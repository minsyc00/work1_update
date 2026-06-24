from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..geometry import wrap_angle
from ..schema import CoverageState, PlannerConfig, Pose2D
from .dynamics_validation import validate_transition_sequence
from .obstacles import path_segment_invalid_reasons
from .patterns import generate_all_region_patterns
from .residuals import assign_residual_backfill, evaluate_tour_coverage_state
from .resources import estimate_repeat_overlap_length, score_cross_agent_ownership_overlap
from .smoothing import build_cover_segment, build_obstacle_aware_transition_segments
from .types import (
    AgentPathPlan,
    CoveragePass,
    CoverageOwnershipMap,
    DecomposedRegion,
    ObstacleField,
    PathPlanningConfig,
    PathSegmentSpec,
    RegionCoveragePattern,
    SingleUsvTourPlan,
)


@dataclass
class ResidualLocalPlanResult:
    appended_count: int = 0
    residual_region_count: int = 0
    repeat_path_penalty_total: float = 0.0
    residual_local_tsp_enabled: bool = True
    diagnostics: Dict[str, str] = field(default_factory=dict)


def append_residual_local_tsp(
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    tours: Dict[int, SingleUsvTourPlan],
    coverage_state: CoverageState | None = None,
    agents: Dict[int, AgentPathPlan] | None = None,
    ownership_map: CoverageOwnershipMap | None = None,
) -> ResidualLocalPlanResult:
    """Append residual coverage through a small local TSP-CPP problem.

    Residual components are first converted into regular decomposed regions and
    coverage patterns. Each agent then greedily solves the local residual tour
    from its current tail pose. The score includes dynamically feasible
    transition length and a repeat-path penalty against all already planned
    segments, so residual repair no longer degenerates into nearest single-line
    appends through a shared corridor.
    """

    result = ResidualLocalPlanResult(residual_local_tsp_enabled=path_config.enable_residual_local_tsp)
    if not config.mission.residual_enable or not path_config.enable_residual_backfill:
        result.diagnostics["status"] = "disabled"
        return result
    if not path_config.enable_residual_local_tsp:
        result.diagnostics["status"] = "local_tsp_disabled"
        return result

    if coverage_state is None:
        coverage_state = evaluate_tour_coverage_state(
            config,
            list(tours.values()),
            resolution=path_config.residual_resolution,
            obstacle_field=obstacle_field,
            include_non_cover_segments=path_config.count_transit_coverage,
        )
    residuals = sorted(coverage_state.residual_components, key=lambda item: len(item.cells), reverse=True)
    residuals = residuals[: max(path_config.max_residual_backfill_regions, 0)]
    if not residuals:
        result.diagnostics["status"] = "no_residuals"
        return result

    backfill = assign_residual_backfill(residuals, list(tours.values()), config)
    result.residual_region_count = len(backfill.residual_regions)
    patterns = _residual_patterns(backfill.residual_regions, config, path_config, obstacle_field)
    if not any(patterns.values()):
        result.diagnostics["status"] = "no_feasible_residual_patterns"
        return result

    existing_segments = _all_tour_segments(tours)
    residual_feasible_ids: set[str] = set()
    residual_candidate_attempt_count = 0
    for agent_id, assigned_region_ids in sorted(backfill.agent_regions.items()):
        tour = tours.get(agent_id)
        if tour is None:
            continue
        remaining = [region_id for region_id in assigned_region_ids if patterns.get(region_id)]
        current_pose = _tour_end_pose(tour, config)
        current_time = _tour_end_time(tour)
        serial = len(tour.segments)
        agent_appended = 0
        while remaining:
            best_choice = None
            for region_id in remaining:
                for pattern in patterns.get(region_id, []):
                    residual_candidate_attempt_count += 1
                    candidate_segments = _build_residual_region_segments(
                        agent_id=agent_id,
                        region_id=region_id,
                        pattern=pattern,
                        current_pose=current_pose,
                        current_time=current_time,
                        config=config,
                        path_config=path_config,
                        obstacle_field=obstacle_field,
                        start_serial=serial,
                    )
                    if not candidate_segments or not _segments_valid(candidate_segments, config, obstacle_field):
                        continue
                    residual_feasible_ids.add(region_id)
                    overlap_length = estimate_repeat_overlap_length(candidate_segments, existing_segments, path_config)
                    repeat_penalty = path_config.repeat_path_penalty_weight * overlap_length
                    cross_agent_score = score_cross_agent_ownership_overlap(
                        candidate_segments,
                        agent_id,
                        ownership_map,
                        path_config,
                        config=config,
                        annotate=True,
                    )
                    score = (
                        sum(segment.length for segment in candidate_segments)
                        + pattern.estimated_time
                        + repeat_penalty
                        + cross_agent_score.penalty
                    )
                    key = (score, cross_agent_score.penalty, repeat_penalty, region_id, pattern.pattern_id)
                    if best_choice is None or key < best_choice[0]:
                        best_choice = (
                            key,
                            region_id,
                            pattern,
                            candidate_segments,
                            repeat_penalty,
                            cross_agent_score.penalty,
                        )
            if best_choice is None:
                result.diagnostics["skipped_infeasible_residuals"] = str(
                    int(result.diagnostics.get("skipped_infeasible_residuals", "0")) + len(remaining)
                )
                break

            _, region_id, pattern, new_segments, repeat_penalty, cross_agent_penalty = best_choice
            selected_id = f"{region_id}_local_tsp_{agent_appended}"
            _mark_residual_segments(new_segments, selected_id)
            tour.segments.extend(new_segments)
            tour.region_order.append(selected_id)
            tour.selected_patterns[selected_id] = pattern
            tour.diagnostics["residual_backfill"] = "true"
            tour.diagnostics["residual_local_tsp"] = "true"
            tour.diagnostics["repeat_path_penalty"] = f"{float(tour.diagnostics.get('repeat_path_penalty', '0') or 0.0) + repeat_penalty:.6f}"
            tour.diagnostics["cross_agent_penalty"] = f"{float(tour.diagnostics.get('cross_agent_penalty', '0') or 0.0) + cross_agent_penalty:.6f}"
            existing_segments.extend(new_segments)
            result.appended_count += 1
            result.repeat_path_penalty_total += repeat_penalty
            agent_appended += 1
            remaining.remove(region_id)
            current_pose = pattern.exit_pose
            current_time = _segment_end_time(new_segments[-1])
            serial += len(new_segments)
        _refresh_tour_metrics(tour, path_config)
        if agents is not None and agent_id in agents:
            agents[agent_id].segments = tour.segments
            agents[agent_id].metrics.update(
                {
                    "total_length": tour.total_length,
                    "total_turn_angle": tour.total_turn_angle,
                    "estimated_time": tour.estimated_time,
                    "objective": tour.objective,
                    "segment_count": float(len(tour.segments)),
                }
            )

    result.diagnostics["status"] = "success" if result.appended_count else "no_append"
    result.diagnostics["residual_candidate_attempt_count"] = str(residual_candidate_attempt_count)
    result.diagnostics["residual_feasible_count"] = str(len(residual_feasible_ids))
    result.diagnostics["residual_infeasible_count"] = str(max(result.residual_region_count - len(residual_feasible_ids), 0))
    result.diagnostics["repeat_path_penalty_total"] = f"{result.repeat_path_penalty_total:.6f}"
    return result


def _residual_patterns(
    residual_regions: Sequence[DecomposedRegion],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> Dict[str, List[RegionCoveragePattern]]:
    raw = generate_all_region_patterns(residual_regions, config, path_config, obstacle_field=obstacle_field)
    patterns = {
        region_id: [pattern for pattern in candidates if pattern.feasible and pattern.passes]
        for region_id, candidates in raw.items()
    }
    for region in residual_regions:
        if not patterns.get(region.region_id):
            patterns[region.region_id] = _fallback_residual_patterns(region, config)
    return patterns


def _fallback_residual_patterns(
    region: DecomposedRegion,
    config: PlannerConfig,
) -> List[RegionCoveragePattern]:
    x_min, y_min, x_max, y_max = region.bounds
    cx = min(max(region.center[0], x_min), x_max)
    cy = min(max(region.center[1], y_min), y_max)
    min_len = max(config.footprint.width_wf * 0.5, 0.25)
    x0 = max(0.0, min(x_min, cx - min_len / 2.0))
    x1 = min(config.mission.area_length_x, max(x_max, cx + min_len / 2.0))
    y0 = max(0.0, min(y_min, cy - min_len / 2.0))
    y1 = min(config.mission.area_length_y, max(y_max, cy + min_len / 2.0))
    candidates = [
        ("x", Pose2D(x0, cy, 0.0), Pose2D(x1, cy, 0.0)),
        ("x_rev", Pose2D(x1, cy, math.pi), Pose2D(x0, cy, math.pi)),
        ("y", Pose2D(cx, y0, math.pi / 2.0), Pose2D(cx, y1, math.pi / 2.0)),
        ("y_rev", Pose2D(cx, y1, -math.pi / 2.0), Pose2D(cx, y0, -math.pi / 2.0)),
    ]
    patterns: List[RegionCoveragePattern] = []
    for idx, (axis, start, end) in enumerate(candidates):
        length = math.hypot(end.x - start.x, end.y - start.y)
        if length <= 1e-9:
            continue
        coverage_pass = CoveragePass(
            pass_id=f"{region.region_id}_fallback_{axis}",
            region_id=region.region_id,
            sequence_index=0,
            scan_axis="x" if axis.startswith("x") else "y",
            start_pose=start,
            end_pose=end,
            center_coordinate=cy if axis.startswith("x") else cx,
            width=config.footprint.width_wf,
            length=length,
        )
        patterns.append(
            RegionCoveragePattern(
                pattern_id=f"{region.region_id}_fallback_pattern_{idx}",
                region_id=region.region_id,
                scan_axis=coverage_pass.scan_axis,
                passes=[coverage_pass],
                entry_pose=start,
                exit_pose=end,
                coverage_length=length,
                turn_length=0.0,
                turn_angle=0.0,
                total_length=length,
                estimated_time=length / max(config.fleet.cover_speed, 1e-6),
                max_curvature=0.0,
                feasible=True,
                source_algorithm="residual_fallback_single_pass",
                metadata={"residual_fallback": "true"},
            )
        )
    return patterns


def _build_residual_region_segments(
    agent_id: int,
    region_id: str,
    pattern: RegionCoveragePattern,
    current_pose: Pose2D,
    current_time: float,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    start_serial: int,
) -> List[PathSegmentSpec]:
    segments: List[PathSegmentSpec] = []
    serial = start_serial
    transit_segments = build_obstacle_aware_transition_segments(
        segment_id=f"agent{agent_id}_residual_local_{serial}_to_{region_id}",
        start=current_pose,
        end=pattern.entry_pose,
        start_time=current_time,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        kind="transit",
        sample_count=48,
    )
    if any(segment.metadata.get("kinematic_feasible") == "false" for segment in transit_segments):
        return []
    for sub_idx, segment in enumerate(transit_segments):
        if segment.length <= 1e-9:
            continue
        segment.metadata.update(
            {
                "region_id": region_id,
                "residual_backfill": "true",
                "residual_connector": "true",
                "residual_tsp_edge": "true",
                "resource_id": f"residual_connector:{region_id}:{sub_idx}",
            }
        )
        segments.append(segment)
        current_time = _segment_end_time(segment)
        serial += 1
    for pass_idx, coverage_pass in enumerate(pattern.passes):
        cover = build_cover_segment(
            segment_id=f"agent{agent_id}_residual_local_{serial}_{coverage_pass.pass_id}",
            start=coverage_pass.start_pose,
            end=coverage_pass.end_pose,
            start_time=current_time,
            speed=max(config.fleet.cover_speed, 1e-6),
            sample_count=12,
        )
        cover.metadata.update(
            {
                "region_id": coverage_pass.region_id,
                "pass_id": coverage_pass.pass_id,
                "residual_backfill": "true",
                "sweep_endpoint_pair": "true",
                "resource_id": f"residual_cover:{coverage_pass.region_id}:{coverage_pass.pass_id}",
            }
        )
        segments.append(cover)
        current_time = _segment_end_time(cover)
        serial += 1
        if pass_idx >= len(pattern.passes) - 1:
            continue
        next_pass = pattern.passes[pass_idx + 1]
        turns = build_obstacle_aware_transition_segments(
            segment_id=f"agent{agent_id}_residual_local_{serial}_{coverage_pass.pass_id}_turn",
            start=coverage_pass.end_pose,
            end=next_pass.start_pose,
            start_time=current_time,
            config=config,
            path_config=path_config,
            obstacle_field=obstacle_field,
            kind="turn",
            sample_count=48,
        )
        if any(segment.metadata.get("kinematic_feasible") == "false" for segment in turns):
            return []
        for sub_idx, turn in enumerate(turns):
            turn.metadata.update(
                {
                    "region_id": coverage_pass.region_id,
                    "residual_backfill": "true",
                    "residual_turn": "true",
                    "resource_id": f"residual_turn:{coverage_pass.pass_id}:{sub_idx}",
                }
            )
            segments.append(turn)
            current_time = _segment_end_time(turn)
            serial += 1
    return segments


def _segments_valid(
    segments: Sequence[PathSegmentSpec],
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
) -> bool:
    if any(path_segment_invalid_reasons(segment, config, obstacle_field) for segment in segments):
        return False
    if any(segment.metadata.get("kinematic_feasible") == "false" for segment in segments):
        return False
    return validate_transition_sequence(segments, config, obstacle_field=obstacle_field, retime=True).valid


def _mark_residual_segments(segments: Sequence[PathSegmentSpec], selected_id: str) -> None:
    for segment in segments:
        segment.metadata["residual_region_visit"] = selected_id


def _all_tour_segments(tours: Dict[int, SingleUsvTourPlan]) -> List[PathSegmentSpec]:
    return [segment for tour in tours.values() for segment in tour.segments]


def _tour_end_pose(tour: SingleUsvTourPlan, config: PlannerConfig) -> Pose2D:
    for segment in reversed(tour.segments):
        if segment.waypoints:
            waypoint = segment.waypoints[-1]
            return Pose2D(waypoint.x, waypoint.y, waypoint.psi)
    state = config.fleet.initial_states_3dof[tour.agent_id]
    return state.pose()


def _tour_end_time(tour: SingleUsvTourPlan) -> float:
    return max((_segment_end_time(segment) for segment in tour.segments), default=0.0)


def _segment_end_time(segment: PathSegmentSpec) -> float:
    if not segment.waypoints or segment.waypoints[-1].time is None:
        return 0.0
    return float(segment.waypoints[-1].time)


def _refresh_tour_metrics(tour: SingleUsvTourPlan, path_config: PathPlanningConfig) -> None:
    tour.total_length = sum(segment.length for segment in tour.segments)
    tour.total_turn_angle = sum(_segment_heading_variation(segment) for segment in tour.segments)
    tour.estimated_time = _tour_end_time(tour)
    tour.objective = (
        path_config.length_weight * tour.total_length
        + path_config.turn_angle_weight * tour.total_turn_angle
        + path_config.time_weight * tour.estimated_time
    )


def _segment_heading_variation(segment: PathSegmentSpec) -> float:
    headings = [waypoint.psi for waypoint in segment.waypoints]
    return sum(abs(wrap_angle(headings[idx] - headings[idx - 1])) for idx in range(1, len(headings)))
