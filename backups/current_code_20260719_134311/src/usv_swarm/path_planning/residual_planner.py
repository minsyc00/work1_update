from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..geometry import wrap_angle
from ..schema import CoverageState, PlannerConfig, Pose2D
from .dynamics_validation import validate_transition_sequence
from .obstacles import clipped_axis_aligned_segments, normalize_obstacle_field, path_segment_invalid_reasons
from .patterns import generate_all_region_patterns
from .residuals import assign_residual_backfill, evaluate_tour_coverage_state
from .resources import estimate_repeat_overlap_length, score_cross_agent_ownership_overlap
from .smoothing import build_cover_segment, build_obstacle_aware_transition_segments, build_transition_segment
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

    started = time.perf_counter()
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
    residual_limit = max(path_config.max_residual_backfill_regions, 0)
    if max(config.mission.area_length_x, config.mission.area_length_y) >= max(path_config.large_map_size_threshold, 1e-6):
        residual_limit = min(residual_limit, 12)
    residuals = sorted(coverage_state.residual_components, key=lambda item: len(item.cells), reverse=True)
    residuals = residuals[:residual_limit]
    if not residuals:
        result.diagnostics["status"] = "no_residuals"
        return result

    backfill = assign_residual_backfill(residuals, list(tours.values()), config)
    result.residual_region_count = len(backfill.residual_regions)
    obstacle_fields_by_agent = {
        agent_id: (
            normalize_obstacle_field(
                obstacle_field.obstacles,
                config.for_agent(agent_id),
                path_config,
            )
            if config.agent_profiles and obstacle_field is not None
            else obstacle_field
        )
        for agent_id in tours
    }
    patterns_by_agent = {
        agent_id: _residual_patterns(
            backfill.residual_regions,
            config.for_agent(agent_id) if config.agent_profiles else config,
            path_config,
            obstacle_fields_by_agent[agent_id],
        )
        for agent_id in tours
    }
    contour_residual_candidate_count = sum(
        1
        for agent_patterns in patterns_by_agent.values()
        for candidates in agent_patterns.values()
        for pattern in candidates
        if pattern.metadata.get("contour_residual_fallback") == "true"
    )
    contour_residual_selected_count = 0
    if not any(patterns for agent_patterns in patterns_by_agent.values() for patterns in agent_patterns.values()):
        result.diagnostics["status"] = "no_feasible_residual_patterns"
        return result

    existing_segments = _all_tour_segments(tours)
    residual_feasible_ids: set[str] = set()
    residual_candidate_attempt_count = 0
    residual_budget_exhausted = False
    residual_budget_reason = ""
    time_budget_sec = max(float(path_config.residual_local_tsp_time_budget_sec), 0.0)
    max_candidate_attempts = max(int(path_config.residual_local_tsp_max_candidate_attempts), 0)
    residual_low_efficiency_filtered_count = 0
    residual_low_efficiency_soft_count = 0
    residual_best_gain_per_meter = 0.0
    residual_gain_per_meter_sum = 0.0
    residual_gain_per_meter_count = 0
    residual_min_positive_gain_per_meter = math.inf
    residual_obstacle_aware_retry_count = 0
    large_map_mode = max(config.mission.area_length_x, config.mission.area_length_y) >= max(
        path_config.large_map_size_threshold,
        1e-6,
    )
    residual_obstacle_aware_retry_limit = 0
    if large_map_mode:
        residual_obstacle_aware_retry_limit = min(
            max(16, min(24, len(backfill.residual_regions) * 4)),
            max(int(path_config.large_map_tsp_obstacle_aware_retry_limit), 0),
        )
    region_cell_counts = {
        region.region_id: int(region.metadata.get("cell_count", "0") or 0)
        for region in backfill.residual_regions
    }
    current_pose = {agent_id: _tour_end_pose(tour, config) for agent_id, tour in tours.items()}
    current_time = {agent_id: _tour_end_time(tour) for agent_id, tour in tours.items()}
    serials = {agent_id: len(tour.segments) for agent_id, tour in tours.items()}
    remaining = {
        region.region_id
        for region in backfill.residual_regions
        if any(agent_patterns.get(region.region_id) for agent_patterns in patterns_by_agent.values())
    }
    projected_covered_cells = int(coverage_state.covered.sum())
    total_coverage_cells = int(coverage_state.covered.size)

    def current_budget_reason() -> str:
        if max_candidate_attempts > 0 and residual_candidate_attempt_count >= max_candidate_attempts:
            return "residual_candidate_budget_exhausted"
        if time_budget_sec > 0.0 and time.perf_counter() - started >= time_budget_sec:
            return "residual_time_budget_exhausted"
        return ""

    while remaining:
        reason = current_budget_reason()
        if reason:
            residual_budget_exhausted = True
            residual_budget_reason = reason
            break
        best_choice = None
        for agent_id, tour in sorted(tours.items()):
            agent_config = config.for_agent(agent_id) if config.agent_profiles else config
            agent_obstacle_field = obstacle_fields_by_agent[agent_id]
            reason = current_budget_reason()
            if reason:
                residual_budget_exhausted = True
                residual_budget_reason = reason
                break
            for region_id in sorted(remaining):
                reason = current_budget_reason()
                if reason:
                    residual_budget_exhausted = True
                    residual_budget_reason = reason
                    break
                for pattern in patterns_by_agent.get(agent_id, {}).get(region_id, []):
                    reason = current_budget_reason()
                    if reason:
                        residual_budget_exhausted = True
                        residual_budget_reason = reason
                        break
                    residual_candidate_attempt_count += 1
                    candidate_segments = _build_residual_region_segments(
                        agent_id=agent_id,
                        region_id=region_id,
                        pattern=pattern,
                        current_pose=current_pose[agent_id],
                        current_time=current_time[agent_id],
                        config=agent_config,
                        path_config=path_config,
                        obstacle_field=agent_obstacle_field,
                        start_serial=serials[agent_id],
                        allow_obstacle_aware=False,
                    )
                    if (
                        (
                            not candidate_segments
                            or not _segments_valid(candidate_segments, agent_config, agent_obstacle_field)
                        )
                        and residual_obstacle_aware_retry_count < residual_obstacle_aware_retry_limit
                    ):
                        residual_obstacle_aware_retry_count += 1
                        candidate_segments = _build_residual_region_segments(
                            agent_id=agent_id,
                            region_id=region_id,
                            pattern=pattern,
                            current_pose=current_pose[agent_id],
                            current_time=current_time[agent_id],
                            config=agent_config,
                            path_config=path_config,
                            obstacle_field=agent_obstacle_field,
                            start_serial=serials[agent_id],
                            allow_obstacle_aware=True,
                        )
                    if not candidate_segments or not _segments_valid(
                        candidate_segments,
                        agent_config,
                        agent_obstacle_field,
                    ):
                        continue
                    residual_feasible_ids.add(region_id)
                    path_length = sum(segment.length for segment in candidate_segments)
                    gain_area = _residual_gain_area(region_id, region_cell_counts, coverage_state)
                    gain_per_meter = gain_area / max(path_length, 1e-9)
                    residual_best_gain_per_meter = max(residual_best_gain_per_meter, gain_per_meter)
                    if gain_per_meter > 0.0:
                        residual_gain_per_meter_sum += gain_per_meter
                        residual_gain_per_meter_count += 1
                        residual_min_positive_gain_per_meter = min(residual_min_positive_gain_per_meter, gain_per_meter)
                    below_efficiency_floor = gain_per_meter + 1e-9 < max(path_config.residual_min_gain_per_path_meter, 0.0)
                    hard_filter_low_efficiency = (
                        not path_config.residual_filter_after_target_only
                        or (projected_covered_cells / max(total_coverage_cells, 1)) + 1e-9
                        >= path_config.target_coverage_fraction
                    )
                    if below_efficiency_floor and hard_filter_low_efficiency:
                        residual_low_efficiency_filtered_count += 1
                        continue
                    if below_efficiency_floor:
                        residual_low_efficiency_soft_count += 1
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
                        path_length
                        + pattern.estimated_time
                        + repeat_penalty
                        + cross_agent_score.penalty
                        - max(path_config.residual_gain_reward_weight, 0.0) * gain_per_meter
                    )
                    key = (
                        -region_cell_counts.get(region_id, 0),
                        -gain_per_meter,
                        score,
                        cross_agent_score.penalty,
                        repeat_penalty,
                        agent_id,
                        region_id,
                        pattern.pattern_id,
                    )
                    if best_choice is None or key < best_choice[0]:
                        best_choice = (
                            key,
                            agent_id,
                            region_id,
                            pattern,
                            candidate_segments,
                            repeat_penalty,
                            cross_agent_score.penalty,
                        )
                if residual_budget_exhausted:
                    break
            if residual_budget_exhausted:
                break
        if residual_budget_exhausted:
            break
        if best_choice is None:
            result.diagnostics["skipped_infeasible_residuals"] = str(
                int(result.diagnostics.get("skipped_infeasible_residuals", "0")) + len(remaining)
            )
            break

        _, agent_id, region_id, pattern, new_segments, repeat_penalty, cross_agent_penalty = best_choice
        if pattern.metadata.get("contour_residual_fallback") == "true":
            contour_residual_selected_count += 1
        tour = tours[agent_id]
        selected_id = f"{region_id}_local_tsp_agent{agent_id}_{serials[agent_id]}"
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
        remaining.remove(region_id)
        projected_covered_cells = min(
            total_coverage_cells,
            projected_covered_cells + max(region_cell_counts.get(region_id, 0), 0),
        )
        current_pose[agent_id] = pattern.exit_pose
        current_time[agent_id] = _segment_end_time(new_segments[-1])
        serials[agent_id] += len(new_segments)

    for agent_id, tour in sorted(tours.items()):
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

    if residual_budget_exhausted:
        result.diagnostics["status"] = "budget_fallback"
    elif result.appended_count:
        result.diagnostics["status"] = "success"
    elif residual_low_efficiency_filtered_count > 0 and residual_feasible_ids:
        result.diagnostics["status"] = "low_efficiency_filtered"
    else:
        result.diagnostics["status"] = "no_append"
    result.diagnostics["residual_candidate_attempt_count"] = str(residual_candidate_attempt_count)
    result.diagnostics["residual_budget_exhausted"] = str(residual_budget_exhausted).lower()
    result.diagnostics["residual_budget_reason"] = residual_budget_reason
    result.diagnostics["residual_elapsed_sec"] = f"{time.perf_counter() - started:.6f}"
    result.diagnostics["residual_time_budget_sec"] = f"{time_budget_sec:.6f}"
    result.diagnostics["residual_max_candidate_attempts"] = str(max_candidate_attempts)
    result.diagnostics["residual_low_efficiency_filtered_count"] = str(residual_low_efficiency_filtered_count)
    result.diagnostics["residual_low_efficiency_soft_count"] = str(residual_low_efficiency_soft_count)
    result.diagnostics["residual_best_gain_per_path_meter"] = f"{residual_best_gain_per_meter:.6f}"
    result.diagnostics["residual_min_positive_gain_per_path_meter"] = (
        f"{0.0 if not math.isfinite(residual_min_positive_gain_per_meter) else residual_min_positive_gain_per_meter:.6f}"
    )
    result.diagnostics["residual_mean_positive_gain_per_path_meter"] = (
        f"{residual_gain_per_meter_sum / max(residual_gain_per_meter_count, 1):.6f}"
    )
    result.diagnostics["residual_min_gain_per_path_meter"] = f"{max(path_config.residual_min_gain_per_path_meter, 0.0):.6f}"
    result.diagnostics["residual_filter_after_target_only"] = str(path_config.residual_filter_after_target_only).lower()
    result.diagnostics["residual_obstacle_aware_retry_count"] = str(residual_obstacle_aware_retry_count)
    result.diagnostics["residual_feasible_count"] = str(len(residual_feasible_ids))
    result.diagnostics["residual_infeasible_count"] = str(max(result.residual_region_count - len(residual_feasible_ids), 0))
    result.diagnostics["repeat_path_penalty_total"] = f"{result.repeat_path_penalty_total:.6f}"
    result.diagnostics["contour_residual_candidate_count"] = str(contour_residual_candidate_count)
    result.diagnostics["contour_residual_selected_count"] = str(contour_residual_selected_count)
    return result


def _residual_gain_area(
    region_id: str,
    region_cell_counts: Dict[str, int],
    coverage_state: CoverageState,
) -> float:
    cell_count = max(region_cell_counts.get(region_id, 0), 0)
    return float(cell_count) * max(float(coverage_state.resolution), 1e-9) ** 2


def _residual_patterns(
    residual_regions: Sequence[DecomposedRegion],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> Dict[str, List[RegionCoveragePattern]]:
    if max(config.mission.area_length_x, config.mission.area_length_y) >= max(path_config.large_map_size_threshold, 1e-6):
        return {
            region.region_id: (
                _contour_residual_patterns(region, config)
                if path_config.enable_contour_residual_fallback
                else []
            )
            + _fallback_residual_patterns(region, config, obstacle_field)[:8]
            for region in residual_regions
        }
    raw = generate_all_region_patterns(residual_regions, config, path_config, obstacle_field=obstacle_field)
    patterns = {
        region_id: [pattern for pattern in candidates if pattern.feasible and pattern.passes]
        for region_id, candidates in raw.items()
    }
    for region in residual_regions:
        if not patterns.get(region.region_id):
            patterns[region.region_id] = (
                _contour_residual_patterns(region, config)
                if path_config.enable_contour_residual_fallback
                else []
            ) + _fallback_residual_patterns(region, config)
    return patterns


def _contour_residual_patterns(
    region: DecomposedRegion,
    config: PlannerConfig,
) -> List[RegionCoveragePattern]:
    x_min, y_min, x_max, y_max = region.bounds
    sensor_inset = config.footprint.width_wf / 2.0
    left = x_min + sensor_inset
    right = x_max - sensor_inset
    bottom = y_min + sensor_inset
    top = y_max - sensor_inset
    radius = config.fleet.min_turn_radius
    min_edge = max(config.footprint.width_wf * 0.25, config.footprint.length_lf * 0.25, 0.25)
    if right - left < 2.0 * radius + min_edge or top - bottom < 2.0 * radius + min_edge:
        return []

    clockwise_specs = [
        (Pose2D(left + radius, bottom, 0.0), Pose2D(right - radius, bottom, 0.0), "bottom"),
        (Pose2D(right, bottom + radius, math.pi / 2.0), Pose2D(right, top - radius, math.pi / 2.0), "right"),
        (Pose2D(right - radius, top, math.pi), Pose2D(left + radius, top, math.pi), "top"),
        (Pose2D(left, top - radius, -math.pi / 2.0), Pose2D(left, bottom + radius, -math.pi / 2.0), "left"),
    ]

    def build_variant(specs: Sequence[tuple[Pose2D, Pose2D, str]], suffix: str) -> RegionCoveragePattern:
        passes: List[CoveragePass] = []
        for index, (start, end, side) in enumerate(specs):
            length = math.hypot(end.x - start.x, end.y - start.y)
            passes.append(
                CoveragePass(
                    pass_id=f"{region.region_id}_contour_{suffix}_{side}",
                    region_id=region.region_id,
                    sequence_index=index,
                    scan_axis="contour",
                    start_pose=start,
                    end_pose=end,
                    center_coordinate=float(index),
                    width=config.footprint.width_wf,
                    length=length,
                )
            )
        coverage_length = sum(item.length for item in passes)
        turn_count = max(len(passes) - 1, 0)
        turn_length = turn_count * math.pi * radius / 2.0
        turn_angle = turn_count * math.pi / 2.0
        estimated_time = (
            coverage_length / max(config.fleet.cover_speed, 1e-6)
            + turn_length / max(config.fleet.turn_speed_max, 1e-6)
            + turn_angle
            / max(config.fleet.turn_speed_max / max(radius, 1e-6), 1e-6)
        )
        return RegionCoveragePattern(
            pattern_id=f"{region.region_id}_contour_pattern_{suffix}",
            region_id=region.region_id,
            scan_axis="contour",
            passes=passes,
            entry_pose=passes[0].start_pose,
            exit_pose=passes[-1].end_pose,
            coverage_length=coverage_length,
            turn_length=turn_length,
            turn_angle=turn_angle,
            total_length=coverage_length + turn_length,
            estimated_time=estimated_time,
            max_curvature=1.0 / max(radius, 1e-6),
            feasible=True,
            source_algorithm="continuous_equidistant_residual_contour",
            metadata={
                "contour_residual_fallback": "true",
                "contour_direction": suffix,
                "contour_turn_radius": f"{radius:.6f}",
                "contour_offset": f"{sensor_inset:.6f}",
                "residual_fallback": "true",
            },
        )

    counterclockwise_specs = [
        (Pose2D(left, bottom + radius, math.pi / 2.0), Pose2D(left, top - radius, math.pi / 2.0), "left"),
        (Pose2D(left + radius, top, 0.0), Pose2D(right - radius, top, 0.0), "top"),
        (Pose2D(right, top - radius, -math.pi / 2.0), Pose2D(right, bottom + radius, -math.pi / 2.0), "right"),
        (Pose2D(right - radius, bottom, math.pi), Pose2D(left + radius, bottom, math.pi), "bottom"),
    ]
    return [
        build_variant(clockwise_specs, "clockwise"),
        build_variant(counterclockwise_specs, "counterclockwise"),
    ]


def _fallback_residual_patterns(
    region: DecomposedRegion,
    config: PlannerConfig,
    obstacle_field: ObstacleField | None = None,
) -> List[RegionCoveragePattern]:
    x_min, y_min, x_max, y_max = region.bounds
    cx = min(max(region.center[0], x_min), x_max)
    cy = min(max(region.center[1], y_min), y_max)
    min_len = max(config.footprint.width_wf * 0.5, 0.25)
    half_width = config.footprint.width_wf / 2.0
    half_length = config.footprint.length_lf / 2.0
    x_motion_low = min(half_length, config.mission.area_length_x / 2.0)
    x_motion_high = max(x_motion_low, config.mission.area_length_x - half_length)
    y_motion_low = min(half_length, config.mission.area_length_y / 2.0)
    y_motion_high = max(y_motion_low, config.mission.area_length_y - half_length)
    x_fixed_low = min(half_width, config.mission.area_length_x / 2.0)
    x_fixed_high = max(x_fixed_low, config.mission.area_length_x - half_width)
    y_fixed_low = min(half_width, config.mission.area_length_y / 2.0)
    y_fixed_high = max(y_fixed_low, config.mission.area_length_y - half_width)

    candidates: List[tuple[str, Pose2D, Pose2D]] = []
    seen: set[tuple[str, float, float, float, bool]] = set()

    def clamp(value: float, low: float, high: float) -> float:
        return min(max(value, low), high)

    def expanded_interval(low: float, high: float, center: float, lower: float, upper: float) -> tuple[float, float] | None:
        low = clamp(low, lower, upper)
        high = clamp(high, lower, upper)
        if high < low:
            low, high = high, low
        if high - low + 1e-9 < min_len:
            low = clamp(center - min_len / 2.0, lower, upper)
            high = clamp(center + min_len / 2.0, lower, upper)
            if high - low + 1e-9 < min_len:
                if low <= lower + 1e-9:
                    high = min(upper, low + min_len)
                else:
                    low = max(lower, high - min_len)
        if high - low <= 1e-9:
            return None
        return low, high

    def fixed_values(center: float, low: float, high: float, lower: float, upper: float) -> List[float]:
        values = [
            center,
            (low + high) / 2.0,
            low + half_width,
            high - half_width,
        ]
        unique: List[float] = []
        for value in values:
            fixed = clamp(value, lower, upper)
            if all(abs(fixed - existing) > 1e-6 for existing in unique):
                unique.append(fixed)
        return unique

    def clipped_intervals(axis: str, fixed: float, low: float, high: float, center: float) -> List[tuple[float, float]]:
        intervals = [(low, high)]
        if obstacle_field is not None:
            intervals = clipped_axis_aligned_segments(
                axis=axis,
                fixed_coord=fixed,
                low=low,
                high=high,
                field=obstacle_field,
                footprint_width=0.0,
                min_length=max(min_len * 0.25, 0.25),
            )
        intervals = [(u0, u1) for u0, u1 in intervals if u1 - u0 > 1e-9]
        intervals.sort(key=lambda item: (_interval_distance_to_center(item, center), -(item[1] - item[0])))
        return intervals[:2]

    def add_candidate(axis: str, fixed: float, u0: float, u1: float, reverse: bool) -> None:
        key = (axis, round(fixed, 6), round(u0, 6), round(u1, 6), reverse)
        if key in seen:
            return
        seen.add(key)
        if axis == "x":
            if reverse:
                candidates.append(("x_rev", Pose2D(u1, fixed, math.pi), Pose2D(u0, fixed, math.pi)))
            else:
                candidates.append(("x", Pose2D(u0, fixed, 0.0), Pose2D(u1, fixed, 0.0)))
        else:
            if reverse:
                candidates.append(("y_rev", Pose2D(fixed, u1, -math.pi / 2.0), Pose2D(fixed, u0, -math.pi / 2.0)))
            else:
                candidates.append(("y", Pose2D(fixed, u0, math.pi / 2.0), Pose2D(fixed, u1, math.pi / 2.0)))

    x_interval = expanded_interval(x_min, x_max, cx, x_motion_low, x_motion_high)
    if x_interval is not None:
        for fixed in fixed_values(cy, y_min, y_max, y_fixed_low, y_fixed_high):
            for u0, u1 in clipped_intervals("x", fixed, x_interval[0], x_interval[1], cx):
                add_candidate("x", fixed, u0, u1, reverse=False)
                add_candidate("x", fixed, u0, u1, reverse=True)
    y_interval = expanded_interval(y_min, y_max, cy, y_motion_low, y_motion_high)
    if y_interval is not None:
        for fixed in fixed_values(cx, x_min, x_max, x_fixed_low, x_fixed_high):
            for u0, u1 in clipped_intervals("y", fixed, y_interval[0], y_interval[1], cy):
                add_candidate("y", fixed, u0, u1, reverse=False)
                add_candidate("y", fixed, u0, u1, reverse=True)

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
                metadata={"residual_fallback": "true", "residual_candidate_axis": axis},
            )
        )
    patterns.sort(key=lambda item: (-item.coverage_length, item.pattern_id))
    return patterns


def _interval_distance_to_center(interval: tuple[float, float], center: float) -> float:
    low, high = interval
    if low <= center <= high:
        return 0.0
    return min(abs(center - low), abs(center - high))


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
    allow_obstacle_aware: bool = True,
) -> List[PathSegmentSpec]:
    segments: List[PathSegmentSpec] = []
    serial = start_serial
    if (
        max(config.mission.area_length_x, config.mission.area_length_y) >= max(path_config.large_map_size_threshold, 1e-6)
        and not allow_obstacle_aware
    ):
        transit_segments = [
            build_transition_segment(
                segment_id=f"agent{agent_id}_residual_local_{serial}_to_{region_id}",
                start=current_pose,
                end=pattern.entry_pose,
                start_time=current_time,
                config=config,
                kind="transit",
                sample_count=24,
                use_bezier=path_config.use_bezier_smoothing,
            )
        ]
    else:
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
