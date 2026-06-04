from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..dubins import dubins_shortest_path
from ..geometry import wrap_angle
from ..schema import PlannerConfig, Pose2D
from .aco import AcoTspResult, solve_aco_tsp_cpp, validate_tsp_solver
from .assignment import balance_region_workload
from .decomposition import decompose_obstacle_aware_area, decompose_rectangular_area
from .dynamics_validation import dynamic_edge_cost, validate_transition_dynamics, validate_transition_sequence
from .graph import build_region_graph
from .obstacles import (
    normalize_obstacle_field,
    path_segment_invalid_length,
    path_segment_invalid_reasons,
    sampled_segment_footprint_collides,
)
from .patterns import generate_all_region_patterns
from .residuals import evaluate_tour_coverage_state
from .smoothing import build_cover_segment, build_obstacle_aware_transition_segments, build_transition_segment
from .types import (
    AgentPathPlan,
    CoveragePass,
    MultiAgentPathPlan,
    ObstacleField,
    PathPlanningConfig,
    PathSegmentSpec,
    RegionCoveragePattern,
    RegionSweepPath,
    RegionVisitNode,
    SingleUsvTourPlan,
    StaticObstacle,
)
from .visualization import _agent_color, _draw_obstacles, _draw_polygon, _new_map_axes, _plot_segment


def run_paper_style_region_tsp_experiment(
    config: PlannerConfig,
    static_obstacles: Sequence[StaticObstacle] | None,
    output_dir: str | Path,
    path_config: PathPlanningConfig | None = None,
    map_id: str = "",
    render: bool = True,
) -> Tuple[MultiAgentPathPlan, Dict[str, object]]:
    """Run the paper-style experiment: sweep inside each region, TSP between regions."""

    started = time.perf_counter()
    path_config = path_config or PathPlanningConfig.from_planner_config(config)
    static_obstacles = list(static_obstacles or [])
    obstacle_field = normalize_obstacle_field(static_obstacles, config, path_config) if static_obstacles else None

    regions = (
        decompose_obstacle_aware_area(config, path_config, obstacle_field)
        if obstacle_field is not None and obstacle_field.inflated_obstacles
        else decompose_rectangular_area(config, path_config)
    )
    base_region_count = len(regions)
    if obstacle_field is not None and obstacle_field.inflated_obstacles:
        regions = _coarsen_paper_style_regions(regions, config)
    raw_patterns = _generate_paper_style_patterns(regions, config, path_config, obstacle_field)
    sweep_paths, feasible_patterns, infeasible_regions = _build_region_sweep_paths(raw_patterns, config, path_config, obstacle_field)
    feasible_regions = [region for region in regions if region.region_id in feasible_patterns]
    graph = build_region_graph(feasible_regions, feasible_patterns, config, obstacle_field=obstacle_field)
    assignment = balance_region_workload(graph, config)

    agents: Dict[int, AgentPathPlan] = {}
    tours: Dict[int, SingleUsvTourPlan] = {}
    tsp_records: Dict[int, Dict[str, object]] = {}
    infeasible_edges: List[Dict[str, object]] = []
    for agent_id, region_ids in sorted(assignment.agent_regions.items()):
        result = _solve_agent_region_tsp(
            agent_id,
            region_ids,
            feasible_patterns,
            sweep_paths,
            config,
            path_config,
            obstacle_field,
        )
        infeasible_edges.extend(result["infeasible_edges"])
        segments = result["segments"]
        metrics = _agent_metrics(segments, config, obstacle_field)
        agents[agent_id] = AgentPathPlan(
            agent_id=agent_id,
            source_algorithm="paper_style_region_tsp",
            segments=segments,
            metrics=metrics,
        )
        selected_patterns = result["selected_patterns"]
        tours[agent_id] = SingleUsvTourPlan(
            agent_id=agent_id,
            region_order=list(result["final_order"]),
            selected_patterns=selected_patterns,
            segments=segments,
            total_length=metrics["total_length"],
            total_turn_angle=metrics["total_turn_angle"],
            estimated_time=metrics["estimated_time"],
            objective=metrics["total_length"] + 0.35 * metrics["total_turn_angle"] + metrics["estimated_time"],
            diagnostics={"planned_region_order": ",".join(result["initial_order"])},
        )
        tsp_records[agent_id] = {
            "assigned_regions": list(region_ids),
            "initial_order": list(result["initial_order"]),
            "final_order": list(result["final_order"]),
            "skipped_regions": [region_id for region_id in region_ids if region_id not in set(result["final_order"])],
            "tsp_node_count": len(region_ids),
            "tsp_solver_metadata": dict(result["tsp_solver_metadata"]),
            "requested_tsp_solver": str(result["tsp_solver_metadata"].get("requested_tsp_solver", path_config.tsp_solver)),
            "effective_tsp_solver": str(result["tsp_solver_metadata"].get("effective_tsp_solver", "deterministic")),
            "tsp_solver_status": str(result["tsp_solver_metadata"].get("tsp_solver_status", "success")),
            "aco_best_objective": result["tsp_solver_metadata"].get("aco_best_objective"),
            "aco_initial_objective": result["tsp_solver_metadata"].get("aco_initial_objective"),
            "aco_iteration_count": int(result["tsp_solver_metadata"].get("aco_iteration_count", 0) or 0),
            "aco_convergence_trace": list(result["tsp_solver_metadata"].get("aco_convergence_trace", [])),
            "aco_accepted_3opt_count": int(result["tsp_solver_metadata"].get("aco_accepted_3opt_count", 0) or 0),
            "candidate_pattern_counts": dict(result["candidate_pattern_counts"]),
            "candidate_attempt_count": int(result["candidate_attempt_count"]),
            "rejected_candidate_count": int(result["rejected_candidate_count"]),
            "selected_pattern_ids": dict(result["selected_pattern_ids"]),
            "coverage_endpoint_count": sum(
                len(sweep_paths[region_id].endpoints)
                for region_id in region_ids
                if region_id in sweep_paths
            ),
            "infeasible_edges": list(result["infeasible_edges"]),
        }

    residual_backfill_count = 0
    coverage_state = evaluate_tour_coverage_state(
        config,
        list(tours.values()),
        resolution=path_config.residual_resolution,
        obstacle_field=obstacle_field,
    )
    for _ in range(max(path_config.residual_backfill_cycles, 0)):
        appended_this_cycle = _append_paper_style_residual_backfill(
            config,
            path_config,
            obstacle_field,
            agents,
            tours,
            coverage_state,
        )
        if appended_this_cycle == 0:
            break
        residual_backfill_count += appended_this_cycle
        for agent_id, agent in agents.items():
            agent.metrics = _agent_metrics(agent.segments, config, obstacle_field)
        coverage_state = evaluate_tour_coverage_state(
            config,
            list(tours.values()),
            resolution=path_config.residual_resolution,
            obstacle_field=obstacle_field,
        )
        if coverage_state.coverage_fraction >= 1.0 - 1e-9:
            break
    totals = _global_metrics(agents)
    visit_nodes = _region_visit_nodes(feasible_patterns)
    report: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "map_id": map_id,
        "algorithm": "paper_style_region_tsp",
        "requested_tsp_solver": path_config.tsp_solver,
        "effective_tsp_solvers": {
            str(agent_id): record.get("effective_tsp_solver", "deterministic")
            for agent_id, record in tsp_records.items()
        },
        "aco_best_objective": sum(
            float(record.get("aco_best_objective") or 0.0)
            for record in tsp_records.values()
        ),
        "aco_initial_objective": sum(
            float(record.get("aco_initial_objective") or 0.0)
            for record in tsp_records.values()
        ),
        "aco_iteration_count": max((int(record.get("aco_iteration_count", 0) or 0) for record in tsp_records.values()), default=0),
        "aco_accepted_3opt_count": sum(int(record.get("aco_accepted_3opt_count", 0) or 0) for record in tsp_records.values()),
        "elapsed_sec": time.perf_counter() - started,
        "region_count": len(regions),
        "base_region_count": base_region_count,
        "feasible_region_count": len(feasible_regions),
        "infeasible_regions": infeasible_regions,
        "infeasible_edges": infeasible_edges,
        "tsp_node_count": len(visit_nodes),
        "coverage_endpoint_count": sum(node.coverage_endpoint_count for node in visit_nodes.values()),
        "agent_tsp_records": tsp_records,
        "coverage_fraction": coverage_state.coverage_fraction,
        "residual_count": len(coverage_state.residual_components),
        "residual_backfill_count": residual_backfill_count,
        "metrics": totals,
    }
    path_plan = MultiAgentPathPlan(
        algorithm_name="paper_style_region_tsp",
        agents=agents,
        metadata={
            "status": "paper_style_region_tsp",
            "requested_tsp_solver": path_config.tsp_solver,
            "effective_tsp_solver": ",".join(
                sorted({str(record.get("effective_tsp_solver", "deterministic")) for record in tsp_records.values()})
            ),
            "tsp_solver_status": ",".join(
                sorted({str(record.get("tsp_solver_status", "success")) for record in tsp_records.values()})
            ),
            "region_count": str(len(feasible_regions)),
            "base_region_count": str(base_region_count),
            "tsp_node_count": str(len(visit_nodes)),
            "coverage_endpoint_count": str(report["coverage_endpoint_count"]),
            "coverage_fraction": f"{coverage_state.coverage_fraction:.6f}",
            "residual_count": str(len(coverage_state.residual_components)),
            "residual_backfill_count": str(residual_backfill_count),
            "invalid_path_length": f"{totals['invalid_path_length']:.6f}",
            "out_of_bounds_segment_count": str(int(totals["out_of_bounds_segment_count"])),
            "obstacle_collision_segment_count": str(int(totals["obstacle_collision_segment_count"])),
            "kinematic_infeasible_segment_count": str(int(totals["kinematic_infeasible_segment_count"])),
            "dynamic_infeasible_segment_count": str(int(totals["dynamic_infeasible_segment_count"])),
            "nmpc_untrackable_count": str(int(totals["nmpc_untrackable_count"])),
            "max_heading_jump": f"{totals['max_heading_jump']:.6f}",
            "max_yaw_rate": f"{totals['max_yaw_rate']:.6f}",
            "max_yaw_acceleration": f"{totals['max_yaw_acceleration']:.6f}",
            "infeasible_region_count": str(len(infeasible_regions)),
            "infeasible_edge_count": str(len(infeasible_edges)),
        },
    )
    if render:
        artifact_dir = _render_paper_style_outputs(
            config,
            obstacle_field,
            feasible_regions,
            sweep_paths,
            assignment.agent_regions,
            tsp_records,
            path_plan,
            report,
            output_dir,
            dpi=path_config.visual_dpi,
        )
        path_plan.metadata["paper_style_output_dir"] = str(artifact_dir)
        path_plan.metadata["paper_style_report"] = str(artifact_dir / "paper_style_region_tsp_report.json")
    return path_plan, report


def _coarsen_paper_style_regions(regions: Sequence, config: PlannerConfig) -> List:
    columns: Dict[Tuple[float, float], List] = {}
    for region in regions:
        x_min, _, x_max, _ = region.bounds
        columns.setdefault((round(x_min, 6), round(x_max, 6)), []).append(region)
    coarsened: List = []
    for idx, ((x_min, x_max), members) in enumerate(sorted(columns.items())):
        y_min = min(member.bounds[1] for member in members)
        y_max = max(member.bounds[3] for member in members)
        width = x_max - x_min
        height = y_max - y_min
        if width <= 1e-9 or height <= 1e-9:
            continue
        preferred_axis = "y" if height >= width else "x"
        polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        coarsened.append(
            replace(
                members[0],
                region_id=f"paper_col_{idx}",
                bounds=(x_min, y_min, x_max, y_max),
                polygon=polygon,
                center=((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
                area=width * height,
                preferred_axis=preferred_axis,
                source_algorithm="paper_style_column_coarsening",
                neighbors=[],
                metadata={
                    "paper_style_coarsened": "true",
                    "source_cell_count": str(len(members)),
                    "source_region_ids": ",".join(member.region_id for member in members),
                    "static_obstacle_aware": "true",
                },
            )
        )
    _populate_region_neighbors(coarsened)
    return coarsened or list(regions)


def _populate_region_neighbors(regions: List) -> None:
    for region in regions:
        region.neighbors.clear()
    for idx, first in enumerate(regions):
        for second in regions[idx + 1 :]:
            if _bounds_touch_or_overlap(first.bounds, second.bounds):
                first.neighbors.append(second.region_id)
                second.neighbors.append(first.region_id)


def _bounds_touch_or_overlap(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
) -> bool:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    x_overlap = min(ax1, bx1) - max(ax0, bx0)
    y_overlap = min(ay1, by1) - max(ay0, by0)
    x_touch = abs(ax1 - bx0) <= 1e-6 or abs(bx1 - ax0) <= 1e-6
    y_touch = abs(ay1 - by0) <= 1e-6 or abs(by1 - ay0) <= 1e-6
    return (x_touch and y_overlap >= -1e-6) or (y_touch and x_overlap >= -1e-6)


def _build_region_sweep_paths(
    raw_patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> Tuple[Dict[str, RegionSweepPath], Dict[str, List[RegionCoveragePattern]], List[Dict[str, object]]]:
    sweep_paths: Dict[str, RegionSweepPath] = {}
    feasible_patterns: Dict[str, List[RegionCoveragePattern]] = {}
    infeasible_regions: List[Dict[str, object]] = []
    for region_id, candidates in sorted(raw_patterns.items()):
        feasible_for_region: List[Tuple[RegionCoveragePattern, RegionSweepPath]] = []
        reasons: List[str] = []
        for pattern in candidates:
            for variant in _pattern_variants(pattern, config, path_config):
                variant = _normalize_pattern_headings(variant)
                valid, reason = _validate_internal_sweep(variant, config, path_config, obstacle_field)
                if valid:
                    feasible_for_region.append((variant, _sweep_path_from_pattern(variant)))
                else:
                    reasons.append(f"{variant.pattern_id}:{reason}")
        if feasible_for_region:
            max_coverage = max(item[0].coverage_length for item in feasible_for_region)
            coverage_ratio = (
                path_config.multi_entry_exit_coverage_floor
                if path_config.enable_multi_entry_exit_patterns
                else 0.8
            )
            coverage_floor = max(0.0, min(1.0, coverage_ratio)) * max_coverage
            feasible_for_region = [item for item in feasible_for_region if item[0].coverage_length + 1e-9 >= coverage_floor]
            feasible_for_region.sort(key=lambda item: _pattern_sort_key(item[0], config))
            limit = max(int(path_config.max_entry_exit_patterns_per_region), 1)
            feasible_for_region = feasible_for_region[:limit]
            feasible_patterns[region_id] = [item[0] for item in feasible_for_region]
            sweep_paths[region_id] = feasible_for_region[0][1]
        else:
            infeasible_regions.append({"region_id": region_id, "reasons": reasons[:6]})
    return sweep_paths, feasible_patterns, infeasible_regions


def _generate_paper_style_patterns(
    regions,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> Dict[str, List[RegionCoveragePattern]]:
    merged: Dict[str, List[RegionCoveragePattern]] = {region.region_id: [] for region in regions}
    for pocket_scale in (0.0, 0.5, 1.0):
        candidate_config = replace(path_config, coverage_turn_pocket_scale=pocket_scale)
        generated = generate_all_region_patterns(regions, config, candidate_config, obstacle_field=obstacle_field)
        for region_id, candidates in generated.items():
            for pattern in candidates:
                suffix = str(pocket_scale).replace(".", "p")
                merged[region_id].append(
                    replace(
                        pattern,
                        pattern_id=f"{pattern.pattern_id}_pocket_{suffix}",
                        metadata={**pattern.metadata, "turn_pocket_scale": f"{pocket_scale:.2f}"},
                    )
                )
    return merged


def _pattern_variants(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> List[RegionCoveragePattern]:
    if not path_config.enable_multi_entry_exit_patterns:
        return _legacy_pattern_variants(pattern, config)

    ordered_low_to_high = sorted(pattern.passes, key=lambda item: (item.center_coordinate, item.sequence_index))
    order_options = [
        ("low_to_high", ordered_low_to_high),
        ("high_to_low", list(reversed(ordered_low_to_high))),
    ]
    variants: List[RegionCoveragePattern] = []
    for order_name, ordered_passes in order_options:
        for start_side in ("min_to_max", "max_to_min"):
            rebuilt = _rebuild_boustrophedon_variant(pattern, ordered_passes, order_name, start_side)
            if rebuilt.passes:
                variants.append(_recalculate_pattern_cost(rebuilt, config))
    return _dedupe_pattern_variants(variants)


def _legacy_pattern_variants(pattern: RegionCoveragePattern, config: PlannerConfig) -> List[RegionCoveragePattern]:
    variants = [_recalculate_pattern_cost(pattern, config)]
    reversed_passes: List[CoveragePass] = []
    for idx, coverage_pass in enumerate(reversed(pattern.passes)):
        reversed_passes.append(
            CoveragePass(
                pass_id=f"{coverage_pass.pass_id}_rev",
                region_id=coverage_pass.region_id,
                sequence_index=idx,
                scan_axis=coverage_pass.scan_axis,
                start_pose=coverage_pass.end_pose,
                end_pose=coverage_pass.start_pose,
                center_coordinate=coverage_pass.center_coordinate,
                width=coverage_pass.width,
                length=coverage_pass.length,
            )
        )
    if reversed_passes:
        variants.append(
            replace(
                pattern,
                pattern_id=f"{pattern.pattern_id}_rev",
                passes=reversed_passes,
                entry_pose=reversed_passes[0].start_pose,
                exit_pose=reversed_passes[-1].end_pose,
                metadata={**pattern.metadata, "direction_variant": "reversed"},
            )
        )
    return _dedupe_pattern_variants([_recalculate_pattern_cost(item, config) for item in variants])


def _rebuild_boustrophedon_variant(
    pattern: RegionCoveragePattern,
    ordered_passes: Sequence[CoveragePass],
    order_name: str,
    start_side: str,
) -> RegionCoveragePattern:
    rebuilt_passes: List[CoveragePass] = []
    first_min_to_max = start_side == "min_to_max"
    for idx, coverage_pass in enumerate(ordered_passes):
        min_point, max_point = _pass_axis_endpoints(coverage_pass)
        use_min_to_max = first_min_to_max if idx % 2 == 0 else not first_min_to_max
        start_point, end_point = (min_point, max_point) if use_min_to_max else (max_point, min_point)
        heading = math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        rebuilt_passes.append(
            CoveragePass(
                pass_id=f"{coverage_pass.pass_id}_{order_name}_{start_side}_{idx}",
                region_id=coverage_pass.region_id,
                sequence_index=idx,
                scan_axis=coverage_pass.scan_axis,
                start_pose=Pose2D(start_point[0], start_point[1], heading),
                end_pose=Pose2D(end_point[0], end_point[1], heading),
                center_coordinate=coverage_pass.center_coordinate,
                width=coverage_pass.width,
                length=coverage_pass.length,
            )
        )
    if not rebuilt_passes:
        return pattern
    return replace(
        pattern,
        pattern_id=f"{pattern.pattern_id}_{order_name}_{start_side}",
        passes=rebuilt_passes,
        entry_pose=rebuilt_passes[0].start_pose,
        exit_pose=rebuilt_passes[-1].end_pose,
        metadata={
            **pattern.metadata,
            "base_pattern_id": pattern.pattern_id,
            "entry_exit_variant": f"{order_name}:{start_side}",
            "scan_order": order_name,
            "entry_side": start_side,
        },
    )


def _pass_axis_endpoints(coverage_pass: CoveragePass) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    points = [
        (coverage_pass.start_pose.x, coverage_pass.start_pose.y),
        (coverage_pass.end_pose.x, coverage_pass.end_pose.y),
    ]
    if coverage_pass.scan_axis == "x":
        points.sort(key=lambda item: (item[0], item[1]))
    else:
        points.sort(key=lambda item: (item[1], item[0]))
    return points[0], points[1]


def _recalculate_pattern_cost(pattern: RegionCoveragePattern, config: PlannerConfig) -> RegionCoveragePattern:
    coverage_length = sum(item.length for item in pattern.passes)
    turn_length = 0.0
    turn_angle = 0.0
    max_curvature = 0.0
    for current_pass, next_pass in zip(pattern.passes[:-1], pattern.passes[1:]):
        transition = dubins_shortest_path(current_pass.end_pose, next_pass.start_pose, config.fleet.min_turn_radius)
        turn_length += transition.total_length
        turn_angle += _dubins_turn_angle(transition.segment_lengths, transition.modes, config.fleet.min_turn_radius)
        max_curvature = max(max_curvature, 1.0 / max(config.fleet.min_turn_radius, 1e-6))
    cover_speed = max(config.fleet.cover_speed, 1e-6)
    turn_speed = max(min(config.fleet.turn_speed_max, config.fleet.cruise_speed), 1e-6)
    yaw_rate = max(config.fleet.turn_speed_max / max(config.fleet.min_turn_radius, 1e-6), 1e-6)
    estimated_time = coverage_length / cover_speed + turn_length / turn_speed + turn_angle / yaw_rate
    entry = pattern.passes[0].start_pose if pattern.passes else pattern.entry_pose
    exit_pose = pattern.passes[-1].end_pose if pattern.passes else pattern.exit_pose
    return replace(
        pattern,
        entry_pose=entry,
        exit_pose=exit_pose,
        coverage_length=coverage_length,
        turn_length=turn_length,
        turn_angle=turn_angle,
        total_length=coverage_length + turn_length,
        estimated_time=estimated_time,
        max_curvature=max_curvature,
        metadata={
            **pattern.metadata,
            "entry_pose": _pose_metadata(entry),
            "exit_pose": _pose_metadata(exit_pose),
            "internal_turn_length": f"{turn_length:.6f}",
            "internal_turn_angle": f"{turn_angle:.6f}",
        },
    )


def _dedupe_pattern_variants(patterns: Sequence[RegionCoveragePattern]) -> List[RegionCoveragePattern]:
    deduped: List[RegionCoveragePattern] = []
    seen = set()
    for pattern in patterns:
        key = tuple(
            (
                round(coverage_pass.start_pose.x, 4),
                round(coverage_pass.start_pose.y, 4),
                round(coverage_pass.end_pose.x, 4),
                round(coverage_pass.end_pose.y, 4),
            )
            for coverage_pass in pattern.passes
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pattern)
    return sorted(deduped, key=lambda item: (item.estimated_time, item.pattern_id))


def _pattern_sort_key(pattern: RegionCoveragePattern, config: PlannerConfig) -> Tuple[float, float, float, float, str]:
    endpoint_penalty = _turn_clearance_penalty(pattern.entry_pose, config) + _turn_clearance_penalty(pattern.exit_pose, config)
    return (-pattern.coverage_length, endpoint_penalty, pattern.estimated_time, pattern.total_length, pattern.pattern_id)


def _dubins_turn_angle(segment_lengths: Tuple[float, float, float], modes: Tuple[str, str, str], turn_radius: float) -> float:
    angle = 0.0
    for length, mode in zip(segment_lengths, modes):
        if mode in {"L", "R"}:
            angle += abs(length / max(turn_radius, 1e-6))
    return angle


def _pose_metadata(pose: Pose2D) -> str:
    return f"{pose.x:.3f},{pose.y:.3f},{pose.psi:.3f}"


def _normalize_pattern_headings(pattern: RegionCoveragePattern) -> RegionCoveragePattern:
    normalized_passes: List[CoveragePass] = []
    for coverage_pass in pattern.passes:
        heading = _line_heading(coverage_pass.start_pose, coverage_pass.end_pose)
        start_pose = Pose2D(coverage_pass.start_pose.x, coverage_pass.start_pose.y, heading)
        end_pose = Pose2D(coverage_pass.end_pose.x, coverage_pass.end_pose.y, heading)
        normalized_passes.append(
            CoveragePass(
                pass_id=coverage_pass.pass_id,
                region_id=coverage_pass.region_id,
                sequence_index=coverage_pass.sequence_index,
                scan_axis=coverage_pass.scan_axis,
                start_pose=start_pose,
                end_pose=end_pose,
                center_coordinate=coverage_pass.center_coordinate,
                width=coverage_pass.width,
                length=coverage_pass.length,
            )
        )
    if not normalized_passes:
        return pattern
    return replace(
        pattern,
        passes=normalized_passes,
        entry_pose=normalized_passes[0].start_pose,
        exit_pose=normalized_passes[-1].end_pose,
    )


def _line_heading(start: Pose2D, end: Pose2D) -> float:
    if abs(end.x - start.x) + abs(end.y - start.y) <= 1e-9:
        return start.psi
    return math.atan2(end.y - start.y, end.x - start.x)


def _validate_internal_sweep(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> Tuple[bool, str]:
    _, reason = _build_internal_sweep_segments(pattern, config, path_config, obstacle_field, start_time=0.0, segment_prefix="validate")
    return reason == "", reason


def _build_internal_sweep_segments(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    start_time: float,
    segment_prefix: str,
) -> Tuple[List[PathSegmentSpec], str]:
    segments: List[PathSegmentSpec] = []
    current_time = start_time
    for pass_idx, coverage_pass in enumerate(pattern.passes):
        cover = build_cover_segment(
            segment_id=f"{segment_prefix}_{coverage_pass.pass_id}",
            start=coverage_pass.start_pose,
            end=coverage_pass.end_pose,
            start_time=current_time,
            speed=max(config.fleet.cover_speed, 1e-6),
        )
        cover.metadata.update({"region_id": pattern.region_id, "pass_id": coverage_pass.pass_id, "sweep_endpoint_pair": "true"})
        reasons = path_segment_invalid_reasons(cover, config, obstacle_field)
        if reasons:
            return [], f"cover_invalid:{','.join(reasons)}"
        segments.append(cover)
        current_time = _segment_end_time(cover)
        if pass_idx >= len(pattern.passes) - 1:
            continue
        next_pass = pattern.passes[pass_idx + 1]
        turn = build_transition_segment(
            segment_id=f"{segment_prefix}_{coverage_pass.pass_id}_uturn",
            start=coverage_pass.end_pose,
            end=next_pass.start_pose,
            start_time=current_time,
            config=config,
            kind="turn",
            sample_count=32,
            use_bezier=path_config.use_bezier_smoothing,
        )
        turn.metadata.update({"region_id": pattern.region_id, "internal_uturn": "true", "kinematic_feasible": "true"})
        reasons = path_segment_invalid_reasons(turn, config, obstacle_field)
        if reasons:
            repaired_turns = build_obstacle_aware_transition_segments(
                segment_id=f"{segment_prefix}_{coverage_pass.pass_id}_uturn_repair",
                start=coverage_pass.end_pose,
                end=next_pass.start_pose,
                start_time=current_time,
                config=config,
                path_config=path_config,
                obstacle_field=obstacle_field,
                kind="turn",
                sample_count=48,
            )
            if not _segments_strictly_valid(repaired_turns, config, obstacle_field):
                return [], f"uturn_invalid:{','.join(reasons)}"
            for repaired in repaired_turns:
                repaired.metadata.update(
                    {
                        "region_id": pattern.region_id,
                        "internal_uturn": "true",
                        "uturn_repair": "true",
                    }
                )
            segments.extend(repaired_turns)
            current_time = _segment_end_time(repaired_turns[-1])
            continue
        turn_report = validate_transition_dynamics(turn, config, obstacle_field=obstacle_field, retime=True)
        if not turn_report.valid:
            return [], f"uturn_dynamic_violation:{','.join(turn_report.reasons)}"
        segments.append(turn)
        current_time = _segment_end_time(turn)
    return segments, ""


def _sweep_path_from_pattern(pattern: RegionCoveragePattern) -> RegionSweepPath:
    endpoints: List[Pose2D] = []
    for coverage_pass in pattern.passes:
        endpoints.extend([coverage_pass.start_pose, coverage_pass.end_pose])
    return RegionSweepPath(
        region_id=pattern.region_id,
        pattern_id=pattern.pattern_id,
        passes=list(pattern.passes),
        endpoints=endpoints,
        entry_pose=pattern.entry_pose,
        exit_pose=pattern.exit_pose,
        feasible=True,
        metadata={"endpoint_count": str(len(endpoints))},
    )


def _region_visit_nodes(patterns: Dict[str, List[RegionCoveragePattern]]) -> Dict[str, RegionVisitNode]:
    nodes: Dict[str, RegionVisitNode] = {}
    for region_id, candidates in patterns.items():
        pattern = candidates[0]
        nodes[region_id] = RegionVisitNode(
            region_id=region_id,
            pattern_id=pattern.pattern_id,
            entry_pose=pattern.entry_pose,
            exit_pose=pattern.exit_pose,
            pass_count=len(pattern.passes),
            coverage_endpoint_count=2 * len(pattern.passes),
            estimated_time=pattern.estimated_time,
        )
    return nodes


def _solve_agent_region_tsp(
    agent_id: int,
    region_ids: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    sweep_paths: Dict[str, RegionSweepPath],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> Dict[str, object]:
    start_pose = config.fleet.initial_states_3dof[agent_id].pose()
    initial_order = _nearest_neighbor_region_order(start_pose, region_ids, patterns, config)
    requested_solver = validate_tsp_solver(path_config.tsp_solver)
    fallback_solver_metadata: Dict[str, object] | None = None
    if requested_solver != "deterministic":
        aco_result = _solve_agent_region_tsp_aco(
            agent_id,
            region_ids,
            patterns,
            config,
            path_config,
            obstacle_field,
            initial_order,
        )
        if aco_result["tsp_solver_metadata"]["tsp_solver_status"] == "success":
            return aco_result
        fallback_solver_metadata = dict(aco_result["tsp_solver_metadata"])
    candidate_pattern_counts = {region_id: len(patterns.get(region_id, [])) for region_id in region_ids}
    candidate_attempt_count = 0
    rejected_candidate_count = 0
    order_rank = {region_id: idx for idx, region_id in enumerate(initial_order)}
    beam_width = max(int(path_config.region_tsp_beam_width), 1)
    branch_limit = max(int(path_config.region_tsp_branch_limit), 1)
    initial_state = {
        "segments": [],
        "final_order": [],
        "selected_patterns": {},
        "selected_pattern_ids": {},
        "remaining": set(initial_order),
        "current_pose": start_pose,
        "current_time": 0.0,
        "serial": 0,
        "score": 0.0,
        "rejections": [],
    }
    beam = [initial_state]
    best_partial = initial_state
    complete_states: List[Dict[str, object]] = []
    terminal_rejections: List[Dict[str, object]] = []

    for _ in range(len(initial_order)):
        next_beam: List[Dict[str, object]] = []
        depth_rejections: List[Dict[str, object]] = []
        for state in beam:
            remaining = set(state["remaining"])
            if not remaining:
                complete_states.append(state)
                continue
            current_pose = state["current_pose"]
            current_time = float(state["current_time"])
            serial = int(state["serial"])
            ordered_candidates: List[Tuple[float, str, RegionCoveragePattern]] = []
            for region_id in sorted(remaining, key=lambda item: order_rank.get(item, 10_000)):
                for pattern in patterns[region_id]:
                    coverage_reward = 2.0 * pattern.coverage_length
                    ordered_candidates.append(
                        (
                            _transition_length(current_pose, pattern.entry_pose, config)
                            + pattern.total_length
                            + _turn_clearance_penalty(pattern.entry_pose, config)
                            + _turn_clearance_penalty(pattern.exit_pose, config)
                            - coverage_reward,
                            region_id,
                            pattern,
                        )
                    )
            ordered_candidates.sort(key=lambda item: (item[0], item[1], item[2].pattern_id))
            candidate_slice = ordered_candidates if not state["final_order"] else ordered_candidates[:branch_limit]
            for _, region_id, candidate_pattern in candidate_slice:
                candidate_attempt_count += 1
                rejected_edges: List[Dict[str, object]] = []
                connector = _build_region_connector(
                    agent_id,
                    serial,
                    current_pose,
                    candidate_pattern.entry_pose,
                    current_time,
                    config,
                    path_config,
                    obstacle_field,
                    to_region=region_id,
                    rejection_sink=rejected_edges,
                )
                if connector is None:
                    rejected_candidate_count += 1
                    depth_rejections.extend(rejected_edges)
                    continue
                connector_end_time = current_time
                if connector:
                    connector_end_time = _segment_end_time(connector[-1])
                sweep_segments, reason = _build_internal_sweep_segments(
                    candidate_pattern,
                    config,
                    path_config,
                    obstacle_field,
                    start_time=connector_end_time,
                    segment_prefix=f"agent{agent_id}_region_{region_id}",
                )
                if reason:
                    rejected_candidate_count += 1
                    depth_rejections.append(
                        {
                            "agent_id": agent_id,
                            "region_id": region_id,
                            "pattern_id": candidate_pattern.pattern_id,
                            "reason": reason,
                        }
                    )
                    continue
                new_remaining = set(remaining)
                new_remaining.remove(region_id)
                new_segments = list(state["segments"]) + list(connector) + list(sweep_segments)
                new_final_order = list(state["final_order"]) + [region_id]
                new_selected_patterns = dict(state["selected_patterns"])
                new_selected_patterns[region_id] = candidate_pattern
                new_selected_pattern_ids = dict(state["selected_pattern_ids"])
                new_selected_pattern_ids[region_id] = candidate_pattern.pattern_id
                step_score = (
                    dynamic_edge_cost(connector, config)
                    + candidate_pattern.estimated_time
                    + _turn_clearance_penalty(candidate_pattern.exit_pose, config)
                )
                next_beam.append(
                    {
                        "segments": new_segments,
                        "final_order": new_final_order,
                        "selected_patterns": new_selected_patterns,
                        "selected_pattern_ids": new_selected_pattern_ids,
                        "remaining": new_remaining,
                        "current_pose": candidate_pattern.exit_pose,
                        "current_time": _segment_end_time(sweep_segments[-1]),
                        "serial": serial + len(connector) + len(sweep_segments),
                        "score": float(state["score"]) + step_score,
                        "rejections": list(state["rejections"]) + depth_rejections,
                    }
                )
                if not state["final_order"] and len(next_beam) >= beam_width:
                    break
        if next_beam:
            next_beam.sort(key=lambda item: (len(item["remaining"]), float(item["score"]), list(item["final_order"])))
            beam = next_beam[:beam_width]
            best_partial = min(
                [best_partial] + beam,
                key=lambda item: (len(item["remaining"]), float(item["score"]), list(item["final_order"])),
            )
        else:
            terminal_rejections.extend(depth_rejections)
            break
    complete_states.extend([state for state in beam if not state["remaining"]])
    if complete_states:
        chosen_state = min(complete_states, key=lambda item: (float(item["score"]), list(item["final_order"])))
        infeasible_edges: List[Dict[str, object]] = []
    else:
        chosen_state = best_partial
        infeasible_edges = terminal_rejections or list(chosen_state["rejections"])
    return {
        "initial_order": initial_order,
        "final_order": list(chosen_state["final_order"]),
        "segments": list(chosen_state["segments"]),
        "infeasible_edges": infeasible_edges,
        "selected_patterns": dict(chosen_state["selected_patterns"]),
        "selected_pattern_ids": dict(chosen_state["selected_pattern_ids"]),
        "candidate_pattern_counts": candidate_pattern_counts,
        "candidate_attempt_count": candidate_attempt_count,
        "rejected_candidate_count": rejected_candidate_count,
        "tsp_solver_metadata": fallback_solver_metadata
        or {
            "requested_tsp_solver": requested_solver,
            "effective_tsp_solver": "deterministic",
            "tsp_solver_status": "success",
            "aco_best_objective": None,
            "aco_initial_objective": None,
            "aco_iteration_count": 0,
            "aco_convergence_trace": [],
            "aco_accepted_3opt_count": 0,
        },
    }


def _solve_agent_region_tsp_aco(
    agent_id: int,
    region_ids: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    initial_order: Sequence[str],
) -> Dict[str, object]:
    start_pose = config.fleet.initial_states_3dof[agent_id].pose()
    candidate_pattern_counts = {region_id: len(patterns.get(region_id, [])) for region_id in region_ids}
    invalid_edges: List[Dict[str, object]] = []
    cost_cache: Dict[Tuple[str, str], float] = {}

    def pattern_key(pattern: RegionCoveragePattern | None) -> str:
        return "__start__" if pattern is None else f"{pattern.region_id}:{pattern.pattern_id}"

    def edge_cost(previous: RegionCoveragePattern | None, candidate_pattern: RegionCoveragePattern) -> float:
        key = (pattern_key(previous), pattern_key(candidate_pattern))
        if key in cost_cache:
            return cost_cache[key]
        current_pose = start_pose if previous is None else previous.exit_pose
        transition = dubins_shortest_path(current_pose, candidate_pattern.entry_pose, config.fleet.min_turn_radius)
        transition_turn = _dubins_turn_angle(transition.segment_lengths, transition.modes, config.fleet.min_turn_radius)
        transition_time = transition.total_length / max(config.fleet.cruise_speed, 1e-6)
        cost_cache[key] = (
            path_config.length_weight * (transition.total_length + candidate_pattern.total_length)
            + path_config.turn_angle_weight * (transition_turn + candidate_pattern.turn_angle)
            + path_config.time_weight * (transition_time + candidate_pattern.estimated_time)
            + _turn_clearance_penalty(candidate_pattern.entry_pose, config)
            + _turn_clearance_penalty(candidate_pattern.exit_pose, config)
        )
        return cost_cache[key]

    result = solve_aco_tsp_cpp(
        region_ids=region_ids,
        patterns=patterns,
        start_pose=start_pose,
        path_config=path_config,
        edge_cost_fn=edge_cost,
        solver=path_config.tsp_solver,
    )
    if result.status != "success":
        return _paper_aco_failure_result(
            initial_order,
            candidate_pattern_counts,
            invalid_edges,
            _paper_aco_metadata(result),
            candidate_attempt_count=len(cost_cache),
        )

    segments: List[PathSegmentSpec] = []
    infeasible_edges: List[Dict[str, object]] = []
    current_pose = start_pose
    current_time = 0.0
    serial = 0
    for region_id in result.region_order:
        pattern = result.selected_patterns[region_id]
        connector = _build_region_connector(
            agent_id,
            serial,
            current_pose,
            pattern.entry_pose,
            current_time,
            config,
            path_config,
            obstacle_field,
            to_region=region_id,
            rejection_sink=infeasible_edges,
        )
        if connector is None:
            metadata = _paper_aco_metadata(
                result,
                status="failed",
                effective_solver="deterministic_fallback",
                extra={"failure_reason": "aco_selected_edge_failed_reassembly"},
            )
            return _paper_aco_failure_result(
                initial_order,
                candidate_pattern_counts,
                infeasible_edges,
                metadata,
                candidate_attempt_count=len(cost_cache),
            )
        connector_end_time = _segment_end_time(connector[-1]) if connector else current_time
        sweep_segments, reason = _build_internal_sweep_segments(
            pattern,
            config,
            path_config,
            obstacle_field,
            start_time=connector_end_time,
            segment_prefix=f"agent{agent_id}_region_{region_id}",
        )
        if reason:
            metadata = _paper_aco_metadata(
                result,
                status="failed",
                effective_solver="deterministic_fallback",
                extra={"failure_reason": reason},
            )
            return _paper_aco_failure_result(
                initial_order,
                candidate_pattern_counts,
                [{"agent_id": agent_id, "region_id": region_id, "reason": reason}],
                metadata,
                candidate_attempt_count=len(cost_cache),
            )
        segments.extend(connector)
        segments.extend(sweep_segments)
        serial += len(connector) + len(sweep_segments)
        current_time = _segment_end_time(sweep_segments[-1])
        current_pose = pattern.exit_pose

    metadata = _paper_aco_metadata(result)
    selected_pattern_ids = {region_id: pattern.pattern_id for region_id, pattern in result.selected_patterns.items()}
    return {
        "initial_order": list(result.metadata.get("initial_order", initial_order)),
        "final_order": list(result.region_order),
        "segments": segments,
        "infeasible_edges": [],
        "selected_patterns": dict(result.selected_patterns),
        "selected_pattern_ids": selected_pattern_ids,
        "candidate_pattern_counts": candidate_pattern_counts,
        "candidate_attempt_count": len(cost_cache),
        "rejected_candidate_count": len([value for value in cost_cache.values() if not math.isfinite(value)]),
        "tsp_solver_metadata": metadata,
    }


def _paper_aco_failure_result(
    initial_order: Sequence[str],
    candidate_pattern_counts: Dict[str, int],
    infeasible_edges: List[Dict[str, object]],
    metadata: Dict[str, object],
    candidate_attempt_count: int,
) -> Dict[str, object]:
    return {
        "initial_order": list(initial_order),
        "final_order": [],
        "segments": [],
        "infeasible_edges": infeasible_edges,
        "selected_patterns": {},
        "selected_pattern_ids": {},
        "candidate_pattern_counts": candidate_pattern_counts,
        "candidate_attempt_count": candidate_attempt_count,
        "rejected_candidate_count": len(infeasible_edges),
        "tsp_solver_metadata": metadata,
    }


def _paper_aco_metadata(
    result: AcoTspResult,
    status: str | None = None,
    effective_solver: str | None = None,
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    return {
        "requested_tsp_solver": result.requested_solver,
        "effective_tsp_solver": effective_solver or result.effective_solver,
        "tsp_solver_status": status or result.status,
        "aco_best_objective": None if not math.isfinite(result.objective) else float(result.objective),
        "aco_initial_objective": None if not math.isfinite(result.initial_objective) else float(result.initial_objective),
        "aco_iteration_count": int(result.metadata.get("iteration_count", 0)),
        "aco_convergence_trace": list(result.convergence_trace),
        "aco_accepted_3opt_count": int(result.accepted_3opt_count),
        **(extra or result.metadata),
    }


def _append_paper_style_residual_backfill(
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    agents: Dict[int, AgentPathPlan],
    tours: Dict[int, SingleUsvTourPlan],
    coverage_state,
) -> int:
    appended = 0
    residuals = sorted(
        coverage_state.residual_components,
        key=lambda item: len(item.cells),
        reverse=True,
    )[: max(path_config.max_residual_backfill_regions, 0)]
    for residual in residuals:
        candidates = _residual_cover_candidates(residual, coverage_state.resolution, config, obstacle_field)
        if not candidates:
            continue
        best_choice = None
        for agent_id, agent in sorted(agents.items()):
            current_pose = _agent_end_pose(agent, config)
            current_time = max((_segment_end_time(segment) for segment in agent.segments), default=0.0)
            for cover in candidates:
                connector = _build_region_connector(
                    agent_id,
                    len(agent.segments),
                    current_pose,
                    Pose2D(cover.waypoints[0].x, cover.waypoints[0].y, cover.waypoints[0].psi),
                    current_time,
                    config,
                    path_config,
                    obstacle_field,
                    to_region=f"residual_{residual.residual_id}",
                )
                if connector is None:
                    continue
                score = sum(segment.length for segment in connector) + cover.length
                if best_choice is None or score < best_choice[0]:
                    best_choice = (score, agent_id, connector, cover)
        if best_choice is None:
            continue
        _, agent_id, connector, cover = best_choice
        agent = agents[agent_id]
        current_time = max((_segment_end_time(segment) for segment in agent.segments), default=0.0)
        timed_connector: List[PathSegmentSpec] = []
        for segment in connector:
            timed_connector.append(segment)
            current_time = _segment_end_time(segment)
        cover = _retime_segment(cover, current_time)
        agent.segments.extend(timed_connector)
        agent.segments.append(cover)
        tour = tours[agent_id]
        tour.segments = agent.segments
        tour.region_order.append(f"residual_{residual.residual_id}")
        tour.diagnostics["paper_style_residual_backfill"] = "true"
        appended += 1
    return appended


def _residual_cover_candidates(
    residual,
    resolution: float,
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
) -> List[PathSegmentSpec]:
    x_min, y_min, x_max, y_max = residual.bounds
    pad = max(float(resolution) / 2.0, 1e-6)
    x0 = max(0.0, x_min - pad)
    x1 = min(config.mission.area_length_x, x_max + pad)
    y0 = max(0.0, y_min - pad)
    y1 = min(config.mission.area_length_y, y_max + pad)
    cx, cy = residual.centroid
    raw: List[Tuple[Pose2D, Pose2D, str]] = [
        (Pose2D(x0, cy, 0.0), Pose2D(x1, cy, 0.0), "x"),
        (Pose2D(x1, cy, math.pi), Pose2D(x0, cy, math.pi), "x_rev"),
        (Pose2D(cx, y0, math.pi / 2.0), Pose2D(cx, y1, math.pi / 2.0), "y"),
        (Pose2D(cx, y1, -math.pi / 2.0), Pose2D(cx, y0, -math.pi / 2.0), "y_rev"),
    ]
    candidates: List[PathSegmentSpec] = []
    for start, end, axis in raw:
        if math.hypot(end.x - start.x, end.y - start.y) <= 1e-9:
            continue
        cover = build_cover_segment(
            segment_id=f"residual_{residual.residual_id}_{axis}",
            start=start,
            end=end,
            start_time=0.0,
            speed=max(config.fleet.cover_speed, 1e-6),
            sample_count=12,
        )
        cover.metadata.update(
            {
                "region_id": f"residual_{residual.residual_id}",
                "residual_backfill": "true",
                "sweep_endpoint_pair": "true",
            }
        )
        if path_segment_invalid_reasons(cover, config, obstacle_field):
            continue
        if obstacle_field is not None and sampled_segment_footprint_collides(
            start,
            end,
            config.footprint.length_lf,
            config.footprint.width_wf,
            obstacle_field,
            sample_spacing=max(config.footprint.width_wf / 2.0, 1e-6),
            inflated=False,
        ):
            continue
        candidates.append(cover)
    return sorted(candidates, key=lambda item: item.length)


def _agent_end_pose(agent: AgentPathPlan, config: PlannerConfig) -> Pose2D:
    for segment in reversed(agent.segments):
        if segment.waypoints:
            waypoint = segment.waypoints[-1]
            return Pose2D(waypoint.x, waypoint.y, waypoint.psi)
    state = config.fleet.initial_states_3dof[agent.agent_id]
    return state.pose()


def _retime_segment(segment: PathSegmentSpec, start_time: float) -> PathSegmentSpec:
    if not segment.waypoints:
        return segment
    original_start = segment.waypoints[0].time or 0.0
    delta = start_time - original_start
    segment.waypoints = [replace(waypoint, time=(waypoint.time or 0.0) + delta) for waypoint in segment.waypoints]
    return segment


def _nearest_neighbor_region_order(
    start_pose: Pose2D,
    region_ids: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
) -> List[str]:
    remaining = set(region_ids)
    order: List[str] = []
    current = start_pose
    while remaining:
        region_id = min(
            sorted(remaining),
            key=lambda item: _transition_length(current, patterns[item][0].entry_pose, config) + patterns[item][0].estimated_time,
        )
        order.append(region_id)
        current = patterns[region_id][0].exit_pose
        remaining.remove(region_id)
    return order


def _two_opt_region_order(
    start_pose: Pose2D,
    order: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
) -> List[str]:
    best = list(order)
    best_cost = _region_order_cost(start_pose, best, patterns, config)
    changed = True
    while changed and len(best) > 3:
        changed = False
        for i in range(0, len(best) - 2):
            for j in range(i + 2, len(best) + 1):
                candidate = best[:i] + list(reversed(best[i:j])) + best[j:]
                cost = _region_order_cost(start_pose, candidate, patterns, config)
                if cost + 1e-9 < best_cost:
                    best = candidate
                    best_cost = cost
                    changed = True
                    break
            if changed:
                break
    return best


def _region_order_cost(
    start_pose: Pose2D,
    order: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
) -> float:
    current = start_pose
    cost = 0.0
    for region_id in order:
        pattern = patterns[region_id][0]
        cost += _transition_length(current, pattern.entry_pose, config) + pattern.total_length
        current = pattern.exit_pose
    return cost


def _build_region_connector(
    agent_id: int,
    serial: int,
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    to_region: str,
    rejection_sink: List[Dict[str, object]] | None = None,
) -> List[PathSegmentSpec] | None:
    segments = build_obstacle_aware_transition_segments(
        segment_id=f"agent{agent_id}_region_edge_{serial}_to_{to_region}",
        start=start,
        end=end,
        start_time=start_time,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        kind="transit",
        sample_count=48,
    )
    report = validate_transition_sequence(segments, config, obstacle_field=obstacle_field, retime=True)
    if not report.valid or not _segments_strictly_valid(segments, config, obstacle_field):
        if rejection_sink is not None:
            rejection_sink.append(
                {
                    "agent_id": agent_id,
                    "from": _pose_label(start),
                    "to_region": to_region,
                    "reason": ",".join(report.reasons) or "dynamic_validation_failed",
                    "segment_sources": [segment.path_source for segment in segments],
                    "max_curvature": report.max_curvature,
                    "max_heading_jump": report.max_heading_jump,
                    "max_yaw_rate": report.max_yaw_rate,
                    "max_yaw_acceleration": report.max_yaw_acceleration,
                    "max_speed": report.max_speed,
                    "max_thrust_required": report.max_thrust_required,
                    "max_yaw_moment_required": report.max_yaw_moment_required,
                }
            )
        return None
    for idx, segment in enumerate(segments):
        segment.metadata.update(
            {
                "to_region": to_region,
                "region_tsp_edge": "true",
                "resource_id": f"region_tsp:{agent_id}:{to_region}:{idx}",
                "dynamic_edge_cost": f"{dynamic_edge_cost([segment], config):.6f}",
            }
        )
    return [segment for segment in segments if segment.length > 1e-9]


def _segments_strictly_valid(
    segments: Sequence[PathSegmentSpec],
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
) -> bool:
    return validate_transition_sequence(segments, config, obstacle_field=obstacle_field, retime=True).valid


def _agent_metrics(
    segments: Sequence[PathSegmentSpec],
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
) -> Dict[str, float]:
    total_length = sum(segment.length for segment in segments)
    total_turn = sum(_segment_heading_variation(segment) for segment in segments)
    estimated_time = max((_segment_end_time(segment) for segment in segments), default=0.0)
    invalid_length = sum(path_segment_invalid_length(segment, config, obstacle_field) for segment in segments)
    out_of_bounds = 0
    collision = 0
    kinematic = 0
    dynamic = 0
    nmpc_untrackable = 0
    max_heading_jump = 0.0
    max_heading_error = 0.0
    max_yaw_rate = 0.0
    max_yaw_acceleration = 0.0
    max_speed = 0.0
    max_acceleration = 0.0
    max_thrust_required = 0.0
    max_yaw_moment_required = 0.0
    for segment in segments:
        reasons = path_segment_invalid_reasons(segment, config, obstacle_field)
        if "out_of_bounds" in reasons:
            out_of_bounds += 1
        if "obstacle_collision" in reasons:
            collision += 1
        if segment.metadata.get("kinematic_feasible") == "false" or segment.path_source == "astar_corridor_edge":
            kinematic += 1
        report = validate_transition_dynamics(segment, config, obstacle_field=obstacle_field, retime=False)
        if not report.valid:
            dynamic += 1
        if not report.nmpc_trackable:
            nmpc_untrackable += 1
        max_heading_jump = max(max_heading_jump, report.max_heading_jump)
        max_heading_error = max(max_heading_error, report.max_heading_error)
        max_yaw_rate = max(max_yaw_rate, report.max_yaw_rate)
        max_yaw_acceleration = max(max_yaw_acceleration, report.max_yaw_acceleration)
        max_speed = max(max_speed, report.max_speed)
        max_acceleration = max(max_acceleration, report.max_acceleration)
        max_thrust_required = max(max_thrust_required, report.max_thrust_required)
        max_yaw_moment_required = max(max_yaw_moment_required, report.max_yaw_moment_required)
    sequence_report = validate_transition_sequence(segments, config, obstacle_field=obstacle_field, retime=False)
    max_heading_jump = max(max_heading_jump, sequence_report.max_heading_jump)
    if not sequence_report.valid:
        dynamic += 1
        nmpc_untrackable += 1
    return {
        "total_length": total_length,
        "coverage_length": sum(segment.length for segment in segments if segment.kind == "cover"),
        "transition_length": sum(segment.length for segment in segments if segment.kind != "cover"),
        "total_turn_angle": total_turn,
        "estimated_time": estimated_time,
        "max_curvature": max((segment.curvature_max for segment in segments), default=0.0),
        "turn_count": float(sum(1 for segment in segments if segment.kind == "turn")),
        "segment_count": float(len(segments)),
        "invalid_path_length": invalid_length,
        "out_of_bounds_segment_count": float(out_of_bounds),
        "obstacle_collision_segment_count": float(collision),
        "kinematic_infeasible_segment_count": float(kinematic),
        "dynamic_infeasible_segment_count": float(dynamic),
        "nmpc_untrackable_count": float(nmpc_untrackable),
        "max_heading_jump": max_heading_jump,
        "max_heading_error": max_heading_error,
        "max_yaw_rate": max_yaw_rate,
        "max_yaw_acceleration": max_yaw_acceleration,
        "max_speed": max_speed,
        "max_acceleration": max_acceleration,
        "max_thrust_required": max_thrust_required,
        "max_yaw_moment_required": max_yaw_moment_required,
    }


def _global_metrics(agents: Dict[int, AgentPathPlan]) -> Dict[str, float]:
    keys = [
        "total_length",
        "coverage_length",
        "transition_length",
        "total_turn_angle",
        "turn_count",
        "invalid_path_length",
        "out_of_bounds_segment_count",
        "obstacle_collision_segment_count",
        "kinematic_infeasible_segment_count",
        "dynamic_infeasible_segment_count",
        "nmpc_untrackable_count",
    ]
    totals = {key: 0.0 for key in keys}
    totals["max_curvature"] = 0.0
    for max_key in (
        "max_heading_jump",
        "max_heading_error",
        "max_yaw_rate",
        "max_yaw_acceleration",
        "max_speed",
        "max_acceleration",
        "max_thrust_required",
        "max_yaw_moment_required",
    ):
        totals[max_key] = 0.0
    for agent in agents.values():
        for key in keys:
            totals[key] += float(agent.metrics.get(key, 0.0))
        totals["max_curvature"] = max(totals["max_curvature"], float(agent.metrics.get("max_curvature", 0.0)))
        for max_key in (
            "max_heading_jump",
            "max_heading_error",
            "max_yaw_rate",
            "max_yaw_acceleration",
            "max_speed",
            "max_acceleration",
            "max_thrust_required",
            "max_yaw_moment_required",
        ):
            totals[max_key] = max(totals[max_key], float(agent.metrics.get(max_key, 0.0)))
    return totals


def _transition_length(start: Pose2D, end: Pose2D, config: PlannerConfig) -> float:
    return math.hypot(end.x - start.x, end.y - start.y) + config.fleet.min_turn_radius * abs(wrap_angle(end.psi - start.psi))


def _turn_clearance_penalty(pose: Pose2D, config: PlannerConfig) -> float:
    clearance = min(
        pose.x,
        config.mission.area_length_x - pose.x,
        pose.y,
        config.mission.area_length_y - pose.y,
    )
    shortfall = max(0.0, config.fleet.min_turn_radius - clearance)
    return 25.0 * shortfall * shortfall


def _segment_heading_variation(segment: PathSegmentSpec) -> float:
    headings = [waypoint.psi for waypoint in segment.waypoints]
    return sum(abs(wrap_angle(headings[idx] - headings[idx - 1])) for idx in range(1, len(headings)))


def _segment_end_time(segment: PathSegmentSpec) -> float:
    if not segment.waypoints or segment.waypoints[-1].time is None:
        return 0.0
    return float(segment.waypoints[-1].time)


def _pose_label(pose: Pose2D) -> List[float]:
    return [round(pose.x, 3), round(pose.y, 3), round(pose.psi, 3)]


def _render_paper_style_outputs(
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
    regions,
    sweep_paths: Dict[str, RegionSweepPath],
    assignment: Dict[int, List[str]],
    tsp_records: Dict[int, Dict[str, object]],
    path_plan: MultiAgentPathPlan,
    report: Dict[str, object],
    output_dir: str | Path,
    dpi: int,
) -> Path:
    output = Path(output_dir) / "paper_style_region_tsp"
    output.mkdir(parents=True, exist_ok=True)

    def save(fig, filename: str) -> None:
        fig.savefig(output / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    fig, ax = _new_map_axes(config, "00 Paper-Style Map")
    _draw_obstacles(ax, obstacle_field, raw=True, inflated=False)
    save(fig, "00_map_and_static_obstacles.png")

    fig, ax = _new_map_axes(config, "01 Inflated Obstacles")
    _draw_obstacles(ax, obstacle_field, raw=True, inflated=True)
    save(fig, "01_obstacle_inflation.png")

    fig, ax = _new_map_axes(config, "02 Free-Space Regions")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    for region in regions:
        _draw_polygon(ax, region.polygon, facecolor="#bde0fe", edgecolor="#457b9d", alpha=0.18, linewidth=0.6)
        ax.text(region.center[0], region.center[1], region.region_id.replace("free_cell_", "c"), fontsize=5, ha="center")
    save(fig, "02_free_space_regions.png")

    fig, ax = _new_map_axes(config, "03 Feasible Region Sweep Modes")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    _draw_sweeps(ax, sweep_paths, color="#0b5fff", endpoints=False)
    save(fig, "03_feasible_region_sweep_modes.png")

    fig, ax = _new_map_axes(config, "04 Region Sweep Patterns")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    _draw_sweeps(ax, sweep_paths, color="#0b5fff", endpoints=False)
    save(fig, "04_region_sweep_patterns.png")

    fig, ax = _new_map_axes(config, "04 Selected Sweep Patterns")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    _draw_actual_cover_endpoints(ax, path_plan, draw_lines=True, add_agent_labels=True)
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "04_selected_region_sweep_patterns.png")

    fig, ax = _new_map_axes(config, "05 Region TSP Nodes")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    for sweep in sweep_paths.values():
        ax.plot(sweep.entry_pose.x, sweep.entry_pose.y, marker=">", color="#0b5fff", markersize=4)
        ax.plot(sweep.exit_pose.x, sweep.exit_pose.y, marker="s", color="#f2c94c", markersize=4)
        ax.text(sweep.entry_pose.x, sweep.entry_pose.y, sweep.region_id.replace("free_cell_", "c"), fontsize=5)
    save(fig, "05_region_tsp_nodes.png")

    fig, ax = _new_map_axes(config, "06 Agent Region TSP Order")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    for agent_id, record in sorted(tsp_records.items()):
        color = _agent_color(agent_id)
        centers = []
        for seq, region_id in enumerate(record.get("final_order", [])):
            sweep = sweep_paths.get(region_id)
            if sweep is None:
                continue
            centers.append((sweep.entry_pose.x, sweep.entry_pose.y))
            ax.text(sweep.entry_pose.x, sweep.entry_pose.y, str(seq + 1), color=color, fontsize=7, ha="center")
        for start, end in zip(centers[:-1], centers[1:]):
            ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.0})
        ax.plot([], [], color=color, label=f"USV {agent_id}")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "06_agent_region_tsp_order.png")

    fig, ax = _new_map_axes(config, "07 Selected Sweep Endpoints")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    _draw_actual_cover_endpoints(ax, path_plan, draw_lines=True, add_agent_labels=True)
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "07_agent_sweep_endpoints.png")

    fig, ax = _new_map_axes(config, "08 Final Region-TSP Coverage Path")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    for agent_id, agent in sorted(path_plan.agents.items()):
        color = _agent_color(agent_id)
        for segment in agent.segments:
            if segment.kind == "cover":
                _plot_segment(ax, segment, color=color, linestyle="-", linewidth=1.8, alpha=0.95)
            else:
                _plot_segment(ax, segment, color=color, linestyle="--", linewidth=1.1, alpha=0.55)
        ax.plot([], [], color=color, label=f"USV {agent_id}")
    _draw_actual_cover_endpoints(ax, path_plan, draw_lines=False, add_agent_labels=False)
    ax.plot([], [], marker="o", color="#111827", linestyle="", label="cover start")
    ax.plot([], [], marker="s", color="#111827", linestyle="", label="cover end")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "08_final_region_tsp_coverage_path.png")

    fig, ax = _new_map_axes(config, "09 Constraint Validation")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    for agent_id, agent in sorted(path_plan.agents.items()):
        color = _agent_color(agent_id)
        for segment in agent.segments:
            dynamic_ok = segment.metadata.get("dynamic_feasible") != "false"
            line_color = color if dynamic_ok else "#d00000"
            linestyle = "-" if dynamic_ok else ":"
            _plot_segment(ax, segment, color=line_color, linestyle=linestyle, linewidth=1.2, alpha=0.85)
            if segment.metadata.get("region_tsp_edge") == "true" and segment.waypoints:
                waypoint = segment.waypoints[min(1, len(segment.waypoints) - 1)]
                label = str(segment.metadata.get("connector") or segment.path_source)
                if not dynamic_ok:
                    label = f"rejected:{segment.metadata.get('dynamic_invalid_reasons', 'dynamic')}"
                ax.text(waypoint.x, waypoint.y, label, fontsize=4.5, color=line_color)
        ax.plot([], [], color=color, label=f"USV {agent_id}")
    for item in report.get("infeasible_edges", []):
        start = item.get("from")
        if isinstance(start, list) and len(start) >= 2:
            ax.plot(float(start[0]), float(start[1]), marker="x", color="#d00000", markersize=4)
    metrics = report.get("metrics", {})
    _info_box_outside(
        ax,
        "\n".join(
            [
                f"invalid_length={metrics.get('invalid_path_length', 0.0):.3f}",
                f"out_of_bounds={metrics.get('out_of_bounds_segment_count', 0.0):.0f}",
                f"collisions={metrics.get('obstacle_collision_segment_count', 0.0):.0f}",
                f"kinematic_bad={metrics.get('kinematic_infeasible_segment_count', 0.0):.0f}",
                f"dynamic_bad={metrics.get('dynamic_infeasible_segment_count', 0.0):.0f}",
                f"max_kappa={metrics.get('max_curvature', 0.0):.3f}",
                f"max_heading_jump={metrics.get('max_heading_jump', 0.0):.3f}",
                f"max_yaw_rate={metrics.get('max_yaw_rate', 0.0):.3f}",
            ]
        ),
        fontsize=8,
    )
    ax.plot([], [], color="#d00000", linestyle=":", label="dynamic rejected")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.42), fontsize=7)
    save(fig, "09_constraint_validation.png")

    with (output / "paper_style_region_tsp_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    return output


def _draw_sweeps(ax, sweep_paths: Dict[str, RegionSweepPath], color: str, endpoints: bool) -> None:
    for sweep in sweep_paths.values():
        for coverage_pass in sweep.passes:
            ax.plot(
                [coverage_pass.start_pose.x, coverage_pass.end_pose.x],
                [coverage_pass.start_pose.y, coverage_pass.end_pose.y],
                color=color,
                linewidth=0.8,
                alpha=0.65,
            )
            if endpoints:
                ax.plot(coverage_pass.start_pose.x, coverage_pass.start_pose.y, marker="o", color="#0b5fff", markersize=2.5)
                ax.plot(coverage_pass.end_pose.x, coverage_pass.end_pose.y, marker="s", color="#f2c94c", markersize=2.5)


def _draw_actual_cover_endpoints(
    ax,
    path_plan: MultiAgentPathPlan,
    draw_lines: bool,
    add_agent_labels: bool,
) -> None:
    for agent_id, agent in sorted(path_plan.agents.items()):
        color = _agent_color(agent_id)
        for segment in agent.segments:
            if segment.kind != "cover" or len(segment.waypoints) < 2:
                continue
            start = segment.waypoints[0]
            end = segment.waypoints[-1]
            if draw_lines:
                ax.plot([start.x, end.x], [start.y, end.y], color=color, linewidth=1.0, alpha=0.75)
            ax.plot(
                start.x,
                start.y,
                marker="o",
                color=color,
                markeredgecolor="#111827",
                markeredgewidth=0.35,
                markersize=3.2,
            )
            ax.plot(
                end.x,
                end.y,
                marker="s",
                color=color,
                markeredgecolor="#111827",
                markeredgewidth=0.35,
                markersize=3.2,
            )
        if add_agent_labels:
            ax.plot([], [], color=color, label=f"USV {agent_id}")


def _info_box_outside(ax, text: str, fontsize: float = 8) -> None:
    ax.figure.subplots_adjust(right=0.76)
    ax.text(
        1.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        clip_on=False,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc"},
    )
