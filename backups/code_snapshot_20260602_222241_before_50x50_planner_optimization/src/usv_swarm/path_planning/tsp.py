from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..dubins import dubins_shortest_path
from ..geometry import wrap_angle
from ..schema import PlannerConfig, Pose2D
from .aco import AcoTspResult, solve_aco_tsp_cpp, validate_tsp_solver
from .astar import turn_aware_astar
from .obstacles import polyline_collides_with_obstacles
from .resources import score_repeat_overlap
from .smoothing import build_cover_segment, build_obstacle_aware_transition_segments
from .types import PathPlanningConfig, RegionCoveragePattern, RegionGraph, SingleUsvTourPlan


def solve_single_usv_tsp_cpp(
    agent_id: int,
    region_ids: Sequence[str],
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig | None = None,
    experiment_record: Optional[Dict[str, Any]] = None,
) -> SingleUsvTourPlan:
    path_config = path_config or PathPlanningConfig.from_planner_config(config)
    requested_solver = validate_tsp_solver(path_config.tsp_solver)
    start_pose = config.fleet.initial_states_3dof[agent_id].pose()
    if not region_ids:
        return SingleUsvTourPlan(agent_id=agent_id, region_order=[], selected_patterns={})
    fallback_solver_metadata: Dict[str, Any] | None = None
    if requested_solver != "deterministic":
        aco_result = _solve_aco_single_usv(agent_id, region_ids, graph, config, path_config)
        if aco_result.status == "success":
            tour = _tour_from_selected_order(
                agent_id,
                list(aco_result.region_order),
                dict(aco_result.selected_patterns),
                start_pose,
                graph,
                config,
                path_config,
                experiment_record,
                aco_result,
            )
            if len(tour.region_order) == len(set(region_ids)):
                return tour
            fallback_solver_metadata = _solver_metadata(
                aco_result,
                status="failed",
                effective_solver="deterministic_fallback",
                extra={"failure_reason": "aco_selected_tour_failed_segment_assembly"},
            )
        else:
            fallback_solver_metadata = _solver_metadata(aco_result)

    initial_order = _astar_seeded_order(start_pose, sorted(region_ids), graph, config, path_config)
    selected, length, turn_angle, estimated_time, objective = _evaluate_order(initial_order, start_pose, graph, config, path_config)
    if experiment_record is not None:
        experiment_record["agent_id"] = agent_id
        experiment_record["assigned_regions"] = list(region_ids)
        experiment_record["initial_order"] = list(initial_order)
        experiment_record["initial_metrics"] = _metrics_dict(length, turn_angle, estimated_time, objective)
        experiment_record["two_opt_improvements"] = []
        experiment_record["two_opt_rejected_count"] = 0
        experiment_record["three_opt_improvements"] = []
    best_order = list(initial_order)
    best_selected = selected
    best_metrics = (length, turn_angle, estimated_time, objective)
    improved = False

    for _ in range(_bounded_2opt_iterations(path_config, len(best_order))):
        changed = False
        for i in range(0, max(len(best_order) - 2, 0)):
            for j in range(i + 2, len(best_order) + 1):
                candidate_order = best_order[:i] + list(reversed(best_order[i:j])) + best_order[j:]
                candidate_selected, c_length, c_turn, c_time, c_objective = _evaluate_order(
                    candidate_order,
                    start_pose,
                    graph,
                    config,
                    path_config,
                )
                if c_objective + 1e-9 < best_metrics[3]:
                    if experiment_record is not None:
                        experiment_record["two_opt_improvements"].append(
                            {
                                "iteration": len(experiment_record["two_opt_improvements"]) + 1,
                                "i": i,
                                "j": j,
                                "before_order": list(best_order),
                                "after_order": list(candidate_order),
                                "before_objective": best_metrics[3],
                                "after_objective": c_objective,
                                "delta_objective": c_objective - best_metrics[3],
                                "after_metrics": _metrics_dict(c_length, c_turn, c_time, c_objective),
                            }
                        )
                    best_order = candidate_order
                    best_selected = candidate_selected
                    best_metrics = (c_length, c_turn, c_time, c_objective)
                    improved = True
                    changed = True
                    break
                if experiment_record is not None:
                    experiment_record["two_opt_rejected_count"] = int(experiment_record.get("two_opt_rejected_count", 0)) + 1
            if changed:
                break
        if not changed:
            break

    for _ in range(_bounded_3opt_iterations(path_config, len(best_order))):
        changed = False
        for candidate_order in _three_opt_candidates(best_order):
            candidate_selected, c_length, c_turn, c_time, c_objective = _evaluate_order(
                candidate_order,
                start_pose,
                graph,
                config,
                path_config,
            )
            if c_objective + 1e-9 < best_metrics[3]:
                if experiment_record is not None:
                    experiment_record["three_opt_improvements"].append(
                        {
                            "iteration": len(experiment_record["three_opt_improvements"]) + 1,
                            "before_order": list(best_order),
                            "after_order": list(candidate_order),
                            "before_objective": best_metrics[3],
                            "after_objective": c_objective,
                            "delta_objective": c_objective - best_metrics[3],
                            "after_metrics": _metrics_dict(c_length, c_turn, c_time, c_objective),
                        }
                    )
                best_order = candidate_order
                best_selected = candidate_selected
                best_metrics = (c_length, c_turn, c_time, c_objective)
                improved = True
                changed = True
                break
        if not changed:
            break

    segments = _assemble_segments(agent_id, best_order, best_selected, start_pose, graph, config, path_config)
    executed_order = _executed_region_order(segments, best_order)
    executed_selected = {region_id: best_selected[region_id] for region_id in executed_order if region_id in best_selected}
    total_length = sum(segment.length for segment in segments)
    total_turn = sum(_segment_heading_variation(segment) for segment in segments)
    estimated_time = _segments_end_time(segments)
    objective = _objective(total_length, total_turn, estimated_time, path_config)
    if experiment_record is not None:
        experiment_record["planned_final_order"] = list(best_order)
        experiment_record["final_order"] = list(executed_order)
        experiment_record["skipped_regions"] = [region_id for region_id in best_order if region_id not in set(executed_order)]
        experiment_record["final_metrics"] = _metrics_dict(total_length, total_turn, estimated_time, objective)
        experiment_record["pattern_selection"] = _pattern_selection_report(best_order, start_pose, graph, config, path_config)
        experiment_record["connection_summary"] = _connection_summary(segments)
        experiment_record["tsp_solver_metadata"] = fallback_solver_metadata or {
            "requested_tsp_solver": requested_solver,
            "effective_tsp_solver": "deterministic",
            "tsp_solver_status": "success",
            "aco_best_objective": None,
            "aco_initial_objective": None,
            "aco_iteration_count": 0,
            "aco_convergence_trace": [],
            "aco_accepted_3opt_count": 0,
        }
    return SingleUsvTourPlan(
        agent_id=agent_id,
        region_order=executed_order,
        selected_patterns=executed_selected,
        segments=segments,
        total_length=total_length,
        total_turn_angle=total_turn,
        estimated_time=estimated_time,
        objective=objective,
        improved=improved,
        diagnostics={
            "initial_order": ",".join(initial_order),
            "planned_region_order": ",".join(best_order),
            "skipped_regions": ",".join(region_id for region_id in best_order if region_id not in set(executed_order)),
            "region_count": str(len(executed_order)),
            "ordering_source": "turn_aware_astar",
            "two_opt_iterations": str(_bounded_2opt_iterations(path_config, len(best_order))),
            "three_opt_iterations": str(_bounded_3opt_iterations(path_config, len(best_order))),
            "requested_tsp_solver": requested_solver,
            "effective_tsp_solver": str((fallback_solver_metadata or {}).get("effective_tsp_solver", "deterministic")),
            "tsp_solver_status": str((fallback_solver_metadata or {}).get("tsp_solver_status", "success")),
        },
    )


def _executed_region_order(segments: Sequence, planned_order: Sequence[str]) -> List[str]:
    seen = set()
    executed = []
    for segment in segments:
        region_id = segment.metadata.get("region_id") or segment.metadata.get("to_region")
        if not region_id or region_id in seen or region_id not in planned_order:
            continue
        seen.add(region_id)
        executed.append(region_id)
    return executed


def solve_multi_agent_tours(
    assignment: Dict[int, List[str]],
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig | None = None,
    experiment_records: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[int, SingleUsvTourPlan]:
    tours: Dict[int, SingleUsvTourPlan] = {}
    for agent_id, region_ids in assignment.items():
        record = None
        if experiment_records is not None:
            record = {}
            experiment_records[agent_id] = record
        tours[agent_id] = solve_single_usv_tsp_cpp(agent_id, region_ids, graph, config, path_config, experiment_record=record)
    return tours


def _solve_aco_single_usv(
    agent_id: int,
    region_ids: Sequence[str],
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> AcoTspResult:
    start_pose = config.fleet.initial_states_3dof[agent_id].pose()

    def edge_cost(previous: RegionCoveragePattern | None, candidate: RegionCoveragePattern) -> float:
        current_pose = start_pose if previous is None else previous.exit_pose
        transition = dubins_shortest_path(current_pose, candidate.entry_pose, config.fleet.min_turn_radius)
        transition_turn = _dubins_turn_angle(transition.segment_lengths, transition.modes, config.fleet.min_turn_radius)
        transition_time = transition.total_length / max(config.fleet.cruise_speed, 1e-6)
        collision_penalty = _transition_collision_penalty(current_pose, candidate.entry_pose, graph)
        infeasible_penalty = 0.0 if candidate.feasible else 1e6
        return (
            path_config.length_weight * (transition.total_length + candidate.total_length)
            + path_config.turn_angle_weight * (transition_turn + candidate.turn_angle)
            + path_config.time_weight * (transition_time + candidate.estimated_time)
            + _pattern_internal_repeat_penalty(candidate, path_config)
            + collision_penalty
            + infeasible_penalty
        )

    return solve_aco_tsp_cpp(
        region_ids=region_ids,
        patterns=graph.patterns,
        start_pose=start_pose,
        path_config=path_config,
        edge_cost_fn=edge_cost,
        solver=path_config.tsp_solver,
    )


def _tour_from_selected_order(
    agent_id: int,
    best_order: Sequence[str],
    best_selected: Dict[str, RegionCoveragePattern],
    start_pose: Pose2D,
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    experiment_record: Optional[Dict[str, Any]],
    aco_result: AcoTspResult,
) -> SingleUsvTourPlan:
    segments = _assemble_segments(agent_id, best_order, best_selected, start_pose, graph, config, path_config)
    executed_order = _executed_region_order(segments, best_order)
    executed_selected = {region_id: best_selected[region_id] for region_id in executed_order if region_id in best_selected}
    total_length = sum(segment.length for segment in segments)
    total_turn = sum(_segment_heading_variation(segment) for segment in segments)
    estimated_time = _segments_end_time(segments)
    objective = _objective(total_length, total_turn, estimated_time, path_config)
    solver_metadata = _solver_metadata(aco_result)
    if experiment_record is not None:
        experiment_record["agent_id"] = agent_id
        experiment_record["assigned_regions"] = list(best_order)
        experiment_record["initial_order"] = list(aco_result.metadata.get("initial_order", best_order))
        experiment_record["initial_metrics"] = _metrics_dict(
            float(aco_result.initial_objective),
            0.0,
            0.0,
            float(aco_result.initial_objective),
        )
        experiment_record["two_opt_improvements"] = []
        experiment_record["two_opt_rejected_count"] = 0
        experiment_record["three_opt_improvements"] = []
        experiment_record["planned_final_order"] = list(best_order)
        experiment_record["final_order"] = list(executed_order)
        experiment_record["skipped_regions"] = [region_id for region_id in best_order if region_id not in set(executed_order)]
        experiment_record["final_metrics"] = _metrics_dict(total_length, total_turn, estimated_time, objective)
        experiment_record["pattern_selection"] = _pattern_selection_report(best_order, start_pose, graph, config, path_config)
        experiment_record["connection_summary"] = _connection_summary(segments)
        experiment_record["tsp_solver_metadata"] = solver_metadata
    return SingleUsvTourPlan(
        agent_id=agent_id,
        region_order=executed_order,
        selected_patterns=executed_selected,
        segments=segments,
        total_length=total_length,
        total_turn_angle=total_turn,
        estimated_time=estimated_time,
        objective=objective,
        improved=aco_result.objective + 1e-9 < aco_result.initial_objective,
        diagnostics={
            "initial_order": ",".join(str(item) for item in aco_result.metadata.get("initial_order", best_order)),
            "planned_region_order": ",".join(best_order),
            "skipped_regions": ",".join(region_id for region_id in best_order if region_id not in set(executed_order)),
            "region_count": str(len(executed_order)),
            "ordering_source": str(aco_result.effective_solver),
            "requested_tsp_solver": str(solver_metadata["requested_tsp_solver"]),
            "effective_tsp_solver": str(solver_metadata["effective_tsp_solver"]),
            "tsp_solver_status": str(solver_metadata["tsp_solver_status"]),
            "aco_best_objective": str(solver_metadata["aco_best_objective"]),
            "aco_initial_objective": str(solver_metadata["aco_initial_objective"]),
            "aco_iteration_count": str(solver_metadata["aco_iteration_count"]),
            "aco_accepted_3opt_count": str(solver_metadata["aco_accepted_3opt_count"]),
        },
    )


def _solver_metadata(
    result: AcoTspResult,
    status: str | None = None,
    effective_solver: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
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


def _astar_seeded_order(
    start_pose: Pose2D,
    region_ids: Sequence[str],
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> List[str]:
    remaining = set(region_ids)
    order: List[str] = []
    current_pose = start_pose
    current_region: str | None = None
    while remaining:
        if current_region is None:
            best_region = min(
                sorted(remaining),
                key=lambda region_id: _best_entry_transition_cost(current_pose, graph.patterns.get(region_id, []), config),
            )
            astar_path = [best_region]
        else:
            allowed = set(remaining)
            allowed.add(current_region)
            candidates = []
            for region_id in sorted(remaining):
                result = turn_aware_astar(
                    graph,
                    current_region,
                    region_id,
                    path_config=path_config,
                    allowed_nodes=allowed,
                )
                candidates.append((result.cost if result.found else float("inf"), region_id, result.path))
            best_cost, best_region, best_path = min(candidates, key=lambda item: (item[0], item[1]))
            astar_path = best_path[1:] if best_path and best_cost < float("inf") else [best_region]

        for path_region in astar_path:
            if path_region not in remaining:
                continue
            order.append(path_region)
            best_pattern = min(
                graph.patterns.get(path_region, []),
                key=lambda pattern: _transition_length(current_pose, pattern.entry_pose, config)
                + pattern.estimated_time
                + _pattern_internal_repeat_penalty(pattern, path_config),
            )
            current_pose = best_pattern.exit_pose
            current_region = path_region
            remaining.remove(path_region)
    return order


def _three_opt_candidates(order: Sequence[str]) -> List[List[str]]:
    if len(order) < 4:
        return []
    candidates: List[List[str]] = []
    n = len(order)
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = list(order[:i])
                b = list(order[i:j])
                c = list(order[j:k])
                d = list(order[k:])
                candidates.append(a + list(reversed(b)) + c + d)
                candidates.append(a + b + list(reversed(c)) + d)
                candidates.append(a + list(reversed(c)) + list(reversed(b)) + d)
    return candidates


def _bounded_2opt_iterations(path_config: PathPlanningConfig, region_count: int) -> int:
    requested = max(path_config.tsp_2opt_iterations, 0)
    if region_count > 30:
        return min(requested, 1)
    if region_count > 18:
        return min(requested, 3)
    return requested


def _bounded_3opt_iterations(path_config: PathPlanningConfig, region_count: int) -> int:
    requested = max(path_config.tsp_3opt_iterations, 0)
    if region_count > 12:
        return 0
    return requested


def _evaluate_order(
    order: Sequence[str],
    start_pose: Pose2D,
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> Tuple[Dict[str, RegionCoveragePattern], float, float, float, float]:
    selected: Dict[str, RegionCoveragePattern] = {}
    current_pose = start_pose
    total_length = 0.0
    total_turn = 0.0
    total_time = 0.0
    for idx, region_id in enumerate(order):
        next_region = order[idx + 1] if idx + 1 < len(order) else None
        pattern = _select_pattern(region_id, current_pose, next_region, graph, config, path_config)
        selected[region_id] = pattern
        transition = dubins_shortest_path(current_pose, pattern.entry_pose, config.fleet.min_turn_radius)
        transition_turn = _dubins_turn_angle(transition.segment_lengths, transition.modes, config.fleet.min_turn_radius)
        total_length += transition.total_length + pattern.total_length
        total_turn += transition_turn + pattern.turn_angle
        total_time += transition.total_length / max(config.fleet.cruise_speed, 1e-6) + pattern.estimated_time
        current_pose = pattern.exit_pose
    objective = _objective(total_length, total_turn, total_time, path_config)
    return selected, total_length, total_turn, total_time, objective


def _metrics_dict(length: float, turn_angle: float, estimated_time: float, objective: float) -> Dict[str, float]:
    return {
        "length": float(length),
        "turn_angle": float(turn_angle),
        "estimated_time": float(estimated_time),
        "objective": float(objective),
    }


def _pattern_selection_report(
    order: Sequence[str],
    start_pose: Pose2D,
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> List[Dict[str, Any]]:
    report: List[Dict[str, Any]] = []
    current_pose = start_pose
    for idx, region_id in enumerate(order):
        next_region = order[idx + 1] if idx + 1 < len(order) else None
        candidates = graph.patterns.get(region_id, [])
        candidate_reports = []
        for pattern in candidates:
            inside = pattern.estimated_time
            connect = _transition_length(current_pose, pattern.entry_pose, config) / max(config.fleet.cruise_speed, 1e-6)
            collision = _transition_collision_penalty(current_pose, pattern.entry_pose, graph)
            lookahead = _lookahead_cost(pattern, next_region, graph, config)
            repeat_penalty = _pattern_internal_repeat_penalty(pattern, path_config)
            infeasible = 0.0 if pattern.feasible else 1e6
            total = inside + connect + collision + lookahead + repeat_penalty + infeasible
            candidate_reports.append(
                {
                    "pattern_id": pattern.pattern_id,
                    "scan_axis": pattern.scan_axis,
                    "inside_cost": float(inside),
                    "connect_cost": float(connect),
                    "collision_penalty": float(collision),
                    "lookahead_cost": float(lookahead),
                    "repeat_penalty": float(repeat_penalty),
                    "infeasible_penalty": float(infeasible),
                    "total_cost": float(total),
                    "pass_count": len(pattern.passes),
                    "entry": [pattern.entry_pose.x, pattern.entry_pose.y, pattern.entry_pose.psi],
                    "exit": [pattern.exit_pose.x, pattern.exit_pose.y, pattern.exit_pose.psi],
                }
            )
        if candidate_reports:
            selected = min(candidate_reports, key=lambda item: item["total_cost"])
            current_pose = next(pattern.exit_pose for pattern in candidates if pattern.pattern_id == selected["pattern_id"])
        else:
            selected = {}
        report.append({"region_id": region_id, "sequence_index": idx, "selected": selected, "candidates": candidate_reports})
    return report


def _connection_summary(segments: Sequence) -> Dict[str, Any]:
    by_kind: Dict[str, int] = {}
    by_connector: Dict[str, int] = {}
    for segment in segments:
        by_kind[segment.kind] = by_kind.get(segment.kind, 0) + 1
        connector = segment.metadata.get("connector", segment.path_source or "unknown")
        by_connector[connector] = by_connector.get(connector, 0) + 1
    return {
        "segment_count": len(segments),
        "by_kind": by_kind,
        "by_connector": by_connector,
    }


def _select_pattern(
    region_id: str,
    current_pose: Pose2D,
    next_region: str | None,
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> RegionCoveragePattern:
    candidates = graph.patterns.get(region_id, [])
    if not candidates:
        raise ValueError(f"region {region_id} has no coverage patterns")
    return min(
        candidates,
        key=lambda pattern: (
            pattern.estimated_time
            + _transition_length(current_pose, pattern.entry_pose, config) / max(config.fleet.cruise_speed, 1e-6)
            + _transition_collision_penalty(current_pose, pattern.entry_pose, graph)
            + _lookahead_cost(pattern, next_region, graph, config)
            + _pattern_internal_repeat_penalty(pattern, path_config)
            + (0.0 if pattern.feasible else 1e6)
        ),
    )


def _assemble_segments(
    agent_id: int,
    order: Sequence[str],
    selected: Dict[str, RegionCoveragePattern],
    start_pose: Pose2D,
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> List:
    segments = []
    current_pose = start_pose
    current_time = 0.0
    serial = 0
    remaining = list(order)
    while remaining:
        feasible_candidates = []
        for idx, region_id in enumerate(remaining):
            pattern = selected[region_id]
            region_segments = _build_region_segments_atomic(
                agent_id=agent_id,
                region_id=region_id,
                pattern=pattern,
                current_pose=current_pose,
                current_time=current_time,
                serial=serial,
                config=config,
                path_config=path_config,
                obstacle_field=graph.obstacle_field,
            )
            if region_segments is None:
                continue
            repeat_weight = path_config.main_repeat_path_penalty_weight if path_config.enable_main_repeat_path_penalty else 0.0
            repeat_score = score_repeat_overlap(
                [segment for segment in region_segments if segment.kind != "cover"],
                segments,
                path_config,
                penalty_weight=repeat_weight,
                annotate=True,
            )
            feasible_candidates.append((repeat_score.penalty, idx, region_id, pattern, region_segments))
        if not feasible_candidates:
            break
        _, accepted_index, accepted_region_id, accepted_pattern, accepted_segments = min(
            feasible_candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        segments.extend(accepted_segments)
        serial += len(accepted_segments)
        current_time = _segment_end_time(accepted_segments[-1])
        current_pose = accepted_pattern.exit_pose
        remaining.pop(accepted_index)
    return segments


def _build_region_segments_atomic(
    agent_id: int,
    region_id: str,
    pattern: RegionCoveragePattern,
    current_pose: Pose2D,
    current_time: float,
    serial: int,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field,
) -> List | None:
    region_segments = []
    next_serial = serial
    transit_segments = build_obstacle_aware_transition_segments(
        segment_id=f"agent{agent_id}_segment{next_serial}_to_{region_id}",
        start=current_pose,
        end=pattern.entry_pose,
        start_time=current_time,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        kind="transit",
    )
    if not _segments_motion_feasible(transit_segments, strict=obstacle_field is not None):
        return None
    for sub_idx, transit in enumerate(transit_segments):
        if transit.length <= 1e-9:
            continue
        transit.metadata["resource_id"] = _transition_resource_id(agent_id, region_id, sub_idx, transit)
        transit.metadata["to_region"] = region_id
        region_segments.append(transit)
        current_time = _segment_end_time(transit)
        next_serial += 1

    for pass_idx, coverage_pass in enumerate(pattern.passes):
        cover = build_cover_segment(
            segment_id=f"agent{agent_id}_segment{next_serial}_{coverage_pass.pass_id}",
            start=coverage_pass.start_pose,
            end=coverage_pass.end_pose,
            start_time=current_time,
            speed=max(config.fleet.cover_speed, 1e-6),
        )
        cover.metadata["resource_id"] = f"cover:{coverage_pass.region_id}:{coverage_pass.pass_id}"
        cover.metadata["region_id"] = coverage_pass.region_id
        cover.metadata["pass_id"] = coverage_pass.pass_id
        region_segments.append(cover)
        current_time = _segment_end_time(cover)
        next_serial += 1
        if pass_idx >= len(pattern.passes) - 1:
            continue
        next_pass = pattern.passes[pass_idx + 1]
        turn_segments = build_obstacle_aware_transition_segments(
            segment_id=f"agent{agent_id}_segment{next_serial}_{coverage_pass.pass_id}_turn",
            start=coverage_pass.end_pose,
            end=next_pass.start_pose,
            start_time=current_time,
            config=config,
            path_config=path_config,
            obstacle_field=obstacle_field,
            kind="turn",
        )
        if not _segments_motion_feasible(turn_segments, strict=obstacle_field is not None):
            return None
        for sub_idx, turn in enumerate(turn_segments):
            turn.metadata["resource_id"] = _turn_resource_id(coverage_pass.pass_id, next_pass.pass_id, sub_idx, turn)
            turn.metadata["region_id"] = coverage_pass.region_id
            region_segments.append(turn)
            current_time = _segment_end_time(turn)
            next_serial += 1
    return region_segments


def _legacy_unused_linear_assembly_marker() -> None:
    return None


def _removed_linear_assembly_placeholder():
    return None


def _segments_motion_feasible(segments: Sequence, strict: bool = True) -> bool:
    return all(
        (not strict or segment.metadata.get("kinematic_feasible", "true") != "false")
        and not segment.metadata.get("invalid_reasons")
        for segment in segments
    )


def _lookahead_cost(
    pattern: RegionCoveragePattern,
    next_region: str | None,
    graph: RegionGraph,
    config: PlannerConfig,
) -> float:
    if next_region is None:
        return 0.0
    candidates = graph.patterns.get(next_region, [])
    if not candidates:
        return 0.0
    return min(_transition_length(pattern.exit_pose, candidate.entry_pose, config) for candidate in candidates) / max(
        config.fleet.cruise_speed,
        1e-6,
    )


def _best_entry_transition_cost(
    current_pose: Pose2D,
    candidates: Sequence[RegionCoveragePattern],
    config: PlannerConfig,
) -> float:
    if not candidates:
        return float("inf")
    return min(
        _transition_length(current_pose, pattern.entry_pose, config) / max(config.fleet.cruise_speed, 1e-6)
        + pattern.estimated_time
        for pattern in candidates
    )


def _transition_length(start: Pose2D, end: Pose2D, config: PlannerConfig) -> float:
    return dubins_shortest_path(start, end, config.fleet.min_turn_radius).total_length


def _transition_collision_penalty(start: Pose2D, end: Pose2D, graph: RegionGraph) -> float:
    if graph.obstacle_field is None:
        return 0.0
    points = [(start.x, start.y), (end.x, end.y)]
    return 1e6 if polyline_collides_with_obstacles(points, graph.obstacle_field, inflated=True) else 0.0


def _transition_resource_id(agent_id: int, region_id: str, sub_idx: int, segment) -> str:
    if segment.metadata.get("connector") == "astar_corridor" and len(segment.waypoints) >= 2:
        start = segment.waypoints[0]
        end = segment.waypoints[-1]
        key = sorted([(round(start.x, 1), round(start.y, 1)), (round(end.x, 1), round(end.y, 1))])
        return f"corridor:{key[0]}:{key[1]}"
    return f"transit:{agent_id}:{region_id}:{sub_idx}"


def _turn_resource_id(pass_id: str, next_pass_id: str, sub_idx: int, segment) -> str:
    if segment.metadata.get("connector") == "astar_corridor" and len(segment.waypoints) >= 2:
        start = segment.waypoints[0]
        end = segment.waypoints[-1]
        key = sorted([(round(start.x, 1), round(start.y, 1)), (round(end.x, 1), round(end.y, 1))])
        return f"corridor:{key[0]}:{key[1]}"
    return f"turn:{pass_id}->{next_pass_id}:{sub_idx}"


def _dubins_turn_angle(segment_lengths: Tuple[float, float, float], modes: Tuple[str, str, str], turn_radius: float) -> float:
    return sum(abs(length / max(turn_radius, 1e-6)) for length, mode in zip(segment_lengths, modes) if mode in {"L", "R"})


def _segment_heading_variation(segment) -> float:
    headings = [waypoint.psi for waypoint in segment.waypoints]
    return sum(abs(wrap_angle(headings[idx] - headings[idx - 1])) for idx in range(1, len(headings)))


def _segment_end_time(segment) -> float:
    if not segment.waypoints or segment.waypoints[-1].time is None:
        return 0.0
    return float(segment.waypoints[-1].time)


def _segments_end_time(segments: Iterable) -> float:
    return max((_segment_end_time(segment) for segment in segments), default=0.0)


def _objective(total_length: float, total_turn: float, total_time: float, path_config: PathPlanningConfig) -> float:
    return (
        path_config.length_weight * total_length
        + path_config.turn_angle_weight * total_turn
        + path_config.time_weight * total_time
    )


def _pattern_internal_repeat_penalty(pattern: RegionCoveragePattern, path_config: PathPlanningConfig) -> float:
    if not path_config.enable_main_repeat_path_penalty:
        return 0.0
    try:
        return float(pattern.metadata.get("internal_repeat_penalty", 0.0))
    except (TypeError, ValueError):
        return 0.0
