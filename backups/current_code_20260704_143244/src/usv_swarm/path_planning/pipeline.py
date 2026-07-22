from __future__ import annotations

import time
from typing import Dict, List

from ..dubins import dubins_shortest_path
from ..geometry import wrap_angle
from ..schema import PlannerConfig, Pose2D
from .assignment import balance_region_workload
from .decomposition import decompose_obstacle_aware_area, decompose_rectangular_area
from .graph import build_region_graph
from .obstacles import normalize_obstacle_field, path_segment_invalid_length, path_segment_invalid_reasons
from .patterns import generate_all_region_patterns
from .residual_planner import append_residual_local_tsp
from .residuals import assign_residual_backfill, evaluate_tour_coverage_state
from .resources import assign_stable_resource_ids, shared_resource_metrics
from .scheduling import apply_resource_window_schedule
from .smoothing import build_cover_segment, build_obstacle_aware_transition_segments
from .tsp import solve_multi_agent_tours
from .types import (
    AgentPathPlan,
    MultiAgentPathPlan,
    PaperReference,
    PathPlanningConfig,
    PathPlanningDiagnostics,
    PathPlanningTrace,
    SingleUsvTourPlan,
    StaticObstacle,
)


def run_paper_fusion_pipeline(
    config: PlannerConfig,
    path_config: PathPlanningConfig | None = None,
    static_obstacles: List[StaticObstacle] | None = None,
    paper_references: List[PaperReference] | None = None,
) -> MultiAgentPathPlan:
    started = time.perf_counter()
    path_config = path_config or PathPlanningConfig.from_planner_config(config)
    trace = (
        PathPlanningTrace(
            enabled=True,
            output_dir=path_config.visual_output_dir,
            map_id=path_config.visual_map_id,
        )
        if path_config.visual_output_dir
        else None
    )
    obstacle_field = normalize_obstacle_field(static_obstacles or [], config, path_config) if static_obstacles else None
    if trace is not None:
        trace.obstacle_field = obstacle_field
    regions = (
        decompose_obstacle_aware_area(config, path_config, obstacle_field)
        if obstacle_field is not None and obstacle_field.inflated_obstacles
        else decompose_rectangular_area(config, path_config)
    )
    if trace is not None:
        trace.regions_before_filter = list(regions)
    raw_patterns = generate_all_region_patterns(regions, config, path_config, obstacle_field=obstacle_field)
    patterns = {region_id: [pattern for pattern in candidates if pattern.feasible] for region_id, candidates in raw_patterns.items()}
    regions = [region for region in regions if patterns.get(region.region_id)]
    patterns = {region.region_id: patterns[region.region_id] for region in regions}
    if trace is not None:
        trace.regions = list(regions)
        trace.patterns = dict(patterns)
    graph = build_region_graph(regions, patterns, config, obstacle_field=obstacle_field)
    if trace is not None:
        trace.graph = graph
    assignment = balance_region_workload(graph, config)
    if trace is not None:
        trace.assignment = assignment
    tours = solve_multi_agent_tours(assignment.agent_regions, graph, config, path_config)
    solver_summary = _solver_summary_from_tours(tours, path_config)
    residual_result = append_residual_local_tsp(config, path_config, obstacle_field, tours)
    residual_backfill_count = residual_result.appended_count
    if trace is not None:
        trace.tours = tours
        trace.residual_backfill_count = residual_backfill_count
    diagnostics = _build_diagnostics(config, tours, assignment.imbalance_ratio, started, path_config, obstacle_field)
    if trace is not None:
        trace.diagnostics = diagnostics
        trace.coverage_state = evaluate_tour_coverage_state(
            config,
            list(tours.values()),
            resolution=path_config.residual_resolution,
            obstacle_field=obstacle_field,
            include_non_cover_segments=path_config.count_transit_coverage,
        )
    agents = _agent_plans_from_tours(tours, list(paper_references or []))
    validity_metrics = _annotate_agents_validity(agents, config, obstacle_field)
    assign_stable_resource_ids(agents, path_config)
    shared_before = shared_resource_metrics(agents, path_config.resource_separation_time)
    conflicts_resolved = apply_resource_window_schedule(agents, separation_time=path_config.resource_separation_time)
    shared_after = shared_resource_metrics(agents, path_config.resource_separation_time)
    if trace is not None:
        trace.agents = agents
        trace.mapf_conflicts_resolved = conflicts_resolved
        trace.metadata.update({key: f"{value:.6f}" for key, value in shared_after.items()})
    path_plan = MultiAgentPathPlan(
        algorithm_name="paper_fusion_planner",
        agents=agents,
        metadata={
            "status": "paper_fusion",
            **solver_summary,
            "region_count": str(len(regions)),
            "load_imbalance_ratio": f"{assignment.imbalance_ratio:.6f}",
            "coverage_fraction": f"{diagnostics.coverage_fraction:.6f}",
            "residual_count": str(int(diagnostics.metrics.get("residual_count", 0.0))),
            "residual_backfill_count": str(residual_backfill_count),
            "residual_local_tsp_enabled": str(path_config.enable_residual_local_tsp).lower(),
            "repeat_path_penalty_total": f"{residual_result.repeat_path_penalty_total:.6f}",
            "planning_time": f"{diagnostics.planning_time:.6f}",
            "assignment_objective": f"{assignment.objective:.6f}",
            "static_obstacle_count": str(len(static_obstacles or [])),
            "static_obstacle_aware": str(obstacle_field is not None).lower(),
            "mapf_scheduler": "resource_window_cbs_hook",
            "mapf_conflicts_resolved": str(conflicts_resolved),
            "mapf_conflicts_resolved_after_residual": str(conflicts_resolved),
            "shared_resource_count": str(int(shared_after["shared_resource_count"])),
            "shared_resource_conflict_count": str(int(shared_before["true_time_conflict_count"])),
            "spatial_overlap_reuse_count": str(int(shared_after["spatial_overlap_reuse_count"])),
            "true_time_conflict_count": str(int(shared_after["true_time_conflict_count"])),
            "invalid_path_length": f"{validity_metrics['invalid_path_length']:.6f}",
            "out_of_bounds_segment_count": str(int(validity_metrics["out_of_bounds_segment_count"])),
            "obstacle_collision_segment_count": str(int(validity_metrics["obstacle_collision_segment_count"])),
            "invalid_segment_count": str(int(validity_metrics["invalid_segment_count"])),
            "kinematic_infeasible_segment_count": str(int(validity_metrics["kinematic_infeasible_segment_count"])),
        },
        paper_references=list(paper_references or []),
    )
    if trace is not None and path_config.visual_output_dir:
        from .visualization import render_path_planning_visual_diagnostics

        visual_result = render_path_planning_visual_diagnostics(
            config=config,
            static_obstacles=static_obstacles or [],
            path_plan=path_plan,
            trace=trace,
            output_dir=path_config.visual_output_dir,
            dpi=path_config.visual_dpi,
            gif_fps=path_config.visual_gif_fps,
        )
        path_plan.metadata["visual_output_dir"] = visual_result["output_dir"]
        path_plan.metadata["visualization_manifest"] = visual_result["manifest"]
    return path_plan


def _solver_summary_from_tours(
    tours: Dict[int, SingleUsvTourPlan],
    path_config: PathPlanningConfig,
) -> Dict[str, str]:
    requested = path_config.tsp_solver
    effective = sorted({tour.diagnostics.get("effective_tsp_solver", "deterministic") for tour in tours.values()})
    statuses = sorted({tour.diagnostics.get("tsp_solver_status", "success") for tour in tours.values()})
    best_objectives = [
        float(tour.diagnostics.get("aco_best_objective", "nan"))
        for tour in tours.values()
        if tour.diagnostics.get("aco_best_objective") not in {None, "None", ""}
    ]
    initial_objectives = [
        float(tour.diagnostics.get("aco_initial_objective", "nan"))
        for tour in tours.values()
        if tour.diagnostics.get("aco_initial_objective") not in {None, "None", ""}
    ]
    return {
        "requested_tsp_solver": requested,
        "effective_tsp_solver": ",".join(effective) if effective else "deterministic",
        "tsp_solver_status": ",".join(statuses) if statuses else "success",
        "aco_best_objective": f"{sum(best_objectives):.6f}" if best_objectives else "",
        "aco_initial_objective": f"{sum(initial_objectives):.6f}" if initial_objectives else "",
    }


def _annotate_agents_validity(
    agents: Dict[int, AgentPathPlan],
    config: PlannerConfig,
    obstacle_field=None,
) -> Dict[str, float]:
    totals = {
        "invalid_path_length": 0.0,
        "invalid_segment_count": 0.0,
        "out_of_bounds_segment_count": 0.0,
        "obstacle_collision_segment_count": 0.0,
        "kinematic_infeasible_segment_count": 0.0,
    }
    for agent in agents.values():
        agent_invalid_length = 0.0
        agent_invalid_count = 0
        agent_out_of_bounds = 0
        agent_obstacle_collision = 0
        agent_kinematic_infeasible = 0
        for segment in agent.segments:
            reasons = path_segment_invalid_reasons(segment, config, obstacle_field)
            invalid_length = path_segment_invalid_length(segment, config, obstacle_field) if reasons else 0.0
            kinematic_infeasible = segment.metadata.get("kinematic_feasible") == "false"
            segment.metadata["collision_free"] = str("obstacle_collision" not in reasons).lower()
            segment.metadata["boundary_safe"] = str("out_of_bounds" not in reasons).lower()
            segment.metadata["invalid_length"] = f"{invalid_length:.6f}"
            if reasons:
                segment.metadata["invalid_reasons"] = ",".join(reasons)
                agent_invalid_count += 1
            else:
                segment.metadata.pop("invalid_reasons", None)
            if "out_of_bounds" in reasons:
                agent_out_of_bounds += 1
            if "obstacle_collision" in reasons:
                agent_obstacle_collision += 1
            if kinematic_infeasible:
                agent_kinematic_infeasible += 1
            agent_invalid_length += invalid_length
        agent.metrics["invalid_path_length"] = agent_invalid_length
        agent.metrics["invalid_segment_count"] = float(agent_invalid_count)
        agent.metrics["out_of_bounds_segment_count"] = float(agent_out_of_bounds)
        agent.metrics["obstacle_collision_segment_count"] = float(agent_obstacle_collision)
        agent.metrics["kinematic_infeasible_segment_count"] = float(agent_kinematic_infeasible)
        totals["invalid_path_length"] += agent_invalid_length
        totals["invalid_segment_count"] += float(agent_invalid_count)
        totals["out_of_bounds_segment_count"] += float(agent_out_of_bounds)
        totals["obstacle_collision_segment_count"] += float(agent_obstacle_collision)
        totals["kinematic_infeasible_segment_count"] += float(agent_kinematic_infeasible)
    return totals


def _agent_plans_from_tours(
    tours: Dict[int, SingleUsvTourPlan],
    paper_references: List[PaperReference],
) -> Dict[int, AgentPathPlan]:
    agents: Dict[int, AgentPathPlan] = {}
    for agent_id, tour in tours.items():
        max_curvature = max((segment.curvature_max for segment in tour.segments), default=0.0)
        agents[agent_id] = AgentPathPlan(
            agent_id=agent_id,
            source_algorithm="paper_fusion_planner",
            segments=list(tour.segments),
            metrics={
                "total_length": tour.total_length,
                "total_turn_angle": tour.total_turn_angle,
                "estimated_time": tour.estimated_time,
                "objective": tour.objective,
                "max_curvature": max_curvature,
                "region_count": float(len(tour.region_order)),
                "segment_count": float(len(tour.segments)),
            },
            paper_references=list(paper_references),
        )
    return agents


def _build_diagnostics(
    config: PlannerConfig,
    tours: Dict[int, SingleUsvTourPlan],
    load_imbalance_ratio: float,
    started: float,
    path_config: PathPlanningConfig,
    obstacle_field=None,
) -> PathPlanningDiagnostics:
    coverage_state = evaluate_tour_coverage_state(
        config,
        list(tours.values()),
        resolution=path_config.residual_resolution,
        obstacle_field=obstacle_field,
        include_non_cover_segments=path_config.count_transit_coverage,
    )
    residuals = coverage_state.residual_components
    total_length = sum(tour.total_length for tour in tours.values())
    max_curvature = max(
        (segment.curvature_max for tour in tours.values() for segment in tour.segments),
        default=0.0,
    )
    return PathPlanningDiagnostics(
        coverage_fraction=coverage_state.coverage_fraction,
        total_length=total_length,
        max_curvature=max_curvature,
        load_imbalance_ratio=load_imbalance_ratio,
        planning_time=time.perf_counter() - started,
        warnings=[] if not residuals else ["coverage_residual_detected"],
        metrics={"residual_count": float(len(residuals))},
    )


def _append_residual_backfill(
    config: PlannerConfig,
    tours: Dict[int, SingleUsvTourPlan],
    path_config: PathPlanningConfig,
    obstacle_field=None,
) -> int:
    appended_total = 0
    for _ in range(max(path_config.residual_backfill_cycles, 1)):
        coverage_state = evaluate_tour_coverage_state(
            config,
            list(tours.values()),
            resolution=path_config.residual_resolution,
            obstacle_field=obstacle_field,
            include_non_cover_segments=path_config.count_transit_coverage,
        )
        result = append_residual_local_tsp(
            config=config,
            path_config=path_config,
            obstacle_field=obstacle_field,
            tours=tours,
            coverage_state=coverage_state,
        )
        if result.appended_count == 0:
            break
        appended_total += result.appended_count
    return appended_total


def _build_residual_pattern_segments(
    agent_id: int,
    region_id: str,
    pattern,
    current_pose: Pose2D,
    current_time: float,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field,
    start_serial: int,
):
    segments = []
    serial = start_serial
    transit_segments = build_obstacle_aware_transition_segments(
        segment_id=f"agent{agent_id}_residual{serial}_to_{region_id}",
        start=current_pose,
        end=pattern.entry_pose,
        start_time=current_time,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        kind="transit",
    )
    if any(segment.metadata.get("kinematic_feasible") == "false" for segment in transit_segments):
        return []
    for sub_idx, segment in enumerate(transit_segments):
        if segment.length <= 1e-9:
            continue
        segment.metadata["resource_id"] = f"residual_transit:{agent_id}:{region_id}:{sub_idx}"
        segment.metadata["region_id"] = region_id
        segments.append(segment)
        current_time = _segment_end_time(segment)
        serial += 1
    for pass_idx, coverage_pass in enumerate(pattern.passes):
        cover = build_cover_segment(
            segment_id=f"agent{agent_id}_residual{serial}_{coverage_pass.pass_id}",
            start=coverage_pass.start_pose,
            end=coverage_pass.end_pose,
            start_time=current_time,
            speed=max(config.fleet.cover_speed, 1e-6),
        )
        cover.metadata["resource_id"] = f"residual_cover:{coverage_pass.region_id}:{coverage_pass.pass_id}"
        cover.metadata["region_id"] = coverage_pass.region_id
        cover.metadata["pass_id"] = coverage_pass.pass_id
        segments.append(cover)
        current_time = _segment_end_time(cover)
        serial += 1
        if pass_idx >= len(pattern.passes) - 1:
            continue
        next_pass = pattern.passes[pass_idx + 1]
        turns = build_obstacle_aware_transition_segments(
            segment_id=f"agent{agent_id}_residual{serial}_{coverage_pass.pass_id}_turn",
            start=coverage_pass.end_pose,
            end=next_pass.start_pose,
            start_time=current_time,
            config=config,
            path_config=path_config,
            obstacle_field=obstacle_field,
            kind="turn",
        )
        if any(segment.metadata.get("kinematic_feasible") == "false" for segment in turns):
            return []
        for sub_idx, turn in enumerate(turns):
            turn.metadata["resource_id"] = f"residual_turn:{coverage_pass.pass_id}->{next_pass.pass_id}:{sub_idx}"
            turn.metadata["region_id"] = coverage_pass.region_id
            segments.append(turn)
            current_time = _segment_end_time(turn)
            serial += 1
    return segments


def _transition_time(start: Pose2D, end: Pose2D, config: PlannerConfig) -> float:
    return dubins_shortest_path(start, end, config.fleet.min_turn_radius).total_length / max(config.fleet.cruise_speed, 1e-6)


def _tour_end_pose(tour: SingleUsvTourPlan, config: PlannerConfig) -> Pose2D:
    for segment in reversed(tour.segments):
        if segment.waypoints:
            waypoint = segment.waypoints[-1]
            return Pose2D(waypoint.x, waypoint.y, waypoint.psi)
    state = config.fleet.initial_states_3dof[tour.agent_id]
    return state.pose()


def _tour_end_time(tour: SingleUsvTourPlan) -> float:
    return max((_segment_end_time(segment) for segment in tour.segments), default=0.0)


def _segment_end_time(segment) -> float:
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


def _segment_heading_variation(segment) -> float:
    headings = [waypoint.psi for waypoint in segment.waypoints]
    return sum(abs(wrap_angle(headings[idx] - headings[idx - 1])) for idx in range(1, len(headings)))
