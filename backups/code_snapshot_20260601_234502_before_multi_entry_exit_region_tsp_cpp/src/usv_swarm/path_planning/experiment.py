from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..schema import PlannerConfig
from .assignment import balance_region_workload
from .decomposition import decompose_obstacle_aware_area, decompose_rectangular_area
from .graph import build_region_graph
from .obstacles import normalize_obstacle_field
from .patterns import generate_all_region_patterns
from .pipeline import _agent_plans_from_tours, _annotate_agents_validity, _append_residual_backfill, _build_diagnostics
from .residuals import evaluate_tour_coverage_state
from .scheduling import apply_resource_window_schedule
from .tsp import solve_multi_agent_tours
from .types import (
    AlgorithmExperimentTrace,
    MultiAgentPathPlan,
    PaperReference,
    PathPlanningConfig,
    StaticObstacle,
)


def run_planning_algorithm_experiment(
    config: PlannerConfig,
    static_obstacles: Sequence[StaticObstacle] | None,
    output_dir: str | Path,
    path_config: PathPlanningConfig | None = None,
    map_id: str = "",
    paper_references: Sequence[PaperReference] | None = None,
    render: bool = True,
) -> Tuple[MultiAgentPathPlan, AlgorithmExperimentTrace]:
    """Run the real planning algorithm stage-by-stage for experiment plots."""

    started = time.perf_counter()
    path_config = path_config or PathPlanningConfig.from_planner_config(config)
    output_path = Path(output_dir)
    trace = AlgorithmExperimentTrace(map_id=map_id, output_dir=str(output_path))
    static_obstacles = list(static_obstacles or [])
    paper_references = list(paper_references or [])

    _record_stage(
        trace,
        "map_loaded",
        started,
        area_length_x=config.mission.area_length_x,
        area_length_y=config.mission.area_length_y,
        usv_count=config.fleet.num_agents or 0,
        static_obstacle_count=len(static_obstacles),
        footprint_length=config.footprint.length_lf,
        footprint_width=config.footprint.width_wf,
        min_turn_radius=config.fleet.min_turn_radius,
    )

    stage_start = time.perf_counter()
    obstacle_field = normalize_obstacle_field(static_obstacles, config, path_config) if static_obstacles else None
    trace.obstacle_field = obstacle_field
    _record_stage(
        trace,
        "obstacle_inflation",
        stage_start,
        inflated_obstacle_count=len(obstacle_field.inflated_obstacles) if obstacle_field is not None else 0,
        safety_margin=obstacle_field.safety_margin if obstacle_field is not None else 0.0,
        footprint_margin=obstacle_field.footprint_margin if obstacle_field is not None else 0.0,
        inflation=float(obstacle_field.metadata.get("inflation", "0")) if obstacle_field is not None else 0.0,
    )

    stage_start = time.perf_counter()
    regions_before_filter = (
        decompose_obstacle_aware_area(config, path_config, obstacle_field)
        if obstacle_field is not None and obstacle_field.inflated_obstacles
        else decompose_rectangular_area(config, path_config)
    )
    trace.regions_before_filter = list(regions_before_filter)
    _record_stage(
        trace,
        "free_space_decomposition",
        stage_start,
        region_count=len(regions_before_filter),
        total_cell_area=sum(region.area for region in regions_before_filter),
        min_cell_area=min((region.area for region in regions_before_filter), default=0.0),
        max_cell_area=max((region.area for region in regions_before_filter), default=0.0),
    )

    stage_start = time.perf_counter()
    raw_patterns = generate_all_region_patterns(regions_before_filter, config, path_config, obstacle_field=obstacle_field)
    patterns = {region_id: [pattern for pattern in candidates if pattern.feasible] for region_id, candidates in raw_patterns.items()}
    regions = [region for region in regions_before_filter if patterns.get(region.region_id)]
    patterns = {region.region_id: patterns[region.region_id] for region in regions}
    trace.regions = list(regions)
    trace.patterns = dict(patterns)
    _record_stage(
        trace,
        "coverage_pattern_generation",
        stage_start,
        region_count=len(regions),
        candidate_pattern_count=sum(len(items) for items in patterns.values()),
        coverage_pass_count=sum(len(pattern.passes) for items in patterns.values() for pattern in items),
        strip_spacing=config.footprint.width_wf * (1.0 - config.mission.overlap_ratio),
    )

    stage_start = time.perf_counter()
    graph = build_region_graph(regions, patterns, config, obstacle_field=obstacle_field)
    trace.graph = graph
    _record_stage(
        trace,
        "region_graph_building",
        stage_start,
        node_count=len(graph.regions),
        edge_count=len(graph.edge_weights),
        mean_node_weight=_mean(graph.node_weights.values()),
    )

    stage_start = time.perf_counter()
    assignment = balance_region_workload(graph, config)
    trace.assignment = assignment
    _record_stage(
        trace,
        "load_balancing_assignment",
        stage_start,
        agent_count=len(assignment.agent_regions),
        objective=assignment.objective,
        imbalance_ratio=assignment.imbalance_ratio,
        max_load=max(assignment.loads.values(), default=0.0),
        min_load=min(assignment.loads.values(), default=0.0),
    )

    stage_start = time.perf_counter()
    tsp_records: Dict[int, Dict] = {}
    tours = solve_multi_agent_tours(assignment.agent_regions, graph, config, path_config, experiment_records=tsp_records)
    trace.tours = tours
    trace.tsp_records = tsp_records
    _record_stage(
        trace,
        "single_usv_tsp_initial_solution",
        stage_start,
        agent_count=len(tsp_records),
        total_initial_objective=sum(record.get("initial_metrics", {}).get("objective", 0.0) for record in tsp_records.values()),
    )
    _record_stage(
        trace,
        "single_usv_tsp_2opt_optimization",
        stage_start,
        total_improvement_count=sum(len(record.get("two_opt_improvements", [])) for record in tsp_records.values()),
        total_rejected_count=sum(int(record.get("two_opt_rejected_count", 0)) for record in tsp_records.values()),
        total_final_objective=sum(record.get("final_metrics", {}).get("objective", 0.0) for record in tsp_records.values()),
    )
    _record_stage(
        trace,
        "pattern_selection",
        stage_start,
        selected_pattern_count=sum(len(record.get("pattern_selection", [])) for record in tsp_records.values()),
    )
    _record_stage(
        trace,
        "obstacle_aware_connection",
        stage_start,
        astar_corridor_count=sum(record.get("connection_summary", {}).get("by_connector", {}).get("astar_corridor", 0) for record in tsp_records.values()),
        total_segment_count=sum(record.get("connection_summary", {}).get("segment_count", 0) for record in tsp_records.values()),
    )

    stage_start = time.perf_counter()
    residual_backfill_count = _append_residual_backfill(config, tours, path_config, obstacle_field)
    trace.residual_backfill_count = residual_backfill_count
    diagnostics = _build_diagnostics(config, tours, assignment.imbalance_ratio, started, path_config, obstacle_field)
    coverage_state = evaluate_tour_coverage_state(config, list(tours.values()), resolution=path_config.residual_resolution, obstacle_field=obstacle_field)
    trace.coverage_state = coverage_state
    agents = _agent_plans_from_tours(tours, list(paper_references))
    validity_metrics = _annotate_agents_validity(agents, config, obstacle_field)
    conflicts_resolved = apply_resource_window_schedule(agents)
    trace.agents = agents
    trace.mapf_conflicts_resolved = conflicts_resolved
    _record_stage(
        trace,
        "final_tsp_cpp_tour",
        stage_start,
        residual_backfill_count=residual_backfill_count,
        coverage_fraction=coverage_state.coverage_fraction,
        residual_count=len(coverage_state.residual_components),
        mapf_conflicts_resolved=conflicts_resolved,
        invalid_path_length=validity_metrics["invalid_path_length"],
        invalid_segment_count=int(validity_metrics["invalid_segment_count"]),
        out_of_bounds_segment_count=int(validity_metrics["out_of_bounds_segment_count"]),
        obstacle_collision_segment_count=int(validity_metrics["obstacle_collision_segment_count"]),
        kinematic_infeasible_segment_count=int(validity_metrics["kinematic_infeasible_segment_count"]),
    )

    path_plan = MultiAgentPathPlan(
        algorithm_name="paper_fusion_planner",
        agents=agents,
        metadata={
            "status": "paper_fusion_algorithm_experiment",
            "region_count": str(len(regions)),
            "load_imbalance_ratio": f"{assignment.imbalance_ratio:.6f}",
            "coverage_fraction": f"{diagnostics.coverage_fraction:.6f}",
            "residual_count": str(int(diagnostics.metrics.get("residual_count", 0.0))),
            "residual_backfill_count": str(residual_backfill_count),
            "planning_time": f"{diagnostics.planning_time:.6f}",
            "assignment_objective": f"{assignment.objective:.6f}",
            "static_obstacle_count": str(len(static_obstacles)),
            "static_obstacle_aware": str(obstacle_field is not None).lower(),
            "mapf_scheduler": "resource_window_cbs_hook",
            "mapf_conflicts_resolved": str(conflicts_resolved),
            "invalid_path_length": f"{validity_metrics['invalid_path_length']:.6f}",
            "out_of_bounds_segment_count": str(int(validity_metrics["out_of_bounds_segment_count"])),
            "obstacle_collision_segment_count": str(int(validity_metrics["obstacle_collision_segment_count"])),
            "invalid_segment_count": str(int(validity_metrics["invalid_segment_count"])),
            "kinematic_infeasible_segment_count": str(int(validity_metrics["kinematic_infeasible_segment_count"])),
        },
        paper_references=list(paper_references),
    )
    trace.path_plan = path_plan
    if render:
        from .visualization import render_algorithm_experiment

        visual_result = render_algorithm_experiment(config, static_obstacles, path_plan, trace, output_path, dpi=path_config.visual_dpi, gif_fps=path_config.visual_gif_fps)
        path_plan.metadata["algorithm_experiment_dir"] = visual_result["output_dir"]
        path_plan.metadata["algorithm_experiment_report"] = visual_result["report"]
    return path_plan, trace


def _record_stage(trace: AlgorithmExperimentTrace, stage: str, stage_started: float, **metrics) -> None:
    trace.stage_metrics[stage] = {"elapsed_sec": time.perf_counter() - stage_started, **metrics}


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
