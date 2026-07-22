from __future__ import annotations

import json
import math
import time
import copy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..dubins import dubins_shortest_path
from ..geometry import wrap_angle
from ..schema import PlannerConfig, Pose2D
from .aco import AcoTspResult, solve_aco_tsp_cpp, validate_tsp_solver
from .assignment import apply_lightweight_load_swap, balance_region_workload
from .decomposition import (
    build_composite_free_space_regions,
    build_free_space_cells,
    build_large_convex_free_space_regions,
    decompose_obstacle_aware_area,
    decompose_rectangular_area,
)
from .dynamics_validation import dynamic_edge_cost, validate_transition_dynamics, validate_transition_sequence
from .graph import build_region_graph
from .obstacles import (
    normalize_obstacle_field,
    path_segment_invalid_length,
    path_segment_invalid_reasons,
    polygon_collides_with_obstacles,
    sampled_segment_footprint_collides,
)
from .patterns import generate_region_patterns
from .performance import build_performance_summary
from .residual_planner import append_residual_local_tsp
from .residuals import evaluate_tour_coverage_state
from .resources import (
    assign_stable_resource_ids,
    build_coverage_ownership_map,
    collect_resource_windows,
    CrossAgentOverlapScore,
    cross_agent_overlap_metrics,
    mark_cross_agent_unavoidable,
    RepeatOverlapScore,
    score_cross_agent_ownership_overlap,
    score_repeat_overlap,
    shared_resource_metrics,
)
from .scheduling import apply_resource_window_schedule
from .smoothing import build_cover_segment, build_obstacle_aware_transition_segments, build_transition_segment
from .types import (
    AgentPathPlan,
    CoveragePass,
    CoverageOwnershipMap,
    MultiAgentPathPlan,
    ObstacleField,
    OpenSweepBreak,
    OpenSweepChain,
    PathPlanningConfig,
    PathSegmentSpec,
    DecomposedRegion,
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
    path_config = _apply_performance_profile(path_config)
    path_config = _apply_large_map_defaults(path_config, config)
    stage_timing_sec: Dict[str, float] = {}
    stage_extra: Dict[str, Dict[str, object]] = {}
    sweep_build_stats: Dict[str, object] = {}
    region_repair_count = 0

    def finish_stage(name: str, stage_started: float, **extra: object) -> None:
        elapsed = time.perf_counter() - stage_started
        stage_timing_sec[name] = elapsed
        if extra:
            stage_extra[name] = dict(extra)
        if path_config.monitor_stages:
            payload = {
                "stage": name,
                "dt_sec": round(elapsed, 3),
                "elapsed_sec": round(time.perf_counter() - started, 3),
                **extra,
            }
            print(json.dumps(payload, ensure_ascii=False), flush=True)

    def emit_region_progress(name: str, stage_started: float, **extra: object) -> None:
        if not path_config.monitor_stages:
            return
        now = time.perf_counter()
        payload = {
            "stage": name,
            "stage_elapsed_sec": round(now - stage_started, 3),
            "elapsed_sec": round(now - started, 3),
            **extra,
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    static_obstacles = list(static_obstacles or [])
    stage_started = time.perf_counter()
    obstacle_field = normalize_obstacle_field(static_obstacles, config, path_config) if static_obstacles else None
    finish_stage(
        "obstacle_normalization",
        stage_started,
        static_obstacle_count=len(static_obstacles),
        inflated_obstacle_count=len(obstacle_field.inflated_obstacles) if obstacle_field is not None else 0,
    )

    stage_started = time.perf_counter()
    large_convex_mode_enabled = False
    composite_mode_enabled = bool(
        obstacle_field is not None
        and obstacle_field.inflated_obstacles
        and path_config.enable_composite_free_space_regions
    )
    raw_free_cells = []
    if obstacle_field is not None and obstacle_field.inflated_obstacles and path_config.enable_large_convex_region_decomposition:
        regions = build_large_convex_free_space_regions(config, path_config, obstacle_field)
        large_convex_mode_enabled = bool(
            regions
            and any(region.metadata.get("convex_region_decomposition") == "true" for region in regions)
        )
        if not large_convex_mode_enabled:
            regions = []
    if large_convex_mode_enabled:
        composite_mode_enabled = False
    elif composite_mode_enabled:
        raw_free_cells = build_free_space_cells(config, path_config, obstacle_field)
        regions = build_composite_free_space_regions(raw_free_cells, config, path_config, obstacle_field)
        if not regions:
            regions = decompose_obstacle_aware_area(config, path_config, obstacle_field)
            composite_mode_enabled = False
    else:
        regions = (
            decompose_obstacle_aware_area(config, path_config, obstacle_field)
            if obstacle_field is not None and obstacle_field.inflated_obstacles
            else decompose_rectangular_area(config, path_config)
        )
    raw_free_regions = list(raw_free_cells) if raw_free_cells else list(regions)
    base_region_count = len(raw_free_cells) if raw_free_cells else len(regions)
    finish_stage(
        "free_space_decomposition",
        stage_started,
        base_region_count=base_region_count,
        large_convex_mode_enabled=large_convex_mode_enabled,
        large_convex_region_count=sum(
            1 for region in regions if region.metadata.get("convex_region_decomposition") == "true"
        ),
        composite_mode_enabled=composite_mode_enabled,
        composite_region_count=len(regions) if composite_mode_enabled else 0,
    )

    stage_started = time.perf_counter()
    coarsened_regions = None
    merge_diagnostics: Dict[str, int] = {}
    performance_merge_fallback = False
    if composite_mode_enabled:
        coarsened_regions = list(regions)
    elif large_convex_mode_enabled:
        coarsened_regions = list(regions)
    elif obstacle_field is not None and obstacle_field.inflated_obstacles:
        coarsened_regions = _coarsen_paper_style_regions(
            regions,
            config,
            obstacle_field=obstacle_field,
            diagnostics=merge_diagnostics,
        )
        regions = _merge_performance_regions(
            coarsened_regions,
            config,
            path_config,
            obstacle_field=obstacle_field,
            diagnostics=merge_diagnostics,
        )
    finish_stage(
        "region_coarsen_merge",
        stage_started,
        coarsened_region_count=len(coarsened_regions) if coarsened_regions is not None else len(regions),
        merged_region_count=len(regions),
        composite_region_count=len(regions) if composite_mode_enabled else 0,
        composite_member_cell_count=sum(len(getattr(region, "member_cells", []) or []) for region in regions),
        merge_rejected_count=sum(merge_diagnostics.values()),
        merge_rejected_by_reason=dict(merge_diagnostics),
    )

    stage_started = time.perf_counter()
    raw_patterns = _generate_paper_style_patterns(
        regions,
        config,
        path_config,
        obstacle_field,
        progress_callback=lambda **extra: emit_region_progress(
            "coverage_pattern_generation_region",
            stage_started,
            **extra,
        ),
    )
    finish_stage(
        "coverage_pattern_generation",
        stage_started,
        region_count=len(regions),
        raw_pattern_count=sum(len(items) for items in raw_patterns.values()),
    )

    stage_started = time.perf_counter()
    sweep_paths, feasible_patterns, infeasible_regions, sweep_segment_templates = _build_region_sweep_paths(
        raw_patterns,
        config,
        path_config,
        obstacle_field,
        stats=sweep_build_stats,
        progress_callback=lambda **extra: emit_region_progress(
            "build_sweep_paths_region",
            stage_started,
            **extra,
        ),
    )
    finish_stage(
        "build_sweep_paths",
        stage_started,
        feasible_region_count=len(feasible_patterns),
        infeasible_region_count=len(infeasible_regions),
        prefiltered_pattern_count=int(sweep_build_stats.get("prefiltered_pattern_count", 0) or 0),
        uturn_cache_hit_count=int(sweep_build_stats.get("uturn_cache_hit_count", 0) or 0),
        uturn_cache_miss_count=int(sweep_build_stats.get("uturn_cache_miss_count", 0) or 0),
        open_chain_region_count=int(sweep_build_stats.get("open_chain_region_count", 0) or 0),
        open_chain_count=int(sweep_build_stats.get("open_chain_count", 0) or 0),
        open_chain_break_count=int(sweep_build_stats.get("open_chain_break_count", 0) or 0),
        open_chain_connected_count=int(sweep_build_stats.get("open_chain_connected_count", 0) or 0),
    )
    composite_split_count = 0
    large_convex_split_count = 0
    if path_config.enable_infeasible_uturn_region_repair and large_convex_mode_enabled and infeasible_regions:
        stage_started = time.perf_counter()
        repair_depth_count = 0
        max_repair_depth = max(1, int(path_config.max_large_map_region_repair_depth))
        for _ in range(max_repair_depth):
            split_regions, split_count = _split_infeasible_large_convex_regions(regions, infeasible_regions, config)
            if not split_count:
                break
            large_convex_split_count += split_count
            repair_depth_count += 1
            regions = split_regions
            raw_patterns = _generate_paper_style_patterns(
                regions,
                config,
                path_config,
                obstacle_field,
                progress_callback=lambda **extra: emit_region_progress(
                    "coverage_pattern_generation_region",
                    stage_started,
                    **extra,
                ),
            )
            sweep_paths, feasible_patterns, infeasible_regions, sweep_segment_templates = _build_region_sweep_paths(
                raw_patterns,
                config,
                path_config,
                obstacle_field,
                stats=sweep_build_stats,
                progress_callback=lambda **extra: emit_region_progress(
                    "build_sweep_paths_region",
                    stage_started,
                    **extra,
                ),
            )
            if not infeasible_regions:
                break
        region_repair_count += large_convex_split_count
        finish_stage(
            "large_convex_region_split_repair",
            stage_started,
            large_convex_split_count=large_convex_split_count,
            large_convex_split_repair_depth=repair_depth_count,
            feasible_region_count=len(feasible_patterns),
            infeasible_region_count=len(infeasible_regions),
        )
    if path_config.enable_infeasible_uturn_region_repair and composite_mode_enabled and infeasible_regions:
        stage_started = time.perf_counter()
        split_regions, composite_split_count = _split_infeasible_composite_regions(regions, infeasible_regions)
        if composite_split_count:
            regions = split_regions
            raw_patterns = _generate_paper_style_patterns(
                regions,
                config,
                path_config,
                obstacle_field,
                progress_callback=lambda **extra: emit_region_progress(
                    "coverage_pattern_generation_region",
                    stage_started,
                    **extra,
                ),
            )
            sweep_paths, feasible_patterns, infeasible_regions, sweep_segment_templates = _build_region_sweep_paths(
                raw_patterns,
                config,
                path_config,
                obstacle_field,
                stats=sweep_build_stats,
                progress_callback=lambda **extra: emit_region_progress(
                    "build_sweep_paths_region",
                    stage_started,
                    **extra,
                ),
            )
            region_repair_count += composite_split_count
        finish_stage(
            "composite_region_split_repair",
            stage_started,
            composite_split_count=composite_split_count,
            feasible_region_count=len(feasible_patterns),
            infeasible_region_count=len(infeasible_regions),
        )
    if (
        path_config.enable_infeasible_uturn_region_repair
        and infeasible_regions
        and coarsened_regions is not None
        and len(regions) < len(coarsened_regions)
    ):
        stage_started = time.perf_counter()
        repaired_regions = _repair_infeasible_merged_regions(regions, coarsened_regions, infeasible_regions)
        region_repair_count += max(0, len(repaired_regions) - len(regions))
        performance_merge_fallback = len(repaired_regions) != len(regions)
        regions = repaired_regions
        raw_patterns = _generate_paper_style_patterns(
            regions,
            config,
            path_config,
            obstacle_field,
            progress_callback=lambda **extra: emit_region_progress(
                "coverage_pattern_generation_region",
                stage_started,
                **extra,
            ),
        )
        sweep_paths, feasible_patterns, infeasible_regions, sweep_segment_templates = _build_region_sweep_paths(
            raw_patterns,
            config,
            path_config,
            obstacle_field,
            stats=sweep_build_stats,
            progress_callback=lambda **extra: emit_region_progress(
                "build_sweep_paths_region",
                stage_started,
                **extra,
            ),
        )
        finish_stage(
            "uturn_region_repair",
            stage_started,
            repaired_region_count=region_repair_count,
            feasible_region_count=len(feasible_patterns),
            infeasible_region_count=len(infeasible_regions),
        )
        if infeasible_regions:
            stage_started = time.perf_counter()
            regions = coarsened_regions
            raw_patterns = _generate_paper_style_patterns(
                regions,
                config,
                path_config,
                obstacle_field,
                progress_callback=lambda **extra: emit_region_progress(
                    "coverage_pattern_generation_region",
                    stage_started,
                    **extra,
                ),
            )
            sweep_paths, feasible_patterns, infeasible_regions, sweep_segment_templates = _build_region_sweep_paths(
                raw_patterns,
                config,
                path_config,
                obstacle_field,
                stats=sweep_build_stats,
                progress_callback=lambda **extra: emit_region_progress(
                    "build_sweep_paths_region",
                    stage_started,
                    **extra,
                ),
            )
            region_repair_count += len(coarsened_regions)
            finish_stage(
                "uturn_region_repair_fallback",
                stage_started,
                fallback_region_count=len(regions),
                feasible_region_count=len(feasible_patterns),
                infeasible_region_count=len(infeasible_regions),
            )
    feasible_regions = [region for region in regions if region.region_id in feasible_patterns]
    stage_started = time.perf_counter()
    graph = build_region_graph(feasible_regions, feasible_patterns, config, obstacle_field=obstacle_field, path_config=path_config)
    finish_stage("region_graph_building", stage_started, feasible_region_count=len(feasible_regions), edge_count=len(graph.edge_weights))
    stage_started = time.perf_counter()
    assignment = balance_region_workload(graph, config)
    load_swap_before = assignment.imbalance_ratio
    if path_config.enable_lightweight_load_swap:
        assignment = apply_lightweight_load_swap(
            assignment,
            graph,
            max_iterations=path_config.load_swap_max_iterations,
        )
    load_swap_count = int(assignment.diagnostics.get("load_swap_count", "0") or 0)
    load_swap_after = assignment.imbalance_ratio
    finish_stage(
        "load_balancing_assignment",
        stage_started,
        agent_count=len(assignment.agent_regions),
        load_swap_count=load_swap_count,
        load_swap_before_imbalance=round(float(load_swap_before), 6),
        load_swap_after_imbalance=round(float(load_swap_after), 6),
    )
    stage_started = time.perf_counter()
    ownership_map = build_coverage_ownership_map(
        feasible_regions,
        assignment.agent_regions,
        config,
        path_config,
        obstacle_field=obstacle_field,
    )
    finish_stage("coverage_ownership_map", stage_started, owned_region_count=len(ownership_map.region_owner))

    agents: Dict[int, AgentPathPlan] = {}
    tours: Dict[int, SingleUsvTourPlan] = {}
    tsp_records: Dict[int, Dict[str, object]] = {}
    infeasible_edges: List[Dict[str, object]] = []
    for agent_id, region_ids in sorted(assignment.agent_regions.items()):
        stage_started = time.perf_counter()
        result = _solve_agent_region_tsp(
            agent_id,
            region_ids,
            feasible_patterns,
            sweep_paths,
            sweep_segment_templates,
            config,
            path_config,
            obstacle_field,
            ownership_map,
        )
        finish_stage(
            f"agent_{agent_id}_region_tsp",
            stage_started,
            assigned_region_count=len(region_ids),
            final_region_count=len(result["final_order"]),
            segment_count=len(result["segments"]),
            infeasible_edge_count=len(result["infeasible_edges"]),
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
            "skipped_region_reasons": dict(result.get("skipped_region_reasons", {})),
            "connector_failure_reasons": dict(result.get("connector_failure_reasons", {})),
            "all_connector_failure_reasons": dict(result.get("all_connector_failure_reasons", {})),
            "main_repeat_overlap_length": float(result.get("main_repeat_overlap_length", 0.0)),
            "main_repeat_penalty_total": float(result.get("main_repeat_penalty_total", 0.0)),
            "cross_agent_overlap_length": float(result.get("cross_agent_overlap_length", 0.0)),
            "cross_agent_penalty_total": float(result.get("cross_agent_penalty_total", 0.0)),
            "unavoidable_cross_agent_overlap_count": int(result.get("unavoidable_cross_agent_overlap_count", 0)),
            "coverage_endpoint_count": sum(
                len(sweep_paths[region_id].endpoints)
                for region_id in region_ids
                if region_id in sweep_paths
            ),
            "infeasible_edges": list(result["infeasible_edges"]),
        }

    main_tsp_executed_region_ids = {
        region_id
        for record in tsp_records.values()
        for region_id in record.get("final_order", [])
    }
    residual_backfill_count = 0
    repeat_path_penalty_total = 0.0
    residual_local_tsp_status = "not_run"
    residual_backfill_diagnostics: List[Dict[str, str]] = []
    stage_started = time.perf_counter()
    skipped_region_recovery = _append_skipped_region_recovery(
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        agents=agents,
        tours=tours,
        tsp_records=tsp_records,
        feasible_patterns=feasible_patterns,
        sweep_segment_templates=sweep_segment_templates,
    )
    finish_stage(
        "skipped_region_recovery",
        stage_started,
        recovered_count=int(skipped_region_recovery.get("recovered_count", 0) or 0),
        failed_count=int(skipped_region_recovery.get("failed_count", 0) or 0),
        connector_cache_size=int(skipped_region_recovery.get("connector_cache_size", 0) or 0),
    )
    if skipped_region_recovery["recovered_count"]:
        for agent_id, agent in agents.items():
            agent.metrics = _agent_metrics(agent.segments, config, obstacle_field)
    coverage_state = evaluate_tour_coverage_state(
        config,
        list(tours.values()),
        resolution=path_config.residual_resolution,
        obstacle_field=obstacle_field,
        include_non_cover_segments=path_config.count_transit_coverage,
    )
    for _ in range(max(path_config.residual_backfill_cycles, 0)):
        if coverage_state.coverage_fraction + 1e-9 >= path_config.target_coverage_fraction:
            break
        stage_started = time.perf_counter()
        residual_result = append_residual_local_tsp(
            config=config,
            path_config=path_config,
            obstacle_field=obstacle_field,
            tours=tours,
            coverage_state=coverage_state,
            agents=agents,
            ownership_map=ownership_map,
        )
        residual_local_tsp_status = residual_result.diagnostics.get("status", "unknown")
        residual_backfill_diagnostics.append(dict(residual_result.diagnostics))
        finish_stage(
            f"residual_backfill_cycle_{len(residual_backfill_diagnostics)}",
            stage_started,
            appended_count=residual_result.appended_count,
            status=residual_local_tsp_status,
            residual_feasible_count=residual_result.diagnostics.get("residual_feasible_count", "0"),
            residual_infeasible_count=residual_result.diagnostics.get("residual_infeasible_count", "0"),
        )
        if residual_result.appended_count == 0:
            break
        residual_backfill_count += residual_result.appended_count
        repeat_path_penalty_total += residual_result.repeat_path_penalty_total
        for agent_id, agent in agents.items():
            agent.metrics = _agent_metrics(agent.segments, config, obstacle_field)
        coverage_state = evaluate_tour_coverage_state(
            config,
            list(tours.values()),
            resolution=path_config.residual_resolution,
            obstacle_field=obstacle_field,
            include_non_cover_segments=path_config.count_transit_coverage,
        )
        if coverage_state.coverage_fraction + 1e-9 >= path_config.target_coverage_fraction:
            break
    assign_stable_resource_ids(agents, path_config)
    shared_before_schedule = shared_resource_metrics(agents, path_config.resource_separation_time)
    mapf_conflicts_resolved_after_residual = apply_resource_window_schedule(
        agents,
        separation_time=path_config.resource_separation_time,
    )
    shared_after_schedule = shared_resource_metrics(agents, path_config.resource_separation_time)
    for agent_id, agent in agents.items():
        agent.metrics = _agent_metrics(agent.segments, config, obstacle_field)
    agent_repeat_metrics = _agent_repeat_metrics(agents, path_config)
    cross_agent_score = cross_agent_overlap_metrics(agents, ownership_map, path_config, config=config, annotate=True)
    totals = _global_metrics(agents)
    visit_nodes = _region_visit_nodes(feasible_patterns)
    corridor_conversion_report = _astar_corridor_conversion_report(agents, infeasible_edges)
    cover_only_coverage_state = evaluate_tour_coverage_state(
        config,
        list(tours.values()),
        resolution=path_config.residual_resolution,
        obstacle_field=obstacle_field,
        include_non_cover_segments=False,
    )
    skipped_region_ids = {
        region_id
        for record in tsp_records.values()
        for region_id in record.get("skipped_regions", [])
    }
    large_map_metadata = _large_map_tsp_metadata_summary(tsp_records)
    skipped_region_diagnostics = _skipped_region_diagnostics(
        skipped_region_ids,
        tsp_records,
        feasible_patterns,
    )
    residual_feasible_count = max(
        (int(item.get("residual_feasible_count", "0") or 0) for item in residual_backfill_diagnostics),
        default=0,
    )
    residual_infeasible_count = max(
        (int(item.get("residual_infeasible_count", "0") or 0) for item in residual_backfill_diagnostics),
        default=0,
    )
    residual_low_efficiency_filtered_count = sum(
        int(item.get("residual_low_efficiency_filtered_count", "0") or 0)
        for item in residual_backfill_diagnostics
    )
    residual_best_gain_per_path_meter = max(
        (float(item.get("residual_best_gain_per_path_meter", "0") or 0.0) for item in residual_backfill_diagnostics),
        default=0.0,
    )
    large_convex_region_count = sum(
        1 for region in regions if region.metadata.get("convex_region_decomposition") == "true"
    )
    rectangle_region_count = sum(
        1 for region in regions if str(region.metadata.get("shape_class", "")).lower() == "rectangle"
    )
    trapezoid_region_count = sum(
        1 for region in regions if str(region.metadata.get("shape_class", "")).lower() == "trapezoid"
    )
    fallback_cell_region_count = sum(
        1
        for region in regions
        if str(region.metadata.get("shape_class", "")).lower() == "fallback_cell"
        or region.source_algorithm in {"obstacle_aware_sweep_decomposition", "large_convex_obstacle_aligned_cell"}
    )
    raw_pattern_list = [pattern for candidates in raw_patterns.values() for pattern in candidates]
    selected_pattern_list = [pattern for tour in tours.values() for pattern in tour.selected_patterns.values()]
    oriented_pattern_count = sum(1 for pattern in raw_pattern_list if pattern.scan_axis.startswith("theta:"))
    axis_aligned_pattern_count = sum(1 for pattern in raw_pattern_list if not pattern.scan_axis.startswith("theta:"))
    selected_oriented_pattern_count = sum(1 for pattern in selected_pattern_list if pattern.scan_axis.startswith("theta:"))
    recovered_region_count = int(skipped_region_recovery.get("recovered_count", 0) or 0)
    target_coverage_met = coverage_state.coverage_fraction + 1e-9 >= path_config.target_coverage_fraction
    cover_only_target_fraction = path_config.target_coverage_fraction
    cover_only_coverage_gap = max(0.0, cover_only_target_fraction - cover_only_coverage_state.coverage_fraction)
    cover_only_target_met = cover_only_coverage_gap <= 1e-9
    region_execution_complete = not skipped_region_ids
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
        "stage_timing_sec": {key: round(value, 6) for key, value in stage_timing_sec.items()},
        "stage_monitor": stage_extra,
        "large_map_mode_enabled": _large_map_mode_enabled(config, path_config),
        "prefiltered_pattern_count": int(sweep_build_stats.get("prefiltered_pattern_count", 0) or 0),
        "prefiltered_variant_count": int(sweep_build_stats.get("prefiltered_variant_count", 0) or 0),
        "uturn_cache_hit_count": int(sweep_build_stats.get("uturn_cache_hit_count", 0) or 0),
        "uturn_cache_miss_count": int(sweep_build_stats.get("uturn_cache_miss_count", 0) or 0),
        "uturn_direct_fail_repair_success_count": int(
            sweep_build_stats.get("uturn_direct_fail_repair_success_count", 0) or 0
        ),
        "open_chain_region_count": int(sweep_build_stats.get("open_chain_region_count", 0) or 0),
        "open_chain_count": int(sweep_build_stats.get("open_chain_count", 0) or 0),
        "open_chain_break_count": int(sweep_build_stats.get("open_chain_break_count", 0) or 0),
        "open_chain_invalid_pass_count": int(sweep_build_stats.get("open_chain_invalid_pass_count", 0) or 0),
        "open_chain_connected_count": int(sweep_build_stats.get("open_chain_connected_count", 0) or 0),
        "open_chain_skipped_count": int(sweep_build_stats.get("open_chain_skipped_count", 0) or 0),
        "open_chain_recovered_coverage_length": float(sweep_build_stats.get("open_chain_recovered_coverage_length", 0.0) or 0.0),
        "rmin_aware_chain_enabled": bool(sweep_build_stats.get("rmin_aware_chain_enabled", False)),
        "turn_stride_distribution": dict(sweep_build_stats.get("turn_stride_distribution", {}) or {}),
        "rmin_180_attempt_count": int(sweep_build_stats.get("rmin_180_attempt_count", 0) or 0),
        "rmin_180_success_count": int(sweep_build_stats.get("rmin_180_success_count", 0) or 0),
        "rmin_180_feasible_ratio": (
            float(sweep_build_stats.get("rmin_180_success_count", 0) or 0)
            / max(float(sweep_build_stats.get("rmin_180_attempt_count", 0) or 0), 1.0)
        ),
        "single_pass_chain_count": int(sweep_build_stats.get("single_pass_chain_count", 0) or 0),
        "short_pass_residual_count": int(sweep_build_stats.get("short_pass_residual_count", 0) or 0),
        "adaptive_retraction_pattern_count": int(sweep_build_stats.get("adaptive_retraction_pattern_count", 0) or 0),
        "adaptive_retracted_pass_count": int(sweep_build_stats.get("adaptive_retracted_pass_count", 0) or 0),
        "adaptive_total_retraction_length": float(sweep_build_stats.get("adaptive_total_retraction_length", 0.0) or 0.0),
        "adaptive_max_retraction_length": float(sweep_build_stats.get("adaptive_max_retraction_length", 0.0) or 0.0),
        "adaptive_retraction_failed_count": int(sweep_build_stats.get("adaptive_retraction_failed_count", 0) or 0),
        "adaptive_retraction_extended_count": int(sweep_build_stats.get("adaptive_retraction_extended_count", 0) or 0),
        "region_repair_count": int(region_repair_count),
        "heading_repair_count": _heading_repair_count(agents),
        "raw_free_cell_count": base_region_count,
        "large_convex_region_count": large_convex_region_count,
        "rectangle_region_count": rectangle_region_count,
        "trapezoid_region_count": trapezoid_region_count,
        "fallback_cell_region_count": fallback_cell_region_count,
        "oriented_pattern_count": oriented_pattern_count,
        "axis_aligned_pattern_count": axis_aligned_pattern_count,
        "selected_oriented_pattern_count": selected_oriented_pattern_count,
        "main_tsp_executed_region_count": len(main_tsp_executed_region_ids),
        "recovered_region_count": recovered_region_count,
        "residual_only_region_count": len(skipped_region_ids),
        "composite_region_enabled": composite_mode_enabled,
        "composite_region_count": sum(1 for region in regions if getattr(region, "member_cells", None)),
        "composite_member_cell_count": sum(len(getattr(region, "member_cells", []) or []) for region in regions),
        "composite_split_count": composite_split_count,
        "large_convex_split_count": large_convex_split_count,
        "coarsened_region_count": len(coarsened_regions) if coarsened_regions is not None else len(regions),
        "merged_region_count": len(regions),
        "feasible_sweep_region_count": len(feasible_regions),
        "merge_rejected_count": sum(merge_diagnostics.values()),
        "merge_rejected_by_reason": dict(merge_diagnostics),
        "raw_astar_edge_rejected_count": _raw_astar_edge_rejected_count(infeasible_edges),
        "astar_corridor_conversion_attempt_count": corridor_conversion_report["attempt_count"],
        "astar_corridor_conversion_success_count": corridor_conversion_report["success_count"],
        "astar_corridor_conversion_failure_count": corridor_conversion_report["failure_count"],
        "corridor_conversion_methods": corridor_conversion_report["methods"],
        "astar_corridor_conversion_failure_reasons": corridor_conversion_report["failure_reasons"],
        "region_count": len(regions),
        "base_region_count": base_region_count,
        "performance_merge_fallback": performance_merge_fallback,
        "feasible_region_count": len(feasible_regions),
        "infeasible_regions": infeasible_regions,
        "infeasible_edges": infeasible_edges,
        "tsp_node_count": len(visit_nodes),
        "coverage_endpoint_count": sum(node.coverage_endpoint_count for node in visit_nodes.values()),
        "agent_tsp_records": tsp_records,
        "coverage_fraction": coverage_state.coverage_fraction,
        "cover_only_coverage_fraction": cover_only_coverage_state.coverage_fraction,
        "cover_only_target_fraction": cover_only_target_fraction,
        "cover_only_coverage_gap": cover_only_coverage_gap,
        "transit_assisted_coverage_fraction": coverage_state.coverage_fraction,
        "free_space_target_coverage": path_config.target_coverage_fraction,
        "residual_count": len(coverage_state.residual_components),
        "residual_backfill_count": residual_backfill_count,
        "residual_feasible_count": residual_feasible_count,
        "residual_infeasible_count": residual_infeasible_count,
        "residual_low_efficiency_filtered_count": residual_low_efficiency_filtered_count,
        "residual_best_gain_per_path_meter": residual_best_gain_per_path_meter,
        "residual_min_gain_per_path_meter": max(path_config.residual_min_gain_per_path_meter, 0.0),
        "residual_local_tsp_enabled": path_config.enable_residual_local_tsp,
        "residual_local_tsp_status": residual_local_tsp_status,
        "residual_backfill_diagnostics": residual_backfill_diagnostics,
        "skipped_region_recovery": skipped_region_recovery,
        "skipped_region_diagnostics": skipped_region_diagnostics,
        "reachable_region_count": len(visit_nodes) - len(skipped_region_ids),
        "region_connection_graph_components": large_map_metadata.get("region_connection_graph_components", {}),
        "skipped_by_no_incoming_edge": len(skipped_region_ids),
        "skipped_by_no_outgoing_edge": len(skipped_region_ids),
        "large_map_connector_cache_size": large_map_metadata.get("large_map_connector_cache_size", 0),
        "large_map_reachability_probe_count": large_map_metadata.get("large_map_reachability_probe_count", 0),
        "large_map_reachability_probe_success_count": large_map_metadata.get("large_map_reachability_probe_success_count", 0),
        "large_map_dead_end_avoidance_count": large_map_metadata.get("large_map_dead_end_avoidance_count", 0),
        "repeat_path_penalty_total": repeat_path_penalty_total,
        "load_swap_count": load_swap_count,
        "load_swap_before_imbalance": float(load_swap_before),
        "load_swap_after_imbalance": float(load_swap_after),
        "shared_resource_count": int(shared_after_schedule["shared_resource_count"]),
        "shared_resource_conflict_count": int(shared_before_schedule["true_time_conflict_count"]),
        "spatial_overlap_reuse_count": int(shared_after_schedule["spatial_overlap_reuse_count"]),
        "true_time_conflict_count": int(shared_after_schedule["true_time_conflict_count"]),
        "mapf_conflicts_resolved_after_residual": mapf_conflicts_resolved_after_residual,
        "resource_separation_time": path_config.resource_separation_time,
        "count_transit_coverage": path_config.count_transit_coverage,
        "enable_short_region_compression": path_config.enable_short_region_compression,
        "short_region_turn_ratio_threshold": path_config.short_region_turn_ratio_threshold,
        "main_repeat_path_penalty_enabled": path_config.enable_main_repeat_path_penalty,
        "main_repeat_overlap_length": sum(item["overlap_length"] for item in agent_repeat_metrics.values()),
        "main_repeat_penalty_total": sum(item["penalty"] for item in agent_repeat_metrics.values()),
        "agent_repeat_overlap": {str(agent_id): item["overlap_length"] for agent_id, item in agent_repeat_metrics.items()},
        "agent_repeat_penalty": {str(agent_id): item["penalty"] for agent_id, item in agent_repeat_metrics.items()},
        "cross_agent_penalty_enabled": path_config.enable_cross_agent_coverage_penalty,
        "cross_agent_overlap_length": cross_agent_score.overlap_length,
        "cross_agent_overlap_by_agent": {str(agent_id): length for agent_id, length in cross_agent_score.overlap_by_agent.items()},
        "cross_agent_overlap_by_kind": dict(cross_agent_score.overlap_by_kind),
        "cross_agent_penalty_total": cross_agent_score.penalty,
        "unavoidable_cross_agent_overlap_count": _unavoidable_cross_agent_overlap_count(agents),
        "coverage_ownership": {
            "cell_count": int(ownership_map.metadata.get("cell_count", "0")),
            "region_count": int(ownership_map.metadata.get("region_count", "0")),
            "conflict_count": int(ownership_map.metadata.get("conflict_count", "0")),
            "region_owner": {region_id: int(owner) for region_id, owner in ownership_map.region_owner.items()},
        },
        "metrics": totals,
    }
    report["performance_summary"] = build_performance_summary(
        agents=agents,
        coverage_state=coverage_state,
        totals=totals,
        repeat_overlap_length=float(report["main_repeat_overlap_length"]),
        path_config=path_config,
    )
    report["target_coverage_status"] = "complete" if target_coverage_met else "incomplete"
    report["cover_only_target_status"] = "complete" if cover_only_target_met else "incomplete"
    report["coverage_quality_status"] = "complete" if target_coverage_met and cover_only_target_met else "incomplete"
    report["region_execution_status"] = "complete" if region_execution_complete else "incomplete"
    report["coverage_status"] = report["target_coverage_status"]
    report["skipped_region_count"] = len(skipped_region_ids)
    report["skipped_regions"] = sorted(skipped_region_ids)
    path_plan = MultiAgentPathPlan(
        algorithm_name="paper_style_region_tsp",
        agents=agents,
        metadata={
            "status": "paper_style_region_tsp",
            "coverage_status": str(report["coverage_status"]),
            "target_coverage_status": str(report["target_coverage_status"]),
            "cover_only_target_status": str(report["cover_only_target_status"]),
            "coverage_quality_status": str(report["coverage_quality_status"]),
            "region_execution_status": str(report["region_execution_status"]),
            "skipped_region_count": str(report["skipped_region_count"]),
            "requested_tsp_solver": path_config.tsp_solver,
            "effective_tsp_solver": ",".join(
                sorted({str(record.get("effective_tsp_solver", "deterministic")) for record in tsp_records.values()})
            ),
            "tsp_solver_status": ",".join(
                sorted({str(record.get("tsp_solver_status", "success")) for record in tsp_records.values()})
            ),
            "region_count": str(len(feasible_regions)),
            "base_region_count": str(base_region_count),
            "performance_merge_fallback": str(performance_merge_fallback).lower(),
            "tsp_node_count": str(len(visit_nodes)),
            "coverage_endpoint_count": str(report["coverage_endpoint_count"]),
            "coverage_fraction": f"{coverage_state.coverage_fraction:.6f}",
            "cover_only_coverage_fraction": f"{cover_only_coverage_state.coverage_fraction:.6f}",
            "cover_only_target_fraction": f"{cover_only_target_fraction:.6f}",
            "cover_only_coverage_gap": f"{cover_only_coverage_gap:.6f}",
            "count_transit_coverage": str(path_config.count_transit_coverage).lower(),
            "enable_short_region_compression": str(path_config.enable_short_region_compression).lower(),
            "residual_count": str(len(coverage_state.residual_components)),
            "residual_backfill_count": str(residual_backfill_count),
            "residual_low_efficiency_filtered_count": str(residual_low_efficiency_filtered_count),
            "residual_local_tsp_enabled": str(path_config.enable_residual_local_tsp).lower(),
            "repeat_path_penalty_total": f"{repeat_path_penalty_total:.6f}",
            "mapf_conflicts_resolved_after_residual": str(mapf_conflicts_resolved_after_residual),
            "shared_resource_count": str(int(shared_after_schedule["shared_resource_count"])),
            "shared_resource_conflict_count": str(int(shared_before_schedule["true_time_conflict_count"])),
            "spatial_overlap_reuse_count": str(int(shared_after_schedule["spatial_overlap_reuse_count"])),
            "true_time_conflict_count": str(int(shared_after_schedule["true_time_conflict_count"])),
            "main_repeat_path_penalty_enabled": str(path_config.enable_main_repeat_path_penalty).lower(),
            "main_repeat_overlap_length": f"{report['main_repeat_overlap_length']:.6f}",
            "main_repeat_penalty_total": f"{report['main_repeat_penalty_total']:.6f}",
            "cross_agent_penalty_enabled": str(path_config.enable_cross_agent_coverage_penalty).lower(),
            "cross_agent_overlap_length": f"{report['cross_agent_overlap_length']:.6f}",
            "cross_agent_penalty_total": f"{report['cross_agent_penalty_total']:.6f}",
            "unavoidable_cross_agent_overlap_count": str(report["unavoidable_cross_agent_overlap_count"]),
            "performance_profile": str(report["performance_summary"]["performance_profile"]),
            "large_map_mode_enabled": str(report["large_map_mode_enabled"]).lower(),
            "prefiltered_pattern_count": str(report["prefiltered_pattern_count"]),
            "uturn_cache_hit_count": str(report["uturn_cache_hit_count"]),
            "uturn_cache_miss_count": str(report["uturn_cache_miss_count"]),
            "region_repair_count": str(report["region_repair_count"]),
            "heading_repair_count": str(report["heading_repair_count"]),
            "large_convex_region_count": str(report["large_convex_region_count"]),
            "main_tsp_executed_region_count": str(report["main_tsp_executed_region_count"]),
            "recovered_region_count": str(report["recovered_region_count"]),
            "residual_only_region_count": str(report["residual_only_region_count"]),
            "load_swap_count": str(report["load_swap_count"]),
            "load_swap_before_imbalance": f"{report['load_swap_before_imbalance']:.6f}",
            "load_swap_after_imbalance": f"{report['load_swap_after_imbalance']:.6f}",
            "target_coverage_fraction": f"{float(report['performance_summary']['target_coverage_fraction']):.6f}",
            "target_coverage_met": str(bool(report["performance_summary"]["target_coverage_met"])).lower(),
            "coverage_length_ratio": f"{float(report['performance_summary']['coverage_length_ratio']):.6f}",
            "transition_length_ratio": f"{float(report['performance_summary']['transition_length_ratio']):.6f}",
            "repeat_transition_ratio": f"{float(report['performance_summary']['repeat_transition_ratio']):.6f}",
            "performance_objective": f"{float(report['performance_summary']['performance_objective']):.6f}",
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
            ownership_map,
            path_config,
            output_dir,
            dpi=path_config.visual_dpi,
            raw_regions=raw_free_regions,
            candidate_regions=regions,
        )
        path_plan.metadata["paper_style_output_dir"] = str(artifact_dir)
        path_plan.metadata["paper_style_report"] = str(artifact_dir / "paper_style_region_tsp_report.json")
    return path_plan, report


def _apply_performance_profile(path_config: PathPlanningConfig) -> PathPlanningConfig:
    profile = str(path_config.performance_profile or "balanced").strip().lower()
    if profile not in {"balanced", "shortest", "low-repeat", "low_repeat"}:
        raise ValueError("performance_profile must be one of: balanced, shortest, low-repeat")
    normalized = "low-repeat" if profile == "low_repeat" else profile
    if normalized == "shortest":
        return replace(
            path_config,
            performance_profile=normalized,
            main_repeat_path_penalty_weight=max(path_config.main_repeat_path_penalty_weight * 0.5, 4.0),
            internal_uturn_repeat_path_penalty_weight=max(path_config.internal_uturn_repeat_path_penalty_weight * 0.5, 4.0),
            transition_length_weight=max(path_config.transition_length_weight, 1.2),
            repeat_transition_weight=max(path_config.repeat_transition_weight * 0.5, 4.0),
        )
    if normalized == "low-repeat":
        return replace(
            path_config,
            performance_profile=normalized,
            main_repeat_path_penalty_weight=max(path_config.main_repeat_path_penalty_weight, 18.0),
            internal_uturn_repeat_path_penalty_weight=max(path_config.internal_uturn_repeat_path_penalty_weight, 18.0),
            repeat_transition_weight=max(path_config.repeat_transition_weight, 18.0),
        )
    return replace(path_config, performance_profile=normalized)


def _large_map_mode_enabled(config: PlannerConfig, path_config: PathPlanningConfig) -> bool:
    threshold = max(float(path_config.large_map_size_threshold), 1e-6)
    return max(config.mission.area_length_x, config.mission.area_length_y) >= threshold


def _large_map_aco_region_limit(path_config: PathPlanningConfig) -> int:
    return max(8, int(path_config.region_tsp_branch_limit))


def _connector_pattern_limit(path_config: PathPlanningConfig) -> int:
    return max(1, int(path_config.large_region_connector_pattern_limit))


def _apply_large_map_defaults(path_config: PathPlanningConfig, config: PlannerConfig) -> PathPlanningConfig:
    if not _large_map_mode_enabled(config, path_config):
        return path_config
    return replace(
        path_config,
        enable_large_map_sweep_prefilter=True,
        max_candidate_axes=1,
        max_prefiltered_patterns_per_region=max(int(path_config.max_prefiltered_patterns_per_region), 4),
        max_prefiltered_variants_per_pattern=min(max(int(path_config.max_prefiltered_variants_per_pattern), 1), 2),
        max_entry_exit_patterns_per_region=max(int(path_config.max_entry_exit_patterns_per_region), 6),
        large_region_connector_pattern_limit=max(int(path_config.large_region_connector_pattern_limit), 6),
        region_tsp_beam_width=max(int(path_config.region_tsp_beam_width), 4),
        region_tsp_branch_limit=max(int(path_config.region_tsp_branch_limit), 16),
        min_sweep_pattern_coverage_fraction=max(path_config.min_sweep_pattern_coverage_fraction, 0.98),
        min_compressed_pattern_coverage_fraction=max(path_config.min_compressed_pattern_coverage_fraction, 0.98),
        enable_uturn_validation_cache=True,
        enable_infeasible_uturn_region_repair=path_config.enable_infeasible_uturn_region_repair,
        enable_motion_lattice_heading_repair=True,
        enable_open_sweep_chain_tsp=False,
    )


def _split_infeasible_composite_regions(regions: Sequence, infeasible_regions: Sequence[Dict[str, object]]) -> Tuple[List, int]:
    infeasible_ids = {str(item.get("region_id", "")) for item in infeasible_regions}
    repaired: List = []
    split_count = 0
    for region in regions:
        member_cells = list(getattr(region, "member_cells", []) or [])
        if region.region_id not in infeasible_ids or len(member_cells) <= 1:
            repaired.append(region)
            continue
        split_count += 1
        for idx, cell in enumerate(member_cells):
            repaired.append(
                DecomposedRegion(
                    region_id=f"{region.region_id}_{cell.cell_id}_{idx}",
                    bounds=cell.bounds,
                    polygon=list(cell.polygon),
                    center=cell.center,
                    area=cell.area,
                    preferred_axis=cell.preferred_axis,
                    source_algorithm="composite_split_repair",
                    neighbors=[],
                    metadata={
                        **dict(cell.metadata),
                        "static_obstacle_aware": "true",
                        "parent_composite_region": region.region_id,
                        "source_region_ids": cell.cell_id,
                        "composite_split_repair": "true",
                    },
                )
            )
    return repaired, split_count


def _split_infeasible_large_convex_regions(
    regions: Sequence,
    infeasible_regions: Sequence[Dict[str, object]],
    config: PlannerConfig,
) -> Tuple[List, int]:
    infeasible_ids = {str(item.get("region_id", "")) for item in infeasible_regions}
    repaired: List = []
    split_count = 0
    min_area = max(config.footprint.length_lf * config.footprint.width_wf, 1e-6)
    for region in regions:
        if (
            region.region_id not in infeasible_ids
            or region.metadata.get("convex_region_decomposition") != "true"
            or region.area <= 1.5 * min_area
        ):
            repaired.append(region)
            continue
        x0, y0, x1, y1 = region.bounds
        width = x1 - x0
        height = y1 - y0
        if width <= 1e-9 or height <= 1e-9:
            repaired.append(region)
            continue
        split_count += 1
        if width >= height:
            mid = (x0 + x1) / 2.0
            split_bounds = [(x0, y0, mid, y1), (mid, y0, x1, y1)]
        else:
            mid = (y0 + y1) / 2.0
            split_bounds = [(x0, y0, x1, mid), (x0, mid, x1, y1)]
        for idx, bounds in enumerate(split_bounds):
            bx0, by0, bx1, by1 = bounds
            polygon = [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]
            preferred_axis = "x" if (bx1 - bx0) >= (by1 - by0) else "y"
            area = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
            repaired.append(
                DecomposedRegion(
                    region_id=f"{region.region_id}_split_{idx}",
                    bounds=bounds,
                    polygon=polygon,
                    center=((bx0 + bx1) / 2.0, (by0 + by1) / 2.0),
                    area=area,
                    preferred_axis=preferred_axis,
                    source_algorithm="large_convex_split_repair",
                    neighbors=[],
                    metadata={
                        **dict(region.metadata),
                        "shape_class": "rectangle",
                        "dominant_scan_axis": preferred_axis,
                        "support_span": f"{(by1 - by0) if preferred_axis == 'x' else (bx1 - bx0):.6f}",
                        "parent_large_region": region.region_id,
                        "large_convex_split_repair": "true",
                        "decomposition_fallback_reason": "infeasible_internal_uturn_split",
                    },
                )
            )
    _populate_region_neighbors(repaired)
    return repaired or list(regions), split_count


def _coarsen_paper_style_regions(
    regions: Sequence,
    config: PlannerConfig,
    obstacle_field: ObstacleField | None = None,
    diagnostics: Dict[str, int] | None = None,
) -> List:
    columns: Dict[Tuple[float, float], List] = {}
    for region in regions:
        x_min, _, x_max, _ = region.bounds
        columns.setdefault((round(x_min, 6), round(x_max, 6)), []).append(region)
    coarsened: List = []
    serial = 0

    def flush_group(group: List) -> None:
        nonlocal serial
        if not group:
            return
        x_min = min(member.bounds[0] for member in group)
        y_min = min(member.bounds[1] for member in group)
        x_max = max(member.bounds[2] for member in group)
        y_max = max(member.bounds[3] for member in group)
        width = x_max - x_min
        height = y_max - y_min
        if width <= 1e-9 or height <= 1e-9:
            return
        preferred_axis = "y" if height >= width else "x"
        polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        coarsened.append(
            replace(
                group[0],
                region_id=f"paper_col_{serial}",
                bounds=(x_min, y_min, x_max, y_max),
                polygon=polygon,
                center=((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
                area=width * height,
                preferred_axis=preferred_axis,
                source_algorithm="paper_style_column_coarsening",
                neighbors=[],
                metadata={
                    "paper_style_coarsened": "true",
                    "source_cell_count": str(len(group)),
                    "source_region_ids": ",".join(member.region_id for member in group),
                    "static_obstacle_aware": "true",
                },
            )
        )
        serial += 1

    for _, members in sorted(columns.items()):
        ordered = sorted(members, key=lambda item: (item.bounds[1], item.bounds[3]))
        group: List = []
        current_bounds: Tuple[float, float, float, float] | None = None
        for member in ordered:
            if not group or current_bounds is None:
                group = [member]
                current_bounds = member.bounds
                continue
            gap = member.bounds[1] - current_bounds[3]
            candidate_bounds = _union_bounds(current_bounds, member.bounds)
            if gap > 1e-6:
                _increment_diagnostic(diagnostics, "obstacle_gap")
                flush_group(group)
                group = [member]
                current_bounds = member.bounds
                continue
            if _bounds_collide_with_obstacles(candidate_bounds, obstacle_field):
                _increment_diagnostic(diagnostics, "obstacle_collision")
                flush_group(group)
                group = [member]
                current_bounds = member.bounds
                continue
            group.append(member)
            current_bounds = candidate_bounds
        flush_group(group)
    _populate_region_neighbors(coarsened)
    return coarsened or list(regions)


def _merge_performance_regions(
    regions: Sequence,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None = None,
    diagnostics: Dict[str, int] | None = None,
) -> List:
    if path_config.performance_profile not in {"balanced", "shortest", "low-repeat"}:
        return list(regions)
    min_width = max(config.footprint.width_wf * path_config.cell_merge_width_factor, 0.0)
    min_coverage_length = max(config.footprint.length_lf * path_config.min_pass_length_factor, 0.0)
    merged: List = []
    rows: Dict[Tuple[float, float], List] = {}
    for region in regions:
        rows.setdefault((round(region.bounds[1], 6), round(region.bounds[3], 6)), []).append(region)
    for _, row_regions in sorted(rows.items()):
        ordered = sorted(row_regions, key=lambda item: (item.bounds[0], item.bounds[2]))
        cursor = None
        pending_sources: List[str] = []
        for region in ordered:
            if cursor is None:
                cursor = region
                pending_sources = [region.region_id]
                continue
            width = cursor.bounds[2] - cursor.bounds[0]
            height = cursor.bounds[3] - cursor.bounds[1]
            should_merge = width < min_width or max(width, height) < min_coverage_length
            adjacent = abs(cursor.bounds[2] - region.bounds[0]) <= 1e-6
            merge_safe, reject_reason = _performance_merge_safe(cursor, region, obstacle_field)
            if should_merge and adjacent and merge_safe:
                cursor = _merge_two_regions(cursor, region, pending_sources + [region.region_id])
                pending_sources.append(region.region_id)
            else:
                if should_merge:
                    if not adjacent:
                        _increment_diagnostic(diagnostics, "non_contiguous")
                    elif not merge_safe:
                        _increment_diagnostic(diagnostics, reject_reason)
                merged.append(cursor)
                cursor = region
                pending_sources = [region.region_id]
        if cursor is not None:
            merged.append(cursor)
    for idx, region in enumerate(merged):
        region.region_id = f"perf_region_{idx}"
        region.metadata["performance_merged"] = str(region.metadata.get("source_cell_count", "1") != "1").lower()
    _populate_region_neighbors(merged)
    return merged or list(regions)


def _increment_diagnostic(diagnostics: Dict[str, int] | None, key: str) -> None:
    if diagnostics is None:
        return
    diagnostics[key] = int(diagnostics.get(key, 0)) + 1


def _union_bounds(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _bounds_polygon(bounds: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    x_min, y_min, x_max, y_max = bounds
    return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]


def _bounds_collide_with_obstacles(
    bounds: Tuple[float, float, float, float],
    obstacle_field: ObstacleField | None,
) -> bool:
    if obstacle_field is None:
        return False
    return polygon_collides_with_obstacles(_bounds_polygon(bounds), obstacle_field, inflated=True)


def _performance_merge_safe(first, second, obstacle_field: ObstacleField | None) -> Tuple[bool, str]:
    x_touch = abs(first.bounds[2] - second.bounds[0]) <= 1e-6 or abs(second.bounds[2] - first.bounds[0]) <= 1e-6
    y_overlap = min(first.bounds[3], second.bounds[3]) - max(first.bounds[1], second.bounds[1])
    if not x_touch or y_overlap <= 1e-6:
        return False, "non_contiguous"
    candidate_bounds = _union_bounds(first.bounds, second.bounds)
    if _bounds_collide_with_obstacles(candidate_bounds, obstacle_field):
        return False, "obstacle_collision"
    return True, ""


def _merge_two_regions(first, second, source_ids: Sequence[str]):
    x_min = min(first.bounds[0], second.bounds[0])
    y_min = min(first.bounds[1], second.bounds[1])
    x_max = max(first.bounds[2], second.bounds[2])
    y_max = max(first.bounds[3], second.bounds[3])
    width = x_max - x_min
    height = y_max - y_min
    preferred_axis = "y" if height >= width else "x"
    polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    return replace(
        first,
        bounds=(x_min, y_min, x_max, y_max),
        polygon=polygon,
        center=((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
        area=max(width, 0.0) * max(height, 0.0),
        preferred_axis=preferred_axis,
        neighbors=[],
        metadata={
            **first.metadata,
            "performance_merged": "true",
            "source_region_ids": ",".join(source_ids),
            "source_cell_count": str(len(source_ids)),
        },
    )


def _repair_infeasible_merged_regions(
    merged_regions: Sequence,
    coarsened_regions: Sequence,
    infeasible_regions: Sequence[Dict[str, object]],
) -> List:
    infeasible_ids = {
        str(item.get("region_id"))
        for item in infeasible_regions
        if item.get("region_id") is not None
    }
    coarsened_by_id = {region.region_id: region for region in coarsened_regions}
    repaired = []
    for region in merged_regions:
        if region.region_id not in infeasible_ids:
            repaired.append(region)
            continue
        source_ids = [
            item.strip()
            for item in str(region.metadata.get("source_region_ids", "")).split(",")
            if item.strip()
        ]
        replacements = [coarsened_by_id[item] for item in source_ids if item in coarsened_by_id]
        repaired.extend(replacements or [region])
    _populate_region_neighbors(repaired)
    return repaired or list(coarsened_regions)


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
    stats: Dict[str, object] | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> Tuple[
    Dict[str, RegionSweepPath],
    Dict[str, List[RegionCoveragePattern]],
    List[Dict[str, object]],
    Dict[str, Tuple[List[PathSegmentSpec], str]],
]:
    stats = stats if stats is not None else {}
    uturn_cache: Dict[Tuple[object, ...], Tuple[bool, str]] = {}
    stats.setdefault("prefiltered_pattern_count", 0)
    stats.setdefault("prefiltered_variant_count", 0)
    stats.setdefault("uturn_cache_hit_count", 0)
    stats.setdefault("uturn_cache_miss_count", 0)
    stats.setdefault("uturn_direct_fail_repair_success_count", 0)
    stats.setdefault("rmin_aware_chain_enabled", False)
    stats.setdefault("turn_stride_distribution", {})
    stats.setdefault("rmin_180_attempt_count", 0)
    stats.setdefault("rmin_180_success_count", 0)
    stats.setdefault("single_pass_chain_count", 0)
    stats.setdefault("short_pass_residual_count", 0)
    stats.setdefault("adaptive_retraction_pattern_count", 0)
    stats.setdefault("adaptive_retracted_pass_count", 0)
    stats.setdefault("adaptive_total_retraction_length", 0.0)
    stats.setdefault("adaptive_max_retraction_length", 0.0)
    stats.setdefault("adaptive_retraction_failed_count", 0)
    stats.setdefault("adaptive_retraction_extended_count", 0)
    sweep_paths: Dict[str, RegionSweepPath] = {}
    feasible_patterns: Dict[str, List[RegionCoveragePattern]] = {}
    infeasible_regions: List[Dict[str, object]] = []
    sweep_segment_templates: Dict[str, Tuple[List[PathSegmentSpec], str]] = {}
    ordered_regions = sorted(raw_patterns.items())
    region_count = len(ordered_regions)
    for region_index, (region_id, candidates) in enumerate(ordered_regions, start=1):
        region_started = time.perf_counter()
        if progress_callback is not None:
            progress_callback(
                event="start",
                region_id=region_id,
                region_index=region_index,
                region_count=region_count,
                raw_candidate_count=len(candidates),
            )
        feasible_for_region: List[Tuple[RegionCoveragePattern, RegionSweepPath]] = []
        reasons: List[str] = []
        raw_candidate_count = len(candidates)
        candidates = _prefilter_region_patterns(candidates, config, path_config)
        stats["prefiltered_pattern_count"] = int(stats["prefiltered_pattern_count"]) + max(0, raw_candidate_count - len(candidates))
        if raw_candidate_count == 0:
            reasons.append("no_candidate_patterns")
        elif not candidates:
            reasons.append("all_candidate_patterns_prefiltered")
        raw_variant_total = 0
        filtered_variant_total = 0
        attempted_variant_count = 0
        feasible_variant_count = 0
        failed_variant_count = 0
        for pattern in candidates:
            variants = _pattern_variants(pattern, config, path_config)
            variants.extend(_short_region_compression_variants(pattern, config, path_config))
            raw_variant_count = len(variants)
            raw_variant_total += raw_variant_count
            variants = _prefilter_pattern_variants(variants, config, path_config)
            filtered_variant_total += len(variants)
            stats["prefiltered_variant_count"] = int(stats["prefiltered_variant_count"]) + max(0, raw_variant_count - len(variants))
            for variant in variants:
                attempted_variant_count += 1
                variant = _annotate_pattern_coverage_quality(_normalize_pattern_headings(variant), config, path_config)
                _accumulate_retraction_stats(stats, variant)
                internal_segments, reason = _build_internal_sweep_segments(
                    variant,
                    config,
                    path_config,
                    obstacle_field,
                    uturn_cache=uturn_cache,
                    stats=stats,
                    start_time=0.0,
                    segment_prefix="validate",
                )
                if not reason:
                    if internal_segments:
                        sweep_segment_templates[_pattern_template_key(variant)] = (copy.deepcopy(internal_segments), "")
                    variant = _annotate_pattern_coverage_quality(
                        _annotate_pattern_internal_repeat(variant, config, path_config, obstacle_field),
                        config,
                        path_config,
                    )
                    feasible_for_region.append((variant, _sweep_path_from_pattern(variant)))
                    feasible_variant_count += 1
                else:
                    failed_variant_count += 1
                    reasons.append(f"{variant.pattern_id}:{reason}")
        if feasible_for_region:
            if _coverage_quality_priority_enabled(config, path_config):
                max_fraction = max(_estimated_pattern_coverage_fraction(item[0], config) for item in feasible_for_region)
                min_fraction = max(0.0, min(1.0, path_config.min_sweep_pattern_coverage_fraction))
                fraction_floor = min(max_fraction, min_fraction)
                high_coverage = [
                    item
                    for item in feasible_for_region
                    if _estimated_pattern_coverage_fraction(item[0], config) + 1e-9 >= fraction_floor
                ]
                if high_coverage:
                    feasible_for_region = high_coverage
            max_coverage = max(item[0].coverage_length for item in feasible_for_region)
            coverage_ratio = (
                path_config.multi_entry_exit_coverage_floor
                if path_config.enable_multi_entry_exit_patterns
                else 0.8
            )
            coverage_floor = max(0.0, min(1.0, coverage_ratio)) * max_coverage
            feasible_for_region = [item for item in feasible_for_region if item[0].coverage_length + 1e-9 >= coverage_floor]
            feasible_for_region.sort(key=lambda item: _pattern_sort_key(item[0], config, path_config))
            limit = max(int(path_config.max_entry_exit_patterns_per_region), 1)
            feasible_for_region = feasible_for_region[:limit]
            feasible_patterns[region_id] = [item[0] for item in feasible_for_region]
            sweep_paths[region_id] = feasible_for_region[0][1]
            status = "feasible"
        else:
            infeasible_regions.append({"region_id": region_id, "reasons": reasons[:6]})
            status = "infeasible"
        if progress_callback is not None:
            progress_callback(
                event="done",
                region_id=region_id,
                region_index=region_index,
                region_count=region_count,
                status=status,
                dt_sec=round(time.perf_counter() - region_started, 3),
                raw_candidate_count=raw_candidate_count,
                candidate_count=len(candidates),
                prefiltered_candidate_count=max(0, raw_candidate_count - len(candidates)),
                raw_variant_count=raw_variant_total,
                variant_count=filtered_variant_total,
                attempted_variant_count=attempted_variant_count,
                feasible_variant_count=feasible_variant_count,
                failed_variant_count=failed_variant_count,
                selected_pattern_count=len(feasible_patterns.get(region_id, [])),
                reason_count=len(reasons),
                sample_reasons=reasons[:3],
                uturn_cache_size=len(uturn_cache),
            )
    return sweep_paths, feasible_patterns, infeasible_regions, sweep_segment_templates


def _prefilter_region_patterns(
    candidates: Sequence[RegionCoveragePattern],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> List[RegionCoveragePattern]:
    if not path_config.enable_large_map_sweep_prefilter:
        return list(candidates)
    limit = max(int(path_config.max_prefiltered_patterns_per_region), 1)
    if len(candidates) <= limit:
        return list(candidates)
    ranked = sorted(candidates, key=lambda item: (_light_pattern_score(item, config, path_config), item.pattern_id))
    selected: List[RegionCoveragePattern] = []
    seen_ids = set()
    for axis in sorted({pattern.scan_axis for pattern in ranked}):
        axis_candidates = [pattern for pattern in ranked if pattern.scan_axis == axis]
        if axis_candidates:
            best = max(
                axis_candidates,
                key=lambda item: (_estimated_pattern_coverage_fraction(item, config), -_light_pattern_score(item, config, path_config), item.pattern_id),
            )
            selected.append(best)
            seen_ids.add(best.pattern_id)
    if len(selected) > limit:
        selected = sorted(selected, key=lambda item: (_light_pattern_score(item, config, path_config), item.pattern_id))[:limit]
        seen_ids = {pattern.pattern_id for pattern in selected}
    for pattern in ranked:
        if len(selected) >= limit:
            break
        if pattern.pattern_id in seen_ids:
            continue
        selected.append(pattern)
        seen_ids.add(pattern.pattern_id)
    return selected or ranked[:1]


def _prefilter_pattern_variants(
    variants: Sequence[RegionCoveragePattern],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> List[RegionCoveragePattern]:
    if not path_config.enable_large_map_sweep_prefilter:
        return list(variants)
    limit = max(int(path_config.max_prefiltered_variants_per_pattern), 1)
    if len(variants) <= limit:
        return list(variants)
    ranked = sorted(variants, key=lambda item: (_light_pattern_score(item, config, path_config), item.pattern_id))
    selected: List[RegionCoveragePattern] = []
    seen_keys = set()
    for variant in ranked:
        key = _variant_diversity_key(variant)
        if key in seen_keys:
            continue
        selected.append(variant)
        seen_keys.add(key)
        if len(selected) >= limit:
            break
    for variant in ranked:
        if len(selected) >= limit:
            break
        if variant in selected:
            continue
        selected.append(variant)
    return selected or ranked[:1]


def _variant_diversity_key(pattern: RegionCoveragePattern) -> Tuple[str, str, str]:
    return (
        pattern.scan_axis,
        str(pattern.metadata.get("scan_order", "")),
        str(pattern.metadata.get("entry_side", "")),
    )


def _light_pattern_score(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig | None = None,
) -> float:
    if not pattern.passes:
        return float("inf")
    coverage_fraction = _estimated_pattern_coverage_fraction(pattern, config)
    coverage_deficit = max(0.0, 0.98 - coverage_fraction)
    short_pass_threshold = max(config.footprint.length_lf, config.footprint.width_wf * 2.0)
    short_pass_ratio = sum(1 for item in pattern.passes if item.length < short_pass_threshold) / max(len(pattern.passes), 1)
    endpoint_penalty = _turn_clearance_penalty(pattern.entry_pose, config) + _turn_clearance_penalty(pattern.exit_pose, config)
    pass_count_penalty = 0.15 * len(pattern.passes)
    quality_penalty = _pattern_quality_penalty(pattern, path_config) if path_config is not None else 0.0
    return (
        500.0 * coverage_deficit
        + quality_penalty
        + pattern.estimated_time
        + 0.8 * pattern.turn_length
        + 20.0 * short_pass_ratio
        + endpoint_penalty
        + pass_count_penalty
        - 0.1 * pattern.coverage_length
    )


def _generate_paper_style_patterns(
    regions,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    progress_callback: Callable[..., None] | None = None,
) -> Dict[str, List[RegionCoveragePattern]]:
    region_list = list(regions)
    merged: Dict[str, List[RegionCoveragePattern]] = {region.region_id: [] for region in region_list}
    pocket_scales = (0.0,) if path_config.enable_adaptive_pass_retraction else (0.0, 0.5, 1.0)
    region_count = len(region_list)
    for region_index, region in enumerate(region_list, start=1):
        region_started = time.perf_counter()
        if progress_callback is not None:
            progress_callback(
                event="start",
                region_id=region.region_id,
                region_index=region_index,
                region_count=region_count,
                preferred_axis=region.preferred_axis,
                area=round(float(region.area), 3),
                bounds=[round(float(value), 3) for value in region.bounds],
                pocket_scale_count=len(pocket_scales),
            )
        pass_count_total = 0
        max_pass_count = 0
        coverage_length_total = 0.0
        oriented_pattern_count = 0
        axis_aligned_pattern_count = 0
        angle_candidate_count = 0
        best_support_span = float("inf")
        best_angle_summary = ""
        for pocket_scale in pocket_scales:
            candidate_config = replace(path_config, coverage_turn_pocket_scale=pocket_scale)
            candidates = generate_region_patterns(region, config, candidate_config, obstacle_field=obstacle_field)
            for pattern in candidates:
                suffix = str(pocket_scale).replace(".", "p")
                merged[region.region_id].append(
                    replace(
                        pattern,
                        pattern_id=f"{pattern.pattern_id}_pocket_{suffix}",
                        metadata={**pattern.metadata, "turn_pocket_scale": f"{pocket_scale:.2f}"},
                    )
                )
                pass_count = len(pattern.passes)
                pass_count_total += pass_count
                max_pass_count = max(max_pass_count, pass_count)
                coverage_length_total += float(pattern.coverage_length)
                if pattern.scan_axis.startswith("theta:"):
                    oriented_pattern_count += 1
                else:
                    axis_aligned_pattern_count += 1
                try:
                    angle_candidate_count = max(
                        angle_candidate_count,
                        int(pattern.metadata.get("angle_candidate_count", "0") or 0),
                    )
                except ValueError:
                    pass
                try:
                    support_span = float(pattern.metadata.get("support_span", "nan"))
                except ValueError:
                    support_span = float("nan")
                if math.isfinite(support_span) and support_span + 1e-9 < best_support_span:
                    best_support_span = support_span
                    best_angle_summary = (
                        f"{pattern.metadata.get('scan_angle_deg', '')}:"
                        f"{pattern.metadata.get('scan_axis_source', '')}:"
                        f"{support_span:.3f}"
                    )
        if progress_callback is not None:
            progress_callback(
                event="done",
                region_id=region.region_id,
                region_index=region_index,
                region_count=region_count,
                dt_sec=round(time.perf_counter() - region_started, 3),
                pattern_count=len(merged[region.region_id]),
                oriented_pattern_count=oriented_pattern_count,
                axis_aligned_pattern_count=axis_aligned_pattern_count,
                angle_candidate_count=angle_candidate_count,
                best_angle_summary=best_angle_summary,
                pass_count_total=pass_count_total,
                max_pass_count=max_pass_count,
                coverage_length_total=round(coverage_length_total, 3),
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


def _short_region_compression_variants(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> List[RegionCoveragePattern]:
    if not path_config.enable_short_region_compression or len(pattern.passes) <= 1:
        return []
    if pattern.coverage_length <= 1e-9:
        return []
    turn_ratio = pattern.turn_length / max(pattern.coverage_length, 1e-9)
    average_pass_length = pattern.coverage_length / max(len(pattern.passes), 1)
    min_useful_pass = max(config.footprint.length_lf * 0.5, config.footprint.width_wf)
    if turn_ratio < path_config.short_region_turn_ratio_threshold and average_pass_length >= min_useful_pass:
        return []
    bounds = _coverage_pass_bounds(pattern.passes)
    if bounds is None:
        return []
    x_min, y_min, x_max, y_max = bounds
    region_area = _pattern_region_area(pattern, bounds)
    width_x = max(x_max - x_min, 0.0)
    width_y = max(y_max - y_min, 0.0)
    narrow_limit = config.footprint.width_wf * max(path_config.compressed_region_width_factor, 1.0)
    min_coverage = max(0.0, min(1.0, path_config.min_compressed_pattern_coverage_fraction))
    candidates: List[RegionCoveragePattern] = []
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    specs = [
        ("x", Pose2D(x_min, cy, 0.0), Pose2D(x_max, cy, 0.0), abs(x_max - x_min), cy),
        ("y", Pose2D(cx, y_min, math.pi / 2.0), Pose2D(cx, y_max, math.pi / 2.0), abs(y_max - y_min), cx),
    ]
    for axis, start, end, length, center in specs:
        if length <= 1e-6:
            continue
        coverage_fraction = min(1.0, max(0.0, length * config.footprint.width_wf / max(region_area, 1e-9)))
        cross_width = width_y if axis == "x" else width_x
        if _large_map_mode_enabled(config, path_config) and cross_width > narrow_limit and coverage_fraction + 1e-9 < min_coverage:
            continue
        coverage_pass = CoveragePass(
            pass_id=f"{pattern.region_id}_{axis}_compressed_pass",
            region_id=pattern.region_id,
            sequence_index=0,
            scan_axis=axis,
            start_pose=start,
            end_pose=end,
            center_coordinate=center,
            width=config.footprint.width_wf,
            length=length,
        )
        cover_speed = max(config.fleet.cover_speed, 1e-6)
        candidates.append(
            RegionCoveragePattern(
                pattern_id=f"{pattern.region_id}_pattern_{axis}_compressed",
                region_id=pattern.region_id,
                scan_axis=axis,
                passes=[coverage_pass],
                entry_pose=start,
                exit_pose=end,
                coverage_length=length,
                turn_length=0.0,
                turn_angle=0.0,
                total_length=length,
                estimated_time=length / cover_speed,
                max_curvature=0.0,
                feasible=True,
                metadata={
                    **pattern.metadata,
                    "compressed_short_region": "true",
                    "compressed_from_pattern": pattern.pattern_id,
                    "source_turn_ratio": f"{turn_ratio:.6f}",
                    "source_average_pass_length": f"{average_pass_length:.6f}",
                    "region_area": f"{region_area:.6f}",
                    "estimated_region_coverage_fraction": f"{coverage_fraction:.6f}",
                    "coverage_deficit": f"{max(0.0, path_config.target_coverage_fraction - coverage_fraction):.6f}",
                },
            )
        )
    return _dedupe_pattern_variants(candidates)


def _coverage_pass_bounds(passes: Sequence[CoveragePass]) -> Tuple[float, float, float, float] | None:
    if not passes:
        return None
    xs: List[float] = []
    ys: List[float] = []
    for coverage_pass in passes:
        half_width = max(coverage_pass.width / 2.0, 0.0)
        dx = coverage_pass.end_pose.x - coverage_pass.start_pose.x
        dy = coverage_pass.end_pose.y - coverage_pass.start_pose.y
        length = math.hypot(dx, dy)
        if length > 1e-9:
            ux, uy = dx / length, dy / length
        else:
            angle = _coverage_pass_scan_angle(coverage_pass)
            ux, uy = math.cos(angle), math.sin(angle)
        px, py = -uy, ux
        for pose in (coverage_pass.start_pose, coverage_pass.end_pose):
            for sign in (-1.0, 1.0):
                xs.append(pose.x + px * half_width * sign)
                ys.append(pose.y + py * half_width * sign)
    return min(xs), min(ys), max(xs), max(ys)


def _pattern_region_bounds(pattern: RegionCoveragePattern) -> Tuple[float, float, float, float] | None:
    raw = pattern.metadata.get("region_bounds")
    if raw:
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if len(parts) == 4:
            try:
                return tuple(float(part) for part in parts)  # type: ignore[return-value]
            except ValueError:
                pass
    return _coverage_pass_bounds(pattern.passes)


def _pattern_region_area(
    pattern: RegionCoveragePattern,
    fallback_bounds: Tuple[float, float, float, float] | None = None,
) -> float:
    try:
        area = float(pattern.metadata.get("region_area", "nan"))
        if math.isfinite(area) and area > 1e-9:
            return area
    except ValueError:
        pass
    bounds = fallback_bounds or _pattern_region_bounds(pattern)
    if bounds is None:
        return max(pattern.coverage_length * max(pattern.passes[0].width if pattern.passes else 1.0, 1e-6), 1e-9)
    x_min, y_min, x_max, y_max = bounds
    return max((x_max - x_min) * (y_max - y_min), 1e-9)


def _estimated_pattern_coverage_fraction(pattern: RegionCoveragePattern, config: PlannerConfig) -> float:
    try:
        value = float(pattern.metadata.get("estimated_region_coverage_fraction", "nan"))
        if math.isfinite(value):
            return max(0.0, min(1.0, value))
    except ValueError:
        pass
    width = config.footprint.width_wf
    return max(0.0, min(1.0, pattern.coverage_length * width / _pattern_region_area(pattern)))


def _annotate_pattern_coverage_quality(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> RegionCoveragePattern:
    fraction = _estimated_pattern_coverage_fraction(pattern, config)
    return replace(
        pattern,
        metadata={
            **pattern.metadata,
            "estimated_region_coverage_fraction": f"{fraction:.6f}",
            "coverage_deficit": f"{max(0.0, path_config.target_coverage_fraction - fraction):.6f}",
        },
    )


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
    angle = _coverage_pass_scan_angle(coverage_pass)
    ux, uy = math.cos(angle), math.sin(angle)
    points.sort(key=lambda item: (item[0] * ux + item[1] * uy, item[0], item[1]))
    return points[0], points[1]


def _coverage_pass_scan_angle(coverage_pass: CoveragePass) -> float:
    if coverage_pass.scan_axis == "x":
        return 0.0
    if coverage_pass.scan_axis == "y":
        return math.pi / 2.0
    if coverage_pass.scan_axis.startswith("theta:"):
        try:
            return float(coverage_pass.scan_axis.split(":", 1)[1])
        except ValueError:
            pass
    return _line_heading(coverage_pass.start_pose, coverage_pass.end_pose)


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
    region_area = _pattern_region_area(pattern)
    coverage_fraction = min(1.0, max(0.0, coverage_length * config.footprint.width_wf / max(region_area, 1e-9)))
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
            "region_area": f"{region_area:.6f}",
            "estimated_region_coverage_fraction": f"{coverage_fraction:.6f}",
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


def _pattern_sort_key(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig | None = None,
) -> Tuple[float, float, float, float, float, float, str]:
    coverage_fraction = _estimated_pattern_coverage_fraction(pattern, config)
    target = 0.98 if path_config is None else path_config.min_sweep_pattern_coverage_fraction
    coverage_deficit = max(0.0, target - coverage_fraction) if path_config is not None and _coverage_quality_priority_enabled(config, path_config) else 0.0
    endpoint_penalty = _turn_clearance_penalty(pattern.entry_pose, config) + _turn_clearance_penalty(pattern.exit_pose, config)
    quality_penalty = _pattern_quality_penalty(pattern, path_config)
    return (
        coverage_deficit,
        endpoint_penalty,
        quality_penalty,
        pattern.estimated_time,
        pattern.total_length,
        -pattern.coverage_length,
        pattern.pattern_id,
    )


def _pattern_quality_penalty(
    pattern: RegionCoveragePattern,
    path_config: PathPlanningConfig | None,
) -> float:
    if path_config is None:
        return _metadata_float(pattern.metadata, "internal_repeat_penalty", 0.0)
    retraction = (
        _metadata_float(pattern.metadata, "total_retraction_length", 0.0)
        + _metadata_float(pattern.metadata, "endpoint_total_retraction_length", 0.0)
    )
    repeat_penalty = _metadata_float(pattern.metadata, "internal_repeat_penalty", 0.0)
    return (
        max(path_config.pattern_retraction_penalty_weight, 0.0) * retraction
        + max(path_config.pattern_turn_penalty_weight, 0.0) * pattern.turn_angle
        + max(path_config.pattern_repeat_penalty_multiplier, 0.0) * repeat_penalty
    )


def _coverage_quality_priority_enabled(config: PlannerConfig, path_config: PathPlanningConfig) -> bool:
    return _large_map_mode_enabled(config, path_config)


def _accumulate_retraction_stats(stats: Dict[str, object], pattern: RegionCoveragePattern) -> None:
    if pattern.metadata.get("boundary_retraction_mode") != "adaptive":
        return
    retracted_pass_count = int(_metadata_float(pattern.metadata, "retracted_pass_count", 0.0))
    total_retraction = _metadata_float(pattern.metadata, "total_retraction_length", 0.0)
    max_retraction = _metadata_float(pattern.metadata, "max_retraction_length", 0.0)
    failed_count = int(_metadata_float(pattern.metadata, "retraction_failed_count", 0.0))
    extended_count = int(_metadata_float(pattern.metadata, "retraction_extended_count", 0.0))
    if retracted_pass_count <= 0 and total_retraction <= 1e-9 and failed_count <= 0 and extended_count <= 0:
        return
    stats["adaptive_retraction_pattern_count"] = int(stats.get("adaptive_retraction_pattern_count", 0) or 0) + 1
    stats["adaptive_retracted_pass_count"] = int(stats.get("adaptive_retracted_pass_count", 0) or 0) + retracted_pass_count
    stats["adaptive_total_retraction_length"] = float(stats.get("adaptive_total_retraction_length", 0.0) or 0.0) + total_retraction
    stats["adaptive_max_retraction_length"] = max(
        float(stats.get("adaptive_max_retraction_length", 0.0) or 0.0),
        max_retraction,
    )
    stats["adaptive_retraction_failed_count"] = int(stats.get("adaptive_retraction_failed_count", 0) or 0) + failed_count
    stats["adaptive_retraction_extended_count"] = int(stats.get("adaptive_retraction_extended_count", 0) or 0) + extended_count


def _annotate_pattern_internal_repeat(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> RegionCoveragePattern:
    if (
        not path_config.enable_main_repeat_path_penalty
        or pattern.metadata.get("open_chain_validation_only") == "true"
    ):
        return replace(
            pattern,
            metadata={
                **pattern.metadata,
                "internal_repeat_overlap_length": "0.000000",
                "internal_repeat_penalty": "0.000000",
                "internal_repeat_hit_ratio": "0.000000",
            },
        )
    segments, reason = _build_internal_sweep_segments(
        pattern,
        config,
        path_config,
        obstacle_field,
        start_time=0.0,
        segment_prefix="internal_repeat_score",
    )
    if reason:
        return pattern
    cover_segments = [segment for segment in segments if segment.kind == "cover"]
    uturn_segments = [segment for segment in segments if segment.kind != "cover"]
    score = score_repeat_overlap(
        uturn_segments,
        cover_segments,
        path_config,
        penalty_weight=path_config.internal_uturn_repeat_path_penalty_weight,
    )
    return replace(
        pattern,
        metadata={
            **pattern.metadata,
            "internal_repeat_overlap_length": f"{score.overlap_length:.6f}",
            "internal_repeat_penalty": f"{score.penalty:.6f}",
            "internal_repeat_hit_ratio": f"{score.hit_ratio:.6f}",
        },
    )


def _metadata_float(metadata: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(metadata.get(key, default))
    except (TypeError, ValueError):
        return default


def _pattern_template_key(pattern: RegionCoveragePattern) -> str:
    return f"{pattern.region_id}:{pattern.pattern_id}:{len(pattern.passes)}"


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
    uturn_cache: Dict[Tuple[object, ...], Tuple[bool, str]] | None = None,
    stats: Dict[str, object] | None = None,
) -> Tuple[bool, str]:
    _, reason = _build_internal_sweep_segments(
        pattern,
        config,
        path_config,
        obstacle_field,
        start_time=0.0,
        segment_prefix="validate",
        uturn_cache=uturn_cache,
        stats=stats,
    )
    return reason == "", reason


def _cached_internal_sweep_segments(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    start_time: float,
    segment_prefix: str,
    cache: Dict[str, Tuple[List[PathSegmentSpec], str]],
) -> Tuple[List[PathSegmentSpec], str]:
    key = _pattern_template_key(pattern)
    if key not in cache:
        segments, reason = _build_internal_sweep_segments(
            pattern,
            config,
            path_config,
            obstacle_field,
            start_time=0.0,
            segment_prefix=segment_prefix,
        )
        cache[key] = (copy.deepcopy(segments), reason)
    templates, reason = cache[key]
    if reason:
        return [], reason
    return _retime_segment_templates(templates, start_time), ""


def _retime_segment_templates(segments: Sequence[PathSegmentSpec], start_time: float) -> List[PathSegmentSpec]:
    if not segments:
        return []
    first_time = 0.0
    for segment in segments:
        if segment.waypoints and segment.waypoints[0].time is not None:
            first_time = float(segment.waypoints[0].time)
            break
    retimed = copy.deepcopy(list(segments))
    for segment in retimed:
        segment.waypoints = [
            replace(waypoint, time=start_time + (float(waypoint.time) - first_time))
            if waypoint.time is not None
            else waypoint
            for waypoint in segment.waypoints
        ]
    return retimed


def _build_internal_sweep_segments(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    start_time: float,
    segment_prefix: str,
    uturn_cache: Dict[Tuple[object, ...], Tuple[bool, str]] | None = None,
    stats: Dict[str, object] | None = None,
) -> Tuple[List[PathSegmentSpec], str]:
    segments, reason = _build_standard_internal_sweep_segments(
        pattern,
        config,
        path_config,
        obstacle_field,
        start_time=start_time,
        segment_prefix=segment_prefix,
        uturn_cache=uturn_cache,
        stats=stats,
    )
    if not reason or not path_config.enable_open_sweep_chain_tsp:
        return segments, reason
    if segment_prefix.startswith("validate"):
        chains, breaks, invalid_passes = _split_pattern_into_open_chains(
            pattern,
            config,
            path_config,
            obstacle_field,
            start_time=start_time,
            segment_prefix=segment_prefix,
            uturn_cache=uturn_cache,
            stats=stats,
            lightweight=True,
        )
        if chains:
            if stats is not None:
                stats["open_chain_region_count"] = int(stats.get("open_chain_region_count", 0) or 0) + 1
                stats["open_chain_count"] = int(stats.get("open_chain_count", 0) or 0) + len(chains)
                stats["open_chain_break_count"] = int(stats.get("open_chain_break_count", 0) or 0) + len(breaks)
                stats["open_chain_invalid_pass_count"] = int(stats.get("open_chain_invalid_pass_count", 0) or 0) + len(invalid_passes)
                stats["single_pass_chain_count"] = int(stats.get("single_pass_chain_count", 0) or 0) + sum(
                    1 for chain in chains if len(chain.passes) == 1
                )
                stats["short_pass_residual_count"] = int(stats.get("short_pass_residual_count", 0) or 0) + _short_pass_count(
                    invalid_passes, config, path_config
                )
            pattern.metadata["open_chain_mode"] = "true"
            pattern.metadata["open_chain_count"] = str(len(chains))
            pattern.metadata["open_chain_break_count"] = str(len(breaks))
            pattern.metadata["open_chain_invalid_pass_count"] = str(len(invalid_passes))
            pattern.metadata["open_chain_validation_only"] = "true"
            _copy_open_chain_rmin_metadata(pattern, chains)
            return [], ""
        return [], reason
    open_segments, open_reason, chains, breaks, invalid_passes = _build_open_chain_sweep_segments(
        pattern,
        config,
        path_config,
        obstacle_field,
        start_time=start_time,
        segment_prefix=segment_prefix,
        uturn_cache=uturn_cache,
        stats=stats,
    )
    if open_segments:
        pattern.metadata["open_chain_mode"] = "true"
        pattern.metadata["open_chain_count"] = str(len(chains))
        pattern.metadata["open_chain_break_count"] = str(len(breaks))
        pattern.metadata["open_chain_invalid_pass_count"] = str(len(invalid_passes))
        connected_chain_ids = sorted({segment.metadata.get("open_chain_id", "") for segment in open_segments if segment.metadata.get("open_chain_id")})
        pattern.metadata["open_chain_connected_count"] = str(len(connected_chain_ids))
        pattern.metadata["open_chain_order"] = ",".join(connected_chain_ids)
        pattern.metadata["open_chain_recovered_coverage_length"] = f"{sum(segment.length for segment in open_segments if segment.kind == 'cover'):.6f}"
        _copy_open_chain_rmin_metadata(pattern, chains)
        return open_segments, ""
    return [], open_reason or reason


def _copy_open_chain_rmin_metadata(
    pattern: RegionCoveragePattern,
    chains: Sequence[OpenSweepChain],
) -> None:
    if not chains:
        return
    first_metadata = chains[0].metadata
    for key in ("chain_order_mode", "turn_stride", "turn_stride_raw", "required_turn_span", "delta"):
        value = first_metadata.get(key)
        if value is not None:
            pattern.metadata[key] = value


def _build_standard_internal_sweep_segments(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    start_time: float,
    segment_prefix: str,
    uturn_cache: Dict[Tuple[object, ...], Tuple[bool, str]] | None = None,
    stats: Dict[str, object] | None = None,
) -> Tuple[List[PathSegmentSpec], str]:
    segments: List[PathSegmentSpec] = []
    current_time = start_time
    use_cache = (
        path_config.enable_uturn_validation_cache
        and uturn_cache is not None
        and segment_prefix.startswith("validate")
    )
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
        cache_key = _uturn_validation_cache_key(coverage_pass.end_pose, next_pass.start_pose, config, path_config, obstacle_field)
        if use_cache and cache_key in uturn_cache:
            if stats is not None:
                stats["uturn_cache_hit_count"] = int(stats.get("uturn_cache_hit_count", 0) or 0) + 1
            cached_valid, cached_reason = uturn_cache[cache_key]
            if not cached_valid:
                return [], cached_reason
            current_time += _estimated_uturn_duration(coverage_pass.end_pose, next_pass.start_pose, config)
            continue
        if use_cache and stats is not None:
            stats["uturn_cache_miss_count"] = int(stats.get("uturn_cache_miss_count", 0) or 0) + 1
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
                reason = f"uturn_invalid:{','.join(reasons)}"
                if use_cache:
                    uturn_cache[cache_key] = (False, reason)
                return [], reason
            if stats is not None:
                stats["uturn_direct_fail_repair_success_count"] = int(
                    stats.get("uturn_direct_fail_repair_success_count", 0) or 0
                ) + 1
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
            if use_cache:
                uturn_cache[cache_key] = (True, "")
            continue
        turn_report = validate_transition_dynamics(turn, config, obstacle_field=obstacle_field, retime=True)
        if not turn_report.valid:
            reason = f"uturn_dynamic_violation:{','.join(turn_report.reasons)}"
            if use_cache:
                uturn_cache[cache_key] = (False, reason)
            return [], reason
        segments.append(turn)
        current_time = _segment_end_time(turn)
        if use_cache:
            uturn_cache[cache_key] = (True, "")
    return segments, ""


def _build_open_chain_sweep_segments(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    start_time: float,
    segment_prefix: str,
    uturn_cache: Dict[Tuple[object, ...], Tuple[bool, str]] | None = None,
    stats: Dict[str, object] | None = None,
) -> Tuple[List[PathSegmentSpec], str, List[OpenSweepChain], List[OpenSweepBreak], List[CoveragePass]]:
    chains, breaks, invalid_passes = _split_pattern_into_open_chains(
        pattern,
        config,
        path_config,
        obstacle_field,
        start_time=start_time,
        segment_prefix=segment_prefix,
        uturn_cache=uturn_cache,
        stats=stats,
        lightweight=False,
    )
    if stats is not None:
        stats["open_chain_count"] = int(stats.get("open_chain_count", 0) or 0) + len(chains)
        stats["open_chain_break_count"] = int(stats.get("open_chain_break_count", 0) or 0) + len(breaks)
        stats["open_chain_invalid_pass_count"] = int(stats.get("open_chain_invalid_pass_count", 0) or 0) + len(invalid_passes)
        stats["single_pass_chain_count"] = int(stats.get("single_pass_chain_count", 0) or 0) + sum(1 for chain in chains if len(chain.passes) == 1)
        stats["short_pass_residual_count"] = int(stats.get("short_pass_residual_count", 0) or 0) + _short_pass_count(
            invalid_passes, config, path_config
        )
    if not chains:
        return [], "open_chain_failed:no_valid_chains", chains, breaks, invalid_passes
    if len(chains) > max(int(path_config.max_open_chains_per_region), 1):
        return [], "open_chain_failed:too_many_chains", chains, breaks, invalid_passes
    segments, reason, connected_chain_ids = _assemble_open_chains_greedy(
        pattern,
        chains,
        config,
        path_config,
        obstacle_field,
        start_time=start_time,
        segment_prefix=segment_prefix,
    )
    if segments:
        if stats is not None:
            stats["open_chain_region_count"] = int(stats.get("open_chain_region_count", 0) or 0) + 1
            stats["open_chain_connected_count"] = int(stats.get("open_chain_connected_count", 0) or 0) + len(connected_chain_ids)
            stats["open_chain_skipped_count"] = int(stats.get("open_chain_skipped_count", 0) or 0) + max(len(chains) - len(connected_chain_ids), 0)
            stats["open_chain_recovered_coverage_length"] = float(stats.get("open_chain_recovered_coverage_length", 0.0) or 0.0) + sum(
                segment.length for segment in segments if segment.kind == "cover"
            )
        return segments, "", chains, breaks, invalid_passes
    return [], reason or "open_chain_failed:no_connectable_chain_order", chains, breaks, invalid_passes


def _split_pattern_into_open_chains(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    start_time: float,
    segment_prefix: str,
    uturn_cache: Dict[Tuple[object, ...], Tuple[bool, str]] | None,
    stats: Dict[str, object] | None,
    lightweight: bool,
) -> Tuple[List[OpenSweepChain], List[OpenSweepBreak], List[CoveragePass]]:
    rmin_groups, rmin_metadata = _rmin_aware_pass_groups(pattern, config, path_config, stats)
    if rmin_groups:
        chains: List[OpenSweepChain] = []
        breaks: List[OpenSweepBreak] = []
        invalid_passes: List[CoveragePass] = []
        for group_index, group in enumerate(rmin_groups):
            group_metadata = {
                **rmin_metadata,
                "rmin_group_index": str(group_index),
                "rmin_group_count": str(len(rmin_groups)),
            }
            group_chains, group_breaks, group_invalid = _split_ordered_passes_into_open_chains(
                pattern=pattern,
                ordered_passes=group,
                config=config,
                path_config=path_config,
                obstacle_field=obstacle_field,
                start_time=start_time,
                segment_prefix=f"{segment_prefix}_rmin_group_{group_index}",
                uturn_cache=uturn_cache,
                stats=stats,
                lightweight=lightweight,
                chain_index_offset=len(chains),
                metadata_extra=group_metadata,
                count_rmin_uturns=True,
            )
            chains.extend(group_chains)
            breaks.extend(group_breaks)
            invalid_passes.extend(group_invalid)
        if chains or invalid_passes:
            return chains, breaks, invalid_passes

    return _split_ordered_passes_into_open_chains(
        pattern=pattern,
        ordered_passes=list(enumerate(pattern.passes)),
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        start_time=start_time,
        segment_prefix=segment_prefix,
        uturn_cache=uturn_cache,
        stats=stats,
        lightweight=lightweight,
        chain_index_offset=0,
        metadata_extra={"chain_order_mode": "sequential"},
        count_rmin_uturns=False,
    )


def _split_ordered_passes_into_open_chains(
    pattern: RegionCoveragePattern,
    ordered_passes: Sequence[Tuple[int, CoveragePass]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    start_time: float,
    segment_prefix: str,
    uturn_cache: Dict[Tuple[object, ...], Tuple[bool, str]] | None,
    stats: Dict[str, object] | None,
    lightweight: bool,
    chain_index_offset: int,
    metadata_extra: Dict[str, str] | None = None,
    count_rmin_uturns: bool = False,
) -> Tuple[List[OpenSweepChain], List[OpenSweepBreak], List[CoveragePass]]:
    chains: List[OpenSweepChain] = []
    breaks: List[OpenSweepBreak] = []
    invalid_passes: List[CoveragePass] = []
    current_passes: List[CoveragePass] = []
    current_indices: List[int] = []
    left_break_reason = ""
    previous_valid_pass: CoveragePass | None = None

    def close_current(right_reason: str = "") -> None:
        nonlocal current_passes, current_indices, left_break_reason
        if not current_passes:
            left_break_reason = right_reason
            return
        chain_index = len(chains)
        chain = _make_open_sweep_chain(
            pattern=pattern,
            chain_index=chain_index_offset + chain_index,
            passes=current_passes,
            pass_indices=current_indices,
            left_break_reason=left_break_reason,
            right_break_reason=right_reason,
            config=config,
            path_config=path_config,
            obstacle_field=obstacle_field,
            start_time=start_time,
            segment_prefix=f"{segment_prefix}_open_chain_{chain_index_offset + chain_index}",
            lightweight=lightweight,
            metadata_extra=metadata_extra,
        )
        if chain.feasible:
            chains.append(chain)
        else:
            invalid_passes.extend(current_passes)
        current_passes = []
        current_indices = []
        left_break_reason = right_reason

    for pass_idx, coverage_pass in ordered_passes:
        cover = _build_cover_for_pass(
            coverage_pass,
            segment_id=f"{segment_prefix}_open_probe_{coverage_pass.pass_id}",
            start_time=start_time,
            reverse=False,
            config=config,
        )
        cover_reasons = path_segment_invalid_reasons(cover, config, obstacle_field)
        if cover_reasons:
            reason = f"cover_invalid:{','.join(cover_reasons)}"
            close_current(reason)
            invalid_passes.append(coverage_pass)
            breaks.append(
                OpenSweepBreak(
                    region_id=pattern.region_id,
                    pattern_id=pattern.pattern_id,
                    before_pass_id=previous_valid_pass.pass_id if previous_valid_pass else None,
                    after_pass_id=coverage_pass.pass_id,
                    reason=reason,
                    direct_reasons=list(cover_reasons),
                )
            )
            previous_valid_pass = None
            continue

        if current_passes and previous_valid_pass is not None:
            uturn_segments, reason, detail = _build_local_uturn_segments(
                segment_id=f"{segment_prefix}_open_probe_{previous_valid_pass.pass_id}_to_{coverage_pass.pass_id}",
                start=previous_valid_pass.end_pose,
                end=coverage_pass.start_pose,
                start_time=start_time,
                config=config,
                path_config=path_config,
                obstacle_field=obstacle_field,
                uturn_cache=uturn_cache,
                stats=stats,
                allow_repair=not lightweight,
            )
            if count_rmin_uturns and stats is not None:
                stats["rmin_180_attempt_count"] = int(stats.get("rmin_180_attempt_count", 0) or 0) + 1
                if uturn_segments:
                    stats["rmin_180_success_count"] = int(stats.get("rmin_180_success_count", 0) or 0) + 1
            if not uturn_segments:
                close_current(reason)
                breaks.append(
                    OpenSweepBreak(
                        region_id=pattern.region_id,
                        pattern_id=pattern.pattern_id,
                        before_pass_id=previous_valid_pass.pass_id,
                        after_pass_id=coverage_pass.pass_id,
                        reason=reason,
                        direct_reasons=list(detail.get("direct_reasons", [])),
                        repair_attempted=bool(detail.get("repair_attempted", False)),
                        repair_success=bool(detail.get("repair_success", False)),
                        rejected_connector_sources=list(detail.get("rejected_connector_sources", [])),
                    )
                )

        current_passes.append(coverage_pass)
        current_indices.append(pass_idx)
        previous_valid_pass = coverage_pass

    close_current("")
    return chains, breaks, invalid_passes


def _rmin_aware_pass_groups(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    stats: Dict[str, object] | None,
) -> Tuple[List[List[Tuple[int, CoveragePass]]] | None, Dict[str, str]]:
    if (
        not path_config.enable_rmin_aware_chain_order
        or path_config.chain_turn_strategy != "rmin_180"
        or len(pattern.passes) < 2
    ):
        return None, {}
    delta = _coverage_strip_spacing(config, path_config)
    if delta <= 1e-9:
        return None, {}
    required_span = _rmin_aware_required_turn_span(config, path_config)
    raw_stride = int(math.ceil(required_span / delta))
    max_stride = max(int(path_config.rmin_chain_max_stride), 1)
    stride = max(1, min(raw_stride, max_stride))
    if stride <= 1:
        return None, {}

    groups: List[List[Tuple[int, CoveragePass]]] = []
    for offset in range(stride):
        group = [(idx, coverage_pass) for idx, coverage_pass in enumerate(pattern.passes) if idx % stride == offset]
        if group:
            groups.append(group)
    if len(groups) <= 1:
        return None, {}

    if stats is not None:
        stats["rmin_aware_chain_enabled"] = True
        distribution = dict(stats.get("turn_stride_distribution", {}) or {})
        key = str(stride)
        distribution[key] = int(distribution.get(key, 0)) + 1
        stats["turn_stride_distribution"] = distribution

    metadata = {
        "chain_order_mode": "rmin_stride",
        "turn_stride": str(stride),
        "turn_stride_raw": str(raw_stride),
        "required_turn_span": f"{required_span:.6f}",
        "delta": f"{delta:.6f}",
    }
    return groups, metadata


def _coverage_strip_spacing(config: PlannerConfig, path_config: PathPlanningConfig) -> float:
    overlap = path_config.overlap_ratio if path_config.overlap_ratio is not None else config.mission.overlap_ratio
    return max(config.footprint.width_wf * (1.0 - overlap), 0.0)


def _rmin_aware_required_turn_span(config: PlannerConfig, path_config: PathPlanningConfig) -> float:
    clearance = max(config.footprint.width_wf * max(path_config.rmin_chain_turn_clearance_factor, 0.0), config.safety.d_safe)
    return max(2.0 * config.fleet.min_turn_radius + clearance, 0.0)


def _short_pass_count(
    passes: Sequence[CoveragePass],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> int:
    threshold = max(config.footprint.length_lf * max(path_config.rmin_chain_min_pass_length_factor, 0.0), 1e-9)
    return sum(1 for coverage_pass in passes if coverage_pass.length < threshold)


def _make_open_sweep_chain(
    pattern: RegionCoveragePattern,
    chain_index: int,
    passes: Sequence[CoveragePass],
    pass_indices: Sequence[int],
    left_break_reason: str,
    right_break_reason: str,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    start_time: float,
    segment_prefix: str,
    lightweight: bool,
    metadata_extra: Dict[str, str] | None = None,
) -> OpenSweepChain:
    pass_list = list(passes)
    if lightweight:
        segments: List[PathSegmentSpec] = []
        reason = ""
    else:
        segments, reason = _build_chain_direction_segments(
            pass_list,
            config,
            path_config,
            obstacle_field,
            start_time=start_time,
            segment_prefix=segment_prefix,
            reverse=False,
        )
    entry_pose = pass_list[0].start_pose
    exit_pose = pass_list[-1].end_pose
    reverse_entry = _reverse_pose(pass_list[-1].end_pose)
    reverse_exit = _reverse_pose(pass_list[0].start_pose)
    metadata = {
        "open_chain_mode": "true",
        "pass_count": str(len(pass_list)),
        "left_break_reason": left_break_reason,
        "right_break_reason": right_break_reason,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    chain = OpenSweepChain(
        chain_id=f"{pattern.pattern_id}_chain_{chain_index}",
        region_id=pattern.region_id,
        pattern_id=pattern.pattern_id,
        pass_indices=list(pass_indices),
        passes=pass_list,
        entry_pose=entry_pose,
        exit_pose=exit_pose,
        reverse_entry_pose=reverse_entry,
        reverse_exit_pose=reverse_exit,
        internal_segments=segments,
        coverage_length=sum(item.length for item in pass_list),
        internal_turn_length=sum(segment.length for segment in segments if segment.kind != "cover"),
        estimated_time=_segment_duration_total(segments),
        max_curvature=max((segment.curvature_max for segment in segments), default=0.0),
        feasible=not reason,
        left_break_reason=left_break_reason,
        right_break_reason=right_break_reason,
        metadata=metadata,
    )
    return chain


def _assemble_open_chains_greedy(
    pattern: RegionCoveragePattern,
    chains: Sequence[OpenSweepChain],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    start_time: float,
    segment_prefix: str,
) -> Tuple[List[PathSegmentSpec], str, List[str]]:
    remaining = list(chains)
    current_pose = pattern.entry_pose
    current_time = start_time
    segments: List[PathSegmentSpec] = []
    connected: List[str] = []
    serial = 0
    failure_reasons: List[str] = []

    while remaining:
        choices: List[Tuple[float, OpenSweepChain, bool, List[PathSegmentSpec], List[PathSegmentSpec], Pose2D]] = []
        for chain in remaining:
            for reverse in (False, True):
                chain_entry = chain.reverse_entry_pose if reverse else chain.entry_pose
                connector = _build_open_chain_connector(
                    segment_id=f"{segment_prefix}_open_chain_connector_{serial}_{chain.chain_id}_{'rev' if reverse else 'fwd'}",
                    start=current_pose,
                    end=chain_entry,
                    start_time=current_time,
                    config=config,
                    path_config=path_config,
                    obstacle_field=obstacle_field,
                )
                if connector is None:
                    failure_reasons.append(f"{chain.chain_id}:connector_failed")
                    continue
                connector_end_time = _segment_end_time(connector[-1]) if connector else current_time
                chain_segments, chain_reason = _build_chain_direction_segments(
                    chain.passes,
                    config,
                    path_config,
                    obstacle_field,
                    start_time=connector_end_time,
                    segment_prefix=f"{segment_prefix}_open_chain_{serial}_{chain.chain_id}_{'rev' if reverse else 'fwd'}",
                    reverse=reverse,
                )
                if chain_reason:
                    failure_reasons.append(f"{chain.chain_id}:{chain_reason}")
                    continue
                score = (
                    path_config.open_chain_connector_penalty_weight * sum(segment.length for segment in connector)
                    + _segment_duration_total(connector)
                    + _segment_duration_total(chain_segments)
                    - path_config.open_chain_coverage_reward_weight * chain.coverage_length
                )
                chain_exit = chain.reverse_exit_pose if reverse else chain.exit_pose
                choices.append((score, chain, reverse, connector, chain_segments, chain_exit))
        if not choices:
            break
        choices.sort(key=lambda item: (item[0], item[1].chain_id, item[2]))
        _, selected, reverse, connector, chain_segments, chain_exit = choices[0]
        for segment in connector:
            segment.metadata.update(
                {
                    "open_chain_connector": "true",
                    "region_id": pattern.region_id,
                    "pattern_id": pattern.pattern_id,
                    "open_chain_to": selected.chain_id,
                    "chain_order_mode": selected.metadata.get("chain_order_mode", ""),
                    "turn_stride": selected.metadata.get("turn_stride", ""),
                }
            )
        for segment in chain_segments:
            segment.metadata.update(
                {
                    "open_chain_mode": "true",
                    "open_chain_id": selected.chain_id,
                    "open_chain_direction": "reverse" if reverse else "forward",
                    "region_id": pattern.region_id,
                    "pattern_id": pattern.pattern_id,
                    "chain_order_mode": selected.metadata.get("chain_order_mode", ""),
                    "turn_stride": selected.metadata.get("turn_stride", ""),
                }
            )
        segments.extend(connector)
        segments.extend(chain_segments)
        connected.append(selected.chain_id)
        remaining = [item for item in remaining if item.chain_id != selected.chain_id]
        current_pose = chain_exit
        current_time = _segment_end_time(chain_segments[-1])
        serial += len(connector) + len(chain_segments)

    if not connected:
        return [], ",".join(failure_reasons[:6]) or "open_chain_no_connected_chains", connected
    final_connector = _build_open_chain_connector(
        segment_id=f"{segment_prefix}_open_chain_exit_to_pattern",
        start=current_pose,
        end=pattern.exit_pose,
        start_time=current_time,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
    )
    if final_connector is None:
        return [], "open_chain_exit_connector_failed", connected
    for segment in final_connector:
        segment.metadata.update(
            {
                "open_chain_connector": "true",
                "open_chain_exit_connector": "true",
                "region_id": pattern.region_id,
                "pattern_id": pattern.pattern_id,
            }
        )
    segments.extend(final_connector)
    if not _segments_strictly_valid(segments, config, obstacle_field):
        return [], "open_chain_sequence_dynamic_validation_failed", connected
    return [segment for segment in segments if segment.length > 1e-9], "", connected


def _build_chain_direction_segments(
    passes: Sequence[CoveragePass],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    start_time: float,
    segment_prefix: str,
    reverse: bool,
) -> Tuple[List[PathSegmentSpec], str]:
    ordered = list(reversed(passes)) if reverse else list(passes)
    segments: List[PathSegmentSpec] = []
    current_time = start_time
    previous_end: Pose2D | None = None
    for idx, coverage_pass in enumerate(ordered):
        cover = _build_cover_for_pass(
            coverage_pass,
            segment_id=f"{segment_prefix}_{coverage_pass.pass_id}",
            start_time=current_time,
            reverse=reverse,
            config=config,
        )
        reasons = path_segment_invalid_reasons(cover, config, obstacle_field)
        if reasons:
            return [], f"cover_invalid:{','.join(reasons)}"
        if previous_end is not None:
            uturn, reason, _ = _build_local_uturn_segments(
                segment_id=f"{segment_prefix}_{ordered[idx - 1].pass_id}_to_{coverage_pass.pass_id}",
                start=previous_end,
                end=cover.waypoints[0] and Pose2D(cover.waypoints[0].x, cover.waypoints[0].y, cover.waypoints[0].psi),
                start_time=current_time,
                config=config,
                path_config=path_config,
                obstacle_field=obstacle_field,
                uturn_cache=None,
                stats=None,
                allow_repair=True,
            )
            if not uturn:
                return [], reason
            segments.extend(uturn)
            current_time = _segment_end_time(uturn[-1])
            cover = _build_cover_for_pass(
                coverage_pass,
                segment_id=f"{segment_prefix}_{coverage_pass.pass_id}",
                start_time=current_time,
                reverse=reverse,
                config=config,
            )
        cover.metadata.update({"open_chain_cover": "true", "pass_id": coverage_pass.pass_id})
        segments.append(cover)
        current_time = _segment_end_time(cover)
        last = cover.waypoints[-1]
        previous_end = Pose2D(last.x, last.y, last.psi)
    return segments, ""


def _build_cover_for_pass(
    coverage_pass: CoveragePass,
    segment_id: str,
    start_time: float,
    reverse: bool,
    config: PlannerConfig,
) -> PathSegmentSpec:
    start = _reverse_pose(coverage_pass.end_pose) if reverse else coverage_pass.start_pose
    end = _reverse_pose(coverage_pass.start_pose) if reverse else coverage_pass.end_pose
    return build_cover_segment(
        segment_id=segment_id,
        start=start,
        end=end,
        start_time=start_time,
        speed=max(config.fleet.cover_speed, 1e-6),
    )


def _build_local_uturn_segments(
    segment_id: str,
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    uturn_cache: Dict[Tuple[object, ...], Tuple[bool, str]] | None,
    stats: Dict[str, object] | None,
    allow_repair: bool = True,
) -> Tuple[List[PathSegmentSpec] | None, str, Dict[str, object]]:
    detail: Dict[str, object] = {"direct_reasons": [], "repair_attempted": False, "repair_success": False, "rejected_connector_sources": []}
    cache_key = _uturn_validation_cache_key(start, end, config, path_config, obstacle_field)
    use_cache = path_config.enable_uturn_validation_cache and uturn_cache is not None
    if use_cache and cache_key in uturn_cache:
        if stats is not None:
            stats["uturn_cache_hit_count"] = int(stats.get("uturn_cache_hit_count", 0) or 0) + 1
        cached_valid, cached_reason = uturn_cache[cache_key]
        if not cached_valid:
            return None, cached_reason, detail
    elif use_cache and stats is not None:
        stats["uturn_cache_miss_count"] = int(stats.get("uturn_cache_miss_count", 0) or 0) + 1

    direct = build_transition_segment(
        segment_id=segment_id,
        start=start,
        end=end,
        start_time=start_time,
        config=config,
        kind="turn",
        sample_count=32,
        use_bezier=path_config.use_bezier_smoothing,
    )
    direct.metadata.update({"internal_uturn": "true", "open_chain_probe": "true"})
    direct_reasons = path_segment_invalid_reasons(direct, config, obstacle_field)
    detail["direct_reasons"] = list(direct_reasons)
    if not direct_reasons:
        report = validate_transition_dynamics(direct, config, obstacle_field=obstacle_field, retime=True)
        if report.valid:
            local_reason = _local_uturn_length_reason([direct], start, end, config, path_config)
            if not local_reason:
                if use_cache:
                    uturn_cache[cache_key] = (True, "")
                return [direct], "", detail
            if use_cache:
                uturn_cache[cache_key] = (False, local_reason)
            return None, local_reason, detail
        direct_reasons = list(report.reasons)
        detail["direct_reasons"] = list(direct_reasons)

    if not allow_repair:
        reason = f"uturn_invalid:{','.join(direct_reasons or ['direct_invalid'])}"
        if use_cache:
            uturn_cache[cache_key] = (False, reason)
        return None, reason, detail

    detail["repair_attempted"] = True
    repaired = build_obstacle_aware_transition_segments(
        segment_id=f"{segment_id}_repair",
        start=start,
        end=end,
        start_time=start_time,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        kind="turn",
        sample_count=48,
    )
    detail["rejected_connector_sources"] = [segment.path_source for segment in repaired]
    if _segments_strictly_valid(repaired, config, obstacle_field):
        detail["repair_success"] = True
        local_reason = _local_uturn_length_reason(repaired, start, end, config, path_config)
        if not local_reason:
            for segment in repaired:
                segment.metadata.update({"internal_uturn": "true", "open_chain_probe": "true", "uturn_repair": "true"})
            if stats is not None:
                stats["uturn_direct_fail_repair_success_count"] = int(stats.get("uturn_direct_fail_repair_success_count", 0) or 0) + 1
            if use_cache:
                uturn_cache[cache_key] = (True, "")
            return repaired, "", detail
        if use_cache:
            uturn_cache[cache_key] = (False, local_reason)
        return None, local_reason, detail

    reasons = sorted(
        {
            reason
            for segment in repaired
            for reason in (segment.metadata.get("dynamic_invalid_reasons", "") or ",".join(path_segment_invalid_reasons(segment, config, obstacle_field))).split(",")
            if reason
        }
    )
    reason = f"uturn_invalid:{','.join(reasons or direct_reasons or ['dynamic_validation_failed'])}"
    if use_cache:
        uturn_cache[cache_key] = (False, reason)
    return None, reason, detail


def _build_open_chain_connector(
    segment_id: str,
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> List[PathSegmentSpec] | None:
    if math.hypot(start.x - end.x, start.y - end.y) <= 1e-6 and abs(wrap_angle(start.psi - end.psi)) <= 1e-4:
        return []
    segments = build_obstacle_aware_transition_segments(
        segment_id=segment_id,
        start=start,
        end=end,
        start_time=start_time,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        kind="turn",
        sample_count=48,
    )
    if not _segments_strictly_valid(segments, config, obstacle_field):
        return None
    return [segment for segment in segments if segment.length > 1e-9]


def _local_uturn_length_reason(
    segments: Sequence[PathSegmentSpec],
    start: Pose2D,
    end: Pose2D,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> str:
    length = sum(segment.length for segment in segments)
    distance = math.hypot(end.x - start.x, end.y - start.y)
    overlap = path_config.overlap_ratio if path_config.overlap_ratio is not None else config.mission.overlap_ratio
    delta = config.footprint.width_wf * (1.0 - overlap)
    limit = max(math.pi * config.fleet.min_turn_radius + 2.5 * delta, 3.0 * distance)
    return "long_bridge_available" if length > limit + 1e-9 else ""


def _reverse_pose(pose: Pose2D) -> Pose2D:
    return Pose2D(pose.x, pose.y, wrap_angle(pose.psi + math.pi))


def _segment_duration_total(segments: Sequence[PathSegmentSpec]) -> float:
    total = 0.0
    for segment in segments:
        if len(segment.waypoints) >= 2:
            total += max((segment.waypoints[-1].time or 0.0) - (segment.waypoints[0].time or 0.0), 0.0)
    return total


def _uturn_validation_cache_key(
    start: Pose2D,
    end: Pose2D,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> Tuple[object, ...]:
    xy_quantum = max(min(config.fleet.min_turn_radius * 0.25, config.footprint.width_wf * 0.25), 0.1)
    heading_quantum = math.pi / 32.0
    return (
        _quantized_pose_key(start, xy_quantum, heading_quantum),
        _quantized_pose_key(end, xy_quantum, heading_quantum),
        round(config.fleet.min_turn_radius, 4),
        bool(path_config.use_bezier_smoothing),
        _obstacle_field_signature(obstacle_field),
    )


def _quantized_pose_key(pose: Pose2D, xy_quantum: float, heading_quantum: float) -> Tuple[int, int, int]:
    return (
        int(round(pose.x / max(xy_quantum, 1e-9))),
        int(round(pose.y / max(xy_quantum, 1e-9))),
        int(round(wrap_angle(pose.psi) / max(heading_quantum, 1e-9))),
    )


def _obstacle_field_signature(obstacle_field: ObstacleField | None) -> Tuple[object, ...]:
    if obstacle_field is None:
        return ("none",)
    obstacle_keys = []
    for obstacle in obstacle_field.inflated_obstacles:
        xs = [point[0] for point in obstacle.polygon]
        ys = [point[1] for point in obstacle.polygon]
        obstacle_keys.append((obstacle.obstacle_id, round(min(xs, default=0.0), 3), round(min(ys, default=0.0), 3), round(max(xs, default=0.0), 3), round(max(ys, default=0.0), 3)))
    return (round(obstacle_field.safety_margin, 3), round(obstacle_field.footprint_margin, 3), tuple(obstacle_keys))


def _estimated_uturn_duration(start: Pose2D, end: Pose2D, config: PlannerConfig) -> float:
    length = _transition_length(start, end, config)
    return length / max(config.fleet.turn_speed_max, 1e-6)


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
        chain_order=[
            item.strip()
            for item in pattern.metadata.get("open_chain_order", "").split(",")
            if item.strip()
        ],
        metadata={
            "endpoint_count": str(len(endpoints)),
            "open_chain_mode": pattern.metadata.get("open_chain_mode", "false"),
            "open_chain_count": pattern.metadata.get("open_chain_count", "0"),
            "open_chain_connected_count": pattern.metadata.get("open_chain_connected_count", "0"),
            "chain_order_mode": pattern.metadata.get("chain_order_mode", ""),
            "turn_stride": pattern.metadata.get("turn_stride", ""),
            "required_turn_span": pattern.metadata.get("required_turn_span", ""),
            "delta": pattern.metadata.get("delta", ""),
        },
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


def _prioritized_candidate_slice(
    ordered_candidates: Sequence[Tuple],
    branch_limit: int,
    per_region_limit: int,
    prioritize_region_execution: bool,
    region_index: int,
) -> List[Tuple]:
    if not prioritize_region_execution:
        return list(ordered_candidates[: max(branch_limit, 1)])
    if not ordered_candidates:
        return []
    per_region_limit = max(int(per_region_limit), 1)
    region_ids = {str(item[region_index]) for item in ordered_candidates}
    target = min(
        len(ordered_candidates),
        max(max(branch_limit, 1), len(region_ids) * min(per_region_limit, 3)),
    )
    selected: List[Tuple] = []
    selected_keys: set[Tuple[str, str]] = set()
    per_region_counts: Dict[str, int] = {}
    for item in ordered_candidates:
        region_id = str(item[region_index])
        pattern = item[-1]
        pattern_id = str(getattr(pattern, "pattern_id", ""))
        if per_region_counts.get(region_id, 0) >= per_region_limit:
            continue
        selected.append(item)
        selected_keys.add((region_id, pattern_id))
        per_region_counts[region_id] = per_region_counts.get(region_id, 0) + 1
        if len(selected) >= target:
            break
    minimum = min(max(branch_limit, 1), len(ordered_candidates))
    if len(selected) < minimum:
        for item in ordered_candidates:
            region_id = str(item[region_index])
            pattern_id = str(getattr(item[-1], "pattern_id", ""))
            key = (region_id, pattern_id)
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)
            if len(selected) >= minimum:
                break
    return selected


def _record_connector_failure(
    failure_map: Dict[str, set[str]],
    region_id: str,
    reason: str,
) -> None:
    failure_map.setdefault(region_id, set()).add(reason or "connector_failed")


def _connector_failure_lists(failure_map: Dict[str, set[str]]) -> Dict[str, List[str]]:
    return {region_id: sorted(reasons) for region_id, reasons in sorted(failure_map.items())}


def _connector_failure_strings(failure_map: Dict[str, set[str]]) -> Dict[str, str]:
    return {region_id: ",".join(sorted(reasons)) for region_id, reasons in sorted(failure_map.items())}


def _connector_failures_from_edges(edges: Sequence[Dict[str, object]]) -> Dict[str, set[str]]:
    failures: Dict[str, set[str]] = {}
    for edge in edges:
        region_id = str(edge.get("region_id") or edge.get("to_region") or "")
        if not region_id:
            continue
        _record_connector_failure(failures, region_id, str(edge.get("reason", "connector_failed")))
    return failures


def _solve_agent_region_tsp(
    agent_id: int,
    region_ids: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    sweep_paths: Dict[str, RegionSweepPath],
    sweep_segment_templates: Dict[str, Tuple[List[PathSegmentSpec], str]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    ownership_map: CoverageOwnershipMap | None,
) -> Dict[str, object]:
    start_pose = config.fleet.initial_states_3dof[agent_id].pose()
    if _large_map_mode_enabled(config, path_config):
        initial_order = list(region_ids)
    else:
        initial_order = _nearest_neighbor_region_order(start_pose, region_ids, patterns, config)
    requested_solver = validate_tsp_solver(path_config.tsp_solver)
    fallback_solver_metadata: Dict[str, object] | None = None
    if requested_solver != "deterministic":
        aco_region_limit = _large_map_aco_region_limit(path_config)
        if _large_map_mode_enabled(config, path_config) and len(region_ids) > aco_region_limit:
            fallback_solver_metadata = {
                "requested_tsp_solver": requested_solver,
                "effective_tsp_solver": "deterministic_fallback",
                "tsp_solver_status": "failed",
                "failure_reason": "large_map_region_count_exceeds_aco_limit",
                "assigned_region_count": len(region_ids),
                "large_map_aco_region_limit": aco_region_limit,
                "aco_best_objective": None,
                "aco_initial_objective": None,
                "aco_iteration_count": 0,
                "aco_convergence_trace": [],
                "aco_accepted_3opt_count": 0,
            }
        else:
            aco_result = _solve_agent_region_tsp_aco(
                agent_id,
                region_ids,
                patterns,
                config,
                path_config,
                obstacle_field,
                initial_order,
                ownership_map,
            )
            if aco_result["tsp_solver_metadata"]["tsp_solver_status"] == "success":
                return aco_result
            fallback_solver_metadata = dict(aco_result["tsp_solver_metadata"])
    if _large_map_mode_enabled(config, path_config):
        return _solve_agent_region_tsp_large_map_greedy(
            agent_id,
            initial_order,
            patterns,
            config,
            path_config,
            obstacle_field,
            ownership_map,
            fallback_solver_metadata,
            sweep_segment_templates,
        )
    candidate_pattern_counts = {region_id: len(patterns.get(region_id, [])) for region_id in region_ids}
    sweep_segment_cache: Dict[str, Tuple[List[PathSegmentSpec], str]] = {
        key: (copy.deepcopy(value[0]), value[1])
        for key, value in sweep_segment_templates.items()
    }
    connector_cache: Dict[Tuple[float, float, float, float, float, float, float, str, bool], Tuple[List[PathSegmentSpec] | None, List[Dict[str, object]]]] = {}
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
        "repeat_overlap_length": 0.0,
        "repeat_penalty": 0.0,
        "cross_agent_overlap_length": 0.0,
        "cross_agent_penalty": 0.0,
        "unavoidable_cross_agent_overlap_count": 0,
        "executed_coverage_length": 0.0,
    }
    beam = [initial_state]
    best_partial = initial_state
    complete_states: List[Dict[str, object]] = []
    terminal_rejections: List[Dict[str, object]] = []
    use_coverage_priority = _coverage_quality_priority_enabled(config, path_config)

    for _ in range(len(initial_order)):
        next_beam: List[Dict[str, object]] = []
        depth_rejections: List[Dict[str, object]] = []
        for state in beam:
            state_entries: List[Dict[str, object]] = []
            state_has_zero_cross_agent_overlap = False
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
                    coverage_deficit = _metadata_float(
                        pattern.metadata,
                        "coverage_deficit",
                        max(0.0, path_config.target_coverage_fraction - _estimated_pattern_coverage_fraction(pattern, config)),
                    ) if use_coverage_priority else 0.0
                    ordered_candidates.append(
                        (
                            _transition_length(current_pose, pattern.entry_pose, config)
                            + pattern.total_length
                            + path_config.coverage_priority_weight * coverage_deficit
                            + _pattern_quality_penalty(pattern, path_config)
                            + _turn_clearance_penalty(pattern.entry_pose, config)
                            + _turn_clearance_penalty(pattern.exit_pose, config)
                            - coverage_reward,
                            region_id,
                            pattern,
                        )
                    )
            ordered_candidates.sort(key=lambda item: (item[0], item[1], item[2].pattern_id))
            candidate_slice = _prioritized_candidate_slice(
                ordered_candidates,
                branch_limit,
                _connector_pattern_limit(path_config),
                bool(path_config.prioritize_region_execution),
                region_index=1,
            )
            obstacle_aware_retry_limit = 0
            obstacle_aware_retry_count = 0
            for _, region_id, candidate_pattern in candidate_slice:
                candidate_attempt_count += 1
                rejected_edges: List[Dict[str, object]] = []
                connector = _build_region_connector_cached(
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
                    allow_obstacle_aware=False,
                    cache=connector_cache,
                )
                if connector is None and obstacle_aware_retry_count < obstacle_aware_retry_limit:
                    obstacle_aware_retry_count += 1
                    repaired_edges: List[Dict[str, object]] = []
                    connector = _build_region_connector_cached(
                        agent_id,
                        serial,
                        current_pose,
                        candidate_pattern.entry_pose,
                        current_time,
                        config,
                        path_config,
                        obstacle_field,
                        to_region=region_id,
                        rejection_sink=repaired_edges,
                        allow_obstacle_aware=True,
                        cache=connector_cache,
                    )
                    if connector is None:
                        rejected_candidate_count += 1
                        depth_rejections.extend(repaired_edges or rejected_edges)
                        continue
                elif connector is None:
                    rejected_candidate_count += 1
                    depth_rejections.extend(rejected_edges)
                    continue
                connector_end_time = current_time
                if connector:
                    connector_end_time = _segment_end_time(connector[-1])
                sweep_segments, reason = _cached_internal_sweep_segments(
                    candidate_pattern,
                    config,
                    path_config,
                    obstacle_field,
                    start_time=connector_end_time,
                    segment_prefix=f"agent{agent_id}_region_{region_id}",
                    cache=sweep_segment_cache,
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
                candidate_segments = list(connector) + list(sweep_segments)
                repeat_score = RepeatOverlapScore(0.0, 0.0, 0.0, 0, 0)
                if path_config.enable_main_repeat_path_penalty:
                    repeat_score = score_repeat_overlap(
                        _non_cover_segments(candidate_segments),
                        list(state["segments"]),
                        path_config,
                        penalty_weight=max(path_config.main_repeat_path_penalty_weight, 0.0)
                        * max(path_config.connector_noncover_repeat_penalty_multiplier, 0.0),
                        annotate=False,
                    )
                cross_agent_score = CrossAgentOverlapScore(0.0, 0.0, 0.0, 0, 0, {}, {})
                if cross_agent_score.overlap_length <= 1e-9:
                    state_has_zero_cross_agent_overlap = True
                internal_repeat_penalty = _metadata_float(candidate_pattern.metadata, "internal_repeat_penalty", 0.0)
                pattern_quality_penalty = _pattern_quality_penalty(candidate_pattern, path_config)
                coverage_deficit = _metadata_float(
                    candidate_pattern.metadata,
                    "coverage_deficit",
                    max(0.0, path_config.target_coverage_fraction - _estimated_pattern_coverage_fraction(candidate_pattern, config)),
                ) if use_coverage_priority else 0.0
                new_segments = list(state["segments"]) + candidate_segments
                new_final_order = list(state["final_order"]) + [region_id]
                new_selected_patterns = dict(state["selected_patterns"])
                new_selected_patterns[region_id] = candidate_pattern
                new_selected_pattern_ids = dict(state["selected_pattern_ids"])
                new_selected_pattern_ids[region_id] = candidate_pattern.pattern_id
                step_score = (
                    dynamic_edge_cost(connector, config)
                    + candidate_pattern.estimated_time
                    + path_config.coverage_priority_weight * coverage_deficit
                    + pattern_quality_penalty
                    + repeat_score.penalty
                    + cross_agent_score.penalty
                    + _turn_clearance_penalty(candidate_pattern.exit_pose, config)
                )
                state_entries.append(
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
                        "repeat_overlap_length": float(state["repeat_overlap_length"]) + repeat_score.overlap_length,
                        "repeat_penalty": float(state["repeat_penalty"]) + repeat_score.penalty + internal_repeat_penalty,
                        "cross_agent_overlap_length": float(state["cross_agent_overlap_length"]) + cross_agent_score.overlap_length,
                        "cross_agent_penalty": float(state["cross_agent_penalty"]) + cross_agent_score.penalty,
                        "unavoidable_cross_agent_overlap_count": int(state["unavoidable_cross_agent_overlap_count"]),
                        "executed_coverage_length": float(state.get("executed_coverage_length", 0.0)) + candidate_pattern.coverage_length,
                        "candidate_segments": candidate_segments,
                        "candidate_cross_agent_overlap_length": cross_agent_score.overlap_length,
                    }
                )
            if state_entries:
                for entry in state_entries:
                    if (
                        float(entry.get("candidate_cross_agent_overlap_length", 0.0)) > 1e-9
                        and not state_has_zero_cross_agent_overlap
                    ):
                        mark_cross_agent_unavoidable(entry.get("candidate_segments", []))
                        entry["unavoidable_cross_agent_overlap_count"] = int(entry["unavoidable_cross_agent_overlap_count"]) + 1
                    entry.pop("candidate_segments", None)
                    entry.pop("candidate_cross_agent_overlap_length", None)
                next_beam.extend(state_entries)
        if next_beam:
            next_beam.sort(
                key=lambda item: (
                    len(item["remaining"]),
                    -float(item.get("executed_coverage_length", 0.0)),
                    float(item["score"]),
                    list(item["final_order"]),
                )
            )
            beam = next_beam[:beam_width]
            best_partial = min(
                [best_partial] + beam,
                key=lambda item: (
                    len(item["remaining"]),
                    -float(item.get("executed_coverage_length", 0.0)),
                    float(item["score"]),
                    list(item["final_order"]),
                ),
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
    all_connector_failures = _connector_failures_from_edges(infeasible_edges)
    skipped_region_reasons = _connector_failure_strings(all_connector_failures)
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
        "main_repeat_overlap_length": float(chosen_state.get("repeat_overlap_length", 0.0)),
        "main_repeat_penalty_total": float(chosen_state.get("repeat_penalty", 0.0)),
        "cross_agent_overlap_length": float(chosen_state.get("cross_agent_overlap_length", 0.0)),
        "cross_agent_penalty_total": float(chosen_state.get("cross_agent_penalty", 0.0)),
        "unavoidable_cross_agent_overlap_count": int(chosen_state.get("unavoidable_cross_agent_overlap_count", 0)),
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
        "skipped_region_reasons": skipped_region_reasons,
        "connector_failure_reasons": skipped_region_reasons,
        "all_connector_failure_reasons": _connector_failure_lists(all_connector_failures),
    }


def _solve_agent_region_tsp_large_map_greedy(
    agent_id: int,
    initial_order: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    ownership_map: CoverageOwnershipMap | None,
    fallback_solver_metadata: Dict[str, object] | None,
    sweep_segment_templates: Dict[str, Tuple[List[PathSegmentSpec], str]] | None = None,
) -> Dict[str, object]:
    current_pose = config.fleet.initial_states_3dof[agent_id].pose()
    current_time = 0.0
    serial = 0
    segments: List[PathSegmentSpec] = []
    selected_patterns: Dict[str, RegionCoveragePattern] = {}
    selected_pattern_ids: Dict[str, str] = {}
    final_order: List[str] = []
    infeasible_edges: List[Dict[str, object]] = []
    candidate_pattern_counts = {region_id: len(patterns.get(region_id, [])) for region_id in initial_order}
    candidate_attempt_count = 0
    rejected_candidate_count = 0
    main_repeat_overlap = 0.0
    main_repeat_penalty = 0.0
    cross_agent_overlap = 0.0
    cross_agent_penalty = 0.0
    connector_cache: Dict[Tuple[float, float, float, float, float, float, float, str, bool], Tuple[List[PathSegmentSpec] | None, List[Dict[str, object]]]] = {}
    sweep_segment_cache: Dict[str, Tuple[List[PathSegmentSpec], str]] = {
        key: (copy.deepcopy(value[0]), value[1])
        for key, value in (sweep_segment_templates or {}).items()
    }
    reachability_probe_count = 0
    reachability_probe_success_count = 0
    dead_end_avoidance_count = 0
    skipped_region_reasons: Dict[str, str] = {}
    connector_failure_reasons: Dict[str, str] = {}
    all_connector_failure_reasons: Dict[str, set[str]] = {}

    use_coverage_priority = _coverage_quality_priority_enabled(config, path_config)
    order_rank = {region_id: idx for idx, region_id in enumerate(initial_order)}
    pattern_limit = _connector_pattern_limit(path_config)
    remaining = list(initial_order)
    if path_config.monitor_stages:
        print(
            json.dumps(
                {
                    "stage": "agent_region_tsp_start",
                    "agent_id": agent_id,
                    "assigned_region_count": len(initial_order),
                    "large_map_greedy": True,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    while remaining:
        ordered_candidates: List[Tuple[float, int, str, RegionCoveragePattern]] = []
        for region_id in remaining:
            candidates = list(patterns.get(region_id, []))
            if not candidates:
                continue
            candidates.sort(key=lambda pattern: (_pattern_sort_key(pattern, config, path_config), pattern.pattern_id))
            for pattern in candidates[:pattern_limit]:
                coverage_deficit = (
                    _metadata_float(
                        pattern.metadata,
                        "coverage_deficit",
                        max(0.0, path_config.target_coverage_fraction - _estimated_pattern_coverage_fraction(pattern, config)),
                    )
                    if use_coverage_priority
                    else 0.0
                )
                ordered_candidates.append(
                    (
                        _transition_length(current_pose, pattern.entry_pose, config)
                        + 0.25 * pattern.estimated_time
                        + _pattern_quality_penalty(pattern, path_config)
                        + path_config.coverage_priority_weight * coverage_deficit
                        + 0.05 * order_rank.get(region_id, 10_000)
                        - 1.5 * pattern.coverage_length,
                        order_rank.get(region_id, 10_000),
                        region_id,
                        pattern,
                    )
                )
        if not ordered_candidates:
            infeasible_edges.extend(
                {"agent_id": agent_id, "region_id": region_id, "reason": "missing_candidate_patterns"}
                for region_id in remaining
            )
            for region_id in remaining:
                skipped_region_reasons.setdefault(region_id, "missing_candidate_patterns")
                _record_connector_failure(all_connector_failure_reasons, region_id, "missing_candidate_patterns")
            break
        ordered_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].pattern_id))
        feasible_choices: List[Dict[str, object]] = []
        local_rejections: List[Dict[str, object]] = []
        candidate_window = max(
            4,
            int(path_config.region_tsp_branch_limit) * max(1, min(pattern_limit, 4)),
        )
        candidate_slice = _prioritized_candidate_slice(
            ordered_candidates,
            candidate_window,
            pattern_limit,
            bool(path_config.prioritize_region_execution),
            region_index=2,
        )
        obstacle_aware_retry_limit = max(2, min(8, int(path_config.region_tsp_branch_limit) // 2))
        attempted_region_ids: List[str] = []
        for candidate_idx, (_, _, region_id, candidate_pattern) in enumerate(candidate_slice):
            if region_id not in attempted_region_ids:
                attempted_region_ids.append(region_id)
            candidate_attempt_count += 1
            connector_rejections: List[Dict[str, object]] = []
            connector = _build_region_connector_cached(
                agent_id,
                serial,
                current_pose,
                candidate_pattern.entry_pose,
                current_time,
                config,
                path_config,
                obstacle_field,
                to_region=region_id,
                rejection_sink=connector_rejections,
                allow_obstacle_aware=False,
                cache=connector_cache,
            )
            if connector is None and candidate_idx < obstacle_aware_retry_limit:
                connector_rejections = []
                connector = _build_region_connector_cached(
                    agent_id,
                    serial,
                    current_pose,
                    candidate_pattern.entry_pose,
                    current_time,
                    config,
                    path_config,
                    obstacle_field,
                    to_region=region_id,
                    rejection_sink=connector_rejections,
                    allow_obstacle_aware=True,
                    cache=connector_cache,
                )
            if connector is None:
                rejected_candidate_count += 1
                local_rejections.extend(connector_rejections)
                if connector_rejections:
                    reason = str(connector_rejections[-1].get("reason", "connector_failed"))
                    connector_failure_reasons.setdefault(region_id, reason)
                    _record_connector_failure(all_connector_failure_reasons, region_id, reason)
                else:
                    connector_failure_reasons.setdefault(region_id, "connector_failed")
                    _record_connector_failure(all_connector_failure_reasons, region_id, "connector_failed")
                continue
            connector_end_time = _segment_end_time(connector[-1]) if connector else current_time
            future_remaining = [item for item in remaining if item != region_id]
            lookahead_reachable = _large_map_lookahead_reachable_count(
                agent_id=agent_id,
                serial=serial + len(connector),
                current_pose=candidate_pattern.exit_pose,
                current_time=connector_end_time + candidate_pattern.estimated_time,
                remaining=future_remaining,
                patterns=patterns,
                config=config,
                path_config=path_config,
                obstacle_field=obstacle_field,
                connector_cache=connector_cache,
            )
            lookahead_probe_limit = max(4, int(path_config.region_tsp_branch_limit))
            reachability_probe_count += min(len(future_remaining), lookahead_probe_limit)
            reachability_probe_success_count += min(lookahead_reachable, lookahead_probe_limit)
            coverage_deficit = (
                _metadata_float(
                    candidate_pattern.metadata,
                    "coverage_deficit",
                    max(0.0, path_config.target_coverage_fraction - _estimated_pattern_coverage_fraction(candidate_pattern, config)),
                )
                if use_coverage_priority
                else 0.0
            )
            connector_repeat_score = RepeatOverlapScore(0.0, 0.0, 0.0, 0, 0)
            if path_config.enable_main_repeat_path_penalty:
                connector_repeat_score = score_repeat_overlap(
                    _non_cover_segments(connector),
                    segments,
                    path_config,
                    penalty_weight=max(path_config.main_repeat_path_penalty_weight, 0.0)
                    * max(path_config.connector_noncover_repeat_penalty_multiplier, 0.0),
                    annotate=False,
                )
            score = (
                sum(segment.length for segment in connector)
                + 0.5 * candidate_pattern.estimated_time
                + _pattern_quality_penalty(candidate_pattern, path_config)
                + connector_repeat_score.penalty
                + path_config.coverage_priority_weight * coverage_deficit
                + 30.0 * max(0, min(len(future_remaining), 3) - lookahead_reachable)
                - 1.0 * candidate_pattern.coverage_length
            )
            feasible_choices.append(
                {
                    "score": score,
                    "region_id": region_id,
                    "pattern": candidate_pattern,
                    "connector": connector,
                    "connector_end_time": connector_end_time,
                    "lookahead_reachable": lookahead_reachable,
                }
            )
            if int(lookahead_reachable) > 0:
                break
            if len(feasible_choices) >= 2:
                break
        if not feasible_choices:
            skipped_region_id = attempted_region_ids[0] if attempted_region_ids else remaining[0]
            if local_rejections:
                infeasible_edges.extend(local_rejections)
                skipped_reason = str(local_rejections[-1].get("reason", "no_valid_large_map_greedy_candidate"))
            else:
                skipped_reason = "no_valid_large_map_greedy_candidate"
                infeasible_edges.append(
                    {
                        "agent_id": agent_id,
                        "region_id": skipped_region_id,
                        "reason": skipped_reason,
                    }
                )
            skipped_region_reasons.setdefault(skipped_region_id, skipped_reason)
            connector_failure_reasons.setdefault(skipped_region_id, skipped_reason)
            _record_connector_failure(all_connector_failure_reasons, skipped_region_id, skipped_reason)
            remaining.remove(skipped_region_id)
            continue

        feasible_choices.sort(
            key=lambda item: (
                -int(item["lookahead_reachable"]),
                float(item["score"]),
                str(item["region_id"]),
                item["pattern"].pattern_id,
            )
        )
        chosen = feasible_choices[0]
        if len(feasible_choices) > 1 and int(chosen["lookahead_reachable"]) > int(feasible_choices[-1]["lookahead_reachable"]):
            dead_end_avoidance_count += 1
        candidate_pattern = chosen["pattern"]
        connector = chosen["connector"]
        region_id = candidate_pattern.region_id
        if path_config.monitor_stages and len(final_order) == 0:
            print(
                json.dumps(
                    {
                        "stage": "agent_region_tsp_choice",
                        "agent_id": agent_id,
                        "region_id": region_id,
                        "pattern_id": candidate_pattern.pattern_id,
                        "remaining_region_count": len(remaining),
                        "candidate_attempt_count": candidate_attempt_count,
                        "connector_cache_size": len(connector_cache),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        sweep_segments, reason = _cached_internal_sweep_segments(
            candidate_pattern,
            config,
            path_config,
            obstacle_field,
            start_time=float(chosen["connector_end_time"]),
            segment_prefix=f"agent{agent_id}_region_{region_id}",
            cache=sweep_segment_cache,
        )
        if reason:
            rejected_candidate_count += 1
            infeasible_edges.append(
                {
                    "agent_id": agent_id,
                    "region_id": region_id,
                    "pattern_id": candidate_pattern.pattern_id,
                    "reason": reason,
                }
            )
            skipped_region_reasons.setdefault(region_id, reason)
            connector_failure_reasons.setdefault(region_id, reason)
            _record_connector_failure(all_connector_failure_reasons, region_id, reason)
            remaining.remove(region_id)
            continue
        candidate_segments = list(connector) + list(sweep_segments)
        repeat_score = RepeatOverlapScore(0.0, 0.0, 0.0, 0, 0)
        if path_config.enable_main_repeat_path_penalty:
            repeat_score = score_repeat_overlap(
                _non_cover_segments(candidate_segments),
                segments,
                path_config,
                penalty_weight=max(path_config.main_repeat_path_penalty_weight, 0.0)
                * max(path_config.connector_noncover_repeat_penalty_multiplier, 0.0),
                annotate=True,
            )
        cross_score = CrossAgentOverlapScore(0.0, 0.0, 0.0, 0, 0, {}, {})
        main_repeat_overlap += repeat_score.overlap_length
        main_repeat_penalty += repeat_score.penalty + _metadata_float(candidate_pattern.metadata, "internal_repeat_penalty", 0.0)
        cross_agent_overlap += cross_score.overlap_length
        cross_agent_penalty += cross_score.penalty
        segments.extend(candidate_segments)
        final_order.append(region_id)
        selected_patterns[region_id] = candidate_pattern
        selected_pattern_ids[region_id] = candidate_pattern.pattern_id
        remaining.remove(region_id)
        serial += len(candidate_segments)
        current_time = _segment_end_time(sweep_segments[-1])
        current_pose = candidate_pattern.exit_pose
        if path_config.monitor_stages and (len(final_order) == 1 or len(final_order) % 5 == 0 or not remaining):
            print(
                json.dumps(
                    {
                        "stage": "agent_region_tsp_progress",
                        "agent_id": agent_id,
                        "visited_region_count": len(final_order),
                        "remaining_region_count": len(remaining),
                        "candidate_attempt_count": candidate_attempt_count,
                        "rejected_candidate_count": rejected_candidate_count,
                        "connector_cache_size": len(connector_cache),
                        "sweep_segment_cache_size": len(sweep_segment_cache),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    metadata = fallback_solver_metadata or {
        "requested_tsp_solver": "deterministic",
        "effective_tsp_solver": "deterministic",
        "tsp_solver_status": "success",
        "aco_best_objective": None,
        "aco_initial_objective": None,
        "aco_iteration_count": 0,
        "aco_convergence_trace": [],
        "aco_accepted_3opt_count": 0,
    }
    metadata = {
        **metadata,
        "large_map_greedy_tsp": True,
        "large_map_greedy_final_region_count": len(final_order),
        "large_map_connector_cache_size": len(connector_cache),
        "large_map_reachability_probe_count": reachability_probe_count,
        "large_map_reachability_probe_success_count": reachability_probe_success_count,
        "large_map_dead_end_avoidance_count": dead_end_avoidance_count,
    }
    return {
        "initial_order": list(initial_order),
        "final_order": final_order,
        "segments": segments,
        "infeasible_edges": infeasible_edges,
        "selected_patterns": selected_patterns,
        "selected_pattern_ids": selected_pattern_ids,
        "candidate_pattern_counts": candidate_pattern_counts,
        "candidate_attempt_count": candidate_attempt_count,
        "rejected_candidate_count": rejected_candidate_count,
        "main_repeat_overlap_length": main_repeat_overlap,
        "main_repeat_penalty_total": main_repeat_penalty,
        "cross_agent_overlap_length": cross_agent_overlap,
        "cross_agent_penalty_total": cross_agent_penalty,
        "unavoidable_cross_agent_overlap_count": 0,
        "tsp_solver_metadata": metadata,
        "skipped_region_reasons": skipped_region_reasons,
        "connector_failure_reasons": connector_failure_reasons,
        "all_connector_failure_reasons": _connector_failure_lists(all_connector_failure_reasons),
    }


def _large_map_lookahead_reachable_count(
    agent_id: int,
    serial: int,
    current_pose: Pose2D,
    current_time: float,
    remaining: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    connector_cache: Dict[Tuple[float, float, float, float, float, float, float, str, bool], Tuple[List[PathSegmentSpec] | None, List[Dict[str, object]]]],
) -> int:
    if not remaining:
        return 0
    ordered = sorted(
        remaining,
        key=lambda region_id: (
            min(
                (
                    _transition_length(current_pose, pattern.entry_pose, config)
                    + _turn_clearance_penalty(pattern.entry_pose, config)
                    for pattern in patterns.get(region_id, [])
                ),
                default=float("inf"),
            ),
            region_id,
        ),
    )
    reachable = 0
    probe_limit = max(4, int(path_config.region_tsp_branch_limit))
    pattern_limit = _connector_pattern_limit(path_config)
    for region_id in ordered[:probe_limit]:
        region_reachable = False
        for pattern in sorted(patterns.get(region_id, []), key=lambda item: (_pattern_sort_key(item, config, path_config), item.pattern_id))[:pattern_limit]:
            if _cheap_region_connector_probe(current_pose, pattern.entry_pose, config, path_config, obstacle_field):
                region_reachable = True
                break
        if region_reachable:
            reachable += 1
    return reachable


def _cheap_region_connector_probe(
    start: Pose2D,
    end: Pose2D,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> bool:
    segment = build_transition_segment(
        segment_id="large_map_lookahead_probe",
        start=start,
        end=end,
        start_time=0.0,
        config=config,
        kind="transit",
        sample_count=16,
        use_bezier=path_config.use_bezier_smoothing,
    )
    return validate_transition_sequence([segment], config, obstacle_field=obstacle_field, retime=True).valid


def _solve_agent_region_tsp_aco(
    agent_id: int,
    region_ids: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    initial_order: Sequence[str],
    ownership_map: CoverageOwnershipMap | None,
) -> Dict[str, object]:
    start_pose = config.fleet.initial_states_3dof[agent_id].pose()
    candidate_pattern_counts = {region_id: len(patterns.get(region_id, [])) for region_id in region_ids}
    invalid_edges: List[Dict[str, object]] = []
    cost_cache: Dict[Tuple[str, str], float] = {}
    invalid_edge_keys: set[Tuple[str, str]] = set()
    use_coverage_priority = _coverage_quality_priority_enabled(config, path_config)

    def pattern_key(pattern: RegionCoveragePattern | None) -> str:
        return "__start__" if pattern is None else f"{pattern.region_id}:{pattern.pattern_id}"

    def edge_cost(previous: RegionCoveragePattern | None, candidate_pattern: RegionCoveragePattern) -> float:
        key = (pattern_key(previous), pattern_key(candidate_pattern))
        if key in cost_cache:
            return cost_cache[key]
        current_pose = start_pose if previous is None else previous.exit_pose
        if _large_map_mode_enabled(config, path_config):
            connector = [
                build_transition_segment(
                    segment_id=f"aco_edge_probe_{agent_id}_{candidate_pattern.region_id}",
                    start=current_pose,
                    end=candidate_pattern.entry_pose,
                    start_time=0.0,
                    config=config,
                    kind="transit",
                    sample_count=16,
                    use_bezier=path_config.use_bezier_smoothing,
                )
            ]
            probe_report = validate_transition_sequence(connector, config, obstacle_field=obstacle_field, retime=True)
            if not probe_report.valid:
                if key not in invalid_edge_keys:
                    invalid_edge_keys.add(key)
                    invalid_edges.append(
                        {
                            "agent_id": agent_id,
                            "to_region": candidate_pattern.region_id,
                            "reason": ",".join(probe_report.reasons),
                            "segment_sources": [segment.path_source for segment in connector],
                            "aco_edge_probe": True,
                        }
                    )
                cost_cache[key] = float("inf")
                return cost_cache[key]
        else:
            validation_rejections: List[Dict[str, object]] = []
            connector = _build_region_connector(
                agent_id,
                0,
                current_pose,
                candidate_pattern.entry_pose,
                0.0,
                config,
                path_config,
                obstacle_field,
                to_region=candidate_pattern.region_id,
                rejection_sink=validation_rejections,
            )
            if connector is None:
                if key not in invalid_edge_keys:
                    invalid_edge_keys.add(key)
                    invalid_edges.extend(validation_rejections)
                cost_cache[key] = float("inf")
                return cost_cache[key]
        transition = dubins_shortest_path(current_pose, candidate_pattern.entry_pose, config.fleet.min_turn_radius)
        transition_turn = _dubins_turn_angle(transition.segment_lengths, transition.modes, config.fleet.min_turn_radius)
        transition_length = sum(segment.length for segment in connector)
        transition_time = sum(_segment_duration(segment) for segment in connector)
        cross_penalty = _approximate_pattern_cross_agent_penalty(
            agent_id,
            current_pose,
            candidate_pattern,
            config,
            path_config,
            ownership_map,
        )
        coverage_deficit = _metadata_float(
            candidate_pattern.metadata,
            "coverage_deficit",
            max(0.0, path_config.target_coverage_fraction - _estimated_pattern_coverage_fraction(candidate_pattern, config)),
        ) if use_coverage_priority else 0.0
        cost_cache[key] = (
            path_config.length_weight * (transition_length + candidate_pattern.total_length)
            + path_config.turn_angle_weight * (transition_turn + candidate_pattern.turn_angle)
            + path_config.time_weight * (transition_time + candidate_pattern.estimated_time)
            + path_config.coverage_priority_weight * coverage_deficit
            + _pattern_quality_penalty(candidate_pattern, path_config)
            + cross_penalty
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
    main_repeat_overlap = 0.0
    main_repeat_penalty = 0.0
    cross_agent_overlap = 0.0
    cross_agent_penalty = 0.0
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
        repeat_weight = (
            path_config.main_repeat_path_penalty_weight
            * max(path_config.connector_noncover_repeat_penalty_multiplier, 0.0)
            if path_config.enable_main_repeat_path_penalty
            else 0.0
        )
        repeat_score = score_repeat_overlap(
            _non_cover_segments(list(connector) + list(sweep_segments)),
            segments,
            path_config,
            penalty_weight=repeat_weight,
            annotate=True,
        )
        cross_score = score_cross_agent_ownership_overlap(
            list(connector) + list(sweep_segments),
            agent_id,
            ownership_map,
            path_config,
            config=config,
            annotate=True,
        )
        main_repeat_overlap += repeat_score.overlap_length
        main_repeat_penalty += repeat_score.penalty + _metadata_float(pattern.metadata, "internal_repeat_penalty", 0.0)
        cross_agent_overlap += cross_score.overlap_length
        cross_agent_penalty += cross_score.penalty
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
        "main_repeat_overlap_length": main_repeat_overlap,
        "main_repeat_penalty_total": main_repeat_penalty,
        "cross_agent_overlap_length": cross_agent_overlap,
        "cross_agent_penalty_total": cross_agent_penalty,
        "unavoidable_cross_agent_overlap_count": 0,
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
        "main_repeat_overlap_length": 0.0,
        "main_repeat_penalty_total": 0.0,
        "cross_agent_overlap_length": 0.0,
        "cross_agent_penalty_total": 0.0,
        "unavoidable_cross_agent_overlap_count": 0,
        "tsp_solver_metadata": metadata,
    }


def _approximate_pattern_cross_agent_penalty(
    agent_id: int,
    current_pose: Pose2D,
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    ownership_map: CoverageOwnershipMap | None,
) -> float:
    if ownership_map is None or not path_config.enable_cross_agent_coverage_penalty:
        return 0.0
    segments: List[PathSegmentSpec] = [
        build_transition_segment(
            segment_id=f"aco_cross_agent_estimate_{agent_id}_{pattern.region_id}",
            start=current_pose,
            end=pattern.entry_pose,
            start_time=0.0,
            config=config,
            kind="transit",
            sample_count=24,
            use_bezier=path_config.use_bezier_smoothing,
        )
    ]
    current_time = _segment_end_time(segments[-1])
    for coverage_pass in pattern.passes:
        segments.append(
            build_cover_segment(
                segment_id=f"aco_cross_agent_estimate_{agent_id}_{coverage_pass.pass_id}",
                start=coverage_pass.start_pose,
                end=coverage_pass.end_pose,
                start_time=current_time,
                speed=max(config.fleet.cover_speed, 1e-6),
            )
        )
        current_time = _segment_end_time(segments[-1])
    return score_cross_agent_ownership_overlap(
        segments,
        agent_id,
        ownership_map,
        path_config,
        config=config,
        annotate=False,
    ).penalty


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


def _append_skipped_region_recovery(
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    agents: Dict[int, AgentPathPlan],
    tours: Dict[int, SingleUsvTourPlan],
    tsp_records: Dict[int, Dict[str, object]],
    feasible_patterns: Dict[str, List[RegionCoveragePattern]],
    sweep_segment_templates: Dict[str, Tuple[List[PathSegmentSpec], str]] | None = None,
) -> Dict[str, object]:
    if path_config.target_coverage_fraction <= 0.0:
        return {"enabled": False, "reason": "target_coverage_zero", "recovered_count": 0, "failed_count": 0}
    skipped = sorted(
        {
            region_id
            for record in tsp_records.values()
            for region_id in record.get("skipped_regions", [])
            if feasible_patterns.get(region_id)
        }
    )
    if not skipped:
        return {"enabled": True, "recovered_count": 0, "failed_count": 0, "failure_reasons": {}}

    recovered: List[str] = []
    failed: Dict[str, str] = {}
    connector_cache: Dict[Tuple[float, float, float, float, float, float, float, str, bool], Tuple[List[PathSegmentSpec] | None, List[Dict[str, object]]]] = {}
    sweep_segment_cache: Dict[str, Tuple[List[PathSegmentSpec], str]] = {
        key: (copy.deepcopy(value[0]), value[1])
        for key, value in (sweep_segment_templates or {}).items()
    }
    recovery_budget = max(1, min(4, int(path_config.max_residual_backfill_regions)))
    max_attempts = min(len(skipped), recovery_budget)
    pattern_limit = 1
    for _ in range(max_attempts):
        best_choice = None
        obstacle_aware_retry_limit = max(2, min(8, int(path_config.region_tsp_branch_limit) // 2))
        obstacle_aware_retry_count = 0
        active_skipped = sorted(
            [region_id for region_id in skipped if region_id not in recovered],
            key=lambda region_id: _skipped_region_recovery_priority(region_id, feasible_patterns, agents, config),
        )[:recovery_budget]
        for region_id in active_skipped:
            if region_id in recovered:
                continue
            for agent_id, agent in sorted(agents.items()):
                current_pose = _agent_end_pose(agent, config)
                current_time = max((_segment_end_time(segment) for segment in agent.segments), default=0.0)
                serial = len(agent.segments)
                for pattern in sorted(feasible_patterns.get(region_id, []), key=lambda item: (_pattern_sort_key(item, config, path_config), item.pattern_id))[:pattern_limit]:
                    rejections: List[Dict[str, object]] = []
                    repaired_rejections: List[Dict[str, object]] = []
                    connector = _build_region_connector_cached(
                        agent_id=agent_id,
                        serial=serial,
                        start=current_pose,
                        end=pattern.entry_pose,
                        start_time=current_time,
                        config=config,
                        path_config=path_config,
                        obstacle_field=obstacle_field,
                        to_region=region_id,
                        rejection_sink=rejections,
                        allow_obstacle_aware=False,
                        cache=connector_cache,
                    )
                    if connector is None and obstacle_aware_retry_count < obstacle_aware_retry_limit:
                        obstacle_aware_retry_count += 1
                        connector = _build_region_connector_cached(
                            agent_id=agent_id,
                            serial=serial,
                            start=current_pose,
                            end=pattern.entry_pose,
                            start_time=current_time,
                            config=config,
                            path_config=path_config,
                            obstacle_field=obstacle_field,
                            to_region=region_id,
                            rejection_sink=repaired_rejections,
                            allow_obstacle_aware=True,
                            cache=connector_cache,
                        )
                    if connector is None:
                        combined_rejections = repaired_rejections or rejections
                        if combined_rejections:
                            reason = str(combined_rejections[-1].get("reason", "connector_failed"))
                            failed.setdefault(region_id, reason)
                            _record_recovery_failure_in_tsp_records(tsp_records, region_id, reason)
                        continue
                    connector_end_time = _segment_end_time(connector[-1]) if connector else current_time
                    sweep_segments, reason = _cached_internal_sweep_segments(
                        pattern,
                        config,
                        path_config,
                        obstacle_field,
                        start_time=connector_end_time,
                        segment_prefix=f"agent{agent_id}_recovered_region_{region_id}",
                        cache=sweep_segment_cache,
                    )
                    if reason:
                        failed.setdefault(region_id, reason)
                        _record_recovery_failure_in_tsp_records(tsp_records, region_id, reason)
                        continue
                    candidate_segments = list(connector) + list(sweep_segments)
                    if not validate_transition_sequence(candidate_segments, config, obstacle_field=obstacle_field, retime=True).valid:
                        failed.setdefault(region_id, "dynamic_validation_failed")
                        _record_recovery_failure_in_tsp_records(tsp_records, region_id, "dynamic_validation_failed")
                        continue
                    score = (
                        sum(segment.length for segment in candidate_segments)
                        + pattern.estimated_time
                        + _pattern_quality_penalty(pattern, path_config)
                        + _turn_clearance_penalty(pattern.exit_pose, config)
                        - 2.0 * pattern.coverage_length
                    )
                    key = (-pattern.coverage_length, score, agent_id, region_id, pattern.pattern_id)
                    if best_choice is None or key < best_choice[0]:
                        best_choice = (key, agent_id, region_id, pattern, candidate_segments)
        if best_choice is None:
            break
        _, agent_id, region_id, pattern, candidate_segments = best_choice
        agent = agents[agent_id]
        tour = tours[agent_id]
        for segment in candidate_segments:
            segment.metadata["skipped_region_recovery"] = "true"
            segment.metadata["region_id"] = region_id
        agent.segments.extend(candidate_segments)
        tour.segments = agent.segments
        tour.region_order.append(region_id)
        tour.selected_patterns[region_id] = pattern
        tour.total_length = sum(segment.length for segment in tour.segments)
        tour.total_turn_angle = sum(_segment_heading_variation(segment) for segment in tour.segments)
        tour.estimated_time = max((_segment_end_time(segment) for segment in tour.segments), default=0.0)
        recovered.append(region_id)
        failed.pop(region_id, None)
        record = tsp_records.get(agent_id)
        if record is not None:
            record.setdefault("final_order", []).append(region_id)
            record.setdefault("selected_pattern_ids", {})[region_id] = pattern.pattern_id
            if region_id not in record.get("assigned_regions", []):
                record.setdefault("assigned_regions", []).append(region_id)
        for any_record in tsp_records.values():
            any_record["skipped_regions"] = [item for item in any_record.get("skipped_regions", []) if item != region_id]
            any_record.get("skipped_region_reasons", {}).pop(region_id, None)
            any_record.get("connector_failure_reasons", {}).pop(region_id, None)
    remaining_failed = [region_id for region_id in skipped if region_id not in set(recovered)]
    for region_id in remaining_failed:
        failed.setdefault(region_id, "no_feasible_recovery_insertion")
    return {
        "enabled": True,
        "recovered_count": len(recovered),
        "failed_count": len(remaining_failed),
        "recovered_regions": recovered,
        "failure_reasons": failed,
        "connector_cache_size": len(connector_cache),
    }


def _record_recovery_failure_in_tsp_records(
    tsp_records: Dict[int, Dict[str, object]],
    region_id: str,
    reason: str,
) -> None:
    for record in tsp_records.values():
        if region_id not in record.get("assigned_regions", []):
            continue
        record.setdefault("connector_failure_reasons", {}).setdefault(region_id, reason)
        record.setdefault("skipped_region_reasons", {}).setdefault(region_id, reason)
        all_reasons = record.setdefault("all_connector_failure_reasons", {})
        region_reasons = all_reasons.setdefault(region_id, [])
        if reason not in region_reasons:
            region_reasons.append(reason)


def _skipped_region_tail_distance(
    region_id: str,
    feasible_patterns: Dict[str, List[RegionCoveragePattern]],
    agents: Dict[int, AgentPathPlan],
    config: PlannerConfig,
) -> float:
    patterns = feasible_patterns.get(region_id, [])
    if not patterns:
        return float("inf")
    best = float("inf")
    for agent in agents.values():
        tail = _agent_end_pose(agent, config)
        for pattern in patterns[:2]:
            best = min(best, math.hypot(tail.x - pattern.entry_pose.x, tail.y - pattern.entry_pose.y))
    return best


def _skipped_region_recovery_priority(
    region_id: str,
    feasible_patterns: Dict[str, List[RegionCoveragePattern]],
    agents: Dict[int, AgentPathPlan],
    config: PlannerConfig,
) -> Tuple[float, float, str]:
    patterns = feasible_patterns.get(region_id, [])
    best_coverage_length = max((pattern.coverage_length for pattern in patterns), default=0.0)
    return (
        -best_coverage_length,
        _skipped_region_tail_distance(region_id, feasible_patterns, agents, config),
        region_id,
    )


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


def _build_region_connector_cached(
    agent_id: int,
    serial: int,
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    to_region: str,
    rejection_sink: List[Dict[str, object]] | None,
    allow_obstacle_aware: bool,
    cache: Dict[Tuple[float, float, float, float, float, float, float, str, bool], Tuple[List[PathSegmentSpec] | None, List[Dict[str, object]]]],
) -> List[PathSegmentSpec] | None:
    key = (
        round(start.x, 3),
        round(start.y, 3),
        round(start.psi, 3),
        round(end.x, 3),
        round(end.y, 3),
        round(end.psi, 3),
        round(start_time, 2),
        to_region,
        bool(allow_obstacle_aware),
    )
    if key in cache:
        cached_segments, cached_rejections = cache[key]
        if cached_segments is None:
            if rejection_sink is not None:
                rejection_sink.extend(copy.deepcopy(cached_rejections))
            return None
        return copy.deepcopy(cached_segments)

    local_rejections: List[Dict[str, object]] = []
    connector = _build_region_connector(
        agent_id=agent_id,
        serial=serial,
        start=start,
        end=end,
        start_time=start_time,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        to_region=to_region,
        rejection_sink=local_rejections,
        allow_obstacle_aware=allow_obstacle_aware,
    )
    cache[key] = (copy.deepcopy(connector), copy.deepcopy(local_rejections))
    if connector is None and rejection_sink is not None:
        rejection_sink.extend(local_rejections)
    return connector


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
    allow_obstacle_aware: bool = True,
) -> List[PathSegmentSpec] | None:
    segment_id = f"agent{agent_id}_region_edge_{serial}_to_{to_region}"
    if allow_obstacle_aware:
        segments = build_obstacle_aware_transition_segments(
            segment_id=segment_id,
            start=start,
            end=end,
            start_time=start_time,
            config=config,
            path_config=path_config,
            obstacle_field=obstacle_field,
            kind="transit",
            sample_count=48,
        )
    else:
        segments = [
            build_transition_segment(
                segment_id=segment_id,
                start=start,
                end=end,
                start_time=start_time,
                config=config,
                kind="transit",
                sample_count=24,
                use_bezier=path_config.use_bezier_smoothing,
            )
        ]
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
                    "segment_connectors": [segment.metadata.get("connector", "") for segment in segments],
                    "segment_dynamic_reasons": [segment.metadata.get("dynamic_invalid_reasons", "") for segment in segments],
                    "astar_corridor_conversion_attempted": any(
                        segment.metadata.get("astar_corridor_conversion_attempted") == "true"
                        for segment in segments
                    ),
                    "astar_corridor_conversion_success": any(
                        segment.metadata.get("astar_corridor_conversion_success") == "true"
                        for segment in segments
                    ),
                    "astar_corridor_conversion_failure_reason": ",".join(
                        sorted(
                            {
                                segment.metadata.get("astar_corridor_conversion_failure_reason", "")
                                for segment in segments
                                if segment.metadata.get("astar_corridor_conversion_failure_reason")
                            }
                        )
                    ),
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


def _agent_repeat_metrics(
    agents: Dict[int, AgentPathPlan],
    path_config: PathPlanningConfig,
) -> Dict[int, Dict[str, float]]:
    metrics: Dict[int, Dict[str, float]] = {}
    weight = path_config.main_repeat_path_penalty_weight if path_config.enable_main_repeat_path_penalty else 0.0
    for agent_id, agent in sorted(agents.items()):
        for segment in agent.segments:
            if segment.kind == "cover":
                segment.metadata["repeat_overlap_length"] = "0.000000"
                segment.metadata["repeat_overlap_hit_ratio"] = "0.000000"
        existing: List[PathSegmentSpec] = []
        total_overlap = 0.0
        total_penalty = 0.0
        total_hits = 0
        total_samples = 0
        for segment in agent.segments:
            if segment.kind == "cover":
                existing.append(segment)
                continue
            score = score_repeat_overlap([segment], existing, path_config, penalty_weight=weight, annotate=True)
            total_overlap += score.overlap_length
            total_penalty += score.penalty
            total_hits += score.hit_count
            total_samples += score.sampled_point_count
            existing.append(segment)
        agent.metrics["main_repeat_overlap_length"] = total_overlap
        agent.metrics["main_repeat_penalty_total"] = total_penalty
        agent.metrics["main_repeat_hit_ratio"] = total_hits / max(total_samples, 1)
        metrics[agent_id] = {
            "overlap_length": total_overlap,
            "penalty": total_penalty,
            "hit_ratio": total_hits / max(total_samples, 1),
            "hit_count": float(total_hits),
            "sampled_point_count": float(total_samples),
        }
    return metrics


def _non_cover_segments(segments: Sequence[PathSegmentSpec]) -> List[PathSegmentSpec]:
    return [segment for segment in segments if segment.kind != "cover"]


def _unavoidable_cross_agent_overlap_count(agents: Dict[int, AgentPathPlan]) -> int:
    return sum(
        1
        for agent in agents.values()
        for segment in agent.segments
        if segment.metadata.get("unavoidable_cross_agent_overlap") == "true"
    )


def _heading_repair_count(agents: Dict[int, AgentPathPlan]) -> int:
    return sum(
        1
        for agent in agents.values()
        for segment in agent.segments
        if segment.metadata.get("heading_repair_applied") == "true"
    )


def _raw_astar_edge_rejected_count(infeasible_edges: Sequence[Dict[str, object]]) -> int:
    count = 0
    for edge in infeasible_edges:
        reason = str(edge.get("reason", ""))
        sources = ",".join(str(item) for item in edge.get("segment_sources", []) or [])
        if "raw_astar_corridor_edge" in reason or "astar_corridor_edge" in sources:
            count += 1
    return count


def _astar_corridor_conversion_report(
    agents: Dict[int, AgentPathPlan],
    infeasible_edges: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    attempt_count = 0
    success_count = 0
    failure_count = 0
    methods: Dict[str, int] = {}
    failure_reasons: Dict[str, int] = {}
    for agent in agents.values():
        for segment in agent.segments:
            if segment.metadata.get("astar_corridor_conversion_attempted") != "true":
                continue
            attempt_count += 1
            if segment.metadata.get("astar_corridor_conversion_success") == "true":
                success_count += 1
                method = segment.metadata.get("corridor_conversion_method", "unknown")
                methods[method] = methods.get(method, 0) + 1
            else:
                failure_count += 1
                reason = segment.metadata.get("astar_corridor_conversion_failure_reason", "unknown")
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    for edge in infeasible_edges:
        if not edge.get("astar_corridor_conversion_attempted"):
            continue
        attempt_count += 1
        if edge.get("astar_corridor_conversion_success"):
            success_count += 1
        else:
            failure_count += 1
            reason = str(edge.get("astar_corridor_conversion_failure_reason") or edge.get("reason") or "unknown")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    return {
        "attempt_count": attempt_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "methods": methods,
        "failure_reasons": failure_reasons,
    }


def _large_map_tsp_metadata_summary(tsp_records: Dict[int, Dict[str, object]]) -> Dict[str, object]:
    cache_size = 0
    probe_count = 0
    probe_success = 0
    dead_end_avoidance = 0
    components: Dict[str, Dict[str, int]] = {}
    for agent_id, record in tsp_records.items():
        metadata = record.get("tsp_solver_metadata", {}) or {}
        cache_size += int(metadata.get("large_map_connector_cache_size", 0) or 0)
        probe_count += int(metadata.get("large_map_reachability_probe_count", 0) or 0)
        probe_success += int(metadata.get("large_map_reachability_probe_success_count", 0) or 0)
        dead_end_avoidance += int(metadata.get("large_map_dead_end_avoidance_count", 0) or 0)
        components[str(agent_id)] = {
            "assigned_region_count": len(record.get("assigned_regions", []) or []),
            "visited_region_count": len(record.get("final_order", []) or []),
            "skipped_region_count": len(record.get("skipped_regions", []) or []),
            "infeasible_edge_count": len(record.get("infeasible_edges", []) or []),
        }
    return {
        "large_map_connector_cache_size": cache_size,
        "large_map_reachability_probe_count": probe_count,
        "large_map_reachability_probe_success_count": probe_success,
        "large_map_dead_end_avoidance_count": dead_end_avoidance,
        "region_connection_graph_components": components,
    }


def _skipped_region_diagnostics(
    skipped_region_ids: Iterable[str],
    tsp_records: Dict[int, Dict[str, object]],
    feasible_patterns: Dict[str, List[RegionCoveragePattern]],
) -> Dict[str, Dict[str, object]]:
    diagnostics: Dict[str, Dict[str, object]] = {}
    for region_id in sorted(skipped_region_ids):
        assigned_agent = None
        skip_reason = "not_selected"
        connector_failure_reason = ""
        all_connector_failure_reasons: List[str] = []
        for agent_id, record in sorted(tsp_records.items()):
            if region_id not in record.get("assigned_regions", []):
                continue
            assigned_agent = int(agent_id)
            skip_reason = str(
                record.get("skipped_region_reasons", {}).get(region_id)
                or record.get("connector_failure_reasons", {}).get(region_id)
                or skip_reason
            )
            connector_failure_reason = str(record.get("connector_failure_reasons", {}).get(region_id, ""))
            raw_reasons = record.get("all_connector_failure_reasons", {}).get(region_id, [])
            if isinstance(raw_reasons, str):
                all_connector_failure_reasons = [item for item in raw_reasons.split(",") if item]
            else:
                all_connector_failure_reasons = [str(item) for item in raw_reasons]
            break
        patterns = feasible_patterns.get(region_id, [])
        diagnostics[region_id] = {
            "assigned_agent": assigned_agent,
            "candidate_pattern_count": len(patterns),
            "best_pattern_coverage_length": max((pattern.coverage_length for pattern in patterns), default=0.0),
            "skip_reason": skip_reason,
            "connector_failure_reason": connector_failure_reason,
            "all_connector_failure_reasons": sorted(set(all_connector_failure_reasons)),
        }
    return diagnostics


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


def _segment_duration(segment: PathSegmentSpec) -> float:
    if len(segment.waypoints) < 2:
        return 0.0
    start = segment.waypoints[0].time or 0.0
    end = segment.waypoints[-1].time or start
    if end > start:
        return end - start
    speed = max(segment.waypoints[0].speed or 1.0, 1e-6)
    return segment.length / speed


def _pose_label(pose: Pose2D) -> List[float]:
    return [round(pose.x, 3), round(pose.y, 3), round(pose.psi, 3)]


def _draw_region_or_member_cells(ax, region, facecolor: str, edgecolor: str, alpha: float, linewidth: float) -> None:
    member_cells = list(getattr(region, "member_cells", []) or [])
    if not member_cells:
        _draw_polygon(ax, region.polygon, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
        return
    for cell in member_cells:
        _draw_polygon(ax, cell.polygon, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=max(linewidth * 0.55, 0.25))
    _draw_polygon(ax, region.polygon, facecolor="none", edgecolor=edgecolor, alpha=min(alpha + 0.35, 1.0), linewidth=linewidth, linestyle="--")


def _render_paper_style_outputs(
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
    regions,
    sweep_paths: Dict[str, RegionSweepPath],
    assignment: Dict[int, List[str]],
    tsp_records: Dict[int, Dict[str, object]],
    path_plan: MultiAgentPathPlan,
    report: Dict[str, object],
    ownership_map: CoverageOwnershipMap | None,
    path_config: PathPlanningConfig,
    output_dir: str | Path,
    dpi: int,
    raw_regions=None,
    candidate_regions=None,
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

    fig, ax = _new_map_axes(config, "02 Free-Space Region Trace")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    for region in raw_regions or regions:
        _draw_region_or_member_cells(ax, region, facecolor="#dbeafe", edgecolor="#93c5fd", alpha=0.10, linewidth=0.35)
    for region in candidate_regions or regions:
        _draw_region_or_member_cells(ax, region, facecolor="none", edgecolor="#f59e0b", alpha=0.95, linewidth=0.8)
    for region in regions:
        _draw_region_or_member_cells(ax, region, facecolor="#86efac", edgecolor="#15803d", alpha=0.26, linewidth=1.0)
        ax.text(region.center[0], region.center[1], region.region_id.replace("free_cell_", "c"), fontsize=5, ha="center")
    ax.plot([], [], color="#93c5fd", linewidth=1.0, label="raw free cells")
    ax.plot([], [], color="#f59e0b", linewidth=1.2, label="merged candidates")
    ax.plot([], [], color="#15803d", linewidth=1.4, label="feasible sweep regions")
    ax.legend(loc="upper right", fontsize=7)
    save(fig, "02_free_space_regions.png")

    fig, ax = _new_map_axes(config, "03 Feasible Region Sweep Modes")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    _draw_sweeps(ax, sweep_paths, color="#0b5fff", endpoints=False)
    save(fig, "03_feasible_region_sweep_modes.png")

    fig, ax = _new_map_axes(config, "04 Region Sweep Patterns")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    _draw_sweeps(ax, sweep_paths, color="#0b5fff", endpoints=False)
    _info_box_outside(
        ax,
        "\n".join(
            [
                f"Delta={_coverage_strip_spacing(config, path_config):.2f}",
                f"Rmin={config.fleet.min_turn_radius:.2f}",
                f"rmin-aware={report.get('rmin_aware_chain_enabled', False)}",
                f"stride={report.get('turn_stride_distribution', {})}",
            ]
        ),
        fontsize=7,
    )
    save(fig, "04_region_sweep_patterns.png")

    fig, ax = _new_map_axes(config, "04 Open Sweep Chain TSP")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    open_chain_seen = False
    for agent_id, agent in sorted(path_plan.agents.items()):
        color = _agent_color(agent_id)
        for segment in agent.segments:
            if segment.metadata.get("open_chain_mode") == "true" or segment.metadata.get("open_chain_connector") == "true":
                open_chain_seen = True
                linestyle = "-" if segment.kind == "cover" else "--"
                linewidth = 2.0 if segment.kind == "cover" else 1.2
                _plot_segment(ax, segment, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.92)
                if segment.metadata.get("open_chain_id") and segment.waypoints:
                    mid = segment.waypoints[len(segment.waypoints) // 2]
                    ax.text(mid.x, mid.y, segment.metadata["open_chain_id"].split("_chain_")[-1], fontsize=5, color=color)
        ax.plot([], [], color=color, label=f"USV {agent_id}")
    if not open_chain_seen:
        ax.text(
            0.5 * config.mission.area_length_x,
            0.5 * config.mission.area_length_y,
            "No open-chain segments selected",
            ha="center",
            va="center",
            fontsize=10,
            color="#374151",
        )
    _info_box_outside(
        ax,
        "\n".join(
            [
                f"regions={report.get('open_chain_region_count', 0)}",
                f"chains={report.get('open_chain_count', 0)}",
                f"breaks={report.get('open_chain_break_count', 0)}",
                f"connected={report.get('open_chain_connected_count', 0)}",
                f"skipped={report.get('open_chain_skipped_count', 0)}",
                f"stride={report.get('turn_stride_distribution', {})}",
                f"180 ok={float(report.get('rmin_180_feasible_ratio', 0.0) or 0.0):.2f}",
                f"single={report.get('single_pass_chain_count', 0)}",
            ]
        ),
        fontsize=7,
    )
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "04_open_sweep_chain_tsp.png")

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

    fig = _shared_resource_timeline_figure(config, obstacle_field, path_plan, report)
    save(fig, "10_shared_resource_timeline.png")

    fig = _repeat_overlap_diagnostics_figure(config, obstacle_field, path_plan, report)
    save(fig, "11_repeat_overlap_diagnostics.png")

    fig = _performance_metric_dashboard_figure(path_plan, report)
    save(fig, "12_performance_metric_dashboard.png")

    fig = _cross_agent_ownership_overlap_figure(config, obstacle_field, regions, path_plan, report, ownership_map)
    save(fig, "13_cross_agent_ownership_overlap.png")

    with (output / "paper_style_region_tsp_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    return output


def _performance_metric_dashboard_figure(
    path_plan: MultiAgentPathPlan,
    report: Dict[str, object],
):
    summary = report.get("performance_summary", {})
    metrics = report.get("metrics", {})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    fig.suptitle("12 Performance Metric Dashboard", fontsize=13)

    ax = axes[0, 0]
    cover = float(metrics.get("coverage_length", 0.0))
    transition = float(metrics.get("transition_length", 0.0))
    ax.bar(["cover", "turn/transit"], [cover, transition], color=["#2563eb", "#f97316"])
    ax.set_ylabel("length [m]")
    ax.set_title("Path Length Composition")
    for idx, value in enumerate([cover, transition]):
        ax.text(idx, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    ax = axes[0, 1]
    ratios = [
        float(summary.get("coverage_length_ratio", 0.0)),
        float(summary.get("transition_length_ratio", 0.0)),
        float(summary.get("repeat_transition_ratio", 0.0)),
        float(summary.get("residual_area_ratio", 0.0)),
    ]
    labels = ["cover ratio", "transition ratio", "repeat/transition", "residual area"]
    colors = ["#2563eb", "#f97316", "#dc2626", "#9333ea"]
    ax.barh(labels, ratios, color=colors)
    ax.set_xlim(0.0, max(1.0, max(ratios, default=0.0) * 1.15))
    ax.set_title("Normalized Performance Ratios")
    for idx, value in enumerate(ratios):
        ax.text(value, idx, f"{value:.3f}", va="center", ha="left", fontsize=8)

    ax = axes[1, 0]
    agent_lengths = summary.get("agent_total_lengths", {})
    if isinstance(agent_lengths, dict) and agent_lengths:
        agent_ids = sorted(agent_lengths, key=lambda item: int(item) if str(item).isdigit() else str(item))
        values = [float(agent_lengths[item]) for item in agent_ids]
        ax.bar([f"USV {item}" for item in agent_ids], values, color=[_agent_color(int(item)) for item in agent_ids])
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("length [m]")
    ax.set_title(f"Agent Load Imbalance = {float(summary.get('agent_load_imbalance', 0.0)):.3f}")

    ax = axes[1, 1]
    ax.axis("off")
    constraint_ok = bool(summary.get("constraint_ok", False))
    lines = [
        f"profile: {summary.get('performance_profile', '')}",
        f"coverage: {float(summary.get('coverage_ratio', 0.0)):.4f} / target {float(summary.get('target_coverage_fraction', 0.0)):.2f}",
        f"target met: {summary.get('target_coverage_met', False)}",
        f"total length: {float(summary.get('total_length', 0.0)):.2f} m",
        f"repeat overlap: {float(summary.get('repeat_overlap_length', 0.0)):.2f} m",
        f"performance objective: {float(summary.get('performance_objective', 0.0)):.2f}",
        f"constraints ok: {constraint_ok}",
        f"invalid/collision/kinematic/dynamic: "
        f"{metrics.get('invalid_path_length', 0.0):.1f}/"
        f"{metrics.get('obstacle_collision_segment_count', 0.0):.0f}/"
        f"{metrics.get('kinematic_infeasible_segment_count', 0.0):.0f}/"
        f"{metrics.get('dynamic_infeasible_segment_count', 0.0):.0f}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


def _cross_agent_ownership_overlap_figure(
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
    regions,
    path_plan: MultiAgentPathPlan,
    report: Dict[str, object],
    ownership_map: CoverageOwnershipMap | None,
):
    fig, ax = _new_map_axes(config, "13 Cross-Agent Ownership Overlap")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    region_owner = ownership_map.region_owner if ownership_map is not None else {}
    for region in regions:
        owner = region_owner.get(region.region_id)
        if owner is None:
            face = "#e5e7eb"
            edge = "#9ca3af"
        else:
            face = _agent_color(owner)
            edge = face
        _draw_polygon(ax, region.polygon, facecolor=face, edgecolor=edge, alpha=0.12, linewidth=0.8)
        ax.text(region.center[0], region.center[1], f"{region.region_id}\nowner={owner}", fontsize=4.8, ha="center", va="center")
    for agent_id, agent in sorted(path_plan.agents.items()):
        base_color = _agent_color(agent_id)
        for segment in agent.segments:
            overlap = _metadata_float(segment.metadata, "cross_agent_overlap_length", 0.0)
            unavoidable = segment.metadata.get("unavoidable_cross_agent_overlap") == "true"
            if overlap > 1e-6:
                color = "#dc2626" if unavoidable else "#f97316"
                linewidth = 2.4
                alpha = 0.92
            else:
                color = base_color
                linewidth = 0.75
                alpha = 0.24
            linestyle = "-" if segment.kind == "cover" else "--" if segment.kind == "transit" else "-."
            _plot_segment(ax, segment, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
            if overlap > 1e-6 and segment.waypoints:
                waypoint = segment.waypoints[len(segment.waypoints) // 2]
                label = f"{segment.kind}:{overlap:.1f}"
                if unavoidable:
                    label += "\nunavoidable"
                ax.text(waypoint.x, waypoint.y, label, fontsize=4.8, color=color)
        ax.plot([], [], color=base_color, label=f"USV {agent_id} owned/path")
    ax.plot([], [], color="#f97316", linewidth=2.4, label="cross-agent overlap")
    ax.plot([], [], color="#dc2626", linewidth=2.4, label="unavoidable overlap")
    _info_box_outside(
        ax,
        "\n".join(
            [
                f"enabled={report.get('cross_agent_penalty_enabled', False)}",
                f"overlap={float(report.get('cross_agent_overlap_length', 0.0)):.3f}",
                f"penalty={float(report.get('cross_agent_penalty_total', 0.0)):.3f}",
                f"by_agent={report.get('cross_agent_overlap_by_agent', {})}",
                f"by_kind={report.get('cross_agent_overlap_by_kind', {})}",
                f"unavoidable={report.get('unavoidable_cross_agent_overlap_count', 0)}",
            ]
        ),
        fontsize=7,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.46), fontsize=7)
    return fig


def _repeat_overlap_diagnostics_figure(
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
    path_plan: MultiAgentPathPlan,
    report: Dict[str, object],
):
    fig, ax = _new_map_axes(config, "11 Repeat-Overlap Diagnostics")
    _draw_obstacles(ax, obstacle_field, raw=False, inflated=True)
    for agent_id, agent in sorted(path_plan.agents.items()):
        base_color = _agent_color(agent_id)
        for segment in agent.segments:
            overlap = _metadata_float(segment.metadata, "repeat_overlap_length", 0.0)
            ratio = _metadata_float(segment.metadata, "repeat_overlap_hit_ratio", 0.0)
            if overlap > 1e-6:
                color = "#d00000" if segment.kind != "cover" and ratio >= 0.25 else "#f97316"
                linewidth = 2.1 if segment.kind != "cover" else 1.5
                alpha = 0.92
            else:
                color = base_color
                linewidth = 0.65
                alpha = 0.22
            linestyle = "-" if segment.kind == "cover" else "--" if segment.kind == "transit" else "-."
            _plot_segment(ax, segment, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
            if overlap > 1e-6 and segment.waypoints:
                waypoint = segment.waypoints[len(segment.waypoints) // 2]
                ax.text(waypoint.x, waypoint.y, f"{segment.kind}:{overlap:.1f}", fontsize=4.8, color=color)
        ax.plot([], [], color=base_color, label=f"USV {agent_id}")
    ax.plot([], [], color="#f97316", linewidth=2.0, label="repeat overlap")
    ax.plot([], [], color="#d00000", linewidth=2.0, label="high repeat turn/transit")
    ax.plot([], [], color="#111827", linestyle="-", label="cover")
    ax.plot([], [], color="#111827", linestyle="--", label="transit")
    ax.plot([], [], color="#111827", linestyle="-.", label="turn")
    _info_box_outside(
        ax,
        "\n".join(
            [
                f"enabled={report.get('main_repeat_path_penalty_enabled', False)}",
                f"overlap={float(report.get('main_repeat_overlap_length', 0.0)):.3f}",
                f"penalty={float(report.get('main_repeat_penalty_total', 0.0)):.3f}",
                f"agent_overlap={report.get('agent_repeat_overlap', {})}",
            ]
        ),
        fontsize=7,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.46), fontsize=7)
    return fig


def _shared_resource_timeline_figure(
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
    path_plan: MultiAgentPathPlan,
    report: Dict[str, object],
):
    windows = collect_resource_windows(path_plan.agents)
    grouped: Dict[str, List] = {}
    for window in windows:
        grouped.setdefault(window.resource_id, []).append(window)
    shared = {
        resource_id: sorted(items, key=lambda item: (item.start, item.end, item.agent_id))
        for resource_id, items in grouped.items()
        if len(items) > 1
    }
    ordered_resources = sorted(
        shared,
        key=lambda item: (shared[item][0].start, item),
    )[:24]
    shared_segment_ids = {
        window.segment_id
        for resource_id in ordered_resources
        for window in shared[resource_id]
    }
    separation_time = float(report.get("resource_separation_time", 0.0) or 0.0)

    fig = plt.figure(figsize=(12.5, 6.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35], wspace=0.28)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_time = fig.add_subplot(grid[0, 1])

    ax_map.set_title("Shared Spatial Resources")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlim(0.0, config.mission.area_length_x)
    ax_map.set_ylim(0.0, config.mission.area_length_y)
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.grid(True, linewidth=0.35, alpha=0.35)
    _draw_obstacles(ax_map, obstacle_field, raw=False, inflated=True)
    for agent_id, agent in sorted(path_plan.agents.items()):
        color = _agent_color(agent_id)
        for segment in agent.segments:
            if segment.segment_id in shared_segment_ids:
                _plot_segment(ax_map, segment, color="#f97316", linestyle="-", linewidth=2.5, alpha=0.9)
            else:
                _plot_segment(ax_map, segment, color=color, linestyle="-" if segment.kind == "cover" else "--", linewidth=0.65, alpha=0.2)
        ax_map.plot([], [], color=color, label=f"USV {agent_id}")
    ax_map.plot([], [], color="#f97316", linewidth=2.5, label="shared resource")
    ax_map.legend(loc="upper right", fontsize=7)

    ax_time.set_title("Resource Time Windows")
    if not ordered_resources:
        ax_time.axis("off")
        ax_time.text(
            0.5,
            0.5,
            "No shared timed resources",
            transform=ax_time.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        return fig

    y_positions = {resource_id: idx for idx, resource_id in enumerate(ordered_resources)}
    conflict_pairs = _timeline_conflict_pairs(shared, ordered_resources, separation_time)
    conflict_windows = {
        (resource_id, first.segment_id)
        for resource_id, first, _ in conflict_pairs
    } | {
        (resource_id, second.segment_id)
        for resource_id, _, second in conflict_pairs
    }
    for resource_id in ordered_resources:
        y = y_positions[resource_id]
        for window in shared[resource_id]:
            width = max(window.end - window.start, 1e-6)
            is_conflict = (resource_id, window.segment_id) in conflict_windows
            ax_time.barh(
                y,
                width,
                left=window.start,
                height=0.55,
                color=_agent_color(window.agent_id),
                edgecolor="#d00000" if is_conflict else "#17803d",
                linewidth=1.6 if is_conflict else 0.8,
                alpha=0.88,
            )
            ax_time.text(
                window.start + width / 2.0,
                y,
                f"A{window.agent_id}",
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )
    for resource_id, first, second in conflict_pairs:
        y = y_positions[resource_id]
        ax_time.plot([second.start, second.start], [y - 0.4, y + 0.4], color="#d00000", linewidth=1.2)
        ax_time.text(second.start, y + 0.42, "conflict", color="#d00000", fontsize=5.5, ha="center")

    labels = [_short_resource_label(resource_id) for resource_id in ordered_resources]
    ax_time.set_yticks(list(y_positions.values()), labels=labels, fontsize=6)
    ax_time.set_xlabel("time [s]")
    ax_time.grid(True, axis="x", linewidth=0.35, alpha=0.35)
    ax_time.invert_yaxis()
    metrics_text = (
        f"shared={report.get('shared_resource_count', 0)}\n"
        f"space reuse={report.get('spatial_overlap_reuse_count', 0)}\n"
        f"true conflicts={report.get('true_time_conflict_count', 0)}\n"
        f"resolved={report.get('mapf_conflicts_resolved_after_residual', 0)}"
    )
    ax_time.text(
        1.02,
        0.98,
        metrics_text,
        transform=ax_time.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc"},
    )
    return fig


def _timeline_conflict_pairs(
    shared: Dict[str, List],
    ordered_resources: Sequence[str],
    separation_time: float,
) -> List[Tuple[str, object, object]]:
    conflicts = []
    for resource_id in ordered_resources:
        windows = shared[resource_id]
        for first, second in zip(windows[:-1], windows[1:]):
            if second.start < first.end + max(separation_time, 0.0) - 1e-9:
                conflicts.append((resource_id, first, second))
    return conflicts


def _short_resource_label(resource_id: str) -> str:
    if len(resource_id) <= 42:
        return resource_id
    head = resource_id[:18]
    tail = resource_id[-20:]
    return f"{head}...{tail}"


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
