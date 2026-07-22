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
from ..schema import PlannerConfig, Pose2D, VehicleFootprint
from .aco import AcoTspResult, solve_aco_tsp_cpp, validate_tsp_solver
from .assignment import assign_heterogeneous_connected_regions, apply_lightweight_load_swap, balance_region_workload
from .decomposition import (
    build_composite_free_space_regions,
    build_free_space_cells,
    build_large_convex_free_space_regions,
    decompose_obstacle_aware_area,
    decompose_rectangular_area,
)
from .dynamics_validation import dynamic_edge_cost, validate_transition_dynamics, validate_transition_sequence
from .graph import build_region_graph, graph_is_connected
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
    BalancedAssignment,
    CoveragePass,
    CoverageOwnershipMap,
    MultiAgentPathPlan,
    ObstacleField,
    OpenSweepBreak,
    OpenSweepChain,
    PathPlanningConfig,
    PathSegmentSpec,
    DecomposedRegion,
    CompositeFreeSpaceRegion,
    FreeSpaceCell,
    RegionCoveragePattern,
    RegionGraph,
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
    heterogeneous_mode = bool(
        path_config.enable_heterogeneous_connected_assignment
        and config.agent_profiles
    )
    decomposition_config = _heterogeneous_decomposition_config(config) if heterogeneous_mode else config
    decomposition_path_config = (
        replace(path_config, large_region_max_area_fraction=1.0)
        if heterogeneous_mode
        else path_config
    )
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
    obstacle_field = (
        normalize_obstacle_field(static_obstacles, decomposition_config, decomposition_path_config)
        if static_obstacles
        else None
    )
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
        regions = build_large_convex_free_space_regions(decomposition_config, decomposition_path_config, obstacle_field)
        large_convex_mode_enabled = bool(
            regions
            and any(region.metadata.get("convex_region_decomposition") == "true" for region in regions)
        )
        if not large_convex_mode_enabled:
            regions = []
    if large_convex_mode_enabled:
        composite_mode_enabled = False
    elif composite_mode_enabled:
        raw_free_cells = build_free_space_cells(decomposition_config, decomposition_path_config, obstacle_field)
        regions = build_composite_free_space_regions(
            raw_free_cells,
            decomposition_config,
            decomposition_path_config,
            obstacle_field,
        )
        if not regions:
            regions = decompose_obstacle_aware_area(decomposition_config, decomposition_path_config, obstacle_field)
            composite_mode_enabled = False
    else:
        regions = (
            decompose_obstacle_aware_area(decomposition_config, decomposition_path_config, obstacle_field)
            if obstacle_field is not None and obstacle_field.inflated_obstacles
            else (
                [_heterogeneous_full_mission_region(config)]
                if heterogeneous_mode
                else decompose_rectangular_area(config, path_config)
            )
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
    coverage_merge_diagnostics: Dict[str, object] = {}
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
    coverage_merge_input_count = len(regions)
    large_convex_region_count = sum(
        1 for region in regions if region.metadata.get("convex_region_decomposition") == "true"
    )
    skip_pre_assignment_coverage_merge = (
        large_convex_mode_enabled
        and int(path_config.coverage_merge_skip_pre_assignment_large_region_count) > 0
        and large_convex_region_count >= int(path_config.coverage_merge_skip_pre_assignment_large_region_count)
    )
    if (
        path_config.enable_coverage_aware_merge
        and obstacle_field is not None
        and obstacle_field.inflated_obstacles
        and regions
    ):
        if skip_pre_assignment_coverage_merge:
            coverage_merge_diagnostics.update(
                {
                    "coverage_merge_status": "skipped_large_convex_pre_assignment",
                    "coverage_merge_budget_exhausted": False,
                    "coverage_merge_budget_reason": "agent_task_merge_preferred",
                    "coverage_merge_validation_count": 0,
                    "coverage_merge_candidate_count": 0,
                    "coverage_merge_accepted_count": 0,
                }
            )
        else:
            coverage_merge_fallback_regions = list(regions)
            regions = _coverage_aware_merge_regions(
                regions,
                config,
                path_config,
                obstacle_field=obstacle_field,
                diagnostics=coverage_merge_diagnostics,
                progress_callback=lambda **extra: emit_region_progress(
                    "coverage_aware_merge_iteration",
                    stage_started,
                    **extra,
                ),
            )
            if len(regions) < len(coverage_merge_fallback_regions):
                coarsened_regions = coverage_merge_fallback_regions
    finish_stage(
        "region_coarsen_merge",
        stage_started,
        coarsened_region_count=len(coarsened_regions) if coarsened_regions is not None else len(regions),
        merged_region_count=len(regions),
        composite_region_count=len(regions) if composite_mode_enabled else 0,
        composite_member_cell_count=sum(len(getattr(region, "member_cells", []) or []) for region in regions),
        merge_rejected_count=sum(merge_diagnostics.values()),
        merge_rejected_by_reason=dict(merge_diagnostics),
        coverage_aware_merge_enabled=bool(path_config.enable_coverage_aware_merge),
        coverage_merge_region_count_before=coverage_merge_input_count,
        coverage_merge_region_count_after=len(regions),
        coverage_merge_status=str(coverage_merge_diagnostics.get("coverage_merge_status", "disabled")),
        coverage_merge_budget_exhausted=bool(coverage_merge_diagnostics.get("coverage_merge_budget_exhausted", False)),
        coverage_merge_budget_reason=str(coverage_merge_diagnostics.get("coverage_merge_budget_reason", "")),
        coverage_merge_validation_count=int(coverage_merge_diagnostics.get("coverage_merge_validation_count", 0) or 0),
        coverage_merge_candidate_count=int(coverage_merge_diagnostics.get("coverage_merge_candidate_count", 0) or 0),
        coverage_merge_accepted_count=int(coverage_merge_diagnostics.get("coverage_merge_accepted_count", 0) or 0),
    )

    stage_started = time.perf_counter()
    raw_patterns = (
        {}
        if heterogeneous_mode
        else _generate_paper_style_patterns(
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
    )
    finish_stage(
        "coverage_pattern_generation",
        stage_started,
        region_count=len(regions),
        raw_pattern_count=sum(len(items) for items in raw_patterns.values()),
    )

    stage_started = time.perf_counter()
    if heterogeneous_mode:
        sweep_paths = {}
        feasible_patterns = {}
        infeasible_regions = []
        sweep_segment_templates = {}
    else:
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
    if not heterogeneous_mode and path_config.enable_infeasible_uturn_region_repair and large_convex_mode_enabled and infeasible_regions:
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
    if not heterogeneous_mode and path_config.enable_infeasible_uturn_region_repair and composite_mode_enabled and infeasible_regions:
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
        and not heterogeneous_mode
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
    controlled_split_records: List[Dict[str, object]] = []
    if heterogeneous_mode:
        stage_started = time.perf_counter()
        regions, controlled_split_records = _split_oversized_heterogeneous_regions(
            regions,
            config,
            path_config,
        )
        if controlled_split_records:
            _populate_region_neighbors(regions)
        finish_stage(
            "heterogeneous_oversized_region_split",
            stage_started,
            split_parent_count=len(controlled_split_records),
            split_child_count=sum(int(record.get("child_count", 0)) for record in controlled_split_records),
            resulting_region_count=len(regions),
        )

    agent_feasible_patterns: Dict[int, Dict[str, List[RegionCoveragePattern]]] = {}
    agent_sweep_paths: Dict[int, Dict[str, RegionSweepPath]] = {}
    agent_sweep_segment_templates: Dict[int, Dict[str, Tuple[List[PathSegmentSpec], str]]] = {}
    agent_obstacle_fields: Dict[int, ObstacleField | None] = {}
    heterogeneous_candidate_diagnostics: Dict[str, object] = {
        "enabled": heterogeneous_mode,
        "agent_feasible_region_counts": {},
        "agent_pattern_counts": {},
        "agent_infeasible_regions": {},
    }
    if heterogeneous_mode:
        stage_started = time.perf_counter()
        for agent_id in range(config.fleet.num_agents or len(config.fleet.initial_states_3dof)):
            agent_config = config.for_agent(agent_id)
            agent_field = normalize_obstacle_field(static_obstacles, agent_config, path_config) if static_obstacles else None
            agent_obstacle_fields[agent_id] = agent_field
            agent_raw_patterns = _generate_paper_style_patterns(
                regions,
                agent_config,
                replace(
                    path_config,
                    max_oriented_sweep_angles_per_region=min(
                        path_config.max_oriented_sweep_angles_per_region,
                        max(path_config.max_agent_pattern_previews_per_region, 1),
                    ),
                ),
                agent_field,
                progress_callback=lambda agent_id=agent_id, **extra: emit_region_progress(
                    "heterogeneous_pattern_generation_region",
                    stage_started,
                    agent_id=agent_id,
                    **extra,
                ),
            )
            agent_stats: Dict[str, object] = {}
            (
                agent_sweep_paths[agent_id],
                agent_feasible_patterns[agent_id],
                agent_infeasible,
                agent_sweep_segment_templates[agent_id],
            ) = _build_region_sweep_paths(
                agent_raw_patterns,
                agent_config,
                path_config,
                agent_field,
                stats=agent_stats,
                progress_callback=lambda agent_id=agent_id, **extra: emit_region_progress(
                    "heterogeneous_sweep_validation_region",
                    stage_started,
                    agent_id=agent_id,
                    **extra,
                ),
            )
            heterogeneous_candidate_diagnostics["agent_feasible_region_counts"][str(agent_id)] = len(
                agent_feasible_patterns[agent_id]
            )
            heterogeneous_candidate_diagnostics["agent_pattern_counts"][str(agent_id)] = sum(
                len(items) for items in agent_feasible_patterns[agent_id].values()
            )
            heterogeneous_candidate_diagnostics["agent_infeasible_regions"][str(agent_id)] = agent_infeasible

        union_patterns: Dict[str, List[RegionCoveragePattern]] = {}
        union_paths: Dict[str, RegionSweepPath] = {}
        union_templates: Dict[str, Tuple[List[PathSegmentSpec], str]] = {}
        for agent_id in sorted(agent_feasible_patterns):
            for region_id, candidates in agent_feasible_patterns[agent_id].items():
                union_patterns.setdefault(region_id, []).extend(candidates)
                if region_id not in union_paths and region_id in agent_sweep_paths[agent_id]:
                    union_paths[region_id] = agent_sweep_paths[agent_id][region_id]
            union_templates.update(agent_sweep_segment_templates[agent_id])
        feasible_patterns = union_patterns
        sweep_paths = union_paths
        sweep_segment_templates = union_templates
        infeasible_regions = [
            {
                "region_id": region.region_id,
                "reason": "no_agent_has_feasible_sweep",
                "agent_failure_reasons": {
                    str(agent_id): next(
                        (
                            item.get("reasons", item.get("reason", "unknown"))
                            for item in heterogeneous_candidate_diagnostics["agent_infeasible_regions"][str(agent_id)]
                            if item.get("region_id") == region.region_id
                        ),
                        "unknown",
                    )
                    for agent_id in agent_feasible_patterns
                },
            }
            for region in regions
            if region.region_id not in feasible_patterns
        ]
        finish_stage(
            "heterogeneous_agent_region_candidates",
            stage_started,
            agent_count=len(agent_feasible_patterns),
            union_feasible_region_count=len(feasible_patterns),
            union_pattern_count=sum(len(items) for items in feasible_patterns.values()),
        )

    feasible_regions = [region for region in regions if region.region_id in feasible_patterns]
    stage_started = time.perf_counter()
    graph = build_region_graph(feasible_regions, feasible_patterns, config, obstacle_field=obstacle_field, path_config=path_config)
    finish_stage("region_graph_building", stage_started, feasible_region_count=len(feasible_regions), edge_count=len(graph.edge_weights))
    stage_started = time.perf_counter()
    assignment = (
        assign_heterogeneous_connected_regions(graph, config, agent_feasible_patterns, path_config)
        if heterogeneous_mode
        else balance_region_workload(graph, config)
    )
    assignment_strategy_diagnostics = dict(assignment.diagnostics)
    assignment_strategy_loads = dict(assignment.loads)
    load_swap_workload_weights = _region_workload_weights(graph, config, path_config)
    load_swap_before = _joint_imbalance_ratio(
        assignment.agent_regions,
        _joint_region_loads(assignment.agent_regions, weights=load_swap_workload_weights),
    )
    if path_config.enable_lightweight_load_swap and not heterogeneous_mode:
        assignment = apply_lightweight_load_swap(
            assignment,
            graph,
            max_iterations=path_config.load_swap_max_iterations,
            workload_weights=load_swap_workload_weights,
        )
    load_swap_count = int(assignment.diagnostics.get("load_swap_count", "0") or 0)
    load_swap_candidate_count = int(assignment.diagnostics.get("load_swap_candidate_count", "0") or 0)
    load_swap_reject_reasons = assignment.diagnostics.get("load_swap_reject_reasons", "")
    load_swap_after = assignment.imbalance_ratio
    finish_stage(
        "load_balancing_assignment",
        stage_started,
        agent_count=len(assignment.agent_regions),
        load_swap_count=load_swap_count,
        load_swap_candidate_count=load_swap_candidate_count,
        load_swap_before_imbalance=round(float(load_swap_before), 6),
        load_swap_after_imbalance=round(float(load_swap_after), 6),
        heterogeneous_assignment=heterogeneous_mode,
        assignment_status=assignment.diagnostics.get("status", "complete"),
    )

    if heterogeneous_mode:
        feasible_patterns = {
            region_id: list(agent_feasible_patterns[agent_id][region_id])
            for agent_id, region_ids in assignment.agent_regions.items()
            for region_id in region_ids
            if region_id in agent_feasible_patterns.get(agent_id, {})
        }
        sweep_paths = {
            region_id: agent_sweep_paths[agent_id][region_id]
            for agent_id, region_ids in assignment.agent_regions.items()
            for region_id in region_ids
            if region_id in agent_sweep_paths.get(agent_id, {})
        }
        sweep_segment_templates = {
            key: value
            for agent_id, templates in agent_sweep_segment_templates.items()
            for key, value in templates.items()
            if any(region_id in assignment.agent_regions.get(agent_id, []) for region_id in feasible_patterns)
        }
        graph = build_region_graph(
            feasible_regions,
            feasible_patterns,
            config,
            obstacle_field=obstacle_field,
            path_config=path_config,
        )

    pre_agent_task_feasible_regions = list(feasible_regions)
    pre_agent_task_feasible_patterns = {region_id: list(patterns) for region_id, patterns in feasible_patterns.items()}
    pre_agent_task_sweep_paths = dict(sweep_paths)
    pre_agent_task_sweep_segment_templates = {
        key: (copy.deepcopy(value[0]), value[1])
        for key, value in sweep_segment_templates.items()
    }

    agent_task_merge_diagnostics: Dict[str, object] = {
        "agent_task_merge_enabled": bool(path_config.enable_agent_task_region_merge),
        "agent_task_merge_status": "disabled",
        "agent_task_merge_region_count_before": len(feasible_regions),
        "agent_task_merge_region_count_after": len(feasible_regions),
        "agent_task_merge_candidate_count": 0,
        "agent_task_merge_accepted_count": 0,
        "agent_task_strip_merge_enabled": bool(path_config.enable_agent_task_lightweight_strip_merge),
        "agent_task_strip_candidate_count": 0,
        "agent_task_strip_accepted_count": 0,
        "agent_task_strip_budget_exhausted": False,
        "agent_task_strip_budget_reason": "",
        "agent_task_strip_elapsed_sec": 0.0,
        "agent_task_strip_rejected_by_reason": {},
        "agent_task_unified_merge_enabled": bool(path_config.agent_task_merge_enable_unified_group_merge),
        "agent_task_unified_candidate_count": 0,
        "agent_task_unified_accepted_count": 0,
        "agent_task_merge_rejected_by_reason": {},
        "agent_task_merge_rejected_candidates": [],
        "agent_task_merge_regions": [],
        "agent_task_merge_split_count": 0,
        "agent_task_merge_unstable_region_count": 0,
        "agent_task_merge_unstable_regions": [],
        "agent_task_merge_pairwise_fallback_enabled": bool(path_config.agent_task_merge_enable_pairwise_fallback),
        "agent_task_merge_min_unified_rectangularity": float(path_config.agent_task_merge_min_unified_rectangularity),
        "agent_task_merge_max_unified_group_size": int(path_config.agent_task_merge_max_unified_group_size),
    }
    if heterogeneous_mode and path_config.enable_agent_task_region_merge:
        stage_started = time.perf_counter()
        (
            feasible_regions,
            feasible_patterns,
            sweep_paths,
            sweep_segment_templates,
            assignment,
            agent_task_merge_diagnostics,
        ) = _heterogeneous_monotone_task_recombination(
            feasible_regions,
            assignment,
            agent_feasible_patterns,
            agent_sweep_paths,
            agent_sweep_segment_templates,
            config,
            path_config,
            agent_obstacle_fields,
            progress_callback=lambda **extra: emit_region_progress(
                "heterogeneous_monotone_merge_progress",
                stage_started,
                **extra,
            ),
        )
        graph = build_region_graph(
            feasible_regions,
            feasible_patterns,
            config,
            obstacle_field=obstacle_field,
            path_config=path_config,
        )
        finish_stage(
            "heterogeneous_monotone_task_recombination",
            stage_started,
            status=agent_task_merge_diagnostics.get("agent_task_merge_status", "unknown"),
            candidate_count=int(agent_task_merge_diagnostics.get("agent_task_merge_candidate_count", 0) or 0),
            accepted_count=int(agent_task_merge_diagnostics.get("agent_task_merge_accepted_count", 0) or 0),
            region_count_after=len(feasible_regions),
        )
    if (path_config.enable_agent_task_region_merge or path_config.enable_agent_task_lightweight_strip_merge) and not heterogeneous_mode:
        stage_started = time.perf_counter()
        (
            candidate_regions,
            candidate_agent_regions,
            agent_task_merge_diagnostics,
        ) = _merge_assigned_agent_task_regions(
            feasible_regions,
            assignment.agent_regions,
            config,
            path_config,
            obstacle_field,
            progress_callback=lambda **extra: emit_region_progress(
                "agent_task_region_merge_progress",
                stage_started,
                **extra,
            ),
        )
        accepted_agent_task_merges = int(agent_task_merge_diagnostics.get("agent_task_merge_accepted_count", 0) or 0)
        if accepted_agent_task_merges > 0:
            candidate_raw_patterns = _generate_paper_style_patterns(
                candidate_regions,
                config,
                path_config,
                obstacle_field,
                progress_callback=lambda **extra: emit_region_progress(
                    "agent_task_merge_pattern_generation_region",
                    stage_started,
                    **extra,
                ),
            )
            candidate_sweep_stats: Dict[str, object] = {}
            (
                candidate_sweep_paths,
                candidate_feasible_patterns,
                candidate_infeasible_regions,
                candidate_sweep_segment_templates,
            ) = _build_region_sweep_paths(
                candidate_raw_patterns,
                config,
                path_config,
                obstacle_field,
                stats=candidate_sweep_stats,
                progress_callback=lambda **extra: emit_region_progress(
                    "agent_task_merge_build_sweep_paths_region",
                    stage_started,
                    **extra,
                ),
            )
            missing_region_ids = [
                region_id
                for region_ids in candidate_agent_regions.values()
                for region_id in region_ids
                if region_id not in candidate_feasible_patterns
            ]
            unstable_region_ids, unstable_region_records = _agent_task_merge_unstable_region_ids(
                candidate_regions,
                candidate_feasible_patterns,
                path_config,
                config=config,
                obstacle_field=obstacle_field,
            )
            if unstable_region_records:
                agent_task_merge_diagnostics["agent_task_merge_unstable_region_count"] = len(unstable_region_records)
                agent_task_merge_diagnostics["agent_task_merge_unstable_regions"] = unstable_region_records
            repair_region_ids = sorted({*missing_region_ids, *unstable_region_ids})
            if repair_region_ids:
                (
                    split_candidate_regions,
                    split_candidate_agent_regions,
                    split_count,
                ) = _split_infeasible_agent_task_merge_candidates(
                    candidate_regions,
                    candidate_agent_regions,
                    feasible_regions,
                    repair_region_ids,
                )
                if split_count > 0:
                    split_raw_patterns: Dict[str, List[RegionCoveragePattern]] = {}
                    split_sweep_paths: Dict[str, RegionSweepPath] = {}
                    split_feasible_patterns: Dict[str, List[RegionCoveragePattern]] = {}
                    split_sweep_segment_templates: Dict[str, Tuple[List[PathSegmentSpec], str]] = {}
                    split_infeasible_regions: List[Dict[str, object]] = []
                    split_regions_to_build = []
                    reused_candidate_count = 0
                    reused_base_count = 0
                    for split_region in split_candidate_regions:
                        region_id = split_region.region_id
                        if region_id in candidate_feasible_patterns:
                            source_raw = candidate_raw_patterns
                            source_paths = candidate_sweep_paths
                            source_patterns = candidate_feasible_patterns
                            source_templates = candidate_sweep_segment_templates
                            reused_candidate_count += 1
                        elif region_id in feasible_patterns:
                            source_raw = raw_patterns
                            source_paths = sweep_paths
                            source_patterns = feasible_patterns
                            source_templates = sweep_segment_templates
                            reused_base_count += 1
                        else:
                            split_regions_to_build.append(split_region)
                            continue
                        split_raw_patterns[region_id] = list(source_raw.get(region_id, []))
                        split_sweep_paths[region_id] = source_paths[region_id]
                        split_feasible_patterns[region_id] = list(source_patterns[region_id])
                        for pattern in source_patterns[region_id]:
                            template_key = _pattern_template_key(pattern)
                            if template_key in source_templates:
                                split_sweep_segment_templates[template_key] = source_templates[template_key]

                    split_sweep_stats: Dict[str, object] = {}
                    if split_regions_to_build:
                        rebuilt_raw_patterns = _generate_paper_style_patterns(
                            split_regions_to_build,
                            config,
                            path_config,
                            obstacle_field,
                            progress_callback=lambda **extra: emit_region_progress(
                                "agent_task_merge_split_pattern_generation_region",
                                stage_started,
                                **extra,
                            ),
                        )
                        (
                            rebuilt_sweep_paths,
                            rebuilt_feasible_patterns,
                            split_infeasible_regions,
                            rebuilt_sweep_segment_templates,
                        ) = _build_region_sweep_paths(
                            rebuilt_raw_patterns,
                            config,
                            path_config,
                            obstacle_field,
                            stats=split_sweep_stats,
                            progress_callback=lambda **extra: emit_region_progress(
                                "agent_task_merge_split_build_sweep_paths_region",
                                stage_started,
                                **extra,
                            ),
                        )
                        split_raw_patterns.update(rebuilt_raw_patterns)
                        split_sweep_paths.update(rebuilt_sweep_paths)
                        split_feasible_patterns.update(rebuilt_feasible_patterns)
                        split_sweep_segment_templates.update(rebuilt_sweep_segment_templates)
                    split_sweep_stats.update(
                        {
                            "reused_candidate_region_count": reused_candidate_count,
                            "reused_base_region_count": reused_base_count,
                            "rebuilt_region_count": len(split_regions_to_build),
                        }
                    )
                    split_missing_region_ids = [
                        region_id
                        for region_ids in split_candidate_agent_regions.values()
                        for region_id in region_ids
                        if region_id not in split_feasible_patterns
                    ]
                    if not split_missing_region_ids:
                        regions = list(split_candidate_regions)
                        raw_patterns = split_raw_patterns
                        sweep_paths = split_sweep_paths
                        feasible_patterns = split_feasible_patterns
                        infeasible_regions = split_infeasible_regions
                        sweep_segment_templates = split_sweep_segment_templates
                        for key, value in split_sweep_stats.items():
                            sweep_build_stats[f"agent_task_merge_split_{key}"] = value
                        feasible_regions = [region for region in regions if region.region_id in feasible_patterns]
                        graph = build_region_graph(
                            feasible_regions,
                            feasible_patterns,
                            config,
                            obstacle_field=obstacle_field,
                            path_config=path_config,
                        )
                        assignment = _joint_assignment_from_regions(split_candidate_agent_regions, graph)
                        agent_task_merge_diagnostics["agent_task_merge_status"] = "success_partial_split"
                        agent_task_merge_diagnostics["agent_task_merge_region_count_after"] = len(feasible_regions)
                        agent_task_merge_diagnostics["agent_task_merge_split_count"] = split_count
                        agent_task_merge_diagnostics["agent_task_merge_missing_region_ids"] = sorted(missing_region_ids)
                    else:
                        agent_task_merge_diagnostics["agent_task_merge_status"] = "fallback_infeasible"
                        agent_task_merge_diagnostics["agent_task_merge_missing_region_ids"] = sorted(split_missing_region_ids)
                        agent_task_merge_diagnostics["agent_task_merge_infeasible_regions"] = split_infeasible_regions
                        agent_task_merge_diagnostics["agent_task_merge_split_count"] = split_count
                else:
                    agent_task_merge_diagnostics["agent_task_merge_status"] = "fallback_infeasible"
                    agent_task_merge_diagnostics["agent_task_merge_missing_region_ids"] = sorted(repair_region_ids)
                    agent_task_merge_diagnostics["agent_task_merge_infeasible_regions"] = candidate_infeasible_regions
            else:
                regions = list(candidate_regions)
                raw_patterns = candidate_raw_patterns
                sweep_paths = candidate_sweep_paths
                feasible_patterns = candidate_feasible_patterns
                infeasible_regions = candidate_infeasible_regions
                sweep_segment_templates = candidate_sweep_segment_templates
                for key, value in candidate_sweep_stats.items():
                    sweep_build_stats[f"agent_task_merge_{key}"] = value
                feasible_regions = [region for region in regions if region.region_id in feasible_patterns]
                graph = build_region_graph(
                    feasible_regions,
                    feasible_patterns,
                    config,
                    obstacle_field=obstacle_field,
                    path_config=path_config,
                )
                assignment = _joint_assignment_from_regions(candidate_agent_regions, graph)
                agent_task_merge_diagnostics["agent_task_merge_status"] = "success"
                agent_task_merge_diagnostics["agent_task_merge_region_count_after"] = len(feasible_regions)
        finish_stage(
            "agent_task_region_merge",
            stage_started,
            status=agent_task_merge_diagnostics.get("agent_task_merge_status", "unknown"),
            region_count_before=int(agent_task_merge_diagnostics.get("agent_task_merge_region_count_before", 0) or 0),
            region_count_after=int(agent_task_merge_diagnostics.get("agent_task_merge_region_count_after", 0) or 0),
            candidate_count=int(agent_task_merge_diagnostics.get("agent_task_merge_candidate_count", 0) or 0),
            accepted_count=int(agent_task_merge_diagnostics.get("agent_task_merge_accepted_count", 0) or 0),
            strip_candidate_count=int(agent_task_merge_diagnostics.get("agent_task_strip_candidate_count", 0) or 0),
            strip_accepted_count=int(agent_task_merge_diagnostics.get("agent_task_strip_accepted_count", 0) or 0),
            strip_budget_exhausted=bool(
                agent_task_merge_diagnostics.get("agent_task_strip_budget_exhausted", False)
            ),
            strip_budget_reason=str(agent_task_merge_diagnostics.get("agent_task_strip_budget_reason", "")),
            unified_candidate_count=int(agent_task_merge_diagnostics.get("agent_task_unified_candidate_count", 0) or 0),
            unified_accepted_count=int(agent_task_merge_diagnostics.get("agent_task_unified_accepted_count", 0) or 0),
            split_count=int(agent_task_merge_diagnostics.get("agent_task_merge_split_count", 0) or 0),
            unstable_region_count=int(
                agent_task_merge_diagnostics.get("agent_task_merge_unstable_region_count", 0) or 0
            ),
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

    joint_optimizer_diagnostics: Dict[str, object] = {
        "joint_optimizer_status": "disabled",
        "global_joint_objective": 0.0,
        "cross_agent_connector_overlap_length": 0.0,
        "cross_agent_crossing_count": 0,
        "joint_improvement_iteration_count": 0,
    }
    joint_precomputed_results: Dict[int, Dict[str, object]] | None = None
    if path_config.enable_joint_region_candidate_optimization and not heterogeneous_mode:
        stage_started = time.perf_counter()
        assignment, ownership_map, joint_precomputed_results, joint_optimizer_diagnostics = _optimize_joint_region_candidate_assignment(
            assignment=assignment,
            graph=graph,
            feasible_regions=feasible_regions,
            feasible_patterns=feasible_patterns,
            sweep_paths=sweep_paths,
            sweep_segment_templates=sweep_segment_templates,
            config=config,
            path_config=path_config,
            obstacle_field=obstacle_field,
            progress_callback=lambda **extra: emit_region_progress(
                "joint_region_candidate_optimization_progress",
                stage_started,
                **extra,
            ),
        )
        finish_stage(
            "joint_region_candidate_optimization",
            stage_started,
            status=joint_optimizer_diagnostics.get("joint_optimizer_status", "unknown"),
            improvement_count=int(joint_optimizer_diagnostics.get("joint_improvement_iteration_count", 0) or 0),
            global_joint_objective=round(float(joint_optimizer_diagnostics.get("global_joint_objective", 0.0) or 0.0), 6),
            joint_mission_makespan=round(float(joint_optimizer_diagnostics.get("joint_mission_makespan", 0.0) or 0.0), 6),
        )
    elif heterogeneous_mode:
        joint_optimizer_diagnostics["joint_optimizer_status"] = "delegated_to_heterogeneous_connected_assignment"

    agents: Dict[int, AgentPathPlan] = {}
    tours: Dict[int, SingleUsvTourPlan] = {}
    tsp_records: Dict[int, Dict[str, object]] = {}
    infeasible_edges: List[Dict[str, object]] = []
    agent_task_runtime_source_fallback_diagnostics: Dict[str, object] = {
        "enabled": False,
        "candidate_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "affected_agent_count": 0,
        "fallback_records": [],
        "accepted_records": [],
        "rejected_records": [],
    }
    large_map_tsp_phase_started = time.perf_counter()
    large_map_tsp_total_time_budget = (
        max(float(path_config.large_map_tsp_total_time_budget_sec), 0.0)
        if _large_map_mode_enabled(config, path_config)
        else 0.0
    )
    for agent_id, region_ids in sorted(assignment.agent_regions.items()):
        stage_started = time.perf_counter()
        agent_config = config.for_agent(agent_id) if heterogeneous_mode else config
        agent_field = agent_obstacle_fields.get(agent_id, obstacle_field)
        if joint_precomputed_results is not None and agent_id in joint_precomputed_results:
            result = copy.deepcopy(joint_precomputed_results[agent_id])
        else:
            effective_path_config = path_config
            if large_map_tsp_total_time_budget > 0.0:
                elapsed_tsp_phase = time.perf_counter() - large_map_tsp_phase_started
                remaining_tsp_budget = large_map_tsp_total_time_budget - elapsed_tsp_phase
                if remaining_tsp_budget <= 1.0:
                    result = _large_map_tsp_budget_fallback_result(
                        agent_id=agent_id,
                        region_ids=region_ids,
                        patterns=feasible_patterns,
                        path_config=path_config,
                        reason="large_map_tsp_total_time_budget_exhausted",
                        elapsed_sec=elapsed_tsp_phase,
                        total_budget_sec=large_map_tsp_total_time_budget,
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
                    metrics = _agent_metrics(segments, agent_config, agent_field)
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
                    tsp_records[agent_id] = _tsp_record_from_result(
                        agent_id,
                        region_ids,
                        result,
                        sweep_paths,
                    )
                    continue
                effective_path_config = replace(
                    path_config,
                    large_map_tsp_agent_time_budget_sec=min(
                        float(path_config.large_map_tsp_agent_time_budget_sec),
                        max(1.0, remaining_tsp_budget),
                    ),
                )
            result = _solve_agent_region_tsp(
                agent_id,
                region_ids,
                feasible_patterns,
                sweep_paths,
                sweep_segment_templates,
                agent_config,
                effective_path_config,
                agent_field,
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
        metrics = _agent_metrics(segments, agent_config, agent_field)
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
        tsp_records[agent_id] = _tsp_record_from_result(agent_id, region_ids, result, sweep_paths)

    stage_started = time.perf_counter()
    (
        feasible_regions,
        feasible_patterns,
        sweep_paths,
        sweep_segment_templates,
        assignment,
        ownership_map,
        agent_task_runtime_source_fallback_diagnostics,
    ) = _apply_agent_task_merge_runtime_source_fallback(
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        current_regions=feasible_regions,
        current_feasible_patterns=feasible_patterns,
        current_sweep_paths=sweep_paths,
        current_sweep_segment_templates=sweep_segment_templates,
        base_regions=pre_agent_task_feasible_regions,
        base_feasible_patterns=pre_agent_task_feasible_patterns,
        base_sweep_paths=pre_agent_task_sweep_paths,
        base_sweep_segment_templates=pre_agent_task_sweep_segment_templates,
        assignment=assignment,
        ownership_map=ownership_map,
        agents=agents,
        tours=tours,
        tsp_records=tsp_records,
        infeasible_edges=infeasible_edges,
    )
    if int(agent_task_runtime_source_fallback_diagnostics.get("accepted_count", 0) or 0) > 0:
        regions = list(feasible_regions)
    finish_stage(
        "agent_task_merge_runtime_source_fallback",
        stage_started,
        candidate_count=int(agent_task_runtime_source_fallback_diagnostics.get("candidate_count", 0) or 0),
        accepted_count=int(agent_task_runtime_source_fallback_diagnostics.get("accepted_count", 0) or 0),
        rejected_count=int(agent_task_runtime_source_fallback_diagnostics.get("rejected_count", 0) or 0),
        affected_agent_count=int(agent_task_runtime_source_fallback_diagnostics.get("affected_agent_count", 0) or 0),
    )

    main_tsp_executed_region_ids = {
        region_id
        for record in tsp_records.values()
        for region_id in record.get("final_order", [])
    }
    residual_backfill_count = 0
    main_integrated_residual_count = 0
    post_backfill_residual_count = 0
    cover_only_residual_backfill_count = 0
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
        agent_feasible_patterns=agent_feasible_patterns if heterogeneous_mode else None,
        agent_obstacle_fields=agent_obstacle_fields if heterogeneous_mode else None,
    )
    finish_stage(
        "skipped_region_recovery",
        stage_started,
        recovered_count=int(skipped_region_recovery.get("recovered_count", 0) or 0),
        failed_count=int(skipped_region_recovery.get("failed_count", 0) or 0),
        connector_cache_size=int(skipped_region_recovery.get("connector_cache_size", 0) or 0),
        connector_attempt_count=int(skipped_region_recovery.get("recovery_connector_attempt_count", 0) or 0),
        prefiltered_count=int(skipped_region_recovery.get("recovery_prefiltered_count", 0) or 0),
        agent_pruned_count=int(skipped_region_recovery.get("recovery_agent_pruned_count", 0) or 0),
        budget_exhausted=bool(skipped_region_recovery.get("budget_exhausted", False)),
        budget_reason=str(skipped_region_recovery.get("budget_reason", "")),
        elapsed_sec=round(float(skipped_region_recovery.get("elapsed_sec", 0.0) or 0.0), 3),
    )
    if skipped_region_recovery["recovered_count"]:
        for agent_id, agent in agents.items():
            agent_config = config.for_agent(agent_id) if heterogeneous_mode else config
            agent.metrics = _agent_metrics(
                agent.segments,
                agent_config,
                agent_obstacle_fields.get(agent_id, obstacle_field),
            )
    coverage_state = evaluate_tour_coverage_state(
        config,
        list(tours.values()),
        resolution=path_config.residual_resolution,
        obstacle_field=obstacle_field,
        include_non_cover_segments=path_config.count_transit_coverage,
    )
    integrated_cover_only_state = evaluate_tour_coverage_state(
        config,
        list(tours.values()),
        resolution=path_config.residual_resolution,
        obstacle_field=obstacle_field,
        include_non_cover_segments=False,
    )
    integrated_cover_only_target = (
        path_config.target_coverage_fraction
        if path_config.cover_only_target_fraction is None
        else max(0.0, min(1.0, float(path_config.cover_only_target_fraction)))
    )
    if (
        path_config.enable_integrated_residual_candidates
        and (
            coverage_state.coverage_fraction + 1e-9 < path_config.target_coverage_fraction
            or integrated_cover_only_state.coverage_fraction + 1e-9 < integrated_cover_only_target
        )
    ):
        stage_started = time.perf_counter()
        integrated_state = integrated_cover_only_state if integrated_cover_only_state.coverage_fraction + 1e-9 < integrated_cover_only_target else coverage_state
        residual_result = append_residual_local_tsp(
            config=config,
            path_config=path_config,
            obstacle_field=obstacle_field,
            tours=tours,
            coverage_state=integrated_state,
            agents=agents,
            ownership_map=ownership_map,
        )
        residual_local_tsp_status = residual_result.diagnostics.get("status", "unknown")
        diagnostics = dict(residual_result.diagnostics)
        diagnostics["coverage_mode"] = "main_integrated"
        residual_backfill_diagnostics.append(diagnostics)
        finish_stage(
            "main_integrated_residual_candidates",
            stage_started,
            appended_count=residual_result.appended_count,
            status=residual_local_tsp_status,
            residual_feasible_count=residual_result.diagnostics.get("residual_feasible_count", "0"),
            residual_infeasible_count=residual_result.diagnostics.get("residual_infeasible_count", "0"),
        )
        if residual_result.appended_count:
            residual_backfill_count += residual_result.appended_count
            main_integrated_residual_count += residual_result.appended_count
            repeat_path_penalty_total += residual_result.repeat_path_penalty_total
            for agent_id, agent in agents.items():
                agent_config = config.for_agent(agent_id) if heterogeneous_mode else config
                agent.metrics = _agent_metrics(
                    agent.segments,
                    agent_config,
                    agent_obstacle_fields.get(agent_id, obstacle_field),
                )
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
        post_backfill_residual_count += residual_result.appended_count
        repeat_path_penalty_total += residual_result.repeat_path_penalty_total
        for agent_id, agent in agents.items():
            agent_config = config.for_agent(agent_id) if heterogeneous_mode else config
            agent.metrics = _agent_metrics(
                agent.segments,
                agent_config,
                agent_obstacle_fields.get(agent_id, obstacle_field),
            )
        coverage_state = evaluate_tour_coverage_state(
            config,
            list(tours.values()),
            resolution=path_config.residual_resolution,
            obstacle_field=obstacle_field,
            include_non_cover_segments=path_config.count_transit_coverage,
        )
        if coverage_state.coverage_fraction + 1e-9 >= path_config.target_coverage_fraction:
            break
    cover_only_target_fraction_for_backfill = (
        path_config.target_coverage_fraction
        if path_config.cover_only_target_fraction is None
        else max(0.0, min(1.0, float(path_config.cover_only_target_fraction)))
    )
    cover_only_backfill_state = evaluate_tour_coverage_state(
        config,
        list(tours.values()),
        resolution=path_config.residual_resolution,
        obstacle_field=obstacle_field,
        include_non_cover_segments=False,
    )
    cover_only_path_config = path_config
    cover_only_cycle_limit = max(path_config.residual_backfill_cycles, 0)
    if _large_map_mode_enabled(config, path_config):
        cover_only_cycle_limit = min(cover_only_cycle_limit, 1)
        cover_only_path_config = replace(
            path_config,
            max_residual_backfill_regions=min(max(path_config.max_residual_backfill_regions, 0), 4),
        )
    for cover_only_cycle_idx in range(cover_only_cycle_limit):
        if cover_only_backfill_state.coverage_fraction + 1e-9 >= cover_only_target_fraction_for_backfill:
            break
        stage_started = time.perf_counter()
        residual_result = append_residual_local_tsp(
            config=config,
            path_config=cover_only_path_config,
            obstacle_field=obstacle_field,
            tours=tours,
            coverage_state=cover_only_backfill_state,
            agents=agents,
            ownership_map=ownership_map,
        )
        residual_local_tsp_status = residual_result.diagnostics.get("status", "unknown")
        diagnostics = dict(residual_result.diagnostics)
        diagnostics["coverage_mode"] = "cover_only"
        residual_backfill_diagnostics.append(diagnostics)
        finish_stage(
            f"cover_only_residual_backfill_cycle_{cover_only_cycle_idx + 1}",
            stage_started,
            appended_count=residual_result.appended_count,
            status=residual_local_tsp_status,
            residual_feasible_count=residual_result.diagnostics.get("residual_feasible_count", "0"),
            residual_infeasible_count=residual_result.diagnostics.get("residual_infeasible_count", "0"),
        )
        if residual_result.appended_count == 0:
            break
        residual_backfill_count += residual_result.appended_count
        post_backfill_residual_count += residual_result.appended_count
        cover_only_residual_backfill_count += residual_result.appended_count
        repeat_path_penalty_total += residual_result.repeat_path_penalty_total
        for agent_id, agent in agents.items():
            agent_config = config.for_agent(agent_id) if heterogeneous_mode else config
            agent.metrics = _agent_metrics(
                agent.segments,
                agent_config,
                agent_obstacle_fields.get(agent_id, obstacle_field),
            )
        coverage_state = evaluate_tour_coverage_state(
            config,
            list(tours.values()),
            resolution=path_config.residual_resolution,
            obstacle_field=obstacle_field,
            include_non_cover_segments=path_config.count_transit_coverage,
        )
        cover_only_backfill_state = evaluate_tour_coverage_state(
            config,
            list(tours.values()),
            resolution=path_config.residual_resolution,
            obstacle_field=obstacle_field,
            include_non_cover_segments=False,
        )
    route_refinement_diagnostics: Dict[str, object] = {
        "refined_connector_count": 0,
        "turn_angle_reduction": 0.0,
        "length_reduction": 0.0,
        "refinement_rejected_reasons": {},
        "route_refinement_status": "disabled",
    }
    if path_config.enable_global_route_refinement:
        stage_started = time.perf_counter()
        if (
            _large_map_mode_enabled(config, path_config)
            and coverage_state.coverage_fraction + 1e-9 < path_config.target_coverage_fraction
        ):
            route_refinement_diagnostics["route_refinement_status"] = "skipped_target_coverage_incomplete"
        else:
            route_refinement_diagnostics = _refine_global_routes(
                agents=agents,
                tours=tours,
                config=config,
                path_config=path_config,
                obstacle_field=obstacle_field,
                baseline_coverage=coverage_state.coverage_fraction,
                agent_obstacle_fields=agent_obstacle_fields if heterogeneous_mode else None,
            )
        finish_stage(
            "route_refinement",
            stage_started,
            refined_connector_count=int(route_refinement_diagnostics.get("refined_connector_count", 0) or 0),
            merged_noncover_window_count=int(route_refinement_diagnostics.get("merged_noncover_window_count", 0) or 0),
            turn_angle_reduction=round(float(route_refinement_diagnostics.get("turn_angle_reduction", 0.0) or 0.0), 6),
            length_reduction=round(float(route_refinement_diagnostics.get("length_reduction", 0.0) or 0.0), 6),
            status=route_refinement_diagnostics.get("route_refinement_status", "unknown"),
        )
        if int(route_refinement_diagnostics.get("refined_connector_count", 0) or 0) > 0:
            coverage_state = evaluate_tour_coverage_state(
                config,
                list(tours.values()),
                resolution=path_config.residual_resolution,
                obstacle_field=obstacle_field,
                include_non_cover_segments=path_config.count_transit_coverage,
            )
    assign_stable_resource_ids(agents, path_config)
    shared_before_schedule = shared_resource_metrics(agents, path_config.resource_separation_time)
    mapf_conflicts_resolved_after_residual = apply_resource_window_schedule(
        agents,
        separation_time=path_config.resource_separation_time,
    )
    shared_after_schedule = shared_resource_metrics(agents, path_config.resource_separation_time)
    for agent_id, agent in agents.items():
        agent_config = config.for_agent(agent_id) if heterogeneous_mode else config
        agent.metrics = _agent_metrics(
            agent.segments,
            agent_config,
            agent_obstacle_fields.get(agent_id, obstacle_field),
        )
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
    residual_low_efficiency_soft_count = sum(
        int(item.get("residual_low_efficiency_soft_count", "0") or 0)
        for item in residual_backfill_diagnostics
    )
    residual_best_gain_per_path_meter = max(
        (float(item.get("residual_best_gain_per_path_meter", "0") or 0.0) for item in residual_backfill_diagnostics),
        default=0.0,
    )
    residual_min_positive_gain_per_path_meter = min(
        (
            float(item.get("residual_min_positive_gain_per_path_meter", "0") or 0.0)
            for item in residual_backfill_diagnostics
            if float(item.get("residual_min_positive_gain_per_path_meter", "0") or 0.0) > 0.0
        ),
        default=0.0,
    )
    residual_mean_positive_gain_values = [
        float(item.get("residual_mean_positive_gain_per_path_meter", "0") or 0.0)
        for item in residual_backfill_diagnostics
        if float(item.get("residual_mean_positive_gain_per_path_meter", "0") or 0.0) > 0.0
    ]
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
    oriented_sweep_skip_reasons: Dict[str, int] = {}
    convex_oriented_candidate_region_count = 0
    for region in regions:
        metadata = getattr(region, "metadata", {}) or {}
        reason = str(metadata.get("oriented_sweep_skip_reason", ""))
        if reason:
            oriented_sweep_skip_reasons[reason] = oriented_sweep_skip_reasons.get(reason, 0) + 1
        try:
            selected_angle_count = int(metadata.get("selected_oriented_angle_count", "0") or 0)
        except (TypeError, ValueError):
            selected_angle_count = 0
        if str(metadata.get("convexity_status", "")).lower() == "convex" and selected_angle_count > 0:
            convex_oriented_candidate_region_count += 1
    recovered_region_count = int(skipped_region_recovery.get("recovered_count", 0) or 0)
    target_coverage_met = coverage_state.coverage_fraction + 1e-9 >= path_config.target_coverage_fraction
    cover_only_target_fraction = (
        path_config.target_coverage_fraction
        if path_config.cover_only_target_fraction is None
        else max(0.0, min(1.0, float(path_config.cover_only_target_fraction)))
    )
    cover_only_coverage_gap = max(0.0, cover_only_target_fraction - cover_only_coverage_state.coverage_fraction)
    cover_only_target_met = cover_only_coverage_gap <= 1e-9
    region_execution_complete = not skipped_region_ids
    constraint_violation_count = int(
        totals.get("out_of_bounds_segment_count", 0.0)
        + totals.get("obstacle_collision_segment_count", 0.0)
        + totals.get("kinematic_infeasible_segment_count", 0.0)
    )
    constraints_legal = constraint_violation_count == 0
    noncover_length_ratio = totals["transition_length"] / max(totals["total_length"], 1e-9)
    turn_angle_per_coverage_meter = totals["total_turn_angle"] / max(totals["coverage_length"], 1e-9)
    connector_noncover_repeat_length = sum(
        float(record.get("connector_noncover_repeat_length", 0.0) or 0.0)
        for record in tsp_records.values()
    )
    connector_noncover_repeat_penalty = sum(
        float(record.get("connector_noncover_repeat_penalty", 0.0) or 0.0)
        for record in tsp_records.values()
    )
    connector_length = sum(float(record.get("connector_length", 0.0) or 0.0) for record in tsp_records.values())
    connector_turn_angle = sum(float(record.get("connector_turn_angle", 0.0) or 0.0) for record in tsp_records.values())
    residual_gain_per_meter_summary = {
        "cycle_count": len(residual_backfill_diagnostics),
        "best": residual_best_gain_per_path_meter,
        "min_positive": residual_min_positive_gain_per_path_meter,
        "mean_positive": sum(residual_mean_positive_gain_values) / max(len(residual_mean_positive_gain_values), 1),
        "threshold": max(path_config.residual_min_gain_per_path_meter, 0.0),
        "filtered_count": residual_low_efficiency_filtered_count,
        "soft_count": residual_low_efficiency_soft_count,
    }
    coverage_merge_rejected_by_reason = dict(
        coverage_merge_diagnostics.get("coverage_merge_rejected_by_reason", {}) or {}
    )
    coverage_merge_objective_delta = float(
        coverage_merge_diagnostics.get("coverage_merge_objective_delta", 0.0) or 0.0
    )
    assignment_assignability = _parse_json_diagnostic(
        assignment_strategy_diagnostics.get("region_assignability_matrix", "{}"),
        {},
    )
    assignment_workload = _parse_json_diagnostic(
        assignment_strategy_diagnostics.get("region_workload_matrix", "{}"),
        {},
    )
    assignment_best_patterns = _parse_json_diagnostic(
        assignment_strategy_diagnostics.get("region_best_pattern_matrix", "{}"),
        {},
    )
    assignment_migrations = _parse_json_diagnostic(
        assignment_strategy_diagnostics.get("boundary_migration_records", "[]"),
        [],
    )
    assignment_exchanges = _parse_json_diagnostic(
        assignment_strategy_diagnostics.get("boundary_exchange_records", "[]"),
        [],
    )
    report: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "map_id": map_id,
        "algorithm": "paper_style_region_tsp",
        "heterogeneous_connected_assignment_enabled": heterogeneous_mode,
        "fleet_profile_id": config.fleet_profile_id,
        "agent_planning_profiles": _agent_planning_profile_report(config),
        "agent_region_assignability_matrix": assignment_assignability,
        "agent_region_estimated_workload_matrix": assignment_workload,
        "agent_region_best_pattern_matrix": assignment_best_patterns,
        "agent_region_assignment": {
            str(agent_id): list(region_ids) for agent_id, region_ids in assignment.agent_regions.items()
        },
        "agent_estimated_loads": {str(agent_id): load for agent_id, load in assignment_strategy_loads.items()},
        "agent_actual_completion_times": {
            str(agent_id): float(agent.metrics.get("estimated_time", 0.0)) for agent_id, agent in agents.items()
        },
        "agent_assignment_connected": {
            str(agent_id): bool(connected) for agent_id, connected in assignment.connected.items()
        },
        "cross_agent_migration_records": assignment_migrations,
        "cross_agent_exchange_records": assignment_exchanges,
        "assignment_unassigned_region_reasons": _parse_json_diagnostic(
            assignment_strategy_diagnostics.get("unassigned_region_reasons", "{}"),
            {},
        ),
        "assignment_reject_reasons": _parse_json_diagnostic(
            assignment_strategy_diagnostics.get("assignment_reject_reasons", "{}"),
            {},
        ),
        "heterogeneous_candidate_diagnostics": heterogeneous_candidate_diagnostics,
        "controlled_region_split_records": controlled_split_records,
        "controlled_region_split_count": len(controlled_split_records),
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
        "open_chain_flexible_exit_variant_attempt_count": int(
            sweep_build_stats.get("open_chain_flexible_exit_variant_attempt_count", 0) or 0
        ),
        "open_chain_flexible_exit_variant_success_count": int(
            sweep_build_stats.get("open_chain_flexible_exit_variant_success_count", 0) or 0
        ),
        "open_chain_flexible_exit_variant_failure_count": int(
            sweep_build_stats.get("open_chain_flexible_exit_variant_failure_count", 0) or 0
        ),
        "open_chain_flexible_exit_variant_failure_reasons": dict(
            sweep_build_stats.get("open_chain_flexible_exit_variant_failure_reasons", {}) or {}
        ),
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
        "oriented_sweep_skip_reasons": oriented_sweep_skip_reasons,
        "convex_oriented_candidate_region_count": convex_oriented_candidate_region_count,
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
        "coverage_aware_merge_enabled": bool(path_config.enable_coverage_aware_merge),
        "coverage_merge_status": str(coverage_merge_diagnostics.get("coverage_merge_status", "disabled")),
        "coverage_merge_budget_exhausted": bool(
            coverage_merge_diagnostics.get("coverage_merge_budget_exhausted", False)
        ),
        "coverage_merge_budget_reason": str(coverage_merge_diagnostics.get("coverage_merge_budget_reason", "")),
        "coverage_merge_elapsed_sec": float(coverage_merge_diagnostics.get("coverage_merge_elapsed_sec", 0.0) or 0.0),
        "coverage_merge_region_count_before": int(
            coverage_merge_diagnostics.get("coverage_merge_region_count_before", coverage_merge_input_count) or coverage_merge_input_count
        ),
        "coverage_merge_region_count_after": int(
            coverage_merge_diagnostics.get("coverage_merge_region_count_after", len(regions)) or len(regions)
        ),
        "coverage_merge_candidate_count": int(
            coverage_merge_diagnostics.get("coverage_merge_candidate_count", 0) or 0
        ),
        "coverage_merge_validation_count": int(
            coverage_merge_diagnostics.get("coverage_merge_validation_count", 0) or 0
        ),
        "coverage_merge_iteration_count": int(
            coverage_merge_diagnostics.get("coverage_merge_iteration_count", 0) or 0
        ),
        "coverage_merge_no_improvement_round_count": int(
            coverage_merge_diagnostics.get("coverage_merge_no_improvement_round_count", 0) or 0
        ),
        "coverage_merge_accepted_count": int(
            coverage_merge_diagnostics.get("coverage_merge_accepted_count", 0) or 0
        ),
        "coverage_merge_rejected_by_reason": coverage_merge_rejected_by_reason,
        "coverage_merge_rejected_candidates": list(
            coverage_merge_diagnostics.get("coverage_merge_rejected_candidates", []) or []
        ),
        "coverage_merge_objective_delta": coverage_merge_objective_delta,
        "coverage_merge_regions": list(coverage_merge_diagnostics.get("coverage_merge_regions", []) or []),
        "agent_task_merge_enabled": bool(path_config.enable_agent_task_region_merge),
        "agent_task_merge_status": str(agent_task_merge_diagnostics.get("agent_task_merge_status", "disabled")),
        "agent_task_merge_region_count_before": int(
            agent_task_merge_diagnostics.get("agent_task_merge_region_count_before", len(feasible_regions)) or 0
        ),
        "agent_task_merge_region_count_after": int(
            agent_task_merge_diagnostics.get("agent_task_merge_region_count_after", len(feasible_regions)) or 0
        ),
        "agent_task_merge_candidate_count": int(
            agent_task_merge_diagnostics.get("agent_task_merge_candidate_count", 0) or 0
        ),
        "agent_task_merge_accepted_count": int(
            agent_task_merge_diagnostics.get("agent_task_merge_accepted_count", 0) or 0
        ),
        "agent_task_strip_merge_enabled": bool(
            agent_task_merge_diagnostics.get("agent_task_strip_merge_enabled", False)
        ),
        "agent_task_strip_candidate_count": int(
            agent_task_merge_diagnostics.get("agent_task_strip_candidate_count", 0) or 0
        ),
        "agent_task_strip_accepted_count": int(
            agent_task_merge_diagnostics.get("agent_task_strip_accepted_count", 0) or 0
        ),
        "agent_task_strip_budget_exhausted": bool(
            agent_task_merge_diagnostics.get("agent_task_strip_budget_exhausted", False)
        ),
        "agent_task_strip_budget_reason": str(
            agent_task_merge_diagnostics.get("agent_task_strip_budget_reason", "")
        ),
        "agent_task_strip_elapsed_sec": float(
            agent_task_merge_diagnostics.get("agent_task_strip_elapsed_sec", 0.0) or 0.0
        ),
        "agent_task_strip_rejected_by_reason": dict(
            agent_task_merge_diagnostics.get("agent_task_strip_rejected_by_reason", {}) or {}
        ),
        "agent_task_strip_regions": list(agent_task_merge_diagnostics.get("agent_task_strip_regions", []) or []),
        "agent_task_unified_merge_enabled": bool(
            agent_task_merge_diagnostics.get("agent_task_unified_merge_enabled", False)
        ),
        "agent_task_merge_pairwise_fallback_enabled": bool(
            agent_task_merge_diagnostics.get("agent_task_merge_pairwise_fallback_enabled", False)
        ),
        "agent_task_merge_min_unified_rectangularity": float(
            agent_task_merge_diagnostics.get("agent_task_merge_min_unified_rectangularity", 0.0) or 0.0
        ),
        "agent_task_merge_max_unified_group_size": int(
            agent_task_merge_diagnostics.get("agent_task_merge_max_unified_group_size", 0) or 0
        ),
        "agent_task_merge_prefer_full_components": bool(
            agent_task_merge_diagnostics.get("agent_task_merge_prefer_full_components", False)
        ),
        "agent_task_merge_full_component_max_regions": int(
            agent_task_merge_diagnostics.get("agent_task_merge_full_component_max_regions", 0) or 0
        ),
        "agent_task_merge_full_component_min_rectangularity": float(
            agent_task_merge_diagnostics.get("agent_task_merge_full_component_min_rectangularity", 0.0) or 0.0
        ),
        "agent_task_full_component_accepted_count": int(
            agent_task_merge_diagnostics.get("agent_task_full_component_accepted_count", 0) or 0
        ),
        "agent_task_unified_candidate_count": int(
            agent_task_merge_diagnostics.get("agent_task_unified_candidate_count", 0) or 0
        ),
        "agent_task_unified_accepted_count": int(
            agent_task_merge_diagnostics.get("agent_task_unified_accepted_count", 0) or 0
        ),
        "agent_task_unified_validation_count": int(
            agent_task_merge_diagnostics.get("agent_task_unified_validation_count", 0) or 0
        ),
        "agent_task_merge_rejected_by_reason": dict(
            agent_task_merge_diagnostics.get("agent_task_merge_rejected_by_reason", {}) or {}
        ),
        "agent_task_merge_rejected_candidates": list(
            agent_task_merge_diagnostics.get("agent_task_merge_rejected_candidates", []) or []
        ),
        "agent_task_merge_regions": list(agent_task_merge_diagnostics.get("agent_task_merge_regions", []) or []),
        "agent_task_merge_split_count": int(
            agent_task_merge_diagnostics.get("agent_task_merge_split_count", 0) or 0
        ),
        "agent_task_merge_unstable_region_count": int(
            agent_task_merge_diagnostics.get("agent_task_merge_unstable_region_count", 0) or 0
        ),
        "agent_task_merge_unstable_regions": list(
            agent_task_merge_diagnostics.get("agent_task_merge_unstable_regions", []) or []
        ),
        "agent_task_runtime_source_fallback": agent_task_runtime_source_fallback_diagnostics,
        "agent_task_runtime_source_fallback_candidate_count": int(
            agent_task_runtime_source_fallback_diagnostics.get("candidate_count", 0) or 0
        ),
        "agent_task_runtime_source_fallback_accepted_count": int(
            agent_task_runtime_source_fallback_diagnostics.get("accepted_count", 0) or 0
        ),
        "agent_task_runtime_source_fallback_rejected_count": int(
            agent_task_runtime_source_fallback_diagnostics.get("rejected_count", 0) or 0
        ),
        "agent_task_runtime_source_fallback_affected_agent_count": int(
            agent_task_runtime_source_fallback_diagnostics.get("affected_agent_count", 0) or 0
        ),
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
        "constraint_violation_count": constraint_violation_count,
        "constraints_legal": constraints_legal,
        "transit_assisted_coverage_fraction": coverage_state.coverage_fraction,
        "free_space_target_coverage": path_config.target_coverage_fraction,
        "residual_count": len(coverage_state.residual_components),
        "residual_backfill_count": residual_backfill_count,
        "main_integrated_residual_count": main_integrated_residual_count,
        "post_backfill_residual_count": post_backfill_residual_count,
        "cover_only_residual_backfill_count": cover_only_residual_backfill_count,
        "residual_feasible_count": residual_feasible_count,
        "residual_infeasible_count": residual_infeasible_count,
        "residual_low_efficiency_filtered_count": residual_low_efficiency_filtered_count,
        "residual_low_efficiency_soft_count": residual_low_efficiency_soft_count,
        "residual_best_gain_per_path_meter": residual_best_gain_per_path_meter,
        "residual_min_positive_gain_per_path_meter": residual_min_positive_gain_per_path_meter,
        "residual_gain_per_meter_summary": residual_gain_per_meter_summary,
        "residual_min_gain_per_path_meter": max(path_config.residual_min_gain_per_path_meter, 0.0),
        "residual_filter_after_target_only": path_config.residual_filter_after_target_only,
        "residual_local_tsp_enabled": path_config.enable_residual_local_tsp,
        "residual_local_tsp_status": residual_local_tsp_status,
        "residual_backfill_diagnostics": residual_backfill_diagnostics,
        "skipped_region_recovery": skipped_region_recovery,
        "short_region_recovery_attempt_count": int(
            skipped_region_recovery.get("short_region_recovery_attempt_count", 0) or 0
        ),
        "short_region_recovery_success_count": int(
            skipped_region_recovery.get("short_region_recovery_success_count", 0) or 0
        ),
        "skipped_recovery_connector_attempt_count": int(
            skipped_region_recovery.get("recovery_connector_attempt_count", 0) or 0
        ),
        "skipped_recovery_prefiltered_count": int(
            skipped_region_recovery.get("recovery_prefiltered_count", 0) or 0
        ),
        "skipped_recovery_agent_pruned_count": int(
            skipped_region_recovery.get("recovery_agent_pruned_count", 0) or 0
        ),
        "skipped_region_diagnostics": skipped_region_diagnostics,
        "reachable_region_count": len(visit_nodes) - len(skipped_region_ids),
        "region_connection_graph_components": large_map_metadata.get("region_connection_graph_components", {}),
        "skipped_by_no_incoming_edge": len(skipped_region_ids),
        "skipped_by_no_outgoing_edge": len(skipped_region_ids),
        "large_map_connector_cache_size": large_map_metadata.get("large_map_connector_cache_size", 0),
        "large_map_reachability_probe_count": large_map_metadata.get("large_map_reachability_probe_count", 0),
        "large_map_reachability_probe_success_count": large_map_metadata.get("large_map_reachability_probe_success_count", 0),
        "large_map_dead_end_avoidance_count": large_map_metadata.get("large_map_dead_end_avoidance_count", 0),
        "large_map_greedy_cheap_probe_collision_only": large_map_metadata.get(
            "large_map_greedy_cheap_probe_collision_only",
            False,
        ),
        "large_map_greedy_obstacle_aware_attempt_count": large_map_metadata.get(
            "large_map_greedy_obstacle_aware_attempt_count", 0
        ),
        "large_map_greedy_obstacle_aware_filtered_count": large_map_metadata.get(
            "large_map_greedy_obstacle_aware_filtered_count", 0
        ),
        "repeat_path_penalty_total": repeat_path_penalty_total,
        "load_swap_count": load_swap_count,
        "load_swap_candidate_count": load_swap_candidate_count,
        "load_swap_reject_reasons": load_swap_reject_reasons,
        "load_swap_before_imbalance": float(load_swap_before),
        "load_swap_after_imbalance": float(load_swap_after),
        "joint_optimizer_status": str(joint_optimizer_diagnostics.get("joint_optimizer_status", "disabled")),
        "global_joint_objective": float(joint_optimizer_diagnostics.get("global_joint_objective", 0.0) or 0.0),
        "cross_agent_connector_overlap_length": float(
            joint_optimizer_diagnostics.get("cross_agent_connector_overlap_length", 0.0) or 0.0
        ),
        "cross_agent_crossing_count": int(joint_optimizer_diagnostics.get("cross_agent_crossing_count", 0) or 0),
        "joint_mission_makespan": float(joint_optimizer_diagnostics.get("joint_mission_makespan", 0.0) or 0.0),
        "joint_total_agent_work_time": float(
            joint_optimizer_diagnostics.get("joint_total_agent_work_time", 0.0) or 0.0
        ),
        "joint_agent_time_imbalance": float(
            joint_optimizer_diagnostics.get("joint_agent_time_imbalance", 0.0) or 0.0
        ),
        "joint_improvement_iteration_count": int(
            joint_optimizer_diagnostics.get("joint_improvement_iteration_count", 0) or 0
        ),
        "joint_candidate_attempt_count": int(joint_optimizer_diagnostics.get("joint_candidate_attempt_count", 0) or 0),
        "joint_candidate_accept_count": int(joint_optimizer_diagnostics.get("joint_candidate_accept_count", 0) or 0),
        "joint_region_count": int(joint_optimizer_diagnostics.get("joint_region_count", 0) or 0),
        "joint_large_map_region_limit": int(joint_optimizer_diagnostics.get("joint_large_map_region_limit", 0) or 0),
        "joint_reject_reasons": dict(joint_optimizer_diagnostics.get("joint_reject_reasons", {}) or {}),
        "refined_connector_count": int(route_refinement_diagnostics.get("refined_connector_count", 0) or 0),
        "merged_noncover_window_count": int(route_refinement_diagnostics.get("merged_noncover_window_count", 0) or 0),
        "turn_angle_reduction": float(route_refinement_diagnostics.get("turn_angle_reduction", 0.0) or 0.0),
        "length_reduction": float(route_refinement_diagnostics.get("length_reduction", 0.0) or 0.0),
        "refinement_rejected_reasons": dict(route_refinement_diagnostics.get("refinement_rejected_reasons", {}) or {}),
        "route_refinement_status": str(route_refinement_diagnostics.get("route_refinement_status", "disabled")),
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
        "noncover_length_ratio": noncover_length_ratio,
        "noncover_repeat_overlap_length": sum(item["overlap_length"] for item in agent_repeat_metrics.values()),
        "connector_noncover_repeat_length": connector_noncover_repeat_length,
        "connector_noncover_repeat_penalty": connector_noncover_repeat_penalty,
        "connector_length": connector_length,
        "connector_turn_angle": connector_turn_angle,
        "turn_angle_per_coverage_meter": turn_angle_per_coverage_meter,
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
    report["coverage_quality_status"] = (
        "complete"
        if target_coverage_met and cover_only_target_met and region_execution_complete and constraints_legal
        else "incomplete"
    )
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
            "coverage_aware_merge_enabled": str(path_config.enable_coverage_aware_merge).lower(),
            "coverage_merge_status": str(report["coverage_merge_status"]),
            "coverage_merge_budget_exhausted": str(report["coverage_merge_budget_exhausted"]).lower(),
            "coverage_merge_budget_reason": str(report["coverage_merge_budget_reason"]),
            "coverage_merge_elapsed_sec": f"{report['coverage_merge_elapsed_sec']:.6f}",
            "coverage_merge_region_count_before": str(report["coverage_merge_region_count_before"]),
            "coverage_merge_region_count_after": str(report["coverage_merge_region_count_after"]),
            "coverage_merge_candidate_count": str(report["coverage_merge_candidate_count"]),
            "coverage_merge_validation_count": str(report["coverage_merge_validation_count"]),
            "coverage_merge_iteration_count": str(report["coverage_merge_iteration_count"]),
            "coverage_merge_accepted_count": str(report["coverage_merge_accepted_count"]),
            "coverage_merge_objective_delta": f"{report['coverage_merge_objective_delta']:.6f}",
            "oriented_pattern_count": str(report["oriented_pattern_count"]),
            "axis_aligned_pattern_count": str(report["axis_aligned_pattern_count"]),
            "selected_oriented_pattern_count": str(report["selected_oriented_pattern_count"]),
            "convex_oriented_candidate_region_count": str(report["convex_oriented_candidate_region_count"]),
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
            "main_integrated_residual_count": str(main_integrated_residual_count),
            "post_backfill_residual_count": str(post_backfill_residual_count),
            "cover_only_residual_backfill_count": str(cover_only_residual_backfill_count),
            "residual_low_efficiency_filtered_count": str(residual_low_efficiency_filtered_count),
            "residual_low_efficiency_soft_count": str(residual_low_efficiency_soft_count),
            "residual_best_gain_per_path_meter": f"{residual_best_gain_per_path_meter:.6f}",
            "residual_min_positive_gain_per_path_meter": f"{residual_min_positive_gain_per_path_meter:.6f}",
            "residual_filter_after_target_only": str(path_config.residual_filter_after_target_only).lower(),
            "residual_local_tsp_enabled": str(path_config.enable_residual_local_tsp).lower(),
            "short_region_recovery_attempt_count": str(report["short_region_recovery_attempt_count"]),
            "short_region_recovery_success_count": str(report["short_region_recovery_success_count"]),
            "repeat_path_penalty_total": f"{repeat_path_penalty_total:.6f}",
            "mapf_conflicts_resolved_after_residual": str(mapf_conflicts_resolved_after_residual),
            "shared_resource_count": str(int(shared_after_schedule["shared_resource_count"])),
            "shared_resource_conflict_count": str(int(shared_before_schedule["true_time_conflict_count"])),
            "spatial_overlap_reuse_count": str(int(shared_after_schedule["spatial_overlap_reuse_count"])),
            "true_time_conflict_count": str(int(shared_after_schedule["true_time_conflict_count"])),
            "main_repeat_path_penalty_enabled": str(path_config.enable_main_repeat_path_penalty).lower(),
            "main_repeat_overlap_length": f"{report['main_repeat_overlap_length']:.6f}",
            "main_repeat_penalty_total": f"{report['main_repeat_penalty_total']:.6f}",
            "noncover_length_ratio": f"{report['noncover_length_ratio']:.6f}",
            "noncover_repeat_overlap_length": f"{report['noncover_repeat_overlap_length']:.6f}",
            "connector_noncover_repeat_length": f"{report['connector_noncover_repeat_length']:.6f}",
            "connector_turn_angle": f"{report['connector_turn_angle']:.6f}",
            "turn_angle_per_coverage_meter": f"{report['turn_angle_per_coverage_meter']:.6f}",
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
            "agent_task_merge_status": str(report["agent_task_merge_status"]),
            "agent_task_merge_region_count_before": str(report["agent_task_merge_region_count_before"]),
            "agent_task_merge_region_count_after": str(report["agent_task_merge_region_count_after"]),
            "agent_task_merge_candidate_count": str(report["agent_task_merge_candidate_count"]),
            "agent_task_merge_accepted_count": str(report["agent_task_merge_accepted_count"]),
            "agent_task_strip_merge_enabled": str(report["agent_task_strip_merge_enabled"]).lower(),
            "agent_task_strip_candidate_count": str(report["agent_task_strip_candidate_count"]),
            "agent_task_strip_accepted_count": str(report["agent_task_strip_accepted_count"]),
            "agent_task_strip_budget_exhausted": str(report["agent_task_strip_budget_exhausted"]).lower(),
            "agent_task_strip_budget_reason": str(report["agent_task_strip_budget_reason"]),
            "agent_task_strip_elapsed_sec": f"{report['agent_task_strip_elapsed_sec']:.6f}",
            "agent_task_unified_merge_enabled": str(report["agent_task_unified_merge_enabled"]).lower(),
            "agent_task_merge_pairwise_fallback_enabled": str(report["agent_task_merge_pairwise_fallback_enabled"]).lower(),
            "agent_task_merge_min_unified_rectangularity": f"{report['agent_task_merge_min_unified_rectangularity']:.6f}",
            "agent_task_merge_max_unified_group_size": str(report["agent_task_merge_max_unified_group_size"]),
            "agent_task_merge_prefer_full_components": str(
                report["agent_task_merge_prefer_full_components"]
            ).lower(),
            "agent_task_merge_full_component_max_regions": str(
                report["agent_task_merge_full_component_max_regions"]
            ),
            "agent_task_merge_full_component_min_rectangularity": f"{report['agent_task_merge_full_component_min_rectangularity']:.6f}",
            "agent_task_full_component_accepted_count": str(report["agent_task_full_component_accepted_count"]),
            "agent_task_unified_candidate_count": str(report["agent_task_unified_candidate_count"]),
            "agent_task_unified_validation_count": str(report["agent_task_unified_validation_count"]),
            "agent_task_unified_accepted_count": str(report["agent_task_unified_accepted_count"]),
            "agent_task_runtime_source_fallback_candidate_count": str(
                report["agent_task_runtime_source_fallback_candidate_count"]
            ),
            "agent_task_runtime_source_fallback_accepted_count": str(
                report["agent_task_runtime_source_fallback_accepted_count"]
            ),
            "agent_task_runtime_source_fallback_rejected_count": str(
                report["agent_task_runtime_source_fallback_rejected_count"]
            ),
            "agent_task_runtime_source_fallback_affected_agent_count": str(
                report["agent_task_runtime_source_fallback_affected_agent_count"]
            ),
            "main_tsp_executed_region_count": str(report["main_tsp_executed_region_count"]),
            "recovered_region_count": str(report["recovered_region_count"]),
            "residual_only_region_count": str(report["residual_only_region_count"]),
            "load_swap_count": str(report["load_swap_count"]),
            "load_swap_candidate_count": str(report["load_swap_candidate_count"]),
            "load_swap_reject_reasons": str(report["load_swap_reject_reasons"]),
            "load_swap_before_imbalance": f"{report['load_swap_before_imbalance']:.6f}",
            "load_swap_after_imbalance": f"{report['load_swap_after_imbalance']:.6f}",
            "joint_optimizer_status": str(report["joint_optimizer_status"]),
            "global_joint_objective": f"{report['global_joint_objective']:.6f}",
            "joint_mission_makespan": f"{report['joint_mission_makespan']:.6f}",
            "joint_total_agent_work_time": f"{report['joint_total_agent_work_time']:.6f}",
            "joint_agent_time_imbalance": f"{report['joint_agent_time_imbalance']:.6f}",
            "cross_agent_connector_overlap_length": f"{report['cross_agent_connector_overlap_length']:.6f}",
            "cross_agent_crossing_count": str(report["cross_agent_crossing_count"]),
            "joint_improvement_iteration_count": str(report["joint_improvement_iteration_count"]),
            "joint_candidate_attempt_count": str(report["joint_candidate_attempt_count"]),
            "joint_candidate_accept_count": str(report["joint_candidate_accept_count"]),
            "joint_region_count": str(report["joint_region_count"]),
            "joint_large_map_region_limit": str(report["joint_large_map_region_limit"]),
            "route_refinement_status": str(report["route_refinement_status"]),
            "refined_connector_count": str(report["refined_connector_count"]),
            "turn_angle_reduction": f"{report['turn_angle_reduction']:.6f}",
            "length_reduction": f"{report['length_reduction']:.6f}",
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


def _merged_region_connector_variant_limit(path_config: PathPlanningConfig) -> int:
    base_limit = _connector_pattern_limit(path_config)
    entry_exit_limit = max(int(path_config.max_entry_exit_patterns_per_region), 1)
    return max(base_limit, min(entry_exit_limit, max(4, base_limit)))


def _pattern_needs_connector_variant_diversity(pattern: RegionCoveragePattern) -> bool:
    metadata = getattr(pattern, "metadata", {}) or {}
    if any(
        str(metadata.get(key, "")).lower() == "true"
        for key in (
            "coverage_aware_merged",
            "agent_task_strip_merge",
            "agent_task_unified_merge",
        )
    ):
        return True
    region_id = str(getattr(pattern, "region_id", ""))
    return (
        "_strip_task_region_" in region_id
        or "_unified_task_region_" in region_id
        or "_task_merge_region_" in region_id
        or region_id.startswith("coverage_merge_region_")
    )


def _connector_pattern_limit_for_region(
    region_id: str,
    candidates: Sequence[RegionCoveragePattern],
    path_config: PathPlanningConfig,
) -> int:
    base_limit = _connector_pattern_limit(path_config)
    if any(_pattern_needs_connector_variant_diversity(pattern) for pattern in candidates):
        return _merged_region_connector_variant_limit(path_config)
    return base_limit


def _tsp_record_from_result(
    agent_id: int,
    region_ids: Sequence[str],
    result: Dict[str, object],
    sweep_paths: Dict[str, RegionSweepPath],
) -> Dict[str, object]:
    final_order = list(result.get("final_order", []))
    final_set = set(final_order)
    initial_order = list(result.get("initial_order", region_ids))
    solver_metadata = dict(result.get("tsp_solver_metadata", {}))
    return {
        "assigned_regions": list(region_ids),
        "initial_order": initial_order,
        "final_order": final_order,
        "skipped_regions": [region_id for region_id in region_ids if region_id not in final_set],
        "tsp_node_count": len(region_ids),
        "tsp_solver_metadata": solver_metadata,
        "requested_tsp_solver": str(solver_metadata.get("requested_tsp_solver", "")),
        "effective_tsp_solver": str(solver_metadata.get("effective_tsp_solver", "deterministic")),
        "tsp_solver_status": str(solver_metadata.get("tsp_solver_status", "success")),
        "aco_best_objective": solver_metadata.get("aco_best_objective"),
        "aco_initial_objective": solver_metadata.get("aco_initial_objective"),
        "aco_iteration_count": int(solver_metadata.get("aco_iteration_count", 0) or 0),
        "aco_convergence_trace": list(solver_metadata.get("aco_convergence_trace", [])),
        "aco_accepted_3opt_count": int(solver_metadata.get("aco_accepted_3opt_count", 0) or 0),
        "candidate_pattern_counts": dict(result.get("candidate_pattern_counts", {})),
        "candidate_attempt_count": int(result.get("candidate_attempt_count", 0) or 0),
        "rejected_candidate_count": int(result.get("rejected_candidate_count", 0) or 0),
        "selected_pattern_ids": dict(result.get("selected_pattern_ids", {})),
        "skipped_region_reasons": dict(result.get("skipped_region_reasons", {})),
        "connector_failure_reasons": dict(result.get("connector_failure_reasons", {})),
        "all_connector_failure_reasons": dict(result.get("all_connector_failure_reasons", {})),
        "main_repeat_overlap_length": float(result.get("main_repeat_overlap_length", 0.0) or 0.0),
        "main_repeat_penalty_total": float(result.get("main_repeat_penalty_total", 0.0) or 0.0),
        "connector_noncover_repeat_length": float(result.get("connector_noncover_repeat_length", 0.0) or 0.0),
        "connector_noncover_repeat_penalty": float(result.get("connector_noncover_repeat_penalty", 0.0) or 0.0),
        "connector_length": float(result.get("connector_length", 0.0) or 0.0),
        "connector_turn_angle": float(result.get("connector_turn_angle", 0.0) or 0.0),
        "cross_agent_overlap_length": float(result.get("cross_agent_overlap_length", 0.0) or 0.0),
        "cross_agent_penalty_total": float(result.get("cross_agent_penalty_total", 0.0) or 0.0),
        "unavoidable_cross_agent_overlap_count": int(result.get("unavoidable_cross_agent_overlap_count", 0) or 0),
        "coverage_endpoint_count": sum(
            len(sweep_paths[region_id].endpoints)
            for region_id in region_ids
            if region_id in sweep_paths
        ),
        "infeasible_edges": list(result.get("infeasible_edges", [])),
    }


def _large_map_tsp_budget_fallback_result(
    agent_id: int,
    region_ids: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    path_config: PathPlanningConfig,
    reason: str,
    elapsed_sec: float,
    total_budget_sec: float,
) -> Dict[str, object]:
    skipped_reasons = {region_id: reason for region_id in region_ids}
    all_reasons = {region_id: [reason] for region_id in region_ids}
    infeasible_edges = [
        {
            "agent_id": agent_id,
            "region_id": region_id,
            "reason": reason,
            "elapsed_sec": round(float(elapsed_sec), 6),
            "total_budget_sec": round(float(total_budget_sec), 6),
        }
        for region_id in region_ids
    ]
    return {
        "initial_order": list(region_ids),
        "final_order": [],
        "segments": [],
        "infeasible_edges": infeasible_edges,
        "selected_patterns": {},
        "selected_pattern_ids": {},
        "candidate_pattern_counts": {region_id: len(patterns.get(region_id, [])) for region_id in region_ids},
        "candidate_attempt_count": 0,
        "rejected_candidate_count": 0,
        "main_repeat_overlap_length": 0.0,
        "main_repeat_penalty_total": 0.0,
        "connector_noncover_repeat_length": 0.0,
        "connector_noncover_repeat_penalty": 0.0,
        "connector_length": 0.0,
        "connector_turn_angle": 0.0,
        "cross_agent_overlap_length": 0.0,
        "cross_agent_penalty_total": 0.0,
        "unavoidable_cross_agent_overlap_count": 0,
        "tsp_solver_metadata": {
            "requested_tsp_solver": path_config.tsp_solver,
            "effective_tsp_solver": "large_map_budget_fallback",
            "tsp_solver_status": "failed",
            "failure_reason": reason,
            "elapsed_sec": float(elapsed_sec),
            "total_budget_sec": float(total_budget_sec),
            "aco_best_objective": None,
            "aco_initial_objective": None,
            "aco_iteration_count": 0,
            "aco_convergence_trace": [],
            "aco_accepted_3opt_count": 0,
        },
        "skipped_region_reasons": skipped_reasons,
        "connector_failure_reasons": dict(skipped_reasons),
        "all_connector_failure_reasons": all_reasons,
    }


def _apply_large_map_defaults(path_config: PathPlanningConfig, config: PlannerConfig) -> PathPlanningConfig:
    if not _large_map_mode_enabled(config, path_config):
        return path_config
    map_span = max(config.mission.area_length_x, config.mission.area_length_y)
    if map_span >= 150.0:
        obstacle_aware_transition_limit = min(max(120.0, 0.8 * map_span), 180.0)
    else:
        obstacle_aware_transition_limit = 80.0
    obstacle_aware_step_attempts = 2 if map_span <= 220.0 else 1
    obstacle_aware_agent_attempts = 20 if map_span <= 220.0 else 10
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
        obstacle_aware_astar_max_expansions=(
            min(max(int(path_config.obstacle_aware_astar_max_expansions), 160), 240)
            if int(path_config.obstacle_aware_astar_max_expansions) > 0
            else 160
        ),
        obstacle_aware_allow_motion_lattice=False,
        obstacle_aware_allow_corridor_conversion=True,
        large_map_tsp_obstacle_aware_retry_limit=1,
        large_map_tsp_max_candidate_attempts_per_step=min(
            max(int(path_config.large_map_tsp_max_candidate_attempts_per_step), 16),
            16,
        ),
        large_map_tsp_agent_time_budget_sec=min(
            max(float(path_config.large_map_tsp_agent_time_budget_sec), 180.0),
            300.0,
        ),
        large_map_tsp_total_time_budget_sec=(
            0.0
            if float(path_config.large_map_tsp_total_time_budget_sec) <= 0.0
            else min(max(float(path_config.large_map_tsp_total_time_budget_sec), 180.0), 1800.0)
        ),
        large_map_tsp_step_time_budget_sec=min(
            max(float(path_config.large_map_tsp_step_time_budget_sec), 8.0),
            12.0,
        ),
        large_map_tsp_enable_lookahead_probe=map_span <= 220.0 and bool(path_config.large_map_tsp_enable_lookahead_probe),
        large_map_tsp_require_cheap_connector_probe=False,
        large_map_tsp_cheap_probe_collision_only=True,
        large_map_tsp_max_obstacle_aware_attempts_per_step=min(
            max(int(path_config.large_map_tsp_max_obstacle_aware_attempts_per_step), obstacle_aware_step_attempts),
            obstacle_aware_step_attempts,
        ),
        large_map_tsp_max_obstacle_aware_attempts_per_agent=min(
            max(int(path_config.large_map_tsp_max_obstacle_aware_attempts_per_agent), obstacle_aware_agent_attempts),
            obstacle_aware_agent_attempts,
        ),
        large_map_tsp_obstacle_aware_max_transition_length=min(
            max(float(path_config.large_map_tsp_obstacle_aware_max_transition_length), obstacle_aware_transition_limit),
            obstacle_aware_transition_limit,
        ),
        max_residual_backfill_regions=min(max(int(path_config.max_residual_backfill_regions), 16), 64),
        residual_backfill_cycles=min(max(int(path_config.residual_backfill_cycles), 2), 6),
        residual_local_tsp_time_budget_sec=min(
            max(float(path_config.residual_local_tsp_time_budget_sec), 8.0),
            60.0,
        ),
        residual_local_tsp_max_candidate_attempts=min(
            max(int(path_config.residual_local_tsp_max_candidate_attempts), 480),
            800,
        ),
        skipped_region_recovery_time_budget_sec=min(
            max(float(path_config.skipped_region_recovery_time_budget_sec), 60.0),
            90.0,
        ),
        enable_open_sweep_chain_tsp=True,
        enable_agent_task_region_merge=bool(path_config.enable_agent_task_region_merge),
        enable_agent_task_lightweight_strip_merge=True,
        agent_task_strip_merge_max_candidate_evaluations=min(
            max(int(path_config.agent_task_strip_merge_max_candidate_evaluations), 1),
            20,
        ),
        agent_task_strip_merge_time_budget_sec=min(
            max(float(path_config.agent_task_strip_merge_time_budget_sec), 1.0),
            8.0,
        ),
        agent_task_strip_merge_use_geometric_preview=True,
        agent_task_strip_merge_max_groups_per_agent=min(
            max(int(path_config.agent_task_strip_merge_max_groups_per_agent), 1),
            6,
        ),
        agent_task_strip_merge_min_rectangularity=min(
            max(float(path_config.agent_task_strip_merge_min_rectangularity), 0.55),
            0.72,
        ),
        agent_task_merge_enable_pairwise_fallback=True,
        agent_task_merge_enable_unified_group_merge=True,
        agent_task_merge_prefer_full_components=True,
        agent_task_merge_full_component_max_regions=max(
            int(path_config.agent_task_merge_full_component_max_regions),
            12,
        ),
        agent_task_merge_full_component_min_rectangularity=min(
            max(float(path_config.agent_task_merge_full_component_min_rectangularity), 0.45),
            0.65,
        ),
        agent_task_merge_max_unified_candidates_per_agent=min(
            max(int(path_config.agent_task_merge_max_unified_candidates_per_agent), 2),
            6,
        ),
        agent_task_merge_min_improvement_ratio=min(
            max(float(path_config.agent_task_merge_min_improvement_ratio), 0.02),
            0.05,
        ),
        agent_task_merge_time_budget_sec=min(max(float(path_config.agent_task_merge_time_budget_sec), 0.0), 20.0),
        agent_task_merge_max_unified_group_size=(
            int(path_config.agent_task_merge_max_unified_group_size)
            if int(path_config.agent_task_merge_max_unified_group_size) > 0
            else 4
        ),
        joint_large_map_region_limit=min(max(int(path_config.joint_large_map_region_limit), 0), 30),
        joint_optimizer_time_budget_sec=min(
            max(float(path_config.joint_optimizer_time_budget_sec), 20.0),
            60.0,
        ),
        joint_eval_agent_time_budget_sec=min(
            max(float(path_config.joint_eval_agent_time_budget_sec), 8.0),
            20.0,
        ),
        joint_eval_step_time_budget_sec=min(
            max(float(path_config.joint_eval_step_time_budget_sec), 2.0),
            4.0,
        ),
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


def _split_infeasible_agent_task_merge_candidates(
    candidate_regions: Sequence,
    candidate_agent_regions: Dict[int, List[str]],
    base_regions: Sequence,
    missing_region_ids: Sequence[str],
) -> Tuple[List, Dict[int, List[str]], int]:
    missing = {str(region_id) for region_id in missing_region_ids}
    if not missing:
        return (
            list(candidate_regions),
            {agent_id: list(region_ids) for agent_id, region_ids in candidate_agent_regions.items()},
            0,
        )
    base_by_id = {region.region_id: region for region in base_regions}
    fallback_by_region: Dict[str, List[str]] = {}
    for region in candidate_regions:
        if region.region_id not in missing:
            continue
        raw_source_ids = str(
            region.metadata.get("merge_fallback_source_ids")
            or region.metadata.get("agent_task_strip_source_ids")
            or region.metadata.get("agent_task_unified_source_ids")
            or region.metadata.get("source_region_ids")
            or ""
        )
        source_ids = [item.strip() for item in raw_source_ids.split(",") if item.strip() in base_by_id]
        if source_ids:
            fallback_by_region[region.region_id] = source_ids
    if not fallback_by_region:
        return (
            list(candidate_regions),
            {agent_id: list(region_ids) for agent_id, region_ids in candidate_agent_regions.items()},
            0,
        )

    repaired_regions: List = []
    emitted: set[str] = set()
    for region in candidate_regions:
        source_ids = fallback_by_region.get(region.region_id)
        if not source_ids:
            if region.region_id not in emitted:
                repaired_regions.append(region)
                emitted.add(region.region_id)
            continue
        for source_id in source_ids:
            if source_id in emitted:
                continue
            repaired_regions.append(base_by_id[source_id])
            emitted.add(source_id)

    repaired_agent_regions: Dict[int, List[str]] = {}
    for agent_id, region_ids in candidate_agent_regions.items():
        repaired_ids: List[str] = []
        for region_id in region_ids:
            source_ids = fallback_by_region.get(region_id)
            if source_ids:
                for source_id in source_ids:
                    if source_id not in repaired_ids:
                        repaired_ids.append(source_id)
            elif region_id not in repaired_ids:
                repaired_ids.append(region_id)
        repaired_agent_regions[agent_id] = repaired_ids

    _populate_region_neighbors(repaired_regions)
    return repaired_regions, repaired_agent_regions, len(fallback_by_region)


def _agent_task_merge_source_region_ids(region) -> List[str]:
    metadata = getattr(region, "metadata", {}) or {}
    raw_source_ids = str(
        metadata.get("merge_fallback_source_ids")
        or metadata.get("agent_task_strip_source_ids")
        or metadata.get("agent_task_unified_source_ids")
        or metadata.get("source_region_ids")
        or ""
    )
    return [item.strip() for item in raw_source_ids.split(",") if item.strip()]


def _is_agent_task_merge_region(region) -> bool:
    metadata = getattr(region, "metadata", {}) or {}
    return (
        metadata.get("agent_task_strip_merge") == "true"
        or metadata.get("agent_task_unified_merge") == "true"
        or str(getattr(region, "region_id", "")).find("_strip_task_region_") >= 0
        or str(getattr(region, "region_id", "")).find("_unified_task_region_") >= 0
    )


def _expand_skipped_agent_task_merge_assignments(
    agent_regions: Dict[int, Sequence[str]],
    tsp_records: Dict[int, Dict[str, object]],
    current_regions: Sequence,
    base_patterns: Dict[str, List[RegionCoveragePattern]],
) -> Tuple[Dict[int, List[str]], List[Dict[str, object]]]:
    region_by_id = {region.region_id: region for region in current_regions}
    expanded_agent_regions = {int(agent_id): list(region_ids) for agent_id, region_ids in agent_regions.items()}
    fallback_records: List[Dict[str, object]] = []
    for raw_agent_id, record in sorted(tsp_records.items()):
        agent_id = int(raw_agent_id)
        replacements: Dict[str, List[str]] = {}
        for skipped_id in record.get("skipped_regions", []) or []:
            region = region_by_id.get(str(skipped_id))
            if region is None or not _is_agent_task_merge_region(region):
                continue
            source_ids = [
                source_id
                for source_id in _agent_task_merge_source_region_ids(region)
                if source_id in base_patterns
            ]
            if not source_ids:
                fallback_records.append(
                    {
                        "agent_id": agent_id,
                        "merged_region_id": str(skipped_id),
                        "source_region_ids": [],
                        "status": "missing_source_patterns",
                    }
                )
                continue
            replacements[str(skipped_id)] = source_ids
            fallback_records.append(
                {
                    "agent_id": agent_id,
                    "merged_region_id": str(skipped_id),
                    "source_region_ids": source_ids,
                    "status": "candidate",
                }
            )
        if not replacements:
            continue
        expanded_ids: List[str] = []
        for region_id in expanded_agent_regions.get(agent_id, []):
            source_ids = replacements.get(region_id)
            if source_ids:
                for source_id in source_ids:
                    if source_id not in expanded_ids:
                        expanded_ids.append(source_id)
            elif region_id not in expanded_ids:
                expanded_ids.append(region_id)
        expanded_agent_regions[agent_id] = expanded_ids
    return expanded_agent_regions, fallback_records


def _selected_pattern_coverage_length(selected_patterns: Dict[str, RegionCoveragePattern]) -> float:
    return sum(float(pattern.coverage_length) for pattern in selected_patterns.values())


def _apply_agent_task_merge_runtime_source_fallback(
    *,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    current_regions: Sequence,
    current_feasible_patterns: Dict[str, List[RegionCoveragePattern]],
    current_sweep_paths: Dict[str, RegionSweepPath],
    current_sweep_segment_templates: Dict[str, Tuple[List[PathSegmentSpec], str]],
    base_regions: Sequence,
    base_feasible_patterns: Dict[str, List[RegionCoveragePattern]],
    base_sweep_paths: Dict[str, RegionSweepPath],
    base_sweep_segment_templates: Dict[str, Tuple[List[PathSegmentSpec], str]],
    assignment: BalancedAssignment,
    ownership_map: CoverageOwnershipMap,
    agents: Dict[int, AgentPathPlan],
    tours: Dict[int, SingleUsvTourPlan],
    tsp_records: Dict[int, Dict[str, object]],
    infeasible_edges: List[Dict[str, object]],
) -> Tuple[
    List,
    Dict[str, List[RegionCoveragePattern]],
    Dict[str, RegionSweepPath],
    Dict[str, Tuple[List[PathSegmentSpec], str]],
    BalancedAssignment,
    CoverageOwnershipMap,
    Dict[str, object],
]:
    expanded_agent_regions, fallback_records = _expand_skipped_agent_task_merge_assignments(
        assignment.agent_regions,
        tsp_records,
        current_regions,
        base_feasible_patterns,
    )
    candidate_records = [record for record in fallback_records if record.get("status") == "candidate"]
    diagnostics: Dict[str, object] = {
        "enabled": bool(candidate_records),
        "candidate_count": len(candidate_records),
        "accepted_count": 0,
        "rejected_count": 0,
        "affected_agent_count": 0,
        "fallback_records": fallback_records,
    }
    if not candidate_records:
        return (
            list(current_regions),
            current_feasible_patterns,
            current_sweep_paths,
            current_sweep_segment_templates,
            assignment,
            ownership_map,
            diagnostics,
        )

    current_region_by_id = {region.region_id: region for region in current_regions}
    base_region_by_id = {region.region_id: region for region in base_regions}
    expanded_region_ids = {
        region_id
        for region_ids in expanded_agent_regions.values()
        for region_id in region_ids
    }
    expanded_region_by_id = {
        region_id: region
        for region_id, region in current_region_by_id.items()
        if region_id in expanded_region_ids
    }
    for region_id in expanded_region_ids:
        if region_id not in expanded_region_by_id and region_id in base_region_by_id:
            expanded_region_by_id[region_id] = base_region_by_id[region_id]
    expanded_regions = [
        region
        for region in list(current_regions) + list(base_regions)
        if region.region_id in expanded_region_by_id
    ]
    seen_regions: set[str] = set()
    expanded_regions = [
        region
        for region in expanded_regions
        if not (region.region_id in seen_regions or seen_regions.add(region.region_id))
    ]
    expanded_feasible_patterns = {
        region_id: list(patterns)
        for region_id, patterns in current_feasible_patterns.items()
        if region_id in expanded_region_ids
    }
    for region_id in expanded_region_ids:
        if region_id not in expanded_feasible_patterns and region_id in base_feasible_patterns:
            expanded_feasible_patterns[region_id] = list(base_feasible_patterns[region_id])
    expanded_sweep_paths = {
        region_id: sweep
        for region_id, sweep in current_sweep_paths.items()
        if region_id in expanded_region_ids
    }
    for region_id in expanded_region_ids:
        if region_id not in expanded_sweep_paths and region_id in base_sweep_paths:
            expanded_sweep_paths[region_id] = base_sweep_paths[region_id]
    expanded_sweep_segment_templates = {
        **base_sweep_segment_templates,
        **current_sweep_segment_templates,
    }
    expanded_graph = build_region_graph(
        expanded_regions,
        expanded_feasible_patterns,
        config,
        obstacle_field=obstacle_field,
        path_config=path_config,
    )
    expanded_assignment = _joint_assignment_from_regions(expanded_agent_regions, expanded_graph)
    expanded_ownership_map = build_coverage_ownership_map(
        expanded_regions,
        expanded_assignment.agent_regions,
        config,
        path_config,
        obstacle_field=obstacle_field,
    )
    affected_agents = sorted({int(record["agent_id"]) for record in candidate_records})
    accepted_agents: set[int] = set()
    rejected_records: List[Dict[str, object]] = []
    accepted_records: List[Dict[str, object]] = []
    for agent_id in affected_agents:
        old_tour = tours.get(agent_id)
        old_record = tsp_records.get(agent_id, {})
        old_coverage = _selected_pattern_coverage_length(old_tour.selected_patterns if old_tour else {})
        agent_fallback_records = [
            record
            for record in candidate_records
            if int(record.get("agent_id", -1)) == agent_id
        ]
        source_expansion_extra_count = sum(
            max(len(record.get("source_region_ids", []) or []) - 1, 0)
            for record in agent_fallback_records
        )
        result = _solve_agent_region_tsp(
            agent_id,
            expanded_assignment.agent_regions.get(agent_id, []),
            expanded_feasible_patterns,
            expanded_sweep_paths,
            expanded_sweep_segment_templates,
            config,
            path_config,
            obstacle_field,
            expanded_ownership_map,
        )
        new_coverage = _selected_pattern_coverage_length(dict(result.get("selected_patterns", {})))
        old_skipped = len(old_record.get("skipped_regions", []) or [])
        new_record = _tsp_record_from_result(
            agent_id,
            expanded_assignment.agent_regions.get(agent_id, []),
            result,
            expanded_sweep_paths,
        )
        new_skipped = len(new_record.get("skipped_regions", []) or [])
        coverage_non_decreasing = new_coverage + 1e-9 >= old_coverage
        direct_improvement = new_coverage > old_coverage + 1e-6 or new_skipped < old_skipped
        neutral_source_expansion = (
            bool(getattr(path_config, "agent_task_runtime_fallback_accept_neutral_source_expansion", True))
            and coverage_non_decreasing
            and bool(agent_fallback_records)
            and new_skipped <= old_skipped + source_expansion_extra_count
        )
        accept = coverage_non_decreasing and (direct_improvement or neutral_source_expansion)
        if accept:
            status = "accepted"
        elif new_coverage + 1e-9 < old_coverage:
            status = "rejected_coverage_loss"
        else:
            status = "rejected_no_coverage_gain"
        record_summary = {
            "agent_id": agent_id,
            "old_coverage_length": round(old_coverage, 6),
            "new_coverage_length": round(new_coverage, 6),
            "old_skipped_count": old_skipped,
            "new_skipped_count": new_skipped,
            "source_expansion_extra_count": source_expansion_extra_count,
            "status": status,
        }
        if not accept:
            rejected_records.append(record_summary)
            continue
        segments = list(result["segments"])
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
        tsp_records[agent_id] = new_record
        accepted_agents.add(agent_id)
        accepted_records.append(record_summary)
    if accepted_agents:
        infeasible_edges[:] = [
            edge
            for edge in infeasible_edges
            if int(edge.get("agent_id", -1)) not in accepted_agents
        ]
        for agent_id in accepted_agents:
            infeasible_edges.extend(tsp_records[agent_id].get("infeasible_edges", []) or [])
        final_agent_regions = {
            agent_id: (
                expanded_assignment.agent_regions.get(agent_id, [])
                if agent_id in accepted_agents
                else list(region_ids)
            )
            for agent_id, region_ids in assignment.agent_regions.items()
        }
        final_region_ids = {region_id for region_ids in final_agent_regions.values() for region_id in region_ids}
        final_regions = [region for region in expanded_regions if region.region_id in final_region_ids]
        final_feasible_patterns = {
            region_id: patterns
            for region_id, patterns in expanded_feasible_patterns.items()
            if region_id in final_region_ids
        }
        final_sweep_paths = {
            region_id: sweep
            for region_id, sweep in expanded_sweep_paths.items()
            if region_id in final_region_ids
        }
        final_graph = build_region_graph(
            final_regions,
            final_feasible_patterns,
            config,
            obstacle_field=obstacle_field,
            path_config=path_config,
        )
        final_assignment = _joint_assignment_from_regions(final_agent_regions, final_graph)
        final_ownership_map = build_coverage_ownership_map(
            final_regions,
            final_assignment.agent_regions,
            config,
            path_config,
            obstacle_field=obstacle_field,
        )
    else:
        final_regions = list(current_regions)
        final_feasible_patterns = current_feasible_patterns
        final_sweep_paths = current_sweep_paths
        expanded_sweep_segment_templates = current_sweep_segment_templates
        final_assignment = assignment
        final_ownership_map = ownership_map
    diagnostics.update(
        {
            "accepted_count": len(accepted_records),
            "rejected_count": len(rejected_records),
            "affected_agent_count": len(affected_agents),
            "accepted_agents": sorted(accepted_agents),
            "accepted_records": accepted_records,
            "rejected_records": rejected_records,
        }
    )
    return (
        final_regions,
        final_feasible_patterns,
        final_sweep_paths,
        expanded_sweep_segment_templates,
        final_assignment,
        final_ownership_map,
        diagnostics,
    )


def _agent_task_merge_unstable_region_ids(
    candidate_regions: Sequence,
    feasible_patterns: Dict[str, List[RegionCoveragePattern]],
    path_config: PathPlanningConfig,
    config: PlannerConfig | None = None,
    obstacle_field: ObstacleField | None = None,
) -> Tuple[List[str], List[Dict[str, object]]]:
    """Find merged agent-task regions whose real sweep is risky for TSP execution.

    The lightweight strip merge uses a geometry preview so it can stay cheap on
    large maps. After real pattern generation, reject merges whose actual pass
    structure no longer provides the promised turn reduction, or whose
    open-chain count exceeds what the downstream connector can assemble.
    """

    unstable_ids: List[str] = []
    records: List[Dict[str, object]] = []
    max_chain_count = max(int(path_config.max_open_chains_per_region), 1)
    for region in candidate_regions:
        metadata = dict(getattr(region, "metadata", {}) or {})
        is_agent_task_merge = (
            metadata.get("agent_task_strip_merge") == "true"
            or metadata.get("agent_task_unified_merge") == "true"
        )
        if not is_agent_task_merge:
            continue
        patterns = list(feasible_patterns.get(region.region_id, []) or [])
        if not patterns:
            continue
        best_pattern = min(
            patterns,
            key=lambda pattern: (
                len(pattern.passes),
                -float(pattern.coverage_length),
                pattern.pattern_id,
            ),
        )
        real_pass_count = len(best_pattern.passes)
        real_max_pass = max((coverage_pass.length for coverage_pass in best_pattern.passes), default=0.0)
        source_pass_count = int(_metadata_float(metadata, "merge_source_pass_count", 0.0))
        preview_pass_count = int(_metadata_float(metadata, "merge_candidate_pass_count", 0.0))
        source_max_pass = _metadata_float(metadata, "merge_source_max_pass_length", 0.0)
        objective_delta = _metadata_float(metadata, "merge_objective_delta", 0.0)
        shape_class = str(metadata.get("shape_class", ""))
        open_chain_count = int(_metadata_float(best_pattern.metadata, "open_chain_count", 0.0))
        open_chain_validation_only = best_pattern.metadata.get("open_chain_validation_only") == "true"

        reasons: List[str] = []
        min_length_gain = max(
            min(float(path_config.agent_task_strip_merge_min_length_gain_factor), 1.10),
            1.0,
        )
        real_length_gain = real_max_pass / max(source_max_pass, 1e-9) if source_max_pass > 1e-9 else 1.0
        real_pass_reduction = max(source_pass_count - real_pass_count, 0)
        real_pass_reduction_ratio = real_pass_reduction / max(source_pass_count, 1)
        pass_count_not_reduced = source_pass_count > 0 and real_pass_count >= source_pass_count
        pass_count_exploded = preview_pass_count > 0 and real_pass_count > max(preview_pass_count * 1.5, preview_pass_count + 4)
        strong_long_pass_gain = source_max_pass > 1e-9 and real_length_gain >= max(min_length_gain, 1.20)
        coherent_boustrophedon_gain = _agent_task_has_coherent_boustrophedon_gain(
            source_pass_count,
            real_pass_count,
            source_max_pass,
            real_max_pass,
        )
        if pass_count_not_reduced and not strong_long_pass_gain:
            reasons.append("real_pass_count_not_reduced")
        if pass_count_exploded and not strong_long_pass_gain:
            reasons.append("preview_underestimated_pass_count")
        if source_max_pass > 1e-9 and real_length_gain < min_length_gain:
            reasons.append("real_long_pass_gain_too_low")
        if open_chain_count > max_chain_count and not strong_long_pass_gain:
            reasons.append("too_many_open_chains")
        if (
            open_chain_validation_only
            and open_chain_count > max(max_chain_count // 2, 8)
            and not strong_long_pass_gain
        ):
            reasons.append("open_chain_validation_only_high_chain_count")
        keep_coherent_negative_objective = (
            bool(getattr(path_config, "agent_task_merge_keep_coherent_negative_objective", True))
            and coherent_boustrophedon_gain
            and not pass_count_exploded
            and open_chain_count <= max(max_chain_count, 8)
        )
        if objective_delta < -1e-9 and shape_class != "rectangle" and not keep_coherent_negative_objective:
            # A non-rectangular merge that loses the preview objective is a poor
            # TSP node unless its real sweep still delivers the requested
            # coherent boustrophedon behavior: fewer passes and longer runs.
            reasons.append("nonrectangular_negative_objective_delta")
        internal_execution_available = False
        internal_execution_reason = ""
        internal_execution_probe_count = 0
        if config is not None and open_chain_validation_only:
            (
                internal_execution_available,
                internal_execution_reason,
                internal_execution_probe_count,
            ) = _agent_task_merge_internal_execution_available(
                region.region_id,
                patterns,
                config,
                path_config,
                obstacle_field,
            )
            if internal_execution_available:
                reasons = [
                    reason
                    for reason in reasons
                    if reason
                    not in {
                        "too_many_open_chains",
                        "open_chain_validation_only_high_chain_count",
                    }
                ]
            else:
                reasons.append("open_chain_execution_unavailable")

        if not reasons:
            continue
        unstable_ids.append(region.region_id)
        records.append(
            {
                "region_id": region.region_id,
                "source_region_ids": [
                    item.strip()
                    for item in str(
                        metadata.get("agent_task_strip_source_ids")
                        or metadata.get("agent_task_unified_source_ids")
                        or metadata.get("source_region_ids")
                        or ""
                    ).split(",")
                    if item.strip()
                ],
                "shape_class": shape_class,
                "reason": ",".join(reasons),
                "source_pass_count": source_pass_count,
                "preview_pass_count": preview_pass_count,
                "real_pass_count": real_pass_count,
                "real_pass_reduction": real_pass_reduction,
                "real_pass_reduction_ratio": round(real_pass_reduction_ratio, 6),
                "source_max_pass_length": round(source_max_pass, 6),
                "real_max_pass_length": round(real_max_pass, 6),
                "real_long_pass_gain_ratio": round(real_length_gain, 6),
                "coherent_boustrophedon_gain": bool(coherent_boustrophedon_gain),
                "open_chain_count": open_chain_count,
                "open_chain_validation_only": bool(open_chain_validation_only),
                "internal_execution_available": bool(internal_execution_available),
                "internal_execution_reason": internal_execution_reason,
                "internal_execution_probe_count": internal_execution_probe_count,
                "objective_delta": round(objective_delta, 6),
            }
        )
    return unstable_ids, records


def _agent_task_merge_internal_execution_available(
    region_id: str,
    patterns: Sequence[RegionCoveragePattern],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> Tuple[bool, str, int]:
    """Probe whether an agent-task merged open-chain pattern can really run.

    Pattern validation may mark a region as ``open_chain_validation_only`` after
    finding chainable cover passes. The main TSP still needs an executable chain
    order and a valid exit pose. This probe keeps merged regions that have at
    least one executable normal or flexible-exit variant, and rejects the rest
    before they can consume a large TSP node.
    """

    probe_patterns = sorted(
        patterns,
        key=lambda pattern: (
            pattern.metadata.get("open_chain_validation_only") == "true",
            len(pattern.passes),
            -float(pattern.coverage_length),
            pattern.pattern_id,
        ),
    )
    max_probe_count = max(1, min(len(probe_patterns), 4))
    failure_reasons: List[str] = []
    for probe_idx, pattern in enumerate(probe_patterns[:max_probe_count]):
        probe_pattern = copy.deepcopy(pattern)
        probe_config = _internal_sweep_execution_path_config(
            probe_pattern,
            replace(path_config, enable_open_sweep_chain_tsp=True),
        )
        segments, reason = _build_internal_sweep_segments(
            probe_pattern,
            config,
            probe_config,
            obstacle_field,
            start_time=0.0,
            segment_prefix=f"agent_task_merge_stability_{region_id}_{probe_idx}",
        )
        if segments and not reason:
            return True, "", probe_idx + 1
        if reason:
            failure_reasons.append(reason)
        else:
            failure_reasons.append("empty_internal_sweep")
    return False, ",".join(failure_reasons[:4]) or "internal_execution_unavailable", max_probe_count


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


def _heterogeneous_decomposition_config(config: PlannerConfig) -> PlannerConfig:
    agent_count = config.fleet.num_agents or len(config.fleet.initial_states_3dof)
    profiles = [config.profile_for_agent(agent_id) for agent_id in range(agent_count)]
    if not profiles:
        return config
    return replace(
        config,
        footprint=replace(
            config.footprint,
            length_lf=min(profile.coverage_length for profile in profiles),
            width_wf=min(profile.coverage_width for profile in profiles),
        ),
        vehicle_footprint=VehicleFootprint(
            length=min(profile.vehicle_length for profile in profiles),
            width=min(profile.vehicle_width for profile in profiles),
        ),
        active_agent_id=None,
    )


def _heterogeneous_full_mission_region(config: PlannerConfig) -> DecomposedRegion:
    width = config.mission.area_length_x
    height = config.mission.area_length_y
    preferred_axis = "x" if width >= height else "y"
    return DecomposedRegion(
        region_id="heterogeneous_mission_region_0",
        bounds=(0.0, 0.0, width, height),
        polygon=[(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)],
        center=(width / 2.0, height / 2.0),
        area=width * height,
        preferred_axis=preferred_axis,
        source_algorithm="heterogeneous_topology_only_decomposition",
        metadata={
            "shape_class": "rectangle",
            "convex_region_decomposition": "true",
            "heterogeneous_topology_only": "true",
            "dominant_scan_axis": preferred_axis,
            "support_span": f"{height if preferred_axis == 'x' else width:.6f}",
        },
    )


def _split_oversized_heterogeneous_regions(
    regions: Sequence[DecomposedRegion],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> Tuple[List[DecomposedRegion], List[Dict[str, object]]]:
    if len(regions) < 1 or (config.fleet.num_agents or 0) < 2:
        return list(regions), []
    agent_count = config.fleet.num_agents or len(config.fleet.initial_states_3dof)
    profiles = [config.profile_for_agent(agent_id) for agent_id in range(agent_count)]
    fastest_rates = [max(profile.effective_strip_spacing * profile.cover_speed, 1e-6) for profile in profiles]
    region_work = {
        region.region_id: region.area / max(fastest_rates)
        for region in regions
    }
    average_target = sum(region_work.values()) / max(agent_count, 1)
    threshold = average_target * max(float(path_config.oversized_region_split_ratio), 1.0)
    if threshold <= 1e-9:
        return list(regions), []

    result: List[DecomposedRegion] = []
    records: List[Dict[str, object]] = []
    for region in regions:
        work = region_work[region.region_id]
        split_count = min(max(int(math.ceil(work / threshold)), 1), max(agent_count, 2))
        if split_count <= 1 or region.area <= 2.0 * min(profile.coverage_width for profile in profiles) ** 2:
            result.append(region)
            continue
        children = _split_region_across_scan_lines(region, split_count)
        if len(children) <= 1 or sum(child.area for child in children) + 1e-6 < region.area * 0.995:
            result.append(region)
            continue
        result.extend(children)
        records.append(
            {
                "parent_region_id": region.region_id,
                "parent_estimated_work": work,
                "average_target_work": average_target,
                "split_threshold": threshold,
                "child_count": len(children),
                "child_region_ids": [child.region_id for child in children],
                "split_axis": "y" if region.preferred_axis == "x" else "x",
                "reason": "estimated_work_exceeds_average_target",
            }
        )
    return result, records


def _split_region_across_scan_lines(region: DecomposedRegion, split_count: int) -> List[DecomposedRegion]:
    split_axis = "y" if region.preferred_axis == "x" else "x"
    low = region.bounds[1] if split_axis == "y" else region.bounds[0]
    high = region.bounds[3] if split_axis == "y" else region.bounds[2]
    if high - low <= 1e-9:
        return [region]
    edges = [low + (high - low) * index / split_count for index in range(split_count + 1)]
    cells = list(getattr(region, "member_cells", []) or [])
    children: List[DecomposedRegion] = []
    for index, (band_low, band_high) in enumerate(zip(edges[:-1], edges[1:])):
        if cells:
            selected_cells = [
                cell
                for cell in cells
                if band_low - 1e-9
                <= (cell.center[1] if split_axis == "y" else cell.center[0])
                and (
                    (cell.center[1] if split_axis == "y" else cell.center[0]) < band_high - 1e-9
                    or index == split_count - 1
                )
            ]
            if not selected_cells:
                continue
            bounds = _cell_group_bounds_for_merge(selected_cells)
            area = sum(cell.area for cell in selected_cells)
            child: DecomposedRegion = CompositeFreeSpaceRegion(
                region_id=f"{region.region_id}_worksplit_{index}",
                bounds=bounds,
                polygon=_bounds_polygon(bounds),
                center=(
                    sum(cell.center[0] * cell.area for cell in selected_cells) / max(area, 1e-9),
                    sum(cell.center[1] * cell.area for cell in selected_cells) / max(area, 1e-9),
                ),
                area=area,
                preferred_axis=region.preferred_axis,
                source_algorithm="heterogeneous_oversized_region_split",
                member_cells=selected_cells,
                metadata={
                    **region.metadata,
                    "controlled_workload_split": "true",
                    "controlled_split_parent": region.region_id,
                    "controlled_split_index": str(index),
                },
            )
        else:
            polygon = _clip_polygon_to_axis_band(region.polygon, split_axis, band_low, band_high)
            area = abs(_polygon_signed_area(polygon))
            if len(polygon) < 3 or area <= 1e-9:
                continue
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            bounds = (min(xs), min(ys), max(xs), max(ys))
            child = DecomposedRegion(
                region_id=f"{region.region_id}_worksplit_{index}",
                bounds=bounds,
                polygon=polygon,
                center=_polygon_centroid(polygon),
                area=area,
                preferred_axis=region.preferred_axis,
                source_algorithm="heterogeneous_oversized_region_split",
                metadata={
                    **region.metadata,
                    "controlled_workload_split": "true",
                    "controlled_split_parent": region.region_id,
                    "controlled_split_index": str(index),
                },
            )
        children.append(child)
    return children or [region]


def _clip_polygon_to_axis_band(
    polygon: Sequence[Tuple[float, float]],
    axis: str,
    low: float,
    high: float,
) -> List[Tuple[float, float]]:
    clipped = list(polygon)
    coordinate_index = 0 if axis == "x" else 1
    for boundary, keep_greater in ((low, True), (high, False)):
        if not clipped:
            break
        output: List[Tuple[float, float]] = []
        previous = clipped[-1]
        previous_inside = (
            previous[coordinate_index] >= boundary - 1e-9
            if keep_greater
            else previous[coordinate_index] <= boundary + 1e-9
        )
        for current in clipped:
            current_inside = (
                current[coordinate_index] >= boundary - 1e-9
                if keep_greater
                else current[coordinate_index] <= boundary + 1e-9
            )
            if current_inside != previous_inside:
                denominator = current[coordinate_index] - previous[coordinate_index]
                ratio = 0.0 if abs(denominator) <= 1e-12 else (boundary - previous[coordinate_index]) / denominator
                intersection = (
                    previous[0] + ratio * (current[0] - previous[0]),
                    previous[1] + ratio * (current[1] - previous[1]),
                )
                output.append(intersection)
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
        clipped = output
    return clipped


def _polygon_signed_area(polygon: Sequence[Tuple[float, float]]) -> float:
    return 0.5 * sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(polygon, list(polygon[1:]) + list(polygon[:1]))
    )


def _polygon_centroid(polygon: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    signed_area = _polygon_signed_area(polygon)
    if abs(signed_area) <= 1e-12:
        return (
            sum(point[0] for point in polygon) / max(len(polygon), 1),
            sum(point[1] for point in polygon) / max(len(polygon), 1),
        )
    factor = 1.0 / (6.0 * signed_area)
    cx = 0.0
    cy = 0.0
    for first, second in zip(polygon, list(polygon[1:]) + list(polygon[:1])):
        cross = first[0] * second[1] - second[0] * first[1]
        cx += (first[0] + second[0]) * cross
        cy += (first[1] + second[1]) * cross
    return cx * factor, cy * factor


def _coverage_aware_merge_regions(
    regions: Sequence,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None = None,
    diagnostics: Dict[str, object] | None = None,
    region_id_prefix: str = "coverage_merge_region",
    progress_callback: Callable[..., None] | None = None,
) -> List:
    """Greedily merge adjacent regions only when coverage economics improve.

    The merge is deliberately conservative: geometry only proposes candidates;
    actual acceptance is driven by the same pattern generation and internal
    sweep validation used by the main planner.
    """

    current = list(regions)
    rejected: Dict[str, int] = {}
    rejected_records: List[Dict[str, object]] = []
    accepted_records: List[Dict[str, object]] = []
    candidate_count = 0
    validation_count = 0
    iteration_count = 0
    no_improvement_rounds = 0
    objective_delta = 0.0
    budget_exhausted = False
    budget_reason = ""
    merge_started = time.perf_counter()
    preview_cache: Dict[Tuple[object, ...], Dict[str, object]] = {}
    before_count = len(current)
    if diagnostics is not None:
        diagnostics["coverage_merge_region_count_before"] = before_count
        diagnostics["coverage_merge_status"] = "running"
    if len(current) < 2 or not path_config.enable_coverage_aware_merge:
        status = "disabled" if not path_config.enable_coverage_aware_merge else "skipped_small_region_set"
        if diagnostics is not None:
            diagnostics.update(
                {
                    "coverage_merge_status": status,
                    "coverage_merge_budget_exhausted": False,
                    "coverage_merge_budget_reason": "",
                    "coverage_merge_elapsed_sec": time.perf_counter() - merge_started,
                    "coverage_merge_region_count_after": len(current),
                    "coverage_merge_candidate_count": 0,
                    "coverage_merge_validation_count": 0,
                    "coverage_merge_iteration_count": 0,
                    "coverage_merge_no_improvement_round_count": 0,
                    "coverage_merge_accepted_count": 0,
                    "coverage_merge_rejected_by_reason": rejected,
                    "coverage_merge_rejected_candidates": rejected_records,
                    "coverage_merge_objective_delta": 0.0,
                    "coverage_merge_regions": [],
                }
            )
        return current

    serial = 0
    max_iterations = max(0, len(current) - 1)
    beam_width = max(int(path_config.coverage_merge_beam_width), 1)
    validate_top_k = max(int(path_config.coverage_merge_validate_top_k), 1)
    max_candidate_evaluations = int(path_config.coverage_merge_max_candidate_evaluations)
    max_validations = int(path_config.coverage_merge_max_validations)
    time_budget_sec = float(path_config.coverage_merge_time_budget_sec)
    no_improvement_patience = max(int(path_config.coverage_merge_no_improvement_patience), 1)

    def current_budget_reason() -> str:
        if max_candidate_evaluations > 0 and candidate_count >= max_candidate_evaluations:
            return "candidate_budget_exhausted"
        if max_validations > 0 and validation_count >= max_validations:
            return "validation_budget_exhausted"
        if time_budget_sec > 0.0 and time.perf_counter() - merge_started >= time_budget_sec:
            return "time_budget_exhausted"
        return ""

    def emit_progress(event: str, **extra: object) -> None:
        if progress_callback is None:
            return
        progress_callback(
            event=event,
            iteration=iteration_count,
            candidate_count=candidate_count,
            validation_count=validation_count,
            accepted_count=len(accepted_records),
            rejected_count=sum(rejected.values()),
            elapsed_sec=round(time.perf_counter() - merge_started, 3),
            budget_exhausted=budget_exhausted,
            budget_reason=budget_reason,
            **extra,
        )

    for _ in range(max_iterations):
        iteration_count += 1
        reason = current_budget_reason()
        if reason:
            budget_exhausted = True
            budget_reason = reason
            emit_progress("budget_stop")
            break
        geometric_candidates: List[Tuple[float, float, int, int, object]] = []
        for first_idx, first in enumerate(current):
            if current_budget_reason():
                break
            for second_idx in range(first_idx + 1, len(current)):
                reason = current_budget_reason()
                if reason:
                    budget_exhausted = True
                    budget_reason = reason
                    break
                second = current[second_idx]
                adjacent, reason = _coverage_merge_regions_can_join(first, second, config, path_config, obstacle_field)
                if not adjacent:
                    _increment_object_diagnostic(rejected, reason)
                    continue
                candidate, reason = _coverage_merge_candidate_from_group(
                    serial,
                    [first, second],
                    config,
                    path_config,
                    obstacle_field,
                    region_id_prefix=region_id_prefix,
                )
                candidate_count += 1
                if candidate is None:
                    _increment_object_diagnostic(rejected, reason)
                    _coverage_merge_record_rejected_candidate(
                        rejected_records,
                        reason,
                        [first, second],
                        path_config,
                    )
                    continue
                boundary_turn_proxy = _coverage_merge_boundary_turn_proxy([first, second], path_config)
                geometric_candidates.append((boundary_turn_proxy, float(candidate.area), first_idx, second_idx, candidate))
            if budget_exhausted:
                break
        if not geometric_candidates:
            no_improvement_rounds += 1
            emit_progress("no_geometric_candidates", no_improvement_rounds=no_improvement_rounds)
            break

        geometric_candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        preview_limit = max(beam_width * max(len(current), 1), beam_width)
        scored: List[Tuple[float, float, int, int, object, Dict[str, object], Dict[str, object]]] = []
        for _, _, first_idx, second_idx, candidate in geometric_candidates[:preview_limit]:
            reason = current_budget_reason()
            if reason:
                budget_exhausted = True
                budget_reason = reason
                break
            sources = [current[first_idx], current[second_idx]]
            before_preview = _coverage_merge_before_preview(sources, config, path_config, obstacle_field, preview_cache)
            after_preview = _coverage_merge_pattern_preview(
                candidate,
                config,
                path_config,
                obstacle_field,
                preview_cache,
                validate_internal=False,
            )
            if not after_preview.get("feasible", False):
                reason = str(after_preview.get("reason", "no_feasible_pattern"))
                _increment_object_diagnostic(rejected, reason)
                _coverage_merge_record_rejected_candidate(
                    rejected_records,
                    reason,
                    sources,
                    path_config,
                    candidate=candidate,
                    before_preview=before_preview,
                    after_preview=after_preview,
                )
                continue
            coverage_fraction = float(after_preview.get("coverage_fraction", 0.0) or 0.0)
            if coverage_fraction + 1e-9 < max(0.0, min(1.0, path_config.coverage_merge_min_coverage_fraction)):
                _increment_object_diagnostic(rejected, "low_coverage_fraction")
                _coverage_merge_record_rejected_candidate(
                    rejected_records,
                    "low_coverage_fraction",
                    sources,
                    path_config,
                    candidate=candidate,
                    before_preview=before_preview,
                    after_preview=after_preview,
                )
                continue
            before_score = float(before_preview.get("score", 0.0) or 0.0)
            after_score = float(after_preview.get("score", 0.0) or 0.0)
            improvement = before_score - after_score
            improvement_ratio = improvement / max(abs(before_score), 1e-9)
            if improvement_ratio + 1e-9 < max(path_config.coverage_merge_min_improvement_ratio, 0.0):
                _increment_object_diagnostic(rejected, "low_objective_improvement")
                _coverage_merge_record_rejected_candidate(
                    rejected_records,
                    "low_objective_improvement",
                    sources,
                    path_config,
                    candidate=candidate,
                    before_preview=before_preview,
                    after_preview=after_preview,
                )
                continue
            scored.append((improvement, improvement_ratio, first_idx, second_idx, candidate, before_preview, after_preview))
        if budget_exhausted:
            emit_progress("budget_stop")
            break
        if not scored:
            no_improvement_rounds += 1
            emit_progress("no_scored_candidates", no_improvement_rounds=no_improvement_rounds)
            if no_improvement_rounds >= no_improvement_patience:
                break
            break

        scored.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        accepted = None
        for improvement, improvement_ratio, first_idx, second_idx, candidate, before_preview, after_preview in scored[:validate_top_k]:
            reason = current_budget_reason()
            if reason:
                budget_exhausted = True
                budget_reason = reason
                break
            validation_count += 1
            validated = _coverage_merge_pattern_preview(
                candidate,
                config,
                path_config,
                obstacle_field,
                preview_cache,
                validate_internal=True,
            )
            if not validated.get("feasible", False):
                reason = str(validated.get("reason", "internal_sweep_infeasible"))
                _increment_object_diagnostic(rejected, reason)
                _coverage_merge_record_rejected_candidate(
                    rejected_records,
                    reason,
                    [current[first_idx], current[second_idx]],
                    path_config,
                    candidate=candidate,
                    before_preview=before_preview,
                    after_preview=validated,
                )
                continue
            coverage_fraction = float(validated.get("coverage_fraction", after_preview.get("coverage_fraction", 0.0)) or 0.0)
            if coverage_fraction + 1e-9 < max(0.0, min(1.0, path_config.coverage_merge_min_coverage_fraction)):
                _increment_object_diagnostic(rejected, "validated_low_coverage_fraction")
                _coverage_merge_record_rejected_candidate(
                    rejected_records,
                    "validated_low_coverage_fraction",
                    [current[first_idx], current[second_idx]],
                    path_config,
                    candidate=candidate,
                    before_preview=before_preview,
                    after_preview=validated,
                )
                continue
            after_score = float(validated.get("score", after_preview.get("score", 0.0)) or 0.0)
            before_score = float(before_preview.get("score", 0.0) or 0.0)
            validated_improvement = before_score - after_score
            validated_ratio = validated_improvement / max(abs(before_score), 1e-9)
            if validated_ratio + 1e-9 < max(path_config.coverage_merge_min_improvement_ratio, 0.0):
                _increment_object_diagnostic(rejected, "validated_low_objective_improvement")
                _coverage_merge_record_rejected_candidate(
                    rejected_records,
                    "validated_low_objective_improvement",
                    [current[first_idx], current[second_idx]],
                    path_config,
                    candidate=candidate,
                    before_preview=before_preview,
                    after_preview=validated,
                )
                continue
            accepted = (validated_improvement, validated_ratio, first_idx, second_idx, candidate, before_preview, validated)
            break
        if budget_exhausted:
            emit_progress("budget_stop")
            break
        if accepted is None:
            no_improvement_rounds += 1
            emit_progress("no_validated_candidate", no_improvement_rounds=no_improvement_rounds)
            if no_improvement_rounds >= no_improvement_patience:
                break
            break

        improvement, improvement_ratio, first_idx, second_idx, candidate, before_preview, after_preview = accepted
        source_ids = [current[first_idx].region_id, current[second_idx].region_id]
        candidate.metadata["coverage_aware_merged"] = "true"
        candidate.metadata["merge_accept_reason"] = "coverage_objective_improved"
        candidate.metadata["merge_objective_before"] = f"{float(before_preview.get('score', 0.0) or 0.0):.6f}"
        candidate.metadata["merge_objective_after"] = f"{float(after_preview.get('score', 0.0) or 0.0):.6f}"
        candidate.metadata["merge_objective_delta"] = f"{improvement:.6f}"
        candidate.metadata["merge_objective_delta_ratio"] = f"{improvement_ratio:.6f}"
        candidate.metadata["merge_best_pattern_id"] = str(after_preview.get("pattern_id", ""))
        candidate.metadata["merge_best_pattern_scan_axis"] = str(after_preview.get("scan_axis", ""))
        candidate.metadata["merge_best_pattern_coverage_fraction"] = f"{float(after_preview.get('coverage_fraction', 0.0) or 0.0):.6f}"
        candidate.metadata["merge_source_pass_count"] = str(int(before_preview.get("pass_count", 0) or 0))
        candidate.metadata["merge_candidate_pass_count"] = str(int(after_preview.get("pass_count", 0) or 0))
        candidate.metadata["merge_source_max_pass_length"] = f"{float(before_preview.get('max_pass_length', 0.0) or 0.0):.6f}"
        candidate.metadata["merge_candidate_max_pass_length"] = f"{float(after_preview.get('max_pass_length', 0.0) or 0.0):.6f}"
        accepted_records.append(
            {
                "region_id": candidate.region_id,
                "source_region_ids": source_ids,
                "source_region_count": len(source_ids),
                "shape_class": candidate.metadata.get("shape_class", ""),
                "scan_support_mode": candidate.metadata.get("scan_support_mode", ""),
                "objective_before": round(float(before_preview.get("score", 0.0) or 0.0), 6),
                "objective_after": round(float(after_preview.get("score", 0.0) or 0.0), 6),
                "objective_delta": round(improvement, 6),
                "boundary_turn_proxy": round(float(before_preview.get("boundary_turn_proxy", 0.0) or 0.0), 6),
                "coverage_fraction": round(float(after_preview.get("coverage_fraction", 0.0) or 0.0), 6),
                "source_pass_count": int(before_preview.get("pass_count", 0) or 0),
                "candidate_pass_count": int(after_preview.get("pass_count", 0) or 0),
                "source_max_pass_length": round(float(before_preview.get("max_pass_length", 0.0) or 0.0), 6),
                "candidate_max_pass_length": round(float(after_preview.get("max_pass_length", 0.0) or 0.0), 6),
                "best_pattern_scan_axis": after_preview.get("scan_axis", ""),
            }
        )
        objective_delta += improvement
        serial += 1
        remove = {first_idx, second_idx}
        current = [region for idx, region in enumerate(current) if idx not in remove]
        current.append(candidate)
        current.sort(key=lambda region: (region.center[0], region.center[1], region.region_id))
        _populate_region_neighbors(current)
        no_improvement_rounds = 0
        emit_progress(
            "accepted",
            accepted_region_id=candidate.region_id,
            region_count=len(current),
            objective_delta=round(objective_delta, 6),
        )

    if diagnostics is not None:
        if budget_exhausted:
            status = "budget_fallback"
        elif accepted_records:
            status = "success"
        else:
            status = "no_improvement"
        diagnostics.update(
            {
                "coverage_merge_status": status,
                "coverage_merge_budget_exhausted": bool(budget_exhausted),
                "coverage_merge_budget_reason": budget_reason,
                "coverage_merge_elapsed_sec": time.perf_counter() - merge_started,
                "coverage_merge_region_count_before": before_count,
                "coverage_merge_region_count_after": len(current),
                "coverage_merge_candidate_count": candidate_count,
                "coverage_merge_validation_count": validation_count,
                "coverage_merge_iteration_count": iteration_count,
                "coverage_merge_no_improvement_round_count": no_improvement_rounds,
                "coverage_merge_accepted_count": len(accepted_records),
                "coverage_merge_rejected_by_reason": rejected,
                "coverage_merge_rejected_candidates": rejected_records,
                "coverage_merge_objective_delta": objective_delta,
                "coverage_merge_regions": accepted_records,
            }
        )
    emit_progress("done", status="budget_fallback" if budget_exhausted else ("success" if accepted_records else "no_improvement"))
    _populate_region_neighbors(current)
    return current


def _heterogeneous_monotone_task_recombination(
    regions: Sequence[DecomposedRegion],
    assignment: BalancedAssignment,
    agent_patterns: Dict[int, Dict[str, List[RegionCoveragePattern]]],
    agent_paths: Dict[int, Dict[str, RegionSweepPath]],
    agent_templates: Dict[int, Dict[str, Tuple[List[PathSegmentSpec], str]]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    agent_obstacle_fields: Dict[int, ObstacleField | None],
    progress_callback: Callable[..., None] | None = None,
) -> Tuple[
    List[DecomposedRegion],
    Dict[str, List[RegionCoveragePattern]],
    Dict[str, RegionSweepPath],
    Dict[str, Tuple[List[PathSegmentSpec], str]],
    BalancedAssignment,
    Dict[str, object],
]:
    """Rebuild each heterogeneous agent's connected allocation as monotone tasks."""

    region_by_id = {region.region_id: region for region in regions}
    final_regions: List[DecomposedRegion] = []
    final_agent_regions: Dict[int, List[str]] = {agent_id: [] for agent_id in assignment.agent_regions}
    final_patterns: Dict[str, List[RegionCoveragePattern]] = {}
    final_paths: Dict[str, RegionSweepPath] = {}
    final_templates: Dict[str, Tuple[List[PathSegmentSpec], str]] = {}
    accepted_records: List[Dict[str, object]] = []
    rejected_records: List[Dict[str, object]] = []
    candidate_count = 0

    for agent_id, region_ids in sorted(assignment.agent_regions.items()):
        source_regions = [region_by_id[region_id] for region_id in region_ids if region_id in region_by_id]
        if len(source_regions) < 2:
            final_regions.extend(source_regions)
            final_agent_regions[agent_id].extend(region.region_id for region in source_regions)
            for region in source_regions:
                final_patterns[region.region_id] = list(agent_patterns.get(agent_id, {}).get(region.region_id, []))
                if region.region_id in agent_paths.get(agent_id, {}):
                    final_paths[region.region_id] = agent_paths[agent_id][region.region_id]
            final_templates.update(agent_templates.get(agent_id, {}))
            continue

        agent_config = config.for_agent(agent_id)
        agent_field = agent_obstacle_fields.get(agent_id)
        local_merge_config = replace(
            path_config,
            coverage_merge_max_area_fraction=1.0,
            agent_task_merge_max_area_fraction=1.0,
            agent_task_merge_min_unified_group_size=2,
            coverage_merge_min_improvement_ratio=max(
                path_config.coverage_merge_min_improvement_ratio,
                path_config.monotone_merge_min_time_gain_ratio,
            ),
            agent_task_merge_min_improvement_ratio=max(
                path_config.agent_task_merge_min_improvement_ratio,
                path_config.monotone_merge_min_time_gain_ratio,
            ),
            agent_task_merge_max_unified_candidates_per_agent=max(
                path_config.agent_task_merge_max_unified_candidates_per_agent,
                path_config.monotone_merge_beam_width,
            ),
        )
        candidate_regions, candidate_assignment, local_diagnostics = _merge_assigned_agent_task_regions(
            source_regions,
            {agent_id: list(region_ids)},
            agent_config,
            local_merge_config,
            agent_field,
            progress_callback=progress_callback,
        )
        candidate_count += int(local_diagnostics.get("agent_task_merge_candidate_count", 0) or 0)
        local_raw = _generate_paper_style_patterns(
            candidate_regions,
            agent_config,
            local_merge_config,
            agent_field,
        )
        local_stats: Dict[str, object] = {}
        local_paths, local_patterns, local_infeasible, local_templates = _build_region_sweep_paths(
            local_raw,
            agent_config,
            local_merge_config,
            agent_field,
            stats=local_stats,
        )
        local_output_regions: List[DecomposedRegion] = []
        for candidate in candidate_regions:
            source_ids = _recombined_source_ids(candidate)
            is_recombined = len(source_ids) >= 2
            if not is_recombined:
                local_output_regions.append(candidate)
                continue
            monotone_axis = _monotone_scan_axis(candidate, path_config.monotone_merge_angle_tolerance_deg)
            failure_reason = ""
            if monotone_axis is None:
                failure_reason = "union_not_scan_monotone"
            elif candidate.region_id not in local_patterns:
                failure_reason = "merged_sweep_infeasible"
            else:
                before_candidates = [
                    min(
                        agent_patterns.get(agent_id, {}).get(source_id, []),
                        key=lambda pattern: (pattern.estimated_time, pattern.turn_angle, pattern.pattern_id),
                        default=None,
                    )
                    for source_id in source_ids
                ]
                before_candidates = [pattern for pattern in before_candidates if pattern is not None]
                after = min(
                    local_patterns[candidate.region_id],
                    key=lambda pattern: (pattern.estimated_time, pattern.turn_angle, pattern.pattern_id),
                )
                before_time = sum(pattern.estimated_time for pattern in before_candidates)
                before_turns = sum(max(len(pattern.passes) - 1, 0) for pattern in before_candidates)
                after_time = after.estimated_time
                after_turns = max(len(after.passes) - 1, 0)
                source_coverage = min(
                    (_estimated_pattern_coverage_fraction(pattern, agent_config) for pattern in before_candidates),
                    default=0.0,
                )
                after_coverage = _estimated_pattern_coverage_fraction(after, agent_config)
                open_breaks = int(_metadata_float(after.metadata, "open_chain_break_count", 0.0))
                time_gain = (before_time - after_time) / max(before_time, 1e-9)
                valid_gain = (
                    time_gain + 1e-9 >= path_config.monotone_merge_min_time_gain_ratio
                    or (after_time <= before_time * 1.02 + 1e-9 and after_turns < before_turns)
                )
                if after_coverage + 1e-9 < source_coverage:
                    failure_reason = "formal_coverage_decreased"
                elif path_config.require_connected_sweep_task and open_breaks > 0:
                    failure_reason = "merged_task_not_single_chain"
                elif not valid_gain:
                    failure_reason = "insufficient_time_or_turn_gain"
                else:
                    if monotone_axis in {"x", "y"}:
                        candidate.preferred_axis = monotone_axis
                    candidate.metadata.update(
                        {
                            "monotone_merge_verified": "true",
                            "monotone_scan_axis": monotone_axis,
                            "monotone_angle_tolerance_deg": f"{path_config.monotone_merge_angle_tolerance_deg:.6f}",
                            "merge_time_before": f"{before_time:.6f}",
                            "merge_time_after": f"{after_time:.6f}",
                            "merge_time_gain_ratio": f"{time_gain:.6f}",
                            "merge_turn_count_before": str(before_turns),
                            "merge_turn_count_after": str(after_turns),
                        }
                    )
                    accepted_records.append(
                        {
                            "agent_id": agent_id,
                            "region_id": candidate.region_id,
                            "source_region_ids": source_ids,
                            "monotone_scan_axis": monotone_axis,
                            "time_before": before_time,
                            "time_after": after_time,
                            "time_gain_ratio": time_gain,
                            "turn_count_before": before_turns,
                            "turn_count_after": after_turns,
                        }
                    )
            if not failure_reason:
                local_output_regions.append(candidate)
                continue
            rejected_records.append(
                {
                    "agent_id": agent_id,
                    "region_id": candidate.region_id,
                    "source_region_ids": source_ids,
                    "reason": failure_reason,
                }
            )
            local_output_regions.extend(region_by_id[source_id] for source_id in source_ids if source_id in region_by_id)

        accepted_ids = {region.region_id for region in local_output_regions if region.region_id in local_patterns}
        final_regions.extend(local_output_regions)
        for region in local_output_regions:
            final_agent_regions[agent_id].append(region.region_id)
            if region.region_id in local_patterns:
                final_patterns[region.region_id] = local_patterns[region.region_id]
                final_paths[region.region_id] = local_paths[region.region_id]
            else:
                final_patterns[region.region_id] = list(agent_patterns.get(agent_id, {}).get(region.region_id, []))
                if region.region_id in agent_paths.get(agent_id, {}):
                    final_paths[region.region_id] = agent_paths[agent_id][region.region_id]
        for key, value in local_templates.items():
            if any(region_id in key for region_id in accepted_ids):
                final_templates[key] = value
        final_templates.update(agent_templates.get(agent_id, {}))

    _populate_region_neighbors(final_regions)
    final_graph = build_region_graph(final_regions, final_patterns, config, obstacle_field=None, path_config=path_config)
    final_assignment = _joint_assignment_from_regions(final_agent_regions, final_graph)
    diagnostics: Dict[str, object] = {
        "agent_task_merge_enabled": True,
        "agent_task_merge_status": "success" if accepted_records else "no_improvement",
        "agent_task_merge_region_count_before": len(regions),
        "agent_task_merge_region_count_after": len(final_regions),
        "agent_task_merge_candidate_count": candidate_count,
        "agent_task_merge_accepted_count": len(accepted_records),
        "agent_task_merge_rejected_count": len(rejected_records),
        "agent_task_merge_regions": accepted_records,
        "agent_task_merge_rejected_candidates": rejected_records,
        "monotone_merge_angle_tolerance_deg": path_config.monotone_merge_angle_tolerance_deg,
        "monotone_merge_require_single_chain": path_config.require_connected_sweep_task,
        "agent_task_strip_merge_enabled": bool(path_config.enable_agent_task_lightweight_strip_merge),
        "agent_task_unified_merge_enabled": bool(path_config.agent_task_merge_enable_unified_group_merge),
        "agent_task_merge_pairwise_fallback_enabled": bool(path_config.agent_task_merge_enable_pairwise_fallback),
        "agent_task_merge_min_unified_rectangularity": float(path_config.agent_task_merge_min_unified_rectangularity),
        "agent_task_merge_max_unified_group_size": int(path_config.agent_task_merge_max_unified_group_size),
        "agent_task_merge_prefer_full_components": bool(path_config.agent_task_merge_prefer_full_components),
        "agent_task_merge_full_component_max_regions": int(path_config.agent_task_merge_full_component_max_regions),
        "agent_task_merge_full_component_min_rectangularity": float(
            path_config.agent_task_merge_full_component_min_rectangularity
        ),
    }
    return (
        final_regions,
        final_patterns,
        final_paths,
        final_templates,
        final_assignment,
        diagnostics,
    )


def _recombined_source_ids(region: DecomposedRegion) -> List[str]:
    metadata = region.metadata or {}
    value = metadata.get("merge_fallback_source_ids") or metadata.get("agent_task_unified_source_ids") or ""
    return [item for item in str(value).split(",") if item]


def _monotone_scan_axis(region: DecomposedRegion, tolerance_deg: float) -> str | None:
    axes = [region.preferred_axis, "y" if region.preferred_axis == "x" else "x"]
    cells = list(getattr(region, "member_cells", []) or [])
    if cells:
        principal = _member_cell_principal_angle(cells)
        if principal is not None:
            tolerance = math.radians(max(float(tolerance_deg), 0.0))
            angles = [principal]
            if tolerance > 1e-9:
                angles.extend([principal - tolerance, principal + tolerance])
            angles = sorted(
                angles,
                key=lambda angle: _member_cell_support_span(cells, angle),
            )
            for angle in angles:
                if _region_is_angle_monotone(region, angle):
                    normalized = angle % math.pi
                    if min(abs(normalized), abs(math.pi - normalized)) <= 1e-6:
                        return "x"
                    if abs(normalized - math.pi / 2.0) <= 1e-6:
                        return "y"
                    return f"theta:{normalized:.6f}"
    for axis in axes:
        if _region_is_axis_monotone(region, axis):
            return axis
    return None


def _member_cell_principal_angle(cells: Sequence[FreeSpaceCell]) -> float | None:
    total_area = sum(max(float(cell.area), 0.0) for cell in cells)
    if total_area <= 1e-12:
        return None
    cx = sum(cell.center[0] * max(float(cell.area), 0.0) for cell in cells) / total_area
    cy = sum(cell.center[1] * max(float(cell.area), 0.0) for cell in cells) / total_area
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for cell in cells:
        area = max(float(cell.area), 0.0)
        x0, y0, x1, y1 = cell.bounds
        sxx += area * ((cell.center[0] - cx) ** 2 + (x1 - x0) ** 2 / 12.0)
        syy += area * ((cell.center[1] - cy) ** 2 + (y1 - y0) ** 2 / 12.0)
        sxy += area * (cell.center[0] - cx) * (cell.center[1] - cy)
    if sxx + syy <= 1e-12:
        return None
    return (0.5 * math.atan2(2.0 * sxy, sxx - syy)) % math.pi


def _member_cell_support_span(cells: Sequence[FreeSpaceCell], angle: float) -> float:
    vx = -math.sin(angle)
    vy = math.cos(angle)
    values = [
        x * vx + y * vy
        for cell in cells
        for x, y in cell.polygon
    ]
    return max(values, default=0.0) - min(values, default=0.0)


def _region_is_angle_monotone(region: DecomposedRegion, angle: float) -> bool:
    cells = list(getattr(region, "member_cells", []) or [])
    if not cells:
        return True
    ux = math.cos(angle)
    uy = math.sin(angle)
    vx = -uy
    vy = ux
    local_cells = [
        [(x * ux + y * uy, x * vx + y * vy) for x, y in cell.polygon]
        for cell in cells
    ]
    events = sorted({round(point[1], 9) for polygon in local_cells for point in polygon})
    samples = [
        (first + second) / 2.0
        for first, second in zip(events[:-1], events[1:])
        if second - first > 1e-9
    ]
    for coordinate in samples:
        intervals: List[Tuple[float, float]] = []
        for polygon in local_cells:
            values: List[float] = []
            for index, start in enumerate(polygon):
                end = polygon[(index + 1) % len(polygon)]
                u0, v0 = start
                u1, v1 = end
                if abs(v1 - v0) <= 1e-12:
                    if abs(coordinate - v0) <= 1e-9:
                        values.extend([u0, u1])
                    continue
                if min(v0, v1) - 1e-9 <= coordinate <= max(v0, v1) + 1e-9:
                    alpha = (coordinate - v0) / (v1 - v0)
                    if -1e-9 <= alpha <= 1.0 + 1e-9:
                        values.append(u0 + alpha * (u1 - u0))
            if len(values) >= 2:
                intervals.append((min(values), max(values)))
        intervals.sort()
        merged_count = 0
        current_end = -float("inf")
        for low, high in intervals:
            if low > current_end + 1e-6:
                merged_count += 1
                current_end = high
            else:
                current_end = max(current_end, high)
        if merged_count > 1:
            return False
    return True


def _region_is_axis_monotone(region: DecomposedRegion, axis: str) -> bool:
    cells = list(getattr(region, "member_cells", []) or [])
    if not cells:
        return True
    event_values = sorted(
        {
            value
            for cell in cells
            for value in ((cell.bounds[1], cell.bounds[3]) if axis == "x" else (cell.bounds[0], cell.bounds[2]))
        }
    )
    samples = [(first + second) / 2.0 for first, second in zip(event_values[:-1], event_values[1:]) if second - first > 1e-9]
    for coordinate in samples:
        intervals = []
        for cell in cells:
            x0, y0, x1, y1 = cell.bounds
            if axis == "x" and y0 - 1e-9 <= coordinate <= y1 + 1e-9:
                intervals.append((x0, x1))
            elif axis == "y" and x0 - 1e-9 <= coordinate <= x1 + 1e-9:
                intervals.append((y0, y1))
        intervals.sort()
        merged_count = 0
        current_end = -float("inf")
        for low, high in intervals:
            if low > current_end + 1e-6:
                merged_count += 1
                current_end = high
            else:
                current_end = max(current_end, high)
        if merged_count > 1:
            return False
    return True


def _merge_assigned_agent_task_regions(
    regions: Sequence,
    agent_regions: Dict[int, List[str]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    progress_callback: Callable[..., None] | None = None,
) -> Tuple[List, Dict[int, List[str]], Dict[str, object]]:
    diagnostics: Dict[str, object] = {
        "agent_task_merge_enabled": bool(path_config.enable_agent_task_region_merge),
        "agent_task_merge_status": "disabled",
        "agent_task_merge_region_count_before": len(regions),
        "agent_task_merge_region_count_after": len(regions),
        "agent_task_merge_candidate_count": 0,
        "agent_task_merge_accepted_count": 0,
        "agent_task_strip_merge_enabled": bool(path_config.enable_agent_task_lightweight_strip_merge),
        "agent_task_strip_candidate_count": 0,
        "agent_task_strip_accepted_count": 0,
        "agent_task_strip_budget_exhausted": False,
        "agent_task_strip_budget_reason": "",
        "agent_task_strip_elapsed_sec": 0.0,
        "agent_task_strip_rejected_by_reason": {},
        "agent_task_strip_regions": [],
        "agent_task_merge_rejected_by_reason": {},
        "agent_task_merge_rejected_candidates": [],
        "agent_task_merge_regions": [],
        "agent_task_merge_split_count": 0,
        "agent_task_merge_unstable_region_count": 0,
        "agent_task_merge_unstable_regions": [],
    }
    if (
        not path_config.enable_agent_task_region_merge
        and not path_config.enable_agent_task_lightweight_strip_merge
    ) or len(regions) < 2:
        return list(regions), {agent_id: list(ids) for agent_id, ids in agent_regions.items()}, diagnostics

    region_by_id = {region.region_id: region for region in regions}
    merged_regions: List = []
    merged_agent_regions: Dict[int, List[str]] = {}
    total_candidates = 0
    total_accepted = 0
    unified_candidate_count = 0
    unified_validation_count = 0
    unified_accepted_count = 0
    strip_candidate_count = 0
    strip_accepted_count = 0
    strip_budget_exhausted = False
    strip_budget_reason = ""
    strip_elapsed_sec = 0.0
    strip_rejected: Dict[str, int] = {}
    strip_records: List[Dict[str, object]] = []
    rejected: Dict[str, int] = {}
    rejected_records: List[Dict[str, object]] = []
    accepted_records: List[Dict[str, object]] = []
    merge_config = replace(
        path_config,
        enable_coverage_aware_merge=True,
        coverage_merge_max_area_fraction=max(
            float(path_config.coverage_merge_max_area_fraction),
            float(path_config.agent_task_merge_max_area_fraction),
        ),
        coverage_merge_min_improvement_ratio=max(float(path_config.agent_task_merge_min_improvement_ratio), 0.0),
        coverage_merge_time_budget_sec=min(
            max(float(path_config.coverage_merge_time_budget_sec), 0.0),
            max(float(path_config.agent_task_merge_time_budget_sec), 0.0),
        )
        or max(float(path_config.coverage_merge_time_budget_sec), 0.0),
    )
    for agent_id, region_ids in sorted(agent_regions.items()):
        source_regions = [region_by_id[region_id] for region_id in region_ids if region_id in region_by_id]
        local_strip_accepted_count = 0
        local_unified_accepted_count = 0
        if progress_callback is not None:
            progress_callback(
                event="agent_start",
                agent_id=agent_id,
                source_region_count=len(source_regions),
            )
        if len(source_regions) < 2:
            merged_agent_regions[agent_id] = [region.region_id for region in source_regions]
            merged_regions.extend(source_regions)
            if progress_callback is not None:
                progress_callback(
                    event="agent_done",
                    agent_id=agent_id,
                    source_region_count=len(source_regions),
                    merged_region_count=len(source_regions),
                    accepted_count=0,
                    unified_accepted_count=0,
            )
            continue
        if path_config.enable_agent_task_lightweight_strip_merge:
            source_regions, strip_diagnostics = _merge_agent_lightweight_strip_regions(
                source_regions,
                agent_id,
                config,
                merge_config,
                obstacle_field,
                progress_callback=progress_callback,
            )
            strip_candidate_count += int(strip_diagnostics.get("agent_task_strip_candidate_count", 0) or 0)
            strip_accepted_count += int(strip_diagnostics.get("agent_task_strip_accepted_count", 0) or 0)
            strip_budget_exhausted = strip_budget_exhausted or bool(
                strip_diagnostics.get("agent_task_strip_budget_exhausted", False)
            )
            if strip_diagnostics.get("agent_task_strip_budget_reason"):
                strip_budget_reason = str(strip_diagnostics.get("agent_task_strip_budget_reason", ""))
            strip_elapsed_sec += float(strip_diagnostics.get("agent_task_strip_elapsed_sec", 0.0) or 0.0)
            total_candidates += int(strip_diagnostics.get("agent_task_strip_candidate_count", 0) or 0)
            total_accepted += int(strip_diagnostics.get("agent_task_strip_accepted_count", 0) or 0)
            for reason, count in dict(strip_diagnostics.get("agent_task_strip_rejected_by_reason", {}) or {}).items():
                strip_rejected[str(reason)] = strip_rejected.get(str(reason), 0) + int(count)
                rejected[str(reason)] = rejected.get(str(reason), 0) + int(count)
            for record in list(strip_diagnostics.get("agent_task_strip_regions", []) or []):
                strip_records.append(record)
                accepted_records.append(record)
            local_strip_accepted_count = int(strip_diagnostics.get("agent_task_strip_accepted_count", 0) or 0)
            if len(source_regions) < 2:
                merged_agent_regions[agent_id] = [region.region_id for region in source_regions]
                merged_regions.extend(source_regions)
                if progress_callback is not None:
                    progress_callback(
                        event="agent_done",
                        agent_id=agent_id,
                        source_region_count=len(source_regions),
                        merged_region_count=len(source_regions),
                        accepted_count=local_strip_accepted_count,
                        strip_accepted_count=local_strip_accepted_count,
                        unified_accepted_count=0,
                    )
                continue
        if path_config.enable_agent_task_region_merge and path_config.agent_task_merge_enable_unified_group_merge:
            source_regions, unified_diagnostics = _merge_agent_unified_region_groups(
                source_regions,
                agent_id,
                config,
                merge_config,
                obstacle_field,
                progress_callback=progress_callback,
            )
            unified_candidate_count += int(unified_diagnostics.get("agent_task_unified_candidate_count", 0) or 0)
            unified_validation_count += int(unified_diagnostics.get("agent_task_unified_validation_count", 0) or 0)
            unified_accepted_count += int(unified_diagnostics.get("agent_task_unified_accepted_count", 0) or 0)
            total_candidates += int(unified_diagnostics.get("agent_task_unified_candidate_count", 0) or 0)
            total_accepted += int(unified_diagnostics.get("agent_task_unified_accepted_count", 0) or 0)
            for reason, count in dict(unified_diagnostics.get("agent_task_unified_rejected_by_reason", {}) or {}).items():
                rejected[str(reason)] = rejected.get(str(reason), 0) + int(count)
            for record in list(unified_diagnostics.get("agent_task_unified_rejected_candidates", []) or []):
                rejected_records.append(record)
            for record in list(unified_diagnostics.get("agent_task_unified_regions", []) or []):
                accepted_records.append(record)
            local_unified_accepted_count = int(unified_diagnostics.get("agent_task_unified_accepted_count", 0) or 0)
            if len(source_regions) < 2:
                merged_agent_regions[agent_id] = [region.region_id for region in source_regions]
                merged_regions.extend(source_regions)
                if progress_callback is not None:
                    progress_callback(
                        event="agent_done",
                        agent_id=agent_id,
                        source_region_count=len(source_regions),
                        merged_region_count=len(source_regions),
                        accepted_count=local_strip_accepted_count + local_unified_accepted_count,
                        strip_accepted_count=local_strip_accepted_count,
                        unified_accepted_count=local_unified_accepted_count,
                    )
                continue
        accepted_count = 0
        if path_config.enable_agent_task_region_merge and path_config.agent_task_merge_enable_pairwise_fallback:
            local_diagnostics: Dict[str, object] = {}
            agent_merged = _coverage_aware_merge_regions(
                source_regions,
                config,
                merge_config,
                obstacle_field=obstacle_field,
                diagnostics=local_diagnostics,
                progress_callback=(
                    (lambda **extra: progress_callback(agent_id=agent_id, event=f"pairwise_{extra.get('event', 'progress')}", **{k: v for k, v in extra.items() if k != 'event'}))
                    if progress_callback is not None
                    else None
                ),
                region_id_prefix=f"agent{agent_id}_task_merge_region",
            )
            total_candidates += int(local_diagnostics.get("coverage_merge_candidate_count", 0) or 0)
            accepted_count = int(local_diagnostics.get("coverage_merge_accepted_count", 0) or 0)
            total_accepted += accepted_count
            for reason, count in dict(local_diagnostics.get("coverage_merge_rejected_by_reason", {}) or {}).items():
                rejected[str(reason)] = rejected.get(str(reason), 0) + int(count)
            for record in list(local_diagnostics.get("coverage_merge_rejected_candidates", []) or []):
                enriched = dict(record)
                enriched["agent_id"] = agent_id
                rejected_records.append(enriched)
            for record in list(local_diagnostics.get("coverage_merge_regions", []) or []):
                enriched = dict(record)
                enriched["agent_id"] = agent_id
                accepted_records.append(enriched)
        else:
            agent_merged = source_regions
            if path_config.enable_agent_task_region_merge:
                _increment_object_diagnostic(rejected, "pairwise_fallback_disabled")
            if progress_callback is not None:
                progress_callback(
                    event="pairwise_skipped",
                    agent_id=agent_id,
                    reason="pairwise_fallback_disabled"
                    if path_config.enable_agent_task_region_merge
                    else "full_agent_task_merge_disabled",
                    source_region_count=len(source_regions),
                )
        merged_agent_regions[agent_id] = [region.region_id for region in agent_merged]
        merged_regions.extend(agent_merged)
        if progress_callback is not None:
            progress_callback(
                event="agent_done",
                agent_id=agent_id,
                source_region_count=len(source_regions),
                merged_region_count=len(agent_merged),
                accepted_count=local_strip_accepted_count + local_unified_accepted_count + accepted_count,
                strip_accepted_count=local_strip_accepted_count,
                unified_accepted_count=local_unified_accepted_count,
            )

    full_component_accepted_count = sum(1 for record in accepted_records if record.get("full_component_merge"))
    if total_accepted <= 0:
        diagnostics.update(
            {
                "agent_task_merge_status": "no_improvement",
                "agent_task_merge_candidate_count": total_candidates,
                "agent_task_strip_merge_enabled": bool(path_config.enable_agent_task_lightweight_strip_merge),
                "agent_task_strip_candidate_count": strip_candidate_count,
                "agent_task_strip_accepted_count": strip_accepted_count,
                "agent_task_strip_budget_exhausted": strip_budget_exhausted,
                "agent_task_strip_budget_reason": strip_budget_reason,
                "agent_task_strip_elapsed_sec": strip_elapsed_sec,
                "agent_task_strip_rejected_by_reason": strip_rejected,
                "agent_task_strip_regions": strip_records,
                "agent_task_unified_merge_enabled": bool(path_config.agent_task_merge_enable_unified_group_merge),
                "agent_task_merge_pairwise_fallback_enabled": bool(path_config.agent_task_merge_enable_pairwise_fallback),
                "agent_task_merge_min_unified_rectangularity": float(path_config.agent_task_merge_min_unified_rectangularity),
                "agent_task_merge_max_unified_group_size": int(path_config.agent_task_merge_max_unified_group_size),
                "agent_task_merge_prefer_full_components": bool(path_config.agent_task_merge_prefer_full_components),
                "agent_task_merge_full_component_max_regions": int(path_config.agent_task_merge_full_component_max_regions),
                "agent_task_merge_full_component_min_rectangularity": float(
                    path_config.agent_task_merge_full_component_min_rectangularity
                ),
                "agent_task_full_component_accepted_count": full_component_accepted_count,
                "agent_task_unified_candidate_count": unified_candidate_count,
                "agent_task_unified_validation_count": unified_validation_count,
                "agent_task_unified_accepted_count": unified_accepted_count,
                "agent_task_merge_rejected_by_reason": rejected,
                "agent_task_merge_rejected_candidates": rejected_records,
                "agent_task_merge_regions": accepted_records,
                "agent_task_merge_split_count": 0,
            }
        )
        return list(regions), {agent_id: list(ids) for agent_id, ids in agent_regions.items()}, diagnostics

    _populate_region_neighbors(merged_regions)
    diagnostics.update(
        {
            "agent_task_merge_status": "candidate_ready",
            "agent_task_merge_region_count_after": len(merged_regions),
            "agent_task_merge_candidate_count": total_candidates,
            "agent_task_merge_accepted_count": total_accepted,
            "agent_task_strip_merge_enabled": bool(path_config.enable_agent_task_lightweight_strip_merge),
            "agent_task_strip_candidate_count": strip_candidate_count,
            "agent_task_strip_accepted_count": strip_accepted_count,
            "agent_task_strip_budget_exhausted": strip_budget_exhausted,
            "agent_task_strip_budget_reason": strip_budget_reason,
            "agent_task_strip_elapsed_sec": strip_elapsed_sec,
            "agent_task_strip_rejected_by_reason": strip_rejected,
            "agent_task_strip_regions": strip_records,
            "agent_task_unified_merge_enabled": bool(path_config.agent_task_merge_enable_unified_group_merge),
            "agent_task_merge_pairwise_fallback_enabled": bool(path_config.agent_task_merge_enable_pairwise_fallback),
            "agent_task_merge_min_unified_rectangularity": float(path_config.agent_task_merge_min_unified_rectangularity),
            "agent_task_merge_max_unified_group_size": int(path_config.agent_task_merge_max_unified_group_size),
            "agent_task_merge_prefer_full_components": bool(path_config.agent_task_merge_prefer_full_components),
            "agent_task_merge_full_component_max_regions": int(path_config.agent_task_merge_full_component_max_regions),
            "agent_task_merge_full_component_min_rectangularity": float(
                path_config.agent_task_merge_full_component_min_rectangularity
            ),
            "agent_task_full_component_accepted_count": full_component_accepted_count,
            "agent_task_unified_candidate_count": unified_candidate_count,
            "agent_task_unified_validation_count": unified_validation_count,
            "agent_task_unified_accepted_count": unified_accepted_count,
            "agent_task_merge_rejected_by_reason": rejected,
            "agent_task_merge_regions": accepted_records,
            "agent_task_merge_split_count": 0,
        }
    )
    return merged_regions, merged_agent_regions, diagnostics


def _merge_agent_lightweight_strip_regions(
    source_regions: Sequence,
    agent_id: int,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    progress_callback: Callable[..., None] | None = None,
) -> Tuple[List, Dict[str, object]]:
    strip_started = time.perf_counter()
    current = list(source_regions)
    rejected: Dict[str, int] = {}
    accepted_records: List[Dict[str, object]] = []
    candidate_count = 0
    preview_cache: Dict[Tuple[object, ...], Dict[str, object]] = {}
    max_groups = max(int(path_config.agent_task_strip_merge_max_groups_per_agent), 0)
    max_candidate_evaluations = max(int(path_config.agent_task_strip_merge_max_candidate_evaluations), 0)
    time_budget_sec = max(float(path_config.agent_task_strip_merge_time_budget_sec), 0.0)
    budget_exhausted = False
    budget_reason = ""
    if max_groups <= 0 or len(current) < 2:
        return current, {
            "agent_task_strip_candidate_count": 0,
            "agent_task_strip_accepted_count": 0,
            "agent_task_strip_budget_exhausted": False,
            "agent_task_strip_budget_reason": "",
            "agent_task_strip_elapsed_sec": time.perf_counter() - strip_started,
            "agent_task_strip_rejected_by_reason": {"disabled": 1} if max_groups <= 0 else {},
            "agent_task_strip_regions": [],
        }

    def current_budget_reason() -> str:
        if max_candidate_evaluations > 0 and candidate_count >= max_candidate_evaluations:
            return "strip_candidate_budget_exhausted"
        if time_budget_sec > 0.0 and time.perf_counter() - strip_started >= time_budget_sec:
            return "strip_time_budget_exhausted"
        return ""

    groups = _agent_task_strip_candidate_groups(current, config, path_config, obstacle_field)
    groups.sort(key=lambda group: _agent_task_strip_group_sort_key(group, path_config))
    accepted_source_ids: set[str] = set()
    min_rectangularity = max(0.0, min(1.0, float(path_config.agent_task_strip_merge_min_rectangularity)))
    min_length_gain = max(float(path_config.agent_task_strip_merge_min_length_gain_factor), 1.0)
    if progress_callback is not None:
        progress_callback(
            event="strip_agent_start",
            agent_id=agent_id,
            candidate_group_count=len(groups),
            max_groups=max_groups,
            min_rectangularity=round(min_rectangularity, 6),
            min_length_gain=round(min_length_gain, 6),
    )
    serial = 0
    for group_index, group in enumerate(groups, start=1):
        reason = current_budget_reason()
        if reason:
            budget_exhausted = True
            budget_reason = reason
            _increment_object_diagnostic(rejected, reason)
            if progress_callback is not None:
                progress_callback(
                    event="strip_budget_stop",
                    agent_id=agent_id,
                    candidate_count=candidate_count,
                    accepted_count=len(accepted_records),
                    budget_reason=budget_reason,
                    elapsed_sec=round(time.perf_counter() - strip_started, 3),
                )
            break
        if len(accepted_records) >= max_groups:
            _increment_object_diagnostic(rejected, "strip_group_accept_limit")
            break
        source_ids = [region.region_id for region in group]
        if any(region_id in accepted_source_ids for region_id in source_ids):
            continue
        candidate_count += 1
        full_component_candidate = _agent_task_group_is_full_component_candidate(group, path_config)
        if progress_callback is not None:
            progress_callback(
                event="strip_candidate_start",
                agent_id=agent_id,
                candidate_index=candidate_count,
                candidate_group_index=group_index,
                source_region_count=len(source_ids),
                source_region_ids=source_ids,
                full_component_candidate=bool(full_component_candidate),
                elapsed_sec=round(time.perf_counter() - strip_started, 3),
            )
        rectangularity = _agent_task_group_rectangularity(group)
        group_min_rectangularity = _agent_task_group_min_rectangularity(group, path_config, min_rectangularity)
        if rectangularity + 1e-9 < group_min_rectangularity:
            _increment_object_diagnostic(rejected, "strip_low_rectangularity")
            continue
        candidate, reason = _coverage_merge_candidate_from_group(
            serial,
            group,
            config,
            path_config,
            obstacle_field,
            region_id_prefix=f"agent{agent_id}_strip_task_region",
        )
        if candidate is None:
            _increment_object_diagnostic(rejected, f"strip_{reason}")
            continue
        shape_class = str(candidate.metadata.get("shape_class", ""))
        if bool(path_config.agent_task_strip_merge_use_geometric_preview):
            before_preview = _agent_task_strip_geometric_before_preview(group, config, path_config)
            after_preview = _agent_task_strip_geometric_candidate_preview(candidate, group, config, path_config)
        else:
            before_preview = _coverage_merge_before_preview(group, config, path_config, obstacle_field, preview_cache)
            after_preview = _coverage_merge_pattern_preview(
                candidate,
                config,
                path_config,
                obstacle_field,
                preview_cache,
                validate_internal=False,
            )
        preview_mode = "geometric" if bool(path_config.agent_task_strip_merge_use_geometric_preview) else "pattern"
        validate_fragmented_strip = (
            len(source_ids) >= 3
            and rectangularity + 1e-9
            < _agent_task_strip_direct_priority_rectangularity(path_config)
        )
        if bool(path_config.agent_task_strip_merge_use_geometric_preview) and (
            len(source_ids) > 4 or validate_fragmented_strip
        ):
            before_preview = _coverage_merge_before_preview(group, config, path_config, obstacle_field, preview_cache)
            after_preview = _coverage_merge_pattern_preview(
                candidate,
                config,
                path_config,
                obstacle_field,
                preview_cache,
                validate_internal=True,
            )
            preview_mode = (
                "validated_pattern_fragmented_strip"
                if validate_fragmented_strip
                else "validated_pattern_large_strip"
            )
            if not after_preview.get("feasible", False):
                _increment_object_diagnostic(
                    rejected,
                    f"strip_large_{after_preview.get('reason', 'no_feasible_pattern')}",
                )
                continue
            large_strip_open_chain_limit = max(8, min(max(int(path_config.max_open_chains_per_region), 1), 16))
            large_strip_open_chain_count = int(after_preview.get("open_chain_count", 0) or 0)
            large_strip_pass_count = int(after_preview.get("pass_count", 0) or 0)
            if large_strip_open_chain_count > large_strip_open_chain_limit:
                _increment_object_diagnostic(rejected, "strip_large_open_chain_count")
                continue
            if large_strip_pass_count > max(16, len(source_ids) * 3):
                _increment_object_diagnostic(rejected, "strip_large_pass_count")
                continue
        if not after_preview.get("feasible", False):
            _increment_object_diagnostic(rejected, f"strip_{after_preview.get('reason', 'no_feasible_pattern')}")
            continue
        coverage_fraction = float(after_preview.get("coverage_fraction", 0.0) or 0.0)
        if coverage_fraction + 1e-9 < max(0.0, min(1.0, float(path_config.coverage_merge_min_coverage_fraction))):
            _increment_object_diagnostic(rejected, "strip_low_coverage_fraction")
            continue
        before_score = float(before_preview.get("score", 0.0) or 0.0)
        raw_after_score = float(after_preview.get("score", 0.0) or 0.0)
        boustrophedon_reward = _agent_task_boustrophedon_merge_reward(before_preview, after_preview, path_config)
        after_score = raw_after_score - boustrophedon_reward
        improvement = before_score - after_score
        improvement_ratio = improvement / max(abs(before_score), 1e-9)
        source_pass_count = int(before_preview.get("pass_count", 0) or 0)
        candidate_pass_count = int(after_preview.get("pass_count", 0) or 0)
        source_max_pass = float(before_preview.get("max_pass_length", 0.0) or 0.0)
        candidate_max_pass = float(after_preview.get("max_pass_length", 0.0) or 0.0)
        length_gain = candidate_max_pass / max(source_max_pass, 1e-9)
        pass_count_ok = candidate_pass_count <= max(source_pass_count, 1)
        objective_ok = improvement_ratio + 1e-9 >= max(float(path_config.agent_task_merge_min_improvement_ratio), 0.0)
        long_sweep_ok = length_gain + 1e-9 >= min_length_gain and pass_count_ok
        if not (objective_ok or long_sweep_ok):
            _increment_object_diagnostic(rejected, "strip_low_long_sweep_gain")
            continue
        candidate.metadata.update(
            {
                "agent_task_strip_merge": "true",
                "agent_task_strip_agent_id": str(agent_id),
                "agent_task_strip_source_count": str(len(source_ids)),
                "agent_task_strip_source_ids": ",".join(source_ids),
                "agent_task_strip_rectangularity": f"{rectangularity:.6f}",
                "agent_task_full_component_merge": str(bool(full_component_candidate)).lower(),
                "merge_accept_reason": (
                    "agent_task_strip_full_component_boustrophedon_gain"
                    if full_component_candidate and shape_class == "rectangle"
                    else (
                        "agent_task_strip_long_boustrophedon_gain"
                        if shape_class == "rectangle"
                        else "agent_task_strip_composite_boustrophedon_gain"
                    )
                ),
                "merge_objective_before": f"{before_score:.6f}",
                "merge_objective_after": f"{after_score:.6f}",
                "merge_raw_objective_after": f"{raw_after_score:.6f}",
                "merge_boustrophedon_reward": f"{boustrophedon_reward:.6f}",
                "merge_objective_delta": f"{improvement:.6f}",
                "merge_objective_delta_ratio": f"{improvement_ratio:.6f}",
                "merge_source_pass_count": str(source_pass_count),
                "merge_candidate_pass_count": str(candidate_pass_count),
                "merge_source_max_pass_length": f"{source_max_pass:.6f}",
                "merge_candidate_max_pass_length": f"{candidate_max_pass:.6f}",
                "merge_long_pass_gain_ratio": f"{length_gain:.6f}",
                "merge_preview_mode": preview_mode,
                "merge_open_chain_count": str(int(after_preview.get("open_chain_count", 0) or 0)),
                "merge_best_pattern_id": str(after_preview.get("pattern_id", "")),
                "merge_best_pattern_scan_axis": str(after_preview.get("scan_axis", "")),
                "merge_best_pattern_coverage_fraction": f"{coverage_fraction:.6f}",
            }
        )
        accepted_records.append(
            {
                "agent_id": agent_id,
                "region_id": candidate.region_id,
                "source_region_ids": source_ids,
                "source_region_count": len(source_ids),
                "rectangularity": round(rectangularity, 6),
                "min_rectangularity": round(group_min_rectangularity, 6),
                "shape_class": shape_class,
                "scan_support_mode": candidate.metadata.get("scan_support_mode", ""),
                "objective_before": round(before_score, 6),
                "objective_after": round(after_score, 6),
                "raw_objective_after": round(raw_after_score, 6),
                "boustrophedon_reward": round(boustrophedon_reward, 6),
                "objective_delta": round(improvement, 6),
                "objective_delta_ratio": round(improvement_ratio, 6),
                "coverage_fraction": round(coverage_fraction, 6),
                "source_pass_count": source_pass_count,
                "candidate_pass_count": candidate_pass_count,
                "source_max_pass_length": round(source_max_pass, 6),
                "candidate_max_pass_length": round(candidate_max_pass, 6),
                "long_pass_gain_ratio": round(length_gain, 6),
                "preview_mode": preview_mode,
                "open_chain_count": int(after_preview.get("open_chain_count", 0) or 0),
                "best_pattern_scan_axis": after_preview.get("scan_axis", ""),
                "strip_merge": True,
                "full_component_merge": bool(full_component_candidate),
            }
        )
        accepted_source_ids.update(source_ids)
        current = [region for region in current if region.region_id not in accepted_source_ids]
        current.append(candidate)
        serial += 1
        if progress_callback is not None:
            progress_callback(
                event="strip_candidate_accepted",
                agent_id=agent_id,
                candidate_index=candidate_count,
                candidate_group_index=group_index,
                candidate_region_id=candidate.region_id,
                source_region_count=len(source_ids),
                source_region_ids=source_ids,
                candidate_pass_count=candidate_pass_count,
                source_pass_count=source_pass_count,
                long_pass_gain_ratio=round(length_gain, 6),
                objective_delta=round(improvement, 6),
            )
    current.sort(key=lambda region: (region.center[0], region.center[1], region.region_id))
    _populate_region_neighbors(current)
    if progress_callback is not None:
        progress_callback(
            event="strip_agent_done",
            agent_id=agent_id,
            candidate_count=candidate_count,
            accepted_count=len(accepted_records),
            budget_exhausted=budget_exhausted,
            budget_reason=budget_reason,
            elapsed_sec=round(time.perf_counter() - strip_started, 3),
            rejected_by_reason=rejected,
        )
    return current, {
        "agent_task_strip_candidate_count": candidate_count,
        "agent_task_strip_accepted_count": len(accepted_records),
        "agent_task_strip_budget_exhausted": budget_exhausted,
        "agent_task_strip_budget_reason": budget_reason,
        "agent_task_strip_elapsed_sec": time.perf_counter() - strip_started,
        "agent_task_strip_rejected_by_reason": rejected,
        "agent_task_strip_regions": accepted_records,
    }


def _agent_task_strip_geometric_before_preview(
    group: Sequence,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> Dict[str, object]:
    previews = [_agent_task_strip_geometric_region_preview(region, config, path_config) for region in group]
    connector_proxy = (
        max(config.fleet.min_turn_radius, 0.0) * math.pi
        + max(config.footprint.length_lf, 0.0)
        + 2.0 * max(config.footprint.width_wf, 0.0)
    ) * max(path_config.transition_length_weight, 0.0)
    boundary_turn_proxy = _coverage_merge_boundary_turn_proxy(group, path_config)
    score = sum(float(preview.get("score", 0.0) or 0.0) for preview in previews)
    score += max(len(group) - 1, 0) * connector_proxy
    score += boundary_turn_proxy
    return {
        "feasible": all(bool(preview.get("feasible", False)) for preview in previews),
        "score": score,
        "feasible_count": sum(1 for preview in previews if preview.get("feasible", False)),
        "connector_proxy": connector_proxy,
        "boundary_turn_proxy": boundary_turn_proxy,
        "pass_count": sum(int(preview.get("pass_count", 0) or 0) for preview in previews),
        "coverage_length": sum(float(preview.get("coverage_length", 0.0) or 0.0) for preview in previews),
        "max_pass_length": max(
            (float(preview.get("max_pass_length", 0.0) or 0.0) for preview in previews),
            default=0.0,
        ),
    }


def _agent_task_strip_geometric_candidate_preview(
    candidate,
    source_group: Sequence,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> Dict[str, object]:
    preview = _agent_task_strip_geometric_region_preview(candidate, config, path_config)
    cells = _coverage_merge_member_cells(source_group)
    source_area = sum(max(float(cell.area), 0.0) for cell in cells)
    bbox_area = _cell_group_bounds_area(candidate.bounds)
    coverage_fraction = min(
        1.0,
        max(
            0.0,
            max(
                source_area / max(bbox_area, 1e-9),
                float(preview.get("coverage_length", 0.0) or 0.0)
                * max(config.footprint.width_wf, 0.0)
                / max(bbox_area, 1e-9),
            ),
        ),
    )
    preview.update(
        {
            "pattern_id": f"{candidate.region_id}_strip_geometric_preview",
            "scan_axis": str(getattr(candidate, "preferred_axis", "x")),
            "coverage_fraction": coverage_fraction,
            "reason": "",
        }
    )
    return preview


def _agent_task_strip_geometric_region_preview(
    region,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> Dict[str, object]:
    x0, y0, x1, y1 = region.bounds
    width = max(x1 - x0, 0.0)
    height = max(y1 - y0, 0.0)
    if width <= 1e-9 or height <= 1e-9:
        return {"feasible": False, "score": float("inf"), "reason": "degenerate_bounds"}
    axis = str(getattr(region, "preferred_axis", "") or region.metadata.get("dominant_scan_axis", "x"))
    if axis not in {"x", "y"}:
        axis = "x" if width >= height else "y"
    along_span = width if axis == "x" else height
    cross_span = height if axis == "x" else width
    pass_count = max(1, int(math.ceil(cross_span / max(config.footprint.width_wf, 1e-6))))
    coverage_length = max(along_span, 0.0) * pass_count
    turn_count = max(pass_count - 1, 0)
    turn_length = turn_count * (
        math.pi * max(config.fleet.min_turn_radius, 0.0)
        + max(config.footprint.width_wf, 0.0)
    )
    turn_angle = turn_count * math.pi
    estimated_time = (
        coverage_length / max(config.fleet.cover_speed, 1e-6)
        + turn_length / max(min(config.fleet.turn_speed_max, config.fleet.cruise_speed), 1e-6)
    )
    score = (
        estimated_time
        + 0.8 * turn_length
        + max(path_config.pattern_turn_penalty_weight, 0.0) * turn_angle
        + 0.15 * pass_count
        - 0.1 * coverage_length
    )
    return {
        "feasible": True,
        "score": score,
        "scan_axis": axis,
        "coverage_fraction": 1.0,
        "coverage_length": coverage_length,
        "max_pass_length": along_span,
        "pass_count": pass_count,
        "turn_length": turn_length,
        "turn_angle": turn_angle,
        "estimated_time": estimated_time,
    }


def _agent_task_strip_candidate_groups(
    regions: Sequence,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> List[List]:
    region_list = list(regions)
    groups: List[List] = []
    seen: set[Tuple[str, ...]] = set()
    max_group_size = max(int(path_config.agent_task_merge_max_unified_group_size), 0)
    if max_group_size <= 0:
        max_group_size = 6
    full_component_limit = _agent_task_full_component_max_regions(path_config)

    def add_group(group: Sequence) -> None:
        key = tuple(sorted(region.region_id for region in group))
        if len(key) < 2 or key in seen:
            return
        seen.add(key)
        groups.append(list(group))

    for component in _agent_task_merge_connected_components(region_list, config, path_config, obstacle_field):
        for axis in ("x", "y"):
            for axis_group in _agent_task_axis_compatible_components(component, axis, config, path_config, obstacle_field):
                ordered = sorted(axis_group, key=lambda region: (region.center[0], region.center[1], region.region_id) if axis == "x" else (region.center[1], region.center[0], region.region_id))
                if len(ordered) <= max_group_size:
                    add_group(ordered)
                    if (
                        len(ordered) >= 3
                        and _agent_task_group_rectangularity(ordered) + 1e-9
                        < _agent_task_strip_direct_priority_rectangularity(path_config)
                    ):
                        # A shallow-L/S component can look attractive as one
                        # long strip in the geometric preview, but its real
                        # polygon sweep often fragments into many open chains.
                        # Keep contiguous sub-strips available so the agent can
                        # still combine executable portions of the component.
                        for window_size in range(len(ordered) - 1, 1, -1):
                            for start in range(0, len(ordered) - window_size + 1):
                                add_group(ordered[start : start + window_size])
                    continue
                if full_component_limit > 0 and len(ordered) <= full_component_limit:
                    add_group(ordered)
                for start in range(0, len(ordered) - max_group_size + 1):
                    add_group(ordered[start : start + max_group_size])
                add_group(ordered[:max_group_size])
                add_group(ordered[-max_group_size:])
    return groups


def _agent_task_strip_group_sort_key(
    group: Sequence,
    path_config: PathPlanningConfig | None = None,
) -> Tuple[int, float, float, float, int, str]:
    cells = _coverage_merge_member_cells(group)
    bounds = _cell_group_bounds_for_merge(cells) if cells else (0.0, 0.0, 0.0, 0.0)
    width = max(bounds[2] - bounds[0], 0.0)
    height = max(bounds[3] - bounds[1], 0.0)
    rectangularity = _agent_task_group_rectangularity(group)
    long_span = max(width, height)
    area = _cell_group_bounds_area(bounds)
    direct_priority_rectangularity = (
        _agent_task_strip_direct_priority_rectangularity(path_config)
        if path_config is not None
        else 0.95
    )
    # Two-region strips are the smallest useful merge and remain cheap to
    # validate. Larger fragmented groups follow regular rectangular groups and
    # pairs, even when they fit inside the nominal group-size limit.
    fragmented_full_component = int(
        len(group) >= 3 and rectangularity < direct_priority_rectangularity
    )
    return (
        fragmented_full_component,
        -long_span,
        -area,
        -rectangularity,
        -len(group),
        ",".join(sorted(region.region_id for region in group)),
    )


def _agent_task_strip_direct_priority_rectangularity(
    path_config: PathPlanningConfig,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            max(
                float(path_config.agent_task_strip_merge_min_rectangularity),
                float(path_config.agent_task_strip_full_component_direct_priority_rectangularity),
            ),
        ),
    )


def _agent_task_unified_group_sort_key(group: Sequence) -> Tuple[float, float, float, int, str]:
    cells = _coverage_merge_member_cells(group)
    bounds = _cell_group_bounds_for_merge(cells) if cells else (0.0, 0.0, 0.0, 0.0)
    width = max(bounds[2] - bounds[0], 0.0)
    height = max(bounds[3] - bounds[1], 0.0)
    long_span = max(width, height)
    area = _cell_group_bounds_area(bounds)
    rectangularity = _agent_task_group_rectangularity(group)
    return (-long_span, -area, -rectangularity, -len(group), min(region.region_id for region in group))


def _agent_task_full_component_max_regions(path_config: PathPlanningConfig) -> int:
    if not bool(getattr(path_config, "agent_task_merge_prefer_full_components", True)):
        return 0
    return max(int(getattr(path_config, "agent_task_merge_full_component_max_regions", 0)), 0)


def _agent_task_group_is_full_component_candidate(
    group: Sequence,
    path_config: PathPlanningConfig,
) -> bool:
    full_component_limit = _agent_task_full_component_max_regions(path_config)
    return full_component_limit > 0 and 2 <= len(group) <= full_component_limit


def _agent_task_group_min_rectangularity(
    group: Sequence,
    path_config: PathPlanningConfig,
    base_min_rectangularity: float,
) -> float:
    if not _agent_task_group_is_full_component_candidate(group, path_config):
        return base_min_rectangularity
    full_min = max(
        0.0,
        min(
            1.0,
            float(getattr(path_config, "agent_task_merge_full_component_min_rectangularity", base_min_rectangularity)),
        ),
    )
    return min(base_min_rectangularity, full_min)


def _merge_agent_unified_region_groups(
    source_regions: Sequence,
    agent_id: int,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    progress_callback: Callable[..., None] | None = None,
) -> Tuple[List, Dict[str, object]]:
    current = list(source_regions)
    rejected: Dict[str, int] = {}
    rejected_records: List[Dict[str, object]] = []
    accepted_records: List[Dict[str, object]] = []
    candidate_count = 0
    validation_count = 0
    objective_delta = 0.0
    preview_cache: Dict[Tuple[object, ...], Dict[str, object]] = {}
    min_group_size = max(2, int(path_config.agent_task_merge_min_unified_group_size))
    candidate_groups = _agent_task_unified_candidate_groups(current, config, path_config, obstacle_field)
    candidate_groups.sort(key=_agent_task_unified_group_sort_key)
    max_candidates = max(int(path_config.agent_task_merge_max_unified_candidates_per_agent), 0)
    max_group_size = max(int(path_config.agent_task_merge_max_unified_group_size), 0)
    min_rectangularity = max(0.0, min(1.0, float(path_config.agent_task_merge_min_unified_rectangularity)))
    time_budget = max(float(path_config.agent_task_merge_time_budget_sec), 0.0)
    started = time.perf_counter()
    accepted_source_ids: set[str] = set()
    serial = 0
    if progress_callback is not None:
        progress_callback(
            event="unified_agent_start",
            agent_id=agent_id,
            candidate_group_count=len(candidate_groups),
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            full_component_max_regions=_agent_task_full_component_max_regions(path_config),
            max_candidates=max_candidates,
            min_rectangularity=round(min_rectangularity, 6),
            full_component_min_rectangularity=round(
                float(getattr(path_config, "agent_task_merge_full_component_min_rectangularity", min_rectangularity)),
                6,
            ),
        )
    budget_exhausted = False
    budget_reason = ""
    for group_index, group in enumerate(candidate_groups, start=1):
        if time_budget > 0.0 and time.perf_counter() - started >= time_budget:
            budget_exhausted = True
            budget_reason = "time_budget_exhausted"
            _increment_object_diagnostic(rejected, budget_reason)
            break
        if max_candidates > 0 and candidate_count >= max_candidates:
            budget_exhausted = True
            budget_reason = "candidate_limit_exhausted"
            _increment_object_diagnostic(rejected, budget_reason)
            break
        if len(group) < min_group_size:
            continue
        full_component_candidate = _agent_task_group_is_full_component_candidate(group, path_config)
        effective_max_group_size = max_group_size
        if full_component_candidate:
            effective_max_group_size = max(effective_max_group_size, len(group))
        if effective_max_group_size > 0 and len(group) > effective_max_group_size:
            _increment_object_diagnostic(rejected, "large_unified_group_size")
            if len(rejected_records) < 64:
                rejected_records.append(
                    {
                        "source_region_ids": [region.region_id for region in group],
                        "reason": "large_unified_group_size",
                        "source_region_count": len(group),
                        "max_group_size": effective_max_group_size,
                        "full_component_candidate": bool(full_component_candidate),
                    }
                )
            if progress_callback is not None:
                progress_callback(
                    event="unified_candidate_skipped",
                    agent_id=agent_id,
                    candidate_group_index=group_index,
                    source_region_count=len(group),
                    source_region_ids=[region.region_id for region in group],
                    reason="large_unified_group_size",
                    max_group_size=effective_max_group_size,
                    full_component_candidate=bool(full_component_candidate),
                )
            continue
        if any(region.region_id in accepted_source_ids for region in group):
            continue
        rectangularity = _agent_task_group_rectangularity(group)
        group_min_rectangularity = _agent_task_group_min_rectangularity(group, path_config, min_rectangularity)
        if rectangularity + 1e-9 < group_min_rectangularity:
            _increment_object_diagnostic(rejected, "low_unified_rectangularity")
            if len(rejected_records) < 64:
                cells = _coverage_merge_member_cells(group)
                bounds = _cell_group_bounds_for_merge(cells) if cells else (0.0, 0.0, 0.0, 0.0)
                source_area = sum(max(float(cell.area), 0.0) for cell in cells)
                rejected_records.append(
                    {
                        "source_region_ids": [region.region_id for region in group],
                        "reason": "low_unified_rectangularity",
                        "source_region_count": len(group),
                        "rectangularity": round(rectangularity, 6),
                        "min_rectangularity": round(group_min_rectangularity, 6),
                        "full_component_candidate": bool(full_component_candidate),
                        "source_area": round(source_area, 6),
                        "bounds_area": round(_cell_group_bounds_area(bounds), 6),
                    }
                )
            if progress_callback is not None:
                progress_callback(
                    event="unified_candidate_skipped",
                    agent_id=agent_id,
                    candidate_group_index=group_index,
                    source_region_count=len(group),
                    source_region_ids=[region.region_id for region in group],
                    reason="low_unified_rectangularity",
                    rectangularity=round(rectangularity, 6),
                    min_rectangularity=round(group_min_rectangularity, 6),
                    full_component_candidate=bool(full_component_candidate),
                )
            continue
        candidate_count += 1
        if progress_callback is not None:
            progress_callback(
                event="unified_candidate_start",
                agent_id=agent_id,
                candidate_index=candidate_count,
                candidate_group_index=group_index,
                source_region_count=len(group),
                source_region_ids=[region.region_id for region in group],
                rectangularity=round(rectangularity, 6),
                full_component_candidate=bool(full_component_candidate),
            )
        candidate, reason = _coverage_merge_candidate_from_group(
            serial,
            group,
            config,
            path_config,
            obstacle_field,
            region_id_prefix=f"agent{agent_id}_unified_task_region",
        )
        if candidate is None:
            _increment_object_diagnostic(rejected, reason)
            _coverage_merge_record_rejected_candidate(
                rejected_records,
                reason,
                group,
                path_config,
            )
            continue
        before_preview = _coverage_merge_before_preview(group, config, path_config, obstacle_field, preview_cache)
        after_preview = _coverage_merge_pattern_preview(
            candidate,
            config,
            path_config,
            obstacle_field,
            preview_cache,
            validate_internal=False,
        )
        if not after_preview.get("feasible", False):
            reason = str(after_preview.get("reason", "no_feasible_pattern"))
            _increment_object_diagnostic(rejected, reason)
            _coverage_merge_record_rejected_candidate(
                rejected_records,
                reason,
                group,
                path_config,
                candidate=candidate,
                before_preview=before_preview,
                after_preview=after_preview,
            )
            continue
        coverage_fraction = float(after_preview.get("coverage_fraction", 0.0) or 0.0)
        if coverage_fraction + 1e-9 < max(0.0, min(1.0, path_config.coverage_merge_min_coverage_fraction)):
            _increment_object_diagnostic(rejected, "low_coverage_fraction")
            _coverage_merge_record_rejected_candidate(
                rejected_records,
                "low_coverage_fraction",
                group,
                path_config,
                candidate=candidate,
                before_preview=before_preview,
                after_preview=after_preview,
            )
            continue
        before_score = float(before_preview.get("score", 0.0) or 0.0)
        raw_after_score = float(after_preview.get("score", 0.0) or 0.0)
        boustrophedon_reward = _agent_task_boustrophedon_merge_reward(before_preview, after_preview, path_config)
        after_score = raw_after_score - boustrophedon_reward
        improvement = before_score - after_score
        improvement_ratio = improvement / max(abs(before_score), 1e-9)
        min_improvement_ratio = max(path_config.coverage_merge_min_improvement_ratio, 0.0)
        preview_coherent_gain = _agent_task_preview_has_coherent_boustrophedon_gain(
            before_preview,
            after_preview,
        )
        if improvement_ratio + 1e-9 < min_improvement_ratio and not preview_coherent_gain:
            _increment_object_diagnostic(rejected, "low_objective_improvement")
            _coverage_merge_record_rejected_candidate(
                rejected_records,
                "low_objective_improvement",
                group,
                path_config,
                candidate=candidate,
                before_preview=before_preview,
                after_preview=after_preview,
            )
            continue
        validation_count += 1
        if progress_callback is not None:
            progress_callback(
                event="unified_candidate_validate",
                agent_id=agent_id,
                candidate_index=candidate_count,
                source_region_count=len(group),
                objective_delta=round(improvement, 6),
                objective_delta_ratio=round(improvement_ratio, 6),
                coherent_boustrophedon_gain=bool(preview_coherent_gain),
            )
        validated = _coverage_merge_pattern_preview(
            candidate,
            config,
            path_config,
            obstacle_field,
            preview_cache,
            validate_internal=True,
        )
        if not validated.get("feasible", False):
            reason = str(validated.get("reason", "internal_sweep_infeasible"))
            _increment_object_diagnostic(rejected, reason)
            _coverage_merge_record_rejected_candidate(
                rejected_records,
                reason,
                group,
                path_config,
                candidate=candidate,
                before_preview=before_preview,
                after_preview=validated,
            )
            continue
        validated_raw_score = float(validated.get("score", raw_after_score) or 0.0)
        validated_reward = _agent_task_boustrophedon_merge_reward(before_preview, validated, path_config)
        validated_score = validated_raw_score - validated_reward
        validated_improvement = before_score - validated_score
        validated_ratio = validated_improvement / max(abs(before_score), 1e-9)
        validated_coherent_gain = _agent_task_preview_has_coherent_boustrophedon_gain(
            before_preview,
            validated,
        )
        if validated_ratio + 1e-9 < min_improvement_ratio and not validated_coherent_gain:
            _increment_object_diagnostic(rejected, "validated_low_objective_improvement")
            _coverage_merge_record_rejected_candidate(
                rejected_records,
                "validated_low_objective_improvement",
                group,
                path_config,
                candidate=candidate,
                before_preview=before_preview,
                after_preview=validated,
            )
            continue
        validated_gain_metrics = _agent_task_boustrophedon_gain_metrics(before_preview, validated)
        validated_accept_reason = (
            "agent_task_unified_coverage_objective_improved"
            if validated_ratio + 1e-9 >= min_improvement_ratio
            else "agent_task_unified_coherent_boustrophedon_gain"
        )
        source_ids = [region.region_id for region in group]
        candidate.metadata.update(
            {
                "agent_task_unified_merge": "true",
                "agent_task_unified_agent_id": str(agent_id),
                "agent_task_unified_source_count": str(len(source_ids)),
                "agent_task_unified_source_ids": ",".join(source_ids),
                "agent_task_unified_rectangularity": f"{rectangularity:.6f}",
                "agent_task_full_component_merge": str(bool(full_component_candidate)).lower(),
                "merge_accept_reason": validated_accept_reason,
                "merge_objective_before": f"{before_score:.6f}",
                "merge_objective_after": f"{validated_score:.6f}",
                "merge_raw_objective_after": f"{validated_raw_score:.6f}",
                "merge_boustrophedon_reward": f"{validated_reward:.6f}",
                "merge_objective_delta": f"{validated_improvement:.6f}",
                "merge_objective_delta_ratio": f"{validated_ratio:.6f}",
                "merge_best_pattern_id": str(validated.get("pattern_id", "")),
                "merge_best_pattern_scan_axis": str(validated.get("scan_axis", "")),
                "merge_best_pattern_coverage_fraction": f"{float(validated.get('coverage_fraction', 0.0) or 0.0):.6f}",
                "merge_source_pass_count": str(int(before_preview.get("pass_count", 0) or 0)),
                "merge_candidate_pass_count": str(int(validated.get("pass_count", 0) or 0)),
                "merge_source_max_pass_length": f"{float(before_preview.get('max_pass_length', 0.0) or 0.0):.6f}",
                "merge_candidate_max_pass_length": f"{float(validated.get('max_pass_length', 0.0) or 0.0):.6f}",
                "merge_pass_reduction": str(int(validated_gain_metrics["pass_reduction"])),
                "merge_pass_reduction_ratio": f"{float(validated_gain_metrics['pass_reduction_ratio']):.6f}",
                "merge_long_pass_gain_ratio": f"{float(validated_gain_metrics['length_gain_ratio']):.6f}",
                "merge_coherent_boustrophedon_gain": str(bool(validated_coherent_gain)).lower(),
            }
        )
        accepted_records.append(
            {
                "agent_id": agent_id,
                "region_id": candidate.region_id,
                "source_region_ids": source_ids,
                "source_region_count": len(source_ids),
                "rectangularity": round(rectangularity, 6),
                "min_rectangularity": round(group_min_rectangularity, 6),
                "shape_class": candidate.metadata.get("shape_class", ""),
                "scan_support_mode": candidate.metadata.get("scan_support_mode", ""),
                "objective_before": round(before_score, 6),
                "objective_after": round(validated_score, 6),
                "raw_objective_after": round(validated_raw_score, 6),
                "boustrophedon_reward": round(validated_reward, 6),
                "objective_delta": round(validated_improvement, 6),
                "objective_delta_ratio": round(validated_ratio, 6),
                "accept_reason": validated_accept_reason,
                "coverage_fraction": round(float(validated.get("coverage_fraction", 0.0) or 0.0), 6),
                "source_pass_count": int(before_preview.get("pass_count", 0) or 0),
                "candidate_pass_count": int(validated.get("pass_count", 0) or 0),
                "source_max_pass_length": round(float(before_preview.get("max_pass_length", 0.0) or 0.0), 6),
                "candidate_max_pass_length": round(float(validated.get("max_pass_length", 0.0) or 0.0), 6),
                "pass_reduction": int(validated_gain_metrics["pass_reduction"]),
                "pass_reduction_ratio": round(float(validated_gain_metrics["pass_reduction_ratio"]), 6),
                "long_pass_gain_ratio": round(float(validated_gain_metrics["length_gain_ratio"]), 6),
                "coherent_boustrophedon_gain": bool(validated_coherent_gain),
                "best_pattern_scan_axis": validated.get("scan_axis", ""),
                "unified_group_merge": True,
                "full_component_merge": bool(full_component_candidate),
            }
        )
        objective_delta += validated_improvement
        accepted_source_ids.update(source_ids)
        current = [region for region in current if region.region_id not in accepted_source_ids]
        current.append(candidate)
        serial += 1
        if progress_callback is not None:
            progress_callback(
                event="unified_candidate_accepted",
                agent_id=agent_id,
                candidate_index=candidate_count,
                source_region_count=len(source_ids),
                candidate_region_id=candidate.region_id,
                objective_delta=round(validated_improvement, 6),
                candidate_pass_count=int(validated.get("pass_count", 0) or 0),
                candidate_max_pass_length=round(float(validated.get("max_pass_length", 0.0) or 0.0), 6),
                full_component_candidate=bool(full_component_candidate),
            )

    current.sort(key=lambda region: (region.center[0], region.center[1], region.region_id))
    _populate_region_neighbors(current)
    if progress_callback is not None:
        progress_callback(
            event="unified_agent_done",
            agent_id=agent_id,
            candidate_count=candidate_count,
            validation_count=validation_count,
            accepted_count=len(accepted_records),
            budget_exhausted=budget_exhausted,
            budget_reason=budget_reason,
            elapsed_sec=round(time.perf_counter() - started, 3),
        )
    return current, {
        "agent_task_unified_candidate_count": candidate_count,
        "agent_task_unified_validation_count": validation_count,
        "agent_task_unified_accepted_count": len(accepted_records),
        "agent_task_unified_rejected_by_reason": rejected,
        "agent_task_unified_rejected_candidates": rejected_records,
        "agent_task_unified_regions": accepted_records,
        "agent_task_unified_objective_delta": objective_delta,
        "agent_task_unified_budget_exhausted": budget_exhausted,
        "agent_task_unified_budget_reason": budget_reason,
    }


def _agent_task_unified_candidate_groups(
    regions: Sequence,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> List[List]:
    region_list = list(regions)
    groups: List[List] = []
    seen: set[Tuple[str, ...]] = set()

    def add_group(group: Sequence) -> None:
        key = tuple(sorted(region.region_id for region in group))
        if len(key) < 2 or key in seen:
            return
        seen.add(key)
        groups.append(list(group))

    for component in _agent_task_merge_connected_components(region_list, config, path_config, obstacle_field):
        add_group(component)
        for axis in ("x", "y"):
            for axis_group in _agent_task_axis_compatible_components(component, axis, config, path_config, obstacle_field):
                add_group(axis_group)

    return groups


def _agent_task_merge_connected_components(
    regions: Sequence,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> List[List]:
    region_list = list(regions)
    adjacency: Dict[str, set[str]] = {region.region_id: set() for region in region_list}
    by_id = {region.region_id: region for region in region_list}
    for idx, first in enumerate(region_list):
        for second in region_list[idx + 1 :]:
            can_join, _ = _coverage_merge_regions_can_join(first, second, config, path_config, obstacle_field)
            if not can_join:
                continue
            adjacency[first.region_id].add(second.region_id)
            adjacency[second.region_id].add(first.region_id)
    components: List[List] = []
    seen: set[str] = set()
    for region in region_list:
        if region.region_id in seen:
            continue
        stack = [region.region_id]
        seen.add(region.region_id)
        component_ids: List[str] = []
        while stack:
            region_id = stack.pop()
            component_ids.append(region_id)
            for neighbor_id in sorted(adjacency.get(region_id, ())):
                if neighbor_id in seen:
                    continue
                seen.add(neighbor_id)
                stack.append(neighbor_id)
        components.append([by_id[region_id] for region_id in sorted(component_ids)])
    return components


def _agent_task_axis_compatible_components(
    regions: Sequence,
    axis: str,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> List[List]:
    region_list = list(regions)
    adjacency: Dict[str, set[str]] = {region.region_id: set() for region in region_list}
    by_id = {region.region_id: region for region in region_list}
    for idx, first in enumerate(region_list):
        for second in region_list[idx + 1 :]:
            if not _agent_task_regions_axis_compatible(first, second, axis, config, path_config, obstacle_field):
                continue
            adjacency[first.region_id].add(second.region_id)
            adjacency[second.region_id].add(first.region_id)
    groups: List[List] = []
    seen: set[str] = set()
    for region in sorted(region_list, key=lambda item: (item.center[0], item.center[1], item.region_id)):
        if region.region_id in seen:
            continue
        stack = [region.region_id]
        seen.add(region.region_id)
        group_ids: List[str] = []
        while stack:
            region_id = stack.pop()
            group_ids.append(region_id)
            for neighbor_id in sorted(adjacency.get(region_id, ())):
                if neighbor_id in seen:
                    continue
                seen.add(neighbor_id)
                stack.append(neighbor_id)
        if len(group_ids) >= 2:
            groups.append([by_id[region_id] for region_id in sorted(group_ids)])
    return groups


def _agent_task_regions_axis_compatible(
    first,
    second,
    axis: str,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> bool:
    can_join, _ = _coverage_merge_regions_can_join(first, second, config, path_config, obstacle_field)
    if not can_join:
        return False
    ax0, ay0, ax1, ay1 = first.bounds
    bx0, by0, bx1, by1 = second.bounds
    gap_limit = max(config.footprint.width_wf * max(path_config.coverage_merge_gap_bridge_width_factor, 0.0), 1e-6)
    if axis == "x":
        y_overlap = min(ay1, by1) - max(ay0, by0)
        min_height = max(min(ay1 - ay0, by1 - by0), 1e-9)
        horizontal_gap = max(bx0 - ax1, ax0 - bx1, 0.0)
        return y_overlap / min_height >= 0.65 and horizontal_gap <= gap_limit
    if axis == "y":
        x_overlap = min(ax1, bx1) - max(ax0, bx0)
        min_width = max(min(ax1 - ax0, bx1 - bx0), 1e-9)
        vertical_gap = max(by0 - ay1, ay0 - by1, 0.0)
        return x_overlap / min_width >= 0.65 and vertical_gap <= gap_limit
    return False


def _cell_group_bounds_area(bounds: Tuple[float, float, float, float]) -> float:
    return max(bounds[2] - bounds[0], 0.0) * max(bounds[3] - bounds[1], 0.0)


def _agent_task_group_rectangularity(group: Sequence) -> float:
    cells = _coverage_merge_member_cells(group)
    if not cells:
        return 0.0
    bounds = _cell_group_bounds_for_merge(cells)
    source_area = sum(max(float(cell.area), 0.0) for cell in cells)
    return source_area / max(_cell_group_bounds_area(bounds), 1e-9)


def _coverage_merge_regions_can_join(
    first,
    second,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> Tuple[bool, str]:
    ax0, ay0, ax1, ay1 = first.bounds
    bx0, by0, bx1, by1 = second.bounds
    x_overlap = min(ax1, bx1) - max(ax0, bx0)
    y_overlap = min(ay1, by1) - max(ay0, by0)
    min_shared = max(config.footprint.width_wf * 0.25, 1e-6)
    if (abs(ax1 - bx0) <= 1e-6 or abs(bx1 - ax0) <= 1e-6) and y_overlap >= min_shared:
        return True, ""
    if (abs(ay1 - by0) <= 1e-6 or abs(by1 - ay0) <= 1e-6) and x_overlap >= min_shared:
        return True, ""
    gap_limit = max(config.footprint.width_wf * max(path_config.coverage_merge_gap_bridge_width_factor, 0.0), 0.0)
    if gap_limit <= 1e-9:
        return False, "non_contiguous"
    horizontal_gap = max(bx0 - ax1, ax0 - bx1, 0.0)
    vertical_gap = max(by0 - ay1, ay0 - by1, 0.0)
    if horizontal_gap <= gap_limit and vertical_gap <= 1e-9 and y_overlap >= min_shared:
        x_min, x_max = (ax1, bx0) if ax1 <= bx0 else (bx1, ax0)
        bridge = [(x_min, max(ay0, by0)), (x_max, max(ay0, by0)), (x_max, min(ay1, by1)), (x_min, min(ay1, by1))]
        if obstacle_field is None or not polygon_collides_with_obstacles(bridge, obstacle_field, inflated=True):
            return True, ""
        return False, "gap_bridge_obstacle_collision"
    if vertical_gap <= gap_limit and horizontal_gap <= 1e-9 and x_overlap >= min_shared:
        y_min, y_max = (ay1, by0) if ay1 <= by0 else (by1, ay0)
        bridge = [(max(ax0, bx0), y_min), (min(ax1, bx1), y_min), (min(ax1, bx1), y_max), (max(ax0, bx0), y_max)]
        if obstacle_field is None or not polygon_collides_with_obstacles(bridge, obstacle_field, inflated=True):
            return True, ""
        return False, "gap_bridge_obstacle_collision"
    return False, "non_contiguous"


def _coverage_merge_candidate_from_group(
    serial: int,
    group: Sequence,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    region_id_prefix: str = "coverage_merge_region",
) -> Tuple[object | None, str]:
    cells = _coverage_merge_member_cells(group)
    if not cells:
        return None, "empty_source_cells"
    max_members = max(int(path_config.coverage_merge_max_members), 1)
    if len(cells) > max_members:
        return None, "too_many_source_cells"
    bounds = _cell_group_bounds_for_merge(cells)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    if width <= 1e-9 or height <= 1e-9:
        return None, "degenerate_bounds"
    mission_area = max(config.mission.area_length_x * config.mission.area_length_y, 1e-9)
    bbox_area = width * height
    max_area = mission_area * max(float(path_config.coverage_merge_max_area_fraction), 1e-6)
    if bbox_area > max_area + 1e-9:
        return None, "area_limit"
    source_area = sum(max(float(cell.area), 0.0) for cell in cells)
    source_region_ids = [region.region_id for region in group]
    deep_source_ids = _coverage_merge_deep_source_ids(group)
    equivalent_source_region_count = sum(
        max(
            int(_metadata_float(getattr(region, "metadata", {}) or {}, "merge_equivalent_source_region_count", 1.0)),
            1,
        )
        for region in group
    )
    preferred_axis = "x" if width >= height else "y"
    support_span = height if preferred_axis == "x" else width
    center = (
        sum(cell.center[0] * max(cell.area, 0.0) for cell in cells) / max(source_area, 1e-9),
        sum(cell.center[1] * max(cell.area, 0.0) for cell in cells) / max(source_area, 1e-9),
    )
    polygon = _bounds_polygon(bounds)
    full_rectangle = abs(source_area - bbox_area) <= max(bbox_area, 1.0) * 1e-6
    bbox_collides = _bounds_collide_with_obstacles(bounds, obstacle_field)
    base_metadata = {
        "coverage_aware_merged": "true",
        "static_obstacle_aware": str(obstacle_field is not None).lower(),
        "source_cell_count": str(len(cells)),
        "source_region_ids": ",".join(deep_source_ids),
        "merge_source_region_ids": ",".join(deep_source_ids),
        "merge_fallback_source_ids": ",".join(source_region_ids),
        "merge_equivalent_source_region_count": str(equivalent_source_region_count),
        "dominant_scan_axis": preferred_axis,
        "support_span": f"{support_span:.6f}",
        "area_priority": f"{bbox_area / mission_area:.6f}",
    }
    if full_rectangle and not bbox_collides:
        return (
            DecomposedRegion(
                region_id=f"{region_id_prefix}_{serial}",
                bounds=bounds,
                polygon=polygon,
                center=((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0),
                area=bbox_area,
                preferred_axis=preferred_axis,
                source_algorithm="coverage_aware_rectangle_merge",
                neighbors=[],
                metadata={
                    **base_metadata,
                    "convex_region_decomposition": "true",
                    "shape_class": "rectangle",
                    "scan_support_mode": "true_polygon",
                },
            ),
            "",
        )
    if not path_config.coverage_merge_allow_nonconvex_composite:
        return None, "obstacle_collision" if bbox_collides else "non_rectangular_union"
    shape_class = "near_convex_composite" if source_area / max(bbox_area, 1e-9) >= 0.85 else "nonconvex_composite"
    return (
        CompositeFreeSpaceRegion(
            region_id=f"{region_id_prefix}_{serial}",
            bounds=bounds,
            polygon=polygon,
            center=center,
            area=source_area,
            preferred_axis=preferred_axis,
            source_algorithm="coverage_aware_composite_merge",
            member_cells=cells,
            neighbors=[],
            metadata={
                **base_metadata,
                "is_composite": "true",
                "composite_bounds_are_envelope": "true",
                "shape_class": shape_class,
                "scan_support_mode": "member_cell_intervals",
            },
        ),
        "",
    )


def _coverage_merge_before_preview(
    regions: Sequence,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    preview_cache: Dict[Tuple[object, ...], Dict[str, object]],
) -> Dict[str, object]:
    score = 0.0
    feasible_count = 0
    previews: List[Dict[str, object]] = []
    for region in regions:
        preview = _coverage_merge_pattern_preview(region, config, path_config, obstacle_field, preview_cache, validate_internal=False)
        previews.append(preview)
        if preview.get("feasible", False):
            feasible_count += 1
        score += float(preview.get("score", 0.0) or 0.0)
    pass_count = sum(int(preview.get("pass_count", 0) or 0) for preview in previews)
    coverage_length = sum(float(preview.get("coverage_length", 0.0) or 0.0) for preview in previews)
    max_pass_length = max(
        (float(preview.get("max_pass_length", 0.0) or 0.0) for preview in previews),
        default=0.0,
    )
    connector_proxy = (
        max(config.fleet.min_turn_radius, 0.0) * math.pi
        + max(config.footprint.length_lf, 0.0)
        + 2.0 * max(config.footprint.width_wf, 0.0)
    ) * max(path_config.transition_length_weight, 0.0)
    boundary_turn_proxy = _coverage_merge_boundary_turn_proxy(regions, path_config)
    score += max(len(regions) - 1, 0) * connector_proxy
    score += boundary_turn_proxy
    return {
        "score": score,
        "feasible_count": feasible_count,
        "connector_proxy": connector_proxy,
        "boundary_turn_proxy": boundary_turn_proxy,
        "pass_count": pass_count,
        "coverage_length": coverage_length,
        "max_pass_length": max_pass_length,
    }


def _agent_task_boustrophedon_gain_metrics(
    before_preview: Dict[str, object],
    after_preview: Dict[str, object],
) -> Dict[str, float]:
    before_pass_count = int(before_preview.get("pass_count", 0) or 0)
    after_pass_count = int(after_preview.get("pass_count", 0) or 0)
    before_max_pass = float(before_preview.get("max_pass_length", 0.0) or 0.0)
    after_max_pass = float(after_preview.get("max_pass_length", 0.0) or 0.0)
    pass_reduction = max(before_pass_count - after_pass_count, 0)
    pass_reduction_ratio = pass_reduction / max(before_pass_count, 1)
    length_gain_ratio = after_max_pass / max(before_max_pass, 1e-9) if before_max_pass > 1e-9 else 1.0
    return {
        "before_pass_count": float(before_pass_count),
        "after_pass_count": float(after_pass_count),
        "before_max_pass_length": before_max_pass,
        "after_max_pass_length": after_max_pass,
        "pass_reduction": float(pass_reduction),
        "pass_reduction_ratio": pass_reduction_ratio,
        "length_gain_ratio": length_gain_ratio,
    }


def _agent_task_has_coherent_boustrophedon_gain(
    source_pass_count: int,
    real_pass_count: int,
    source_max_pass: float,
    real_max_pass: float,
) -> bool:
    pass_reduction = max(int(source_pass_count) - int(real_pass_count), 0)
    pass_reduction_ratio = pass_reduction / max(int(source_pass_count), 1)
    length_gain_ratio = real_max_pass / max(source_max_pass, 1e-9) if source_max_pass > 1e-9 else 1.0
    return (
        (pass_reduction >= 2 and length_gain_ratio >= 1.10)
        or (pass_reduction_ratio >= 0.20 and length_gain_ratio >= 1.05)
        or (pass_reduction >= 3 and length_gain_ratio >= 1.05)
        or (pass_reduction >= 1 and length_gain_ratio >= 1.45)
    )


def _agent_task_preview_has_coherent_boustrophedon_gain(
    before_preview: Dict[str, object],
    after_preview: Dict[str, object],
) -> bool:
    metrics = _agent_task_boustrophedon_gain_metrics(before_preview, after_preview)
    return _agent_task_has_coherent_boustrophedon_gain(
        int(metrics["before_pass_count"]),
        int(metrics["after_pass_count"]),
        float(metrics["before_max_pass_length"]),
        float(metrics["after_max_pass_length"]),
    )


def _agent_task_boustrophedon_merge_reward(
    before_preview: Dict[str, object],
    after_preview: Dict[str, object],
    path_config: PathPlanningConfig,
) -> float:
    """Reward merged candidates that create longer continuous sweep passes.

    The normal preview score already captures time and turn length, but it can
    under-value the user's desired behavior: one agent should execute a larger
    assigned block as a coherent boustrophedon task.  This small adjustment keeps
    coverage feasibility unchanged while making long-pass, low-turn candidates
    easier to select.
    """

    metrics = _agent_task_boustrophedon_gain_metrics(before_preview, after_preview)
    before_max_pass = float(metrics["before_max_pass_length"])
    pass_reduction = float(metrics["pass_reduction"])
    length_gain_ratio = float(metrics["length_gain_ratio"])
    long_pass_gain = max(length_gain_ratio - 1.0, 0.0)
    turn_unit_reward = max(path_config.turn_angle_weight, 0.0) * math.pi + max(path_config.turn_count_weight, 0.0)
    pass_reward = pass_reduction * max(turn_unit_reward, 1.0)
    long_pass_reward = before_max_pass * long_pass_gain * 0.35
    return max(pass_reward + long_pass_reward, 0.0)


def _coverage_merge_boundary_turn_proxy(regions: Sequence, path_config: PathPlanningConfig) -> float:
    base_turn_cost = (
        max(path_config.turn_angle_weight, 0.0) * math.pi
        + max(path_config.turn_count_weight, 0.0)
    )
    boundary_count = max(len(regions) - 1, 0)
    if boundary_count <= 0 or base_turn_cost <= 0.0:
        return 0.0
    continuity_bonus = 0.0
    tolerance = math.radians(max(path_config.oriented_sweep_angle_tolerance_deg, 0.0))
    for first, second in zip(regions[:-1], regions[1:]):
        first_angle = _coverage_merge_region_axis_angle(first)
        second_angle = _coverage_merge_region_axis_angle(second)
        delta = abs((first_angle - second_angle) % math.pi)
        delta = min(delta, math.pi - delta)
        if delta <= tolerance:
            compatibility = 1.0
        else:
            compatibility = max(0.0, 1.0 - delta / (math.pi / 2.0))
        continuity_bonus += 0.75 * compatibility
    return base_turn_cost * (boundary_count + continuity_bonus)


def _coverage_merge_region_axis_angle(region) -> float:
    metadata = getattr(region, "metadata", {}) or {}
    axis = str(
        metadata.get(
            "merge_best_pattern_scan_axis",
            metadata.get("dominant_scan_axis", getattr(region, "preferred_axis", "x")),
        )
    )
    if axis == "theta":
        axis = str(getattr(region, "preferred_axis", "x"))
    return _scan_axis_angle_rad(axis)


def _coverage_merge_record_rejected_candidate(
    records: List[Dict[str, object]],
    reason: str,
    source_regions: Sequence,
    path_config: PathPlanningConfig,
    candidate=None,
    before_preview: Dict[str, object] | None = None,
    after_preview: Dict[str, object] | None = None,
    limit: int = 64,
) -> None:
    if len(records) >= max(limit, 0):
        return
    before_preview = before_preview or {}
    after_preview = after_preview or {}
    before_score = float(before_preview.get("score", 0.0) or 0.0)
    after_score = float(after_preview.get("score", 0.0) or 0.0)
    record: Dict[str, object] = {
        "source_region_ids": [region.region_id for region in source_regions],
        "reason": reason or "unknown",
        "boundary_turn_proxy": round(
            float(before_preview.get("boundary_turn_proxy", _coverage_merge_boundary_turn_proxy(source_regions, path_config)) or 0.0),
            6,
        ),
        "objective_before": round(before_score, 6),
        "objective_after": round(after_score, 6),
        "objective_delta": round(before_score - after_score, 6),
        "coverage_fraction": round(float(after_preview.get("coverage_fraction", 0.0) or 0.0), 6),
        "source_pass_count": int(before_preview.get("pass_count", 0) or 0),
        "candidate_pass_count": int(after_preview.get("pass_count", 0) or 0),
        "source_max_pass_length": round(float(before_preview.get("max_pass_length", 0.0) or 0.0), 6),
        "candidate_max_pass_length": round(float(after_preview.get("max_pass_length", 0.0) or 0.0), 6),
        "source_coverage_length": round(float(before_preview.get("coverage_length", 0.0) or 0.0), 6),
        "candidate_coverage_length": round(float(after_preview.get("coverage_length", 0.0) or 0.0), 6),
    }
    if candidate is not None:
        record.update(
            {
                "candidate_region_id": candidate.region_id,
                "candidate_area": round(float(candidate.area), 6),
                "shape_class": str(candidate.metadata.get("shape_class", "")),
            }
        )
    records.append(record)


def _coverage_merge_pattern_preview(
    region,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    preview_cache: Dict[Tuple[object, ...], Dict[str, object]],
    validate_internal: bool,
) -> Dict[str, object]:
    cache_key = (
        "validated" if validate_internal else "preview",
        region.region_id,
        tuple(round(float(value), 6) for value in region.bounds),
        str(region.metadata.get("merge_source_region_ids", region.metadata.get("source_region_ids", ""))),
        str(region.metadata.get("shape_class", "")),
        len(getattr(region, "member_cells", []) or []),
    )
    if cache_key in preview_cache:
        return preview_cache[cache_key]
    preview_limit = max(int(path_config.coverage_merge_preview_pattern_limit), 1)
    preview_config = replace(
        path_config,
        enable_large_map_sweep_prefilter=True,
        max_candidate_axes=1,
        max_oriented_sweep_angles_per_region=max(0, min(int(path_config.max_oriented_sweep_angles_per_region), 1)),
        include_axis_aligned_sweep_fallbacks=False,
        max_prefiltered_patterns_per_region=preview_limit,
        max_prefiltered_variants_per_pattern=max(1, min(int(path_config.max_prefiltered_variants_per_pattern), 2)),
        max_entry_exit_patterns_per_region=max(1, min(int(path_config.max_entry_exit_patterns_per_region), preview_limit, 2)),
    )
    raw_patterns = generate_region_patterns(region, config, preview_config, obstacle_field=obstacle_field)
    if validate_internal:
        stats: Dict[str, object] = {}
        _, feasible, infeasible, _ = _build_region_sweep_paths(
            {region.region_id: raw_patterns},
            config,
            preview_config,
            obstacle_field,
            stats=stats,
        )
        patterns = feasible.get(region.region_id, [])
        reason = ",".join(str(item) for record in infeasible for item in record.get("reasons", [])) if infeasible else ""
    else:
        patterns = [pattern for pattern in raw_patterns if pattern.feasible and pattern.passes]
        reason = "no_preview_pattern"
    if not patterns:
        result = {"feasible": False, "score": float("inf"), "reason": reason or "no_feasible_pattern"}
        preview_cache[cache_key] = result
        return result
    patterns = [_annotate_pattern_coverage_quality(pattern, config, path_config) for pattern in patterns]
    patterns.sort(key=lambda pattern: (_light_pattern_score(pattern, config, path_config), pattern.pattern_id))
    best = patterns[0]
    coverage_fraction = _estimated_pattern_coverage_fraction(best, config)
    coverage_deficit = max(0.0, max(path_config.coverage_merge_min_coverage_fraction, 0.0) - coverage_fraction)
    score = (
        _light_pattern_score(best, config, path_config)
        + max(path_config.coverage_priority_weight, 0.0) * coverage_deficit
        + 0.1 * max(len(best.passes) - 1, 0)
    )
    result = {
        "feasible": True,
        "score": score,
        "pattern_id": best.pattern_id,
        "scan_axis": best.scan_axis,
        "coverage_fraction": coverage_fraction,
        "coverage_length": best.coverage_length,
        "max_pass_length": max((coverage_pass.length for coverage_pass in best.passes), default=0.0),
        "pass_count": len(best.passes),
        "turn_length": best.turn_length,
        "turn_angle": best.turn_angle,
        "open_chain_count": int(_metadata_float(best.metadata, "open_chain_count", 0.0)),
        "open_chain_validation_only": best.metadata.get("open_chain_validation_only") == "true",
    }
    preview_cache[cache_key] = result
    return result


def _coverage_merge_member_cells(group: Sequence) -> List[FreeSpaceCell]:
    cells: List[FreeSpaceCell] = []
    seen: set[str] = set()
    for region in group:
        member_cells = list(getattr(region, "member_cells", []) or [])
        if member_cells:
            for cell in member_cells:
                if cell.cell_id in seen:
                    continue
                cells.append(cell)
                seen.add(cell.cell_id)
            continue
        cell_id = region.region_id
        if cell_id in seen:
            continue
        cells.append(
            FreeSpaceCell(
                cell_id=cell_id,
                bounds=region.bounds,
                polygon=list(region.polygon),
                center=region.center,
                area=region.area,
                preferred_axis=region.preferred_axis,
                source_algorithm=region.source_algorithm,
                neighbors=list(region.neighbors),
                metadata=dict(region.metadata),
            )
        )
        seen.add(cell_id)
    return cells


def _coverage_merge_deep_source_ids(group: Sequence) -> List[str]:
    source_ids: List[str] = []
    for region in group:
        encoded = str(region.metadata.get("merge_source_region_ids") or region.metadata.get("source_region_ids") or "")
        items = [item.strip() for item in encoded.split(",") if item.strip()] or [region.region_id]
        for item in items:
            if item not in source_ids:
                source_ids.append(item)
    return source_ids


def _cell_group_bounds_for_merge(cells: Sequence[FreeSpaceCell]) -> Tuple[float, float, float, float]:
    return (
        min(cell.bounds[0] for cell in cells),
        min(cell.bounds[1] for cell in cells),
        max(cell.bounds[2] for cell in cells),
        max(cell.bounds[3] for cell in cells),
    )


def _increment_object_diagnostic(diagnostics: Dict[str, int], key: str) -> None:
    diagnostics[key or "unknown"] = int(diagnostics.get(key or "unknown", 0)) + 1


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
        fallback_source = str(region.metadata.get("merge_fallback_source_ids") or region.metadata.get("source_region_ids", ""))
        source_ids = [
            item.strip()
            for item in fallback_source.split(",")
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
    stats.setdefault("open_chain_flexible_exit_variant_attempt_count", 0)
    stats.setdefault("open_chain_flexible_exit_variant_success_count", 0)
    stats.setdefault("open_chain_flexible_exit_variant_failure_count", 0)
    stats.setdefault("open_chain_flexible_exit_variant_failure_reasons", {})
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
        region_budget_exhausted = False
        region_time_budget = max(float(path_config.sweep_region_validation_time_budget_sec), 0.0)
        stop_after_first_feasible = (
            bool(path_config.large_map_stop_after_first_feasible_sweep_variant)
            and _large_map_mode_enabled(config, path_config)
        )
        for pattern in candidates:
            if region_budget_exhausted:
                break
            stop_after_first_for_pattern = stop_after_first_feasible and not _pattern_needs_connector_variant_diversity(pattern)
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
                    if _pattern_needs_connector_variant_diversity(variant):
                        variant = replace(
                            variant,
                            metadata={
                                **variant.metadata,
                                "connector_variant_diversity_preserved": "true",
                            },
                        )
                    feasible_for_region.append((variant, _sweep_path_from_pattern(variant)))
                    feasible_variant_count += 1
                    flex_result = _build_open_chain_flexible_exit_variant(
                        variant,
                        config,
                        path_config,
                        obstacle_field,
                        stats,
                    )
                    if flex_result is not None:
                        flex_variant, flex_segments = flex_result
                        flex_variant = _annotate_pattern_coverage_quality(
                            _annotate_pattern_internal_repeat(flex_variant, config, path_config, obstacle_field),
                            config,
                            path_config,
                        )
                        if _pattern_needs_connector_variant_diversity(flex_variant):
                            flex_variant = replace(
                                flex_variant,
                                metadata={
                                    **flex_variant.metadata,
                                    "connector_variant_diversity_preserved": "true",
                                },
                            )
                        sweep_segment_templates[_pattern_template_key(flex_variant)] = (copy.deepcopy(flex_segments), "")
                        feasible_for_region.append((flex_variant, _sweep_path_from_pattern(flex_variant)))
                        feasible_variant_count += 1
                    if stop_after_first_for_pattern:
                        break
                else:
                    failed_variant_count += 1
                    reasons.append(f"{variant.pattern_id}:{reason}")
                if region_time_budget > 0.0 and time.perf_counter() - region_started >= region_time_budget:
                    region_budget_exhausted = True
                    reasons.append("region_validation_time_budget_exhausted")
                    break
            if stop_after_first_for_pattern and feasible_for_region:
                break
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
            if any(_pattern_needs_connector_variant_diversity(item[0]) for item in feasible_for_region):
                limit = max(limit, _merged_region_connector_variant_limit(path_config))
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
                region_budget_exhausted=region_budget_exhausted,
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
    if any(_pattern_needs_connector_variant_diversity(variant) for variant in variants):
        limit = max(limit, _merged_region_connector_variant_limit(path_config))
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


def _pattern_is_agent_task_merge_pattern(pattern: RegionCoveragePattern) -> bool:
    metadata = getattr(pattern, "metadata", {}) or {}
    if any(
        str(metadata.get(key, "")).lower() == "true"
        for key in (
            "agent_task_strip_merge",
            "agent_task_unified_merge",
        )
    ):
        return True
    region_id = str(getattr(pattern, "region_id", ""))
    return (
        "_strip_task_region_" in region_id
        or "_unified_task_region_" in region_id
        or "_task_merge_region_" in region_id
    )


def _record_open_chain_flexible_exit_variant_failure(
    stats: Dict[str, object] | None,
    reason: str,
) -> None:
    if stats is None:
        return
    stats["open_chain_flexible_exit_variant_failure_count"] = (
        int(stats.get("open_chain_flexible_exit_variant_failure_count", 0) or 0) + 1
    )
    reasons = dict(stats.get("open_chain_flexible_exit_variant_failure_reasons", {}) or {})
    reasons[reason] = int(reasons.get(reason, 0) or 0) + 1
    stats["open_chain_flexible_exit_variant_failure_reasons"] = reasons


def _build_open_chain_flexible_exit_variant(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    stats: Dict[str, object] | None,
) -> Tuple[RegionCoveragePattern, List[PathSegmentSpec]] | None:
    if not bool(getattr(path_config, "enable_open_chain_flexible_exit_variants", True)):
        return None
    if not path_config.enable_open_sweep_chain_tsp:
        return None
    if pattern.metadata.get("open_chain_validation_only") != "true":
        return None
    if int(getattr(path_config, "open_chain_flexible_exit_variant_limit", 1)) <= 0:
        return None
    if (
        bool(getattr(path_config, "open_chain_flexible_exit_variants_for_agent_task_only", True))
        and not _pattern_is_agent_task_merge_pattern(pattern)
    ):
        return None
    if pattern.metadata.get("open_chain_flexible_exit_variant") == "true":
        return None

    if stats is not None:
        stats["open_chain_flexible_exit_variant_attempt_count"] = (
            int(stats.get("open_chain_flexible_exit_variant_attempt_count", 0) or 0) + 1
        )
    probe_pattern = copy.deepcopy(pattern)
    probe_pattern.metadata = dict(probe_pattern.metadata)
    probe_path_config = replace(path_config, open_chain_allow_flexible_exit=True)
    segments, reason = _build_internal_sweep_segments(
        probe_pattern,
        config,
        probe_path_config,
        obstacle_field,
        start_time=0.0,
        segment_prefix=f"flex_exit_probe_{pattern.region_id}_{pattern.pattern_id}",
    )
    if reason:
        _record_open_chain_flexible_exit_variant_failure(stats, reason)
        return None
    if not segments:
        _record_open_chain_flexible_exit_variant_failure(stats, "empty_probe_segments")
        return None
    if probe_pattern.metadata.get("open_chain_flexible_exit") != "true":
        _record_open_chain_flexible_exit_variant_failure(stats, "flexible_exit_not_needed")
        return None

    pattern_id = f"{pattern.pattern_id}_flex_exit"
    metadata = {
        **probe_pattern.metadata,
        "open_chain_flexible_exit_variant": "true",
        "open_chain_flexible_exit_variant_from": pattern.pattern_id,
        "open_chain_validation_only": "false",
        "entry_pose": _pose_metadata(probe_pattern.entry_pose),
        "exit_pose": _pose_metadata(probe_pattern.exit_pose),
    }
    variant = replace(
        probe_pattern,
        pattern_id=pattern_id,
        metadata=metadata,
    )
    for segment in segments:
        segment.metadata["pattern_id"] = pattern_id
        segment.metadata["open_chain_flexible_exit_variant"] = "true"
    if stats is not None:
        stats["open_chain_flexible_exit_variant_success_count"] = (
            int(stats.get("open_chain_flexible_exit_variant_success_count", 0) or 0) + 1
        )
    return variant, [segment for segment in segments if segment.length > 1e-9]


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
                shape_class=str(region.metadata.get("shape_class", "")),
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
                convexity_status=str(region.metadata.get("convexity_status", "")),
                selected_angle_sources=str(region.metadata.get("selected_angle_sources", "")),
                oriented_sweep_skip_reason=str(region.metadata.get("oriented_sweep_skip_reason", "")),
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
                rebuilt = _apply_external_connector_pocket(rebuilt, config, path_config)
                variants.append(_recalculate_pattern_cost(rebuilt, config))
    return _dedupe_pattern_variants(variants)


def _apply_external_connector_pocket(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> RegionCoveragePattern:
    if config.active_agent_id is None or not path_config.enable_adaptive_pass_retraction or not pattern.passes:
        return pattern
    vehicle_half_length = (
        config.vehicle_footprint.length / 2.0
        if config.vehicle_footprint is not None
        else config.footprint.length_lf / 2.0
    )
    required = max(config.fleet.min_turn_radius + vehicle_half_length, vehicle_half_length)
    ratio_limit = max(0.0, min(float(path_config.max_pass_retraction_ratio), 1.0))
    min_length = max(
        config.footprint.width_wf * path_config.retraction_min_pass_length_factor,
        config.footprint.length_lf * path_config.retraction_min_pass_length_factor,
    )
    passes = list(pattern.passes)
    total_retraction = 0.0

    first = passes[0]
    backward = (-math.cos(first.start_pose.psi), -math.sin(first.start_pose.psi))
    entry_clearance = _ray_distance_to_mission_boundary(first.start_pose, backward, config)
    entry_retraction = min(
        max(required - entry_clearance, 0.0),
        max(first.length - min_length, 0.0),
        first.length * ratio_limit,
    )
    if entry_retraction > 1e-9:
        passes[0] = _retract_coverage_pass_endpoint(first, entry_retraction, at_entry=True)
        total_retraction += entry_retraction

    last = passes[-1]
    forward = (math.cos(last.end_pose.psi), math.sin(last.end_pose.psi))
    exit_clearance = _ray_distance_to_mission_boundary(last.end_pose, forward, config)
    exit_retraction = min(
        max(required - exit_clearance, 0.0),
        max(last.length - min_length, 0.0),
        last.length * ratio_limit,
    )
    if exit_retraction > 1e-9:
        passes[-1] = _retract_coverage_pass_endpoint(last, exit_retraction, at_entry=False)
        total_retraction += exit_retraction

    if total_retraction <= 1e-9:
        return pattern
    return replace(
        pattern,
        passes=passes,
        entry_pose=passes[0].start_pose,
        exit_pose=passes[-1].end_pose,
        metadata={
            **pattern.metadata,
            "external_connector_pocket_retraction": f"{total_retraction:.6f}",
            "external_connector_required_clearance": f"{required:.6f}",
        },
    )


def _ray_distance_to_mission_boundary(
    pose: Pose2D,
    direction: Tuple[float, float],
    config: PlannerConfig,
) -> float:
    dx, dy = direction
    distances: List[float] = []
    if dx > 1e-12:
        distances.append((config.mission.area_length_x - pose.x) / dx)
    elif dx < -1e-12:
        distances.append((0.0 - pose.x) / dx)
    if dy > 1e-12:
        distances.append((config.mission.area_length_y - pose.y) / dy)
    elif dy < -1e-12:
        distances.append((0.0 - pose.y) / dy)
    positive = [distance for distance in distances if distance >= -1e-9]
    return max(0.0, min(positive, default=float("inf")))


def _retract_coverage_pass_endpoint(
    coverage_pass: CoveragePass,
    distance: float,
    *,
    at_entry: bool,
) -> CoveragePass:
    length = max(float(coverage_pass.length), 0.0)
    if length <= 1e-9 or distance <= 1e-9:
        return coverage_pass
    ratio = min(distance / length, 1.0)
    start = coverage_pass.start_pose
    end = coverage_pass.end_pose
    if at_entry:
        start = Pose2D(
            start.x + (end.x - start.x) * ratio,
            start.y + (end.y - start.y) * ratio,
            start.psi,
        )
    else:
        end = Pose2D(
            end.x + (start.x - end.x) * ratio,
            end.y + (start.y - end.y) * ratio,
            end.psi,
        )
    return replace(
        coverage_pass,
        start_pose=start,
        end_pose=end,
        length=max(length - distance, 0.0),
    )


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
    return _pattern_quality_components(pattern, path_config)["total_quality_penalty"]


def _pattern_quality_components(
    pattern: RegionCoveragePattern,
    path_config: PathPlanningConfig | None,
) -> Dict[str, float]:
    coverage_length = max(float(pattern.coverage_length), 1e-9)
    pass_count = max(len(pattern.passes), 1)
    total_retraction = _metadata_float(pattern.metadata, "total_retraction_length", 0.0)
    endpoint_retraction = _metadata_float(pattern.metadata, "endpoint_total_retraction_length", 0.0)
    retraction_length = total_retraction + endpoint_retraction
    retraction_ratio = retraction_length / coverage_length
    internal_repeat_overlap = _metadata_float(pattern.metadata, "internal_repeat_overlap_length", 0.0)
    internal_repeat_penalty = _metadata_float(pattern.metadata, "internal_repeat_penalty", 0.0)
    internal_repeat_ratio = internal_repeat_overlap / coverage_length
    turn_angle_per_pass = float(pattern.turn_angle) / pass_count
    turn_angle_per_coverage_meter = float(pattern.turn_angle) / coverage_length
    failed_retractions = _metadata_float(pattern.metadata, "retraction_failed_count", 0.0)
    extended_retractions = _metadata_float(pattern.metadata, "retraction_extended_count", 0.0)
    coverage_deficit = _metadata_float(pattern.metadata, "coverage_deficit", 0.0)
    if path_config is None:
        return {
            "coverage_deficit": coverage_deficit,
            "retraction_length": retraction_length,
            "retraction_ratio": retraction_ratio,
            "retraction_penalty": 0.0,
            "turn_angle_per_pass": turn_angle_per_pass,
            "turn_angle_per_coverage_meter": turn_angle_per_coverage_meter,
            "turn_penalty": 0.0,
            "internal_repeat_overlap_length": internal_repeat_overlap,
            "internal_repeat_ratio": internal_repeat_ratio,
            "internal_repeat_penalty": internal_repeat_penalty,
            "repeat_penalty": internal_repeat_penalty,
            "retraction_failure_penalty": 0.0,
            "total_quality_penalty": internal_repeat_penalty,
        }
    retraction_penalty = max(path_config.pattern_retraction_penalty_weight, 0.0) * 50.0 * retraction_ratio
    turn_penalty = max(path_config.pattern_turn_penalty_weight, 0.0) * (
        turn_angle_per_pass + 5.0 * turn_angle_per_coverage_meter
    )
    repeat_penalty = max(path_config.pattern_repeat_penalty_multiplier, 0.0) * (
        internal_repeat_penalty / coverage_length
        + max(path_config.main_repeat_path_penalty_weight, 0.0) * 20.0 * internal_repeat_ratio
    )
    retraction_failure_penalty = 25.0 * failed_retractions + 5.0 * extended_retractions
    total = retraction_penalty + turn_penalty + repeat_penalty + retraction_failure_penalty
    components = {
        "coverage_deficit": coverage_deficit,
        "retraction_length": retraction_length,
        "retraction_ratio": retraction_ratio,
        "retraction_penalty": retraction_penalty,
        "turn_angle_per_pass": turn_angle_per_pass,
        "turn_angle_per_coverage_meter": turn_angle_per_coverage_meter,
        "turn_penalty": turn_penalty,
        "internal_repeat_overlap_length": internal_repeat_overlap,
        "internal_repeat_ratio": internal_repeat_ratio,
        "internal_repeat_penalty": internal_repeat_penalty,
        "repeat_penalty": repeat_penalty,
        "retraction_failure_penalty": retraction_failure_penalty,
        "total_quality_penalty": total,
    }
    if path_config.report_score_components:
        pattern.metadata.update({f"quality_{key}": f"{value:.6f}" for key, value in components.items()})
    return components


def _coverage_quality_priority_enabled(config: PlannerConfig, path_config: PathPlanningConfig) -> bool:
    return _large_map_mode_enabled(config, path_config)


def _region_workload_weights(
    graph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> Dict[str, float]:
    initial_poses = [state.pose() for state in config.fleet.initial_states_3dof]
    weights: Dict[str, float] = {}
    for region_id in graph.regions:
        candidates = [pattern for pattern in graph.patterns.get(region_id, []) if pattern.feasible]
        if not candidates:
            weights[region_id] = float(graph.node_weights.get(region_id, 0.0))
            continue
        best_pattern = min(candidates, key=lambda pattern: (_pattern_sort_key(pattern, config, path_config), pattern.pattern_id))
        initial_connection = min(
            (_transition_length(pose, best_pattern.entry_pose, config) for pose in initial_poses),
            default=0.0,
        )
        weights[region_id] = max(
            1e-6,
            best_pattern.estimated_time
            + 0.25 * best_pattern.total_length
            + max(path_config.turn_angle_weight, 0.0) * best_pattern.turn_angle
            + _pattern_quality_penalty(best_pattern, path_config)
            + 0.1 * initial_connection,
        )
    return weights


def _optimize_joint_region_candidate_assignment(
    assignment: BalancedAssignment,
    graph: RegionGraph,
    feasible_regions: Sequence[DecomposedRegion],
    feasible_patterns: Dict[str, List[RegionCoveragePattern]],
    sweep_paths: Dict[str, RegionSweepPath],
    sweep_segment_templates: Dict[str, Tuple[List[PathSegmentSpec], str]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    progress_callback: Callable[..., None] | None = None,
) -> Tuple[BalancedAssignment, CoverageOwnershipMap, Dict[int, Dict[str, object]] | None, Dict[str, object]]:
    """Improve assignment/order choices by evaluating real region-TSP solutions.

    This is intentionally a conservative wrapper around the existing solver:
    candidate assignments are generated through connected boundary moves, swaps,
    and simple reorders, then scored only after the normal connector/sweep
    validation chain has built concrete segments.
    """

    optimizer_started = time.perf_counter()
    time_budget = max(float(path_config.joint_optimizer_time_budget_sec), 0.0)

    def elapsed() -> float:
        return time.perf_counter() - optimizer_started

    def budget_exhausted() -> bool:
        return time_budget > 0.0 and elapsed() >= time_budget

    def emit(event: str, **payload: object) -> None:
        if progress_callback is None:
            return
        progress_callback(
            event=event,
            elapsed_sec=round(elapsed(), 3),
            time_budget_sec=round(time_budget, 3),
            **payload,
        )

    diagnostics: Dict[str, object] = {
        "joint_optimizer_status": "disabled",
        "global_joint_objective": 0.0,
        "cross_agent_connector_overlap_length": 0.0,
        "cross_agent_crossing_count": 0,
        "joint_mission_makespan": 0.0,
        "joint_total_agent_work_time": 0.0,
        "joint_agent_time_imbalance": 0.0,
        "joint_improvement_iteration_count": 0,
        "joint_candidate_attempt_count": 0,
        "joint_candidate_accept_count": 0,
        "joint_reject_reasons": {},
    }
    ownership_map = build_coverage_ownership_map(
        feasible_regions,
        assignment.agent_regions,
        config,
        path_config,
        obstacle_field=obstacle_field,
    )
    if not path_config.enable_joint_region_candidate_optimization:
        return assignment, ownership_map, None, diagnostics
    if max(path_config.joint_improvement_iterations, 0) <= 0:
        diagnostics["joint_optimizer_status"] = "iteration_budget_zero"
        return assignment, ownership_map, None, diagnostics
    if not feasible_regions:
        diagnostics["joint_optimizer_status"] = "empty_region_set"
        return assignment, ownership_map, None, diagnostics
    assigned_region_count = sum(len(region_ids) for region_ids in assignment.agent_regions.values())
    diagnostics["joint_region_count"] = assigned_region_count
    emit(
        "start",
        assigned_region_count=assigned_region_count,
        agent_count=len(assignment.agent_regions),
    )
    if max(config.mission.area_length_x, config.mission.area_length_y) < max(path_config.large_map_size_threshold, 1e-6):
        diagnostics["joint_optimizer_status"] = "small_map_skipped"
        emit("skipped", reason="small_map_skipped")
        return assignment, ownership_map, None, diagnostics
    joint_region_limit = max(int(path_config.joint_large_map_region_limit), 0)
    if joint_region_limit > 0 and assigned_region_count > joint_region_limit:
        diagnostics.update(
            {
                "joint_optimizer_status": "skipped_large_region_budget",
                "joint_region_count": assigned_region_count,
                "joint_large_map_region_limit": joint_region_limit,
            }
        )
        emit(
            "skipped",
            reason="skipped_large_region_budget",
            assigned_region_count=assigned_region_count,
            joint_large_map_region_limit=joint_region_limit,
        )
        return assignment, ownership_map, None, diagnostics

    eval_agent_budget = max(float(path_config.joint_eval_agent_time_budget_sec), 0.0)
    eval_step_budget = max(float(path_config.joint_eval_step_time_budget_sec), 0.0)
    eval_path_config = replace(
        path_config,
        monitor_stages=False,
        large_map_tsp_agent_time_budget_sec=(
            min(float(path_config.large_map_tsp_agent_time_budget_sec), eval_agent_budget)
            if eval_agent_budget > 0.0
            else float(path_config.large_map_tsp_agent_time_budget_sec)
        ),
        large_map_tsp_step_time_budget_sec=(
            min(float(path_config.large_map_tsp_step_time_budget_sec), eval_step_budget)
            if eval_step_budget > 0.0
            else float(path_config.large_map_tsp_step_time_budget_sec)
        ),
        large_map_tsp_max_candidate_attempts_per_step=min(
            max(int(path_config.large_map_tsp_max_candidate_attempts_per_step), 1),
            max(4, int(path_config.joint_connector_edge_limit)),
        ),
        large_map_tsp_max_obstacle_aware_attempts_per_agent=min(
            max(int(path_config.large_map_tsp_max_obstacle_aware_attempts_per_agent), 0),
            4,
        ),
    )
    limited_patterns = _joint_limited_patterns(feasible_patterns, config, eval_path_config)
    current_regions = {agent_id: list(region_ids) for agent_id, region_ids in assignment.agent_regions.items()}
    emit(
        "initial_evaluation_start",
        eval_agent_time_budget_sec=round(float(eval_path_config.large_map_tsp_agent_time_budget_sec), 3),
        eval_step_time_budget_sec=round(float(eval_path_config.large_map_tsp_step_time_budget_sec), 3),
    )
    try:
        current = _evaluate_joint_assignment_solution(
            agent_regions=current_regions,
            feasible_regions=feasible_regions,
            feasible_patterns=limited_patterns,
            sweep_paths=sweep_paths,
            sweep_segment_templates=sweep_segment_templates,
            config=config,
            path_config=eval_path_config,
            report_path_config=path_config,
            obstacle_field=obstacle_field,
        )
    except Exception as exc:  # pragma: no cover - safety fallback for large experiment runs
        diagnostics["joint_optimizer_status"] = "fallback"
        diagnostics["joint_fallback_reason"] = f"initial_evaluation_failed:{type(exc).__name__}:{exc}"
        emit("fallback", reason=diagnostics["joint_fallback_reason"])
        return assignment, ownership_map, None, diagnostics
    emit(
        "initial_evaluation_done",
        valid=bool(current.get("valid", False)),
        skipped_region_count=int(current.get("skipped_region_count", 0) or 0),
        objective=round(float(current.get("objective", 0.0) or 0.0), 6),
    )
    if budget_exhausted():
        diagnostics["joint_optimizer_status"] = "budget_fallback"
        diagnostics["joint_fallback_reason"] = "joint_optimizer_time_budget_exhausted_after_initial_evaluation"
        diagnostics["joint_elapsed_sec"] = elapsed()
        emit("fallback", reason=diagnostics["joint_fallback_reason"])
        return assignment, ownership_map, None, diagnostics
    if not current.get("valid", False):
        diagnostics["joint_optimizer_status"] = "fallback"
        diagnostics["joint_fallback_reason"] = str(current.get("failure_reason", "initial_solution_invalid"))
        emit("fallback", reason=diagnostics["joint_fallback_reason"])
        return assignment, ownership_map, None, diagnostics
    if int(current.get("skipped_region_count", 0) or 0) > 0 or _joint_solution_budget_exhausted(current):
        diagnostics["joint_optimizer_status"] = "fallback"
        diagnostics["joint_fallback_reason"] = "initial_evaluation_incomplete_or_budget_exhausted"
        diagnostics["joint_initial_skipped_region_count"] = int(current.get("skipped_region_count", 0) or 0)
        emit(
            "fallback",
            reason=diagnostics["joint_fallback_reason"],
            skipped_region_count=diagnostics["joint_initial_skipped_region_count"],
        )
        return assignment, ownership_map, None, diagnostics

    reject_reasons: Dict[str, int] = {}
    accepted = 0
    attempted = 0
    evaluated_signatures = {_joint_assignment_signature(current_regions)}
    effective_iterations = min(
        max(int(path_config.joint_improvement_iterations), 0),
        1 if assigned_region_count > 40 else 2,
    )
    effective_candidate_limit = min(
        max(int(path_config.joint_connector_edge_limit), 1),
        2 if assigned_region_count > 40 else 4,
    )
    diagnostics["joint_effective_iteration_limit"] = effective_iterations
    diagnostics["joint_effective_candidate_limit"] = effective_candidate_limit
    for iteration in range(effective_iterations):
        if budget_exhausted():
            diagnostics["joint_optimizer_status"] = "budget_fallback"
            diagnostics["joint_fallback_reason"] = "joint_optimizer_time_budget_exhausted"
            diagnostics["joint_elapsed_sec"] = elapsed()
            emit("fallback", reason=diagnostics["joint_fallback_reason"], iteration=iteration)
            return assignment, ownership_map, None, diagnostics
        candidates = _joint_assignment_neighbors(
            current_regions,
            graph,
            config,
            path_config,
            max_candidates=effective_candidate_limit,
        )
        emit(
            "iteration_start",
            iteration=iteration + 1,
            candidate_count=len(candidates),
            current_objective=round(float(current.get("objective", 0.0) or 0.0), 6),
        )
        best_candidate = None
        best_operation = ""
        for operation, candidate_regions in candidates:
            if budget_exhausted():
                diagnostics["joint_optimizer_status"] = "budget_fallback"
                diagnostics["joint_fallback_reason"] = "joint_optimizer_time_budget_exhausted"
                diagnostics["joint_elapsed_sec"] = elapsed()
                emit("fallback", reason=diagnostics["joint_fallback_reason"], iteration=iteration + 1)
                return assignment, ownership_map, None, diagnostics
            signature = _joint_assignment_signature(candidate_regions)
            if signature in evaluated_signatures:
                _joint_increment_reason(reject_reasons, "duplicate_candidate")
                continue
            evaluated_signatures.add(signature)
            attempted += 1
            try:
                candidate = _evaluate_joint_assignment_solution(
                    agent_regions=candidate_regions,
                    feasible_regions=feasible_regions,
                    feasible_patterns=limited_patterns,
                    sweep_paths=sweep_paths,
                    sweep_segment_templates=sweep_segment_templates,
                    config=config,
                    path_config=eval_path_config,
                    report_path_config=path_config,
                    obstacle_field=obstacle_field,
                )
            except Exception as exc:  # pragma: no cover - candidate-level fallback
                _joint_increment_reason(reject_reasons, f"evaluation_failed:{type(exc).__name__}")
                continue
            if not candidate.get("valid", False):
                _joint_increment_reason(reject_reasons, str(candidate.get("failure_reason", "invalid_candidate")))
                continue
            if not _joint_solution_improves(candidate, current):
                _joint_increment_reason(reject_reasons, "objective_not_improved")
                continue
            if best_candidate is None or float(candidate["objective"]) < float(best_candidate["objective"]):
                best_candidate = candidate
                best_operation = operation
        if best_candidate is None:
            emit("iteration_done", iteration=iteration + 1, accepted=False, reject_reasons=dict(reject_reasons))
            break
        current = best_candidate
        current_regions = {agent_id: list(region_ids) for agent_id, region_ids in current["agent_regions"].items()}
        accepted += 1
        diagnostics[f"joint_iteration_{iteration + 1}_operation"] = best_operation
        emit(
            "iteration_done",
            iteration=iteration + 1,
            accepted=True,
            operation=best_operation,
            objective=round(float(current.get("objective", 0.0) or 0.0), 6),
        )

    optimized_assignment = _joint_assignment_from_regions(current_regions, graph)
    diagnostics.update(
        {
            "joint_optimizer_status": "success" if accepted else "no_improvement",
            "global_joint_objective": float(current.get("objective", 0.0)),
            "cross_agent_connector_overlap_length": float(current.get("cross_agent_connector_overlap_length", 0.0)),
            "cross_agent_crossing_count": int(current.get("cross_agent_crossing_count", 0)),
            "joint_mission_makespan": float(current.get("mission_makespan", 0.0)),
            "joint_total_agent_work_time": float(current.get("total_agent_work_time", 0.0)),
            "joint_agent_time_imbalance": float(current.get("agent_time_imbalance", 0.0)),
            "joint_improvement_iteration_count": accepted,
            "joint_candidate_attempt_count": attempted,
            "joint_candidate_accept_count": accepted,
            "joint_reject_reasons": dict(reject_reasons),
            "joint_executed_region_count": int(current.get("executed_region_count", 0)),
            "joint_skipped_region_count": int(current.get("skipped_region_count", 0)),
            "joint_noncover_repeat_overlap_length": float(current.get("noncover_repeat_overlap_length", 0.0)),
            "joint_load_imbalance": float(current.get("load_imbalance", 0.0)),
            "joint_elapsed_sec": elapsed(),
        }
    )
    return optimized_assignment, current["ownership_map"], current["results"], diagnostics


def _joint_solution_budget_exhausted(solution: Dict[str, object]) -> bool:
    results = solution.get("results", {})
    if not isinstance(results, dict):
        return False
    for result in results.values():
        if not isinstance(result, dict):
            continue
        metadata = result.get("tsp_solver_metadata", {})
        if isinstance(metadata, dict) and bool(metadata.get("large_map_greedy_budget_exhausted", False)):
            return True
    return False


def _joint_limited_patterns(
    feasible_patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> Dict[str, List[RegionCoveragePattern]]:
    limit = max(int(path_config.joint_candidate_patterns_per_region), 1)
    return {
        region_id: sorted(
            [pattern for pattern in patterns if pattern.feasible],
            key=lambda pattern: (_pattern_sort_key(pattern, config, path_config), pattern.pattern_id),
        )[:limit]
        for region_id, patterns in feasible_patterns.items()
    }


def _evaluate_joint_assignment_solution(
    agent_regions: Dict[int, List[str]],
    feasible_regions: Sequence[DecomposedRegion],
    feasible_patterns: Dict[str, List[RegionCoveragePattern]],
    sweep_paths: Dict[str, RegionSweepPath],
    sweep_segment_templates: Dict[str, Tuple[List[PathSegmentSpec], str]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    report_path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> Dict[str, object]:
    ownership_map = build_coverage_ownership_map(
        feasible_regions,
        agent_regions,
        config,
        report_path_config,
        obstacle_field=obstacle_field,
    )
    results: Dict[int, Dict[str, object]] = {}
    agents: Dict[int, AgentPathPlan] = {}
    assigned_count = sum(len(region_ids) for region_ids in agent_regions.values())
    for agent_id, region_ids in sorted(agent_regions.items()):
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
        results[agent_id] = result
        agents[agent_id] = AgentPathPlan(
            agent_id=agent_id,
            source_algorithm="paper_style_region_tsp_joint_eval",
            segments=list(result["segments"]),
            metrics=_agent_metrics(result["segments"], config, obstacle_field),
        )
    totals = _global_metrics(agents)
    invalid_count = (
        totals["out_of_bounds_segment_count"]
        + totals["obstacle_collision_segment_count"]
        + totals["kinematic_infeasible_segment_count"]
        + totals["dynamic_infeasible_segment_count"]
    )
    if invalid_count > 0:
        return {
            "valid": False,
            "failure_reason": "constraint_violation",
            "agent_regions": agent_regions,
            "results": results,
            "ownership_map": ownership_map,
        }
    executed_count = sum(len(result.get("final_order", [])) for result in results.values())
    skipped_count = max(assigned_count - executed_count, 0)
    repeat_metrics = _agent_repeat_metrics(agents, report_path_config)
    noncover_repeat = sum(item["overlap_length"] for item in repeat_metrics.values())
    cross_score = cross_agent_overlap_metrics(
        agents,
        ownership_map,
        report_path_config,
        config=config,
        annotate=False,
    )
    cross_connector_overlap = (
        float(cross_score.overlap_by_kind.get("transit", 0.0))
        + float(cross_score.overlap_by_kind.get("turn", 0.0))
    )
    crossing_count = _cross_agent_crossing_count(agents)
    load_imbalance = _joint_imbalance_ratio(agent_regions, _joint_region_loads(agent_regions, feasible_patterns, report_path_config))
    time_metrics = _joint_agent_time_metrics(agents)
    objective = _joint_solution_objective(
        skipped_count=skipped_count,
        executed_count=executed_count,
        totals=totals,
        noncover_repeat=noncover_repeat,
        cross_agent_overlap=cross_score.overlap_length,
        crossing_count=crossing_count,
        load_imbalance=load_imbalance,
        mission_makespan=time_metrics["mission_makespan"],
        agent_time_imbalance=time_metrics["agent_time_imbalance"],
        report_path_config=report_path_config,
    )
    return {
        "valid": True,
        "objective": objective,
        "agent_regions": {agent_id: list(region_ids) for agent_id, region_ids in agent_regions.items()},
        "results": results,
        "ownership_map": ownership_map,
        "executed_region_count": executed_count,
        "skipped_region_count": skipped_count,
        "noncover_repeat_overlap_length": noncover_repeat,
        "cross_agent_overlap_length": cross_score.overlap_length,
        "cross_agent_connector_overlap_length": cross_connector_overlap,
        "cross_agent_crossing_count": crossing_count,
        "load_imbalance": load_imbalance,
        "mission_makespan": time_metrics["mission_makespan"],
        "total_agent_work_time": time_metrics["total_agent_work_time"],
        "agent_time_imbalance": time_metrics["agent_time_imbalance"],
        "transition_length": totals["transition_length"],
        "total_turn_angle": totals["total_turn_angle"],
        "total_length": totals["total_length"],
    }


def _joint_solution_improves(candidate: Dict[str, object], current: Dict[str, object]) -> bool:
    if int(candidate.get("executed_region_count", 0)) < int(current.get("executed_region_count", 0)):
        return False
    if int(candidate.get("skipped_region_count", 0)) > int(current.get("skipped_region_count", 0)):
        return False
    if float(candidate.get("load_imbalance", 0.0)) > float(current.get("load_imbalance", 0.0)) + 1e-6:
        return False
    return float(candidate.get("objective", math.inf)) + 1e-6 < float(current.get("objective", math.inf))


def _joint_agent_time_metrics(agents: Dict[int, AgentPathPlan]) -> Dict[str, float]:
    times = [float(agent.metrics.get("estimated_time", 0.0) or 0.0) for agent in agents.values()]
    if not times:
        return {
            "mission_makespan": 0.0,
            "total_agent_work_time": 0.0,
            "agent_time_imbalance": 0.0,
        }
    mean_time = sum(times) / max(len(times), 1)
    return {
        "mission_makespan": max(times),
        "total_agent_work_time": sum(times),
        "agent_time_imbalance": 0.0 if mean_time <= 1e-9 else (max(times) - min(times)) / mean_time,
    }


def _joint_solution_objective(
    *,
    skipped_count: int,
    executed_count: int,
    totals: Dict[str, float],
    noncover_repeat: float,
    cross_agent_overlap: float,
    crossing_count: int,
    load_imbalance: float,
    mission_makespan: float,
    agent_time_imbalance: float,
    report_path_config: PathPlanningConfig,
) -> float:
    return (
        1_000_000.0 * skipped_count
        - 10_000.0 * executed_count
        + float(totals.get("transition_length", 0.0))
        + 0.05 * float(totals.get("total_length", 0.0))
        + max(report_path_config.global_noncover_repeat_weight, 0.0) * noncover_repeat
        + max(report_path_config.global_cross_agent_overlap_weight, 0.0) * cross_agent_overlap
        + 10.0 * max(report_path_config.global_cross_agent_overlap_weight, 0.0) * crossing_count
        + max(report_path_config.global_turn_angle_weight, 0.0) * float(totals.get("total_turn_angle", 0.0))
        + max(report_path_config.turn_count_weight, 0.0) * float(totals.get("turn_count", 0.0))
        + max(report_path_config.time_weight, 0.0) * mission_makespan
        + max(report_path_config.load_balance_weight, 0.0) * mission_makespan * agent_time_imbalance
        + 1000.0 * load_imbalance
    )


def _joint_assignment_neighbors(
    agent_regions: Dict[int, List[str]],
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    max_candidates: int,
) -> List[Tuple[str, Dict[int, List[str]]]]:
    if max_candidates <= 0:
        return []
    weights = graph.node_weights
    loads = _joint_region_loads(agent_regions, weights=weights)
    agents_by_heavy = sorted(agent_regions, key=lambda agent_id: loads.get(agent_id, 0.0), reverse=True)
    agents_by_light = sorted(agent_regions, key=lambda agent_id: loads.get(agent_id, 0.0))
    candidates: List[Tuple[str, Dict[int, List[str]]]] = []
    seen: set[Tuple[Tuple[int, Tuple[str, ...]], ...]] = set()

    def add(operation: str, regions: Dict[int, List[str]]) -> None:
        if len(candidates) >= max_candidates:
            return
        signature = _joint_assignment_signature(regions)
        if signature in seen:
            return
        if not _joint_assignment_connected(regions, graph):
            return
        seen.add(signature)
        candidates.append((operation, regions))

    balance_budget = max(1, max_candidates // 2)
    for operation, moved in _joint_load_balancing_move_candidates(
        agent_regions,
        graph,
        config,
        max_candidates=balance_budget,
    ):
        add(operation, moved)
        if len(candidates) >= max_candidates:
            return candidates

    scan_compat_budget = max(1, max_candidates // 3)
    for operation, moved in _joint_scan_axis_compatible_move_candidates(
        agent_regions,
        graph,
        config,
        path_config,
        max_candidates=scan_compat_budget,
    ):
        add(operation, moved)
        if len(candidates) >= max_candidates:
            return candidates

    for agent_id, region_ids in sorted(agent_regions.items()):
        reorder_budget = max(1, max_candidates // 3)
        for operation, reordered_ids in _joint_two_opt_order_candidates(
            agent_id,
            region_ids,
            graph,
            config,
            path_config,
            max_candidates=reorder_budget,
        ):
            reordered = {candidate_agent: list(candidate_regions) for candidate_agent, candidate_regions in agent_regions.items()}
            reordered[agent_id] = reordered_ids
            add(operation, reordered)
            if len(candidates) >= max_candidates:
                return candidates

    for source in agents_by_heavy:
        if len(agent_regions.get(source, [])) <= 1:
            continue
        for target in agents_by_light:
            if source == target:
                continue
            for region_id in _joint_boundary_regions(agent_regions, graph, source, target):
                moved = {agent_id: list(region_ids) for agent_id, region_ids in agent_regions.items()}
                moved[source].remove(region_id)
                moved[target].append(region_id)
                moved[target] = _joint_order_for_agent(target, moved[target], graph, config, reverse=False)
                add(f"move:{region_id}:{source}->{target}", moved)
                if len(candidates) >= max_candidates:
                    return candidates

    for source in agents_by_heavy:
        for target in agents_by_light:
            if source >= target or not agent_regions.get(source) or not agent_regions.get(target):
                continue
            source_candidates = _joint_boundary_regions(agent_regions, graph, source, target)[:2]
            target_candidates = _joint_boundary_regions(agent_regions, graph, target, source)[:2]
            for left in source_candidates:
                for right in target_candidates:
                    swapped = {agent_id: list(region_ids) for agent_id, region_ids in agent_regions.items()}
                    swapped[source].remove(left)
                    swapped[target].remove(right)
                    swapped[source].append(right)
                    swapped[target].append(left)
                    swapped[source] = _joint_order_for_agent(source, swapped[source], graph, config, reverse=False)
                    swapped[target] = _joint_order_for_agent(target, swapped[target], graph, config, reverse=False)
                    add(f"swap:{left}<->{right}", swapped)
                    if len(candidates) >= max_candidates:
                        return candidates

    for agent_id, region_ids in sorted(agent_regions.items()):
        if len(region_ids) <= 2:
            continue
        for reverse in (False, True):
            reordered = {candidate_agent: list(candidate_regions) for candidate_agent, candidate_regions in agent_regions.items()}
            reordered[agent_id] = _joint_order_for_agent(agent_id, region_ids, graph, config, reverse=reverse)
            if reordered[agent_id] != list(region_ids):
                add(f"reorder:{agent_id}:{'reverse' if reverse else 'nearest'}", reordered)
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def _joint_load_balancing_move_candidates(
    agent_regions: Dict[int, List[str]],
    graph: RegionGraph,
    config: PlannerConfig,
    max_candidates: int,
) -> List[Tuple[str, Dict[int, List[str]]]]:
    if max_candidates <= 0 or len(agent_regions) < 2:
        return []
    loads = _joint_region_loads(agent_regions, weights=graph.node_weights)
    current_imbalance = _joint_imbalance_ratio(agent_regions, loads)
    if current_imbalance <= 1e-9:
        return []
    agents_by_heavy = sorted(agent_regions, key=lambda agent_id: loads.get(agent_id, 0.0), reverse=True)
    agents_by_light = sorted(agent_regions, key=lambda agent_id: loads.get(agent_id, 0.0))
    proposals: List[Tuple[float, float, int, int, str, Dict[int, List[str]]]] = []
    seen: set[Tuple[Tuple[int, Tuple[str, ...]], ...]] = set()
    for source in agents_by_heavy:
        source_regions = list(agent_regions.get(source, []))
        if len(source_regions) <= 1:
            continue
        for target in agents_by_light:
            if source == target:
                continue
            for region_id in _joint_boundary_regions(agent_regions, graph, source, target):
                if region_id not in source_regions:
                    continue
                moved = {agent_id: list(region_ids) for agent_id, region_ids in agent_regions.items()}
                moved[source].remove(region_id)
                moved[target].append(region_id)
                moved[target] = _joint_order_for_agent(target, moved[target], graph, config)
                if not _joint_assignment_connected(moved, graph):
                    continue
                moved_loads = dict(loads)
                weight = float(graph.node_weights.get(region_id, 0.0))
                moved_loads[source] = moved_loads.get(source, 0.0) - weight
                moved_loads[target] = moved_loads.get(target, 0.0) + weight
                candidate_imbalance = _joint_imbalance_ratio(moved, moved_loads)
                gain = current_imbalance - candidate_imbalance
                if gain <= 1e-9:
                    continue
                signature = _joint_assignment_signature(moved)
                if signature in seen:
                    continue
                seen.add(signature)
                proposals.append((gain, weight, source, target, region_id, moved))
    proposals.sort(key=lambda item: (-item[0], -item[1], item[2], item[3], item[4]))
    return [
        (f"load_balance_move:{region_id}:{source}->{target}", moved)
        for gain, weight, source, target, region_id, moved in proposals[:max_candidates]
    ]


def _joint_scan_axis_compatible_move_candidates(
    agent_regions: Dict[int, List[str]],
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    max_candidates: int,
) -> List[Tuple[str, Dict[int, List[str]]]]:
    """Move connected boundary regions when doing so joins scan-compatible neighbors.

    This expands the joint optimizer's search space toward the real objective:
    adjacent regions with compatible sweep directions should have a chance to be
    planned by the same USV, so their boustrophedon tracks can be ordered with
    fewer inter-region turns. The final acceptance still goes through the full
    region-TSP and connector validation objective.
    """

    if max_candidates <= 0 or len(agent_regions) < 2:
        return []
    current_loads = _joint_region_loads(agent_regions, weights=graph.node_weights)
    current_imbalance = _joint_imbalance_ratio(agent_regions, current_loads)
    current_score = _joint_scan_axis_assignment_score(agent_regions, graph, config, path_config)
    proposals: List[Tuple[float, float, float, int, int, str, Dict[int, List[str]]]] = []
    seen: set[Tuple[Tuple[int, Tuple[str, ...]], ...]] = set()
    agent_ids = sorted(agent_regions)
    for source in agent_ids:
        source_regions = list(agent_regions.get(source, []))
        if len(source_regions) <= 1:
            continue
        for target in agent_ids:
            if source == target:
                continue
            for region_id in _joint_boundary_regions(agent_regions, graph, source, target):
                if region_id not in source_regions:
                    continue
                moved = {agent_id: list(region_ids) for agent_id, region_ids in agent_regions.items()}
                moved[source].remove(region_id)
                moved[target].append(region_id)
                moved[target] = _joint_order_for_agent(target, moved[target], graph, config)
                if not _joint_assignment_connected(moved, graph):
                    continue
                moved_loads = dict(current_loads)
                weight = float(graph.node_weights.get(region_id, 0.0))
                moved_loads[source] = moved_loads.get(source, 0.0) - weight
                moved_loads[target] = moved_loads.get(target, 0.0) + weight
                candidate_imbalance = _joint_imbalance_ratio(moved, moved_loads)
                if candidate_imbalance > current_imbalance + 1e-6:
                    continue
                candidate_score = _joint_scan_axis_assignment_score(moved, graph, config, path_config)
                gain = candidate_score - current_score
                if gain <= 1e-9:
                    continue
                signature = _joint_assignment_signature(moved)
                if signature in seen:
                    continue
                seen.add(signature)
                proposals.append((gain, -candidate_imbalance, weight, source, target, region_id, moved))
    proposals.sort(key=lambda item: (-item[0], item[1], -item[2], item[3], item[4], item[5]))
    return [
        (f"scan_axis_move:{region_id}:{source}->{target}", moved)
        for gain, _, weight, source, target, region_id, moved in proposals[:max_candidates]
    ]


def _joint_scan_axis_assignment_score(
    agent_regions: Dict[int, List[str]],
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> float:
    score = 0.0
    for region_ids in agent_regions.values():
        region_set = set(region_ids)
        counted_edges: set[Tuple[str, str]] = set()
        for region_id in region_ids:
            for neighbor_id in graph.adjacency.get(region_id, []):
                if neighbor_id not in region_set:
                    continue
                edge = tuple(sorted((region_id, neighbor_id)))
                if edge in counted_edges:
                    continue
                counted_edges.add(edge)
                score += _joint_scan_axis_edge_score(region_id, neighbor_id, graph, config, path_config)
    return score


def _joint_scan_axis_edge_score(
    left_region_id: str,
    right_region_id: str,
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> float:
    left_axis = _joint_region_best_scan_axis(left_region_id, graph, config, path_config)
    right_axis = _joint_region_best_scan_axis(right_region_id, graph, config, path_config)
    left_angle = _scan_axis_angle_rad(left_axis)
    right_angle = _scan_axis_angle_rad(right_axis)
    delta = abs((left_angle - right_angle) % math.pi)
    delta = min(delta, math.pi - delta)
    tolerance = math.radians(max(path_config.oriented_sweep_angle_tolerance_deg, 0.0))
    compatibility = 1.0 if delta <= tolerance else max(0.0, 1.0 - delta / (math.pi / 2.0))
    left_weight = float(graph.node_weights.get(left_region_id, 1.0))
    right_weight = float(graph.node_weights.get(right_region_id, 1.0))
    return compatibility * math.sqrt(max(min(left_weight, right_weight), 1e-9))


def _joint_region_best_scan_axis(
    region_id: str,
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> str:
    candidates = [pattern for pattern in graph.patterns.get(region_id, []) if pattern.feasible]
    if candidates:
        best = min(candidates, key=lambda pattern: (_pattern_sort_key(pattern, config, path_config), pattern.pattern_id))
        return best.scan_axis
    region = graph.regions.get(region_id)
    return region.preferred_axis if region is not None else "x"


def _joint_region_scan_axis_switch_penalty(
    previous_region_id: str,
    candidate_region_id: str,
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> float:
    return _scan_axis_switch_penalty_between_axes(
        _joint_region_best_scan_axis(previous_region_id, graph, config, path_config),
        _joint_region_best_scan_axis(candidate_region_id, graph, config, path_config),
        path_config,
    )


def _joint_two_opt_order_candidates(
    agent_id: int,
    region_ids: Sequence[str],
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    max_candidates: int,
) -> List[Tuple[str, List[str]]]:
    order = list(region_ids)
    if len(order) <= 3 or max_candidates <= 0:
        return []
    base_cost = _joint_center_route_cost(agent_id, order, graph, config, path_config)
    improvements: List[Tuple[float, int, int, List[str]]] = []
    for i in range(0, len(order) - 2):
        for j in range(i + 2, len(order) + 1):
            candidate = order[:i] + list(reversed(order[i:j])) + order[j:]
            if candidate == order:
                continue
            cost = _joint_center_route_cost(agent_id, candidate, graph, config, path_config)
            delta = base_cost - cost
            if delta > 1e-9:
                improvements.append((delta, i, j, candidate))
    improvements.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    return [
        (f"reorder_2opt:{agent_id}:{i}:{j}", candidate)
        for _, i, j, candidate in improvements[:max_candidates]
    ]


def _joint_center_route_cost(
    agent_id: int,
    order: Sequence[str],
    graph: RegionGraph,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> float:
    if not order:
        return 0.0
    if agent_id < len(config.fleet.initial_states_3dof):
        current_x = config.fleet.initial_states_3dof[agent_id].x
        current_y = config.fleet.initial_states_3dof[agent_id].y
        current_heading = config.fleet.initial_states_3dof[agent_id].psi
    else:
        first_center = graph.regions[order[0]].center
        current_x, current_y, current_heading = first_center[0], first_center[1], 0.0
    cost = 0.0
    previous_region_id: str | None = None
    for region_id in order:
        center = graph.regions[region_id].center
        dx = center[0] - current_x
        dy = center[1] - current_y
        distance = math.hypot(dx, dy)
        if distance > 1e-9:
            heading = math.atan2(dy, dx)
            cost += max(path_config.turn_angle_weight, 0.0) * abs(wrap_angle(heading - current_heading))
            current_heading = heading
        cost += distance
        if previous_region_id is not None:
            cost += _joint_region_scan_axis_switch_penalty(
                previous_region_id,
                region_id,
                graph,
                config,
                path_config,
            )
        previous_region_id = region_id
        current_x, current_y = center
    return cost


def _joint_boundary_regions(
    agent_regions: Dict[int, List[str]],
    graph: RegionGraph,
    source: int,
    target: int,
) -> List[str]:
    source_regions = list(agent_regions.get(source, []))
    target_set = set(agent_regions.get(target, []))
    if not source_regions:
        return []
    if not target_set:
        return sorted(source_regions, key=lambda region_id: -graph.node_weights.get(region_id, 0.0))[:4]
    boundary = [
        region_id
        for region_id in source_regions
        if any(neighbor in target_set for neighbor in graph.adjacency.get(region_id, []))
    ]
    if not boundary:
        target_centers = [graph.regions[region_id].center for region_id in target_set if region_id in graph.regions]
        if not target_centers:
            return []
        boundary = sorted(
            source_regions,
            key=lambda region_id: min(
                math.hypot(graph.regions[region_id].center[0] - cx, graph.regions[region_id].center[1] - cy)
                for cx, cy in target_centers
            ),
        )[:3]
    return sorted(boundary, key=lambda region_id: -graph.node_weights.get(region_id, 0.0))[:4]


def _joint_order_for_agent(
    agent_id: int,
    region_ids: Sequence[str],
    graph: RegionGraph,
    config: PlannerConfig,
    reverse: bool = False,
) -> List[str]:
    remaining = set(region_ids)
    if not remaining:
        return []
    if agent_id < len(config.fleet.initial_states_3dof):
        current = config.fleet.initial_states_3dof[agent_id].pose()
    else:
        first_center = graph.regions[next(iter(remaining))].center
        current = Pose2D(first_center[0], first_center[1], 0.0)
    order: List[str] = []
    while remaining:
        region_id = min(
            remaining,
            key=lambda item: (
                math.hypot(graph.regions[item].center[0] - current.x, graph.regions[item].center[1] - current.y),
                item,
            ),
        )
        order.append(region_id)
        center = graph.regions[region_id].center
        current = Pose2D(center[0], center[1], current.psi)
        remaining.remove(region_id)
    if reverse:
        order.reverse()
    return order


def _joint_assignment_connected(agent_regions: Dict[int, List[str]], graph: RegionGraph) -> bool:
    return all(graph_is_connected(graph, region_ids) for region_ids in agent_regions.values() if region_ids)


def _joint_assignment_signature(agent_regions: Dict[int, List[str]]) -> Tuple[Tuple[int, Tuple[str, ...]], ...]:
    return tuple((agent_id, tuple(region_ids)) for agent_id, region_ids in sorted(agent_regions.items()))


def _joint_region_loads(
    agent_regions: Dict[int, List[str]],
    patterns: Dict[str, List[RegionCoveragePattern]] | None = None,
    path_config: PathPlanningConfig | None = None,
    weights: Dict[str, float] | None = None,
) -> Dict[int, float]:
    loads: Dict[int, float] = {}
    for agent_id, region_ids in agent_regions.items():
        total = 0.0
        for region_id in region_ids:
            if weights is not None:
                total += float(weights.get(region_id, 0.0))
                continue
            candidates = list((patterns or {}).get(region_id, []))
            if candidates:
                best = min(candidates, key=lambda pattern: (pattern.estimated_time, pattern.total_length, pattern.pattern_id))
                total += best.estimated_time + 0.25 * best.total_length + (path_config.turn_angle_weight if path_config else 0.35) * best.turn_angle
            else:
                total += 1.0
        loads[agent_id] = total
    return loads


def _joint_imbalance_ratio(agent_regions: Dict[int, List[str]], loads: Dict[int, float]) -> float:
    active = [loads.get(agent_id, 0.0) for agent_id in agent_regions]
    avg = sum(active) / max(len(active), 1)
    if avg <= 1e-9:
        return 0.0
    return (max(active) - min(active)) / avg


def _joint_assignment_from_regions(agent_regions: Dict[int, List[str]], graph: RegionGraph) -> BalancedAssignment:
    loads = _joint_region_loads(agent_regions, weights=graph.node_weights)
    connected = {agent_id: graph_is_connected(graph, region_ids) for agent_id, region_ids in agent_regions.items()}
    imbalance = _joint_imbalance_ratio(agent_regions, loads)
    return BalancedAssignment(
        agent_regions={agent_id: list(region_ids) for agent_id, region_ids in agent_regions.items()},
        loads=loads,
        connected=connected,
        imbalance_ratio=imbalance,
        objective=max(loads.values(), default=0.0) + imbalance,
        diagnostics={"joint_assignment": "true"},
    )


def _joint_increment_reason(reasons: Dict[str, int], reason: str) -> None:
    reasons[reason] = reasons.get(reason, 0) + 1


def _cross_agent_crossing_count(agents: Dict[int, AgentPathPlan]) -> int:
    count = 0
    agent_items = sorted(agents.items())
    for left_idx, (left_agent, left_plan) in enumerate(agent_items):
        left_segments = [segment for segment in left_plan.segments if segment.kind != "cover"]
        for right_agent, right_plan in agent_items[left_idx + 1 :]:
            right_segments = [segment for segment in right_plan.segments if segment.kind != "cover"]
            for left_segment in left_segments:
                for right_segment in right_segments:
                    if _segments_cross(left_segment, right_segment):
                        count += 1
    return count


def _segments_cross(first: PathSegmentSpec, second: PathSegmentSpec) -> bool:
    first_points = [(waypoint.x, waypoint.y) for waypoint in first.waypoints]
    second_points = [(waypoint.x, waypoint.y) for waypoint in second.waypoints]
    for a0, a1 in zip(first_points[:-1], first_points[1:]):
        for b0, b1 in zip(second_points[:-1], second_points[1:]):
            if _line_segments_cross(a0, a1, b0, b1):
                return True
    return False


def _line_segments_cross(
    a0: Tuple[float, float],
    a1: Tuple[float, float],
    b0: Tuple[float, float],
    b1: Tuple[float, float],
) -> bool:
    if min(math.hypot(a0[0] - b0[0], a0[1] - b0[1]), math.hypot(a0[0] - b1[0], a0[1] - b1[1]), math.hypot(a1[0] - b0[0], a1[1] - b0[1]), math.hypot(a1[0] - b1[0], a1[1] - b1[1])) <= 1e-6:
        return False
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return orient(a0, a1, b0) * orient(a0, a1, b1) < 0.0 and orient(b0, b1, a0) * orient(b0, b1, a1) < 0.0


def _refine_global_routes(
    agents: Dict[int, AgentPathPlan],
    tours: Dict[int, SingleUsvTourPlan],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    baseline_coverage: float,
    agent_obstacle_fields: Dict[int, ObstacleField | None] | None = None,
) -> Dict[str, object]:
    original_agent_segments = {agent_id: copy.deepcopy(agent.segments) for agent_id, agent in agents.items()}
    original_tour_segments = {agent_id: copy.deepcopy(tour.segments) for agent_id, tour in tours.items()}
    rejected: Dict[str, int] = {}
    refined_count = 0
    merged_window_count = 0
    length_reduction = 0.0
    turn_reduction = 0.0
    iterations = max(int(path_config.route_refinement_iterations), 0)
    if iterations <= 0:
        return {
            "refined_connector_count": 0,
            "merged_noncover_window_count": 0,
            "turn_angle_reduction": 0.0,
            "length_reduction": 0.0,
            "refinement_rejected_reasons": {},
            "route_refinement_status": "iteration_budget_zero",
        }

    for _ in range(iterations):
        changed = False
        for agent_id, agent in sorted(agents.items()):
            agent_config = config.for_agent(agent_id) if config.agent_profiles else config
            agent_field = (
                agent_obstacle_fields.get(agent_id, obstacle_field)
                if agent_obstacle_fields is not None
                else obstacle_field
            )
            (
                window_segments,
                window_refined_count,
                window_merged_count,
                window_length_reduction,
                window_turn_reduction,
            ) = _refine_noncover_segment_windows(agent.segments, agent_config, path_config, agent_field, rejected)
            if window_refined_count:
                agent.segments = window_segments
                refined_count += window_refined_count
                merged_window_count += window_merged_count
                length_reduction += window_length_reduction
                turn_reduction += window_turn_reduction
                changed = True
            new_segments: List[PathSegmentSpec] = []
            for segment in agent.segments:
                if segment.kind == "cover" or len(segment.waypoints) < 2:
                    new_segments.append(segment)
                    continue
                alternatives = _route_refinement_alternatives(segment, agent_config, path_config, agent_field)
                best_segments = [segment]
                best_score = _route_refinement_score(best_segments, path_config)
                best_length = segment.length
                best_turn = _segment_heading_variation(segment)
                for candidate_segments, source in alternatives:
                    if not candidate_segments:
                        _joint_increment_reason(rejected, f"{source}:empty")
                        continue
                    if not validate_transition_sequence(
                        candidate_segments,
                        agent_config,
                        obstacle_field=agent_field,
                        retime=True,
                    ).valid:
                        _joint_increment_reason(rejected, f"{source}:invalid")
                        continue
                    candidate_length = sum(item.length for item in candidate_segments)
                    candidate_turn = _path_heading_variation(candidate_segments)
                    candidate_score = _route_refinement_score(candidate_segments, path_config)
                    improves = (
                        candidate_length + 1e-6 < best_length
                        or candidate_turn + 1e-6 < best_turn
                        or candidate_score + 1e-6 < best_score
                    )
                    if not improves:
                        _joint_increment_reason(rejected, f"{source}:not_improved")
                        continue
                    if candidate_score < best_score + 1e-6 or candidate_length <= best_length * 1.05:
                        best_segments = _mark_refined_segments(candidate_segments, segment, source)
                        best_score = candidate_score
                        best_length = candidate_length
                        best_turn = candidate_turn
                if best_segments != [segment]:
                    refined_count += 1
                    length_reduction += max(0.0, segment.length - best_length)
                    turn_reduction += max(0.0, _segment_heading_variation(segment) - best_turn)
                    changed = True
                new_segments.extend(best_segments)
            agent.segments = _retime_agent_segments(new_segments)
            if agent_id in tours:
                tours[agent_id].segments = agent.segments
        if not changed:
            break

    for agent_id, agent in agents.items():
        agent_config = config.for_agent(agent_id) if config.agent_profiles else config
        agent_field = (
            agent_obstacle_fields.get(agent_id, obstacle_field)
            if agent_obstacle_fields is not None
            else obstacle_field
        )
        agent.metrics = _agent_metrics(agent.segments, agent_config, agent_field)
        if agent_id in tours:
            _refresh_tour_from_segments(tours[agent_id], path_config)
    refined_coverage = evaluate_tour_coverage_state(
        config,
        list(tours.values()),
        resolution=path_config.residual_resolution,
        obstacle_field=obstacle_field,
        include_non_cover_segments=path_config.count_transit_coverage,
    )
    if refined_coverage.coverage_fraction + 1e-9 < baseline_coverage:
        for agent_id, original in original_agent_segments.items():
            agents[agent_id].segments = original
            agent_config = config.for_agent(agent_id) if config.agent_profiles else config
            agent_field = (
                agent_obstacle_fields.get(agent_id, obstacle_field)
                if agent_obstacle_fields is not None
                else obstacle_field
            )
            agents[agent_id].metrics = _agent_metrics(original, agent_config, agent_field)
        for agent_id, original in original_tour_segments.items():
            tours[agent_id].segments = original
            _refresh_tour_from_segments(tours[agent_id], path_config)
        _joint_increment_reason(rejected, "coverage_declined_reverted")
        return {
            "refined_connector_count": 0,
            "merged_noncover_window_count": 0,
            "turn_angle_reduction": 0.0,
            "length_reduction": 0.0,
            "refinement_rejected_reasons": dict(rejected),
            "route_refinement_status": "coverage_declined_reverted",
        }
    return {
        "refined_connector_count": refined_count,
        "merged_noncover_window_count": merged_window_count,
        "turn_angle_reduction": turn_reduction,
        "length_reduction": length_reduction,
        "refinement_rejected_reasons": dict(rejected),
        "route_refinement_status": "success" if refined_count else "no_improvement",
    }


def _refine_noncover_segment_windows(
    segments: Sequence[PathSegmentSpec],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    rejected: Dict[str, int],
) -> Tuple[List[PathSegmentSpec], int, int, float, float]:
    if len(segments) < 2:
        return list(segments), 0, 0, 0.0, 0.0
    refined: List[PathSegmentSpec] = []
    refined_count = 0
    merged_window_count = 0
    length_reduction = 0.0
    turn_reduction = 0.0
    idx = 0
    while idx < len(segments):
        segment = segments[idx]
        if segment.kind == "cover" or len(segment.waypoints) < 2:
            refined.append(segment)
            idx += 1
            continue
        best_window_size = 1
        best_segments: List[PathSegmentSpec] | None = None
        best_source = ""
        best_score_gain = 0.0
        best_length_reduction = 0.0
        best_turn_reduction = 0.0
        max_window_size = min(3, len(segments) - idx)
        for window_size in range(2, max_window_size + 1):
            window = list(segments[idx : idx + window_size])
            if any(item.kind == "cover" or len(item.waypoints) < 2 for item in window):
                break
            original_score = _route_refinement_score(window, path_config)
            original_length = sum(item.length for item in window)
            original_turn = _path_heading_variation(window)
            proxy = _route_refinement_window_proxy(window)
            for candidate_segments, source in _route_refinement_alternatives(proxy, config, path_config, obstacle_field):
                source = f"window_{source}"
                if not candidate_segments:
                    _joint_increment_reason(rejected, f"{source}:empty")
                    continue
                if not validate_transition_sequence(candidate_segments, config, obstacle_field=obstacle_field, retime=True).valid:
                    _joint_increment_reason(rejected, f"{source}:invalid")
                    continue
                candidate_score = _route_refinement_score(candidate_segments, path_config)
                candidate_length = sum(item.length for item in candidate_segments)
                candidate_turn = _path_heading_variation(candidate_segments)
                score_gain = original_score - candidate_score
                if score_gain <= 1e-6:
                    _joint_increment_reason(rejected, f"{source}:not_improved")
                    continue
                if candidate_length > original_length * 1.05 and candidate_turn >= original_turn:
                    _joint_increment_reason(rejected, f"{source}:too_long")
                    continue
                if score_gain > best_score_gain + 1e-9:
                    best_window_size = window_size
                    best_segments = candidate_segments
                    best_source = source
                    best_score_gain = score_gain
                    best_length_reduction = max(0.0, original_length - candidate_length)
                    best_turn_reduction = max(0.0, original_turn - candidate_turn)
        if best_segments is not None:
            window = list(segments[idx : idx + best_window_size])
            marked = _mark_refined_window_segments(best_segments, window, best_source)
            refined.extend(marked)
            refined_count += 1
            merged_window_count += 1
            length_reduction += best_length_reduction
            turn_reduction += best_turn_reduction
            idx += best_window_size
            continue
        refined.append(segment)
        idx += 1
    return refined, refined_count, merged_window_count, length_reduction, turn_reduction


def _route_refinement_window_proxy(window: Sequence[PathSegmentSpec]) -> PathSegmentSpec:
    first = window[0]
    last = window[-1]
    kind = "turn" if all(segment.kind == "turn" for segment in window) else "transit"
    segment_ids = ",".join(segment.segment_id for segment in window)
    start = first.waypoints[0]
    end = last.waypoints[-1]
    return PathSegmentSpec(
        segment_id=f"{first.segment_id}_to_{last.segment_id}_merged",
        kind=kind,
        source_algorithm="route_refinement",
        waypoints=[copy.deepcopy(start), copy.deepcopy(end)],
        length=sum(segment.length for segment in window),
        path_source="route_refinement_window",
        metadata={
            "merged_original_segment_ids": segment_ids,
            "route_refinement_window_size": str(len(window)),
        },
    )


def _route_refinement_alternatives(
    segment: PathSegmentSpec,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> List[Tuple[List[PathSegmentSpec], str]]:
    start_wp = segment.waypoints[0]
    end_wp = segment.waypoints[-1]
    start = Pose2D(start_wp.x, start_wp.y, start_wp.psi)
    end = Pose2D(end_wp.x, end_wp.y, end_wp.psi)
    start_time = start_wp.time or 0.0
    sample_count = max(24, len(segment.waypoints))
    alternatives: List[Tuple[List[PathSegmentSpec], str]] = []
    for use_bezier, source in ((False, "dubins_refinement"), (True, "bezier_refinement")):
        candidate = build_transition_segment(
            segment_id=segment.segment_id,
            start=start,
            end=end,
            start_time=start_time,
            config=config,
            kind=segment.kind,
            sample_count=sample_count,
            use_bezier=use_bezier,
        )
        alternatives.append(([candidate], source))
    if obstacle_field is not None:
        alternatives.append(
            (
                build_obstacle_aware_transition_segments(
                    segment_id=segment.segment_id,
                    start=start,
                    end=end,
                    start_time=start_time,
                    config=config,
                    path_config=path_config,
                    obstacle_field=obstacle_field,
                    kind=segment.kind,
                    sample_count=sample_count,
                ),
                "obstacle_aware_refinement",
            )
        )
    return alternatives


def _route_refinement_score(segments: Sequence[PathSegmentSpec], path_config: PathPlanningConfig) -> float:
    return (
        sum(segment.length for segment in segments)
        + max(path_config.global_turn_angle_weight, 0.0)
        * _path_heading_variation(segments)
        + max(path_config.turn_count_weight, 0.0) * _path_turn_count(segments)
    )


def _mark_refined_segments(
    candidate_segments: Sequence[PathSegmentSpec],
    original: PathSegmentSpec,
    source: str,
) -> List[PathSegmentSpec]:
    refined: List[PathSegmentSpec] = []
    for idx, segment in enumerate(copy.deepcopy(list(candidate_segments))):
        segment.segment_id = original.segment_id if len(candidate_segments) == 1 else f"{original.segment_id}_refined_{idx}"
        segment.metadata.update(original.metadata)
        segment.metadata["route_refined"] = "true"
        segment.metadata["route_refinement_source"] = source
        refined.append(segment)
    return refined


def _mark_refined_window_segments(
    candidate_segments: Sequence[PathSegmentSpec],
    window: Sequence[PathSegmentSpec],
    source: str,
) -> List[PathSegmentSpec]:
    proxy = _route_refinement_window_proxy(window)
    refined = _mark_refined_segments(candidate_segments, proxy, source)
    segment_ids = ",".join(segment.segment_id for segment in window)
    for segment in refined:
        segment.metadata["merged_original_segment_ids"] = segment_ids
        segment.metadata["route_refinement_window_size"] = str(len(window))
    return refined


def _retime_agent_segments(segments: Sequence[PathSegmentSpec]) -> List[PathSegmentSpec]:
    current_time = 0.0
    retimed: List[PathSegmentSpec] = []
    for segment in copy.deepcopy(list(segments)):
        if not segment.waypoints:
            retimed.append(segment)
            continue
        original_start = segment.waypoints[0].time or 0.0
        segment.waypoints = [
            replace(waypoint, time=current_time + ((waypoint.time or original_start) - original_start))
            for waypoint in segment.waypoints
        ]
        current_time = _segment_end_time(segment)
        retimed.append(segment)
    return retimed


def _refresh_tour_from_segments(tour: SingleUsvTourPlan, path_config: PathPlanningConfig) -> None:
    tour.total_length = sum(segment.length for segment in tour.segments)
    tour.total_turn_angle = _path_heading_variation(tour.segments)
    tour.estimated_time = max((_segment_end_time(segment) for segment in tour.segments), default=0.0)
    tour.objective = tour.total_length + max(path_config.turn_angle_weight, 0.0) * tour.total_turn_angle + tour.estimated_time


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
    effective_path_config = _internal_sweep_execution_path_config(pattern, path_config)
    segments, reason = _build_internal_sweep_segments(
        pattern,
        config,
        effective_path_config,
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


def _parse_json_diagnostic(value: object, default: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _agent_planning_profile_report(config: PlannerConfig) -> Dict[str, Dict[str, object]]:
    count = config.fleet.num_agents or len(config.fleet.initial_states_3dof)
    result: Dict[str, Dict[str, object]] = {}
    for agent_id in range(count):
        profile = config.profile_for_agent(agent_id)
        state = config.fleet.initial_states_3dof[agent_id]
        result[str(agent_id)] = {
            "initial_state": {"x": state.x, "y": state.y, "psi": state.psi},
            "coverage_length": profile.coverage_length,
            "coverage_width": profile.coverage_width,
            "overlap_ratio": profile.overlap_ratio,
            "effective_strip_spacing": profile.effective_strip_spacing,
            "vehicle_length": profile.vehicle_length,
            "vehicle_width": profile.vehicle_width,
            "min_turn_radius": profile.min_turn_radius,
            "cruise_speed": profile.cruise_speed,
            "cover_speed": profile.cover_speed,
            "turn_speed_max": profile.turn_speed_max,
            "yaw_rate_limit": profile.yaw_rate_limit,
            "max_thrust": profile.max_thrust,
            "max_yaw_moment": profile.max_yaw_moment,
            "max_mission_time": profile.max_mission_time,
            "fingerprint": profile.fingerprint,
        }
    return result


def _internal_sweep_execution_path_config(
    pattern: RegionCoveragePattern,
    path_config: PathPlanningConfig,
) -> PathPlanningConfig:
    if (
        pattern.metadata.get("open_chain_flexible_exit_variant") == "true"
        and not bool(getattr(path_config, "open_chain_allow_flexible_exit", False))
    ):
        return replace(path_config, open_chain_allow_flexible_exit=True)
    return path_config


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
        effective_path_config = _internal_sweep_execution_path_config(pattern, path_config)
        segments, reason = _build_internal_sweep_segments(
            pattern,
            config,
            effective_path_config,
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
            dubins_turn = build_transition_segment(
                segment_id=f"{segment_prefix}_{coverage_pass.pass_id}_uturn_dubins",
                start=coverage_pass.end_pose,
                end=next_pass.start_pose,
                start_time=current_time,
                config=config,
                kind="turn",
                sample_count=48,
                use_bezier=False,
            )
            dubins_reasons = path_segment_invalid_reasons(dubins_turn, config, obstacle_field)
            if not dubins_reasons:
                dubins_report = validate_transition_dynamics(
                    dubins_turn,
                    config,
                    obstacle_field=obstacle_field,
                    retime=True,
                )
                if dubins_report.valid:
                    dubins_turn.metadata.update(
                        {
                            "region_id": pattern.region_id,
                            "internal_uturn": "true",
                            "uturn_repair": "dubins_after_bezier",
                            "kinematic_feasible": "true",
                        }
                    )
                    segments.append(dubins_turn)
                    current_time = _segment_end_time(dubins_turn)
                    if stats is not None:
                        stats["uturn_bezier_fail_dubins_success_count"] = int(
                            stats.get("uturn_bezier_fail_dubins_success_count", 0) or 0
                        ) + 1
                    if use_cache:
                        uturn_cache[cache_key] = (True, "")
                    continue
            allow_validation_repair = not (
                segment_prefix.startswith("validate")
                and _large_map_mode_enabled(config, path_config)
                and not path_config.large_map_validate_internal_uturn_repair
            )
            if not allow_validation_repair:
                reason = f"uturn_invalid:{','.join(reasons)}"
                if use_cache:
                    uturn_cache[cache_key] = (False, reason)
                return [], reason
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
        lightweight=True,
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


def _open_chain_entry_exit(chain: OpenSweepChain, reverse: bool) -> Tuple[Pose2D, Pose2D]:
    return (
        chain.reverse_entry_pose if reverse else chain.entry_pose,
        chain.reverse_exit_pose if reverse else chain.exit_pose,
    )


def _pose_anchor_matches(first: Pose2D, second: Pose2D, config: PlannerConfig) -> bool:
    xy_tol = max(1e-5, config.footprint.width_wf * 1e-4)
    heading_tol = 1e-3
    return math.hypot(first.x - second.x, first.y - second.y) <= xy_tol and abs(wrap_angle(first.psi - second.psi)) <= heading_tol


def _open_chain_candidate_limit(
    candidate_count: int,
    path_config: PathPlanningConfig,
    config: PlannerConfig,
    anchored: bool = False,
) -> int:
    if candidate_count <= 0:
        return 0
    if anchored:
        return min(candidate_count, max(4, int(path_config.open_chain_tsp_beam_width)))
    if _large_map_mode_enabled(config, path_config):
        return min(candidate_count, max(2, min(int(path_config.open_chain_tsp_beam_width), 4)))
    if path_config.open_chain_tsp_beam_width > 0:
        return min(candidate_count, max(2, int(path_config.open_chain_tsp_beam_width)))
    return candidate_count


def _build_open_chain_execution_choice(
    pattern: RegionCoveragePattern,
    chain: OpenSweepChain,
    reverse: bool,
    current_pose: Pose2D,
    current_time: float,
    serial: int,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    segment_prefix: str,
) -> Tuple[List[PathSegmentSpec], List[PathSegmentSpec], Pose2D, str]:
    connector = _build_open_chain_connector(
        segment_id=f"{segment_prefix}_open_chain_connector_{serial}_{chain.chain_id}_{'rev' if reverse else 'fwd'}",
        start=current_pose,
        end=_open_chain_entry_exit(chain, reverse)[0],
        start_time=current_time,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
    )
    if connector is None:
        return [], [], _open_chain_entry_exit(chain, reverse)[1], "connector_failed"
    connector_end_time = _segment_end_time(connector[-1]) if connector else current_time
    if not reverse and chain.internal_segments:
        chain_segments = _retime_segment_templates(chain.internal_segments, connector_end_time)
        chain_reason = ""
    else:
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
        return connector, [], _open_chain_entry_exit(chain, reverse)[1], chain_reason
    return connector, chain_segments, _open_chain_entry_exit(chain, reverse)[1], ""


def _annotate_open_chain_execution_segments(
    connector: Sequence[PathSegmentSpec],
    chain_segments: Sequence[PathSegmentSpec],
    pattern: RegionCoveragePattern,
    chain: OpenSweepChain,
    reverse: bool,
) -> None:
    for segment in connector:
        segment.metadata.update(
            {
                "open_chain_connector": "true",
                "region_id": pattern.region_id,
                "pattern_id": pattern.pattern_id,
                "open_chain_to": chain.chain_id,
                "chain_order_mode": chain.metadata.get("chain_order_mode", ""),
                "turn_stride": chain.metadata.get("turn_stride", ""),
            }
        )
    for segment in chain_segments:
        segment.metadata.update(
            {
                "open_chain_mode": "true",
                "open_chain_id": chain.chain_id,
                "open_chain_direction": "reverse" if reverse else "forward",
                "region_id": pattern.region_id,
                "pattern_id": pattern.pattern_id,
                "chain_order_mode": chain.metadata.get("chain_order_mode", ""),
                "turn_stride": chain.metadata.get("turn_stride", ""),
            }
        )


def _assemble_open_chains_endpoint_anchored(
    pattern: RegionCoveragePattern,
    chains: Sequence[OpenSweepChain],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    start_time: float,
    segment_prefix: str,
) -> Tuple[List[PathSegmentSpec], str, List[str]]:
    final_candidates: List[Tuple[float, str, bool, OpenSweepChain]] = []
    for chain in chains:
        for reverse in (False, True):
            chain_entry, chain_exit = _open_chain_entry_exit(chain, reverse)
            if not _pose_anchor_matches(chain_exit, pattern.exit_pose, config):
                continue
            final_candidates.append(
                (
                    math.hypot(chain_entry.x - pattern.entry_pose.x, chain_entry.y - pattern.entry_pose.y),
                    chain.chain_id,
                    reverse,
                    chain,
                )
            )
    if not final_candidates:
        return [], "open_chain_endpoint_anchor_unavailable", []
    final_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    failure_reasons: List[str] = []
    max_final_candidates = min(len(final_candidates), max(2, int(path_config.open_chain_tsp_beam_width)))
    for _, _, final_reverse, final_chain in final_candidates[:max_final_candidates]:
        remaining = [chain for chain in chains if chain.chain_id != final_chain.chain_id]
        current_pose = pattern.entry_pose
        current_time = start_time
        segments: List[PathSegmentSpec] = []
        connected: List[str] = []
        serial = 0
        failed = False
        while remaining:
            chain_entry_candidates: List[Tuple[float, str, bool, OpenSweepChain, Pose2D]] = []
            for chain in remaining:
                for reverse in (False, True):
                    chain_entry, _ = _open_chain_entry_exit(chain, reverse)
                    chain_entry_candidates.append(
                        (
                            math.hypot(chain_entry.x - current_pose.x, chain_entry.y - current_pose.y),
                            chain.chain_id,
                            reverse,
                            chain,
                            chain_entry,
                        )
                    )
            chain_entry_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            candidate_limit = _open_chain_candidate_limit(
                len(chain_entry_candidates),
                path_config,
                config,
                anchored=True,
            )
            choices: List[Tuple[float, OpenSweepChain, bool, List[PathSegmentSpec], List[PathSegmentSpec], Pose2D]] = []
            for _, _, reverse, chain, _ in chain_entry_candidates[:candidate_limit]:
                connector, chain_segments, chain_exit, reason = _build_open_chain_execution_choice(
                    pattern,
                    chain,
                    reverse,
                    current_pose,
                    current_time,
                    serial,
                    config,
                    path_config,
                    obstacle_field,
                    segment_prefix,
                )
                if reason:
                    failure_reasons.append(f"{chain.chain_id}:{reason}")
                    continue
                score = (
                    path_config.open_chain_connector_penalty_weight * sum(segment.length for segment in connector)
                    + _segment_duration_total(connector)
                    + _segment_duration_total(chain_segments)
                    - path_config.open_chain_coverage_reward_weight * chain.coverage_length
                )
                choices.append((score, chain, reverse, connector, chain_segments, chain_exit))
            if not choices:
                failed = True
                break
            choices.sort(key=lambda item: (item[0], item[1].chain_id, item[2]))
            _, selected, reverse, connector, chain_segments, chain_exit = choices[0]
            _annotate_open_chain_execution_segments(connector, chain_segments, pattern, selected, reverse)
            segments.extend(connector)
            segments.extend(chain_segments)
            connected.append(selected.chain_id)
            remaining = [item for item in remaining if item.chain_id != selected.chain_id]
            current_pose = chain_exit
            current_time = _segment_end_time(chain_segments[-1])
            serial += len(connector) + len(chain_segments)
        if failed:
            continue
        connector, chain_segments, chain_exit, reason = _build_open_chain_execution_choice(
            pattern,
            final_chain,
            final_reverse,
            current_pose,
            current_time,
            serial,
            config,
            path_config,
            obstacle_field,
            segment_prefix,
        )
        if reason:
            failure_reasons.append(f"{final_chain.chain_id}:{reason}")
            continue
        if not _pose_anchor_matches(chain_exit, pattern.exit_pose, config):
            failure_reasons.append(f"{final_chain.chain_id}:exit_anchor_mismatch")
            continue
        _annotate_open_chain_execution_segments(connector, chain_segments, pattern, final_chain, final_reverse)
        segments.extend(connector)
        segments.extend(chain_segments)
        connected.append(final_chain.chain_id)
        if not _segments_strictly_valid(segments, config, obstacle_field):
            failure_reasons.append("endpoint_anchored_sequence_dynamic_validation_failed")
            continue
        for segment in segments:
            if segment.metadata.get("open_chain_connector") == "true":
                segment.metadata["open_chain_endpoint_anchored"] = "true"
        return [segment for segment in segments if segment.length > 1e-9], "", connected
    return [], ",".join(failure_reasons[:6]) or "open_chain_endpoint_anchored_failed", []


def _assemble_open_chains_greedy(
    pattern: RegionCoveragePattern,
    chains: Sequence[OpenSweepChain],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    start_time: float,
    segment_prefix: str,
) -> Tuple[List[PathSegmentSpec], str, List[str]]:
    anchored_segments, anchored_reason, anchored_connected = _assemble_open_chains_endpoint_anchored(
        pattern,
        chains,
        config,
        path_config,
        obstacle_field,
        start_time=start_time,
        segment_prefix=segment_prefix,
    )
    if anchored_segments:
        return anchored_segments, "", anchored_connected
    remaining = list(chains)
    current_pose = pattern.entry_pose
    current_time = start_time
    segments: List[PathSegmentSpec] = []
    connected: List[str] = []
    serial = 0
    failure_reasons: List[str] = [anchored_reason] if anchored_reason else []

    while remaining:
        choices: List[Tuple[float, OpenSweepChain, bool, List[PathSegmentSpec], List[PathSegmentSpec], Pose2D]] = []
        chain_entry_candidates: List[Tuple[float, str, bool, OpenSweepChain, Pose2D]] = []
        for chain in remaining:
            for reverse in (False, True):
                chain_entry = chain.reverse_entry_pose if reverse else chain.entry_pose
                chain_entry_candidates.append(
                    (
                        math.hypot(chain_entry.x - current_pose.x, chain_entry.y - current_pose.y),
                        chain.chain_id,
                        reverse,
                        chain,
                        chain_entry,
                    )
                )
        chain_entry_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        candidate_limit = _open_chain_candidate_limit(
            len(chain_entry_candidates),
            path_config,
            config,
            anchored=False,
        )
        for _, _, reverse, chain, chain_entry in chain_entry_candidates[:candidate_limit]:
                connector, chain_segments, chain_exit, chain_reason = _build_open_chain_execution_choice(
                    pattern,
                    chain,
                    reverse,
                    current_pose,
                    current_time,
                    serial,
                    config,
                    path_config,
                    obstacle_field,
                    segment_prefix,
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
                choices.append((score, chain, reverse, connector, chain_segments, chain_exit))
        if not choices:
            break
        choices.sort(key=lambda item: (item[0], item[1].chain_id, item[2]))
        _, selected, reverse, connector, chain_segments, chain_exit = choices[0]
        _annotate_open_chain_execution_segments(connector, chain_segments, pattern, selected, reverse)
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
        if bool(getattr(path_config, "open_chain_allow_flexible_exit", True)):
            if not _segments_strictly_valid(segments, config, obstacle_field):
                return [], "open_chain_sequence_dynamic_validation_failed", connected
            old_exit = pattern.exit_pose
            pattern.exit_pose = current_pose
            pattern.metadata.update(
                {
                    "open_chain_flexible_exit": "true",
                    "open_chain_exit_connector_failed": "true",
                    "open_chain_nominal_exit": f"{old_exit.x:.6f},{old_exit.y:.6f},{old_exit.psi:.6f}",
                    "open_chain_actual_exit": f"{current_pose.x:.6f},{current_pose.y:.6f},{current_pose.psi:.6f}",
                }
            )
            for segment in segments:
                segment.metadata["open_chain_flexible_exit"] = "true"
            return [segment for segment in segments if segment.length > 1e-9], "", connected
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
    if _segments_strictly_valid([direct], config, obstacle_field):
        return [direct] if direct.length > 1e-9 else []
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
    merged_region_ids = {
        str(item[region_index])
        for item in ordered_candidates
        if _pattern_needs_connector_variant_diversity(item[-1])
    }
    merged_extra = len(merged_region_ids) * max(0, min(per_region_limit, 6) - 3)
    target = min(
        len(ordered_candidates),
        max(max(branch_limit, 1), len(region_ids) * min(per_region_limit, 3) + merged_extra),
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


def _scan_axis_switch_penalty(
    previous_pattern: RegionCoveragePattern | None,
    candidate_pattern: RegionCoveragePattern,
    path_config: PathPlanningConfig,
) -> float:
    if previous_pattern is None:
        return 0.0
    return _scan_axis_switch_penalty_between_axes(previous_pattern.scan_axis, candidate_pattern.scan_axis, path_config)


def _scan_axis_switch_penalty_between_axes(
    previous_axis: str,
    candidate_axis: str,
    path_config: PathPlanningConfig,
) -> float:
    previous_angle = _scan_axis_angle_rad(previous_axis)
    candidate_angle = _scan_axis_angle_rad(candidate_axis)
    delta = abs(wrap_angle(candidate_angle - previous_angle))
    delta = min(delta, abs(math.pi - delta))
    tolerance = math.radians(max(float(path_config.oriented_sweep_angle_tolerance_deg), 0.0))
    if delta <= tolerance + 1e-9:
        return 0.0
    normalized = delta / max(math.pi / 2.0, 1e-9)
    return (
        max(path_config.turn_count_weight, 0.0) * (1.0 + normalized)
        + max(path_config.turn_angle_weight, 0.0) * delta
    )


def _scan_axis_angle_rad(scan_axis: str) -> float:
    if scan_axis == "x":
        return 0.0
    if scan_axis == "y":
        return math.pi / 2.0
    if scan_axis.startswith("theta:"):
        try:
            return float(scan_axis.split(":", 1)[1]) % math.pi
        except ValueError:
            return 0.0
    return 0.0


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
    connector_cache: Dict[Tuple[object, ...], Tuple[List[PathSegmentSpec] | None, List[Dict[str, object]]]] = {}
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
        "connector_noncover_repeat_length": 0.0,
        "connector_noncover_repeat_penalty": 0.0,
        "connector_length": 0.0,
        "connector_turn_angle": 0.0,
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
            previous_pattern = None
            if state["final_order"]:
                previous_region_id = list(state["final_order"])[-1]
                previous_pattern = dict(state["selected_patterns"]).get(previous_region_id)
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
                            + _scan_axis_switch_penalty(previous_pattern, pattern, path_config)
                            + _turn_clearance_penalty(pattern.entry_pose, config)
                            + _turn_clearance_penalty(pattern.exit_pose, config)
                            - coverage_reward,
                            region_id,
                            pattern,
                        )
                    )
            ordered_candidates.sort(key=lambda item: (item[0], item[1], item[2].pattern_id))
            max_candidate_pattern_limit = max(
                [_connector_pattern_limit(path_config)]
                + [
                    _connector_pattern_limit_for_region(region_id, patterns.get(region_id, []), path_config)
                    for region_id in remaining
                ]
            )
            candidate_slice = _prioritized_candidate_slice(
                ordered_candidates,
                branch_limit,
                max_candidate_pattern_limit,
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
                connector_repeat_score = RepeatOverlapScore(0.0, 0.0, 0.0, 0, 0)
                connector_non_cover_segments = _non_cover_segments(connector)
                existing_non_cover_segments = _non_cover_segments(list(state["segments"]))
                if path_config.enable_main_repeat_path_penalty:
                    repeat_weight = (
                        max(path_config.main_repeat_path_penalty_weight, 0.0)
                        * max(path_config.connector_noncover_repeat_penalty_multiplier, 0.0)
                    )
                    repeat_score = score_repeat_overlap(
                        connector_non_cover_segments,
                        existing_non_cover_segments,
                        path_config,
                        penalty_weight=repeat_weight,
                        annotate=False,
                    )
                    connector_repeat_score = score_repeat_overlap(
                        connector_non_cover_segments,
                        existing_non_cover_segments,
                        path_config,
                        penalty_weight=repeat_weight,
                        annotate=False,
                    )
                cross_agent_score = score_cross_agent_ownership_overlap(
                    connector_non_cover_segments,
                    agent_id,
                    ownership_map,
                    path_config,
                    config=config,
                    annotate=False,
                )
                if cross_agent_score.overlap_length <= 1e-9:
                    state_has_zero_cross_agent_overlap = True
                internal_repeat_penalty = _metadata_float(candidate_pattern.metadata, "internal_repeat_penalty", 0.0)
                pattern_quality_penalty = _pattern_quality_penalty(candidate_pattern, path_config)
                scan_axis_switch_penalty = _scan_axis_switch_penalty(previous_pattern, candidate_pattern, path_config)
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
                connector_components = _connector_score_components(
                    connector,
                    connector_repeat_score,
                    candidate_pattern,
                    path_config,
                    coverage_deficit,
                )
                step_score = (
                    dynamic_edge_cost(connector, config)
                    + _connector_economy_penalty(connector_components, path_config, include_repeat=False)
                    + candidate_pattern.estimated_time
                    + path_config.coverage_priority_weight * coverage_deficit
                    + pattern_quality_penalty
                    + scan_axis_switch_penalty
                    + repeat_score.penalty
                    + cross_agent_score.penalty
                    + _turn_clearance_penalty(candidate_pattern.exit_pose, config)
                )
                _annotate_connector_score_components(connector, connector_components, path_config)
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
                        "connector_noncover_repeat_length": float(state["connector_noncover_repeat_length"])
                        + connector_repeat_score.overlap_length,
                        "connector_noncover_repeat_penalty": float(state["connector_noncover_repeat_penalty"])
                        + connector_repeat_score.penalty,
                        "connector_length": float(state["connector_length"]) + connector_components["connector_length"],
                        "connector_turn_angle": float(state["connector_turn_angle"]) + connector_components["connector_turn_angle"],
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
        "connector_noncover_repeat_length": float(chosen_state.get("connector_noncover_repeat_length", 0.0)),
        "connector_noncover_repeat_penalty": float(chosen_state.get("connector_noncover_repeat_penalty", 0.0)),
        "connector_length": float(chosen_state.get("connector_length", 0.0)),
        "connector_turn_angle": float(chosen_state.get("connector_turn_angle", 0.0)),
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
    _restart_depth: int = 0,
    _initial_forbidden_region_ids: set[str] | None = None,
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
    connector_noncover_repeat_length = 0.0
    connector_noncover_repeat_penalty = 0.0
    connector_length_total = 0.0
    connector_turn_angle_total = 0.0
    cross_agent_overlap = 0.0
    cross_agent_penalty = 0.0
    connector_cache: Dict[Tuple[object, ...], Tuple[List[PathSegmentSpec] | None, List[Dict[str, object]]]] = {}
    sweep_segment_cache: Dict[str, Tuple[List[PathSegmentSpec], str]] = {
        key: (copy.deepcopy(value[0]), value[1])
        for key, value in (sweep_segment_templates or {}).items()
    }
    reachability_probe_count = 0
    reachability_probe_success_count = 0
    reachability_probe_coverage_length = 0.0
    dead_end_avoidance_count = 0
    tsp_started = time.perf_counter()
    agent_time_budget = max(float(path_config.large_map_tsp_agent_time_budget_sec), 0.0)
    step_time_budget = max(float(path_config.large_map_tsp_step_time_budget_sec), 0.0)
    max_step_attempts = max(int(path_config.large_map_tsp_max_candidate_attempts_per_step), 0)
    obstacle_aware_retry_limit_config = max(int(path_config.large_map_tsp_obstacle_aware_retry_limit), 0)
    max_obstacle_aware_attempts_per_step = max(
        int(path_config.large_map_tsp_max_obstacle_aware_attempts_per_step),
        0,
    )
    max_obstacle_aware_attempts_per_agent = max(
        int(path_config.large_map_tsp_max_obstacle_aware_attempts_per_agent),
        0,
    )
    obstacle_aware_max_transition_length = max(
        float(path_config.large_map_tsp_obstacle_aware_max_transition_length),
        0.0,
    )
    map_span = max(float(config.mission.area_length_x), float(config.mission.area_length_y))
    initial_obstacle_aware_max_transition_length = obstacle_aware_max_transition_length
    if obstacle_aware_max_transition_length > 0.0:
        initial_obstacle_aware_max_transition_length = min(
            max(1.75 * obstacle_aware_max_transition_length, 0.85 * map_span),
            0.95 * map_span,
        )
    enable_lookahead_probe = bool(path_config.large_map_tsp_enable_lookahead_probe)
    require_cheap_connector_probe = bool(path_config.large_map_tsp_require_cheap_connector_probe)
    cheap_probe_collision_only = bool(path_config.large_map_tsp_cheap_probe_collision_only)
    obstacle_aware_attempt_count = 0
    obstacle_aware_filtered_count = 0
    budget_exhausted = False
    budget_reason = ""
    skipped_region_reasons: Dict[str, str] = {}
    connector_failure_reasons: Dict[str, str] = {}
    all_connector_failure_reasons: Dict[str, set[str]] = {}
    deferred_region_counts: Dict[str, int] = {}
    deferred_region_count = 0
    deferred_initial_anchor_count = 0
    failed_internal_pattern_ids: set[str] = set()
    initial_forbidden_region_ids = set(_initial_forbidden_region_ids or set())

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
        if agent_time_budget > 0.0 and time.perf_counter() - tsp_started >= agent_time_budget:
            budget_exhausted = True
            budget_reason = "large_map_tsp_agent_time_budget_exhausted"
            if path_config.monitor_stages:
                print(
                    json.dumps(
                        {
                            "stage": "agent_region_tsp_budget",
                            "agent_id": agent_id,
                            "reason": budget_reason,
                            "visited_region_count": len(final_order),
                            "remaining_region_count": len(remaining),
                            "elapsed_sec": round(time.perf_counter() - tsp_started, 3),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            for region_id in list(remaining):
                skipped_region_reasons.setdefault(region_id, budget_reason)
                connector_failure_reasons.setdefault(region_id, budget_reason)
                _record_connector_failure(all_connector_failure_reasons, region_id, budget_reason)
                infeasible_edges.append({"agent_id": agent_id, "region_id": region_id, "reason": budget_reason})
            remaining.clear()
            break
        ordered_candidates: List[Tuple[float, int, str, RegionCoveragePattern]] = []
        previous_pattern = selected_patterns.get(final_order[-1]) if final_order else None
        max_candidate_pattern_limit = pattern_limit
        for region_id in remaining:
            if not final_order and region_id in initial_forbidden_region_ids:
                continue
            candidates = [
                pattern
                for pattern in patterns.get(region_id, [])
                if pattern.pattern_id not in failed_internal_pattern_ids
            ]
            if not candidates:
                continue
            region_pattern_limit = _connector_pattern_limit_for_region(region_id, candidates, path_config)
            max_candidate_pattern_limit = max(max_candidate_pattern_limit, region_pattern_limit)
            candidates.sort(key=lambda pattern: (_pattern_sort_key(pattern, config, path_config), pattern.pattern_id))
            for pattern in candidates[:region_pattern_limit]:
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
                        + _scan_axis_switch_penalty(previous_pattern, pattern, path_config)
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
            int(path_config.region_tsp_branch_limit) * max(1, min(max_candidate_pattern_limit, 4)),
        )
        candidate_slice = _prioritized_candidate_slice(
            ordered_candidates,
            candidate_window,
            max_candidate_pattern_limit,
            bool(path_config.prioritize_region_execution),
            region_index=2,
        )
        initial_entry_step = len(final_order) == 0
        if initial_entry_step:
            candidate_slice.sort(
                key=lambda item: (
                    _transition_length(current_pose, item[3].entry_pose, config),
                    int(_pattern_needs_connector_variant_diversity(item[3])),
                    item[0],
                    item[1],
                    item[2],
                    item[3].pattern_id,
                )
            )
        obstacle_aware_retry_limit = min(
            obstacle_aware_retry_limit_config,
            max(2, min(8, int(path_config.region_tsp_branch_limit) // 2)),
        )
        if initial_entry_step and obstacle_aware_retry_limit_config > 0:
            obstacle_aware_retry_limit = min(
                max(obstacle_aware_retry_limit, 4),
                max(4, min(8, int(path_config.region_tsp_branch_limit) // 2)),
            )
        step_obstacle_aware_attempt_limit = max_obstacle_aware_attempts_per_step
        if (
            initial_entry_step
            and step_obstacle_aware_attempt_limit > 0
            and obstacle_aware_retry_limit > 0
        ):
            step_obstacle_aware_attempt_limit = max(
                step_obstacle_aware_attempt_limit,
                min(4, obstacle_aware_retry_limit),
            )
        effective_max_obstacle_aware_attempts_per_agent = max_obstacle_aware_attempts_per_agent
        if initial_entry_step and effective_max_obstacle_aware_attempts_per_agent > 0:
            effective_max_obstacle_aware_attempts_per_agent = max(
                effective_max_obstacle_aware_attempts_per_agent,
                24,
            )
        effective_obstacle_aware_max_transition_length = (
            initial_obstacle_aware_max_transition_length
            if initial_entry_step
            else obstacle_aware_max_transition_length
        )
        attempted_region_ids: List[str] = []
        step_started = time.perf_counter()
        step_budget_exhausted = False
        step_attempt_limit_exhausted = False
        step_attempt_count = 0
        step_obstacle_aware_attempt_count = 0
        if path_config.monitor_stages:
            print(
                json.dumps(
                    {
                        "stage": "agent_region_tsp_step_start",
                        "agent_id": agent_id,
                        "visited_region_count": len(final_order),
                        "remaining_region_count": len(remaining),
                        "candidate_slice_count": len(candidate_slice),
                        "candidate_window": candidate_window,
                        "max_step_attempts": max_step_attempts,
                        "step_time_budget_sec": step_time_budget,
                        "obstacle_aware_retry_limit": obstacle_aware_retry_limit,
                        "obstacle_aware_step_attempt_limit": step_obstacle_aware_attempt_limit,
                        "obstacle_aware_agent_attempt_limit": effective_max_obstacle_aware_attempts_per_agent,
                        "obstacle_aware_max_transition_length": round(
                            effective_obstacle_aware_max_transition_length,
                            6,
                        ),
                        "connector_cache_size": len(connector_cache),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        for candidate_idx, (_, _, region_id, candidate_pattern) in enumerate(candidate_slice):
            if step_time_budget > 0.0 and time.perf_counter() - step_started >= step_time_budget:
                step_budget_exhausted = True
                break
            if max_step_attempts > 0 and step_attempt_count >= max_step_attempts:
                step_attempt_limit_exhausted = True
                break
            if region_id not in attempted_region_ids:
                attempted_region_ids.append(region_id)
            candidate_attempt_count += 1
            step_attempt_count += 1
            connector_rejections: List[Dict[str, object]] = []
            candidate_transition_length = _transition_length(current_pose, candidate_pattern.entry_pose, config)
            candidate_started = time.perf_counter()
            if path_config.monitor_stages and (candidate_idx < 3 or _pattern_needs_connector_variant_diversity(candidate_pattern)):
                print(
                    json.dumps(
                        {
                            "stage": "agent_region_tsp_candidate_start",
                            "agent_id": agent_id,
                            "candidate_idx": candidate_idx,
                            "region_id": region_id,
                            "pattern_id": candidate_pattern.pattern_id,
                            "transition_length": round(candidate_transition_length, 6),
                            "is_merged_candidate": _pattern_needs_connector_variant_diversity(candidate_pattern),
                            "step_attempt_count": step_attempt_count,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            cheap_probe_failed = False
            if require_cheap_connector_probe:
                cheap_probe_failed = not _cheap_region_connector_probe(
                    current_pose,
                    candidate_pattern.entry_pose,
                    config,
                    path_config,
                    obstacle_field,
                    collision_only=cheap_probe_collision_only,
                )
            connector: List[PathSegmentSpec] | None = None
            if not cheap_probe_failed:
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
            direct_failed_for_geometry = any(
                token in str(rejection.get("reason", ""))
                for rejection in connector_rejections
                for token in ("out_of_bounds", "obstacle_collision")
            )
            step_elapsed_before_retry = time.perf_counter() - step_started
            retry_budget_remaining = (
                step_time_budget <= 0.0
                or step_elapsed_before_retry < 0.75 * step_time_budget
            )
            initial_merged_retry_blocked = (
                initial_entry_step
                and _pattern_needs_connector_variant_diversity(candidate_pattern)
                and effective_obstacle_aware_max_transition_length > 0.0
                and candidate_transition_length > effective_obstacle_aware_max_transition_length
            )
            retry_with_obstacle_aware = (
                (cheap_probe_failed or direct_failed_for_geometry)
                and connector is None
                and not feasible_choices
                and retry_budget_remaining
                and not initial_merged_retry_blocked
                and candidate_idx < obstacle_aware_retry_limit
                and (
                    step_obstacle_aware_attempt_limit <= 0
                    or step_obstacle_aware_attempt_count < step_obstacle_aware_attempt_limit
                )
                and (
                    effective_max_obstacle_aware_attempts_per_agent <= 0
                    or obstacle_aware_attempt_count < effective_max_obstacle_aware_attempts_per_agent
                )
                and (
                    effective_obstacle_aware_max_transition_length <= 0.0
                    or candidate_transition_length <= effective_obstacle_aware_max_transition_length
                )
            )
            if retry_with_obstacle_aware:
                connector_rejections = []
                step_obstacle_aware_attempt_count += 1
                obstacle_aware_attempt_count += 1
                obstacle_aware_path_config = path_config
                if initial_entry_step:
                    obstacle_aware_path_config = replace(
                        path_config,
                        obstacle_aware_allow_motion_lattice=True,
                        obstacle_aware_astar_max_expansions=max(
                            int(path_config.obstacle_aware_astar_max_expansions),
                            480,
                        ),
                    )
                if path_config.monitor_stages:
                    print(
                        json.dumps(
                            {
                                "stage": "agent_region_tsp_obstacle_aware_attempt",
                                "agent_id": agent_id,
                                "region_id": region_id,
                                "candidate_idx": candidate_idx,
                                "cheap_probe_failed": cheap_probe_failed,
                                "transition_length": round(candidate_transition_length, 6),
                                "max_transition_length": round(
                                    effective_obstacle_aware_max_transition_length,
                                    6,
                                ),
                                "astar_max_expansions": int(obstacle_aware_path_config.obstacle_aware_astar_max_expansions),
                                "motion_lattice_enabled": bool(
                                    obstacle_aware_path_config.obstacle_aware_allow_motion_lattice
                                ),
                                "step_obstacle_aware_attempt_count": step_obstacle_aware_attempt_count,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                connector = _build_region_connector_cached(
                    agent_id,
                    serial,
                    current_pose,
                    candidate_pattern.entry_pose,
                    current_time,
                    config,
                    obstacle_aware_path_config,
                    obstacle_field,
                    to_region=region_id,
                    rejection_sink=connector_rejections,
                    allow_obstacle_aware=True,
                    cache=connector_cache,
                )
            elif cheap_probe_failed:
                reason = "cheap_connector_probe_failed"
                if candidate_idx < obstacle_aware_retry_limit:
                    obstacle_aware_filtered_count += 1
                    if (
                        effective_max_obstacle_aware_attempts_per_agent > 0
                        and obstacle_aware_attempt_count >= effective_max_obstacle_aware_attempts_per_agent
                    ):
                        reason = "cheap_connector_probe_failed_obstacle_aware_retry_filtered_agent_limit"
                    elif step_obstacle_aware_attempt_limit > 0 and step_obstacle_aware_attempt_count >= step_obstacle_aware_attempt_limit:
                        reason = "cheap_connector_probe_failed_obstacle_aware_retry_filtered_step_limit"
                    elif (
                        effective_obstacle_aware_max_transition_length > 0.0
                        and candidate_transition_length > effective_obstacle_aware_max_transition_length
                    ):
                        reason = "cheap_connector_probe_failed_obstacle_aware_retry_filtered_distance"
                    elif not retry_budget_remaining:
                        reason = "cheap_connector_probe_failed_obstacle_aware_retry_filtered_step_time_guard"
                    elif initial_merged_retry_blocked:
                        reason = "cheap_connector_probe_failed_obstacle_aware_retry_filtered_initial_merged"
                connector_rejections.append(
                    {
                        "agent_id": agent_id,
                        "from": _pose_label(current_pose),
                        "to_region": region_id,
                        "reason": reason,
                        "transition_length": round(candidate_transition_length, 6),
                        "max_transition_length": round(
                            effective_obstacle_aware_max_transition_length,
                            6,
                        ),
                        "step_obstacle_aware_attempt_count": step_obstacle_aware_attempt_count,
                    }
                )
            elif connector is None and candidate_idx < obstacle_aware_retry_limit and not connector_rejections:
                obstacle_aware_filtered_count += 1
                if (
                    effective_max_obstacle_aware_attempts_per_agent > 0
                    and obstacle_aware_attempt_count >= effective_max_obstacle_aware_attempts_per_agent
                ):
                    filtered_reason = "obstacle_aware_retry_filtered_agent_limit"
                elif step_obstacle_aware_attempt_limit > 0 and step_obstacle_aware_attempt_count >= step_obstacle_aware_attempt_limit:
                    filtered_reason = "obstacle_aware_retry_filtered_step_limit"
                elif (
                        effective_obstacle_aware_max_transition_length > 0.0
                        and candidate_transition_length > effective_obstacle_aware_max_transition_length
                    ):
                        filtered_reason = "obstacle_aware_retry_filtered_distance"
                elif not retry_budget_remaining:
                    filtered_reason = "obstacle_aware_retry_filtered_step_time_guard"
                elif initial_merged_retry_blocked:
                    filtered_reason = "obstacle_aware_retry_filtered_initial_merged"
                else:
                    filtered_reason = "obstacle_aware_retry_filtered"
                connector_rejections.append(
                    {
                        "agent_id": agent_id,
                        "from": _pose_label(current_pose),
                        "to_region": region_id,
                        "reason": filtered_reason,
                        "transition_length": round(candidate_transition_length, 6),
                        "max_transition_length": round(
                            effective_obstacle_aware_max_transition_length,
                            6,
                        ),
                        "step_obstacle_aware_attempt_count": step_obstacle_aware_attempt_count,
                    }
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
                if path_config.monitor_stages and (candidate_idx < 3 or _pattern_needs_connector_variant_diversity(candidate_pattern)):
                    print(
                        json.dumps(
                            {
                                "stage": "agent_region_tsp_candidate_done",
                                "agent_id": agent_id,
                                "candidate_idx": candidate_idx,
                                "region_id": region_id,
                                "status": "rejected",
                                "dt_sec": round(time.perf_counter() - candidate_started, 3),
                                "reason": connector_rejections[-1].get("reason", "connector_failed") if connector_rejections else "connector_failed",
                                "step_attempt_count": step_attempt_count,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                continue
            if path_config.monitor_stages and (candidate_idx < 3 or _pattern_needs_connector_variant_diversity(candidate_pattern)):
                print(
                    json.dumps(
                        {
                            "stage": "agent_region_tsp_candidate_done",
                            "agent_id": agent_id,
                            "candidate_idx": candidate_idx,
                            "region_id": region_id,
                            "status": "connector_feasible",
                            "dt_sec": round(time.perf_counter() - candidate_started, 3),
                            "connector_segment_count": len(connector),
                            "connector_length": round(sum(segment.length for segment in connector), 6),
                            "step_attempt_count": step_attempt_count,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            connector_end_time = _segment_end_time(connector[-1]) if connector else current_time
            future_remaining = [item for item in remaining if item != region_id]
            lookahead_probe_limit = max(4, int(path_config.region_tsp_branch_limit))
            lookahead_coverage_length = 0.0
            if enable_lookahead_probe:
                lookahead_summary = _large_map_lookahead_reachable_count(
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
                    collision_only=cheap_probe_collision_only,
                    return_coverage=True,
                )
                if isinstance(lookahead_summary, tuple):
                    lookahead_reachable = int(lookahead_summary[0])
                    lookahead_coverage_length = float(lookahead_summary[1])
                else:
                    lookahead_reachable = int(lookahead_summary)
                reachability_probe_count += min(len(future_remaining), lookahead_probe_limit)
                reachability_probe_success_count += min(lookahead_reachable, lookahead_probe_limit)
                reachability_probe_coverage_length += lookahead_coverage_length
            else:
                lookahead_reachable = 0
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
            connector_cross_agent_score = score_cross_agent_ownership_overlap(
                _non_cover_segments(connector),
                agent_id,
                ownership_map,
                path_config,
                config=config,
                annotate=False,
            )
            connector_components = _connector_score_components(
                connector,
                connector_repeat_score,
                candidate_pattern,
                path_config,
                coverage_deficit,
            )
            score = (
                _connector_economy_penalty(connector_components, path_config, include_repeat=True)
                + 0.5 * candidate_pattern.estimated_time
                + _pattern_quality_penalty(candidate_pattern, path_config)
                + _scan_axis_switch_penalty(previous_pattern, candidate_pattern, path_config)
                + connector_cross_agent_score.penalty
                + path_config.coverage_priority_weight * coverage_deficit
                + 55.0 * max(0, min(len(future_remaining), 6) - lookahead_reachable)
                - 1.0 * candidate_pattern.coverage_length
                - 0.20 * lookahead_coverage_length
            )
            feasible_choices.append(
                {
                    "score": score,
                    "region_id": region_id,
                    "pattern": candidate_pattern,
                    "connector": connector,
                    "connector_end_time": connector_end_time,
                    "lookahead_reachable": lookahead_reachable,
                    "lookahead_coverage_length": lookahead_coverage_length,
                    "equivalent_region_count": _pattern_equivalent_region_count(candidate_pattern),
                    "execution_coverage_length": (
                        float(candidate_pattern.coverage_length) + lookahead_coverage_length
                    ),
                    "connector_repeat_score": connector_repeat_score,
                    "connector_cross_agent_score": connector_cross_agent_score,
                    "connector_score_components": connector_components,
                }
            )
            max_feasible_choice_count = max(2, min(4, int(path_config.region_tsp_branch_limit)))
            if len(feasible_choices) >= max_feasible_choice_count:
                break
        if not feasible_choices:
            skipped_region_id = attempted_region_ids[0] if attempted_region_ids else remaining[0]
            pending_infeasible_edges: List[Dict[str, object]] = []
            if step_budget_exhausted:
                skipped_reason = "large_map_tsp_step_time_budget_exhausted"
                pending_infeasible_edges.append(
                    {
                        "agent_id": agent_id,
                        "region_id": skipped_region_id,
                        "reason": skipped_reason,
                        "step_elapsed_sec": round(time.perf_counter() - step_started, 6),
                        "step_attempt_count": step_attempt_count,
                    }
                )
            elif step_attempt_limit_exhausted:
                skipped_reason = "large_map_tsp_step_attempt_limit_exhausted"
                pending_infeasible_edges.append(
                    {
                        "agent_id": agent_id,
                        "region_id": skipped_region_id,
                        "reason": skipped_reason,
                        "step_attempt_count": step_attempt_count,
                    }
                )
            elif local_rejections:
                pending_infeasible_edges.extend(local_rejections)
                skipped_reason = str(local_rejections[-1].get("reason", "no_valid_large_map_greedy_candidate"))
            else:
                skipped_reason = "no_valid_large_map_greedy_candidate"
                pending_infeasible_edges.append(
                    {
                        "agent_id": agent_id,
                        "region_id": skipped_region_id,
                        "reason": skipped_reason,
                    }
                )
            retryable_skip = (
                skipped_reason
                not in {
                    "missing_candidate_patterns",
                    "candidate_patterns_exhausted",
                }
                and (
                    skipped_reason.startswith("large_map_tsp_step_")
                    or skipped_reason in {
                        "no_valid_large_map_greedy_candidate",
                        "connector_failed",
                        "cheap_connector_probe_failed",
                    }
                    or any(
                        token in skipped_reason
                        for token in (
                            "obstacle_collision",
                            "out_of_bounds",
                            "kinematic_infeasible",
                            "obstacle_aware_retry_filtered",
                            "cheap_connector_probe_failed",
                        )
                    )
                )
            )
            max_region_deferrals = 2
            already_deferred = deferred_region_counts.get(skipped_region_id, 0)
            should_defer_region = (
                bool(path_config.prioritize_region_execution)
                and retryable_skip
                and len(remaining) > 1
                and already_deferred < max_region_deferrals
            )
            if should_defer_region:
                deferred_region_counts[skipped_region_id] = already_deferred + 1
                deferred_region_count += 1
                if initial_entry_step:
                    deferred_initial_anchor_count += 1
                remaining.remove(skipped_region_id)
                remaining.append(skipped_region_id)
                if path_config.monitor_stages:
                    print(
                        json.dumps(
                            {
                                "stage": "agent_region_tsp_defer_region",
                                "agent_id": agent_id,
                                "region_id": skipped_region_id,
                                "reason": skipped_reason,
                                "defer_count": deferred_region_counts[skipped_region_id],
                                "visited_region_count": len(final_order),
                                "remaining_region_count": len(remaining),
                                "step_attempt_count": step_attempt_count,
                                "step_elapsed_sec": round(time.perf_counter() - step_started, 3),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                continue
            infeasible_edges.extend(pending_infeasible_edges)
            skipped_region_reasons.setdefault(skipped_region_id, skipped_reason)
            connector_failure_reasons.setdefault(skipped_region_id, skipped_reason)
            _record_connector_failure(all_connector_failure_reasons, skipped_region_id, skipped_reason)
            if path_config.monitor_stages:
                print(
                    json.dumps(
                        {
                            "stage": "agent_region_tsp_skip",
                            "agent_id": agent_id,
                            "region_id": skipped_region_id,
                            "reason": skipped_reason,
                            "visited_region_count": len(final_order),
                            "remaining_region_count": len(remaining),
                            "step_attempt_count": step_attempt_count,
                            "rejected_candidate_count": rejected_candidate_count,
                            "connector_cache_size": len(connector_cache),
                            "step_elapsed_sec": round(time.perf_counter() - step_started, 3),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if path_config.monitor_stages and (step_budget_exhausted or step_attempt_limit_exhausted):
                print(
                    json.dumps(
                        {
                            "stage": "agent_region_tsp_budget",
                            "agent_id": agent_id,
                            "reason": skipped_reason,
                            "skipped_region_id": skipped_region_id,
                            "visited_region_count": len(final_order),
                            "remaining_region_count": len(remaining),
                            "step_attempt_count": step_attempt_count,
                            "step_obstacle_aware_attempt_count": step_obstacle_aware_attempt_count,
                            "step_elapsed_sec": round(time.perf_counter() - step_started, 3),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            remaining.remove(skipped_region_id)
            continue

        feasible_choices.sort(key=_large_map_feasible_choice_sort_key)
        chosen = feasible_choices[0]
        if len(feasible_choices) > 1 and int(chosen["lookahead_reachable"]) > int(feasible_choices[-1]["lookahead_reachable"]):
            dead_end_avoidance_count += 1
        candidate_pattern = chosen["pattern"]
        connector = chosen["connector"]
        region_id = candidate_pattern.region_id
        chosen_equivalent_region_count = int(chosen.get("equivalent_region_count", 1) or 1)
        if path_config.monitor_stages and (
            len(final_order) == 0 or chosen_equivalent_region_count > 1
        ):
            print(
                json.dumps(
                    {
                        "stage": "agent_region_tsp_choice",
                        "agent_id": agent_id,
                        "region_id": region_id,
                        "pattern_id": candidate_pattern.pattern_id,
                        "remaining_region_count": len(remaining),
                        "equivalent_region_count": chosen_equivalent_region_count,
                        "lookahead_reachable_equivalent_region_count": int(
                            chosen.get("lookahead_reachable", 0) or 0
                        ),
                        "execution_coverage_length": round(
                            float(chosen.get("execution_coverage_length", 0.0) or 0.0),
                            6,
                        ),
                        "candidate_attempt_count": candidate_attempt_count,
                        "connector_cache_size": len(connector_cache),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        commit_started = time.perf_counter()
        if path_config.monitor_stages:
            print(
                json.dumps(
                    {
                        "stage": "agent_region_tsp_commit_start",
                        "agent_id": agent_id,
                        "region_id": region_id,
                        "pattern_id": candidate_pattern.pattern_id,
                        "visited_region_count": len(final_order),
                        "remaining_region_count": len(remaining),
                        "is_merged_candidate": _pattern_needs_connector_variant_diversity(candidate_pattern),
                        "pattern_pass_count": len(candidate_pattern.passes),
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
            failed_internal_pattern_ids.add(candidate_pattern.pattern_id)
            infeasible_edges.append(
                {
                    "agent_id": agent_id,
                    "region_id": region_id,
                    "pattern_id": candidate_pattern.pattern_id,
                    "reason": reason,
                }
            )
            _record_connector_failure(all_connector_failure_reasons, region_id, reason)
            has_remaining_pattern_variant = any(
                pattern.pattern_id not in failed_internal_pattern_ids
                for pattern in patterns.get(region_id, [])
            )
            if has_remaining_pattern_variant and region_id in remaining:
                remaining.remove(region_id)
                remaining.append(region_id)
                if path_config.monitor_stages:
                    print(
                        json.dumps(
                            {
                                "stage": "agent_region_tsp_internal_sweep_retry",
                                "agent_id": agent_id,
                                "region_id": region_id,
                                "pattern_id": candidate_pattern.pattern_id,
                                "reason": reason,
                                "remaining_pattern_variant_count": sum(
                                    1
                                    for pattern in patterns.get(region_id, [])
                                    if pattern.pattern_id not in failed_internal_pattern_ids
                                ),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                continue
            skipped_region_reasons.setdefault(region_id, reason)
            connector_failure_reasons.setdefault(region_id, reason)
            remaining.remove(region_id)
            continue
        candidate_segments = list(connector) + list(sweep_segments)
        if path_config.monitor_stages:
            print(
                json.dumps(
                    {
                        "stage": "agent_region_tsp_commit_sweep_ready",
                        "agent_id": agent_id,
                        "region_id": region_id,
                        "pattern_id": candidate_pattern.pattern_id,
                        "sweep_segment_count": len(sweep_segments),
                        "commit_dt_sec": round(time.perf_counter() - commit_started, 3),
                        "sweep_segment_cache_size": len(sweep_segment_cache),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        connector_components = dict(chosen.get("connector_score_components", {}) or {})
        if not connector_components:
            connector_components = _connector_score_components(
                connector,
                RepeatOverlapScore(0.0, 0.0, 0.0, 0, 0),
                candidate_pattern,
                path_config,
                0.0,
            )
        _annotate_connector_score_components(connector, connector_components, path_config)
        repeat_score = RepeatOverlapScore(0.0, 0.0, 0.0, 0, 0)
        connector_non_cover_segments = _non_cover_segments(connector)
        if path_config.enable_main_repeat_path_penalty:
            repeat_score = score_repeat_overlap(
                connector_non_cover_segments,
                _non_cover_segments(segments),
                path_config,
                penalty_weight=max(path_config.main_repeat_path_penalty_weight, 0.0)
                * max(path_config.connector_noncover_repeat_penalty_multiplier, 0.0),
                annotate=True,
            )
        cross_score = score_cross_agent_ownership_overlap(
            connector_non_cover_segments,
            agent_id,
            ownership_map,
            path_config,
            config=config,
            annotate=True,
        )
        if path_config.monitor_stages:
            print(
                json.dumps(
                    {
                        "stage": "agent_region_tsp_commit_overlap_ready",
                        "agent_id": agent_id,
                        "region_id": region_id,
                        "pattern_id": candidate_pattern.pattern_id,
                        "connector_noncover_segment_count": len(connector_non_cover_segments),
                        "repeat_sampled_point_count": repeat_score.sampled_point_count,
                        "cross_agent_sampled_point_count": cross_score.sampled_point_count,
                        "commit_dt_sec": round(time.perf_counter() - commit_started, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        main_repeat_overlap += repeat_score.overlap_length
        main_repeat_penalty += repeat_score.penalty + _metadata_float(candidate_pattern.metadata, "internal_repeat_penalty", 0.0)
        connector_repeat_score = chosen.get("connector_repeat_score", RepeatOverlapScore(0.0, 0.0, 0.0, 0, 0))
        connector_noncover_repeat_length += float(getattr(connector_repeat_score, "overlap_length", 0.0))
        connector_noncover_repeat_penalty += float(getattr(connector_repeat_score, "penalty", 0.0))
        connector_length_total += float(connector_components.get("connector_length", 0.0))
        connector_turn_angle_total += float(connector_components.get("connector_turn_angle", 0.0))
        cross_agent_overlap += cross_score.overlap_length
        cross_agent_penalty += cross_score.penalty
        segments.extend(candidate_segments)
        final_order.append(region_id)
        selected_patterns[region_id] = candidate_pattern
        selected_pattern_ids[region_id] = candidate_pattern.pattern_id
        skipped_region_reasons.pop(region_id, None)
        connector_failure_reasons.pop(region_id, None)
        all_connector_failure_reasons.pop(region_id, None)
        remaining.remove(region_id)
        serial += len(candidate_segments)
        current_time = _segment_end_time(sweep_segments[-1])
        current_pose = candidate_pattern.exit_pose
        if path_config.monitor_stages:
            print(
                json.dumps(
                    {
                        "stage": "agent_region_tsp_progress",
                        "agent_id": agent_id,
                        "last_region_id": region_id,
                        "visited_region_count": len(final_order),
                        "remaining_region_count": len(remaining),
                        "commit_dt_sec": round(time.perf_counter() - commit_started, 3),
                        "candidate_segment_count": len(candidate_segments),
                        "sweep_segment_count": len(sweep_segments),
                        "candidate_attempt_count": candidate_attempt_count,
                        "rejected_candidate_count": rejected_candidate_count,
                        "connector_cache_size": len(connector_cache),
                        "obstacle_aware_attempt_count": obstacle_aware_attempt_count,
                        "obstacle_aware_filtered_count": obstacle_aware_filtered_count,
                        "lookahead_probe_enabled": enable_lookahead_probe,
                        "cheap_connector_probe_required": require_cheap_connector_probe,
                        "cheap_connector_probe_collision_only": cheap_probe_collision_only,
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
        "large_map_reachability_probe_coverage_length": round(reachability_probe_coverage_length, 6),
        "large_map_dead_end_avoidance_count": dead_end_avoidance_count,
        "large_map_greedy_budget_exhausted": budget_exhausted,
        "large_map_greedy_budget_reason": budget_reason,
        "large_map_greedy_elapsed_sec": round(time.perf_counter() - tsp_started, 6),
        "large_map_greedy_step_time_budget_sec": step_time_budget,
        "large_map_greedy_agent_time_budget_sec": agent_time_budget,
        "large_map_greedy_max_candidate_attempts_per_step": max_step_attempts,
        "large_map_greedy_obstacle_aware_attempt_count": obstacle_aware_attempt_count,
        "large_map_greedy_obstacle_aware_filtered_count": obstacle_aware_filtered_count,
        "large_map_greedy_max_obstacle_aware_attempts_per_step": max_obstacle_aware_attempts_per_step,
        "large_map_greedy_max_obstacle_aware_attempts_per_agent": max_obstacle_aware_attempts_per_agent,
        "large_map_greedy_obstacle_aware_max_transition_length": obstacle_aware_max_transition_length,
        "large_map_greedy_initial_obstacle_aware_max_transition_length": initial_obstacle_aware_max_transition_length,
        "large_map_greedy_obstacle_aware_astar_max_expansions": int(path_config.obstacle_aware_astar_max_expansions),
        "large_map_greedy_lookahead_probe_enabled": enable_lookahead_probe,
        "large_map_greedy_cheap_connector_probe_required": require_cheap_connector_probe,
        "large_map_greedy_cheap_probe_collision_only": cheap_probe_collision_only,
        "large_map_greedy_deferred_region_count": deferred_region_count,
        "large_map_greedy_deferred_initial_anchor_count": deferred_initial_anchor_count,
        "large_map_greedy_failed_internal_pattern_count": len(failed_internal_pattern_ids),
    }
    result = {
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
        "connector_noncover_repeat_length": connector_noncover_repeat_length,
        "connector_noncover_repeat_penalty": connector_noncover_repeat_penalty,
        "connector_length": connector_length_total,
        "connector_turn_angle": connector_turn_angle_total,
        "cross_agent_overlap_length": cross_agent_overlap,
        "cross_agent_penalty_total": cross_agent_penalty,
        "unavoidable_cross_agent_overlap_count": 0,
        "tsp_solver_metadata": metadata,
        "skipped_region_reasons": skipped_region_reasons,
        "connector_failure_reasons": connector_failure_reasons,
        "all_connector_failure_reasons": _connector_failure_lists(all_connector_failure_reasons),
    }
    return _large_map_dead_end_restart_result(
        result=result,
        agent_id=agent_id,
        initial_order=initial_order,
        patterns=patterns,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        ownership_map=ownership_map,
        fallback_solver_metadata=fallback_solver_metadata,
        sweep_segment_templates=sweep_segment_templates,
        restart_depth=_restart_depth,
        initial_forbidden_region_ids=initial_forbidden_region_ids,
    )


def _large_map_dead_end_restart_result(
    *,
    result: Dict[str, object],
    agent_id: int,
    initial_order: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    ownership_map: CoverageOwnershipMap | None,
    fallback_solver_metadata: Dict[str, object] | None,
    sweep_segment_templates: Dict[str, Tuple[List[PathSegmentSpec], str]] | None,
    restart_depth: int,
    initial_forbidden_region_ids: set[str],
) -> Dict[str, object]:
    metadata = dict(result.get("tsp_solver_metadata", {}) or {})
    metadata.setdefault("large_map_dead_end_restart_attempt_count", 0)
    metadata.setdefault("large_map_dead_end_restart_accepted_count", 0)
    metadata.setdefault("large_map_dead_end_restart_forbidden_initial_regions", [])
    result["tsp_solver_metadata"] = metadata
    restart_limit = max(int(path_config.large_map_dead_end_restart_limit), 0)
    trigger_ratio = min(max(float(path_config.large_map_dead_end_restart_trigger_ratio), 0.0), 1.0)
    final_order = list(result.get("final_order", []) or [])
    if (
        not bool(path_config.enable_large_map_dead_end_restart)
        or restart_depth >= restart_limit
        or len(initial_order) <= 1
        or not final_order
        or len(final_order) / max(len(initial_order), 1) >= trigger_ratio
    ):
        return result

    first_region_id = final_order[0]
    forbidden = set(initial_forbidden_region_ids)
    forbidden.add(first_region_id)
    restarted = _solve_agent_region_tsp_large_map_greedy(
        agent_id,
        initial_order,
        patterns,
        config,
        path_config,
        obstacle_field,
        ownership_map,
        fallback_solver_metadata,
        sweep_segment_templates,
        _restart_depth=restart_depth + 1,
        _initial_forbidden_region_ids=forbidden,
    )
    base_score = _large_map_result_execution_score(
        result,
        agent_id=agent_id,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
    )
    restart_score = _large_map_result_execution_score(
        restarted,
        agent_id=agent_id,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
    )
    selected = restarted if restart_score > base_score else result
    selected_metadata = dict(selected.get("tsp_solver_metadata", {}) or {})
    selected_metadata.update(
        {
            "large_map_dead_end_restart_attempt_count": 1,
            "large_map_dead_end_restart_accepted_count": int(selected is restarted),
            "large_map_dead_end_restart_forbidden_initial_regions": sorted(forbidden),
            "large_map_dead_end_restart_base_score": list(base_score),
            "large_map_dead_end_restart_candidate_score": list(restart_score),
        }
    )
    selected["tsp_solver_metadata"] = selected_metadata
    return selected


def _large_map_result_execution_score(
    result: Dict[str, object],
    *,
    agent_id: int,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
) -> Tuple[float, float, float, float, int, float, float]:
    selected_patterns = dict(result.get("selected_patterns", {}) or {})
    segments = list(result.get("segments", []) or [])
    coverage_tour = SingleUsvTourPlan(
        agent_id=agent_id,
        region_order=list(result.get("final_order", []) or []),
        selected_patterns=selected_patterns,
        segments=segments,
    )
    coverage_state = evaluate_tour_coverage_state(
        config,
        [coverage_tour],
        resolution=max(float(path_config.residual_resolution), 1.0),
        obstacle_field=obstacle_field,
        include_non_cover_segments=False,
    )
    equivalent_region_count = sum(
        _pattern_equivalent_region_count(pattern)
        for pattern in selected_patterns.values()
    )
    coverage_length = _selected_pattern_coverage_length(selected_patterns)
    final_count = len(result.get("final_order", []) or [])
    cross_agent_overlap = float(result.get("cross_agent_overlap_length", 0.0) or 0.0)
    connector_length = float(result.get("connector_length", 0.0) or 0.0)
    turn_angle = float(result.get("connector_turn_angle", 0.0) or 0.0)
    return (
        float(coverage_state.coverage_fraction),
        -cross_agent_overlap,
        equivalent_region_count,
        coverage_length,
        final_count,
        -connector_length,
        -turn_angle,
    )


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
    connector_cache: Dict[Tuple[object, ...], Tuple[List[PathSegmentSpec] | None, List[Dict[str, object]]]],
    collision_only: bool = False,
    return_coverage: bool = False,
) -> int | Tuple[int, float]:
    if not remaining:
        return (0, 0.0) if return_coverage else 0
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
    reachable_coverage = 0.0
    probe_limit = max(4, int(path_config.region_tsp_branch_limit))
    for region_id in ordered[:probe_limit]:
        region_reachable = False
        region_patterns = list(patterns.get(region_id, []))
        pattern_limit = _connector_pattern_limit_for_region(region_id, region_patterns, path_config)
        for pattern in sorted(region_patterns, key=lambda item: (_pattern_sort_key(item, config, path_config), item.pattern_id))[:pattern_limit]:
            if _cheap_region_connector_probe(
                current_pose,
                pattern.entry_pose,
                config,
                path_config,
                obstacle_field,
                collision_only=collision_only,
            ):
                region_reachable = True
                break
        if region_reachable:
            reachable += 1
            reachable_coverage += max((pattern.coverage_length for pattern in region_patterns), default=0.0)
    if return_coverage:
        return reachable, reachable_coverage
    return reachable


def _large_map_feasible_choice_sort_key(item: Dict[str, object]) -> Tuple[object, ...]:
    equivalent_region_count = int(item.get("equivalent_region_count", 1) or 1)
    lookahead_reachable = int(item.get("lookahead_reachable", 0) or 0)
    pattern = item["pattern"]
    return (
        -lookahead_reachable,
        -float(item.get("lookahead_coverage_length", 0.0) or 0.0),
        -equivalent_region_count,
        float(item["score"]),
        str(item["region_id"]),
        pattern.pattern_id,
    )


def _pattern_equivalent_region_count(pattern: RegionCoveragePattern) -> int:
    """Count base regions represented by a merged TSP candidate."""

    metadata = dict(getattr(pattern, "metadata", {}) or {})
    is_merged_candidate = (
        metadata.get("coverage_aware_merged") == "true"
        or metadata.get("agent_task_strip_merge") == "true"
        or metadata.get("agent_task_unified_merge") == "true"
        or bool(metadata.get("merge_fallback_source_ids"))
    )
    if not is_merged_candidate:
        return 1
    explicit_count = int(_metadata_float(metadata, "merge_equivalent_source_region_count", 0.0))
    if explicit_count > 0:
        return explicit_count
    for key in (
        "merge_fallback_source_ids",
        "agent_task_strip_source_ids",
        "agent_task_unified_source_ids",
    ):
        raw = metadata.get(key)
        if not raw:
            continue
        source_ids = {item.strip() for item in str(raw).split(",") if item.strip()}
        if source_ids:
            return max(len(source_ids), 1)
    return 1


def _cheap_region_connector_probe(
    start: Pose2D,
    end: Pose2D,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    collision_only: bool = False,
) -> bool:
    segment = build_transition_segment(
        segment_id="large_map_lookahead_probe",
        start=start,
        end=end,
        start_time=0.0,
        config=config,
        kind="transit",
        sample_count=_connector_sample_count(start, end, config, base=16),
        use_bezier=path_config.use_bezier_smoothing,
    )
    if collision_only:
        reasons = set(path_segment_invalid_reasons(segment, config, obstacle_field))
        return not ({"out_of_bounds", "obstacle_collision"} & reasons)
    return validate_transition_sequence([segment], config, obstacle_field=obstacle_field, retime=True).valid


def _connector_sample_count(
    start: Pose2D,
    end: Pose2D,
    config: PlannerConfig,
    base: int = 24,
    max_count: int = 128,
) -> int:
    approx_length = _transition_length(start, end, config)
    target_step = max(0.75, 0.60 * max(config.fleet.min_turn_radius, 1e-6))
    return max(int(base), min(int(max_count), int(math.ceil(approx_length / target_step)) + 1))


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
    connector_noncover_repeat_length = 0.0
    connector_noncover_repeat_penalty = 0.0
    connector_length_total = 0.0
    connector_turn_angle_total = 0.0
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
        coverage_deficit = _metadata_float(
            pattern.metadata,
            "coverage_deficit",
            max(0.0, path_config.target_coverage_fraction - _estimated_pattern_coverage_fraction(pattern, config)),
        ) if use_coverage_priority else 0.0
        connector_repeat_score = score_repeat_overlap(
            _non_cover_segments(connector),
            segments,
            path_config,
            penalty_weight=repeat_weight,
            annotate=False,
        )
        connector_components = _connector_score_components(
            connector,
            connector_repeat_score,
            pattern,
            path_config,
            coverage_deficit,
        )
        _annotate_connector_score_components(connector, connector_components, path_config)
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
        connector_noncover_repeat_length += connector_repeat_score.overlap_length
        connector_noncover_repeat_penalty += connector_repeat_score.penalty
        connector_length_total += connector_components["connector_length"]
        connector_turn_angle_total += connector_components["connector_turn_angle"]
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
        "connector_noncover_repeat_length": connector_noncover_repeat_length,
        "connector_noncover_repeat_penalty": connector_noncover_repeat_penalty,
        "connector_length": connector_length_total,
        "connector_turn_angle": connector_turn_angle_total,
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
        "connector_noncover_repeat_length": 0.0,
        "connector_noncover_repeat_penalty": 0.0,
        "connector_length": 0.0,
        "connector_turn_angle": 0.0,
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
    agent_feasible_patterns: Dict[int, Dict[str, List[RegionCoveragePattern]]] | None = None,
    agent_obstacle_fields: Dict[int, ObstacleField | None] | None = None,
) -> Dict[str, object]:
    recovery_started = time.perf_counter()
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

    large_map_mode = max(config.mission.area_length_x, config.mission.area_length_y) >= max(
        path_config.large_map_size_threshold,
        1e-6,
    )
    recovered: List[str] = []
    failed: Dict[str, str] = {}
    budget_exhausted = False
    budget_reason = ""
    time_budget_sec = max(float(path_config.skipped_region_recovery_time_budget_sec), 0.0)
    recovery_connector_attempt_count = 0
    recovery_prefiltered_count = 0
    recovery_agent_pruned_count = 0
    large_map_recovery_connector_attempt_limit = (
        max(128, int(path_config.large_map_tsp_max_candidate_attempts_per_step) * 8)
        if large_map_mode
        else 0
    )
    large_map_recovery_agent_limit = min(max(3, len(agents)), len(agents)) if large_map_mode else max(1, len(agents))
    large_map_recovery_transition_limit = (
        min(
            max(config.mission.area_length_x, config.mission.area_length_y) * 0.8,
            max(float(path_config.large_map_tsp_obstacle_aware_max_transition_length), 1.0),
        )
        if large_map_mode
        else 0.0
    )

    def current_time_budget_reason() -> str:
        if time_budget_sec > 0.0 and time.perf_counter() - recovery_started >= time_budget_sec:
            return "skipped_region_recovery_time_budget_exhausted"
        return ""

    def current_budget_reason() -> str:
        reason = current_time_budget_reason()
        if reason:
            return reason
        if (
            large_map_recovery_connector_attempt_limit > 0
            and recovery_connector_attempt_count >= large_map_recovery_connector_attempt_limit
        ):
            return "skipped_region_recovery_connector_attempt_budget_exhausted"
        return ""

    connector_cache: Dict[Tuple[object, ...], Tuple[List[PathSegmentSpec] | None, List[Dict[str, object]]]] = {}
    sweep_segment_cache: Dict[str, Tuple[List[PathSegmentSpec], str]] = {
        key: (copy.deepcopy(value[0]), value[1])
        for key, value in (sweep_segment_templates or {}).items()
    }
    recovery_budget = max(1, min(len(skipped), max(4, int(path_config.max_residual_backfill_regions))))
    max_attempts = min(len(skipped), recovery_budget)
    short_region_attempt_count = 0
    short_region_success_count = 0
    for _ in range(max_attempts):
        reason = current_budget_reason()
        if reason:
            budget_exhausted = True
            budget_reason = reason
            break
        best_choice = None
        obstacle_aware_retry_limit = 1 if large_map_mode else max(2, min(8, int(path_config.region_tsp_branch_limit) // 2))
        obstacle_aware_retry_count = 0
        active_skipped = sorted(
            [region_id for region_id in skipped if region_id not in recovered],
            key=lambda region_id: _skipped_region_recovery_priority(region_id, feasible_patterns, agents, config),
        )[:recovery_budget]
        for region_id in active_skipped:
            reason = current_budget_reason()
            if reason:
                budget_exhausted = True
                budget_reason = reason
                break
            if region_id in recovered:
                continue
            short_recovery_candidate = _is_short_region_recovery_candidate(region_id, feasible_patterns, config)
            if short_recovery_candidate and path_config.enable_short_region_connector_recovery:
                short_region_attempt_count += 1
            candidate_patterns_by_agent: Dict[int, List[RegionCoveragePattern]] = {}
            for agent_id in agents:
                candidate_source = (
                    agent_feasible_patterns.get(agent_id, {})
                    if agent_feasible_patterns is not None
                    else feasible_patterns
                )
                agent_config = config.for_agent(agent_id) if agent_feasible_patterns is not None else config
                candidates = _skipped_region_recovery_pattern_candidates(
                    region_id,
                    candidate_source,
                    agent_config,
                    path_config,
                    short_recovery_candidate,
                )
                if candidates:
                    candidate_patterns_by_agent[agent_id] = candidates
            if not candidate_patterns_by_agent:
                failed.setdefault(region_id, "missing_candidate_patterns")
                continue
            agent_items = sorted(
                (
                    (agent_id, agents[agent_id])
                    for agent_id in candidate_patterns_by_agent
                ),
                key=lambda item: min(
                    _transition_length(
                        _agent_end_pose(item[1], config),
                        pattern.entry_pose,
                        config.for_agent(item[0]) if agent_feasible_patterns is not None else config,
                    )
                    for pattern in candidate_patterns_by_agent[item[0]]
                ),
            )
            agent_limit = (
                min(len(agent_items), max(large_map_recovery_agent_limit, 3))
                if short_recovery_candidate and path_config.enable_short_region_connector_recovery
                else min(len(agent_items), large_map_recovery_agent_limit)
            )
            if large_map_mode and len(agent_items) > agent_limit:
                recovery_agent_pruned_count += len(agent_items) - agent_limit
                agent_items = agent_items[:agent_limit]
            for agent_id, agent in agent_items:
                reason = current_budget_reason()
                if reason:
                    budget_exhausted = True
                    budget_reason = reason
                    break
                agent_config = config.for_agent(agent_id) if agent_feasible_patterns is not None else config
                agent_field = (
                    agent_obstacle_fields.get(agent_id, obstacle_field)
                    if agent_obstacle_fields is not None
                    else obstacle_field
                )
                pattern_candidates = candidate_patterns_by_agent[agent_id]
                current_pose = _agent_end_pose(agent, config)
                current_time = max((_segment_end_time(segment) for segment in agent.segments), default=0.0)
                serial = len(agent.segments)
                for pattern in pattern_candidates:
                    reason = current_time_budget_reason()
                    if reason:
                        budget_exhausted = True
                        budget_reason = reason
                        break
                    candidate_transition_length = _transition_length(current_pose, pattern.entry_pose, agent_config)
                    if (
                        large_map_recovery_transition_limit > 0.0
                        and candidate_transition_length > large_map_recovery_transition_limit
                        and not short_recovery_candidate
                    ):
                        recovery_prefiltered_count += 1
                        reason = "recovery_connector_prefiltered_distance"
                        failed.setdefault(region_id, reason)
                        _record_recovery_failure_in_tsp_records(tsp_records, region_id, reason)
                        continue
                    if large_map_mode and not _cheap_region_connector_probe(
                        current_pose,
                        pattern.entry_pose,
                        agent_config,
                        path_config,
                        agent_field,
                        collision_only=True,
                    ):
                        recovery_prefiltered_count += 1
                        reason = "recovery_cheap_connector_probe_failed"
                        failed.setdefault(region_id, reason)
                        _record_recovery_failure_in_tsp_records(tsp_records, region_id, reason)
                        continue
                    recovery_connector_attempt_count += 1
                    rejections: List[Dict[str, object]] = []
                    repaired_rejections: List[Dict[str, object]] = []
                    connector = _build_region_connector_cached(
                        agent_id=agent_id,
                        serial=serial,
                        start=current_pose,
                        end=pattern.entry_pose,
                        start_time=current_time,
                        config=agent_config,
                        path_config=path_config,
                        obstacle_field=agent_field,
                        to_region=region_id,
                        rejection_sink=rejections,
                        allow_obstacle_aware=False,
                        cache=connector_cache,
                    )
                    reason = current_budget_reason()
                    if reason:
                        budget_exhausted = True
                        budget_reason = reason
                        break
                    if connector is None and obstacle_aware_retry_count < obstacle_aware_retry_limit:
                        obstacle_aware_retry_count += 1
                        recovery_connector_attempt_count += 1
                        connector = _build_region_connector_cached(
                            agent_id=agent_id,
                            serial=serial,
                            start=current_pose,
                            end=pattern.entry_pose,
                            start_time=current_time,
                            config=agent_config,
                            path_config=path_config,
                            obstacle_field=agent_field,
                            to_region=region_id,
                            rejection_sink=repaired_rejections,
                            allow_obstacle_aware=True,
                            cache=connector_cache,
                        )
                        reason = current_time_budget_reason()
                        if reason:
                            budget_exhausted = True
                            budget_reason = reason
                            break
                    if connector is None:
                        combined_rejections = repaired_rejections or rejections
                        if combined_rejections:
                            reason = str(combined_rejections[-1].get("reason", "connector_failed"))
                            failed.setdefault(region_id, reason)
                            _record_recovery_failure_in_tsp_records(tsp_records, region_id, reason)
                        continue
                    connector_end_time = _segment_end_time(connector[-1]) if connector else current_time
                    recovery_path_config = (
                        replace(path_config, enable_open_sweep_chain_tsp=True)
                        if short_recovery_candidate and path_config.enable_short_region_connector_recovery
                        else path_config
                    )
                    sweep_segments, reason = _cached_internal_sweep_segments(
                        pattern,
                        agent_config,
                        recovery_path_config,
                        agent_field,
                        start_time=connector_end_time,
                        segment_prefix=f"agent{agent_id}_recovered_region_{region_id}",
                        cache=sweep_segment_cache,
                    )
                    if reason:
                        failed.setdefault(region_id, reason)
                        _record_recovery_failure_in_tsp_records(tsp_records, region_id, reason)
                        continue
                    candidate_segments = list(connector) + list(sweep_segments)
                    if not validate_transition_sequence(
                        candidate_segments,
                        agent_config,
                        obstacle_field=agent_field,
                        retime=True,
                    ).valid:
                        failed.setdefault(region_id, "dynamic_validation_failed")
                        _record_recovery_failure_in_tsp_records(tsp_records, region_id, "dynamic_validation_failed")
                        continue
                    score = (
                        sum(segment.length for segment in candidate_segments)
                        + pattern.estimated_time
                        + _pattern_quality_penalty(pattern, path_config)
                        + _turn_clearance_penalty(pattern.exit_pose, agent_config)
                        - 2.0 * pattern.coverage_length
                    )
                    key = (-pattern.coverage_length, score, agent_id, region_id, pattern.pattern_id)
                    if best_choice is None or key < best_choice[0]:
                        best_choice = (key, agent_id, region_id, pattern, candidate_segments)
                if budget_exhausted:
                    break
            if budget_exhausted:
                break
        if budget_exhausted:
            break
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
        tour.total_turn_angle = _path_heading_variation(tour.segments)
        tour.estimated_time = max((_segment_end_time(segment) for segment in tour.segments), default=0.0)
        recovered.append(region_id)
        if _is_short_region_recovery_candidate(region_id, feasible_patterns, config):
            short_region_success_count += 1
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
        if path_config.monitor_stages:
            print(
                json.dumps(
                    {
                        "stage": "skipped_region_recovery_progress",
                        "recovered_count": len(recovered),
                        "remaining_count": len([item for item in skipped if item not in set(recovered)]),
                        "connector_cache_size": len(connector_cache),
                        "elapsed_sec": round(time.perf_counter() - recovery_started, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    remaining_failed = [region_id for region_id in skipped if region_id not in set(recovered)]
    for region_id in remaining_failed:
        failed.setdefault(region_id, budget_reason or "no_feasible_recovery_insertion")
    return {
        "enabled": True,
        "recovered_count": len(recovered),
        "failed_count": len(remaining_failed),
        "recovered_regions": recovered,
        "failure_reasons": failed,
        "connector_cache_size": len(connector_cache),
        "short_region_recovery_attempt_count": short_region_attempt_count,
        "short_region_recovery_success_count": short_region_success_count,
        "recovery_connector_attempt_count": recovery_connector_attempt_count,
        "recovery_prefiltered_count": recovery_prefiltered_count,
        "recovery_agent_pruned_count": recovery_agent_pruned_count,
        "budget_exhausted": budget_exhausted,
        "budget_reason": budget_reason,
        "elapsed_sec": time.perf_counter() - recovery_started,
    }


def _is_short_region_recovery_candidate(
    region_id: str,
    feasible_patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
) -> bool:
    patterns = feasible_patterns.get(region_id, [])
    if not patterns:
        return False
    best_length = max((pattern.coverage_length for pattern in patterns), default=0.0)
    min_pass_count = min((len(pattern.passes) for pattern in patterns), default=999)
    length_threshold = max(config.footprint.length_lf * 3.0, config.footprint.width_wf * 4.0)
    return best_length <= length_threshold + 1e-9 or min_pass_count <= 1


def _skipped_region_recovery_pattern_candidates(
    region_id: str,
    feasible_patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    short_recovery_candidate: bool,
) -> List[RegionCoveragePattern]:
    patterns = sorted(
        feasible_patterns.get(region_id, []),
        key=lambda item: (_pattern_sort_key(item, config, path_config), item.pattern_id),
    )
    if not patterns:
        return []
    limit = 1
    if path_config.enable_short_region_connector_recovery and short_recovery_candidate:
        limit = max(2, min(max(int(path_config.large_region_connector_pattern_limit), 2), len(patterns)))
    if any(_pattern_needs_connector_variant_diversity(pattern) for pattern in patterns):
        limit = max(limit, min(_merged_region_connector_variant_limit(path_config), len(patterns)))
    selected = list(patterns[:limit])
    if path_config.enable_short_region_connector_recovery and short_recovery_candidate:
        selected.extend(_reverse_region_pattern(pattern) for pattern in patterns[:limit])
    deduped: List[RegionCoveragePattern] = []
    seen: set[str] = set()
    for pattern in selected:
        if pattern.pattern_id in seen:
            continue
        seen.add(pattern.pattern_id)
        deduped.append(pattern)
    return deduped


def _reverse_region_pattern(pattern: RegionCoveragePattern) -> RegionCoveragePattern:
    reversed_passes: List[CoveragePass] = []
    for sequence_index, coverage_pass in enumerate(reversed(pattern.passes)):
        start = coverage_pass.end_pose
        end = coverage_pass.start_pose
        heading = _line_heading(start, end)
        start_pose = Pose2D(start.x, start.y, heading)
        end_pose = Pose2D(end.x, end.y, heading)
        reversed_passes.append(
            replace(
                coverage_pass,
                pass_id=f"{coverage_pass.pass_id}_recovery_reverse",
                sequence_index=sequence_index,
                start_pose=start_pose,
                end_pose=end_pose,
            )
        )
    if not reversed_passes:
        return pattern
    return replace(
        pattern,
        pattern_id=f"{pattern.pattern_id}_recovery_reverse",
        passes=reversed_passes,
        entry_pose=reversed_passes[0].start_pose,
        exit_pose=reversed_passes[-1].end_pose,
        metadata={**pattern.metadata, "recovery_reverse_pattern": "true"},
    )


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
    cache: Dict[Tuple[object, ...], Tuple[List[PathSegmentSpec] | None, List[Dict[str, object]]]],
) -> List[PathSegmentSpec] | None:
    profile_fingerprint = (
        config.profile_for_agent(agent_id).fingerprint
        if config.agent_profiles and agent_id in config.agent_profiles
        else (
            f"legacy:{config.footprint.length_lf:.6g}:{config.footprint.width_wf:.6g}:"
            f"{config.fleet.min_turn_radius:.6g}"
        )
    )
    key = (
        int(agent_id),
        profile_fingerprint,
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
    sample_count = _connector_sample_count(start, end, config, base=48 if allow_obstacle_aware else 24)
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
            sample_count=sample_count,
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
                sample_count=sample_count,
                use_bezier=path_config.use_bezier_smoothing,
            )
        ]
    report = validate_transition_sequence(segments, config, obstacle_field=obstacle_field, retime=True)
    if not report.valid:
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
        dynamic_cost = (
            segment.length
            if _large_map_mode_enabled(config, path_config)
            else dynamic_edge_cost([segment], config)
        )
        segment.metadata.update(
            {
                "to_region": to_region,
                "region_tsp_edge": "true",
                "resource_id": f"region_tsp:{agent_id}:{to_region}:{idx}",
                "dynamic_edge_cost": f"{dynamic_cost:.6f}",
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
    total_turn = _path_heading_variation(segments)
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
        "turn_count": float(_path_turn_count(segments)),
        "turn_segment_count": float(sum(1 for segment in segments if segment.kind == "turn")),
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
        "estimated_time",
        "turn_count",
        "turn_segment_count",
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


def _connector_score_components(
    connector: Sequence[PathSegmentSpec],
    repeat_score: RepeatOverlapScore,
    pattern: RegionCoveragePattern,
    path_config: PathPlanningConfig,
    coverage_deficit: float,
) -> Dict[str, float]:
    connector_length = sum(segment.length for segment in connector)
    connector_turn_angle = _path_heading_variation(connector)
    connector_turn_count = float(_path_turn_count(connector))
    pattern_quality_penalty = _pattern_quality_penalty(pattern, path_config)
    return {
        "connector_length": connector_length,
        "connector_turn_angle": connector_turn_angle,
        "connector_turn_count": connector_turn_count,
        "connector_noncover_repeat_length": repeat_score.overlap_length,
        "connector_noncover_repeat_penalty": repeat_score.penalty,
        "pattern_quality_penalty": pattern_quality_penalty,
        "coverage_deficit": coverage_deficit,
        "coverage_deficit_penalty": path_config.coverage_priority_weight * coverage_deficit,
        "connector_economy_penalty": _connector_economy_penalty(
            {
                "connector_length": connector_length,
                "connector_turn_angle": connector_turn_angle,
                "connector_turn_count": connector_turn_count,
                "connector_noncover_repeat_penalty": repeat_score.penalty,
            },
            path_config,
            include_repeat=True,
        ),
    }


def _connector_economy_penalty(
    components: Dict[str, float],
    path_config: PathPlanningConfig,
    include_repeat: bool = True,
) -> float:
    penalty = (
        max(path_config.transition_length_weight, 0.0) * float(components.get("connector_length", 0.0) or 0.0)
        + max(path_config.turn_angle_weight, 0.0) * float(components.get("connector_turn_angle", 0.0) or 0.0)
        + max(path_config.turn_count_weight, 0.0) * float(components.get("connector_turn_count", 0.0) or 0.0)
    )
    if include_repeat:
        penalty += float(components.get("connector_noncover_repeat_penalty", 0.0) or 0.0)
    return penalty


def _annotate_connector_score_components(
    connector: Sequence[PathSegmentSpec],
    components: Dict[str, float],
    path_config: PathPlanningConfig,
) -> None:
    if not path_config.report_score_components:
        return
    for segment in connector:
        segment.metadata.update(
            {
                "connector_length_total": f"{components.get('connector_length', 0.0):.6f}",
                "connector_turn_angle_total": f"{components.get('connector_turn_angle', 0.0):.6f}",
                "connector_turn_count_total": f"{components.get('connector_turn_count', 0.0):.6f}",
                "connector_noncover_repeat_length": f"{components.get('connector_noncover_repeat_length', 0.0):.6f}",
                "connector_noncover_repeat_penalty": f"{components.get('connector_noncover_repeat_penalty', 0.0):.6f}",
                "connector_pattern_quality_penalty": f"{components.get('pattern_quality_penalty', 0.0):.6f}",
                "connector_coverage_deficit_penalty": f"{components.get('coverage_deficit_penalty', 0.0):.6f}",
                "connector_economy_penalty": f"{components.get('connector_economy_penalty', 0.0):.6f}",
            }
        )


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
    obstacle_aware_attempt_count = 0
    obstacle_aware_filtered_count = 0
    cheap_probe_collision_only = False
    components: Dict[str, Dict[str, int]] = {}
    for agent_id, record in tsp_records.items():
        metadata = record.get("tsp_solver_metadata", {}) or {}
        cache_size += int(metadata.get("large_map_connector_cache_size", 0) or 0)
        probe_count += int(metadata.get("large_map_reachability_probe_count", 0) or 0)
        probe_success += int(metadata.get("large_map_reachability_probe_success_count", 0) or 0)
        dead_end_avoidance += int(metadata.get("large_map_dead_end_avoidance_count", 0) or 0)
        obstacle_aware_attempt_count += int(metadata.get("large_map_greedy_obstacle_aware_attempt_count", 0) or 0)
        obstacle_aware_filtered_count += int(metadata.get("large_map_greedy_obstacle_aware_filtered_count", 0) or 0)
        cheap_probe_collision_only = cheap_probe_collision_only or bool(
            metadata.get("large_map_greedy_cheap_probe_collision_only", False)
        )
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
        "large_map_greedy_cheap_probe_collision_only": cheap_probe_collision_only,
        "large_map_greedy_obstacle_aware_attempt_count": obstacle_aware_attempt_count,
        "large_map_greedy_obstacle_aware_filtered_count": obstacle_aware_filtered_count,
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


def _path_heading_variation(segments: Sequence[PathSegmentSpec]) -> float:
    total = 0.0
    previous_heading: float | None = None
    for segment in segments:
        if not segment.waypoints:
            continue
        if previous_heading is not None:
            total += abs(wrap_angle(segment.waypoints[0].psi - previous_heading))
        total += _segment_heading_variation(segment)
        previous_heading = segment.waypoints[-1].psi
    return total


def _path_turn_count(segments: Sequence[PathSegmentSpec], threshold_rad: float = math.radians(5.0)) -> int:
    count = 0
    previous_heading: float | None = None
    threshold = max(float(threshold_rad), 0.0)
    for segment in segments:
        if not segment.waypoints:
            continue
        headings = [waypoint.psi for waypoint in segment.waypoints]
        if previous_heading is not None and abs(wrap_angle(headings[0] - previous_heading)) > threshold:
            count += 1
        for idx in range(1, len(headings)):
            if abs(wrap_angle(headings[idx] - headings[idx - 1])) > threshold:
                count += 1
        previous_heading = headings[-1]
    return count


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
