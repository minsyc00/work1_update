from __future__ import annotations

import json
import math
from dataclasses import replace
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from usv_swarm import (  # noqa: E402
    PathPlanningLayer,
    StaticObstacle,
    build_experiment_output_dir,
    load_map_for_planner,
    load_fleet_profile_json,
    plan_global_coverage,
    run_paper_style_region_tsp_experiment,
    run_planning_algorithm_experiment,
)
from usv_swarm.schema import AgentPlanningProfile, CoverageFootprint, CoverageResidual, FleetConfig, MissionConfig, PlannerConfig, PlannerWeights, Pose2D, SafetyMargins, State3DOF, State6DOF, VehicleFootprint  # noqa: E402
from usv_swarm.path_planning.adapters.runtime_adapter import path_plan_to_trajectory_references  # noqa: E402
from usv_swarm.dubins import dubins_shortest_path  # noqa: E402
from usv_swarm.path_planning.aco import solve_aco_tsp_cpp  # noqa: E402
from usv_swarm.path_planning.assignment import assign_heterogeneous_connected_regions, apply_lightweight_load_swap, balance_region_workload  # noqa: E402
from usv_swarm.path_planning.astar import obstacle_aware_grid_astar, sailing_safety_weight, turn_aware_astar  # noqa: E402
from usv_swarm.path_planning.coverage import (  # noqa: E402
    RectangularCoverageModel,
    build_coverage_state,
    find_residual_components,
    mark_coverage_passes,
)
from usv_swarm.path_planning.decomposition import (  # noqa: E402
    build_composite_free_space_regions,
    build_free_space_cells,
    build_large_convex_free_space_regions,
    concave_vertex_indices,
    decompose_obstacle_aware_area,
    decompose_polygon_interface,
    decompose_rectangular_area,
)
from usv_swarm.path_planning.dynamics_validation import validate_transition_dynamics, validate_transition_sequence  # noqa: E402
from usv_swarm.path_planning.graph import build_region_graph, graph_is_connected  # noqa: E402
from usv_swarm.path_planning.obstacles import (  # noqa: E402
    circle_obstacle,
    distance_point_to_polygon,
    ellipse_obstacle,
    normalize_obstacle_field,
    obstacle_bounds,
    path_segment_invalid_reasons,
    point_in_any_obstacle,
    point_in_polygon,
    polyline_collides_with_obstacles,
    polygon_collides_with_obstacles,
    polygon_obstacle,
    rectangle_obstacle,
    sampled_segment_footprint_collides,
)
import usv_swarm.path_planning.paper_style_experiment as paper_style_experiment  # noqa: E402
from usv_swarm.path_planning.paper_style_experiment import (  # noqa: E402
    _build_region_sweep_paths,
    _split_pattern_into_open_chains,
    _coverage_aware_merge_regions,
    _coarsen_paper_style_regions,
    _estimated_pattern_coverage_fraction,
    _generate_paper_style_patterns,
    _merge_performance_regions,
    _prefilter_region_patterns,
    _short_region_compression_variants,
)
from usv_swarm.path_planning.patterns import generate_all_region_patterns, generate_region_patterns  # noqa: E402
from usv_swarm.path_planning.performance import build_performance_summary  # noqa: E402
from usv_swarm.path_planning.residuals import assign_residual_backfill, evaluate_tour_coverage_state  # noqa: E402
from usv_swarm.path_planning.residual_planner import append_residual_local_tsp  # noqa: E402
import usv_swarm.path_planning.residual_planner as residual_planner  # noqa: E402
from usv_swarm.path_planning.resources import estimate_repeat_overlap_length, score_cross_agent_ownership_overlap, score_repeat_overlap, shared_resource_metrics  # noqa: E402
from usv_swarm.path_planning.scheduling import apply_resource_window_schedule  # noqa: E402
from usv_swarm.path_planning.smoothing import _convert_corridor_to_trackable_segments, build_obstacle_aware_transition_segments  # noqa: E402
from usv_swarm.path_planning.tsp import solve_multi_agent_tours, solve_single_usv_tsp_cpp  # noqa: E402
from usv_swarm.path_planning.types import (  # noqa: E402
    AgentPathPlan,
    BalancedAssignment,
    CompositeFreeSpaceRegion,
    CoveragePass,
    CoverageOwnershipMap,
    DecomposedRegion,
    FreeSpaceCell,
    OpenSweepBreak,
    OpenSweepChain,
    ObstacleField,
    PathPlanningConfig,
    PathSegmentSpec,
    PathWaypoint,
    RegionGraph,
    RegionCoveragePattern,
    RegionSweepPath,
    SingleUsvTourPlan,
)
from test_framework import build_test_config  # noqa: E402


class PathPlanningLayerTests(unittest.TestCase):
    def test_layer_runs_paper_fusion_pipeline_from_existing_plan(self) -> None:
        config = build_test_config()
        planning_result = plan_global_coverage(config)
        layer = PathPlanningLayer()
        path_plan = layer.plan_from_config(config, planning_result=planning_result)
        self.assertEqual(path_plan.algorithm_name, "paper_fusion_planner")
        self.assertEqual(path_plan.metadata["status"], "paper_fusion")
        self.assertEqual(set(path_plan.agents.keys()), set(range(config.fleet.num_agents or 0)))
        for agent_id, agent_plan in path_plan.agents.items():
            self.assertEqual(agent_plan.source_algorithm, "paper_fusion_planner")
            self.assertGreater(len(agent_plan.segments), 0)
            self.assertLessEqual(
                agent_plan.metrics["max_curvature"],
                1.0 / config.fleet.min_turn_radius + 1e-3,
            )

    def test_layer_exposes_registered_algorithms(self) -> None:
        layer = PathPlanningLayer()
        self.assertIn("paper_fusion_planner", layer.available_algorithms())

    def test_path_planning_types_are_instantiable(self) -> None:
        path_config = PathPlanningConfig(max_regions_per_agent=2)
        region = DecomposedRegion(
            region_id="r0",
            bounds=(0.0, 0.0, 10.0, 4.0),
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)],
            center=(5.0, 2.0),
            area=40.0,
            preferred_axis="x",
        )
        self.assertEqual(path_config.max_regions_per_agent, 2)
        self.assertEqual(region.center, (5.0, 2.0))

    def test_rectangular_coverage_model_and_residuals(self) -> None:
        config = build_test_config()
        model = RectangularCoverageModel.from_config(config)
        self.assertAlmostEqual(model.strip_spacing, config.footprint.width_wf * (1.0 - config.mission.overlap_ratio))
        self.assertGreaterEqual(
            model.turn_buffer,
            config.fleet.min_turn_radius + config.footprint.length_lf / 2.0 + config.safety.d_safe,
        )

        state = build_coverage_state(config)
        passes = [
            (Pose2D(2.0, y, 0.0), Pose2D(config.mission.area_length_x - 2.0, y, 0.0))
            for y in (2.0, 6.0, 10.0, 14.0, 16.0)
        ]
        mark_coverage_passes(state, passes, model, eta_cov=config.footprint.eta_cov)
        residuals = find_residual_components(state, (0.0, 0.0, config.mission.area_length_x, config.mission.area_length_y))
        self.assertEqual(residuals, [])
        self.assertAlmostEqual(state.coverage_fraction, 1.0)

        empty_state = build_coverage_state(config)
        self.assertGreater(len(find_residual_components(empty_state)), 0)

    def test_rectangular_and_concave_decomposition(self) -> None:
        config = build_test_config()
        regions = decompose_rectangular_area(config)
        self.assertGreaterEqual(len(regions), config.fleet.num_agents or 0)
        self.assertTrue(all(region.bounds[0] >= 0.0 and region.bounds[2] <= config.mission.area_length_x for region in regions))
        self.assertTrue(all(region.bounds[1] >= 0.0 and region.bounds[3] <= config.mission.area_length_y for region in regions))
        self.assertTrue(all(region.neighbors or len(regions) == 1 for region in regions))

        l_shape = [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (2.0, 2.0), (2.0, 6.0), (0.0, 6.0)]
        self.assertGreater(len(concave_vertex_indices(l_shape)), 0)
        concave_regions = decompose_polygon_interface(l_shape, preferred_axis="x")
        self.assertGreater(len(concave_regions), 1)

    def test_large_convex_decomposition_keeps_open_rectangle_whole(self) -> None:
        config = build_test_config()
        regions = build_large_convex_free_space_regions(
            config,
            replace(PathPlanningConfig.from_planner_config(config), large_region_max_area_fraction=1.0),
            ObstacleField(),
        )
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].metadata.get("shape_class"), "rectangle")
        self.assertEqual(regions[0].metadata.get("convex_region_decomposition"), "true")

    def test_trapezoid_pattern_uses_true_polygon_intersections(self) -> None:
        config = build_test_config()
        polygon = [(2.0, 0.0), (10.0, 0.0), (8.0, 6.0), (0.0, 6.0)]
        region = DecomposedRegion(
            region_id="trapezoid_region",
            bounds=(0.0, 0.0, 10.0, 6.0),
            polygon=polygon,
            center=(5.0, 3.0),
            area=48.0,
            preferred_axis="x",
            source_algorithm="large_convex_free_space_decomposition",
            metadata={
                "convex_region_decomposition": "true",
                "shape_class": "trapezoid",
                "dominant_scan_axis": "x",
            },
        )
        patterns = generate_region_patterns(region, config, PathPlanningConfig.from_planner_config(config))
        self.assertGreaterEqual(len(patterns), 1)
        for coverage_pass in patterns[0].passes:
            self.assertTrue(point_in_polygon((coverage_pass.start_pose.x, coverage_pass.start_pose.y), polygon))
            self.assertTrue(point_in_polygon((coverage_pass.end_pose.x, coverage_pass.end_pose.y), polygon))

    def test_oriented_sweep_generates_tilted_passes_for_rotated_rectangle(self) -> None:
        config = replace(
            _build_visual_test_config(),
            mission=MissionConfig(area_length_x=60.0, area_length_y=40.0, overlap_ratio=0.2, local_control_hz=5.0),
            footprint=CoverageFootprint(length_lf=2.0, width_wf=2.0, eta_cov=0.7),
            safety=SafetyMargins(d_safe=0.0, boundary_margin_x=0.0, boundary_margin_y=0.0),
        )
        center = (30.0, 20.0)
        length = 24.0
        width = 6.0
        angle = math.radians(30.0)
        ux, uy = math.cos(angle), math.sin(angle)
        vx, vy = -math.sin(angle), math.cos(angle)
        polygon = [
            (center[0] + sx * length / 2.0 * ux + sy * width / 2.0 * vx, center[1] + sx * length / 2.0 * uy + sy * width / 2.0 * vy)
            for sx, sy in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        ]
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        region = DecomposedRegion(
            region_id="rotated_rectangle",
            bounds=(min(xs), min(ys), max(xs), max(ys)),
            polygon=polygon,
            center=center,
            area=length * width,
            preferred_axis="x",
            source_algorithm="large_convex_free_space_decomposition",
            metadata={
                "convex_region_decomposition": "true",
                "shape_class": "convex_polygon",
                "dominant_scan_axis": "theta",
            },
        )
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_oriented_sweep_patterns=True,
            max_oriented_sweep_angles_per_region=4,
            include_axis_aligned_sweep_fallbacks=True,
        )

        patterns = generate_region_patterns(region, config, path_config)
        oriented = [pattern for pattern in patterns if pattern.scan_axis.startswith("theta:")]
        axis_aligned = [pattern for pattern in patterns if pattern.scan_axis in {"x", "y"}]

        self.assertTrue(oriented)
        self.assertTrue(axis_aligned)
        best_oriented = min(oriented, key=lambda item: len(item.passes))
        self.assertLess(len(best_oriented.passes), min(len(pattern.passes) for pattern in axis_aligned))
        self.assertIn(best_oriented.metadata["scan_axis_source"], {"minimum_span_direction", "edge_direction", "long_axis_direction"})
        self.assertGreater(float(best_oriented.metadata["scan_angle_deg"]), 1.0)
        self.assertLess(abs(float(best_oriented.metadata["scan_angle_deg"]) - 90.0), 89.0)
        for coverage_pass in best_oriented.passes:
            self.assertLessEqual(distance_point_to_polygon((coverage_pass.start_pose.x, coverage_pass.start_pose.y), polygon), 1e-6)
            self.assertLessEqual(distance_point_to_polygon((coverage_pass.end_pose.x, coverage_pass.end_pose.y), polygon), 1e-6)
        if len(best_oriented.passes) >= 2:
            heading_delta = abs((best_oriented.passes[0].start_pose.psi - best_oriented.passes[1].start_pose.psi + math.pi) % (2.0 * math.pi) - math.pi)
            self.assertAlmostEqual(heading_delta, math.pi, delta=1e-6)

    def test_oriented_sweep_keeps_axis_aligned_fallbacks_for_convex_rectangle(self) -> None:
        config = _build_visual_test_config()
        region = DecomposedRegion(
            region_id="axis_rectangle",
            bounds=(2.0, 2.0, 12.0, 8.0),
            polygon=[(2.0, 2.0), (12.0, 2.0), (12.0, 8.0), (2.0, 8.0)],
            center=(7.0, 5.0),
            area=60.0,
            preferred_axis="x",
            source_algorithm="large_convex_free_space_decomposition",
            metadata={"convex_region_decomposition": "true", "shape_class": "rectangle"},
        )
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_oriented_sweep_patterns=True,
            include_axis_aligned_sweep_fallbacks=True,
        )

        patterns = generate_region_patterns(region, config, path_config)
        axes = {pattern.scan_axis for pattern in patterns}

        self.assertIn("x", axes)
        self.assertIn("y", axes)

    def test_oriented_sweep_uses_geometry_convexity_without_shape_class(self) -> None:
        config = _build_visual_test_config()
        angle = math.radians(25.0)
        center = (8.0, 5.0)
        length = 8.0
        width = 3.0
        ux, uy = math.cos(angle), math.sin(angle)
        vx, vy = -math.sin(angle), math.cos(angle)
        polygon = [
            (
                center[0] + sx * length / 2.0 * ux + sy * width / 2.0 * vx,
                center[1] + sx * length / 2.0 * uy + sy * width / 2.0 * vy,
            )
            for sx, sy in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        ]
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        region = DecomposedRegion(
            region_id="convex_no_shape_class",
            bounds=(min(xs), min(ys), max(xs), max(ys)),
            polygon=polygon,
            center=center,
            area=length * width,
            preferred_axis="x",
            metadata={},
        )
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_oriented_sweep_patterns=True,
            include_axis_aligned_sweep_fallbacks=True,
        )

        patterns = generate_region_patterns(region, config, path_config)
        oriented = [pattern for pattern in patterns if pattern.scan_axis.startswith("theta:")]

        self.assertTrue(oriented)
        self.assertEqual(region.metadata["convexity_status"], "convex")
        self.assertEqual(region.metadata["oriented_sweep_skip_reason"], "")
        self.assertGreater(int(region.metadata["selected_oriented_angle_count"]), 0)

    def test_short_region_recovery_candidates_include_reverse_pattern(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_short_region_connector_recovery=True,
            large_region_connector_pattern_limit=2,
        )
        first = _manual_pattern("short_a", (2.0, 2.0, 6.0, 4.0), 1, 3.0, 1.0, config, region_id="short_region")
        second = _manual_pattern("short_b", (2.0, 4.0, 6.0, 6.0), 1, 3.0, 1.0, config, region_id="short_region")

        candidates = paper_style_experiment._skipped_region_recovery_pattern_candidates(
            "short_region",
            {"short_region": [first, second]},
            config,
            path_config,
            short_recovery_candidate=True,
        )

        pattern_ids = {pattern.pattern_id for pattern in candidates}
        self.assertIn("short_a_recovery_reverse", pattern_ids)
        self.assertIn("short_b_recovery_reverse", pattern_ids)
        reversed_pattern = next(pattern for pattern in candidates if pattern.pattern_id == "short_a_recovery_reverse")
        self.assertAlmostEqual(reversed_pattern.entry_pose.x, first.exit_pose.x)
        self.assertAlmostEqual(reversed_pattern.exit_pose.x, first.entry_pose.x)

    def test_region_patterns_and_dubins_feasibility(self) -> None:
        config = build_test_config()
        regions = decompose_rectangular_area(config)
        patterns = generate_all_region_patterns(regions, config)
        for region in regions:
            candidates = patterns[region.region_id]
            self.assertGreaterEqual(len(candidates), 1)
            for pattern in candidates:
                self.assertGreaterEqual(pattern.estimated_time, 0.0)
                self.assertGreaterEqual(pattern.coverage_length, 0.0)
                self.assertLessEqual(pattern.max_curvature, 1.0 / config.fleet.min_turn_radius + 1e-3)
                centers = [coverage_pass.center_coordinate for coverage_pass in pattern.passes]
                for first, second in zip(centers, centers[1:]):
                    self.assertLessEqual(abs(second - first), config.footprint.width_wf + 1e-6)

    def test_region_graph_and_balanced_assignment_for_multiple_fleets(self) -> None:
        for agent_count in (2, 4, 8):
            config = _build_config_for_agents(agent_count)
            regions = decompose_rectangular_area(config)
            patterns = generate_all_region_patterns(regions, config)
            graph = build_region_graph(regions, patterns, config)
            self.assertTrue(graph_is_connected(graph))
            self.assertEqual(set(graph.node_weights), {region.region_id for region in regions})
            self.assertGreater(len(graph.edge_weights), 0)

            assignment = balance_region_workload(graph, config)
            assigned = sorted(region_id for region_ids in assignment.agent_regions.values() for region_id in region_ids)
            self.assertEqual(assigned, sorted(graph.regions))
            self.assertTrue(all(assignment.connected.values()))
            self.assertLessEqual(assignment.imbalance_ratio, 0.10 + 1e-6)

    def test_turn_aware_astar_prefers_safer_branch(self) -> None:
        graph = _build_toy_astar_graph()
        result = turn_aware_astar(graph, "a", "d")
        self.assertTrue(result.found)
        self.assertEqual(result.path, ["a", "c", "d"])
        self.assertGreater(sailing_safety_weight(4), sailing_safety_weight(0))
        restricted = turn_aware_astar(graph, "a", "d", allowed_nodes={"a", "b", "d"})
        self.assertTrue(restricted.found)
        self.assertEqual(restricted.path, ["a", "b", "d"])

    def test_single_usv_tsp_cpp_and_adapter(self) -> None:
        config = build_test_config()
        regions = decompose_rectangular_area(config)
        patterns = generate_all_region_patterns(regions, config)
        graph = build_region_graph(regions, patterns, config)
        assignment = balance_region_workload(graph, config)
        region_ids = assignment.agent_regions[0]
        tour = solve_single_usv_tsp_cpp(0, region_ids, graph, config)
        self.assertEqual(sorted(tour.region_order), sorted(region_ids))
        self.assertEqual(set(tour.selected_patterns), set(region_ids))
        self.assertGreater(len(tour.segments), 0)
        self.assertGreater(tour.objective, 0.0)
        self.assertEqual(tour.diagnostics["ordering_source"], "turn_aware_astar")

        tour_3opt = solve_single_usv_tsp_cpp(
            0,
            region_ids,
            graph,
            config,
            path_config=PathPlanningConfig(
                overlap_ratio=config.mission.overlap_ratio,
                coverage_resolution=config.footprint.width_wf / 2.0,
                residual_resolution=config.footprint.width_wf / 2.0,
                tsp_3opt_iterations=1,
            ),
        )
        self.assertLessEqual(tour_3opt.objective, tour.objective + 1e-6)

        layer = PathPlanningLayer()
        path_plan = layer.plan_from_config(config)
        refs = path_plan_to_trajectory_references(path_plan)
        self.assertEqual(set(refs), set(range(config.fleet.num_agents or 0)))
        self.assertTrue(all(ref.samples for ref in refs.values()))

    def test_aco_tsp_cpp_is_seed_reproducible(self) -> None:
        config = build_test_config()
        regions = decompose_rectangular_area(config)[:4]
        patterns = generate_all_region_patterns(regions, config)
        region_ids = [region.region_id for region in regions]
        start_pose = config.fleet.initial_states_3dof[0].pose()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            tsp_solver="aco",
            aco_ant_count=8,
            aco_iterations=8,
            aco_random_seed=7,
        )

        def edge_cost(previous, candidate) -> float:
            pose = start_pose if previous is None else previous.exit_pose
            return dubins_shortest_path(pose, candidate.entry_pose, config.fleet.min_turn_radius).total_length + candidate.estimated_time

        first = solve_aco_tsp_cpp(region_ids, patterns, start_pose, path_config, edge_cost)
        second = solve_aco_tsp_cpp(region_ids, patterns, start_pose, path_config, edge_cost)
        self.assertEqual(first.status, "success")
        self.assertEqual(first.region_order, second.region_order)
        self.assertEqual(
            {key: value.pattern_id for key, value in first.selected_patterns.items()},
            {key: value.pattern_id for key, value in second.selected_patterns.items()},
        )

    def test_fa3aco_keeps_best_solution_not_worse_than_initial(self) -> None:
        config = build_test_config()
        regions = decompose_rectangular_area(config)[:5]
        patterns = generate_all_region_patterns(regions, config)
        region_ids = [region.region_id for region in regions]
        start_pose = config.fleet.initial_states_3dof[0].pose()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            tsp_solver="fa3aco",
            aco_ant_count=8,
            aco_iterations=8,
            aco_random_seed=11,
        )

        def edge_cost(previous, candidate) -> float:
            pose = start_pose if previous is None else previous.exit_pose
            return dubins_shortest_path(pose, candidate.entry_pose, config.fleet.min_turn_radius).total_length + candidate.estimated_time

        result = solve_aco_tsp_cpp(region_ids, patterns, start_pose, path_config, edge_cost)
        self.assertEqual(result.status, "success")
        self.assertLessEqual(result.objective, result.initial_objective + 1e-9)
        self.assertGreaterEqual(result.accepted_3opt_count, 0)

    def test_tsp_solver_invalid_value_is_rejected(self) -> None:
        config = build_test_config()
        regions = decompose_rectangular_area(config)
        patterns = generate_all_region_patterns(regions, config)
        graph = build_region_graph(regions, patterns, config)
        with self.assertRaises(ValueError):
            solve_single_usv_tsp_cpp(
                0,
                [region.region_id for region in regions[:2]],
                graph,
                config,
                path_config=replace(PathPlanningConfig.from_planner_config(config), tsp_solver="bad_solver"),
            )

    def test_path_planner_accepts_aco_and_fa3aco_solvers(self) -> None:
        config = _build_visual_test_config()
        for solver in ("aco", "fa3aco"):
            path_config = replace(
                PathPlanningConfig.from_planner_config(config),
                tsp_solver=solver,
                aco_ant_count=5,
                aco_iterations=5,
                aco_random_seed=3,
                residual_backfill_cycles=0,
            )
            path_plan = PathPlanningLayer().plan_from_config(config, path_config=path_config)
            self.assertEqual(path_plan.metadata["requested_tsp_solver"], solver)
            self.assertIn(path_plan.metadata["effective_tsp_solver"], {solver, "deterministic_fallback"})
            self.assertEqual(set(path_plan.agents), {0})

    def test_residual_backfill_assigns_nearest_available_agent(self) -> None:
        config = build_test_config()
        regions = decompose_rectangular_area(config)
        patterns = generate_all_region_patterns(regions, config)
        graph = build_region_graph(regions, patterns, config)
        assignment = balance_region_workload(graph, config)
        tours = list(solve_multi_agent_tours(assignment.agent_regions, graph, config).values())
        empty_state = build_coverage_state(config)
        residuals = find_residual_components(empty_state, (0.0, 0.0, config.mission.area_length_x, config.mission.area_length_y))
        backfill = assign_residual_backfill(residuals[:1], tours, config)
        self.assertEqual(len(backfill.residual_regions), 1)
        self.assertEqual(sum(len(items) for items in backfill.agent_regions.values()), 1)
        self.assertEqual(backfill.diagnostics["assigned_count"], "1")

    def test_static_obstacle_shapes_normalize_and_inflate(self) -> None:
        config = build_test_config()
        obstacles = _build_static_obstacles()
        field = normalize_obstacle_field(obstacles, config)
        self.assertEqual(len(field.obstacles), 4)
        self.assertEqual(len(field.inflated_obstacles), 4)
        self.assertAlmostEqual(field.footprint_margin, config.footprint.width_wf / 2.0)
        self.assertAlmostEqual(float(field.metadata["inflation"]), config.safety.d_safe + config.footprint.width_wf / 2.0)
        raw_rect_bounds = obstacle_bounds(field.obstacles[0])
        inflated_rect_bounds = obstacle_bounds(field.inflated_obstacles[0])
        self.assertAlmostEqual(inflated_rect_bounds[2] - inflated_rect_bounds[0], (raw_rect_bounds[2] - raw_rect_bounds[0]) + 2.0 * float(field.metadata["inflation"]))
        self.assertGreater(len(field.obstacles[1].polygon), 8)
        raw_circle_bounds = obstacle_bounds(field.obstacles[1])
        inflated_circle_bounds = obstacle_bounds(field.inflated_obstacles[1])
        self.assertLess(inflated_circle_bounds[0], raw_circle_bounds[0])
        self.assertGreater(inflated_circle_bounds[2], raw_circle_bounds[2])
        self.assertFalse(point_in_any_obstacle((0.0, 7.0), field, inflated=True))

    def test_obstacle_aware_free_space_decomposition_and_patterns(self) -> None:
        config = build_test_config()
        field = normalize_obstacle_field(_build_static_obstacles(), config)
        cells = build_free_space_cells(config, PathPlanningConfig.from_planner_config(config), field)
        self.assertGreater(len(cells), 0)
        for cell in cells:
            self.assertFalse(point_in_any_obstacle(cell.center, field, inflated=True))
            self.assertTrue(set(cell.neighbors).issubset({other.cell_id for other in cells}))

        regions = decompose_obstacle_aware_area(config, PathPlanningConfig.from_planner_config(config), field)
        patterns = generate_all_region_patterns(regions, config, obstacle_field=field)
        self.assertGreater(len(regions), 0)
        self.assertTrue(any(region.metadata.get("static_obstacle_aware") == "true" for region in regions))
        self.assertTrue(any(patterns[region.region_id] for region in regions))
        for candidates in patterns.values():
            for pattern in candidates:
                self.assertEqual(pattern.metadata["static_obstacle_aware"], "true")

    def test_composite_region_scans_member_cells_without_covering_obstacle_hole(self) -> None:
        base = _build_visual_test_config()
        config = replace(
            base,
            mission=MissionConfig(area_length_x=6.0, area_length_y=4.0, overlap_ratio=0.2, local_control_hz=5.0),
            footprint=CoverageFootprint(length_lf=1.0, width_wf=1.0, eta_cov=0.7),
            safety=SafetyMargins(d_safe=0.0, boundary_margin_x=0.0, boundary_margin_y=0.0),
            fleet=replace(base.fleet, min_turn_radius=0.5),
        )
        cells = [
            FreeSpaceCell("left", (0.0, 0.0, 2.0, 4.0), [(0.0, 0.0), (2.0, 0.0), (2.0, 4.0), (0.0, 4.0)], (1.0, 2.0), 8.0, "x"),
            FreeSpaceCell("right", (4.0, 0.0, 6.0, 4.0), [(4.0, 0.0), (6.0, 0.0), (6.0, 4.0), (4.0, 4.0)], (5.0, 2.0), 8.0, "x"),
            FreeSpaceCell("bottom", (2.0, 0.0, 4.0, 1.0), [(2.0, 0.0), (4.0, 0.0), (4.0, 1.0), (2.0, 1.0)], (3.0, 0.5), 2.0, "x"),
            FreeSpaceCell("top", (2.0, 3.0, 4.0, 4.0), [(2.0, 3.0), (4.0, 3.0), (4.0, 4.0), (2.0, 4.0)], (3.0, 3.5), 2.0, "x"),
        ]
        cells[0].neighbors = ["bottom", "top"]
        cells[1].neighbors = ["bottom", "top"]
        cells[2].neighbors = ["left", "right"]
        cells[3].neighbors = ["left", "right"]
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            composite_max_member_cells=8,
            composite_max_region_area_fraction=1.0,
        )
        field = normalize_obstacle_field(
            [rectangle_obstacle("hole", center=(3.0, 2.0), width=0.8, height=0.8)],
            config,
            path_config,
        )
        composite = CompositeFreeSpaceRegion(
            region_id="ring",
            bounds=(0.0, 0.0, 6.0, 4.0),
            polygon=[(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)],
            center=(3.0, 2.0),
            area=sum(cell.area for cell in cells),
            preferred_axis="x",
            source_algorithm="unit_test_composite",
            member_cells=cells,
            metadata={"is_composite": "true"},
        )
        self.assertEqual(composite.metadata["is_composite"], "true")
        self.assertEqual(len(composite.member_cells), 4)

        patterns = generate_all_region_patterns([composite], config, path_config, obstacle_field=field)
        candidates = patterns[composite.region_id]
        self.assertTrue(candidates)
        x_pattern = next(pattern for pattern in candidates if pattern.scan_axis == "x")
        split_lines = {}
        for coverage_pass in x_pattern.passes:
            self.assertFalse(
                sampled_segment_footprint_collides(
                    coverage_pass.start_pose,
                    coverage_pass.end_pose,
                    config.footprint.length_lf,
                    config.footprint.width_wf,
                    field,
                    sample_spacing=0.25,
                    inflated=False,
                )
            )
            split_lines.setdefault(round(coverage_pass.center_coordinate, 3), 0)
            split_lines[round(coverage_pass.center_coordinate, 3)] += 1
        self.assertTrue(any(count >= 2 for count in split_lines.values()))

    def test_composite_builder_bridges_artificial_thin_gaps_in_parallel_sweep_block(self) -> None:
        config = replace(
            _build_visual_test_config(),
            mission=MissionConfig(area_length_x=4.0, area_length_y=12.0, overlap_ratio=0.2, local_control_hz=5.0),
            footprint=CoverageFootprint(length_lf=1.0, width_wf=1.0, eta_cov=0.7),
            safety=SafetyMargins(d_safe=0.0, boundary_margin_x=0.0, boundary_margin_y=0.0),
        )
        cells = [
            FreeSpaceCell("lower", (0.0, 0.0, 4.0, 5.0), [(0.0, 0.0), (4.0, 0.0), (4.0, 5.0), (0.0, 5.0)], (2.0, 2.5), 20.0, "x"),
            FreeSpaceCell("upper", (0.0, 5.4, 4.0, 12.0), [(0.0, 5.4), (4.0, 5.4), (4.0, 12.0), (0.0, 12.0)], (2.0, 8.7), 26.4, "x"),
        ]
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            composite_max_member_cells=8,
            composite_max_region_area_fraction=1.0,
            composite_gap_bridge_factor=0.75,
        )
        composites = build_composite_free_space_regions(cells, config, path_config, obstacle_field=None)
        self.assertEqual(len(composites), 1)
        self.assertEqual(len(composites[0].member_cells), 2)
        self.assertEqual(composites[0].bounds, (0.0, 0.0, 4.0, 12.0))

    def test_merged_composite_pattern_bridges_thin_free_gap_into_long_passes(self) -> None:
        config = replace(
            _build_visual_test_config(),
            mission=MissionConfig(area_length_x=14.0, area_length_y=8.0, overlap_ratio=0.2, local_control_hz=5.0),
            footprint=CoverageFootprint(length_lf=1.0, width_wf=1.0, eta_cov=0.7),
            safety=SafetyMargins(d_safe=0.0, boundary_margin_x=0.0, boundary_margin_y=0.0),
            fleet=replace(_build_visual_test_config().fleet, min_turn_radius=0.5),
        )
        cells = [
            FreeSpaceCell("left", (2.0, 2.0, 6.0, 6.0), [(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)], (4.0, 4.0), 16.0, "x"),
            FreeSpaceCell("right", (6.4, 2.0, 10.4, 6.0), [(6.4, 2.0), (10.4, 2.0), (10.4, 6.0), (6.4, 6.0)], (8.4, 4.0), 16.0, "x"),
        ]
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            coverage_merge_gap_bridge_width_factor=1.0,
        )
        composite = CompositeFreeSpaceRegion(
            region_id="merged_strip",
            bounds=(2.0, 2.0, 10.4, 6.0),
            polygon=[(2.0, 2.0), (10.4, 2.0), (10.4, 6.0), (2.0, 6.0)],
            center=(6.2, 4.0),
            area=32.0,
            preferred_axis="x",
            source_algorithm="unit_test_composite",
            member_cells=cells,
            metadata={"coverage_aware_merged": "true", "agent_task_strip_merge": "true"},
        )

        x_pattern = next(pattern for pattern in generate_region_patterns(composite, config, path_config) if pattern.scan_axis == "x")

        self.assertGreater(int(x_pattern.metadata["composite_gap_bridge_count"]), 0)
        self.assertGreater(float(x_pattern.metadata["composite_gap_bridge_length"]), 0.0)
        self.assertEqual(x_pattern.metadata["pass_obstacle_collision_check_skipped"], "true")
        self.assertGreater(max(coverage_pass.length for coverage_pass in x_pattern.passes), 7.5)

    def test_plain_composite_pattern_does_not_bridge_thin_gap_without_merge_metadata(self) -> None:
        config = replace(
            _build_visual_test_config(),
            mission=MissionConfig(area_length_x=14.0, area_length_y=8.0, overlap_ratio=0.2, local_control_hz=5.0),
            footprint=CoverageFootprint(length_lf=1.0, width_wf=1.0, eta_cov=0.7),
            safety=SafetyMargins(d_safe=0.0, boundary_margin_x=0.0, boundary_margin_y=0.0),
            fleet=replace(_build_visual_test_config().fleet, min_turn_radius=0.5),
        )
        cells = [
            FreeSpaceCell("left", (2.0, 2.0, 6.0, 6.0), [(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)], (4.0, 4.0), 16.0, "x"),
            FreeSpaceCell("right", (6.4, 2.0, 10.4, 6.0), [(6.4, 2.0), (10.4, 2.0), (10.4, 6.0), (6.4, 6.0)], (8.4, 4.0), 16.0, "x"),
        ]
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            coverage_merge_gap_bridge_width_factor=1.0,
        )
        composite = CompositeFreeSpaceRegion(
            region_id="plain_strip",
            bounds=(2.0, 2.0, 10.4, 6.0),
            polygon=[(2.0, 2.0), (10.4, 2.0), (10.4, 6.0), (2.0, 6.0)],
            center=(6.2, 4.0),
            area=32.0,
            preferred_axis="x",
            source_algorithm="unit_test_composite",
            member_cells=cells,
            metadata={"is_composite": "true"},
        )

        x_pattern = next(pattern for pattern in generate_region_patterns(composite, config, path_config) if pattern.scan_axis == "x")

        self.assertEqual(int(x_pattern.metadata["composite_gap_bridge_count"]), 0)
        self.assertEqual(x_pattern.metadata["pass_obstacle_collision_check_skipped"], "true")
        self.assertLessEqual(max(coverage_pass.length for coverage_pass in x_pattern.passes), 4.1)

    def test_open_chain_split_on_single_invalid_uturn(self) -> None:
        config = _open_chain_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_open_sweep_chain_tsp=True,
            enable_uturn_validation_cache=False,
            enable_rmin_aware_chain_order=False,
        )
        barrier = polygon_obstacle("barrier", [(0.0, 2.0), (20.0, 2.0), (20.0, 3.0), (0.0, 3.0)])
        field = ObstacleField(obstacles=[barrier], inflated_obstacles=[barrier])
        pattern = _open_chain_test_pattern(
            [
                (1.0, 1.0, 19.0, 1.0, 0.0),
                (19.0, 4.0, 1.0, 4.0, math.pi),
                (1.0, 5.0, 19.0, 5.0, 0.0),
            ]
        )

        chains, breaks, invalid_passes = _split_pattern_into_open_chains(
            pattern,
            config,
            path_config,
            field,
            start_time=0.0,
            segment_prefix="unit_open_chain",
            uturn_cache={},
            stats={},
            lightweight=False,
        )

        self.assertEqual([len(chain.passes) for chain in chains], [1, 2])
        self.assertEqual(invalid_passes, [])
        self.assertEqual(len(breaks), 1)
        self.assertIsInstance(chains[0], OpenSweepChain)
        self.assertIsInstance(breaks[0], OpenSweepBreak)
        self.assertEqual(breaks[0].before_pass_id, "pass_0")
        self.assertEqual(breaks[0].after_pass_id, "pass_1")
        self.assertIn("uturn_invalid", breaks[0].reason)

    def test_invalid_cover_pass_breaks_and_is_residual_candidate(self) -> None:
        config = _open_chain_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_open_sweep_chain_tsp=True,
            enable_uturn_validation_cache=False,
            enable_rmin_aware_chain_order=False,
        )
        barrier = polygon_obstacle("barrier", [(0.0, 2.0), (20.0, 2.0), (20.0, 3.0), (0.0, 3.0)])
        field = ObstacleField(obstacles=[barrier], inflated_obstacles=[barrier])
        pattern = _open_chain_test_pattern(
            [
                (1.0, 1.0, 19.0, 1.0, 0.0),
                (1.0, 2.5, 19.0, 2.5, 0.0),
                (19.0, 4.0, 1.0, 4.0, math.pi),
            ]
        )

        chains, breaks, invalid_passes = _split_pattern_into_open_chains(
            pattern,
            config,
            path_config,
            field,
            start_time=0.0,
            segment_prefix="unit_invalid_cover",
            uturn_cache={},
            stats={},
            lightweight=False,
        )

        self.assertEqual([len(chain.passes) for chain in chains], [1, 1])
        self.assertEqual([coverage_pass.pass_id for coverage_pass in invalid_passes], ["pass_1"])
        self.assertTrue(any("cover_invalid" in item.reason for item in breaks))

    def test_rmin_aware_chain_order_uses_turn_stride_for_tight_spacing(self) -> None:
        config = PlannerConfig(
            mission=MissionConfig(area_length_x=30.0, area_length_y=12.0, overlap_ratio=0.2, local_control_hz=5.0),
            fleet=FleetConfig(
                initial_states_3dof=[State3DOF(x=2.0, y=1.0, psi=0.0)],
                initial_states_6dof=[],
                cruise_speed=1.2,
                cover_speed=1.0,
                turn_speed_max=0.8,
                max_thrust=4.0,
                max_yaw_moment=4.0,
                min_turn_radius=2.0,
            ),
            footprint=CoverageFootprint(length_lf=4.0, width_wf=2.0, eta_cov=0.7),
            weights=PlannerWeights(),
            safety=SafetyMargins(d_safe=0.0, boundary_margin_x=0.0, boundary_margin_y=0.0),
        )
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_open_sweep_chain_tsp=True,
            enable_uturn_validation_cache=False,
            enable_rmin_aware_chain_order=True,
            rmin_chain_turn_clearance_factor=0.25,
        )
        ys = [1.0 + 1.6 * idx for idx in range(6)]
        lines = []
        for idx, y in enumerate(ys):
            if idx % 2 == 0:
                lines.append((4.0, y, 26.0, y, 0.0))
            else:
                lines.append((26.0, y, 4.0, y, math.pi))
        pattern = _open_chain_test_pattern(lines)

        chains, breaks, invalid_passes = _split_pattern_into_open_chains(
            pattern,
            config,
            path_config,
            obstacle_field=None,
            start_time=0.0,
            segment_prefix="unit_rmin_stride",
            uturn_cache={},
            stats={},
            lightweight=False,
        )

        self.assertEqual([chain.pass_indices for chain in chains], [[0, 3], [1, 4], [2, 5]])
        self.assertEqual(invalid_passes, [])
        self.assertEqual(breaks, [])
        self.assertTrue(all(chain.metadata.get("chain_order_mode") == "rmin_stride" for chain in chains))
        self.assertTrue(all(chain.metadata.get("turn_stride") == "3" for chain in chains))

    def test_static_obstacle_layer_integration(self) -> None:
        config = build_test_config()
        layer = PathPlanningLayer()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_residual_backfill=False,
            enable_residual_local_tsp=False,
        )
        path_plan = layer.plan_from_config(config, static_obstacles=_build_static_obstacles(), path_config=path_config)
        self.assertEqual(path_plan.metadata["static_obstacle_aware"], "true")
        self.assertEqual(path_plan.metadata["static_obstacle_count"], "4")
        self.assertEqual(path_plan.metadata["mapf_scheduler"], "resource_window_cbs_hook")
        self.assertGreater(int(path_plan.metadata["region_count"]), 0)
        self.assertEqual(set(path_plan.agents), set(range(config.fleet.num_agents or 0)))
        self.assertEqual(path_plan.metadata["invalid_path_length"], "0.000000")
        self.assertEqual(path_plan.metadata["invalid_segment_count"], "0")
        self.assertEqual(path_plan.metadata["kinematic_infeasible_segment_count"], "0")

    def test_obstacle_aware_grid_astar_routes_around_blocker(self) -> None:
        config = build_test_config()
        field = normalize_obstacle_field([rectangle_obstacle("wall", center=(24.0, 9.0), width=4.0, height=4.0)], config)
        result = obstacle_aware_grid_astar(
            start=(4.0, 4.0),
            goal=(44.0, 14.0),
            bounds=(0.0, 0.0, config.mission.area_length_x, config.mission.area_length_y),
            obstacle_field=field,
            resolution=2.0,
        )
        self.assertTrue(result.found)
        self.assertFalse(polyline_collides_with_obstacles(result.points, field, inflated=True))

    def test_static_map_json_loads_planner_config_without_usv_assets(self) -> None:
        map_path = ROOT / "maps" / "static_obstacle_map_50x50_simple" / "static_obstacle_map_50x50_simple.json"
        base = _build_config_for_agents(3)
        config, obstacles = load_map_for_planner(map_path, base.fleet)
        self.assertEqual(config.mission.area_length_x, 50.0)
        self.assertEqual(config.mission.area_length_y, 50.0)
        self.assertEqual(config.footprint.length_lf, 4.0)
        self.assertEqual(config.footprint.width_wf, 2.0)
        self.assertEqual(config.fleet.num_agents, 3)
        self.assertEqual(config.fleet.min_turn_radius, 2.0)
        self.assertEqual(len(obstacles), 4)
        self.assertFalse(any(obstacle.kind == "circle" for obstacle in obstacles))
        self.assertEqual(
            build_experiment_output_dir(map_path, config, outputs_root=ROOT / "outputs").name,
            "static_obstacle_map_50x50_simple_usv3_footprint4x2_rmin2",
        )

    def test_dubins_collision_reports_blocked_when_curvature_corridor_is_unavailable(self) -> None:
        config = build_test_config()
        field = normalize_obstacle_field([rectangle_obstacle("blocker", center=(24.0, 9.0), width=4.0, height=4.0)], config)
        segments = build_obstacle_aware_transition_segments(
            segment_id="blocked_transition",
            start=Pose2D(4.0, 4.0, 0.0),
            end=Pose2D(44.0, 14.0, 0.0),
            start_time=0.0,
            config=config,
            path_config=PathPlanningConfig.from_planner_config(config),
            obstacle_field=field,
        )
        self.assertTrue(
            any(
                segment.metadata.get("connector")
                in {"astar_corridor", "smoothed_astar_corridor", "motion_lattice_no_astar", "blocked_dubins_no_astar"}
                for segment in segments
            )
        )
        self.assertTrue(all(segment.curvature_max <= 1.0 / config.fleet.min_turn_radius + 1e-3 for segment in segments))
        self.assertTrue(all(segment.waypoints for segment in segments))
        self.assertTrue(
            all(not path_segment_invalid_reasons(segment, config, field) for segment in segments)
            or any(segment.metadata.get("kinematic_feasible") == "false" for segment in segments)
        )

    def test_boundary_invalid_dubins_falls_back_to_safe_corridor(self) -> None:
        config = _build_visual_test_config()
        config.fleet.min_turn_radius = 4.0
        segments = build_obstacle_aware_transition_segments(
            segment_id="boundary_transition",
            start=Pose2D(1.0, 1.0, 3.141592653589793),
            end=Pose2D(1.0, 8.0, 1.5707963267948966),
            start_time=0.0,
            config=config,
            path_config=PathPlanningConfig.from_planner_config(config),
            obstacle_field=None,
            sample_count=80,
        )
        self.assertTrue(any(segment.metadata.get("direct_invalid_reasons") == "out_of_bounds" for segment in segments))
        self.assertTrue(all(segment.path_source != "astar_corridor_edge" for segment in segments))
        self.assertTrue(any(segment.metadata.get("connector") == "astar_corridor_conversion_failed" for segment in segments))
        self.assertTrue(any(segment.metadata.get("kinematic_feasible") == "false" for segment in segments))

    def test_astar_corridor_converts_sharp_corner_to_trackable_subsegments(self) -> None:
        config = _build_visual_test_config()
        config.fleet.min_turn_radius = 0.5
        path_config = PathPlanningConfig.from_planner_config(config)
        corridor = [(2.0, 2.0), (6.0, 2.0), (6.0, 6.0)]
        segments = _convert_corridor_to_trackable_segments(
            segment_id="corner_corridor",
            corridor_points=corridor,
            start=Pose2D(2.0, 2.0, 0.0),
            end=Pose2D(6.0, 6.0, math.pi / 2.0),
            start_time=0.0,
            config=config,
            path_config=path_config,
            obstacle_field=None,
            kind="transit",
            sample_count=64,
        )
        self.assertIsNotNone(segments)
        assert segments is not None
        self.assertTrue(all(segment.path_source != "astar_corridor_edge" for segment in segments))
        self.assertTrue(all(segment.metadata.get("astar_corridor_conversion_success") == "true" for segment in segments))
        self.assertTrue(validate_transition_sequence(segments, config, obstacle_field=None, retime=True).valid)
        self.assertLessEqual(max(segment.curvature_max for segment in segments), 1.0 / config.fleet.min_turn_radius + 1e-3)

    def test_transition_dynamics_validator_rejects_raw_sharp_corner(self) -> None:
        config = _build_visual_test_config()
        sharp = PathSegmentSpec(
            segment_id="sharp_corner",
            kind="transit",
            source_algorithm="test",
            waypoints=[
                PathWaypoint(x=2.0, y=2.0, psi=0.0, time=0.0, speed=1.0),
                PathWaypoint(x=4.0, y=2.0, psi=0.0, time=2.0, speed=1.0),
                PathWaypoint(x=4.0, y=4.0, psi=math.pi / 2.0, time=4.0, speed=1.0),
            ],
            curvature_max=10.0,
            length=4.0,
            path_source="raw_polyline",
        )
        report = validate_transition_dynamics(sharp, config)
        self.assertFalse(report.valid)
        self.assertIn("curvature_exceeded", report.reasons)
        self.assertEqual(sharp.metadata["dynamic_feasible"], "false")

    def test_transition_dynamics_validator_retimes_high_curvature_dubins(self) -> None:
        config = _build_visual_test_config()
        config.fleet.min_turn_radius = 0.5
        segment = build_obstacle_aware_transition_segments(
            segment_id="tight_but_retimed",
            start=Pose2D(2.0, 2.0, 0.0),
            end=Pose2D(4.0, 3.0, math.pi / 2.0),
            start_time=0.0,
            config=config,
            path_config=PathPlanningConfig.from_planner_config(config),
            obstacle_field=None,
            sample_count=48,
        )[0]
        before_speed = max(waypoint.speed or 0.0 for waypoint in segment.waypoints)
        report = validate_transition_dynamics(segment, config, retime=True)
        after_speed = max(waypoint.speed or 0.0 for waypoint in segment.waypoints)
        self.assertTrue(report.valid, report.reasons)
        self.assertLessEqual(after_speed, before_speed)
        self.assertEqual(segment.metadata["dynamic_feasible"], "true")

    def test_resource_window_scheduler_delays_conflicting_corridor(self) -> None:
        agents = {
            0: AgentPathPlan(
                agent_id=0,
                source_algorithm="test",
                segments=[_resource_segment("a0", 0.0, 5.0, "corridor:shared")],
            ),
            1: AgentPathPlan(
                agent_id=1,
                source_algorithm="test",
                segments=[_resource_segment("a1", 2.0, 4.0, "corridor:shared")],
            ),
        }
        conflicts = apply_resource_window_schedule(agents, separation_time=0.5)
        self.assertEqual(conflicts, 1)
        self.assertGreaterEqual(agents[1].segments[0].waypoints[0].time or 0.0, 5.5)

    def test_shared_resource_metrics_distinguish_reuse_from_true_conflict(self) -> None:
        agents = {
            0: AgentPathPlan(
                agent_id=0,
                source_algorithm="test",
                segments=[_resource_segment("a0", 0.0, 5.0, "corridor:shared")],
            ),
            1: AgentPathPlan(
                agent_id=1,
                source_algorithm="test",
                segments=[_resource_segment("a1", 2.0, 4.0, "corridor:shared")],
            ),
        }
        before = shared_resource_metrics(agents, separation_time=0.5)
        self.assertEqual(int(before["shared_resource_count"]), 1)
        self.assertEqual(int(before["true_time_conflict_count"]), 1)
        apply_resource_window_schedule(agents, separation_time=0.5)
        after = shared_resource_metrics(agents, separation_time=0.5)
        self.assertEqual(int(after["spatial_overlap_reuse_count"]), 1)
        self.assertEqual(int(after["true_time_conflict_count"]), 0)

    def test_repeat_path_overlap_penalty_detects_used_corridor(self) -> None:
        path_config = replace(PathPlanningConfig(), shared_resource_grid_size=0.5)
        existing = [_plain_segment("used", [(0.0, 0.0), (4.0, 0.0)])]
        overlapping = [_plain_segment("overlap", [(1.0, 0.0), (3.0, 0.0)])]
        separate = [_plain_segment("separate", [(1.0, 3.0), (3.0, 3.0)])]
        self.assertGreater(estimate_repeat_overlap_length(overlapping, existing, path_config), 0.0)
        self.assertEqual(estimate_repeat_overlap_length(separate, existing, path_config), 0.0)

    def test_cross_agent_ownership_penalty_prefers_unclaimed_corridor(self) -> None:
        path_config = replace(PathPlanningConfig(), shared_resource_grid_size=1.0, cross_agent_overlap_grid_size=1.0)
        ownership = CoverageOwnershipMap(
            resolution=1.0,
            owner_by_cell={"1_0": 0, "2_0": 0, "3_0": 0},
            region_owner={"left": 0},
        )
        crossing = [_plain_segment("crossing", [(0.0, 0.0), (4.0, 0.0)])]
        separate = [_plain_segment("separate", [(0.0, 2.0), (4.0, 2.0)])]

        crossing_score = score_cross_agent_ownership_overlap(crossing, 1, ownership, path_config, annotate=True)
        separate_score = score_cross_agent_ownership_overlap(separate, 1, ownership, path_config, annotate=True)

        self.assertGreater(crossing_score.overlap_length, 0.0)
        self.assertGreater(crossing_score.penalty, separate_score.penalty)
        self.assertEqual(separate_score.overlap_length, 0.0)
        self.assertGreater(float(crossing[0].metadata["cross_agent_overlap_length"]), 0.0)

    def test_cross_agent_initial_escape_distance_reduces_start_penalty(self) -> None:
        ownership = CoverageOwnershipMap(
            resolution=1.0,
            owner_by_cell={"0_0": 0, "1_0": 0, "2_0": 0, "3_0": 0},
            region_owner={"left": 0},
        )
        no_escape = replace(
            PathPlanningConfig(),
            shared_resource_grid_size=1.0,
            cross_agent_overlap_grid_size=1.0,
            cross_agent_initial_escape_free_distance=0.0,
        )
        with_escape = replace(no_escape, cross_agent_initial_escape_free_distance=2.0)
        segment = [_plain_segment("escape", [(0.0, 0.0), (4.0, 0.0)])]

        full_score = score_cross_agent_ownership_overlap(segment, 1, ownership, no_escape)
        escaped_score = score_cross_agent_ownership_overlap(segment, 1, ownership, with_escape)

        self.assertGreater(full_score.overlap_length, escaped_score.overlap_length)
        self.assertGreater(escaped_score.overlap_length, 0.0)

    def test_performance_summary_reports_ratios_and_constraints(self) -> None:
        config = _build_visual_test_config()
        state = build_coverage_state(config, resolution=1.0)
        state.covered[:, :] = True
        state.coverage_ratio[:, :] = 1.0
        agents = {
            0: AgentPathPlan(
                agent_id=0,
                source_algorithm="test",
                segments=[],
                metrics={"total_length": 10.0, "estimated_time": 5.0},
            ),
            1: AgentPathPlan(
                agent_id=1,
                source_algorithm="test",
                segments=[],
                metrics={"total_length": 20.0, "estimated_time": 15.0},
            ),
        }
        totals = {
            "total_length": 30.0,
            "coverage_length": 12.0,
            "transition_length": 18.0,
            "estimated_time": 20.0,
            "turn_count": 3.0,
            "invalid_path_length": 0.0,
            "out_of_bounds_segment_count": 0.0,
            "obstacle_collision_segment_count": 0.0,
            "kinematic_infeasible_segment_count": 0.0,
            "dynamic_infeasible_segment_count": 0.0,
            "nmpc_untrackable_count": 0.0,
        }
        summary = build_performance_summary(
            agents,
            state,
            totals,
            repeat_overlap_length=6.0,
            path_config=PathPlanningConfig(target_coverage_fraction=0.99),
        )
        self.assertAlmostEqual(summary["coverage_length_ratio"], 0.4)
        self.assertAlmostEqual(summary["transition_length_ratio"], 0.6)
        self.assertAlmostEqual(summary["repeat_transition_ratio"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["turn_count"], 3.0)
        self.assertEqual(summary["agent_work_times"], {"0": 5.0, "1": 15.0})
        self.assertAlmostEqual(summary["mission_makespan"], 15.0)
        self.assertAlmostEqual(summary["total_agent_work_time"], 20.0)
        self.assertAlmostEqual(summary["agent_time_imbalance"], 1.0)
        self.assertGreater(summary["performance_objective"], summary["total_length"])
        self.assertTrue(summary["constraint_ok"])
        self.assertTrue(summary["target_coverage_met"])

    def test_actual_motion_footprint_can_count_transit_coverage(self) -> None:
        config = _build_visual_test_config()
        tour = SingleUsvTourPlan(
            agent_id=0,
            region_order=[],
            selected_patterns={},
            segments=[
                PathSegmentSpec(
                    segment_id="transit_line",
                    kind="transit",
                    source_algorithm="unit_test",
                    waypoints=[
                        PathWaypoint(0.5, 0.5, 0.0, time=0.0),
                        PathWaypoint(config.mission.area_length_x - 0.5, 0.5, 0.0, time=10.0),
                    ],
                    length=config.mission.area_length_x - 1.0,
                )
            ],
        )

        cover_only = evaluate_tour_coverage_state(config, [tour], resolution=1.0)
        actual_footprint = evaluate_tour_coverage_state(
            config,
            [tour],
            resolution=1.0,
            include_non_cover_segments=True,
        )

        self.assertEqual(cover_only.coverage_fraction, 0.0)
        self.assertGreater(actual_footprint.coverage_fraction, cover_only.coverage_fraction)

    def test_residual_local_tsp_visits_multiple_residual_regions(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            max_residual_backfill_regions=2,
            residual_backfill_cycles=1,
            enable_residual_local_tsp=True,
        )
        state = build_coverage_state(config, resolution=1.0)
        state.residual_components = [
            CoverageResidual(0, [(3, 3)], (4.0, 4.0), (3.0, 3.0, 5.0, 5.0)),
            CoverageResidual(1, [(10, 4)], (11.0, 4.0), (10.0, 3.0, 12.0, 5.0)),
        ]
        tours = {
            0: SingleUsvTourPlan(
                agent_id=0,
                region_order=[],
                selected_patterns={},
            )
        }
        result = append_residual_local_tsp(
            config=config,
            path_config=path_config,
            obstacle_field=None,
            tours=tours,
            coverage_state=state,
        )
        self.assertEqual(result.appended_count, 2)
        self.assertEqual(len(tours[0].region_order), 2)
        self.assertEqual(tours[0].diagnostics["residual_local_tsp"], "true")

    def test_residual_local_tsp_filters_low_efficiency_candidates(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            max_residual_backfill_regions=1,
            residual_backfill_cycles=1,
            enable_residual_local_tsp=True,
            residual_min_gain_per_path_meter=10.0,
            residual_filter_after_target_only=False,
        )
        state = build_coverage_state(config, resolution=1.0)
        state.residual_components = [
            CoverageResidual(0, [(3, 3)], (4.0, 4.0), (3.0, 3.0, 5.0, 5.0)),
        ]
        tours = {
            0: SingleUsvTourPlan(
                agent_id=0,
                region_order=[],
                selected_patterns={},
            )
        }

        result = append_residual_local_tsp(
            config=config,
            path_config=path_config,
            obstacle_field=None,
            tours=tours,
            coverage_state=state,
        )

        self.assertEqual(result.appended_count, 0)
        self.assertEqual(result.diagnostics["status"], "low_efficiency_filtered")
        self.assertGreater(int(result.diagnostics["residual_low_efficiency_filtered_count"]), 0)

    def test_residual_local_tsp_soft_keeps_low_efficiency_candidates_before_target(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            max_residual_backfill_regions=1,
            residual_backfill_cycles=1,
            enable_residual_local_tsp=True,
            residual_min_gain_per_path_meter=10.0,
            residual_filter_after_target_only=True,
        )
        state = build_coverage_state(config, resolution=1.0)
        state.residual_components = [
            CoverageResidual(0, [(3, 3)], (4.0, 4.0), (3.0, 3.0, 5.0, 5.0)),
        ]
        tours = {
            0: SingleUsvTourPlan(
                agent_id=0,
                region_order=[],
                selected_patterns={},
            )
        }

        result = append_residual_local_tsp(
            config=config,
            path_config=path_config,
            obstacle_field=None,
            tours=tours,
            coverage_state=state,
        )

        self.assertEqual(result.appended_count, 1)
        self.assertEqual(result.diagnostics["status"], "success")
        self.assertEqual(result.diagnostics["residual_low_efficiency_filtered_count"], "0")
        self.assertGreater(int(result.diagnostics["residual_low_efficiency_soft_count"]), 0)

    def test_pattern_quality_penalty_prefers_less_retraction_turn_and_repeat(self) -> None:
        config = _build_visual_test_config()
        path_config = PathPlanningConfig.from_planner_config(config)
        clean = _manual_pattern(
            "clean",
            (1.0, 1.0, 9.0, 5.0),
            pass_count=2,
            pass_length=8.0,
            estimated_fraction=0.99,
            config=config,
        )
        noisy = replace(
            _manual_pattern(
                "noisy",
                (1.0, 1.0, 9.0, 5.0),
                pass_count=2,
                pass_length=8.0,
                estimated_fraction=0.99,
                config=config,
            ),
            turn_angle=12.0,
            metadata={
                "estimated_region_coverage_fraction": "0.990000",
                "coverage_deficit": "0.000000",
                "total_retraction_length": "4.000000",
                "internal_repeat_overlap_length": "3.000000",
                "internal_repeat_penalty": "36.000000",
                "retraction_failed_count": "1",
                "retraction_extended_count": "1",
            },
        )

        clean_penalty = paper_style_experiment._pattern_quality_penalty(clean, path_config)
        noisy_penalty = paper_style_experiment._pattern_quality_penalty(noisy, path_config)

        self.assertLess(clean_penalty, noisy_penalty)
        self.assertIn("quality_total_quality_penalty", noisy.metadata)
        self.assertLess(
            paper_style_experiment._pattern_sort_key(clean, config, path_config),
            paper_style_experiment._pattern_sort_key(noisy, config, path_config),
        )

    def test_connector_score_components_report_noncover_repeat(self) -> None:
        config = _build_visual_test_config()
        path_config = PathPlanningConfig.from_planner_config(config)
        existing = [_plain_segment("existing", [(0.0, 0.0), (5.0, 0.0)])]
        connector = [_plain_segment("connector", [(0.0, 0.0), (5.0, 0.0)])]
        pattern = _manual_pattern(
            "p",
            (1.0, 1.0, 5.0, 3.0),
            pass_count=1,
            pass_length=4.0,
            estimated_fraction=1.0,
            config=config,
        )
        repeat_score = score_repeat_overlap(
            connector,
            existing,
            path_config,
            penalty_weight=path_config.main_repeat_path_penalty_weight
            * path_config.connector_noncover_repeat_penalty_multiplier,
            annotate=False,
        )

        components = paper_style_experiment._connector_score_components(
            connector,
            repeat_score,
            pattern,
            path_config,
            coverage_deficit=0.0,
        )
        paper_style_experiment._annotate_connector_score_components(connector, components, path_config)

        self.assertGreater(components["connector_noncover_repeat_length"], 0.0)
        self.assertGreater(components["connector_noncover_repeat_penalty"], 0.0)
        self.assertGreater(components["connector_economy_penalty"], components["connector_length"])
        self.assertIn("connector_noncover_repeat_length", connector[0].metadata)
        self.assertIn("connector_economy_penalty", connector[0].metadata)

        straight = _plain_segment("straight", [(0.0, 0.0), (4.0, 0.0)])
        turning = PathSegmentSpec(
            segment_id="turning",
            kind="transit",
            source_algorithm="test",
            waypoints=[
                PathWaypoint(0.0, 0.0, 0.0, 0.0, 1.0),
                PathWaypoint(2.0, 0.0, math.pi / 2.0, 1.0, 1.0),
                PathWaypoint(2.0, 2.0, math.pi / 2.0, 2.0, 1.0),
            ],
            length=4.0,
            path_source="raw_polyline",
        )
        straight_components = paper_style_experiment._connector_score_components(
            [straight],
            score_repeat_overlap([straight], [], path_config, penalty_weight=path_config.main_repeat_path_penalty_weight, annotate=False),
            pattern,
            path_config,
            coverage_deficit=0.0,
        )
        turning_components = paper_style_experiment._connector_score_components(
            [turning],
            score_repeat_overlap([turning], [], path_config, penalty_weight=path_config.main_repeat_path_penalty_weight, annotate=False),
            pattern,
            path_config,
            coverage_deficit=0.0,
        )
        self.assertGreater(
            turning_components["connector_economy_penalty"],
            straight_components["connector_economy_penalty"],
        )

    def test_lightweight_load_swap_moves_boundary_region_when_connected_and_improving(self) -> None:
        regions = {
            "a": _toy_region("a", 0.0, 0.0),
            "b": _toy_region("b", 1.0, 0.0),
            "c": _toy_region("c", 2.0, 0.0),
            "d": _toy_region("d", 3.0, 0.0),
        }
        graph = RegionGraph(
            regions=regions,
            adjacency={"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]},
            node_weights={region_id: 1.0 for region_id in regions},
            edge_weights={},
            edge_metadata={},
            patterns={},
        )
        assignment = BalancedAssignment(
            agent_regions={0: ["a", "b", "c"], 1: ["d"]},
            loads={0: 3.0, 1: 1.0},
            connected={0: True, 1: True},
            imbalance_ratio=1.0,
            objective=4.0,
            diagnostics={},
        )

        swapped = apply_lightweight_load_swap(assignment, graph, max_iterations=4)

        self.assertEqual(swapped.diagnostics["load_swap_count"], "1")
        self.assertEqual(swapped.agent_regions[0], ["a", "b"])
        self.assertEqual(swapped.agent_regions[1], ["c", "d"])
        self.assertLess(swapped.imbalance_ratio, assignment.imbalance_ratio)

    def test_lightweight_load_swap_can_assign_boundary_region_to_idle_agent(self) -> None:
        regions = {
            "a": _toy_region("a", 0.0, 0.0),
            "b": _toy_region("b", 1.0, 0.0),
            "c": _toy_region("c", 2.0, 0.0),
            "d": _toy_region("d", 3.0, 0.0),
        }
        graph = RegionGraph(
            regions=regions,
            adjacency={"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]},
            node_weights={region_id: 1.0 for region_id in regions},
            edge_weights={},
            edge_metadata={},
            patterns={},
        )
        assignment = BalancedAssignment(
            agent_regions={0: ["a", "b", "c"], 1: ["d"], 2: []},
            loads={0: 3.0, 1: 1.0, 2: 0.0},
            connected={0: True, 1: True, 2: True},
            imbalance_ratio=1.0,
            objective=4.0,
            diagnostics={},
        )

        swapped = apply_lightweight_load_swap(assignment, graph, max_iterations=1)

        self.assertEqual(swapped.diagnostics["load_swap_count"], "1")
        self.assertEqual(len(swapped.agent_regions[2]), 1)
        self.assertTrue(graph_is_connected(graph, swapped.agent_regions[0]))
        self.assertTrue(graph_is_connected(graph, swapped.agent_regions[2]))
        self.assertEqual(len(swapped.agent_regions[0]), 2)
        self.assertEqual(swapped.diagnostics["load_swap_candidate_count"], "3")

    def test_joint_assignment_neighbors_preserve_connected_agent_subgraphs(self) -> None:
        config = _build_visual_test_config()
        path_config = PathPlanningConfig.from_planner_config(config)
        regions = {
            "a": _toy_region("a", 0.0, 0.0),
            "b": _toy_region("b", 1.0, 0.0),
            "c": _toy_region("c", 2.0, 0.0),
            "d": _toy_region("d", 3.0, 0.0),
        }
        graph = RegionGraph(
            regions=regions,
            adjacency={"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]},
            node_weights={region_id: 1.0 for region_id in regions},
            edge_weights={},
            edge_metadata={},
            patterns={},
        )

        neighbors = paper_style_experiment._joint_assignment_neighbors(
            {0: ["a", "b", "c"], 1: ["d"]},
            graph,
            config,
            path_config,
            max_candidates=8,
        )

        self.assertTrue(neighbors)
        for _, candidate in neighbors:
            self.assertTrue(all(graph_is_connected(graph, region_ids) for region_ids in candidate.values()))

    def test_joint_assignment_neighbors_include_two_opt_reorder_candidates(self) -> None:
        config = _build_visual_test_config()
        path_config = PathPlanningConfig.from_planner_config(config)
        regions = {
            "a": _toy_region("a", 1.0, 1.0),
            "b": _toy_region("b", 5.0, 1.0),
            "c": _toy_region("c", 1.0, 5.0),
            "d": _toy_region("d", 5.0, 5.0),
        }
        adjacency = {
            region_id: [other for other in regions if other != region_id]
            for region_id in regions
        }
        graph = RegionGraph(
            regions=regions,
            adjacency=adjacency,
            node_weights={region_id: 1.0 for region_id in regions},
            edge_weights={},
            edge_metadata={},
            patterns={},
        )
        original_order = ["a", "d", "c", "b"]
        original_cost = paper_style_experiment._joint_center_route_cost(0, original_order, graph, config, path_config)

        neighbors = paper_style_experiment._joint_assignment_neighbors(
            {0: original_order, 1: []},
            graph,
            config,
            path_config,
            max_candidates=4,
        )
        two_opt_neighbors = [
            candidate[0]
            for operation, candidate in neighbors
            if operation.startswith("reorder_2opt:0")
        ]

        self.assertTrue(two_opt_neighbors)
        self.assertTrue(
            any(
                paper_style_experiment._joint_center_route_cost(0, candidate[0], graph, config, path_config)
                + 1e-9
                < original_cost
                for candidate in two_opt_neighbors
            )
        )

    def test_joint_assignment_neighbors_prioritize_load_reducing_boundary_move(self) -> None:
        config = _build_visual_test_config()
        path_config = PathPlanningConfig.from_planner_config(config)
        regions = {
            "a": _toy_region("a", 0.0, 0.0),
            "b": _toy_region("b", 1.0, 0.0),
            "c": _toy_region("c", 2.0, 0.0),
            "d": _toy_region("d", 3.0, 0.0),
            "e": _toy_region("e", 4.0, 0.0),
        }
        graph = RegionGraph(
            regions=regions,
            adjacency={
                "a": ["b"],
                "b": ["a", "c"],
                "c": ["b", "d"],
                "d": ["c", "e"],
                "e": ["d"],
            },
            node_weights={region_id: 1.0 for region_id in regions},
            edge_weights={},
            edge_metadata={},
            patterns={},
        )
        original = {0: ["a", "b", "c", "d"], 1: ["e"]}
        original_loads = paper_style_experiment._joint_region_loads(original, weights=graph.node_weights)
        original_imbalance = paper_style_experiment._joint_imbalance_ratio(original, original_loads)

        neighbors = paper_style_experiment._joint_assignment_neighbors(
            original,
            graph,
            config,
            path_config,
            max_candidates=1,
        )

        self.assertEqual(len(neighbors), 1)
        operation, moved = neighbors[0]
        self.assertEqual(operation, "load_balance_move:d:0->1")
        self.assertEqual(moved[0], ["a", "b", "c"])
        self.assertIn("d", moved[1])
        self.assertTrue(all(graph_is_connected(graph, region_ids) for region_ids in moved.values()))
        moved_loads = paper_style_experiment._joint_region_loads(moved, weights=graph.node_weights)
        self.assertLess(
            paper_style_experiment._joint_imbalance_ratio(moved, moved_loads),
            original_imbalance,
        )

    def test_joint_assignment_neighbors_include_scan_axis_compatible_boundary_move(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            oriented_sweep_angle_tolerance_deg=5.0,
        )
        regions = {
            "a": _toy_region("a", 0.0, 0.0),
            "b": _toy_region("b", 1.0, 0.0),
            "c": _toy_region("c", 2.0, 0.0),
        }
        base_c_pattern = _manual_pattern(
            "c_y",
            bounds=(2.0, 0.0, 2.5, 0.5),
            pass_count=1,
            pass_length=0.5,
            estimated_fraction=1.0,
            config=config,
            region_id="c",
        )
        c_y_pattern = replace(
            base_c_pattern,
            scan_axis="y",
            passes=[replace(coverage_pass, scan_axis="y") for coverage_pass in base_c_pattern.passes],
        )
        graph = RegionGraph(
            regions=regions,
            adjacency={"a": ["b"], "b": ["a", "c"], "c": ["b"]},
            node_weights={region_id: 1.0 for region_id in regions},
            edge_weights={},
            edge_metadata={},
            patterns={
                "a": [
                    _manual_pattern(
                        "a_x",
                        bounds=(0.0, 0.0, 0.5, 0.5),
                        pass_count=1,
                        pass_length=0.5,
                        estimated_fraction=1.0,
                        config=config,
                        region_id="a",
                    )
                ],
                "b": [
                    _manual_pattern(
                        "b_x",
                        bounds=(1.0, 0.0, 1.5, 0.5),
                        pass_count=1,
                        pass_length=0.5,
                        estimated_fraction=1.0,
                        config=config,
                        region_id="b",
                    )
                ],
                "c": [c_y_pattern],
            },
        )
        original = {0: ["a"], 1: ["b", "c"]}
        original_score = paper_style_experiment._joint_scan_axis_assignment_score(
            original,
            graph,
            config,
            path_config,
        )

        neighbors = paper_style_experiment._joint_assignment_neighbors(
            original,
            graph,
            config,
            path_config,
            max_candidates=3,
        )

        operations = [operation for operation, _ in neighbors]
        self.assertIn("scan_axis_move:b:1->0", operations)
        moved = next(candidate for operation, candidate in neighbors if operation == "scan_axis_move:b:1->0")
        self.assertEqual(moved[0], ["a", "b"])
        self.assertEqual(moved[1], ["c"])
        moved_loads = paper_style_experiment._joint_region_loads(moved, weights=graph.node_weights)
        original_loads = paper_style_experiment._joint_region_loads(original, weights=graph.node_weights)
        self.assertLessEqual(
            paper_style_experiment._joint_imbalance_ratio(moved, moved_loads),
            paper_style_experiment._joint_imbalance_ratio(original, original_loads) + 1e-9,
        )
        self.assertGreater(
            paper_style_experiment._joint_scan_axis_assignment_score(moved, graph, config, path_config),
            original_score,
        )

    def test_joint_center_route_cost_penalizes_scan_axis_switches(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            turn_count_weight=20.0,
            turn_angle_weight=0.0,
            oriented_sweep_angle_tolerance_deg=5.0,
        )
        regions = {
            "a": _toy_region("a", 0.0, 0.0),
            "b": _toy_region("b", 1.0, 0.0),
            "c": _toy_region("c", 2.0, 0.0),
        }
        base_b_pattern = _manual_pattern(
            "b_y",
            bounds=(1.0, 0.0, 1.5, 0.5),
            pass_count=1,
            pass_length=0.5,
            estimated_fraction=1.0,
            config=config,
            region_id="b",
        )
        b_y_pattern = replace(
            base_b_pattern,
            scan_axis="y",
            passes=[replace(coverage_pass, scan_axis="y") for coverage_pass in base_b_pattern.passes],
        )
        graph = RegionGraph(
            regions=regions,
            adjacency={"a": ["b", "c"], "b": ["a", "c"], "c": ["a", "b"]},
            node_weights={region_id: 1.0 for region_id in regions},
            edge_weights={},
            edge_metadata={},
            patterns={
                "a": [
                    _manual_pattern(
                        "a_x",
                        bounds=(0.0, 0.0, 0.5, 0.5),
                        pass_count=1,
                        pass_length=0.5,
                        estimated_fraction=1.0,
                        config=config,
                        region_id="a",
                    )
                ],
                "b": [b_y_pattern],
                "c": [
                    _manual_pattern(
                        "c_x",
                        bounds=(2.0, 0.0, 2.5, 0.5),
                        pass_count=1,
                        pass_length=0.5,
                        estimated_fraction=1.0,
                        config=config,
                        region_id="c",
                    )
                ],
            },
        )

        switching_cost = paper_style_experiment._joint_center_route_cost(
            0,
            ["a", "b", "c"],
            graph,
            config,
            path_config,
        )
        grouped_cost = paper_style_experiment._joint_center_route_cost(
            0,
            ["a", "c", "b"],
            graph,
            config,
            path_config,
        )

        self.assertLess(grouped_cost, switching_cost)

    def test_joint_solution_improves_rejects_skips_and_worse_imbalance(self) -> None:
        current = {"executed_region_count": 3, "skipped_region_count": 0, "load_imbalance": 0.1, "objective": 100.0}
        skipped = {"executed_region_count": 2, "skipped_region_count": 1, "load_imbalance": 0.0, "objective": 10.0}
        imbalanced = {"executed_region_count": 3, "skipped_region_count": 0, "load_imbalance": 0.4, "objective": 10.0}
        improved = {"executed_region_count": 3, "skipped_region_count": 0, "load_imbalance": 0.1, "objective": 90.0}

        self.assertFalse(paper_style_experiment._joint_solution_improves(skipped, current))
        self.assertFalse(paper_style_experiment._joint_solution_improves(imbalanced, current))
        self.assertTrue(paper_style_experiment._joint_solution_improves(improved, current))

    def test_joint_solution_objective_penalizes_makespan_and_time_imbalance(self) -> None:
        path_config = replace(
            PathPlanningConfig(),
            time_weight=2.0,
            load_balance_weight=3.0,
            turn_count_weight=4.0,
            global_noncover_repeat_weight=0.0,
            global_cross_agent_overlap_weight=0.0,
            global_turn_angle_weight=0.0,
        )
        totals = {"transition_length": 10.0, "total_length": 100.0, "total_turn_angle": 0.0, "turn_count": 0.0}
        fast_balanced = paper_style_experiment._joint_solution_objective(
            skipped_count=0,
            executed_count=2,
            totals=totals,
            noncover_repeat=0.0,
            cross_agent_overlap=0.0,
            crossing_count=0,
            load_imbalance=0.0,
            mission_makespan=20.0,
            agent_time_imbalance=0.0,
            report_path_config=path_config,
        )
        slow_balanced = paper_style_experiment._joint_solution_objective(
            skipped_count=0,
            executed_count=2,
            totals=totals,
            noncover_repeat=0.0,
            cross_agent_overlap=0.0,
            crossing_count=0,
            load_imbalance=0.0,
            mission_makespan=30.0,
            agent_time_imbalance=0.0,
            report_path_config=path_config,
        )
        fast_imbalanced = paper_style_experiment._joint_solution_objective(
            skipped_count=0,
            executed_count=2,
            totals=totals,
            noncover_repeat=0.0,
            cross_agent_overlap=0.0,
            crossing_count=0,
            load_imbalance=0.0,
            mission_makespan=20.0,
            agent_time_imbalance=0.5,
            report_path_config=path_config,
        )
        high_turn_count = paper_style_experiment._joint_solution_objective(
            skipped_count=0,
            executed_count=2,
            totals={**totals, "turn_count": 5.0},
            noncover_repeat=0.0,
            cross_agent_overlap=0.0,
            crossing_count=0,
            load_imbalance=0.0,
            mission_makespan=20.0,
            agent_time_imbalance=0.0,
            report_path_config=path_config,
        )

        self.assertGreater(slow_balanced, fast_balanced)
        self.assertGreater(fast_imbalanced, fast_balanced)
        self.assertGreater(high_turn_count, fast_balanced)

    def test_scan_axis_switch_penalty_rewards_continuous_sweep_direction(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            turn_count_weight=5.0,
            turn_angle_weight=1.0,
            oriented_sweep_angle_tolerance_deg=5.0,
        )
        previous = _manual_pattern(
            "previous_x",
            bounds=(0.0, 0.0, 6.0, 4.0),
            pass_count=2,
            pass_length=6.0,
            estimated_fraction=1.0,
            config=config,
            region_id="previous",
        )
        same_axis = _manual_pattern(
            "same_x",
            bounds=(6.0, 0.0, 12.0, 4.0),
            pass_count=2,
            pass_length=6.0,
            estimated_fraction=1.0,
            config=config,
            region_id="same",
        )
        y_axis = replace(
            same_axis,
            pattern_id="switch_y",
            scan_axis="y",
            passes=[replace(coverage_pass, scan_axis="y") for coverage_pass in same_axis.passes],
        )
        near_axis = replace(
            same_axis,
            pattern_id="near_theta",
            scan_axis=f"theta:{math.radians(3.0):.9f}",
            passes=[
                replace(coverage_pass, scan_axis=f"theta:{math.radians(3.0):.9f}")
                for coverage_pass in same_axis.passes
            ],
        )

        self.assertAlmostEqual(
            paper_style_experiment._scan_axis_switch_penalty(previous, same_axis, path_config),
            0.0,
        )
        self.assertAlmostEqual(
            paper_style_experiment._scan_axis_switch_penalty(previous, near_axis, path_config),
            0.0,
        )
        self.assertGreater(
            paper_style_experiment._scan_axis_switch_penalty(previous, y_axis, path_config),
            5.0,
        )

    def test_path_heading_variation_counts_segment_boundary_jump(self) -> None:
        config = _build_visual_test_config()
        first = PathSegmentSpec(
            segment_id="first",
            kind="transit",
            source_algorithm="test",
            waypoints=[
                PathWaypoint(0.5, 2.0, 0.0, 0.0, 1.0),
                PathWaypoint(2.0, 2.0, 0.0, 1.0, 1.0),
            ],
            length=1.5,
            path_source="unit",
        )
        second = PathSegmentSpec(
            segment_id="second",
            kind="transit",
            source_algorithm="test",
            waypoints=[
                PathWaypoint(2.0, 2.0, math.pi / 2.0, 1.0, 1.0),
                PathWaypoint(2.0, 4.0, math.pi / 2.0, 2.0, 1.0),
            ],
            length=2.0,
            path_source="unit",
        )

        self.assertAlmostEqual(paper_style_experiment._segment_heading_variation(first), 0.0)
        self.assertAlmostEqual(paper_style_experiment._segment_heading_variation(second), 0.0)
        self.assertAlmostEqual(
            paper_style_experiment._path_heading_variation([first, second]),
            math.pi / 2.0,
        )
        self.assertEqual(paper_style_experiment._path_turn_count([first, second]), 1)
        metrics = paper_style_experiment._agent_metrics([first, second], config, obstacle_field=None)
        self.assertAlmostEqual(metrics["total_turn_angle"], math.pi / 2.0)
        self.assertEqual(metrics["turn_count"], 1.0)
        self.assertEqual(metrics["turn_segment_count"], 0.0)

    def test_route_refinement_replaces_inefficient_noncover_segment(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            route_refinement_iterations=1,
        )
        inefficient = _plain_segment(
            "zigzag_transit",
            [(0.0, 0.0), (0.0, 2.0), (4.0, 2.0), (4.0, 0.0)],
        )
        agent = AgentPathPlan(agent_id=0, source_algorithm="test", segments=[inefficient])
        tour = SingleUsvTourPlan(agent_id=0, region_order=[], selected_patterns={}, segments=[inefficient])

        diagnostics = paper_style_experiment._refine_global_routes(
            agents={0: agent},
            tours={0: tour},
            config=config,
            path_config=path_config,
            obstacle_field=None,
            baseline_coverage=0.0,
        )

        self.assertEqual(diagnostics["route_refinement_status"], "success")
        self.assertGreater(diagnostics["refined_connector_count"], 0)
        self.assertLess(agent.segments[0].length, inefficient.length)
        self.assertEqual(agent.segments[0].metadata["route_refined"], "true")

    def test_route_refinement_merges_consecutive_noncover_segments(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            route_refinement_iterations=1,
        )
        first = _plain_segment("transit_a", [(0.5, 2.0), (2.0, 4.0), (4.0, 4.0)])
        second = _plain_segment("transit_b", [(4.0, 4.0), (6.0, 4.0), (8.0, 2.0)])
        agent = AgentPathPlan(agent_id=0, source_algorithm="test", segments=[first, second])
        tour = SingleUsvTourPlan(agent_id=0, region_order=[], selected_patterns={}, segments=[first, second])
        original_length = first.length + second.length

        diagnostics = paper_style_experiment._refine_global_routes(
            agents={0: agent},
            tours={0: tour},
            config=config,
            path_config=path_config,
            obstacle_field=None,
            baseline_coverage=0.0,
        )

        self.assertEqual(diagnostics["route_refinement_status"], "success")
        self.assertGreater(diagnostics["merged_noncover_window_count"], 0)
        self.assertEqual(len(agent.segments), 1)
        self.assertLess(agent.segments[0].length, original_length)
        self.assertEqual(agent.segments[0].metadata["route_refined"], "true")
        self.assertEqual(agent.segments[0].metadata["route_refinement_window_size"], "2")
        self.assertEqual(agent.segments[0].metadata["merged_original_segment_ids"], "transit_a,transit_b")

    def test_visual_diagnostics_exports_expected_artifacts(self) -> None:
        config = _build_visual_test_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path_config = replace(
                PathPlanningConfig.from_planner_config(config),
                visual_output_dir=tmpdir,
                visual_map_id="unit_visual_map",
                visual_dpi=80,
                visual_gif_fps=2,
                max_residual_backfill_regions=2,
                residual_backfill_cycles=1,
                tsp_2opt_iterations=0,
            )
            path_plan = PathPlanningLayer().plan_from_config(
                config,
                static_obstacles=[rectangle_obstacle("visual_block", center=(8.0, 5.0), width=1.0, height=1.0)],
                path_config=path_config,
            )
            expected = [
                "00_initial_map.png",
                "01_inflated_obstacles.png",
                "02_free_space_decomposition.png",
                "04_multi_usv_assignment.png",
                "05_agent_0_tsp_route.png",
                "06_agent_0_coverage_passes.png",
                "10_final_multi_usv_path_plan.png",
                "11_coverage_heatmap.png",
                "route_monitor.gif",
                "visualization_manifest.json",
            ]
            for filename in expected:
                artifact = pathlib.Path(tmpdir) / filename
                self.assertTrue(artifact.exists(), filename)
                self.assertGreater(artifact.stat().st_size, 0, filename)
            self.assertEqual(path_plan.metadata["visual_output_dir"], tmpdir)

    def test_full_algorithm_experiment_exports_stage_artifacts_and_report(self) -> None:
        config = _build_visual_test_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path_config = replace(
                PathPlanningConfig.from_planner_config(config),
                visual_dpi=80,
                visual_gif_fps=2,
                max_residual_backfill_regions=2,
                residual_backfill_cycles=1,
                tsp_2opt_iterations=1,
            )
            path_plan, trace = run_planning_algorithm_experiment(
                config=config,
                static_obstacles=[rectangle_obstacle("experiment_block", center=(8.0, 5.0), width=1.0, height=1.0)],
                output_dir=tmpdir,
                path_config=path_config,
                map_id="unit_algorithm_experiment",
            )
            steps_dir = pathlib.Path(tmpdir) / "algorithm_steps"
            expected = [
                "00_map_and_static_obstacles.png",
                "01_obstacle_inflation.png",
                "02_sweep_lines_and_free_cells.png",
                "04_candidate_coverage_patterns.png",
                "06_balanced_assignment.png",
                "07_agent_0_tsp_initial_order.png",
                "08_agent_0_pattern_selection.png",
                "09_agent_0_2opt_iterations.png",
                "11_final_single_usv_tsp_cpp_tours.png",
                "12_algorithm_process.gif",
                "algorithm_experiment_report.json",
            ]
            for filename in expected:
                artifact = steps_dir / filename
                self.assertTrue(artifact.exists(), filename)
                self.assertGreater(artifact.stat().st_size, 0, filename)
            self.assertIn("single_usv_tsp_initial_solution", trace.stage_metrics)
            self.assertIn(0, trace.tsp_records)
            self.assertIn("initial_order", trace.tsp_records[0])
            self.assertIn("pattern_selection", trace.tsp_records[0])
            self.assertEqual(path_plan.metadata["algorithm_experiment_dir"], str(steps_dir))

    def test_paper_style_region_tsp_experiment_uses_region_nodes_not_scan_endpoints(self) -> None:
        config = _build_visual_test_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path_config = replace(
                PathPlanningConfig.from_planner_config(config),
                visual_dpi=60,
                tsp_2opt_iterations=0,
                residual_backfill_cycles=0,
            )
            path_plan, report = run_paper_style_region_tsp_experiment(
                config=config,
                static_obstacles=[],
                output_dir=tmpdir,
                path_config=path_config,
                map_id="unit_paper_style",
            )
            output_dir = pathlib.Path(tmpdir) / "paper_style_region_tsp"
            expected = [
                "04_region_sweep_patterns.png",
                "04_selected_region_sweep_patterns.png",
                "05_region_tsp_nodes.png",
                "06_agent_region_tsp_order.png",
                "07_agent_sweep_endpoints.png",
                "08_final_region_tsp_coverage_path.png",
                "09_constraint_validation.png",
                "10_shared_resource_timeline.png",
                "11_repeat_overlap_diagnostics.png",
                "12_performance_metric_dashboard.png",
                "13_cross_agent_ownership_overlap.png",
                "paper_style_region_tsp_report.json",
            ]
            for filename in expected:
                artifact = output_dir / filename
                self.assertTrue(artifact.exists(), filename)
                self.assertGreater(artifact.stat().st_size, 0, filename)
            self.assertEqual(path_plan.algorithm_name, "paper_style_region_tsp")
            self.assertGreater(int(path_plan.metadata["coverage_endpoint_count"]), int(path_plan.metadata["tsp_node_count"]))
            self.assertEqual(path_plan.metadata["invalid_path_length"], "0.000000")
            self.assertEqual(path_plan.metadata["out_of_bounds_segment_count"], "0")
            self.assertEqual(path_plan.metadata["obstacle_collision_segment_count"], "0")
            self.assertEqual(path_plan.metadata["kinematic_infeasible_segment_count"], "0")
            self.assertEqual(report["tsp_node_count"], int(path_plan.metadata["tsp_node_count"]))
            if report["infeasible_edges"]:
                self.assertLess(report["coverage_fraction"], 1.0)
                self.assertGreater(int(path_plan.metadata["infeasible_edge_count"]), 0)
            else:
                self.assertGreater(report["coverage_fraction"], 0.95)
            report_from_file = json.loads((output_dir / "paper_style_region_tsp_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report_from_file["algorithm"], "paper_style_region_tsp")
            self.assertIn("main_repeat_overlap_length", report_from_file)
            self.assertIn("agent_repeat_overlap", report_from_file)
            self.assertIn("performance_summary", report_from_file)
            self.assertIn("transition_length_ratio", report_from_file["performance_summary"])
            self.assertIn("cross_agent_overlap_length", report_from_file)
            for agent in path_plan.agents.values():
                self.assertGreater(len(agent.segments), 0)
                for segment in agent.segments:
                    self.assertNotEqual(segment.path_source, "astar_corridor_edge")
                    self.assertNotEqual(segment.metadata.get("kinematic_feasible"), "false")
                    self.assertNotEqual(segment.metadata.get("dynamic_feasible"), "false")

    def test_large_map_greedy_continues_after_unreachable_candidate(self) -> None:
        config = _build_config_for_agents(1)
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            region_tsp_branch_limit=2,
            max_residual_backfill_regions=4,
        )
        patterns = {
            f"r{idx}": [
                _manual_pattern(
                    pattern_id=f"p{idx}",
                    bounds=(2.0 + 3.0 * idx, 2.0, 4.0 + 3.0 * idx, 4.0),
                    pass_count=1,
                    pass_length=2.0,
                    estimated_fraction=1.0,
                    config=config,
                    region_id=f"r{idx}",
                )
            ]
            for idx in range(3)
        }

        original_connector = paper_style_experiment._build_region_connector_cached
        original_internal = paper_style_experiment._build_internal_sweep_segments
        original_lookahead = paper_style_experiment._large_map_lookahead_reachable_count

        def fake_connector(*args, **kwargs):
            if kwargs.get("to_region") == "r0":
                kwargs.get("rejection_sink", []).append(
                    {"agent_id": 0, "region_id": "r0", "reason": "unit_unreachable"}
                )
                return None
            return []

        def fake_internal(pattern, *_args, **_kwargs):
            start = pattern.entry_pose
            end = pattern.exit_pose
            return [
                PathSegmentSpec(
                    segment_id=f"unit_{pattern.region_id}",
                    kind="cover",
                    source_algorithm="unit",
                    waypoints=[
                        PathWaypoint(start.x, start.y, start.psi, 0.0, 1.0),
                        PathWaypoint(end.x, end.y, end.psi, 1.0, 1.0),
                    ],
                    length=pattern.coverage_length,
                    metadata={"region_id": pattern.region_id},
                )
            ], ""

        try:
            paper_style_experiment._build_region_connector_cached = fake_connector
            paper_style_experiment._build_internal_sweep_segments = fake_internal
            paper_style_experiment._large_map_lookahead_reachable_count = lambda **_kwargs: 1
            result = paper_style_experiment._solve_agent_region_tsp_large_map_greedy(
                agent_id=0,
                initial_order=["r0", "r1", "r2"],
                patterns=patterns,
                config=config,
                path_config=path_config,
                obstacle_field=None,
                ownership_map=None,
                fallback_solver_metadata=None,
            )
        finally:
            paper_style_experiment._build_region_connector_cached = original_connector
            paper_style_experiment._build_internal_sweep_segments = original_internal
            paper_style_experiment._large_map_lookahead_reachable_count = original_lookahead

        self.assertEqual(result["final_order"], ["r1", "r2"])
        self.assertEqual(result["skipped_region_reasons"]["r0"], "unit_unreachable")

    def test_large_map_greedy_compares_multiple_positive_lookahead_choices(self) -> None:
        config = _build_config_for_agents(1)
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            region_tsp_branch_limit=4,
            large_map_tsp_enable_lookahead_probe=True,
        )
        patterns = {
            "near": [
                _manual_pattern(
                    pattern_id="near",
                    bounds=(2.0, 2.0, 4.0, 4.0),
                    pass_count=1,
                    pass_length=2.0,
                    estimated_fraction=1.0,
                    config=config,
                    region_id="near",
                )
            ],
            "far": [
                _manual_pattern(
                    pattern_id="far",
                    bounds=(9.0, 2.0, 11.0, 4.0),
                    pass_count=1,
                    pass_length=2.0,
                    estimated_fraction=1.0,
                    config=config,
                    region_id="far",
                )
            ],
            "tail_a": [
                _manual_pattern(
                    pattern_id="tail_a",
                    bounds=(12.0, 2.0, 14.0, 4.0),
                    pass_count=1,
                    pass_length=2.0,
                    estimated_fraction=1.0,
                    config=config,
                    region_id="tail_a",
                )
            ],
            "tail_b": [
                _manual_pattern(
                    pattern_id="tail_b",
                    bounds=(15.0, 2.0, 17.0, 4.0),
                    pass_count=1,
                    pass_length=2.0,
                    estimated_fraction=1.0,
                    config=config,
                    region_id="tail_b",
                )
            ],
            "tail_c": [
                _manual_pattern(
                    pattern_id="tail_c",
                    bounds=(18.0, 2.0, 20.0, 4.0),
                    pass_count=1,
                    pass_length=2.0,
                    estimated_fraction=1.0,
                    config=config,
                    region_id="tail_c",
                )
            ],
        }

        original_connector = paper_style_experiment._build_region_connector_cached
        original_internal = paper_style_experiment._build_internal_sweep_segments
        original_lookahead = paper_style_experiment._large_map_lookahead_reachable_count

        def fake_connector(*args, **kwargs):
            start = args[2]
            end = args[3]
            to_region = kwargs.get("to_region", "")
            return [
                PathSegmentSpec(
                    segment_id=f"connector_{to_region}",
                    kind="transit",
                    source_algorithm="unit",
                    waypoints=[
                        PathWaypoint(start.x, start.y, start.psi, 0.0, 1.0),
                        PathWaypoint(end.x, end.y, end.psi, 1.0, 1.0),
                    ],
                    length=math.hypot(end.x - start.x, end.y - start.y),
                    path_source="unit_connector",
                    metadata={"connector": "unit"},
                )
            ]

        def fake_internal(pattern, *_args, **_kwargs):
            start = pattern.entry_pose
            end = pattern.exit_pose
            return [
                PathSegmentSpec(
                    segment_id=f"cover_{pattern.region_id}",
                    kind="cover",
                    source_algorithm="unit",
                    waypoints=[
                        PathWaypoint(start.x, start.y, start.psi, 1.0, 1.0),
                        PathWaypoint(end.x, end.y, end.psi, 2.0, 1.0),
                    ],
                    length=pattern.coverage_length,
                    metadata={"region_id": pattern.region_id},
                )
            ], ""

        def fake_lookahead(**kwargs):
            count = 3 if kwargs["current_pose"].x > 8.0 else 1
            if kwargs.get("return_coverage"):
                return count, float(count * 10.0)
            return count

        try:
            paper_style_experiment._build_region_connector_cached = fake_connector
            paper_style_experiment._build_internal_sweep_segments = fake_internal
            paper_style_experiment._large_map_lookahead_reachable_count = fake_lookahead
            result = paper_style_experiment._solve_agent_region_tsp_large_map_greedy(
                agent_id=0,
                initial_order=["near", "far", "tail_a", "tail_b", "tail_c"],
                patterns=patterns,
                config=config,
                path_config=path_config,
                obstacle_field=None,
                ownership_map=None,
                fallback_solver_metadata=None,
            )
        finally:
            paper_style_experiment._build_region_connector_cached = original_connector
            paper_style_experiment._build_internal_sweep_segments = original_internal
            paper_style_experiment._large_map_lookahead_reachable_count = original_lookahead

        self.assertEqual(result["final_order"][0], "far")
        self.assertGreater(result["tsp_solver_metadata"]["large_map_dead_end_avoidance_count"], 0)

    def test_large_map_greedy_restarts_after_initial_dead_end(self) -> None:
        config = _build_config_for_agents(1)
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            region_tsp_branch_limit=4,
            enable_large_map_dead_end_restart=True,
            large_map_dead_end_restart_limit=1,
            large_map_dead_end_restart_trigger_ratio=0.75,
        )
        patterns = {
            "near": [
                _manual_pattern(
                    pattern_id="near",
                    bounds=(2.0, 2.0, 4.0, 4.0),
                    pass_count=1,
                    pass_length=2.0,
                    estimated_fraction=1.0,
                    config=config,
                    region_id="near",
                )
            ],
            "far": [
                _manual_pattern(
                    pattern_id="far",
                    bounds=(9.0, 2.0, 11.0, 4.0),
                    pass_count=1,
                    pass_length=2.0,
                    estimated_fraction=1.0,
                    config=config,
                    region_id="far",
                )
            ],
            "tail": [
                _manual_pattern(
                    pattern_id="tail",
                    bounds=(12.0, 2.0, 14.0, 4.0),
                    pass_count=1,
                    pass_length=2.0,
                    estimated_fraction=1.0,
                    config=config,
                    region_id="tail",
                )
            ],
        }

        original_connector = paper_style_experiment._build_region_connector_cached
        original_internal = paper_style_experiment._build_internal_sweep_segments
        original_lookahead = paper_style_experiment._large_map_lookahead_reachable_count

        def fake_connector(*args, **kwargs):
            serial = int(args[1])
            start = args[2]
            if serial > 0 and start.x < 5.0:
                kwargs.get("rejection_sink", []).append(
                    {"agent_id": 0, "region_id": kwargs.get("to_region", ""), "reason": "unit_dead_end"}
                )
                return None
            return []

        def fake_internal(pattern, *_args, **_kwargs):
            start = pattern.entry_pose
            end = pattern.exit_pose
            return [
                PathSegmentSpec(
                    segment_id=f"cover_{pattern.region_id}",
                    kind="cover",
                    source_algorithm="unit",
                    waypoints=[
                        PathWaypoint(start.x, start.y, start.psi, 0.0, 1.0),
                        PathWaypoint(end.x, end.y, end.psi, 1.0, 1.0),
                    ],
                    length=pattern.coverage_length,
                    metadata={"region_id": pattern.region_id},
                )
            ], ""

        def fake_lookahead(**kwargs):
            if int(kwargs.get("serial", 0)) <= 0:
                return 1 if kwargs["current_pose"].x < 5.0 else 0
            return 2 if kwargs["current_pose"].x > 8.0 else 0

        try:
            paper_style_experiment._build_region_connector_cached = fake_connector
            paper_style_experiment._build_internal_sweep_segments = fake_internal
            paper_style_experiment._large_map_lookahead_reachable_count = fake_lookahead
            result = paper_style_experiment._solve_agent_region_tsp_large_map_greedy(
                agent_id=0,
                initial_order=["near", "far", "tail"],
                patterns=patterns,
                config=config,
                path_config=path_config,
                obstacle_field=None,
                ownership_map=None,
                fallback_solver_metadata=None,
            )
        finally:
            paper_style_experiment._build_region_connector_cached = original_connector
            paper_style_experiment._build_internal_sweep_segments = original_internal
            paper_style_experiment._large_map_lookahead_reachable_count = original_lookahead

        self.assertEqual(set(result["final_order"]), {"near", "far", "tail"})
        self.assertNotEqual(result["final_order"][0], "near")
        metadata = result["tsp_solver_metadata"]
        self.assertEqual(metadata["large_map_dead_end_restart_attempt_count"], 1)
        self.assertEqual(metadata["large_map_dead_end_restart_accepted_count"], 1)
        self.assertEqual(metadata["large_map_dead_end_restart_forbidden_initial_regions"], ["near"])

    def test_large_map_choice_counts_regions_represented_by_merged_pattern(self) -> None:
        config = _build_config_for_agents(1)
        regular = _manual_pattern(
            pattern_id="regular",
            bounds=(1.0, 1.0, 5.0, 3.0),
            pass_count=1,
            pass_length=4.0,
            estimated_fraction=1.0,
            config=config,
            region_id="regular",
        )
        merged = _manual_pattern(
            pattern_id="merged",
            bounds=(1.0, 1.0, 5.0, 3.0),
            pass_count=1,
            pass_length=4.0,
            estimated_fraction=1.0,
            config=config,
            region_id="merged",
        )
        merged = replace(
            merged,
            metadata={
                **merged.metadata,
                "coverage_aware_merged": "true",
                "merge_source_region_ids": ",".join(f"cell_{idx}" for idx in range(20)),
                "merge_equivalent_source_region_count": "3",
            },
        )
        regular_choice = {
            "score": 0.0,
            "region_id": regular.region_id,
            "pattern": regular,
            "lookahead_reachable": 3,
            "lookahead_coverage_length": 90.0,
            "equivalent_region_count": paper_style_experiment._pattern_equivalent_region_count(regular),
            "execution_coverage_length": 100.0,
        }
        merged_choice = {
            "score": 1000.0,
            "region_id": merged.region_id,
            "pattern": merged,
            "lookahead_reachable": 3,
            "lookahead_coverage_length": 90.0,
            "equivalent_region_count": paper_style_experiment._pattern_equivalent_region_count(merged),
            "execution_coverage_length": 90.0,
        }

        self.assertEqual(regular_choice["equivalent_region_count"], 1)
        self.assertEqual(merged_choice["equivalent_region_count"], 3)
        ordered = sorted(
            [regular_choice, merged_choice],
            key=paper_style_experiment._large_map_feasible_choice_sort_key,
        )
        self.assertIs(ordered[0], merged_choice)

    def test_large_map_greedy_prefers_connector_with_less_cross_agent_overlap(self) -> None:
        config = _build_config_for_agents(1)
        config.fleet.initial_states_3dof[0].y = 2.0
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            region_tsp_branch_limit=4,
            cross_agent_transit_penalty_weight=100.0,
            cross_agent_initial_escape_free_distance=0.0,
        )
        patterns = {
            "bad": [
                _manual_pattern(
                    pattern_id="bad",
                    bounds=(4.0, 1.0, 8.0, 3.0),
                    pass_count=1,
                    pass_length=4.0,
                    estimated_fraction=1.0,
                    config=config,
                    region_id="bad",
                )
            ],
            "good": [
                _manual_pattern(
                    pattern_id="good",
                    bounds=(0.0, 5.0, 4.0, 7.0),
                    pass_count=1,
                    pass_length=4.0,
                    estimated_fraction=1.0,
                    config=config,
                    region_id="good",
                )
            ],
        }
        ownership_map = CoverageOwnershipMap(
            resolution=1.0,
            owner_by_cell={"2_2": 1, "4_2": 1},
            region_owner={"bad": 0, "good": 0},
        )

        original_connector = paper_style_experiment._build_region_connector_cached
        original_internal = paper_style_experiment._build_internal_sweep_segments
        original_lookahead = paper_style_experiment._large_map_lookahead_reachable_count

        def fake_connector(*args, **kwargs):
            start = args[2]
            end = args[3]
            to_region = kwargs.get("to_region", "")
            midpoint = (2.0, 2.0) if to_region == "bad" else (0.0, 4.0)
            return [
                PathSegmentSpec(
                    segment_id=f"connector_{to_region}",
                    kind="transit",
                    source_algorithm="unit",
                    waypoints=[
                        PathWaypoint(start.x, start.y, 0.0, 0.0, 1.0),
                        PathWaypoint(midpoint[0], midpoint[1], 0.0, 1.0, 1.0),
                        PathWaypoint(end.x, end.y, 0.0, 2.0, 1.0),
                    ],
                    length=math.hypot(midpoint[0] - start.x, midpoint[1] - start.y)
                    + math.hypot(end.x - midpoint[0], end.y - midpoint[1]),
                    path_source="unit_connector",
                    metadata={"connector": "unit"},
                )
            ]

        def fake_internal(pattern, *_args, **_kwargs):
            start = pattern.entry_pose
            end = pattern.exit_pose
            return [
                PathSegmentSpec(
                    segment_id=f"cover_{pattern.region_id}",
                    kind="cover",
                    source_algorithm="unit",
                    waypoints=[
                        PathWaypoint(start.x, start.y, start.psi, 2.0, 1.0),
                        PathWaypoint(end.x, end.y, end.psi, 3.0, 1.0),
                    ],
                    length=pattern.coverage_length,
                    metadata={"region_id": pattern.region_id},
                )
            ], ""

        try:
            paper_style_experiment._build_region_connector_cached = fake_connector
            paper_style_experiment._build_internal_sweep_segments = fake_internal
            paper_style_experiment._large_map_lookahead_reachable_count = lambda **_kwargs: 0
            result = paper_style_experiment._solve_agent_region_tsp_large_map_greedy(
                agent_id=0,
                initial_order=["bad", "good"],
                patterns=patterns,
                config=config,
                path_config=path_config,
                obstacle_field=None,
                ownership_map=ownership_map,
                fallback_solver_metadata=None,
            )
        finally:
            paper_style_experiment._build_region_connector_cached = original_connector
            paper_style_experiment._build_internal_sweep_segments = original_internal
            paper_style_experiment._large_map_lookahead_reachable_count = original_lookahead

        self.assertEqual(result["final_order"][0], "good")
        self.assertGreater(result["cross_agent_overlap_length"], 0.0)

    def test_adaptive_pass_retraction_preserves_full_passes_when_turn_is_safe(self) -> None:
        config = _build_visual_test_config()
        config.mission = replace(config.mission, area_length_x=30.0)
        region = DecomposedRegion(
            region_id="center_region",
            bounds=(12.0, 2.0, 18.0, 6.0),
            polygon=[(12.0, 2.0), (18.0, 2.0), (18.0, 6.0), (12.0, 6.0)],
            center=(15.0, 4.0),
            area=24.0,
            preferred_axis="x",
        )
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            max_candidate_axes=1,
            enable_adaptive_pass_retraction=True,
        )

        pattern = generate_region_patterns(region, config, path_config)[0]

        self.assertEqual(pattern.metadata["boundary_retraction_mode"], "adaptive")
        self.assertEqual(pattern.metadata["retracted_pass_count"], "0")
        self.assertAlmostEqual(float(pattern.metadata["total_retraction_length"]), 0.0)
        self.assertTrue(all(abs(item.length - 6.0) <= 1e-6 for item in pattern.passes))

    def test_adaptive_pass_retraction_retracts_boundary_uturn_minimally(self) -> None:
        config = _build_visual_test_config()
        region = DecomposedRegion(
            region_id="left_boundary_region",
            bounds=(0.0, 2.0, 6.0, 8.0),
            polygon=[(0.0, 2.0), (6.0, 2.0), (6.0, 8.0), (0.0, 8.0)],
            center=(3.0, 5.0),
            area=36.0,
            preferred_axis="y",
        )
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            max_candidate_axes=1,
            enable_adaptive_pass_retraction=True,
            retraction_search_iterations=16,
        )

        pattern = generate_region_patterns(region, config, path_config)[0]

        self.assertEqual(pattern.metadata["boundary_retraction_mode"], "adaptive")
        self.assertGreater(int(pattern.metadata["retracted_pass_count"]), 0)
        self.assertGreater(float(pattern.metadata["total_retraction_length"]), 0.0)
        self.assertEqual(pattern.metadata["retraction_failed_count"], "0")
        self.assertTrue(pattern.feasible)
        min_length = max(
            config.footprint.width_wf * 0.25,
            config.footprint.length_lf * path_config.retraction_min_pass_length_factor,
        )
        self.assertTrue(all(item.length + 1e-6 >= min_length for item in pattern.passes))

    def test_agent_profile_separates_sensor_spacing_from_vehicle_hull(self) -> None:
        config = _build_config_for_agents(2)
        config.agent_profiles = {
            0: AgentPlanningProfile(
                agent_id=0,
                coverage_length=8.0,
                coverage_width=6.0,
                overlap_ratio=0.25,
                vehicle_length=3.0,
                vehicle_width=1.0,
                min_turn_radius=4.0,
                cruise_speed=3.0,
                cover_speed=1.5,
                turn_speed_max=1.2,
                max_thrust=4.0,
                max_yaw_moment=2.0,
            )
        }
        agent_config = config.for_agent(0)

        self.assertAlmostEqual(agent_config.footprint.width_wf, 6.0)
        self.assertAlmostEqual(agent_config.mission.overlap_ratio, 0.25)
        self.assertAlmostEqual(config.profile_for_agent(0).effective_strip_spacing, 4.5)
        self.assertEqual(agent_config.vehicle_footprint, VehicleFootprint(length=3.0, width=1.0))
        field = normalize_obstacle_field([], agent_config)
        self.assertAlmostEqual(field.footprint_margin, 0.5)

    def test_heterogeneous_pattern_uses_overlap_spacing_and_agent_cache_identity(self) -> None:
        config = _build_visual_test_config()
        config.agent_profiles = {
            0: AgentPlanningProfile(
                agent_id=0,
                coverage_length=3.0,
                coverage_width=2.0,
                overlap_ratio=0.25,
                vehicle_length=3.0,
                vehicle_width=1.0,
                min_turn_radius=1.0,
                cruise_speed=2.0,
                cover_speed=1.2,
                turn_speed_max=1.0,
                max_thrust=2.0,
                max_yaw_moment=1.0,
            )
        }
        region = DecomposedRegion(
            region_id="spacing_region",
            bounds=(6.0, 1.0, 12.0, 6.0),
            polygon=[(6.0, 1.0), (12.0, 1.0), (12.0, 6.0), (6.0, 6.0)],
            center=(9.0, 3.5),
            area=30.0,
            preferred_axis="x",
        )
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            max_candidate_axes=1,
            enable_oriented_sweep_patterns=False,
        )

        pattern = generate_region_patterns(region, config.for_agent(0), path_config)[0]

        self.assertEqual(len(pattern.passes), 3)
        centers = [coverage_pass.center_coordinate for coverage_pass in pattern.passes]
        self.assertAlmostEqual(centers[1] - centers[0], 1.5)
        self.assertAlmostEqual(centers[2] - centers[1], 1.5)
        self.assertIn("_agent_0", pattern.pattern_id)
        self.assertEqual(pattern.metadata["agent_id"], "0")

    def test_fleet_profile_loader_reads_independent_agent_parameters(self) -> None:
        payload = {
            "fleet_profile_id": "heterogeneous_test",
            "agents": [
                {
                    "agent_id": 0,
                    "initial_state": {"x": 1.0, "y": 2.0, "psi_deg": 90.0},
                    "coverage_footprint": {"length_lf": 6.0, "width_wf": 4.0, "overlap_ratio": 0.2},
                    "vehicle_footprint": {"length": 3.0, "width": 1.2},
                    "motion_constraints": {
                        "min_turn_radius": 3.0,
                        "cruise_speed": 2.5,
                        "cover_speed": 1.1,
                        "turn_speed_max": 0.9,
                        "max_thrust": 3.0,
                        "max_yaw_moment": 1.5,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "fleet.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            fleet, profiles, profile_id = load_fleet_profile_json(path)

        self.assertEqual(profile_id, "heterogeneous_test")
        self.assertEqual(fleet.num_agents, 1)
        self.assertAlmostEqual(fleet.initial_states_3dof[0].psi, math.pi / 2.0)
        self.assertAlmostEqual(profiles[0].effective_strip_spacing, 3.2)
        self.assertAlmostEqual(profiles[0].vehicle_width, 1.2)

    def test_heterogeneous_assignment_respects_feasibility_and_connectivity(self) -> None:
        config = _build_config_for_agents(2)
        config.agent_profiles = {
            agent_id: AgentPlanningProfile(
                agent_id=agent_id,
                coverage_length=4.0,
                coverage_width=3.0 if agent_id == 0 else 1.5,
                overlap_ratio=0.1,
                vehicle_length=3.0,
                vehicle_width=1.0,
                min_turn_radius=3.0 if agent_id == 0 else 1.0,
                cruise_speed=2.0,
                cover_speed=1.4 if agent_id == 0 else 1.0,
                turn_speed_max=1.0,
                max_thrust=2.0,
                max_yaw_moment=1.0,
            )
            for agent_id in range(2)
        }
        regions = {
            region_id: DecomposedRegion(
                region_id=region_id,
                bounds=(float(index), 0.0, float(index + 1), 1.0),
                polygon=[(index, 0.0), (index + 1, 0.0), (index + 1, 1.0), (index, 1.0)],
                center=(index + 0.5, 0.5),
                area=1.0,
                preferred_axis="x",
            )
            for index, region_id in enumerate(("open", "boundary", "narrow"))
        }
        graph = RegionGraph(
            regions=regions,
            adjacency={"open": ["boundary"], "boundary": ["open", "narrow"], "narrow": ["boundary"]},
            node_weights={region_id: 1.0 for region_id in regions},
            edge_weights={("open", "boundary"): 1.0, ("boundary", "narrow"): 1.0},
        )
        agent_patterns = {
            0: {
                "open": [_assignment_test_pattern("open", 0, 2.0)],
                "boundary": [_assignment_test_pattern("boundary", 0, 2.0)],
            },
            1: {
                "boundary": [_assignment_test_pattern("boundary", 1, 3.0)],
                "narrow": [_assignment_test_pattern("narrow", 1, 3.0)],
            },
        }

        assignment = assign_heterogeneous_connected_regions(
            graph,
            config,
            agent_patterns,
            PathPlanningConfig(joint_assignment_iterations=4),
        )

        self.assertIn("open", assignment.agent_regions[0])
        self.assertIn("narrow", assignment.agent_regions[1])
        self.assertEqual(
            {region_id for region_ids in assignment.agent_regions.values() for region_id in region_ids},
            set(regions),
        )
        self.assertTrue(all(assignment.connected.values()))

    def test_heterogeneous_assignment_enforces_agent_mission_time_limit(self) -> None:
        config = _build_config_for_agents(1)
        config.agent_profiles = {
            0: AgentPlanningProfile(
                agent_id=0,
                coverage_length=4.0,
                coverage_width=2.0,
                overlap_ratio=0.0,
                vehicle_length=2.0,
                vehicle_width=1.0,
                min_turn_radius=1.0,
                cruise_speed=2.0,
                cover_speed=1.0,
                turn_speed_max=1.0,
                max_thrust=2.0,
                max_yaw_moment=1.0,
                max_mission_time=2.0,
            )
        }
        regions = {
            region_id: DecomposedRegion(
                region_id=region_id,
                bounds=(float(index), 0.0, float(index + 1), 1.0),
                polygon=[(index, 0.0), (index + 1, 0.0), (index + 1, 1.0), (index, 1.0)],
                center=(index + 0.5, 0.5),
                area=1.0,
                preferred_axis="x",
            )
            for index, region_id in enumerate(("first", "second"))
        }
        graph = RegionGraph(
            regions=regions,
            adjacency={"first": ["second"], "second": ["first"]},
            node_weights={"first": 1.0, "second": 1.0},
            edge_weights={("first", "second"): 1.0},
        )
        patterns = {
            0: {
                region_id: [_assignment_test_pattern(region_id, 0, 1.0)]
                for region_id in regions
            }
        }

        assignment = assign_heterogeneous_connected_regions(
            graph,
            config,
            patterns,
            PathPlanningConfig(joint_assignment_iterations=2),
        )

        self.assertEqual(len(assignment.agent_regions[0]), 1)
        self.assertIn("mission_time_limit", assignment.diagnostics["unassigned_region_reasons"])

    def test_oriented_composite_sweep_uses_true_member_cell_union(self) -> None:
        config = _build_visual_test_config()
        cells = [
            FreeSpaceCell(
                "left",
                (2.0, 2.0, 7.0, 8.0),
                [(2.0, 2.0), (7.0, 2.0), (7.0, 8.0), (2.0, 8.0)],
                (4.5, 5.0),
                30.0,
                "x",
            ),
            FreeSpaceCell(
                "right",
                (7.0, 2.0, 12.0, 8.0),
                [(7.0, 2.0), (12.0, 2.0), (12.0, 8.0), (7.0, 8.0)],
                (9.5, 5.0),
                30.0,
                "x",
            ),
        ]
        region = CompositeFreeSpaceRegion(
            region_id="assigned_union",
            bounds=(2.0, 2.0, 12.0, 8.0),
            polygon=[(2.0, 2.0), (12.0, 2.0), (12.0, 8.0), (2.0, 8.0)],
            center=(7.0, 5.0),
            area=60.0,
            preferred_axis="x",
            member_cells=cells,
            metadata={"coverage_aware_merged": "true", "agent_task_unified_merge": "true"},
        )
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            oriented_sweep_min_area_factor=1.0,
            max_oriented_sweep_angles_per_region=4,
            monotone_merge_angle_tolerance_deg=10.0,
        )

        patterns = generate_region_patterns(region, config, path_config)
        oriented = [pattern for pattern in patterns if pattern.scan_axis.startswith("theta:")]

        self.assertTrue(oriented)
        self.assertTrue(all(pattern.metadata["scan_line_monotone"] == "true" for pattern in oriented))
        self.assertTrue(all(pattern.metadata["oriented_composite_intersections"] == "true" for pattern in oriented))

    def test_heterogeneous_connector_cache_is_profile_scoped(self) -> None:
        config = _build_config_for_agents(2)
        config.agent_profiles = {
            agent_id: AgentPlanningProfile(
                agent_id=agent_id,
                coverage_length=4.0,
                coverage_width=2.0 + agent_id,
                overlap_ratio=0.1,
                vehicle_length=2.0 + agent_id,
                vehicle_width=1.0,
                min_turn_radius=1.0 + agent_id,
                cruise_speed=2.0,
                cover_speed=1.0,
                turn_speed_max=1.0,
                max_thrust=2.0,
                max_yaw_moment=1.0,
            )
            for agent_id in range(2)
        }
        path_config = PathPlanningConfig.from_planner_config(config)
        cache = {}
        start = Pose2D(4.0, 4.0, 0.0)
        end = Pose2D(6.0, 4.0, 0.0)

        for agent_id in range(2):
            paper_style_experiment._build_region_connector_cached(
                agent_id=agent_id,
                serial=0,
                start=start,
                end=end,
                start_time=0.0,
                config=config.for_agent(agent_id),
                path_config=path_config,
                obstacle_field=None,
                to_region="same_region",
                rejection_sink=[],
                allow_obstacle_aware=False,
                cache=cache,
            )

        self.assertEqual(len(cache), 2)
        self.assertEqual({key[0] for key in cache}, {0, 1})
        self.assertEqual(len({key[1] for key in cache}), 2)

    def test_contour_residual_fallback_builds_turn_radius_feasible_chain(self) -> None:
        config = _build_visual_test_config()
        region = DecomposedRegion(
            region_id="compact_residual",
            bounds=(2.0, 1.0, 14.0, 9.0),
            polygon=[(2.0, 1.0), (14.0, 1.0), (14.0, 9.0), (2.0, 9.0)],
            center=(8.0, 5.0),
            area=96.0,
            preferred_axis="x",
        )
        path_config = PathPlanningConfig.from_planner_config(config)

        patterns = residual_planner._contour_residual_patterns(region, config)

        self.assertEqual(len(patterns), 2)
        self.assertTrue(all(pattern.metadata["contour_residual_fallback"] == "true" for pattern in patterns))
        self.assertTrue(
            all(
                paper_style_experiment._validate_internal_sweep(pattern, config, path_config, None)[0]
                for pattern in patterns
            )
        )
        self.assertTrue(
            all(pattern.max_curvature <= 1.0 / config.fleet.min_turn_radius + 1e-9 for pattern in patterns)
        )

    def test_oversized_region_split_preserves_area_without_overlap(self) -> None:
        config = _build_config_for_agents(2)
        config.agent_profiles = {
            agent_id: AgentPlanningProfile(
                agent_id=agent_id,
                coverage_length=4.0,
                coverage_width=2.0,
                overlap_ratio=0.0,
                vehicle_length=3.0,
                vehicle_width=1.0,
                min_turn_radius=1.0,
                cruise_speed=2.0,
                cover_speed=1.0,
                turn_speed_max=1.0,
                max_thrust=2.0,
                max_yaw_moment=1.0,
            )
            for agent_id in range(2)
        }
        region = DecomposedRegion(
            region_id="large",
            bounds=(0.0, 0.0, 10.0, 10.0),
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            center=(5.0, 5.0),
            area=100.0,
            preferred_axis="x",
        )

        children, records = paper_style_experiment._split_oversized_heterogeneous_regions(
            [region],
            config,
            PathPlanningConfig(oversized_region_split_ratio=1.25),
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(len(children), 2)
        self.assertAlmostEqual(sum(child.area for child in children), region.area)
        self.assertLessEqual(children[0].bounds[3], children[1].bounds[1] + 1e-9)

    def test_paper_style_generates_multiple_entry_exit_candidates(self) -> None:
        map_path = ROOT / "maps" / "static_obstacle_map_15x15_rect_triangle_small" / "static_obstacle_map_15x15_rect_triangle_small.json"
        base = _build_config_for_agents(2)
        base.fleet.min_turn_radius = 0.5
        config, obstacles = load_map_for_planner(map_path, base.fleet)
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            residual_backfill_cycles=0,
            max_entry_exit_patterns_per_region=16,
        )
        field = normalize_obstacle_field(obstacles, config, path_config)
        regions = _coarsen_paper_style_regions(decompose_obstacle_aware_area(config, path_config, field), config)
        raw_patterns = _generate_paper_style_patterns(regions, config, path_config, field)
        _, feasible_patterns, _, _ = _build_region_sweep_paths(raw_patterns, config, path_config, field)
        candidate_counts = [len(candidates) for candidates in feasible_patterns.values()]
        self.assertTrue(candidate_counts)
        self.assertGreater(max(candidate_counts), 1)
        self.assertTrue(
            any(
                "entry_exit_variant" in pattern.metadata
                for candidates in feasible_patterns.values()
                for pattern in candidates
            )
        )

    def test_merged_region_preserves_entry_exit_variants_in_large_map_mode(self) -> None:
        base = _build_visual_test_config()
        config = replace(
            base,
            mission=MissionConfig(area_length_x=200.0, area_length_y=200.0, overlap_ratio=0.2, local_control_hz=5.0),
            fleet=replace(base.fleet, min_turn_radius=0.5),
            safety=SafetyMargins(d_safe=0.0, boundary_margin_x=0.0, boundary_margin_y=0.0),
        )
        region = DecomposedRegion(
            region_id="agent0_strip_task_region_0",
            bounds=(10.0, 10.0, 30.0, 20.0),
            polygon=[(10.0, 10.0), (30.0, 10.0), (30.0, 20.0), (10.0, 20.0)],
            center=(20.0, 15.0),
            area=200.0,
            preferred_axis="x",
            metadata={
                "coverage_aware_merged": "true",
                "agent_task_strip_merge": "true",
                "shape_class": "rectangle",
            },
        )
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            large_map_stop_after_first_feasible_sweep_variant=True,
            enable_large_map_sweep_prefilter=True,
            max_prefiltered_patterns_per_region=1,
            max_prefiltered_variants_per_pattern=1,
            max_entry_exit_patterns_per_region=8,
            large_region_connector_pattern_limit=2,
            max_candidate_axes=1,
        )
        raw_patterns = {region.region_id: generate_region_patterns(region, config, path_config)}

        _, feasible_patterns, infeasible_regions, _ = _build_region_sweep_paths(
            raw_patterns,
            config,
            path_config,
            obstacle_field=None,
        )

        self.assertFalse(infeasible_regions)
        variants = feasible_patterns[region.region_id]
        self.assertGreaterEqual(len(variants), 4)
        self.assertTrue(all(pattern.metadata.get("connector_variant_diversity_preserved") == "true" for pattern in variants))
        self.assertGreaterEqual(
            paper_style_experiment._connector_pattern_limit_for_region(region.region_id, variants, path_config),
            4,
        )

    def test_sweep_path_build_reports_no_candidate_patterns_reason(self) -> None:
        config = _build_visual_test_config()
        path_config = PathPlanningConfig.from_planner_config(config)

        _, feasible_patterns, infeasible_regions, _ = _build_region_sweep_paths(
            {"empty_region": []},
            config,
            path_config,
            obstacle_field=None,
        )

        self.assertEqual(feasible_patterns, {})
        self.assertEqual(infeasible_regions[0]["region_id"], "empty_region")
        self.assertEqual(infeasible_regions[0]["reasons"], ["no_candidate_patterns"])

    def test_large_region_compression_requires_high_coverage(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_short_region_compression=True,
            min_compressed_pattern_coverage_fraction=0.98,
            large_map_size_threshold=10.0,
        )
        wide_pattern = _manual_pattern(
            "wide",
            bounds=(0.0, 0.0, 10.0, 10.0),
            pass_count=5,
            pass_length=10.0,
            estimated_fraction=1.0,
            turn_length=500.0,
            config=config,
        )
        self.assertEqual(_short_region_compression_variants(wide_pattern, config, path_config), [])

        narrow_pattern = _manual_pattern(
            "narrow",
            bounds=(0.0, 0.0, 10.0, 2.0),
            pass_count=2,
            pass_length=10.0,
            estimated_fraction=1.0,
            turn_length=500.0,
            config=config,
        )
        compressed = _short_region_compression_variants(narrow_pattern, config, path_config)
        self.assertTrue(compressed)
        self.assertGreaterEqual(max(_estimated_pattern_coverage_fraction(pattern, config) for pattern in compressed), 0.98)

    def test_large_map_prefilter_preserves_high_coverage_candidate(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_large_map_sweep_prefilter=True,
            max_prefiltered_patterns_per_region=1,
        )
        low = _manual_pattern(
            "low",
            bounds=(0.0, 0.0, 10.0, 10.0),
            pass_count=1,
            pass_length=10.0,
            estimated_fraction=0.2,
            estimated_time=1.0,
            config=config,
        )
        high = _manual_pattern(
            "high",
            bounds=(0.0, 0.0, 10.0, 10.0),
            pass_count=5,
            pass_length=10.0,
            estimated_fraction=1.0,
            estimated_time=50.0,
            config=config,
        )
        selected = _prefilter_region_patterns([low, high], config, path_config)
        self.assertEqual([pattern.pattern_id for pattern in selected], ["high"])

    def test_region_graph_weight_uses_high_coverage_workload(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            target_coverage_fraction=0.99,
            min_sweep_pattern_coverage_fraction=0.95,
            coverage_priority_weight=500.0,
            large_map_size_threshold=10.0,
        )
        region = DecomposedRegion(
            region_id="r0",
            bounds=(0.0, 0.0, 10.0, 10.0),
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            center=(5.0, 5.0),
            area=100.0,
            preferred_axis="x",
        )
        low = _manual_pattern("low", region_id="r0", bounds=region.bounds, pass_count=1, pass_length=10.0, estimated_fraction=0.2, estimated_time=1.0, config=config)
        high = _manual_pattern("high", region_id="r0", bounds=region.bounds, pass_count=5, pass_length=10.0, estimated_fraction=1.0, estimated_time=50.0, config=config)
        graph = build_region_graph([region], {"r0": [low, high]}, config, path_config=path_config)
        self.assertAlmostEqual(graph.node_weights["r0"], 50.0)

    def test_paper_style_coarsen_preserves_obstacle_gap(self) -> None:
        config = _build_visual_test_config()
        upper = DecomposedRegion(
            region_id="upper",
            bounds=(0.0, 6.0, 2.0, 10.0),
            polygon=[(0.0, 6.0), (2.0, 6.0), (2.0, 10.0), (0.0, 10.0)],
            center=(1.0, 8.0),
            area=8.0,
            preferred_axis="y",
        )
        lower = DecomposedRegion(
            region_id="lower",
            bounds=(0.0, 0.0, 2.0, 4.0),
            polygon=[(0.0, 0.0), (2.0, 0.0), (2.0, 4.0), (0.0, 4.0)],
            center=(1.0, 2.0),
            area=8.0,
            preferred_axis="y",
        )
        field = normalize_obstacle_field(
            [rectangle_obstacle("gap_obstacle", center=(1.0, 5.0), width=1.0, height=1.0)],
            config,
            PathPlanningConfig.from_planner_config(config),
        )
        diagnostics: dict[str, int] = {}
        coarsened = _coarsen_paper_style_regions([lower, upper], config, obstacle_field=field, diagnostics=diagnostics)
        self.assertEqual(len(coarsened), 2)
        self.assertGreaterEqual(diagnostics.get("obstacle_gap", 0), 1)

    def test_performance_merge_rejects_obstacle_crossing_bbox(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            performance_profile="balanced",
            cell_merge_width_factor=4.0,
        )
        left = DecomposedRegion(
            region_id="left",
            bounds=(0.0, 0.0, 2.0, 10.0),
            polygon=[(0.0, 0.0), (2.0, 0.0), (2.0, 10.0), (0.0, 10.0)],
            center=(1.0, 5.0),
            area=20.0,
            preferred_axis="y",
        )
        right = DecomposedRegion(
            region_id="right",
            bounds=(2.0, 0.0, 4.0, 10.0),
            polygon=[(2.0, 0.0), (4.0, 0.0), (4.0, 10.0), (2.0, 10.0)],
            center=(3.0, 5.0),
            area=20.0,
            preferred_axis="y",
        )
        field = normalize_obstacle_field(
            [rectangle_obstacle("splitter", center=(2.0, 5.0), width=0.5, height=2.0)],
            config,
            path_config,
        )
        diagnostics: dict[str, int] = {}
        merged = _merge_performance_regions([left, right], config, path_config, obstacle_field=field, diagnostics=diagnostics)
        self.assertEqual(len(merged), 2)
        self.assertGreaterEqual(diagnostics.get("obstacle_collision", 0), 1)

    def test_coverage_aware_merge_accepts_economic_rectangle(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            coverage_merge_min_improvement_ratio=0.0,
            coverage_merge_min_coverage_fraction=0.75,
            coverage_merge_max_area_fraction=1.0,
        )
        left = DecomposedRegion(
            region_id="left",
            bounds=(2.0, 1.0, 8.0, 7.0),
            polygon=[(2.0, 1.0), (8.0, 1.0), (8.0, 7.0), (2.0, 7.0)],
            center=(5.0, 4.0),
            area=36.0,
            preferred_axis="x",
        )
        right = DecomposedRegion(
            region_id="right",
            bounds=(8.0, 1.0, 14.0, 7.0),
            polygon=[(8.0, 1.0), (14.0, 1.0), (14.0, 7.0), (8.0, 7.0)],
            center=(11.0, 4.0),
            area=36.0,
            preferred_axis="x",
        )

        diagnostics: dict[str, object] = {}
        merged = _coverage_aware_merge_regions([left, right], config, path_config, diagnostics=diagnostics)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].metadata["coverage_aware_merged"], "true")
        self.assertEqual(merged[0].metadata["shape_class"], "rectangle")
        self.assertEqual(diagnostics["coverage_merge_accepted_count"], 1)

    def test_coverage_aware_merge_budget_fallback_keeps_regions(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            coverage_merge_min_improvement_ratio=0.0,
            coverage_merge_min_coverage_fraction=0.75,
            coverage_merge_max_area_fraction=1.0,
            coverage_merge_max_candidate_evaluations=1,
            coverage_merge_max_validations=10,
            coverage_merge_time_budget_sec=60.0,
        )
        left = DecomposedRegion(
            region_id="left",
            bounds=(2.0, 1.0, 8.0, 7.0),
            polygon=[(2.0, 1.0), (8.0, 1.0), (8.0, 7.0), (2.0, 7.0)],
            center=(5.0, 4.0),
            area=36.0,
            preferred_axis="x",
        )
        right = DecomposedRegion(
            region_id="right",
            bounds=(8.0, 1.0, 14.0, 7.0),
            polygon=[(8.0, 1.0), (14.0, 1.0), (14.0, 7.0), (8.0, 7.0)],
            center=(11.0, 4.0),
            area=36.0,
            preferred_axis="x",
        )

        diagnostics: dict[str, object] = {}
        merged = _coverage_aware_merge_regions([left, right], config, path_config, diagnostics=diagnostics)

        self.assertEqual([region.region_id for region in merged], ["left", "right"])
        self.assertEqual(diagnostics["coverage_merge_status"], "budget_fallback")
        self.assertTrue(diagnostics["coverage_merge_budget_exhausted"])
        self.assertEqual(diagnostics["coverage_merge_budget_reason"], "candidate_budget_exhausted")
        self.assertEqual(diagnostics["coverage_merge_candidate_count"], 1)

    def test_coverage_merge_preview_rewards_removing_region_boundary_turns(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            turn_angle_weight=0.0,
            turn_count_weight=5.0,
            coverage_merge_min_coverage_fraction=0.75,
        )
        left = DecomposedRegion(
            region_id="left",
            bounds=(2.0, 1.0, 8.0, 7.0),
            polygon=[(2.0, 1.0), (8.0, 1.0), (8.0, 7.0), (2.0, 7.0)],
            center=(5.0, 4.0),
            area=36.0,
            preferred_axis="x",
        )
        right = DecomposedRegion(
            region_id="right",
            bounds=(8.0, 1.0, 14.0, 7.0),
            polygon=[(8.0, 1.0), (14.0, 1.0), (14.0, 7.0), (8.0, 7.0)],
            center=(11.0, 4.0),
            area=36.0,
            preferred_axis="x",
        )

        before = paper_style_experiment._coverage_merge_before_preview(
            [left, right],
            config,
            path_config,
            obstacle_field=None,
            preview_cache={},
        )
        vertical = replace(right, region_id="vertical", preferred_axis="y")
        mixed_axis_before = paper_style_experiment._coverage_merge_before_preview(
            [left, vertical],
            config,
            path_config,
            obstacle_field=None,
            preview_cache={},
        )

        self.assertAlmostEqual(before["boundary_turn_proxy"], 8.75)
        self.assertAlmostEqual(mixed_axis_before["boundary_turn_proxy"], 5.0)
        self.assertGreater(before["boundary_turn_proxy"], mixed_axis_before["boundary_turn_proxy"])
        self.assertGreater(before["score"], before["boundary_turn_proxy"])

    def test_coverage_aware_merge_rejects_obstacle_crossing_rectangle_when_composite_disabled(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            coverage_merge_min_improvement_ratio=0.0,
            coverage_merge_max_area_fraction=1.0,
            coverage_merge_allow_nonconvex_composite=False,
        )
        left = DecomposedRegion(
            region_id="left",
            bounds=(2.0, 1.0, 7.0, 8.0),
            polygon=[(2.0, 1.0), (7.0, 1.0), (7.0, 8.0), (2.0, 8.0)],
            center=(4.5, 4.5),
            area=35.0,
            preferred_axis="y",
        )
        right = DecomposedRegion(
            region_id="right",
            bounds=(7.0, 1.0, 12.0, 8.0),
            polygon=[(7.0, 1.0), (12.0, 1.0), (12.0, 8.0), (7.0, 8.0)],
            center=(9.5, 4.5),
            area=35.0,
            preferred_axis="y",
        )
        field = normalize_obstacle_field(
            [rectangle_obstacle("splitter", center=(7.0, 4.5), width=0.8, height=2.0)],
            config,
            path_config,
        )

        diagnostics: dict[str, object] = {}
        merged = _coverage_aware_merge_regions([left, right], config, path_config, obstacle_field=field, diagnostics=diagnostics)

        self.assertEqual(len(merged), 2)
        rejected = diagnostics["coverage_merge_rejected_by_reason"]
        self.assertGreaterEqual(rejected.get("obstacle_collision", 0), 1)
        rejected_candidates = diagnostics["coverage_merge_rejected_candidates"]
        self.assertTrue(rejected_candidates)
        self.assertEqual(rejected_candidates[0]["source_region_ids"], ["left", "right"])
        self.assertGreaterEqual(rejected_candidates[0]["boundary_turn_proxy"], 0.0)

    def test_agent_task_region_merge_combines_same_agent_adjacent_regions(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_agent_task_region_merge=True,
            agent_task_merge_min_improvement_ratio=0.0,
            agent_task_merge_max_area_fraction=1.0,
            coverage_merge_min_coverage_fraction=0.75,
        )
        left = DecomposedRegion(
            region_id="left",
            bounds=(2.0, 1.0, 8.0, 7.0),
            polygon=[(2.0, 1.0), (8.0, 1.0), (8.0, 7.0), (2.0, 7.0)],
            center=(5.0, 4.0),
            area=36.0,
            preferred_axis="x",
        )
        right = DecomposedRegion(
            region_id="right",
            bounds=(8.0, 1.0, 14.0, 7.0),
            polygon=[(8.0, 1.0), (14.0, 1.0), (14.0, 7.0), (8.0, 7.0)],
            center=(11.0, 4.0),
            area=36.0,
            preferred_axis="x",
        )

        merged_regions, merged_assignment, diagnostics = paper_style_experiment._merge_assigned_agent_task_regions(
            [left, right],
            {0: ["left", "right"]},
            config,
            path_config,
            obstacle_field=None,
        )

        self.assertEqual(diagnostics["agent_task_merge_status"], "candidate_ready")
        self.assertEqual(diagnostics["agent_task_merge_accepted_count"], 1)
        self.assertEqual(len(merged_regions), 1)
        self.assertEqual(len(merged_assignment[0]), 1)
        self.assertTrue(merged_assignment[0][0].startswith("agent0_task_merge_region_"))
        self.assertEqual(merged_regions[0].metadata["coverage_aware_merged"], "true")
        self.assertIn("left", merged_regions[0].metadata["merge_source_region_ids"])
        self.assertIn("right", merged_regions[0].metadata["merge_source_region_ids"])

    def test_agent_task_unified_merge_combines_three_regions_into_long_sweep_region(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_agent_task_region_merge=True,
            agent_task_merge_enable_unified_group_merge=True,
            agent_task_merge_min_unified_group_size=3,
            agent_task_merge_min_improvement_ratio=0.0,
            agent_task_merge_max_area_fraction=1.0,
            coverage_merge_min_coverage_fraction=0.75,
        )
        regions = []
        for idx, x0 in enumerate((2.0, 6.0, 10.0)):
            x1 = x0 + 4.0
            regions.append(
                DecomposedRegion(
                    region_id=f"strip_{idx}",
                    bounds=(x0, 1.0, x1, 7.0),
                    polygon=[(x0, 1.0), (x1, 1.0), (x1, 7.0), (x0, 7.0)],
                    center=((x0 + x1) / 2.0, 4.0),
                    area=24.0,
                    preferred_axis="x",
                )
            )

        merged_regions, merged_assignment, diagnostics = paper_style_experiment._merge_assigned_agent_task_regions(
            regions,
            {0: [region.region_id for region in regions]},
            config,
            path_config,
            obstacle_field=None,
        )

        self.assertEqual(diagnostics["agent_task_unified_accepted_count"], 1)
        self.assertEqual(len(merged_regions), 1)
        self.assertEqual(merged_assignment[0], [merged_regions[0].region_id])
        merged = merged_regions[0]
        self.assertEqual(merged.metadata["agent_task_unified_merge"], "true")
        self.assertEqual(merged.metadata["agent_task_unified_source_count"], "3")
        self.assertEqual(merged.bounds, (2.0, 1.0, 14.0, 7.0))
        patterns = generate_region_patterns(merged, config, path_config)
        longest_pass = max(coverage_pass.length for pattern in patterns for coverage_pass in pattern.passes)
        self.assertGreater(longest_pass, 10.0)

    def test_agent_task_unified_merge_prefers_full_component_beyond_window_limit(self) -> None:
        base_config = _build_visual_test_config()
        config = replace(
            base_config,
            mission=replace(base_config.mission, area_length_x=40.0, area_length_y=15.0),
        )
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_agent_task_region_merge=True,
            agent_task_merge_enable_unified_group_merge=True,
            agent_task_merge_min_unified_group_size=3,
            agent_task_merge_max_unified_group_size=4,
            agent_task_merge_prefer_full_components=True,
            agent_task_merge_full_component_max_regions=8,
            agent_task_merge_min_improvement_ratio=0.0,
            agent_task_merge_max_area_fraction=1.0,
            coverage_merge_max_area_fraction=1.0,
            coverage_merge_min_coverage_fraction=0.75,
        )
        regions = []
        for idx in range(6):
            x0 = 2.0 + 4.0 * idx
            x1 = x0 + 4.0
            regions.append(
                DecomposedRegion(
                    region_id=f"long_agent_strip_{idx}",
                    bounds=(x0, 1.0, x1, 7.0),
                    polygon=[(x0, 1.0), (x1, 1.0), (x1, 7.0), (x0, 7.0)],
                    center=((x0 + x1) / 2.0, 4.0),
                    area=24.0,
                    preferred_axis="x",
                )
            )

        merged_regions, merged_assignment, diagnostics = paper_style_experiment._merge_assigned_agent_task_regions(
            regions,
            {0: [region.region_id for region in regions]},
            config,
            path_config,
            obstacle_field=None,
        )

        self.assertEqual(diagnostics["agent_task_unified_accepted_count"], 1)
        self.assertEqual(diagnostics["agent_task_full_component_accepted_count"], 1)
        self.assertEqual(len(merged_regions), 1)
        self.assertEqual(merged_assignment[0], [merged_regions[0].region_id])
        merged = merged_regions[0]
        self.assertEqual(merged.metadata["agent_task_unified_merge"], "true")
        self.assertEqual(merged.metadata["agent_task_full_component_merge"], "true")
        self.assertEqual(merged.metadata["agent_task_unified_source_count"], "6")
        self.assertEqual(merged.bounds, (2.0, 1.0, 26.0, 7.0))
        patterns = generate_region_patterns(merged, config, path_config)
        longest_pass = max(coverage_pass.length for pattern in patterns for coverage_pass in pattern.passes)
        self.assertGreater(longest_pass, 22.0)

    def test_agent_task_strip_sort_deprioritizes_fragmented_oversized_component(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            agent_task_merge_max_unified_group_size=4,
            agent_task_strip_merge_min_rectangularity=0.55,
        )

        def region(region_id: str, x0: float, y0: float) -> DecomposedRegion:
            return DecomposedRegion(
                region_id=region_id,
                bounds=(x0, y0, x0 + 4.0, y0 + 4.0),
                polygon=[(x0, y0), (x0 + 4.0, y0), (x0 + 4.0, y0 + 4.0), (x0, y0 + 4.0)],
                center=(x0 + 2.0, y0 + 2.0),
                area=16.0,
                preferred_axis="x",
            )

        aligned = [region(f"aligned_{idx}", 4.0 * idx, 0.0) for idx in range(6)]
        fragmented = [
            region("l_0", 0.0, 0.0),
            region("l_1", 4.0, 0.0),
            region("l_2", 8.0, 0.0),
            region("l_3", 12.0, 0.0),
            region("l_4", 0.0, 4.0),
            region("l_5", 0.0, 8.0),
        ]

        aligned_full_key = paper_style_experiment._agent_task_strip_group_sort_key(aligned, path_config)
        aligned_window_key = paper_style_experiment._agent_task_strip_group_sort_key(aligned[:4], path_config)
        fragmented_full_key = paper_style_experiment._agent_task_strip_group_sort_key(fragmented, path_config)
        fragmented_window_key = paper_style_experiment._agent_task_strip_group_sort_key(fragmented[:4], path_config)

        self.assertLess(aligned_full_key, aligned_window_key)
        self.assertLess(fragmented_window_key, fragmented_full_key)

    def test_agent_task_strip_generates_subwindows_for_fragmented_small_component(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            agent_task_merge_max_unified_group_size=6,
            agent_task_strip_merge_min_rectangularity=0.55,
            agent_task_strip_full_component_direct_priority_rectangularity=0.95,
        )

        def region(region_id: str, x0: float, y0: float) -> DecomposedRegion:
            return DecomposedRegion(
                region_id=region_id,
                bounds=(x0, y0, x0 + 4.0, y0 + 4.0),
                polygon=[(x0, y0), (x0 + 4.0, y0), (x0 + 4.0, y0 + 4.0), (x0, y0 + 4.0)],
                center=(x0 + 2.0, y0 + 2.0),
                area=16.0,
                preferred_axis="x",
            )

        fragmented = [
            region("a", 0.0, 0.0),
            region("b", 4.0, 0.0),
            region("c", 8.0, 0.0),
            region("d", 0.0, 4.0),
        ]
        original_components = paper_style_experiment._agent_task_merge_connected_components
        original_axis_components = paper_style_experiment._agent_task_axis_compatible_components
        try:
            paper_style_experiment._agent_task_merge_connected_components = lambda *_args: [fragmented]
            paper_style_experiment._agent_task_axis_compatible_components = lambda *_args: [fragmented]
            groups = paper_style_experiment._agent_task_strip_candidate_groups(
                fragmented,
                config,
                path_config,
                obstacle_field=None,
            )
        finally:
            paper_style_experiment._agent_task_merge_connected_components = original_components
            paper_style_experiment._agent_task_axis_compatible_components = original_axis_components

        group_ids = {tuple(sorted(item.region_id for item in group)) for group in groups}
        self.assertIn(("a", "b", "c", "d"), group_ids)
        self.assertIn(("a", "b", "c"), group_ids)
        self.assertIn(("a", "b"), group_ids)
        ordered = sorted(groups, key=lambda group: paper_style_experiment._agent_task_strip_group_sort_key(group, path_config))
        self.assertLess(len(ordered[0]), len(fragmented))

    def test_coverage_merge_accumulates_equivalent_source_region_count(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            coverage_merge_max_area_fraction=1.0,
        )

        def region(region_id: str, x0: float, equivalent_count: int = 1) -> DecomposedRegion:
            metadata = {}
            if equivalent_count > 1:
                metadata = {
                    "coverage_aware_merged": "true",
                    "merge_equivalent_source_region_count": str(equivalent_count),
                }
            return DecomposedRegion(
                region_id=region_id,
                bounds=(x0, 1.0, x0 + 2.0, 3.0),
                polygon=[(x0, 1.0), (x0 + 2.0, 1.0), (x0 + 2.0, 3.0), (x0, 3.0)],
                center=(x0 + 1.0, 2.0),
                area=4.0,
                preferred_axis="x",
                metadata=metadata,
            )

        candidate, reason = paper_style_experiment._coverage_merge_candidate_from_group(
            0,
            [region("already_merged", 1.0, equivalent_count=2), region("base", 3.0)],
            config,
            path_config,
            obstacle_field=None,
        )

        self.assertEqual(reason, "")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.metadata["merge_equivalent_source_region_count"], "3")
        patterns = generate_region_patterns(candidate, config, path_config)
        self.assertTrue(patterns)
        self.assertTrue(
            all(
                paper_style_experiment._pattern_equivalent_region_count(pattern) == 3
                for pattern in patterns
            )
        )

    def test_agent_task_unified_merge_accepts_axis_compatible_subgroup(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_agent_task_region_merge=True,
            agent_task_merge_enable_unified_group_merge=True,
            agent_task_merge_min_unified_group_size=3,
            agent_task_merge_min_improvement_ratio=0.0,
            agent_task_merge_max_area_fraction=1.0,
            coverage_merge_min_coverage_fraction=0.70,
            coverage_merge_allow_nonconvex_composite=False,
        )
        regions = []
        for idx, x0 in enumerate((1.0, 5.0, 9.0)):
            x1 = x0 + 4.0
            regions.append(
                DecomposedRegion(
                    region_id=f"main_strip_{idx}",
                    bounds=(x0, 1.0, x1, 5.0),
                    polygon=[(x0, 1.0), (x1, 1.0), (x1, 5.0), (x0, 5.0)],
                    center=((x0 + x1) / 2.0, 3.0),
                    area=16.0,
                    preferred_axis="x",
                )
            )
        regions.append(
            DecomposedRegion(
                region_id="side_branch",
                bounds=(5.0, 5.0, 9.0, 8.0),
                polygon=[(5.0, 5.0), (9.0, 5.0), (9.0, 8.0), (5.0, 8.0)],
                center=(7.0, 6.5),
                area=12.0,
                preferred_axis="y",
            )
        )

        merged_regions, merged_assignment, diagnostics = paper_style_experiment._merge_assigned_agent_task_regions(
            regions,
            {0: [region.region_id for region in regions]},
            config,
            path_config,
            obstacle_field=None,
        )

        unified = [region for region in merged_regions if region.metadata.get("agent_task_unified_merge") == "true"]
        self.assertEqual(diagnostics["agent_task_unified_accepted_count"], 1)
        self.assertEqual(len(unified), 1)
        self.assertIn("side_branch", merged_assignment[0])
        self.assertEqual(unified[0].metadata["agent_task_unified_source_count"], "3")
        self.assertNotIn("side_branch", unified[0].metadata["agent_task_unified_source_ids"])
        patterns = generate_region_patterns(unified[0], config, path_config)
        longest_pass = max(coverage_pass.length for pattern in patterns for coverage_pass in pattern.passes)
        self.assertGreater(longest_pass, 10.0)

    def test_lightweight_strip_merge_runs_without_full_agent_task_merge(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_agent_task_region_merge=False,
            enable_agent_task_lightweight_strip_merge=True,
            agent_task_strip_merge_max_groups_per_agent=2,
            agent_task_strip_merge_min_rectangularity=0.80,
            agent_task_strip_merge_min_length_gain_factor=1.10,
            agent_task_merge_min_improvement_ratio=0.0,
            agent_task_merge_max_area_fraction=1.0,
            coverage_merge_max_area_fraction=1.0,
            coverage_merge_min_coverage_fraction=0.70,
            agent_task_merge_max_unified_group_size=4,
        )
        regions = []
        for idx, x0 in enumerate((2.0, 6.0, 10.0)):
            x1 = x0 + 4.0
            regions.append(
                DecomposedRegion(
                    region_id=f"strip_{idx}",
                    bounds=(x0, 1.0, x1, 7.0),
                    polygon=[(x0, 1.0), (x1, 1.0), (x1, 7.0), (x0, 7.0)],
                    center=((x0 + x1) / 2.0, 4.0),
                    area=24.0,
                    preferred_axis="x",
                )
            )

        merged_regions, merged_assignment, diagnostics = paper_style_experiment._merge_assigned_agent_task_regions(
            regions,
            {0: [region.region_id for region in regions]},
            config,
            path_config,
            obstacle_field=None,
        )

        self.assertEqual(diagnostics["agent_task_merge_status"], "candidate_ready")
        self.assertEqual(diagnostics["agent_task_strip_accepted_count"], 1)
        self.assertEqual(diagnostics["agent_task_unified_accepted_count"], 0)
        self.assertEqual(len(merged_regions), 1)
        self.assertEqual(merged_assignment[0], [merged_regions[0].region_id])
        merged = merged_regions[0]
        self.assertEqual(merged.metadata["agent_task_strip_merge"], "true")
        self.assertEqual(merged.metadata["agent_task_strip_source_count"], "3")
        self.assertGreater(float(merged.metadata["merge_long_pass_gain_ratio"]), 1.10)

    def test_lightweight_strip_merge_accepts_nonrectangular_composite_gain(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_agent_task_region_merge=False,
            enable_agent_task_lightweight_strip_merge=True,
            agent_task_strip_merge_max_groups_per_agent=2,
            agent_task_strip_merge_min_rectangularity=0.60,
            agent_task_strip_merge_min_length_gain_factor=1.20,
            agent_task_merge_min_improvement_ratio=0.0,
            agent_task_merge_max_area_fraction=1.0,
            coverage_merge_max_area_fraction=1.0,
            coverage_merge_min_coverage_fraction=0.70,
            coverage_merge_allow_nonconvex_composite=True,
            agent_task_merge_max_unified_group_size=4,
        )
        left_column = DecomposedRegion(
            region_id="left_column",
            bounds=(0.0, 0.0, 4.0, 8.0),
            polygon=[(0.0, 0.0), (4.0, 0.0), (4.0, 8.0), (0.0, 8.0)],
            center=(2.0, 4.0),
            area=32.0,
            preferred_axis="x",
        )
        lower_bar = DecomposedRegion(
            region_id="lower_bar",
            bounds=(4.0, 0.0, 12.0, 4.0),
            polygon=[(4.0, 0.0), (12.0, 0.0), (12.0, 4.0), (4.0, 4.0)],
            center=(8.0, 2.0),
            area=32.0,
            preferred_axis="x",
        )

        merged_regions, merged_assignment, diagnostics = paper_style_experiment._merge_assigned_agent_task_regions(
            [left_column, lower_bar],
            {0: ["left_column", "lower_bar"]},
            config,
            path_config,
            obstacle_field=None,
        )

        self.assertEqual(diagnostics["agent_task_merge_status"], "candidate_ready")
        self.assertEqual(diagnostics["agent_task_strip_accepted_count"], 1)
        self.assertEqual(len(merged_regions), 1)
        merged = merged_regions[0]
        self.assertEqual(merged_assignment[0], [merged.region_id])
        self.assertEqual(merged.metadata["agent_task_strip_merge"], "true")
        self.assertNotEqual(merged.metadata["shape_class"], "rectangle")
        self.assertEqual(merged.metadata["scan_support_mode"], "member_cell_intervals")
        self.assertEqual(merged.metadata["merge_accept_reason"], "agent_task_strip_composite_boustrophedon_gain")
        self.assertGreater(float(merged.metadata["merge_long_pass_gain_ratio"]), 1.20)
        patterns = generate_region_patterns(merged, config, path_config)
        longest_pass = max(coverage_pass.length for pattern in patterns for coverage_pass in pattern.passes)
        self.assertGreater(longest_pass, 9.5)

    def test_infeasible_agent_task_merge_candidate_splits_back_to_sources(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_agent_task_region_merge=False,
            enable_agent_task_lightweight_strip_merge=True,
            agent_task_strip_merge_min_rectangularity=0.60,
            agent_task_strip_merge_min_length_gain_factor=1.20,
            agent_task_merge_max_area_fraction=1.0,
            coverage_merge_max_area_fraction=1.0,
            coverage_merge_min_coverage_fraction=0.70,
            coverage_merge_allow_nonconvex_composite=True,
        )
        left_column = DecomposedRegion(
            region_id="left_column",
            bounds=(0.0, 0.0, 4.0, 8.0),
            polygon=[(0.0, 0.0), (4.0, 0.0), (4.0, 8.0), (0.0, 8.0)],
            center=(2.0, 4.0),
            area=32.0,
            preferred_axis="x",
        )
        lower_bar = DecomposedRegion(
            region_id="lower_bar",
            bounds=(4.0, 0.0, 12.0, 4.0),
            polygon=[(4.0, 0.0), (12.0, 0.0), (12.0, 4.0), (4.0, 4.0)],
            center=(8.0, 2.0),
            area=32.0,
            preferred_axis="x",
        )
        merged_regions, merged_assignment, diagnostics = paper_style_experiment._merge_assigned_agent_task_regions(
            [left_column, lower_bar],
            {0: ["left_column", "lower_bar"]},
            config,
            path_config,
            obstacle_field=None,
        )
        merged_id = merged_regions[0].region_id

        repaired_regions, repaired_assignment, split_count = paper_style_experiment._split_infeasible_agent_task_merge_candidates(
            merged_regions,
            merged_assignment,
            [left_column, lower_bar],
            [merged_id],
        )

        self.assertEqual(diagnostics["agent_task_strip_accepted_count"], 1)
        self.assertEqual(split_count, 1)
        self.assertEqual([region.region_id for region in repaired_regions], ["left_column", "lower_bar"])
        self.assertEqual(repaired_assignment[0], ["left_column", "lower_bar"])

    def test_runtime_source_fallback_expands_skipped_agent_task_merge(self) -> None:
        config = _build_visual_test_config()
        source_a = DecomposedRegion(
            region_id="source_a",
            bounds=(0.0, 0.0, 4.0, 4.0),
            polygon=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            center=(2.0, 2.0),
            area=16.0,
            preferred_axis="x",
        )
        source_b = DecomposedRegion(
            region_id="source_b",
            bounds=(4.0, 0.0, 8.0, 4.0),
            polygon=[(4.0, 0.0), (8.0, 0.0), (8.0, 4.0), (4.0, 4.0)],
            center=(6.0, 2.0),
            area=16.0,
            preferred_axis="x",
        )
        merged = DecomposedRegion(
            region_id="agent0_strip_task_region_0",
            bounds=(0.0, 0.0, 8.0, 4.0),
            polygon=[(0.0, 0.0), (8.0, 0.0), (8.0, 4.0), (0.0, 4.0)],
            center=(4.0, 2.0),
            area=32.0,
            preferred_axis="x",
            metadata={
                "agent_task_strip_merge": "true",
                "merge_fallback_source_ids": "source_a,source_b",
            },
        )
        base_patterns = {
            "source_a": [
                _manual_pattern(
                    "source_a_pattern",
                    source_a.bounds,
                    pass_count=1,
                    pass_length=4.0,
                    estimated_fraction=1.0,
                    config=config,
                    region_id="source_a",
                )
            ],
            "source_b": [
                _manual_pattern(
                    "source_b_pattern",
                    source_b.bounds,
                    pass_count=1,
                    pass_length=4.0,
                    estimated_fraction=1.0,
                    config=config,
                    region_id="source_b",
                )
            ],
        }

        expanded, records = paper_style_experiment._expand_skipped_agent_task_merge_assignments(
            {0: ["kept_region", "agent0_strip_task_region_0"]},
            {0: {"skipped_regions": ["agent0_strip_task_region_0"]}},
            [merged],
            base_patterns,
        )

        self.assertEqual(expanded[0], ["kept_region", "source_a", "source_b"])
        self.assertEqual(records[0]["status"], "candidate")
        self.assertEqual(records[0]["source_region_ids"], ["source_a", "source_b"])

    def test_runtime_source_fallback_accepts_neutral_source_expansion(self) -> None:
        config = _build_visual_test_config()
        path_config = PathPlanningConfig.from_planner_config(config)
        source_a = DecomposedRegion(
            region_id="source_a",
            bounds=(0.0, 0.0, 4.0, 4.0),
            polygon=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            center=(2.0, 2.0),
            area=16.0,
            preferred_axis="x",
        )
        source_b = DecomposedRegion(
            region_id="source_b",
            bounds=(4.0, 0.0, 8.0, 4.0),
            polygon=[(4.0, 0.0), (8.0, 0.0), (8.0, 4.0), (4.0, 4.0)],
            center=(6.0, 2.0),
            area=16.0,
            preferred_axis="x",
        )
        merged = DecomposedRegion(
            region_id="agent0_strip_task_region_0",
            bounds=(0.0, 0.0, 8.0, 4.0),
            polygon=[(0.0, 0.0), (8.0, 0.0), (8.0, 4.0), (0.0, 4.0)],
            center=(4.0, 2.0),
            area=32.0,
            preferred_axis="x",
            metadata={
                "agent_task_strip_merge": "true",
                "merge_fallback_source_ids": "source_a,source_b",
            },
        )
        merged_pattern = _manual_pattern(
            "merged_pattern",
            merged.bounds,
            pass_count=1,
            pass_length=4.0,
            estimated_fraction=1.0,
            config=config,
            region_id=merged.region_id,
        )
        source_pattern = _manual_pattern(
            "source_a_pattern",
            source_a.bounds,
            pass_count=1,
            pass_length=8.0,
            estimated_fraction=1.0,
            config=config,
            region_id=source_a.region_id,
        )
        source_b_pattern = _manual_pattern(
            "source_b_pattern",
            source_b.bounds,
            pass_count=1,
            pass_length=4.0,
            estimated_fraction=1.0,
            config=config,
            region_id=source_b.region_id,
        )

        def sweep(pattern: RegionCoveragePattern) -> RegionSweepPath:
            return RegionSweepPath(
                region_id=pattern.region_id,
                pattern_id=pattern.pattern_id,
                passes=list(pattern.passes),
                endpoints=[pattern.entry_pose, pattern.exit_pose],
                entry_pose=pattern.entry_pose,
                exit_pose=pattern.exit_pose,
            )

        old_solver = paper_style_experiment._solve_agent_region_tsp

        def fake_solver(*args, **kwargs):
            region_ids = list(args[1])
            self.assertEqual(region_ids, ["source_a", "source_b"])
            return {
                "initial_order": region_ids,
                "final_order": ["source_a"],
                "selected_patterns": {"source_a": source_pattern},
                "selected_pattern_ids": {"source_a": source_pattern.pattern_id},
                "segments": [],
                "tsp_solver_metadata": {"tsp_solver_status": "success"},
            }

        try:
            paper_style_experiment._solve_agent_region_tsp = fake_solver
            (
                final_regions,
                final_patterns,
                _final_sweeps,
                _final_templates,
                final_assignment,
                _final_ownership,
                diagnostics,
            ) = paper_style_experiment._apply_agent_task_merge_runtime_source_fallback(
                config=config,
                path_config=path_config,
                obstacle_field=None,
                current_regions=[merged],
                current_feasible_patterns={merged.region_id: [merged_pattern]},
                current_sweep_paths={merged.region_id: sweep(merged_pattern)},
                current_sweep_segment_templates={},
                base_regions=[source_a, source_b],
                base_feasible_patterns={
                    source_a.region_id: [source_pattern],
                    source_b.region_id: [source_b_pattern],
                },
                base_sweep_paths={
                    source_a.region_id: sweep(source_pattern),
                    source_b.region_id: sweep(source_b_pattern),
                },
                base_sweep_segment_templates={},
                assignment=BalancedAssignment(
                    agent_regions={0: [merged.region_id]},
                    loads={0: merged_pattern.estimated_time},
                    connected={0: True},
                    imbalance_ratio=0.0,
                    objective=0.0,
                ),
                ownership_map=CoverageOwnershipMap(resolution=1.0),
                agents={0: AgentPathPlan(agent_id=0, source_algorithm="unit")},
                tours={
                    0: SingleUsvTourPlan(
                        agent_id=0,
                        region_order=[],
                        selected_patterns={merged.region_id: merged_pattern},
                    )
                },
                tsp_records={0: {"skipped_regions": [merged.region_id], "final_order": []}},
                infeasible_edges=[],
            )
        finally:
            paper_style_experiment._solve_agent_region_tsp = old_solver

        self.assertEqual(diagnostics["accepted_count"], 1)
        self.assertEqual(diagnostics["accepted_records"][0]["status"], "accepted")
        self.assertEqual(diagnostics["accepted_records"][0]["source_expansion_extra_count"], 1)
        self.assertEqual(final_assignment.agent_regions[0], ["source_a", "source_b"])
        self.assertEqual({region.region_id for region in final_regions}, {"source_a", "source_b"})
        self.assertEqual(set(final_patterns), {"source_a", "source_b"})

    def test_agent_task_merge_unstable_real_pattern_is_flagged(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            max_open_chains_per_region=8,
            agent_task_strip_merge_min_length_gain_factor=1.15,
        )
        merged = DecomposedRegion(
            region_id="agent0_strip_task_region_0",
            bounds=(0.0, 0.0, 12.0, 8.0),
            polygon=[(0.0, 0.0), (12.0, 0.0), (12.0, 8.0), (0.0, 8.0)],
            center=(6.0, 4.0),
            area=96.0,
            preferred_axis="x",
            metadata={
                "agent_task_strip_merge": "true",
                "agent_task_strip_source_ids": "left_column,lower_bar",
                "merge_source_pass_count": "4",
                "merge_candidate_pass_count": "2",
                "merge_source_max_pass_length": "10.0",
                "merge_objective_delta": "5.0",
                "shape_class": "near_convex_composite",
            },
        )
        pattern = _manual_pattern(
            "merged_real",
            merged.bounds,
            pass_count=4,
            pass_length=10.0,
            estimated_fraction=1.0,
            config=config,
            region_id=merged.region_id,
        )
        pattern.metadata["open_chain_count"] = "12"
        pattern.metadata["open_chain_validation_only"] = "true"

        unstable_ids, records = paper_style_experiment._agent_task_merge_unstable_region_ids(
            [merged],
            {merged.region_id: [pattern]},
            path_config,
        )

        self.assertEqual(unstable_ids, [merged.region_id])
        self.assertEqual(records[0]["source_region_ids"], ["left_column", "lower_bar"])
        self.assertIn("real_pass_count_not_reduced", records[0]["reason"])
        self.assertIn("too_many_open_chains", records[0]["reason"])

    def test_agent_task_merge_validation_only_pattern_must_execute(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_open_sweep_chain_tsp=True,
            max_open_chains_per_region=8,
        )
        merged = DecomposedRegion(
            region_id="agent0_strip_task_region_0",
            bounds=(0.0, 0.0, 24.0, 8.0),
            polygon=[(0.0, 0.0), (24.0, 0.0), (24.0, 8.0), (0.0, 8.0)],
            center=(12.0, 4.0),
            area=192.0,
            preferred_axis="x",
            metadata={
                "agent_task_strip_merge": "true",
                "agent_task_strip_source_ids": "left,right",
                "merge_source_pass_count": "8",
                "merge_candidate_pass_count": "4",
                "merge_source_max_pass_length": "10.0",
                "merge_objective_delta": "12.0",
                "shape_class": "near_convex_composite",
            },
        )
        pattern = _manual_pattern(
            "merged_validation_only",
            merged.bounds,
            pass_count=4,
            pass_length=24.0,
            estimated_fraction=1.0,
            config=config,
            region_id=merged.region_id,
        )
        pattern.metadata["open_chain_validation_only"] = "true"
        pattern.metadata["open_chain_count"] = "4"

        old_builder = paper_style_experiment._build_internal_sweep_segments

        def fake_builder(*args, **kwargs):
            return [], "open_chain_exit_connector_failed"

        try:
            paper_style_experiment._build_internal_sweep_segments = fake_builder
            unstable_ids, records = paper_style_experiment._agent_task_merge_unstable_region_ids(
                [merged],
                {merged.region_id: [pattern]},
                path_config,
                config=config,
                obstacle_field=None,
            )
        finally:
            paper_style_experiment._build_internal_sweep_segments = old_builder

        self.assertEqual(unstable_ids, [merged.region_id])
        self.assertIn("open_chain_execution_unavailable", records[0]["reason"])
        self.assertEqual(records[0]["internal_execution_reason"], "open_chain_exit_connector_failed")
        self.assertEqual(records[0]["internal_execution_probe_count"], 1)

    def test_agent_task_merge_keeps_same_pass_count_when_long_pass_gain_is_strong(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            max_open_chains_per_region=8,
            agent_task_strip_merge_min_length_gain_factor=1.15,
        )
        merged = DecomposedRegion(
            region_id="agent0_strip_task_region_0",
            bounds=(0.0, 0.0, 24.0, 8.0),
            polygon=[(0.0, 0.0), (24.0, 0.0), (24.0, 8.0), (0.0, 8.0)],
            center=(12.0, 4.0),
            area=192.0,
            preferred_axis="x",
            metadata={
                "agent_task_strip_merge": "true",
                "agent_task_strip_source_ids": "left_column,lower_bar",
                "merge_source_pass_count": "4",
                "merge_candidate_pass_count": "2",
                "merge_source_max_pass_length": "10.0",
                "merge_objective_delta": "5.0",
                "shape_class": "near_convex_composite",
            },
        )
        pattern = _manual_pattern(
            "merged_real",
            merged.bounds,
            pass_count=4,
            pass_length=16.0,
            estimated_fraction=1.0,
            config=config,
            region_id=merged.region_id,
        )
        pattern.metadata["open_chain_count"] = "2"

        unstable_ids, records = paper_style_experiment._agent_task_merge_unstable_region_ids(
            [merged],
            {merged.region_id: [pattern]},
            path_config,
        )

        self.assertEqual(unstable_ids, [])
        self.assertEqual(records, [])

    def test_agent_task_merge_splits_high_open_chain_negative_objective_region(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            max_open_chains_per_region=8,
            agent_task_strip_merge_min_length_gain_factor=1.15,
        )
        merged = DecomposedRegion(
            region_id="agent0_strip_task_region_0",
            bounds=(0.0, 0.0, 28.0, 8.0),
            polygon=[(0.0, 0.0), (28.0, 0.0), (28.0, 8.0), (0.0, 8.0)],
            center=(14.0, 4.0),
            area=224.0,
            preferred_axis="x",
            metadata={
                "agent_task_strip_merge": "true",
                "agent_task_strip_source_ids": "a,b,c",
                "merge_source_pass_count": "4",
                "merge_candidate_pass_count": "2",
                "merge_source_max_pass_length": "10.0",
                "merge_objective_delta": "-5.0",
                "shape_class": "near_convex_composite",
            },
        )
        pattern = _manual_pattern(
            "merged_real",
            merged.bounds,
            pass_count=4,
            pass_length=16.0,
            estimated_fraction=1.0,
            config=config,
            region_id=merged.region_id,
        )
        pattern.metadata["open_chain_count"] = "16"
        pattern.metadata["open_chain_validation_only"] = "true"

        unstable_ids, records = paper_style_experiment._agent_task_merge_unstable_region_ids(
            [merged],
            {merged.region_id: [pattern]},
            path_config,
        )

        self.assertEqual(unstable_ids, [merged.region_id])
        self.assertIn("nonrectangular_negative_objective_delta", records[0]["reason"])

    def test_agent_task_merge_keeps_high_open_chain_region_when_internal_execution_builds(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_open_sweep_chain_tsp=True,
            max_open_chains_per_region=12,
            agent_task_strip_merge_min_length_gain_factor=1.10,
        )
        merged = DecomposedRegion(
            region_id="agent0_strip_task_region_0",
            bounds=(1.0, 1.0, 12.0, 11.0),
            polygon=[(1.0, 1.0), (12.0, 1.0), (12.0, 11.0), (1.0, 11.0)],
            center=(6.5, 6.0),
            area=110.0,
            preferred_axis="x",
            metadata={
                "agent_task_strip_merge": "true",
                "agent_task_strip_source_ids": "a,b,c",
                "merge_source_pass_count": "3",
                "merge_candidate_pass_count": "1",
                "merge_source_max_pass_length": "7.0",
                "merge_objective_delta": "10.0",
                "shape_class": "near_convex_composite",
            },
        )
        pattern = _manual_pattern(
            "merged_real",
            merged.bounds,
            pass_count=1,
            pass_length=8.0,
            estimated_fraction=1.0,
            config=config,
            region_id=merged.region_id,
        )
        pattern.metadata["open_chain_count"] = "10"
        pattern.metadata["open_chain_validation_only"] = "true"

        unstable_ids, records = paper_style_experiment._agent_task_merge_unstable_region_ids(
            [merged],
            {merged.region_id: [pattern]},
            path_config,
            config=config,
            obstacle_field=None,
        )

        self.assertEqual(unstable_ids, [])
        self.assertEqual(records, [])

    def test_agent_task_merge_keeps_nonrectangular_negative_objective_when_coherent_gain_is_real(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            max_open_chains_per_region=8,
            agent_task_strip_merge_min_length_gain_factor=1.15,
        )
        merged = DecomposedRegion(
            region_id="agent0_strip_task_region_0",
            bounds=(0.0, 0.0, 80.0, 18.0),
            polygon=[(0.0, 0.0), (80.0, 0.0), (80.0, 18.0), (0.0, 18.0)],
            center=(40.0, 9.0),
            area=1440.0,
            preferred_axis="x",
            metadata={
                "agent_task_strip_merge": "true",
                "agent_task_strip_source_ids": "a,b,c,d",
                "merge_source_pass_count": "14",
                "merge_candidate_pass_count": "8",
                "merge_source_max_pass_length": "50.0",
                "merge_objective_delta": "-8.0",
                "shape_class": "near_convex_composite",
            },
        )
        pattern = _manual_pattern(
            "merged_real",
            merged.bounds,
            pass_count=8,
            pass_length=78.0,
            estimated_fraction=1.0,
            config=config,
            region_id=merged.region_id,
        )
        pattern.metadata["open_chain_count"] = "4"
        pattern.metadata["open_chain_validation_only"] = "false"

        unstable_ids, records = paper_style_experiment._agent_task_merge_unstable_region_ids(
            [merged],
            {merged.region_id: [pattern]},
            path_config,
        )

        self.assertEqual(unstable_ids, [])
        self.assertEqual(records, [])

    def test_agent_task_merge_can_disable_coherent_negative_objective_preservation(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            max_open_chains_per_region=8,
            agent_task_strip_merge_min_length_gain_factor=1.15,
            agent_task_merge_keep_coherent_negative_objective=False,
        )
        merged = DecomposedRegion(
            region_id="agent0_strip_task_region_0",
            bounds=(0.0, 0.0, 80.0, 18.0),
            polygon=[(0.0, 0.0), (80.0, 0.0), (80.0, 18.0), (0.0, 18.0)],
            center=(40.0, 9.0),
            area=1440.0,
            preferred_axis="x",
            metadata={
                "agent_task_strip_merge": "true",
                "agent_task_strip_source_ids": "a,b,c,d",
                "merge_source_pass_count": "14",
                "merge_candidate_pass_count": "8",
                "merge_source_max_pass_length": "50.0",
                "merge_objective_delta": "-8.0",
                "shape_class": "near_convex_composite",
            },
        )
        pattern = _manual_pattern(
            "merged_real",
            merged.bounds,
            pass_count=8,
            pass_length=78.0,
            estimated_fraction=1.0,
            config=config,
            region_id=merged.region_id,
        )
        pattern.metadata["open_chain_count"] = "4"
        pattern.metadata["open_chain_validation_only"] = "false"

        unstable_ids, records = paper_style_experiment._agent_task_merge_unstable_region_ids(
            [merged],
            {merged.region_id: [pattern]},
            path_config,
        )

        self.assertEqual(unstable_ids, [merged.region_id])
        self.assertIn("nonrectangular_negative_objective_delta", records[0]["reason"])
        self.assertTrue(records[0]["coherent_boustrophedon_gain"])

    def test_agent_task_boustrophedon_merge_reward_prefers_longer_passes(self) -> None:
        config = _build_visual_test_config()
        path_config = PathPlanningConfig.from_planner_config(config)
        before = {"pass_count": 6, "max_pass_length": 10.0}
        unchanged = {"pass_count": 6, "max_pass_length": 10.0}
        longer = {"pass_count": 6, "max_pass_length": 16.0}

        unchanged_reward = paper_style_experiment._agent_task_boustrophedon_merge_reward(
            before,
            unchanged,
            path_config,
        )
        longer_reward = paper_style_experiment._agent_task_boustrophedon_merge_reward(
            before,
            longer,
            path_config,
        )

        self.assertEqual(unchanged_reward, 0.0)
        self.assertGreater(longer_reward, unchanged_reward)

    def test_agent_task_group_sort_prefers_long_sweep_span(self) -> None:
        def region(region_id: str, bounds: tuple[float, float, float, float]) -> DecomposedRegion:
            x0, y0, x1, y1 = bounds
            return DecomposedRegion(
                region_id=region_id,
                bounds=bounds,
                polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                center=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                area=(x1 - x0) * (y1 - y0),
                preferred_axis="x",
            )

        short_group = [region("short_a", (0.0, 0.0, 6.0, 2.0)), region("short_b", (6.0, 0.0, 12.0, 2.0))]
        long_group = [region("long_a", (0.0, 0.0, 14.0, 2.0)), region("long_b", (14.0, 0.0, 32.0, 2.4))]

        ordered = sorted(
            [short_group, long_group],
            key=paper_style_experiment._agent_task_strip_group_sort_key,
        )

        self.assertEqual([item.region_id for item in ordered[0]], ["long_a", "long_b"])

    def test_open_chain_assembly_anchors_exit_chain_without_final_backtrack(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_open_sweep_chain_tsp=True,
            open_chain_tsp_beam_width=4,
        )
        pattern = _manual_pattern(
            "open_anchor",
            (1.0, 1.0, 12.0, 5.0),
            pass_count=2,
            pass_length=8.0,
            estimated_fraction=1.0,
            config=config,
            region_id="merged_region",
        )

        def chain(chain_id: str, coverage_pass: CoveragePass, pass_index: int) -> OpenSweepChain:
            return OpenSweepChain(
                chain_id=chain_id,
                region_id=pattern.region_id,
                pattern_id=pattern.pattern_id,
                pass_indices=[pass_index],
                passes=[coverage_pass],
                entry_pose=coverage_pass.start_pose,
                exit_pose=coverage_pass.end_pose,
                reverse_entry_pose=Pose2D(
                    coverage_pass.end_pose.x,
                    coverage_pass.end_pose.y,
                    (coverage_pass.end_pose.psi + math.pi) % (2.0 * math.pi),
                ),
                reverse_exit_pose=Pose2D(
                    coverage_pass.start_pose.x,
                    coverage_pass.start_pose.y,
                    (coverage_pass.start_pose.psi + math.pi) % (2.0 * math.pi),
                ),
                coverage_length=coverage_pass.length,
                estimated_time=coverage_pass.length / max(config.fleet.cover_speed, 1e-6),
            )

        first_chain = chain("chain_first", pattern.passes[0], 0)
        final_chain = chain("chain_final", pattern.passes[1], 1)

        segments, reason, connected = paper_style_experiment._assemble_open_chains_greedy(
            pattern,
            [final_chain, first_chain],
            config,
            path_config,
            obstacle_field=None,
            start_time=0.0,
            segment_prefix="unit",
        )

        self.assertEqual(reason, "")
        self.assertEqual(connected[-1], "chain_final")
        self.assertTrue(any(segment.metadata.get("open_chain_endpoint_anchored") == "true" for segment in segments))
        self.assertFalse(any(segment.metadata.get("open_chain_exit_connector") == "true" for segment in segments))
        last = segments[-1].waypoints[-1]
        self.assertAlmostEqual(last.x, pattern.exit_pose.x)
        self.assertAlmostEqual(last.y, pattern.exit_pose.y)

    def test_open_chain_flexible_exit_accepts_actual_chain_exit(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_open_sweep_chain_tsp=True,
            open_chain_allow_flexible_exit=True,
        )
        pattern = _manual_pattern(
            "open_flexible",
            (1.0, 1.0, 12.0, 5.0),
            pass_count=1,
            pass_length=4.0,
            estimated_fraction=1.0,
            config=config,
            region_id="merged_region",
        )
        actual_exit = pattern.passes[0].end_pose
        nominal_exit = Pose2D(11.0, 4.0, 0.0)
        pattern.exit_pose = nominal_exit
        chain = OpenSweepChain(
            chain_id="chain_actual_exit",
            region_id=pattern.region_id,
            pattern_id=pattern.pattern_id,
            pass_indices=[0],
            passes=[pattern.passes[0]],
            entry_pose=pattern.passes[0].start_pose,
            exit_pose=actual_exit,
            reverse_entry_pose=Pose2D(actual_exit.x, actual_exit.y, math.pi),
            reverse_exit_pose=Pose2D(pattern.passes[0].start_pose.x, pattern.passes[0].start_pose.y, math.pi),
            coverage_length=pattern.passes[0].length,
            estimated_time=pattern.passes[0].length / max(config.fleet.cover_speed, 1e-6),
        )
        old_connector = paper_style_experiment._build_open_chain_connector

        def fake_connector(*args, **kwargs):
            segment_id = str(kwargs.get("segment_id", args[0] if args else ""))
            if segment_id.endswith("_open_chain_exit_to_pattern"):
                return None
            return old_connector(*args, **kwargs)

        try:
            paper_style_experiment._build_open_chain_connector = fake_connector
            segments, reason, connected = paper_style_experiment._assemble_open_chains_greedy(
                pattern,
                [chain],
                config,
                path_config,
                obstacle_field=None,
                start_time=0.0,
                segment_prefix="unit",
            )
        finally:
            paper_style_experiment._build_open_chain_connector = old_connector

        self.assertEqual(reason, "")
        self.assertEqual(connected, ["chain_actual_exit"])
        self.assertEqual(pattern.metadata.get("open_chain_flexible_exit"), "true")
        self.assertAlmostEqual(pattern.exit_pose.x, actual_exit.x)
        self.assertAlmostEqual(pattern.exit_pose.y, actual_exit.y)
        self.assertFalse(any(segment.metadata.get("open_chain_exit_connector") == "true" for segment in segments))
        last = segments[-1].waypoints[-1]
        self.assertAlmostEqual(last.x, actual_exit.x)
        self.assertAlmostEqual(last.y, actual_exit.y)

    def test_open_chain_flexible_exit_variant_exposes_actual_exit_candidate(self) -> None:
        config = _build_visual_test_config()
        path_config = replace(
            PathPlanningConfig.from_planner_config(config),
            enable_open_sweep_chain_tsp=True,
            enable_open_chain_flexible_exit_variants=True,
            open_chain_allow_flexible_exit=False,
        )
        pattern = _manual_pattern(
            "open_flexible_candidate",
            (1.0, 1.0, 12.0, 5.0),
            pass_count=1,
            pass_length=4.0,
            estimated_fraction=1.0,
            config=config,
            region_id="agent0_strip_task_region_0",
        )
        actual_exit = pattern.passes[0].end_pose
        nominal_exit = Pose2D(11.0, 4.0, 0.0)
        pattern.exit_pose = nominal_exit
        pattern.metadata.update(
            {
                "agent_task_strip_merge": "true",
                "open_chain_mode": "true",
                "open_chain_validation_only": "true",
            }
        )
        stats: dict[str, object] = {}
        old_connector = paper_style_experiment._build_open_chain_connector
        old_standard_builder = paper_style_experiment._build_standard_internal_sweep_segments

        def fake_connector(*args, **kwargs):
            segment_id = str(kwargs.get("segment_id", args[0] if args else ""))
            if segment_id.endswith("_open_chain_exit_to_pattern"):
                return None
            return old_connector(*args, **kwargs)

        def force_standard_failure(*args, **kwargs):
            return [], "forced_standard_internal_failure"

        try:
            paper_style_experiment._build_open_chain_connector = fake_connector
            paper_style_experiment._build_standard_internal_sweep_segments = force_standard_failure
            result = paper_style_experiment._build_open_chain_flexible_exit_variant(
                pattern,
                config,
                path_config,
                obstacle_field=None,
                stats=stats,
            )
        finally:
            paper_style_experiment._build_open_chain_connector = old_connector
            paper_style_experiment._build_standard_internal_sweep_segments = old_standard_builder

        self.assertIsNotNone(result)
        variant, segments = result
        self.assertEqual(pattern.exit_pose, nominal_exit)
        self.assertEqual(variant.metadata.get("open_chain_flexible_exit_variant"), "true")
        self.assertEqual(variant.metadata.get("open_chain_flexible_exit_variant_from"), pattern.pattern_id)
        self.assertAlmostEqual(variant.exit_pose.x, actual_exit.x)
        self.assertAlmostEqual(variant.exit_pose.y, actual_exit.y)
        self.assertTrue(variant.pattern_id.endswith("_flex_exit"))
        self.assertTrue(segments)
        self.assertFalse(any(segment.metadata.get("open_chain_exit_connector") == "true" for segment in segments))
        self.assertEqual(int(stats.get("open_chain_flexible_exit_variant_attempt_count", 0)), 1)
        self.assertEqual(int(stats.get("open_chain_flexible_exit_variant_success_count", 0)), 1)

    def test_large_map_defaults_enable_budgeted_agent_task_merge(self) -> None:
        config = _build_visual_test_config()
        large_config = replace(config, mission=replace(config.mission, area_length_x=200.0, area_length_y=200.0))
        base = replace(
            PathPlanningConfig.from_planner_config(large_config),
            enable_agent_task_region_merge=True,
            enable_agent_task_lightweight_strip_merge=False,
        )

        applied = paper_style_experiment._apply_large_map_defaults(base, large_config)

        self.assertTrue(applied.enable_agent_task_region_merge)
        self.assertTrue(applied.enable_agent_task_lightweight_strip_merge)
        self.assertTrue(applied.agent_task_merge_enable_unified_group_merge)
        self.assertTrue(applied.agent_task_merge_enable_pairwise_fallback)
        self.assertTrue(applied.agent_task_merge_prefer_full_components)
        self.assertGreaterEqual(applied.agent_task_merge_full_component_max_regions, 12)
        self.assertLessEqual(applied.agent_task_merge_full_component_min_rectangularity, 0.65)
        self.assertLessEqual(applied.agent_task_merge_max_unified_candidates_per_agent, 6)
        self.assertEqual(applied.agent_task_merge_max_unified_group_size, 4)
        self.assertLessEqual(applied.obstacle_aware_astar_max_expansions, 240)
        self.assertFalse(applied.obstacle_aware_allow_motion_lattice)
        self.assertGreaterEqual(applied.large_map_tsp_agent_time_budget_sec, 180.0)
        self.assertEqual(applied.large_map_tsp_total_time_budget_sec, 0.0)
        self.assertGreaterEqual(applied.large_map_tsp_step_time_budget_sec, 8.0)
        self.assertTrue(applied.large_map_tsp_enable_lookahead_probe)
        self.assertLessEqual(applied.large_map_tsp_max_candidate_attempts_per_step, 16)
        self.assertEqual(applied.large_map_tsp_obstacle_aware_retry_limit, 1)
        self.assertFalse(applied.large_map_tsp_require_cheap_connector_probe)
        self.assertEqual(applied.large_map_tsp_max_obstacle_aware_attempts_per_step, 2)
        self.assertLessEqual(applied.large_map_tsp_max_obstacle_aware_attempts_per_agent, 20)
        self.assertLessEqual(applied.large_map_tsp_obstacle_aware_max_transition_length, 180.0)
        self.assertTrue(applied.agent_task_strip_merge_use_geometric_preview)
        self.assertLessEqual(applied.agent_task_strip_merge_time_budget_sec, 8.0)
        self.assertLessEqual(applied.joint_large_map_region_limit, 30)
        self.assertLessEqual(applied.joint_optimizer_time_budget_sec, 60.0)
        self.assertLessEqual(applied.joint_eval_agent_time_budget_sec, 20.0)
        self.assertLessEqual(applied.joint_eval_step_time_budget_sec, 4.0)
        self.assertGreaterEqual(applied.skipped_region_recovery_time_budget_sec, 8.0)
        self.assertGreaterEqual(applied.residual_local_tsp_time_budget_sec, 8.0)
        self.assertGreaterEqual(applied.residual_local_tsp_max_candidate_attempts, 480)


def _manual_pattern(
    pattern_id: str,
    bounds: tuple[float, float, float, float],
    pass_count: int,
    pass_length: float,
    estimated_fraction: float,
    config: PlannerConfig,
    region_id: str = "r0",
    estimated_time: float | None = None,
    turn_length: float = 0.0,
) -> RegionCoveragePattern:
    x_min, y_min, x_max, y_max = bounds
    passes: list[CoveragePass] = []
    for idx in range(pass_count):
        y = y_min + (y_max - y_min) * (idx + 0.5) / max(pass_count, 1)
        heading = 0.0 if idx % 2 == 0 else math.pi
        if idx % 2 == 0:
            start = Pose2D(x_min, y, heading)
            end = Pose2D(min(x_min + pass_length, x_max), y, heading)
        else:
            start = Pose2D(min(x_min + pass_length, x_max), y, heading)
            end = Pose2D(x_min, y, heading)
        passes.append(
            CoveragePass(
                pass_id=f"{pattern_id}_pass_{idx}",
                region_id=region_id,
                sequence_index=idx,
                scan_axis="x",
                start_pose=start,
                end_pose=end,
                center_coordinate=y,
                width=config.footprint.width_wf,
                length=abs(end.x - start.x),
            )
        )
    coverage_length = sum(item.length for item in passes)
    return RegionCoveragePattern(
        pattern_id=pattern_id,
        region_id=region_id,
        scan_axis="x",
        passes=passes,
        entry_pose=passes[0].start_pose,
        exit_pose=passes[-1].end_pose,
        coverage_length=coverage_length,
        turn_length=turn_length,
        turn_angle=0.0,
        total_length=coverage_length + turn_length,
        estimated_time=estimated_time if estimated_time is not None else coverage_length / max(config.fleet.cover_speed, 1e-6),
        max_curvature=0.0,
        feasible=True,
        metadata={
            "region_bounds": f"{x_min:.6f},{y_min:.6f},{x_max:.6f},{y_max:.6f}",
            "region_area": f"{max((x_max - x_min) * (y_max - y_min), 1e-9):.6f}",
            "estimated_region_coverage_fraction": f"{estimated_fraction:.6f}",
        },
    )


def _build_config_for_agents(agent_count: int):
    base = build_test_config()
    if agent_count == 1:
        y_values = [base.mission.area_length_y / 2.0]
    else:
        y_values = [
            1.0 + (base.mission.area_length_y - 2.0) * idx / max(agent_count - 1, 1)
            for idx in range(agent_count)
        ]
    states_3dof = [State3DOF(x=0.0, y=y, psi=0.0) for y in y_values]
    states_6dof = [State6DOF(x=state.x, y=state.y, psi=state.psi) for state in states_3dof]
    base.fleet = FleetConfig(
        initial_states_3dof=states_3dof,
        initial_states_6dof=states_6dof,
        cruise_speed=base.fleet.cruise_speed,
        cover_speed=base.fleet.cover_speed,
        turn_speed_max=base.fleet.turn_speed_max,
        max_thrust=base.fleet.max_thrust,
        max_yaw_moment=base.fleet.max_yaw_moment,
        min_turn_radius=base.fleet.min_turn_radius,
    )
    return base


def _build_visual_test_config() -> PlannerConfig:
    states_3dof = [State3DOF(x=0.0, y=2.0, psi=0.0)]
    states_6dof = [State6DOF(x=state.x, y=state.y, psi=state.psi) for state in states_3dof]
    return PlannerConfig(
        mission=MissionConfig(area_length_x=16.0, area_length_y=10.0, overlap_ratio=0.2, local_control_hz=5.0),
        fleet=FleetConfig(
            initial_states_3dof=states_3dof,
            initial_states_6dof=states_6dof,
            cruise_speed=2.0,
            cover_speed=1.2,
            turn_speed_max=1.0,
            max_thrust=2.0,
            max_yaw_moment=1.0,
            min_turn_radius=2.0,
        ),
        footprint=CoverageFootprint(length_lf=3.0, width_wf=2.0, eta_cov=0.7),
        weights=PlannerWeights(),
        safety=SafetyMargins(d_safe=0.6, boundary_margin_x=0.2, boundary_margin_y=0.2),
    )


def _assignment_test_pattern(region_id: str, agent_id: int, estimated_time: float) -> RegionCoveragePattern:
    start = Pose2D(0.0, 0.0, 0.0)
    end = Pose2D(1.0, 0.0, 0.0)
    coverage_pass = CoveragePass(
        pass_id=f"{region_id}_agent_{agent_id}_pass",
        region_id=region_id,
        sequence_index=0,
        scan_axis="x",
        start_pose=start,
        end_pose=end,
        center_coordinate=0.0,
        width=1.0,
        length=1.0,
    )
    return RegionCoveragePattern(
        pattern_id=f"{region_id}_agent_{agent_id}_pattern",
        region_id=region_id,
        scan_axis="x",
        passes=[coverage_pass],
        entry_pose=start,
        exit_pose=end,
        coverage_length=1.0,
        turn_length=0.0,
        turn_angle=0.0,
        total_length=1.0,
        estimated_time=estimated_time,
        max_curvature=0.0,
        feasible=True,
        metadata={"estimated_region_coverage_fraction": "1.0", "agent_id": str(agent_id)},
    )


def _build_toy_astar_graph() -> RegionGraph:
    regions = {
        "a": _toy_region("a", 0.0, 0.0),
        "b": _toy_region("b", 1.0, 0.0, danger=4),
        "c": _toy_region("c", 0.0, 1.0),
        "d": _toy_region("d", 1.0, 1.0),
    }
    adjacency = {"a": ["b", "c"], "b": ["a", "d"], "c": ["a", "d"], "d": ["b", "c"]}
    edge_weights = {
        ("a", "b"): 1.0,
        ("a", "c"): 1.0,
        ("b", "d"): 1.0,
        ("c", "d"): 1.0,
    }
    return RegionGraph(
        regions=regions,
        adjacency=adjacency,
        node_weights={region_id: 1.0 for region_id in regions},
        edge_weights=edge_weights,
        edge_metadata={key: {"heading_change": 0.0, "dubins_length": value} for key, value in edge_weights.items()},
        patterns={},
    )


def _build_static_obstacles() -> list[StaticObstacle]:
    return [
        rectangle_obstacle("rect", center=(12.0, 4.0), width=1.2, height=1.2),
        circle_obstacle("circle", center=(24.0, 4.0), radius=0.6),
        ellipse_obstacle("ellipse", center=(34.0, 4.0), radii=(0.8, 0.5), psi=0.25),
        polygon_obstacle("poly", [(42.0, 3.0), (43.4, 3.4), (42.8, 4.5)]),
    ]


def _toy_region(region_id: str, x: float, y: float, danger: int = 0) -> DecomposedRegion:
    return DecomposedRegion(
        region_id=region_id,
        bounds=(x, y, x + 0.5, y + 0.5),
        polygon=[(x, y), (x + 0.5, y), (x + 0.5, y + 0.5), (x, y + 0.5)],
        center=(x, y),
        area=0.25,
        preferred_axis="x",
        metadata={"danger_neighbors": str(danger)},
    )


def _plain_segment(segment_id: str, points: list[tuple[float, float]]) -> PathSegmentSpec:
    waypoints = [
        PathWaypoint(x=x, y=y, psi=0.0, time=float(idx), speed=1.0)
        for idx, (x, y) in enumerate(points)
    ]
    length = sum(
        math.hypot(points[idx][0] - points[idx - 1][0], points[idx][1] - points[idx - 1][1])
        for idx in range(1, len(points))
    )
    return PathSegmentSpec(
        segment_id=segment_id,
        kind="transit",
        source_algorithm="test",
        waypoints=waypoints,
        length=length,
        path_source="raw_polyline",
    )


def _resource_segment(segment_id: str, start_time: float, end_time: float, resource_id: str) -> PathSegmentSpec:
    return PathSegmentSpec(
        segment_id=segment_id,
        kind="transit",
        source_algorithm="test",
        waypoints=[
            PathWaypoint(x=0.0, y=0.0, psi=0.0, time=start_time, speed=1.0),
            PathWaypoint(x=1.0, y=0.0, psi=0.0, time=end_time, speed=1.0),
        ],
        length=1.0,
        metadata={"resource_id": resource_id},
    )


def _open_chain_test_config() -> PlannerConfig:
    return PlannerConfig(
        mission=MissionConfig(area_length_x=20.0, area_length_y=8.0, overlap_ratio=0.1, local_control_hz=5.0),
        fleet=FleetConfig(
            initial_states_3dof=[State3DOF(x=1.0, y=1.0, psi=0.0)],
            initial_states_6dof=[],
            cruise_speed=1.2,
            cover_speed=1.0,
            turn_speed_max=0.8,
            max_thrust=4.0,
            max_yaw_moment=4.0,
            min_turn_radius=0.5,
        ),
        footprint=CoverageFootprint(length_lf=1.0, width_wf=0.5, eta_cov=0.7),
        weights=PlannerWeights(),
        safety=SafetyMargins(d_safe=0.0, boundary_margin_x=0.0, boundary_margin_y=0.0),
    )


def _open_chain_test_pattern(lines: list[tuple[float, float, float, float, float]]) -> RegionCoveragePattern:
    passes = []
    for idx, (x0, y0, x1, y1, heading) in enumerate(lines):
        start = Pose2D(x0, y0, heading)
        end = Pose2D(x1, y1, heading)
        passes.append(
            CoveragePass(
                pass_id=f"pass_{idx}",
                region_id="open_region",
                sequence_index=idx,
                scan_axis="x",
                start_pose=start,
                end_pose=end,
                center_coordinate=y0,
                width=0.5,
                length=math.hypot(x1 - x0, y1 - y0),
            )
        )
    return RegionCoveragePattern(
        pattern_id="open_pattern",
        region_id="open_region",
        scan_axis="x",
        passes=passes,
        entry_pose=passes[0].start_pose,
        exit_pose=passes[-1].end_pose,
        coverage_length=sum(item.length for item in passes),
        turn_length=0.0,
        turn_angle=0.0,
        total_length=sum(item.length for item in passes),
        estimated_time=sum(item.length for item in passes),
        max_curvature=0.0,
    )


if __name__ == "__main__":
    unittest.main()
