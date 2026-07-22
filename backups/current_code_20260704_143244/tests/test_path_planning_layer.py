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
    plan_global_coverage,
    run_paper_style_region_tsp_experiment,
    run_planning_algorithm_experiment,
)
from usv_swarm.schema import CoverageFootprint, CoverageResidual, FleetConfig, MissionConfig, PlannerConfig, PlannerWeights, Pose2D, SafetyMargins, State3DOF, State6DOF  # noqa: E402
from usv_swarm.path_planning.adapters.runtime_adapter import path_plan_to_trajectory_references  # noqa: E402
from usv_swarm.dubins import dubins_shortest_path  # noqa: E402
from usv_swarm.path_planning.aco import solve_aco_tsp_cpp  # noqa: E402
from usv_swarm.path_planning.assignment import apply_lightweight_load_swap, balance_region_workload  # noqa: E402
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
                metrics={"total_length": 10.0},
            ),
            1: AgentPathPlan(
                agent_id=1,
                source_algorithm="test",
                segments=[],
                metrics={"total_length": 20.0},
            ),
        }
        totals = {
            "total_length": 30.0,
            "coverage_length": 12.0,
            "transition_length": 18.0,
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

    def test_joint_solution_improves_rejects_skips_and_worse_imbalance(self) -> None:
        current = {"executed_region_count": 3, "skipped_region_count": 0, "load_imbalance": 0.1, "objective": 100.0}
        skipped = {"executed_region_count": 2, "skipped_region_count": 1, "load_imbalance": 0.0, "objective": 10.0}
        imbalanced = {"executed_region_count": 3, "skipped_region_count": 0, "load_imbalance": 0.4, "objective": 10.0}
        improved = {"executed_region_count": 3, "skipped_region_count": 0, "load_imbalance": 0.1, "objective": 90.0}

        self.assertFalse(paper_style_experiment._joint_solution_improves(skipped, current))
        self.assertFalse(paper_style_experiment._joint_solution_improves(imbalanced, current))
        self.assertTrue(paper_style_experiment._joint_solution_improves(improved, current))

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
        region = DecomposedRegion(
            region_id="center_region",
            bounds=(4.0, 2.0, 10.0, 6.0),
            polygon=[(4.0, 2.0), (10.0, 2.0), (10.0, 6.0), (4.0, 6.0)],
            center=(7.0, 4.0),
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
        min_length = max(config.footprint.width_wf * 0.25, config.footprint.length_lf * path_config.retraction_min_pass_length_factor)
        self.assertTrue(all(item.length + 1e-6 >= min_length for item in pattern.passes))

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
