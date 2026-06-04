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
from usv_swarm.schema import CoverageFootprint, FleetConfig, MissionConfig, PlannerConfig, PlannerWeights, Pose2D, SafetyMargins, State3DOF, State6DOF  # noqa: E402
from usv_swarm.path_planning.adapters.runtime_adapter import path_plan_to_trajectory_references  # noqa: E402
from usv_swarm.path_planning.assignment import balance_region_workload  # noqa: E402
from usv_swarm.path_planning.astar import obstacle_aware_grid_astar, sailing_safety_weight, turn_aware_astar  # noqa: E402
from usv_swarm.path_planning.coverage import (  # noqa: E402
    RectangularCoverageModel,
    build_coverage_state,
    find_residual_components,
    mark_coverage_passes,
)
from usv_swarm.path_planning.decomposition import (  # noqa: E402
    build_free_space_cells,
    concave_vertex_indices,
    decompose_obstacle_aware_area,
    decompose_polygon_interface,
    decompose_rectangular_area,
)
from usv_swarm.path_planning.dynamics_validation import validate_transition_dynamics  # noqa: E402
from usv_swarm.path_planning.graph import build_region_graph, graph_is_connected  # noqa: E402
from usv_swarm.path_planning.obstacles import (  # noqa: E402
    circle_obstacle,
    ellipse_obstacle,
    normalize_obstacle_field,
    obstacle_bounds,
    path_segment_invalid_reasons,
    point_in_any_obstacle,
    polyline_collides_with_obstacles,
    polygon_collides_with_obstacles,
    polygon_obstacle,
    rectangle_obstacle,
)
from usv_swarm.path_planning.patterns import generate_all_region_patterns  # noqa: E402
from usv_swarm.path_planning.residuals import assign_residual_backfill  # noqa: E402
from usv_swarm.path_planning.scheduling import apply_resource_window_schedule  # noqa: E402
from usv_swarm.path_planning.smoothing import build_obstacle_aware_transition_segments  # noqa: E402
from usv_swarm.path_planning.tsp import solve_multi_agent_tours, solve_single_usv_tsp_cpp  # noqa: E402
from usv_swarm.path_planning.types import (  # noqa: E402
    AgentPathPlan,
    DecomposedRegion,
    PathPlanningConfig,
    PathSegmentSpec,
    PathWaypoint,
    RegionGraph,
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

        regions = decompose_rectangular_area(config)
        patterns = generate_all_region_patterns(regions, config)
        state = build_coverage_state(config)
        passes = []
        for candidates in patterns.values():
            best = min(candidates, key=lambda item: item.estimated_time)
            passes.extend((coverage_pass.start_pose, coverage_pass.end_pose) for coverage_pass in best.passes)
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
                    self.assertLessEqual(abs(second - first), RectangularCoverageModel.from_config(config).strip_spacing + 1e-6)

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

    def test_static_obstacle_layer_integration(self) -> None:
        config = build_test_config()
        layer = PathPlanningLayer()
        path_plan = layer.plan_from_config(config, static_obstacles=_build_static_obstacles())
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
        self.assertTrue(all(not path_segment_invalid_reasons(segment, config, None) for segment in segments))
        self.assertTrue(all(segment.metadata.get("connector") in {"astar_corridor", "smoothed_astar_corridor", "local_safe_corridor_edge"} for segment in segments))

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
                "05_region_tsp_nodes.png",
                "06_agent_region_tsp_order.png",
                "07_agent_sweep_endpoints.png",
                "08_final_region_tsp_coverage_path.png",
                "09_constraint_validation.png",
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
            for agent in path_plan.agents.values():
                self.assertGreater(len(agent.segments), 0)
                for segment in agent.segments:
                    self.assertNotEqual(segment.path_source, "astar_corridor_edge")
                    self.assertNotEqual(segment.metadata.get("kinematic_feasible"), "false")
                    self.assertNotEqual(segment.metadata.get("dynamic_feasible"), "false")


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


if __name__ == "__main__":
    unittest.main()
