from __future__ import annotations

import math
import pathlib
import sys
import unittest
from dataclasses import replace


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
for path in (SRC, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_framework import build_test_config
from usv_swarm.path_planning.crown import (
    CrownGeometricMode,
    CrownInstance,
    CrownMcppConfig,
    CrownMotionPrimitive,
    CrownOperation,
    CrownPricingDuals,
    CrownPricingPrecedenceDual,
    CrownPricingRestrictions,
    build_continuous_responsibility_regions,
    certify_continuous_pattern_coverage,
    CrownRoute,
    CrownTimeExpandedModeGraph,
    CrownTimedRoute,
    conservative_tube_cells,
    find_continuous_conflicts,
    prepare_crown_problem,
    price_mode_graph_exact,
    service_workload_lower_bound,
    solve_crown_graph_bpc,
    solve_crown_lns,
    solve_prepared_crown_problem,
    solve_joint_exact,
)
from usv_swarm.path_planning.crown.motion import segment_to_motion_primitives
from usv_swarm.path_planning.layer import PathPlanningLayer
from usv_swarm.path_planning.map_loader import load_map_for_planner
from usv_swarm.path_planning.types import (
    CoveragePass,
    DecomposedRegion,
    ObstacleField,
    PathPlanningConfig,
    PathPlanningRequest,
    PathSegmentSpec,
    PathWaypoint,
    RegionCoveragePattern,
    StaticObstacle,
)
from usv_swarm.schema import (
    CoverageFootprint,
    FleetConfig,
    MissionConfig,
    Pose2D,
    SafetyMargins,
    State3DOF,
    State6DOF,
    VehicleFootprint,
)


def _primitive(
    agent_id: str,
    primitive_id: str,
    duration: float,
    resource_id: str,
    *,
    start: Pose2D = Pose2D(0.0, 0.0, 0.0),
    end: Pose2D = Pose2D(1.0, 0.0, 0.0),
    kind: str = "cover",
) -> CrownMotionPrimitive:
    segment = PathSegmentSpec(
        segment_id=primitive_id,
        kind=kind,
        source_algorithm="crown-test",
        waypoints=[
            PathWaypoint(start.x, start.y, start.psi, time=0.0),
            PathWaypoint(end.x, end.y, end.psi, time=duration),
        ],
        length=math.hypot(end.x - start.x, end.y - start.y),
    )
    return CrownMotionPrimitive(
        primitive_id=primitive_id,
        agent_id=agent_id,
        kind=kind,
        start_pose=start,
        end_pose=end,
        duration=duration,
        energy=duration,
        resource_ids=(resource_id,),
        segment=segment,
    )


def _mode(agent_id: str, task_id: str, name: str, duration: float, resource: str):
    # A stationary synthetic service keeps the declared duration invariant
    # under the current-aware retiming routine used by graph pricing.
    origin = Pose2D(0.0, 0.0, 0.0)
    primitive = _primitive(
        agent_id,
        f"{agent_id}:{name}",
        duration,
        resource,
        start=origin,
        end=origin,
    )
    coverage_pass = CoveragePass(
        pass_id=f"{agent_id}:{name}:pass",
        region_id=task_id,
        sequence_index=0,
        scan_axis="x",
        start_pose=primitive.start_pose,
        end_pose=primitive.end_pose,
        center_coordinate=0.0,
        width=1.0,
        length=1.0,
    )
    pattern = RegionCoveragePattern(
        pattern_id=f"{agent_id}:{name}",
        region_id=task_id,
        scan_axis="x",
        passes=[coverage_pass],
        entry_pose=primitive.start_pose,
        exit_pose=primitive.end_pose,
        coverage_length=1.0,
        turn_length=0.0,
        turn_angle=0.0,
        total_length=1.0,
        estimated_time=duration,
        max_curvature=0.0,
    )
    return CrownGeometricMode(
        agent_id=agent_id,
        task_id=task_id,
        mode_id=pattern.pattern_id,
        pattern=pattern,
        service_segments=(primitive.segment,),
        nominal_service_primitives=(primitive,),
        nominal_duration=duration,
        nominal_energy=duration,
    )


def _shared_corridor_graphs():
    planner = build_test_config()
    path = replace(
        PathPlanningConfig.from_planner_config(planner),
        crown_time_step=0.5,
        crown_return_to_start=False,
        crown_enable_continuous_conflict_validation=False,
        crown_lns_iterations=16,
        crown_lns_time_budget_sec=2.0,
        crown_lns_pool_reopt_interval=4,
        crown_pricing_label_limit=100_000,
    )
    crown = CrownMcppConfig.from_path_config(path)
    graphs = {}
    for index in range(3):
        agent_id = str(index)
        task_id = f"task-{index}"
        shared = _mode(agent_id, task_id, "shared", 1.0, "shared-corridor")
        private = _mode(agent_id, task_id, "private", 1.5, f"private-{index}")
        graphs[agent_id] = CrownTimeExpandedModeGraph(
            agent_id=agent_id,
            numeric_agent_id=index,
            profile=planner.profile_for_agent(index),
            planner_config=planner.for_agent(index),
            path_config=path,
            crown_config=crown,
            obstacle_field=None,
            modes_by_task={task_id: (shared, private)},
            start_pose=Pose2D(0.0, 0.0, 0.0),
            goal_pose=Pose2D(0.0, 0.0, 0.0),
        )
    return graphs, tuple(f"task-{index}" for index in range(3))


class CrownFullAlgorithmTests(unittest.TestCase):
    def test_turn_penalty_changes_primary_duration_and_secondary_energy(self) -> None:
        config = build_test_config()
        base_profile = config.profile_for_agent(0)
        penalized_profile = replace(
            base_profile,
            turn_time_penalty_per_rad=12.0,
            turn_energy_penalty_per_rad=20.0,
            turn_maneuver_time_penalty=30.0,
            turn_maneuver_energy_penalty=100.0,
        )
        segment = PathSegmentSpec(
            segment_id="curved-transit",
            kind="transit",
            source_algorithm="test",
            waypoints=[
                PathWaypoint(0.0, 0.0, 0.0, time=0.0),
                PathWaypoint(1.0, 1.0, math.pi / 2.0, time=1.0),
            ],
            length=math.sqrt(2.0),
        )
        crown = CrownMcppConfig(primitive_max_duration=2.0)

        def totals(profile):
            primitives = segment_to_motion_primitives(
                segment,
                agent_id="0",
                profile=profile,
                crown_config=crown,
                planning_distance=1.0,
                primitive_prefix="test-turn-cost",
            )
            return (
                sum(primitive.duration for primitive in primitives),
                sum(primitive.energy for primitive in primitives),
            )

        base_duration, base_energy = totals(base_profile)
        duration, energy = totals(penalized_profile)
        turn_angle = math.pi / 2.0
        self.assertAlmostEqual(
            duration - base_duration,
            12.0 * turn_angle + 30.0,
            places=9,
        )
        self.assertAlmostEqual(
            energy - base_energy,
            12.0 * turn_angle * base_profile.transit_power
            + 30.0 * base_profile.transit_power
            + 20.0 * turn_angle
            + 100.0,
            places=9,
        )

    def test_real_obstacle_map_lns_constructs_continuous_global_incumbent(self) -> None:
        states = [State3DOF(2.25, 2.25, 0.0), State3DOF(2.25, 12.0, 0.0)]
        fleet = FleetConfig(
            initial_states_3dof=states,
            initial_states_6dof=[
                State6DOF(x=state.x, y=state.y, psi=state.psi)
                for state in states
            ],
            num_agents=2,
            cruise_speed=2.0,
            cover_speed=1.2,
            turn_speed_max=1.0,
            max_thrust=2.0,
            max_yaw_moment=1.0,
            min_turn_radius=0.5,
        )
        map_path = (
            ROOT
            / "maps"
            / "static_obstacle_map_15x15_rect_triangle_small"
            / "static_obstacle_map_15x15_rect_triangle_small.json"
        )
        config, obstacles = load_map_for_planner(map_path, fleet)
        config = replace(
            config,
            fleet=replace(config.fleet, min_turn_radius=0.5),
            vehicle_footprint=VehicleFootprint(1.0, 0.5),
        )
        path = replace(
            PathPlanningConfig.from_planner_config(config),
            crown_engine="certified_lns",
            crown_return_to_start=False,
            crown_lns_time_budget_sec=30.0,
            crown_lns_iterations=1,
            crown_root_exact_pricing=False,
            crown_include_sequential_baseline=True,
            crown_baseline_time_budget_sec=30.0,
            crown_connector_max_expansions=500,
            # Match the map's declared fleet/profile regression contract.  An
            # eight-mode library is a different, substantially more expensive
            # instance and cannot be used to judge the profile's 30 s seed.
            crown_mode_limit_per_region_agent=4,
            max_candidate_axes=2,
        )

        problem = prepare_crown_problem(config, path, obstacles)
        solution = solve_prepared_crown_problem(problem)

        self.assertTrue(problem.responsibility_certificate.valid)
        self.assertEqual(problem.responsibility_certificate.gap_area, 0.0)
        self.assertEqual(problem.responsibility_certificate.overlap_area, 0.0)
        self.assertEqual(
            sorted(task for route in solution.timed_routes for task in route.task_ids),
            sorted(problem.task_ids),
        )
        self.assertTrue(solution.solution_status.startswith("certified_lns"))
        self.assertIsNotNone(solution.baseline_makespan)
        self.assertLess(solution.makespan, float("inf"))

    def test_continuous_responsibility_partition_has_no_gap_spill_or_overlap(self) -> None:
        config = replace(
            build_test_config(),
            mission=MissionConfig(10.0, 10.0, overlap_ratio=0.2, local_control_hz=5.0),
        )
        path = PathPlanningConfig.from_planner_config(config)
        field = ObstacleField(
            inflated_obstacles=[
                StaticObstacle(
                    obstacle_id="oblique-triangle",
                    kind="polygon",
                    polygon=[(3.1, 1.2), (7.4, 4.8), (4.2, 8.9)],
                )
            ]
        )

        regions, certificate = build_continuous_responsibility_regions(
            config,
            path,
            field,
        )

        self.assertTrue(certificate.valid)
        self.assertAlmostEqual(certificate.gap_area, 0.0)
        self.assertAlmostEqual(certificate.spill_area, 0.0)
        self.assertAlmostEqual(certificate.overlap_area, 0.0)
        self.assertGreater(len(regions), 1)
        self.assertTrue(
            all(
                region.metadata["crown_geometry_role"]
                == "exact_polygon_not_envelope"
                for region in regions
            )
        )

    def test_continuous_mode_coverage_rejects_subgrid_boundary_gaps(self) -> None:
        config = replace(
            build_test_config(),
            mission=MissionConfig(4.0, 2.0, overlap_ratio=0.2, local_control_hz=5.0),
            footprint=CoverageFootprint(2.0, 2.0, 0.7),
        )
        region = DecomposedRegion(
            region_id="continuous-cell",
            bounds=(0.0, 0.0, 4.0, 2.0),
            polygon=[(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)],
            center=(2.0, 1.0),
            area=8.0,
            preferred_axis="x",
        )

        def pattern(start_x: float, end_x: float) -> RegionCoveragePattern:
            coverage_pass = CoveragePass(
                pass_id="continuous-pass",
                region_id=region.region_id,
                sequence_index=0,
                scan_axis="x",
                start_pose=Pose2D(start_x, 1.0, 0.0),
                end_pose=Pose2D(end_x, 1.0, 0.0),
                center_coordinate=1.0,
                width=2.0,
                length=end_x - start_x,
            )
            return RegionCoveragePattern(
                pattern_id="continuous-pattern",
                region_id=region.region_id,
                scan_axis="x",
                passes=[coverage_pass],
                entry_pose=coverage_pass.start_pose,
                exit_pose=coverage_pass.end_pose,
                coverage_length=end_x - start_x,
                turn_length=0.0,
                turn_angle=0.0,
                total_length=end_x - start_x,
                estimated_time=1.0,
                max_curvature=0.0,
            )

        complete = certify_continuous_pattern_coverage(
            region,
            pattern(1.0, 3.0),
            config,
        )
        incomplete = certify_continuous_pattern_coverage(
            region,
            pattern(1.1, 2.9),
            config,
        )

        self.assertTrue(complete.valid)
        self.assertEqual(complete.missing_area, 0.0)
        self.assertFalse(incomplete.valid)
        self.assertGreater(incomplete.missing_area, 0.0)

    def test_conservative_tube_contains_crossed_cells(self) -> None:
        cells = set(
            conservative_tube_cells(
                (0.1, 0.1),
                (2.9, 0.1),
                radius=0.2,
                grid_size=1.0,
            )
        )
        self.assertTrue({(0, 0), (1, 0), (2, 0)}.issubset(cells))

    def test_exact_graph_bpc_matches_independent_continuous_oracle(self) -> None:
        graphs, tasks = _shared_corridor_graphs()
        routes_by_agent = {}
        for agent_id in graphs:
            task = f"task-{agent_id}"
            routes_by_agent[agent_id] = (
                CrownRoute(f"{agent_id}:empty", agent_id, (), ()),
                CrownRoute(
                    f"{agent_id}:shared",
                    agent_id,
                    (task,),
                    (CrownOperation("shared", 1.0, ("shared-corridor",), 1.0),),
                ),
                CrownRoute(
                    f"{agent_id}:private",
                    agent_id,
                    (task,),
                    (CrownOperation("private", 1.5, (), 1.5),),
                ),
            )
        oracle = solve_joint_exact(CrownInstance(tuple(graphs), tasks, routes_by_agent))
        solution = solve_crown_graph_bpc(graphs, tasks, horizon=3.0)

        self.assertEqual(solution.objective, oracle.objective)
        self.assertEqual(solution.objective, (1.5, 4.0))
        self.assertEqual(solution.solution_status, "exact_graph_bpc")
        self.assertEqual(solution.lower_bound, solution.upper_bound)
        self.assertGreater(solution.conflict_separation_rounds, 0)
        self.assertGreater(solution.resource_precedence_branches, 0)
        self.assertGreater(solution.pricing_labels, 0)

    def test_precedence_branch_dual_is_priced_at_first_resource_entry(self) -> None:
        graphs, _ = _shared_corridor_graphs()
        graph = graphs["0"]
        shared = graph.modes_for_task("task-0")[0]
        graph.modes_by_task = {"task-0": (shared,)}
        graph.__post_init__()
        result = price_mode_graph_exact(
            graph,
            horizon_slots=8,
            duals=CrownPricingDuals(
                agent_dual=0.0,
                task_duals={"task-0": 100.0},
                precedence_duals=(
                    CrownPricingPrecedenceDual(
                        resource_id="shared-corridor",
                        role="after",
                        horizon_slots=8,
                        dual=-1.0,
                    ),
                ),
            ),
            restrictions=CrownPricingRestrictions(
                required_tasks=frozenset({"task-0"}),
            ),
        )

        self.assertIsNotNone(result.route)
        first_entry = min(
            slot
            for resource_id, slot in result.route.occupied_resource_slots
            if resource_id == "shared-corridor"
        )
        coefficient = 8 - first_entry
        self.assertAlmostEqual(result.reduced_cost, -100.0 + coefficient)

    def test_wait_edges_are_explicit_and_occupy_stationary_tube(self) -> None:
        graphs, _ = _shared_corridor_graphs()
        graph = graphs["0"]
        shared = graph.modes_for_task("task-0")[0]
        graph.modes_by_task = {"task-0": (shared,)}
        graph.__post_init__()
        result = price_mode_graph_exact(
            graph,
            horizon_slots=8,
            duals=CrownPricingDuals(
                agent_dual=0.0,
                task_duals={"task-0": 10.0},
                makespan_dual=-1.0,
            ),
            restrictions=CrownPricingRestrictions(
                required_tasks=frozenset({"task-0"}),
                forbidden_resource_slots=frozenset(
                    {("shared-corridor", 0), ("shared-corridor", 1)}
                ),
            ),
        )
        self.assertIsNotNone(result.route)
        route = result.route
        self.assertEqual(route.base_route.operations[0].kind, "wait")
        self.assertGreater(route.base_route.operations[0].duration, 0.0)
        self.assertTrue(
            any(resource.startswith("tube:") and slot == 0 for resource, slot in route.occupied_resource_slots)
        )

    def test_continuous_crossing_conflict_maps_to_canonical_resource(self) -> None:
        graphs, _ = _shared_corridor_graphs()
        left = _primitive(
            "0",
            "left-cross",
            2.0,
            "tube:cross",
            start=Pose2D(-1.0, 0.0, 0.0),
            end=Pose2D(1.0, 0.0, 0.0),
        )
        right = _primitive(
            "1",
            "right-cross",
            2.0,
            "tube:cross",
            start=Pose2D(0.0, -1.0, math.pi / 2.0),
            end=Pose2D(0.0, 1.0, math.pi / 2.0),
        )

        def timed(primitive: CrownMotionPrimitive) -> CrownTimedRoute:
            base = CrownRoute(
                primitive.primitive_id,
                primitive.agent_id,
                (),
                (primitive.to_operation(),),
            )
            return CrownTimedRoute(
                timed_route_id=primitive.primitive_id,
                base_route=base,
                start_slots=(0,),
                duration_slots=(4,),
                finish_slot=4,
                time_step=0.5,
                occupied_resource_slots=tuple(("tube:cross", slot) for slot in range(4)),
                energy=primitive.energy,
            )

        conflicts = find_continuous_conflicts((timed(left), timed(right)), graphs)
        self.assertTrue(conflicts)
        self.assertAlmostEqual(conflicts[0].minimum_distance, 0.0)
        self.assertTrue(conflicts[0].mapped_resources)

    def test_lns_returns_feasible_solution_with_valid_lower_bound(self) -> None:
        graphs, tasks = _shared_corridor_graphs()
        solution = solve_crown_lns(graphs, tasks, horizon=3.0)

        self.assertEqual(solution.makespan, 1.5)
        self.assertLessEqual(solution.lower_bound, solution.makespan)
        self.assertEqual(service_workload_lower_bound(graphs, tasks), 1.0)
        self.assertTrue(solution.solution_status.startswith("certified_lns"))
        self.assertTrue(solution.anytime_trace)

    def test_lns_bootstrap_jointly_repairs_disconnected_greedy_ownership(self) -> None:
        planner = build_test_config()
        path = replace(
            PathPlanningConfig.from_planner_config(planner),
            crown_time_step=1.0,
            crown_return_to_start=False,
            crown_enable_continuous_conflict_validation=False,
            crown_root_exact_pricing=False,
            crown_include_sequential_baseline=False,
            crown_lns_iterations=2,
            crown_lns_time_budget_sec=2.0,
        )
        crown = CrownMcppConfig.from_path_config(path)
        tasks = ("a", "b", "c", "d")
        graphs = {}
        for index in range(2):
            agent_id = str(index)
            modes = {
                task: (
                    _mode(
                        agent_id,
                        task,
                        task,
                        1.0,
                        f"private:{agent_id}:{task}",
                    ),
                )
                for task in tasks
            }
            graph = CrownTimeExpandedModeGraph(
                agent_id=agent_id,
                numeric_agent_id=index,
                profile=planner.profile_for_agent(index),
                planner_config=planner.for_agent(index),
                path_config=path,
                crown_config=crown,
                obstacle_field=None,
                modes_by_task=modes,
                start_pose=Pose2D(0.0, 0.0, 0.0),
                goal_pose=Pose2D(0.0, 0.0, 0.0),
            )
            all_mode_ids = tuple(mode.mode_id for value in modes.values() for mode in value)
            for previous in (None,) + all_mode_ids:
                for following in all_mode_ids:
                    graph.connection_segment_cache[(previous, following)] = None
            if agent_id == "0":
                allowed_tasks = ("a", "b")
            else:
                allowed_tasks = ("c", "d")
            first = modes[allowed_tasks[0]][0].mode_id
            second = modes[allowed_tasks[1]][0].mode_id
            graph.connection_segment_cache[(None, first)] = ()
            graph.connection_segment_cache[(first, second)] = ()
            graphs[agent_id] = graph

        solution = solve_crown_lns(graphs, tasks, horizon=6.0)

        ownership = {route.agent_id: route.task_ids for route in solution.timed_routes}
        self.assertEqual(ownership, {"0": ("a", "b"), "1": ("c", "d")})
        self.assertEqual(solution.makespan, 2.0)

    def test_service_primitives_are_retimed_by_arrival_current(self) -> None:
        class ReversingCurrent:
            time_invariant = False

            def velocity(self, x: float, y: float, time: float):
                del x, y
                return (0.5, 0.0) if time < 1.0 else (-0.5, 0.0)

        planner = build_test_config()
        path = replace(
            PathPlanningConfig.from_planner_config(planner),
            crown_time_step=1.0,
            crown_return_to_start=False,
        )
        crown = CrownMcppConfig.from_path_config(path)
        primitive = _primitive(
            "0",
            "moving-service",
            1.0,
            "canonical-service-resource",
            start=Pose2D(0.0, 0.0, 0.0),
            end=Pose2D(1.0, 0.0, 0.0),
        )
        mode = _mode("0", "task", "moving", 1.0, "unused")
        mode = replace(
            mode,
            service_segments=(primitive.segment,),
            nominal_service_primitives=(primitive,),
            nominal_duration=primitive.duration,
            nominal_energy=primitive.energy,
        )
        graph = CrownTimeExpandedModeGraph(
            agent_id="0",
            numeric_agent_id=0,
            profile=planner.profile_for_agent(0),
            planner_config=planner.for_agent(0),
            path_config=path,
            crown_config=crown,
            obstacle_field=None,
            modes_by_task={"task": (mode,)},
            start_pose=Pose2D(0.0, 0.0, 0.0),
            goal_pose=Pose2D(0.0, 0.0, 0.0),
            current_field=ReversingCurrent(),
        )

        early = graph.service_primitives(mode.mode_id, 0)
        late = graph.service_primitives(mode.mode_id, 2)

        self.assertIsNotNone(early)
        self.assertIsNotNone(late)
        self.assertLess(early[0].duration, late[0].duration)
        self.assertEqual(early[0].resource_ids, primitive.resource_ids)
        self.assertEqual(late[0].resource_ids, primitive.resource_ids)

    def test_path_planning_layer_runs_complete_small_geometric_pipeline(self) -> None:
        config = build_test_config()
        start = State3DOF(0.5, 2.0, 0.0)
        config = replace(
            config,
            mission=MissionConfig(
                12.0,
                4.0,
                overlap_ratio=0.2,
                local_control_hz=5.0,
            ),
            fleet=replace(
                config.fleet,
                initial_states_3dof=[start],
                initial_states_6dof=[State6DOF(x=0.5, y=2.0, psi=0.0)],
                num_agents=1,
                min_turn_radius=0.5,
                cruise_speed=2.0,
                cover_speed=1.5,
                turn_speed_max=1.0,
            ),
            footprint=CoverageFootprint(4.0, 4.0, 0.7),
            vehicle_footprint=VehicleFootprint(1.0, 0.5),
            safety=SafetyMargins(
                d_safe=0.5,
                boundary_margin_x=0.2,
                boundary_margin_y=0.2,
                delta_safe_max=1.0,
                t_block=8.0,
            ),
        )
        path = replace(
            PathPlanningConfig.from_planner_config(config),
            max_regions_per_agent=1,
            max_candidate_axes=1,
            enable_oriented_sweep_patterns=False,
            crown_engine="bpc",
            crown_horizon=30.0,
            crown_primitive_max_duration=4.0,
            crown_return_to_start=False,
            crown_pricing_label_limit=100_000,
        )
        plan = PathPlanningLayer().plan_paths(
            PathPlanningRequest(config=config, path_config=path),
            algorithm_name="crown_mcpp",
        )

        self.assertEqual(plan.algorithm_name, "crown_mcpp")
        self.assertEqual(plan.metadata["status"], "exact_graph_bpc")
        self.assertEqual(plan.metadata["coverage_validated"], "true")
        self.assertGreaterEqual(float(plan.metadata["coverage_fraction"]), 0.99)
        self.assertEqual(plan.agents[0].source_algorithm, "crown_mcpp")
        self.assertTrue(plan.agents[0].segments)


if __name__ == "__main__":
    unittest.main()
