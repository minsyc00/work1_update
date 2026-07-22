from __future__ import annotations

import math
import pathlib
import random
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from usv_swarm.path_planning.crown import (
    CrownInstance,
    CrownMode,
    CrownOperation,
    CrownRoute,
    CrownSegmentRouteCandidate,
    assert_schedule_resource_feasible,
    bpc_solution_to_path_plan,
    build_crown_instance_from_segment_candidates,
    build_shared_corridor_counterexample,
    compare_joint_and_sequential,
    enumerate_agent_routes,
    run_shared_corridor_proof_experiment,
    schedule_selected_routes_exact,
    solve_crown_bpc,
    solve_joint_exact,
)
from usv_swarm.path_planning.types import PathPlanningConfig, PathSegmentSpec, PathWaypoint


class CrownMcppTests(unittest.TestCase):
    def test_exact_scheduler_chooses_best_conflict_orientation(self) -> None:
        long_then_tail = CrownRoute(
            route_id="a:route",
            agent_id="a",
            task_ids=("x",),
            operations=(
                CrownOperation("shared", 2.0, ("channel",)),
                CrownOperation("tail", 1.0),
            ),
        )
        short = CrownRoute(
            route_id="b:route",
            agent_id="b",
            task_ids=("y",),
            operations=(CrownOperation("shared", 1.0, ("channel",)),),
        )

        schedule = schedule_selected_routes_exact((long_then_tail, short))

        self.assertTrue(math.isclose(schedule.makespan, 3.0))
        self.assertTrue(math.isclose(schedule.starts[("a", 0)], 0.0))
        self.assertTrue(math.isclose(schedule.starts[("b", 0)], 2.0))
        self.assertEqual(schedule.conflict_pairs, 1)
        assert_schedule_resource_feasible((long_then_tail, short), schedule)

    def test_route_enumerator_covers_every_subset_order_and_mode(self) -> None:
        modes = tuple(
            CrownMode(
                task_id=task,
                mode_id=f"{task}-{mode}",
                agent_id="a",
                operations=(CrownOperation(f"cover-{task}-{mode}", 1.0),),
            )
            for task in ("x", "y")
            for mode in ("forward", "reverse")
        )

        routes = enumerate_agent_routes("a", ("x", "y"), modes)

        # empty + four one-task modes + 2! orders * 2^2 mode combinations
        self.assertEqual(len(routes), 13)
        self.assertEqual(sum(not route.task_ids for route in routes), 1)
        self.assertEqual(
            {route.task_ids for route in routes if len(route.task_ids) == 2},
            {("x", "y"), ("y", "x")},
        )

    def test_exact_scheduler_uses_wait_energy_as_lexicographic_tiebreak(self) -> None:
        expensive_wait = CrownRoute(
            "a:shared",
            "a",
            ("x",),
            (CrownOperation("shared", 1.0, ("channel",)),),
        )
        cheap_wait = CrownRoute(
            "b:shared",
            "b",
            ("y",),
            (CrownOperation("shared", 1.0, ("channel",)),),
        )

        schedule = schedule_selected_routes_exact(
            (expensive_wait, cheap_wait),
            wait_energy_rates={"a": 10.0, "b": 1.0},
        )

        self.assertEqual(schedule.makespan, 2.0)
        self.assertEqual(schedule.starts[("a", 0)], 0.0)
        self.assertEqual(schedule.starts[("b", 0)], 1.0)
        self.assertEqual(schedule.waiting_energy, 1.0)

    def test_joint_exact_strictly_beats_strong_sequential_postprocessing(self) -> None:
        instance, assignment = build_shared_corridor_counterexample(4, 0.5)

        comparison = compare_joint_and_sequential(instance, assignment)

        sequential = comparison["sequential"]
        joint = comparison["joint"]
        self.assertTrue(math.isclose(sequential.statistics["nominal_makespan"], 1.0))
        self.assertTrue(math.isclose(sequential.makespan, 4.0))
        self.assertTrue(math.isclose(joint.makespan, 1.5))
        self.assertTrue(math.isclose(comparison["joint_gain_ratio"], 4.0 / 1.5))
        self.assertIs(comparison["strict_improvement"], True)
        self.assertEqual(sum(route.mode_ids == ("private",) for route in joint.routes), 3)

    def test_crown_bpc_matches_independent_enumeration_oracle(self) -> None:
        instance, _ = build_shared_corridor_counterexample(3, 0.5)

        oracle = solve_joint_exact(instance)
        bpc = solve_crown_bpc(instance, horizon=3.0, time_step=0.5)

        self.assertEqual(bpc.objective, oracle.objective)
        self.assertEqual(bpc.lower_bound, bpc.upper_bound)
        self.assertEqual(bpc.upper_bound, oracle.makespan)
        self.assertEqual(bpc.optimality_gap, 0.0)
        self.assertEqual(bpc.energy_lower_bound, bpc.energy_upper_bound)
        self.assertEqual(bpc.energy_upper_bound, oracle.total_energy)
        self.assertEqual(bpc.energy_optimality_gap, 0.0)
        self.assertGreater(bpc.pricing_iterations, 0)
        self.assertGreater(bpc.generated_columns, len(instance.agent_ids))
        self.assertGreater(bpc.conflict_separation_rounds, 0)
        self.assertTrue(bpc.active_conflict_resources)

    def test_proof_experiment_reports_theoretical_and_observed_ratio(self) -> None:
        report = run_shared_corridor_proof_experiment(
            agent_count=3,
            epsilon=0.25,
            verify_bpc=True,
        )

        self.assertTrue(math.isclose(report["joint_gain_ratio"], 3.0 / 1.25))
        self.assertTrue(math.isclose(report["theoretical_ratio"], 3.0 / 1.25))
        self.assertIs(report["bpc_matches_enum"], True)

    def test_exact_cover_can_change_task_assignment(self) -> None:
        routes = {
            "a": (
                CrownRoute("a:empty", "a", (), ()),
                CrownRoute("a:x", "a", ("x",), (CrownOperation("x", 3.0),)),
                CrownRoute("a:y", "a", ("y",), (CrownOperation("y", 1.0),)),
            ),
            "b": (
                CrownRoute("b:empty", "b", (), ()),
                CrownRoute("b:x", "b", ("x",), (CrownOperation("x", 1.0),)),
                CrownRoute("b:y", "b", ("y",), (CrownOperation("y", 3.0),)),
            ),
        }
        instance = CrownInstance(("a", "b"), ("x", "y"), routes)

        comparison = compare_joint_and_sequential(
            instance,
            {"a": ("x",), "b": ("y",)},
        )

        self.assertTrue(math.isclose(comparison["sequential"].makespan, 3.0))
        self.assertTrue(math.isclose(comparison["joint"].makespan, 1.0))
        self.assertEqual(comparison["task_assignment_changes"], 2)

    def test_existing_path_segments_round_trip_through_public_adapter(self) -> None:
        path_config = PathPlanningConfig(resource_separation_time=0.0)
        candidates = {}
        for agent_id in (0, 1):
            task_id = f"task-{agent_id}"
            common = PathSegmentSpec(
                segment_id=f"common-{agent_id}",
                kind="transit",
                source_algorithm="test",
                waypoints=[
                    PathWaypoint(0.0, 0.0, 0.0, time=0.0),
                    PathWaypoint(1.0, 0.0, 0.0, time=1.0),
                ],
                length=1.0,
            )
            private = PathSegmentSpec(
                segment_id=f"private-{agent_id}",
                kind="transit",
                source_algorithm="test",
                waypoints=[
                    PathWaypoint(0.0, 10.0 + agent_id, 0.0, time=0.0),
                    PathWaypoint(1.5, 10.0 + agent_id, 0.0, time=1.5),
                ],
                length=1.5,
            )
            candidates[agent_id] = (
                CrownSegmentRouteCandidate("common", (task_id,), (common,), ("common",)),
                CrownSegmentRouteCandidate("private", (task_id,), (private,), ("private",)),
            )

        instance = build_crown_instance_from_segment_candidates(
            (0, 1),
            ("task-0", "task-1"),
            candidates,
            path_config,
        )
        solution = solve_crown_bpc(instance, horizon=2.0, time_step=0.5)
        path_plan = bpc_solution_to_path_plan(solution)

        self.assertEqual(solution.makespan, 1.5)
        self.assertEqual(set(path_plan.agents), {0, 1})
        self.assertEqual(path_plan.algorithm_name, "crown_bpc_minimal_exact")
        self.assertEqual(path_plan.metadata["optimality_gap"], "0.000000000")
        for agent in path_plan.agents.values():
            self.assertTrue(agent.segments)
            self.assertEqual(agent.segments[0].metadata["crown_retimed"], "true")

    def test_bpc_matches_oracle_on_deterministic_random_integer_instances(self) -> None:
        for seed in range(10):
            rng = random.Random(seed)
            routes = {}
            for agent_id in ("a", "b"):
                candidates = [CrownRoute(f"{agent_id}:empty", agent_id, (), ())]
                for task_id in ("x", "y"):
                    for mode in range(2):
                        duration = float(rng.randint(1, 3))
                        resources = ("channel",) if rng.random() < 0.5 else ()
                        candidates.append(
                            CrownRoute(
                                f"{agent_id}:{task_id}:{mode}",
                                agent_id,
                                (task_id,),
                                (
                                    CrownOperation(
                                        f"{task_id}-{mode}",
                                        duration,
                                        resources,
                                        energy=duration,
                                    ),
                                ),
                                mode_ids=(str(mode),),
                            )
                        )
                routes[agent_id] = tuple(candidates)
            instance = CrownInstance(("a", "b"), ("x", "y"), routes)

            oracle = solve_joint_exact(instance)
            bpc = solve_crown_bpc(instance, horizon=6.0, time_step=1.0)

            self.assertEqual(bpc.objective, oracle.objective, msg=f"seed={seed}")


if __name__ == "__main__":
    unittest.main()
