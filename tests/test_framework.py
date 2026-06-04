from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from usv_swarm import (  # noqa: E402
    AgentRuntimeState,
    CoverageFootprint,
    FleetConfig,
    MissionConfig,
    PlannerConfig,
    PlannerWeights,
    SafetyMargins,
    State3DOF,
    State6DOF,
    SwarmRuntime,
    simulate_swarm_closed_loop,
    plan_global_coverage,
)
from usv_swarm.planning import _find_first_conflict, build_boustrophedon_strips  # noqa: E402


def build_test_config() -> PlannerConfig:
    states_3dof = [
        State3DOF(x=0.0, y=1.0, psi=0.0),
        State3DOF(x=0.0, y=7.0, psi=0.0),
        State3DOF(x=0.0, y=13.0, psi=0.0),
    ]
    states_6dof = [State6DOF(x=s.x, y=s.y, psi=s.psi) for s in states_3dof]
    return PlannerConfig(
        mission=MissionConfig(area_length_x=48.0, area_length_y=18.0, overlap_ratio=0.2, local_control_hz=5.0),
        fleet=FleetConfig(
            initial_states_3dof=states_3dof,
            initial_states_6dof=states_6dof,
            cruise_speed=3.0,
            cover_speed=2.0,
            turn_speed_max=1.2,
            max_thrust=2.0,
            max_yaw_moment=1.0,
            min_turn_radius=3.5,
        ),
        footprint=CoverageFootprint(length_lf=4.0, width_wf=4.0, eta_cov=0.7),
        weights=PlannerWeights(),
        safety=SafetyMargins(d_safe=2.5, boundary_margin_x=0.2, boundary_margin_y=0.2, delta_safe_max=1.0, t_block=8.0),
    )


class FrameworkTests(unittest.TestCase):
    def test_strip_generation(self) -> None:
        config = build_test_config()
        strips = build_boustrophedon_strips(config)
        self.assertGreaterEqual(len(strips), 4)
        self.assertEqual(strips[0].scan_axis, "x")
        self.assertAlmostEqual(strips[0].start_pose.y, config.footprint.width_wf / 2.0)

    def test_global_plan_has_no_remaining_conflicts(self) -> None:
        config = build_test_config()
        plan = plan_global_coverage(config)
        self.assertEqual(len(plan.refs), config.fleet.num_agents)
        self.assertIsNone(_find_first_conflict(plan.reservations.reservations))
        self.assertGreater(plan.reservations.makespan, 0.0)
        for agent_id, task_range in plan.assignments.assignments.items():
            if task_range != (-1, -1):
                self.assertGreater(len(plan.refs[agent_id].samples), 0)
        for path in plan.paths.values():
            for segment in path.segments:
                if segment.segment_type in {"turn", "transit"}:
                    self.assertLessEqual(segment.max_curvature, 1.0 / config.fleet.min_turn_radius + 1e-3)

    def test_runtime_control_step(self) -> None:
        config = build_test_config()
        plan = plan_global_coverage(config)
        runtime = SwarmRuntime(config, plan)
        result = runtime.control_step(
            AgentRuntimeState(
                agent_id=0,
                time=0.0,
                state3=config.fleet.initial_states_3dof[0],
                state6=config.fleet.initial_states_6dof[0],
            )
        )
        self.assertLessEqual(abs(result.cmd.thrust), config.fleet.max_thrust + 1e-9)
        self.assertLessEqual(abs(result.cmd.yaw_moment), config.fleet.max_yaw_moment + 1e-9)
        self.assertGreater(len(result.local_ref), 0)
        self.assertGreaterEqual(runtime.coverage.state.coverage_fraction, 0.0)

    def test_closed_loop_simulation_smoke(self) -> None:
        config = build_test_config()
        plan = plan_global_coverage(config)
        log = simulate_swarm_closed_loop(config, planning_result=plan, total_time=0.4)
        self.assertGreaterEqual(len(log.frames), 2)
        self.assertGreaterEqual(log.final_coverage_fraction, 0.0)
        self.assertIn(0, log.frames[-1].agent_states)


if __name__ == "__main__":
    unittest.main()
