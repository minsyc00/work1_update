from __future__ import annotations

import math
import pickle
import pathlib
import sys
import time
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from usv_swarm import (  # noqa: E402
    AgentRuntimeState,
    ControlInput,
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
from usv_swarm.control import CoverageTracker, USV3DOFModel  # noqa: E402
from usv_swarm.geometry import rotated_rectangle_local_mask, rotated_rectangle_mask  # noqa: E402
from usv_swarm.nmpc_backend import NMPCBackendSolveResult, NMPCSolveRequest  # noqa: E402
from usv_swarm.nmpc import NMPCResult  # noqa: E402
from usv_swarm.safety_filter import filter_control_cbf_qp  # noqa: E402
from usv_swarm.schema import PathRequirement, Pose2D, ReservationEntry, TrajectorySample  # noqa: E402
from usv_swarm.planning import _find_first_conflict, _find_first_conflict_indexed, build_boustrophedon_strips, solve_cbs_mapf  # noqa: E402


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

    def test_indexed_mapf_conflict_detector_matches_legacy_detector(self) -> None:
        pose = Pose2D(0.0, 0.0, 0.0)
        reservations = {
            0: [
                ReservationEntry(
                    agent_id=0,
                    seq_index=0,
                    resource_id="shared",
                    kind="cover",
                    t_enter=0.0,
                    t_exit=2.0,
                    from_node="a",
                    to_node="b",
                    start_pose=pose,
                    end_pose=pose,
                )
            ],
            1: [
                ReservationEntry(
                    agent_id=1,
                    seq_index=0,
                    resource_id="shared",
                    kind="cover",
                    t_enter=1.0,
                    t_exit=3.0,
                    from_node="c",
                    to_node="d",
                    start_pose=pose,
                    end_pose=pose,
                )
            ],
        }

        legacy = _find_first_conflict(reservations)
        indexed = _find_first_conflict_indexed(reservations)

        self.assertIsNotNone(legacy)
        self.assertIsNotNone(indexed.conflict)
        self.assertEqual(indexed.conflict.kind, legacy.kind)
        self.assertEqual(indexed.conflict.start_time, legacy.start_time)
        self.assertGreater(indexed.checks, 0)

    def test_cbs_budget_exhaustion_uses_prioritized_resource_window_fallback(self) -> None:
        config = build_test_config()
        config.mission.mapf_max_expanded_nodes = 1
        config.mission.mapf_max_conflicts = 1
        pose = Pose2D(1.0, 1.0, 0.0)
        requirements = {
            agent_id: [
                PathRequirement(
                    agent_id=agent_id,
                    seq_index=0,
                    kind="cover",
                    resource_id="shared_strip",
                    duration=1.0,
                    from_node=f"agent:{agent_id}:in",
                    to_node=f"agent:{agent_id}:out",
                    start_pose=pose,
                    end_pose=pose,
                )
            ]
            for agent_id in range(3)
        }

        table = solve_cbs_mapf(config, requirements)

        self.assertTrue(table.budget_exhausted)
        self.assertTrue(table.fallback_used)
        self.assertIn("fallback", table.solver_status)
        self.assertEqual(table.unresolved_conflict_count, 0)
        self.assertIsNone(_find_first_conflict(table.reservations))
        self.assertGreaterEqual(table.makespan, 3.0)

    def test_runtime_control_step(self) -> None:
        config = build_test_config()
        plan = plan_global_coverage(config)
        runtime = SwarmRuntime(config, plan)
        self.assertEqual(runtime.nmpc_by_agent[0].integration_method, "rk4")
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

    def test_local_rotated_rectangle_mask_matches_full_grid_mask(self) -> None:
        x_coords = np.arange(0.25, 24.0, 0.5)
        y_coords = np.arange(0.25, 18.0, 0.5)
        full = rotated_rectangle_mask(x_coords, y_coords, 7.3, 8.8, 0.72, 4.0, 2.0)
        local, row_slice, col_slice = rotated_rectangle_local_mask(x_coords, y_coords, 7.3, 8.8, 0.72, 4.0, 2.0)
        reconstructed = np.zeros_like(full, dtype=bool)
        reconstructed[row_slice, col_slice] = local
        self.assertTrue(np.array_equal(full, reconstructed))
        self.assertLess(local.size, full.size)

    def test_local_rotated_rectangle_mask_clips_at_boundary(self) -> None:
        x_coords = np.arange(0.25, 10.0, 0.5)
        y_coords = np.arange(0.25, 10.0, 0.5)
        full = rotated_rectangle_mask(x_coords, y_coords, 0.3, 0.4, 0.9, 4.0, 2.0)
        local, row_slice, col_slice = rotated_rectangle_local_mask(x_coords, y_coords, 0.3, 0.4, 0.9, 4.0, 2.0)
        reconstructed = np.zeros_like(full, dtype=bool)
        reconstructed[row_slice, col_slice] = local
        self.assertTrue(np.array_equal(full, reconstructed))
        self.assertGreaterEqual(row_slice.start, 0)
        self.assertGreaterEqual(col_slice.start, 0)

    def test_coverage_tracker_local_update_matches_full_mask_result(self) -> None:
        config = build_test_config()
        tracker = CoverageTracker(config)
        pose = State3DOF(x=9.0, y=6.0, psi=0.53).pose()
        tracker.update(pose)
        full_mask = rotated_rectangle_mask(
            tracker.state.x_coords,
            tracker.state.y_coords,
            pose.x,
            pose.y,
            pose.psi,
            config.footprint.length_lf,
            config.footprint.width_wf,
        )
        expected = np.zeros_like(tracker.state.covered, dtype=bool)
        expected[full_mask] = True
        self.assertTrue(np.array_equal(tracker.state.covered, expected))
        self.assertLess(tracker.updated_cell_count, tracker.state.covered.size)

    def test_fast_tracker_control_step_avoids_nmpc(self) -> None:
        config = build_test_config()
        config.mission.control_mode = "fast_tracker"
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
        self.assertEqual(runtime.nmpc_by_agent, {})
        self.assertEqual(runtime.profiler.nmpc_called_count, 0)
        self.assertEqual(result.safety_status.mode, "fast_tracker")
        self.assertLessEqual(abs(result.cmd.thrust), config.fleet.max_thrust + 1e-9)
        self.assertLessEqual(abs(result.cmd.yaw_moment), config.fleet.max_yaw_moment + 1e-9)
        self.assertGreater(runtime.profiler.cbf_filter_called_count, 0)

    def test_cbf_filter_reduces_boundary_outward_control(self) -> None:
        config = build_test_config()
        state = State3DOF(x=0.3, y=5.0, psi=math.pi, u=1.0, v=0.0, r=0.0)
        nominal = ControlInput(thrust=config.fleet.max_thrust, yaw_moment=0.0)

        result = filter_control_cbf_qp(
            state,
            nominal,
            {},
            [],
            config,
            current_time=0.0,
            dt=0.2,
        )

        self.assertTrue(result.filtered)
        self.assertIn("boundary", result.active_constraints)
        self.assertLessEqual(result.control.thrust, nominal.thrust)

    def test_cbf_filter_reduces_head_on_neighbor_control(self) -> None:
        config = build_test_config()
        config.safety.d_safe = 1.0
        state = State3DOF(x=5.0, y=5.0, psi=0.0, u=1.2, v=0.0, r=0.0)
        nominal = ControlInput(thrust=config.fleet.max_thrust, yaw_moment=0.0)
        predictions = {
            1: [
                TrajectorySample(time=0.0, x=6.25, y=5.0, psi=math.pi, u_ref=0.0, r_ref=0.0, segment_type="predicted"),
                TrajectorySample(time=0.2, x=6.20, y=5.0, psi=math.pi, u_ref=0.0, r_ref=0.0, segment_type="predicted"),
            ]
        }

        result = filter_control_cbf_qp(
            state,
            nominal,
            predictions,
            [],
            config,
            current_time=0.0,
            dt=0.2,
        )

        self.assertTrue(result.filtered)
        self.assertTrue(any(item.startswith("agent:") for item in result.active_constraints))
        self.assertLessEqual(result.control.thrust, nominal.thrust)

    def test_hybrid_nmpc_update_interval_skips_solver_between_updates(self) -> None:
        config = build_test_config()
        config.mission.control_mode = "hybrid_nmpc"
        config.mission.nmpc_update_interval_steps = 100
        plan = plan_global_coverage(config)
        runtime = SwarmRuntime(config, plan)
        runtime._last_nmpc_step[0] = 0
        result = runtime.control_step(
            AgentRuntimeState(
                agent_id=0,
                time=runtime.dt,
                state3=config.fleet.initial_states_3dof[0],
                state6=config.fleet.initial_states_6dof[0],
            )
        )
        self.assertEqual(runtime.profiler.nmpc_called_count, 0)
        self.assertEqual(result.safety_status.mode, "hybrid_tracker")
        self.assertIn("nmpc skipped by scheduler", result.safety_status.warnings)

    def test_hybrid_nmpc_timeout_falls_back_to_tracker(self) -> None:
        config = build_test_config()
        config.mission.control_mode = "hybrid_nmpc"
        config.mission.nmpc_update_interval_steps = 1
        config.mission.nmpc_max_wall_time_ms = 0.001
        plan = plan_global_coverage(config)
        runtime = SwarmRuntime(config, plan)
        runtime.nmpc_by_agent[0] = _SlowFeasibleNMPC(runtime.horizon_steps)
        result = runtime.control_step(
            AgentRuntimeState(
                agent_id=0,
                time=0.0,
                state3=config.fleet.initial_states_3dof[0],
                state6=config.fleet.initial_states_6dof[0],
            )
        )
        self.assertEqual(runtime.profiler.nmpc_called_count, 1)
        self.assertEqual(runtime.profiler.timeout_count, 1)
        self.assertGreaterEqual(runtime.profiler.fallback_count, 1)
        self.assertEqual(result.safety_status.mode, "nmpc_timeout_tracker_fallback")
        self.assertIn("nmpc timeout fallback", result.safety_status.warnings)

    def test_thread_backend_control_steps_batches_nmpc_solves(self) -> None:
        config = build_test_config()
        config.mission.control_mode = "full_nmpc"
        config.mission.nmpc_parallel_backend = "thread"
        config.mission.nmpc_max_wall_time_ms = 1000.0
        plan = plan_global_coverage(config)
        runtime = SwarmRuntime(config, plan)
        for agent_id in range(config.fleet.num_agents or 0):
            runtime.nmpc_by_agent[agent_id] = _SlowFeasibleNMPC(runtime.horizon_steps, sleep_s=0.001)
        runtime_states = {
            agent_id: AgentRuntimeState(
                agent_id=agent_id,
                time=0.0,
                state3=config.fleet.initial_states_3dof[agent_id],
                state6=config.fleet.initial_states_6dof[agent_id],
            )
            for agent_id in range(config.fleet.num_agents or 0)
        }

        results = runtime.control_steps(runtime_states)

        self.assertEqual(set(results), set(runtime_states))
        self.assertEqual(runtime.profiler.nmpc_called_count, config.fleet.num_agents)
        self.assertTrue(all(result.safety_status.mode.startswith("full_nmpc") for result in results.values()))
        self.assertEqual(runtime.nmpc_parallel_backend, "thread")
        self.assertEqual(runtime.nmpc_parallel_backend_effective, "thread")

    def test_thread_backend_with_casadi_controllers_uses_safe_serial_fallback(self) -> None:
        config = build_test_config()
        config.mission.control_mode = "hybrid_nmpc"
        config.mission.nmpc_parallel_backend = "thread"
        config.mission.nmpc_update_interval_steps = 100
        plan = plan_global_coverage(config)
        runtime = SwarmRuntime(config, plan)
        runtime._last_nmpc_step = {agent_id: 0 for agent_id in range(config.fleet.num_agents or 0)}
        runtime_states = {
            agent_id: AgentRuntimeState(
                agent_id=agent_id,
                time=runtime.dt,
                state3=config.fleet.initial_states_3dof[agent_id],
                state6=config.fleet.initial_states_6dof[agent_id],
            )
            for agent_id in range(config.fleet.num_agents or 0)
        }

        results = runtime.control_steps(runtime_states)

        self.assertEqual(set(results), set(runtime_states))
        self.assertEqual(runtime.profiler.nmpc_called_count, 0)
        self.assertEqual(runtime.nmpc_parallel_backend, "thread")
        self.assertEqual(runtime.nmpc_parallel_backend_effective, "serial_casadi_thread_disabled")

    def test_nmpc_solve_request_is_pickleable_for_process_backend(self) -> None:
        config = build_test_config()
        request = NMPCSolveRequest(
            agent_id=0,
            state=config.fleet.initial_states_3dof[0],
            previous_control=ControlInput.zero(),
            ref_window=[
                TrajectorySample(time=0.0, x=1.0, y=1.0, psi=0.0, u_ref=0.0, r_ref=0.0, segment_type="test")
            ],
            preferred_velocities=np.zeros((2, 2), dtype=float),
            neighbor_predictions=[],
            obstacle_predictions=[],
            mismatch=np.zeros(3, dtype=float),
            safe_distance=config.safety.d_safe,
        )

        restored = pickle.loads(pickle.dumps(request))

        self.assertEqual(restored.agent_id, request.agent_id)
        self.assertEqual(restored.state.x, request.state.x)
        self.assertEqual(restored.preferred_velocities.shape, (2, 2))

    def test_process_backend_hard_timeout_falls_back_without_blocking_runtime(self) -> None:
        config = build_test_config()
        config.mission.control_mode = "full_nmpc"
        config.mission.nmpc_parallel_backend = "process"
        config.mission.nmpc_max_wall_time_ms = 5.0
        plan = plan_global_coverage(config)
        runtime = SwarmRuntime(config, plan)
        runtime.nmpc_process_backend = _TimeoutProcessBackend()
        runtime_states = {
            agent_id: AgentRuntimeState(
                agent_id=agent_id,
                time=0.0,
                state3=config.fleet.initial_states_3dof[agent_id],
                state6=config.fleet.initial_states_6dof[agent_id],
            )
            for agent_id in range(config.fleet.num_agents or 0)
        }

        results = runtime.control_steps(runtime_states)

        self.assertEqual(set(results), set(runtime_states))
        self.assertEqual(runtime.nmpc_parallel_backend_effective, "process")
        self.assertEqual(runtime.profiler.nmpc_called_count, config.fleet.num_agents)
        self.assertEqual(runtime.profiler.nmpc_hard_timeout_count, config.fleet.num_agents)
        self.assertTrue(all("nmpc hard timeout fallback" in result.safety_status.warnings for result in results.values()))

    def test_closed_loop_simulation_smoke(self) -> None:
        config = build_test_config()
        config.mission.control_mode = "fast_tracker"
        plan = plan_global_coverage(config)
        log = simulate_swarm_closed_loop(config, planning_result=plan, total_time=0.4)
        self.assertGreaterEqual(len(log.frames), 2)
        self.assertGreaterEqual(log.final_coverage_fraction, 0.0)
        self.assertIn(0, log.frames[-1].agent_states)
        self.assertIn("summary", log.runtime_profile)
        self.assertEqual(log.runtime_profile["summary"]["nmpc_called_count"], 0.0)
        self.assertEqual(log.runtime_profile["summary"]["dynamics_integration_method"], "rk4")
        self.assertEqual(log.runtime_profile["summary"]["nmpc_integration_method"], "rk4")
        self.assertFalse(log.runtime_profile["summary"]["plant_nmpc_integration_mismatch"])
        self.assertIn("cbf_filter_called_count", log.runtime_profile["summary"])
        self.assertGreater(log.runtime_profile["summary"]["cbf_filter_called_count"], 0.0)

    def test_residual_detection_interval_is_step_level_not_agent_level(self) -> None:
        config = build_test_config()
        config.mission.control_mode = "fast_tracker"
        config.mission.coverage_residual_interval_steps = 10
        plan = plan_global_coverage(config)
        log = simulate_swarm_closed_loop(config, planning_result=plan, total_time=0.4)
        summary = log.runtime_profile["summary"]
        self.assertEqual(summary["residual_detection_count"], 1.0)
        self.assertGreater(summary["coverage_update_count"], summary["residual_detection_count"])
        self.assertIn("avg_coverage_update_time_ms", summary)

    def test_3dof_integrators_keep_zero_state_stationary(self) -> None:
        model = _undamped_model()
        state = State3DOF(x=1.0, y=2.0, psi=0.3, u=0.0, v=0.0, r=0.0)
        for method in ("explicit_euler", "semi_implicit_euler", "rk4"):
            next_state = model.step(state, ControlInput.zero(), 0.1, np.zeros(3), integration_method=method)
            self.assertAlmostEqual(next_state.x, state.x)
            self.assertAlmostEqual(next_state.y, state.y)
            self.assertAlmostEqual(next_state.psi, state.psi)
            self.assertAlmostEqual(next_state.u, 0.0)
            self.assertAlmostEqual(next_state.v, 0.0)
            self.assertAlmostEqual(next_state.r, 0.0)

    def test_rk4_matches_constant_velocity_straight_line_solution(self) -> None:
        model = _undamped_model()
        state = State3DOF(x=1.0, y=-2.0, psi=0.4, u=2.0, v=0.0, r=0.0)
        dt = 0.5
        next_state = model.step(state, ControlInput.zero(), dt, np.zeros(3), integration_method="rk4")
        self.assertAlmostEqual(next_state.x, state.x + state.u * math.cos(state.psi) * dt, places=9)
        self.assertAlmostEqual(next_state.y, state.y + state.u * math.sin(state.psi) * dt, places=9)
        self.assertAlmostEqual(next_state.psi, state.psi, places=9)

    def test_rk4_reduces_constant_yaw_rate_arc_error(self) -> None:
        model = _undamped_model()
        state = State3DOF(x=0.0, y=0.0, psi=0.0, u=2.0, v=0.0, r=0.6)
        dt = 0.5
        expected_x = state.u / state.r * math.sin(state.r * dt)
        expected_y = state.u / state.r * (1.0 - math.cos(state.r * dt))
        explicit = model.step(state, ControlInput.zero(), dt, np.zeros(3), integration_method="explicit_euler")
        semi = model.step(state, ControlInput.zero(), dt, np.zeros(3), integration_method="semi_implicit_euler")
        rk4 = model.step(state, ControlInput.zero(), dt, np.zeros(3), integration_method="rk4")
        explicit_error = math.hypot(explicit.x - expected_x, explicit.y - expected_y)
        semi_error = math.hypot(semi.x - expected_x, semi.y - expected_y)
        rk4_error = math.hypot(rk4.x - expected_x, rk4.y - expected_y)
        self.assertLess(rk4_error, explicit_error)
        self.assertLess(rk4_error, semi_error)

    def test_explicit_euler_uses_pre_update_pose_for_position(self) -> None:
        model = _undamped_model()
        state = State3DOF(x=0.0, y=0.0, psi=0.0, u=1.0, v=0.0, r=0.0)
        control = ControlInput(thrust=1.0, yaw_moment=1.0)
        next_state = model.step(state, control, 0.2, np.zeros(3), integration_method="explicit_euler")
        self.assertAlmostEqual(next_state.x, 0.2, places=9)
        self.assertAlmostEqual(next_state.y, 0.0, places=9)
        self.assertGreater(next_state.u, state.u)
        self.assertGreater(next_state.r, state.r)


class _SlowFeasibleNMPC:
    def __init__(self, horizon_steps: int, sleep_s: float = 0.002) -> None:
        self.horizon_steps = horizon_steps
        self.sleep_s = sleep_s

    def solve(self, **kwargs) -> NMPCResult:
        time.sleep(self.sleep_s)
        state = kwargs["state"]
        predicted_states = np.zeros((6, self.horizon_steps + 1), dtype=float)
        predicted_states[0, :] = state.x
        predicted_states[1, :] = state.y
        predicted_states[2, :] = state.psi
        predicted_states[3, :] = state.u
        predicted_states[4, :] = state.v
        predicted_states[5, :] = state.r
        return NMPCResult(
            feasible=True,
            control=ControlInput(thrust=0.0, yaw_moment=0.0),
            predicted_states=predicted_states,
            predicted_controls=np.zeros((2, self.horizon_steps), dtype=float),
            objective=0.0,
            solver_status="fake_success",
            min_margin=float("inf"),
        )


class _TimeoutProcessBackend:
    worker_restart_count = 1

    def solve_many(self, requests, timeout_ms):
        return {
            request.agent_id: NMPCBackendSolveResult(
                agent_id=request.agent_id,
                result=None,
                solve_time_ms=float(timeout_ms),
                timed_out=True,
                error="test_hard_timeout",
            )
            for request in requests
        }


def _undamped_model() -> USV3DOFModel:
    return USV3DOFModel(
        mass_u=1.0,
        mass_v=1.0,
        mass_r=1.0,
        damp_u=0.0,
        damp_v=0.0,
        damp_r=0.0,
        cross_coupling=0.0,
    )


if __name__ == "__main__":
    unittest.main()
