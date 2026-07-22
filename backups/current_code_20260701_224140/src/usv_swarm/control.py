from __future__ import annotations

import math
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import connected_components, rotated_rectangle_local_mask, unit_heading, wrap_angle
from .nmpc_backend import NMPCBackendSolveResult, NMPCSolveRequest, ProcessNMPCBackend
from .nmpc import CasadiNMPCController, create_nmpc_controller
from .safety_filter import SafetyFilterResult, filter_control_cbf_qp
from .schema import (
    AgentRuntimeState,
    ControlInput,
    ControlStepResult,
    CoverageResidual,
    CoverageState,
    DynamicObstacleSample,
    DynamicObstacleTrack,
    PlannerConfig,
    PlanningResult,
    Pose2D,
    SafetyStatus,
    State3DOF,
    TrajectoryReference,
    TrajectorySample,
)


@dataclass
class _SimResult:
    feasible: bool
    cost: float
    control: ControlInput
    predicted_samples: List[TrajectorySample]
    min_margin: float


@dataclass
class _ControlContext:
    agent_state: AgentRuntimeState
    ref_window: List[TrajectorySample]
    predictions: Dict[int, List[TrajectorySample]]
    obstacles: List[DynamicObstacleTrack]
    mismatch: np.ndarray
    delta_safe: float
    preferred_velocity: np.ndarray
    tracker: _SimResult
    step_index: int
    should_call_nmpc: bool


@dataclass
class _NmpcSolveOutcome:
    result: _SimResult
    solve_time_ms: float
    hard_timeout: bool = False
    error: str = ""


@dataclass
class ControlProfiler:
    control_step_count: int = 0
    nmpc_called_count: int = 0
    timeout_count: int = 0
    fallback_count: int = 0
    total_control_time_ms: float = 0.0
    total_nmpc_time_ms: float = 0.0
    max_control_time_ms: float = 0.0
    max_nmpc_time_ms: float = 0.0
    cbf_filter_called_count: int = 0
    cbf_filter_failed_count: int = 0
    cbf_slack_used_count: int = 0
    total_cbf_filter_time_ms: float = 0.0
    max_cbf_filter_time_ms: float = 0.0
    safety_min_margin: float = float("inf")
    nmpc_hard_timeout_count: int = 0
    nmpc_worker_restart_count: int = 0
    nmpc_solver_backend_requested: str = ""
    nmpc_solver_backend_effective: str = ""
    acados_available: bool = False
    acados_fallback_reason: str = ""

    def summary(self) -> Dict[str, float]:
        steps = max(self.control_step_count, 1)
        calls = max(self.nmpc_called_count, 1)
        return {
            "control_step_count": float(self.control_step_count),
            "nmpc_called_count": float(self.nmpc_called_count),
            "timeout_count": float(self.timeout_count),
            "fallback_count": float(self.fallback_count),
            "avg_control_time_ms": self.total_control_time_ms / steps,
            "avg_nmpc_solve_time_ms": self.total_nmpc_time_ms / calls if self.nmpc_called_count else 0.0,
            "max_control_time_ms": self.max_control_time_ms,
            "max_nmpc_solve_time_ms": self.max_nmpc_time_ms,
            "cbf_filter_called_count": float(self.cbf_filter_called_count),
            "cbf_filter_failed_count": float(self.cbf_filter_failed_count),
            "cbf_slack_used_count": float(self.cbf_slack_used_count),
            "avg_cbf_filter_time_ms": self.total_cbf_filter_time_ms / max(self.cbf_filter_called_count, 1) if self.cbf_filter_called_count else 0.0,
            "max_cbf_filter_time_ms": self.max_cbf_filter_time_ms,
            "safety_min_margin": self.safety_min_margin if self.safety_min_margin != float("inf") else float("inf"),
            "nmpc_hard_timeout_count": float(self.nmpc_hard_timeout_count),
            "nmpc_worker_restart_count": float(self.nmpc_worker_restart_count),
            "nmpc_solver_backend_requested": self.nmpc_solver_backend_requested,
            "nmpc_solver_backend_effective": self.nmpc_solver_backend_effective,
            "acados_available": self.acados_available,
            "acados_fallback_reason": self.acados_fallback_reason,
        }


class USV3DOFModel:
    def __init__(
        self,
        mass_u: float = 1.0,
        mass_v: float = 1.4,
        mass_r: float = 1.0,
        damp_u: float = 0.8,
        damp_v: float = 1.1,
        damp_r: float = 0.9,
        cross_coupling: float = 0.3,
    ) -> None:
        self.mass_u = mass_u
        self.mass_v = mass_v
        self.mass_r = mass_r
        self.damp_u = damp_u
        self.damp_v = damp_v
        self.damp_r = damp_r
        self.cross_coupling = cross_coupling

    def derivatives(self, state: State3DOF, control: ControlInput, mismatch: np.ndarray) -> np.ndarray:
        u_dot = (control.thrust - self.damp_u * state.u) / self.mass_u + mismatch[0]
        v_dot = (-self.damp_v * state.v + self.cross_coupling * state.r) / self.mass_v + mismatch[1]
        r_dot = (control.yaw_moment - self.damp_r * state.r) / self.mass_r + mismatch[2]
        x_dot = state.u * math.cos(state.psi) - state.v * math.sin(state.psi)
        y_dot = state.u * math.sin(state.psi) + state.v * math.cos(state.psi)
        return np.array([x_dot, y_dot, state.r, u_dot, v_dot, r_dot], dtype=float)

    def step(
        self,
        state: State3DOF,
        control: ControlInput,
        dt: float,
        mismatch: np.ndarray,
        integration_method: str = "rk4",
    ) -> State3DOF:
        method = _normalized_integration_method(integration_method)
        if method == "explicit_euler":
            return self._step_explicit_euler(state, control, dt, mismatch)
        if method == "semi_implicit_euler":
            return self._step_semi_implicit_euler(state, control, dt, mismatch)
        return self._step_rk4(state, control, dt, mismatch)

    def _step_explicit_euler(self, state: State3DOF, control: ControlInput, dt: float, mismatch: np.ndarray) -> State3DOF:
        derivative = self.derivatives(state, control, mismatch)
        vector = state.as_vector() + dt * derivative
        return _state_from_vector(vector)

    def _step_semi_implicit_euler(self, state: State3DOF, control: ControlInput, dt: float, mismatch: np.ndarray) -> State3DOF:
        derivative = self.derivatives(state, control, mismatch)
        u = state.u + dt * derivative[3]
        v = state.v + dt * derivative[4]
        r = state.r + dt * derivative[5]
        psi = wrap_angle(state.psi + dt * r)
        x_dot = u * math.cos(psi) - v * math.sin(psi)
        y_dot = u * math.sin(psi) + v * math.cos(psi)
        return State3DOF(x=state.x + dt * x_dot, y=state.y + dt * y_dot, psi=psi, u=u, v=v, r=r)

    def _step_rk4(self, state: State3DOF, control: ControlInput, dt: float, mismatch: np.ndarray) -> State3DOF:
        y0 = state.as_vector()
        k1 = self.derivatives(state, control, mismatch)
        k2 = self.derivatives(_state_from_vector(y0 + 0.5 * dt * k1), control, mismatch)
        k3 = self.derivatives(_state_from_vector(y0 + 0.5 * dt * k2), control, mismatch)
        k4 = self.derivatives(_state_from_vector(y0 + dt * k3), control, mismatch)
        return _state_from_vector(y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


class Parallel6DOFEstimator:
    def __init__(self, delta_safe_max: float) -> None:
        self.delta_safe_max = delta_safe_max

    def estimate(self, runtime_state: AgentRuntimeState) -> Tuple[np.ndarray, float]:
        if runtime_state.state6 is None:
            return np.zeros(3, dtype=float), 0.0
        mismatch = np.array(
            [
                0.2 * (runtime_state.state6.u - runtime_state.state3.u),
                0.2 * (runtime_state.state6.v - runtime_state.state3.v),
                0.2 * (runtime_state.state6.r - runtime_state.state3.r),
            ],
            dtype=float,
        )
        attitude_mag = math.hypot(runtime_state.state6.phi, runtime_state.state6.theta)
        delta_safe = min(self.delta_safe_max, 0.4 * attitude_mag + 0.5 * float(np.linalg.norm(mismatch)))
        return mismatch, delta_safe


class CoverageTracker:
    def __init__(self, config: PlannerConfig) -> None:
        resolution = min(config.footprint.width_wf / 2.0, config.footprint.length_lf / 2.0)
        resolution = max(resolution, 0.25)
        x_coords = np.arange(resolution / 2.0, config.mission.area_length_x + 1e-9, resolution)
        y_coords = np.arange(resolution / 2.0, config.mission.area_length_y + 1e-9, resolution)
        if x_coords.size == 0:
            x_coords = np.array([config.mission.area_length_x / 2.0], dtype=float)
        if y_coords.size == 0:
            y_coords = np.array([config.mission.area_length_y / 2.0], dtype=float)
        self.eta_cov = config.footprint.eta_cov
        self.footprint_length = config.footprint.length_lf
        self.footprint_width = config.footprint.width_wf
        self.update_count = 0
        self.updated_cell_count = 0
        self.total_update_time_ms = 0.0
        self.max_update_time_ms = 0.0
        self.residual_detection_count = 0
        self.total_residual_detection_time_ms = 0.0
        self.max_residual_detection_time_ms = 0.0
        self.state = CoverageState(
            resolution=resolution,
            x_coords=x_coords,
            y_coords=y_coords,
            coverage_ratio=np.zeros((y_coords.size, x_coords.size), dtype=float),
            covered=np.zeros((y_coords.size, x_coords.size), dtype=bool),
        )

    def update(self, pose: Pose2D) -> CoverageState:
        started = time.perf_counter()
        mask, row_slice, col_slice = rotated_rectangle_local_mask(
            self.state.x_coords,
            self.state.y_coords,
            pose.x,
            pose.y,
            pose.psi,
            self.footprint_length,
            self.footprint_width,
        )
        if mask.size:
            local_ratio = self.state.coverage_ratio[row_slice, col_slice]
            local_covered = self.state.covered[row_slice, col_slice]
            local_ratio[mask] = 1.0
            local_covered[:, :] = local_ratio >= self.eta_cov
            self.updated_cell_count += int(mask.size)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.update_count += 1
        self.total_update_time_ms += elapsed_ms
        self.max_update_time_ms = max(self.max_update_time_ms, elapsed_ms)
        return self.state

    def detect_residuals(self, min_component_cells: int = 3) -> List[CoverageResidual]:
        started = time.perf_counter()
        residual_mask = ~self.state.covered
        components = connected_components(residual_mask)
        residuals: List[CoverageResidual] = []
        for residual_id, cells in enumerate(components):
            if len(cells) < min_component_cells:
                continue
            xs = [self.state.x_coords[col] for row, col in cells]
            ys = [self.state.y_coords[row] for row, col in cells]
            residuals.append(
                CoverageResidual(
                    residual_id=residual_id,
                    cells=cells,
                    centroid=(float(np.mean(xs)), float(np.mean(ys))),
                    bounds=(min(xs), min(ys), max(xs), max(ys)),
                )
            )
        self.state.residual_components = residuals
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.residual_detection_count += 1
        self.total_residual_detection_time_ms += elapsed_ms
        self.max_residual_detection_time_ms = max(self.max_residual_detection_time_ms, elapsed_ms)
        return residuals

    def profiler_summary(self) -> Dict[str, float]:
        update_count = max(self.update_count, 1)
        residual_count = max(self.residual_detection_count, 1)
        return {
            "coverage_update_count": float(self.update_count),
            "coverage_updated_cell_count": float(self.updated_cell_count),
            "avg_coverage_update_time_ms": self.total_update_time_ms / update_count,
            "max_coverage_update_time_ms": self.max_update_time_ms,
            "residual_detection_count": float(self.residual_detection_count),
            "avg_residual_detection_time_ms": self.total_residual_detection_time_ms / residual_count if self.residual_detection_count else 0.0,
            "max_residual_detection_time_ms": self.max_residual_detection_time_ms,
        }


class SwarmRuntime:
    def __init__(self, config: PlannerConfig, planning_result: PlanningResult) -> None:
        self.config = config
        self.planning_result = planning_result
        self.model = USV3DOFModel()
        self.estimator = Parallel6DOFEstimator(delta_safe_max=config.safety.delta_safe_max)
        self.coverage = CoverageTracker(config)
        self.dt = 1.0 / max(config.mission.local_control_hz, 1e-6)
        self.control_mode = _normalized_control_mode(config.mission.control_mode)
        self.integration_method = _normalized_integration_method(config.mission.dynamics_integration_method)
        self.nmpc_integration_method = _normalized_nmpc_integration_method(config.mission.nmpc_integration_method)
        self.nmpc_update_interval_steps = max(1, int(config.mission.nmpc_update_interval_steps))
        self.nmpc_max_wall_time_ms = max(float(config.mission.nmpc_max_wall_time_ms), 0.0)
        self.nmpc_parallel_backend = _normalized_nmpc_parallel_backend(config.mission.nmpc_parallel_backend)
        self.nmpc_parallel_backend_effective = self.nmpc_parallel_backend
        self.nmpc_solver_backend_requested = _normalized_nmpc_solver_backend(config.mission.nmpc_solver_backend)
        self.nmpc_solver_backend_effective = "none"
        self.acados_available = False
        self.acados_fallback_reason = ""
        self.horizon_steps = min(
            max(int(config.mission.nmpc_horizon_steps_cap), 1),
            max(6, int(round(max(config.mission.nmpc_horizon_seconds, self.dt) / self.dt))),
        )
        self.neighbor_radius = 4.0 * config.safety.d_safe
        self.max_neighbors = max(1, (config.fleet.num_agents or 1) - 1)
        self.max_obstacles = 4
        self.nmpc_process_backend = self._build_process_nmpc_backend() if self.control_mode != "fast_tracker" and self.nmpc_parallel_backend == "process" else None
        self.nmpc_by_agent = self._build_nmpc_controllers() if self.control_mode != "fast_tracker" and self.nmpc_process_backend is None else {}
        self._last_nmpc_step = {agent_id: -10**9 for agent_id in range(config.fleet.num_agents or 0)}
        self._last_valid_nmpc: Dict[int, _SimResult] = {}
        self.profiler = ControlProfiler()
        self._sync_nmpc_solver_profile()
        self.last_agent_profile: Dict[str, float | int | str | bool] = {}

    def close(self) -> None:
        if self.nmpc_process_backend is not None:
            self.nmpc_process_backend.close(terminate_workers=True)

    def _sync_nmpc_solver_profile(self) -> None:
        controllers = list(self.nmpc_by_agent.values())
        if controllers:
            effective = sorted({str(getattr(item, "solver_backend_effective", "unknown")) for item in controllers})
            fallbacks = [str(getattr(item, "acados_fallback_reason", "")) for item in controllers if getattr(item, "acados_fallback_reason", "")]
            self.nmpc_solver_backend_effective = ",".join(effective)
            self.acados_available = any(bool(getattr(item, "acados_available", False)) for item in controllers)
            self.acados_fallback_reason = fallbacks[0] if fallbacks else ""
        elif self.control_mode == "fast_tracker":
            self.nmpc_solver_backend_effective = "none"
            self.acados_available = False
            self.acados_fallback_reason = ""
        elif self.nmpc_process_backend is not None:
            if self.nmpc_solver_backend_effective in {"", "none"}:
                self.nmpc_solver_backend_effective = "process_worker_pending"
        self.profiler.nmpc_solver_backend_requested = self.nmpc_solver_backend_requested
        self.profiler.nmpc_solver_backend_effective = self.nmpc_solver_backend_effective
        self.profiler.acados_available = self.acados_available
        self.profiler.acados_fallback_reason = self.acados_fallback_reason

    def _build_nmpc_controllers(self) -> Dict[int, object]:
        return {
            agent_id: create_nmpc_controller(
                config=self.config,
                horizon_steps=self.horizon_steps,
                dt=self.dt,
                max_neighbors=self.max_neighbors,
                max_obstacles=self.max_obstacles,
                mass_u=self.model.mass_u,
                mass_v=self.model.mass_v,
                mass_r=self.model.mass_r,
                damp_u=self.model.damp_u,
                damp_v=self.model.damp_v,
                damp_r=self.model.damp_r,
                cross_coupling=self.model.cross_coupling,
            )
            for agent_id in range(self.config.fleet.num_agents or 0)
        }

    def _build_process_nmpc_backend(self) -> ProcessNMPCBackend:
        return ProcessNMPCBackend(
            self.config,
            horizon_steps=self.horizon_steps,
            dt=self.dt,
            max_neighbors=self.max_neighbors,
            max_obstacles=self.max_obstacles,
            model_params={
                "mass_u": self.model.mass_u,
                "mass_v": self.model.mass_v,
                "mass_r": self.model.mass_r,
                "damp_u": self.model.damp_u,
                "damp_v": self.model.damp_v,
                "damp_r": self.model.damp_r,
                "cross_coupling": self.model.cross_coupling,
            },
            max_workers=max(1, self.config.fleet.num_agents or 1),
        )

    def control_step(
        self,
        agent_state: AgentRuntimeState,
        shared_predictions: Optional[Dict[int, Sequence[TrajectorySample]]] = None,
        obstacle_tracks: Optional[Sequence[DynamicObstacleTrack]] = None,
    ) -> ControlStepResult:
        self.nmpc_parallel_backend_effective = "process" if self.nmpc_parallel_backend == "process" and self.nmpc_process_backend is not None else "serial"
        wall_started = time.perf_counter()
        context = self._prepare_control_context(agent_state, shared_predictions, obstacle_tracks)
        outcome = self._solve_nmpc_for_context(context) if context.should_call_nmpc else None
        return self._finalize_control_context(context, outcome, wall_started)

    def control_steps(
        self,
        runtime_states: Dict[int, AgentRuntimeState],
        shared_predictions: Optional[Dict[int, Sequence[TrajectorySample]]] = None,
        obstacle_tracks: Optional[Sequence[DynamicObstacleTrack]] = None,
    ) -> Dict[int, ControlStepResult]:
        if self.nmpc_parallel_backend == "process" and self.nmpc_process_backend is not None:
            return self._control_steps_process_backend(runtime_states, shared_predictions, obstacle_tracks)

        if self.nmpc_parallel_backend != "thread" or len(runtime_states) <= 1 or not self._thread_backend_can_run_nmpc():
            effective_backend = (
                "serial_casadi_thread_disabled"
                if self.nmpc_parallel_backend == "thread" and not self._thread_backend_can_run_nmpc()
                else "serial"
            )
            self.nmpc_parallel_backend_effective = effective_backend
            predictions = dict(shared_predictions or {})
            results: Dict[int, ControlStepResult] = {}
            for agent_id in sorted(runtime_states):
                result = self.control_step(runtime_states[agent_id], predictions if predictions else None, obstacle_tracks)
                results[agent_id] = result
                predictions[agent_id] = result.predicted_samples
            self.nmpc_parallel_backend_effective = effective_backend
            return results

        self.nmpc_parallel_backend_effective = "thread"
        started_by_agent: Dict[int, float] = {}
        contexts: Dict[int, _ControlContext] = {}
        for agent_id in sorted(runtime_states):
            started_by_agent[agent_id] = time.perf_counter()
            contexts[agent_id] = self._prepare_control_context(runtime_states[agent_id], shared_predictions, obstacle_tracks)

        outcomes: Dict[int, _NmpcSolveOutcome] = {}
        to_solve = [context for context in contexts.values() if context.should_call_nmpc]
        if to_solve:
            with ThreadPoolExecutor(max_workers=max(1, min(len(to_solve), self.max_neighbors + 1))) as executor:
                future_to_agent = {executor.submit(self._solve_nmpc_for_context, context): context.agent_state.agent_id for context in to_solve}
                for future in as_completed(future_to_agent):
                    outcomes[future_to_agent[future]] = future.result()

        results: Dict[int, ControlStepResult] = {}
        for agent_id in sorted(contexts):
            results[agent_id] = self._finalize_control_context(contexts[agent_id], outcomes.get(agent_id), started_by_agent[agent_id])
        return results

    def _thread_backend_can_run_nmpc(self) -> bool:
        """Avoid threaded IPOPT/Opti solves; CasADi controllers are not reliably thread-safe."""

        return all(not isinstance(controller, CasadiNMPCController) for controller in self.nmpc_by_agent.values())

    def _control_steps_process_backend(
        self,
        runtime_states: Dict[int, AgentRuntimeState],
        shared_predictions: Optional[Dict[int, Sequence[TrajectorySample]]],
        obstacle_tracks: Optional[Sequence[DynamicObstacleTrack]],
    ) -> Dict[int, ControlStepResult]:
        self.nmpc_parallel_backend_effective = "process"
        started_by_agent: Dict[int, float] = {}
        contexts: Dict[int, _ControlContext] = {}
        for agent_id in sorted(runtime_states):
            started_by_agent[agent_id] = time.perf_counter()
            contexts[agent_id] = self._prepare_control_context(runtime_states[agent_id], shared_predictions, obstacle_tracks)

        outcomes: Dict[int, _NmpcSolveOutcome] = {}
        to_solve = [context for context in contexts.values() if context.should_call_nmpc]
        if to_solve and self.nmpc_process_backend is not None:
            requests = [self._build_nmpc_solve_request(context) for context in to_solve]
            backend_results = self.nmpc_process_backend.solve_many(requests, timeout_ms=self.nmpc_max_wall_time_ms)
            self.profiler.nmpc_worker_restart_count = self.nmpc_process_backend.worker_restart_count
            for context in to_solve:
                backend_result = backend_results.get(context.agent_state.agent_id)
                if backend_result is None:
                    outcomes[context.agent_state.agent_id] = _NmpcSolveOutcome(
                        result=context.tracker,
                        solve_time_ms=self.nmpc_max_wall_time_ms,
                        hard_timeout=True,
                        error="missing_process_result",
                    )
                    continue
                outcomes[context.agent_state.agent_id] = self._process_backend_result_to_outcome(context, backend_result)

        results: Dict[int, ControlStepResult] = {}
        for agent_id in sorted(contexts):
            results[agent_id] = self._finalize_control_context(contexts[agent_id], outcomes.get(agent_id), started_by_agent[agent_id])
        return results

    def _prepare_control_context(
        self,
        agent_state: AgentRuntimeState,
        shared_predictions: Optional[Dict[int, Sequence[TrajectorySample]]],
        obstacle_tracks: Optional[Sequence[DynamicObstacleTrack]],
    ) -> _ControlContext:
        ref_window = self._reference_window(agent_state.agent_id, agent_state.time)
        predictions = self._normalize_predictions(shared_predictions, agent_state.time, exclude_agent=agent_state.agent_id)
        obstacles = list(obstacle_tracks or [])
        mismatch, delta_safe = self.estimator.estimate(agent_state)
        preferred_velocity = self._compute_rvo_velocity(agent_state, ref_window, predictions, obstacles)
        tracker = self._solve_fast_tracker(agent_state, ref_window, predictions, obstacles, mismatch, delta_safe, preferred_velocity)
        step_index = int(round(agent_state.time / max(self.dt, 1e-9)))
        should_call = self._should_call_nmpc(agent_state, tracker, predictions, obstacles, delta_safe, step_index)
        return _ControlContext(
            agent_state=agent_state,
            ref_window=ref_window,
            predictions=predictions,
            obstacles=obstacles,
            mismatch=mismatch,
            delta_safe=delta_safe,
            preferred_velocity=preferred_velocity,
            tracker=tracker,
            step_index=step_index,
            should_call_nmpc=should_call,
        )

    def _solve_nmpc_for_context(self, context: _ControlContext) -> _NmpcSolveOutcome:
        if self.nmpc_parallel_backend == "process" and self.nmpc_process_backend is not None:
            request = self._build_nmpc_solve_request(context)
            backend_results = self.nmpc_process_backend.solve_many([request], timeout_ms=self.nmpc_max_wall_time_ms)
            self.profiler.nmpc_worker_restart_count = self.nmpc_process_backend.worker_restart_count
            backend_result = backend_results.get(context.agent_state.agent_id)
            if backend_result is None:
                return _NmpcSolveOutcome(
                    result=context.tracker,
                    solve_time_ms=self.nmpc_max_wall_time_ms,
                    hard_timeout=True,
                    error="missing_process_result",
                )
            return self._process_backend_result_to_outcome(context, backend_result)
        solve_started = time.perf_counter()
        result = self._solve_true_nmpc(
            context.agent_state,
            context.ref_window,
            context.predictions,
            context.obstacles,
            context.mismatch,
            context.delta_safe,
            context.preferred_velocity,
        )
        return _NmpcSolveOutcome(result=result, solve_time_ms=(time.perf_counter() - solve_started) * 1000.0)

    def _build_nmpc_solve_request(self, context: _ControlContext) -> NMPCSolveRequest:
        preferred_horizon = np.repeat(context.preferred_velocity.reshape(1, 2), self.horizon_steps, axis=0)
        return NMPCSolveRequest(
            agent_id=context.agent_state.agent_id,
            state=context.agent_state.state3,
            previous_control=context.agent_state.previous_control,
            ref_window=list(context.ref_window),
            preferred_velocities=preferred_horizon,
            neighbor_predictions=[list(samples) for samples in self._nearest_neighbor_predictions(context.agent_state, context.predictions)],
            obstacle_predictions=self._predict_obstacles(context.obstacles, context.agent_state.time),
            mismatch=context.mismatch,
            safe_distance=self.config.safety.d_safe + context.delta_safe,
        )

    def _process_backend_result_to_outcome(
        self,
        context: _ControlContext,
        backend_result: NMPCBackendSolveResult,
    ) -> _NmpcSolveOutcome:
        self._record_process_solver_metadata(backend_result)
        if backend_result.timed_out:
            return _NmpcSolveOutcome(
                result=context.tracker,
                solve_time_ms=max(backend_result.solve_time_ms, self.nmpc_max_wall_time_ms + 1.0),
                hard_timeout=True,
                error=backend_result.error,
            )
        if backend_result.result is None:
            failed = _SimResult(
                feasible=False,
                cost=float("inf"),
                control=ControlInput.zero(),
                predicted_samples=context.tracker.predicted_samples,
                min_margin=float("-inf"),
            )
            return _NmpcSolveOutcome(result=failed, solve_time_ms=backend_result.solve_time_ms, error=backend_result.error)
        return _NmpcSolveOutcome(
            result=self._nmpc_result_to_sim_result(context.agent_state, backend_result.result),
            solve_time_ms=backend_result.solve_time_ms,
            error=backend_result.error,
        )

    def _record_process_solver_metadata(self, backend_result: NMPCBackendSolveResult) -> None:
        if backend_result.solver_backend_effective:
            self.nmpc_solver_backend_effective = backend_result.solver_backend_effective
        self.acados_available = bool(backend_result.acados_available)
        if backend_result.acados_fallback_reason:
            self.acados_fallback_reason = backend_result.acados_fallback_reason
        self._sync_nmpc_solver_profile()

    def _finalize_control_context(
        self,
        context: _ControlContext,
        nmpc_outcome: Optional[_NmpcSolveOutcome],
        wall_started: float,
    ) -> ControlStepResult:
        agent_state = context.agent_state
        best = context.tracker
        mode = self.control_mode if self.control_mode == "fast_tracker" else "hybrid_tracker"
        nmpc_called = nmpc_outcome is not None
        nmpc_time_ms = nmpc_outcome.solve_time_ms if nmpc_outcome is not None else 0.0
        if nmpc_outcome is not None:
            nmpc_result = nmpc_outcome.result
            self.profiler.nmpc_called_count += 1
            self.profiler.total_nmpc_time_ms += nmpc_time_ms
            self.profiler.max_nmpc_time_ms = max(self.profiler.max_nmpc_time_ms, nmpc_time_ms)
            if nmpc_outcome.hard_timeout:
                self.profiler.nmpc_hard_timeout_count += 1
            timed_out = nmpc_outcome.hard_timeout or (self.nmpc_max_wall_time_ms > 0.0 and nmpc_time_ms > self.nmpc_max_wall_time_ms)
            if nmpc_result.feasible and not timed_out:
                best = nmpc_result
                mode = "full_nmpc" if self.control_mode == "full_nmpc" else "hybrid_nmpc"
                self._last_valid_nmpc[agent_state.agent_id] = nmpc_result
                self._last_nmpc_step[agent_state.agent_id] = context.step_index
            else:
                self.profiler.fallback_count += 1
                mode = "nmpc_timeout_tracker_fallback" if timed_out else "nmpc_infeasible_tracker_fallback"
                if timed_out:
                    self.profiler.timeout_count += 1
                self._last_nmpc_step[agent_state.agent_id] = context.step_index

        if not best.feasible:
            self.profiler.fallback_count += 1
            best = self._safe_hold(agent_state, context.ref_window, context.predictions, context.obstacles, context.mismatch, context.delta_safe)
            mode = "degraded_safe_hold"

        safety_filter_result = self._apply_safety_filter(agent_state, best.control, context)
        safety_filter_warnings: List[str] = []
        if safety_filter_result.filtered or not safety_filter_result.feasible:
            best = self._rollout_control(
                agent_state,
                safety_filter_result.control,
                context.ref_window,
                context.predictions,
                context.obstacles,
                context.mismatch,
                context.delta_safe,
                preferred_velocity=context.preferred_velocity,
            )
            mode = f"{mode}+cbf_filtered" if safety_filter_result.feasible else f"{mode}+cbf_limited"
            safety_filter_warnings.append("cbf safety filter adjusted control")
        if safety_filter_result.active_constraints:
            safety_filter_warnings.append("cbf active constraints: " + ",".join(safety_filter_result.active_constraints[:4]))
        if not safety_filter_result.feasible:
            safety_filter_warnings.append("cbf filter could not find strictly safe candidate")
        if safety_filter_result.slack_used:
            safety_filter_warnings.append("cbf slack used")
        best.min_margin = min(best.min_margin, safety_filter_result.min_predicted_margin)

        self.coverage.update(agent_state.state3.pose())
        warnings: List[str] = []
        if nmpc_called and nmpc_time_ms > self.nmpc_max_wall_time_ms > 0.0:
            warnings.append("nmpc timeout fallback")
        if nmpc_outcome is not None and nmpc_outcome.hard_timeout:
            warnings.append("nmpc hard timeout fallback")
        if nmpc_outcome is not None and nmpc_outcome.error:
            warnings.append(f"nmpc backend error: {nmpc_outcome.error}")
        if self.control_mode != "full_nmpc" and not nmpc_called:
            warnings.append("nmpc skipped by scheduler")
        if best.min_margin < self.config.safety.d_safe:
            warnings.append("reduced safety margin")
        if self.coverage.state.residual_components:
            warnings.append("residual coverage present")
        warnings.extend(safety_filter_warnings)
        elapsed_ms = (time.perf_counter() - wall_started) * 1000.0
        self.profiler.control_step_count += 1
        self.profiler.total_control_time_ms += elapsed_ms
        self.profiler.max_control_time_ms = max(self.profiler.max_control_time_ms, elapsed_ms)
        self.last_agent_profile = {
            "agent_id": agent_state.agent_id,
            "mode": mode,
            "control_time_ms": elapsed_ms,
            "nmpc_called": nmpc_called,
            "nmpc_solve_time_ms": nmpc_time_ms,
            "fallback_count": self.profiler.fallback_count,
            "timeout_count": self.profiler.timeout_count,
            "cbf_filter_called": self.profiler.cbf_filter_called_count,
            "cbf_filter_failed": self.profiler.cbf_filter_failed_count,
        }
        return ControlStepResult(
            cmd=best.control,
            safety_status=SafetyStatus(mode=mode, min_margin=best.min_margin, warnings=warnings),
            local_ref=context.ref_window,
            predicted_samples=best.predicted_samples,
        )

    def _apply_safety_filter(
        self,
        agent_state: AgentRuntimeState,
        nominal_control: ControlInput,
        context: _ControlContext,
    ) -> SafetyFilterResult:
        result = filter_control_cbf_qp(
            agent_state.state3,
            nominal_control,
            context.predictions,
            context.obstacles,
            self.config,
            current_time=agent_state.time,
            dt=self.dt,
            delta_safe=context.delta_safe,
        )
        self.profiler.cbf_filter_called_count += 1
        self.profiler.total_cbf_filter_time_ms += result.solve_time_ms
        self.profiler.max_cbf_filter_time_ms = max(self.profiler.max_cbf_filter_time_ms, result.solve_time_ms)
        if not result.feasible:
            self.profiler.cbf_filter_failed_count += 1
        if result.slack_used:
            self.profiler.cbf_slack_used_count += 1
        self.profiler.safety_min_margin = min(self.profiler.safety_min_margin, result.min_predicted_margin)
        return result

    def _reference_window(self, agent_id: int, current_time: float) -> List[TrajectorySample]:
        ref = self.planning_result.refs.get(agent_id, TrajectoryReference(agent_id=agent_id, samples=[], horizon_time=0.0))
        if not ref.samples:
            return []
        start_index = 0
        while start_index < len(ref.samples) and ref.samples[start_index].time < current_time:
            start_index += 1
        if start_index >= len(ref.samples):
            start_index = len(ref.samples) - 1
        window = list(ref.samples[start_index : start_index + self.horizon_steps])
        while len(window) < self.horizon_steps:
            window.append(ref.samples[-1])
        return window

    def _normalize_predictions(
        self,
        shared_predictions: Optional[Dict[int, Sequence[TrajectorySample]]],
        current_time: float,
        exclude_agent: Optional[int] = None,
    ) -> Dict[int, List[TrajectorySample]]:
        normalized: Dict[int, List[TrajectorySample]] = {}
        if shared_predictions:
            for agent_id, samples in shared_predictions.items():
                if exclude_agent is not None and agent_id == exclude_agent:
                    continue
                normalized[agent_id] = list(samples)
        else:
            for agent_id, reference in self.planning_result.refs.items():
                if exclude_agent is not None and agent_id == exclude_agent:
                    continue
                normalized[agent_id] = self._reference_window(agent_id, current_time)
        return normalized

    def _compute_rvo_velocity(
        self,
        agent_state: AgentRuntimeState,
        ref_window: Sequence[TrajectorySample],
        predictions: Dict[int, Sequence[TrajectorySample]],
        obstacle_tracks: Sequence[DynamicObstacleTrack],
    ) -> np.ndarray:
        if ref_window:
            target = np.array([ref_window[min(1, len(ref_window) - 1)].x, ref_window[min(1, len(ref_window) - 1)].y], dtype=float)
            desired = target - np.array([agent_state.state3.x, agent_state.state3.y], dtype=float)
            desired_speed = ref_window[0].u_ref
        else:
            desired = unit_heading(agent_state.state3.psi)
            desired_speed = self.config.fleet.cover_speed
        norm = np.linalg.norm(desired)
        if norm > 1e-9:
            preferred = desired / norm * min(desired_speed, self.config.fleet.cruise_speed)
        else:
            preferred = unit_heading(agent_state.state3.psi) * min(desired_speed, self.config.fleet.cruise_speed)

        current_position = np.array([agent_state.state3.x, agent_state.state3.y], dtype=float)
        repulsion = np.zeros(2, dtype=float)
        for samples in predictions.values():
            if not samples:
                continue
            other = np.array([samples[0].x, samples[0].y], dtype=float)
            offset = current_position - other
            distance = float(np.linalg.norm(offset))
            if distance < 1e-6 or distance > self.neighbor_radius:
                continue
            repulsion += ((self.neighbor_radius - distance) / self.neighbor_radius) * (offset / distance)

        for track in obstacle_tracks:
            sample = _sample_obstacle(track, agent_state.time)
            if sample is None:
                continue
            other = np.array([sample.x, sample.y], dtype=float)
            offset = current_position - other
            distance = float(np.linalg.norm(offset))
            threshold = self.neighbor_radius + track.radius
            if distance < 1e-6 or distance > threshold:
                continue
            repulsion += 1.5 * ((threshold - distance) / threshold) * (offset / distance)

        preferred = preferred + 0.6 * repulsion
        speed = float(np.linalg.norm(preferred))
        if speed > self.config.fleet.cruise_speed:
            preferred = preferred / speed * self.config.fleet.cruise_speed
        return preferred

    def _solve_fast_tracker(
        self,
        agent_state: AgentRuntimeState,
        ref_window: Sequence[TrajectorySample],
        predictions: Dict[int, Sequence[TrajectorySample]],
        obstacle_tracks: Sequence[DynamicObstacleTrack],
        mismatch: np.ndarray,
        delta_safe: float,
        preferred_velocity: np.ndarray,
    ) -> _SimResult:
        state = agent_state.state3
        target = self._lookahead_reference(state, ref_window)
        preferred_speed = float(np.linalg.norm(preferred_velocity))
        if preferred_speed > 1e-6:
            desired_heading = math.atan2(preferred_velocity[1], preferred_velocity[0])
            desired_speed = min(preferred_speed, self.config.fleet.cruise_speed)
        elif target is not None:
            desired_heading = math.atan2(target.y - state.y, target.x - state.x)
            desired_speed = min(max(target.u_ref, 0.0), self.config.fleet.cruise_speed)
        else:
            desired_heading = state.psi
            desired_speed = self.config.fleet.cover_speed

        heading_error = wrap_angle(desired_heading - state.psi)
        speed_scale = max(0.25, math.cos(min(abs(heading_error), math.pi / 2.0)))
        desired_speed *= speed_scale
        r_abs_max = max(self.config.fleet.turn_speed_max / max(self.config.fleet.min_turn_radius, 1e-6), 0.2)
        desired_r = float(np.clip(1.8 * heading_error, -r_abs_max, r_abs_max))
        thrust = self.model.mass_u * (desired_speed - state.u) / max(self.dt, 1e-6) + self.model.damp_u * state.u
        yaw_moment = self.model.mass_r * (desired_r - state.r) / max(self.dt, 1e-6) + self.model.damp_r * state.r
        control = ControlInput(
            thrust=float(np.clip(thrust, -self.config.fleet.max_thrust, self.config.fleet.max_thrust)),
            yaw_moment=float(np.clip(yaw_moment, -self.config.fleet.max_yaw_moment, self.config.fleet.max_yaw_moment)),
        )
        return self._rollout_control(
            agent_state,
            control,
            ref_window,
            predictions,
            obstacle_tracks,
            mismatch,
            delta_safe,
            preferred_velocity=preferred_velocity,
        )

    def _lookahead_reference(
        self,
        state: State3DOF,
        ref_window: Sequence[TrajectorySample],
    ) -> Optional[TrajectorySample]:
        if not ref_window:
            return None
        lookahead_distance = max(self.config.footprint.length_lf, self.config.fleet.cruise_speed * self.dt * 2.0)
        for sample in ref_window:
            if math.hypot(sample.x - state.x, sample.y - state.y) >= lookahead_distance:
                return sample
        return ref_window[-1]

    def _should_call_nmpc(
        self,
        agent_state: AgentRuntimeState,
        tracker: _SimResult,
        predictions: Dict[int, Sequence[TrajectorySample]],
        obstacle_tracks: Sequence[DynamicObstacleTrack],
        delta_safe: float,
        step_index: int,
    ) -> bool:
        if self.control_mode == "fast_tracker":
            return False
        if self.nmpc_parallel_backend == "process" and self.nmpc_process_backend is not None:
            pass
        elif agent_state.agent_id not in self.nmpc_by_agent:
            return False
        if self.control_mode == "full_nmpc":
            return True
        steps_since = step_index - self._last_nmpc_step.get(agent_state.agent_id, -10**9)
        interval_due = steps_since >= self.nmpc_update_interval_steps
        high_risk = not tracker.feasible
        return interval_due or high_risk

    def _solve_true_nmpc(
        self,
        agent_state: AgentRuntimeState,
        ref_window: Sequence[TrajectorySample],
        predictions: Dict[int, Sequence[TrajectorySample]],
        obstacle_tracks: Sequence[DynamicObstacleTrack],
        mismatch: np.ndarray,
        delta_safe: float,
        preferred_velocity: np.ndarray,
    ) -> _SimResult:
        preferred_horizon = np.repeat(preferred_velocity.reshape(1, 2), self.horizon_steps, axis=0)
        neighbor_predictions = self._nearest_neighbor_predictions(agent_state, predictions)
        obstacle_predictions = self._predict_obstacles(obstacle_tracks, agent_state.time)
        result = self.nmpc_by_agent[agent_state.agent_id].solve(
            state=agent_state.state3,
            previous_control=agent_state.previous_control,
            ref_window=ref_window,
            preferred_velocities=preferred_horizon,
            neighbor_predictions=neighbor_predictions,
            obstacle_predictions=obstacle_predictions,
            mismatch=mismatch,
            safe_distance=self.config.safety.d_safe + delta_safe,
        )
        return self._nmpc_result_to_sim_result(agent_state, result)

    def _nmpc_result_to_sim_result(self, agent_state: AgentRuntimeState, result) -> _SimResult:
        predicted_samples = [
            TrajectorySample(
                time=agent_state.time + idx * self.dt,
                x=float(result.predicted_states[0, idx]),
                y=float(result.predicted_states[1, idx]),
                psi=float(result.predicted_states[2, idx]),
                u_ref=float(result.predicted_states[3, idx]),
                r_ref=float(result.predicted_states[5, idx]),
                segment_type="predicted",
            )
            for idx in range(result.predicted_states.shape[1])
        ]
        return _SimResult(
            feasible=result.feasible,
            cost=result.objective,
            control=result.control,
            predicted_samples=predicted_samples,
            min_margin=result.min_margin,
        )

    def _nearest_neighbor_predictions(
        self,
        agent_state: AgentRuntimeState,
        predictions: Dict[int, Sequence[TrajectorySample]],
    ) -> List[Sequence[TrajectorySample]]:
        current = np.array([agent_state.state3.x, agent_state.state3.y], dtype=float)
        ranked: List[Tuple[float, Sequence[TrajectorySample]]] = []
        for samples in predictions.values():
            if not samples:
                continue
            distance = float(np.linalg.norm(current - np.array([samples[0].x, samples[0].y], dtype=float)))
            ranked.append((distance, samples))
        ranked.sort(key=lambda item: item[0])
        return [samples for _, samples in ranked[: self.max_neighbors]]

    def _predict_obstacles(
        self,
        obstacle_tracks: Sequence[DynamicObstacleTrack],
        current_time: float,
    ) -> List[List[Tuple[float, float, float]]]:
        predictions: List[List[Tuple[float, float, float]]] = []
        for track in obstacle_tracks[: self.max_obstacles]:
            samples: List[Tuple[float, float, float]] = []
            for step in range(self.horizon_steps + 1):
                sample = _sample_obstacle(track, current_time + step * self.dt)
                if sample is None:
                    continue
                samples.append((sample.x, sample.y, track.radius))
            if samples:
                predictions.append(samples)
        return predictions

    def _rollout_control(
        self,
        agent_state: AgentRuntimeState,
        control: ControlInput,
        ref_window: Sequence[TrajectorySample],
        predictions: Dict[int, Sequence[TrajectorySample]],
        obstacle_tracks: Sequence[DynamicObstacleTrack],
        mismatch: np.ndarray,
        delta_safe: float,
        preferred_velocity: np.ndarray,
    ) -> _SimResult:
        state = State3DOF(**vars(agent_state.state3))
        predicted_samples: List[TrajectorySample] = []
        total_cost = 0.0
        min_margin = float("inf")
        feasible = True
        previous_control = agent_state.previous_control

        for step in range(self.horizon_steps):
            time = agent_state.time + step * self.dt
            state = self.model.step(state, control, self.dt, mismatch, integration_method=self.integration_method)
            ref = ref_window[min(step, len(ref_window) - 1)] if ref_window else None
            if ref is not None:
                total_cost += self.config.weights.w_pos * ((state.x - ref.x) ** 2 + (state.y - ref.y) ** 2)
                total_cost += self.config.weights.w_psi * (wrap_angle(state.psi - ref.psi) ** 2)
                predicted_velocity = np.array(
                    [
                        state.u * math.cos(state.psi) - state.v * math.sin(state.psi),
                        state.u * math.sin(state.psi) + state.v * math.cos(state.psi),
                    ],
                    dtype=float,
                )
                total_cost += self.config.weights.w_vel * float(np.sum((predicted_velocity - preferred_velocity) ** 2))
            total_cost += self.config.weights.w_u * (control.thrust**2 + control.yaw_moment**2)
            total_cost += self.config.weights.w_du * (
                (control.thrust - previous_control.thrust) ** 2 + (control.yaw_moment - previous_control.yaw_moment) ** 2
            )
            previous_control = control

            hard_boundary_margin = min(
                state.x,
                self.config.mission.area_length_x - state.x,
                state.y,
                self.config.mission.area_length_y - state.y,
            )
            soft_boundary_margin = min(
                state.x - self.config.safety.boundary_margin_x,
                self.config.mission.area_length_x - state.x - self.config.safety.boundary_margin_x,
                state.y - self.config.safety.boundary_margin_y,
                self.config.mission.area_length_y - state.y - self.config.safety.boundary_margin_y,
            )
            min_margin = min(min_margin, soft_boundary_margin)
            if hard_boundary_margin < 0.0:
                feasible = False
                total_cost += self.config.weights.w_soft * abs(hard_boundary_margin) * 100.0
            elif soft_boundary_margin < 0.0:
                total_cost += self.config.weights.w_soft * abs(soft_boundary_margin) * 10.0

            safe_distance = self.config.safety.d_safe + delta_safe
            for samples in predictions.values():
                if not samples:
                    continue
                sample = samples[min(step, len(samples) - 1)]
                distance = math.hypot(state.x - sample.x, state.y - sample.y)
                margin = distance - safe_distance
                min_margin = min(min_margin, margin)
                if margin < 0.0:
                    feasible = False
                    total_cost += self.config.weights.w_soft * abs(margin) * 100.0

            for track in obstacle_tracks:
                sample = _sample_obstacle(track, time)
                if sample is None:
                    continue
                distance = math.hypot(state.x - sample.x, state.y - sample.y)
                margin = distance - (safe_distance + track.radius)
                min_margin = min(min_margin, margin)
                if margin < 0.0:
                    feasible = False
                    total_cost += self.config.weights.w_soft * abs(margin) * 100.0

            predicted_samples.append(
                TrajectorySample(
                    time=time,
                    x=state.x,
                    y=state.y,
                    psi=state.psi,
                    u_ref=state.u,
                    r_ref=state.r,
                    segment_type="predicted",
                )
            )

        return _SimResult(feasible=feasible, cost=total_cost, control=control, predicted_samples=predicted_samples, min_margin=min_margin)

    def _safe_hold(
        self,
        agent_state: AgentRuntimeState,
        ref_window: Sequence[TrajectorySample],
        predictions: Dict[int, Sequence[TrajectorySample]],
        obstacle_tracks: Sequence[DynamicObstacleTrack],
        mismatch: np.ndarray,
        delta_safe: float,
    ) -> _SimResult:
        current = np.array([agent_state.state3.x, agent_state.state3.y], dtype=float)
        threat_vector = np.zeros(2, dtype=float)
        nearest_margin = float("inf")
        safe_distance = self.config.safety.d_safe + delta_safe
        for samples in predictions.values():
            if not samples:
                continue
            other = np.array([samples[0].x, samples[0].y], dtype=float)
            offset = current - other
            distance = float(np.linalg.norm(offset))
            nearest_margin = min(nearest_margin, distance - safe_distance)
            if distance > 1e-9:
                threat_vector += offset / distance
        for track in obstacle_tracks:
            sample = _sample_obstacle(track, agent_state.time)
            if sample is None:
                continue
            other = np.array([sample.x, sample.y], dtype=float)
            offset = current - other
            distance = float(np.linalg.norm(offset))
            nearest_margin = min(nearest_margin, distance - safe_distance - track.radius)
            if distance > 1e-9:
                threat_vector += 1.5 * offset / distance
        if np.linalg.norm(threat_vector) < 1e-9:
            center = np.array(
                [self.config.mission.area_length_x / 2.0 - current[0], self.config.mission.area_length_y / 2.0 - current[1]],
                dtype=float,
            )
            threat_vector = center if np.linalg.norm(center) > 1e-9 else unit_heading(agent_state.state3.psi)
        desired_heading = math.atan2(threat_vector[1], threat_vector[0])
        control = ControlInput(
            thrust=float(np.clip(-1.2 * agent_state.state3.u, -self.config.fleet.max_thrust, self.config.fleet.max_thrust)),
            yaw_moment=float(
                np.clip(
                    2.5 * wrap_angle(desired_heading - agent_state.state3.psi) - 0.8 * agent_state.state3.r,
                    -self.config.fleet.max_yaw_moment,
                    self.config.fleet.max_yaw_moment,
                )
            ),
        )
        result = self._rollout_control(
            agent_state,
            control,
            ref_window,
            predictions,
            obstacle_tracks,
            mismatch,
            delta_safe,
            preferred_velocity=np.zeros(2, dtype=float),
        )
        result.feasible = True
        result.min_margin = nearest_margin if nearest_margin != float("inf") else result.min_margin
        return result


def _normalized_control_mode(control_mode: str) -> str:
    mode = str(control_mode or "hybrid_nmpc").strip().lower()
    allowed = {"fast_tracker", "hybrid_nmpc", "full_nmpc"}
    if mode not in allowed:
        raise ValueError(f"Unsupported control_mode '{control_mode}'. Expected one of {sorted(allowed)}")
    return mode


def _normalized_integration_method(integration_method: str) -> str:
    method = str(integration_method or "rk4").strip().lower()
    allowed = {"explicit_euler", "semi_implicit_euler", "rk4"}
    if method not in allowed:
        raise ValueError(f"Unsupported integration_method '{integration_method}'. Expected one of {sorted(allowed)}")
    return method


def _normalized_nmpc_integration_method(integration_method: str) -> str:
    method = str(integration_method or "rk4").strip().lower()
    allowed = {"explicit_euler", "rk4"}
    if method not in allowed:
        raise ValueError(f"Unsupported nmpc_integration_method '{integration_method}'. Expected one of {sorted(allowed)}")
    return method


def _normalized_nmpc_parallel_backend(parallel_backend: str) -> str:
    backend = str(parallel_backend or "serial").strip().lower()
    allowed = {"serial", "thread", "process"}
    if backend not in allowed:
        raise ValueError(f"Unsupported nmpc_parallel_backend '{parallel_backend}'. Expected one of {sorted(allowed)}")
    return backend


def _normalized_nmpc_solver_backend(solver_backend: str) -> str:
    backend = str(solver_backend or "auto").strip().lower()
    allowed = {"auto", "casadi", "acados"}
    if backend not in allowed:
        raise ValueError(f"Unsupported nmpc_solver_backend '{solver_backend}'. Expected one of {sorted(allowed)}")
    return backend


def _state_from_vector(vector: np.ndarray) -> State3DOF:
    return State3DOF(
        x=float(vector[0]),
        y=float(vector[1]),
        psi=wrap_angle(float(vector[2])),
        u=float(vector[3]),
        v=float(vector[4]),
        r=float(vector[5]),
    )


def _sample_obstacle(track: DynamicObstacleTrack, time: float) -> Optional[DynamicObstacleSample]:
    if not track.samples:
        return None
    if time <= track.samples[0].time:
        return track.samples[0]
    if time >= track.samples[-1].time:
        last = track.samples[-1]
        dt = time - last.time
        return DynamicObstacleSample(time=time, x=last.x + last.vx * dt, y=last.y + last.vy * dt, vx=last.vx, vy=last.vy)
    for first, second in zip(track.samples[:-1], track.samples[1:]):
        if first.time <= time <= second.time:
            alpha = (time - first.time) / max(second.time - first.time, 1e-9)
            return DynamicObstacleSample(
                time=time,
                x=first.x + alpha * (second.x - first.x),
                y=first.y + alpha * (second.y - first.y),
                vx=first.vx + alpha * (second.vx - first.vx),
                vy=first.vy + alpha * (second.vy - first.vy),
            )
    return track.samples[-1]
