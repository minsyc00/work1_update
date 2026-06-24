from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import connected_components, rotated_rectangle_mask, unit_heading, wrap_angle
from .nmpc import CasadiNMPCController
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

    def step(self, state: State3DOF, control: ControlInput, dt: float, mismatch: np.ndarray) -> State3DOF:
        u_dot = (control.thrust - self.damp_u * state.u) / self.mass_u + mismatch[0]
        v_dot = (-self.damp_v * state.v + self.cross_coupling * state.r) / self.mass_v + mismatch[1]
        r_dot = (control.yaw_moment - self.damp_r * state.r) / self.mass_r + mismatch[2]

        u = state.u + dt * u_dot
        v = state.v + dt * v_dot
        r = state.r + dt * r_dot
        psi = wrap_angle(state.psi + dt * r)
        x_dot = u * math.cos(psi) - v * math.sin(psi)
        y_dot = u * math.sin(psi) + v * math.cos(psi)
        x = state.x + dt * x_dot
        y = state.y + dt * y_dot
        return State3DOF(x=x, y=y, psi=psi, u=u, v=v, r=r)


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
        self.state = CoverageState(
            resolution=resolution,
            x_coords=x_coords,
            y_coords=y_coords,
            coverage_ratio=np.zeros((y_coords.size, x_coords.size), dtype=float),
            covered=np.zeros((y_coords.size, x_coords.size), dtype=bool),
        )

    def update(self, pose: Pose2D) -> CoverageState:
        mask = rotated_rectangle_mask(
            self.state.x_coords,
            self.state.y_coords,
            pose.x,
            pose.y,
            pose.psi,
            self.footprint_length,
            self.footprint_width,
        )
        self.state.coverage_ratio[mask] = 1.0
        self.state.covered = self.state.coverage_ratio >= self.eta_cov
        return self.state

    def detect_residuals(self, min_component_cells: int = 3) -> List[CoverageResidual]:
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
        return residuals


class SwarmRuntime:
    def __init__(self, config: PlannerConfig, planning_result: PlanningResult) -> None:
        self.config = config
        self.planning_result = planning_result
        self.model = USV3DOFModel()
        self.estimator = Parallel6DOFEstimator(delta_safe_max=config.safety.delta_safe_max)
        self.coverage = CoverageTracker(config)
        self.horizon_steps = max(10, int(round(config.mission.local_control_hz * 3.0)))
        self.dt = 1.0 / max(config.mission.local_control_hz, 1e-6)
        self.neighbor_radius = 4.0 * config.safety.d_safe
        self.max_neighbors = max(1, (config.fleet.num_agents or 1) - 1)
        self.max_obstacles = 4
        self.nmpc_by_agent = {
            agent_id: CasadiNMPCController(
                config=config,
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
            for agent_id in range(config.fleet.num_agents or 0)
        }

    def control_step(
        self,
        agent_state: AgentRuntimeState,
        shared_predictions: Optional[Dict[int, Sequence[TrajectorySample]]] = None,
        obstacle_tracks: Optional[Sequence[DynamicObstacleTrack]] = None,
    ) -> ControlStepResult:
        ref_window = self._reference_window(agent_state.agent_id, agent_state.time)
        predictions = self._normalize_predictions(shared_predictions, agent_state.time, exclude_agent=agent_state.agent_id)
        obstacles = list(obstacle_tracks or [])
        mismatch, delta_safe = self.estimator.estimate(agent_state)
        preferred_velocity = self._compute_rvo_velocity(agent_state, ref_window, predictions, obstacles)
        best = self._solve_true_nmpc(agent_state, ref_window, predictions, obstacles, mismatch, delta_safe, preferred_velocity)
        if not best.feasible:
            best = self._safe_hold(agent_state, ref_window, predictions, obstacles, mismatch, delta_safe)
            mode = "degraded_safe_hold"
        else:
            mode = "nominal"
        self.coverage.update(agent_state.state3.pose())
        if self.config.mission.residual_enable:
            self.coverage.detect_residuals()
        warnings: List[str] = []
        if best.min_margin < self.config.safety.d_safe:
            warnings.append("reduced safety margin")
        if self.coverage.state.residual_components:
            warnings.append("residual coverage present")
        return ControlStepResult(
            cmd=best.control,
            safety_status=SafetyStatus(mode=mode, min_margin=best.min_margin, warnings=warnings),
            local_ref=ref_window,
            predicted_samples=best.predicted_samples,
        )

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
            state = self.model.step(state, control, self.dt, mismatch)
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
