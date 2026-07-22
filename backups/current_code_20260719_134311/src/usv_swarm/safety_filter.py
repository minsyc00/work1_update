from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .geometry import wrap_angle
from .schema import ControlInput, DynamicObstacleTrack, PlannerConfig, State3DOF, TrajectorySample


@dataclass
class SafetyFilterResult:
    control: ControlInput
    feasible: bool
    filtered: bool
    active_constraints: List[str] = field(default_factory=list)
    min_predicted_margin: float = float("inf")
    slack_used: bool = False
    solve_time_ms: float = 0.0
    fallback_reason: str = ""


def filter_control_cbf_qp(
    state: State3DOF,
    nominal_control: ControlInput,
    predictions: Dict[int, Sequence[TrajectorySample]],
    obstacle_tracks: Sequence[DynamicObstacleTrack],
    config: PlannerConfig,
    *,
    current_time: float,
    dt: float,
    delta_safe: float = 0.0,
) -> SafetyFilterResult:
    """Project a nominal control through a lightweight CBF-style safety filter.

    The first implementation intentionally avoids a heavy QP dependency. It
    evaluates a deterministic candidate set around the nominal command and
    selects the closest command that satisfies one-step-ahead CBF margins.
    """

    started = time.perf_counter()
    mode = str(config.mission.safety_filter_mode or "hybrid_cbf_qp").strip().lower()
    nominal = _clip_control(nominal_control, config)
    if mode in {"off", "none", "disabled"}:
        return SafetyFilterResult(control=nominal, feasible=True, filtered=False, solve_time_ms=(time.perf_counter() - started) * 1000.0)

    current_margin, _ = _evaluate_candidate_margin(
        state,
        ControlInput.zero(),
        predictions,
        obstacle_tracks,
        config,
        current_time=current_time - dt,
        dt=0.0,
        delta_safe=delta_safe,
    )
    required_margin = -config.mission.safety_min_margin_epsilon
    if current_margin < required_margin:
        required_margin = current_margin + config.mission.cbf_alpha * max(dt, 0.0) * (-current_margin)

    candidates = _candidate_controls(nominal, config)
    best_feasible: tuple[float, ControlInput, float, List[str]] | None = None
    best_any: tuple[float, ControlInput, float, List[str]] | None = None
    nominal_active: List[str] = []
    for candidate in candidates:
        margin, active = _evaluate_candidate_margin(
            state,
            candidate,
            predictions,
            obstacle_tracks,
            config,
            current_time=current_time,
            dt=dt,
            delta_safe=delta_safe,
        )
        if abs(candidate.thrust - nominal.thrust) <= 1e-9 and abs(candidate.yaw_moment - nominal.yaw_moment) <= 1e-9:
            nominal_active = active
        control_distance = ((candidate.thrust - nominal.thrust) / max(config.fleet.max_thrust, 1e-6)) ** 2
        control_distance += ((candidate.yaw_moment - nominal.yaw_moment) / max(config.fleet.max_yaw_moment, 1e-6)) ** 2
        violation_penalty = 1.0e3 * max(-margin, 0.0) ** 2
        score = control_distance + violation_penalty
        if best_any is None or score < best_any[0]:
            best_any = (score, candidate, margin, active)
        if margin >= required_margin:
            if best_feasible is None or score < best_feasible[0]:
                best_feasible = (score, candidate, margin, active)

    selected = best_feasible or best_any
    if selected is None:
        return SafetyFilterResult(
            control=nominal,
            feasible=False,
            filtered=False,
            min_predicted_margin=-float("inf"),
            solve_time_ms=(time.perf_counter() - started) * 1000.0,
            fallback_reason="no_candidate_controls",
        )

    _, control, margin, active = selected
    filtered = abs(control.thrust - nominal.thrust) > 1e-9 or abs(control.yaw_moment - nominal.yaw_moment) > 1e-9
    reported_active = sorted(set(active + (nominal_active if filtered else [])))
    return SafetyFilterResult(
        control=control,
        feasible=best_feasible is not None,
        filtered=filtered,
        active_constraints=reported_active,
        min_predicted_margin=margin,
        slack_used=best_feasible is None and bool(config.mission.cbf_allow_slack),
        solve_time_ms=(time.perf_counter() - started) * 1000.0,
        fallback_reason="" if best_feasible is not None else "no_strictly_safe_candidate",
    )


def _candidate_controls(nominal: ControlInput, config: PlannerConfig) -> List[ControlInput]:
    thrust_limit = config.fleet.max_thrust
    yaw_limit = config.fleet.max_yaw_moment
    candidates: List[ControlInput] = [nominal]
    for thrust in (
        nominal.thrust,
        0.5 * nominal.thrust,
        0.0,
        -0.5 * thrust_limit,
        -thrust_limit,
    ):
        for yaw in (
            nominal.yaw_moment,
            -yaw_limit,
            -0.5 * yaw_limit,
            0.0,
            0.5 * yaw_limit,
            yaw_limit,
        ):
            candidates.append(_clip_control(ControlInput(float(thrust), float(yaw)), config))

    unique: List[ControlInput] = []
    seen: set[tuple[float, float]] = set()
    for candidate in candidates:
        key = (round(candidate.thrust, 9), round(candidate.yaw_moment, 9))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _evaluate_candidate_margin(
    state: State3DOF,
    control: ControlInput,
    predictions: Dict[int, Sequence[TrajectorySample]],
    obstacle_tracks: Sequence[DynamicObstacleTrack],
    config: PlannerConfig,
    *,
    current_time: float,
    dt: float,
    delta_safe: float,
) -> tuple[float, List[str]]:
    safe_distance = config.safety.d_safe + delta_safe
    horizon_steps = 2
    predicted = State3DOF(**vars(state))
    min_margin = float("inf")
    active: List[str] = []
    for step in range(1, horizon_steps + 1):
        predicted = _predict_state(predicted, control, dt)
        t = current_time + step * dt
        boundary_margin = _boundary_margin(predicted, config)
        if boundary_margin < min_margin:
            min_margin = boundary_margin
        if boundary_margin <= config.mission.safety_min_margin_epsilon:
            active.append("boundary")

        for agent_id, samples in predictions.items():
            if not samples:
                continue
            sample = samples[min(step - 1, len(samples) - 1)]
            margin = math.hypot(predicted.x - sample.x, predicted.y - sample.y) - safe_distance
            if margin < min_margin:
                min_margin = margin
            if margin <= config.mission.safety_min_margin_epsilon:
                active.append(f"agent:{agent_id}")

        for track in obstacle_tracks:
            sample = _sample_obstacle(track, t)
            if sample is None:
                continue
            margin = math.hypot(predicted.x - sample.x, predicted.y - sample.y) - safe_distance - track.radius
            if margin < min_margin:
                min_margin = margin
            if margin <= config.mission.safety_min_margin_epsilon:
                active.append(f"obstacle:{track.obstacle_id}")
    return min_margin, sorted(set(active))


def _predict_state(state: State3DOF, control: ControlInput, dt: float) -> State3DOF:
    mass_u = 1.0
    mass_v = 1.4
    mass_r = 1.0
    damp_u = 0.8
    damp_v = 1.1
    damp_r = 0.9
    cross_coupling = 0.3

    u_dot = (control.thrust - damp_u * state.u) / mass_u
    v_dot = (-damp_v * state.v + cross_coupling * state.r) / mass_v
    r_dot = (control.yaw_moment - damp_r * state.r) / mass_r
    u = state.u + dt * u_dot
    v = state.v + dt * v_dot
    r = state.r + dt * r_dot
    psi = wrap_angle(state.psi + dt * r)
    x = state.x + dt * (u * math.cos(psi) - v * math.sin(psi))
    y = state.y + dt * (u * math.sin(psi) + v * math.cos(psi))
    return State3DOF(x=x, y=y, psi=psi, u=u, v=v, r=r)


def _boundary_margin(state: State3DOF, config: PlannerConfig) -> float:
    return min(
        state.x - config.safety.boundary_margin_x,
        config.mission.area_length_x - state.x - config.safety.boundary_margin_x,
        state.y - config.safety.boundary_margin_y,
        config.mission.area_length_y - state.y - config.safety.boundary_margin_y,
    )


def _clip_control(control: ControlInput, config: PlannerConfig) -> ControlInput:
    return ControlInput(
        thrust=float(np.clip(control.thrust, -config.fleet.max_thrust, config.fleet.max_thrust)),
        yaw_moment=float(np.clip(control.yaw_moment, -config.fleet.max_yaw_moment, config.fleet.max_yaw_moment)),
    )


def _sample_obstacle(track: DynamicObstacleTrack, time_value: float):
    if not track.samples:
        return None
    best = min(track.samples, key=lambda sample: abs(sample.time - time_value))
    return best
