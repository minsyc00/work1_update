from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import casadi as ca
import numpy as np

from .schema import ControlInput, PlannerConfig, State3DOF, TrajectorySample


@dataclass
class NMPCResult:
    feasible: bool
    control: ControlInput
    predicted_states: np.ndarray
    predicted_controls: np.ndarray
    objective: float
    solver_status: str
    min_margin: float


class AcadosUnavailableError(RuntimeError):
    """Raised when the optional Acados backend cannot be constructed."""


class AcadosNMPCController:
    """Optional Acados NMPC backend.

    The project keeps Acados optional because it requires a native acados
    installation and generated solver artifacts. This class is intentionally
    lazy: importing the package does not require Acados, while selecting the
    backend gives a clear error if the runtime is unavailable.
    """

    solver_backend_requested = "acados"
    solver_backend_effective = "acados"
    acados_available = False
    acados_fallback_reason = ""

    def __init__(self, *args, **kwargs) -> None:
        try:
            import acados_template  # noqa: F401
        except BaseException as exc:
            self.acados_fallback_reason = f"acados_template unavailable: {type(exc).__name__}: {exc}"
            raise AcadosUnavailableError(self.acados_fallback_reason) from exc

        self.acados_available = True
        self.acados_fallback_reason = (
            "acados_template is importable, but generated Acados solver support "
            "is not configured in this source checkout"
        )
        raise AcadosUnavailableError(self.acados_fallback_reason)


class CasadiNMPCController:
    def __init__(
        self,
        config: PlannerConfig,
        horizon_steps: int,
        dt: float,
        max_neighbors: int,
        max_obstacles: int,
        mass_u: float,
        mass_v: float,
        mass_r: float,
        damp_u: float,
        damp_v: float,
        damp_r: float,
        cross_coupling: float,
    ) -> None:
        self.config = config
        self.horizon_steps = horizon_steps
        self.dt = dt
        self.max_neighbors = max(1, max_neighbors)
        self.max_obstacles = max(1, max_obstacles)
        self.mass_u = mass_u
        self.mass_v = mass_v
        self.mass_r = mass_r
        self.damp_u = damp_u
        self.damp_v = damp_v
        self.damp_r = damp_r
        self.cross_coupling = cross_coupling
        self.integration_method = _normalized_nmpc_integration_method(config.mission.nmpc_integration_method)
        self.solver_backend_requested = str(getattr(config.mission, "nmpc_solver_backend", "casadi") or "casadi").strip().lower()
        self.solver_backend_effective = "casadi"
        self.acados_available = False
        self.acados_fallback_reason = ""
        self._build_problem()
        self._last_x_guess: np.ndarray | None = None
        self._last_u_guess: np.ndarray | None = None

    def _build_problem(self) -> None:
        opti = ca.Opti()
        h = self.horizon_steps
        nx = 6
        nu = 2

        x = opti.variable(nx, h + 1)
        u = opti.variable(nu, h)
        s_nei = opti.variable(self.max_neighbors, h)
        s_obs = opti.variable(self.max_obstacles, h)
        s_bound = opti.variable(4, h)

        x0_p = opti.parameter(nx)
        prev_u_p = opti.parameter(nu)
        mismatch_p = opti.parameter(3)
        safe_distance_p = opti.parameter()

        ref_x_p = opti.parameter(h)
        ref_y_p = opti.parameter(h)
        ref_psi_p = opti.parameter(h)
        ref_u_p = opti.parameter(h)
        ref_r_p = opti.parameter(h)
        pref_vx_p = opti.parameter(h)
        pref_vy_p = opti.parameter(h)

        neigh_x_p = opti.parameter(self.max_neighbors, h + 1)
        neigh_y_p = opti.parameter(self.max_neighbors, h + 1)
        obs_x_p = opti.parameter(self.max_obstacles, h + 1)
        obs_y_p = opti.parameter(self.max_obstacles, h + 1)
        obs_r_p = opti.parameter(self.max_obstacles, h + 1)

        objective = 0
        gamma = 0.4
        gamma_b = 0.5
        du_thrust_max = 0.7 * self.config.fleet.max_thrust
        du_yaw_max = 0.7 * self.config.fleet.max_yaw_moment
        u_upper = 1.4 * self.config.fleet.cruise_speed
        u_lower = -0.25 * self.config.fleet.cruise_speed
        r_abs_max = max(self.config.fleet.turn_speed_max / max(self.config.fleet.min_turn_radius, 1e-6), 0.2)

        opti.subject_to(x[:, 0] == x0_p)
        opti.subject_to(ca.vec(s_nei) >= 0)
        opti.subject_to(ca.vec(s_obs) >= 0)
        opti.subject_to(ca.vec(s_bound) >= 0)

        for k in range(h):
            xk = x[:, k]
            uk = u[:, k]
            xkp = x[:, k + 1]
            thrust = uk[0]
            yaw = uk[1]

            opti.subject_to(xkp == self._symbolic_next_state(xk, uk, mismatch_p))

            opti.subject_to(opti.bounded(-self.config.fleet.max_thrust, thrust, self.config.fleet.max_thrust))
            opti.subject_to(opti.bounded(-self.config.fleet.max_yaw_moment, yaw, self.config.fleet.max_yaw_moment))
            opti.subject_to(opti.bounded(u_lower, xkp[3], u_upper))
            opti.subject_to(opti.bounded(-u_upper, xkp[4], u_upper))
            opti.subject_to(opti.bounded(-r_abs_max, xkp[5], r_abs_max))
            opti.subject_to(opti.bounded(0.0, xkp[0], self.config.mission.area_length_x))
            opti.subject_to(opti.bounded(0.0, xkp[1], self.config.mission.area_length_y))

            prev = prev_u_p if k == 0 else u[:, k - 1]
            opti.subject_to(opti.bounded(-du_thrust_max, thrust - prev[0], du_thrust_max))
            opti.subject_to(opti.bounded(-du_yaw_max, yaw - prev[1], du_yaw_max))

            vx = xk[3] * ca.cos(xk[2]) - xk[4] * ca.sin(xk[2])
            vy = xk[3] * ca.sin(xk[2]) + xk[4] * ca.cos(xk[2])
            objective += self.config.weights.w_pos * ((xk[0] - ref_x_p[k]) ** 2 + (xk[1] - ref_y_p[k]) ** 2)
            objective += self.config.weights.w_psi * (1.0 - ca.cos(xk[2] - ref_psi_p[k]))
            objective += 0.5 * self.config.weights.w_vel * ((xk[3] - ref_u_p[k]) ** 2 + (xk[5] - ref_r_p[k]) ** 2)
            objective += self.config.weights.w_vel * ((vx - pref_vx_p[k]) ** 2 + (vy - pref_vy_p[k]) ** 2)
            objective += self.config.weights.w_u * (thrust**2 + yaw**2)
            objective += self.config.weights.w_du * ((thrust - prev[0]) ** 2 + (yaw - prev[1]) ** 2)

            for j in range(self.max_neighbors):
                h_cur = (xk[0] - neigh_x_p[j, k]) ** 2 + (xk[1] - neigh_y_p[j, k]) ** 2 - safe_distance_p**2
                h_next = (xkp[0] - neigh_x_p[j, k + 1]) ** 2 + (xkp[1] - neigh_y_p[j, k + 1]) ** 2 - safe_distance_p**2
                if self.config.mission.cbf_allow_slack:
                    opti.subject_to(h_cur + s_nei[j, k] >= 0)
                    opti.subject_to(h_next - (1.0 - gamma) * h_cur + s_nei[j, k] >= 0)
                    objective += self.config.weights.w_soft * s_nei[j, k] ** 2
                else:
                    opti.subject_to(h_next - (1.0 - gamma) * h_cur >= 0)

            for j in range(self.max_obstacles):
                safe_radius_cur = safe_distance_p + obs_r_p[j, k]
                safe_radius_next = safe_distance_p + obs_r_p[j, k + 1]
                h_cur = (xk[0] - obs_x_p[j, k]) ** 2 + (xk[1] - obs_y_p[j, k]) ** 2 - safe_radius_cur**2
                h_next = (xkp[0] - obs_x_p[j, k + 1]) ** 2 + (xkp[1] - obs_y_p[j, k + 1]) ** 2 - safe_radius_next**2
                if self.config.mission.cbf_allow_slack:
                    opti.subject_to(h_cur + s_obs[j, k] >= 0)
                    opti.subject_to(h_next - (1.0 - gamma) * h_cur + s_obs[j, k] >= 0)
                    objective += self.config.weights.w_soft * s_obs[j, k] ** 2
                else:
                    opti.subject_to(h_next - (1.0 - gamma) * h_cur >= 0)

            boundary_pairs = (
                (xk[0] - self.config.safety.boundary_margin_x, xkp[0] - self.config.safety.boundary_margin_x, s_bound[0, k]),
                (
                    self.config.mission.area_length_x - self.config.safety.boundary_margin_x - xk[0],
                    self.config.mission.area_length_x - self.config.safety.boundary_margin_x - xkp[0],
                    s_bound[1, k],
                ),
                (xk[1] - self.config.safety.boundary_margin_y, xkp[1] - self.config.safety.boundary_margin_y, s_bound[2, k]),
                (
                    self.config.mission.area_length_y - self.config.safety.boundary_margin_y - xk[1],
                    self.config.mission.area_length_y - self.config.safety.boundary_margin_y - xkp[1],
                    s_bound[3, k],
                ),
            )
            for h_cur, h_next, slack in boundary_pairs:
                if self.config.mission.cbf_allow_slack:
                    opti.subject_to(h_cur + slack >= 0)
                    opti.subject_to(h_next - (1.0 - gamma_b) * h_cur + slack >= 0)
                    objective += 0.5 * self.config.weights.w_soft * slack**2
                else:
                    opti.subject_to(h_next - (1.0 - gamma_b) * h_cur >= 0)

        objective += self.config.weights.w_pos * ((x[0, h] - ref_x_p[h - 1]) ** 2 + (x[1, h] - ref_y_p[h - 1]) ** 2)
        objective += self.config.weights.w_psi * (1.0 - ca.cos(x[2, h] - ref_psi_p[h - 1]))

        opti.minimize(objective)
        solver_name = "ipopt"
        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "print_level": 0,
            "sb": "yes",
            "max_iter": 60,
            "tol": 1e-3,
            "acceptable_tol": 3e-3,
            "warm_start_init_point": "yes",
        }
        try:
            opti.solver(solver_name, p_opts, s_opts)
        except RuntimeError:
            solver_name = "sqpmethod"
            opti.solver(
                solver_name,
                {"expand": True},
                {"print_header": False, "print_iteration": False, "print_status": False, "qpsol": "qrqp"},
            )

        self.opti = opti
        self.solver_name = solver_name
        self.x = x
        self.u = u
        self.s_nei = s_nei
        self.s_obs = s_obs
        self.s_bound = s_bound
        self.x0_p = x0_p
        self.prev_u_p = prev_u_p
        self.mismatch_p = mismatch_p
        self.safe_distance_p = safe_distance_p
        self.ref_x_p = ref_x_p
        self.ref_y_p = ref_y_p
        self.ref_psi_p = ref_psi_p
        self.ref_u_p = ref_u_p
        self.ref_r_p = ref_r_p
        self.pref_vx_p = pref_vx_p
        self.pref_vy_p = pref_vy_p
        self.neigh_x_p = neigh_x_p
        self.neigh_y_p = neigh_y_p
        self.obs_x_p = obs_x_p
        self.obs_y_p = obs_y_p
        self.obs_r_p = obs_r_p

    def _symbolic_dynamics(self, xk, uk, mismatch_p):
        thrust = uk[0]
        yaw = uk[1]
        u_dot = (thrust - self.damp_u * xk[3]) / self.mass_u + mismatch_p[0]
        v_dot = (-self.damp_v * xk[4] + self.cross_coupling * xk[5]) / self.mass_v + mismatch_p[1]
        r_dot = (yaw - self.damp_r * xk[5]) / self.mass_r + mismatch_p[2]
        x_dot = xk[3] * ca.cos(xk[2]) - xk[4] * ca.sin(xk[2])
        y_dot = xk[3] * ca.sin(xk[2]) + xk[4] * ca.cos(xk[2])
        psi_dot = xk[5]
        return ca.vertcat(x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot)

    def _symbolic_next_state(self, xk, uk, mismatch_p):
        if self.integration_method == "explicit_euler":
            return xk + self.dt * self._symbolic_dynamics(xk, uk, mismatch_p)
        k1 = self._symbolic_dynamics(xk, uk, mismatch_p)
        k2 = self._symbolic_dynamics(xk + 0.5 * self.dt * k1, uk, mismatch_p)
        k3 = self._symbolic_dynamics(xk + 0.5 * self.dt * k2, uk, mismatch_p)
        k4 = self._symbolic_dynamics(xk + self.dt * k3, uk, mismatch_p)
        return xk + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def solve(
        self,
        state: State3DOF,
        previous_control: ControlInput,
        ref_window: Sequence[TrajectorySample],
        preferred_velocities: np.ndarray,
        neighbor_predictions: Sequence[Sequence[TrajectorySample]],
        obstacle_predictions: Sequence[Sequence[Tuple[float, float, float]]],
        mismatch: np.ndarray,
        safe_distance: float,
    ) -> NMPCResult:
        h = self.horizon_steps
        refs = _pad_reference_window(ref_window, h)
        pref = _pad_preferred_velocities(preferred_velocities, h)
        neigh_x, neigh_y = _pack_neighbor_predictions(neighbor_predictions, self.max_neighbors, h)
        obs_x, obs_y, obs_r = _pack_obstacle_predictions(obstacle_predictions, self.max_obstacles, h)

        self.opti.set_value(self.x0_p, np.array([state.x, state.y, state.psi, state.u, state.v, state.r], dtype=float))
        self.opti.set_value(self.prev_u_p, np.array([previous_control.thrust, previous_control.yaw_moment], dtype=float))
        self.opti.set_value(self.mismatch_p, mismatch.astype(float))
        self.opti.set_value(self.safe_distance_p, float(safe_distance))
        self.opti.set_value(self.ref_x_p, np.array([sample.x for sample in refs], dtype=float))
        self.opti.set_value(self.ref_y_p, np.array([sample.y for sample in refs], dtype=float))
        self.opti.set_value(self.ref_psi_p, np.array([sample.psi for sample in refs], dtype=float))
        self.opti.set_value(self.ref_u_p, np.array([sample.u_ref for sample in refs], dtype=float))
        self.opti.set_value(self.ref_r_p, np.array([sample.r_ref for sample in refs], dtype=float))
        self.opti.set_value(self.pref_vx_p, pref[:, 0])
        self.opti.set_value(self.pref_vy_p, pref[:, 1])
        self.opti.set_value(self.neigh_x_p, neigh_x)
        self.opti.set_value(self.neigh_y_p, neigh_y)
        self.opti.set_value(self.obs_x_p, obs_x)
        self.opti.set_value(self.obs_y_p, obs_y)
        self.opti.set_value(self.obs_r_p, obs_r)

        x_guess = self._build_x_guess(state, refs)
        u_guess = self._build_u_guess(previous_control, refs)
        self.opti.set_initial(self.x, x_guess)
        self.opti.set_initial(self.u, u_guess)
        self.opti.set_initial(self.s_nei, 0.0)
        self.opti.set_initial(self.s_obs, 0.0)
        self.opti.set_initial(self.s_bound, 0.0)

        try:
            solution = self.opti.solve()
            x_value = np.array(solution.value(self.x), dtype=float)
            u_value = np.array(solution.value(self.u), dtype=float)
            objective = float(solution.value(self.opti.f))
            status = str(self.opti.stats().get("return_status", "success"))
            self._last_x_guess = x_value
            self._last_u_guess = u_value
            min_margin = _estimate_min_margin(x_value, neigh_x, neigh_y, obs_x, obs_y, obs_r, safe_distance, self.config)
            return NMPCResult(
                feasible=True,
                control=ControlInput(float(u_value[0, 0]), float(u_value[1, 0])),
                predicted_states=x_value,
                predicted_controls=u_value,
                objective=objective,
                solver_status=status,
                min_margin=min_margin,
            )
        except RuntimeError as exc:
            self.opti.set_initial(self.s_nei, 1.0)
            self.opti.set_initial(self.s_obs, 1.0)
            self.opti.set_initial(self.s_bound, 1.0)
            return NMPCResult(
                feasible=False,
                control=ControlInput.zero(),
                predicted_states=np.zeros((6, h + 1), dtype=float),
                predicted_controls=np.zeros((2, h), dtype=float),
                objective=float("inf"),
                solver_status=str(exc),
                min_margin=float("-inf"),
            )

    def _build_x_guess(self, state: State3DOF, refs: Sequence[TrajectorySample]) -> np.ndarray:
        if self._last_x_guess is not None and self._last_x_guess.shape == (6, self.horizon_steps + 1):
            guess = np.concatenate([self._last_x_guess[:, 1:], self._last_x_guess[:, -1:]], axis=1)
            guess[:, 0] = np.array([state.x, state.y, state.psi, state.u, state.v, state.r], dtype=float)
            return guess
        guess = np.zeros((6, self.horizon_steps + 1), dtype=float)
        guess[:, 0] = np.array([state.x, state.y, state.psi, state.u, state.v, state.r], dtype=float)
        for idx in range(1, self.horizon_steps + 1):
            ref = refs[min(idx - 1, len(refs) - 1)]
            guess[:, idx] = np.array([ref.x, ref.y, ref.psi, ref.u_ref, 0.0, ref.r_ref], dtype=float)
        return guess

    def _build_u_guess(self, previous_control: ControlInput, refs: Sequence[TrajectorySample]) -> np.ndarray:
        if self._last_u_guess is not None and self._last_u_guess.shape == (2, self.horizon_steps):
            guess = np.concatenate([self._last_u_guess[:, 1:], self._last_u_guess[:, -1:]], axis=1)
            return guess
        guess = np.zeros((2, self.horizon_steps), dtype=float)
        guess[0, :] = np.clip([sample.u_ref for sample in refs], -self.config.fleet.max_thrust, self.config.fleet.max_thrust)
        guess[1, :] = np.clip([sample.r_ref for sample in refs], -self.config.fleet.max_yaw_moment, self.config.fleet.max_yaw_moment)
        guess[:, 0] = np.array([previous_control.thrust, previous_control.yaw_moment], dtype=float)
        return guess


def create_nmpc_controller(
    *,
    config: PlannerConfig,
    horizon_steps: int,
    dt: float,
    max_neighbors: int,
    max_obstacles: int,
    mass_u: float,
    mass_v: float,
    mass_r: float,
    damp_u: float,
    damp_v: float,
    damp_r: float,
    cross_coupling: float,
):
    requested = _normalized_nmpc_solver_backend(getattr(config.mission, "nmpc_solver_backend", "auto"))
    kwargs = dict(
        config=config,
        horizon_steps=horizon_steps,
        dt=dt,
        max_neighbors=max_neighbors,
        max_obstacles=max_obstacles,
        mass_u=mass_u,
        mass_v=mass_v,
        mass_r=mass_r,
        damp_u=damp_u,
        damp_v=damp_v,
        damp_r=damp_r,
        cross_coupling=cross_coupling,
    )
    if requested in {"auto", "acados"}:
        try:
            controller = AcadosNMPCController(**kwargs)
            controller.solver_backend_requested = requested
            controller.solver_backend_effective = "acados"
            controller.acados_available = True
            controller.acados_fallback_reason = ""
            return controller
        except AcadosUnavailableError as exc:
            if requested == "acados":
                raise
            fallback = CasadiNMPCController(**kwargs)
            fallback.solver_backend_requested = "auto"
            fallback.solver_backend_effective = "casadi"
            fallback.acados_available = False
            fallback.acados_fallback_reason = str(exc)
            return fallback
    controller = CasadiNMPCController(**kwargs)
    controller.solver_backend_requested = "casadi"
    controller.solver_backend_effective = "casadi"
    controller.acados_available = False
    controller.acados_fallback_reason = ""
    return controller


def _pad_reference_window(ref_window: Sequence[TrajectorySample], horizon_steps: int) -> List[TrajectorySample]:
    refs = list(ref_window)
    if not refs:
        refs = [TrajectorySample(time=0.0, x=0.0, y=0.0, psi=0.0, u_ref=0.0, r_ref=0.0, segment_type="idle")]
    while len(refs) < horizon_steps:
        refs.append(refs[-1])
    return refs[:horizon_steps]


def _pad_preferred_velocities(preferred_velocities: np.ndarray, horizon_steps: int) -> np.ndarray:
    preferred = np.asarray(preferred_velocities, dtype=float)
    if preferred.ndim == 1:
        preferred = np.repeat(preferred.reshape(1, 2), horizon_steps, axis=0)
    if preferred.shape[0] < horizon_steps:
        pad = np.repeat(preferred[-1:, :], horizon_steps - preferred.shape[0], axis=0)
        preferred = np.vstack([preferred, pad])
    return preferred[:horizon_steps]


def _pack_neighbor_predictions(
    neighbor_predictions: Sequence[Sequence[TrajectorySample]],
    max_neighbors: int,
    horizon_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    far_value = 1e3
    neigh_x = np.full((max_neighbors, horizon_steps + 1), far_value, dtype=float)
    neigh_y = np.full((max_neighbors, horizon_steps + 1), far_value, dtype=float)
    for idx, prediction in enumerate(neighbor_predictions[:max_neighbors]):
        samples = list(prediction)
        if not samples:
            continue
        while len(samples) < horizon_steps + 1:
            samples.append(samples[-1])
        for step in range(horizon_steps + 1):
            neigh_x[idx, step] = samples[step].x
            neigh_y[idx, step] = samples[step].y
    return neigh_x, neigh_y


def _pack_obstacle_predictions(
    obstacle_predictions: Sequence[Sequence[Tuple[float, float, float]]],
    max_obstacles: int,
    horizon_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    far_value = 1e3
    obs_x = np.full((max_obstacles, horizon_steps + 1), far_value, dtype=float)
    obs_y = np.full((max_obstacles, horizon_steps + 1), far_value, dtype=float)
    obs_r = np.zeros((max_obstacles, horizon_steps + 1), dtype=float)
    for idx, prediction in enumerate(obstacle_predictions[:max_obstacles]):
        samples = list(prediction)
        if not samples:
            continue
        while len(samples) < horizon_steps + 1:
            samples.append(samples[-1])
        for step in range(horizon_steps + 1):
            obs_x[idx, step], obs_y[idx, step], obs_r[idx, step] = samples[step]
    return obs_x, obs_y, obs_r


def _normalized_nmpc_integration_method(integration_method: str) -> str:
    method = str(integration_method or "rk4").strip().lower()
    allowed = {"explicit_euler", "rk4"}
    if method not in allowed:
        raise ValueError(f"Unsupported nmpc_integration_method '{integration_method}'. Expected one of {sorted(allowed)}")
    return method


def _normalized_nmpc_solver_backend(solver_backend: str) -> str:
    backend = str(solver_backend or "auto").strip().lower()
    allowed = {"auto", "casadi", "acados"}
    if backend not in allowed:
        raise ValueError(f"Unsupported nmpc_solver_backend '{solver_backend}'. Expected one of {sorted(allowed)}")
    return backend


def _estimate_min_margin(
    predicted_states: np.ndarray,
    neigh_x: np.ndarray,
    neigh_y: np.ndarray,
    obs_x: np.ndarray,
    obs_y: np.ndarray,
    obs_r: np.ndarray,
    safe_distance: float,
    config: PlannerConfig,
) -> float:
    min_margin = float("inf")
    for step in range(predicted_states.shape[1]):
        x = predicted_states[0, step]
        y = predicted_states[1, step]
        min_margin = min(
            min_margin,
            x - config.safety.boundary_margin_x,
            config.mission.area_length_x - x - config.safety.boundary_margin_x,
            y - config.safety.boundary_margin_y,
            config.mission.area_length_y - y - config.safety.boundary_margin_y,
        )
        for idx in range(neigh_x.shape[0]):
            margin = math.hypot(x - neigh_x[idx, step], y - neigh_y[idx, step]) - safe_distance
            min_margin = min(min_margin, margin)
        for idx in range(obs_x.shape[0]):
            margin = math.hypot(x - obs_x[idx, step], y - obs_y[idx, step]) - safe_distance - obs_r[idx, step]
            min_margin = min(min_margin, margin)
    return float(min_margin)
