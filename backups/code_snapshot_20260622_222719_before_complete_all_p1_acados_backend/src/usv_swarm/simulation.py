from __future__ import annotations

import copy
import math
import pathlib
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from .control import SwarmRuntime, _sample_obstacle
from .geometry import wrap_angle
from .planning import plan_global_coverage
from .schema import (
    AgentRuntimeState,
    ControlInput,
    DynamicObstacleSample,
    DynamicObstacleTrack,
    PlannerConfig,
    PlanningResult,
    SafetyStatus,
    State3DOF,
    State6DOF,
)


@dataclass
class SimulationFrame:
    time: float
    agent_states: Dict[int, State3DOF]
    agent_controls: Dict[int, ControlInput]
    safety_status: Dict[int, SafetyStatus]
    obstacle_positions: Dict[str, Tuple[float, float, float]]
    coverage_snapshot: np.ndarray
    coverage_fraction: float


@dataclass
class SimulationLog:
    config: PlannerConfig
    plan: PlanningResult
    obstacle_tracks: List[DynamicObstacleTrack]
    frames: List[SimulationFrame] = field(default_factory=list)
    runtime_profile: Dict[str, object] = field(default_factory=dict)

    def trajectory(self, agent_id: int) -> List[Tuple[float, float]]:
        return [(frame.agent_states[agent_id].x, frame.agent_states[agent_id].y) for frame in self.frames if agent_id in frame.agent_states]

    @property
    def final_coverage_fraction(self) -> float:
        return self.frames[-1].coverage_fraction if self.frames else 0.0


def build_crossing_obstacle_scenario(config: PlannerConfig, total_time: float, dt: float) -> List[DynamicObstacleTrack]:
    lx = config.mission.area_length_x
    ly = config.mission.area_length_y
    return [
        _linear_obstacle_track(
            obstacle_id="obs_a",
            start=(0.25 * lx, 0.2 * ly),
            end=(0.78 * lx, 0.82 * ly),
            radius=1.2,
            total_time=total_time,
            dt=dt,
        ),
        _linear_obstacle_track(
            obstacle_id="obs_b",
            start=(0.82 * lx, 0.7 * ly),
            end=(0.18 * lx, 0.35 * ly),
            radius=1.0,
            total_time=total_time,
            dt=dt,
        ),
    ]


def simulate_swarm_closed_loop(
    config: PlannerConfig,
    planning_result: Optional[PlanningResult] = None,
    obstacle_tracks: Optional[Sequence[DynamicObstacleTrack]] = None,
    total_time: Optional[float] = None,
) -> SimulationLog:
    plan = planning_result or plan_global_coverage(config)
    runtime = SwarmRuntime(config, plan)
    dt = runtime.dt
    obstacle_tracks = list(obstacle_tracks or [])
    if total_time is None:
        total_time = max(ref.horizon_time for ref in plan.refs.values()) if plan.refs else 30.0
        total_time = min(max(total_time, 20.0), 60.0)

    states3 = {idx: copy.deepcopy(state) for idx, state in enumerate(config.fleet.initial_states_3dof)}
    states6 = {
        idx: copy.deepcopy(config.fleet.initial_states_6dof[idx]) if config.fleet.initial_states_6dof else State6DOF(x=state.x, y=state.y, psi=state.psi)
        for idx, state in states3.items()
    }
    previous_controls = {idx: ControlInput.zero() for idx in states3}
    predictions_cache: Dict[int, Sequence] = {}
    log = SimulationLog(config=config, plan=plan, obstacle_tracks=list(obstacle_tracks))
    log.runtime_profile = {
        "control_mode": runtime.control_mode,
        "dynamics_integration_method": runtime.integration_method,
        "nmpc_integration_method": runtime.nmpc_integration_method,
        "plant_nmpc_integration_mismatch": runtime.integration_method != runtime.nmpc_integration_method,
        "nmpc_parallel_backend": runtime.nmpc_parallel_backend,
        "nmpc_parallel_backend_effective": runtime.nmpc_parallel_backend_effective,
        "horizon_steps": runtime.horizon_steps,
        "dt": dt,
        "steps": [],
    }

    runtime.coverage.state.covered[:] = False
    runtime.coverage.state.coverage_ratio[:] = 0.0
    for state in states3.values():
        runtime.coverage.update(state.pose())
    if config.mission.residual_enable:
        runtime.coverage.detect_residuals()
    log.frames.append(_build_frame(0.0, states3, previous_controls, {idx: SafetyStatus("init", float("inf")) for idx in states3}, obstacle_tracks, runtime))

    steps = int(math.ceil(total_time / dt))
    residual_interval_steps = max(1, int(config.mission.coverage_residual_interval_steps))
    for step in range(steps):
        step_wall_started = time.perf_counter()
        current_time = step * dt
        current_predictions = dict(predictions_cache)
        step_results = {}
        step_safety: Dict[int, SafetyStatus] = {}
        nmpc_calls_before = runtime.profiler.nmpc_called_count
        fallback_before = runtime.profiler.fallback_count
        timeout_before = runtime.profiler.timeout_count
        coverage_time_before = runtime.coverage.total_update_time_ms
        coverage_cells_before = runtime.coverage.updated_cell_count
        residual_count_before = runtime.coverage.residual_detection_count
        residual_time_before = runtime.coverage.total_residual_detection_time_ms
        if runtime.nmpc_parallel_backend in {"thread", "process"}:
            runtime_states = {
                agent_id: AgentRuntimeState(
                    agent_id=agent_id,
                    time=current_time,
                    state3=states3[agent_id],
                    state6=states6.get(agent_id),
                    previous_control=previous_controls[agent_id],
                )
                for agent_id in sorted(states3)
            }
            step_results = runtime.control_steps(runtime_states, current_predictions if current_predictions else None, obstacle_tracks)
            for agent_id, result in step_results.items():
                step_safety[agent_id] = result.safety_status
                current_predictions[agent_id] = result.predicted_samples
        else:
            for agent_id in sorted(states3):
                runtime_state = AgentRuntimeState(
                    agent_id=agent_id,
                    time=current_time,
                    state3=states3[agent_id],
                    state6=states6.get(agent_id),
                    previous_control=previous_controls[agent_id],
                )
                result = runtime.control_step(runtime_state, current_predictions if current_predictions else None, obstacle_tracks)
                step_results[agent_id] = result
                step_safety[agent_id] = result.safety_status
                current_predictions[agent_id] = result.predicted_samples
        next_states3: Dict[int, State3DOF] = {}
        next_states6: Dict[int, State6DOF] = {}
        for agent_id in sorted(states3):
            runtime_state = AgentRuntimeState(
                agent_id=agent_id,
                time=current_time,
                state3=states3[agent_id],
                state6=states6.get(agent_id),
                previous_control=previous_controls[agent_id],
            )
            mismatch, _ = runtime.estimator.estimate(runtime_state)
            next_state3 = runtime.model.step(
                states3[agent_id],
                step_results[agent_id].cmd,
                dt,
                mismatch,
                integration_method=runtime.integration_method,
            )
            next_state3.psi = wrap_angle(next_state3.psi)
            next_state6 = _propagate_state6_surrogate(states6.get(agent_id), next_state3, current_time + dt, agent_id)
            next_states3[agent_id] = next_state3
            next_states6[agent_id] = next_state6
            previous_controls[agent_id] = step_results[agent_id].cmd
            runtime.coverage.update(next_state3.pose())

        if config.mission.residual_enable and ((step + 1) % residual_interval_steps == 0):
            runtime.coverage.detect_residuals()

        states3 = next_states3
        states6 = next_states6
        predictions_cache = {agent_id: step_results[agent_id].predicted_samples for agent_id in step_results}
        frame_time = current_time + dt
        log.frames.append(_build_frame(frame_time, states3, previous_controls, step_safety, obstacle_tracks, runtime))
        step_wall_time = time.perf_counter() - step_wall_started
        log.runtime_profile["steps"].append(
            {
                "step": step,
                "time": current_time,
                "wall_time_ms": step_wall_time * 1000.0,
                "real_time_factor": dt / max(step_wall_time, 1e-9),
                "nmpc_called_count": runtime.profiler.nmpc_called_count - nmpc_calls_before,
                "fallback_count": runtime.profiler.fallback_count - fallback_before,
                "timeout_count": runtime.profiler.timeout_count - timeout_before,
                "coverage_update_time_ms": runtime.coverage.total_update_time_ms - coverage_time_before,
                "coverage_updated_cell_count": runtime.coverage.updated_cell_count - coverage_cells_before,
                "residual_detection_count": runtime.coverage.residual_detection_count - residual_count_before,
                "residual_detection_time_ms": runtime.coverage.total_residual_detection_time_ms - residual_time_before,
            }
        )

        if runtime.coverage.state.coverage_fraction >= 0.995 and all(frame_time >= plan.refs[agent].horizon_time for agent in plan.refs):
            break
    step_profiles = list(log.runtime_profile.get("steps", []))
    avg_step_ms = float(np.mean([item["wall_time_ms"] for item in step_profiles])) if step_profiles else 0.0
    avg_rtf = float(np.mean([item["real_time_factor"] for item in step_profiles])) if step_profiles else 0.0
    log.runtime_profile["summary"] = {
        **runtime.profiler.summary(),
        **runtime.coverage.profiler_summary(),
        "dynamics_integration_method": runtime.integration_method,
        "nmpc_integration_method": runtime.nmpc_integration_method,
        "plant_nmpc_integration_mismatch": runtime.integration_method != runtime.nmpc_integration_method,
        "nmpc_parallel_backend": runtime.nmpc_parallel_backend,
        "nmpc_parallel_backend_effective": runtime.nmpc_parallel_backend_effective,
        "avg_step_wall_time_ms": avg_step_ms,
        "avg_real_time_factor": avg_rtf,
        "step_count": len(step_profiles),
    }
    runtime.close()
    return log


def render_simulation_animation(log: SimulationLog, output_path: str | pathlib.Path, fps: int = 8) -> pathlib.Path:
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = log.config
    footprint = config.footprint

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, config.mission.area_length_x)
    ax.set_ylim(0, config.mission.area_length_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Multi-USV Closed-Loop Coverage Simulation")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    first_frame = log.frames[0]
    coverage_im = ax.imshow(
        first_frame.coverage_snapshot.astype(float),
        origin="lower",
        extent=[0, config.mission.area_length_x, 0, config.mission.area_length_y],
        cmap="Blues",
        alpha=0.28,
        vmin=0.0,
        vmax=1.0,
    )

    for strip in log.plan.strips:
        ax.plot(
            [strip.start_pose.x, strip.end_pose.x],
            [strip.start_pose.y, strip.end_pose.y],
            linestyle=":",
            linewidth=0.7,
            color="#b0b0b0",
            alpha=0.9,
        )

    colors = ["#0b5fff", "#ff6b00", "#00a676", "#c13bff", "#d7263d", "#2e4057"]
    agent_lines = {}
    agent_markers = {}
    agent_footprints = {}
    for agent_id, ref in log.plan.refs.items():
        color = colors[agent_id % len(colors)]
        ax.plot([sample.x for sample in ref.samples], [sample.y for sample in ref.samples], linestyle="--", linewidth=1.0, color=color, alpha=0.25)
        (line,) = ax.plot([], [], color=color, linewidth=2.2, label=f"USV {agent_id}")
        (marker,) = ax.plot([], [], marker="o", color=color, markersize=6)
        polygon = patches.Polygon([[0, 0]], closed=True, fill=False, edgecolor=color, linewidth=1.5)
        ax.add_patch(polygon)
        agent_lines[agent_id] = line
        agent_markers[agent_id] = marker
        agent_footprints[agent_id] = polygon

    obstacle_patches: Dict[str, patches.Circle] = {}
    for obstacle_id, (x, y, radius) in first_frame.obstacle_positions.items():
        circle = patches.Circle((x, y), radius=radius, fill=False, edgecolor="#e63946", linewidth=2.0, linestyle="-")
        ax.add_patch(circle)
        obstacle_patches[obstacle_id] = circle

    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
    )
    ax.legend(loc="upper right")

    def update(frame_index: int):
        frame = log.frames[frame_index]
        coverage_im.set_data(frame.coverage_snapshot.astype(float))
        for agent_id, line in agent_lines.items():
            trajectory = [
                (past.agent_states[agent_id].x, past.agent_states[agent_id].y)
                for past in log.frames[: frame_index + 1]
                if agent_id in past.agent_states
            ]
            xs = [item[0] for item in trajectory]
            ys = [item[1] for item in trajectory]
            line.set_data(xs, ys)
            state = frame.agent_states[agent_id]
            agent_markers[agent_id].set_data([state.x], [state.y])
            agent_footprints[agent_id].set_xy(_footprint_polygon(state, footprint.length_lf, footprint.width_wf))

        for obstacle_id, circle in obstacle_patches.items():
            if obstacle_id in frame.obstacle_positions:
                x, y, radius = frame.obstacle_positions[obstacle_id]
                circle.center = (x, y)
                circle.radius = radius

        warning_count = sum(len(status.warnings) for status in frame.safety_status.values())
        status_text.set_text(
            f"t = {frame.time:5.1f}s\n"
            f"coverage = {frame.coverage_fraction*100:5.1f}%\n"
            f"warnings = {warning_count}"
        )
        artists = [coverage_im, status_text, *agent_lines.values(), *agent_markers.values(), *agent_footprints.values(), *obstacle_patches.values()]
        return artists

    anim = animation.FuncAnimation(fig, update, frames=len(log.frames), interval=max(30, int(1000 / max(fps, 1))), blit=False)
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    return output_path


def _build_frame(
    time: float,
    states3: Dict[int, State3DOF],
    controls: Dict[int, ControlInput],
    safety: Dict[int, SafetyStatus],
    obstacle_tracks: Sequence[DynamicObstacleTrack],
    runtime: SwarmRuntime,
) -> SimulationFrame:
    obstacle_positions: Dict[str, Tuple[float, float, float]] = {}
    for track in obstacle_tracks:
        sample = _sample_obstacle(track, time)
        if sample is not None:
            obstacle_positions[track.obstacle_id] = (sample.x, sample.y, track.radius)
    return SimulationFrame(
        time=time,
        agent_states={agent_id: copy.deepcopy(state) for agent_id, state in states3.items()},
        agent_controls={agent_id: copy.deepcopy(cmd) for agent_id, cmd in controls.items()},
        safety_status={agent_id: copy.deepcopy(status) for agent_id, status in safety.items()},
        obstacle_positions=obstacle_positions,
        coverage_snapshot=runtime.coverage.state.covered.copy(),
        coverage_fraction=runtime.coverage.state.coverage_fraction,
    )


def _linear_obstacle_track(
    obstacle_id: str,
    start: Tuple[float, float],
    end: Tuple[float, float],
    radius: float,
    total_time: float,
    dt: float,
) -> DynamicObstacleTrack:
    steps = max(2, int(math.ceil(total_time / dt)) + 1)
    vx = (end[0] - start[0]) / max(total_time, 1e-9)
    vy = (end[1] - start[1]) / max(total_time, 1e-9)
    samples = []
    for idx in range(steps):
        alpha = idx / max(steps - 1, 1)
        time = alpha * total_time
        x = start[0] + alpha * (end[0] - start[0])
        y = start[1] + alpha * (end[1] - start[1])
        samples.append(DynamicObstacleSample(time=time, x=x, y=y, vx=vx, vy=vy))
    return DynamicObstacleTrack(obstacle_id=obstacle_id, radius=radius, samples=samples)


def _propagate_state6_surrogate(previous: Optional[State6DOF], state3: State3DOF, time: float, agent_id: int) -> State6DOF:
    phi = 0.03 * math.sin(0.35 * time + 0.4 * agent_id)
    theta = 0.02 * math.cos(0.28 * time + 0.3 * agent_id)
    return State6DOF(
        x=state3.x,
        y=state3.y,
        z=0.0,
        phi=phi,
        theta=theta,
        psi=state3.psi,
        u=state3.u * (1.0 + 0.02 * math.sin(0.2 * time + agent_id)),
        v=state3.v + 0.01 * math.cos(0.31 * time + agent_id),
        w=0.0,
        p=0.02 * math.cos(0.35 * time + agent_id),
        q=-0.015 * math.sin(0.28 * time + agent_id),
        r=state3.r * (1.0 + 0.03 * math.cos(0.24 * time + agent_id)),
    )


def _footprint_polygon(state: State3DOF, length: float, width: float) -> List[Tuple[float, float]]:
    c = math.cos(state.psi)
    s = math.sin(state.psi)
    half_l = length / 2.0
    half_w = width / 2.0
    corners = [
        (+half_l, +half_w),
        (+half_l, -half_w),
        (-half_l, -half_w),
        (-half_l, +half_w),
    ]
    polygon = []
    for local_x, local_y in corners:
        x = state.x + c * local_x - s * local_y
        y = state.y + s * local_x + c * local_y
        polygon.append((x, y))
    return polygon
