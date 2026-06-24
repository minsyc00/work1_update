from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from usv_swarm import (  # noqa: E402
    CoverageFootprint,
    FleetConfig,
    MissionConfig,
    PlannerConfig,
    PlannerWeights,
    SafetyMargins,
    State3DOF,
    State6DOF,
    plan_global_coverage,
)
from usv_swarm.simulation import (  # noqa: E402
    build_crossing_obstacle_scenario,
    render_simulation_animation,
    simulate_swarm_closed_loop,
)


def build_closed_loop_demo_config() -> PlannerConfig:
    initial_states_3dof = [
        State3DOF(x=2.0, y=2.0, psi=0.0),
        State3DOF(x=2.0, y=10.0, psi=0.0),
        State3DOF(x=2.0, y=18.0, psi=0.0),
    ]
    initial_states_6dof = [
        State6DOF(x=state.x, y=state.y, psi=state.psi, u=state.u, v=state.v, r=state.r)
        for state in initial_states_3dof
    ]
    return PlannerConfig(
        mission=MissionConfig(area_length_x=52.0, area_length_y=24.0, overlap_ratio=0.15, global_replan_hz=0.5, local_control_hz=4.0),
        fleet=FleetConfig(
            initial_states_3dof=initial_states_3dof,
            initial_states_6dof=initial_states_6dof,
            cruise_speed=2.8,
            cover_speed=1.9,
            turn_speed_max=1.2,
            max_thrust=2.6,
            max_yaw_moment=1.4,
            min_turn_radius=4.2,
        ),
        footprint=CoverageFootprint(length_lf=4.0, width_wf=5.0, eta_cov=0.7),
        weights=PlannerWeights(w_soft=25.0),
        safety=SafetyMargins(d_safe=2.8, boundary_margin_x=0.25, boundary_margin_y=0.25, delta_safe_max=1.2, t_block=8.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-USV closed-loop coverage simulation and export a GIF animation.")
    parser.add_argument("--total-time", type=float, default=28.0, help="Simulation horizon in seconds.")
    parser.add_argument("--fps", type=int, default=6, help="GIF frame rate.")
    parser.add_argument("--control-mode", choices=["fast_tracker", "hybrid_nmpc", "full_nmpc"], default="hybrid_nmpc", help="Online control mode.")
    parser.add_argument("--nmpc-update-interval", type=int, default=5, help="Hybrid NMPC update interval in control steps.")
    parser.add_argument("--nmpc-horizon-seconds", type=float, default=1.2, help="NMPC prediction horizon in seconds.")
    parser.add_argument("--nmpc-horizon-cap", type=int, default=10, help="Maximum NMPC horizon steps.")
    parser.add_argument("--nmpc-max-wall-time-ms", type=float, default=80.0, help="NMPC solve-time budget before tracker fallback.")
    parser.add_argument(
        "--nmpc-parallel-backend",
        choices=["serial", "thread", "process"],
        default="serial",
        help="NMPC execution backend. Use process for worker-local CasADi with hard timeout.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "outputs" / "usv_swarm_closed_loop.gif"),
        help="Output animation path.",
    )
    args = parser.parse_args()

    config = build_closed_loop_demo_config()
    config.mission.control_mode = args.control_mode
    config.mission.nmpc_update_interval_steps = args.nmpc_update_interval
    config.mission.nmpc_horizon_seconds = args.nmpc_horizon_seconds
    config.mission.nmpc_horizon_steps_cap = args.nmpc_horizon_cap
    config.mission.nmpc_max_wall_time_ms = args.nmpc_max_wall_time_ms
    config.mission.nmpc_parallel_backend = args.nmpc_parallel_backend
    plan = plan_global_coverage(config)
    total_time = args.total_time
    obstacle_tracks = build_crossing_obstacle_scenario(config, total_time=total_time, dt=1.0 / config.mission.local_control_hz)
    log = simulate_swarm_closed_loop(config, planning_result=plan, obstacle_tracks=obstacle_tracks, total_time=total_time)
    output_path = pathlib.Path(args.output)
    render_simulation_animation(log, output_path, fps=args.fps)

    print(f"frames: {len(log.frames)}")
    print(f"final coverage: {log.final_coverage_fraction:.3f}")
    print(f"runtime profile: {log.runtime_profile.get('summary', {})}")
    print(f"animation: {output_path}")
    for agent_id in range(config.fleet.num_agents or 0):
        trajectory = log.trajectory(agent_id)
        print(f"agent {agent_id}: states={len(trajectory)}")


if __name__ == "__main__":
    main()
