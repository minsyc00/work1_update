from __future__ import annotations

import pathlib
import sys

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
    plan_global_coverage,
)


def build_demo_config() -> PlannerConfig:
    initial_states_3dof = [
        State3DOF(x=0.0, y=2.0, psi=0.0),
        State3DOF(x=0.0, y=8.0, psi=0.0),
        State3DOF(x=0.0, y=14.0, psi=0.0),
    ]
    initial_states_6dof = [
        State6DOF(x=state.x, y=state.y, psi=state.psi, u=state.u, v=state.v, r=state.r)
        for state in initial_states_3dof
    ]
    return PlannerConfig(
        mission=MissionConfig(area_length_x=60.0, area_length_y=24.0, overlap_ratio=0.15, local_control_hz=5.0),
        fleet=FleetConfig(
            initial_states_3dof=initial_states_3dof,
            initial_states_6dof=initial_states_6dof,
            cruise_speed=3.0,
            cover_speed=2.0,
            turn_speed_max=1.5,
            max_thrust=2.5,
            max_yaw_moment=1.5,
            min_turn_radius=4.0,
        ),
        footprint=CoverageFootprint(length_lf=4.0, width_wf=5.0, eta_cov=0.7),
        weights=PlannerWeights(),
        safety=SafetyMargins(d_safe=3.0, boundary_margin_x=0.2, boundary_margin_y=0.2, delta_safe_max=1.0, t_block=8.0),
    )


def main() -> None:
    config = build_demo_config()
    plan = plan_global_coverage(config)
    runtime = SwarmRuntime(config, plan)
    print(f"strips: {len(plan.strips)}")
    print(f"makespan: {plan.reservations.makespan:.2f}s")
    for agent_id in range(config.fleet.num_agents or 0):
        task_range = plan.assignments.assignments[agent_id]
        ref_count = len(plan.refs[agent_id].samples)
        print(f"agent {agent_id}: strips={task_range}, ref_samples={ref_count}")
        step = runtime.control_step(
            AgentRuntimeState(
                agent_id=agent_id,
                time=0.0,
                state3=config.fleet.initial_states_3dof[agent_id],
                state6=config.fleet.initial_states_6dof[agent_id],
            )
        )
        print(
            f"  cmd=(thrust={step.cmd.thrust:.2f}, yaw={step.cmd.yaw_moment:.2f}) "
            f"mode={step.safety_status.mode} margin={step.safety_status.min_margin:.2f}"
        )
    print(f"coverage fraction after one update: {runtime.coverage.state.coverage_fraction:.3f}")


if __name__ == "__main__":
    main()
