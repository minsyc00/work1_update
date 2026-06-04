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
    PathPlanningConfig,
    PathPlanningLayer,
    PlannerConfig,
    PlannerWeights,
    SafetyMargins,
    State3DOF,
    State6DOF,
)
from usv_swarm.path_planning.adapters.runtime_adapter import path_plan_to_trajectory_references  # noqa: E402


def build_path_planning_demo_config(agent_count: int = 4) -> PlannerConfig:
    y_values = [2.0 + 20.0 * idx / max(agent_count - 1, 1) for idx in range(agent_count)]
    initial_states_3dof = [State3DOF(x=0.5, y=y, psi=0.0) for y in y_values]
    initial_states_6dof = [State6DOF(x=state.x, y=state.y, psi=state.psi) for state in initial_states_3dof]
    return PlannerConfig(
        mission=MissionConfig(area_length_x=64.0, area_length_y=24.0, overlap_ratio=0.15, local_control_hz=5.0),
        fleet=FleetConfig(
            initial_states_3dof=initial_states_3dof,
            initial_states_6dof=initial_states_6dof,
            cruise_speed=3.0,
            cover_speed=2.0,
            turn_speed_max=1.4,
            max_thrust=2.5,
            max_yaw_moment=1.4,
            min_turn_radius=4.0,
        ),
        footprint=CoverageFootprint(length_lf=4.0, width_wf=5.0, eta_cov=0.7),
        weights=PlannerWeights(),
        safety=SafetyMargins(d_safe=3.0, boundary_margin_x=0.25, boundary_margin_y=0.25),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper-fusion path planning layer demo.")
    parser.add_argument("--agents", type=int, default=4, help="Number of homogeneous USVs.")
    parser.add_argument("--three-opt", action="store_true", help="Enable one deterministic 3-opt pass.")
    args = parser.parse_args()

    config = build_path_planning_demo_config(agent_count=max(args.agents, 1))
    path_config = PathPlanningConfig.from_planner_config(config)
    if args.three_opt:
        path_config = PathPlanningConfig(
            overlap_ratio=path_config.overlap_ratio,
            coverage_resolution=path_config.coverage_resolution,
            residual_resolution=path_config.residual_resolution,
            tsp_3opt_iterations=1,
        )

    path_plan = PathPlanningLayer().plan_from_config(config, path_config=path_config)
    refs = path_plan_to_trajectory_references(path_plan)

    print(f"algorithm: {path_plan.algorithm_name}")
    print(f"status: {path_plan.metadata.get('status')}")
    print(f"regions: {path_plan.metadata.get('region_count')}")
    print(f"coverage_fraction: {path_plan.metadata.get('coverage_fraction')}")
    print(f"load_imbalance_ratio: {path_plan.metadata.get('load_imbalance_ratio')}")
    print(f"planning_time: {path_plan.metadata.get('planning_time')}s")
    for agent_id in sorted(path_plan.agents):
        plan = path_plan.agents[agent_id]
        print(
            f"agent {agent_id}: regions={plan.metrics.get('region_count', 0):.0f}, "
            f"segments={plan.metrics.get('segment_count', 0):.0f}, "
            f"length={plan.metrics.get('total_length', 0):.2f}, "
            f"turn={plan.metrics.get('total_turn_angle', 0):.2f}, "
            f"max_kappa={plan.metrics.get('max_curvature', 0):.3f}, "
            f"ref_samples={len(refs[agent_id].samples)}"
        )


if __name__ == "__main__":
    main()
