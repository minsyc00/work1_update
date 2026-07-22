from __future__ import annotations

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
    PathPlanningLayer,
    PlannerConfig,
    PlannerWeights,
    SafetyMargins,
    State3DOF,
    State6DOF,
)
from usv_swarm.path_planning.adapters.runtime_adapter import path_plan_to_trajectory_references  # noqa: E402
from usv_swarm.path_planning.obstacles import (  # noqa: E402
    circle_obstacle,
    ellipse_obstacle,
    polygon_obstacle,
    rectangle_obstacle,
)


def build_static_obstacle_demo_config() -> PlannerConfig:
    states_3dof = [
        State3DOF(x=0.5, y=2.0, psi=0.0),
        State3DOF(x=0.5, y=9.0, psi=0.0),
        State3DOF(x=0.5, y=16.0, psi=0.0),
    ]
    states_6dof = [State6DOF(x=state.x, y=state.y, psi=state.psi) for state in states_3dof]
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
        safety=SafetyMargins(d_safe=2.5, boundary_margin_x=0.2, boundary_margin_y=0.2),
    )


def build_static_obstacles():
    return [
        rectangle_obstacle("rect_pier", center=(12.0, 4.0), width=1.2, height=1.2),
        circle_obstacle("round_buoy", center=(24.0, 4.0), radius=0.6),
        ellipse_obstacle("elliptic_reef", center=(34.0, 4.0), radii=(0.8, 0.5), psi=0.25),
        polygon_obstacle("poly_rock", [(42.0, 3.0), (43.4, 3.4), (42.8, 4.5)]),
    ]


def main() -> None:
    config = build_static_obstacle_demo_config()
    static_obstacles = build_static_obstacles()
    path_plan = PathPlanningLayer().plan_from_config(config, static_obstacles=static_obstacles)
    refs = path_plan_to_trajectory_references(path_plan)

    print(f"algorithm: {path_plan.algorithm_name}")
    print(f"static_obstacle_aware: {path_plan.metadata.get('static_obstacle_aware')}")
    print(f"static_obstacle_count: {path_plan.metadata.get('static_obstacle_count')}")
    print(f"free-space regions: {path_plan.metadata.get('region_count')}")
    print(f"coverage_fraction: {path_plan.metadata.get('coverage_fraction')}")
    print(f"load_imbalance_ratio: {path_plan.metadata.get('load_imbalance_ratio')}")
    for agent_id in sorted(path_plan.agents):
        agent_plan = path_plan.agents[agent_id]
        print(
            f"agent {agent_id}: regions={agent_plan.metrics.get('region_count', 0):.0f}, "
            f"segments={agent_plan.metrics.get('segment_count', 0):.0f}, "
            f"length={agent_plan.metrics.get('total_length', 0.0):.2f}, "
            f"max_kappa={agent_plan.metrics.get('max_curvature', 0.0):.3f}, "
            f"ref_samples={len(refs[agent_id].samples)}"
        )


if __name__ == "__main__":
    main()
