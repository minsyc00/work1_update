from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from usv_swarm import (  # noqa: E402
    FleetConfig,
    PathPlanningConfig,
    State3DOF,
    State6DOF,
    build_experiment_output_dir,
    load_map_for_planner,
    load_map_json,
    run_paper_style_region_tsp_experiment,
)


def build_two_usv_fleet(area_length_x: float = 20.0, area_length_y: float = 20.0, min_turn_radius: float = 2.0) -> FleetConfig:
    start_x = min(max(area_length_x * 0.1, 1.0), max(area_length_x - 1.0, 1.0))
    lower_y = min(max(area_length_y * 0.12, 1.0), max(area_length_y - 1.0, 1.0))
    upper_y = max(min(area_length_y * 0.88, area_length_y - 1.0), lower_y)
    states = [
        State3DOF(x=start_x, y=lower_y, psi=math.pi / 2.0),
        State3DOF(x=start_x, y=upper_y, psi=-math.pi / 2.0),
    ]
    return FleetConfig(
        initial_states_3dof=states,
        initial_states_6dof=[State6DOF(x=state.x, y=state.y, psi=state.psi) for state in states],
        cruise_speed=2.0,
        cover_speed=1.2,
        turn_speed_max=1.0,
        max_thrust=2.0,
        max_yaw_moment=1.0,
        min_turn_radius=min_turn_radius,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the paper-style region-sweep + inter-region TSP coverage experiment."
    )
    parser.add_argument(
        "--map",
        type=str,
        default=str(
            ROOT
            / "maps"
            / "static_obstacle_map_20x20_two_obstacles"
            / "static_obstacle_map_20x20_two_obstacles.json"
        ),
        help="Static obstacle map JSON.",
    )
    parser.add_argument("--outputs-root", type=str, default=str(ROOT / "outputs"), help="Root output directory.")
    parser.add_argument("--dpi", type=int, default=140, help="PNG DPI.")
    parser.add_argument("--tsp-2opt-iterations", type=int, default=8, help="Maximum 2-opt iterations per USV.")
    parser.add_argument("--rmin", type=float, default=None, help="Override map min_turn_radius for this experiment.")
    parser.add_argument("--tsp-solver", choices=["deterministic", "aco", "fa3aco"], default="deterministic", help="Region TSP-CPP solver.")
    parser.add_argument("--aco-ants", type=int, default=30, help="Number of ants for ACO/FA3ACO.")
    parser.add_argument("--aco-iterations", type=int, default=80, help="ACO/FA3ACO iteration count.")
    parser.add_argument("--aco-seed", type=int, default=42, help="ACO/FA3ACO random seed.")
    args = parser.parse_args()

    map_path = pathlib.Path(args.map)
    map_data = load_map_json(map_path)
    mission_area = map_data.get("mission_area", {})
    area_length_x = float(mission_area.get("length_x", 20.0))
    area_length_y = float(mission_area.get("length_y", 20.0))
    config, static_obstacles = load_map_for_planner(
        map_path,
        build_two_usv_fleet(area_length_x, area_length_y, min_turn_radius=args.rmin or 2.0),
    )
    if args.rmin is not None:
        config.fleet = replace(config.fleet, min_turn_radius=float(args.rmin))
    output_dir = build_experiment_output_dir(map_path, config, outputs_root=args.outputs_root)
    path_config = replace(
        PathPlanningConfig.from_planner_config(config),
        visual_map_id=str(map_data.get("map_id") or map_path.stem),
        visual_dpi=args.dpi,
        tsp_2opt_iterations=args.tsp_2opt_iterations,
        tsp_solver=args.tsp_solver,
        aco_ant_count=args.aco_ants,
        aco_iterations=args.aco_iterations,
        aco_random_seed=args.aco_seed,
    )

    path_plan, report = run_paper_style_region_tsp_experiment(
        config=config,
        static_obstacles=static_obstacles,
        output_dir=output_dir,
        path_config=path_config,
        map_id=str(map_data.get("map_id") or map_path.stem),
        render=True,
    )

    print(f"paper_style_dir: {path_plan.metadata.get('paper_style_output_dir')}")
    print(f"report: {path_plan.metadata.get('paper_style_report')}")
    print(f"coverage_fraction: {path_plan.metadata.get('coverage_fraction')}")
    print(f"requested_tsp_solver: {path_plan.metadata.get('requested_tsp_solver')}")
    print(f"effective_tsp_solver: {path_plan.metadata.get('effective_tsp_solver')}")
    print(f"tsp_solver_status: {path_plan.metadata.get('tsp_solver_status')}")
    print(f"tsp_node_count: {path_plan.metadata.get('tsp_node_count')}")
    print(f"coverage_endpoint_count: {path_plan.metadata.get('coverage_endpoint_count')}")
    print(f"invalid_path_length: {path_plan.metadata.get('invalid_path_length')}")
    print(f"out_of_bounds_segment_count: {path_plan.metadata.get('out_of_bounds_segment_count')}")
    print(f"obstacle_collision_segment_count: {path_plan.metadata.get('obstacle_collision_segment_count')}")
    print(f"kinematic_infeasible_segment_count: {path_plan.metadata.get('kinematic_infeasible_segment_count')}")
    if report.get("infeasible_regions"):
        print(f"infeasible_regions: {report['infeasible_regions']}")
    if report.get("infeasible_edges"):
        print(f"infeasible_edges: {report['infeasible_edges']}")


if __name__ == "__main__":
    main()
