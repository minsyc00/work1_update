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
    run_planning_algorithm_experiment,
)


def build_two_usv_fleet() -> FleetConfig:
    states = [
        State3DOF(x=2.0, y=2.0, psi=math.pi / 2.0),
        State3DOF(x=2.0, y=18.0, psi=-math.pi / 2.0),
    ]
    return FleetConfig(
        initial_states_3dof=states,
        initial_states_6dof=[State6DOF(x=state.x, y=state.y, psi=state.psi) for state in states],
        cruise_speed=2.0,
        cover_speed=1.2,
        turn_speed_max=1.0,
        max_thrust=2.0,
        max_yaw_moment=1.0,
        min_turn_radius=2.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full path-planning algorithm experiment and export algorithm-step figures.")
    parser.add_argument(
        "--map",
        type=str,
        default=str(ROOT / "maps" / "static_obstacle_map_20x20_two_obstacles" / "static_obstacle_map_20x20_two_obstacles.json"),
        help="Static obstacle map JSON.",
    )
    parser.add_argument("--outputs-root", type=str, default=str(ROOT / "outputs"), help="Root output directory.")
    parser.add_argument("--dpi", type=int, default=140, help="PNG DPI.")
    parser.add_argument("--gif-fps", type=int, default=4, help="Algorithm-process GIF FPS.")
    args = parser.parse_args()

    map_path = pathlib.Path(args.map)
    map_data = load_map_json(map_path)
    config, static_obstacles = load_map_for_planner(map_path, build_two_usv_fleet())
    output_dir = build_experiment_output_dir(map_path, config, outputs_root=args.outputs_root)
    path_config = replace(
        PathPlanningConfig.from_planner_config(config),
        visual_map_id=str(map_data.get("map_id") or map_path.stem),
        visual_dpi=args.dpi,
        visual_gif_fps=args.gif_fps,
        tsp_2opt_iterations=8,
        max_residual_backfill_regions=8,
        residual_backfill_cycles=2,
    )

    path_plan, trace = run_planning_algorithm_experiment(
        config=config,
        static_obstacles=static_obstacles,
        output_dir=output_dir,
        path_config=path_config,
        map_id=str(map_data.get("map_id") or map_path.stem),
        render=True,
    )
    print(f"algorithm_steps: {path_plan.metadata.get('algorithm_experiment_dir')}")
    print(f"report: {path_plan.metadata.get('algorithm_experiment_report')}")
    print(f"coverage_fraction: {path_plan.metadata.get('coverage_fraction')}")
    print(f"region_count: {path_plan.metadata.get('region_count')}")
    print(f"stage_count: {len(trace.stage_metrics)}")
    for agent_id, record in sorted(trace.tsp_records.items()):
        print(
            f"agent {agent_id}: initial_regions={len(record.get('initial_order', []))}, "
            f"final_regions={len(record.get('final_order', []))}, "
            f"2opt_improvements={len(record.get('two_opt_improvements', []))}"
        )


if __name__ == "__main__":
    main()
