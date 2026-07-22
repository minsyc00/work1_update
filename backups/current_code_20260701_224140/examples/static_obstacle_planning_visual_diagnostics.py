from __future__ import annotations

import argparse
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
    PathPlanningLayer,
    State3DOF,
    State6DOF,
    build_experiment_output_dir,
    load_map_for_planner,
    load_map_json,
)


def build_default_fleet() -> FleetConfig:
    initial_states_3dof = [
        State3DOF(x=0.0, y=5.0, psi=0.0),
        State3DOF(x=0.0, y=25.0, psi=0.0),
        State3DOF(x=0.0, y=45.0, psi=0.0),
    ]
    return FleetConfig(
        initial_states_3dof=initial_states_3dof,
        initial_states_6dof=[State6DOF(x=state.x, y=state.y, psi=state.psi) for state in initial_states_3dof],
        cruise_speed=2.0,
        cover_speed=1.4,
        turn_speed_max=1.0,
        max_thrust=2.0,
        max_yaw_moment=1.0,
        min_turn_radius=2.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run static-obstacle coverage planning and export step-by-step visual diagnostics.")
    parser.add_argument(
        "--map",
        type=str,
        default=str(ROOT / "maps" / "static_obstacle_map_50x50_simple" / "static_obstacle_map_50x50_simple.json"),
        help="Static obstacle map JSON.",
    )
    parser.add_argument("--outputs-root", type=str, default=str(ROOT / "outputs"), help="Root directory for experiment outputs.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG DPI.")
    parser.add_argument("--gif-fps", type=int, default=6, help="Route monitor GIF frame rate.")
    args = parser.parse_args()

    map_path = pathlib.Path(args.map)
    map_data = load_map_json(map_path)
    fleet = build_default_fleet()
    config, static_obstacles = load_map_for_planner(map_path, fleet)
    output_dir = build_experiment_output_dir(map_path, config, outputs_root=args.outputs_root)
    path_config = replace(
        PathPlanningConfig.from_planner_config(config),
        visual_output_dir=str(output_dir),
        visual_map_id=str(map_data.get("map_id") or map_path.stem),
        visual_dpi=args.dpi,
        visual_gif_fps=args.gif_fps,
    )

    path_plan = PathPlanningLayer().plan_from_config(
        config=config,
        static_obstacles=static_obstacles,
        path_config=path_config,
    )
    print(f"output_dir: {path_plan.metadata.get('visual_output_dir', output_dir)}")
    print(f"manifest: {path_plan.metadata.get('visualization_manifest', '')}")
    print(f"coverage_fraction: {path_plan.metadata.get('coverage_fraction', '')}")
    print(f"region_count: {path_plan.metadata.get('region_count', '')}")
    print(f"mapf_conflicts_resolved: {path_plan.metadata.get('mapf_conflicts_resolved', '')}")


if __name__ == "__main__":
    main()
