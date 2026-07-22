from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys
from time import perf_counter
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
    VehicleFootprint,
    build_experiment_output_dir,
    load_map_for_planner,
    load_map_json,
    load_fleet_profile_json,
    resolve_default_fleet_profile_path,
    run_crown_mcpp_experiment,
    validate_fleet_profile_for_map,
)


def _fleet(length_x: float, length_y: float, count: int, turn_radius: float) -> FleetConfig:
    count = max(count, 1)
    # Manual fleets have an explicitly declared hull later in ``main``.  This
    # helper only chooses conservative interior seed poses; it never infers a
    # physical body from the map's sensing footprint.
    x = max(2.1, min(length_x * 0.15, length_x - 2.1))
    if count == 1:
        fractions = [0.15]
    else:
        fractions = [0.15 + 0.70 * index / (count - 1) for index in range(count)]
    states = [
        State3DOF(
            x=x,
            y=max(1.1, min(length_y * fractions[index], length_y - 1.1)),
            psi=0.0,
        )
        for index in range(count)
    ]
    return FleetConfig(
        initial_states_3dof=states,
        initial_states_6dof=[
            State6DOF(x=state.x, y=state.y, psi=state.psi) for state in states
        ],
        cruise_speed=2.0,
        cover_speed=1.2,
        turn_speed_max=1.0,
        max_thrust=2.0,
        max_yaw_moment=1.0,
        min_turn_radius=turn_radius,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete CROWN-MCPP experiment.")
    parser.add_argument(
        "--map",
        default=str(
            ROOT
            / "maps"
            / "static_obstacle_map_20x20_two_obstacles"
            / "static_obstacle_map_20x20_two_obstacles.json"
        ),
    )
    parser.add_argument(
        "--fleet-profile",
        default=None,
        help="Fleet JSON; defaults to notes.default_fleet_profile beside the selected map.",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=None,
        help="Manual fleet size. Requires explicit physical vehicle dimensions.",
    )
    parser.add_argument("--engine", choices=["auto", "bpc", "certified_lns"], default=None)
    parser.add_argument("--time-step", type=float, default=None)
    parser.add_argument("--horizon", type=float, default=None)
    parser.add_argument("--lns-seconds", type=float, default=None)
    parser.add_argument("--baseline-seconds", type=float, default=None)
    parser.add_argument("--lns-iterations", type=int, default=None)
    parser.add_argument("--connector-max-expansions", type=int, default=None)
    parser.add_argument("--connector-grid-resolution", type=float, default=None)
    parser.add_argument("--mode-limit", type=int, default=None)
    parser.add_argument("--max-candidate-axes", type=int, default=None)
    parser.add_argument("--resource-grid-size", type=float, default=None)
    parser.add_argument(
        "--turn-power",
        type=float,
        default=None,
        help="Override turning power for every USV profile.",
    )
    parser.add_argument(
        "--turn-time-penalty-per-rad",
        type=float,
        default=None,
        help="Extra operational seconds charged per radian of heading change.",
    )
    parser.add_argument(
        "--turn-energy-penalty-per-rad",
        type=float,
        default=None,
        help="Extra energy charged per radian of heading change.",
    )
    parser.add_argument(
        "--turn-maneuver-time-penalty",
        type=float,
        default=None,
        help="Fixed operational seconds charged once per turning maneuver.",
    )
    parser.add_argument(
        "--turn-maneuver-energy-penalty",
        type=float,
        default=None,
        help="Fixed energy charged once per turning maneuver.",
    )
    root_group = parser.add_mutually_exclusive_group()
    root_group.add_argument(
        "--root-exact-pricing",
        dest="root_exact_pricing",
        action="store_true",
    )
    root_group.add_argument(
        "--no-root-exact-pricing",
        dest="root_exact_pricing",
        action="store_false",
    )
    parser.set_defaults(root_exact_pricing=None)
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument(
        "--sequential-baseline",
        dest="include_sequential_baseline",
        action="store_true",
    )
    baseline_group.add_argument(
        "--no-sequential-baseline",
        dest="include_sequential_baseline",
        action="store_false",
    )
    parser.set_defaults(include_sequential_baseline=None)
    parser.add_argument(
        "--turn-radius",
        type=float,
        default=None,
        help="Override the map motion-constraint radius for a declared fleet model.",
    )
    parser.add_argument(
        "--vehicle-length",
        type=float,
        default=None,
        help="Physical hull length for a manual fleet; never inferred from sensor coverage.",
    )
    parser.add_argument(
        "--vehicle-width",
        type=float,
        default=None,
        help="Physical hull width for a manual fleet; never inferred from sensor coverage.",
    )
    parser.add_argument("--outputs-root", default=str(ROOT / "outputs"))
    return_group = parser.add_mutually_exclusive_group()
    return_group.add_argument(
        "--return-to-start",
        dest="return_to_start",
        action="store_true",
    )
    return_group.add_argument(
        "--no-return",
        dest="return_to_start",
        action="store_false",
    )
    parser.set_defaults(return_to_start=None)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    map_path = pathlib.Path(args.map)
    data = load_map_json(map_path)
    mission = data.get("mission_area", {})
    length_x = float(mission.get("length_x", 20.0))
    length_y = float(mission.get("length_y", 20.0))
    manual_fleet = any(
        value is not None
        for value in (
            args.agents,
            args.turn_radius,
            args.vehicle_length,
            args.vehicle_width,
        )
    )
    if args.fleet_profile and manual_fleet:
        parser.error(
            "--fleet-profile cannot be combined with manual fleet geometry options"
        )
    if (args.vehicle_length is None) != (args.vehicle_width is None):
        parser.error("--vehicle-length and --vehicle-width must be supplied together")
    profile_data = {}
    profile_path = None
    if manual_fleet:
        if args.vehicle_length is None or args.vehicle_width is None:
            parser.error(
                "manual fleet configuration requires --vehicle-length and --vehicle-width; "
                "the coverage footprint is not a physical hull"
            )
        if args.agents is not None and args.agents <= 0:
            parser.error("--agents must be positive")
        if args.turn_radius is not None and args.turn_radius <= 0.0:
            parser.error("--turn-radius must be positive")
        if args.vehicle_length <= 0.0 or args.vehicle_width <= 0.0:
            parser.error("physical vehicle dimensions must be positive")
        recommended_count = int(data.get("notes", {}).get("recommended_usv_count", 2) or 2)
        turn_radius = float(
            args.turn_radius
            if args.turn_radius is not None
            else data.get("motion_constraints", {}).get("min_turn_radius", 2.0)
        )
        config, obstacles = load_map_for_planner(
            map_path,
            _fleet(length_x, length_y, args.agents or recommended_count, turn_radius),
        )
        config = replace(
            config,
            fleet=replace(config.fleet, min_turn_radius=turn_radius),
            vehicle_footprint=VehicleFootprint(
                length=args.vehicle_length,
                width=args.vehicle_width,
            ),
        )
    else:
        profile_path = (
            pathlib.Path(args.fleet_profile)
            if args.fleet_profile
            else resolve_default_fleet_profile_path(map_path)
        )
        if profile_path is None:
            parser.error(
                "the selected map has no default fleet profile; provide --fleet-profile "
                "or explicit --agents/--vehicle-length/--vehicle-width"
            )
        profile_data = validate_fleet_profile_for_map(map_path, profile_path)
        fleet, agent_profiles, fleet_profile_id = load_fleet_profile_json(profile_path)
        config, obstacles = load_map_for_planner(
            map_path,
            fleet,
            agent_profiles=agent_profiles,
            fleet_profile_id=fleet_profile_id,
        )

    turn_overrides = (
        args.turn_power,
        args.turn_time_penalty_per_rad,
        args.turn_energy_penalty_per_rad,
        args.turn_maneuver_time_penalty,
        args.turn_maneuver_energy_penalty,
    )
    if any(value is not None for value in turn_overrides):
        profiles = {
            agent_id: replace(
                config.profile_for_agent(agent_id),
                turn_power=(
                    args.turn_power
                    if args.turn_power is not None
                    else config.profile_for_agent(agent_id).turn_power
                ),
                turn_time_penalty_per_rad=(
                    args.turn_time_penalty_per_rad
                    if args.turn_time_penalty_per_rad is not None
                    else config.profile_for_agent(agent_id).turn_time_penalty_per_rad
                ),
                turn_energy_penalty_per_rad=(
                    args.turn_energy_penalty_per_rad
                    if args.turn_energy_penalty_per_rad is not None
                    else config.profile_for_agent(agent_id).turn_energy_penalty_per_rad
                ),
                turn_maneuver_time_penalty=(
                    args.turn_maneuver_time_penalty
                    if args.turn_maneuver_time_penalty is not None
                    else config.profile_for_agent(agent_id).turn_maneuver_time_penalty
                ),
                turn_maneuver_energy_penalty=(
                    args.turn_maneuver_energy_penalty
                    if args.turn_maneuver_energy_penalty is not None
                    else config.profile_for_agent(agent_id).turn_maneuver_energy_penalty
                ),
            )
            for agent_id in range(len(config.fleet.initial_states_3dof))
        }
        config = replace(config, agent_profiles=profiles)

    defaults = profile_data.get("planning_defaults", {})
    if not isinstance(defaults, dict):
        parser.error("fleet planning_defaults must be a JSON object")

    def selected(name: str, override, fallback):
        return override if override is not None else defaults.get(name, fallback)

    base = PathPlanningConfig.from_planner_config(config)
    path = replace(
        base,
        crown_engine=str(selected("engine", args.engine, "auto")),
        crown_time_step=float(selected("time_step", args.time_step, 1.0)),
        crown_horizon=args.horizon,
        crown_lns_time_budget_sec=float(
            selected("lns_time_budget_sec", args.lns_seconds, 60.0)
        ),
        crown_baseline_time_budget_sec=float(
            selected("baseline_time_budget_sec", args.baseline_seconds, 30.0)
        ),
        crown_include_sequential_baseline=bool(
            selected(
                "include_sequential_baseline",
                args.include_sequential_baseline,
                True,
            )
        ),
        crown_lns_iterations=int(
            selected("lns_iterations", args.lns_iterations, 500)
        ),
        crown_connector_max_expansions=int(
            selected(
                "connector_max_expansions",
                args.connector_max_expansions,
                2000,
            )
        ),
        obstacle_aware_grid_resolution=float(
            selected(
                "connector_grid_resolution",
                args.connector_grid_resolution,
                base.obstacle_aware_grid_resolution
                or base.coverage_resolution
                or config.footprint.width_wf * 0.5,
            )
        ),
        crown_mode_limit_per_region_agent=int(
            selected("mode_limit_per_region_agent", args.mode_limit, 8)
        ),
        crown_root_exact_pricing=bool(
            selected("root_exact_pricing", args.root_exact_pricing, True)
        ),
        crown_resource_grid_size=float(
            selected("resource_grid_size", args.resource_grid_size, 1.0)
        ),
        crown_return_to_start=bool(
            selected("return_to_start", args.return_to_start, True)
        ),
        max_candidate_axes=int(
            selected("max_candidate_axes", args.max_candidate_axes, base.max_candidate_axes)
        ),
    )
    output = build_experiment_output_dir(map_path, config, args.outputs_root)
    experiment_started = perf_counter()
    try:
        _, report = run_crown_mcpp_experiment(
            config,
            obstacles,
            output,
            path,
            map_id=str(data.get("map_id") or map_path.stem),
            render=not args.no_render,
        )
    except Exception as error:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak *= 1024.0
        failure_report = {
            "algorithm": "CROWN-MCPP",
            "map_id": str(data.get("map_id") or map_path.stem),
            "solution_status": "failed_before_first_feasible_solution",
            "runtime_sec": perf_counter() - experiment_started,
            "peak_rss_mb": peak / (1024.0 * 1024.0),
            "configured_lns_time_budget_sec": path.crown_lns_time_budget_sec,
            "fleet_profile_id": config.fleet_profile_id or None,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        failure_path = output / "crown_mcpp" / "crown_mcpp_failure_report.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_report["artifacts"] = {"failure_report": str(failure_path)}
        failure_path.write_text(
            json.dumps(failure_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(failure_report, indent=2, ensure_ascii=False))
        raise
    report["fleet_profile"] = str(profile_path) if profile_path is not None else None
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
