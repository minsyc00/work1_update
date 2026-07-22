from __future__ import annotations

import argparse
import json
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
    load_fleet_profile_json,
    load_map_for_planner,
    load_map_json,
    run_paper_style_region_tsp_experiment,
)


def build_usv_fleet(
    area_length_x: float = 20.0,
    area_length_y: float = 20.0,
    min_turn_radius: float = 2.0,
    count: int = 2,
) -> FleetConfig:
    start_x = min(max(area_length_x * 0.1, 1.0), max(area_length_x - 1.0, 1.0))
    count = max(int(count), 1)
    if count == 1:
        fractions = [0.5]
    elif count == 2:
        fractions = [0.12, 0.88]
    elif count == 3:
        fractions = [0.12, 0.50, 0.88]
    else:
        fractions = [0.12 + 0.76 * idx / max(count - 1, 1) for idx in range(count)]
    states = []
    for idx, fraction in enumerate(fractions):
        y = min(max(area_length_y * fraction, 1.0), max(area_length_y - 1.0, 1.0))
        if count == 3 and idx == 1:
            psi = 0.0
        elif fraction < 0.5:
            psi = math.pi / 2.0
        elif fraction > 0.5:
            psi = -math.pi / 2.0
        else:
            psi = 0.0
        states.append(State3DOF(x=start_x, y=y, psi=psi))
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


def build_two_usv_fleet(area_length_x: float = 20.0, area_length_y: float = 20.0, min_turn_radius: float = 2.0) -> FleetConfig:
    return build_usv_fleet(area_length_x, area_length_y, min_turn_radius=min_turn_radius, count=2)


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
    parser.add_argument("--performance-profile", choices=["balanced", "shortest", "low-repeat"], default="balanced", help="Path performance optimization profile.")
    parser.add_argument(
        "--target-coverage",
        type=_coverage_fraction_arg,
        default=0.99,
        help="Target free-space coverage fraction in (0, 1].",
    )
    parser.add_argument(
        "--cover-only-target",
        type=_coverage_fraction_arg,
        default=None,
        help="Target coverage fraction using cover segments only. Defaults to --target-coverage.",
    )
    parser.add_argument(
        "--max-residual-backfill-regions",
        type=int,
        default=None,
        help="Maximum skipped/residual regions to try recovering per pass.",
    )
    parser.add_argument(
        "--residual-backfill-cycles",
        type=int,
        default=None,
        help="Maximum residual-local-TSP backfill cycles.",
    )
    parser.add_argument("--run-parameter-sweep", action="store_true", help="Run a small non-rendered parameter sweep before rendering the selected result.")
    parser.add_argument("--usv-count", type=int, default=None, help="Override USV count. If omitted, maps >=50m use notes.recommended_usv_count when available.")
    parser.add_argument(
        "--fleet-profile",
        type=str,
        default=None,
        help="Independent heterogeneous fleet JSON. It defines agent count, starts, footprints, and motion limits.",
    )
    parser.add_argument(
        "--residual-filter-low-efficiency-always",
        action="store_true",
        help="Filter low-efficiency residual candidates even before the total target coverage is reached.",
    )
    parser.add_argument("--no-score-components", action="store_true", help="Do not write detailed pattern/connector score components to metadata.")
    parser.add_argument("--monitor-stages", action="store_true", help="Print JSON-line timing diagnostics for every planning stage.")
    parser.add_argument("--no-render", action="store_true", help="Run planning and write a report without generating PNG/GIF artifacts.")
    args = parser.parse_args()

    map_path = pathlib.Path(args.map)
    map_data = load_map_json(map_path)
    mission_area = map_data.get("mission_area", {})
    area_length_x = float(mission_area.get("length_x", 20.0))
    area_length_y = float(mission_area.get("length_y", 20.0))
    recommended_count = int(map_data.get("notes", {}).get("recommended_usv_count", 2) or 2)
    default_count = recommended_count if max(area_length_x, area_length_y) >= 50.0 else 2
    if args.fleet_profile and args.usv_count is not None:
        parser.error("--usv-count cannot be combined with --fleet-profile")
    if args.fleet_profile and args.rmin is not None:
        parser.error("--rmin cannot be combined with --fleet-profile; set each agent radius in the profile")
    usv_count = args.usv_count or default_count
    agent_profiles = None
    fleet_profile_id = ""
    if args.fleet_profile:
        fleet, agent_profiles, fleet_profile_id = load_fleet_profile_json(args.fleet_profile)
    else:
        fleet = build_usv_fleet(area_length_x, area_length_y, min_turn_radius=args.rmin or 2.0, count=usv_count)
    config, static_obstacles = load_map_for_planner(
        map_path,
        fleet,
        agent_profiles=agent_profiles,
        fleet_profile_id=fleet_profile_id,
    )
    if args.rmin is not None:
        config.fleet = replace(config.fleet, min_turn_radius=float(args.rmin))
    output_dir = build_experiment_output_dir(map_path, config, outputs_root=args.outputs_root)
    base_path_config = PathPlanningConfig.from_planner_config(config)
    path_config = replace(
        base_path_config,
        visual_map_id=str(map_data.get("map_id") or map_path.stem),
        visual_dpi=args.dpi,
        tsp_2opt_iterations=args.tsp_2opt_iterations,
        tsp_solver=args.tsp_solver,
        aco_ant_count=args.aco_ants,
        aco_iterations=args.aco_iterations,
        aco_random_seed=args.aco_seed,
        performance_profile=args.performance_profile,
        target_coverage_fraction=args.target_coverage,
        cover_only_target_fraction=args.cover_only_target,
        residual_filter_after_target_only=not args.residual_filter_low_efficiency_always,
        report_score_components=not args.no_score_components,
        max_residual_backfill_regions=(
            base_path_config.max_residual_backfill_regions
            if args.max_residual_backfill_regions is None
            else max(0, int(args.max_residual_backfill_regions))
        ),
        residual_backfill_cycles=(
            base_path_config.residual_backfill_cycles
            if args.residual_backfill_cycles is None
            else max(0, int(args.residual_backfill_cycles))
        ),
        monitor_stages=args.monitor_stages,
    )
    sweep_records = []
    if args.run_parameter_sweep:
        path_config, sweep_records = _select_sweep_config(
            config=config,
            static_obstacles=static_obstacles,
            output_dir=output_dir,
            base_path_config=path_config,
            map_id=str(map_data.get("map_id") or map_path.stem),
        )

    path_plan, report = run_paper_style_region_tsp_experiment(
        config=config,
        static_obstacles=static_obstacles,
        output_dir=output_dir,
        path_config=path_config,
        map_id=str(map_data.get("map_id") or map_path.stem),
        render=not args.no_render,
    )
    if args.no_render:
        artifact_dir = pathlib.Path(output_dir) / "paper_style_region_tsp"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        report_path = artifact_dir / "paper_style_region_tsp_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        path_plan.metadata["paper_style_output_dir"] = str(artifact_dir)
        path_plan.metadata["paper_style_report"] = str(report_path)
    if sweep_records:
        report["parameter_sweep"] = sweep_records
        report_path = pathlib.Path(path_plan.metadata.get("paper_style_report", ""))
        if report_path:
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        path_plan.metadata["parameter_sweep_count"] = str(len(sweep_records))

    print(f"paper_style_dir: {path_plan.metadata.get('paper_style_output_dir')}")
    print(f"report: {path_plan.metadata.get('paper_style_report')}")
    print(f"usv_count: {len(config.fleet.initial_states_3dof)}")
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
    if report.get("performance_summary"):
        summary = report["performance_summary"]
        print(f"performance_profile: {summary.get('performance_profile')}")
        print(f"transition_length_ratio: {summary.get('transition_length_ratio'):.6f}")
        print(f"repeat_transition_ratio: {summary.get('repeat_transition_ratio'):.6f}")
        print(f"target_coverage_met: {summary.get('target_coverage_met')}")


def _select_sweep_config(
    config,
    static_obstacles,
    output_dir: pathlib.Path,
    base_path_config: PathPlanningConfig,
    map_id: str,
) -> tuple[PathPlanningConfig, list[dict]]:
    candidates = [
        (6.0, 0.65, 1.0, 0.5),
        (12.0, 0.65, 1.5, 1.0),
        (18.0, 0.80, 1.5, 0.5),
        (12.0, 0.80, 1.0, 0.0),
    ]
    records = []
    best = None
    for idx, (repeat_weight, coverage_floor, merge_factor, turn_pocket_scale) in enumerate(candidates):
        candidate_config = replace(
            base_path_config,
            main_repeat_path_penalty_weight=repeat_weight,
            internal_uturn_repeat_path_penalty_weight=repeat_weight,
            repeat_transition_weight=repeat_weight,
            multi_entry_exit_coverage_floor=coverage_floor,
            cell_merge_width_factor=merge_factor,
            coverage_turn_pocket_scale=turn_pocket_scale,
        )
        _, report = run_paper_style_region_tsp_experiment(
            config=config,
            static_obstacles=static_obstacles,
            output_dir=output_dir / "_parameter_sweep",
            path_config=candidate_config,
            map_id=f"{map_id}_sweep_{idx}",
            render=False,
        )
        summary = report.get("performance_summary", {})
        metrics = report.get("metrics", {})
        record = {
            "index": idx,
            "main_repeat_weight": repeat_weight,
            "coverage_floor": coverage_floor,
            "cell_merge_width_factor": merge_factor,
            "turn_pocket_scale": turn_pocket_scale,
            "coverage_fraction": report.get("coverage_fraction", 0.0),
            "performance_objective": summary.get("performance_objective", float("inf")),
            "target_coverage_met": bool(summary.get("target_coverage_met", False)),
            "constraint_ok": bool(summary.get("constraint_ok", False)),
            "total_length": metrics.get("total_length", 0.0),
            "transition_length_ratio": summary.get("transition_length_ratio", 0.0),
            "repeat_transition_ratio": summary.get("repeat_transition_ratio", 0.0),
        }
        records.append(record)
        key = (
            not record["constraint_ok"],
            not record["target_coverage_met"],
            float(record["performance_objective"]),
            -float(record["coverage_fraction"]),
            float(record["total_length"]),
        )
        if best is None or key < best[0]:
            best = (key, candidate_config)
    return (best[1] if best is not None else base_path_config), records


def _coverage_fraction_arg(value: str) -> float:
    fraction = float(value)
    if fraction <= 0.0 or fraction > 1.0:
        raise argparse.ArgumentTypeError("--target-coverage must be in (0, 1].")
    return fraction


if __name__ == "__main__":
    main()
