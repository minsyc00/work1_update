"""Reproducible end-to-end experiment/report entry for CROWN-MCPP."""

from __future__ import annotations

import json
import math
from pathlib import Path
import resource
import sys
from time import perf_counter
from typing import Any, Dict, Optional, Sequence, Tuple

from ...schema import PlannerConfig
from ...geometry import wrap_angle
from ..types import MultiAgentPathPlan, PaperReference, PathPlanningConfig, StaticObstacle
from .pipeline import run_crown_mcpp_pipeline


def run_crown_mcpp_experiment(
    config: PlannerConfig,
    static_obstacles: Optional[Sequence[StaticObstacle]],
    output_dir: str | Path,
    path_config: Optional[PathPlanningConfig] = None,
    *,
    map_id: str = "",
    paper_references: Optional[Sequence[PaperReference]] = None,
    render: bool = True,
) -> Tuple[MultiAgentPathPlan, Dict[str, Any]]:
    """Run CROWN, persist its certificate, and optionally render core plots."""

    started = perf_counter()
    output = Path(output_dir) / "crown_mcpp"
    output.mkdir(parents=True, exist_ok=True)
    plan = run_crown_mcpp_pipeline(
        config,
        path_config,
        static_obstacles,
        paper_references,
    )
    baseline = _optional_float(plan.metadata.get("sequential_baseline_makespan"))
    makespan = _optional_float(plan.metadata.get("makespan")) or 0.0
    lower_bound = _optional_float(plan.metadata.get("lower_bound")) or 0.0
    effective_path_config = path_config or PathPlanningConfig.from_planner_config(config)
    active_agent_count = sum(
        bool(agent.segments) for agent in plan.agents.values()
    )
    turn_metrics = {
        agent_id: _turn_metrics(agent.segments)
        for agent_id, agent in plan.agents.items()
    }
    total_turn_angle = sum(
        metrics["total_turn_angle_rad"] for metrics in turn_metrics.values()
    )
    total_turn_maneuvers = sum(
        metrics["turn_maneuver_count"] for metrics in turn_metrics.values()
    )
    report: Dict[str, Any] = {
        "algorithm": "CROWN-MCPP",
        "map_id": map_id,
        "solution_status": plan.metadata.get("status", ""),
        "certification_scope": plan.metadata.get("certification_scope", ""),
        "runtime_sec": perf_counter() - started,
        "preparation_runtime_sec": _optional_float(
            plan.metadata.get("preparation_runtime_sec")
        ),
        "solve_runtime_sec": _optional_float(plan.metadata.get("solve_runtime_sec")),
        "materialization_validation_runtime_sec": _optional_float(
            plan.metadata.get("materialization_validation_runtime_sec")
        ),
        "peak_rss_mb": _peak_rss_mb(),
        "fleet_size": len(plan.agents),
        "active_agent_count": active_agent_count,
        "configured_lns_time_budget_sec": (
            effective_path_config.crown_lns_time_budget_sec
        ),
        "configured_return_to_start": effective_path_config.crown_return_to_start,
        "connector_grid_resolution": (
            effective_path_config.obstacle_aware_grid_resolution
        ),
        "fleet_profile_id": config.fleet_profile_id or None,
        "fleet": {
            str(agent_id): {
                "initial_pose": [
                    profile_state.x,
                    profile_state.y,
                    profile_state.psi,
                ],
                "coverage_footprint": [
                    profile.coverage_length,
                    profile.coverage_width,
                ],
                "vehicle_footprint": [
                    profile.vehicle_length,
                    profile.vehicle_width,
                ],
                "min_turn_radius": profile.min_turn_radius,
                "turn_power": profile.turn_power,
                "turn_time_penalty_per_rad": profile.turn_time_penalty_per_rad,
                "turn_energy_penalty_per_rad": profile.turn_energy_penalty_per_rad,
                "turn_maneuver_time_penalty": profile.turn_maneuver_time_penalty,
                "turn_maneuver_energy_penalty": profile.turn_maneuver_energy_penalty,
            }
            for agent_id, profile_state in enumerate(config.fleet.initial_states_3dof)
            for profile in (config.profile_for_agent(agent_id),)
        },
        "fixed_region_count": int(plan.metadata.get("fixed_region_count", "0") or 0),
        "mode_count": int(plan.metadata.get("mode_count", "0") or 0),
        "makespan": makespan,
        "total_energy": _optional_float(plan.metadata.get("total_energy")) or 0.0,
        "total_turn_angle_rad": total_turn_angle,
        "total_turn_angle_deg": math.degrees(total_turn_angle),
        "turn_maneuver_count": total_turn_maneuvers,
        "lower_bound": lower_bound,
        "upper_bound": _optional_float(plan.metadata.get("upper_bound")) or makespan,
        "optimality_gap": _optional_float(plan.metadata.get("optimality_gap")) or 0.0,
        "root_lp_lower_bound": _optional_float(plan.metadata.get("root_lp_lower_bound")),
        "service_lower_bound": _optional_float(plan.metadata.get("service_lower_bound")),
        "sequential_baseline_makespan": baseline,
        "joint_gain_ratio": (
            baseline / makespan
            if baseline is not None and makespan > 0.0
            else None
        ),
        "joint_not_worse_than_sequential_baseline": (
            plan.metadata.get("joint_not_worse_than_sequential_baseline") == "true"
        ),
        "coverage_fraction": _optional_float(plan.metadata.get("coverage_fraction")),
        "coverage_target": _optional_float(plan.metadata.get("coverage_target")),
        "continuous_conflict_validated": (
            plan.metadata.get("continuous_conflict_validated") == "true"
        ),
        "generated_columns": int(plan.metadata.get("generated_columns", "0") or 0),
        "pricing_iterations": int(plan.metadata.get("pricing_iterations", "0") or 0),
        "branch_nodes": int(plan.metadata.get("branch_nodes", "0") or 0),
        "resource_precedence_branches": int(
            plan.metadata.get("resource_precedence_branches", "0") or 0
        ),
        "route_variable_branches": int(
            plan.metadata.get("route_variable_branches", "0") or 0
        ),
        "conflict_separation_rounds": int(
            plan.metadata.get("conflict_separation_rounds", "0") or 0
        ),
        "anytime_trace": json.loads(plan.metadata.get("anytime_trace_json", "[]")),
        "agents": {
            str(agent_id): {
                "segment_count": len(agent.segments),
                "metrics": {**dict(agent.metrics), **turn_metrics[agent_id]},
            }
            for agent_id, agent in plan.agents.items()
        },
        "artifacts": {},
    }
    if report["anytime_trace"]:
        solver_first = float(report["anytime_trace"][0]["time"])
        report["solver_first_feasible_sec"] = solver_first
        report["end_to_end_first_feasible_sec"] = (
            float(report["preparation_runtime_sec"] or 0.0) + solver_first
        )
    else:
        report["solver_first_feasible_sec"] = None
        report["end_to_end_first_feasible_sec"] = None
    report_path = output / "crown_mcpp_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report["artifacts"]["report"] = str(report_path)
    if render:
        route_path = _render_routes(plan, static_obstacles or (), output, map_id)
        trace_path = _render_anytime(report["anytime_trace"], output)
        report["artifacts"].update(
            {"routes": str(route_path), "anytime": str(trace_path)}
        )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plan.metadata["crown_output_dir"] = str(output)
    plan.metadata["crown_report"] = str(report_path)
    return plan, report


def _peak_rss_mb() -> float:
    """Return the process peak resident set size using platform units."""

    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _optional_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _turn_metrics(segments) -> Dict[str, float | int]:
    """Return sampling-invariant heading variation and maneuver count.

    CROWN materializes short primitives, so counting waypoint heading changes
    would make the result depend on primitive duration.  We instead aggregate
    every primitive back to its source geometric segment and count one
    maneuver when that source changes heading by more than five degrees.
    """

    total_angle = 0.0
    angle_by_source: Dict[str, float] = {}
    for ordinal, segment in enumerate(segments):
        if len(segment.waypoints) < 2 or segment.kind == "wait":
            continue
        segment_angle = sum(
            abs(wrap_angle(current.psi - previous.psi))
            for previous, current in zip(
                segment.waypoints[:-1], segment.waypoints[1:]
            )
        )
        total_angle += segment_angle
        source_id = str(
            segment.metadata.get("source_segment_id")
            or segment.segment_id
            or f"segment-{ordinal}"
        )
        angle_by_source[source_id] = angle_by_source.get(source_id, 0.0) + segment_angle
    threshold = math.radians(5.0)
    return {
        "total_turn_angle_rad": total_angle,
        "total_turn_angle_deg": math.degrees(total_angle),
        "turn_maneuver_count": sum(
            angle > threshold for angle in angle_by_source.values()
        ),
    }
def _render_routes(
    plan: MultiAgentPathPlan,
    obstacles: Sequence[StaticObstacle],
    output: Path,
    map_id: str,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    figure, axis = plt.subplots(figsize=(9, 7))
    for obstacle in obstacles:
        if obstacle.polygon:
            axis.add_patch(
                Polygon(obstacle.polygon, closed=True, color="0.25", alpha=0.65)
            )
        elif obstacle.center is not None and obstacle.radius is not None:
            axis.add_patch(
                Circle(obstacle.center, obstacle.radius, color="0.25", alpha=0.65)
            )
    for agent_id, agent in sorted(plan.agents.items()):
        color = f"C{agent_id % 10}"
        labelled = False
        for segment in agent.segments:
            if len(segment.waypoints) < 2:
                continue
            axis.plot(
                [point.x for point in segment.waypoints],
                [point.y for point in segment.waypoints],
                color=color,
                linewidth=2.0 if segment.kind == "cover" else 1.1,
                alpha=0.9,
                label=f"agent {agent_id}" if not labelled else None,
            )
            labelled = True
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(f"CROWN-MCPP routes{': ' + map_id if map_id else ''}")
    if plan.agents:
        axis.legend(loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output / "crown_mcpp_routes.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _render_anytime(trace: Sequence[Dict[str, float]], output: Path) -> Path:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    if trace:
        times = [float(item.get("time", 0.0)) for item in trace]
        lower = [float(item.get("lower_bound", 0.0)) for item in trace]
        upper = [float(item.get("upper_bound", 0.0)) for item in trace]
        axis.step(times, lower, where="post", label="LB")
        axis.step(times, upper, where="post", label="UB")
    axis.set_xlabel("wall time (s)")
    axis.set_ylabel("makespan")
    axis.set_title("CROWN-MCPP anytime certificate")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    path = output / "crown_mcpp_anytime.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path
