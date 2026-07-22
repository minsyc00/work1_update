from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from ..schema import PlannerConfig, Pose2D
from .types import (
    AlgorithmExperimentTrace,
    MultiAgentPathPlan,
    ObstacleField,
    PathPlanningTrace,
    PathSegmentSpec,
    StaticObstacle,
)


AGENT_COLORS = ["#0b5fff", "#f2c94c", "#00a676", "#c13bff", "#d7263d", "#2e4057", "#f4a261", "#0081a7"]
OBSTACLE_COLOR = "#e63946"
INFLATED_COLOR = "#9d0208"
RESIDUAL_COLOR = "#8a2be2"
ASTAR_COLOR = "#ff9f1c"


def render_path_planning_visual_diagnostics(
    config: PlannerConfig,
    static_obstacles: Sequence[StaticObstacle],
    path_plan: MultiAgentPathPlan,
    trace: PathPlanningTrace,
    output_dir: str | Path,
    dpi: int = 180,
    gif_fps: int = 6,
) -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "map_id": trace.map_id or "",
        "algorithm_name": path_plan.algorithm_name,
        "metrics": dict(path_plan.metadata),
        "artifacts": [],
    }

    def save(fig, filename: str, stage: str) -> None:
        target = output / filename
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        manifest["artifacts"].append({"stage": stage, "file": filename, "bytes": target.stat().st_size})

    fig, ax = _new_map_axes(config, "00 Initial Map")
    _draw_obstacles(ax, trace.obstacle_field, raw=True, inflated=False)
    _draw_initial_states(ax, config)
    save(fig, "00_initial_map.png", "initial_map")

    fig, ax = _new_map_axes(config, "01 Inflated Static Obstacles")
    _draw_obstacles(ax, trace.obstacle_field, raw=True, inflated=True)
    _draw_initial_states(ax, config)
    save(fig, "01_inflated_obstacles.png", "inflated_obstacles")

    fig, ax = _new_map_axes(config, "02 Free-Space Decomposition")
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    _draw_regions(ax, trace.regions_before_filter or trace.regions, facecolor="#bde0fe", edgecolor="#457b9d", alpha=0.28, label_ids=True)
    save(fig, "02_free_space_decomposition.png", "free_space_decomposition")

    fig, ax = _new_map_axes(config, "03 Region Graph")
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    _draw_region_graph(ax, trace)
    save(fig, "03_region_graph.png", "region_graph")

    fig, ax = _new_map_axes(config, "04 Multi-USV Assignment")
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    _draw_assignment(ax, trace)
    save(fig, "04_multi_usv_assignment.png", "multi_usv_assignment")

    for agent_id in sorted(trace.tours):
        fig, ax = _new_map_axes(config, f"05 USV {agent_id} TSP-CPP Region Route")
        _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
        _draw_assignment_regions_for_agent(ax, trace, agent_id)
        _draw_tsp_route(ax, trace, agent_id)
        save(fig, f"05_agent_{agent_id}_tsp_route.png", f"agent_{agent_id}_tsp_route")

    for agent_id in sorted(trace.tours):
        fig, ax = _new_map_axes(config, f"06 USV {agent_id} Coverage Passes")
        _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
        _draw_agent_coverage_passes(ax, trace, agent_id)
        _draw_agent_segments(ax, path_plan, agent_id, only_agent_color=True, alpha=0.45)
        save(fig, f"06_agent_{agent_id}_coverage_passes.png", f"agent_{agent_id}_coverage_passes")

    fig, ax = _new_map_axes(config, "07 Obstacle-Aware Connections")
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    _draw_connections(ax, path_plan)
    save(fig, "07_obstacle_aware_connections.png", "obstacle_aware_connections")

    fig, ax = _new_schedule_axes(path_plan)
    _draw_resource_schedule(ax, path_plan)
    save(fig, "08_mapf_resource_schedule.png", "mapf_resource_schedule")

    fig, ax = _new_map_axes(config, "09 Residual Backfill")
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    _draw_residuals(ax, trace)
    _draw_residual_segments(ax, path_plan)
    save(fig, "09_residual_backfill.png", "residual_backfill")

    fig, ax = _new_map_axes(config, "10 Final Multi-USV Path Plan")
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    _draw_all_agent_segments(ax, path_plan)
    save(fig, "10_final_multi_usv_path_plan.png", "final_multi_usv_path_plan")

    fig, ax = _new_map_axes(config, "11 Coverage Heatmap")
    _draw_coverage_heatmap(ax, config, trace)
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    save(fig, "11_coverage_heatmap.png", "coverage_heatmap")

    gif_path = output / "route_monitor.gif"
    _render_route_monitor_gif(config, trace, path_plan, gif_path, fps=gif_fps)
    manifest["artifacts"].append({"stage": "route_monitor", "file": "route_monitor.gif", "bytes": gif_path.stat().st_size})

    manifest_path = output / "visualization_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    return {"output_dir": str(output), "manifest": str(manifest_path)}


def render_algorithm_experiment(
    config: PlannerConfig,
    static_obstacles: Sequence[StaticObstacle],
    path_plan: MultiAgentPathPlan,
    trace: AlgorithmExperimentTrace,
    output_dir: str | Path,
    dpi: int = 180,
    gif_fps: int = 6,
) -> Dict[str, str]:
    base = Path(output_dir)
    output = base if base.name == "algorithm_steps" else base / "algorithm_steps"
    output.mkdir(parents=True, exist_ok=True)
    artifacts: List[Dict[str, str | int]] = []

    def save(fig, filename: str, stage: str) -> None:
        target = output / filename
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        artifacts.append({"stage": stage, "file": filename, "bytes": target.stat().st_size})

    fig, ax = _new_map_axes(config, "00 Map Loaded: Static Obstacles + USV Starts")
    _draw_obstacles(ax, trace.obstacle_field, raw=True, inflated=False)
    _draw_initial_states(ax, config)
    _stage_box(ax, trace, "map_loaded", "Map JSON + fleet config -> PlannerConfig + StaticObstacle[]")
    save(fig, "00_map_and_static_obstacles.png", "map_loaded")

    fig, ax = _new_map_axes(config, "01 Obstacle Inflation")
    _draw_obstacles(ax, trace.obstacle_field, raw=True, inflated=True)
    _stage_box(ax, trace, "obstacle_inflation", "inflation = d_safe + max(lf,wf)/2 + extra")
    save(fig, "01_obstacle_inflation.png", "obstacle_inflation")

    fig, ax = _new_map_axes(config, "02 Sweep Lines and Free-Space Cells")
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    _draw_sweep_lines(ax, trace.regions_before_filter)
    _draw_regions(ax, trace.regions_before_filter, facecolor="#bde0fe", edgecolor="#457b9d", alpha=0.18, label_ids=True)
    _stage_box(ax, trace, "free_space_decomposition", "Obstacle vertices + grid breaks -> free-space cells")
    save(fig, "02_sweep_lines_and_free_cells.png", "free_space_decomposition")

    fig, ax = _new_map_axes(config, "03 Valid Decomposition Cells")
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    _draw_valid_and_filtered_cells(ax, trace)
    _stage_box(ax, trace, "free_space_decomposition", "Valid cells retain coverage-feasible free space")
    save(fig, "03_decomposition_valid_cells.png", "decomposition_valid_cells")

    fig, ax = _new_map_axes(config, "04 Candidate Coverage Patterns")
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    _draw_candidate_patterns(ax, trace)
    _stage_box(ax, trace, "coverage_pattern_generation", "Delta = wf * (1-rho); candidates keep entry/exit pose")
    save(fig, "04_candidate_coverage_patterns.png", "coverage_pattern_generation")

    fig, ax = _new_map_axes(config, "05 Region Graph and Weights")
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    _draw_region_graph(ax, trace)  # duck-typed with PathPlanningTrace
    _stage_box(ax, trace, "region_graph_building", "W(region)=min pattern time; edges use Dubins + collision cost")
    save(fig, "05_region_graph_weights.png", "region_graph_building")

    fig, ax = _new_map_axes(config, "06 Balanced Multi-USV Assignment")
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    _draw_assignment(ax, trace)
    _stage_box(ax, trace, "load_balancing_assignment", "min max_i W_i + imbalance penalty")
    save(fig, "06_balanced_assignment.png", "load_balancing_assignment")

    for agent_id in sorted(trace.tsp_records):
        fig, ax = _new_map_axes(config, f"07 USV {agent_id} Initial TSP-CPP Order")
        _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
        _draw_order(ax, trace, agent_id, trace.tsp_records[agent_id].get("initial_order", []), title_prefix="initial")
        _stage_box(ax, trace, "single_usv_tsp_initial_solution", "Turn-aware A* seeded region order")
        save(fig, f"07_agent_{agent_id}_tsp_initial_order.png", f"agent_{agent_id}_tsp_initial_order")

    for agent_id in sorted(trace.tsp_records):
        fig, ax = _new_map_axes(config, f"08 USV {agent_id} Pattern Selection")
        _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
        _draw_pattern_selection(ax, trace, agent_id)
        _stage_box(ax, trace, "pattern_selection", "C = C_inside + C_connect + C_lookahead")
        save(fig, f"08_agent_{agent_id}_pattern_selection.png", f"agent_{agent_id}_pattern_selection")

    for agent_id in sorted(trace.tsp_records):
        fig = _draw_2opt_iterations(config, trace, agent_id)
        save(fig, f"09_agent_{agent_id}_2opt_iterations.png", f"agent_{agent_id}_2opt_iterations")

    fig, ax = _new_map_axes(config, "10 Obstacle-Aware Connections")
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    _draw_connections(ax, path_plan)
    _stage_box(ax, trace, "obstacle_aware_connection", "Dubins first; if colliding, use grid A* corridor -> Dubins/Bezier subsegments")
    save(fig, "10_obstacle_aware_connections.png", "obstacle_aware_connection")

    fig, ax = _new_map_axes(config, "11 Final Single-USV TSP-CPP Tours")
    _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
    _draw_all_agent_segments(ax, path_plan)
    _stage_box(ax, trace, "final_tsp_cpp_tour", "Final cover/transit/turn/residual segments after scheduling")
    save(fig, "11_final_single_usv_tsp_cpp_tours.png", "final_tsp_cpp_tour")

    gif_path = output / "12_algorithm_process.gif"
    _render_algorithm_process_gif(output, gif_path, fps=gif_fps)
    artifacts.append({"stage": "algorithm_process", "file": "12_algorithm_process.gif", "bytes": gif_path.stat().st_size})

    report_path = output / "algorithm_experiment_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(_algorithm_experiment_report(trace, path_plan, artifacts), file, indent=2, ensure_ascii=False)
    return {"output_dir": str(output), "report": str(report_path)}


def _stage_box(ax, trace: AlgorithmExperimentTrace, stage: str, formula: str) -> None:
    metrics = trace.stage_metrics.get(stage, {})
    metric_lines = [f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}" for key, value in list(metrics.items())[:5]]
    text = f"{stage}\n{formula}"
    if metric_lines:
        text += "\n" + "\n".join(metric_lines)
    _info_box_outside(ax, text, fontsize=7)


def _draw_sweep_lines(ax, regions) -> None:
    x_edges = sorted({round(value, 6) for region in regions for value in (region.bounds[0], region.bounds[2])})
    y_edges = sorted({round(value, 6) for region in regions for value in (region.bounds[1], region.bounds[3])})
    for x_value in x_edges:
        ax.axvline(x_value, color="#90a4ae", linewidth=0.35, alpha=0.45)
    for y_value in y_edges:
        ax.axhline(y_value, color="#90a4ae", linewidth=0.35, alpha=0.45)


def _draw_valid_and_filtered_cells(ax, trace: AlgorithmExperimentTrace) -> None:
    valid_ids = {region.region_id for region in trace.regions}
    for region in trace.regions_before_filter:
        is_valid = region.region_id in valid_ids
        color = "#2a9d8f" if is_valid else "#adb5bd"
        alpha = 0.25 if is_valid else 0.14
        _draw_polygon(ax, region.polygon, facecolor=color, edgecolor=color, alpha=alpha, linewidth=0.8)
        narrow = region.metadata.get("narrow_width", "")
        label = _short_region_label(region.region_id)
        if narrow:
            label += f"\nw={float(narrow):.1f}"
        ax.text(region.center[0], region.center[1], label, fontsize=4.8, ha="center", va="center", color="#1d3557")


def _draw_candidate_patterns(ax, trace: AlgorithmExperimentTrace) -> None:
    for region in trace.regions:
        _draw_polygon(ax, region.polygon, facecolor="#f1faee", edgecolor="#a8dadc", alpha=0.12, linewidth=0.4)
    for idx, (region_id, patterns) in enumerate(sorted(trace.patterns.items())):
        for pattern_idx, pattern in enumerate(patterns):
            if pattern.scan_axis.startswith("theta:"):
                color = "#7b2cbf"
            else:
                color = "#0b5fff" if pattern.scan_axis == "x" else "#ff6b00"
            alpha = 0.28 if pattern_idx > 0 else 0.55
            for coverage_pass in pattern.passes:
                ax.plot(
                    [coverage_pass.start_pose.x, coverage_pass.end_pose.x],
                    [coverage_pass.start_pose.y, coverage_pass.end_pose.y],
                    color=color,
                    linewidth=0.65,
                    alpha=alpha,
                )
            if idx % 3 == 0 and pattern.passes:
                label = pattern.scan_axis
                if pattern.scan_axis.startswith("theta:"):
                    label = f"theta={pattern.metadata.get('scan_angle_deg', '?')}"
                ax.text(pattern.entry_pose.x, pattern.entry_pose.y, label, fontsize=5, color=color)


def _draw_order(ax, trace: AlgorithmExperimentTrace, agent_id: int, order: Sequence[str], title_prefix: str) -> None:
    if trace.graph is None:
        return
    color = _agent_color(agent_id)
    centers = []
    for seq, region_id in enumerate(order):
        region = trace.graph.regions.get(region_id)
        if region is None:
            continue
        centers.append(region.center)
        _draw_polygon(ax, region.polygon, facecolor=color, edgecolor=color, alpha=0.18, linewidth=0.8)
        ax.text(region.center[0], region.center[1], str(seq + 1), fontsize=7, ha="center", va="center", color=color)
    for start, end in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.0})
    ax.plot([], [], color=color, label=f"USV {agent_id} {title_prefix} order")
    ax.legend(loc="upper right", fontsize=8)


def _draw_pattern_selection(ax, trace: AlgorithmExperimentTrace, agent_id: int) -> None:
    if trace.graph is None:
        return
    record = trace.tsp_records.get(agent_id, {})
    color = _agent_color(agent_id)
    for item in record.get("pattern_selection", []):
        region = trace.graph.regions.get(item.get("region_id", ""))
        selected = item.get("selected", {})
        if region is None or not selected:
            continue
        _draw_polygon(ax, region.polygon, facecolor=color, edgecolor=color, alpha=0.16, linewidth=0.8)
        entry = selected.get("entry", [region.center[0], region.center[1], 0.0])
        exit_pose = selected.get("exit", [region.center[0], region.center[1], 0.0])
        ax.plot(entry[0], entry[1], marker=">", color=color, markersize=4)
        ax.plot(exit_pose[0], exit_pose[1], marker="s", color=color, markersize=3)
        ax.text(
            region.center[0],
            region.center[1],
            f"{selected.get('scan_axis','?')}\nC={selected.get('total_cost',0.0):.1f}",
            fontsize=5.5,
            ha="center",
            va="center",
            color="#111111",
        )


def _draw_2opt_iterations(config: PlannerConfig, trace: AlgorithmExperimentTrace, agent_id: int):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    record = trace.tsp_records.get(agent_id, {})
    initial = record.get("initial_order", [])
    final = record.get("final_order", initial)
    improvements = record.get("two_opt_improvements", [])
    for ax, title, order in zip(axes, ("Initial TSP-CPP order", "After 2-opt/3-opt"), (initial, final)):
        _setup_existing_map_axes(ax, config, f"09 USV {agent_id} {title}")
        _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
        _draw_order(ax, trace, agent_id, order, title_prefix=title.lower())
    before_obj = record.get("initial_metrics", {}).get("objective", 0.0)
    after_obj = record.get("final_metrics", {}).get("objective", 0.0)
    _info_box_outside(
        axes[1],
        f"accepted 2-opt={len(improvements)}\nobjective {before_obj:.2f} -> {after_obj:.2f}\ndelta={after_obj-before_obj:.2f}",
        fontsize=8,
    )
    return fig


def _render_algorithm_process_gif(output_dir: Path, output_path: Path, fps: int) -> None:
    frames = [
        ("00_map_and_static_obstacles.png", "Map loaded"),
        ("01_obstacle_inflation.png", "Obstacle inflation"),
        ("02_sweep_lines_and_free_cells.png", "Sweep decomposition"),
        ("04_candidate_coverage_patterns.png", "Candidate coverage patterns"),
        ("05_region_graph_weights.png", "Region graph weights"),
        ("06_balanced_assignment.png", "Balanced assignment"),
        ("10_obstacle_aware_connections.png", "Obstacle-aware connections"),
        ("11_final_single_usv_tsp_cpp_tours.png", "Final TSP-CPP tours"),
    ]
    existing = [(output_dir / filename, title) for filename, title in frames if (output_dir / filename).exists()]
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(index: int):
        path, title = existing[index]
        ax.clear()
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(f"{index + 1}/{len(existing)} {title}", fontsize=12)
        ax.axis("off")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(existing), interval=max(30, int(1000 / max(fps, 1))), blit=False)
    anim.save(output_path, writer=animation.PillowWriter(fps=max(fps, 1)))
    plt.close(fig)


def _algorithm_experiment_report(trace: AlgorithmExperimentTrace, path_plan: MultiAgentPathPlan, artifacts: List[Dict[str, str | int]]) -> Dict[str, object]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "map_id": trace.map_id,
        "output_dir": trace.output_dir,
        "path_plan_metadata": dict(path_plan.metadata),
        "stage_metrics": trace.stage_metrics,
        "tsp_records": trace.tsp_records,
        "agents": {
            str(agent_id): {
                "segment_count": len(agent.segments),
                "metrics": agent.metrics,
            }
            for agent_id, agent in path_plan.agents.items()
        },
        "artifacts": artifacts,
    }


def _new_map_axes(config: PlannerConfig, title: str):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(0.0, config.mission.area_length_x)
    ax.set_ylim(0.0, config.mission.area_length_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.add_patch(
        patches.Rectangle(
            (0.0, 0.0),
            config.mission.area_length_x,
            config.mission.area_length_y,
            fill=False,
            edgecolor="#111111",
            linewidth=1.5,
        )
    )
    return fig, ax


def _new_schedule_axes(path_plan: MultiAgentPathPlan):
    resources = _resource_windows(path_plan)
    height = min(max(5.0, 0.22 * max(len(resources), 1)), 18.0)
    fig, ax = plt.subplots(figsize=(12, height))
    ax.set_title("08 MAPF/CBS-Style Resource Window Schedule")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("resource")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)
    return fig, ax


def _draw_initial_states(ax, config: PlannerConfig) -> None:
    for agent_id, state in enumerate(config.fleet.initial_states_3dof):
        color = _agent_color(agent_id)
        ax.plot(state.x, state.y, marker="o", color=color, markersize=5)
        ax.arrow(state.x, state.y, 1.2 * math.cos(state.psi), 1.2 * math.sin(state.psi), color=color, width=0.04, alpha=0.9)
        ax.text(state.x + 0.4, state.y + 0.4, f"USV {agent_id}", fontsize=7, color=color)


def _draw_obstacles(ax, field: ObstacleField | None, raw: bool, inflated: bool) -> None:
    if field is None:
        return
    if raw:
        for obstacle in field.obstacles:
            _draw_polygon(ax, obstacle.polygon, facecolor=OBSTACLE_COLOR, edgecolor=OBSTACLE_COLOR, alpha=0.22, linewidth=1.2)
            _label_polygon(ax, obstacle.polygon, obstacle.obstacle_id, OBSTACLE_COLOR)
    if inflated:
        for obstacle in field.inflated_obstacles:
            _draw_polygon(ax, obstacle.polygon, facecolor="none", edgecolor=INFLATED_COLOR, alpha=1.0, linewidth=1.1, linestyle="--")


def _draw_regions(ax, regions, facecolor: str, edgecolor: str, alpha: float, label_ids: bool = False) -> None:
    for region in regions:
        _draw_polygon(ax, region.polygon, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=0.7)
        if label_ids:
            ax.text(region.center[0], region.center[1], _short_region_label(region.region_id), fontsize=4.5, ha="center", va="center", color="#1d3557")


def _draw_region_graph(ax, trace: PathPlanningTrace) -> None:
    graph = trace.graph
    if graph is None:
        return
    _draw_regions(ax, graph.regions.values(), facecolor="#f1faee", edgecolor="#a8dadc", alpha=0.18)
    drawn = set()
    for region_id, neighbors in graph.adjacency.items():
        region = graph.regions.get(region_id)
        if region is None:
            continue
        for neighbor_id in neighbors:
            key = tuple(sorted((region_id, neighbor_id)))
            if key in drawn:
                continue
            drawn.add(key)
            neighbor = graph.regions.get(neighbor_id)
            if neighbor is None:
                continue
            ax.plot(
                [region.center[0], neighbor.center[0]],
                [region.center[1], neighbor.center[1]],
                color="#6c757d",
                linewidth=0.5,
                alpha=0.5,
            )
    weights = list(graph.node_weights.values()) or [0.0]
    max_weight = max(max(weights), 1e-9)
    for region_id, region in graph.regions.items():
        weight = graph.node_weights.get(region_id, 0.0) / max_weight
        ax.scatter(region.center[0], region.center[1], s=18 + 28 * weight, color="#1d3557", alpha=0.8)


def _draw_assignment(ax, trace: PathPlanningTrace) -> None:
    if trace.assignment is None or trace.graph is None:
        return
    for agent_id, region_ids in sorted(trace.assignment.agent_regions.items()):
        color = _agent_color(agent_id)
        for region_id in region_ids:
            region = trace.graph.regions.get(region_id)
            if region is not None:
                _draw_polygon(ax, region.polygon, facecolor=color, edgecolor=color, alpha=0.22, linewidth=0.9)
        load = trace.assignment.loads.get(agent_id, 0.0)
        ax.plot([], [], color=color, linewidth=5, label=f"USV {agent_id}: load={load:.1f}")
    ax.legend(loc="upper right", fontsize=8)


def _draw_assignment_regions_for_agent(ax, trace: PathPlanningTrace, agent_id: int) -> None:
    if trace.assignment is None or trace.graph is None:
        return
    color = _agent_color(agent_id)
    for region_id in trace.assignment.agent_regions.get(agent_id, []):
        region = trace.graph.regions.get(region_id)
        if region is not None:
            _draw_polygon(ax, region.polygon, facecolor=color, edgecolor=color, alpha=0.20, linewidth=0.8)


def _draw_tsp_route(ax, trace: PathPlanningTrace, agent_id: int) -> None:
    if trace.graph is None:
        return
    tour = trace.tours.get(agent_id)
    if tour is None:
        return
    color = _agent_color(agent_id)
    centers = []
    for seq, region_id in enumerate(tour.region_order):
        region = trace.graph.regions.get(region_id)
        if region is None:
            continue
        centers.append(region.center)
        ax.text(region.center[0], region.center[1], str(seq + 1), fontsize=7, color=color, ha="center", va="center")
    for start, end in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.0, "alpha": 0.9})
    if centers:
        ax.plot([pt[0] for pt in centers], [pt[1] for pt in centers], color=color, linewidth=1.0, alpha=0.7, label=f"USV {agent_id} TSP order")
        ax.legend(loc="upper right", fontsize=8)


def _draw_agent_coverage_passes(ax, trace: PathPlanningTrace, agent_id: int) -> None:
    tour = trace.tours.get(agent_id)
    if tour is None:
        return
    color = _agent_color(agent_id)
    for pattern in tour.selected_patterns.values():
        for coverage_pass in pattern.passes:
            ax.plot(
                [coverage_pass.start_pose.x, coverage_pass.end_pose.x],
                [coverage_pass.start_pose.y, coverage_pass.end_pose.y],
                color=color,
                linewidth=1.5,
                alpha=0.9,
            )
            ax.plot(coverage_pass.start_pose.x, coverage_pass.start_pose.y, marker=".", color=color, markersize=3)


def _draw_connections(ax, path_plan: MultiAgentPathPlan) -> None:
    for agent_id, agent in sorted(path_plan.agents.items()):
        for segment in agent.segments:
            if segment.kind == "cover":
                continue
            connector = segment.metadata.get("connector", "")
            invalid = bool(segment.metadata.get("invalid_reasons")) or segment.metadata.get("kinematic_feasible") == "false"
            color = "#d00000" if invalid else (ASTAR_COLOR if connector == "astar_corridor" else _agent_color(agent_id))
            linewidth = 3.0 if invalid else (2.4 if connector == "astar_corridor" else 1.0)
            linestyle = "-" if connector == "astar_corridor" else _style_for_segment(segment)
            _plot_segment(ax, segment, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.82)


def _draw_residuals(ax, trace: PathPlanningTrace) -> None:
    if trace.coverage_state is None:
        return
    for residual in trace.coverage_state.residual_components:
        x0, y0, x1, y1 = residual.bounds
        ax.add_patch(
            patches.Rectangle(
                (x0, y0),
                max(x1 - x0, trace.coverage_state.resolution),
                max(y1 - y0, trace.coverage_state.resolution),
                facecolor=RESIDUAL_COLOR,
                edgecolor=RESIDUAL_COLOR,
                alpha=0.22,
                linewidth=0.8,
            )
        )
        ax.text(residual.centroid[0], residual.centroid[1], f"R{residual.residual_id}", fontsize=7, color=RESIDUAL_COLOR)


def _draw_residual_segments(ax, path_plan: MultiAgentPathPlan) -> None:
    for agent_id, agent in sorted(path_plan.agents.items()):
        for segment in agent.segments:
            resource = segment.metadata.get("resource_id", "")
            if "residual" in resource or "_residual" in segment.segment_id:
                _plot_segment(ax, segment, color=RESIDUAL_COLOR, linestyle="-", linewidth=1.8, alpha=0.9)


def _draw_all_agent_segments(ax, path_plan: MultiAgentPathPlan) -> None:
    for agent_id in sorted(path_plan.agents):
        _draw_agent_segments(ax, path_plan, agent_id, only_agent_color=False, alpha=0.88)
    ax.legend(loc="upper right", fontsize=8)


def _draw_agent_segments(ax, path_plan: MultiAgentPathPlan, agent_id: int, only_agent_color: bool, alpha: float) -> None:
    agent = path_plan.agents.get(agent_id)
    if agent is None:
        return
    color = _agent_color(agent_id)
    labeled = False
    for segment in agent.segments:
        invalid = bool(segment.metadata.get("invalid_reasons")) or segment.metadata.get("kinematic_feasible") == "false"
        segment_color = color
        if invalid and not only_agent_color:
            segment_color = "#d00000"
        label = f"USV {agent_id}" if not labeled else None
        width = max(_width_for_segment(segment), 2.8) if invalid and not only_agent_color else _width_for_segment(segment)
        _plot_segment(ax, segment, color=segment_color, linestyle=_style_for_segment(segment), linewidth=width, alpha=alpha, label=label)
        labeled = True


def _draw_resource_schedule(ax, path_plan: MultiAgentPathPlan) -> None:
    resources = _resource_windows(path_plan)
    if not resources:
        ax.text(0.5, 0.5, "No timed resources", transform=ax.transAxes, ha="center", va="center")
        return
    sorted_resources = sorted(resources, key=lambda item: item[1][0][0])[:80]
    y_labels = []
    for row, (resource_id, windows) in enumerate(sorted_resources):
        y_labels.append(_short_resource_label(resource_id))
        for start, end, agent_id, kind in windows:
            ax.barh(row, max(end - start, 0.05), left=start, height=0.72, color=_agent_color(agent_id), alpha=0.75)
            ax.text(start, row, kind[:1], fontsize=6, va="center", ha="left", color="white")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=6)
    if len(resources) > len(sorted_resources):
        ax.set_title(f"08 MAPF/CBS-Style Resource Window Schedule (first {len(sorted_resources)} of {len(resources)} resources)")


def _draw_coverage_heatmap(ax, config: PlannerConfig, trace: PathPlanningTrace) -> None:
    if trace.coverage_state is None:
        return
    state = trace.coverage_state
    ax.imshow(
        state.coverage_ratio.astype(float),
        origin="lower",
        extent=[0, config.mission.area_length_x, 0, config.mission.area_length_y],
        cmap="Blues",
        alpha=0.55,
        vmin=0.0,
        vmax=1.0,
    )
    _draw_residuals(ax, trace)


def _render_route_monitor_gif(
    config: PlannerConfig,
    trace: PathPlanningTrace,
    path_plan: MultiAgentPathPlan,
    output_path: Path,
    fps: int,
) -> None:
    segments = _timed_segments(path_plan)
    step = max(1, int(math.ceil(max(len(segments), 1) / 140)))
    frame_indices = list(range(0, len(segments) + 1, step))
    if frame_indices[-1] != len(segments):
        frame_indices.append(len(segments))

    fig, ax = _new_map_axes(config, "Route Monitor")

    def update(frame_number: int):
        upto = frame_indices[frame_number]
        ax.clear()
        _setup_existing_map_axes(ax, config, "Route Monitor")
        _draw_obstacles(ax, trace.obstacle_field, raw=False, inflated=True)
        length = 0.0
        for _, agent_id, segment in segments[:upto]:
            length += segment.length
            _plot_segment(ax, segment, color=_agent_color(agent_id), linestyle=_style_for_segment(segment), linewidth=_width_for_segment(segment), alpha=0.85)
        if upto > 0:
            _, agent_id, current = segments[upto - 1]
            _plot_segment(ax, current, color="#111111", linestyle="-", linewidth=2.7, alpha=0.95)
            segment_kind = current.kind
        else:
            agent_id = -1
            segment_kind = "start"
        coverage = path_plan.metadata.get("coverage_fraction", "0")
        _info_box_outside(
            ax,
            f"segment {upto}/{len(segments)}\nagent {agent_id}\nkind {segment_kind}\ncoverage {coverage}\nlength {length:.1f} m",
            fontsize=8,
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=max(30, int(1000 / max(fps, 1))), blit=False)
    anim.save(output_path, writer=animation.PillowWriter(fps=max(fps, 1)))
    plt.close(fig)


def _setup_existing_map_axes(ax, config: PlannerConfig, title: str) -> None:
    ax.set_xlim(0.0, config.mission.area_length_x)
    ax.set_ylim(0.0, config.mission.area_length_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.add_patch(patches.Rectangle((0.0, 0.0), config.mission.area_length_x, config.mission.area_length_y, fill=False, edgecolor="#111111", linewidth=1.4))


def _info_box_outside(ax, text: str, fontsize: float = 8) -> None:
    ax.figure.subplots_adjust(right=0.76)
    ax.text(
        1.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        clip_on=False,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc"},
    )


def _resource_windows(path_plan: MultiAgentPathPlan) -> List[Tuple[str, List[Tuple[float, float, int, str]]]]:
    grouped: Dict[str, List[Tuple[float, float, int, str]]] = {}
    for agent_id, agent in path_plan.agents.items():
        for segment in agent.segments:
            resource = segment.metadata.get("resource_id")
            if not resource:
                continue
            start, end = _segment_time_bounds(segment)
            grouped.setdefault(resource, []).append((start, end, agent_id, segment.kind))
    return [(resource, sorted(windows)) for resource, windows in grouped.items()]


def _timed_segments(path_plan: MultiAgentPathPlan) -> List[Tuple[float, int, PathSegmentSpec]]:
    items = []
    for agent_id, agent in path_plan.agents.items():
        for segment in agent.segments:
            items.append((_segment_time_bounds(segment)[0], agent_id, segment))
    return sorted(items, key=lambda item: (item[0], item[1], item[2].segment_id))


def _segment_time_bounds(segment: PathSegmentSpec) -> Tuple[float, float]:
    times = [float(waypoint.time) for waypoint in segment.waypoints if waypoint.time is not None]
    if not times:
        return (0.0, 0.0)
    return (min(times), max(times))


def _plot_segment(
    ax,
    segment: PathSegmentSpec,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    label: str | None = None,
) -> None:
    points = [(waypoint.x, waypoint.y) for waypoint in segment.waypoints]
    if len(points) < 2:
        return
    ax.plot([pt[0] for pt in points], [pt[1] for pt in points], color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)


def _draw_polygon(ax, polygon: Sequence[Tuple[float, float]], facecolor: str, edgecolor: str, alpha: float, linewidth: float, linestyle: str = "-") -> None:
    if len(polygon) < 3:
        return
    ax.add_patch(patches.Polygon(list(polygon), closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth, linestyle=linestyle))


def _label_polygon(ax, polygon: Sequence[Tuple[float, float]], label: str, color: str) -> None:
    if not polygon:
        return
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    ax.text(float(np.mean(xs)), float(np.mean(ys)), label, fontsize=6, color=color, ha="center", va="center")


def _style_for_segment(segment: PathSegmentSpec) -> str:
    if "residual" in segment.metadata.get("resource_id", ""):
        return "-"
    if segment.kind == "cover":
        return "-"
    if segment.kind == "turn":
        return "-."
    return "--"


def _width_for_segment(segment: PathSegmentSpec) -> float:
    if segment.metadata.get("connector") == "astar_corridor":
        return 2.1
    if segment.kind == "cover":
        return 1.25
    return 1.0


def _agent_color(agent_id: int) -> str:
    return AGENT_COLORS[agent_id % len(AGENT_COLORS)]


def _short_region_label(region_id: str) -> str:
    return region_id.replace("free_cell_", "c").replace("region_", "r")


def _short_resource_label(resource_id: str) -> str:
    if len(resource_id) <= 28:
        return resource_id
    return resource_id[:25] + "..."
