from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..schema import CoverageFootprint, FleetConfig, MissionConfig, PlannerConfig, PlannerWeights, SafetyMargins
from .obstacles import circle_obstacle, ellipse_obstacle, polygon_obstacle, rectangle_obstacle
from .types import StaticObstacle


def load_map_json(path: str | Path) -> Dict[str, Any]:
    """Read a static map asset JSON file without injecting fleet state."""

    map_path = Path(path)
    with map_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"map JSON must contain an object: {map_path}")
    return data


def load_map_for_planner(
    map_json: str | Path,
    fleet_config: FleetConfig,
    weights: PlannerWeights | None = None,
    safety: SafetyMargins | None = None,
) -> Tuple[PlannerConfig, List[StaticObstacle]]:
    """Build a planner config plus static obstacles from a map asset.

    The map file intentionally does not contain USV initial states. Fleet state,
    speeds, thrust limits, and controller-facing parameters are supplied by the
    experiment config and combined here with the static map geometry.
    """

    data = load_map_json(map_json)
    mission_area = data.get("mission_area", {})
    if mission_area.get("type", "rectangle") != "rectangle":
        raise ValueError("only rectangle mission_area is supported by the current planner")
    origin = mission_area.get("origin", [0.0, 0.0])
    if len(origin) != 2 or abs(float(origin[0])) > 1e-9 or abs(float(origin[1])) > 1e-9:
        raise ValueError("planner currently expects map origin [0, 0]")

    notes = data.get("notes", {})
    footprint_data = data.get("coverage_footprint", {})
    motion_data = data.get("motion_constraints", {})
    overlap_ratio = float(notes.get("recommended_overlap_ratio", 0.1))
    d_safe = float(notes.get("recommended_d_safe", 1.0))
    min_turn_radius = float(motion_data.get("min_turn_radius", fleet_config.min_turn_radius))

    mission = MissionConfig(
        area_length_x=float(mission_area["length_x"]),
        area_length_y=float(mission_area["length_y"]),
        overlap_ratio=overlap_ratio,
    )
    footprint = CoverageFootprint(
        length_lf=float(footprint_data["length_lf"]),
        width_wf=float(footprint_data["width_wf"]),
    )
    fleet = replace(fleet_config, min_turn_radius=min_turn_radius)
    planner_config = PlannerConfig(
        mission=mission,
        fleet=fleet,
        footprint=footprint,
        weights=weights or PlannerWeights(),
        safety=safety or SafetyMargins(d_safe=d_safe),
    )
    obstacles = [_parse_static_obstacle(item) for item in data.get("static_obstacles", [])]
    return planner_config, obstacles


def build_experiment_output_dir(
    map_json: str | Path,
    config: PlannerConfig,
    outputs_root: str | Path = "outputs",
) -> Path:
    data = load_map_json(map_json)
    map_id = str(data.get("map_id") or Path(map_json).stem)
    footprint = config.footprint
    fleet = config.fleet
    suffix = (
        f"{map_id}_usv{fleet.num_agents or len(fleet.initial_states_3dof)}"
        f"_footprint{_compact_number(footprint.length_lf)}x{_compact_number(footprint.width_wf)}"
        f"_rmin{_compact_number(fleet.min_turn_radius)}"
    )
    return Path(outputs_root) / suffix


def _parse_static_obstacle(item: Dict[str, Any]) -> StaticObstacle:
    obstacle_id = str(item.get("obstacle_id") or item.get("id") or "static_obstacle")
    kind = str(item.get("type", "")).lower()
    yaw = math.radians(float(item.get("yaw_deg", 0.0)))
    if kind == "rectangle":
        center = _point(item["center"])
        width, height = item["size"]
        return rectangle_obstacle(obstacle_id, center=center, width=float(width), height=float(height), psi=yaw)
    if kind == "ellipse":
        center = _point(item["center"])
        rx, ry = item["radii"]
        return ellipse_obstacle(obstacle_id, center=center, radii=(float(rx), float(ry)), psi=yaw)
    if kind == "circle":
        return circle_obstacle(obstacle_id, center=_point(item["center"]), radius=float(item["radius"]))
    if kind == "polygon":
        return polygon_obstacle(obstacle_id, [_point(point) for point in item["vertices"]])
    raise ValueError(f"unsupported static obstacle type in map JSON: {kind}")


def _point(value: Any) -> Tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"point must contain two coordinates: {value!r}")
    return (float(value[0]), float(value[1]))


def _compact_number(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) <= 1e-9:
        return str(int(round(number)))
    return f"{number:.3f}".rstrip("0").rstrip(".").replace(".", "p")
