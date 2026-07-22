from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..schema import (
    AgentPlanningProfile,
    CoverageFootprint,
    FleetConfig,
    MissionConfig,
    PlannerConfig,
    PlannerWeights,
    SafetyMargins,
    State3DOF,
    State6DOF,
    VehicleFootprint,
)
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
    agent_profiles: Dict[int, AgentPlanningProfile] | None = None,
    fleet_profile_id: str = "",
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
    profile_map = dict(agent_profiles or {})
    vehicle_footprint = None
    if profile_map:
        vehicle_footprint = VehicleFootprint(
            length=max(profile.vehicle_length for profile in profile_map.values()),
            width=max(profile.vehicle_width for profile in profile_map.values()),
        )
    planner_config = PlannerConfig(
        mission=mission,
        fleet=fleet,
        footprint=footprint,
        weights=weights or PlannerWeights(),
        safety=safety or SafetyMargins(d_safe=d_safe),
        agent_profiles=profile_map,
        vehicle_footprint=vehicle_footprint,
        fleet_profile_id=fleet_profile_id,
    )
    planner_config.validate_agent_profiles()
    obstacles = [_parse_static_obstacle(item) for item in data.get("static_obstacles", [])]
    return planner_config, obstacles


def load_fleet_profile_json(path: str | Path) -> Tuple[FleetConfig, Dict[int, AgentPlanningProfile], str]:
    """Load an independent heterogeneous fleet description.

    Each agent owns its initial state, coverage footprint, physical hull, and
    motion limits.  The loader also accepts flat keys to keep hand-authored
    experiment files compact.
    """

    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"fleet profile JSON must contain an object: {profile_path}")
    agents = data.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("fleet profile must contain a non-empty agents list")

    profiles: Dict[int, AgentPlanningProfile] = {}
    states_3dof: List[State3DOF] = []
    for expected_id, item in enumerate(sorted(agents, key=lambda value: int(value.get("agent_id", 0)))):
        if not isinstance(item, dict):
            raise ValueError("each fleet profile agent must be an object")
        agent_id = int(item.get("agent_id", expected_id))
        if agent_id != expected_id:
            raise ValueError("fleet profile agent_id values must be contiguous from zero")
        initial = item.get("initial_state", {})
        coverage = item.get("coverage_footprint", {})
        vehicle = item.get("vehicle_footprint", {})
        motion = item.get("motion_constraints", {})
        psi = _heading_radians(initial)
        state = State3DOF(
            x=float(initial.get("x", item.get("initial_x", 0.0))),
            y=float(initial.get("y", item.get("initial_y", 0.0))),
            psi=psi,
        )
        states_3dof.append(state)
        profiles[agent_id] = AgentPlanningProfile(
            agent_id=agent_id,
            coverage_length=float(coverage.get("length_lf", item.get("coverage_length", 4.0))),
            coverage_width=float(coverage.get("width_wf", item.get("coverage_width", 2.0))),
            overlap_ratio=float(coverage.get("overlap_ratio", item.get("overlap_ratio", 0.1))),
            vehicle_length=float(vehicle.get("length", item.get("vehicle_length", coverage.get("length_lf", 4.0)))),
            vehicle_width=float(vehicle.get("width", item.get("vehicle_width", coverage.get("width_wf", 2.0)))),
            min_turn_radius=float(motion.get("min_turn_radius", item.get("min_turn_radius", 2.0))),
            cruise_speed=float(motion.get("cruise_speed", item.get("cruise_speed", 2.0))),
            cover_speed=float(motion.get("cover_speed", item.get("cover_speed", 1.2))),
            turn_speed_max=float(motion.get("turn_speed_max", item.get("turn_speed_max", 1.0))),
            max_thrust=float(motion.get("max_thrust", item.get("max_thrust", 2.0))),
            max_yaw_moment=float(motion.get("max_yaw_moment", item.get("max_yaw_moment", 1.0))),
            max_mission_time=_optional_float(motion.get("max_mission_time", item.get("max_mission_time"))),
        )

    first = profiles[0]
    fleet = FleetConfig(
        initial_states_3dof=states_3dof,
        initial_states_6dof=[State6DOF(x=state.x, y=state.y, psi=state.psi) for state in states_3dof],
        cruise_speed=first.cruise_speed,
        cover_speed=first.cover_speed,
        turn_speed_max=first.turn_speed_max,
        max_thrust=first.max_thrust,
        max_yaw_moment=first.max_yaw_moment,
        min_turn_radius=first.min_turn_radius,
        num_agents=len(states_3dof),
    )
    return fleet, profiles, str(data.get("fleet_profile_id") or profile_path.stem)


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
    if config.fleet_profile_id:
        suffix += f"_fleet-{_safe_identifier(config.fleet_profile_id)}"
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


def _heading_radians(initial: Dict[str, Any]) -> float:
    if "psi_rad" in initial:
        return float(initial["psi_rad"])
    if "psi_deg" in initial:
        return math.radians(float(initial["psi_deg"]))
    return float(initial.get("psi", 0.0))


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _compact_number(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) <= 1e-9:
        return str(int(round(number)))
    return f"{number:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _safe_identifier(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)
    return cleaned.strip("-") or "heterogeneous"
