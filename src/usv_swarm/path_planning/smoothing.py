from __future__ import annotations

import heapq
import math
from typing import Dict, List, Tuple

import numpy as np

from ..dubins import dubins_shortest_path, sample_dubins_path
from ..geometry import polyline_length, sample_quintic_bezier, straight_segment_points, unit_heading, wrap_angle
from ..schema import PlannerConfig, Pose2D
from .astar import obstacle_aware_grid_astar
from .dynamics_validation import retime_segment_for_dynamics, validate_transition_dynamics
from .obstacles import (
    path_segment_invalid_reasons,
    segment_collides_with_obstacles,
)
from .types import ObstacleField, PathPlanningConfig, PathSegmentSpec, PathWaypoint


def build_cover_segment(
    segment_id: str,
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    speed: float,
    sample_count: int = 12,
) -> PathSegmentSpec:
    points, headings = straight_segment_points(start, end, max(sample_count, 2))
    length = polyline_length(points)
    duration = length / max(speed, 1e-6)
    waypoints = _waypoints_from_points(points, headings, start_time, duration, speed)
    return PathSegmentSpec(
        segment_id=segment_id,
        kind="cover",
        source_algorithm="paper_fusion_planner",
        waypoints=waypoints,
        curvature_max=0.0,
        length=length,
        path_source="straight",
    )


def build_transition_segment(
    segment_id: str,
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    config: PlannerConfig,
    kind: str = "transit",
    sample_count: int = 24,
    use_bezier: bool = True,
) -> PathSegmentSpec:
    turn_radius = config.fleet.min_turn_radius
    dubins_path = dubins_shortest_path(start, end, turn_radius)
    speed = max(config.fleet.turn_speed_max if kind == "turn" else config.fleet.cruise_speed, 1e-6)
    if dubins_path.total_length <= 1e-9:
        return PathSegmentSpec(
            segment_id=segment_id,
            kind=kind,
            source_algorithm="paper_fusion_planner",
            waypoints=[
                PathWaypoint(start.x, start.y, start.psi, time=start_time, speed=0.0),
                PathWaypoint(end.x, end.y, end.psi, time=start_time, speed=0.0),
            ],
            curvature_max=0.0,
            length=0.0,
            path_source="stationary",
            metadata={"dubins_modes": "-".join(dubins_path.modes)},
        )

    if use_bezier:
        candidate = _try_bezier_segment(segment_id, kind, start, end, start_time, speed, turn_radius, sample_count, dubins_path.total_length)
        if candidate is not None:
            candidate.metadata["dubins_modes"] = "-".join(dubins_path.modes)
            return candidate

    step = max(dubins_path.total_length / max(sample_count - 1, 1), turn_radius / 8.0)
    points, headings, max_curvature = sample_dubins_path(dubins_path, step_size=step)
    duration = dubins_path.total_length / speed
    return PathSegmentSpec(
        segment_id=segment_id,
        kind=kind,
        source_algorithm="paper_fusion_planner",
        waypoints=_waypoints_from_points(points, headings, start_time, duration, speed),
        curvature_max=max_curvature,
        length=polyline_length(points),
        path_source="dubins_fallback",
        metadata={"dubins_modes": "-".join(dubins_path.modes)},
    )


def build_obstacle_aware_transition_segments(
    segment_id: str,
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    kind: str = "transit",
    sample_count: int = 24,
) -> List[PathSegmentSpec]:
    direct = build_transition_segment(
        segment_id=segment_id,
        start=start,
        end=end,
        start_time=start_time,
        config=config,
        kind=kind,
        sample_count=sample_count,
        use_bezier=path_config.use_bezier_smoothing,
    )
    direct_reasons = path_segment_invalid_reasons(direct, config, obstacle_field)
    if not direct_reasons:
        _annotate_validity(direct, config, obstacle_field, direct_reasons)
        direct.metadata["connector"] = direct.path_source
        return [direct]
    _annotate_validity(direct, config, obstacle_field, direct_reasons)

    search_field = obstacle_field or ObstacleField()
    astar = None
    search_bounds = _turn_radius_safe_bounds(config, start, end)
    for resolution in _astar_resolution_candidates(config, path_config):
        astar = obstacle_aware_grid_astar(
            start=(start.x, start.y),
            goal=(end.x, end.y),
            bounds=search_bounds,
            obstacle_field=search_field,
            resolution=resolution,
            path_config=path_config,
        )
        if astar.found and len(astar.points) >= 2:
            break
    if astar is None:
        raise RuntimeError("A* resolution candidate generation failed")
    if not astar.found or len(astar.points) < 2:
        lattice = _build_motion_lattice_segment(
            segment_id=f"{segment_id}_motion_lattice_no_astar",
            start=start,
            end=end,
            start_time=start_time,
            config=config,
            path_config=path_config,
            obstacle_field=obstacle_field,
            kind=kind,
            sample_count=max(sample_count, 24),
        )
        if lattice is not None:
            lattice.metadata.update(
                {
                    "connector": "motion_lattice_no_astar",
                    "astar_found": str(astar.found).lower(),
                    "astar_expanded": str(astar.expanded),
                    "direct_invalid_reasons": ",".join(direct_reasons),
                }
            )
            return [lattice]
        safe_edge = _build_corridor_edge_segment(f"{segment_id}_safe_edge", start, end, start_time, config, kind)
        safe_edge_reasons = path_segment_invalid_reasons(safe_edge, config, obstacle_field)
        if not safe_edge_reasons:
            _annotate_validity(safe_edge, config, obstacle_field, safe_edge_reasons)
            safe_edge.metadata.update(
                {
                    "connector": "local_safe_corridor_edge",
                    "kinematic_feasible": "false",
                    "astar_found": str(astar.found).lower(),
                    "astar_expanded": str(astar.expanded),
                    "direct_invalid_reasons": ",".join(direct_reasons),
                    "tracking_note": "No curvature-feasible connector was found; this edge is marked for later global replan.",
                }
            )
            return [safe_edge]
        _annotate_validity(direct, config, obstacle_field, direct_reasons)
        direct.metadata["connector"] = "blocked_dubins_no_astar"
        direct.metadata["kinematic_feasible"] = "false"
        direct.metadata["astar_found"] = "false"
        direct.metadata["direct_invalid_reasons"] = ",".join(direct_reasons)
        return [direct]

    smoothed = _build_smoothed_corridor_segment(
        segment_id=f"{segment_id}_smooth_astar",
        corridor_points=astar.points,
        start=start,
        end=end,
        start_time=start_time,
        config=config,
        obstacle_field=obstacle_field,
        kind=kind,
        sample_count=max(sample_count, 24),
    )
    if smoothed is not None:
        smoothed.metadata.update(
            {
                "connector": "smoothed_astar_corridor",
                "astar_cost": f"{astar.cost:.6f}",
                "astar_expanded": str(astar.expanded),
                "corridor_point_count": str(len(astar.points)),
                "direct_invalid_reasons": ",".join(direct_reasons),
            }
        )
        return [smoothed]

    lattice = _build_motion_lattice_segment(
        segment_id=f"{segment_id}_motion_lattice",
        start=start,
        end=end,
        start_time=start_time,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        kind=kind,
        sample_count=max(sample_count, 24),
    )
    if lattice is not None:
        lattice.metadata.update(
            {
                "connector": "motion_lattice",
                "astar_cost": f"{astar.cost:.6f}",
                "astar_expanded": str(astar.expanded),
                "corridor_point_count": str(len(astar.points)),
                "direct_invalid_reasons": ",".join(direct_reasons),
            }
        )
        return [lattice]

    corridor = _simplify_corridor_points(astar.points, search_field)
    corridor[0] = (start.x, start.y)
    corridor[-1] = (end.x, end.y)
    poses = _poses_from_corridor(corridor, start, end)
    segments: List[PathSegmentSpec] = []
    current_time = start_time
    for idx, (pose_a, pose_b) in enumerate(zip(poses[:-1], poses[1:])):
        sub_id = f"{segment_id}_astar_{idx}"
        candidate = build_transition_segment(
            segment_id=sub_id,
            start=pose_a,
            end=pose_b,
            start_time=current_time,
            config=config,
            kind=kind,
            sample_count=max(8, sample_count // 2),
            use_bezier=path_config.use_bezier_smoothing,
        )
        candidate_reasons = path_segment_invalid_reasons(candidate, config, obstacle_field)
        if candidate_reasons:
            fallback = _build_corridor_edge_segment(sub_id, pose_a, pose_b, current_time, config, kind)
            fallback_reasons = path_segment_invalid_reasons(fallback, config, obstacle_field)
            fallback.metadata["fallback_from"] = candidate.path_source
            fallback.metadata["fallback_invalid_reasons"] = ",".join(candidate_reasons)
            fallback.metadata["kinematic_feasible"] = "false"
            candidate = fallback
            candidate_reasons = fallback_reasons
        _annotate_validity(candidate, config, obstacle_field, candidate_reasons)
        candidate.metadata.update(
            {
                "connector": "motion_lattice" if candidate.path_source == "motion_lattice" else "astar_corridor",
                "astar_cost": f"{astar.cost:.6f}",
                "astar_expanded": str(astar.expanded),
                "corridor_index": str(idx),
                "corridor_point_count": str(len(corridor)),
                "direct_invalid_reasons": ",".join(direct_reasons),
            }
        )
        segments.append(candidate)
        current_time = _segment_end_time(candidate)
    return segments


def _build_smoothed_corridor_segment(
    segment_id: str,
    corridor_points: List[Tuple[float, float]],
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
    kind: str,
    sample_count: int,
) -> PathSegmentSpec | None:
    if len(corridor_points) < 2:
        return None
    radius = max(config.fleet.min_turn_radius, 1e-6)
    shaped = _corridor_control_points(corridor_points, start, end, radius)
    if len(shaped) < 2:
        return None
    points = _chaikin_smooth(shaped, iterations=3)
    points[0] = (start.x, start.y)
    points[-1] = (end.x, end.y)
    points = _resample_polyline(points, max(sample_count, 24))
    headings = _headings_from_points(points, start.psi, end.psi)
    max_curvature = _polyline_max_curvature(points)
    if max_curvature > 1.0 / radius + 1e-3:
        return None
    length = polyline_length(points)
    speed = max(config.fleet.turn_speed_max if kind == "turn" else config.fleet.cruise_speed, 1e-6)
    duration = length / speed
    segment = PathSegmentSpec(
        segment_id=segment_id,
        kind=kind,
        source_algorithm="paper_fusion_planner",
        waypoints=_waypoints_from_points(points, headings, start_time, duration, speed),
        curvature_max=max_curvature,
        length=length,
        path_source="smoothed_astar_corridor",
        metadata={
            "kinematic_feasible": "true",
            "motion_model": "curvature_bounded_corridor_smoothing",
        },
    )
    reasons = path_segment_invalid_reasons(segment, config, obstacle_field)
    if reasons:
        return None
    _annotate_validity(segment, config, obstacle_field, reasons)
    return segment


def _corridor_control_points(
    corridor_points: List[Tuple[float, float]],
    start: Pose2D,
    end: Pose2D,
    radius: float,
) -> List[Tuple[float, float]]:
    controls = [(start.x, start.y)]
    start_ahead = (start.x + radius * math.cos(start.psi), start.y + radius * math.sin(start.psi))
    end_back = (end.x - radius * math.cos(end.psi), end.y - radius * math.sin(end.psi))
    controls.append(start_ahead)
    controls.extend(corridor_points[1:-1])
    controls.append(end_back)
    controls.append((end.x, end.y))
    return _drop_near_duplicate_points(controls)


def _chaikin_smooth(points: List[Tuple[float, float]], iterations: int) -> List[Tuple[float, float]]:
    smoothed = list(points)
    for _ in range(max(iterations, 0)):
        if len(smoothed) < 3:
            break
        new_points = [smoothed[0]]
        for p0, p1 in zip(smoothed[:-1], smoothed[1:]):
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_points.extend([q, r])
        new_points.append(smoothed[-1])
        smoothed = new_points
    return _drop_near_duplicate_points(smoothed)


def _resample_polyline(points: List[Tuple[float, float]], sample_count: int) -> List[Tuple[float, float]]:
    if len(points) <= 2:
        return points
    total = polyline_length(points)
    if total <= 1e-9:
        return points
    targets = [total * idx / max(sample_count - 1, 1) for idx in range(sample_count)]
    result: List[Tuple[float, float]] = []
    cursor = 0.0
    segment_idx = 0
    for target in targets:
        while segment_idx < len(points) - 2:
            segment_length = math.hypot(points[segment_idx + 1][0] - points[segment_idx][0], points[segment_idx + 1][1] - points[segment_idx][1])
            if cursor + segment_length >= target:
                break
            cursor += segment_length
            segment_idx += 1
        p0 = points[segment_idx]
        p1 = points[segment_idx + 1]
        segment_length = max(math.hypot(p1[0] - p0[0], p1[1] - p0[1]), 1e-9)
        alpha = max(0.0, min(1.0, (target - cursor) / segment_length))
        result.append((p0[0] + alpha * (p1[0] - p0[0]), p0[1] + alpha * (p1[1] - p0[1])))
    return _drop_near_duplicate_points(result)


def _headings_from_points(points: List[Tuple[float, float]], start_heading: float, end_heading: float) -> List[float]:
    if len(points) < 2:
        return [start_heading]
    headings: List[float] = []
    for idx, point in enumerate(points):
        if idx == 0:
            headings.append(start_heading)
        elif idx == len(points) - 1:
            headings.append(end_heading)
        else:
            prev_point = points[idx - 1]
            next_point = points[idx + 1]
            headings.append(math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]))
    return headings


def _polyline_max_curvature(points: List[Tuple[float, float]]) -> float:
    max_curvature = 0.0
    for p0, p1, p2 in zip(points[:-2], points[1:-1], points[2:]):
        a = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        b = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        c = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
        denom = max(a * b * c, 1e-12)
        area2 = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
        max_curvature = max(max_curvature, 2.0 * area2 / denom)
    return max_curvature


def _drop_near_duplicate_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return points
    result = [points[0]]
    for point in points[1:]:
        if math.hypot(point[0] - result[-1][0], point[1] - result[-1][1]) > 1e-9:
            result.append(point)
    return result


def _build_motion_lattice_segment(
    segment_id: str,
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    kind: str,
    sample_count: int,
) -> PathSegmentSpec | None:
    result = _motion_lattice_search(start, end, config, path_config, obstacle_field, sample_count)
    if result is None:
        return None
    points, headings, expanded = result
    length = polyline_length(points)
    speed = max(config.fleet.turn_speed_max if kind == "turn" else config.fleet.cruise_speed, 1e-6)
    duration = length / speed
    segment = PathSegmentSpec(
        segment_id=segment_id,
        kind=kind,
        source_algorithm="paper_fusion_planner",
        waypoints=_waypoints_from_points(points, headings, start_time, duration, speed),
        curvature_max=1.0 / max(config.fleet.min_turn_radius, 1e-6),
        length=length,
        path_source="motion_lattice",
        metadata={
            "kinematic_feasible": "true",
            "motion_model": "3dof_unicycle_primitives",
            "lattice_expanded": str(expanded),
        },
    )
    reasons = path_segment_invalid_reasons(segment, config, obstacle_field)
    if reasons:
        return None
    _annotate_validity(segment, config, obstacle_field, reasons)
    dynamic_report = validate_transition_dynamics(segment, config, obstacle_field=obstacle_field, retime=True)
    if (
        not dynamic_report.valid
        and "heading_tangent_mismatch" in dynamic_report.reasons
        and path_config.enable_motion_lattice_heading_repair
    ):
        repaired = _repair_motion_lattice_heading(segment, start, end, config, obstacle_field)
        if repaired is not None:
            return repaired
    return segment


def _repair_motion_lattice_heading(
    segment: PathSegmentSpec,
    start: Pose2D,
    end: Pose2D,
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
) -> PathSegmentSpec | None:
    points = [(waypoint.x, waypoint.y) for waypoint in segment.waypoints]
    if len(points) < 3:
        return None
    headings = _headings_from_points(points, start.psi, end.psi)
    repaired = PathSegmentSpec(
        segment_id=segment.segment_id,
        kind=segment.kind,
        source_algorithm=segment.source_algorithm,
        waypoints=_waypoints_from_points(
            points,
            headings,
            segment.waypoints[0].time or 0.0,
            max((segment.waypoints[-1].time or 0.0) - (segment.waypoints[0].time or 0.0), segment.length / max(config.fleet.cruise_speed, 1e-6)),
            max(config.fleet.turn_speed_max if segment.kind == "turn" else config.fleet.cruise_speed, 1e-6),
        ),
        control_points=list(segment.control_points),
        curvature_max=segment.curvature_max,
        length=segment.length,
        path_source=segment.path_source,
        metadata={
            **segment.metadata,
            "heading_repair_applied": "true",
            "heading_repair_method": "tangent_resample_retime",
            "pre_repair_dynamic_reasons": segment.metadata.get("dynamic_invalid_reasons", ""),
        },
    )
    retime_segment_for_dynamics(repaired, config)
    reasons = path_segment_invalid_reasons(repaired, config, obstacle_field)
    if reasons:
        return None
    report = validate_transition_dynamics(repaired, config, obstacle_field=obstacle_field, retime=True)
    if not report.valid:
        return None
    _annotate_validity(repaired, config, obstacle_field, reasons)
    repaired.metadata["dynamic_feasible"] = "true"
    repaired.metadata["nmpc_trackable"] = "proxy_pass"
    return repaired


def _motion_lattice_search(
    start: Pose2D,
    end: Pose2D,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    sample_count: int,
) -> Tuple[List[Tuple[float, float]], List[float], int] | None:
    turn_radius = max(config.fleet.min_turn_radius, 1e-6)
    xy_resolution = max(min(float(path_config.coverage_resolution or config.footprint.width_wf * 0.5), turn_radius * 0.5), 0.5)
    heading_bins = 16
    heading_step = 2.0 * math.pi / heading_bins
    straight_step = xy_resolution
    max_expanded = 16000
    bounds = (0.0, 0.0, config.mission.area_length_x, config.mission.area_length_y)

    def key_for(pose: Pose2D) -> Tuple[int, int, int]:
        heading_idx = int(round((wrap_pi_positive(pose.psi) / (2.0 * math.pi)) * heading_bins)) % heading_bins
        return (int(round(pose.x / xy_resolution)), int(round(pose.y / xy_resolution)), heading_idx)

    def pose_for_key(key: Tuple[int, int, int], fallback: Pose2D | None = None) -> Pose2D:
        if fallback is not None:
            return fallback
        return Pose2D(key[0] * xy_resolution, key[1] * xy_resolution, key[2] * heading_step)

    start_key = key_for(start)
    open_set: List[Tuple[float, int, Tuple[int, int, int]]] = []
    heapq.heappush(open_set, (0.0, 0, start_key))
    poses: Dict[Tuple[int, int, int], Pose2D] = {start_key: start}
    g_score: Dict[Tuple[int, int, int], float] = {start_key: 0.0}
    parents: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], List[Tuple[float, float]], List[float]]] = {}
    closed = set()
    serial = 0
    expanded = 0

    while open_set and expanded < max_expanded:
        _, _, current_key = heapq.heappop(open_set)
        if current_key in closed:
            continue
        closed.add(current_key)
        expanded += 1
        current_pose = pose_for_key(current_key, poses.get(current_key))
        final = build_transition_segment(
            segment_id="lattice_goal_connector",
            start=current_pose,
            end=end,
            start_time=0.0,
            config=config,
            kind="transit",
            sample_count=max(12, sample_count // 2),
            use_bezier=False,
        )
        if math.hypot(current_pose.x - end.x, current_pose.y - end.y) <= max(2.0 * xy_resolution, turn_radius) and not path_segment_invalid_reasons(final, config, obstacle_field):
            points, headings = _reconstruct_lattice_points(current_key, parents, start)
            final_points = [(waypoint.x, waypoint.y) for waypoint in final.waypoints]
            final_headings = [waypoint.psi for waypoint in final.waypoints]
            points.extend(final_points[1:])
            headings.extend(final_headings[1:])
            return _dedupe_points_and_headings(points, headings) + (expanded,)

        for mode in ("straight", "left", "right"):
            primitive_points, primitive_headings, next_pose, cost = _sample_motion_primitive(
                current_pose,
                mode,
                straight_step,
                turn_radius,
                heading_step,
            )
            if not _primitive_within_bounds(primitive_points, bounds):
                continue
            primitive_segment = PathSegmentSpec(
                segment_id="primitive",
                kind="transit",
                source_algorithm="motion_lattice",
                waypoints=[PathWaypoint(x=pt[0], y=pt[1], psi=psi) for pt, psi in zip(primitive_points, primitive_headings)],
            )
            if path_segment_invalid_reasons(primitive_segment, config, obstacle_field):
                continue
            next_key = key_for(next_pose)
            candidate_g = g_score[current_key] + cost
            if candidate_g + 1e-9 >= g_score.get(next_key, float("inf")):
                continue
            g_score[next_key] = candidate_g
            poses[next_key] = next_pose
            parents[next_key] = (current_key, primitive_points, primitive_headings)
            serial += 1
            priority = candidate_g + _lattice_heuristic(next_pose, end, turn_radius)
            heapq.heappush(open_set, (priority, serial, next_key))
    return None


def _sample_motion_primitive(
    start: Pose2D,
    mode: str,
    straight_step: float,
    turn_radius: float,
    heading_step: float,
) -> Tuple[List[Tuple[float, float]], List[float], Pose2D, float]:
    if mode == "straight":
        end = Pose2D(start.x + straight_step * math.cos(start.psi), start.y + straight_step * math.sin(start.psi), start.psi)
        points, headings = straight_segment_points(start, end, 3)
        return points, headings, end, straight_step

    sign = 1.0 if mode == "left" else -1.0
    dtheta = sign * heading_step
    count = max(4, int(math.ceil(turn_radius * abs(dtheta) / max(turn_radius / 8.0, 1e-6))) + 1)
    points: List[Tuple[float, float]] = []
    headings: List[float] = []
    for idx in range(count):
        alpha = idx / max(count - 1, 1)
        theta = start.psi + dtheta * alpha
        x = start.x + sign * turn_radius * (math.sin(theta) - math.sin(start.psi))
        y = start.y - sign * turn_radius * (math.cos(theta) - math.cos(start.psi))
        points.append((x, y))
        headings.append(theta)
    end = Pose2D(points[-1][0], points[-1][1], headings[-1])
    return points, headings, end, turn_radius * abs(dtheta)


def _reconstruct_lattice_points(
    current_key: Tuple[int, int, int],
    parents: Dict[Tuple[int, int, int], Tuple[Tuple[int, int, int], List[Tuple[float, float]], List[float]]],
    start: Pose2D,
) -> Tuple[List[Tuple[float, float]], List[float]]:
    chunks: List[Tuple[List[Tuple[float, float]], List[float]]] = []
    key = current_key
    while key in parents:
        parent, points, headings = parents[key]
        chunks.append((points, headings))
        key = parent
    points = [(start.x, start.y)]
    headings = [start.psi]
    for chunk_points, chunk_headings in reversed(chunks):
        points.extend(chunk_points[1:])
        headings.extend(chunk_headings[1:])
    return _dedupe_points_and_headings(points, headings)


def _dedupe_points_and_headings(points: List[Tuple[float, float]], headings: List[float]) -> Tuple[List[Tuple[float, float]], List[float]]:
    if not points:
        return points, headings
    deduped_points = [points[0]]
    deduped_headings = [headings[0]]
    for point, heading in zip(points[1:], headings[1:]):
        if math.hypot(point[0] - deduped_points[-1][0], point[1] - deduped_points[-1][1]) <= 1e-9:
            deduped_headings[-1] = heading
            continue
        deduped_points.append(point)
        deduped_headings.append(heading)
    return deduped_points, deduped_headings


def _primitive_within_bounds(points: List[Tuple[float, float]], bounds: Tuple[float, float, float, float]) -> bool:
    x_min, y_min, x_max, y_max = bounds
    return all(x_min - 1e-9 <= x <= x_max + 1e-9 and y_min - 1e-9 <= y <= y_max + 1e-9 for x, y in points)


def _lattice_heuristic(pose: Pose2D, goal: Pose2D, turn_radius: float) -> float:
    distance = math.hypot(goal.x - pose.x, goal.y - pose.y)
    heading_error = abs(wrap_angle(goal.psi - pose.psi))
    return distance + turn_radius * heading_error


def wrap_pi_positive(angle: float) -> float:
    return angle % (2.0 * math.pi)


def _try_bezier_segment(
    segment_id: str,
    kind: str,
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    speed: float,
    turn_radius: float,
    sample_count: int,
    dubins_length: float,
) -> PathSegmentSpec | None:
    base_distance = max(math.hypot(end.x - start.x, end.y - start.y), 1e-6)
    for multiplier in (0.8, 1.0, 1.25, 1.6, 2.0, 2.6, 3.2):
        scale = max(turn_radius * 0.85, min(base_distance * 0.35 * multiplier, 3.0 * turn_radius))
        p0 = np.array([start.x, start.y], dtype=float)
        p5 = np.array([end.x, end.y], dtype=float)
        t0 = unit_heading(start.psi)
        t1 = unit_heading(end.psi)
        p1 = p0 + scale * t0
        p2 = p1 + 0.75 * scale * t0
        p4 = p5 - scale * t1
        p3 = p4 - 0.75 * scale * t1
        control_points: List[Tuple[float, float]] = [tuple(p0), tuple(p1), tuple(p2), tuple(p3), tuple(p4), tuple(p5)]
        points, headings, max_curvature = sample_quintic_bezier(control_points, max(sample_count, 2))
        length = polyline_length(points)
        if max_curvature <= 1.0 / max(turn_radius, 1e-6) + 1e-3 and length <= dubins_length * 1.2:
            duration = length / max(speed, 1e-6)
            return PathSegmentSpec(
                segment_id=segment_id,
                kind=kind,
                source_algorithm="paper_fusion_planner",
                waypoints=_waypoints_from_points(points, headings, start_time, duration, speed),
                control_points=control_points,
                curvature_max=max_curvature,
                length=length,
                path_source="bezier",
            )
    return None


def _build_corridor_edge_segment(
    segment_id: str,
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    config: PlannerConfig,
    kind: str,
) -> PathSegmentSpec:
    points, headings = straight_segment_points(start, end, 2)
    length = polyline_length(points)
    speed = max(config.fleet.turn_speed_max if kind == "turn" else config.fleet.cruise_speed, 1e-6)
    duration = length / speed
    return PathSegmentSpec(
        segment_id=segment_id,
        kind=kind,
        source_algorithm="paper_fusion_planner",
        waypoints=_waypoints_from_points(points, headings, start_time, duration, speed),
        curvature_max=0.0,
        length=length,
        path_source="astar_corridor_edge",
        metadata={"tracking_note": "corridor edge is later smoothed by downstream trajectory tracking"},
    )


def _astar_resolution_candidates(config: PlannerConfig, path_config: PathPlanningConfig) -> List[float]:
    base = float(path_config.coverage_resolution or config.footprint.width_wf * 0.5)
    candidates = [
        base,
        config.footprint.width_wf * 0.5,
        config.fleet.min_turn_radius * 0.5,
        config.footprint.width_wf,
        config.fleet.min_turn_radius,
    ]
    deduped: List[float] = []
    for value in candidates:
        value = max(float(value), 0.25)
        if not any(abs(value - existing) <= 1e-9 for existing in deduped):
            deduped.append(value)
    return deduped


def _turn_radius_safe_bounds(config: PlannerConfig, start: Pose2D, end: Pose2D) -> Tuple[float, float, float, float]:
    radius = max(config.fleet.min_turn_radius, 0.0)
    x_min = min(max(radius, 0.0), start.x, end.x)
    y_min = min(max(radius, 0.0), start.y, end.y)
    x_max = max(min(config.mission.area_length_x - radius, config.mission.area_length_x), start.x, end.x)
    y_max = max(min(config.mission.area_length_y - radius, config.mission.area_length_y), start.y, end.y)
    return (x_min, y_min, x_max, y_max)


def _annotate_validity(
    segment: PathSegmentSpec,
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
    reasons: List[str] | None = None,
) -> None:
    reasons = list(reasons if reasons is not None else path_segment_invalid_reasons(segment, config, obstacle_field))
    segment.metadata["collision_free"] = str("obstacle_collision" not in reasons).lower()
    segment.metadata["boundary_safe"] = str("out_of_bounds" not in reasons).lower()
    segment.metadata["curvature_feasible"] = str(
        segment.curvature_max <= 1.0 / max(config.fleet.min_turn_radius, 1e-6) + 1e-3
    ).lower()
    if "kinematic_feasible" not in segment.metadata:
        segment.metadata["kinematic_feasible"] = str(
            not reasons
            and segment.path_source != "astar_corridor_edge"
            and segment.metadata["curvature_feasible"] == "true"
        ).lower()
    segment.metadata.setdefault("dynamic_feasible", "unknown")
    segment.metadata.setdefault("nmpc_trackable", "not_checked")
    if reasons:
        segment.metadata["invalid_reasons"] = ",".join(reasons)
    else:
        segment.metadata.pop("invalid_reasons", None)
    if not reasons and segment.path_source != "astar_corridor_edge":
        validate_transition_dynamics(segment, config, obstacle_field=obstacle_field, retime=False)


def _simplify_corridor_points(points: List[Tuple[float, float]], obstacle_field: ObstacleField) -> List[Tuple[float, float]]:
    if len(points) <= 2:
        return list(points)
    simplified = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        chosen = anchor + 1
        for candidate in range(len(points) - 1, anchor, -1):
            if not segment_collides_with_obstacles(points[anchor], points[candidate], obstacle_field, inflated=True):
                chosen = candidate
                break
        simplified.append(points[chosen])
        anchor = chosen
    return _drop_collinear_points(simplified)


def _drop_collinear_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(points) <= 2:
        return points
    result = [points[0]]
    for prev, current, nxt in zip(points[:-2], points[1:-1], points[2:]):
        v1 = (current[0] - prev[0], current[1] - prev[1])
        v2 = (nxt[0] - current[0], nxt[1] - current[1])
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        if cross > 1e-9:
            result.append(current)
    result.append(points[-1])
    return result


def _poses_from_corridor(points: List[Tuple[float, float]], start: Pose2D, end: Pose2D) -> List[Pose2D]:
    poses: List[Pose2D] = []
    for idx, point in enumerate(points):
        if idx == 0:
            heading = start.psi
        elif idx == len(points) - 1:
            heading = end.psi
        else:
            next_point = points[idx + 1]
            prev_point = points[idx - 1]
            if math.hypot(next_point[0] - point[0], next_point[1] - point[1]) > 1e-9:
                heading = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
            else:
                heading = math.atan2(point[1] - prev_point[1], point[0] - prev_point[0])
        poses.append(Pose2D(point[0], point[1], heading))
    return poses


def _segment_end_time(segment: PathSegmentSpec) -> float:
    if not segment.waypoints or segment.waypoints[-1].time is None:
        return 0.0
    return float(segment.waypoints[-1].time)


def _waypoints_from_points(
    points: List[Tuple[float, float]],
    headings: List[float],
    start_time: float,
    duration: float,
    speed: float,
) -> List[PathWaypoint]:
    waypoints: List[PathWaypoint] = []
    count = max(len(points), 1)
    for idx, ((x, y), psi) in enumerate(zip(points, headings)):
        alpha = idx / max(count - 1, 1)
        waypoints.append(PathWaypoint(x=x, y=y, psi=psi, time=start_time + alpha * duration, speed=speed))
    return waypoints
