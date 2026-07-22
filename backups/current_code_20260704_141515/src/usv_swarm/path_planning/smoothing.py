from __future__ import annotations

import heapq
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..dubins import dubins_shortest_path, sample_dubins_path
from ..geometry import polyline_length, sample_quintic_bezier, straight_segment_points, unit_heading, wrap_angle
from ..schema import PlannerConfig, Pose2D
from .astar import obstacle_aware_grid_astar
from .dynamics_validation import retime_segment_for_dynamics, validate_transition_dynamics, validate_transition_sequence
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
        _annotate_validity(direct, config, obstacle_field, direct_reasons)
        direct.metadata["connector"] = "blocked_dubins_no_astar"
        direct.metadata["kinematic_feasible"] = "false"
        direct.metadata["dynamic_feasible"] = "false"
        direct.metadata["astar_found"] = "false"
        direct.metadata["astar_corridor_conversion_attempted"] = "false"
        direct.metadata["astar_corridor_conversion_success"] = "false"
        direct.metadata["astar_corridor_conversion_failure_reason"] = "astar_not_found"
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
                "astar_corridor_conversion_attempted": "true",
                "astar_corridor_conversion_success": "true",
                "corridor_conversion_method": "smoothed_astar_corridor",
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
                "astar_corridor_conversion_attempted": "true",
                "astar_corridor_conversion_success": "true",
                "corridor_conversion_method": "motion_lattice_after_astar",
                "astar_corridor_conversion_fallback_reason": "smoothed_corridor_failed",
                "astar_cost": f"{astar.cost:.6f}",
                "astar_expanded": str(astar.expanded),
                "corridor_point_count": str(len(astar.points)),
                "direct_invalid_reasons": ",".join(direct_reasons),
            }
        )
        return [lattice]

    converted = _convert_corridor_to_trackable_segments(
        segment_id=segment_id,
        corridor_points=astar.points,
        start=start,
        end=end,
        start_time=start_time,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        kind=kind,
        sample_count=max(sample_count, 24),
    )
    if converted:
        for idx, segment in enumerate(converted):
            segment.metadata.update(
                {
                    "connector": "astar_corridor",
                    "astar_corridor_conversion_attempted": "true",
                    "astar_corridor_conversion_success": "true",
                    "astar_cost": f"{astar.cost:.6f}",
                    "astar_expanded": str(astar.expanded),
                    "corridor_index": str(idx),
                    "corridor_point_count": str(len(astar.points)),
                    "direct_invalid_reasons": ",".join(direct_reasons),
                }
            )
        return converted

    _annotate_validity(direct, config, obstacle_field, direct_reasons)
    direct.metadata.update(
        {
            "connector": "astar_corridor_conversion_failed",
            "kinematic_feasible": "false",
            "dynamic_feasible": "false",
            "astar_found": "true",
            "astar_expanded": str(astar.expanded),
            "astar_cost": f"{astar.cost:.6f}",
            "corridor_point_count": str(len(astar.points)),
            "astar_corridor_conversion_attempted": "true",
            "astar_corridor_conversion_success": "false",
            "astar_corridor_conversion_failure_reason": "all_conversion_methods_failed",
            "direct_invalid_reasons": ",".join(direct_reasons),
        }
    )
    return [direct]


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


def _convert_corridor_to_trackable_segments(
    segment_id: str,
    corridor_points: List[Tuple[float, float]],
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    kind: str,
    sample_count: int,
) -> List[PathSegmentSpec] | None:
    search_field = obstacle_field or ObstacleField()
    last_failure = ""
    for corridor in _corridor_conversion_variants(corridor_points, start, end, search_field):
        filleted = _build_filleted_corridor_segments(
            segment_id=f"{segment_id}_fillet",
            corridor_points=corridor,
            start=start,
            end=end,
            start_time=start_time,
            config=config,
            path_config=path_config,
            obstacle_field=obstacle_field,
            kind=kind,
            sample_count=sample_count,
        )
        if filleted and _transition_sequence_valid(filleted, config, obstacle_field):
            _mark_corridor_conversion(filleted, "filleted_astar_corridor")
            return filleted
        if filleted:
            last_failure = _sequence_invalid_reason(filleted, config, obstacle_field)

    for corridor in _corridor_conversion_variants(corridor_points, start, end, search_field):
        segmented = _build_segmented_corridor_transitions(
            segment_id=f"{segment_id}_segmented",
            corridor_points=corridor,
            start=start,
            end=end,
            start_time=start_time,
            config=config,
            path_config=path_config,
            obstacle_field=obstacle_field,
            kind=kind,
            sample_count=sample_count,
        )
        if segmented and _transition_sequence_valid(segmented, config, obstacle_field):
            _mark_corridor_conversion(segmented, "segmented_dubins_lattice")
            return segmented
        if segmented:
            last_failure = _sequence_invalid_reason(segmented, config, obstacle_field)

    if last_failure:
        return None
    return None


def _corridor_conversion_variants(
    corridor_points: List[Tuple[float, float]],
    start: Pose2D,
    end: Pose2D,
    obstacle_field: ObstacleField,
) -> List[List[Tuple[float, float]]]:
    variants: List[List[Tuple[float, float]]] = []
    raw = _drop_collinear_points(_drop_near_duplicate_points(list(corridor_points)))
    simplified = _simplify_corridor_points(raw, obstacle_field)
    for candidate in (simplified, raw):
        if len(candidate) < 2:
            continue
        candidate = list(candidate)
        candidate[0] = (start.x, start.y)
        candidate[-1] = (end.x, end.y)
        candidate = _drop_near_duplicate_points(candidate)
        if len(candidate) < 2:
            continue
        if not any(_same_point_sequence(candidate, existing) for existing in variants):
            variants.append(candidate)
    return variants


def _same_point_sequence(first: Sequence[Tuple[float, float]], second: Sequence[Tuple[float, float]]) -> bool:
    if len(first) != len(second):
        return False
    return all(math.hypot(a[0] - b[0], a[1] - b[1]) <= 1e-9 for a, b in zip(first, second))


def _build_segmented_corridor_transitions(
    segment_id: str,
    corridor_points: List[Tuple[float, float]],
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    kind: str,
    sample_count: int,
) -> List[PathSegmentSpec] | None:
    poses = _poses_from_corridor(corridor_points, start, end)
    if len(poses) < 2:
        return None
    segments: List[PathSegmentSpec] = []
    current_time = start_time
    for idx, (pose_a, pose_b) in enumerate(zip(poses[:-1], poses[1:])):
        sub_id = f"{segment_id}_{idx}"
        candidate = build_transition_segment(
            segment_id=sub_id,
            start=pose_a,
            end=pose_b,
            start_time=current_time,
            config=config,
            kind=kind,
            sample_count=max(12, sample_count // 2),
            use_bezier=path_config.use_bezier_smoothing,
        )
        if not _single_segment_trackable(candidate, config, obstacle_field):
            lattice = _build_motion_lattice_segment(
                segment_id=f"{sub_id}_lattice",
                start=pose_a,
                end=pose_b,
                start_time=current_time,
                config=config,
                path_config=path_config,
                obstacle_field=obstacle_field,
                kind=kind,
                sample_count=max(24, sample_count // 2),
            )
            if lattice is None or not _single_segment_trackable(lattice, config, obstacle_field):
                return None
            candidate = lattice
        candidate.metadata.update(
            {
                "corridor_conversion_method": "segmented_dubins_lattice",
                "corridor_subsegment": str(idx),
            }
        )
        segments.append(candidate)
        current_time = _segment_end_time(candidate)
    return segments


def _build_filleted_corridor_segments(
    segment_id: str,
    corridor_points: List[Tuple[float, float]],
    start: Pose2D,
    end: Pose2D,
    start_time: float,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    kind: str,
    sample_count: int,
) -> List[PathSegmentSpec] | None:
    points = _drop_near_duplicate_points(list(corridor_points))
    if len(points) < 2:
        return None
    radius = max(config.fleet.min_turn_radius, 1e-6)
    first_heading = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
    last_heading = math.atan2(points[-1][1] - points[-2][1], points[-1][0] - points[-2][0])
    core_points = list(points)
    segments: List[PathSegmentSpec] = []
    current_time = start_time

    start_adapter = _maybe_build_heading_adapter(
        segment_id=f"{segment_id}_start_adapter",
        pose=start,
        target_heading=first_heading,
        next_point=points[1],
        start_time=current_time,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        kind=kind,
        sample_count=sample_count,
        forward=True,
    )
    if start_adapter == "failed":
        return None
    if isinstance(start_adapter, PathSegmentSpec):
        segments.append(start_adapter)
        current_time = _segment_end_time(start_adapter)
        last = start_adapter.waypoints[-1]
        core_points[0] = (last.x, last.y)
    elif abs(wrap_angle(start.psi - first_heading)) > 0.35:
        return None

    end_adapter = _maybe_build_heading_adapter(
        segment_id=f"{segment_id}_end_adapter",
        pose=end,
        target_heading=last_heading,
        next_point=points[-2],
        start_time=0.0,
        config=config,
        path_config=path_config,
        obstacle_field=obstacle_field,
        kind=kind,
        sample_count=sample_count,
        forward=False,
    )
    if end_adapter == "failed":
        return None
    if isinstance(end_adapter, PathSegmentSpec):
        first = end_adapter.waypoints[0]
        core_points[-1] = (first.x, first.y)
    elif abs(wrap_angle(end.psi - last_heading)) > 0.35:
        return None

    fillet = _build_filleted_centerline_segment(
        segment_id=f"{segment_id}_centerline",
        points=core_points,
        start_time=current_time,
        config=config,
        obstacle_field=obstacle_field,
        kind=kind,
        sample_count=sample_count,
    )
    if fillet is None:
        return None
    segments.append(fillet)
    current_time = _segment_end_time(fillet)

    if isinstance(end_adapter, PathSegmentSpec):
        _retime_segment_from(end_adapter, current_time, config)
        segments.append(end_adapter)
    if not _transition_sequence_valid(segments, config, obstacle_field):
        return None
    for idx, segment in enumerate(segments):
        segment.metadata.update(
            {
                "corridor_conversion_method": "filleted_astar_corridor",
                "corridor_subsegment": str(idx),
            }
        )
    return segments


def _maybe_build_heading_adapter(
    segment_id: str,
    pose: Pose2D,
    target_heading: float,
    next_point: Tuple[float, float],
    start_time: float,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    kind: str,
    sample_count: int,
    forward: bool,
) -> PathSegmentSpec | str | None:
    heading_error = abs(wrap_angle(pose.psi - target_heading))
    if heading_error <= 0.35:
        return None
    radius = max(config.fleet.min_turn_radius, 1e-6)
    distance = math.hypot(next_point[0] - pose.x, next_point[1] - pose.y)
    if distance <= radius * 1.25:
        return "failed"
    adapter_length = min(max(radius, config.footprint.width_wf * 0.5), distance * 0.45)
    if forward:
        adapter_pose = Pose2D(
            pose.x + adapter_length * math.cos(target_heading),
            pose.y + adapter_length * math.sin(target_heading),
            target_heading,
        )
        start_pose = pose
        end_pose = adapter_pose
    else:
        adapter_pose = Pose2D(
            pose.x - adapter_length * math.cos(target_heading),
            pose.y - adapter_length * math.sin(target_heading),
            target_heading,
        )
        start_pose = adapter_pose
        end_pose = pose
    adapter = build_transition_segment(
        segment_id=segment_id,
        start=start_pose,
        end=end_pose,
        start_time=start_time,
        config=config,
        kind=kind,
        sample_count=max(16, sample_count // 2),
        use_bezier=path_config.use_bezier_smoothing,
    )
    if not _single_segment_trackable(adapter, config, obstacle_field):
        lattice = _build_motion_lattice_segment(
            segment_id=f"{segment_id}_lattice",
            start=start_pose,
            end=end_pose,
            start_time=start_time,
            config=config,
            path_config=path_config,
            obstacle_field=obstacle_field,
            kind=kind,
            sample_count=max(24, sample_count // 2),
        )
        if lattice is None or not _single_segment_trackable(lattice, config, obstacle_field):
            return "failed"
        adapter = lattice
    adapter.metadata["heading_adapter"] = "true"
    return adapter


def _build_filleted_centerline_segment(
    segment_id: str,
    points: List[Tuple[float, float]],
    start_time: float,
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
    kind: str,
    sample_count: int,
) -> PathSegmentSpec | None:
    points = _drop_near_duplicate_points(points)
    if len(points) < 2:
        return None
    radius = max(config.fleet.min_turn_radius, 1e-6)
    sample_spacing = max(radius / 6.0, polyline_length(points) / max(sample_count, 12))
    fillet = _fillet_polyline(points, radius, sample_spacing)
    if fillet is None:
        return None
    fillet_points, headings = fillet
    length = polyline_length(fillet_points)
    speed = max(config.fleet.turn_speed_max if kind == "turn" else config.fleet.cruise_speed, 1e-6)
    duration = length / speed
    segment = PathSegmentSpec(
        segment_id=segment_id,
        kind=kind,
        source_algorithm="paper_fusion_planner",
        waypoints=_waypoints_from_points(fillet_points, headings, start_time, duration, speed),
        curvature_max=1.0 / radius,
        length=length,
        path_source="filleted_astar_corridor",
        metadata={
            "kinematic_feasible": "true",
            "motion_model": "curvature_bounded_corridor_fillet",
        },
    )
    if not _single_segment_trackable(segment, config, obstacle_field):
        return None
    return segment


def _fillet_polyline(
    points: List[Tuple[float, float]],
    radius: float,
    sample_spacing: float,
) -> Tuple[List[Tuple[float, float]], List[float]] | None:
    if len(points) < 2:
        return None
    if len(points) == 2:
        heading = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
        count = max(2, int(math.ceil(polyline_length(points) / max(sample_spacing, 1e-6))) + 1)
        straight_points = [
            (
                points[0][0] + (points[1][0] - points[0][0]) * idx / max(count - 1, 1),
                points[0][1] + (points[1][1] - points[0][1]) * idx / max(count - 1, 1),
            )
            for idx in range(count)
        ]
        return straight_points, [heading for _ in straight_points]

    corner_data: Dict[int, Tuple[Tuple[float, float], Tuple[float, float], List[Tuple[float, float]], List[float]]] = {}
    for idx in range(1, len(points) - 1):
        prev_point = points[idx - 1]
        point = points[idx]
        next_point = points[idx + 1]
        incoming_length = math.hypot(point[0] - prev_point[0], point[1] - prev_point[1])
        outgoing_length = math.hypot(next_point[0] - point[0], next_point[1] - point[1])
        if incoming_length <= 1e-9 or outgoing_length <= 1e-9:
            return None
        u_in = ((point[0] - prev_point[0]) / incoming_length, (point[1] - prev_point[1]) / incoming_length)
        u_out = ((next_point[0] - point[0]) / outgoing_length, (next_point[1] - point[1]) / outgoing_length)
        cross = u_in[0] * u_out[1] - u_in[1] * u_out[0]
        dot = max(-1.0, min(1.0, u_in[0] * u_out[0] + u_in[1] * u_out[1]))
        turn_angle = math.atan2(cross, dot)
        if abs(turn_angle) <= 1e-3:
            continue
        tangent_distance = radius * math.tan(abs(turn_angle) / 2.0)
        if tangent_distance >= min(incoming_length, outgoing_length) * 0.92:
            return None
        tangent_in = (point[0] - u_in[0] * tangent_distance, point[1] - u_in[1] * tangent_distance)
        tangent_out = (point[0] + u_out[0] * tangent_distance, point[1] + u_out[1] * tangent_distance)
        sign = 1.0 if turn_angle > 0.0 else -1.0
        left_normal = (-u_in[1], u_in[0])
        center = (tangent_in[0] + sign * radius * left_normal[0], tangent_in[1] + sign * radius * left_normal[1])
        arc_points, arc_headings = _sample_circular_arc(center, tangent_in, tangent_out, sign, radius, sample_spacing)
        if len(arc_points) < 2:
            return None
        corner_data[idx] = (tangent_in, tangent_out, arc_points, arc_headings)

    result_points: List[Tuple[float, float]] = [points[0]]
    first_heading = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
    result_headings: List[float] = [first_heading]
    cursor = points[0]
    for idx in range(1, len(points) - 1):
        next_line_end = corner_data[idx][0] if idx in corner_data else points[idx]
        line_heading = math.atan2(next_line_end[1] - cursor[1], next_line_end[0] - cursor[0]) if math.hypot(next_line_end[0] - cursor[0], next_line_end[1] - cursor[1]) > 1e-9 else result_headings[-1]
        _append_line_samples(result_points, result_headings, cursor, next_line_end, line_heading, sample_spacing)
        if idx in corner_data:
            _, tangent_out, arc_points, arc_headings = corner_data[idx]
            _append_points_and_headings(result_points, result_headings, arc_points, arc_headings)
            cursor = tangent_out
        else:
            cursor = points[idx]
    final_heading = math.atan2(points[-1][1] - cursor[1], points[-1][0] - cursor[0]) if math.hypot(points[-1][0] - cursor[0], points[-1][1] - cursor[1]) > 1e-9 else result_headings[-1]
    _append_line_samples(result_points, result_headings, cursor, points[-1], final_heading, sample_spacing)
    return _dedupe_points_and_headings(result_points, result_headings)


def _sample_circular_arc(
    center: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
    sign: float,
    radius: float,
    sample_spacing: float,
) -> Tuple[List[Tuple[float, float]], List[float]]:
    a0 = math.atan2(start[1] - center[1], start[0] - center[0])
    a1 = math.atan2(end[1] - center[1], end[0] - center[0])
    if sign > 0.0:
        while a1 < a0:
            a1 += 2.0 * math.pi
    else:
        while a1 > a0:
            a1 -= 2.0 * math.pi
    delta = a1 - a0
    count = max(4, int(math.ceil(abs(delta) * radius / max(sample_spacing, 1e-6))) + 1)
    points: List[Tuple[float, float]] = []
    headings: List[float] = []
    for idx in range(count):
        alpha = idx / max(count - 1, 1)
        angle = a0 + delta * alpha
        points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
        headings.append(angle + sign * math.pi / 2.0)
    return points, headings


def _append_line_samples(
    points: List[Tuple[float, float]],
    headings: List[float],
    start: Tuple[float, float],
    end: Tuple[float, float],
    heading: float,
    sample_spacing: float,
) -> None:
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length <= 1e-9:
        return
    count = max(2, int(math.ceil(length / max(sample_spacing, 1e-6))) + 1)
    line_points = [
        (
            start[0] + (end[0] - start[0]) * idx / max(count - 1, 1),
            start[1] + (end[1] - start[1]) * idx / max(count - 1, 1),
        )
        for idx in range(count)
    ]
    _append_points_and_headings(points, headings, line_points, [heading for _ in line_points])


def _append_points_and_headings(
    points: List[Tuple[float, float]],
    headings: List[float],
    new_points: Sequence[Tuple[float, float]],
    new_headings: Sequence[float],
) -> None:
    for point, heading in zip(new_points, new_headings):
        if points and math.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) <= 1e-9:
            headings[-1] = heading
            continue
        points.append(point)
        headings.append(heading)


def _retime_segment_from(segment: PathSegmentSpec, start_time: float, config: PlannerConfig) -> None:
    if not segment.waypoints:
        return
    duration = max((segment.waypoints[-1].time or 0.0) - (segment.waypoints[0].time or 0.0), segment.length / max(config.fleet.cruise_speed, 1e-6))
    speed = max((waypoint.speed or 0.0 for waypoint in segment.waypoints), default=config.fleet.cruise_speed)
    points = [(waypoint.x, waypoint.y) for waypoint in segment.waypoints]
    headings = [waypoint.psi for waypoint in segment.waypoints]
    segment.waypoints = _waypoints_from_points(points, headings, start_time, duration, speed)
    retime_segment_for_dynamics(segment, config)


def _single_segment_trackable(
    segment: PathSegmentSpec,
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
) -> bool:
    if segment.path_source == "astar_corridor_edge":
        return False
    reasons = path_segment_invalid_reasons(segment, config, obstacle_field)
    if reasons:
        _annotate_validity(segment, config, obstacle_field, reasons)
        return False
    report = validate_transition_dynamics(segment, config, obstacle_field=obstacle_field, retime=True)
    return report.valid and segment.metadata.get("kinematic_feasible") != "false"


def _transition_sequence_valid(
    segments: Sequence[PathSegmentSpec],
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
) -> bool:
    if not segments or any(segment.path_source == "astar_corridor_edge" for segment in segments):
        return False
    report = validate_transition_sequence(segments, config, obstacle_field=obstacle_field, retime=True)
    return report.valid and all(
        segment.metadata.get("kinematic_feasible") != "false"
        and segment.metadata.get("dynamic_feasible") != "false"
        for segment in segments
    )


def _sequence_invalid_reason(
    segments: Sequence[PathSegmentSpec],
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
) -> str:
    report = validate_transition_sequence(segments, config, obstacle_field=obstacle_field, retime=True)
    return ",".join(report.reasons) or "dynamic_validation_failed"


def _mark_corridor_conversion(segments: Sequence[PathSegmentSpec], method: str) -> None:
    for segment in segments:
        segment.metadata["astar_corridor_conversion_attempted"] = "true"
        segment.metadata["astar_corridor_conversion_success"] = "true"
        segment.metadata["corridor_conversion_method"] = method


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
    if not dynamic_report.valid:
        return None
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
