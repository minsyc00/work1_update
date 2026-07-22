from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple

from ..dubins import dubins_shortest_path, sample_dubins_path
from ..geometry import wrap_angle
from ..schema import PlannerConfig, Pose2D
from .coverage import RectangularCoverageModel
from .obstacles import (
    clipped_axis_aligned_segments,
    point_in_mission_bounds,
    point_in_polygon,
    polyline_collides_with_obstacles,
    polyline_out_of_mission_bounds,
    sampled_segment_footprint_collides,
)
from .types import CoveragePass, DecomposedRegion, FreeSpaceCell, ObstacleField, PathPlanningConfig, RegionCoveragePattern


ScanAxisCandidate = Tuple[str, float | None, str, float | None, int]


def candidate_scan_axes(
    region: DecomposedRegion,
    max_axes: int = 2,
    config: PlannerConfig | None = None,
    path_config: PathPlanningConfig | None = None,
) -> List[str]:
    if config is not None and path_config is not None:
        return [item[0] for item in _candidate_scan_specs(region, config, path_config)]
    axes = [region.preferred_axis]
    axes.append("y" if region.preferred_axis == "x" else "x")
    deduped: List[str] = []
    for axis in axes:
        if axis not in deduped:
            deduped.append(axis)
    return deduped[: max(max_axes, 1)]


def generate_region_patterns(
    region: DecomposedRegion,
    config: PlannerConfig,
    path_config: PathPlanningConfig | None = None,
    obstacle_field: ObstacleField | None = None,
) -> List[RegionCoveragePattern]:
    path_config = path_config or PathPlanningConfig.from_planner_config(config)
    patterns: List[RegionCoveragePattern] = []
    for axis, angle, source, support_span, angle_candidate_count in _candidate_scan_specs(region, config, path_config):
        pattern = _build_pattern_for_axis(
            region,
            config,
            path_config,
            axis,
            obstacle_field,
            scan_angle=angle,
            scan_axis_source=source,
            support_span=support_span,
            angle_candidate_count=angle_candidate_count,
        )
        if pattern.passes:
            patterns.append(pattern)
    return sorted(patterns, key=lambda item: (not item.feasible, item.estimated_time, item.total_length))


def generate_all_region_patterns(
    regions: Iterable[DecomposedRegion],
    config: PlannerConfig,
    path_config: PathPlanningConfig | None = None,
    obstacle_field: ObstacleField | None = None,
) -> Dict[str, List[RegionCoveragePattern]]:
    return {region.region_id: generate_region_patterns(region, config, path_config, obstacle_field) for region in regions}


def _candidate_scan_specs(
    region: DecomposedRegion,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> List[ScanAxisCandidate]:
    if not _region_supports_oriented_sweep(region, config, path_config):
        return [
            (
                axis,
                _axis_angle(axis),
                "axis_aligned",
                _support_span_for_axis(region.polygon, axis) if region.polygon else None,
                0,
            )
            for axis in candidate_scan_axes(region, path_config.max_candidate_axes)
        ]

    tolerance = math.radians(max(path_config.oriented_sweep_angle_tolerance_deg, 0.01))
    raw_angles: List[Tuple[float, str]] = []
    for angle in _polygon_edge_angles(region.polygon):
        raw_angles.append((angle, "edge_direction"))
    principal_angle = _principal_axis_angle(region.polygon)
    if principal_angle is not None:
        raw_angles.append((principal_angle, "long_axis_direction"))

    deduped_angles: List[Tuple[float, str, float]] = []
    for angle, source in raw_angles:
        normalized = _normalize_angle_pi(angle)
        span = _support_span_for_angle(region.polygon, normalized)
        existing_index = _find_similar_angle_index(deduped_angles, normalized, tolerance)
        if existing_index is None:
            deduped_angles.append((normalized, source, span))
            continue
        existing_angle, existing_source, existing_span = deduped_angles[existing_index]
        if span + 1e-9 < existing_span or (
            abs(span - existing_span) <= 1e-9 and _angle_source_priority(source) < _angle_source_priority(existing_source)
        ):
            deduped_angles[existing_index] = (existing_angle, source, span)

    axis_fallbacks: List[ScanAxisCandidate] = []
    fallback_axes = candidate_scan_axes(region, 2) if path_config.include_axis_aligned_sweep_fallbacks else []
    for axis in fallback_axes:
        axis_fallbacks.append((axis, _axis_angle(axis), "axis_aligned_fallback", _support_span_for_axis(region.polygon, axis), len(deduped_angles)))

    all_angle_records = list(deduped_angles)
    all_angle_records.extend((angle, f"{axis}_fallback", span or 0.0) for axis, angle, _, span, _ in axis_fallbacks if angle is not None)
    minimum_span_angle: float | None = None
    if all_angle_records:
        minimum_span_angle = min(all_angle_records, key=lambda item: (item[2], _angle_source_priority(item[1])))[0]

    non_axis_records: List[Tuple[float, str, float]] = []
    for angle, source, span in deduped_angles:
        if _angle_matches_axis(angle, tolerance):
            continue
        if minimum_span_angle is not None and _angle_diff_mod_pi(angle, minimum_span_angle) <= tolerance:
            source = "minimum_span_direction"
        non_axis_records.append((angle, source, span))

    limit = max(int(path_config.max_oriented_sweep_angles_per_region), 0)
    selected_angles: List[Tuple[float, str, float]] = []
    if limit > 0:
        minimum_non_axis = None
        if minimum_span_angle is not None:
            for record in non_axis_records:
                if _angle_diff_mod_pi(record[0], minimum_span_angle) <= tolerance:
                    minimum_non_axis = record
                    break
        if minimum_non_axis is not None:
            selected_angles.append(minimum_non_axis)
        for record in sorted(non_axis_records, key=lambda item: (item[2], _angle_source_priority(item[1]), item[0])):
            if len(selected_angles) >= limit:
                break
            if _find_similar_angle_index(selected_angles, record[0], tolerance) is None:
                selected_angles.append(record)

    specs: List[ScanAxisCandidate] = [
        (_theta_axis(angle), angle, source, span, len(deduped_angles)) for angle, source, span in selected_angles
    ]
    specs.extend(axis_fallbacks)
    if not specs:
        specs = [
            (
                axis,
                _axis_angle(axis),
                "axis_aligned",
                _support_span_for_axis(region.polygon, axis) if region.polygon else None,
                len(deduped_angles),
            )
            for axis in candidate_scan_axes(region, path_config.max_candidate_axes)
        ]
    return _dedupe_scan_specs(specs, tolerance)


def _region_supports_oriented_sweep(
    region: DecomposedRegion,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
) -> bool:
    if not path_config.enable_oriented_sweep_patterns:
        return False
    if getattr(region, "member_cells", None):
        return False
    if len(region.polygon) < 3:
        return False
    min_area = (
        max(config.footprint.width_wf, 1e-6)
        * max(config.footprint.length_lf, 1e-6)
        * max(path_config.oriented_sweep_min_area_factor, 0.0)
    )
    if region.area + 1e-9 < min_area:
        return False
    shape_class = str(region.metadata.get("shape_class", "")).lower()
    if shape_class in {"rectangle", "trapezoid", "convex_polygon", "large_convex"}:
        return True
    return region.metadata.get("convex_region_decomposition") == "true"


def _dedupe_scan_specs(specs: List[ScanAxisCandidate], tolerance: float) -> List[ScanAxisCandidate]:
    deduped: List[ScanAxisCandidate] = []
    for spec in specs:
        axis, angle, source, support_span, angle_candidate_count = spec
        if axis in {"x", "y"}:
            if any(existing[0] == axis for existing in deduped):
                continue
            deduped.append(spec)
            continue
        if angle is None:
            deduped.append(spec)
            continue
        if _find_similar_angle_index([(item[1] or 0.0, item[2], item[3] or 0.0) for item in deduped if item[0].startswith("theta:")], angle, tolerance) is not None:
            continue
        deduped.append((axis, angle, source, support_span, angle_candidate_count))
    return deduped


def _axis_angle(axis: str) -> float | None:
    if axis == "x":
        return 0.0
    if axis == "y":
        return math.pi / 2.0
    return _parse_theta_axis(axis)


def _theta_axis(angle: float) -> str:
    return f"theta:{_normalize_angle_pi(angle):.6f}"


def _parse_theta_axis(axis: str) -> float | None:
    if not axis.startswith("theta:"):
        return None
    try:
        return _normalize_angle_pi(float(axis.split(":", 1)[1]))
    except ValueError:
        return None


def _normalize_angle_pi(angle: float) -> float:
    normalized = math.fmod(angle, math.pi)
    if normalized < 0.0:
        normalized += math.pi
    if abs(normalized - math.pi) <= 1e-9:
        return 0.0
    return normalized


def _angle_diff_mod_pi(first: float, second: float) -> float:
    diff = abs(_normalize_angle_pi(first) - _normalize_angle_pi(second))
    return min(diff, math.pi - diff)


def _angle_matches_axis(angle: float, tolerance: float) -> bool:
    return _angle_diff_mod_pi(angle, 0.0) <= tolerance or _angle_diff_mod_pi(angle, math.pi / 2.0) <= tolerance


def _find_similar_angle_index(
    records: List[Tuple[float, str, float]],
    angle: float,
    tolerance: float,
) -> int | None:
    for idx, (existing_angle, _, _) in enumerate(records):
        if _angle_diff_mod_pi(existing_angle, angle) <= tolerance:
            return idx
    return None


def _angle_source_priority(source: str) -> int:
    priorities = {
        "minimum_span_direction": 0,
        "edge_direction": 1,
        "long_axis_direction": 2,
        "axis_aligned_fallback": 3,
        "x_fallback": 4,
        "y_fallback": 4,
    }
    return priorities.get(source, 5)


def _polygon_edge_angles(polygon: List[Tuple[float, float]]) -> List[float]:
    angles: List[float] = []
    for idx, start in enumerate(polygon):
        end = polygon[(idx + 1) % len(polygon)]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if math.hypot(dx, dy) <= 1e-9:
            continue
        angles.append(_normalize_angle_pi(math.atan2(dy, dx)))
    return angles


def _principal_axis_angle(polygon: List[Tuple[float, float]]) -> float | None:
    if len(polygon) < 2:
        return None
    cx = sum(point[0] for point in polygon) / len(polygon)
    cy = sum(point[1] for point in polygon) / len(polygon)
    sxx = sum((point[0] - cx) ** 2 for point in polygon)
    syy = sum((point[1] - cy) ** 2 for point in polygon)
    sxy = sum((point[0] - cx) * (point[1] - cy) for point in polygon)
    if sxx + syy <= 1e-12:
        return None
    return _normalize_angle_pi(0.5 * math.atan2(2.0 * sxy, sxx - syy))


def _support_span_for_axis(polygon: List[Tuple[float, float]], axis: str) -> float | None:
    angle = _axis_angle(axis)
    if angle is None or not polygon:
        return None
    return _support_span_for_angle(polygon, angle)


def _support_span_for_angle(polygon: List[Tuple[float, float]], angle: float) -> float:
    if not polygon:
        return 0.0
    vx = -math.sin(angle)
    vy = math.cos(angle)
    values = [point[0] * vx + point[1] * vy for point in polygon]
    return max(values) - min(values)


def _scan_axis_metadata(
    axis: str,
    source: str,
    support_span: float | None,
    angle_candidate_count: int,
    angle: float | None = None,
) -> Dict[str, str]:
    resolved_angle = _axis_angle(axis) if angle is None else _normalize_angle_pi(angle)
    metadata = {
        "scan_axis_source": source,
        "angle_candidate_count": str(int(angle_candidate_count)),
        "support_span": "" if support_span is None else f"{support_span:.6f}",
    }
    if resolved_angle is not None:
        metadata.update(
            {
                "scan_angle_rad": f"{resolved_angle:.6f}",
                "scan_angle_deg": f"{math.degrees(resolved_angle):.3f}",
            }
        )
    return metadata


def _build_pattern_for_axis(
    region: DecomposedRegion,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    axis: str,
    obstacle_field: ObstacleField | None = None,
    scan_angle: float | None = None,
    scan_axis_source: str = "axis_aligned",
    support_span: float | None = None,
    angle_candidate_count: int = 0,
) -> RegionCoveragePattern:
    x_min, y_min, x_max, y_max = region.bounds
    width = config.footprint.width_wf
    margin_x = min(max(config.safety.boundary_margin_x, 0.0), max((x_max - x_min) / 2.0, 0.0))
    margin_y = min(max(config.safety.boundary_margin_y, 0.0), max((y_max - y_min) / 2.0, 0.0))
    turn_pocket = (
        max(config.fleet.min_turn_radius * max(path_config.coverage_turn_pocket_scale, 0.0), 0.0)
        if obstacle_field is not None
        else 0.0
    )
    min_pass_length = _minimum_pass_length(config, path_config)
    axis_metadata = _scan_axis_metadata(axis, scan_axis_source, support_span, angle_candidate_count, scan_angle)
    theta_angle = _parse_theta_axis(axis)

    if getattr(region, "member_cells", None):
        return _build_composite_pattern_for_axis(
            region,
            config,
            path_config,
            axis,
            obstacle_field,
            extra_metadata=axis_metadata,
        )
    if theta_angle is not None and _use_true_polygon_scan_intersections(region):
        return _build_oriented_polygon_pattern_for_angle(
            region,
            config,
            path_config,
            theta_angle,
            obstacle_field,
            axis,
            scan_axis_source,
            support_span,
            angle_candidate_count,
        )
    if theta_angle is not None:
        return _finalize_pattern(
            region,
            config,
            axis,
            [],
            obstacle_field,
            source="unsupported_oriented_boustrophedon_candidate",
            extra_metadata=axis_metadata,
        )
    if _use_true_polygon_scan_intersections(region):
        return _build_polygon_pattern_for_axis(
            region,
            config,
            path_config,
            axis,
            obstacle_field,
            extra_metadata=axis_metadata,
        )

    if axis == "x":
        cross_width = max(y_max - y_min, 0.0)
        pass_count = _coverage_pass_count(cross_width, config, path_config)
        centers = _coverage_centers(y_min, y_max, width, pass_count)
        x0, x1 = _initial_scan_interval(x_min, x_max, margin_x, turn_pocket, min_pass_length, path_config)
        passes: List[CoveragePass] = []
        sequence_index = 0
        for center_y in centers:
            intervals = [(x0, x1)]
            if obstacle_field is not None:
                intervals = clipped_axis_aligned_segments("x", center_y, x0, x1, obstacle_field, width, min_length=min_pass_length)
            for interval_start, interval_end in intervals:
                interval_start, interval_end = _scan_interval_for_segment(interval_start, interval_end, turn_pocket, min_pass_length, path_config)
                if interval_end - interval_start < min_pass_length:
                    continue
                if sequence_index % 2 == 0:
                    start = Pose2D(interval_start, center_y, 0.0)
                    end = Pose2D(interval_end, center_y, 0.0)
                else:
                    start = Pose2D(interval_end, center_y, math.pi)
                    end = Pose2D(interval_start, center_y, math.pi)
                passes.append(
                    CoveragePass(
                        pass_id=f"{region.region_id}_x_pass_{sequence_index}",
                        region_id=region.region_id,
                        sequence_index=sequence_index,
                        scan_axis="x",
                        start_pose=start,
                        end_pose=end,
                        center_coordinate=center_y,
                        width=width,
                        length=max(abs(interval_end - interval_start), 0.0),
                    )
                )
                sequence_index += 1
    else:
        cross_width = max(x_max - x_min, 0.0)
        pass_count = _coverage_pass_count(cross_width, config, path_config)
        centers = _coverage_centers(x_min, x_max, width, pass_count)
        y0, y1 = _initial_scan_interval(y_min, y_max, margin_y, turn_pocket, min_pass_length, path_config)
        passes = []
        sequence_index = 0
        for center_x in centers:
            intervals = [(y0, y1)]
            if obstacle_field is not None:
                intervals = clipped_axis_aligned_segments("y", center_x, y0, y1, obstacle_field, width, min_length=min_pass_length)
            for interval_start, interval_end in intervals:
                interval_start, interval_end = _scan_interval_for_segment(interval_start, interval_end, turn_pocket, min_pass_length, path_config)
                if interval_end - interval_start < min_pass_length:
                    continue
                if sequence_index % 2 == 0:
                    start = Pose2D(center_x, interval_start, math.pi / 2.0)
                    end = Pose2D(center_x, interval_end, math.pi / 2.0)
                else:
                    start = Pose2D(center_x, interval_end, -math.pi / 2.0)
                    end = Pose2D(center_x, interval_start, -math.pi / 2.0)
                passes.append(
                    CoveragePass(
                        pass_id=f"{region.region_id}_y_pass_{sequence_index}",
                        region_id=region.region_id,
                        sequence_index=sequence_index,
                        scan_axis="y",
                        start_pose=start,
                        end_pose=end,
                        center_coordinate=center_x,
                        width=width,
                        length=max(abs(interval_end - interval_start), 0.0),
                    )
                )
                sequence_index += 1

    passes, retraction_metadata = _apply_adaptive_pass_retraction(
        passes,
        config,
        path_config,
        obstacle_field,
        min_pass_length,
    )
    passes, endpoint_retraction_metadata = _apply_boundary_endpoint_retraction(
        passes,
        config,
        path_config,
        min_pass_length,
    )
    retraction_metadata = {**retraction_metadata, **endpoint_retraction_metadata}
    coverage_length = sum(item.length for item in passes)
    region_area = max((x_max - x_min) * (y_max - y_min), 1e-9)
    estimated_coverage_fraction = min(1.0, max(0.0, coverage_length * width / region_area))
    turn_length = 0.0
    turn_angle = 0.0
    max_curvature = 0.0
    feasible = True
    collision_free = True
    boundary_safe = True
    for coverage_pass in passes:
        if not point_in_mission_bounds((coverage_pass.start_pose.x, coverage_pass.start_pose.y), config) or not point_in_mission_bounds(
            (coverage_pass.end_pose.x, coverage_pass.end_pose.y),
            config,
        ):
            feasible = False
            boundary_safe = False
    if obstacle_field is not None:
        for coverage_pass in passes:
            if sampled_segment_footprint_collides(
                coverage_pass.start_pose,
                coverage_pass.end_pose,
                config.footprint.length_lf,
                config.footprint.width_wf,
                obstacle_field,
                sample_spacing=max(config.footprint.width_wf / 2.0, 1e-6),
                inflated=False,
            ):
                feasible = False
                collision_free = False
                break
    for current_pass, next_pass in zip(passes[:-1], passes[1:]):
        transition = dubins_shortest_path(current_pass.end_pose, next_pass.start_pose, config.fleet.min_turn_radius)
        turn_length += transition.total_length
        max_curvature = max(max_curvature, 1.0 / config.fleet.min_turn_radius)
        turn_angle += _dubins_turn_angle(transition.segment_lengths, transition.modes, config.fleet.min_turn_radius)
        if max_curvature > 1.0 / config.fleet.min_turn_radius + 1e-3:
            feasible = False
        if obstacle_field is not None:
            points, _, _ = sample_dubins_path(
                transition,
                step_size=max(config.fleet.min_turn_radius / 8.0, 0.25),
            )
            if polyline_out_of_mission_bounds(points, config):
                feasible = False
                boundary_safe = False
            if polyline_collides_with_obstacles(points, obstacle_field, inflated=True):
                feasible = False
                collision_free = False
        else:
            points, _, _ = sample_dubins_path(
                transition,
                step_size=max(config.fleet.min_turn_radius / 8.0, 0.25),
            )
            if polyline_out_of_mission_bounds(points, config):
                feasible = False
                boundary_safe = False

    cover_speed = max(config.fleet.cover_speed, 1e-6)
    turn_speed = max(min(config.fleet.turn_speed_max, config.fleet.cruise_speed), 1e-6)
    yaw_rate = max(config.fleet.turn_speed_max / max(config.fleet.min_turn_radius, 1e-6), 1e-6)
    estimated_time = coverage_length / cover_speed + turn_length / turn_speed + turn_angle / yaw_rate

    entry = passes[0].start_pose if passes else Pose2D(region.center[0], region.center[1], 0.0)
    exit_pose = passes[-1].end_pose if passes else entry
    total_length = coverage_length + turn_length
    return RegionCoveragePattern(
        pattern_id=f"{region.region_id}_pattern_{axis}",
        region_id=region.region_id,
        scan_axis=axis,
        passes=passes,
        entry_pose=entry,
        exit_pose=exit_pose,
        coverage_length=coverage_length,
        turn_length=turn_length,
        turn_angle=turn_angle,
        total_length=total_length,
        estimated_time=estimated_time,
        max_curvature=max_curvature,
        feasible=feasible,
        metadata={
            "pass_count": str(len(passes)),
            "source": "boustrophedon_candidate",
            "collision_free": str(collision_free).lower(),
            "boundary_safe": str(boundary_safe).lower(),
            "static_obstacle_aware": str(obstacle_field is not None).lower(),
            "region_bounds": f"{x_min:.6f},{y_min:.6f},{x_max:.6f},{y_max:.6f}",
            "region_area": f"{region_area:.6f}",
            "estimated_region_coverage_fraction": f"{estimated_coverage_fraction:.6f}",
            "shape_class": str(region.metadata.get("shape_class", "rectangle")),
            "dominant_scan_axis": str(region.metadata.get("dominant_scan_axis", axis)),
            "support_span": str(region.metadata.get("support_span", "")),
            **axis_metadata,
            **retraction_metadata,
        },
    )


def _use_true_polygon_scan_intersections(region: DecomposedRegion) -> bool:
    shape_class = str(region.metadata.get("shape_class", "")).lower()
    if shape_class in {"trapezoid", "convex_polygon", "large_convex", "rectangle"}:
        return bool(region.metadata.get("convex_region_decomposition") == "true" or shape_class != "rectangle")
    return False


def _build_polygon_pattern_for_axis(
    region: DecomposedRegion,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    axis: str,
    obstacle_field: ObstacleField | None = None,
    extra_metadata: Dict[str, str] | None = None,
) -> RegionCoveragePattern:
    x_min, y_min, x_max, y_max = region.bounds
    width = config.footprint.width_wf
    turn_pocket = (
        max(config.fleet.min_turn_radius * max(path_config.coverage_turn_pocket_scale, 0.0), 0.0)
        if obstacle_field is not None
        else 0.0
    )
    min_pass_length = _minimum_pass_length(config, path_config)
    passes: List[CoveragePass] = []
    sequence_index = 0
    previous_end: Pose2D | None = None
    clip_against_obstacles = obstacle_field is not None and region.metadata.get("convex_region_decomposition") != "true"

    if axis == "x":
        support_values = [point[1] for point in region.polygon]
        low, high = min(support_values), max(support_values)
        centers = _coverage_centers(low, high, width, _coverage_pass_count(max(high - low, 0.0), config, path_config))
        for center_y in centers:
            intervals = _polygon_axis_aligned_intervals("x", center_y, region.polygon)
            if clip_against_obstacles:
                clipped: List[Tuple[float, float]] = []
                for interval_start, interval_end in intervals:
                    clipped.extend(
                        clipped_axis_aligned_segments(
                            "x",
                            center_y,
                            interval_start,
                            interval_end,
                            obstacle_field,
                            width,
                            min_length=min_pass_length,
                        )
                    )
                intervals = clipped
            intervals = [_scan_interval_for_segment(a, b, turn_pocket, min_pass_length, path_config) for a, b in intervals]
            intervals = [(a, b) for a, b in intervals if b - a >= min_pass_length]
            for interval_start, interval_end, forward in _ordered_intervals(intervals, axis, center_y, previous_end, sequence_index):
                if forward:
                    start = Pose2D(interval_start, center_y, 0.0)
                    end = Pose2D(interval_end, center_y, 0.0)
                else:
                    start = Pose2D(interval_end, center_y, math.pi)
                    end = Pose2D(interval_start, center_y, math.pi)
                passes.append(
                    CoveragePass(
                        pass_id=f"{region.region_id}_x_pass_{sequence_index}",
                        region_id=region.region_id,
                        sequence_index=sequence_index,
                        scan_axis="x",
                        start_pose=start,
                        end_pose=end,
                        center_coordinate=center_y,
                        width=width,
                        length=max(abs(interval_end - interval_start), 0.0),
                    )
                )
                previous_end = end
                sequence_index += 1
    else:
        support_values = [point[0] for point in region.polygon]
        low, high = min(support_values), max(support_values)
        centers = _coverage_centers(low, high, width, _coverage_pass_count(max(high - low, 0.0), config, path_config))
        for center_x in centers:
            intervals = _polygon_axis_aligned_intervals("y", center_x, region.polygon)
            if clip_against_obstacles:
                clipped = []
                for interval_start, interval_end in intervals:
                    clipped.extend(
                        clipped_axis_aligned_segments(
                            "y",
                            center_x,
                            interval_start,
                            interval_end,
                            obstacle_field,
                            width,
                            min_length=min_pass_length,
                        )
                    )
                intervals = clipped
            intervals = [_scan_interval_for_segment(a, b, turn_pocket, min_pass_length, path_config) for a, b in intervals]
            intervals = [(a, b) for a, b in intervals if b - a >= min_pass_length]
            for interval_start, interval_end, forward in _ordered_intervals(intervals, axis, center_x, previous_end, sequence_index):
                if forward:
                    start = Pose2D(center_x, interval_start, math.pi / 2.0)
                    end = Pose2D(center_x, interval_end, math.pi / 2.0)
                else:
                    start = Pose2D(center_x, interval_end, -math.pi / 2.0)
                    end = Pose2D(center_x, interval_start, -math.pi / 2.0)
                passes.append(
                    CoveragePass(
                        pass_id=f"{region.region_id}_y_pass_{sequence_index}",
                        region_id=region.region_id,
                        sequence_index=sequence_index,
                        scan_axis="y",
                        start_pose=start,
                        end_pose=end,
                        center_coordinate=center_x,
                        width=width,
                        length=max(abs(interval_end - interval_start), 0.0),
                    )
                )
                previous_end = end
                sequence_index += 1

    passes, retraction_metadata = _apply_adaptive_pass_retraction(
        passes,
        config,
        path_config,
        obstacle_field,
        min_pass_length,
    )
    passes, endpoint_retraction_metadata = _apply_boundary_endpoint_retraction(
        passes,
        config,
        path_config,
        min_pass_length,
    )
    return _finalize_pattern(
        region,
        config,
        axis,
        passes,
        obstacle_field,
        source="convex_polygon_boustrophedon_candidate",
        extra_metadata={
            **retraction_metadata,
            **endpoint_retraction_metadata,
            "shape_class": str(region.metadata.get("shape_class", "convex_polygon")),
            "dominant_scan_axis": str(region.metadata.get("dominant_scan_axis", axis)),
            "support_span": str(region.metadata.get("support_span", "")),
            "true_polygon_intersections": "true",
            "region_bounds": f"{x_min:.6f},{y_min:.6f},{x_max:.6f},{y_max:.6f}",
            **(extra_metadata or {}),
        },
    )


def _build_oriented_polygon_pattern_for_angle(
    region: DecomposedRegion,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    angle: float,
    obstacle_field: ObstacleField | None,
    axis_label: str,
    scan_axis_source: str,
    support_span: float | None,
    angle_candidate_count: int,
) -> RegionCoveragePattern:
    x_min, y_min, x_max, y_max = region.bounds
    width = config.footprint.width_wf
    turn_pocket = (
        max(config.fleet.min_turn_radius * max(path_config.coverage_turn_pocket_scale, 0.0), 0.0)
        if obstacle_field is not None
        else 0.0
    )
    min_pass_length = _minimum_pass_length(config, path_config)
    angle = _normalize_angle_pi(angle)
    u_vec = (math.cos(angle), math.sin(angle))
    v_vec = (-math.sin(angle), math.cos(angle))
    local_polygon = [_project_oriented(point, u_vec, v_vec) for point in region.polygon]
    if len(local_polygon) < 3:
        return _finalize_pattern(
            region,
            config,
            axis_label,
            [],
            obstacle_field,
            source="oriented_polygon_boustrophedon_candidate",
            extra_metadata=_scan_axis_metadata(axis_label, scan_axis_source, support_span, angle_candidate_count, angle),
        )

    v_values = [point[1] for point in local_polygon]
    v_min, v_max = min(v_values), max(v_values)
    centers = _coverage_centers(v_min, v_max, width, _coverage_pass_count(max(v_max - v_min, 0.0), config, path_config))
    passes: List[CoveragePass] = []
    sequence_index = 0
    previous_end: Pose2D | None = None
    for center_v in centers:
        intervals = _polygon_local_line_intervals(center_v, local_polygon)
        intervals = [_scan_interval_for_segment(a, b, turn_pocket, min_pass_length, path_config) for a, b in intervals]
        intervals = [(a, b) for a, b in intervals if b - a >= min_pass_length]
        for interval_start, interval_end, forward in _ordered_oriented_intervals(
            intervals,
            center_v,
            u_vec,
            v_vec,
            previous_end,
            sequence_index,
        ):
            heading = angle if forward else wrap_angle(angle + math.pi)
            start_u, end_u = (interval_start, interval_end) if forward else (interval_end, interval_start)
            start_point = _unproject_oriented(start_u, center_v, u_vec, v_vec)
            end_point = _unproject_oriented(end_u, center_v, u_vec, v_vec)
            start = Pose2D(start_point[0], start_point[1], heading)
            end = Pose2D(end_point[0], end_point[1], heading)
            passes.append(
                CoveragePass(
                    pass_id=f"{region.region_id}_theta_{int(round(math.degrees(angle) * 1000.0))}_pass_{sequence_index}",
                    region_id=region.region_id,
                    sequence_index=sequence_index,
                    scan_axis=axis_label,
                    start_pose=start,
                    end_pose=end,
                    center_coordinate=center_v,
                    width=width,
                    length=max(abs(interval_end - interval_start), 0.0),
                )
            )
            previous_end = end
            sequence_index += 1

    passes, retraction_metadata = _apply_adaptive_pass_retraction(
        passes,
        config,
        path_config,
        obstacle_field,
        min_pass_length,
    )
    passes, endpoint_retraction_metadata = _apply_boundary_endpoint_retraction(
        passes,
        config,
        path_config,
        min_pass_length,
    )
    return _finalize_pattern(
        region,
        config,
        axis_label,
        passes,
        obstacle_field,
        source="oriented_polygon_boustrophedon_candidate",
        extra_metadata={
            **retraction_metadata,
            **endpoint_retraction_metadata,
            **_scan_axis_metadata(axis_label, scan_axis_source, support_span, angle_candidate_count, angle),
            "shape_class": str(region.metadata.get("shape_class", "convex_polygon")),
            "dominant_scan_axis": str(region.metadata.get("dominant_scan_axis", axis_label)),
            "true_polygon_intersections": "true",
            "oriented_polygon_intersections": "true",
            "region_bounds": f"{x_min:.6f},{y_min:.6f},{x_max:.6f},{y_max:.6f}",
        },
    )


def _project_oriented(
    point: Tuple[float, float],
    u_vec: Tuple[float, float],
    v_vec: Tuple[float, float],
) -> Tuple[float, float]:
    return point[0] * u_vec[0] + point[1] * u_vec[1], point[0] * v_vec[0] + point[1] * v_vec[1]


def _unproject_oriented(
    u_value: float,
    v_value: float,
    u_vec: Tuple[float, float],
    v_vec: Tuple[float, float],
) -> Tuple[float, float]:
    return u_value * u_vec[0] + v_value * v_vec[0], u_value * u_vec[1] + v_value * v_vec[1]


def _polygon_local_line_intervals(
    fixed_v: float,
    local_polygon: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    if len(local_polygon) < 3:
        return []
    intersections: List[float] = []
    for idx, start in enumerate(local_polygon):
        end = local_polygon[(idx + 1) % len(local_polygon)]
        u0, v0 = start
        u1, v1 = end
        if abs(v1 - v0) <= 1e-12:
            if abs(fixed_v - v0) <= 1e-9:
                intersections.extend([u0, u1])
            continue
        low_v, high_v = min(v0, v1), max(v0, v1)
        if low_v - 1e-9 <= fixed_v <= high_v + 1e-9:
            alpha = (fixed_v - v0) / (v1 - v0)
            if -1e-9 <= alpha <= 1.0 + 1e-9:
                intersections.append(u0 + alpha * (u1 - u0))
    values = sorted(set(round(value, 9) for value in intersections))
    intervals: List[Tuple[float, float]] = []
    for idx in range(len(values) - 1):
        low = float(values[idx])
        high = float(values[idx + 1])
        if high <= low + 1e-9:
            continue
        midpoint = ((low + high) / 2.0, fixed_v)
        if point_in_polygon(midpoint, local_polygon):
            intervals.append((low, high))
    return _merge_intervals(intervals)


def _ordered_oriented_intervals(
    intervals: List[Tuple[float, float]],
    fixed_v: float,
    u_vec: Tuple[float, float],
    v_vec: Tuple[float, float],
    previous_end: Pose2D | None,
    sequence_index: int,
) -> List[Tuple[float, float, bool]]:
    if not intervals:
        return []
    if previous_end is None:
        forward = sequence_index % 2 == 0
        ordered = intervals if forward else list(reversed(intervals))
        return [(low, high, forward) for low, high in ordered]
    remaining = list(intervals)
    result: List[Tuple[float, float, bool]] = []
    current = previous_end
    while remaining:
        best_idx = 0
        best_forward = True
        best_cost = float("inf")
        for idx, (low, high) in enumerate(remaining):
            start_forward = _unproject_oriented(low, fixed_v, u_vec, v_vec)
            start_reverse = _unproject_oriented(high, fixed_v, u_vec, v_vec)
            forward_cost = math.hypot(current.x - start_forward[0], current.y - start_forward[1])
            reverse_cost = math.hypot(current.x - start_reverse[0], current.y - start_reverse[1])
            if forward_cost < best_cost:
                best_idx = idx
                best_forward = True
                best_cost = forward_cost
            if reverse_cost < best_cost:
                best_idx = idx
                best_forward = False
                best_cost = reverse_cost
        low, high = remaining.pop(best_idx)
        result.append((low, high, best_forward))
        end_u = high if best_forward else low
        end_point = _unproject_oriented(end_u, fixed_v, u_vec, v_vec)
        current = Pose2D(end_point[0], end_point[1], 0.0)
    return result


def _build_composite_pattern_for_axis(
    region: DecomposedRegion,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    axis: str,
    obstacle_field: ObstacleField | None,
    extra_metadata: Dict[str, str] | None = None,
) -> RegionCoveragePattern:
    x_min, y_min, x_max, y_max = region.bounds
    width = config.footprint.width_wf
    turn_pocket = (
        max(config.fleet.min_turn_radius * max(path_config.coverage_turn_pocket_scale, 0.0), 0.0)
        if obstacle_field is not None
        else 0.0
    )
    min_pass_length = _minimum_pass_length(config, path_config)
    member_cells = list(getattr(region, "member_cells", []) or [])
    passes: List[CoveragePass] = []
    sequence_index = 0
    previous_end: Pose2D | None = None

    if axis == "x":
        cross_width = max(y_max - y_min, 0.0)
        pass_count = _coverage_pass_count(cross_width, config, path_config)
        centers = _coverage_centers(y_min, y_max, width, pass_count)
        for center_y in centers:
            intervals = _composite_axis_aligned_segments(
                member_cells,
                axis="x",
                fixed_coord=center_y,
                obstacle_field=obstacle_field,
                footprint_width=width,
                min_length=min_pass_length,
            )
            intervals = [_scan_interval_for_segment(a, b, turn_pocket, min_pass_length, path_config) for a, b in intervals]
            intervals = [(a, b) for a, b in intervals if b - a >= min_pass_length]
            for interval_start, interval_end, forward in _ordered_intervals(intervals, axis, center_y, previous_end, sequence_index):
                if forward:
                    start = Pose2D(interval_start, center_y, 0.0)
                    end = Pose2D(interval_end, center_y, 0.0)
                else:
                    start = Pose2D(interval_end, center_y, math.pi)
                    end = Pose2D(interval_start, center_y, math.pi)
                passes.append(
                    CoveragePass(
                        pass_id=f"{region.region_id}_x_pass_{sequence_index}",
                        region_id=region.region_id,
                        sequence_index=sequence_index,
                        scan_axis="x",
                        start_pose=start,
                        end_pose=end,
                        center_coordinate=center_y,
                        width=width,
                        length=max(abs(interval_end - interval_start), 0.0),
                    )
                )
                previous_end = end
                sequence_index += 1
    else:
        cross_width = max(x_max - x_min, 0.0)
        pass_count = _coverage_pass_count(cross_width, config, path_config)
        centers = _coverage_centers(x_min, x_max, width, pass_count)
        for center_x in centers:
            intervals = _composite_axis_aligned_segments(
                member_cells,
                axis="y",
                fixed_coord=center_x,
                obstacle_field=obstacle_field,
                footprint_width=width,
                min_length=min_pass_length,
            )
            intervals = [_scan_interval_for_segment(a, b, turn_pocket, min_pass_length, path_config) for a, b in intervals]
            intervals = [(a, b) for a, b in intervals if b - a >= min_pass_length]
            for interval_start, interval_end, forward in _ordered_intervals(intervals, axis, center_x, previous_end, sequence_index):
                if forward:
                    start = Pose2D(center_x, interval_start, math.pi / 2.0)
                    end = Pose2D(center_x, interval_end, math.pi / 2.0)
                else:
                    start = Pose2D(center_x, interval_end, -math.pi / 2.0)
                    end = Pose2D(center_x, interval_start, -math.pi / 2.0)
                passes.append(
                    CoveragePass(
                        pass_id=f"{region.region_id}_y_pass_{sequence_index}",
                        region_id=region.region_id,
                        sequence_index=sequence_index,
                        scan_axis="y",
                        start_pose=start,
                        end_pose=end,
                        center_coordinate=center_x,
                        width=width,
                        length=max(abs(interval_end - interval_start), 0.0),
                    )
                )
                previous_end = end
                sequence_index += 1

    passes, retraction_metadata = _apply_adaptive_pass_retraction(
        passes,
        config,
        path_config,
        obstacle_field,
        min_pass_length,
    )
    passes, endpoint_retraction_metadata = _apply_boundary_endpoint_retraction(
        passes,
        config,
        path_config,
        min_pass_length,
    )
    retraction_metadata = {**retraction_metadata, **endpoint_retraction_metadata, **(extra_metadata or {})}
    return _finalize_pattern(
        region,
        config,
        axis,
        passes,
        obstacle_field,
        source="composite_boustrophedon_candidate",
        extra_metadata=retraction_metadata,
    )


def _finalize_pattern(
    region: DecomposedRegion,
    config: PlannerConfig,
    axis: str,
    passes: List[CoveragePass],
    obstacle_field: ObstacleField | None,
    source: str,
    extra_metadata: Dict[str, str] | None = None,
) -> RegionCoveragePattern:
    x_min, y_min, x_max, y_max = region.bounds
    coverage_length = sum(item.length for item in passes)
    region_area = max(float(region.area), 1e-9)
    estimated_coverage_fraction = min(1.0, max(0.0, coverage_length * config.footprint.width_wf / region_area))
    turn_length = 0.0
    turn_angle = 0.0
    max_curvature = 0.0
    feasible = True
    collision_free = True
    boundary_safe = True
    for coverage_pass in passes:
        if not point_in_mission_bounds((coverage_pass.start_pose.x, coverage_pass.start_pose.y), config) or not point_in_mission_bounds(
            (coverage_pass.end_pose.x, coverage_pass.end_pose.y),
            config,
        ):
            feasible = False
            boundary_safe = False
    if obstacle_field is not None:
        for coverage_pass in passes:
            if sampled_segment_footprint_collides(
                coverage_pass.start_pose,
                coverage_pass.end_pose,
                config.footprint.length_lf,
                config.footprint.width_wf,
                obstacle_field,
                sample_spacing=max(config.footprint.width_wf / 2.0, 1e-6),
                inflated=False,
            ):
                feasible = False
                collision_free = False
                break
    for current_pass, next_pass in zip(passes[:-1], passes[1:]):
        transition = dubins_shortest_path(current_pass.end_pose, next_pass.start_pose, config.fleet.min_turn_radius)
        turn_length += transition.total_length
        max_curvature = max(max_curvature, 1.0 / config.fleet.min_turn_radius)
        turn_angle += _dubins_turn_angle(transition.segment_lengths, transition.modes, config.fleet.min_turn_radius)
        if max_curvature > 1.0 / config.fleet.min_turn_radius + 1e-3:
            feasible = False
        points, _, _ = sample_dubins_path(
            transition,
            step_size=max(config.fleet.min_turn_radius / 8.0, 0.25),
        )
        if polyline_out_of_mission_bounds(points, config):
            feasible = False
            boundary_safe = False
        if obstacle_field is not None and polyline_collides_with_obstacles(points, obstacle_field, inflated=True):
            feasible = False
            collision_free = False

    cover_speed = max(config.fleet.cover_speed, 1e-6)
    turn_speed = max(min(config.fleet.turn_speed_max, config.fleet.cruise_speed), 1e-6)
    yaw_rate = max(config.fleet.turn_speed_max / max(config.fleet.min_turn_radius, 1e-6), 1e-6)
    estimated_time = coverage_length / cover_speed + turn_length / turn_speed + turn_angle / yaw_rate
    entry = passes[0].start_pose if passes else Pose2D(region.center[0], region.center[1], 0.0)
    exit_pose = passes[-1].end_pose if passes else entry
    return RegionCoveragePattern(
        pattern_id=f"{region.region_id}_pattern_{axis}",
        region_id=region.region_id,
        scan_axis=axis,
        passes=passes,
        entry_pose=entry,
        exit_pose=exit_pose,
        coverage_length=coverage_length,
        turn_length=turn_length,
        turn_angle=turn_angle,
        total_length=coverage_length + turn_length,
        estimated_time=estimated_time,
        max_curvature=max_curvature,
        feasible=feasible,
        metadata={
            "pass_count": str(len(passes)),
            "source": source,
            "collision_free": str(collision_free).lower(),
            "boundary_safe": str(boundary_safe).lower(),
            "static_obstacle_aware": str(obstacle_field is not None).lower(),
            "region_bounds": f"{x_min:.6f},{y_min:.6f},{x_max:.6f},{y_max:.6f}",
            "region_area": f"{region_area:.6f}",
            "estimated_region_coverage_fraction": f"{estimated_coverage_fraction:.6f}",
            "is_composite": str(bool(getattr(region, "member_cells", None))).lower(),
            "source_cell_count": str(len(getattr(region, "member_cells", []) or [])),
            "shape_class": str(region.metadata.get("shape_class", "")),
            "dominant_scan_axis": str(region.metadata.get("dominant_scan_axis", axis)),
            "support_span": str(region.metadata.get("support_span", "")),
            **(extra_metadata or {}),
        },
    )


def _polygon_axis_aligned_intervals(
    axis: str,
    fixed_coord: float,
    polygon: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    if len(polygon) < 3:
        return []
    intersections: List[float] = []
    for idx, start in enumerate(polygon):
        end = polygon[(idx + 1) % len(polygon)]
        x0, y0 = start
        x1, y1 = end
        if axis == "x":
            if abs(y1 - y0) <= 1e-12:
                if abs(fixed_coord - y0) <= 1e-9:
                    intersections.extend([x0, x1])
                continue
            low_y, high_y = min(y0, y1), max(y0, y1)
            if low_y - 1e-9 <= fixed_coord <= high_y + 1e-9:
                alpha = (fixed_coord - y0) / (y1 - y0)
                if -1e-9 <= alpha <= 1.0 + 1e-9:
                    intersections.append(x0 + alpha * (x1 - x0))
        else:
            if abs(x1 - x0) <= 1e-12:
                if abs(fixed_coord - x0) <= 1e-9:
                    intersections.extend([y0, y1])
                continue
            low_x, high_x = min(x0, x1), max(x0, x1)
            if low_x - 1e-9 <= fixed_coord <= high_x + 1e-9:
                alpha = (fixed_coord - x0) / (x1 - x0)
                if -1e-9 <= alpha <= 1.0 + 1e-9:
                    intersections.append(y0 + alpha * (y1 - y0))
    values = sorted(set(round(value, 9) for value in intersections))
    intervals: List[Tuple[float, float]] = []
    for idx in range(len(values) - 1):
        low = float(values[idx])
        high = float(values[idx + 1])
        if high <= low + 1e-9:
            continue
        midpoint_value = (low + high) / 2.0
        midpoint = (midpoint_value, fixed_coord) if axis == "x" else (fixed_coord, midpoint_value)
        if point_in_polygon(midpoint, polygon):
            intervals.append((low, high))
    return _merge_intervals(intervals)


def _composite_axis_aligned_segments(
    member_cells: List[FreeSpaceCell],
    axis: str,
    fixed_coord: float,
    obstacle_field: ObstacleField | None,
    footprint_width: float,
    min_length: float,
) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    for cell in member_cells:
        x0, y0, x1, y1 = cell.bounds
        if axis == "x":
            if y0 - 1e-9 <= fixed_coord <= y1 + 1e-9:
                intervals.append((x0, x1))
        else:
            if x0 - 1e-9 <= fixed_coord <= x1 + 1e-9:
                intervals.append((y0, y1))
    intervals = _merge_intervals(intervals)
    if obstacle_field is not None:
        clipped: List[Tuple[float, float]] = []
        for low, high in intervals:
            clipped.extend(
                clipped_axis_aligned_segments(
                    axis,
                    fixed_coord,
                    low,
                    high,
                    obstacle_field,
                    footprint_width,
                    min_length=min_length,
                )
            )
        intervals = _merge_intervals(clipped)
    return [(low, high) for low, high in intervals if high - low >= min_length]


def _merge_intervals(intervals: List[Tuple[float, float]], tol: float = 1e-9) -> List[Tuple[float, float]]:
    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals if abs(b - a) > tol)
    if not ordered:
        return []
    merged = [ordered[0]]
    for low, high in ordered[1:]:
        prev_low, prev_high = merged[-1]
        if low <= prev_high + tol:
            merged[-1] = (prev_low, max(prev_high, high))
        else:
            merged.append((low, high))
    return merged


def _ordered_intervals(
    intervals: List[Tuple[float, float]],
    axis: str,
    fixed_coord: float,
    previous_end: Pose2D | None,
    sequence_index: int,
) -> List[Tuple[float, float, bool]]:
    if not intervals:
        return []
    if previous_end is None:
        forward = sequence_index % 2 == 0
        ordered = intervals if forward else list(reversed(intervals))
        return [(low, high, forward) for low, high in ordered]
    remaining = list(intervals)
    result: List[Tuple[float, float, bool]] = []
    current = previous_end
    while remaining:
        best_idx = 0
        best_forward = True
        best_cost = float("inf")
        for idx, (low, high) in enumerate(remaining):
            if axis == "x":
                start_forward = (low, fixed_coord)
                start_reverse = (high, fixed_coord)
            else:
                start_forward = (fixed_coord, low)
                start_reverse = (fixed_coord, high)
            forward_cost = math.hypot(current.x - start_forward[0], current.y - start_forward[1])
            reverse_cost = math.hypot(current.x - start_reverse[0], current.y - start_reverse[1])
            if forward_cost < best_cost:
                best_idx = idx
                best_forward = True
                best_cost = forward_cost
            if reverse_cost < best_cost:
                best_idx = idx
                best_forward = False
                best_cost = reverse_cost
        low, high = remaining.pop(best_idx)
        result.append((low, high, best_forward))
        if axis == "x":
            current = Pose2D(high if best_forward else low, fixed_coord, 0.0 if best_forward else math.pi)
        else:
            current = Pose2D(fixed_coord, high if best_forward else low, math.pi / 2.0 if best_forward else -math.pi / 2.0)
    return result


def _coverage_pass_count(cross_width: float, config: PlannerConfig, path_config: PathPlanningConfig) -> int:
    if path_config.enable_adaptive_pass_retraction:
        return max(1, int(math.ceil(max(cross_width, 0.0) / max(config.footprint.width_wf, 1e-6))))
    model = RectangularCoverageModel.from_config(config)
    return 1 if cross_width <= model.width else int(math.ceil((cross_width - model.width) / model.strip_spacing) + 1)


def _minimum_pass_length(config: PlannerConfig, path_config: PathPlanningConfig) -> float:
    return max(
        config.footprint.width_wf * 0.25,
        config.footprint.length_lf * max(path_config.retraction_min_pass_length_factor, 0.0),
        1e-6,
    )


def _initial_scan_interval(
    low: float,
    high: float,
    boundary_margin: float,
    turn_pocket: float,
    min_length: float,
    path_config: PathPlanningConfig,
) -> Tuple[float, float]:
    if path_config.enable_adaptive_pass_retraction:
        return low, high
    return _buffered_interval(low, high, max(boundary_margin, turn_pocket), min_length)


def _scan_interval_for_segment(
    low: float,
    high: float,
    turn_pocket: float,
    min_length: float,
    path_config: PathPlanningConfig,
) -> Tuple[float, float]:
    if path_config.enable_adaptive_pass_retraction:
        return low, high
    return _buffered_interval(low, high, turn_pocket, min_length)


def _apply_adaptive_pass_retraction(
    passes: List[CoveragePass],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    min_pass_length: float,
) -> Tuple[List[CoveragePass], Dict[str, str]]:
    if not path_config.enable_adaptive_pass_retraction:
        return passes, _retraction_metadata(False, 0, 0.0, 0.0, 0, 0, "")
    if len(passes) <= 1:
        return passes, _retraction_metadata(True, 0, 0.0, 0.0, 0, 0, "")

    updated = list(passes)
    retracted_pass_ids = set()
    total_retraction = 0.0
    max_retraction = 0.0
    extended_count = 0
    failed_count = 0
    reasons: List[str] = []

    for idx in range(len(updated) - 1):
        current_pass = updated[idx]
        next_pass = updated[idx + 1]
        result = _find_pair_retraction(
            current_pass,
            next_pass,
            config,
            path_config,
            obstacle_field,
            min_pass_length,
        )
        if result is None:
            failed_count += 1
            reasons.append(f"{current_pass.pass_id}->{next_pass.pass_id}:no_feasible_retraction")
            continue
        exit_retraction, entry_retraction, extended = result
        if exit_retraction <= 1e-9 and entry_retraction <= 1e-9:
            continue
        current_pass, next_pass = _retract_pass_pair(current_pass, next_pass, exit_retraction, entry_retraction)
        updated[idx] = current_pass
        updated[idx + 1] = next_pass
        if exit_retraction > 1e-9:
            retracted_pass_ids.add(current_pass.pass_id)
        if entry_retraction > 1e-9:
            retracted_pass_ids.add(next_pass.pass_id)
        total_retraction += exit_retraction + entry_retraction
        max_retraction = max(max_retraction, exit_retraction, entry_retraction)
        if extended:
            extended_count += 1
        reasons.append(f"{current_pass.pass_id}->{next_pass.pass_id}:uturn_boundary_or_obstacle")

    return updated, _retraction_metadata(
        True,
        len(retracted_pass_ids),
        total_retraction,
        max_retraction,
        failed_count,
        extended_count,
        ",".join(reasons[:6]),
    )


def _apply_boundary_endpoint_retraction(
    passes: List[CoveragePass],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    min_pass_length: float,
) -> Tuple[List[CoveragePass], Dict[str, str]]:
    if not path_config.enable_adaptive_pass_retraction or not passes:
        return passes, _endpoint_retraction_metadata(0, 0.0, 0.0, "")
    updated = list(passes)
    ratio_limit = max(0.0, min(1.0, path_config.max_pass_retraction_ratio))
    target_clearance = max(config.fleet.min_turn_radius, 0.0)
    total = 0.0
    max_value = 0.0
    touched: set[str] = set()
    reasons: List[str] = []

    first = updated[0]
    entry_clearance = _mission_boundary_clearance(first.start_pose, config)
    if entry_clearance + 1e-9 < target_clearance:
        distance = min(
            target_clearance - entry_clearance,
            max(_pass_length(first) - min_pass_length, 0.0),
            _pass_length(first) * ratio_limit,
        )
        if distance > 1e-9:
            updated[0] = _with_retracted_entry(first, distance)
            total += distance
            max_value = max(max_value, distance)
            touched.add(first.pass_id)
            reasons.append(f"{first.pass_id}:entry_boundary")

    last = updated[-1]
    exit_clearance = _mission_boundary_clearance(last.end_pose, config)
    if exit_clearance + 1e-9 < target_clearance:
        distance = min(
            target_clearance - exit_clearance,
            max(_pass_length(last) - min_pass_length, 0.0),
            _pass_length(last) * ratio_limit,
        )
        if distance > 1e-9:
            updated[-1] = _with_retracted_exit(last, distance)
            total += distance
            max_value = max(max_value, distance)
            touched.add(last.pass_id)
            reasons.append(f"{last.pass_id}:exit_boundary")
    return updated, _endpoint_retraction_metadata(len(touched), total, max_value, ",".join(reasons[:4]))


def _endpoint_retraction_metadata(
    retracted_pass_count: int,
    total_retraction: float,
    max_retraction: float,
    reason: str,
) -> Dict[str, str]:
    return {
        "endpoint_retracted_pass_count": str(int(retracted_pass_count)),
        "endpoint_total_retraction_length": f"{total_retraction:.6f}",
        "endpoint_max_retraction_length": f"{max_retraction:.6f}",
        "endpoint_retraction_reason": reason,
    }


def _mission_boundary_clearance(pose: Pose2D, config: PlannerConfig) -> float:
    return min(
        pose.x,
        config.mission.area_length_x - pose.x,
        pose.y,
        config.mission.area_length_y - pose.y,
    )


def _retraction_metadata(
    enabled: bool,
    retracted_pass_count: int,
    total_retraction: float,
    max_retraction: float,
    failed_count: int,
    extended_count: int,
    reason: str,
) -> Dict[str, str]:
    return {
        "boundary_retraction_mode": "adaptive" if enabled else "fixed_pocket",
        "retracted_pass_count": str(int(retracted_pass_count)),
        "total_retraction_length": f"{total_retraction:.6f}",
        "max_retraction_length": f"{max_retraction:.6f}",
        "retraction_failed_count": str(int(failed_count)),
        "retraction_extended_count": str(int(extended_count)),
        "retraction_reason": reason,
    }


def _find_pair_retraction(
    current_pass: CoveragePass,
    next_pass: CoveragePass,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    min_pass_length: float,
) -> Tuple[float, float, bool] | None:
    if _pass_pair_valid(current_pass, next_pass, config, obstacle_field):
        return 0.0, 0.0, False

    ratio_limit = max(0.0, min(1.0, path_config.max_pass_retraction_ratio))
    current_length = _pass_length(current_pass)
    next_length = _pass_length(next_pass)
    max_exit = min(max(current_length - min_pass_length, 0.0), current_length * ratio_limit)
    max_entry = min(max(next_length - min_pass_length, 0.0), next_length * ratio_limit)
    if max_exit <= 1e-9 and max_entry <= 1e-9:
        return None

    nominal = min(config.footprint.length_lf / 2.0, min(current_length, next_length) / 2.0)
    nominal_exit = min(max_exit, nominal)
    nominal_entry = min(max_entry, nominal)
    candidates = [
        (nominal_exit, nominal_entry, False),
        (max_exit, max_entry, True),
        (max_exit, nominal_entry, True),
        (nominal_exit, max_entry, True),
        (max_exit, 0.0, True),
        (0.0, max_entry, True),
    ]

    best: Tuple[float, float, bool] | None = None
    best_score = float("inf")
    for exit_limit, entry_limit, extended in candidates:
        if exit_limit <= 1e-9 and entry_limit <= 1e-9:
            continue
        trial_current, trial_next = _retract_pass_pair(current_pass, next_pass, exit_limit, entry_limit)
        if not _pass_pair_valid(trial_current, trial_next, config, obstacle_field):
            continue
        low = 0.0
        high = 1.0
        for _ in range(max(int(path_config.retraction_search_iterations), 1)):
            mid = (low + high) / 2.0
            mid_current, mid_next = _retract_pass_pair(current_pass, next_pass, exit_limit * mid, entry_limit * mid)
            if _pass_pair_valid(mid_current, mid_next, config, obstacle_field):
                high = mid
            else:
                low = mid
        exit_retraction = exit_limit * high
        entry_retraction = entry_limit * high
        score = exit_retraction + entry_retraction
        if score < best_score:
            best = (exit_retraction, entry_retraction, extended)
            best_score = score
    return best


def _pass_pair_valid(
    current_pass: CoveragePass,
    next_pass: CoveragePass,
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
) -> bool:
    for coverage_pass in (current_pass, next_pass):
        if _pass_length(coverage_pass) <= 1e-9:
            return False
        if not point_in_mission_bounds((coverage_pass.start_pose.x, coverage_pass.start_pose.y), config):
            return False
        if not point_in_mission_bounds((coverage_pass.end_pose.x, coverage_pass.end_pose.y), config):
            return False
        if obstacle_field is not None and sampled_segment_footprint_collides(
            coverage_pass.start_pose,
            coverage_pass.end_pose,
            config.footprint.length_lf,
            config.footprint.width_wf,
            obstacle_field,
            sample_spacing=max(config.footprint.width_wf / 2.0, 1e-6),
            inflated=False,
        ):
            return False
    transition = dubins_shortest_path(current_pass.end_pose, next_pass.start_pose, config.fleet.min_turn_radius)
    points, _, _ = sample_dubins_path(
        transition,
        step_size=max(config.fleet.min_turn_radius / 8.0, 0.25),
    )
    if polyline_out_of_mission_bounds(points, config):
        return False
    if obstacle_field is not None and polyline_collides_with_obstacles(points, obstacle_field, inflated=True):
        return False
    return True


def _retract_pass_pair(
    current_pass: CoveragePass,
    next_pass: CoveragePass,
    exit_retraction: float,
    entry_retraction: float,
) -> Tuple[CoveragePass, CoveragePass]:
    return (
        _with_retracted_exit(current_pass, max(exit_retraction, 0.0)),
        _with_retracted_entry(next_pass, max(entry_retraction, 0.0)),
    )


def _with_retracted_exit(coverage_pass: CoveragePass, distance: float) -> CoveragePass:
    start = coverage_pass.start_pose
    end = coverage_pass.end_pose
    length = _pass_length(coverage_pass)
    if length <= 1e-9 or distance <= 1e-9:
        return coverage_pass
    ratio = min(distance / length, 1.0)
    new_end = Pose2D(
        end.x + (start.x - end.x) * ratio,
        end.y + (start.y - end.y) * ratio,
        end.psi,
    )
    return _replace_pass_endpoint(coverage_pass, start, new_end)


def _with_retracted_entry(coverage_pass: CoveragePass, distance: float) -> CoveragePass:
    start = coverage_pass.start_pose
    end = coverage_pass.end_pose
    length = _pass_length(coverage_pass)
    if length <= 1e-9 or distance <= 1e-9:
        return coverage_pass
    ratio = min(distance / length, 1.0)
    new_start = Pose2D(
        start.x + (end.x - start.x) * ratio,
        start.y + (end.y - start.y) * ratio,
        start.psi,
    )
    return _replace_pass_endpoint(coverage_pass, new_start, end)


def _replace_pass_endpoint(coverage_pass: CoveragePass, start: Pose2D, end: Pose2D) -> CoveragePass:
    length = math.hypot(end.x - start.x, end.y - start.y)
    return replace(
        coverage_pass,
        start_pose=start,
        end_pose=end,
        length=length,
    )


def _pass_length(coverage_pass: CoveragePass) -> float:
    return math.hypot(
        coverage_pass.end_pose.x - coverage_pass.start_pose.x,
        coverage_pass.end_pose.y - coverage_pass.start_pose.y,
    )


def _coverage_centers(low: float, high: float, footprint_width: float, pass_count: int) -> List[float]:
    if pass_count <= 1:
        return [(low + high) / 2.0]
    first = low + footprint_width / 2.0
    last = high - footprint_width / 2.0
    if last < first:
        return [(low + high) / 2.0]
    return [first + (last - first) * idx / max(pass_count - 1, 1) for idx in range(pass_count)]


def _buffered_interval(low: float, high: float, desired_buffer: float, min_length: float) -> Tuple[float, float]:
    length = max(high - low, 0.0)
    if length <= min_length:
        midpoint = (low + high) / 2.0
        half = length / 2.0
        return midpoint - half, midpoint + half
    max_buffer = max((length - min_length) / 2.0, 0.0)
    buffer = min(max(desired_buffer, 0.0), max_buffer)
    return low + buffer, high - buffer


def _dubins_turn_angle(segment_lengths: Tuple[float, float, float], modes: Tuple[str, str, str], turn_radius: float) -> float:
    angle = 0.0
    for length, mode in zip(segment_lengths, modes):
        if mode in {"L", "R"}:
            angle += abs(length / max(turn_radius, 1e-6))
    return angle


def heading_change_between_patterns(first: RegionCoveragePattern, second: RegionCoveragePattern) -> float:
    return abs(wrap_angle(second.entry_pose.psi - first.exit_pose.psi))
