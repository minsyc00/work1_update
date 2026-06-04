from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

from ..dubins import dubins_shortest_path, sample_dubins_path
from ..geometry import wrap_angle
from ..schema import PlannerConfig, Pose2D
from .coverage import RectangularCoverageModel
from .obstacles import (
    clipped_axis_aligned_segments,
    point_in_mission_bounds,
    polyline_collides_with_obstacles,
    polyline_out_of_mission_bounds,
    sampled_segment_footprint_collides,
)
from .types import CoveragePass, DecomposedRegion, ObstacleField, PathPlanningConfig, RegionCoveragePattern


def candidate_scan_axes(region: DecomposedRegion, max_axes: int = 2) -> List[str]:
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
    for axis in candidate_scan_axes(region, path_config.max_candidate_axes):
        pattern = _build_pattern_for_axis(region, config, path_config, axis, obstacle_field)
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


def _build_pattern_for_axis(
    region: DecomposedRegion,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    axis: str,
    obstacle_field: ObstacleField | None = None,
) -> RegionCoveragePattern:
    model = RectangularCoverageModel.from_config(config)
    x_min, y_min, x_max, y_max = region.bounds
    lf = config.footprint.length_lf
    width = config.footprint.width_wf
    delta = model.strip_spacing
    margin_x = min(max(config.safety.boundary_margin_x, 0.0), max((x_max - x_min) / 2.0, 0.0))
    margin_y = min(max(config.safety.boundary_margin_y, 0.0), max((y_max - y_min) / 2.0, 0.0))
    turn_pocket = (
        max(config.fleet.min_turn_radius * max(path_config.coverage_turn_pocket_scale, 0.0), 0.0)
        if obstacle_field is not None
        else 0.0
    )
    min_pass_length = max(width * 0.25, 1e-6)

    if axis == "x":
        cross_width = max(y_max - y_min, 0.0)
        pass_count = 1 if cross_width <= width else int(math.ceil((cross_width - width) / delta) + 1)
        centers = _coverage_centers(y_min, y_max, width, pass_count)
        x0, x1 = _buffered_interval(x_min, x_max, max(margin_x, turn_pocket), min_pass_length)
        passes: List[CoveragePass] = []
        sequence_index = 0
        for idx, center_y in enumerate(centers):
            intervals = [(x0, x1)]
            if obstacle_field is not None:
                intervals = clipped_axis_aligned_segments("x", center_y, x0, x1, obstacle_field, width, min_length=min_pass_length)
            for interval_start, interval_end in intervals:
                interval_start, interval_end = _buffered_interval(interval_start, interval_end, turn_pocket, min_pass_length)
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
        pass_count = 1 if cross_width <= width else int(math.ceil((cross_width - width) / delta) + 1)
        centers = _coverage_centers(x_min, x_max, width, pass_count)
        y0, y1 = _buffered_interval(y_min, y_max, max(margin_y, turn_pocket), min_pass_length)
        passes = []
        sequence_index = 0
        for idx, center_x in enumerate(centers):
            intervals = [(y0, y1)]
            if obstacle_field is not None:
                intervals = clipped_axis_aligned_segments("y", center_x, y0, y1, obstacle_field, width, min_length=min_pass_length)
            for interval_start, interval_end in intervals:
                interval_start, interval_end = _buffered_interval(interval_start, interval_end, turn_pocket, min_pass_length)
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

    coverage_length = sum(item.length for item in passes)
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
        },
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
