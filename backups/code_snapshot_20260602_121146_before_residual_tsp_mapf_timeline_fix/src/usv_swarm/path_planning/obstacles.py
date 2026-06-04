from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from ..geometry import rotated_rectangle_mask
from ..schema import PlannerConfig, Pose2D
from .types import ObstacleField, PathPlanningConfig, StaticObstacle


Point = Tuple[float, float]
Bounds = Tuple[float, float, float, float]


def normalize_obstacle_field(
    static_obstacles: Sequence[StaticObstacle] | None,
    config: PlannerConfig,
    path_config: PathPlanningConfig | None = None,
) -> ObstacleField:
    path_config = path_config or PathPlanningConfig.from_planner_config(config)
    safety_margin = config.safety.d_safe + path_config.obstacle_inflation_extra
    footprint_margin = max(config.footprint.length_lf, config.footprint.width_wf) / 2.0
    inflation = safety_margin + footprint_margin
    normalized = [_normalized_obstacle(obstacle, 0.0, path_config) for obstacle in static_obstacles or []]
    inflated = [_normalized_obstacle(obstacle, inflation, path_config) for obstacle in static_obstacles or []]
    return ObstacleField(
        obstacles=normalized,
        inflated_obstacles=inflated,
        safety_margin=safety_margin,
        footprint_margin=footprint_margin,
        metadata={"inflation": f"{inflation:.6f}", "obstacle_count": str(len(normalized))},
    )


def rectangle_obstacle(
    obstacle_id: str,
    center: Point,
    width: float,
    height: float,
    psi: float = 0.0,
) -> StaticObstacle:
    return StaticObstacle(obstacle_id=obstacle_id, kind="rectangle", center=center, width=width, height=height, psi=psi)


def circle_obstacle(obstacle_id: str, center: Point, radius: float) -> StaticObstacle:
    return StaticObstacle(obstacle_id=obstacle_id, kind="circle", center=center, radius=radius)


def ellipse_obstacle(obstacle_id: str, center: Point, radii: Point, psi: float = 0.0) -> StaticObstacle:
    return StaticObstacle(obstacle_id=obstacle_id, kind="ellipse", center=center, radii=radii, psi=psi)


def polygon_obstacle(obstacle_id: str, polygon: Sequence[Point]) -> StaticObstacle:
    return StaticObstacle(obstacle_id=obstacle_id, kind="polygon", polygon=list(polygon))


def obstacle_bounds(obstacle: StaticObstacle) -> Bounds:
    if not obstacle.polygon:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [point[0] for point in obstacle.polygon]
    ys = [point[1] for point in obstacle.polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def point_in_any_obstacle(point: Point, field: ObstacleField, inflated: bool = True) -> bool:
    obstacles = field.inflated_obstacles if inflated else field.obstacles
    return any(point_in_polygon(point, obstacle.polygon) for obstacle in obstacles)


def segment_collides_with_obstacles(start: Point, end: Point, field: ObstacleField, inflated: bool = True) -> bool:
    obstacles = field.inflated_obstacles if inflated else field.obstacles
    return any(segment_intersects_polygon(start, end, obstacle.polygon) for obstacle in obstacles)


def polygon_collides_with_obstacles(polygon: Sequence[Point], field: ObstacleField, inflated: bool = True) -> bool:
    obstacles = field.inflated_obstacles if inflated else field.obstacles
    return any(polygons_intersect(polygon, obstacle.polygon) for obstacle in obstacles)


def pose_footprint_collides(
    pose: Pose2D,
    length: float,
    width: float,
    field: ObstacleField,
    inflated: bool = True,
) -> bool:
    return polygon_collides_with_obstacles(rotated_rectangle_polygon(pose.x, pose.y, pose.psi, length, width), field, inflated)


def sampled_segment_footprint_collides(
    start: Pose2D,
    end: Pose2D,
    length: float,
    width: float,
    field: ObstacleField,
    sample_spacing: float,
    inflated: bool = False,
) -> bool:
    distance = math.hypot(end.x - start.x, end.y - start.y)
    count = max(2, int(math.ceil(distance / max(sample_spacing, 1e-6))) + 1)
    heading = math.atan2(end.y - start.y, end.x - start.x) if distance > 1e-9 else start.psi
    for idx in range(count):
        alpha = idx / max(count - 1, 1)
        pose = Pose2D(
            start.x + alpha * (end.x - start.x),
            start.y + alpha * (end.y - start.y),
            heading,
        )
        if pose_footprint_collides(pose, length, width, field, inflated=inflated):
            return True
    return False


def path_segment_spec_collides(segment, field: ObstacleField) -> bool:
    waypoints = getattr(segment, "waypoints", [])
    points = [(waypoint.x, waypoint.y) for waypoint in waypoints]
    return polyline_collides_with_obstacles(points, field, inflated=True)


def point_in_mission_bounds(point: Point, config: PlannerConfig, margin: float = 0.0, tol: float = 1e-9) -> bool:
    x, y = point
    return (
        margin - tol <= x <= config.mission.area_length_x - margin + tol
        and margin - tol <= y <= config.mission.area_length_y - margin + tol
    )


def polyline_out_of_mission_bounds(points: Sequence[Point], config: PlannerConfig, margin: float = 0.0) -> bool:
    return any(not point_in_mission_bounds(point, config, margin=margin) for point in points)


def path_segment_spec_out_of_bounds(segment, config: PlannerConfig, margin: float = 0.0) -> bool:
    waypoints = getattr(segment, "waypoints", [])
    points = [(waypoint.x, waypoint.y) for waypoint in waypoints]
    return polyline_out_of_mission_bounds(points, config, margin=margin)


def path_segment_invalid_reasons(
    segment,
    config: PlannerConfig,
    field: ObstacleField | None = None,
    margin: float = 0.0,
) -> List[str]:
    reasons: List[str] = []
    if path_segment_spec_out_of_bounds(segment, config, margin=margin):
        reasons.append("out_of_bounds")
    if field is not None and path_segment_spec_collides(segment, field):
        reasons.append("obstacle_collision")
    return reasons


def path_segment_invalid_length(
    segment,
    config: PlannerConfig,
    field: ObstacleField | None = None,
    margin: float = 0.0,
) -> float:
    waypoints = getattr(segment, "waypoints", [])
    points = [(waypoint.x, waypoint.y) for waypoint in waypoints]
    return polyline_invalid_length(points, config, field=field, margin=margin)


def polyline_invalid_length(
    points: Sequence[Point],
    config: PlannerConfig,
    field: ObstacleField | None = None,
    margin: float = 0.0,
) -> float:
    if len(points) < 2:
        return 0.0
    invalid = 0.0
    for start, end in zip(points[:-1], points[1:]):
        midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        out_of_bounds = (
            not point_in_mission_bounds(start, config, margin=margin)
            or not point_in_mission_bounds(end, config, margin=margin)
            or not point_in_mission_bounds(midpoint, config, margin=margin)
        )
        obstacle_collision = field is not None and segment_collides_with_obstacles(start, end, field, inflated=True)
        if out_of_bounds or obstacle_collision:
            invalid += math.hypot(end[0] - start[0], end[1] - start[1])
    return invalid


def polyline_collides_with_obstacles(points: Sequence[Point], field: ObstacleField, inflated: bool = True) -> bool:
    if any(point_in_any_obstacle(point, field, inflated=inflated) for point in points):
        return True
    return any(segment_collides_with_obstacles(points[idx], points[idx + 1], field, inflated=inflated) for idx in range(len(points) - 1))


def clearance_to_obstacles(point: Point, field: ObstacleField, inflated: bool = True) -> float:
    obstacles = field.inflated_obstacles if inflated else field.obstacles
    if not obstacles:
        return float("inf")
    return min(distance_point_to_polygon(point, obstacle.polygon) for obstacle in obstacles)


def danger_neighbor_count(point: Point, field: ObstacleField, radius: float) -> int:
    return sum(1 for obstacle in field.inflated_obstacles if distance_point_to_polygon(point, obstacle.polygon) <= radius)


def clipped_axis_aligned_segments(
    axis: str,
    fixed_coord: float,
    low: float,
    high: float,
    field: ObstacleField,
    footprint_width: float,
    min_length: float,
) -> List[Tuple[float, float]]:
    blocked: List[Tuple[float, float]] = []
    band_half_width = footprint_width / 2.0
    for obstacle in field.inflated_obstacles:
        x_min, y_min, x_max, y_max = obstacle_bounds(obstacle)
        if axis == "x":
            if y_min - band_half_width <= fixed_coord <= y_max + band_half_width:
                blocked.extend(
                    _polygon_band_intervals(
                        axis="x",
                        fixed_coord=fixed_coord,
                        polygon=obstacle.polygon,
                        band_half_width=band_half_width,
                        fallback=(x_min, x_max),
                        low=low,
                        high=high,
                    )
                )
        else:
            if x_min - band_half_width <= fixed_coord <= x_max + band_half_width:
                blocked.extend(
                    _polygon_band_intervals(
                        axis="y",
                        fixed_coord=fixed_coord,
                        polygon=obstacle.polygon,
                        band_half_width=band_half_width,
                        fallback=(y_min, y_max),
                        low=low,
                        high=high,
                    )
                )
    segments = subtract_intervals((low, high), blocked)
    return [(a, b) for a, b in segments if b - a >= min_length]


def subtract_intervals(base: Tuple[float, float], blocked: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    intervals = [(max(base[0], a), min(base[1], b)) for a, b in blocked if min(base[1], b) > max(base[0], a)]
    intervals.sort()
    result: List[Tuple[float, float]] = []
    cursor = base[0]
    for start, end in intervals:
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < base[1]:
        result.append((cursor, base[1]))
    return result


def rotated_rectangle_polygon(center_x: float, center_y: float, psi: float, length: float, width: float) -> List[Point]:
    c = math.cos(psi)
    s = math.sin(psi)
    local = [
        (-length / 2.0, -width / 2.0),
        (length / 2.0, -width / 2.0),
        (length / 2.0, width / 2.0),
        (-length / 2.0, width / 2.0),
    ]
    return [(center_x + c * x - s * y, center_y + s * x + c * y) for x, y in local]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if not polygon:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if _point_on_segment(point, polygon[j], polygon[i]):
            return True
        intersects = (yi > y) != (yj > y)
        if intersects:
            denom = yj - yi
            if abs(denom) <= 1e-12:
                j = i
                continue
            x_cross = (xj - xi) * (y - yi) / denom + xi
            if x <= x_cross:
                inside = not inside
        j = i
    return inside


def segment_intersects_polygon(start: Point, end: Point, polygon: Sequence[Point]) -> bool:
    if not polygon:
        return False
    if point_in_polygon(start, polygon) or point_in_polygon(end, polygon):
        return True
    return any(_segments_intersect(start, end, edge_start, edge_end) for edge_start, edge_end in polygon_edges(polygon))


def polygons_intersect(first: Sequence[Point], second: Sequence[Point]) -> bool:
    if not first or not second:
        return False
    if any(point_in_polygon(point, second) for point in first):
        return True
    if any(point_in_polygon(point, first) for point in second):
        return True
    return any(_segments_intersect(a0, a1, b0, b1) for a0, a1 in polygon_edges(first) for b0, b1 in polygon_edges(second))


def polygon_edges(polygon: Sequence[Point]) -> Iterable[Tuple[Point, Point]]:
    for idx, point in enumerate(polygon):
        yield point, polygon[(idx + 1) % len(polygon)]


def distance_point_to_polygon(point: Point, polygon: Sequence[Point]) -> float:
    if point_in_polygon(point, polygon):
        return 0.0
    return min(_distance_point_to_segment(point, start, end) for start, end in polygon_edges(polygon))


def _polygon_band_intervals(
    axis: str,
    fixed_coord: float,
    polygon: Sequence[Point],
    band_half_width: float,
    fallback: Tuple[float, float],
    low: float,
    high: float,
) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    offsets = (-band_half_width, 0.0, band_half_width)
    for offset in offsets:
        intervals.extend(_polygon_line_intervals(axis, fixed_coord + offset, polygon))
    if not intervals:
        intervals = [fallback]
    clipped = [(max(low, start), min(high, end)) for start, end in intervals if min(high, end) > max(low, start)]
    return _merge_intervals(clipped)


def _polygon_line_intervals(axis: str, fixed_coord: float, polygon: Sequence[Point]) -> List[Tuple[float, float]]:
    if not polygon:
        return []
    intersections: List[float] = []
    for start, end in polygon_edges(polygon):
        x0, y0 = start
        x1, y1 = end
        if axis == "x":
            if abs(y1 - y0) <= 1e-12:
                if abs(fixed_coord - y0) <= 1e-9:
                    intersections.extend([x0, x1])
                continue
            alpha = (fixed_coord - y0) / (y1 - y0)
            if -1e-9 <= alpha <= 1.0 + 1e-9:
                intersections.append(x0 + alpha * (x1 - x0))
        else:
            if abs(x1 - x0) <= 1e-12:
                if abs(fixed_coord - x0) <= 1e-9:
                    intersections.extend([y0, y1])
                continue
            alpha = (fixed_coord - x0) / (x1 - x0)
            if -1e-9 <= alpha <= 1.0 + 1e-9:
                intersections.append(y0 + alpha * (y1 - y0))
    if len(intersections) < 2:
        return []
    values = sorted(set(round(value, 9) for value in intersections))
    intervals: List[Tuple[float, float]] = []
    for idx in range(0, len(values) - 1, 2):
        start = float(values[idx])
        end = float(values[idx + 1])
        if end > start + 1e-9:
            intervals.append((start, end))
    return intervals


def _merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    sorted_items = sorted(intervals)
    merged = [sorted_items[0]]
    for start, end in sorted_items[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1e-9:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _normalized_obstacle(obstacle: StaticObstacle, inflation: float, path_config: PathPlanningConfig) -> StaticObstacle:
    kind = obstacle.kind.lower()
    segments = max(path_config.obstacle_circle_segments, 8)
    if kind == "rectangle":
        center = obstacle.center or _polygon_centroid(obstacle.polygon)
        width = max(float(obstacle.width or _bounds_width(obstacle.polygon)), 0.0) + 2.0 * inflation
        height = max(float(obstacle.height or _bounds_height(obstacle.polygon)), 0.0) + 2.0 * inflation
        polygon = rotated_rectangle_polygon(center[0], center[1], obstacle.psi, width, height)
    elif kind == "circle":
        if obstacle.center is None or obstacle.radius is None:
            raise ValueError(f"circle obstacle {obstacle.obstacle_id} requires center and radius")
        polygon = _ellipse_polygon(obstacle.center, (obstacle.radius + inflation, obstacle.radius + inflation), 0.0, segments)
    elif kind == "ellipse":
        if obstacle.center is None or obstacle.radii is None:
            raise ValueError(f"ellipse obstacle {obstacle.obstacle_id} requires center and radii")
        polygon = _ellipse_polygon(
            obstacle.center,
            (obstacle.radii[0] + inflation, obstacle.radii[1] + inflation),
            obstacle.psi,
            segments,
        )
    elif kind == "polygon":
        if not obstacle.polygon:
            raise ValueError(f"polygon obstacle {obstacle.obstacle_id} requires polygon points")
        polygon = _inflate_polygon_radially(obstacle.polygon, inflation)
    else:
        raise ValueError(f"unsupported static obstacle kind: {obstacle.kind}")
    return StaticObstacle(
        obstacle_id=obstacle.obstacle_id,
        kind=kind,
        polygon=_ensure_ccw(polygon),
        center=obstacle.center,
        radius=obstacle.radius,
        radii=obstacle.radii,
        width=obstacle.width,
        height=obstacle.height,
        psi=obstacle.psi,
        inflation_radius=inflation,
        metadata=dict(obstacle.metadata),
    )


def _ellipse_polygon(center: Point, radii: Point, psi: float, segments: int) -> List[Point]:
    c = math.cos(psi)
    s = math.sin(psi)
    points: List[Point] = []
    for idx in range(segments):
        theta = 2.0 * math.pi * idx / segments
        x = radii[0] * math.cos(theta)
        y = radii[1] * math.sin(theta)
        points.append((center[0] + c * x - s * y, center[1] + s * x + c * y))
    return points


def _inflate_polygon_radially(polygon: Sequence[Point], margin: float) -> List[Point]:
    if margin <= 1e-12:
        return list(polygon)
    centroid = _polygon_centroid(polygon)
    inflated: List[Point] = []
    for x, y in polygon:
        dx = x - centroid[0]
        dy = y - centroid[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-9:
            inflated.append((x, y))
        else:
            inflated.append((x + margin * dx / norm, y + margin * dy / norm))
    return inflated


def _polygon_centroid(polygon: Sequence[Point]) -> Point:
    if not polygon:
        return (0.0, 0.0)
    return (sum(point[0] for point in polygon) / len(polygon), sum(point[1] for point in polygon) / len(polygon))


def _ensure_ccw(polygon: Sequence[Point]) -> List[Point]:
    points = list(polygon)
    area = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    if area < 0.0:
        points.reverse()
    return points


def _bounds_width(polygon: Sequence[Point]) -> float:
    if not polygon:
        return 0.0
    xs = [point[0] for point in polygon]
    return max(xs) - min(xs)


def _bounds_height(polygon: Sequence[Point]) -> float:
    if not polygon:
        return 0.0
    ys = [point[1] for point in polygon]
    return max(ys) - min(ys)


def _segments_intersect(a0: Point, a1: Point, b0: Point, b1: Point) -> bool:
    o1 = _orientation(a0, a1, b0)
    o2 = _orientation(a0, a1, b1)
    o3 = _orientation(b0, b1, a0)
    o4 = _orientation(b0, b1, a1)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    return (
        abs(o1) <= 1e-9 and _point_on_segment(b0, a0, a1)
        or abs(o2) <= 1e-9 and _point_on_segment(b1, a0, a1)
        or abs(o3) <= 1e-9 and _point_on_segment(a0, b0, b1)
        or abs(o4) <= 1e-9 and _point_on_segment(a1, b0, b1)
    )


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    if abs(_orientation(start, end, point)) > 1e-9:
        return False
    return (
        min(start[0], end[0]) - 1e-9 <= point[0] <= max(start[0], end[0]) + 1e-9
        and min(start[1], end[1]) - 1e-9 <= point[1] <= max(start[1], end[1]) + 1e-9
    )


def _distance_point_to_segment(point: Point, start: Point, end: Point) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
    proj_x = sx + t * dx
    proj_y = sy + t * dy
    return math.hypot(px - proj_x, py - proj_y)
