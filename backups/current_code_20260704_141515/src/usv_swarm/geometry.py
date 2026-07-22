from __future__ import annotations

import math
from collections import deque
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .schema import Pose2D


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def mean_heading(headings: Iterable[float]) -> float:
    headings = list(headings)
    if not headings:
        return 0.0
    return math.atan2(sum(math.sin(h) for h in headings), sum(math.cos(h) for h in headings))


def distance_xy(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def polyline_length(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(
        sum(
            math.hypot(points[idx + 1][0] - points[idx][0], points[idx + 1][1] - points[idx][1])
            for idx in range(len(points) - 1)
        )
    )


def approximate_dubins_length(start: Pose2D, end: Pose2D, turn_radius: float) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    base = math.hypot(dx, dy)
    if base < 1e-9:
        return turn_radius * abs(wrap_angle(end.psi - start.psi))
    course = math.atan2(dy, dx)
    turn_1 = abs(wrap_angle(course - start.psi))
    turn_2 = abs(wrap_angle(end.psi - course))
    return base + turn_radius * (turn_1 + turn_2)


def unit_heading(psi: float) -> np.ndarray:
    return np.array([math.cos(psi), math.sin(psi)], dtype=float)


def bezier_point(control_points: Sequence[Tuple[float, float]], t: float) -> np.ndarray:
    cps = np.asarray(control_points, dtype=float)
    omt = 1.0 - t
    coeffs = np.array(
        [
            omt**5,
            5 * omt**4 * t,
            10 * omt**3 * t**2,
            10 * omt**2 * t**3,
            5 * omt * t**4,
            t**5,
        ],
        dtype=float,
    )
    return coeffs @ cps


def bezier_first_derivative(control_points: Sequence[Tuple[float, float]], t: float) -> np.ndarray:
    cps = np.asarray(control_points, dtype=float)
    diffs = 5.0 * (cps[1:] - cps[:-1])
    omt = 1.0 - t
    coeffs = np.array(
        [
            omt**4,
            4 * omt**3 * t,
            6 * omt**2 * t**2,
            4 * omt * t**3,
            t**4,
        ],
        dtype=float,
    )
    return coeffs @ diffs


def bezier_second_derivative(control_points: Sequence[Tuple[float, float]], t: float) -> np.ndarray:
    cps = np.asarray(control_points, dtype=float)
    diffs = 20.0 * (cps[2:] - 2 * cps[1:-1] + cps[:-2])
    omt = 1.0 - t
    coeffs = np.array(
        [
            omt**3,
            3 * omt**2 * t,
            3 * omt * t**2,
            t**3,
        ],
        dtype=float,
    )
    return coeffs @ diffs


def bezier_curvature(control_points: Sequence[Tuple[float, float]], t: float) -> float:
    d1 = bezier_first_derivative(control_points, t)
    d2 = bezier_second_derivative(control_points, t)
    denom = float(np.linalg.norm(d1) ** 3)
    if denom < 1e-9:
        return 0.0
    numer = d1[0] * d2[1] - d1[1] * d2[0]
    return numer / denom


def sample_quintic_bezier(control_points: Sequence[Tuple[float, float]], sample_count: int) -> Tuple[List[Tuple[float, float]], List[float], float]:
    points: List[Tuple[float, float]] = []
    headings: List[float] = []
    max_curvature = 0.0
    for idx in range(sample_count):
        t = idx / max(sample_count - 1, 1)
        pt = bezier_point(control_points, t)
        tangent = bezier_first_derivative(control_points, t)
        heading = math.atan2(tangent[1], tangent[0]) if np.linalg.norm(tangent) > 1e-9 else 0.0
        curvature = abs(bezier_curvature(control_points, t))
        points.append((float(pt[0]), float(pt[1])))
        headings.append(heading)
        max_curvature = max(max_curvature, curvature)
    return points, headings, max_curvature


def straight_segment_points(start: Pose2D, end: Pose2D, sample_count: int) -> Tuple[List[Tuple[float, float]], List[float]]:
    points: List[Tuple[float, float]] = []
    headings: List[float] = []
    heading = math.atan2(end.y - start.y, end.x - start.x) if abs(end.x - start.x) + abs(end.y - start.y) > 1e-9 else start.psi
    for idx in range(sample_count):
        alpha = idx / max(sample_count - 1, 1)
        x = start.x + alpha * (end.x - start.x)
        y = start.y + alpha * (end.y - start.y)
        points.append((x, y))
        headings.append(heading)
    return points, headings


def rotated_rectangle_mask(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    center_x: float,
    center_y: float,
    psi: float,
    length: float,
    width: float,
) -> np.ndarray:
    xx, yy = np.meshgrid(x_coords, y_coords)
    dx = xx - center_x
    dy = yy - center_y
    c = math.cos(psi)
    s = math.sin(psi)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return (np.abs(local_x) <= length / 2.0) & (np.abs(local_y) <= width / 2.0)


def rotated_rectangle_local_mask(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    center_x: float,
    center_y: float,
    psi: float,
    length: float,
    width: float,
) -> Tuple[np.ndarray, slice, slice]:
    """Return a footprint mask over only the rotated rectangle's AABB window."""

    if x_coords.size == 0 or y_coords.size == 0:
        return np.zeros((0, 0), dtype=bool), slice(0, 0), slice(0, 0)

    half_l = length / 2.0
    half_w = width / 2.0
    c = math.cos(psi)
    s = math.sin(psi)
    corners = (
        (half_l, half_w),
        (half_l, -half_w),
        (-half_l, half_w),
        (-half_l, -half_w),
    )
    world_x = [center_x + c * lx - s * ly for lx, ly in corners]
    world_y = [center_y + s * lx + c * ly for lx, ly in corners]
    min_x = min(world_x)
    max_x = max(world_x)
    min_y = min(world_y)
    max_y = max(world_y)

    col_start = max(int(np.searchsorted(x_coords, min_x, side="left")), 0)
    col_stop = min(int(np.searchsorted(x_coords, max_x, side="right")), x_coords.size)
    row_start = max(int(np.searchsorted(y_coords, min_y, side="left")), 0)
    row_stop = min(int(np.searchsorted(y_coords, max_y, side="right")), y_coords.size)
    row_slice = slice(row_start, row_stop)
    col_slice = slice(col_start, col_stop)
    if row_start >= row_stop or col_start >= col_stop:
        return np.zeros((0, 0), dtype=bool), row_slice, col_slice

    xx, yy = np.meshgrid(x_coords[col_slice], y_coords[row_slice])
    dx = xx - center_x
    dy = yy - center_y
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    mask = (np.abs(local_x) <= half_l) & (np.abs(local_y) <= half_w)
    return mask, row_slice, col_slice


def connected_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[List[Tuple[int, int]]] = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            component: List[Tuple[int, int]] = []
            queue: deque[Tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            while queue:
                cr, cc = queue.popleft()
                component.append((cr, cc))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if nr < 0 or nr >= height or nc < 0 or nc >= width:
                        continue
                    if visited[nr, nc] or not mask[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))
            components.append(component)
    return components
