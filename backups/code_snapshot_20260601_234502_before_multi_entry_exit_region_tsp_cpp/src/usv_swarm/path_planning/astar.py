from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from ..geometry import wrap_angle
from .obstacles import point_in_any_obstacle, segment_collides_with_obstacles
from .graph import edge_key
from .types import ObstacleField, PathPlanningConfig, RegionGraph


@dataclass(order=True)
class _OpenNode:
    priority: float
    serial: int
    node_id: str = field(compare=False)


@dataclass(frozen=True)
class AStarResult:
    path: List[str]
    cost: float
    expanded: int
    found: bool


@dataclass(frozen=True)
class GridAStarResult:
    points: List[Tuple[float, float]]
    cost: float
    expanded: int
    found: bool


def sailing_safety_weight(danger_neighbor_count: int) -> float:
    if danger_neighbor_count <= 0:
        return 1.0
    return 1.0 + 0.5 * (2.0 ** max(danger_neighbor_count - 1, 0))


def goal_pilot_factor(theta: float) -> float:
    return 3.0 / max(4.0 - math.sin(theta), 1e-6)


def turn_aware_astar(
    graph: RegionGraph,
    start: str,
    goal: str,
    path_config: PathPlanningConfig | None = None,
    danger_counter: Optional[Callable[[str], int]] = None,
    allowed_nodes: Optional[Iterable[str]] = None,
) -> AStarResult:
    path_config = path_config or PathPlanningConfig()
    allowed: Optional[Set[str]] = None if allowed_nodes is None else set(allowed_nodes)
    if allowed is not None and (start not in allowed or goal not in allowed):
        return AStarResult(path=[], cost=float("inf"), expanded=0, found=False)
    if start not in graph.regions or goal not in graph.regions:
        return AStarResult(path=[], cost=float("inf"), expanded=0, found=False)
    if start == goal:
        return AStarResult(path=[start], cost=0.0, expanded=0, found=True)

    open_set: List[_OpenNode] = []
    serial = 0
    heapq.heappush(open_set, _OpenNode(priority=0.0, serial=serial, node_id=start))
    came_from: Dict[str, str] = {}
    g_score: Dict[str, float] = {start: 0.0}
    heading_to_node: Dict[str, float] = {start: 0.0}
    expanded = 0

    while open_set:
        current = heapq.heappop(open_set).node_id
        expanded += 1
        if current == goal:
            return AStarResult(
                path=_reconstruct_path(came_from, current),
                cost=g_score[current],
                expanded=expanded,
                found=True,
            )

        current_heading = heading_to_node.get(current, 0.0)
        for neighbor in graph.adjacency.get(current, []):
            if allowed is not None and neighbor not in allowed:
                continue
            step_cost, new_heading = _edge_cost(graph, current, neighbor, current_heading, path_config, danger_counter)
            candidate_g = g_score[current] + step_cost
            if candidate_g + 1e-9 >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = candidate_g
            heading_to_node[neighbor] = new_heading
            serial += 1
            priority = candidate_g + _heuristic(graph, neighbor, goal)
            heapq.heappush(open_set, _OpenNode(priority=priority, serial=serial, node_id=neighbor))

    return AStarResult(path=[], cost=float("inf"), expanded=expanded, found=False)


def obstacle_aware_grid_astar(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    bounds: Tuple[float, float, float, float],
    obstacle_field: ObstacleField,
    resolution: float,
    path_config: PathPlanningConfig | None = None,
) -> GridAStarResult:
    path_config = path_config or PathPlanningConfig()
    resolution = max(resolution, 1e-6)
    x_min, y_min, x_max, y_max = bounds

    def to_cell(point: Tuple[float, float]) -> Tuple[int, int]:
        return (
            int(round((point[0] - x_min) / resolution)),
            int(round((point[1] - y_min) / resolution)),
        )

    def to_point(cell: Tuple[int, int]) -> Tuple[float, float]:
        return (x_min + cell[0] * resolution, y_min + cell[1] * resolution)

    start_cell = to_cell(start)
    goal_cell = to_cell(goal)
    max_i = int(round((x_max - x_min) / resolution))
    max_j = int(round((y_max - y_min) / resolution))
    if point_in_any_obstacle(start, obstacle_field, inflated=True) or point_in_any_obstacle(goal, obstacle_field, inflated=True):
        return GridAStarResult(points=[], cost=float("inf"), expanded=0, found=False)

    open_set: List[Tuple[float, int, Tuple[int, int]]] = []
    serial = 0
    heapq.heappush(open_set, (0.0, serial, start_cell))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start_cell: 0.0}
    expanded = 0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        _, _, current = heapq.heappop(open_set)
        expanded += 1
        if current == goal_cell:
            cells = _reconstruct_grid_path(came_from, current)
            return GridAStarResult(points=[to_point(cell) for cell in cells], cost=g_score[current], expanded=expanded, found=True)
        current_point = to_point(current)
        for di, dj in neighbors:
            neighbor = (current[0] + di, current[1] + dj)
            if neighbor[0] < 0 or neighbor[0] > max_i or neighbor[1] < 0 or neighbor[1] > max_j:
                continue
            neighbor_point = to_point(neighbor)
            if point_in_any_obstacle(neighbor_point, obstacle_field, inflated=True):
                continue
            if segment_collides_with_obstacles(current_point, neighbor_point, obstacle_field, inflated=True):
                continue
            step = resolution * (math.sqrt(2.0) if di != 0 and dj != 0 else 1.0)
            danger = path_config.astar_safety_weight * sailing_safety_weight(
                int(segment_collides_with_obstacles(current_point, neighbor_point, obstacle_field, inflated=False))
            )
            tentative = g_score[current] + step + danger
            if tentative + 1e-9 >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            serial += 1
            priority = tentative + math.hypot(goal_cell[0] - neighbor[0], goal_cell[1] - neighbor[1]) * resolution
            heapq.heappush(open_set, (priority, serial, neighbor))
    return GridAStarResult(points=[], cost=float("inf"), expanded=expanded, found=False)


def _edge_cost(
    graph: RegionGraph,
    current: str,
    neighbor: str,
    current_heading: float,
    path_config: PathPlanningConfig,
    danger_counter: Optional[Callable[[str], int]],
) -> Tuple[float, float]:
    key = edge_key(current, neighbor)
    base = graph.edge_weights.get(key, 1.0)
    meta = graph.edge_metadata.get(key, {})
    current_region = graph.regions[current]
    neighbor_region = graph.regions[neighbor]
    dx = neighbor_region.center[0] - current_region.center[0]
    dy = neighbor_region.center[1] - current_region.center[1]
    edge_heading = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-9 else current_heading
    heading_change = abs(wrap_angle(edge_heading - current_heading))
    danger = danger_counter(neighbor) if danger_counter is not None else int(float(neighbor_region.metadata.get("danger_neighbors", "0")))
    safety = sailing_safety_weight(danger)
    boundary_risk = float(neighbor_region.metadata.get("boundary_risk", "0.0"))
    cost = (
        base * safety
        + path_config.astar_heading_weight * heading_change
        + path_config.astar_safety_weight * (safety - 1.0)
        + path_config.astar_boundary_weight * boundary_risk
        + path_config.curvature_weight * max(float(meta.get("curvature_violation", 0.0)), 0.0)
    )
    return cost, edge_heading


def _heuristic(graph: RegionGraph, node: str, goal: str) -> float:
    node_region = graph.regions[node]
    goal_region = graph.regions[goal]
    dx = goal_region.center[0] - node_region.center[0]
    dy = goal_region.center[1] - node_region.center[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        return 0.0
    theta = math.atan2(dy, dx)
    return distance * goal_pilot_factor(theta)


def _reconstruct_path(came_from: Dict[str, str], current: str) -> List[str]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _reconstruct_grid_path(
    came_from: Dict[Tuple[int, int], Tuple[int, int]],
    current: Tuple[int, int],
) -> List[Tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
