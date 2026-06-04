from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, List, Set, Tuple

from ..dubins import dubins_shortest_path
from ..geometry import wrap_angle
from ..schema import PlannerConfig, Pose2D
from .obstacles import clearance_to_obstacles, polyline_collides_with_obstacles
from .types import DecomposedRegion, ObstacleField, RegionCoveragePattern, RegionGraph


def edge_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def build_region_graph(
    regions: Iterable[DecomposedRegion],
    patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
    obstacle_field: ObstacleField | None = None,
) -> RegionGraph:
    region_map = {region.region_id: region for region in regions}
    adjacency: Dict[str, List[str]] = {region_id: [] for region_id in region_map}
    node_weights: Dict[str, float] = {}
    edge_weights: Dict[Tuple[str, str], float] = {}
    edge_metadata: Dict[Tuple[str, str], Dict[str, float]] = {}

    for region_id, candidates in patterns.items():
        if not candidates:
            continue
        feasible = [candidate for candidate in candidates if candidate.feasible]
        best = min(feasible or candidates, key=lambda item: item.estimated_time)
        node_weights[region_id] = best.estimated_time

    region_items = list(region_map.values())
    for idx, region_a in enumerate(region_items):
        for region_b in region_items[idx + 1 :]:
            if region_b.region_id in region_a.neighbors or _bounds_touch_or_near(region_a.bounds, region_b.bounds):
                adjacency[region_a.region_id].append(region_b.region_id)
                adjacency[region_b.region_id].append(region_a.region_id)
                weight, meta = _region_transition_weight(region_a, region_b, patterns, config, obstacle_field)
                key = edge_key(region_a.region_id, region_b.region_id)
                edge_weights[key] = weight
                edge_metadata[key] = meta

    return RegionGraph(
        regions=region_map,
        adjacency={key: sorted(value) for key, value in adjacency.items()},
        node_weights=node_weights,
        edge_weights=edge_weights,
        edge_metadata=edge_metadata,
        patterns=patterns,
        obstacle_field=obstacle_field,
        metadata={"static_obstacle_aware": str(obstacle_field is not None).lower()},
    )


def graph_is_connected(graph: RegionGraph, nodes: Iterable[str] | None = None) -> bool:
    node_set: Set[str] = set(nodes or graph.regions.keys())
    if not node_set:
        return True
    start = next(iter(node_set))
    visited = {start}
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph.adjacency.get(node, []):
            if neighbor in node_set and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited == node_set


def connected_components(graph: RegionGraph, nodes: Iterable[str] | None = None) -> List[List[str]]:
    remaining: Set[str] = set(nodes or graph.regions.keys())
    components: List[List[str]] = []
    while remaining:
        start = next(iter(remaining))
        component = []
        queue: deque[str] = deque([start])
        remaining.remove(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in graph.adjacency.get(node, []):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def region_center_pose(region: DecomposedRegion, psi: float = 0.0) -> Pose2D:
    return Pose2D(region.center[0], region.center[1], psi)


def _bounds_touch_or_near(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    tolerance: float = 1e-6,
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x_gap = max(bx0 - ax1, ax0 - bx1, 0.0)
    y_gap = max(by0 - ay1, ay0 - by1, 0.0)
    x_overlap = min(ax1, bx1) - max(ax0, bx0)
    y_overlap = min(ay1, by1) - max(ay0, by0)
    return (x_gap <= tolerance and y_overlap >= -tolerance) or (y_gap <= tolerance and x_overlap >= -tolerance)


def _region_transition_weight(
    region_a: DecomposedRegion,
    region_b: DecomposedRegion,
    patterns: Dict[str, List[RegionCoveragePattern]],
    config: PlannerConfig,
    obstacle_field: ObstacleField | None = None,
) -> Tuple[float, Dict[str, float]]:
    candidates_a = patterns.get(region_a.region_id, [])
    candidates_b = patterns.get(region_b.region_id, [])
    if candidates_a and candidates_b:
        best_a = min(candidates_a, key=lambda item: item.estimated_time)
        best_b = min(candidates_b, key=lambda item: item.estimated_time)
        start = best_a.exit_pose
        end = best_b.entry_pose
    else:
        start = region_center_pose(region_a, 0.0)
        end = region_center_pose(region_b, 0.0)
    dubins = dubins_shortest_path(start, end, config.fleet.min_turn_radius)
    euclidean = math.hypot(region_a.center[0] - region_b.center[0], region_a.center[1] - region_b.center[1])
    heading_change = abs(wrap_angle(end.psi - start.psi))
    collision_penalty = 0.0
    clearance = float("inf")
    if obstacle_field is not None:
        clearance = min(
            clearance_to_obstacles(region_a.center, obstacle_field, inflated=True),
            clearance_to_obstacles(region_b.center, obstacle_field, inflated=True),
        )
        if polyline_collides_with_obstacles([(start.x, start.y), (end.x, end.y)], obstacle_field, inflated=True):
            collision_penalty = 1e6
    weight = dubins.total_length / max(config.fleet.cruise_speed, 1e-6) + heading_change + collision_penalty
    return weight, {
        "euclidean": euclidean,
        "dubins_length": dubins.total_length,
        "heading_change": heading_change,
        "collision_penalty": collision_penalty,
        "obstacle_clearance": clearance,
    }
