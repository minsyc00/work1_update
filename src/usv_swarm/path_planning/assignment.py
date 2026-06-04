from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from ..schema import PlannerConfig
from .graph import graph_is_connected
from .types import BalancedAssignment, RegionGraph


def balance_region_workload(graph: RegionGraph, config: PlannerConfig) -> BalancedAssignment:
    agent_count = config.fleet.num_agents or len(config.fleet.initial_states_3dof)
    agent_regions: Dict[int, List[str]] = {agent_id: [] for agent_id in range(agent_count)}
    loads: Dict[int, float] = {agent_id: 0.0 for agent_id in range(agent_count)}
    connected: Dict[int, bool] = {agent_id: True for agent_id in range(agent_count)}

    if not graph.regions:
        return BalancedAssignment(
            agent_regions=agent_regions,
            loads=loads,
            connected=connected,
            imbalance_ratio=0.0,
            objective=0.0,
            diagnostics={"status": "empty_graph"},
        )

    ordered_regions = _ordered_regions_for_assignment(graph)
    active_count = min(agent_count, len(ordered_regions))
    active_agents = _ordered_agent_ids(config, graph)[:active_count]
    chunks = _minimax_contiguous_partition(ordered_regions, active_agents, graph)
    for agent_id, region_ids in chunks.items():
        agent_regions[agent_id] = region_ids
        loads[agent_id] = _load(region_ids, graph)

    _improve_by_boundary_migration(agent_regions, loads, graph, max_iterations=20)
    for agent_id, region_ids in agent_regions.items():
        connected[agent_id] = graph_is_connected(graph, region_ids)

    active_loads = [loads[agent_id] for agent_id in active_agents if agent_regions[agent_id]]
    avg_load = sum(active_loads) / max(len(active_loads), 1)
    imbalance_ratio = 0.0 if avg_load <= 1e-9 else (max(active_loads) - min(active_loads)) / avg_load
    objective = max(active_loads, default=0.0) + imbalance_ratio
    diagnostics = {"active_agents": str(len(active_loads)), "region_count": str(len(ordered_regions))}
    if len(ordered_regions) < agent_count:
        diagnostics["warning"] = "fewer_regions_than_agents"
    if imbalance_ratio > 0.10:
        diagnostics["imbalance_warning"] = "load_difference_exceeds_default_tolerance"

    return BalancedAssignment(
        agent_regions=agent_regions,
        loads=loads,
        connected=connected,
        imbalance_ratio=imbalance_ratio,
        objective=objective,
        diagnostics=diagnostics,
    )


def _ordered_regions_for_assignment(graph: RegionGraph) -> List[str]:
    if not graph.regions:
        return []
    first_region = next(iter(graph.regions.values()))
    if first_region.preferred_axis == "x":
        return sorted(graph.regions, key=lambda region_id: (graph.regions[region_id].center[1], graph.regions[region_id].center[0]))
    return sorted(graph.regions, key=lambda region_id: (graph.regions[region_id].center[0], graph.regions[region_id].center[1]))


def _ordered_agent_ids(config: PlannerConfig, graph: RegionGraph) -> List[int]:
    if not graph.regions:
        return list(range(config.fleet.num_agents or 0))
    first_region = next(iter(graph.regions.values()))
    if first_region.preferred_axis == "x":
        return [
            agent_id
            for agent_id, _ in sorted(
                enumerate(config.fleet.initial_states_3dof),
                key=lambda pair: (pair[1].y, pair[1].x),
            )
        ]
    return [
        agent_id
        for agent_id, _ in sorted(
            enumerate(config.fleet.initial_states_3dof),
            key=lambda pair: (pair[1].x, pair[1].y),
        )
    ]


def _minimax_contiguous_partition(
    ordered_regions: Sequence[str],
    active_agents: Sequence[int],
    graph: RegionGraph,
) -> Dict[int, List[str]]:
    n_regions = len(ordered_regions)
    n_agents = len(active_agents)
    result: Dict[int, List[str]] = {agent_id: [] for agent_id in active_agents}
    if n_regions == 0 or n_agents == 0:
        return result

    prefix = [0.0]
    for region_id in ordered_regions:
        prefix.append(prefix[-1] + graph.node_weights.get(region_id, 0.0))

    def block_cost(start: int, end: int) -> float:
        return prefix[end] - prefix[start]

    inf = float("inf")
    dp = [[inf] * (n_regions + 1) for _ in range(n_agents + 1)]
    choice = [[-1] * (n_regions + 1) for _ in range(n_agents + 1)]
    dp[0][0] = 0.0
    for agent_idx in range(1, n_agents + 1):
        for end in range(agent_idx, n_regions + 1):
            for start in range(agent_idx - 1, end):
                candidate = max(dp[agent_idx - 1][start], block_cost(start, end))
                if candidate < dp[agent_idx][end]:
                    dp[agent_idx][end] = candidate
                    choice[agent_idx][end] = start

    end = n_regions
    for agent_idx in range(n_agents, 0, -1):
        start = choice[agent_idx][end]
        agent_id = active_agents[agent_idx - 1]
        result[agent_id] = list(ordered_regions[start:end])
        end = start
    return result


def _improve_by_boundary_migration(
    agent_regions: Dict[int, List[str]],
    loads: Dict[int, float],
    graph: RegionGraph,
    max_iterations: int,
) -> None:
    for _ in range(max_iterations):
        active_agents = [agent_id for agent_id, regions in agent_regions.items() if regions]
        if len(active_agents) < 2:
            return
        heavy = max(active_agents, key=lambda agent_id: loads[agent_id])
        light = min(active_agents, key=lambda agent_id: loads[agent_id])
        current_gap = loads[heavy] - loads[light]
        if current_gap <= 1e-9:
            return
        candidate = _best_boundary_region_to_move(agent_regions[heavy], agent_regions[light], graph, current_gap)
        if candidate is None:
            return
        region_weight = graph.node_weights.get(candidate, 0.0)
        agent_regions[heavy].remove(candidate)
        agent_regions[light].append(candidate)
        agent_regions[light].sort(key=lambda region_id: graph.regions[region_id].center)
        loads[heavy] -= region_weight
        loads[light] += region_weight


def _best_boundary_region_to_move(
    source: List[str],
    target: List[str],
    graph: RegionGraph,
    current_gap: float,
) -> str | None:
    if len(source) <= 1:
        return None
    target_set = set(target)
    best_region: str | None = None
    best_gap = current_gap
    for region_id in list(source):
        if target_set and not any(neighbor in target_set for neighbor in graph.adjacency.get(region_id, [])):
            continue
        new_source = [item for item in source if item != region_id]
        new_target = list(target) + [region_id]
        if not graph_is_connected(graph, new_source) or not graph_is_connected(graph, new_target):
            continue
        weight = graph.node_weights.get(region_id, 0.0)
        new_gap = abs(current_gap - 2.0 * weight)
        if new_gap + 1e-9 < best_gap:
            best_gap = new_gap
            best_region = region_id
    return best_region


def _load(region_ids: Sequence[str], graph: RegionGraph) -> float:
    return float(sum(graph.node_weights.get(region_id, 0.0) for region_id in region_ids))
