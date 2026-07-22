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


def apply_lightweight_load_swap(
    assignment: BalancedAssignment,
    graph: RegionGraph,
    max_iterations: int = 4,
    workload_weights: Dict[str, float] | None = None,
) -> BalancedAssignment:
    """Move boundary regions from heavy to light agents when it safely improves load balance."""

    agent_regions = {agent_id: list(region_ids) for agent_id, region_ids in assignment.agent_regions.items()}
    weights = workload_weights or graph.node_weights
    loads = {agent_id: _weighted_load(region_ids, weights) for agent_id, region_ids in agent_regions.items()}
    before = _imbalance_ratio(agent_regions, loads, include_empty=True)
    swap_count = 0
    candidate_count = 0
    reject_reasons: Dict[str, int] = {}
    for _ in range(max(max_iterations, 0)):
        source_agents = [agent_id for agent_id, regions in agent_regions.items() if regions]
        if not source_agents or len(agent_regions) < 2:
            break
        heavy = max(source_agents, key=lambda agent_id: loads.get(agent_id, 0.0))
        light = min(agent_regions, key=lambda agent_id: loads.get(agent_id, 0.0))
        if heavy == light:
            _increment_reason(reject_reasons, "already_balanced")
            break
        current = _imbalance_ratio(agent_regions, loads, include_empty=True)
        candidate, tried, reasons = _best_swap_region(agent_regions, loads, graph, weights, heavy, light, current)
        candidate_count += tried
        for reason, count in reasons.items():
            reject_reasons[reason] = reject_reasons.get(reason, 0) + count
        if candidate is None:
            break
        region_id = candidate
        weight = weights.get(region_id, graph.node_weights.get(region_id, 0.0))
        agent_regions[heavy].remove(region_id)
        agent_regions[light].append(region_id)
        agent_regions[light].sort(key=lambda item: graph.regions[item].center)
        loads[heavy] -= weight
        loads[light] += weight
        swap_count += 1

    connected = {agent_id: graph_is_connected(graph, region_ids) for agent_id, region_ids in agent_regions.items()}
    after = _imbalance_ratio(agent_regions, loads, include_empty=True)
    diagnostics = dict(assignment.diagnostics)
    diagnostics.update(
        {
            "load_swap_count": str(swap_count),
            "load_swap_candidate_count": str(candidate_count),
            "load_swap_before_imbalance": f"{before:.6f}",
            "load_swap_after_imbalance": f"{after:.6f}",
            "load_swap_reject_reasons": ",".join(
                f"{reason}:{count}" for reason, count in sorted(reject_reasons.items())
            ),
        }
    )
    active_loads = [loads[agent_id] for agent_id in agent_regions]
    return BalancedAssignment(
        agent_regions=agent_regions,
        loads=loads,
        connected=connected,
        imbalance_ratio=after,
        objective=max(active_loads, default=0.0) + after,
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


def _best_swap_region(
    agent_regions: Dict[int, List[str]],
    loads: Dict[int, float],
    graph: RegionGraph,
    weights: Dict[str, float],
    heavy: int,
    light: int,
    current_imbalance: float,
) -> Tuple[str | None, int, Dict[str, int]]:
    source = agent_regions.get(heavy, [])
    target = agent_regions.get(light, [])
    reasons: Dict[str, int] = {}
    if len(source) <= 1:
        _increment_reason(reasons, "source_single_region")
        return None, 0, reasons
    target_set = set(target)
    best_region: str | None = None
    best_imbalance = current_imbalance
    candidate_count = 0
    for region_id in list(source):
        candidate_count += 1
        if target_set and not any(neighbor in target_set for neighbor in graph.adjacency.get(region_id, [])):
            _increment_reason(reasons, "not_boundary_to_light_agent")
            continue
        new_source = [item for item in source if item != region_id]
        new_target = list(target) + [region_id]
        if not graph_is_connected(graph, new_source):
            _increment_reason(reasons, "source_disconnect")
            continue
        if not graph_is_connected(graph, new_target):
            _increment_reason(reasons, "target_disconnect")
            continue
        trial_regions = {agent_id: list(regions) for agent_id, regions in agent_regions.items()}
        trial_loads = dict(loads)
        weight = weights.get(region_id, graph.node_weights.get(region_id, 0.0))
        trial_regions[heavy] = new_source
        trial_regions[light] = new_target
        trial_loads[heavy] = trial_loads.get(heavy, 0.0) - weight
        trial_loads[light] = trial_loads.get(light, 0.0) + weight
        trial_imbalance = _imbalance_ratio(trial_regions, trial_loads, include_empty=True)
        if trial_imbalance + 1e-9 < best_imbalance:
            best_imbalance = trial_imbalance
            best_region = region_id
        else:
            _increment_reason(reasons, "no_imbalance_improvement")
    return best_region, candidate_count, reasons


def _increment_reason(reasons: Dict[str, int], reason: str) -> None:
    reasons[reason] = reasons.get(reason, 0) + 1


def _imbalance_ratio(agent_regions: Dict[int, List[str]], loads: Dict[int, float], *, include_empty: bool = False) -> float:
    active_loads = [
        loads.get(agent_id, 0.0)
        for agent_id, regions in agent_regions.items()
        if include_empty or regions
    ]
    if not active_loads:
        return 0.0
    avg_load = sum(active_loads) / len(active_loads)
    if avg_load <= 1e-9:
        return 0.0
    return (max(active_loads) - min(active_loads)) / avg_load


def _load(region_ids: Sequence[str], graph: RegionGraph) -> float:
    return float(sum(graph.node_weights.get(region_id, 0.0) for region_id in region_ids))


def _weighted_load(region_ids: Sequence[str], weights: Dict[str, float]) -> float:
    return float(sum(weights.get(region_id, 0.0) for region_id in region_ids))
