from __future__ import annotations

import json
import math
from typing import Dict, List, Sequence, Tuple

from ..schema import PlannerConfig
from .graph import graph_is_connected
from .types import BalancedAssignment, PathPlanningConfig, RegionCoveragePattern, RegionGraph


def assign_heterogeneous_connected_regions(
    graph: RegionGraph,
    config: PlannerConfig,
    agent_patterns: Dict[int, Dict[str, List[RegionCoveragePattern]]],
    path_config: PathPlanningConfig,
) -> BalancedAssignment:
    """Build a capability-aware connected assignment using estimated completion time.

    The growth step is rooted at the agents' starts.  Improvement operations are
    deliberately local: a boundary migration or swap is accepted only when both
    partitions remain connected and the lexicographic assignment objective
    improves.
    """

    agent_count = config.fleet.num_agents or len(config.fleet.initial_states_3dof)
    agents = list(range(agent_count))
    agent_regions: Dict[int, List[str]] = {agent_id: [] for agent_id in agents}
    workload, assignable, best_pattern_ids = _heterogeneous_workload_matrix(
        graph,
        config,
        agent_patterns,
        path_config,
    )
    remaining = set(graph.regions)
    unassigned_reasons: Dict[str, str] = {}

    # Distinct roots let the subsequent growth preserve one connected component
    # per active agent whenever the region graph itself is connected.
    for agent_id in agents:
        feasible = [region_id for region_id in remaining if assignable[agent_id].get(region_id, False)]
        if not feasible:
            continue
        state = config.fleet.initial_states_3dof[agent_id]
        root = min(
            feasible,
            key=lambda region_id: (
                math.hypot(graph.regions[region_id].center[0] - state.x, graph.regions[region_id].center[1] - state.y)
                / max(config.profile_for_agent(agent_id).cruise_speed, 1e-6)
                + workload[agent_id][region_id],
                -graph.regions[region_id].area,
                region_id,
            ),
        )
        agent_regions[agent_id].append(root)
        remaining.remove(root)

    disconnected_seed_count = 0
    while remaining:
        loads = _heterogeneous_loads(agent_regions, workload)
        choices: List[Tuple[Tuple[float, float, float, float, str], int, str]] = []
        for region_id in remaining:
            region = graph.regions[region_id]
            for agent_id in agents:
                if not assignable[agent_id].get(region_id, False):
                    continue
                owned = agent_regions[agent_id]
                if owned and not any(neighbor in owned for neighbor in graph.adjacency.get(region_id, [])):
                    continue
                trial_loads = dict(loads)
                trial_loads[agent_id] += workload[agent_id][region_id]
                mission_limit = config.profile_for_agent(agent_id).max_mission_time
                if mission_limit is not None and trial_loads[agent_id] > mission_limit + 1e-9:
                    continue
                same_axis_neighbors = sum(
                    1
                    for neighbor in graph.adjacency.get(region_id, [])
                    if neighbor in owned
                    and _best_scan_axis(agent_patterns[agent_id].get(neighbor, []))
                    == _best_scan_axis(agent_patterns[agent_id].get(region_id, []))
                )
                state = config.fleet.initial_states_3dof[agent_id]
                distance = math.hypot(region.center[0] - state.x, region.center[1] - state.y)
                score = (
                    max(trial_loads.values(), default=0.0),
                    trial_loads[agent_id],
                    -float(same_axis_neighbors),
                    distance,
                    region_id,
                )
                choices.append((score, agent_id, region_id))
        if choices:
            _, agent_id, region_id = min(choices, key=lambda item: item[0])
            agent_regions[agent_id].append(region_id)
            remaining.remove(region_id)
            continue

        # A disconnected graph component cannot be reached by frontier growth.
        # Seed it explicitly but expose the condition instead of silently
        # claiming connectedness.
        region_id = max(remaining, key=lambda item: (graph.regions[item].area, item))
        loads = _heterogeneous_loads(agent_regions, workload)
        feasible_agents = [
            agent_id
            for agent_id in agents
            if assignable[agent_id].get(region_id, False)
            and (
                config.profile_for_agent(agent_id).max_mission_time is None
                or loads[agent_id] + workload[agent_id][region_id]
                <= float(config.profile_for_agent(agent_id).max_mission_time) + 1e-9
            )
        ]
        if not feasible_agents:
            geometric_feasible_agents = [
                agent_id for agent_id in agents if assignable[agent_id].get(region_id, False)
            ]
            unassigned_reasons[region_id] = (
                "mission_time_limit" if geometric_feasible_agents else "no_agent_has_feasible_pattern"
            )
            remaining.remove(region_id)
            continue
        agent_id = min(
            feasible_agents,
            key=lambda item: (loads[item] + workload[item][region_id], loads[item], item),
        )
        agent_regions[agent_id].append(region_id)
        remaining.remove(region_id)
        disconnected_seed_count += 1

    migration_records: List[Dict[str, object]] = []
    exchange_records: List[Dict[str, object]] = []
    reject_reasons: Dict[str, int] = {}
    for _ in range(max(int(path_config.joint_assignment_iterations), 0)):
        current_objective = _heterogeneous_assignment_objective(agent_regions, workload, graph)
        candidate = _best_heterogeneous_boundary_migration(
            agent_regions,
            graph,
            workload,
            assignable,
            current_objective,
            reject_reasons,
            config,
        )
        if candidate is not None:
            source, target, region_id, objective = candidate
            agent_regions[source].remove(region_id)
            agent_regions[target].append(region_id)
            migration_records.append(
                {"region_id": region_id, "from_agent": source, "to_agent": target, "objective": list(objective)}
            )
            continue
        exchange = _best_heterogeneous_boundary_exchange(
            agent_regions,
            graph,
            workload,
            assignable,
            current_objective,
            reject_reasons,
            config,
        )
        if exchange is None:
            break
        first_agent, second_agent, first_region, second_region, objective = exchange
        agent_regions[first_agent].remove(first_region)
        agent_regions[second_agent].remove(second_region)
        agent_regions[first_agent].append(second_region)
        agent_regions[second_agent].append(first_region)
        exchange_records.append(
            {
                "first_agent": first_agent,
                "second_agent": second_agent,
                "first_region": first_region,
                "second_region": second_region,
                "objective": list(objective),
            }
        )

    for region_ids in agent_regions.values():
        region_ids.sort(key=lambda region_id: graph.regions[region_id].center)
    loads = _heterogeneous_loads(agent_regions, workload)
    connected = {agent_id: graph_is_connected(graph, region_ids) for agent_id, region_ids in agent_regions.items()}
    active_loads = [load for agent_id, load in loads.items() if agent_regions[agent_id]]
    average = sum(active_loads) / max(len(active_loads), 1)
    imbalance = 0.0 if average <= 1e-9 else (max(active_loads) - min(active_loads)) / average
    objective = _heterogeneous_assignment_objective(agent_regions, workload, graph)
    diagnostics = {
        "status": "complete" if not unassigned_reasons and all(connected.values()) else "partial",
        "assignment_method": "heterogeneous_connected_multisource_growth",
        "assignment_objective": path_config.assignment_objective,
        "region_assignability_matrix": json.dumps(assignable, sort_keys=True),
        "region_workload_matrix": json.dumps(workload, sort_keys=True),
        "region_best_pattern_matrix": json.dumps(best_pattern_ids, sort_keys=True),
        "unassigned_region_reasons": json.dumps(unassigned_reasons, sort_keys=True),
        "boundary_migration_records": json.dumps(migration_records, sort_keys=True),
        "boundary_exchange_records": json.dumps(exchange_records, sort_keys=True),
        "assignment_reject_reasons": json.dumps(reject_reasons, sort_keys=True),
        "disconnected_component_seed_count": str(disconnected_seed_count),
        "estimated_makespan": f"{max(active_loads, default=0.0):.6f}",
    }
    return BalancedAssignment(
        agent_regions=agent_regions,
        loads=loads,
        connected=connected,
        imbalance_ratio=imbalance,
        objective=float(objective[2]),
        diagnostics=diagnostics,
    )


def _heterogeneous_workload_matrix(
    graph: RegionGraph,
    config: PlannerConfig,
    agent_patterns: Dict[int, Dict[str, List[RegionCoveragePattern]]],
    path_config: PathPlanningConfig,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, bool]], Dict[int, Dict[str, str]]]:
    workload: Dict[int, Dict[str, float]] = {}
    assignable: Dict[int, Dict[str, bool]] = {}
    best_pattern_ids: Dict[int, Dict[str, str]] = {}
    target = max(float(path_config.min_sweep_pattern_coverage_fraction), 0.0)
    for agent_id, region_patterns in agent_patterns.items():
        workload[agent_id] = {}
        assignable[agent_id] = {}
        best_pattern_ids[agent_id] = {}
        profile = config.profile_for_agent(agent_id)
        state = config.fleet.initial_states_3dof[agent_id]
        for region_id, region in graph.regions.items():
            candidates = []
            for pattern in region_patterns.get(region_id, []):
                try:
                    fraction = float(pattern.metadata.get("estimated_region_coverage_fraction", "0"))
                except (TypeError, ValueError):
                    fraction = 0.0
                if not pattern.feasible or fraction + 1e-9 < target:
                    continue
                failed_retractions = float(pattern.metadata.get("retraction_failed_pass_count", "0") or 0.0)
                open_breaks = float(pattern.metadata.get("open_chain_break_count", "0") or 0.0)
                endpoint_clearance = float(pattern.metadata.get("endpoint_min_clearance", "0") or 0.0)
                endpoint_penalty = 1.0 / max(endpoint_clearance, 0.25) if endpoint_clearance > 0.0 else 0.0
                estimated = (
                    pattern.coverage_length / max(profile.cover_speed, 1e-6)
                    + pattern.turn_length / max(profile.turn_speed_max, 1e-6)
                    + pattern.turn_angle / max(profile.yaw_rate_limit, 1e-6)
                    + failed_retractions * 5.0
                    + open_breaks * 8.0
                    + endpoint_penalty
                )
                transit = math.hypot(pattern.entry_pose.x - state.x, pattern.entry_pose.y - state.y) / max(
                    profile.cruise_speed,
                    1e-6,
                )
                candidates.append((estimated + 0.15 * transit, pattern))
            if not candidates:
                assignable[agent_id][region_id] = False
                continue
            best_cost, best_pattern = min(candidates, key=lambda item: (item[0], item[1].total_length, item[1].pattern_id))
            if profile.max_mission_time is not None and best_cost > profile.max_mission_time + 1e-9:
                assignable[agent_id][region_id] = False
                continue
            assignable[agent_id][region_id] = True
            workload[agent_id][region_id] = float(best_cost)
            best_pattern_ids[agent_id][region_id] = best_pattern.pattern_id
    return workload, assignable, best_pattern_ids


def _heterogeneous_loads(
    agent_regions: Dict[int, List[str]],
    workload: Dict[int, Dict[str, float]],
) -> Dict[int, float]:
    return {
        agent_id: sum(workload.get(agent_id, {}).get(region_id, float("inf")) for region_id in region_ids)
        for agent_id, region_ids in agent_regions.items()
    }


def _heterogeneous_assignment_objective(
    agent_regions: Dict[int, List[str]],
    workload: Dict[int, Dict[str, float]],
    graph: RegionGraph,
) -> Tuple[float, float, float, float]:
    assigned = {region_id for region_ids in agent_regions.values() for region_id in region_ids}
    unassigned = set(graph.regions) - assigned
    unassigned_area = sum(graph.regions[region_id].area for region_id in unassigned)
    loads = _heterogeneous_loads(agent_regions, workload)
    active = [loads[agent_id] for agent_id, region_ids in agent_regions.items() if region_ids]
    return (float(len(unassigned)), unassigned_area, max(active, default=0.0), sum(active))


def _best_heterogeneous_boundary_migration(
    agent_regions: Dict[int, List[str]],
    graph: RegionGraph,
    workload: Dict[int, Dict[str, float]],
    assignable: Dict[int, Dict[str, bool]],
    current_objective: Tuple[float, float, float, float],
    reject_reasons: Dict[str, int],
    config: PlannerConfig,
) -> Tuple[int, int, str, Tuple[float, float, float, float]] | None:
    best = None
    for source, source_regions in agent_regions.items():
        if len(source_regions) <= 1:
            continue
        for region_id in source_regions:
            remaining_source = [item for item in source_regions if item != region_id]
            if not graph_is_connected(graph, remaining_source):
                _increment_reason(reject_reasons, "migration_source_disconnect")
                continue
            for target, target_regions in agent_regions.items():
                if target == source or not assignable.get(target, {}).get(region_id, False):
                    continue
                if target_regions and not any(neighbor in target_regions for neighbor in graph.adjacency.get(region_id, [])):
                    continue
                trial = {agent_id: list(regions) for agent_id, regions in agent_regions.items()}
                trial[source].remove(region_id)
                trial[target].append(region_id)
                if not _heterogeneous_loads_respect_mission_limits(trial, workload, config):
                    _increment_reason(reject_reasons, "migration_mission_time_limit")
                    continue
                if not graph_is_connected(graph, trial[target]):
                    _increment_reason(reject_reasons, "migration_target_disconnect")
                    continue
                objective = _heterogeneous_assignment_objective(trial, workload, graph)
                if objective >= current_objective:
                    _increment_reason(reject_reasons, "migration_no_lexicographic_improvement")
                    continue
                record = (source, target, region_id, objective)
                if best is None or objective < best[3]:
                    best = record
    return best


def _best_heterogeneous_boundary_exchange(
    agent_regions: Dict[int, List[str]],
    graph: RegionGraph,
    workload: Dict[int, Dict[str, float]],
    assignable: Dict[int, Dict[str, bool]],
    current_objective: Tuple[float, float, float, float],
    reject_reasons: Dict[str, int],
    config: PlannerConfig,
) -> Tuple[int, int, str, str, Tuple[float, float, float, float]] | None:
    best = None
    agents = sorted(agent_regions)
    for index, first_agent in enumerate(agents):
        for second_agent in agents[index + 1 :]:
            for first_region in agent_regions[first_agent]:
                for second_region in agent_regions[second_agent]:
                    if second_region not in graph.adjacency.get(first_region, []):
                        continue
                    if not assignable.get(first_agent, {}).get(second_region, False):
                        continue
                    if not assignable.get(second_agent, {}).get(first_region, False):
                        continue
                    trial = {agent_id: list(regions) for agent_id, regions in agent_regions.items()}
                    trial[first_agent].remove(first_region)
                    trial[second_agent].remove(second_region)
                    trial[first_agent].append(second_region)
                    trial[second_agent].append(first_region)
                    if not _heterogeneous_loads_respect_mission_limits(trial, workload, config):
                        _increment_reason(reject_reasons, "exchange_mission_time_limit")
                        continue
                    if not graph_is_connected(graph, trial[first_agent]) or not graph_is_connected(graph, trial[second_agent]):
                        _increment_reason(reject_reasons, "exchange_disconnect")
                        continue
                    objective = _heterogeneous_assignment_objective(trial, workload, graph)
                    if objective >= current_objective:
                        _increment_reason(reject_reasons, "exchange_no_lexicographic_improvement")
                        continue
                    record = (first_agent, second_agent, first_region, second_region, objective)
                    if best is None or objective < best[4]:
                        best = record
    return best


def _heterogeneous_loads_respect_mission_limits(
    agent_regions: Dict[int, List[str]],
    workload: Dict[int, Dict[str, float]],
    config: PlannerConfig,
) -> bool:
    loads = _heterogeneous_loads(agent_regions, workload)
    for agent_id, load in loads.items():
        limit = config.profile_for_agent(agent_id).max_mission_time
        if limit is not None and load > limit + 1e-9:
            return False
    return True


def _best_scan_axis(patterns: Sequence[RegionCoveragePattern]) -> str:
    feasible = [pattern for pattern in patterns if pattern.feasible]
    if not feasible:
        return ""
    return min(feasible, key=lambda pattern: (pattern.estimated_time, pattern.pattern_id)).scan_axis


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
