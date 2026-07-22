"""Continuous-time exact scheduler for a fixed set of CROWN routes."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isclose
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from .types import CrownRoute, CrownSchedule, OperationKey


_TOL = 1.0e-10


@dataclass(frozen=True)
class _ConflictPair:
    left: OperationKey
    right: OperationKey
    resources: Tuple[str, ...]


def _topological_schedule(
    nodes: Sequence[OperationKey],
    edges: Mapping[Tuple[OperationKey, OperationKey], float],
    durations: Mapping[OperationKey, float],
) -> Optional[Tuple[Dict[OperationKey, float], float]]:
    successors: Dict[OperationKey, list[Tuple[OperationKey, float]]] = {
        node: [] for node in nodes
    }
    indegree = {node: 0 for node in nodes}
    for (source, target), lag in edges.items():
        successors[source].append((target, lag))
        indegree[target] += 1

    ready = sorted(node for node in nodes if indegree[node] == 0)
    starts = {node: 0.0 for node in nodes}
    processed = 0
    while ready:
        node = ready.pop(0)
        processed += 1
        for target, lag in sorted(successors[node], key=lambda item: item[0]):
            starts[target] = max(starts[target], starts[node] + lag)
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()

    if processed != len(nodes):
        return None
    makespan = max((starts[node] + durations[node] for node in nodes), default=0.0)
    return starts, makespan


def _conflict_pairs(routes: Sequence[CrownRoute]) -> Tuple[_ConflictPair, ...]:
    pairs = []
    for left_index, left_route in enumerate(routes):
        for right_route in routes[left_index + 1 :]:
            if left_route.agent_id == right_route.agent_id:
                raise ValueError("exactly one selected route is allowed per agent")
            for left_operation_index, left_operation in enumerate(left_route.operations):
                left_resources = set(left_operation.resource_ids)
                if not left_resources:
                    continue
                for right_operation_index, right_operation in enumerate(right_route.operations):
                    shared = tuple(sorted(left_resources.intersection(right_operation.resource_ids)))
                    if shared:
                        pairs.append(
                            _ConflictPair(
                                left=(left_route.agent_id, left_operation_index),
                                right=(right_route.agent_id, right_operation_index),
                                resources=shared,
                            )
                        )
    return tuple(pairs)


def schedule_selected_routes_exact(
    routes: Sequence[CrownRoute],
    *,
    separation_time: float = 0.0,
    resource_capacities: Optional[Mapping[str, int]] = None,
    wait_energy_rates: Optional[Mapping[str, float]] = None,
) -> CrownSchedule:
    """Return the minimum-makespan schedule for fixed route columns.

    Every pair of cross-agent operations sharing a unary resource yields one
    disjunction.  The solver enumerates both precedence orientations.  For a
    fixed orientation set, a directed acyclic precedence graph is solved by a
    longest-path pass; cyclic orientations are infeasible.  This is exponential
    in the number of conflicts by design and therefore serves as the exact
    small-instance oracle.
    """

    selected = tuple(routes)
    if separation_time < 0.0:
        raise ValueError("separation_time must be non-negative")
    if len({route.agent_id for route in selected}) != len(selected):
        raise ValueError("routes must contain at most one route per agent")

    capacities = dict(resource_capacities or {})
    used_resources: Set[str] = {
        resource
        for route in selected
        for operation in route.operations
        for resource in operation.resource_ids
    }
    non_unary = [resource for resource in used_resources if capacities.get(resource, 1) != 1]
    if non_unary:
        raise NotImplementedError(
            "the continuous exact scheduler currently supports unary resources only; "
            f"non-unary resources: {sorted(non_unary)}"
        )

    nodes = tuple(
        (route.agent_id, operation_index)
        for route in selected
        for operation_index, _ in enumerate(route.operations)
    )
    durations = {
        (route.agent_id, operation_index): operation.duration
        for route in selected
        for operation_index, operation in enumerate(route.operations)
    }
    base_edges: Dict[Tuple[OperationKey, OperationKey], float] = {}
    for route in selected:
        for operation_index in range(len(route.operations) - 1):
            source = (route.agent_id, operation_index)
            target = (route.agent_id, operation_index + 1)
            base_edges[(source, target)] = durations[source]

    conflicts = _conflict_pairs(selected)
    wait_rates = dict(wait_energy_rates or {})
    if any(rate < 0.0 for rate in wait_rates.values()):
        raise ValueError("wait energy rates must be non-negative")
    service_times = {route.agent_id: route.nominal_duration for route in selected}
    best_objective = (inf, inf)
    best_starts: Optional[Dict[OperationKey, float]] = None
    orientations_evaluated = 0

    def search(index: int, edges: Dict[Tuple[OperationKey, OperationKey], float]) -> None:
        nonlocal best_objective, best_starts, orientations_evaluated
        partial = _topological_schedule(nodes, edges, durations)
        if partial is None:
            return
        partial_starts, lower_bound = partial
        if lower_bound > best_objective[0] + _TOL:
            return
        if index == len(conflicts):
            orientations_evaluated += 1
            completion = {}
            for route in selected:
                completion[route.agent_id] = (
                    partial_starts[(route.agent_id, len(route.operations) - 1)]
                    + durations[(route.agent_id, len(route.operations) - 1)]
                    if route.operations
                    else 0.0
                )
            waiting_energy = sum(
                wait_rates.get(agent_id, 0.0)
                * max(0.0, completion[agent_id] - service_times[agent_id])
                for agent_id in completion
            )
            objective = (lower_bound, waiting_energy)
            if (
                objective[0] < best_objective[0] - _TOL
                or (
                    abs(objective[0] - best_objective[0]) <= _TOL
                    and objective[1] < best_objective[1] - _TOL
                )
            ):
                best_objective = objective
                best_starts = partial_starts
            return

        conflict = conflicts[index]
        choices = ((conflict.left, conflict.right), (conflict.right, conflict.left))
        for source, target in choices:
            next_edges = dict(edges)
            lag = durations[source] + separation_time
            next_edges[(source, target)] = max(next_edges.get((source, target), 0.0), lag)
            search(index + 1, next_edges)

    search(0, base_edges)
    if best_starts is None:
        # With wait allowed and unary resources this should only be reachable if
        # an inconsistent extension is added in the future.
        raise RuntimeError("no feasible orientation of the resource conflicts exists")

    finishes = {node: best_starts[node] + durations[node] for node in nodes}
    completion_times = {}
    for route in selected:
        if route.operations:
            completion_times[route.agent_id] = finishes[
                (route.agent_id, len(route.operations) - 1)
            ]
        else:
            completion_times[route.agent_id] = 0.0

    actual_makespan = max(completion_times.values(), default=0.0)
    if not isclose(actual_makespan, best_objective[0], rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise AssertionError("internal makespan inconsistency")
    return CrownSchedule(
        starts=best_starts,
        finishes=finishes,
        agent_completion_times=completion_times,
        makespan=actual_makespan,
        waiting_energy=best_objective[1],
        conflict_pairs=len(conflicts),
        orientations_evaluated=orientations_evaluated,
    )


def assert_schedule_resource_feasible(
    routes: Sequence[CrownRoute],
    schedule: CrownSchedule,
    *,
    separation_time: float = 0.0,
) -> None:
    """Raise ``AssertionError`` if a returned schedule has an overlap."""

    route_by_agent = {route.agent_id: route for route in routes}
    windows: Dict[str, list[Tuple[float, float, str]]] = {}
    for (agent_id, operation_index), start in schedule.starts.items():
        operation = route_by_agent[agent_id].operations[operation_index]
        finish = schedule.finishes[(agent_id, operation_index)]
        for resource in operation.resource_ids:
            windows.setdefault(resource, []).append((start, finish, agent_id))

    for resource, resource_windows in windows.items():
        ordered = sorted(resource_windows)
        for left_index, (left_start, left_finish, left_agent) in enumerate(ordered):
            for right_start, right_finish, right_agent in ordered[left_index + 1 :]:
                if left_agent == right_agent:
                    continue
                separated = (
                    left_finish + separation_time <= right_start + _TOL
                    or right_finish + separation_time <= left_start + _TOL
                )
                if not separated:
                    raise AssertionError(
                        f"resource {resource!r} overlaps for {left_agent!r} and "
                        f"{right_agent!r}: {(left_start, left_finish)} vs "
                        f"{(right_start, right_finish)}"
                    )
