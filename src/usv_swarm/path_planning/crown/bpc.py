"""Minimal exact branch-price-and-cut solver for finite CROWN columns."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .lp import LinearProgramInfeasible, LinearProgramSolution, solve_linear_program
from .resource_model import build_time_expanded_route_universe, violated_resource_slots
from .types import CrownBpcSolution, CrownInstance, CrownTimedRoute, ResourceSlot


_TOL = 1.0e-7


@dataclass(frozen=True)
class _BranchNode:
    forced_task_agents: FrozenSet[Tuple[str, str]] = frozenset()
    forbidden_task_agents: FrozenSet[Tuple[str, str]] = frozenset()
    fixed_routes: FrozenSet[Tuple[str, str]] = frozenset()
    forbidden_routes: FrozenSet[str] = frozenset()
    depth: int = 0


@dataclass
class _Statistics:
    generated_column_ids: Set[str] = field(default_factory=set)
    pricing_iterations: int = 0
    branch_nodes: int = 0
    conflict_separation_rounds: int = 0


@dataclass(frozen=True)
class _MasterResult:
    columns: Tuple[CrownTimedRoute, ...]
    route_values: np.ndarray
    objective: float
    equality_duals: Mapping[Tuple[str, str], float]
    inequality_duals: Mapping[Tuple[object, ...], float]


def _fixed_route_map(node: _BranchNode) -> Optional[Mapping[str, str]]:
    result: Dict[str, str] = {}
    for agent_id, route_id in node.fixed_routes:
        previous = result.get(agent_id)
        if previous is not None and previous != route_id:
            return None
        result[agent_id] = route_id
    return result


def _eligible_routes(
    instance: CrownInstance,
    universe: Mapping[str, Tuple[CrownTimedRoute, ...]],
    node: _BranchNode,
) -> Mapping[str, Tuple[CrownTimedRoute, ...]]:
    fixed = _fixed_route_map(node)
    if fixed is None:
        return {agent_id: () for agent_id in instance.agent_ids}
    forced: Dict[str, str] = {}
    for task_id, agent_id in node.forced_task_agents:
        if task_id in forced and forced[task_id] != agent_id:
            return {candidate_agent: () for candidate_agent in instance.agent_ids}
        if (task_id, agent_id) in node.forbidden_task_agents:
            return {candidate_agent: () for candidate_agent in instance.agent_ids}
        forced[task_id] = agent_id
    result = {}
    for agent_id in instance.agent_ids:
        candidates = []
        for route in universe[agent_id]:
            if route.timed_route_id in node.forbidden_routes:
                continue
            if agent_id in fixed and route.timed_route_id != fixed[agent_id]:
                continue
            tasks = set(route.task_ids)
            invalid = False
            for task_id, forced_agent in forced.items():
                if forced_agent == agent_id and task_id not in tasks:
                    invalid = True
                    break
                if forced_agent != agent_id and task_id in tasks:
                    invalid = True
                    break
            if invalid:
                continue
            if any(
                forbidden_agent == agent_id and task_id in tasks
                for task_id, forbidden_agent in node.forbidden_task_agents
            ):
                continue
            candidates.append(route)
        result[agent_id] = tuple(candidates)
    return result


def _find_integer_seed(
    instance: CrownInstance,
    eligible: Mapping[str, Tuple[CrownTimedRoute, ...]],
    active_resources: Set[ResourceSlot],
) -> Optional[Tuple[CrownTimedRoute, ...]]:
    target_tasks = frozenset(instance.task_ids)
    counts: Dict[ResourceSlot, int] = {}
    selected: list[CrownTimedRoute] = []

    def search(agent_index: int, covered: FrozenSet[str]) -> bool:
        if agent_index == len(instance.agent_ids):
            return covered == target_tasks
        agent_id = instance.agent_ids[agent_index]
        for route in eligible[agent_id]:
            route_tasks = frozenset(route.task_ids)
            if covered.intersection(route_tasks):
                continue
            touched = []
            feasible = True
            for resource in route.occupied_resource_slots:
                if resource not in active_resources:
                    continue
                new_count = counts.get(resource, 0) + 1
                if new_count > instance.capacity(resource[0]):
                    feasible = False
                    break
                counts[resource] = new_count
                touched.append(resource)
            if feasible:
                selected.append(route)
                if search(agent_index + 1, covered.union(route_tasks)):
                    return True
                selected.pop()
            for resource in touched:
                counts[resource] -= 1
                if counts[resource] == 0:
                    del counts[resource]
        return False

    if not search(0, frozenset()):
        return None
    return tuple(selected)


def _solve_restricted_master(
    instance: CrownInstance,
    columns: Sequence[CrownTimedRoute],
    active_resources: Set[ResourceSlot],
    *,
    stage: int,
) -> _MasterResult:
    routes = tuple(sorted(columns, key=lambda route: route.timed_route_id))
    route_count = len(routes)
    has_makespan = stage == 1
    variable_count = route_count + int(has_makespan)

    equality_names: list[Tuple[str, str]] = []
    equality_rows = []
    equality_rhs = []
    for agent_id in instance.agent_ids:
        row = np.zeros(variable_count)
        for index, route in enumerate(routes):
            if route.agent_id == agent_id:
                row[index] = 1.0
        equality_names.append(("agent", agent_id))
        equality_rows.append(row)
        equality_rhs.append(1.0)
    for task_id in instance.task_ids:
        row = np.zeros(variable_count)
        for index, route in enumerate(routes):
            if task_id in route.task_ids:
                row[index] = 1.0
        equality_names.append(("task", task_id))
        equality_rows.append(row)
        equality_rhs.append(1.0)

    inequality_names: list[Tuple[object, ...]] = []
    inequality_rows = []
    inequality_rhs = []
    if has_makespan:
        makespan_index = route_count
        for agent_id in instance.agent_ids:
            row = np.zeros(variable_count)
            for index, route in enumerate(routes):
                if route.agent_id == agent_id:
                    row[index] = route.finish_time
            row[makespan_index] = -1.0
            inequality_names.append(("makespan", agent_id))
            inequality_rows.append(row)
            inequality_rhs.append(0.0)
    for resource_slot in sorted(active_resources):
        row = np.zeros(variable_count)
        for index, route in enumerate(routes):
            if resource_slot in route.occupied_resource_slots:
                row[index] = 1.0
        inequality_names.append(("resource",) + resource_slot)
        inequality_rows.append(row)
        inequality_rhs.append(float(instance.capacity(resource_slot[0])))

    objective = np.zeros(variable_count)
    if has_makespan:
        objective[-1] = 1.0
    else:
        for index, route in enumerate(routes):
            objective[index] = route.energy

    result: LinearProgramSolution = solve_linear_program(
        objective,
        equality_rows,
        equality_rhs,
        inequality_rows,
        inequality_rhs,
    )
    return _MasterResult(
        columns=routes,
        route_values=result.values[:route_count],
        objective=result.objective,
        equality_duals=dict(zip(equality_names, result.equality_duals)),
        inequality_duals=dict(zip(inequality_names, result.inequality_duals)),
    )


def _reduced_cost(
    route: CrownTimedRoute,
    master: _MasterResult,
    active_resources: Set[ResourceSlot],
    *,
    stage: int,
) -> float:
    cost = 0.0 if stage == 1 else route.energy
    dual_contribution = master.equality_duals[("agent", route.agent_id)]
    dual_contribution += sum(
        master.equality_duals[("task", task_id)] for task_id in route.task_ids
    )
    if stage == 1:
        dual_contribution += (
            master.inequality_duals[("makespan", route.agent_id)] * route.finish_time
        )
    occupied = set(route.occupied_resource_slots)
    dual_contribution += sum(
        master.inequality_duals[("resource",) + resource_slot]
        for resource_slot in active_resources
        if resource_slot in occupied
    )
    return cost - dual_contribution


def _column_generation(
    instance: CrownInstance,
    eligible: Mapping[str, Tuple[CrownTimedRoute, ...]],
    seed: Sequence[CrownTimedRoute],
    active_resources: Set[ResourceSlot],
    *,
    stage: int,
    statistics: _Statistics,
) -> _MasterResult:
    pool = {route.timed_route_id: route for route in seed}
    while True:
        master = _solve_restricted_master(
            instance,
            tuple(pool.values()),
            active_resources,
            stage=stage,
        )
        additions = []
        for agent_id in instance.agent_ids:
            priced = [
                (_reduced_cost(route, master, active_resources, stage=stage), route)
                for route in eligible[agent_id]
                if route.timed_route_id not in pool
            ]
            if not priced:
                continue
            reduced_cost, route = min(priced, key=lambda item: (item[0], item[1].timed_route_id))
            if reduced_cost < -_TOL:
                additions.append(route)
        statistics.pricing_iterations += 1
        if not additions:
            return master
        for route in additions:
            pool[route.timed_route_id] = route
            statistics.generated_column_ids.add(route.timed_route_id)


def _is_integral(values: np.ndarray) -> bool:
    return all(abs(value - round(value)) <= _TOL for value in values)


def _selected_integral_routes(master: _MasterResult) -> Tuple[CrownTimedRoute, ...]:
    return tuple(
        route
        for route, value in zip(master.columns, master.route_values)
        if value >= 1.0 - _TOL
    )


def _branch_children(
    instance: CrownInstance,
    node: _BranchNode,
    master: _MasterResult,
) -> Tuple[_BranchNode, _BranchNode]:
    # Ryan-Foster-style task-agent branching remains visible to pricing.
    for task_id in instance.task_ids:
        for agent_id in instance.agent_ids:
            assignment_value = sum(
                value
                for route, value in zip(master.columns, master.route_values)
                if route.agent_id == agent_id and task_id in route.task_ids
            )
            if _TOL < assignment_value < 1.0 - _TOL:
                forced = _BranchNode(
                    forced_task_agents=node.forced_task_agents.union({(task_id, agent_id)}),
                    forbidden_task_agents=node.forbidden_task_agents,
                    fixed_routes=node.fixed_routes,
                    forbidden_routes=node.forbidden_routes,
                    depth=node.depth + 1,
                )
                forbidden = _BranchNode(
                    forced_task_agents=node.forced_task_agents,
                    forbidden_task_agents=node.forbidden_task_agents.union({(task_id, agent_id)}),
                    fixed_routes=node.fixed_routes,
                    forbidden_routes=node.forbidden_routes,
                    depth=node.depth + 1,
                )
                return forced, forbidden

    # If assignment aggregates are integral, branch on a fractional timed route.
    candidates = [
        (abs(value - 0.5), route.timed_route_id, route)
        for route, value in zip(master.columns, master.route_values)
        if _TOL < value < 1.0 - _TOL
    ]
    if not candidates:
        raise RuntimeError("fractional master has no branchable variable")
    _, _, route = min(candidates)
    included = _BranchNode(
        forced_task_agents=node.forced_task_agents,
        forbidden_task_agents=node.forbidden_task_agents,
        fixed_routes=node.fixed_routes.union({(route.agent_id, route.timed_route_id)}),
        forbidden_routes=node.forbidden_routes,
        depth=node.depth + 1,
    )
    excluded = _BranchNode(
        forced_task_agents=node.forced_task_agents,
        forbidden_task_agents=node.forbidden_task_agents,
        fixed_routes=node.fixed_routes,
        forbidden_routes=node.forbidden_routes.union({route.timed_route_id}),
        depth=node.depth + 1,
    )
    return included, excluded


def _solve_phase(
    instance: CrownInstance,
    universe: Mapping[str, Tuple[CrownTimedRoute, ...]],
    active_resources: Set[ResourceSlot],
    *,
    stage: int,
    statistics: _Statistics,
) -> Tuple[float, Tuple[CrownTimedRoute, ...]]:
    incumbent = inf
    incumbent_routes: Optional[Tuple[CrownTimedRoute, ...]] = None
    stack = [_BranchNode()]

    while stack:
        node = stack.pop()
        statistics.branch_nodes += 1
        while True:
            eligible = _eligible_routes(instance, universe, node)
            if any(not eligible[agent_id] for agent_id in instance.agent_ids):
                break
            seed = _find_integer_seed(instance, eligible, active_resources)
            if seed is None:
                break
            statistics.generated_column_ids.update(route.timed_route_id for route in seed)
            try:
                master = _column_generation(
                    instance,
                    eligible,
                    seed,
                    active_resources,
                    stage=stage,
                    statistics=statistics,
                )
            except LinearProgramInfeasible:
                break
            if master.objective >= incumbent - _TOL:
                break
            if _is_integral(master.route_values):
                selected = _selected_integral_routes(master)
                violations = violated_resource_slots(selected, instance.resource_capacities)
                new_resources = set(violations).difference(active_resources)
                if new_resources:
                    active_resources.update(new_resources)
                    statistics.conflict_separation_rounds += 1
                    # Crucially restart the RMP and exact pricing with new duals.
                    continue
                candidate = (
                    max((route.finish_time for route in selected), default=0.0)
                    if stage == 1
                    else sum(route.energy for route in selected)
                )
                if candidate < incumbent - _TOL:
                    incumbent = candidate
                    incumbent_routes = selected
                break

            included, excluded = _branch_children(instance, node, master)
            # LIFO explores the include branch first and retains deterministic output.
            stack.append(excluded)
            stack.append(included)
            break

    if incumbent_routes is None:
        raise ValueError("the finite time-expanded CROWN master is infeasible")
    return incumbent, incumbent_routes


def solve_crown_bpc(
    instance: CrownInstance,
    *,
    horizon: float,
    time_step: float = 1.0,
    wait_energy_rate: Optional[Union[float, Mapping[str, float]]] = None,
    max_timed_columns: int = 200_000,
) -> CrownBpcSolution:
    """Solve the finite time-expanded model to lexicographic global optimality.

    The pricing oracle exhaustively scans the finite timed-route universe, so a
    terminated pricing loop proves that no negative-reduced-cost column exists.
    Branch decisions are propagated into that oracle.  Shared resource rows are
    separated lazily from integer solutions and *always* followed by re-pricing.
    """

    universe, horizon_slots = build_time_expanded_route_universe(
        instance,
        horizon=horizon,
        time_step=time_step,
        wait_energy_rate=(instance.wait_energy_rates if wait_energy_rate is None else wait_energy_rate),
        max_timed_columns=max_timed_columns,
    )
    statistics = _Statistics()
    active_resources: Set[ResourceSlot] = set()
    best_makespan, _ = _solve_phase(
        instance,
        universe,
        active_resources,
        stage=1,
        statistics=statistics,
    )

    lexicographic_universe = {
        agent_id: tuple(
            route
            for route in routes
            if route.finish_time <= best_makespan + _TOL
        )
        for agent_id, routes in universe.items()
    }
    best_energy, selected = _solve_phase(
        instance,
        lexicographic_universe,
        active_resources,
        stage=2,
        statistics=statistics,
    )
    final_makespan = max((route.finish_time for route in selected), default=0.0)
    remaining_violations = violated_resource_slots(selected, instance.resource_capacities)
    if remaining_violations:
        raise AssertionError("BPC returned a resource-infeasible solution")
    if abs(final_makespan - best_makespan) > _TOL:
        raise AssertionError("lexicographic second stage changed the optimal makespan")

    return CrownBpcSolution(
        timed_routes=tuple(sorted(selected, key=lambda route: route.agent_id)),
        makespan=best_makespan,
        total_energy=best_energy,
        lower_bound=best_makespan,
        upper_bound=best_makespan,
        optimality_gap=0.0,
        energy_lower_bound=best_energy,
        energy_upper_bound=best_energy,
        energy_optimality_gap=0.0,
        active_conflict_resources=tuple(sorted(active_resources)),
        generated_columns=len(statistics.generated_column_ids),
        pricing_iterations=statistics.pricing_iterations,
        branch_nodes=statistics.branch_nodes,
        conflict_separation_rounds=statistics.conflict_separation_rounds,
        time_step=time_step,
        horizon_slots=horizon_slots,
    )
