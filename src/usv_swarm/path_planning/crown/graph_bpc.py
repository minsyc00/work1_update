"""Branch-price-and-cut directly over time-expanded CROWN mode graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor, inf
from time import perf_counter
from typing import Dict, FrozenSet, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .conflicts import CrownResourceMappingError, find_continuous_conflicts
from .lp import LinearProgramInfeasible, solve_linear_program
from .mode_graph import CrownTimeExpandedModeGraph
from .pricing import (
    CrownPricingDuals,
    CrownPricingPrecedenceDual,
    CrownPricingRestrictions,
    PricingLabelLimitExceeded,
    PricingTimeLimitExceeded,
    price_mode_graph_exact,
)
from .resource_model import violated_resource_slots
from .types import CrownBpcSolution, CrownRoute, CrownTimedRoute, ResourceSlot


_TOL = 1.0e-7


class CrownColumnLimitExceeded(RuntimeError):
    """Raised instead of silently weakening an exact BPC run."""


@dataclass(frozen=True)
class CrownRootRelaxation:
    objective: float
    route_pool: Tuple[CrownTimedRoute, ...]
    task_duals: Mapping[str, float]
    agent_duals: Mapping[str, float]
    makespan_duals: Mapping[str, float]
    resource_duals: Mapping[ResourceSlot, float]
    service_lower_bound: float
    pricing_iterations: int
    pricing_labels: int
    exact: bool = True


@dataclass(frozen=True)
class CrownPoolDualGuidance:
    """Dual prices of a restricted route-pool LP used only for LNS guidance."""

    task_duals: Mapping[str, float]
    agent_duals: Mapping[str, float]
    makespan_duals: Mapping[str, float]
    resource_duals: Mapping[ResourceSlot, float]
    objective: float


@dataclass(frozen=True)
class _GraphBranchNode:
    forced_task_agents: FrozenSet[Tuple[str, str]] = frozenset()
    forbidden_task_agents: FrozenSet[Tuple[str, str]] = frozenset()
    required_arcs: FrozenSet[Tuple[str, str, str]] = frozenset()
    forbidden_arcs: FrozenSet[Tuple[str, str, str]] = frozenset()
    resource_precedences: FrozenSet[Tuple[str, str, str]] = frozenset()
    fixed_routes: FrozenSet[Tuple[str, str]] = frozenset()
    forbidden_routes: FrozenSet[str] = frozenset()
    depth: int = 0


@dataclass(frozen=True)
class _GraphMasterResult:
    columns: Tuple[CrownTimedRoute, ...]
    route_values: np.ndarray
    objective: float
    equality_duals: Mapping[Tuple[str, str], float]
    inequality_duals: Mapping[Tuple[object, ...], float]
    artificial_sum: float = 0.0


@dataclass
class _GraphStatistics:
    generated_column_ids: Set[str] = field(default_factory=set)
    pricing_iterations: int = 0
    pricing_labels: int = 0
    pricing_labels_dominated: int = 0
    branch_nodes: int = 0
    resource_precedence_branches: int = 0
    route_variable_branches: int = 0
    conflict_separation_rounds: int = 0
    active_resources_peak: int = 0
    root_lp_lower_bound: float = 0.0
    anytime_trace: list[Mapping[str, float]] = field(default_factory=list)


def _empty_timed_route(graph: CrownTimeExpandedModeGraph) -> CrownTimedRoute:
    route_id = f"graph:{graph.agent_id}:empty"
    base = CrownRoute(route_id, graph.agent_id, (), ())
    return CrownTimedRoute(
        timed_route_id=route_id,
        base_route=base,
        start_slots=(),
        duration_slots=(),
        finish_slot=0,
        time_step=graph.crown_config.time_step,
        occupied_resource_slots=(),
        energy=0.0,
    )


def _route_resource_first_entry(
    route: CrownTimedRoute,
    resource_id: str,
) -> Optional[int]:
    return min(
        (
            slot
            for candidate_id, slot in route.occupied_resource_slots
            if candidate_id == resource_id
        ),
        default=None,
    )


def _solve_graph_master(
    agent_ids: Sequence[str],
    task_ids: Sequence[str],
    capacities: Mapping[str, int],
    columns: Sequence[CrownTimedRoute],
    active_resources: Set[ResourceSlot],
    node: _GraphBranchNode,
    *,
    stage: int,
    feasibility: bool,
    horizon_slots: int,
) -> _GraphMasterResult:
    routes = tuple(sorted(columns, key=lambda route: route.timed_route_id))
    route_count = len(routes)
    has_makespan = stage == 1 and not feasibility
    artificial_count = (len(agent_ids) + len(task_ids)) if feasibility else 0
    variable_count = route_count + int(has_makespan) + artificial_count
    makespan_index = route_count if has_makespan else None
    artificial_offset = route_count + int(has_makespan)

    equality_names = []
    equality_rows = []
    equality_rhs = []
    for agent_index, agent_id in enumerate(agent_ids):
        row = np.zeros(variable_count)
        for route_index, route in enumerate(routes):
            if route.agent_id == agent_id:
                row[route_index] = 1.0
        if feasibility:
            row[artificial_offset + agent_index] = 1.0
        equality_names.append(("agent", agent_id))
        equality_rows.append(row)
        equality_rhs.append(1.0)
    for task_index, task_id in enumerate(task_ids):
        row = np.zeros(variable_count)
        for route_index, route in enumerate(routes):
            if task_id in route.task_ids:
                row[route_index] = 1.0
        if feasibility:
            row[artificial_offset + len(agent_ids) + task_index] = 1.0
        equality_names.append(("task", task_id))
        equality_rows.append(row)
        equality_rhs.append(1.0)

    inequality_names = []
    inequality_rows = []
    inequality_rhs = []
    if has_makespan and makespan_index is not None:
        for agent_id in agent_ids:
            row = np.zeros(variable_count)
            for route_index, route in enumerate(routes):
                if route.agent_id == agent_id:
                    row[route_index] = route.finish_time
            row[makespan_index] = -1.0
            inequality_names.append(("makespan", agent_id))
            inequality_rows.append(row)
            inequality_rhs.append(0.0)
    for resource_slot in sorted(active_resources):
        row = np.zeros(variable_count)
        for route_index, route in enumerate(routes):
            if resource_slot in route.occupied_resource_slots:
                row[route_index] = 1.0
        inequality_names.append(("resource",) + resource_slot)
        inequality_rows.append(row)
        inequality_rhs.append(float(capacities.get(resource_slot[0], 1)))
    for before_agent, after_agent, resource_id in sorted(
        node.resource_precedences
    ):
        row = np.zeros(variable_count)
        for route_index, route in enumerate(routes):
            first_entry = _route_resource_first_entry(route, resource_id)
            if first_entry is None:
                continue
            if route.agent_id == before_agent:
                row[route_index] = horizon_slots + first_entry
            elif route.agent_id == after_agent:
                row[route_index] = horizon_slots - first_entry
        inequality_names.append(
            ("precedence", before_agent, after_agent, resource_id)
        )
        inequality_rows.append(row)
        inequality_rhs.append(float(2 * horizon_slots - 1))

    objective = np.zeros(variable_count)
    if feasibility:
        objective[artificial_offset:] = 1.0
    elif stage == 1:
        objective[makespan_index] = 1.0
    else:
        for route_index, route in enumerate(routes):
            objective[route_index] = route.energy

    result = solve_linear_program(
        objective,
        equality_rows,
        equality_rhs,
        inequality_rows,
        inequality_rhs,
    )
    artificial_sum = (
        float(sum(result.values[artificial_offset:])) if feasibility else 0.0
    )
    return _GraphMasterResult(
        columns=routes,
        route_values=result.values[:route_count],
        objective=result.objective,
        equality_duals=dict(zip(equality_names, result.equality_duals)),
        inequality_duals=dict(zip(inequality_names, result.inequality_duals)),
        artificial_sum=artificial_sum,
    )


def _forced_agent_map(node: _GraphBranchNode) -> Optional[Mapping[str, str]]:
    result: Dict[str, str] = {}
    for task_id, agent_id in node.forced_task_agents:
        if task_id in result and result[task_id] != agent_id:
            return None
        if (task_id, agent_id) in node.forbidden_task_agents:
            return None
        result[task_id] = agent_id
    return result


def _fixed_route_map(node: _GraphBranchNode) -> Optional[Mapping[str, str]]:
    result: Dict[str, str] = {}
    for agent_id, route_id in node.fixed_routes:
        if agent_id in result and result[agent_id] != route_id:
            return None
        result[agent_id] = route_id
    return result


def _route_allowed(
    route: CrownTimedRoute,
    node: _GraphBranchNode,
    forced: Mapping[str, str],
    fixed: Mapping[str, str],
    *,
    makespan_limit: Optional[float],
) -> bool:
    if route.timed_route_id in node.forbidden_routes:
        return False
    if route.agent_id in fixed and route.timed_route_id != fixed[route.agent_id]:
        return False
    if makespan_limit is not None and route.finish_time > makespan_limit + _TOL:
        return False
    tasks = set(route.task_ids)
    for task_id, forced_agent in forced.items():
        if forced_agent == route.agent_id and task_id not in tasks:
            return False
        if forced_agent != route.agent_id and task_id in tasks:
            return False
    if any(
        forbidden_agent == route.agent_id and task_id in tasks
        for task_id, forbidden_agent in node.forbidden_task_agents
    ):
        return False
    arcs = set(zip(route.task_ids[:-1], route.task_ids[1:]))
    if any(
        agent_id == route.agent_id and (left, right) not in arcs
        for agent_id, left, right in node.required_arcs
    ):
        return False
    if any(
        agent_id == route.agent_id and (left, right) in arcs
        for agent_id, left, right in node.forbidden_arcs
    ):
        return False
    return True


def _pricing_restrictions(
    graph: CrownTimeExpandedModeGraph,
    node: _GraphBranchNode,
    forced: Mapping[str, str],
) -> CrownPricingRestrictions:
    required_tasks = frozenset(
        task_id for task_id, agent_id in forced.items() if agent_id == graph.agent_id
    )
    forbidden_tasks = frozenset(
        task_id
        for task_id, forced_agent in forced.items()
        if forced_agent != graph.agent_id
    ).union(
        task_id
        for task_id, forbidden_agent in node.forbidden_task_agents
        if forbidden_agent == graph.agent_id
    )
    required_successors = {
        left: right
        for agent_id, left, right in node.required_arcs
        if agent_id == graph.agent_id
    }
    return CrownPricingRestrictions(
        required_tasks=required_tasks,
        forbidden_tasks=forbidden_tasks,
        required_successors=required_successors,
        forbidden_arcs=frozenset(
            (left, right)
            for agent_id, left, right in node.forbidden_arcs
            if agent_id == graph.agent_id
        ),
        forbidden_route_ids=node.forbidden_routes,
    )


def _duals_for_agent(
    master: _GraphMasterResult,
    graph: CrownTimeExpandedModeGraph,
    task_ids: Sequence[str],
    active_resources: Set[ResourceSlot],
    node: _GraphBranchNode,
    *,
    stage: int,
    feasibility: bool,
    horizon_slots: int,
) -> CrownPricingDuals:
    return CrownPricingDuals(
        agent_dual=master.equality_duals[("agent", graph.agent_id)],
        task_duals={
            task_id: master.equality_duals[("task", task_id)]
            for task_id in task_ids
        },
        makespan_dual=(
            0.0
            if feasibility or stage == 2
            else master.inequality_duals[("makespan", graph.agent_id)]
        ),
        resource_duals={
            resource: master.inequality_duals[("resource",) + resource]
            for resource in active_resources
        },
        precedence_duals=tuple(
            CrownPricingPrecedenceDual(
                resource_id=resource_id,
                role=("before" if before_agent == graph.agent_id else "after"),
                horizon_slots=horizon_slots,
                dual=master.inequality_duals[
                    ("precedence", before_agent, after_agent, resource_id)
                ],
            )
            for before_agent, after_agent, resource_id in sorted(
                node.resource_precedences
            )
            if graph.agent_id in {before_agent, after_agent}
        ),
        stage=(1 if feasibility else stage),
    )


def _column_generation(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    capacities: Mapping[str, int],
    global_pool: Dict[str, CrownTimedRoute],
    node: _GraphBranchNode,
    active_resources: Set[ResourceSlot],
    *,
    stage: int,
    horizon_slots: int,
    makespan_limit: Optional[float],
    statistics: _GraphStatistics,
    deadline: Optional[float] = None,
) -> Optional[_GraphMasterResult]:
    forced = _forced_agent_map(node)
    fixed = _fixed_route_map(node)
    if forced is None or fixed is None:
        return None

    def eligible_pool() -> Tuple[CrownTimedRoute, ...]:
        return tuple(
            route
            for route in global_pool.values()
            if _route_allowed(
                route,
                node,
                forced,
                fixed,
                makespan_limit=makespan_limit,
            )
        )

    for feasibility in (True, False):
        while True:
            columns = eligible_pool()
            try:
                master = _solve_graph_master(
                    tuple(graphs),
                    task_ids,
                    capacities,
                    columns,
                    active_resources,
                    node,
                    stage=stage,
                    feasibility=feasibility,
                    horizon_slots=horizon_slots,
                )
            except LinearProgramInfeasible:
                return None
            additions = []
            for agent_id, graph in graphs.items():
                if agent_id in fixed:
                    continue
                pricing_horizon = horizon_slots
                if makespan_limit is not None:
                    pricing_horizon = min(
                        pricing_horizon,
                        int(floor(makespan_limit / graph.crown_config.time_step + _TOL)),
                    )
                result = price_mode_graph_exact(
                    graph,
                    horizon_slots=pricing_horizon,
                    duals=_duals_for_agent(
                        master,
                        graph,
                        task_ids,
                        active_resources,
                        node,
                        stage=stage,
                        feasibility=feasibility,
                        horizon_slots=horizon_slots,
                    ),
                    restrictions=_pricing_restrictions(graph, node, forced),
                    label_limit=graph.crown_config.pricing_label_limit,
                    exact=True,
                    deadline=deadline,
                )
                statistics.pricing_labels += result.labels_created
                statistics.pricing_labels_dominated += result.labels_dominated
                if (
                    result.route is not None
                    and result.reduced_cost < -_TOL
                    and result.route.timed_route_id not in global_pool
                ):
                    additions.append(result.route)
            statistics.pricing_iterations += 1
            for route in additions:
                global_pool[route.timed_route_id] = route
                statistics.generated_column_ids.add(route.timed_route_id)
            column_limit = min(
                graph.crown_config.max_timed_columns for graph in graphs.values()
            )
            if len(global_pool) > column_limit:
                raise CrownColumnLimitExceeded(
                    f"exact graph BPC exceeded {column_limit} timed columns"
                )
            if additions:
                continue
            if feasibility:
                if master.artificial_sum > _TOL:
                    return None
                break
            return master
    raise AssertionError("unreachable column-generation state")


def _is_integral(values: np.ndarray) -> bool:
    return all(abs(value - round(value)) <= _TOL for value in values)


def _selected_routes(master: _GraphMasterResult) -> Tuple[CrownTimedRoute, ...]:
    return tuple(
        route
        for route, value in zip(master.columns, master.route_values)
        if value >= 1.0 - _TOL
    )


def _branch_children(
    agent_ids: Sequence[str],
    task_ids: Sequence[str],
    node: _GraphBranchNode,
    master: _GraphMasterResult,
    active_resources: Set[ResourceSlot],
    capacities: Mapping[str, int],
    statistics: _GraphStatistics,
) -> Tuple[_GraphBranchNode, _GraphBranchNode]:
    for task_id in task_ids:
        for agent_id in agent_ids:
            value = sum(
                route_value
                for route, route_value in zip(master.columns, master.route_values)
                if route.agent_id == agent_id and task_id in route.task_ids
            )
            if _TOL < value < 1.0 - _TOL:
                return (
                    _GraphBranchNode(
                        forced_task_agents=node.forced_task_agents.union({(task_id, agent_id)}),
                        forbidden_task_agents=node.forbidden_task_agents,
                        required_arcs=node.required_arcs,
                        forbidden_arcs=node.forbidden_arcs,
                        resource_precedences=node.resource_precedences,
                        fixed_routes=node.fixed_routes,
                        forbidden_routes=node.forbidden_routes,
                        depth=node.depth + 1,
                    ),
                    _GraphBranchNode(
                        forced_task_agents=node.forced_task_agents,
                        forbidden_task_agents=node.forbidden_task_agents.union({(task_id, agent_id)}),
                        required_arcs=node.required_arcs,
                        forbidden_arcs=node.forbidden_arcs,
                        resource_precedences=node.resource_precedences,
                        fixed_routes=node.fixed_routes,
                        forbidden_routes=node.forbidden_routes,
                        depth=node.depth + 1,
                    ),
                )

    for agent_id in agent_ids:
        for left in task_ids:
            for right in task_ids:
                if left == right:
                    continue
                value = sum(
                    route_value
                    for route, route_value in zip(master.columns, master.route_values)
                    if route.agent_id == agent_id
                    and (left, right) in set(zip(route.task_ids[:-1], route.task_ids[1:]))
                )
                if _TOL < value < 1.0 - _TOL:
                    return (
                        _GraphBranchNode(
                            forced_task_agents=node.forced_task_agents.union(
                                {(left, agent_id), (right, agent_id)}
                            ),
                            forbidden_task_agents=node.forbidden_task_agents,
                            required_arcs=node.required_arcs.union({(agent_id, left, right)}),
                            forbidden_arcs=node.forbidden_arcs,
                            resource_precedences=node.resource_precedences,
                            fixed_routes=node.fixed_routes,
                            forbidden_routes=node.forbidden_routes,
                            depth=node.depth + 1,
                        ),
                        _GraphBranchNode(
                            forced_task_agents=node.forced_task_agents,
                            forbidden_task_agents=node.forbidden_task_agents,
                            required_arcs=node.required_arcs,
                            forbidden_arcs=node.forbidden_arcs.union({(agent_id, left, right)}),
                            resource_precedences=node.resource_precedences,
                            fixed_routes=node.fixed_routes,
                            forbidden_routes=node.forbidden_routes,
                            depth=node.depth + 1,
                        ),
                    )

    # Branch on the first visit to a registered unary narrow resource.  First
    # visits are totally ordered in every capacity-feasible integer solution,
    # even when a route later re-enters the same corridor.  This makes the two
    # children a complete disjunction without imposing an unsafe single-visit
    # assumption.  Fine safety-tube cells remain cut resources, not corridor
    # precedence candidates.
    precedence_candidates = []
    narrow_resources = sorted(
        {
            resource_id
            for resource_id, _ in active_resources
            if not resource_id.startswith("tube:")
            and capacities.get(resource_id, 1) == 1
        }
    )
    ordered_agents = tuple(sorted(agent_ids))
    for resource_id in narrow_resources:
        for left_index, left_agent in enumerate(ordered_agents):
            for right_agent in ordered_agents[left_index + 1 :]:
                if (
                    (left_agent, right_agent, resource_id)
                    in node.resource_precedences
                    or (right_agent, left_agent, resource_id)
                    in node.resource_precedences
                ):
                    continue
                left_routes = [
                    (route, value, _route_resource_first_entry(route, resource_id))
                    for route, value in zip(master.columns, master.route_values)
                    if route.agent_id == left_agent and value > _TOL
                ]
                right_routes = [
                    (route, value, _route_resource_first_entry(route, resource_id))
                    for route, value in zip(master.columns, master.route_values)
                    if route.agent_id == right_agent and value > _TOL
                ]
                left_before = 0.0
                right_before = 0.0
                ties = 0.0
                for _, left_value, left_entry in left_routes:
                    if left_entry is None:
                        continue
                    for _, right_value, right_entry in right_routes:
                        if right_entry is None:
                            continue
                        mass = left_value * right_value
                        if left_entry < right_entry:
                            left_before += mass
                        elif right_entry < left_entry:
                            right_before += mass
                        else:
                            ties += mass
                left_child_mass = left_before + ties
                right_child_mass = right_before + ties
                if left_child_mass <= _TOL or right_child_mass <= _TOL:
                    continue
                precedence_candidates.append(
                    (
                        -min(left_child_mass, right_child_mass),
                        -max(left_child_mass, right_child_mass),
                        resource_id,
                        left_agent,
                        right_agent,
                    )
                )
    if precedence_candidates:
        _, _, resource_id, left_agent, right_agent = min(
            precedence_candidates
        )
        statistics.resource_precedence_branches += 1
        return (
            _GraphBranchNode(
                forced_task_agents=node.forced_task_agents,
                forbidden_task_agents=node.forbidden_task_agents,
                required_arcs=node.required_arcs,
                forbidden_arcs=node.forbidden_arcs,
                resource_precedences=node.resource_precedences.union(
                    {(left_agent, right_agent, resource_id)}
                ),
                fixed_routes=node.fixed_routes,
                forbidden_routes=node.forbidden_routes,
                depth=node.depth + 1,
            ),
            _GraphBranchNode(
                forced_task_agents=node.forced_task_agents,
                forbidden_task_agents=node.forbidden_task_agents,
                required_arcs=node.required_arcs,
                forbidden_arcs=node.forbidden_arcs,
                resource_precedences=node.resource_precedences.union(
                    {(right_agent, left_agent, resource_id)}
                ),
                fixed_routes=node.fixed_routes,
                forbidden_routes=node.forbidden_routes,
                depth=node.depth + 1,
            ),
        )

    fractional = [
        (abs(value - 0.5), route.timed_route_id, route)
        for route, value in zip(master.columns, master.route_values)
        if _TOL < value < 1.0 - _TOL
    ]
    if not fractional:
        raise RuntimeError("fractional graph master has no compatible branch")
    _, _, route = min(fractional)
    statistics.route_variable_branches += 1
    return (
        _GraphBranchNode(
            forced_task_agents=node.forced_task_agents,
            forbidden_task_agents=node.forbidden_task_agents,
            required_arcs=node.required_arcs,
            forbidden_arcs=node.forbidden_arcs,
            resource_precedences=node.resource_precedences,
            fixed_routes=node.fixed_routes.union({(route.agent_id, route.timed_route_id)}),
            forbidden_routes=node.forbidden_routes,
            depth=node.depth + 1,
        ),
        _GraphBranchNode(
            forced_task_agents=node.forced_task_agents,
            forbidden_task_agents=node.forbidden_task_agents,
            required_arcs=node.required_arcs,
            forbidden_arcs=node.forbidden_arcs,
            resource_precedences=node.resource_precedences,
            fixed_routes=node.fixed_routes,
            forbidden_routes=node.forbidden_routes.union({route.timed_route_id}),
            depth=node.depth + 1,
        ),
    )


def service_workload_lower_bound(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
) -> float:
    """Solve Theorem 7's fractional pure-service workload relaxation."""

    variables = []
    for agent_id, graph in graphs.items():
        for task_id in task_ids:
            for mode in graph.modes_for_task(task_id):
                # For an arbitrary user-supplied time-varying current the
                # t=0 nominal duration need not be a lower bound over all
                # departure times.  Falling back to zero is weak but preserves
                # Theorem 7's certification claim.  Built-in zero/uniform
                # fields are time invariant, so their exact service duration
                # is safe here.
                duration_lb = (
                    mode.nominal_duration
                    if bool(getattr(graph.current_field, "time_invariant", False))
                    else 0.0
                )
                variables.append((agent_id, task_id, duration_lb))
    if any(not any(task == task_id for _, task, _ in variables) for task_id in task_ids):
        return inf
    variable_count = len(variables) + 1
    load_index = len(variables)
    objective = np.zeros(variable_count)
    objective[load_index] = 1.0
    equalities = []
    equality_rhs = []
    for task_id in task_ids:
        row = np.zeros(variable_count)
        for index, (_, candidate_task, _) in enumerate(variables):
            if candidate_task == task_id:
                row[index] = 1.0
        equalities.append(row)
        equality_rhs.append(1.0)
    inequalities = []
    inequality_rhs = []
    for agent_id in graphs:
        row = np.zeros(variable_count)
        for index, (candidate_agent, _, duration) in enumerate(variables):
            if candidate_agent == agent_id:
                row[index] = duration
        row[load_index] = -1.0
        inequalities.append(row)
        inequality_rhs.append(0.0)
    try:
        result = solve_linear_program(
            objective,
            equalities,
            equality_rhs,
            inequalities,
            inequality_rhs,
        )
        return result.objective
    except RuntimeError:
        # Large, highly degenerate homogeneous mode libraries can expose a
        # numerically dual-infeasible final basis in the lightweight simplex.
        # Preserve certification with a weaker analytic workload bound rather
        # than treating a numerical lower-bound failure as route infeasibility.
        minimum_by_task = [
            min(
                duration
                for _, candidate_task, duration in variables
                if candidate_task == task_id
            )
            for task_id in task_ids
        ]
        return max(
            max(minimum_by_task, default=0.0),
            sum(minimum_by_task) / max(len(graphs), 1),
        )


def _record_anytime(
    statistics: _GraphStatistics,
    started: float,
    lower_bound: float,
    upper_bound: float,
    active_resources: Set[ResourceSlot],
) -> None:
    gap = (
        max(0.0, (upper_bound - lower_bound) / upper_bound)
        if upper_bound < inf and upper_bound > 0.0
        else inf
    )
    statistics.anytime_trace.append(
        {
            "time": perf_counter() - started,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "gap": gap,
            "columns": float(len(statistics.generated_column_ids)),
            "active_resources": float(len(active_resources)),
        }
    )


def _solve_phase(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    capacities: Mapping[str, int],
    global_pool: Dict[str, CrownTimedRoute],
    active_resources: Set[ResourceSlot],
    *,
    stage: int,
    horizon_slots: int,
    makespan_limit: Optional[float],
    service_lower_bound: float,
    statistics: _GraphStatistics,
    started: float,
) -> Tuple[float, Tuple[CrownTimedRoute, ...]]:
    incumbent = inf
    incumbent_routes: Optional[Tuple[CrownTimedRoute, ...]] = None
    stack = [_GraphBranchNode()]
    while stack:
        node = stack.pop()
        statistics.branch_nodes += 1
        while True:
            master = _column_generation(
                graphs,
                task_ids,
                capacities,
                global_pool,
                node,
                active_resources,
                stage=stage,
                horizon_slots=horizon_slots,
                makespan_limit=makespan_limit,
                statistics=statistics,
                deadline=None,
            )
            if master is None:
                break
            if node.depth == 0 and stage == 1:
                statistics.root_lp_lower_bound = max(
                    statistics.root_lp_lower_bound,
                    master.objective,
                )
            if master.objective >= incumbent - _TOL:
                break
            if _is_integral(master.route_values):
                selected = _selected_routes(master)
                violations = set(
                    violated_resource_slots(selected, capacities)
                ).difference(active_resources)
                if violations:
                    active_resources.update(violations)
                    statistics.conflict_separation_rounds += 1
                    statistics.active_resources_peak = max(
                        statistics.active_resources_peak,
                        len(active_resources),
                    )
                    continue
                if any(
                    graph.crown_config.enable_continuous_conflict_validation
                    for graph in graphs.values()
                ):
                    continuous_conflicts = find_continuous_conflicts(selected, graphs)
                    if continuous_conflicts:
                        mapped = {
                            resource
                            for conflict in continuous_conflicts
                            for resource in conflict.mapped_resources
                        }
                        new_mapped = mapped.difference(active_resources)
                        if not new_mapped:
                            raise CrownResourceMappingError(
                                "continuous conflicts remain after all mapped resource cuts "
                                "are active"
                            )
                        active_resources.update(new_mapped)
                        statistics.conflict_separation_rounds += 1
                        statistics.active_resources_peak = max(
                            statistics.active_resources_peak,
                            len(active_resources),
                        )
                        continue
                candidate = (
                    max((route.finish_time for route in selected), default=0.0)
                    if stage == 1
                    else sum(route.energy for route in selected)
                )
                if candidate < incumbent - _TOL:
                    incumbent = candidate
                    incumbent_routes = selected
                    if stage == 1:
                        _record_anytime(
                            statistics,
                            started,
                            max(service_lower_bound, statistics.root_lp_lower_bound),
                            incumbent,
                            active_resources,
                        )
                break
            included, excluded = _branch_children(
                tuple(graphs),
                task_ids,
                node,
                master,
                active_resources,
                capacities,
                statistics,
            )
            stack.append(excluded)
            stack.append(included)
            break
    if incumbent_routes is None:
        raise ValueError("time-expanded graph master is infeasible")
    return incumbent, incumbent_routes


def solve_crown_graph_bpc(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    *,
    horizon: float,
    resource_capacities: Optional[Mapping[str, int]] = None,
    initial_routes: Sequence[CrownTimedRoute] = (),
) -> CrownBpcSolution:
    """Solve the graph-generated finite model to lexicographic optimality."""

    if not graphs:
        raise ValueError("at least one mode graph is required")
    steps = {graph.crown_config.time_step for graph in graphs.values()}
    if len(steps) != 1:
        raise ValueError("all mode graphs must share one time step")
    time_step = next(iter(steps))
    horizon_slots = int(floor(horizon / time_step + _TOL))
    if horizon_slots <= 0:
        raise ValueError("horizon must contain at least one time slot")
    capacities = dict(resource_capacities or {})
    if any(
        resource_id.startswith("tube:") and capacity != 1
        for resource_id, capacity in capacities.items()
    ):
        raise ValueError("conservative safety-tube resources must have capacity one")
    global_pool = {
        route.timed_route_id: route
        for route in (_empty_timed_route(graph) for graph in graphs.values())
    }
    for route in initial_routes:
        if route.agent_id not in graphs:
            raise ValueError("initial CROWN route references an unknown agent")
        if not set(route.task_ids).issubset(set(task_ids)):
            raise ValueError("initial CROWN route references an unknown task")
        if abs(route.time_step - time_step) > _TOL:
            raise ValueError("initial CROWN route uses a different time step")
        global_pool[route.timed_route_id] = route
    statistics = _GraphStatistics()
    statistics.generated_column_ids.update(global_pool)
    active_resources: Set[ResourceSlot] = set()
    started = perf_counter()
    service_lb = service_workload_lower_bound(graphs, task_ids)

    makespan, _ = _solve_phase(
        graphs,
        task_ids,
        capacities,
        global_pool,
        active_resources,
        stage=1,
        horizon_slots=horizon_slots,
        makespan_limit=None,
        service_lower_bound=service_lb,
        statistics=statistics,
        started=started,
    )
    energy, selected = _solve_phase(
        graphs,
        task_ids,
        capacities,
        global_pool,
        active_resources,
        stage=2,
        horizon_slots=horizon_slots,
        makespan_limit=makespan,
        service_lower_bound=service_lb,
        statistics=statistics,
        started=started,
    )
    final_makespan = max((route.finish_time for route in selected), default=0.0)
    if abs(final_makespan - makespan) > _TOL:
        raise AssertionError("energy stage changed the optimal graph makespan")
    if violated_resource_slots(selected, capacities):
        raise AssertionError("graph BPC returned a resource-infeasible solution")
    _record_anytime(
        statistics,
        started,
        makespan,
        makespan,
        active_resources,
    )
    return CrownBpcSolution(
        timed_routes=tuple(sorted(selected, key=lambda route: route.agent_id)),
        makespan=makespan,
        total_energy=energy,
        lower_bound=makespan,
        upper_bound=makespan,
        optimality_gap=0.0,
        energy_lower_bound=energy,
        energy_upper_bound=energy,
        energy_optimality_gap=0.0,
        active_conflict_resources=tuple(sorted(active_resources)),
        generated_columns=len(statistics.generated_column_ids),
        pricing_iterations=statistics.pricing_iterations,
        branch_nodes=statistics.branch_nodes,
        conflict_separation_rounds=statistics.conflict_separation_rounds,
        time_step=time_step,
        horizon_slots=horizon_slots,
        pricing_labels=statistics.pricing_labels,
        pricing_labels_dominated=statistics.pricing_labels_dominated,
        resource_precedence_branches=statistics.resource_precedence_branches,
        route_variable_branches=statistics.route_variable_branches,
        root_lp_lower_bound=statistics.root_lp_lower_bound,
        service_lower_bound=service_lb,
        solution_status="exact_graph_bpc",
        anytime_trace=tuple(statistics.anytime_trace),
        baseline_makespan=(
            max((route.finish_time for route in initial_routes), default=0.0)
            if initial_routes
            else None
        ),
        baseline_energy=(
            sum(route.energy for route in initial_routes)
            if initial_routes
            else None
        ),
    )


def solve_crown_root_relaxation(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    *,
    horizon: float,
    resource_capacities: Optional[Mapping[str, int]] = None,
    active_resources: Optional[Set[ResourceSlot]] = None,
    deadline: Optional[float] = None,
) -> CrownRootRelaxation:
    """Run exact root column generation for LNS guidance and a valid LB."""

    if not graphs:
        raise ValueError("at least one mode graph is required")
    steps = {graph.crown_config.time_step for graph in graphs.values()}
    if len(steps) != 1:
        raise ValueError("all mode graphs must share one time step")
    time_step = next(iter(steps))
    horizon_slots = int(floor(horizon / time_step + _TOL))
    capacities = dict(resource_capacities or {})
    resources = set(active_resources or set())
    pool = {
        route.timed_route_id: route
        for route in (_empty_timed_route(graph) for graph in graphs.values())
    }
    statistics = _GraphStatistics()
    statistics.generated_column_ids.update(pool)
    master = _column_generation(
        graphs,
        task_ids,
        capacities,
        pool,
        _GraphBranchNode(),
        resources,
        stage=1,
        horizon_slots=horizon_slots,
        makespan_limit=None,
        statistics=statistics,
        deadline=deadline,
    )
    if master is None:
        raise ValueError("root graph relaxation is infeasible")
    return CrownRootRelaxation(
        objective=master.objective,
        route_pool=tuple(sorted(pool.values(), key=lambda route: route.timed_route_id)),
        task_duals={
            task_id: master.equality_duals[("task", task_id)] for task_id in task_ids
        },
        agent_duals={
            agent_id: master.equality_duals[("agent", agent_id)] for agent_id in graphs
        },
        makespan_duals={
            agent_id: master.inequality_duals[("makespan", agent_id)]
            for agent_id in graphs
        },
        resource_duals={
            resource: master.inequality_duals[("resource",) + resource]
            for resource in resources
        },
        service_lower_bound=service_workload_lower_bound(graphs, task_ids),
        pricing_iterations=statistics.pricing_iterations,
        pricing_labels=statistics.pricing_labels,
        exact=True,
    )


def solve_route_pool_dual_guidance(
    agent_ids: Sequence[str],
    task_ids: Sequence[str],
    columns: Sequence[CrownTimedRoute],
    *,
    resource_capacities: Optional[Mapping[str, int]] = None,
) -> CrownPoolDualGuidance:
    """Solve the current pool LP and expose congestion/assignment prices.

    This restricted-master value is deliberately *not* reported as a global
    lower bound.  Its duals only bias CROWN-LNS neighborhood selection and
    route repricing; certified bounds still come from exact root pricing or the
    service relaxation.
    """

    capacities = dict(resource_capacities or {})
    per_resource_agents: Dict[ResourceSlot, Set[str]] = {}
    for route in columns:
        for resource in route.occupied_resource_slots:
            per_resource_agents.setdefault(resource, set()).add(route.agent_id)
    resources = {
        resource
        for resource, agents in per_resource_agents.items()
        if len(agents) > capacities.get(resource[0], 1)
    }
    master = _solve_graph_master(
        tuple(agent_ids),
        tuple(task_ids),
        capacities,
        tuple(columns),
        resources,
        _GraphBranchNode(),
        stage=1,
        feasibility=False,
        horizon_slots=max((route.finish_slot for route in columns), default=1),
    )
    return CrownPoolDualGuidance(
        task_duals={
            task_id: master.equality_duals[("task", task_id)]
            for task_id in task_ids
        },
        agent_duals={
            agent_id: master.equality_duals[("agent", agent_id)]
            for agent_id in agent_ids
        },
        makespan_duals={
            agent_id: master.inequality_duals[("makespan", agent_id)]
            for agent_id in agent_ids
        },
        resource_duals={
            resource: master.inequality_duals[("resource",) + resource]
            for resource in resources
        },
        objective=master.objective,
    )
