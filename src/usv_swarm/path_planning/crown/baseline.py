"""Conflict-unaware assignment followed by prioritized route deconfliction."""

from __future__ import annotations

from math import floor
from time import perf_counter
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .mode_graph import CrownTimeExpandedModeGraph
from .pricing import (
    CrownPricingDuals,
    CrownPricingRestrictions,
    price_mode_graph_exact,
)
from .types import CrownTimedRoute, ResourceSlot


def _greedy_service_assignment(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
) -> Mapping[str, Tuple[str, ...]]:
    assignments: Dict[str, list[str]] = {agent_id: [] for agent_id in graphs}
    loads = {agent_id: 0.0 for agent_id in graphs}
    for task_id in sorted(
        task_ids,
        key=lambda task: (
            sum(bool(graph.modes_for_task(task)) for graph in graphs.values()),
            task,
        ),
    ):
        candidates = []
        for agent_id, graph in graphs.items():
            modes = graph.modes_for_task(task_id)
            if not modes:
                continue
            limit = graph.crown_config.max_tasks_per_route
            if limit is not None and len(assignments[agent_id]) >= limit:
                continue
            service = min(mode.nominal_duration for mode in modes)
            candidates.append((loads[agent_id] + service, service, agent_id))
        if not candidates:
            raise ValueError(f"sequential baseline cannot assign task {task_id!r}")
        _, service, agent_id = min(candidates)
        assignments[agent_id].append(task_id)
        loads[agent_id] += service
    return {agent_id: tuple(tasks) for agent_id, tasks in assignments.items()}


def _full_slots(
    selected: Sequence[CrownTimedRoute],
    capacities: Mapping[str, int],
) -> frozenset[ResourceSlot]:
    counts: Dict[ResourceSlot, int] = {}
    for route in selected:
        for resource in route.occupied_resource_slots:
            counts[resource] = counts.get(resource, 0) + 1
    return frozenset(
        resource
        for resource, count in counts.items()
        if count >= capacities.get(resource[0], 1)
    )


def build_crown_sequential_baseline(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    *,
    horizon: float,
    resource_capacities: Optional[Mapping[str, int]] = None,
    deadline: Optional[float] = None,
) -> Tuple[CrownTimedRoute, ...]:
    """Build a same-model sequential baseline for seeding and comparison.

    Responsibility is fixed using conflict-unaware pure-service loads.  Each
    single-agent route is then minimized independently, while already planned
    agents are treated as immutable time-space reservations.  Failed priority
    orders are retried deterministically.  Unlike CROWN, later conflict costs
    never feed back into the assignment of earlier agents.
    """

    if not graphs:
        raise ValueError("sequential baseline requires at least one graph")
    capacities = dict(resource_capacities or {})
    assignment = _greedy_service_assignment(graphs, task_ids)
    time_steps = {graph.crown_config.time_step for graph in graphs.values()}
    if len(time_steps) != 1:
        raise ValueError("sequential baseline graphs must share one time step")
    time_step = next(iter(time_steps))
    horizon_slots = int(floor(horizon / time_step + 1.0e-9))

    def attempt(
        candidate_assignment: Mapping[str, Tuple[str, ...]],
        *,
        attempt_deadline: Optional[float],
        fixed_order: bool,
        allowed_modes: Optional[Mapping[str, frozenset[str]]] = None,
    ) -> Optional[Tuple[CrownTimedRoute, ...]]:
        base_order = tuple(
            sorted(
                graphs,
                key=lambda agent: (-len(candidate_assignment[agent]), agent),
            )
        )
        orders = [base_order, tuple(reversed(base_order))]
        for offset in range(1, len(base_order)):
            orders.append(base_order[offset:] + base_order[:offset])
        for order in orders:
            if attempt_deadline is not None and perf_counter() >= attempt_deadline:
                break
            selected = []
            failed = False
            for agent_id in order:
                if attempt_deadline is not None and perf_counter() >= attempt_deadline:
                    failed = True
                    break
                own_tasks = candidate_assignment[agent_id]
                result = price_mode_graph_exact(
                    graphs[agent_id],
                    horizon_slots=horizon_slots,
                    duals=CrownPricingDuals(
                        agent_dual=0.0,
                        task_duals={},
                        makespan_dual=-1.0,
                        stage=1,
                    ),
                    restrictions=CrownPricingRestrictions(
                        required_tasks=frozenset(own_tasks),
                        forbidden_tasks=frozenset(
                            set(task_ids).difference(own_tasks)
                        ),
                        required_successors=(
                            dict(zip(own_tasks[:-1], own_tasks[1:]))
                            if fixed_order
                            else {}
                        ),
                        forbidden_resource_slots=_full_slots(selected, capacities),
                        allowed_mode_ids=(
                            None
                            if allowed_modes is None
                            else allowed_modes.get(agent_id)
                        ),
                    ),
                    label_limit=graphs[agent_id].crown_config.pricing_label_limit,
                    exact=False,
                    deadline=attempt_deadline,
                )
                if result.route is None:
                    failed = True
                    break
                selected.append(result.route)
            if not failed:
                by_agent = {route.agent_id: route for route in selected}
                return tuple(by_agent[agent_id] for agent_id in graphs)
        return None

    greedy_deadline = deadline
    if deadline is not None:
        greedy_deadline = perf_counter() + max(
            0.1,
            0.25 * max(0.0, deadline - perf_counter()),
        )
    result = attempt(
        assignment,
        attempt_deadline=greedy_deadline,
        fixed_order=False,
    )
    if result is not None:
        return result

    # Service-only load balancing can assign an obstacle-separated subset to
    # an agent that has no connected mode order.  Repair ownership/order once,
    # still without using any inter-agent conflict prices, then retain the
    # original prioritized post-deconfliction semantics.
    if deadline is None or perf_counter() < deadline:
        from .lns import (
            _bcd_contiguous_connectivity_fleet_paths,
            _connectivity_fleet_paths,
        )

        remaining = None if deadline is None else max(0.0, deadline - perf_counter())
        connectivity_deadline = (
            perf_counter() + 60.0
            if remaining is None
            else perf_counter() + max(0.1, 0.65 * remaining)
        )
        paths = None
        if 8 <= len(task_ids) <= 12:
            paths = _bcd_contiguous_connectivity_fleet_paths(
                graphs,
                task_ids,
                connectivity_deadline,
            )
        if paths is None and perf_counter() < connectivity_deadline:
            paths = _connectivity_fleet_paths(
                graphs,
                task_ids,
                connectivity_deadline,
                beam_width=64,
                direct_only=True,
            )
        if paths is None and perf_counter() < connectivity_deadline:
            paths = _connectivity_fleet_paths(
                graphs,
                task_ids,
                connectivity_deadline,
                beam_width=64,
                direct_only=False,
            )
        if paths is not None:
            connected_assignment = {
                agent_id: paths[agent_id][0] for agent_id in graphs
            }
            allowed_modes = {
                agent_id: frozenset(paths[agent_id][1]) for agent_id in graphs
            }
            result = attempt(
                connected_assignment,
                attempt_deadline=deadline,
                fixed_order=True,
                allowed_modes=allowed_modes,
            )
            if result is not None:
                return result

    raise ValueError("sequential baseline could not be deconflicted within the horizon")
