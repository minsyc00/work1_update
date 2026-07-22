"""Exact route-column enumeration for small CROWN instances."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations, permutations, product
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .types import CrownConnection, CrownMode, CrownOperation, CrownRoute


ConnectionKey = Tuple[Optional[str], Optional[str]]


def _copy_operations(
    operations: Iterable[CrownOperation],
    prefix: str,
) -> Tuple[CrownOperation, ...]:
    return tuple(
        replace(operation, operation_id=f"{prefix}:{index}:{operation.operation_id}")
        for index, operation in enumerate(operations)
    )


def enumerate_agent_routes(
    agent_id: str,
    task_ids: Sequence[str],
    modes: Sequence[CrownMode],
    connections: Sequence[CrownConnection] = (),
    *,
    max_tasks_per_route: Optional[int] = None,
    require_connections: bool = False,
    include_empty_route: bool = True,
) -> Tuple[CrownRoute, ...]:
    """Enumerate every subset/order/mode route for one agent.

    This routine is intentionally exponential and is the exact route oracle
    for unit tests and small instances.  Missing connections represent a
    zero-duration connector unless ``require_connections`` is true.  Depot is
    denoted by ``None`` in ``CrownConnection``.
    """

    ordered_tasks = tuple(dict.fromkeys(task_ids))
    if len(ordered_tasks) != len(tuple(task_ids)):
        raise ValueError("task_ids must be unique")
    maximum = len(ordered_tasks) if max_tasks_per_route is None else max_tasks_per_route
    if maximum < 0:
        raise ValueError("max_tasks_per_route must be non-negative")

    agent_modes = tuple(mode for mode in modes if mode.agent_id == agent_id)
    agent_mode_ids = [mode.mode_id for mode in agent_modes]
    if len(set(agent_mode_ids)) != len(agent_mode_ids):
        raise ValueError("mode_id values must be unique per agent")
    modes_by_task: Dict[str, Tuple[CrownMode, ...]] = {}
    for task_id in ordered_tasks:
        candidates = tuple(
            sorted(
                (
                    mode
                    for mode in agent_modes
                    if mode.task_id == task_id
                ),
                key=lambda mode: mode.mode_id,
            )
        )
        modes_by_task[task_id] = candidates

    agent_connections = tuple(
        connection for connection in connections if connection.agent_id == agent_id
    )
    connection_keys = [
        (connection.from_mode_id, connection.to_mode_id)
        for connection in agent_connections
    ]
    if len(set(connection_keys)) != len(connection_keys):
        raise ValueError("connection endpoint pairs must be unique per agent")
    connection_map: Mapping[ConnectionKey, CrownConnection] = {
        key: connection for key, connection in zip(connection_keys, agent_connections)
    }
    routes = []
    if include_empty_route:
        routes.append(
            CrownRoute(
                route_id=f"{agent_id}:empty",
                agent_id=agent_id,
                task_ids=(),
                operations=(),
                mode_ids=(),
            )
        )

    for task_count in range(1, min(maximum, len(ordered_tasks)) + 1):
        for task_subset in combinations(ordered_tasks, task_count):
            for task_order in permutations(task_subset):
                choices = tuple(modes_by_task[task_id] for task_id in task_order)
                if any(not task_modes for task_modes in choices):
                    continue
                for selected_modes in product(*choices):
                    operations = []
                    feasible = True
                    mode_ids = tuple(mode.mode_id for mode in selected_modes)
                    transition_pairs = tuple(
                        zip((None,) + mode_ids, mode_ids + (None,))
                    )

                    for position, mode in enumerate(selected_modes):
                        incoming_key = transition_pairs[position]
                        incoming = connection_map.get(incoming_key)
                        if incoming is None and require_connections:
                            feasible = False
                            break
                        if incoming is not None:
                            operations.extend(
                                _copy_operations(
                                    incoming.operations,
                                    f"connection-{position}",
                                )
                            )
                        operations.extend(
                            _copy_operations(mode.operations, f"mode-{position}-{mode.mode_id}")
                        )

                    if not feasible:
                        continue
                    outgoing_key = transition_pairs[-1]
                    outgoing = connection_map.get(outgoing_key)
                    if outgoing is None and require_connections:
                        continue
                    if outgoing is not None:
                        operations.extend(
                            _copy_operations(outgoing.operations, "connection-return")
                        )

                    order_token = ",".join(task_order)
                    mode_token = ",".join(mode_ids)
                    routes.append(
                        CrownRoute(
                            route_id=f"{agent_id}:tasks[{order_token}]:modes[{mode_token}]",
                            agent_id=agent_id,
                            task_ids=tuple(task_order),
                            operations=tuple(operations),
                            mode_ids=mode_ids,
                        )
                    )

    return tuple(sorted(routes, key=lambda route: route.route_id))


def enumerate_route_universe(
    agent_ids: Sequence[str],
    task_ids: Sequence[str],
    modes: Sequence[CrownMode],
    connections: Sequence[CrownConnection] = (),
    *,
    max_tasks_per_route: Optional[int] = None,
    require_connections: bool = False,
) -> Mapping[str, Tuple[CrownRoute, ...]]:
    """Enumerate deterministic complete route pools for every agent."""

    return {
        agent_id: enumerate_agent_routes(
            agent_id,
            task_ids,
            modes,
            connections,
            max_tasks_per_route=max_tasks_per_route,
            require_connections=require_connections,
            include_empty_route=True,
        )
        for agent_id in agent_ids
    }
