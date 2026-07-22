"""Finite time-grid expansion and resource occupancy for CROWN-BPC."""

from __future__ import annotations

from math import ceil, isfinite
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from .types import CrownInstance, CrownRoute, CrownTimedRoute, ResourceSlot


def duration_to_slots(duration: float, time_step: float) -> int:
    """Conservatively round a positive duration up to grid slots."""

    if not isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be finite and positive")
    return max(1, int(ceil(duration / time_step - 1.0e-12)))


def _wait_rate_for_agent(
    wait_energy_rate: Union[float, Mapping[str, float]],
    agent_id: str,
) -> float:
    rate = (
        float(wait_energy_rate.get(agent_id, 0.0))
        if isinstance(wait_energy_rate, Mapping)
        else float(wait_energy_rate)
    )
    if not isfinite(rate) or rate < 0.0:
        raise ValueError("wait energy rates must be finite and non-negative")
    return rate


def expand_route_on_time_grid(
    route: CrownRoute,
    *,
    horizon_slots: int,
    time_step: float,
    separation_slots: int = 0,
    wait_energy_rate: float = 0.0,
    max_columns: Optional[int] = None,
) -> Tuple[CrownTimedRoute, ...]:
    """Enumerate every precedence-feasible integer start vector for a route."""

    if horizon_slots < 0:
        raise ValueError("horizon_slots must be non-negative")
    if separation_slots < 0:
        raise ValueError("separation_slots must be non-negative")
    if max_columns is not None and max_columns <= 0:
        raise ValueError("max_columns must be positive when provided")
    if not route.operations:
        return (
            CrownTimedRoute(
                timed_route_id=f"{route.route_id}@empty",
                base_route=route,
                start_slots=(),
                duration_slots=(),
                finish_slot=0,
                time_step=time_step,
                occupied_resource_slots=(),
                energy=route.energy,
            ),
        )

    durations = tuple(
        duration_to_slots(operation.duration, time_step) for operation in route.operations
    )
    results = []

    def append_timed_route(starts: Tuple[int, ...]) -> None:
        if max_columns is not None and len(results) >= max_columns:
            raise ValueError(
                "one route exceeded its remaining timed-column budget; reduce the "
                "horizon or coarsen time_step"
            )
        finish_slot = starts[-1] + durations[-1]
        occupied = set()
        for operation, start, duration in zip(route.operations, starts, durations):
            for resource_id in operation.resource_ids:
                for slot in range(start, start + duration + separation_slots):
                    occupied.add((resource_id, slot))
        service_slots = sum(durations)
        idle_slots = finish_slot - service_slots
        energy = route.energy + wait_energy_rate * idle_slots * time_step
        start_token = ",".join(str(start) for start in starts)
        results.append(
            CrownTimedRoute(
                timed_route_id=f"{route.route_id}@[{start_token}]",
                base_route=route,
                start_slots=starts,
                duration_slots=durations,
                finish_slot=finish_slot,
                time_step=time_step,
                occupied_resource_slots=tuple(sorted(occupied)),
                energy=energy,
            )
        )

    def search(operation_index: int, earliest_start: int, starts: Tuple[int, ...]) -> None:
        duration = durations[operation_index]
        latest_start = horizon_slots - duration
        for start in range(earliest_start, latest_start + 1):
            next_starts = starts + (start,)
            if operation_index + 1 == len(durations):
                append_timed_route(next_starts)
            else:
                search(operation_index + 1, start + duration, next_starts)

    search(0, 0, ())
    return tuple(results)


def build_time_expanded_route_universe(
    instance: CrownInstance,
    *,
    horizon: float,
    time_step: float,
    wait_energy_rate: Union[float, Mapping[str, float]] = 0.0,
    max_timed_columns: int = 200_000,
) -> Tuple[Mapping[str, Tuple[CrownTimedRoute, ...]], int]:
    """Build the complete finite column universe used by exact pricing.

    No column is sampled or truncated.  The explicit limit only fails fast when
    a requested instance is outside the intended minimal-exact regime.
    """

    if not isfinite(horizon) or horizon < 0.0:
        raise ValueError("horizon must be finite and non-negative")
    if not isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be finite and positive")
    if max_timed_columns <= 0:
        raise ValueError("max_timed_columns must be positive")
    horizon_slots = int(ceil(horizon / time_step - 1.0e-12))
    separation_slots = (
        int(ceil(instance.resource_separation / time_step - 1.0e-12))
        if instance.resource_separation > 0.0
        else 0
    )
    universe: Dict[str, Tuple[CrownTimedRoute, ...]] = {}
    count = 0
    for agent_id in instance.agent_ids:
        columns = []
        wait_rate = _wait_rate_for_agent(wait_energy_rate, agent_id)
        for route in instance.routes_by_agent[agent_id]:
            used_count = sum(len(values) for values in universe.values()) + len(columns)
            columns.extend(
                expand_route_on_time_grid(
                    route,
                    horizon_slots=horizon_slots,
                    time_step=time_step,
                    separation_slots=separation_slots,
                    wait_energy_rate=wait_rate,
                    max_columns=max_timed_columns - used_count,
                )
            )
            count = sum(len(values) for values in universe.values()) + len(columns)
            if count > max_timed_columns:
                raise ValueError(
                    "time expansion exceeded max_timed_columns; reduce the horizon, "
                    "coarsen time_step, or use the scalable CROWN-LNS engine"
                )
        if not columns:
            raise ValueError(f"agent {agent_id!r} has no route within the time horizon")
        universe[agent_id] = tuple(
            sorted(columns, key=lambda column: column.timed_route_id)
        )
    return universe, horizon_slots


def violated_resource_slots(
    selected_routes: Sequence[CrownTimedRoute],
    capacities: Mapping[str, int],
) -> Tuple[ResourceSlot, ...]:
    """Return every violated capacity resource in a timed integer solution."""

    counts: Dict[ResourceSlot, int] = {}
    for route in selected_routes:
        for resource_slot in route.occupied_resource_slots:
            counts[resource_slot] = counts.get(resource_slot, 0) + 1
    return tuple(
        sorted(
            resource_slot
            for resource_slot, count in counts.items()
            if count > capacities.get(resource_slot[0], 1)
        )
    )
