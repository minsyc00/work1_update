"""Exact elementary resource-constrained pricing on the CROWN mode graph."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil, inf
from time import perf_counter
from typing import Dict, FrozenSet, Iterator, Mapping, Optional, Sequence, Tuple

from .mode_graph import CrownGeometricMode, CrownTimeExpandedModeGraph
from .motion import (
    CrownMotionPrimitive,
    CurrentInfeasibleError,
    build_wait_primitive,
    retime_motion_primitive,
)
from .types import CrownRoute, CrownTimedRoute, ResourceSlot


_TOL = 1.0e-9


class PricingLabelLimitExceeded(RuntimeError):
    pass


class PricingTimeLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class CrownPricingPrecedenceDual:
    """Dual of one first-entry resource-precedence branch row."""

    resource_id: str
    role: str
    horizon_slots: int
    dual: float

    def __post_init__(self) -> None:
        if self.role not in {"before", "after"}:
            raise ValueError("precedence pricing role must be before or after")
        if self.horizon_slots <= 0:
            raise ValueError("precedence pricing horizon must be positive")


@dataclass(frozen=True)
class CrownPricingDuals:
    agent_dual: float
    task_duals: Mapping[str, float]
    makespan_dual: float = 0.0
    resource_duals: Mapping[ResourceSlot, float] = field(default_factory=dict)
    precedence_duals: Tuple[CrownPricingPrecedenceDual, ...] = ()
    stage: int = 1

    def __post_init__(self) -> None:
        if self.stage not in {1, 2}:
            raise ValueError("pricing stage must be one or two")


@dataclass(frozen=True)
class CrownPricingRestrictions:
    required_tasks: FrozenSet[str] = frozenset()
    forbidden_tasks: FrozenSet[str] = frozenset()
    forbidden_mode_ids: FrozenSet[str] = frozenset()
    allowed_mode_ids: Optional[FrozenSet[str]] = None
    required_successors: Mapping[str, str] = field(default_factory=dict)
    forbidden_arcs: FrozenSet[Tuple[str, str]] = frozenset()
    required_before: FrozenSet[Tuple[str, str]] = frozenset()
    forbidden_resource_slots: FrozenSet[ResourceSlot] = frozenset()
    forbidden_route_ids: FrozenSet[str] = frozenset()

    def __post_init__(self) -> None:
        if self.required_tasks.intersection(self.forbidden_tasks):
            raise ValueError("a task cannot be both required and forbidden")


@dataclass(frozen=True)
class CrownPricingResult:
    route: Optional[CrownTimedRoute]
    reduced_cost: float
    labels_created: int
    labels_dominated: int
    complete_routes_evaluated: int
    exact: bool


@dataclass(frozen=True)
class _Label:
    last_mode_id: Optional[str]
    time_slot: int
    energy: float
    visited: FrozenSet[str]
    task_order: Tuple[str, ...]
    mode_ids: Tuple[str, ...]
    primitives: Tuple[CrownMotionPrimitive, ...]
    start_slots: Tuple[int, ...]
    duration_slots: Tuple[int, ...]
    occupied: FrozenSet[ResourceSlot]
    partial_reduced_cost: float


@dataclass
class _PricingCounters:
    labels_created: int = 0
    labels_dominated: int = 0
    complete_routes_evaluated: int = 0


def _duration_slots(duration: float, time_step: float) -> int:
    return max(1, int(ceil(duration / time_step - 1.0e-12)))


def _schedule_primitive_sequence(
    graph: CrownTimeExpandedModeGraph,
    primitives: Sequence[CrownMotionPrimitive],
    *,
    earliest_slot: int,
    horizon_slots: int,
    initial_energy: float,
    initial_occupied: FrozenSet[ResourceSlot],
    restrictions: CrownPricingRestrictions,
    enumerate_all_waits: bool = True,
    deadline: Optional[float] = None,
) -> Iterator[
    Tuple[
        Tuple[CrownMotionPrimitive, ...],
        Tuple[int, ...],
        Tuple[int, ...],
        int,
        float,
        FrozenSet[ResourceSlot],
    ]
]:
    """Enumerate waits at a graph node, then execute one graph edge atomically.

    A connection or service mode is a finite motion edge.  Waiting is allowed at
    its tail node; its internal primitives execute consecutively.  This matches
    the paper's time-expanded graph semantics and avoids introducing artificial
    mid-primitive stopping states.
    """

    if not primitives:
        yield (), (), (), earliest_slot, initial_energy, initial_occupied
        return
    separation_slots = max(
        0,
        int(
            ceil(
                graph.path_config.resource_separation_time
                / graph.crown_config.time_step
                - 1.0e-12
            )
        ),
    )

    for edge_start_slot in range(earliest_slot, horizon_slots + 1):
        if deadline is not None and perf_counter() >= deadline:
            return
        current_slot = edge_start_slot
        energy = initial_energy
        occupied = initial_occupied
        retimed = []
        starts = []
        durations = []
        feasible = True
        wait_slots = edge_start_slot - earliest_slot
        if wait_slots > 0:
            wait = build_wait_primitive(
                primitive_id=(
                    f"wait:{graph.agent_id}:{earliest_slot}:{edge_start_slot}:"
                    f"{primitives[0].primitive_id}"
                ),
                agent_id=graph.agent_id,
                pose=primitives[0].start_pose,
                duration=wait_slots * graph.crown_config.time_step,
                profile=graph.profile,
                crown_config=graph.crown_config,
                planning_distance=graph.planning_distance,
            )
            wait_resources = frozenset(
                (resource_id, slot)
                for resource_id in wait.resource_ids
                for slot in range(earliest_slot, edge_start_slot + separation_slots)
            )
            if wait_resources.intersection(restrictions.forbidden_resource_slots):
                continue
            energy += wait.energy
            if (
                graph.profile.battery_capacity is not None
                and energy > graph.profile.battery_capacity + _TOL
            ):
                continue
            occupied = occupied.union(wait_resources)
            retimed.append(wait)
            starts.append(earliest_slot)
            durations.append(wait_slots)
        for primitive in primitives:
            try:
                timed_primitive = retime_motion_primitive(
                    primitive,
                    profile=graph.profile,
                    current_field=graph.current_field,
                    absolute_start_time=current_slot * graph.crown_config.time_step,
                )
            except CurrentInfeasibleError:
                feasible = False
                break
            duration = _duration_slots(
                timed_primitive.duration,
                graph.crown_config.time_step,
            )
            rounding_wait = max(
                0.0,
                duration * graph.crown_config.time_step - timed_primitive.duration,
            )
            if rounding_wait > _TOL:
                timed_primitive = replace(
                    timed_primitive,
                    energy=(
                        timed_primitive.energy
                        + graph.profile.wait_power * rounding_wait
                    ),
                    metadata={
                        **dict(timed_primitive.metadata or {}),
                        "rounding_wait_duration": f"{rounding_wait:.12g}",
                    },
                )
            finish_slot = current_slot + duration
            if finish_slot > horizon_slots:
                feasible = False
                break
            new_resources = frozenset(
                (resource_id, slot)
                for resource_id in timed_primitive.resource_ids
                for slot in range(current_slot, finish_slot + separation_slots)
            )
            if new_resources.intersection(restrictions.forbidden_resource_slots):
                feasible = False
                break
            new_energy = energy + timed_primitive.energy
            if (
                graph.profile.battery_capacity is not None
                and new_energy > graph.profile.battery_capacity + _TOL
            ):
                feasible = False
                break
            energy = new_energy
            occupied = occupied.union(new_resources)
            retimed.append(timed_primitive)
            starts.append(current_slot)
            durations.append(duration)
            current_slot = finish_slot
        if feasible:
            yield (
                tuple(retimed),
                tuple(starts),
                tuple(durations),
                current_slot,
                energy,
                occupied,
            )
            if not enumerate_all_waits:
                return


def _resource_reduced_cost(
    before: FrozenSet[ResourceSlot],
    after: FrozenSet[ResourceSlot],
    duals: CrownPricingDuals,
) -> float:
    return -sum(
        duals.resource_duals.get(resource, 0.0)
        for resource in after.difference(before)
    )


def _partial_cost_increment(
    old_energy: float,
    new_energy: float,
    old_occupied: FrozenSet[ResourceSlot],
    new_occupied: FrozenSet[ResourceSlot],
    duals: CrownPricingDuals,
    task_id: Optional[str],
) -> float:
    increment = new_energy - old_energy if duals.stage == 2 else 0.0
    increment += _resource_reduced_cost(old_occupied, new_occupied, duals)
    for precedence in duals.precedence_duals:
        old_slots = tuple(
            slot for resource_id, slot in old_occupied
            if resource_id == precedence.resource_id
        )
        if old_slots:
            continue
        new_slots = tuple(
            slot for resource_id, slot in new_occupied
            if resource_id == precedence.resource_id
        )
        if not new_slots:
            continue
        first_entry = min(new_slots)
        coefficient = precedence.horizon_slots + (
            first_entry if precedence.role == "before" else -first_entry
        )
        increment -= precedence.dual * coefficient
    if task_id is not None:
        increment -= duals.task_duals.get(task_id, 0.0)
    return increment


def _arc_allowed(
    label: _Label,
    task_id: str,
    restrictions: CrownPricingRestrictions,
) -> bool:
    if task_id in restrictions.forbidden_tasks:
        return False
    if any(second == task_id and first not in label.visited for first, second in restrictions.required_before):
        return False
    if label.task_order:
        previous = label.task_order[-1]
        if (previous, task_id) in restrictions.forbidden_arcs:
            return False
        required = restrictions.required_successors.get(previous)
        if required is not None and required != task_id:
            return False
    return True


def _can_finish(label: _Label, restrictions: CrownPricingRestrictions) -> bool:
    if not restrictions.required_tasks.issubset(label.visited):
        return False
    if label.task_order and label.task_order[-1] in restrictions.required_successors:
        return False
    return all(
        first in label.visited and second in label.visited
        for first, second in restrictions.required_before
        if first in label.visited or second in label.visited
    )


def _dominance_key(
    label: _Label,
    restrictions: CrownPricingRestrictions,
    duals: CrownPricingDuals,
) -> Tuple[object, ...]:
    key: Tuple[object, ...] = (label.last_mode_id, label.time_slot, label.visited)
    if duals.precedence_duals:
        resource_ids = sorted(
            {precedence.resource_id for precedence in duals.precedence_duals}
        )
        key += (
            tuple(
                (
                    resource_id,
                    min(
                        (
                            slot
                            for candidate_id, slot in label.occupied
                            if candidate_id == resource_id
                        ),
                        default=-1,
                    ),
                )
                for resource_id in resource_ids
            ),
        )
    if restrictions.forbidden_route_ids:
        key += (label.task_order, label.mode_ids, label.start_slots)
    return key


def _is_dominated(
    label: _Label,
    fronts: Dict[Tuple[object, ...], list[Tuple[float, float]]],
    restrictions: CrownPricingRestrictions,
    duals: CrownPricingDuals,
) -> bool:
    key = _dominance_key(label, restrictions, duals)
    front = fronts.setdefault(key, [])
    if any(
        energy <= label.energy + _TOL and cost <= label.partial_reduced_cost + _TOL
        for energy, cost in front
    ):
        return True
    front[:] = [
        (energy, cost)
        for energy, cost in front
        if not (
            label.energy <= energy + _TOL
            and label.partial_reduced_cost <= cost + _TOL
        )
    ]
    front.append((label.energy, label.partial_reduced_cost))
    return False


def _route_from_label(
    graph: CrownTimeExpandedModeGraph,
    label: _Label,
) -> CrownTimedRoute:
    operations = tuple(
        replace(
            primitive.to_operation(),
            operation_id=f"route-operation:{index}:{primitive.primitive_id}",
        )
        for index, primitive in enumerate(label.primitives)
    )
    task_token = ",".join(label.task_order)
    mode_token = ",".join(label.mode_ids)
    start_token = ",".join(str(slot) for slot in label.start_slots)
    route_id = (
        f"graph:{graph.agent_id}:tasks[{task_token}]:"
        f"modes[{mode_token}]:starts[{start_token}]"
    )
    base_route = CrownRoute(
        route_id=route_id,
        agent_id=graph.agent_id,
        task_ids=label.task_order,
        operations=operations,
        mode_ids=label.mode_ids,
        metadata={
            "path_segments": tuple(primitive.segment for primitive in label.primitives),
            "source": "time-expanded-mode-graph-pricing",
        },
    )
    return CrownTimedRoute(
        timed_route_id=route_id,
        base_route=base_route,
        start_slots=label.start_slots,
        duration_slots=label.duration_slots,
        finish_slot=label.time_slot,
        time_step=graph.crown_config.time_step,
        occupied_resource_slots=tuple(sorted(label.occupied)),
        energy=label.energy,
    )


def build_fixed_mode_timed_route(
    graph: CrownTimeExpandedModeGraph,
    mode_ids: Sequence[str],
    *,
    horizon_slots: int,
    earliest_slot: int = 0,
) -> Optional[CrownTimedRoute]:
    """Materialize one already-connected mode path without label search.

    This is an incumbent constructor, not a pricing oracle.  Every connection
    and service primitive still comes from the certified time-expanded graph,
    is retimed at its actual departure slot, and carries the same conservative
    resource occupancy as a priced route.
    """

    restrictions = CrownPricingRestrictions()
    current_slot = earliest_slot
    energy = 0.0
    occupied: FrozenSet[ResourceSlot] = frozenset()
    primitives = []
    starts = []
    durations = []
    tasks = []
    previous: Optional[str] = None

    def append_edge(edge_primitives: Sequence[CrownMotionPrimitive]) -> bool:
        nonlocal current_slot, energy, occupied
        result = next(
            _schedule_primitive_sequence(
                graph,
                edge_primitives,
                earliest_slot=current_slot,
                horizon_slots=horizon_slots,
                initial_energy=energy,
                initial_occupied=occupied,
                restrictions=restrictions,
                enumerate_all_waits=False,
            ),
            None,
        )
        if result is None:
            return False
        timed, edge_starts, edge_durations, current_slot, energy, occupied = result
        primitives.extend(timed)
        starts.extend(edge_starts)
        durations.extend(edge_durations)
        return True

    for mode_id in mode_ids:
        if mode_id not in graph.mode_lookup:
            return None
        connection = graph.connection(previous, mode_id, current_slot)
        if not connection.feasible or not append_edge(connection.primitives):
            return None
        service = graph.service_primitives(mode_id, current_slot)
        if service is None or not append_edge(service):
            return None
        mode = graph.mode_lookup[mode_id]
        tasks.append(mode.task_id)
        previous = mode_id
    if graph.goal_pose_explicit or graph.crown_config.return_to_start:
        connection = graph.connection(previous, None, current_slot)
        if not connection.feasible or not append_edge(connection.primitives):
            return None
    finish_time = current_slot * graph.crown_config.time_step
    if (
        graph.profile.max_mission_time is not None
        and finish_time > graph.profile.max_mission_time + _TOL
    ):
        return None
    operations = tuple(
        replace(
            primitive.to_operation(),
            operation_id=f"fixed-route-operation:{index}:{primitive.primitive_id}",
        )
        for index, primitive in enumerate(primitives)
    )
    route_id = (
        f"fixed:{graph.agent_id}:tasks[{','.join(tasks)}]:"
        f"modes[{','.join(mode_ids)}]:start[{earliest_slot}]"
    )
    base_route = CrownRoute(
        route_id=route_id,
        agent_id=graph.agent_id,
        task_ids=tuple(tasks),
        operations=operations,
        mode_ids=tuple(mode_ids),
        metadata={
            "path_segments": tuple(primitive.segment for primitive in primitives),
            "source": "fixed-connected-mode-path-incumbent",
        },
    )
    return CrownTimedRoute(
        timed_route_id=route_id,
        base_route=base_route,
        start_slots=tuple(starts),
        duration_slots=tuple(durations),
        finish_slot=current_slot,
        time_step=graph.crown_config.time_step,
        occupied_resource_slots=tuple(sorted(occupied)),
        energy=energy,
    )


def price_mode_graph_exact(
    graph: CrownTimeExpandedModeGraph,
    *,
    horizon_slots: int,
    duals: CrownPricingDuals,
    restrictions: Optional[CrownPricingRestrictions] = None,
    label_limit: Optional[int] = None,
    exact: bool = True,
    deadline: Optional[float] = None,
) -> CrownPricingResult:
    """Find the minimum-reduced-cost elementary route by exact labels.

    Safe dominance is applied only at the same time-expanded mode node and with
    the same visited task set.  If ``exact`` is true, hitting ``label_limit`` is
    an explicit failure and no lower-bound claim may be made.
    """

    rules = restrictions or CrownPricingRestrictions()
    limit = label_limit or graph.crown_config.pricing_label_limit
    counters = _PricingCounters()
    initial = _Label(
        last_mode_id=None,
        time_slot=0,
        energy=0.0,
        visited=frozenset(),
        task_order=(),
        mode_ids=(),
        primitives=(),
        start_slots=(),
        duration_slots=(),
        occupied=frozenset(),
        partial_reduced_cost=0.0,
    )
    stack = [initial]
    fronts: Dict[Tuple[object, ...], list[Tuple[float, float]]] = {}
    best_route: Optional[CrownTimedRoute] = None
    best_reduced_cost = inf

    while stack:
        if deadline is not None and perf_counter() >= deadline:
            if exact:
                raise PricingTimeLimitExceeded(
                    f"exact pricing hit its deadline for agent {graph.agent_id}"
                )
            break
        label = stack.pop()
        counters.labels_created += 1
        if counters.labels_created > limit:
            if exact:
                raise PricingLabelLimitExceeded(
                    f"exact pricing exceeded {limit} labels for agent {graph.agent_id}"
                )
            break
        if _is_dominated(label, fronts, rules, duals):
            counters.labels_dominated += 1
            continue

        if _can_finish(label, rules):
            connection = graph.connection(label.last_mode_id, None, label.time_slot)
            if connection.feasible:
                for (
                    retimed,
                    starts,
                    durations,
                    finish_slot,
                    energy,
                    occupied,
                ) in _schedule_primitive_sequence(
                    graph,
                    connection.primitives,
                    earliest_slot=label.time_slot,
                    horizon_slots=horizon_slots,
                    initial_energy=label.energy,
                    initial_occupied=label.occupied,
                    restrictions=rules,
                    enumerate_all_waits=exact,
                    deadline=deadline,
                ):
                    if (
                        graph.profile.max_mission_time is not None
                        and finish_slot * graph.crown_config.time_step
                        > graph.profile.max_mission_time + _TOL
                    ):
                        continue
                    complete = replace(
                        label,
                        time_slot=finish_slot,
                        energy=energy,
                        primitives=label.primitives + retimed,
                        start_slots=label.start_slots + starts,
                        duration_slots=label.duration_slots + durations,
                        occupied=occupied,
                        partial_reduced_cost=(
                            label.partial_reduced_cost
                            + _partial_cost_increment(
                                label.energy,
                                energy,
                                label.occupied,
                                occupied,
                                duals,
                                None,
                            )
                        ),
                    )
                    route = _route_from_label(graph, complete)
                    if route.timed_route_id in rules.forbidden_route_ids:
                        continue
                    counters.complete_routes_evaluated += 1
                    reduced_cost = (
                        complete.partial_reduced_cost
                        - duals.agent_dual
                        - duals.makespan_dual * route.finish_time
                    )
                    if (
                        reduced_cost < best_reduced_cost - _TOL
                        or (
                            abs(reduced_cost - best_reduced_cost) <= _TOL
                            and (
                                best_route is None
                                or (route.finish_time, route.energy, route.timed_route_id)
                                < (
                                    best_route.finish_time,
                                    best_route.energy,
                                    best_route.timed_route_id,
                                )
                            )
                        )
                    ):
                        best_reduced_cost = reduced_cost
                        best_route = route

        if (
            graph.crown_config.max_tasks_per_route is not None
            and len(label.visited) >= graph.crown_config.max_tasks_per_route
        ):
            continue
        for task_id in reversed(graph.task_ids):
            if task_id in label.visited or not _arc_allowed(label, task_id, rules):
                continue
            for mode in reversed(graph.modes_for_task(task_id)):
                if mode.mode_id in rules.forbidden_mode_ids:
                    continue
                if (
                    rules.allowed_mode_ids is not None
                    and mode.mode_id not in rules.allowed_mode_ids
                ):
                    continue
                connection = graph.connection(
                    label.last_mode_id,
                    mode.mode_id,
                    label.time_slot,
                )
                if not connection.feasible:
                    continue
                for (
                    connection_primitives,
                    connection_starts,
                    connection_durations,
                    arrival_slot,
                    arrival_energy,
                    arrival_occupied,
                ) in _schedule_primitive_sequence(
                    graph,
                    connection.primitives,
                    earliest_slot=label.time_slot,
                    horizon_slots=horizon_slots,
                    initial_energy=label.energy,
                    initial_occupied=label.occupied,
                    restrictions=rules,
                    enumerate_all_waits=exact,
                    deadline=deadline,
                ):
                    service_at_arrival = graph.service_primitives(
                        mode.mode_id,
                        arrival_slot,
                    )
                    if service_at_arrival is None:
                        continue
                    for (
                        service_primitives,
                        service_starts,
                        service_durations,
                        finish_slot,
                        energy,
                        occupied,
                    ) in _schedule_primitive_sequence(
                        graph,
                        service_at_arrival,
                        earliest_slot=arrival_slot,
                        horizon_slots=horizon_slots,
                        initial_energy=arrival_energy,
                        initial_occupied=arrival_occupied,
                        restrictions=rules,
                        enumerate_all_waits=exact,
                        deadline=deadline,
                    ):
                        if (
                            graph.profile.max_mission_time is not None
                            and finish_slot * graph.crown_config.time_step
                            > graph.profile.max_mission_time + _TOL
                        ):
                            continue
                        new_label = _Label(
                            last_mode_id=mode.mode_id,
                            time_slot=finish_slot,
                            energy=energy,
                            visited=label.visited.union({task_id}),
                            task_order=label.task_order + (task_id,),
                            mode_ids=label.mode_ids + (mode.mode_id,),
                            primitives=(
                                label.primitives
                                + connection_primitives
                                + service_primitives
                            ),
                            start_slots=(
                                label.start_slots
                                + connection_starts
                                + service_starts
                            ),
                            duration_slots=(
                                label.duration_slots
                                + connection_durations
                                + service_durations
                            ),
                            occupied=occupied,
                            partial_reduced_cost=(
                                label.partial_reduced_cost
                                + _partial_cost_increment(
                                    label.energy,
                                    energy,
                                    label.occupied,
                                    occupied,
                                    duals,
                                    task_id,
                                )
                            ),
                        )
                        stack.append(new_label)

    return CrownPricingResult(
        route=best_route,
        reduced_cost=best_reduced_cost,
        labels_created=counters.labels_created,
        labels_dominated=counters.labels_dominated,
        complete_routes_evaluated=counters.complete_routes_evaluated,
        exact=exact and counters.labels_created <= limit,
    )
