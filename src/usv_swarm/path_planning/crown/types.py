"""Finite exact model used by the minimal CROWN-MCPP implementation.

The classes in this module deliberately describe a small, solver-independent
model.  A route column already contains an agent, an ordered task subset, its
coverage/connection operations, and the shared resources used by those
operations.  Consequently, selecting one route per agent jointly decides task
ownership, visit order, coverage mode and resource use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple


OperationKey = Tuple[str, int]
ResourceSlot = Tuple[str, int]


def _frozen_mapping(value: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class CrownOperation:
    """One non-preemptive operation in a route.

    ``resource_ids`` contains unary shared resources.  Operations of different
    agents that name the same resource may not overlap.  Operations in one
    route always follow their tuple order.
    """

    operation_id: str
    duration: float
    resource_ids: Tuple[str, ...] = ()
    energy: float = 0.0
    kind: str = "motion"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id must be non-empty")
        if not isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("operation duration must be finite and positive")
        if not isfinite(self.energy) or self.energy < 0.0:
            raise ValueError("operation energy must be finite and non-negative")
        if len(set(self.resource_ids)) != len(self.resource_ids):
            raise ValueError("resource_ids must not contain duplicates")
        object.__setattr__(self, "resource_ids", tuple(self.resource_ids))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class CrownMode:
    """Agent-specific way of covering exactly one atomic task."""

    task_id: str
    mode_id: str
    agent_id: str
    operations: Tuple[CrownOperation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.task_id or not self.mode_id or not self.agent_id:
            raise ValueError("task_id, mode_id and agent_id must be non-empty")
        if not self.operations:
            raise ValueError("a coverage mode must contain at least one operation")
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class CrownConnection:
    """Connection primitive between two modes (``None`` denotes depot)."""

    agent_id: str
    from_mode_id: Optional[str]
    to_mode_id: Optional[str]
    operations: Tuple[CrownOperation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must be non-empty")
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True)
class CrownRoute:
    """Complete, untimed route column for one agent."""

    route_id: str
    agent_id: str
    task_ids: Tuple[str, ...]
    operations: Tuple[CrownOperation, ...]
    mode_ids: Tuple[str, ...] = ()
    fixed_energy: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.route_id or not self.agent_id:
            raise ValueError("route_id and agent_id must be non-empty")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("a route may cover each task at most once")
        if not isfinite(self.fixed_energy) or self.fixed_energy < 0.0:
            raise ValueError("fixed_energy must be finite and non-negative")
        operation_ids = [operation.operation_id for operation in self.operations]
        if len(set(operation_ids)) != len(operation_ids):
            raise ValueError("operation_id values must be unique inside a route")
        object.__setattr__(self, "task_ids", tuple(self.task_ids))
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "mode_ids", tuple(self.mode_ids))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def nominal_duration(self) -> float:
        return sum(operation.duration for operation in self.operations)

    @property
    def energy(self) -> float:
        return self.fixed_energy + sum(operation.energy for operation in self.operations)


@dataclass(frozen=True)
class CrownInstance:
    """Finite CROWN problem relative to a supplied route universe."""

    agent_ids: Tuple[str, ...]
    task_ids: Tuple[str, ...]
    routes_by_agent: Mapping[str, Tuple[CrownRoute, ...]]
    resource_separation: float = 0.0
    resource_capacities: Mapping[str, int] = field(default_factory=dict)
    wait_energy_rates: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.agent_ids:
            raise ValueError("an instance must contain at least one agent")
        if len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("agent_ids must be unique")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique")
        if not isfinite(self.resource_separation) or self.resource_separation < 0.0:
            raise ValueError("resource_separation must be finite and non-negative")

        known_tasks = set(self.task_ids)
        normalized = {}
        route_ids = set()
        for agent_id in self.agent_ids:
            routes = tuple(self.routes_by_agent.get(agent_id, ()))
            if not routes:
                raise ValueError(f"agent {agent_id!r} has no route candidates")
            for route in routes:
                if route.agent_id != agent_id:
                    raise ValueError("route stored under the wrong agent")
                if not set(route.task_ids).issubset(known_tasks):
                    raise ValueError("route references a task outside the instance")
                if route.route_id in route_ids:
                    raise ValueError("route_id values must be globally unique")
                route_ids.add(route.route_id)
            normalized[agent_id] = routes

        capacities = dict(self.resource_capacities)
        if any((not isinstance(value, int)) or value <= 0 for value in capacities.values()):
            raise ValueError("resource capacities must be positive integers")
        wait_rates = dict(self.wait_energy_rates)
        if not set(wait_rates).issubset(set(self.agent_ids)):
            raise ValueError("wait_energy_rates references an unknown agent")
        if any(not isfinite(value) or value < 0.0 for value in wait_rates.values()):
            raise ValueError("wait energy rates must be finite and non-negative")
        object.__setattr__(self, "agent_ids", tuple(self.agent_ids))
        object.__setattr__(self, "task_ids", tuple(self.task_ids))
        object.__setattr__(self, "routes_by_agent", MappingProxyType(normalized))
        object.__setattr__(self, "resource_capacities", MappingProxyType(capacities))
        object.__setattr__(self, "wait_energy_rates", MappingProxyType(wait_rates))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def capacity(self, resource_id: str) -> int:
        return self.resource_capacities.get(resource_id, 1)

    def wait_energy_rate(self, agent_id: str) -> float:
        return self.wait_energy_rates.get(agent_id, 0.0)


@dataclass(frozen=True)
class CrownSchedule:
    """Exact start times for operations of a selected route set."""

    starts: Mapping[OperationKey, float]
    finishes: Mapping[OperationKey, float]
    agent_completion_times: Mapping[str, float]
    makespan: float
    waiting_energy: float = 0.0
    conflict_pairs: int = 0
    orientations_evaluated: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "starts", MappingProxyType(dict(self.starts)))
        object.__setattr__(self, "finishes", MappingProxyType(dict(self.finishes)))
        object.__setattr__(
            self,
            "agent_completion_times",
            MappingProxyType(dict(self.agent_completion_times)),
        )


@dataclass(frozen=True)
class CrownSolution:
    """Lexicographic solution of the continuous-time finite-route model."""

    routes: Tuple[CrownRoute, ...]
    schedule: CrownSchedule
    makespan: float
    total_energy: float
    method: str
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    optimality_gap: Optional[float] = None
    statistics: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes", tuple(self.routes))
        object.__setattr__(self, "statistics", _frozen_mapping(self.statistics))

    @property
    def objective(self) -> Tuple[float, float]:
        return (self.makespan, self.total_energy)


@dataclass(frozen=True)
class CrownTimedRoute:
    """A route column with every operation placed on the finite time grid."""

    timed_route_id: str
    base_route: CrownRoute
    start_slots: Tuple[int, ...]
    duration_slots: Tuple[int, ...]
    finish_slot: int
    time_step: float
    occupied_resource_slots: Tuple[ResourceSlot, ...]
    energy: float

    @property
    def agent_id(self) -> str:
        return self.base_route.agent_id

    @property
    def task_ids(self) -> Tuple[str, ...]:
        return self.base_route.task_ids

    @property
    def finish_time(self) -> float:
        return self.finish_slot * self.time_step


@dataclass(frozen=True)
class CrownBpcSolution:
    """Certified solution of the finite time-expanded master problem."""

    timed_routes: Tuple[CrownTimedRoute, ...]
    makespan: float
    total_energy: float
    lower_bound: float
    upper_bound: float
    optimality_gap: float
    energy_lower_bound: float
    energy_upper_bound: float
    energy_optimality_gap: float
    active_conflict_resources: Tuple[ResourceSlot, ...]
    generated_columns: int
    pricing_iterations: int
    branch_nodes: int
    conflict_separation_rounds: int
    time_step: float
    horizon_slots: int
    pricing_labels: int = 0
    pricing_labels_dominated: int = 0
    resource_precedence_branches: int = 0
    route_variable_branches: int = 0
    root_lp_lower_bound: Optional[float] = None
    service_lower_bound: Optional[float] = None
    solution_status: str = "exact_finite_columns"
    anytime_trace: Tuple[Mapping[str, float], ...] = ()
    baseline_makespan: Optional[float] = None
    baseline_energy: Optional[float] = None

    @property
    def objective(self) -> Tuple[float, float]:
        return (self.makespan, self.total_energy)


def ensure_route_tuple(routes: Sequence[CrownRoute]) -> Tuple[CrownRoute, ...]:
    """Return a deterministic tuple without changing route semantics."""

    return tuple(sorted(routes, key=lambda route: route.route_id))
