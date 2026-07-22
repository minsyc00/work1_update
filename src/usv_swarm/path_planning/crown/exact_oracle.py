"""Full-enumeration oracle for the finite continuous-time CROWN model."""

from __future__ import annotations

from math import inf
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

from .exact_scheduler import schedule_selected_routes_exact
from .types import CrownInstance, CrownRoute, CrownSolution


_TOL = 1.0e-10


def _lexicographically_better(
    candidate: Tuple[float, float],
    incumbent: Tuple[float, float],
) -> bool:
    if candidate[0] < incumbent[0] - _TOL:
        return True
    return abs(candidate[0] - incumbent[0]) <= _TOL and candidate[1] < incumbent[1] - _TOL


def solve_joint_exact(instance: CrownInstance) -> CrownSolution:
    """Jointly select one route per agent and schedule all shared resources.

    The result is globally lexicographically optimal relative to the finite
    route universe in ``instance``.  This exponential solver is intentionally
    independent from CROWN-BPC and is used to certify its small-instance output.
    """

    task_universe = frozenset(instance.task_ids)
    best_objective = (inf, inf)
    best_routes: Optional[Tuple[CrownRoute, ...]] = None
    best_schedule = None
    complete_combinations = 0
    partial_nodes = 0

    def search(
        agent_index: int,
        covered_tasks: Set[str],
        selected_routes: list[CrownRoute],
    ) -> None:
        nonlocal best_objective, best_routes, best_schedule
        nonlocal complete_combinations, partial_nodes
        partial_nodes += 1
        if agent_index == len(instance.agent_ids):
            if frozenset(covered_tasks) != task_universe:
                return
            complete_combinations += 1
            schedule = schedule_selected_routes_exact(
                selected_routes,
                separation_time=instance.resource_separation,
                resource_capacities=instance.resource_capacities,
                wait_energy_rates=instance.wait_energy_rates,
            )
            energy = sum(route.energy for route in selected_routes) + schedule.waiting_energy
            objective = (schedule.makespan, energy)
            if _lexicographically_better(objective, best_objective):
                best_objective = objective
                best_routes = tuple(selected_routes)
                best_schedule = schedule
            return

        agent_id = instance.agent_ids[agent_index]
        for route in sorted(instance.routes_by_agent[agent_id], key=lambda item: item.route_id):
            route_tasks = set(route.task_ids)
            if covered_tasks.intersection(route_tasks):
                continue
            selected_routes.append(route)
            search(agent_index + 1, covered_tasks.union(route_tasks), selected_routes)
            selected_routes.pop()

    search(0, set(), [])
    if best_routes is None or best_schedule is None:
        raise ValueError("the route universe contains no exact-cover solution")

    return CrownSolution(
        routes=best_routes,
        schedule=best_schedule,
        makespan=best_objective[0],
        total_energy=best_objective[1],
        method="CROWN-ENUM",
        lower_bound=best_objective[0],
        upper_bound=best_objective[0],
        optimality_gap=0.0,
        statistics={
            "complete_route_combinations": complete_combinations,
            "partial_search_nodes": partial_nodes,
        },
    )


def solve_sequential_exact_post(
    instance: CrownInstance,
    fixed_assignment: Mapping[str, Sequence[str]],
) -> CrownSolution:
    """Strong sequential baseline: nominal local routes, then exact scheduling.

    The assignment is immutable and every agent independently chooses its
    shortest nominal route.  Only after those choices are frozen is the exact
    conflict scheduler called.  Thus any gap to ``solve_joint_exact`` is caused
    by the inability of post-processing to feed waiting back into assignment,
    order or mode decisions—not by a weak greedy scheduler.
    """

    expected_agents = set(instance.agent_ids)
    if set(fixed_assignment) != expected_agents:
        raise ValueError("fixed_assignment must contain exactly the instance agents")
    assigned_sets = {
        agent_id: frozenset(task_ids)
        for agent_id, task_ids in fixed_assignment.items()
    }
    covered = [task for tasks in assigned_sets.values() for task in tasks]
    if len(covered) != len(set(covered)) or set(covered) != set(instance.task_ids):
        raise ValueError("fixed_assignment must assign every task exactly once")

    selected = []
    for agent_id in instance.agent_ids:
        eligible = [
            route
            for route in instance.routes_by_agent[agent_id]
            if frozenset(route.task_ids) == assigned_sets[agent_id]
        ]
        if not eligible:
            raise ValueError(f"agent {agent_id!r} has no route for its fixed assignment")
        selected.append(
            min(
                eligible,
                key=lambda route: (route.nominal_duration, route.energy, route.route_id),
            )
        )

    nominal_makespan = max((route.nominal_duration for route in selected), default=0.0)
    schedule = schedule_selected_routes_exact(
        selected,
        separation_time=instance.resource_separation,
        resource_capacities=instance.resource_capacities,
        wait_energy_rates=instance.wait_energy_rates,
    )
    energy = sum(route.energy for route in selected) + schedule.waiting_energy
    return CrownSolution(
        routes=tuple(selected),
        schedule=schedule,
        makespan=schedule.makespan,
        total_energy=energy,
        method="Sequential-ExactPost",
        statistics={
            "nominal_makespan": nominal_makespan,
            "deconfliction_penalty": schedule.makespan - nominal_makespan,
        },
    )


def compare_joint_and_sequential(
    instance: CrownInstance,
    fixed_assignment: Mapping[str, Sequence[str]],
) -> Mapping[str, object]:
    """Run the fair two-method comparison and return paper-facing metrics."""

    sequential = solve_sequential_exact_post(instance, fixed_assignment)
    joint = solve_joint_exact(instance)
    gain = sequential.makespan - joint.makespan
    ratio = sequential.makespan / joint.makespan if joint.makespan > 0.0 else inf
    sequential_assignment = {
        route.agent_id: frozenset(route.task_ids) for route in sequential.routes
    }
    joint_assignment = {route.agent_id: frozenset(route.task_ids) for route in joint.routes}
    task_transfers = sum(
        sequential_assignment[agent_id] != joint_assignment[agent_id]
        for agent_id in instance.agent_ids
    )
    mode_changes = sum(
        sequential_route.mode_ids != joint_route.mode_ids
        for sequential_route, joint_route in zip(sequential.routes, joint.routes)
    )
    return {
        "sequential": sequential,
        "joint": joint,
        "joint_gain": gain,
        "joint_gain_ratio": ratio,
        "strict_improvement": gain > _TOL,
        "task_assignment_changes": task_transfers,
        "route_or_mode_changes": mode_changes,
    }
