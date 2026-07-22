"""Deterministic benchmark instances for CROWN correctness experiments."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

from .types import CrownInstance, CrownOperation, CrownRoute


def build_shared_corridor_counterexample(
    agent_count: int,
    epsilon: float,
) -> Tuple[CrownInstance, Mapping[str, Sequence[str]]]:
    """Build the constructive separation between joint and sequential planning.

    Agent ``k`` must cover task ``k``.  Its nominally shortest route occupies a
    single unary corridor for one time unit; its private route takes
    ``1 + epsilon`` and uses no shared resource.  Independent shortest-route
    selection therefore serializes all agents to makespan ``K``, whereas the
    joint solution uses at most one common route and finishes in
    ``1 + epsilon``.  The ratio is ``K / (1 + epsilon)``.
    """

    if agent_count < 2:
        raise ValueError("agent_count must be at least two")
    if epsilon <= 0.0 or epsilon >= agent_count - 1.0:
        raise ValueError("epsilon must satisfy 0 < epsilon < agent_count - 1")

    agent_ids = tuple(f"agent-{index}" for index in range(agent_count))
    task_ids = tuple(f"task-{index}" for index in range(agent_count))
    routes_by_agent = {}
    assignment = {}
    for index, (agent_id, task_id) in enumerate(zip(agent_ids, task_ids)):
        common = CrownRoute(
            route_id=f"{agent_id}:common",
            agent_id=agent_id,
            task_ids=(task_id,),
            operations=(
                CrownOperation(
                    operation_id="common-corridor",
                    duration=1.0,
                    resource_ids=("shared-corridor",),
                    energy=1.0,
                    kind="cover",
                ),
            ),
            mode_ids=("common",),
        )
        private = CrownRoute(
            route_id=f"{agent_id}:private",
            agent_id=agent_id,
            task_ids=(task_id,),
            operations=(
                CrownOperation(
                    operation_id="private-lane",
                    duration=1.0 + epsilon,
                    energy=1.0 + epsilon,
                    kind="cover",
                ),
            ),
            mode_ids=("private",),
        )
        routes_by_agent[agent_id] = (common, private)
        assignment[agent_id] = (task_id,)

    return (
        CrownInstance(
            agent_ids=agent_ids,
            task_ids=task_ids,
            routes_by_agent=routes_by_agent,
            metadata={
                "family": "shared-corridor-counterexample",
                "epsilon": epsilon,
                "expected_sequential_makespan": float(agent_count),
                "expected_joint_makespan": 1.0 + epsilon,
                "expected_ratio": agent_count / (1.0 + epsilon),
            },
        ),
        assignment,
    )
