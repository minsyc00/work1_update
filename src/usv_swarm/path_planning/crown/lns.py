"""Dual-guided scalable CROWN-LNS sharing the exact route-column model."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot, inf
from random import Random
from time import perf_counter
from typing import Dict, Mapping, Optional, Sequence, Tuple

from ..dynamics_validation import validate_transition_sequence
from ..obstacles import path_segment_invalid_reasons, segment_collides_with_obstacles
from ..smoothing import build_transition_segment
from .conflicts import assert_continuous_conflict_free
from .graph_bpc import (
    CrownRootRelaxation,
    service_workload_lower_bound,
    solve_crown_root_relaxation,
    solve_route_pool_dual_guidance,
)
from .mode_graph import CrownTimeExpandedModeGraph
from .lp import LinearProgramInfeasible
from .pricing import (
    CrownPricingDuals,
    CrownPricingRestrictions,
    PricingLabelLimitExceeded,
    PricingTimeLimitExceeded,
    build_fixed_mode_timed_route,
    price_mode_graph_exact,
)
from .types import CrownBpcSolution, CrownTimedRoute, ResourceSlot


_TOL = 1.0e-9


@dataclass(frozen=True)
class _LnsCandidate:
    assignment: Mapping[str, Tuple[str, ...]]
    routes: Tuple[CrownTimedRoute, ...]

    @property
    def makespan(self) -> float:
        return max((route.finish_time for route in self.routes), default=0.0)

    @property
    def energy(self) -> float:
        return sum(route.energy for route in self.routes)

    @property
    def objective(self) -> Tuple[float, float]:
        return (self.makespan, self.energy)


def _better(left: Tuple[float, float], right: Tuple[float, float]) -> bool:
    return left[0] < right[0] - _TOL or (
        abs(left[0] - right[0]) <= _TOL and left[1] < right[1] - _TOL
    )


def _greedy_assignment(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    task_prices: Mapping[str, float],
) -> Mapping[str, Tuple[str, ...]]:
    assigned: Dict[str, list[str]] = {agent_id: [] for agent_id in graphs}
    loads = {agent_id: 0.0 for agent_id in graphs}
    ordered_tasks = sorted(
        task_ids,
        key=lambda task: (
            -task_prices.get(task, 0.0),
            sum(bool(graph.modes_for_task(task)) for graph in graphs.values()),
            task,
        ),
    )
    for task_id in ordered_tasks:
        choices = []
        for agent_id, graph in graphs.items():
            modes = graph.modes_for_task(task_id)
            if not modes:
                continue
            limit = graph.crown_config.max_tasks_per_route
            if limit is not None and len(assigned[agent_id]) >= limit:
                continue
            service = min(mode.nominal_duration for mode in modes)
            choices.append((loads[agent_id] + service, service, agent_id))
        if not choices:
            raise ValueError(f"no CROWN agent can cover task {task_id!r}")
        _, service, selected_agent = min(choices)
        assigned[selected_agent].append(task_id)
        loads[selected_agent] += service
    return {agent_id: tuple(tasks) for agent_id, tasks in assigned.items()}


def _geometric_task_order(
    graph: CrownTimeExpandedModeGraph,
    tasks: Sequence[str],
) -> Tuple[str, ...]:
    """Cheap nearest-mode order used only to bootstrap an LNS incumbent."""

    remaining = set(tasks)
    ordered = []
    current = graph.start_pose
    while remaining:
        choices = []
        for task_id in remaining:
            for mode in graph.modes_for_task(task_id):
                choices.append(
                    (
                        hypot(mode.entry_pose.x - current.x, mode.entry_pose.y - current.y)
                        + mode.nominal_duration,
                        task_id,
                        mode.mode_id,
                        mode.exit_pose,
                    )
                )
        if not choices:
            break
        _, task_id, _, exit_pose = min(choices, key=lambda item: item[:3])
        ordered.append(task_id)
        remaining.remove(task_id)
        current = exit_pose
    return tuple(ordered)


def _connectivity_mode_path(
    graph: CrownTimeExpandedModeGraph,
    tasks: Sequence[str],
    deadline: float,
) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """Find one depot-rooted elementary mode path before timed pricing."""

    memo = set()

    def search(
        last_mode_id: Optional[str],
        remaining: frozenset[str],
        task_order: Tuple[str, ...],
        mode_order: Tuple[str, ...],
    ) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        if perf_counter() >= deadline:
            return None
        if not remaining:
            return task_order, mode_order
        key = (last_mode_id, remaining)
        if key in memo:
            return None
        memo.add(key)
        current_pose = (
            graph.start_pose
            if last_mode_id is None
            else graph.mode_lookup[last_mode_id].exit_pose
        )
        candidates = []
        for task_id in remaining:
            for mode in graph.modes_for_task(task_id):
                candidates.append(
                    (
                        hypot(
                            mode.entry_pose.x - current_pose.x,
                            mode.entry_pose.y - current_pose.y,
                        ),
                        task_id,
                        mode.mode_id,
                    )
                )
        for _, task_id, mode_id in sorted(candidates):
            if not graph.connection(last_mode_id, mode_id, 0).feasible:
                continue
            result = search(
                mode_id,
                remaining.difference({task_id}),
                task_order + (task_id,),
                mode_order + (mode_id,),
            )
            if result is not None:
                return result
        return None

    return search(None, frozenset(tasks), (), ())


def _greedy_connectivity_fleet_paths(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    deadline: float,
) -> Optional[
    Mapping[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]
]:
    """Build a scalable load-balanced fleet path using certified edges.

    The joint beam is useful on small, highly ambiguous instances but its
    state expansion is unsuitable for 50--65 responsibility units.  This
    constructor couples ownership, order, and mode choice one task at a time;
    every accepted extension is checked by the same obstacle-aware connector
    used by pricing.  Deterministic agent rotations provide cheap alternative
    tie breaks if one greedy chain reaches a dead end.
    """

    agent_ids = tuple(graphs)
    if not task_ids:
        return {agent_id: ((), ()) for agent_id in agent_ids}
    maximum_attempts = max(1, min(len(agent_ids), 6))
    for rotation in range(maximum_attempts):
        remaining = set(task_ids)
        task_paths = {agent_id: [] for agent_id in agent_ids}
        mode_paths = {agent_id: [] for agent_id in agent_ids}
        last_modes: Dict[str, Optional[str]] = {
            agent_id: None for agent_id in agent_ids
        }
        loads = {agent_id: 0.0 for agent_id in agent_ids}
        rotated = agent_ids[rotation:] + agent_ids[:rotation]
        rank = {agent_id: index for index, agent_id in enumerate(rotated)}

        while remaining and perf_counter() < deadline:
            candidates = []
            for agent_id in agent_ids:
                graph = graphs[agent_id]
                limit = graph.crown_config.max_tasks_per_route
                if limit is not None and len(task_paths[agent_id]) >= limit:
                    continue
                current_pose = (
                    graph.start_pose
                    if last_modes[agent_id] is None
                    else graph.mode_lookup[last_modes[agent_id]].exit_pose
                )
                other_max = max(
                    (load for other, load in loads.items() if other != agent_id),
                    default=0.0,
                )
                for task_id in remaining:
                    for mode in graph.modes_for_task(task_id):
                        distance = hypot(
                            mode.entry_pose.x - current_pose.x,
                            mode.entry_pose.y - current_pose.y,
                        )
                        estimated = (
                            loads[agent_id]
                            + distance / max(graph.profile.cruise_speed, _TOL)
                            + mode.nominal_duration
                        )
                        candidates.append(
                            (
                                max(other_max, estimated),
                                estimated,
                                distance,
                                rank[agent_id],
                                task_id,
                                mode.mode_id,
                                agent_id,
                            )
                        )
            selected = None
            for _, _, _, _, task_id, mode_id, agent_id in sorted(candidates):
                if perf_counter() >= deadline:
                    return None
                graph = graphs[agent_id]
                connection = graph.connection(
                    last_modes[agent_id], mode_id, 0
                )
                if connection.feasible:
                    selected = (task_id, mode_id, agent_id, connection.duration)
                    break
            if selected is None:
                break
            task_id, mode_id, agent_id, connector_duration = selected
            mode = graphs[agent_id].mode_lookup[mode_id]
            task_paths[agent_id].append(task_id)
            mode_paths[agent_id].append(mode_id)
            last_modes[agent_id] = mode_id
            loads[agent_id] += connector_duration + mode.nominal_duration
            remaining.remove(task_id)

        if not remaining:
            return {
                agent_id: (
                    tuple(task_paths[agent_id]),
                    tuple(mode_paths[agent_id]),
                )
                for agent_id in agent_ids
            }
    return None


def _sequential_agent_connectivity_fleet_path(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    deadline: float,
    diagnostics: Optional[Dict[str, object]] = None,
) -> Optional[Mapping[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]]:
    """Build a conservative seed from a few locally connected agent paths.

    Each agent greedily consumes locally reachable tasks.  After a bounded
    number of failed edge trials, the next physical depot takes over the
    residual tasks instead of spending the entire budget trying to escape one
    poor endpoint.  The resulting routes are serialized before certification,
    so this bootstrap cannot introduce an inter-agent collision.
    """

    agent_ids = tuple(graphs)
    maximum_covered = 0
    last_remaining: Tuple[str, ...] = tuple(task_ids)
    remaining = set(task_ids)
    fleet_paths: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
        agent_id: ((), ()) for agent_id in agent_ids
    }
    reference_graph = graphs[agent_ids[0]]
    largest_seed_tasks = set(
        sorted(
            task_ids,
            key=lambda task_id: (
                -min(
                    mode.nominal_duration
                    for mode in reference_graph.modes_for_task(task_id)
                ),
                task_id,
            ),
        )[: len(agent_ids)]
    )
    largest_seed_tasks = frozenset(largest_seed_tasks)
    maximum_edge_trials = 12
    for agent_index, selected_agent in enumerate(agent_ids):
        graph = graphs[selected_agent]
        limit = graph.crown_config.max_tasks_per_route
        remaining_agents = len(agent_ids) - agent_index
        now = perf_counter()
        agent_deadline = now + max(0.0, deadline - now) / max(
            remaining_agents,
            1,
        )
        task_path = []
        mode_path = []
        previous: Optional[str] = None
        while (
            remaining
            and perf_counter() < agent_deadline
            and (limit is None or len(task_path) < limit)
        ):
            current = (
                graph.start_pose
                if previous is None
                else graph.mode_lookup[previous].exit_pose
            )
            candidates = [
                (
                    (
                        0
                        if task_id in largest_seed_tasks
                        else (
                            1
                            if (
                                mode.pattern.metadata.get("source")
                                == "crown_sensor_offset_observation"
                                or mode.nominal_duration <= 10.0
                            )
                            else 2
                        )
                    ),
                    int(
                        graph.obstacle_field is not None
                        and segment_collides_with_obstacles(
                            (current.x, current.y),
                            (mode.entry_pose.x, mode.entry_pose.y),
                            graph.obstacle_field,
                            inflated=True,
                        )
                    ),
                    hypot(
                        mode.entry_pose.x - current.x,
                        mode.entry_pose.y - current.y,
                    ),
                    mode.nominal_duration,
                    task_id,
                    mode.mode_id,
                )
                for task_id in remaining
                for mode in graph.modes_for_task(task_id)
            ]
            selected = None
            edge_trial_limit = (
                len(candidates)
                if len(remaining) <= 12
                else max(
                    maximum_edge_trials,
                    2
                    * sum(
                        task_id in largest_seed_tasks
                        for task_id in remaining
                    ),
                )
            )
            for trial, (_, _, _, _, task_id, mode_id) in enumerate(
                sorted(candidates),
                start=1,
            ):
                if trial > edge_trial_limit:
                    break
                if perf_counter() >= agent_deadline:
                    break
                if graph.connection(previous, mode_id, 0).feasible:
                    selected = (task_id, mode_id)
                    break
            if selected is None:
                break
            task_id, mode_id = selected
            task_path.append(task_id)
            mode_path.append(mode_id)
            remaining.remove(task_id)
            previous = mode_id
            covered = len(task_ids) - len(remaining)
            if covered > maximum_covered:
                maximum_covered = covered
                last_remaining = tuple(sorted(remaining))
                if diagnostics is not None:
                    diagnostics["single_agent_max_covered"] = maximum_covered
                    diagnostics["single_agent_remaining"] = last_remaining[:8]
        fleet_paths[selected_agent] = (tuple(task_path), tuple(mode_path))
        if not remaining:
            if diagnostics is not None:
                diagnostics["single_agent_max_covered"] = len(task_ids)
                diagnostics["single_agent_remaining"] = ()
            return fleet_paths
    if diagnostics is not None:
        diagnostics["single_agent_max_covered"] = maximum_covered
        diagnostics["single_agent_remaining"] = last_remaining[:8]
    return None


def _task_reference_point(
    graph: CrownTimeExpandedModeGraph,
    task_id: str,
) -> Tuple[float, float]:
    """Return a stable spatial representative for a responsibility unit."""

    modes = graph.modes_for_task(task_id)
    points = [
        (pose.x, pose.y)
        for mode in modes
        for pose in (mode.entry_pose, mode.exit_pose)
    ]
    if not points:
        return (graph.start_pose.x, graph.start_pose.y)
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _fixed_task_order_mode_path(
    graph: CrownTimeExpandedModeGraph,
    tasks: Sequence[str],
    deadline: float,
    *,
    beam_width: int = 8,
) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """Choose connected modes for one spatially local fixed task order."""

    # State fields are accumulated duration, last mode, and selected modes.
    states: list[Tuple[float, Optional[str], Tuple[str, ...]]] = [
        (0.0, None, ())
    ]
    for task_id in tasks:
        if perf_counter() >= deadline:
            return None
        next_by_last: Dict[str, Tuple[float, str, Tuple[str, ...]]] = {}
        for cost, previous, selected_modes in states:
            for mode in graph.modes_for_task(task_id):
                if perf_counter() >= deadline:
                    return None
                connection = graph.connection(previous, mode.mode_id, 0)
                if not connection.feasible:
                    continue
                candidate = (
                    cost + connection.duration + mode.nominal_duration,
                    mode.mode_id,
                    selected_modes + (mode.mode_id,),
                )
                incumbent = next_by_last.get(mode.mode_id)
                if incumbent is None or candidate < incumbent:
                    next_by_last[mode.mode_id] = candidate
        if not next_by_last:
            return None
        states = sorted(next_by_last.values())[: max(1, beam_width)]
    if not states:
        return None
    _, _, modes = min(states)
    return tuple(tasks), modes


def _bcd_contiguous_connectivity_fleet_paths(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    deadline: float,
) -> Optional[Mapping[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]]:
    """Split the exact BCD order into load-balanced contiguous fleet blocks."""

    agent_ids = tuple(graphs)
    if not task_ids:
        return {agent_id: ((), ()) for agent_id in agent_ids}
    reference = graphs[agent_ids[0]]

    def bcd_key(task_id: str) -> Tuple[int, str]:
        try:
            serial = int(task_id.split("crown_cell_", 1)[1].split(":", 1)[0])
        except (IndexError, ValueError):
            serial = 10**9
        return serial, task_id

    ordered = tuple(sorted(task_ids, key=bcd_key))
    workload = {
        task_id: min(
            mode.nominal_duration
            for mode in reference.modes_for_task(task_id)
        )
        for task_id in ordered
    }
    chunks = []
    cursor = 0
    remaining_work = sum(workload.values())
    for agent_index in range(len(agent_ids)):
        agents_left = len(agent_ids) - agent_index
        tasks_left = len(ordered) - cursor
        if agent_index == len(agent_ids) - 1:
            chunks.append(ordered[cursor:])
            cursor = len(ordered)
            break
        target = remaining_work / max(agents_left, 1)
        chunk = []
        load = 0.0
        while cursor < len(ordered) and tasks_left - len(chunk) > agents_left - 1:
            task_id = ordered[cursor]
            if chunk and load >= target:
                break
            chunk.append(task_id)
            load += workload[task_id]
            cursor += 1
        chunks.append(tuple(chunk))
        remaining_work -= load
    if cursor != len(ordered) or len(chunks) != len(agent_ids):
        return None

    for reverse_all in (False, True):
        paths = {}
        complete = True
        selected_chunks = tuple(reversed(chunks)) if reverse_all else tuple(chunks)
        for agent_id, source_chunk in zip(agent_ids, selected_chunks):
            if perf_counter() >= deadline:
                return None
            order_candidates = (
                tuple(reversed(source_chunk)) if reverse_all else tuple(source_chunk),
                tuple(source_chunk) if reverse_all else tuple(reversed(source_chunk)),
            )
            path = None
            for order in order_candidates:
                path = _fixed_task_order_mode_path(
                    graphs[agent_id],
                    order,
                    deadline,
                )
                if path is not None:
                    break
            if path is None:
                complete = False
                break
            paths[agent_id] = path
        if complete:
            return paths
    return None


def _spatial_connectivity_fleet_paths(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    deadline: float,
) -> Optional[Mapping[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]]:
    """Construct a large-map incumbent from local clusters and exact edges.

    Global optimistic beams spend most of their work on cross-map temporary
    edges.  This constructor first clusters tasks around the physical depot
    poses while balancing nominal service, then validates only short local
    chains with the authoritative obstacle-aware connector.
    """

    agent_ids = tuple(graphs)
    if not task_ids:
        return {agent_id: ((), ()) for agent_id in agent_ids}
    reference_graph = graphs[agent_ids[0]]
    centers = {
        task_id: _task_reference_point(reference_graph, task_id)
        for task_id in task_ids
    }
    service = {
        task_id: min(
            mode.nominal_duration
            for graph in graphs.values()
            for mode in graph.modes_for_task(task_id)
        )
        for task_id in task_ids
    }
    mean_service = sum(service.values()) / max(len(service), 1)
    distance_weights = (0.25, 1.0, 4.0)
    for distance_weight in distance_weights:
        if perf_counter() >= deadline:
            return None
        assigned: Dict[str, list[str]] = {agent_id: [] for agent_id in agent_ids}
        loads = {agent_id: 0.0 for agent_id in agent_ids}
        # Large jobs establish balanced clusters first; deterministic spatial
        # tie breaks keep repeat runs identical.
        ordered_tasks = sorted(
            task_ids,
            key=lambda task_id: (
                -service[task_id],
                centers[task_id][0],
                centers[task_id][1],
                task_id,
            ),
        )
        for task_id in ordered_tasks:
            cx, cy = centers[task_id]
            choices = []
            for agent_id in agent_ids:
                graph = graphs[agent_id]
                limit = graph.crown_config.max_tasks_per_route
                if limit is not None and len(assigned[agent_id]) >= limit:
                    continue
                depot_distance = hypot(
                    cx - graph.start_pose.x,
                    cy - graph.start_pose.y,
                )
                choices.append(
                    (
                        loads[agent_id]
                        + service[task_id]
                        + distance_weight * mean_service * depot_distance
                        / max(
                            graph.planner_config.mission.area_length_x
                            + graph.planner_config.mission.area_length_y,
                            _TOL,
                        ),
                        loads[agent_id],
                        depot_distance,
                        agent_id,
                    )
                )
            if not choices:
                break
            _, _, _, selected_agent = min(choices)
            assigned[selected_agent].append(task_id)
            loads[selected_agent] += service[task_id]
        if sum(len(tasks) for tasks in assigned.values()) != len(task_ids):
            continue

        candidate_paths = {}
        complete = True
        for agent_id in agent_ids:
            graph = graphs[agent_id]
            remaining = set(assigned[agent_id])
            current = (graph.start_pose.x, graph.start_pose.y)
            nearest_order = []
            while remaining:
                task_id = min(
                    remaining,
                    key=lambda item: (
                        hypot(
                            centers[item][0] - current[0],
                            centers[item][1] - current[1],
                        ),
                        centers[item][0],
                        centers[item][1],
                        item,
                    ),
                )
                nearest_order.append(task_id)
                current = centers[task_id]
                remaining.remove(task_id)
            order_candidates = (
                tuple(nearest_order),
                tuple(reversed(nearest_order)),
                tuple(
                    sorted(
                        assigned[agent_id],
                        key=lambda item: (
                            centers[item][0], centers[item][1], item
                        ),
                    )
                ),
                tuple(
                    sorted(
                        assigned[agent_id],
                        key=lambda item: (
                            centers[item][1], centers[item][0], item
                        ),
                    )
                ),
            )
            path = None
            for order in order_candidates:
                path = _fixed_task_order_mode_path(
                    graph,
                    order,
                    deadline,
                )
                if path is not None:
                    break
            if path is None:
                complete = False
                break
            candidate_paths[agent_id] = path
        if complete:
            return candidate_paths
    return None


def _connectivity_fleet_paths(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    deadline: float,
    *,
    beam_width: int = 64,
    direct_only: bool = False,
) -> Optional[
    Mapping[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]
]:
    """Jointly choose ownership, order, and one connected mode per task.

    The former LNS bootstrap assigned tasks using service duration alone and
    only afterwards checked each agent's connector graph.  On obstacle maps
    that easily creates disconnected per-agent subsets even when a fleet-wide
    solution exists.  This bounded beam search keeps those three decisions
    coupled.  It is an incumbent constructor only; all timing, resource, and
    objective decisions are still made by the route pricing model below.
    """

    agent_ids = tuple(graphs)
    tasks = frozenset(task_ids)
    if not tasks:
        return {agent_id: ((), ()) for agent_id in agent_ids}

    # State fields are (remaining, task paths, mode paths, last modes, loads).
    initial = (
        tasks,
        tuple(() for _ in agent_ids),
        tuple(() for _ in agent_ids),
        tuple(None for _ in agent_ids),
        tuple(0.0 for _ in agent_ids),
    )
    states = [initial]
    direct_cache: Dict[Tuple[str, Optional[str], Optional[str]], bool] = {}

    def edge_feasible(
        graph: CrownTimeExpandedModeGraph,
        from_mode_id: Optional[str],
        to_mode_id: Optional[str],
    ) -> bool:
        if not direct_only:
            return graph.connection(from_mode_id, to_mode_id, 0).feasible
        # This phase is intentionally an optimistic geometric beam.  Building
        # and dynamically validating a throw-away direct segment for every
        # state/task/mode triple made the nominally cheap phase quadratic in
        # expensive geometry calls on 200/400 m instances.  Selected complete
        # paths are still materialized and checked with the authoritative
        # obstacle-aware connector below before they can become an incumbent.
        if (
            to_mode_id is None
            and not graph.goal_pose_explicit
            and not graph.crown_config.return_to_start
        ):
            return True
        if to_mode_id is not None:
            return True
        key = (graph.agent_id, from_mode_id, to_mode_id)
        if key in direct_cache:
            return direct_cache[key]
        if (
            to_mode_id is None
            and not graph.goal_pose_explicit
            and not graph.crown_config.return_to_start
        ):
            direct_cache[key] = True
            return True
        start, end = graph._poses_for_pair(from_mode_id, to_mode_id)
        if (
            abs(start.x - end.x) <= _TOL
            and abs(start.y - end.y) <= _TOL
            and abs(start.psi - end.psi) <= _TOL
        ):
            direct_cache[key] = True
            return True
        segment = build_transition_segment(
            segment_id="crown_lns_direct_connectivity_probe",
            start=start,
            end=end,
            start_time=0.0,
            config=graph.planner_config,
            kind="transit",
            use_bezier=graph.path_config.use_bezier_smoothing,
        )
        feasible = not path_segment_invalid_reasons(
            segment,
            graph.planner_config,
            graph.obstacle_field,
        ) and validate_transition_sequence(
            (segment,),
            graph.planner_config,
            obstacle_field=graph.obstacle_field,
            retime=True,
        ).valid
        direct_cache[key] = feasible
        # A failed straight/Dubins probe is not a reachability proof on an
        # obstacle map.  Keep the optimistic edge in the cheap beam and let
        # the selected complete paths invoke the exact obstacle-aware
        # connector.  The previous false return systematically removed the
        # very detour edges needed around islands and narrow obstacles.
        return True

    def terminal_feasible(
        remaining: frozenset[str],
        last_modes: Tuple[Optional[str], ...],
    ) -> bool:
        if remaining:
            return True
        for index, agent_id in enumerate(agent_ids):
            graph = graphs[agent_id]
            if (
                graph.goal_pose_explicit or graph.crown_config.return_to_start
            ) and not edge_feasible(graph, last_modes[index], None):
                return False
        return True

    for _ in range(len(tasks)):
        if perf_counter() >= deadline:
            return None
        next_by_key = {}
        for remaining, task_paths, mode_paths, last_modes, loads in states:
            candidates = []
            for agent_index, agent_id in enumerate(agent_ids):
                graph = graphs[agent_id]
                limit = graph.crown_config.max_tasks_per_route
                if limit is not None and len(task_paths[agent_index]) >= limit:
                    continue
                current_pose = (
                    graph.start_pose
                    if last_modes[agent_index] is None
                    else graph.mode_lookup[last_modes[agent_index]].exit_pose
                )
                for task_id in remaining:
                    for mode in graph.modes_for_task(task_id):
                        candidates.append(
                            (
                                hypot(
                                    mode.entry_pose.x - current_pose.x,
                                    mode.entry_pose.y - current_pose.y,
                                ),
                                loads[agent_index],
                                agent_index,
                                task_id,
                                mode.mode_id,
                            )
                        )

            # Nearby, currently under-loaded extensions are tested first.  We
            # do not truncate the list: sparse connector graphs often require
            # a non-nearest bridge around an obstacle.
            for _, _, agent_index, task_id, mode_id in sorted(candidates):
                if perf_counter() >= deadline:
                    return None
                agent_id = agent_ids[agent_index]
                graph = graphs[agent_id]
                if not edge_feasible(graph, last_modes[agent_index], mode_id):
                    continue
                mode = graph.mode_lookup[mode_id]
                new_remaining = remaining.difference({task_id})
                new_task_paths = list(task_paths)
                new_task_paths[agent_index] = task_paths[agent_index] + (task_id,)
                new_mode_paths = list(mode_paths)
                new_mode_paths[agent_index] = mode_paths[agent_index] + (mode_id,)
                new_last_modes = list(last_modes)
                new_last_modes[agent_index] = mode_id
                new_loads = list(loads)
                if direct_only:
                    start_pose = (
                        graph.start_pose
                        if last_modes[agent_index] is None
                        else graph.mode_lookup[last_modes[agent_index]].exit_pose
                    )
                    connector_duration = hypot(
                        mode.entry_pose.x - start_pose.x,
                        mode.entry_pose.y - start_pose.y,
                    ) / max(graph.profile.cruise_speed, _TOL)
                else:
                    connector_duration = graph.connection(
                        last_modes[agent_index], mode_id, 0
                    ).duration
                new_loads[agent_index] += connector_duration + mode.nominal_duration
                last_tuple = tuple(new_last_modes)
                if not terminal_feasible(new_remaining, last_tuple):
                    continue
                state = (
                    new_remaining,
                    tuple(new_task_paths),
                    tuple(new_mode_paths),
                    last_tuple,
                    tuple(new_loads),
                )
                # The remaining set and current endpoints determine all future
                # connectivity; task counts are needed for route-size limits.
                key = (
                    new_remaining,
                    last_tuple,
                    tuple(len(path) for path in new_task_paths),
                )
                score = (
                    max(new_loads),
                    sum(new_loads),
                    tuple(new_task_paths),
                    tuple(new_mode_paths),
                )
                incumbent = next_by_key.get(key)
                if incumbent is None or score < incumbent[0]:
                    next_by_key[key] = (score, state)
        if not next_by_key:
            return None
        states = [
            item[1]
            for item in sorted(next_by_key.values(), key=lambda item: item[0])[
                : max(1, beam_width)
            ]
        ]

    if not states:
        return None
    ordered_states = sorted(
        states,
        key=lambda state: (
            max(state[4]),
            sum(state[4]),
            state[1],
            state[2],
        ),
    )
    for _, task_paths, mode_paths, _, _ in ordered_states:
        if direct_only:
            # The cheap probe deliberately bypasses A*/lattice construction.
            # Materialize only the selected edges now so a current-dependent
            # or custom connector model cannot create a false incumbent.
            verified = True
            for index, agent_id in enumerate(agent_ids):
                if perf_counter() >= deadline:
                    return None
                graph = graphs[agent_id]
                previous: Optional[str] = None
                for mode_id in mode_paths[index]:
                    if perf_counter() >= deadline:
                        return None
                    if not graph.connection(previous, mode_id, 0).feasible:
                        verified = False
                        break
                    previous = mode_id
                if verified and (
                    graph.goal_pose_explicit or graph.crown_config.return_to_start
                ) and not graph.connection(previous, None, 0).feasible:
                    verified = False
                if not verified:
                    break
            if not verified:
                continue
        return {
            agent_id: (task_paths[index], mode_paths[index])
            for index, agent_id in enumerate(agent_ids)
        }
    return None


def _fixed_connectivity_candidate(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    connectivity_paths: Mapping[str, Tuple[Tuple[str, ...], Tuple[str, ...]]],
    *,
    horizon_slots: int,
) -> Optional[_LnsCandidate]:
    """Serialize connected paths into an immediate certified LNS incumbent."""

    agent_ids = tuple(graphs)
    orders = [
        tuple(sorted(agent_ids, key=lambda agent: (-len(connectivity_paths[agent][0]), agent))),
        tuple(sorted(agent_ids, key=lambda agent: (len(connectivity_paths[agent][0]), agent))),
    ]
    for order in orders:
        by_agent = {}
        next_slot = 0
        feasible = True
        for agent_id in order:
            route = build_fixed_mode_timed_route(
                graphs[agent_id],
                connectivity_paths[agent_id][1],
                horizon_slots=horizon_slots,
                earliest_slot=next_slot,
            )
            if route is None:
                feasible = False
                break
            by_agent[agent_id] = route
            if route.task_ids:
                next_slot = route.finish_slot + 1
        if not feasible:
            continue
        routes = tuple(by_agent[agent_id] for agent_id in agent_ids)
        try:
            assert_continuous_conflict_free(routes, graphs)
        except ValueError:
            continue
        return _LnsCandidate(
            assignment={
                agent_id: tuple(connectivity_paths[agent_id][0])
                for agent_id in agent_ids
            },
            routes=routes,
        )
    return None


def _forbidden_slots(
    routes: Sequence[CrownTimedRoute],
    capacities: Mapping[str, int],
) -> frozenset[ResourceSlot]:
    counts: Dict[ResourceSlot, int] = {}
    for route in routes:
        for resource in route.occupied_resource_slots:
            counts[resource] = counts.get(resource, 0) + 1
    return frozenset(
        resource
        for resource, count in counts.items()
        if count >= capacities.get(resource[0], 1)
    )


def _price_assigned_route(
    graph: CrownTimeExpandedModeGraph,
    assigned_tasks: Sequence[str],
    all_tasks: Sequence[str],
    *,
    horizon_slots: int,
    forbidden_slots: frozenset[ResourceSlot],
    task_prices: Mapping[str, float],
    resource_prices: Mapping[ResourceSlot, float],
    fixed_order: bool,
    deadline: Optional[float],
    allowed_mode_ids: Optional[frozenset[str]] = None,
) -> Optional[CrownTimedRoute]:
    required_successors = (
        dict(zip(assigned_tasks[:-1], assigned_tasks[1:]))
        if fixed_order
        else {}
    )
    restrictions = CrownPricingRestrictions(
        required_tasks=frozenset(assigned_tasks),
        forbidden_tasks=frozenset(set(all_tasks).difference(assigned_tasks)),
        required_successors=required_successors,
        forbidden_resource_slots=forbidden_slots,
        allowed_mode_ids=allowed_mode_ids,
    )
    result = price_mode_graph_exact(
        graph,
        horizon_slots=horizon_slots,
        duals=CrownPricingDuals(
            agent_dual=0.0,
            task_duals=task_prices,
            makespan_dual=-1.0,
            resource_duals={resource: -abs(price) for resource, price in resource_prices.items()},
            stage=1,
        ),
        restrictions=restrictions,
        label_limit=graph.crown_config.pricing_label_limit,
        exact=False,
        deadline=deadline,
    )
    return result.route


def _build_candidate(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    assignment: Mapping[str, Sequence[str]],
    capacities: Mapping[str, int],
    *,
    horizon_slots: int,
    task_prices: Mapping[str, float],
    resource_prices: Mapping[ResourceSlot, float],
    rng: Random,
    fixed_order: bool = False,
    deadline: Optional[float] = None,
    allowed_mode_ids_by_agent: Optional[Mapping[str, frozenset[str]]] = None,
) -> Optional[_LnsCandidate]:
    agent_ids = list(graphs)
    orderings = [
        sorted(agent_ids, key=lambda agent: (-len(assignment[agent]), agent)),
        sorted(agent_ids, key=lambda agent: (len(assignment[agent]), agent)),
    ]
    for _ in range(min(6, max(1, len(agent_ids)))):
        shuffled = list(agent_ids)
        rng.shuffle(shuffled)
        orderings.append(shuffled)

    for order in orderings:
        selected = []
        feasible = True
        for agent_id in order:
            if deadline is not None and perf_counter() >= deadline:
                return None
            route = _price_assigned_route(
                graphs[agent_id],
                assignment[agent_id],
                task_ids,
                horizon_slots=horizon_slots,
                forbidden_slots=_forbidden_slots(selected, capacities),
                task_prices=task_prices,
                resource_prices=resource_prices,
                fixed_order=fixed_order,
                deadline=deadline,
                allowed_mode_ids=(
                    None
                    if allowed_mode_ids_by_agent is None
                    else allowed_mode_ids_by_agent.get(agent_id)
                ),
            )
            if route is None:
                feasible = False
                break
            selected.append(route)
        if feasible:
            by_agent = {route.agent_id: route for route in selected}
            return _LnsCandidate(
                assignment={agent: tuple(assignment[agent]) for agent in graphs},
                routes=tuple(by_agent[agent] for agent in graphs),
            )
    return None


def _add_to_pool(
    route_pool: Dict[str, Dict[str, CrownTimedRoute]],
    routes: Sequence[CrownTimedRoute],
    limit: int,
) -> None:
    for route in routes:
        pool = route_pool.setdefault(route.agent_id, {})
        pool[route.timed_route_id] = route
        if len(pool) <= limit:
            continue
        ordered = sorted(
            pool.values(),
            key=lambda candidate: (
                candidate.finish_time,
                candidate.energy,
                len(candidate.occupied_resource_slots),
                candidate.timed_route_id,
            ),
        )[:limit]
        route_pool[route.agent_id] = {
            candidate.timed_route_id: candidate for candidate in ordered
        }


def _pool_recombine(
    agent_ids: Sequence[str],
    task_ids: Sequence[str],
    route_pool: Mapping[str, Mapping[str, CrownTimedRoute]],
    capacities: Mapping[str, int],
    *,
    deadline: float,
) -> Optional[_LnsCandidate]:
    target = frozenset(task_ids)
    ordered_agents = sorted(agent_ids, key=lambda agent: len(route_pool.get(agent, {})))
    best_objective = (inf, inf)
    best_routes: Optional[Tuple[CrownTimedRoute, ...]] = None
    resource_counts: Dict[ResourceSlot, int] = {}

    def search(
        index: int,
        covered: frozenset[str],
        selected: list[CrownTimedRoute],
        current_makespan: float,
        current_energy: float,
    ) -> None:
        nonlocal best_objective, best_routes
        if perf_counter() >= deadline:
            return
        if current_makespan > best_objective[0] + _TOL:
            return
        if index == len(ordered_agents):
            if covered != target:
                return
            objective = (current_makespan, current_energy)
            if _better(objective, best_objective):
                best_objective = objective
                best_routes = tuple(selected)
            return
        agent_id = ordered_agents[index]
        for route in route_pool.get(agent_id, {}).values():
            tasks = frozenset(route.task_ids)
            if covered.intersection(tasks):
                continue
            touched = []
            feasible = True
            for resource in route.occupied_resource_slots:
                count = resource_counts.get(resource, 0) + 1
                if count > capacities.get(resource[0], 1):
                    feasible = False
                    break
                resource_counts[resource] = count
                touched.append(resource)
            if feasible:
                selected.append(route)
                search(
                    index + 1,
                    covered.union(tasks),
                    selected,
                    max(current_makespan, route.finish_time),
                    current_energy + route.energy,
                )
                selected.pop()
            for resource in touched:
                resource_counts[resource] -= 1
                if resource_counts[resource] == 0:
                    del resource_counts[resource]

    search(0, frozenset(), [], 0.0, 0.0)
    if best_routes is None:
        return None
    assignment = {agent: () for agent in agent_ids}
    for route in best_routes:
        assignment[route.agent_id] = route.task_ids
    by_agent = {route.agent_id: route for route in best_routes}
    return _LnsCandidate(
        assignment=assignment,
        routes=tuple(by_agent[agent] for agent in agent_ids),
    )


def _resource_pressure(
    route_pool: Mapping[str, Mapping[str, CrownTimedRoute]],
) -> Mapping[ResourceSlot, float]:
    counts: Dict[ResourceSlot, int] = {}
    for routes in route_pool.values():
        for route in routes.values():
            for resource in route.occupied_resource_slots:
                counts[resource] = counts.get(resource, 0) + 1
    return {resource: float(max(0, count - 1)) for resource, count in counts.items()}


def _relocation_neighbor(
    current: _LnsCandidate,
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_prices: Mapping[str, float],
    destroy_fraction: float,
) -> Optional[Mapping[str, Tuple[str, ...]]]:
    if not current.routes:
        return None
    bottleneck = max(current.routes, key=lambda route: route.finish_time).agent_id
    source_tasks = list(current.assignment[bottleneck])
    if not source_tasks:
        return None
    chain_size = min(
        len(source_tasks),
        max(1, int(ceil(len(source_tasks) * destroy_fraction))),
    )
    starts = range(len(source_tasks) - chain_size + 1)
    chain_start = max(
        starts,
        key=lambda index: (
            sum(
                task_prices.get(task, 0.0)
                for task in source_tasks[index : index + chain_size]
            ),
            -index,
        ),
    )
    chain = source_tasks[chain_start : chain_start + chain_size]
    targets = [
        agent_id
        for agent_id, graph in graphs.items()
        if agent_id != bottleneck
        and all(graph.modes_for_task(task) for task in chain)
        and (
            graph.crown_config.max_tasks_per_route is None
            or len(current.assignment[agent_id]) + len(chain)
            <= graph.crown_config.max_tasks_per_route
        )
    ]
    if not targets:
        return None
    target = min(targets, key=lambda agent: (len(current.assignment[agent]), agent))
    result = {agent: list(tasks) for agent, tasks in current.assignment.items()}
    del result[bottleneck][chain_start : chain_start + chain_size]
    result[target].extend(chain)
    return {agent: tuple(tasks) for agent, tasks in result.items()}


def _chain_exchange_neighbor(
    current: _LnsCandidate,
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    rng: Random,
    destroy_fraction: float,
) -> Optional[Mapping[str, Tuple[str, ...]]]:
    nonempty = [agent for agent, tasks in current.assignment.items() if tasks]
    if len(nonempty) < 2:
        return None
    left, right = rng.sample(nonempty, 2)
    left_tasks = list(current.assignment[left])
    right_tasks = list(current.assignment[right])
    left_size = min(len(left_tasks), max(1, int(ceil(len(left_tasks) * destroy_fraction))))
    right_size = min(len(right_tasks), max(1, int(ceil(len(right_tasks) * destroy_fraction))))
    left_start = rng.randrange(len(left_tasks) - left_size + 1)
    right_start = rng.randrange(len(right_tasks) - right_size + 1)
    left_chain = left_tasks[left_start : left_start + left_size]
    right_chain = right_tasks[right_start : right_start + right_size]
    if not all(graphs[left].modes_for_task(task) for task in right_chain) or not all(
        graphs[right].modes_for_task(task) for task in left_chain
    ):
        return None
    if (
        graphs[left].crown_config.max_tasks_per_route is not None
        and len(left_tasks) - left_size + right_size
        > graphs[left].crown_config.max_tasks_per_route
    ) or (
        graphs[right].crown_config.max_tasks_per_route is not None
        and len(right_tasks) - right_size + left_size
        > graphs[right].crown_config.max_tasks_per_route
    ):
        return None
    result = {agent: list(tasks) for agent, tasks in current.assignment.items()}
    result[left][left_start : left_start + left_size] = right_chain
    result[right][right_start : right_start + right_size] = left_chain
    return {agent: tuple(tasks) for agent, tasks in result.items()}


def _select_neighborhood(
    rng: Random,
    task_prices: Mapping[str, float],
    makespan_prices: Mapping[str, float],
    resource_prices: Mapping[ResourceSlot, float],
) -> int:
    """Sample one of the four core neighborhoods using current master duals."""

    task_signal = sum(abs(value) for value in task_prices.values()) / max(len(task_prices), 1)
    bottleneck_signal = sum(abs(value) for value in makespan_prices.values()) / max(
        len(makespan_prices), 1
    )
    conflict_signal = sum(abs(value) for value in resource_prices.values()) / max(
        len(resource_prices), 1
    )
    weights = (
        1.0 + task_signal + bottleneck_signal,
        1.0 + task_signal,
        1.0,
        1.0 + 2.0 * conflict_signal,
    )
    draw = rng.random() * sum(weights)
    cumulative = 0.0
    for index, weight in enumerate(weights, start=1):
        cumulative += weight
        if draw <= cumulative:
            return index
    return 4


def solve_crown_lns(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    *,
    horizon: float,
    resource_capacities: Optional[Mapping[str, int]] = None,
    initial_routes: Sequence[CrownTimedRoute] = (),
) -> CrownBpcSolution:
    """Run the four CROWN-LNS neighborhoods with pool recombination/certificate."""

    if not graphs:
        raise ValueError("at least one mode graph is required")
    steps = {graph.crown_config.time_step for graph in graphs.values()}
    if len(steps) != 1:
        raise ValueError("all graphs must use the same CROWN time step")
    time_step = next(iter(steps))
    horizon_slots = int(horizon / time_step + _TOL)
    config = next(iter(graphs.values())).crown_config
    capacities = dict(resource_capacities or {})
    rng = Random(config.lns_random_seed)
    started = perf_counter()
    deadline = started + config.lns_time_budget_sec
    service_lb = service_workload_lower_bound(graphs, task_ids)
    root: Optional[CrownRootRelaxation] = None
    if config.root_exact_pricing:
        # A scalable run must reserve time for constructing a certified
        # conflict-free incumbent.  Exact root pricing may consume its slice,
        # but cannot starve the LNS and turn a solvable instance into a
        # spurious "no initial solution" failure.
        root_deadline = min(
            deadline,
            started + max(0.1, config.lns_time_budget_sec * 0.25),
        )
        try:
            root = solve_crown_root_relaxation(
                graphs,
                task_ids,
                horizon=horizon,
                resource_capacities=capacities,
                deadline=root_deadline,
            )
        except (PricingLabelLimitExceeded, PricingTimeLimitExceeded, ValueError):
            root = None
    task_prices = root.task_duals if root is not None else {task: 0.0 for task in task_ids}
    makespan_prices = root.makespan_duals if root is not None else {agent: 0.0 for agent in graphs}
    resource_prices: Mapping[ResourceSlot, float] = (
        root.resource_duals if root is not None else {}
    )
    route_pool: Dict[str, Dict[str, CrownTimedRoute]] = {agent: {} for agent in graphs}
    if root is not None:
        _add_to_pool(
            route_pool,
            root.route_pool,
            config.lns_max_route_pool_per_agent,
        )
    if initial_routes:
        _add_to_pool(
            route_pool,
            initial_routes,
            config.lns_max_route_pool_per_agent,
        )
    initial = _pool_recombine(
        tuple(graphs),
        task_ids,
        route_pool,
        capacities,
        deadline=min(deadline, perf_counter() + max(0.1, config.lns_time_budget_sec * 0.1)),
    )
    initialization_diagnostics = {
        "pool_recombined": initial is not None,
        "bcd_contiguous_connectivity": False,
        "single_agent_connectivity": False,
        "spatial_connectivity": False,
        "greedy_connectivity": False,
        "direct_beam_connectivity": False,
        "exact_beam_connectivity": False,
        "fixed_connectivity_candidate": False,
        "fixed_order_pricing_candidate": False,
        "free_order_pricing_candidate": False,
    }
    if initial is None:
        # Small obstacle instances can have many Euclidean-near assignments
        # whose selected entry headings are not jointly connectable.  Keeping
        # only 64 optimistic beam states discarded the first valid chain on
        # the bundled 20 m map.  A wider small-instance beam is inexpensive;
        # retain the bounded 64-state search once task count is large.
        connectivity_beam_width = min(
            config.lns_max_route_pool_per_agent,
            200 if len(task_ids) <= 12 else 64,
        )
        connectivity_paths = None
        # The exact BCD cell order is also a strong deterministic seed on the
        # bundled 15 m obstacle map (eight cells).  Reserving it only for
        # instances above twelve tasks forced that small case into a much
        # broader joint beam and could exhaust the whole first-solution
        # budget despite a connected contiguous partition existing.
        if len(task_ids) >= 8:
            bcd_budget_fraction = 0.50 if len(task_ids) <= 12 else 0.25
            connectivity_paths = _bcd_contiguous_connectivity_fleet_paths(
                graphs,
                task_ids,
                min(
                    deadline,
                    perf_counter()
                    + max(
                        0.1,
                        config.lns_time_budget_sec * bcd_budget_fraction,
                    ),
                ),
            )
            initialization_diagnostics["bcd_contiguous_connectivity"] = (
                connectivity_paths is not None
            )
        if connectivity_paths is None and len(task_ids) > 12:
            connectivity_paths = _sequential_agent_connectivity_fleet_path(
                graphs,
                task_ids,
                min(
                    deadline,
                    perf_counter()
                    + max(0.1, config.lns_time_budget_sec * 0.60),
                ),
                diagnostics=initialization_diagnostics,
            )
            initialization_diagnostics["single_agent_connectivity"] = (
                connectivity_paths is not None
            )
        if connectivity_paths is None and len(task_ids) > 12:
            connectivity_paths = _spatial_connectivity_fleet_paths(
                graphs,
                task_ids,
                min(
                    deadline,
                    perf_counter()
                    + max(0.1, config.lns_time_budget_sec * 0.10),
                ),
            )
            initialization_diagnostics["spatial_connectivity"] = (
                connectivity_paths is not None
            )
        if connectivity_paths is None and len(task_ids) > 12:
            connectivity_paths = _greedy_connectivity_fleet_paths(
                graphs,
                task_ids,
                min(
                    deadline,
                    perf_counter()
                    + max(0.1, config.lns_time_budget_sec * 0.05),
                ),
            )
            initialization_diagnostics["greedy_connectivity"] = (
                connectivity_paths is not None
            )
        if connectivity_paths is None:
            connectivity_paths = _connectivity_fleet_paths(
                graphs,
                task_ids,
                min(deadline, perf_counter() + max(0.1, config.lns_time_budget_sec * 0.5)),
                beam_width=connectivity_beam_width,
                direct_only=True,
            )
            initialization_diagnostics["direct_beam_connectivity"] = (
                connectivity_paths is not None
            )
        if connectivity_paths is None and perf_counter() < deadline:
            connectivity_paths = _connectivity_fleet_paths(
                graphs,
                task_ids,
                deadline,
                beam_width=connectivity_beam_width,
                direct_only=False,
            )
            initialization_diagnostics["exact_beam_connectivity"] = (
                connectivity_paths is not None
            )
        if connectivity_paths is not None:
            assignment = {
                agent_id: connectivity_paths[agent_id][0]
                for agent_id in graphs
            }
            allowed_modes = {
                agent_id: frozenset(connectivity_paths[agent_id][1])
                for agent_id in graphs
            }
            initial = _fixed_connectivity_candidate(
                graphs,
                connectivity_paths,
                horizon_slots=horizon_slots,
            )
            initialization_diagnostics["fixed_connectivity_candidate"] = (
                initial is not None
            )
        else:
            assignment = _greedy_assignment(graphs, task_ids, task_prices)
            assignment = {
                agent_id: _geometric_task_order(graphs[agent_id], tasks)
                for agent_id, tasks in assignment.items()
            }
            allowed_modes = None
        if initial is None:
            initial = _build_candidate(
                graphs,
                task_ids,
                assignment,
                capacities,
                horizon_slots=horizon_slots,
                task_prices=task_prices,
                resource_prices={},
                rng=rng,
                fixed_order=True,
                deadline=deadline,
                allowed_mode_ids_by_agent=allowed_modes,
            )
            initialization_diagnostics["fixed_order_pricing_candidate"] = (
                initial is not None
            )
        if initial is None and perf_counter() < deadline:
            initial = _build_candidate(
                graphs,
                task_ids,
                assignment,
                capacities,
                horizon_slots=horizon_slots,
                task_prices=task_prices,
                resource_prices={},
                rng=rng,
                fixed_order=False,
                deadline=deadline,
            )
            initialization_diagnostics["free_order_pricing_candidate"] = (
                initial is not None
            )
    if initial is None:
        elapsed = perf_counter() - started
        cache_sizes = {
            agent_id: len(graph.connection_segment_cache)
            for agent_id, graph in graphs.items()
        }
        raise ValueError(
            "CROWN-LNS could not construct an initial conflict-free solution; "
            f"elapsed={elapsed:.3f}s diagnostics={initialization_diagnostics} "
            f"connection_cache_sizes={cache_sizes}"
        )
    current = best = initial
    _add_to_pool(route_pool, best.routes, config.lns_max_route_pool_per_agent)
    lower_bound = max(service_lb, root.objective if root is not None else 0.0)
    trace = [
        {
            "time": perf_counter() - started,
            "lower_bound": lower_bound,
            "upper_bound": best.makespan,
            "gap": max(0.0, (best.makespan - lower_bound) / max(best.makespan, _TOL)),
            "iteration": 0.0,
        }
    ]
    iterations = 0
    while iterations < config.lns_iterations and perf_counter() < deadline:
        iterations += 1
        neighborhood = _select_neighborhood(
            rng,
            task_prices,
            makespan_prices,
            resource_prices,
        )
        if neighborhood == 1:
            proposed_assignment = _relocation_neighbor(
                current,
                graphs,
                task_prices,
                config.lns_destroy_fraction,
            )
            fixed_order = False
        elif neighborhood == 2:
            proposed_assignment = _chain_exchange_neighbor(
                current,
                graphs,
                rng,
                config.lns_destroy_fraction,
            )
            fixed_order = False
        elif neighborhood == 3:
            # Mode/endpoint DP analogue: preserve the complete responsibility
            # order while pricing all mode choices and departure times.
            proposed_assignment = current.assignment
            fixed_order = True
        else:
            # Conflict-resource replanning uses route-pool congestion prices.
            proposed_assignment = current.assignment
            fixed_order = False
        if proposed_assignment is None:
            continue
        candidate = _build_candidate(
            graphs,
            task_ids,
            proposed_assignment,
            capacities,
            horizon_slots=horizon_slots,
            task_prices=task_prices,
            resource_prices=(resource_prices or _resource_pressure(route_pool)),
            rng=rng,
            fixed_order=fixed_order,
            deadline=deadline,
        )
        if candidate is None:
            continue
        _add_to_pool(route_pool, candidate.routes, config.lns_max_route_pool_per_agent)
        if _better(candidate.objective, current.objective):
            current = candidate
        if _better(candidate.objective, best.objective):
            best = candidate
            trace.append(
                {
                    "time": perf_counter() - started,
                    "lower_bound": lower_bound,
                    "upper_bound": best.makespan,
                    "gap": max(
                        0.0,
                        (best.makespan - lower_bound) / max(best.makespan, _TOL),
                    ),
                    "iteration": float(iterations),
                }
            )
        if iterations % config.lns_pool_reopt_interval == 0:
            pool_columns = tuple(
                route
                for routes in route_pool.values()
                for route in routes.values()
            )
            try:
                guidance = solve_route_pool_dual_guidance(
                    tuple(graphs),
                    task_ids,
                    pool_columns,
                    resource_capacities=capacities,
                )
            except (LinearProgramInfeasible, ValueError):
                guidance = None
            if guidance is not None:
                task_prices = guidance.task_duals
                makespan_prices = guidance.makespan_duals
                resource_prices = guidance.resource_duals
            recombined = _pool_recombine(
                tuple(graphs),
                task_ids,
                route_pool,
                capacities,
                deadline=min(deadline, perf_counter() + 1.0),
            )
            if recombined is not None and _better(recombined.objective, best.objective):
                best = current = recombined
                trace.append(
                    {
                        "time": perf_counter() - started,
                        "lower_bound": lower_bound,
                        "upper_bound": best.makespan,
                        "gap": max(
                            0.0,
                            (best.makespan - lower_bound) / max(best.makespan, _TOL),
                        ),
                        "iteration": float(iterations),
                    }
                )

    if config.enable_continuous_conflict_validation:
        assert_continuous_conflict_free(best.routes, graphs)
    gap = max(0.0, (best.makespan - lower_bound) / max(best.makespan, _TOL))
    pool_size = sum(len(routes) for routes in route_pool.values())
    status = (
        "certified_lns_root_exact_pricing"
        if root is not None
        else "certified_lns_service_bound"
    )
    return CrownBpcSolution(
        timed_routes=tuple(sorted(best.routes, key=lambda route: route.agent_id)),
        makespan=best.makespan,
        total_energy=best.energy,
        lower_bound=lower_bound,
        upper_bound=best.makespan,
        optimality_gap=gap,
        energy_lower_bound=0.0,
        energy_upper_bound=best.energy,
        energy_optimality_gap=(1.0 if best.energy > _TOL else 0.0),
        active_conflict_resources=tuple(
            sorted(
                resource
                for resource, price in resource_prices.items()
                if abs(price) > _TOL
            )
        ),
        generated_columns=pool_size,
        pricing_iterations=(root.pricing_iterations if root is not None else 0) + iterations,
        branch_nodes=0,
        conflict_separation_rounds=0,
        time_step=time_step,
        horizon_slots=horizon_slots,
        pricing_labels=root.pricing_labels if root is not None else 0,
        root_lp_lower_bound=root.objective if root is not None else None,
        service_lower_bound=service_lb,
        solution_status=status,
        anytime_trace=tuple(trace),
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
