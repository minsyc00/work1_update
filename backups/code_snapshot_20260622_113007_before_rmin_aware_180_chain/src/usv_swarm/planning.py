from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dubins import dubins_shortest_path, sample_dubins_path
from .geometry import (
    distance_xy,
    mean_heading,
    polyline_length,
    sample_quintic_bezier,
    straight_segment_points,
    unit_heading,
    wrap_angle,
)
from .schema import (
    AssignmentPlan,
    ConstraintWindow,
    MAPFReservationTable,
    PathRequirement,
    PlannerConfig,
    PlanningResult,
    Pose2D,
    ReservationEntry,
    SmoothedPath,
    StripTask,
    TimedPathSegment,
    TrajectoryReference,
    TrajectorySample,
)


@dataclass
class _Conflict:
    kind: str
    agent_a: int
    agent_b: int
    start_time: float
    end_time: float
    resource_id_a: str
    resource_id_b: str
    from_node_a: str
    to_node_a: str
    from_node_b: str
    to_node_b: str


@dataclass(order=True)
class _CBSNode:
    priority: float
    serial: int
    constraints: Dict[int, List[ConstraintWindow]] = field(compare=False)
    reservations: Dict[int, List[ReservationEntry]] = field(compare=False)


def choose_scan_axis(config: PlannerConfig) -> str:
    lx = config.mission.area_length_x
    ly = config.mission.area_length_y
    if abs(lx - ly) > 1e-6:
        return "x" if lx >= ly else "y"
    avg_heading = mean_heading(state.psi for state in config.fleet.initial_states_3dof)
    dx = abs(math.cos(avg_heading))
    dy = abs(math.sin(avg_heading))
    return "x" if dx >= dy else "y"


def build_boustrophedon_strips(config: PlannerConfig) -> List[StripTask]:
    axis = choose_scan_axis(config)
    lx = config.mission.area_length_x
    ly = config.mission.area_length_y
    wf = config.footprint.width_wf
    lf = config.footprint.length_lf
    rho = config.mission.overlap_ratio
    delta = max(wf * (1.0 - rho), 1e-6)
    h = config.fleet.min_turn_radius + config.footprint.length_lf / 2.0 + config.safety.d_safe
    width = ly if axis == "x" else lx
    if width <= wf:
        strip_count = 1
    else:
        strip_count = int(math.ceil((width - wf) / delta) + 1)

    strips: List[StripTask] = []
    for idx in range(strip_count):
        center = width / 2.0 if strip_count == 1 else min(wf / 2.0 + idx * delta, width - wf / 2.0)
        if axis == "x":
            x_min = min(max(lf / 2.0, config.safety.boundary_margin_x), lx / 2.0)
            x_max = max(lx - x_min, x_min)
            if idx % 2 == 0:
                start = Pose2D(x_min, center, 0.0)
                end = Pose2D(x_max, center, 0.0)
            else:
                start = Pose2D(x_max, center, math.pi)
                end = Pose2D(x_min, center, math.pi)
            pocket_left = Pose2D(max(x_min - h, x_min), center, math.pi)
            pocket_right = Pose2D(min(x_max + h, x_max), center, 0.0)
            strip_length = max(x_max - x_min, 0.0)
        else:
            y_min = min(max(lf / 2.0, config.safety.boundary_margin_y), ly / 2.0)
            y_max = max(ly - y_min, y_min)
            if idx % 2 == 0:
                start = Pose2D(center, y_min, math.pi / 2.0)
                end = Pose2D(center, y_max, math.pi / 2.0)
            else:
                start = Pose2D(center, y_max, -math.pi / 2.0)
                end = Pose2D(center, y_min, -math.pi / 2.0)
            pocket_left = Pose2D(center, max(y_min - h, y_min), -math.pi / 2.0)
            pocket_right = Pose2D(center, min(y_max + h, y_max), math.pi / 2.0)
            strip_length = max(y_max - y_min, 0.0)
        strips.append(
            StripTask(
                strip_id=idx,
                start_pose=start,
                end_pose=end,
                nominal_heading=start.psi,
                strip_length=strip_length,
                pocket_left=pocket_left,
                pocket_right=pocket_right,
                scan_axis=axis,
                center_coordinate=center,
            )
        )
    return strips


def _sort_agent_ids_by_axis(config: PlannerConfig, axis: str) -> List[int]:
    states = config.fleet.initial_states_3dof
    if axis == "x":
        return [idx for idx, _ in sorted(enumerate(states), key=lambda pair: (pair[1].y, pair[1].x))]
    return [idx for idx, _ in sorted(enumerate(states), key=lambda pair: (pair[1].x, pair[1].y))]


def _estimate_block_cost(config: PlannerConfig, agent_id: int, block: Sequence[StripTask], average_cover_time: float) -> float:
    if not block:
        return 0.0
    state = config.fleet.initial_states_3dof[agent_id]
    cruise_speed = max(config.fleet.cruise_speed, 1e-6)
    cover_speed = max(config.fleet.cover_speed, 1e-6)
    turn_speed = max(min(config.fleet.turn_speed_max, config.fleet.cruise_speed), 1e-6)

    cover_time = sum(strip.strip_length / cover_speed for strip in block)
    transit_time = dubins_shortest_path(state.pose(), block[0].start_pose, config.fleet.min_turn_radius).total_length / cruise_speed
    turn_count = max(len(block) - 1, 0)
    for current_strip, next_strip in zip(block[:-1], block[1:]):
        transit_time += (
            dubins_shortest_path(current_strip.end_pose, next_strip.start_pose, config.fleet.min_turn_radius).total_length / turn_speed
        )
    imbalance = abs(cover_time - average_cover_time)
    return cover_time + transit_time + config.weights.lambda1 * turn_count + config.weights.lambda2 * imbalance


def solve_contiguous_partition(config: PlannerConfig, strips: Sequence[StripTask]) -> AssignmentPlan:
    axis = strips[0].scan_axis if strips else choose_scan_axis(config)
    agent_order = _sort_agent_ids_by_axis(config, axis)
    n_agents = config.fleet.num_agents or len(agent_order)
    n_strips = len(strips)
    assignments: Dict[int, Tuple[int, int]] = {agent_id: (-1, -1) for agent_id in range(n_agents)}
    ordered_tasks: Dict[int, List[StripTask]] = {agent_id: [] for agent_id in range(n_agents)}
    estimated_cost: Dict[int, float] = {agent_id: 0.0 for agent_id in range(n_agents)}

    if n_strips == 0:
        return AssignmentPlan(assignments=assignments, estimated_cost=estimated_cost, ordered_tasks=ordered_tasks, agent_order=agent_order)

    active_agents = min(n_agents, n_strips)
    active_agent_ids = agent_order[:active_agents]
    average_cover_time = sum(strip.strip_length / max(config.fleet.cover_speed, 1e-6) for strip in strips) / active_agents

    cost_cache: Dict[Tuple[int, int, int], float] = {}
    for agent_id in active_agent_ids:
        for start in range(n_strips):
            for end in range(start, n_strips):
                cost_cache[(agent_id, start, end)] = _estimate_block_cost(config, agent_id, strips[start : end + 1], average_cover_time)

    inf = float("inf")
    dp = np.full((active_agents + 1, n_strips + 1), inf, dtype=float)
    choice = np.full((active_agents + 1, n_strips + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for m in range(1, active_agents + 1):
        for j in range(m, n_strips + 1):
            for i in range(m - 1, j):
                block_cost = cost_cache[(active_agent_ids[m - 1], i, j - 1)]
                candidate = max(dp[m - 1, i], block_cost)
                if candidate < dp[m, j]:
                    dp[m, j] = candidate
                    choice[m, j] = i

    j = n_strips
    for m in range(active_agents, 0, -1):
        i = int(choice[m, j])
        agent_id = active_agent_ids[m - 1]
        block = list(strips[i:j])
        ordered_tasks[agent_id] = block
        assignments[agent_id] = (block[0].strip_id, block[-1].strip_id)
        estimated_cost[agent_id] = cost_cache[(agent_id, i, j - 1)]
        j = i

    for agent_id in range(n_agents):
        if not ordered_tasks[agent_id]:
            assignments[agent_id] = (-1, -1)
            estimated_cost[agent_id] = 0.0

    return AssignmentPlan(
        assignments=assignments,
        estimated_cost=estimated_cost,
        ordered_tasks=ordered_tasks,
        agent_order=agent_order,
    )


def _terminal_side(strip: StripTask) -> str:
    if strip.scan_axis == "x":
        return "right" if strip.end_pose.x >= strip.start_pose.x else "left"
    return "top" if strip.end_pose.y >= strip.start_pose.y else "bottom"


def build_path_requirements(config: PlannerConfig, assignments: AssignmentPlan) -> Dict[int, List[PathRequirement]]:
    requirements: Dict[int, List[PathRequirement]] = {agent_id: [] for agent_id in range(config.fleet.num_agents or 0)}
    cruise_speed = max(config.fleet.cruise_speed, 1e-6)
    cover_speed = max(config.fleet.cover_speed, 1e-6)
    turn_speed = max(config.fleet.turn_speed_max, 1e-6)

    for agent_id, tasks in assignments.ordered_tasks.items():
        if not tasks:
            continue
        seq_index = 0
        current_pose = config.fleet.initial_states_3dof[agent_id].pose()
        first = tasks[0]
        transit_duration = dubins_shortest_path(current_pose, first.start_pose, config.fleet.min_turn_radius).total_length / cruise_speed
        requirements[agent_id].append(
            PathRequirement(
                agent_id=agent_id,
                seq_index=seq_index,
                kind="transit",
                resource_id=f"transit:{agent_id}:{first.strip_id}",
                duration=transit_duration,
                from_node=f"start:{agent_id}",
                to_node=f"strip:{first.strip_id}:entry",
                start_pose=current_pose,
                end_pose=first.start_pose,
                strip_id=first.strip_id,
            )
        )
        seq_index += 1

        for idx, strip in enumerate(tasks):
            requirements[agent_id].append(
                PathRequirement(
                    agent_id=agent_id,
                    seq_index=seq_index,
                    kind="cover",
                    resource_id=f"strip:{strip.strip_id}",
                    duration=strip.strip_length / cover_speed,
                    from_node=f"strip:{strip.strip_id}:entry",
                    to_node=f"strip:{strip.strip_id}:exit",
                    start_pose=strip.start_pose,
                    end_pose=strip.end_pose,
                    strip_id=strip.strip_id,
                )
            )
            seq_index += 1

            if idx == len(tasks) - 1:
                continue
            next_strip = tasks[idx + 1]
            side = _terminal_side(strip)
            hold_pose = strip.end_pose
            hold_duration = (config.fleet.min_turn_radius + config.footprint.length_lf / 2.0 + config.safety.d_safe) / turn_speed
            requirements[agent_id].append(
                PathRequirement(
                    agent_id=agent_id,
                    seq_index=seq_index,
                    kind="hold",
                    resource_id=f"pocket:{side}:{min(strip.strip_id, next_strip.strip_id)}",
                    duration=hold_duration,
                    from_node=f"strip:{strip.strip_id}:exit",
                    to_node=f"strip:{strip.strip_id}:exit",
                    start_pose=hold_pose,
                    end_pose=hold_pose,
                    strip_id=strip.strip_id,
                )
            )
            seq_index += 1
            turn_duration = (
                dubins_shortest_path(strip.end_pose, next_strip.start_pose, config.fleet.min_turn_radius).total_length / turn_speed
            )
            requirements[agent_id].append(
                PathRequirement(
                    agent_id=agent_id,
                    seq_index=seq_index,
                    kind="turn",
                    resource_id=f"edge:strip:{strip.strip_id}->{next_strip.strip_id}",
                    duration=turn_duration,
                    from_node=f"strip:{strip.strip_id}:exit",
                    to_node=f"strip:{next_strip.strip_id}:entry",
                    start_pose=strip.end_pose,
                    end_pose=next_strip.start_pose,
                    strip_id=next_strip.strip_id,
                )
            )
            seq_index += 1

    return requirements


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) - max(a0, b0) > 1e-9


def _constraint_applies(requirement: PathRequirement, constraint: ConstraintWindow) -> bool:
    if requirement.resource_id != constraint.resource_id:
        return False
    if constraint.from_node is not None and requirement.from_node != constraint.from_node:
        return False
    if constraint.to_node is not None and requirement.to_node != constraint.to_node:
        return False
    return True


def _schedule_agent(requirements: Sequence[PathRequirement], constraints: Sequence[ConstraintWindow]) -> List[ReservationEntry]:
    reservations: List[ReservationEntry] = []
    current_time = 0.0
    sorted_constraints = sorted(constraints, key=lambda item: (item.start_time, item.end_time))
    for requirement in requirements:
        t_enter = current_time
        if requirement.duration < 1e-9:
            t_exit = t_enter
        else:
            while True:
                shifted = False
                for constraint in sorted_constraints:
                    if not _constraint_applies(requirement, constraint):
                        continue
                    if _overlaps(t_enter, t_enter + requirement.duration, constraint.start_time, constraint.end_time):
                        t_enter = constraint.end_time
                        shifted = True
                        break
                if not shifted:
                    break
            t_exit = t_enter + requirement.duration
        reservations.append(
            ReservationEntry(
                agent_id=requirement.agent_id,
                seq_index=requirement.seq_index,
                resource_id=requirement.resource_id,
                kind=requirement.kind,
                t_enter=t_enter,
                t_exit=t_exit,
                from_node=requirement.from_node,
                to_node=requirement.to_node,
                start_pose=requirement.start_pose,
                end_pose=requirement.end_pose,
                strip_id=requirement.strip_id,
            )
        )
        current_time = t_exit
    return reservations


def _find_first_conflict(reservations: Dict[int, List[ReservationEntry]]) -> Optional[_Conflict]:
    ordered_entries = list(
        itertools.chain.from_iterable(
            sorted(agent_reservations, key=lambda item: item.t_enter) for agent_reservations in reservations.values()
        )
    )
    earliest: Optional[_Conflict] = None
    for first_index in range(len(ordered_entries)):
        a = ordered_entries[first_index]
        for second_index in range(first_index + 1, len(ordered_entries)):
            b = ordered_entries[second_index]
            if a.agent_id == b.agent_id:
                continue
            if not _overlaps(a.t_enter, a.t_exit, b.t_enter, b.t_exit):
                continue
            overlap_start = max(a.t_enter, b.t_enter)
            overlap_end = min(a.t_exit, b.t_exit)
            candidate: Optional[_Conflict] = None
            if a.resource_id == b.resource_id:
                candidate = _Conflict(
                    kind="resource",
                    agent_a=a.agent_id,
                    agent_b=b.agent_id,
                    start_time=overlap_start,
                    end_time=overlap_end,
                    resource_id_a=a.resource_id,
                    resource_id_b=b.resource_id,
                    from_node_a=a.from_node,
                    to_node_a=a.to_node,
                    from_node_b=b.from_node,
                    to_node_b=b.to_node,
                )
            elif a.from_node == b.to_node and a.to_node == b.from_node:
                candidate = _Conflict(
                    kind="edge",
                    agent_a=a.agent_id,
                    agent_b=b.agent_id,
                    start_time=overlap_start,
                    end_time=overlap_end,
                    resource_id_a=a.resource_id,
                    resource_id_b=b.resource_id,
                    from_node_a=a.from_node,
                    to_node_a=a.to_node,
                    from_node_b=b.from_node,
                    to_node_b=b.to_node,
                )
            if candidate is not None and (earliest is None or candidate.start_time < earliest.start_time):
                earliest = candidate
    return earliest


def solve_cbs_mapf(config: PlannerConfig, requirements: Dict[int, List[PathRequirement]]) -> MAPFReservationTable:
    constraints: Dict[int, List[ConstraintWindow]] = {agent_id: [] for agent_id in requirements}
    root_reservations = {
        agent_id: _schedule_agent(agent_requirements, constraints[agent_id])
        for agent_id, agent_requirements in requirements.items()
    }
    makespan = max((entries[-1].t_exit for entries in root_reservations.values() if entries), default=0.0)
    serial = 0
    open_set: List[_CBSNode] = [
        _CBSNode(priority=makespan, serial=serial, constraints=constraints, reservations=root_reservations)
    ]
    serial += 1
    resolved_conflicts = 0

    while open_set:
        node = heapq.heappop(open_set)
        conflict = _find_first_conflict(node.reservations)
        if conflict is None:
            final_makespan = max((entries[-1].t_exit for entries in node.reservations.values() if entries), default=0.0)
            return MAPFReservationTable(
                reservations=node.reservations,
                conflicts_resolved=resolved_conflicts,
                makespan=final_makespan,
            )

        resolved_conflicts += 1
        branch_specs = [
            (
                conflict.agent_a,
                ConstraintWindow(
                    agent_id=conflict.agent_a,
                    resource_id=conflict.resource_id_a,
                    start_time=conflict.start_time,
                    end_time=conflict.end_time,
                    from_node=conflict.from_node_a if conflict.kind == "edge" else None,
                    to_node=conflict.to_node_a if conflict.kind == "edge" else None,
                ),
            ),
            (
                conflict.agent_b,
                ConstraintWindow(
                    agent_id=conflict.agent_b,
                    resource_id=conflict.resource_id_b,
                    start_time=conflict.start_time,
                    end_time=conflict.end_time,
                    from_node=conflict.from_node_b if conflict.kind == "edge" else None,
                    to_node=conflict.to_node_b if conflict.kind == "edge" else None,
                ),
            ),
        ]
        for agent_id, constraint in branch_specs:
            child_constraints = {key: list(value) for key, value in node.constraints.items()}
            child_constraints[agent_id].append(constraint)
            child_reservations = {key: list(value) for key, value in node.reservations.items()}
            child_reservations[agent_id] = _schedule_agent(requirements[agent_id], child_constraints[agent_id])
            child_makespan = max((entries[-1].t_exit for entries in child_reservations.values() if entries), default=0.0)
            heapq.heappush(
                open_set,
                _CBSNode(
                    priority=child_makespan,
                    serial=serial,
                    constraints=child_constraints,
                    reservations=child_reservations,
                ),
            )
            serial += 1

    return MAPFReservationTable(reservations=root_reservations, conflicts_resolved=resolved_conflicts, makespan=makespan)


def _build_feasible_transition_segment(
    start_pose: Pose2D,
    end_pose: Pose2D,
    turn_radius: float,
    sample_count: int,
) -> TimedPathSegment:
    dubins_path = dubins_shortest_path(start_pose, end_pose, turn_radius)
    if dubins_path.total_length < 1e-8:
        return TimedPathSegment(
            segment_type="turn",
            start_time=0.0,
            end_time=0.0,
            start_pose=start_pose,
            end_pose=end_pose,
            points=[(start_pose.x, start_pose.y), (end_pose.x, end_pose.y)],
            headings=[start_pose.psi, end_pose.psi],
            control_points=None,
            max_curvature=0.0,
            length=0.0,
            path_source="stationary",
            dubins_modes=dubins_path.modes,
        )

    base_distance = max(distance_xy(start_pose, end_pose), 1e-6)
    segment_lengths = dubins_path.segment_lengths
    tangent_scale = max(turn_radius * 0.85, min(max(segment_lengths[0], segment_lengths[-1], base_distance * 0.2), 2.5 * turn_radius))
    control_points: List[Tuple[float, float]] = []
    best_bezier: Optional[TimedPathSegment] = None

    for scale_multiplier in (0.8, 1.0, 1.25, 1.5, 1.8, 2.2, 2.8, 3.5):
        t0 = unit_heading(start_pose.psi)
        t1 = unit_heading(end_pose.psi)
        p0 = np.array([start_pose.x, start_pose.y], dtype=float)
        p5 = np.array([end_pose.x, end_pose.y], dtype=float)
        scale = tangent_scale * scale_multiplier
        p1 = p0 + scale * t0
        p2 = p1 + 0.75 * scale * t0
        p4 = p5 - scale * t1
        p3 = p4 - 0.75 * scale * t1
        control_points = [tuple(p0), tuple(p1), tuple(p2), tuple(p3), tuple(p4), tuple(p5)]
        points, headings, max_curvature = sample_quintic_bezier(control_points, sample_count=sample_count)
        bezier_length = polyline_length(points)
        feasible = max_curvature <= 1.0 / max(turn_radius, 1e-6) + 1e-3
        feasible = feasible and bezier_length <= dubins_path.total_length * 1.15
        if feasible:
            best_bezier = TimedPathSegment(
                segment_type="turn",
                start_time=0.0,
                end_time=0.0,
                start_pose=start_pose,
                end_pose=end_pose,
                points=points,
                headings=headings,
                control_points=control_points,
                max_curvature=max_curvature,
                length=bezier_length,
                path_source="bezier",
                dubins_modes=dubins_path.modes,
            )
            break

    if best_bezier is not None:
        return best_bezier

    dubins_points, dubins_headings, dubins_max_curvature = sample_dubins_path(
        dubins_path,
        step_size=max(dubins_path.total_length / max(sample_count - 1, 1), turn_radius / 8.0),
    )
    return TimedPathSegment(
        segment_type="turn",
        start_time=0.0,
        end_time=0.0,
        start_pose=start_pose,
        end_pose=end_pose,
        points=dubins_points,
        headings=dubins_headings,
        control_points=None,
        max_curvature=dubins_max_curvature,
        length=polyline_length(dubins_points),
        path_source="dubins_fallback",
        dubins_modes=dubins_path.modes,
    )


def build_smoothed_paths(config: PlannerConfig, reservations: MAPFReservationTable) -> Dict[int, SmoothedPath]:
    control_hz = max(config.mission.local_control_hz, 1.0)
    paths: Dict[int, SmoothedPath] = {}
    for agent_id, agent_reservations in reservations.reservations.items():
        if not agent_reservations:
            paths[agent_id] = SmoothedPath(agent_id=agent_id, segments=[], total_length=0.0, max_curvature=0.0)
            continue
        segments: List[TimedPathSegment] = []
        total_length = 0.0
        global_max_curvature = 0.0
        last_end_time = agent_reservations[0].t_enter
        last_pose = agent_reservations[0].start_pose

        for reservation in agent_reservations:
            if reservation.t_enter - last_end_time > 1e-9:
                hold_segment = TimedPathSegment(
                    segment_type="wait",
                    start_time=last_end_time,
                    end_time=reservation.t_enter,
                    start_pose=last_pose,
                    end_pose=last_pose,
                    points=[(last_pose.x, last_pose.y), (last_pose.x, last_pose.y)],
                    headings=[last_pose.psi, last_pose.psi],
                    control_points=None,
                    max_curvature=0.0,
                    length=0.0,
                    path_source="wait",
                )
                segments.append(hold_segment)

            duration = max(reservation.t_exit - reservation.t_enter, 0.0)
            sample_count = max(2, int(round(duration * control_hz)) + 1)
            if reservation.kind == "cover":
                points, headings = straight_segment_points(reservation.start_pose, reservation.end_pose, sample_count)
                segment = TimedPathSegment(
                    segment_type="cover",
                    start_time=reservation.t_enter,
                    end_time=reservation.t_exit,
                    start_pose=reservation.start_pose,
                    end_pose=reservation.end_pose,
                    points=points,
                    headings=headings,
                    control_points=None,
                    max_curvature=0.0,
                    length=polyline_length(points),
                    path_source="straight",
                )
            elif reservation.kind == "hold":
                segment = TimedPathSegment(
                    segment_type="hold",
                    start_time=reservation.t_enter,
                    end_time=reservation.t_exit,
                    start_pose=reservation.start_pose,
                    end_pose=reservation.end_pose,
                    points=[(reservation.start_pose.x, reservation.start_pose.y)] * sample_count,
                    headings=[reservation.start_pose.psi] * sample_count,
                    control_points=None,
                    max_curvature=0.0,
                    length=0.0,
                    path_source="hold",
                )
            else:
                segment = _build_feasible_transition_segment(
                    reservation.start_pose,
                    reservation.end_pose,
                    turn_radius=config.fleet.min_turn_radius,
                    sample_count=sample_count,
                )
                segment.segment_type = "turn" if reservation.kind == "turn" else "transit"
                segment.start_time = reservation.t_enter
                segment.end_time = reservation.t_exit
            segments.append(segment)
            total_length += segment.length
            global_max_curvature = max(global_max_curvature, segment.max_curvature)
            last_end_time = reservation.t_exit
            last_pose = reservation.end_pose

        paths[agent_id] = SmoothedPath(
            agent_id=agent_id,
            segments=segments,
            total_length=total_length,
            max_curvature=global_max_curvature,
        )
    return paths


def build_time_parameterized_references(config: PlannerConfig, paths: Dict[int, SmoothedPath]) -> Dict[int, TrajectoryReference]:
    refs: Dict[int, TrajectoryReference] = {}
    for agent_id, path in paths.items():
        samples: List[TrajectorySample] = []
        for segment in path.segments:
            duration = max(segment.end_time - segment.start_time, 0.0)
            count = max(len(segment.points), 2)
            for idx, ((x, y), psi) in enumerate(zip(segment.points, segment.headings)):
                if samples and idx == 0:
                    continue
                alpha = idx / max(count - 1, 1)
                time = segment.start_time + alpha * duration
                speed = segment.length / duration if duration > 1e-9 else 0.0
                if idx == 0:
                    yaw_rate = 0.0
                else:
                    prev_heading = segment.headings[idx - 1]
                    prev_alpha = (idx - 1) / max(count - 1, 1)
                    prev_time = segment.start_time + prev_alpha * duration
                    dt = max(time - prev_time, 1e-9)
                    yaw_rate = wrap_angle(psi - prev_heading) / dt
                samples.append(
                    TrajectorySample(
                        time=time,
                        x=x,
                        y=y,
                        psi=psi,
                        u_ref=speed,
                        r_ref=yaw_rate,
                        segment_type=segment.segment_type,
                    )
                )
        horizon_time = samples[-1].time if samples else 0.0
        refs[agent_id] = TrajectoryReference(agent_id=agent_id, samples=samples, horizon_time=horizon_time)
    return refs


def plan_global_coverage(config: PlannerConfig) -> PlanningResult:
    strips = build_boustrophedon_strips(config)
    assignments = solve_contiguous_partition(config, strips)
    requirements = build_path_requirements(config, assignments)
    reservations = solve_cbs_mapf(config, requirements)
    paths = build_smoothed_paths(config, reservations)
    refs = build_time_parameterized_references(config, paths)
    return PlanningResult(
        strips=strips,
        assignments=assignments,
        reservations=reservations,
        paths=paths,
        refs=refs,
    )
