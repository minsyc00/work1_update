"""Continuous trajectory-tube validation and canonical conflict resources."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, hypot
from typing import Mapping, Optional, Sequence, Tuple

from .mode_graph import CrownTimeExpandedModeGraph
from .types import CrownTimedRoute, ResourceSlot


_TOL = 1.0e-9


class CrownResourceMappingError(RuntimeError):
    pass


@dataclass(frozen=True)
class CrownContinuousConflict:
    left_agent: str
    right_agent: str
    left_operation_index: int
    right_operation_index: int
    start_time: float
    end_time: float
    minimum_time: float
    minimum_distance: float
    required_distance: float
    mapped_resources: Tuple[ResourceSlot, ...]


@dataclass(frozen=True)
class _MotionWindow:
    agent_id: str
    operation_index: int
    start_time: float
    motion_end_time: float
    allocated_end_time: float
    start: Tuple[float, float]
    end: Tuple[float, float]
    resource_ids: Tuple[str, ...]
    time_step: float

    def position_velocity(self, time: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        duration = self.motion_end_time - self.start_time
        if duration <= _TOL or time >= self.motion_end_time - _TOL:
            return self.end, (0.0, 0.0)
        alpha = max(0.0, min(1.0, (time - self.start_time) / duration))
        position = (
            self.start[0] + alpha * (self.end[0] - self.start[0]),
            self.start[1] + alpha * (self.end[1] - self.start[1]),
        )
        velocity = (
            (self.end[0] - self.start[0]) / duration,
            (self.end[1] - self.start[1]) / duration,
        )
        return position, velocity


def _operation_windows(route: CrownTimedRoute) -> Tuple[_MotionWindow, ...]:
    windows = []
    for index, (operation, start_slot, duration_slots) in enumerate(
        zip(
            route.base_route.operations,
            route.start_slots,
            route.duration_slots,
        )
    ):
        metadata = operation.metadata
        required = ("start_x", "start_y", "end_x", "end_y")
        if any(key not in metadata for key in required):
            continue
        start_time = start_slot * route.time_step
        windows.append(
            _MotionWindow(
                agent_id=route.agent_id,
                operation_index=index,
                start_time=start_time,
                motion_end_time=start_time + operation.duration,
                allocated_end_time=(start_slot + duration_slots) * route.time_step,
                start=(float(metadata["start_x"]), float(metadata["start_y"])),
                end=(float(metadata["end_x"]), float(metadata["end_y"])),
                resource_ids=tuple(operation.resource_ids),
                time_step=route.time_step,
            )
        )
    return tuple(windows)


def _minimum_distance(
    left: _MotionWindow,
    right: _MotionWindow,
    start_time: float,
    end_time: float,
) -> Tuple[float, float]:
    breakpoints = {start_time, end_time}
    for point in (left.motion_end_time, right.motion_end_time):
        if start_time < point < end_time:
            breakpoints.add(point)
    ordered = sorted(breakpoints)
    best_distance = float("inf")
    best_time = start_time
    for interval_start, interval_end in zip(ordered[:-1], ordered[1:]):
        left_position, left_velocity = left.position_velocity(interval_start)
        right_position, right_velocity = right.position_velocity(interval_start)
        delta = (
            left_position[0] - right_position[0],
            left_position[1] - right_position[1],
        )
        relative_velocity = (
            left_velocity[0] - right_velocity[0],
            left_velocity[1] - right_velocity[1],
        )
        duration = interval_end - interval_start
        denominator = relative_velocity[0] ** 2 + relative_velocity[1] ** 2
        candidates = [0.0, duration]
        if denominator > _TOL:
            stationary = -(
                delta[0] * relative_velocity[0]
                + delta[1] * relative_velocity[1]
            ) / denominator
            candidates.append(max(0.0, min(duration, stationary)))
        for offset in candidates:
            dx = delta[0] + relative_velocity[0] * offset
            dy = delta[1] + relative_velocity[1] * offset
            distance = hypot(dx, dy)
            if distance < best_distance:
                best_distance = distance
                best_time = interval_start + offset
    if not ordered[:-1]:
        left_position, _ = left.position_velocity(start_time)
        right_position, _ = right.position_velocity(start_time)
        return hypot(
            left_position[0] - right_position[0],
            left_position[1] - right_position[1],
        ), start_time
    return best_distance, best_time


def _mapped_resources(
    left: _MotionWindow,
    right: _MotionWindow,
    minimum_time: float,
) -> Tuple[ResourceSlot, ...]:
    shared = set(left.resource_ids).intersection(right.resource_ids)
    slot = int(floor(minimum_time / left.time_step + _TOL))
    candidates = []
    for resource_id in shared:
        for candidate_slot in (slot - 1, slot, slot + 1):
            if candidate_slot < 0:
                continue
            left_active = (
                left.start_time / left.time_step - _TOL
                <= candidate_slot
                < left.allocated_end_time / left.time_step + _TOL
            )
            right_active = (
                right.start_time / right.time_step - _TOL
                <= candidate_slot
                < right.allocated_end_time / right.time_step + _TOL
            )
            if left_active and right_active:
                candidates.append((resource_id, candidate_slot))
    return tuple(sorted(set(candidates)))


def find_continuous_conflicts(
    routes: Sequence[CrownTimedRoute],
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
) -> Tuple[CrownContinuousConflict, ...]:
    """Validate every overlapping cross-agent primitive pair in continuous time."""

    windows_by_agent = {
        route.agent_id: _operation_windows(route) for route in routes
    }
    conflicts = []
    ordered_routes = tuple(sorted(routes, key=lambda route: route.agent_id))
    for left_index, left_route in enumerate(ordered_routes):
        for right_route in ordered_routes[left_index + 1 :]:
            required_distance = max(
                graphs[left_route.agent_id].planning_distance,
                graphs[right_route.agent_id].planning_distance,
            )
            for left in windows_by_agent[left_route.agent_id]:
                for right in windows_by_agent[right_route.agent_id]:
                    overlap_start = max(left.start_time, right.start_time)
                    overlap_end = min(left.allocated_end_time, right.allocated_end_time)
                    if overlap_end < overlap_start + _TOL:
                        continue
                    distance, minimum_time = _minimum_distance(
                        left,
                        right,
                        overlap_start,
                        overlap_end,
                    )
                    if distance + _TOL >= required_distance:
                        continue
                    mapped = _mapped_resources(left, right, minimum_time)
                    if not mapped:
                        raise CrownResourceMappingError(
                            "continuous collision has no common conservative tube resource; "
                            "the resource grid/tube implementation is not complete"
                        )
                    conflicts.append(
                        CrownContinuousConflict(
                            left_agent=left.agent_id,
                            right_agent=right.agent_id,
                            left_operation_index=left.operation_index,
                            right_operation_index=right.operation_index,
                            start_time=overlap_start,
                            end_time=overlap_end,
                            minimum_time=minimum_time,
                            minimum_distance=distance,
                            required_distance=required_distance,
                            mapped_resources=mapped,
                        )
                    )
    return tuple(conflicts)


def assert_continuous_conflict_free(
    routes: Sequence[CrownTimedRoute],
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
) -> None:
    conflicts = find_continuous_conflicts(routes, graphs)
    if conflicts:
        first = conflicts[0]
        raise AssertionError(
            f"continuous conflict between {first.left_agent} and {first.right_agent}: "
            f"distance={first.minimum_distance:.6f}, required={first.required_distance:.6f}"
        )
