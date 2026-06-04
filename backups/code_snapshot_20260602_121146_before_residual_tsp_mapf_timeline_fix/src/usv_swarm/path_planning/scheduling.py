from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

from .types import AgentPathPlan, PathSegmentSpec, PathWaypoint


def apply_resource_window_schedule(
    agents: Dict[int, AgentPathPlan],
    separation_time: float = 0.0,
) -> int:
    """Resolve simple resource-window conflicts by delaying later segments.

    This is a lightweight CBS-compatible hook for path-planning segments. Each
    segment can expose a shared `metadata["resource_id"]` such as a narrow
    corridor, entrance, turning pocket, or coverage strip. If two agents would
    occupy the same resource at overlapping times, the later segment and all
    subsequent segments of that agent are shifted by the necessary delay.
    """

    original: List[Tuple[float, int, int, PathSegmentSpec, List[float | None]]] = []
    for agent_id, agent in agents.items():
        for idx, segment in enumerate(agent.segments):
            times = [waypoint.time for waypoint in segment.waypoints]
            start, _ = _segment_time_bounds_from_times(times)
            original.append((start, agent_id, idx, segment, times))
    original.sort(key=lambda item: (item[0], item[1], item[2]))

    shift_by_agent: Dict[int, float] = {agent_id: 0.0 for agent_id in agents}
    resource_end: Dict[str, float] = {}
    conflicts = 0
    for _, agent_id, _, segment, times in original:
        resource_id = segment.metadata.get("resource_id")
        start, end = _segment_time_bounds_from_times(times)
        duration = max(end - start, 0.0)
        offset = shift_by_agent[agent_id]
        shifted_start = start + offset
        shifted_end = shifted_start + duration
        if resource_id:
            reserved_until = resource_end.get(resource_id)
            if reserved_until is not None and shifted_start < reserved_until - 1e-9:
                delay = reserved_until - shifted_start + max(separation_time, 0.0)
                offset += delay
                shift_by_agent[agent_id] = offset
                shifted_start += delay
                shifted_end += delay
                conflicts += 1
            resource_end[resource_id] = max(resource_end.get(resource_id, shifted_end), shifted_end)
        _retime_segment(segment, times, offset)

    for agent in agents.values():
        agent.metrics["mapf_conflicts_resolved"] = float(conflicts)
        agent.metrics["scheduled_estimated_time"] = max(
            (_segment_time_bounds(segment)[1] for segment in agent.segments),
            default=0.0,
        )
    return conflicts


def _retime_segment(segment: PathSegmentSpec, original_times: List[float | None], offset: float) -> None:
    retimed: List[PathWaypoint] = []
    for waypoint, original_time in zip(segment.waypoints, original_times):
        retimed.append(replace(waypoint, time=None if original_time is None else float(original_time) + offset))
    segment.waypoints = retimed


def _segment_time_bounds(segment: PathSegmentSpec) -> Tuple[float, float]:
    return _segment_time_bounds_from_times([waypoint.time for waypoint in segment.waypoints])


def _segment_time_bounds_from_times(times: List[float | None]) -> Tuple[float, float]:
    numeric = [float(time) for time in times if time is not None]
    if not numeric:
        return (0.0, 0.0)
    return (min(numeric), max(numeric))
