from __future__ import annotations

from typing import Dict, Optional, Sequence

from ...geometry import polyline_length, wrap_angle
from ...schema import (
    DynamicObstacleTrack,
    PlannerConfig,
    PlanningResult,
    Pose2D,
    SmoothedPath,
    TimedPathSegment,
    TrajectoryReference,
    TrajectorySample,
)
from ..types import (
    AgentPathPlan,
    MultiAgentPathPlan,
    PaperReference,
    PathPlanningRequest,
    PathSegmentSpec,
    PathWaypoint,
    StaticObstacle,
)


def build_request_from_planning_result(
    config: PlannerConfig,
    planning_result: PlanningResult,
    static_obstacles: Optional[Sequence[StaticObstacle]] = None,
    dynamic_obstacles: Optional[Sequence[DynamicObstacleTrack]] = None,
    paper_references: Optional[Sequence[PaperReference]] = None,
) -> PathPlanningRequest:
    return PathPlanningRequest(
        config=config,
        path_config=None,
        strips=list(planning_result.strips),
        assignments=planning_result.assignments,
        static_obstacles=list(static_obstacles or []),
        dynamic_obstacles=list(dynamic_obstacles or []),
        existing_plan=planning_result,
        paper_references=list(paper_references or []),
        metadata={
            "bootstrap_source": "usv_swarm.planning.plan_global_coverage",
            "path_count": str(len(planning_result.paths)),
        },
    )


def planning_result_to_path_plan(
    planning_result: PlanningResult,
    algorithm_name: str,
    paper_references: Optional[Sequence[PaperReference]] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> MultiAgentPathPlan:
    agents: Dict[int, AgentPathPlan] = {}
    for agent_id, smoothed_path in planning_result.paths.items():
        segments = []
        for segment_index, segment in enumerate(smoothed_path.segments):
            waypoints = [
                PathWaypoint(
                    x=point[0],
                    y=point[1],
                    psi=heading,
                )
                for point, heading in zip(segment.points, segment.headings)
            ]
            segments.append(
                PathSegmentSpec(
                    segment_id=f"agent{agent_id}_segment{segment_index}",
                    kind=segment.segment_type,
                    source_algorithm=algorithm_name,
                    waypoints=waypoints,
                    control_points=list(segment.control_points or []),
                    curvature_max=segment.max_curvature,
                    length=segment.length,
                    path_source=segment.path_source,
                    metadata={
                        "dubins_modes": "" if segment.dubins_modes is None else "-".join(segment.dubins_modes),
                    },
                )
            )
        agents[agent_id] = AgentPathPlan(
            agent_id=agent_id,
            source_algorithm=algorithm_name,
            segments=segments,
            metrics={
                "total_length": smoothed_path.total_length,
                "max_curvature": smoothed_path.max_curvature,
                "segment_count": float(len(segments)),
            },
            paper_references=list(paper_references or []),
        )
    return MultiAgentPathPlan(
        algorithm_name=algorithm_name,
        agents=agents,
        metadata=dict(metadata or {}),
        paper_references=list(paper_references or []),
    )


def path_plan_to_smoothed_paths(path_plan: MultiAgentPathPlan) -> Dict[int, SmoothedPath]:
    paths: Dict[int, SmoothedPath] = {}
    for agent_id, agent_plan in path_plan.agents.items():
        segments = []
        total_length = 0.0
        max_curvature = 0.0
        for segment in agent_plan.segments:
            if not segment.waypoints:
                continue
            points = [(waypoint.x, waypoint.y) for waypoint in segment.waypoints]
            headings = [waypoint.psi for waypoint in segment.waypoints]
            start_waypoint = segment.waypoints[0]
            end_waypoint = segment.waypoints[-1]
            timed_segment = TimedPathSegment(
                segment_type=segment.kind,
                start_time=float(start_waypoint.time or 0.0),
                end_time=float(end_waypoint.time or start_waypoint.time or 0.0),
                start_pose=Pose2D(start_waypoint.x, start_waypoint.y, start_waypoint.psi),
                end_pose=Pose2D(end_waypoint.x, end_waypoint.y, end_waypoint.psi),
                points=points,
                headings=headings,
                control_points=list(segment.control_points or []),
                max_curvature=segment.curvature_max,
                length=segment.length if segment.length > 0.0 else polyline_length(points),
                path_source=segment.path_source,
                dubins_modes=_decode_dubins_modes(segment.metadata.get("dubins_modes", "")),
            )
            segments.append(timed_segment)
            total_length += timed_segment.length
            max_curvature = max(max_curvature, timed_segment.max_curvature)
        paths[agent_id] = SmoothedPath(
            agent_id=agent_id,
            segments=segments,
            total_length=total_length,
            max_curvature=max_curvature,
        )
    return paths


def path_plan_to_trajectory_references(path_plan: MultiAgentPathPlan) -> Dict[int, TrajectoryReference]:
    refs: Dict[int, TrajectoryReference] = {}
    for agent_id, smoothed_path in path_plan_to_smoothed_paths(path_plan).items():
        samples = []
        for segment in smoothed_path.segments:
            for idx, ((x, y), psi) in enumerate(zip(segment.points, segment.headings)):
                if samples and idx == 0:
                    continue
                count = max(len(segment.points), 2)
                alpha = idx / max(count - 1, 1)
                time = segment.start_time + alpha * max(segment.end_time - segment.start_time, 0.0)
                if idx == 0:
                    yaw_rate = 0.0
                else:
                    prev_heading = segment.headings[idx - 1]
                    prev_alpha = (idx - 1) / max(count - 1, 1)
                    prev_time = segment.start_time + prev_alpha * max(segment.end_time - segment.start_time, 0.0)
                    yaw_rate = wrap_angle(psi - prev_heading) / max(time - prev_time, 1e-9)
                speed = segment.length / max(segment.end_time - segment.start_time, 1e-9)
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
        refs[agent_id] = TrajectoryReference(
            agent_id=agent_id,
            samples=samples,
            horizon_time=samples[-1].time if samples else 0.0,
        )
    return refs


def _decode_dubins_modes(value: str) -> Optional[tuple[str, str, str]]:
    if not value:
        return None
    parts = tuple(part for part in value.split("-") if part)
    if len(parts) != 3:
        return None
    return parts  # type: ignore[return-value]
