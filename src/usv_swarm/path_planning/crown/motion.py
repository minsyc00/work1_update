"""Nonholonomic motion primitives, current-aware costs, and conservative tubes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, ceil, floor, hypot, isfinite, radians, sqrt
from typing import Mapping, Optional, Protocol, Sequence, Tuple

from ...geometry import wrap_angle
from ...schema import AgentPlanningProfile, Pose2D
from ..types import PathSegmentSpec, PathWaypoint
from .config import CrownMcppConfig
from .types import CrownOperation


Point = Tuple[float, float]
Cell = Tuple[int, int]


class CrownCurrentField(Protocol):
    """Known deterministic current used by service and connector costing."""

    def velocity(self, x: float, y: float, time: float) -> Point:
        ...


@dataclass(frozen=True)
class ZeroCurrentField:
    time_invariant: bool = True

    def velocity(self, x: float, y: float, time: float) -> Point:
        del x, y, time
        return (0.0, 0.0)


@dataclass(frozen=True)
class UniformCurrentField:
    vx: float = 0.0
    vy: float = 0.0
    time_invariant: bool = True

    def velocity(self, x: float, y: float, time: float) -> Point:
        del x, y, time
        return (self.vx, self.vy)


class CurrentInfeasibleError(ValueError):
    pass


@dataclass(frozen=True)
class CrownMotionPrimitive:
    primitive_id: str
    agent_id: str
    kind: str
    start_pose: Pose2D
    end_pose: Pose2D
    duration: float
    energy: float
    resource_ids: Tuple[str, ...]
    segment: PathSegmentSpec
    task_id: Optional[str] = None
    mode_id: Optional[str] = None
    metadata: Mapping[str, str] = None

    def to_operation(self) -> CrownOperation:
        return CrownOperation(
            operation_id=self.primitive_id,
            duration=self.duration,
            resource_ids=self.resource_ids,
            energy=self.energy,
            kind=self.kind,
            metadata={
                "start_x": f"{self.start_pose.x:.12g}",
                "start_y": f"{self.start_pose.y:.12g}",
                "end_x": f"{self.end_pose.x:.12g}",
                "end_y": f"{self.end_pose.y:.12g}",
                "task_id": self.task_id or "",
                "mode_id": self.mode_id or "",
                **dict(self.metadata or {}),
            },
        )


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1.0e-18:
        return hypot(point[0] - start[0], point[1] - start[1])
    alpha = max(
        0.0,
        min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator),
    )
    projection = (start[0] + alpha * dx, start[1] + alpha * dy)
    return hypot(point[0] - projection[0], point[1] - projection[1])


def conservative_tube_cells(
    start: Point,
    end: Point,
    *,
    radius: float,
    grid_size: float,
) -> Tuple[Cell, ...]:
    """Rasterize every square cell that may intersect a swept center disk."""

    if radius < 0.0 or grid_size <= 0.0:
        raise ValueError("tube radius must be non-negative and grid_size positive")
    half_diagonal = grid_size / sqrt(2.0)
    threshold = radius + half_diagonal
    x_min = floor((min(start[0], end[0]) - threshold) / grid_size)
    x_max = ceil((max(start[0], end[0]) + threshold) / grid_size)
    y_min = floor((min(start[1], end[1]) - threshold) / grid_size)
    y_max = ceil((max(start[1], end[1]) + threshold) / grid_size)
    cells = []
    for ix in range(x_min, x_max + 1):
        for iy in range(y_min, y_max + 1):
            center = ((ix + 0.5) * grid_size, (iy + 0.5) * grid_size)
            if _point_segment_distance(center, start, end) <= threshold + 1.0e-12:
                cells.append((ix, iy))
    return tuple(cells)


def _commanded_speed(kind: str, profile: AgentPlanningProfile) -> float:
    if kind == "cover":
        return profile.cover_speed
    if kind == "turn":
        return profile.turn_speed_max
    return profile.cruise_speed


def _power(kind: str, profile: AgentPlanningProfile) -> float:
    if kind == "cover":
        return profile.cover_power
    if kind == "turn":
        return profile.turn_power
    return profile.transit_power


def _pair_duration(
    start: PathWaypoint,
    end: PathWaypoint,
    *,
    kind: str,
    profile: AgentPlanningProfile,
    current_field: CrownCurrentField,
    absolute_time: float,
) -> float:
    distance = hypot(end.x - start.x, end.y - start.y)
    heading_change = abs(wrap_angle(end.psi - start.psi))
    yaw_time = heading_change / max(profile.yaw_rate_limit, 1.0e-9)
    original_time = (
        max(0.0, float(end.time) - float(start.time))
        if start.time is not None and end.time is not None
        else 0.0
    )
    if distance <= 1.0e-12:
        return max(yaw_time, original_time, 1.0e-6)

    unit = ((end.x - start.x) / distance, (end.y - start.y) / distance)
    midpoint = ((start.x + end.x) * 0.5, (start.y + end.y) * 0.5)
    current = current_field.velocity(midpoint[0], midpoint[1], absolute_time)
    along = current[0] * unit[0] + current[1] * unit[1]
    perpendicular_x = current[0] - along * unit[0]
    perpendicular_y = current[1] - along * unit[1]
    perpendicular_squared = perpendicular_x**2 + perpendicular_y**2
    commanded = _commanded_speed(kind, profile)
    if perpendicular_squared >= commanded**2 - 1.0e-12:
        raise CurrentInfeasibleError("cross-current exceeds commanded through-water speed")
    ground_speed = along + sqrt(max(commanded**2 - perpendicular_squared, 0.0))
    if ground_speed <= 1.0e-9:
        raise CurrentInfeasibleError("adverse current makes the primitive unreachable")
    return max(distance / ground_speed, yaw_time, 1.0e-6)


def _collapse_collinear_waypoints(
    waypoints: Sequence[PathWaypoint],
) -> Tuple[PathWaypoint, ...]:
    """Remove rendering samples that do not define a new motion primitive."""

    if len(waypoints) <= 2:
        return tuple(waypoints)
    result = [waypoints[0]]
    for index in range(1, len(waypoints) - 1):
        previous = result[-1]
        current = waypoints[index]
        following = waypoints[index + 1]
        dx = following.x - previous.x
        dy = following.y - previous.y
        chord = hypot(dx, dy)
        if chord <= 1.0e-12:
            result.append(current)
            continue
        cross_distance = abs(
            dx * (previous.y - current.y)
            - (previous.x - current.x) * dy
        ) / chord
        chord_heading = atan2(dy, dx)
        heading_error = max(
            abs(wrap_angle(previous.psi - chord_heading)),
            abs(wrap_angle(current.psi - chord_heading)),
            abs(wrap_angle(following.psi - chord_heading)),
        )
        if cross_distance <= 1.0e-8 and heading_error <= 1.0e-6:
            continue
        result.append(current)
    result.append(waypoints[-1])
    return tuple(result)


def segment_to_motion_primitives(
    segment: PathSegmentSpec,
    *,
    agent_id: str,
    profile: AgentPlanningProfile,
    crown_config: CrownMcppConfig,
    planning_distance: float,
    current_field: Optional[CrownCurrentField] = None,
    primitive_prefix: str,
    task_id: Optional[str] = None,
    mode_id: Optional[str] = None,
    absolute_start_time: float = 0.0,
) -> Tuple[CrownMotionPrimitive, ...]:
    """Split a path into short, current-aware primitives with swept tubes."""

    if len(segment.waypoints) < 2:
        return ()
    field = current_field or ZeroCurrentField()
    primitives = []
    running_time = absolute_start_time
    primitive_index = 0
    motion_waypoints = _collapse_collinear_waypoints(segment.waypoints)
    motion_pairs = tuple(zip(motion_waypoints[:-1], motion_waypoints[1:]))
    segment_turn_angle = sum(
        abs(wrap_angle(end.psi - start.psi)) for start, end in motion_pairs
    )
    is_turn_maneuver = segment_turn_angle > radians(5.0)
    for source_index, (start, end) in enumerate(motion_pairs):
        pair_turn_angle = abs(wrap_angle(end.psi - start.psi))
        kinematic_duration = _pair_duration(
            start,
            end,
            kind=segment.kind,
            profile=profile,
            current_field=field,
            absolute_time=running_time,
        )
        # The base kinematic duration lets translation and yaw happen in
        # parallel.  Real USVs additionally pay for slowing, steering and
        # settling.  Add that operational penalty once here; retimed
        # primitives preserve it through their original duration.
        subdivision_count = max(
            1,
            int(ceil(kinematic_duration / crown_config.primitive_max_duration)),
        )
        for subdivision in range(subdivision_count):
            alpha0 = subdivision / subdivision_count
            alpha1 = (subdivision + 1) / subdivision_count
            delta_heading = wrap_angle(end.psi - start.psi)
            start_pose = Pose2D(
                start.x + alpha0 * (end.x - start.x),
                start.y + alpha0 * (end.y - start.y),
                wrap_angle(start.psi + alpha0 * delta_heading),
            )
            end_pose = Pose2D(
                start.x + alpha1 * (end.x - start.x),
                start.y + alpha1 * (end.y - start.y),
                wrap_angle(start.psi + alpha1 * delta_heading),
            )
            duration = (
                kinematic_duration / subdivision_count
                + profile.turn_time_penalty_per_rad
                * pair_turn_angle
                / subdivision_count
            )
            maneuver_start = (
                is_turn_maneuver and source_index == 0 and subdivision == 0
            )
            if maneuver_start:
                duration += profile.turn_maneuver_time_penalty
            primitive_turn_angle = pair_turn_angle / subdivision_count
            cells = conservative_tube_cells(
                (start_pose.x, start_pose.y),
                (end_pose.x, end_pose.y),
                radius=planning_distance / 2.0,
                grid_size=crown_config.resource_grid_size,
            )
            resource_ids = tuple(
                f"tube:{ix}:{iy}" for ix, iy in cells
            )
            primitive_segment = PathSegmentSpec(
                segment_id=f"{segment.segment_id}:primitive:{primitive_index}",
                kind=segment.kind,
                source_algorithm="crown_motion_primitive",
                waypoints=[
                    PathWaypoint(start_pose.x, start_pose.y, start_pose.psi, time=0.0),
                    PathWaypoint(end_pose.x, end_pose.y, end_pose.psi, time=duration),
                ],
                curvature_max=segment.curvature_max,
                length=hypot(end_pose.x - start_pose.x, end_pose.y - start_pose.y),
                path_source=segment.path_source,
                metadata={**dict(segment.metadata), "source_segment_id": segment.segment_id},
            )
            energy = (
                _power(segment.kind, profile) * duration
                + profile.turn_energy_penalty_per_rad * primitive_turn_angle
                + (
                    profile.turn_maneuver_energy_penalty
                    if maneuver_start
                    else 0.0
                )
            )
            primitives.append(
                CrownMotionPrimitive(
                    primitive_id=f"{primitive_prefix}:{source_index}:{subdivision}",
                    agent_id=agent_id,
                    kind=segment.kind,
                    start_pose=start_pose,
                    end_pose=end_pose,
                    duration=duration,
                    energy=energy,
                    resource_ids=resource_ids,
                    segment=primitive_segment,
                    task_id=task_id,
                    mode_id=mode_id,
                    metadata={
                        "source_segment_id": segment.segment_id,
                        "tube_cell_count": str(len(cells)),
                        "turn_angle_rad": f"{primitive_turn_angle:.12g}",
                        "turn_time_penalty": (
                            f"{profile.turn_time_penalty_per_rad * primitive_turn_angle:.12g}"
                        ),
                        "turn_energy_penalty": (
                            f"{profile.turn_energy_penalty_per_rad * primitive_turn_angle:.12g}"
                        ),
                        "turn_maneuver_start": str(maneuver_start).lower(),
                        "turn_maneuver_time_penalty": (
                            f"{profile.turn_maneuver_time_penalty if maneuver_start else 0.0:.12g}"
                        ),
                        "turn_maneuver_energy_penalty": (
                            f"{profile.turn_maneuver_energy_penalty if maneuver_start else 0.0:.12g}"
                        ),
                    },
                )
            )
            primitive_index += 1
            running_time += duration
    return tuple(primitives)


def primitives_to_operations(
    primitives: Sequence[CrownMotionPrimitive],
) -> Tuple[CrownOperation, ...]:
    return tuple(primitive.to_operation() for primitive in primitives)


def build_wait_primitive(
    *,
    primitive_id: str,
    agent_id: str,
    pose: Pose2D,
    duration: float,
    profile: AgentPlanningProfile,
    crown_config: CrownMcppConfig,
    planning_distance: float,
) -> CrownMotionPrimitive:
    """Build a stationary graph edge with a conservative occupancy tube."""

    if duration <= 0.0:
        raise ValueError("wait duration must be positive")
    cells = conservative_tube_cells(
        (pose.x, pose.y),
        (pose.x, pose.y),
        radius=planning_distance / 2.0,
        grid_size=crown_config.resource_grid_size,
    )
    segment = PathSegmentSpec(
        segment_id=primitive_id,
        kind="wait",
        source_algorithm="crown_wait_primitive",
        waypoints=[
            PathWaypoint(pose.x, pose.y, pose.psi, time=0.0, speed=0.0),
            PathWaypoint(pose.x, pose.y, pose.psi, time=duration, speed=0.0),
        ],
        length=0.0,
        path_source="time_expanded_wait_edge",
        metadata={"stationary": "true"},
    )
    return CrownMotionPrimitive(
        primitive_id=primitive_id,
        agent_id=agent_id,
        kind="wait",
        start_pose=pose,
        end_pose=pose,
        duration=duration,
        energy=profile.wait_power * duration,
        resource_ids=tuple(f"tube:{ix}:{iy}" for ix, iy in cells),
        segment=segment,
        metadata={"stationary": "true", "tube_cell_count": str(len(cells))},
    )


def retime_motion_primitive(
    primitive: CrownMotionPrimitive,
    *,
    profile: AgentPlanningProfile,
    current_field: CrownCurrentField,
    absolute_start_time: float,
) -> CrownMotionPrimitive:
    """Re-evaluate one fixed-geometry primitive in a time-varying current."""

    start = primitive.segment.waypoints[0]
    end = primitive.segment.waypoints[-1]
    duration = _pair_duration(
        start,
        end,
        kind=primitive.kind,
        profile=profile,
        current_field=current_field,
        absolute_time=absolute_start_time,
    )
    turn_angle = abs(wrap_angle(end.psi - start.psi))
    maneuver_start = (
        str((primitive.metadata or {}).get("turn_maneuver_start", "false")).lower()
        == "true"
    )
    segment = replace(
        primitive.segment,
        waypoints=[
            replace(start, time=0.0),
            replace(end, time=duration),
        ],
        metadata=dict(primitive.segment.metadata),
    )
    return replace(
        primitive,
        duration=duration,
        energy=(
            _power(primitive.kind, profile) * duration
            + profile.turn_energy_penalty_per_rad * turn_angle
            + (
                profile.turn_maneuver_energy_penalty
                if maneuver_start
                else 0.0
            )
        ),
        segment=segment,
    )
