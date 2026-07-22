from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Sequence, Tuple

from ..geometry import polyline_length, wrap_angle
from ..schema import PlannerConfig
from .obstacles import path_segment_invalid_reasons
from .types import ObstacleField, PathSegmentSpec, PathWaypoint


_MASS_U = 1.0
_MASS_R = 1.0
_DAMP_U = 0.8
_DAMP_R = 0.9


@dataclass(frozen=True)
class TransitionDynamicsReport:
    valid: bool
    reasons: Tuple[str, ...]
    max_curvature: float = 0.0
    max_heading_jump: float = 0.0
    max_heading_error: float = 0.0
    max_yaw_rate: float = 0.0
    max_yaw_acceleration: float = 0.0
    max_speed: float = 0.0
    max_acceleration: float = 0.0
    max_thrust_required: float = 0.0
    max_yaw_moment_required: float = 0.0
    nmpc_trackable: bool = True


def validate_transition_sequence(
    segments: Sequence[PathSegmentSpec],
    config: PlannerConfig,
    obstacle_field: ObstacleField | None = None,
    retime: bool = False,
) -> TransitionDynamicsReport:
    reports: List[TransitionDynamicsReport] = []
    reasons: List[str] = []
    max_boundary_heading_jump = 0.0
    previous: PathWaypoint | None = None
    for segment in segments:
        report = validate_transition_dynamics(segment, config, obstacle_field=obstacle_field, retime=retime)
        reports.append(report)
        if not report.valid:
            reasons.extend(report.reasons)
        if previous is not None and segment.waypoints:
            first = segment.waypoints[0]
            pose_gap = math.hypot(first.x - previous.x, first.y - previous.y)
            heading_jump = abs(wrap_angle(first.psi - previous.psi))
            max_boundary_heading_jump = max(max_boundary_heading_jump, heading_jump)
            if pose_gap > 1e-3:
                reasons.append("segment_pose_gap")
            if heading_jump > 0.35:
                reasons.append("segment_heading_jump")
        if segment.waypoints:
            previous = segment.waypoints[-1]

    merged = _merge_reports(reports, reasons)
    max_heading_jump = max(merged.max_heading_jump, max_boundary_heading_jump)
    valid = merged.valid and max_boundary_heading_jump <= 0.35 + 1e-9
    if not valid and max_boundary_heading_jump > 0.35:
        reasons = list(merged.reasons)
    else:
        reasons = list(merged.reasons)
    return replace(
        merged,
        valid=valid,
        reasons=tuple(sorted(set(reasons))),
        max_heading_jump=max_heading_jump,
        nmpc_trackable=valid,
    )


def validate_transition_dynamics(
    segment: PathSegmentSpec,
    config: PlannerConfig,
    obstacle_field: ObstacleField | None = None,
    retime: bool = False,
) -> TransitionDynamicsReport:
    if retime:
        retime_segment_for_dynamics(segment, config)
    report = _compute_segment_report(segment, config, obstacle_field)
    _annotate_segment_with_report(segment, report)
    return report


def retime_segment_for_dynamics(segment: PathSegmentSpec, config: PlannerConfig) -> None:
    if len(segment.waypoints) < 2 or segment.length <= 1e-9:
        return
    curvature = max(segment.curvature_max, _sampled_curvature(segment.waypoints))
    speed_limit = _dynamic_speed_limit(curvature, config, segment.kind)
    existing_speed = max((waypoint.speed or 0.0 for waypoint in segment.waypoints), default=0.0)
    if existing_speed <= 1e-9:
        existing_speed = config.fleet.turn_speed_max if segment.kind == "turn" else config.fleet.cruise_speed
    target_speed = min(existing_speed, speed_limit)
    if target_speed <= 1e-6:
        target_speed = 1e-6
    duration = segment.length / target_speed
    start_time = segment.waypoints[0].time or 0.0
    cumulative = _cumulative_distances(segment.waypoints)
    total = max(cumulative[-1], 1e-9)
    segment.waypoints = [
        replace(
            waypoint,
            time=start_time + duration * distance / total,
            speed=target_speed,
        )
        for waypoint, distance in zip(segment.waypoints, cumulative)
    ]


def dynamic_edge_cost(segments: Sequence[PathSegmentSpec], config: PlannerConfig) -> float:
    report = validate_transition_sequence(segments, config, retime=False)
    duration = sum(_segment_duration(segment) for segment in segments)
    control_cost = report.max_thrust_required + report.max_yaw_moment_required
    turn_cost = sum(_segment_heading_variation(segment) for segment in segments)
    return duration + 0.2 * turn_cost + 0.1 * control_cost


def _compute_segment_report(
    segment: PathSegmentSpec,
    config: PlannerConfig,
    obstacle_field: ObstacleField | None,
) -> TransitionDynamicsReport:
    reasons = list(path_segment_invalid_reasons(segment, config, obstacle_field))
    if segment.path_source == "astar_corridor_edge":
        reasons.append("raw_astar_corridor_edge")
    if segment.metadata.get("kinematic_feasible") == "false":
        reasons.append("kinematic_infeasible")

    waypoints = segment.waypoints
    if len(waypoints) < 2:
        report = TransitionDynamicsReport(valid=not reasons, reasons=tuple(sorted(set(reasons))))
        return report

    max_curvature = _effective_curvature(segment)
    curvature_limit = 1.0 / max(config.fleet.min_turn_radius, 1e-6)
    if max_curvature > curvature_limit + 1e-3:
        reasons.append("curvature_exceeded")

    max_heading_jump = 0.0
    max_heading_error = 0.0
    max_yaw_rate = 0.0
    max_yaw_acceleration = 0.0
    max_speed = 0.0
    max_acceleration = 0.0
    max_thrust = 0.0
    max_yaw_moment = 0.0
    max_steady_yaw_moment = 0.0
    previous_speed = waypoints[0].speed or _interval_speed(waypoints[0], waypoints[1])
    previous_r = 0.0
    previous_time = waypoints[0].time or 0.0
    control_dt = 1.0 / max(config.mission.local_control_hz, 1e-6)

    for idx, (first, second) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        dt = _interval_dt(first, second)
        distance = math.hypot(second.x - first.x, second.y - first.y)
        speed = _interval_speed(first, second)
        dpsi = wrap_angle(second.psi - first.psi)
        r_ref = dpsi / dt
        max_heading_jump = max(max_heading_jump, abs(dpsi))
        max_yaw_rate = max(max_yaw_rate, abs(r_ref))
        max_steady_yaw_moment = max(max_steady_yaw_moment, abs(_DAMP_R * r_ref))
        max_speed = max(max_speed, abs(speed), abs(first.speed or 0.0), abs(second.speed or 0.0))
        if distance > 1e-6:
            motion_heading = math.atan2(second.y - first.y, second.x - first.x)
            avg_heading = first.psi + 0.5 * dpsi
            max_heading_error = max(max_heading_error, abs(wrap_angle(avg_heading - motion_heading)))
        if idx > 0:
            elapsed = max((first.time or previous_time) - previous_time, dt, control_dt)
            acceleration = (speed - previous_speed) / max(elapsed, 1e-6)
            yaw_acceleration = (r_ref - previous_r) / max(elapsed, 1e-6)
            max_acceleration = max(max_acceleration, abs(acceleration))
            max_yaw_acceleration = max(max_yaw_acceleration, abs(yaw_acceleration))
            max_thrust = max(max_thrust, abs(_MASS_U * acceleration + _DAMP_U * speed))
            max_yaw_moment = max(max_yaw_moment, abs(_MASS_R * yaw_acceleration + _DAMP_R * r_ref))
        else:
            max_thrust = max(max_thrust, abs(_DAMP_U * speed))
            max_yaw_moment = max(max_yaw_moment, abs(_DAMP_R * r_ref))
        previous_speed = speed
        previous_r = r_ref
        previous_time = second.time or (previous_time + dt)

    speed_limit = _dynamic_speed_limit(max_curvature, config, segment.kind)
    yaw_rate_limit = _yaw_rate_limit(config)
    accel_limit = max(config.fleet.max_thrust / _MASS_U, 1e-6)
    heading_error_limit = 0.45
    heading_jump_limit = 0.80

    if max_speed > speed_limit + 1e-3:
        reasons.append("speed_exceeded")
    if max_yaw_rate > yaw_rate_limit + 1e-3:
        reasons.append("yaw_rate_exceeded")
    if max_acceleration > accel_limit + 1e-3:
        reasons.append("acceleration_exceeded")
    if max_thrust > config.fleet.max_thrust + 1e-3:
        reasons.append("thrust_exceeded")
    if max_steady_yaw_moment > config.fleet.max_yaw_moment + 1e-3:
        reasons.append("yaw_moment_exceeded")
    if max_heading_error > heading_error_limit + 1e-3:
        reasons.append("heading_tangent_mismatch")
    if max_heading_jump > heading_jump_limit + 1e-3:
        reasons.append("heading_jump")

    valid = not reasons
    return TransitionDynamicsReport(
        valid=valid,
        reasons=tuple(sorted(set(reasons))),
        max_curvature=max_curvature,
        max_heading_jump=max_heading_jump,
        max_heading_error=max_heading_error,
        max_yaw_rate=max_yaw_rate,
        max_yaw_acceleration=max_yaw_acceleration,
        max_speed=max_speed,
        max_acceleration=max_acceleration,
        max_thrust_required=max_thrust,
        max_yaw_moment_required=max_yaw_moment,
        nmpc_trackable=valid,
    )


def _annotate_segment_with_report(segment: PathSegmentSpec, report: TransitionDynamicsReport) -> None:
    segment.metadata["curvature_feasible"] = str("curvature_exceeded" not in report.reasons).lower()
    segment.metadata["kinematic_feasible"] = str(
        "raw_astar_corridor_edge" not in report.reasons and "kinematic_infeasible" not in report.reasons
    ).lower()
    segment.metadata["dynamic_feasible"] = str(report.valid).lower()
    segment.metadata["nmpc_trackable"] = "proxy_pass" if report.valid else "proxy_fail"
    segment.metadata["dynamic_validation"] = "3dof_rollout_proxy"
    segment.metadata["max_curvature_checked"] = f"{report.max_curvature:.6f}"
    segment.metadata["max_heading_jump"] = f"{report.max_heading_jump:.6f}"
    segment.metadata["max_heading_error"] = f"{report.max_heading_error:.6f}"
    segment.metadata["max_yaw_rate"] = f"{report.max_yaw_rate:.6f}"
    segment.metadata["max_yaw_acceleration"] = f"{report.max_yaw_acceleration:.6f}"
    segment.metadata["max_speed"] = f"{report.max_speed:.6f}"
    segment.metadata["max_acceleration"] = f"{report.max_acceleration:.6f}"
    segment.metadata["max_thrust_required"] = f"{report.max_thrust_required:.6f}"
    segment.metadata["max_yaw_moment_required"] = f"{report.max_yaw_moment_required:.6f}"
    if report.reasons:
        segment.metadata["dynamic_invalid_reasons"] = ",".join(report.reasons)
    else:
        segment.metadata.pop("dynamic_invalid_reasons", None)


def _merge_reports(reports: Sequence[TransitionDynamicsReport], extra_reasons: Sequence[str]) -> TransitionDynamicsReport:
    reasons = list(extra_reasons)
    for report in reports:
        reasons.extend(report.reasons)
    return TransitionDynamicsReport(
        valid=not reasons,
        reasons=tuple(sorted(set(reasons))),
        max_curvature=max((report.max_curvature for report in reports), default=0.0),
        max_heading_jump=max((report.max_heading_jump for report in reports), default=0.0),
        max_heading_error=max((report.max_heading_error for report in reports), default=0.0),
        max_yaw_rate=max((report.max_yaw_rate for report in reports), default=0.0),
        max_yaw_acceleration=max((report.max_yaw_acceleration for report in reports), default=0.0),
        max_speed=max((report.max_speed for report in reports), default=0.0),
        max_acceleration=max((report.max_acceleration for report in reports), default=0.0),
        max_thrust_required=max((report.max_thrust_required for report in reports), default=0.0),
        max_yaw_moment_required=max((report.max_yaw_moment_required for report in reports), default=0.0),
        nmpc_trackable=not reasons,
    )


def _dynamic_speed_limit(curvature: float, config: PlannerConfig, kind: str) -> float:
    base_limit = config.fleet.turn_speed_max if kind == "turn" else config.fleet.cruise_speed
    base_limit = min(base_limit, config.fleet.cruise_speed)
    thrust_speed_limit = config.fleet.max_thrust / _DAMP_U if _DAMP_U > 1e-9 else base_limit
    if curvature <= 1e-9:
        return max(min(base_limit, thrust_speed_limit), 1e-6)
    yaw_rate_speed_limit = _yaw_rate_limit(config) / curvature
    yaw_moment_speed_limit = (config.fleet.max_yaw_moment / _DAMP_R) / curvature if _DAMP_R > 1e-9 else base_limit
    return max(min(base_limit, thrust_speed_limit, yaw_rate_speed_limit, yaw_moment_speed_limit), 1e-6)


def _yaw_rate_limit(config: PlannerConfig) -> float:
    radius = max(config.fleet.min_turn_radius, 1e-6)
    turn_speed_limit = max(config.fleet.turn_speed_max, 1e-6) / radius
    moment_steady_limit = config.fleet.max_yaw_moment / _DAMP_R if _DAMP_R > 1e-9 else turn_speed_limit
    return max(min(turn_speed_limit, moment_steady_limit), 1e-6)


def _sampled_curvature(waypoints: Sequence[PathWaypoint]) -> float:
    max_curvature = 0.0
    for first, second in zip(waypoints[:-1], waypoints[1:]):
        ds = math.hypot(second.x - first.x, second.y - first.y)
        if ds <= 1e-9:
            continue
        max_curvature = max(max_curvature, abs(wrap_angle(second.psi - first.psi)) / ds)
    return max_curvature


def _effective_curvature(segment: PathSegmentSpec) -> float:
    trusted_sources = {
        "bezier",
        "dubins_fallback",
        "motion_lattice",
        "smoothed_astar_corridor",
        "stationary",
        "straight",
    }
    if segment.path_source in trusted_sources:
        return max(segment.curvature_max, 0.0)
    return max(segment.curvature_max, _sampled_curvature(segment.waypoints))


def _interval_dt(first: PathWaypoint, second: PathWaypoint) -> float:
    if first.time is not None and second.time is not None and second.time > first.time:
        return second.time - first.time
    speed = second.speed or first.speed or 1.0
    distance = math.hypot(second.x - first.x, second.y - first.y)
    return max(distance / max(abs(speed), 1e-6), 1e-6)


def _interval_speed(first: PathWaypoint, second: PathWaypoint) -> float:
    if first.speed is not None and second.speed is not None:
        return 0.5 * (first.speed + second.speed)
    distance = math.hypot(second.x - first.x, second.y - first.y)
    return distance / _interval_dt(first, second)


def _cumulative_distances(waypoints: Sequence[PathWaypoint]) -> List[float]:
    distances = [0.0]
    for first, second in zip(waypoints[:-1], waypoints[1:]):
        distances.append(distances[-1] + math.hypot(second.x - first.x, second.y - first.y))
    return distances


def _segment_duration(segment: PathSegmentSpec) -> float:
    if len(segment.waypoints) < 2:
        return 0.0
    start = segment.waypoints[0].time or 0.0
    end = segment.waypoints[-1].time or start
    if end > start:
        return end - start
    return polyline_length([(waypoint.x, waypoint.y) for waypoint in segment.waypoints]) / max(
        segment.waypoints[0].speed or 1.0,
        1e-6,
    )


def _segment_heading_variation(segment: PathSegmentSpec) -> float:
    return sum(
        abs(wrap_angle(segment.waypoints[idx].psi - segment.waypoints[idx - 1].psi))
        for idx in range(1, len(segment.waypoints))
    )
