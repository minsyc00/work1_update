"""Heterogeneous coverage modes and lazy obstacle-aware connection graph."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import atan2, cos, hypot, pi, sin
from typing import Dict, Mapping, Optional, Sequence, Tuple

from ...schema import AgentPlanningProfile, PlannerConfig, Pose2D
from ..dynamics_validation import validate_transition_sequence
from ..patterns import expand_entry_exit_pattern_variants, generate_region_patterns
from ..smoothing import build_obstacle_aware_transition_segments
from ..obstacles import clearance_to_obstacles
from ..tsp import build_region_service_segments
from ..types import (
    CoveragePass,
    DecomposedRegion,
    ObstacleField,
    PathPlanningConfig,
    PathSegmentSpec,
    RegionCoveragePattern,
)
from .config import CrownMcppConfig
from .geometry import (
    certify_continuous_pattern_coverage,
    repair_continuous_pattern_coverage,
)
from .motion import (
    CrownCurrentField,
    CrownMotionPrimitive,
    CurrentInfeasibleError,
    ZeroCurrentField,
    retime_motion_primitive,
    segment_to_motion_primitives,
)


ModePair = Tuple[Optional[str], Optional[str]]


@dataclass(frozen=True)
class CrownGeometricMode:
    agent_id: str
    task_id: str
    mode_id: str
    pattern: RegionCoveragePattern
    service_segments: Tuple[PathSegmentSpec, ...]
    nominal_service_primitives: Tuple[CrownMotionPrimitive, ...]
    nominal_duration: float
    nominal_energy: float

    @property
    def entry_pose(self) -> Pose2D:
        return self.pattern.entry_pose

    @property
    def exit_pose(self) -> Pose2D:
        return self.pattern.exit_pose


@dataclass(frozen=True)
class CrownGeometricConnection:
    agent_id: str
    from_mode_id: Optional[str]
    to_mode_id: Optional[str]
    segments: Tuple[PathSegmentSpec, ...]
    primitives: Tuple[CrownMotionPrimitive, ...]
    feasible: bool
    failure_reason: str = ""

    @property
    def duration(self) -> float:
        return sum(primitive.duration for primitive in self.primitives)

    @property
    def energy(self) -> float:
        return sum(primitive.energy for primitive in self.primitives)


@dataclass
class CrownTimeExpandedModeGraph:
    """One agent's mode graph with time-dependent primitive expansion."""

    agent_id: str
    numeric_agent_id: int
    profile: AgentPlanningProfile
    planner_config: PlannerConfig
    path_config: PathPlanningConfig
    crown_config: CrownMcppConfig
    obstacle_field: Optional[ObstacleField]
    modes_by_task: Mapping[str, Tuple[CrownGeometricMode, ...]]
    start_pose: Pose2D
    goal_pose: Pose2D
    goal_pose_explicit: bool = False
    current_field: CrownCurrentField = field(default_factory=ZeroCurrentField)
    connection_segment_cache: Dict[ModePair, Optional[Tuple[PathSegmentSpec, ...]]] = field(
        default_factory=dict
    )
    connection_cache: Dict[Tuple[Optional[str], Optional[str], int], CrownGeometricConnection] = field(
        default_factory=dict
    )
    service_cache: Dict[Tuple[str, int], Tuple[CrownMotionPrimitive, ...]] = field(
        default_factory=dict
    )
    diagnostics: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mode_lookup: Dict[str, CrownGeometricMode] = {
            mode.mode_id: mode
            for modes in self.modes_by_task.values()
            for mode in modes
        }
        expected = sum(len(modes) for modes in self.modes_by_task.values())
        if len(self.mode_lookup) != expected:
            raise ValueError("mode_id values must be unique inside an agent mode graph")

    @property
    def task_ids(self) -> Tuple[str, ...]:
        return tuple(self.modes_by_task)

    @property
    def planning_distance(self) -> float:
        return self.planner_config.safety.d_safe + 2.0 * self.crown_config.total_position_error

    def modes_for_task(self, task_id: str) -> Tuple[CrownGeometricMode, ...]:
        return self.modes_by_task.get(task_id, ())

    def service_primitives(
        self,
        mode_id: str,
        start_slot: int,
    ) -> Optional[Tuple[CrownMotionPrimitive, ...]]:
        key = (mode_id, start_slot)
        if key in self.service_cache:
            return self.service_cache[key]
        mode = self.mode_lookup[mode_id]
        primitives = []
        absolute_time = start_slot * self.crown_config.time_step
        try:
            # A coverage mode has fixed geometry and a fixed finite primitive
            # decomposition.  Only duration/energy depend on the departure
            # time.  Retiming those primitives preserves canonical tube IDs
            # (and user-supplied finite-model resources) across all slots.
            for primitive in mode.nominal_service_primitives:
                timed = retime_motion_primitive(
                    primitive,
                    profile=self.profile,
                    current_field=self.current_field,
                    absolute_start_time=absolute_time,
                )
                primitives.append(timed)
                absolute_time += timed.duration
        except CurrentInfeasibleError:
            self.diagnostics["time_dependent_service_infeasible"] = (
                self.diagnostics.get("time_dependent_service_infeasible", 0) + 1
            )
            return None
        result = tuple(primitives)
        self.service_cache[key] = result
        return result

    def _poses_for_pair(self, from_mode_id: Optional[str], to_mode_id: Optional[str]) -> Tuple[Pose2D, Pose2D]:
        start = self.start_pose if from_mode_id is None else self.mode_lookup[from_mode_id].exit_pose
        end = self.goal_pose if to_mode_id is None else self.mode_lookup[to_mode_id].entry_pose
        return start, end

    def _connection_segments(
        self,
        from_mode_id: Optional[str],
        to_mode_id: Optional[str],
    ) -> Optional[Tuple[PathSegmentSpec, ...]]:
        pair = (from_mode_id, to_mode_id)
        if pair in self.connection_segment_cache:
            return self.connection_segment_cache[pair]
        if from_mode_id is None and to_mode_id is None:
            self.connection_segment_cache[pair] = ()
            return ()
        if (
            not self.goal_pose_explicit
            and not self.crown_config.return_to_start
            and to_mode_id is None
        ):
            self.connection_segment_cache[pair] = ()
            return ()
        start, end = self._poses_for_pair(from_mode_id, to_mode_id)
        if (
            abs(start.x - end.x) <= 1.0e-9
            and abs(start.y - end.y) <= 1.0e-9
            and abs(start.psi - end.psi) <= 1.0e-9
        ):
            self.connection_segment_cache[pair] = ()
            return ()
        segments = tuple(
            build_obstacle_aware_transition_segments(
                segment_id=(
                    f"crown_agent{self.numeric_agent_id}_"
                    f"{from_mode_id or 'depot'}_to_{to_mode_id or 'goal'}"
                ),
                start=start,
                end=end,
                start_time=0.0,
                config=self.planner_config,
                path_config=self.path_config,
                obstacle_field=self.obstacle_field,
                kind="transit",
            )
        )
        feasible = all(
            segment.metadata.get("kinematic_feasible", "true") != "false"
            and not segment.metadata.get("invalid_reasons")
            for segment in segments
        )
        if feasible and segments:
            feasible = validate_transition_sequence(
                segments,
                self.planner_config,
                obstacle_field=self.obstacle_field,
                retime=True,
            ).valid
        if not feasible:
            self.connection_segment_cache[pair] = None
            self.diagnostics["geometric_connection_infeasible"] = (
                self.diagnostics.get("geometric_connection_infeasible", 0) + 1
            )
            return None
        self.connection_segment_cache[pair] = segments
        return segments

    def connection(
        self,
        from_mode_id: Optional[str],
        to_mode_id: Optional[str],
        departure_slot: int,
    ) -> CrownGeometricConnection:
        key = (from_mode_id, to_mode_id, departure_slot)
        if key in self.connection_cache:
            return self.connection_cache[key]
        segments = self._connection_segments(from_mode_id, to_mode_id)
        if segments is None:
            result = CrownGeometricConnection(
                self.agent_id,
                from_mode_id,
                to_mode_id,
                (),
                (),
                False,
                "geometric_connection_infeasible",
            )
            self.connection_cache[key] = result
            return result
        primitives = []
        absolute_time = departure_slot * self.crown_config.time_step
        try:
            for segment_index, segment in enumerate(segments):
                segment_primitives = segment_to_motion_primitives(
                    segment,
                    agent_id=self.agent_id,
                    profile=self.profile,
                    crown_config=self.crown_config,
                    planning_distance=self.planning_distance,
                    current_field=self.current_field,
                    primitive_prefix=(
                        f"connection:{from_mode_id or 'depot'}:"
                        f"{to_mode_id or 'goal'}:{departure_slot}:{segment_index}"
                    ),
                    absolute_start_time=absolute_time,
                )
                primitives.extend(segment_primitives)
                absolute_time += sum(item.duration for item in segment_primitives)
        except CurrentInfeasibleError as error:
            result = CrownGeometricConnection(
                self.agent_id,
                from_mode_id,
                to_mode_id,
                segments,
                (),
                False,
                str(error),
            )
            self.connection_cache[key] = result
            self.diagnostics["current_connection_infeasible"] = (
                self.diagnostics.get("current_connection_infeasible", 0) + 1
            )
            return result
        result = CrownGeometricConnection(
            self.agent_id,
            from_mode_id,
            to_mode_id,
            segments,
            tuple(primitives),
            True,
        )
        self.connection_cache[key] = result
        return result


def _effective_coverage_config(
    config: PlannerConfig,
    agent_id: int,
    crown_config: CrownMcppConfig,
) -> PlannerConfig:
    agent_config = config.for_agent(agent_id)
    effective_width = agent_config.footprint.width_wf - 2.0 * crown_config.total_position_error
    effective_length = agent_config.footprint.length_lf - 2.0 * crown_config.total_position_error
    if effective_width <= 0.0 or effective_length <= 0.0:
        raise ValueError(
            f"agent {agent_id} has no positive effective coverage footprint after error margin"
        )
    return replace(
        agent_config,
        footprint=replace(
            agent_config.footprint,
            width_wf=effective_width,
            length_lf=effective_length,
        ),
    )


def _repair_pattern_mission_boundary(
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
) -> RegionCoveragePattern:
    """Shift/clip straight passes so the physical hull stays in the mission.

    Narrow responsibility bands touching the outer boundary are legitimately
    coverable because the sensing footprint extends beyond the hull center.
    Raw polygon intersections can nevertheless place the centerline too close
    to the boundary.  This deterministic repair moves the centerline inward;
    the subsequent continuous swept-footprint and full dynamics/obstacle
    validators decide whether the repaired mode is admissible.  The final
    pipeline still audits the shared finite coverage grid once globally.
    """

    vehicle = config.vehicle_footprint
    if vehicle is None:
        return pattern
    repaired = []
    total_length = 0.0
    changed = False
    for coverage_pass in pattern.passes:
        dx = coverage_pass.end_pose.x - coverage_pass.start_pose.x
        dy = coverage_pass.end_pose.y - coverage_pass.start_pose.y
        heading = atan2(dy, dx) if hypot(dx, dy) > 1.0e-12 else coverage_pass.start_pose.psi
        if abs(dx) > 1.0e-9 and abs(dy) > 1.0e-9:
            # Oriented passes already come from polygon clipping; changing
            # their line direction would change the advertised scan angle.
            repaired.append(coverage_pass)
            total_length += coverage_pass.length
            continue
        half_x = 0.5 * (
            abs(cos(heading)) * vehicle.length
            + abs(sin(heading)) * vehicle.width
        )
        half_y = 0.5 * (
            abs(sin(heading)) * vehicle.length
            + abs(cos(heading)) * vehicle.width
        )
        maneuver_margin = max(config.fleet.min_turn_radius, 0.0)
        x_low = half_x + maneuver_margin
        x_high = config.mission.area_length_x - half_x - maneuver_margin
        y_low = half_y + maneuver_margin
        y_high = config.mission.area_length_y - half_y - maneuver_margin
        if x_low > x_high or y_low > y_high:
            return pattern

        if abs(dx) >= abs(dy):
            center_y = min(max(0.5 * (coverage_pass.start_pose.y + coverage_pass.end_pose.y), y_low), y_high)
            start_x = min(max(coverage_pass.start_pose.x, x_low), x_high)
            end_x = min(max(coverage_pass.end_pose.x, x_low), x_high)
            start = Pose2D(start_x, center_y, 0.0 if end_x >= start_x else 3.141592653589793)
            end = Pose2D(end_x, center_y, start.psi)
        else:
            center_x = min(max(0.5 * (coverage_pass.start_pose.x + coverage_pass.end_pose.x), x_low), x_high)
            start_y = min(max(coverage_pass.start_pose.y, y_low), y_high)
            end_y = min(max(coverage_pass.end_pose.y, y_low), y_high)
            vertical_heading = 1.5707963267948966 if end_y >= start_y else -1.5707963267948966
            start = Pose2D(center_x, start_y, vertical_heading)
            end = Pose2D(center_x, end_y, vertical_heading)
        length = hypot(end.x - start.x, end.y - start.y)
        changed = changed or (
            abs(start.x - coverage_pass.start_pose.x) > 1.0e-9
            or abs(start.y - coverage_pass.start_pose.y) > 1.0e-9
            or abs(end.x - coverage_pass.end_pose.x) > 1.0e-9
            or abs(end.y - coverage_pass.end_pose.y) > 1.0e-9
        )
        total_length += length
        repaired.append(
            replace(
                coverage_pass,
                start_pose=start,
                end_pose=end,
                length=length,
            )
        )
    if not changed or not repaired:
        return pattern
    return replace(
        pattern,
        passes=repaired,
        entry_pose=repaired[0].start_pose,
        exit_pose=repaired[-1].end_pose,
        coverage_length=total_length,
        total_length=total_length + pattern.turn_length,
        metadata={**pattern.metadata, "crown_boundary_centerline_repair": "true"},
    )


def _apply_connector_pockets(
    region: DecomposedRegion,
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
) -> RegionCoveragePattern:
    """Retract pass endpoints while preserving longitudinal sensor coverage.

    A centerline ending exactly at a mission boundary can be coverage-valid
    but unusable as a nonholonomic route node because the hull points out of
    the map.  The rectangular sensor already covers half its length beyond the
    center, so that amount is a legitimate maneuver pocket.  The finite-grid
    validator below remains the authority and rejects any over-retraction.
    """

    if not pattern.passes:
        return pattern
    maximum_inset = max(config.footprint.length_lf * 0.5, 0.0)
    vehicle = config.vehicle_footprint
    minimum_maneuver_span = 2.0 * max(
        config.fleet.min_turn_radius,
        0.0 if vehicle is None else 0.5 * vehicle.length,
    )
    repaired = []
    coverage_length = 0.0
    applied = []
    for source in pattern.passes:
        dx = source.end_pose.x - source.start_pose.x
        dy = source.end_pose.y - source.start_pose.y
        length = hypot(dx, dy)
        if length <= 1.0e-9 or maximum_inset <= 1.0e-12:
            repaired.append(source)
            coverage_length += length
            applied.append(0.0)
            continue
        ux, uy = dx / length, dy / length
        support = [x * ux + y * uy for x, y in region.polygon]
        start_projection = source.start_pose.x * ux + source.start_pose.y * uy
        end_projection = source.end_pose.x * ux + source.end_pose.y * uy
        start_coverage_allowance = min(support) + maximum_inset - start_projection
        end_coverage_allowance = end_projection + maximum_inset - max(support)
        inset = min(
            maximum_inset,
            max(0.0, start_coverage_allowance),
            max(0.0, end_coverage_allowance),
            # Never use sensor overhang to collapse a short physical service
            # leg into an in-place observation pose.  A point with a declared
            # Dubins heading can be service-valid yet unreachable from every
            # depot in a narrow responsibility sliver.  Preserving this
            # maneuver span gives the connector a real approach/exit edge.
            max(0.0, 0.5 * (length - minimum_maneuver_span)),
        )
        start = Pose2D(
            source.start_pose.x + inset * ux,
            source.start_pose.y + inset * uy,
            source.start_pose.psi,
        )
        end = Pose2D(
            source.end_pose.x - inset * ux,
            source.end_pose.y - inset * uy,
            source.end_pose.psi,
        )
        repaired_length = hypot(end.x - start.x, end.y - start.y)
        repaired.append(
            replace(
                source,
                start_pose=start,
                end_pose=end,
                length=repaired_length,
            )
        )
        coverage_length += repaired_length
        applied.append(inset)
    return replace(
        pattern,
        passes=repaired,
        entry_pose=repaired[0].start_pose,
        exit_pose=repaired[-1].end_pose,
        coverage_length=coverage_length,
        total_length=coverage_length + pattern.turn_length,
        metadata={
            **pattern.metadata,
            "crown_connector_pockets": ",".join(f"{value:.9f}" for value in applied),
            "crown_connector_pocket_coverage_validated": "pending",
        },
    )


def _remove_redundant_coverage_passes(
    region: DecomposedRegion,
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    target_fraction: float,
) -> RegionCoveragePattern:
    """Drop scan lines whose removal preserves the continuous certificate."""

    if len(pattern.passes) <= 1:
        return pattern
    current = pattern
    removed = 0
    changed = True
    while changed and len(current.passes) > 1:
        changed = False
        for index in range(len(current.passes)):
            passes = [
                replace(item, sequence_index=new_index)
                for new_index, item in enumerate(
                    current.passes[:index] + current.passes[index + 1 :]
                )
            ]
            candidate = replace(
                current,
                passes=passes,
                entry_pose=passes[0].start_pose,
                exit_pose=passes[-1].end_pose,
                coverage_length=sum(item.length for item in passes),
            )
            certificate = certify_continuous_pattern_coverage(
                region,
                candidate,
                config,
            )
            if certificate.coverage_fraction + 1.0e-9 < target_fraction:
                continue
            current = candidate
            removed += 1
            changed = True
            break
    if not removed:
        return pattern
    return replace(
        current,
        metadata={
            **current.metadata,
            "crown_redundant_passes_removed": str(removed),
        },
    )


def _sensor_offset_observation_patterns(
    region: DecomposedRegion,
    config: PlannerConfig,
    obstacle_field: Optional[ObstacleField],
) -> Tuple[RegionCoveragePattern, ...]:
    """Cover a tiny responsibility sliver from nearby maneuverable water.

    Exact BCD can create a free-space polygon that is smaller than the
    vehicle's nonholonomic maneuver envelope.  Requiring the service
    centerline to remain inside that polygon then creates a mode which is
    coverage-valid but unreachable.  A sensor footprint is not a solid hull:
    its centerline may legitimately run in adjacent free water as long as the
    exact swept sensor rectangle covers the responsibility polygon and the
    ordinary hull/dynamics validators accept the service trajectory.
    """

    x0, y0, x1, y1 = region.bounds
    coverage_length = config.footprint.length_lf
    coverage_width = config.footprint.width_wf
    fits_x = (
        x1 - x0 <= coverage_length + 1.0e-9
        and y1 - y0 <= coverage_width + 1.0e-9
    )
    fits_y = (
        y1 - y0 <= coverage_length + 1.0e-9
        and x1 - x0 <= coverage_width + 1.0e-9
    )
    if not (fits_x or fits_y):
        return ()
    vehicle = config.vehicle_footprint
    service_span = max(
        2.0 * config.fleet.min_turn_radius,
        0.0 if vehicle is None else vehicle.length,
        1.0e-3,
    )

    def samples(low: float, high: float) -> Tuple[float, ...]:
        if low > high + 1.0e-9:
            return ()
        if high - low <= 1.0e-9:
            return (0.5 * (low + high),)
        return tuple(
            low + (high - low) * fraction
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
        )

    scored = []
    serial = 0
    for axis in ("x", "y"):
        if axis == "x" and not fits_x:
            continue
        if axis == "y" and not fits_y:
            continue
        along_sensor_span = coverage_length + service_span
        if axis == "x":
            along_values = samples(
                x1 - 0.5 * along_sensor_span,
                x0 + 0.5 * along_sensor_span,
            )
            cross_values = samples(
                y1 - 0.5 * coverage_width,
                y0 + 0.5 * coverage_width,
            )
        else:
            along_values = samples(
                y1 - 0.5 * along_sensor_span,
                y0 + 0.5 * along_sensor_span,
            )
            cross_values = samples(
                x1 - 0.5 * coverage_width,
                x0 + 0.5 * coverage_width,
            )
        for along in along_values:
            for cross in cross_values:
                if axis == "x":
                    start = Pose2D(along - 0.5 * service_span, cross, 0.0)
                    end = Pose2D(along + 0.5 * service_span, cross, 0.0)
                    center = (along, cross)
                    center_coordinate = cross
                    heading = 0.0
                else:
                    start = Pose2D(cross, along - 0.5 * service_span, 0.5 * pi)
                    end = Pose2D(cross, along + 0.5 * service_span, 0.5 * pi)
                    center = (cross, along)
                    center_coordinate = cross
                    heading = 0.5 * pi
                pass_spec = CoveragePass(
                    pass_id=f"{region.region_id}:sensor-offset:{serial}",
                    region_id=region.region_id,
                    sequence_index=0,
                    scan_axis=axis,
                    start_pose=start,
                    end_pose=end,
                    center_coordinate=center_coordinate,
                    width=coverage_width,
                    length=service_span,
                )
                pattern = RegionCoveragePattern(
                    pattern_id=f"{region.region_id}:sensor-offset-pattern:{serial}",
                    region_id=region.region_id,
                    scan_axis=axis,
                    passes=[pass_spec],
                    entry_pose=start,
                    exit_pose=end,
                    coverage_length=service_span,
                    turn_length=0.0,
                    turn_angle=0.0,
                    total_length=service_span,
                    estimated_time=service_span / max(config.fleet.cover_speed, 1.0e-9),
                    max_curvature=0.0,
                    feasible=True,
                    metadata={
                        "source": "crown_sensor_offset_observation",
                        "crown_service_centerline_outside_responsibility_allowed": "true",
                        "crown_responsibility_coverage_validation": "pending",
                        "scan_angle_rad": f"{heading:.12g}",
                    },
                )
                certificate = certify_continuous_pattern_coverage(
                    region,
                    pattern,
                    config,
                )
                if certificate.coverage_fraction + 1.0e-9 < 0.99:
                    serial += 1
                    continue
                obstacle_clearance = (
                    float("inf")
                    if obstacle_field is None
                    else clearance_to_obstacles(
                        center,
                        obstacle_field,
                        inflated=True,
                    )
                )
                boundary_clearance = min(
                    center[0],
                    center[1],
                    config.mission.area_length_x - center[0],
                    config.mission.area_length_y - center[1],
                )
                scored.append(
                    (
                        -min(obstacle_clearance, boundary_clearance),
                        pattern.pattern_id,
                        pattern,
                    )
                )
                serial += 1
    return tuple(item[2] for item in sorted(scored))


def _select_endpoint_diverse_modes(
    candidates: Sequence[CrownGeometricMode],
    limit: int,
) -> Tuple[CrownGeometricMode, ...]:
    """Keep service quality while avoiding a one-sided mode library."""

    ordered = sorted(
        candidates,
        key=lambda mode: (
            mode.nominal_duration,
            mode.nominal_energy,
            mode.mode_id,
        ),
    )
    if len(ordered) <= limit:
        return tuple(ordered)
    selected = [ordered.pop(0)]
    while ordered and len(selected) < limit:
        candidate = max(
            ordered,
            key=lambda mode: (
                min(
                    hypot(
                        mode.entry_pose.x - chosen.entry_pose.x,
                        mode.entry_pose.y - chosen.entry_pose.y,
                    )
                    for chosen in selected
                ),
                -mode.nominal_duration,
                -mode.nominal_energy,
                mode.mode_id,
            ),
        )
        selected.append(candidate)
        ordered.remove(candidate)
    return tuple(selected)


def build_agent_mode_graph(
    agent_id: int,
    regions: Sequence[DecomposedRegion],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    crown_config: CrownMcppConfig,
    obstacle_field: Optional[ObstacleField],
    *,
    current_field: Optional[CrownCurrentField] = None,
    goal_pose: Optional[Pose2D] = None,
) -> CrownTimeExpandedModeGraph:
    """Generate validated heterogeneous modes for every fixed responsibility unit."""

    mode_config = _effective_coverage_config(config, agent_id, crown_config)
    profile = config.profile_for_agent(agent_id)
    modes_by_task: Dict[str, Tuple[CrownGeometricMode, ...]] = {}
    planning_distance = config.safety.d_safe + 2.0 * crown_config.total_position_error
    field = current_field or ZeroCurrentField()
    for region in regions:
        base_patterns = generate_region_patterns(
            region,
            mode_config,
            path_config,
            obstacle_field,
        )
        if region.source_algorithm == "crown_atomic_sweep_band_preprocessing" and base_patterns:
            minimum_pass_count = min(len(pattern.passes) for pattern in base_patterns)
            base_patterns = [
                pattern
                for pattern in base_patterns
                if len(pattern.passes) == minimum_pass_count
            ]
        offset_patterns = _sensor_offset_observation_patterns(
            region,
            mode_config,
            obstacle_field,
        )
        patterns = [
            variant
            for pattern in tuple(offset_patterns) + tuple(base_patterns)
            for variant in expand_entry_exit_pattern_variants(pattern)
        ]
        candidates = []
        x0, y0, x1, y1 = region.bounds
        long_responsibility = max(x1 - x0, y1 - y0) > 2.0 * max(
            mode_config.footprint.length_lf,
            mode_config.footprint.width_wf,
        )
        candidate_evaluation_limit = (
            max(crown_config.mode_limit_per_region_agent, 4)
            if long_responsibility
            else crown_config.mode_limit_per_region_agent
        )
        for pattern in patterns:
            # ``base_patterns`` are already ordered by estimated service
            # quality and each base pattern emits entry/exit-diverse variants.
            # Once the declared finite mode budget is filled, constructing
            # and validating additional modes only to discard them below is a
            # major preparation-time cost on 200/400 m maps.
            if len(candidates) >= candidate_evaluation_limit:
                break
            pattern = _repair_pattern_mission_boundary(pattern, mode_config)
            pattern = _apply_connector_pockets(region, pattern, mode_config)
            pattern = _remove_redundant_coverage_passes(
                region,
                pattern,
                mode_config,
                path_config.target_coverage_fraction,
            )
            continuous_coverage = certify_continuous_pattern_coverage(
                region,
                pattern,
                mode_config,
            )
            if (
                continuous_coverage.coverage_fraction + 1.0e-9
                < path_config.target_coverage_fraction
            ):
                repaired_pattern = repair_continuous_pattern_coverage(
                    region,
                    pattern,
                    mode_config,
                    maximum_relative_missing=1.0,
                    maximum_repair_passes=8,
                )
                repaired_coverage = certify_continuous_pattern_coverage(
                    region,
                    repaired_pattern,
                    mode_config,
                )
                if (
                    repaired_coverage.coverage_fraction
                    > continuous_coverage.coverage_fraction + 1.0e-12
                ):
                    pattern = repaired_pattern
                    continuous_coverage = repaired_coverage
            if (
                continuous_coverage.coverage_fraction + 1.0e-9
                < path_config.target_coverage_fraction
            ):
                continue
            coverage_fraction = continuous_coverage.coverage_fraction
            pattern = replace(
                pattern,
                metadata={
                    **pattern.metadata,
                    "crown_validated_coverage_fraction": f"{coverage_fraction:.9f}",
                    "crown_coverage_model": "continuous_swept_error_shrunk_footprint",
                    "crown_continuous_coverage_fraction": (
                        f"{continuous_coverage.coverage_fraction:.12g}"
                    ),
                    "crown_continuous_missing_area": (
                        f"{continuous_coverage.missing_area:.12g}"
                    ),
                    "crown_continuous_coverage_tolerance": (
                        f"{continuous_coverage.tolerance:.12g}"
                    ),
                    "crown_continuous_coverage_complete": str(
                        continuous_coverage.valid
                    ).lower(),
                    "crown_continuous_coverage_target_met": "true",
                    "crown_connector_pocket_coverage_validated": "true",
                },
            )
            segments = tuple(
                build_region_service_segments(
                    agent_id,
                    pattern,
                    mode_config,
                    path_config,
                    obstacle_field,
                )
            )
            if not segments:
                continue
            if any(
                segment.metadata.get("kinematic_feasible", "true") == "false"
                or bool(segment.metadata.get("invalid_reasons"))
                for segment in segments
            ):
                continue
            if not validate_transition_sequence(
                segments,
                mode_config,
                obstacle_field=obstacle_field,
                retime=True,
            ).valid:
                continue
            try:
                primitives = tuple(
                    primitive
                    for segment_index, segment in enumerate(segments)
                    for primitive in segment_to_motion_primitives(
                        segment,
                        agent_id=str(agent_id),
                        profile=profile,
                        crown_config=crown_config,
                        planning_distance=planning_distance,
                        current_field=field,
                        primitive_prefix=f"mode:{pattern.pattern_id}:{segment_index}",
                        task_id=region.region_id,
                        mode_id=pattern.pattern_id,
                    )
                )
            except CurrentInfeasibleError:
                continue
            if not primitives:
                continue
            energy = sum(primitive.energy for primitive in primitives)
            duration = sum(primitive.duration for primitive in primitives)
            if (
                profile.max_mission_time is not None
                and duration > profile.max_mission_time + 1.0e-9
            ):
                continue
            if profile.battery_capacity is not None and energy > profile.battery_capacity + 1.0e-9:
                continue
            candidates.append(
                CrownGeometricMode(
                    agent_id=str(agent_id),
                    task_id=region.region_id,
                    mode_id=pattern.pattern_id,
                    pattern=pattern,
                    service_segments=segments,
                    nominal_service_primitives=primitives,
                    nominal_duration=duration,
                    nominal_energy=energy,
                )
            )
        modes_by_task[region.region_id] = _select_endpoint_diverse_modes(
            candidates,
            crown_config.mode_limit_per_region_agent,
        )

    start_pose = config.fleet.initial_states_3dof[agent_id].pose()
    return CrownTimeExpandedModeGraph(
        agent_id=str(agent_id),
        numeric_agent_id=agent_id,
        profile=profile,
        planner_config=mode_config,
        path_config=path_config,
        crown_config=crown_config,
        obstacle_field=obstacle_field,
        modes_by_task=modes_by_task,
        start_pose=start_pose,
        goal_pose=goal_pose or start_pose,
        goal_pose_explicit=goal_pose is not None,
        current_field=field,
    )


def clone_agent_mode_graph(
    template: CrownTimeExpandedModeGraph,
    agent_id: int,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    crown_config: CrownMcppConfig,
    *,
    goal_pose: Optional[Pose2D] = None,
) -> CrownTimeExpandedModeGraph:
    """Reuse a validated mode library for an identical physical agent.

    Coverage geometry, service dynamics, and conservative tube resources are
    identical for agents with the same planning profile.  Rebuilding them for
    every homogeneous USV dominated preparation on the large bundled maps.
    This clone changes only agent-scoped identifiers and depot/goal state;
    lazy connection caches deliberately start empty.
    """

    profile = config.profile_for_agent(agent_id)
    if profile.fingerprint != template.profile.fingerprint:
        raise ValueError("mode graph template profile does not match target agent")
    source_id = template.numeric_agent_id

    def rename(value: str) -> str:
        return value.replace(
            f"agent_{source_id}", f"agent_{agent_id}"
        ).replace(
            f"agent{source_id}", f"agent{agent_id}"
        )

    def metadata(values: Mapping[str, str]) -> Dict[str, str]:
        renamed = {
            key: rename(value) if isinstance(value, str) else value
            for key, value in dict(values or {}).items()
        }
        if "agent_id" in renamed:
            renamed["agent_id"] = str(agent_id)
        return renamed

    def segment(source: PathSegmentSpec) -> PathSegmentSpec:
        return replace(
            source,
            segment_id=rename(source.segment_id),
            waypoints=list(source.waypoints),
            metadata=metadata(source.metadata),
        )

    modes_by_task: Dict[str, Tuple[CrownGeometricMode, ...]] = {}
    for task_id, source_modes in template.modes_by_task.items():
        cloned_modes = []
        for source_mode in source_modes:
            mode_id = rename(source_mode.mode_id)
            passes = [
                replace(item, pass_id=rename(item.pass_id))
                for item in source_mode.pattern.passes
            ]
            pattern = replace(
                source_mode.pattern,
                pattern_id=mode_id,
                passes=passes,
                metadata=metadata(source_mode.pattern.metadata),
            )
            primitives = tuple(
                replace(
                    primitive,
                    primitive_id=rename(primitive.primitive_id),
                    agent_id=str(agent_id),
                    mode_id=mode_id,
                    segment=segment(primitive.segment),
                    metadata=metadata(primitive.metadata),
                )
                for primitive in source_mode.nominal_service_primitives
            )
            cloned_modes.append(
                replace(
                    source_mode,
                    agent_id=str(agent_id),
                    mode_id=mode_id,
                    pattern=pattern,
                    service_segments=tuple(
                        segment(item) for item in source_mode.service_segments
                    ),
                    nominal_service_primitives=primitives,
                )
            )
        modes_by_task[task_id] = tuple(cloned_modes)

    mode_config = _effective_coverage_config(config, agent_id, crown_config)
    start_pose = config.fleet.initial_states_3dof[agent_id].pose()
    return CrownTimeExpandedModeGraph(
        agent_id=str(agent_id),
        numeric_agent_id=agent_id,
        profile=profile,
        planner_config=mode_config,
        path_config=path_config,
        crown_config=crown_config,
        obstacle_field=template.obstacle_field,
        modes_by_task=modes_by_task,
        start_pose=start_pose,
        goal_pose=goal_pose or start_pose,
        goal_pose_explicit=goal_pose is not None,
        current_field=template.current_field,
    )
