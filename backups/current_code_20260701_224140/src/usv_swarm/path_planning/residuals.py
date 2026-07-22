from __future__ import annotations

import math
from typing import List

from ..dubins import dubins_shortest_path
from ..schema import CoverageResidual, CoverageState, PlannerConfig, Pose2D
from .coverage import RectangularCoverageModel, build_coverage_state, find_residual_components, mark_coverage_passes
from .obstacles import point_in_any_obstacle
from .types import DecomposedRegion, ObstacleField, ResidualBackfillPlan, SingleUsvTourPlan


def evaluate_tour_coverage_residuals(
    config: PlannerConfig,
    tours: List[SingleUsvTourPlan],
    resolution: float | None = None,
    obstacle_field: ObstacleField | None = None,
    include_non_cover_segments: bool = False,
) -> List[CoverageResidual]:
    return evaluate_tour_coverage_state(
        config,
        tours,
        resolution,
        obstacle_field,
        include_non_cover_segments=include_non_cover_segments,
    ).residual_components


def evaluate_tour_coverage_state(
    config: PlannerConfig,
    tours: List[SingleUsvTourPlan],
    resolution: float | None = None,
    obstacle_field: ObstacleField | None = None,
    include_non_cover_segments: bool = False,
) -> CoverageState:
    state = build_coverage_state(config, resolution=resolution)
    if obstacle_field is not None:
        for row, y in enumerate(state.y_coords):
            for col, x in enumerate(state.x_coords):
                if point_in_any_obstacle((float(x), float(y)), obstacle_field, inflated=True):
                    state.coverage_ratio[row, col] = 1.0
                    state.covered[row, col] = True
    model = RectangularCoverageModel.from_config(config)
    for tour in tours:
        pass_segments = []
        for segment in tour.segments:
            if not _segment_counts_as_coverage(segment.kind, include_non_cover_segments):
                continue
            pass_segments.extend(_segment_swept_passes(segment))
        if not pass_segments:
            for pattern in tour.selected_patterns.values():
                for coverage_pass in pattern.passes:
                    pass_segments.append((coverage_pass.start_pose, coverage_pass.end_pose))
        mark_coverage_passes(state, pass_segments, model, eta_cov=config.footprint.eta_cov)
    find_residual_components(state, (0.0, 0.0, config.mission.area_length_x, config.mission.area_length_y))
    return state


def _segment_counts_as_coverage(kind: str, include_non_cover_segments: bool) -> bool:
    if kind == "cover":
        return True
    if not include_non_cover_segments:
        return False
    return kind not in {"wait", "hold"}


def _segment_swept_passes(segment) -> List[tuple[Pose2D, Pose2D]]:
    if len(segment.waypoints) < 2:
        return []
    if segment.kind == "cover":
        start = segment.waypoints[0]
        end = segment.waypoints[-1]
        return [(Pose2D(start.x, start.y, start.psi), Pose2D(end.x, end.y, end.psi))]
    passes: List[tuple[Pose2D, Pose2D]] = []
    for first, second in zip(segment.waypoints, segment.waypoints[1:]):
        if math.hypot(second.x - first.x, second.y - first.y) <= 1e-9:
            continue
        passes.append((Pose2D(first.x, first.y, first.psi), Pose2D(second.x, second.y, second.psi)))
    return passes


def residual_to_region(
    residual: CoverageResidual,
    region_id: str,
    preferred_axis: str = "x",
    padding: float = 0.0,
    area_bounds: tuple[float, float, float, float] | None = None,
) -> DecomposedRegion:
    x_min, y_min, x_max, y_max = residual.bounds
    if padding > 0.0:
        x_min -= padding
        y_min -= padding
        x_max += padding
        y_max += padding
    if area_bounds is not None:
        ax0, ay0, ax1, ay1 = area_bounds
        x_min = max(ax0, x_min)
        y_min = max(ay0, y_min)
        x_max = min(ax1, x_max)
        y_max = min(ay1, y_max)
    polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    area = max(x_max - x_min, 0.0) * max(y_max - y_min, 0.0)
    return DecomposedRegion(
        region_id=region_id,
        bounds=(x_min, y_min, x_max, y_max),
        polygon=polygon,
        center=residual.centroid,
        area=area,
        preferred_axis=preferred_axis,
        source_algorithm="coverage_residual_backfill",
        metadata={"residual_id": str(residual.residual_id), "cell_count": str(len(residual.cells))},
    )


def assign_residual_backfill(
    residuals: List[CoverageResidual],
    tours: List[SingleUsvTourPlan],
    config: PlannerConfig,
    preferred_axis: str = "x",
) -> ResidualBackfillPlan:
    padding = max(config.footprint.width_wf / 2.0, 1e-6)
    area_bounds = (0.0, 0.0, config.mission.area_length_x, config.mission.area_length_y)
    residual_regions = [
        residual_to_region(
            residual,
            region_id=f"residual_region_{residual.residual_id}",
            preferred_axis=preferred_axis,
            padding=padding,
            area_bounds=area_bounds,
        )
        for residual in residuals
    ]
    agent_regions = {tour.agent_id: [] for tour in tours}
    estimated_start_times = {tour.agent_id: _tour_end_time(tour) for tour in tours}
    estimated_transition_cost = {}

    for region in residual_regions:
        best_agent = min(
            tours,
            key=lambda tour: estimated_start_times[tour.agent_id] + _transition_time_to_region(tour, region, config),
        )
        transition_time = _transition_time_to_region(best_agent, region, config)
        estimated_transition_cost[(best_agent.agent_id, region.region_id)] = transition_time
        agent_regions[best_agent.agent_id].append(region.region_id)
        estimated_start_times[best_agent.agent_id] += transition_time + _residual_service_time(region, config)

    return ResidualBackfillPlan(
        residual_regions=residual_regions,
        agent_regions=agent_regions,
        estimated_start_times=estimated_start_times,
        estimated_transition_cost=estimated_transition_cost,
        diagnostics={
            "residual_count": str(len(residual_regions)),
            "assigned_count": str(sum(len(items) for items in agent_regions.values())),
        },
    )


def _tour_end_time(tour: SingleUsvTourPlan) -> float:
    return max(
        (
            waypoint.time or 0.0
            for segment in tour.segments
            for waypoint in segment.waypoints[-1:]
        ),
        default=0.0,
    )


def _tour_end_pose(tour: SingleUsvTourPlan, config: PlannerConfig):
    for segment in reversed(tour.segments):
        if segment.waypoints:
            waypoint = segment.waypoints[-1]
            return waypoint
    state = config.fleet.initial_states_3dof[tour.agent_id]
    return state


def _transition_time_to_region(tour: SingleUsvTourPlan, region: DecomposedRegion, config: PlannerConfig) -> float:
    end = _tour_end_pose(tour, config)
    dx = region.center[0] - end.x
    dy = region.center[1] - end.y
    heading = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-9 else end.psi
    target_pose = Pose2D(region.center[0], region.center[1], heading)
    start_pose = Pose2D(end.x, end.y, end.psi)
    return dubins_shortest_path(start_pose, target_pose, config.fleet.min_turn_radius).total_length / max(
        config.fleet.cruise_speed,
        1e-6,
    )


def _residual_service_time(region: DecomposedRegion, config: PlannerConfig) -> float:
    short_side = min(region.bounds[2] - region.bounds[0], region.bounds[3] - region.bounds[1])
    long_side = max(region.bounds[2] - region.bounds[0], region.bounds[3] - region.bounds[1])
    pass_count = max(1, int(math.ceil(short_side / max(config.footprint.width_wf * (1.0 - config.mission.overlap_ratio), 1e-6))))
    return pass_count * long_side / max(config.fleet.cover_speed, 1e-6)
