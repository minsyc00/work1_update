from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from ..schema import PlannerConfig
from .obstacles import point_in_any_obstacle, point_in_polygon
from .types import AgentPathPlan, CoverageOwnershipMap, DecomposedRegion, ObstacleField, PathPlanningConfig, PathSegmentSpec


@dataclass(frozen=True)
class ResourceWindow:
    resource_id: str
    start: float
    end: float
    agent_id: int
    kind: str
    segment_id: str


@dataclass(frozen=True)
class RepeatOverlapScore:
    overlap_length: float
    penalty: float
    hit_ratio: float
    sampled_point_count: int
    hit_count: int


@dataclass(frozen=True)
class CrossAgentOverlapScore:
    overlap_length: float
    penalty: float
    hit_ratio: float
    sampled_point_count: int
    hit_count: int
    overlap_by_agent: Dict[int, float]
    overlap_by_kind: Dict[str, float]


def stable_resource_id(segment: PathSegmentSpec, path_config: PathPlanningConfig) -> str:
    points = _segment_points(segment)
    if len(points) < 2:
        return segment.metadata.get("resource_id", f"stationary:{segment.segment_id}")
    grid = max(float(path_config.shared_resource_grid_size), 1e-6)
    start_key = _quantized_point(points[0], grid)
    end_key = _quantized_point(points[-1], grid)
    mid_key = _quantized_point(points[len(points) // 2], grid)
    axis = _dominant_axis(points[0], points[-1])
    region_id = segment.metadata.get("region_id") or segment.metadata.get("to_region") or "global"
    pass_id = segment.metadata.get("pass_id", "")
    connector = segment.metadata.get("connector", segment.path_source or "")
    if segment.kind == "cover":
        residual = "residual:" if segment.metadata.get("residual_backfill") == "true" or "residual" in region_id else ""
        return f"{residual}cover:{region_id}:{axis}:{mid_key}:{pass_id}"
    if segment.kind == "turn":
        return f"turn_pocket:{region_id}:{mid_key}:{axis}"
    ordered = sorted([start_key, end_key])
    prefix = "corridor"
    if segment.metadata.get("region_tsp_edge") == "true":
        prefix = "region_corridor"
    if "residual" in region_id or "residual" in segment.segment_id:
        prefix = "residual_corridor"
    if connector in {"astar_corridor", "smoothed_astar_corridor", "motion_lattice", "motion_lattice_no_astar"}:
        prefix = f"{prefix}:{connector}"
    return f"{prefix}:{ordered[0]}:{ordered[1]}"


def assign_stable_resource_ids(agents: Dict[int, AgentPathPlan], path_config: PathPlanningConfig) -> None:
    for agent in agents.values():
        for segment in agent.segments:
            previous = segment.metadata.get("resource_id")
            if previous:
                segment.metadata.setdefault("legacy_resource_id", previous)
            segment.metadata["resource_id"] = stable_resource_id(segment, path_config)


def build_coverage_ownership_map(
    regions: Sequence[DecomposedRegion],
    agent_regions: Dict[int, Sequence[str]],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None = None,
) -> CoverageOwnershipMap:
    grid = max(float(path_config.cross_agent_overlap_grid_size or path_config.shared_resource_grid_size), 1e-6)
    region_lookup = {region.region_id: region for region in regions}
    region_owner: Dict[str, int] = {}
    owner_by_cell: Dict[str, int] = {}
    conflict_count = 0
    for agent_id, region_ids in sorted(agent_regions.items()):
        for region_id in region_ids:
            region = region_lookup.get(region_id)
            if region is None:
                continue
            region_owner[region_id] = int(agent_id)
            x_min, y_min, x_max, y_max = region.bounds
            x_steps = max(int(math.ceil((x_max - x_min) / grid)), 1)
            y_steps = max(int(math.ceil((y_max - y_min) / grid)), 1)
            for ix in range(x_steps + 1):
                x = min(max(x_min + ix * grid, 0.0), config.mission.area_length_x)
                for iy in range(y_steps + 1):
                    y = min(max(y_min + iy * grid, 0.0), config.mission.area_length_y)
                    point = (float(x), float(y))
                    if not point_in_polygon(point, region.polygon):
                        continue
                    if obstacle_field is not None and point_in_any_obstacle(point, obstacle_field, inflated=True):
                        continue
                    key = _quantized_point(point, grid)
                    previous = owner_by_cell.get(key)
                    if previous is not None and previous != int(agent_id):
                        conflict_count += 1
                        continue
                    owner_by_cell[key] = int(agent_id)
    return CoverageOwnershipMap(
        resolution=grid,
        owner_by_cell=owner_by_cell,
        region_owner=region_owner,
        metadata={
            "cell_count": str(len(owner_by_cell)),
            "region_count": str(len(region_owner)),
            "conflict_count": str(conflict_count),
        },
    )


def estimate_repeat_overlap_length(
    candidate_segments: Sequence[PathSegmentSpec],
    existing_segments: Sequence[PathSegmentSpec],
    path_config: PathPlanningConfig,
) -> float:
    return score_repeat_overlap(candidate_segments, existing_segments, path_config, penalty_weight=1.0).overlap_length


def score_cross_agent_ownership_overlap(
    candidate_segments: Sequence[PathSegmentSpec],
    agent_id: int,
    ownership_map: CoverageOwnershipMap | None,
    path_config: PathPlanningConfig,
    config: PlannerConfig | None = None,
    annotate: bool = False,
) -> CrossAgentOverlapScore:
    if (
        ownership_map is None
        or not path_config.enable_cross_agent_coverage_penalty
        or not ownership_map.owner_by_cell
    ):
        if annotate:
            for segment in candidate_segments:
                _annotate_cross_agent(segment, 0.0, 0.0, 0, 0, False)
        return CrossAgentOverlapScore(0.0, 0.0, 0.0, 0, 0, {}, {})
    grid = max(float(path_config.cross_agent_overlap_grid_size or ownership_map.resolution or path_config.shared_resource_grid_size), 1e-6)
    escape_remaining = _initial_escape_distance(candidate_segments, agent_id, ownership_map, grid, path_config, config)
    total_overlap = 0.0
    total_penalty = 0.0
    total_points = 0
    total_hits = 0
    by_agent: Dict[int, float] = {}
    by_kind: Dict[str, float] = {}
    for segment in candidate_segments:
        points = _sample_segment_points(segment, grid)
        if len(points) < 2:
            if annotate:
                _annotate_cross_agent(segment, 0.0, 0.0, 0, len(points), False)
            continue
        segment_overlap = 0.0
        segment_penalty = 0.0
        segment_hits = 0
        for start, end in zip(points[:-1], points[1:]):
            distance = math.hypot(end[0] - start[0], end[1] - start[1])
            if distance <= 1e-9:
                continue
            midpoint = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
            owner = ownership_map.owner_by_cell.get(_quantized_point(midpoint, grid))
            if owner is None or owner == agent_id:
                total_points += 1
                continue
            charge_length = distance
            if escape_remaining > 0.0:
                free_length = min(escape_remaining, charge_length)
                charge_length -= free_length
                escape_remaining -= free_length
            total_points += 1
            if charge_length <= 1e-9:
                continue
            weight = (
                path_config.cross_agent_cover_penalty_weight
                if segment.kind == "cover"
                else path_config.cross_agent_transit_penalty_weight
            )
            penalty = max(float(weight), 0.0) * charge_length
            segment_overlap += charge_length
            segment_penalty += penalty
            segment_hits += 1
            total_overlap += charge_length
            total_penalty += penalty
            total_hits += 1
            by_agent[owner] = by_agent.get(owner, 0.0) + charge_length
            by_kind[segment.kind] = by_kind.get(segment.kind, 0.0) + charge_length
        if annotate:
            _annotate_cross_agent(segment, segment_overlap, segment_penalty, segment_hits, max(len(points) - 1, 0), False)
    return CrossAgentOverlapScore(
        overlap_length=total_overlap,
        penalty=total_penalty,
        hit_ratio=total_hits / max(total_points, 1),
        sampled_point_count=total_points,
        hit_count=total_hits,
        overlap_by_agent=by_agent,
        overlap_by_kind=by_kind,
    )


def score_repeat_overlap(
    candidate_segments: Sequence[PathSegmentSpec],
    existing_segments: Sequence[PathSegmentSpec],
    path_config: PathPlanningConfig,
    penalty_weight: float,
    annotate: bool = False,
) -> RepeatOverlapScore:
    grid = max(float(path_config.shared_resource_grid_size), 1e-6)
    occupied = _occupied_cells(existing_segments, grid)
    if not occupied:
        if annotate:
            for segment in candidate_segments:
                _annotate_repeat(segment, 0.0, 0, 0)
        return RepeatOverlapScore(0.0, 0.0, 0.0, 0, 0)
    overlap = 0.0
    total_points = 0
    total_hits = 0
    for segment in candidate_segments:
        points = _sample_segment_points(segment, grid)
        if len(points) < 2:
            if annotate:
                _annotate_repeat(segment, 0.0, 0, len(points))
            continue
        occupied_hits = sum(1 for point in points if _quantized_point(point, grid) in occupied)
        ratio = occupied_hits / max(len(points), 1)
        segment_overlap = segment.length * ratio
        overlap += segment_overlap
        total_points += len(points)
        total_hits += occupied_hits
        if annotate:
            _annotate_repeat(segment, segment_overlap, occupied_hits, len(points))
    hit_ratio = total_hits / max(total_points, 1)
    return RepeatOverlapScore(
        overlap_length=overlap,
        penalty=max(float(penalty_weight), 0.0) * overlap,
        hit_ratio=hit_ratio,
        sampled_point_count=total_points,
        hit_count=total_hits,
    )


def repeat_overlap_metrics(
    segments: Sequence[PathSegmentSpec],
    path_config: PathPlanningConfig,
    penalty_weight: float,
    annotate: bool = False,
) -> RepeatOverlapScore:
    existing: List[PathSegmentSpec] = []
    total_overlap = 0.0
    total_hits = 0
    total_points = 0
    for segment in segments:
        score = score_repeat_overlap([segment], existing, path_config, penalty_weight, annotate=annotate)
        total_overlap += score.overlap_length
        total_hits += score.hit_count
        total_points += score.sampled_point_count
        existing.append(segment)
    return RepeatOverlapScore(
        overlap_length=total_overlap,
        penalty=max(float(penalty_weight), 0.0) * total_overlap,
        hit_ratio=total_hits / max(total_points, 1),
        sampled_point_count=total_points,
        hit_count=total_hits,
    )


def _annotate_repeat(segment: PathSegmentSpec, overlap: float, hits: int, samples: int) -> None:
    segment.metadata["repeat_overlap_length"] = f"{overlap:.6f}"
    segment.metadata["repeat_overlap_hit_count"] = str(hits)
    segment.metadata["repeat_overlap_sample_count"] = str(samples)
    segment.metadata["repeat_overlap_hit_ratio"] = f"{hits / max(samples, 1):.6f}"


def mark_cross_agent_unavoidable(segments: Sequence[PathSegmentSpec]) -> None:
    for segment in segments:
        if float(segment.metadata.get("cross_agent_overlap_length", "0") or 0.0) > 1e-9:
            segment.metadata["unavoidable_cross_agent_overlap"] = "true"


def cross_agent_overlap_metrics(
    agents: Dict[int, AgentPathPlan],
    ownership_map: CoverageOwnershipMap | None,
    path_config: PathPlanningConfig,
    config: PlannerConfig | None = None,
    annotate: bool = True,
) -> CrossAgentOverlapScore:
    total_overlap = 0.0
    total_penalty = 0.0
    total_points = 0
    total_hits = 0
    by_agent: Dict[int, float] = {}
    by_kind: Dict[str, float] = {}
    for agent_id, agent in sorted(agents.items()):
        score = score_cross_agent_ownership_overlap(
            agent.segments,
            agent_id,
            ownership_map,
            path_config,
            config=config,
            annotate=annotate,
        )
        total_overlap += score.overlap_length
        total_penalty += score.penalty
        total_points += score.sampled_point_count
        total_hits += score.hit_count
        for owner, length in score.overlap_by_agent.items():
            by_agent[owner] = by_agent.get(owner, 0.0) + length
        for kind, length in score.overlap_by_kind.items():
            by_kind[kind] = by_kind.get(kind, 0.0) + length
        agent.metrics["cross_agent_overlap_length"] = score.overlap_length
        agent.metrics["cross_agent_penalty_total"] = score.penalty
    return CrossAgentOverlapScore(
        overlap_length=total_overlap,
        penalty=total_penalty,
        hit_ratio=total_hits / max(total_points, 1),
        sampled_point_count=total_points,
        hit_count=total_hits,
        overlap_by_agent=by_agent,
        overlap_by_kind=by_kind,
    )


def _annotate_cross_agent(segment: PathSegmentSpec, overlap: float, penalty: float, hits: int, samples: int, unavoidable: bool) -> None:
    segment.metadata["cross_agent_overlap_length"] = f"{overlap:.6f}"
    segment.metadata["cross_agent_penalty"] = f"{penalty:.6f}"
    segment.metadata["cross_agent_overlap_hit_count"] = str(hits)
    segment.metadata["cross_agent_overlap_sample_count"] = str(samples)
    segment.metadata["cross_agent_overlap_hit_ratio"] = f"{hits / max(samples, 1):.6f}"
    if unavoidable:
        segment.metadata["unavoidable_cross_agent_overlap"] = "true"
    elif segment.metadata.get("unavoidable_cross_agent_overlap") != "true":
        segment.metadata.pop("unavoidable_cross_agent_overlap", None)


def _initial_escape_distance(
    candidate_segments: Sequence[PathSegmentSpec],
    agent_id: int,
    ownership_map: CoverageOwnershipMap,
    grid: float,
    path_config: PathPlanningConfig,
    config: PlannerConfig | None,
) -> float:
    first_point = None
    for segment in candidate_segments:
        points = _segment_points(segment)
        if points:
            first_point = points[0]
            break
    if first_point is None:
        return 0.0
    owner = ownership_map.owner_by_cell.get(_quantized_point(first_point, grid))
    if owner is None or owner == agent_id:
        return 0.0
    if path_config.cross_agent_initial_escape_free_distance is not None:
        return max(float(path_config.cross_agent_initial_escape_free_distance), 0.0)
    if config is None:
        return 0.0
    return max(config.footprint.length_lf, 2.0 * config.fleet.min_turn_radius)


def collect_resource_windows(agents: Dict[int, AgentPathPlan]) -> List[ResourceWindow]:
    windows: List[ResourceWindow] = []
    for agent_id, agent in agents.items():
        for segment in agent.segments:
            resource_id = segment.metadata.get("resource_id")
            if not resource_id:
                continue
            start, end = segment_time_bounds(segment)
            windows.append(
                ResourceWindow(
                    resource_id=resource_id,
                    start=start,
                    end=end,
                    agent_id=agent_id,
                    kind=segment.kind,
                    segment_id=segment.segment_id,
                )
            )
    return windows


def shared_resource_metrics(
    agents: Dict[int, AgentPathPlan],
    separation_time: float = 0.0,
) -> Dict[str, float]:
    grouped: Dict[str, List[ResourceWindow]] = {}
    for window in collect_resource_windows(agents):
        grouped.setdefault(window.resource_id, []).append(window)
    shared = {resource: windows for resource, windows in grouped.items() if len(windows) > 1}
    conflicts = 0
    for windows in shared.values():
        ordered = sorted(windows, key=lambda item: (item.start, item.end, item.agent_id))
        for first, second in zip(ordered[:-1], ordered[1:]):
            if second.start < first.end + max(separation_time, 0.0) - 1e-9:
                conflicts += 1
    return {
        "shared_resource_count": float(len(shared)),
        "spatial_overlap_reuse_count": float(sum(len(windows) - 1 for windows in shared.values())),
        "true_time_conflict_count": float(conflicts),
        "resource_window_count": float(sum(len(windows) for windows in grouped.values())),
    }


def segment_time_bounds(segment: PathSegmentSpec) -> Tuple[float, float]:
    times = [float(waypoint.time) for waypoint in segment.waypoints if waypoint.time is not None]
    if not times:
        return (0.0, 0.0)
    return (min(times), max(times))


def all_segments(agents: Dict[int, AgentPathPlan]) -> List[PathSegmentSpec]:
    return [segment for agent in agents.values() for segment in agent.segments]


def _occupied_cells(segments: Sequence[PathSegmentSpec], grid: float) -> set[str]:
    cells = set()
    for segment in segments:
        for point in _sample_segment_points(segment, grid):
            cells.add(_quantized_point(point, grid))
    return cells


def _sample_segment_points(segment: PathSegmentSpec, spacing: float) -> List[Tuple[float, float]]:
    points = _segment_points(segment)
    if len(points) < 2:
        return points
    sampled: List[Tuple[float, float]] = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        count = max(int(math.ceil(distance / max(spacing, 1e-6))), 1)
        for idx in range(1, count + 1):
            alpha = idx / count
            sampled.append((start[0] + alpha * (end[0] - start[0]), start[1] + alpha * (end[1] - start[1])))
    return sampled


def _segment_points(segment: PathSegmentSpec) -> List[Tuple[float, float]]:
    return [(float(waypoint.x), float(waypoint.y)) for waypoint in segment.waypoints]


def _quantized_point(point: Tuple[float, float], grid: float) -> str:
    return f"{round(point[0] / grid):d}_{round(point[1] / grid):d}"


def _dominant_axis(start: Tuple[float, float], end: Tuple[float, float]) -> str:
    return "x" if abs(end[0] - start[0]) >= abs(end[1] - start[1]) else "y"
