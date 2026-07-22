"""Continuous responsibility geometry and coverage certificates for CROWN."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, cos, hypot, sin
from typing import Iterable, Optional, Sequence, Tuple

from shapely import constrained_delaunay_triangles, make_valid, set_precision
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

from ...schema import PlannerConfig, Pose2D
from ..decomposition import choose_region_sweep_axis
from ..types import (
    CoveragePass,
    DecomposedRegion,
    ObstacleField,
    PathPlanningConfig,
    RegionCoveragePattern,
)


_TOL = 1.0e-9


@dataclass(frozen=True)
class CrownResponsibilityCertificate:
    """Area identities proving a continuous exact-once responsibility partition."""

    free_space_area: float
    responsibility_area: float
    gap_area: float
    spill_area: float
    overlap_area: float
    tolerance: float
    cell_count: int

    @property
    def valid(self) -> bool:
        return (
            self.gap_area <= self.tolerance
            and self.spill_area <= self.tolerance
            and self.overlap_area <= self.tolerance
        )


@dataclass(frozen=True)
class CrownCoverageCertificate:
    """Continuous under-approximation certificate for one coverage mode."""

    responsibility_area: float
    covered_area: float
    missing_area: float
    coverage_fraction: float
    tolerance: float

    @property
    def valid(self) -> bool:
        return self.missing_area <= self.tolerance


def _polygonal_parts(geometry: object) -> Tuple[Polygon, ...]:
    if isinstance(geometry, Polygon):
        return (geometry,) if geometry.area > _TOL else ()
    if isinstance(geometry, MultiPolygon):
        return tuple(part for part in geometry.geoms if part.area > _TOL)
    if isinstance(geometry, GeometryCollection) or hasattr(geometry, "geoms"):
        return tuple(
            part
            for item in geometry.geoms
            for part in _polygonal_parts(item)
        )
    return ()


def _precision(config: PlannerConfig) -> float:
    scale = max(config.mission.area_length_x, config.mission.area_length_y, 1.0)
    return max(scale * 1.0e-10, 1.0e-9)


def _clean_polygon(points: Sequence[Tuple[float, float]], precision: float) -> object:
    if len(points) < 3:
        return GeometryCollection()
    return set_precision(make_valid(Polygon(points)), precision)


def _free_space_geometry(
    config: PlannerConfig,
    obstacle_field: Optional[ObstacleField],
    precision: float,
) -> object:
    mission = set_precision(
        Polygon(
            (
                (0.0, 0.0),
                (config.mission.area_length_x, 0.0),
                (config.mission.area_length_x, config.mission.area_length_y),
                (0.0, config.mission.area_length_y),
            )
        ),
        precision,
    )
    if obstacle_field is None or not obstacle_field.inflated_obstacles:
        return mission
    obstacles = tuple(
        _clean_polygon(obstacle.polygon, precision)
        for obstacle in obstacle_field.inflated_obstacles
        if len(obstacle.polygon) >= 3
    )
    forbidden = unary_union(obstacles).intersection(mission) if obstacles else GeometryCollection()
    return set_precision(make_valid(mission.difference(forbidden)), precision)


def _ring_critical_x_coordinates(coordinates: Sequence[Tuple[float, float]]) -> set[float]:
    """Return vertical BCD events, not every polygon-approximation vertex.

    Ellipses and circles enter the planner as accurate many-sided polygons.
    Cutting at every sampled vertex creates dozens of sub-footprint slivers.
    Boustrophedon connectivity changes only at local x extrema (including a
    vertical plateau), so intermediate samples must remain on the curved cell
    boundary instead of becoming responsibility boundaries.
    """

    points = [(float(x), float(y)) for x, y in coordinates]
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    count = len(points)
    if count == 0:
        return set()
    values = {min(x for x, _ in points), max(x for x, _ in points)}
    tolerance = 1.0e-12
    for index, (x, _) in enumerate(points):
        previous = (index - 1) % count
        while previous != index and abs(points[previous][0] - x) <= tolerance:
            previous = (previous - 1) % count
        following = (index + 1) % count
        while following != index and abs(points[following][0] - x) <= tolerance:
            following = (following + 1) % count
        if previous == index or following == index:
            values.add(x)
            continue
        left_delta = points[previous][0] - x
        right_delta = points[following][0] - x
        if left_delta * right_delta >= -tolerance:
            values.add(x)
    return values


def _boundary_x_coordinates(geometry: object) -> Tuple[float, ...]:
    values = set()
    for polygon in _polygonal_parts(geometry):
        values.update(_ring_critical_x_coordinates(tuple(polygon.exterior.coords)))
        for ring in polygon.interiors:
            values.update(_ring_critical_x_coordinates(tuple(ring.coords)))
    return tuple(sorted(values))


def _obstacle_event_x_coordinates(
    free_space: object,
    obstacle_field: Optional[ObstacleField],
) -> Tuple[float, ...]:
    if obstacle_field is None:
        return _boundary_x_coordinates(free_space)
    values = set()
    protected = set()
    for polygon in _polygonal_parts(free_space):
        exterior = {float(x) for x, _ in polygon.exterior.coords}
        values.update(exterior)
        protected.update(exterior)
        for ring in polygon.interiors:
            values.update(float(x) for x, _ in ring.coords)

    def snapped_x(raw_x: float) -> Optional[float]:
        if not values:
            return None
        candidate = min(values, key=lambda value: abs(value - raw_x))
        scale = max(abs(raw_x), abs(candidate), 1.0)
        return candidate if abs(candidate - raw_x) <= scale * 1.0e-8 else None

    smooth_samples = set()
    for obstacle in obstacle_field.inflated_obstacles:
        if obstacle.kind in {"circle", "ellipse"}:
            for raw_x, _ in obstacle.polygon:
                candidate = snapped_x(float(raw_x))
                if candidate is not None:
                    smooth_samples.add(candidate)
            for raw_x in _ring_critical_x_coordinates(tuple(obstacle.polygon)):
                candidate = snapped_x(raw_x)
                if candidate is not None:
                    protected.add(candidate)
        else:
            # Polygonal corners are real geometry rather than smooth-shape
            # sampling points.  Retaining their events keeps the resulting
            # cells simple enough for exact polygon sweep modes.
            for raw_x, _ in obstacle.polygon:
                candidate = snapped_x(float(raw_x))
                if candidate is not None:
                    protected.add(candidate)
    return tuple(sorted(values.difference(smooth_samples).union(protected)))


def _convex_parts(polygon: Polygon, tolerance: float) -> Tuple[Polygon, ...]:
    if (
        not polygon.interiors
        and polygon.convex_hull.area - polygon.area <= tolerance
    ):
        return (polygon,)
    triangles = constrained_delaunay_triangles(polygon)
    result = []
    for triangle in _polygonal_parts(triangles):
        clipped = make_valid(triangle.intersection(polygon))
        for part in _polygonal_parts(clipped):
            if part.area > tolerance:
                result.append(part)
    return tuple(result)


def _vertical_bcd_cells(
    free_space: object,
    config: PlannerConfig,
    precision: float,
    tolerance: float,
    obstacle_field: Optional[ObstacleField] = None,
) -> Tuple[Polygon, ...]:
    """Polygonize exact vertical-connectivity event slabs."""

    critical_x = _obstacle_event_x_coordinates(free_space, obstacle_field)
    height = config.mission.area_length_y
    padding = max(config.mission.area_length_x, height, 1.0)
    cuts = []
    for x in critical_x[1:-1]:
        line = LineString(((x, -padding), (x, height + padding)))
        clipped = line.intersection(free_space)
        if not clipped.is_empty:
            cuts.append(clipped)
    network = unary_union((free_space.boundary, *cuts))
    raw_cells = []
    convexify_polygonal_map = bool(obstacle_field) and not any(
        obstacle.kind in {"circle", "ellipse"}
        for obstacle in obstacle_field.inflated_obstacles
    )
    for candidate in polygonize(network):
        if not free_space.covers(candidate.representative_point()):
            continue
        clipped = set_precision(make_valid(candidate.intersection(free_space)), precision)
        for polygon in _polygonal_parts(clipped):
            if polygon.interiors or (
                convexify_polygonal_map
                and polygon.convex_hull.area - polygon.area > tolerance
            ):
                raw_cells.extend(_convex_parts(polygon, tolerance))
            elif polygon.area > tolerance:
                raw_cells.append(polygon)
    if not raw_cells:
        for polygon in _polygonal_parts(free_space):
            raw_cells.extend(_convex_parts(polygon, tolerance))
    return tuple(
        sorted(
            raw_cells,
            key=lambda polygon: (
                round(polygon.bounds[0], 12),
                round(polygon.bounds[1], 12),
                round(polygon.bounds[2], 12),
                round(polygon.bounds[3], 12),
                round(polygon.area, 12),
            ),
        )
    )


def _regions_from_polygons(
    polygons: Sequence[Polygon],
    preferred_axis: str,
) -> Tuple[DecomposedRegion, ...]:
    regions = []
    for index, polygon in enumerate(polygons):
        coordinates = [(float(x), float(y)) for x, y in tuple(polygon.exterior.coords)[:-1]]
        x0, y0, x1, y1 = (float(value) for value in polygon.bounds)
        centroid = polygon.centroid
        is_convex = (
            not polygon.interiors
            and polygon.convex_hull.area - polygon.area <= max(polygon.area * 1.0e-10, _TOL)
        )
        regions.append(
            DecomposedRegion(
                region_id=f"crown_cell_{index}",
                bounds=(x0, y0, x1, y1),
                polygon=coordinates,
                center=(float(centroid.x), float(centroid.y)),
                area=float(polygon.area),
                preferred_axis=preferred_axis,
                source_algorithm="crown_continuous_vertical_bcd",
                metadata={
                    "crown_continuous_responsibility": "true",
                    "crown_geometry_role": "exact_polygon_not_envelope",
                    "shape_class": (
                        "convex_polygon" if is_convex else "boustrophedon_cell"
                    ),
                    "convexity_status": "convex" if is_convex else "concave",
                },
            )
        )
    for index, left in enumerate(polygons):
        for right_index in range(index + 1, len(polygons)):
            right = polygons[right_index]
            shared = left.boundary.intersection(right.boundary)
            if shared.length <= _TOL:
                continue
            regions[index].neighbors.append(regions[right_index].region_id)
            regions[right_index].neighbors.append(regions[index].region_id)
    return tuple(regions)


def _responsibility_certificate(
    free_space: object,
    polygons: Sequence[Polygon],
    tolerance: float,
) -> CrownResponsibilityCertificate:
    union = unary_union(tuple(polygons)) if polygons else GeometryCollection()
    free_area = float(free_space.area)
    responsibility_area = float(sum(polygon.area for polygon in polygons))
    certificate = CrownResponsibilityCertificate(
        free_space_area=free_area,
        responsibility_area=responsibility_area,
        gap_area=float(free_space.difference(union).area),
        spill_area=float(union.difference(free_space).area),
        overlap_area=max(0.0, responsibility_area - float(union.area)),
        tolerance=tolerance,
        cell_count=len(polygons),
    )
    if not certificate.valid:
        raise ValueError(
            "CROWN continuous responsibility partition failed: "
            f"gap={certificate.gap_area:.12g}, "
            f"spill={certificate.spill_area:.12g}, "
            f"overlap={certificate.overlap_area:.12g}, "
            f"tolerance={certificate.tolerance:.12g}"
        )
    return certificate


def build_continuous_responsibility_regions(
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: Optional[ObstacleField],
    fallback_regions: Sequence[DecomposedRegion] = (),
) -> Tuple[Tuple[DecomposedRegion, ...], CrownResponsibilityCertificate]:
    """Build/certify an interior-disjoint continuous partition of free space."""

    precision = _precision(config)
    free_space = _free_space_geometry(config, obstacle_field, precision)
    tolerance = max(float(free_space.area) * 1.0e-10, precision * precision * 16.0)
    if obstacle_field is not None and obstacle_field.inflated_obstacles:
        polygons = _vertical_bcd_cells(
            free_space,
            config,
            precision,
            tolerance,
            obstacle_field,
        )
        regions = _regions_from_polygons(
            polygons,
            choose_region_sweep_axis(config, path_config),
        )
    else:
        regions = tuple(fallback_regions)
        polygons = tuple(
            part
            for region in regions
            for part in _polygonal_parts(_clean_polygon(region.polygon, precision))
        )
    certificate = _responsibility_certificate(free_space, polygons, tolerance)
    for region in regions:
        region.metadata.update(
            {
                "crown_partition_gap_area": f"{certificate.gap_area:.12g}",
                "crown_partition_spill_area": f"{certificate.spill_area:.12g}",
                "crown_partition_overlap_area": f"{certificate.overlap_area:.12g}",
                "crown_partition_tolerance": f"{certificate.tolerance:.12g}",
            }
        )
    return regions, certificate


def certify_continuous_responsibility_regions(
    config: PlannerConfig,
    obstacle_field: Optional[ObstacleField],
    regions: Sequence[DecomposedRegion],
) -> CrownResponsibilityCertificate:
    """Re-certify a partition after topology-preserving cell refinement."""

    precision = _precision(config)
    free_space = _free_space_geometry(config, obstacle_field, precision)
    polygons = tuple(
        part
        for region in regions
        for part in _polygonal_parts(_clean_polygon(region.polygon, precision))
    )
    tolerance = max(float(free_space.area) * 1.0e-10, precision * precision * 16.0)
    return _responsibility_certificate(free_space, polygons, tolerance)


def _pass_swept_rectangle(
    start: Pose2D,
    end: Pose2D,
    footprint_length: float,
    footprint_width: float,
) -> Polygon:
    dx, dy = end.x - start.x, end.y - start.y
    distance = hypot(dx, dy)
    heading = atan2(dy, dx) if distance > _TOL else start.psi
    ux, uy = cos(heading), sin(heading)
    half_length = 0.5 * footprint_length
    extended_start = (start.x - half_length * ux, start.y - half_length * uy)
    extended_end = (end.x + half_length * ux, end.y + half_length * uy)
    if hypot(extended_end[0] - extended_start[0], extended_end[1] - extended_start[1]) <= _TOL:
        vx, vy = -uy, ux
        half_width = 0.5 * footprint_width
        return Polygon(
            (
                (start.x - half_length * ux - half_width * vx, start.y - half_length * uy - half_width * vy),
                (start.x + half_length * ux - half_width * vx, start.y + half_length * uy - half_width * vy),
                (start.x + half_length * ux + half_width * vx, start.y + half_length * uy + half_width * vy),
                (start.x - half_length * ux + half_width * vx, start.y - half_length * uy + half_width * vy),
            )
        )
    return LineString((extended_start, extended_end)).buffer(
        0.5 * footprint_width,
        cap_style="flat",
        join_style="mitre",
    )


def certify_continuous_pattern_coverage(
    region: DecomposedRegion,
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
) -> CrownCoverageCertificate:
    """Prove coverage using the exact swept rectangle of every straight pass."""

    precision = _precision(config)
    responsibility = _clean_polygon(region.polygon, precision)
    footprints = tuple(
        _pass_swept_rectangle(
            coverage_pass.start_pose,
            coverage_pass.end_pose,
            config.footprint.length_lf,
            config.footprint.width_wf,
        )
        for coverage_pass in pattern.passes
    )
    covered = unary_union(footprints) if footprints else GeometryCollection()
    missing = responsibility.difference(covered)
    area = float(responsibility.area)
    missing_area = float(missing.area)
    tolerance = max(area * 1.0e-9, precision * precision * 16.0)
    covered_area = max(0.0, area - missing_area)
    return CrownCoverageCertificate(
        responsibility_area=area,
        covered_area=covered_area,
        missing_area=missing_area,
        coverage_fraction=(1.0 if area <= tolerance else covered_area / area),
        tolerance=tolerance,
    )


def repair_continuous_pattern_coverage(
    region: DecomposedRegion,
    pattern: RegionCoveragePattern,
    config: PlannerConfig,
    *,
    maximum_relative_missing: float = 0.01,
    maximum_repair_passes: int = 4,
) -> RegionCoveragePattern:
    """Add local free-space passes for small curved-boundary corner gaps.

    Axis-aligned rectangular footprints can leave a tiny triangular corner at
    an obstacle tangent even when the ordinary sweep covers more than 99.9% of
    a BCD cell.  Each added centerline lies inside the exact missing component;
    downstream hull, dynamics, obstacle and transition validators remain
    authoritative and reject an unsafe repair.
    """

    if not pattern.passes or maximum_repair_passes <= 0:
        return pattern
    precision = _precision(config)
    responsibility = _clean_polygon(region.polygon, precision)
    tolerance = max(float(responsibility.area) * 1.0e-10, precision * precision * 16.0)
    repaired = pattern
    added = 0
    while added < maximum_repair_passes:
        footprints = tuple(
            _pass_swept_rectangle(
                coverage_pass.start_pose,
                coverage_pass.end_pose,
                config.footprint.length_lf,
                config.footprint.width_wf,
            )
            for coverage_pass in repaired.passes
        )
        covered = unary_union(footprints) if footprints else GeometryCollection()
        missing = make_valid(responsibility.difference(covered))
        missing_area = float(missing.area)
        if missing_area <= tolerance:
            return repaired
        if missing_area > max(
            float(responsibility.area) * maximum_relative_missing,
            tolerance,
        ):
            return pattern
        parts = sorted(_polygonal_parts(missing), key=lambda part: -part.area)
        if not parts:
            return pattern
        component = parts[0]
        x0, y0, x1, y1 = (float(value) for value in component.bounds)
        point = component.representative_point()
        scan_axis = repaired.scan_axis if repaired.scan_axis in {"x", "y"} else region.preferred_axis
        if scan_axis == "y":
            center = float(point.x)
            vehicle = config.vehicle_footprint
            half_hull_along = (
                max(config.fleet.min_turn_radius, 0.0)
                + (0.0 if vehicle is None else 0.5 * vehicle.length)
            )
            y0 = max(y0, half_hull_along)
            y1 = min(y1, config.mission.area_length_y - half_hull_along)
            start = Pose2D(center, y0, 0.5 * 3.141592653589793)
            end = Pose2D(center, y1, 0.5 * 3.141592653589793)
            length = max(y1 - y0, 0.0)
            center_coordinate = center
        else:
            center = float(point.y)
            vehicle = config.vehicle_footprint
            half_hull_along = (
                max(config.fleet.min_turn_radius, 0.0)
                + (0.0 if vehicle is None else 0.5 * vehicle.length)
            )
            x0 = max(x0, half_hull_along)
            x1 = min(x1, config.mission.area_length_x - half_hull_along)
            start = Pose2D(x0, center, 0.0)
            end = Pose2D(x1, center, 0.0)
            length = max(x1 - x0, 0.0)
            center_coordinate = center
        repair_pass = CoveragePass(
            pass_id=f"{pattern.pattern_id}:continuous-repair:{added}",
            region_id=region.region_id,
            sequence_index=0,
            scan_axis=scan_axis,
            start_pose=start,
            end_pose=end,
            center_coordinate=center_coordinate,
            width=config.footprint.width_wf,
            length=length,
        )
        existing = list(repaired.passes)
        insert_front = abs(center_coordinate - existing[0].center_coordinate) <= abs(
            center_coordinate - existing[-1].center_coordinate
        )
        neighbor = existing[0] if insert_front else existing[-1]
        repair_dx = repair_pass.end_pose.x - repair_pass.start_pose.x
        repair_dy = repair_pass.end_pose.y - repair_pass.start_pose.y
        neighbor_dx = neighbor.end_pose.x - neighbor.start_pose.x
        neighbor_dy = neighbor.end_pose.y - neighbor.start_pose.y
        if repair_dx * neighbor_dx + repair_dy * neighbor_dy >= 0.0:
            reverse_heading = atan2(-repair_dy, -repair_dx)
            repair_pass = replace(
                repair_pass,
                start_pose=Pose2D(
                    repair_pass.end_pose.x,
                    repair_pass.end_pose.y,
                    reverse_heading,
                ),
                end_pose=Pose2D(
                    repair_pass.start_pose.x,
                    repair_pass.start_pose.y,
                    reverse_heading,
                ),
            )
        passes = (
            [repair_pass] + existing if insert_front else existing + [repair_pass]
        )
        passes = [
            replace(coverage_pass, sequence_index=index)
            for index, coverage_pass in enumerate(passes)
        ]
        repaired = replace(
            repaired,
            passes=passes,
            entry_pose=passes[0].start_pose,
            exit_pose=passes[-1].end_pose,
            coverage_length=repaired.coverage_length + length,
            total_length=repaired.total_length + length,
            metadata={
                **repaired.metadata,
                "crown_continuous_coverage_repair": "true",
                "crown_continuous_coverage_repair_passes": str(added + 1),
            },
        )
        added += 1
    return repaired
