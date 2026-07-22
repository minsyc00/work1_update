from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from ..geometry import mean_heading
from ..schema import PlannerConfig
from .coverage import RectangularCoverageModel
from .obstacles import obstacle_bounds, point_in_any_obstacle, polygon_collides_with_obstacles
from .types import CompositeFreeSpaceRegion, DecomposedRegion, FreeSpaceCell, ObstacleField, PathPlanningConfig


def choose_region_sweep_axis(config: PlannerConfig, path_config: PathPlanningConfig | None = None) -> str:
    if path_config is not None and path_config.sweep_axis in {"x", "y"}:
        return path_config.sweep_axis
    lx = config.mission.area_length_x
    ly = config.mission.area_length_y
    if abs(lx - ly) > 1e-6:
        return "x" if lx >= ly else "y"
    avg_heading = mean_heading(state.psi for state in config.fleet.initial_states_3dof)
    return "x" if abs(math.cos(avg_heading)) >= abs(math.sin(avg_heading)) else "y"


def _region_from_bounds(
    region_id: str,
    bounds: Tuple[float, float, float, float],
    preferred_axis: str,
    source_algorithm: str,
    metadata: Dict[str, str] | None = None,
) -> DecomposedRegion:
    x_min, y_min, x_max, y_max = bounds
    polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    area = max(x_max - x_min, 0.0) * max(y_max - y_min, 0.0)
    return DecomposedRegion(
        region_id=region_id,
        bounds=bounds,
        polygon=polygon,
        center=((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
        area=area,
        preferred_axis=preferred_axis,
        source_algorithm=source_algorithm,
        metadata=dict(metadata or {}),
    )


def decompose_rectangular_area(
    config: PlannerConfig,
    path_config: PathPlanningConfig | None = None,
) -> List[DecomposedRegion]:
    path_config = path_config or PathPlanningConfig.from_planner_config(config)
    axis = choose_region_sweep_axis(config, path_config)
    model = RectangularCoverageModel.from_config(config)
    lx = config.mission.area_length_x
    ly = config.mission.area_length_y
    cross_width = ly if axis == "x" else lx
    strip_count = 1 if cross_width <= model.width else int(math.ceil((cross_width - model.width) / model.strip_spacing) + 1)
    agent_count = max(config.fleet.num_agents or 1, 1)
    min_regions = max(agent_count, strip_count)
    balanced_multiple = int(math.ceil(min_regions / agent_count) * agent_count)
    max_regions = max(min_regions, agent_count * max(path_config.max_regions_per_agent, 1))
    region_count = min(max_regions, balanced_multiple)
    region_count = max(1, region_count)

    regions: List[DecomposedRegion] = []
    if axis == "x":
        edges = [ly * idx / region_count for idx in range(region_count + 1)]
        for idx in range(region_count):
            regions.append(
                _region_from_bounds(
                    f"region_{idx}",
                    (0.0, edges[idx], lx, edges[idx + 1]),
                    preferred_axis="x",
                    source_algorithm="rectangular_band_decomposition",
                    metadata={"band_index": str(idx), "band_count": str(region_count)},
                )
            )
    else:
        edges = [lx * idx / region_count for idx in range(region_count + 1)]
        for idx in range(region_count):
            regions.append(
                _region_from_bounds(
                    f"region_{idx}",
                    (edges[idx], 0.0, edges[idx + 1], ly),
                    preferred_axis="y",
                    source_algorithm="rectangular_band_decomposition",
                    metadata={"band_index": str(idx), "band_count": str(region_count)},
                )
            )
    _populate_axis_aligned_neighbors(regions)
    return regions


def decompose_obstacle_aware_area(
    config: PlannerConfig,
    path_config: PathPlanningConfig | None,
    obstacle_field: ObstacleField,
) -> List[DecomposedRegion]:
    path_config = path_config or PathPlanningConfig.from_planner_config(config)
    if path_config.enable_large_convex_region_decomposition:
        large_regions = build_large_convex_free_space_regions(config, path_config, obstacle_field)
        if large_regions and _large_regions_are_valid(large_regions, obstacle_field):
            return large_regions
    cells = build_free_space_cells(config, path_config, obstacle_field)
    regions = [
        DecomposedRegion(
            region_id=cell.cell_id,
            bounds=cell.bounds,
            polygon=list(cell.polygon),
            center=cell.center,
            area=cell.area,
            preferred_axis=cell.preferred_axis,
            source_algorithm=cell.source_algorithm,
            neighbors=list(cell.neighbors),
            metadata=dict(cell.metadata),
        )
        for cell in cells
    ]
    _populate_axis_aligned_neighbors(regions)
    for region in regions:
        region.metadata["static_obstacle_aware"] = "true"
    return regions


def build_large_convex_free_space_regions(
    config: PlannerConfig,
    path_config: PathPlanningConfig | None,
    obstacle_field: ObstacleField,
) -> List[DecomposedRegion]:
    """Build large obstacle-free rectangular regions before falling back to cells.

    The decomposition is intentionally conservative: it only emits rectangular
    blocks whose full polygon has been checked against inflated obstacles. This
    keeps the main sweep planner working with large simple regions without
    relying on an external polygon-boolean dependency.
    """

    path_config = path_config or PathPlanningConfig.from_planner_config(config)
    coarse_cells = _build_obstacle_aligned_free_cells(config, path_config, obstacle_field)
    if not coarse_cells:
        return []
    x_edges = sorted({edge for cell in coarse_cells for edge in (cell.bounds[0], cell.bounds[2])})
    y_edges = sorted({edge for cell in coarse_cells for edge in (cell.bounds[1], cell.bounds[3])})
    if len(x_edges) < 2 or len(y_edges) < 2:
        return []
    cell_by_index: Dict[Tuple[int, int], FreeSpaceCell] = {}
    for cell in coarse_cells:
        ix = _edge_index(x_edges, cell.bounds[0])
        iy = _edge_index(y_edges, cell.bounds[1])
        if ix is not None and iy is not None:
            cell_by_index[(ix, iy)] = cell

    unvisited = set(cell_by_index)
    groups: List[List[FreeSpaceCell]] = []
    while unvisited:
        best_indices: set[Tuple[int, int]] | None = None
        best_key: Tuple[float, float, int, int] | None = None
        for seed in sorted(unvisited):
            indices = _grow_large_rect_indices(seed, unvisited, x_edges, y_edges)
            bounds = _indices_bounds(indices, x_edges, y_edges)
            area = max(bounds[2] - bounds[0], 0.0) * max(bounds[3] - bounds[1], 0.0)
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            aspect_balance = min(width, height) / max(max(width, height), 1e-9)
            key = (area, aspect_balance, len(indices), -seed[0] * 10_000 - seed[1])
            if best_key is None or key > best_key:
                best_key = key
                best_indices = indices
        if not best_indices:
            seed = min(unvisited)
            best_indices = {seed}
        groups.append([cell_by_index[index] for index in sorted(best_indices)])
        unvisited.difference_update(best_indices)

    mission_area = max(config.mission.area_length_x * config.mission.area_length_y, 1e-9)
    max_area = mission_area * max(float(path_config.large_region_max_area_fraction), 1e-6)
    min_width = max(config.footprint.width_wf * max(path_config.large_region_min_width_factor, 0.0), 1e-6)
    regions: List[DecomposedRegion] = []
    serial = 0
    for group in groups:
        bounds = _cell_group_bounds(group)
        for split_bounds in _split_large_region_bounds(bounds, max_area):
            width = split_bounds[2] - split_bounds[0]
            height = split_bounds[3] - split_bounds[1]
            if width <= 1e-9 or height <= 1e-9:
                continue
            polygon = [
                (split_bounds[0], split_bounds[1]),
                (split_bounds[2], split_bounds[1]),
                (split_bounds[2], split_bounds[3]),
                (split_bounds[0], split_bounds[3]),
            ]
            if polygon_collides_with_obstacles(_shrink_axis_aligned_polygon(polygon), obstacle_field, inflated=True):
                continue
            preferred_axis = "x" if width >= height else "y"
            shape_class = "rectangle" if min(width, height) + 1e-9 >= min_width else "fallback_cell"
            area = width * height
            support_span = height if preferred_axis == "x" else width
            regions.append(
                DecomposedRegion(
                    region_id=f"large_region_{serial}",
                    bounds=split_bounds,
                    polygon=polygon,
                    center=((split_bounds[0] + split_bounds[2]) / 2.0, (split_bounds[1] + split_bounds[3]) / 2.0),
                    area=area,
                    preferred_axis=preferred_axis,
                    source_algorithm="large_convex_free_space_decomposition",
                    metadata={
                        "static_obstacle_aware": "true",
                        "convex_region_decomposition": "true",
                        "shape_class": shape_class,
                        "dominant_scan_axis": preferred_axis,
                        "support_span": f"{support_span:.6f}",
                        "area_priority": f"{area / mission_area:.6f}",
                        "source_cell_count": str(len(group)),
                        "source_region_ids": ",".join(cell.cell_id for cell in group),
                        "decomposition_fallback_reason": "" if shape_class != "fallback_cell" else "below_large_region_min_width",
                    },
                )
            )
            serial += 1
    _populate_axis_aligned_neighbors(regions)
    return regions


def build_composite_free_space_regions(
    cells: Sequence[FreeSpaceCell],
    config: PlannerConfig,
    path_config: PathPlanningConfig | None = None,
    obstacle_field: ObstacleField | None = None,
) -> List[CompositeFreeSpaceRegion]:
    """Merge adjacent free cells into logical coverage regions.

    The composite region keeps the source cells as the true free-space support.
    Its bounding polygon is only a coarse envelope used by existing graphing and
    visualization code.
    """

    path_config = path_config or PathPlanningConfig.from_planner_config(config)
    if not cells:
        return []
    max_members = max(int(path_config.composite_max_member_cells), 1)
    mission_area = max(config.mission.area_length_x * config.mission.area_length_y, 1e-9)
    max_area = max(
        config.footprint.length_lf * config.footprint.width_wf,
        mission_area * max(float(path_config.composite_max_region_area_fraction), 1e-6),
    )
    cell_by_id = {cell.cell_id: cell for cell in cells}
    visited: set[str] = set()
    composites: List[CompositeFreeSpaceRegion] = []

    for seed in sorted(cells, key=lambda item: (item.preferred_axis, item.center[0], item.center[1], item.cell_id)):
        if seed.cell_id in visited:
            continue
        group: List[FreeSpaceCell] = []
        queued = {seed.cell_id}
        queue = [seed]
        axis = seed.preferred_axis
        area = 0.0
        sorted_cells = sorted(cells, key=lambda item: (item.center[0], item.center[1], item.cell_id))
        while queue:
            cell = queue.pop(0)
            queued.discard(cell.cell_id)
            if cell.cell_id in visited:
                continue
            if group and cell.preferred_axis != axis:
                continue
            if group and (len(group) >= max_members or area + cell.area > max_area):
                continue
            group.append(cell)
            visited.add(cell.cell_id)
            area += cell.area
            for neighbor_id in sorted(cell.neighbors):
                neighbor = cell_by_id.get(neighbor_id)
                if neighbor is None or neighbor.cell_id in visited or neighbor.cell_id in queued:
                    continue
                if neighbor.preferred_axis != axis:
                    continue
                if not _composite_cells_can_join(cell, neighbor, config, axis):
                    continue
                queue.append(neighbor)
                queued.add(neighbor.cell_id)
            for candidate in sorted_cells:
                if candidate.cell_id in visited or candidate.cell_id in queued:
                    continue
                if candidate.preferred_axis != axis:
                    continue
                if not _composite_cells_can_bridge_gap(cell, candidate, config, path_config, obstacle_field, axis):
                    continue
                queue.append(candidate)
                queued.add(candidate.cell_id)
        if group:
            composites.append(_composite_region_from_cells(len(composites), group, axis))

    _populate_composite_neighbors(composites)
    return composites


def _build_obstacle_aligned_free_cells(
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField,
) -> List[FreeSpaceCell]:
    lx = config.mission.area_length_x
    ly = config.mission.area_length_y
    min_size = path_config.min_free_cell_size or max(config.footprint.width_wf * 0.5, 1e-6)
    x_breaks = [0.0, lx]
    y_breaks = [0.0, ly]
    for obstacle in obstacle_field.inflated_obstacles:
        x_min, y_min, x_max, y_max = obstacle_bounds(obstacle)
        x_breaks.extend([max(0.0, min(lx, x_min)), max(0.0, min(lx, x_max))])
        y_breaks.extend([max(0.0, min(ly, y_min)), max(0.0, min(ly, y_max))])
        for x_value, y_value in _salient_obstacle_coordinates(obstacle):
            x_breaks.append(max(0.0, min(lx, x_value)))
            y_breaks.append(max(0.0, min(ly, y_value)))
    x_edges = _unique_sorted_edges(x_breaks, lx)
    y_edges = _unique_sorted_edges(y_breaks, ly)
    cells: List[FreeSpaceCell] = []
    serial = 0
    default_axis = choose_region_sweep_axis(config, path_config)
    for ix in range(len(x_edges) - 1):
        for iy in range(len(y_edges) - 1):
            x0, x1 = x_edges[ix], x_edges[ix + 1]
            y0, y1 = y_edges[iy], y_edges[iy + 1]
            if x1 - x0 < min_size or y1 - y0 < min_size:
                continue
            polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            if point_in_any_obstacle(center, obstacle_field, inflated=True):
                continue
            if polygon_collides_with_obstacles(_shrink_axis_aligned_polygon(polygon), obstacle_field, inflated=True):
                continue
            axis = "x" if (x1 - x0) >= (y1 - y0) else "y"
            if abs((x1 - x0) - (y1 - y0)) <= 1e-9:
                axis = default_axis
            cells.append(
                FreeSpaceCell(
                    cell_id=f"large_free_cell_{serial}",
                    bounds=(x0, y0, x1, y1),
                    polygon=polygon,
                    center=center,
                    area=(x1 - x0) * (y1 - y0),
                    preferred_axis=axis,
                    source_algorithm="large_convex_obstacle_aligned_cell",
                    obstacle_ids=[obstacle.obstacle_id for obstacle in obstacle_field.inflated_obstacles],
                    metadata={
                        "ix": str(ix),
                        "iy": str(iy),
                        "static_obstacle_aware": "true",
                        "convex_region_decomposition_seed": "true",
                        "free_area": f"{(x1 - x0) * (y1 - y0):.6f}",
                        "narrow_width": f"{min(x1 - x0, y1 - y0):.6f}",
                    },
                )
            )
            serial += 1
    _populate_free_space_neighbors(cells)
    return cells


def _edge_index(edges: Sequence[float], value: float) -> int | None:
    rounded = round(value, 9)
    for idx, edge in enumerate(edges[:-1]):
        if round(edge, 9) == rounded:
            return idx
    return None


def _grow_large_rect_indices(
    seed: Tuple[int, int],
    available: set[Tuple[int, int]],
    x_edges: Sequence[float],
    y_edges: Sequence[float],
) -> set[Tuple[int, int]]:
    ix0 = ix1 = seed[0]
    iy0 = iy1 = seed[1]
    selected = {seed}
    while True:
        candidates: List[Tuple[float, set[Tuple[int, int]], Tuple[int, int, int, int]]] = []
        if ix0 > 0:
            strip = {(ix0 - 1, iy) for iy in range(iy0, iy1 + 1)}
            if strip <= available:
                candidates.append((_indices_area(selected | strip, x_edges, y_edges), strip, (ix0 - 1, ix1, iy0, iy1)))
        if ix1 + 1 < len(x_edges) - 1:
            strip = {(ix1 + 1, iy) for iy in range(iy0, iy1 + 1)}
            if strip <= available:
                candidates.append((_indices_area(selected | strip, x_edges, y_edges), strip, (ix0, ix1 + 1, iy0, iy1)))
        if iy0 > 0:
            strip = {(ix, iy0 - 1) for ix in range(ix0, ix1 + 1)}
            if strip <= available:
                candidates.append((_indices_area(selected | strip, x_edges, y_edges), strip, (ix0, ix1, iy0 - 1, iy1)))
        if iy1 + 1 < len(y_edges) - 1:
            strip = {(ix, iy1 + 1) for ix in range(ix0, ix1 + 1)}
            if strip <= available:
                candidates.append((_indices_area(selected | strip, x_edges, y_edges), strip, (ix0, ix1, iy0, iy1 + 1)))
        if not candidates:
            break
        _, strip, bounds = max(candidates, key=lambda item: (item[0], len(item[1])))
        selected |= strip
        ix0, ix1, iy0, iy1 = bounds
    return selected


def _indices_bounds(
    indices: set[Tuple[int, int]],
    x_edges: Sequence[float],
    y_edges: Sequence[float],
) -> Tuple[float, float, float, float]:
    ix_values = [idx[0] for idx in indices]
    iy_values = [idx[1] for idx in indices]
    return (
        x_edges[min(ix_values)],
        y_edges[min(iy_values)],
        x_edges[max(ix_values) + 1],
        y_edges[max(iy_values) + 1],
    )


def _indices_area(indices: set[Tuple[int, int]], x_edges: Sequence[float], y_edges: Sequence[float]) -> float:
    x0, y0, x1, y1 = _indices_bounds(indices, x_edges, y_edges)
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def _cell_group_bounds(cells: Sequence[FreeSpaceCell]) -> Tuple[float, float, float, float]:
    return (
        min(cell.bounds[0] for cell in cells),
        min(cell.bounds[1] for cell in cells),
        max(cell.bounds[2] for cell in cells),
        max(cell.bounds[3] for cell in cells),
    )


def _split_large_region_bounds(
    bounds: Tuple[float, float, float, float],
    max_area: float,
) -> List[Tuple[float, float, float, float]]:
    x0, y0, x1, y1 = bounds
    width = max(x1 - x0, 0.0)
    height = max(y1 - y0, 0.0)
    area = width * height
    if area <= max(max_area, 1e-9):
        return [bounds]
    split_count = int(math.ceil(area / max(max_area, 1e-9)))
    result: List[Tuple[float, float, float, float]] = []
    if width >= height:
        edges = [x0 + width * idx / split_count for idx in range(split_count + 1)]
        result.extend((edges[idx], y0, edges[idx + 1], y1) for idx in range(split_count))
    else:
        edges = [y0 + height * idx / split_count for idx in range(split_count + 1)]
        result.extend((x0, edges[idx], x1, edges[idx + 1]) for idx in range(split_count))
    return result


def _large_regions_are_valid(regions: Sequence[DecomposedRegion], obstacle_field: ObstacleField) -> bool:
    if not regions:
        return False
    for idx, region in enumerate(regions):
        if region.area <= 1e-9:
            return False
        if polygon_collides_with_obstacles(_shrink_axis_aligned_polygon(region.polygon), obstacle_field, inflated=True):
            return False
        for other in regions[idx + 1 :]:
            if _bounds_overlap_area(region.bounds, other.bounds) > 1e-8:
                return False
    return True


def _bounds_overlap_area(first: Tuple[float, float, float, float], second: Tuple[float, float, float, float]) -> float:
    x_overlap = min(first[2], second[2]) - max(first[0], second[0])
    y_overlap = min(first[3], second[3]) - max(first[1], second[1])
    if x_overlap <= 0.0 or y_overlap <= 0.0:
        return 0.0
    return x_overlap * y_overlap


def build_free_space_cells(
    config: PlannerConfig,
    path_config: PathPlanningConfig | None,
    obstacle_field: ObstacleField,
) -> List[FreeSpaceCell]:
    path_config = path_config or PathPlanningConfig.from_planner_config(config)
    lx = config.mission.area_length_x
    ly = config.mission.area_length_y
    min_size = path_config.min_free_cell_size or max(config.footprint.width_wf * 0.5, 1e-6)
    x_breaks = [0.0, lx]
    y_breaks = [0.0, ly]
    grid_step = _free_space_grid_step(config, path_config)
    x_breaks.extend(_regular_breaks(lx, grid_step))
    y_breaks.extend(_regular_breaks(ly, grid_step))
    for obstacle in obstacle_field.inflated_obstacles:
        x_min, y_min, x_max, y_max = obstacle_bounds(obstacle)
        x_breaks.extend([max(0.0, min(lx, x_min)), max(0.0, min(lx, x_max))])
        y_breaks.extend([max(0.0, min(ly, y_min)), max(0.0, min(ly, y_max))])
        for x_value, y_value in _salient_obstacle_coordinates(obstacle):
            x_breaks.append(max(0.0, min(lx, x_value)))
            y_breaks.append(max(0.0, min(ly, y_value)))
    x_edges = _unique_sorted_edges(x_breaks, lx)
    y_edges = _unique_sorted_edges(y_breaks, ly)

    cells: List[FreeSpaceCell] = []
    serial = 0
    default_axis = choose_region_sweep_axis(config, path_config)
    for ix in range(len(x_edges) - 1):
        for iy in range(len(y_edges) - 1):
            x0, x1 = x_edges[ix], x_edges[ix + 1]
            y0, y1 = y_edges[iy], y_edges[iy + 1]
            if x1 - x0 < min_size or y1 - y0 < min_size:
                continue
            polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            if point_in_any_obstacle(center, obstacle_field, inflated=True):
                continue
            if polygon_collides_with_obstacles(_shrink_axis_aligned_polygon(polygon), obstacle_field, inflated=True):
                continue
            axis = "x" if (x1 - x0) >= (y1 - y0) else "y"
            if abs((x1 - x0) - (y1 - y0)) <= 1e-9:
                axis = default_axis
            cells.append(
                FreeSpaceCell(
                    cell_id=f"free_cell_{serial}",
                    bounds=(x0, y0, x1, y1),
                    polygon=polygon,
                    center=center,
                    area=(x1 - x0) * (y1 - y0),
                    preferred_axis=axis,
                    obstacle_ids=[obstacle.obstacle_id for obstacle in obstacle_field.inflated_obstacles],
                    metadata={
                        "ix": str(ix),
                        "iy": str(iy),
                        "static_obstacle_aware": "true",
                        "free_area": f"{(x1 - x0) * (y1 - y0):.6f}",
                        "narrow_width": f"{min(x1 - x0, y1 - y0):.6f}",
                    },
                )
            )
            serial += 1
    _populate_free_space_neighbors(cells)
    return cells


def _composite_region_from_cells(serial: int, cells: Sequence[FreeSpaceCell], preferred_axis: str) -> CompositeFreeSpaceRegion:
    x_min = min(cell.bounds[0] for cell in cells)
    y_min = min(cell.bounds[1] for cell in cells)
    x_max = max(cell.bounds[2] for cell in cells)
    y_max = max(cell.bounds[3] for cell in cells)
    area = sum(cell.area for cell in cells)
    if area <= 1e-9:
        center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
    else:
        center = (
            sum(cell.center[0] * cell.area for cell in cells) / area,
            sum(cell.center[1] * cell.area for cell in cells) / area,
        )
    polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    source_ids = ",".join(cell.cell_id for cell in cells)
    envelope_axis = "x" if (x_max - x_min) >= (y_max - y_min) else "y"
    return CompositeFreeSpaceRegion(
        region_id=f"composite_region_{serial}",
        bounds=(x_min, y_min, x_max, y_max),
        polygon=polygon,
        center=center,
        area=area,
        preferred_axis=envelope_axis if len(cells) > 1 else preferred_axis,
        source_algorithm="composite_free_space_decomposition",
        member_cells=list(cells),
        neighbors=[],
        metadata={
            "is_composite": "true",
            "static_obstacle_aware": "true",
            "source_cell_count": str(len(cells)),
            "source_region_ids": source_ids,
            "composite_bounds_are_envelope": "true",
        },
    )


def _composite_cells_can_join(first: FreeSpaceCell, second: FreeSpaceCell, config: PlannerConfig, axis: str) -> bool:
    ax0, ay0, ax1, ay1 = first.bounds
    bx0, by0, bx1, by1 = second.bounds
    shared_y = min(ay1, by1) - max(ay0, by0)
    shared_x = min(ax1, bx1) - max(ax0, bx0)
    touches_vertical = abs(ax1 - bx0) <= 1e-9 or abs(bx1 - ax0) <= 1e-9
    touches_horizontal = abs(ay1 - by0) <= 1e-9 or abs(by1 - ay0) <= 1e-9
    min_shared = max(config.footprint.width_wf * 0.25, 1e-6)
    if axis == "x":
        return touches_horizontal and shared_x >= min_shared
    return touches_vertical and shared_y >= min_shared


def _composite_cells_can_bridge_gap(
    first: FreeSpaceCell,
    second: FreeSpaceCell,
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: ObstacleField | None,
    axis: str,
) -> bool:
    ax0, ay0, ax1, ay1 = first.bounds
    bx0, by0, bx1, by1 = second.bounds
    gap_limit = max(config.footprint.width_wf * max(path_config.composite_gap_bridge_factor, 0.0), 0.0)
    if gap_limit <= 1e-9:
        return False
    x_overlap = min(ax1, bx1) - max(ax0, bx0)
    y_overlap = min(ay1, by1) - max(ay0, by0)
    vertical_gap = max(by0 - ay1, ay0 - by1, 0.0)
    horizontal_gap = max(bx0 - ax1, ax0 - bx1, 0.0)
    if axis == "x" and vertical_gap <= gap_limit and horizontal_gap <= 1e-9 and x_overlap >= config.footprint.width_wf * 0.25:
        x_min, x_max = max(ax0, bx0), min(ax1, bx1)
        y_min, y_max = (ay1, by0) if ay1 <= by0 else (by1, ay0)
    elif axis == "y" and horizontal_gap <= gap_limit and vertical_gap <= 1e-9 and y_overlap >= config.footprint.width_wf * 0.25:
        x_min, x_max = (ax1, bx0) if ax1 <= bx0 else (bx1, ax0)
        y_min, y_max = max(ay0, by0), min(ay1, by1)
    else:
        return False
    if x_max <= x_min or y_max <= y_min:
        return False
    bridge_polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    return obstacle_field is None or not polygon_collides_with_obstacles(bridge_polygon, obstacle_field, inflated=True)


def _populate_composite_neighbors(regions: List[CompositeFreeSpaceRegion]) -> None:
    cell_owner: Dict[str, str] = {}
    for region in regions:
        region.neighbors.clear()
        for cell in region.member_cells:
            cell_owner[cell.cell_id] = region.region_id
    by_id = {region.region_id: region for region in regions}
    for region in regions:
        neighbors: set[str] = set()
        for cell in region.member_cells:
            for neighbor_cell_id in cell.neighbors:
                owner = cell_owner.get(neighbor_cell_id)
                if owner and owner != region.region_id:
                    neighbors.add(owner)
        region.neighbors.extend(sorted(neighbors))
    for region in regions:
        for neighbor_id in list(region.neighbors):
            neighbor = by_id.get(neighbor_id)
            if neighbor is not None and region.region_id not in neighbor.neighbors:
                neighbor.neighbors.append(region.region_id)
                neighbor.neighbors.sort()


def polygon_signed_area(polygon: Sequence[Tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    acc = 0.0
    for idx, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(idx + 1) % len(polygon)]
        acc += x0 * y1 - x1 * y0
    return acc / 2.0


def concave_vertex_indices(polygon: Sequence[Tuple[float, float]]) -> List[int]:
    if len(polygon) < 4:
        return []
    ccw = polygon_signed_area(polygon) > 0.0
    indices: List[int] = []
    for idx in range(len(polygon)):
        x_prev, y_prev = polygon[idx - 1]
        x_curr, y_curr = polygon[idx]
        x_next, y_next = polygon[(idx + 1) % len(polygon)]
        v1 = (x_curr - x_prev, y_curr - y_prev)
        v2 = (x_next - x_curr, y_next - y_curr)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if (ccw and cross < -1e-9) or ((not ccw) and cross > 1e-9):
            indices.append(idx)
    return indices


def decompose_polygon_interface(
    polygon: Sequence[Tuple[float, float]],
    preferred_axis: str = "x",
) -> List[DecomposedRegion]:
    """Lightweight concave-region interface.

    The first implementation is intentionally conservative: it detects concave
    vertices and returns axis-aligned slabs over the polygon bounding box. This
    keeps the rectangular main flow stable while giving later exact
    decomposition code a typed entrypoint.
    """

    if len(polygon) < 3:
        return []
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    concave = concave_vertex_indices(polygon)
    split_count = max(1, len(concave) + 1)
    regions: List[DecomposedRegion] = []
    if preferred_axis == "x":
        edges = [y_min + (y_max - y_min) * idx / split_count for idx in range(split_count + 1)]
        for idx in range(split_count):
            regions.append(
                _region_from_bounds(
                    f"poly_region_{idx}",
                    (x_min, edges[idx], x_max, edges[idx + 1]),
                    preferred_axis="x",
                    source_algorithm="concave_interface_slab_decomposition",
                    metadata={"concave_vertices": str(len(concave))},
                )
            )
    else:
        edges = [x_min + (x_max - x_min) * idx / split_count for idx in range(split_count + 1)]
        for idx in range(split_count):
            regions.append(
                _region_from_bounds(
                    f"poly_region_{idx}",
                    (edges[idx], y_min, edges[idx + 1], y_max),
                    preferred_axis="y",
                    source_algorithm="concave_interface_slab_decomposition",
                    metadata={"concave_vertices": str(len(concave))},
                )
            )
    _populate_axis_aligned_neighbors(regions)
    return regions


def _bounds_touch_or_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x_overlap = min(ax1, bx1) - max(ax0, bx0)
    y_overlap = min(ay1, by1) - max(ay0, by0)
    x_touch = abs(ax1 - bx0) <= 1e-9 or abs(bx1 - ax0) <= 1e-9
    y_touch = abs(ay1 - by0) <= 1e-9 or abs(by1 - ay0) <= 1e-9
    return (x_touch and y_overlap >= -1e-9) or (y_touch and x_overlap >= -1e-9)


def _populate_axis_aligned_neighbors(regions: List[DecomposedRegion]) -> None:
    for region in regions:
        region.neighbors.clear()
    for idx, region_a in enumerate(regions):
        for region_b in regions[idx + 1 :]:
            if _bounds_touch_or_overlap(region_a.bounds, region_b.bounds):
                region_a.neighbors.append(region_b.region_id)
                region_b.neighbors.append(region_a.region_id)


def _populate_free_space_neighbors(cells: List[FreeSpaceCell]) -> None:
    for cell in cells:
        cell.neighbors.clear()
    for idx, cell_a in enumerate(cells):
        for cell_b in cells[idx + 1 :]:
            if _bounds_touch_or_overlap(cell_a.bounds, cell_b.bounds):
                cell_a.neighbors.append(cell_b.cell_id)
                cell_b.neighbors.append(cell_a.cell_id)


def _unique_sorted_edges(values: List[float], max_value: float) -> List[float]:
    clipped = [max(0.0, min(max_value, value)) for value in values]
    clipped.extend([0.0, max_value])
    unique = sorted(set(round(value, 9) for value in clipped))
    return [float(value) for value in unique if 0.0 <= value <= max_value]


def _regular_breaks(max_value: float, step: float) -> List[float]:
    if step <= 1e-9:
        return []
    count = max(0, int(math.floor(max_value / step)))
    return [idx * step for idx in range(1, count + 1) if idx * step < max_value - 1e-9]


def _free_space_grid_step(config: PlannerConfig, path_config: PathPlanningConfig) -> float:
    resolution = float(path_config.coverage_resolution or config.footprint.width_wf)
    footprint_scale = max(config.footprint.length_lf, config.footprint.width_wf) * 2.0
    return max(resolution, footprint_scale, 1e-6)


def _salient_obstacle_coordinates(obstacle) -> List[Tuple[float, float]]:
    if obstacle.kind in {"circle", "ellipse"}:
        x_min, y_min, x_max, y_max = obstacle_bounds(obstacle)
        return [((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)]
    return list(obstacle.polygon)


def _shrink_axis_aligned_polygon(polygon: List[Tuple[float, float]], eps: float = 1e-6) -> List[Tuple[float, float]]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max - x_min <= 2.0 * eps or y_max - y_min <= 2.0 * eps:
        return polygon
    return [
        (x_min + eps, y_min + eps),
        (x_max - eps, y_min + eps),
        (x_max - eps, y_max - eps),
        (x_min + eps, y_max - eps),
    ]
