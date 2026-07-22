from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..geometry import connected_components, rotated_rectangle_mask
from ..schema import CoverageResidual, CoverageState, PlannerConfig, Pose2D


@dataclass(frozen=True)
class RectangularCoverageModel:
    length: float
    width: float
    overlap_ratio: float
    min_turn_radius: float
    safe_distance: float

    @property
    def strip_spacing(self) -> float:
        return max(self.width * (1.0 - self.overlap_ratio), 1e-6)

    @property
    def turn_buffer(self) -> float:
        return self.min_turn_radius + self.length / 2.0 + self.safe_distance

    @classmethod
    def from_config(cls, config: PlannerConfig) -> "RectangularCoverageModel":
        return cls(
            length=config.footprint.length_lf,
            width=config.footprint.width_wf,
            overlap_ratio=config.mission.overlap_ratio,
            min_turn_radius=config.fleet.min_turn_radius,
            safe_distance=config.safety.d_safe,
        )


def build_coverage_state(config: PlannerConfig, resolution: float | None = None) -> CoverageState:
    cell = max(float(resolution or config.footprint.width_wf / 2.0), 1e-6)
    x_count = max(1, int(math.ceil(config.mission.area_length_x / cell)))
    y_count = max(1, int(math.ceil(config.mission.area_length_y / cell)))
    x_coords = np.linspace(cell / 2.0, config.mission.area_length_x - cell / 2.0, x_count)
    y_coords = np.linspace(cell / 2.0, config.mission.area_length_y - cell / 2.0, y_count)
    coverage_ratio = np.zeros((len(y_coords), len(x_coords)), dtype=float)
    covered = np.zeros_like(coverage_ratio, dtype=bool)
    return CoverageState(
        resolution=cell,
        x_coords=x_coords,
        y_coords=y_coords,
        coverage_ratio=coverage_ratio,
        covered=covered,
    )


def sample_pose_along_segment(start: Pose2D, end: Pose2D, spacing: float) -> List[Pose2D]:
    distance = math.hypot(end.x - start.x, end.y - start.y)
    count = max(2, int(math.ceil(distance / max(spacing, 1e-6))) + 1)
    heading = math.atan2(end.y - start.y, end.x - start.x) if distance > 1e-9 else start.psi
    poses: List[Pose2D] = []
    for idx in range(count):
        alpha = idx / max(count - 1, 1)
        poses.append(
            Pose2D(
                x=start.x + alpha * (end.x - start.x),
                y=start.y + alpha * (end.y - start.y),
                psi=heading,
            )
        )
    return poses


def mark_rectangular_swept_segment(
    state: CoverageState,
    start: Pose2D,
    end: Pose2D,
    length: float,
    width: float,
    eta_cov: float,
    sample_spacing: float | None = None,
) -> CoverageState:
    spacing = max(float(sample_spacing or width / 2.0), 1e-6)
    poses = sample_pose_along_segment(start, end, spacing)
    hits = np.zeros_like(state.coverage_ratio, dtype=float)
    for pose in poses:
        mask = rotated_rectangle_mask(
            state.x_coords,
            state.y_coords,
            pose.x,
            pose.y,
            pose.psi,
            length,
            width,
        )
        hits[mask] += 1.0
    if poses:
        # A grid cell is counted once the swept rectangular footprint intersects
        # its center; ratio is kept as a confidence-like [0, 1] value.
        state.coverage_ratio = np.maximum(state.coverage_ratio, (hits > 0.0).astype(float))
    state.covered = state.covered | (state.coverage_ratio >= eta_cov)
    return state


def mark_coverage_passes(
    state: CoverageState,
    passes: Sequence[Tuple[Pose2D, Pose2D]],
    model: RectangularCoverageModel,
    eta_cov: float,
) -> CoverageState:
    for start, end in passes:
        mark_rectangular_swept_segment(
            state,
            start,
            end,
            model.length,
            model.width,
            eta_cov=eta_cov,
            sample_spacing=model.width / 2.0,
        )
    return state


def find_residual_components(
    state: CoverageState,
    area_bounds: Tuple[float, float, float, float] | None = None,
) -> List[CoverageResidual]:
    uncovered = ~state.covered
    if area_bounds is not None:
        x_min, y_min, x_max, y_max = area_bounds
        xx, yy = np.meshgrid(state.x_coords, state.y_coords)
        in_bounds = (xx >= x_min) & (xx <= x_max) & (yy >= y_min) & (yy <= y_max)
        uncovered = uncovered & in_bounds

    residuals: List[CoverageResidual] = []
    for residual_id, component in enumerate(connected_components(uncovered)):
        xs = [float(state.x_coords[col]) for _, col in component]
        ys = [float(state.y_coords[row]) for row, _ in component]
        residuals.append(
            CoverageResidual(
                residual_id=residual_id,
                cells=component,
                centroid=(float(np.mean(xs)), float(np.mean(ys))),
                bounds=(min(xs), min(ys), max(xs), max(ys)),
            )
        )
    state.residual_components = residuals
    return residuals
