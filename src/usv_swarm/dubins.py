from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .schema import Pose2D


def mod2pi(theta: float) -> float:
    return theta % (2.0 * math.pi)


@dataclass(frozen=True)
class DubinsPath:
    start: Pose2D
    end: Pose2D
    turn_radius: float
    modes: Tuple[str, str, str]
    normalized_lengths: Tuple[float, float, float]
    total_length: float

    @property
    def segment_lengths(self) -> Tuple[float, float, float]:
        return tuple(length * self.turn_radius for length in self.normalized_lengths)


def _lsl(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    p_sq = 2.0 + d * d - 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))
    if p_sq < 0.0:
        return None
    tmp = math.atan2(math.cos(beta) - math.cos(alpha), d + math.sin(alpha) - math.sin(beta))
    return mod2pi(-alpha + tmp), math.sqrt(max(p_sq, 0.0)), mod2pi(beta - tmp)


def _rsr(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    p_sq = 2.0 + d * d - 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(beta) - math.sin(alpha))
    if p_sq < 0.0:
        return None
    tmp = math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta))
    return mod2pi(alpha - tmp), math.sqrt(max(p_sq, 0.0)), mod2pi(-beta + tmp)


def _lsr(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    p_sq = -2.0 + d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) + math.sin(beta))
    if p_sq < 0.0:
        return None
    p = math.sqrt(max(p_sq, 0.0))
    tmp = math.atan2(-math.cos(alpha) - math.cos(beta), d + math.sin(alpha) + math.sin(beta)) - math.atan2(-2.0, p)
    return mod2pi(-alpha + tmp), p, mod2pi(tmp - beta)


def _rsl(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    p_sq = d * d - 2.0 + 2.0 * math.cos(alpha - beta) - 2.0 * d * (math.sin(alpha) + math.sin(beta))
    if p_sq < 0.0:
        return None
    p = math.sqrt(max(p_sq, 0.0))
    tmp = math.atan2(math.cos(alpha) + math.cos(beta), d - math.sin(alpha) - math.sin(beta)) - math.atan2(2.0, p)
    return mod2pi(alpha - tmp), p, mod2pi(beta - tmp)


def _rlr(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    tmp = (6.0 - d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = mod2pi(2.0 * math.pi - math.acos(tmp))
    t = mod2pi(alpha - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta)) + p / 2.0)
    q = mod2pi(alpha - beta - t + p)
    return t, p, q


def _lrl(alpha: float, beta: float, d: float) -> Optional[Tuple[float, float, float]]:
    tmp = (6.0 - d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (-math.sin(alpha) + math.sin(beta))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = mod2pi(2.0 * math.pi - math.acos(tmp))
    t = mod2pi(-alpha - math.atan2(math.cos(alpha) - math.cos(beta), d + math.sin(alpha) - math.sin(beta)) + p / 2.0)
    q = mod2pi(mod2pi(beta) - alpha - t + p)
    return t, p, q


_PATH_GENERATORS: Dict[Tuple[str, str, str], Callable[[float, float, float], Optional[Tuple[float, float, float]]]] = {
    ("L", "S", "L"): _lsl,
    ("R", "S", "R"): _rsr,
    ("L", "S", "R"): _lsr,
    ("R", "S", "L"): _rsl,
    ("R", "L", "R"): _rlr,
    ("L", "R", "L"): _lrl,
}


def dubins_shortest_path(start: Pose2D, end: Pose2D, turn_radius: float) -> DubinsPath:
    if turn_radius <= 0.0:
        raise ValueError("turn_radius must be positive")

    dx = end.x - start.x
    dy = end.y - start.y
    distance = math.hypot(dx, dy)
    theta = mod2pi(math.atan2(dy, dx)) if distance > 1e-12 else 0.0
    d = distance / turn_radius
    alpha = mod2pi(start.psi - theta)
    beta = mod2pi(end.psi - theta)

    best_path: Optional[DubinsPath] = None
    for modes, generator in _PATH_GENERATORS.items():
        candidate = generator(alpha, beta, d)
        if candidate is None:
            continue
        total = turn_radius * sum(candidate)
        if best_path is None or total < best_path.total_length:
            best_path = DubinsPath(
                start=start,
                end=end,
                turn_radius=turn_radius,
                modes=modes,
                normalized_lengths=candidate,
                total_length=total,
            )

    if best_path is None:
        # This should rarely happen once all six Dubins families are considered.
        heading_delta = mod2pi(end.psi - start.psi)
        fallback = (heading_delta, distance / max(turn_radius, 1e-9), 0.0)
        best_path = DubinsPath(
            start=start,
            end=end,
            turn_radius=turn_radius,
            modes=("L", "S", "L"),
            normalized_lengths=fallback,
            total_length=turn_radius * sum(fallback),
        )
    return best_path


def advance_pose_along_mode(pose: Pose2D, mode: str, segment_length: float, turn_radius: float) -> Pose2D:
    if segment_length <= 0.0:
        return pose
    if mode == "S":
        x = pose.x + segment_length * math.cos(pose.psi)
        y = pose.y + segment_length * math.sin(pose.psi)
        return Pose2D(x=x, y=y, psi=pose.psi)
    delta = segment_length / turn_radius
    if mode == "L":
        psi_new = pose.psi + delta
        x = pose.x + turn_radius * (math.sin(psi_new) - math.sin(pose.psi))
        y = pose.y + turn_radius * (-math.cos(psi_new) + math.cos(pose.psi))
        return Pose2D(x=x, y=y, psi=psi_new)
    if mode == "R":
        psi_new = pose.psi - delta
        x = pose.x + turn_radius * (math.sin(pose.psi) - math.sin(psi_new))
        y = pose.y + turn_radius * (math.cos(psi_new) - math.cos(pose.psi))
        return Pose2D(x=x, y=y, psi=psi_new)
    raise ValueError(f"unsupported mode: {mode}")


def sample_dubins_path(
    path: DubinsPath,
    step_size: float = 0.25,
    *,
    max_heading_step: float | None = None,
) -> Tuple[List[Tuple[float, float]], List[float], float]:
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if max_heading_step is not None and max_heading_step <= 0.0:
        raise ValueError("max_heading_step must be positive when provided")
    pose = path.start
    points: List[Tuple[float, float]] = [(pose.x, pose.y)]
    headings: List[float] = [pose.psi]
    max_curvature = 0.0

    for mode, segment_length in zip(path.modes, path.segment_lengths):
        remaining = float(segment_length)
        while remaining > 1e-9:
            mode_step = step_size
            if mode in ("L", "R") and max_heading_step is not None:
                # A distance-only sample count is unsafe on a long Dubins
                # path containing a short, tight arc: the whole arc may be
                # represented by one large heading jump and then be rejected
                # by the dynamics validator even though the analytic path is
                # curvature feasible.  Bound the angular discretization on
                # turns without needlessly oversampling long straight legs.
                mode_step = min(
                    mode_step,
                    path.turn_radius * max_heading_step,
                )
            ds = min(mode_step, remaining)
            pose = advance_pose_along_mode(pose, mode, ds, path.turn_radius)
            points.append((pose.x, pose.y))
            headings.append(pose.psi)
            remaining -= ds
            if mode in ("L", "R"):
                max_curvature = max(max_curvature, 1.0 / path.turn_radius)

    if math.hypot(points[-1][0] - path.end.x, points[-1][1] - path.end.y) > 1e-5 or abs(headings[-1] - path.end.psi) > 1e-5:
        points.append((path.end.x, path.end.y))
        headings.append(path.end.psi)
    return points, headings, max_curvature
