from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    psi: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi], dtype=float)


@dataclass
class State3DOF:
    x: float
    y: float
    psi: float
    u: float = 0.0
    v: float = 0.0
    r: float = 0.0

    def pose(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.psi)

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.u, self.v, self.r], dtype=float)


@dataclass
class State6DOF:
    x: float
    y: float
    z: float = 0.0
    phi: float = 0.0
    theta: float = 0.0
    psi: float = 0.0
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0


@dataclass
class ControlInput:
    thrust: float
    yaw_moment: float

    @staticmethod
    def zero() -> "ControlInput":
        return ControlInput(0.0, 0.0)


@dataclass
class DynamicObstacleSample:
    time: float
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class DynamicObstacleTrack:
    obstacle_id: str
    radius: float
    samples: List[DynamicObstacleSample] = field(default_factory=list)


@dataclass
class MissionConfig:
    area_length_x: float
    area_length_y: float
    overlap_ratio: float = 0.1
    global_replan_hz: float = 0.5
    local_control_hz: float = 5.0
    residual_enable: bool = True


@dataclass
class FleetConfig:
    initial_states_3dof: List[State3DOF]
    initial_states_6dof: List[State6DOF]
    cruise_speed: float
    cover_speed: float
    turn_speed_max: float
    max_thrust: float
    max_yaw_moment: float
    min_turn_radius: float
    num_agents: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_agents is None:
            self.num_agents = len(self.initial_states_3dof)
        if len(self.initial_states_3dof) != self.num_agents:
            raise ValueError("initial_states_3dof length must match num_agents")
        if self.initial_states_6dof and len(self.initial_states_6dof) != self.num_agents:
            raise ValueError("initial_states_6dof length must match num_agents")


@dataclass
class CoverageFootprint:
    length_lf: float
    width_wf: float
    eta_cov: float = 0.7


@dataclass
class PlannerWeights:
    lambda1: float = 1.0
    lambda2: float = 0.5
    w_pos: float = 4.0
    w_psi: float = 1.0
    w_vel: float = 2.0
    w_u: float = 0.1
    w_du: float = 0.5
    w_soft: float = 20.0


@dataclass
class SafetyMargins:
    d_safe: float
    boundary_margin_x: float = 0.5
    boundary_margin_y: float = 0.5
    delta_safe_max: float = 1.0
    t_block: float = 8.0


@dataclass
class PlannerConfig:
    mission: MissionConfig
    fleet: FleetConfig
    footprint: CoverageFootprint
    weights: PlannerWeights
    safety: SafetyMargins


@dataclass(frozen=True)
class StripTask:
    strip_id: int
    start_pose: Pose2D
    end_pose: Pose2D
    nominal_heading: float
    strip_length: float
    pocket_left: Pose2D
    pocket_right: Pose2D
    scan_axis: str
    center_coordinate: float


@dataclass
class AssignmentPlan:
    assignments: Dict[int, Tuple[int, int]]
    estimated_cost: Dict[int, float]
    ordered_tasks: Dict[int, List[StripTask]]
    agent_order: List[int]


@dataclass
class PathRequirement:
    agent_id: int
    seq_index: int
    kind: str
    resource_id: str
    duration: float
    from_node: str
    to_node: str
    start_pose: Pose2D
    end_pose: Pose2D
    strip_id: Optional[int] = None


@dataclass
class ConstraintWindow:
    agent_id: int
    resource_id: str
    start_time: float
    end_time: float
    from_node: Optional[str] = None
    to_node: Optional[str] = None


@dataclass
class ReservationEntry:
    agent_id: int
    seq_index: int
    resource_id: str
    kind: str
    t_enter: float
    t_exit: float
    from_node: str
    to_node: str
    start_pose: Pose2D
    end_pose: Pose2D
    strip_id: Optional[int] = None


@dataclass
class MAPFReservationTable:
    reservations: Dict[int, List[ReservationEntry]]
    conflicts_resolved: int
    makespan: float


@dataclass
class TimedPathSegment:
    segment_type: str
    start_time: float
    end_time: float
    start_pose: Pose2D
    end_pose: Pose2D
    points: List[Tuple[float, float]]
    headings: List[float]
    control_points: Optional[List[Tuple[float, float]]] = None
    max_curvature: float = 0.0
    length: float = 0.0
    path_source: str = "bezier"
    dubins_modes: Optional[Tuple[str, str, str]] = None


@dataclass
class SmoothedPath:
    agent_id: int
    segments: List[TimedPathSegment]
    total_length: float
    max_curvature: float


@dataclass
class TrajectorySample:
    time: float
    x: float
    y: float
    psi: float
    u_ref: float
    r_ref: float
    segment_type: str


@dataclass
class TrajectoryReference:
    agent_id: int
    samples: List[TrajectorySample]
    horizon_time: float


@dataclass
class CoverageResidual:
    residual_id: int
    cells: List[Tuple[int, int]]
    centroid: Tuple[float, float]
    bounds: Tuple[float, float, float, float]


@dataclass
class CoverageState:
    resolution: float
    x_coords: np.ndarray
    y_coords: np.ndarray
    coverage_ratio: np.ndarray
    covered: np.ndarray
    residual_components: List[CoverageResidual] = field(default_factory=list)

    @property
    def coverage_fraction(self) -> float:
        return float(np.count_nonzero(self.covered) / self.covered.size)


@dataclass
class PlanningResult:
    strips: List[StripTask]
    assignments: AssignmentPlan
    reservations: MAPFReservationTable
    paths: Dict[int, SmoothedPath]
    refs: Dict[int, TrajectoryReference]


@dataclass
class AgentRuntimeState:
    agent_id: int
    time: float
    state3: State3DOF
    state6: Optional[State6DOF] = None
    previous_control: ControlInput = field(default_factory=ControlInput.zero)


@dataclass
class SafetyStatus:
    mode: str
    min_margin: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class ControlStepResult:
    cmd: ControlInput
    safety_status: SafetyStatus
    local_ref: List[TrajectorySample]
    predicted_samples: List[TrajectorySample]


def ensure_sequence(sequence: Optional[Sequence[State6DOF]]) -> List[State6DOF]:
    return list(sequence or [])
