from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    coverage_residual_interval_steps: int = 1
    control_mode: str = "hybrid_nmpc"
    nmpc_update_interval_steps: int = 5
    nmpc_max_wall_time_ms: float = 80.0
    nmpc_horizon_seconds: float = 1.2
    nmpc_horizon_steps_cap: int = 10
    nmpc_parallel_backend: str = "serial"
    nmpc_solver_backend: str = "auto"
    dynamics_integration_method: str = "rk4"
    nmpc_integration_method: str = "rk4"
    safety_filter_mode: str = "hybrid_cbf_qp"
    cbf_alpha: float = 0.8
    cbf_allow_slack: bool = False
    cbf_slack_weight: float = 1.0e4
    cbf_qp_max_iter: int = 30
    cbf_qp_timeout_ms: float = 5.0
    safety_min_margin_epsilon: float = 1.0e-3
    mapf_solver: str = "auto"
    mapf_max_expanded_nodes: int = 5000
    mapf_max_conflicts: int = 2000
    mapf_max_wall_time_ms: float = 2000.0
    mapf_suboptimality_bound: float = 1.2
    mapf_fallback: str = "prioritized_resource_windows"


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


@dataclass(frozen=True)
class VehicleFootprint:
    """Physical hull envelope used for collision and boundary validation."""

    length: float
    width: float

    def __post_init__(self) -> None:
        if self.length <= 0.0 or self.width <= 0.0:
            raise ValueError("vehicle footprint dimensions must be positive")


@dataclass(frozen=True)
class AgentPlanningProfile:
    """Coverage, geometry, and motion capabilities for one USV."""

    agent_id: int
    coverage_length: float
    coverage_width: float
    overlap_ratio: float
    vehicle_length: float
    vehicle_width: float
    min_turn_radius: float
    cruise_speed: float
    cover_speed: float
    turn_speed_max: float
    max_thrust: float
    max_yaw_moment: float
    max_mission_time: Optional[float] = None

    def __post_init__(self) -> None:
        positive = {
            "coverage_length": self.coverage_length,
            "coverage_width": self.coverage_width,
            "vehicle_length": self.vehicle_length,
            "vehicle_width": self.vehicle_width,
            "min_turn_radius": self.min_turn_radius,
            "cruise_speed": self.cruise_speed,
            "cover_speed": self.cover_speed,
            "turn_speed_max": self.turn_speed_max,
            "max_thrust": self.max_thrust,
            "max_yaw_moment": self.max_yaw_moment,
        }
        invalid = [name for name, value in positive.items() if float(value) <= 0.0]
        if invalid:
            raise ValueError(f"agent profile values must be positive: {','.join(invalid)}")
        if not 0.0 <= float(self.overlap_ratio) < 1.0:
            raise ValueError("agent overlap_ratio must be in [0, 1)")
        if self.max_mission_time is not None and self.max_mission_time <= 0.0:
            raise ValueError("agent max_mission_time must be positive when provided")

    @property
    def effective_strip_spacing(self) -> float:
        return self.coverage_width * (1.0 - self.overlap_ratio)

    @property
    def yaw_rate_limit(self) -> float:
        return self.turn_speed_max / self.min_turn_radius

    @property
    def fingerprint(self) -> str:
        values = (
            self.coverage_length,
            self.coverage_width,
            self.overlap_ratio,
            self.vehicle_length,
            self.vehicle_width,
            self.min_turn_radius,
            self.cruise_speed,
            self.cover_speed,
            self.turn_speed_max,
            self.max_thrust,
            self.max_yaw_moment,
        )
        return ":".join(f"{value:.6g}" for value in values)

    def coverage_footprint(self, eta_cov: float = 0.7) -> CoverageFootprint:
        return CoverageFootprint(
            length_lf=self.coverage_length,
            width_wf=self.coverage_width,
            eta_cov=eta_cov,
        )

    def vehicle_footprint(self) -> VehicleFootprint:
        return VehicleFootprint(length=self.vehicle_length, width=self.vehicle_width)


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
    agent_profiles: Dict[int, AgentPlanningProfile] = field(default_factory=dict)
    vehicle_footprint: Optional[VehicleFootprint] = None
    active_agent_id: Optional[int] = None
    fleet_profile_id: str = ""

    def profile_for_agent(self, agent_id: int) -> AgentPlanningProfile:
        if agent_id in self.agent_profiles:
            return self.agent_profiles[agent_id]
        vehicle = self.vehicle_footprint or VehicleFootprint(
            length=self.footprint.length_lf,
            width=self.footprint.width_wf,
        )
        return AgentPlanningProfile(
            agent_id=agent_id,
            coverage_length=self.footprint.length_lf,
            coverage_width=self.footprint.width_wf,
            overlap_ratio=self.mission.overlap_ratio,
            vehicle_length=vehicle.length,
            vehicle_width=vehicle.width,
            min_turn_radius=self.fleet.min_turn_radius,
            cruise_speed=self.fleet.cruise_speed,
            cover_speed=self.fleet.cover_speed,
            turn_speed_max=self.fleet.turn_speed_max,
            max_thrust=self.fleet.max_thrust,
            max_yaw_moment=self.fleet.max_yaw_moment,
        )

    def for_agent(self, agent_id: int) -> "PlannerConfig":
        """Return a planner view resolved to one agent's capabilities."""

        profile = self.profile_for_agent(agent_id)
        return replace(
            self,
            mission=replace(self.mission, overlap_ratio=profile.overlap_ratio),
            fleet=replace(
                self.fleet,
                cruise_speed=profile.cruise_speed,
                cover_speed=profile.cover_speed,
                turn_speed_max=profile.turn_speed_max,
                max_thrust=profile.max_thrust,
                max_yaw_moment=profile.max_yaw_moment,
                min_turn_radius=profile.min_turn_radius,
            ),
            footprint=profile.coverage_footprint(self.footprint.eta_cov),
            vehicle_footprint=profile.vehicle_footprint(),
            active_agent_id=agent_id,
        )

    def validate_agent_profiles(self) -> None:
        agent_count = self.fleet.num_agents or len(self.fleet.initial_states_3dof)
        unknown = sorted(set(self.agent_profiles) - set(range(agent_count)))
        if unknown:
            raise ValueError(f"agent profile ids outside fleet range: {unknown}")


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
    solver_status: str = "solved"
    expanded_nodes: int = 0
    open_set_peak: int = 0
    conflict_checks: int = 0
    budget_exhausted: bool = False
    fallback_used: bool = False
    unresolved_conflict_count: int = 0
    unresolved_conflicts: List[str] = field(default_factory=list)


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
