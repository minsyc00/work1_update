from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..schema import AssignmentPlan, CoverageState, DynamicObstacleTrack, PlannerConfig, PlanningResult, Pose2D, StripTask


class DynamicsModelTag(str, Enum):
    THREE_DOF = "3dof"
    SIX_DOF = "6dof"


@dataclass(frozen=True)
class PaperReference:
    paper_id: str
    title: str = ""
    authors: Tuple[str, ...] = ()
    year: Optional[int] = None
    venue: str = ""
    notes: str = ""


@dataclass(frozen=True)
class PathWaypoint:
    x: float
    y: float
    psi: float
    time: Optional[float] = None
    speed: Optional[float] = None


@dataclass
class PathSegmentSpec:
    segment_id: str
    kind: str
    source_algorithm: str
    waypoints: List[PathWaypoint] = field(default_factory=list)
    control_points: List[Tuple[float, float]] = field(default_factory=list)
    curvature_max: float = 0.0
    length: float = 0.0
    path_source: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentPathPlan:
    agent_id: int
    source_algorithm: str
    segments: List[PathSegmentSpec] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    paper_references: List[PaperReference] = field(default_factory=list)


@dataclass
class MultiAgentPathPlan:
    algorithm_name: str
    agents: Dict[int, AgentPathPlan]
    metadata: Dict[str, str] = field(default_factory=dict)
    paper_references: List[PaperReference] = field(default_factory=list)


@dataclass(frozen=True)
class PathPlanningConfig:
    """Algorithm-level parameters kept separate from the runtime planner config."""

    sweep_axis: Optional[str] = None
    overlap_ratio: Optional[float] = None
    coverage_resolution: Optional[float] = None
    residual_resolution: Optional[float] = None
    max_regions_per_agent: int = 4
    max_candidate_axes: int = 2
    length_weight: float = 1.0
    turn_angle_weight: float = 0.35
    time_weight: float = 1.0
    load_balance_weight: float = 0.5
    safety_weight: float = 0.25
    boundary_weight: float = 0.1
    curvature_weight: float = 100.0
    astar_heading_weight: float = 0.35
    astar_safety_weight: float = 0.5
    astar_boundary_weight: float = 0.2
    tsp_2opt_iterations: int = 25
    tsp_3opt_iterations: int = 0
    use_bezier_smoothing: bool = True
    load_imbalance_tolerance: float = 0.10
    obstacle_circle_segments: int = 24
    obstacle_inflation_extra: float = 0.0
    coverage_turn_pocket_scale: float = 1.0
    min_free_cell_size: Optional[float] = None
    enable_residual_backfill: bool = True
    max_residual_backfill_regions: int = 12
    residual_backfill_cycles: int = 3
    visual_output_dir: Optional[str] = None
    visual_map_id: Optional[str] = None
    visual_dpi: int = 180
    visual_gif_fps: int = 6

    @classmethod
    def from_planner_config(cls, config: PlannerConfig) -> "PathPlanningConfig":
        resolution = max(config.footprint.width_wf * 0.5, 1e-6)
        return cls(
            overlap_ratio=config.mission.overlap_ratio,
            coverage_resolution=resolution,
            residual_resolution=resolution,
        )


@dataclass
class DecomposedRegion:
    region_id: str
    bounds: Tuple[float, float, float, float]
    polygon: List[Tuple[float, float]]
    center: Tuple[float, float]
    area: float
    preferred_axis: str
    source_algorithm: str = "rectangular_decomposition"
    neighbors: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class StaticObstacle:
    obstacle_id: str
    kind: str
    polygon: List[Tuple[float, float]] = field(default_factory=list)
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None
    radii: Optional[Tuple[float, float]] = None
    width: Optional[float] = None
    height: Optional[float] = None
    psi: float = 0.0
    inflation_radius: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ObstacleField:
    obstacles: List[StaticObstacle] = field(default_factory=list)
    inflated_obstacles: List[StaticObstacle] = field(default_factory=list)
    safety_margin: float = 0.0
    footprint_margin: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class FreeSpaceCell:
    cell_id: str
    bounds: Tuple[float, float, float, float]
    polygon: List[Tuple[float, float]]
    center: Tuple[float, float]
    area: float
    preferred_axis: str
    source_algorithm: str = "obstacle_aware_sweep_decomposition"
    neighbors: List[str] = field(default_factory=list)
    obstacle_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class CoveragePass:
    pass_id: str
    region_id: str
    sequence_index: int
    scan_axis: str
    start_pose: Pose2D
    end_pose: Pose2D
    center_coordinate: float
    width: float
    length: float


@dataclass
class RegionCoveragePattern:
    pattern_id: str
    region_id: str
    scan_axis: str
    passes: List[CoveragePass]
    entry_pose: Pose2D
    exit_pose: Pose2D
    coverage_length: float
    turn_length: float
    turn_angle: float
    total_length: float
    estimated_time: float
    max_curvature: float
    feasible: bool = True
    source_algorithm: str = "paper_fusion_pattern"
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionVisitNode:
    region_id: str
    pattern_id: str
    entry_pose: Pose2D
    exit_pose: Pose2D
    pass_count: int
    coverage_endpoint_count: int
    estimated_time: float


@dataclass
class RegionSweepPath:
    region_id: str
    pattern_id: str
    passes: List[CoveragePass]
    endpoints: List[Pose2D]
    entry_pose: Pose2D
    exit_pose: Pose2D
    feasible: bool = True
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ObstacleAwareCoveragePattern:
    pattern: RegionCoveragePattern
    collision_free: bool
    clipped_pass_count: int
    obstacle_clearance: float
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RegionGraph:
    regions: Dict[str, DecomposedRegion]
    adjacency: Dict[str, List[str]]
    node_weights: Dict[str, float]
    edge_weights: Dict[Tuple[str, str], float]
    edge_metadata: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)
    patterns: Dict[str, List[RegionCoveragePattern]] = field(default_factory=dict)
    obstacle_field: Optional[ObstacleField] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class BalancedAssignment:
    agent_regions: Dict[int, List[str]]
    loads: Dict[int, float]
    connected: Dict[int, bool]
    imbalance_ratio: float
    objective: float
    diagnostics: Dict[str, str] = field(default_factory=dict)


@dataclass
class SingleUsvTourPlan:
    agent_id: int
    region_order: List[str]
    selected_patterns: Dict[str, RegionCoveragePattern]
    segments: List[PathSegmentSpec] = field(default_factory=list)
    total_length: float = 0.0
    total_turn_angle: float = 0.0
    estimated_time: float = 0.0
    objective: float = 0.0
    improved: bool = False
    diagnostics: Dict[str, str] = field(default_factory=dict)


@dataclass
class ResidualBackfillPlan:
    residual_regions: List[DecomposedRegion]
    agent_regions: Dict[int, List[str]]
    estimated_start_times: Dict[int, float]
    estimated_transition_cost: Dict[Tuple[int, str], float]
    diagnostics: Dict[str, str] = field(default_factory=dict)


@dataclass
class PathPlanningDiagnostics:
    coverage_fraction: float = 0.0
    total_length: float = 0.0
    max_curvature: float = 0.0
    load_imbalance_ratio: float = 0.0
    planning_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PathPlanningTrace:
    enabled: bool = False
    output_dir: Optional[str] = None
    map_id: Optional[str] = None
    obstacle_field: Optional[ObstacleField] = None
    regions_before_filter: List[DecomposedRegion] = field(default_factory=list)
    regions: List[DecomposedRegion] = field(default_factory=list)
    patterns: Dict[str, List[RegionCoveragePattern]] = field(default_factory=dict)
    graph: Optional[RegionGraph] = None
    assignment: Optional[BalancedAssignment] = None
    tours: Dict[int, SingleUsvTourPlan] = field(default_factory=dict)
    agents: Dict[int, AgentPathPlan] = field(default_factory=dict)
    coverage_state: Optional[CoverageState] = None
    diagnostics: Optional[PathPlanningDiagnostics] = None
    residual_backfill_count: int = 0
    mapf_conflicts_resolved: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlgorithmExperimentTrace:
    map_id: str = ""
    output_dir: Optional[str] = None
    obstacle_field: Optional[ObstacleField] = None
    regions_before_filter: List[DecomposedRegion] = field(default_factory=list)
    regions: List[DecomposedRegion] = field(default_factory=list)
    patterns: Dict[str, List[RegionCoveragePattern]] = field(default_factory=dict)
    graph: Optional[RegionGraph] = None
    assignment: Optional[BalancedAssignment] = None
    tours: Dict[int, SingleUsvTourPlan] = field(default_factory=dict)
    agents: Dict[int, AgentPathPlan] = field(default_factory=dict)
    coverage_state: Optional[CoverageState] = None
    path_plan: Optional[MultiAgentPathPlan] = None
    stage_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tsp_records: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    residual_backfill_count: int = 0
    mapf_conflicts_resolved: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class PathPlanningRequest:
    config: PlannerConfig
    path_config: Optional[PathPlanningConfig] = None
    strips: List[StripTask] = field(default_factory=list)
    assignments: Optional[AssignmentPlan] = None
    static_obstacles: List[StaticObstacle] = field(default_factory=list)
    dynamic_obstacles: List[DynamicObstacleTrack] = field(default_factory=list)
    existing_plan: Optional[PlanningResult] = None
    preferred_models: Tuple[DynamicsModelTag, ...] = (
        DynamicsModelTag.THREE_DOF,
        DynamicsModelTag.SIX_DOF,
    )
    paper_references: List[PaperReference] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
