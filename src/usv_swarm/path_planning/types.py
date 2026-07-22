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
    turn_count_weight: float = 0.2
    time_weight: float = 1.0
    load_balance_weight: float = 0.5
    safety_weight: float = 0.25
    boundary_weight: float = 0.1
    curvature_weight: float = 100.0
    astar_heading_weight: float = 0.35
    astar_safety_weight: float = 0.5
    astar_boundary_weight: float = 0.2
    tsp_solver: str = "deterministic"
    tsp_2opt_iterations: int = 25
    tsp_3opt_iterations: int = 0
    region_tsp_beam_width: int = 6
    region_tsp_branch_limit: int = 12
    aco_ant_count: int = 30
    aco_iterations: int = 80
    aco_alpha: float = 1.0
    aco_beta: float = 3.0
    aco_rho: float = 0.35
    aco_q: float = 100.0
    aco_random_seed: int = 42
    fa3aco_rho_min: float = 0.08
    fa3aco_rho_max: float = 0.55
    fa3aco_rho_decay: float = 0.035
    fa3aco_fractional_order: float = 0.65
    fa3aco_memory_depth: int = 4
    fa3aco_enable_3opt: bool = True
    performance_profile: str = "balanced"
    target_coverage_fraction: float = 0.99
    cell_merge_width_factor: float = 1.5
    min_pass_length_factor: float = 2.5
    transition_length_weight: float = 1.0
    repeat_transition_weight: float = 12.0
    residual_penalty_weight: float = 100.0
    count_transit_coverage: bool = True
    enable_short_region_compression: bool = True
    short_region_turn_ratio_threshold: float = 6.0
    compressed_region_width_factor: float = 1.25
    min_compressed_pattern_coverage_fraction: float = 0.98
    min_sweep_pattern_coverage_fraction: float = 0.95
    coverage_priority_weight: float = 500.0
    pattern_retraction_penalty_weight: float = 2.0
    pattern_turn_penalty_weight: float = 0.5
    pattern_repeat_penalty_multiplier: float = 1.5
    connector_noncover_repeat_penalty_multiplier: float = 2.5
    use_bezier_smoothing: bool = True
    load_imbalance_tolerance: float = 0.10
    enable_lightweight_load_swap: bool = True
    load_swap_max_iterations: int = 4
    obstacle_circle_segments: int = 24
    obstacle_inflation_extra: float = 0.0
    coverage_turn_pocket_scale: float = 1.0
    enable_adaptive_pass_retraction: bool = True
    retraction_search_iterations: int = 16
    max_pass_retraction_ratio: float = 0.75
    retraction_min_pass_length_factor: float = 0.25
    enable_oriented_sweep_patterns: bool = True
    max_oriented_sweep_angles_per_region: int = 4
    oriented_sweep_angle_tolerance_deg: float = 5.0
    include_axis_aligned_sweep_fallbacks: bool = True
    oriented_sweep_min_area_factor: float = 4.0
    enable_multi_entry_exit_patterns: bool = True
    max_entry_exit_patterns_per_region: int = 12
    multi_entry_exit_coverage_floor: float = 0.65
    min_free_cell_size: Optional[float] = None
    enable_residual_backfill: bool = True
    enable_residual_local_tsp: bool = True
    repeat_path_penalty_weight: float = 8.0
    shared_resource_grid_size: float = 1.0
    resource_separation_time: float = 0.5
    max_residual_backfill_regions: int = 12
    residual_backfill_cycles: int = 3
    residual_local_tsp_time_budget_sec: float = 20.0
    residual_local_tsp_max_candidate_attempts: int = 400
    residual_min_gain_per_path_meter: float = 0.03
    residual_gain_reward_weight: float = 20.0
    skipped_region_recovery_time_budget_sec: float = 20.0
    cover_only_target_fraction: Optional[float] = None
    residual_filter_after_target_only: bool = True
    report_score_components: bool = True
    enable_main_repeat_path_penalty: bool = True
    main_repeat_path_penalty_weight: float = 12.0
    internal_uturn_repeat_path_penalty_weight: float = 12.0
    enable_cross_agent_coverage_penalty: bool = True
    cross_agent_transit_penalty_weight: float = 10.0
    cross_agent_cover_penalty_weight: float = 16.0
    cross_agent_initial_escape_free_distance: Optional[float] = None
    cross_agent_overlap_grid_size: Optional[float] = None
    enable_large_map_sweep_prefilter: bool = False
    large_map_size_threshold: float = 50.0
    max_prefiltered_patterns_per_region: int = 4
    max_prefiltered_variants_per_pattern: int = 4
    enable_uturn_validation_cache: bool = True
    enable_infeasible_uturn_region_repair: bool = True
    enable_motion_lattice_heading_repair: bool = True
    max_large_map_region_repair_depth: int = 2
    large_map_stop_after_first_feasible_sweep_variant: bool = True
    large_map_validate_internal_uturn_repair: bool = False
    sweep_region_validation_time_budget_sec: float = 30.0
    obstacle_aware_grid_resolution: Optional[float] = None
    obstacle_aware_astar_max_expansions: int = 0
    obstacle_aware_motion_lattice_max_expansions: int = 16000
    obstacle_aware_allow_motion_lattice: bool = True
    obstacle_aware_allow_corridor_conversion: bool = True
    large_map_tsp_agent_time_budget_sec: float = 480.0
    large_map_tsp_total_time_budget_sec: float = 0.0
    large_map_tsp_step_time_budget_sec: float = 60.0
    large_map_tsp_max_candidate_attempts_per_step: int = 12
    large_map_tsp_obstacle_aware_retry_limit: int = 1
    large_map_tsp_max_obstacle_aware_attempts_per_step: int = 1
    large_map_tsp_max_obstacle_aware_attempts_per_agent: int = 8
    large_map_tsp_obstacle_aware_max_transition_length: float = 80.0
    large_map_tsp_enable_lookahead_probe: bool = True
    large_map_tsp_require_cheap_connector_probe: bool = False
    large_map_tsp_cheap_probe_collision_only: bool = False
    enable_large_map_dead_end_restart: bool = True
    large_map_dead_end_restart_limit: int = 1
    large_map_dead_end_restart_trigger_ratio: float = 0.50
    enable_large_convex_region_decomposition: bool = True
    large_region_max_area_fraction: float = 0.45
    large_region_min_width_factor: float = 1.0
    large_region_connector_pattern_limit: int = 6
    prioritize_region_execution: bool = True
    enable_coverage_aware_merge: bool = True
    coverage_merge_beam_width: int = 8
    coverage_merge_max_members: int = 48
    coverage_merge_max_area_fraction: float = 0.35
    coverage_merge_min_improvement_ratio: float = 0.03
    coverage_merge_min_coverage_fraction: float = 0.98
    coverage_merge_allow_nonconvex_composite: bool = True
    coverage_merge_gap_bridge_width_factor: float = 2.0
    coverage_merge_preview_pattern_limit: int = 4
    coverage_merge_validate_top_k: int = 3
    coverage_merge_max_candidate_evaluations: int = 400
    coverage_merge_max_validations: int = 80
    coverage_merge_time_budget_sec: float = 180.0
    coverage_merge_no_improvement_patience: int = 3
    coverage_merge_skip_pre_assignment_large_region_count: int = 40
    enable_short_region_connector_recovery: bool = True
    enable_agent_task_region_merge: bool = True
    enable_agent_task_lightweight_strip_merge: bool = False
    agent_task_strip_merge_max_candidate_evaluations: int = 24
    agent_task_strip_merge_time_budget_sec: float = 20.0
    agent_task_strip_merge_use_geometric_preview: bool = True
    agent_task_strip_merge_max_groups_per_agent: int = 3
    agent_task_strip_merge_min_rectangularity: float = 0.82
    agent_task_strip_full_component_direct_priority_rectangularity: float = 0.95
    agent_task_strip_merge_min_length_gain_factor: float = 1.15
    agent_task_merge_max_area_fraction: float = 0.50
    agent_task_merge_min_improvement_ratio: float = 0.0
    agent_task_merge_time_budget_sec: float = 60.0
    agent_task_merge_enable_pairwise_fallback: bool = True
    agent_task_merge_enable_unified_group_merge: bool = True
    agent_task_merge_min_unified_group_size: int = 3
    agent_task_merge_max_unified_group_size: int = 0
    agent_task_merge_prefer_full_components: bool = True
    agent_task_merge_full_component_max_regions: int = 16
    agent_task_merge_full_component_min_rectangularity: float = 0.55
    agent_task_merge_max_unified_candidates_per_agent: int = 8
    agent_task_merge_min_unified_rectangularity: float = 0.65
    agent_task_merge_keep_coherent_negative_objective: bool = True
    agent_task_runtime_fallback_accept_neutral_source_expansion: bool = True
    enable_composite_free_space_regions: bool = True
    composite_max_member_cells: int = 32
    composite_max_region_area_fraction: float = 0.20
    composite_gap_bridge_factor: float = 0.75
    enable_open_sweep_chain_tsp: bool = False
    max_open_chains_per_region: int = 24
    open_chain_tsp_beam_width: int = 8
    open_chain_coverage_reward_weight: float = 600.0
    open_chain_connector_penalty_weight: float = 1.0
    open_chain_skip_penalty_weight: float = 1000.0
    open_chain_allow_flexible_exit: bool = False
    enable_open_chain_flexible_exit_variants: bool = True
    open_chain_flexible_exit_variant_limit: int = 1
    open_chain_flexible_exit_variants_for_agent_task_only: bool = True
    enable_rmin_aware_chain_order: bool = True
    chain_turn_strategy: str = "rmin_180"
    rmin_chain_turn_clearance_factor: float = 0.25
    rmin_chain_max_stride: int = 8
    rmin_chain_min_pass_length_factor: float = 1.0
    enable_joint_region_candidate_optimization: bool = True
    joint_candidate_patterns_per_region: int = 6
    joint_connector_edge_limit: int = 12
    joint_improvement_iterations: int = 8
    joint_large_map_region_limit: int = 40
    joint_optimizer_time_budget_sec: float = 90.0
    joint_eval_agent_time_budget_sec: float = 30.0
    joint_eval_step_time_budget_sec: float = 4.0
    enable_integrated_residual_candidates: bool = True
    enable_global_route_refinement: bool = True
    route_refinement_iterations: int = 3
    global_cross_agent_overlap_weight: float = 20.0
    global_noncover_repeat_weight: float = 18.0
    global_turn_angle_weight: float = 1.0
    enable_heterogeneous_connected_assignment: bool = True
    assignment_objective: str = "lexicographic_makespan"
    max_agent_pattern_previews_per_region: int = 6
    monotone_merge_angle_tolerance_deg: float = 10.0
    monotone_merge_beam_width: int = 12
    monotone_merge_min_time_gain_ratio: float = 0.02
    oversized_region_split_ratio: float = 1.25
    joint_assignment_iterations: int = 10
    require_connected_sweep_task: bool = True
    enable_contour_residual_fallback: bool = True
    monitor_stages: bool = False
    visual_output_dir: Optional[str] = None
    visual_map_id: Optional[str] = None
    visual_dpi: int = 180
    visual_gif_fps: int = 6
    # CROWN-MCPP: finite-model accuracy, exact/scalable engines and certificates.
    crown_engine: str = "auto"
    crown_time_step: float = 1.0
    crown_horizon: Optional[float] = None
    crown_mode_limit_per_region_agent: int = 8
    crown_exact_max_agents: int = 4
    crown_exact_max_regions: int = 10
    crown_max_tasks_per_route: Optional[int] = None
    crown_max_timed_columns: int = 200_000
    crown_position_error_map: float = 0.0
    crown_position_error_discretization: float = 0.0
    crown_position_error_tracking: float = 0.0
    crown_resource_grid_size: Optional[float] = None
    crown_primitive_max_duration: float = 2.0
    crown_enable_continuous_conflict_validation: bool = True
    crown_return_to_start: bool = True
    crown_goal_poses: Optional[Tuple[Tuple[float, float, float], ...]] = None
    crown_include_sequential_baseline: bool = True
    crown_baseline_time_budget_sec: float = 30.0
    crown_connector_max_expansions: int = 2000
    crown_root_exact_pricing: bool = True
    crown_pricing_label_limit: int = 1_000_000
    crown_lns_iterations: int = 500
    crown_lns_time_budget_sec: float = 60.0
    crown_lns_destroy_fraction: float = 0.2
    crown_lns_pool_reopt_interval: int = 20
    crown_lns_random_seed: int = 42
    crown_lns_max_route_pool_per_agent: int = 200
    crown_report_anytime_trace: bool = True

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
    # Optional finite-grid audit points.  CROWN responsibility semantics are
    # carried by ``polygon`` and certified continuously; these samples are a
    # secondary regression check and never replace missing polygon geometry.
    required_coverage_points: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class CompositeFreeSpaceRegion(DecomposedRegion):
    """A logical coverage region made from multiple obstacle-free cells.

    ``bounds`` is only an index/visualization envelope; coverage generation must
    use ``member_cells`` so obstacle holes or missing cells are not treated as
    free space.
    """

    member_cells: List["FreeSpaceCell"] = field(default_factory=list)
    free_intervals_by_axis: Dict[str, List[Tuple[float, List[Tuple[float, float]]]]] = field(default_factory=dict)


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
class OpenSweepBreak:
    region_id: str
    pattern_id: str
    before_pass_id: Optional[str]
    after_pass_id: Optional[str]
    reason: str
    direct_reasons: List[str] = field(default_factory=list)
    repair_attempted: bool = False
    repair_success: bool = False
    rejected_connector_sources: List[str] = field(default_factory=list)


@dataclass
class OpenSweepChain:
    chain_id: str
    region_id: str
    pattern_id: str
    pass_indices: List[int]
    passes: List[CoveragePass]
    entry_pose: Pose2D
    exit_pose: Pose2D
    reverse_entry_pose: Pose2D
    reverse_exit_pose: Pose2D
    internal_segments: List[PathSegmentSpec] = field(default_factory=list)
    coverage_length: float = 0.0
    internal_turn_length: float = 0.0
    estimated_time: float = 0.0
    max_curvature: float = 0.0
    feasible: bool = True
    left_break_reason: str = ""
    right_break_reason: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


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
    open_chains: List[OpenSweepChain] = field(default_factory=list)
    chain_order: List[str] = field(default_factory=list)
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
class CoverageOwnershipMap:
    resolution: float
    owner_by_cell: Dict[str, int] = field(default_factory=dict)
    region_owner: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)


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
