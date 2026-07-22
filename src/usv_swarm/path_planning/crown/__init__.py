"""CROWN-MCPP minimal exact solvers.

``solve_joint_exact`` is an independent continuous-time full-enumeration oracle.
``solve_crown_bpc`` is the finite time-expanded branch-price-and-cut solver.
The two implementations intentionally share data types but not optimization
logic, allowing small-instance cross-validation.
"""

from .adapters import (
    CrownSegmentRouteCandidate,
    bpc_solution_to_path_plan,
    build_crown_instance_from_segment_candidates,
    crown_route_from_path_segments,
)
from .bpc import solve_crown_bpc
from .baseline import build_crown_sequential_baseline
from .config import CrownMcppConfig
from .conflicts import (
    CrownContinuousConflict,
    CrownResourceMappingError,
    assert_continuous_conflict_free,
    find_continuous_conflicts,
)
from .exact_oracle import (
    compare_joint_and_sequential,
    solve_joint_exact,
    solve_sequential_exact_post,
)
from .exact_scheduler import (
    assert_schedule_resource_feasible,
    schedule_selected_routes_exact,
)
from .experiment import run_shared_corridor_proof_experiment
from .full_experiment import run_crown_mcpp_experiment
from .instances import build_shared_corridor_counterexample
from .graph_bpc import (
    CrownRootRelaxation,
    service_workload_lower_bound,
    solve_crown_graph_bpc,
    solve_crown_root_relaxation,
)
from .geometry import (
    CrownCoverageCertificate,
    CrownResponsibilityCertificate,
    build_continuous_responsibility_regions,
    certify_continuous_pattern_coverage,
    certify_continuous_responsibility_regions,
)
from .lns import solve_crown_lns
from .mode_graph import (
    CrownGeometricConnection,
    CrownGeometricMode,
    CrownTimeExpandedModeGraph,
    build_agent_mode_graph,
)
from .motion import (
    CrownMotionPrimitive,
    CurrentInfeasibleError,
    UniformCurrentField,
    ZeroCurrentField,
    conservative_tube_cells,
)
from .pipeline import (
    CrownPreparedProblem,
    prepare_crown_problem,
    run_crown_mcpp_pipeline,
    solve_prepared_crown_problem,
)
from .pricing import (
    CrownPricingDuals,
    CrownPricingPrecedenceDual,
    CrownPricingRestrictions,
    CrownPricingResult,
    PricingLabelLimitExceeded,
    price_mode_graph_exact,
)
from .resource_model import build_time_expanded_route_universe, expand_route_on_time_grid
from .route_enumerator import enumerate_agent_routes, enumerate_route_universe
from .types import (
    CrownBpcSolution,
    CrownConnection,
    CrownInstance,
    CrownMode,
    CrownOperation,
    CrownRoute,
    CrownSchedule,
    CrownSolution,
    CrownTimedRoute,
)

__all__ = [
    "CrownBpcSolution",
    "CrownContinuousConflict",
    "CrownCoverageCertificate",
    "CrownConnection",
    "CrownGeometricConnection",
    "CrownGeometricMode",
    "CrownInstance",
    "CrownMode",
    "CrownMcppConfig",
    "CrownMotionPrimitive",
    "CrownOperation",
    "CrownRoute",
    "CrownRootRelaxation",
    "CrownSchedule",
    "CrownSegmentRouteCandidate",
    "CrownSolution",
    "CrownTimedRoute",
    "CrownTimeExpandedModeGraph",
    "CrownPreparedProblem",
    "CrownPricingDuals",
    "CrownPricingPrecedenceDual",
    "CrownPricingRestrictions",
    "CrownPricingResult",
    "CrownResourceMappingError",
    "CrownResponsibilityCertificate",
    "CurrentInfeasibleError",
    "PricingLabelLimitExceeded",
    "UniformCurrentField",
    "ZeroCurrentField",
    "assert_continuous_conflict_free",
    "assert_schedule_resource_feasible",
    "bpc_solution_to_path_plan",
    "build_crown_instance_from_segment_candidates",
    "build_continuous_responsibility_regions",
    "build_crown_sequential_baseline",
    "build_agent_mode_graph",
    "build_shared_corridor_counterexample",
    "build_time_expanded_route_universe",
    "compare_joint_and_sequential",
    "conservative_tube_cells",
    "certify_continuous_pattern_coverage",
    "certify_continuous_responsibility_regions",
    "crown_route_from_path_segments",
    "enumerate_agent_routes",
    "enumerate_route_universe",
    "expand_route_on_time_grid",
    "find_continuous_conflicts",
    "prepare_crown_problem",
    "price_mode_graph_exact",
    "run_shared_corridor_proof_experiment",
    "run_crown_mcpp_pipeline",
    "run_crown_mcpp_experiment",
    "schedule_selected_routes_exact",
    "solve_crown_bpc",
    "solve_crown_graph_bpc",
    "solve_crown_lns",
    "solve_crown_root_relaxation",
    "solve_prepared_crown_problem",
    "solve_joint_exact",
    "solve_sequential_exact_post",
    "service_workload_lower_bound",
]
