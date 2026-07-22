"""End-to-end CROWN-MCPP planning pipeline over the existing geometry stack."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from math import ceil, hypot
from time import perf_counter
from typing import Mapping, Optional, Sequence, Tuple

from shapely import make_valid
from shapely.geometry import Polygon, box

from ...schema import PlannerConfig, Pose2D, VehicleFootprint
from ..coverage import build_coverage_state
from ..decomposition import decompose_obstacle_aware_area, decompose_rectangular_area
from ..obstacles import (
    distance_point_to_polygon,
    normalize_obstacle_field,
    point_in_any_obstacle,
    point_in_polygon,
)
from ..residuals import evaluate_tour_coverage_state
from ..types import (
    DecomposedRegion,
    MultiAgentPathPlan,
    ObstacleField,
    PaperReference,
    PathPlanningConfig,
    SingleUsvTourPlan,
    StaticObstacle,
)
from .adapters import bpc_solution_to_path_plan
from .baseline import build_crown_sequential_baseline
from .config import CrownMcppConfig
from .graph_bpc import solve_crown_graph_bpc
from .geometry import (
    CrownResponsibilityCertificate,
    _convex_parts,
    build_continuous_responsibility_regions,
    certify_continuous_responsibility_regions,
)
from .lns import solve_crown_lns
from .mode_graph import (
    CrownTimeExpandedModeGraph,
    build_agent_mode_graph,
    clone_agent_mode_graph,
)
from .motion import CrownCurrentField, ZeroCurrentField
from .types import CrownBpcSolution


@dataclass(frozen=True)
class CrownPreparedProblem:
    planner_config: PlannerConfig
    path_config: PathPlanningConfig
    crown_config: CrownMcppConfig
    regions: Tuple[DecomposedRegion, ...]
    obstacle_field: Optional[ObstacleField]
    graphs: Mapping[str, CrownTimeExpandedModeGraph]
    horizon: float
    responsibility_certificate: CrownResponsibilityCertificate

    @property
    def task_ids(self) -> Tuple[str, ...]:
        return tuple(region.region_id for region in self.regions)


def _common_safety_config(config: PlannerConfig) -> PlannerConfig:
    profiles = [
        config.profile_for_agent(agent_id)
        for agent_id in range(config.fleet.num_agents or len(config.fleet.initial_states_3dof))
    ]
    return replace(
        config,
        vehicle_footprint=VehicleFootprint(
            length=max(profile.vehicle_length for profile in profiles),
            width=max(profile.vehicle_width for profile in profiles),
        ),
    )


def _automatic_horizon(
    graphs: Mapping[str, CrownTimeExpandedModeGraph],
    task_ids: Sequence[str],
    config: PlannerConfig,
) -> float:
    time_step = max(
        graph.crown_config.time_step for graph in graphs.values()
    )
    service_bound = 0.0
    for task_id in task_ids:
        durations = [
            sum(
                max(
                    1,
                    int(ceil(primitive.duration / time_step - 1.0e-12)),
                )
                * time_step
                for primitive in mode.nominal_service_primitives
            )
            for graph in graphs.values()
            for mode in graph.modes_for_task(task_id)
        ]
        if not durations:
            raise ValueError(f"task {task_id!r} has no feasible heterogeneous coverage mode")
        service_bound += max(durations)
    diagonal = hypot(config.mission.area_length_x, config.mission.area_length_y)
    minimum_speed = min(
        min(graph.profile.cruise_speed, graph.profile.cover_speed, graph.profile.turn_speed_max)
        for graph in graphs.values()
    )
    # Service primitives occupy integer time slots, so summing their nominal
    # continuous durations can underestimate the horizon by hundreds of slots
    # after a curved path is sampled finely.  The service term above therefore
    # applies exactly the same per-primitive rounding as pricing.  Connector
    # geometry is lazy; scale its continuous detour allowance when the time
    # grid is coarser than a nominal primitive.
    primitive_duration = min(
        graph.crown_config.primitive_max_duration for graph in graphs.values()
    )
    connector_rounding = max(1.0, time_step / primitive_duration)
    connector_bound = (
        3.0
        * connector_rounding
        * (len(task_ids) + len(graphs) + 1)
        * diagonal
        / minimum_speed
    )
    return max(service_bound + connector_bound, time_step)


def _split_uncovered_regions_into_atomic_bands(
    regions: Sequence[DecomposedRegion],
    uncovered: Sequence[str],
    config: PlannerConfig,
    crown: CrownMcppConfig,
) -> Tuple[DecomposedRegion, ...]:
    """Refine an infeasible exact polygon without losing responsibility area."""

    uncovered_set = set(uncovered)
    refined = []
    for region in regions:
        if region.region_id not in uncovered_set:
            refined.append(region)
            continue
        x_min, y_min, x_max, y_max = region.bounds
        axis = (
            region.preferred_axis
            if region.preferred_axis in {"x", "y"}
            else ("x" if x_max - x_min >= y_max - y_min else "y")
        )
        source_polygon = make_valid(Polygon(region.polygon))
        tolerance = max(float(source_polygon.area) * 1.0e-10, 1.0e-12)
        convex_parts = _convex_parts(source_polygon, tolerance)
        part_specs = []
        refinement_kind = "convex"
        if len(convex_parts) > 1:
            part_specs = [
                (index, 0, part) for index, part in enumerate(convex_parts)
            ]
        else:
            cross_width = (
                (y_max - y_min) if axis == "x" else (x_max - x_min)
            )
            if cross_width <= 1.0e-9:
                refined.append(region)
                continue
            refinement_kind = "band"
            band_count = 2
            for index in range(band_count):
                if axis == "x":
                    low = y_min + cross_width * index / band_count
                    high = y_min + cross_width * (index + 1) / band_count
                    band = box(x_min, low, x_max, high)
                else:
                    low = x_min + cross_width * index / band_count
                    high = x_min + cross_width * (index + 1) / band_count
                    band = box(low, y_min, high, y_max)
                intersection = make_valid(source_polygon.intersection(band))
                parts = (
                    (intersection,)
                    if isinstance(intersection, Polygon)
                    else tuple(
                        part
                        for part in getattr(intersection, "geoms", ())
                        if isinstance(part, Polygon)
                    )
                )
                part_specs.extend(
                    (index, part_index, part)
                    for part_index, part in enumerate(parts)
                )

        children = []
        for index, part_index, part in part_specs:
            if part.area <= 1.0e-12:
                continue
            bx0, by0, bx1, by1 = (float(value) for value in part.bounds)
            centroid = part.centroid
            child = DecomposedRegion(
                region_id=(
                    f"{region.region_id}:crown-{refinement_kind}:"
                    f"{index}:part:{part_index}"
                ),
                bounds=(bx0, by0, bx1, by1),
                polygon=[
                    (float(x), float(y))
                    for x, y in tuple(part.exterior.coords)[:-1]
                ],
                center=(float(centroid.x), float(centroid.y)),
                area=float(part.area),
                preferred_axis=axis,
                source_algorithm=(
                    "crown_exact_convex_refinement"
                    if refinement_kind == "convex"
                    else "crown_atomic_sweep_band_preprocessing"
                ),
                metadata={
                    **region.metadata,
                    "crown_parent_region_id": region.region_id,
                    "crown_refinement_kind": refinement_kind,
                    "crown_refinement_index": str(index),
                },
            )
            children.append(child)
        if len(children) <= 1:
            refined.append(region)
            continue
        for point in region.required_coverage_points:
            containing = [
                child for child in children if point_in_polygon(point, child.polygon)
            ]
            owner = min(
                containing or children,
                key=lambda child: (
                    distance_point_to_polygon(point, child.polygon),
                    hypot(point[0] - child.center[0], point[1] - child.center[1]),
                    child.region_id,
                ),
            )
            owner.required_coverage_points.append(point)
        for child in children:
            child.metadata["crown_required_grid_point_count"] = str(
                len(child.required_coverage_points)
            )
        refined.extend(children)
    return tuple(refined)


def _assign_finite_grid_responsibilities(
    regions: Sequence[DecomposedRegion],
    config: PlannerConfig,
    path_config: PathPlanningConfig,
    obstacle_field: Optional[ObstacleField],
) -> Tuple[Tuple[float, float], ...]:
    """Assign polygon-contained grid points and return uncovered slivers."""

    if not regions:
        return ()
    for region in regions:
        region.required_coverage_points.clear()
    state = build_coverage_state(
        config,
        resolution=path_config.coverage_resolution,
    )
    ordered = tuple(sorted(regions, key=lambda region: region.region_id))
    missing = []
    for y in state.y_coords:
        for x in state.x_coords:
            point = (float(x), float(y))
            if (
                obstacle_field is not None
                and point_in_any_obstacle(point, obstacle_field, inflated=True)
            ):
                continue
            containing = [
                region for region in ordered if point_in_polygon(point, region.polygon)
            ]
            if not containing:
                missing.append(point)
                continue
            owner = min(containing, key=lambda region: region.region_id)
            owner.required_coverage_points.append(point)
    for region in ordered:
        region.metadata["crown_required_grid_point_count"] = str(
            len(region.required_coverage_points)
        )
    assigned = sum(len(region.required_coverage_points) for region in ordered)
    expected = sum(
        1
        for y in state.y_coords
        for x in state.x_coords
        if obstacle_field is None
        or not point_in_any_obstacle(
            (float(x), float(y)), obstacle_field, inflated=True
        )
    )
    if assigned + len(missing) != expected:
        raise AssertionError("CROWN finite-grid responsibility audit is inconsistent")
    return tuple(missing)


def prepare_crown_problem(
    config: PlannerConfig,
    path_config: Optional[PathPlanningConfig] = None,
    static_obstacles: Optional[Sequence[StaticObstacle]] = None,
    *,
    current_field: Optional[CrownCurrentField] = None,
) -> CrownPreparedProblem:
    """Build fixed cells, heterogeneous modes, connectors and error margins."""

    path = path_config or PathPlanningConfig.from_planner_config(config)
    crown = CrownMcppConfig.from_path_config(path)
    config.validate_agent_profiles()
    agent_count = config.fleet.num_agents or len(config.fleet.initial_states_3dof)
    if crown.goal_poses is not None and len(crown.goal_poses) != agent_count:
        raise ValueError("CROWN goal_poses length must match the fleet size")
    safe_config = _common_safety_config(config)
    safe_path = replace(
        path,
        # Exact BCD legitimately creates small trapezoids at obstacle event
        # lines.  Edge-aligned modes are essential for covering those cells;
        # the ordinary large-region area heuristic must not suppress them.
        oriented_sweep_min_area_factor=0.0,
        obstacle_inflation_extra=(
            path.obstacle_inflation_extra + crown.total_position_error
        ),
        obstacle_aware_astar_max_expansions=(
            crown.connector_max_expansions
            if path.obstacle_aware_astar_max_expansions <= 0
            else min(
                path.obstacle_aware_astar_max_expansions,
                crown.connector_max_expansions,
            )
        ),
        obstacle_aware_motion_lattice_max_expansions=min(
            path.obstacle_aware_motion_lattice_max_expansions,
            crown.connector_max_expansions,
        ),
    )
    obstacle_field = (
        normalize_obstacle_field(static_obstacles, safe_config, safe_path)
        if static_obstacles
        else None
    )
    fallback_regions = tuple(
        decompose_obstacle_aware_area(safe_config, safe_path, obstacle_field)
        if obstacle_field is not None and obstacle_field.inflated_obstacles
        else decompose_rectangular_area(safe_config, safe_path)
    )
    regions, responsibility_certificate = build_continuous_responsibility_regions(
        safe_config,
        safe_path,
        obstacle_field,
        fallback_regions,
    )
    if not regions:
        raise ValueError("CROWN decomposition produced no fixed responsibility units")
    missing_responsibilities = _assign_finite_grid_responsibilities(
        regions,
        safe_config,
        safe_path,
        obstacle_field,
    )
    if missing_responsibilities:
        raise AssertionError(
            "continuous CROWN responsibility polygons left finite-grid gaps: "
            + ",".join(str(point) for point in missing_responsibilities[:8])
        )
    field = current_field or ZeroCurrentField()

    # Do not pre-generate every region pattern for every agent here.  The mode
    # graph below is the authoritative feasibility check and already triggers
    # exact responsibility refinement for uncovered cells.  The former probe
    # duplicated the most expensive geometry work and bypassed homogeneous
    # fleet mode-library reuse.
    def build_graphs(
        fixed_regions: Sequence[DecomposedRegion],
    ) -> Mapping[str, CrownTimeExpandedModeGraph]:
        result = {}
        templates = {}
        for agent_id in range(agent_count):
            goal = None
            if crown.goal_poses is not None:
                raw_goal = crown.goal_poses[agent_id]
                goal = Pose2D(float(raw_goal[0]), float(raw_goal[1]), float(raw_goal[2]))
            profile = config.profile_for_agent(agent_id)
            template_key = (profile.fingerprint, profile.max_mission_time)
            template = templates.get(template_key)
            if template is None:
                graph = build_agent_mode_graph(
                    agent_id,
                    fixed_regions,
                    config,
                    safe_path,
                    crown,
                    obstacle_field,
                    current_field=field,
                    goal_pose=goal,
                )
                templates[template_key] = graph
            else:
                graph = clone_agent_mode_graph(
                    template,
                    agent_id,
                    config,
                    safe_path,
                    crown,
                    goal_pose=goal,
                )
            result[str(agent_id)] = graph
        return result

    graphs = build_graphs(regions)
    task_ids = tuple(region.region_id for region in regions)
    uncovered = [
        task_id
        for task_id in task_ids
        if not any(graph.modes_for_task(task_id) for graph in graphs.values())
    ]
    refinement_round = 0
    while uncovered and refinement_round < 6:
        refinement_round += 1
        refined_regions = _split_uncovered_regions_into_atomic_bands(
            regions,
            uncovered,
            config,
            crown,
        )
        if tuple(region.region_id for region in refined_regions) != tuple(
            region.region_id for region in regions
        ):
            regions = refined_regions
            graphs = build_graphs(regions)
            task_ids = tuple(region.region_id for region in regions)
        uncovered = [
            task_id
            for task_id in task_ids
            if not any(graph.modes_for_task(task_id) for graph in graphs.values())
        ]
    if uncovered:
        raise ValueError(
            "fixed responsibility units without any feasible agent mode: "
            + ",".join(uncovered)
        )
    responsibility_certificate = certify_continuous_responsibility_regions(
        safe_config,
        obstacle_field,
        regions,
    )
    horizon = crown.horizon or _automatic_horizon(graphs, task_ids, config)
    return CrownPreparedProblem(
        planner_config=config,
        path_config=safe_path,
        crown_config=crown,
        regions=regions,
        obstacle_field=obstacle_field,
        graphs=graphs,
        horizon=horizon,
        responsibility_certificate=responsibility_certificate,
    )


def solve_prepared_crown_problem(problem: CrownPreparedProblem) -> CrownBpcSolution:
    baseline: Tuple = ()
    baseline_started = perf_counter()
    if problem.crown_config.include_sequential_baseline:
        try:
            baseline = build_crown_sequential_baseline(
                problem.graphs,
                problem.task_ids,
                horizon=problem.horizon,
                deadline=(
                    perf_counter()
                    + problem.crown_config.baseline_time_budget_sec
                ),
            )
        except ValueError:
            # A failed prioritized order is not a proof of model
            # infeasibility; exact BPC can still recover through joint pricing.
            baseline = ()
    baseline_runtime = (
        perf_counter() - baseline_started
        if problem.crown_config.include_sequential_baseline
        else 0.0
    )
    engine = problem.crown_config.engine
    if engine == "auto":
        engine = (
            "bpc"
            if len(problem.graphs) <= problem.crown_config.exact_max_agents
            and len(problem.regions) <= problem.crown_config.exact_max_regions
            else "certified_lns"
        )
    if engine == "bpc":
        solution = solve_crown_graph_bpc(
            problem.graphs,
            problem.task_ids,
            horizon=problem.horizon,
            initial_routes=baseline,
        )
    elif engine in {"lns", "certified_lns"}:
        solution = solve_crown_lns(
            problem.graphs,
            problem.task_ids,
            horizon=problem.horizon,
            initial_routes=baseline,
        )
    else:
        raise ValueError(f"unsupported CROWN engine: {engine}")
    if baseline_runtime > 0.0 and solution.anytime_trace:
        solution = replace(
            solution,
            anytime_trace=tuple(
                {
                    **dict(item),
                    "time": float(item.get("time", 0.0)) + baseline_runtime,
                }
                for item in solution.anytime_trace
            ),
        )
    return solution


def run_crown_mcpp_pipeline(
    config: PlannerConfig,
    path_config: Optional[PathPlanningConfig] = None,
    static_obstacles: Optional[Sequence[StaticObstacle]] = None,
    paper_references: Optional[Sequence[PaperReference]] = None,
    *,
    current_field: Optional[CrownCurrentField] = None,
) -> MultiAgentPathPlan:
    """Run the complete CROWN-MCPP chain and materialize executable paths."""

    pipeline_started = perf_counter()
    problem = prepare_crown_problem(
        config,
        path_config,
        static_obstacles,
        current_field=current_field,
    )
    preparation_finished = perf_counter()
    solution = solve_prepared_crown_problem(problem)
    solve_finished = perf_counter()
    plan = bpc_solution_to_path_plan(
        solution,
        paper_references=tuple(paper_references or ()),
    )
    plan.algorithm_name = "crown_mcpp"
    for agent in plan.agents.values():
        agent.source_algorithm = "crown_mcpp"
    selected_tasks = [
        task_id for route in solution.timed_routes for task_id in route.task_ids
    ]
    if sorted(selected_tasks) != sorted(problem.task_ids):
        raise AssertionError("CROWN materialization lost the exact-once responsibility cover")
    coverage_state = evaluate_tour_coverage_state(
        config,
        [
            SingleUsvTourPlan(
                agent_id=agent_id,
                region_order=[],
                selected_patterns={},
                segments=list(agent.segments),
            )
            for agent_id, agent in plan.agents.items()
        ],
        resolution=problem.path_config.coverage_resolution,
        obstacle_field=problem.obstacle_field,
        include_non_cover_segments=False,
    )
    if (
        coverage_state.coverage_fraction + 1.0e-9
        < problem.path_config.target_coverage_fraction
    ):
        raise AssertionError(
            "CROWN selected modes failed end-to-end coverage validation: "
            f"{coverage_state.coverage_fraction:.9f} < "
            f"{problem.path_config.target_coverage_fraction:.9f}"
        )
    plan.metadata.update(
        {
            "status": solution.solution_status,
            "engine": solution.solution_status,
            "fixed_region_count": str(len(problem.regions)),
            "covered_responsibility_count": str(len(selected_tasks)),
            "coverage_fraction": f"{coverage_state.coverage_fraction:.9f}",
            "coverage_target": (
                f"{problem.path_config.target_coverage_fraction:.9f}"
            ),
            "coverage_validated": "true",
            "continuous_responsibility_validated": str(
                problem.responsibility_certificate.valid
            ).lower(),
            "responsibility_gap_area": (
                f"{problem.responsibility_certificate.gap_area:.12g}"
            ),
            "responsibility_spill_area": (
                f"{problem.responsibility_certificate.spill_area:.12g}"
            ),
            "responsibility_overlap_area": (
                f"{problem.responsibility_certificate.overlap_area:.12g}"
            ),
            "agent_count": str(len(problem.graphs)),
            "mode_count": str(
                sum(
                    len(modes)
                    for graph in problem.graphs.values()
                    for modes in graph.modes_by_task.values()
                )
            ),
            "horizon": f"{problem.horizon:.9f}",
            "root_lp_lower_bound": (
                ""
                if solution.root_lp_lower_bound is None
                else f"{solution.root_lp_lower_bound:.9f}"
            ),
            "service_lower_bound": (
                ""
                if solution.service_lower_bound is None
                else f"{solution.service_lower_bound:.9f}"
            ),
            "sequential_baseline_makespan": (
                ""
                if solution.baseline_makespan is None
                else f"{solution.baseline_makespan:.9f}"
            ),
            "sequential_baseline_energy": (
                ""
                if solution.baseline_energy is None
                else f"{solution.baseline_energy:.9f}"
            ),
            "joint_not_worse_than_sequential_baseline": str(
                solution.baseline_makespan is None
                or solution.makespan <= solution.baseline_makespan + 1.0e-9
            ).lower(),
            "pricing_labels": str(solution.pricing_labels),
            "pricing_labels_dominated": str(solution.pricing_labels_dominated),
            "position_error_total": f"{problem.crown_config.total_position_error:.9f}",
            "effective_planning_distance": f"{config.safety.d_safe + 2.0 * problem.crown_config.total_position_error:.9f}",
            "continuous_conflict_validated": str(
                problem.crown_config.enable_continuous_conflict_validation
            ).lower(),
            "return_to_start": str(problem.crown_config.return_to_start).lower(),
            "certification_scope": (
                "continuous_responsibility_fixed_modes_time_grid_horizon"
            ),
            "anytime_trace_json": json.dumps(
                list(solution.anytime_trace),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "static_obstacle_count": str(len(static_obstacles or ())),
            "preparation_runtime_sec": (
                f"{preparation_finished - pipeline_started:.9f}"
            ),
            "solve_runtime_sec": f"{solve_finished - preparation_finished:.9f}",
            "materialization_validation_runtime_sec": (
                f"{perf_counter() - solve_finished:.9f}"
            ),
        }
    )
    return plan
