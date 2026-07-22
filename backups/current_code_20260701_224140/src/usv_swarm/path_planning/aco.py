from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

from ..schema import Pose2D
from .types import PathPlanningConfig, RegionCoveragePattern


VALID_TSP_SOLVERS = {"deterministic", "aco", "fa3aco"}
EdgeCostFn = Callable[[RegionCoveragePattern | None, RegionCoveragePattern], float]


@dataclass
class AcoTspResult:
    status: str
    requested_solver: str
    effective_solver: str
    region_order: List[str] = field(default_factory=list)
    selected_patterns: Dict[str, RegionCoveragePattern] = field(default_factory=dict)
    objective: float = float("inf")
    initial_objective: float = float("inf")
    convergence_trace: List[float] = field(default_factory=list)
    accepted_3opt_count: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class _AcoSolution:
    route: List[RegionCoveragePattern]
    objective: float
    edge_keys: List[Tuple[str, str]]

    @property
    def region_order(self) -> List[str]:
        return [pattern.region_id for pattern in self.route]

    @property
    def selected_patterns(self) -> Dict[str, RegionCoveragePattern]:
        return {pattern.region_id: pattern for pattern in self.route}


def validate_tsp_solver(solver: str) -> str:
    normalized = str(solver or "deterministic").strip().lower()
    if normalized not in VALID_TSP_SOLVERS:
        allowed = ", ".join(sorted(VALID_TSP_SOLVERS))
        raise ValueError(f"unsupported tsp_solver {solver!r}; expected one of: {allowed}")
    return normalized


def solve_aco_tsp_cpp(
    region_ids: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    start_pose: Pose2D,
    path_config: PathPlanningConfig,
    edge_cost_fn: EdgeCostFn,
    solver: str | None = None,
) -> AcoTspResult:
    requested = validate_tsp_solver(solver or path_config.tsp_solver)
    if requested == "deterministic":
        raise ValueError("solve_aco_tsp_cpp only accepts aco or fa3aco")
    region_ids = sorted(dict.fromkeys(region_ids))
    if not region_ids:
        return AcoTspResult(status="success", requested_solver=requested, effective_solver=requested, objective=0.0)
    if any(not patterns.get(region_id) for region_id in region_ids):
        return _failed_result(requested, "missing_candidate_patterns", region_ids)

    rng = random.Random(path_config.aco_random_seed)
    edge_cache: Dict[Tuple[str, str], float] = {}

    def cached_cost(previous: RegionCoveragePattern | None, candidate: RegionCoveragePattern) -> float:
        key = (_node_key(previous), _node_key(candidate))
        if key not in edge_cache:
            edge_cache[key] = float(edge_cost_fn(previous, candidate))
        return edge_cache[key]

    greedy = _greedy_solution(region_ids, patterns, cached_cost)
    if greedy is None:
        return _failed_result(requested, "no_finite_initial_tour", region_ids, edge_evaluation_count=len(edge_cache))

    best = greedy
    pheromone: Dict[Tuple[str, str], float] = {}
    convergence: List[float] = []
    accepted_3opt = 0
    ant_count = max(int(path_config.aco_ant_count), 1)
    iterations = max(int(path_config.aco_iterations), 1)

    for iteration in range(iterations):
        iteration_solutions: List[_AcoSolution] = []
        for _ in range(ant_count):
            ant = _construct_ant_solution(
                region_ids,
                patterns,
                cached_cost,
                pheromone,
                path_config,
                rng,
                use_fractional_memory=requested == "fa3aco",
            )
            if ant is not None:
                iteration_solutions.append(ant)
        if iteration_solutions:
            iteration_best = min(iteration_solutions, key=lambda item: item.objective)
            if requested == "fa3aco" and path_config.fa3aco_enable_3opt:
                improved = _three_opt_improve(iteration_best, patterns, cached_cost, max_candidates=80)
                if improved.objective + 1e-9 < iteration_best.objective:
                    iteration_best = improved
                    accepted_3opt += 1
            if iteration_best.objective + 1e-9 < best.objective:
                best = iteration_best
        rho = _evaporation_rate(iteration, path_config, requested)
        _update_pheromone(pheromone, iteration_solutions, best, rho, path_config.aco_q)
        convergence.append(float(best.objective))

    return AcoTspResult(
        status="success",
        requested_solver=requested,
        effective_solver=requested,
        region_order=best.region_order,
        selected_patterns=best.selected_patterns,
        objective=float(best.objective),
        initial_objective=float(greedy.objective),
        convergence_trace=convergence,
        accepted_3opt_count=accepted_3opt,
        metadata={
            "initial_order": greedy.region_order,
            "edge_evaluation_count": len(edge_cache),
            "iteration_count": iterations,
            "ant_count": ant_count,
            "start_pose": [start_pose.x, start_pose.y, start_pose.psi],
        },
    )


def _construct_ant_solution(
    region_ids: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    cost_fn: EdgeCostFn,
    pheromone: Dict[Tuple[str, str], float],
    path_config: PathPlanningConfig,
    rng: random.Random,
    use_fractional_memory: bool,
) -> _AcoSolution | None:
    remaining = set(region_ids)
    current: RegionCoveragePattern | None = None
    route: List[RegionCoveragePattern] = []
    edge_keys: List[Tuple[str, str]] = []
    objective = 0.0
    probability_history: List[Dict[str, float]] = []
    while remaining:
        options: List[Tuple[RegionCoveragePattern, float, float, Tuple[str, str]]] = []
        current_key = _node_key(current)
        for region_id in sorted(remaining):
            for candidate in patterns.get(region_id, []):
                edge_key = (current_key, _node_key(candidate))
                cost = float(cost_fn(current, candidate))
                if not math.isfinite(cost):
                    continue
                tau = max(pheromone.get(edge_key, 1.0), 1e-9)
                eta = 1.0 / max(cost, 1e-9)
                desirability = (tau ** path_config.aco_alpha) * (eta ** path_config.aco_beta)
                options.append((candidate, cost, desirability, edge_key))
        if not options:
            return None
        total = sum(item[2] for item in options)
        if total <= 0.0 or not math.isfinite(total):
            return None
        probabilities = {item[3][1]: item[2] / total for item in options}
        if use_fractional_memory:
            weighted_options = []
            for candidate, cost, desirability, edge_key in options:
                memory = _fractional_memory(edge_key[1], probability_history, path_config)
                weighted_options.append((candidate, cost, desirability * (1.0 + memory), edge_key))
            options = weighted_options
            total = sum(item[2] for item in options)
            if total <= 0.0 or not math.isfinite(total):
                return None
            probabilities = {item[3][1]: item[2] / total for item in options}
        selected, cost, _, edge_key = _weighted_choice(options, rng)
        probability_history.append(probabilities)
        route.append(selected)
        edge_keys.append(edge_key)
        objective += cost
        remaining.remove(selected.region_id)
        current = selected
    return _AcoSolution(route=route, objective=objective, edge_keys=edge_keys)


def _greedy_solution(
    region_ids: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    cost_fn: EdgeCostFn,
) -> _AcoSolution | None:
    remaining = set(region_ids)
    current: RegionCoveragePattern | None = None
    route: List[RegionCoveragePattern] = []
    edge_keys: List[Tuple[str, str]] = []
    objective = 0.0
    while remaining:
        candidates = []
        current_key = _node_key(current)
        for region_id in sorted(remaining):
            for candidate in patterns.get(region_id, []):
                cost = float(cost_fn(current, candidate))
                if math.isfinite(cost):
                    candidates.append((cost, candidate.pattern_id, candidate, (current_key, _node_key(candidate))))
        if not candidates:
            return None
        cost, _, selected, edge_key = min(candidates, key=lambda item: (item[0], item[1]))
        route.append(selected)
        edge_keys.append(edge_key)
        objective += cost
        remaining.remove(selected.region_id)
        current = selected
    return _AcoSolution(route=route, objective=objective, edge_keys=edge_keys)


def _three_opt_improve(
    solution: _AcoSolution,
    patterns: Dict[str, List[RegionCoveragePattern]],
    cost_fn: EdgeCostFn,
    max_candidates: int,
) -> _AcoSolution:
    best = solution
    checked = 0
    for order in _three_opt_orders(solution.region_order):
        candidate = _select_patterns_for_order(order, patterns, cost_fn)
        checked += 1
        if candidate is not None and candidate.objective + 1e-9 < best.objective:
            best = candidate
        if checked >= max_candidates:
            break
    return best


def _select_patterns_for_order(
    order: Sequence[str],
    patterns: Dict[str, List[RegionCoveragePattern]],
    cost_fn: EdgeCostFn,
) -> _AcoSolution | None:
    current: RegionCoveragePattern | None = None
    route: List[RegionCoveragePattern] = []
    edge_keys: List[Tuple[str, str]] = []
    objective = 0.0
    for region_id in order:
        candidates = []
        current_key = _node_key(current)
        for pattern in patterns.get(region_id, []):
            cost = float(cost_fn(current, pattern))
            if math.isfinite(cost):
                candidates.append((cost, pattern.pattern_id, pattern, (current_key, _node_key(pattern))))
        if not candidates:
            return None
        cost, _, selected, edge_key = min(candidates, key=lambda item: (item[0], item[1]))
        route.append(selected)
        edge_keys.append(edge_key)
        objective += cost
        current = selected
    return _AcoSolution(route=route, objective=objective, edge_keys=edge_keys)


def _three_opt_orders(order: Sequence[str]) -> List[List[str]]:
    if len(order) < 4:
        return []
    result: List[List[str]] = []
    n = len(order)
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                a = list(order[:i])
                b = list(order[i:j])
                c = list(order[j:k])
                d = list(order[k:])
                result.append(a + list(reversed(b)) + c + d)
                result.append(a + b + list(reversed(c)) + d)
                result.append(a + list(reversed(c)) + list(reversed(b)) + d)
    return result


def _weighted_choice(
    options: Sequence[Tuple[RegionCoveragePattern, float, float, Tuple[str, str]]],
    rng: random.Random,
) -> Tuple[RegionCoveragePattern, float, float, Tuple[str, str]]:
    total = sum(item[2] for item in options)
    threshold = rng.random() * total
    running = 0.0
    for option in options:
        running += option[2]
        if running >= threshold:
            return option
    return options[-1]


def _update_pheromone(
    pheromone: Dict[Tuple[str, str], float],
    iteration_solutions: Sequence[_AcoSolution],
    best: _AcoSolution,
    rho: float,
    q: float,
) -> None:
    for key in list(pheromone):
        pheromone[key] = max((1.0 - rho) * pheromone[key], 1e-9)
    elite_count = max(1, len(iteration_solutions) // 4)
    for solution in sorted(iteration_solutions, key=lambda item: item.objective)[:elite_count]:
        _deposit(pheromone, solution, q)
    _deposit(pheromone, best, q * 0.5)


def _deposit(pheromone: Dict[Tuple[str, str], float], solution: _AcoSolution, q: float) -> None:
    amount = q / max(solution.objective, 1e-9)
    for edge_key in solution.edge_keys:
        pheromone[edge_key] = pheromone.get(edge_key, 1.0) + amount


def _evaporation_rate(iteration: int, path_config: PathPlanningConfig, solver: str) -> float:
    if solver != "fa3aco":
        return min(max(path_config.aco_rho, 0.0), 0.95)
    rho = path_config.fa3aco_rho_min + (
        path_config.fa3aco_rho_max - path_config.fa3aco_rho_min
    ) * math.exp(-path_config.fa3aco_rho_decay * iteration)
    return min(max(rho, 0.0), 0.95)


def _fractional_memory(
    candidate_key: str,
    probability_history: Sequence[Dict[str, float]],
    path_config: PathPlanningConfig,
) -> float:
    depth = max(int(path_config.fa3aco_memory_depth), 0)
    if depth <= 0 or not probability_history:
        return 0.0
    nu = min(max(path_config.fa3aco_fractional_order, 1e-6), 0.999)
    memory = 0.0
    recent = list(probability_history[-depth:])
    for k, probability_map in enumerate(reversed(recent), start=1):
        weight = abs(math.gamma(k - nu) / (math.gamma(1.0 - nu) * math.gamma(k + 1.0)))
        memory += weight * probability_map.get(candidate_key, 0.0)
    return memory


def _node_key(pattern: RegionCoveragePattern | None) -> str:
    if pattern is None:
        return "__start__"
    return f"{pattern.region_id}:{pattern.pattern_id}"


def _failed_result(requested: str, reason: str, region_ids: Sequence[str], **metadata: object) -> AcoTspResult:
    return AcoTspResult(
        status="failed",
        requested_solver=requested,
        effective_solver="deterministic_fallback",
        metadata={"failure_reason": reason, "region_ids": list(region_ids), **metadata},
    )
