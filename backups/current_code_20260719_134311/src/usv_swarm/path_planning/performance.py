from __future__ import annotations

from typing import Dict

from ..schema import CoverageState
from .types import AgentPathPlan, PathPlanningConfig


def build_performance_summary(
    agents: Dict[int, AgentPathPlan],
    coverage_state: CoverageState,
    totals: Dict[str, float],
    repeat_overlap_length: float,
    path_config: PathPlanningConfig,
) -> Dict[str, float | bool | str | Dict[str, float]]:
    total_length = float(totals.get("total_length", 0.0))
    coverage_length = float(totals.get("coverage_length", 0.0))
    transition_length = float(totals.get("transition_length", 0.0))
    turn_count = float(totals.get("turn_count", 0.0))
    residual_cells = sum(len(item.cells) for item in coverage_state.residual_components)
    residual_area_ratio = residual_cells / max(int(coverage_state.covered.size), 1)
    agent_lengths = {str(agent_id): float(agent.metrics.get("total_length", 0.0)) for agent_id, agent in agents.items()}
    agent_times = {str(agent_id): float(agent.metrics.get("estimated_time", 0.0)) for agent_id, agent in agents.items()}
    length_values = list(agent_lengths.values())
    mean_load = sum(length_values) / max(len(length_values), 1)
    load_imbalance = (max(length_values, default=0.0) - min(length_values, default=0.0)) / max(mean_load, 1e-9)
    time_values = list(agent_times.values())
    mean_time = sum(time_values) / max(len(time_values), 1)
    mission_makespan = max(time_values, default=float(totals.get("estimated_time", 0.0)))
    total_agent_work_time = sum(time_values) if time_values else float(totals.get("estimated_time", 0.0))
    time_imbalance = (max(time_values, default=0.0) - min(time_values, default=0.0)) / max(mean_time, 1e-9)
    constraint_ok = all(
        float(totals.get(key, 0.0)) <= 1e-9
        for key in (
            "invalid_path_length",
            "out_of_bounds_segment_count",
            "obstacle_collision_segment_count",
            "kinematic_infeasible_segment_count",
            "dynamic_infeasible_segment_count",
            "nmpc_untrackable_count",
        )
    )
    objective = (
        total_length
        + path_config.transition_length_weight * transition_length
        + path_config.repeat_transition_weight * repeat_overlap_length
        + path_config.residual_penalty_weight * residual_area_ratio
        + path_config.time_weight * mission_makespan
        + path_config.load_balance_weight * mission_makespan * time_imbalance
        + path_config.turn_count_weight * turn_count
    )
    return {
        "performance_profile": path_config.performance_profile,
        "target_coverage_fraction": float(path_config.target_coverage_fraction),
        "count_transit_coverage": bool(path_config.count_transit_coverage),
        "coverage_ratio": float(coverage_state.coverage_fraction),
        "target_coverage_met": coverage_state.coverage_fraction + 1e-9 >= path_config.target_coverage_fraction,
        "total_length": total_length,
        "coverage_length": coverage_length,
        "transition_length": transition_length,
        "coverage_length_ratio": _safe_ratio(coverage_length, total_length),
        "transition_length_ratio": _safe_ratio(transition_length, total_length),
        "repeat_overlap_length": float(repeat_overlap_length),
        "repeat_overlap_ratio": _safe_ratio(repeat_overlap_length, total_length),
        "repeat_transition_ratio": _safe_ratio(repeat_overlap_length, transition_length),
        "residual_count": float(len(coverage_state.residual_components)),
        "residual_area_ratio": residual_area_ratio,
        "turn_count": turn_count,
        "turn_density": _safe_ratio(turn_count, total_length),
        "agent_load_imbalance": load_imbalance,
        "agent_total_lengths": agent_lengths,
        "agent_work_times": agent_times,
        "mission_makespan": mission_makespan,
        "total_agent_work_time": total_agent_work_time,
        "agent_time_imbalance": time_imbalance,
        "constraint_ok": constraint_ok,
        "performance_objective": objective,
    }


def relative_improvement_summary(
    baseline: Dict[str, float | bool | str | Dict[str, float]],
    optimized: Dict[str, float | bool | str | Dict[str, float]],
) -> Dict[str, float]:
    return {
        "total_length_delta_ratio": _delta_ratio(_float_metric(optimized, "total_length"), _float_metric(baseline, "total_length")),
        "transition_length_delta_ratio": _delta_ratio(_float_metric(optimized, "transition_length"), _float_metric(baseline, "transition_length")),
        "repeat_overlap_delta_ratio": _delta_ratio(_float_metric(optimized, "repeat_overlap_length"), _float_metric(baseline, "repeat_overlap_length")),
        "coverage_delta": _float_metric(optimized, "coverage_ratio") - _float_metric(baseline, "coverage_ratio"),
        "turn_count_delta_ratio": _delta_ratio(_float_metric(optimized, "turn_count"), _float_metric(baseline, "turn_count")),
        "mission_makespan_delta_ratio": _delta_ratio(_float_metric(optimized, "mission_makespan"), _float_metric(baseline, "mission_makespan")),
        "agent_time_imbalance_delta_ratio": _delta_ratio(_float_metric(optimized, "agent_time_imbalance"), _float_metric(baseline, "agent_time_imbalance")),
        "objective_delta_ratio": _delta_ratio(_float_metric(optimized, "performance_objective"), _float_metric(baseline, "performance_objective")),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1e-9)


def _delta_ratio(current: float, baseline: float) -> float:
    return (current - baseline) / max(abs(baseline), 1e-9)


def _float_metric(summary: Dict[str, float | bool | str | Dict[str, float]], key: str) -> float:
    value = summary.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0
