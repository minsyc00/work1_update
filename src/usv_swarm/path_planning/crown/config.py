"""Configuration and validation for the complete CROWN-MCPP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional, Tuple

from ..types import PathPlanningConfig


@dataclass(frozen=True)
class CrownMcppConfig:
    engine: str = "auto"
    time_step: float = 1.0
    horizon: Optional[float] = None
    mode_limit_per_region_agent: int = 8
    exact_max_agents: int = 4
    exact_max_regions: int = 10
    max_tasks_per_route: Optional[int] = None
    max_timed_columns: int = 200_000
    position_error_map: float = 0.0
    position_error_discretization: float = 0.0
    position_error_tracking: float = 0.0
    resource_grid_size: float = 1.0
    primitive_max_duration: float = 2.0
    enable_continuous_conflict_validation: bool = True
    return_to_start: bool = True
    goal_poses: Optional[Tuple[Tuple[float, float, float], ...]] = None
    include_sequential_baseline: bool = True
    baseline_time_budget_sec: float = 30.0
    connector_max_expansions: int = 2000
    root_exact_pricing: bool = True
    pricing_label_limit: int = 1_000_000
    lns_iterations: int = 500
    lns_time_budget_sec: float = 60.0
    lns_destroy_fraction: float = 0.2
    lns_pool_reopt_interval: int = 20
    lns_random_seed: int = 42
    lns_max_route_pool_per_agent: int = 200
    report_anytime_trace: bool = True

    def __post_init__(self) -> None:
        if self.engine not in {"auto", "bpc", "lns", "certified_lns"}:
            raise ValueError("crown engine must be auto, bpc, lns, or certified_lns")
        positive = {
            "time_step": self.time_step,
            "resource_grid_size": self.resource_grid_size,
            "primitive_max_duration": self.primitive_max_duration,
            "lns_time_budget_sec": self.lns_time_budget_sec,
            "baseline_time_budget_sec": self.baseline_time_budget_sec,
        }
        invalid = [name for name, value in positive.items() if not isfinite(value) or value <= 0.0]
        if invalid:
            raise ValueError(f"CROWN positive values invalid: {','.join(invalid)}")
        integer_positive = {
            "mode_limit_per_region_agent": self.mode_limit_per_region_agent,
            "exact_max_agents": self.exact_max_agents,
            "exact_max_regions": self.exact_max_regions,
            "max_timed_columns": self.max_timed_columns,
            "pricing_label_limit": self.pricing_label_limit,
            "lns_iterations": self.lns_iterations,
            "lns_pool_reopt_interval": self.lns_pool_reopt_interval,
            "lns_max_route_pool_per_agent": self.lns_max_route_pool_per_agent,
            "connector_max_expansions": self.connector_max_expansions,
        }
        invalid_integer = [name for name, value in integer_positive.items() if value <= 0]
        if invalid_integer:
            raise ValueError(f"CROWN integer values invalid: {','.join(invalid_integer)}")
        errors = (
            self.position_error_map,
            self.position_error_discretization,
            self.position_error_tracking,
        )
        if any(not isfinite(value) or value < 0.0 for value in errors):
            raise ValueError("CROWN position errors must be finite and non-negative")
        if self.horizon is not None and (not isfinite(self.horizon) or self.horizon <= 0.0):
            raise ValueError("CROWN horizon must be positive when provided")
        if self.max_tasks_per_route is not None and self.max_tasks_per_route <= 0:
            raise ValueError("CROWN max_tasks_per_route must be positive when provided")
        if not 0.0 < self.lns_destroy_fraction <= 1.0:
            raise ValueError("CROWN LNS destroy fraction must be in (0, 1]")
        if self.goal_poses is not None:
            if not self.goal_poses:
                raise ValueError("CROWN goal_poses cannot be empty when provided")
            if any(
                len(pose) != 3 or any(not isfinite(float(value)) for value in pose)
                for pose in self.goal_poses
            ):
                raise ValueError("each CROWN goal pose must contain finite x, y, psi")

    @property
    def total_position_error(self) -> float:
        return (
            self.position_error_map
            + self.position_error_discretization
            + self.position_error_tracking
        )

    @classmethod
    def from_path_config(cls, path_config: PathPlanningConfig) -> "CrownMcppConfig":
        return cls(
            engine=path_config.crown_engine,
            time_step=path_config.crown_time_step,
            horizon=path_config.crown_horizon,
            mode_limit_per_region_agent=path_config.crown_mode_limit_per_region_agent,
            exact_max_agents=path_config.crown_exact_max_agents,
            exact_max_regions=path_config.crown_exact_max_regions,
            max_tasks_per_route=path_config.crown_max_tasks_per_route,
            max_timed_columns=path_config.crown_max_timed_columns,
            position_error_map=path_config.crown_position_error_map,
            position_error_discretization=path_config.crown_position_error_discretization,
            position_error_tracking=path_config.crown_position_error_tracking,
            resource_grid_size=(
                path_config.crown_resource_grid_size
                if path_config.crown_resource_grid_size is not None
                else path_config.shared_resource_grid_size
            ),
            primitive_max_duration=path_config.crown_primitive_max_duration,
            enable_continuous_conflict_validation=(
                path_config.crown_enable_continuous_conflict_validation
            ),
            return_to_start=path_config.crown_return_to_start,
            goal_poses=path_config.crown_goal_poses,
            include_sequential_baseline=path_config.crown_include_sequential_baseline,
            baseline_time_budget_sec=path_config.crown_baseline_time_budget_sec,
            connector_max_expansions=path_config.crown_connector_max_expansions,
            root_exact_pricing=path_config.crown_root_exact_pricing,
            pricing_label_limit=path_config.crown_pricing_label_limit,
            lns_iterations=path_config.crown_lns_iterations,
            lns_time_budget_sec=path_config.crown_lns_time_budget_sec,
            lns_destroy_fraction=path_config.crown_lns_destroy_fraction,
            lns_pool_reopt_interval=path_config.crown_lns_pool_reopt_interval,
            lns_random_seed=path_config.crown_lns_random_seed,
            lns_max_route_pool_per_agent=path_config.crown_lns_max_route_pool_per_agent,
            report_anytime_trace=path_config.crown_report_anytime_trace,
        )
