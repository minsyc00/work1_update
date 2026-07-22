from __future__ import annotations

from typing import List, Optional, Sequence

from ..planning import plan_global_coverage
from ..schema import DynamicObstacleTrack, PlannerConfig, PlanningResult
from .adapters.runtime_adapter import build_request_from_planning_result
from .algorithms.crown_mcpp_planner import CrownMcppPlanner
from .algorithms.paper_fusion_planner import PaperFusionPlanner
from .registry import PathPlannerRegistry
from .types import MultiAgentPathPlan, PaperReference, PathPlanningConfig, PathPlanningRequest, StaticObstacle


class PathPlanningLayer:
    def __init__(self, default_algorithm: str = "paper_fusion_planner") -> None:
        self.registry = PathPlannerRegistry()
        self.registry.register("paper_fusion_planner", PaperFusionPlanner)
        self.registry.register("crown_mcpp", CrownMcppPlanner)
        self.default_algorithm = default_algorithm

    def available_algorithms(self) -> List[str]:
        return self.registry.names()

    def build_request(
        self,
        config: PlannerConfig,
        planning_result: Optional[PlanningResult] = None,
        static_obstacles: Optional[Sequence[StaticObstacle]] = None,
        dynamic_obstacles: Optional[Sequence[DynamicObstacleTrack]] = None,
        paper_references: Optional[Sequence[PaperReference]] = None,
        path_config: Optional[PathPlanningConfig] = None,
    ) -> PathPlanningRequest:
        existing_plan = planning_result or plan_global_coverage(config)
        request = build_request_from_planning_result(
            config=config,
            planning_result=existing_plan,
            static_obstacles=static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
            paper_references=paper_references,
        )
        request.path_config = path_config
        return request

    def plan_paths(
        self,
        request: PathPlanningRequest,
        algorithm_name: Optional[str] = None,
    ) -> MultiAgentPathPlan:
        planner = self.registry.create(algorithm_name or self.default_algorithm)
        return planner.plan(request)

    def plan_from_config(
        self,
        config: PlannerConfig,
        planning_result: Optional[PlanningResult] = None,
        static_obstacles: Optional[Sequence[StaticObstacle]] = None,
        dynamic_obstacles: Optional[Sequence[DynamicObstacleTrack]] = None,
        paper_references: Optional[Sequence[PaperReference]] = None,
        path_config: Optional[PathPlanningConfig] = None,
        algorithm_name: Optional[str] = None,
    ) -> MultiAgentPathPlan:
        request = self.build_request(
            config=config,
            planning_result=planning_result,
            static_obstacles=static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
            paper_references=paper_references,
            path_config=path_config,
        )
        return self.plan_paths(request, algorithm_name=algorithm_name)
