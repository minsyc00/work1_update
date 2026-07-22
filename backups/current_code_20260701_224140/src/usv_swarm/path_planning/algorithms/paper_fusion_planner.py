from __future__ import annotations

from ..base import PathPlanningAlgorithm
from ..pipeline import run_paper_fusion_pipeline
from ..types import MultiAgentPathPlan, PathPlanningRequest


class PaperFusionPlanner(PathPlanningAlgorithm):
    name = "paper_fusion_planner"

    def plan(self, request: PathPlanningRequest) -> MultiAgentPathPlan:
        return run_paper_fusion_pipeline(
            config=request.config,
            path_config=request.path_config,
            static_obstacles=request.static_obstacles,
            paper_references=request.paper_references,
        )
