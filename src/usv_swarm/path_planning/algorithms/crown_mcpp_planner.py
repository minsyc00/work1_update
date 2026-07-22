from __future__ import annotations

from ..base import PathPlanningAlgorithm
from ..crown.pipeline import run_crown_mcpp_pipeline
from ..types import MultiAgentPathPlan, PathPlanningRequest


class CrownMcppPlanner(PathPlanningAlgorithm):
    name = "crown_mcpp"

    def plan(self, request: PathPlanningRequest) -> MultiAgentPathPlan:
        return run_crown_mcpp_pipeline(
            config=request.config,
            path_config=request.path_config,
            static_obstacles=request.static_obstacles,
            paper_references=request.paper_references,
        )
