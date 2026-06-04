from __future__ import annotations

from abc import ABC, abstractmethod

from .types import MultiAgentPathPlan, PathPlanningRequest


class PathPlanningAlgorithm(ABC):
    name: str = "unnamed_path_planner"

    @abstractmethod
    def plan(self, request: PathPlanningRequest) -> MultiAgentPathPlan:
        raise NotImplementedError
