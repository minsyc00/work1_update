from __future__ import annotations

from typing import Callable, Dict, List, Type

from .base import PathPlanningAlgorithm


class PathPlannerRegistry:
    def __init__(self) -> None:
        self._constructors: Dict[str, Callable[[], PathPlanningAlgorithm]] = {}

    def register(self, name: str, constructor: Callable[[], PathPlanningAlgorithm]) -> None:
        self._constructors[name] = constructor

    def create(self, name: str) -> PathPlanningAlgorithm:
        if name not in self._constructors:
            raise KeyError(f"Unknown path planner: {name}")
        return self._constructors[name]()

    def names(self) -> List[str]:
        return sorted(self._constructors.keys())
