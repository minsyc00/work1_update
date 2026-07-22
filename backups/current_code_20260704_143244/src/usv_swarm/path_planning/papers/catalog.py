from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from ..types import PaperReference


@dataclass
class PaperFusionRule:
    stage_name: str
    purpose: str
    source_paper_ids: Tuple[str, ...] = ()
    notes: str = ""


@dataclass
class PaperFusionProfile:
    profile_name: str
    references: List[PaperReference] = field(default_factory=list)
    fusion_rules: List[PaperFusionRule] = field(default_factory=list)
