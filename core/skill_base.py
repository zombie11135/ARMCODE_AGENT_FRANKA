# core/skill_base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SkillResult:
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error: Optional[str] = None


class BaseSkill(ABC):
    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}

    def __init__(self, context):
        self.context = context

    @abstractmethod
    def run(self, **kwargs) -> SkillResult:
        pass