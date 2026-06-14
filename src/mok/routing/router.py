from __future__ import annotations

from dataclasses import dataclass, field
import re

from mok.models.backends import RequestPayload
from mok.models.registry import ModelRegistry


CODE_PATTERN = re.compile(r"```|python|function|traceback|stack trace|bug|refactor", re.IGNORECASE)
VISION_PATTERN = re.compile(r"image|screenshot|photo|diagram|chart|figure", re.IGNORECASE)


@dataclass(slots=True)
class RouteDecision:
    expert_name: str
    confidence: float
    reason: str
    secondary_experts: list[str] = field(default_factory=list)


class RulesRouter:
    """R0 rules router using modality and keyword heuristics."""

    def route(self, payload: RequestPayload, registry: ModelRegistry) -> RouteDecision:
        if payload.modality_flags.get("has_image"):
            expert = registry.find_first_by_role("vision") or registry.find_first_by_role("general")
            return RouteDecision(
                expert_name=expert.name,
                confidence=0.95,
                reason="image modality flag",
            )

        if VISION_PATTERN.search(payload.prompt):
            expert = registry.find_first_by_role("vision") or registry.find_first_by_role("general")
            return RouteDecision(
                expert_name=expert.name,
                confidence=0.88,
                reason="vision keyword match",
            )

        if CODE_PATTERN.search(payload.prompt):
            expert = registry.find_first_by_role("code") or registry.find_first_by_role("general")
            return RouteDecision(
                expert_name=expert.name,
                confidence=0.84,
                reason="code keyword match",
            )

        expert = registry.find_first_by_role("general") or registry.find_first_by_role("coordinator")
        return RouteDecision(
            expert_name=expert.name,
            confidence=0.65,
            reason="default general route",
        )
