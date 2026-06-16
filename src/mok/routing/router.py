from __future__ import annotations

from dataclasses import dataclass, field
import re

from mok.models.backends import RequestPayload
from mok.models.registry import ModelRegistry


CODE_PATTERN = re.compile(r"```|python|function|traceback|stack trace|bug|refactor", re.IGNORECASE)
VISION_PATTERN = re.compile(r"image|screenshot|photo|diagram|chart|figure", re.IGNORECASE)

# Ordered fallback role priority when the preferred role is unavailable
_FALLBACK_ROLES = ["general", "coordinator", "code", "vision"]


class RoutingError(RuntimeError):
    """Raised when no expert can be found in the registry."""


@dataclass(slots=True)
class RouteDecision:
    expert_name: str
    confidence: float
    reason: str
    router_tier: str = "R0"
    secondary_experts: list[str] = field(default_factory=list)


def _resolve(registry: ModelRegistry, *roles: str) -> str:
    """Return the name of the first expert found for any of *roles*.

    Raises RoutingError if the registry contains no matching expert at all.
    """
    for role in roles:
        expert = registry.find_first_by_role(role)
        if expert is not None:
            return expert.name
    # Last-resort: any expert at all
    all_experts = registry.all()
    if all_experts:
        return all_experts[0].name
    raise RoutingError(
        "Registry is empty — cannot route request. "
        "Add at least one expert to the config and restart."
    )


class RulesRouter:
    """R0 rules router using modality and keyword heuristics."""

    def route(self, payload: RequestPayload, registry: ModelRegistry) -> RouteDecision:
        if payload.modality_flags.get("has_image"):
            return RouteDecision(
                expert_name=_resolve(registry, "vision", "general", "coordinator"),
                confidence=0.95,
                reason="image modality flag",
                router_tier="R0",
            )

        if VISION_PATTERN.search(payload.prompt):
            return RouteDecision(
                expert_name=_resolve(registry, "vision", "general", "coordinator"),
                confidence=0.88,
                reason="vision keyword match",
                router_tier="R0",
            )

        if CODE_PATTERN.search(payload.prompt):
            return RouteDecision(
                expert_name=_resolve(registry, "code", "general", "coordinator"),
                confidence=0.84,
                reason="code keyword match",
                router_tier="R0",
            )

        return RouteDecision(
            expert_name=_resolve(registry, "general", "coordinator", "code"),
            confidence=0.65,
            reason="default general route",
            router_tier="R0",
        )
