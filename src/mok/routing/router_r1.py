"""
src/mok/routing/router_r1.py

R1 Zero-Shot Router — P3.1.

Escalates low-confidence R0 routing decisions to the always-resident
coordinator expert, which classifies the request via a single zero-shot
inference call.

Design
------
1. R0 produces a RouteDecision.
2. If confidence >= escalate_below threshold, R0 decision is returned as-is.
3. Otherwise, ZeroShotRouter asks the coordinator:
       "Which expert role best matches this request? Options: {roles}. Prompt: {prompt}"
4. The coordinator replies with a role name (or partial match).
5. ZeroShotRouter resolves the role to an expert name and returns a new
   RouteDecision with router_tier="R1".
6. On any failure (coordinator unreachable, parse error, unknown role),
   falls back silently to the original R0 decision.

No external dependencies.  Coordinator backend call is synchronous.
"""
from __future__ import annotations

import logging
import re

from mok.models.backends import ExpertBackend, RequestPayload
from mok.models.registry import ModelRegistry
from mok.routing.router import RouteDecision, _resolve

logger = logging.getLogger(__name__)

# Default threshold: escalate when R0 confidence is below this value
_DEFAULT_ESCALATE_BELOW = 0.65

# Classification prompt template sent to the coordinator
_CLASSIFY_PROMPT = (
    "You are a routing classifier. "
    "Given a user request, pick the BEST expert role from the list below. "
    "Reply with ONLY the role name and nothing else.\n\n"
    "Available roles: {roles}\n\n"
    "User request: {prompt}\n\n"
    "Best role:"
)


class ZeroShotRouter:
    """
    R1 router: wraps RulesRouter and escalates low-confidence routes
    to the coordinator expert for zero-shot classification.

    Parameters
    ----------
    backends        : backend map (same dict used by OrchestratorRuntime)
    escalate_below  : R0 confidence below which escalation is triggered.
                      Defaults to 0.65 (R1_escalate_below from routing.json).
    coordinator_role: Registry role name of the always-resident classifier.
    """

    def __init__(
        self,
        backends: dict[str, ExpertBackend],
        escalate_below: float = _DEFAULT_ESCALATE_BELOW,
        coordinator_role: str = "coordinator",
    ) -> None:
        self._backends = backends
        self._escalate_below = escalate_below
        self._coordinator_role = coordinator_role

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def route(
        self,
        payload: RequestPayload,
        registry: ModelRegistry,
        r0_decision: RouteDecision,
    ) -> RouteDecision:
        """
        Optionally escalate r0_decision to a coordinator classification.

        Returns a RouteDecision with router_tier="R0" (pass-through) or
        router_tier="R1" (escalated and reclassified).

        Never raises — falls back to r0_decision on any error.
        """
        if r0_decision.confidence >= self._escalate_below:
            # R0 is confident enough; no escalation needed
            return r0_decision

        logger.debug(
            "[R1] Escalating req=%s (R0 confidence=%.2f < %.2f)",
            payload.request_id,
            r0_decision.confidence,
            self._escalate_below,
        )

        coordinator = registry.find_first_by_role(self._coordinator_role)
        if coordinator is None:
            logger.debug("[R1] No coordinator expert found; falling back to R0.")
            return r0_decision

        backend = self._backends.get(coordinator.backend)
        if backend is None:
            logger.debug("[R1] No backend for coordinator '%s'; falling back to R0.", coordinator.backend)
            return r0_decision

        # Collect available role names from the registry
        available_roles = sorted({e.role for e in registry.all()})
        classify_prompt = _CLASSIFY_PROMPT.format(
            roles=", ".join(available_roles),
            prompt=payload.prompt[:500],   # truncate for safety
        )

        try:
            classify_payload = RequestPayload(
                prompt=classify_prompt,
                request_id=payload.request_id + "_r1",
                parameters={"temperature": 0.0, "max_tokens": 16},
            )
            response = backend.generate(coordinator, classify_payload)
            chosen_role = _parse_role(response.text, available_roles)
        except Exception as exc:
            logger.debug("[R1] Coordinator call failed (%s); falling back to R0.", exc)
            return r0_decision

        if chosen_role is None:
            logger.debug(
                "[R1] Could not parse role from coordinator response %r; falling back to R0.",
                response.text[:80],
            )
            return r0_decision

        # Resolve role → expert name
        try:
            expert_name = _resolve(registry, chosen_role)
        except Exception:
            return r0_decision

        logger.debug(
            "[R1] Reclassified req=%s from '%s' (R0) to '%s' (R1).",
            payload.request_id,
            r0_decision.expert_name,
            expert_name,
        )

        return RouteDecision(
            expert_name=expert_name,
            confidence=0.80,   # fixed confidence for a successful R1 override
            reason=f"R1 zero-shot: coordinator chose '{chosen_role}'",
            router_tier="R1",
            secondary_experts=r0_decision.secondary_experts,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_role(text: str, available_roles: list[str]) -> str | None:
    """
    Extract the first matching role name from the coordinator's response.

    Looks for an exact match (case-insensitive) in the first 80 chars.
    Returns None if no known role is found.
    """
    snippet = text.strip()[:80].lower()
    for role in available_roles:
        if re.search(rf"\b{re.escape(role)}\b", snippet):
            return role
    return None
