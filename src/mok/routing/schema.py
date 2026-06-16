"""
src/mok/routing/schema.py

Typed route contract — mok.route.v1

This is the single source of truth for what the router produces
and what the runtime consumes.  Lock this schema before Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RouteRequest:
    """Everything the router needs to make a decision."""

    request_id: str
    prompt: str
    modality_flags: dict[str, bool] = field(default_factory=dict)
    context_tokens: int = 0          # estimated input token count
    parameters: dict[str, Any] = field(default_factory=dict)  # pass-through to backend

    @classmethod
    def from_payload(cls, payload) -> "RouteRequest":
        """Construct from a RequestPayload (loose duck-type)."""
        return cls(
            request_id=getattr(payload, "request_id", "req-unknown"),
            prompt=getattr(payload, "prompt", ""),
            modality_flags=dict(getattr(payload, "modality_flags", {})),
            parameters=dict(getattr(payload, "parameters", {})),
        )


@dataclass(frozen=True)
class RouteRecord:
    """The router's decision, frozen for tracing and evaluation.

    All fields that downstream consumers may branch on are here.
    Do not add mutable state.
    """

    request_id: str
    chosen_expert: str
    confidence: float
    reason: str
    router_tier: str                  # "R0" | "R1" | "R2"
    fallback_chain: list[str] = field(default_factory=list)  # ordered alternatives
    latency_ms: int = 0

    @classmethod
    def from_decision(cls, request_id: str, decision, latency_ms: int = 0) -> "RouteRecord":
        """Construct from a RouteDecision produced by any router tier."""
        return cls(
            request_id=request_id,
            chosen_expert=decision.expert_name,
            confidence=decision.confidence,
            reason=decision.reason,
            router_tier=getattr(decision, "router_tier", "R0"),
            fallback_chain=list(getattr(decision, "secondary_experts", [])),
            latency_ms=latency_ms,
        )

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "chosen_expert": self.chosen_expert,
            "confidence": self.confidence,
            "reason": self.reason,
            "router_tier": self.router_tier,
            "fallback_chain": self.fallback_chain,
            "latency_ms": self.latency_ms,
        }
