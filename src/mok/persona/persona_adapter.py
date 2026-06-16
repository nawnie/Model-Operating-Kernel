"""
src/mok/persona/persona_adapter.py

MoK Persona Adapter
====================
Reads a UserProfile and adjusts ConsultationEngine behavior at query time.

This is the mechanism that makes each MoK instance unique over time.
Two users who start from the same base model end up with different
effective behavior because their profiles push the adapter in different
directions.

Adjustments made
----------------
  expert_tier_bias      — push cheap vs strong expert selection based on preferred_experts
  challenge_threshold   — raise/lower challenge aggressiveness based on correction_rate
  max_turns             — shorten consultation loops if user prefers fast answers
  confidence_threshold  — tighten if user's gate fail rate is high
  preferred_expert_hint — pass the user's top expert as a hint to ConsultationEngine

The adapter does NOT change model weights. It changes runtime parameters.
The RSI flywheel changes weights via fine-tuning. Both move together.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mok.persona.user_profile import UserProfile

if TYPE_CHECKING:
    from mok.orchestration.consultation import ConsultationEngine, ResourceContext

logger = logging.getLogger(__name__)

# Thresholds for behavioral adjustments
HIGH_CORRECTION_RATE = 0.15     # above this: become more cautious
LOW_CORRECTION_RATE = 0.05      # below this: user trusts MoK more
HIGH_GATE_FAIL_RATE = 0.20      # fraction of sessions with gate failures
FAST_ANSWER_AVG_TURNS = 1.5     # below this: user wants fewer expert turns
DEEP_ANSWER_AVG_TURNS = 3.0     # above this: user likes thorough consultation


@dataclass
class PersonaAdjustments:
    """The set of behavioral adjustments the adapter applies.

    Passed to ConsultationEngine before each query.
    """
    # Expert selection bias: +1.0 = strongly prefer cheap, -1.0 = strongly prefer strong
    expert_tier_bias: float = 0.0

    # Challenge threshold multiplier: <1.0 = challenge more, >1.0 = challenge less
    challenge_threshold_multiplier: float = 1.0

    # Max consultation turns override (None = use engine default)
    max_consultation_turns: int | None = None

    # Confidence threshold adjustment (additive): positive = require higher confidence
    confidence_threshold_delta: float = 0.0

    # Hint to the engine: prefer this expert_id when ambiguous
    preferred_expert_hint: str | None = None

    # Hint to the engine: bias toward this task type's expert
    dominant_task_type: str = "general"

    def describe(self) -> str:
        parts = []
        if abs(self.expert_tier_bias) > 0.1:
            direction = "cheap" if self.expert_tier_bias > 0 else "strong"
            parts.append(f"expert_bias={direction}({self.expert_tier_bias:+.2f})")
        if abs(self.challenge_threshold_multiplier - 1.0) > 0.05:
            parts.append(f"challenge_multiplier={self.challenge_threshold_multiplier:.2f}")
        if self.max_consultation_turns is not None:
            parts.append(f"max_turns={self.max_consultation_turns}")
        if abs(self.confidence_threshold_delta) > 0.01:
            parts.append(f"confidence_delta={self.confidence_threshold_delta:+.2f}")
        if self.preferred_expert_hint:
            parts.append(f"preferred_expert={self.preferred_expert_hint}")
        return ", ".join(parts) if parts else "no adjustments"


class PersonaAdapter:
    """Translates a UserProfile into runtime adjustments for ConsultationEngine.

    Usage
    -----
    adapter = PersonaAdapter(profile)
    adjustments = adapter.compute_adjustments(prompt)
    engine.apply_persona(adjustments)   # ConsultationEngine picks this up per-call
    """

    def __init__(self, profile: UserProfile) -> None:
        self.profile = profile

    def compute_adjustments(self, prompt: str) -> PersonaAdjustments:
        """Compute behavioral adjustments for this query based on the user profile."""
        adj = PersonaAdjustments()
        p = self.profile

        if p.sessions < 5:
            # Not enough history — use defaults
            logger.debug("[PersonaAdapter] insufficient history (%d sessions), using defaults", p.sessions)
            return adj

        # ── Expert tier bias ──────────────────────────────────────────────────
        # If the user's top expert is cheap, bias toward cheap
        top = p.top_expert
        if top and "3b" in top.lower() or (top and "fast" in top.lower()):
            adj.expert_tier_bias = 0.3   # mild cheap preference
        elif top and ("7b" in top.lower() or "strong" in top.lower() or "general" in top.lower()):
            adj.expert_tier_bias = -0.3  # mild strong preference
        adj.preferred_expert_hint = top

        # ── Challenge threshold ────────────────────────────────────────────────
        # High correction rate → MoK is being too passive → challenge more aggressively
        if p.correction_rate > HIGH_CORRECTION_RATE:
            adj.challenge_threshold_multiplier = 0.7  # challenge 30% more aggressively
        elif p.correction_rate < LOW_CORRECTION_RATE and p.sessions >= 20:
            adj.challenge_threshold_multiplier = 1.2  # user trusts MoK; relax slightly

        # ── Consultation depth ─────────────────────────────────────────────────
        if p.prefers_fast_answers:
            adj.max_consultation_turns = 1   # one expert call, then synthesize
        elif p.avg_consultation_turns >= DEEP_ANSWER_AVG_TURNS:
            adj.max_consultation_turns = 4   # user wants thorough consultation

        # ── Confidence threshold ───────────────────────────────────────────────
        # High gate failure rate → tighten confidence requirements
        gate_fail_sessions = sum(p.gate_fail_patterns.values())
        gate_fail_rate = gate_fail_sessions / p.sessions if p.sessions else 0.0
        if gate_fail_rate > HIGH_GATE_FAIL_RATE:
            adj.confidence_threshold_delta = 0.1  # require slightly higher confidence

        # ── Task type hint ─────────────────────────────────────────────────────
        adj.dominant_task_type = p.dominant_task_type

        logger.debug(
            "[PersonaAdapter] user=%s sessions=%d adjustments: %s",
            p.user_id, p.sessions, adj.describe()
        )
        return adj

    def apply_to_engine(
        self,
        engine: "ConsultationEngine",
        adjustments: PersonaAdjustments,
    ) -> None:
        """Apply persona adjustments to a ConsultationEngine instance.

        Modifies engine thresholds in place for this query.
        Call before engine.handle() and restore after if needed.
        """
        # Adjust cheap/strong VRAM thresholds based on expert tier bias
        if adjustments.expert_tier_bias > 0.1:
            # Bias toward cheap: lower the cheap threshold slightly
            engine.CHEAP_VRAM_THRESHOLD_GB = max(1.5, engine.CHEAP_VRAM_THRESHOLD_GB - 0.5)
        elif adjustments.expert_tier_bias < -0.1:
            # Bias toward strong: raise the cheap threshold so we escalate sooner
            engine.CHEAP_VRAM_THRESHOLD_GB = min(4.0, engine.CHEAP_VRAM_THRESHOLD_GB + 0.5)

        # Apply max consultation turns to all sessions
        if adjustments.max_consultation_turns is not None:
            from mok.orchestration.consultation import ConsultationSession
            ConsultationSession.MAX_TURNS = adjustments.max_consultation_turns

        # Apply confidence threshold
        if adjustments.confidence_threshold_delta != 0.0:
            engine._persona_confidence_delta = adjustments.confidence_threshold_delta

        # Store hint for expert selection
        engine._preferred_expert_hint = adjustments.preferred_expert_hint
        engine._dominant_task_type = adjustments.dominant_task_type

    def summary(self) -> dict:
        """Return a human-readable summary of the profile and current adjustments."""
        p = self.profile
        adj = self.compute_adjustments("")
        return {
            "user_id": p.user_id,
            "sessions": p.sessions,
            "correction_rate": p.correction_rate,
            "dominant_task_type": p.dominant_task_type,
            "top_expert": p.top_expert,
            "avg_consultation_turns": p.avg_consultation_turns,
            "avg_quality_score": p.avg_quality_score,
            "top_gate_failure": p.top_gate_failure,
            "prefers_fast_answers": p.prefers_fast_answers,
            "current_adjustments": adj.describe(),
        }
