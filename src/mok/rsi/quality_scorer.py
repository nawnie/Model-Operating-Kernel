"""
src/mok/rsi/quality_scorer.py

RSI Quality Scorer
==================
Scores a ConsultationResult on four dimensions.
Records scoring >= threshold are eligible for the replay buffer.
Records that fail are discarded — the flywheel only trains on what worked.

Dimensions
----------
1. gate_discipline      — gate fired correctly (not "pending" or "no_backend")
2. no_copy_enforcement  — MOK_FINAL differs sufficiently from raw expert text
3. challenge_discipline — MoK pushed back on weak output when it should have
4. confidence_calibration — stated confidence matches gate outcome

Each dimension scores 0.0 or 1.0. Total score is 0.0–1.0.
Default eligibility threshold: 0.6 (3 of 4 dimensions is close enough for v1).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mok.orchestration.consultation import ConsultationResult


# Bad gate values — these mean the consultation loop did not really complete
_FAILED_GATES = frozenset({
    "pending",
    "no_backend",
    "no_expert_available",
    "expert_output_insufficient",
})

_HIGH_CONFIDENCE_GATES = frozenset({
    "expert_answer_accepted_and_synthesized",
    "disagreement_analyzed_and_position_chosen_with_reason",
    "followup_completed_and_new_info_extracted",
})


@dataclass
class QualityScoreBreakdown:
    gate_discipline: float = 0.0
    no_copy_enforcement: float = 0.0
    challenge_discipline: float = 0.0
    confidence_calibration: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.gate_discipline
            + self.no_copy_enforcement
            + self.challenge_discipline
            + self.confidence_calibration
        ) / 4.0

    def is_eligible(self, threshold: float = 0.6) -> bool:
        return self.total >= threshold

    def to_dict(self) -> dict:
        return {
            "gate_discipline": self.gate_discipline,
            "no_copy_enforcement": self.no_copy_enforcement,
            "challenge_discipline": self.challenge_discipline,
            "confidence_calibration": self.confidence_calibration,
            "total": round(self.total, 4),
            "eligible": self.is_eligible(),
        }


def _record_id(user: str, mok_action: str, gate: str) -> str:
    """Canonical record dedup key.
    Defined here for backwards-compat import; source of truth is replay_buffer._record_id.
    """
    key = f"{user[:80]}|{mok_action}|{gate}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _token_overlap_ratio(a: str, b: str) -> float:
    """Rough token overlap between two strings. 1.0 = identical, 0.0 = no overlap."""
    if not a or not b:
        return 0.0
    tok_a = set(re.findall(r"\w+", a.lower()))
    tok_b = set(re.findall(r"\w+", b.lower()))
    if not tok_a or not tok_b:
        return 0.0
    return len(tok_a & tok_b) / max(len(tok_a), len(tok_b))


def _raw_expert_text(result: "ConsultationResult") -> str:
    """Concatenate all raw expert findings from the result."""
    parts: list[str] = []
    for reply in result.accepted_findings:
        parts.append(reply)
    for session in result.sessions:
        for turn in session.turns:
            parts.extend(turn.reply.findings)
    return " ".join(parts)


def score(result: "ConsultationResult") -> QualityScoreBreakdown:
    """Score a ConsultationResult on all four RSI quality dimensions."""
    breakdown = QualityScoreBreakdown()

    # 1. Gate discipline
    if result.gate not in _FAILED_GATES:
        breakdown.gate_discipline = 1.0

    # 2. No-copy enforcement
    raw_expert = _raw_expert_text(result)
    overlap = _token_overlap_ratio(result.final_answer, raw_expert)
    if overlap < 0.70:
        breakdown.no_copy_enforcement = 1.0
    elif overlap < 0.85:
        breakdown.no_copy_enforcement = 0.5

    # 3. Challenge discipline
    vague_turns = 0
    challenged_turns = 0
    for session in result.sessions:
        for turn in session.turns:
            if turn.reply.quality in ("vague", "overconfident"):
                vague_turns += 1
                if turn.challenged:
                    challenged_turns += 1

    if vague_turns == 0:
        breakdown.challenge_discipline = 1.0
    else:
        breakdown.challenge_discipline = challenged_turns / vague_turns

    # 4. Confidence calibration
    confidence = result.confidence
    gate = result.gate
    gate_passed = gate not in _FAILED_GATES

    if gate_passed and confidence in ("medium", "high"):
        breakdown.confidence_calibration = 1.0
    elif not gate_passed and confidence == "low":
        breakdown.confidence_calibration = 1.0
    elif gate_passed and confidence == "low":
        breakdown.confidence_calibration = 0.5
    else:
        breakdown.confidence_calibration = 0.0

    return breakdown
