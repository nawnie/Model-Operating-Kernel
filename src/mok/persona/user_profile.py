"""
src/mok/persona/user_profile.py

MoK User Profile
================
Tracks per-user behavioral patterns accumulated across sessions.
This is what makes each MoK instance diverge toward its owner.

The profile is updated after every ConsultationResult:
  - Which experts were used (and how often they were accepted vs challenged)
  - What kinds of tasks the user gives (inferred from prompt patterns)
  - How often MoK gets corrected (correction_rate)
  - Which gate failure modes appear repeatedly (user's "weak spots")
  - Average consultation depth (how many expert turns per query)

After enough sessions, the PersonaAdapter reads this profile and adjusts
ConsultationEngine behavior at query time.

Persistence
-----------
Profiles saved as JSON to `profile_dir/{user_id}.json`.
Default profile_dir: ~/.mok/profiles/
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mok.orchestration.consultation import ConsultationResult

logger = logging.getLogger(__name__)

DEFAULT_PROFILE_DIR = Path.home() / ".mok" / "profiles"

# Task type inference patterns
_TASK_PATTERNS: dict[str, re.Pattern] = {
    "code": re.compile(r"(```|def |class |bug|error:|refactor|test|import |function)", re.I),
    "planning": re.compile(r"\b(plan|design|architect|roadmap|strategy|outline|steps)\b", re.I),
    "research": re.compile(r"\b(research|find|search|look up|what is|explain|summarize)\b", re.I),
    "review": re.compile(r"\b(review|critique|check|audit|assess|evaluate|analyse|analyze)\b", re.I),
    "creative": re.compile(r"\b(write|generate|create|draft|compose|story|idea)\b", re.I),
}


def _infer_task_type(prompt: str) -> str:
    for task_type, pattern in _TASK_PATTERNS.items():
        if pattern.search(prompt):
            return task_type
    return "general"


@dataclass
class UserProfile:
    """Per-user behavioral profile accumulated from ConsultationResult traces.

    Fields
    ------
    user_id               : unique identifier (anonymous by default)
    sessions              : total number of handled requests
    preferred_experts     : {expert_id: accept_count} — which helpers the user's tasks favor
    expert_challenge_rate : {expert_id: float} — how often MoK challenged that expert for this user
    task_type_counts      : {task_type: count} — distribution of task types
    correction_count      : times MoK was corrected (recorded externally via mark_corrected())
    gate_fail_patterns    : {gate_condition: count} — repeated gate failures = user's weak spots
    avg_consultation_turns: rolling average of expert turns per query
    avg_quality_score     : rolling average of quality scores (0.0–1.0)
    first_seen            : ISO timestamp of first interaction
    last_updated          : ISO timestamp of last profile update
    """

    user_id: str = "anonymous"
    sessions: int = 0
    preferred_experts: dict[str, int] = field(default_factory=dict)
    expert_challenge_rate: dict[str, float] = field(default_factory=dict)
    task_type_counts: dict[str, int] = field(default_factory=dict)
    correction_count: int = 0
    gate_fail_patterns: dict[str, int] = field(default_factory=dict)
    avg_consultation_turns: float = 0.0
    avg_quality_score: float = 0.0
    first_seen: str = ""
    last_updated: str = ""

    # ------------------------------------------------------------------
    # Update API
    # ------------------------------------------------------------------

    def update_from_result(
        self,
        result: "ConsultationResult",
        prompt: str,
        quality_score: float,
    ) -> None:
        """Update profile from one ConsultationResult."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.first_seen:
            self.first_seen = now
        self.last_updated = now
        self.sessions += 1

        # Task type
        task_type = _infer_task_type(prompt)
        self.task_type_counts[task_type] = self.task_type_counts.get(task_type, 0) + 1

        # Expert usage + challenge rate
        for session in result.sessions:
            expert_id = session.expert.name
            turns = session.turns
            if not turns:
                continue

            # Count accepted turns (not challenged, quality good)
            accepted = sum(1 for t in turns if not t.challenged and t.reply.quality == "good")
            self.preferred_experts[expert_id] = self.preferred_experts.get(expert_id, 0) + accepted

            # Challenge rate for this expert
            vague_turns = sum(1 for t in turns if t.reply.quality in ("vague", "overconfident"))
            challenged = sum(1 for t in turns if t.challenged)
            if vague_turns > 0:
                rate = challenged / vague_turns
                # Rolling average
                prev = self.expert_challenge_rate.get(expert_id, rate)
                self.expert_challenge_rate[expert_id] = round((prev + rate) / 2, 3)

        # Consultation depth (rolling average)
        total_turns = sum(len(s.turns) for s in result.sessions)
        prev_avg = self.avg_consultation_turns
        self.avg_consultation_turns = round(
            ((prev_avg * (self.sessions - 1)) + total_turns) / self.sessions, 2
        )

        # Quality score (rolling average)
        prev_q = self.avg_quality_score
        self.avg_quality_score = round(
            ((prev_q * (self.sessions - 1)) + quality_score) / self.sessions, 4
        )

        # Gate failure patterns
        if result.gate in ("pending", "no_backend", "no_expert_available", "expert_output_insufficient"):
            self.gate_fail_patterns[result.gate] = self.gate_fail_patterns.get(result.gate, 0) + 1

    def mark_corrected(self) -> None:
        """Call when a user explicitly corrects MoK's answer."""
        self.correction_count += 1
        self.last_updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ------------------------------------------------------------------
    # Computed properties for PersonaAdapter
    # ------------------------------------------------------------------

    @property
    def correction_rate(self) -> float:
        if self.sessions == 0:
            return 0.0
        return round(self.correction_count / self.sessions, 3)

    @property
    def dominant_task_type(self) -> str:
        if not self.task_type_counts:
            return "general"
        return max(self.task_type_counts, key=self.task_type_counts.get)

    @property
    def top_expert(self) -> str | None:
        if not self.preferred_experts:
            return None
        return max(self.preferred_experts, key=self.preferred_experts.get)

    @property
    def top_gate_failure(self) -> str | None:
        if not self.gate_fail_patterns:
            return None
        return max(self.gate_fail_patterns, key=self.gate_fail_patterns.get)

    @property
    def prefers_fast_answers(self) -> bool:
        """True if user's average consultation depth is low (prefers fewer turns)."""
        return self.avg_consultation_turns <= 1.5

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, profile_dir: Path = DEFAULT_PROFILE_DIR) -> Path:
        profile_dir.mkdir(parents=True, exist_ok=True)
        path = profile_dir / f"{self.user_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, user_id: str, profile_dir: Path = DEFAULT_PROFILE_DIR) -> "UserProfile":
        path = profile_dir / f"{user_id}.json"
        if not path.exists():
            return cls(user_id=user_id)
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            profile = cls()
            for k, v in d.items():
                if hasattr(profile, k):
                    setattr(profile, k, v)
            profile.user_id = user_id
            return profile
        except Exception as e:
            logger.warning("[UserProfile] failed to load %s: %s — starting fresh", path, e)
            return cls(user_id=user_id)

    def to_dict(self) -> dict:
        return asdict(self)
