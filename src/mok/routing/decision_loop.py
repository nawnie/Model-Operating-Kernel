"""
src/mok/routing/decision_loop.py

MoK Decision Loop
=================
Stateless evaluator that maps (task state, resource context, available experts)
→ MoKDecision.

This is the seven-branch logic kernel.  It does NOT call any model.
It produces a decision that ConsultationEngine acts on.

The seven branches
------------------
  ANSWER_DIRECT         — task is simple/short; no helper needed
  RETRIEVE_MEMORY       — need documents or memory before reasoning
  CALL_CHEAP_HELPER     — delegate to resident small model (≤3 GB)
  CALL_STRONG_HELPER    — escalate to on-demand larger model (>3 GB)
  MULTI_EXPERT          — consult two helpers and compare answers
  CONFIDENCE_SUFFICIENT — already have enough; stop gathering
  REPORT_UNCERTAINTY    — cannot answer reliably; state limits clearly

Designed for unit testing without any model dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any


class MoKDecision(str, Enum):
    ANSWER_DIRECT = "answer_direct"
    RETRIEVE_MEMORY = "retrieve_memory"
    CALL_CHEAP_HELPER = "call_cheap_helper"
    CALL_STRONG_HELPER = "call_strong_helper"
    MULTI_EXPERT = "multi_expert"
    CONFIDENCE_SUFFICIENT = "confidence_sufficient"
    REPORT_UNCERTAINTY = "report_uncertainty"


@dataclass
class TaskState:
    """Everything the decision loop needs to know about the current task.

    Populated by the caller before invoking DecisionLoop.evaluate().
    """
    prompt: str
    current_confidence: float = 0.0     # 0.0–1.0; from prior loop iteration or router
    findings_so_far: list[str] = field(default_factory=list)
    experts_called: list[str] = field(default_factory=list)
    retrieval_done: bool = False
    iteration: int = 0                  # how many times the loop has already run
    max_iterations: int = 3

    @property
    def is_first_iteration(self) -> bool:
        return self.iteration == 0

    @property
    def has_findings(self) -> bool:
        return bool(self.findings_so_far)

    @property
    def at_limit(self) -> bool:
        return self.iteration >= self.max_iterations


@dataclass
class AvailableExperts:
    """Snapshot of which experts can currently be used.

    vram_free_gb is checked before each entry here, so entries in each
    list are already budget-viable at this moment.
    """
    cheap: list[str] = field(default_factory=list)     # ≤3 GB VRAM, already loadable
    strong: list[str] = field(default_factory=list)    # >3 GB VRAM, loadable if VRAM free
    retrieval: list[str] = field(default_factory=list) # embedding / retrieval models
    vram_free_gb: float = 8.0
    time_budget_s: float = 30.0

    @property
    def has_cheap(self) -> bool:
        return bool(self.cheap)

    @property
    def has_strong(self) -> bool:
        return bool(self.strong)

    @property
    def has_retrieval(self) -> bool:
        return bool(self.retrieval)

    @property
    def can_multi_expert(self) -> bool:
        """True when both a cheap and a strong expert are available."""
        return self.has_cheap and self.has_strong

    @property
    def is_time_tight(self) -> bool:
        return self.time_budget_s <= 10.0


@dataclass(frozen=True)
class DecisionRecord:
    """Frozen output from DecisionLoop.evaluate().

    Carries the decision plus the reasoning for tracing/training.
    """
    decision: MoKDecision
    reason: str
    preferred_expert_tier: str = ""   # "cheap" | "strong" | "both" | "retrieval" | "none"
    confidence_after: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "preferred_expert_tier": self.preferred_expert_tier,
            "confidence_after": self.confidence_after,
        }


# ---------------------------------------------------------------------------
# Heuristic classifiers (stateless helpers)
# ---------------------------------------------------------------------------

_SIMPLE_RE = re.compile(
    r"^(what is|what are|define|how many|who is|list the|name the|when did)\b",
    re.IGNORECASE,
)
_RETRIEVAL_RE = re.compile(
    r"\b(document|file|memory|recall|last session|plan\.md|previous|earlier|stored)\b",
    re.IGNORECASE,
)
_CODE_RE = re.compile(
    r"(```|def |class |traceback|bug|error:|refactor|test case|import )",
    re.IGNORECASE,
)
_COMPLEX_RE = re.compile(
    r"\b(design|architect|multi.step|compare|evaluate|trade.off|explain why|analyse|analyze)\b",
    re.IGNORECASE,
)


def _prompt_is_simple(prompt: str) -> bool:
    return bool(_SIMPLE_RE.match(prompt.strip())) and len(prompt) < 100


def _prompt_needs_retrieval(prompt: str) -> bool:
    return bool(_RETRIEVAL_RE.search(prompt))


def _prompt_needs_code_expert(prompt: str) -> bool:
    return bool(_CODE_RE.search(prompt))


def _prompt_is_complex(prompt: str) -> bool:
    return bool(_COMPLEX_RE.search(prompt))


# ---------------------------------------------------------------------------
# DecisionLoop — the main evaluator
# ---------------------------------------------------------------------------

class DecisionLoop:
    """Stateless evaluator: (TaskState, AvailableExperts) → DecisionRecord.

    Rule priority (highest to lowest)
    -----------------------------------
    1. Already confident enough       → CONFIDENCE_SUFFICIENT
    2. Loop at iteration limit        → REPORT_UNCERTAINTY (if still uncertain)
    3. Task is trivially simple       → ANSWER_DIRECT
    4. No resources at all            → REPORT_UNCERTAINTY
    5. Need retrieval and not done    → RETRIEVE_MEMORY
    6. Complex + both experts avail   → MULTI_EXPERT
    7. Code task + code expert avail  → CALL_STRONG_HELPER (prefer code specialist)
    8. Complex + strong avail         → CALL_STRONG_HELPER
    9. Any cheap expert avail         → CALL_CHEAP_HELPER
    10. Only retrieval avail          → RETRIEVE_MEMORY
    11. Nothing available             → REPORT_UNCERTAINTY
    """

    CONFIDENCE_THRESHOLD = 0.75    # above this: stop and answer
    MINIMUM_VIABLE_CONFIDENCE = 0.35  # below this after N iterations: report uncertainty

    def evaluate(self, task: TaskState, experts: AvailableExperts) -> DecisionRecord:
        """Return the decision MoK should take right now."""

        # Rule 1: already confident enough
        if task.current_confidence >= self.CONFIDENCE_THRESHOLD and task.has_findings:
            return DecisionRecord(
                decision=MoKDecision.CONFIDENCE_SUFFICIENT,
                reason=f"Confidence {task.current_confidence:.2f} exceeds threshold {self.CONFIDENCE_THRESHOLD}",
                preferred_expert_tier="none",
                confidence_after=task.current_confidence,
            )

        # Rule 2: hit iteration limit with low confidence
        if task.at_limit:
            return DecisionRecord(
                decision=MoKDecision.REPORT_UNCERTAINTY,
                reason=(
                    f"Hit max iterations ({task.max_iterations}) with confidence "
                    f"{task.current_confidence:.2f}. Reporting limits to user."
                ),
                preferred_expert_tier="none",
                confidence_after=task.current_confidence,
            )

        # Rule 3: trivially simple prompt
        if _prompt_is_simple(task.prompt) and task.is_first_iteration:
            return DecisionRecord(
                decision=MoKDecision.ANSWER_DIRECT,
                reason="Prompt matches simple fact-query pattern and is short",
                preferred_expert_tier="none",
                confidence_after=0.9,
            )

        # Rule 4: no experts and no resources
        if not experts.has_cheap and not experts.has_strong and not experts.has_retrieval:
            return DecisionRecord(
                decision=MoKDecision.REPORT_UNCERTAINTY,
                reason="No experts available within resource budget",
                preferred_expert_tier="none",
                confidence_after=0.0,
            )

        # Rule 5: task needs retrieval and it hasn't been done
        if _prompt_needs_retrieval(task.prompt) and not task.retrieval_done:
            if experts.has_retrieval:
                return DecisionRecord(
                    decision=MoKDecision.RETRIEVE_MEMORY,
                    reason="Prompt references prior session or stored documents; retrieve first",
                    preferred_expert_tier="retrieval",
                    confidence_after=task.current_confidence,
                )

        # Rule 6: complex task + both experts available + time not tight
        if _prompt_is_complex(task.prompt) and experts.can_multi_expert and not experts.is_time_tight:
            return DecisionRecord(
                decision=MoKDecision.MULTI_EXPERT,
                reason="Complex task; two experts available; time budget allows comparison",
                preferred_expert_tier="both",
                confidence_after=0.0,
            )

        # Rule 7: code task + strong code-specialist available
        if _prompt_needs_code_expert(task.prompt) and experts.has_strong:
            return DecisionRecord(
                decision=MoKDecision.CALL_STRONG_HELPER,
                reason="Code-specific task; routing to strong specialist",
                preferred_expert_tier="strong",
                confidence_after=0.0,
            )

        # Rule 8: complex task + strong available
        if _prompt_is_complex(task.prompt) and experts.has_strong:
            return DecisionRecord(
                decision=MoKDecision.CALL_STRONG_HELPER,
                reason="Complex task; escalating to strong helper",
                preferred_expert_tier="strong",
                confidence_after=0.0,
            )

        # Rule 9: cheap expert available (default path)
        if experts.has_cheap:
            return DecisionRecord(
                decision=MoKDecision.CALL_CHEAP_HELPER,
                reason="Task within cheap-expert capability range; using resident small model",
                preferred_expert_tier="cheap",
                confidence_after=0.0,
            )

        # Rule 10: only retrieval left
        if experts.has_retrieval:
            return DecisionRecord(
                decision=MoKDecision.RETRIEVE_MEMORY,
                reason="Only retrieval model available; using memory lookup",
                preferred_expert_tier="retrieval",
                confidence_after=task.current_confidence,
            )

        # Rule 11: truly nothing
        return DecisionRecord(
            decision=MoKDecision.REPORT_UNCERTAINTY,
            reason="No viable experts or retrieval models within current resource constraints",
            preferred_expert_tier="none",
            confidence_after=0.0,
        )

    def run_loop(
        self,
        prompt: str,
        experts: AvailableExperts,
        max_iterations: int = 3,
    ) -> list[DecisionRecord]:
        """Simulate the full decision loop without calling any model.

        Useful for testing routing logic and generating training traces.
        Returns the sequence of decisions MoK would make.
        """
        task = TaskState(prompt=prompt, max_iterations=max_iterations)
        records: list[DecisionRecord] = []

        for _ in range(max_iterations + 1):
            record = self.evaluate(task, experts)
            records.append(record)

            # Terminal decisions — stop the loop
            if record.decision in (
                MoKDecision.ANSWER_DIRECT,
                MoKDecision.CONFIDENCE_SUFFICIENT,
                MoKDecision.REPORT_UNCERTAINTY,
            ):
                break

            # Simulate progress: after calling a helper, confidence grows slightly
            task = TaskState(
                prompt=prompt,
                current_confidence=min(task.current_confidence + 0.3, 0.9),
                findings_so_far=task.findings_so_far + [f"finding_{task.iteration}"],
                experts_called=task.experts_called + [record.preferred_expert_tier],
                retrieval_done=(record.decision == MoKDecision.RETRIEVE_MEMORY or task.retrieval_done),
                iteration=task.iteration + 1,
                max_iterations=max_iterations,
            )

        return records
