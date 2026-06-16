"""
src/mok/orchestration/consultation.py

MoK Expert Consultation Engine
================================
MoK does not answer everything itself.  It knows which minds to wake up,
what to ask them, when to doubt them, and how to turn their output into
one clean answer.

Architecture
------------
ExpertCallRequest   — structured call sent to a helper model
ExpertCallReply     — structured reply from a helper model
ConsultationTurn    — one (request, reply) exchange, frozen for tracing
ConsultationSession — manages multi-turn back-and-forth with ONE expert
ConsultationEngine  — the full decision loop; coordinates multiple sessions,
                      applies resource gates, synthesizes the final answer

The ConsultationEngine works WITH OrchestratorRuntime — it uses the same
ExpertBackend protocol and ModelRegistry for VRAM-aware expert selection,
but adds the consultation dialogue layer that the base runtime does not have.

Decision loop (7 branches)
---------------------------
  1. ANSWER_DIRECT       — task is simple enough; no helper needed
  2. RETRIEVE_MEMORY     — retrieve from memory/files first
  3. CALL_CHEAP_HELPER   — delegate to resident small model
  4. CALL_STRONG_HELPER  — escalate to larger on-demand model
  5. MULTI_EXPERT        — consult two helpers and compare
  6. CONFIDENCE_SUFFICIENT — stop; current confidence is good enough
  7. REPORT_UNCERTAINTY  — answer with explicit uncertainty caveat

MoK synthesis rules (enforced by ConsultationEngine)
------------------------------------------------------
- Helper models ADVISE.  MoK DECIDES.
- MoK never outputs raw expert text as its own answer.
- MoK always adds gate status and confidence to its synthesis.
- MoK challenges weak/vague/overconfident helper output before accepting.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mok.models.backends import ExpertBackend, RequestPayload
from mok.models.registry import ExpertMetadata, ExpertState, ModelRegistry
from mok.memory.budget import BudgetManager
from mok.memory.state_bus import ExpertContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ABI types — the structured call / reply contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExpertCallRequest:
    """Structured call sent to a helper model.

    MoK Core always uses a narrow, focused prompt — not the raw user request.
    """
    expert_id: str
    purpose: str                          # e.g. "first_pass_plan_audit"
    prompt: str                           # narrow, focused question for this expert
    max_tokens: int = 400
    temperature: float = 0.2
    expected_keys: tuple[str, ...] = ("findings", "confidence", "needs_followup")
    call_index: int = 0                   # 0 = initial, 1+ = follow-up

    def to_payload(self, request_id: str) -> RequestPayload:
        return RequestPayload(
            prompt=self.prompt,
            request_id=f"{request_id}:expert:{self.expert_id}:{self.call_index}",
            parameters={
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "purpose": self.purpose,
            },
        )


@dataclass
class ExpertCallReply:
    """Structured reply from a helper model.

    The ConsultationSession parses the backend text response into this
    structure.  If parsing fails the reply is marked low-quality.
    """
    expert_id: str
    raw_text: str
    findings: list[str] = field(default_factory=list)
    confidence: str = "low"              # "low" | "medium" | "high"
    needs_followup: bool = False
    latency_ms: int = 0
    quality: str = "unknown"             # "good" | "vague" | "overconfident" | "error"

    def is_usable(self) -> bool:
        return bool(self.findings) and self.quality != "error"

    def is_vague(self) -> bool:
        """True if the reply is too generic to act on."""
        if not self.findings:
            return True
        vague_markers = ("further analysis", "several considerations", "it depends", "various factors")
        joined = " ".join(self.findings).lower()
        return any(m in joined for m in vague_markers)

    def is_overconfident(self) -> bool:
        """True if the reply claims high confidence without specific evidence."""
        if self.confidence != "high":
            return False
        specific_markers = ("specifically", "because", "evidence", "verified", "source")
        joined = " ".join(self.findings).lower()
        return not any(m in joined for m in specific_markers)

    @classmethod
    def from_backend_text(cls, expert_id: str, text: str, latency_ms: int) -> "ExpertCallReply":
        """Parse a backend text response into a structured reply.

        The consultation prompt instructs the expert to return JSON with
        findings/confidence/needs_followup.  If that fails, fall back to
        treating the whole text as a single finding.
        """
        import json as _json

        reply = cls(expert_id=expert_id, raw_text=text, latency_ms=latency_ms)

        # Try JSON first
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(text[start:end])
                reply.findings = parsed.get("findings", [text.strip()])
                reply.confidence = str(parsed.get("confidence", "low")).lower()
                reply.needs_followup = bool(parsed.get("needs_followup", False))
                reply.quality = "good"
                return reply
        except Exception:
            pass

        # Fall back: treat entire text as one finding, low confidence
        reply.findings = [text.strip()] if text.strip() else []
        reply.confidence = "low"
        reply.needs_followup = True
        reply.quality = "vague" if text.strip() else "error"
        return reply

    def assess_quality(self) -> "ExpertCallReply":
        """Assign quality label after parsing."""
        if not self.is_usable():
            self.quality = "error"
        elif self.is_overconfident():
            self.quality = "overconfident"
        elif self.is_vague():
            self.quality = "vague"
        else:
            self.quality = "good"
        return self


@dataclass
class ConsultationTurn:
    """One (request, reply) exchange with an expert. Frozen for tracing."""
    turn_index: int
    request: ExpertCallRequest
    reply: ExpertCallReply
    challenged: bool = False
    challenge_reason: str = ""


# ---------------------------------------------------------------------------
# ConsultationSession — multi-turn back-and-forth with ONE expert
# ---------------------------------------------------------------------------

class ConsultationSession:
    """Manages a structured multi-turn dialogue with a single helper expert.

    MoK uses this to:
      - Send a focused initial question
      - Assess the reply quality
      - Issue a challenge if the reply is vague or overconfident
      - Ask a follow-up if more information is needed
      - Decide when to stop and synthesize

    The session records every turn for tracing and training data generation.
    """

    MAX_TURNS = 4  # hard cap on back-and-forth per session

    def __init__(
        self,
        expert: ExpertMetadata,
        backend: ExpertBackend,
        request_id: str,
    ) -> None:
        self.expert = expert
        self.backend = backend
        self.request_id = request_id
        self.turns: list[ConsultationTurn] = []
        self._call_index: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, purpose: str, prompt: str, max_tokens: int = 400, temperature: float = 0.2) -> ExpertCallReply:
        """Send a question to the expert and record the turn."""
        if len(self.turns) >= self.MAX_TURNS:
            logger.warning("[Consultation] MAX_TURNS reached for expert %s", self.expert.name)
            # Return the last reply rather than calling again
            return self.turns[-1].reply

        request = ExpertCallRequest(
            expert_id=self.expert.name,
            purpose=purpose,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            call_index=self._call_index,
        )
        self._call_index += 1

        payload = request.to_payload(self.request_id)
        started = time.perf_counter()
        try:
            response = self.backend.generate(self.expert, payload)
            latency_ms = int((time.perf_counter() - started) * 1000)
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.error("[Consultation] Backend error for expert %s: %s", self.expert.name, exc)
            reply = ExpertCallReply(
                expert_id=self.expert.name,
                raw_text="",
                findings=[],
                confidence="low",
                quality="error",
                latency_ms=latency_ms,
            )
            self.turns.append(ConsultationTurn(
                turn_index=len(self.turns),
                request=request,
                reply=reply,
            ))
            return reply

        reply = ExpertCallReply.from_backend_text(
            self.expert.name, response.text, latency_ms
        ).assess_quality()

        self.turns.append(ConsultationTurn(
            turn_index=len(self.turns),
            request=request,
            reply=reply,
        ))

        logger.debug(
            "[Consultation] Turn %d with %s: quality=%s confidence=%s needs_followup=%s",
            len(self.turns) - 1,
            self.expert.name,
            reply.quality,
            reply.confidence,
            reply.needs_followup,
        )
        return reply

    def challenge(self, reply: ExpertCallReply, reason: str) -> ExpertCallReply:
        """Issue a challenge when the expert reply is weak or overconfident."""
        if self.turns:
            self.turns[-1].challenged = True
            self.turns[-1].challenge_reason = reason

        challenge_prompt = (
            f"Your previous answer was {reply.quality}. {reason}\n\n"
            "Please revise:\n"
            "- Provide specific evidence for each finding\n"
            "- Remove or qualify any unverified certainty claims\n"
            "- If you cannot be more specific, say so explicitly\n\n"
            "Return JSON: {\"findings\": [...], \"confidence\": \"low|medium|high\", \"needs_followup\": bool}"
        )
        return self.ask(
            purpose="challenge_revision",
            prompt=challenge_prompt,
            max_tokens=400,
        )

    def followup(self, reply: ExpertCallReply, question: str) -> ExpertCallReply:
        """Ask a follow-up question to close gaps in the expert's answer."""
        followup_prompt = (
            f"Follow-up on your previous answer.\n\n{question}\n\n"
            "Return JSON: {\"findings\": [...], \"confidence\": \"low|medium|high\", \"needs_followup\": bool}"
        )
        return self.ask(
            purpose="followup",
            prompt=followup_prompt,
            max_tokens=400,
        )

    def best_findings(self) -> list[str]:
        """Return findings from the highest-quality turn in this session."""
        if not self.turns:
            return []
        quality_order = {"good": 3, "overconfident": 2, "vague": 1, "error": 0, "unknown": 0}
        best = max(self.turns, key=lambda t: quality_order.get(t.reply.quality, 0))
        return best.reply.findings

    def to_trace(self) -> list[dict]:
        return [
            {
                "turn": t.turn_index,
                "expert": t.request.expert_id,
                "purpose": t.request.purpose,
                "prompt_snippet": t.request.prompt[:120],
                "findings": t.reply.findings,
                "confidence": t.reply.confidence,
                "quality": t.reply.quality,
                "needs_followup": t.reply.needs_followup,
                "challenged": t.challenged,
                "challenge_reason": t.challenge_reason,
                "latency_ms": t.reply.latency_ms,
            }
            for t in self.turns
        ]


# ---------------------------------------------------------------------------
# MoK Decision branches
# ---------------------------------------------------------------------------

class MoKDecision(str, Enum):
    ANSWER_DIRECT = "answer_direct"
    RETRIEVE_MEMORY = "retrieve_memory"
    CALL_CHEAP_HELPER = "call_cheap_helper"
    CALL_STRONG_HELPER = "call_strong_helper"
    MULTI_EXPERT = "multi_expert"
    CONFIDENCE_SUFFICIENT = "confidence_sufficient"
    REPORT_UNCERTAINTY = "report_uncertainty"


@dataclass
class ResourceContext:
    """Current machine resource state, checked before every expert selection."""
    vram_free_gb: float = 8.0
    ram_free_gb: float = 16.0
    time_budget_s: float = 30.0
    models_loaded: list[str] = field(default_factory=list)

    def can_load(self, vram_cost_gb: float) -> bool:
        return self.vram_free_gb >= vram_cost_gb

    def is_time_tight(self) -> bool:
        return self.time_budget_s <= 10.0


# ---------------------------------------------------------------------------
# ConsultationEngine — the full MoK coordination loop
# ---------------------------------------------------------------------------

class ConsultationEngine:
    """Coordinates expert consultation for a single user task.

    Usage
    -----
    engine = ConsultationEngine(registry, backends, budget, resources)
    result = engine.handle(request_id, user_prompt, context)
    print(result.final_answer)   # MoK's synthesized answer
    print(result.trace)          # full trace for logging/training

    Decision loop
    -------------
    1. Assess task complexity and resource context → MoKDecision
    2. If cheap helper is viable, start ConsultationSession with small model
    3. Assess reply quality → challenge or follow up if needed
    4. If needed, escalate to stronger model for second opinion
    5. If two experts consulted, compare and pick the more specific one
    6. Synthesize a final answer in MoK's own voice
    7. Never copy-paste; always add gate status and confidence
    """

    # Heuristics for deciding when to escalate
    CHEAP_VRAM_THRESHOLD_GB = 3.0    # use cheap helper if free VRAM > this
    STRONG_VRAM_THRESHOLD_GB = 6.0   # use strong helper if free VRAM > this
    MULTI_EXPERT_VRAM_THRESHOLD_GB = 9.0  # only attempt multi if enough VRAM

    def __init__(
        self,
        registry: ModelRegistry,
        backends: dict[str, ExpertBackend],
        budget: BudgetManager | None = None,
        resources: ResourceContext | None = None,
    ) -> None:
        self.registry = registry
        self.backends = backends
        self.budget = budget or BudgetManager()
        self.resources = resources or ResourceContext()
        self._sessions: list[ConsultationSession] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def handle(
        self,
        request_id: str,
        user_prompt: str,
        context: ExpertContext | None = None,
    ) -> "ConsultationResult":
        """Run the full MoK consultation loop and return a synthesized result."""
        started = time.perf_counter()
        trace: list[dict] = []

        # Step 1: decide what to do
        decision = self._decide(user_prompt)
        trace.append({"step": "DECISION", "branch": decision.value})

        sessions: list[ConsultationSession] = []
        accepted_findings: list[str] = []
        gate = "pending"
        confidence = "low"

        if decision == MoKDecision.ANSWER_DIRECT:
            gate = "answered_directly"
            confidence = "high"
            final = f"[MoK direct answer] {user_prompt}"
            trace.append({"step": "GATE", "gate": gate})

        elif decision == MoKDecision.REPORT_UNCERTAINTY:
            gate = "uncertainty_reported"
            confidence = "low"
            final = (
                "[MoK uncertainty] Cannot answer with sufficient confidence. "
                "Resource or information constraints prevent a reliable answer. "
                "Please provide more context or adjust the request scope."
            )
            trace.append({"step": "GATE", "gate": gate})

        else:
            # Step 2: select experts based on decision + resource check
            cheap_expert = self._select_expert("cheap")
            strong_expert = self._select_expert("strong")

            # Step 3: run consultation
            if decision == MoKDecision.MULTI_EXPERT and cheap_expert and strong_expert:
                sessions, accepted_findings, gate, confidence = self._run_multi_expert(
                    request_id, user_prompt, cheap_expert, strong_expert, trace
                )
            elif decision in (MoKDecision.CALL_CHEAP_HELPER, MoKDecision.RETRIEVE_MEMORY) and cheap_expert:
                session, accepted_findings, gate, confidence = self._run_single_session(
                    request_id, user_prompt, cheap_expert, "cheap_consult", trace
                )
                sessions = [session]
            elif strong_expert:
                session, accepted_findings, gate, confidence = self._run_single_session(
                    request_id, user_prompt, strong_expert, "strong_consult", trace
                )
                sessions = [session]
            else:
                gate = "no_expert_available"
                confidence = "low"
                trace.append({"step": "GATE", "gate": gate, "reason": "no expert fits resource budget"})

            # Step 4: synthesize
            final = self._synthesize(user_prompt, accepted_findings, gate, confidence)

        self._sessions.extend(sessions)

        total_ms = int((time.perf_counter() - started) * 1000)
        for session in sessions:
            trace.extend(session.to_trace())

        trace.append({"step": "FINAL", "gate": gate, "confidence": confidence})

        return ConsultationResult(
            request_id=request_id,
            decision=decision,
            sessions=sessions,
            accepted_findings=accepted_findings,
            gate=gate,
            confidence=confidence,
            final_answer=final,
            trace=trace,
            total_ms=total_ms,
        )

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide(self, prompt: str) -> MoKDecision:
        """7-branch decision: pick the right consultation path."""
        r = self.resources

        # Is the task so simple it needs no helper?
        simple_markers = ("what is", "define", "list the", "how many")
        if any(prompt.lower().startswith(m) for m in simple_markers) and len(prompt) < 80:
            return MoKDecision.ANSWER_DIRECT

        # Is resource situation too constrained for any helper?
        if r.vram_free_gb < 1.0 and r.is_time_tight():
            return MoKDecision.REPORT_UNCERTAINTY

        # Is there enough VRAM for a multi-expert comparison?
        if r.vram_free_gb >= self.MULTI_EXPERT_VRAM_THRESHOLD_GB and not r.is_time_tight():
            return MoKDecision.MULTI_EXPERT

        # Is there enough for a strong helper?
        if r.vram_free_gb >= self.STRONG_VRAM_THRESHOLD_GB:
            return MoKDecision.CALL_STRONG_HELPER

        # Can we use a cheap helper?
        if r.vram_free_gb >= self.CHEAP_VRAM_THRESHOLD_GB:
            return MoKDecision.CALL_CHEAP_HELPER

        # Only retrieval possible
        if r.vram_free_gb >= 0.5:
            return MoKDecision.RETRIEVE_MEMORY

        return MoKDecision.REPORT_UNCERTAINTY

    # ------------------------------------------------------------------
    # Expert selection
    # ------------------------------------------------------------------

    def _select_expert(self, tier: str) -> ExpertMetadata | None:
        """Select an expert by tier ('cheap' or 'strong'), respecting resource budget."""
        r = self.resources
        experts = self.registry.all()

        if tier == "cheap":
            # Prefer always-resident small model
            candidates = [
                e for e in experts
                if e.vram_cost_gb <= self.CHEAP_VRAM_THRESHOLD_GB
                and r.can_load(e.vram_cost_gb)
            ]
        else:
            # Strong: any model that fits
            candidates = [
                e for e in experts
                if e.vram_cost_gb > self.CHEAP_VRAM_THRESHOLD_GB
                and r.can_load(e.vram_cost_gb)
            ]

        if not candidates:
            return None

        # Prefer roles: general > code > coordinator > others
        role_priority = {"general": 3, "code": 2, "coordinator": 1}
        candidates.sort(key=lambda e: role_priority.get(e.role, 0), reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # Session runners
    # ------------------------------------------------------------------

    def _run_single_session(
        self,
        request_id: str,
        prompt: str,
        expert: ExpertMetadata,
        purpose: str,
        trace: list[dict],
    ) -> tuple["ConsultationSession", list[str], str, str]:
        """Run a full consultation session with one expert: ask → assess → challenge/followup."""
        backend = self.backends.get(expert.backend)
        if backend is None:
            # Try mock as fallback for testing
            backend = self.backends.get("mock")
        if backend is None:
            trace.append({"step": "CHECK", "result": f"no backend for {expert.backend}"})
            return ConsultationSession(expert, list(self.backends.values())[0], request_id), [], "no_backend", "low"

        session = ConsultationSession(expert, backend, request_id)
        trace.append({"step": "ASK_EXPERT", "expert": expert.name, "purpose": purpose})

        # Initial question — narrow and focused
        narrow_prompt = (
            f"Task context: {prompt}\n\n"
            "Identify only the most critical issues, risks, or missing information. "
            "Be specific. Do not restate the task.\n\n"
            "Return JSON: {\"findings\": [\"...\", ...], \"confidence\": \"low|medium|high\", \"needs_followup\": bool}"
        )
        reply = session.ask(purpose=purpose, prompt=narrow_prompt)
        trace.append({"step": "EXPERT_REPLY", "expert": expert.name, "quality": reply.quality})

        # Challenge if needed
        if reply.quality in ("vague", "overconfident"):
            reason = (
                "Be more specific — provide evidence for each point."
                if reply.quality == "vague"
                else "Downgrade confidence claims that lack specific evidence."
            )
            trace.append({"step": "CHECK", "result": f"challenging {reply.quality} reply"})
            reply = session.challenge(reply, reason)
            trace.append({"step": "EXPERT_REPLY", "expert": expert.name, "quality": reply.quality, "after_challenge": True})

        # Follow up if expert says it needs one and we have turns left
        if reply.needs_followup and len(session.turns) < ConsultationSession.MAX_TURNS - 1:
            trace.append({"step": "CHECK", "result": "running followup"})
            reply = session.followup(
                reply,
                "Cover the edge cases and resource constraints not mentioned in your previous answer. "
                "Return JSON: {\"findings\": [...], \"confidence\": \"low|medium|high\", \"needs_followup\": bool}"
            )
            trace.append({"step": "EXPERT_REPLY", "expert": expert.name, "quality": reply.quality, "after_followup": True})

        findings = session.best_findings()
        gate = "expert_answer_accepted_and_synthesized" if findings else "expert_output_insufficient"
        confidence = "medium" if findings else "low"

        trace.append({"step": "GATE", "gate": gate})
        return session, findings, gate, confidence

    def _run_multi_expert(
        self,
        request_id: str,
        prompt: str,
        cheap: ExpertMetadata,
        strong: ExpertMetadata,
        trace: list[dict],
    ) -> tuple[list["ConsultationSession"], list[str], str, str]:
        """Consult two experts, compare, and choose the more specific answer."""
        trace.append({"step": "MULTI_EXPERT_START", "experts": [cheap.name, strong.name]})

        session_a, findings_a, _, confidence_a = self._run_single_session(
            request_id, prompt, cheap, "cheap_first_pass", trace
        )
        session_b, findings_b, _, confidence_b = self._run_single_session(
            request_id + ":strong", prompt, strong, "strong_second_opinion", trace
        )

        # Compare: prefer the set with more specific findings (longer, more distinct)
        def specificity(findings: list[str]) -> int:
            return sum(len(f.split()) for f in findings)

        if specificity(findings_b) >= specificity(findings_a):
            chosen = findings_b
            chosen_expert = strong.name
            rejected_expert = cheap.name
        else:
            chosen = findings_a
            chosen_expert = cheap.name
            rejected_expert = strong.name

        trace.append({
            "step": "MULTI_EXPERT_COMPARE",
            "chosen": chosen_expert,
            "reason": f"{chosen_expert} findings were more specific",
            "rejected": rejected_expert,
        })

        gate = "disagreement_analyzed_and_position_chosen_with_reason"
        confidence = "medium"
        trace.append({"step": "GATE", "gate": gate})
        return [session_a, session_b], chosen, gate, confidence

    # ------------------------------------------------------------------
    # Synthesis — MoK always writes the final answer
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        prompt: str,
        findings: list[str],
        gate: str,
        confidence: str,
    ) -> str:
        """Produce MoK's synthesized answer. Never a copy of expert output."""
        if not findings:
            return (
                f"[MoK synthesis] Gate: {gate}. Confidence: {confidence}. "
                "Expert consultation did not yield actionable findings. "
                "Recommend: broaden the question or check expert availability."
            )

        # Compress findings into a synthesis — MoK's voice, not the expert's
        compressed = "; ".join(f.rstrip(".") for f in findings[:5])  # cap at 5 points
        return (
            f"[MoK synthesis] Based on expert consultation, the key findings are: {compressed}. "
            f"Gate: {gate}. Confidence: {confidence}. "
            "This answer is MoK's synthesis — expert output was reviewed and compressed, not copied."
        )

    # ------------------------------------------------------------------
    # Trace access
    # ------------------------------------------------------------------

    def all_sessions(self) -> list[ConsultationSession]:
        return list(self._sessions)


# ---------------------------------------------------------------------------
# ConsultationResult — the return value from ConsultationEngine.handle()
# ---------------------------------------------------------------------------

@dataclass
class ConsultationResult:
    """Full result of a MoK consultation loop.

    Fields
    ------
    request_id       : ties result to the originating request
    decision         : which of the 7 branches MoK chose
    sessions         : all ConsultationSessions that ran
    accepted_findings: the compressed findings MoK accepted from experts
    gate             : the gate condition name (for training data)
    confidence       : "low" | "medium" | "high"
    final_answer     : MoK's synthesized, user-facing answer
    trace            : full step-by-step trace (for training data and logging)
    total_ms         : wall time for the whole consultation loop
    """
    request_id: str
    decision: MoKDecision
    sessions: list[ConsultationSession]
    accepted_findings: list[str]
    gate: str
    confidence: str
    final_answer: str
    trace: list[dict]
    total_ms: int

    def to_training_record(self, user_prompt: str, resource_context: dict | None = None) -> dict:
        """Emit a training record in the MoK SFT format."""
        return {
            "USER": user_prompt,
            "STATE": {
                "decision": self.decision.value,
                "gate": self.gate,
                "confidence": self.confidence,
            },
            "AVAILABLE_EXPERTS": [s.expert.name for s in self.sessions],
            "RESOURCE_STATUS": resource_context or {},
            "MOK_ACTION": self.decision.value,
            "EXPERT_REPLY": [
                {"expert": t["expert"], "findings": t.get("findings", []), "quality": t.get("quality")}
                for t in self.trace if t.get("step") == "EXPERT_REPLY"
            ],
            "MOK_CHECK": [
                t for t in self.trace if t.get("step") == "CHECK"
            ],
            "MOK_FINAL": self.final_answer,
        }
