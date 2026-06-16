"""
tests/test_consultation.py

Tests for the MoK consultation engine and decision loop.

Covers:
  - ExpertCallRequest / ExpertCallReply parsing
  - ConsultationSession: ask, challenge, followup, best_findings
  - ConsultationEngine: single expert, multi-expert, resource gating
  - DecisionLoop: all 7 branches + run_loop simulation
  - ConsultationResult.to_training_record()
"""
from __future__ import annotations

import pytest

from mok.models.backends import BackendResponse, MockBackend, RequestPayload
from mok.models.registry import ExpertMetadata, ExpertState, ModelRegistry
from mok.orchestration.consultation import (
    ConsultationEngine,
    ConsultationSession,
    ExpertCallReply,
    ExpertCallRequest,
    MoKDecision,
    ResourceContext,
)
from mok.routing.decision_loop import (
    AvailableExperts,
    DecisionLoop,
    DecisionRecord,
    MoKDecision as LoopDecision,
    TaskState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_expert(name: str, role: str, vram: float, state: str = "resident") -> ExpertMetadata:
    return ExpertMetadata(
        name=name,
        role=role,
        kind="llm",
        backend="mock",
        api_url=None,
        base_id=None,
        adapter_path=None,
        vram_cost_gb=vram,
        ram_cost_gb=vram * 1.5,
        current_device="cuda",
        state=ExpertState(state),
    )


@pytest.fixture
def cheap_expert() -> ExpertMetadata:
    return _make_expert("fast_3b", "general", 2.5)


@pytest.fixture
def strong_expert() -> ExpertMetadata:
    return _make_expert("general_7b", "general", 7.0)


@pytest.fixture
def code_expert() -> ExpertMetadata:
    return _make_expert("coder_7b", "code", 6.5)


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend()


@pytest.fixture
def rich_resources() -> ResourceContext:
    return ResourceContext(vram_free_gb=12.0, ram_free_gb=24.0, time_budget_s=60.0)


@pytest.fixture
def tight_resources() -> ResourceContext:
    return ResourceContext(vram_free_gb=1.5, ram_free_gb=4.0, time_budget_s=8.0)


@pytest.fixture
def registry(cheap_expert, strong_expert, code_expert) -> ModelRegistry:
    return ModelRegistry([cheap_expert, strong_expert, code_expert])


@pytest.fixture
def backends(mock_backend) -> dict:
    return {"mock": mock_backend}


# ---------------------------------------------------------------------------
# ExpertCallRequest
# ---------------------------------------------------------------------------

class TestExpertCallRequest:
    def test_to_payload_sets_request_id(self, cheap_expert):
        req = ExpertCallRequest(
            expert_id=cheap_expert.name,
            purpose="test_purpose",
            prompt="What are the risks?",
            call_index=2,
        )
        payload = req.to_payload("req-abc")
        assert "req-abc" in payload.request_id
        assert cheap_expert.name in payload.request_id
        assert "2" in payload.request_id

    def test_to_payload_carries_parameters(self):
        req = ExpertCallRequest(
            expert_id="x",
            purpose="audit",
            prompt="check this",
            max_tokens=200,
            temperature=0.1,
        )
        payload = req.to_payload("r1")
        assert payload.parameters["max_tokens"] == 200
        assert payload.parameters["temperature"] == 0.1


# ---------------------------------------------------------------------------
# ExpertCallReply
# ---------------------------------------------------------------------------

class TestExpertCallReply:
    def test_parse_valid_json(self):
        text = '{"findings": ["issue A", "issue B"], "confidence": "medium", "needs_followup": true}'
        reply = ExpertCallReply.from_backend_text("expert_x", text, 100)
        assert reply.findings == ["issue A", "issue B"]
        assert reply.confidence == "medium"
        assert reply.needs_followup is True
        assert reply.quality == "good"

    def test_parse_fallback_on_prose(self):
        text = "There are several considerations here. Further analysis may be required."
        reply = ExpertCallReply.from_backend_text("expert_x", text, 50)
        assert len(reply.findings) == 1
        assert reply.confidence == "low"
        assert reply.quality == "vague"

    def test_is_vague_detection(self):
        reply = ExpertCallReply(
            expert_id="x",
            raw_text="",
            findings=["several considerations related to the task.", "further analysis may be required."],
            confidence="low",
        )
        assert reply.is_vague() is True

    def test_is_overconfident_detection(self):
        reply = ExpertCallReply(
            expert_id="x",
            raw_text="",
            findings=["The answer is definitively resolved."],
            confidence="high",
        )
        assert reply.is_overconfident() is True

    def test_good_reply_not_flagged(self):
        reply = ExpertCallReply(
            expert_id="x",
            raw_text="",
            findings=["Specifically, the VRAM allocation is missing a bounds check."],
            confidence="medium",
        )
        reply.assess_quality()
        assert reply.quality == "good"
        assert reply.is_vague() is False
        assert reply.is_overconfident() is False

    def test_json_embedded_in_prose(self):
        text = 'Sure, here you go: {"findings": ["risk 1"], "confidence": "high", "needs_followup": false} hope that helps!'
        reply = ExpertCallReply.from_backend_text("e", text, 30)
        assert "risk 1" in reply.findings
        assert reply.confidence == "high"


# ---------------------------------------------------------------------------
# ConsultationSession
# ---------------------------------------------------------------------------

class TestConsultationSession:
    def test_ask_records_turn(self, cheap_expert, mock_backend):
        session = ConsultationSession(cheap_expert, mock_backend, "req-1")
        reply = session.ask("audit", "What risks exist?")
        assert len(session.turns) == 1
        assert session.turns[0].request.purpose == "audit"

    def test_challenge_marks_turn(self, cheap_expert, mock_backend):
        session = ConsultationSession(cheap_expert, mock_backend, "req-2")
        reply = session.ask("audit", "check this")
        # Manually make it vague so challenge fires
        session.turns[-1].reply.quality = "vague"
        session.challenge(reply, "Too vague — be specific.")
        assert session.turns[0].challenged is True
        assert len(session.turns) == 2

    def test_followup_increments_turns(self, cheap_expert, mock_backend):
        session = ConsultationSession(cheap_expert, mock_backend, "req-3")
        reply = session.ask("first_pass", "initial question")
        session.followup(reply, "What edge cases did you miss?")
        assert len(session.turns) == 2

    def test_max_turns_enforced(self, cheap_expert, mock_backend):
        session = ConsultationSession(cheap_expert, mock_backend, "req-4")
        for _ in range(ConsultationSession.MAX_TURNS + 2):
            session.ask("loop", "another question")
        assert len(session.turns) == ConsultationSession.MAX_TURNS

    def test_best_findings_returns_nonempty(self, cheap_expert, mock_backend):
        session = ConsultationSession(cheap_expert, mock_backend, "req-5")
        session.ask("audit", "find issues")
        findings = session.best_findings()
        assert isinstance(findings, list)

    def test_to_trace_structure(self, cheap_expert, mock_backend):
        session = ConsultationSession(cheap_expert, mock_backend, "req-6")
        session.ask("test", "question")
        trace = session.to_trace()
        assert len(trace) == 1
        assert "turn" in trace[0]
        assert "expert" in trace[0]
        assert "quality" in trace[0]


# ---------------------------------------------------------------------------
# ConsultationEngine
# ---------------------------------------------------------------------------

class TestConsultationEngine:
    def test_single_cheap_expert_consult(self, registry, backends, cheap_expert):
        resources = ResourceContext(vram_free_gb=4.0, time_budget_s=30.0)
        engine = ConsultationEngine(registry, backends, resources=resources)
        result = engine.handle("req-10", "Identify the main risks in this deployment plan.")
        assert result.final_answer.startswith("[MoK synthesis]")
        assert result.gate != "pending"
        assert result.confidence in ("low", "medium", "high")

    def test_direct_answer_for_simple_prompt(self, registry, backends):
        resources = ResourceContext(vram_free_gb=4.0, time_budget_s=30.0)
        engine = ConsultationEngine(registry, backends, resources=resources)
        result = engine.handle("req-11", "What is a transformer model?")
        assert result.decision == MoKDecision.ANSWER_DIRECT

    def test_uncertainty_when_no_vram(self, registry, backends):
        resources = ResourceContext(vram_free_gb=0.3, time_budget_s=5.0)
        engine = ConsultationEngine(registry, backends, resources=resources)
        result = engine.handle("req-12", "Analyze the architecture of this complex distributed system.")
        assert result.decision == MoKDecision.REPORT_UNCERTAINTY

    def test_multi_expert_when_vram_sufficient(self, registry, backends):
        resources = ResourceContext(
            vram_free_gb=12.0,
            time_budget_s=60.0,
            models_loaded=[],
        )
        engine = ConsultationEngine(registry, backends, resources=resources)
        result = engine.handle("req-13", "Compare and evaluate these two architectural approaches.")
        # Multi-expert requires both cheap and strong experts visible
        assert result.decision in (MoKDecision.MULTI_EXPERT, MoKDecision.CALL_STRONG_HELPER)

    def test_final_answer_never_just_raw_expert_text(self, registry, backends):
        """MoK synthesis must not be a verbatim copy of MockBackend output."""
        resources = ResourceContext(vram_free_gb=4.0, time_budget_s=30.0)
        engine = ConsultationEngine(registry, backends, resources=resources)
        result = engine.handle("req-14", "Audit this plan for missing assumptions.")
        # MockBackend always starts with "[expert_name] Mock..."
        # MoK synthesis must NOT start that way
        assert not result.final_answer.startswith("[fast_3b]")
        assert not result.final_answer.startswith("[general_7b]")
        assert "MoK synthesis" in result.final_answer or "MoK" in result.final_answer

    def test_trace_contains_expected_steps(self, registry, backends):
        resources = ResourceContext(vram_free_gb=4.0, time_budget_s=30.0)
        engine = ConsultationEngine(registry, backends, resources=resources)
        result = engine.handle("req-15", "Review this code for bugs.")
        step_names = set()
        for t in result.trace:
            if isinstance(t, dict) and "step" in t:
                step_names.add(t["step"])
        assert "DECISION" in step_names
        assert "GATE" in step_names or "FINAL" in step_names

    def test_to_training_record_has_required_keys(self, registry, backends):
        resources = ResourceContext(vram_free_gb=4.0, time_budget_s=30.0)
        engine = ConsultationEngine(registry, backends, resources=resources)
        result = engine.handle("req-16", "Critique this plan.")
        record = result.to_training_record(
            user_prompt="Critique this plan.",
            resource_context={"vram_free_gb": 4.0},
        )
        for key in ("USER", "STATE", "AVAILABLE_EXPERTS", "RESOURCE_STATUS",
                    "MOK_ACTION", "EXPERT_REPLY", "MOK_CHECK", "MOK_FINAL"):
            assert key in record, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# DecisionLoop
# ---------------------------------------------------------------------------

class TestDecisionLoop:
    @pytest.fixture
    def loop(self) -> DecisionLoop:
        return DecisionLoop()

    @pytest.fixture
    def rich_experts(self) -> AvailableExperts:
        return AvailableExperts(
            cheap=["fast_3b"],
            strong=["general_7b"],
            retrieval=["embed_1b"],
            vram_free_gb=12.0,
            time_budget_s=60.0,
        )

    @pytest.fixture
    def cheap_only_experts(self) -> AvailableExperts:
        return AvailableExperts(
            cheap=["fast_3b"],
            strong=[],
            retrieval=[],
            vram_free_gb=3.0,
            time_budget_s=30.0,
        )

    @pytest.fixture
    def no_experts(self) -> AvailableExperts:
        return AvailableExperts(vram_free_gb=0.5, time_budget_s=5.0)

    def test_answer_direct_for_simple_prompt(self, loop, rich_experts):
        task = TaskState(prompt="What is a language model?")
        record = loop.evaluate(task, rich_experts)
        assert record.decision == LoopDecision.ANSWER_DIRECT

    def test_confidence_sufficient_when_threshold_met(self, loop, rich_experts):
        task = TaskState(
            prompt="Analyze this deployment",
            current_confidence=0.9,
            findings_so_far=["finding A"],
        )
        record = loop.evaluate(task, rich_experts)
        assert record.decision == LoopDecision.CONFIDENCE_SUFFICIENT

    def test_report_uncertainty_when_no_experts(self, loop, no_experts):
        task = TaskState(prompt="Complex analysis of a distributed system design.")
        record = loop.evaluate(task, no_experts)
        assert record.decision == LoopDecision.REPORT_UNCERTAINTY

    def test_report_uncertainty_at_iteration_limit(self, loop, rich_experts):
        task = TaskState(prompt="Design a complete microservices architecture.", iteration=3, max_iterations=3)
        record = loop.evaluate(task, rich_experts)
        assert record.decision == LoopDecision.REPORT_UNCERTAINTY

    def test_retrieve_memory_when_referenced(self, loop, rich_experts):
        task = TaskState(prompt="What did we decide about the memory system in the last session?")
        record = loop.evaluate(task, rich_experts)
        assert record.decision == LoopDecision.RETRIEVE_MEMORY

    def test_multi_expert_for_complex_task_with_resources(self, loop, rich_experts):
        task = TaskState(prompt="Compare and evaluate these two architectural trade-offs.")
        record = loop.evaluate(task, rich_experts)
        assert record.decision == LoopDecision.MULTI_EXPERT

    def test_call_strong_for_code_task(self, loop):
        experts = AvailableExperts(
            cheap=["fast_3b"],
            strong=["coder_7b"],
            vram_free_gb=7.0,
            time_budget_s=30.0,
        )
        task = TaskState(prompt="```python\ndef foo():\n    pass\n``` Review this function for bugs.")
        record = loop.evaluate(task, experts)
        assert record.decision == LoopDecision.CALL_STRONG_HELPER

    def test_call_cheap_when_only_cheap_available(self, loop, cheap_only_experts):
        task = TaskState(prompt="Summarize the key points from this text.")
        record = loop.evaluate(task, cheap_only_experts)
        assert record.decision == LoopDecision.CALL_CHEAP_HELPER

    def test_run_loop_terminates(self, loop, rich_experts):
        records = loop.run_loop("Analyse the trade-offs in this distributed system design.", rich_experts)
        assert len(records) >= 1
        assert len(records) <= rich_experts.time_budget_s  # sanity
        # Final decision must be terminal
        final = records[-1].decision
        assert final in (
            LoopDecision.ANSWER_DIRECT,
            LoopDecision.CONFIDENCE_SUFFICIENT,
            LoopDecision.REPORT_UNCERTAINTY,
        )

    def test_run_loop_simple_prompt_single_step(self, loop, rich_experts):
        records = loop.run_loop("What is 2 + 2?", rich_experts)
        assert records[0].decision == LoopDecision.ANSWER_DIRECT
        assert len(records) == 1

    def test_decision_record_to_dict(self, loop, rich_experts):
        task = TaskState(prompt="Summarize this plan.")
        record = loop.evaluate(task, rich_experts)
        d = record.to_dict()
        assert "decision" in d
        assert "reason" in d
        assert "preferred_expert_tier" in d

    def test_retrieval_only_fallback(self, loop):
        experts = AvailableExperts(
            cheap=[],
            strong=[],
            retrieval=["embed_1b"],
            vram_free_gb=0.8,
            time_budget_s=30.0,
        )
        task = TaskState(prompt="Analyse the deep complexity of this multi-expert system.")
        record = loop.evaluate(task, experts)
        assert record.decision == LoopDecision.RETRIEVE_MEMORY