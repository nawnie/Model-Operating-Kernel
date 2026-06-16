"""
tests/test_persona.py

Unit tests for the persona subsystem:
  mok.persona.user_profile — UserProfile tracking + properties
  mok.persona.persona_adapter — PersonaAdapter behavioral adjustments
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mok.persona.user_profile import UserProfile, _infer_task_type
from mok.persona.persona_adapter import (
    PersonaAdapter,
    PersonaAdjustments,
    HIGH_CORRECTION_RATE,
    LOW_CORRECTION_RATE,
    FAST_ANSWER_AVG_TURNS,
    DEEP_ANSWER_AVG_TURNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_result(
    expert_name: str = "fast_3b",
    num_turns: int = 1,
    turns_quality: str = "good",
    gate: str = "expert_answer_accepted_and_synthesized",
):
    class _Turn:
        def __init__(self):
            self.challenged = (turns_quality in ("vague", "overconfident"))
            self.reply = type("R", (), {"quality": turns_quality, "findings": []})()

    class _Expert:
        name = expert_name

    class _Session:
        def __init__(self):
            self.expert = _Expert()
            self.turns = [_Turn() for _ in range(num_turns)]

    class _Result:
        def __init__(self):
            self.gate = gate
            self.sessions = [_Session()]
            self.confidence = "medium"
            self.final_answer = "[MoK synthesis] MoK answer."
            self.accepted_findings = []

    return _Result()


def _build_profile(
    sessions: int = 10,
    correction_count: int = 0,
    preferred_experts: dict | None = None,
    avg_turns: float = 1.0,
    gate_fails: dict | None = None,
    task_counts: dict | None = None,
) -> UserProfile:
    p = UserProfile(user_id="test_user")
    p.sessions = sessions
    p.correction_count = correction_count
    p.preferred_experts = preferred_experts or {"fast_3b": 8, "general_7b": 2}
    p.avg_consultation_turns = avg_turns
    p.gate_fail_patterns = gate_fails or {}
    p.task_type_counts = task_counts or {"code": 5, "general": 5}
    return p


# ---------------------------------------------------------------------------
# UserProfile — task inference
# ---------------------------------------------------------------------------

class TestTaskInference:
    def test_code_prompt(self):
        assert _infer_task_type("write a python function") == "code"

    def test_planning_prompt(self):
        assert _infer_task_type("design a roadmap for this service") == "planning"

    def test_research_prompt(self):
        assert _infer_task_type("explain what transformers are") == "research"

    def test_review_prompt(self):
        assert _infer_task_type("review this pull request") == "review"

    def test_creative_prompt(self):
        assert _infer_task_type("write a short story about robots") == "creative"

    def test_general_fallback(self):
        assert _infer_task_type("xyz undefined jargon") == "general"


# ---------------------------------------------------------------------------
# UserProfile — update_from_result
# ---------------------------------------------------------------------------

class TestUserProfileUpdate:
    def test_session_count_increments(self):
        p = UserProfile(user_id="u1")
        result = _make_mock_result()
        p.update_from_result(result, prompt="write a python sort", quality_score=0.8)
        assert p.sessions == 1

    def test_task_type_recorded(self):
        p = UserProfile(user_id="u1")
        result = _make_mock_result()
        p.update_from_result(result, prompt="design a roadmap strategy", quality_score=0.7)
        assert "planning" in p.task_type_counts

    def test_preferred_expert_increments_on_good_turn(self):
        p = UserProfile(user_id="u1")
        result = _make_mock_result(expert_name="fast_3b", turns_quality="good")
        p.update_from_result(result, prompt="test", quality_score=0.9)
        assert p.preferred_experts.get("fast_3b", 0) >= 1

    def test_avg_consultation_turns_rolling_average(self):
        p = UserProfile(user_id="u1")
        for i in range(4):
            result = _make_mock_result(num_turns=2)
            p.update_from_result(result, prompt="test", quality_score=0.8)
        assert p.avg_consultation_turns == pytest.approx(2.0, abs=0.01)

    def test_avg_quality_score_rolling_average(self):
        p = UserProfile(user_id="u1")
        scores = [0.8, 0.6, 1.0]
        for i, q in enumerate(scores):
            result = _make_mock_result()
            p.update_from_result(result, prompt="test", quality_score=q)
        expected = sum(scores) / len(scores)
        assert p.avg_quality_score == pytest.approx(expected, rel=0.05)

    def test_gate_failure_recorded(self):
        p = UserProfile(user_id="u1")
        result = _make_mock_result(gate="no_backend")
        p.update_from_result(result, prompt="test", quality_score=0.3)
        assert "no_backend" in p.gate_fail_patterns

    def test_good_gate_not_recorded_as_failure(self):
        p = UserProfile(user_id="u1")
        result = _make_mock_result(gate="expert_answer_accepted_and_synthesized")
        p.update_from_result(result, prompt="test", quality_score=0.9)
        assert len(p.gate_fail_patterns) == 0

    def test_first_seen_set_on_first_update(self):
        p = UserProfile(user_id="u1")
        result = _make_mock_result()
        p.update_from_result(result, prompt="test", quality_score=0.8)
        assert p.first_seen != ""

    def test_mark_corrected_increments(self):
        p = UserProfile(user_id="u1")
        p.mark_corrected()
        p.mark_corrected()
        assert p.correction_count == 2


# ---------------------------------------------------------------------------
# UserProfile — computed properties
# ---------------------------------------------------------------------------

class TestUserProfileProperties:
    def test_correction_rate_zero_sessions(self):
        p = UserProfile(user_id="u1")
        assert p.correction_rate == 0.0

    def test_correction_rate_calculation(self):
        p = _build_profile(sessions=10, correction_count=2)
        assert p.correction_rate == pytest.approx(0.2)

    def test_dominant_task_type(self):
        p = _build_profile(task_counts={"code": 8, "research": 2})
        assert p.dominant_task_type == "code"

    def test_dominant_task_type_default(self):
        p = UserProfile(user_id="u1")
        assert p.dominant_task_type == "general"

    def test_top_expert(self):
        p = _build_profile(preferred_experts={"fast_3b": 10, "general_7b": 3})
        assert p.top_expert == "fast_3b"

    def test_top_expert_none_when_empty(self):
        p = UserProfile(user_id="u1")
        assert p.top_expert is None

    def test_prefers_fast_answers_true_when_low_turns(self):
        p = _build_profile(avg_turns=1.0)
        assert p.prefers_fast_answers is True

    def test_prefers_fast_answers_false_when_high_turns(self):
        p = _build_profile(avg_turns=3.0)
        assert p.prefers_fast_answers is False

    def test_top_gate_failure_none_when_empty(self):
        p = UserProfile(user_id="u1")
        assert p.top_gate_failure is None

    def test_top_gate_failure_returns_most_common(self):
        p = _build_profile(gate_fails={"no_backend": 5, "pending": 2})
        assert p.top_gate_failure == "no_backend"


# ---------------------------------------------------------------------------
# UserProfile — persistence
# ---------------------------------------------------------------------------

class TestUserProfilePersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        p = _build_profile(sessions=7, correction_count=1)
        p.user_id = "persist_user"
        p.save(profile_dir=tmp_path)
        loaded = UserProfile.load("persist_user", profile_dir=tmp_path)
        assert loaded.sessions == 7
        assert loaded.correction_count == 1
        assert loaded.user_id == "persist_user"

    def test_load_missing_returns_fresh(self, tmp_path):
        p = UserProfile.load("nonexistent_user", profile_dir=tmp_path)
        assert p.sessions == 0
        assert p.user_id == "nonexistent_user"

    def test_to_dict_has_all_fields(self):
        p = _build_profile()
        d = p.to_dict()
        for k in ("user_id", "sessions", "correction_count", "preferred_experts",
                  "avg_consultation_turns", "avg_quality_score"):
            assert k in d


# ---------------------------------------------------------------------------
# PersonaAdjustments
# ---------------------------------------------------------------------------

class TestPersonaAdjustments:
    def test_defaults_are_neutral(self):
        adj = PersonaAdjustments()
        assert adj.expert_tier_bias == 0.0
        assert adj.challenge_threshold_multiplier == 1.0
        assert adj.max_consultation_turns is None
        assert adj.confidence_threshold_delta == 0.0
        assert adj.preferred_expert_hint is None

    def test_describe_no_adjustments(self):
        adj = PersonaAdjustments()
        assert adj.describe() == "no adjustments"

    def test_describe_with_bias(self):
        adj = PersonaAdjustments(expert_tier_bias=0.5)
        desc = adj.describe()
        assert "cheap" in desc

    def test_describe_strong_bias(self):
        adj = PersonaAdjustments(expert_tier_bias=-0.5)
        desc = adj.describe()
        assert "strong" in desc


# ---------------------------------------------------------------------------
# PersonaAdapter
# ---------------------------------------------------------------------------

class TestPersonaAdapter:
    def test_insufficient_history_returns_defaults(self):
        p = UserProfile(user_id="new_user")
        p.sessions = 3  # below minimum (5)
        adapter = PersonaAdapter(p)
        adj = adapter.compute_adjustments("test prompt")
        assert adj.expert_tier_bias == 0.0
        assert adj.max_consultation_turns is None

    def test_cheap_expert_bias_for_fast_3b_top_expert(self):
        p = _build_profile(sessions=10, preferred_experts={"fast_3b": 9, "general_7b": 1})
        adapter = PersonaAdapter(p)
        adj = adapter.compute_adjustments("test")
        assert adj.expert_tier_bias > 0.0  # bias toward cheap

    def test_strong_expert_bias_for_general_7b_top_expert(self):
        p = _build_profile(sessions=10, preferred_experts={"general_7b": 9, "fast_3b": 1})
        adapter = PersonaAdapter(p)
        adj = adapter.compute_adjustments("test")
        assert adj.expert_tier_bias < 0.0  # bias toward strong

    def test_high_correction_rate_lowers_challenge_multiplier(self):
        # HIGH_CORRECTION_RATE = 0.15
        p = _build_profile(sessions=10, correction_count=3)  # 30% correction rate
        adapter = PersonaAdapter(p)
        adj = adapter.compute_adjustments("test")
        assert adj.challenge_threshold_multiplier < 1.0

    def test_low_correction_rate_raises_challenge_multiplier(self):
        # LOW_CORRECTION_RATE = 0.05 and sessions >= 20
        p = _build_profile(sessions=25, correction_count=1)  # 4% correction rate
        adapter = PersonaAdapter(p)
        adj = adapter.compute_adjustments("test")
        assert adj.challenge_threshold_multiplier >= 1.0

    def test_fast_answer_preference_limits_turns(self):
        p = _build_profile(sessions=10, avg_turns=1.0)  # prefers_fast_answers = True
        adapter = PersonaAdapter(p)
        adj = adapter.compute_adjustments("test")
        assert adj.max_consultation_turns == 1

    def test_deep_answer_preference_raises_turns(self):
        p = _build_profile(sessions=10, avg_turns=3.5)  # above DEEP_ANSWER_AVG_TURNS
        adapter = PersonaAdapter(p)
        adj = adapter.compute_adjustments("test")
        assert adj.max_consultation_turns == 4

    def test_high_gate_fail_rate_raises_confidence_delta(self):
        # Gate fail rate > 0.20
        p = _build_profile(sessions=10, gate_fails={"no_backend": 5, "pending": 3})
        adapter = PersonaAdapter(p)
        adj = adapter.compute_adjustments("test")
        assert adj.confidence_threshold_delta > 0.0

    def test_preferred_expert_hint_set_to_top_expert(self):
        p = _build_profile(sessions=10, preferred_experts={"fast_3b": 8, "general_7b": 2})
        adapter = PersonaAdapter(p)
        adj = adapter.compute_adjustments("test")
        assert adj.preferred_expert_hint == "fast_3b"

    def test_dominant_task_type_propagated(self):
        p = _build_profile(sessions=10, task_counts={"code": 8, "general": 2})
        adapter = PersonaAdapter(p)
        adj = adapter.compute_adjustments("test")
        assert adj.dominant_task_type == "code"

    def test_summary_has_expected_keys(self):
        p = _build_profile(sessions=10)
        adapter = PersonaAdapter(p)
        s = adapter.summary()
        for k in ("user_id", "sessions", "correction_rate", "dominant_task_type",
                  "top_expert", "current_adjustments"):
            assert k in s
