"""
tests/test_flywheel.py

Unit tests for the RSI flywheel subsystem:
  quality_scorer, replay_buffer, trace_accumulator, finetune_trigger
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from mok.rsi.quality_scorer import QualityScoreBreakdown, score as quality_score
from mok.rsi.replay_buffer import BufferRecord, BufferStats, ReplayBuffer
from mok.rsi.trace_accumulator import TraceAccumulator
from mok.rsi.finetune_trigger import FineTuneTrigger


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mock_result(
    gate: str = "expert_answer_accepted_and_synthesized",
    confidence: str = "medium",
    final_answer: str = "[MoK synthesis] Synthesized answer here, not copied.",
    accepted_findings: list[str] | None = None,
):
    """Build a minimal duck-typed ConsultationResult for scoring."""

    class _Turn:
        def __init__(self, quality="good", challenged=False, findings=None):
            self.challenged = challenged
            self.reply = type("R", (), {
                "quality": quality,
                "findings": findings or ["Expert finding 1."],
            })()

    class _Session:
        def __init__(self, turns=None):
            self.turns = turns or [_Turn()]

    class _Result:
        def __init__(self):
            self.gate = gate
            self.confidence = confidence
            self.final_answer = final_answer
            self.accepted_findings = accepted_findings or ["Expert finding 1."]
            self.sessions = [_Session()]
            self.decision = type("D", (), {"value": "call_cheap_helper"})()
            self.trace = [{"step": "DECISION"}, {"step": "FINAL"}]
            self.request_id = "test_req_001"

        def to_training_record(self, user_prompt="", resource_context=None):
            return {
                "USER": user_prompt or "test query",
                "STATE": {"decision": self.decision.value, "gate": self.gate, "confidence": self.confidence},
                "AVAILABLE_EXPERTS": ["fast_3b"],
                "RESOURCE_STATUS": {"vram_free_gb": 4.0, "time_budget_s": 30.0},
                "MOK_ACTION": self.decision.value,
                "EXPERT_REPLY": [{"expert": "fast_3b", "findings": self.accepted_findings}],
                "MOK_CHECK": [{"step": "CHECK", "result": "ok"}],
                "MOK_FINAL": self.final_answer,
                "messages": [],
                "trace": self.trace,
            }

    return _Result()


def _make_record(lane: str = "single_expert_consult", quality: float = 0.75) -> BufferRecord:
    uid = f"{lane}_{int(time.time() * 1000000) % 999999}"
    return BufferRecord(
        record_id=uid,
        timestamp=time.time(),
        quality_score=quality,
        lane=lane,
        USER=f"test query for {lane}",
        STATE={"decision": "call_cheap_helper", "gate": "expert_answer_accepted_and_synthesized", "confidence": "medium"},
        AVAILABLE_EXPERTS=["fast_3b"],
        RESOURCE_STATUS={"vram_free_gb": 4.0, "time_budget_s": 30.0},
        MOK_ACTION="call_cheap_helper",
        EXPERT_REPLY=[{"expert": "fast_3b", "findings": ["finding"], "quality": "good"}],
        MOK_CHECK=[{"step": "CHECK", "result": "ok"}],
        MOK_FINAL="[MoK synthesis] Distinct MoK answer not from expert.",
        trace=[{"step": "DECISION"}, {"step": "FINAL"}],
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
    )


# ---------------------------------------------------------------------------
# QualityScorer
# ---------------------------------------------------------------------------

class TestQualityScorer:
    def test_breakdown_is_dataclass(self):
        b = QualityScoreBreakdown()
        assert b.gate_discipline == 0.0
        assert b.total == 0.0

    def test_good_result_passes_gate(self):
        result = _make_mock_result()
        bd = quality_score(result)
        assert bd.gate_discipline == 1.0

    def test_failed_gate_scores_zero(self):
        result = _make_mock_result(gate="pending")
        bd = quality_score(result)
        assert bd.gate_discipline == 0.0

    def test_no_backend_gate_scores_zero(self):
        result = _make_mock_result(gate="no_backend")
        bd = quality_score(result)
        assert bd.gate_discipline == 0.0

    def test_distinct_final_answer_passes_no_copy(self):
        result = _make_mock_result(
            final_answer="[MoK synthesis] An entirely different answer with different tokens."
        )
        bd = quality_score(result)
        assert bd.no_copy_enforcement >= 0.5

    def test_copied_final_answer_fails_no_copy(self):
        # Final answer is exactly the expert finding
        result = _make_mock_result(
            accepted_findings=["Expert finding 1."],
            final_answer="Expert finding 1.",
        )
        bd = quality_score(result)
        assert bd.no_copy_enforcement < 1.0

    def test_no_vague_turns_full_challenge_credit(self):
        result = _make_mock_result()
        result.sessions[0].turns[0].reply.quality = "good"
        bd = quality_score(result)
        assert bd.challenge_discipline == 1.0

    def test_vague_unchallenged_loses_challenge_credit(self):
        result = _make_mock_result()

        class _Turn:
            challenged = False
            reply = type("R", (), {"quality": "vague", "findings": ["vague finding"]})()

        result.sessions[0].turns = [_Turn()]
        bd = quality_score(result)
        assert bd.challenge_discipline == 0.0

    def test_vague_challenged_earns_challenge_credit(self):
        result = _make_mock_result()

        class _Turn:
            challenged = True
            reply = type("R", (), {"quality": "vague", "findings": ["vague finding"]})()

        result.sessions[0].turns = [_Turn()]
        bd = quality_score(result)
        assert bd.challenge_discipline == 1.0

    def test_medium_confidence_good_gate_passes_calibration(self):
        result = _make_mock_result(
            gate="expert_answer_accepted_and_synthesized", confidence="medium"
        )
        bd = quality_score(result)
        assert bd.confidence_calibration == 1.0

    def test_low_confidence_failed_gate_passes_calibration(self):
        result = _make_mock_result(gate="no_backend", confidence="low")
        bd = quality_score(result)
        assert bd.confidence_calibration == 1.0

    def test_high_confidence_failed_gate_fails_calibration(self):
        result = _make_mock_result(gate="no_backend", confidence="high")
        bd = quality_score(result)
        assert bd.confidence_calibration == 0.0

    def test_is_eligible_at_threshold_0_6(self):
        # 3 of 4 = 0.75 total => eligible
        bd = QualityScoreBreakdown(gate_discipline=1.0, no_copy_enforcement=1.0,
                                   challenge_discipline=1.0, confidence_calibration=0.0)
        assert bd.total == pytest.approx(0.75)
        assert bd.is_eligible(0.6) is True

    def test_is_not_eligible_below_threshold(self):
        bd = QualityScoreBreakdown(gate_discipline=0.0, no_copy_enforcement=0.5,
                                   challenge_discipline=0.0, confidence_calibration=0.0)
        assert bd.is_eligible(0.6) is False

    def test_to_dict_has_all_keys(self):
        bd = quality_score(_make_mock_result())
        d = bd.to_dict()
        for k in ("gate_discipline", "no_copy_enforcement", "challenge_discipline",
                  "confidence_calibration", "total", "eligible"):
            assert k in d


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_write_single_record(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        rec = _make_record()
        result = buf.write(rec)
        assert result is True

    def test_read_all_returns_written(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        ids = []
        for i in range(5):
            r = _make_record()
            r.record_id = f"rall_{i}"
            buf.write(r)
            ids.append(r.record_id)
        records = buf.read_all()
        assert len(records) == 5

    def test_dedup_rejects_same_id(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        r = _make_record()
        r.record_id = "fixed_dedup_id"
        first = buf.write(r)
        second = buf.write(r)
        assert first is True
        assert second is False

    def test_stats_total_after_writes(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        for i in range(3):
            r = _make_record()
            r.record_id = f"stat_{i}"
            buf.write(r)
        s = buf.stats()
        assert s.total_records == 3

    def test_stats_mean_quality(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        for i, q in enumerate([0.8, 0.6, 1.0]):
            r = _make_record(quality=q)
            r.record_id = f"mq_{i}"
            buf.write(r)
        s = buf.stats()
        assert s.mean_quality_score == pytest.approx(0.8, rel=0.05)

    def test_mark_triggered_resets_since_trigger(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        r = _make_record()
        r.record_id = "trig_r"
        buf.write(r)
        buf.mark_triggered()
        new_records = buf.read_since_trigger()
        assert len(new_records) == 0

    def test_write_batch(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        records = []
        for i in range(8):
            r = _make_record()
            r.record_id = f"batch_{i}"
            records.append(r)
        written = buf.write_batch(records)
        assert written == 8

    def test_diversity_flagging(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        # Write 20 records in one lane (dominant)
        dominant = []
        for i in range(20):
            r = _make_record(lane="single_expert_consult")
            r.record_id = f"dom_{i}"
            dominant.append(r)
        buf.write_batch(dominant)
        # Add a minority lane record — should not be flagged
        r = _make_record(lane="expert_challenge")
        r.record_id = "minority_rec"
        buf.write(r)
        all_recs = buf.read_all()
        minority = [x for x in all_recs if x.lane == "expert_challenge"]
        assert len(minority) == 1
        assert not minority[0].diversity_flagged

    def test_from_dict_roundtrip(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        r = _make_record()
        r.record_id = "roundtrip_1"
        buf.write(r)
        all_recs = buf.read_all()
        assert all_recs[0].record_id == "roundtrip_1"
        assert all_recs[0].lane == r.lane


# ---------------------------------------------------------------------------
# TraceAccumulator
# ---------------------------------------------------------------------------

class TestTraceAccumulator:
    def test_accept_high_quality(self, tmp_path):
        acc = TraceAccumulator(pool_dir=tmp_path, quality_threshold=0.0)
        result = _make_mock_result()
        ir = acc.ingest(result, user_id="u1", lane="test")
        assert ir.accepted is True
        assert isinstance(ir.quality_score, float)

    def test_reject_below_threshold(self, tmp_path):
        # Gate=no_backend (gate_discipline=0) + confidence=high (calibration=0) => total <= 0.5
        acc = TraceAccumulator(pool_dir=tmp_path, quality_threshold=0.6)
        result = _make_mock_result(gate="no_backend", confidence="high",
                                   final_answer="[MoK synthesis] distinct answer here")
        ir = acc.ingest(result, user_id="u1", lane="test")
        assert ir.accepted is False

    def test_stats_keys_present(self, tmp_path):
        acc = TraceAccumulator(pool_dir=tmp_path, quality_threshold=0.0)
        result = _make_mock_result()
        acc.ingest(result, user_id="u1", lane="test")
        s = acc.stats()
        assert "accumulator" in s
        assert "buffer" in s
        assert s["accumulator"]["total_ingested"] == 1

    def test_acceptance_rate(self, tmp_path):
        acc = TraceAccumulator(pool_dir=tmp_path, quality_threshold=0.0)
        for i in range(4):
            r = _make_mock_result()
            r.request_id = f"ar_req_{i}"
            acc.ingest(r, user_id="u1", lane="lane_a")
        s = acc.stats()["accumulator"]
        # All accepted (threshold=0) but dedup may reject 2nd+ identical records
        assert s["total_ingested"] == 4
        assert "acceptance_rate" in s

    def test_rejected_record_reason_contains_threshold(self, tmp_path):
        acc = TraceAccumulator(pool_dir=tmp_path, quality_threshold=0.99)
        result = _make_mock_result(gate="no_backend", confidence="high")
        ir = acc.ingest(result, user_id="u1", lane="test")
        assert ir.accepted is False
        assert "threshold" in ir.reason


# ---------------------------------------------------------------------------
# FineTuneTrigger
# ---------------------------------------------------------------------------

class TestFineTuneTrigger:
    def _trigger(self, tmp_path, min_new=5, min_q=0.5, min_lanes=2, cooldown=0.0):
        return FineTuneTrigger(
            pool_dir=tmp_path,
            min_new_records=min_new,
            min_quality_score=min_q,
            min_distinct_lanes=min_lanes,
            min_interval_hours=cooldown,
        )

    def test_no_trigger_empty_buffer(self, tmp_path):
        t = self._trigger(tmp_path)
        result = t.check()
        assert result.should_trigger is False
        assert "insufficient" in result.reason

    def test_triggers_when_conditions_met(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        lanes = ["lane_a", "lane_b", "lane_c"]
        for i in range(15):
            r = _make_record(lane=lanes[i % 3], quality=0.9)
            r.record_id = f"trig_{i}"
            buf.write(r)
        t = self._trigger(tmp_path, min_new=5, min_q=0.7, min_lanes=2)
        result = t.check()
        assert result.should_trigger is True
        assert result.batch_path is not None
        assert result.batch_path.exists()

    def test_no_trigger_low_quality(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        for i in range(10):
            r = _make_record(quality=0.1)
            r.record_id = f"lq_{i}"
            buf.write(r)
        t = self._trigger(tmp_path, min_new=5, min_q=0.7)
        result = t.check()
        assert result.should_trigger is False
        assert "quality" in result.reason

    def test_no_trigger_insufficient_lanes(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        for i in range(10):
            r = _make_record(lane="only_lane", quality=0.9)
            r.record_id = f"il_{i}"
            buf.write(r)
        t = self._trigger(tmp_path, min_new=5, min_q=0.5, min_lanes=3)
        result = t.check()
        assert result.should_trigger is False
        assert "lane" in result.reason

    def test_cooldown_blocks_retrigger(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        for i in range(20):
            r = _make_record(lane=["a", "b", "c"][i % 3], quality=0.9)
            r.record_id = f"cd_{i}"
            buf.write(r)
        buf.mark_triggered()  # simulate recent trigger
        t = self._trigger(tmp_path, min_new=5, min_q=0.5, cooldown=24.0)
        result = t.check()
        assert result.should_trigger is False
        assert "cooldown" in result.reason

    def test_batch_file_schema(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        for i in range(10):
            r = _make_record(lane=["x", "y", "z"][i % 3], quality=0.9)
            r.record_id = f"schema_{i}"
            buf.write(r)
        t = self._trigger(tmp_path, min_new=5, min_q=0.5, min_lanes=2)
        result = t.check()
        assert result.should_trigger
        with open(result.batch_path, encoding="utf-8") as f:
            first = json.loads(f.readline())
        required = ("id", "lane", "dataset", "quality_score", "USER", "MOK_FINAL", "messages")
        for k in required:
            assert k in first, f"Missing field: {k}"

    def test_trigger_marks_buffer(self, tmp_path):
        buf = ReplayBuffer(tmp_path)
        for i in range(10):
            r = _make_record(lane=["p", "q"][i % 2], quality=0.9)
            r.record_id = f"mark_{i}"
            buf.write(r)
        t = self._trigger(tmp_path, min_new=5, min_q=0.5, min_lanes=2)
        t.check()
        # After trigger, since_trigger should be empty
        new_recs = buf.read_since_trigger()
        assert len(new_recs) == 0
