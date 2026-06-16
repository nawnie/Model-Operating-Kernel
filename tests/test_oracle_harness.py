"""
tests/test_oracle_harness.py

Tests for the OracleHarness, rouge_l, and ScorerProtocol additions to
mok.evaluation.oracle (P4.3).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mok.evaluation.oracle import (
    OracleExample,
    OracleHarness,
    ScorerProtocol,
    _lcs_length,
    rouge_l,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_traces(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _trace(
    request_id: str = "r1",
    expert: str = "coder",
    reference: str = "print hello world",
    response: str = "print('hello world')",
) -> dict:
    return {
        "request_id": request_id,
        "route_expert": expert,
        "reference": reference,
        "expert_response": response,
    }


# ---------------------------------------------------------------------------
# _lcs_length
# ---------------------------------------------------------------------------

class TestLcsLength:
    def test_identical_sequences(self):
        assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_empty_sequences(self):
        assert _lcs_length([], ["a", "b"]) == 0
        assert _lcs_length(["a"], []) == 0

    def test_no_overlap(self):
        assert _lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_partial_overlap(self):
        assert _lcs_length(["a", "b", "c"], ["a", "x", "c"]) == 2


# ---------------------------------------------------------------------------
# rouge_l
# ---------------------------------------------------------------------------

class TestRougeL:
    def test_identical_strings_score_one(self):
        assert rouge_l("hello world", "hello world") == pytest.approx(1.0)

    def test_empty_hypothesis_scores_zero(self):
        assert rouge_l("", "hello world") == pytest.approx(0.0)

    def test_empty_reference_scores_zero(self):
        assert rouge_l("hello world", "") == pytest.approx(0.0)

    def test_no_overlap_scores_zero(self):
        assert rouge_l("foo bar", "baz qux") == pytest.approx(0.0)

    def test_partial_overlap_between_zero_and_one(self):
        score = rouge_l("the cat sat", "the dog sat")
        assert 0.0 < score < 1.0

    def test_case_insensitive(self):
        assert rouge_l("Hello World", "hello world") == pytest.approx(1.0)

    def test_symmetry_approximate(self):
        # ROUGE-L is not perfectly symmetric but should be close for similar lengths
        s1 = rouge_l("the quick brown fox", "the quick fox")
        s2 = rouge_l("the quick fox", "the quick brown fox")
        assert abs(s1 - s2) < 0.2


# ---------------------------------------------------------------------------
# ScorerProtocol
# ---------------------------------------------------------------------------

class TestScorerProtocol:
    def test_default_scorer_uses_rouge_l(self):
        scorer = ScorerProtocol()
        assert scorer.score("hello world", "hello world") == pytest.approx(1.0)

    def test_custom_scorer_used_when_provided(self):
        def always_one(r, ref):
            return 1.0

        harness = OracleHarness(scorer=always_one)
        assert harness.score_response("anything", "anything else") == pytest.approx(1.0)

    def test_callable_protocol_accepted(self):
        scorer = ScorerProtocol()
        harness = OracleHarness(scorer=scorer)
        score = harness.score_response("hello", "hello")
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# OracleHarness.score_response
# ---------------------------------------------------------------------------

class TestScoreResponse:
    def test_perfect_match(self):
        h = OracleHarness()
        assert h.score_response("def sort(x): return sorted(x)", "def sort(x): return sorted(x)") == pytest.approx(1.0)

    def test_empty_response_zero(self):
        h = OracleHarness()
        assert h.score_response("", "some reference") == pytest.approx(0.0)

    def test_partial_match_between_zero_and_one(self):
        h = OracleHarness()
        score = h.score_response("return sorted list", "return a sorted list of items")
        assert 0.0 < score < 1.0

    def test_score_in_unit_interval(self):
        h = OracleHarness()
        score = h.score_response("hello world foo bar", "baz qux hello")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# OracleHarness.evaluate_batch
# ---------------------------------------------------------------------------

class TestEvaluateBatch:
    def test_basic_evaluation(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        _write_traces(traces, [_trace("r1", response="hello world", reference="hello world")])
        h = OracleHarness()
        result = h.evaluate_batch(traces)
        assert result["count"] == 1
        assert "mean_regret" in result
        assert "oracle_match_rate" in result

    def test_multiple_traces(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        _write_traces(traces, [_trace(f"r{i}") for i in range(5)])
        h = OracleHarness()
        result = h.evaluate_batch(traces)
        assert result["count"] == 5

    def test_per_expert_in_result(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        _write_traces(traces, [_trace("r1", expert="coder")])
        h = OracleHarness()
        result = h.evaluate_batch(traces)
        assert "coder" in result["per_expert"]

    def test_examples_list_in_result(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        _write_traces(traces, [_trace("r1")])
        result = OracleHarness().evaluate_batch(traces)
        assert len(result["examples"]) == 1
        assert isinstance(result["examples"][0], OracleExample)

    def test_skips_trace_with_no_response(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        _write_traces(traces, [
            {"request_id": "r1", "route_expert": "coder", "reference": "hi"},  # no expert_response
            _trace("r2"),
        ])
        result = OracleHarness().evaluate_batch(traces)
        assert result["count"] == 1

    def test_skips_trace_with_no_request_id(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        _write_traces(traces, [
            {"route_expert": "coder", "reference": "hi", "expert_response": "hi"},  # no rid
            _trace("r2"),
        ])
        result = OracleHarness().evaluate_batch(traces)
        assert result["count"] == 1

    def test_expert_responses_override(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        _write_traces(traces, [
            {"request_id": "r1", "route_expert": "coder", "reference": "hello world"},
        ])
        expert_responses = {"r1": {"coder": "hello world", "general": "greetings"}}
        result = OracleHarness().evaluate_batch(traces, expert_responses=expert_responses)
        assert result["count"] == 1
        assert "coder" in result["per_expert"]
        assert "general" in result["per_expert"]

    def test_count_zero_on_empty_file(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        traces.write_text("", encoding="utf-8")
        result = OracleHarness().evaluate_batch(traces)
        assert result["count"] == 0
        assert result["mean_regret"] == pytest.approx(0.0)

    def test_skips_malformed_lines(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        with traces.open("w") as fh:
            fh.write("not json\n")
            fh.write(json.dumps(_trace("r1")) + "\n")
        result = OracleHarness().evaluate_batch(traces)
        assert result["count"] == 1

    def test_oracle_match_rate_one_when_always_routed_to_best(self, tmp_path: Path):
        # When there's only one expert, it's always the oracle expert
        traces = tmp_path / "t.jsonl"
        _write_traces(traces, [_trace("r1", expert="coder"), _trace("r2", expert="coder")])
        result = OracleHarness().evaluate_batch(traces)
        assert result["oracle_match_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# OracleHarness.write_oracle_scores
# ---------------------------------------------------------------------------

class TestWriteOracleScores:
    def test_writes_jsonl(self, tmp_path: Path):
        out = tmp_path / "scores.jsonl"
        examples = [
            OracleExample("r1", "coder", {"coder": 0.9, "general": 0.6}),
            OracleExample("r2", "general", {"coder": 0.4, "general": 0.8}),
        ]
        n = OracleHarness().write_oracle_scores(examples, out)
        assert n == 2
        assert out.exists()

    def test_output_parseable(self, tmp_path: Path):
        out = tmp_path / "scores.jsonl"
        examples = [OracleExample("r1", "coder", {"coder": 0.9})]
        OracleHarness().write_oracle_scores(examples, out)
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert lines[0]["request_id"] == "r1"
        assert lines[0]["expert_scores"]["coder"] == pytest.approx(0.9)

    def test_no_overwrite_skips_existing(self, tmp_path: Path):
        out = tmp_path / "scores.jsonl"
        examples = [OracleExample("r1", "coder", {"coder": 0.9})]
        OracleHarness().write_oracle_scores(examples, out, overwrite=True)
        n2 = OracleHarness().write_oracle_scores(examples, out, overwrite=False)
        assert n2 == 0

    def test_creates_parent_dirs(self, tmp_path: Path):
        out = tmp_path / "deep" / "nested" / "scores.jsonl"
        OracleHarness().write_oracle_scores([], out)
        assert out.exists()

    def test_empty_examples_writes_empty_file(self, tmp_path: Path):
        out = tmp_path / "scores.jsonl"
        n = OracleHarness().write_oracle_scores([], out)
        assert n == 0
        assert out.read_text() == ""

    def test_end_to_end_with_export(self, tmp_path: Path):
        """write_oracle_scores output is loadable by export_training_pairs."""
        from mok.evaluation.export import export_training_pairs
        # Build traces JSONL
        traces = tmp_path / "traces.jsonl"
        _write_traces(traces, [_trace("r1", expert="coder")])
        # Build oracle scores via harness
        oracle_out = tmp_path / "oracle.jsonl"
        examples = [OracleExample("r1", "coder", {"coder": 0.85})]
        OracleHarness().write_oracle_scores(examples, oracle_out)
        # Export should succeed
        pairs_out = tmp_path / "pairs.jsonl"
        n = export_training_pairs(traces, oracle_out, pairs_out)
        assert n == 1
