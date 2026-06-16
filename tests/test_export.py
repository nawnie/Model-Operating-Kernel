"""
tests/test_export.py

Tests for mok.evaluation.export (R2 training pair exporter).
All offline — uses tmp_path for disk I/O.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mok.evaluation.export import (
    export_training_pairs,
    export_training_pairs_csv,
    load_training_pairs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trace(
    request_id: str = "r1",
    expert: str = "coder",
    confidence: float = 0.84,
    tier: str = "R0",
    prompt: str = "write a sort function",
) -> dict:
    return {
        "request_id": request_id,
        "prompt": prompt,
        "route_expert": expert,
        "route_confidence": confidence,
        "route_reason": "code keyword match",
        "router_tier": tier,
        "success": True,
        "modality_flags": {"has_image": False},
    }


def _oracle(request_id: str = "r1", expert_scores: dict | None = None) -> dict:
    if expert_scores is None:
        expert_scores = {"coder": 0.92, "general": 0.61}
    return {"request_id": request_id, "expert_scores": expert_scores}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Basic join
# ---------------------------------------------------------------------------

class TestExportTrainingPairs:
    def test_basic_join(self, tmp_path: Path):
        traces = tmp_path / "traces.jsonl"
        oracle = tmp_path / "oracle.jsonl"
        out    = tmp_path / "pairs.jsonl"
        _write_jsonl(traces, [_trace("r1", "coder")])
        _write_jsonl(oracle, [_oracle("r1", {"coder": 0.92})])
        n = export_training_pairs(traces, oracle, out)
        assert n == 1
        pairs = load_training_pairs(out)
        assert pairs[0]["request_id"] == "r1"
        assert pairs[0]["expert"] == "coder"
        assert pairs[0]["oracle_score"] == pytest.approx(0.92)

    def test_multiple_traces(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.jsonl"
        _write_jsonl(traces, [_trace(f"r{i}", "coder") for i in range(5)])
        _write_jsonl(oracle, [_oracle(f"r{i}", {"coder": 0.9}) for i in range(5)])
        n = export_training_pairs(traces, oracle, out)
        assert n == 5

    def test_skips_trace_with_no_oracle_entry(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.jsonl"
        _write_jsonl(traces, [_trace("r1"), _trace("r2")])
        _write_jsonl(oracle, [_oracle("r1", {"coder": 0.9})])  # r2 missing
        n = export_training_pairs(traces, oracle, out)
        assert n == 1

    def test_skips_trace_when_expert_not_in_oracle_scores(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.jsonl"
        _write_jsonl(traces, [_trace("r1", expert="coder")])
        _write_jsonl(oracle, [_oracle("r1", {"vision": 0.9})])  # coder score missing
        n = export_training_pairs(traces, oracle, out)
        assert n == 0

    def test_output_fields_complete(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.jsonl"
        _write_jsonl(traces, [_trace("r1", "coder", confidence=0.84, tier="R0")])
        _write_jsonl(oracle, [_oracle("r1", {"coder": 0.92})])
        export_training_pairs(traces, oracle, out)
        pair = load_training_pairs(out)[0]
        assert "request_id" in pair
        assert "prompt" in pair
        assert "modality" in pair
        assert "expert" in pair
        assert "oracle_score" in pair
        assert "router_tier" in pair
        assert "confidence" in pair
        assert "route_reason" in pair

    def test_router_tier_preserved(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.jsonl"
        _write_jsonl(traces, [_trace("r1", tier="R1")])
        _write_jsonl(oracle, [_oracle("r1", {"coder": 0.80})])
        export_training_pairs(traces, oracle, out)
        assert load_training_pairs(out)[0]["router_tier"] == "R1"

    def test_prompt_truncated_to_1000_chars(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.jsonl"
        _write_jsonl(traces, [_trace("r1", prompt="x" * 2000)])
        _write_jsonl(oracle, [_oracle("r1", {"coder": 0.9})])
        export_training_pairs(traces, oracle, out)
        assert len(load_training_pairs(out)[0]["prompt"]) == 1000

    def test_min_oracle_score_filter(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.jsonl"
        _write_jsonl(traces, [_trace("r1"), _trace("r2")])
        _write_jsonl(oracle, [
            _oracle("r1", {"coder": 0.90}),
            _oracle("r2", {"coder": 0.30}),  # below threshold
        ])
        n = export_training_pairs(traces, oracle, out, min_oracle_score=0.70)
        assert n == 1

    def test_max_pairs_cap(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.jsonl"
        _write_jsonl(traces, [_trace(f"r{i}") for i in range(10)])
        _write_jsonl(oracle, [_oracle(f"r{i}", {"coder": 0.9}) for i in range(10)])
        n = export_training_pairs(traces, oracle, out, max_pairs=3)
        assert n == 3

    def test_no_overwrite_skips_existing(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.jsonl"
        _write_jsonl(traces, [_trace("r1")])
        _write_jsonl(oracle, [_oracle("r1", {"coder": 0.9})])
        export_training_pairs(traces, oracle, out, overwrite=True)
        n2 = export_training_pairs(traces, oracle, out, overwrite=False)
        assert n2 == 0

    def test_raises_on_missing_trace_file(self, tmp_path: Path):
        oracle = tmp_path / "o.jsonl"
        oracle.write_text("{}\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            export_training_pairs(tmp_path / "ghost.jsonl", oracle, tmp_path / "out.jsonl")

    def test_raises_on_missing_oracle_file(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        traces.write_text("{}\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            export_training_pairs(traces, tmp_path / "ghost.jsonl", tmp_path / "out.jsonl")

    def test_skips_malformed_trace_lines(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.jsonl"
        traces.write_text("not json\n" + json.dumps(_trace("r1")) + "\n", encoding="utf-8")
        _write_jsonl(oracle, [_oracle("r1", {"coder": 0.9})])
        n = export_training_pairs(traces, oracle, out)
        assert n == 1

    def test_empty_trace_file(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.jsonl"
        traces.write_text("", encoding="utf-8")
        _write_jsonl(oracle, [_oracle()])
        n = export_training_pairs(traces, oracle, out)
        assert n == 0

    def test_creates_output_parent_dirs(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "deep" / "nested" / "pairs.jsonl"
        _write_jsonl(traces, [_trace("r1")])
        _write_jsonl(oracle, [_oracle("r1", {"coder": 0.9})])
        export_training_pairs(traces, oracle, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

class TestExportTrainingPairsCSV:
    def test_writes_csv_with_header(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.csv"
        _write_jsonl(traces, [_trace("r1")])
        _write_jsonl(oracle, [_oracle("r1", {"coder": 0.9})])
        n = export_training_pairs_csv(traces, oracle, out)
        assert n == 1
        with out.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert "expert" in rows[0]
        assert "oracle_score" in rows[0]

    def test_empty_traces_writes_header_only(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.csv"
        traces.write_text("", encoding="utf-8")
        oracle.write_text("", encoding="utf-8")
        export_training_pairs_csv(traces, oracle, out)
        with out.open(encoding="utf-8") as fh:
            content = fh.read()
        assert "request_id" in content


# ---------------------------------------------------------------------------
# load_training_pairs
# ---------------------------------------------------------------------------

class TestLoadTrainingPairs:
    def test_load_returns_list(self, tmp_path: Path):
        traces = tmp_path / "t.jsonl"
        oracle = tmp_path / "o.jsonl"
        out    = tmp_path / "p.jsonl"
        _write_jsonl(traces, [_trace("r1"), _trace("r2")])
        _write_jsonl(oracle, [
            _oracle("r1", {"coder": 0.9}),
            _oracle("r2", {"coder": 0.7}),
        ])
        export_training_pairs(traces, oracle, out)
        pairs = load_training_pairs(out)
        assert len(pairs) == 2

    def test_load_returns_empty_on_missing_file(self, tmp_path: Path):
        result = load_training_pairs(tmp_path / "ghost.jsonl")
        assert result == []
