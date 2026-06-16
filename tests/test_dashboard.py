"""
tests/test_dashboard.py

Tests for mok.telemetry.dashboard (P6.2).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mok.telemetry.dashboard import build_report, render_report


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
    tier: str = "R0",
    total_ms: int = 100,
    success: bool = True,
) -> dict:
    return {
        "request_id": request_id,
        "route_expert": expert,
        "router_tier": tier,
        "total_ms": total_ms,
        "success": success,
    }


def _oracle(request_id: str, expert: str, score: float = 0.9) -> dict:
    return {
        "request_id": request_id,
        "expert_scores": {expert: score, "general": 0.5},
    }


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_raises_on_missing_trace(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            build_report(tmp_path / "ghost.jsonl")

    def test_empty_trace_file(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        t.write_text("", encoding="utf-8")
        stats = build_report(t)
        assert stats["total_traces"] == 0

    def test_single_trace(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, [_trace("r1", "coder")])
        stats = build_report(t)
        assert stats["total_traces"] == 1
        assert "coder" in stats["per_expert"]

    def test_per_expert_percentage(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, [
            _trace("r1", "coder"),
            _trace("r2", "coder"),
            _trace("r3", "vision"),
            _trace("r4", "vision"),
        ])
        stats = build_report(t)
        assert stats["per_expert"]["coder"]["pct"] == pytest.approx(50.0)
        assert stats["per_expert"]["vision"]["pct"] == pytest.approx(50.0)

    def test_tier_distribution(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, [
            _trace("r1", tier="R0"),
            _trace("r2", tier="R0"),
            _trace("r3", tier="R1"),
        ])
        stats = build_report(t)
        assert "R0" in stats["tier_distribution"]
        assert stats["tier_distribution"]["R0"] == pytest.approx(66.7, abs=0.2)

    def test_mean_latency(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, [
            _trace("r1", total_ms=100),
            _trace("r2", total_ms=200),
            _trace("r3", total_ms=300),
        ])
        stats = build_report(t)
        assert stats["mean_latency_ms"] == 200

    def test_routing_errors_counted(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, [
            _trace("r1", success=True),
            _trace("r2", success=False),
        ])
        stats = build_report(t)
        assert stats["routing_errors"] == 1

    def test_oracle_match_rate_computed(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        o = tmp_path / "o.jsonl"
        _write_traces(t, [
            _trace("r1", "coder"),
            _trace("r2", "coder"),
        ])
        with o.open("w") as fh:
            # r1: coder is oracle (score 0.9 vs general 0.5) → match
            fh.write(json.dumps({"request_id": "r1", "expert_scores": {"coder": 0.9, "general": 0.5}}) + "\n")
            # r2: coder not oracle (general scores higher) → mismatch
            fh.write(json.dumps({"request_id": "r2", "expert_scores": {"coder": 0.4, "general": 0.9}}) + "\n")
        stats = build_report(t, oracle_scores_path=o)
        assert stats["oracle_match_rate"] == pytest.approx(0.5)

    def test_last_n_limits_traces(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, [_trace(f"r{i}", "coder") for i in range(10)])
        stats = build_report(t, last_n=5)
        assert stats["total_traces"] == 5

    def test_no_oracle_path_returns_none_match_rate(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, [_trace("r1")])
        stats = build_report(t)
        assert stats["oracle_match_rate"] is None

    def test_skips_malformed_trace_lines(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        with t.open("w") as fh:
            fh.write("not json\n")
            fh.write(json.dumps(_trace("r1")) + "\n")
        stats = build_report(t)
        assert stats["total_traces"] == 1

    def test_per_expert_mean_latency(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, [
            _trace("r1", "coder", total_ms=200),
            _trace("r2", "coder", total_ms=400),
        ])
        stats = build_report(t)
        assert stats["per_expert"]["coder"]["mean_latency_ms"] == 300


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------

class TestRenderReport:
    def test_returns_string(self):
        stats = {
            "total_traces": 0, "last_n": 1000,
            "per_expert": {}, "tier_distribution": {},
            "oracle_match_rate": None, "mean_regret": None,
            "mean_latency_ms": 0, "p95_latency_ms": 0, "routing_errors": 0,
        }
        assert isinstance(render_report(stats), str)

    def test_contains_expert_names(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, [_trace("r1", "coder"), _trace("r2", "vision")])
        stats = build_report(t)
        report = render_report(stats)
        assert "coder" in report
        assert "vision" in report

    def test_contains_latency(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, [_trace("r1", total_ms=250)])
        stats = build_report(t)
        report = render_report(stats)
        assert "250ms" in report or "250" in report

    def test_contains_tier_distribution(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, [_trace("r1", tier="R0"), _trace("r2", tier="R1")])
        stats = build_report(t)
        report = render_report(stats)
        assert "R0" in report
        assert "R1" in report

    def test_routing_errors_shown(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, [_trace("r1", success=False)])
        stats = build_report(t)
        report = render_report(stats)
        assert "Routing errors: 1" in report

    def test_oracle_match_rate_shown_when_available(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        o = tmp_path / "o.jsonl"
        _write_traces(t, [_trace("r1", "coder")])
        with o.open("w") as fh:
            fh.write(json.dumps({"request_id": "r1", "expert_scores": {"coder": 0.9}}) + "\n")
        stats = build_report(t, oracle_scores_path=o)
        report = render_report(stats)
        assert "oracle_match_rate" in report

    def test_handles_run_eval_stats_shape(self):
        """render_report also accepts stats from run_eval.run_eval()."""
        from mok.evaluation.oracle import OracleExample
        run_eval_stats = {
            "count": 5.0,
            "mean_regret": 0.05,
            "oracle_match_rate": 0.80,
            "routing_errors": 0,
            "mean_latency_ms": 200,
            "p95_latency_ms": 350,
            "examples": [],
        }
        report = render_report(run_eval_stats)
        assert isinstance(report, str)
        assert "0.8000" in report or "0.80" in report


# ---------------------------------------------------------------------------
# Integration: run_eval end-to-end
# ---------------------------------------------------------------------------

class TestRunEval:
    def test_run_eval_produces_stats(self, config_path: Path):
        from evaluation.run_eval import run_eval
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))

        stats = run_eval(
            config_path=config_path,
            prompts_path=Path(__file__).resolve().parent.parent / "evaluation" / "prompts.jsonl",
            oracle_path=Path(__file__).resolve().parent.parent / "evaluation" / "oracle_labels.jsonl",
        )
        assert isinstance(stats["count"], (int, float))
        assert "routing_errors" in stats
