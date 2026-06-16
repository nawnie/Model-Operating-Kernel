"""
tests/test_telemetry.py

Tests for mok.telemetry.compact and mok.telemetry.analyze.
All offline — uses tmp_path for disk I/O.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mok.telemetry.compact import (
    _COLUMNS,
    compact_all,
    compact_traces,
)
from mok.telemetry.analyze import (
    ExpertStats,
    TraceReport,
    analyze_traces,
    analyze_many,
    format_report,
    write_summary_csv,
    _mean,
    _percentile,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _trace(
    *,
    request_id: str = "req-1",
    expert: str = "coder",
    success: bool = True,
    confidence: float = 0.84,
    total_ms: int = 120,
    backend_latency_ms: int = 100,
    vram_pressure_gb: float = 2.5,
    router_tier: str = "R0",
    error_type: str | None = None,
    evicted: list | None = None,
    prompt: str = "write a python function",
    prompt_tokens: int = 8,
    fallback_chain: list | None = None,
) -> dict:
    return {
        "request_id": request_id,
        "prompt": prompt,
        "route_expert": expert,
        "route_confidence": confidence,
        "route_reason": "code keyword match",
        "router_tier": router_tier,
        "success": success,
        "error_type": error_type,
        "total_ms": total_ms,
        "backend_latency_ms": backend_latency_ms,
        "vram_pressure_gb": vram_pressure_gb,
        "prompt_tokens": prompt_tokens,
        "experts_called": [expert],
        "evicted": evicted or [],
        "fallback_chain": fallback_chain or [],
        "modality_flags": {},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# ===========================================================================
# compact.py
# ===========================================================================

class TestCompactTraces:
    def test_produces_csv(self, tmp_path: Path):
        src = tmp_path / "runtime.jsonl"
        _write_jsonl(src, [_trace(request_id="r1"), _trace(request_id="r2")])
        out = compact_traces(src, tmp_path)
        assert out.exists()
        assert out.suffix == ".csv"

    def test_csv_has_correct_header(self, tmp_path: Path):
        src = tmp_path / "runtime.jsonl"
        _write_jsonl(src, [_trace()])
        out = compact_traces(src, tmp_path)
        with out.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert set(reader.fieldnames) == set(_COLUMNS)

    def test_csv_row_count_matches_input(self, tmp_path: Path):
        src = tmp_path / "runtime.jsonl"
        _write_jsonl(src, [_trace(request_id=f"r{i}") for i in range(10)])
        out = compact_traces(src, tmp_path)
        with out.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 10

    def test_csv_values_correct(self, tmp_path: Path):
        src = tmp_path / "runtime.jsonl"
        _write_jsonl(src, [_trace(
            request_id="req-42", expert="vision", success=False,
            confidence=0.95, total_ms=200, error_type="backend_error",
            evicted=["coder"], fallback_chain=["general"],
        )])
        out = compact_traces(src, tmp_path)
        with out.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        row = rows[0]
        assert row["request_id"] == "req-42"
        assert row["route_expert"] == "vision"
        assert row["success"] == "0"
        assert row["error_type"] == "backend_error"
        assert row["evicted"] == "coder"
        assert row["fallback_chain"] == "general"
        assert row["ts_index"] == "0"

    def test_prompt_excerpt_truncated(self, tmp_path: Path):
        long_prompt = "x" * 500
        src = tmp_path / "runtime.jsonl"
        _write_jsonl(src, [_trace(prompt=long_prompt)])
        out = compact_traces(src, tmp_path)
        with out.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows[0]["prompt_excerpt"]) == 120

    def test_skips_malformed_lines(self, tmp_path: Path):
        src = tmp_path / "runtime.jsonl"
        src.write_text(
            "not json at all\n" + json.dumps(_trace()) + "\n\n",
            encoding="utf-8",
        )
        out = compact_traces(src, tmp_path)
        with out.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            compact_traces(tmp_path / "ghost.jsonl", tmp_path)

    def test_no_overwrite_returns_existing(self, tmp_path: Path):
        src = tmp_path / "runtime.jsonl"
        _write_jsonl(src, [_trace()])
        out1 = compact_traces(src, tmp_path, overwrite=True)
        mtime1 = out1.stat().st_mtime
        out2 = compact_traces(src, tmp_path, overwrite=False)
        assert out2.stat().st_mtime == mtime1

    def test_default_output_dir_is_same_as_input(self, tmp_path: Path):
        src = tmp_path / "runtime.jsonl"
        _write_jsonl(src, [_trace()])
        out = compact_traces(src)
        assert out.parent == tmp_path

    def test_compact_all_handles_multiple_files(self, tmp_path: Path):
        for name in ["a.jsonl", "b.jsonl", "c.jsonl"]:
            _write_jsonl(tmp_path / name, [_trace()])
        results = compact_all(tmp_path, tmp_path)
        assert len(results) == 3
        for p in results:
            assert p.suffix == ".csv"


# ===========================================================================
# analyze.py — helpers
# ===========================================================================

class TestHelpers:
    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_values(self):
        assert _mean([1, 2, 3]) == pytest.approx(2.0)

    def test_percentile_empty(self):
        assert _percentile([], 95) == 0.0

    def test_percentile_p50(self):
        values = list(range(1, 101))   # 1..100
        assert _percentile(values, 50) == 50.0

    def test_percentile_p95(self):
        values = list(range(1, 101))
        assert _percentile(values, 95) == 95.0

    def test_percentile_single(self):
        assert _percentile([42], 95) == 42.0


# ===========================================================================
# analyze.py — analyze_traces
# ===========================================================================

class TestAnalyzeTraces:
    def test_empty_file(self, tmp_path: Path):
        src = tmp_path / "empty.jsonl"
        src.write_text("", encoding="utf-8")
        report = analyze_traces(src)
        assert report.total_traces == 0
        assert report.success_rate == 0.0

    def test_missing_file_returns_empty_report(self, tmp_path: Path):
        report = analyze_traces(tmp_path / "ghost.jsonl")
        assert report.total_traces == 0

    def test_total_traces(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        _write_jsonl(src, [_trace(request_id=f"r{i}") for i in range(7)])
        report = analyze_traces(src)
        assert report.total_traces == 7

    def test_success_rate(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        _write_jsonl(src, [
            _trace(success=True),
            _trace(success=True),
            _trace(success=False, error_type="backend_error"),
        ])
        report = analyze_traces(src)
        assert report.success_rate == pytest.approx(2 / 3)

    def test_expert_call_counts(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        traces = (
            [_trace(expert="coder")] * 4 +
            [_trace(expert="vision")] * 2
        )
        _write_jsonl(src, traces)
        report = analyze_traces(src)
        assert report.expert_stats["coder"].call_count == 4
        assert report.expert_stats["vision"].call_count == 2

    def test_mean_latency(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        _write_jsonl(src, [
            _trace(total_ms=100),
            _trace(total_ms=200),
            _trace(total_ms=300),
        ])
        report = analyze_traces(src)
        assert report.mean_latency_ms == pytest.approx(200.0)

    def test_p95_latency(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        latencies = list(range(100, 200))   # 100 values: 100..199
        _write_jsonl(src, [_trace(total_ms=ms) for ms in latencies])
        report = analyze_traces(src)
        assert report.p95_latency_ms >= 194

    def test_error_breakdown(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        _write_jsonl(src, [
            _trace(success=False, error_type="backend_error"),
            _trace(success=False, error_type="backend_error"),
            _trace(success=False, error_type="routing_error"),
        ])
        report = analyze_traces(src)
        assert report.error_counts["backend_error"] == 2
        assert report.error_counts["routing_error"] == 1

    def test_tier_distribution(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        _write_jsonl(src, [
            _trace(router_tier="R0"),
            _trace(router_tier="R0"),
            _trace(router_tier="R1"),
        ])
        report = analyze_traces(src)
        assert report.tier_counts["R0"] == 2
        assert report.tier_counts["R1"] == 1

    def test_regret_rate_low_confidence(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        # 2 high-confidence, 1 low-confidence (below 0.70)
        _write_jsonl(src, [
            _trace(confidence=0.90, success=True),
            _trace(confidence=0.85, success=True),
            _trace(confidence=0.55, success=True),
        ])
        report = analyze_traces(src)
        assert report.low_confidence_count == 1
        assert report.regret_rate == pytest.approx(1 / 3)

    def test_failed_routes_not_counted_for_regret(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        _write_jsonl(src, [
            _trace(confidence=0.50, success=False, error_type="routing_error"),
        ])
        report = analyze_traces(src)
        assert report.low_confidence_count == 0

    def test_eviction_count(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        _write_jsonl(src, [
            _trace(evicted=["coder", "vision"]),
            _trace(evicted=["general"]),
            _trace(evicted=[]),
        ])
        report = analyze_traces(src)
        assert report.total_evictions == 3

    def test_expert_success_rate(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        _write_jsonl(src, [
            _trace(expert="coder", success=True),
            _trace(expert="coder", success=True),
            _trace(expert="coder", success=False, error_type="backend_error"),
        ])
        report = analyze_traces(src)
        es = report.expert_stats["coder"]
        assert es.success_rate == pytest.approx(2 / 3)

    def test_skips_malformed_lines(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        src.write_text("bad json\n" + json.dumps(_trace()) + "\n", encoding="utf-8")
        report = analyze_traces(src)
        assert report.total_traces == 1

    def test_top_experts_sorted_by_count(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        traces = (
            [_trace(expert="general")] * 5 +
            [_trace(expert="coder")] * 3 +
            [_trace(expert="vision")] * 1
        )
        _write_jsonl(src, traces)
        report = analyze_traces(src)
        top = report.top_experts(n=2)
        assert top[0].expert == "general"
        assert top[1].expert == "coder"


class TestFormatReport:
    def test_format_report_contains_key_sections(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        _write_jsonl(src, [_trace(expert="coder"), _trace(expert="vision")])
        report = analyze_traces(src)
        text = format_report(report)
        assert "MoK Trace Analysis" in text
        assert "Total traces" in text
        assert "Expert call distribution" in text
        assert "coder" in text

    def test_format_empty_report(self, tmp_path: Path):
        report = analyze_traces(tmp_path / "ghost.jsonl")
        text = format_report(report)
        assert "MoK Trace Analysis" in text
        assert "0" in text


class TestAnalyzeMany:
    def test_analyze_many(self, tmp_path: Path):
        for name in ["a.jsonl", "b.jsonl"]:
            _write_jsonl(tmp_path / name, [_trace(), _trace()])
        reports = analyze_many(tmp_path)
        assert len(reports) == 2
        for r in reports:
            assert r.total_traces == 2


class TestWriteSummaryCSV:
    def test_writes_csv(self, tmp_path: Path):
        src = tmp_path / "t.jsonl"
        _write_jsonl(src, [_trace(expert="coder"), _trace(expert="vision")])
        report = analyze_traces(src)
        out = tmp_path / "summary.csv"
        write_summary_csv(report, out)
        assert out.exists()
        with out.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2
        experts = {r["expert"] for r in rows}
        assert "coder" in experts
        assert "vision" in experts
