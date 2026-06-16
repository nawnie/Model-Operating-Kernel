"""
tests/test_rsi.py

Unit tests for mok.routing.rsi — Routing Signal Integrator.
All tests are fully offline; no real JSONL files required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mok.routing.rsi import (
    KeywordSignal,
    RSIReport,
    RoutingSignalIntegrator,
    _tokenise,
)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def test_tokenise_removes_stop_words():
    tokens = _tokenise("please help me fix this python function")
    assert "please" not in tokens
    assert "help" not in tokens
    assert "python" in tokens
    assert "function" in tokens


def test_tokenise_min_length():
    tokens = _tokenise("a an ok do is are the code")
    # single / two-char words and stop words stripped
    assert "code" in tokens
    assert "ok" not in tokens   # 2 chars
    assert "an" not in tokens


def test_tokenise_case_fold():
    tokens = _tokenise("Python FUNCTION traceback")
    assert "python" in tokens
    assert "function" in tokens
    assert "traceback" in tokens


# ---------------------------------------------------------------------------
# Helpers — build a JSONL trace file in tmp_path
# ---------------------------------------------------------------------------

def _write_traces(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _code_trace(request_id: str = "req-1", confidence: float = 0.84,
                success: bool = True) -> dict:
    return {
        "request_id": request_id,
        "prompt": "write a python function to sort a list",
        "route_expert": "coder",
        "route_confidence": confidence,
        "route_reason": "code keyword match",
        "router_tier": "R0",
        "success": success,
        "error_type": None,
    }


def _vision_trace(request_id: str = "req-2") -> dict:
    return {
        "request_id": request_id,
        "prompt": "describe this screenshot of the dashboard diagram",
        "route_expert": "vision",
        "route_confidence": 0.95,
        "route_reason": "image modality flag",
        "router_tier": "R0",
        "success": True,
        "error_type": None,
    }


def _error_trace(request_id: str = "req-e") -> dict:
    return {
        "request_id": request_id,
        "prompt": "do something",
        "route_expert": "unknown",
        "route_confidence": 0.0,
        "route_reason": "routing_failed",
        "router_tier": "R0",
        "success": False,
        "error_type": "routing_error",
    }


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def test_ingest_log_returns_count(tmp_path: Path) -> None:
    log = tmp_path / "trace.jsonl"
    _write_traces(log, [_code_trace(), _vision_trace()])
    rsi = RoutingSignalIntegrator()
    count = rsi.ingest_log(log)
    assert count == 2


def test_ingest_log_missing_file_returns_zero(tmp_path: Path) -> None:
    rsi = RoutingSignalIntegrator()
    count = rsi.ingest_log(tmp_path / "nonexistent.jsonl")
    assert count == 0


def test_ingest_log_skips_malformed_lines(tmp_path: Path) -> None:
    log = tmp_path / "trace.jsonl"
    log.write_text('{"bad": true}\nnot json at all\n' + json.dumps(_code_trace()) + "\n",
                   encoding="utf-8")
    rsi = RoutingSignalIntegrator()
    count = rsi.ingest_log(log)
    # The valid line + the bad-but-parseable {"bad":true} — both parse; only
    # the valid one produces a full TraceRecord via defaults.  Just ensure no crash.
    assert count >= 1


def test_clear_resets_records(tmp_path: Path) -> None:
    log = tmp_path / "trace.jsonl"
    _write_traces(log, [_code_trace()])
    rsi = RoutingSignalIntegrator()
    rsi.ingest_log(log)
    rsi.clear()
    report = rsi.report()
    assert report.total_traces == 0


# ---------------------------------------------------------------------------
# Report basics
# ---------------------------------------------------------------------------

def test_report_total_traces(tmp_path: Path) -> None:
    log = tmp_path / "trace.jsonl"
    _write_traces(log, [_code_trace(), _vision_trace(), _error_trace()])
    rsi = RoutingSignalIntegrator()
    rsi.ingest_log(log)
    report = rsi.report()
    assert report.total_traces == 3


def test_report_success_rate(tmp_path: Path) -> None:
    log = tmp_path / "trace.jsonl"
    _write_traces(log, [_code_trace(success=True), _error_trace()])
    rsi = RoutingSignalIntegrator()
    rsi.ingest_log(log)
    report = rsi.report()
    assert report.success_rate == pytest.approx(0.5)


def test_report_expert_call_counts(tmp_path: Path) -> None:
    log = tmp_path / "trace.jsonl"
    traces = [_code_trace(f"r{i}") for i in range(3)] + [_vision_trace()]
    _write_traces(log, traces)
    rsi = RoutingSignalIntegrator()
    rsi.ingest_log(log)
    report = rsi.report()
    assert report.expert_call_counts["coder"] == 3
    assert report.expert_call_counts["vision"] == 1


def test_report_error_breakdown(tmp_path: Path) -> None:
    log = tmp_path / "trace.jsonl"
    _write_traces(log, [_error_trace("e1"), _error_trace("e2")])
    rsi = RoutingSignalIntegrator()
    rsi.ingest_log(log)
    report = rsi.report()
    assert report.error_breakdown.get("routing_error", 0) == 2


def test_report_tier_distribution(tmp_path: Path) -> None:
    log = tmp_path / "trace.jsonl"
    _write_traces(log, [_code_trace(), _vision_trace()])
    rsi = RoutingSignalIntegrator()
    rsi.ingest_log(log)
    report = rsi.report()
    assert report.tier_distribution.get("R0", 0) == 2


# ---------------------------------------------------------------------------
# Keyword signal extraction
# ---------------------------------------------------------------------------

def _many_code_traces(tmp_path: Path, n: int = 6) -> Path:
    """Write n code-expert traces that share the token 'python'."""
    log = tmp_path / "trace.jsonl"
    traces = [
        {
            "request_id": f"req-{i}",
            "prompt": f"python function sort list example {i}",
            "route_expert": "coder",
            "route_confidence": 0.90,
            "route_reason": "code keyword match",
            "router_tier": "R0",
            "success": True,
            "error_type": None,
        }
        for i in range(n)
    ]
    _write_traces(log, traces)
    return log


def test_signals_extracted_for_frequent_token(tmp_path: Path) -> None:
    log = _many_code_traces(tmp_path, n=6)
    rsi = RoutingSignalIntegrator()
    rsi.ingest_log(log)
    report = rsi.report()
    tokens = [s.token for s in report.top_signals]
    assert "python" in tokens, f"Expected 'python' in signals; got {tokens}"


def test_signals_respect_min_occurrences(tmp_path: Path) -> None:
    """Token appearing only twice should not produce a signal (MIN=3)."""
    log = tmp_path / "trace.jsonl"
    traces = [
        {
            "request_id": f"req-{i}",
            "prompt": f"raretoken{i} sort list",
            "route_expert": "coder",
            "route_confidence": 0.90,
            "route_reason": "code match",
            "router_tier": "R0",
            "success": True,
            "error_type": None,
        }
        for i in range(2)
    ]
    _write_traces(log, traces)
    rsi = RoutingSignalIntegrator()
    rsi.ingest_log(log)
    report = rsi.report()
    # "raretoken0" / "raretoken1" are unique per trace → each appears once
    tokens = [s.token for s in report.top_signals]
    for t in tokens:
        assert not t.startswith("raretoken"), f"Rare token {t} leaked into signals"


def test_signals_ignored_when_low_confidence(tmp_path: Path) -> None:
    log = tmp_path / "trace.jsonl"
    traces = [
        {
            "request_id": f"req-{i}",
            "prompt": "python function example",
            "route_expert": "coder",
            "route_confidence": 0.50,   # below MIN_CONFIDENCE 0.75
            "route_reason": "code match",
            "router_tier": "R0",
            "success": True,
            "error_type": None,
        }
        for i in range(5)
    ]
    _write_traces(log, traces)
    rsi = RoutingSignalIntegrator()
    rsi.ingest_log(log)
    report = rsi.report()
    assert report.top_signals == []


# ---------------------------------------------------------------------------
# keyword_patches
# ---------------------------------------------------------------------------

def test_keyword_patches_returns_dict(tmp_path: Path) -> None:
    log = _many_code_traces(tmp_path, n=6)
    rsi = RoutingSignalIntegrator()
    rsi.ingest_log(log)
    patches = rsi.keyword_patches()
    assert isinstance(patches, dict)
    if "coder" in patches:
        assert "python" in patches["coder"]


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def test_report_to_markdown_contains_key_sections(tmp_path: Path) -> None:
    log = _many_code_traces(tmp_path, n=6)
    rsi = RoutingSignalIntegrator()
    rsi.ingest_log(log)
    md = rsi.report().to_markdown()
    assert "# RSI Report" in md
    assert "Expert call distribution" in md
    assert "coder" in md


def test_report_to_markdown_no_crash_on_empty() -> None:
    rsi = RoutingSignalIntegrator()
    md = rsi.report().to_markdown()
    assert "# RSI Report" in md
    assert "0" in md   # total_traces = 0
