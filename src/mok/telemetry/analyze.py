"""
src/mok/telemetry/analyze.py

Stdlib trace analysis (P2.5).

Reads JSONL trace files and produces a structured summary:
  - Route distribution (which expert was chosen, how often)
  - Mean / p50 / p95 latency per expert and overall
  - Success rate and error breakdown
  - Mean confidence per router tier
  - Regret proxy: fraction of low-confidence routes (< 0.70)
  - RSI signal: which token patterns dominate each expert's traffic

No external dependencies — pure stdlib.

Usage
-----
    from mok.telemetry.analyze import analyze_traces, print_report

    report = analyze_traces(Path("traces/runtime.jsonl"))
    print_report(report)

    # Or from the CLI:
    python -m mok --analyze-traces traces/runtime.jsonl
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ExpertStats:
    expert: str
    call_count: int = 0
    success_count: int = 0
    latencies_ms: list[int] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.call_count if self.call_count else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return _mean(self.latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        return _percentile(self.latencies_ms, 95)

    @property
    def mean_confidence(self) -> float:
        return _mean(self.confidences)


@dataclass
class TraceReport:
    source_path: str
    total_traces: int = 0
    success_count: int = 0
    # Per-expert breakdowns
    expert_stats: dict[str, ExpertStats] = field(default_factory=dict)
    # Tier distribution
    tier_counts: dict[str, int] = field(default_factory=dict)
    # Error breakdown
    error_counts: dict[str, int] = field(default_factory=dict)
    # Overall latency
    all_latencies_ms: list[int] = field(default_factory=list)
    # Low-confidence routes (regret proxy)
    low_confidence_count: int = 0
    low_confidence_threshold: float = 0.70
    # Eviction events
    total_evictions: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_traces if self.total_traces else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return _mean(self.all_latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        return _percentile(self.all_latencies_ms, 95)

    @property
    def regret_rate(self) -> float:
        """Fraction of successful routes with confidence < threshold."""
        denom = self.success_count or 1
        return self.low_confidence_count / denom

    def top_experts(self, n: int = 5) -> list[ExpertStats]:
        return sorted(
            self.expert_stats.values(),
            key=lambda s: -s.call_count,
        )[:n]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_traces(
    jsonl_path: Path,
    *,
    low_confidence_threshold: float = 0.70,
) -> TraceReport:
    """
    Parse a JSONL trace file and return a TraceReport.

    Skips blank lines and malformed JSON silently.
    Returns an empty report (total_traces=0) if the file does not exist.
    """
    report = TraceReport(
        source_path=str(jsonl_path),
        low_confidence_threshold=low_confidence_threshold,
    )

    if not jsonl_path.exists():
        return report

    for obj in _iter_jsonl(jsonl_path):
        report.total_traces += 1

        expert     = str(obj.get("route_expert", "unknown"))
        success    = bool(obj.get("success", True))
        latency    = int(obj.get("total_ms", 0))
        confidence = float(obj.get("route_confidence", 0.0))
        tier       = str(obj.get("router_tier", "R0"))
        error_type = obj.get("error_type") or None
        evicted    = obj.get("evicted", [])

        if success:
            report.success_count += 1
        if error_type:
            report.error_counts[error_type] = report.error_counts.get(error_type, 0) + 1

        report.all_latencies_ms.append(latency)
        report.tier_counts[tier] = report.tier_counts.get(tier, 0) + 1
        report.total_evictions += len(evicted) if isinstance(evicted, list) else 0

        if success and confidence < low_confidence_threshold:
            report.low_confidence_count += 1

        # Per-expert
        if expert not in report.expert_stats:
            report.expert_stats[expert] = ExpertStats(expert=expert)
        es = report.expert_stats[expert]
        es.call_count += 1
        if success:
            es.success_count += 1
        es.latencies_ms.append(latency)
        es.confidences.append(confidence)

    return report


def analyze_many(trace_dir: Path, **kwargs) -> list[TraceReport]:
    """Analyze all *.jsonl files in trace_dir. Returns one report per file."""
    return [
        analyze_traces(p, **kwargs)
        for p in sorted(trace_dir.glob("*.jsonl"))
    ]


# ---------------------------------------------------------------------------
# Text report formatter
# ---------------------------------------------------------------------------

def format_report(report: TraceReport) -> str:
    """
    Render a TraceReport as a plain-text summary table.
    Suitable for printing to stdout or writing to a .txt file.
    """
    lines: list[str] = []
    w = lines.append

    w("=" * 60)
    w("MoK Trace Analysis")
    w(f"Source : {report.source_path}")
    w("=" * 60)

    w(f"\nTotal traces    : {report.total_traces}")
    w(f"Success rate    : {report.success_rate:.1%}")
    w(f"Mean latency    : {report.mean_latency_ms:.0f} ms")
    w(f"p95 latency     : {report.p95_latency_ms:.0f} ms")
    w(f"Regret proxy    : {report.regret_rate:.1%}  "
      f"(confidence < {report.low_confidence_threshold:.0%})")
    w(f"Total evictions : {report.total_evictions}")

    if report.tier_counts:
        w("\nRouter tier distribution")
        w("-" * 30)
        for tier, count in sorted(report.tier_counts.items()):
            pct = count / max(report.total_traces, 1)
            w(f"  {tier:<8} {count:>5}  ({pct:.1%})")

    if report.expert_stats:
        w("\nExpert call distribution")
        w("-" * 60)
        header = f"  {'Expert':<18} {'Calls':>6}  {'%':>5}  {'Succ%':>6}  {'AvgMs':>6}  {'p95Ms':>6}  {'AvgConf':>7}"
        w(header)
        w("  " + "-" * 56)
        for es in report.top_experts(n=20):
            pct = es.call_count / max(report.total_traces, 1)
            w(
                f"  {es.expert:<18} {es.call_count:>6}  {pct:>4.1%}  "
                f"{es.success_rate:>5.1%}  {es.mean_latency_ms:>6.0f}  "
                f"{es.p95_latency_ms:>6.0f}  {es.mean_confidence:>7.2f}"
            )

    if report.error_counts:
        w("\nError breakdown")
        w("-" * 30)
        for etype, count in sorted(report.error_counts.items(), key=lambda x: -x[1]):
            w(f"  {etype:<30} {count:>5}")

    w("\n" + "=" * 60)
    return "\n".join(lines)


def print_report(report: TraceReport) -> None:
    print(format_report(report))


# ---------------------------------------------------------------------------
# CSV export (optional — complements compact.py which works on raw JSONL)
# ---------------------------------------------------------------------------

def write_summary_csv(report: TraceReport, path: Path) -> None:
    """
    Write per-expert stats to a CSV for further analysis.
    Columns: expert, calls, success_rate, mean_ms, p95_ms, mean_confidence
    """
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["expert", "calls", "success_rate",
                         "mean_ms", "p95_ms", "mean_confidence"])
        for es in sorted(report.expert_stats.values(), key=lambda s: -s.call_count):
            writer.writerow([
                es.expert,
                es.call_count,
                round(es.success_rate, 4),
                round(es.mean_latency_ms, 1),
                round(es.p95_latency_ms, 1),
                round(es.mean_confidence, 4),
            ])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: list, pct: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, math.ceil(len(sorted_vals) * pct / 100) - 1)
    return float(sorted_vals[idx])


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
