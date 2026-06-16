"""
src/mok/telemetry/dashboard.py

Text-mode regret dashboard (P6.2).

Reads a runtime trace JSONL and optional oracle scores JSONL and prints
a human-readable report to stdout.  No external dependencies.

Report format
-------------
    Route Distribution (last N traces)
      coder:    42%  mean_regret=0.03  mean_latency=312ms
      instruct: 31%  mean_regret=0.08  mean_latency=425ms
      vision:   18%  mean_regret=0.12  mean_latency=890ms
      core:      9%  (coordinator, not evaluated)

    Router tier distribution
      R0: 74%   R1: 18%   R2: 8%

    Overall oracle_match_rate: 0.87
    Mean latency: 412ms (p95: 890ms)
    Circuit breaker trips: 0
    Routing errors: 0

Usage
-----
    from mok.telemetry.dashboard import build_report, render_report

    stats = build_report(
        trace_path=Path("traces/runtime.jsonl"),
        oracle_scores_path=Path("traces/oracle_scores.jsonl"),  # optional
        last_n=1000,
    )
    print(render_report(stats))
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * pct)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

def build_report(
    trace_path: Path,
    oracle_scores_path: Path | None = None,
    last_n: int = 1000,
) -> dict[str, Any]:
    """
    Build a report dict from trace and optional oracle score files.

    Parameters
    ----------
    trace_path          : runtime JSONL trace file
    oracle_scores_path  : optional oracle scores JSONL
                          ({"request_id": ..., "expert_scores": {...}})
    last_n              : only consider the last N traces

    Returns
    -------
    dict with keys used by render_report().
    """
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    # Load (up to) last_n traces
    all_traces = list(_iter_jsonl(trace_path))
    traces = all_traces[-last_n:] if len(all_traces) > last_n else all_traces

    # Load oracle scores
    oracle: dict[str, dict[str, float]] = {}
    if oracle_scores_path and oracle_scores_path.exists():
        for obj in _iter_jsonl(oracle_scores_path):
            rid = obj.get("request_id", "")
            scores = obj.get("expert_scores", {})
            if rid and isinstance(scores, dict):
                oracle[rid] = {k: float(v) for k, v in scores.items()}

    # Accumulate per-expert stats
    expert_counts: dict[str, int] = defaultdict(int)
    expert_latencies: dict[str, list[int]] = defaultdict(list)
    expert_regrets: dict[str, list[float]] = defaultdict(list)
    tier_counts: dict[str, int] = defaultdict(int)
    oracle_matches = 0
    oracle_total = 0
    routing_errors = 0
    all_latencies: list[int] = []

    for trace in traces:
        expert = trace.get("route_expert", "unknown")
        ms = int(trace.get("total_ms", 0))
        tier = str(trace.get("router_tier", "R0"))
        success = trace.get("success", True)

        expert_counts[expert] += 1
        expert_latencies[expert].append(ms)
        all_latencies.append(ms)
        tier_counts[tier] += 1

        if not success:
            routing_errors += 1

        rid = trace.get("request_id", "")
        if rid in oracle:
            scores = oracle[rid]
            if expert in scores:
                oracle_total += 1
                expert_score = scores[expert]
                oracle_score = max(scores.values())
                regret = oracle_score - expert_score
                expert_regrets[expert].append(regret)
                if expert == max(scores, key=scores.__getitem__):
                    oracle_matches += 1

    total = len(traces)
    all_latencies.sort()

    per_expert = {}
    for exp, cnt in expert_counts.items():
        lat = sorted(expert_latencies[exp])
        regrets = expert_regrets.get(exp, [])
        per_expert[exp] = {
            "count": cnt,
            "pct": round(cnt / total * 100, 1) if total else 0.0,
            "mean_regret": round(mean(regrets), 4) if regrets else None,
            "mean_latency_ms": int(mean(lat)) if lat else 0,
        }

    tier_dist = {
        tier: round(cnt / total * 100, 1) if total else 0.0
        for tier, cnt in tier_counts.items()
    }

    return {
        "total_traces": total,
        "last_n": last_n,
        "per_expert": per_expert,
        "tier_distribution": tier_dist,
        "oracle_match_rate": round(oracle_matches / oracle_total, 4) if oracle_total else None,
        "mean_regret": round(
            mean([r for rs in expert_regrets.values() for r in rs]), 4
        ) if any(expert_regrets.values()) else None,
        "mean_latency_ms": int(mean(all_latencies)) if all_latencies else 0,
        "p95_latency_ms": int(_percentile(all_latencies, 0.95)),
        "routing_errors": routing_errors,
    }


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------

def render_report(stats: dict[str, Any]) -> str:
    """
    Render a build_report() dict as a human-readable text report.

    Also accepts a dict returned directly by run_eval.run_eval() — the
    function tolerates missing keys gracefully.
    """
    lines: list[str] = []
    total = stats.get("total_traces", stats.get("count", 0))
    last_n = stats.get("last_n", total)

    lines.append(f"Route Distribution (last {min(total, last_n)} traces)")

    per_expert = stats.get("per_expert", {})
    for exp, info in sorted(per_expert.items(), key=lambda x: -x[1]["count"]):
        pct = info["pct"]
        mean_lat = info["mean_latency_ms"]
        regret = info.get("mean_regret")
        regret_str = f"  mean_regret={regret:.4f}" if regret is not None else "  (not evaluated)"
        lat_str = f"  mean_latency={mean_lat}ms"
        lines.append(f"  {exp:<12} {pct:5.1f}%{regret_str}{lat_str}")

    lines.append("")

    tier_dist = stats.get("tier_distribution", {})
    if tier_dist:
        tier_parts = "   ".join(
            f"{t}: {pct:.1f}%" for t, pct in sorted(tier_dist.items())
        )
        lines.append(f"Router tier distribution")
        lines.append(f"  {tier_parts}")
        lines.append("")

    oracle_rate = stats.get("oracle_match_rate")
    mean_regret = stats.get("mean_regret")
    if oracle_rate is not None:
        lines.append(f"Overall oracle_match_rate: {oracle_rate:.4f}")
    if mean_regret is not None:
        lines.append(f"Overall mean_regret:       {mean_regret:.4f}")

    mean_ms = stats.get("mean_latency_ms", 0)
    p95_ms = stats.get("p95_latency_ms", 0)
    lines.append(f"Mean latency: {mean_ms}ms (p95: {p95_ms}ms)")

    errors = stats.get("routing_errors", 0)
    lines.append(f"Routing errors: {errors}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="MoK regret dashboard")
    parser.add_argument("trace", help="Path to runtime JSONL trace file")
    parser.add_argument("--oracle", default=None, help="Path to oracle scores JSONL")
    parser.add_argument("--last-n", type=int, default=1000)
    args = parser.parse_args()

    stats = build_report(
        trace_path=Path(args.trace),
        oracle_scores_path=Path(args.oracle) if args.oracle else None,
        last_n=args.last_n,
    )
    print(render_report(stats))


if __name__ == "__main__":
    main()
