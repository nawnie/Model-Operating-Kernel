"""
src/mok/telemetry/compact.py

Trace compaction — JSONL → CSV (P1.5).

Reads a runtime trace JSONL file (one TraceEvent per line) and writes a
compact CSV that any analytics tool can query (DuckDB, pandas, Excel,
sqlite3's .import, etc.).

No external dependencies — pure stdlib (json, csv, pathlib).

Usage
-----
    from mok.telemetry.compact import compact_traces

    out = compact_traces(
        jsonl_path=Path("traces/runtime.jsonl"),
        output_dir=Path("traces/compact/"),
    )
    # out → Path("traces/compact/runtime.csv")

CLI
---
    python -m mok --compact-traces traces/runtime.jsonl

Column schema (matches TraceEvent v1)
--------------------------------------
    request_id, prompt_excerpt, route_expert, route_confidence, route_reason,
    router_tier, experts_called, evicted, success, error_type,
    total_ms, backend_latency_ms, vram_pressure_gb,
    prompt_tokens, fallback_chain, ts_index

    prompt_excerpt  = first 120 chars of prompt (full prompt omitted for size)
    experts_called  = pipe-joined list  e.g. "coder|vision"
    evicted         = pipe-joined list
    fallback_chain  = pipe-joined list
    ts_index        = 0-based line number (proxy for time when no wall clock)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Column definition — order matters for CSV header
# ---------------------------------------------------------------------------

_COLUMNS = [
    "ts_index",
    "request_id",
    "success",
    "error_type",
    "route_expert",
    "route_confidence",
    "route_reason",
    "router_tier",
    "total_ms",
    "backend_latency_ms",
    "vram_pressure_gb",
    "prompt_tokens",
    "experts_called",
    "evicted",
    "fallback_chain",
    "prompt_excerpt",
]

_PROMPT_EXCERPT_LEN = 120


def _pipe(value) -> str:
    """Encode a list as a pipe-joined string. Empty list → ''."""
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    return str(value) if value is not None else ""


def _row_from_event(idx: int, obj: dict) -> dict:
    """Project one raw JSONL object onto the CSV column schema."""
    prompt = str(obj.get("prompt", ""))
    return {
        "ts_index":           idx,
        "request_id":         obj.get("request_id", ""),
        "success":            int(bool(obj.get("success", True))),
        "error_type":         obj.get("error_type") or "",
        "route_expert":       obj.get("route_expert", ""),
        "route_confidence":   obj.get("route_confidence", 0.0),
        "route_reason":       obj.get("route_reason", ""),
        "router_tier":        obj.get("router_tier", "R0"),
        "total_ms":           obj.get("total_ms", 0),
        "backend_latency_ms": obj.get("backend_latency_ms", 0),
        "vram_pressure_gb":   obj.get("vram_pressure_gb", 0.0),
        "prompt_tokens":      obj.get("prompt_tokens", 0),
        "experts_called":     _pipe(obj.get("experts_called", [])),
        "evicted":            _pipe(obj.get("evicted", [])),
        "fallback_chain":     _pipe(obj.get("fallback_chain", [])),
        "prompt_excerpt":     prompt[:_PROMPT_EXCERPT_LEN],
    }


def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield parsed objects from a JSONL file; skip blank/malformed lines."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def compact_traces(
    jsonl_path: Path,
    output_dir: Path | None = None,
    *,
    overwrite: bool = True,
) -> Path:
    """
    Compact a JSONL trace file into a CSV.

    Parameters
    ----------
    jsonl_path  : path to the source .jsonl file
    output_dir  : directory for the output CSV; defaults to same dir as input
    overwrite   : if False and the CSV already exists, return its path unchanged

    Returns
    -------
    Path of the written CSV file.

    Raises
    ------
    FileNotFoundError if jsonl_path does not exist.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Trace file not found: {jsonl_path}")

    out_dir = output_dir or jsonl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / (jsonl_path.stem + ".csv")

    if csv_path.exists() and not overwrite:
        return csv_path

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        for idx, obj in enumerate(_iter_jsonl(jsonl_path)):
            writer.writerow(_row_from_event(idx, obj))

    return csv_path


def compact_all(
    trace_dir: Path,
    output_dir: Path | None = None,
    *,
    overwrite: bool = True,
) -> list[Path]:
    """
    Compact every *.jsonl file found directly inside trace_dir.

    Returns list of CSV paths written.
    """
    results: list[Path] = []
    for jsonl in sorted(trace_dir.glob("*.jsonl")):
        results.append(compact_traces(jsonl, output_dir, overwrite=overwrite))
    return results
