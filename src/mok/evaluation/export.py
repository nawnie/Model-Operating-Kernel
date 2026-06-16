"""
src/mok/evaluation/export.py

R2 training pair exporter — P4.1.

Joins runtime JSONL trace events with oracle score files to produce
(prompt, modality, expert, oracle_score, router_tier) training pairs
for the R2 learned router.

No external dependencies — pure stdlib (json, csv, pathlib).

Usage
-----
    from mok.evaluation.export import export_training_pairs

    n = export_training_pairs(
        jsonl_path=Path("traces/runtime.jsonl"),
        oracle_scores_path=Path("traces/oracle_scores.jsonl"),
        output_path=Path("training/r2_pairs.jsonl"),
    )
    print(f"Exported {n} training pairs")

Training pair format (JSONL)
----------------------------
One JSON object per line:

    {
        "request_id":   "req-abc123",
        "prompt":       "write a python sort function",
        "modality":     {"has_image": false},
        "expert":       "coder",
        "oracle_score": 0.92,
        "router_tier":  "R0",
        "confidence":   0.84,
        "route_reason": "code keyword match"
    }

Oracle score file format (JSONL)
---------------------------------
    {"request_id": "req-abc123", "expert_scores": {"coder": 0.92, "general": 0.61}}

If an oracle entry has no score for the routed expert, that trace is skipped.
If a trace has no matching oracle entry, it is skipped (no imputation).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_training_pairs(
    jsonl_path: Path,
    oracle_scores_path: Path,
    output_path: Path,
    *,
    min_oracle_score: float = 0.0,
    max_pairs: int | None = None,
    overwrite: bool = True,
) -> int:
    """
    Join trace events with oracle scores and write R2 training pairs.

    Parameters
    ----------
    jsonl_path          : path to the runtime trace JSONL file
    oracle_scores_path  : path to the oracle scores JSONL file
    output_path         : where to write the training pairs JSONL
    min_oracle_score    : discard pairs where oracle_score < this value
    max_pairs           : cap total output rows (None = no cap)
    overwrite           : if False and output exists, return 0 immediately

    Returns
    -------
    Number of training pairs written.

    Raises
    ------
    FileNotFoundError if either input file does not exist.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Trace file not found: {jsonl_path}")
    if not oracle_scores_path.exists():
        raise FileNotFoundError(f"Oracle scores file not found: {oracle_scores_path}")

    if output_path.exists() and not overwrite:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load oracle scores into memory keyed by request_id
    oracle: dict[str, dict[str, float]] = _load_oracle(oracle_scores_path)

    written = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for trace in _iter_jsonl(jsonl_path):
            if max_pairs is not None and written >= max_pairs:
                break

            request_id = trace.get("request_id", "")
            expert = trace.get("route_expert", "")

            if not request_id or not expert:
                continue

            expert_scores = oracle.get(request_id)
            if expert_scores is None:
                continue  # no oracle entry for this trace

            oracle_score = expert_scores.get(expert)
            if oracle_score is None:
                continue  # oracle doesn't cover the routed expert

            if oracle_score < min_oracle_score:
                continue

            pair = {
                "request_id":   request_id,
                "prompt":       str(trace.get("prompt", ""))[:1000],
                "modality":     trace.get("modality_flags", {}),
                "expert":       expert,
                "oracle_score": round(float(oracle_score), 6),
                "router_tier":  str(trace.get("router_tier", "R0")),
                "confidence":   float(trace.get("route_confidence", 0.0)),
                "route_reason": str(trace.get("route_reason", "")),
            }
            out_fh.write(json.dumps(pair) + "\n")
            written += 1

    return written


def export_training_pairs_csv(
    jsonl_path: Path,
    oracle_scores_path: Path,
    output_path: Path,
    **kwargs,
) -> int:
    """
    Same as export_training_pairs but writes a CSV instead of JSONL.
    Returns the number of rows written (excluding header).
    """
    import csv

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Trace file not found: {jsonl_path}")
    if not oracle_scores_path.exists():
        raise FileNotFoundError(f"Oracle scores file not found: {oracle_scores_path}")

    # Write to a tmp JSONL buffer first, then convert
    tmp_path = output_path.with_suffix(".tmp.jsonl")
    try:
        n = export_training_pairs(jsonl_path, oracle_scores_path, tmp_path, **kwargs)
        if n == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "request_id,prompt,expert,oracle_score,router_tier,confidence,route_reason\n",
                encoding="utf-8",
            )
            return 0

        _FIELDS = [
            "request_id", "prompt", "expert", "oracle_score",
            "router_tier", "confidence", "route_reason",
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in _iter_jsonl(tmp_path):
                writer.writerow(row)
        return n
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_training_pairs(path: Path) -> list[dict]:
    """Load a training pairs JSONL file back into a list of dicts."""
    if not path.exists():
        return []
    return list(_iter_jsonl(path))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_oracle(path: Path) -> dict[str, dict[str, float]]:
    """
    Parse an oracle scores JSONL into {request_id: {expert: score}}.
    Skips blank/malformed lines silently.
    """
    result: dict[str, dict[str, float]] = {}
    for obj in _iter_jsonl(path):
        rid = obj.get("request_id", "")
        scores = obj.get("expert_scores", {})
        if rid and isinstance(scores, dict):
            result[rid] = {k: float(v) for k, v in scores.items()}
    return result


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
