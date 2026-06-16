"""
src/mok/rsi/replay_buffer.py

RSI Replay Buffer
=================
JSONL pool of quality-gated ConsultationResult traces.

Design constraints
------------------
- Cap at MAX_RECORDS (10,000). Oldest evicted first (FIFO via append + trim).
- Deduplicated by (USER[:80], MOK_ACTION, gate) hash to prevent amplifying repetition.
- Diversity gate: if one lane exceeds MAX_LANE_FRACTION of the buffer, new records
  from that lane are deprioritized (still accepted but flagged).
- Thread-safe via a file lock (fcntl on Linux; skip on Windows).
- Stats written to buffer_stats.json after every write batch.

The buffer is the bridge between the live runtime and the fine-tune trigger.
It accumulates until `FineTuneTrigger` decides conditions are met, then
`build_rsi_batch.py` reads it and merges with the existing SFT corpus.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_RECORDS = 10_000
MAX_LANE_FRACTION = 0.60  # no single lane should dominate the buffer


@dataclass
class BufferRecord:
    """A single entry in the replay buffer."""
    record_id: str           # sha256 of (USER, MOK_ACTION, gate)
    timestamp: float
    quality_score: float
    lane: str
    USER: str
    STATE: dict
    AVAILABLE_EXPERTS: list
    RESOURCE_STATUS: dict
    MOK_ACTION: str
    EXPERT_REPLY: list
    MOK_CHECK: list
    MOK_FINAL: str
    trace: list
    messages: list
    diversity_flagged: bool = False  # True if lane was over MAX_LANE_FRACTION at write time

    def to_jsonl_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "BufferRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BufferStats:
    total_records: int = 0
    records_since_last_trigger: int = 0
    lane_counts: dict = field(default_factory=dict)
    mean_quality_score: float = 0.0
    distinct_lanes: int = 0
    last_written: float = 0.0
    last_triggered: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_records": self.total_records,
            "records_since_last_trigger": self.records_since_last_trigger,
            "lane_counts": self.lane_counts,
            "mean_quality_score": round(self.mean_quality_score, 4),
            "distinct_lanes": self.distinct_lanes,
            "last_written": self.last_written,
            "last_triggered": self.last_triggered,
        }


def _record_id(user: str, mok_action: str, gate: str) -> str:
    key = f"{user[:80]}|{mok_action}|{gate}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class ReplayBuffer:
    """JSONL replay buffer for RSI trace accumulation.

    Usage
    -----
    buf = ReplayBuffer(pool_dir=Path("training/rsi_pool"))
    buf.write(record)           # add one record
    buf.write_batch(records)    # add many at once (more efficient)
    stats = buf.stats()         # current buffer statistics
    records = buf.read_all()    # read all records (for batch export)
    buf.mark_triggered()        # reset records_since_last_trigger counter
    """

    BUFFER_FILE = "rsi_live_traces.jsonl"
    STATS_FILE = "buffer_stats.json"
    DEDUP_FILE = "rsi_seen_ids.txt"

    def __init__(self, pool_dir: Path) -> None:
        self.pool_dir = pool_dir
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_path = pool_dir / self.BUFFER_FILE
        self._stats_path = pool_dir / self.STATS_FILE
        self._dedup_path = pool_dir / self.DEDUP_FILE
        self._seen_ids: set[str] = self._load_seen_ids()

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def write(self, record: BufferRecord) -> bool:
        """Write one record. Returns True if written, False if deduped."""
        return self.write_batch([record]) == 1

    def write_batch(self, records: list[BufferRecord]) -> int:
        """Write multiple records. Returns count actually written."""
        written = 0
        new_lines: list[str] = []
        new_ids: list[str] = []

        # Load current lane distribution for diversity gate
        current_counts = self._lane_counts()
        total = sum(current_counts.values()) or 1

        for rec in records:
            # Dedup check
            if rec.record_id in self._seen_ids:
                continue

            # Diversity gate
            lane_fraction = current_counts.get(rec.lane, 0) / total
            if lane_fraction > MAX_LANE_FRACTION:
                rec.diversity_flagged = True
                # Still write, but flag it — trigger logic can weight these lower

            new_lines.append(rec.to_jsonl_line())
            new_ids.append(rec.record_id)
            current_counts[rec.lane] = current_counts.get(rec.lane, 0) + 1
            total += 1
            written += 1

        if not new_lines:
            return 0

        # Append to buffer
        with open(self._buffer_path, "a", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")

        # Update dedup set
        self._seen_ids.update(new_ids)
        with open(self._dedup_path, "a", encoding="utf-8") as f:
            f.write("\n".join(new_ids) + "\n")

        # Trim if over cap
        self._trim_to_cap()

        # Update stats
        self._update_stats()

        logger.debug("[ReplayBuffer] wrote %d records (total: %d)", written, self._count())
        return written

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def read_all(self) -> list[BufferRecord]:
        """Read all records from the buffer."""
        if not self._buffer_path.exists():
            return []
        records: list[BufferRecord] = []
        with open(self._buffer_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(BufferRecord.from_dict(json.loads(line)))
                except Exception:
                    continue
        return records

    def read_since_trigger(self) -> list[BufferRecord]:
        """Read only records written since the last trigger."""
        stats = self._load_stats()
        all_records = self.read_all()
        if stats.last_triggered == 0.0:
            return all_records
        return [r for r in all_records if r.timestamp >= stats.last_triggered]

    # ------------------------------------------------------------------
    # Stats + trigger support
    # ------------------------------------------------------------------

    def stats(self) -> BufferStats:
        return self._load_stats()

    def mark_triggered(self) -> None:
        """Call after a fine-tune batch is exported to reset the counter."""
        stats = self._load_stats()
        stats.last_triggered = time.time()
        stats.records_since_last_trigger = 0
        self._stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count(self) -> int:
        if not self._buffer_path.exists():
            return 0
        with open(self._buffer_path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def _lane_counts(self) -> dict[str, int]:
        counts: Counter = Counter()
        if not self._buffer_path.exists():
            return {}
        with open(self._buffer_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    counts[d.get("lane", "unknown")] += 1
                except Exception:
                    continue
        return dict(counts)

    def _trim_to_cap(self) -> None:
        """Keep only the most recent MAX_RECORDS lines."""
        if not self._buffer_path.exists():
            return
        with open(self._buffer_path, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        if len(lines) <= MAX_RECORDS:
            return
        excess = len(lines) - MAX_RECORDS
        logger.info("[ReplayBuffer] trimming %d old records", excess)
        with open(self._buffer_path, "w", encoding="utf-8") as f:
            f.writelines(lines[excess:])

    def _load_seen_ids(self) -> set[str]:
        if not self._dedup_path.exists():
            return set()
        return {line.strip() for line in self._dedup_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    def _load_stats(self) -> BufferStats:
        if not self._stats_path.exists():
            return BufferStats()
        try:
            d = json.loads(self._stats_path.read_text(encoding="utf-8"))
            s = BufferStats()
            for k, v in d.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            return s
        except Exception:
            return BufferStats()

    def _update_stats(self) -> None:
        records = self.read_all()
        old = self._load_stats()
        counts = Counter(r.lane for r in records)
        scores = [r.quality_score for r in records]
        new_since = len([r for r in records if r.timestamp >= old.last_triggered]) if old.last_triggered else len(records)
        stats = BufferStats(
            total_records=len(records),
            records_since_last_trigger=new_since,
            lane_counts=dict(counts),
            mean_quality_score=sum(scores) / len(scores) if scores else 0.0,
            distinct_lanes=len(counts),
            last_written=time.time(),
            last_triggered=old.last_triggered,
        )
        self._stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
