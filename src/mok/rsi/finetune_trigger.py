"""
src/mok/rsi/finetune_trigger.py

RSI Fine-Tune Trigger
=====================
Watches the replay buffer. When conditions are met, writes a fine-tune
batch file to training/rsi_pool/ and marks the buffer as triggered.

Trigger conditions (all must be met)
--------------------------------------
1. Buffer has >= MIN_NEW_RECORDS new records since last trigger
2. Mean quality score of new batch >= MIN_QUALITY_SCORE
3. At least MIN_DISTINCT_LANES distinct lanes represented in new batch
4. Not triggered more recently than MIN_TRIGGER_INTERVAL_HOURS hours ago

When triggered
--------------
- Writes `rsi_batch_{timestamp}.jsonl` to the pool dir
- Calls buffer.mark_triggered() to reset the counter
- Logs a trigger event to `trigger_log.jsonl`

The batch file is the input to build_rsi_batch.py, which merges it
with the existing SFT corpus at the configured ratio.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from mok.rsi.replay_buffer import BufferRecord, BufferStats, ReplayBuffer

logger = logging.getLogger(__name__)

MIN_NEW_RECORDS = 500
MIN_QUALITY_SCORE = 0.70
MIN_DISTINCT_LANES = 3
MIN_TRIGGER_INTERVAL_HOURS = 4.0


@dataclass
class TriggerCheckResult:
    should_trigger: bool
    reason: str
    new_records: int
    mean_quality: float
    distinct_lanes: int
    batch_path: Path | None = None


class FineTuneTrigger:
    """Watches the replay buffer and fires when fine-tune conditions are met.

    Usage
    -----
    trigger = FineTuneTrigger(pool_dir=Path("training/rsi_pool"))
    result = trigger.check()
    if result.should_trigger:
        print(f"Batch ready: {result.batch_path}")
    """

    TRIGGER_LOG = "trigger_log.jsonl"

    def __init__(
        self,
        pool_dir: Path,
        min_new_records: int = MIN_NEW_RECORDS,
        min_quality_score: float = MIN_QUALITY_SCORE,
        min_distinct_lanes: int = MIN_DISTINCT_LANES,
        min_interval_hours: float = MIN_TRIGGER_INTERVAL_HOURS,
    ) -> None:
        self.pool_dir = pool_dir
        self.buffer = ReplayBuffer(pool_dir)
        self.min_new_records = min_new_records
        self.min_quality_score = min_quality_score
        self.min_distinct_lanes = min_distinct_lanes
        self.min_interval_seconds = min_interval_hours * 3600
        self._trigger_log_path = pool_dir / self.TRIGGER_LOG

    def check(self) -> TriggerCheckResult:
        """Check conditions and trigger if all are met. Returns TriggerCheckResult."""
        stats = self.buffer.stats()
        new_records = self.buffer.read_since_trigger()

        n = len(new_records)
        mean_q = sum(r.quality_score for r in new_records) / n if n else 0.0
        distinct_lanes = len({r.lane for r in new_records})

        # Check cooldown
        time_since_trigger = time.time() - stats.last_triggered if stats.last_triggered else float("inf")
        if time_since_trigger < self.min_interval_seconds:
            hours_remaining = (self.min_interval_seconds - time_since_trigger) / 3600
            return TriggerCheckResult(
                should_trigger=False,
                reason=f"cooldown: {hours_remaining:.1f}h remaining before next trigger",
                new_records=n,
                mean_quality=mean_q,
                distinct_lanes=distinct_lanes,
            )

        # Check record count
        if n < self.min_new_records:
            return TriggerCheckResult(
                should_trigger=False,
                reason=f"insufficient new records: {n} < {self.min_new_records}",
                new_records=n,
                mean_quality=mean_q,
                distinct_lanes=distinct_lanes,
            )

        # Check quality
        if mean_q < self.min_quality_score:
            return TriggerCheckResult(
                should_trigger=False,
                reason=f"quality below threshold: {mean_q:.3f} < {self.min_quality_score}",
                new_records=n,
                mean_quality=mean_q,
                distinct_lanes=distinct_lanes,
            )

        # Check diversity
        if distinct_lanes < self.min_distinct_lanes:
            return TriggerCheckResult(
                should_trigger=False,
                reason=f"insufficient lane diversity: {distinct_lanes} < {self.min_distinct_lanes}",
                new_records=n,
                mean_quality=mean_q,
                distinct_lanes=distinct_lanes,
            )

        # All conditions met — write batch
        batch_path = self._write_batch(new_records, mean_q)
        self.buffer.mark_triggered()
        self._log_trigger(n, mean_q, distinct_lanes, batch_path)

        logger.info(
            "[FineTuneTrigger] TRIGGERED: %d records, quality=%.3f, lanes=%d → %s",
            n, mean_q, distinct_lanes, batch_path,
        )

        return TriggerCheckResult(
            should_trigger=True,
            reason=f"all conditions met: {n} records, quality={mean_q:.3f}, {distinct_lanes} lanes",
            new_records=n,
            mean_quality=mean_q,
            distinct_lanes=distinct_lanes,
            batch_path=batch_path,
        )

    def _write_batch(self, records: list[BufferRecord], mean_quality: float) -> Path:
        """Write the new records as a dated batch file."""
        ts = int(time.time())
        batch_path = self.pool_dir / f"rsi_batch_{ts}.jsonl"

        with open(batch_path, "w", encoding="utf-8") as f:
            for rec in records:
                # Write in SFT-compatible format
                row = {
                    "id": f"rsi_{rec.record_id}",
                    "lane": rec.lane,
                    "dataset": f"rsi_live_batch_{ts}",
                    "quality_score": rec.quality_score,
                    "USER": rec.USER,
                    "STATE": rec.STATE,
                    "AVAILABLE_EXPERTS": rec.AVAILABLE_EXPERTS,
                    "RESOURCE_STATUS": rec.RESOURCE_STATUS,
                    "MOK_ACTION": rec.MOK_ACTION,
                    "EXPERT_REPLY": rec.EXPERT_REPLY,
                    "MOK_CHECK": rec.MOK_CHECK,
                    "MOK_FINAL": rec.MOK_FINAL,
                    "messages": rec.messages,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info("[FineTuneTrigger] wrote %d records to %s", len(records), batch_path)
        return batch_path

    def _log_trigger(
        self,
        record_count: int,
        mean_quality: float,
        distinct_lanes: int,
        batch_path: Path,
    ) -> None:
        event = {
            "timestamp": time.time(),
            "record_count": record_count,
            "mean_quality": round(mean_quality, 4),
            "distinct_lanes": distinct_lanes,
            "batch_file": str(batch_path.name),
        }
        with open(self._trigger_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
