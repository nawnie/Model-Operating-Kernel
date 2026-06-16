"""
src/mok/rsi/trace_accumulator.py

RSI Trace Accumulator
=====================
Sits between the runtime and the replay buffer.
Receives ConsultationResult objects, scores them, and writes eligible
records to the replay buffer.

The accumulator is the entry point for the RSI flywheel.
Every call to ConsultationEngine flows through here automatically
when wired into OrchestratorRuntime.

Flow
----
  ConsultationEngine.handle() → ConsultationResult
       ↓
  TraceAccumulator.ingest(result, user_id, lane)
       ↓
  QualityScorer.score(result) → QualityScoreBreakdown
       ↓
  if score >= threshold: ReplayBuffer.write(BufferRecord)
       ↓
  FineTuneTrigger.check() → emit batch if conditions met

Thread safety: each ingest() call is independent. The ReplayBuffer
handles its own file locking. The accumulator itself is stateless
between calls (all state lives in the buffer files).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mok.rsi import quality_scorer
from mok.rsi.replay_buffer import BufferRecord, BufferStats, ReplayBuffer, _record_id
from mok.rsi.quality_scorer import QualityScoreBreakdown

if TYPE_CHECKING:
    from mok.orchestration.consultation import ConsultationResult

logger = logging.getLogger(__name__)

DEFAULT_POOL_DIR = Path("training/rsi_pool")
DEFAULT_QUALITY_THRESHOLD = 0.6


@dataclass
class IngestResult:
    """Outcome of one TraceAccumulator.ingest() call."""
    record_id: str
    quality_score: float
    breakdown: QualityScoreBreakdown
    accepted: bool          # True if written to buffer
    reason: str             # why accepted or rejected


class TraceAccumulator:
    """Receives ConsultationResult traces, scores them, writes eligible ones to ReplayBuffer.

    Usage
    -----
    accumulator = TraceAccumulator(pool_dir=Path("training/rsi_pool"))

    # After every ConsultationEngine.handle() call:
    ingest_result = accumulator.ingest(consultation_result, user_id="user_123", lane="single_expert_consult")

    # Check buffer stats:
    stats = accumulator.buffer.stats()
    """

    def __init__(
        self,
        pool_dir: Path = DEFAULT_POOL_DIR,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    ) -> None:
        self.pool_dir = pool_dir
        self.quality_threshold = quality_threshold
        self.buffer = ReplayBuffer(pool_dir)
        self._total_ingested = 0
        self._total_accepted = 0
        self._total_rejected = 0

    def ingest(
        self,
        result: "ConsultationResult",
        user_id: str = "anonymous",
        lane: str = "unknown",
    ) -> IngestResult:
        """Score a ConsultationResult and write to buffer if eligible."""
        self._total_ingested += 1

        # Score the result
        breakdown = quality_scorer.score(result)
        score = breakdown.total

        # Build the record id for deduplication
        training_record = result.to_training_record(
            user_prompt=result.request_id,  # placeholder — overridden below
        )
        user_text = training_record.get("USER", result.request_id)
        rec_id = _record_id(user_text, result.decision.value, result.gate)

        # Reject if below threshold
        if score < self.quality_threshold:
            self._total_rejected += 1
            reason = (
                f"quality {score:.2f} below threshold {self.quality_threshold:.2f}: "
                f"gate={breakdown.gate_discipline:.1f} "
                f"nocopy={breakdown.no_copy_enforcement:.1f} "
                f"challenge={breakdown.challenge_discipline:.1f} "
                f"conf={breakdown.confidence_calibration:.1f}"
            )
            logger.debug("[TraceAccumulator] rejected %s (%s)", rec_id, reason)
            return IngestResult(
                record_id=rec_id,
                quality_score=score,
                breakdown=breakdown,
                accepted=False,
                reason=reason,
            )

        # Build buffer record
        record = BufferRecord(
            record_id=rec_id,
            timestamp=time.time(),
            quality_score=round(score, 4),
            lane=lane,
            USER=user_text,
            STATE=training_record.get("STATE", {}),
            AVAILABLE_EXPERTS=training_record.get("AVAILABLE_EXPERTS", []),
            RESOURCE_STATUS=training_record.get("RESOURCE_STATUS", {}),
            MOK_ACTION=training_record.get("MOK_ACTION", result.decision.value),
            EXPERT_REPLY=training_record.get("EXPERT_REPLY", []),
            MOK_CHECK=training_record.get("MOK_CHECK", []),
            MOK_FINAL=result.final_answer,
            trace=result.trace,
            messages=self._build_messages(user_text, result),
        )

        written = self.buffer.write(record)
        if written:
            self._total_accepted += 1
            reason = f"quality {score:.2f} >= threshold {self.quality_threshold:.2f}"
            logger.info(
                "[TraceAccumulator] accepted %s (lane=%s score=%.2f)",
                rec_id, lane, score,
            )
        else:
            self._total_rejected += 1
            reason = "duplicate record (deduped)"

        return IngestResult(
            record_id=rec_id,
            quality_score=score,
            breakdown=breakdown,
            accepted=written,
            reason=reason,
        )

    def stats(self) -> dict:
        """Return accumulator + buffer statistics."""
        buf_stats = self.buffer.stats()
        return {
            "accumulator": {
                "total_ingested": self._total_ingested,
                "total_accepted": self._total_accepted,
                "total_rejected": self._total_rejected,
                "acceptance_rate": (
                    round(self._total_accepted / self._total_ingested, 3)
                    if self._total_ingested else 0.0
                ),
            },
            "buffer": buf_stats.to_dict(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, user_text: str, result: "ConsultationResult") -> list[dict]:
        """Build ChatML messages for SFT fine-tuning from the consultation result."""
        import json as _json
        return [
            {
                "role": "system",
                "content": (
                    "You are MoK Core — a coordinator model. "
                    "You consult helper models, challenge weak output, "
                    "and synthesize every final answer yourself. "
                    "You never copy expert output verbatim."
                ),
            },
            {"role": "user", "content": user_text},
            {
                "role": "assistant",
                "content": _json.dumps({
                    "decision": result.decision.value,
                    "gate": result.gate,
                    "confidence": result.confidence,
                    "trace": result.trace,
                    "final": result.final_answer,
                }, ensure_ascii=False),
            },
        ]
