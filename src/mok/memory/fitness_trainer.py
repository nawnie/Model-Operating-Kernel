"""
src/mok/memory/fitness_trainer.py

FitnessTrainer — the memory coach.

Responsibilities
----------------
1. SCORE    — compute fitness for every ACTIVE card based on accumulated usage signals.
2. COMPRESS — detect near-duplicate cards (Jaccard ≥ threshold) and merge them into
              a single SUMMARY card. Both originals are ARCHIVED with a superseded_by pointer.
3. ARCHIVE  — move low-fitness cards to CardState.ARCHIVED so they stop being retrieved.
4. PURGE    — delete ARCHIVED cards older than a configurable age.
5. PROMOTE  — move cards that clear all training thresholds into
              CardState.TRAINING_CANDIDATE.  They stay there until a batch is flushed.
6. FLUSH    — emit a training batch (list of TRAINING_CANDIDATE cards) and reset their state.

The trainer does NOT:
- Load or save the CardStore (the caller controls persistence).
- Retrain model weights (that is a separate, gated process).
- Import torch, numpy, or any external library.

Integration pattern
-------------------
    trainer = FitnessTrainer(store)

    # On every request where a card was used:
    signal = UsageSignal(card_id=cid, outcome_score=0.85, ...)
    trainer.record_signal(signal)

    # Periodically (e.g., after every N requests or on a timer):
    report = trainer.tick()          # score → compress → archive → promote

    # When the training pipeline is ready to consume a batch:
    batch = trainer.flush_training_batch()
    # → send batch to cloud training job
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from mok.memory.card import (
    CardState,
    CardType,
    FitnessScore,
    MemoryCard,
    UsageSignal,
    jaccard_similarity,
    new_card_id,
    recency_decay,
)
from mok.memory.card_store import CardStore


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Fitness scoring
RECENCY_HALF_LIFE_SECONDS: float = 86_400.0   # 1 day
MIN_OUTCOME_SAMPLES: int = 1                   # need ≥ N outcomes to use mean

# Compression
JACCARD_THRESHOLD: float = 0.80               # ≥ this → merge pair

# Archive triggers
MIN_WEIGHT_TO_STAY_ACTIVE: float = 0.05       # below this → archive candidate
MIN_USAGE_TO_AVOID_ARCHIVE: int = 1           # never-used cards can be archived
NEVER_USED_ARCHIVE_AGE_SECONDS: float = 7 * 86_400.0   # 7 days of zero use → archive

# Purge
PURGE_ARCHIVED_AFTER_SECONDS: float = 30 * 86_400.0    # 30 days

# Training gate — card must clear ALL of these
TRAINING_MIN_USAGE: int = 5
TRAINING_MIN_TRUST: float = 0.65
TRAINING_MIN_MEAN_OUTCOME: float = 0.70
TRAINING_MIN_STABILITY: float = 0.75
TRAINING_BATCH_SIZE: int = 10              # flush only when ≥ this many candidates


# ---------------------------------------------------------------------------
# FitnessReport
# ---------------------------------------------------------------------------

@dataclass
class FitnessReport:
    """Summary of what happened during a tick()."""
    scored:     int = 0
    compressed: int = 0    # pairs merged
    archived:   int = 0
    purged:     int = 0
    promoted:   int = 0
    scores:     dict[str, FitnessScore] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FitnessTrainer
# ---------------------------------------------------------------------------

class FitnessTrainer:
    """
    The memory coach.

    Thread safety: CardStore is thread-safe; FitnessTrainer itself is NOT —
    call it from a single background thread or hold an external lock.
    """

    def __init__(
        self,
        store: CardStore,
        *,
        recency_half_life: float = RECENCY_HALF_LIFE_SECONDS,
        jaccard_threshold: float = JACCARD_THRESHOLD,
        training_min_usage: int = TRAINING_MIN_USAGE,
        training_min_trust: float = TRAINING_MIN_TRUST,
        training_min_outcome: float = TRAINING_MIN_MEAN_OUTCOME,
        training_min_stability: float = TRAINING_MIN_STABILITY,
        training_batch_size: int = TRAINING_BATCH_SIZE,
        signal_log_path: Path | None = None,
    ) -> None:
        self.store = store
        self.recency_half_life = recency_half_life
        self.jaccard_threshold = jaccard_threshold
        self.training_min_usage = training_min_usage
        self.training_min_trust = training_min_trust
        self.training_min_outcome = training_min_outcome
        self.training_min_stability = training_min_stability
        self.training_batch_size = training_batch_size
        self._signal_log_path = signal_log_path
        self._pending_signals: list[UsageSignal] = []

    # ------------------------------------------------------------------
    # 1. Signal ingestion
    # ------------------------------------------------------------------

    def record_signal(self, signal: UsageSignal) -> None:
        """
        Log that card `signal.card_id` was used with the given outcome.

        Updates the card's usage_count, last_used_at, and outcome_scores
        immediately so scores are always current.

        The raw signal is also appended to the JSONL log if a path was given.
        """
        card = self.store.get(signal.card_id)
        if card is None:
            return   # card was purged or never existed — ignore

        # Update card fitness fields in-place
        self.store.update(
            signal.card_id,
            usage_count=card.usage_count + 1,
            last_used_at=time.time(),
            outcome_scores=card.outcome_scores + [max(0.0, min(1.0, signal.outcome_score))],
        )

        self._pending_signals.append(signal)
        if self._signal_log_path:
            self._append_signal_log(signal)

    def record_signals(self, signals: list[UsageSignal]) -> None:
        for sig in signals:
            self.record_signal(sig)

    # ------------------------------------------------------------------
    # 2. Scoring
    # ------------------------------------------------------------------

    def score_card(self, card: MemoryCard) -> FitnessScore:
        """Compute fitness for a single card."""
        # Recency: how fresh is the most recent use?
        recency = recency_decay(card.seconds_since_last_use, self.recency_half_life)

        # Usage factor: log-scaled so diminishing returns past ~20 uses
        import math
        usage = math.log1p(card.usage_count)   # log(1+n), 0 for n=0

        # Outcome factor: mean outcome if enough samples, else 0
        if len(card.outcome_scores) >= MIN_OUTCOME_SAMPLES:
            outcome = card.mean_outcome
        else:
            outcome = 0.0

        weight = usage * recency * (0.5 + 0.5 * outcome)
        trust  = card.trust_score * card.stability_score

        eligible = (
            card.usage_count    >= self.training_min_usage
            and card.trust_score  >= self.training_min_trust
            and outcome           >= self.training_min_outcome
            and card.stability_score >= self.training_min_stability
        )

        notes_parts = []
        if card.usage_count == 0:
            notes_parts.append("never used")
        if recency < 0.1:
            notes_parts.append("stale")
        if trust < 0.3:
            notes_parts.append("low trust")
        if eligible:
            notes_parts.append("TRAINING ELIGIBLE")

        return FitnessScore(
            card_id=card.card_id,
            weight=weight,
            trust=trust,
            retention_priority=weight * trust,
            training_eligible=eligible,
            notes="; ".join(notes_parts) if notes_parts else "healthy",
            recency_factor=recency,
            usage_factor=usage,
            outcome_factor=outcome,
        )

    def score_all(self) -> dict[str, FitnessScore]:
        """Score every ACTIVE card. Returns {card_id: FitnessScore}."""
        return {c.card_id: self.score_card(c) for c in self.store.active()}

    # ------------------------------------------------------------------
    # 3. Compression
    # ------------------------------------------------------------------

    def find_redundant_pairs(self) -> list[tuple[str, str, float]]:
        """
        Return (card_id_a, card_id_b, similarity) for all ACTIVE pairs where
        Jaccard similarity ≥ self.jaccard_threshold.

        O(n²) — fine for typical card counts (<1000); use sampling at larger scale.
        """
        active = self.store.active()
        pairs: list[tuple[str, str, float]] = []
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                if a.card_type != b.card_type:
                    continue   # only compress within the same type
                sim = jaccard_similarity(a.content, b.content)
                if sim >= self.jaccard_threshold:
                    pairs.append((a.card_id, b.card_id, sim))
        return pairs

    def compress(self, card_id_a: str, card_id_b: str) -> MemoryCard | None:
        """
        Merge two ACTIVE cards into a new SUMMARY card.

        The merged card:
        - content     = shorter card's content (higher information density)
        - trust_score = max(a.trust, b.trust)
        - usage_count = a.usage_count + b.usage_count
        - outcome_scores = combined list
        - compressed_from = [a.card_id, b.card_id]

        Both originals are ARCHIVED with superseded_by pointing at the new card.
        Returns None if either card is not found or not ACTIVE.
        """
        a = self.store.get(card_id_a)
        b = self.store.get(card_id_b)
        if a is None or b is None:
            return None
        if not (a.is_active and b.is_active):
            return None

        # Prefer shorter content — tighter card
        if len(b.content) < len(a.content):
            a, b = b, a

        merged_id = new_card_id()
        merged = MemoryCard(
            card_id=merged_id,
            card_type=CardType.SUMMARY,
            content=a.content,
            source=f"compressed from {a.card_id}, {b.card_id}",
            tags=list(set(a.tags) | set(b.tags)),
            trust_score=max(a.trust_score, b.trust_score),
            stability_score=min(a.stability_score, b.stability_score),
            usage_count=a.usage_count + b.usage_count,
            outcome_scores=a.outcome_scores + b.outcome_scores,
            last_used_at=max(a.last_used_at, b.last_used_at),
            compressed_from=[a.card_id, b.card_id],
            metadata={"merged_from_types": [str(a.card_type), str(b.card_type)]},
        )
        self.store.upsert(merged)

        # Archive originals with pointer to merged
        self.store.update(a.card_id, state=CardState.ARCHIVED, superseded_by=merged_id)
        self.store.update(b.card_id, state=CardState.ARCHIVED, superseded_by=merged_id)

        return merged

    def compress_all_redundant(self) -> int:
        """Run compression on all redundant pairs. Returns number of merges performed."""
        merges = 0
        pairs = self.find_redundant_pairs()
        seen: set[str] = set()
        for id_a, id_b, _ in pairs:
            if id_a in seen or id_b in seen:
                continue   # card already consumed in an earlier merge this pass
            result = self.compress(id_a, id_b)
            if result is not None:
                seen.add(id_a)
                seen.add(id_b)
                merges += 1
        return merges

    # ------------------------------------------------------------------
    # 4. Archive
    # ------------------------------------------------------------------

    def archive_low_value(self, scores: dict[str, FitnessScore] | None = None) -> int:
        """
        Move weak cards to ARCHIVED.

        A card is archived if:
          - it has never been used AND is older than NEVER_USED_ARCHIVE_AGE_SECONDS, OR
          - its computed weight falls below MIN_WEIGHT_TO_STAY_ACTIVE

        Returns the number of cards archived.
        """
        if scores is None:
            scores = self.score_all()
        now = time.time()
        archived = 0
        for card in self.store.active():
            score = scores.get(card.card_id)

            never_used_too_old = (
                card.usage_count == 0
                and (now - card.created_at) > NEVER_USED_ARCHIVE_AGE_SECONDS
            )
            low_weight = score is not None and score.weight < MIN_WEIGHT_TO_STAY_ACTIVE

            if never_used_too_old or low_weight:
                self.store.update(card.card_id, state=CardState.ARCHIVED)
                archived += 1
        return archived

    # ------------------------------------------------------------------
    # 5. Purge
    # ------------------------------------------------------------------

    def purge_old_archived(self, older_than_seconds: float = PURGE_ARCHIVED_AFTER_SECONDS) -> int:
        """
        Hard-delete ARCHIVED cards whose last_used_at (or created_at if never used)
        is older than older_than_seconds.

        Returns number of cards purged.
        """
        now = time.time()
        purged = 0
        for card in self.store.by_state(CardState.ARCHIVED):
            age_anchor = card.last_used_at if card.last_used_at > 0 else card.created_at
            if (now - age_anchor) > older_than_seconds:
                self.store.update(card.card_id, state=CardState.PURGED)
                purged += 1
        return purged

    # ------------------------------------------------------------------
    # 6. Promote to training candidates
    # ------------------------------------------------------------------

    def promote_eligible(self, scores: dict[str, FitnessScore] | None = None) -> int:
        """
        Move ACTIVE cards that are training_eligible → TRAINING_CANDIDATE.
        Does NOT automatically flush a batch — that is a separate gated call.
        Returns number of cards promoted.
        """
        if scores is None:
            scores = self.score_all()
        promoted = 0
        for card in self.store.active():
            score = scores.get(card.card_id)
            if score and score.training_eligible:
                self.store.update(
                    card.card_id,
                    state=CardState.TRAINING_CANDIDATE,
                    training_promoted_at=time.time(),
                )
                promoted += 1
        return promoted

    def flush_training_batch(self) -> list[MemoryCard]:
        """
        If enough TRAINING_CANDIDATE cards exist (≥ batch_size), return them
        as a list and reset their state to ACTIVE.

        Returns [] if the batch is not full yet — this prevents premature training.

        The caller is responsible for submitting the batch to the training pipeline.
        """
        candidates = self.store.by_state(CardState.TRAINING_CANDIDATE)
        if len(candidates) < self.training_batch_size:
            return []

        batch = candidates[:self.training_batch_size]
        for card in batch:
            # Reset to ACTIVE — the card keeps its fitness; training happens separately
            self.store.update(card.card_id, state=CardState.ACTIVE)
        return batch

    def pending_training_count(self) -> int:
        return self.store.count(CardState.TRAINING_CANDIDATE)

    # ------------------------------------------------------------------
    # 7. Full tick — run all phases in order
    # ------------------------------------------------------------------

    def tick(
        self,
        *,
        compress: bool = True,
        archive: bool = True,
        purge: bool = True,
        promote: bool = True,
    ) -> FitnessReport:
        """
        Run a complete fitness pass:
            score → [compress] → [archive] → [purge] → [promote]

        Returns a FitnessReport summarising what happened.

        Call this periodically (e.g., every N requests, or every hour).
        Not every-request — that would be wasteful.
        """
        report = FitnessReport()

        scores = self.score_all()
        report.scored = len(scores)
        report.scores = scores

        if compress:
            report.compressed = self.compress_all_redundant()
            # Re-score after merges (active set may have changed)
            scores = self.score_all()
            report.scores = scores

        if archive:
            report.archived = self.archive_low_value(scores)

        if purge:
            report.purged = self.purge_old_archived()

        if promote:
            report.promoted = self.promote_eligible(scores)

        return report

    # ------------------------------------------------------------------
    # 8. Signal log persistence
    # ------------------------------------------------------------------

    def _append_signal_log(self, signal: UsageSignal) -> None:
        assert self._signal_log_path is not None
        self._signal_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._signal_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(signal.to_dict()) + "\n")

    def drain_pending_signals(self) -> list[UsageSignal]:
        """Return and clear the in-memory pending signal list (useful for testing)."""
        signals = list(self._pending_signals)
        self._pending_signals.clear()
        return signals

    # ------------------------------------------------------------------
    # 9. Stats / diagnostics
    # ------------------------------------------------------------------

    def health_summary(self) -> dict:
        """Quick snapshot of the memory layer's current state."""
        active    = self.store.count(CardState.ACTIVE)
        archived  = self.store.count(CardState.ARCHIVED)
        training  = self.store.count(CardState.TRAINING_CANDIDATE)
        purged    = self.store.count(CardState.PURGED)
        total     = active + archived + training + purged

        scores = self.score_all()
        if scores:
            weights = [s.weight for s in scores.values()]
            mean_w  = sum(weights) / len(weights)
            trusts  = [s.trust  for s in scores.values()]
            mean_t  = sum(trusts)  / len(trusts)
        else:
            mean_w = mean_t = 0.0

        return {
            "total":              total,
            "active":             active,
            "archived":           archived,
            "training_candidate": training,
            "purged":             purged,
            "mean_weight":        round(mean_w, 4),
            "mean_trust":         round(mean_t, 4),
            "pending_signals":    len(self._pending_signals),
        }
