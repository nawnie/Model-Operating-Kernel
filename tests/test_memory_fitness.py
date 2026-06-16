"""
tests/test_memory_fitness.py

Tests for the MoK Memory Fitness Trainer.
Covers: card.py, card_store.py, fitness_trainer.py
All offline — no network, no disk I/O (tmp_path for persistence tests).
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

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
from mok.memory.fitness_trainer import (
    FitnessTrainer,
    TRAINING_BATCH_SIZE,
    TRAINING_MIN_USAGE,
    TRAINING_MIN_TRUST,
    TRAINING_MIN_MEAN_OUTCOME,
    TRAINING_MIN_STABILITY,
)


# ===========================================================================
# Helpers
# ===========================================================================

def make_store() -> CardStore:
    return CardStore()


def make_trainer(store: CardStore | None = None, **kwargs) -> FitnessTrainer:
    s = store if store is not None else make_store()
    return FitnessTrainer(s, **kwargs)


def make_signal(card_id: str, outcome: float = 0.9, request_id: str = "req-1") -> UsageSignal:
    return UsageSignal(card_id=card_id, outcome_score=outcome, request_id=request_id)


def add_healthy_card(store: CardStore, content: str = "Python → coder") -> str:
    return store.add(
        content=content,
        card_type=CardType.ROUTING_HINT,
        source="trace",
        trust_score=0.9,
    )


# ===========================================================================
# card.py — helpers
# ===========================================================================

class TestJaccardSimilarity:
    def test_identical_texts(self):
        assert jaccard_similarity("python function sort", "python function sort") == pytest.approx(1.0)

    def test_completely_different(self):
        sim = jaccard_similarity("apple orange banana", "car train plane")
        assert sim == pytest.approx(0.0)

    def test_partial_overlap(self):
        sim = jaccard_similarity("python function sort list", "python sort example")
        assert 0.0 < sim < 1.0

    def test_empty_strings(self):
        assert jaccard_similarity("", "") == pytest.approx(1.0)

    def test_one_empty(self):
        assert jaccard_similarity("python", "") == pytest.approx(0.0)

    def test_stop_words_excluded(self):
        # "is a the" are all stop words — should be ignored
        sim = jaccard_similarity("python is a language", "python the language")
        assert sim == pytest.approx(1.0)

    def test_threshold_detection(self):
        # Two near-duplicate routing hints
        a = "route python questions to the coder expert"
        b = "route python requests to coder expert"
        assert jaccard_similarity(a, b) >= 0.6


class TestRecencyDecay:
    def test_just_used_is_nearly_one(self):
        val = recency_decay(0.0, half_life_seconds=86400)
        assert val == pytest.approx(1.0)

    def test_half_life(self):
        val = recency_decay(86400.0, half_life_seconds=86400)
        assert val == pytest.approx(0.5, rel=0.01)

    def test_never_used_is_zero(self):
        assert recency_decay(None) == 0.0

    def test_very_old_approaches_zero(self):
        val = recency_decay(86400 * 30, half_life_seconds=86400)
        assert val < 0.01


class TestMemoryCard:
    def test_created_at_defaults_to_wall_clock_time(self):
        before = time.time()
        card = MemoryCard(card_id="c1", card_type=CardType.FACT,
                          content="x", source="test")
        after = time.time()
        assert before <= card.created_at <= after

    def test_mean_outcome_no_scores(self):
        card = MemoryCard(card_id="c1", card_type=CardType.FACT,
                          content="x", source="test")
        assert card.mean_outcome == 0.0

    def test_mean_outcome_with_scores(self):
        card = MemoryCard(card_id="c1", card_type=CardType.FACT,
                          content="x", source="test",
                          outcome_scores=[0.8, 1.0, 0.6])
        assert card.mean_outcome == pytest.approx(0.8)

    def test_is_active(self):
        card = MemoryCard(card_id="c1", card_type=CardType.FACT,
                          content="x", source="test", state=CardState.ACTIVE)
        assert card.is_active is True
        assert card.is_retrievable is True

    def test_archived_not_retrievable(self):
        card = MemoryCard(card_id="c1", card_type=CardType.FACT,
                          content="x", source="test", state=CardState.ARCHIVED)
        assert card.is_retrievable is False

    def test_roundtrip_serialise(self):
        card = MemoryCard(
            card_id="card-abc123",
            card_type=CardType.ROUTING_HINT,
            content="route to coder",
            source="trace",
            tags=["python"],
            trust_score=0.8,
            usage_count=3,
            outcome_scores=[0.9, 0.7],
        )
        d = card.to_dict()
        card2 = MemoryCard.from_dict(d)
        assert card2.card_id == card.card_id
        assert card2.content == card.content
        assert card2.usage_count == 3
        assert card2.outcome_scores == [0.9, 0.7]


class TestUsageSignal:
    def test_defaults(self):
        sig = UsageSignal(card_id="c1", outcome_score=0.8)
        assert sig.signal_id != ""
        assert sig.timestamp > 0

    def test_roundtrip(self):
        sig = UsageSignal(card_id="c1", outcome_score=0.75, request_id="req-5",
                          router_tier="R0", expert_used="coder")
        sig2 = UsageSignal.from_dict(sig.to_dict())
        assert sig2.card_id == "c1"
        assert sig2.outcome_score == pytest.approx(0.75)
        assert sig2.router_tier == "R0"


# ===========================================================================
# card_store.py
# ===========================================================================

class TestCardStore:
    def test_add_and_get(self):
        store = make_store()
        cid = store.add(content="python → coder", card_type=CardType.ROUTING_HINT, source="test")
        card = store.get(cid)
        assert card is not None
        assert card.content == "python → coder"
        assert card.state == CardState.ACTIVE

    def test_get_missing_returns_none(self):
        store = make_store()
        assert store.get("nonexistent") is None

    def test_update_field(self):
        store = make_store()
        cid = store.add(content="x", source="test")
        store.update(cid, usage_count=5)
        assert store.get(cid).usage_count == 5

    def test_update_invalid_field_raises(self):
        store = make_store()
        cid = store.add(content="x", source="test")
        with pytest.raises(ValueError, match="no field"):
            store.update(cid, nonexistent_field=42)

    def test_update_missing_card_returns_false(self):
        store = make_store()
        assert store.update("bad-id", usage_count=1) is False

    def test_remove(self):
        store = make_store()
        cid = store.add(content="x", source="test")
        assert store.remove(cid) is True
        assert store.get(cid) is None

    def test_remove_missing_returns_false(self):
        store = make_store()
        assert store.remove("ghost") is False

    def test_active_filters_non_active(self):
        store = make_store()
        cid = store.add(content="x", source="test")
        store.update(cid, state=CardState.ARCHIVED)
        assert store.active() == []

    def test_count_by_state(self):
        store = make_store()
        c1 = store.add(content="a", source="test")
        c2 = store.add(content="b", source="test")
        store.update(c2, state=CardState.ARCHIVED)
        assert store.count(CardState.ACTIVE) == 1
        assert store.count(CardState.ARCHIVED) == 1
        assert store.count() == 2

    def test_by_tag(self):
        store = make_store()
        c1 = store.add(content="a", source="test", tags=["python"])
        c2 = store.add(content="b", source="test", tags=["vision"])
        results = store.by_tag("python")
        assert len(results) == 1
        assert results[0].card_id == c1

    def test_filter(self):
        store = make_store()
        c1 = store.add(content="a", source="test", trust_score=0.9)
        c2 = store.add(content="b", source="test", trust_score=0.3)
        high_trust = store.filter(lambda c: c.trust_score >= 0.8)
        assert len(high_trust) == 1
        assert high_trust[0].card_id == c1

    def test_contains(self):
        store = make_store()
        cid = store.add(content="x", source="test")
        assert cid in store
        assert "ghost" not in store

    def test_len(self):
        store = make_store()
        assert len(store) == 0
        store.add(content="x", source="test")
        assert len(store) == 1

    def test_persistence_roundtrip(self, tmp_path: Path):
        p = tmp_path / "cards.json"
        store = CardStore(path=p)
        cid = store.add(content="persist me", source="test", trust_score=0.8)
        store.save()

        store2 = CardStore(path=p)
        card = store2.get(cid)
        assert card is not None
        assert card.content == "persist me"
        assert card.trust_score == pytest.approx(0.8)

    def test_reloaded_old_card_archives_by_wall_clock_age(self, tmp_path: Path):
        p = tmp_path / "cards.json"
        store = CardStore(path=p)
        cid = store.add(content="persistently stale", source="test")
        store.update(cid, created_at=time.time() - (8 * 86400))
        store.save()

        reloaded = CardStore(path=p)
        trainer = make_trainer(reloaded)
        assert trainer.archive_low_value() == 1
        assert reloaded.get(cid).state == CardState.ARCHIVED

    def test_reloaded_old_archived_card_purges_by_wall_clock_age(self, tmp_path: Path):
        p = tmp_path / "cards.json"
        store = CardStore(path=p)
        cid = store.add(content="old archived", source="test")
        store.update(
            cid,
            state=CardState.ARCHIVED,
            created_at=time.time() - (31 * 86400),
        )
        store.save()

        reloaded = CardStore(path=p)
        trainer = make_trainer(reloaded)
        assert trainer.purge_old_archived(older_than_seconds=30 * 86400) == 1
        assert reloaded.get(cid).state == CardState.PURGED

    def test_load_tolerates_corrupt_card(self, tmp_path: Path):
        p = tmp_path / "cards.json"
        # Write a file with one corrupt card and one valid card
        import json
        good = MemoryCard(card_id="c-good", card_type=CardType.FACT,
                          content="good", source="test")
        payload = {
            "version": 1,
            "cards": [
                {"broken": True},          # missing required fields
                good.to_dict(),
            ]
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
        store = CardStore(path=p)
        assert store.get("c-good") is not None


# ===========================================================================
# fitness_trainer.py
# ===========================================================================

class TestSignalIngestion:
    def test_record_signal_updates_card(self):
        store = make_store()
        cid = add_healthy_card(store)
        trainer = make_trainer(store)
        trainer.record_signal(make_signal(cid, outcome=0.9))
        card = store.get(cid)
        assert card.usage_count == 1
        assert card.outcome_scores == [0.9]
        assert card.last_used_at > 0

    def test_record_signal_clamps_outcome(self):
        store = make_store()
        cid = add_healthy_card(store)
        trainer = make_trainer(store)
        trainer.record_signal(UsageSignal(card_id=cid, outcome_score=1.5))  # above 1
        assert store.get(cid).outcome_scores == [1.0]

    def test_record_signal_ignores_missing_card(self):
        store = make_store()
        trainer = make_trainer(store)
        # No exception
        trainer.record_signal(UsageSignal(card_id="ghost", outcome_score=0.5))

    def test_multiple_signals_accumulate(self):
        store = make_store()
        cid = add_healthy_card(store)
        trainer = make_trainer(store)
        for i in range(5):
            trainer.record_signal(make_signal(cid, outcome=0.8, request_id=f"r{i}"))
        assert store.get(cid).usage_count == 5
        assert len(store.get(cid).outcome_scores) == 5

    def test_pending_signals_drained(self):
        store = make_store()
        cid = add_healthy_card(store)
        trainer = make_trainer(store)
        trainer.record_signal(make_signal(cid))
        signals = trainer.drain_pending_signals()
        assert len(signals) == 1
        assert trainer.drain_pending_signals() == []

    def test_signal_log_written(self, tmp_path: Path):
        import json as _json
        log_path = tmp_path / "signals.jsonl"
        store = make_store()
        cid = add_healthy_card(store)
        trainer = FitnessTrainer(store, signal_log_path=log_path)
        trainer.record_signal(make_signal(cid, outcome=0.75))
        lines = log_path.read_text().splitlines()
        assert len(lines) == 1
        obj = _json.loads(lines[0])
        assert obj["card_id"] == cid
        assert obj["outcome_score"] == pytest.approx(0.75)


class TestScoring:
    def test_never_used_card_has_zero_weight(self):
        store = make_store()
        cid = add_healthy_card(store)
        trainer = make_trainer(store)
        score = trainer.score_card(store.get(cid))
        assert score.weight == pytest.approx(0.0)

    def test_used_card_has_positive_weight(self):
        store = make_store()
        cid = add_healthy_card(store)
        trainer = make_trainer(store)
        for _ in range(3):
            trainer.record_signal(make_signal(cid, outcome=0.9))
        score = trainer.score_card(store.get(cid))
        assert score.weight > 0

    def test_more_usage_higher_weight(self):
        store = make_store()
        c1 = add_healthy_card(store, "card one")
        c2 = add_healthy_card(store, "card two")
        trainer = make_trainer(store)
        for _ in range(1):
            trainer.record_signal(make_signal(c1, outcome=0.8))
        for _ in range(10):
            trainer.record_signal(make_signal(c2, outcome=0.8))
        s1 = trainer.score_card(store.get(c1))
        s2 = trainer.score_card(store.get(c2))
        assert s2.weight > s1.weight

    def test_score_all_covers_active_only(self):
        store = make_store()
        c1 = add_healthy_card(store, "active")
        c2 = add_healthy_card(store, "archived")
        store.update(c2, state=CardState.ARCHIVED)
        trainer = make_trainer(store)
        scores = trainer.score_all()
        assert c1 in scores
        assert c2 not in scores

    def test_training_eligible_flag(self):
        store = make_store()
        cid = store.add(content="proven routing hint", source="trace",
                        card_type=CardType.ROUTING_HINT, trust_score=0.9)
        trainer = make_trainer(
            store,
            training_min_usage=3,
            training_min_trust=0.7,
            training_min_outcome=0.7,
            training_min_stability=0.7,
        )
        for _ in range(3):
            trainer.record_signal(make_signal(cid, outcome=0.85))
        score = trainer.score_card(store.get(cid))
        assert score.training_eligible is True

    def test_below_threshold_not_eligible(self):
        store = make_store()
        cid = add_healthy_card(store)
        trainer = make_trainer(store, training_min_usage=10)
        for _ in range(3):   # only 3, need 10
            trainer.record_signal(make_signal(cid, outcome=0.9))
        score = trainer.score_card(store.get(cid))
        assert score.training_eligible is False


class TestCompression:
    def test_find_redundant_pairs(self):
        store = make_store()
        store.add(content="route python code questions to coder expert", source="t",
                  card_type=CardType.ROUTING_HINT)
        store.add(content="route python coding requests to coder expert", source="t",
                  card_type=CardType.ROUTING_HINT)
        trainer = make_trainer(store, jaccard_threshold=0.5)
        pairs = trainer.find_redundant_pairs()
        assert len(pairs) >= 1

    def test_compress_creates_summary_card(self):
        store = make_store()
        c1 = store.add(content="python → coder expert for code tasks", source="t",
                        card_type=CardType.ROUTING_HINT)
        c2 = store.add(content="python → coder expert programming tasks", source="t",
                        card_type=CardType.ROUTING_HINT)
        trainer = make_trainer(store)
        merged = trainer.compress(c1, c2)
        assert merged is not None
        assert merged.card_type == CardType.SUMMARY
        assert c1 in merged.compressed_from
        assert c2 in merged.compressed_from

    def test_compress_archives_originals(self):
        store = make_store()
        c1 = store.add(content="a python expert routing hint", source="t",
                        card_type=CardType.ROUTING_HINT)
        c2 = store.add(content="b python expert routing hint", source="t",
                        card_type=CardType.ROUTING_HINT)
        trainer = make_trainer(store)
        merged = trainer.compress(c1, c2)
        assert store.get(c1).state == CardState.ARCHIVED
        assert store.get(c2).state == CardState.ARCHIVED
        assert store.get(c1).superseded_by == merged.card_id

    def test_compress_missing_card_returns_none(self):
        store = make_store()
        c1 = store.add(content="x", source="t")
        trainer = make_trainer(store)
        assert trainer.compress(c1, "nonexistent") is None

    def test_compress_already_archived_returns_none(self):
        store = make_store()
        c1 = store.add(content="a", source="t")
        c2 = store.add(content="b", source="t")
        store.update(c1, state=CardState.ARCHIVED)
        trainer = make_trainer(store)
        assert trainer.compress(c1, c2) is None

    def test_compress_inherits_max_trust(self):
        store = make_store()
        c1 = store.add(content="python routing to coder", source="t",
                        card_type=CardType.ROUTING_HINT, trust_score=0.6)
        c2 = store.add(content="python routing to coder expert", source="t",
                        card_type=CardType.ROUTING_HINT, trust_score=0.9)
        trainer = make_trainer(store)
        merged = trainer.compress(c1, c2)
        assert merged.trust_score == pytest.approx(0.9)

    def test_cross_type_not_compressed(self):
        store = make_store()
        store.add(content="python code sort function", source="t",
                  card_type=CardType.ROUTING_HINT)
        store.add(content="python code sort function", source="t",
                  card_type=CardType.FACT)
        trainer = make_trainer(store, jaccard_threshold=0.5)
        pairs = trainer.find_redundant_pairs()
        assert pairs == []


class TestArchive:
    def test_never_used_old_card_archived(self):
        store = make_store()
        cid = store.add(content="stale card", source="test")
        # Fake old creation time
        store.update(cid, created_at=time.time() - (8 * 86400))
        trainer = make_trainer(store)
        n = trainer.archive_low_value()
        assert n == 1
        assert store.get(cid).state == CardState.ARCHIVED

    def test_recently_used_card_not_archived(self):
        store = make_store()
        cid = add_healthy_card(store)
        trainer = make_trainer(store)
        for _ in range(5):
            trainer.record_signal(make_signal(cid, outcome=0.9))
        n = trainer.archive_low_value()
        assert n == 0
        assert store.get(cid).state == CardState.ACTIVE


class TestPurge:
    def test_purges_old_archived(self):
        store = make_store()
        cid = store.add(content="old archived", source="test")
        store.update(cid, state=CardState.ARCHIVED,
                     created_at=time.time() - (31 * 86400))
        trainer = make_trainer(store)
        n = trainer.purge_old_archived(older_than_seconds=30 * 86400)
        assert n == 1
        assert store.get(cid).state == CardState.PURGED

    def test_recently_archived_not_purged(self):
        store = make_store()
        cid = store.add(content="recent archived", source="test")
        store.update(cid, state=CardState.ARCHIVED)
        trainer = make_trainer(store)
        n = trainer.purge_old_archived(older_than_seconds=30 * 86400)
        assert n == 0


class TestPromotion:
    def _make_eligible_card(self, store: CardStore, trainer: FitnessTrainer,
                             usage: int = 5, outcome: float = 0.85) -> str:
        cid = store.add(content="proven card", source="trace",
                        card_type=CardType.ROUTING_HINT, trust_score=0.9)
        for _ in range(usage):
            trainer.record_signal(make_signal(cid, outcome=outcome))
        return cid

    def test_eligible_card_promoted(self):
        store = make_store()
        trainer = make_trainer(
            store,
            training_min_usage=3,
            training_min_trust=0.7,
            training_min_outcome=0.7,
            training_min_stability=0.7,
        )
        cid = self._make_eligible_card(store, trainer, usage=3)
        n = trainer.promote_eligible()
        assert n == 1
        assert store.get(cid).state == CardState.TRAINING_CANDIDATE

    def test_ineligible_card_not_promoted(self):
        store = make_store()
        trainer = make_trainer(store, training_min_usage=10)
        cid = add_healthy_card(store)
        trainer.record_signal(make_signal(cid, outcome=0.9))
        n = trainer.promote_eligible()
        assert n == 0


class TestFlushBatch:
    def test_batch_not_flushed_when_too_small(self):
        store = make_store()
        trainer = make_trainer(
            store,
            training_min_usage=1,
            training_min_trust=0.1,
            training_min_outcome=0.1,
            training_min_stability=0.1,
            training_batch_size=5,
        )
        cid = store.add(content="one card", source="test", trust_score=0.9)
        trainer.record_signal(make_signal(cid, outcome=0.9))
        trainer.promote_eligible()
        batch = trainer.flush_training_batch()
        assert batch == []

    def test_batch_flushed_when_full(self):
        store = make_store()
        trainer = make_trainer(
            store,
            training_min_usage=1,
            training_min_trust=0.1,
            training_min_outcome=0.1,
            training_min_stability=0.1,
            training_batch_size=3,
        )
        cids = []
        for i in range(3):
            cid = store.add(content=f"card {i} routing hint", source="test",
                             trust_score=0.9, card_type=CardType.ROUTING_HINT)
            trainer.record_signal(make_signal(cid, outcome=0.9))
            cids.append(cid)
        trainer.promote_eligible()
        batch = trainer.flush_training_batch()
        assert len(batch) == 3
        # Cards reset to ACTIVE after flush
        for cid in cids:
            assert store.get(cid).state == CardState.ACTIVE


class TestTick:
    def test_tick_returns_report(self):
        store = make_store()
        add_healthy_card(store)
        trainer = make_trainer(store)
        report = trainer.tick()
        assert report.scored >= 0
        assert isinstance(report.scores, dict)

    def test_tick_all_phases(self):
        store = make_store()
        cid = add_healthy_card(store)
        trainer = make_trainer(store)
        # Give it a signal
        trainer.record_signal(make_signal(cid, outcome=0.8))
        report = trainer.tick(compress=True, archive=True, purge=True, promote=True)
        assert report.scored == 1


class TestHealthSummary:
    def test_health_summary_keys(self):
        store = make_store()
        add_healthy_card(store)
        trainer = make_trainer(store)
        h = trainer.health_summary()
        assert "active" in h
        assert "archived" in h
        assert "training_candidate" in h
        assert "mean_weight" in h
        assert "mean_trust" in h

    def test_health_summary_counts(self):
        store = make_store()
        add_healthy_card(store)
        trainer = make_trainer(store)
        h = trainer.health_summary()
        assert h["active"] == 1
        assert h["total"] == 1
