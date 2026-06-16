"""
tests/test_prefetch.py

Tests for mok.memory.prefetch (NextExpertPredictor — P5.1)
and the prefetch_hints extension to BudgetManager.propose_evictions() (P5.2).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mok.memory.budget import BudgetManager
from mok.memory.prefetch import NextExpertPredictor
from mok.models.registry import ExpertMetadata, ExpertState, ModelRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_traces(path: Path, expert_sequence: list[str]) -> None:
    """Write a minimal JSONL trace file with the given expert sequence."""
    with path.open("w", encoding="utf-8") as fh:
        for expert in expert_sequence:
            fh.write(json.dumps({"route_expert": expert, "request_id": "r"}) + "\n")


def _expert(
    name: str,
    state: ExpertState = ExpertState.IDLE,
    vram: float = 4.0,
    load_seq: int = 0,
) -> ExpertMetadata:
    return ExpertMetadata(
        name=name, role=name, kind="llm", backend="mock",
        api_url=None, base_id=None, adapter_path=None,
        vram_cost_gb=vram, ram_cost_gb=0.5,
        current_device="gpu", state=state,
        load_sequence=load_seq,
    )


# ---------------------------------------------------------------------------
# NextExpertPredictor.from_trace_jsonl
# ---------------------------------------------------------------------------

class TestFromTraceJsonl:
    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            NextExpertPredictor.from_trace_jsonl(tmp_path / "ghost.jsonl")

    def test_empty_file_produces_empty_predictor(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        t.write_text("", encoding="utf-8")
        p = NextExpertPredictor.from_trace_jsonl(t)
        assert p.known_experts() == set()

    def test_single_record_no_transitions(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, ["coder"])
        p = NextExpertPredictor.from_trace_jsonl(t)
        assert p.known_experts() == set()

    def test_two_records_one_transition(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, ["coder", "instruct"])
        p = NextExpertPredictor.from_trace_jsonl(t)
        assert "coder" in p.known_experts()
        assert p.transition_count("coder", "instruct") == pytest.approx(1.0)

    def test_skips_malformed_lines(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        with t.open("w") as fh:
            fh.write("not json\n")
            fh.write(json.dumps({"route_expert": "coder"}) + "\n")
            fh.write(json.dumps({"route_expert": "instruct"}) + "\n")
        p = NextExpertPredictor.from_trace_jsonl(t)
        assert p.transition_count("coder", "instruct") == pytest.approx(1.0)

    def test_repeated_same_expert_not_counted(self, tmp_path: Path):
        t = tmp_path / "t.jsonl"
        _write_traces(t, ["coder", "coder", "instruct"])
        p = NextExpertPredictor.from_trace_jsonl(t)
        # coder→coder should be ignored; only coder→instruct counted
        assert p.transition_count("coder", "coder") == pytest.approx(0.0)
        assert p.transition_count("coder", "instruct") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# NextExpertPredictor.predict
# ---------------------------------------------------------------------------

class TestPredict:
    @pytest.fixture()
    def registry(self, config_path: Path) -> ModelRegistry:
        return ModelRegistry.from_json(config_path)

    def test_returns_empty_for_unknown_expert(self, registry: ModelRegistry):
        p = NextExpertPredictor.empty()
        assert p.predict("unknown", registry) == []

    def test_returns_sorted_by_probability(self, tmp_path: Path, registry: ModelRegistry):
        t = tmp_path / "t.jsonl"
        # coder→general 3x, coder→vision 1x
        _write_traces(t, ["coder", "instruct", "coder", "instruct", "coder", "instruct", "coder", "vision"])
        p = NextExpertPredictor.from_trace_jsonl(t)
        hints = p.predict("coder", registry)
        # instruct should come first (higher probability)
        if len(hints) >= 2:
            assert hints[0][1] >= hints[1][1]

    def test_probabilities_from_single_source(self, tmp_path: Path, registry: ModelRegistry):
        t = tmp_path / "t.jsonl"
        _write_traces(t, ["coder", "instruct"])
        p = NextExpertPredictor.from_trace_jsonl(t)
        hints = p.predict("coder", registry)
        names = [h[0] for h in hints]
        probs = [h[1] for h in hints]
        assert "instruct" in names
        assert all(0.0 <= pr <= 1.0 for pr in probs)

    def test_top_k_limits_results(self, tmp_path: Path, registry: ModelRegistry):
        t = tmp_path / "t.jsonl"
        _write_traces(t, ["coder", "instruct", "coder", "vision"])
        p = NextExpertPredictor.from_trace_jsonl(t)
        hints = p.predict("coder", registry, top_k=1)
        assert len(hints) <= 1

    def test_excludes_current_expert(self, tmp_path: Path, registry: ModelRegistry):
        t = tmp_path / "t.jsonl"
        _write_traces(t, ["coder", "instruct", "coder", "coder"])
        p = NextExpertPredictor.from_trace_jsonl(t)
        hints = p.predict("coder", registry)
        assert all(name != "coder" for name, _ in hints)

    def test_excludes_experts_not_in_registry(self, tmp_path: Path, registry: ModelRegistry):
        t = tmp_path / "t.jsonl"
        _write_traces(t, ["coder", "ghost_expert"])  # ghost_expert not in registry
        p = NextExpertPredictor.from_trace_jsonl(t)
        hints = p.predict("coder", registry)
        names = [h[0] for h in hints]
        assert "ghost_expert" not in names

    def test_empty_predictor_returns_empty(self, registry: ModelRegistry):
        p = NextExpertPredictor.empty()
        assert p.predict("coder", registry) == []


# ---------------------------------------------------------------------------
# BudgetManager.propose_evictions with prefetch_hints (P5.2)
# ---------------------------------------------------------------------------

class TestProposeEvictionsWithHints:
    def _make_manager(self, ceiling: float = 14.0) -> BudgetManager:
        return BudgetManager(ceiling_gb=ceiling, landing_zone_gb=2.0)

    def test_no_hints_behaviour_unchanged(self):
        bm = self._make_manager(ceiling=10.0)
        # 3 idle experts × 4 GB = 12 GB loaded; target needs 4 GB
        # usable = 8 GB, projected = 12+4 = 16, so must evict
        experts = [
            _expert("a", ExpertState.IDLE, 4.0, load_seq=0),
            _expert("b", ExpertState.IDLE, 4.0, load_seq=1),
            _expert("c", ExpertState.IDLE, 4.0, load_seq=2),
        ]
        for e in experts:
            e.state = ExpertState.IDLE
        target = _expert("target", ExpertState.OFFLINE, 4.0)
        evictions = bm.propose_evictions(target, experts)
        assert len(evictions) > 0

    def test_hinted_expert_evicted_last(self):
        """
        "b" is hinted (predicted next).  With two IDLE experts and need
        to evict one, "a" (non-hinted) should be evicted before "b".
        """
        bm = BudgetManager(ceiling_gb=10.0, landing_zone_gb=2.0)
        # usable = 8 GB; a(4)+b(4) = 8 loaded; target needs 4 → must evict 1
        a = _expert("a", ExpertState.IDLE, 4.0, load_seq=0)
        b = _expert("b", ExpertState.IDLE, 4.0, load_seq=1)
        target = _expert("target", ExpertState.OFFLINE, 4.0)
        evictions = bm.propose_evictions(target, [a, b], prefetch_hints=["b"])
        # b is hinted → a should be evicted first
        assert evictions[0] == "a"
        assert "b" not in evictions

    def test_hinted_evicted_when_no_other_option(self):
        """If only the hinted expert can free enough VRAM, it must be evicted."""
        bm = BudgetManager(ceiling_gb=6.0, landing_zone_gb=0.5)
        # usable = 5.5 GB; b(5) loaded; target needs 4 → must evict b
        b = _expert("b", ExpertState.IDLE, 5.0, load_seq=0)
        target = _expert("target", ExpertState.OFFLINE, 4.0)
        evictions = bm.propose_evictions(target, [b], prefetch_hints=["b"])
        assert "b" in evictions

    def test_empty_hints_same_as_none(self):
        bm = BudgetManager(ceiling_gb=10.0, landing_zone_gb=2.0)
        a = _expert("a", ExpertState.IDLE, 4.0, load_seq=0)
        b = _expert("b", ExpertState.IDLE, 4.0, load_seq=1)
        target = _expert("target", ExpertState.OFFLINE, 4.0)
        ev_none = bm.propose_evictions(target, [a, b], prefetch_hints=None)
        ev_empty = bm.propose_evictions(target, [a, b], prefetch_hints=[])
        assert ev_none == ev_empty

    def test_no_eviction_needed_with_hints(self):
        bm = BudgetManager(ceiling_gb=20.0, landing_zone_gb=2.0)
        a = _expert("a", ExpertState.IDLE, 2.0)
        target = _expert("target", ExpertState.OFFLINE, 2.0)
        evictions = bm.propose_evictions(target, [a], prefetch_hints=["a"])
        assert evictions == []

    def test_hints_do_not_protect_core(self):
        """core expert is always protected regardless of hints."""
        bm = BudgetManager(ceiling_gb=8.0, landing_zone_gb=1.0)
        # usable = 7; core(4)+b(4) = 8; target(4) → must evict something
        core = _expert("core", ExpertState.IDLE, 4.0, load_seq=0)
        b = _expert("b", ExpertState.IDLE, 4.0, load_seq=1)
        target = _expert("target", ExpertState.OFFLINE, 4.0)
        evictions = bm.propose_evictions(target, [core, b], prefetch_hints=["b"])
        assert "core" not in evictions
