import pytest
from mok.memory.budget import BudgetManager
from mok.models.registry import ExpertMetadata, ExpertState


def make_expert(name: str, role: str, state: ExpertState, vram_cost_gb: float) -> ExpertMetadata:
    return ExpertMetadata(
        name=name,
        role=role,
        kind="full",
        backend="local",
        api_url=None,
        base_id=None,
        adapter_path=None,
        vram_cost_gb=vram_cost_gb,
        ram_cost_gb=1.0,
        current_device="cuda" if state != ExpertState.OFFLINE else "cpu",
        state=state,
    )


def test_budget_proposes_idle_evictions() -> None:
    core = make_expert("core", "coordinator", ExpertState.RESIDENT, 3.0)
    core.load_sequence = 1
    coder = make_expert("coder", "code", ExpertState.IDLE, 3.0)
    coder.load_sequence = 2
    instruct = make_expert("instruct", "general", ExpertState.IDLE, 3.0)
    instruct.load_sequence = 3
    vision = make_expert("vision", "vision", ExpertState.OFFLINE, 3.0)

    manager = BudgetManager(ceiling_gb=10.0, landing_zone_gb=1.0)
    evictions = manager.propose_evictions(vision, [core, coder, instruct, vision])

    assert evictions == ["coder"]


# ---------------------------------------------------------------------------
# AdapterSlot + propose_adapter_swap (P3.4)
# ---------------------------------------------------------------------------

from mok.memory.budget import AdapterSlot


def _slot(aid: str, role: str = "code", seq: int = 0) -> AdapterSlot:
    return AdapterSlot(adapter_id=aid, role=role, vram_cost_gb=0.25, load_seq=seq)


def test_adapter_swap_empty_returns_no_evictions():
    bm = BudgetManager()
    target = _slot("coder-lora")
    assert bm.propose_adapter_swap([], target, max_slots=4) == []


def test_adapter_swap_slot_available_no_eviction():
    bm = BudgetManager()
    current = [_slot("instruct-lora", seq=1)]
    target = _slot("coder-lora")
    assert bm.propose_adapter_swap(current, target, max_slots=4) == []


def test_adapter_swap_already_loaded_no_eviction():
    bm = BudgetManager()
    current = [_slot("coder-lora", seq=1), _slot("instruct-lora", seq=2)]
    target = _slot("coder-lora")
    assert bm.propose_adapter_swap(current, target, max_slots=4) == []


def test_adapter_swap_at_cap_evicts_oldest():
    bm = BudgetManager()
    current = [
        _slot("alpha", seq=1),
        _slot("beta", seq=2),
        _slot("gamma", seq=3),
        _slot("delta", seq=4),
    ]
    target = _slot("epsilon")
    evicted = bm.propose_adapter_swap(current, target, max_slots=4)
    assert evicted == ["alpha"]


def test_adapter_swap_evicts_only_one_when_one_slot_needed():
    bm = BudgetManager()
    current = [_slot("a", seq=10), _slot("b", seq=20), _slot("c", seq=30)]
    target = _slot("new")
    evicted = bm.propose_adapter_swap(current, target, max_slots=3)
    assert len(evicted) == 1
    assert evicted[0] == "a"


def test_adapter_swap_max_slots_one_always_evicts():
    bm = BudgetManager()
    current = [_slot("old", seq=5)]
    target = _slot("new")
    evicted = bm.propose_adapter_swap(current, target, max_slots=1)
    assert evicted == ["old"]


def test_adapter_vram_cost_sums_correctly():
    bm = BudgetManager()
    adapters = [_slot("a"), _slot("b")]
    cost = bm.adapter_vram_cost(adapters, base_vram_cost_gb=4.5)
    assert cost == pytest.approx(4.5 + 0.25 + 0.25)


def test_adapter_vram_cost_empty_adapters():
    bm = BudgetManager()
    assert bm.adapter_vram_cost([], base_vram_cost_gb=4.5) == pytest.approx(4.5)
