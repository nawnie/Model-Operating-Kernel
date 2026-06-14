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
