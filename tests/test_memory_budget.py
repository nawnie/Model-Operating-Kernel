from mok.memory.budget import BudgetManager
from mok.models.registry import AssetState, ExpertMetadata, ModelRegistry


def make_expert(
    name: str,
    vram_cost_gb: float,
    state: AssetState = AssetState.OFFLINE,
    device: str = "cpu",
) -> ExpertMetadata:
    return ExpertMetadata(
        name=name,
        role=f"{name} role",
        backend="mock",
        api_url=f"mock://{name}",
        vram_cost_gb=vram_cost_gb,
        ram_cost_gb=vram_cost_gb + 1.0,
        state=state,
        current_device=device,
    )


def test_vram_pressure_counts_gpu_resident_states_only() -> None:
    registry = ModelRegistry()
    registry.register_expert(make_expert("core", 3.0, AssetState.RESIDENT, "cuda:0"))
    registry.register_expert(make_expert("coder", 4.0, AssetState.IDLE, "cuda:0"))
    registry.register_expert(make_expert("vision", 5.0, AssetState.STAGED, "cpu"))

    budget = BudgetManager(registry, max_vram_gb=14.5, landing_zone_gb=3.5)

    assert budget.calculate_vram_pressure() == 7.0


def test_allocation_clearance_evicts_idle_expert_when_needed() -> None:
    registry = ModelRegistry()
    registry.register_expert(make_expert("core", 4.0, AssetState.RESIDENT, "cuda:0"))
    registry.register_expert(make_expert("coder", 5.0, AssetState.IDLE, "cuda:0"))
    registry.register_expert(make_expert("vision", 4.0, AssetState.OFFLINE, "cpu"))

    budget = BudgetManager(registry, max_vram_gb=14.5, landing_zone_gb=3.5)
    evictions = budget.request_allocation_clearance("vision")

    assert evictions == ["coder"]
    coder = registry.get_expert("coder")
    assert coder is not None
    assert coder.state == AssetState.OFFLINE
    assert coder.current_device == "cpu"


def test_resident_core_is_not_evicted() -> None:
    registry = ModelRegistry()
    registry.register_expert(make_expert("core", 9.0, AssetState.RESIDENT, "cuda:0"))
    registry.register_expert(make_expert("vision", 4.0, AssetState.OFFLINE, "cpu"))

    budget = BudgetManager(registry, max_vram_gb=14.5, landing_zone_gb=3.5)
    evictions = budget.request_allocation_clearance("vision")

    assert evictions == []
    core = registry.get_expert("core")
    assert core is not None
    assert core.state == AssetState.RESIDENT
