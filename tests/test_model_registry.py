from mok.models.registry import AssetState, ExpertMetadata, ModelRegistry


def test_register_and_fetch_expert() -> None:
    registry = ModelRegistry()
    expert = ExpertMetadata(
        name="coder",
        role="code generation",
        backend="mock",
        api_url="mock://coder",
        vram_cost_gb=4.0,
        ram_cost_gb=6.0,
    )

    registry.register_expert(expert)

    assert registry.get_expert("coder") == expert
    assert registry.get_expert("missing") is None


def test_update_state_changes_lifecycle_and_device() -> None:
    registry = ModelRegistry()
    registry.register_expert(
        ExpertMetadata(
            name="vision",
            role="image understanding",
            backend="mock",
            api_url="mock://vision",
            vram_cost_gb=5.5,
            ram_cost_gb=7.5,
        )
    )

    registry.update_state("vision", AssetState.ACTIVE, "cuda:0")
    expert = registry.get_expert("vision")

    assert expert is not None
    assert expert.state == AssetState.ACTIVE
    assert expert.current_device == "cuda:0"


def test_get_experts_by_state() -> None:
    registry = ModelRegistry()
    registry.register_expert(
        ExpertMetadata(
            name="core",
            role="coordinator",
            backend="mock",
            api_url="mock://core",
            vram_cost_gb=3.0,
            ram_cost_gb=4.0,
            state=AssetState.RESIDENT,
            current_device="cuda:0",
        )
    )
    registry.register_expert(
        ExpertMetadata(
            name="coder",
            role="code generation",
            backend="mock",
            api_url="mock://coder",
            vram_cost_gb=4.0,
            ram_cost_gb=6.0,
            state=AssetState.IDLE,
            current_device="cuda:0",
        )
    )

    resident = registry.get_experts_by_state(AssetState.RESIDENT)

    assert len(resident) == 1
    assert resident[0].name == "core"
