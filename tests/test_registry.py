from pathlib import Path

from mok.models.registry import ExpertState, ModelRegistry


def test_registry_loads_experts_from_json() -> None:
    registry = ModelRegistry.from_json(
        Path(r"C:\Users\Shawn\Desktop\MoK-Project\configs\example_experts.json")
    )

    coder = registry.get("coder")

    assert coder.role == "code"
    assert coder.state == ExpertState.IDLE
    assert registry.find_first_by_role("vision").name == "vision"
    assert coder.file_format is None
