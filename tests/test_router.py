from pathlib import Path

from mok.models.backends import RequestPayload
from mok.models.registry import ModelRegistry
from mok.routing.router import RulesRouter


def build_registry() -> ModelRegistry:
    return ModelRegistry.from_json(
        Path(r"C:\Users\Shawn\Desktop\MoK-Project\configs\example_experts.json")
    )


def test_router_selects_coder_for_code_prompt() -> None:
    route = RulesRouter().route(
        RequestPayload(prompt="write a python function to sort a list"),
        build_registry(),
    )
    assert route.expert_name == "coder"


def test_router_selects_vision_for_image_requests() -> None:
    route = RulesRouter().route(
        RequestPayload(prompt="describe this screenshot", modality_flags={"has_image": True}),
        build_registry(),
    )
    assert route.expert_name == "vision"
