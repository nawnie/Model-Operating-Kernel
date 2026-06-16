from pathlib import Path

import pytest

from mok.models.backends import RequestPayload
from mok.models.registry import ModelRegistry
from mok.routing.router import RulesRouter


def test_router_selects_coder_for_code_prompt(config_path: Path) -> None:
    route = RulesRouter().route(
        RequestPayload(prompt="write a python function to sort a list"),
        ModelRegistry.from_json(config_path),
    )
    assert route.expert_name == "coder"


def test_router_selects_vision_for_image_requests(config_path: Path) -> None:
    route = RulesRouter().route(
        RequestPayload(prompt="describe this screenshot", modality_flags={"has_image": True}),
        ModelRegistry.from_json(config_path),
    )
    assert route.expert_name == "vision"
