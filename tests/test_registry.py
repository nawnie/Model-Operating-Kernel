import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mok.models.registry import ExpertMetadata, ExpertState, ModelRegistry


def test_registry_loads_experts_from_json(config_path: Path) -> None:
    registry = ModelRegistry.from_json(config_path)
    coder = registry.get("coder")
    assert coder.role == "code"
    assert coder.state == ExpertState.IDLE
    assert registry.find_first_by_role("vision").name == "vision"
    assert coder.file_format is None


# ---------------------------------------------------------------------------
# Ollama hydration (P2.3)
# ---------------------------------------------------------------------------

def _make_expert(
    name: str = "coder",
    backend: str = "ollama",
    base_id: str | None = "qwen2.5:7b",
) -> ExpertMetadata:
    return ExpertMetadata(
        name=name, role="code", kind="llm", backend=backend,
        api_url=None, base_id=base_id, adapter_path=None,
        vram_cost_gb=4.5, ram_cost_gb=1.0, current_device="cpu",
        state=ExpertState.OFFLINE,
    )


def test_hydrate_from_ollama_fills_architecture():
    expert = _make_expert()
    assert expert.architecture is None
    expert.hydrate_from_ollama([
        {"name": "qwen2.5:7b", "details": {"family": "qwen2", "quantization_level": "Q4_K_M"}},
    ])
    assert expert.architecture == "qwen2"


def test_hydrate_from_ollama_fills_quantization():
    expert = _make_expert()
    expert.hydrate_from_ollama([
        {"name": "qwen2.5:7b", "details": {"family": "qwen2", "quantization_level": "Q4_K_M"}},
    ])
    assert expert.quantization == "Q4_K_M"


def test_hydrate_from_ollama_no_op_when_backend_not_ollama():
    expert = _make_expert(backend="llama_cpp")
    expert.hydrate_from_ollama([
        {"name": "qwen2.5:7b", "details": {"family": "qwen2"}},
    ])
    assert expert.architecture is None


def test_hydrate_from_ollama_no_op_when_base_id_none():
    expert = _make_expert(base_id=None)
    expert.hydrate_from_ollama([
        {"name": "qwen2.5:7b", "details": {"family": "qwen2"}},
    ])
    assert expert.architecture is None


def test_hydrate_from_ollama_no_op_when_model_not_listed():
    expert = _make_expert(base_id="mistral:7b")
    expert.hydrate_from_ollama([
        {"name": "qwen2.5:7b", "details": {"family": "qwen2"}},
    ])
    assert expert.architecture is None


def test_hydrate_from_ollama_does_not_overwrite_existing():
    expert = _make_expert()
    expert.architecture = "already-set"
    expert.hydrate_from_ollama([
        {"name": "qwen2.5:7b", "details": {"family": "qwen2"}},
    ])
    assert expert.architecture == "already-set"


def test_registry_hydrate_from_ollama_server_returns_count(config_path: Path):
    registry = ModelRegistry.from_json(config_path)
    ollama_resp = json.dumps({"models": [
        {"name": "qwen2.5:7b", "details": {"family": "qwen2", "quantization_level": "Q4_K_M"}},
        {"name": "codestral:7b", "details": {"family": "mistral"}},
    ]}).encode("utf-8")
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    ctx.read = MagicMock(return_value=ollama_resp)
    with patch("urllib.request.urlopen", return_value=ctx):
        updated = registry.hydrate_from_ollama_server()
    assert isinstance(updated, int)
    assert updated >= 0


def test_registry_hydrate_from_ollama_server_returns_zero_on_connection_error(config_path: Path):
    from urllib.error import URLError
    registry = ModelRegistry.from_json(config_path)
    with patch("urllib.request.urlopen", side_effect=URLError("refused")):
        result = registry.hydrate_from_ollama_server()
    assert result == 0
