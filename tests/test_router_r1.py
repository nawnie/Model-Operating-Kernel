"""
tests/test_router_r1.py

Tests for mok.routing.router_r1 (ZeroShotRouter — R1 tier).
All coordinator backend calls are mocked.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mok.models.backends import BackendResponse, RequestPayload
from mok.models.registry import ExpertMetadata, ExpertState, ModelRegistry
from mok.routing.router import RouteDecision
from mok.routing.router_r1 import ZeroShotRouter, _parse_role


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def registry(config_path: Path) -> ModelRegistry:
    return ModelRegistry.from_json(config_path)


def _r0(expert: str = "general", confidence: float = 0.65, tier: str = "R0") -> RouteDecision:
    return RouteDecision(
        expert_name=expert,
        confidence=confidence,
        reason="default general route",
        router_tier=tier,
    )


def _payload(prompt: str = "what is recursion") -> RequestPayload:
    return RequestPayload(prompt=prompt, request_id="req-r1-test")


def _mock_backend(reply: str = "code") -> MagicMock:
    backend = MagicMock()
    backend.generate.return_value = BackendResponse(text=reply, latency_ms=10)
    return backend


# ---------------------------------------------------------------------------
# _parse_role helper
# ---------------------------------------------------------------------------

class TestParseRole:
    def test_exact_match(self):
        assert _parse_role("code", ["code", "vision", "general"]) == "code"

    def test_case_insensitive(self):
        assert _parse_role("CODE", ["code", "vision"]) == "code"

    def test_first_word_in_longer_response(self):
        assert _parse_role("vision - handles image tasks", ["code", "vision", "general"]) == "vision"

    def test_returns_none_when_no_match(self):
        assert _parse_role("I don't know", ["code", "vision"]) is None

    def test_returns_none_on_empty_response(self):
        assert _parse_role("", ["code", "vision"]) is None

    def test_does_not_partial_match(self):
        # "coordinator" should not match "code" via partial string
        result = _parse_role("coordinator", ["code", "coordinator"])
        assert result == "coordinator"


# ---------------------------------------------------------------------------
# ZeroShotRouter — pass-through
# ---------------------------------------------------------------------------

class TestZeroShotRouterPassThrough:
    def test_passes_through_high_confidence(self, registry: ModelRegistry):
        backend = _mock_backend()
        r1 = ZeroShotRouter(backends={"mock": backend}, escalate_below=0.65)
        r0 = _r0(confidence=0.84)   # above threshold
        result = r1.route(_payload(), registry, r0)
        assert result is r0
        backend.generate.assert_not_called()

    def test_passes_through_at_threshold_boundary(self, registry: ModelRegistry):
        backend = _mock_backend()
        r1 = ZeroShotRouter(backends={"mock": backend}, escalate_below=0.65)
        r0 = _r0(confidence=0.65)   # exactly at threshold — not below, no escalation
        result = r1.route(_payload(), registry, r0)
        assert result is r0
        backend.generate.assert_not_called()


# ---------------------------------------------------------------------------
# ZeroShotRouter — escalation
# ---------------------------------------------------------------------------

class TestZeroShotRouterEscalation:
    # NOTE: coordinator expert in example_experts.json has backend="local"
    def test_escalates_low_confidence(self, registry: ModelRegistry):
        backend = _mock_backend(reply="code")
        r1 = ZeroShotRouter(backends={"local": backend}, escalate_below=0.65)
        r0 = _r0(confidence=0.50)   # below threshold → escalate
        result = r1.route(_payload(), registry, r0)
        assert result.router_tier == "R1"

    def test_escalation_sets_r1_tier(self, registry: ModelRegistry):
        backend = _mock_backend(reply="code")
        r1 = ZeroShotRouter(backends={"local": backend}, escalate_below=0.65)
        result = r1.route(_payload(), registry, _r0(confidence=0.40))
        assert result.router_tier == "R1"

    def test_escalation_uses_coordinator_response(self, registry: ModelRegistry):
        # coordinator replies "code" → should route to coder expert
        backend = _mock_backend(reply="code")
        r1 = ZeroShotRouter(backends={"local": backend}, escalate_below=0.65)
        result = r1.route(_payload(), registry, _r0(confidence=0.40))
        # The coder expert has role "code" in example_experts.json
        assert result.expert_name == "coder"

    def test_escalation_confidence_is_fixed_at_0_80(self, registry: ModelRegistry):
        backend = _mock_backend(reply="general")
        r1 = ZeroShotRouter(backends={"local": backend}, escalate_below=0.65)
        result = r1.route(_payload(), registry, _r0(confidence=0.40))
        assert result.confidence == pytest.approx(0.80)

    def test_escalation_reason_mentions_r1(self, registry: ModelRegistry):
        backend = _mock_backend(reply="code")
        r1 = ZeroShotRouter(backends={"local": backend}, escalate_below=0.65)
        result = r1.route(_payload(), registry, _r0(confidence=0.40))
        assert "R1" in result.reason or "zero-shot" in result.reason.lower()

    def test_calls_coordinator_backend(self, registry: ModelRegistry):
        backend = _mock_backend(reply="code")
        r1 = ZeroShotRouter(backends={"local": backend}, escalate_below=0.65)
        r1.route(_payload(), registry, _r0(confidence=0.40))
        backend.generate.assert_called_once()


# ---------------------------------------------------------------------------
# ZeroShotRouter — fallback on failure
# ---------------------------------------------------------------------------

class TestZeroShotRouterFallback:
    def test_falls_back_when_no_coordinator_in_registry(self, tmp_path: Path):
        """Registry with no coordinator expert — R1 must return r0."""
        import json
        cfg = tmp_path / "no_coord.json"
        cfg.write_text(json.dumps({"experts": [{
            "name": "coder", "role": "code", "kind": "llm",
            "backend": "mock", "api_url": None, "base_id": None,
            "adapter_path": None, "vram_cost_gb": 4.0, "ram_cost_gb": 0.5,
            "current_device": "cpu", "state": "offline",
        }]}), encoding="utf-8")
        reg = ModelRegistry.from_json(cfg)
        r0 = _r0(confidence=0.40)
        r1 = ZeroShotRouter(backends={"mock": _mock_backend()}, escalate_below=0.65)
        result = r1.route(_payload(), reg, r0)
        assert result is r0

    def test_falls_back_when_coordinator_backend_missing(self, registry: ModelRegistry):
        r0 = _r0(confidence=0.40)
        r1 = ZeroShotRouter(backends={}, escalate_below=0.65)   # no backends
        result = r1.route(_payload(), registry, r0)
        assert result is r0

    def test_falls_back_when_backend_raises(self, registry: ModelRegistry):
        backend = MagicMock()
        backend.generate.side_effect = RuntimeError("coordinator crashed")
        r1 = ZeroShotRouter(backends={"mock": backend}, escalate_below=0.65)
        r0 = _r0(confidence=0.40)
        result = r1.route(_payload(), registry, r0)
        assert result is r0

    def test_falls_back_when_coordinator_returns_unknown_role(self, registry: ModelRegistry):
        backend = _mock_backend(reply="flying_saucer")
        r1 = ZeroShotRouter(backends={"mock": backend}, escalate_below=0.65)
        r0 = _r0(confidence=0.40)
        result = r1.route(_payload(), registry, r0)
        assert result is r0

    def test_falls_back_when_coordinator_returns_empty_response(self, registry: ModelRegistry):
        backend = _mock_backend(reply="")
        r1 = ZeroShotRouter(backends={"mock": backend}, escalate_below=0.65)
        r0 = _r0(confidence=0.40)
        result = r1.route(_payload(), registry, r0)
        assert result is r0


# ---------------------------------------------------------------------------
# Runtime integration (R1 wired into OrchestratorRuntime)
# ---------------------------------------------------------------------------

class TestRuntimeWithR1:
    def test_runtime_uses_r1_when_r0_confidence_low(self, config_path: Path, tmp_path: Path):
        from mok.models.backends import BackendResponse
        from mok.memory.budget import BudgetManager
        from mok.models.registry import ModelRegistry
        from mok.orchestration.runtime import OrchestratorRuntime
        from mok.routing.router import RulesRouter

        registry = ModelRegistry.from_json(config_path)
        # example_experts.json uses backends "local" and "vllm"
        shared_backend = MagicMock()
        shared_backend.generate.return_value = BackendResponse(text="hello", latency_ms=5)
        backends = {"local": shared_backend, "vllm": shared_backend}

        # R1 router whose coordinator always replies "general"
        r1 = ZeroShotRouter(backends={"local": shared_backend}, escalate_below=0.99)  # always escalate

        runtime = OrchestratorRuntime(
            registry=registry,
            router=RulesRouter(),
            budget_manager=BudgetManager(),
            backends=backends,
            r1_router=r1,
        )
        result = runtime.handle_request(_payload(prompt="what is recursion"))
        assert result.ok

    def test_runtime_without_r1_still_works(self, config_path: Path, tmp_path: Path):
        from mok.models.backends import BackendResponse
        from mok.memory.budget import BudgetManager
        from mok.models.registry import ModelRegistry
        from mok.orchestration.runtime import OrchestratorRuntime
        from mok.routing.router import RulesRouter

        registry = ModelRegistry.from_json(config_path)
        shared_backend = MagicMock()
        shared_backend.generate.return_value = BackendResponse(text="hi", latency_ms=3)
        backends = {"local": shared_backend, "vllm": shared_backend}

        runtime = OrchestratorRuntime(
            registry=registry,
            router=RulesRouter(),
            budget_manager=BudgetManager(),
            backends=backends,
            r1_router=None,
        )
        result = runtime.handle_request(_payload())
        assert result.ok
