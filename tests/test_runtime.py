from pathlib import Path

import pytest

from mok.models.backends import HTTPBackend, MockBackend, RequestPayload
from mok.models.registry import ModelRegistry
from mok.memory.budget import BudgetManager
from mok.orchestration.runtime import OrchestratorRuntime
from mok.routing.circuit_breaker import ExpertCircuitBreakerRegistry
from mok.routing.router import RoutingError, RulesRouter


def _make_runtime(config_path, tmp_path, breakers=None):
    return OrchestratorRuntime(
        registry=ModelRegistry.from_json(config_path),
        router=RulesRouter(),
        budget_manager=BudgetManager(),
        backends={"local": MockBackend(), "vllm": MockBackend(), "http": HTTPBackend()},
        trace_logger=None,
        circuit_breakers=breakers,
    )


def test_runtime_handles_request_and_logs_trace(config_path: Path, tmp_path: Path) -> None:
    trace_path = tmp_path / "runtime.jsonl"
    runtime = OrchestratorRuntime.from_config(
        config_path=config_path,
        trace_path=trace_path,
        backends={
            "local": MockBackend(),
            "vllm": MockBackend(),
            "http": HTTPBackend(),
        },
    )
    result = runtime.handle_request(
        RequestPayload(prompt="please fix this python bug", request_id="req-42")
    )
    assert result.expert_name == "coder"
    assert "Mock code specialist response" in result.text
    assert trace_path.exists()
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_runtime_rejects_request_when_circuit_open(config_path: Path, tmp_path: Path) -> None:
    breakers = ExpertCircuitBreakerRegistry(failure_threshold=1)
    breakers.trip("coder")
    runtime = _make_runtime(config_path, tmp_path, breakers=breakers)
    with pytest.raises(RoutingError, match="circuit-broken"):
        runtime.handle_request(
            RequestPayload(prompt="please fix this python bug", request_id="req-trip")
        )


def test_runtime_records_success_clears_failure_streak(config_path: Path, tmp_path: Path) -> None:
    breakers = ExpertCircuitBreakerRegistry(failure_threshold=3)
    runtime = _make_runtime(config_path, tmp_path, breakers=breakers)
    runtime.handle_request(
        RequestPayload(prompt="please fix this python bug", request_id="req-ok")
    )
    assert breakers._get_or_create("coder")._consecutive_failures == 0
