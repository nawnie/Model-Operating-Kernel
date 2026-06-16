"""
tests/test_vram_state.py

Tests for P5.3 (VRAMProfile + BudgetManager._effective_vram)
and P5.4 (ExpertContext / state_bus).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mok.memory.budget import BudgetManager
from mok.memory.state_bus import ExpertContext
from mok.models.registry import ExpertMetadata, ExpertState, ModelRegistry, VRAMProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expert(
    name: str = "coder",
    vram: float = 4.0,
    state: ExpertState = ExpertState.IDLE,
    vram_profile: VRAMProfile | None = None,
) -> ExpertMetadata:
    e = ExpertMetadata(
        name=name, role=name, kind="llm", backend="mock",
        api_url=None, base_id=None, adapter_path=None,
        vram_cost_gb=vram, ram_cost_gb=0.5,
        current_device="gpu", state=state,
    )
    e.vram_profile = vram_profile
    return e


# ---------------------------------------------------------------------------
# VRAMProfile
# ---------------------------------------------------------------------------

class TestVRAMProfile:
    def test_effective_gb_uses_static_when_no_measurement(self):
        p = VRAMProfile(static_gb=4.0)
        assert p.effective_gb == pytest.approx(4.0)

    def test_effective_gb_uses_measured_when_set(self):
        p = VRAMProfile(static_gb=4.0, measured_peak_gb=5.2)
        assert p.effective_gb == pytest.approx(5.2)

    def test_effective_gb_ignores_activation(self):
        # activation_gb is stored but effective_gb only cares about peak
        p = VRAMProfile(static_gb=4.0, measured_peak_gb=5.2, activation_gb=6.0)
        assert p.effective_gb == pytest.approx(5.2)

    def test_measured_none_falls_back_to_static(self):
        p = VRAMProfile(static_gb=3.5, measured_peak_gb=None)
        assert p.effective_gb == pytest.approx(3.5)

    def test_zero_static(self):
        p = VRAMProfile(static_gb=0.0)
        assert p.effective_gb == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BudgetManager._effective_vram
# ---------------------------------------------------------------------------

class TestEffectiveVram:
    def test_falls_back_to_vram_cost_gb_when_no_profile(self):
        bm = BudgetManager()
        e = _expert(vram=4.0, vram_profile=None)
        assert bm._effective_vram(e) == pytest.approx(4.0)

    def test_uses_profile_measured_peak(self):
        bm = BudgetManager()
        profile = VRAMProfile(static_gb=4.0, measured_peak_gb=5.5)
        e = _expert(vram=4.0, vram_profile=profile)
        assert bm._effective_vram(e) == pytest.approx(5.5)

    def test_pressure_uses_effective_vram(self):
        bm = BudgetManager(ceiling_gb=20.0, landing_zone_gb=2.0)
        profile = VRAMProfile(static_gb=4.0, measured_peak_gb=6.0)
        e = _expert("coder", vram=4.0, state=ExpertState.IDLE, vram_profile=profile)
        # pressure should use 6.0, not 4.0
        pressure = bm.current_pressure_gb([e])
        assert pressure == pytest.approx(6.0)

    def test_can_activate_respects_measured_peak(self):
        # usable = 8 GB; expert with measured_peak=7 → fits; with measured_peak=9 → does not
        bm = BudgetManager(ceiling_gb=10.0, landing_zone_gb=2.0)
        profile_fits = VRAMProfile(static_gb=4.0, measured_peak_gb=7.0)
        profile_too_big = VRAMProfile(static_gb=4.0, measured_peak_gb=9.0)
        assert bm.can_activate(_expert("a", vram_profile=profile_fits, state=ExpertState.OFFLINE), [])
        assert not bm.can_activate(_expert("b", vram_profile=profile_too_big, state=ExpertState.OFFLINE), [])


    def test_propose_evictions_uses_effective_vram(self):
        bm = BudgetManager(ceiling_gb=12.0, landing_zone_gb=2.0)
        target = _expert("vision", vram=5.0, state=ExpertState.OFFLINE)
        measured = _expert(
            "coder",
            vram=2.0,
            state=ExpertState.IDLE,
            vram_profile=VRAMProfile(static_gb=2.0, measured_peak_gb=6.0),
        )
        second = _expert("instruct", vram=3.0, state=ExpertState.IDLE)
        measured.load_sequence = 1
        second.load_sequence = 2

        evictions = bm.propose_evictions(target, [measured, second])

        assert evictions == ["coder"]


# ---------------------------------------------------------------------------
# ExpertContext construction
# ---------------------------------------------------------------------------

class TestExpertContextInit:
    def test_defaults(self):
        ctx = ExpertContext(request_id="r1")
        assert ctx.history == []
        assert ctx.artifacts == {}
        assert ctx.task_plan is None
        assert ctx.step_index == 0
        assert ctx.metadata == {}

    def test_repr_no_plan(self):
        ctx = ExpertContext(request_id="r1")
        assert "no-plan" in repr(ctx)

    def test_repr_with_plan(self):
        ctx = ExpertContext(request_id="r1", task_plan=["code", "general"])
        assert "step=0/2" in repr(ctx)


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

class TestHistory:
    def test_add_message(self):
        ctx = ExpertContext(request_id="r1")
        ctx.add_message("user", "hello")
        assert ctx.history == [{"role": "user", "content": "hello"}]

    def test_last_message_empty(self):
        ctx = ExpertContext(request_id="r1")
        assert ctx.last_message() is None

    def test_last_message_returns_most_recent(self):
        ctx = ExpertContext(request_id="r1")
        ctx.add_message("user", "first")
        ctx.add_message("assistant", "second")
        assert ctx.last_message() == {"role": "assistant", "content": "second"}

    def test_multiple_messages_preserved_in_order(self):
        ctx = ExpertContext(request_id="r1")
        for i in range(5):
            ctx.add_message("user", str(i))
        assert [m["content"] for m in ctx.history] == ["0", "1", "2", "3", "4"]


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

class TestArtifacts:
    def test_set_and_get(self):
        ctx = ExpertContext(request_id="r1")
        ctx.set_artifact("code", "def f(): pass")
        assert ctx.get_artifact("code") == "def f(): pass"

    def test_get_missing_returns_default(self):
        ctx = ExpertContext(request_id="r1")
        assert ctx.get_artifact("missing") == ""
        assert ctx.get_artifact("missing", "fallback") == "fallback"

    def test_overwrite(self):
        ctx = ExpertContext(request_id="r1")
        ctx.set_artifact("x", "v1")
        ctx.set_artifact("x", "v2")
        assert ctx.get_artifact("x") == "v2"

    def test_multiple_artifacts(self):
        ctx = ExpertContext(request_id="r1")
        ctx.set_artifact("a", "alpha")
        ctx.set_artifact("b", "beta")
        assert set(ctx.artifacts.keys()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Task-plan helpers
# ---------------------------------------------------------------------------

class TestTaskPlan:
    def test_current_step_role_no_plan(self):
        ctx = ExpertContext(request_id="r1")
        assert ctx.current_step_role is None

    def test_current_step_role_returns_correct(self):
        ctx = ExpertContext(request_id="r1", task_plan=["code", "general", "vision"])
        assert ctx.current_step_role == "code"

    def test_is_final_step_no_plan(self):
        ctx = ExpertContext(request_id="r1")
        assert ctx.is_final_step is True

    def test_is_final_step_single_step(self):
        ctx = ExpertContext(request_id="r1", task_plan=["code"])
        assert ctx.is_final_step is True

    def test_is_not_final_step(self):
        ctx = ExpertContext(request_id="r1", task_plan=["code", "general"])
        assert ctx.is_final_step is False

    def test_advance_increments(self):
        ctx = ExpertContext(request_id="r1", task_plan=["code", "general"])
        ctx.advance()
        assert ctx.step_index == 1
        assert ctx.current_step_role == "general"
        assert ctx.is_final_step is True

    def test_advance_does_not_exceed_plan_length(self):
        ctx = ExpertContext(request_id="r1", task_plan=["code"])
        ctx.advance()
        ctx.advance()   # second advance — should not go past 1
        assert ctx.step_index == 1

    def test_advance_returns_self(self):
        ctx = ExpertContext(request_id="r1", task_plan=["code", "general"])
        result = ctx.advance()
        assert result is ctx

    def test_advance_no_plan_is_noop(self):
        ctx = ExpertContext(request_id="r1")
        ctx.advance()
        assert ctx.step_index == 0


# ---------------------------------------------------------------------------
# to_dict / serialisation
# ---------------------------------------------------------------------------

class TestToDict:
    def test_to_dict_keys(self):
        ctx = ExpertContext(request_id="r1", task_plan=["code"])
        d = ctx.to_dict()
        assert "request_id" in d
        assert "history_len" in d
        assert "artifact_keys" in d
        assert "task_plan" in d
        assert "step_index" in d

    def test_to_dict_history_len(self):
        ctx = ExpertContext(request_id="r1")
        ctx.add_message("user", "hi")
        ctx.add_message("assistant", "hey")
        assert ctx.to_dict()["history_len"] == 2


# ---------------------------------------------------------------------------
# Runtime integration — context parameter accepted
# ---------------------------------------------------------------------------

class TestRuntimeContext:
    def test_handle_request_accepts_context(self, config_path: Path):
        from mok.memory.budget import BudgetManager
        from mok.models.backends import BackendResponse
        from mok.models.registry import ModelRegistry
        from mok.orchestration.runtime import OrchestratorRuntime
        from mok.routing.router import RulesRouter

        registry = ModelRegistry.from_json(config_path)
        backend = MagicMock()
        backend.generate.return_value = BackendResponse(text="ok", latency_ms=5)
        backends = {"local": backend, "vllm": backend}

        runtime = OrchestratorRuntime(
            registry=registry, router=RulesRouter(),
            budget_manager=BudgetManager(), backends=backends,
        )
        from mok.models.backends import RequestPayload
        ctx = ExpertContext(request_id="r1", task_plan=["code", "general"])
        result = runtime.handle_request(
            RequestPayload(prompt="write a sort function", request_id="r1"),
            context=ctx,
        )
        assert result.ok

    def test_handle_request_without_context_still_works(self, config_path: Path):
        from mok.memory.budget import BudgetManager
        from mok.models.backends import BackendResponse
        from mok.models.registry import ModelRegistry
        from mok.orchestration.runtime import OrchestratorRuntime
        from mok.routing.router import RulesRouter

        registry = ModelRegistry.from_json(config_path)
        backend = MagicMock()
        backend.generate.return_value = BackendResponse(text="hi", latency_ms=3)
        backends = {"local": backend, "vllm": backend}

        runtime = OrchestratorRuntime(
            registry=registry, router=RulesRouter(),
            budget_manager=BudgetManager(), backends=backends,
        )
        from mok.models.backends import RequestPayload
        result = runtime.handle_request(RequestPayload(prompt="hello", request_id="r2"))
        assert result.ok
