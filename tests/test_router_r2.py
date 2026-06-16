"""
tests/test_router_r2.py

Tests for mok.routing.router_r2 (LearnedRouter — R2 tier).
All inference is pure numpy — no torch, no onnxruntime required.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mok.models.backends import RequestPayload
from mok.models.registry import ModelRegistry
from mok.routing.router_r2 import (
    HIDDEN,
    VOCAB_SIZE,
    LearnedRouter,
    _mlp_forward,
    _vectorize,
    make_untrained_checkpoint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROLES = ["code", "general", "vision", "coordinator"]


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "router.npz"
    make_untrained_checkpoint(ckpt, ROLES)
    return ckpt


@pytest.fixture()
def learned_router(checkpoint: Path) -> LearnedRouter:
    return LearnedRouter.from_checkpoint(checkpoint)


@pytest.fixture()
def registry(config_path: Path) -> ModelRegistry:
    return ModelRegistry.from_json(config_path)


def _payload(prompt: str = "write a sort function") -> RequestPayload:
    return RequestPayload(prompt=prompt, request_id="req-r2-test")


# ---------------------------------------------------------------------------
# _vectorize
# ---------------------------------------------------------------------------

class TestVectorize:
    def test_returns_correct_shape(self):
        vec = _vectorize("hello world", vocab_size=512)
        assert vec.shape == (512,)

    def test_returns_float32(self):
        vec = _vectorize("hello", vocab_size=128)
        assert vec.dtype == np.float32

    def test_l2_normalised(self):
        vec = _vectorize("hello world foo bar", vocab_size=512)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5

    def test_empty_string_returns_zeros(self):
        vec = _vectorize("", vocab_size=64)
        assert np.all(vec == 0.0)
        assert float(np.linalg.norm(vec)) == pytest.approx(0.0)

    def test_same_text_same_vector(self):
        v1 = _vectorize("explain recursion", vocab_size=256)
        v2 = _vectorize("explain recursion", vocab_size=256)
        np.testing.assert_array_equal(v1, v2)

    def test_same_text_same_vector_across_hash_seeds(self):
        script = (
            "import json; "
            "from mok.routing.router_r2 import _vectorize; "
            "print(json.dumps(_vectorize('explain recursion', vocab_size=64).tolist()))"
        )
        env_a = os.environ.copy()
        env_b = os.environ.copy()
        env_a["PYTHONHASHSEED"] = "1"
        env_b["PYTHONHASHSEED"] = "2"
        src_path = Path(__file__).resolve().parents[1] / "src"
        env_a["PYTHONPATH"] = str(src_path)
        env_b["PYTHONPATH"] = str(src_path)

        first = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env_a,
        )
        second = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env_b,
        )

        assert first.stdout == second.stdout

    def test_different_text_different_vector(self):
        v1 = _vectorize("code python", vocab_size=256)
        v2 = _vectorize("image analysis", vocab_size=256)
        assert not np.array_equal(v1, v2)

    def test_custom_vocab_size(self):
        vec = _vectorize("hello", vocab_size=32)
        assert vec.shape == (32,)


# ---------------------------------------------------------------------------
# _mlp_forward
# ---------------------------------------------------------------------------

class TestMlpForward:
    def _random_weights(self, n_roles: int = 4):
        rng = np.random.default_rng(42)
        W1 = rng.standard_normal((VOCAB_SIZE, HIDDEN)).astype(np.float32)
        b1 = np.zeros(HIDDEN, dtype=np.float32)
        W2 = rng.standard_normal((HIDDEN, n_roles)).astype(np.float32)
        b2 = np.zeros(n_roles, dtype=np.float32)
        return W1, b1, W2, b2

    def test_output_shape(self):
        W1, b1, W2, b2 = self._random_weights(4)
        vec = _vectorize("hello")
        out = _mlp_forward(vec, W1, b1, W2, b2)
        assert out.shape == (4,)

    def test_output_sums_to_one(self):
        W1, b1, W2, b2 = self._random_weights(4)
        vec = _vectorize("code please")
        out = _mlp_forward(vec, W1, b1, W2, b2)
        assert abs(float(out.sum()) - 1.0) < 1e-5

    def test_all_probabilities_non_negative(self):
        W1, b1, W2, b2 = self._random_weights(3)
        vec = _vectorize("describe the image")
        out = _mlp_forward(vec, W1, b1, W2, b2)
        assert (out >= 0).all()

    def test_deterministic(self):
        W1, b1, W2, b2 = self._random_weights(4)
        vec = _vectorize("hello world")
        out1 = _mlp_forward(vec, W1, b1, W2, b2)
        out2 = _mlp_forward(vec, W1, b1, W2, b2)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# make_untrained_checkpoint
# ---------------------------------------------------------------------------

class TestMakeUntrainedCheckpoint:
    def test_creates_file(self, tmp_path: Path):
        ckpt = tmp_path / "r2.npz"
        make_untrained_checkpoint(ckpt, ROLES)
        assert ckpt.exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        ckpt = tmp_path / "deep" / "nested" / "r2.npz"
        make_untrained_checkpoint(ckpt, ROLES)
        assert ckpt.exists()

    def test_checkpoint_has_required_keys(self, tmp_path: Path):
        ckpt = tmp_path / "r2.npz"
        make_untrained_checkpoint(ckpt, ROLES)
        data = np.load(ckpt, allow_pickle=False)
        for key in ("W1", "b1", "W2", "b2", "roles_json"):
            assert key in data.files

    def test_weight_shapes(self, tmp_path: Path):
        roles = ["a", "b", "c"]
        ckpt = tmp_path / "r2.npz"
        make_untrained_checkpoint(ckpt, roles, vocab_size=64, hidden=16)
        data = np.load(ckpt, allow_pickle=False)
        assert data["W1"].shape == (64, 16)
        assert data["b1"].shape == (16,)
        assert data["W2"].shape == (16, 3)
        assert data["b2"].shape == (3,)

    def test_reproducible_with_seed(self, tmp_path: Path):
        ckpt1 = tmp_path / "a.npz"
        ckpt2 = tmp_path / "b.npz"
        make_untrained_checkpoint(ckpt1, ROLES, seed=7)
        make_untrained_checkpoint(ckpt2, ROLES, seed=7)
        d1 = np.load(ckpt1, allow_pickle=False)
        d2 = np.load(ckpt2, allow_pickle=False)
        np.testing.assert_array_equal(d1["W1"], d2["W1"])

    def test_different_seeds_differ(self, tmp_path: Path):
        ckpt1 = tmp_path / "a.npz"
        ckpt2 = tmp_path / "b.npz"
        make_untrained_checkpoint(ckpt1, ROLES, seed=1)
        make_untrained_checkpoint(ckpt2, ROLES, seed=2)
        d1 = np.load(ckpt1, allow_pickle=False)
        d2 = np.load(ckpt2, allow_pickle=False)
        assert not np.array_equal(d1["W1"], d2["W1"])


# ---------------------------------------------------------------------------
# LearnedRouter.from_checkpoint
# ---------------------------------------------------------------------------

class TestLearnedRouterLoad:
    def test_loads_npz(self, checkpoint: Path):
        router = LearnedRouter.from_checkpoint(checkpoint)
        assert isinstance(router, LearnedRouter)

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            LearnedRouter.from_checkpoint(tmp_path / "ghost.npz")

    def test_roles_preserved(self, checkpoint: Path):
        router = LearnedRouter.from_checkpoint(checkpoint)
        assert router._roles == ROLES

    def test_repr_contains_roles(self, checkpoint: Path):
        router = LearnedRouter.from_checkpoint(checkpoint)
        r = repr(router)
        assert "LearnedRouter" in r
        assert "numpy" in r

    def test_raises_on_corrupt_npz(self, tmp_path: Path):
        bad = tmp_path / "bad.npz"
        bad.write_bytes(b"definitely not a zip file")
        with pytest.raises(Exception):
            LearnedRouter.from_checkpoint(bad)


# ---------------------------------------------------------------------------
# LearnedRouter.route
# ---------------------------------------------------------------------------

class TestLearnedRouterRoute:
    def test_returns_route_decision(self, learned_router: LearnedRouter, registry: ModelRegistry):
        result = learned_router.route(_payload(), registry)
        from mok.routing.router import RouteDecision
        assert isinstance(result, RouteDecision)

    def test_router_tier_is_r2(self, learned_router: LearnedRouter, registry: ModelRegistry):
        result = learned_router.route(_payload(), registry)
        assert result.router_tier == "R2"

    def test_confidence_in_unit_interval(self, learned_router: LearnedRouter, registry: ModelRegistry):
        result = learned_router.route(_payload(), registry)
        assert 0.0 <= result.confidence <= 1.0

    def test_expert_name_is_valid(self, learned_router: LearnedRouter, registry: ModelRegistry):
        result = learned_router.route(_payload(), registry)
        all_names = {e.name for e in registry.all()}
        assert result.expert_name in all_names

    def test_reason_mentions_r2(self, learned_router: LearnedRouter, registry: ModelRegistry):
        result = learned_router.route(_payload(), registry)
        assert "R2" in result.reason or "learned" in result.reason.lower()

    def test_deterministic_for_same_prompt(self, learned_router: LearnedRouter, registry: ModelRegistry):
        r1 = learned_router.route(_payload("python recursion"), registry)
        r2 = learned_router.route(_payload("python recursion"), registry)
        assert r1.expert_name == r2.expert_name
        assert r1.confidence == pytest.approx(r2.confidence)

    def test_different_prompts_may_differ(self, checkpoint: Path, registry: ModelRegistry):
        # With random weights this isn't guaranteed, but the vectorise step
        # produces different vectors so route() completes without error.
        router = LearnedRouter.from_checkpoint(checkpoint)
        r1 = router.route(_payload("write code"), registry)
        r2 = router.route(_payload("describe the image"), registry)
        # No assertion on equality — just verify both succeed
        assert r1.router_tier == r2.router_tier == "R2"

    def test_empty_prompt_does_not_crash(self, learned_router: LearnedRouter, registry: ModelRegistry):
        result = learned_router.route(_payload(""), registry)
        assert result.router_tier == "R2"


# ---------------------------------------------------------------------------
# Runtime integration
# ---------------------------------------------------------------------------

class TestRuntimeWithR2:
    def test_runtime_uses_r2_when_configured(self, config_path: Path, tmp_path: Path):
        from unittest.mock import MagicMock
        from mok.memory.budget import BudgetManager
        from mok.models.backends import BackendResponse
        from mok.models.registry import ModelRegistry
        from mok.orchestration.runtime import OrchestratorRuntime
        from mok.routing.router import RulesRouter

        registry = ModelRegistry.from_json(config_path)
        shared_backend = MagicMock()
        shared_backend.generate.return_value = BackendResponse(text="hello", latency_ms=5)
        backends = {"local": shared_backend, "vllm": shared_backend}

        ckpt = tmp_path / "r2.npz"
        make_untrained_checkpoint(ckpt, ["code", "general", "vision", "coordinator"])
        r2 = LearnedRouter.from_checkpoint(ckpt)

        runtime = OrchestratorRuntime(
            registry=registry,
            router=RulesRouter(),
            budget_manager=BudgetManager(),
            backends=backends,
            r2_router=r2,
        )
        result = runtime.handle_request(_payload("what is recursion"))
        assert result.ok
        assert result.route.router_tier == "R2"

    def test_runtime_without_r2_still_works(self, config_path: Path, tmp_path: Path):
        from unittest.mock import MagicMock
        from mok.memory.budget import BudgetManager
        from mok.models.backends import BackendResponse
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
            r2_router=None,
        )
        result = runtime.handle_request(_payload())
        assert result.ok
