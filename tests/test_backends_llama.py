"""
tests/test_backends_llama.py

Tests for mok.models.backends_llama (LlamaCppBackend).
All HTTP calls are mocked — no server required.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mok.models.backends import BackendInvocationError, RequestPayload
from mok.models.backends_llama import DEFAULT_BASE_URL, DEFAULT_CHAT_PATH, LlamaCppBackend
from mok.models.registry import ExpertMetadata, ExpertState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expert(
    *,
    api_url: str | None = "http://localhost:8080/v1/chat/completions",
    model_path: str | None = "/models/qwen.gguf",
    context_limit: int = 4096,
    name: str = "general",
) -> ExpertMetadata:
    return ExpertMetadata(
        name=name,
        role="general",
        kind="llm",
        backend="llama_cpp",
        api_url=api_url,
        base_id=None,
        adapter_path=None,
        vram_cost_gb=4.5,
        ram_cost_gb=1.0,
        current_device="cpu",
        state=ExpertState.OFFLINE,
        model_path=model_path,
        context_limit=context_limit,
    )


def _payload(prompt: str = "hello") -> RequestPayload:
    return RequestPayload(prompt=prompt, request_id="req-test")


def _mock_response(text: str = "Hi!", model: str = "qwen") -> dict:
    return {
        "id": "cmpl-1",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 10},
    }


def _urlopen_ctx(response_dict: dict):
    """Build a mock context manager returned by urlopen."""
    raw = json.dumps(response_dict).encode("utf-8")
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    ctx.read = MagicMock(return_value=raw)
    return ctx


# ---------------------------------------------------------------------------
# Basic generate
# ---------------------------------------------------------------------------

class TestLlamaCppGenerate:
    def test_returns_backend_response(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx(_mock_response("Hello from llama!"))
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx):
            result = backend.generate(_expert(), _payload())
        assert result.text == "Hello from llama!"

    def test_uses_expert_api_url(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx(_mock_response())
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx) as mock_open:
            backend.generate(_expert(api_url="http://myhost:9999/v1/chat/completions"), _payload())
        req = mock_open.call_args[0][0]
        assert "myhost:9999" in req.full_url

    def test_falls_back_to_default_url_when_api_url_none(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx(_mock_response())
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx) as mock_open:
            backend.generate(_expert(api_url=None), _payload())
        req = mock_open.call_args[0][0]
        assert DEFAULT_BASE_URL in req.full_url
        assert DEFAULT_CHAT_PATH in req.full_url

    def test_uses_model_path_as_model_id(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx(_mock_response())
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx) as mock_open:
            backend.generate(_expert(model_path="/models/qwen.gguf"), _payload())
        body = json.loads(mock_open.call_args[0][0].data)
        assert body["model"] == "/models/qwen.gguf"

    def test_falls_back_to_expert_name_when_no_model_path(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx(_mock_response())
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx) as mock_open:
            backend.generate(_expert(model_path=None, name="general"), _payload())
        body = json.loads(mock_open.call_args[0][0].data)
        assert body["model"] == "general"

    def test_prompt_in_messages(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx(_mock_response())
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx) as mock_open:
            backend.generate(_expert(), _payload(prompt="explain recursion"))
        body = json.loads(mock_open.call_args[0][0].data)
        assert body["messages"][0]["content"] == "explain recursion"
        assert body["messages"][0]["role"] == "user"

    def test_stream_is_false(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx(_mock_response())
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx) as mock_open:
            backend.generate(_expert(), _payload())
        body = json.loads(mock_open.call_args[0][0].data)
        assert body["stream"] is False

    def test_max_tokens_set_from_context_limit(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx(_mock_response())
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx) as mock_open:
            backend.generate(_expert(context_limit=2048), _payload())
        body = json.loads(mock_open.call_args[0][0].data)
        assert body["max_tokens"] == 2048

    def test_max_tokens_capped_at_4096(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx(_mock_response())
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx) as mock_open:
            backend.generate(_expert(context_limit=99999), _payload())
        body = json.loads(mock_open.call_args[0][0].data)
        assert body["max_tokens"] == 4096

    def test_payload_parameters_forwarded(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx(_mock_response())
        payload = RequestPayload(prompt="hi", parameters={"temperature": 0.3, "seed": 42})
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx) as mock_open:
            backend.generate(_expert(), payload)
        body = json.loads(mock_open.call_args[0][0].data)
        assert body["temperature"] == 0.3
        assert body["seed"] == 42

    def test_metadata_contains_token_counts(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx(_mock_response())
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx):
            result = backend.generate(_expert(), _payload())
        assert result.metadata["prompt_tokens"] == 4
        assert result.metadata["completion_tokens"] == 10


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestLlamaCppErrors:
    def test_raises_on_http_error(self):
        from urllib.error import HTTPError
        backend = LlamaCppBackend()
        err = HTTPError("http://x", 503, "Service Unavailable", {}, None)
        err.read = lambda n=300: b"overloaded"
        with patch("mok.models.backends_llama.urlrequest.urlopen", side_effect=err):
            with pytest.raises(BackendInvocationError, match="HTTP 503"):
                backend.generate(_expert(), _payload())

    def test_raises_on_url_error(self):
        from urllib.error import URLError
        backend = LlamaCppBackend()
        with patch("mok.models.backends_llama.urlrequest.urlopen",
                   side_effect=URLError("connection refused")):
            with pytest.raises(BackendInvocationError, match="connection error"):
                backend.generate(_expert(), _payload())

    def test_raises_on_non_json_response(self):
        backend = LlamaCppBackend()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        ctx.read = MagicMock(return_value=b"not json")
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx):
            with pytest.raises(BackendInvocationError, match="non-JSON"):
                backend.generate(_expert(), _payload())

    def test_raises_on_missing_choices(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx({"id": "x", "model": "m", "choices": []})
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx):
            with pytest.raises(BackendInvocationError, match="unexpected response shape"):
                backend.generate(_expert(), _payload())

    def test_raises_on_missing_choices_key(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx({"id": "x", "model": "m"})
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx):
            with pytest.raises(BackendInvocationError, match="unexpected response shape"):
                backend.generate(_expert(), _payload())


# ---------------------------------------------------------------------------
# ping / list_local_models
# ---------------------------------------------------------------------------

class TestLlamaCppPing:
    def test_ping_true_on_health_200(self):
        backend = LlamaCppBackend()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx):
            assert backend.ping() is True

    def test_ping_false_on_connection_refused(self):
        from urllib.error import URLError
        backend = LlamaCppBackend()
        with patch("mok.models.backends_llama.urlrequest.urlopen",
                   side_effect=URLError("refused")):
            assert backend.ping() is False

    def test_list_models_returns_ids(self):
        backend = LlamaCppBackend()
        ctx = _urlopen_ctx({"data": [{"id": "qwen.gguf"}, {"id": "llama.gguf"}]})
        with patch("mok.models.backends_llama.urlrequest.urlopen", return_value=ctx):
            models = backend.list_local_models()
        assert "qwen.gguf" in models
        assert "llama.gguf" in models

    def test_list_models_returns_empty_on_error(self):
        from urllib.error import URLError
        backend = LlamaCppBackend()
        with patch("mok.models.backends_llama.urlrequest.urlopen",
                   side_effect=URLError("refused")):
            assert backend.list_local_models() == []
