"""
tests/test_backends_ollama.py

Tests for mok.models.backends_ollama.OllamaBackend.
All network calls are mocked — no Ollama process required.
"""
from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from mok.models.backends import BackendInvocationError, RequestPayload
from mok.models.backends_ollama import DEFAULT_BASE_URL, OllamaBackend
from mok.models.registry import ExpertMetadata, ExpertState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_expert(
    *,
    name: str = "test-llm",
    role: str = "general",
    backend: str = "ollama",
    base_id: str | None = "llama3.2:3b",
    api_url: str | None = None,
    context_limit: int = 4096,
) -> ExpertMetadata:
    return ExpertMetadata(
        name=name,
        role=role,
        kind="local",
        backend=backend,
        api_url=api_url,
        base_id=base_id,
        adapter_path=None,
        vram_cost_gb=2.0,
        ram_cost_gb=4.0,
        current_device="cpu",
        state=ExpertState.IDLE,
        context_limit=context_limit,
    )


def make_payload(prompt: str = "Hello", **params) -> RequestPayload:
    return RequestPayload(prompt=prompt, request_id="req-test", parameters=params)


def mock_urlopen(response_body: dict):
    """Context-manager mock that returns a JSON body."""
    raw = json.dumps(response_body).encode("utf-8")
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    resp.read.return_value = raw
    resp.status = 200
    return resp


# ---------------------------------------------------------------------------
# generate() — happy path
# ---------------------------------------------------------------------------

class TestOllamaBackendGenerate:
    def test_returns_backend_response(self):
        expert = make_expert()
        payload = make_payload("Tell me a joke")
        ollama_resp = {
            "model": "llama3.2:3b",
            "response": "Why did the chicken cross the road?",
            "done": True,
            "eval_count": 12,
        }
        backend = OllamaBackend()
        with patch("mok.routing.circuit_breaker.time"):  # unrelated
            pass
        with patch("mok.models.backends_ollama.urlrequest.urlopen") as mock_open:
            mock_open.return_value = mock_urlopen(ollama_resp)
            result = backend.generate(expert, payload)

        assert result.text == "Why did the chicken cross the road?"
        assert result.latency_ms >= 0
        assert result.metadata["backend"] == "ollama"
        assert result.metadata["done"] is True
        assert result.metadata["eval_count"] == 12

    def test_uses_base_id_as_model_tag(self):
        expert = make_expert(base_id="codestral:latest")
        payload = make_payload("write a sort function")
        sent_bodies = []

        def capture_open(req, timeout=None):
            sent_bodies.append(json.loads(req.data.decode()))
            return mock_urlopen({"response": "done", "done": True})

        backend = OllamaBackend()
        with patch("mok.models.backends_ollama.urlrequest.urlopen", side_effect=capture_open):
            backend.generate(expert, payload)

        assert sent_bodies[0]["model"] == "codestral:latest"

    def test_falls_back_to_expert_name_when_no_base_id(self):
        expert = make_expert(name="my-local-expert", base_id=None)
        payload = make_payload("hello")
        sent_bodies = []

        def capture_open(req, timeout=None):
            sent_bodies.append(json.loads(req.data.decode()))
            return mock_urlopen({"response": "hi", "done": True})

        backend = OllamaBackend()
        with patch("mok.models.backends_ollama.urlrequest.urlopen", side_effect=capture_open):
            backend.generate(expert, payload)

        assert sent_bodies[0]["model"] == "my-local-expert"

    def test_uses_expert_api_url_override(self):
        expert = make_expert(api_url="http://remote:11434")
        payload = make_payload("hello")
        called_urls = []

        def capture_open(req, timeout=None):
            called_urls.append(req.full_url)
            return mock_urlopen({"response": "remote reply", "done": True})

        backend = OllamaBackend()
        with patch("mok.models.backends_ollama.urlrequest.urlopen", side_effect=capture_open):
            backend.generate(expert, payload)

        assert called_urls[0].startswith("http://remote:11434")

    def test_uses_default_base_url_when_no_api_url(self):
        expert = make_expert(api_url=None)
        payload = make_payload("hello")
        called_urls = []

        def capture_open(req, timeout=None):
            called_urls.append(req.full_url)
            return mock_urlopen({"response": "ok", "done": True})

        backend = OllamaBackend(base_url="http://localhost:11434")
        with patch("mok.models.backends_ollama.urlrequest.urlopen", side_effect=capture_open):
            backend.generate(expert, payload)

        assert called_urls[0].startswith("http://localhost:11434")

    def test_forwards_parameters_as_options(self):
        expert = make_expert()
        payload = make_payload("hi", temperature=0.7, seed=42)
        sent_bodies = []

        def capture_open(req, timeout=None):
            sent_bodies.append(json.loads(req.data.decode()))
            return mock_urlopen({"response": "cool", "done": True})

        backend = OllamaBackend()
        with patch("mok.models.backends_ollama.urlrequest.urlopen", side_effect=capture_open):
            backend.generate(expert, payload)

        options = sent_bodies[0]["options"]
        assert options["temperature"] == 0.7
        assert options["seed"] == 42

    def test_passes_context_limit_as_num_ctx(self):
        expert = make_expert(context_limit=2048)
        payload = make_payload("hi")
        sent_bodies = []

        def capture_open(req, timeout=None):
            sent_bodies.append(json.loads(req.data.decode()))
            return mock_urlopen({"response": "ok", "done": True})

        backend = OllamaBackend()
        with patch("mok.models.backends_ollama.urlrequest.urlopen", side_effect=capture_open):
            backend.generate(expert, payload)

        assert sent_bodies[0]["options"]["num_ctx"] == 2048

    def test_stream_is_always_false(self):
        expert = make_expert()
        payload = make_payload("hi")
        sent_bodies = []

        def capture_open(req, timeout=None):
            sent_bodies.append(json.loads(req.data.decode()))
            return mock_urlopen({"response": "ok", "done": True})

        backend = OllamaBackend()
        with patch("mok.models.backends_ollama.urlrequest.urlopen", side_effect=capture_open):
            backend.generate(expert, payload)

        assert sent_bodies[0]["stream"] is False


# ---------------------------------------------------------------------------
# generate() — error paths
# ---------------------------------------------------------------------------

class TestOllamaBackendErrors:
    def test_raises_on_http_error(self):
        from urllib.error import HTTPError
        expert = make_expert()
        payload = make_payload("hi")
        backend = OllamaBackend()

        err = HTTPError(url="http://x", code=404, msg="Not Found", hdrs=None, fp=None)
        with patch("mok.models.backends_ollama.urlrequest.urlopen", side_effect=err):
            with pytest.raises(BackendInvocationError, match="404"):
                backend.generate(expert, payload)

    def test_raises_on_url_error(self):
        from urllib.error import URLError
        expert = make_expert()
        payload = make_payload("hi")
        backend = OllamaBackend()

        err = URLError(reason="Connection refused")
        with patch("mok.models.backends_ollama.urlrequest.urlopen", side_effect=err):
            with pytest.raises(BackendInvocationError, match="unreachable"):
                backend.generate(expert, payload)

    def test_raises_on_timeout(self):
        expert = make_expert()
        payload = make_payload("hi")
        backend = OllamaBackend()

        with patch("mok.models.backends_ollama.urlrequest.urlopen", side_effect=TimeoutError()):
            with pytest.raises(BackendInvocationError, match="timed out"):
                backend.generate(expert, payload)

    def test_raises_on_non_json_response(self):
        expert = make_expert()
        payload = make_payload("hi")
        backend = OllamaBackend()

        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = b"not json at all"
        resp.status = 200

        with patch("mok.models.backends_ollama.urlrequest.urlopen", return_value=resp):
            with pytest.raises(BackendInvocationError, match="non-JSON"):
                backend.generate(expert, payload)

    def test_raises_when_response_field_missing(self):
        expert = make_expert()
        payload = make_payload("hi")
        backend = OllamaBackend()

        with patch("mok.models.backends_ollama.urlrequest.urlopen") as mock_open:
            mock_open.return_value = mock_urlopen({"done": True})  # no "response" key... wait
            # Actually {} gives response="" which is falsy but still a str — test the
            # case where "response" is an int (wrong type)
            mock_open.return_value = mock_urlopen({"response": 123, "done": True})
            with pytest.raises(BackendInvocationError, match="'response' field"):
                backend.generate(expert, payload)


# ---------------------------------------------------------------------------
# ping() and list_local_models()
# ---------------------------------------------------------------------------

class TestOllamaBackendUtils:
    def test_ping_returns_true_on_200(self):
        backend = OllamaBackend()
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.status = 200
        with patch("mok.models.backends_ollama.urlrequest.urlopen", return_value=resp):
            assert backend.ping() is True

    def test_ping_returns_false_on_error(self):
        from urllib.error import URLError
        backend = OllamaBackend()
        with patch("mok.models.backends_ollama.urlrequest.urlopen",
                   side_effect=URLError("refused")):
            assert backend.ping() is False

    def test_list_local_models_parses_tags(self):
        backend = OllamaBackend()
        body = json.dumps({"models": [
            {"name": "llama3.2:3b"},
            {"name": "codestral:latest"},
        ]}).encode()
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = body
        with patch("mok.models.backends_ollama.urlrequest.urlopen", return_value=resp):
            models = backend.list_local_models()
        assert "llama3.2:3b" in models
        assert "codestral:latest" in models

    def test_list_local_models_returns_empty_on_error(self):
        from urllib.error import URLError
        backend = OllamaBackend()
        with patch("mok.models.backends_ollama.urlrequest.urlopen",
                   side_effect=URLError("refused")):
            assert backend.list_local_models() == []
