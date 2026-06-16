"""
src/mok/models/backends_ollama.py

Ollama backend for the MoK expert system — P2.2.

Maps ExpertMetadata → Ollama /api/generate (streaming=False).

No third-party dependencies; uses only stdlib urllib.
Ollama must already be running (default: http://localhost:11434).

ExpertMetadata wiring
---------------------
- expert.base_id   : Ollama model tag  e.g. "llama3.2:3b", "codestral:latest"
                     Falls back to expert.name if base_id is None.
- expert.api_url   : Override base URL  e.g. "http://remote-host:11434"
                     Falls back to DEFAULT_BASE_URL if None.
- expert.context_limit : passed as "num_ctx" option.
- payload.parameters   : forwarded verbatim as Ollama options
                         (temperature, top_p, top_k, seed, …).

Response contract
-----------------
Returns BackendResponse(text=..., latency_ms=..., metadata={...}).
Raises BackendInvocationError on any HTTP or parse error.
"""
from __future__ import annotations

import json
import time
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from mok.models.backends import BackendInvocationError, BackendResponse, RequestPayload
from mok.models.registry import ExpertMetadata


DEFAULT_BASE_URL = "http://localhost:11434"
_GENERATE_PATH = "/api/generate"
_TIMEOUT_SECONDS = 120      # generous; large models can be slow to respond


class OllamaBackend:
    """
    Pure-stdlib Ollama backend.

    One instance is safe to share across requests; it holds no mutable state.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        # Strip trailing slash once so we never double-slash the path
        self._default_base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # ExpertBackend protocol
    # ------------------------------------------------------------------

    def generate(self, expert: ExpertMetadata, payload: RequestPayload) -> BackendResponse:
        base_url = (
            expert.api_url.rstrip("/") if expert.api_url else self._default_base_url
        )
        model_tag = expert.base_id or expert.name
        url = base_url + _GENERATE_PATH

        options: dict = {
            "num_ctx": expert.context_limit,
        }
        # Forward any caller-supplied options (temperature, seed, etc.)
        for k, v in payload.parameters.items():
            options[k] = v

        body = json.dumps(
            {
                "model": model_tag,
                "prompt": payload.prompt,
                "stream": False,
                "options": options,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        req = urlrequest.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urlrequest.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as exc:
            raise BackendInvocationError(
                f"Ollama HTTP {exc.code} for expert '{expert.name}' "
                f"(model={model_tag!r}): {exc.reason}"
            ) from exc
        except URLError as exc:
            raise BackendInvocationError(
                f"Ollama unreachable for expert '{expert.name}' "
                f"(url={url!r}): {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise BackendInvocationError(
                f"Ollama timed out after {_TIMEOUT_SECONDS}s "
                f"for expert '{expert.name}'"
            ) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BackendInvocationError(
                f"Ollama returned non-JSON for expert '{expert.name}': {exc}"
            ) from exc

        # Ollama /api/generate (non-streaming) returns {"response": "...", ...}
        text = obj.get("response", "")
        if not isinstance(text, str):
            raise BackendInvocationError(
                f"Ollama response missing 'response' field for '{expert.name}'. "
                f"Keys present: {list(obj.keys())}"
            )

        # Surface useful eval metrics when Ollama provides them
        metadata: dict = {
            "backend": "ollama",
            "model": obj.get("model", model_tag),
            "done": obj.get("done", True),
        }
        for key in ("eval_count", "eval_duration", "prompt_eval_count", "total_duration"):
            if key in obj:
                metadata[key] = obj[key]

        return BackendResponse(text=text, latency_ms=latency_ms, metadata=metadata)

    # ------------------------------------------------------------------
    # Utility — probe Ollama health without routing a request
    # ------------------------------------------------------------------

    def ping(self, base_url: str | None = None) -> bool:
        """Return True if Ollama is reachable at base_url (or the default)."""
        target = (base_url or self._default_base_url).rstrip("/")
        req = urlrequest.Request(target + "/api/tags", method="GET")
        try:
            with urlrequest.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_local_models(self, base_url: str | None = None) -> list[str]:
        """
        Return the list of model tags currently pulled in Ollama.
        Returns [] if Ollama is unreachable.
        """
        target = (base_url or self._default_base_url).rstrip("/")
        req = urlrequest.Request(target + "/api/tags", method="GET")
        try:
            with urlrequest.urlopen(req, timeout=10) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in obj.get("models", [])]
        except Exception:
            return []
