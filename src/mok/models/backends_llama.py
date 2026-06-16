"""
src/mok/models/backends_llama.py

llama-cpp-python HTTP backend for the MoK expert system — P2.1.

Wraps the OpenAI-compatible REST API served by `llama-cpp-python --server`
(or any OpenAI-compat endpoint such as LM Studio, llama.cpp server, etc.).

No third-party dependencies; uses only stdlib urllib.

ExpertMetadata wiring
---------------------
- expert.api_url       : Full chat/completions endpoint URL.
                         e.g. "http://localhost:8080/v1/chat/completions"
                         Falls back to DEFAULT_CHAT_URL if None.
- expert.model_path    : Reported as the model identifier in the POST body.
                         Falls back to expert.name if None.
- expert.context_limit : Passed as "max_tokens" cap in the request.
- payload.parameters   : Forwarded as top-level JSON keys
                         (temperature, top_p, seed, stop, …).

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


DEFAULT_BASE_URL   = "http://localhost:8080"
DEFAULT_CHAT_PATH  = "/v1/chat/completions"
_TIMEOUT_SECONDS   = 120


class LlamaCppBackend:
    """
    Pure-stdlib backend for llama-cpp-python (OpenAI-compat) servers.

    One instance is safe to share across requests; it holds no mutable state.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self._default_base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # ExpertBackend protocol
    # ------------------------------------------------------------------

    def generate(self, expert: ExpertMetadata, payload: RequestPayload) -> BackendResponse:
        endpoint = (
            expert.api_url
            if expert.api_url
            else self._default_base_url + DEFAULT_CHAT_PATH
        )
        model_id = expert.model_path or expert.name

        body: dict = {
            "model": model_id,
            "messages": [{"role": "user", "content": payload.prompt}],
            "stream": False,
        }

        # Forward caller parameters (temperature, top_p, seed, stop, etc.)
        if payload.parameters:
            body.update(payload.parameters)

        # Respect context window limit as max_tokens cap
        if expert.context_limit:
            body.setdefault("max_tokens", min(expert.context_limit, 4096))

        raw = self._post(endpoint, body)

        try:
            choices = raw["choices"]
            text = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendInvocationError(
                f"LlamaCppBackend: unexpected response shape from {endpoint}: {exc}\n"
                f"raw={json.dumps(raw)[:300]}"
            ) from exc

        usage = raw.get("usage", {})
        return BackendResponse(
            text=text,
            latency_ms=0,   # caller measures wall time; llama.cpp doesn't always report it
            metadata={
                "model":         raw.get("model", model_id),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )

    # ------------------------------------------------------------------
    # Utility — health probe
    # ------------------------------------------------------------------

    def ping(self, base_url: str | None = None) -> bool:
        """
        Returns True if the server at base_url is reachable (GET /health or /v1/models).
        Never raises; returns False on any error.
        """
        base = (base_url or self._default_base_url).rstrip("/")
        for path in ("/health", "/v1/models"):
            try:
                req = urlrequest.Request(base + path, method="GET")
                with urlrequest.urlopen(req, timeout=5):
                    return True
            except Exception:
                continue
        return False

    def list_local_models(self, base_url: str | None = None) -> list[str]:
        """
        Query /v1/models and return model IDs. Returns [] on any error.
        """
        base = (base_url or self._default_base_url).rstrip("/")
        try:
            req = urlrequest.Request(base + "/v1/models", method="GET")
            with urlrequest.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            return [m["id"] for m in raw.get("data", [])]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _post(self, url: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urlrequest.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
                raw_bytes = resp.read()
        except HTTPError as exc:
            body_snip = exc.read(300).decode("utf-8", errors="replace")
            raise BackendInvocationError(
                f"LlamaCppBackend HTTP {exc.code} from {url}: {body_snip}"
            ) from exc
        except URLError as exc:
            raise BackendInvocationError(
                f"LlamaCppBackend connection error to {url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise BackendInvocationError(
                f"LlamaCppBackend timed out after {_TIMEOUT_SECONDS}s ({url})"
            ) from exc

        try:
            return json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise BackendInvocationError(
                f"LlamaCppBackend: non-JSON response from {url}: "
                f"{raw_bytes[:200].decode('utf-8', errors='replace')}"
            ) from exc
