from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Protocol
from urllib import request as urlrequest

from mok.models.registry import ExpertMetadata


@dataclass(slots=True)
class RequestPayload:
    prompt: str
    request_id: str = "req-1"
    modality_flags: dict[str, bool] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackendResponse:
    text: str
    latency_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BackendInvocationError(RuntimeError):
    """Raised when an expert backend cannot complete a request."""


class ExpertBackend(Protocol):
    def generate(self, expert: ExpertMetadata, payload: RequestPayload) -> BackendResponse:
        ...


class MockBackend:
    """Cheap local backend for exercising the runtime loop."""

    def generate(self, expert: ExpertMetadata, payload: RequestPayload) -> BackendResponse:
        started = time.perf_counter()
        prompt = payload.prompt.strip()
        if expert.role == "code":
            text = (
                f"[{expert.name}] Mock code specialist response.\n"
                f"Suggested next step: write a small function or test for: {prompt}"
            )
        elif expert.role == "vision":
            text = (
                f"[{expert.name}] Mock vision specialist response.\n"
                f"Observed image-oriented request: {prompt}"
            )
        elif expert.role == "coordinator":
            text = (
                f"[{expert.name}] Mock coordinator plan.\n"
                f"Plan the task, then hand off work for: {prompt}"
            )
        else:
            text = (
                f"[{expert.name}] Mock general expert response.\n"
                f"Handled request: {prompt}"
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return BackendResponse(text=text, latency_ms=latency_ms, metadata={"backend": "mock"})


class HTTPBackend:
    """HTTP adapter with a minimal JSON contract for future expert services."""

    def generate(self, expert: ExpertMetadata, payload: RequestPayload) -> BackendResponse:
        if not expert.api_url:
            raise BackendInvocationError(f"Expert {expert.name} has no api_url configured.")
        started = time.perf_counter()
        body = json.dumps(
            {
                "model_name": expert.name,
                "prompt": payload.prompt,
                "parameters": payload.parameters,
                "modality_flags": payload.modality_flags,
            }
        ).encode("utf-8")
        http_request = urlrequest.Request(
            expert.api_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(http_request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except Exception as exc:  # pragma: no cover - exercised only against real services
            raise BackendInvocationError(f"HTTP backend failed for {expert.name}: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        return BackendResponse(text=raw, latency_ms=latency_ms, metadata={"backend": "http"})
