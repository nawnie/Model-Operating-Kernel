from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TraceEvent:
    request_id: str
    prompt: str
    modality_flags: dict[str, bool]
    route_expert: str
    route_confidence: float
    route_reason: str
    experts_called: list[str]
    evicted: list[str]
    total_ms: int
    backend_latency_ms: int
    vram_pressure_gb: float
    success: bool
    # v1 additions
    router_tier: str = "R0"
    fallback_chain: list[str] = field(default_factory=list)
    error_type: str | None = None   # None | "backend_error" | "budget_exhausted" | "routing_error" | "no_backend" | "internal_error"
    prompt_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    vram_peak_gb: float | None = None  # measured peak from telemetry (P5.3)


class JsonlTraceLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: TraceEvent) -> None:
        payload = asdict(event)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
