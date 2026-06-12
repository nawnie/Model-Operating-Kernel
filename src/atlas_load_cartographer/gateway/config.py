"""Runtime configuration loader for the Atlas Load Cartographer gateway."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ExpertConfig(BaseModel):
    route: str
    adapter_path: str
    anchors: list[str] = Field(default_factory=list)
    description: str = ""


class GuardrailConfig(BaseModel):
    vram_soft_limit_gb: float = 14.0
    vram_hard_limit_gb: float = 14.5
    adapter_lru_limit: int = 3
    max_context_tokens: int = 8192
    eviction_policy: str = "lru"


class GatewayConfig(BaseModel):
    default_route: str = "default"
    telemetry_db_path: str = "data/mok_telemetry.db"
    experts: list[ExpertConfig] = Field(default_factory=list)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)

    @property
    def expert_mapping(self) -> dict[str, str]:
        return {expert.route: expert.adapter_path for expert in self.experts}

    @property
    def route_anchors(self) -> dict[str, list[str]]:
        return {expert.route: expert.anchors for expert in self.experts}


def load_gateway_config(path: str | Path = "configs/experts_v04.json") -> GatewayConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return GatewayConfig.model_validate(payload)
