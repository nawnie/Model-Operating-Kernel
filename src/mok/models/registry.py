"""Model registry for MOK-managed expert assets.

The registry is the source of truth for what expert models exist, what role each
model serves, what backend owns it, and where the model currently lives in the
local memory lifecycle.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AssetState(str, Enum):
    """Hardware lifecycle state for a managed model asset."""

    OFFLINE = "offline"  # On disk.
    STAGED = "staged"  # Paged into system RAM.
    RESIDENT = "resident"  # Permanent in VRAM, reserved for the core coordinator.
    ACTIVE = "active"  # In VRAM and currently executing.
    IDLE = "idle"  # In VRAM but waiting; eligible for eviction.


class ExpertMetadata(BaseModel):
    """Metadata and live placement state for one expert model."""

    name: str
    role: str
    backend: str
    api_url: str
    vram_cost_gb: float = Field(..., description="Estimated VRAM footprint when active")
    ram_cost_gb: float = Field(..., description="Estimated system RAM footprint when staged")
    state: AssetState = AssetState.OFFLINE
    current_device: str = "cpu"


class ModelRegistry:
    """In-memory registry for expert model metadata and lifecycle state."""

    def __init__(self) -> None:
        self._experts: Dict[str, ExpertMetadata] = {}

    def register_expert(self, expert: ExpertMetadata) -> None:
        self._experts[expert.name] = expert

    def get_expert(self, name: str) -> Optional[ExpertMetadata]:
        return self._experts.get(name)

    def list_experts(self) -> List[ExpertMetadata]:
        return list(self._experts.values())

    def get_experts_by_state(self, state: AssetState) -> List[ExpertMetadata]:
        return [expert for expert in self._experts.values() if expert.state == state]

    def update_state(self, name: str, state: AssetState, device: str) -> None:
        if name in self._experts:
            self._experts[name].state = state
            self._experts[name].current_device = device
