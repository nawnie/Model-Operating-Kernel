"""Model registry for MOK-managed expert assets.

The registry is the source of truth for what expert models exist, what role each
model serves, what backend owns it, and where the model currently lives in the
local memory lifecycle.
"""

from __future__ import annotations

from enum import Enum
from time import time
from typing import Any, Dict, List, Optional

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

    # Runtime scheduling metadata. These values let the budget manager make
    # eviction decisions without guessing or mutating state too early.
    pinned: bool = False
    can_evict: bool = True
    priority: int = 100
    loaded_at: Optional[float] = None
    last_used_at: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelRegistry:
    """In-memory registry for expert model metadata and lifecycle state."""

    def __init__(self) -> None:
        self._experts: Dict[str, ExpertMetadata] = {}

    def register_expert(self, expert: ExpertMetadata) -> None:
        """Register or replace an expert.

        RESIDENT experts are treated as protected core assets by default. They
        are pinned and marked non-evictable unless the caller explicitly changes
        them later.
        """

        if expert.state == AssetState.RESIDENT:
            expert.pinned = True
            expert.can_evict = False
            if expert.loaded_at is None:
                expert.loaded_at = time()
            if expert.last_used_at is None:
                expert.last_used_at = expert.loaded_at
        self._experts[expert.name] = expert

    def get_expert(self, name: str) -> Optional[ExpertMetadata]:
        return self._experts.get(name)

    def list_experts(self) -> List[ExpertMetadata]:
        return list(self._experts.values())

    def get_experts_by_state(self, state: AssetState) -> List[ExpertMetadata]:
        return [expert for expert in self._experts.values() if expert.state == state]

    def update_state(self, name: str, state: AssetState, device: str) -> None:
        """Update lifecycle state and device placement for an expert."""

        expert = self._experts.get(name)
        if expert is None:
            return

        now = time()
        expert.state = state
        expert.current_device = device

        if state in {AssetState.RESIDENT, AssetState.ACTIVE, AssetState.IDLE}:
            if expert.loaded_at is None:
                expert.loaded_at = now
            expert.last_used_at = now

        if state == AssetState.OFFLINE:
            expert.loaded_at = None
            expert.current_device = "cpu"

        if state == AssetState.RESIDENT:
            expert.pinned = True
            expert.can_evict = False

    def mark_used(self, name: str) -> None:
        """Refresh an expert's last-used timestamp without changing its state."""

        expert = self._experts.get(name)
        if expert is not None:
            expert.last_used_at = time()

    def get_eviction_candidates(self) -> List[ExpertMetadata]:
        """Return IDLE experts that may be evicted, oldest and lowest priority first."""

        candidates = [
            expert
            for expert in self.get_experts_by_state(AssetState.IDLE)
            if expert.can_evict and not expert.pinned
        ]
        return sorted(
            candidates,
            key=lambda expert: (
                expert.priority,
                expert.last_used_at if expert.last_used_at is not None else 0.0,
                expert.name,
            ),
        )
