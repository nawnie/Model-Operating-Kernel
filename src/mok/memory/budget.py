"""VRAM budget guardrails for split-loaded MOK expert models."""

from __future__ import annotations

import logging
from typing import List

from mok.models.registry import AssetState, ModelRegistry

logger = logging.getLogger("MOK.BudgetManager")


class BudgetManager:
    """Gatekeeper for model allocation under a protected VRAM budget.

    The budget manager does not move neural weights by itself. It decides whether
    an incoming expert can be promoted into GPU memory safely and returns the
    experts that must be evicted first.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        max_vram_gb: float = 14.5,
        landing_zone_gb: float = 3.5,
    ) -> None:
        """Create a budget manager with a reserved landing zone.

        The landing zone prevents the runtime from filling VRAM completely. It is
        reserved for context growth, backend overhead, buffers, and temporary
        allocations that appear during real inference.
        """

        if max_vram_gb <= 0:
            raise ValueError("max_vram_gb must be greater than zero")
        if landing_zone_gb < 0:
            raise ValueError("landing_zone_gb cannot be negative")
        if landing_zone_gb >= max_vram_gb:
            raise ValueError("landing_zone_gb must be smaller than max_vram_gb")

        self.registry = registry
        self.max_vram_gb = max_vram_gb
        self.landing_zone_gb = landing_zone_gb
        self.usable_vram_gb = max_vram_gb - landing_zone_gb

    def calculate_vram_pressure(self) -> float:
        """Sum the estimated VRAM cost of models currently in GPU memory."""

        loaded_experts = (
            self.registry.get_experts_by_state(AssetState.ACTIVE)
            + self.registry.get_experts_by_state(AssetState.IDLE)
            + self.registry.get_experts_by_state(AssetState.RESIDENT)
        )
        return sum(expert.vram_cost_gb for expert in loaded_experts)

    def request_allocation_clearance(self, incoming_expert_name: str) -> List[str]:
        """Return expert names that must be evicted before loading a target expert.

        The current v0.1 strategy evicts IDLE experts first. RESIDENT experts are
        never selected for eviction because they represent the core coordinator.
        ACTIVE experts are not evicted in this first pass because that would
        require request cancellation or scheduling support.
        """

        target = self.registry.get_expert(incoming_expert_name)
        if target is None:
            return []

        if target.state in {AssetState.ACTIVE, AssetState.IDLE, AssetState.RESIDENT}:
            return []

        required_space = target.vram_cost_gb
        eviction_list: List[str] = []

        while (self.calculate_vram_pressure() + required_space) > self.usable_vram_gb:
            idle_candidates = self.registry.get_experts_by_state(AssetState.IDLE)

            if not idle_candidates:
                logger.warning(
                    "[BUDGET] Critical memory saturation. No IDLE experts left to evict."
                )
                break

            evict_target = idle_candidates[0]
            eviction_list.append(evict_target.name)
            self.registry.update_state(evict_target.name, AssetState.OFFLINE, "cpu")
            logger.info("[BUDGET] Scheduled eviction flag for: %s", evict_target.name)

        return eviction_list
