"""VRAM budget guardrails for split-loaded MOK expert models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from mok.models.registry import AssetState, ModelRegistry

logger = logging.getLogger("MOK.BudgetManager")


@dataclass(frozen=True)
class EvictionPlan:
    """A non-mutating allocation decision returned by the budget manager."""

    incoming_expert: str
    required_vram_gb: float
    current_pressure_gb: float
    usable_vram_gb: float
    projected_pressure_gb: float
    evictions: List[str] = field(default_factory=list)
    can_allocate: bool = True
    reason: str = ""


class BudgetManager:
    """Gatekeeper for model allocation under a protected VRAM budget.

    The budget manager does not move neural weights by itself. It decides whether
    an incoming expert can be promoted into GPU memory safely and returns an
    eviction plan for the runtime to execute.
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

    def request_allocation_clearance(self, incoming_expert_name: str) -> EvictionPlan:
        """Plan evictions required before loading a target expert.

        This method is intentionally non-mutating. It does not mark experts as
        OFFLINE. The runtime must execute the returned evictions and update the
        registry only after backend offload succeeds.
        """

        current_pressure = self.calculate_vram_pressure()
        target = self.registry.get_expert(incoming_expert_name)
        if target is None:
            return EvictionPlan(
                incoming_expert=incoming_expert_name,
                required_vram_gb=0.0,
                current_pressure_gb=current_pressure,
                usable_vram_gb=self.usable_vram_gb,
                projected_pressure_gb=current_pressure,
                can_allocate=False,
                reason="incoming expert is not registered",
            )

        if target.state in {AssetState.ACTIVE, AssetState.IDLE, AssetState.RESIDENT}:
            return EvictionPlan(
                incoming_expert=incoming_expert_name,
                required_vram_gb=0.0,
                current_pressure_gb=current_pressure,
                usable_vram_gb=self.usable_vram_gb,
                projected_pressure_gb=current_pressure,
                can_allocate=True,
                reason="incoming expert is already in GPU memory",
            )

        required_space = target.vram_cost_gb
        projected_pressure = current_pressure + required_space
        evictions: List[str] = []

        for candidate in self.registry.get_eviction_candidates():
            if projected_pressure <= self.usable_vram_gb:
                break
            evictions.append(candidate.name)
            projected_pressure -= candidate.vram_cost_gb
            logger.info("[BUDGET] Planned eviction: %s", candidate.name)

        can_allocate = projected_pressure <= self.usable_vram_gb
        if not can_allocate:
            logger.warning(
                "[BUDGET] Allocation blocked for %s. Pressure %.2fGB exceeds usable %.2fGB.",
                incoming_expert_name,
                projected_pressure,
                self.usable_vram_gb,
            )

        return EvictionPlan(
            incoming_expert=incoming_expert_name,
            required_vram_gb=required_space,
            current_pressure_gb=current_pressure,
            usable_vram_gb=self.usable_vram_gb,
            projected_pressure_gb=projected_pressure,
            evictions=evictions,
            can_allocate=can_allocate,
            reason="ok" if can_allocate else "insufficient evictable idle capacity",
        )
