from __future__ import annotations

from dataclasses import dataclass

from mok.models.registry import ExpertMetadata, ExpertState


@dataclass(slots=True)
class BudgetManager:
    ceiling_gb: float = 14.5
    landing_zone_gb: float = 3.5

    @property
    def usable_vram_gb(self) -> float:
        return self.ceiling_gb - self.landing_zone_gb

    def current_pressure_gb(self, experts: list[ExpertMetadata]) -> float:
        return round(
            sum(expert.vram_cost_gb for expert in experts if expert.is_loaded),
            3,
        )

    def can_activate(self, target: ExpertMetadata, experts: list[ExpertMetadata]) -> bool:
        projected = self.current_pressure_gb(experts)
        if not target.is_loaded:
            projected += target.vram_cost_gb
        return projected <= self.usable_vram_gb

    def propose_evictions(
        self,
        target: ExpertMetadata,
        experts: list[ExpertMetadata],
    ) -> list[str]:
        if self.can_activate(target, experts):
            return []

        projected = self.current_pressure_gb(experts)
        if not target.is_loaded:
            projected += target.vram_cost_gb

        evictable = sorted(
            (
                expert
                for expert in experts
                if expert.state == ExpertState.IDLE and expert.name != "core"
            ),
            key=lambda expert: expert.load_sequence,
        )
        evictions: list[str] = []
        for expert in evictable:
            if projected <= self.usable_vram_gb:
                break
            projected -= expert.vram_cost_gb
            evictions.append(expert.name)
        return evictions
