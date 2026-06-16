from __future__ import annotations

from dataclasses import dataclass

from mok.models.registry import ExpertMetadata, ExpertState


# ---------------------------------------------------------------------------
# Adapter slot descriptor (P3.4)
# ---------------------------------------------------------------------------

@dataclass
class AdapterSlot:
    """
    Lightweight descriptor for a resident LoRA adapter.

    adapter_id   : unique name, e.g. "coder-lora"
    role         : expert role this adapter serves
    vram_cost_gb : VRAM consumed while the adapter is loaded
    load_seq     : monotonic counter — lower = loaded earlier (evict first)
    """
    adapter_id: str
    role: str
    vram_cost_gb: float = 0.25
    load_seq: int = 0


@dataclass(slots=True)
class BudgetManager:
    ceiling_gb: float = 14.5
    landing_zone_gb: float = 3.5

    @property
    def usable_vram_gb(self) -> float:
        return self.ceiling_gb - self.landing_zone_gb

    @staticmethod
    def _effective_vram(expert: ExpertMetadata) -> float:
        """Use measured_peak_gb from VRAMProfile when available."""
        if expert.vram_profile is not None:
            return expert.vram_profile.effective_gb
        return expert.vram_cost_gb

    def current_pressure_gb(self, experts: list[ExpertMetadata]) -> float:
        return round(
            sum(self._effective_vram(e) for e in experts if e.is_loaded),
            3,
        )

    def can_activate(self, target: ExpertMetadata, experts: list[ExpertMetadata]) -> bool:
        projected = self.current_pressure_gb(experts)
        if not target.is_loaded:
            projected += self._effective_vram(target)
        return projected <= self.usable_vram_gb

    def propose_evictions(
        self,
        target: ExpertMetadata,
        experts: list[ExpertMetadata],
        prefetch_hints: list[str] | None = None,
    ) -> list[str]:
        """
        Determine which experts to evict to make room for *target*.

        Parameters
        ----------
        target          : expert to activate
        experts         : current expert roster from the registry
        prefetch_hints  : expert names predicted to be needed soon
                          (from NextExpertPredictor.predict).
                          These experts are deprioritised for eviction —
                          they are only evicted if no other IDLE expert
                          can free enough VRAM.

        Returns [] when target is already loaded or fits in the budget.
        """
        if self.can_activate(target, experts):
            return []

        projected = self.current_pressure_gb(experts)
        if not target.is_loaded:
            projected += self._effective_vram(target)

        protected = set(prefetch_hints) if prefetch_hints else set()

        # Two-pass eviction: non-hinted first, hinted last
        idle_experts = [
            expert
            for expert in experts
            if expert.state == ExpertState.IDLE and expert.name != "core"
        ]
        non_hinted = sorted(
            (e for e in idle_experts if e.name not in protected),
            key=lambda e: e.load_sequence,
        )
        hinted = sorted(
            (e for e in idle_experts if e.name in protected),
            key=lambda e: e.load_sequence,
        )

        evictions: list[str] = []
        for expert in non_hinted + hinted:
            if projected <= self.usable_vram_gb:
                break
            projected -= self._effective_vram(expert)
            evictions.append(expert.name)
        return evictions

    # ------------------------------------------------------------------
    # Adapter-slot management (P3.4)
    # ------------------------------------------------------------------

    def propose_adapter_swap(
        self,
        current_adapters: list[AdapterSlot],
        target_adapter: AdapterSlot,
        max_slots: int = 4,
    ) -> list[str]:
        """
        Determine which currently-loaded adapters to evict to make room
        for target_adapter, respecting the slot cap.

        Returns a list of adapter_ids to evict (oldest first).
        Returns [] if target is already loaded or there is a free slot.
        """
        # Already loaded?
        if any(a.adapter_id == target_adapter.adapter_id for a in current_adapters):
            return []

        # Slot available?
        if len(current_adapters) < max_slots:
            return []

        # Evict oldest first (lowest load_seq)
        sorted_by_age = sorted(current_adapters, key=lambda a: a.load_seq)
        slots_to_free = len(current_adapters) - max_slots + 1
        return [a.adapter_id for a in sorted_by_age[:slots_to_free]]

    def adapter_vram_cost(
        self,
        current_adapters: list[AdapterSlot],
        base_vram_cost_gb: float = 4.5,
    ) -> float:
        """Total VRAM: base model + all resident adapters."""
        return base_vram_cost_gb + sum(a.vram_cost_gb for a in current_adapters)
