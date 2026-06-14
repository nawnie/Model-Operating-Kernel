from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from mok.memory.budget import BudgetManager
from mok.models.backends import ExpertBackend, RequestPayload
from mok.models.registry import ExpertState, ModelRegistry
from mok.routing.router import RouteDecision, RulesRouter
from mok.telemetry.events import JsonlTraceLogger, TraceEvent


@dataclass(slots=True)
class RuntimeResult:
    request_id: str
    expert_name: str
    route: RouteDecision
    text: str
    evicted: list[str]
    total_ms: int


class OrchestratorRuntime:
    def __init__(
        self,
        registry: ModelRegistry,
        router: RulesRouter,
        budget_manager: BudgetManager,
        backends: dict[str, ExpertBackend],
        trace_logger: JsonlTraceLogger | None = None,
    ) -> None:
        self.registry = registry
        self.router = router
        self.budget_manager = budget_manager
        self.backends = backends
        self.trace_logger = trace_logger

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        trace_path: Path | None,
        backends: dict[str, ExpertBackend],
    ) -> "OrchestratorRuntime":
        logger = JsonlTraceLogger(trace_path) if trace_path else None
        return cls(
            registry=ModelRegistry.from_json(config_path),
            router=RulesRouter(),
            budget_manager=BudgetManager(),
            backends=backends,
            trace_logger=logger,
        )

    def handle_request(self, payload: RequestPayload) -> RuntimeResult:
        started = time.perf_counter()
        route = self.router.route(payload, self.registry)
        target = self.registry.get(route.expert_name)

        evicted = self.budget_manager.propose_evictions(target, self.registry.all())
        for expert_name in evicted:
            self.registry.evict(expert_name)

        if not self.budget_manager.can_activate(target, self.registry.all()):
            raise RuntimeError(f"Unable to activate expert {target.name} within the VRAM budget.")

        self.registry.promote(target.name, ExpertState.ACTIVE)
        backend = self.backends.get(target.backend)
        if backend is None:
            raise RuntimeError(f"No backend registered for {target.backend}.")
        response = backend.generate(target, payload)
        self.registry.mark_idle(target.name)

        total_ms = int((time.perf_counter() - started) * 1000)
        if self.trace_logger:
            self.trace_logger.log(
                TraceEvent(
                    request_id=payload.request_id,
                    prompt=payload.prompt,
                    modality_flags=payload.modality_flags,
                    route_expert=route.expert_name,
                    route_confidence=route.confidence,
                    route_reason=route.reason,
                    experts_called=[target.name],
                    evicted=evicted,
                    total_ms=total_ms,
                    backend_latency_ms=response.latency_ms,
                    vram_pressure_gb=self.budget_manager.current_pressure_gb(self.registry.all()),
                    success=True,
                )
            )
        return RuntimeResult(
            request_id=payload.request_id,
            expert_name=target.name,
            route=route,
            text=response.text,
            evicted=evicted,
            total_ms=total_ms,
        )
