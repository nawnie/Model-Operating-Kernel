from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import time

from mok.memory.budget import BudgetManager
from mok.models.backends import BackendInvocationError, ExpertBackend, RequestPayload
from mok.models.registry import ExpertState, ModelRegistry
from mok.routing.circuit_breaker import ExpertCircuitBreakerRegistry
from mok.routing.router import RouteDecision, RoutingError, RulesRouter
from mok.routing.router_r1 import ZeroShotRouter
from mok.routing.router_r2 import LearnedRouter
from mok.memory.state_bus import ExpertContext
from mok.telemetry.events import JsonlTraceLogger, TraceEvent

# RSI + Persona (optional — imported lazily to avoid hard dependency)
try:
    from mok.orchestration.consultation import ConsultationEngine, ResourceContext
    from mok.rsi.trace_accumulator import TraceAccumulator
    from mok.persona.user_profile import UserProfile
    from mok.persona.persona_adapter import PersonaAdapter
    _CONSULTATION_AVAILABLE = True
except ImportError:
    _CONSULTATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeResult:
    request_id: str
    expert_name: str
    route: RouteDecision
    text: str
    evicted: list[str]
    total_ms: int
    error: str | None = None
    # Consultation path extras (None when using legacy single-expert path)
    consultation_gate: str | None = None
    consultation_confidence: str | None = None
    quality_score: float | None = None
    rsi_accepted: bool | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class OrchestratorRuntime:
    def __init__(
        self,
        registry: ModelRegistry,
        router: RulesRouter,
        budget_manager: BudgetManager,
        backends: dict[str, ExpertBackend],
        trace_logger: JsonlTraceLogger | None = None,
        circuit_breakers: ExpertCircuitBreakerRegistry | None = None,
        r1_router: ZeroShotRouter | None = None,
        r2_router: LearnedRouter | None = None,
        consultation_engine: "ConsultationEngine | None" = None,
        trace_accumulator: "TraceAccumulator | None" = None,
    ) -> None:
        self.registry = registry
        self.router = router
        self.budget_manager = budget_manager
        self.backends = backends
        self.trace_logger = trace_logger
        self.circuit_breakers = circuit_breakers or ExpertCircuitBreakerRegistry()
        self.r1_router = r1_router  # None = R0 only; set to enable R1 escalation
        self.r2_router = r2_router  # None = no R2; set to enable learned routing
        self.consultation_engine = consultation_engine  # None = legacy path
        self.trace_accumulator = trace_accumulator      # None = no RSI logging

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        trace_path: Path | None,
        backends: dict[str, ExpertBackend],
    ) -> "OrchestratorRuntime":
        logger_inst = JsonlTraceLogger(trace_path) if trace_path else None
        return cls(
            registry=ModelRegistry.from_json(config_path),
            router=RulesRouter(),
            budget_manager=BudgetManager(),
            backends=backends,
            trace_logger=logger_inst,
        )

    def handle_request(
        self,
        payload: RequestPayload,
        context: ExpertContext | None = None,
        user_id: str = "anonymous",
        lane: str = "unknown",
    ) -> RuntimeResult:
        started = time.perf_counter()
        error_type: str | None = None
        error_msg: str | None = None
        evicted: list[str] = []
        target_name = "unknown"
        route: RouteDecision | None = None

        # ── Consultation path (RSI-enabled) ───────────────────────────────────
        if self.consultation_engine is not None and _CONSULTATION_AVAILABLE:
            try:
                c_result = self.consultation_engine.handle(
                    request_id=payload.request_id,
                    user_prompt=payload.prompt,
                    context=context,
                )
                quality_score = None
                rsi_accepted = None
                if self.trace_accumulator is not None:
                    ingest = self.trace_accumulator.ingest(c_result, user_id=user_id, lane=lane)
                    quality_score = ingest.quality_score
                    rsi_accepted = ingest.accepted
                total_ms = int((time.perf_counter() - started) * 1000)
                # Fabricate a minimal RouteDecision for compatibility
                from mok.routing.router import RouteDecision as _RD
                pseudo_route = _RD(
                    expert_name=c_result.sessions[0].expert.name if c_result.sessions else "mok_core",
                    confidence=1.0 if c_result.gate not in ("pending", "no_backend") else 0.0,
                    reason=f"consultation:{c_result.decision.value}",
                )
                return RuntimeResult(
                    request_id=payload.request_id,
                    expert_name=pseudo_route.expert_name,
                    route=pseudo_route,
                    text=c_result.final_answer,
                    evicted=[],
                    total_ms=total_ms,
                    consultation_gate=c_result.gate,
                    consultation_confidence=c_result.confidence,
                    quality_score=quality_score,
                    rsi_accepted=rsi_accepted,
                )
            except Exception as exc:
                logger.warning(
                    "[Runtime] consultation path failed (%s), falling through to legacy path", exc
                )
                # Fall through to legacy path below

        # ── Legacy single-expert path ─────────────────────────────────────────
        try:
            route = self.router.route(payload, self.registry)
            # R1 escalation: if configured and R0 confidence is low, re-route
            if self.r1_router is not None:
                route = self.r1_router.route(payload, self.registry, route)
            # R2 override: if a learned router is configured, apply it now
            if self.r2_router is not None:
                route = self.r2_router.route(payload, self.registry)
            target_name = route.expert_name

            # Circuit breaker check — reject immediately if expert is OPEN
            if not self.circuit_breakers.allow(target_name):
                error_type = "circuit_open"
                raise RoutingError(
                    f"Expert '{target_name}' is circuit-broken and not accepting requests."
                )

            target = self.registry.get(route.expert_name)

            evicted = self.budget_manager.propose_evictions(target, self.registry.all())
            for expert_name in evicted:
                self.registry.evict(expert_name)

            if not self.budget_manager.can_activate(target, self.registry.all()):
                error_type = "budget_exhausted"
                raise RuntimeError(
                    f"Unable to activate expert {target.name} within the VRAM budget."
                )

            self.registry.promote(target.name, ExpertState.ACTIVE)
            backend = self.backends.get(target.backend)
            if backend is None:
                error_type = "no_backend"
                raise RuntimeError(f"No backend registered for '{target.backend}'.")

            response = backend.generate(target, payload)
            self.circuit_breakers.success(target_name)
            self.registry.mark_idle(target.name)

        except RoutingError as exc:
            error_type = error_type or "routing_error"
            error_msg = str(exc)
            logger.error("[Runtime] Routing failed for %s: %s", payload.request_id, exc)
            total_ms = int((time.perf_counter() - started) * 1000)
            self._trace_failure(payload, route, target_name, evicted, total_ms, error_type, error_msg)
            raise

        except BackendInvocationError as exc:
            error_type = "backend_error"
            error_msg = str(exc)
            logger.error("[Runtime] Backend error for %s: %s", payload.request_id, exc)
            self.circuit_breakers.failure(target_name)
            self.registry.mark_idle(target_name)
            total_ms = int((time.perf_counter() - started) * 1000)
            self._trace_failure(payload, route, target_name, evicted, total_ms, error_type, error_msg)
            raise

        except Exception as exc:
            error_type = error_type or "internal_error"
            error_msg = str(exc)
            logger.exception("[Runtime] Unhandled error for %s", payload.request_id)
            total_ms = int((time.perf_counter() - started) * 1000)
            self._trace_failure(payload, route, target_name, evicted, total_ms, error_type, error_msg)
            raise RuntimeError(f"Runtime error [{error_type}]: {exc}") from exc

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
                    router_tier=getattr(route, "router_tier", "R0"),
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

    def _trace_failure(
        self,
        payload: RequestPayload,
        route: RouteDecision | None,
        target_name: str,
        evicted: list[str],
        total_ms: int,
        error_type: str,
        error_msg: str,
    ) -> None:
        if not self.trace_logger:
            return
        self.trace_logger.log(
            TraceEvent(
                request_id=payload.request_id,
                prompt=payload.prompt,
                modality_flags=payload.modality_flags,
                route_expert=target_name,
                route_confidence=getattr(route, "confidence", 0.0) if route else 0.0,
                route_reason=getattr(route, "reason", "unknown") if route else "routing_failed",
                router_tier=getattr(route, "router_tier", "R0") if route else "R0",
                experts_called=[target_name],
                evicted=evicted,
                total_ms=total_ms,
                backend_latency_ms=0,
                vram_pressure_gb=self.budget_manager.current_pressure_gb(self.registry.all()),
                success=False,
                error_type=error_type,
            )
        )
