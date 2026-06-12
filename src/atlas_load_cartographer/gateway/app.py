"""FastAPI gateway prototype for the Model Operating Kernel."""

from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI
from pydantic import BaseModel

from atlas_load_cartographer.gateway.config import GatewayConfig, load_gateway_config
from atlas_load_cartographer.gateway.embedding_router import EmbeddingRouter
from atlas_load_cartographer.gateway.preload_manager import PreloadManager
from atlas_load_cartographer.gateway.telemetry import RouteEvent, TelemetryStore


class RouteRequest(BaseModel):
    prompt: str


class RouteResponse(BaseModel):
    route: str
    embedding_score: float
    route_ms: float
    memory_staged_hit: bool
    matched_terms: list[str]


def build_app(config: GatewayConfig | None = None) -> FastAPI:
    """Build the MOK gateway app with explicit dependency wiring."""

    gateway_config = config or load_gateway_config()
    router = EmbeddingRouter(
        route_anchors=gateway_config.route_anchors,
        default_route=gateway_config.default_route,
    )
    preload_manager = PreloadManager(gateway_config.expert_mapping)
    telemetry = TelemetryStore(gateway_config.telemetry_db_path)

    app = FastAPI(
        title="Model Operating Kernel Gateway",
        version="0.4.0-alpha",
        description="Atlas Load Cartographer gateway for route selection, adapter staging, and telemetry.",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "kernel": "model-operating-kernel"}

    @app.post("/route", response_model=RouteResponse)
    async def route_prompt(request: RouteRequest) -> RouteResponse:
        total_start = perf_counter()
        route_result = router.route(request.prompt)
        staged = await preload_manager.stage_expert_to_ram(route_result.route)
        total_ms = (perf_counter() - total_start) * 1000

        telemetry.record_route_event(
            RouteEvent(
                route=route_result.route,
                success=True,
                route_ms=route_result.route_ms,
                total_ms=round(total_ms, 6),
                embedding_score=route_result.score,
                memory_staged_hit=staged,
                prompt_chars=len(request.prompt),
            )
        )

        return RouteResponse(
            route=route_result.route,
            embedding_score=route_result.score,
            route_ms=route_result.route_ms,
            memory_staged_hit=staged,
            matched_terms=list(route_result.matched_terms),
        )

    @app.get("/telemetry/routes")
    async def route_telemetry() -> dict[str, list[dict[str, float | int | str]]]:
        return {"routes": telemetry.route_summary()}

    return app


app = build_app()
