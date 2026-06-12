"""Starter semantic/keyword router for the Model Operating Kernel gateway.

This is intentionally dependency-light for v0.4-alpha. It can be replaced by a
real embedding model while preserving the same route result shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class RouteResult:
    route: str
    score: float
    route_ms: float
    matched_terms: tuple[str, ...]


class EmbeddingRouter:
    """Deterministic route selector with keyword anchors and fallback routing."""

    def __init__(self, route_anchors: dict[str, list[str]], default_route: str = "default") -> None:
        self.route_anchors = {
            route: tuple(anchor.lower() for anchor in anchors)
            for route, anchors in route_anchors.items()
        }
        self.default_route = default_route

    def route(self, prompt: str) -> RouteResult:
        """Select the strongest route for a prompt.

        The score is a simple normalized anchor-hit score. This gives telemetry a
        stable field today and leaves room for true embedding cosine distance in
        v0.5+.
        """

        start = perf_counter()
        prompt_lower = prompt.lower()
        best_route = self.default_route
        best_score = 0.0
        best_terms: tuple[str, ...] = ()

        for route, anchors in self.route_anchors.items():
            if not anchors:
                continue
            matched = tuple(anchor for anchor in anchors if anchor in prompt_lower)
            score = len(matched) / len(anchors)
            if score > best_score:
                best_route = route
                best_score = score
                best_terms = matched

        route_ms = (perf_counter() - start) * 1000
        return RouteResult(
            route=best_route,
            score=round(best_score, 6),
            route_ms=round(route_ms, 6),
            matched_terms=best_terms,
        )
