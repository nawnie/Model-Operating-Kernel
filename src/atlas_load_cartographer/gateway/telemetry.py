"""SQLite telemetry for Model Operating Kernel route events."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from time import time


@dataclass(frozen=True)
class RouteEvent:
    route: str
    success: bool
    route_ms: float
    total_ms: float
    embedding_score: float | None = None
    memory_staged_hit: bool = False
    prompt_chars: int = 0
    error: str | None = None


class TelemetryStore:
    """Small SQLite-backed syscall trace for MOK route execution."""

    def __init__(self, db_path: str | Path = "data/mok_telemetry.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS route_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    route TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    route_ms REAL NOT NULL,
                    total_ms REAL NOT NULL,
                    embedding_score REAL,
                    memory_staged_hit INTEGER DEFAULT 0,
                    prompt_chars INTEGER DEFAULT 0,
                    error TEXT
                )
                """
            )
            conn.commit()

    def record_route_event(self, event: RouteEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO route_events (
                    created_at,
                    route,
                    success,
                    route_ms,
                    total_ms,
                    embedding_score,
                    memory_staged_hit,
                    prompt_chars,
                    error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time(),
                    event.route,
                    int(event.success),
                    event.route_ms,
                    event.total_ms,
                    event.embedding_score,
                    int(event.memory_staged_hit),
                    event.prompt_chars,
                    event.error,
                ),
            )
            conn.commit()

    def route_summary(self) -> list[dict[str, float | int | str]]:
        query = """
        SELECT
            route,
            COUNT(*) as total_invocations,
            AVG(route_ms) as routing_overhead_ms,
            AVG(embedding_score) as mean_semantic_confidence,
            AVG(total_ms) as complete_execution_ms,
            AVG(memory_staged_hit) as staged_hit_rate
        FROM route_events
        WHERE success = 1
        GROUP BY route
        ORDER BY total_invocations DESC
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]
