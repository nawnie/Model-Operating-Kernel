# Telemetry Schema

The MOK telemetry database is a local SQLite syscall trace for route execution.

## Table: route_events

```sql
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
);
```

## Operational trace query

```sql
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
ORDER BY total_invocations DESC;
```

## Why this matters

The project needs empirical proof that the MOK route layer is lightweight. SQLite telemetry lets the repo show:

- which routes are used most
- whether staging hits are happening
- routing overhead in milliseconds
- complete gateway overhead in milliseconds
- confidence trends by expert route
