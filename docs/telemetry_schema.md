# Telemetry Schema

MOK telemetry should measure model lifecycle events, memory pressure, route choices, and backend execution outcomes.

This schema is intentionally early. It should evolve as the runtime grows.

## Table: model_events

```sql
CREATE TABLE IF NOT EXISTS model_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    event_type TEXT NOT NULL,
    model_name TEXT NOT NULL,
    role TEXT,
    previous_state TEXT,
    next_state TEXT,
    previous_device TEXT,
    next_device TEXT,
    vram_pressure_gb REAL,
    usable_vram_gb REAL,
    success INTEGER NOT NULL,
    error TEXT
);
```

## Table: orchestration_events

```sql
CREATE TABLE IF NOT EXISTS orchestration_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    prompt_chars INTEGER DEFAULT 0,
    selected_expert TEXT NOT NULL,
    evicted_models TEXT,
    route_ms REAL,
    load_ms REAL,
    execution_ms REAL,
    total_ms REAL,
    success INTEGER NOT NULL,
    error TEXT
);
```

## Operational trace query

```sql
SELECT
    selected_expert,
    COUNT(*) as total_invocations,
    AVG(route_ms) as avg_route_ms,
    AVG(load_ms) as avg_load_ms,
    AVG(execution_ms) as avg_execution_ms,
    AVG(total_ms) as avg_total_ms
FROM orchestration_events
WHERE success = 1
GROUP BY selected_expert
ORDER BY total_invocations DESC;
```

## Why this matters

The project needs empirical proof that the runtime can coordinate multiple models without losing control of memory.

Telemetry should show:

- which experts are used most
- which models are evicted most often
- how close the system gets to the usable VRAM ceiling
- how much time routing, loading, execution, and offloading cost
- whether the resident core coordinator stays protected
