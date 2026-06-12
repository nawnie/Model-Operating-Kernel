# CLAUDE.md — Start Here

Claude, this repository is ready for agentic development.

## Immediate mission

Implement **v0.5 of the Model Operating Kernel gateway**.

The repo already defines the concept and contains starter modules. Your job is to make the first practical gateway loop work cleanly:

```text
prompt → route selection → adapter RAM staging → telemetry record → JSON response
```

## Read first

1. `README.md`
2. `AGENTS.md`
3. `docs/v0_5_mok_spec.md`
4. `src/atlas_load_cartographer/gateway/preload_manager.py`
5. `src/atlas_load_cartographer/gateway/embedding_router.py`
6. `src/atlas_load_cartographer/gateway/telemetry.py`

## Do not do these things

- Do not commit model weights.
- Do not add huge dependencies casually.
- Do not assume datacenter GPUs.
- Do not turn this into LangChain.
- Do not remove telemetry.
- Do not ignore VRAM/resource constraints.

## First implementation target

Create or complete:

```text
src/atlas_load_cartographer/gateway/config.py
src/atlas_load_cartographer/gateway/app.py
configs/experts_v04.json
tests/test_embedding_router.py
tests/test_preload_manager.py
tests/test_gateway_app.py
```

## Expected `/route` response shape

```json
{
  "route": "coding",
  "embedding_score": 0.5,
  "route_ms": 0.123,
  "memory_staged_hit": true,
  "matched_terms": ["python", "code"]
}
```

## Expected `/telemetry/routes` response shape

```json
{
  "routes": [
    {
      "route": "coding",
      "total_invocations": 3,
      "routing_overhead_ms": 0.12,
      "mean_semantic_confidence": 0.5,
      "complete_execution_ms": 4.2,
      "staged_hit_rate": 1.0
    }
  ]
}
```

## Suggested next branch

```bash
git checkout -b feature/v0.5-gateway-lifecycle
```

## Suggested commit message

```text
Implement v0.5 MOK gateway lifecycle
```

## Definition of done

- `pytest` passes.
- FastAPI app starts.
- `/route` returns deterministic route data.
- Missing adapter files fail safely and are recorded in telemetry.
- Telemetry summary endpoint works.
- README/docs stay aligned with implementation.
