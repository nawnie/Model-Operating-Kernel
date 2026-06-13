# CLAUDE.md — Start Here

Claude, this repository is ready for agentic development.

## Immediate mission

Build the first working **Model Operating Kernel** runtime loop.

MOK is a multi-model runtime. It coordinates a small resident core model with specialized expert models such as coder, instruct, vision, reasoning, memory, and tool models.

The first practical loop should be:

```text
prompt -> core chooses expert -> budget returns eviction plan -> runtime offloads idle models -> selected expert loads -> mock response returns -> trace logs
```

## Read first

1. `README.md`
2. `docs/runtime_mvp.md`
3. `AGENTS.md`
4. `src/mok/models/registry.py`
5. `src/mok/memory/budget.py`
6. `tests/test_model_registry.py`
7. `tests/test_memory_budget.py`

Read `docs/research_plan.md` only after the runtime task is understood. Do not start training work yet.

## Project boundary

This repo is MOK-only. Keep implementation under `src/mok/`.

Do not add Atlas naming, Atlas paths, or old adapter-router assumptions.

## First implementation target

Create or complete:

```text
src/mok/models/backends.py
src/mok/orchestration/__init__.py
src/mok/orchestration/runtime.py
src/mok/telemetry/__init__.py
src/mok/telemetry/events.py
configs/example_experts.json
tests/test_mock_backend.py
tests/test_orchestration_runtime.py
```

## Expected model registry concept

```json
{
  "name": "coder",
  "role": "code generation",
  "backend": "mock",
  "api_url": "mock://coder",
  "vram_cost_gb": 4.0,
  "ram_cost_gb": 6.0,
  "state": "offline",
  "current_device": "cpu",
  "pinned": false,
  "can_evict": true,
  "priority": 100
}
```

## Expected budget behavior

The budget manager returns an eviction plan. It must not mutate the registry directly.

The runtime executes the plan and only updates model state after backend offload/load succeeds.

## Expected orchestration response shape

```json
{
  "selected_expert": "coder",
  "evicted": ["vision"],
  "core_state": "resident",
  "expert_state": "idle",
  "response": "mock response from coder",
  "trace_id": "mock-trace-id"
}
```

## Suggested next branch

```bash
git checkout -b feature/runtime-mvp-mock-loop
```

## Suggested commit message

```text
Implement MOK runtime MVP mock loop
```

## Definition of done

- `pytest` passes.
- Registry tracks core and expert models.
- Budget manager protects the landing zone.
- Budget manager returns a non-mutating eviction plan.
- Resident core coordinator is preserved.
- Idle experts are selected for offload before loading a new expert.
- Mock backend proves the loop without real model weights.
- Runtime trace records route, evictions, state changes, and timing.
