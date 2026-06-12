# CLAUDE.md — Start Here

Claude, this repository is ready for agentic development.

## Immediate mission

Build the first working **Model Operating Kernel** runtime loop.

MOK is a multi-model runtime. It coordinates a small resident core model with specialized expert models such as coder, instruct, vision, reasoning, memory, and tool models.

The first practical loop should be:

```text
prompt -> core chooses expert -> budget checks VRAM -> idle models offload -> selected expert loads -> mock response returns
```

## Read first

1. `README.md`
2. `AGENTS.md`
3. `src/mok/models/registry.py`
4. `src/mok/memory/budget.py`
5. `tests/test_model_registry.py`
6. `tests/test_memory_budget.py`

## Project boundary

This repo is MOK-only. Keep implementation under `src/mok/`.

## First implementation target

Create or complete:

```text
src/mok/models/backends.py
src/mok/orchestration/__init__.py
src/mok/orchestration/runtime.py
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
  "current_device": "cpu"
}
```

## Expected orchestration response shape

```json
{
  "selected_expert": "coder",
  "evicted": ["vision"],
  "core_state": "resident",
  "expert_state": "active",
  "response": "mock response from coder"
}
```

## Suggested next branch

```bash
git checkout -b feature/phase1-multi-model-runtime
```

## Suggested commit message

```text
Implement phase 1 MOK multi-model runtime loop
```

## Definition of done

- `pytest` passes.
- Registry tracks core and expert models.
- Budget manager protects the landing zone.
- Resident core coordinator is preserved.
- Idle experts are selected for offload before loading a new expert.
- Mock backend proves the loop without real model weights.
