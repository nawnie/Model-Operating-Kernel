# AGENTS.md — Model Operating Kernel

This repository is intended to be worked on by AI coding agents such as Claude, Codex, GPT, and local assistants.

## Project identity

**Repo name:** Model Operating Kernel  
**Architecture category:** Model Operating Kernel (MOK)  
**Current milestone:** Runtime MVP — registry, budget planning, mock backends, and orchestration loop

MOK is a local-first multi-model runtime. Its first goal is to make a working system that coordinates multiple specialized models — such as a core coordinator, instruct model, coder model, vision model, reasoning model, and tool model — under one shared hardware budget.

This is not an Atlas project, not a LoRA-adapter project, and not a generic agent framework. Shared ideas may exist, but this repo is new and should stand on its own.

## Read order

1. `README.md`
2. `docs/runtime_mvp.md`
3. `src/mok/models/registry.py`
4. `src/mok/memory/budget.py`
5. `tests/test_model_registry.py`
6. `tests/test_memory_budget.py`
7. `docs/research_plan.md` only after the runtime task is understood

## Build track vs research track

### Build now

- model registry
- non-mutating budget plans
- mock backends
- runtime orchestrator
- telemetry trace
- API or CLI loop

### Do later

- trained router
- trained coordinator
- LoRA/adapters as the main expert strategy
- oracle evaluation harness
- memory-policy learning
- benchmark report

Research can improve the runtime later, but it must not block the first working loop.

## Core idea

MOK treats models as managed compute assets:

- **Core coordinator model**: resident model that stays loaded and decides what expert should be used.
- **Expert models**: specialized models for coding, instruction following, vision, reasoning, tools, memory, or other roles.
- **Model registry**: tracks every expert, backend, memory estimate, lifecycle state, and device location.
- **Budget manager**: protects VRAM by returning an eviction plan before loading a new expert.
- **Load/offload manager**: future component that physically stages, loads, unloads, and routes models.

## Hardware target

Default target is a consumer workstation, especially:

- 16GB VRAM GPU
- 32GB system RAM
- Windows 11 / WSL-friendly workflow
- local-first inference stack

Agents must avoid assuming A100/H100-class memory, cluster infrastructure, or unlimited context.

## Non-negotiable constraints

1. **Do not commit model weights or checkpoints.**
   - `models/`, `.models/`, `data/`, `*.gguf`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.ckpt`, and DB files are ignored for a reason.

2. **MOK must remain multi-model, not Atlas.**
   - Do not add Atlas naming, Atlas package paths, or Atlas-specific assumptions.
   - Do not frame the system as a single adapter router.

3. **Consumer hardware first.**
   - Any feature that risks uncontrolled VRAM growth needs a guardrail, config limit, or explicit TODO.

4. **Split loading and offloading are first-class.**
   - The system must track what is offline, staged in RAM, resident in VRAM, active, or idle.

5. **Budget planning must not lie to the registry.**
   - The budget manager returns an eviction plan.
   - The runtime updates registry state only after backend offload/load succeeds.

6. **Do not turn this into a generic chat-agent framework.**
   - MOK is a runtime/control layer for model assets.

## Current high-priority tasks

### Task 1 — Add mock backend clients

Create:

```text
src/mok/models/backends.py
```

The first backend layer can simulate model load latency, unload latency, generation latency, and failures. Do not require real model weights for the first full-system test.

### Task 2 — Add the first orchestration loop

Create:

```text
src/mok/orchestration/runtime.py
```

The first loop should do:

```text
prompt -> core decides route -> budget plans clearance -> runtime offloads idle experts -> runtime loads selected expert -> mock response returns -> trace logs
```

### Task 3 — Add telemetry trace

Create:

```text
src/mok/telemetry/events.py
```

Record route, model states, evictions, load time, execution time, memory pressure, and success/failure.

### Task 4 — Add CLI or FastAPI entrypoint

Create one minimal entrypoint after the runtime loop works:

```text
src/mok/api/app.py
```

or

```text
src/mok/cli.py
```

### Task 5 — Keep docs aligned

Update `docs/runtime_mvp.md` when runtime behavior changes. Update `docs/research_plan.md` only for later training/eval work.

## Style rules

- Use Python 3.10+.
- Prefer type hints.
- Keep module boundaries clean.
- Use `pathlib.Path` for filesystem paths.
- Avoid global mutable runtime state unless it is explicitly scoped.
- Do not bury errors. Resource failures should become observable.
- Write comments that explain resource-management intent, not obvious Python mechanics.

## Commands

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest
```

## Agent handoff summary

Implement the mock backend and first orchestration loop with tests. Do not start training work until the runtime MVP completes.
