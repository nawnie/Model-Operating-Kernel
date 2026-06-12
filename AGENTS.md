# AGENTS.md — Model Operating Kernel

This repository is intended to be worked on by AI coding agents such as Claude, Codex, GPT, and local assistants.

## Project identity

**Repo name:** Model Operating Kernel  
**Architecture category:** Model Operating Kernel (MOK)  
**Current milestone:** Phase 1 — model registry and memory budget manager

MOK is a local-first multi-model runtime. Its first goal is to make a working system that coordinates multiple specialized models — such as a core coordinator, instruct model, coder model, vision model, reasoning model, and tool model — under one shared hardware budget.

This is not an Atlas project, not a LoRA-adapter project, and not a generic agent framework. Shared ideas may exist, but this repo is new and should stand on its own.

## Core idea

MOK treats models as managed compute assets:

- **Core coordinator model**: resident model that stays loaded and decides what expert should be used.
- **Expert models**: specialized models for coding, instruction following, vision, reasoning, tools, memory, or other roles.
- **Model registry**: tracks every expert, backend, memory estimate, lifecycle state, and device location.
- **Budget manager**: protects VRAM by deciding what must be offloaded before loading a new expert.
- **Load/offload manager**: future component that will physically stage, load, unload, and route models.

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

5. **Do not turn this into a generic chat-agent framework.**
   - MOK is a runtime/control layer for model assets.

## Preferred development flow

Use small, reviewable commits:

1. Keep architecture docs and implementation aligned.
2. Add tests for every core component.
3. Prefer dependency-light modules until a real backend is required.
4. Put runtime config in `configs/`.
5. Put design docs in `docs/`.
6. Put local-only artifacts under ignored directories.

## Current high-priority tasks

### Task 1 — Finish the model registry

Work from:

```text
src/mok/models/registry.py
```

The registry should track:

- model name
- model role
- backend type
- API URL or local endpoint
- estimated VRAM cost
- estimated RAM cost
- lifecycle state
- current device

### Task 2 — Finish the budget manager

Work from:

```text
src/mok/memory/budget.py
```

The budget manager should:

1. Calculate estimated VRAM pressure.
2. Preserve a landing zone for context growth and backend overhead.
3. Refuse to evict the resident core coordinator.
4. Prefer evicting idle experts first.
5. Return the model names that must be offloaded before loading a new expert.

### Task 3 — Add mock backend clients

Create:

```text
src/mok/models/backends.py
```

The first backend layer can simulate model load latency, unload latency, and generation latency. Do not require real model weights for the first full-system test.

### Task 4 — Add the first orchestration loop

Create a simple API or CLI loop that does:

```text
prompt -> core decides route -> budget checks incoming expert -> offload idle experts -> load selected expert -> execute mock response
```

### Task 5 — Keep docs pitch-ready

The repo should remain legible to outside engineers. Update `README.md`, `docs/architecture.md`, and future specs when implementation changes.

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

Start by reading:

1. `README.md`
2. `src/mok/models/registry.py`
3. `src/mok/memory/budget.py`
4. `tests/test_model_registry.py`
5. `tests/test_memory_budget.py`

Then implement the mock backend and first orchestration loop with tests.
