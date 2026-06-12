# AGENTS.md — Model Operating Kernel

This repository is intended to be worked on by AI coding agents such as Claude, Codex, GPT, and local assistants.

## Project identity

**Repo name:** Model Operating Kernel  
**Architecture category:** Model Operating Kernel (MOK)  
**First implementation:** Atlas Load Cartographer  
**Current milestone:** v0.4-alpha → v0.5 gateway integration

MOK is not a normal LLM app, not a generic agent framework, and not a LangChain-style wrapper. It is a local-first orchestration kernel for multi-expert model execution under hard consumer-hardware constraints.

The core metaphor is traditional operating-system engineering applied to local AI inference:

- Base model = resident kernel space / hardware backbone
- LoRA adapter = lightweight thread/expert execution context
- KV-cache = virtual memory / page table
- Router = syscall dispatcher / interrupt front-end
- Preload manager = disk I/O and RAM staging scheduler
- SQLite telemetry = operational trace log
- VRAM ceiling = memory protection boundary

## Hardware target

Default target is a consumer workstation, especially:

- 16GB VRAM GPU
- 32GB system RAM
- Windows 11 / WSL-friendly workflow
- Local-first inference stack

Agents must avoid assuming A100/H100-class memory, cluster infrastructure, or unlimited context.

## Non-negotiable constraints

1. **Do not commit model weights or checkpoints.**
   - `models/`, `adapters/`, `.models/`, `data/`, `*.gguf`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.ckpt`, and DB files are ignored for a reason.

2. **Keep the repo runnable before SGLang is fully integrated.**
   - The router and telemetry layers should work with lightweight local tests.
   - SGLang integration should be added behind a client abstraction.

3. **Consumer hardware first.**
   - Any feature that risks uncontrolled VRAM growth needs a guardrail, config limit, or explicit TODO.

4. **Telemetry is proof.**
   - Every route, staged-hit, failure, and timing event should become measurable.

5. **Do not turn this into a generic agent framework.**
   - MOK is a resource manager and execution gateway, not a chat personality system.

## Preferred development flow

Use small, reviewable commits:

1. Keep architecture docs and implementation aligned.
2. Add tests for every gateway component.
3. Prefer dependency-light modules until a real backend is required.
4. Put runtime config in `configs/`.
5. Put design docs in `docs/`.
6. Put local-only artifacts under ignored directories.

## Current high-priority tasks

### Task 1 — Finish v0.5 gateway lifecycle

Wire these modules together:

- `EmbeddingRouter`
- `PreloadManager`
- `TelemetryStore`
- FastAPI `/route` endpoint
- FastAPI `/telemetry/routes` endpoint

Expected behavior:

1. Receive prompt.
2. Route prompt.
3. Attempt async RAM staging for selected expert.
4. Record route timing, embedding score, staged-hit flag, prompt size, and success.
5. Return route result JSON.

### Task 2 — Add SGLang client shim

Create a thin backend abstraction, likely:

```text
src/atlas_load_cartographer/backends/
├── __init__.py
├── base.py
└── sglang_client.py
```

The v0.5 version may be a stub or HTTP client wrapper. Do not hardwire SGLang into the router.

### Task 3 — Add VRAM guardrail config

Add config fields for:

- `vram_soft_limit_gb`
- `vram_hard_limit_gb`
- `adapter_lru_limit`
- `max_context_tokens`
- `eviction_policy`

Do not implement unsafe GPU operations without measurement.

### Task 4 — Benchmark route overhead

Add a script that compares:

- router overhead
- preload staging time
- staged-hit rate
- fake/full backend latency placeholder

Target file:

```text
scripts/benchmark_gateway.py
```

### Task 5 — Keep docs pitch-ready

The repo should remain legible to outside engineers. Update `README.md`, `docs/architecture.md`, and `docs/v0_5_mok_spec.md` when implementation changes.

## Style rules

- Use Python 3.10+.
- Prefer type hints.
- Keep module boundaries clean.
- Use `pathlib.Path` for filesystem paths.
- Avoid global mutable runtime state unless guarded by FastAPI lifecycle setup.
- Do not bury errors. Route failures should become telemetry events.
- Write comments that explain kernel/resource-management intent, not obvious Python mechanics.

## Commands

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest
uvicorn atlas_load_cartographer.gateway.app:app --reload --port 8787
```

## Agent handoff summary

Start by reading:

1. `README.md`
2. `docs/v0_5_mok_spec.md`
3. `src/atlas_load_cartographer/gateway/preload_manager.py`
4. `src/atlas_load_cartographer/gateway/embedding_router.py`
5. `src/atlas_load_cartographer/gateway/telemetry.py`

Then implement the v0.5 gateway lifecycle and open a PR with tests.
