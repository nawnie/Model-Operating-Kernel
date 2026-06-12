# Model Operating Kernel (MOK)

> A lightweight, local-first kernel pattern for routing, staging, and protecting multi-expert LLM execution on consumer hardware.

**Model Operating Kernel** is the architecture layer for running dynamic multi-expert LLM pipelines inside strict local hardware limits. Instead of treating models as remote API abstractions, MOK treats base models, LoRA adapters, KV-caches, vector routes, telemetry, and tool inputs as volatile compute resources that must be scheduled, context-switched, staged, isolated, and governed.

The first implementation in this repository is **Atlas Load Cartographer**, a 16GB-VRAM-oriented prototype for safe adapter routing, async preload staging, SQLite telemetry, and local inference gateway control.

---

## Why this exists

Most AI orchestration software falls into two extremes:

1. **Over-abstracted agent frameworks** that treat models like black-box APIs and add heavy coordination overhead.
2. **Hard-silicon serving stacks** that optimize GPU execution but assume large production cards, clusters, or datacenter-style deployment.

MOK fills the lane between them: a resource-protective orchestration kernel for consumer desktop systems.

It borrows from traditional kernel design:

- **Scheduling**: decide which model route should run and when.
- **Context switching**: treat adapters as lightweight expert execution contexts.
- **Virtual memory**: treat KV-cache and prompt state like pinned page references.
- **Interrupt handling**: prepare future mid-generation route switches.
- **Memory protection**: enforce VRAM ceilings before instability starts.
- **Telemetry**: record proof that routing overhead is cheaper than full-model swapping.

---

## Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │                    USER / CONSUMER LAYER               │
       │               (Chat UI, Workflows, API Requests)       │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                       MODEL OPERATING KERNEL (MOK)                          │
 │                                                                             │
 │  ┌────────────────────────┐ ┌────────────────────────┐ ┌─────────────────┐  │
 │  │    SCHEDULING LAYER    │ │     ROUTING KERNEL     │ │ MEMORY VIRTUAL. │  │
 │  │ (Batching, Prefill)    │ │ (Regex / Embed Anchors)│ │ (KV Cache Map)  │  │
 │  └────────────────────────┘ └────────────────────────┘ └─────────────────┘  │
 │  ┌───────────────────────────────────────────────────────────────────────┐  │
 │  │                     RESOURCE ISOLATION & GOVERNANCE                   │  │
 │  │          (VRAM Floor Protection, Land-Zone Cushioning, Telemetry)     │  │
 │  └───────────────────────────────────────────────────────────────────────┘  │
 └─────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                  RAW EXECUTION HARDWARE                │
       │            (SGLang/vLLM layer, VRAM, GPU Cores)        │
       └────────────────────────────────────────────────────────┘
```

---

## Current status

**Version:** `0.4-alpha`  
**Hardware target:** local consumer workstation, especially 16GB VRAM cards  
**Primary implementation:** Atlas Load Cartographer  
**Runtime stance:** local-first, adapter-aware, telemetry-driven

This repository currently contains the day-one architecture scaffold:

- MOK README / manifesto
- Python package skeleton
- async adapter preload manager
- deterministic starter embedding/keyword router
- SQLite route telemetry helpers
- FastAPI gateway prototype
- config schema for local experts
- docs for v0.5 preload lifecycle and telemetry

---

## Repository layout

```text
Model-Operating-Kernel/
├── .gitignore
├── LICENSE
├── README.md
├── AGENTS.md
├── configs/
│   └── experts_v04.json
├── docs/
│   ├── architecture.md
│   ├── telemetry_schema.md
│   └── v0_5_mok_spec.md
├── src/
│   └── atlas_load_cartographer/
│       ├── __init__.py
│       └── gateway/
│           ├── __init__.py
│           ├── app.py
│           ├── config.py
│           ├── embedding_router.py
│           ├── preload_manager.py
│           └── telemetry.py
└── tests/
    ├── test_embedding_router.py
    └── test_preload_manager.py
```

Runtime-heavy directories such as `models/`, `adapters/`, `.models/`, `data/`, and checkpoint formats are intentionally ignored.

---

## Quick start

```powershell
# Clone
 git clone https://github.com/nawnie/Model-Operating-Kernel.git
 cd Model-Operating-Kernel

# Create venv
 py -3.11 -m venv .venv
 .\.venv\Scripts\Activate.ps1

# Install package for development
 pip install -e .[dev]

# Run tests
 pytest

# Launch the gateway prototype
 uvicorn atlas_load_cartographer.gateway.app:app --reload --port 8787
```

Then test routing:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8787/route `
  -ContentType "application/json" `
  -Body '{"prompt":"Write a Python preload manager for LoRA adapters"}'
```

---

## Core principle

A full model swap is a cold reboot.  
A LoRA adapter swap is a thread context switch.  
A KV-cache is virtual memory.  
A route event is a syscall trace.  
A 16GB GPU is not a toy — it is a constrained execution chamber that needs a kernel.

---

## Roadmap

### v0.4-alpha

- [x] Repository identity and MOK terminology
- [x] Atlas Load Cartographer package scaffold
- [x] Async preload manager
- [x] Route telemetry schema
- [x] Local expert config example
- [x] FastAPI gateway skeleton

### v0.5

- [ ] Combine embedding router + preload manager in gateway lifecycle
- [ ] Add route confidence thresholds
- [ ] Add adapter staged-hit telemetry
- [ ] Add VRAM guardrail config
- [ ] Add SGLang client shim
- [ ] Add benchmark scripts for route overhead vs model swap overhead

### v0.6+

- [ ] Token-level interrupt hooks
- [ ] Mid-generation route exception handling
- [ ] KV-cache page table abstraction
- [ ] Adapter eviction policy
- [ ] Route replay harness
- [ ] Proof-deck charts from SQLite telemetry

---

## License

MIT. See [`LICENSE`](LICENSE).
