# Model Operating Kernel (MOK)

> A multi-model runtime architecture for building one usable AI system out of several specialized models: a resident core coordinator plus expert models for instruction, coding, vision, tools, memory, reasoning, and future domains.

**Model Operating Kernel (MOK)** is a new architecture for making multiple models behave like one coordinated model system. It is a runtime layer for split loading, offloading, routing, and resource budgeting across independent model assets.

The first goal is simple and concrete:

> **Build a working local model system that can load, offload, route, and coordinate multiple expert models under constrained hardware.**

A traditional Mixture-of-Experts model routes between experts inside one trained model. MOK explores a runtime-level version of that idea: separate specialist models are treated as managed expert assets coordinated by a core process.

---

## The core idea

A MOK system has three major parts:

1. **Core coordinator model**
   - The always-on controller.
   - Understands the user request.
   - Decides which expert model is needed.
   - Maintains task state and final response control.
   - Should usually stay `resident` in VRAM.

2. **Expert model pool**
   - Specialist models for different capabilities.
   - Initial targets: instruct, coding, vision, reasoning, tool use, and memory/retrieval.
   - Experts may be local LLMs, VLMs, quantized models, adapters, or external backend processes.

3. **Load/offload manager**
   - Keeps the system inside hardware limits.
   - Loads the needed model or expert.
   - Offloads inactive experts.
   - Tracks RAM, VRAM, cache state, and route cost.

The point is to make a working model-of-models runtime.

---

## Why this exists

Local AI systems are becoming more capable, but they are still awkward when one task needs multiple abilities.

A strong coding model may be weak at vision. A vision model may be weak at long instruction following. A small instruct model may be fast but unable to handle deep code. A large model may be powerful but too expensive to keep loaded all the time on a 16GB GPU.

MOK solves that at the runtime level.

Instead of forcing one model to do everything, MOK treats models like managed compute resources:

- load the expert needed now
- offload what is not needed
- keep a small/core coordinator available
- pass structured state between experts
- prevent VRAM crashes
- measure routing and load costs
- make the system feel like one coherent model

---

## What MOK is

MOK is a **runtime-level MoE-style architecture**.

It coordinates multiple independent models as if they were expert regions of one larger system.

Examples:

| User need | Core decision | Expert used |
|---|---|---|
| Write or debug code | route to code expert | coder model |
| Explain a general question | stay with core or instruct expert | instruct model |
| Analyze an image | load vision expert | vision model |
| Use a local tool | route to tool executor | tool model / tool layer |
| Retrieve project memory | route to memory expert | retrieval model |
| Multi-step task | coordinate several experts | core + selected experts |

This is related to Mixture-of-Experts in spirit, but it is not limited to one trained MoE model. MOK manages a pool of real models at runtime.

---

## What MOK is not

MOK is **not**:

- a LoRA-only adapter router
- a prompt-chain framework
- a chatbot personality system
- a LangChain clone
- a research note pretending to be a product
- a wrapper that assumes unlimited GPU memory

---

## Target architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                      USER / APP LAYER                       │
│        chat UI, API call, local workflow, automation         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  CORE COORDINATOR MODEL                     │
│   understands task, keeps state, chooses expert, merges work │
└───────────────┬──────────────────────────────┬──────────────┘
                │                              │
                ▼                              ▼
┌─────────────────────────────┐   ┌───────────────────────────┐
│      MODEL ROUTER            │   │    MEMORY / STATE BUS      │
│ intent, modality, cost, risk │   │ task state, cache, context │
└───────────────┬─────────────┘   └─────────────┬─────────────┘
                │                               │
                ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  LOAD / OFFLOAD MANAGER                     │
│        VRAM budget, RAM staging, model cache, eviction       │
└───────────────┬──────────────────────────────┬──────────────┘
                │                              │
                ▼                              ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────┐
│   INSTRUCT EXPERT   │ │    CODER EXPERT     │ │ VISION EXP. │
└─────────────────────┘ └─────────────────────┘ └─────────────┘
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────┐
│  REASONING EXPERT   │ │  MEMORY / RAG EXP.  │ │ TOOL EXPERT │
└─────────────────────┘ └─────────────────────┘ └─────────────┘
```

---

## First working milestone

The first milestone is not a whitepaper. It is a minimal working MOK runtime.

### Milestone 1: working multi-model loop

Build a prototype that can:

1. Start or simulate a lightweight core coordinator.
2. Register multiple expert models in a model registry.
3. Accept a prompt through an API or CLI.
4. Decide whether the request needs core, coder, instruct, vision, or another expert.
5. Check the memory budget before promoting an expert.
6. Load the selected expert if it is not active.
7. Offload inactive experts when memory limits require it.
8. Run the expert or mock backend.
9. Return the result through the core coordinator.
10. Log route, load time, offload time, memory pressure, and final result status.

### Initial expert set

The first practical expert set should be:

- **Core coordinator**: small instruct/reasoning model
- **Coder expert**: code-specialized local model
- **Instruct expert**: general instruction model
- **Vision expert**: image understanding model
- **Memory/retrieval expert**: project/document context helper

The exact models can change. The architecture should not depend on one model name.

---

## Model lifecycle states

MOK tracks models as assets with explicit hardware lifecycle states:

| State | Meaning |
|---|---|
| `offline` | On disk only. Not staged in RAM or resident in VRAM. |
| `staged` | Paged or prepared in system RAM. Ready for faster promotion. |
| `resident` | Permanently held in VRAM. Reserved for the core coordinator. |
| `active` | In VRAM and currently executing. |
| `idle` | In VRAM but waiting. Eligible for eviction under pressure. |

### Required manager behavior

- Know which models are available.
- Know which models are currently in RAM or VRAM.
- Know estimated RAM/VRAM cost per model.
- Load only what is needed.
- Offload least-needed models first.
- Preserve the resident core coordinator.
- Avoid crashing the GPU by crossing hard memory limits.
- Record every load/offload event.

This is the heart of the project.

---

## Current repository status

**Status:** early MOK runtime scaffold  
**Goal:** working multi-model MOK prototype  
**Hardware target:** local consumer hardware, especially 16GB VRAM systems  
**Primary concern:** split loading, offloading, routing, and coordination

The current codebase contains the first MOK-native core pieces:

- `src/mok/models/registry.py`
- `src/mok/memory/budget.py`
- `tests/test_model_registry.py`
- `tests/test_memory_budget.py`

---

## Near-term build plan

### Phase 1 — model registry

Define each model:

- name
- role
- backend
- local path or server endpoint
- modality support
- estimated RAM cost
- estimated VRAM cost
- lifecycle state
- current device

### Phase 2 — core coordinator

Create the first coordinator loop:

```text
prompt -> classify need -> choose expert -> call expert -> merge result -> respond
```

This can begin with deterministic routing before any learned router exists.

### Phase 3 — load/offload manager

Implement model lifecycle control:

- resident core model
- loaded models
- idle models
- memory budget
- eviction policy
- load queue
- offload queue
- telemetry

### Phase 4 — first expert integration

Integrate at least two real experts:

- core/instruct model
- coder model

Then add vision once the model lifecycle is stable.

### Phase 5 — benchmark and prove it

Measure:

- route decision time
- model load time
- model unload time
- memory pressure
- time to first token
- total response time
- failure modes

---

## Suggested repo direction for Claude/Codex

If you are an AI coding agent working on this repo, your first job is to turn this into a working MOK runtime.

Start from this structure:

```text
src/mok/
├── __init__.py
├── core/
│   ├── coordinator.py
│   └── state.py
├── models/
│   ├── registry.py
│   ├── lifecycle.py
│   └── backends.py
├── routing/
│   └── router.py
├── memory/
│   └── budget.py
├── telemetry/
│   └── events.py
└── api/
    └── app.py
```

Do not assume the first implementation must use LoRA. LoRA may become one expert type, but MOK is about coordinating multiple models and managing their load/offload lifecycle.

---

## Development commands

```powershell
git clone https://github.com/nawnie/Model-Operating-Kernel.git
cd Model-Operating-Kernel
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
pytest
```

---

## Core principle

A single model does not have to do everything.

MOK is the runtime that decides:

- which model should think
- which model should see
- which model should code
- which model should remember
- which model should be loaded
- which model should be offloaded
- how the result becomes one coherent answer

The first win is a working prototype. The theory comes after the model actually runs.

---

## License

MIT. See [`LICENSE`](LICENSE).
