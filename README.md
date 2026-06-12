# Model Operating Kernel (MOK)

> A new multi-model architecture for building one usable AI system out of several specialized models: a core coordinator plus expert models for instruct, coding, vision, tools, memory, and future domains.

**Model Operating Kernel (MOK)** is not the Atlas adapter project, not a LoRA-only router, and not a normal AI agent wrapper. This repository starts a new architecture: a runtime that makes multiple models behave like one coordinated model system.

The first goal is simple and concrete:

> **Build a working local model system that can load, offload, route, and coordinate multiple expert models under constrained hardware.**

A traditional Mixture-of-Experts model usually routes between experts inside one trained model. MOK explores a runtime-level version of that idea: separate specialist models are treated as swappable expert modules managed by a core process.

---

## The core idea

A MOK system has three major parts:

1. **Core coordinator model**
   - The always-on controller.
   - Understands the user request.
   - Decides which expert model is needed.
   - Maintains task state and final response control.

2. **Expert model pool**
   - Specialist models for different capabilities.
   - Initial targets: instruct, coding, vision, reasoning, tool use, and memory/retrieval.
   - Experts may be local LLMs, VLMs, adapters, quantized models, or external backend processes.

3. **Load/offload manager**
   - Keeps the system inside hardware limits.
   - Loads the needed model or expert.
   - Offloads inactive experts.
   - Tracks RAM, VRAM, cache state, and route cost.

The point is not to prove a name. The point is to make a working model-of-models runtime.

---

## Why this exists

Local AI systems are becoming more capable, but they are still awkward when one task needs multiple abilities.

A strong coding model may be weak at vision. A vision model may be weak at long instruction following. A small instruct model may be fast but unable to handle deep code. A large model may be powerful but too expensive to keep loaded all the time on a 16GB GPU.

MOK is meant to solve that at the runtime level.

Instead of forcing one model to do everything, MOK treats models like managed compute resources:

- load the expert needed now
- offload what is not needed
- keep a small/core coordinator available
- pass structured state between experts
- prevent VRAM crashes
- measure routing and load costs
- eventually make the system feel like one coherent model

---

## What MOK is

MOK is a **runtime-level MoE-style architecture**.

It is designed to coordinate multiple independent models as if they were expert regions of one larger system.

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

- the Atlas adapter project
- a LoRA-only adapter router
- a prompt-chain framework
- a chatbot personality system
- a LangChain clone
- a research note pretending to be a product
- a wrapper that assumes unlimited GPU memory

Some early scaffold code may still contain names from prior experiments. Those should be cleaned up as the repo moves toward the actual MOK runtime.

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

1. Start a lightweight core coordinator.
2. Register multiple expert models in a model registry.
3. Accept a prompt through an API or CLI.
4. Decide whether the request needs core, coder, instruct, vision, or another expert.
5. Load the selected expert if it is not active.
6. Offload inactive experts when memory limits require it.
7. Run the expert.
8. Return the result through the core coordinator.
9. Log route, load time, offload time, memory pressure, and final result status.

### Initial expert set

The first practical expert set should be:

- **Core coordinator**: small instruct/reasoning model
- **Coder expert**: code-specialized local model
- **Instruct expert**: general instruction model
- **Vision expert**: image understanding model
- **Memory/retrieval expert**: project/document context helper

The exact models can change. The architecture should not depend on one model name.

---

## Load and offload strategy

MOK should treat models as resources with lifecycle states.

```text
unloaded -> staged -> loaded -> active -> idle -> offloaded
```

### Required manager behavior

- Know which models are available.
- Know which models are currently loaded.
- Know estimated RAM/VRAM cost per model.
- Load only what is needed.
- Offload least-needed models first.
- Prefer keeping the core coordinator alive.
- Avoid crashing the GPU by crossing hard memory limits.
- Record every load/offload event.

This is the heart of the project.

---

## Current repository status

**Status:** early scaffold / direction correction  
**Goal:** working multi-model MOK prototype  
**Hardware target:** local consumer hardware, especially 16GB VRAM systems  
**Primary concern:** split loading, offloading, routing, and coordination

The current codebase is only a starting scaffold. Some names and files may still reflect older adapter-routing experiments. Those should be renamed or replaced as the MOK runtime becomes real.

---

## Near-term build plan

### Phase 0 — clean the scaffold

- Rename old project/package references that imply this is Atlas.
- Keep useful ideas only where they serve MOK.
- Replace adapter-first language with model-pool language.
- Make the repo understandable to outside coding agents.

### Phase 1 — model registry

Create a registry that defines each model:

- name
- role
- backend
- local path or server endpoint
- modality support
- estimated RAM cost
- estimated VRAM cost
- load command
- unload command
- health check

### Phase 2 — core coordinator

Create the first coordinator loop:

```text
prompt -> classify need -> choose expert -> call expert -> merge result -> respond
```

This can begin with deterministic routing before any learned router exists.

### Phase 3 — load/offload manager

Implement model lifecycle control:

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

If you are an AI coding agent working on this repo, your first job is not to expand the old adapter scaffold.

Your first job is to turn this into a working MOK runtime.

Start by creating or refactoring toward this structure:

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
