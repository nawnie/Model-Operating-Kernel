# Runtime MVP

## Purpose

This is the build contract for the first working Model Operating Kernel.

The MVP is not a training project, a LoRA library, or a whitepaper. The MVP is a working local runtime that coordinates multiple model assets under a shared hardware budget.

## Success condition

A user sends one prompt. MOK chooses between at least three experts, checks the memory budget, plans or executes idle-model offloads, activates the selected expert, returns a mock or real response, marks the expert idle, and records the route.

```text
prompt -> core coordinator -> router -> budget plan -> evict/load -> expert response -> trace
```

## MVP components

### 1. Core coordinator

The coordinator is the always-available control brain. In the first MVP it may be a deterministic function, a small local instruct model, or a mock coordinator.

Responsibilities:

- accept the user request
- preserve task state
- decide whether an expert is needed
- merge or return the expert response
- keep the runtime coherent from the user's point of view

### 2. Model registry

The registry is the source of truth for model assets.

It tracks:

- model name
- role
- backend
- endpoint or local handle
- estimated VRAM cost
- estimated RAM cost
- lifecycle state
- current device
- scheduling metadata such as priority, pinning, and last-used timestamps

### 3. Budget manager

The budget manager protects the machine before a model is loaded.

It must:

- calculate current VRAM pressure
- preserve a landing zone for context growth and backend overhead
- protect resident core assets
- prefer evicting idle, non-pinned experts first
- return an eviction plan without mutating registry state

The runtime executes the plan. The budget manager only decides.

### 4. Expert model pool

The first pool should include:

- core coordinator
- instruct expert
- coder expert
- vision expert or mock vision expert
- memory/retrieval expert or mock memory expert

Experts may be full models, quantized models, local server endpoints, remote-compatible endpoints, adapters, or mock backends.

### 5. Mock backends

Mock backends are mandatory before real model weights are wired in.

They should simulate:

- load time
- unload time
- generation time
- VRAM/RAM cost
- response payload
- failure conditions

The point is to prove lifecycle control before expensive inference is involved.

### 6. Runtime orchestrator

The orchestrator performs the actual loop:

1. Receive prompt.
2. Ask coordinator/router for selected expert.
3. Ask budget manager for allocation clearance.
4. Execute planned evictions.
5. Load or activate the selected expert.
6. Run the backend.
7. Mark the expert idle.
8. Log telemetry.
9. Return response.

## Required routes for first test

The first deterministic router should support:

| Route | Trigger examples | Expert |
|---|---|---|
| instruct | general question, explanation, summary | instruct expert |
| coder | code, bug, function, Python, API | coder expert |
| vision | image, screenshot, diagram, photo | vision expert |
| memory | retrieve, remember, notes, docs | memory expert |
| core | simple coordination or fallback | core coordinator |

## Telemetry MVP

Every request should log:

- request id
- selected expert
- initial model states
- evictions planned
- evictions executed
- load time
- execution time
- final model states
- budget pressure before and after
- success or failure

## Definition of done

The MVP is complete when:

- `pytest` passes.
- The registry can hold at least five model assets.
- The budget manager returns non-mutating eviction plans.
- A mock runtime can complete one end-to-end prompt.
- The resident core is never evicted.
- At least one idle expert is evicted before another expert is loaded.
- No real model weights are required for the proof loop.

## Not in MVP

Do not start with:

- router training
- coordinator distillation
- LoRA expert training
- oracle evaluation
- vision fine-tuning
- benchmark papers

Those belong in the research track after the runtime works.
