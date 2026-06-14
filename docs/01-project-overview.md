# MoK Project Overview

## Definition

Model Operating Kernel (MoK) is framed as a new architecture rather than a single model. The product is the kernel layer that decides which model runs, when a model is loaded or evicted, how state moves between experts, and how the system stays inside a fixed local hardware budget.

Target runtime in the source PDF: one 16 GB VRAM consumer GPU.

## Main thesis

MoK is trying to make a local model-of-models feel like one coherent system by combining:

- a router
- a resident core coordinator
- a shared-base LoRA expert library
- a learned memory policy
- a VRAM cost model

The runtime is the product. The learned pieces sit inside that runtime and must prove they beat the heuristic versions they replace.

## What the source says already exists

The PDF describes an early repo state with:

- a model registry with expert metadata and lifecycle states
- a budget manager with a 14.5 GB ceiling and 3.5 GB landing zone
- FIFO eviction over idle experts
- tests for registry and budget guardrails
- immediate implementation targets around `backends.py`, `runtime.py`, and `example_experts.json`

This means the first job is not "train everything." The first job is getting a runnable, trace-producing loop.

## Core design choices

- MoK should act like a kernel for models, not a prompt-chaining wrapper.
- Control-plane decisions should stay lightweight and replaceable.
- Heavy inference belongs in the data plane.
- Heuristics must exist before learned policies.
- Every learned component needs a measurable acceptance bar.

## Five trainable components

1. Router
Routes requests to the right expert or expert set, with quality and cost awareness.

2. Core coordinator
Breaks down work, emits structured routes, and merges outputs into one final answer.

3. LoRA expert library
Uses one shared quantized base for text experts so multiple adapters can fit in a 16 GB runtime budget.

4. Memory policy
Predicts what to prefetch and what to evict instead of relying on FIFO forever.

5. VRAM cost model
Predicts memory pressure and latency so the runtime can stay inside safe operating limits.

## Shared-base strategy

The source recommends a hybrid approach:

- text experts should mostly be LoRA adapters on one shared base
- vision should remain a separate full model loaded on demand

Named base-model candidates from the PDF:

- `Qwen2.5-7B-Instruct`
- `Llama-3.1-8B-Instruct`
- `Mistral-7B-Instruct-v0.3`

Default recommendation in the source: `Qwen2.5-7B-Instruct`, pending a quantized bake-off on the actual 16 GB target.

## What success looks like early

The early project should be considered healthy only if it can:

- run a documented end-to-end runtime loop
- emit one structured trace per request
- route through a simple `R0` heuristic router
- benchmark experts and routes against an oracle harness
- choose a shared base with a written decision memo

## Practical priority order

1. Runtime loop
2. Trace logging
3. Rules router
4. Base-model bake-off
5. First coder adapter
6. First learned router

That ordering keeps the project grounded in observable runtime behavior instead of training in the dark.
