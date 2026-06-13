# Research Plan

## Purpose

This document holds the training and research track for Model Operating Kernel.

It is intentionally separate from `docs/runtime_mvp.md`. The runtime must work first. Training should improve routing, coordination, memory policy, and expert quality only after the mock runtime proves the control loop.

## Research boundary

Research work must not block the runtime MVP.

The first priority remains:

```text
registry + budget manager + mock backends + orchestrator + telemetry
```

Research becomes useful when it replaces a working heuristic with a measured learned component.

## Trainable component 1: router

The router decides which expert or experts should receive a request.

### Baseline

Start with deterministic routing:

- keyword rules
- modality detection
- simple cost thresholds
- hard safety or fallback rules

### Later learned router

After trace data exists, train or fit a small classifier using:

- request text
- modality flags
- history summary
- current loaded experts
- estimated VRAM pressure
- prior route outcomes

Targets:

- selected expert
- confidence
- optional expert ordering for multi-step tasks
- fallback/escalation decision

Research question:

> Can a small router improve expert selection quality without adding meaningful latency?

## Trainable component 2: core coordinator

The coordinator decomposes tasks, calls experts, and merges outputs.

### Baseline

Start with a deterministic or prompted coordinator that emits structured route plans.

### Later trained coordinator

Possible training methods:

- supervised fine-tuning on teacher traces
- preference tuning from successful vs wasteful routes
- schema-constrained generation

Research question:

> Can a small coordinator reliably produce valid plans and merge expert outputs better than a heuristic coordinator?

## Trainable component 3: expert library

MOK can manage many expert asset types. An expert may be:

- a full local model
- a quantized model
- a local server endpoint
- an OpenAI-compatible endpoint
- a vision-language model
- a tool executor
- an adapter
- a mock backend during testing

Adapters and LoRA experts are allowed, but they are not the whole architecture.

Research question:

> Which capabilities deserve separate full models, and which are cheap enough to represent as adapters or lightweight specialists?

## Trainable component 4: memory policy

The memory policy decides what to keep warm, what to stage, and what to evict.

### Baseline

Start with:

- resident core protection
- idle-first eviction
- non-mutating eviction plans
- fixed landing zone
- conservative VRAM estimates

### Later learned policy

Train or fit a policy using:

- route traces
- load latency
- offload latency
- recent expert usage
- long-session behavior
- failures and OOM warnings

Research question:

> Can the system predict the next useful expert well enough to reduce visible load latency?

## Data to collect before training

Every request should collect:

- request id
- prompt hash or sanitized prompt text
- modality flags
- selected expert
- route confidence
- model states before route
- VRAM estimate before route
- eviction plan
- executed evictions
- load time
- generation time
- output success/failure
- final model states
- quality label when available

## Evaluation suites

The research track should use separate suites:

| Suite | Purpose |
|---|---|
| ROUTE | route choice quality and route latency |
| DOMAIN | per-expert task quality |
| SESSION | long-running memory and offload stability |
| STRESS | VRAM pressure, eviction behavior, and concurrency |

## Ablation rule

Every learned component must be compared against the heuristic runtime baseline.

If a learned component cannot beat the heuristic on quality, latency, or stability, keep the heuristic.

## Research gates

Research work may advance only when:

1. The runtime MVP works with mock backends.
2. Trace logging exists.
3. A heuristic baseline is measured.
4. The learned component has a test suite.
5. The learned component can be removed without breaking the runtime.

## Immediate research backlog

Do not implement these before the MVP, but keep them as future tracks:

- router dataset schema
- coordinator trace schema
- memory-policy replay simulator
- oracle evaluation harness
- cost-vs-quality benchmark report
- expert escalation thresholds
- adapter-vs-full-model comparison
- vision expert integration study
