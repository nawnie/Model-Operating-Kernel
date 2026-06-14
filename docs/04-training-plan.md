# Training Plan

## Order of work

The source document is clear that nothing learned should be trained before there is a runnable baseline and a trace stream. The practical order is:

1. `R0` heuristic router
2. trace logging
3. oracle route evaluation
4. shared-base model decision
5. first LoRA adapters
6. `R2` learned router
7. coordinator training
8. memory policy
9. VRAM cost model refinement
10. `R3` cost-aware policy

## Router plan

### R0

- keyword, regex, and modality detection
- serves as the first runnable baseline
- produces the first useful route traces

### R1

- zero-shot routing tier
- used as an escalation layer for ambiguous cases

### R2

- small multi-label classifier
- calibrated confidence
- cost-aware head
- target fast path under 5 ms

Acceptance goals from the source:

- beats `R0/R1` on F1 and regret
- `ECE < 0.05`
- clean shadow period before promotion

### R3

- contextual bandit or offline RL layer
- only starts after sufficient logged traces and counterfactual data exist
- must beat `R2` by measured reward, not by intuition

## Coordinator plan

The coordinator should be trained after the runtime can already emit structured route and outcome traces.

Stages:

- teacher trace generation
- supervised distillation
- DPO preference tuning
- constrained decoding hardening

What it must own:

- decomposition
- machine-readable routing plans
- merge logic for final answers

## LoRA expert library plan

Start with a shared quantized text base and train adapters in this order:

1. coder
2. instruct
3. reasoning
4. tool-use
5. memory / RAG

Vision stays separate as an on-demand full model rather than being forced into the shared-base path.

## Memory and cost plan

### Memory policy

- start with simple next-expert prediction
- benchmark `P0` heuristic prediction before considering a tiny transformer
- only claim wins if cold-miss rate improves on realistic sessions

### VRAM cost model

- start from install-time profiling
- include context-length sensitivity
- refine online from observed telemetry
- preserve hard safety headroom

## Core training rule

Every learned component must remain reversible to its heuristic baseline. If the learned version cannot beat the baseline on the agreed harness, the heuristic stays in place.
