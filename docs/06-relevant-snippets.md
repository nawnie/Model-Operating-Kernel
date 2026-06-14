# Relevant Snippets

This file keeps the most useful design snippets from the source PDF in one place, cleaned up for fast reference.

## Architecture claim

"MoK is a new architecture, not another model."

Why it matters:

- this is the framing that keeps the project from collapsing into "just train a bigger model"
- the runtime kernel is the product boundary

## The five things worth training

- router
- core coordinator
- LoRA expert library
- memory policy
- VRAM cost model

Why it matters:

- this is the real project scope for the learned system
- if work falls outside these buckets, it should justify itself

## Router staging

- `R0`: rules
- `R1`: zero-shot routing tier
- `R2`: trained classifier
- `R3`: cost-aware policy

Why it matters:

- MoK is supposed to earn its way from heuristics to learned routing
- there is always a fallback path

## Key runtime rule

"We do not train in the dark."

Why it matters:

- traces come before serious training
- evaluation comes before promotion

## Trace schema fields to preserve

- request id and turn index
- request text or hash
- modality flags
- resident experts
- VRAM pressure before routing
- router decision and confidence
- coordinator plan
- experts called, evicted, prefetched
- load, offload, TTFT, total latency
- peak VRAM, context length, OOM
- final answer and later quality label

Why it matters:

- this is the training data backbone for router, coordinator, and memory work

## Named metrics

- route decision time
- adapter/model load time
- offload time
- time to first token
- total response time
- OOM rate
- cold-miss rate
- escalation rate
- routing F1
- regret vs oracle

Why it matters:

- these are the metrics the project should center, not vague "felt better" judgments

## Shared-base guidance

The text path should use one shared quantized base plus adapters, while vision stays a separate full model loaded on demand.

Why it matters:

- this is the core memory-efficiency trick
- forcing vision into the same path would likely distort the design too early

## Acceptance bars that should not get lost

- `R2` fast path under 5 ms
- `ECE < 0.05`
- adapter attach under roughly 100 ms
- local runtime target stays within the 16 GB budget
- OOM is a release-blocking failure

## Roadmap discipline

"Never start a T-phase before its E-dependency logs usable traces."

Why it matters:

- this is the best anti-chaos rule in the whole document
