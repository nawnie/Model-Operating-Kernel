# v0.5 Model Operating Kernel Spec

## Purpose

v0.5 turns Model Operating Kernel from a named architecture into a measurable gateway lifecycle.

The target loop is:

```text
prompt -> route selection -> adapter RAM staging -> telemetry record -> backend execution hook
```

The first implementation, Atlas Load Cartographer, is designed for a constrained local workstation rather than a datacenter inference cluster.

## Kernel framing

| OS kernel concept | MOK equivalent |
|---|---|
| Process scheduler | route and request scheduler |
| Thread context switch | LoRA adapter swap |
| Page cache | filesystem-to-RAM adapter staging |
| Virtual memory | KV-cache and prompt-state mapping |
| Memory protection | VRAM hard and soft limits |
| Syscall trace | SQLite route telemetry |

## v0.5 lifecycle

1. A prompt enters the FastAPI gateway.
2. The router selects a route using deterministic anchors or embeddings.
3. The preload manager stages the selected route adapter into OS page cache.
4. The gateway records route timing, confidence, staged-hit status, and success state.
5. The backend execution layer receives the selected route.

v0.5 may use a stubbed backend. The important part is that routing, staging, and telemetry are independently testable.

## Resource guardrails

Default guardrails:

- Soft VRAM limit: 14.0GB
- Hard VRAM limit: 14.5GB
- Adapter LRU limit: 3
- Max context tokens: 8192
- Eviction policy: LRU

A future GPU monitor should enforce these values before adapter loading or context growth can destabilize the workstation.

## Telemetry contract

Every route event should record route, success, route timing, total timing, semantic score, staged-hit status, prompt size, and any error.

The proof metric is not only answer quality. The proof metric is that routing and staging overhead remain cheaper than full model swaps.

## Next implementation target

Next work should add a backend abstraction, SGLang client shim, VRAM guardrail checker, benchmark script, and route replay harness.
