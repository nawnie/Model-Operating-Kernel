# Phase 1 Model Operating Kernel Spec

## Purpose

Phase 1 turns Model Operating Kernel into a concrete multi-model runtime foundation.

The target loop is:

```text
prompt -> core coordinator -> expert selection -> budget clearance -> load/offload -> expert execution -> merged response
```

The first implementation should be designed for a constrained local workstation rather than a datacenter inference cluster.

## Kernel framing

| OS kernel concept | MOK equivalent |
|---|---|
| Process scheduler | model/expert scheduler |
| Resident kernel process | core coordinator model |
| Process table | model registry |
| Memory protection | VRAM budget manager |
| Swap policy | load/offload manager |
| Device state | offline/staged/resident/active/idle lifecycle |
| Operational trace | telemetry events |

## Phase 1 lifecycle

1. The core coordinator receives a prompt.
2. A routing policy selects the most appropriate expert model.
3. The budget manager checks whether the expert can fit inside the usable VRAM budget.
4. Idle experts are selected for offload when clearance is needed.
5. The selected expert is promoted into the active pool.
6. The expert runs through a mock or real backend.
7. The response is returned through the coordinator.

Phase 1 may use mock backends. The important part is that registry state, budget pressure, and eviction decisions are testable.

## Resource guardrails

Default guardrails:

- Total VRAM ceiling: 14.5GB
- Landing zone: 3.5GB
- Usable VRAM budget: 11.0GB
- Core coordinator: resident and protected
- First eviction target: idle experts

A future GPU monitor should enforce these values using real device statistics before model loading or context growth can destabilize the workstation.

## Model lifecycle states

| State | Meaning |
|---|---|
| `offline` | On disk only. |
| `staged` | Prepared in system RAM. |
| `resident` | Permanently kept in VRAM, usually the core coordinator. |
| `active` | In VRAM and currently executing. |
| `idle` | In VRAM but not executing; eligible for eviction. |

## Next implementation target

Next work should add a mock backend, a simple runtime orchestrator, example expert config, and tests for the full prompt-to-expert loop.
