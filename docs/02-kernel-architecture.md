# Kernel Architecture

## Kernel analogy

The source document is explicit that MoK should be treated like an operating-system layer for models.

- Router = process scheduler
- Budget manager + lifecycle states = pager / virtual memory
- Expert call contract = system-call ABI
- Backends = device drivers
- Core coordinator = resident daemon
- Memory/state bus = IPC layer
- Next-expert predictor = page-cache prefetcher

This analogy matters because it defines what the product is. The product is not one giant model. The product is the layer that makes many models act like one local system.

## Control plane vs data plane

Control plane responsibilities:

- route selection
- task decomposition
- memory and budget clearance
- eviction and prefetch decisions
- telemetry and evaluation

Data plane responsibilities:

- heavy inference
- adapter attach/detach
- full-model load/offload
- streaming outputs

Rule: control-plane logic should stay small, inspectable, and replaceable. Heavy GPU work should stay outside it.

## Core runtime concepts

### Expert

An expert is any callable model asset that fits the MoK ABI. It may be:

- a LoRA adapter on a shared base
- a full text model
- a multimodal vision model
- a remote or local backend target

What matters is not the training origin. What matters is that the expert exposes metadata, declared costs, and a stable invocation contract.

### Lifecycle

The source describes a lifecycle like:

- `offline`
- `staged`
- `resident`
- `active`
- `idle`

These states are central to budgeting and eviction decisions.

### Budget manager

The source calls out:

- a hard ceiling of 14.5 GB
- a 3.5 GB landing zone
- FIFO as the current baseline eviction policy

The learned memory policy is a later improvement, not a prerequisite for the first runtime loop.

## Initial subsystem map

- `src/mok/models/registry.py`
  - expert metadata and lifecycle tracking
- `src/mok/models/backends.py`
  - backend adapters such as mock and HTTP
- `src/mok/orchestration/runtime.py`
  - prompt -> core -> route -> load/offload -> answer loop
- `src/mok/routing/router.py`
  - `R0` rules router first, later `R1/R2/R3`
- `src/mok/telemetry/events.py`
  - structured trace emission

## Early invariants

- no training phase starts before traces exist
- no learned component ships without a heuristic fallback
- active experts are never evicted
- the resident core remains protected
- the route schema should be frozen early enough to avoid churn across training artifacts

## Architectural stance

The cleanest first implementation is a hybrid local system:

- shared-base LoRA strategy for most text experts
- separate full-model vision path
- telemetry-first runtime
- offline evaluation harness before aggressive policy learning

This is the smallest architecture that still honors the paper's claims.
