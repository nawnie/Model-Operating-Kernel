# Model Operating Kernel

Model Operating Kernel (MoK) is a runtime kernel for local model orchestration under constrained hardware. The goal is to make many experts behave like one coherent system through routing, budgeting, memory policy, structured coordination, and telemetry-driven learning.

## Initial priorities

- get the runtime loop working
- log one structured trace per request
- ship a heuristic router before learned routing
- choose a shared text base for LoRA experts
- validate everything on the 16 GB target runtime

## Expected layout

```text
src/mok/
  models/
  orchestration/
  routing/
  telemetry/
  evaluation/
configs/
tests/
```

## Runtime principles

- telemetry before serious training
- heuristic fallback for every learned component
- vision stays a separate full-model path unless proven otherwise
- OOM is a runtime blocker
