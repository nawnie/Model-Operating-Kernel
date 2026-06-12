# Model Operating Kernel Architecture

## Summary

Model Operating Kernel is a local-first architecture for managing multi-expert model execution like an operating system manages hardware resources.

It is not a replacement for serving engines. It sits above them as a resource-aware orchestration layer.

## Layers

```text
User/API layer
    ↓
MOK gateway
    ↓
Router
    ↓
Preload manager + telemetry + guardrails
    ↓
Execution backend
    ↓
Local GPU / system memory / storage
```

## First implementation: Atlas Load Cartographer

Atlas Load Cartographer is the first implementation of MOK. Its job is to map user prompts to expert routes, stage likely adapter resources, and record traceable operational data.

## Design principles

### 1. Full model swaps are expensive

The base model should stay resident whenever possible. Expert adapters are treated as lightweight execution contexts.

### 2. Routing must be measurable

Routing should never become mysterious overhead. Every route decision should emit timing and confidence telemetry.

### 3. Local hardware needs protection

The default target is a 16GB VRAM workstation. Memory ceilings are part of the architecture, not an afterthought.

### 4. Staging is separate from execution

The preload manager warms adapter files into system memory. It does not load them into VRAM. This separation keeps disk I/O, RAM staging, and GPU execution easier to test.

### 5. The architecture should stay backend-neutral

The gateway should remain abstract enough to test without requiring a specific model server.
