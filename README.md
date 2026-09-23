# Model Operating Kernel

**MoK is the router/runtime layer around models and expert backends.**

It decides *where a task should go*, checks whether the machine can afford the selected expert, invokes the bounded backend, records what happened, and fails over explicitly when a route is unavailable.

MoK is **not Kairo**. Some historical MoK research drifted toward adaptive reasoning, memory, and self-improvement experiments that eventually became better represented by Kairo. Kairo remains a separate research line. The useful surviving identity of MoK is much closer to its original name: an operating kernel for models.

```text
request
  |
  v
route (R0 / optional R1 / optional R2)
  |
  v
health + circuit breaker
  |
  v
VRAM / resource budget
  |
  v
backend invocation
  |
  +--> Ollama
  +--> llama.cpp
  +--> HTTP / OpenAI-compatible lane
  +--> mock/test expert
  |
  v
trace + receipt + fallback evidence
```

## Active direction: router specialist

The current refocus is deliberately narrow:

1. **Discover/register experts** with explicit roles, backend type, context limit, resource cost, and trust metadata.
2. **Route** a request to the smallest appropriate expert using deterministic rules first.
3. **Escalate** only when routing confidence requires it.
4. **Budget** VRAM/RAM before activating an expert.
5. **Protect** unhealthy experts with circuit breakers and bounded fallbacks.
6. **Invoke** local backends through a stable execution boundary.
7. **Record** the selected route, confidence, resource pressure, latency, failures, and fallback behavior.
8. **Return a receipt** that a higher-level orchestrator can verify or reroute.

This makes MoK suitable as a specialist beneath [Shawn Core](https://github.com/nawnie/shawn-core-mcp): Shawn Core owns the whole user problem; MoK owns model/backend selection and resource-aware dispatch.

Kairo can be registered as one expert when a task needs Kairo's evidence-oriented program reasoning. Kairo does not own model dispatch.

## What already exists

The repository is not an empty redesign. Existing implementation includes:

- model/expert registry and typed backend configuration;
- deterministic R0 rules routing;
- optional R1 zero-shot escalation;
- optional R2 learned-router inference from NumPy or ONNX checkpoints;
- Ollama and llama.cpp backend adapters;
- generic HTTP/mock backend lanes;
- VRAM-aware activation and eviction logic;
- per-expert circuit breakers;
- JSONL runtime traces;
- GGUF inspection helpers;
- routing/evaluation harnesses and tests;
- experimental consultation, memory, RSI, persona, and companion code retained as historical/research branches.

The refocus does **not** require deleting the experimental code. It means the public product contract is the router/runtime core, while research branches are labeled as research instead of silently defining MoK's purpose.

## Quick start

Install and run the test suite:

```powershell
python -m pip install -e .
python -m pytest -q
```

Route a request with the starter configuration:

```powershell
python run_mok.py "write Python to reverse a list"
```

For actual local backends, copy and edit:

```text
configs/real_experts.json
```

Then run:

```powershell
python -m mok --config configs/real_experts.json "inspect this task and choose the appropriate expert"
```

The included configuration documents Ollama, llama.cpp, model identity, context limits, estimated VRAM/RAM cost, device state, and trust metadata.

## Router tiers

### R0 — deterministic rules

Cheap, inspectable routing. Use this whenever the task can be classified without calling another model.

### R1 — coordinator-assisted routing

Optional escalation when R0 confidence is insufficient.

### R2 — learned router

Optional compact classifier. The current implementation supports NumPy checkpoints and ONNX inference. It is an optimization/research lane, not a requirement for normal operation.

**Default policy:** use the lowest routing tier that can make a defensible decision.

## Integration boundary

A higher-level orchestrator should hand MoK something like:

```json
{
  "task": "review this Python failure",
  "requirements": ["code", "local"],
  "max_vram_gb": 10,
  "needs_vision": false
}
```

MoK should return a bounded routing/execution receipt:

```json
{
  "expert": "coder",
  "router_tier": "R0",
  "confidence": 0.93,
  "backend": "ollama",
  "resource_check": "passed",
  "fallback_used": false,
  "status": "succeeded"
}
```

That contract is intentionally smaller than a general autonomous agent. MoK answers **which expert, can we afford it, did it run, and what happened?**

## Relationship to the rest of the stack

- [Shawn Core](https://github.com/nawnie/shawn-core-mcp) — whole-problem orchestration and specialist handoffs.
- [Kairo](https://github.com/nawnie/kairo) — separate evidence-oriented program reasoning/research line.
- [Atlas Core](https://github.com/nawnie/atlas-core) — provenance, canonical state, approvals, execution evidence, and recovery.
- [AIWF Studio](https://github.com/nawnie/AIWF-Studio) — local creative AI workspace.
- [ReTrain](https://github.com/nawnie/ReTrain) — local training workflows.
- [Model Speedometer](https://github.com/nawnie/model-speedometer) — local inference observability.

## Status

The router/runtime core is runnable, but this refocus is still in progress. Existing research modules are broader than the active product contract. The next engineering milestone is to make the router-specialist boundary boring and dependable: explicit inputs, deterministic first-pass routing, resource checks, bounded invocation, fallbacks, and receipts.

Generated traces, private datasets, model assets, and local runtime state remain outside version control.
