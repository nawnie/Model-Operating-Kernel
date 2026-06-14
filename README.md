# Model Operating Kernel (MoK)

Model Operating Kernel is a local-first runtime scaffold for coordinating multiple model and expert backends on consumer hardware.

MoK explores a practical model-orchestration layer: expert registration, route selection, VRAM budgeting, backend invocation, trace logging, and evaluation records. The current repository is an early buildable slice, not the final architecture.

## Current Status

This repo contains both research notes and a runnable starter codebase. The implementation is intentionally small so the routing, budgeting, and trace contracts can be tested before heavier model-serving work is added.

## Repository Layout

- `docs/` - project, architecture, training, roadmap, and kickoff documents.
- `sources/` - original source material plus extracted text where available.
- `templates/` - future-file templates kept for reference.
- `src/mok/` - initial Python package.
- `configs/` - example expert registry configuration.
- `tests/` - basic unit tests for the first runtime slice.

## Starter Runtime

The current runtime slice includes:

- expert registry with lifecycle states;
- simple VRAM budget manager with idle-expert eviction;
- `R0` rules router;
- mock backend plus HTTP backend stub;
- runtime loop from prompt to route to budget to backend to trace;
- JSONL trace logging;
- oracle-regret skeleton for later evaluation work;
- GGUF metadata inspection without loading models for inference.

## Quick Start

```powershell
python -m pip install pytest
python -m pytest -q
python run_mok.py "write Python to reverse a list"
python run_mok.py --has-image "describe this screenshot"
python run_mok.py --inspect-gguf "C:\path\to\model.gguf"
python run_mok.py --scan-gguf-dir "C:\path\to\models"
```

## Design Goals

MoK is being built around a few core contracts:

- keep expert metadata explicit and machine-readable;
- separate routing decisions from backend execution;
- treat VRAM as a managed budget, not an accident;
- record traces so route decisions can be replayed and evaluated;
- allow local models, HTTP backends, adapters, and future multimodal experts to share one invocation contract.

## GGUF Support

The starter can inspect GGUF model files without loading them for inference. This is useful for:

- reading architecture and context length from local GGUF assets;
- identifying quantization type;
- scanning a model directory to catalog local executors;
- hydrating registry entries that point at GGUF files.

## Near-Term Roadmap

The next development targets are:

- a stable route schema;
- richer trace fields;
- failure recovery for timeouts, malformed output, tool failure, and memory pressure;
- replay evaluation over actual expert outputs;
- stronger expert registry validation;
- backend integration beyond the current mock and HTTP stub paths.

## Project Boundary

MoK is not an in-model Mixture-of-Experts implementation. It is a runtime coordination layer that decides which external expert, model, adapter, or backend should handle a request, then records what happened so the system can be improved safely.
