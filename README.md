# Model Operating Kernel (MoK)

[![Pytest](https://github.com/nawnie/Model-Operating-Kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/nawnie/Model-Operating-Kernel/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-early%20runnable%20slice-yellow)
![Runtime](https://img.shields.io/badge/runtime-local--first-brightgreen)
![Target](https://img.shields.io/badge/target-consumer%20hardware-lightgrey)
![MoE](https://img.shields.io/badge/MoE-runtime%20orchestration-informational)

Model Operating Kernel is a local-first runtime for coordinating model and expert backends on consumer hardware.

MoK is not an in-model Mixture-of-Experts implementation. It is a runtime control layer: it registers experts, selects routes, manages VRAM pressure, invokes local or HTTP-backed models, records traces, and exports evaluation data for routing work.

The repository is an early runnable slice of the full design. It is meant to show the contracts that matter: routing, budgeting, memory, telemetry, and backend execution. Heavier training and serving work comes after those contracts stay stable under local tests.

For reviewers, MoK is public proof of the control layer. It is not a claim that every planned training path is finished.

## Current Status

The current runtime includes:

- expert registry with lifecycle state
- VRAM budget accounting and idle-expert eviction
- rule-based `R0` routing
- learned-router scaffolding
- per-expert circuit breakers
- mock, HTTP, Ollama, and llama.cpp-style backends
- GGUF metadata inspection
- JSONL trace logging
- oracle scoring and training-pair export
- smoke evaluation harnesses
- companion runtime and terminal controls

The next operational goal is to collect real local traces, measure actual VRAM behavior, and validate routing quality against repeatable eval sets.

## Repository Layout

```text
configs/       runtime and expert configuration
docs/          architecture, training, roadmap, and research notes
evaluation/    smoke prompts, oracle labels, and eval runners
sources/       source material and OCR extraction
src/mok/       Python package
templates/     starter config and dependency templates
tests/         unit and integration tests
run_mok.py     command entrypoint
```

Generated traces, private training data, local datasets, and model assets should stay out of version control.

## Quick Start

```powershell
cd C:\Users\Shawn\Desktop\MoK-Project
python -m pip install -e .
python -m pytest -q
python run_mok.py "write Python to reverse a list"
```

Useful local commands:

```powershell
python run_mok.py --has-image "describe this screenshot"
python run_mok.py --inspect-gguf "C:\path\to\model.gguf"
python run_mok.py --scan-gguf-dir "C:\path\to\models"
python run_mok.py --config configs\real_experts.json "write Python to reverse a list"
```

## Runtime Design

MoK is built around a few core contracts:

- expert metadata stays explicit and machine-readable
- routing decisions are separate from backend execution
- VRAM is treated as a managed budget
- every request can produce trace data for replay and evaluation
- local models, HTTP backends, adapters, and future multimodal experts share one invocation contract

Configured backend keys:

- `mock`: no-server backend for tests and dry runs
- `http`: generic JSON HTTP backend
- `ollama`: Ollama `/api/generate` backend using `base_id` as the model tag
- `llama_cpp`: OpenAI-compatible chat backend for `llama-server` or `llama-cpp-python`
- `vllm`: reserved for later high-throughput serving work

The default local coordinator/general model is `mok-core:1b`, built from `configs/ollama/Modelfile.mok-core-1b` around `gemma3:1b`.

## Evaluation

Run the local MoK Core smoke set:

```powershell
python evaluation\run_mok_core_smoke.py
```

The smoke runner checks route choice and behavior cues such as project-file safety, tool-result skepticism, and current-information handling. Results and traces are written under `traces/`.

## Companion Runtime

The companion process is a lightweight interface for a small local MoK assistant. It can be controlled from the command layer while the runtime remains separate from the terminal UI.

Common lifecycle commands:

```powershell
mok wakeup
mok sleep
python run_mok.py companion lifecycle
```

## GGUF Support

MoK can inspect GGUF files without loading them for inference. This is used for:

- reading architecture metadata
- checking context length
- identifying quantization
- scanning local model directories
- hydrating registry entries from local model assets

## Development Checks

```powershell
python -m pytest -q
python evaluation\run_mok_core_smoke.py
```
