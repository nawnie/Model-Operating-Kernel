# MoK-Project

This folder now contains both the research pack and a runnable starter codebase for Model Operating Kernel (MoK).

## What exists now

- `docs/`
  - project, architecture, training, roadmap, and kickoff docs
- `sources/`
  - original PDF plus OCR extract
- `templates/`
  - future-file templates kept for reference
- `src/mok/`
  - initial Python package
- `configs/`
  - example expert registry config
- `tests/`
  - basic unit tests for the first runtime slice

## What the starter implements

- expert registry with lifecycle states
- simple VRAM budget manager with idle-expert eviction
- `R0` rules router
- mock backend plus HTTP backend stub
- runtime loop from prompt -> route -> budget -> backend -> trace
- JSONL trace logging
- oracle-regret skeleton for later evaluation work

## Quick start

```powershell
cd C:\Users\Shawn\Desktop\MoK-Project
python -m pip install pytest
python -m pytest -q
python run_mok.py "write python to reverse a list"
python run_mok.py --has-image "describe this screenshot"
python run_mok.py --inspect-gguf C:\path\to\model.gguf
python run_mok.py --scan-gguf-dir C:\path\to\models
```

## Current goal

This is the first buildable layer from the roadmap, not the final architecture. It is meant to make MoK concrete enough to iterate on:

- registry shape
- route records
- trace logging
- budget behavior
- expert invocation contract

The next logical step after this scaffold is to add a real route schema, richer trace fields, and the oracle-eval harness over actual expert outputs.

## GGUF support

The starter can now inspect GGUF model files without loading them for inference. This is useful for:

- reading architecture and context length from local GGUF assets
- identifying quantization type
- scanning a model directory to catalog local executors
- hydrating registry entries that point at GGUF files
