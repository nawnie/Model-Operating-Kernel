# Model Operating Kernel

Model Operating Kernel, or MoK, is a local-first runtime for coordinating models, tools, and expert backends on consumer hardware.

A model can be impressive. A system that knows which model to call, what the machine can afford, and what happened afterward is more useful.

## What MoK does

- registers model and expert backends;
- routes requests through explicit policies;
- treats VRAM as a managed budget;
- supports mock, HTTP, Ollama, and llama.cpp-style backends;
- records JSONL traces for replay and evaluation;
- keeps routing decisions separate from backend execution.

MoK is not an in-model Mixture-of-Experts implementation. It is the operating layer around models: the part that keeps the wiring from becoming folklore.

## Current state

This is an early runnable slice. It proves the control contracts and local execution path. The next work is measured routing behavior on real local traces, not a claim that every planned serving or training path is complete.

## Quick start

~~~powershell
python -m pip install -e .
python -m pytest -q
python run_mok.py "write Python to reverse a list"
~~~

## The public stack

- [AIWF Studio](https://github.com/nawnie/AIWF-Studio) — local creative AI.
- [ReTrain](https://github.com/nawnie/ReTrain) — local training workflows.
- [Cartographer SDK](https://github.com/nawnie/atlas-core) — supporting context, lineage, and change infrastructure.
- [RNV1](https://github.com/nawnie/Rnv1) — the longer-term embodied and edge-AI direction.

Generated traces, private datasets, model assets, and local runtime state remain outside version control.
