# Training Environment

## Default environment assumptions

- OS for planning and early runtime work: Windows-first
- Python baseline: 3.10
- Target runtime hardware: local 16 GB VRAM consumer GPU
- Heavy training hardware: rented A100 or H100 when needed

This project should not be designed as if the full training stack must run comfortably on the target runtime card. The local card is for runtime truth, evaluation smoke, and some light router work. The cloud is for the expensive parts.

## Environment split

### Local runtime environment

Purpose:

- runtime loop development
- trace generation
- expert registry and budget work
- light routing experiments
- local evaluation smoke tests

Expected responsibilities:

- vLLM-style serving or equivalent adapter-capable runtime
- telemetry capture to JSONL, DuckDB, and Parquet
- basic router inference and export validation

### Local router-training environment

Purpose:

- `R2` router training
- calibration
- ONNX export
- route-eval harness work

This should be lightweight enough to run locally when possible.

### Cloud training environment

Purpose:

- shared-base LoRA adapter training
- coordinator distillation and preference tuning
- larger evaluation bursts
- oracle refresh runs

The PDF-native assumption is that this work happens on rented high-memory GPUs.

## Default stack

### Runtime serving

- `vLLM` as the default first choice
- `LoRAX` or similar as a fallback if adapter concurrency or hot-swap behavior is weak
- `FastAPI` or equivalent lightweight service layer only if needed around the runtime

### Adapter training

- `transformers`
- `PEFT`
- `TRL`
- `bitsandbytes`

### Coordinator training

- teacher traces generated externally or by API
- SFT first
- DPO second

### Router training

- PyTorch
- sentence encoder backbone
- ONNX export for fast inference

### Telemetry and evaluation

- JSONL trace logs
- DuckDB
- Parquet
- optional experiment tracking through Weights & Biases or MLflow

## Why this stack is the default

This matches the architecture in the source PDF instead of forcing the project into local trainer tools that solve adjacent problems. Tools like `kohya_ss` and `EveryDream2` can still be referenced later if they help with a specific training lane, but they should not define the system architecture up front.

## Runtime-local constraints to preserve

- always validate on the real 16 GB target
- treat OOM as a release-blocking runtime failure
- preserve at least 1 GB of hard headroom in cost estimates
- benchmark adapter attach/detach and full-model load separately

## Immediate environment tasks

- define dependency splits by runtime, router, and training concerns
- write a base-model bake-off memo template
- define the trace schema before serious training scripts
- standardize how local and cloud outputs are versioned
