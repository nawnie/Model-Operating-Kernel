#!/usr/bin/env python
"""
evaluation/run_eval.py

End-to-end evaluation script for the MoK routing stack.

Loads canonical prompts from evaluation/prompts.jsonl, runs them through
the OrchestratorRuntime (with a configurable backend), computes regret
against evaluation/oracle_labels.jsonl, and prints a report.

Usage
-----
    python evaluation/run_eval.py \\
        --config configs/example_experts.json \\
        --prompts evaluation/prompts.jsonl \\
        --oracle evaluation/oracle_labels.jsonl \\
        [--trace traces/eval.jsonl] \\
        [--backend mock]

The ``--backend mock`` flag (default) uses a no-op backend that always
returns a fixed response.  Replace with ``ollama`` or ``llama_cpp`` for
live evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mok.evaluation.oracle import OracleExample, OracleHarness, compute_regret
from mok.memory.budget import BudgetManager
from mok.models.backends import BackendResponse, ExpertBackend, RequestPayload
from mok.models.registry import ExpertMetadata, ModelRegistry
from mok.orchestration.runtime import OrchestratorRuntime
from mok.routing.router import RulesRouter
from mok.telemetry.dashboard import render_report


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

class MockBackend(ExpertBackend):
    """No-op backend that returns a canned response for offline evaluation."""

    def generate(self, expert: ExpertMetadata, payload: RequestPayload) -> BackendResponse:
        return BackendResponse(
            text=f"[mock response from {expert.name}]",
            latency_ms=10,
        )


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _load_oracle(path: Path) -> dict[str, dict]:
    """Load oracle labels keyed by request_id."""
    result = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            result[obj["request_id"]] = obj
    return result


def _iter_prompts(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(
    config_path: Path,
    prompts_path: Path,
    oracle_path: Path,
    trace_path: Path | None = None,
    backend_name: str = "mock",
) -> dict:
    registry = ModelRegistry.from_json(config_path)

    backend = MockBackend()
    backends = {b: backend for b in ("mock", "ollama", "llama_cpp", "local", "vllm")}

    runtime = OrchestratorRuntime(
        registry=registry,
        router=RulesRouter(),
        budget_manager=BudgetManager(),
        backends=backends,
        trace_logger=None,
    )

    oracle_labels = _load_oracle(oracle_path)
    examples: list[OracleExample] = []
    latencies: list[int] = []
    routing_errors = 0

    for prompt_rec in _iter_prompts(prompts_path):
        rid = prompt_rec["request_id"]
        payload = RequestPayload(
            prompt=prompt_rec["prompt"],
            request_id=rid,
            modality_flags=prompt_rec.get("modality_flags", {}),
        )
        try:
            result = runtime.handle_request(payload)
            latencies.append(result.total_ms)
            oracle_entry = oracle_labels.get(rid)
            if oracle_entry:
                expert_scores = oracle_entry["expert_scores"]
                routed_expert = result.expert_name
                if routed_expert in expert_scores:
                    examples.append(OracleExample(
                        request_id=rid,
                        chosen_expert=routed_expert,
                        expert_scores=expert_scores,
                    ))
        except Exception as exc:
            routing_errors += 1
            print(f"  [WARN] {rid}: {exc}", file=sys.stderr)

    regret_stats = compute_regret(examples)
    harness = OracleHarness()
    latencies.sort()
    p95_ms = latencies[int(len(latencies) * 0.95)] if latencies else 0
    mean_ms = int(sum(latencies) / len(latencies)) if latencies else 0

    return {
        **regret_stats,
        "routing_errors": routing_errors,
        "mean_latency_ms": mean_ms,
        "p95_latency_ms": p95_ms,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MoK routing evaluation")
    parser.add_argument("--config", default="configs/example_experts.json")
    parser.add_argument("--prompts", default="evaluation/prompts.jsonl")
    parser.add_argument("--oracle", default="evaluation/oracle_labels.jsonl")
    parser.add_argument("--trace", default=None)
    parser.add_argument("--backend", default="mock")
    args = parser.parse_args()

    stats = run_eval(
        config_path=Path(args.config),
        prompts_path=Path(args.prompts),
        oracle_path=Path(args.oracle),
        trace_path=Path(args.trace) if args.trace else None,
        backend_name=args.backend,
    )
    print(render_report(stats))


if __name__ == "__main__":
    main()
