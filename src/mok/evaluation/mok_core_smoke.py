from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from mok.__main__ import build_default_backends
from mok.memory.budget import BudgetManager
from mok.models.backends import RequestPayload
from mok.models.registry import ModelRegistry
from mok.orchestration.runtime import OrchestratorRuntime
from mok.routing.router import RulesRouter
from mok.telemetry.events import JsonlTraceLogger


@dataclass(slots=True)
class SmokeScenario:
    request_id: str
    prompt: str
    expected_expert: str
    required_all: list[str]
    required_any: list[str]
    forbidden: list[str]
    modality_flags: dict[str, bool]


@dataclass(slots=True)
class SmokeResult:
    request_id: str
    expected_expert: str
    actual_expert: str
    route_ok: bool
    required_all_ok: bool
    required_any_ok: bool
    forbidden_ok: bool
    passed: bool
    missing_required_all: list[str]
    matched_required: list[str]
    forbidden_hits: list[str]
    total_ms: int
    response_preview: str
    error: str | None = None


def load_scenarios(path: Path) -> list[SmokeScenario]:
    scenarios: list[SmokeScenario] = []
    for idx, obj in enumerate(_iter_jsonl(path), start=1):
        try:
            scenarios.append(
                SmokeScenario(
                    request_id=str(obj["request_id"]),
                    prompt=str(obj["prompt"]),
                    expected_expert=str(obj["expected_expert"]),
                    required_all=[str(term) for term in obj.get("required_all", [])],
                    required_any=[str(term) for term in obj.get("required_any", [])],
                    forbidden=[str(term) for term in obj.get("forbidden", [])],
                    modality_flags=dict(obj.get("modality_flags", {})),
                )
            )
        except KeyError as exc:
            raise ValueError(f"{path}:{idx} missing required field {exc}") from exc
    return scenarios


def score_response(scenario: SmokeScenario, actual_expert: str, response_text: str, total_ms: int) -> SmokeResult:
    normalized = response_text.lower()
    matched_required = [
        term for term in scenario.required_any
        if term.lower() in normalized
    ]
    missing_required_all = [
        term for term in scenario.required_all
        if term.lower() not in normalized
    ]
    forbidden_hits = [
        term for term in scenario.forbidden
        if term.lower() in normalized
    ]
    route_ok = actual_expert == scenario.expected_expert
    required_all_ok = not missing_required_all
    required_any_ok = bool(matched_required) if scenario.required_any else True
    forbidden_ok = not forbidden_hits
    return SmokeResult(
        request_id=scenario.request_id,
        expected_expert=scenario.expected_expert,
        actual_expert=actual_expert,
        route_ok=route_ok,
        required_all_ok=required_all_ok,
        required_any_ok=required_any_ok,
        forbidden_ok=forbidden_ok,
        passed=route_ok and required_all_ok and required_any_ok and forbidden_ok,
        missing_required_all=missing_required_all,
        matched_required=matched_required,
        forbidden_hits=forbidden_hits,
        total_ms=total_ms,
        response_preview=response_text.strip().replace("\n", " ")[:300],
    )


def summarize_results(results: Iterable[SmokeResult]) -> dict[str, Any]:
    rows = list(results)
    if not rows:
        return {"count": 0, "passed": 0, "pass_rate": 0.0, "failures": []}
    failures = [row.request_id for row in rows if not row.passed]
    passed = len(rows) - len(failures)
    return {
        "count": len(rows),
        "passed": passed,
        "pass_rate": round(passed / len(rows), 4),
        "failures": failures,
        "mean_latency_ms": int(sum(row.total_ms for row in rows) / len(rows)),
    }


def run_smoke(
    *,
    config_path: Path,
    scenarios_path: Path,
    trace_path: Path | None,
    results_path: Path | None,
) -> dict[str, Any]:
    registry = ModelRegistry.from_json(config_path)
    runtime = OrchestratorRuntime(
        registry=registry,
        router=RulesRouter(),
        budget_manager=BudgetManager(),
        backends=build_default_backends(),
        trace_logger=JsonlTraceLogger(trace_path) if trace_path else None,
    )

    results: list[SmokeResult] = []
    for scenario in load_scenarios(scenarios_path):
        payload = RequestPayload(
            prompt=scenario.prompt,
            request_id=scenario.request_id,
            modality_flags=scenario.modality_flags,
        )
        try:
            runtime_result = runtime.handle_request(payload)
            results.append(
                score_response(
                    scenario,
                    runtime_result.expert_name,
                    runtime_result.text,
                    runtime_result.total_ms,
                )
            )
        except Exception as exc:
            results.append(
                SmokeResult(
                    request_id=scenario.request_id,
                    expected_expert=scenario.expected_expert,
                    actual_expert="",
                    route_ok=False,
                    required_all_ok=False,
                    required_any_ok=False,
                    forbidden_ok=True,
                    passed=False,
                    missing_required_all=scenario.required_all,
                    matched_required=[],
                    forbidden_hits=[],
                    total_ms=0,
                    response_preview="",
                    error=str(exc),
                )
            )

    if results_path:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as fh:
            for result in results:
                fh.write(json.dumps(asdict(result), sort_keys=True) + "\n")

    summary = summarize_results(results)
    summary["results"] = results
    return summary


def render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "MoK Core Smoke Eval",
        f"  count: {summary['count']}",
        f"  passed: {summary['passed']}",
        f"  pass_rate: {summary['pass_rate']:.2%}",
        f"  mean_latency_ms: {summary.get('mean_latency_ms', 0)}",
    ]
    failures = summary.get("failures", [])
    if failures:
        lines.append(f"  failures: {', '.join(failures)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run live MoK Core behavior smoke scenarios.")
    parser.add_argument("--config", default="configs/real_experts.json")
    parser.add_argument("--scenarios", default="evaluation/mok_core_smoke.jsonl")
    parser.add_argument("--trace", default="traces/mok_core_smoke_trace.jsonl")
    parser.add_argument("--results", default="traces/mok_core_smoke_results.jsonl")
    args = parser.parse_args(argv)

    summary = run_smoke(
        config_path=Path(args.config),
        scenarios_path=Path(args.scenarios),
        trace_path=Path(args.trace) if args.trace else None,
        results_path=Path(args.results) if args.results else None,
    )
    print(render_summary(summary))
    return 0 if not summary["failures"] else 1


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
