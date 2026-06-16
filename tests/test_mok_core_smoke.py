from pathlib import Path

from mok.evaluation.mok_core_smoke import (
    SmokeScenario,
    load_scenarios,
    score_response,
    summarize_results,
)


def test_load_scenarios_reads_jsonl() -> None:
    scenarios = load_scenarios(Path("evaluation/mok_core_smoke.jsonl"))

    assert len(scenarios) >= 5
    assert scenarios[0].request_id.startswith("mok-smoke-")
    assert scenarios[0].expected_expert == "general"


def test_score_response_passes_when_route_and_terms_match() -> None:
    scenario = SmokeScenario(
        request_id="r1",
        prompt="prompt",
        expected_expert="general",
        required_all=["config"],
        required_any=["verify", "check"],
        forbidden=["guess"],
        modality_flags={},
    )

    result = score_response(
        scenario,
        actual_expert="general",
        response_text="First verify the config and check related files.",
        total_ms=10,
    )

    assert result.passed is True
    assert result.matched_required == ["verify", "check"]
    assert result.missing_required_all == []


def test_score_response_fails_on_forbidden_term() -> None:
    scenario = SmokeScenario(
        request_id="r1",
        prompt="prompt",
        expected_expert="general",
        required_all=[],
        required_any=["verify"],
        forbidden=["guess"],
        modality_flags={},
    )

    result = score_response(
        scenario,
        actual_expert="general",
        response_text="Verify later, but guess now.",
        total_ms=10,
    )

    assert result.passed is False
    assert result.forbidden_hits == ["guess"]


def test_score_response_fails_when_required_all_missing() -> None:
    scenario = SmokeScenario(
        request_id="r1",
        prompt="prompt",
        expected_expert="general",
        required_all=["config"],
        required_any=["verify"],
        forbidden=[],
        modality_flags={},
    )

    result = score_response(
        scenario,
        actual_expert="general",
        response_text="Verify related tests.",
        total_ms=10,
    )

    assert result.passed is False
    assert result.missing_required_all == ["config"]


def test_summarize_results_counts_failures() -> None:
    scenario = SmokeScenario("r1", "prompt", "general", [], ["verify"], [], {})
    passed = score_response(scenario, "general", "verify first", 10)
    failed = score_response(scenario, "coder", "verify first", 20)

    summary = summarize_results([passed, failed])

    assert summary["count"] == 2
    assert summary["passed"] == 1
    assert summary["failures"] == ["r1"]
