from mok.evaluation.oracle import OracleExample, compute_regret


def test_compute_regret_reports_summary() -> None:
    examples = [
        OracleExample(
            request_id="req-1",
            chosen_expert="coder",
            expert_scores={"coder": 0.8, "instruct": 0.3},
        ),
        OracleExample(
            request_id="req-2",
            chosen_expert="instruct",
            expert_scores={"coder": 0.7, "instruct": 0.4},
        ),
    ]

    summary = compute_regret(examples)

    assert summary["count"] == 2.0
    assert summary["mean_regret"] == 0.15
    assert summary["oracle_match_rate"] == 0.5
