from pathlib import Path

from mok.models.backends import HTTPBackend, MockBackend, RequestPayload
from mok.orchestration.runtime import OrchestratorRuntime


def test_runtime_handles_request_and_logs_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / "runtime.jsonl"
    runtime = OrchestratorRuntime.from_config(
        config_path=Path(r"C:\Users\Shawn\Desktop\MoK-Project\configs\example_experts.json"),
        trace_path=trace_path,
        backends={
            "local": MockBackend(),
            "vllm": MockBackend(),
            "http": HTTPBackend(),
        },
    )

    result = runtime.handle_request(
        RequestPayload(prompt="please fix this python bug", request_id="req-42")
    )

    assert result.expert_name == "coder"
    assert "Mock code specialist response" in result.text
    assert trace_path.exists()
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
