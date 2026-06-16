from __future__ import annotations

from pathlib import Path

from mok.companion.eyes import attention_state_path, infer_gaze, read_attention_state, write_attention_state


def test_attention_state_round_trip_clamps_gaze(tmp_path: Path) -> None:
    path = tmp_path / "attention.json"

    written = write_attention_state(
        path,
        state="watching",
        target_kind="window",
        target_label="VS Code - service.py",
        gaze_x=2.0,
        gaze_y=-2.0,
        detail={"window": "VS Code"},
    )
    read = read_attention_state(path)

    assert written.gaze_x == 1.0
    assert written.gaze_y == -1.0
    assert read.state == "watching"
    assert read.target_kind == "window"
    assert read.target_label == "VS Code - service.py"
    assert read.detail == {"window": "VS Code"}


def test_inferred_gaze_is_stable_and_changes_by_target() -> None:
    first = infer_gaze("window", "Editor")
    second = infer_gaze("window", "Editor")
    third = infer_gaze("files", "allowed project files")

    assert first == second
    assert first != third
    assert all(-1.0 <= value <= 1.0 for value in (*first, *third))


def test_attention_state_path_uses_storage_dir(tmp_path: Path) -> None:
    assert attention_state_path(tmp_path) == tmp_path / "attention.json"
