from __future__ import annotations

from pathlib import Path

from mok.__main__ import build_parser
from mok.companion.cartographer import card_from_file, card_from_window_title
from mok.companion.config import CompanionConfig
from mok.companion.llama_cpp import build_server_command
from mok.companion.models import best_gguf_model, find_mmproj_for
from mok.companion.observer import observe_files
from mok.companion.startup import write_startup_scripts
from mok.companion.storage import DatacardStore
from mok.companion.control import state_label
from mok.companion.cli import format_lifecycle_result
from mok.companion.terminal import format_lifecycle, format_status
from mok.companion.wakeup import build_wakeup_prompt, fallback_wakeup_note


def test_companion_config_computes_twenty_percent_headroom(tmp_path: Path) -> None:
    config = CompanionConfig(
        storage_dir=tmp_path,
        inference_vram_gb=3.0,
        inference_ram_mb=1000,
        reservation_headroom=0.20,
    )

    config.save(tmp_path / "config.json")

    assert config.reserved_vram_gb == 3.6
    assert config.reserved_ram_mb == 1200


def test_companion_config_roundtrip_allow_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    watched = tmp_path / "watched"
    watched.mkdir()
    config = CompanionConfig(storage_dir=tmp_path).with_allow_paths([watched])
    config.save(config_path)

    loaded = CompanionConfig.load(config_path)

    assert loaded.allow_paths == [watched.resolve()]
    assert loaded.db_path == tmp_path / "memory.sqlite3"


def test_cartographer_builds_trainable_file_card(tmp_path: Path) -> None:
    path = tmp_path / "roadmap.md"
    content = "MoK roadmap\n\n" + ("router memory atlas training " * 40)

    card = card_from_file(path, content)

    assert card.lane == "planning"
    assert card.trainable is True
    assert "mok" in card.tags


def test_window_title_card_is_not_trainable() -> None:
    card = card_from_window_title("Codex - MoK Project")

    assert card.source_type == "window"
    assert card.trainable is False
    assert card.lane == "ai-work"


def test_datacard_store_upserts_and_searches(tmp_path: Path) -> None:
    store = DatacardStore(tmp_path / "memory.sqlite3")
    card = card_from_file(tmp_path / "notes.md", "MoK cartographer atlas memory lane " * 20)

    card_id = store.upsert_card(card)
    results = store.search("cartographer atlas")
    stats = store.stats()

    assert results[0].card_id == card_id
    assert stats["cards"] == 1
    assert stats["lanes"]["docs"] == 1


def test_datacard_store_logs_recent_chat(tmp_path: Path) -> None:
    store = DatacardStore(tmp_path / "memory.sqlite3")
    store.log_chat_message("user", "hello")
    store.log_chat_message("mok", "hi")

    recent = store.recent_chat()

    assert [m["role"] for m in recent] == ["user", "mok"]
    assert recent[-1]["content"] == "hi"


def test_file_observer_respects_allow_paths_and_extensions(tmp_path: Path) -> None:
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "notes.md").write_text("MoK memory cartographer " * 20, encoding="utf-8")
    (watched / "ignore.exe").write_text("no", encoding="utf-8")
    config = CompanionConfig(storage_dir=tmp_path, allow_paths=[watched])

    cards, result = observe_files(config)

    assert result.scanned == 1
    assert result.stored == 1
    assert len(cards) == 1
    assert cards[0].title == "notes.md"


def test_cli_accepts_companion_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["companion", "status"])

    assert args.command == "companion"
    assert args.companion_command == "status"


def test_cli_accepts_terminal_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["companion", "terminal"])

    assert args.command == "companion"
    assert args.companion_command == "terminal"


def test_cli_accepts_lifecycle_commands() -> None:
    parser = build_parser()
    for command in ("lifecycle", "start", "wakeup", "pause", "resume", "stop", "sleep", "restart"):
        args = parser.parse_args(["companion", command])
        assert args.command == "companion"
        assert args.companion_command == command


def test_cli_accepts_lifecycle_json_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["companion", "wakeup", "--json"])

    assert args.companion_command == "wakeup"
    assert args.json is True


def test_wakeup_lifecycle_output_hides_process_details() -> None:
    text = format_lifecycle_result(
        "wakeup",
        {
            "state": "running",
            "server_running": True,
            "watcher_running": True,
            "server": [{"pid": 1, "command_line": "llama-server.exe --model secret.gguf"}],
            "wakeup_note": "Back online.",
        },
    )

    assert text == "Back online."
    assert "command_line" not in text
    assert "llama-server" not in text


def test_terminal_status_format() -> None:
    text = format_status({"cards": 2, "trainable": 1, "events": 3, "lanes": {"code": 2}, "db_path": "x.db"})

    assert "cards=2" in text
    assert "code" in text


def test_terminal_lifecycle_format() -> None:
    text = format_lifecycle(
        {
            "state": "running",
            "server_running": True,
            "watcher_running": True,
            "server": [{"pid": 1}],
            "watcher": [{"pid": 2}],
            "wakeup_note": "Back online. I remember the last thread.",
        }
    )

    assert "state=running" in text
    assert "server_pids=[1]" in text
    assert "Back online" in text


def test_state_label() -> None:
    assert state_label(True, True) == "running"
    assert state_label(True, False) == "paused"
    assert state_label(False, False) == "stopped"


def test_wakeup_prompt_and_fallback(tmp_path: Path) -> None:
    card = card_from_file(tmp_path / "notes.md", "MoK memory atlas " * 20)
    prompt = build_wakeup_prompt([{"role": "user", "content": "last chat", "created_at": 1.0}], [card])
    fallback = fallback_wakeup_note([{"role": "user", "content": "last chat", "created_at": 1.0}], [])

    assert "just started back up" in prompt
    assert "Do not use a required phrase" in prompt
    assert "last chat" in prompt
    assert fallback.startswith("I am back online")


def test_model_discovery_prefers_3b_instruct_q4(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    weak = root / "umt5-xxl-encoder-Q5_K_M.gguf"
    good = root / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
    projector = root / "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
    weak.write_bytes(b"0" * 1024)
    good.write_bytes(b"0" * 2048)
    projector.write_bytes(b"0" * 512)

    candidate = best_gguf_model([root], max_bytes=7 * 1024 * 1024)

    assert candidate is not None
    assert candidate.path == good
    assert candidate.mmproj_path == projector
    assert candidate.total_size_bytes == 2560


def test_find_mmproj_for_model(tmp_path: Path) -> None:
    model = tmp_path / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
    projector = tmp_path / "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
    model.write_bytes(b"model")
    projector.write_bytes(b"projector")

    assert find_mmproj_for(model) == projector


def test_build_server_command_uses_llama_cpp_fields(tmp_path: Path) -> None:
    server = tmp_path / "llama-server.exe"
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    config = CompanionConfig(
        storage_dir=tmp_path,
        llama_cpp_server_path=server,
        model_path=model,
        mmproj_path=mmproj,
        context_limit=4096,
        gpu_layers=99,
    )

    command = build_server_command(config)

    assert str(server) == command[0]
    assert "--model" in command
    assert str(model) in command
    assert "--mmproj" in command
    assert str(mmproj) in command
    assert "4096" in command


def test_write_startup_scripts(tmp_path: Path) -> None:
    server = tmp_path / "llama-server.exe"
    model = tmp_path / "model.gguf"
    config_path = tmp_path / "config.json"
    config = CompanionConfig(
        storage_dir=tmp_path,
        llama_cpp_server_path=server,
        model_path=model,
        llama_cpp_url="http://localhost:8181/v1/chat/completions",
    )

    scripts = write_startup_scripts(config, config_path=config_path, root=tmp_path)

    assert scripts.llama_script.exists()
    assert scripts.watch_script.exists()
    assert "llama-server.exe" in scripts.llama_script.read_text(encoding="utf-8")
    assert "run_mok.py companion" in scripts.watch_script.read_text(encoding="utf-8")
    assert "8181" in scripts.llama_script.read_text(encoding="utf-8")
