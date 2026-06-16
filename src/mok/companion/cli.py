from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mok.companion.config import CompanionConfig, DEFAULT_CONFIG_PATH
from mok.companion.control import (
    lifecycle_status,
    pause_mok,
    restart_mok,
    resume_mok,
    start_mok,
    stop_mok,
)
from mok.companion.llama_cpp import (
    LlamaCppCompanionError,
    build_server_command,
    find_installed_llama_server,
    format_server_command,
    start_server,
)
from mok.companion.models import best_gguf_model, find_gguf_models
from mok.companion.service import CompanionService
from mok.companion.startup import install_startup_tasks, startup_status, uninstall_startup_tasks
from mok.companion.storage import DatacardStore
from mok.companion.terminal import run_terminal


def add_companion_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("companion", help="Run the headless MoK companion.")
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Companion config path.",
    )
    commands = parser.add_subparsers(dest="companion_command", required=True)

    init = commands.add_parser("init", help="Create or update companion config.")
    init.add_argument("--allow-path", action="append", default=[], help="Folder or file MoK may observe.")
    init.add_argument("--storage-dir", help="Storage directory for memory.sqlite3 and config.")
    init.add_argument("--poll-seconds", type=float, help="Watch loop interval.")
    init.add_argument("--inference-vram-gb", type=float, help="Estimated VRAM needed by the resident model.")
    init.add_argument("--inference-ram-mb", type=int, help="Estimated RAM needed by the resident companion.")
    init.add_argument("--headroom", type=float, help="Reservation headroom ratio, default 0.20.")
    init.add_argument("--model-path", help="GGUF model path for llama.cpp.")
    init.add_argument("--mmproj-path", help="Multimodal projector path for llama.cpp.")
    init.add_argument("--llama-server-path", help="Path to llama-server.exe.")
    init.add_argument("--llama-url", help="OpenAI-compatible llama.cpp chat completions URL.")
    init.add_argument("--context-limit", type=int, help="llama.cpp context size.")
    init.add_argument("--gpu-layers", default=None, help="llama.cpp GPU layers, e.g. auto, all, 999.")
    init.add_argument("--auto-model", action="store_true", help="Auto-select the best local GGUF under the cap.")

    scan = commands.add_parser("scan", help="Scan allowlisted files once.")
    scan.add_argument("--json", action="store_true", help="Print JSON report.")

    watch = commands.add_parser("watch", help="Run the companion watch loop.")
    watch.add_argument("--iterations", type=int, help="Stop after N ticks.")

    recall = commands.add_parser("recall", help="Search stored datacards.")
    recall.add_argument("query", nargs="?", default="", help="Search query.")
    recall.add_argument("--lane", help="Restrict search to one lane.")
    recall.add_argument("--limit", type=int, default=10)
    recall.add_argument("--json", action="store_true", help="Print full JSON cards.")

    model = commands.add_parser("model", help="Inspect or auto-select local GGUF models.")
    model.add_argument("--list", action="store_true", help="List discovered GGUF candidates.")
    model.add_argument("--auto", action="store_true", help="Select the best discovered candidate.")
    model.add_argument("--limit", type=int, default=10)
    model.add_argument("--json", action="store_true", help="Print JSON.")

    serve = commands.add_parser("serve", help="Run llama.cpp server for the companion.")
    serve.add_argument("--background", action="store_true", help="Start server in the background.")
    serve.add_argument("--print-command", action="store_true", help="Only print the command.")

    chat = commands.add_parser("chat", help="Chat with the llama.cpp companion using recalled cards.")
    chat.add_argument("query", help="Question or instruction.")
    chat.add_argument("--lane", help="Restrict memory recall to one lane.")
    chat.add_argument("--limit", type=int, default=6)

    commands.add_parser("terminal", help="Open the standalone MoK terminal window.")
    for name, help_text in (
        ("lifecycle", "Print MoK runtime lifecycle status."),
        ("start", "Start llama.cpp server and companion watcher."),
        ("wakeup", "Alias for start."),
        ("pause", "Pause watcher while keeping model server available."),
        ("resume", "Resume watcher after pause."),
        ("stop", "Stop watcher and llama.cpp server."),
        ("sleep", "Alias for stop."),
        ("restart", "Restart watcher and llama.cpp server."),
    ):
        lifecycle = commands.add_parser(name, help=help_text)
        lifecycle.add_argument("--json", action="store_true", help="Print raw lifecycle JSON.")
    commands.add_parser("install-startup", help="Install login startup tasks for server and watcher.")
    commands.add_parser("uninstall-startup", help="Remove login startup tasks.")
    startup = commands.add_parser("startup-status", help="Inspect startup task status.")
    startup.add_argument("--json", action="store_true", help="Print JSON startup status.")

    status = commands.add_parser("status", help="Print companion memory status.")
    status.add_argument("--json", action="store_true", help="Print JSON status.")


def handle_companion_command(args: argparse.Namespace) -> bool:
    if not hasattr(args, "companion_command"):
        return False
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    config_path = Path(args.config_path)
    if args.companion_command == "init":
        config = CompanionConfig.load(config_path)
        if args.storage_dir:
            data = config.to_dict()
            data["storage_dir"] = args.storage_dir
            config = CompanionConfig.from_dict(data)
        if args.allow_path:
            config = config.with_allow_paths([Path(p) for p in args.allow_path])
        if args.poll_seconds is not None:
            data = config.to_dict()
            data["poll_seconds"] = args.poll_seconds
            config = CompanionConfig.from_dict(data)
        if args.inference_vram_gb is not None:
            data = config.to_dict()
            data["inference_vram_gb"] = args.inference_vram_gb
            config = CompanionConfig.from_dict(data)
        if args.inference_ram_mb is not None:
            data = config.to_dict()
            data["inference_ram_mb"] = args.inference_ram_mb
            config = CompanionConfig.from_dict(data)
        if args.headroom is not None:
            data = config.to_dict()
            data["reservation_headroom"] = args.headroom
            config = CompanionConfig.from_dict(data)
        if args.model_path:
            data = config.to_dict()
            data["model_path"] = args.model_path
            config = CompanionConfig.from_dict(data)
        if args.mmproj_path:
            data = config.to_dict()
            data["mmproj_path"] = args.mmproj_path
            config = CompanionConfig.from_dict(data)
        if args.llama_server_path:
            data = config.to_dict()
            data["llama_cpp_server_path"] = args.llama_server_path
            config = CompanionConfig.from_dict(data)
        if args.llama_url:
            data = config.to_dict()
            data["llama_cpp_url"] = args.llama_url
            config = CompanionConfig.from_dict(data)
        if args.context_limit is not None:
            data = config.to_dict()
            data["context_limit"] = args.context_limit
            config = CompanionConfig.from_dict(data)
        if args.gpu_layers is not None:
            data = config.to_dict()
            data["gpu_layers"] = int(args.gpu_layers) if str(args.gpu_layers).isdigit() else args.gpu_layers
            config = CompanionConfig.from_dict(data)
        if not config.llama_cpp_server_path:
            server = find_installed_llama_server()
            if server:
                data = config.to_dict()
                data["llama_cpp_server_path"] = str(server)
                config = CompanionConfig.from_dict(data)
        if args.auto_model or not config.model_path:
            candidate = best_gguf_model(max_bytes=config.max_model_bytes)
            if candidate:
                data = config.to_dict()
                data["model_path"] = str(candidate.path)
                data["mmproj_path"] = str(candidate.mmproj_path) if candidate.mmproj_path else None
                data["inference_vram_gb"] = max(candidate.total_size_gb, config.inference_vram_gb)
                config = CompanionConfig.from_dict(data)
        saved = config.save(config_path)
        DatacardStore(config.db_path)
        print(f"config={saved}")
        print(f"db={config.db_path}")
        print(f"allow_paths={len(config.allow_paths)}")
        print(f"inference_vram_gb={config.inference_vram_gb}")
        print(f"inference_ram_mb={config.inference_ram_mb}")
        print(f"headroom={config.reservation_headroom}")
        print(f"reserved_vram_gb={config.reserved_vram_gb}")
        print(f"reserved_ram_mb={config.reserved_ram_mb}")
        print(f"llama_server={config.llama_cpp_server_path}")
        print(f"model={config.model_path}")
        print(f"mmproj={config.mmproj_path}")
        return True

    service = CompanionService.from_config_path(config_path)
    if args.companion_command == "scan":
        report = service.scan_once()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"scanned={report['scanned']} stored={report['stored']} skipped={report['skipped']}")
            for error in report["errors"]:
                print(f"error={error}")
        return True

    if args.companion_command == "watch":
        service.watch(iterations=args.iterations)
        return True

    if args.companion_command == "recall":
        cards = service.store.search(args.query, lane=args.lane, limit=args.limit)
        if args.json:
            print(json.dumps([card_to_dict(card) for card in cards], indent=2))
        else:
            for card in cards:
                print(f"{card.card_id} lane={card.lane} trainable={card.trainable} {card.title}")
                print(f"  {card.summary}")
        return True

    if args.companion_command == "model":
        if args.auto:
            candidate = best_gguf_model(max_bytes=service.config.max_model_bytes)
            if not candidate:
                raise SystemExit("No GGUF model found under the configured cap.")
            data = service.config.to_dict()
            data["model_path"] = str(candidate.path)
            data["mmproj_path"] = str(candidate.mmproj_path) if candidate.mmproj_path else None
            data["inference_vram_gb"] = max(candidate.total_size_gb, service.config.inference_vram_gb)
            config = CompanionConfig.from_dict(data)
            config.save(config_path)
            if args.json:
                print(json.dumps(candidate.to_dict(), indent=2))
            else:
                print(f"selected={candidate.path}")
                print(f"size_gb={candidate.size_gb}")
                print(f"total_size_gb={candidate.total_size_gb}")
                print(f"mmproj={candidate.mmproj_path}")
            return True
        candidates = find_gguf_models(max_bytes=service.config.max_model_bytes, limit=args.limit)
        if args.json:
            print(json.dumps([candidate.to_dict() for candidate in candidates], indent=2))
        else:
            for candidate in candidates:
                print(f"score={candidate.score} size_gb={candidate.size_gb} path={candidate.path}")
                if candidate.mmproj_path:
                    print(f"  mmproj={candidate.mmproj_path}")
        return True

    if args.companion_command == "serve":
        try:
            if args.print_command:
                print(format_server_command(service.config))
                return True
            result = start_server(service.config, background=args.background)
            if args.background:
                print(f"llama_server_pid={result.pid}")
                print(f"url={service.config.llama_cpp_url}")
            return True
        except LlamaCppCompanionError as exc:
            raise SystemExit(str(exc)) from exc

    if args.companion_command == "chat":
        try:
            print(service.chat(args.query, lane=args.lane, limit=args.limit))
        except LlamaCppCompanionError as exc:
            print(f"llama.cpp error: {exc}")
            try:
                print("Start server with:")
                print(format_server_command(service.config))
            except LlamaCppCompanionError:
                pass
            raise SystemExit(1) from exc
        return True

    if args.companion_command == "terminal":
        run_terminal(config_path)
        return True

    if args.companion_command in {"lifecycle", "start", "wakeup", "pause", "resume", "stop", "sleep", "restart"}:
        actions = {
            "lifecycle": lifecycle_status,
            "start": start_mok,
            "wakeup": start_mok,
            "pause": pause_mok,
            "resume": resume_mok,
            "stop": stop_mok,
            "sleep": stop_mok,
            "restart": restart_mok,
        }
        result = actions[args.companion_command](config_path)
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2))
        else:
            print(format_lifecycle_result(args.companion_command, result))
        return True

    if args.companion_command == "install-startup":
        try:
            result = install_startup_tasks(config_path)
        except Exception as exc:
            raise SystemExit(f"startup install failed: {exc}") from exc
        print(f"method={result['method']}")
        print(f"llama_task={result['llama_task']}")
        print(f"watch_task={result['watch_task']}")
        print(f"llama_script={result['llama_script']}")
        print(f"watch_script={result['watch_script']}")
        print(f"log_dir={result['log_dir']}")
        if result["scheduler_error"]:
            print(f"scheduler_error={result['scheduler_error']}")
        return True

    if args.companion_command == "uninstall-startup":
        result = uninstall_startup_tasks()
        print(json.dumps(result, indent=2))
        return True

    if args.companion_command == "startup-status":
        result = startup_status()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for task_name, info in result.items():
                print(f"{task_name}: installed={info['installed']}")
                if info["output"]:
                    print(info["output"])
        return True

    if args.companion_command == "status":
        stats = service.store.stats()
        stats["inference_vram_gb"] = service.config.inference_vram_gb
        stats["inference_ram_mb"] = service.config.inference_ram_mb
        stats["headroom"] = service.config.reservation_headroom
        stats["reserved_vram_gb"] = service.config.reserved_vram_gb
        stats["reserved_ram_mb"] = service.config.reserved_ram_mb
        stats["allow_paths"] = [str(p) for p in service.config.allow_paths]
        stats["llama_server"] = str(service.config.llama_cpp_server_path) if service.config.llama_cpp_server_path else None
        stats["llama_url"] = service.config.llama_cpp_url
        stats["model"] = str(service.config.model_path) if service.config.model_path else None
        stats["mmproj"] = str(service.config.mmproj_path) if service.config.mmproj_path else None
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"db={stats['db_path']}")
            print(f"cards={stats['cards']} trainable={stats['trainable']} events={stats['events']}")
            print(f"lanes={stats['lanes']}")
            print(f"inference_vram_gb={stats['inference_vram_gb']} inference_ram_mb={stats['inference_ram_mb']}")
            print(f"headroom={stats['headroom']}")
            print(f"reserved_vram_gb={stats['reserved_vram_gb']} reserved_ram_mb={stats['reserved_ram_mb']}")
            print(f"allow_paths={stats['allow_paths']}")
            print(f"llama_server={stats['llama_server']}")
            print(f"llama_url={stats['llama_url']}")
            print(f"model={stats['model']}")
            print(f"mmproj={stats['mmproj']}")
        return True

    return False


def card_to_dict(card) -> dict:
    return {
        "card_id": card.card_id,
        "lane": card.lane,
        "source_type": card.source_type,
        "source_uri": card.source_uri,
        "title": card.title,
        "summary": card.summary,
        "tags": card.tags,
        "trust_score": card.trust_score,
        "trainable": card.trainable,
        "created_at": card.created_at,
        "updated_at": card.updated_at,
    }


def format_lifecycle_result(command: str, result: dict) -> str:
    state = result.get("state", "unknown")
    if command in {"wakeup", "start", "restart"}:
        note = str(result.get("wakeup_note") or "").strip()
        if note:
            return note
        if result.get("wakeup_note_error"):
            return f"MoK is awake, but the wakeup note failed: {result['wakeup_note_error']}"
        return f"MoK is {state}."
    if command in {"sleep", "stop"}:
        return "MoK is asleep. llama.cpp server and watcher are off."
    if command == "pause":
        return "MoK watcher is paused. llama.cpp server is still available."
    if command == "resume":
        return f"MoK watcher resumed. State: {state}."

    parts = [
        f"state={state}",
        f"server_running={bool(result.get('server_running'))}",
        f"watcher_running={bool(result.get('watcher_running'))}",
        f"paused={bool(result.get('paused'))}",
    ]
    return "\n".join(parts)
