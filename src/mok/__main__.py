from __future__ import annotations

import argparse
import json
from pathlib import Path

from mok.models.backends import (
    BackendInvocationError,
    ExpertBackend,
    HTTPBackend,
    MockBackend,
    RequestPayload,
)
from mok.models.backends_llama import LlamaCppBackend
from mok.models.backends_ollama import OllamaBackend
from mok.models.gguf import inspect_gguf_file, scan_gguf_directory
from mok.orchestration.runtime import OrchestratorRuntime
from mok.companion.cli import add_companion_parser, handle_companion_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MoK starter runtime.")
    subparsers = parser.add_subparsers(dest="command")
    add_companion_parser(subparsers)
    parser.add_argument("prompt", nargs="?", help="Prompt to route through the runtime.")
    parser.add_argument(
        "--config",
        default=str(Path("configs") / "example_experts.json"),
        help="Expert config path.",
    )
    parser.add_argument(
        "--trace-path",
        default=str(Path("traces") / "runtime.jsonl"),
        help="JSONL trace output path.",
    )
    parser.add_argument(
        "--has-image",
        action="store_true",
        help="Mark the request as containing an image input.",
    )
    parser.add_argument(
        "--inspect-gguf",
        help="Inspect a GGUF model file and print parsed metadata.",
    )
    parser.add_argument(
        "--scan-gguf-dir",
        help="Scan a directory recursively for GGUF files and print summaries.",
    )
    return parser


def build_default_backends() -> dict[str, ExpertBackend]:
    """Backends available from the CLI without requiring caller-side wiring."""
    return {
        "local": MockBackend(),
        "mock": MockBackend(),
        "vllm": MockBackend(),
        "http": HTTPBackend(),
        "ollama": OllamaBackend(),
        "llama_cpp": LlamaCppBackend(),
    }


def main() -> None:
    args = build_parser().parse_args()
    if handle_companion_command(args):
        return
    if args.inspect_gguf:
        inspection = inspect_gguf_file(Path(args.inspect_gguf))
        print(json.dumps(inspection.to_dict(), indent=2, sort_keys=True))
        return
    if args.scan_gguf_dir:
        inspections = scan_gguf_directory(Path(args.scan_gguf_dir))
        print(json.dumps([inspection.to_dict() for inspection in inspections], indent=2, sort_keys=True))
        return
    if not args.prompt:
        raise SystemExit("prompt is required unless --inspect-gguf or --scan-gguf-dir is used")
    runtime = OrchestratorRuntime.from_config(
        config_path=Path(args.config),
        trace_path=Path(args.trace_path),
        backends=build_default_backends(),
    )
    try:
        result = runtime.handle_request(
            RequestPayload(
                prompt=args.prompt,
                modality_flags={"has_image": args.has_image},
            )
        )
    except BackendInvocationError as exc:
        raise SystemExit(f"MoK backend failed: {exc}") from exc
    except RuntimeError as exc:
        raise SystemExit(f"MoK runtime failed: {exc}") from exc
    print(f"expert={result.expert_name}")
    print(f"confidence={result.route.confidence:.2f}")
    print(result.text)


if __name__ == "__main__":
    main()
