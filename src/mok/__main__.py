from __future__ import annotations

import argparse
import json
from pathlib import Path

from mok.models.backends import HTTPBackend, MockBackend, RequestPayload
from mok.models.gguf import inspect_gguf_file, scan_gguf_directory
from mok.orchestration.runtime import OrchestratorRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MoK starter runtime.")
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


def main() -> None:
    args = build_parser().parse_args()
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
        backends={
            "local": MockBackend(),
            "vllm": MockBackend(),
            "http": HTTPBackend(),
        },
    )
    result = runtime.handle_request(
        RequestPayload(
            prompt=args.prompt,
            modality_flags={"has_image": args.has_image},
        )
    )
    print(f"expert={result.expert_name}")
    print(f"confidence={result.route.confidence:.2f}")
    print(result.text)


if __name__ == "__main__":
    main()
