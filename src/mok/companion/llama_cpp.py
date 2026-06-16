from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from mok.companion.config import CompanionConfig


class LlamaCppCompanionError(RuntimeError):
    """Raised when the companion cannot talk to llama.cpp."""


def find_installed_llama_server() -> Path | None:
    candidates = [
        Path.home() / "Tools" / "llama.cpp" / "current.txt",
        Path.home() / "Tools" / "llama.cpp" / "b9660" / "llama-server.exe",
    ]
    current_txt = candidates[0]
    if current_txt.exists():
        text = current_txt.read_text(encoding="utf-8-sig").strip()
        if text:
            path = Path(text)
            if path.exists():
                return path
    for path in candidates[1:]:
        if path.exists():
            return path
    return None


def build_server_command(config: CompanionConfig) -> list[str]:
    if not config.llama_cpp_server_path:
        raise LlamaCppCompanionError("llama_cpp_server_path is not configured.")
    if not config.model_path:
        raise LlamaCppCompanionError("model_path is not configured.")
    parsed = urlparse(config.llama_cpp_url)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 8080)
    command = [
        str(config.llama_cpp_server_path),
        "--model",
        str(config.model_path),
        "--alias",
        config.model_name,
        "--host",
        host,
        "--port",
        port,
        "--ctx-size",
        str(config.context_limit),
        "--gpu-layers",
        str(config.gpu_layers),
    ]
    if config.mmproj_path:
        command.extend(["--mmproj", str(config.mmproj_path)])
    return command


def start_server(config: CompanionConfig, *, background: bool = False) -> subprocess.Popen | int:
    command = build_server_command(config)
    if background:
        return subprocess.Popen(
            command,
            cwd=str(Path(command[0]).parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
    return subprocess.call(command, cwd=str(Path(command[0]).parent))


def chat_completion(config: CompanionConfig, prompt: str) -> str:
    body = {
        "model": config.model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the local MoK companion. Use retrieved memory only when it is relevant. "
                    "Be concise, practical, and say when memory is weak."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": config.chat_temperature,
        "max_tokens": config.chat_max_tokens,
        "stream": False,
    }
    req = urlrequest.Request(
        config.llama_cpp_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise LlamaCppCompanionError(f"llama.cpp HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise LlamaCppCompanionError(f"llama.cpp server is not reachable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LlamaCppCompanionError("llama.cpp request timed out") from exc

    try:
        payload = json.loads(raw)
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LlamaCppCompanionError(f"Unexpected llama.cpp response: {raw[:500]}") from exc


def wait_until_ready(config: CompanionConfig, *, timeout_seconds: float = 120.0) -> bool:
    parsed = urlparse(config.llama_cpp_url)
    base = f"{parsed.scheme or 'http'}://{parsed.netloc}"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            req = urlrequest.Request(base + "/v1/models", method="GET")
            with urlrequest.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def format_server_command(config: CompanionConfig) -> str:
    return " ".join(quote_arg(part) for part in build_server_command(config))


def quote_arg(value: str) -> str:
    if " " not in value and "\t" not in value:
        return value
    return '"' + value.replace('"', '\\"') + '"'
