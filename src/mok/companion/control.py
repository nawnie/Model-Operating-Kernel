from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from mok.companion.config import CompanionConfig, DEFAULT_CONFIG_PATH
from mok.companion.llama_cpp import start_server, wait_until_ready
from mok.companion.service import CompanionService
from mok.companion.startup import write_startup_scripts
from mok.companion.wakeup import generate_wakeup_note


@dataclass(slots=True)
class ProcessInfo:
    pid: int
    command_line: str

    def to_dict(self) -> dict:
        return {"pid": self.pid, "command_line": self.command_line}


def lifecycle_status(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    config = CompanionConfig.load(config_path)
    server = find_processes("llama-server.exe")
    watcher = find_processes("run_mok.py companion --config-path", require=" watch")
    paused = pause_flag(config).exists()
    return {
        "server_running": bool(server),
        "watcher_running": bool(watcher),
        "paused": paused,
        "server": [p.to_dict() for p in server],
        "watcher": [p.to_dict() for p in watcher],
        "state": state_label(bool(server), bool(watcher), paused=paused),
        "config_path": str(config_path),
    }


def start_mok(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    config = CompanionConfig.load(config_path)
    clear_pause(config)
    if not find_processes("llama-server.exe"):
        start_server(config, background=True)
    if not find_processes("run_mok.py companion --config-path", require=" watch"):
        scripts = write_startup_scripts(config, config_path=config_path)
        subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(scripts.watch_script),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
    time.sleep(1)
    wait_until_ready(config, timeout_seconds=120)
    status = lifecycle_status(config_path)
    try:
        note = generate_wakeup_note(CompanionService.from_config_path(config_path))
        status["wakeup_note"] = note.text
        status["wakeup_note_generated_by_model"] = note.generated_by_model
    except Exception as exc:
        status["wakeup_note_error"] = str(exc)
    return status


def pause_mok(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    config = CompanionConfig.load(config_path)
    set_pause(config)
    stop_processes("run_mok.py companion --config-path", require=" watch")
    time.sleep(0.5)
    return lifecycle_status(config_path)


def resume_mok(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    config = CompanionConfig.load(config_path)
    clear_pause(config)
    if not find_processes("run_mok.py companion --config-path", require=" watch"):
        scripts = write_startup_scripts(config, config_path=config_path)
        subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(scripts.watch_script),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
    time.sleep(1)
    return lifecycle_status(config_path)


def stop_mok(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    config = CompanionConfig.load(config_path)
    set_pause(config)
    stop_processes("run_mok.py companion --config-path", require=" watch")
    stop_processes("start_companion_watch.ps1")
    stop_processes("llama-server.exe")
    time.sleep(0.5)
    return lifecycle_status(config_path)


def restart_mok(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    stop_mok(config_path)
    clear_pause(CompanionConfig.load(config_path))
    return start_mok(config_path)


def find_processes(command_substring: str, *, require: str | None = None) -> list[ProcessInfo]:
    script = (
        "Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.CommandLine -like '*{escape_ps_like(command_substring)}*' }} | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    import json

    raw = json.loads(completed.stdout)
    rows = raw if isinstance(raw, list) else [raw]
    current_pid = str(subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", "$PID"],
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip())
    processes: list[ProcessInfo] = []
    for row in rows:
        pid = int(row.get("ProcessId", 0))
        command = str(row.get("CommandLine", ""))
        if str(pid) == current_pid:
            continue
        if "Get-CimInstance Win32_Process" in command:
            continue
        if require and require not in command:
            continue
        processes.append(ProcessInfo(pid=pid, command_line=command))
    return processes


def stop_processes(command_substring: str, *, require: str | None = None) -> int:
    processes = find_processes(command_substring, require=require)
    stopped = 0
    for process in processes:
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/F", "/T"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0:
            stopped += 1
    return stopped


def state_label(server_running: bool, watcher_running: bool, *, paused: bool = False) -> str:
    if paused and server_running:
        return "paused"
    if paused and not server_running:
        return "stopped"
    if server_running and watcher_running:
        return "running"
    if server_running and not watcher_running:
        return "paused"
    if not server_running and watcher_running:
        return "watcher_only"
    return "stopped"


def pause_flag(config: CompanionConfig) -> Path:
    return config.storage_dir / "watch.paused"


def set_pause(config: CompanionConfig) -> None:
    config.storage_dir.mkdir(parents=True, exist_ok=True)
    pause_flag(config).write_text("paused\n", encoding="utf-8")


def clear_pause(config: CompanionConfig) -> None:
    path = pause_flag(config)
    if path.exists():
        path.unlink()


def escape_ps_like(value: str) -> str:
    return value.replace("'", "''")
