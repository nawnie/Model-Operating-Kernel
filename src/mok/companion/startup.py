from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from mok.companion.config import CompanionConfig, DEFAULT_CONFIG_PATH
from mok.companion.llama_cpp import build_server_command


LLAMA_TASK_NAME = "MoK-Companion-LlamaServer"
WATCH_TASK_NAME = "MoK-Companion-Watch"
LLAMA_STARTUP_NAME = "MoK-Companion-LlamaServer.cmd"
WATCH_STARTUP_NAME = "MoK-Companion-Watch.cmd"


@dataclass(slots=True)
class StartupScripts:
    llama_script: Path
    watch_script: Path
    log_dir: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def write_startup_scripts(
    config: CompanionConfig,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    root: Path | None = None,
) -> StartupScripts:
    storage = config.storage_dir
    scripts_dir = storage / "startup"
    log_dir = storage / "logs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    root = root or project_root()

    llama_script = scripts_dir / "start_llama_server.ps1"
    watch_script = scripts_dir / "start_companion_watch.ps1"

    server_command = build_server_command(config)
    server_exe = server_command[0]
    server_args = server_command[1:]
    port = "8080"
    if ":8080" not in config.llama_cpp_url:
        port = config.llama_cpp_url.split(":")[-1].split("/")[0]

    llama_script.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"$LogPath = {ps_quote(str(log_dir / 'llama-server.log'))}",
                "function Write-Log($Message) {",
                "  $stamp = Get-Date -Format o",
                "  Add-Content -Path $LogPath -Value \"$stamp $Message\"",
                "}",
                f"$existing = Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue",
                "if ($existing) { Write-Log 'llama.cpp already listening'; exit 0 }",
                f"$Server = {ps_quote(server_exe)}",
                "$Args = @(" + ", ".join(ps_quote(arg) for arg in server_args) + ")",
                "Write-Log \"starting $Server $($Args -join ' ')\"",
                "Start-Process -FilePath $Server -ArgumentList $Args -WorkingDirectory (Split-Path $Server) -WindowStyle Hidden",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    watch_script.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Continue'",
                "$env:PYTHONIOENCODING = 'utf-8'",
                f"$ProjectRoot = {ps_quote(str(root))}",
                f"$ConfigPath = {ps_quote(str(config_path))}",
                f"$LogPath = {ps_quote(str(log_dir / 'companion-watch.log'))}",
                f"$PausePath = {ps_quote(str(storage / 'watch.paused'))}",
                "Set-Location $ProjectRoot",
                "while ($true) {",
                "  if (Test-Path $PausePath) { Start-Sleep -Seconds 5; continue }",
                "  $stamp = Get-Date -Format o",
                "  Add-Content -Path $LogPath -Value \"$stamp starting companion watch\"",
                "  python run_mok.py companion --config-path $ConfigPath watch >> $LogPath 2>&1",
                "  Start-Sleep -Seconds 10",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return StartupScripts(llama_script=llama_script, watch_script=watch_script, log_dir=log_dir)


def install_startup_tasks(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    config = CompanionConfig.load(config_path)
    scripts = write_startup_scripts(config, config_path=config_path)
    remove_llama_startup_entry()
    method = "task_scheduler"
    try:
        register_task(WATCH_TASK_NAME, scripts.watch_script)
    except RuntimeError as exc:
        method = "startup_folder"
        write_startup_folder_entries(scripts)
        scheduler_error = str(exc)
    else:
        scheduler_error = ""
    return {
        "method": method,
        "llama_task": None,
        "watch_task": WATCH_TASK_NAME,
        "llama_script": str(scripts.llama_script),
        "watch_script": str(scripts.watch_script),
        "log_dir": str(scripts.log_dir),
        "scheduler_error": scheduler_error,
        "server_starts_on_login": False,
    }


def uninstall_startup_tasks() -> dict:
    results = {}
    for task_name in (LLAMA_TASK_NAME, WATCH_TASK_NAME):
        completed = subprocess.run(
            ["schtasks", "/Delete", "/TN", task_name, "/F"],
            capture_output=True,
            text=True,
        )
        results[task_name] = completed.returncode == 0
    startup = startup_folder()
    for name in (LLAMA_STARTUP_NAME, WATCH_STARTUP_NAME):
        path = startup / name
        if path.exists():
            path.unlink()
            results[name] = True
        else:
            results[name] = False
    return results


def remove_llama_startup_entry() -> None:
    subprocess.run(
        ["schtasks", "/Delete", "/TN", LLAMA_TASK_NAME, "/F"],
        capture_output=True,
        text=True,
    )
    path = startup_folder() / LLAMA_STARTUP_NAME
    if path.exists():
        path.unlink()


def startup_status() -> dict:
    results = {}
    for task_name in (LLAMA_TASK_NAME, WATCH_TASK_NAME):
        completed = subprocess.run(
            ["schtasks", "/Query", "/TN", task_name, "/FO", "LIST"],
            capture_output=True,
            text=True,
        )
        results[task_name] = {
            "installed": completed.returncode == 0,
            "output": completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip(),
        }
    startup = startup_folder()
    for name in (LLAMA_STARTUP_NAME, WATCH_STARTUP_NAME):
        path = startup / name
        results[name] = {
            "installed": path.exists(),
            "output": str(path) if path.exists() else "",
        }
    return results


def register_task(task_name: str, script_path: Path) -> None:
    task_run = (
        "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden "
        f"-File {ps_quote(str(script_path))}"
    )
    completed = subprocess.run(
        ["schtasks", "/Create", "/TN", task_name, "/SC", "ONLOGON", "/TR", task_run, "/F"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to register {task_name}: {completed.stderr or completed.stdout}")


def write_startup_folder_entries(scripts: StartupScripts) -> None:
    startup = startup_folder()
    startup.mkdir(parents=True, exist_ok=True)
    remove_llama_startup_entry()
    (startup / WATCH_STARTUP_NAME).write_text(
        startup_cmd_for(scripts.watch_script),
        encoding="utf-8",
    )


def startup_folder() -> Path:
    return (
        Path.home()
        / "AppData"
        / "Roaming"
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
    )


def startup_cmd_for(script_path: Path) -> str:
    return (
        "@echo off\n"
        f"powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File {cmd_quote(str(script_path))}\n"
    )


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def cmd_quote(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'
