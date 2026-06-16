from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from mok.companion.cartographer import card_from_file, card_from_window_title
from mok.companion.config import CompanionConfig
from mok.companion.storage import Datacard


@dataclass(slots=True)
class ObservationResult:
    scanned: int = 0
    stored: int = 0
    skipped: int = 0
    errors: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "stored": self.stored,
            "skipped": self.skipped,
            "errors": list(self.errors or []),
        }


def iter_allowed_files(config: CompanionConfig) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    errors: list[str] = []
    extensions = {ext.lower() for ext in config.file_extensions}
    deny_names = {name.lower() for name in config.deny_dir_names}

    for root in config.allow_paths:
        try:
            resolved = root.expanduser().resolve()
        except OSError as exc:
            errors.append(f"{root}: {exc}")
            continue
        if not resolved.exists():
            errors.append(f"{resolved}: path does not exist")
            continue
        if resolved.is_file():
            candidates = [resolved]
        else:
            candidates = []
            for path in resolved.rglob("*"):
                if any(part.lower() in deny_names for part in path.parts):
                    continue
                if path.is_file():
                    candidates.append(path)
        for path in candidates:
            if path.suffix.lower() not in extensions:
                continue
            try:
                if path.stat().st_size > config.max_file_bytes:
                    continue
            except OSError as exc:
                errors.append(f"{path}: {exc}")
                continue
            files.append(path)
            if len(files) >= config.max_cards_per_scan:
                return files, errors
    return files, errors


def read_text_file(path: Path) -> str | None:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def observe_files(config: CompanionConfig) -> tuple[list[Datacard], ObservationResult]:
    cards: list[Datacard] = []
    files, errors = iter_allowed_files(config)
    result = ObservationResult(scanned=len(files), errors=errors)
    for path in files:
        content = read_text_file(path)
        if not content:
            result.skipped += 1
            continue
        cards.append(card_from_file(path, content))
        result.stored += 1
    return cards, result


def observe_active_window(config: CompanionConfig) -> Datacard | None:
    if not config.capture_window_titles:
        return None
    title = get_active_window_title()
    if not title:
        return None
    lowered = title.lower()
    if any(keyword.lower() in lowered for keyword in config.deny_window_keywords):
        return None
    return card_from_window_title(title)


def get_active_window_title() -> str:
    script = r"""
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class Win32Window {
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll", CharSet=CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
}
"@
$builder = New-Object System.Text.StringBuilder 512
$handle = [Win32Window]::GetForegroundWindow()
[void][Win32Window]::GetWindowText($handle, $builder, $builder.Capacity)
$builder.ToString()
"""
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()
