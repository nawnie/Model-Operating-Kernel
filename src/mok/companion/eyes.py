from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mok.companion.config import DEFAULT_COMPANION_DIR


DEFAULT_ATTENTION_FILENAME = "attention.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(slots=True)
class AttentionState:
    state: str = "idle"
    target_kind: str = "idle"
    target_label: str = ""
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gaze_x"] = clamp(float(data["gaze_x"]), -1.0, 1.0)
        data["gaze_y"] = clamp(float(data["gaze_y"]), -1.0, 1.0)
        return data


def attention_state_path(storage_dir: Path | None = None) -> Path:
    return (storage_dir or DEFAULT_COMPANION_DIR) / DEFAULT_ATTENTION_FILENAME


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def infer_gaze(target_kind: str, target_label: str = "") -> tuple[float, float]:
    """Map an attention target to an approximate cartoon-eye direction."""

    kind = (target_kind or "idle").lower()
    label = target_label or ""
    biases = {
        "idle": (0.0, 0.0),
        "stale": (0.0, 0.35),
        "window": (0.25, -0.05),
        "files": (-0.45, 0.25),
        "file": (-0.45, 0.25),
        "scan": (-0.55, 0.20),
        "chat": (0.0, -0.30),
        "routing": (0.35, -0.30),
        "model": (0.45, -0.25),
        "error": (0.0, 0.55),
    }
    bias_x, bias_y = biases.get(kind, (0.0, 0.0))
    digest = hashlib.blake2b(f"{kind}:{label}".encode("utf-8", errors="replace"), digest_size=2).digest()
    jitter_x = (digest[0] / 255.0) * 2.0 - 1.0
    jitter_y = (digest[1] / 255.0) * 2.0 - 1.0
    return (
        clamp(bias_x + jitter_x * 0.35, -1.0, 1.0),
        clamp(bias_y + jitter_y * 0.25, -1.0, 1.0),
    )


def normalize_state(raw: dict[str, Any] | None) -> AttentionState:
    raw = dict(raw or {})
    kind = str(raw.get("target_kind") or raw.get("kind") or "idle")
    label = str(raw.get("target_label") or raw.get("label") or "")
    inferred_x, inferred_y = infer_gaze(kind, label)
    return AttentionState(
        state=str(raw.get("state") or "idle"),
        target_kind=kind,
        target_label=label,
        gaze_x=clamp(float(raw.get("gaze_x", inferred_x)), -1.0, 1.0),
        gaze_y=clamp(float(raw.get("gaze_y", inferred_y)), -1.0, 1.0),
        detail=dict(raw.get("detail") or {}),
        updated_at=str(raw.get("updated_at") or utc_now()),
    )


def read_attention_state(path: Path) -> AttentionState:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return normalize_state(None)
    except (OSError, json.JSONDecodeError):
        return normalize_state({"state": "error", "target_kind": "error", "target_label": "attention state unreadable"})
    return normalize_state(raw if isinstance(raw, dict) else None)


def write_attention_state(
    path: Path,
    *,
    state: str,
    target_kind: str,
    target_label: str = "",
    gaze_x: float | None = None,
    gaze_y: float | None = None,
    detail: dict[str, Any] | None = None,
) -> AttentionState:
    inferred_x, inferred_y = infer_gaze(target_kind, target_label)
    attention = AttentionState(
        state=state,
        target_kind=target_kind,
        target_label=target_label,
        gaze_x=clamp(float(gaze_x if gaze_x is not None else inferred_x), -1.0, 1.0),
        gaze_y=clamp(float(gaze_y if gaze_y is not None else inferred_y), -1.0, 1.0),
        detail=dict(detail or {}),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(attention.to_dict(), indent=2), encoding="utf-8")
    temp_path.replace(path)
    return attention


def is_stale(state: AttentionState, stale_seconds: float) -> bool:
    if stale_seconds <= 0:
        return False
    try:
        updated = datetime.fromisoformat(state.updated_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    return (datetime.now(timezone.utc) - updated).total_seconds() > stale_seconds


def run_overlay(
    *,
    state_path: Path,
    topmost: bool = False,
    frameless: bool = False,
    demo: bool = False,
    poll_ms: int = 150,
    stale_seconds: float = 15.0,
) -> None:
    try:
        import tkinter as tk
    except ImportError as exc:
        raise SystemExit("Tkinter is required for the MoK eyes overlay.") from exc

    root = tk.Tk()
    root.title("MoK Eyes")
    root.geometry("260x118")
    root.resizable(False, False)
    root.attributes("-topmost", bool(topmost))
    if frameless:
        root.overrideredirect(True)

    overlay = EyeOverlay(
        root=root,
        canvas=tk.Canvas(root, width=260, height=118, highlightthickness=0),
        state_path=state_path,
        topmost=topmost,
        demo=demo,
        poll_ms=max(40, poll_ms),
        stale_seconds=stale_seconds,
    )
    overlay.canvas.pack(fill="both", expand=True)
    root.bind("<Escape>", lambda _event: root.destroy())
    root.bind("t", lambda _event: overlay.toggle_topmost())
    root.bind("T", lambda _event: overlay.toggle_topmost())
    root.bind("<ButtonPress-1>", overlay.begin_drag)
    root.bind("<B1-Motion>", overlay.drag)
    overlay.tick()
    root.mainloop()


class EyeOverlay:
    def __init__(
        self,
        *,
        root: Any,
        canvas: Any,
        state_path: Path,
        topmost: bool,
        demo: bool,
        poll_ms: int,
        stale_seconds: float,
    ) -> None:
        self.root = root
        self.canvas = canvas
        self.state_path = state_path
        self.topmost = topmost
        self.demo = demo
        self.poll_ms = poll_ms
        self.stale_seconds = stale_seconds
        self.gaze_x = 0.0
        self.gaze_y = 0.0
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.next_blink_at = time.monotonic() + random.uniform(2.5, 5.5)
        self.blink_until = 0.0

    def toggle_topmost(self) -> None:
        self.topmost = not self.topmost
        self.root.attributes("-topmost", self.topmost)

    def begin_drag(self, event: Any) -> None:
        self.drag_offset_x = int(event.x)
        self.drag_offset_y = int(event.y)

    def drag(self, event: Any) -> None:
        x = self.root.winfo_pointerx() - self.drag_offset_x
        y = self.root.winfo_pointery() - self.drag_offset_y
        self.root.geometry(f"+{x}+{y}")

    def tick(self) -> None:
        state = self.demo_state() if self.demo else read_attention_state(self.state_path)
        if is_stale(state, self.stale_seconds):
            state = normalize_state({"state": "stale", "target_kind": "stale", "target_label": "no recent watch update"})
        self.gaze_x += (state.gaze_x - self.gaze_x) * 0.24
        self.gaze_y += (state.gaze_y - self.gaze_y) * 0.24
        self.draw(state)
        self.root.after(self.poll_ms, self.tick)

    def demo_state(self) -> AttentionState:
        targets = [
            ("watching", "window", "VS Code - service.py"),
            ("watching", "files", "allowlisted project files"),
            ("thinking", "routing", "selecting expert"),
            ("idle", "idle", "waiting for activity"),
        ]
        state, kind, label = targets[int(time.monotonic() // 2) % len(targets)]
        return normalize_state(
            {
                "state": state,
                "target_kind": kind,
                "target_label": label,
                "gaze_x": math.sin(time.monotonic() * 1.8) * 0.75,
                "gaze_y": math.cos(time.monotonic()) * 0.45,
            }
        )

    def draw(self, state: AttentionState) -> None:
        now = time.monotonic()
        if now >= self.next_blink_at:
            self.blink_until = now + 0.12
            self.next_blink_at = now + random.uniform(2.5, 6.0)
        blinking = now < self.blink_until
        quiet = state.state.lower() in {"sleep", "sleeping", "stale"} or state.target_kind.lower() == "stale"

        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 260, 118, fill="#1d1e24", outline="")
        self.canvas.create_text(130, 16, text="MoK is awake" if not quiet else "MoK is quiet", fill="#d8d8df", font=("Segoe UI", 10, "bold"))
        self.draw_eye(86, 58, blinking=blinking, quiet=quiet)
        self.draw_eye(174, 58, blinking=blinking, quiet=quiet)
        label = state.target_label or state.target_kind or state.state
        if len(label) > 34:
            label = f"{label[:31]}..."
        self.canvas.create_text(130, 103, text=f"{state.target_kind}: {label}", fill="#b9bac3", font=("Segoe UI", 8))

    def draw_eye(self, cx: int, cy: int, *, blinking: bool, quiet: bool) -> None:
        if blinking:
            self.canvas.create_line(cx - 32, cy, cx + 32, cy, fill="#f6f3e8", width=5)
            return
        openness = 0.58 if quiet else 1.0
        half_h = 22 * openness
        self.canvas.create_oval(cx - 32, cy - half_h, cx + 32, cy + half_h, fill="#f6f3e8", outline="#08090b", width=3)
        pupil_x = cx + self.gaze_x * 15
        pupil_y = cy + self.gaze_y * 10 * openness
        self.canvas.create_oval(pupil_x - 10, pupil_y - 10, pupil_x + 10, pupil_y + 10, fill="#111217", outline="")
        self.canvas.create_oval(pupil_x - 4, pupil_y - 5, pupil_x - 1, pupil_y - 2, fill="#ffffff", outline="")
        if quiet:
            self.canvas.create_arc(cx - 32, cy - 24, cx + 32, cy + 10, start=0, extent=180, fill="#1d1e24", outline="#1d1e24")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open the MoK cartoon eyes attention cue.")
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--storage-dir", type=Path, default=None)
    parser.add_argument("--topmost", action="store_true", help="Keep the eyes above other windows.")
    parser.add_argument("--frameless", action="store_true", help="Remove normal window decorations.")
    parser.add_argument("--demo", action="store_true", help="Animate fake attention targets for visual testing.")
    parser.add_argument("--poll-ms", type=int, default=150, help="Attention state poll interval.")
    parser.add_argument("--stale-seconds", type=float, default=15.0, help="Dim eyes if state updates stop.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_overlay(
        state_path=args.state_path or attention_state_path(args.storage_dir),
        topmost=args.topmost,
        frameless=args.frameless,
        demo=args.demo,
        poll_ms=args.poll_ms,
        stale_seconds=args.stale_seconds,
    )


if __name__ == "__main__":
    main()
