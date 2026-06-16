from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext

from mok.companion.config import DEFAULT_CONFIG_PATH
from mok.companion.control import (
    lifecycle_status,
    pause_mok,
    restart_mok,
    resume_mok,
    start_mok,
    stop_mok,
)
from mok.companion.llama_cpp import LlamaCppCompanionError
from mok.companion.service import CompanionService
from mok.companion.wakeup import latest_wakeup_note


class MoKTerminal:
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = config_path
        self.service = CompanionService.from_config_path(config_path)
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()

        self.root = tk.Tk()
        self.root.title("MoK Terminal")
        self.root.geometry("900x620")
        self.root.minsize(640, 420)
        self.root.configure(bg="#101410")

        self.output = scrolledtext.ScrolledText(
            self.root,
            bg="#101410",
            fg="#d7f5cf",
            insertbackground="#d7f5cf",
            selectbackground="#345c38",
            font=("Consolas", 11),
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=14,
            pady=12,
        )
        self.output.pack(fill=tk.BOTH, expand=True)
        self.output.configure(state=tk.DISABLED)

        bottom = tk.Frame(self.root, bg="#0b0f0b")
        bottom.pack(fill=tk.X)

        self.prompt = tk.Label(bottom, text="mok>", bg="#0b0f0b", fg="#8ee36f", font=("Consolas", 11))
        self.prompt.pack(side=tk.LEFT, padx=(12, 4), pady=10)

        self.entry = tk.Entry(
            bottom,
            bg="#172017",
            fg="#f4fff1",
            insertbackground="#f4fff1",
            relief=tk.FLAT,
            font=("Consolas", 11),
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), pady=10, ipady=6)
        self.entry.bind("<Return>", self._on_submit)
        self.entry.bind("<Control-l>", lambda _event: self._clear())

        self.send = tk.Button(
            bottom,
            text="Send",
            command=self._submit_current,
            bg="#8ee36f",
            fg="#071007",
            relief=tk.FLAT,
            font=("Consolas", 10, "bold"),
            padx=14,
        )
        self.send.pack(side=tk.RIGHT, padx=(0, 12), pady=10)

        self._write("system", "MoK Terminal ready. Type /help for commands.")
        note = latest_wakeup_note(self.service)
        if note:
            self._write("mok", note)
        self.root.after(100, self._drain_queue)
        self.entry.focus_set()

    def run(self) -> None:
        self.root.mainloop()

    def _on_submit(self, _event) -> str:
        self._submit_current()
        return "break"

    def _submit_current(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self._write("you", text)
        self._dispatch(text)

    def _dispatch(self, text: str) -> None:
        if text in {"/exit", "/quit"}:
            self.root.destroy()
            return
        if text == "/clear":
            self._clear()
            return
        if text == "/help":
            self._write(
                "system",
                "Commands: /help, /clear, /status, /lifecycle, /wakeup, /sleep, /start, /pause, /resume, /stop, /restart, /scan, /recall <query>, /exit. Anything else is chat.",
            )
            return
        if text == "/status":
            self._write("system", format_status(self.service.store.stats()))
            return
        if text in {"/lifecycle", "/start", "/wakeup", "/pause", "/resume", "/stop", "/sleep", "/restart"}:
            threading.Thread(target=self._control_worker, args=(text,), daemon=True).start()
            return
        if text == "/scan":
            threading.Thread(target=self._scan_worker, daemon=True).start()
            return
        if text.startswith("/recall"):
            query = text.removeprefix("/recall").strip()
            self._recall(query)
            return
        threading.Thread(target=self._chat_worker, args=(text,), daemon=True).start()

    def _chat_worker(self, text: str) -> None:
        self.queue.put(("system", "thinking..."))
        try:
            answer = self.service.chat(text)
        except LlamaCppCompanionError as exc:
            answer = f"llama.cpp error: {exc}"
        except Exception as exc:
            answer = f"terminal error: {exc}"
        self.queue.put(("mok", answer))

    def _scan_worker(self) -> None:
        try:
            report = self.service.scan_once()
            self.queue.put(("system", f"scan complete: scanned={report['scanned']} stored={report['stored']} skipped={report['skipped']}"))
        except Exception as exc:
            self.queue.put(("system", f"scan failed: {exc}"))

    def _control_worker(self, command: str) -> None:
        actions = {
            "/lifecycle": lifecycle_status,
            "/start": start_mok,
            "/wakeup": start_mok,
            "/pause": pause_mok,
            "/resume": resume_mok,
            "/stop": stop_mok,
            "/sleep": stop_mok,
            "/restart": restart_mok,
        }
        try:
            result = actions[command](self.config_path)
            self.queue.put(("system", format_lifecycle(result)))
        except Exception as exc:
            self.queue.put(("system", f"{command} failed: {exc}"))

    def _recall(self, query: str) -> None:
        cards = self.service.store.search(query, limit=8)
        if not cards:
            self._write("system", "No matching cards.")
            return
        lines = []
        for card in cards:
            lines.append(f"{card.card_id} lane={card.lane} title={card.title}\n  {card.summary}")
        self._write("memory", "\n".join(lines))

    def _drain_queue(self) -> None:
        while True:
            try:
                role, text = self.queue.get_nowait()
            except queue.Empty:
                break
            self._write(role, text)
        self.root.after(100, self._drain_queue)

    def _write(self, role: str, text: str) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, f"\n{role}> {text.strip()}\n")
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)

    def _clear(self) -> str:
        self.output.configure(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.configure(state=tk.DISABLED)
        return "break"


def format_status(stats: dict) -> str:
    return (
        f"cards={stats['cards']} trainable={stats['trainable']} events={stats['events']}\n"
        f"lanes={stats['lanes']}\n"
        f"db={stats['db_path']}"
    )


def format_lifecycle(status: dict) -> str:
    base = (
        f"state={status['state']}\n"
        f"server_running={status['server_running']} watcher_running={status['watcher_running']} paused={status.get('paused', False)}\n"
        f"server_pids={[p['pid'] for p in status['server']]}\n"
        f"watcher_pids={[p['pid'] for p in status['watcher']]}"
    )
    if status.get("wakeup_note"):
        return base + f"\nwakeup_note={status['wakeup_note']}"
    return base


def run_terminal(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    MoKTerminal(config_path).run()
