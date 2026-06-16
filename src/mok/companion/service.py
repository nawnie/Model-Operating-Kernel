from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from mok.companion.config import CompanionConfig, DEFAULT_CONFIG_PATH
from mok.companion.eyes import attention_state_path, write_attention_state
from mok.companion.llama_cpp import chat_completion
from mok.companion.observer import observe_active_window, observe_files
from mok.companion.storage import Datacard, DatacardStore


@dataclass(slots=True)
class CompanionService:
    config: CompanionConfig
    store: DatacardStore

    @classmethod
    def from_config_path(cls, path: Path = DEFAULT_CONFIG_PATH) -> "CompanionService":
        config = CompanionConfig.load(path)
        return cls(config=config, store=DatacardStore(config.db_path))

    def scan_once(self) -> dict:
        cards, result = observe_files(self.config)
        for card in cards:
            self.store.upsert_card(card)
        self.store.log_event("scan", "allow_paths", result.to_dict())
        return result.to_dict()

    def observe_window_once(self) -> Datacard | None:
        card = observe_active_window(self.config)
        if card is None:
            return None
        self.store.upsert_card(card)
        self.store.log_event("window", card.source_uri, {"title": card.title, "lane": card.lane})
        return card

    def tick(self) -> dict:
        scan = self.scan_once()
        window = self.observe_window_once()
        self.publish_attention_state(scan=scan, window=window)
        return {
            "scan": scan,
            "window": window.title if window else None,
            "stats": self.store.stats(),
        }

    def publish_attention_state(self, *, scan: dict, window: Datacard | None) -> None:
        if window is not None:
            target_kind = "window"
            target_label = window.title
            state = "watching"
        elif scan.get("scanned", 0):
            target_kind = "files"
            target_label = f"{scan['scanned']} allowlisted files"
            state = "watching"
        else:
            target_kind = "idle"
            target_label = "watch loop"
            state = "idle"

        try:
            write_attention_state(
                attention_state_path(self.config.storage_dir),
                state=state,
                target_kind=target_kind,
                target_label=target_label,
                detail={"scan": scan, "window": window.title if window else None},
            )
        except OSError as exc:
            self.store.log_event("attention", "state_write_failed", {"error": str(exc)})

    def watch(self, *, iterations: int | None = None) -> None:
        count = 0
        while iterations is None or count < iterations:
            report = self.tick()
            print(
                "tick "
                f"cards={report['stats']['cards']} "
                f"scan_stored={report['scan']['stored']} "
                f"window={report['window'] or '-'}"
            )
            count += 1
            time.sleep(self.config.poll_seconds)

    def chat(self, query: str, *, lane: str | None = None, limit: int = 6) -> str:
        self.store.log_chat_message("user", query)
        cards = self.store.search(query, lane=lane, limit=limit)
        self.publish_chat_attention(query=query, recalled_cards=len(cards))
        prompt = build_memory_prompt(query, cards)
        answer = chat_completion(self.config, prompt)
        self.store.log_chat_message("mok", answer)
        return answer

    def publish_chat_attention(self, *, query: str, recalled_cards: int) -> None:
        try:
            write_attention_state(
                attention_state_path(self.config.storage_dir),
                state="thinking",
                target_kind="chat",
                target_label=query[:120],
                detail={"recalled_cards": recalled_cards},
            )
        except OSError as exc:
            self.store.log_event("attention", "state_write_failed", {"error": str(exc)})


def build_memory_prompt(query: str, cards: list[Datacard]) -> str:
    if not cards:
        return f"User request:\n{query}\n\nNo relevant MoK memory cards were found."
    memory_lines: list[str] = []
    for i, card in enumerate(cards, start=1):
        memory_lines.append(
            f"[{i}] lane={card.lane} source={card.source_uri}\n"
            f"title={card.title}\n"
            f"summary={card.summary}"
        )
    memory = "\n\n".join(memory_lines)
    return (
        "Relevant MoK memory cards:\n"
        f"{memory}\n\n"
        "User request:\n"
        f"{query}\n\n"
        "Answer using the memory cards when they help. Do not invent file contents."
    )
