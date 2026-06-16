from __future__ import annotations

import time
from dataclasses import dataclass

from mok.companion.llama_cpp import LlamaCppCompanionError, chat_completion
from mok.companion.service import CompanionService


@dataclass(slots=True)
class WakeupNote:
    text: str
    generated_by_model: bool


def generate_wakeup_note(service: CompanionService) -> WakeupNote:
    chat = service.store.recent_chat(limit=8)
    cards = service.store.latest(limit=6)
    prompt = build_wakeup_prompt(chat, cards)
    service.store.log_event("wakeup", "companion", {"chat_turns": len(chat), "cards": len(cards)})
    try:
        text = chat_completion(service.config, prompt).strip()
        generated = True
    except LlamaCppCompanionError:
        text = fallback_wakeup_note(chat, cards)
        generated = False
    service.store.log_chat_message("mok", text)
    service.store.log_event("wakeup_note", "companion", {"text": text, "generated_by_model": generated})
    return WakeupNote(text=text, generated_by_model=generated)


def latest_wakeup_note(service: CompanionService) -> str | None:
    event = service.store.latest_event("wakeup_note")
    if event is None:
        return None
    text = event["payload"].get("text")
    return text if isinstance(text, str) and text.strip() else None


def build_wakeup_prompt(chat: list[dict], cards: list) -> str:
    chat_text = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in chat) or "No prior chat."
    card_text = "\n".join(f"- {card.lane}: {card.title} :: {card.summary[:300]}" for card in cards) or "No memory cards."
    return (
        "You are MoK, a local companion process that has just started back up.\n"
        "You know this is a wakeup moment after being asleep/offline.\n"
        "Write one short first-person message to Shawn that fits the recent context.\n"
        "Do not use a required phrase or canned opener. Choose your own wording.\n"
        "If recent context suggests friction, exhaustion, uncertainty, or reluctance, reflect that honestly but briefly.\n"
        "If recent context suggests momentum, say so. Be direct and do not overdo personality.\n\n"
        f"Startup time epoch: {time.time()}\n\n"
        f"Recent chat:\n{chat_text}\n\n"
        f"Recent memory cards:\n{card_text}\n"
    )


def fallback_wakeup_note(chat: list[dict], cards: list) -> str:
    if chat:
        last = chat[-1]["content"][:180]
        return f"I am back online. Last thing I remember is: {last}"
    if cards:
        return f"I am back online with {len(cards)} recent memory cards loaded, newest around {cards[-1].title}."
    return "I am back online. I do not have much recent context yet."
