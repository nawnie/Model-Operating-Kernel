"""
src/mok/memory/card_store.py

CardStore — CRUD + JSON persistence for MemoryCards.

Design constraints
------------------
- Purely in-memory during a session; persists to a single JSON file.
- Thread-safe via a single RLock (reads are also locked — simple is correct here).
- No external dependencies.
- Does NOT score, compress, or archive cards — that is FitnessTrainer's job.
  The store only answers: add, get, update, query, persist.

Persistence format
------------------
{
  "version": 1,
  "cards": [ {<card dict>}, ... ]
}

Usage
-----
    store = CardStore()                    # in-memory only
    store = CardStore(path=Path("cards.json"))   # load + auto-save

    cid = store.add(content="route Python → coder", card_type=CardType.ROUTING_HINT)
    card = store.get(cid)
    store.update(cid, usage_count=card.usage_count + 1)
    store.save()
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

from mok.memory.card import (
    CardState,
    CardType,
    MemoryCard,
    new_card_id,
)


_FORMAT_VERSION = 1


class CardStore:
    """
    In-memory card registry with optional JSON persistence.

    All public methods are thread-safe.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._cards: dict[str, MemoryCard] = {}
        self._lock = threading.RLock()
        self._path = path
        if path and path.exists():
            self._load(path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        card_type: CardType | str = CardType.FACT,
        source: str = "unknown",
        tags: list[str] | None = None,
        trust_score: float = 0.5,
        metadata: dict | None = None,
        card_id: str | None = None,
    ) -> str:
        """
        Create a new ACTIVE card and return its card_id.

        If card_id is supplied (e.g. when loading from a file), that ID is used;
        otherwise a fresh UUID-based ID is generated.
        """
        cid = card_id or new_card_id()
        card = MemoryCard(
            card_id=cid,
            card_type=card_type,
            content=content,
            source=source,
            tags=list(tags or []),
            trust_score=max(0.0, min(1.0, trust_score)),
            metadata=dict(metadata or {}),
            created_at=time.time(),
        )
        with self._lock:
            self._cards[cid] = card
        return cid

    def get(self, card_id: str) -> MemoryCard | None:
        with self._lock:
            return self._cards.get(card_id)

    def get_many(self, card_ids: list[str]) -> list[MemoryCard]:
        with self._lock:
            return [self._cards[cid] for cid in card_ids if cid in self._cards]

    def update(self, card_id: str, **fields) -> bool:
        """
        Patch one or more fields on an existing card.
        Returns False if the card does not exist.
        Raises ValueError if a field name is not a valid MemoryCard attribute.
        """
        with self._lock:
            card = self._cards.get(card_id)
            if card is None:
                return False
            for key, value in fields.items():
                if not hasattr(card, key):
                    raise ValueError(f"MemoryCard has no field {key!r}")
                object.__setattr__(card, key, value)  # works for both regular and slots
            return True

    def remove(self, card_id: str) -> bool:
        """Hard-delete a card from the store. Prefer state=PURGED for soft delete."""
        with self._lock:
            return self._cards.pop(card_id, None) is not None

    def upsert(self, card: MemoryCard) -> None:
        """Insert or replace a card by its card_id (used by load and compress)."""
        with self._lock:
            self._cards[card.card_id] = card

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all(self) -> list[MemoryCard]:
        with self._lock:
            return list(self._cards.values())

    def active(self) -> list[MemoryCard]:
        with self._lock:
            return [c for c in self._cards.values() if c.state == CardState.ACTIVE]

    def by_state(self, state: CardState) -> list[MemoryCard]:
        with self._lock:
            return [c for c in self._cards.values() if c.state == state]

    def by_type(self, card_type: CardType | str) -> list[MemoryCard]:
        with self._lock:
            return [c for c in self._cards.values() if str(c.card_type) == str(card_type)]

    def by_tag(self, tag: str) -> list[MemoryCard]:
        with self._lock:
            return [c for c in self._cards.values() if tag in c.tags]

    def filter(self, predicate: Callable[[MemoryCard], bool]) -> list[MemoryCard]:
        with self._lock:
            return [c for c in self._cards.values() if predicate(c)]

    def count(self, state: CardState | None = None) -> int:
        with self._lock:
            if state is None:
                return len(self._cards)
            return sum(1 for c in self._cards.values() if c.state == state)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        """Serialise all cards to JSON. Uses self._path if path is not given."""
        target = path or self._path
        if target is None:
            raise ValueError("No path specified for CardStore.save()")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "version": _FORMAT_VERSION,
                "cards":   [c.to_dict() for c in self._cards.values()],
            }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cards = raw.get("cards", [])
        with self._lock:
            for d in cards:
                try:
                    card = MemoryCard.from_dict(d)
                    self._cards[card.card_id] = card
                except (KeyError, ValueError, TypeError):
                    # Corrupt card — skip gracefully
                    continue

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, card_id: str) -> bool:
        with self._lock:
            return card_id in self._cards
