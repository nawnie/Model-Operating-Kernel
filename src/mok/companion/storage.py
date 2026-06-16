from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1


@dataclass(slots=True)
class Datacard:
    card_id: str
    lane: str
    source_type: str
    source_uri: str
    title: str
    content: str
    summary: str
    tags: list[str] = field(default_factory=list)
    trust_score: float = 0.5
    trainable: bool = False
    content_hash: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def build(
        cls,
        *,
        lane: str,
        source_type: str,
        source_uri: str,
        title: str,
        content: str,
        summary: str,
        tags: list[str] | None = None,
        trust_score: float = 0.5,
        trainable: bool = False,
    ) -> "Datacard":
        content_hash = hash_content(source_type, source_uri, content)
        return cls(
            card_id="dc-" + content_hash[:16],
            lane=lane,
            source_type=source_type,
            source_uri=source_uri,
            title=title,
            content=content,
            summary=summary,
            tags=list(tags or []),
            trust_score=max(0.0, min(1.0, trust_score)),
            trainable=trainable,
            content_hash=content_hash,
        )


def hash_content(source_type: str, source_uri: str, content: str) -> str:
    h = hashlib.sha256()
    h.update(source_type.encode("utf-8", errors="replace"))
    h.update(b"\0")
    h.update(source_uri.encode("utf-8", errors="replace"))
    h.update(b"\0")
    h.update(content.encode("utf-8", errors="replace"))
    return h.hexdigest()


class DatacardStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS datacards (
                    card_id TEXT PRIMARY KEY,
                    lane TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    trust_score REAL NOT NULL,
                    trainable INTEGER NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cards_lane ON datacards(lane)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cards_source ON datacards(source_uri)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cards_updated ON datacards(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_created ON chat_messages(created_at)")
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def upsert_card(self, card: Datacard) -> str:
        now = time.time()
        if not card.content_hash:
            card.content_hash = hash_content(card.source_type, card.source_uri, card.content)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT card_id, created_at FROM datacards WHERE content_hash = ?",
                (card.content_hash,),
            ).fetchone()
            created_at = float(existing["created_at"]) if existing else card.created_at
            card_id = str(existing["card_id"]) if existing else card.card_id
            conn.execute(
                """
                INSERT INTO datacards (
                    card_id, lane, source_type, source_uri, title, content, summary,
                    tags_json, trust_score, trainable, content_hash, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(card_id) DO UPDATE SET
                    lane=excluded.lane,
                    source_type=excluded.source_type,
                    source_uri=excluded.source_uri,
                    title=excluded.title,
                    content=excluded.content,
                    summary=excluded.summary,
                    tags_json=excluded.tags_json,
                    trust_score=excluded.trust_score,
                    trainable=excluded.trainable,
                    content_hash=excluded.content_hash,
                    updated_at=excluded.updated_at
                """,
                (
                    card_id,
                    card.lane,
                    card.source_type,
                    card.source_uri,
                    card.title,
                    card.content,
                    card.summary,
                    json.dumps(card.tags),
                    card.trust_score,
                    1 if card.trainable else 0,
                    card.content_hash,
                    created_at,
                    now,
                ),
            )
        return card_id

    def log_event(self, event_type: str, source_uri: str, payload: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events(event_type, source_uri, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (event_type, source_uri, json.dumps(payload), time.time()),
            )

    def log_chat_message(self, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages(role, content, created_at)
                VALUES (?, ?, ?)
                """,
                (role, content, time.time()),
            )

    def latest_event(self, event_type: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT event_type, source_uri, payload_json, created_at
                FROM events
                WHERE event_type = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (event_type,),
            ).fetchone()
        if row is None:
            return None
        return {
            "event_type": row["event_type"],
            "source_uri": row["source_uri"],
            "payload": json.loads(row["payload_json"]),
            "created_at": float(row["created_at"]),
        }

    def recent_chat(self, limit: int = 12) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM chat_messages
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {"role": row["role"], "content": row["content"], "created_at": float(row["created_at"])}
            for row in reversed(rows)
        ]

    def latest(self, limit: int = 10) -> list[Datacard]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM datacards ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [row_to_card(row) for row in rows]

    def search(self, query: str, *, lane: str | None = None, limit: int = 10) -> list[Datacard]:
        terms = [term.lower() for term in query.split() if term.strip()]
        if not terms:
            return self.latest(limit)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM datacards WHERE (? IS NULL OR lane = ?) ORDER BY updated_at DESC",
                (lane, lane),
            ).fetchall()
        cards = [row_to_card(row) for row in rows]
        ranked = sorted(
            ((score_card(card, terms), card) for card in cards),
            key=lambda item: item[0],
            reverse=True,
        )
        return [card for score, card in ranked if score > 0][:limit]

    def stats(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM datacards").fetchone()["n"]
            lanes = conn.execute(
                "SELECT lane, COUNT(*) AS n FROM datacards GROUP BY lane ORDER BY n DESC"
            ).fetchall()
            trainable = conn.execute(
                "SELECT COUNT(*) AS n FROM datacards WHERE trainable = 1"
            ).fetchone()["n"]
            events = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        return {
            "db_path": str(self.path),
            "cards": int(total),
            "trainable": int(trainable),
            "events": int(events),
            "lanes": {str(row["lane"]): int(row["n"]) for row in lanes},
        }


def row_to_card(row: sqlite3.Row) -> Datacard:
    return Datacard(
        card_id=row["card_id"],
        lane=row["lane"],
        source_type=row["source_type"],
        source_uri=row["source_uri"],
        title=row["title"],
        content=row["content"],
        summary=row["summary"],
        tags=json.loads(row["tags_json"]),
        trust_score=float(row["trust_score"]),
        trainable=bool(row["trainable"]),
        content_hash=row["content_hash"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def score_card(card: Datacard, terms: Iterable[str]) -> int:
    haystack = " ".join(
        [card.lane, card.source_uri, card.title, card.summary, card.content, " ".join(card.tags)]
    ).lower()
    score = 0
    for term in terms:
        if term in haystack:
            score += haystack.count(term)
    return score
