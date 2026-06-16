"""
src/mok/memory/card.py

Data layer for the MoK Memory Fitness Trainer.

Concepts
--------
MemoryCard   — a single unit of structured knowledge (routing hint, expert
               profile, user preference, task pattern, fact).
CardState    — lifecycle state of a card.
UsageSignal  — one logged usage event: card was retrieved, outcome observed.
FitnessScore — computed quality metrics for a single card at a point in time.

No IO, no external dependencies.  The card_store and fitness_trainer modules
build on these types.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Card lifecycle state
# ---------------------------------------------------------------------------

class CardState(str, Enum):
    ACTIVE             = "active"             # normal — retrievable, scoreable
    ARCHIVED           = "archived"           # low-value; cold storage, not retrieved
    TRAINING_CANDIDATE = "training_candidate" # meets promotion thresholds; queued for batch
    PURGED             = "purged"             # deleted; kept as tombstone only


# ---------------------------------------------------------------------------
# Card types (open set — extend as needed)
# ---------------------------------------------------------------------------

class CardType(str, Enum):
    ROUTING_HINT   = "routing_hint"    # "when user says X → route to coder"
    EXPERT_PROFILE = "expert_profile"  # capabilities, quirks, cost of an expert
    USER_PREF      = "user_pref"       # style, verbosity, preferred tools
    TASK_PATTERN   = "task_pattern"    # "write-then-test" → code → tester chain
    FACT           = "fact"            # domain knowledge pulled in from a trace
    HEURISTIC      = "heuristic"       # router rule not yet compiled into R0
    SUMMARY        = "summary"         # compressed merge of 2+ other cards


# ---------------------------------------------------------------------------
# MemoryCard
# ---------------------------------------------------------------------------

@dataclass
class MemoryCard:
    """
    One unit of knowledge managed by the fitness trainer.

    Identity fields are immutable after creation.
    Fitness fields (usage_count, outcome_scores, etc.) are updated in-place
    by FitnessTrainer after each usage signal is processed.
    """

    # Identity (set at creation, never changed)
    card_id: str
    card_type: CardType | str
    content: str                         # the actual knowledge text
    source: str                          # where this came from: "trace", "user", "rsi", …
    tags: list[str] = field(default_factory=list)

    # Lifecycle
    state: CardState = CardState.ACTIVE
    created_at: float = field(default_factory=time.time)

    # Fitness tracking (mutated by FitnessTrainer)
    usage_count: int = 0
    last_used_at: float = 0.0           # wall-clock timestamp; 0 = never used
    outcome_scores: list[float] = field(default_factory=list)   # per-usage 0–1
    trust_score: float = 0.5            # source-quality × stability; 0–1
    stability_score: float = 1.0        # decreases when content is edited; 0–1

    # Compression provenance
    compressed_from: list[str] = field(default_factory=list)    # source card IDs
    superseded_by: str | None = None                             # card_id that replaced this

    # Training gate
    training_promoted_at: float | None = None                   # set when promoted

    # Arbitrary metadata (version vectors, user context, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def mean_outcome(self) -> float:
        """Average outcome score across all usage signals. 0 if never used."""
        if not self.outcome_scores:
            return 0.0
        return sum(self.outcome_scores) / len(self.outcome_scores)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def seconds_since_last_use(self) -> float | None:
        """None if never used."""
        if self.last_used_at == 0.0:
            return None
        return time.time() - self.last_used_at

    @property
    def is_active(self) -> bool:
        return self.state == CardState.ACTIVE

    @property
    def is_retrievable(self) -> bool:
        """Only ACTIVE cards are returned to callers."""
        return self.state == CardState.ACTIVE

    # ------------------------------------------------------------------
    # Serialisation (no json import needed — card_store handles that)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "card_id":              self.card_id,
            "card_type":            self.card_type.value if hasattr(self.card_type, "value") else str(self.card_type),
            "content":              self.content,
            "source":               self.source,
            "tags":                 self.tags,
            "state":                self.state.value,
            "created_at":           self.created_at,
            "usage_count":          self.usage_count,
            "last_used_at":         self.last_used_at,
            "outcome_scores":       self.outcome_scores,
            "trust_score":          self.trust_score,
            "stability_score":      self.stability_score,
            "compressed_from":      self.compressed_from,
            "superseded_by":        self.superseded_by,
            "training_promoted_at": self.training_promoted_at,
            "metadata":             self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryCard":
        return cls(
            card_id=d["card_id"],
            card_type=d.get("card_type", CardType.FACT),
            content=d["content"],
            source=d.get("source", "unknown"),
            tags=d.get("tags", []),
            state=CardState(d.get("state", "active")),
            created_at=float(d.get("created_at", time.time())),
            usage_count=int(d.get("usage_count", 0)),
            last_used_at=float(d.get("last_used_at", 0.0)),
            outcome_scores=list(d.get("outcome_scores", [])),
            trust_score=float(d.get("trust_score", 0.5)),
            stability_score=float(d.get("stability_score", 1.0)),
            compressed_from=list(d.get("compressed_from", [])),
            superseded_by=d.get("superseded_by"),
            training_promoted_at=d.get("training_promoted_at"),
            metadata=dict(d.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# UsageSignal
# ---------------------------------------------------------------------------

@dataclass
class UsageSignal:
    """
    One observed usage event for a memory card.

    Logged every time a card is retrieved and an outcome is available.
    The stream of signals is the primary input to FitnessTrainer.
    """

    signal_id:      str    = field(default_factory=lambda: str(uuid.uuid4()))
    card_id:        str    = ""
    request_id:     str    = ""
    context_excerpt: str   = ""     # truncated prompt (max ~200 chars)
    outcome_score:  float  = 0.0   # 0 (useless / wrong) → 1 (perfect / confirmed)
    router_tier:    str    = "R0"
    expert_used:    str    = ""
    timestamp:      float  = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "signal_id":      self.signal_id,
            "card_id":        self.card_id,
            "request_id":     self.request_id,
            "context_excerpt": self.context_excerpt,
            "outcome_score":  self.outcome_score,
            "router_tier":    self.router_tier,
            "expert_used":    self.expert_used,
            "timestamp":      self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UsageSignal":
        return cls(
            signal_id=d.get("signal_id", str(uuid.uuid4())),
            card_id=d.get("card_id", ""),
            request_id=d.get("request_id", ""),
            context_excerpt=d.get("context_excerpt", ""),
            outcome_score=float(d.get("outcome_score", 0.0)),
            router_tier=d.get("router_tier", "R0"),
            expert_used=d.get("expert_used", ""),
            timestamp=float(d.get("timestamp", time.time())),
        )


# ---------------------------------------------------------------------------
# FitnessScore
# ---------------------------------------------------------------------------

@dataclass
class FitnessScore:
    """
    Computed quality snapshot for one card at one point in time.

    weight             — how much this card earns its place in memory.
                         High = frequently used AND recently used AND good outcomes.
    trust              — how reliable the card's content is.
                         High = good source × stable content × high outcome correlation.
    retention_priority — combined rank: weight × trust.
    training_eligible  — True if this card has met all thresholds for the
                         training batch gate.
    notes              — human-readable explanation of the score.
    """

    card_id:            str
    weight:             float    # 0–∞  (not normalised — compare within corpus)
    trust:              float    # 0–1
    retention_priority: float    # weight × trust
    training_eligible:  bool
    notes:              str = ""

    # Individual sub-scores for debugging
    recency_factor:     float = 0.0
    usage_factor:       float = 0.0
    outcome_factor:     float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_card_id() -> str:
    return f"card-{uuid.uuid4().hex[:12]}"


def recency_decay(seconds_since: float | None, half_life_seconds: float = 86_400.0) -> float:
    """
    Exponential decay: 1.0 at t=0, 0.5 at t=half_life_seconds.
    Returns 0.0 if the card has never been used.
    """
    if seconds_since is None:
        return 0.0
    lam = math.log(2) / half_life_seconds
    return math.exp(-lam * seconds_since)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Token-level Jaccard similarity.  Cheap proxy for semantic overlap.
    Suitable for detecting near-duplicate cards without embeddings.
    """
    tokens_a = _tokenise(text_a)
    tokens_b = _tokenise(text_b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    inter = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(inter) / len(union)


_STOP_WORDS = frozenset(
    "a an the is are was were be been have has had do does did "
    "will would could should may might i you he she it we they "
    "me him her us them my your his its our their and or but not "
    "in on at by for with to of this that these those so if then "
    "when where how what which who".split()
)


def _tokenise(text: str) -> frozenset[str]:
    import re
    tokens = re.findall(r"[a-z][a-z0-9_]{1,}", text.lower())
    return frozenset(t for t in tokens if t not in _STOP_WORDS)
