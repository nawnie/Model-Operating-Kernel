"""
src/mok/memory/prefetch.py

Next-expert predictor (P5.1) — bigram transition table.

Reads runtime JSONL traces and builds a transition frequency table:
    (current_expert) → {next_expert: count}

The ``predict()`` method returns ranked (expert_name, probability) pairs
so the BudgetManager can warm-load the most-likely-needed expert before
it is requested, reducing cold-miss latency.

Design notes
------------
* Pure stdlib — no numpy/torch at inference time.
* The bigram table is keyed by current expert name only.  A future R3
  implementation can extend the key to (current_expert, prompt_modality)
  or replace it with an MLP while keeping the same ``predict()`` interface.
* Thread-safe reads (builds table at construction time from an immutable
  snapshot of the trace file).

Usage
-----
    predictor = NextExpertPredictor.from_trace_jsonl(Path("traces/runtime.jsonl"))
    hints = predictor.predict("coder", registry)
    # → [("general", 0.61), ("vision", 0.25), ("coordinator", 0.14)]
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from mok.models.registry import ModelRegistry


# ---------------------------------------------------------------------------
# NextExpertPredictor
# ---------------------------------------------------------------------------

class NextExpertPredictor:
    """
    Predict the next expert needed based on the current expert.

    Uses a bigram transition table built from historical JSONL traces.
    The table stores, for each (from_expert, to_expert) pair, the number
    of consecutive request transitions observed in the trace.

    Parameters
    ----------
    transitions : raw count table {from_expert: {to_expert: count}}
    """

    def __init__(self, transitions: dict[str, dict[str, int]]) -> None:
        # Normalise to probabilities once at construction time
        self._probs: dict[str, dict[str, float]] = {}
        for from_expert, counts in transitions.items():
            total = sum(counts.values())
            if total > 0:
                self._probs[from_expert] = {
                    to_exp: cnt / total for to_exp, cnt in counts.items()
                }

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_trace_jsonl(cls, path: Path) -> "NextExpertPredictor":
        """
        Build a predictor from a runtime trace JSONL file.

        Consecutive ``route_expert`` values in the file are treated as
        bigram transitions.  Blank or malformed lines are skipped.

        Raises FileNotFoundError if *path* does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        prev_expert: str | None = None

        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                expert = record.get("route_expert", "")
                if not expert:
                    continue
                if prev_expert is not None and prev_expert != expert:
                    transitions[prev_expert][expert] += 1
                prev_expert = expert

        return cls(dict(transitions))

    @classmethod
    def empty(cls) -> "NextExpertPredictor":
        """Return a predictor with no transition data (always returns [])."""
        return cls({})

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        current_expert: str,
        registry: ModelRegistry,
        *,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return the most likely next experts given *current_expert*.

        Parameters
        ----------
        current_expert : name of the expert currently handling the request
        registry       : used to filter predictions to experts that exist
                         and are not the current expert
        top_k          : if set, return at most this many results

        Returns
        -------
        List of (expert_name, probability) tuples, sorted by probability
        descending.  Returns [] when no transition data exists for
        *current_expert*.
        """
        dist = self._probs.get(current_expert, {})
        if not dist:
            return []

        valid_names = {e.name for e in registry.all()} - {current_expert}
        ranked = sorted(
            ((name, prob) for name, prob in dist.items() if name in valid_names),
            key=lambda x: x[1],
            reverse=True,
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def transition_count(self, from_expert: str, to_expert: str) -> float:
        """
        Return the stored transition probability from → to.
        Returns 0.0 if no data is available.
        """
        return self._probs.get(from_expert, {}).get(to_expert, 0.0)

    def known_experts(self) -> set[str]:
        """Set of experts that appear as *sources* in the transition table."""
        return set(self._probs.keys())

    def __repr__(self) -> str:
        return f"NextExpertPredictor(known={sorted(self.known_experts())})"
