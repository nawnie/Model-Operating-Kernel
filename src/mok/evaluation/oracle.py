from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(slots=True)
class OracleExample:
    request_id: str
    chosen_expert: str
    expert_scores: dict[str, float]

    @property
    def oracle_expert(self) -> str:
        return max(self.expert_scores, key=self.expert_scores.__getitem__)

    @property
    def oracle_score(self) -> float:
        return self.expert_scores[self.oracle_expert]

    @property
    def chosen_score(self) -> float:
        return self.expert_scores[self.chosen_expert]


def compute_regret(examples: list[OracleExample]) -> dict[str, float]:
    if not examples:
        return {"count": 0, "mean_regret": 0.0, "oracle_match_rate": 0.0}
    regrets = [example.oracle_score - example.chosen_score for example in examples]
    matches = [example.oracle_expert == example.chosen_expert for example in examples]
    return {
        "count": float(len(examples)),
        "mean_regret": round(mean(regrets), 6),
        "oracle_match_rate": mean(1.0 if matched else 0.0 for matched in matches),
    }


# ---------------------------------------------------------------------------
# ROUGE-L implementation (pure stdlib — no external NLP deps)
# ---------------------------------------------------------------------------

def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute the length of the longest common subsequence of token lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Use two-row DP to save memory
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 between *hypothesis* and *reference*.

    Tokenisation: lower-case whitespace split.
    Returns 0.0 when either string is empty.
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0.0:
        return 0.0
    return round(2.0 * precision * recall / (precision + recall), 6)


# ---------------------------------------------------------------------------
# Pluggable scorer protocol
# ---------------------------------------------------------------------------

class ScorerProtocol:
    """
    Interface for pluggable response scorers.

    Subclass and override ``score`` to replace ROUGE-L with, e.g.,
    embedding cosine similarity or an LLM-as-judge approach.
    """

    def score(self, response: str, reference: str) -> float:
        """
        Return a scalar score in [0, 1] measuring response quality.

        Higher = better.  The default implementation is ROUGE-L F1.
        """
        return rouge_l(response, reference)


# ---------------------------------------------------------------------------
# OracleHarness (P4.3)
# ---------------------------------------------------------------------------

import json
from pathlib import Path
from typing import Callable


class OracleHarness:
    """
    Run oracle evaluation over a set of traces and expert responses.

    Workflow
    --------
    1. Load a trace JSONL where each record has:
           request_id, route_expert, expert_response (optional),
           reference (ground-truth answer, optional)
    2. Score each (expert_response, reference) pair with the scorer.
    3. Aggregate into an oracle scores JSONL compatible with export.py.
    4. Also compute regret metrics via compute_regret().

    The harness is intentionally minimal — it does not call live backends.
    Feed it pre-collected (response, reference) pairs.

    Parameters
    ----------
    scorer      : callable or ScorerProtocol instance.
                  Must accept (response: str, reference: str) -> float.
                  Defaults to ROUGE-L F1.
    experts     : ordered list of expert names to include in score dicts.
                  When None, only the routed expert is scored.
    """

    def __init__(
        self,
        scorer: Callable[[str, str], float] | ScorerProtocol | None = None,
        experts: list[str] | None = None,
    ) -> None:
        if scorer is None:
            scorer = ScorerProtocol()
        self._scorer = scorer if callable(scorer) else scorer.score
        self._experts = experts

    # ------------------------------------------------------------------
    # Per-response scoring
    # ------------------------------------------------------------------

    def score_response(self, response: str, reference: str) -> float:
        """
        Score a single (response, reference) pair.

        Returns a float in [0, 1].  Delegates to the configured scorer.
        """
        return float(self._scorer(response, reference))

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        trace_path: Path,
        *,
        expert_responses: dict[str, dict[str, str]] | None = None,
    ) -> dict:
        """
        Evaluate all traces in *trace_path* and return aggregate metrics.

        Parameters
        ----------
        trace_path        : JSONL trace file.  Each record may contain:
                              request_id, route_expert, reference,
                              expert_response  (optional, used when
                              expert_responses dict is not supplied)
        expert_responses  : {request_id: {expert_name: response_text}}
                            Optional override.  When provided, scores
                            all listed experts for each request; otherwise
                            only the routed expert is scored from the
                            trace's ``expert_response`` field.

        Returns
        -------
        dict with keys:
            count            : int   — number of traces evaluated
            mean_regret      : float — from compute_regret()
            oracle_match_rate: float — fraction where routed == oracle expert
            per_expert       : dict[expert, mean_score]
            examples         : list[OracleExample]
        """
        examples: list[OracleExample] = []
        per_expert_scores: dict[str, list[float]] = {}

        for trace in _iter_jsonl(trace_path):
            rid = trace.get("request_id", "")
            chosen = trace.get("route_expert", "")
            reference = str(trace.get("reference", ""))
            if not rid or not chosen:
                continue

            if expert_responses is not None:
                responses_for_req = expert_responses.get(rid, {})
            else:
                raw_response = trace.get("expert_response", "")
                responses_for_req = {chosen: str(raw_response)} if raw_response else {}

            if not responses_for_req:
                continue

            expert_scores: dict[str, float] = {}
            for expert, response in responses_for_req.items():
                score = self.score_response(response, reference)
                expert_scores[expert] = score
                per_expert_scores.setdefault(expert, []).append(score)

            if chosen not in expert_scores:
                continue

            examples.append(OracleExample(
                request_id=rid,
                chosen_expert=chosen,
                expert_scores=expert_scores,
            ))

        regret_stats = compute_regret(examples)
        per_expert_mean = {
            expert: round(mean(scores), 6)
            for expert, scores in per_expert_scores.items()
        }
        return {
            "count": int(regret_stats["count"]),
            "mean_regret": regret_stats["mean_regret"],
            "oracle_match_rate": regret_stats["oracle_match_rate"],
            "per_expert": per_expert_mean,
            "examples": examples,
        }

    # ------------------------------------------------------------------
    # Write oracle scores compatible with export.py
    # ------------------------------------------------------------------

    def write_oracle_scores(
        self,
        examples: list[OracleExample],
        output_path: Path,
        *,
        overwrite: bool = True,
    ) -> int:
        """
        Write oracle scores to a JSONL file consumable by export_training_pairs().

        Format per line:
            {"request_id": "...", "expert_scores": {"coder": 0.92, ...}}

        Returns the number of records written.
        """
        if output_path.exists() and not overwrite:
            return 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with output_path.open("w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps({
                    "request_id": ex.request_id,
                    "expert_scores": ex.expert_scores,
                }) + "\n")
                written += 1
        return written


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
