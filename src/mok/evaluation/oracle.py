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
