"""
src/mok/world_eater/absorption_map.py

AbsorptionMap — the data contract between S_Surfer and WorldEater.

S_Surfer surveys Model B against Model A and produces an AbsorptionMap.
WorldEater reads the map and executes the consumption.

Design
------
Each layer in Model B gets one of three verdicts:

  SKIP      — A already knows this well (similarity > SKIP_THRESHOLD).
               Absorbing would only add noise. Leave A's layer untouched.

  ABSORB    — B knows this differently or extends it usefully.
               WorldEater will align, delta-filter, and merge at rate α.

  DISCARD   — B's layer has no useful correspondence to A.
               Could be domain noise or just incompatible structure.

The map also carries metadata: estimated novelty gain, which layers
dominated the absorption decision, and provenance for multi-model runs.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class LayerAction(str, Enum):
    SKIP    = "skip"     # A already has this knowledge
    ABSORB  = "absorb"   # valuable, merge into A
    DISCARD = "discard"  # noise or incompatible


@dataclass
class LayerCorrespondence:
    """One row in the AbsorptionMap — a single layer-pair decision.

    Attributes
    ----------
    name_a          : tensor name in Model A (e.g. "blk.12.attn_q.weight")
    name_b          : tensor name in Model B
    depth_a         : normalised layer depth in A (0.0 = input, 1.0 = output)
    depth_b         : normalised layer depth in B
    role            : functional role ("attn_q", "ffn_up", "embed", …)
    similarity      : principal-angle cosine similarity [0, 1]
                      1 = identical subspaces, 0 = completely orthogonal
    novelty         : fraction of B's subspace not spanned by A [0, 1]
                      high novelty + good role match → ABSORB
    action          : SKIP / ABSORB / DISCARD
    alpha           : absorption rate applied by WorldEater (0.0–1.0)
                      computed by conflict-check; None until WorldEater fills it
    shape_a         : weight matrix shape in A
    shape_b         : weight matrix shape in B
    notes           : human-readable reason for the decision
    """
    name_a:      str
    name_b:      str
    depth_a:     float
    depth_b:     float
    role:        str
    similarity:  float
    novelty:     float
    action:      LayerAction
    alpha:       float | None = None     # filled in by WorldEater
    shape_a:     tuple[int, ...] = field(default_factory=tuple)
    shape_b:     tuple[int, ...] = field(default_factory=tuple)
    notes:       str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LayerCorrespondence":
        d = dict(d)
        d["action"] = LayerAction(d["action"])
        d["shape_a"] = tuple(d.get("shape_a", ()))
        d["shape_b"] = tuple(d.get("shape_b", ()))
        return cls(**d)


@dataclass
class AbsorptionMap:
    """The full survey result produced by S_Surfer.

    Attributes
    ----------
    model_a_path    : path to the host model (will be modified)
    model_b_path    : path to the prey model (read-only)
    survey_time     : ISO timestamp of when the survey was run
    architecture    : detected architecture tag ("llama", "mistral", …)
    layers          : ordered list of LayerCorrespondence entries
    n_skip          : count of SKIP decisions
    n_absorb        : count of ABSORB decisions
    n_discard       : count of DISCARD decisions
    mean_similarity : average similarity across all layers
    mean_novelty    : average novelty across ABSORB layers
    estimated_gain  : heuristic score for how much A might improve (0–1)
    meal_index      : which meal this is in a multi-model consumption run (0-indexed)
    prior_meals     : list of previously consumed model paths (for provenance)
    metadata        : arbitrary extra info
    """
    model_a_path:    str
    model_b_path:    str
    survey_time:     str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    architecture:    str = "unknown"
    layers:          list[LayerCorrespondence] = field(default_factory=list)
    n_skip:          int = 0
    n_absorb:        int = 0
    n_discard:       int = 0
    mean_similarity: float = 0.0
    mean_novelty:    float = 0.0
    estimated_gain:  float = 0.0
    meal_index:      int = 0
    prior_meals:     list[str] = field(default_factory=list)
    metadata:        dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def absorb_layers(self) -> list[LayerCorrespondence]:
        return [l for l in self.layers if l.action == LayerAction.ABSORB]

    @property
    def skip_layers(self) -> list[LayerCorrespondence]:
        return [l for l in self.layers if l.action == LayerAction.SKIP]

    @property
    def discard_layers(self) -> list[LayerCorrespondence]:
        return [l for l in self.layers if l.action == LayerAction.DISCARD]

    def recompute_stats(self) -> None:
        """Recompute aggregate stats from the layer list. Call after modifying layers."""
        self.n_skip    = sum(1 for l in self.layers if l.action == LayerAction.SKIP)
        self.n_absorb  = sum(1 for l in self.layers if l.action == LayerAction.ABSORB)
        self.n_discard = sum(1 for l in self.layers if l.action == LayerAction.DISCARD)

        all_sim = [l.similarity for l in self.layers]
        self.mean_similarity = round(sum(all_sim) / len(all_sim), 4) if all_sim else 0.0

        absorb_nov = [l.novelty for l in self.absorb_layers]
        self.mean_novelty = round(sum(absorb_nov) / len(absorb_nov), 4) if absorb_nov else 0.0

        # Heuristic gain: fraction of layers absorbed * mean novelty of those layers
        if self.layers:
            self.estimated_gain = round(
                (self.n_absorb / len(self.layers)) * self.mean_novelty, 4
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layers"] = [l.to_dict() for l in self.layers]
        return d

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AbsorptionMap":
        d = json.loads(path.read_text(encoding="utf-8"))
        layers = [LayerCorrespondence.from_dict(l) for l in d.pop("layers", [])]
        return cls(**d, layers=layers)

    def summary(self) -> str:
        lines = [
            f"AbsorptionMap  meal #{self.meal_index}",
            f"  A: {Path(self.model_a_path).name}",
            f"  B: {Path(self.model_b_path).name}",
            f"  arch: {self.architecture}",
            f"  layers: {len(self.layers)} total  "
            f"SKIP={self.n_skip}  ABSORB={self.n_absorb}  DISCARD={self.n_discard}",
            f"  mean similarity: {self.mean_similarity:.3f}",
            f"  mean novelty (absorb):  {self.mean_novelty:.3f}",
            f"  estimated gain: {self.estimated_gain:.3f}",
        ]
        return "\n".join(lines)
