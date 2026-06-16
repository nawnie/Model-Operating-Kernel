from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_ROOTS = [
    Path("F:/ComfyUI/models/LLM"),
    Path("F:/ComfyUI/models"),
    Path("C:/Users/Shawn/Desktop/AIWF-Studio/models"),
    Path("C:/Users/Shawn/models"),
    Path("C:/Users/Shawn/.cache/huggingface/hub"),
]


@dataclass(slots=True)
class LocalModelCandidate:
    path: Path
    size_bytes: int
    score: int
    mmproj_path: Path | None = None

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024 ** 3), 3)

    @property
    def total_size_bytes(self) -> int:
        total = self.size_bytes
        if self.mmproj_path:
            try:
                total += self.mmproj_path.stat().st_size
            except OSError:
                pass
        return total

    @property
    def total_size_gb(self) -> float:
        return round(self.total_size_bytes / (1024 ** 3), 3)

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "size_gb": self.size_gb,
            "total_size_bytes": self.total_size_bytes,
            "total_size_gb": self.total_size_gb,
            "score": self.score,
            "mmproj_path": str(self.mmproj_path) if self.mmproj_path else None,
        }


def find_gguf_models(
    roots: list[Path] | None = None,
    *,
    max_bytes: int = 7 * 1024 * 1024 * 1024,
    limit: int = 20,
) -> list[LocalModelCandidate]:
    candidates: list[LocalModelCandidate] = []
    for root in roots or DEFAULT_MODEL_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.gguf"):
            if path.name.lower().startswith("mmproj"):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > max_bytes:
                continue
            candidates.append(
                LocalModelCandidate(
                    path=path,
                    size_bytes=size,
                    score=score_model_path(path, size),
                    mmproj_path=find_mmproj_for(path),
                )
            )
    return sorted(candidates, key=lambda c: (c.score, -c.size_bytes), reverse=True)[:limit]


def best_gguf_model(
    roots: list[Path] | None = None,
    *,
    max_bytes: int = 7 * 1024 * 1024 * 1024,
) -> LocalModelCandidate | None:
    candidates = find_gguf_models(roots=roots, max_bytes=max_bytes, limit=1)
    return candidates[0] if candidates else None


def score_model_path(path: Path, size_bytes: int) -> int:
    name = path.name.lower()
    score = 0
    if "3b" in name:
        score += 100
    if "instruct" in name:
        score += 50
    if "qwen" in name:
        score += 30
    if "llama" in name:
        score += 20
    if "q4_k_m" in name:
        score += 40
    if "q8" in name:
        score += 20
    if "vl" in name:
        score += 10
    if "encoder" in name or "umt5" in name or "clip" in name:
        score -= 200
    if size_bytes < 800 * 1024 * 1024:
        score -= 50
    return score


def find_mmproj_for(model_path: Path) -> Path | None:
    parent = model_path.parent
    try:
        projectors = list(parent.glob("mmproj*.gguf"))
    except OSError:
        return None
    if not projectors:
        return None
    stem_terms = set(model_path.stem.lower().replace("_", "-").split("-"))
    ranked = sorted(
        projectors,
        key=lambda p: len(stem_terms & set(p.stem.lower().replace("_", "-").split("-"))),
        reverse=True,
    )
    return ranked[0]
