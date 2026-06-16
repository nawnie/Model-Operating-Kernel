from __future__ import annotations

import re
from pathlib import Path

from mok.companion.storage import Datacard


CODE_EXTENSIONS = {".py", ".ps1", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".cs"}
DOC_EXTENSIONS = {".md", ".txt", ".rst"}
DATA_EXTENSIONS = {".json", ".jsonl", ".csv", ".toml", ".yaml", ".yml"}


def card_from_file(path: Path, content: str) -> Datacard:
    lane = lane_for_file(path, content)
    summary = summarize_text(content)
    tags = tags_for_file(path, content, lane)
    return Datacard.build(
        lane=lane,
        source_type="file",
        source_uri=str(path),
        title=path.name,
        content=content,
        summary=summary,
        tags=tags,
        trust_score=0.75,
        trainable=is_training_candidate(lane, content),
    )


def card_from_window_title(title: str) -> Datacard:
    clean = " ".join(title.split())
    lane = lane_for_window_title(clean)
    return Datacard.build(
        lane=lane,
        source_type="window",
        source_uri=f"window:{clean}",
        title=clean[:120],
        content=clean,
        summary=clean[:240],
        tags=["window", lane],
        trust_score=0.35,
        trainable=False,
    )


def lane_for_file(path: Path, content: str) -> str:
    suffix = path.suffix.lower()
    text = content.lower()
    if suffix in CODE_EXTENSIONS:
        return "code"
    if suffix in DATA_EXTENSIONS:
        return "data"
    if "roadmap" in path.name.lower() or "plan" in path.name.lower():
        return "planning"
    if "research" in path.name.lower() or re.search(r"\b(arxiv|paper|benchmark|eval)\b", text):
        return "research"
    if suffix in DOC_EXTENSIONS:
        return "docs"
    return "general"


def lane_for_window_title(title: str) -> str:
    text = title.lower()
    if any(term in text for term in ["visual studio", "vscode", "pycharm", ".py", "powershell"]):
        return "workstation"
    if any(term in text for term in ["browser", "chrome", "edge", "firefox"]):
        return "browser"
    if any(term in text for term in ["codex", "chatgpt", "claude", "gemini"]):
        return "ai-work"
    return "activity"


def summarize_text(content: str, *, max_chars: int = 700) -> str:
    lines = [line.strip() for line in content.replace("\r\n", "\n").split("\n")]
    meaningful = [line for line in lines if line and not line.startswith("#" * 6)]
    if not meaningful:
        return content[:max_chars]
    summary = " ".join(meaningful[:8])
    return summary[:max_chars]


def tags_for_file(path: Path, content: str, lane: str) -> list[str]:
    tags = {lane, path.suffix.lower().lstrip(".") or "file"}
    lowered = content.lower()
    for term in ["mok", "gguf", "router", "training", "memory", "atlas", "cartographer"]:
        if term in lowered or term in path.name.lower():
            tags.add(term)
    return sorted(tags)


def is_training_candidate(lane: str, content: str) -> bool:
    if lane not in {"docs", "planning", "research", "code"}:
        return False
    if len(content.strip()) < 200:
        return False
    unstable_markers = ["todo maybe", "scratch", "temporary", "delete this"]
    text = content.lower()
    return not any(marker in text for marker in unstable_markers)
