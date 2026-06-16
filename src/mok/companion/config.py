from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


DEFAULT_COMPANION_DIR = Path.home() / ".mok" / "companion"
DEFAULT_CONFIG_PATH = DEFAULT_COMPANION_DIR / "config.json"
DEFAULT_DB_PATH = DEFAULT_COMPANION_DIR / "memory.sqlite3"
DEFAULT_LLAMA_CPP_URL = "http://localhost:8080/v1/chat/completions"


@dataclass(slots=True)
class CompanionConfig:
    storage_dir: Path = DEFAULT_COMPANION_DIR
    allow_paths: list[Path] = field(default_factory=list)
    deny_dir_names: list[str] = field(
        default_factory=lambda: [
            ".git",
            ".hg",
            ".svn",
            ".venv",
            "venv",
            "env",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "traces",
        ]
    )
    deny_window_keywords: list[str] = field(
        default_factory=lambda: [
            "password",
            "passcode",
            "credential",
            "secret",
            "token",
            "bank",
            "wallet",
        ]
    )
    file_extensions: list[str] = field(
        default_factory=lambda: [
            ".txt",
            ".md",
            ".json",
            ".jsonl",
            ".py",
            ".ps1",
            ".toml",
            ".yaml",
            ".yml",
            ".csv",
            ".log",
        ]
    )
    max_file_bytes: int = 512_000
    poll_seconds: float = 5.0
    capture_window_titles: bool = True
    max_cards_per_scan: int = 500
    inference_vram_gb: float = 2.0
    inference_ram_mb: int = 1536
    reservation_headroom: float = 0.20
    reserved_vram_gb: float = 2.4
    reserved_ram_mb: int = 1844
    max_model_bytes: int = 7 * 1024 * 1024 * 1024
    llama_cpp_url: str = DEFAULT_LLAMA_CPP_URL
    llama_cpp_server_path: Path | None = None
    model_path: Path | None = None
    mmproj_path: Path | None = None
    model_name: str = "mok-companion"
    context_limit: int = 8192
    gpu_layers: int = 999
    chat_temperature: float = 0.35
    chat_max_tokens: int = 512

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "memory.sqlite3"

    @property
    def computed_reserved_vram_gb(self) -> float:
        return round(self.inference_vram_gb * (1.0 + self.reservation_headroom), 3)

    @property
    def computed_reserved_ram_mb(self) -> int:
        return int(round(self.inference_ram_mb * (1.0 + self.reservation_headroom)))

    @classmethod
    def default(cls) -> "CompanionConfig":
        return cls()

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "CompanionConfig":
        if not path.exists():
            return cls.default()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "CompanionConfig":
        data = dict(raw)
        data["storage_dir"] = Path(data.get("storage_dir", DEFAULT_COMPANION_DIR))
        data["allow_paths"] = [Path(p) for p in data.get("allow_paths", [])]
        if data.get("llama_cpp_server_path"):
            data["llama_cpp_server_path"] = Path(data["llama_cpp_server_path"])
        if data.get("model_path"):
            data["model_path"] = Path(data["model_path"])
        if data.get("mmproj_path"):
            data["mmproj_path"] = Path(data["mmproj_path"])
        return cls(**data)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["storage_dir"] = str(self.storage_dir)
        data["allow_paths"] = [str(p) for p in self.allow_paths]
        data["llama_cpp_server_path"] = str(self.llama_cpp_server_path) if self.llama_cpp_server_path else None
        data["model_path"] = str(self.model_path) if self.model_path else None
        data["mmproj_path"] = str(self.mmproj_path) if self.mmproj_path else None
        return data

    def save(self, path: Path = DEFAULT_CONFIG_PATH) -> Path:
        self.reserved_vram_gb = self.computed_reserved_vram_gb
        self.reserved_ram_mb = self.computed_reserved_ram_mb
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def with_allow_paths(self, paths: list[Path]) -> "CompanionConfig":
        seen: set[str] = set()
        merged: list[Path] = []
        for path in [*self.allow_paths, *paths]:
            resolved = path.expanduser().resolve()
            key = str(resolved).lower()
            if key not in seen:
                seen.add(key)
                merged.append(resolved)
        data = self.to_dict()
        data["allow_paths"] = [str(p) for p in merged]
        return CompanionConfig.from_dict(data)
