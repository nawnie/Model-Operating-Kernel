from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

from mok.models.gguf import GGUFInspection, inspect_gguf_file


class ExpertState(str, Enum):
    OFFLINE = "offline"
    STAGED = "staged"
    RESIDENT = "resident"
    ACTIVE = "active"
    IDLE = "idle"



@dataclass
class VRAMProfile:
    """
    VRAM cost descriptor for a single expert.

    ``static_gb``        : from config — used when no measured data available.
    ``measured_peak_gb`` : observed peak during a real forward pass (from
                           telemetry).  Used by BudgetManager when set.
    ``activation_gb``    : peak delta during the forward pass itself (may
                           spike above resident cost).
    """
    static_gb: float
    measured_peak_gb: float | None = None
    activation_gb: float | None = None

    @property
    def effective_gb(self) -> float:
        """Return measured_peak_gb if available, else static_gb."""
        return self.measured_peak_gb if self.measured_peak_gb is not None else self.static_gb


@dataclass(slots=True)
class ExpertMetadata:
    name: str
    role: str
    kind: str
    backend: str
    api_url: str | None
    base_id: str | None
    adapter_path: str | None
    vram_cost_gb: float
    ram_cost_gb: float
    current_device: str
    state: ExpertState
    model_path: str | None = None
    file_format: str | None = None
    architecture: str | None = None
    quantization: str | None = None
    context_limit: int = 8192
    trust_score: float = 1.0
    load_sequence: int = 0
    vram_profile: VRAMProfile | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ExpertMetadata":
        return cls(
            name=str(data["name"]),
            role=str(data["role"]),
            kind=str(data["kind"]),
            backend=str(data["backend"]),
            api_url=data.get("api_url") if data.get("api_url") else None,
            base_id=data.get("base_id") if data.get("base_id") else None,
            adapter_path=data.get("adapter_path") if data.get("adapter_path") else None,
            vram_cost_gb=float(data["vram_cost_gb"]),
            ram_cost_gb=float(data["ram_cost_gb"]),
            current_device=str(data.get("current_device", "cpu")),
            state=ExpertState(str(data["state"])),
            model_path=data.get("model_path") if data.get("model_path") else None,
            file_format=data.get("file_format") if data.get("file_format") else None,
            architecture=data.get("architecture") if data.get("architecture") else None,
            quantization=data.get("quantization") if data.get("quantization") else None,
            context_limit=int(data.get("context_limit", 8192)),
            trust_score=float(data.get("trust_score", 1.0)),
        )

    @property
    def is_loaded(self) -> bool:
        return self.state in {
            ExpertState.STAGED,
            ExpertState.RESIDENT,
            ExpertState.ACTIVE,
            ExpertState.IDLE,
        }

    def hydrate_from_local_artifact(self) -> None:
        if self.file_format != "gguf" or not self.model_path:
            return
        path = Path(self.model_path)
        if not path.exists():
            return
        inspection = inspect_gguf_file(path)
        self._apply_gguf_inspection(inspection)

    def _apply_gguf_inspection(self, inspection: GGUFInspection) -> None:
        if not self.architecture and inspection.architecture:
            self.architecture = inspection.architecture
        if not self.quantization and inspection.quantization_label:
            self.quantization = inspection.quantization_label
        if self.context_limit == 8192 and inspection.context_length:
            self.context_limit = inspection.context_length

    def hydrate_from_ollama(self, ollama_models: list[dict[str, Any]]) -> None:
        """
        Enrich metadata from an /api/tags model entry if this expert matches.

        ollama_models: the list under the 'models' key returned by GET /api/tags.
        Each entry looks like:
          {"name": "qwen2.5:7b", "details": {"parameter_size": "7B", ...}, ...}
        """
        if self.backend != "ollama" or not self.base_id:
            return
        tag = self.base_id
        for entry in ollama_models:
            if entry.get("name") == tag:
                details = entry.get("details", {})
                if not self.architecture:
                    self.architecture = details.get("family") or details.get("architecture")
                if not self.quantization:
                    self.quantization = details.get("quantization_level")
                # Ollama doesn't expose context_length in /api/tags — leave as-is
                break


class ModelRegistry:
    def __init__(self, experts: list[ExpertMetadata]) -> None:
        self._experts = {expert.name: expert for expert in experts}
        for expert in self._experts.values():
            expert.hydrate_from_local_artifact()
        sequence = 0
        for expert in self._experts.values():
            if expert.is_loaded:
                sequence += 1
                expert.load_sequence = sequence
        self._sequence = sequence

    @classmethod
    def from_json(cls, path: Path) -> "ModelRegistry":
        payload = json.loads(path.read_text(encoding="utf-8"))
        experts = [ExpertMetadata.from_dict(item) for item in payload["experts"]]
        return cls(experts)

    def all(self) -> list[ExpertMetadata]:
        return list(self._experts.values())

    def get(self, name: str) -> ExpertMetadata:
        return self._experts[name]

    def find_first_by_role(self, role: str) -> ExpertMetadata | None:
        for expert in self._experts.values():
            if expert.role == role:
                return expert
        return None

    def promote(self, name: str, state: ExpertState) -> ExpertMetadata:
        expert = self.get(name)
        if not expert.is_loaded:
            self._sequence += 1
            expert.load_sequence = self._sequence
        expert.state = state
        expert.current_device = "cuda"
        return expert

    def mark_idle(self, name: str) -> ExpertMetadata:
        expert = self.get(name)
        if expert.role != "coordinator":
            expert.state = ExpertState.IDLE
        return expert

    def evict(self, name: str) -> ExpertMetadata:
        expert = self.get(name)
        expert.state = ExpertState.OFFLINE
        expert.current_device = "cpu"
        return expert

    def hydrate_from_ollama_server(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 5.0,
    ) -> int:
        """
        Query the running Ollama server at base_url for its model list and
        enrich any matching expert metadata entries.

        Returns the number of experts that were updated.
        Never raises -- logs silently on connection errors.
        """
        import json
        from urllib import request as urlrequest
        from urllib.error import URLError, HTTPError

        try:
            req = urlrequest.Request(base_url.rstrip("/") + "/api/tags", method="GET")
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                data: dict = json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, OSError, json.JSONDecodeError):
            return 0

        models: list[dict] = data.get("models", [])
        updated = 0
        for expert in self._experts.values():
            before = (expert.architecture, expert.quantization)
            expert.hydrate_from_ollama(models)
            if (expert.architecture, expert.quantization) != before:
                updated += 1
        return updated
