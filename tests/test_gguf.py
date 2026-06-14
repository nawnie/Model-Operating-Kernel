from __future__ import annotations

from pathlib import Path
import struct

from mok.models.gguf import inspect_gguf_file, scan_gguf_directory
from mok.models.registry import ModelRegistry


def _write_gguf_string(handle, value: str) -> None:
    encoded = value.encode("utf-8")
    handle.write(struct.pack("<Q", len(encoded)))
    handle.write(encoded)


def _write_metadata_entry(handle, key: str, value_type: int, value) -> None:
    _write_gguf_string(handle, key)
    handle.write(struct.pack("<I", value_type))
    if value_type == 4:
        handle.write(struct.pack("<I", value))
    elif value_type == 8:
        _write_gguf_string(handle, value)
    else:
        raise AssertionError(f"unsupported test metadata type {value_type}")


def write_fake_gguf(path: Path, *, architecture: str = "llama", context_length: int = 4096) -> None:
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<I", 3))
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<Q", 4))

        _write_metadata_entry(handle, "general.architecture", 8, architecture)
        _write_metadata_entry(handle, "general.name", 8, "Tiny GGUF")
        _write_metadata_entry(handle, "general.file_type", 4, 12)
        _write_metadata_entry(handle, f"{architecture}.context_length", 4, context_length)

        _write_gguf_string(handle, "blk.0.attn.weight")
        handle.write(struct.pack("<I", 2))
        handle.write(struct.pack("<Q", 16))
        handle.write(struct.pack("<Q", 16))
        handle.write(struct.pack("<I", 12))
        handle.write(struct.pack("<Q", 0))

        current = handle.tell()
        padding = (32 - (current % 32)) % 32
        handle.write(b"\x00" * padding)
        handle.write(b"\x00" * 32)


def test_inspect_gguf_file_reads_summary(tmp_path: Path) -> None:
    gguf_path = tmp_path / "tiny.gguf"
    write_fake_gguf(gguf_path)

    inspection = inspect_gguf_file(gguf_path)

    assert inspection.version == 3
    assert inspection.architecture == "llama"
    assert inspection.context_length == 4096
    assert inspection.quantization_label == "Q4_K"
    assert inspection.tensor_count == 1
    assert inspection.tensors[0].name == "blk.0.attn.weight"


def test_scan_gguf_directory_finds_models(tmp_path: Path) -> None:
    first = tmp_path / "a.gguf"
    second_dir = tmp_path / "nested"
    second_dir.mkdir()
    second = second_dir / "b.gguf"
    write_fake_gguf(first)
    write_fake_gguf(second, architecture="qwen2", context_length=32768)

    inspections = scan_gguf_directory(tmp_path)

    assert [item.architecture for item in inspections] == ["llama", "qwen2"]


def test_registry_hydrates_gguf_backed_expert(tmp_path: Path) -> None:
    gguf_path = tmp_path / "local.gguf"
    write_fake_gguf(gguf_path, architecture="qwen2", context_length=32768)
    config_path = tmp_path / "experts.json"
    config_path.write_text(
        """
{
  "experts": [
    {
      "name": "local-gguf",
      "role": "general",
      "kind": "full",
      "backend": "local",
      "api_url": null,
      "base_id": null,
      "adapter_path": null,
      "vram_cost_gb": 4.0,
      "ram_cost_gb": 6.0,
      "current_device": "cpu",
      "state": "offline",
      "model_path": "__MODEL_PATH__",
      "file_format": "gguf"
    }
  ]
}
""".replace("__MODEL_PATH__", str(gguf_path).replace("\\", "\\\\")),
        encoding="utf-8",
    )

    registry = ModelRegistry.from_json(config_path)
    expert = registry.get("local-gguf")

    assert expert.architecture == "qwen2"
    assert expert.quantization == "Q4_K"
    assert expert.context_limit == 32768
