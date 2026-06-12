import pytest

from atlas_load_cartographer.gateway.preload_manager import PreloadManager


@pytest.mark.asyncio
async def test_preload_manager_stages_existing_file(tmp_path) -> None:
    adapter_file = tmp_path / "adapter.local"
    adapter_file.write_bytes(b"mok-test")

    manager = PreloadManager({"sample": adapter_file})

    assert await manager.stage_expert_to_ram("sample") is True
    assert "sample" in manager.staged_experts


@pytest.mark.asyncio
async def test_preload_manager_rejects_missing_file(tmp_path) -> None:
    manager = PreloadManager({"missing": tmp_path / "missing.local"})

    assert await manager.stage_expert_to_ram("missing") is False
