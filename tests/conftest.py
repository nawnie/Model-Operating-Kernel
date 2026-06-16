"""Shared pytest fixtures for the MoK test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

# Resolve relative to this file so tests work on any OS / any checkout location
_REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = _REPO_ROOT / "configs" / "example_experts.json"


@pytest.fixture
def config_path() -> Path:
    """Return the path to the canonical example_experts.json config."""
    return CONFIG_PATH


@pytest.fixture
def experts_config(tmp_path: Path) -> Path:
    """Write a minimal in-memory expert config and return its path.
    
    Use this when you want a fully isolated config without touching disk state.
    """
    cfg = tmp_path / "experts.json"
    cfg.write_text(
        CONFIG_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return cfg
