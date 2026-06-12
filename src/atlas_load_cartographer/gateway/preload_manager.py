"""Async preload lifecycle manager for Model Operating Kernel adapters.

The preload manager performs non-blocking filesystem-to-RAM staging for likely
LoRA adapter routes. It intentionally does not load adapters into VRAM. Its job
is to touch the adapter payload early enough that the operating system can place
it in the page cache before the execution backend asks for it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger("MOK.PreloadManager")


class PreloadManager:
    """Stage likely expert adapter files into the operating-system page cache."""

    def __init__(self, expert_mapping: Mapping[str, str | Path], block_size_mb: int = 4) -> None:
        self.expert_mapping = {route: Path(path) for route, path in expert_mapping.items()}
        self.staged_experts: set[str] = set()
        self.block_size = block_size_mb * 1024 * 1024
        self.lock = asyncio.Lock()

    async def stage_expert_to_ram(self, route: str) -> bool:
        """Warm a route's adapter file into system memory.

        Returns True when the route is already staged or staging succeeds.
        Returns False when the route is unknown, missing, or fails to read.
        """

        adapter_path = self.expert_mapping.get(route)
        if adapter_path is None:
            logger.debug("Unknown expert route requested for staging: %s", route)
            return False

        async with self.lock:
            if route in self.staged_experts:
                return True

            if not adapter_path.exists():
                logger.error("Kernel fault: adapter resource missing at %s", adapter_path)
                return False

            if not adapter_path.is_file():
                logger.error("Kernel fault: adapter path is not a file: %s", adapter_path)
                return False

            try:
                logger.info("[KERNEL] Page-warming expert file: %s -> %s", route, adapter_path)
                await asyncio.to_thread(self._force_os_page_warm, adapter_path)
                self.staged_experts.add(route)
                return True
            except OSError as exc:
                logger.error("Kernel paging exception for %s: %s", route, exc)
                return False

    def _force_os_page_warm(self, file_path: Path) -> None:
        """Sequential read burst to encourage kernel readahead/page caching."""

        with file_path.open("rb") as file_handle:
            while file_handle.read(self.block_size):
                # Intentionally discard bytes: the OS page cache is the target.
                pass

    async def evict_ram_stage(self, route: str) -> None:
        """Remove internal staged tracking for a route.

        This does not force the operating system to evict pages from RAM. It only
        clears MOK's management state so a later route can be staged again.
        """

        async with self.lock:
            if route in self.staged_experts:
                self.staged_experts.remove(route)
                logger.info("[KERNEL] Evicted expert %s tracking state.", route)
