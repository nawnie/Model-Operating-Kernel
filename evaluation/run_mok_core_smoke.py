#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mok.evaluation.mok_core_smoke import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
