#!/usr/bin/env python3
"""Entrypoint mỏng cho `knee_mri.cli.similarity`.

Chạy được cả hai cách::

    python scripts/similarity.py --env local --limit 8
    knee-similarity --env local --limit 8    # sau khi pip install -e .
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee_mri.cli import similarity  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(similarity())
