#!/usr/bin/env python3
"""CLI entry point for the standalone Exp5 external enrichment workflow."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from package.cli import main


if __name__ == "__main__":
    main()
