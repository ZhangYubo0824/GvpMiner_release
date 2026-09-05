#!/usr/bin/env python3
"""CLI entry point for Exp3 occurrence-level motif sequence representations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp3_motif_representation.representation import main


if __name__ == "__main__":
    main()
