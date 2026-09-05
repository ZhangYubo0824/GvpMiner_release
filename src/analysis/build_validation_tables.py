#!/usr/bin/env python3

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motif_validation_tables import validation_cli


if __name__ == "__main__":
    raise SystemExit(validation_cli())
