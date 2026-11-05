#!/usr/bin/env python3
"""Server-portable entry point for stability-filtered motif discovery."""

from motif_stability.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
