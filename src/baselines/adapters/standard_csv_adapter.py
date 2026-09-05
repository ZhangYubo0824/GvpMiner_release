#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from adapter_common import build_parser, copy_standard_csvs


def adapt(*, family: str, method: str, input_dir: Path, output_dir: Path) -> None:
    copy_standard_csvs(family=family, method=method, input_dir=input_dir, output_dir=output_dir)


def main() -> None:
    args = build_parser("Validate and copy model-exported standard baseline motif CSV outputs.").parse_args()
    adapt(family=args.family, method=args.method, input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
