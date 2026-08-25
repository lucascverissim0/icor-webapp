#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from icor.audit import audit_repository

DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit ICOR source data without modifying it.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = audit_repository(args.root).to_dict()
        rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            args.output.write_text(rendered, encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Audit failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
