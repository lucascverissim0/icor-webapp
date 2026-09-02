"""Acquire the official EEA 2025 provisional model aggregate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from icor.evidence.eea_provisional_acquisition import acquire_2025_provisional


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    result = acquire_2025_provisional(args.destination)
    print(json.dumps(asdict(result), default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
