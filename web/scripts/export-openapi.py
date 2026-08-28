"""Export the FastAPI schema deterministically for the TypeScript client."""

from __future__ import annotations

import json
from pathlib import Path

from icor.api.app import create_app

WEB_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = WEB_ROOT / "openapi.json"
    serialized = json.dumps(
        create_app(snapshot_root=WEB_ROOT / ".openapi-no-snapshot").openapi(),
        indent=2,
        sort_keys=True,
    )
    output.write_text(f"{serialized}\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
