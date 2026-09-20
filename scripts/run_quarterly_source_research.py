#!/usr/bin/env python3
"""Run or preview the bounded quarterly official-source discovery task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI, OpenAIError

from icor.research.quarterly import (
    BATCHES,
    BUDGET_USD,
    MAX_OUTPUT_TOKENS_PER_BATCH,
    MAX_TOOL_CALLS_PER_BATCH,
    MODEL,
    TARGETS,
    ResearchError,
    conservative_run_ceiling_usd,
    require_api_key,
    run_quarterly_research,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="validate configuration without API use"
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path, default=Path(".local/quarterly-research"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "targets": [target.key for target in TARGETS],
                    "batches": list(BATCHES),
                    "max_tool_calls_per_batch": MAX_TOOL_CALLS_PER_BATCH,
                    "max_output_tokens_per_batch": MAX_OUTPUT_TOKENS_PER_BATCH,
                    "budget_usd": str(BUDGET_USD),
                    "conservative_run_ceiling_usd": str(conservative_run_ceiling_usd()),
                    "api_called": False,
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        client = OpenAI(api_key=require_api_key())
        destination = run_quarterly_research(client, args.output, model=args.model)
    except ResearchError as error:
        print(f"quarterly research failed: {error}", file=sys.stderr)
        return 2
    except OpenAIError as error:
        print(
            f"quarterly research failed at OpenAI ({type(error).__name__}); "
            "verify model access, the dedicated project key, and account limits",
            file=sys.stderr,
        )
        return 3
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
