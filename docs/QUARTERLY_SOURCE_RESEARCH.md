# Quarterly official-source research

This workflow uses the OpenAI Responses API and web search to discover candidate
official vehicle-registration and fleet releases. It does not ingest data, alter an
active snapshot, or approve a source. Every JSON report is marked for human review.

The GitHub Actions schedule runs at 06:17 UTC on January, April, July, and October 1.
It defaults to the explicitly requested `o4-mini` model. OpenAI now describes that
model as deprecated and succeeded by GPT-5 mini, so a future owner may set the
repository variable `OPENAI_RESEARCH_MODEL` without changing code if access ends.

## Client setup

1. Create a dedicated OpenAI API project for this quarterly task and set its monthly
   project budget/notification controls. A platform budget is not a guaranteed
   per-request cutoff, so use a project that is not shared with other workloads.
2. Add its key to the GitHub repository Actions secret `OPENAI_API_KEY`. Never put the
   key in source, an issue, a command argument, or a report.
3. Run the `Quarterly official-source research` workflow manually once and inspect the
   uploaded JSON artifact. The scheduled cadence then needs no local computer.

Preview the exact bounded configuration without a key or API call:

    uv run python scripts/run_quarterly_source_research.py --dry-run

The job covers 33 governed targets: the EU-level EEA release, all 27 EU member states,
the UK, and all four EFTA states. Four regional response batches ensure that every
authority receives explicit search attention. Each batch allows at most 24 web-search
calls and 12,000 output tokens; response storage is disabled and URLs outside each
authority's official domains are rejected. At prices checked on 2026-09-12, the
conservative ceiling for the complete four-batch run is $2.0512, well below the $20
run budget. A metered check also stops the workflow after any batch that reaches the
budget. This is an engineering estimate, not a billing guarantee: OpenAI pricing and
account behavior can change, so reconcile each report with the project's usage
dashboard.

## Review boundary

Before any candidate can enter the evidence ledger, a human must verify publication
terms, download URL, checksum, measure, period, geography, revision status, model
granularity, and overlap with existing releases. Registration year, sales year, and
vehicle model year remain distinct. Candidate reports are discovery aids, not sales
truth and not training data.
