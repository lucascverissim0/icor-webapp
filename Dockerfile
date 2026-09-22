# syntax=docker/dockerfile:1.7
#
# The authenticated client preview.
#
# Two properties this file exists to guarantee, both by construction rather than
# by convention:
#
#   1. There is no `COPY . .` anywhere. Every file is named. The repository's Git
#      history still contains a leaked OpenAI key, so the history must never be
#      able to reach an image, and an allowlist is the only way to be sure.
#   2. The runtime image installs the `preview` extra only, so the `openai` and
#      `streamlit` packages are physically absent from the running container.
#      That does not un-leak the key, but it removes the client from anything
#      that could use it.

# ---------------------------------------------------------------- client bundle
FROM node:24.15.0-bookworm-slim AS web
WORKDIR /build
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/tsconfig.json web/vite.config.ts web/index.html ./
COPY web/src ./src
# Release gate 3: the bundle must be built in client-release mode.
ENV VITE_ICOR_CLIENT_RELEASE=verified
RUN npm run build -- --outDir /client-release --emptyOutDir \
 && test -f /client-release/index.html \
 && test -d /client-release/assets

# ------------------------------------------------------------ python dependencies
FROM python:3.12-slim-bookworm AS deps
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock .python-version ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra preview

# ------------------------------------------------------------------- runtime
FROM python:3.12-slim-bookworm
RUN useradd --system --uid 10001 --home /app --shell /usr/sbin/nologin icor
WORKDIR /app
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY --from=deps                  /app/.venv                        /app/.venv
COPY --chown=icor:icor            src                               /app/src
COPY --chown=icor:icor            scripts/run_container_preview.py  /app/scripts/
# `data/` cannot be excluded wholesale: the API reads this catalogue at start-up
# with no fallback.
COPY --chown=icor:icor            data/icor_supported_models.txt    /app/data/
COPY --from=web --chown=icor:icor /client-release                   /app/client-release
# The pruned, client-scoped snapshot. Baked rather than mounted: it is immutable,
# read-only at runtime, and pinning it to the image digest means code and data
# roll back together.
COPY --chown=icor:icor            .local/client-evidence            /srv/icor/evidence

# The fail-closed runner requires a coverage database to exist. Create an empty
# one at build time rather than teaching the runner a container exception.
RUN install -d -o icor -g icor /var/lib/icor \
 && PYTHONPATH=/app/src /app/.venv/bin/python -c "\
from pathlib import Path; \
from icor.infrastructure.sqlite_coverage_repository import SQLiteCoverageRepository; \
SQLiteCoverageRepository(Path('/var/lib/icor/production-coverage.sqlite3'))" \
 && chown icor:icor /var/lib/icor/production-coverage.sqlite3

# Fail the build, not the deploy, if the baked snapshot is not a verifiable
# client-scoped snapshot. A bad bake becomes a red build instead of a machine
# that crash-loops in production.
RUN PYTHONPATH=/app/src ICOR_EVIDENCE_ACTIVE_ROOT=/srv/icor/evidence \
    /app/.venv/bin/python -c "\
from pathlib import Path; \
from icor.infrastructure.snapshot_store import SnapshotStore; \
manifest, _ = SnapshotStore(Path('/srv/icor/evidence')).open_active_snapshot(); \
assert manifest.scope == 'client-release', manifest.scope; \
print('baked', manifest.snapshot_id)"

ENV ICOR_PREVIEW_HOST_MODE=container \
    ICOR_CLIENT_RELEASE_MODE=verified \
    ICOR_PREVIEW_ASSET_ROOT=/app/client-release \
    ICOR_EVIDENCE_ACTIVE_ROOT=/srv/icor/evidence \
    ICOR_COVERAGE_DB=/var/lib/icor/production-coverage.sqlite3 \
    ICOR_PREVIEW_PORT=8080 \
    SQLITE_TMPDIR=/tmp

USER icor
EXPOSE 8080
ENTRYPOINT ["python", "/app/scripts/run_container_preview.py"]
