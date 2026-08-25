# ICOR Web App — Durable Handoff

Last updated: 2026-08-25

## Project objective

Repair and evolve the ICOR web app into a reliable, secure, maintainable, and exceptionally user-friendly product. Development is local-first so Lucas can review progress incrementally. The existing multi-user web deployment is out of scope for modification or deployment until Lucas explicitly authorizes it.

On 2026-08-25 Lucas clarified the business outcome: the app must forecast how many vehicles will require windshield replacement, broken down by brand, model, year, and the exact windshield design. A model/year may have multiple incompatible windshields because of generation, body/design/facelift, trim, equipment, or other configuration differences. The intended customer is a windshield manufacturer using the forecast for product and demand planning. The output therefore ultimately needs to resolve demand to a canonical windshield-compatible vehicle configuration, and ideally to the manufacturer's windshield SKU/part family, rather than stopping at a model or generation label.

## Repository and access

- GitHub owner: `lucascverissim0`
- Repository: `lucascverissim0/icor-webapp` (public)
- Local clone: `C:\Users\LucasCravoVERISSIMO\icor-webapp`
- Remote: `https://github.com/lucascverissim0/icor-webapp.git`
- Branch: `main`
- Review baseline HEAD: `1ba1d7c`
- At the end of the initial review, the tracked worktree was clean.
- Existing Windows Git Credential Manager credentials identified the account as `lucascverissim0` without exposing the token.
- All future Git network commands must be non-interactive. Use `GCM_INTERACTIVE=never`; if authentication expires, stop rather than triggering an account-selection pop-up.
- Clearing a Codex conversation or terminal screen does not normally erase Windows Credential Manager credentials. Credentials can still expire or be revoked externally.

## User decisions and working rules

- Preserve advancement across cleared Codex conversations in this file.
- Every final response must state whether the conversation context is safe to clear, as required by `AGENTS.md`.
- Never store secrets or private/customer data here.
- Run and demonstrate a local app as work progresses.
- Do not modify or deploy the current multi-user web version yet.
- The initial repository review was explicitly read-only. No application code was changed during it.
- On 2026-08-25 Lucas authorized starting the approved delivery sequence and requested brief completion reports covering what changed, what happened, and the next steps.
- The company has a proprietary vehicle-to-windshield fitment catalog and one year of reliably tracked replacement-related history. Integrate it in a later subproject as fitment truth and limited calibration/validation evidence; do not treat one year as sufficient long-run training history.
- Lucas requires all development to remain on a separate long-lived development branch/worktree until the final product has been fully reviewed and he explicitly authorizes a merge. The current actively used application on `main` must remain intact. Do not merge, push, deploy, or modify production from the development worktree without explicit authorization.

## Application map

- `ui/app.py`: landing page, custom authentication, dataset browsing, and Script 1 trigger.
- `ui/pages/02_Model_Researcher.py`: authenticated Model Researcher UI, Script 2 execution, output selection/download, and run logs.
- `scripts/script1.py`: builds the shared passenger-car workbook, calculates recommendation scores, and queries OpenAI for BEV counterparts.
- `scripts/script2.py`: researches a chosen model/generation, forecasts continuation, calls OpenAI, and produces Excel output.
- `scripts/wikipedia_gen.py`: obtains and caches model-generation windows from Wikipedia-derived research.
- `scripts/build_local_generations_db.py`: builds the local generation database.
- `data/`: Top-100 market JSON data, ICOR supported-model mapping, generated workbook, and assets.
- `.streamlit/config.toml`, `.devcontainer/devcontainer.json`, `requirements.txt`, and `DEPLOYMENT.md`: runtime and deployment configuration.

## Critical security action still required

A real OpenAI API key was committed to public Git history in `scripts/script1.py`, introduced by commit `cbef0ed6...` and removed by `d557aef...` on 2025-08-19. It must be treated as compromised even though it is absent from current HEAD.

Required owner actions:

1. Revoke the exposed key in the OpenAI dashboard.
2. Create a replacement key and update the deployment secret.
3. Confirm revocation/rotation in a future session without placing either key in chat or the repository.
4. Plan a coordinated Git history rewrite later. Do not rewrite history without explicit authorization.

Never reproduce the exposed value in output, documentation, tests, commands, or commits.

## Confirmed review findings

### Critical/high priority

1. **ICOR support scoring is nonfunctional.** `scripts/script1.py:59-67` parses a tab-delimited list, but `data/icor_supported_models.txt` is a custom dictionary-like mapping. Across the current datasets, the parsed supported set intersects with zero normalized model names. At `script1.py:380-384`, all models therefore become unsupported and receive the same ICOR score of 10.

2. **All authenticated users have the same privileges.** `ui/app.py:120-177` creates authenticated sessions but no role or administrator boundary. Every authenticated user can trigger Script 1 at `ui/app.py:263-279`, spending the shared OpenAI key and rebuilding shared canonical data.

3. **Shared mutable files create races and cross-user leakage.** Script 1 writes directly to `data/passenger_car_data.xlsx` at `script1.py:348` without locking or atomic replacement. Model Researcher writes shared output files and `load_latest_output` at `ui/pages/02_Model_Researcher.py:190-195` can serve the globally latest result. Concurrent users can overwrite, read, or receive another user's result.

4. **Market data duplicates produce contradictory results.** Examples include repeated EU 2021 models and Toyota Aygo X twice in EU 2025. Script 1 groups and sums duplicates (`script1.py:301-305`), while Script 2 overwrites earlier entries (`script2.py:179-224`). Verified example: Toyota Aygo X becomes 172,000 units in Script 1 but 70,000 in Script 2. World 2016 contains 50 records with `units_sold: null`; Script 1 converts these to zero but counts appearances, while Script 2 skips them.

5. **Transient OpenAI failures poison the BEV cache.** Missing credentials or any API exception returns `False` at `script1.py:69-90`, and `has_bev_counterpart_cached` persists the false result at `script1.py:92-110`. An outage can therefore become a durable factual “no BEV” result and distort scores. Cache writes are not atomic and concurrent corruption is silently reset to an empty mapping.

6. **Output filenames are unsafe.** `script2.py:1315-1318` replaces spaces only. Natural data values such as `Nissan X-Trail / Rogue` retain `/`, producing unintended nested paths or save failures. Use a strict filename slug, unique per-run/per-user directories, and a resolved-path containment check.

7. **The custom authentication fallback is weak.** `ui/app.py:123-136` supports plaintext secret passwords and has no rate limiting, backoff, or lockout. Require hashed credentials and robust throttling or an external identity provider before production hardening.

8. **The optional `streamlit-authenticator` branch is broken for the declared dependency range.** At `ui/pages/02_Model_Researcher.py:93-103`, `authenticator.login("Login", "main")` uses the current API incorrectly and attempts to unpack a rendered call that returns `None`. The documented `[users]` deployment currently follows the custom fallback instead, so this is a latent configuration failure rather than a confirmed login bypass.

### Correctness and reliability

9. **NaN aggregation can silently damage recommendations.** `script1.py:438` uses `max(a or 0, b or 0)`. Because `NaN` is truthy, a missing regional value can win over a valid value and propagate NaN into rankings. Directly reproduced with Python: `max(float('nan') or 0, 5 or 0)` returns NaN.

10. **Forecasting uses a stale reference year.** `script2.py:523` and callers hard-code 2025. The current project date is 2026, and the logic will become increasingly stale. The scoring deadline year 2035 and selected year 2030 are also hard-coded policy inputs that need central configuration and tests.

11. **Generation-history filtering is overbroad.** At `script2.py:1078`, `or allow_year_match_if_model` causes every local generation within the window to enter history when the main caller passes `True` at `script2.py:1273-1276`, contradicting the stated “THIS GENERATION ONLY” behavior.

12. **OpenAI output is parsed without a strict schema.** `script2.py:807-975` requests legacy JSON-object mode and trusts arbitrary fields/types after parsing. Missing or malformed fields can raise runtime errors. Prefer strict JSON Schema/Structured Outputs, validate all fields, add bounded retry/timeout behavior, and pin the intended model/version where feasible.

13. **Workbook sheet-name caching can remain stale after a rebuild.** `_excel_sheet_names` at `ui/app.py:229-238` uses `st.cache_data` with a fixed path, no TTL/mtime argument, and no cache clear after Script 1.

14. **Generated Excel content may allow formula injection.** User/model/AI-derived strings are passed to openpyxl at `script2.py:924-974`. Text beginning with formula control characters should be escaped before workbook creation.

15. **Wikipedia generation cache freshness is indefinite.** `wikipedia_gen.py:330-347` reuses cached windows without TTL once it has enough entries. Its default cache path is `cache/gen_windows`, while the tracked placeholder is under `ui/cache/gen_windows`, creating misleading configuration.

16. **Backend logs are visible by default.** `ui/pages/02_Model_Researcher.py:34` defaults `debug.show_run_log` to `True` despite a comment saying it is off. The last 80 subprocess lines can be shown to every authenticated user at lines 372-375.

17. **Output files have no cleanup, quota, or lifecycle.** Long-running shared subprocesses (up to 420 seconds) and accumulated estimator outputs can exhaust limited hosted resources.

18. **PostHog events include usernames and vehicle-query details.** The project should document its analytics purpose, retention, legal basis/consent, and processor configuration before serving EU clients.

### Dependencies, configuration, and maintainability

19. **Known vulnerable pins/configuration exist.** `requests==2.32.3` is affected by CVE-2024-47081, fixed in 2.32.4. `streamlit==1.37.1` has a Windows-specific unauthenticated SSRF/NTLM exposure fixed in 1.54.0; the Linux Streamlit Community Cloud deployment is not affected by that Windows-specific condition, but exposed local Windows runs are.

20. **The devcontainer weakens security and reproducibility.** `.devcontainer/devcontainer.json:20-22` upgrades system packages non-reproducibly, installs requirements, then installs unpinned Streamlit again (overriding the declared pin), and disables CORS/XSRF. Streamlit's secure defaults should remain enabled outside a narrowly isolated development need.

21. **Runtime targets conflict.** The devcontainer uses Python 3.11, `DEPLOYMENT.md` says Python 3.12, and the inspected local machine provides Python 3.14.3. The locally installed packages do not represent `requirements.txt` and several application dependencies are absent.

22. **Dependency resolution is not reproducible.** `openai>=1.10.0` and `streamlit-authenticator>=0.4` are unbounded, with no lockfile or hashes. Model alias `gpt-5` is also unpinned, so behavior can drift.

23. **There are no automated tests or CI quality gates.** No test suite, CI workflow, linter configuration, type checker, dependency audit, or lockfile exists. `scripts/script2.py` is about 1,300 lines and combines data access, forecasting, API calls, filesystem behavior, and workbook generation.

24. **Script 1 runs its full pipeline at import time.** It performs network and filesystem work at module scope, making safe unit testing and reuse difficult.

25. **Documentation is insufficient.** `README.md` contains only the project title. Architecture, local setup, data provenance, scoring rules, API costs, privacy, testing, and recovery procedures are undocumented.

26. **Broad exception handling hides failures.** Multiple `except Exception`/silent fallbacks can conceal corrupt data, bad configuration, or partial failures while the UI reports success.

### Product-model gaps confirmed after the objective clarification

27. **The current entity key is too coarse for the business decision.** The main forecast is keyed only by `Model` and `Generation`; the researcher accepts free-text model, optional generation, and start year. There is no canonical vehicle-configuration or windshield-design identity covering market, body style, facelift/build interval, trim/equipment, ADAS/camera/HUD/sensor/acoustic/heated features, left/right-hand drive, VIN applicability, glazing/OEM reference, or windshield SKU. The heuristic body-style/performance classifier in Script 2 estimates sales shares; it does not establish windshield compatibility.

28. **The replacement model is an uncalibrated constant-rate heuristic.** Script 1 multiplies estimated fleet by `REPL_RATE_MEAN = 0.021`; Script 2 multiplies it by the same `REPAIR_RATE = 0.021`. The repository contains no historical breakage, claim, replacement, repair-versus-replace, shipment, or installation outcome data from which to estimate or validate that rate by vehicle, age, geography, exposure, or windshield design. The output labels also alternate between repairs and replacements, which are different business outcomes.

29. **The fleet exposure base is incomplete and cannot support market-size claims.** The input is only annual Top-100 sales lists for EU and World, not full registrations or vehicle parc. It omits the long tail and does not model geography below those aggregates, imports/exports, scrappage/deregistration, mileage, or vehicle-use exposure. Some source years are incomplete or estimated as already noted, and provenance is not documented.

30. **The two fleet-survival implementations disagree.** Script 1 applies `(1 - 0.0556) ** age`, so a cohort decays in the year after sale. Script 2 treats `age <= 1` as survival 1.0 and begins decay afterward. The same cohort therefore produces different fleet and replacement values depending on which workflow is used.

31. **Forecast quality is not measurable.** Script 2 uses an LLM to produce future annual sales and a self-reported confidence field, but there is no time-series backtest, holdout evaluation, benchmark model, calibrated prediction interval, error metric, or model/data version attached to each result. The strategic score is likewise not validated against later demand or commercial outcomes.

32. **Outputs are not yet procurement/manufacturing decisions.** The UI displays wide tables by model/generation and a composite opportunity score. It cannot aggregate by windshield SKU/part family, show base/upside/downside demand, expose data-quality/identity confidence, reconcile totals, or let planners trace a forecast back through vehicle compatibility, fleet exposure, replacement-rate assumptions, and sources.

These are confirmed gaps in the current repository, not proof that no suitable proprietary data exists outside it. Whether ICOR has historical replacement/claims/shipments and a vehicle-to-windshield fitment catalog is now the most important discovery question.

## Data notes

- Twenty-two Top-100 JSON files cover EU/world datasets from 2015 through 2025.
- EU 2022 has 93 rows and EU 2024 has 97 rather than 100; provenance/completeness needs confirmation.
- `icor_supported_models.txt` contains duplicate keys, including Ford Kuga, Mazda 3, Mercedes C Class, and Mercedes E Class. Script 2's custom parser may accumulate some duplicate-year content, but the format is not valid JSON and is interpreted differently by Script 1.
- The JSON files are valid UTF-8. Mojibake seen in one PowerShell display was a terminal rendering issue, not confirmed file corruption.
- `data/passenger_car_data.xlsx` is a binary generated/data artifact. A read-only follow-up inspection confirmed 29 sheets. `Windshield_Repl_By_Year_EU` has 339 model/generation rows and annual columns from 2016 through 2035; `ICOR_SO_All` has 637 rows and 13 scoring/output columns. These are materialized values produced by the constant-rate logic described in finding 28, not observed replacement outcomes.

## Initial read-only verification evidence

Run on 2026-08-25:

- `python -m compileall -q ui scripts` with `PYTHONPYCACHEPREFIX` redirected to the temporary directory: exit 0, `COMPILEALL_OK`.
- `git status --short --branch`: `## main...origin/main`, no changes at that checkpoint.
- Direct Python reproduction confirmed the NaN behavior and that `Nissan X-Trail / Rogue` creates a path separator in the generated filename.
- Git object checks and source inspection found no current key in HEAD, but history search confirmed the historical exposure described above.
- No representative application run was attempted because only Python 3.14.3 and a mismatched/incomplete local dependency set were present; installing a local environment would have violated the then-active read-only constraint.
- No test/lint/security-audit command was available from the repository because those tools/configurations are absent.
- A read-only workbook inspection confirmed the generated sheet inventory and the key sheet shapes/columns recorded in Data notes. No workbook was modified or exported.

## Recommended implementation order

1. Lucas revokes and rotates the historically exposed OpenAI key; record confirmation without recording the secret.
2. Establish an isolated, reproducible local development environment and a safe local-secret workflow. Do not touch production deployment/secrets.
3. Add characterization tests for ICOR parsing/scoring, duplicate aggregation, NaN handling, generation filtering, safe paths, cache failure semantics, and output isolation.
4. Fix the ICOR parser and define one validated source-data schema with an explicit duplicate policy.
5. Separate pure scoring/forecasting logic from network, UI, and filesystem side effects.
6. Make all generated/cache writes atomic; use unique local run directories and safe filename slugs.
7. Add strict OpenAI response schemas, validation, retries/timeouts, and failure states that are not cached as facts.
8. Improve the local UX iteratively with Lucas reviewing the running local app.
9. Upgrade and lock dependencies, enable secure defaults, and add CI/tests/static/security checks.
10. Design production roles, authentication, per-user isolation, privacy, quotas, and deployment migration only after Lucas explicitly expands scope to the multi-user web version.
11. After key rotation, separately plan the public Git history cleanup; do not perform it casually because it requires force-updating history and coordination.

The clarified product objective changes the recommended product sequence. Before rebuilding forecasting behavior, define the forecast target precisely (replacement events versus distinct vehicles; horizon, geography, and decision cadence), inventory available proprietary outcome and fitment data, and design the canonical vehicle-configuration-to-windshield-SKU model. Preserve the security/environment/test foundation above, but do not invest further in the current constant-rate/LLM forecast as if it were production-valid. Establish a deterministic baseline, backtesting and uncertainty reporting first; add more complex statistical or machine-learning models only when the available data demonstrates that they improve out-of-sample accuracy.

## Local development status

- No local server is currently running.
- No dedicated virtual environment has yet been created.
- No application code fixes have yet been made.
- `uv 0.11.3` is installed locally, but only Python 3.14.3 is currently available. The approved foundation design standardizes the project on Python 3.12; obtaining that runtime belongs to implementation after the worktree/review gate.
- The first subproject specification is committed at `docs/superpowers/specs/2026-08-25-foundation-design.md` in commit `fbd10de`. It covers a reproducible Python 3.12/uv foundation, safe configuration, deterministic audits, characterization tests, secure devcontainer defaults, CI, and documentation. It explicitly excludes forecast changes and proprietary-data ingestion.
- Lucas has authorized an isolated long-lived development branch/worktree. The written foundation specification is still awaiting his content review. After that approval, create the detailed TDD implementation plan and execute it inline because proactive subagent delegation is disabled for this session.

## Terminal identity

- Renaming a Codex conversation does not necessarily rename the integrated terminal.
- A PowerShell session can set `$Host.UI.RawUI.WindowTitle = 'Icor web app'`, but Codex may override it and a child process cannot reliably rename the parent terminal.
- A persistent conditional PowerShell profile or repository-local launcher can be added later if Lucas authorizes that environment change.
- Opening this clone as the Codex project/worktree is the most reliable built-in indication that the terminal belongs to ICOR.

## Current checkpoint

The initial deep review and business-objective alignment review are complete. Lucas approved the delivery direction, disclosed the availability and one-year limitation of proprietary fitment/replacement data, and authorized work to begin on a separate long-lived development branch/worktree. The foundation subproject design is committed as `fbd10de`; application behavior remains unchanged. Create the isolated branch/worktree while returning local `main` to `origin/main`, then await Lucas's content review of the foundation specification before writing the implementation plan. No local server is running.
