# ICOR Web App — Durable Handoff

Last updated: 2026-09-14

## Project objective

Repair and evolve the ICOR web app into a reliable, secure, maintainable, and exceptionally user-friendly product. Development is local-first so Lucas can review progress incrementally. The existing multi-user web deployment is out of scope for modification or deployment until Lucas explicitly authorizes it.

On 2026-08-25 Lucas clarified the business outcome: the app must forecast how many vehicles will require windshield replacement, broken down by brand, model, year, and the exact windshield design. A model/year may have multiple incompatible windshields because of generation, body/design/facelift, trim, equipment, or other configuration differences. The intended customer is a windshield manufacturer using the forecast for product and demand planning. The output therefore ultimately needs to resolve demand to a canonical windshield-compatible vehicle configuration, and ideally to the manufacturer's windshield SKU/part family, rather than stopping at a model or generation label.

## Repository and access

- GitHub owner: `lucascverissim0`
- Repository: `lucascverissim0/icor-webapp` (public)
- Local clone: `C:\Users\LucasCravoVERISSIMO\icor-webapp`
- Remote: `https://github.com/lucascverissim0/icor-webapp.git`
- Production branch: `main`
- Development branch: `development/windshield-demand-platform`
- Review baseline HEAD: `1ba1d7c`
- At the end of the initial review, the tracked worktree was clean.
- Existing Windows Git Credential Manager credentials identified the account as `lucascverissim0` without exposing the token.
- All future Git network commands must be non-interactive. Use `GCM_INTERACTIVE=never`; if authentication expires, stop rather than triggering an account-selection pop-up.
- Clearing a Codex conversation or terminal screen does not normally erase Windows Credential Manager credentials. Credentials can still expire or be revoked externally.
- Protected local production checkout: `C:\Users\LucasCravoVERISSIMO\icor-webapp`, branch `main`, reset to and tracking `origin/main` at `1ba1d7c` on 2026-08-25.
- Long-lived development worktree: `C:\Users\LucasCravoVERISSIMO\icor-webapp-development`, branch `development/windshield-demand-platform`, created at `f5d0bf4` on 2026-08-25.
- All subsequent edits, tests, local servers, and commits must run from the development worktree. Treat the production checkout as read-only. Merging to `main`, pushing, or deploying requires Lucas's explicit authorization after final review.

## User decisions and working rules

- This Codex conversation and all of its terminal work are anchored to
  `C:\Users\LucasCravoVERISSIMO\icor-webapp-development`. Run every project
  command from that development worktree; do not switch this conversation to
  `Video_app`, the protected production checkout, or another repository.
- Lucas explicitly requires strict isolation from the simultaneously open Video Flow
  terminal session: do not read, write, launch, reference, or transfer files, context,
  processes, or decisions between that session and this ICOR conversation.
- Preserve advancement across cleared Codex conversations in this file.
- Lucas requires Codex to optimize for practical productivity and token efficiency on
  all repository work. `AGENTS.md` now defines this as using the shortest reliable path,
  reusing durable context and existing artifacts, batching independent work, and keeping
  output evidence-focused without weakening correctness, security, or verification.
- Every final response must state whether the conversation context is safe to clear, as required by `AGENTS.md`.
- Never store secrets or private/customer data here.
- Run and demonstrate a local app as work progresses.
- Do not modify or deploy the current multi-user web version yet.
- The initial repository review was explicitly read-only. No application code was changed during it.
- On 2026-08-25 Lucas authorized starting the approved delivery sequence and requested brief completion reports covering what changed, what happened, and the next steps.
- The company has a proprietary vehicle-to-windshield fitment catalog and one year of reliably tracked replacement-related history. Integrate it in a later subproject as fitment truth and limited calibration/validation evidence; do not treat one year as sufficient long-run training history.
- Lucas requires all development to remain on a separate long-lived development branch/worktree until the final product has been fully reviewed and he explicitly authorizes a merge. The current actively used application on `main` must remain intact. Do not merge, push, deploy, or modify production from the development worktree without explicit authorization.
- On 2026-08-25 Lucas approved applying one full year of fleet attrition to a
  one-year-old vehicle cohort. Both forecast workflows must use that convention
  consistently when `ICOR-030` is remediated.
- Lucas requested a new responsive, polished local web experience in the development
  branch rather than treating Streamlit as the long-term product UI. Preserve the
  working Streamlit app as a behavioral reference during migration; design the new
  experience and run it locally before considering any production deployment.
- Lucas approved the recommended product-slice direction: the new web experience must
  begin with the future decision-planning workflow centered on canonical windshield
  configurations/SKUs. Current forecast data may appear only as clearly labelled
  prototype evidence; recreating the existing Streamlit screens is not the product goal.
- On 2026-08-25 Lucas selected visual Option B, the planner workbench, and authorized
  creation of a navigable local app that can be adapted while product data is added.
- On 2026-08-25 Lucas approved the recommended technical architecture: a modular
  monolith with a Vite/React/TypeScript planner, a FastAPI adapter, and Python
  domain/application layers over a replaceable read-only demonstration repository.
  The written specification is at
  `docs/superpowers/specs/2026-08-25-planner-webapp-design.md`. Lucas subsequently
  approved the complete specification and authorized implementation.
- On 2026-08-26 Lucas approved the opportunity-ranking design: a new page will rank
  brands, models, and model years by forecast windshield replacements and separately
  apply a moderate production-readiness advantage. Raw demand remains unchanged.
  Exact configuration/SKU production matches are preferred, with an explicitly
  lower-confidence brand/model/model-year fallback when exact identity is unknown.
- Production coverage will use shared backend persistence, initially local SQLite.
  The initial explainable score allocates 80 points to relative demand and 20 points
  to readiness; exact matches receive full readiness weight and broad fallbacks half.
  The ranking strategy boundary must allow a later cost-basis strategy without
  migrating coverage records.
- On 2026-08-26 Lucas clarified that company information is not yet available. The
  next product slice must therefore begin with publicly obtainable vehicle make,
  model, and model-year data. Windshield SKU/fitment integration is deferred until
  proprietary inputs are available, but the architecture must retain a clean future
  mapping boundary. Lucas also reiterated that this is a new web experience expected
  to be substantially better than the first Streamlit version.
- The first real-data geographic scope is the EU aggregate. After the complete EU web
  app is finished and reviewed, the product must expand worldwide because the company
  operates globally. Geography must therefore remain a first-class, replaceable data
  dimension even though the initial UI and validation are EU-only.
- The authenticated landing page must show the full EU opportunity ranking immediately.
  A separate second page must provide targeted vehicle search by make, model, and model
  year. Both views must use the same canonical dataset and calculation services.
- Opportunity order must be driven by estimated windshield replacements and expose
  downside, base, and upside values. The ranking must also show fleet size and fleet
  growth as separate columns so users can make informed decisions; these explanatory
  measures must not be hidden inside an opaque composite score.
- On 2026-08-26 Lucas requested an architectural restart of the new web application
  from a clean product and data design rather than extending the fixture-led prototype.
  The desired historical scope is worldwide passenger-car sales/registrations by brand
  and model from 1995 through the latest available period, with explicit forecast
  horizons through 2028 and 2031. Lucas invited alternative approaches and wants the
  product to be designed collaboratively before implementation.
- On 2026-08-26 Lucas approved using each trustworthy dataset for the years it covers,
  adding further datasets for older periods, and reconciling overlaps rather than
  forcing one source to cover everything. The real-data design must retain every raw
  observation and its provenance, use overlaps to detect agreement or conflict, and
  expose source/data confidence alongside forecast uncertainty. Confidence must remain
  explainable and must not hide missing coverage or turn estimated values into observed
  facts.
- On 2026-08-26 Lucas approved the recommended treatment for incomplete 1995-2009 EU
  model coverage: retain and display observed national evidence, and also permit
  explicitly labelled low-confidence EU estimates derived from it. Observations and
  estimates must remain separate, every estimate must expose its inputs and
  limitations, and uncertainty must widen automatically as coverage weakens.
- On 2026-08-26 Lucas approved the evidence-led real-data architecture: immutable raw
  source releases; normalized observations; a separately governed canonical vehicle
  registry; dependency-aware evidence reconciliation; distinct observed, reconciled,
  estimated, and forecast values; versioned estimate/forecast methods; and API/UI
  provenance drill-down. A deterministic, explainable reconciliation engine will come
  first behind replaceable strategy boundaries; a probabilistic fusion model may be
  introduced only after coverage and source behavior are measured.
- On 2026-08-26 Lucas approved the reconciliation and confidence rules. Measures with
  different meanings are never merged; source observations remain immutable;
  dependency groups prevent correlated publications from being counted as independent;
  deterministic precedence selects a reconciled value; configurable overlap thresholds
  initially classify <=2% as concordant, 2-10% as review, and >10% as conflict; and
  conflicts are never averaged automatically. Evidence confidence is an explainable
  100-point composition of authority (25), publication/revision status (10), coverage
  (25), identity quality (20), and independent agreement (20), with High/Medium/Low/
  Very low bands, component reasons, and caps for provisional or inferred evidence.
  Forecast certainty remains separate and must be backtest-calibrated; uncalibrated
  windshield outputs remain assumption-led opportunity estimates.
- On 2026-08-26 Lucas approved the forecasting chain: reconcile registration history;
  estimate missing EU model-years hierarchically while preserving observation status;
  reconstruct active vehicle cohorts with evidence-calibrated geography/segment
  survival curves; select simple future-registration models through rolling-origin
  backtests; apply an explicit age/geography/vehicle replacement-hazard distribution;
  and propagate all major uncertainty through simulation to P10/P50/P90 outputs for
  2028 and 2031. Modelled history stays separate from observations, forecast confidence
  depends on empirical error/coverage and evidence depth, and insufficiently validated
  outputs are labelled experimental. Replacement outputs remain assumption-led until
  defensible hazard evidence or proprietary history is integrated.
- On 2026-08-26 Lucas approved the real-data UI design. The authenticated landing page
  opens directly to the complete EU opportunity ranking with 2028/2031 horizon,
  geography, search, confidence, and estimate-inclusion controls; separate active-fleet
  and fleet-growth columns; P10/P50/P90 replacement opportunity; evidence/forecast
  confidence; status; freshness; and URL-addressable state. Row evidence workspaces
  expose history, assumptions, source comparisons, confidence components, identity,
  conflicts, missingness, and reproducibility versions. A separate vehicle-search page
  serves make/model/model-year research. Observed and estimated values remain visually
  distinct, warnings cannot rely on color, missing periods are not visually invented,
  provenance remains available on mobile, and internal source diagnostics stay outside
  the decision-focused user workflow.
- On 2026-08-26 Lucas approved the final operational-safety and testing section. Source
  ingestion uses immutable manifests and atomic candidate-snapshot promotion; a failed
  build preserves the last known-good snapshot and never falls back to fixtures. The
  required verification covers parser contracts, identity, reconciliation/confidence,
  estimation/forecast invariants and backtests, source aggregate reconciliation,
  snapshot/versioned API behavior, authorization/security, frontend/browser/
  accessibility/responsive behavior, and a deterministic clean-room rebuild.
- The complete approved design is committed as `c0071db` at
  `docs/superpowers/specs/2026-08-26-real-data-evidence-forecasting-design.md`. Its
  placeholder, consistency, ambiguity, scope, and whitespace self-review passed.
- On 2026-08-26 Lucas explicitly approved beginning the build from the complete written
  design. The first executable subproject plan covers immutable evidence contracts,
  release storage, the SQLite evidence ledger, validation, deterministic candidate
  builds, atomic last-known-good snapshot promotion, and a clean-room CLI. It is at
  `docs/superpowers/plans/2026-08-26-evidence-snapshot-foundation-implementation.md`.

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
- Preliminary source research on 2026-08-26 confirmed that no reviewed free official
  source provides one complete worldwide make/model series from 1995 to the present.
  OICA supplies country/category totals, and ACEA states that its public registration
  figures are by country and brand rather than model. Commercial candidates include
  MarkLines (country/brand/model data and claimed 99% global-sales coverage), JATO
  Volumes/ModelMix (registration volumes across 40+ markets), and S&P Global Mobility
  (new-registration coverage across 150+ countries). Licensing, historical depth,
  redistribution rights, exact model granularity, corrections, and export/API terms
  must be verified with vendors before selecting a canonical source.
- Follow-up official-source validation on 2026-08-26 identified the European
  Environment Agency's Regulation (EU) 2019/631 passenger-car monitoring dataset as
  the strongest public first-party source for an initial real EU registration layer.
  It contains member-state submissions with manufacturer/make and commercial-name
  fields and currently spans 2010-2024. The EEA labels 2023 final and 2024
  provisional. It does not satisfy the requested 1995-present worldwide history, so
  using it requires an explicit product decision to launch the EU real-data phase at
  2010 and defer 1995-2009/worldwide completeness to licensed or additional sources.
- Additional official-source validation found complementary national evidence. The UK
  Department for Transport/DVLA publishes model-level first registrations for Great
  Britain from 2001 and model-level licensed-stock data from 1994 Q4. Germany's KBA
  publishes registration tables by brand/model series, including overlapping annual
  FZ 10 datasets. France's SDES publishes corrected make/model fleet evidence and
  documents VIN/type-based correction methods, though the reviewed national fleet
  series is substantially later than 1995. These sources can validate identity,
  overlap, and national trends but cannot simply be summed into an EU total: scopes,
  measures, suppression rules, revisions, and upstream dependencies differ. Agreement
  between EEA and a national register is useful validation but is not necessarily
  independent evidence.
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

- A private local visual-companion session is active on port 61376 for the new
  planner design. Its first screen compares three information hierarchies using
  product-specific content: executive dashboard, planner workbench, and guided
  planning flow. Lucas selected the planner workbench (Option B). The exact session
  URL contains a private local access key and is intentionally not recorded here.
  The session files live under ignored `.superpowers/`; `.gitignore` excludes that
  directory. The companion server auto-exits after four idle hours; the Streamlit
  server remains separate on port 8501.
- On 2026-08-25 the local app startup failure was reproduced as
  `StreamlitSecretNotFoundError` at `ui/app.py:120`: `st.secrets.get("users", {})`
  forced Streamlit to parse a missing local secrets file before rendering login. The
  call now uses the existing exception-safe `_safe_get`, so the documented secret-free
  offline startup path remains available. `tests/test_app_startup.py` runs the real
  Streamlit app with no injected secrets and verifies a clean login page with username
  and password fields. TDD evidence: the test first failed on the exact secrets
  traceback and then passed after the one-line fix.
- Fresh verification after the startup fix: `uv lock --check` resolved 96 packages;
  `uv run ruff check src tests scripts/audit_baseline.py` reported `All checks passed!`;
  `uv run python -m compileall -q ui/app.py` exited zero; and
  `uv run pytest -p no:cacheprovider` reported 15 passed with the four documented strict
  XFAILs (`ICOR-001`, `ICOR-006`, `ICOR-009`, and `ICOR-030`). The local Streamlit
  health endpoint continued to return `ok` on port 8501. Its stdout file still contains
  the historical pre-fix traceback because no post-fix browser session has rewritten
  that log; the Streamlit AppTest regression is the fresh render-level verification.
- A local Streamlit server is running from the development worktree at `http://127.0.0.1:8501` (also reachable as `http://localhost:8501`). Its health endpoint returned HTTP 200 with `ok`, the browser was opened to the app, and the startup log reported Uvicorn listening on `127.0.0.1:8501` with no stderr output. The listener PID at verification time was 7604. Clearing the Codex conversation does not stop it; stopping that process or rebooting does.
- An isolated `.venv` now uses uv-managed CPython 3.12.13 with 96 locked packages. `uv.lock` is authoritative and `requirements.txt` is an exact production export.
- The global Python 3.14.3/pytest 9.1 process hung while finalizing intentional RED failures. Collection exited normally, and disabling AnyIO/capture did not change the hang. The locked Python 3.12.13 environment exits normally with pytest 9.1.1; this isolates the problem to the unsupported global runtime rather than pytest 9 itself.
- Foundation Task 1 is committed as `2704d75`: Python 3.12 selector, bounded project metadata, lock/export, toolchain tests, and ignored local test/lint/output caches.
- Foundation Task 2 is committed as `2ace28e`: installed import-safe `icor` package and validated non-secret settings. A subprocess regression test caught and prevented a pytest-only import-path illusion; six combined toolchain/configuration tests passed after the package installation fix.
- Foundation Task 3 is committed as `5936cf4`: deterministic read-only audit core. Its tests passed 3/3 and verified 22 market files, 2,190 records, and 50 null-unit findings. The audit parses forecast constants through AST without importing side-effecting scripts.
- Foundation Task 4 is committed as `88d6364`: read-only baseline-audit CLI. Its deterministic two-run test passed, preserved the canonical workbook byte-for-byte, and the manual ignored-local run reported 58 total structured findings.
- Checkpoint security verification initially found `PYSEC-2026-1845` in pytest 8.4.2. The constraint and lock were raised to pytest 9.1.1 in `f28a106`. Fresh results afterward: 10 tests passed, Ruff reported no errors, and pip-audit reported no known third-party vulnerabilities. Pip-audit skipped only the expected unpublished local package `icor-windshield-demand`.
- Foundation Task 5 is committed as `4bebab1`: tests clear integration credentials, prohibit sockets, scan tracked text files without leaking matching values, safely AST-load individual Script 1 functions, and preserve four strict known-defect XFAILs (`ICOR-001`, `ICOR-006`, `ICOR-009`, and `ICOR-030`).
- Foundation Task 6 is committed as `4840beb`: the devcontainer now uses Python 3.12 Bookworm, pinned uv 0.11.3, the locked sync, and secure Streamlit defaults. Read-only GitHub Actions configuration covers Windows/Linux lint, lock, tests, and a Linux dependency audit. It has been validated locally but not executed remotely because nothing was pushed.
- Foundation Task 7 is committed as `e940dec`: the README and local development guide document the business objective, worktree isolation, Python/uv setup, audit/lint/test/app commands, local-secret safety, and the prototype status of the current forecasts. The committed Streamlit secret template contains only empty values and an intentionally disabled example user.
- Fresh Task 8 verification on 2026-08-25: `uv lock --check` resolved 96 packages; regenerating `requirements.txt` produced no diff; `uv run ruff check src tests scripts/audit_baseline.py` reported `All checks passed!`; `uv run pytest` reported 14 passed and exactly 4 expected XFAILs; `uv run pip-audit` reported no known vulnerabilities and skipped only the unpublished local package.
- Two independent `audit_baseline.py` outputs had the identical SHA-256 `0B764BC05BF85ED38905E44D667EC80BDE68DE50D375F981DBE8A5F0194ECDC6`. The canonical workbook remained byte-identical at Git object hash `b39db4fd9b4e51cd2d1a138d6f0b347a9c561494`.
- Production isolation was freshly verified: the protected checkout is clean on `main`; both `HEAD` and `origin/main` equal `1ba1d7c41a5fa8354134685b5c85509a0b8f6137`; `git diff --exit-code origin/main --` returned zero.
- No existing UI or forecasting behavior has been changed. The new foundation package and audit are additive.
- The first subproject specification is committed at `docs/superpowers/specs/2026-08-25-foundation-design.md` in commit `fbd10de`. It covers a reproducible Python 3.12/uv foundation, safe configuration, deterministic audits, characterization tests, secure devcontainer defaults, CI, and documentation. It explicitly excludes forecast changes and proprietary-data ingestion.
- Lucas authorized an isolated long-lived development branch/worktree. It now exists at `C:\Users\LucasCravoVERISSIMO\icor-webapp-development` on `development/windshield-demand-platform`; local `main` is restored to the production baseline. Continue the approved inline implementation plan at Task 5 because proactive subagent delegation is disabled for this session.
- Lucas approved the foundation specification on 2026-08-25 and clarified that its non-goals are deferrals only: the full product program must review and may replace forecast formulas, identity handling, proprietary-data integration, statistical methods, and machine-learning models. No current algorithm is exempt from evidence-based review.
- For model-controlled architecture and forecasting evaluation work, current official OpenAI guidance supports GPT-5.6 Sol as the flagship choice and `max` reasoning effort for the hardest quality-first work, with `xhigh` as the comparison setting. The root model of an already-running Codex turn cannot be silently changed by repository code; preserve this recommendation for future model-selectable sessions and explicitly benchmark app-side model choices rather than assuming them.
- On 2026-08-25 Lucas requested the Codex effort setting be changed from Fast to the recommended level. The installed GPT-5.6-Sol metadata identifies `medium` as its default reasoning level and exposes Fast separately as a speed tier. The machine-level Codex config now contains `model_reasoning_effort = "medium"`; `codex --strict-config --version` accepted the configuration. This applies to new sessions, not retroactively to an already-running turn.
- The approved planner implementation plan is committed as `296c48c`. The product
  slice is implemented through `4810036`: typed Python domain/application layers, a
  replaceable deterministic demonstration repository with eight windshield
  configurations, FastAPI endpoints and OpenAPI export, a responsive React planner,
  URL-backed filters/sort/pagination, traceable configuration detail, error/empty/loading
  states, accessibility checks, and isolated browser/contract/CI gates. Final visual
  refinements are recorded in `8b4f7ed` and `46bd3b1`.
- A final visual-QA defect found at 1440px was corrected with a browser regression:
  the contextual comparison is retained only at widths that can fit useful filter,
  comparison, and detail columns; 1440px now uses the uncluttered full-detail route.
- Fresh planner verification on 2026-08-26: Python reported 50 passed plus the four
  documented strict XFAILs; Ruff and `uv lock --check` passed; pip-audit and npm audit
  reported no known vulnerabilities; OpenAPI drift check passed; frontend unit tests
  reported 33 passed; lint, TypeScript/build, and the eight-test Chromium planner suite
  passed.
  Final desktop, mobile, and deep-link screenshots were inspected from the live app.
- On 2026-08-26 the managed launcher was freshly started as background PID 9732 and
  serves the local planner at `http://127.0.0.1:5300/planner` and its FastAPI docs at
  `http://127.0.0.1:8140/docs`. Both ports were verified listening, and Edge was
  verified responsive, maximized, and foregrounded with the title `ICOR Planner`.
  Launcher logs are under ignored `.local/planner-20260826-115540.*.log`. Clearing the
  Codex conversation or terminal screen does not stop this background process; stopping
  PID 9732 or rebooting does. The documented default ports remain 5173/8000.
- The approved opportunity-ranking implementation plan is committed as `39b010e`.
  Implementation is committed through `8ec548f`: reconciled model-year demonstration
  demand, immutable coverage/ranking semantics, versioned transactional SQLite CRUD,
  canonical exact-before-fallback matching, opportunity application services, six
  versioned FastAPI endpoints, generated OpenAPI types, and the responsive React
  opportunity/coverage workbench.
- The implemented `demand_readiness_v1` policy leaves raw downside/base/upside demand
  unchanged, uses tie-aware relative base-demand percentiles for 0–80 points, and uses
  exact/fallback/uncovered base units for 0–20 readiness points. Exact matches take
  precedence over broader vehicle-year fallbacks; fallbacks receive half readiness
  weight and demand is never counted twice.
- Production coverage defaults to ignored shared-local SQLite at
  `.local/production-coverage.sqlite3` and can be redirected with
  `ICOR_COVERAGE_DB`. It is local prototype state without authentication,
  authorization, attribution, backup, or audit-grade history; the UI and documentation
  prohibit secrets/customer data and explain the lower precision of fallback records.
- Fresh opportunity-slice verification on 2026-08-26: `uv lock --check` and Ruff
  passed; pytest reported 93 passed plus the four documented strict XFAILs; pip-audit
  and npm audit reported no known vulnerabilities; OpenAPI drift passed; all 47
  frontend tests, ESLint, TypeScript/production build, and all 13 Chromium journeys
  passed. Browser coverage includes exact create/edit/delete, fallback confirmation,
  committed ranking refetch, planner regressions, keyboard focus, WCAG serious/critical
  checks, and 390px/1440px overflow checks. Captured opportunity desktop/mobile pixels
  were inspected with no clipping, overflow, hierarchy, or responsive-layout defect.
- A fresh current-code review launcher runs as background PID 18272 at
  `http://127.0.0.1:5310/opportunities`; its API and docs use port 8150. Both the API
  health/opportunity reads and web route returned HTTP 200, and the opportunity page
  was opened in the default browser. Ignored logs are
  `.local/opportunities-20260826-125116.out.log` and `.err.log`. Clearing conversation
  context does not stop it; stopping PID 18272 or rebooting does.
- This product slice remains fixture-only. Proprietary fitment/outcome ingestion,
  calibrated forecasting and backtesting, authentication/roles, multi-user behavior,
  deployment, pushing, and merging remain explicitly deferred. The existing strict
  XFAIL defects `ICOR-001`, `ICOR-006`, `ICOR-009`, and `ICOR-030` are not silently
  treated as fixed.

## Terminal identity

- Renaming a Codex conversation does not necessarily rename the integrated terminal.
- A PowerShell session can set `$Host.UI.RawUI.WindowTitle = 'Icor web app'`, but Codex may override it and a child process cannot reliably rename the parent terminal.
- A persistent conditional PowerShell profile or repository-local launcher can be added later if Lucas authorizes that environment change.
- Opening this clone as the Codex project/worktree is the most reliable built-in indication that the terminal belongs to ICOR.

## Current checkpoint

The full real-data evidence and forecasting design is approved for implementation and
committed as `c0071db`. It defines the EEA/KBA/UK initial source strategy, immutable
evidence ledger, canonical identity governance, dependency-aware reconciliation,
explainable confidence, separate low-confidence 1995-2009 EU estimates, cohort fleet
reconstruction, backtested 2028/2031 uncertainty forecasts, atomic snapshots, and the
real-data opportunity/search experience. Lucas approved beginning implementation on
2026-08-26. The first staged plan is written and self-reviewed at
`docs/superpowers/plans/2026-08-26-evidence-snapshot-foundation-implementation.md`;
it covers the evidence and atomic-snapshot foundation. The phase began with no
application-code changes.

Task 1 of that foundation is committed on `development/windshield-demand-platform`:
immutable slotted
domain contracts now exist in `src/icor/domain/evidence.py` and
`src/icor/domain/snapshots.py`, with 41 focused contract tests. They enforce UTC
timestamps, identifier/hash/count/coverage validation, original-plus-normalized source
labels, five-component capped evidence confidence with 40/60/80 bands, prohibition on
publishing ambiguous/rejected/unresolved identity mappings, ordered uncertainty
intervals and inputs, and deterministic snapshot manifest invariants. Fresh
verification: focused pytest 41 passed; `uv run ruff check src tests` passed; full
pytest passed 134 tests with exactly the four documented strict XFAILs. The existing
unrelated `AGENTS.md` modification remains untouched. The detailed ignored task report
is `.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-1-report.md`.

The approved opportunity-ranking and production-coverage slice is implemented and
verified on `development/windshield-demand-platform`. It is live for review at
`http://127.0.0.1:5310/opportunities`, while the earlier configuration planner remains
available at `http://127.0.0.1:5300/planner`. The new page ranks brands, models, and
model years, drills into contributing configuration/model-year demand, and manages
exact or deliberately broader local production coverage. All forecast and fitment
values remain clearly labelled deterministic demonstration evidence; local coverage
records do not validate the forecast.

Production remains read-only on `main` at `1ba1d7c`; nothing has been pushed, merged,
or deployed. The next real-data phase requires Lucas to provide secure local exports
and field definitions for the proprietary vehicle-to-windshield fitment catalog and
the reliably tracked replacement/production history. That phase must approve canonical
identity mapping, data-quality checks, ingestion, reconciliation, baseline/backtest
design, uncertainty, and access controls before any value is labelled validated.
Streamlit remains a temporary behavioral reference, and `ICOR-001`, `ICOR-006`,
`ICOR-009`, and `ICOR-030` remain strict XFAILs for their separate TDD remediation.
A concurrent uncommitted `AGENTS.md` change remains untouched.

Task 2 is committed and added the deterministic evidence serialization and manifest boundary in
`src/icor/evidence/`. `canonical_json_bytes` emits sorted compact UTF-8 JSON with one
trailing newline; it explicitly serializes contract dataclasses, enums, UTC timestamps,
dates, decimals, tuples, and paths while rejecting floats. `sha256_file` hashes exact
file bytes. Strict release and snapshot manifest loaders reject malformed/non-UTF-8
JSON, duplicate or unknown/missing keys, invalid enum/date/hash values, unsafe artifact
paths, and duplicate snapshot release IDs before domain validation. Release-manifest
writes use a same-directory temporary file, flush/fsync, and atomic replacement. Fresh
verification: focused evidence tests passed 21/21; Ruff passed; full pytest reported
155 passed with exactly the four documented strict XFAILs. The unrelated `AGENTS.md`
modification remains untouched. Task report:
`.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-2-report.md`.

Task 2 fix round 1 closes a writer-boundary gap: `write_release_manifest` now validates
the domain object's artifact path before creating its temporary file, so otherwise-valid
domain instances cannot persist absolute or traversing paths that the strict loader
would reject. TDD evidence: the new focused writer regression first failed with no
`ManifestError`, then `tests/evidence/test_release_manifests.py` passed 18/18 and Ruff
reported `All checks passed!`. The deferred `load_snapshot_manifest` minor was not
changed. The unrelated `AGENTS.md` modification remains untouched.

Task 3 is committed as `6955489` (`feat: store immutable source releases`).
`src/icor/infrastructure/release_store.py` now stages only source artifacts whose
byte count and SHA-256 match a `ReleaseManifest`, copying them through a same-root
`.staging/<uuid>` directory before publishing the complete release directory under
`<source_id>/<release_id>`. The store atomically writes the manifest, validates
identifiers, detects incomplete or tampered releases, prevents replacement by different
content, permits exact idempotent restaging, and lists releases in stable ID order.
`.local/evidence/` is explicitly ignored. TDD evidence: the new focused test file first
failed at collection because `icor.infrastructure.release_store` did not exist, then
passed 10/10. Fresh verification: `uv run ruff check src tests` reported `All checks
passed!`; `uv run pytest -p no:cacheprovider` reported 166 passed with exactly the four
documented strict XFAILs (`ICOR-001`, `ICOR-006`, `ICOR-009`, and `ICOR-030`). No parser
or real source data was added; the unrelated `AGENTS.md` modification remains untouched.

Task 3 fix round 1/5 is committed as `c5f30a0` (`fix: harden immutable release storage`).
The immutable release store now serializes every stage under an atomic per-release lock,
uses native non-replacing directory publication (`MoveFileW` on Windows and Linux
`renameat2` with `RENAME_NOREPLACE`), and fails closed on unsupported platforms. It
enforces global release-ID uniqueness across source IDs, rejects symlinked or
out-of-root store paths, and requires the manifest artifact path to exactly match the
stored `artifact<source suffix>` filename. TDD evidence: the first added regressions
reported three expected safety failures, and the subsequent global-lock regression
reported one expected failure. Fresh focused verification: 14 passed; three symlink
regressions skipped only because Windows returned `WinError 1314` while creating test
symlinks. Ruff passed. The ignored detailed report is
`.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-3-report.md`.

Task 3 fix round 2/5 is committed as `0c4ebc3` (`fix: make release staging concurrency
safe`). Symlink regressions now skip only the explicit Windows developer-privilege
denial (`WinError 1314`); every other symlink setup error fails the test. A real
two-thread test synchronizes two distinct releases after both observe `.locks` absent.
The former check-then-create failure was reproduced as `FileExistsError`; root and
shared directory setup now uses idempotent creation followed by the existing symlink,
directory-type, and containment checks. Fresh focused verification: 15 passed and the
three expected Windows-privilege symlink skips; Ruff passed. The unrelated `AGENTS.md`
modification remains untouched.

Task 4 adds an immutable versioned SQLite evidence ledger in
`src/icor/infrastructure/sqlite_evidence_repository.py`, behind the
`EvidenceRepository` application protocol. Version 1 migrates new databases and
rejects corrupt, missing, or newer schema metadata; stores normalized releases,
observations, vehicles, mappings, published values with ordered inputs, and snapshots;
uses bound parameters, explicit transactions, foreign keys, WAL/FULL durability, and
SQLite read-only URI connections. No update/delete API exists. Its real temporary-DB
tests cover migration safety, immutability, rollback, references, ordered input
retention, unresolved-input rejection, and deterministic reads. TDD evidence: the new
test module first failed at collection because the repository module was absent, then
passed 16/16. Fresh Ruff output was `All checks passed!`; full pytest ran 194 collected
tests with 187 passed, 3 Windows symlink-privilege skips, and exactly the four documented
strict XFAILs. The unrelated `AGENTS.md` modification remains untouched. Detailed
ignored report: `.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-4-report.md`.

Task 4 fix round 1 hardens the SQLite ledger after review. Published values now reject
each input observation unless canonical vehicle, measure, unit, geography and version,
and exact period match. SQLite v1 now has a normalized `snapshot_release` table with
ordered membership and database foreign keys rather than a JSON release-ID bundle.
Schema inspection validates all required tables, columns, primary/unique keys, and
foreign keys; migration executes each statement inside an explicit transaction so a
late DDL error rolls back without even a schema-version table. Tests now exercise the
actual SQLite read-only connection, full published-value and snapshot round-trips,
canonical identity uniqueness, all list orderings, membership foreign keys, schema
corruption, and migration rollback. TDD evidence: 9 regressions first failed for the
known missing checks; focused verification passed 29/29; Ruff reported `All checks
passed!`; full pytest reported 200 passed, 3 Windows symlink-privilege skips, and the
four documented strict XFAILs. The unrelated `AGENTS.md` modification remains
untouched. Detailed ignored report:
`.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-4-report.md`.

Task 4 fix round 3 closes the remaining schema-fingerprint gaps. SQL normalization now
lowercases and collapses whitespace only outside single-quoted literals, preserving
literal case and escaped quotes; a changed enum literal is therefore rejected. The v1
migration now declares named unique indexes for canonical vehicle identity and source
row location. Schema validation compares the exact set of application-owned tables and
explicit indexes (while SQLite autoindexes remain deliberately excluded because their
SQL is null), rejecting either a missing required index or an unexpected one. TDD RED
reported the expected enum-literal and two index failures; focused ledger verification
then passed 36 tests and Ruff passed. The deferred module-wide E501 suppression and
unrelated `AGENTS.md` change remain untouched. Detailed ignored report:
`.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-4-report.md`.

Task 4 fix round 2 completes the v1 schema-integrity boundary. Instead of checking
only names, the ledger now derives a canonical fingerprint for every `CREATE TABLE`
statement from the v1 migration and compares it to SQLite's persisted schema. This
rejects altered column types/nullability, primary and unique keys, foreign keys, and
enum/relationship checks. TDD corruption regressions first failed for a removed
source-release measure enum check, a changed required type, and relaxed `NOT NULL`,
then passed. An honest temporary candidate-schema probe omitted `snapshot_release` and
failed the membership-integrity expectation with `OperationalError: no such table:
snapshot_release`; after restoring the contract, the actual membership foreign-key test
passed. Fresh focused verification: 32 passed; Ruff passed. The module-wide E501
suppression remains intentionally deferred, and the unrelated `AGENTS.md` modification
remains untouched. Detailed ignored report:
`.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-4-report.md`.

Task 4 fix round 4 makes schema fingerprints insensitive to optional whitespace around
SQL punctuation without weakening their semantic checks. `_normalize_schema_sql` now
extracts lexical tokens, lowercases unquoted tokens, and copies quoted tokens verbatim,
including doubled-quote escapes. Regressions cover equivalent `t (x TEXT)`/`t(x text)`
formatting, meaningful type-token changes, and escaped literal changes. The exact
punctuation regression first failed with `create table t (x text)` versus
`create table t(x text)`, then passed after the tokenizer change. Fresh verification:
`uv run pytest tests/infrastructure/test_sqlite_evidence_repository.py -v` reported
38 passed, and `uv run ruff check src tests` reported `All checks passed!`. The
deferred module-wide E501 suppression and unrelated `AGENTS.md` modification remain
untouched. Detailed ignored report:
`.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-4-report.md`.

Task 4 fix round 5 closes the SQLite lexical-boundary defect in schema fingerprints.
The normalizer now recognizes SQLite's contiguous symbolic multi-character operators
longest-first, including the three-character JSON extraction operator, while split
operator characters remain separate tokens. Table-driven regressions cover `!=`,
`<>`, `<=`, `>=`, `==`, `||`, `<<`, `>>`, `->`, and `->>` against whitespace-split
forms; exact quoted-literal/escaped-quote preservation; and valid punctuation
whitespace equivalence. The initial operator table failed all 10 cases because each
pair produced an identical fingerprint, then passed all 10 after the matcher. A
mutation check reproduced the 10 failures after temporarily removing the matcher and
passed all 17 normalization cases after restoration. Fresh final verification:
`uv run pytest tests/infrastructure/test_sqlite_evidence_repository.py -v` reported
52 passed, and `uv run ruff check src tests` reported `All checks passed!`. The
deferred module-wide E501 suppression and unrelated `AGENTS.md` modification remain
untouched. Detailed ignored report:
`.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-4-report.md`.

Task 5 adds read-only release and candidate-snapshot quality gates in
`src/icor/evidence/validation.py`. `ReleaseValidator` blocks promotion for unavailable
or unreadable artifacts, checksum/byte-size mismatch, missing terms metadata,
non-conserving record counts, and reversed coverage. `SnapshotValidator` opens the
ledger using SQLite `mode=ro`, blocks promotion for hash/release/count mismatches and
detects orphan inputs, negative values, invalid or unordered intervals, and unresolved
published mappings. Findings are frozen, sorted by severity/code/record ID, and use
fixed sanitized messages that contain no paths, raw rows, credentials, or stack traces.
TDD RED was the expected missing validation-module import; focused GREEN reported 16
passed. Fresh Ruff passed. The full Python suite reported 239 passed, 3 Windows
symlink-privilege skips, and the four documented strict XFAILs. Task 6 must checkpoint,
VACUUM, and close candidate SQLite files before hashing so this file-hash validation
does not depend on a WAL sidecar. Detailed ignored report:
`.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-5-report.md`.
Task 5 is committed with `feat: enforce evidence snapshot quality gates`.

Task 5 fix round 1 hardens the release/snapshot gates. Ordered P10/P50/P90 intervals
now reject negative bounds; orphan inputs check both the observation and published-value
parent; manifest release membership is reconciled against releases actually used by
observations while unused stored releases remain allowed; and publication validation
joins inputs through observations and identity mappings so forged published statuses
cannot hide unresolved links. Database-derived record IDs are sanitized to the domain
identifier grammar before findings are created, preventing BLOB/mixed-type sort crashes
and raw-ID leakage. TDD RED reported the five expected missing invariant failures;
focused GREEN reported 23 passed and Ruff passed. A linked-observation mutation check
failed when its mapping-status join was removed and passed after restoration. The simultaneous checksum/byte-size
mismatch behavior is deliberately unchanged for the deferred Minor. Detailed ignored
report: `.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-5-report.md`.

Task 6 implements deterministic candidate builds and atomic last-known-good promotion.
`SnapshotBuildRequest` now makes the UTC `build_as_of` instant an explicit identity
input alongside sorted release IDs and artifact hashes, all eight method/registry
versions, and the deterministic seed. `SnapshotBuilder` verifies releases, loads an
isolated scratch ledger, replays every record in stable primary-key order, checkpoints
WAL, switches to a single-file journal, runs `VACUUM`, closes handles, hashes the final
database, and writes canonical `snapshot.json` and `validation.json` artifacts beneath
`candidates/<snapshot_id>`. `SnapshotStore` verifies candidate and copied target bytes
before an fsynced atomic `active.json` replacement; interrupted writes, changed hashes,
missing files, and invalid candidates leave the previous pointer unchanged. Previous
snapshot directories are never deleted or replaced, repeat promotion is idempotent,
and active repositories open read-only with typed unavailable errors and no fixture
fallback. A Windows regression also corrected `SnapshotValidator` to close its
read-only SQLite connection rather than relying on the transaction-only connection
context manager. TDD RED was the expected two missing-module collection errors; focused
GREEN reported 18 passed. Fresh Ruff reported `All checks passed!`; full pytest
reported 264 passed, 3 documented Windows symlink-privilege skips, and the four strict
legacy XFAILs. No server or other process was started. Detailed ignored report:
`.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-6-report.md`.

Task 6 fix round 1 closes the identity and publication review findings. Builder and
promotion now share one canonical identity function covering deterministic time, seed,
all version fields, and persisted release hashes; promotion also requires canonical
candidate status and warnings. Releases are compared with a second verified stored
state after loading. A narrow no-follow filesystem boundary rejects symlink/reparse and
out-of-root components, flushes copied files and publication directories, publishes
read-only targets, and revalidates the stable target inside pointer replacement. An
interprocess atomic-directory lock serializes the complete promotion and makes
same-ID retries byte-for-byte idempotent. TDD RED reproduced 13 forged-identity
acceptances, release replacement, the missing filesystem seam, missing durability and
locking injection, and a real Windows junction escape. Final focused verification was
36 passed with four explicit `WinError 1314` symlink-privilege skips; full pytest was
282 passed, 7 skipped, and the four documented strict XFAILs. `uv run ruff check src
tests` passed. The detailed ignored report is
`.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-6-report.md`.
The unrelated `AGENTS.md` modification remains untouched; no server or external process
was started.

Task 6 fix round 2 closes the remaining exact-set, stable-publication, and crash-lock
blockers. Promotion now rejects any mismatch between the complete persisted
`source_release` ID set and manifest release IDs. Final verification and pointer
replacement run while no-follow identity handles remain open for the target directory
and all three files: Windows handles deny write/delete sharing, while POSIX descriptors
anchor device/inode identity; held file bytes and path identities are checked before
and after replacement, with atomic restoration of the prior pointer on detected change.
The promotion lock is now an OS-owned byte/range lock (`msvcrt.locking` or
`fcntl.flock`) whose ownership disappears on process death. RED reproduced extra-release
acceptance, a real post-verification target rename, and a subprocess crash leaving the
old directory lock until timeout. Final focused verification reported 39 passed and
four explicit Windows symlink-privilege skips; full pytest reported 285 passed, 7
skipped, and the four documented strict XFAILs. `uv run ruff check src tests` passed.
Detailed ignored evidence is in
`.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-6-report.md`.
The unrelated `AGENTS.md` change remains untouched; no server or external process was
started.

Task 6 fix round 3 closes the remaining POSIX verify-to-seal race. Stable no-follow
target handles are now acquired and their identities/content sealed before any final
promotion validation. The same held handles are re-hashed and path identities rechecked
immediately after validation and across pointer replacement, so an in-place POSIX write
after verification cannot become the trusted baseline. A deterministic cross-platform
regression uses real descriptors with POSIX write-sharing semantics and reproduced the
old acceptance (`DID NOT RAISE`); it now fails typed while preserving exact LKG pointer
bytes. Final focused verification reported 40 passed with four explicit Windows
symlink-privilege skips; full pytest reported 286 passed, 7 skipped, and the four
documented strict XFAILs. `uv run ruff check src tests` passed. Detailed ignored
evidence is in
`.superpowers/sdd/2026-08-26-evidence-snapshot-foundation-implementation/task-6-report.md`.
The unrelated `AGENTS.md` change remains untouched; no server or external process was
started.

Task 7 adds the clean-room evidence snapshot CLI in
`scripts/build_evidence_snapshot.py`. `stage-release`, `build`, `promote`, `status`,
and `verify` require an explicit root; roots resolving outside this repository require
`--allow-external-root`. Every application write stays beneath that root, stdout is one
canonical JSON object, stderr is static and sanitized, and exit classes are 0 success,
2 invalid/unsupported input, 3 failed validation/operation, and 4 unavailable active
state. Production composition intentionally registers no source parser: an unregistered
manifest parser returns typed `unsupported_parser`. The only loader is injected from
`tests/integration/test_clean_room_evidence_snapshot.py`; it reads exactly two fictional
Example Motors rows, writes normalized-label observations/mappings, and publishes no
model value. No EEA/KBA/UK parser or real source extract was added.

Task 7 strict TDD first failed at collection because the CLI module did not exist. The
focused suite then reached 6/6 after covering deterministic byte-identical builds across
two external temporary roots, promotion/status/verification, socket prohibition,
outside-root write containment, unavailable status, invalid input, unsupported default
parser, tampered-candidate exit 3, canonical JSON, raw-row exclusion, and unexpected
loader-error sanitization. The latter regression independently failed with a raw-message
`RuntimeError` before the static error boundary and passed afterward. Fresh full pytest
reported 292 passed, 7 documented Windows symlink-privilege skips, and the four strict
legacy XFAILs. `uv run ruff check src tests scripts/build_evidence_snapshot.py` and
`git diff --check` passed; the latter emitted only existing Windows line-ending warnings.
No network request, server, production/customer data, push, merge, deploy, main-branch,
or Streamlit action occurred. The unrelated pre-existing `AGENTS.md` modification remains
untouched. The next planned work is Task 8 foundation documentation/checkpointing; its
planned broad `ruff check ... scripts` currently encounters 486 pre-existing findings in
untouched legacy scripts and needs an explicit maintained-scope decision rather than an
unrelated Task 7 rewrite.

Task 7 fix round 1 closes all three Important review findings. The CLI now keeps the
lexical explicit root after policy resolution and holds a reparse-aware root identity
for every command: Windows permits child writes but denies root deletion/replacement,
while POSIX holds a no-follow directory descriptor and revalidates device/inode identity.
Substitution after `_safe_root` and a real Windows junction root are rejected as typed
`invalid_root` without writing the external target. Stored-release integrity failures
during `build` now translate to exit 3 `snapshot_validation_failed`; malformed staging
input remains sanitized exit 2. `SnapshotStore.open_active_snapshot()` resolves and
verifies one active pointer/immutable target and returns its matching manifest plus
read-only repository; CLI `verify` no longer resolves active state twice. A deterministic
promotion seam advances `active.json` between pointer read and result construction and
proves the reported snapshot ID, manifest count, and repository count still come from
the same earlier immutable snapshot. RED reported four expected failures, six passes,
and one explicit `WinError 1314` symlink skip. Final relevant verification reported 50
passed and five explicit Windows symlink-privilege skips across integration,
snapshot-build, and snapshot-store suites; the real junction test passed. Scoped Ruff
and `git diff --check` passed. Argparse help behavior remains deferred as requested. No
network, server, production/customer data, push, merge, deploy, main-branch, or Streamlit
action occurred; the unrelated `AGENTS.md` edit remains untouched.

Task 7 fix round 2 closes the remaining POSIX lexical-root race. `pin_root` now validates
the held POSIX directory descriptor through `/proc/self/fd/<fd>` or `/dev/fd/<fd>`, yields
that descriptor-relative alias as the operation root, and fails closed when neither
supported alias resolves to the held directory's exact device/inode. The same
`SnapshotFilesystem` instance and anchored operation root are injected into
`ReleaseStore`, `SnapshotBuilder`, and `SnapshotStore`; the filesystem permits the one
live descriptor alias while retaining no-follow/reparse checks for every descendant.
Windows continues yielding the lexical root while its handle denies delete sharing. A
deterministic POSIX-semantics test allows lexical-root rename after pin acquisition,
replaces it with a redirect immediately before staging, and reproduced the old external
`releases/` write. It now stages exact bytes only beneath the original pinned directory,
leaves the replacement target empty, and returns typed `invalid_root` when the final
lexical identity check detects substitution. Focused GREEN was 1 passed; the full
affected integration/release-store/snapshot-build/snapshot-store set reported 66 passed
and eight explicit Windows symlink-privilege skips. The real Windows junction tests
passed, scoped Ruff passed, and `git diff --check` passed with only line-ending warnings.
WSL is not installed, so direct Linux execution was unavailable; deterministic simulated
POSIX rename semantics plus fail-closed descriptor-alias selection are the retained
evidence. Missing status/verify exit 4, exit classification, coherent verification, and
source-neutral parsing remain unchanged. Argparse help remains deferred. No network,
server, production/customer data, push, merge, deploy, main-branch, or Streamlit action
occurred; the unrelated `AGENTS.md` edit remains untouched.

Task 7 fix round 3 corrects descriptor-alias containment in `ReleaseStore`. The store
now compares a resolved candidate against the resolved pinned-root identity while
retaining the lexical descriptor-alias path for every returned path and filesystem
operation. A cross-platform regression performs real `stage`, `get`, and `verify`
operations through an alias whose resolved target differs from its spelling: POSIX uses
the live alias yielded by `pin_root`, while Windows uses a privilege-independent junction
witness after replacing the user-facing lexical root. The regression first failed at
the `.locks` containment boundary, then passed with exact artifact bytes under the
original pinned target and no entries under the external replacement. Final affected
verification reported 67 passed and eight explicit Windows symlink-privilege skips;
the junction witness and earlier substitution tests passed. Scoped Ruff passed.
Fail-closed unsupported-POSIX behavior, mid-command substitution protection, source-
neutral parsing, exit classifications, and coherent active verification remain
unchanged. Argparse help remains deferred. No network, server, production/customer
data, push, merge, deploy, main-branch, or Streamlit action occurred; the unrelated
`AGENTS.md` edit remains untouched.

Task 7 fix round 4 corrects the cross-platform alias regression without changing
production behavior. Unsupported POSIX descriptor anchoring must now fail before the
test body with the exact fail-closed message and empty lexical/external roots; supported
POSIX must complete real release `stage`, `get`, and `verify` operations before the
exact post-substitution no-follow recheck error is accepted. Stage/get/verify results
also directly assert descriptor/junction-spelled artifact and manifest paths. A
temporary physical-path-return mutation failed on the new literal assertion, then the
restored implementation passed. Fresh verification reported 1 focused pass, 67 passes
and eight explicit Windows symlink-privilege skips across affected suites, and clean
scoped Ruff. The unrelated `AGENTS.md` change and deferred argparse Minor remain
untouched; no server or external process was started.

Task 8 records the evidence-snapshot foundation checkpoint from exact predecessor
`d79613348f430f529392426036534b26e3a974c8` on
`development/windshield-demand-platform`. Its documentation and security-regression
changes are committed as `db39675` (`docs: document evidence snapshot foundation`).
`README.md` and `docs/DEVELOPMENT.md`
now document the explicit local evidence commands, source-terms review before
acquisition, immutable release/snapshot storage, atomic active-pointer recovery, and
the irreversible implications of deleting local ignored evidence state. They explicitly
state the current limits: no real EEA/KBA/UK parser, forecast, API replacement, or
fixture fallback. The default CLI deliberately returns typed `unsupported_parser` for
the fictional sample build until a future reviewed parser is supplied through application
composition; it must not fall back to fixture data.

The repository-security boundary now has two focused behavior regressions: `git
check-ignore` proves candidate evidence SQLite state is ignored, and the retained
fictional registration CSV is run through the same credential-shape scanner used for
tracked text files and produces no finding. TDD RED was
`uv run pytest tests/test_repository_security.py -v`, which reported 3 passed and 2
expected helper-boundary `NameError` failures. After extracting the scanner and
`git check-ignore` helpers without weakening the existing tracked-file check, the same
focused command reported 5 passed in 0.31s. The documented manifest-validation command
was also run and printed only `sample-registration-2024`.

Fresh Task 8 foundation verification on 2026-08-27:

- `uv lock --check` reported `Resolved 105 packages in 2ms`.
- `uv run ruff check src tests scripts/audit_baseline.py scripts/build_evidence_snapshot.py`
  reported `All checks passed!`. This is the maintained foundation gate; the broad
  `scripts` diagnostic remains excluded because it has 486 pre-existing untouched
  legacy-script findings.
- `uv run pytest -p no:cacheprovider -q` reported `300 passed, 8 skipped, 4 xfailed in
  23.64s`. Every skip was the documented Windows symlink-privilege (`WinError 1314`)
  limitation. The strict legacy XFAILs remain `ICOR-001`, `ICOR-006`, `ICOR-009`, and
  `ICOR-030`.
- `uv run pip-audit` reported no known vulnerabilities and skipped only the unpublished
  local package `icor-windshield-demand`.
- `git diff --check` exited 0 with only existing Windows line-ending warnings. Before
  adding this handoff entry, `git status --short` contained only the intended
  `README.md`, `docs/DEVELOPMENT.md`, and `tests/test_repository_security.py` changes
  plus the unrelated pre-existing `AGENTS.md` modification.

No evidence CLI, Streamlit, deployment, source acquisition, production/customer-data,
push, merge, or main-branch action occurred in Task 8. No new process was started.
Previously documented local processes remain running: Python PID 9732 (planner launcher)
and uv PID 18272 (opportunity-review launcher); the older Streamlit PID 7604 is no
longer running. The next plan is an approved EEA release acquisition/profile/parser and
source-level snapshot implementation, including terms review and parser composition;
it must keep KBA/UK parsing, forecasting, API replacement, and fixture fallback out of
scope until separately planned.

Task 8 review fix round 1 corrects the operator guide in `docs/DEVELOPMENT.md`.
The promotion command now consumes an assigned PowerShell `$candidateSnapshotId` rather
than a shell-invalid angle-bracket literal. The guide now states the precise
source-neutral outcomes: the fictional build returns `unsupported_parser`/exit 2, a
promotion with no candidate rejects/exit 3, and both `status` and `verify` return
`{"active_snapshot_id": null, "state": "unavailable"}`/exit 4 until an active snapshot
exists. Its maintained Ruff command now checks `src`, `tests`,
`scripts/audit_baseline.py`, and `scripts/build_evidence_snapshot.py`. Fresh focused
integration verification reported 3 passed; direct status/verify emitted the documented
unavailable payload and direct missing-candidate promotion emitted the documented
validation-rejected payload. The scoped Ruff gate and `git diff --check` passed. No
process was started or stopped; the review correction remains documentation-only.

The final whole-branch integrity review from exact predecessor
`dcd483d0a2f824ec8e7a1dc3bb1e3bd5b10cba24` is fixed in implementation commit
`ec4a7f7` (`fix: harden evidence snapshot integrity`). Snapshot loaders now consume
private per-build artifact copies that are copied through the no-follow filesystem
boundary, fsynced, checked against the stored release SHA-256 and byte count, paired
with the canonical release manifest, and made read-only before parsing. The original
stored release is still verified again after loading. The regression transiently
substituted values 999/888, read through the loader, and restored the stored artifact;
RED produced a promotable database containing those transient values, while GREEN
contains only the sealed verified values 10/5.

SQLite evidence now permits one identity-mapping row per observation. Repository
writes require mapping vehicle/status attribution to equal the observation and require
every published input to have exactly one publishable mapping that also equals the
published value. Independent read-only candidate validation rejects missing,
duplicate, non-publishable, and contradictory mappings even after raw SQLite
corruption and a recomputed candidate checksum. Repository RED reported five expected
failures; raw-corruption validation RED reported three; and the promotion regression
initially failed with `DID NOT RAISE SnapshotPromotionError`. GREEN reported 57
repository passes and 27 combined validation/promotion passes.

Release staging now uses the shared OS-owned byte/range lock primitive at a stable
per-release lock-file path instead of an ownership directory. A subprocess deliberately
exited with code 73 while holding the old staging lock; RED left an indefinitely stale
directory, while GREEN released lock ownership with process death and allowed the next
stage to complete. Domain constructors now reject every non-finite `Decimal` for
observations, published values, and P10/P50/P90 bounds. The 20-case NaN/sNaN/positive-
infinity/negative-infinity matrix failed RED and passed GREEN.

Fresh final-review verification on 2026-08-27:

- Cross-component evidence/snapshot verification reported `205 passed, 8 skipped in
  26.80s`; every skip was the documented Windows symlink-privilege limitation.
- `uv run pytest -q` reported `331 passed, 8 skipped, 4 xfailed in 44.72s`. The strict
  XFAILs remain `ICOR-001`, `ICOR-006`, `ICOR-009`, and `ICOR-030`.
- `uv run ruff check src tests scripts/audit_baseline.py scripts/build_evidence_snapshot.py`
  reported `All checks passed!`.
- `uv lock --check` reported `Resolved 105 packages in 2ms`.
- `uv run pip-audit` reported no known vulnerabilities and skipped only the unpublished
  local package `icor-windshield-demand`.
- `git diff --check` exited 0 with only existing Windows line-ending warnings.

Direct POSIX lock/permission execution remains a CI responsibility because this host is
Windows; the implementation reuses the existing `fcntl.flock` promotion-lock branch.
An ignored candidate made with an earlier schema fingerprint may still require the
already-documented operator cleanup before rebuilding the same immutable snapshot ID.
Path-based loaders remain trusted in-process application code; private sealed copies
close the mutable release-store path boundary, not deliberate permission reversal by a
hostile loader. No source acquisition, production/customer data, server/process
lifecycle, push, merge, deploy, or main-branch action occurred. The unrelated
`AGENTS.md` modification remains untouched and unstaged.

Final review fix round 2 closes the zero-input publication residual found by the scoped
re-review. The previous mapping checks all began at `published_value_input`; deleting
every join row therefore made an otherwise valid `published_value` invisible to
validation. Repository writes already inherit the domain invariant that `input_ids` is
non-empty, while repository reads reconstruct the domain value and fail on corruption,
but neither gives promotion a deterministic validation finding. Snapshot validation now
starts with a corruption-tolerant query rooted at `published_value`, emits
`snapshot.missing_input` for every value with zero inputs, and then retains the existing
exactly-one-mapping, publishability, and vehicle/status-coherence checks for each input.

The promotion regression builds a valid published candidate, deletes all
`published_value_input` rows through raw SQLite, runs `VACUUM`, recomputes the database
SHA-256, and writes the canonical manifest. RED produced an empty validation report and
failed with `DID NOT RAISE SnapshotPromotionError`; GREEN reported one pass and asserts
both the exact `snapshot.missing_input` finding and that no active pointer is created.
Fresh affected verification reported `117 passed, 3 skipped in 19.40s`, with only the
documented Windows symlink-privilege skips. Fresh full verification reported `332
passed, 8 skipped, 4 xfailed in 32.93s`; the strict XFAILs remain `ICOR-001`,
`ICOR-006`, `ICOR-009`, and `ICOR-030`. Maintained Ruff reported `All checks passed!`,
`uv lock --check` resolved 105 packages in 1 ms, and `uv run pip-audit` found no known
vulnerabilities while skipping only the unpublished local package. No source
acquisition, production/customer data, server/process lifecycle, push, merge, deploy,
or main-branch action occurred. The unrelated `AGENTS.md` modification remains
untouched and unstaged.

Coordinator rulings retained from the completed subagent-driven implementation:

- The canonical serialization example uses SHA-256
  `5ba45a928128f18ed081de659501374802968f3fc00d37ec9158bab5dd210777`, the digest of
  the exact specified UTF-8 bytes. If this ruling is wrong, only the deterministic
  serialization test and dependent hash fixtures require rework.
- `SnapshotBuildRequest.build_as_of` is deterministic and identity-bearing; wall-clock
  time is reserved for the active pointer's `promoted_at`. If this ruling is wrong,
  snapshot identities and manifests require a versioned migration.
- The shipped CLI source registry remains empty until a real approved parser exists;
  the fictional two-row loader is injected only by integration tests. If this ruling
  is wrong, the later EEA implementation must revise the CLI composition API.
- The maintained Ruff gate is scoped to `src`, `tests`, `scripts/audit_baseline.py`,
  and `scripts/build_evidence_snapshot.py`; 486 findings in untouched legacy scripts
  remain diagnostic and out of this foundation's scope. Expanding that ruling would
  require a separate legacy-script remediation slice.
- The fictional-sample safety regression exercises the repository's real credential
  scanner boundary rather than matching source-text keywords. Reverting this ruling
  would affect only the security test, not production behavior.

## 2026-08-27 main integration and local review attempt

Lucas explicitly authorized pushing the current ICOR development version to `main` and
opening it locally for review. The development branch was refreshed against the public
remote and confirmed to be a strict fast-forward: `origin/main` remained
`1ba1d7c41a5fa8354134685b5c85509a0b8f6137`, and it was an ancestor of the development
HEAD with no commits on the remote-only side. The branch was 81 commits ahead after the
test-only stabilization commit `945db54` (`test: stabilize Streamlit startup gate`).

The full Python suite initially exposed a timing-only failure in
`tests/test_app_startup.py`: the real Streamlit AppTest took 8.40 seconds in isolation
but exceeded its fixed 10-second timeout after the rest of the suite. The application
render passed in isolation with unchanged login assertions. The test harness timeout was
raised to 30 seconds; no application behavior changed. Fresh full verification then
reported `332 passed, 8 skipped, 4 xfailed in 172.69s`. The eight skips were the
documented Windows symlink-privilege cases, and the strict XFAILs remained `ICOR-001`,
`ICOR-006`, `ICOR-009`, and `ICOR-030`. `uv lock --check`, maintained Ruff, and
`uv run pip-audit` passed, with pip-audit skipping only the unpublished local package.

Frontend verification reported 47/47 Vitest tests passing; OpenAPI generation had no
drift; TypeScript, ESLint, and the production Vite build passed. A four-worker Playwright
run encountered load-related 30-second timeouts, after which a failed exact-coverage test
left shared test-database state that caused the next fallback test's duplicate locator.
The complete unchanged browser suite was rerun with one worker and reported 13/13
Chromium tests passing, including accessibility, keyboard, responsive, planner recovery,
deep-link, and exact/fallback coverage mutation flows.

The non-force command `git push origin HEAD:main` was attempted with
`GCM_INTERACTIVE=never` and failed before changing the remote because Git Credential
Manager had no usable non-interactive credential. GitHub CLI is not installed. Do not
claim that `main` was updated; the remote remained at `1ba1d7c`. Lucas must re-establish
GitHub authentication before retrying the same fast-forward push. Do not open an account
selector or expose credential material. The pre-existing unrelated working-tree edit to
`AGENTS.md` remains unstaged and was not included in `945db54`.

The verified local planner was started with launcher PID 30512 on
`http://127.0.0.1:5300/opportunities`; its API is at `http://127.0.0.1:8140`.
Both endpoints returned HTTP 200, and `/api/health` returned `status: ok`,
`fixture_ready: true`, and `data_version: demo-planner-v1`. Port 8000 was already owned
by an unrelated existing listener, so the reviewed launcher uses 8140 instead. The
browser open command completed for the opportunity page. Logs are ignored local files
`.local/planner-20260827-094023.stdout.log` and
`.local/planner-20260827-094023.stderr.log`. Clearing Codex context does not stop this
server; stopping PID 30512 or rebooting does.

## 2026-08-27 real-data source validation checkpoint

Lucas requested beginning the replacement of demonstration data with trustworthy free
real data. This is the next approved product milestone after the evidence-snapshot
foundation; no application code, source release, or runtime data was changed during
this initial source-validation checkpoint.

Current official EEA evidence was rechecked on 2026-08-27. The passenger-car CO2
monitoring family now covers 2010-2025. The newest release is 2025 provisional,
published 2026-06-25; the latest finalized release is 2024. EEA metadata says the
dataset contains country-reported newly registered passenger-car records, including
make, commercial name, type approval/type/variant/version, fuel and technical fields.
It is public under CC BY 4.0 with DG CLIMA attribution. EEA also states that provisional
data can still contain inconsistencies and manufacturer corrections, while final data
is published after that review. The initial recommendation is therefore to ingest the
2024 final release as the first source-level snapshot and add 2025 provisional only as
a separately labelled later release. Lucas was asked to confirm whether to follow that
recommendation or include provisional 2025 immediately.

The existing local planner remains active at `http://127.0.0.1:5300/opportunities`
with API port 8140 and still serves `demo-planner-v1`; this investigation did not
restart or stop it. The tracked pre-existing `AGENTS.md` and `docs/CODEX_HANDOFF.md`
working-tree edits remain uncommitted and must be preserved.

## 2026-08-27 official-source ingestion implementation

Lucas authorized completing all recommended free official sources without further
questions. The approved design and implementation plan are in
`docs/superpowers/specs/2026-08-26-real-data-evidence-forecasting-design.md` and
`docs/superpowers/plans/2026-08-27-official-source-ingestion-implementation.md`.
Implementation commits are `b044050` (shared parsing contracts), `8efaccd` (EEA/KBA),
`29a08cf` (UK/production registry), and `26408b8` (pinned acquisition and live fixes).

Four official files were acquired, checksum-verified, and staged below ignored
`.local/evidence`: EEA `co2cars_2024Fv30` final (138,252,239 bytes,
SHA-256 `122dab33e931ea04d3ddb4bb2691dae85dc0da14428fc17873d3fb1f648b7b67`),
KBA FZ10 December 2024 v3 (177,108 bytes,
`856b9afe515d51aa52bcb34d645dce2c5cdeaf47ef398b4e0a754c1bd5813dbf`),
DfT VEH0160 GB (28,355,274 bytes,
`312d09ecabc0f0bcd85d5d2b10ddebf222ba39bb1f833ad60b725708f4f4f06c`),
and DfT VEH0120 GB (65,878,628 bytes,
`3bf96499b09fbb5a9710e1257a2dbc2a8a538190cf823430e2dee23709bb73d3`).
Terms are respectively CC BY 4.0 with EEA/DG CLIMA attribution, DL-DE/BY-2.0 with
KBA attribution, and OGL v3.0 with Crown copyright/DfT attribution.

Live parsing found and fixed two contract gaps with regressions: 628 EEA records lack
an identifiable country/make/commercial-name tuple and are now explicitly rejected
(10,781,686 accepted of 10,782,314), and VEH0120 legitimately contains 1994-1999
quarter columns. Parser scratch databases now use the OS temporary directory because
candidate release copies are intentionally read-only. KBA accepted 417 of 479 physical
rows and reconciled its model-series detail to the published 2,817,331 registration
total. VEH0160 accepted 60,406 Cars rows of 106,148; VEH0120 accepted 77,299
Cars+Licensed rows of 245,043. SORN and non-car rows are excluded, and provisional
columns after 2025 Q4 are not used.

The deterministic build at `2026-08-27T12:00:00Z`, seed `20260827`, produced candidate
`snapshot-a92867b966f81d7966fe`: database SHA-256
`442aa0226156f8c8f62e6e2964bbb590a3395e63ab460a7725ef9f31fd5a4a07`,
542,455 observations, zero warnings, and zero published values. It was not promoted;
no active pointer changed. This is intentional because make/model/model-year identity,
cross-source reconciliation, windshield fitment, estimation, and forecasting remain
future reviewed work. The planner/API still serve `demo-planner-v1`; fresh HTTP checks
returned 200 for ports 5300 and 8140 even though the original launcher PID is no longer
the owning process.

Files added or materially updated include `src/icor/evidence/{normalization.py,
source_records.py,source_registry.py,acquisition.py}`, source adapters under
`src/icor/evidence/sources/`, `scripts/{acquire_official_evidence.py,
build_evidence_snapshot.py}`, their evidence tests, this handoff, README, and the
development guide. Focused verification passed: source adapters 13/13, acquisition 3/3,
and registry/clean-room 12 passed with one Windows symlink-privilege skip. Fresh final
verification reported `360 passed, 8 skipped, 4 xfailed in 89.70s`; all eight skips are
the documented Windows symlink-privilege cases and the four strict XFAILs remain
ICOR-001/006/009/030. `uv lock --check` resolved 105 packages, maintained Ruff passed,
`uv run pip-audit` found no known vulnerabilities while skipping only the unpublished
local package, and `git diff --check` exited zero with line-ending warnings only.

A final read-only ledger reconciliation reported 295,130 EEA observations totalling
10,781,686 registrations; 417 KBA observations totalling 2,817,331; 73,930 VEH0160
observations totalling 55,507,758 historical quarterly registrations; and 172,978
VEH0120 observations totalling 2,491,116,917 historical quarter-end licensed vehicles.
There are 542,404 unresolved and 51 rejected-mapping observations, zero published
values, and no `.local/evidence/active.json`. The database checksum was independently
recomputed and matched the candidate manifest. Exact ignored artifacts created only for
diagnosis, the incomplete EEA download, the stale pre-reconciliation EEA release, and
the failed build directory were removed; the four current staged releases and validated
candidate remain. Preserve the unrelated pre-existing `AGENTS.md` edit. Do not push,
merge, deploy, promote the candidate, or change the protected `main` checkout without
explicit authorization. The long-lived development branch/worktree is intentionally
kept as-is.

## 2026-08-27 source-evidence review workspace

Lucas authorized continuing through all data sources and opening the finished local
app for review. The local/internal review slice is implemented on
`development/windshield-demand-platform` in commits `dc4c15f` (plan), `3536a4a`
(strict read-only candidate service), `f71494e` (typed evidence API), `621063d`
(React workspace), `0128f9a` (operator documentation), and `3dfb8a7` (URL state,
browser journey, and responsive/accessibility fixes). Nothing was pushed, merged,
deployed, promoted, or written to the protected `main` checkout.

The new `/evidence` workspace reviews the exact validated candidate configured by
`ICOR_EVIDENCE_CANDIDATE`. It shows candidate/snapshot provenance, four official
release ledgers and terms links, record counts, raw publisher labels, mapping status,
confidence, row locators, bounded filters, URL-backed search state, and 25-row
pagination. It prominently states that reported labels are not canonical vehicle
identities, the candidate is not active, published values are zero, and the data does
not feed forecasts. The existing planner/opportunity health contract remains
`demo-planner-v1`; canonical identity, reconciliation, estimation, forecasting, and
windshield fitment remain later reviewed milestones.

Candidate validation is fail-closed. The service requires the exact candidate
directory, rejects symlinks/incomplete or non-candidate manifests, verifies directory
identity, database SHA-256, schema, observation/published counts, and release identity,
then opens SQLite with `mode=ro` and `PRAGMA query_only`. Search is bound, treats SQL
wildcards literally, is limited to 100 characters, and page size is capped at 100.
Missing/invalid configuration returns typed `503 evidence_unavailable` without fixture
fallback, filesystem paths, tracebacks, or raw exceptions.

Fresh final verification on the completed tree:

- `uv lock --check` resolved 105 packages; maintained Ruff reported all checks passed.
- Full Python: `371 passed, 8 skipped, 4 xfailed in 54.93s`; the skips are documented
  Windows symlink-privilege cases and the strict XFAILs remain ICOR-001/006/009/030.
- Evidence API regression: 5/5 passed after the import-only Ruff correction.
- Frontend: 53/53 Vitest tests passed; TypeScript, ESLint, production Vite build, and
  regenerated OpenAPI drift check all exited zero.
- Real-candidate Chromium journey: 4/4 passed in 1.3 minutes, covering four releases,
  URL-backed `ALFA ROMEO` search, 390px/1440px reflow, keyboard reachability, and zero
  serious/critical axe findings. Reviewed screenshots are ignored local files
  `.local/review/evidence-mobile.png` and `.local/review/evidence-desktop.png`.
- `git diff --check` exited zero apart from informational Windows line-ending warnings.

A fresh persistent review launcher is running with launcher PID 16408. Its web process
owns `127.0.0.1:5320` as PID 55528 and API process owns `127.0.0.1:8160` as PID 49356.
Live checks returned HTTP 200 for health and `/evidence`; the API reported candidate
`snapshot-a92867b966f81d7966fe`, status `candidate`, 542,455 observations, four
releases, and zero published values. A live `ALFA ROMEO` query returned 4,797 rows with
`ALFA ROMEO` first. Logs are ignored files `.local/evidence-review-20260827.stdout.log`
and `.local/evidence-review-20260827.stderr.log`; stderr contains normal Uvicorn startup
only. Clearing Codex context does not stop the server; stopping launcher PID 16408 (and
its children if necessary) or rebooting does. The older demo instance on ports
5300/8140 was not stopped or changed.

Preserve the pre-existing unrelated modifications to `AGENTS.md` and this handoff.
The long-lived development branch/worktree remains intentionally available for the
next real-data milestone. Do not push, merge, deploy, promote the candidate, or change
the protected checkout without explicit authorization.

Context safety: SAFE TO CLEAR — durable handoff is current.

## 2026-08-28 restart checkpoint: implementation in progress

Lucas asked for a lossless checkpoint before restarting the computer. Work remains
strictly isolated in `C:\Users\LucasCravoVERISSIMO\icor-webapp-development` on branch
`development/windshield-demand-platform`; the protected checkout, production,
deployment, remote, and unrelated Video app processes were not changed. A reboot will
stop the previously documented local ICOR launcher and servers.

The approved implementation plan is committed as `02df582` at
`docs/superpowers/plans/2026-08-27-multiyear-generation-planner-implementation.md`.
The following tested implementation slices are committed:

- `f979eba feat: add official source inventory contracts`
- `6cf83f2 feat: separate vehicle year semantics`
- `424e4dd feat: ingest UK vehicle age evidence`
- `7b7460d feat: define deterministic generation resolution`

The evidence schema is version 3 and now stores registration cohort, manufacture,
and model year independently while reading and migrating older databases. The new
generation domain includes validated registry entries, one selected assignment,
ranked alternatives, method, confidence, training weight, resolver/registry versions,
and stable estimated-generation identities. Resolution precedence is exact identity,
descriptor overlap, unique window, active-month coverage, and the approved newer-
launch tie-break.

Two official finalized DfT/DVLA VEH0124 licensed-stock files are staged immutably:

- `uk-dft-veh0124-am-2025-final-20260429`: 54,874,318 bytes, SHA-256
  `86fe32407fde0a92cb1fd4724e2b586917100d975b4d64dd8c972644418ecc3a`,
  566,977 raw / 173,252 accepted / 393,725 rejected rows.
- `uk-dft-veh0124-nz-2025-final-20260429`: 39,295,208 bytes, SHA-256
  `6a04aebfe77953a4686a633e6081351c1853119488e6738958df738756115984`,
  399,612 raw / 138,302 accepted / 261,310 rejected rows.

The loader preserves `YearFirstUsed` and `YearManufacture` separately. Official
`[x]`, `[z]`, blank, and null year markers become null fields with
`registration_cohort_year_missing`, `manufacture_year_missing`, or
`year_semantics_missing` validation flags; it never invents a year. Both complete raw
files were parsed directly after the fix and returned exactly the manifest counts
above. Focused verification immediately before checkpoint was:

- `python -m pytest tests/evidence/sources/test_uk_dft_age.py tests/domain/test_generations.py tests/generations -q` — 13 passed.
- UK acquisition/registry/parser focused suite — 11 passed.
- generation domain/resolver/estimator focused suite — 11 passed.
- Ruff over all new parser and generation files — all checks passed.
- The last full baseline before these slices was 416 passed, 8 skipped, 4 xfailed;
  the eight skips are Windows symlink privilege cases and the four expected failures
  are ICOR-001/006/009/030. A new full-suite run is still required before completion.

A deterministic six-release snapshot build was started with build time
`2026-08-28T08:00:00+00:00` and seed `20260828`, then cleanly interrupted with Ctrl-C
at Lucas's restart request. The incomplete candidate directory is
`.local/evidence/candidates/.build-de2cb5abc8d543bf8c1ce7b10afd0dbb`; its database
was actively growing (approximately 2.39 GB at the last check). It is not promoted.
The active snapshot remains unchanged and verified by `status` as
`snapshot-2f13ba3f0cd083c7eea8`, SHA-256
`05677e564f10794ae296799fb609ffadbb5b93cfff8b8bd79ae1e327e28df968`,
542,455 observations, using only the four earlier releases. No snapshot build process
is active at checkpoint.

Resume by reading this file and `AGENTS.md`, checking Git status, and confirming the
active snapshot. Preserve the existing unrelated modifications to `AGENTS.md` and
this accumulated handoff. Then rerun the deterministic six-release build using the
four active release IDs plus the two VEH0124 IDs above. The builder creates a new
candidate; do not promote until `verify` passes. After that, continue the plan at EEA
2010 onward acquisition, generation persistence and assignment transformation,
cohort/survival/opportunity computation, snapshot-backed APIs/UI/export, and full
clean-room verification. The current high-level plan has source/year semantics and
the generation contracts substantially implemented; generation persistence,
historical EEA acquisition, cohorts, opportunities, APIs/UI/export, final snapshot
promotion, and full verification remain incomplete.

Context safety: SAFE TO CLEAR — durable handoff is current.

## 2026-08-27 canonical official-registration product

Lucas authorized moving beyond demo-only behavior and asked for the web app to be
built through a finalized real-data slice. The default React/FastAPI route now opens
`/registrations` and serves finalized 2024 EU-27 passenger-car registrations from the
verified EEA release. It ranks exact-normalized make/model families, supports bounded
URL-backed search and pagination, and explicitly states that registration year is not
model year. Windshield fitment, model year, replacement demand, and forecasts are not
inferred. The existing `/planner` and `/opportunities` workflows remain secondary,
clearly labelled prototypes using demonstration forecast data.

Implementation on `development/windshield-demand-platform` is committed through
`5de6c24`. The task commits are `9b4fe7d` (plan), `7d6a544` (nullable canonical model
year), `a03e544` (exact-normalized identity), `d2dd56b` (registration query service),
`a85d175` (typed API), `7161708` (official-data landing page), `1b53f57` (active
snapshot composition), `ecee2e9` (atomic batch replay performance), `a9403ea`
(efficient pre-join aggregation), `74b7ea1` (activation/browser/docs), and `5de6c24`
(review formatting). Nothing was pushed, merged, or deployed, and the protected main
checkout was not changed.

The deterministic official build used the four staged release IDs documented above,
build time `2026-08-27T12:00:00Z`, and seed `20260827`. It produced and locally
promoted `snapshot-2f13ba3f0cd083c7eea8`, database SHA-256
`05677e564f10794ae296799fb609ffadbb5b93cfff8b8bd79ae1e327e28df968`,
542,455 observations, zero warnings, and zero published forecast values. A repeated
build invocation returned the identical snapshot ID and digest. Active status and a
separate full repository verification both reported 542,455 observations.

Independent read-only SQL audit found 10,401 canonical model families, 542,404
exact-normalized mappings, 51 rejected ambiguous observations, zero orphaned mappings,
and source counts of 295,130 EEA, 417 KBA, 172,978 VEH0120, and 73,930 VEH0160.
The product query reports 10,506,946 EU-27 registrations across 6,929 ranked model
families. The first three are DACIA SANDERO (257,883), RENAULT CLIO (211,260), and
PEUGEOT 208 (178,064). The service uses only finalized EEA member-state rows for this
ranking, so overlapping KBA and non-EU EEA rows are excluded.

Fresh final verification on the finished tree:

- Full Python: `406 passed, 8 skipped, 4 xfailed in 65.70s`. The skips are the
  documented Windows symlink-privilege cases; strict legacy XFAILs remain
  ICOR-001/006/009/030.
- Frontend: OpenAPI generation had no drift; all 62 Vitest tests passed; TypeScript,
  ESLint, and the Vite production build exited zero.
- Combined real-data Chromium: 7/7 passed in 3.3 minutes, covering the official
  landing/search journey, evidence search, 390px and 1440px layouts for both pages,
  keyboard reachability, and no serious/critical accessibility findings.
- `uv lock --check` resolved 105 packages, Ruff reported all checks passed, and
  `pip-audit` found no known vulnerabilities (only the unpublished local package was
  skipped). The complete committed-range `git diff --check` review found one extra
  EOF blank line; it was removed in `5de6c24`, and the focused identity suite passed
  8/8 afterward.

A persistent local-only review launcher is running as PID 3316 with web port 5340 and
API port 8180. Fresh HTTP checks returned 200/`ok`; the live API reported the active
snapshot, 10,506,946 registrations, 6,929 families, and DACIA SANDERO at 257,883.
Open `http://127.0.0.1:5340/` for the official product and `/evidence` for provenance.
Ignored logs are `.local/official-review-20260827.stdout.log` and
`.local/official-review-20260827.stderr.log`. Clearing Codex context does not stop the
server; stopping launcher PID 3316 and its children, or rebooting, does. Older review
and demo instances on ports 5320/8160 and 5300/8140 were not changed.

Preserve the pre-existing unrelated `AGENTS.md` modification. This handoff remains
intentionally uncommitted because it already contained pre-existing user/session
changes that were not safe to absorb into a feature commit. Keep the development
branch and worktree in place. Do not push, merge, deploy, or modify the protected main
checkout without explicit authorization.

## 2026-08-27 local official-product restart

Lucas reported that the ICOR page had become unreachable while he was navigating and
asked for the app to be started again. The previously documented listeners on ports
5340 and 8180 were no longer active. The existing official-data application was
restarted without source, data, Git, production, or deployment changes. The launcher
is PID 30064, the Vite web process is PID 19296 on `127.0.0.1:5340`, and the FastAPI
process is PID 35500 on `127.0.0.1:8180`. The launcher uses the active evidence root
and exact candidate `snapshot-2f13ba3f0cd083c7eea8`.

Fresh live verification returned HTTP 200 for `/`, `/evidence`, `/planner`, and
`/opportunities`, and `/api/health` returned `status: ok`. The official registration
summary returned the expected snapshot, 10,506,946 registrations, 6,929 model
families, EU-27 geography, and finalized 2024 release identity. Ignored process logs
are `.local/official-review-restart-20260827.stdout.log` and
`.local/official-review-restart-20260827.stderr.log`. Clearing Codex context does not
stop the app; stopping launcher PID 30064 and its children, or rebooting, does.

Preserve the existing unrelated `AGENTS.md` modification and the accumulated handoff
edits. Do not push, merge, deploy, or modify the protected main checkout without
explicit authorization.

## 2026-08-27 multi-year generation-aware product request

Lucas confirmed that the demonstration data remaining in `/planner` and
`/opportunities` is unacceptable for the intended product. He requested ingestion of
all trustworthy historical years that can be obtained, separation by vehicle year,
mapping model records into generation classes, and replacement of both demonstration
workflows only after the historical and generation data is complete enough for review.
This is an architectural milestone spanning evidence acquisition, year semantics,
canonical identity/generation mapping, forecasts, APIs, and both web workflows.

Initial design investigation reconfirmed that only finalized EEA 2024 feeds the
official registration page and that the planner/opportunity repository remains
`demo-planner-v1`. Public-source semantics require an explicit decision before design:
first-registration year cannot truthfully be renamed model year. UK DfT publishes
`YearFirstUsed` and `YearManufacture` as separate fields, while EU registrations can
retain type-approval/type/variant/version identifiers that are stronger generation
signals than calendar year. Volkswagen's official history identifies Golf VII as
2012-2019 with a transition into Golf VIII, disproving the example 2020-2026 Mk7 range
and illustrating boundary overlap. The recommended rule awaiting Lucas's approval is
to retain first-registration year as a cohort year, use true manufacture/model-year
values only where sourced, map generations through authoritative identifiers and
reviewed evidence with confidence, and leave ambiguous records explicitly unmapped.
No application implementation or source acquisition was started while this semantic
decision remains open. The local app continues to run under the restart recorded
above.

Lucas approved using first-registration year as credible proxy evidence for assigning
as many vehicles as possible to a generation, prioritizing broad machine-learning
coverage. The implementation must preserve the publisher's first-registration year,
record that the generation assignment is inferred rather than manufacturer-confirmed,
and attach provenance and confidence so inferred labels do not silently become ground
truth. The remaining source-policy decision is whether generation mappings may use
reputable licensed non-government sources when official records and manufacturer
archives are incomplete.

Lucas approved the recommended hybrid source hierarchy: government registration and
type-approval evidence first, manufacturer generation archives second, and reputable
licensed open vehicle registries when corroborated. Forums, dealer listings, and
unsupported AI-generated mappings are not accepted as truth. The generation design
must now specify how transition-year observations are represented without forcing
known ambiguity into incorrect hard training labels.

Lucas clarified that the ML dataset requires one concrete generation approximation
per usable vehicle rather than unresolved probabilistic candidate labels. The revised
recommended rule is deterministic: exact type/variant/version or manufacturer evidence
wins; otherwise first-registration year is the primary generation signal. A unique
generation active in that year is high-confidence. Transition years use additional
model/body/type evidence where available, then a documented deterministic market-date
tie-break, with lower confidence if ambiguity remains. The chosen generation, method,
candidate alternatives, evidence, and confidence remain stored so the hard label is
usable for ML without erasing approximation risk.

Lucas approved the historical-data foundation: ingest every validated finalized EEA
annual release available from 2010 onward; retain separately labelled provisional
years; use the already acquired UK 2001-2025 first-registration and 1994-2025 active-
fleet histories; add available KBA years and further licensed European national
sources through isolated adapters; preserve immutable artifacts, checksums, licences,
source semantics, and row provenance; store registration/manufacture/model year as
separate fields; introduce a versioned market-aware generation registry; and promote
only complete validated multi-year snapshots. “All data” means every legally reusable
release that passes validation, not untraceable scraped values.

Lucas approved deterministic generation resolution. Raw publisher labels remain;
exact type-approval/type/variant/version/body/manufacturer identifiers take precedence;
first-registration date is matched to market-specific generation/facelift windows;
overlaps use detailed descriptors, active-month coverage, and then the approved newer-
generation tie-break. Where no sourced window exists, the pipeline creates a stable
chronological estimated generation from registration continuity and structural changes
without inventing an official Mk name. Every usable registration receives one hard
generation ID plus alternatives, evidence, resolver version, confidence, and ML
training weight. Overlapping publications are reconciled to avoid duplicate counts.

Lucas approved replacing `demo-planner-v1` at runtime with one promoted real-data
snapshot and no demo fallback. The planner will expose historical registrations,
cohort-based active-fleet reconstruction, generation evidence, confidence, assumptions,
and horizons. Opportunities will rank real generation-level P10/P50/P90 windshield
replacement opportunity while keeping production readiness separate. Until proprietary
fitment truth is integrated, claims stop at generation/body/facelift level and disclose
that multiple windshield configurations can remain within a generation. The official
registration, evidence, planner, opportunity, and versioned ML-export workflows all use
the same snapshot; every value retains provenance and no LLM-generated value enters
calculation or training data.

Lucas approved the final validation/completion design and the complete architectural
design. The durable specification is committed as `d01c74b` at
`docs/superpowers/specs/2026-08-27-multiyear-generation-planner-design.md`. It defines
the EU-first multi-year source foundation, separate year semantics, deterministic
generation resolution for every usable canonical observation, estimated-generation
fallbacks, confidence/training weights, reconciliation, cohort/fleet reconstruction,
assumption-led generation-level replacement opportunity, snapshot-backed UI/ML export,
failure behavior, verification, exact completion criteria, and delivery sequence. Its
self-review found no placeholders or missing required sections, and `git diff --check`
for the specification exited zero. The Superpowers design workflow now requires Lucas
to review and approve the written specification before an implementation plan is
created. No application behavior, source data, active snapshot, server process, or
production state changed during specification work.

## 2026-08-28 latest-checkpoint pointer

The restart checkpoint headed `2026-08-28 restart checkpoint: implementation in
progress` earlier in this file is the authoritative latest state and supersedes the
older design-stage next-action text immediately above. The specification and plan are
approved, five implementation commits through `7b7460d` are present, the interrupted
six-release candidate is not promoted, and the active snapshot is still
`snapshot-2f13ba3f0cd083c7eea8`. Resume from that checkpoint after reboot. No snapshot
build process is active; reboot will stop any remaining local web/API processes.

## 2026-08-28 second restart checkpoint: generation persistence complete

Lucas requested another restart-safe checkpoint after work resumed. The verified
generation-planning schema and mapping slice is committed as
`c9729f7 feat: extend snapshot for generation planning`. The branch head is now
`c9729f7`; only the pre-existing unrelated `AGENTS.md` change and this accumulated
handoff remain uncommitted.

The evidence ledger is now schema version 4 with a forward-only v3-to-v4 migration.
It immutably persists generation registry entries, exactly one selected assignment
per usable observation, ranked alternatives, cohort estimates and observation
lineage, generation-level opportunity intervals and cohort lineage, and annual
completeness records. Snapshot reproducibility versions now include the generation
registry and resolver. Canonical snapshot replay retains every new derived table.

Promotion validation is fail-closed for generation-enabled candidates: it reports
missing usable-observation assignments, missing generation schema or completeness,
orphan/incompatible lineage, invalid training weights, method-version mismatch,
invalid/reversed generation windows, and invalid cohort/opportunity intervals. The
official build composition now runs a deterministic post-load generation finalizer.
Registration observations use their registration period as the cohort signal; age
evidence uses `YearFirstUsed`, then separately disclosed manufacture year only when
first-use is absent; aggregate stock with no cohort/manufacture semantics remains
evidence-only. Sparse histories receive one broad, explicitly named estimated
generation rather than an invented manufacturer designation, with low confidence and
training weight 0.35. The batch result requires assigned count to equal usable count.

Fresh checkpoint verification:

- Focused domain, resolver, mapping, snapshot-build, schema-migration, source-registry,
  and promotion-validation suite: 132 passed, 1 skipped in 11.49 seconds. The one skip
  is the documented Windows symlink privilege case.
- Ruff over every changed implementation and test file: all checks passed.
- `git diff --check` exited zero before commit.

The active local evidence pointer was not changed and remains
`snapshot-2f13ba3f0cd083c7eea8` with the four earlier official releases, 542,455
observations, and digest
`05677e564f10794ae296799fb609ffadbb5b93cfff8b8bd79ae1e327e28df968`.
No new candidate was built or promoted during this resumed slice and no ICOR build
process is active. Reboot will stop any remaining local web/API processes.

Resume by reading `AGENTS.md` and this file, verifying branch/status/active snapshot,
and continuing with finalized EEA 2010-onward acquisition and annual parser
generalization. Do not rerun the large six-release build yet: the generation-enabled
validator intentionally requires completeness materialization, which is not implemented
until the cohort/opportunity slice. After historical EEA acquisition, implement
reconciliation, cohort survival, seeded P10/P50/P90 opportunity and completeness;
then replace the demo runtime/API/UI/export, run the deterministic build twice, verify,
and only then promote locally. Production, remote, protected checkout, push, merge,
and deployment remain untouched and unauthorized.

Checkpoint correction: adding generation version fields initially made the legacy
eight-field active manifest fail strict decoding and identity recomputation. This was
diagnosed before handoff, fixed test-first, and committed as
`f27399b fix: preserve legacy snapshot compatibility`. Legacy manifests decode with
explicit generation-v0 defaults and may verify against their original identity;
generation-enabled manifests retain the stricter new identity. The focused manifest,
identity, and snapshot-store suite passed 54 tests with 3 Windows symlink-privilege
skips, Ruff passed, and the CLI again confirmed active snapshot
`snapshot-2f13ba3f0cd083c7eea8` with its original digest and 542,455 observations.
The checkpoint branch head is `f27399b`.

## 2026-08-28 latest checkpoint: remote Codespaces pivot approved

Lucas stopped the local full-snapshot path because the working computer should not
retain or construct the remaining multi-gigabyte evidence database. The new approved
direction is an authenticated GitHub Codespaces preview using Lucas's personal GitHub
account, with no new software installed on the working computer. Initial access is for
Lucas and the ICOR manager; later employee access will migrate to a separately designed
OIDC/SSO boundary.

The complete architecture, components/data flow, security, failure/recovery, and
verification design was approved in conversation. The written specification is
committed at
`docs/superpowers/specs/2026-08-28-codespaces-preview-storage-design.md` in commits
`996427f` and formatting correction `dc3053a`. Branch HEAD is `dc3053a` on
`development/windshield-demand-platform`. The specification requires a private
development branch, direct remote acquisition of the 20 checksum-pinned official
releases, a persistent `/workspaces` evidence root, atomic validation/promotion, a
same-origin compiled React/FastAPI service, individually named preview credentials held
only in Codespaces secrets, no demo fallback, and unchanged production/main state.
Per the design workflow, Lucas must review and approve the written specification before
the implementation plan is created. No deployment implementation, remote branch push,
Codespace, GitHub secret, or public port has been created yet.

No GitHub CLI or other software will be installed locally. Existing Git 2.54.0 and Git
Credential Manager successfully performed authenticated read-only access to the private
remote `https://github.com/lucascverissim0/icor-webapp.git`. Remote `main` remains
`1ba1d7c41a5fa8354134685b5c85509a0b8f6137`; the development branch does not yet exist
remotely. Codespace creation will use GitHub's website after the reviewed implementation
is committed and pushed.

Multiple orphaned continuations of the earlier local 20-release command repeatedly
restarted after Lucas requested remote storage. Every process chain was stopped, and
only its exact validated `.build-*` candidate directory was removed. Across the four
cleanups, 6,701,900,080 bytes of incomplete staging data were reclaimed. The final
authoritative process/staging check reported zero snapshot-build processes and zero
`.build-*` directories.

To prevent that orphaned command from consuming more disk, the complete immutable local
release store was preserved by an atomic, reversible rename from
`.local/evidence/releases` to `.local/evidence/releases.local-build-paused`. Do not
delete the paused directory. A deliberately approved future local build would first
verify that `.local/evidence/releases` is absent and then rename the paused directory
back to exactly `.local/evidence/releases`. Remote Codespaces acquisition will download
official sources directly and does not depend on this local directory.

After the pause, the active snapshot CLI verified state `active`, snapshot
`snapshot-2f13ba3f0cd083c7eea8`, database SHA-256
`05677e564f10794ae296799fb609ffadbb5b93cfff8b8bd79ae1e327e28df968`, 542,455
observations, the four earlier official releases, and zero warnings. The active pointer,
promoted snapshot, production checkout, remote `main`, and unrelated `AGENTS.md` change
were not modified.

The implementation work after `a965bc7` remains uncommitted exactly as shown by Git
status: API/application/domain/repository performance and snapshot-runtime changes,
frontend opportunity/schema changes, completeness reporting and integration tests,
README/development documentation, this handoff, plus the pre-existing unrelated
`AGENTS.md`. Preserve these changes. Do not stage `AGENTS.md`.

Resume by reading `AGENTS.md`, this latest checkpoint, and the committed Codespaces
design. Verify branch HEAD, active snapshot, zero local build processes/staging, and the
paused release-store path. Then obtain Lucas's written-spec approval, invoke the
`writing-plans` workflow, and implement the preview test-first. Before any GitHub push,
commit and re-run the full application gates; push only
`development/windshield-demand-platform`. Never merge, push `main`, modify the protected
production checkout, expose an unauthenticated port, or put evidence/secrets in Git.

## 2026-08-28 Codespaces implementation start: product baseline recovered

Lucas approved beginning the authenticated Codespaces preview build. The committed
implementation plan is `2faa6d0` at
`docs/superpowers/plans/2026-08-28-codespaces-preview-storage-implementation.md`.
Before preview-specific implementation, the accumulated generation-aware runtime slice
was recovered and verified on branch `development/windshield-demand-platform`, whose
pre-baseline-commit HEAD was `2faa6d0`.

A performance regression in `GenerationRegistry.candidates()` was reproduced
test-first: the registry rescanned every unrelated generation for every observation.
The focused failing test observed 1,000 unrelated reads. Indexing entries by canonical
vehicle and market reduced a controlled 50,000-entry lookup benchmark from about
2.47 ms to about 0.0023 ms per lookup while preserving resolver behavior. The focused
registry/resolver/mapping suite passed 7 tests and Ruff passed afterward.

Fresh recovered-product verification:

- Focused product suite: 33 passed in 22.83 seconds. The initial sandbox run failed
  before application code because Windows denied pytest's default temp directory; the
  same exact suite passed outside that restriction.
- Frontend: 14 files / 62 Vitest tests passed; TypeScript and ESLint exited zero.
- Ruff over every changed Python implementation and test file: all checks passed.
- `git diff --check` exited zero.
- `.local/evidence/releases.local-build-paused` exists,
  `.local/evidence/releases` does not, and the candidate staging count is zero.

Several local diagnostic launch attempts were stopped after the authoritative
Codespaces checkpoint was found at the end of this handoff. Their exact scheduler
entry, temporary helper files, logs, and build root were removed; no candidate was
published or promoted. The paused immutable release directory, active snapshot,
protected production checkout, remote, and deployment were not changed. Codespaces
implementation now proceeds exclusively from the approved remote-storage plan.

## 2026-08-28 Codespaces checkpoint: preview configuration boundary

The first preview-specific TDD slice is complete. `src/icor/preview/config.py` now
strictly decodes individually named Argon2id password verifiers, a base64url signing
secret of at least 32 decoded bytes, and a bounded 300-43,200 second session lifetime.
It rejects absent, malformed, weak, duplicate-casefolded, and plaintext configuration
without echoing submitted values. `argon2-cffi` 25.1.0 is a direct locked dependency;
`argon2-cffi-bindings` 26.1.0 is locked transitively.

The test was observed RED with `ModuleNotFoundError: No module named 'icor.preview'`
before implementation. Fresh GREEN verification is 14 passed in 0.16 seconds, and
Ruff reports all checks passed for the configuration implementation and tests. The
generation runtime checkpoint remains `fc723af`; `AGENTS.md` remains the only unrelated
unstaged user change. No GitHub branch, Codespace, secret, port, production checkout,
active snapshot, or paused local release store was changed.

Resume with Task 3 of
`docs/superpowers/plans/2026-08-28-codespaces-preview-storage-implementation.md`:
signed sessions, Argon2id credential checks, bounded login throttling, authentication
middleware, and security headers, continuing strict RED-GREEN cycles.

## 2026-08-28 Codespaces checkpoint: sessions and authentication middleware

Task 3 is complete test-first. `src/icor/preview/auth.py` issues canonical compact
HMAC-SHA256 sessions with a 128-bit nonce and strict issue/expiry bounds, rejects
malformed, tampered, expired, and key-rotated tokens, verifies passwords with Argon2id,
and uses a dummy verifier for unknown accounts. Its login throttle retains only keyed
digests of normalized account/address pairs, enforces five failures per 15 minutes,
resets after success, expires stale attempts, and caps retained buckets.

`src/icor/preview/security.py` allows anonymous access only to `/healthz` and
`/auth/login`, protects application/assets/API/docs/export paths, attaches the verified
username to request state, and adds CSP, nosniff, no-referrer, and deny-framing headers.
Authentication failures and auth routes are non-cacheable. Fresh verification is 17
passed in 1.92 seconds with Ruff fully clean. The only initial integration failures were
the repository's intentional socket guard blocking TestClient's Windows loopback
socketpair; adding the same explicit loopback-only marker used by existing API tests
made the middleware tests exercise production code without broadening network access.

No GitHub, Codespace, port, secret, evidence, production, `main`, active-snapshot, or
paused-release state changed. `AGENTS.md` remains the only unrelated unstaged user
change. Resume with Task 4: the login/logout routes, fail-closed preview factory, and
safe same-origin SPA/static resolver.

## 2026-08-28 Codespaces checkpoint: authenticated same-origin preview

Task 4 is complete test-first. `icor.preview.app:create_preview_app` is a separate
preview composition over the existing FastAPI product routes. Startup fails closed
when preview security configuration, the active generation snapshot, or
`web/dist/index.html` is unavailable. `/healthz` is the sole data-free lifecycle
response; login uses a size-limited form, generic Argon2id rejection, and the bounded
throttle. Success issues a short-lived `Secure`, `HttpOnly`, `SameSite=Strict` cookie,
and POST logout clears the browser session.

`icor.preview.static.resolve_asset` rejects dot traversal, backslashes, percent-encoded
ambiguity, NULs, missing files, and resolved symlink escape. Existing API routes win
before the SPA catch-all; unknown `/api/*` paths remain JSON 404s. Compiled hashed
assets receive their proper content types, while extensionless application navigation
falls back to the compiled index on the same origin.

The module-absence RED run produced 14 expected failures and one already-passing
environment-isolation assertion. Fresh combined preview/local-API verification is 46
passed in 12.12 seconds with one documented Windows symlink-privilege skip. Ruff and
`git diff --check` pass. No remote, Codespace, port, secret, evidence, production,
`main`, active-snapshot, or paused-release state changed; `AGENTS.md` remains excluded.
Resume with Task 5, the idempotent 20-release Codespaces bootstrap.

Context safety: SAFE TO CLEAR — durable handoff is current.

## 2026-08-28 Codespaces checkpoint: bootstrap, runner, and operator runbook

Tasks 5-7 are complete test-first. The bootstrap owns an exact 20-release plan,
validates Codespaces/Python 3.12/Node/npm/uv 0.11.3/lockfile/disk prerequisites,
verifies and reuses staged releases, and invokes only repository public CLIs with
argument arrays. It builds with `2026-08-27T12:00:00+00:00` and seed `20260827`,
requires the exact completeness identity before promotion, preserves the active
pointer on failure, and compiles the locked React client. Focused bootstrap plus
official-source verification is 19 passed with Ruff clean.

The explicit runner rejects non-Codespaces execution, invalid preview configuration,
missing active state, missing `web/dist`, missing coverage state, and weak/missing
export authorization before binding the authenticated preview factory. The local
runner remains loopback-only. The devcontainer no longer auto-starts Streamlit or any
server; port 8000 is only silently forwarded, carries no public-visibility directive,
and is labeled `ICOR authenticated preview`. Runner/toolchain/repository-security
verification is 14 passed with Ruff clean.

The interactive credential helper has no plaintext password argument, performs hidden
double entry, emits Argon2id verifiers, and generates independent 32-byte base64url
session keys. Its focused suite is 4 passed with Ruff clean. README, development, and
deployment documentation now cover browser-created Codespaces, the three secret
names, private bootstrap and smoke tests, temporary public sharing, shutdown,
identity capture, retention, rebuild, and deletion recovery. The one secret-helper
smoke value printed during local verification was disposable, was not stored, and is
not configured anywhere.

No Codespace, GitHub secret, forwarded/public port, evidence acquisition, candidate,
promotion, production checkout, `main`, active snapshot, or paused local release store
changed. `AGENTS.md` remains the only unrelated unstaged user change. Next run Task 8:
complete local backend/frontend/security gates, authenticated fixture smoke tests,
protected-state proof, final handoff checkpoint, and development-branch push review.

## 2026-08-28 Codespaces checkpoint: complete local verification

Task 8 is complete. Integrity checks found no tracked secret, local evidence/database,
compiled frontend, private key, or unrelated staged file. The protected production
checkout remains clean at `1ba1d7c41a5fa8354134685b5c85509a0b8f6137`, exactly
matching `origin/main`; `AGENTS.md` remains the only unrelated unstaged user change.

Fresh backend gates passed: lockfile check, Ruff, 544 tests, and `pip-audit`. Pytest
reported 11 documented Windows-symlink or unconfigured-real-snapshot skips and four
documented characterization XFAILs. No known third-party vulnerability was found;
only the unpublished local package was unauditable. Focused preview/security/toolchain
verification additionally passed 77 tests with one Windows symlink skip.

Fresh frontend gates passed: OpenAPI drift, 62 Vitest tests, TypeScript, ESLint, Vite
production build, and all 20 Chromium scenarios. The bundle contains no secret value or
verifier, absolute local path, demo-repository reference, or hard-coded local API origin;
the literal `ICOR_EXPORT_TOKEN` UI instruction is intentionally not a credential.
Playwright used sealed ignored official evidence candidate
`snapshot-2f13ba3f0cd083c7eea8` and generation candidate
`snapshot-a48d61af9e4307b42b7b`. The first browser run overlapped a queued fixture update
and loaded stale assertions; the stable full rerun passed. Commit `cd38cbe` records the
fixture composition and coverage UI integration.

The live authenticated loopback smoke passed with two disposable named users:
anonymous health, denial for six protected route classes, same-origin app/API/official
data access, logout, tampered and expired cookie rejection, and safe shutdown. No
disposable credential or session key was stored.

The local active pointer is still `snapshot-2f13ba3f0cd083c7eea8`, pointer SHA-256
`A02B9BCEB32B7C88BDD47F571636921BFA5531B0CE9B64A10BE2EC5F43E5AA05`.
The paused-release marker remains present and `.local/evidence/releases` remains absent.
No active evidence, GitHub, Codespace, public port, production checkout, or `main` state
changed.

A disposable local browser composition is running at `http://127.0.0.1:5173/` under
hidden local runner PID `25576` with API port 8000. It combines official sealed
registration/evidence data with explicit computed browser fixtures; it is not the final
remote 20-release build. Clearing conversation context does not stop it. The detached
runner can outlive this terminal; shutting down the computer stops it. To stop it
manually, terminate PID `25576` and its child process tree.

Next is Task 9: inspect and, only with explicit authorization, push the development
branch. The user must create the private Codespace from that branch, add the three
documented secrets outside Git/logs/conversation, run the private 20-release bootstrap
and full gates, and only then temporarily make port 8000 public for review. Do not push
or merge `main`, publish a port, or start the remote acquisition without those actions.

## 2026-08-29 release checkpoint: clean-room browser gate repaired

Lucas authorized completing the previously documented Task 9 sequence: push only
`development/windshield-demand-platform`, create the Codespace from that branch,
configure the three documented Codespaces secrets, run the exact 20-release remote
build and complete verification, verify authentication privately, temporarily expose
port 8000 for manager review, and return it to private. This authorization does not
permit a merge or push to `main` or a production deployment.

The outgoing audit confirmed the local development worktree is still isolated at
`45e9eca5732b480eda1a1fa40758305e019a2507`. Both the protected production checkout
and remote `main` remain at `1ba1d7c41a5fa8354134685b5c85509a0b8f6137`. The remote
development branch was absent at the time of the audit. The paused local release store
remains present, `.local/evidence/releases` remains absent, and no abandoned
`.build-*` candidate exists. The unrelated unstaged `AGENTS.md` edit is preserved and
must remain excluded from commits.

Fresh pre-push gates first passed lockfile verification, the maintained Ruff scope,
544 backend tests with 11 documented skips and four characterization XFAILs, 62 Vitest
tests, OpenAPI drift, TypeScript, ESLint, the production bundle, and Python/npm audits
with no known third-party vulnerability. The first bare Playwright run then correctly
exposed a release-blocking clean-room defect: CI supplied neither required sealed
candidate path, so its API composition failed closed before tests ran. Supplying the
two recorded ignored candidates proved the configuration diagnosis, but a four-worker
large-candidate run reproduced resource/process failure; the same evidence file passed
4/4 serially.

The repair is test-only and does not change application, preview, Codespaces, or
production runtime behavior. `scripts/e2e_fixture.py` builds a tiny deterministic
sealed EEA candidate through production release staging, snapshot validation, and
canonical identity boundaries when explicit candidates are absent.
`scripts/run_e2e_dev.py` injects that candidate only into the browser-test composition,
rejects partial explicit configuration, preserves two explicit paths unchanged, and
uses a port-scoped ignored fixture root for concurrent runs. Explicit real-candidate
runs retain their exact 542,455-observation, four-release, 6,929-model, and official
registration assertions and automatically use one worker. CI Ruff coverage now includes
the harness and all Codespaces security scripts touched by the release plan.

TDD evidence recorded the missing module and missing runner environment as RED, then
focused GREEN at 2/2. Final changed-tree verification is: expanded Ruff clean; 546
backend tests passed with the same 11 skips and four XFAILs; 62 Vitest tests passed;
OpenAPI, TypeScript, ESLint, Vite production build, lockfile, `pip-audit`, and
`npm audit --audit-level=high` passed; bare clean-room Playwright passed 20/20 in
40.6 seconds after review hardening; and the explicit large sealed-candidate suite
passed 20/20 serially in 3.3 minutes. Independent read-only review found no Critical or
Important issue; both Minor suggestions (concurrent fixture isolation and the complete
environment matrix) were implemented and reverified.

No test server is running. No GitHub branch, Codespace, secret, port, evidence release,
snapshot promotion, deployment, production checkout, or `main` state changed during
this checkpoint. Next: commit only the intended harness/CI/handoff files, re-audit the
commit, push only `development/windshield-demand-platform`, then perform the documented
Codespaces secret/bootstrap/private-auth/public-review/private-port sequence.


Context safety: SAFE TO CLEAR — durable handoff is current.

## 2026-08-30 private Codespaces preview: acquisition checkpoint

Lucas explicitly authorized pushing only `development/windshield-demand-platform`,
creating a private Codespace, configuring the three documented Codespaces secrets,
running the exact 20-release build and full gates, verifying private authentication,
temporarily exposing port 8000 for manager review, and returning it to private. This
does not authorize a merge/push to `main` or a production deployment. Lucas asked to
defer choosing personal passwords and use generated temporary credentials meanwhile.
Strong generated credentials for the named `Lucas` and `manager` preview accounts are
stored only in Windows Credential Manager targets `ICOR-Preview-Lucas` and
`ICOR-Preview-Manager`; no password was shown or placed in Git, logs, or documentation.
The user-level repository-scoped Codespaces secrets `ICOR_PREVIEW_USERS`,
`ICOR_PREVIEW_SESSION_SECRET`, and `ICOR_EXPORT_TOKEN` are configured. The original
Git Credential Manager token was preserved, and the separate GitHub CLI token remains
in the system keyring.

The portable official GitHub CLI 2.98.0 is installed at
`C:\Users\LucasCravoVERISSIMO\tools\gh-2.98.0\bin\gh.exe`; its Windows amd64 ZIP
matched the official SHA-256
`C28C7B3B584967A05B74D9EAF7481BFF24DDC34930BF2D6E442C148236561EB1`.
The definitive private Codespace is
`icor-windshield-preview-final-pjqjxgg6qrx4f9r94` (`standardLinux32gb`, West Europe,
30-minute idle timeout, 168-hour retention). Failed predecessor Codespaces were
deleted. The definitive Codespace is on branch
`development/windshield-demand-platform`; port 8000 is currently **private**, and no
manager-facing exposure has occurred.

Four reviewed provisioning fixes were committed and pushed only to the development
branch: `1be37ec` pins the Codespaces SSH Feature and lock digest; `578fca2` enforces
canonical SSH-before-Node Feature ordering; `25a4121` pins the devcontainer base image
and removes its stale Yarn APT source; and `d9f95fe` normalizes the platform suffix in
`uv --version`. The Codespace then built successfully with Python 3.12.11, Node
24.15.0, npm 11.12.1, and uv 0.11.3. Its only unrelated worktree difference is the
known final-newline change in `.devcontainer/devcontainer-lock.json` made by
Codespaces. The local unrelated `AGENTS.md` edit remains unstaged and excluded.

The first live `--prepare` attempt exposed two coordinator defects: historical EEA
sources were incorrectly sent to the generic direct downloader, and the CLI received
the releases directory instead of the evidence root. Commit `ff84287` fixes both
test-first: EEA 2010-2023 uses `acquire_eea_history.py` followed by checksum-gated
`acquire_official_evidence.py --artifact`; every acquisition gets the correct evidence
root; all-valid releases remain no-op; and the downloads root is resolved and rejected
if it is a file, symlink, unavailable, or outside the evidence tree. Independent review
found the symlink boundary before commit; the regression was observed RED, then the
complete bootstrap suite passed 17 tests with one Windows symlink-privilege skip.
Focused acquisition/parser verification passed 28 tests, Ruff and `git diff --check`
passed, and the fresh final backend suite passed 549 tests with 12 documented skips and
four known characterization XFAILs. A prior full-suite attempt had one Windows
`MoveFileW` access-denied fixture error; the exact test passed immediately in isolation
and the full rerun passed. Independent re-review found no Critical or Important issue.
The verified application-code checkpoint and the Codespace are at
`ff842875a2daa054d1d9c4238e69308a0c2cbdf0`; after this checkpoint is committed, the
remote development branch is one documentation-only commit ahead. Protected remote
`main` remains unchanged at `1ba1d7c41a5fa8354134685b5c85509a0b8f6137`.

The corrected bootstrap environment check returns
`{"release_count":20,"state":"ready"}`, but live acquisition is not yet complete.
The reviewed 2010 adapter reproduced every pinned aggregate count (162,167 groups,
285,764 source rows, 282,966 accepted rows, 2,798 rejected rows, and 12,939,010
registrations), while its current artifact was 10,582,166 bytes with SHA-256
`d7eed251b30a3cc8d14ad9106e30c2655938b277192f88d051f8ce34538dba07`; the pinned
artifact was 10,582,165 bytes with a different digest. Row-level comparison against the
preserved validated local release found exactly 128 changed rows. The only changes were
case differences in `Mk`, `Cn`, `T`, `Va`, or `Ve`, plus one trailing space in `Mk`;
all row ordering, counts, and nonpresentation values were unchanged. This proves the
official SQL endpoint's case/trailing-space-insensitive grouping returns unstable
representative text. The checksum gate correctly rejected it; no release was staged,
no candidate or active snapshot was created or changed, and the exact remote download
fragment plus local temporary comparison copy were removed after validation.

This newly discovered issue upgrades the next step to a bounded data-contract design
requiring Lucas's explicit approval under the brainstorming workflow. Recommended
design: canonicalize every historical EEA grouping label deterministically using the
same NFC/whitespace/casefold identity rule before CSV serialization; issue new immutable
`-r1` release IDs for all 14 historical releases rather than reusing IDs for different
bytes; keep the parser backward-compatible with the preserved old IDs; derive and pin
all 14 new sizes/digests; verify the current and preserved 2010 exports canonicalize to
identical bytes; then rerun local gates, independent review, development-only push,
remote acquisition, build/completeness/promotion, frontend/browser/security/audit
gates, private authentication, temporary manager exposure, and immediate return to
private. Do not weaken checksum validation or stage the nondeterministic artifacts.

No application server is running in the Codespace. Port 8000 has a GitHub forwarded
URL but is private. The next action is to obtain Lucas's explicit approval for the
bounded canonicalization/new-release-revision design before any implementation.
## 2026-08-31 canonical EEA releases and direct-source redirect checkpoint

Lucas explicitly approved the bounded historical EEA canonicalization/new-release
revision design and reiterated that Codex must work only in the ICOR web app even after
terminal or conversation context is cleared. This conversation remains anchored to
`C:\Users\LucasCravoVERISSIMO\icor-webapp-development`; the unrelated unstaged
`AGENTS.md` edit remains preserved and excluded. Protected remote `main` remains
`1ba1d7c41a5fa8354134685b5c85509a0b8f6137`, and no production deployment or merge
was performed.

Commit `e666c433317fd420753ef560c7b0e51709c5d5cf` was pushed only to
`development/windshield-demand-platform`. Historical EEA 2010-2023 group labels are now
serialized deterministically: `MS` uses NFC, collapsed whitespace, and uppercase ISO
country-code casing; `Mk`, `Cn`, `TAN`, `T`, `Va`, `Ve`, and `Ft` use NFC, collapsed
whitespace, and casefolding. All 14 releases have new immutable `-r1` IDs and exact
canonical sizes/SHA-256 pins. The loader remains compatible with both preserved legacy
IDs and new `-r1` IDs and defensively uppercases geography before aggregation.

The initial independent review found that casefolding `MS` would break the application's
uppercase EU27 geography filters. That finding was verified against
`src/icor/application/registrations.py`, reproduced RED at both acquisition and loader
boundaries, and corrected test-first. Independent re-review found no remaining Critical
or Important issue. All 14 canonical artifacts derived from the preserved verified
release store passed the real `build_manifest` checksum/size/identity gate. A live 2010
export with independently varying raw presentation canonicalized to the same bytes as
the preserved release. Focused local verification passed 32 tests with one documented
Windows symlink skip; the private Linux Codespace passed all 33 focused tests with no
skip. Local lockfile and full Ruff gates passed, and the fresh complete backend suite
passed 551 tests with 12 documented skips and four known characterization XFAILs.
Temporary local derivation artifacts and diagnostic copies were removed.

The definitive private Codespace
`icor-windshield-preview-final-pjqjxgg6qrx4f9r94` is at application commit `e666c43`
with only its known devcontainer-lock final-newline difference. Its environment check
returns `{"release_count":20,"state":"ready"}`. Real acquisition immutably staged all
14 canonical EEA history releases (2010-2023) before stopping at EEA 2024. No candidate
build, promotion, or active snapshot change occurred.

EEA 2024 failed only because the strict URL allowlist rejected the official endpoint's
new HTTP 302 target. The public landing URL now redirects to the exact versioned
EEA-managed object
`https://dis2datalake.blob.core.windows.net/discodata/co2emission/v7r2/co2cars_2024fv30.zip`.
The blob reports 138,252,239 bytes, last-modified 2026-08-07 08:49:05 UTC, and its
verified SHA-256 remains the existing pinned
`122dab33e931ea04d3ddb4bb2691dae85dc0da14428fc17873d3fb1f648b7b67`.
The verified archive is retained only as an unstaged acquisition download in the
private Codespace. KBA and all four UK direct sources return HTTP 200 at their existing
exact allowlisted URLs with no redirect.

The next bounded design requires Lucas's explicit approval under the brainstorming
workflow: replace only EEA 2024's download URL with the exact versioned EEA-managed blob
while keeping its release ID, size, checksum, counts, terms, parser, and all strict URL
validation unchanged. Add a failing exact-URL regression test, implement the metadata
change, rerun focused/full gates and independent review, push only the development
branch, then resume acquisition using the already verified archive. Port 8000 remains
private and no application server is running.

## 2026-08-31 EEA 2024 versioned-source implementation checkpoint

Lucas explicitly approved the bounded list from the preceding checkpoint and asked
Codex to keep all durable working context in the GitHub repository so work can resume
after conversation or terminal context is cleared. The mandatory startup and handoff
rules in AGENTS.md already enforce that policy; the separate pre-existing unstaged
AGENTS.md productivity edit remains preserved and excluded.

The approved EEA 2024 source change is implemented locally on
development/windshield-demand-platform. Only the source URL now points to the exact
versioned EEA-managed blob documented above. The immutable release ID, byte size,
SHA-256, source counts, publication time, licence terms, parser, schema, suffix, and
strict exact-URL validation remain unchanged. The regression also pins those invariant
identity fields and validates the exact new URL.

TDD RED was observed as one focused assertion failure showing the obsolete moving EEA
landing URL versus the versioned blob; the other four acquisition tests passed. Focused
GREEN is 5/5. Fresh complete verification is: uv lock --check resolved 107 packages;
the complete maintained Ruff gate passed; backend pytest reported 552 passed, 12
documented skips, and four known characterization XFAILs in 103.26 seconds; and
git diff --check exited zero with informational Windows line-ending warnings only.
Independent read-only review found no Critical, Important, or Minor issue and confirmed
the immutable contract and fail-closed redirect validation remain intact.

Local commit c2fb8e7917fbd43e289ca9afec2979d178206c04 contains only the EEA
metadata, regression, and this handoff. Its post-commit diff/check audit passed, and the
unrelated AGENTS.md edit remains unstaged. The attempted push was rejected before Git
executed because this public repository's tracked handoff contains detailed operational
information such as the Codespace identity, local paths, credential-store target names,
and preview workflow metadata. No secret values are present, and much of the handoff is
already tracked remotely, but an explicit user approval for this exact public payload
is required before retrying. Before this handoff-only checkpoint commit, local HEAD is
one commit ahead of remote development HEAD
09d8253b410d871587e420d365810343b112e6a3; after it, the branch will be two commits
ahead.

The protected main checkout, production, active snapshot, Codespaces releases,
candidate/promotion state, port visibility, and secrets are unchanged. Port 8000
remains private and no application server is running. Next: obtain explicit approval
to push the two local commits, including the detailed tracked handoff, to the public
lucascverissim0/icor-webapp development branch; then update the private Codespace,
resume acquisition from EEA 2024 using the already verified archive, and continue the
documented build, completeness, promotion, application, browser, security, audit,
private-authentication, temporary manager-review, and immediate return-to-private
sequence.

## 2026-08-31 portable all-release build checkpoint

Lucas explicitly approved publishing the two preceding local commits, including this
detailed tracked handoff, and stated that all important state must be committed and
pushed so development can resume from any machine. Treat a material checkpoint as
portable only after its handoff is committed and pushed to the development branch; do
not report that it is safe to clear or switch machines while newer material state
exists only in a terminal or Codespace process.

The approved development push succeeded. Remote
`development/windshield-demand-platform` reached
`3b0d81a9dc5665d21bcf521bfc81011fe4bb7f0f`; protected `main` remained unchanged.
Commit `c2fb8e7` pins the exact EEA 2024 versioned source and its regression, and commit
`3b0d81a` records the prior push boundary. The local branch was synchronized except for
the preserved unrelated unstaged `AGENTS.md` edit.

The definitive private Codespace was fast-forwarded to `3b0d81a`. The retained EEA
2024 archive was reverified at 138,252,239 bytes and staged as
`eea-co2cars-2024-final-v30-r1`. Acquisition then completed for all 20 pinned releases,
and a repeated acquisition was a 20-manifest no-op. No active snapshot exists; status
correctly returned `{"active_snapshot_id":null,"state":"unavailable"}` with exit 4.
Port 8000 remains private and no application server is running.

The first complete build was silently stopped when the Codespace hit its 30-minute idle
timeout: GitHub defines idle as absence of user-indicative activity, and silent terminal
sessions do not reset the timeout. The intact `.build-*` tree proved abrupt container
termination rather than a normal Python exception, because `SnapshotBuilder.build`
always removes staging in `finally`. The exact abandoned scratch path was verified to
be beneath `/workspaces/.icor/evidence/candidates`, confirmed to have no live worker or
active pointer, and permanently removed with the project's containment-checked
`SnapshotFilesystem.cleanup_directory`; all 20 release manifests remained intact.

An exact-command retry used harmless terminal heartbeats every 120 seconds and stayed
alive for more than three hours, proving the idle mitigation. It then exposed a separate
capacity limit in the 32 GB workspace. During canonical replay the builder must retain
the 12 GB scratch database while writing an approximately 11 GB final WAL and then
checkpointing that WAL into the main database. Workspace use reached 100% (28 KB free)
before the checkpoint could finish. A graceful SIGINT could not be delivered while
SQLite was inside its C checkpoint. The single capacity-doomed worker was therefore
terminated, and only its exact verified staging directory
`.build-463d04c93ecc4b6db1b2aef43d65caed` was permanently removed with
`SnapshotFilesystem.cleanup_directory`. Workspace capacity returned to 26 GB free;
the release store, repository, active pointer, and port state were untouched.

The Codespace exposes `/tmp` as a separate 118 GB filesystem with 108 GB free. A unique
retry root `/tmp/icor-evidence-build-20260831T1520Z` now contains a verified copy of all
20 immutable release manifests/artifacts (685 MB). The initial preflight correctly
refused before building because its manifest-count command assumed the wrong directory
depth; a recursive verification then proved exactly 20 manifests. No candidate was
created by that refused preflight.

The deterministic large-volume retry is currently running in the private Codespace as
PID 43316 under a 120-second terminal heartbeat. Its command is:

`uv run python scripts/bootstrap_codespaces_preview.py --build --workspaces-root / --evidence-root /tmp/icor-evidence-build-20260831T1520Z`

Its first heartbeat was `2026-08-31T15:22:38+00:00`. This build performs candidate
validation but no promotion. If the terminal session is lost, first inspect PID 43316,
the exact temporary root, and `/tmp` capacity; do not start a duplicate worker while it
is alive. After success, run the independent completeness report against the temporary
candidate and require zero warnings plus the exact 20 release IDs. Then import only the
finished immutable candidate into `/workspaces/.icor/evidence/candidates` through a
containment-checked staging/publish operation, rerun candidate validation there, verify
active state is still unavailable, and promote only that verified candidate. Continue
with active status/verification, idempotent `--prepare`, backend/frontend/security/audit
gates, private authentication, and private browser review. Keep port 8000 private until
all private gates pass. Record and push every material outcome before reporting the
workspace portable.

## 2026-09-01 promoted evidence, durable bootstrap, and authenticated preview checkpoint

Lucas repeatedly approved continuing the bounded build and reiterated that every
material state needed to resume from another machine must be committed and pushed to
GitHub. Work remained on `development/windshield-demand-platform`; protected
`main` and production were not changed. The unrelated local `AGENTS.md` productivity
edit remains unstaged and excluded.

The capacity-safe evidence workflow completed after the preceding temporary-root build.
Commit `692ab0c` records the portable evidence-build checkpoint. The exact active
snapshot is `snapshot-b238373eba183733bf60` with database SHA-256
`856415af05386ff0fb39109eacfabc135ee7a3ca950776e1aea578c008879c26`,
1,529,210 observations, zero published values, zero warnings, and the exact 20 pinned
releases: canonical EEA CO2 cars 2010-2023 `-r1`, EEA 2024 v30, KBA FZ10 2024-12,
and the four pinned UK DfT 2025 releases. Candidate validation, completeness, import
into the persistent evidence root, promotion, active status, and full repository
verification all passed.

Commit `e4f8861` makes `--prepare` portable on a low-capacity machine without
weakening validation. It reuses the active snapshot only when all 20 release
manifests/artifacts validate, the active snapshot exactly matches the intended release
plan, and production matching returns the actual snapshot ID. Capacity is reprobed
before every non-reuse path. TDD and independent review passed. The real private
Codespace had about 5.2 GB free and returned:
`{"release_count":20,"reused":true,"snapshot_id":"snapshot-b238373eba183733bf60","start_command":"python scripts/run_codespaces_preview.py","state":"prepared"}`.

The first post-bootstrap Linux integration run exposed a POSIX durability defect.
After active-pointer replacement, `SnapshotFilesystem` attempted to reopen its pinned
`/proc/self/fd/<n>` root with `O_NOFOLLOW|O_DIRECTORY`, raising
`NotADirectoryError` even though promotion had succeeded. Commit `f49521d` retains
the verified pinned root handle and fsyncs that exact handle directly while preserving
nofollow directory opens elsewhere. Independent review found and drove a nested
`pin_root` reentrancy regression; the final implementation saves/restores prior pin
state. Both failing Linux promotion integrations and both POSIX unit regressions pass.
Fresh local verification at that checkpoint was 557 passed, 14 skipped, four XFAILs;
the private Linux Codespace reported 567 passed, four skipped, four XFAILs. Lock,
maintained Ruff, and `pip-audit` passed locally and remotely.

Frontend post-promotion gates passed in the private Codespace: OpenAPI compatibility,
62 Vitest tests, TypeScript, ESLint, the production Vite build, and npm high-severity
audit with zero vulnerabilities. The first Playwright run failed before assertions
because the pinned Chromium revision was absent. After the exact
`npx playwright install chromium`, the second failed because Linux browser libraries
such as `libatk-1.0.so.0` were absent. `npx playwright install-deps chromium`
completed successfully. The final browser run passed all 20 assertions in 3.0 minutes.
The harness teardown initially waited on service children left by the two environmental
failures; only the exact verified Vite/uvicorn test-service PIDs were terminated, after
which Playwright finalized cleanly. No Playwright, Vite, or test uvicorn process
remained. These browser binaries/libraries persist in the current Codespace but must be
installed again in a replacement Codespace.

The persistent production-coverage database did not yet exist. It was initialized
through the supported `SQLiteCoverageRepository` adapter at
`/workspaces/.icor/production-coverage.sqlite3`; schema version 1 was asserted and
the initial coverage list was empty. This ignored runtime file is intentionally not in
Git and exists only in the persistent private Codespace workspace. After initialization,
`uv run python scripts/run_codespaces_preview.py --check` returned
`{"state":"ready"}`.

Actual preview startup then exposed a separate configuration bug: the runner correctly
set `ICOR_EVIDENCE_ACTIVE_ROOT=/workspaces/.icor/evidence`, but
`create_preview_app` always supplied the repository-local default to the core factory,
overriding the environment. Commit `51dd3a6` resolves roots explicitly in this order:
a supplied `snapshot_root`, then `ICOR_EVIDENCE_ACTIVE_ROOT`, then the repository
default. The regression was observed RED against the repository-local path and GREEN
after implementation. Independent review requested a conflicting explicit-versus-env
assertion; it was added, and final re-review found no remaining issue.

Fresh local verification for `51dd3a6`: `uv lock --check` passed; the exact
CI-maintained Ruff scope passed; pytest reported 558 passed, 14 documented Windows or
real-snapshot skips, and four known characterization XFAILs in 251.18 seconds;
`pip-audit` found no known vulnerabilities and skipped only the unpublished local
package; and `git diff --check` passed with informational Windows line-ending warnings.
The accidentally broader Ruff probe over every legacy scraper reported 486 pre-existing
findings and was not the maintained gate; no legacy scraper was changed. Focused Linux
verification after fast-forward reported 16 passed and Ruff clean. Commit `51dd3a6`
was pushed to the public development branch; local and remote hashes matched.

The live authenticated preview then started successfully from the persistent snapshot.
Factory planning took about 13 minutes and performed substantial SQLite I/O before
uvicorn reported application startup complete on port 8000. Anonymous results were:
`/healthz` 200; application root, a compiled asset, opportunities API, docs,
`openapi.json`, and export paths all 401. Passwords remained only in Windows
Credential Manager targets `ICOR-Preview-Lucas` and `ICOR-Preview-Manager`.
Both stored credentials produced 303 login redirects to `/` with Secure, HttpOnly,
SameSite=strict, and Path=/ cookie flags; no password, cookie, session secret, or export
token was printed or placed on a command line.

Signed same-origin live checks returned 200 for both `Lucas` and `manager` on the
compiled application, planner options, grouped opportunities, evidence summary, and
completeness endpoints. A localhost request carrying the in-memory export capability
returned 200 for `/api/exports/ml.csv?cutoff=2024-12-31` with CSV media type,
`Cache-Control: no-store`, attachment disposition, and the expected
`observation_id,snapshot_id,release_id,` header prefix. Logout returned 303 to
`/auth/login` and `Max-Age=0`. Port 8000 remained private throughout at
`https://icor-windshield-preview-final-pjqjxgg6qrx4f9r94-8000.app.github.dev`;
no manager-facing public exposure occurred.

Final post-smoke snapshot status and verification both passed again. They reported the
same snapshot ID and SHA, 20 release IDs, zero warnings, manifest observation count
1,529,210, repository observation count 1,529,210, manifest/repository published value
counts zero, and state `verified`. The temporary preview was stopped with a clean
uvicorn shutdown; the loopback tunnel and both diagnostic SSH shells were closed.
Port 8000 is private but no application server is currently running.

The definitive private Codespace is
`icor-windshield-preview-final-pjqjxgg6qrx4f9r94`, checked out at
`51dd3a6222da116e22cd35683cca3f7e9fb497b3`. Its only worktree difference is the
known final-newline drift in `.devcontainer/devcontainer-lock.json`. To restart from
another authorized machine, use the documented portable GitHub CLI or an installed
official `gh`, open an interactive Codespace shell so Codespaces secrets are injected,
run `cd /workspaces/icor-webapp`, confirm
`/home/vscode/.local/bin/uv run python scripts/run_codespaces_preview.py --check`
returns ready, then run the same command without `--check`. Allow roughly 13 minutes
for the 11 GB snapshot factory to become healthy. Keep port 8000 private unless Lucas
explicitly chooses the already-authorized bounded manager-review window, and return it
to private immediately afterward. Before new feature work, fast-forward this development
branch, read this handoff, preserve unrelated worktree changes, and never merge or push
to `main` without separate explicit authorization.

## 2026-09-01 ICOR stabilization execution checkpoint

Lucas approved and commit `33e5cb2` records the design at
`docs/superpowers/specs/2026-09-01-icor-search-data-performance-stabilization-design.md`.
The executable plan is
`docs/superpowers/plans/2026-09-01-icor-stabilization-implementation.md`. Lucas directed
Codex to use the recommended choices without further approval questions and begin building.

Live diagnosis found the main 2024 EU27 Golf label at 61,900 registrations plus variants.
The focused query took 9.668 seconds; a full request was interrupted after 90 seconds.
The 10.75 GB snapshot has 1,529,210 observations and 1,736,619 cohort estimates. Planner
and Opportunity materialize unbounded projections in Python; Evidence performs unindexed
aggregates. Official EEA model-level data is loaded for 2010-2024 final, while 2025
provisional is available but not loaded and 2000-2009 EU model values require explicit
estimate labels. The apparent 1914 registrations are valid UK vintage active-fleet
first-use/manufacture years, not annual sales, so semantics must be fixed rather than data
deleted.

Execution is inline because this task has no permitted subagent delegation. The first task
is the TDD fix for route-scope preservation and multi-token registration search. Preserve
the unrelated local `AGENTS.md` modification and do not touch protected `main`.

Task 1 reached verified green before commit. The frontend regression first failed because
search submission dropped `geography` and `year`; after spreading route state it passed,
with 5 registration-page tests green. Backend regressions first returned zero for both
`Example Motors Alpha` and its punctuated equivalent; tokenized AND-across-make-or-model
matching then passed 26 registration application/API tests. Ruff passed for the changed
Python scope, TypeScript completed with exit code 0, and `git diff --check` reported no
errors. Wildcard escaping remains covered.

Task 2 reached verified green before commit. Schema v5 adds immutable registration-family,
registration-label, evidence-release-summary, and planner-option projections plus composite
indexes for registration scope, Evidence filters, canonical search, and reverse opportunity
lineage. Writable v2/v3/v4 databases migrate forward; sealed read-only snapshots remain
immutable. Projection materialization runs after canonical replay and before finalization,
so it is deterministic and included in the snapshot checksum. Red tests first observed
schema version 4, then a missing rebuild method, then empty candidate projections. Green
verification reported 77 passed and one documented Windows symlink-privilege skip across
the full repository and snapshot-build files; Ruff and `git diff --check` passed. A reversed
input build remained byte-identical.

## 2026-09-03 ICOR stabilization completed and active snapshot verified

All eight stabilization tasks are complete on
development/windshield-demand-platform. The implementation commits are f4bddd6,
8369df6, 8708e3b, 06bbb99, 768fad8, 2c2ef01, c52927f, and 164956c.
The last commit adds bounded and cached Evidence, Planner, and Opportunity reads,
preserves live coverage invalidation, fixes the Evidence browser workflow, and prevents
explicit test/service injection from silently composing the 11 GB active snapshot.
Protected main was not changed. The unrelated local AGENTS.md edit remains unstaged.

The atomically promoted active snapshot is
snapshot-fcb3cdb004a4b7c4042b, database SHA-256
9733d748a239a34184edce41c23df073ea0fb2fea034b7e60aead89e6fe7de65,
manifest SHA-256 79f8b4d6fcd404219b9cfbe666819752c531c109a68389c1f32f246cca45ccf6,
schema 5, build-as-of 2026-08-27T12:00:00+00:00, and deterministic seed
20260827. It contains 1,555,677 observations, zero published values, 21 releases,
and 100 non-blocking snapshot.generic_vehicle_label warnings. The exact releases are
EEA CO2 cars 2010-2024 final revisions v2 through v30, EEA 2025 provisional v31,
KBA FZ10 2024-12 v3, and UK DfT VEH0120 GB, VEH0124 AM/NZ, and VEH0160 GB 2025.
The final supported status command reported this snapshot active, and the final
supported verify command independently matched its SHA, all 21 release IDs,
manifest/repository observation counts, zero published-value counts, and state
verified.

Independent semantic checks reported 1,377,325 assigned observations, 1,780,398
cohorts, 617 completeness records, 85,306 estimated generations, 85,306 generation
entries, 945,232 evidence-only records, 609,963 forecastable records, 111,600
opportunities, and 1,134,350 rejected source records. Sourced generations remain zero
and are labelled accordingly. EU27 Golf 2024 final totals 61,900 and its label
breakdown sums exactly to 61,900 from eea-co2cars-2024-final-v30-r1. EU27 exposes
2010-2025 with 2024 final and 2025 provisional; unsupported EU27 2000 and UK annual
registration queries return unavailable rather than misleading zeroes.

Production-sized timings in seconds were: Registration cold 0.156 and warm p95 0.127;
Evidence summary cold 0.014 and warm p95 0.006; Evidence rows cold 2.613 and warm p95
0.018; Planner options cold/warm p95 0.046/0.045; Planner configurations cold 4.106
and warm p95 0.008; Opportunity brand cold/warm p95 1.285/0.006, model
2.019/0.006, and model-year 6.072/0.027. All interactive warm targets pass. The
model-year first uncached request remains the documented cold-path ceiling; subsequent
immutable reads are bounded and cached, while every Opportunity request checks live
coverage existence before using an uncovered cache.

Final local verification after all code changes: 582 backend tests passed, 14
documented Windows/real-snapshot tests skipped, and four known characterization tests
XFAILed in 60.82 seconds; maintained Ruff and git diff --check passed. Frontend
verification passed 68 Vitest tests across 14 files, TypeScript, ESLint, the Vite
production build (1,955 modules), OpenAPI regeneration/compatibility, and all 20
Playwright workflows in 44.7 seconds. uv lock --check, pip-audit, and npm audit
passed with no known vulnerabilities; only the unpublished local Python package was
not auditable on PyPI.

The authenticated production preview composition was exercised locally against the
verified active database and compiled assets. Health returned 200; anonymous
application/API requests returned 401; login returned 303 with Secure, HttpOnly,
SameSite=strict, Path=/ cookies; authenticated Registrations, Evidence, Planner,
Opportunities, and Completeness pages/APIs returned 200; logout expired the cookie and
subsequent API access returned 401. The Codespaces runner correctly rejected the
current Windows host, so the private Codespaces preview was not restarted and no
public port or server remains running. A future authorized Codespaces restart must
transfer/rebuild this exact active snapshot and use the documented private runner; do
not describe the older remote preview as serving this snapshot.

The former physical snapshot snapshot-2f13ba3f0cd083c7eea8 remains retained, but its
old active state was incompatible with schema 5 and resolved unavailable before this
promotion; treat it as a physical recovery artifact, not an application-ready rollback.
The branch and these commits remain local and were not pushed or merged. No further
product-code work is pending in this stabilization plan.

## 2026-09-03 local review server restarted

At Lucas's request, the current React/FastAPI development application was started from
`C:\Users\LucasCravoVERISSIMO\icor-webapp-development` against the verified local
active evidence root at `.local/evidence`. The protected production checkout and the
older Streamlit prototype were not used or modified. A fresh ephemeral export
capability was generated only in the server process environment and was not printed or
persisted.

The launcher is running in the background as PID 17492. The frontend is available at
`http://127.0.0.1:5173/` and the API at `http://127.0.0.1:8000/`. Startup completed
successfully: the frontend, `/openapi.json`, and the proxied
`/api/v1/registrations/summary` endpoint each returned HTTP 200. The local development
factory does not expose `/healthz` (it correctly returned 404), so readiness was
verified through the documented OpenAPI and application endpoints instead. The app was
opened in the Windows default browser. Logs are
`.local/planner-20260903-164741.stdout.log` and
`.local/planner-20260903-164741.stderr.log`. Closing the browser does not stop the
server; ending the launcher process or restarting Windows does.

No source code, snapshot, branch, remote, or production state changed. The pre-existing
unstaged `AGENTS.md` modification remains preserved.

## 2026-09-03 registration-year generation proxy slice

Lucas made vehicle year/generation discoverability the immediate priority and confirmed
that work should proceed without the unavailable proprietary ICOR catalogue, using the
registration-year proxy. Registration year remains explicitly identified as proxy
evidence rather than a manufacturer-confirmed model year. The example was corrected
before implementation: Volkswagen's official history identifies Golf VIII as the
generation since 2019, so a European Golf registered in 2024 is Mk8, not Mk7.

Read-only diagnosis against active snapshot `snapshot-fcb3cdb004a4b7c4042b` showed
that a 2024 EU27 `Volkswagen Golf` search returns 24 separately published families
totalling 181,082 registrations because manufacturer spelling variants, trims, engines,
and body strings remain separate exact-normalized identities. The two largest were
`volkswagen vw / golf` (116,976) and `volkswagen / golf` (61,900). The snapshot
contains zero sourced generations; its current entries are broad low-confidence
estimated windows. EEA 2024 technical observations do retain useful TAN/T/Va/Ve
identifiers, including Golf type CD/CDV, but the historical aggregate observations do
not materialize that detail even though the immutable source artifacts retain it.

The first UI/API slice is implemented but intentionally does not invent Mk/generation
names. Every registration ranking row now carries the selected registration year in
`model_year` together with mandatory basis `registration_year_proxy`. The
Registrations page presents this as, for example, `2024 generation-year proxy`, and
the explanatory boundary states that registration year is the generation reference
while manufacturer model year remains unavailable. The summary continues to report
`model_year_available=false` because no source supplied a true model year. The
OpenAPI document and generated TypeScript contract include the new required basis.

Changed files are `src/icor/application/registrations.py`,
`src/icor/api/schemas.py`, their application/API tests,
`web/src/features/registrations/RegistrationsPage.tsx`, its unit/browser tests, and
the regenerated `web/openapi.json` and `web/src/lib/api/schema.ts`.

TDD RED reported the expected three failures: the service still returned null and the
row contract rejected `model_year_basis`. Focused GREEN reported 30 backend tests and
7 registration-page tests passing; scoped Ruff, TypeScript, and ESLint passed. Final
verification reported 582 backend tests passed, 14 documented skips, and four known
characterization XFAILs; all 68 frontend tests passed; the Vite production build
completed with 1,955 modules; OpenAPI regenerated successfully; and `git diff --check`
passed with informational Windows line-ending warnings only. The OpenAPI drift command
returned exit 1 only because it correctly displayed the intended checked-in contract
change relative to HEAD. The focused Chromium registration journey then passed 3/3,
including populated behavior and 390/1440 px reflow.

This is not completion of all-brand named generation mapping. The next slice must
create a reviewed public generation-window registry, normalize publisher aliases before
aggregation, reuse retained EEA technical identifiers where possible, assign a named
generation plus basis/confidence to every publishable row, and then rebuild/promote a
validated snapshot. Do not use the repository's legacy Top-100 generation labels as
truth: they label the 2020 Golf as Mk7 despite Volkswagen launching Mk8 in Europe by
then. No active snapshot, remote, production, protected checkout, or deployment changed.

The local review app was restarted so the API loads this new contract. Launcher PID
30268 serves the frontend at `http://127.0.0.1:5173/` and API at
`http://127.0.0.1:8000/`; ignored logs are
`.local/planner-proxy-20260903-175932.stdout.log` and the matching stderr log.
Fresh live checks returned HTTP 200 for both surfaces. A real active-snapshot EU27 2024
`Volkswagen Golf` request returned `model_year=2024`,
`model_year_basis=registration_year_proxy`, and 116,976 registrations for the
highest-ranked publisher family. Stopping launcher PID 30268 and its children or
restarting Windows stops this local app.

## 2026-09-07 reviewed public generation catalog and registration UI slice

Lucas authorized the next generation-mapping steps without proprietary ICOR fitment data
and emphasized correctness and the windshield-demand product goal. The safe boundary is
now source-backed named generation where an annual registration cohort has exactly one
reviewed match, with the explicit registration-year proxy retained for unsupported or
transition-year ambiguity. The app still does not claim exact windshield/SKU fitment from
generation alone; body, facelift, trim, equipment, ADAS/camera/HUD/sensors, drive side,
and other fitment distinctions remain required downstream.

Added `src/icor/generations/public_catalog.py`, a strict in-code public catalog contract.
It normalizes only explicitly reviewed make/model aliases, rejects duplicate aliases,
invalid/non-HTTPS evidence URLs, duplicate/unordered windows, and a non-final open-ended
window. Catalog entries carry stable IDs, manufacturer-confirmed identity, generation
window, evidence URL, dependency group, confidence reason, and optional platform/body/
facelift metadata. `entry_for_year` returns a generation only when exactly one reviewed
window covers the annual cohort; overlapping transition years fail safely to no named
generation instead of choosing one. `GenerationMappingService` now uses reviewed entries
before its estimated fallback, and future official snapshots identify the changed
semantics as `public-generation-registry-v1`.

The initial reviewed profile is Volkswagen Golf for Europe: Golf Mk6 (2008-2012, PQ35),
Golf Mk7 (September 2012-2019, MQB), and Golf Mk8 (from October 2019). The evidence is
Volkswagen Newsroom's official Golf VI and Golf VII histories and Volkswagen's 50th-
anniversary history, which states that Golf VIII was presented in October 2019 and that
the 2024 update is an evolutionary stage of the eighth generation. Exact publisher
aliases observed in the active 2024 data were reviewed for Volkswagen/VW spellings and
Golf, GTE, engine/gearbox, Variant, Life, Style, and eHybrid labels. These aliases do not
merge registration families or erase body/trim text. `Golf Plus` is intentionally not
included because it is a distinct derivative.

Registration ranking rows now expose `generation_name`, `generation_basis`,
`generation_confidence`, and `generation_source_url` alongside the existing registration-
year proxy fields. The Registrations page shows the named generation and a manufacturer
source link when supported; otherwise it continues to show the generation-year proxy.
Its interpretation boundary explains both paths and continues to state that manufacturer
model year is unavailable. OpenAPI and the generated TypeScript client were regenerated.
Tests cover explicit alias matching, exclusion of Golf Plus, catalog validation,
transition-year ambiguity, reviewed mapping precedence, API serialization, proxy fallback,
and both UI states.

Live verification against unchanged active snapshot `snapshot-fcb3cdb004a4b7c4042b`
returned 24 EU27 2024 families for the `Volkswagen Golf` search. Twenty-three families,
totalling 181,080 registrations, now return `Golf Mk8`, basis
`manufacturer_generation_window`, confidence `high`, and the Volkswagen source. The only
proxy result is `volkswagen / golf plus`, with 2 registrations. The major
`volkswagen vw / golf` and `volkswagen / golf` rows return 116,976 and 61,900 respectively.
This runtime enrichment does not mutate or rebuild the active snapshot. No remote,
production, protected checkout, push, merge, or deployment changed.

Fresh verification after the final edits: focused catalog/source/mapping tests passed
7/7; the complete backend suite passed 586 tests with the same 14 documented environment
skips and four known characterization XFAILs in 73.91 seconds; Ruff passed; and
`git diff --check` passed with informational CRLF warnings only. Earlier in this same
final code slice, all 69 frontend tests passed, ESLint passed, TypeScript passed, the Vite
production build completed with 1,955 modules, and the regenerated OpenAPI client passed
the focused 8-test registration suite. Focused Chromium registration journeys passed
3/3, including 390 px and 1440 px layouts.

The local review app is active through launcher PID 16172. The frontend listens at
`http://127.0.0.1:5173/` and the API at `http://127.0.0.1:8000/`. Logs are
`.local/planner-generations-20260907-173611.stdout.log` and the matching stderr log. The
large active snapshot required about three minutes of CPU-bound startup validation; both
ports and the live response were verified after completion. Closing the browser or
clearing a terminal screen does not stop it; terminating the launcher process tree or
restarting Windows does.

This is the validated foundation and first real vehicle profile, not completion of all-
brand generation mapping. Next, expand the catalog model by model from primary
manufacturer sources, prioritizing high-volume EU27 families; add a separate reviewed
publisher-identity alias layer before registration aggregation so spelling variants can
be combined without losing source lineage; use EEA technical identifiers to separate
generation/body/facelift where supported; measure named-generation coverage and surface
it in Completeness; then build, verify, and only explicitly promote a new immutable
snapshot. Never bulk-infer generation names from registration year or the legacy Top-100
labels.

Lucas also asked whether to clear the terminal periodically. Clearing the visible screen
with `cls` or `Clear-Host` is harmless but unnecessary for performance. Closing/restarting
the terminal can stop foreground commands or discard useful scrollback; clearing the
Codex conversation is safe only when the final response says the durable handoff is
current.

## 2026-09-08 forecasting-first workflow and ICOR readiness slice

Lucas clarified that the product must lead with forecast windshield replacements by
model/generation for upcoming years, not with data-quality review. He required the home
route to open Opportunities, a separate vehicle/model search second, and the evidence
pages after those decision tools. He also asked why data does not extend through today,
what the former Generation planner does, why brand-only opportunities were not useful,
how the score is calculated, and for the old ICOR worked-model list to improve readiness.

The app shell and routes now implement that workflow. `/` redirects to `/opportunities`
with `groupBy=model_year`; Opportunities and Model search are first under `Decision
tools`, while Official registrations, Source evidence, and Completeness are grouped under
`Data & audit`. The former Generation planner is now named `Model search` and explains
that users find a brand, model, and generation and inspect its upcoming windshield
replacement range. The shell title is `Windshield replacement forecasts` and the ICOR
brand link also returns to Opportunities. The old `/planner` route remains stable for
bookmarks and implementation compatibility.

Opportunities now defaults to vehicle/model-year rows and visibly presents make, model,
registration-year proxy, reviewed generation when available, forecast replacement range,
ICOR worked status, and the score breakdown. The score remains the approved strategy:
up to 80 points from the row's relative forecast-demand percentile plus up to 20 points
from readiness. Exact configuration coverage receives full readiness weight; conservative
vehicle-year fallback and the legacy worked-model list receive half weight. The coverage
editor is retained but moved behind a `Manage ICOR worked-model coverage` disclosure so
it does not compete with the decision view. Brand/model summary modes remain available
as optional aggregations.

Added `src/icor/application/worked_models.py`, a strict read-only line parser for
`data/icor_supported_models.txt`. It reads source rows individually so duplicate Python-
mapping keys such as Ford Kuga 2012 and 2020 are both preserved, normalizes explicit VW
publisher aliases, skips the suspicious `support audi a6` label, rejects malformed rows
with a line number, and matches only the exact recorded vehicle year. It deliberately
does not extrapolate a legacy label across unrecorded years or treat the legacy G1/G2
labels as authoritative public generation evidence. The snapshot opportunity repository
loads these identities into a per-connection temporary table, applies half readiness
credit after exact/manual fallback precedence, exposes `icor_worked_base_units`, and
returns the reviewed public generation name/basis where the existing public catalog has
an unambiguous match. OpenAPI and the generated TypeScript contract contain the new
fields.

The Opportunities page obtains evidence freshness from the registration summary rather
than hard-coding a year. It currently says `Registration evidence through 2025
(provisional)` and states that official model-level registrations are annual, not a live
daily feed, while forecast horizons continue beyond observed releases. This is the honest
current boundary: the EEA published 2025 provisional new-car data on 25 June 2026, but
there is no official complete 2026 model-level microdata release as of 8 September 2026.
No proxy method was added for 2026 YTD data.

TDD covered legacy aliases, duplicate rows, strict non-extrapolation, malformed input,
snapshot score contribution, default model-year search, home redirect, forecast-first
shell, dynamic freshness, generation/worked badges, visible score components, and renamed
vehicle search. Fresh verification: `uv run pytest -q` passed 589 tests with 14 documented
Windows/optional-integration skips and four known characterization XFAILs; `uv run ruff
check src tests` passed; all 69 Vitest tests passed; ESLint passed; the TypeScript/Vite
production build passed with 1,955 modules; and the final complete Chromium suite passed
21/21, including the explicit home-route test. `git diff --check` passed with
informational CRLF warnings only.

The unchanged real active snapshot `snapshot-fcb3cdb004a4b7c4042b` was validated through
the updated API. A model-year opportunity request returned HTTP 200, 28,510 ranked groups,
and the first result `citroen / c4 / 2010` with 46,266 ICOR-worked forecast units and 10.0
readiness points, proving the legacy catalog affects the score conservatively. Its first
uncached query took about 9.85 seconds and an identical cached query took 0.194 seconds.
The registration summary reports latest year 2025. The first result has no reviewed public
generation name, so the UI honestly displays `Generation not yet verified`; only the
reviewed Volkswagen Golf profile currently has a manufacturer-backed Mk name. Do not
claim all-brand named-generation completion or fill gaps with guesses. Expanding reviewed
manufacturer generation profiles remains the next evidence task.

The verified local review app was restarted from this branch. Launcher PID 28576 serves
the frontend at `http://127.0.0.1:5173/` and API at `http://127.0.0.1:8000/`; both returned
HTTP 200. Logs are `.local/forecast-workspace-20260908.out.log` and
`.local/forecast-workspace-20260908.err.log`. Startup of the large real snapshot remains
CPU-bound for roughly three minutes. Closing the browser or clearing terminal output does
not stop it; terminating PID 28576's process tree or restarting Windows does.

No active evidence snapshot, remote, protected checkout, production environment, push,
merge, or deployment changed. The pre-existing unstaged `AGENTS.md` modification remains
preserved. Clearing the terminal display remains harmless but provides no performance
benefit; preserve useful scrollback when diagnosing a running command.

## 2026-09-08 guided vehicle forecast and visible score formula slice

Lucas asked for the score computation details to be visible and for Model search to
support a text search followed by brand, model, model year or direct generation, then
aggregate all cars of that generation still circulating and forecast windshield
replacements for Europe, Belgium, France, Spain, the Netherlands, England, Germany, and
Poland. The implementation preserves the earlier no-proxy/no-ICOR-data boundary.

Opportunities now contains an always-visible `How the opportunity score is calculated`
panel. It states the exact approved formula: demand percentile times 80; readiness equals
`(exact units + 0.5 × fallback units) / total units × 20`; total score is their sum with
a maximum of 100. The wording also makes clear that scoring ranks opportunities and does
not change the underlying windshield replacement forecast.

The `/planner` Model search route now uses `VehicleForecastSearch`. It provides a search
bar, forecastable Brand and Model selectors, a mutually exclusive Model year / Generation
directly choice, and forecast-horizon selection. Results show registration cohorts,
surviving fleet P50 after decay, windshield replacements P50, and P10-P90 replacement
ranges for EU27, BE, FR, ES, NL, GB, DE, and PL. GB is labelled `United Kingdom (GB;
England is not separable)` because the active source cannot truthfully isolate England.
Unavailable evidence is shown as `Unavailable — not zero`. The responsive table scrolls
inside its card at 390 px and its scroll region is labelled and keyboard focusable.

Added the typed read-only endpoints `GET /api/v1/vehicle-forecasts/options` and
`GET /api/v1/vehicle-forecasts`, their Pydantic/OpenAPI/TypeScript contracts, and
`SnapshotVehicleForecastRepository`. Search only offers canonical identities with
opportunity data. For reviewed catalog profiles, explicit reviewed aliases are combined
and each annual cohort is mapped independently to a manufacturer-backed generation; an
ambiguous transition year is excluded instead of guessed. Unreviewed identities retain
the existing estimated generation entry with explicitly low confidence and must not be
described as manufacturer truth.

The fleet calculation uses the snapshot's registration cohorts and already-computed
active-fleet P10/P50/P90 values, which embody constant annual retention/fleet decay. It
then applies the existing age-band and geography replacement hazard and seeded triangular
uncertainty propagation. The result exposes the survival, hazard, uncertainty, and
calibration methods and states that calibration is assumption-led until proprietary ICOR
fitment and replacement-history data exist.

A live-snapshot diagnostic caught and prevented a material error: the reason code
`forecast-registration-cohort` is also used for estimates filling historical gaps, so it
must not be treated as synonymous with a future sale. The final boundary is the latest
observed or reconciled registration year for the selected vehicle. All cohorts through
that year are included, including explicit historical gap estimates; only years after
that evidence boundary are excluded and disclosed as `excluded_forecast_cohort_years`.
The UI states this plainly. Options with no usable horizon fail safely rather than
raising an index error.

Read-only validation against unchanged active snapshot
`snapshot-fcb3cdb004a4b7c4042b` selected Volkswagen Golf model year 2020 as Golf Mk8 and
included cohort years 2020-2025, excluded future sales cohorts 2026-2028, and disclosed
the only transition year relevant to Mk8, 2019. Transition year 2019 is not offered as a
selectable reviewed model year. Its 2028 P50 results were: EU27 2,297,451
registration-cohort units, 1,675,705 surviving fleet, 58,433 replacements; BE 37,459,
26,473, 986; FR 92,365, 67,500, 2,343; ES 47,547, 34,324, 1,228; NL 21,126, 15,192, 542;
GB 1, 1, 0; DE 1,714,677, 1,253,454, 43,175; and PL 30,748, 22,478, 775. These values are
snapshot/model outputs, not independently calibrated ICOR truth.

Verification: the complete backend suite passed 594 tests with the same 14 documented
Windows/optional-integration skips and four known characterization XFAILs. After the final
transition-year guard, its focused repository/API tests passed 5/5 against both the
fixture and real snapshot. Ruff passed for `src`, `tests`, and changed
`scripts/e2e_app.py`; all 70 frontend tests passed; ESLint and TypeScript passed; the
production Vite build passed with 1,956 modules; and the final complete Chromium suite
passed 21/21, including guided year and generation selection, all eight target market
rows, accessibility, keyboard focus, and 390/1440 px overflow checks. OpenAPI and
generated TypeScript types are current, and `git diff --check` passed with informational
CRLF warnings only.

The refreshed local review app is active under launcher PID 41040 using logs
`.local/vehicle-forecast-final-20260908.out.log` and the matching stderr log. Both
`http://127.0.0.1:5173/planner` and port 8000 returned HTTP 200. The live options endpoint
returned Volkswagen Golf first, and the live forecast returned Golf Mk8, included years
2020-2025, future exclusions 2026-2028, target markets EU27/BE/FR/ES/NL/GB/DE/PL, and
EU27 replacement P50 58,433. No snapshot, remote, protected checkout, production
environment, push, merge, or deployment changed. The pre-existing unstaged `AGENTS.md`
modification remains preserved. Clearing the terminal display remains harmless and does
not stop this launcher; closing its host process tree or restarting Windows does.

## 2026-09-08 Model search manual-entry repair

Lucas reported that manual entry did not work in Model search. The live API diagnosis
showed that exact typed identities were already accepted by the backend; the defect was
the frontend interaction. The main search text only refreshed suggestion dropdowns and
never selected its exact result, while Brand and Model were non-editable `<select>`
controls. A user who typed `Ford Focus` and pressed Search therefore still had to discover
and choose separate dropdown entries, and could not type an identity directly.

`web/src/features/planner/VehicleForecastSearch.tsx` now supports both intended paths.
Submitting an exact full make/model such as `Ford Focus`, or an exact model such as
`Golf`, derives the best exact result from the ordered forecastable API matches and
selects it automatically. Brand and Model are editable, accessible inputs backed by
suggestion datalists, so a user can also type both values directly even when they were
not present in the initial 200 suggestions. Matching is case-insensitive for suggestions;
the backend remains the source of truth for exact normalized identity validation.

The selection query stays disabled until both fields are present. A misspelled or
non-forecastable exact pair returns no horizons, keeps Calculate disabled, and displays
`No forecastable vehicle matches this exact brand and model` instead of guessing. Search
feedback reports the forecastable match count and whether an exact model was selected.
Auto-selection is derived from the existing React Query result rather than an effect or
a second API request, preventing cascading renders, duplicate large-snapshot queries,
and stale-request races. Changing either identity resets year, generation, horizon, and
any previous forecast.

Added component regressions for exact search auto-selection and direct manual entry of a
brand/model outside the suggestion list; both passed. Fresh verification reported all
71 frontend tests passing, TypeScript and ESLint passing, the Vite production build
passing with 1,956 modules, and the complete Chromium suite passing 21/21, including
accessibility and 390/1440 px behavior. A headless browser against the real running local
app confirmed `Ford Focus` auto-selected as `ford / focus` with 16 available years, and
direct `Toyota / Corolla` entry returned 16 years and horizons 2028/2031.

The existing Vite development server hot-reloaded the frontend change; launcher PID
41040 remains the local review-app owner, with frontend/API on
`http://127.0.0.1:5173` and `http://127.0.0.1:8000`. No backend, snapshot, remote,
protected checkout, production environment, push, merge, or deployment changed. The
pre-existing unstaged `AGENTS.md` modification and all earlier development changes remain
preserved.

## 2026-09-08 pre-owner review and official ICOR branding

Lucas requested a deep operational review before presenting the product to ICOR's
owner and asked for the real Belgian company's colours and logo. The identity was
confirmed from ICOR SA's official site at `https://icor.be/`: Wavre, Belgium,
automotive-glass accessories and tools. The official website stylesheet uses cyan
`#00A3D9` as its dominant brand colour, and the official materials pair it with dark
navy and white.

The application shell now uses an ICOR palette: official cyan `#00A3D9`, navy
`#152F4A`, white, blue-grey surfaces, and accessible darker blue interactive states.
The placeholder `I` monogram was replaced on desktop and mobile by ICOR's exact
official white wordmark/tagline. The original 151 x 45 PNG from
`https://icor.be/images/icor-blanc.png` is embedded in a self-contained SVG wrapper so
the demo does not depend on the public site at runtime. The embedded 21,644 bytes match
the official source SHA-256
`55c76bf6c858444e0eaf3889e4f737b928ffedc1fc5652563a9d585d74efd95c`.
Provenance and preservation rules are recorded in `web/src/assets/README.md`. The
browser title is now `ICOR | Windshield demand forecasts`, its description states the
forecast purpose, and its theme colour matches the navy shell.

The branding regression was observed RED before the official logo existed and then
passed 5/5 after implementation. Final frontend gates passed: all 71 Vitest tests,
TypeScript, ESLint, Vite production build with 1,956 modules, final focused shell tests
5/5, and npm audit with zero known vulnerabilities. The complete Chromium suite passed
21/21, including all planner/opportunity/data journeys, visible keyboard focus,
automated serious/critical accessibility checks, and 390/1440 px overflow checks.

The complete backend gates passed: `uv lock --check`, maintained Ruff, 594 tests,
14 documented Windows/optional-integration skips, the four historical strict XFAILs,
and pip-audit with no known third-party vulnerability (the unpublished local package
cannot be checked against PyPI). The active pointer still identifies immutable snapshot
`snapshot-fcb3cdb004a4b7c4042b`; the promoted `snapshot.json` SHA-256 exactly matches
the pointer's recorded manifest digest
`79f8b4d6fcd404219b9cfbe666819752c531c109a68389c1f32f246cca45ccf6`.
Its manifest still records database SHA-256
`9733d748a239a34184edce41c23df073ea0fb2fea034b7e60aead89e6fe7de65`,
1,555,677 observations, 21 releases, zero published values, and 100 non-blocking
generic-label warnings. A fresh full 12.8 GB status/hash scan was stopped after it
remained I/O-bound much longer than the product checks; its exact orphaned read-only
process tree was terminated. No candidate, pointer, snapshot, or source data changed.

Live real-snapshot review passed. All six routes returned 200 with no browser console
errors, page errors, failed network responses, or 1440 px page overflow. Opportunities
and Model search also rendered at 390 px with the official logo visible, working mobile
menu, no errors, and exact viewport width. The owner landing route redirected to
model-year Opportunities, showed the score formula and 2025 provisional freshness,
and rendered ranked vehicles. Twenty-five live opportunity rows had zero arithmetic
or range failures: `demand_points + readiness_points = total_points` within the API's
one-decimal serialization, with the 80/20 caps preserved. A real Volkswagen Golf 2020
forecast returned Golf Mk8, cohorts 2020-2025, future exclusions 2026-2028, all eight
requested markets, ordered non-negative P10/P50/P90 ranges, and HTTP 422 for an invalid
vehicle. Live manual Ford Focus entry and direct Volkswagen Golf generation selection
both completed without browser errors.

The final exact code was restarted cleanly under hidden launcher PID `39636`. The
frontend is `http://127.0.0.1:5173/`, the API is `http://127.0.0.1:8000/`, and logs are
`.local/owner-demo-20260908.out.log` and `.local/owner-demo-20260908.err.log`. Snapshot
composition took about five and a half minutes on this run. The first uncached
model-year Opportunity API read took 19.8 seconds and the first full browser landing
took 15.4 seconds; the warmed landing then took 2.18 seconds with no console errors.
The live caches are currently warm. Closing the browser or clearing terminal output
does not stop the launcher; terminating PID 39636's process tree or restarting Windows
does.

Readiness verdict: the current loopback app is ready for a controlled owner
demonstration on this computer or by screen sharing. It is not a production-ready or
externally shareable service: the local development composition has no user login;
forecast calibration and fitment remain assumption-led until proprietary ICOR data is
integrated; only Volkswagen Golf currently has reviewed manufacturer-backed generation
names; evidence is annual through 2025 provisional rather than live through today; GB
cannot be separated into England; and historical OpenAI-key revocation/rotation still
requires owner confirmation. Do not call the forecasts validated ICOR demand, expose
port 5173/8000 publicly, push, merge, or deploy without the separate approved hardening
and release sequence.

No remote, protected checkout, production environment, active snapshot, push, merge,
or deployment changed. The pre-existing `AGENTS.md` modification and accumulated
development changes remain preserved.

## 2026-09-08 generation coverage, Odoo, global warming, and pre-2000 assessment

Lucas confirmed that Great Britain can remain whole, proprietary ICOR information will
arrive later, and every model exposed to a client must have a correct generation name.
He also disclosed that ICOR uses Odoo and asked about future integration, globally warm
forecast execution, trusted pre-2000 evidence, and current data quality. This checkpoint
is analysis only; no application, data, snapshot, cache, Odoo, remote, or deployment
state changed.

Read-only active-snapshot measurement found 22,893 forecastable raw canonical vehicle
IDs/make-model pairs, 85,543 canonical vehicles in total, 85,306 generation entries,
1,094 distinct generation display labels, 111,600 opportunity estimates, and 1,780,398
cohort estimates. The Model search default is not a complete catalog: its repository
sorts the identities and returns only the first 200. All 85,306 snapshot generation
entries have `identity_kind=estimated`; manufacturer-backed Golf names are runtime
enrichment from the one reviewed public profile. Therefore, manually sourcing a
generation for every raw identity is neither tractable nor correct: raw identities
contain trims, engines, spelling variants, and publisher-specific labels. The required
boundary is to canonicalize aliases into genuine model series first, attach reviewed
month-precise generation windows and technical identifiers second, and expose only
verified model/generation identities in client-facing search/opportunities until their
coverage is complete. Annual transition cohorts must remain mixed/ambiguous unless
technical type, VIN/type-approval, or monthly evidence can split them.

Recommended generation source order is: manufacturer histories/brochures and formal
type-approval evidence; a licensed structured aftermarket vehicle tree such as TecDoc
with vehicle/range/design/build windows; regulatory technical identifiers; and secondary
sources only as corroboration. TecAlliance states that TecDoc supplies standardized
vehicle/product/linkage data and API/ERP integration, but licensing, historical depth,
territories, redistribution rights, and the exact generation field population must be
tested on a representative ICOR vehicle sample before selection. The public raw list
must not be bulk-labelled through year-only guesses.

Odoo integration is feasible, conditional on ICOR's edition/version, hosting, API plan,
modules, access rights, and actual data model. Odoo 19's official JSON-2 API uses bearer
API keys and database-specific model/field documentation; external API access is limited
to Custom plans. Older XML-RPC/JSON-RPC interfaces are deprecated on Odoo's published
roadmap. The safe design is a dedicated least-privilege read-only bot with a rotated
secret outside Git, an incremental server-side sync keyed by `write_date` plus record
ID, immutable raw pulls, and a reviewed mapping from Odoo product/SKU/order/manufacturing
records to canonical vehicle/generation/configuration. The app must not query Odoo in a
user request. Before implementation ICOR must define “worked on” (quoted, sold,
invoiced, manufactured, fitted, or technically developed) and identify where vehicle,
generation, windshield SKU/OE number, and fitment attributes are stored. If external API
access is unavailable, a controlled Odoo export or small server-side Odoo module is the
fallback.

For worldwide low-latency use, do not rebuild “the entire world” inside page requests.
Use scheduled acquisition/Odoo-sync jobs, versioned precomputed model-generation-market-
horizon aggregates, candidate validation, atomic last-known-good promotion, and a fast
serving store/cache. Check sources several times daily, rebuild only when inputs change,
run a nightly deterministic forecast refresh if business assumptions require it, prewarm
all published model/generation routes immediately after promotion, and retain the prior
snapshot/cache if any stage fails. A weekly full integrity rebuild can detect drift.
Global execution still requires a licensed global registration/VIO source; the current
snapshot is not worldwide.

Current data-quality assessment: source provenance and immutability are strong for the
implemented official scope, but product identity and forecast calibration are not yet
client-grade. The snapshot contains 1,555,677 observations, of which 1,555,195 are
normalized-label usable and 1,377,325 are assigned; completeness records report 609,963
forecastable and 945,232 evidence-only records, zero sourced generations, and 1,134,350
source-level rejected records across loader scopes. Every 1,780,398 cohort and every
111,600 opportunity estimate is labelled low confidence. EEA model-level evidence is
annual from 2010 through 2024 final plus 2025 provisional and is a regulatory CO2
monitoring source with noisy commercial labels, not a complete world sales/parc source.
The forecast hazard, survival, and uncertainty remain assumption-led until ICOR history
or defensible external calibration is integrated.

Trusted pre-2000 evidence exists only in partial forms. The UK DfT/DVLA official
VEH0120 stock file covers Great Britain by make/generic model/model from 1994 Q4, while
model-level first registrations begin in 2001; current stock by old first-use/manufacture
year can describe surviving vintage cohorts but cannot reconstruct complete historic
annual sales. ACEA publishes European historical totals by country/manufacturer, not the
required model detail. No reviewed free official source provides comprehensive global
model-level pre-2000 sales/parc. The realistic route is licensed vendor evidence plus
national-register backfills and manufacturer archives, all stored separately with scope,
licence, revision, and confidence. S&P advertises model-level global new registrations
but its currently described marketplace history starts in 2019; MarkLines advertises
model-by-country sales with 99% global-sales coverage; their exact pre-2000 depth and
redistribution terms require commercial confirmation. S&P's VIO product is especially
relevant to aftermarket fleet exposure, while TecDoc is relevant to vehicle identity
and fitment rather than sales counts.

## 2026-09-08 verified client-release candidate

Lucas authorized finishing the first build so it can be shared with the client and
emphasized mistake avoidance. The implemented release boundary is a deliberately narrow
authenticated Volkswagen Golf pilot, because every vehicle exposed to a client must
have a correct generation name and Golf is currently the only manufacturer-reviewed
profile. The broader internal workspace remains unchanged when client-release mode is
absent.

Client-release mode is enabled only when the backend uses
ICOR_CLIENT_RELEASE_MODE=verified and the frontend bundle was compiled with
VITE_ICOR_CLIENT_RELEASE=verified. SnapshotVehicleForecastRepository then rejects every
unreviewed identity. SnapshotOpportunityRepository joins only explicit reviewed aliases,
consolidates them as canonical Volkswagen Golf, and excludes annual transition cohorts
when month-level windows overlap. The client opportunity API rejects brand/model
groupings that cannot guarantee a generation on every row. Canonical drill-down expands
through all reviewed publisher aliases.

The authenticated preview now has a deny-by-default client allowlist. It exposes only
read-only Opportunities, Model search, vehicle forecasts, and registration freshness.
It returns 404 for registrations, evidence, completeness, ML export, API documentation,
coverage management, unknown APIs, and non-login mutations. The client frontend removes
internal/audit/admin navigation, grouping controls, and the mutable coverage editor,
shows a Verified client preview badge, and explicitly labels the catalog and forecasts.
The preview runner can serve a separately compiled bundle from
ICOR_PREVIEW_ASSET_ROOT, preventing internal/client asset mixing.

The release procedure, credential rules, limitations, smoke test, and shutdown
requirements are documented in docs/CLIENT_RELEASE.md and linked from README.md. The
verified client bundle was built in ignored .local/client-release. The existing
web/dist directory could not be emptied on Windows because index.html returned EPERM;
the separate output both avoided that local packaging lock and is the safer release
boundary.

Fresh verification on the final implementation:

- focused RED tests first failed because verified-only constructors and client-release
  composition did not exist; after implementation, the focused repository/API/preview
  set passed 44 tests with one Windows symlink-privilege skip;
- uv lock --check passed; maintained Ruff passed after one mechanical import-order fix;
- the complete backend suite passed 599 tests, 14 documented Windows/optional-real-data
  skips, and the four historical strict XFAILs in 65.89 seconds;
- pip-audit found no known third-party vulnerabilities and skipped only the unpublished
  local package;
- all 73 Vitest tests, TypeScript, ESLint, and npm audit passed, with zero known npm
  vulnerabilities;
- the verified client Vite production bundle passed with 1,956 modules and was written
  to .local/client-release;
- the complete Chromium suite passed 21/21, including workflow, accessibility, keyboard,
  and 390/1440 px responsive gates;
- a real-snapshot invariant returned exactly 11 client opportunity rows, all with
  manufacturer_generation_window identity, only Volkswagen Golf in Model search, and
  HTTP 422 for prohibited brand grouping;
- a final authenticated composition smoke test against the compiled client bundle and
  active snapshot passed login, the 11 verified rows, all internal-route/API denials,
  logout, and post-logout protection.

The active snapshot remains snapshot-fcb3cdb004a4b7c4042b. The existing local internal
owner-demo launcher PID 39636 and frontend/API processes remain active on
http://127.0.0.1:5173 and http://127.0.0.1:8000; they are not the authenticated client
composition and must not be exposed. No remote, protected checkout, production
environment, active snapshot, push, merge, deployment, port visibility, or external
sharing changed.

The complete application and client-release candidate is committed on the isolated
development branch as e284cd2 (feat: prepare verified client preview). AGENTS.md and
this durable handoff remain the only unstaged tracked changes; they were intentionally
excluded from the product commit. The branch has not been pushed.

The release candidate is ready for an owner-controlled sharing step.
Before any external URL is created, Lucas must confirm the historically exposed OpenAI
key was revoked/rotated, approve the development-branch push and authenticated preview
deployment, provide the intended reviewer identity and review window, and confirm that
the Volkswagen Golf-only pilot scope is acceptable. Do not broaden the catalog with
guessed generation names.

## 2026-09-08 full-catalog release requirement and data dependency

Lucas clarified that OpenAI-key revocation must wait until immediately after the urgent
client review. The new React/FastAPI preview does not require that historical key, so it
must remain absent from the client build and runtime. Rotation is still required as soon
as the review ends.

Lucas authorized merging and proceeding toward the final release, rejected the Golf-only
pilot as the client scope, and required all models to be present with correct generation
names. He requested reuse of the prior web-app accounts and passwords. Local inspection
found no tracked or local credential file containing those values, which is correct.
The earlier checkpoint records repository-scoped Codespaces secrets for the prior
accounts. Reuse must occur by retaining those secrets in the same repository; passwords
and hashes must not be retrieved, copied, printed, committed, or moved through chat.

The full-catalog requirement is not satisfiable from current inputs without making false
claims. The active snapshot contains 22,893 forecastable raw make/model identities. The
legacy ICOR worked list has only 148 year records across 128 distinct noisy labels and
includes generic values such as G1. All 85,306 snapshot generation entries are estimated,
and only Volkswagen Golf has a reviewed manufacturer profile. Therefore neither the raw
snapshot nor the legacy list can be presented as correct all-model generation truth.

Current official-source research reconfirmed TecDoc as the appropriate acquisition
boundary. TecAlliance describes TecDoc Reference Data/Data Package/Web Service as a
standardized global vehicle/type/linkage source with more than 260,000 vehicle types,
construction windows, model design identifiers, and API/bulk integration. TecDoc VIO
adds harmonized official parc evidence across more than 75 countries linked to the
reference vehicle tree. Access, historical depth, field population, client-display and
derived-data rights, and windshield linkage rights require a commercial account and
licence. NHTSA vPIC is authoritative manufacturer-submitted US/VIN evidence but cannot
satisfy the European/global all-model generation requirement. EU type approval confirms
type/variant/version semantics but no reviewed open comprehensive client-ready generation
API was found.

The exact vendor/Odoo intake and acceptance contract is now documented in
docs/VEHICLE_CATALOG_INTAKE.md. Required inputs are a licensed TecDoc sample/export or
equivalent ICOR/Odoo reference export plus rights confirmation. Until one is provided,
do not expand the verified client mode, merge it to main, or deploy it as the requested
all-model final version.

An attempted combined non-interactive fetch/status/push command was rejected by the
managed approval reviewer because the exact remote payload was not separately approved.
No fetch or push ran. Although Lucas broadly authorized proceeding, a future push must
name the development branch and GitHub repository explicitly in the approval request.
No merge, remote, production, account secret, deployment, or active snapshot changed in
this checkpoint.

## 2026-09-08 authoritative-catalog follow-up

Official ICOR-site research found a public new-articles catalogue with many useful
vehicle/product descriptions and construction-year fragments, plus the ICOR Shop user
guide. The catalogue is product-oriented, contains abbreviated and mixed-granularity
vehicle labels, and is not a complete canonical all-model generation source. It may be
used later as an ICOR fitment supplement, but it cannot satisfy the client's requirement
that every exposed model already have a correct generation identity.

The release/key timing was clarified in docs/CLIENT_RELEASE.md, and the authoritative
TecDoc-or-Odoo field, licence, sample, and acceptance contract was added at
docs/VEHICLE_CATALOG_INTAKE.md. Both files passed git diff --check and were committed
locally as 3fa0b08 (docs: define authoritative vehicle catalog intake). The development
branch is now 13 commits ahead of its last-known remote tracking state. AGENTS.md and
this handoff remain the only uncommitted tracked files; they are intentionally excluded
from product commits.

The blocking input remains unchanged: obtain an existing ICOR TecDoc Reference
Data/Data Package/Web Service/VIO sample or an equivalent Odoo export with the documented
canonical vehicle fields and rights. No truthful code-only path can manufacture the
missing all-model generation truth. No push, merge, deployment, credential, production
checkout, active snapshot, or running-server state changed in this follow-up.

## 2026-09-09 vehicle-year catalog, validated forecast, and local client review

Lucas confirmed that proprietary ICOR data will not be available before the client
meeting, asked for real names across the available vehicle years, authorized testing,
pushing, and launching the application, and reiterated that the historical OpenAI key
cannot be revoked until immediately after the meeting.

The client-release boundary now exposes every forecastable official-source make/model
and observed registration cohort year. The UI and API describe these truthfully as
vehicle/registration years rather than claiming they are universal manufacturer model
years. Direct estimated-generation selection is disabled in client mode. A forecast
identity is source-specific, for example `Ford Focus — 2025 registration cohort`, with
basis `official_source_registration_cohort`. Manufacturer generation names remain
available only where independently reviewed. This does not create the missing licensed
all-model generation/fitment truth described above.

Forecasting now uses `validated-recency-damped-ensemble-v2`: a deterministic fixed
50/50 ensemble of the latest observation and a five-year trend whose future increments
are damped by 0.8 per step. Outputs are non-negative, and diagnostics use multi-step
rolling WAPE. The reproducible benchmark and limitations are in
`scripts/benchmark_registration_forecasts.py` and `docs/FORECAST_VALIDATION.md`.
Against the promoted snapshot's 16,239 eligible series and newest-two-year outer
holdout, production WAPE was 0.503518 versus 0.628230 for the replaced baseline, a
19.85% relative error reduction. No claim is made that one algorithm is universally
best; this method won the recorded same-data benchmark without introducing unvalidated
high-capacity ML.

The immutable candidate was fully validated and atomically promoted:
`snapshot-eefbf3566f2106e5965c`, database SHA-256
`2349509ac863995632670d40002b0f6170fe1be1dcfffaf8455a416795c6662d`,
1,555,677 source observations, all 21 expected releases, zero published proprietary
values, and 100 previously classified generic-label warnings. The active pointer
manifest digest is
`8baf20c2af9d24ac6fc4b5145f8c72ee44a59fcaaa9115697fa1bce27cd8c6e1`.

Fresh release verification on the final product commit:

- complete backend: 605 passed, 14 documented Windows/optional-data skips, and the four
  historical strict XFAILs;
- maintained changed-file Ruff, `uv lock --check`, and pip-audit passed; pip-audit
  found no known third-party vulnerabilities and skipped only the unpublished local
  package;
- frontend: 15 files and all 74 Vitest tests passed; generated OpenAPI schema was
  unchanged; TypeScript and ESLint passed;
- verified client production bundle built successfully with 1,956 modules in
  `.local/client-release-20260909`;
- complete Chromium E2E passed 21/21, including workflow, accessibility, visible
  keyboard focus, and 390/1440 px responsive checks;
- shipped frontend runtime audit (`npm audit --omit=dev`) found zero vulnerabilities.
  The full development-tree audit newly reports two high-severity instances of
  GHSA-2883-xcg3-v3hh through `openapi-typescript -> @redocly/openapi-core 1.34.19 ->
  js-yaml 4.3.1`. Both available forced overrides were tested and proved incompatible
  with schema generation, so neither was retained. The affected generator is
  development-only, consumes the repository-generated OpenAPI file, and is not bundled
  into the browser release; monitor upstream for a compatible fix.

The scoped product, tests, benchmark, and documentation are committed as `e7e7c94`
(`feat: expand vehicle-year catalog and validate forecasts`). `AGENTS.md` and this
handoff remain intentionally outside the product commit. The exact current-tree
OpenAI-style key scan returned zero matching files. The new React/FastAPI client and
deterministic forecast do not require OpenAI, but preserved legacy Streamlit tooling
still has the dependency. The old key remains in four historical commits and must be
revoked immediately after the client meeting; history rewriting cannot revoke it.

The development branch was fetched and confirmed zero commits behind and 14 ahead of
`origin/development/windshield-demand-platform`. The managed reviewer nevertheless
rejected the push because Lucas's message did not explicitly name both
`https://github.com/lucascverissim0/icor-webapp.git` and
`development/windshield-demand-platform`. No workaround, push, merge, main-branch
change, or deployment occurred. The next push requires Lucas to authorize that exact
repository and branch in a new message.

The local client-release frontend is running at `http://127.0.0.1:5173/` and was
opened in Lucas's default browser. The client-release API is bound only to
`http://127.0.0.1:8000/`, serves the new active snapshot, and uses explicit
`client_release=True`. Live checks returned 28,512 opportunity rows; Ford Focus 2025
used the official registration-cohort identity, eight markets, non-negative ordered
intervals, and no estimated generation label. Launcher PIDs are 5536 (web) and 38060
(API wrapper; serving child 39388); logs are
`.local/client-web-20260909.*.log` and `.local/client-api-20260909.*.log`.
This local development composition is unauthenticated and localhost-only: three
internal data routes are hidden, while API docs/OpenAPI remain reachable. It is for
Lucas's on-computer review only and must not be forwarded publicly. The existing remote
account/password secrets were not read or changed; the authenticated client-sharing
runner must use those repository secrets after the authorized push.

## 2026-09-09 registration-cohort and forecast sanity audit

Lucas questioned the unusually low Volkswagen Golf 2020 Belgium registration-cohort
figure and asked for a careful read-only review of the numbers and forecast quality.
No application code, active snapshot, source release, pointer, remote, production,
protected checkout, or deployment state changed. The localhost client composition was
still reachable on ports 5173/8000. Browser control could not initialize because the
Windows sandbox failed with `SetTokenInformation` error 1344, so the live API and
immutable snapshot were inspected directly outside that broken sandbox boundary.

The live Volkswagen Golf 2020/horizon-2028 endpoint returned 7,653 Belgium
registrations. This is not a clean source observation: it combines 7,401 reconciled EEA
registrations with 251.9200 synthetic `forecast-registration-cohort` Mk7 registrations
and a zero synthetic Mk6 row, while labelling the result
`official_source_registration_cohort` and `source-reported`. The four observed
components are 7,075 `volkswagen vw / golf`, 298 `volkswagen / golf`, 25
`volkswagen vw / golf gte`, and 3 `vw / golf`, all from
`eea-co2cars-2020-final-v22-r1`.

The source itself is coherent. All Belgian EEA passenger-car rows sum to 431,922 for
2020, only 431 (0.10%) above the 431,491 FEBIAC national total. All EEA labels
containing Golf sum to 8,941: the 7,401 core rows, 1,538 Golf Sportsvan rows, and two
one-unit long-form aliases. Published Belgian model totals report 8,937 Golf
registrations. The apparent low value is therefore mainly an identity/scope problem:
Golf Sportsvan is deliberately separate for windshield fitment, but the UI does not
make that comparison boundary clear enough, and the reviewed Golf aliases omit
historical punctuation variants such as `volkswagen, vw`.

Historical label drift is material. The current alias set makes Belgian core-Golf
observed totals appear as 80 in 2011, 60 in 2012, 180 in 2013, 145 in 2014, and zero in
2015-2017, while source rows under `volkswagen, vw / golf` are 18,034, 12,345, 14,686,
15,675, 13,652, 13,242, and 13,741. The source-specific catalog remains fragmented
rather than a canonical cross-year model series.

The more serious generation defect is in `GenerationPlanningService`: every generation
series is forecast through 2028/2031 without enforcing the generation's end month.
Golf Mk6 and Mk7 therefore receive registrations after discontinuation. Germany 2020
sums 114,559 observed Golf Mk8 units, 4,234 observed Golf GTE units, and 361,690.3075
synthetic Mk7 units, producing the implausible displayed total 480,483. In the 2028
materialization, 287 exact canonical-vehicle/geography/year keys contain both observed
and synthetic cohorts. Across client-selectable years through each vehicle's latest
evidence boundary, 65,736 market/vehicle/year keys are synthetic-only but the client
selection contract describes them as official/source-reported; 287 are mixed.

The Opportunities model-year query has a separate semantic error. It joins only
`opportunity_input.input_position = 0` and assigns the entire generation/geography/
horizon opportunity to that first cohort year. A Belgium Golf Mk8 opportunity shown as
2020 contains nine cohorts from 2020 through 2028; a Mk7 row shown as 2013 contains
sixteen cohorts from 2013 through 2028. These are not cohort-year forecasts.

Fresh benchmark execution against active snapshot `snapshot-eefbf3566f2106e5965c`
reproduced 16,239 series, production WAPE 0.503518, replaced-baseline WAPE 0.628230,
and 19.85% relative improvement. A same-holdout challenger gave last-observation WAPE
0.516167, so production improves on that strong naive baseline by only about 2.45%.
Production first-step WAPE was 0.357571, second-step WAPE 0.690450, signed bias was
+0.130621 of holdout demand, median per-series WAPE was 0.655179, and p90 was 2.345647.
The registration forecaster is a reasonable conservative baseline, not a client-grade
engine. Final windshield demand is less validated: constant survival, age-band hazards,
+/-20% hazard bounds, and triangular uncertainty remain assumptions with no empirical
coverage test against proprietary replacement outcomes.

Release verdict: do not present the current numbers as reliable client forecasts and do
not push/deploy this release unchanged. Enforce generation windows, separate observed/
estimated/forecast rows in every response, make model-year opportunities genuinely
cohort-specific, complete reviewed alias canonicalization, rebuild/promote a new
immutable snapshot, add aggregate/discontinuation invariants, rerun benchmarks with
naive baselines/bias/interval coverage, and re-audit high-volume samples. The historical
OpenAI key still requires revocation immediately after the client meeting as separately
documented.

## 2026-09-09 concise-report preference

Lucas said the preceding audit report was too long to read under time pressure. All
future completion reports must be brief and decision-focused: lead with the verdict,
include only the most important evidence, risk, and next action, and keep exhaustive
technical detail in this handoff unless Lucas asks for it. `AGENTS.md` now makes this a
durable repository directive. No application code, data, snapshot, server, remote, or
production state changed.

## 2026-09-10 evidence-anchored windshield hazard and discontinuation correction

Lucas clarified that a discontinued generation must continue producing replacement
demand while vehicles from its historical cohorts remain in circulation, and asked for
the strongest credible free windshield-demand assumption available. Public research
identified the latest France Assureurs motor report as the best national frequency
anchor: its July 2026 report for calendar 2025 gives 60.6 glass claims per 1,000 covered
exposures for the subscribed glass guarantee among first-category vehicles outside
fleets. A Crédit Agricole Assurances/Pacifica, BCA Expertise, Europ Assistance, and
Institut Louis Bachelier study using 2022 operating data reports a 71% windshield-
replacement operation frequency per glass claim, alongside 12% windshield repair and
17% replacement of other glazed elements. It says Pacifica supplied the glass-claim
data and their representativeness was checked against France Assureurs. The study's
narrative separately says 15% repair, a minor internal discrepancy that does not change
the reported 71% replacement share used here.

The implemented versioned baseline is therefore 0.0606 x 0.71 = 0.043026, or 4.3026%
annual windshield-replacement events per active vehicle, with explicit planning
scenarios of 3.44208% / 4.30260% / 5.16312% (-20% / base / +20%). These are not
empirically calibrated P10/P90 quantiles. The former unsupported age bands (2%-6%) and
GB +10% multiplier are no longer used. Age and country remain accepted inputs for
contract validation and future evidence-backed calibration, but the default hazard is
flat because the reviewed free evidence does not support age- or country-specific
effects. Applying insured French claim behavior to all surviving vehicles and other
markets remains an explicit proxy limitation. Exact evidence, formula, interpretation,
and URLs are recorded in `docs/WINDSHIELD_DEMAND_ASSUMPTIONS.md`.

`GenerationPlanningService` now removes any assigned or forecast registration cohort
after a generation's documented end year and filters forecast output at that same
boundary. It does not remove the discontinued generation: every valid earlier cohort
continues through the survival model to the 2028/2031 horizon and continues generating
replacement opportunity while vehicles remain active. The hazard method is versioned
as `france-insurance-windshield-hazard-v2` in official snapshot identity, generation
planning, and live guided vehicle forecasts. The unsupported runtime GB multiplier was
removed. Focused regressions cover the exact 4.3026% calculation, flat age behavior,
the 20% scenario interval, absence of an unsupported country premium, and continued
positive demand from a generation ending in 2022 with no post-2022 cohorts.
The scoped implementation, tests, and source methodology are committed locally as
`b3cd2ff` (`fix: anchor windshield demand and stop ended cohorts`). The development
branch is 15 commits ahead of its last-known remote tracking state; nothing was pushed.

Fresh verification: the focused backend set passed 15 tests; maintained Ruff passed;
`git diff --check` passed with informational CRLF warnings only; and all 74 frontend
Vitest tests passed. Two complete backend runs reached respectively 605 and 606 passes,
14 documented skips, and four historical XFAILs, but each encountered one different
known intermittent Windows `MoveFileW` access-denied race in a temporary ReleaseStore
test. Both affected tests passed immediately in isolated reruns. No forecasting test
failed after implementation.

The corrected 12.8 GB immutable snapshot was not rebuilt or promoted. Only about
15.3 GB remained free after verification, while this repository's canonical build can
temporarily require both a similarly sized database and WAL before finalization; a
capacity failure would be expected. Several old candidate copies exist, including
candidate copies of promoted snapshots, but none was deleted because Lucas did not
authorize material immutable-artifact cleanup. The active pointer therefore remains
`snapshot-eefbf3566f2106e5965c`, whose persisted opportunity values still use the old
hazard/window logic. The already-running localhost client composition was not restarted
and must not be presented as containing this correction. No remote, push, merge,
deployment, protected checkout, active pointer, source release, or production state
changed. `AGENTS.md` and this accumulated handoff remain preserved user/session changes.

Next: obtain explicit authorization either to remove only verified redundant candidate
copies or provide at least 30 GB additional temporary disk space; then rebuild the exact
21-release snapshot, require zero new blocking findings, independently verify and
promote it atomically, re-audit high-volume discontinued generations, rerun application
gates, and restart the local client composition. Do not push or deploy the old active
snapshot as corrected.

## 2026-09-10 corrected engine rebuilt, promoted, and cleaned

Lucas requested no installation, maximum safe ICOR cleanup, and a rebuild with the
evidence-backed windshield numbers. No software, package, dependency, or runtime was
installed; all work used the existing `.venv`, `node_modules`, and retained local
releases. Cleanup stayed inside this development worktree and did not touch personal
files or unrelated processes.

Verified old candidates, inactive snapshots, caches, a paused release-store duplicate,
old client bundles, the superseded active snapshot, the promoted candidate duplicate,
and four unreferenced diagnostic databases were permanently removed. Total reclaimed
space was 65,828,749,237 bytes (about 61.31 GiB). The deleted generated artifacts are
not directly recoverable but are reproducible from the preserved code and complete
21-release store. The last material-cleanup check showed 97,460,903,936 bytes free.

The exact prior 21 releases, UTC timestamp `2026-08-27T12:00:00+00:00`, and seed
`20260827` rebuilt `snapshot-7e0eb1d25f73ee96c0f9`, database SHA-256
`033633ecf9378a646c47214a64c3ead8965706e7362bf9914062c8432ba8b5f1`. It has
1,555,677 observations, zero proprietary published values, all expected releases, and
only the same 100 generic-label warnings. Atomic promotion, independent status, and
the full repository recount all returned that exact ID, hash, and observation count.

Completeness passed with 85,358 generations, 883,240 cohorts, and 111,694 opportunities.
SQL audits found zero cohorts after generation end, zero opportunities missing the
`0.043026` base-rate or plus/minus-20-percent assumptions, and zero negative or
unordered intervals. All opportunity rows use `france-insurance-windshield-hazard-v2`.
Golf Mk7 ends in 2019 and has no later cohort. Nevertheless, 111,558 positive rows and
11,559,428.3839 P50 events remain in post-discontinuation horizons, confirming that
surviving historical cars keep generating replacement demand. Stored opportunity P50
is a seeded propagated median, so its ratio to fleet P50 need not equal the exact
4.3026% underlying hazard.

Fresh no-install gates passed: 15 focused backend tests, Ruff, all 74 frontend tests,
TypeScript, and the 1,956-module Vite production build. The verified client bundle was
rebuilt in `.local/client-release`. The loopback-only composition is running at
`http://127.0.0.1:5173/` (launcher 26428, child 18324) and
`http://127.0.0.1:8000/` (launcher 5168, child 18152), with logs under
`.local/client-{web,api}-20260910.*.log`. Live HTTP checks passed, returned 28,512
opportunity rows, and both Golf 2018 and Ford Focus 2025 reported
`official_source_registration_cohort`, the new hazard method, and active data version
`snapshot-7e0eb1d25f73ee96c0f9`. Golf retained positive 2028 demand while forecast sales
cohorts were excluded.

The localhost composition is unauthenticated and must not be exposed publicly. Nothing
was pushed, merged, deployed, or changed in the protected checkout. Product code
remains local at `b3cd2ff`; the branch remains 15 commits ahead. Preserve the existing
`AGENTS.md` and accumulated handoff changes outside product commits.

A final ignored-cache pass removed another 479,893 bytes (`.pytest_cache`,
`.ruff_cache`, `web/.eslintcache`, and redundant internal `web/dist`) while retaining
the verified client bundle. Total reclaimed space is 65,829,229,130 bytes (about
61.31 GiB). The immediate cleanup reading was 115,831,685,120 bytes free; the final
post-service/log verification settled at 114,881,503,232 bytes free.

## 2026-09-11 cohort attribution production-hardening checkpoint

Lucas authorized the next production-readiness steps. The opportunity engine now
materializes and validates exact per-registration-cohort downside/base/upside units in
schema v6 instead of assigning each generation/horizon total to input position zero.
Planner model-year demand and opportunity model-year drill-down use those bounded
attributions and reconcile exactly to every rounded opportunity interval. The public
generation registry is v2 and accepts the historical `Volkswagen, VW` spelling. The
snapshot validator fails closed on missing attribution, lineage mismatches, interval
ordering, or total reconciliation failures. v5 remains readable; writable v5 databases
migrate atomically to v6 with a tested deterministic constrained allocation.

The exact active v5 snapshot was copied into an immutable v6 candidate because the new
alias occurs in zero observations in this dataset; 1,377,325 assignment markers and
85,358 generation-entry markers were safely advanced to registry v2 without changing
assignment identities. The first full 21-release rebuild ended before canonical replay;
its never-published scratch data was removed. A WAL packaging-order defect in the
one-off migration path was caught by strict promotion, corrected to
projection-then-checkpoint/DELETE/VACUUM, and a stale never-active target was removed
only after confirming the active pointer still named the prior snapshot. No failed
candidate was activated.

`snapshot-a20e1c00232b3603c1a1` is now active. Its database SHA-256 is
`dd6ea0338085d9d23ac55930e61419c101ac4b00c6da5a286112e48752c85f9d`, with
1,555,677 observations and the same 100 generic-label warnings. Independent checks
found schema 6, 111,694 opportunities, 883,240 attribution rows, zero unordered
intervals, zero reconciliation failures, zero missing inputs, zero lineage failures,
matching manifest identity/checksum, and a 50-row model-year query in 265.27 ms. The
active immutable directory contains exactly the database, manifest, and validation
report.

Fresh gates passed: backend 609 passed / 14 documented environment skips / four tracked
XFAILs; Ruff; `uv lock --check`; frontend 74/74; TypeScript plus the 1,956-module Vite
build; ESLint; OpenAPI generation/diff; Python dependency audit; production npm audit
with zero vulnerabilities; current tracked OpenAI-key-shape scan with zero files; HTTP
web/API 200 with registry v2; and Chromium 21/21. The complete npm developer tree still
fails its strict high-severity audit on the dev-only Redocly -> `js-yaml` advisory;
`npm audit fix --dry-run` identifies patch releases, but no install/update was performed.
The historical OpenAI key still requires owner-confirmed revocation/rotation.

The refreshed client-release composition is running loopback-only at
`http://127.0.0.1:5173/` (child PID 3296) and `http://127.0.0.1:8000/` (child PID 1416)
with a newly generated ephemeral export token that was neither printed nor persisted.
It must not be exposed publicly. Nothing was pushed, merged, deployed, or changed in
the protected checkout. Product code and tests are committed locally as `089e97b`
(`fix: attribute opportunity demand by cohort`); the branch is 16 commits ahead of its
last-known remote state. Preserve the unrelated/pre-existing `AGENTS.md` and accumulated
handoff modifications.

## 2026-09-11 ranked generation labels and ranking-detail route

Lucas requested generation names for the cars visible in the opportunity ranking and
a dedicated explanation page for each rank. A display-only reviewed generation catalog
now augments, but does not alter, the persisted public generation registry. Nineteen
manufacturer/model profiles and twenty generation windows cover all 25 rows in the
default live ranking, including both 2012 and 2020 Ford Kuga cohorts. Names, platform
codes, year windows, source URLs, and confidence reasons were checked against official
manufacturer history or launch material from Volkswagen, Skoda, Ford, Hyundai,
Citroen/Stellantis, Volvo, Audi, BMW, Opel/Stellantis, Mazda, and Honda. The UI exposes
the supporting manufacturer URL and explicitly warns that a registration-year cohort
can include transition-year stock. The old client-side masking that hid reviewed
non-Golf names was removed.

A read-only `GET /api/v1/opportunities/{group_id}` endpoint returns the exact ranked
row under the same grouping and filters as the list. The new reloadable
`/opportunities/$groupId` page preserves ranking filters in its back link and explains
the 80-point demand component, 20-point readiness component, percentile, downside/base/
upside forecast, exact/fallback/uncovered demand, and each contributing market/horizon
row. Typed API/schema/OpenAPI clients, error/retry states, responsive styling, unit
tests, and Chromium coverage were added.

Verification: after the final Kuga window addition the full backend suite passed 629
tests with 14 documented environment skips and four tracked XFAILs; the isolated
catalog and generation-mapping suite also passed 24 tests. Ruff passed.
All 75 frontend tests passed; after the final explanatory copy change the focused
opportunities suite passed 7/7, TypeScript and ESLint passed, and the 1,956-module Vite
production build succeeded. The complete opportunities Chromium spec passed 7/7,
including reload, responsive overflow, keyboard, and serious accessibility checks.
A final live check against active `snapshot-a20e1c00232b3603c1a1` returned health
`ok`, all 25 default ranked rows with non-null generation names and manufacturer
source URLs, correct Kuga Mk2/Mk3 labels, and an exact list/detail identity match. The
refreshed loopback-only app is running at `http://127.0.0.1:5173/` (PID 38396) and
`http://127.0.0.1:8000/` (launcher 31888, worker 36424); logs are
`.local/client-{web,api}-20260911.*.log`. Do not expose either listener publicly.

No snapshot, active pointer, remote, push, merge, deployment, protected checkout, or
production state changed. Preserve the unrelated/pre-existing `AGENTS.md` changes.

## 2026-09-11 opportunity detail context and estimated fleet

Lucas requested that opening a ranked Opportunity preserve the original Opportunities
panel and show the total estimated fleet, including totals per world region. On wide
screens the reloadable detail route now uses the established split-detail pattern: the
ranked Opportunities workbench remains visible on the left with the selected card
highlighted, while the selected explanation is a sticky panel on the right. Narrower
screens retain the focused single-column detail page and back link.

A new read-only `GET /api/v1/opportunities/{group_id}/fleet` contract returns estimated
active-fleet P50 units by world region and forecast horizon under the exact same group,
market, and horizon filters as the selected rank. Snapshot calculation uses the actual
selected cohort inputs and sums their unrounded `active_fleet_p50` values before one
final half-up rounding, so a model-year does not accidentally inherit the whole
generation fleet and country-level rounding cannot drift from the region total. The UI
shows a total and region breakdown separately for every horizon; it deliberately never
adds 2028 and 2031 because that would double-count vehicles across time.

The current governed source scope maps EU27/EEA/UK country markets to Europe. Explicit
canonical future region labels pass through, while any not-yet-governed geography is
shown as `Other / unclassified` instead of being guessed. The active snapshot currently
contains Europe only, so the live breakdown has one region row per horizon. For the
top live Skoda Octavia 2017 rank, the final endpoint returned Europe 102,209 vehicles
for 2028 and 86,091 for 2031 from active snapshot
`snapshot-a20e1c00232b3603c1a1`.

Backend/domain/API/snapshot tests cover exact demo totals, typed 404 behavior,
repository delegation, and cohort-specific snapshot fleet selection. Frontend tests
cover typed URL serialization, horizon totals, region rows, and detail rendering.
Chromium verifies that the wide left ranking remains visible, the selected card is
highlighted, fleet totals render, reload works, mobile/desktop overflow is absent,
keyboard access works, and no serious accessibility violations are present.

Fresh verification: focused backend opportunity/snapshot suite 32/32; full backend
631 passed, 14 documented Windows/real-snapshot skips, and four tracked XFAILs;
maintained Ruff scope (`src tests`) passed; all 76 frontend Vitest tests passed;
TypeScript/Vite production build passed with 1,956 modules; ESLint passed; OpenAPI was
regenerated; full Opportunities Chromium spec passed 7/7 and the final focused detail
scenario passed 1/1; `git diff --check` passed with informational CRLF warnings only.
A broader `ruff check .` was also attempted and reports 507 pre-existing legacy
Streamlit/script findings outside the maintained lint scope; none is in the modified
`src` or `tests` files. CUA/image viewing remained unavailable because the Windows
sandbox still fails with error 1344, but headless Chromium exercised the rendered UI
and produced `.local/opportunity-fleet-detail.png`.

The verified client bundle was rebuilt in ignored `.local/client-release`. The final
loopback-only composition is reachable at `http://127.0.0.1:5173/` (Vite PID 38396)
and `http://127.0.0.1:8000/` (API launcher PID 3516, worker PID 33200). The API is
run through ignored `.local/run_client_api.py`; re-resolve process IDs in a later
session rather than assuming they remain stable, and do not expose either listener
publicly. No snapshot,
active pointer, production deployment, protected checkout, remote, push, or merge was
changed. The work remains uncommitted alongside the preceding generation-detail work;
preserve all pre-existing modifications.

## 2026-09-11 product visual-system refinement

Lucas approved the opportunity-detail functionality and requested a design pass that
no longer looked vibe-coded. The frontend now uses a restrained analyst-product system:
a deeper neutral navigation rail with consistent Lucide icons and a clear active state;
warm neutral page surfaces; a single teal decision accent; stronger type hierarchy;
tabular numerals; compact rectangular evidence/status treatments; subtle borders and
shadows; and a consistent reduced-radius component language. The browser theme color
was aligned with the new navigation rail.

The Opportunities experience received the most focused refinement. The hero and
methodology areas now have deliberate information hierarchy, summary metrics read as
a compact KPI strip, and the former stack of generic rounded cards is presented as a
dense ranked decision table. Selected rows use an inset state instead of a glow. Row
actions use consistent iconography, and repeated paragraph-length scoring copy was
reduced to a concise demand/readiness split plus percentile; the complete transparent
method remains in the methodology section and dedicated detail page. The detail panel
uses the same restrained surface, metric, score, progress, fleet, and contribution
patterns. Mobile retains the single-column flow and desktop retains the established
wide split-detail behavior.

Changed for this pass: `web/src/app/AppShell.tsx`, `web/src/app/styles.css`,
`web/src/features/opportunities/OpportunityRanking.tsx`, `web/index.html`, and the
corresponding concise-score expectations in `web/tests/opportunities-page.test.tsx`.
No API, forecast, snapshot, data, routing, or production behavior changed.

Fresh verification: all 76 frontend Vitest tests passed; focused app-shell and
opportunity tests passed 13/13; TypeScript plus both normal and ignored client-release
Vite builds passed with 1,956 modules; ESLint passed; the three-test cross-page
responsive Chromium suite passed at 390px and 1440px; the opportunity Chromium suite
initially passed six functional/responsive cases and found one 4.34:1 freshness-note
contrast issue, which was corrected by darkening the muted text token; the focused
keyboard/WCAG rerun then passed with zero serious or critical violations. Playwright
captured `.local/review/opportunities-mobile.png` and
`.local/review/opportunities-desktop.png`; both were reviewed for hierarchy and
overflow. Native CUA remained unavailable because of the known Windows sandbox error
1344, so browser automation and captured renders were used instead. `git diff --check`
passed with informational CRLF warnings only. After the final score-copy tightening,
the focused 13 tests and the refreshed 1,956-module client-release build passed again.

The ignored `.local/client-release` bundle was refreshed. The loopback-only development
frontend remains reachable at `http://127.0.0.1:5173/` and hot-reloads these source
changes; the API remains healthy at `http://127.0.0.1:8000/`. Re-resolve process IDs in
a later session and do not expose either listener publicly. Nothing was pushed, merged,
deployed, committed, or changed in the protected checkout, active snapshot, or remote.

## 2026-09-11 TailAdmin human-template redesign

Lucas found the custom refinement still visibly AI-designed and explicitly asked to use
a suitable human template from the internet. The prior custom visual direction is now
superseded by a selective adaptation of TailAdmin's free React dashboard template,
referenced at upstream commit `21dc917cb6cb22b5f1d12e5af57359a849d19aa8`.
TailAdmin was selected because its published free template is made for React 19,
TypeScript, and Tailwind CSS 4 (matching ICOR's frontend stack), has a public Figma
community design, and is MIT-licensed. The source reference remains ignored under
`.local/template-reference/tailadmin`; no dependency or demo application was imported.

`web/src/app/tailadmin-adaptation.css` maps the template's actual gray and brand scales,
shadows, radii, white sidebar/sticky-header shell, navigation states, panel anatomy,
KPI-card proportions, badge/button treatments, table-like ranking rows, and responsive
spacing onto ICOR's existing semantic components. `OpportunitiesPage.tsx` now uses
meaningful Lucide icons in the three KPI cards, following the template's metric-card
pattern. `web/src/main.tsx` loads the adaptation after the legacy stylesheet so the
licensed template layer is isolated and reviewable. `web/THIRD_PARTY_NOTICES.md`
records the upstream source, exact revision, copyright, and full MIT license. Existing
ICOR content, forecast logic, API contracts, routing, and interactions remain intact.

Fresh verification: all 76 frontend Vitest tests passed and the final focused shell/
opportunity rerun passed 13/13; TypeScript and the 1,957-module Vite production build
passed; ESLint passed. The combined opportunity and cross-page responsive Chromium run
passed nine functional/responsive cases at 390px, 1440px, and the wide split-detail
layout, then reported only two template-palette contrast classes (small sidebar section
labels and the active navigation blue). Both were darkened; the focused keyboard/WCAG
rerun passed with zero serious or critical violations. Desktop and mobile template
renders were captured and the desktop render was visually inspected. The final ignored
`.local/client-release` bundle was refreshed successfully.

The loopback-only frontend remains at `http://127.0.0.1:5173/` and the API remains at
`http://127.0.0.1:8000/`; do not expose them publicly. Nothing was committed, pushed,
merged, deployed, or changed in the protected checkout, active snapshot, or remote.

## 2026-09-11 sharp corners and complete governed opportunity access

Lucas requested sharp rather than rounded corners and asked to add all data to the web
app. The TailAdmin adaptation now applies a zero-radius visual contract across the app
shell, panels, metric cards, tables, navigation, buttons, inputs, badges, disclosures,
ranking/detail surfaces, and mobile equivalents. A Chromium regression explicitly
checks representative opportunity surfaces and controls resolve to `0px` radius.

For "all data," implementation follows the safe product meaning: expose the complete
client-safe governed opportunity dataset, while retaining the existing release boundary
around internal audit/admin diagnostics and not implying that unavailable proprietary
fitment data exists. The Opportunities request now uses the API's supported maximum of
100 records per page instead of 25. The page shows an explicit dataset-coverage strip
for ranked records, official model labels, and represented registrations; pagination
shows the exact current record range and total, and adds First/Previous/Next/Last
navigation so every record is reachable. Targeted model lookup remains available on
the existing Model search page rather than attempting to render the entire dataset into
one browser DOM.

Live verification against active snapshot `snapshot-a20e1c00232b3603c1a1` returned
143,894 model-year opportunity rows across 1,439 pages at 100 rows per page, 8,007
official model labels, 10,496,547 represented registrations, and EU27 plus 30 named
country markets. The request returned exactly 100 real rows. These counts supersede the
older 25-row/default-filter figures when no market or horizon filter is applied.

Changed in this pass: `web/src/features/opportunities/OpportunitiesPage.tsx`,
`web/src/app/tailadmin-adaptation.css`, `web/tests/opportunities-page.test.tsx`, and
`web/e2e/opportunities.spec.ts`. Fresh verification: all 76 frontend Vitest tests;
TypeScript/Vite production build; ESLint; the combined ten-test opportunity/responsive
Chromium run at 390px, 1440px, and wide split detail; keyboard/WCAG checks; and the
new focused sharp-corner browser regression all passed. The final ignored
`.local/client-release` bundle was rebuilt with 1,957 modules. Desktop sharp-corner
render was captured and visually inspected.

The local development frontend is listening only on `127.0.0.1:5173` (PID 38396), and
the restarted active-snapshot API is listening only on `127.0.0.1:8000` (worker PID
32068). Logs are `.local/api-sharp-data.{stdout,stderr}.log`. Do not expose either
listener publicly. Nothing was committed, pushed, merged, deployed, or changed in the
protected checkout, active pointer, snapshot contents, or remote.
## 2026-09-12 pagination, detail latency, vehicle dropdown, and launch audit

Lucas requested repairs for non-working First/Previous/Next/Last opportunity
pagination, slow opportunity detail loading, and a Model Research Brand control whose
native datalist arrow did not produce a reliable dropdown. He also requested a full
web-app review and launch preparation, while explicitly deferring deployment-method
instructions until later.

The pagination defect was a route/data race: React Query deliberately retained the
previous page as placeholder data, but button targets and disabled states were derived
from that stale response. Navigation now derives exclusively from requested route
state, prevents concurrent page changes while fetching, and announces the requested
loading page. A Chromium regression exercises Next -> Last -> Previous with delayed
responses and verifies pages 2, 1,439, and 1,438 rather than only testing a single
button.

Opportunity details now seed the selected ranking row into the exact detail query cache
before navigation, so the detail shell and explanation render immediately. The former
configuration endpoint reconstructed and serialized full planning records (about 584
KB and 11.5 seconds for a representative 54-row result) although the page needed only
six fields. A typed read-only `/api/v1/opportunities/{group_id}/contributions` endpoint
now performs a bounded aggregate and returns compact contribution rows. Against active
`snapshot-a20e1c00232b3603c1a1`, the isolated production-sized query returned 54 rows
in about 0.72 seconds. Detail, contribution, and fleet requests use AbortSignals so
navigation cancels stale work. Snapshot repository detail lookup also reuses a cached
ranking page when safe (no manual coverage) instead of recalculating the full score CTE.

Model Research Brand and Model controls are now real dependent HTML selects rather
than browser-dependent datalists. The API supplies all 464 governed forecastable
brands; selecting a brand requests its model options. Because the snapshot is
immutable for the process lifetime, the complete canonical option set is cached after
its initial load. The active-snapshot Volkswagen dependent lookup fell from roughly
1.2 seconds to 0.11 seconds. The UI retains a fallback for an empty/older brand catalog,
and the deterministic browser fixture plus every prior text-input journey was updated
to the typed select contract.

The broader review found and fixed stale cross-page browser journeys, the missing E2E
brand fixture, request cancellation, empty-catalog compatibility, and actual mojibake
in opportunity pagination/test copy. No additional serious/critical accessibility,
responsive overflow, dependency vulnerability, contract drift, maintained-code lint,
or secret-in-current-tree finding remains. The release checklist now correctly treats
the historically exposed OpenAI key as a mandatory pre-internet-launch revocation and
rotation action; deleting it from the current tree is insufficient because it remains
in Git history.

Fresh final gates on the finished code: backend 633 passed, 14 documented Windows or
real-snapshot skips, and four tracked characterization XFAILs; frontend 76/76 passed;
full Chromium 25/25 passed, including pagination, detail reload, both dropdown paths,
keyboard access, WCAG serious/critical checks, and 390/1100/1440 responsive coverage.
Ruff, ESLint, TypeScript, Vite production build (1,957 modules), OpenAPI regeneration,
`uv lock --check`, `git diff --check`, and the broken-text scan passed. `pip-audit`
reported no known third-party Python vulnerabilities (the local ICOR package is not on
PyPI); production `npm audit` reported zero vulnerabilities. The required active-
snapshot forecast benchmark passed for 16,239 series using
`validated-recency-damped-ensemble-v2`: WAPE 0.503518 versus 0.628230, a 19.85% relative
error reduction.

A fresh verified client bundle was built with
`VITE_ICOR_CLIENT_RELEASE=verified` at ignored `.local/client-release`. Fresh-process
snapshot integrity verification of the 9.2 GB ledger takes roughly two minutes before
the server binds; a future hosting plan must allow an adequate startup window and use
persistent evidence storage. This is cold-start behavior, not per-detail latency.

The stale loopback API was stopped and a current-code replacement is running under uv
launcher PID 25792 and worker PID 10948 with logs
`.local/api-final.{stdout,stderr}.log`. It binds only to `127.0.0.1:8000`;
`/api/health` returned `ok`, and the new contribution endpoint returned 54 rows for the
top Skoda Octavia 2017 opportunity. Re-resolve PIDs in a later session rather than
assuming they persist. The frontend development listener remains loopback-only. No
internet deployment, public port, active snapshot mutation, protected checkout change,
commit, push, merge, or remote action was performed. The application is technically
packaged for review, but public launch remains blocked until Lucas revokes/rotates the
historically exposed key and explicitly authorizes a deployment approach.

## 2026-09-12 final client-release blocker repair and quarterly search assessment

Lucas asked whether an OpenAI API key could support a quarterly model/model-year sales
refresh capped at 20 USD per run, requested that no refresh be run or built yet, and
asked for the rest of the client app to be made launch-ready. Official OpenAI
documentation checked on 2026-09-12 confirms that Responses API web search is currently
priced at 10 USD per 1,000 calls plus search-content/model tokens, and that projects can
have custom rate and spend limits. A quarterly run below 20 USD is technically
practical, but a platform project monthly hard limit is not by itself a per-run cap.
Any future implementation must therefore use a dedicated project/key plus an
application-side preflight estimate, metered counters, a conservative stop threshold
below 20 USD, and post-run usage reconciliation. No API request or sales search was
performed and no quarterly workflow was implemented.

The model must not be treated as the source of sales truth. A trustworthy future
workflow should use web search only for discovery, restrict retrieval to official
registration authorities and manufacturer or other governed primary publications,
retain every source URL/release/checksum and measure definition, reject ambiguous
model/model-year matches, reconcile overlaps without averaging conflicts, and promote
an immutable candidate snapshot only after automated validation and human review.
Results can be decision-grade where official model-year evidence exists; unavailable,
conflicting, estimated, registration-year, sales-year, and model-year values must stay
distinct. Broad web-derived numbers without that evidence chain would not be
trustworthy.

The verified client-release deep-link blocker found in the preceding audit is repaired.
ClientReleaseMiddleware now allows the client-facing /opportunities/{group_id} SPA
route while continuing to deny registrations, evidence, unknown/internal routes, and
non-read-only surfaces. The authenticated preview regression now covers a percent-
encoded opportunity detail URL plus continued denial of /registrations,
/opportunities-internal, and an unknown API. The active snapshot validation document
was synchronized to snapshot-a20e1c00232b3603c1a1 and the freshly rerun benchmark:
16,239 series, production WAPE 0.503518 versus 0.628230, a 19.85 percent relative
improvement.

Fresh post-fix verification: focused authenticated preview suite 48 passed with one
Windows symlink skip; preview Ruff scope passed; full backend 633 passed, 14 documented
Windows/real-snapshot skips, and four legacy characterization XFAILs; frontend 76/76
passed; and the verified VITE_ICOR_CLIENT_RELEASE=verified bundle rebuilt successfully
with 1,957 modules into ignored .local/client-release. The same-session pre-fix full
Chromium suite had passed 25/25; no frontend code changed in this repair. git diff
--check passed with informational CRLF warnings only. The loopback API and frontend
remain on ports 8000 and 5173; the running API process predates the middleware edit, so
restart it before any new local authenticated release smoke test. Nothing was committed,
pushed, merged, deployed, exposed publicly, or changed in the active snapshot.

The code and release bundle are now prepared for the next launch stage except for the
owner actions already documented: revoke/rotate the historically exposed key before
internet exposure, commit and push the reviewed development work, select and authorize
hosting, configure named reviewer/session/export secrets, provide persistent storage
for the 9.2 GB evidence ledger and its cold-start verification window, then perform the
authenticated owner and public HTTPS smoke tests. Do not promise a live URL until those
operational steps pass.

## 2026-09-12 bounded quarterly official-source research implementation

Lucas superseded the earlier request not to build the quarterly refresh and authorized
implementation, with the intent that the client API key be the remaining credential. A
new isolated discovery workflow now searches quarterly for newer EEA, UK DfT/DVLA, KBA
FZ10, and French SDES official vehicle-registration/fleet releases. It uses one OpenAI
Responses API request, defaults to the explicitly requested `o4-mini`, enables web search
for discovery, requests strict structured output, and validates every returned HTTPS URL
against the target publisher domain. It never downloads, ingests, promotes, or modifies an
active evidence snapshot; every immutable JSON result is marked pending human review.

The workflow is implemented in `src/icor/research/quarterly.py` with the CLI
`scripts/run_quarterly_source_research.py`, focused tests in
`tests/research/test_quarterly.py`, operating guidance in
`docs/QUARTERLY_SOURCE_RESEARCH.md`, and the scheduled GitHub Actions workflow
`.github/workflows/quarterly-source-research.yml`. The schedule is 06:17 UTC on January,
April, July, and October 1 and retains each report artifact for 120 days. CI lint scope now
includes the CLI. Actual scheduling becomes active only after this work is reviewed,
committed, pushed/merged into the repository default branch, and the dedicated project key
is added as the `OPENAI_API_KEY` Actions secret. No secret was created or stored.

The request is bounded to one response, 40 web-search tool calls, and 16,000 output tokens;
response storage is disabled. At official prices checked on 2026-09-12, the conservative
single-request ceiling is USD 0.6904 against the requested USD 20 run budget. This is not
an upstream billing guarantee: the client should use a dedicated OpenAI project with its
own monthly budget/alerts and reconcile the report estimate with platform usage. Official
OpenAI documentation now labels `o4-mini` deprecated and succeeded by GPT-5 mini. The
requested model remains the default; `OPENAI_RESEARCH_MODEL` is an explicit escape hatch
if the client project lacks access.

Fresh verification: focused research tests passed 8/8; new-code Ruff passed; the no-key
dry run reported `api_called: false` with the USD 0.6904 ceiling; an offline closed-loopback
SDK probe reached transport timeout, proving OpenAI Python 1.109.1 accepted the full request
shape without making an external request; full backend verification passed 641 tests with
14 documented Windows/real-snapshot skips and the four tracked characterization XFAILs;
and `git diff --check` passed with informational CRLF warnings only. No paid API request,
source search, snapshot mutation, server restart, deployment, commit, push, merge, or
protected-checkout change occurred.

Forecast-quality interpretation remains unchanged: the production registration forecaster
has newest-two-year WAPE 0.503518 versus 0.628230 for the replaced selector, a 19.85 percent
relative improvement across 16,239 eligible series. A 50.35 percent WAPE is still not high
accuracy, and the final windshield-demand layer remains assumption-led. A more complex ML
challenger must not be promoted without multi-year replacement outcomes/fitment truth and a
strict rolling-origin plus locked-holdout win over the current baseline.

## 2026-09-12 exploratory country-aware ML registration benchmark

Lucas asked to test an ML model that includes region. The governed snapshot contains
country/model registration outcomes but no country-level windshield replacement
outcomes, weather/road exposure, mileage, or fitment outcomes. The experiment therefore
tested geographic signal for registration forecasting only; it did not claim to train
or validate a north/south windshield-breakage effect.

An ignored local experiment at `.local/benchmark_regional_ml.py` used scikit-learn
1.7.2 `HistGradientBoostingRegressor` with absolute-error loss and a fixed seed.
Scikit-learn was installed only into the local development virtual environment and was
not added to production dependencies. Features were country (categorical), forecast
step, target year, history length, latest registrations, recent three- and five-year
means, recent five-year slope, recent three-year standard deviation, log versions of
latest/three-year mean, and the production recency/damped prediction. An ablation used
the same features and hyperparameters without country. Training used 127,865 earlier
rolling-origin examples. The outer test held out the newest two years, giving 32,478
points across 16,239 series and 29 countries in active snapshot
`snapshot-a20e1c00232b3603c1a1`.

Two identical fixed-seed runs produced: production WAPE 0.503518; ML without country
0.498622; ML with country 0.495473; and the regional model with test-country labels
shuffled 0.498232. The regional challenger reduced WAPE by 0.008045 absolute / 1.60%
relative versus production and by 0.003149 absolute / 0.63% relative versus ML without
geography. It beat production in 24 of 29 country slices but worsened Germany, France,
Cyprus, Italy, and Lithuania. The ignored result artifact is
`.local/regional-ml-benchmark.json`, SHA-256
`1423B2CD5434367E3E35C41106B72AE49B20F46E158301E9971F2F5603F2A570`.

Do not promote this challenger: the gain is small, it is not a windshield-outcome
validation, and it lacks a preregistered locked holdout, interval calibration, and
production dependency/test review. Keep `validated-recency-damped-ensemble-v2` in
production. A windshield-hazard ML test needs governed exposure and replacement events
by geography/time plus vehicle age/model/fitment, mileage, weather/freeze-thaw/hail, and
road/grit variables with consistent measure definitions.

## 2026-09-12 complete European authority scope and residual challenger

Lucas requested as many European official authorities as possible within the existing
USD 20 quarterly run budget and asked for the best practical forecast improvement.
The quarterly discovery registry now covers 33 governed targets: the EEA EU-level
release, all 27 EU member states, the UK, and all four EFTA states. The former single
broad response is replaced by four regional batches so every authority receives
explicit search attention. Each batch permits 24 web-search calls and 12,000 output
tokens, validates candidates only against that batch's exact publisher/domain
allowlist, disables response storage, and is capped at 24 candidates. Reports retain
all response IDs and batch metadata. A cumulative metered check runs after every batch.

Official OpenAI documentation was rechecked on 2026-09-12: o4-mini remains USD 1.10
per million input tokens and USD 4.40 per million output tokens, web search remains
USD 10 per 1,000 calls plus search-content tokens, the model supports a 200,000-token
context and 100,000 maximum output tokens, and it is deprecated/succeeded by GPT-5
mini. The explicitly requested o4-mini default remains unchanged. The new complete-run
conservative ceiling is USD 2.0512, leaving substantial headroom under USD 20. No paid
API call or real quarterly search was run.

The ignored regional benchmark was extended with two challengers. A dependency-free
per-series median residual correction worsened WAPE from 0.503518 to 0.559946 and was
rejected. A country-aware histogram-gradient-boosting residual model retained the
production forecast as its anchor and learned only corrections from 127,865 earlier
rolling-origin examples. On the same 32,478 newest-two-year test points it reached
WAPE 0.468392, a 6.98 percent relative reduction versus production and materially
better than the raw regional model's 0.495473. The result artifact is
`.local/regional-ml-benchmark.json`, SHA-256
`D831F1AA002B8BE762FB2225FA826D56E700FC552D90B8E4A0E48FDC3224404B`.

Do not promote the residual challenger yet. The newest-two-year test has now informed
model development, so it is no longer a pristine holdout. The required path is:
formalize deterministic snapshot-build training/artifact handling; test multiple
earlier temporal cutoffs; enforce country-level regression and interval-calibration
gates; then evaluate exactly once on a newly frozen unseen source release. Registration
accuracy is also distinct from windshield-demand accuracy. Final hazard forecasts need
ICOR's governed replacement outcomes and fitment truth plus vehicle age, mileage,
weather/freeze-thaw/hail, and road/grit exposures.

Fresh verification: focused research tests passed 8/8; focused new-code Ruff passed;
the no-key dry run reported 33 targets, four batches, `api_called: false`, and the USD
2.0512 ceiling; full backend verification passed 641 tests with 14 documented
Windows/real-snapshot skips and four tracked characterization XFAILs; and `git diff
--check` passed with informational CRLF warnings only. A broad `ruff check src scripts
tests` additionally exposed 486 pre-existing legacy-script lint findings outside the
maintained CI scope; none were introduced or modified in this work. No snapshot,
production forecaster, server, deployment, commit, push, merge, protected checkout, or
remote state was changed. Existing loopback server state was not revalidated.

## 2026-09-12 strict market-context ML validation, compact free data, and cleanup

Lucas asked to improve the machine-learning model, identify needed data and acquire
free compact inputs, and reclaim app-related disk space. Production remains on
`validated-recency-damped-ensemble-v2`: changing it would have been scientifically
unsafe because the newest holdout had already influenced earlier challenger
development and the active snapshot is precomputed. A new ignored local experiment,
`.local/benchmark_market_context_ml.py`, instead tested a country-aware histogram
gradient-boosting residual correction with a conservative 50 percent blend. Its
features use only forecast-origin information: each series' history, country-market
history, cross-country vehicle history, and country/vehicle market shares.

The benchmark initially used relative series endpoints; review caught that this mixed
calendar periods, and that result was discarded. The corrected evaluation uses strict
global folds in which all training targets precede the test period. Across 16,239
eligible series, the 50 percent market-context blend improved WAPE versus production
in every fold: 0.611754 versus 0.654037 (6.46 percent) for 2020-2021; 0.356360 versus
0.375763 (5.16 percent) for 2022-2023; and 0.351307 versus 0.360313 (2.50 percent) for
2024-2025. The immutable result is
`.local/market-context-ml-benchmark.json`, SHA-256
`E9D5326095337AA5B1AE295601015D8679312CC5114AE4D58710E952BBCF8BBC`.
This is a materially better registration challenger, not proof of better windshield
replacement forecasts. It must remain out of production until it wins once on a
newly frozen unseen official release and receives deterministic training/artifact,
interval-calibration, dependency, and snapshot-build integration review.

Four compact World Bank World Development Indicators were downloaded keylessly for
the 29 forecast countries and 1995-2025 into ignored
`.local/external-features/world-bank-wdi-20260912`: constant-price GDP per capita,
population, unemployment, and consumer inflation. The four official JSON responses
total 804,448 bytes. WDI is public and CC BY 4.0; only lagged/origin-available values
were tested. The WDI augmentation did not consistently beat the internal
market-context model (at the chosen 50 percent blend: 5.96, 4.72, and 2.43 percent
fold improvements versus 6.46, 5.16, and 2.50 percent without WDI), so it was
correctly rejected from the challenger rather than adding complexity.

The decisive missing data for windshield-demand ML remains multiple years of governed
replacement events joined to vehicle generation/model year and exact windshield
configuration/SKU, with exposure denominators. Useful covariates are vehicle age,
mileage, geography/date, weather/freeze-thaw/hail, and road/grit exposure. Official
registrations, macro indicators, and climate aggregates are free or low-cost and can
be compactly aggregated, but no free public dataset located provides the necessary
vehicle-to-windshield replacement outcome truth. ICOR's one reliable proprietary year
is suitable for limited calibration and a future temporal holdout, not long-run
training by itself.

Disk cleanup verified that candidate
`.local/evidence/candidates/snapshot-a20e1c00232b3603c1a1` had the same snapshot ID
and database SHA-256 as the active snapshot copy, then removed only that duplicate,
reclaiming 9,239,546,168 bytes (8.605 GiB). The active 8.6 GiB snapshot and older
rollback snapshot were preserved. Removing 82 regenerable pytest/E2E/lint/build
directories reclaimed another 231,645,208 bytes (220.9 MiB). Total `.local` usage
fell from 26.559 GiB to 17.740 GiB.

Fresh verification: production benchmark still reports 16,239 series, WAPE 0.503518
versus legacy 0.628230, and 19.85 percent relative reduction on active
`snapshot-a20e1c00232b3603c1a1`; focused registration forecast tests passed 4/4;
the corrected challenger script compiled; and both loopback listeners returned HTTP
200 at `127.0.0.1:8000/api/health` and `127.0.0.1:5173/`. No production source,
active snapshot, remote, protected checkout, deployment, commit, push, or merge was
changed.


## 2026-09-12 strict ICOR registration model race

Lucas instructed Codex to obtain the best possible ML forecast using free data. A
repository-selection error initially sent one isolated commit to the unrelated Trading
repository; it was immediately neutralized there by revert commit `2469069`. No Trading
application behavior or live state remains changed by that work. All subsequent work
was anchored exclusively to `C:\Users\LucasCravoVERISSIMO\icor-webapp-development`.

The existing ignored market-context experiment was extended with a preregistered race
across six scikit-learn residual learners: three absolute-error histogram-gradient
boosters of different capacity, one squared-error histogram booster, Extra Trees, and
Random Forest. Candidate parameters and residual blend values (0.25, 0.50, 0.75, 1.00)
were declared before reading the 2024-2025 confirmation. Selection used pooled
2020-2023 WAPE and required every eligible candidate to beat production in both
development folds.

The selected model was the flexible country-aware absolute-error histogram booster
(`learning_rate=0.035`, `max_iter=400`, `max_leaf_nodes=63`,
`min_samples_leaf=50`, `l2_regularization=5`) blended at 75 percent with the production
anchor. It improved WAPE from 0.654037 to 0.587175 in 2020-2021 (10.22 percent), from
0.375763 to 0.360766 in 2022-2023 (3.99 percent), and, only after selection, from
0.360313 to 0.350763 on the 14,448-point 2024-2025 confirmation (2.65 percent). It won
23 of 29 country slices; regressions remained in ES, HR, IE, IT, LT, and SI.

The reproducible ignored script is `.local/benchmark_model_race.py`. Its immutable
result is `.local/model-race-benchmark.json`, SHA-256
`27E7F866FDCBC48DB6D8CBC427CBE8D86ABEA1DADD2A4B9980554658D5C6C991`.
The script compiled successfully. It read the active snapshot in SQLite read-only mode
and did not mutate source data, the active pointer, production code, forecasts, servers,
the protected checkout, or any remote.

This is the strongest strictly selected registration challenger measured so far, not a
validated windshield-replacement model. No free public data found supplies the missing
vehicle/configuration/SKU-level replacement outcomes and exposure denominators. The
challenger was not promoted because the confirmation fold is now consumed and six
country regressions remain. The next safe gate is a newly frozen official snapshot,
followed by deterministic model-artifact integration, prediction-interval calibration,
and explicit country-level non-regression policy. Existing loopback server state was
not changed or revalidated in this pass.

## 2026-09-12 model-trust audit

Lucas asked what remains to make the ICOR model the best possible with available data
while making it trustworthy. A read-only audit confirmed that the system is a chain,
not one validated windshield ML model: registration forecasting is backtested, while
survival is constant-retention, replacement hazard is a French fleet-average proxy,
and the displayed demand interval propagates assumption bands rather than empirically
calibrated forecast errors.

The audit found a blocking validation defect in the ignored ML challenger. Its folds
are created from the current `cohort_estimate` snapshot, which already contains linear
interpolations made with values on both sides of a gap. The benchmark excludes future
forecast rows but does not exclude estimated rows or reconstruct data from source
release vintages. Consequently, an origin-period aggregate feature or held-out target
can contain an interpolation informed by a later year, and records may include source
revisions unavailable at the simulated forecast date. Exact fold composition was:
2020-2021, 4,162 of 19,636 targets estimated; 2022-2023, 3,707 of 19,386 estimated;
and 2024-2025, 1,008 of 14,448 estimated. The reported challenger WAPE therefore must
not be treated as promotion-grade evidence until the benchmark is rebuilt as-of each
release date using reconciled observed targets only.

Coverage and identity checks exposed additional trust limits. The benchmark's 16,239
eligible contiguous series represent only 29.1% of the 55,847 non-forecast series in
the snapshot, while production forecasts series with materially shorter histories.
The snapshot has 1,373,185 low-confidence generation assignments versus 4,140 high-
confidence assignments; low-confidence weights are mostly 0.35, but the challenger
loads materialized cohorts without applying assignment training weights. Validation
must therefore report performance and coverage by identity confidence, history length,
country, volume decile, horizon, and observed-versus-estimated status, and evaluate at
the finest identity level genuinely supported by source data.

No code, active snapshot, production model, server, remote, deployment, commit, push,
or merge was changed. A temporary ignored read-only audit helper was removed after
use. The next recommended implementation is a promotion-gate benchmark that uses
publication-date
vintages, observed-only targets, full-population coverage cohorts, multiple metrics,
country/segment non-regression checks, and empirically calibrated intervals. Only then
should the residual challenger be rerun and evaluated once on a newly frozen release.

## 2026-09-12 promotion-grade forecast safeguards and corrected validation

Lucas instructed Codex to follow the ICOR recommendations and maximize application
quality and forecast accuracy without making unsupported claims. The recommended
promotion safeguards are now maintained code. `src/icor/forecasting/promotion_gate.py`
defines a fail-closed contract requiring an observed-only, as-of-origin benchmark on
a newly frozen unseen snapshot; at least 80% evaluated target volume; at least 2%
relative overall WAPE improvement and improvement at every horizon; no material
greater-than-two-point WAPE regressions by country, identity confidence, history
length, or volume decile; and empirically measured 80% interval coverage within five
points. Contract validation also requires every segment family to partition the exact
evaluated volume. A challenger passing these gates is eligible for review, not
automatically promoted and not proof of windshield-replacement accuracy.

`src/icor/forecasting/snapshot_readiness.py` and
`scripts/audit_forecast_promotion.py` add a read-only active-snapshot readiness
audit. Against active `snapshot-a20e1c00232b3603c1a1`, it found 757,883 mapped
observed registration rows and the same number assigned, but only 4,140 high-confidence
versus 753,743 low-confidence generation assignments. The materialized snapshot has
95,176 interpolated and 139,770 forecast cohort rows. Interpolation is now an explicit
warning rather than a claim that the raw observations are unusable. The blocking fact
is publication history: only annual release years 2024 and 2025 were published within
the allowed historical-origin window. EEA 2010-2024 is primarily a 2026 backfill and
UK history is a cumulative 2026 release, so multiple genuine as-of forecast origins
cannot be reconstructed. The audit correctly exits nonzero with
`insufficient_as_of_publication_vintages`.

`scripts/benchmark_registration_forecasts.py` now excludes interpolated and
forecast targets, uses only reconciled observed-source cohorts, splits every history
at real gaps, and requires five observed annual values so three remain for training
and two for holdout. The corrected diagnostic retained 24,466 contiguous runs.
Production `validated-recency-damped-ensemble-v2` reported WAPE 0.697648 versus
0.829732 for the replaced mean/linear selector, a 15.92% relative reduction. This
supersedes the earlier 0.503518 versus 0.628230 result contaminated by interpolation.
`docs/FORECAST_VALIDATION.md` now states the corrected evidence and explicitly
marks prior residual-ML scores as ineligible. Production predictions were not changed:
the existing deterministic ensemble still wins its honest available baseline, while
the residual ML candidate lacks release vintages, an unseen holdout, calibrated
intervals, and segment non-regression evidence. Promoting it would reduce trust rather
than establish better accuracy.

New focused coverage comprises 25 forecasting tests. Fresh complete backend verification
passed 654 tests, with 14 documented Windows/optional-real-snapshot skips and the four
tracked characterization XFAILs. Maintained Ruff and `uv lock --check` passed;
`pip-audit` found no known vulnerability and skipped only the unpublished local
package. Frontend verification passed 76 Vitest tests, TypeScript, ESLint, the
1,957-module production build, and all 25 Chromium E2E/responsive/accessibility tests.
OpenAPI artifacts regenerated twice to identical SHA-256 values:
`E4774BE673B54A63C135CAD07A2FF00031255CB10A0FC7328DE75E566510C9D9`
for `web/openapi.json` and
`FA6B09AA4949B23F51D03CC0E75AA9C802D2742E883A84272768B7FDD06EBBD1`
for the TypeScript schema. The normal drift command remains nonzero only because the
required generated API artifacts are intentionally uncommitted relative to HEAD.

The complete npm audit initially found the development-only Redocly/js-yaml advisory.
`npm audit fix` updated only the compatible locked transitive tree; the full
audit now reports zero vulnerabilities, 76 frontend tests and the production build
still pass. A clean `npm ci` could not replace the native Rolldown binary held
by the running local Vite process (Windows EPERM); `npm install` restored the
exact lockfile tree without stopping the user's app. The verified client-mode bundle
was rebuilt in ignored `.local/client-release`.

The loopback app is healthy: `http://127.0.0.1:5173/` returned 200 and is owned
by PID 38396; `http://127.0.0.1:8000/api/health` returned 200 and is owned by
PID 10948. Do not expose these development listeners publicly. Nothing was committed,
pushed, merged, deployed, or changed in the protected checkout, active snapshot,
source releases, or remote. The decisive next accuracy input is historical release
vintages plus a newly frozen official release; final windshield-demand calibration
still requires governed multi-year ICOR replacement outcomes and fitment truth.

## 2026-09-12 free-data survival challenger and model-improvement research

Lucas authorized the best safe accuracy work possible from free internet data. The
deep-research skill was used to review authoritative public sources and produce the
cited, self-contained `docs/FREE_DATA_MODEL_IMPROVEMENT.md`. The research covers DfT
vehicle licensing, EEA passenger-car registrations/attributes, Eurostat traffic,
Copernicus ERA5-Land, current and historical France Assureurs glass frequency,
CASdatasets historical windscreen claims, and EU 2026/699 repair/maintenance access.
The report separates usable targets from contextual covariates and data whose reuse or
transfer rights are insufficient for production.

The strongest immediate free-data result is UK fleet survival. The official DfT
`df_VEH0160_UK.csv` was downloaded to ignored local path
`.local/downloads/df_VEH0160_UK-20260715.csv`: 10,092,936 bytes, SHA-256
`F5390DFB66087B4299FFFA2FE77C32FE35CF2DCBDFB0DB70D38C6D4926F7ABCE`, official URL
`https://assets.publishing.service.gov.uk/media/6a54d2eca6586e258d371d71/df_VEH0160_UK.csv`.
Cars first registrations for complete calendar years 2015-2025 were paired with the
active snapshot's governed UK DfT `df_VEH0124` end-of-year licensed-stock cohorts.

New `src/icor/forecasting/survival_calibration.py` implements a fail-closed,
exposure-weighted licensed-stock curve: age one is anchored to registrations, later
ages use same-cohort longitudinal transitions, administrative growth is capped so the
curve stays monotone, zero-registration cohorts cannot affect transitions, unsupported
ages fail, and complete cohorts can be excluded from calibration. New
`scripts/benchmark_survival_calibration.py` validates the CP1252 source schema and all
quarter values, uses Cars only, reads the active snapshot read-only, records the source
hash, and runs leave-one-registration-cohort-out scoring against production's constant
0.9444 retention. Tests are in `tests/forecasting/test_survival_calibration.py` and
`tests/forecasting/test_survival_benchmark.py`.

The completed benchmark on active snapshot `snapshot-a20e1c00232b3603c1a1` evaluated
54 aggregate UK cohort-age points at ages 1-9 covering 115,051,473 actual vehicle-years.
The candidate WAPE was 0.006715 versus 0.127699 for constant retention, a 94.74%
relative error reduction; weighted bias improved from -0.126372 to -0.004368. Exact
candidate versus baseline WAPE by age was: age 1, 0.004560/0.004255; age 2,
0.004466/0.045217; age 3, 0.006569/0.070691; age 4, 0.009267/0.130284; age 5,
0.009132/0.177136; age 6, 0.008377/0.218630; age 7, 0.006615/0.254588; age 8,
0.006360/0.287710; age 9, 0.007282/0.317497. A later defensive change excluding
zero-registration transitions cannot alter this dataset because all 2015-2025
registration totals are positive. A final redundant full benchmark rerun was manually
stopped after prolonged Windows snapshot verification; the earlier completed run is
the recorded benchmark.

This result is research-only and production was deliberately unchanged. It aggregates
all UK Cars across makes/models, uses one current revised publication rather than
historical as-of vintages, predicts administrative licensed stock rather than physical
survival, and proves neither generation-level performance nor transfer to another
country. Required promotion work is governed DfT vintage retention, chronological
as-of validation, high-volume make/model checks, and equivalent local-country stock
evidence or an explicit low-confidence transfer policy. The report proposes ICOR
operating labels of <=10% WAPE strong, <=20% useful/accurate aggregate, 20-30%
directional, 30-50% weak, and >50% not decision-grade; these are governance thresholds,
not an industry universal, and never replace bias, baseline skill or segment checks.

Fresh verification passed 9 focused new tests and all 31 forecasting tests. The full
backend suite passed 663 tests with 14 documented Windows/optional-real-snapshot skips
and four pre-existing characterization XFAILs. Maintained Ruff and `git diff --check`
passed; the latter emitted only existing LF-to-CRLF informational warnings. No active
snapshot, governed release, runtime service, production forecast, protected checkout,
remote, commit, push, merge, or deployment was changed.

## 2026-09-14 client WAPE check

Lucas asked for the current ICOR WAPE and whether it could be improved before a client
message. A fresh read-only run of
`scripts/benchmark_registration_forecasts.py --root .local/evidence` reproduced the
active snapshot result exactly: production registration WAPE 0.697648 versus 0.829732
for the replaced baseline, a 15.92% relative error reduction across 24,466 contiguous
observed-source cohort series with the newest two years held out. Snapshot and method
remain `snapshot-a20e1c00232b3603c1a1` and
`validated-recency-damped-ensemble-v2`. The benchmark still lacks sufficient
as-of-publication vintages and is diagnostic rather than promotion-grade.

There remains no validated end-to-end WAPE for windshield replacements. Registration
WAPE, UK aggregate licensed-stock survival WAPE, and final replacement-demand accuracy
measure different components and must not be blended or presented as one number. The
best survival result remains the research-only 0.006715 WAPE versus 0.127699 for the
constant-retention baseline on 54 UK aggregate cohort-age points; a redundant fresh
rerun was stopped after the active snapshot scan took several minutes because the
immutable prior result and source hash were already recorded. The final demand rate
remains the documented French 4.3026% fleet-average planning proxy with +/-20%
scenarios, not calibrated P10/P90 intervals.

The safe client wording is therefore to call the current output an evidence-backed,
assumption-led planning estimate and avoid claiming a validated windshield-demand
accuracy percentage. Improving the headline WAPE honestly requires new frozen official
release vintages for registration validation and governed multi-year ICOR replacement
outcomes plus exact fitment/SKU truth for final-demand calibration. The local API and
Vite development UI both returned HTTP 200 on 2026-09-14. They remain loopback-only and
must not be shared as client URLs. No code, active snapshot, governed source, release
bundle, protected checkout, remote, deployment, commit, push, or merge was changed.
## 2026-09-20 Claude audit and modelling plan (no code changed)

Lucas returned to this project via Claude Code after prior Codex work and asked for a full
verification, a running local app, and a plan to improve fleet size and windshield-replacement
forecasting. A stale third clone at `OneDrive - EPSA DEV\Desktop\Lucas\icor-webapp` (branch
`main`, 170 commits behind, the abandoned Streamlit app) was audited first by mistake and is
not the product; this worktree is.

Verified this session, with evidence:

- Full backend suite `uv run pytest`: **665 passed, 14 skipped, 4 xfailed** (Windows symlink
  and optional-real-snapshot skips; the four xfails are the existing characterization markers).
  `uv run python -m pytest tests/forecasting -q` passed 36.
- The local API and Vite UI were already running from an earlier session on PIDs 10948 (uvicorn
  :8000) and 38396 (vite :5173), in client-release mode — `/api/v1/opportunities` rejects any
  `group_by` other than `model_year` with `client_release_scope`. A fresh
  `run_planner_dev.py` attempt exited 1 with "Port 5173 is already in use"; the pre-existing
  processes were left untouched. Loopback only; never to be shared as a client URL.
- Live ranking sample: 143,894 rows, summary `base_units` 11,677,269,
  `exact_covered_base_units` **0**, snapshot `snapshot-a20e1c00232b3603c1a1`.
- `production_coverage` in `.local/production-coverage.sqlite3` holds **zero rows** and nothing
  seeds it, so exact ICOR readiness is structurally 0 for every row and the opportunity score
  reduces to a fleet-size percentile. Score is a rank transform
  (`demand_points = demand_percentile * 80`), so a constant hazard change cannot reorder rows.
- Generation coverage: only 525 of 143,894 groups resolve to a reviewed generation (19
  hard-coded profiles in `public_catalog.py` against 85,543 canonical vehicles).
- `uncertainty.py` feeds P10/P90 into `random.triangular(low, high, mode)` as support bounds,
  so the propagated interval is systematically **narrower** than its inputs. Build path uses 256
  draws, query path 2000; the snapshot the ranking UI reads is the 256-draw one.
- `snapshot_vehicle_forecast_repository.py:337` hardcodes
  `survival_method="constant-annual-retention-v1"` instead of reading it, so the vehicle-forecast
  API will misreport the method after any version bump. The planner channel reads the manifest
  correctly.
- Eurostat `road_eqs_carage` was fetched live and is usable free with no API key: 42 geographies,
  6 age categories, 2013-2025; **25 geographies have at least 10 years across 5 age bands**,
  including BE, DE, FR, IT, ES, NL, PL and EU27. Belgian parc 6,136,034 in 2025 reconciles with
  ACEA. It has **no make/model dimension**, so it cannot become `observation` rows
  (`original_make`/`original_model` are NOT NULL and the finalizer fails the build on any
  unassigned usable observation); it must be a checksum-pinned research/calibration input, the
  same classification already given to `road_tf_vehage`.
- Repository state: `github.com/lucascverissim0/icor-webapp` was **public**; 16 commits unpushed
  on this branch, 42 modified files (+5,022/-134) and ~19 untracked paths including
  `survival_calibration.py`, `promotion_gate.py`, `snapshot_readiness.py` and `src/icor/research/`.
  A scan of the full pending diff and all untracked files found **no secrets**; `.local/` is
  gitignored so the snapshot will not be pushed.

Decisions taken by Lucas this session: make the GitHub repository **private and then push** the
local work as a backup; improve **fleet size first, then demand**; and **acquire additional
publication vintages** so `audit_forecast_promotion.py` can exit 0 instead of 3. He confirmed the
historical OpenAI key will be rotated but not immediately, so it remains compromised and every
internet-launch gate stays closed.

The full approved plan lives outside the repository at
`C:\Users\LucasCravoVERISSIMO\.claude\plans\ok-as-you-can-mossy-shannon.md`. Its ordered next
actions are: (1) make the repo private, commit the pending work as three logical units
(opportunity drill-down; forecasting governance; quarterly source research) and push; (2) add
real tests for `CohortSurvivalModel`, add a survival injection point to
`GenerationPlanningService`, and fix the hardcoded `survival_method` literal; (3) acquire extra
EEA/DfT vintages until the readiness audit passes; (4) give the calibrated curve a monotone
parametric tail beyond its ~9-age support and a P10/P90 band derived from the
leave-one-cohort-out error distribution; (5) add Eurostat as a pinned calibration input plus a
fleet-validation harness scoring modelled parc against published national parc; (6) promote
through the gate and rebuild the snapshot; (7) fix the uncertainty propagation bug and the
256/2000 draw split; (8) decompose the hazard and anchor the European total against Belron's
published job volume; (9) integrate ICOR's real product catalogue once the client supplies it.

No code, test, snapshot, release, remote, commit, push, or deployment was changed this session.
The two development processes described above remain running and are unaffected by clearing the
conversation; stopping them requires closing their terminal or killing those PIDs.

## 2026-09-20 Part 0 execution: three commits made, push and privacy blocked on account access

Lucas asked Claude to execute the approved plan at
`C:\Users\LucasCravoVERISSIMO\.claude\plans\ok-as-you-can-mossy-shannon.md` end to end,
including making the GitHub repository private. Part 0 was executed as far as local work
allows; the two remote actions are blocked and were not attempted beyond a read-only check.

Fresh verification before committing, on the exact tree that was committed:

- `uv run pytest`: **665 passed, 14 skipped, 4 xfailed in 43.95s** (Windows symlink and
  optional-real-snapshot skips; the four xfails are the existing characterization markers).
- `cd web && npm test`: **15 test files passed, 76 tests passed** (vitest 4.1.11, 45.57s).
- `uv run ruff check` over the CI file list plus the three benchmark/e2e scripts:
  **All checks passed!**
- `git diff --cached --check`: clean for every commit. One real defect was found and fixed
  while staging: `.github/workflows/quarterly-source-research.yml` ended with a blank line at
  EOF; the trailing newline was normalised before it was committed.

The 42 modified and 21 untracked paths were committed as three logical units, on
`development/windshield-demand-platform`:

- `d5a2758 feat: add opportunity drill-down and ranking catalog` — 39 files. The opportunity
  detail channel (API, schemas, service, snapshot opportunity repository, React
  `OpportunityDetailPage`), the TailAdmin styling pass and `web/THIRD_PARTY_NOTICES.md`, the
  switch from `official_public_generation_catalog` to `ranking_public_generation_catalog` in
  `registrations.py` and `snapshot_vehicle_forecast_repository.py`, the new planner brand
  filter, and the `/opportunities/` path allowance in `ClientReleaseMiddleware`.
- `af3847f feat: add forecasting promotion gate and survival calibration` — 14 files.
  `promotion_gate.py`, `snapshot_readiness.py`, `survival_calibration.py`, their tests,
  `audit_forecast_promotion.py`, `benchmark_survival_calibration.py`, the corrected
  `benchmark_registration_forecasts.py` (reconciled observed-source rows only, contiguous-run
  splitting, minimum five observed years, fail-closed on an empty denominator),
  `FREE_DATA_MODEL_IMPROVEMENT.md`, `FORECAST_VALIDATION.md`, and the restored
  `CLIENT_RELEASE.md` gate #1 requiring key revocation before any internet launch.
- `4716722 feat: add quarterly source research package` — 7 files. `src/icor/research/`,
  `run_quarterly_source_research.py`, `tests/research/`, `QUARTERLY_SOURCE_RESEARCH.md`, the
  scheduled workflow, and the `ci.yml` lint-list registration of both new scripts. `ci.yml` was
  deliberately placed in this commit rather than the previous one so CI never references
  `run_quarterly_source_research.py` before that file exists.

`docs/CLIENT_RELEASE.md` and `.github/workflows/ci.yml` each carry hunks belonging to two of
the three units. They were kept whole and assigned to the unit that owns their dominant change
rather than split mid-file, so every commit is a complete, buildable tree.

**Blocked: the repository is still public and nothing has been pushed.** The cause is account
identity, not permissions policy:

- `github.com/lucascverissim0/icor-webapp` still returns HTTP 200 unauthenticated, so it
  remains **public**.
- This machine's Chrome is signed into GitHub as **`lverissimo-01`**, not `lucascverissim0`.
  `https://github.com/lucascverissim0/icor-webapp/settings` returns "Page not found" for that
  session, so the visibility control is unreachable from this browser.
- The Git Credential Manager credential is the same wrong account.
  `git push --dry-run origin development/windshield-demand-platform` returned
  `remote: Permission to lucascverissim0/icor-webapp.git denied to lverissimo-01.` and
  `error: 403`. No objects were transferred; a dry run writes nothing.
- `gh` is not installed on this machine and no `GH_TOKEN`/`GITHUB_TOKEN` is set. Reading the
  stored credential directly was refused by the harness, which is correct.

Unblocking requires Lucas to sign in as `lucascverissim0` (in Chrome, so the visibility change
can be made, and for Git, so the push is authorised) or to grant `lverissimo-01` admin and push
rights on that repository. Until then the 20 local commits (16 pre-existing plus the four made here) are the only
copy of this work and
the leaked-key exposure window stays open.

The `.local/` snapshot remains gitignored and was not committed. The earlier no-secrets scan of
this diff still holds: the three commits contain no credentials. No snapshot, release, remote,
push, deployment, or history rewrite occurred. The development API and Vite processes from the
earlier session were not touched.

## 2026-09-21 Part B and Part C complete: directives, three model fixes, four commits

Lucas asked for the last run's results and what remained before deploying to a real host, and
asked for a directives rule about doing all necessary work and delegating to cheaper models.
The approved plan for this session lives outside the repository at
`C:\Users\LucasCravoVERISSIMO\.claude\plans\ok-great-what-are-vivid-parnas.md`.

Decisions Lucas took this session:

- **Release gate #1 is waived**: deploy to the internet with the historically exposed OpenAI
  key not yet rotated. It stays compromised; this is an explicit authorization, not a skipped
  gate, and rotation remains the only real fix.
- **Host: Fly.io**, region `ams`, deployed end to end by Claude. `flyctl` is not installed and
  `fly auth login` is blocked on Lucas.
- **Ordering: modelling first, deploy when correct.** Lucas accepted that this rules out
  shipping today; revised span is roughly 3-5 weeks.
- **Generation catalogue in scope now, sourced from Wikidata (CC0)**, chosen because
  `docs/VEHICLE_CATALOG_INTAKE.md` forbids invented labels and requires per-generation licence
  and provenance metadata. `scripts/wikipedia_gen.py` is kept as a manual spot-check fallback
  only and must never be wired into the snapshot build.
- **GitHub**: Lucas will sign in as `lucascverissim0`; the repo is then made private and the
  branch pushed. Still not done - see blockers.

### The OpenAI key is not needed by the web app (verified, not assumed)

- Every `OPENAI_API_KEY` reader is the legacy Streamlit app (`ui/app.py:215`,
  `ui/pages/02_Model_Researcher.py:310`, `scripts/script1.py:41-48`, `scripts/script2.py:36-38`)
  or the offline quarterly research module (`src/icor/research/quarterly.py:472-474`).
- `src/icor/api/` and `src/icor/preview/` never import `icor.research` - zero grep hits.
- `src/icor/` imports no `openai`. The only `openai` imports in the tree are
  `scripts/run_quarterly_source_research.py:11` and `scripts/script1.py:36`.
- A tracked-file scan for an `sk-` prefixed key literal returns no hits. The key exists only in
  commits `cbef0ed`, `d557aef`, `d444ea2`, `50fafd3`.
- Therefore the realistic path to the host is the image build copying `.git` in, and
  **`.dockerignore` does not exist**. Part G of the plan now carries five containment measures,
  including a COPY allowlist and moving `openai` out of the runtime dependency set so the SDK is
  physically absent from the container.

### Commits made this session, on `development/windshield-demand-platform`

- `9f7d362 docs: require complete work and cheapest-adequate-model delegation` - two
  `AGENTS.md` Engineering workflow bullets. The section already covered token efficiency in its
  top bullet, so this extends rather than duplicates; there was no prior rule on model choice,
  delegation, or a general completeness mandate.
- `3c91add fix: report the real survival method on the vehicle-forecast channel` -
  `snapshot_vehicle_forecast_repository.py:337` hardcoded
  `survival_method="constant-annual-retention-v1"`. It now holds a `CohortSurvivalModel` and
  reads `.method`, matching what `generation_planning.py:244` already did. Without this the two
  API channels would have disagreed after the Part D version bump.
- `45c54fe fix: propagate P10/P90 as quantiles and unify the draw count` - the important one.
  `uncertainty.py` passed input P10/P90 to `random.triangular` as the *support* of the
  distribution. A triangular's own 10th percentile sits inside its lower bound, so every
  propagated interval was narrower than its inputs: measured, with a fixed hazard and a fleet
  interval of relative width 0.222, the propagated width was **0.122, 45% too narrow**. Every
  opportunity band shown to date has been overconfident. Replaced with a split-normal quantile
  function (two half-normals sharing a median), which reproduces asymmetric input intervals
  exactly, clamped at zero. No new dependency; `statistics.NormalDist` is stdlib.
  The build path drew 256 samples and the live query path 2000, so the ranking page and the
  forecast page disagreed about the same vehicle; both now read `DEFAULT_DRAW_COUNT = 2000`.
  Measured cost of that choice: **2.13 ms/estimate at 256 draws, 16.21 ms at 2000**, i.e. about
  **4 min versus 30 min across 111,694 opportunities** in an offline build, and 16 ms on a live
  request. Correctness was chosen over build speed; the value is one constant if Lucas wants it
  revisited. `method` bumped to `quantile-matched-split-normal-propagation-v2`.
- `ab4d396 test: cover the survival model and make it injectable` - `CohortSurvivalModel` had
  two tests and no coverage of age zero, compounding, monotonicity, interval bracketing or any
  of its four validation paths. Added those, plus an optional `survival=` argument on
  `GenerationPlanningService` so a calibrated curve can replace the assumed one without editing
  the service.

### Verification (fresh output, this tree)

- `uv run pytest`: **683 passed, 14 skipped, 4 xfailed** in 191.43s. Baseline was 665 passed;
  the 18 new tests are the ones listed above and no existing test regressed.
- `cd web && npm test`: **15 test files passed, 76 tests passed** (vitest 4.1.11).
- `uv run ruff check` on every file touched: all checks passed.
- `git diff --cached --check` clean for all four commits.

### Snapshot consequence

`uncertainty.method` changed, so the active snapshot `snapshot-a20e1c00232b3603c1a1` now
predates the current application method. The corrected, wider intervals do **not** reach the UI
until Part F rebuilds and promotes a snapshot. Until then the ranking still shows the
45%-too-narrow bands. Do not show the current build to a client as if the intervals were fixed.

### Blockers, unchanged and both on Lucas

1. **The repository is still public and nothing is pushed - now 24 local commits.** This
   machine is signed in as `lverissimo-01`; the repo belongs to `lucascverissim0`. These 24
   commits remain the only copy of the work.
2. **`flyctl` is not installed and `fly auth login` has not been run.** Docker and `gh` are also
   absent; Fly's `--remote-only` build means Docker is not needed.

### Next actions, in order

1. Part A: Lucas signs in as `lucascverissim0`; make the repo private; push the 24 commits.
2. Part D: acquire EEA/DfT vintages until `scripts/audit_forecast_promotion.py` exits 0; add the
   monotone tail and the leave-one-cohort-out P10/P90 band; pin Eurostat `road_eqs_carage` as a
   calibration input; build the fleet-validation harness.
3. Part E: Wikidata generation catalogue. Revised estimate **9-16 working days**, dominated by
   manual curation. A design pass established that the root cause of "Golf needed 20 spellings"
   is that `identity.py` never splits engine/trim tokens, so a tokenization tier goes in the
   matcher and the identity-layer fix is recorded as accepted debt. **Open question for Lucas
   before curation starts:** re-baseline the success target as volume-weighted coverage rather
   than raw group count, because intake gate 1 is not reachable against 143,894 raw groups by
   any free source.
4. Part F: rebuild and promote a snapshot; confirm the new `survival_method` and
   `uncertainty_method` appear in both API channels.
5. Part G: `.dockerignore`, Dockerfile, `fly.toml`, host-agnostic runner replacing the
   `CODESPACES` gate at `src/icor/preview/runner.py:22`, HSTS and `X-Robots-Tag` headers, then
   deploy.

No snapshot, release, remote, push, deployment, or history rewrite occurred this session. No
long-running local process was started.

## 2026-09-21 Part A pushed, Windows CI root-caused, survival provenance fixed

Lucas asked for a Git auth pop-up, then confirmation that the web app works, then the
forecasts made as accurate as possible including ingesting more data. Deployment stays
deferred; `flyctl` cannot be installed on this machine. The approved plan for this session
is outside the repository at
`C:\Users\LucasCravoVERISSIMO\.claude\plans\ok-give-me-the-cozy-crab.md`.

Decisions Lucas took this session:

- ICOR's proprietary fitment catalogue and replacement history are **not available yet**,
  so the windshield hazard stays explicitly assumption-led.
- The generation catalogue is taken to **volume-weighted coverage** (~80% of forecast
  volume), not the unreachable full raw-group coverage.
- The deploy host is **decided later**; only host-agnostic readiness work is in scope.
- **Push first, make the repository private afterwards.** The repository was still public
  at the time of the push and remains so.

### No auth pop-up was needed; the earlier 403 was misdiagnosed

`git-credential-manager github list` returns **both** `lucascverissim0` and
`lverissimo-01`. The recorded
`Permission to lucascverissim0/icor-webapp.git denied to lverissimo-01` was caused by the
remote URL carrying no username, so Git offered the wrong account's token - not by a
missing or expired credential. `GCM_INTERACTIVE=never git push --dry-run
https://lucascverissim0@github.com/...` succeeded with no prompt.

`origin` is now `https://lucascverissim0@github.com/lucascverissim0/icor-webapp.git` so the
account is pinned. **27 commits are pushed**; the branch is level with
`origin/development/windshield-demand-platform` at `f1c0582`. The work is no longer a
single local copy.

`api.github.com/repos/lucascverissim0/icor-webapp` still returns 200 unauthenticated:
the repository is **still public**, and the leaked key remains in commits `cbef0ed`,
`d557aef`, `d444ea2`, `50fafd3`. Changing visibility needs Chrome signed in as
`lucascverissim0`; it is currently signed in as `lverissimo-01`.

### Root cause of the windows-latest CI failure, red since at least 2026-09-01

Verified, not inferred. `stage-release` verifies
`tests/fixtures/sources/sample-registration.csv` against its manifest's `artifact_bytes`
(128) and `sha256` (`098d71a1...f715fb`). The repository had **no `.gitattributes`**, and
Git for Windows defaults to `core.autocrlf=true`, so `actions/checkout` on windows-latest
rewrites that artifact as **131 bytes** of CRLF with digest `b4e126ea...a954ff`. Staging
then raises `ReleaseIntegrityError`, which `build_evidence_snapshot.py:323` maps to
**exit code 2** - exactly the `assert 2 == 0` seen in three clean-room integration tests
(`test_build_rejects_unregistered_parser_with_typed_safe_output`,
`test_failed_snapshot_validation_exits_three`,
`test_loader_failure_is_sanitized_without_raw_row_output`).

It never reproduced locally because this worktree predates `core.autocrlf` being set:
`git ls-files --eol` still reports `i/lf w/lf` for the fixture.

Fixed in `3d53ac3` by pinning only `tests/fixtures/** -text`. A repository-wide
`text=auto` was rejected: twelve legacy `data/*.txt` files are stored with CRLF in the
index and feed the legacy Streamlit scripts. A regression test now checks the fixture's
bytes against its manifest so a future CRLF checkout names its own cause.

### Survival provenance defect, fixed in `f1c0582`

`CohortSurvivalModel.method` and `.assumption_ids` were **class** attributes, so every
instance reported the default 0.92/0.9444/0.965 retentions whatever it was constructed
with. `assumption_ids` is written into `opportunity_estimate` in the immutable snapshot
and surfaced through the API, so promoting a calibrated curve would have stamped
provably false provenance onto client-visible evidence. Both are now per-instance and the
IDs are derived from the retentions themselves. This is a prerequisite for C1.

`test_survival_method_is_read_from_the_model_not_a_literal` patched the class attribute,
which an instance now shadows; it patches the repository's own model instead, which is a
stronger check.

### Verification, fresh output on this tree

- `uv run pytest`: **690 passed, 14 skipped, 4 xfailed** in 165.72s. Baseline was 683; the
  seven added tests are the survival provenance set plus the fixture-integrity test.
- `cd web && npm run typecheck && npm run lint && npm test -- --run && npm run build &&
  npm run openapi:check`: all pass, **15 files / 76 tests**, exit 0.
- `uv run ruff check src tests` and the full CI script list: **All checks passed!**
- `uv lock --check`: clean. `uv run pip-audit`: **no known vulnerabilities**.
- `uv run python scripts/audit_baseline.py`: exit 0.
- `git diff --cached --check`: clean for both commits.

A first local `npm run e2e` attempt was invoked through `timeout`, which this shell
refuses; it exited 0 without running Playwright at all. Do not read that as a pass. A
genuine local run was started afterwards and its result is recorded separately.

### Still open

1. **`npm run e2e` hangs on CI, but passes locally.** A clean local run with nothing else
   loading the machine: **25 passed, exit 0, 1.1m**. An earlier local run made while the
   pytest suite was running reported 3 failures (planner, opportunities, accessibility),
   all bare 30s timeouts; that was machine contention, not a defect, and must not be
   recorded as a failing suite. On CI, run 35585580069's `Planner web (Linux)` job sat in
   `Run npm run e2e` for over an hour with every prior step green, and the 2026-09-01 runs
   show the same step `cancelled`. Unresolved, and it blocks a green CI.
2. **windows-latest `uv run pytest` still failed on `f1c0582`**, after the `.gitattributes`
   fix. `git -c core.autocrlf=true checkout-index` with the new attributes does produce the
   correct 128-byte LF artifact, so the CRLF mechanism is genuinely fixed; the remaining
   Windows failure is therefore something else and is **not yet identified**. Its log could
   not be read: the API log endpoint returns 403 unauthenticated, and the Actions job page
   streams while a job is running so it never reaches document_idle for the browser tools.
   Lucas chose to skip CI for now and keep verifying locally. Reading job logs later needs
   a fine-grained PAT with Actions: read.
3. The repository is still public and the key is still unrotated.
4. Parts C, D and E of the plan are not started. The active snapshot
   `snapshot-a20e1c00232b3603c1a1` still predates the current uncertainty method, so the
   ranking page still serves the 45%-too-narrow bands.

### Free-data research completed this session (read-only, no code)

**Publication vintages, for `audit_forecast_promotion.py` (`_MINIMUM_TEMPORAL_ORIGINS = 3`,
currently 2 eligible).** EEA's legacy per-year ZIPs are dead on the live site, and the
Wayback captures return the wrapper page rather than the binary. The usable route is EEA's
public unauthenticated **Discodata SQL API**, which still serves every year/status table
(`co2cars_2020Fv22`, `co2cars_2021Fv24`, `co2cars_2022Fv26`, `co2cars_2023Fv28`,
`co2cars_2024Fv30`, `co2cars_2025Pv31`), each with its own `sdi.eea.europa.eu` catalogue
publication date: 2022 Final published **2024-02-01** (lag ~397 days, inside the 550-day
rule), 2023 Provisional published **2024-05-29** (lag ~150 days). That supplies the third
and fourth genuine origin. Licence CC-BY 4.0. KBA FZ10 has stable per-year URLs
(`fz10_n_<YEAR>.html`, 2018 and 2023 verified live) but its reuse terms were not verified.
UK DfT keeps permanent dated quarterly release pages back to ~2014 under OGL v3.0, but
whether the attached data actually differs per release is unverified.

**Non-UK survival calibration.** Only three countries publish cohort stock at a
granularity that supports a per-age, per-make retention curve from free data:
**Netherlands** (RDW `opendata.rdw.nl/.../m9d7-ebf2`, per-vehicle microdata, CC0 - richer
than UK VEH0124 because it is not pre-binned; the "from 2017" note needs clarifying),
**Norway** (SSB Statbank table **08581**, age x make, 2008-2024, NLOD), and **Switzerland**
(BFS `px-x-1103020100_108`, make x year of first registration, ~1990-2024 - existence
corroborated but not cleanly fetched, verify before committing effort). Denmark (DST BIL8)
and Belgium (Statbel) give per-age national aggregates without make. **Germany, France,
Italy, Spain and Austria cannot**: their stock data is broad age bands, or never crossed
with make, or access-restricted (France's RSVERO is CASD-gated). Sweden has no free path.
Eurostat `road_eqs_carage` is confirmed live with exactly five age bands
(<2, 2-5, 5-10, 10-20, >=20) and no make dimension: a cross-country sanity check, **not** a
calibration source, which matches the classification already recorded for `road_tf_vehage`.

No snapshot, release, deployment, merge or history rewrite occurred.

## 2026-09-22 calibrated survival promoted; snapshot rebuild in progress

Lucas asked to end the day with the best version of the web app. He chose the larger
of two options: add the governed input the calibration needed and do a single rebuild
carrying both the corrected uncertainty propagation and the calibrated survival curve,
accepting the risk that it might not finish today.

### The constant survival model had the wrong shape, not just the wrong level

Reproduced on this tree with
`scripts/benchmark_survival_calibration.py --root .local/evidence --uk-registrations
.local/downloads/df_VEH0160_UK-20260715.csv`:

| Measure | Constant retention | Calibrated curve |
|---|---:|---:|
| WAPE | 0.131908 | **0.007013** |
| Weighted bias | -0.130606 | **-0.004712** |
| Points / vehicle-years | 55 / 117,339,249 | same |

A 94.68% relative reduction. Per-age constant WAPE runs 0.0043 at age one, 0.045 at
age two, 0.177 at age five, 0.317 at age nine, 0.344 at age ten.

The earlier record of 0.006715 versus 0.127699 on 54 points is superseded: the current
run evaluates 55 points because age ten is now supported.

The important finding is the *shape*. The constant model **over-attrits young vehicles**
(56.4% of a cohort remaining at age ten against an observed 85.4%) and **under-attrits
old ones** (31.9% at age twenty against an observed 16.7%). The errors cross over around
age sixteen, so no single retention constant could have corrected either. Observed annual
retention is roughly 0.98-1.00 through the first decade, falls to about 0.79 between ages
eleven and twenty, then rises again as durable survivors remain.

### What was built

- `extend_with_monotone_tail` continues a curve past its calibrated support at the
  geometric mean of its last five observed transitions. It fits no parametric shape on
  purpose: neither Weibull nor Gompertz reproduces the three-phase shape above. A curve
  whose recent transitions were all capped at 1.0 is rejected rather than extended into
  an immortal fleet. Not exercised by this build - maximum cohort age is 30 against a
  support of 40, verified before starting.
- `calibrate_licensed_stock_band` measures the spread of the same annual transition
  across the eleven registration cohorts available at every age and compounds the
  quantile transitions around the pooled median, which is left exactly as benchmarked.
  Compounding assumes a cohort that decays faster than the median keeps doing so; that is
  the conservative reading and matches the data. Relative band width against the constant
  model: **0.043 versus 0.237 at age five**, 0.129 versus 0.471 at age ten, and honestly
  wider at old ages, 1.79 versus 1.45 at age thirty.
- `CalibratedCohortSurvivalModel` serves that band through the interface
  `GenerationPlanningService` already expected, carrying its own method and assumption
  IDs. `reason_code` distinguishes the measured geography from every transfer, and the
  planner now asks the model for that code instead of hardcoding
  `assumption-led-survival-not-calibrated`.

### Why the curve is an artifact, not a build-time calculation

`scripts/calibrate_survival_curve.py` writes
`src/icor/forecasting/survival_curves/uk_licensed_stock.json`, carrying the three share
curves, cohort counts, the source's URL, byte count and SHA-256, and its stated
limitations. `load_promoted_survival_model()` reads that committed artifact and the build
uses it. A snapshot's curve is therefore a reviewable fact in the repository rather than
a side effect of whichever files were on disk at build time.

### The geography decision, which is the subtle part

The curve is calibrated on **UK** licensed stock (`df_VEH0124`, UK-scoped) anchored by
**UK** registrations (`df_VEH0160_UK`, 10,092,936 bytes, SHA-256
`f5390dfb...7abce`, OGL v3.0).

Two alternatives were rejected, both for concrete reasons:

- *Anchor UK stock with the GB registrations already in the snapshot.* GB excludes
  Northern Ireland, so the scope mismatch would have scaled the age-one anchor, and
  therefore every cohort at every age, by roughly 2.5%.
- *Load `df_VEH0160_UK` into the evidence set as a governed release.* GB registrations
  from the same DVLA register are already there, so a UK-wide copy would double count and
  would create a `UK` geography the product does not target. The file is read only to
  calibrate. Parser counts were derived exactly (raw 62,967 / accepted 33,589 /
  rejected 29,378) in case that decision is ever revisited.

`calibrated_geography` is therefore **UK**, not GB. Every product geography - EU27, BE,
FR, ES, NL, GB, DE, PL - receives the curve as a transfer and each cohort records
`licensed-stock-calibrated-survival-transferred-from-uk`. GB is a 97% subset of UK rather
than the same thing, so it is labelled a transfer too; that is pedantic but honest, and
the label names its source so a reader can judge the distance.

### Snapshot versioning gap, closed

`SnapshotVersions` had no `uncertainty_method` field, so a snapshot could not record which
propagation produced its intervals. `CLIENT_RELEASE.md` gate 7 therefore had nothing to
compare and could not detect that the active snapshot predated the split-normal fix. The
field now exists, defaulted so manifests written before it still load, and
`_LEGACY_VERSION_FIELDS` excludes it. `tests/api/test_completeness_api.py` passed its
versions positionally and silently shifted when the field was inserted; it now uses
keywords.

`OFFICIAL_SOURCE_VERSIONS.survival_method` is now `uk-dft-licensed-stock-band-v1`.

### Verification

- `uv run pytest`: **702 passed, 14 skipped, 4 xfailed**. Baseline at the start of the day
  was 683.
- `uv run ruff check src tests scripts/calibrate_survival_curve.py`: all checks passed.
- One transient `test_open_active_snapshot_returns_matching_manifest_and_repository`
  teardown error appeared in a single full run and did not reproduce on rerun or in
  isolation; it looks like a Windows file lock on the 9.2 GB database, not a defect.

### Commits

`52ae5e3` tail, band and model; `9606957` promotion, curve artifact and version bump;
`94c1b2a` documentation. All pushed.

### What promotion does NOT license anyone to claim

Registration WAPE (0.6976), UK licensed-stock survival WAPE (0.007013) and final
windshield-replacement accuracy measure different things and must never be blended into
one number. **Calibrating the fleet does not calibrate the rate applied to it**: the
4.3026% French insurance proxy is untouched and still assumption-led, so final demand
intervals remain assumption-led. `docs/WINDSHIELD_DEMAND_ASSUMPTIONS.md` now states the
two separately rather than declaring both uncalibrated.

### Snapshot rebuild status at the time of writing

A rebuild started at 09:41 local time covering all 21 releases of the active snapshot,
with `--build-as-of 2026-09-22T12:00:00+00:00 --deterministic-seed 20260922`, writing to
`.local/evidence/candidates/.build-82025dce027c4ac9926eecaff3caa44b`. It is the first
snapshot to carry both the split-normal uncertainty propagation and the calibrated
survival curve.

**It has not been promoted.** Until it is, the active snapshot remains
`snapshot-a20e1c00232b3603c1a1`, whose intervals are the measured 45%-too-narrow ones and
whose survival is the constant model. If this rebuild is abandoned, delete only the
`.build-` candidate directory; the active pointer and both retained snapshots are
untouched by a failed or abandoned build.

Remaining after the build finishes: inspect the candidate with
`scripts/report_snapshot_completeness.py`, promote only a zero-error candidate, confirm
`survival_method` and `uncertainty_method` appear in both the ranking channel
(`snapshot_opportunity_repository`, reads stored quantiles) and the vehicle-forecast
channel (`snapshot_vehicle_forecast_repository`, recomputes them), and re-run the client
smoke test.

## 2026-09-22 DEFECT: GB registrations are double counted, inflating GB demand up to ~2x

Found while sanity-checking the rebuilt snapshot's modelled fleet against published parc
figures. **This is a pre-existing defect present in the active production snapshot, not a
regression introduced by the survival calibration.** It is the most consequential finding
of the session and it affects client-visible numbers.

### Evidence

GB cohort registrations as reconciled in the snapshot, against actual GB new car
registrations:

| Cohort year | Reconciled | Actual | Ratio |
|---|---:|---:|---:|
| 2015 | 5,156,812 | ~2.63M | **1.96x** |
| 2018 | 4,645,967 | ~2.37M | **1.96x** |
| 2019 | 3,984,634 | ~2.31M | **1.72x** |
| 2023 | 2,132,509 | ~1.90M | 1.12x |
| 2024 | 2,114,755 | ~1.95M | 1.08x |

The inflation appears exactly in the years where two sources both cover GB, and
disappears after the UK stopped reporting to the EEA. Per-source totals confirm the two
are measuring the same vehicles:

- 2018: `eea-co2-monitoring` 2,355,350 and `uk-dft-veh0160` 2,341,505
- 2019: `eea-co2-monitoring` 2,304,560 and `uk-dft-veh0160` 2,295,409

Consequence: modelled GB fleet for 2028 from 2001+ cohorts alone is **41,129,260**, above
the entire GB car parc of roughly 33-34 million, which also excludes pre-2001 vehicles
still licensed.

### Root cause

`RegistrationReconciler.reconcile` in `src/icor/forecasting/reconciliation.py` selects one
winner **per dependency group** and then **sums across groups**:

    value=sum((item.value for item in selected), start=Decimal(0))

That is correct only when dependency groups measure *disjoint populations*. Here they
measure the *same* population from two independent publishers, which is corroboration,
not addition. The dependency-group concept conflates "sources that are correlated" with
"sources that cover different vehicles".

The same overlap was already recognised for Germany and handled: `docs/DEVELOPMENT.md`
records that EEA and KBA share a dependency group "so their overlap is not treated as
independent confirmation". UK DfT was left in its own group `uk-dvla-vehicle-register`
while EEA uses `european-passenger-car-registrations-<year>`, so the GB overlap was never
handled.

### Why it is not a one-line fix

EEA dependency groups are **per year** (`european-passenger-car-registrations-2024`),
while the single UK DfT release spans **2001-2025**. A release therefore cannot simply be
moved into the matching EEA group; the overlap is per (geography, year), but
`dependency_group` is a property of a release. Resolving this needs either a
geography-and-year-scoped overlap rule or a reconciler that understands corroboration
separately from addition. It needs design, then a rebuild, which currently costs 4h15m.

### Interaction with the survival calibration

The two defects were partially cancelling. The constant survival curve over-attrited
young vehicles, which masked part of the inflated registration input. Replacing it with
the accurate curve retains more of an input that is too large, so the GB fleet overshoot
becomes *more* visible: modelled GB 2028 fleet rises from 35,919,876 to 41,129,260.

This is expected and is not an argument against the calibration. It is an argument for
fixing the input. Lucas was shown this trade-off explicitly and chose to promote the
better model now and fix GB next.

### Until it is fixed

Do not quote any GB figure - fleet, opportunity ranking position, or demand - to a client.
Non-overlapping geographies are unaffected by this specific defect: Germany is covered by
the shared EEA/KBA group, and the sanity check on the new snapshot gives DE 43.5M and FR
23.4M for 2028, both below their national parcs and therefore plausible.
