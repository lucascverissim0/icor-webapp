# ICOR Web App — Durable Handoff

Last updated: 2026-08-26

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
