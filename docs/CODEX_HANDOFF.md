# ICOR Web App — Durable Handoff

Last updated: 2026-08-31

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
