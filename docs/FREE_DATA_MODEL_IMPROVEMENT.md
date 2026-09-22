# ICOR free-data model improvement assessment

Date: 2026-09-12

## Executive decision

The largest immediately testable weakness is fleet survival, not the choice of a
more complicated registration algorithm. ICOR currently applies one constant 94.44%
annual retention rate to every cohort, age and country. Official UK registration and
licensed-stock data support an age-shaped empirical challenger.

On the active snapshot `snapshot-a20e1c00232b3603c1a1`, leave-one-registration-
cohort-out validation produced:

| Measure | Constant ICOR retention | UK empirical challenger |
|---|---:|---:|
| WAPE, ages 1-9 | 12.7699% | 0.6715% |
| Weighted bias | -12.6372% | -0.4368% |
| Evaluation | 54 cohort-age points / 115,051,473 actual vehicle-years | same |

This is a 94.74% relative WAPE reduction for the aggregate UK Cars licensed-stock
target.

**Promotion status, 2026-09-22: this curve is now the production survival model.**
Reproduced on the current tree it scores WAPE 0.007013 against 0.131908, a 94.68%
relative reduction, with weighted bias -0.0047 against -0.1306 across 55 cohort-age
points. It clears all four survival gates stated further down this page. The caveats
below are unchanged by promotion and still bound what may be claimed: the evidence is
one current DfT publication vintage, the score aggregates makes and models, licensed
stock is an administrative proxy rather than physical survival, and generation-level
or non-UK accuracy has not been demonstrated.

## What ICOR predicts today

The model is a deterministic evidence-and-assumption pipeline:

1. Official registration and licensed-stock releases are ingested with release,
   checksum, schema, publication-status and provenance controls.
2. Source make/model records are normalized and assigned to canonical vehicle
   generations.
3. Observed annual registrations are reconciled and gaps are explicitly estimated.
4. Future registrations use a fixed 50/50 blend of the last observation and a damped
   five-year linear trend.
5. Each registration cohort is decayed by the calibrated UK licensed-stock
   retention band (`uk-dft-licensed-stock-band-v1`). Its median is the pooled
   curve validated here; its P10/P90 come from the spread of the same annual
   transition measured across eleven registration cohorts. Until 2026-09-22 this
   step applied constant annual retention of P50 0.9444, P10 0.92 and P90 0.965.
6. Surviving fleet is multiplied by a flat windshield-replacement hazard. The current
   central assumption is `0.0606 × 0.71 = 0.043026` replacements per active
   vehicle-year.
7. Survival, hazard and registration scenarios are propagated to the opportunity
   outputs.

This is not presently an end-to-end machine-learning model. The production
registration forecaster is a guarded statistical rule; survival and hazard are fixed
assumptions. That distinction matters because adding a powerful regressor cannot
repair a structurally wrong target, weak identity mapping or missing outcome labels.

## What was implemented

`src/icor/forecasting/survival_calibration.py` adds a fail-closed empirical curve:

- age-one stock is anchored to registrations using pooled vehicle exposure;
- later retention uses same-cohort longitudinal transitions, avoiding comparisons
  between different cohort mixes;
- administrative increases from imports, relicensing or corrections are capped at
  100%, so the curve cannot grow with age;
- a complete cohort can be excluded for cross-validation; and
- unsupported ages raise an error instead of being silently extrapolated.

`scripts/benchmark_survival_calibration.py` independently parses DfT UK Cars first
registrations for complete calendar years 2015-2025, reads only the governed
`df_VEH0124` licensed-stock observations in the active immutable snapshot, removes
each evaluated cohort from training, and compares against production retention.

The exact DfT registration artifact used was 10,092,936 bytes with SHA-256
`F5390DFB66087B4299FFFA2FE77C32FE35CF2DCBDFB0DB70D38C6D4926F7ABCE`.
DfT describes `df_VEH0160_UK` as quarterly first registrations by make, generic model,
model and fuel, and `df_VEH0124` as end-of-year licensed vehicles by make/model and
year first used/manufacture.[1]

## Why the result is plausible, and why it is not sufficient

The constant model compounds 0.9444 every year, leaving only 59.92% of a cohort at
age nine. The DfT cohorts imply about 87.6% at age nine. The empirical curve therefore
corrects a large, directional undercount that grows with vehicle age.

The comparison is strong against cohort leakage: the cohort being scored never
contributes to its curve. It is an aggregate country score and can conceal weak
make/model segments. It is not a temporal publication-vintage backtest. DfT says
the database is virtually complete for licensed/SORN totals, but it is administrative;
individual fields contain errors, generic-model methodology is revised, and vehicle
records change with import, export, scrappage and keeper updates.[2] The target should
therefore be called a licensed-stock retention curve, not a physical survival curve.

The active ICOR snapshot also lacks enough historical as-of publication vintages for
promotion-grade registration validation: only 2024 and 2025 origins are currently
eligible. Archiving each future official release without overwriting old vintages is
the highest-value data-engineering action.

## Free-data landscape and recommended use

| Need | Best free source found | What it can improve | Decision |
|---|---|---|---|
| Cohort stock/survival | UK DfT `df_VEH0160_UK` + `df_VEH0124` | Age-shaped UK licensed-stock retention | **Promoted 2026-09-22**; UK-measured, transferred elsewhere with a per-cohort label |
| EU registrations and vehicle attributes | EEA new car CO2 monitoring | Country, manufacturer, type/variant/version and technical covariates | Continue governed annual ingestion; archive every vintage[3] |
| Traffic exposure | Eurostat road traffic by vehicle type and age (`road_tf_vehage`) | Country/age exposure priors | Research covariate only; it is aggregate, not model-level[4] |
| Weather exposure | Copernicus ERA5-Land | Hail, freezing, temperature and precipitation features | Add only after geographically aligned claims/replacement labels exist[5] |
| Glass-claim level | France Assureurs annual motor report | National frequency anchor and drift monitoring | Keep the current 2025 anchor of 60.6 claims per 1,000 covered vehicles and use 2024's 62.2 as history; both are all-glass claims, not windshield replacements[6] |
| Historical windscreen claims | CASdatasets French private motor data | Exploratory age/class/region hazard relationships | Research only: 2003-2004, unknown private portfolio, and reuse rights must be cleared[7] |
| Exact windshield fitment | Manufacturer repair/maintenance information | VIN/OE-part applicability and generation splits | Do not assume it is free or freely redistributable; EU rules require machine-readable independent-operator access but access conditions still apply[8] |

Open weather or macroeconomic covariates are not targets. They can only improve a
hazard or registration model when joined to leakage-safe, dated outcomes. ICOR's
previous macro residual challenger did not improve robustly across countries, so
additional generic features should not be added merely because they are free.

## What is required for a genuinely better production model

### 1. Replace the survival constant country by country

Use the UK challenger as the method template, not as a Europe-wide coefficient.
Discover official cohort-stock tables for each country, preserve source vintages, and
fit hierarchical age curves only where local data are sparse. Require monotonicity,
credible tails, and separate treatment of imports/exports when available.

Promotion gate:

- chronological, as-of-origin validation rather than one revised current file;
- WAPE below 10% on licensed stock and at least 10% better than constant retention;
- weighted bias within ±5%;
- no critical age band above 20% WAPE; and
- country and high-volume generation-level validation, or an explicit low-confidence
  transfer label.

### 2. Improve registration forecasts only with clean vintages

Retain the simple production rule as the baseline. After at least four to five annual
publication vintages are archived, compare it with ETS/damped trend, pooled
hierarchical models, and gradient boosting using only information available at each
forecast origin. Evaluate per horizon, country, generation-confidence band and demand
scale. Never train on interpolated values or later revisions as if they were observed
history.

Promotion gate:

- portfolio WAPE at or below 20%;
- critical country WAPE at or below 30%;
- at least 10% relative improvement over seasonal-naive/last-value baselines;
- weighted bias within ±10%; and
- no material regression by horizon or high-volume country.

Those are ICOR governance targets, not universal scientific cutoffs. WAPE is total
absolute error divided by total actual volume; lower is better, and it becomes
undefined when aggregate actual demand is near zero.[9] Forecast errors must be
measured on held-out future observations, not training residuals.[10]

### 3. Obtain a windshield-replacement outcome

The final demand target is windshield replacement, but the public anchor is glass
claims. A reliable hazard model requires, at minimum, dated counts of windshield
replacements and covered/exposed vehicles by geography, vehicle age and preferably
generation or OE part. Free sources currently found do not supply that modern,
model-level denominator and outcome together.

Until such labels exist:

- keep the hazard assumption visibly assumption-led;
- update the national glass-frequency anchor annually;
- separate windshield repair, windshield replacement and other glazing;
- use ERA5/traffic only for scenario sensitivity, not fitted accuracy claims; and
- report uncertainty much wider than ordinary model error.

### 4. Improve vehicle identity and fitment coverage

Registration accuracy does not guarantee part-level accuracy. ICOR needs auditable
generation boundaries and windshield/OE-part fitment dates. Use manufacturer RMI
data only under confirmed access and reuse terms; do not scrape or republish protected
catalog data. Low-confidence generation assignments must remain excluded from model
training or down-weighted and reported separately.

## WAPE interpretation for ICOR

There is no universal WAPE value at which a forecast becomes accurate. Difficulty,
horizon, aggregation, intermittency and the business loss function change the answer.
For ICOR, the following operating labels are reasonable when computed on clean,
future, observed-only data:

| WAPE | ICOR interpretation |
|---:|---|
| ≤10% | strong |
| >10% to 20% | useful/accurate at aggregate level |
| >20% to 30% | directional; investigate segments |
| >30% to 50% | weak |
| >50% | not decision-grade |

These labels never override baseline skill, bias, coverage or segment stability. A 15%
portfolio WAPE can conceal a failed country; a 35% WAPE can still add value for a
highly volatile sparse series if it decisively beats the best naive forecast. Always
publish WAPE with weighted bias, horizon results, coverage, and baseline improvement.

## Ordered execution plan

1. Preserve the current UK benchmark and its exact input hash as a reproducible
   research artifact.
2. Add governed acquisition manifests for every quarterly/annual DfT vintage and
   retain all old releases.
3. Add time-vintage tests and country-segment promotion gates for survival.
4. Research equivalent official cohort-stock releases for Germany, France, Italy,
   Spain and other priority markets; never transfer the UK curve silently.
5. Refresh the France glass-frequency baseline annually; retain the current 2025
   evidence and do not confuse claim frequency with replacement frequency.
6. Seek a lawful, reusable modern replacement-outcome dataset. Only then benchmark
   age, weather, traffic and vehicle-feature hazard models.
7. Revisit registration ML after sufficient as-of vintages exist; promote only a
   challenger that wins the untouched gate across horizons and countries.

## Sources

1. UK Department for Transport, *Vehicle licensing statistics data files*, updated
   15 July 2026: https://www.gov.uk/government/statistical-data-sets/vehicle-licensing-statistics-data-files
2. UK Department for Transport, *Vehicle licensing statistics: notes and definitions*,
   updated 29 April 2026: https://www.gov.uk/government/publications/vehicles-statistics-information/vehicle-licensing-statistics-notes-and-definitions
3. European Environment Agency, *Monitoring of CO2 emissions from passenger cars*
   data hub: https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b
4. Eurostat, road-traffic vehicle dataset and navigation entry for `road_tf_vehage`:
   https://ec.europa.eu/eurostat/web/products-datasets/-/road_tf_veh
5. ECMWF/Copernicus, *ERA5-Land hourly data from 1950 to present*, DOI
   10.24381/cds.e2161bac: https://www.ecmwf.int/en/forecasts/datasets/era5-land-hourly-data-1950-present
6. France Assureurs, *Le marche de l'assurance automobile des particuliers en 2025*,
   July 2026: https://www.franceassureurs.fr/wp-content/uploads/le-marche-de-lassurance-automobile-des-particuliers-2025.pdf;
   historical 2024 study: https://www.franceassureurs.fr/wp-content/uploads/lassurance-automobile-des-particuliers-en-2024.pdf
7. CASdatasets, *French claims for private motor*:
   https://dutangc.github.io/CASdatasets/reference/fremotorclaim.html
8. Commission Delegated Regulation (EU) 2026/699, section 6.1.2:
   https://eur-lex.europa.eu/legal-content/en/ALL/?uri=CELEX:32026R0699
9. Amazon Web Services, *Evaluating Predictor Accuracy — WAPE*:
   https://docs.aws.amazon.com/forecast/latest/dg/metrics.html
10. Hyndman and Athanasopoulos, *Forecasting: Principles and Practice*, time-series
    cross-validation: https://otexts.com/fpp3/tscv.html
