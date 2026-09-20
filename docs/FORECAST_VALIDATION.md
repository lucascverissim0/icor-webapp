# Forecast validation

## Production decision

The registration forecast is
`validated-recency-damped-ensemble-v2`. It averages:

- the latest observed annual registration value; and
- a five-year linear trend whose future increments decay by 20% per step.

The 50/50 weight is fixed globally. It is not selected separately for each short
vehicle series, because that approach overfit the available histories.

## Corrected observed-source diagnostic

Run:

    uv run python scripts/benchmark_registration_forecasts.py --root .local/evidence

Against active snapshot `snapshot-a20e1c00232b3603c1a1`, the corrected benchmark:

- excludes every interpolated and forecast cohort target;
- splits each reconciled source-backed history at gaps instead of filling them;
- retained 24,466 contiguous runs with at least five annual values;
- removed each series' newest two years before model fitting;
- predicted those unseen years;
- aggregated error with demand-weighted absolute percentage error (WAPE);
- compared the production method with the replaced rolling-origin mean/linear
  selector.

Verified result:

| Method | Newest-two-year WAPE |
| --- | ---: |
| Replaced mean/linear selector | 0.829732 |
| Validated recency/damped ensemble | 0.697648 |

The relative error reduction is 15.92%. Lower is better. This supersedes the earlier
0.503518 versus 0.628230 diagnostic, which included 95,176 interpolation-derived
cohort rows in the available evaluation population. The corrected score is more
honest but remains a diagnostic, not a promotion-grade result.

Run the fail-closed readiness audit:

    uv run python scripts/audit_forecast_promotion.py --root .local/evidence

The current snapshot has only two origin-eligible annual release years (2024 and
2025), so it cannot reconstruct multiple historical forecast origins. Its EEA
2010-2024 history was acquired as a 2026 backfill, and its UK history is supplied by
a cumulative 2026 release. The production method therefore remains the safest current
baseline, but its historical method ID must not be interpreted as a claim that final
windshield demand is empirically validated.

## Challenger promotion policy

`src/icor/forecasting/promotion_gate.py` now fails closed unless a challenger has:

- observed-only targets reconstructed from evidence published by each forecast origin;
- a newly frozen holdout snapshot that was never used during development;
- at least 80% of eligible target volume evaluated;
- at least 2% overall relative WAPE improvement and improvement at every horizon;
- no greater-than-2-percentage-point material regressions by country, identity
  confidence, history length, or volume decile; and
- empirically measured 80% interval coverage within a five-point tolerance.

These thresholds are predeclared application policy. Passing them proves eligibility
for review, not windshield-replacement accuracy.

An additional exploratory rolling-origin comparison covered 566,656 forecast points.
It rejected recent mean, median, unrestricted linear trend, damped trend alone, CAGR,
and per-series candidate selection as universal replacements for the robust recency
anchor. This is why the production method is a conservative fixed ensemble rather than
an unnecessarily complex learner.

## Scientific boundary

There is no defensible universal “best algorithm in the market.” The M4 competition
found that the leading methods were combinations, while the M5 retail competition found
strong results from LightGBM when many related series and explanatory variables were
available. Those are different data regimes:

- M4 results: https://doi.org/10.1016/j.ijforecast.2019.04.014
- M5 results: https://doi.org/10.1016/j.ijforecast.2021.11.013

ICOR currently has short annual registration histories, noisy publisher model labels,
and no pre-meeting proprietary windshield replacement outcomes or explanatory features.
Introducing LightGBM, neural forecasting, or another high-capacity model now would not
create missing signal and must not be called an accuracy improvement without a superior
locked holdout result.

## Remaining calibration limits

This validation covers future registrations only. Windshield replacement hazard now
uses the public French insurance-sector anchor documented in
`docs/WINDSHIELD_DEMAND_ASSUMPTIONS.md`; it is still a cross-market planning proxy,
not a fitted ICOR outcome model. Cohort survival and exact vehicle-to-windshield
fitment remain explicit versioned assumptions. They cannot be trained or independently
validated until ICOR supplies replacement history and fitment truth. The UI and API
must continue to label final windshield-demand ranges as assumption-led planning
estimates.

When that evidence arrives, the next challenger should be a hierarchical
gradient-boosted model with model, geography, age, weather/road exposure, vehicle parc,
replacement history, and fitment features. It is promoted only if rolling-origin and
locked newest-period tests improve point accuracy and interval calibration over this
production baseline.

## Country-aware residual challenger

A 2026-09-12 read-only experiment tested a safer pooled-ML structure: retain the
production recency/damped forecast as an anchor, then train a country-aware histogram
gradient-boosted model to predict only its residual error. On the same newest-two-year
outer test (32,478 points across 16,239 series), it reached WAPE 0.468392 versus
0.503518 for production, a 6.98% relative reduction. A raw country-aware model reached
0.495473, and a dependency-free per-series bias correction worsened WAPE to 0.559946.

This identifies a historical experimental challenger, not a production promotion. The newest-two-year
test has now informed model development and is no longer a pristine locked holdout.
Before integration, the residual approach must pass multiple earlier temporal cutoffs,
country-level regression guardrails, calibrated prediction-interval tests, a newly
frozen unseen snapshot, deterministic artifact/version handling, and production
dependency review. Until those gates pass, `validated-recency-damped-ensemble-v2`
remains the honest production choice.


## Strict multi-model confirmation race

A subsequent 2026-09-12 experiment compared six fixed residual learners using only
forecast-origin registration and market-context features. Model and blend selection
used 2020-2023 only; a candidate was eligible only if it beat production in both
two-year development folds. The selected challenger was a country-aware histogram
gradient booster with absolute-error loss, 63 leaves, 50 minimum samples per leaf,
and a 75% residual blend.

| Period | Production WAPE | Challenger WAPE | Relative reduction |
| --- | ---: | ---: | ---: |
| 2020-2021 development | 0.654037 | 0.587175 | 10.22% |
| 2022-2023 development | 0.375763 | 0.360766 | 3.99% |
| 2024-2025 confirmation | 0.360313 | 0.350763 | 2.65% |

The confirmation contained 14,448 points across 29 countries. The challenger improved
23 country slices and regressed in Spain, Croatia, Ireland, Italy, Lithuania, and
Slovenia. Extra Trees, Random Forest, squared-error boosting, and more conservative
boosting were all inferior under the predeclared development rule. The immutable local
artifact is `.local/model-race-benchmark.json`, SHA-256
`27E7F866FDCBC48DB6D8CBC427CBE8D86ABEA1DADD2A4B9980554658D5C6C991`.

This was the strongest strictly selected registration challenger in the contaminated
materialized-cohort experiment, but it is not eligible for promotion. The 2024-2025
fold has now been consumed, no newly frozen official
release remains unseen, and registration accuracy is not windshield-replacement
accuracy. Production promotion still requires a new snapshot, deterministic artifact
integration, interval calibration, and explicit country-regression guardrails.
