# Forecast validation

## Production decision

The registration forecast is
`validated-recency-damped-ensemble-v2`. It averages:

- the latest observed annual registration value; and
- a five-year linear trend whose future increments decay by 20% per step.

The 50/50 weight is fixed globally. It is not selected separately for each short
vehicle series, because that approach overfit the available histories.

## Reproducible outer validation

Run:

    uv run python scripts/benchmark_registration_forecasts.py --root .local/evidence

Against active snapshot `snapshot-fcb3cdb004a4b7c4042b`, the benchmark:

- retained 16,274 contiguous series with at least eight annual values;
- removed each series' newest two years before model fitting;
- predicted those unseen years;
- aggregated error with demand-weighted absolute percentage error (WAPE);
- compared the production method with the replaced rolling-origin mean/linear
  selector.

Verified result:

| Method | Newest-two-year WAPE |
| --- | ---: |
| Replaced mean/linear selector | 0.626247 |
| Validated recency/damped ensemble | 0.498932 |

The relative error reduction is 20.33%. Lower is better.

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
