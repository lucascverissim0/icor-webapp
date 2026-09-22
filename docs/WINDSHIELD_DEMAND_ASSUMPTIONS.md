# Windshield demand assumptions

## Recommended free-data baseline

The annual windshield-replacement point estimate is **4.3026% of the active
fleet**. It is derived from two public French insurance-sector datasets:

1. France Assureurs reports **60.6 glass-damage claims per 1,000 covered
   exposures for the relevant subscribed guarantee among first-category vehicles
   outside fleets in 2025**.
2. A Crédit Agricole Assurances/Pacifica, BCA Expertise, Europ Assistance, and
   Institut Louis Bachelier study reports that **71% of 2022 glass claims led to
   a windshield replacement**, 12% to windshield repair, and 17% to replacement
   of another glazed element. Pacifica supplied the glass-claim operations and
   the study says their representativeness was checked against France Assureurs.

The reproducible calculation is:

    0.0606 glass claims per covered vehicle-year
    x 0.71 windshield replacements per glass claim
    = 0.043026 windshield replacements per active vehicle-year

Sources:

- France Assureurs, *Le marché de l'assurance automobile des particuliers en
  2025* (July 2026):
  https://www.franceassureurs.fr/wp-content/uploads/le-marche-de-lassurance-automobile-des-particuliers-2025.pdf
- Crédit Agricole Assurances et al., *Les émissions de CO2e de la gestion des
  sinistres automobiles en France* (2022 operating data), pages 11 and 14:
  https://www.bca.fr/wp-content/uploads/2024/08/CAA_livreBlanc_BATflipbook_compressed-1.pdf

## Model parameters

| Scenario | Annual replacement rate | Meaning |
| --- | ---: | --- |
| Downside | 3.44208% | Base rate minus 20% |
| Base | 4.30260% | Latest national claim frequency x observed replacement mix |
| Upside | 5.16312% | Base rate plus 20% |

The downside/upside values are planning scenarios, not empirically calibrated
P10/P90 quantiles. The 20% band explicitly represents transfer uncertainty from
one French insurer's operation mix to the national fleet and from France to other
markets. It must be replaced by market-specific calibration when comparable claim
or installation outcomes become available.

## Boundaries that prevent false precision

- The free evidence supports a fleet-average event rate. It does **not** support
  the previous age-band escalation or the previous 10% Great Britain multiplier,
  so neither is used.
- France Assureurs reports claim frequency by subscribed guarantee for insured
  first-category vehicles outside fleets. Applying it to every surviving vehicle
  assumes uncovered/self-paid behavior does not materially change physical
  replacement demand. The result is a European planning proxy, not a measured
  rate for every country, vehicle age, model, or windshield design.
- Forecast demand is the expected number of replacement events, not a count of
  distinct vehicles and not a claim that every event maps to one ICOR SKU.
- A discontinued generation receives no forecast registration cohorts after its
  documented end year. Its earlier cohorts remain in the active fleet after that
  date, decay through the survival model, and continue to generate replacement
  demand.
- Fleet survival is now calibrated against UK DfT licensed stock and is no
  longer an assumption; see `docs/FREE_DATA_MODEL_IMPROVEMENT.md`. Outside the
  UK it is an explicit transfer, and each cohort records which it received.
- The windshield replacement rate on this page and exact vehicle-to-windshield
  fitment remain uncalibrated. Final demand intervals therefore remain
  assumption-led until ICOR replacement history and fitment truth are
  integrated and backtested. Calibrating the fleet does not calibrate the rate
  applied to it.
