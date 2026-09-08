# Authoritative all-model vehicle catalog intake

## Why an external catalog is required

The active ICOR snapshot contains official registration evidence but not a governed
all-model generation catalog. Its 22,893 forecastable raw make/model identities include
publisher spellings, engines, trims, body variants, and aliases. The legacy ICOR list
contains 128 distinct labels and generic values such as G1; it is not generation truth.

Client-visible generation names must therefore come from licensed reference data or
manufacturer/type-approval evidence. They must never be invented from registration year,
an LLM, or the legacy generic labels.

## Preferred source package

Request TecAlliance access for:

1. TecDoc Reference Data or TecDoc Data Package covering European passenger cars.
2. Vehicle/model/type identifiers and construction windows.
3. TecDoc Web Service access if scheduled incremental refresh is required.
4. TecDoc VIO if ICOR wants licensed parc evidence linked directly to the TecDoc vehicle
   tree.
5. Vehicle-to-glazing/OE/article linkages if the licence permits windshield fitment use.

Before purchase or ingestion, obtain written confirmation of:

- covered countries and historical construction depth;
- update frequency, revision behavior, and stable identifier policy;
- internal application, client display, derived analytics, caching, and redistribution
  rights;
- whether generation, model design/chassis code, body construction, facelift/build
  window, drive side, and glazing/ADAS fitment fields are populated for ICOR's target
  vehicles;
- API rate limits or bulk-delivery format and permitted retention after subscription
  termination.

## Minimum sample fields

Provide a representative, non-secret sample export containing at least:

- manufacturer ID and name;
- model-series ID and name;
- vehicle type or kType ID;
- generation or range name where populated;
- model design, chassis/platform, sales designation, and type designation;
- construction/body type;
- construction month from and to;
- market/country applicability;
- left/right-hand-drive applicability where available;
- source release/version and extraction timestamp;
- windshield article, OE, or linkage identifiers and fitment criteria when licensed.

The sample must include transition years, facelifts, body variants, and at least the
legacy ICOR worked-model set. Do not include API keys, passwords, customer records,
orders, VINs, or personal data.

## Odoo alternative or supplement

If ICOR already stores the necessary vehicle and glazing reference data in Odoo, provide
a schema-only description and a controlled read-only export with:

- Odoo model and field technical names;
- stable record IDs and write_date;
- make, model series, generation/design/chassis, build window, body, drive side;
- windshield SKU, OE reference, equipment/ADAS criteria, and fitment relationship;
- the business meaning of worked on: quoted, sold, invoiced, manufactured, fitted, or
  technically developed.

Use a least-privilege read-only integration user. Never put the Odoo URL credential,
database secret, or API key in Git or chat.

## Import acceptance gates

The full client catalog is releasable only when:

1. Every visible make/model/year resolves to exactly one sourced generation.
2. Ambiguous transition years are excluded or split by a stronger technical identifier.
3. Publisher aliases consolidate into one canonical model series without losing raw
   lineage.
4. Generic or estimated generation labels are absent from the client API and UI.
5. Construction windows are month-precise where transitions overlap.
6. Every generation retains source, licence, release, method, and confidence metadata.
7. Fitment is not claimed unless the licensed vehicle-to-windshield linkage supports it.
8. Automated completeness, uniqueness, interval, alias, and regression checks pass.
