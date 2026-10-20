# Eurocode mapping review — reviewer brief

ICOR identifies vehicles by **Eurocode 4**: a four-character code for one
windscreen family (one generation of a model, sometimes shared by several models
or bodies). The app knows vehicles only by make and model. This review links every
vehicle identity that carries forecast demand to the codes in ICOR's catalogue, so
demand can be shown per Eurocode.

ICOR's catalogue is confidential and this repository is public. Every example in
this brief is fictional (`Q…` codes, which the real catalogue does not use). Real
codes, catalogue text and review outputs exist only under the git-ignored
`.local/private/eurocode/`.

## What you receive

A batch file `.local/private/eurocode/review/batches/<batch>.md` with:

- **Catalogue rows** for the batch's makes: `r<row> | code | brand | description |
  sop/eop | scope | parser flags`. The description is free text: model, generation
  marker (`II`, `MK3`, chassis codes), body (`HBK` hatchback, `SAL` saloon, `BRK`/`EST`
  estate, `CPE` coupé, `CAB` convertible), drive side (`LHD`/`RHD`), years, Dutch notes
  (`VAN` before a year means "from", `TOT` means "until", `MET` "with", `ZONDER`
  "without", `REEKS` series). `sop`/`eop` are the catalogue's start/end columns;
  `eop` is usually empty and never trusted alone. Truck and bus rows are already
  removed; `conversion` (hearse, camper) and `non_european` rows must never be used.
- **Identities to decide**: each with its forecast demand, demand by registration
  year, right-hand-drive share, top raw spellings and, where raw labels name a body,
  the units per body. An identity is the app's folded make|model; `raw|make|model`
  keys are vehicles the app could not fold.

## What you decide

Write one JSON file, `.local/private/eurocode/review/out/<batch>.<reviewer>.json`:

```json
{
  "batch": "<batch>",
  "reviewer": "a",
  "families": {
    "<batch>/arrow": {
      "label": "Zephyr Arrow",
      "make": "zephyr",
      "codes": [
        {"code": "Q101", "label": "Arrow I", "first_year": 2005, "bodies": ["all"],
         "rows": [101, 102], "note": "rows agree"},
        {"code": "Q102", "label": "Arrow II", "first_year": 2012, "first_year_latest": 2013,
         "bodies": ["hatchback"], "replaces": ["Q101"], "rows": [103]},
        {"code": "Q103", "label": "Arrow II Estate", "first_year": 2013, "last_year": 2016,
         "bodies": ["estate"], "replaces": ["Q101"], "rows": [104]},
        {"code": "Q104", "label": "Arrow II Cabriolet", "first_year": 2014, "last_year": 2018,
         "bodies": ["convertible"], "role": "minor", "rows": [105]}
      ]
    }
  },
  "identities": {
    "zephyr|arrow": {"family": "<batch>/arrow", "why": "all spellings are the Arrow"},
    "zephyr|arrow estate": {"family": "<batch>/arrow", "body": "estate", "why": "names the estate"},
    "zephyr|arrow 2.0 tdi": {"family": "<batch>/arrow", "why": "engine trim of the Arrow"},
    "zephyr|grand": {"status": "ambiguous_identity", "why": "spellings split between two models"},
    "zephyr|model missing": {"status": "not_a_vehicle", "why": "no model given"},
    "zephyr|nova ev": {"status": "not_in_catalogue", "why": "no catalogue row for this model"}
  }
}
```

### Code facts (you state facts; the software computes the years)

| Field | Meaning |
|---|---|
| `code` | A code that appears in the batch's catalogue rows. Never invent or alter one. |
| `rows` | The `r<row>` numbers (as integers) that support this code — only rows of this code. |
| `first_year` | The earliest year the vehicle can be registered: the earliest start the cited rows give (the `sop` column or a start year in the text). |
| `first_year_latest` | Only when the cited rows disagree on the start: the latest such start, the year from which the code is certainly in volume. Omit otherwise. |
| `last_year` | The last production year, only when a cited row states it or it is beyond doubt (the model was discontinued with no successor in this family). Otherwise omit it and let `replaces` end the code. |
| `bodies` | `["all"]` when the code serves every body of the family, or the bodies it serves: `hatchback`, `saloon`, `estate`, `coupe`, `convertible`, `suv`, `mpv`, `van`, `pickup`. |
| `role` | `main` (default) for mainstream bodies; `minor` for a niche body with its own code (convertible, van, pickup, a coupé of a hatch/saloon family). A minor code receives demand only from registrations whose label names that body. |
| `drive` | `both` (default), or `LHD`/`RHD` when the code exists for one drive side only. When a generation has an RHD-specific code, its unmarked counterpart is `LHD`. |
| `replaces` | Codes of the same family that this code succeeds, for the bodies it covers. |
| `label` | A short human label for the code, such as model and generation. |
| `note` | Optional: any doubt, contradiction or external fact you relied on. |

How the software turns facts into registration years (per drive side):

- A code is active from `first_year`.
- A replaced code stays active until its successor's `first_year_latest`. The
  successor's launch span is shared, because registration year is not production
  year.
- A stated `last_year` keeps the code active one more year, for stock registered
  after production ends. When `last_year` equals the successor's certain start,
  the shared launch year already covers that stock.
- The last code of a family stays active for future years; never invent a future code.
- A code replaced for some of its bodies must also end for its other bodies,
  either by a successor for them or by a stated `last_year`. Validation refuses it
  otherwise.
- **Without a body in the label:** every active `main` code applies, so two
  mainstream bodies with different codes are shared, never split.
- **With a body:** codes that state the body win; otherwise `all`-body codes apply;
  otherwise the main codes apply.

A year that two codes can explain is shown as a range: certain units for a code,
plus "up to X shared". That is honest and acceptable. A single code where two were
possible is a false certainty and is the worst error. **When unsure, choose the
fact that leaves years shared:** an earlier `first_year`, a later end, `both`
drive sides.

### Families

- A family is one vehicle line as the market knows it, across its generations:
  every code a registration of that model can carry. Key it `<batch>/<slug>`.
- Sibling models are separate families even when the names overlap: a model,
  its `PLUS`, `SPORTSVAN`, `GRAND`, `CROSS`, `ALLROAD` or `COUNTRY` derivative,
  and an `ACTIVE TOURER` versus a `GRAN TOURER`, all with their own windscreens.
  Trim words (`GTI`, `RS`, `AMG`, `M`, engine codes) are not siblings.
- **Prefix traps:** a model named `208` is not `2008`; a model named `C3` is not
  `C3 PICASSO` or `C3 AIRCROSS`. Match whole names.
- A code shared by two models (for example one windscreen for a hatchback and a
  related SUV) belongs to both families.
- A renamed model that continued the same line (an identity spelled as both
  names) may be one family with the codes of both names in succession.
- Rebadged twins: the catalogue may list one code under several brands. Use the
  code under the identity's own make when it is listed there.

### Identities

- `mapped` (default, with `family`): the identity names one family for
  essentially all of its demand. Check the raw spellings, not just the key: an
  identity whose spellings mix two models is `ambiguous_identity`.
- Set `body` when the identity itself names a body (`avant`, `variant`, `touring`,
  `sw`, `cabrio`, `coupe`). Engine and trim names (`320d`, `a180`, `2.0 tdi`)
  map to the model's family without a body.
- `ambiguous_identity`: it could be several families (`amg`, the make alone,
  `grand`).
- `not_in_catalogue`: a real vehicle the catalogue has no code for. Search
  carefully (other spellings, chassis codes, rebadges) before choosing this.
- `not_a_vehicle`: junk (`model missing`, a dealer word, a number fragment).
- Every identity in the batch must be decided, and every family must be used.

## Checklist before you finish

1. Every code comes from the batch rows; every cited row belongs to its code.
2. For each mapped identity, its demand years fall inside the family's codes.
   `validate` lists any demand outside every code: explain it in `why` or fix it.
3. Generations follow one another without gaps or overlaps you cannot justify;
   facelifts that kept the code are the same fact.
4. Bodies with different codes are separated; niche bodies are `minor`.
5. RHD-only codes are `RHD`, and their unmarked counterparts are `LHD`.
6. Siblings and prefix traps are separate families.
7. Run `uv run python scripts/eurocode_review.py validate <your file>` until it
   exits 0 with no problems, and read `uv run python scripts/eurocode_review.py
   preview <your file>` to check every family's timeline is what you intend.

## Rules

- Work blind: do not open other reviewers' files in `review/out/` or anything in
  `review/compare/`.
- Write only your own output file. Do not edit any other file, do not run git,
  and do not copy codes or catalogue text anywhere else.
- Public knowledge may support a fact (a launch year, whether two bodies share a
  windscreen), but every code must come from the catalogue. Record any conflict
  with the catalogue in `note`.
