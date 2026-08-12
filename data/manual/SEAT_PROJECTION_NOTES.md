# Seat projections, source, verification, and join notes

## Source

**Electoral Calculus MRP Poll, January 2026**, conducted on behalf of PLMR.

- Page: <https://www.electoralcalculus.co.uk/blogs/ec_vipoll_20260113.html>
- Data table: <https://www.electoralcalculus.co.uk/blogs/DataTables_VIDec2025.xlsx>
  (linked from the page as `DataTables_VIDec2025.xlsx`), sheet `Seats`.
- Both cached locally under `data/raw/seat_projections/` (`ec_vipoll_20260113.html`,
  `electoral_calculus_DataTables_VIDec2025.xlsx`) so the join is reproducible
  without a further network fetch.
- Method: MRP (Multi-level Regression and Post-stratification), constituency-level
  regression predictions for every Westminster seat, not a uniform-swing estimate.
  The page states explicitly: "gives the estimated result in each Westminster
  constituency."

## Publication date, verified, not assumed

- **Fieldwork: 1–8 December 2025.**
- **Published: 13 January 2026.** Verified two independent ways in the raw HTML:
  - `<META name="date" content="2026-01-13T09:00:00Z">`
  - `<H4 class="dateline">This page first posted 13 January 2026</H4>`
- **2026-01-13 is before the 2026-04-01 cutoff.** Not a close call, nearly
  eleven weeks clear, and the fieldwork itself (Dec 2025) is further back still.
- `commonslibrary.parliament.uk` was not tried (known Cloudflare-blocked per the
  brief). `electoralcalculus.co.uk` responded normally to a scripted `curl`
  fetch with a standard browser User-Agent, no blocking encountered on either
  the HTML page or the `.xlsx` data file.
- Sanity check on the data itself: the page's own headline figures (Reform UK
  31% VI / 335 seats, CON 92, LIB 60) match the seat-count I get by tallying
  the `Seats` sheet's "Predicted Winner (with TV)" column independently
  (335 Reform, 92 CON, 60 LIB), confirms I pulled the right table, not a
  stale or mismatched one.

## What was NOT used

Later Electoral Calculus MRPs exist (April 2026, published 2026-04-23; July
2026), both postdate the cutoff and were excluded. More in Common's 2026
MRPs found in search were all for devolved/local elections (Senedd, Holyrood,
London/Birmingham locals) or postdated the cutoff (e.g. "Final Projections
Ahead of the 2026 Elections", published 6 May 2026), none gave Westminster
constituency-level predictions pre-cutoff, so Electoral Calculus's January
release is the one used.

## Coverage of the 405

The `Seats` sheet covers 632 GB constituencies (no Northern Ireland seats, consistent with the page's "Total GB" summary row). All 405 roster
constituencies (`data/processed/attributes.parquet`, all `party_name` in
`{Labour, Labour (Co-op)}`) matched to exactly one row each.

**Match rate: 405/405 (100.0%).** Zero unmatched.

### Why a naive normalized-name join alone would have failed

Running only `plp_sim.attributes._normalize_constituency_name` (diacritics,
`&`→`and`, case/whitespace) on both sides gave **378/405 (93.3%)**, leaving 27
unmatched. All 27 were a single systematic pattern: Electoral Calculus lists
many seats root-word-first with the compass direction trailing (its own
internal sorting convention), while the official 2024-boundary name leads
with the direction, e.g. official **"North Durham"** vs EC **"Durham
North"**; official **"North Ayrshire and Arran"** vs EC **"Ayrshire North and
Arran"**; official **"Mid and South Pembrokeshire"** vs EC **"Pembrokeshire
Mid and South"**. Two further one-off patterns: EC drops the "Kingston upon"
prefix on both Hull seats, and spells "City of Durham" as "Durham, City of".

I wrote a small deterministic reordering transform (detect a leading
compass-direction phrase, North/South/East/West/Mid/Central, optionally
compound or joined by "and", and try the EC-style reordering as a fallback
candidate before giving up on a name), plus the two one-off substitutions.
Applied on top of the existing fold, this took every one of the 27 to a
unique match. No fuzzy/approximate string matching was used, every
transform is an exact, named, deterministic rule, checked by hand against
each of the 27 names before being trusted (see `official_to_ec_style` /
`official_candidates` in the build script).

**Cross-check on the join itself:** independently of my roster, Electoral
Calculus's own `Seats` sheet carries a `Winner 2024` column. For all 405
matched rows, `Winner 2024 == 'LAB'`, i.e. the seats I matched by name are
also the seats EC itself says Labour won in 2024. This is a second, source-side
confirmation that the join landed on the right rows, not just a name collision.
Also confirmed no EC seat was claimed by two different roster constituencies
(405 distinct EC keys used for 405 roster rows).

## Output

`data/manual/seat_projections.csv`, 405 rows, header:

    constituency,projected_winner,projected_labour_share,projected_margin,at_risk,source_url,published_date

- `projected_winner`: EC's "Predicted Winner (with TV)" [tactical voting]
  column, mapped from EC's party codes to this project's naming convention
  (`Conservative`, `Labour`, `Liberal Democrat`, `Reform UK`, `Green Party`,
  `Scottish National Party`, `Plaid Cymru`, `Independent`, `Other`), plus
  `Your Party` (Corbyn/Sultana, formed 2025, not in the existing
  `schemas.RUNNER_UP_PARTIES` set, so it passes through as its own label
  rather than being forced into "Other").
- `projected_labour_share`: EC's MRP-predicted Labour vote share, as a
  percentage (0–100, matching the scale of `majority_pct`/`vote_share` in
  `attributes.parquet`; the source stores it as a 0–1 fraction).
- `projected_margin` = `projected_labour_share − (highest projected share
  among all other parties)`. One formula, always well-defined: negative when
  Labour is projected to lose (the size of the gap to the party that beats
  it), positive when Labour is projected to hold (the comfort margin over the
  best challenger). Not the source's raw "majority" field, computed here
  directly from the per-party shares in the `Seats` sheet.
- `at_risk` = `projected_winner != Labour` (boolean). Uses the "with
  tactical voting" prediction, EC's headline seat call, not the "no TV"
  alternative also present in the sheet.
- `source_url` / `published_date`: as above for all 405 rows (single source,
  single publication date).

## Headline numbers

- **366 of 405** Labour-held seats (**90.4%**) are projected `at_risk` (lost)
  under this MRP.
- Of those 366: Reform UK projected winner in 267, Green Party in 48, SNP in
  34, Conservative in 15, Your Party in 2.
- The 39 seats projected held are mostly comfortable (e.g. Liverpool Garston
  +14.5pts, Bootle +10.7pts) but the list also includes seats projected to
  survive by well under a point (Ealing Central and Acton +0.10,
  Mitcham and Morden +0.07, Enfield North +0.07), right at the model's noise
  floor, worth flagging as "held" only in the most technical sense.
- Closest *lost* seats (smallest margin against Labour) are effectively
  tied: Lothian East (Reform UK, −0.20), Sheffield Heeley (Green, −0.23),
  Hexham (Conservative, −0.27), Tooting (Green, −0.29).

This is a single-source, single-poll projection (n≈5,500), not an ensemble or
average of multiple MRPs, treat any individual seat's classification as
indicative of poll conditions in Dec 2025, not a certainty, and note this
explicitly if it's surfaced to a persona as "your seat is projected to
fall", that framing is appropriate for the 366, but the 39 held seats,
especially the ~5 held by under a point, are one bad poll away from flipping.
