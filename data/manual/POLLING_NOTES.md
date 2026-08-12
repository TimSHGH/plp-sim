# Polling context: sourcing and honesty notes

## Coverage

- **Date window actually covered:** 2025-01-03 to 2026-03-30 (fieldwork end dates).
  The task's target window was ~2025-01-01 to 2026-03-31; the actual data starts
  two days later than the nominal start (the first VI poll with a fully
  attributable fieldwork-end date and citation is 3 January 2025) and the last
  admitted row is 2026-03-30, one day short of the hard cutoff, because the
  next candidate rows all failed the publication-date check below.
- **Row count:** 419
- **Pollsters (25):** BMG Research, Convergent Opinion, Deltapoll, Find Out Now,
  Find Out Now/Electoral Calculus (MRP), Focaldata, Freshwater Strategy, Good
  Growth Foundation, Ipsos, JL Partners, Lord Ashcroft Polls, More in Common,
  More in Common (MRP), Opinium, Redfield & Wilton Strategies, Savanta, Stack
  Data Strategy (MRP), Survation, Survation (MRP), Techne, Trajectory
  Partnership, Verian, Whitestone Insight, YouGov, YouGov (MRP).
- 357 rows carry voting-intention figures; 231 rows carry `leader_net`
  (Keir Starmer's net satisfaction/approval). 169 rows have both, because the
  same release (mainly More in Common, which publishes VI and trackers in one
  document) reports both in a single row; the remainder are VI-only or
  approval-only releases.

## Sources tried

**Used:**
- Wikipedia, "Opinion polling for the next United Kingdom general election"
  (`https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_United_Kingdom_general_election`), the "National poll results" tables for 2025 and 2026, for Labour/Conservative/
  Reform UK/Lib Dem/Green voting intention.
- Wikipedia, "Leadership approval opinion polling for the next United Kingdom
  general election"
  (`https://en.wikipedia.org/wiki/Leadership_approval_opinion_polling_for_the_next_United_Kingdom_general_election`), the "Leadership approval" tables for 2025 and 2026, for Keir Starmer's net
  satisfaction (he was Labour leader throughout the whole window; the
  Burnham-headed tables that appear later in this same article, covering
  fieldwork from ~July 2026 onward, are out of window and were not used).
- Both were fetched via Wikipedia's REST HTML endpoint
  (`/api/rest_v1/page/html/<title>`) rather than scraped as rendered pages, so
  the party-affiliation-colour cells and citation footnotes parse reliably.
  Wikitext (`action=raw`) was pulled in parallel and used specifically to
  read citation `date=`/`access-date=`/`publication-date=` template
  parameters for the cutoff check below (the rendered HTML carries fieldwork
  dates in the table but the *publication* date only inside the footnote's
  markup).
- Every row's `source_url` is the pollster's own primary document (PDF/xlsx/
  web page) taken from Wikipedia's own citation for that row, not the
  Wikipedia article itself, confirmed zero rows fell back to the article URL
  for lack of a per-row citation.

**Tried and failed / not used:**
- Politico's "Poll of Polls" UK page returns HTTP 200 but serves a Cloudflare
  interstitial ("Just a moment...") to a scripted fetch, not usable, matching
  the brief's own warning.
- Ipsos's UK political-monitor landing page (`ipsos.com/en-uk/political-monitor`
  and a guessed `-satisfaction-ratings` variant) both 404. Ipsos's actual
  tracker releases *are* represented in the CSV, Wikipedia's per-row
  citations link straight to the individual Ipsos article for each wave (21
  rows), so the tracker's data made it in even though the landing/index page
  didn't resolve.
- YouGov's own tracker index (`yougov.co.uk/topics/politics/trackers`) 301-
  redirects and wasn't pursued further, for the same reason: individual
  YouGov releases are already captured via Wikipedia's per-row citations,
  which point at YouGov's own PDF/article for each wave (89 + 2 MRP rows).

## The cutoff check went past fieldwork date, this project's stated failure mode, caught this time

Fieldwork end date alone is not sufficient to establish admissibility. A poll's
*fieldwork* can end before 2026-04-01 while its *publication*, the point at
which the number actually became public and could have been part of the
pre-cutoff information environment, lands after it. I checked every row with
fieldwork ending on or after 2026-03-20 (13 rows) against its citation's own
`date=`/`access-date=`/`publication-date=` field in the underlying wikitext,
not just the fieldwork date shown in the rendered table. Four failed and were
excluded even though their fieldwork end date alone would have passed:

| Pollster | Fieldwork end | Citation publication/access date | Why excluded |
|---|---|---|---|
| Ipsos | 2026-03-24 | 2 April 2026 | Article itself dated 2 Apr 2026 |
| More in Common | 2026-03-30 | 1 April 2026 | Dated exactly 2026-04-01, not *strictly before* the cutoff |
| Lord Ashcroft Polls | 2026-03-30 | 4 April 2026 | URL path is `/2026/04/`; access-dated 4 Apr |
| More in Common (MRP) | 2026-03-30 | 12 April 2026 | Monthly MRP write-up, published well after fieldwork closed |

This is exactly the shape of the two prior leaks the brief describes: a
plausible-looking, correctly-dated-sounding row that turns out to postdate the
cutoff once you check the actual publication record rather than the fieldwork
window. All four are omitted from `polling_context.csv` and are not part of
the summary paragraph below. I did not attempt to verify publication dates for
every one of the 419 admitted rows individually, that was only feasible/
necessary in the two-week window before the cutoff, since observed
publication lag elsewhere in the dataset is 0–5 days after fieldwork end for
ordinary trackers (long enough to matter only right at the boundary) and up to
~12 days for monthly MRP write-ups (which is why both MRP-tagged rows in that
window failed the check). Rows further from the cutoff carry no realistic risk
of the same failure mode.

## What the political weather looked like immediately before the cutoff

Over the fifteen months to 30 March 2026, Labour's voting intention fell from
the high-to-mid 20s (30% in Deltapoll's 3 January 2025 poll; 24–26% across
several pollsters that same month) to the mid-to-high teens by the final weeks
before the cutoff (16% in Find Out Now's 27 March 2026 poll; 18% in YouGov's
30 March 2026 poll; 15–21% across the last dozen polls). Reform UK moved the
opposite direction, from the low-to-mid 20s in January 2025 (22–25%) to the
largest-party position in most late-March 2026 polls (23–28%, e.g. 28% in BMG
Research's 26 March 2026 poll). The Green Party also rose sharply over the
window, reaching 18–20% in the final polls before the cutoff (20% in Verian's
23 March 2026 poll), often running level with or ahead of Labour. Keir
Starmer's net satisfaction, already deeply negative at the start of the
window (-42 in Deltapoll's 3 January 2025 poll; -41 in YouGov's 13 January
2025 poll), worsened further by the cutoff, with trackers recording -46 to -50
in the final two weeks (-46, BMG Research, 26 March 2026; -48, YouGov, 19
March 2026; -50, YouGov, 23 March 2026). Immediately before the cutoff, then,
Labour was polling in third or fourth place behind Reform UK and, in several
polls, a resurgent Green Party, with the Prime Minister's personal net rating
among the worst recorded for a sitting PM this far into a parliamentary term.
