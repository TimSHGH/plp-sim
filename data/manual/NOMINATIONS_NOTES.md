# Nominations data: sourcing and honesty notes

## The event

A UK Labour Party leadership contest did take place in mid-2026, triggered by Keir Starmer's
resignation on 22 June 2026. PLP (MP) nominations ran **9 July 2026 to 18:00, 15 July 2026**.
Source for the timetable: LabourList, "Timetable for leadership contest confirmed by NEC"
(https://labourlist.org/2026/06/timetable-for-leadership-contest-confirmed-by-nec/), corroborated
by Wikipedia's "2026 Labour Party leadership election" article
(https://en.wikipedia.org/wiki/2026_Labour_Party_leadership_election).

Andy Burnham (newly returned as MP for Makerfield after an 18 June 2026 by-election) was
nominated by 379 of ~405 sitting Labour MPs and was the only candidate to clear the 81-nomination
threshold. One MP, Neil Coyle, nominated Catherine West instead, after Burnham had already
mathematically won.

`nominations-open` used for `days_from_open` = **2026-07-09**.

## The source: the Labour Party's own live nomination-tracker page, via Wayback Machine

The party published a live, continuously-updated results page during the nomination window:
`https://labour.org.uk/labour-leadership-2026/labour-leadership-2026-plp-nominations/`. It lists
every Labour MP who has nominated, their constituency, and who they nominated, this is the
per-MP tracker the task asked for. It does **not** timestamp individual nominations; it only shows
the cumulative state at the moment it's viewed.

The Wayback Machine (web.archive.org) crawled this exact page four times during the nomination
window. I fetched all four snapshots directly and diffed them by MP name to find, for each MP, the
**earliest snapshot in which their name first appears** against a candidate:

| Snapshot fetched (UTC)   | Cumulative rows | New MPs vs. prior snapshot | Source URL |
|---|---|---|---|
| 2026-07-09 18:54:56 | 322 | 322 (all, since none earlier exists) | http://web.archive.org/web/20260709185456/https://labour.org.uk/labour-leadership-2026/labour-leadership-2026-plp-nominations/ |
| 2026-07-10 08:46:23 | 322 | 0 | (not used as a row source, identical content to 07-09) |
| 2026-07-13 21:52:55 | 349 | 27 | http://web.archive.org/web/20260713215255/https://labour.org.uk/labour-leadership-2026/labour-leadership-2026-plp-nominations/ |
| 2026-07-16 14:39:53 | 380 | 31 | http://web.archive.org/web/20260716143953/https://labour.org.uk/labour-leadership-2026/labour-leadership-2026-plp-nominations/ |

No candidate ever switched between snapshots for any MP, and the cumulative counts (322 → 349 →
380 total rows, 379 Burnham + 1 West at the end) match the counts reported independently by ITV,
RTE and LabourList for 13 and 15 July. I checked the CDX index for additional/denser crawls of
this URL between 2026-07-08 and 2026-07-18 and there are none, these four are the entirety of
what Wayback Machine captured.

## What `declared_date` actually means here, READ THIS BEFORE MODELING

**This is not a self-reported declaration date. It is "nominated on or before this date," derived
from when a name first appears in an archived snapshot of a live tally page.** I did not find any
source that states an exact date for an individual MP's nomination (that would have supported
`confidence=high`). All 380 rows are `confidence=medium` on that basis, per the task's own
definition ("date only datable to the [snapshot's] publication date").

Consequently **`declared_date` takes only three distinct values in the whole file**:
`2026-07-09`, `2026-07-13`, `2026-07-16`. This is a genuine limitation of Wayback Machine's crawl
frequency for this URL, not a shortcut I took, I looked for denser trackers (BBC, Guardian,
PoliticsHome, LabourList day-by-day pieces) that might name individuals on the days in between and
found none with named, dated per-MP detail. The true declaration dates for the 322 "day-1" MPs
could genuinely span any point from 9 July (nominations opening) up to the 18:54 snapshot time,
and are indistinguishable from each other in this data. The gap between the 07-10 and 07-13
snapshots (over 3 days) is the widest: the 27 MPs in that bucket could have nominated any day from
07-10 through 07-13. The final bucket (31 MPs, dated 2026-07-16) is a special case: nominations
were confirmed closed at 18:00 on 2026-07-15 (see timetable source above), a full day before this
snapshot was crawled, so the true window for that bucket is 07-13 to 07-15, and I have recorded
the crawl date (07-16) rather than the close date because that is the literal date I can point to
in the cited source, treat 07-15 as the true upper bound for that bucket if you need one.

**Practical implication:** this dataset supports a coarse three-level "early / mid / late
nominator" categorical, or a binary "day-1 bandwagon vs. rest" split. It does **not** support a
day-by-day or continuous hazard/event-time model of nomination timing, there is no source data at
that resolution for this contest. If the modeling plan (e.g. the cascade / declaration-ordering
work mentioned in the project README) needs finer-grained ordering, this file cannot provide it
honestly; the alternative would have been to leave `declared_date` blank for everyone, which
seemed like a strictly worse outcome given the three real buckets that are supportable.

## Coverage and bias

- 380 rows / ~405 sitting Labour MPs at the time ≈ **93.8% coverage** of the PLP.
- Confidence breakdown: **380 medium, 0 high, 0 low.**
- Candidate breakdown: 379 Burnham, 1 West (Neil Coyle).
- The ~23-25 MPs missing are, per LabourList's reporting
  (https://labourlist.org/2026/07/who-are-labour-mps-backing-to-become-the-partys-next-leader-and-prime-minister/),
  MPs who **declined to nominate anyone**, this is not random attrition, it is a distinct group
  (non-nominators) that is systematically excluded from this file because the tracker page only
  lists MPs who nominated. If "declared vs. did not declare" itself matters to the model, this file
  alone will make coverage look like ~94% when the true denominator includes a real "abstained"
  category this file is silent on.
- Within the 380 who did nominate, the tracker is a complete population list, not a sample, there
  is no over-representation of frontbenchers or high-profile MPs the way a hand-curated media list
  would have. The known bias is entirely in the **date** axis (three buckets, see above), not in
  *which* MPs are covered.
- I did not attempt to source `constituency` from a second source, it comes from the same Labour
  Party page and matches the MP's current seat name as Labour lists it (which may differ slightly
  from the Commons Library's naming after the 2024 boundary changes).
