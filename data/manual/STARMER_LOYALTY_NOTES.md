# Starmer loyalty data: sourcing and honesty notes

## Gate 1, contamination check

**PRESSURE BEGAN (at Westminster-PLP level): 2026-05-07 to 2026-05-09**, hardening into an organised,
named campaign from **2026-05-11** onward. First individually-dated Westminster MP break: Catherine
West, 9 May 2026 ("Labour's schedule to shift Starmer from office" reporting; corroborated by
`2026_Labour_Party_leadership_crisis`, https://en.wikipedia.org/wiki/2026_Labour_Party_leadership_crisis).
There IS an earlier public resignation call, Anas Sarwar, leader of Scottish Labour, on **9 February
2026** (https://www.aljazeera.com/news/2026/2/9/uk-pm-starmers-communications-chief-quits-amid-epstein-scandal-fallout,
corroborated by https://foreignpolicy.com/2026/02/10/keir-starmer-resign-labour-party-uk-peter-mandelson-jeffrey-epstein-files/), but Sarwar is a member of the Scottish Parliament (MSP), not one of the ~405 Westminster Labour MPs
this file's roster covers, so his call is not a construction-input risk for this roster. Two senior
Starmer aides (Chief of Staff Morgan McSweeney, Communications Director Tim Allan) also resigned in
February 2026 over the same Mandelson/Epstein affair, but they are staff, not MPs, so likewise fall
outside the roster.

I specifically searched for any Westminster Labour MP taking a public position (either direction)
between January and 31 March 2026 and found none. The Wikipedia crisis-timeline article's own
earliest Westminster-MP entry is Catherine West on 9 May. No dated tracker, letter, or news report
names an individual Westminster MP before that date.

**PRE-CUTOFF CONTAMINATION: NO**, all Westminster Labour MP positions in this file postdate
2026-04-01 by five weeks or more. (The 9 February Sarwar call is a genuine precursor to the crisis
and is documented above for context, but it is not an MP-roster row and creates no leakage.)

**PER-MP TRACKER: FOUND.** Two independent, per-MP, quote-and-constituency trackers exist for the
"calling for exit" side:
- LabourList, "Which Labour MPs are calling for Starmer to go, and who is still backing PM?" (live-updated, timestamped 15 May 2026, 10:00am): https://labourlist.org/2026/05/labourlist-labour-mp-starmer-resignation-tracker/
- New Statesman, "Tracked: the Labour MPs calling for Keir Starmer to go": https://www.newstatesman.com/politics/uk-politics/2026/05/tracked-the-labour-mps-calling-for-keir-starmer-to-go

And one for the "backing" side:
- Left Foot Forward, "Over 100 Labour MPs sign letter backing Keir Starmer and opposing leadership contest" (full signatory list), corroborated by Wikipedia and AOL as an event on 13 May 2026: https://leftfootforward.org/2026/05/over-100-labour-mps-sign-letter-backing-keir-starmer-and-opposing-leadership-contest/

A dedicated, timestamped resignations live-blog also exists and was used to date-anchor the subset
of MPs whose position coincided with a ministerial/PPS resignation:
https://labourlist.org/2026/05/live-updates-government-resignations/

## What went into the file, and why

**called_for_exit (101 rows).** Source: the LabourList resignation tracker (chosen over New
Statesman as the primary citation, see "Quality note" below). It lists MPs alphabetically with
constituency and a short quote fragment for each; I used that quote fragment as the `quote` field
for every row so the citation is self-consistent (the URL you check will contain the fragment shown).
This includes 2 suspended (non-whip) MPs, Diane Abbott and Karl Turner, whom the tracker lists
separately as also calling for resignation; they may not be part of your ~405-MP "sitting Labour MP"
roster, in which case they will simply fail to join, which is fine.

Of these 101, **14 got a specific `position_date` and were upgraded to `confidence=high`** because I
could independently pin an exact date via a second source (either the live-blog above, or direct news
search): Catherine West (9 May), Sally Jameson, Tom Rutland, Joe Morris, Naushabah Khan, Melanie Ward,
Gordon McKee (all 11 May, tied to PPS resignations reported same-day by GB News/Politics.co.uk), Miatta
Fahnbulleh, Jess Phillips, Alex Davies-Jones, Zubir Ahmed (all 12 May, junior-minister resignations
reported by Al Jazeera/UPI/US News), Wes Streeting (14 May, Health Secretary resignation, reported by
Euronews/CNN/Al Jazeera), Rosie Wrighting (18 May per the live-blog, note this conflicts with a
Wikipedia summary that groups her with the 14 May batch; I trusted the dedicated, timestamped live-blog
over the more compressed Wikipedia summary), and Steve Race (19 May per the live-blog). **The date for
these 14 is corroborated on a different page than the one cited in `source_url`** (which stays the
main tracker, for quote consistency), if you spot-check dates specifically, check the live-blog URL
above rather than the row's `source_url`. The remaining 87 rows have no `position_date`: the tracker
is a cumulative alphabetical list, not a day-by-day one, and I found no independent dating for most
individual entries within the time-box.

**backed_starmer (110 rows).** Source: the Left Foot Forward article listing the full text of a
110/111-signatory letter to colleagues stating "This is no time for a leadership contest," reported
13 May 2026 and corroborated independently by an AOL/Yahoo report also dated 13 May and by the
Wikipedia crisis timeline. **Confidence is medium for all 110** per the task's own definition
(named in a group, not individually quoted), and there is a specific, documented reason it should
stay at medium and not be trusted at face value: **at least two of the original 111 signatories
publicly denied signing.** Ealing Central & Acton MP Rupa Huq said on social media "Surprised to see
my name on this list when I haven't either signed any letter supporting the PM or called for the PM
to go... Not very courteous of colleagues to put names down without their approval", I excluded her
entirely rather than mark her either way, since her own statement contradicts both categories. The
Times reportedly identified a second MP added without consent, but I could not find that MP named in
any source I could reach, so I could not exclude them specifically; treat the 110 as "reported
signatories," not "110 confirmed positions," and expect roughly 1 of them to be a false positive I
could not identify. I did not source constituency for this group (Left Foot Forward's list is names
only); I found constituencies for 14 of the 110 via a secondary Wikipedia lookup and left the rest
blank rather than fill from memory, recommend joining on name for this group.

## Quality note: why LabourList over New Statesman for called_for_exit

Both trackers list almost the same ~97-101 MPs. I used LabourList as the sole `source_url` because
New Statesman's version, when extracted, showed two signs of drift I couldn't reconcile: (1) a quote
attributed to Jo White that reads as internally contradictory ("orderly change of leadership...back
him"), and (2) a quote attributed to Fabian Hamilton referencing "allow Andy Burnham" to take over, premature, since Burnham had no path to the leadership until his 18 June by-election win, five weeks
later. I could not verify these two names or a further three New Statesman-only names (Luke Charters,
Patrick Hurley, Polly Billington) against LabourList's list or independent search, so I **excluded all
five from the file** rather than include a quote I could not confirm. This is a deliberate coverage
sacrifice in favour of not risking a fabricated-looking citation.

## Coverage and bias

- 211 rows / ~405 sitting Labour MPs ≈ **52% coverage** of the PLP (99 sitting MPs in
  called_for_exit + 110 in backed_starmer, plus 2 suspended MPs = 211 total; ~194 MPs are absent
  from both trackers and get no row).
- Position breakdown: **101 called_for_exit, 110 backed_starmer.** (No `silent` rows emitted, per
  instructions, the ~194 absent MPs are not encoded as anything.)
- Confidence breakdown: **14 high, 197 medium, 0 low.**

## Bias section, read before treating "absent" as "silent"

**Yes, these trackers systematically over-represent the vocal, and the two sides are vocal for
different reasons, which matters a lot if you infer anything about the missing ~194.**

- The **called_for_exit** side is disproportionately backbenchers, several already-rebellious
  "usual suspects" (Corbyn-aligned figures like John McDonnell, Richard Burgon, Ian Lavery, Diane
  Abbott), and MPs in marginal seats hit hard by the May local elections, i.e. people with a
  standing incentive to be publicly critical, or a fresh personal reason (electoral fear) to be
  loud right then. A press call for a "timetable" is also cheap relative to actually voting no
  confidence, which inflates this list further.
- The **backed_starmer** side is a coordinated letter, i.e. an organised whip/loyalist operation,
  not decentralised individual statements. That itself is informative (loyalists needed to be
  organised to compete for the same media day the resignation tracker was accumulating), but it
  also means these 110 names reflect who was willing to be organised into signing something, not
  spontaneous public backing at the same rate as the rebels' spontaneity. And per the Rupa Huq case,
  at least one name is confirmed appended without consent, the letter's list is not perfectly self-
  reported the way the resignation quotes are.
- **The absent ~194 MPs are not one thing.** Genuinely loyal-but-quiet MPs, genuinely undecided
  fence-sitters, and MPs who were simply never asked by a journalist or never approached for the
  letter are all indistinguishable in "absence from this file." Given the mechanics above (rebels
  self-select into press coverage; loyalists self-select into an organised letter), I'd expect the
  absent group to skew toward the least-engaged/least-networked MPs in the PLP, not a random sample
  and not necessarily "closet Starmer loyalists." **Do not treat silence as a validated third class, treat it as missing data with a plausible skew toward disengagement, not toward either
  position.**

## Honest view on usability

This is a substantially stronger validation target than the leadership-nomination file: genuine
binary choice under real uncertainty (resignation was not a foregone conclusion in May, unlike the
379/380 Burnham coronation), good per-MP granularity (individual quotes, not a cumulative snapshot),
and 52% roster coverage split roughly 46%/54% between the two positions, real variance on both
sides. The two soft spots are (1) the missing ~194 are not safely treated as "loyal," only as
"unclassified," and (2) only 14/211 rows have a specific date, most of `called_for_exit` and all of
`backed_starmer` are dated to a single day (13 or a specific mid-May date) or left blank, so this
file supports a binary/categorical target well but not a day-level hazard model of when defection
happened, similar in spirit to the nominations file's date-resolution limit.

---

## Verification

Independent check of the compiled rows. Both cited pages were re-fetched and
every row cross-referenced against them.

- **211/211 rows are named in their cited source.** No fabricated attributions.
- Sources are cleanly separated and not mixed: LabourList supplies all 101 `called_for_exit`,
  Left Foot Forward all 110 `backed_starmer`.
- No MP appears twice; no MP appears on both sides. Rupa Huq is correctly absent.
- Every row carries a supporting quote.

**Two silent name normalisations found, worth knowing before the join.** The Left Foot Forward
signatory list misspells two MPs, and the compiled file quietly corrects them:

| In the source | In this file | Real MP |
|---|---|---|
| Jenny Riddle-Carpenter | Jenny Riddell-Carpenter | Riddell-Carpenter |
| Michelle Scroogham | Michelle Scrogham | Scrogham |

The correction is right for joining to the Members API, but it departs from the "as it appears in
the source" rule, so it is recorded here rather than left implicit. Both corrected rows are
`backed_starmer`; if either is wrong it inflates the loyalist count, which makes the target harder
rather than easier, the conservative direction.

**Standing caveat carried forward:** the ~194 MPs absent from both trackers are unclassified, not
loyal. Rebels seek visibility and loyalists mostly stay quiet, so a derived `silent` class would
conflate loyalty with non-participation. Model this as `called_for_exit` vs `backed_starmer` on the
211 classified MPs, and treat the remainder as missing.
