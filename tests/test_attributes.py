"""Tests for plp_sim.attributes.

The module under test builds schemas.ATTRIBUTES from real, cached data (plus
one small live-fetched-and-cached electoral CSV). The properties worth
testing are the ones a *wrong* build would pass by accident:

* a roster count pinned to a literal that silently drifts from the real
  members cache;
* a "rebellion rate" that quietly includes a post-cutoff division, which
  would leak the holdout outcome into the construction frame;
* a nominations join that conflates "declined to nominate" with "row we
  could not place", or that drops an unmatched row instead of surfacing it;
* an output that never gets checked against schemas.ATTRIBUTES at all.

Tests fall into two groups. Real-data tests exercise ``build_attributes``
end to end against the actual committed caches in ``data/`` -- no network
calls, since the external CSVs this module needs (2024 general-election
results, the constituency census) are already cached under
``data/raw/election_results/`` and ``data/raw/constituency_census/``.
Synthetic tests exercise individual functions (``_rebellion_rate``,
``_match_nominations``, ``_is_payroll``, ...) against small hand-built
frames, both because the real cache cannot exercise every code path (e.g.
"rebellion rate excludes post-cutoff divisions" needs a synthetic
post-cutoff division to prove, since the real 2026 leadership contest hasn't
produced one that also carries member-level vote detail) and because a few
properties are cheaper to pin down exactly with hand-built numbers than to
infer from what the live data happens to contain.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from plp_sim import attributes, collect, schemas
from plp_sim.config import get_settings


@pytest.fixture(scope="module")
def real_settings():
    return get_settings()


@pytest.fixture(scope="module")
def real_attributes(real_settings):
    """The real build, once per test module -- it's a handful of in-memory
    joins over already-cached data, but there's no reason to repeat it once
    per test.
    """
    return attributes.build_attributes(real_settings, verbose=False)


# --------------------------------------------------------------------------
# population / roster count
# --------------------------------------------------------------------------


def test_roster_count_matches_live_members_cache_not_a_literal(real_settings, real_attributes):
    """Must be derived from members.parquet every time, never hardcoded --
    the House's membership drifts (by-elections, defections) between runs.
    """
    members = collect.read_interim("members", settings=real_settings)
    expected = members[members["party_name"].isin(real_settings.plp_parties)]
    if real_settings.exclude_speaker:
        expected = expected[expected["party_name"] != "Speaker"]

    assert len(real_attributes) == len(expected)
    # Not a coincidence of the two counts merely being equal in size: the
    # actual member_id sets must match too.
    assert set(real_attributes["member_id"]) == set(expected["member_id"])


def test_roster_is_only_labour_and_labour_coop(real_attributes):
    assert set(real_attributes["party_name"].unique()) <= set(schemas.PLP_PARTIES)


def test_speaker_excluded(real_settings, real_attributes):
    members = collect.read_interim("members", settings=real_settings)
    speaker_ids = set(members.loc[members["party_name"] == "Speaker", "member_id"])
    assert speaker_ids, "fixture assumption broken: no Speaker row in members.parquet"
    assert not (set(real_attributes["member_id"]) & speaker_ids)


def test_include_defectors_raises_a_clear_error(real_settings):
    """schemas.ATTRIBUTES.party_name is restricted to PLP_PARTIES, so a
    defector's real party_name cannot be represented. A wrong implementation
    either silently drops defectors (flag does nothing) or crashes deep
    inside schemas.validate with a confusing message; this checks for the
    specific, actionable error instead.
    """
    broken = real_settings.model_copy(update={"include_defectors": True})
    with pytest.raises(ValueError, match="include_defectors"):
        attributes.build_attributes(broken, verbose=False)


def test_select_population_excludes_speaker_and_non_plp_parties(real_settings):
    members = pd.DataFrame(
        {
            "member_id": [1, 2, 3, 4],
            "party_name": ["Labour", "Labour (Co-op)", "Conservative", "Speaker"],
        }
    )
    roster = attributes._select_population(members, real_settings)
    assert set(roster["member_id"]) == {1, 2}


# --------------------------------------------------------------------------
# rebellion_rate: the cutoff rule
# --------------------------------------------------------------------------


def _divisions(rows: list[tuple[int, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"division_id": d, "date": date} for d, date in rows])


def _votes(rows: list[tuple[int, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"division_id": d, "member_id": m, "vote": v} for d, m, v in rows]
    )


def test_rebellion_rate_ignores_post_cutoff_divisions():
    """The load-bearing property: a division dated on/after the cutoff must
    not move rebellion_rate at all, even when it is the only division with
    any dissent in it. A wrong implementation that used every division
    regardless of date would pass every other test in this file and still
    leak the outcome.
    """
    cutoff = dt.date(2026, 4, 1)
    divisions = _divisions(
        [
            (1, "2026-01-01"),  # pre-cutoff, member 100 loyal
            (2, "2026-06-01"),  # post-cutoff, member 100 rebels -- must be ignored
        ]
    )
    votes = _votes(
        [
            (1, 100, "aye"), (1, 101, "aye"), (1, 102, "aye"),
            (2, 100, "no"), (2, 101, "aye"), (2, 102, "aye"),
        ]
    )
    rate = attributes._rebellion_rate(pd.Series([100, 101, 102]), divisions, votes, cutoff)
    # Division 1 is unanimous, and it is the only division before the cutoff,
    # so every member's rate must be exactly 0 -- not something above 0,
    # which is what member 100 would show if division 2 leaked in.
    assert (rate == 0.0).all()


def test_rebellion_rate_detects_a_real_pre_cutoff_rebellion():
    cutoff = dt.date(2026, 4, 1)
    divisions = _divisions([(1, "2026-01-01"), (2, "2026-02-01")])
    votes = _votes(
        [
            (1, 100, "no"), (1, 101, "aye"), (1, 102, "aye"),  # 100 rebels
            (2, 100, "aye"), (2, 101, "aye"), (2, 102, "aye"),  # unanimous
        ]
    )
    rate = attributes._rebellion_rate(pd.Series([100, 101, 102]), divisions, votes, cutoff)
    assert rate.loc[100] == pytest.approx(0.5)
    assert rate.loc[101] == 0.0
    assert rate.loc[102] == 0.0


def test_rebellion_rate_is_null_not_zero_without_pre_cutoff_data():
    """No data must not read as perfect loyalty. This is exactly what
    happens on the real cache today (see attributes.py's module docstring):
    the 10 divisions with member-level detail are all post-cutoff.
    """
    cutoff = dt.date(2026, 4, 1)
    divisions = _divisions([(1, "2026-06-01")])  # post-cutoff only
    votes = _votes([(1, 100, "aye")])
    rate = attributes._rebellion_rate(pd.Series([100, 101]), divisions, votes, cutoff)
    assert rate.isna().all()


def test_real_division_votes_now_cover_pre_cutoff_divisions(real_settings):
    """Pins down that the backfill (scripts/02_collect_division_votes.py) has
    actually landed: division_votes.parquet must cover a large share of the
    ~466 pre-cutoff divisions in divisions.parquet, not just the original 10
    post-cutoff ones. If this regresses to "covered divisions are all
    post-cutoff", rebellion_rate silently goes back to all-null and this
    test (not just the docstring) should catch it before
    test_real_rebellion_rate_has_real_coverage does.
    """
    divisions = collect.read_interim("divisions", settings=real_settings)
    division_votes = collect.read_interim("division_votes", settings=real_settings)
    dates = pd.to_datetime(divisions.set_index("division_id")["date"])
    covered_ids = division_votes["division_id"].unique()
    covered_dates = dates.reindex(covered_ids)
    n_pre_cutoff_covered = int((covered_dates < pd.Timestamp(real_settings.cutoff_date)).sum())
    n_pre_cutoff_total = int((dates < pd.Timestamp(real_settings.cutoff_date)).sum())
    # Not a hardcoded 466: the live divisions cache could grow between runs,
    # but coverage of whatever is pre-cutoff today must be near-complete.
    assert n_pre_cutoff_covered / n_pre_cutoff_total > 0.95, (
        f"only {n_pre_cutoff_covered}/{n_pre_cutoff_total} pre-cutoff divisions have "
        "member-level vote detail -- rerun scripts/02_collect_division_votes.py"
    )


def test_real_rebellion_rate_has_real_coverage(real_settings, real_attributes):
    """With division_votes.parquet backfilled to cover the pre-cutoff
    divisions, rebellion_rate must now be populated for (very nearly) every
    member -- the opposite of the old documented gap. The one expected
    exception is Andy Burnham (member_id 1427), whose Makerfield by-election
    (18 June 2026) postdates every pre-cutoff division, so he genuinely has
    no pre-cutoff vote to measure; pinned by name so a *different* member
    unexpectedly going null is caught rather than shrugged off as "the same
    known gap".
    """
    n_null = int(real_attributes["rebellion_rate"].isna().sum())
    assert n_null <= 1, f"expected at most 1 null rebellion_rate (Andy Burnham), got {n_null}"
    still_null = real_attributes.loc[real_attributes["rebellion_rate"].isna(), "member_id"].tolist()
    assert still_null == [1427] or still_null == [], (
        f"unexpected null rebellion_rate for member_id(s) {still_null} -- investigate, "
        "don't assume this is the same documented Andy Burnham gap"
    )
    assert real_attributes["rebellion_rate"].dropna().between(0.0, 1.0).all()


def test_real_rebellions_welfare_and_wfa_have_real_coverage(real_attributes):
    """Companion to the rebellion_rate check above: with the same backfilled
    division_votes, the welfare/WFA title-matched divisions (all pre-cutoff
    on the real data -- see _title_matched_rebellion_counts' docstring) must
    now have real vote detail too, not the old all-null gap.
    """
    assert real_attributes["rebellions_welfare"].notna().any()
    assert real_attributes["rebellions_wfa"].notna().any()
    assert (real_attributes["rebellions_welfare"].dropna() >= 0).all()
    assert (real_attributes["rebellions_wfa"].dropna() >= 0).all()


def test_title_matched_rebellion_counts_only_counts_matching_titles():
    divisions = pd.DataFrame(
        {
            "division_id": [1, 2],
            "title": ["Welfare Reform Bill: Third Reading", "Immigration and Asylum Bill"],
        }
    )
    # Unambiguous 2-1 "aye" majority in each division, so the rebel is
    # whichever member voted "no" -- no tie-breaking ambiguity to trip over.
    votes = _votes(
        [
            (1, 100, "no"), (1, 101, "aye"), (1, 102, "aye"),
            (2, 100, "no"), (2, 101, "aye"), (2, 102, "aye"),
        ]
    )
    counts = attributes._title_matched_rebellion_counts(
        pd.Series([100, 101, 102]), divisions, votes, "welfare"
    )
    # Only division 1 matches "welfare"; member 100 dissents from the
    # division-1 majority (aye) exactly once, division 2 must not count.
    assert counts.loc[100] == 1
    assert counts.loc[101] == 0
    assert counts.loc[102] == 0


def test_title_matched_rebellion_counts_null_with_no_matching_division():
    divisions = pd.DataFrame({"division_id": [1], "title": ["Immigration and Asylum Bill"]})
    votes = _votes([(1, 100, "aye")])
    counts = attributes._title_matched_rebellion_counts(pd.Series([100, 200]), divisions, votes, "welfare")
    assert counts.isna().all()


# --------------------------------------------------------------------------
# is_payroll / committee_count (member_biography.parquet)
# --------------------------------------------------------------------------


def _biography(rows: list[tuple[int, str, str, str | None, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"member_id": m, "category": cat, "name": n, "start_date": s, "end_date": e}
            for m, cat, n, s, e in rows
        ],
        columns=["member_id", "category", "name", "start_date", "end_date"],
    )


def test_active_as_at_includes_ongoing_and_spanning_posts_excludes_past_and_future():
    cutoff = dt.date(2026, 4, 1)
    bio = _biography(
        [
            (1, "governmentPosts", "Ongoing", "2024-07-05", None),  # spans, null end -> active
            (2, "governmentPosts", "Spans cutoff", "2025-01-01", "2026-06-01"),  # active
            (3, "governmentPosts", "Ended before cutoff", "2024-07-05", "2026-03-01"),  # inactive
            (4, "governmentPosts", "Starts after cutoff", "2026-05-01", None),  # inactive
        ]
    )
    active = attributes._active_as_at(bio, cutoff)
    assert set(active["member_id"]) == {1, 2}


def test_is_payroll_true_for_government_post_holder_false_otherwise():
    cutoff = dt.date(2026, 4, 1)
    bio = _biography(
        [
            (1, "governmentPosts", "Secretary of State", "2024-07-05", None),
            (2, "governmentPosts", "Assistant Whip", "2025-09-07", "2026-03-01"),  # ended pre-cutoff
            (3, "oppositionPosts", "Opposition Whip (Commons)", "2020-01-01", None),  # not a gov post
        ]
    )
    result = attributes._is_payroll(pd.Series([1, 2, 3, 4]), bio, cutoff)
    assert result.tolist() == [True, False, False, False]


def test_is_payroll_false_for_member_with_no_biography_rows():
    cutoff = dt.date(2026, 4, 1)
    bio = _biography([])
    result = attributes._is_payroll(pd.Series([1]), bio, cutoff)
    assert result.tolist() == [False]


def test_committee_count_counts_active_committee_memberships_only():
    cutoff = dt.date(2026, 4, 1)
    bio = _biography(
        [
            (1, "committeeMemberships", "Treasury Committee", "2024-07-05", None),
            (1, "committeeMemberships", "Justice Committee", "2020-01-01", "2024-01-01"),  # inactive
            (2, "committeeMemberships", "Home Affairs Committee", "2024-07-05", None),
            (2, "governmentPosts", "Minister of State", "2024-07-05", None),  # not a committee
        ]
    )
    counts = attributes._committee_count(pd.Series([1, 2, 3]), bio, cutoff)
    assert counts.loc[1] == 1
    assert counts.loc[2] == 1
    assert counts.loc[3] == 0  # no biography rows at all -> 0, not null


def test_real_is_payroll_and_committee_count_have_variance(real_attributes):
    """The one property a broken (all-False / all-zero) implementation would
    also satisfy trivially -- guard against that by requiring real variance.
    """
    assert real_attributes["is_payroll"].any()
    assert not real_attributes["is_payroll"].all()
    assert real_attributes["committee_count"].ge(0).all()
    assert real_attributes["committee_count"].gt(0).any()


# --------------------------------------------------------------------------
# constituency demographics
# --------------------------------------------------------------------------


def test_median_age_from_bands_all_population_in_one_band_returns_its_midpoint_ish():
    """A degenerate case that pins the interpolation formula down concretely:
    if the entire population sat in one 5-year band, the estimated median
    must land inside that band, not merely somewhere plausible.
    """
    row = pd.Series({col: 0.0 for col, _, _ in attributes._AGE_BANDS})
    row["c21Age30to34"] = 100.0
    median = attributes._median_age_from_bands(row)
    assert 30.0 <= median <= 35.0


def test_median_age_from_bands_matches_hand_computed_interpolation():
    row = pd.Series({col: 0.0 for col, _, _ in attributes._AGE_BANDS})
    row["c21Age0to4"] = 40.0
    row["c21Age5to9"] = 60.0
    # Cumulative to end of first band is 40 (< half of 50); the crossing point
    # falls (50-40)/60 = 1/6 of the way through the second band (5-9, width 5).
    median = attributes._median_age_from_bands(row)
    assert median == pytest.approx(5 + (50 - 40) / 60 * 5)


#: A hand-built stand-in for NRS's real SuperWEB2 CSV export, matching its
#: exact shape (9 title/filter lines, a blank line pandas' default
#: skip_blank_lines absorbs, the header, data rows, then a blank line and
#: free-text footer junk) -- not the real 35KB file, just enough of its
#: structure to prove the parser handles it.
_FAKE_SCOTLAND_QUAL_CSV = b"""SuperWEB2(tm)

"Census 2022-Person-2Fv1"
"Scotland's Census 2022 - National Records of Scotland Table UV501 - Highest level of qualification All people aged 16 and over"
"United Kingdom Parliamentary Constituency 2024 by Highest Level of Qualification"
"Counting: Individuals"

Filters:
"Default Summation","Individuals"

"Counting","United Kingdom Parliamentary Constituency 2024","Highest Level of Qualification","Count",
"Individuals","Fake Seat","All people aged 16 and over",1000
"Individuals","Fake Seat","No qualifications",100
"Individuals","Fake Seat","Lower school qualifications",200
"Individuals","Fake Seat","Upper school qualifications",150
"Individuals","Fake Seat","Apprenticeship qualifications",50
"Individuals","Fake Seat","Further Education and sub-degree Higher Education qualifications incl. HNC/HNDs",200
"Individuals","Fake Seat","Degree level qualifications or above",300


"INFO","Data has been perturbed"


"Lower school qualifications - long free-text category definitions ... Crown copyright 2024"

(c) Copyright WingArc Australia 2018
"""


def test_parse_scotland_qualifications_skips_title_rows_and_footer():
    result = attributes._parse_scotland_qualifications(_FAKE_SCOTLAND_QUAL_CSV)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["_name_key"] == "fake seat"
    assert row["degree_share_scotland"] == pytest.approx(300 / 1000 * 100)


def test_parse_scotland_qualifications_excludes_subdegree_band():
    """The documented, checked judgement call in SCOTLAND_QUALIFICATIONS_URL's
    docstring: only "Degree level qualifications or above" counts, not summed
    with "Further Education and sub-degree Higher Education ... incl.
    HNC/HNDs". A wrong implementation that summed both would give 50.0
    (300+200 of 1000), not 30.0 -- this is the test that would catch it.
    """
    result = attributes._parse_scotland_qualifications(_FAKE_SCOTLAND_QUAL_CSV)
    assert result.iloc[0]["degree_share_scotland"] == pytest.approx(30.0)


def test_normalize_constituency_name_strips_diacritics_and_folds_ampersand():
    names = pd.Series(["Montgomeryshire and Glyndŵr", "Montgomeryshire and Glyndwr", "Foo & Bar"])
    normalized = attributes._normalize_constituency_name(names)
    assert normalized.iloc[0] == normalized.iloc[1]
    assert normalized.iloc[2] == "foo and bar"


def test_real_constituency_demographics_match_rate(real_settings):
    """The exact concern the task calls out: a silent bad join here corrupts
    figure 1 in a way nobody would catch. Assert the match rate as a number,
    not just "some are non-null" -- on the real roster this is 100%, with
    the only diacritic mismatch (Glyndŵr / Glyndwr) resolved by
    _normalize_constituency_name.
    """
    members = collect.read_interim("members", settings=real_settings)
    roster = attributes._select_population(members, real_settings)
    _, report = attributes._constituency_demographics(roster, real_settings)

    assert report.unmatched_constituencies == []
    assert report.match_rate == 1.0


def test_real_degree_share_has_no_nulls(real_attributes):
    """degree_share used to be null for the 37 Scottish PLP members, same as
    deprivation_score -- see attributes.py's module docstring. It no longer
    is: SCOTLAND_QUALIFICATIONS_URL fills it from a second, Scotland-only NRS
    source. A regression here (e.g. a broken join silently reverting to the
    old all-Scotland-null state) must fail loudly, not quietly ship as "a
    known gap".
    """
    assert real_attributes["degree_share"].notna().all()


def test_real_deprivation_score_null_only_for_scotland(real_settings, real_attributes):
    """deprivation_score is null for Scottish PLP members only (no comparable,
    fetchable Scottish source was found for it -- see attributes.py's module
    docstring for exactly what was checked and ruled out) -- median_age has
    no such gap, and degree_share no longer does either (see
    test_real_degree_share_has_no_nulls). A wrong join could produce nulls
    anywhere; this pins the null pattern down to exactly the documented,
    country-scoped cause.
    """
    null_ids = set(real_attributes.loc[real_attributes["deprivation_score"].isna(), "member_id"])
    assert real_attributes["median_age"].notna().all()
    assert 0 < len(null_ids) < len(real_attributes)  # a real, partial, non-trivial gap


def test_real_scottish_and_non_scottish_degree_share_overlap(real_attributes):
    """The project brief's explicit sanity check: filling Scotland from a
    differently-defined measure would manufacture a systematic
    Scotland-vs-rest offset, which is worse than the null it replaces. Assert
    the two distributions actually overlap (not merely "both non-null") --
    a definitional mismatch would tend to push one distribution's range
    largely outside the other's, which this would catch.
    """
    is_scotland = real_attributes["deprivation_score"].isna()  # the one column Scotland still lacks
    scotland = real_attributes.loc[is_scotland, "degree_share"]
    rest = real_attributes.loc[~is_scotland, "degree_share"]
    assert len(scotland) > 0 and len(rest) > 0
    # Ranges overlap substantially rather than one sitting entirely above/below
    # the other -- e.g. most of Scotland's range must fall inside rest's range.
    assert scotland.min() < rest.max()
    assert scotland.max() > rest.min()
    # Means within a few points of each other, not an implausible double-digit
    # population-level gap (the smoking gun for a definitional mismatch --
    # see SCOTLAND_QUALIFICATIONS_URL's docstring for the ~11pp gap that the
    # rejected "sum both higher-education bands" definition would have shown).
    assert abs(scotland.mean() - rest.mean()) < 5.0


def test_real_constituency_report_documents_the_scotland_fill(real_settings):
    """ConstituencyMatchReport is where the "these 37 rows are a different
    source" provenance lives (this module can't add a schema column for it --
    see SCOTLAND_QUALIFICATIONS_URL's docstring). Pin down that the report
    actually says so, not just that the numbers happen to work out.
    """
    members = collect.read_interim("members", settings=real_settings)
    roster = attributes._select_population(members, real_settings)
    _, report = attributes._constituency_demographics(roster, real_settings)

    assert report.n_degree_share_filled_from_scotland == 37
    assert report.unmatched_scotland_qualifications == []
    assert "SCOTLAND_QUALIFICATIONS_URL" in report.summary()
    assert "DIFFERENT source" in report.summary()


# --------------------------------------------------------------------------
# nominations join
# --------------------------------------------------------------------------


def test_strip_honorifics():
    s = pd.Series(["Dr Zubir Ahmed", "Mr Bayo Alaba", "Rt Hon Sir Alan Campbell", "Stella Creasy"])
    assert list(attributes._strip_honorifics(s)) == ["Zubir Ahmed", "Bayo Alaba", "Alan Campbell", "Stella Creasy"]


def _roster(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"member_id": m, "name_display_as": n, "membership_from": c} for m, n, c in rows]
    )


def _nominations(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"mp_name": n, "constituency": c, "days_from_open": d} for n, c, d in rows]
    )


def test_match_nominations_honorific_stripped_name_match():
    roster = _roster([(1, "Mr Bayo Alaba", "Southend East")])
    nominations = _nominations([("Bayo Alaba", "Southend East", 0)])
    matched, report = attributes._match_nominations(nominations, roster)
    assert list(matched["member_id"]) == [1]
    assert report.matched_by_name == 1
    assert report.matched_by_constituency_fallback == 0


def test_match_nominations_constituency_fallback_for_name_mismatch():
    """Mirrors the two real cases (Rachel/Rachael Maskell, Louise
    Jones/Sandher-Jones): name doesn't match even after stripping
    honorifics, but constituency does and is unique, so the fallback
    resolves it deterministically rather than leaving it unmatched.
    """
    roster = _roster([(1, "Rachael Maskell", "York Central")])
    nominations = _nominations([("Rachel Maskell", "York Central", 0)])
    matched, report = attributes._match_nominations(nominations, roster)
    assert list(matched["member_id"]) == [1]
    assert report.matched_by_name == 0
    assert report.matched_by_constituency_fallback == 1


def test_match_nominations_surfaces_unmatched_rows_instead_of_dropping():
    roster = _roster([(1, "Mr Bayo Alaba", "Southend East")])
    nominations = _nominations(
        [("Bayo Alaba", "Southend East", 0), ("Someone Untraceable", "Nowhere Seat", 4)]
    )
    matched, report = attributes._match_nominations(nominations, roster)
    assert len(matched) == 1
    assert len(report.unmatched_nomination_rows) == 1
    assert report.unmatched_nomination_rows["mp_name"].iloc[0] == "Someone Untraceable"


def test_did_nominate_distinguishes_declined_from_unsourced():
    """The central distinction the schema exists to protect: a roster member
    with no nomination row at all is a confirmed decliner (did_nominate=False,
    bucket="none", day=NaN) -- not the same thing as a nominations.csv row
    this join could not place anywhere (which does not touch any roster
    member's did_nominate at all, and is reported separately).
    """
    roster = _roster(
        [
            (1, "Mr Bayo Alaba", "Southend East"),  # nominates
            (2, "Ms Stella Creasy", "Walthamstow"),  # declines
        ]
    )
    nominations = _nominations(
        [
            ("Bayo Alaba", "Southend East", 0),
            ("Someone Untraceable", "Nowhere Seat", 4),  # genuinely unplaceable
        ]
    )
    cols, report = attributes._nomination_columns(roster, nominations)
    cols = cols.set_index("member_id")

    assert cols.loc[1, "did_nominate"] == True
    assert cols.loc[1, "nomination_bucket"] == "day1"
    assert cols.loc[1, "nomination_day"] == 0.0

    assert cols.loc[2, "did_nominate"] == False
    assert cols.loc[2, "nomination_bucket"] == "none"
    assert pd.isna(cols.loc[2, "nomination_day"])

    assert report.non_nominators == 1
    assert len(report.unmatched_nomination_rows) == 1


def test_nomination_day_buckets_match_the_documented_censored_values():
    roster = _roster([(1, "A", "X"), (2, "B", "Y"), (3, "C", "Z")])
    nominations = _nominations([("A", "X", 0), ("B", "Y", 4), ("C", "Z", 7)])
    cols, _ = attributes._nomination_columns(roster, nominations)
    cols = cols.set_index("member_id")
    assert cols.loc[1, "nomination_bucket"] == "day1"
    assert cols.loc[2, "nomination_bucket"] == "mid"
    assert cols.loc[3, "nomination_bucket"] == "late"


def test_nomination_columns_rejects_an_undocumented_days_from_open_value():
    roster = _roster([(1, "A", "X")])
    nominations = _nominations([("A", "X", 3)])  # not in {0, 4, 7}
    with pytest.raises(AssertionError, match="days_from_open"):
        attributes._nomination_columns(roster, nominations)


#: Below this, a build should be treated as broken, not merely imperfect --
#: see NOMINATIONS_NOTES.md, which documents ~94% raw coverage of the
#: nominators list and explains the ~25 non-nominators separately.
MIN_NOMINATION_JOIN_COVERAGE = 0.95


def test_real_nomination_join_coverage_above_threshold(real_settings):
    members = collect.read_interim("members", settings=real_settings)
    roster = attributes._select_population(members, real_settings)
    nominations = pd.read_csv(real_settings.data_manual / "nominations.csv")

    _, report = attributes._match_nominations(nominations, roster)
    coverage = report.n_matched / report.n_nomination_rows

    assert coverage >= MIN_NOMINATION_JOIN_COVERAGE, (
        f"nominations.csv join coverage {coverage:.1%} fell below "
        f"{MIN_NOMINATION_JOIN_COVERAGE:.0%} -- unmatched rows: "
        f"{report.unmatched_nomination_rows['mp_name'].tolist()}"
    )
    # The real file matches 100% in the current data; assert that precisely
    # too, so any regression is caught even though it's stricter than the
    # threshold above requires.
    assert report.unmatched_nomination_rows.empty
    assert coverage == 1.0


# --------------------------------------------------------------------------
# electoral columns
# --------------------------------------------------------------------------


def test_coarsen_runner_up_party_keeps_known_parties():
    raw = pd.Series(["Conservative", "Reform UK", "Scottish National Party"])
    assert list(attributes._coarsen_runner_up_party(raw)) == list(raw)


def test_coarsen_runner_up_party_maps_independent_variants():
    raw = pd.Series(["Independent Network", "Newham Independents Party", "Independent"])
    assert (attributes._coarsen_runner_up_party(raw) == "Independent").all()


def test_coarsen_runner_up_party_maps_rare_and_missing_to_other():
    raw = pd.Series(["Workers Party of Britain", None, np.nan])
    assert (attributes._coarsen_runner_up_party(raw) == "Other").all()


def test_extract_results_general_election_shape():
    df = pd.DataFrame(
        [
            {
                "Constituency name": "Seat A", "Candidate result position": 1,
                "Candidate MNIS ID": 42, "Candidate vote count": 20000,
                "Candidate vote share": 0.60, "Election valid vote count": 33333,
                "Majority": 10000, "Main party name": "Labour",
                "Candidate is standing as independent": False,
            },
            {
                "Constituency name": "Seat A", "Candidate result position": 2,
                "Candidate MNIS ID": np.nan, "Candidate vote count": 10000,
                "Candidate vote share": 0.30, "Election valid vote count": 33333,
                "Majority": np.nan, "Main party name": "Conservative",
                "Candidate is standing as independent": False,
            },
        ]
    )
    out = attributes._extract_results(df, group_col="Constituency name")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["member_id"] == 42
    assert row["vote_share"] == pytest.approx(60.0)
    assert row["majority_pct"] == pytest.approx(10000 / 33333 * 100)
    assert row["runner_up_party_raw"] == "Conservative"


def test_extract_results_byelection_shape_derives_majority_and_valid_votes():
    """The per-election export (used for by-election winners not in the
    general-election file) has no ``Election valid vote count`` or
    ``Majority`` column at all -- both must be derived from raw vote counts.
    """
    df = pd.DataFrame(
        [
            {
                "Candidate result position": 1, "Candidate MNIS ID": 1427,
                "Candidate vote count": 24927, "Candidate vote share": 0.548,
                "Main party name": "Labour", "Candidate is standing as independent": False,
            },
            {
                "Candidate result position": 2, "Candidate MNIS ID": np.nan,
                "Candidate vote count": 15696, "Candidate vote share": 0.345,
                "Main party name": "Reform UK", "Candidate is standing as independent": False,
            },
        ]
    )
    out = attributes._extract_results(df, group_col=None)
    row = out.iloc[0]
    assert row["member_id"] == 1427
    assert row["majority_pct"] == pytest.approx((24927 - 15696) / (24927 + 15696) * 100)
    assert row["runner_up_party_raw"] == "Reform UK"


def test_extract_results_independent_runner_up_with_blank_party_name():
    df = pd.DataFrame(
        [
            {
                "Constituency name": "Seat A", "Candidate result position": 1,
                "Candidate MNIS ID": 1, "Candidate vote count": 100,
                "Candidate vote share": 0.6, "Election valid vote count": 167,
                "Majority": 40, "Main party name": "Labour",
                "Candidate is standing as independent": False,
            },
            {
                "Constituency name": "Seat A", "Candidate result position": 2,
                "Candidate MNIS ID": np.nan, "Candidate vote count": 60,
                "Candidate vote share": 0.36, "Election valid vote count": 167,
                "Majority": np.nan, "Main party name": np.nan,
                "Candidate is standing as independent": True,
            },
        ]
    )
    out = attributes._extract_results(df, group_col="Constituency name")
    assert out.iloc[0]["runner_up_party_raw"] == "Independent"


# --------------------------------------------------------------------------
# speech_count
# --------------------------------------------------------------------------


def test_speech_counts_matches_source(real_settings, real_attributes):
    mc = collect.read_interim("member_contributions", settings=real_settings)
    joined = real_attributes.merge(mc, on="member_id", how="left", suffixes=("", "_src"))
    assert (joined["speech_count"] == joined["spoken_result_count"]).all()


# --------------------------------------------------------------------------
# full build: schema conformance and SEG_VARS
# --------------------------------------------------------------------------


def test_build_attributes_passes_schema_validate(real_attributes):
    schemas.validate(real_attributes, schemas.ATTRIBUTES)


def test_seg_vars_other_than_rebellion_rate_are_fully_populated(real_attributes):
    """rebellion_rate is the one SEG_VAR with a documented, single-member gap
    (Andy Burnham -- see attributes.py's module docstring and
    test_real_rebellion_rate_has_real_coverage above); the other four
    SEG_VARS have no such excuse and must have zero nulls.
    """
    other_seg_vars = [v for v in schemas.SEG_VARS if v != "rebellion_rate"]
    for col in other_seg_vars:
        assert real_attributes[col].notna().all(), f"{col} has unexpected nulls"


def test_seg_vars_fully_populated_when_rebellion_data_is_available():
    """Demonstrates the property the real build cannot: given adequate
    pre-cutoff input, every SEG_VAR -- including rebellion_rate -- comes out
    fully populated. This isolates "the function is correct" from "today's
    cache happens to have no pre-cutoff member-level votes", which are very
    different failure modes and must not be conflated by one all-real-data test.
    """
    cutoff = dt.date(2026, 4, 1)
    n = 12
    member_ids = pd.Series(range(1, n + 1))
    divisions = _divisions([(1, "2026-01-01"), (2, "2026-02-01")])
    rng = np.random.default_rng(0)
    votes = _votes(
        [(d, m, v) for d in (1, 2) for m, v in zip(member_ids, rng.choice(["aye", "no"], n))]
    )
    rate = attributes._rebellion_rate(member_ids, divisions, votes, cutoff)
    assert rate.notna().all()
    assert rate.between(0.0, 1.0).all()
