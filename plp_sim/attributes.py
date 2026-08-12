"""Builds the attribute table: one row per Labour MP, conforming to
``schemas.ATTRIBUTES``.

Computes rather than collects. Inputs are the cached tables in
``data/interim/``, the hand-compiled ``data/manual/nominations.csv``, and 2024
election results, which this module fetches and caches itself because they do
not come from a Parliament API.

Read ``data/manual/NOMINATIONS_NOTES.md`` before changing anything nomination
related; it bounds what those columns can claim.

One deliberate gap: ``deprivation_score`` is null for the 37 Scottish seats.
The source recuts England and Wales census data onto 2024 boundaries and never
merged Scotland's deprivation table. Scotland's own measure (SIMD) is defined
differently and is not comparable, so filling the gap with it would manufacture
a Scotland effect. Left null instead. ``median_age`` and ``degree_share`` are
populated for Scotland, from comparable sources.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from plp_sim import collect, schemas
from plp_sim.config import Settings, get_settings

# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------

#: Parties the README documents as ex-Labour breakaways founded by MPs who
#: left the Labour whip -- as opposed to a pre-existing minor party. This is
#: the only slice of "ex-Labour independents" this module can safely add
#: under ``config.include_defectors``: a plain ``party_name == "Independent"``
#: mixes genuine ex-Labour defectors with MPs who were never Labour (e.g.
#: Gaza-independents elected as independents from the start), and nothing in
#: the collected data distinguishes the two -- collect.py's Members/Search
#: flatten keeps only ``latestParty``, never a party-history endpoint. Reported
#: as a real gap rather than guessed at with a name-based heuristic.
DEFECTOR_PARTIES: tuple[str, ...] = ("Your Party", "Restore Britain")


def _select_population(members: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """The PLP roster per ``settings``: never a literal count.

    ``config.include_defectors=True`` is deliberately unsupported here (see
    module docstring / ``DEFECTOR_PARTIES``): a defector's real party_name
    (Independent / Your Party / Restore Britain) cannot be represented under
    ``schemas.ATTRIBUTES``, which restricts ``party_name`` to
    ``schemas.PLP_PARTIES``. Raising here gives a specific, actionable error
    instead of a confusing failure deep inside ``schemas.validate``.
    """
    if settings.include_defectors:
        raise ValueError(
            "config.include_defectors=True has no representable output under the "
            "current schema: schemas.ATTRIBUTES restricts party_name to "
            "schemas.PLP_PARTIES, so a defector's real party_name (Independent / "
            "Your Party / Restore Britain) would fail schemas.validate(). This is a "
            "genuine gap between config.py's flag and schemas.py's constraint that "
            "attributes.py cannot paper over without owning one of those two files -- "
            "it does not. Run with include_defectors=False (the default), or extend "
            "the schema first."
        )
    roster = members[members["party_name"].isin(settings.plp_parties)].copy()
    if settings.exclude_speaker:
        # Belt-and-braces: plp_parties already excludes "Speaker", but this
        # keeps the exclusion explicit and correct even if plp_parties ever
        # changed to something broader.
        roster = roster[roster["party_name"] != "Speaker"]
    return roster.reset_index(drop=True)


# --------------------------------------------------------------------------
# electoral columns (2024 GE results + one by-election)
# --------------------------------------------------------------------------

#: 2024 UK general election, per electionresults.parliament.uk's own numbering.
GENERAL_ELECTION_ID = 6
GENERAL_ELECTION_CANDIDACIES_URL = (
    f"https://electionresults.parliament.uk/general-elections/{GENERAL_ELECTION_ID}/candidacies.csv"
)

#: By-election winners postdating the 2024 GE are not in that file at all --
#: it is scoped to one election. There is no generic "every by-election since
#: X" endpoint this module walks automatically; each by-election has its own
#: numeric election id on electionresults.parliament.uk, found by hand from
#: the member's own page (https://electionresults.parliament.uk/members/<id>).
#: This is that manual lookup table. Andy Burnham (member_id 1427, Makerfield,
#: 18 June 2026) is the only current PLP member affected. If another
#: by-election winner joins the roster, ``build_attributes`` raises a clear,
#: specific error naming the missing member_id rather than silently emitting
#: a null electoral column -- add them here.
BYELECTION_RESULT_URLS: dict[int, str] = {
    1427: "https://electionresults.parliament.uk/elections/4555/candidate-results.csv",
}


def _ensure_cached(url: str, cache_path: Path) -> Path:
    """Fetch a URL's raw bytes to ``cache_path`` if not already there; return
    ``cache_path`` either way.

    Mirrors ``collect.py``'s cache-or-fetch contract (same request always
    resolves to the same file; a rerun with the file already on disk makes
    zero network calls) without reusing its private, Parliament-API-shaped
    helpers, which wrap every response in a JSON envelope this isn't. A
    ``<name>.meta.json`` sidecar records when and from where it was fetched.
    """
    if not cache_path.exists():
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        meta_path = cache_path.with_name(cache_path.name + ".meta.json")
        meta_path.write_text(
            json.dumps({"fetched_at": dt.datetime.now(dt.UTC).isoformat(), "url": url}, indent=2)
        )
    return cache_path


def _cached_csv_fetch(url: str, cache_path: Path) -> pd.DataFrame:
    """A clean, single-header CSV, fetched (and cached) via ``_ensure_cached``.

    Not every CSV this module needs is this simple -- see
    ``_fetch_scotland_qualifications`` for one that isn't (a SuperWEB2 export
    with title rows and a text footer) and does its own parsing on top of the
    same ``_ensure_cached`` contract rather than through this function.
    """
    _ensure_cached(url, cache_path)
    return pd.read_csv(cache_path, encoding="utf-8-sig")


def _fetch_general_election_results(settings: Settings) -> pd.DataFrame:
    cache_path = settings.data_raw / "election_results" / "general_election_2024_candidacies.csv"
    return _cached_csv_fetch(GENERAL_ELECTION_CANDIDACIES_URL, cache_path)


def _extract_results(df: pd.DataFrame, *, group_col: str | None) -> pd.DataFrame:
    """Reduce a candidacies-style export (one row per candidate) to one row
    per winning, MNIS-identified candidate: member_id, vote_share (0-100),
    majority_pct (0-100), runner_up_party_raw.

    Handles both shapes electionresults.parliament.uk actually serves: the
    general-election export (many constituencies, ``Election valid vote
    count`` and ``Majority`` columns already computed on the winner's row)
    and a single-election by-election export (no constituency column, no
    valid-vote or majority column -- both derived here from raw vote counts).
    """
    work = df if group_col is not None else df.assign(_grp=0)
    key = group_col or "_grp"

    rows: list[dict[str, object]] = []
    for _, g in work.groupby(key):
        winners = g[g["Candidate result position"] == 1]
        if winners.empty or pd.isna(winners.iloc[0].get("Candidate MNIS ID")):
            continue
        w = winners.iloc[0]
        runner_ups = g[g["Candidate result position"] == 2]

        if "Election valid vote count" in g.columns and pd.notna(w.get("Election valid vote count")):
            valid_votes = float(w["Election valid vote count"])
        else:
            valid_votes = float(g["Candidate vote count"].sum())

        if "Majority" in g.columns and pd.notna(w.get("Majority")):
            majority = float(w["Majority"])
        else:
            runner_up_votes = float(runner_ups.iloc[0]["Candidate vote count"]) if not runner_ups.empty else 0.0
            majority = float(w["Candidate vote count"]) - runner_up_votes

        if not runner_ups.empty:
            ru = runner_ups.iloc[0]
            if pd.notna(ru["Main party name"]):
                runner_up_raw = str(ru["Main party name"])
            elif bool(ru.get("Candidate is standing as independent", False)):
                runner_up_raw = "Independent"
            else:
                runner_up_raw = None
        else:
            runner_up_raw = None

        rows.append(
            {
                "member_id": int(w["Candidate MNIS ID"]),
                "vote_share": float(w["Candidate vote share"]) * 100.0,
                "majority_pct": (majority / valid_votes * 100.0) if valid_votes else np.nan,
                "runner_up_party_raw": runner_up_raw,
            }
        )
    return pd.DataFrame(rows, columns=["member_id", "vote_share", "majority_pct", "runner_up_party_raw"])


def _coarsen_runner_up_party(raw: pd.Series) -> pd.Series:
    """Map a free-text runner-up party name onto ``schemas.RUNNER_UP_PARTIES``.

    Exact members of the allowed set pass through. Anything with "independent"
    in it (e.g. "Independent Network", "Newham Independents Party") is treated
    as ``Independent`` -- a documented judgement call, not a silent guess.
    Everything else, including a genuinely missing runner-up (which would
    otherwise violate the non-nullable schema), becomes ``Other``.
    """
    allowed = set(schemas.RUNNER_UP_PARTIES)

    def coarsen(v: object) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "Other"
        s = str(v)
        if s in allowed:
            return s
        if "independent" in s.lower():
            return "Independent"
        return "Other"

    return raw.map(coarsen)


def _combine_election_results(roster: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Electoral columns for every roster member with a findable result.

    Combines the 2024 GE export with any by-election results named in
    ``BYELECTION_RESULT_URLS``. Does not itself guarantee full coverage --
    ``build_attributes`` asserts that explicitly and names any gap.
    """
    ge_results = _fetch_general_election_results(settings)
    frames = [_extract_results(ge_results, group_col="Constituency name")]

    roster_ids = set(roster["member_id"])
    for member_id, url in BYELECTION_RESULT_URLS.items():
        if member_id not in roster_ids:
            continue
        cache_path = settings.data_raw / "election_results" / f"byelection_{member_id}.csv"
        by_df = _cached_csv_fetch(url, cache_path)
        frames.append(_extract_results(by_df, group_col=None))

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="member_id", keep="first")
    combined["runner_up_party"] = _coarsen_runner_up_party(combined["runner_up_party_raw"])
    return combined[["member_id", "majority_pct", "vote_share", "runner_up_party"]], ge_results


def _first_elected_and_intake(roster: pd.DataFrame, ge_results: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """``first_elected`` is the start of the member's current, continuous
    Commons membership (``membership_start_date`` from the Members API --
    real, sourced data already collected, no GE file needed for this part).

    ``is_2024_intake`` is exactly "started on the 2024 general election's own
    polling date", read from the fetched GE file rather than hardcoded, so it
    can't silently drift from the electoral data it's paired with.
    """
    first_elected = pd.to_datetime(roster["membership_start_date"])
    polling_date = pd.to_datetime(ge_results["General election polling date"].iloc[0]).normalize()
    is_2024_intake = first_elected.dt.normalize() == polling_date
    return first_elected, is_2024_intake


# --------------------------------------------------------------------------
# rebellion rate (pre-cutoff only) and the two post-cutoff rebellion counts
# --------------------------------------------------------------------------


def _majority_vote_per_division(votes: pd.DataFrame) -> pd.Series:
    """Modal Aye/No per division among the roster rows given.

    The whip-proxy definition. The Parliament APIs expose no whip direction or
    party line for a division, only the named Aye/No lists, so "the Labour majority
    position" is defined as whichever way more roster members actually voted.
    Abstentions are absent from both lists and so do not count. A rebel is a member
    whose vote differs from that mode.

    This is a revealed majority, not a sourced whip instruction: on a division with
    few Labour members present, "the majority" can be a small number. It is the
    conventional definition where no whip record exists.

    Ties resolve to the first-encountered vote value, deterministically.
    """
    return votes.groupby("division_id")["vote"].agg(lambda s: s.value_counts().idxmax())


def _rebellion_rate(
    member_ids: pd.Series,
    divisions: pd.DataFrame,
    division_votes: pd.DataFrame,
    cutoff_date: dt.date,
) -> pd.Series:
    """Per-member share of strictly-pre-cutoff divisions voted against the
    PLP majority in that division (see ``_majority_vote_per_division`` for
    the exact whip-proxy definition), among divisions the member actually
    voted in.

    A member with zero pre-cutoff divisions recorded gets NaN, not 0: "no
    data" and "perfect loyalty" are different claims. On the real,
    collected data (``division_votes.parquet`` after
    ``scripts/02_collect_division_votes.py``, which pulls vote detail for
    every one of the ~466 pre-cutoff divisions in ``divisions.parquet``)
    this is null for exactly one of 405 roster members -- Andy Burnham
    (member_id 1427), whose Makerfield by-election (18 June 2026) postdates
    every pre-cutoff division, so he genuinely has no pre-cutoff vote to
    measure. Everyone else has real coverage. Tested against synthetic
    pre/post-cutoff data in tests/test_attributes.py; those synthetic tests
    still pass unchanged and remain the place that pins down the
    cutoff-exclusion property itself, independent of whatever the live cache
    happens to contain.
    """
    cutoff_ts = pd.Timestamp(cutoff_date)
    pre_cutoff_ids = set(divisions.loc[pd.to_datetime(divisions["date"]) < cutoff_ts, "division_id"])

    roster_ids = set(member_ids)
    votes = division_votes[
        division_votes["division_id"].isin(pre_cutoff_ids) & division_votes["member_id"].isin(roster_ids)
    ]
    if votes.empty:
        return pd.Series(np.nan, index=pd.Index(member_ids, name="member_id"), dtype="float64")

    majority = _majority_vote_per_division(votes)
    rebelled = votes.join(majority.rename("majority_vote"), on="division_id")
    rebelled = rebelled.assign(rebelled=rebelled["vote"] != rebelled["majority_vote"])
    per_member = rebelled.groupby("member_id")["rebelled"].mean()
    return per_member.reindex(member_ids).astype("float64")


def _title_matched_rebellion_counts(
    member_ids: pd.Series,
    divisions: pd.DataFrame,
    division_votes: pd.DataFrame,
    title_contains: str,
) -> pd.Series:
    """Per-member rebellion count across divisions whose title contains
    ``title_contains`` and which have member-level detail.

    NaN for a member with no matching divisions recorded, meaning no data rather
    than zero rebellions.

    Deliberately not cutoff-filtered, unlike ``_rebellion_rate``. These columns are
    ``POST_CUTOFF_VARS``, validation targets rather than construction inputs, so a
    post-cutoff division is expected here. Using one for ``rebellion_rate`` is what
    would leak.

    Uses the same whip-proxy majority as ``_majority_vote_per_division``.
    """
    matched_ids = set(divisions.loc[divisions["title"].str.contains(title_contains, case=False, na=False), "division_id"])
    roster_ids = set(member_ids)
    votes = division_votes[
        division_votes["division_id"].isin(matched_ids) & division_votes["member_id"].isin(roster_ids)
    ]
    if votes.empty:
        return pd.Series(
            pd.array([pd.NA] * len(member_ids), dtype="Int64"), index=pd.Index(member_ids, name="member_id")
        )

    majority = _majority_vote_per_division(votes)
    rebelled = votes.join(majority.rename("majority_vote"), on="division_id")
    rebelled = rebelled.assign(rebelled=(rebelled["vote"] != rebelled["majority_vote"]).astype("int64"))
    per_member = rebelled.groupby("member_id")["rebelled"].sum()
    return per_member.reindex(member_ids).astype("Int64")


# --------------------------------------------------------------------------
# payroll status and committee membership, both as at cutoff_date
# (data/interim/member_biography.parquet, via collect.fetch_all_member_biographies)
# --------------------------------------------------------------------------


def _active_as_at(bio: pd.DataFrame, cutoff_date: dt.date) -> pd.DataFrame:
    """Rows of a (member_id, category, name, start_date, end_date) frame whose
    date range spans ``cutoff_date`` -- ``start_date <= cutoff <= end_date``,
    with a null ``end_date`` read as "still ongoing" (the API's own convention
    for a current post, confirmed against live data: e.g. the sitting Home
    Secretary's post has ``end_date=None`` while every ex-minister's has a
    real end date).
    """
    cutoff_ts = pd.Timestamp(cutoff_date)
    start = pd.to_datetime(bio["start_date"])
    end = pd.to_datetime(bio["end_date"])
    return bio[(start <= cutoff_ts) & (end.isna() | (end >= cutoff_ts))]


def _is_payroll(
    member_ids: pd.Series, member_biography: pd.DataFrame, cutoff_date: dt.date
) -> pd.Series:
    """Whether a member holds a government post at ``cutoff_date``.

    Ministers and whips are both captured: whip appointments appear as ordinary
    ``governmentPosts`` entries rather than a separate category, so "holds a
    government post" already means "minister or whip".

    Parliamentary Private Secretaries are **not** captured. Searching every post
    name across all four Biography categories for all 405 members returns nothing;
    the API does not record PPS appointments. The real payroll vote customarily
    includes several dozen PPSs, so this is a documented undercount.

    A member whose Biography is empty is ``False``, not null. The fetch succeeded
    and found nothing, which is a real negative.
    """
    gov_posts = member_biography[member_biography["category"] == "governmentPosts"]
    active_ids = set(_active_as_at(gov_posts, cutoff_date)["member_id"])
    return member_ids.isin(active_ids)


def _committee_count(
    member_ids: pd.Series, member_biography: pd.DataFrame, cutoff_date: dt.date
) -> pd.Series:
    """Number of distinct ``committeeMemberships`` rows active as at
    ``cutoff_date`` (see ``_active_as_at``).

    The API does not distinguish a standing select committee (e.g. "Treasury
    Committee") from an ad hoc Public Bill Committee assignment (e.g.
    "Approved Premises (Substance Testing) Bill") within this category --
    both are counted the same way here, since nothing in the response marks
    one as more durable than the other. A member with zero active
    committee memberships (including one with no ``member_biography`` rows
    at all) gets 0, not null: schema-non-nullable, and "not on a committee"
    is exactly as real a fact as "on two committees".
    """
    committees = member_biography[member_biography["category"] == "committeeMemberships"]
    active = _active_as_at(committees, cutoff_date)
    counts = active.groupby("member_id").size()
    return counts.reindex(member_ids, fill_value=0).astype("int64")


# --------------------------------------------------------------------------
# nominations (see data/manual/NOMINATIONS_NOTES.md)
# --------------------------------------------------------------------------

#: A fixed, small set of honorific prefixes, stripped from both sides of the
#: nomination join before comparing names. Not fuzzy matching: the prefix set
#: is closed and exact, and every row that still fails to match after this is
#: reported, never guessed at.
_HONORIFIC_PREFIX = re.compile(
    r"^(Rt Hon |Right Hon |Sir |Dame |Dr |Mr |Mrs |Ms |Miss |Professor |Prof )+"
)

#: Bucket per the three crawl-derived ``days_from_open`` values that actually
#: occur in nominations.csv (0/4/7 -- see NOMINATIONS_NOTES.md). Anything else
#: would indicate the source file no longer matches its own documentation.
_NOMINATION_BUCKET_BY_DAY: dict[int, str] = {0: "day1", 4: "mid", 7: "late"}


def _strip_honorifics(names: pd.Series) -> pd.Series:
    return names.str.replace(_HONORIFIC_PREFIX, "", regex=True).str.strip()


@dataclass
class NominationMatchReport:
    """What the nominations.csv -> roster join actually did, for printing and
    for tests -- never silently swallowed.
    """

    n_nomination_rows: int
    n_roster: int
    matched_by_name: int
    matched_by_constituency_fallback: int
    unmatched_nomination_rows: pd.DataFrame = field(repr=False)
    non_nominators: int = 0

    @property
    def n_matched(self) -> int:
        return self.matched_by_name + self.matched_by_constituency_fallback

    def summary(self) -> str:
        return (
            f"nominations.csv join: {self.n_matched}/{self.n_nomination_rows} rows matched to the "
            f"{self.n_roster}-member roster "
            f"({self.matched_by_name} by honorific-stripped name+constituency, "
            f"{self.matched_by_constituency_fallback} by constituency-only fallback); "
            f"{len(self.unmatched_nomination_rows)} nomination row(s) unmatched; "
            f"{self.non_nominators} roster member(s) have no nomination row (non-nominators, "
            "did_nominate=False per NOMINATIONS_NOTES.md, not null)."
        )


def _match_nominations(
    nominations: pd.DataFrame, roster: pd.DataFrame
) -> tuple[pd.DataFrame, NominationMatchReport]:
    """Join nominations.csv onto the roster by name and constituency.

    Two deterministic stages, neither of which is approximate string
    matching:

    1. Strip the fixed honorific set from both sides and match on (stripped
       name, constituency) exactly. Covers the common case where one side
       carries "Dr"/"Sir"/"Dame"/etc. and the other doesn't.
    2. For rows still unmatched, fall back to an exact match on constituency
       alone -- constituency is unique per sitting member (asserted below),
       so this is still an exact-key join, just on the other of the two
       supplied keys. This recovers the two real cases where the *name*
       itself differs: a spelling variant ("Rachel" vs "Rachael" Maskell) and
       a surname change ("Louise Jones" vs "Louise Sandher-Jones"), both
       verified by hand against the roster before relying on this fallback,
       not applied blindly to arbitrary rows.

    Anything left unmatched after both stages is returned in the report
    rather than dropped.
    """
    assert not roster["membership_from"].duplicated().any(), (
        "roster has duplicate constituencies -- the constituency-only fallback below "
        "assumes constituency uniquely identifies a member"
    )

    nom = nominations.copy()
    nom["_name_key"] = _strip_honorifics(nom["mp_name"])
    ros = roster[["member_id", "name_display_as", "membership_from"]].rename(
        columns={"name_display_as": "name", "membership_from": "constituency"}
    ).copy()
    ros["_name_key"] = _strip_honorifics(ros["name"])

    stage1 = nom.merge(ros, on=["_name_key", "constituency"], how="left")
    matched_by_name = int(stage1["member_id"].notna().sum())

    remainder = stage1[stage1["member_id"].isna()].drop(columns=["member_id", "name"])
    stage2 = remainder.merge(ros[["member_id", "constituency"]], on="constituency", how="left")
    matched_by_fallback = int(stage2["member_id"].notna().sum())

    matched = pd.concat(
        [stage1[stage1["member_id"].notna()], stage2[stage2["member_id"].notna()]],
        ignore_index=True,
    )
    assert not matched["member_id"].duplicated().any(), (
        "a roster member matched more than one nominations.csv row -- the join key is "
        "no longer unique, investigate before trusting did_nominate"
    )

    unmatched = stage2[stage2["member_id"].isna()][list(nominations.columns)]

    report = NominationMatchReport(
        n_nomination_rows=len(nominations),
        n_roster=len(roster),
        matched_by_name=matched_by_name,
        matched_by_constituency_fallback=matched_by_fallback,
        unmatched_nomination_rows=unmatched.reset_index(drop=True),
        non_nominators=int((~roster["member_id"].isin(matched["member_id"])).sum()),
    )
    return matched, report


def _nomination_columns(
    roster: pd.DataFrame, nominations: pd.DataFrame
) -> tuple[pd.DataFrame, NominationMatchReport]:
    """did_nominate / nomination_bucket / nomination_day for every roster
    member.

    Per NOMINATIONS_NOTES.md, nominations.csv's tracker is a complete list of
    everyone who nominated -- a roster member absent from it is a confirmed
    non-nominator (did_nominate=False, bucket="none"), not an unknown. Null is
    reserved for a nomination row this join genuinely cannot place (see
    ``NominationMatchReport.unmatched_nomination_rows``); with 100% match
    coverage in the current data, that path is not exercised here but is
    exercised in tests.
    """
    bad_days = set(nominations["days_from_open"].unique()) - set(_NOMINATION_BUCKET_BY_DAY)
    assert not bad_days, (
        f"nominations.csv has days_from_open value(s) {bad_days} outside the documented "
        f"censored set {sorted(_NOMINATION_BUCKET_BY_DAY)} -- NOMINATIONS_NOTES.md may be stale"
    )

    matched, report = _match_nominations(nominations, roster)
    matched = matched.assign(
        nomination_bucket=matched["days_from_open"].map(_NOMINATION_BUCKET_BY_DAY),
        nomination_day=matched["days_from_open"].astype("float64"),
    )

    per_member = matched.set_index("member_id")[["nomination_bucket", "nomination_day"]]
    out = pd.DataFrame(index=pd.Index(roster["member_id"], name="member_id")).join(per_member)

    nominated = out.index.isin(matched["member_id"])
    out["did_nominate"] = pd.array(nominated, dtype="boolean")
    out.loc[~nominated, "nomination_bucket"] = "none"
    # nomination_day is already NaN for non-nominators via the left join.

    return out.reset_index(), report


# --------------------------------------------------------------------------
# speech count
# --------------------------------------------------------------------------


def _speech_counts(member_ids: pd.Series, member_contributions: pd.DataFrame) -> pd.Series:
    s = member_contributions.set_index("member_id")["spoken_result_count"]
    return s.reindex(member_ids).astype("Int64")


# --------------------------------------------------------------------------
# constituency demographics (deprivation_score / median_age / degree_share)
# --------------------------------------------------------------------------

#: A community-built recut of Census 2021 (England & Wales, via the ONS's own
#: Nomis API / "create a custom dataset" tool) and National Records of
#: Scotland's Census 2022 (via Scotland's Census custom-dataset tool) onto the
#: 2024 Westminster Parliamentary Constituency boundaries -- exactly the "ONS
#: publishes Census 2021 data recut to 2024 constituency boundaries" source
#: this column was built for, just not served directly by ons.gov.uk: ONS's
#: own constituency-boundary release
#: (ons.gov.uk/releases/westminsterparliamentaryconstituenciesdataenglandandwalescensus2021)
#: was cancelled outright (confirmed by fetching it -- the page states the
#: release "will be released with NHS England and Wales health areas"
#: instead, with no constituency-boundary data of its own), and
#: commonslibrary.parliament.uk -- the other natural source -- returns HTTP
#: 403 to a scripted fetch (confirmed).
#: This GitHub-hosted CSV is the source that is actually fetchable, built
#: with an open, inspectable methodology (R build scripts at
#: https://github.com/ralphascott/UKGE24_wpc_census_summaries) by the same
#: kind of academic exercise that has produced the standard British Election
#: Study constituency-results file for every past election -- not an
#: unsourced scrape. Pinned to a specific commit, not `main`, so the cached
#: file and this URL stay reproducible even if the repo is updated later.
#:
#: Great Britain only -- 632 of 650 constituencies; Northern Ireland's 18
#: seats aren't covered (NRS Northern Ireland runs a separate census and
#: wasn't merged in). This does not cost any PLP coverage: Labour does not
#: contest Northern Ireland seats, so every current PLP member's constituency
#: is in Great Britain by construction.
CONSTITUENCY_CENSUS_COMMIT = "d37264281d7ffa9eff6a86a01008c15e331022cc"
CONSTITUENCY_CENSUS_URL = (
    "https://raw.githubusercontent.com/ralphascott/UKGE24_wpc_census_summaries/"
    f"{CONSTITUENCY_CENSUS_COMMIT}/2024-UK-General-Election-Census-Constituency-Summaries-File-v1.1.csv"
)

#: NRS Census 2022 Table UV501 ("Highest level of qualification"), the source
#: that fills ``degree_share`` for Scottish constituencies -- see the module
#: docstring's "known, documented gaps" section for why the sibling gap,
#: ``deprivation_score``, could NOT be filled the same way (checked and ruled
#: out, not merely unfetched).
#:
#: **Different source from every other row's ``degree_share`` -- this is the
#: provenance flag the project brief asked to make visible.** This module
#: cannot add a new column to ``schemas.ATTRIBUTES`` (out of scope: attributes.py
#: does not own schemas.py), so provenance is surfaced the way this module
#: already surfaces every other join concern that can't live in the output
#: frame itself: printed in ``ConstituencyMatchReport.summary()`` on every
#: build (see its ``n_degree_share_filled_from_scotland`` field), not silently
#: folded into the number.
#:
#: Fetchable and stable, unlike NRS's own interactive Flexible Table Builder
#: (scotlandscensus.gov.uk/census-results/flexible-table-builder/, a SuperWEB2
#: deployment): that tool gates every query behind a click-through "Terms &
#: Conditions to access ... as a guest user" page, which this module does not
#: automate -- accepting a ToS programmatically on someone's behalf is not
#: something a scripted, unattended fetch should do, and even accepted it
#: does not obviously produce a stable, cache-once URL of the kind this
#: module's whole caching contract depends on. This CSV instead comes from
#: the UK Data Service's open CKAN mirror of NRS's Census 2022 standard
#: outputs (statistics.ukdataservice.ac.uk/api/3/action/package_search,
#: organization ``national-records-of-scotland``) as a plain, unauthenticated
#: static file, already cross-tabulated to 2024 Westminster Parliamentary
#: Constituency boundaries by NRS itself -- no geography-conversion
#: approximation needed. Confirmed by fetching it: all 57 Scottish
#: constituencies present, all 37 PLP-held ones among them, 100% match by
#: normalized name.
#:
#: **Definition, and why only one of the two "higher" Scottish categories is
#: used.** NRS's classification has seven mutually exclusive bands; England &
#: Wales' Census 2021 "Level 4 qualifications or above" (``c21QualLevel4`` in
#: ``CONSTITUENCY_CENSUS_URL``, per ONS's own published definition) bundles
#: degree-and-above together with sub-degree Higher Education (HNC/HND, NVQ
#: 4-5). NRS's Scotland-only equivalent, by contrast, keeps those as two
#: *separate* bands: "Further Education and sub-degree Higher Education
#: qualifications incl. HNC/HNDs" and "Degree level qualifications or above".
#: Naively summing both to mirror ONS's bundling was tried and rejected: it
#: produces a Scottish mean of ~45% against England & Wales' ~34% -- a
#: ~11-point gap with no plausible substantive basis, and exactly the
#: "systematic Scotland-higher cluster" the project brief says is evidence of
#: a definitional mismatch, not a discovery. Using ONLY "Degree level
#: qualifications or above" instead gives a Scottish mean of ~32% against
#: England & Wales' ~34%, with matching spread (Scotland ~19-56%, England &
#: Wales ~18-65%) -- consistent with Scotland's "sub-degree HE" band actually
#: sweeping in vaguer "other post-school, pre-Higher-Education" qualifications
#: (per NRS's own category footnote) that correspond better to England &
#: Wales' Level 3 / Other bands than to Level 4+. This is a documented
#: judgement call, checked against the numbers it produces, not assumed.
SCOTLAND_QUALIFICATIONS_URL = (
    "https://ukds-ckan.s3.eu-west-1.amazonaws.com/2022/NRS/UV501/"
    "Census_2022_UV501_Highest_level_of_qualification_United_Kingdom_Parliamentary_Constituency_2024.csv"
)

#: The two category labels (verbatim, from the source's own
#: "Highest Level of Qualification" column) this module reads out of the
#: seven NRS publishes -- see ``SCOTLAND_QUALIFICATIONS_URL``'s docstring for
#: why "Further Education and sub-degree Higher Education qualifications
#: incl. HNC/HNDs" is deliberately excluded from the numerator.
_SCOTLAND_QUAL_TOTAL_CATEGORY = "All people aged 16 and over"
_SCOTLAND_QUAL_DEGREE_CATEGORY = "Degree level qualifications or above"


def _parse_scotland_qualifications(raw_bytes: bytes) -> pd.DataFrame:
    """Parse NRS's SuperWEB2 CSV export into (constituency, degree_share).

    Not a clean single-header table like the other CSVs this module fetches:
    it opens with 9 lines of title/filter metadata, then the data rows, then
    a blank line, an "INFO","Data has been perturbed" statistical-disclosure-
    control notice, a free-text category-definitions paragraph, and a
    copyright line. Handled by filtering to the two known category labels
    (``_SCOTLAND_QUAL_TOTAL_CATEGORY`` / ``_SCOTLAND_QUAL_DEGREE_CATEGORY``)
    rather than trusting row position past the header, so the footer text
    (which parses as extra, malformed "rows") is dropped regardless of its
    exact shape.

    Perturbation note: NRS applies small random Statistical Disclosure
    Control adjustments to protect against identifying individuals in
    small cells -- standard census-output practice, and immaterial at
    constituency-level aggregates (sub-category counts were checked to sum
    back to each constituency's stated total to within a handful of people).
    """
    raw = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig", skiprows=9)
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw[
        raw["Highest Level of Qualification"].isin(
            (_SCOTLAND_QUAL_TOTAL_CATEGORY, _SCOTLAND_QUAL_DEGREE_CATEGORY)
        )
    ]
    raw = raw.dropna(subset=["United Kingdom Parliamentary Constituency 2024"])

    pivoted = raw.pivot(
        index="United Kingdom Parliamentary Constituency 2024",
        columns="Highest Level of Qualification",
        values="Count",
    )
    assert not pivoted.index.duplicated().any(), (
        "Scotland qualifications source has duplicate constituency rows -- the pivot "
        "above assumes normalized constituency name uniquely identifies a row"
    )

    degree_share = (
        pivoted[_SCOTLAND_QUAL_DEGREE_CATEGORY] / pivoted[_SCOTLAND_QUAL_TOTAL_CATEGORY] * 100.0
    )
    return pd.DataFrame(
        {
            "_name_key": _normalize_constituency_name(pd.Series(degree_share.index)),
            "degree_share_scotland": degree_share.to_numpy(),
        }
    )


def _fetch_scotland_qualifications(settings: Settings) -> pd.DataFrame:
    cache_path = settings.data_raw / "constituency_census" / "scotland_census2022_uv501_qualifications.csv"
    _ensure_cached(SCOTLAND_QUALIFICATIONS_URL, cache_path)
    return _parse_scotland_qualifications(cache_path.read_bytes())


#: (Census column, band lower bound, band width in years). These are the UK
#: 2021 Census's own standard age bands (irregular widths by design -- e.g.
#: the "10 to 15" band is 6 years wide, "16 to 19" is 4 -- not something
#: chosen here), each stored in the source as a percentage of the
#: constituency's population. The final band is open-ended and cannot be
#: linearly interpolated into; see ``_median_age_from_bands``.
_AGE_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("c21Age0to4", 0, 5), ("c21Age5to9", 5, 5), ("c21Age10to15", 10, 6),
    ("c21Age16to19", 16, 4), ("c21Age20to24", 20, 5), ("c21Age25to29", 25, 5),
    ("c21Age30to34", 30, 5), ("c21Age35to39", 35, 5), ("c21Age40to44", 40, 5),
    ("c21Age45to49", 45, 5), ("c21Age50to54", 50, 5), ("c21Age55to59", 55, 5),
    ("c21Age60to64", 60, 5), ("c21Age65to69", 65, 5), ("c21Age70to74", 70, 5),
    ("c21Age75to79", 75, 5), ("c21Age80to84", 80, 5), ("c21Age85plus", 85, None),
)


def _median_age_from_bands(row: pd.Series) -> float:
    """Median age by linear interpolation over Census age-band shares.

    The source publishes banded percentages, never an exact median, so this is the
    standard grouped-median estimator: accumulate band shares in order until the
    running total crosses 50%, then interpolate within that band.

    Checked against known demographics rather than assumed correct: the lowest
    estimates are university seats (Leeds Central ~24.5) and the highest are
    retirement coasts (North Norfolk ~55.5), which a wrong band boundary would not
    reproduce by accident.

    The open-ended 85+ band returns its lower bound, unreachable for any real
    constituency.
    """
    total = sum(row[col] for col, _, _ in _AGE_BANDS)
    half = total / 2.0
    cum = 0.0
    for col, lower, width in _AGE_BANDS:
        pct = row[col]
        if cum + pct >= half:
            if not width:
                return float(lower)
            frac = (half - cum) / pct if pct > 0 else 0.0
            return lower + frac * width
        cum += pct
    return float("nan")


def _normalize_constituency_name(names: pd.Series) -> pd.Series:
    """Fold cosmetic spelling differences before joining constituency names
    across the two sources -- a fixed, deterministic set of transforms, not
    approximate/fuzzy string matching: Unicode NFKD-decompose and drop
    combining marks (strips Welsh diacritics, e.g. "Glyndŵr" -> "Glyndwr" --
    the one and only diacritic anywhere in the 405-member roster, and the
    census source spells everything without diacritics throughout), fold "&"
    to "and", lowercase, and strip whitespace. Applied identically to both
    sides of the join so it can't introduce an asymmetric match.
    """
    def fold(s: object) -> object:
        if pd.isna(s):
            return s
        decomposed = unicodedata.normalize("NFKD", str(s))
        no_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
        return no_marks.replace("&", "and").strip().lower()

    return names.map(fold)


@dataclass
class ConstituencyMatchReport:
    """What the constituency-census join actually did -- printed and tested,

    never silently swallowed (see ``build_attributes``'s "silent bad join"
    concern in the module docstring). Deliberately separates two different
    failure modes that a single null count would conflate: a constituency
    *name* that never found a matching row in the census source at all
    (``unmatched_constituencies`` -- this would be the actual join bug), vs.
    a name that matched fine but whose census row has a genuinely null field
    (Scotland's missing deprivation table, and formerly its qualification
    table too -- a real, documented source gap, not a join failure).

    Also where the Scotland provenance flag for ``degree_share`` lives (see
    ``SCOTLAND_QUALIFICATIONS_URL``'s docstring): this module cannot add a
    provenance column to ``schemas.ATTRIBUTES`` itself, so
    ``n_degree_share_filled_from_scotland`` and this report's ``summary()``
    are how "these rows come from a different source" stays visible rather
    than buried.
    """

    n_roster_constituencies: int
    n_matched: int
    unmatched_constituencies: list[str] = field(default_factory=list)
    n_null_degree_or_deprivation_among_matched: int = 0
    n_degree_share_filled_from_scotland: int = 0
    unmatched_scotland_qualifications: list[str] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_roster_constituencies if self.n_roster_constituencies else 0.0

    def summary(self) -> str:
        return (
            f"constituency census join: {self.n_matched}/{self.n_roster_constituencies} "
            f"distinct roster constituencies matched by name ({self.match_rate:.1%})"
            + (f"; UNMATCHED: {self.unmatched_constituencies}" if self.unmatched_constituencies else "")
            + f"; of matched constituencies, {self.n_null_degree_or_deprivation_among_matched} member(s) "
            "had null degree_share/deprivation_score before the Scotland fill below "
            "(CONSTITUENCY_CENSUS_URL's tables are missing for Scotland, not a join failure -- "
            "see module docstring); "
            f"{self.n_degree_share_filled_from_scotland} of those had degree_share filled from "
            "SCOTLAND_QUALIFICATIONS_URL -- a DIFFERENT source than every English/Welsh row, "
            "on a checked-comparable definition (see that constant's docstring); "
            "deprivation_score remains null for all of them -- no comparable, fetchable Scottish "
            "source was found for it (checked and ruled out; see module docstring)"
            + (
                f"; UNMATCHED against the Scotland qualifications source: "
                f"{self.unmatched_scotland_qualifications}"
                if self.unmatched_scotland_qualifications
                else ""
            )
        )


def _fetch_constituency_census(settings: Settings) -> pd.DataFrame:
    cache_path = settings.data_raw / "constituency_census" / "census21_wpc_gb.csv"
    return _cached_csv_fetch(CONSTITUENCY_CENSUS_URL, cache_path)


def _constituency_demographics(
    roster: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, ConstituencyMatchReport]:
    """deprivation_score, median_age and degree_share per roster member.

    Joined on constituency name via ``_normalize_constituency_name``. Returns the
    frame plus a :class:`ConstituencyMatchReport`, so a mismatch is always
    accounted for rather than silently nulled.

    * ``degree_share`` is the Census 2021 share with Level 4 qualifications or
      above, already 0-100 in the source.
    * ``deprivation_score`` is ``100 - c21DeprivedNone``: households deprived in at
      least one of the Census's four dimensions. This is the Census measure, not
      MHCLG's Index of Multiple Deprivation, which has no official
      constituency-level aggregation.
    * ``median_age`` comes from ``_median_age_from_bands``.

    ``degree_share`` is back-filled for Scottish seats from a second NRS source on
    a comparable definition. ``deprivation_score`` is not: no comparable Scottish
    source is fetchable, and substituting SIMD would invent a Scotland effect.
    """
    raw_census = _fetch_constituency_census(settings)
    # Built as a fresh, small frame rather than assigned onto the 242-column
    # source frame column-by-column -- purely to avoid pandas' fragmentation
    # warning from repeated inserts on a wide frame; nothing semantic depends
    # on this shape.
    census = pd.DataFrame({
        "_name_key": _normalize_constituency_name(raw_census["ConstituencyName"]),
        "deprivation_score": 100.0 - raw_census["c21DeprivedNone"],
        "degree_share": raw_census["c21QualLevel4"],
        "median_age": raw_census.apply(_median_age_from_bands, axis=1),
    })
    assert not census["_name_key"].duplicated().any(), (
        "constituency census source has duplicate normalized names -- the join below "
        "assumes the normalized name uniquely identifies a constituency"
    )

    roster_names = roster[["member_id", "membership_from"]].rename(
        columns={"membership_from": "constituency"}
    )
    roster_names = roster_names.assign(
        _name_key=_normalize_constituency_name(roster_names["constituency"])
    )

    joined = roster_names.merge(
        census[["_name_key", "deprivation_score", "median_age", "degree_share"]],
        on="_name_key",
        how="left",
    )
    assert not joined["member_id"].duplicated().any(), (
        "constituency join produced duplicate member rows -- a normalized name matched "
        "more than one census row, investigate before trusting this join"
    )

    matched_keys = set(census["_name_key"])
    is_matched = roster_names["_name_key"].isin(matched_keys)
    unmatched_constituencies = sorted(
        roster_names.loc[~is_matched, "constituency"].unique().tolist()
    )
    n_null_among_matched = int(
        (joined.loc[is_matched.to_numpy(), ["degree_share", "deprivation_score"]].isna().any(axis=1)).sum()
    )

    # Back-fill degree_share only, for whichever roster members are still
    # null after the main join (Scotland -- see module docstring). Never
    # touches deprivation_score (no comparable Scottish source exists for it)
    # and never overwrites a real English/Welsh value: `needs_fill` is
    # computed from `joined` as it stands right now, before this merge.
    needs_fill = joined["degree_share"].isna()
    scotland_qual = _fetch_scotland_qualifications(settings)
    joined = joined.merge(scotland_qual, on="_name_key", how="left", validate="many_to_one")
    filled_now = needs_fill & joined["degree_share_scotland"].notna()
    joined.loc[filled_now, "degree_share"] = joined.loc[filled_now, "degree_share_scotland"]
    unmatched_scotland_qualifications = sorted(
        roster_names.loc[(needs_fill & ~filled_now).to_numpy(), "constituency"].unique().tolist()
    )
    joined = joined.drop(columns=["degree_share_scotland"])

    report = ConstituencyMatchReport(
        n_roster_constituencies=roster_names["constituency"].nunique(),
        n_matched=roster_names.loc[is_matched, "constituency"].nunique(),
        unmatched_constituencies=unmatched_constituencies,
        n_null_degree_or_deprivation_among_matched=n_null_among_matched,
        n_degree_share_filled_from_scotland=int(filled_now.sum()),
        unmatched_scotland_qualifications=unmatched_scotland_qualifications,
    )
    return joined[["member_id", "deprivation_score", "median_age", "degree_share"]], report


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def build_attributes(settings: Settings | None = None, *, verbose: bool = True) -> pd.DataFrame:
    """Build the full ``schemas.ATTRIBUTES`` table from cached real data.

    Validates against ``schemas.ATTRIBUTES`` before returning: code
    that drifted from the contract should break here, not at whatever reads
    ``data/processed/attributes.parquet`` next.
    """
    settings = settings or get_settings()

    members = collect.read_interim("members", settings=settings)
    divisions = collect.read_interim("divisions", settings=settings)
    division_votes = collect.read_interim("division_votes", settings=settings)
    member_contributions = collect.read_interim("member_contributions", settings=settings)
    member_biography = collect.read_interim("member_biography", settings=settings)
    nominations = pd.read_csv(settings.data_manual / "nominations.csv")

    roster = _select_population(members, settings)

    # Live-API-derived expectation, recomputed independently here rather than
    # trusting _select_population's own arithmetic -- and never a literal
    # count, since the House's membership drifts between runs.
    expected = members[members["party_name"].isin(settings.plp_parties)]
    if settings.exclude_speaker:
        expected = expected[expected["party_name"] != "Speaker"]
    assert len(roster) == len(expected), (
        f"roster size {len(roster)} != members-cache-derived expectation {len(expected)}"
    )

    out = pd.DataFrame(
        {
            "member_id": roster["member_id"].astype("int64"),
            "name": roster["name_display_as"],
            "constituency": roster["membership_from"],
            "party_name": roster["party_name"],
        }
    )

    election_results, ge_results = _combine_election_results(roster, settings)
    out = out.merge(election_results, on="member_id", how="left", validate="one_to_one")
    missing_electoral = out.loc[out["majority_pct"].isna(), "member_id"].tolist()
    assert not missing_electoral, (
        f"no election result for member_id(s) {missing_electoral} -- add them to "
        "BYELECTION_RESULT_URLS with their electionresults.parliament.uk "
        "candidate-results.csv URL (find it via https://electionresults.parliament.uk"
        "/members/<member_id>)"
    )

    first_elected, is_2024_intake = _first_elected_and_intake(roster, ge_results)
    out["first_elected"] = first_elected.to_numpy()
    out["is_2024_intake"] = is_2024_intake.to_numpy()

    out["is_payroll"] = _is_payroll(
        roster["member_id"], member_biography, settings.cutoff_date
    ).to_numpy()
    out["committee_count"] = _committee_count(
        roster["member_id"], member_biography, settings.cutoff_date
    ).to_numpy()
    # `role` (a single free-text "current post" label) is a separate,
    # out-of-scope ask from `is_payroll`/`committee_count`: a member can hold
    # several simultaneous posts (e.g. a Cabinet post plus a party post), and
    # nothing in the API ranks them by seniority, so picking "the" role would
    # be an invented heuristic rather than a read off the data. Left null,
    # documented, not guessed at.
    out["role"] = None

    out["rebellion_rate"] = _rebellion_rate(
        roster["member_id"], divisions, division_votes, settings.cutoff_date
    ).to_numpy()
    # `.array`, not `.to_numpy()`: these are nullable Int64 series, and
    # `.to_numpy()` silently upcasts a nullable-int-with-NA to float64, which
    # then fails schema validation on dtype rather than on anything semantic.
    out["rebellions_welfare"] = _title_matched_rebellion_counts(
        roster["member_id"], divisions, division_votes, "welfare"
    ).array
    out["rebellions_wfa"] = _title_matched_rebellion_counts(
        roster["member_id"], divisions, division_votes, "winter fuel"
    ).array

    nomination_cols, nomination_report = _nomination_columns(roster, nominations)
    out = out.merge(nomination_cols, on="member_id", how="left", validate="one_to_one")

    # deprivation_score / median_age / degree_share: HELDOUT_VARS, so a gap
    # here only weakens frame-error measurement, never leaks into
    # construction -- but see _constituency_demographics' docstring and
    # ConstituencyMatchReport for exactly what is and isn't populated
    # (Scotland's deprivation_score, specifically -- degree_share is filled
    # for Scotland from a second source, see SCOTLAND_QUALIFICATIONS_URL).
    demographics, constituency_report = _constituency_demographics(roster, settings)
    out = out.merge(demographics, on="member_id", how="left", validate="one_to_one")

    out["speech_count"] = _speech_counts(roster["member_id"], member_contributions).array

    out = out[schemas.ATTRIBUTES.names]
    schemas.validate(out, schemas.ATTRIBUTES)

    if verbose:
        print(f"roster: {len(out)} members ({expected['party_name'].value_counts().to_dict()})")
        print(nomination_report.summary())
        n_null_rebellion = int(out["rebellion_rate"].isna().sum())
        print(
            f"rebellion_rate: null for {n_null_rebellion}/{len(out)} members "
            "(see _rebellion_rate docstring for the whip-proxy definition and "
            "the one documented exception)"
        )
        n_null_welfare = int(out["rebellions_welfare"].isna().sum())
        n_null_wfa = int(out["rebellions_wfa"].isna().sum())
        print(
            f"rebellions_welfare: null for {n_null_welfare}/{len(out)}; "
            f"rebellions_wfa: null for {n_null_wfa}/{len(out)} "
            "(see _title_matched_rebellion_counts docstring for which divisions matched)"
        )
        n_payroll = int(out["is_payroll"].sum())
        print(
            f"is_payroll: {n_payroll}/{len(out)} members "
            "(see _is_payroll docstring for exactly what counts)"
        )
        print(constituency_report.summary())

    return out


if __name__ == "__main__":
    settings = get_settings()
    attrs = build_attributes(settings)
    settings.data_processed.mkdir(parents=True, exist_ok=True)
    out_path = settings.data_processed / "attributes.parquet"
    attrs.to_parquet(out_path, index=False)
    print(f"wrote {len(attrs)} rows to {out_path}")
