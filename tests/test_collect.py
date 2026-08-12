"""Tests for plp_sim.collect.

No real network calls: every httpx.Client here is built on a MockTransport
that serves canned pages, so these tests exercise pagination, caching, and
resumability logic against a fake API rather than the live one.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import tempfile
from pathlib import Path

import httpx
import pandas as pd
import pytest

from plp_sim import collect
from plp_sim.config import Settings


@pytest.fixture
def settings():
    """An isolated Settings pointing at a throwaway directory.

    Deliberately not pytest's ``tmp_path``: that fixture names each test's
    directory after the test itself (e.g. ``test_cache_file_has_fetch_meta0``),
    numbered sequentially within a session. Multiple agents run pytest against
    this repo concurrently, and cache-key collisions (fetch_members' page-0
    request, for instance, hashes identically regardless of the mock server's
    total) mean two same-named tests in two processes could in principle
    share a directory. ``tempfile.mkdtemp`` gives each test a directory with a
    cryptographically random suffix instead, which is safe under concurrent
    processes by construction, not by convention.
    """
    tmp = Path(tempfile.mkdtemp(prefix="plp_sim_test_collect_"))
    try:
        s = Settings(
            data_raw=tmp / "raw",
            data_interim=tmp / "interim",
            data_manual=tmp / "manual",
            data_processed=tmp / "processed",
            outputs=tmp / "outputs",
            cache_dir=tmp / "cache",
        )
        s.ensure_dirs()
        yield s
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _member_item(member_id: int, party: str = "Labour") -> dict:
    return {
        "value": {
            "id": member_id,
            "nameDisplayAs": f"Member {member_id}",
            "nameListAs": f"{member_id}, Member",
            "nameFullTitle": f"Member {member_id} MP",
            "gender": "F",
            "latestParty": {"id": 15, "name": party, "abbreviation": "Lab"},
            "latestHouseMembership": {
                "membershipFrom": "Somewhere",
                "membershipFromId": 1,
                "membershipStartDate": "2024-07-04T00:00:00",
                "membershipEndDate": None,
                "membershipStatus": {
                    "statusDescription": "Current Member",
                    "statusStartDate": "2024-07-04T00:00:00",
                },
            },
            "thumbnailUrl": None,
        },
        "links": [],
    }


def _members_handler(n_total: int, page_size: int = 20):
    """A MockTransport handler mimicking Members/Search's pagination contract."""

    def handler(request: httpx.Request) -> httpx.Response:
        skip = int(request.url.params.get("skip", "0"))
        take = int(request.url.params.get("take", str(page_size)))
        ids = list(range(skip, min(skip + take, n_total)))
        items = [_member_item(i) for i in ids]
        return httpx.Response(200, json={"items": items, "totalResults": n_total})

    return handler


def _counting_transport(handler):
    """Wrap a handler to also count how many requests actually hit it."""
    calls = {"n": 0}

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return handler(request)

    return httpx.MockTransport(wrapped), calls


# --------------------------------------------------------------------------
# members
# --------------------------------------------------------------------------


def test_fetch_members_paginates_and_matches_total(settings):
    transport, calls = _counting_transport(_members_handler(n_total=45))
    with httpx.Client(transport=transport) as client:
        df = collect.fetch_members(client, settings=settings)

    assert len(df) == 45
    # 20 + 20 + 5, then one more (empty) page to confirm exhaustion.
    assert calls["n"] == 4
    assert set(df["member_id"]) == set(range(45))


def test_fetch_members_asserts_on_count_mismatch(settings):
    def bad_handler(request: httpx.Request) -> httpx.Response:
        # Always claims a bigger total than it actually ever returns.
        skip = int(request.url.params.get("skip", "0"))
        if skip >= 20:
            return httpx.Response(200, json={"items": [], "totalResults": 45})
        items = [_member_item(i) for i in range(skip, skip + 20)]
        return httpx.Response(200, json={"items": items, "totalResults": 45})

    with (
        httpx.Client(transport=httpx.MockTransport(bad_handler)) as client,
        pytest.raises(AssertionError, match="totalResults"),
    ):
        collect.fetch_members(client, settings=settings)


def test_second_run_makes_zero_network_calls(settings):
    transport, calls = _counting_transport(_members_handler(n_total=45))
    with httpx.Client(transport=transport) as client:
        collect.fetch_members(client, settings=settings)
    assert calls["n"] == 4

    # A fresh client whose transport would error on any request: if fetch_members
    # tries the network at all here, the test fails loudly rather than silently
    # re-fetching.
    def explode(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected network call: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(explode)) as client:
        df = collect.fetch_members(client, settings=settings)
    assert len(df) == 45


def test_dry_run_with_no_client_raises_on_cache_miss(settings):
    with pytest.raises(collect.CollectError):
        collect.fetch_members(None, settings=settings)


def test_force_bypasses_cache(settings):
    transport, calls = _counting_transport(_members_handler(n_total=10))
    with httpx.Client(transport=transport) as client:
        collect.fetch_members(client, settings=settings)
    assert calls["n"] == 2  # one page of 10, one empty page to confirm exhaustion

    with httpx.Client(transport=transport) as client:
        collect.fetch_members(client, settings=settings, force=True)
    assert calls["n"] == 4


def test_cache_file_has_fetch_metadata(settings):
    transport, _ = _counting_transport(_members_handler(n_total=5))
    with httpx.Client(transport=transport) as client:
        collect.fetch_members(client, settings=settings)

    files = list((settings.data_raw / "members").glob("*.json"))
    assert len(files) == 2  # one page of 5, one empty page to confirm exhaustion
    for f in files:
        record = json.loads(f.read_text())
        assert set(record) == {"fetched_at", "url", "params", "payload"}
        dt.datetime.fromisoformat(record["fetched_at"])  # doesn't raise


def test_plan_members_unknown_before_first_fetch(settings):
    plan = collect.plan_members(settings)
    assert plan.total_needed is None
    assert plan.missing == 1


def test_plan_members_matches_after_fetch(settings):
    transport, _ = _counting_transport(_members_handler(n_total=45))
    with httpx.Client(transport=transport) as client:
        collect.fetch_members(client, settings=settings)

    plan = collect.plan_members(settings)
    assert plan.total_needed == 3
    assert plan.cached == 3
    assert plan.missing == 0


def test_cached_members_none_until_fully_cached(settings):
    assert collect.cached_members(settings) is None

    transport, _ = _counting_transport(_members_handler(n_total=25))
    with httpx.Client(transport=transport) as client:
        collect.fetch_members(client, settings=settings)

    df = collect.cached_members(settings)
    assert df is not None
    assert len(df) == 25


def test_plp_member_ids_filters_by_configured_parties(settings):
    members = pd.DataFrame(
        {
            "member_id": [1, 2, 3, 4],
            "party_name": ["Labour", "Labour (Co-op)", "Conservative", "Green Party"],
        }
    )
    ids = collect.plp_member_ids(members, settings)
    assert ids == [1, 2]


def test_fetch_members_writes_interim_parquet(settings):
    transport, _ = _counting_transport(_members_handler(n_total=5))
    with httpx.Client(transport=transport) as client:
        collect.fetch_members(client, settings=settings)

    df = collect.read_interim("members", settings=settings)
    assert len(df) == 5
    assert "member_id" in df.columns


# --------------------------------------------------------------------------
# divisions
# --------------------------------------------------------------------------


def _division_item(division_id: int) -> dict:
    return {
        "DivisionId": division_id,
        "Date": "2025-01-01T00:00:00",
        "Number": division_id,
        "IsDeferred": False,
        "Title": f"Division {division_id}",
        "AyeCount": 300,
        "NoCount": 100,
        "EVELType": "",
    }


def _divisions_handler(n_total: int, page_size: int = 25):
    def handler(request: httpx.Request) -> httpx.Response:
        if "searchTotalResults" in str(request.url):
            return httpx.Response(200, json=n_total)
        skip = int(request.url.params.get("queryParameters.skip", "0"))
        take = int(request.url.params.get("queryParameters.take", str(page_size)))
        ids = list(range(skip, min(skip + take, n_total)))
        return httpx.Response(200, json=[_division_item(i) for i in ids])

    return handler


def test_fetch_divisions_paginates_and_matches_total(settings):
    since = dt.date(2024, 7, 9)
    transport, calls = _counting_transport(_divisions_handler(n_total=60))
    with httpx.Client(transport=transport) as client:
        df = collect.fetch_divisions(since, client, settings=settings)

    assert len(df) == 60
    assert calls["n"] == 1 + 3  # 1 total-count call + 3 pages of 25/25/10

    with httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(
        AssertionError("unexpected network call")
    ))) as client:
        df2 = collect.fetch_divisions(since, client, settings=settings)
    assert len(df2) == 60


def test_cached_divisions_round_trips(settings):
    since = dt.date(2024, 7, 9)
    assert collect.cached_divisions(since, settings) is None

    transport, _ = _counting_transport(_divisions_handler(n_total=30))
    with httpx.Client(transport=transport) as client:
        collect.fetch_divisions(since, client, settings=settings)

    df = collect.cached_divisions(since, settings)
    assert df is not None
    assert len(df) == 30
    assert set(df["division_id"]) == set(range(30))


# --------------------------------------------------------------------------
# division votes
# --------------------------------------------------------------------------


def _division_detail_handler():
    def handler(request: httpx.Request) -> httpx.Response:
        division_id = int(str(request.url).rsplit("/", 1)[-1].removesuffix(".json"))
        return httpx.Response(
            200,
            json={
                "Ayes": [{"MemberId": 1, "Name": "A", "Party": "Labour"}],
                "Noes": [
                    {"MemberId": 2, "Name": "B", "Party": "Labour"},
                    {"MemberId": 3, "Name": "C", "Party": "Conservative"},
                ],
                "DivisionId": division_id,
            },
        )

    return handler


def test_fetch_division_votes_flattens_ayes_and_noes(settings):
    transport, calls = _counting_transport(_division_detail_handler())
    with httpx.Client(transport=transport) as client:
        df = collect.fetch_division_votes(101, client, settings=settings)

    assert calls["n"] == 1
    assert len(df) == 3
    assert set(df.columns) == {"division_id", "member_id", "name", "party", "vote"}
    assert sorted(df["vote"]) == ["aye", "no", "no"]


def test_fetch_all_division_votes_is_individually_cached_and_resumable(settings):
    transport, calls = _counting_transport(_division_detail_handler())
    with httpx.Client(transport=transport) as client:
        collect.fetch_all_division_votes([101, 102, 103], client, settings=settings)
    assert calls["n"] == 3

    with httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(
        AssertionError("unexpected network call")
    ))) as client:
        df = collect.fetch_all_division_votes([101, 102, 103], client, settings=settings)
    assert len(df) == 9  # 3 votes x 3 divisions


def test_plan_division_votes_reports_partial_cache(settings):
    transport, _ = _counting_transport(_division_detail_handler())
    with httpx.Client(transport=transport) as client:
        collect.fetch_division_votes(101, client, settings=settings)

    plan = collect.plan_division_votes([101, 102, 103], settings)
    assert plan.cached == 1
    assert plan.missing == 2


# --------------------------------------------------------------------------
# member contributions
# --------------------------------------------------------------------------


def _contributions_handler():
    def handler(request: httpx.Request) -> httpx.Response:
        member_id = int(request.url.params["queryParameters.memberId"])
        return httpx.Response(
            200,
            json={
                "SpokenResultCount": member_id * 10,
                "WrittenResultCount": 0,
                "CorrectionsResultCount": 0,
                "DivisionsResultCount": member_id,
            },
        )

    return handler


def test_fetch_member_contributions_parses_counts(settings):
    transport, calls = _counting_transport(_contributions_handler())
    with httpx.Client(transport=transport) as client:
        result = collect.fetch_member_contributions(7, client, settings=settings)

    assert calls["n"] == 1
    assert result == {
        "member_id": 7,
        "spoken_result_count": 70,
        "written_result_count": 0,
        "corrections_result_count": 0,
        "divisions_result_count": 7,
    }


def test_fetch_all_member_contributions_resumable(settings):
    transport, calls = _counting_transport(_contributions_handler())
    with httpx.Client(transport=transport) as client:
        collect.fetch_all_member_contributions([1, 2, 3], client, settings=settings)
    assert calls["n"] == 3

    with httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(
        AssertionError("unexpected network call")
    ))) as client:
        df = collect.fetch_all_member_contributions([1, 2, 3], client, settings=settings)
    assert len(df) == 3

    interim = collect.read_interim("member_contributions", settings=settings)
    assert len(interim) == 3


# --------------------------------------------------------------------------
# member biography (posts + committee memberships)
# --------------------------------------------------------------------------


def _biography_handler():
    """Mimics the real, verified /api/Members/{id}/Biography response shape:

    a top-level ``value`` envelope containing (among other keys not needed
    here) governmentPosts / oppositionPosts / otherPosts / committeeMemberships
    arrays of {name, startDate, endDate} objects.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "value": {
                    "governmentPosts": [
                        {"name": "Assistant Whip", "startDate": "2025-09-07T00:00:00", "endDate": None}
                    ],
                    "oppositionPosts": [],
                    "otherPosts": [],
                    "committeeMemberships": [
                        {
                            "name": "Treasury Committee",
                            "startDate": "2024-07-05T00:00:00",
                            "endDate": None,
                        }
                    ],
                },
                "links": [],
            },
        )

    return handler


def test_fetch_member_biography_flattens_all_four_categories(settings):
    transport, calls = _counting_transport(_biography_handler())
    with httpx.Client(transport=transport) as client:
        df = collect.fetch_member_biography(101, client, settings=settings)

    assert calls["n"] == 1
    assert set(df.columns) == {"member_id", "category", "name", "start_date", "end_date"}
    assert len(df) == 2  # one government post + one committee membership
    assert set(df["category"]) == {"governmentPosts", "committeeMemberships"}
    assert (df["member_id"] == 101).all()


def test_fetch_member_biography_empty_for_member_with_no_posts(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"value": {}, "links": []})

    transport, _ = _counting_transport(handler)
    with httpx.Client(transport=transport) as client:
        df = collect.fetch_member_biography(101, client, settings=settings)
    assert len(df) == 0
    assert set(df.columns) == {"member_id", "category", "name", "start_date", "end_date"}


def test_fetch_all_member_biographies_is_individually_cached_and_resumable(settings):
    transport, calls = _counting_transport(_biography_handler())
    with httpx.Client(transport=transport) as client:
        collect.fetch_all_member_biographies([101, 102, 103], client, settings=settings)
    assert calls["n"] == 3

    with httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(
        AssertionError("unexpected network call")
    ))) as client:
        df = collect.fetch_all_member_biographies([101, 102, 103], client, settings=settings)
    assert len(df) == 6  # 2 rows x 3 members

    interim = collect.read_interim("member_biography", settings=settings)
    assert len(interim) == 6


def test_plan_member_biographies_reports_partial_cache(settings):
    transport, _ = _counting_transport(_biography_handler())
    with httpx.Client(transport=transport) as client:
        collect.fetch_member_biography(101, client, settings=settings)

    plan = collect.plan_member_biographies([101, 102, 103], settings)
    assert plan.cached == 1
    assert plan.missing == 2


# --------------------------------------------------------------------------
# Early Day Motions (list + per-motion signatories)
# --------------------------------------------------------------------------


def _edm_item(motion_id: int, sponsors_count: int = 0) -> dict:
    return {
        "Id": motion_id,
        "Status": 0,
        "StatusDate": "2025-01-01T00:00:00",
        "MemberId": 1000 + motion_id,
        "PrimarySponsor": {"Name": f"Sponsor {motion_id}", "Party": "Labour"},
        "Title": f"Motion {motion_id}",
        "MotionText": f"That this House notes motion {motion_id}.",
        "AmendmentToMotionId": None,
        "UIN": motion_id,
        "AmendmentSuffix": None,
        "DateTabled": "2025-01-01T00:00:00",
        "UINWithAmendmentSuffix": str(motion_id),
        "SponsorsCount": sponsors_count,
    }


def _edms_handler(n_total: int):
    """A MockTransport handler mimicking EarlyDayMotions/list's envelope and

    pagination contract: total lives in ``PagingInfo.Total`` on every page,
    the items live under ``Response``, same "total learned from page 0,
    page until empty" shape as Members/Search. Page size always comes from
    the request's own ``parameters.take`` (the real client always sends
    ``EDM_PAGE_SIZE``), not a separate handler argument, so the test can't
    silently drift from what the code under test actually requests.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        skip = int(request.url.params.get("parameters.skip", "0"))
        take = int(request.url.params["parameters.take"])
        ids = list(range(skip, min(skip + take, n_total)))
        return httpx.Response(
            200,
            json={
                "PagingInfo": {"Skip": skip, "Take": take, "Total": n_total},
                "StatusCode": 200,
                "Success": True,
                "Errors": [],
                "Response": [_edm_item(i) for i in ids],
            },
        )

    return handler


def test_fetch_edms_paginates_and_matches_total(settings):
    since, until = dt.date(2024, 7, 4), dt.date(2026, 3, 31)
    # 250 spans 3 pages of EDM_PAGE_SIZE=100 (100+100+50), then one more
    # (empty) page to confirm exhaustion: same "confirm with an empty page"
    # shape as fetch_members.
    transport, calls = _counting_transport(_edms_handler(n_total=250))
    with httpx.Client(transport=transport) as client:
        df = collect.fetch_edms(since, until, client, settings=settings)

    assert len(df) == 250
    assert calls["n"] == 4
    assert set(df["motion_id"]) == set(range(250))
    assert "sponsors_count" in df.columns

    with httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(
        AssertionError("unexpected network call")
    ))) as client:
        df2 = collect.fetch_edms(since, until, client, settings=settings)
    assert len(df2) == 250


def test_fetch_edms_asserts_on_count_mismatch(settings):
    def bad_handler(request: httpx.Request) -> httpx.Response:
        skip = int(request.url.params.get("parameters.skip", "0"))
        if skip >= 100:
            return httpx.Response(
                200,
                json={"PagingInfo": {"Total": 999}, "Response": []},
            )
        items = [_edm_item(i) for i in range(skip, skip + 100)]
        return httpx.Response(200, json={"PagingInfo": {"Total": 999}, "Response": items})

    with (
        httpx.Client(transport=httpx.MockTransport(bad_handler)) as client,
        pytest.raises(AssertionError, match="PagingInfo.Total"),
    ):
        collect.fetch_edms(dt.date(2024, 7, 4), dt.date(2026, 3, 31), client, settings=settings)


def test_cached_edms_none_until_fully_cached(settings):
    since, until = dt.date(2024, 7, 4), dt.date(2026, 3, 31)
    assert collect.cached_edms(since, until, settings) is None

    transport, _ = _counting_transport(_edms_handler(n_total=30))
    with httpx.Client(transport=transport) as client:
        collect.fetch_edms(since, until, client, settings=settings)

    df = collect.cached_edms(since, until, settings)
    assert df is not None
    assert len(df) == 30


def test_plan_edms_unknown_before_first_fetch(settings):
    plan = collect.plan_edms(dt.date(2024, 7, 4), dt.date(2026, 3, 31), settings)
    assert plan.total_needed is None
    assert plan.missing == 1


def test_plan_edms_matches_after_fetch(settings):
    since, until = dt.date(2024, 7, 4), dt.date(2026, 3, 31)
    transport, _ = _counting_transport(_edms_handler(n_total=250))
    with httpx.Client(transport=transport) as client:
        collect.fetch_edms(since, until, client, settings=settings)

    plan = collect.plan_edms(since, until, settings)
    assert plan.total_needed == 3  # ceil(250/100)
    assert plan.cached == 3
    assert plan.missing == 0


def _edm_detail_handler():
    """Mimics the real ``EarlyDayMotion/{id}`` detail response: a

    ``Response`` envelope wrapping a single motion with a full ``Sponsors``
    array (primary sponsor plus every signatory), each entry keyed directly
    by ``MemberId``: no name-matching required.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        motion_id = int(str(request.url).rsplit("/", 1)[-1])
        return httpx.Response(
            200,
            json={
                "PagingInfo": None,
                "StatusCode": 200,
                "Success": True,
                "Errors": [],
                "Response": {
                    "Id": motion_id,
                    "Sponsors": [
                        {
                            "MemberId": 1,
                            "Member": {"Name": "A", "Party": "Labour"},
                            "SponsoringOrder": 1,
                            "CreatedWhen": "2025-01-01T00:00:00",
                            "IsWithdrawn": False,
                            "WithdrawnDate": None,
                        },
                        {
                            "MemberId": 2,
                            "Member": {"Name": "B", "Party": "Labour"},
                            "SponsoringOrder": None,
                            "CreatedWhen": "2025-01-02T00:00:00",
                            "IsWithdrawn": True,
                            "WithdrawnDate": "2025-02-01T00:00:00",
                        },
                    ],
                    "SponsorsCount": 0,
                },
            },
        )

    return handler


def test_fetch_edm_signatures_flattens_sponsors(settings):
    transport, calls = _counting_transport(_edm_detail_handler())
    with httpx.Client(transport=transport) as client:
        df = collect.fetch_edm_signatures(555, client, settings=settings)

    assert calls["n"] == 1
    assert len(df) == 2
    assert set(df["member_id"]) == {1, 2}
    assert (df["motion_id"] == 555).all()
    assert df.loc[df["member_id"] == 2, "is_withdrawn"].iloc[0]
    assert not df.loc[df["member_id"] == 1, "is_withdrawn"].iloc[0]


def test_fetch_all_edm_signatures_is_individually_cached_and_resumable(settings):
    transport, calls = _counting_transport(_edm_detail_handler())
    with httpx.Client(transport=transport) as client:
        collect.fetch_all_edm_signatures([101, 102, 103], client, settings=settings)
    assert calls["n"] == 3

    with httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(
        AssertionError("unexpected network call")
    ))) as client:
        df = collect.fetch_all_edm_signatures([101, 102, 103], client, settings=settings)
    assert len(df) == 6  # 2 signatories x 3 motions

    interim = collect.read_interim("edm_signatures", settings=settings)
    assert len(interim) == 6


def test_plan_edm_signatures_reports_partial_cache(settings):
    transport, _ = _counting_transport(_edm_detail_handler())
    with httpx.Client(transport=transport) as client:
        collect.fetch_edm_signatures(101, client, settings=settings)

    plan = collect.plan_edm_signatures([101, 102, 103], settings)
    assert plan.cached == 1
    assert plan.missing == 2


def test_fetch_edms_writes_interim_parquet(settings):
    since, until = dt.date(2024, 7, 4), dt.date(2026, 3, 31)
    transport, _ = _counting_transport(_edms_handler(n_total=5))
    with httpx.Client(transport=transport) as client:
        collect.fetch_edms(since, until, client, settings=settings)

    df = collect.read_interim("edms", settings=settings)
    assert len(df) == 5
    assert "motion_id" in df.columns
