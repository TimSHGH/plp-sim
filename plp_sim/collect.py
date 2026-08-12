"""Fetches raw data from the Parliament APIs and caches it.

Pure I/O. Nothing here computes a column that ends up in ``ATTRIBUTES`` or
``HOLDOUT``; that happens downstream against a fixed cache, so it can be re-run
for free.

Sources:

* **Members** ``members-api.parliament.uk`` -- the roster of current Commons
  members, and per-member biography for posts and committees. The roster
  endpoint carries no posts data, hence the second call.
* **Divisions** ``commonsvotes-api.parliament.uk`` -- division overviews, and
  per-division member-level Aye/No lists.
* **Hansard** ``hansard-api.parliament.uk`` -- spoken contribution counts.
  ``take=1`` suffices: the endpoint returns aggregate counts regardless of how
  many rows it returns.
* **EDMs** ``oralquestionsandmotions-api.parliament.uk`` -- Early Day Motions
  and their signatories.

Caching contract. Every response is stored as
``{fetched_at, url, params, payload}`` under ``data_raw/<source>/``, keyed by a
hash of the request, so identical requests resolve to the same file and a
re-run makes no network calls. ``force=True`` bypasses it for one call.

Because pagination is fully determined by ``take`` and a known total, the
``plan_*`` functions can list exactly which cache files a full collection needs
and check them on disk. A real dry run, not an estimate.

One caveat worth knowing: an EDM's ``SponsorsCount`` is unreliable on the
detail endpoint (observed 0 against 124 actual sponsors). Derive signature
counts from ``edm_signatures``, never from that field.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from plp_sim.config import Settings, get_settings

MEMBERS_BASE = "https://members-api.parliament.uk/api/Members/Search"
DIVISIONS_SEARCH_BASE = "https://commonsvotes-api.parliament.uk/data/divisions.json/search"
DIVISIONS_TOTAL_BASE = (
    "https://commonsvotes-api.parliament.uk/data/divisions.json/searchTotalResults"
)
DIVISION_DETAIL_BASE = "https://commonsvotes-api.parliament.uk/data/division"
HANSARD_CONTRIBUTIONS_BASE = "https://hansard-api.parliament.uk/search/contributions/Spoken.json"
MEMBER_BIOGRAPHY_BASE = "https://members-api.parliament.uk/api/Members"
#: Plural "EarlyDayMotions": the singular form 404s. Verified against the live API.
EDM_LIST_BASE = "https://oralquestionsandmotions-api.parliament.uk/EarlyDayMotions/list"
#: Singular "EarlyDayMotion": the per-motion detail route, distinct from the
#: list route above. Also verified live.
EDM_DETAIL_BASE = "https://oralquestionsandmotions-api.parliament.uk/EarlyDayMotion"

#: State opening of the current Parliament. Default lower bound for the
#: divisions pull: behavioural history (rebellion_rate) is built over the
#: whole sitting period, not just since cutoff_date. Exposed as a CLI default
#: in scripts/01_collect.py, not buried as a silent assumption in here.
CURRENT_PARLIAMENT_START = dt.date(2024, 7, 9)

MEMBERS_PAGE_SIZE = 20

#: ~2 requests/second, shared across every source in this process.
_MIN_INTERVAL_S = 0.5


class CollectError(RuntimeError):
    """Raised when a fetch can't proceed as requested (e.g. a dry-run cache miss)."""


class _RateLimiter:
    """Process-wide throttle. Single-threaded collection only: this project

    never issues concurrent requests to these APIs, so a simple last-call
    timestamp is sufficient; no lock is needed.
    """

    def __init__(self, min_interval_s: float) -> None:
        self._min_interval = min_interval_s
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is not None:
            remaining = self._min_interval - (time.monotonic() - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


_rate_limiter = _RateLimiter(_MIN_INTERVAL_S)


def _is_retryable(exc: BaseException) -> bool:
    """429 or 5xx get retried with backoff; 4xx other than 429 is a real error."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    reraise=True,
)
def _get_json(client: httpx.Client, url: str, params: dict[str, Any] | None) -> Any:
    _rate_limiter.wait()
    resp = client.get(url, params=params, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def _cache_key(url: str, params: dict[str, Any] | None) -> str:
    """Deterministic filename stem for a request: same request, same file."""
    raw = url + "?" + json.dumps(params or {}, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    slug = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0][:40] or "root"
    slug = "".join(c if c.isalnum() or c in "-_." else "_" for c in slug)
    return f"{slug}_{digest}"


def _cache_path(settings: Settings, source: str, url: str, params: dict[str, Any] | None) -> Path:
    d = settings.data_raw / source
    return d / f"{_cache_key(url, params)}.json"


def cache_exists(settings: Settings, source: str, url: str, params: dict[str, Any] | None) -> bool:
    """Whether a request is already cached, with no network call."""
    return _cache_path(settings, source, url, params).exists()


def _read_cache(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["payload"]


def _fetch_cached(
    client: httpx.Client | None,
    settings: Settings,
    source: str,
    url: str,
    params: dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> Any:
    """Return the parsed payload for ``url``, from cache when possible.

    Raises :class:`CollectError` if the response is not cached and no client
    was given: that is what makes a dry run (``client=None``) a hard
    guarantee of zero network calls rather than a promise kept by convention.
    """
    path = _cache_path(settings, source, url, params)
    if path.exists() and not force:
        return _read_cache(path)

    if client is None:
        raise CollectError(f"not cached and no client given (dry run?): {url} {params}")

    payload = _get_json(client, url, params)
    record = {
        "fetched_at": dt.datetime.now(dt.UTC).isoformat(),
        "url": url,
        "params": params or {},
        "payload": payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f)
    return payload


# --------------------------------------------------------------------------
# members
# --------------------------------------------------------------------------


def _members_page_params(skip: int) -> dict[str, Any]:
    return {"House": 1, "IsCurrentMember": "true", "take": MEMBERS_PAGE_SIZE, "skip": skip}


def _flatten_member(item: dict[str, Any]) -> dict[str, Any]:
    """One row of raw-but-flattened fields from a Members/Search item.

    Flattening nested JSON into columns is parsing, not derivation: every
    value here is copied straight from the API, nothing is computed from it.
    """
    v = item["value"]
    party = v.get("latestParty") or {}
    membership = v.get("latestHouseMembership") or {}
    status = membership.get("membershipStatus") or {}
    return {
        "member_id": v["id"],
        "name_display_as": v.get("nameDisplayAs"),
        "name_list_as": v.get("nameListAs"),
        "name_full_title": v.get("nameFullTitle"),
        "gender": v.get("gender"),
        "party_id": party.get("id"),
        "party_name": party.get("name"),
        "party_abbreviation": party.get("abbreviation"),
        "membership_from": membership.get("membershipFrom"),
        "membership_from_id": membership.get("membershipFromId"),
        "membership_start_date": membership.get("membershipStartDate"),
        "membership_end_date": membership.get("membershipEndDate"),
        "membership_status_description": status.get("statusDescription"),
        "membership_status_start_date": status.get("statusStartDate"),
        "thumbnail_url": v.get("thumbnailUrl"),
    }


def _paginate_members(
    client: httpx.Client | None, settings: Settings, *, force: bool
) -> tuple[list[dict[str, Any]], int]:
    """Walk every members page, from cache or network. Shared by the live

    fetch and by :func:`cached_members`, which calls this with ``client=None``
    so a partial cache surfaces as a :class:`CollectError` rather than a
    network call.
    """
    rows: list[dict[str, Any]] = []
    skip = 0
    total_results: int | None = None
    while True:
        payload = _fetch_cached(
            client, settings, "members", MEMBERS_BASE, _members_page_params(skip), force=force
        )
        items = payload.get("items", [])
        if total_results is None:
            total_results = payload["totalResults"]
        if not items:
            break
        rows.extend(_flatten_member(item) for item in items)
        skip += MEMBERS_PAGE_SIZE
    assert total_results is not None, "members API returned no pages at all"
    return rows, total_results


def fetch_members(
    client: httpx.Client | None,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """All current Commons members, paginated to exhaustion.

    Asserts the number of rows collected equals the API's own
    ``totalResults``: never a hardcoded count, since the House's membership
    drifts (by-elections, defections, deaths) between any two runs.
    """
    settings = settings or get_settings()
    rows, total_results = _paginate_members(client, settings, force=force)
    assert len(rows) == total_results, (
        f"collected {len(rows)} members but API reports totalResults={total_results}"
    )
    df = pd.DataFrame(rows)
    write_interim(df, "members", settings=settings)
    return df


def cached_members(settings: Settings | None = None) -> pd.DataFrame | None:
    """Reassemble the members table purely from disk, or ``None`` if the

    cache doesn't yet cover every page. Used for dry-run planning and by
    scripts that need to know who's in scope for a further pull (e.g. which
    members are PLP) without touching the network.
    """
    settings = settings or get_settings()
    try:
        rows, total_results = _paginate_members(None, settings, force=False)
    except CollectError:
        return None
    if len(rows) != total_results:
        return None
    return pd.DataFrame(rows)


def plan_members(settings: Settings | None = None) -> FetchPlan:
    """What a full members pull would need, read entirely from disk.

    The first page carries ``totalResults``, which fixes every subsequent
    ``skip`` value deterministically, so once page 0 is cached, the whole
    plan is knowable without another network call.
    """
    settings = settings or get_settings()
    page0_path = _cache_path(settings, "members", MEMBERS_BASE, _members_page_params(0))
    if not page0_path.exists():
        return FetchPlan(source="members", total_needed=None, cached=0, missing=1,
                          note="first page not cached; total page count unknown until fetched")

    total = _read_cache(page0_path)["totalResults"]
    n_pages = (total + MEMBERS_PAGE_SIZE - 1) // MEMBERS_PAGE_SIZE
    cached = 0
    for skip in range(0, n_pages * MEMBERS_PAGE_SIZE, MEMBERS_PAGE_SIZE):
        if _cache_path(settings, "members", MEMBERS_BASE, _members_page_params(skip)).exists():
            cached += 1
    return FetchPlan(source="members", total_needed=n_pages, cached=cached,
                      missing=n_pages - cached)


# --------------------------------------------------------------------------
# divisions (commonsvotes-api)
# --------------------------------------------------------------------------

DIVISIONS_PAGE_SIZE = 25


def _divisions_total_params(since: dt.date) -> dict[str, Any]:
    return {"queryParameters.startDate": since.isoformat()}


def _divisions_page_params(since: dt.date, skip: int) -> dict[str, Any]:
    return {
        "queryParameters.startDate": since.isoformat(),
        "queryParameters.skip": skip,
        "queryParameters.take": DIVISIONS_PAGE_SIZE,
    }


def _flatten_division(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "division_id": item["DivisionId"],
        "date": item.get("Date"),
        "number": item.get("Number"),
        "is_deferred": item.get("IsDeferred"),
        "title": item.get("Title"),
        "aye_count": item.get("AyeCount"),
        "no_count": item.get("NoCount"),
        "evel_type": item.get("EVELType"),
    }


def _paginate_divisions(
    client: httpx.Client | None, settings: Settings, since: dt.date, *, force: bool
) -> tuple[list[dict[str, Any]], int]:
    """Walk every division-search page, from cache or network.

    The search endpoint returns a bare array with no total, so the count
    comes from the sibling ``searchTotalResults`` endpoint and is fetched
    (and cached) first: a separate cached call, so a dry run can tell
    whether it's already known without paging.
    """
    total = _fetch_cached(
        client, settings, "divisions_total", DIVISIONS_TOTAL_BASE,
        _divisions_total_params(since), force=force,
    )
    rows: list[dict[str, Any]] = []
    skip = 0
    while skip < total:
        page = _fetch_cached(
            client, settings, "divisions", DIVISIONS_SEARCH_BASE,
            _divisions_page_params(since, skip), force=force,
        )
        if not page:
            break
        rows.extend(_flatten_division(item) for item in page)
        skip += DIVISIONS_PAGE_SIZE
    return rows, total


def fetch_divisions(
    since: dt.date,
    client: httpx.Client | None,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Division overviews (no member-level votes) from ``since`` to now."""
    settings = settings or get_settings()
    rows, total = _paginate_divisions(client, settings, since, force=force)
    assert len(rows) == total, (
        f"collected {len(rows)} divisions but searchTotalResults reports {total}"
    )
    df = pd.DataFrame(rows)
    write_interim(df, "divisions", settings=settings)
    return df


def cached_divisions(since: dt.date, settings: Settings | None = None) -> pd.DataFrame | None:
    """Reassemble the division overview table purely from disk, or ``None``

    if the cache doesn't yet cover every page. Lets a script choose which
    division ids to pull vote detail for without a network call.
    """
    settings = settings or get_settings()
    try:
        rows, total = _paginate_divisions(None, settings, since, force=False)
    except CollectError:
        return None
    if len(rows) != total:
        return None
    return pd.DataFrame(rows)


def plan_divisions(since: dt.date, settings: Settings | None = None) -> FetchPlan:
    settings = settings or get_settings()
    total_path = _cache_path(
        settings, "divisions_total", DIVISIONS_TOTAL_BASE, _divisions_total_params(since)
    )
    if not total_path.exists():
        return FetchPlan(source="divisions", total_needed=None, cached=0, missing=1,
                          note="total-results count not cached; page count unknown until fetched")

    total = _read_cache(total_path)
    n_pages = (total + DIVISIONS_PAGE_SIZE - 1) // DIVISIONS_PAGE_SIZE if total else 0
    cached = 0
    for skip in range(0, n_pages * DIVISIONS_PAGE_SIZE, DIVISIONS_PAGE_SIZE):
        params = _divisions_page_params(since, skip)
        if _cache_path(settings, "divisions", DIVISIONS_SEARCH_BASE, params).exists():
            cached += 1
    # +1 for the total-results call itself, already known to be cached here.
    return FetchPlan(source="divisions", total_needed=n_pages + 1, cached=cached + 1,
                      missing=n_pages - cached)


# --------------------------------------------------------------------------
# division votes (commonsvotes-api detail: per-member Aye/No)
# --------------------------------------------------------------------------


def _flatten_division_votes(division_id: int, detail: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for vote, key in (("aye", "Ayes"), ("no", "Noes")):
        for m in detail.get(key) or []:
            rows.append({
                "division_id": division_id,
                "member_id": m.get("MemberId"),
                "name": m.get("Name"),
                "party": m.get("Party"),
                "vote": vote,
            })
    return pd.DataFrame(rows, columns=["division_id", "member_id", "name", "party", "vote"])


def fetch_division_votes(
    division_id: int,
    client: httpx.Client | None,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Member-level Aye/No list for one division, long format (one row per vote).

    This is the endpoint the search results don't carry: ``AyeCount`` and
    ``NoCount`` on the overview are aggregates only; the named lists live
    here, keyed by the division's own id (not its ext id).
    """
    settings = settings or get_settings()
    url = f"{DIVISION_DETAIL_BASE}/{division_id}.json"
    detail = _fetch_cached(client, settings, "division_votes", url, None, force=force)
    return _flatten_division_votes(division_id, detail)


def fetch_all_division_votes(
    division_ids: list[int],
    client: httpx.Client | None,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Vote lists for many divisions, concatenated and written to interim."""
    settings = settings or get_settings()
    parts = [
        fetch_division_votes(did, client, settings=settings, force=force)
        for did in division_ids
    ]
    df = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["division_id", "member_id", "name", "party", "vote"])
    )
    write_interim(df, "division_votes", settings=settings)
    return df


def plan_division_votes(division_ids: list[int], settings: Settings | None = None) -> FetchPlan:
    settings = settings or get_settings()
    cached = sum(
        1 for did in division_ids
        if cache_exists(settings, "division_votes", f"{DIVISION_DETAIL_BASE}/{did}.json", None)
    )
    return FetchPlan(
        source="division_votes", total_needed=len(division_ids), cached=cached,
        missing=len(division_ids) - cached,
    )


# --------------------------------------------------------------------------
# member contributions (hansard-api: spoken-contribution counts)
# --------------------------------------------------------------------------


def _contribution_params(member_id: int) -> dict[str, Any]:
    # take=1 keeps the payload tiny; the *ResultCount fields are aggregate
    # counts independent of how many result rows come back.
    return {"queryParameters.memberId": member_id, "queryParameters.take": 1}


def fetch_member_contributions(
    member_id: int,
    client: httpx.Client | None,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Spoken-contribution counts for one member, straight off the API.

    Returns the aggregate count fields (``SpokenResultCount`` and siblings)
    plus the member id. Not a speech_count column yet: that mapping is the
    attribute-construction module's call to make, including how to treat a
    member with zero contributions versus one who hasn't been fetched.
    """
    settings = settings or get_settings()
    payload = _fetch_cached(
        client, settings, "member_contributions", HANSARD_CONTRIBUTIONS_BASE,
        _contribution_params(member_id), force=force,
    )
    return {
        "member_id": member_id,
        "spoken_result_count": payload.get("SpokenResultCount"),
        "written_result_count": payload.get("WrittenResultCount"),
        "corrections_result_count": payload.get("CorrectionsResultCount"),
        "divisions_result_count": payload.get("DivisionsResultCount"),
    }


def fetch_all_member_contributions(
    member_ids: list[int],
    client: httpx.Client | None,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Contribution counts for many members, concatenated and written to interim.

    This is the "long pull": one request per member, ~400+ of them for the
    PLP at 2 req/s, so resumability here matters more than anywhere else in
    this module.
    """
    settings = settings or get_settings()
    rows = [
        fetch_member_contributions(mid, client, settings=settings, force=force)
        for mid in member_ids
    ]
    df = pd.DataFrame(rows)
    write_interim(df, "member_contributions", settings=settings)
    return df


def plan_member_contributions(member_ids: list[int], settings: Settings | None = None) -> FetchPlan:
    settings = settings or get_settings()
    cached = sum(
        1 for mid in member_ids
        if cache_exists(
            settings, "member_contributions", HANSARD_CONTRIBUTIONS_BASE,
            _contribution_params(mid),
        )
    )
    return FetchPlan(
        source="member_contributions", total_needed=len(member_ids), cached=cached,
        missing=len(member_ids) - cached,
    )


# --------------------------------------------------------------------------
# member biography (members-api: posts + committee memberships)
# --------------------------------------------------------------------------

#: The four post/membership arrays the Biography endpoint actually returns
#: (verified against live data: see the module docstring). There is no
#: ``parliamentaryPosts`` key; that was a guess before verification and the
#: real shape is these four.
BIOGRAPHY_CATEGORIES: tuple[str, ...] = (
    "governmentPosts", "oppositionPosts", "otherPosts", "committeeMemberships",
)


def _flatten_member_biography(member_id: int, payload: dict[str, Any]) -> pd.DataFrame:
    """One row per (member, post-or-committee-membership), long format: the same shape as ``_flatten_division_votes``: one row per event rather
    than one wide row per member, since a member can hold any number of
    posts across the four categories. ``category`` keeps the source array
    name verbatim so downstream code can tell a government post from a
    committee membership without re-deriving it.
    """
    v = payload.get("value") or {}
    rows: list[dict[str, Any]] = []
    for category in BIOGRAPHY_CATEGORIES:
        for p in v.get(category) or []:
            rows.append({
                "member_id": member_id,
                "category": category,
                "name": p.get("name"),
                "start_date": p.get("startDate"),
                "end_date": p.get("endDate"),
            })
    return pd.DataFrame(rows, columns=["member_id", "category", "name", "start_date", "end_date"])


def fetch_member_biography(
    member_id: int,
    client: httpx.Client | None,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Post and committee history for one member, long format (one row per
    post/membership). A member with no posts and no committee memberships
    at all yields zero rows, not a row of nulls: same convention as
    ``fetch_division_votes`` returning an empty frame for an uncontested
    division.
    """
    settings = settings or get_settings()
    url = f"{MEMBER_BIOGRAPHY_BASE}/{member_id}/Biography"
    payload = _fetch_cached(client, settings, "member_biography", url, None, force=force)
    return _flatten_member_biography(member_id, payload)


def fetch_all_member_biographies(
    member_ids: list[int],
    client: httpx.Client | None,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Biographies for many members, concatenated and written to interim.

    One request per member: same "long pull" shape as
    ``fetch_all_member_contributions``, and individually cached the same
    way, so it is resumable on the same terms.
    """
    settings = settings or get_settings()
    parts = [
        fetch_member_biography(mid, client, settings=settings, force=force)
        for mid in member_ids
    ]
    df = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["member_id", "category", "name", "start_date", "end_date"])
    )
    write_interim(df, "member_biography", settings=settings)
    return df


def plan_member_biographies(member_ids: list[int], settings: Settings | None = None) -> FetchPlan:
    settings = settings or get_settings()
    cached = sum(
        1 for mid in member_ids
        if cache_exists(
            settings, "member_biography", f"{MEMBER_BIOGRAPHY_BASE}/{mid}/Biography", None
        )
    )
    return FetchPlan(
        source="member_biography", total_needed=len(member_ids), cached=cached,
        missing=len(member_ids) - cached,
    )


def plp_member_ids(members: pd.DataFrame, settings: Settings | None = None) -> list[int]:
    """Which fetched members are in scope for the (expensive) Hansard pull.

    Uses ``settings.plp_parties`` directly: the same population definition
    the rest of the project uses, so this scoping decision can't drift from
    it. This is a fetch-scope decision, not the ATTRIBUTES population
    definition itself: exclude_speaker/include_defectors are applied later,
    by attribute construction, against the full members cache.
    """
    settings = settings or get_settings()
    in_scope = members[members["party_name"].isin(settings.plp_parties)]
    return sorted(int(x) for x in in_scope["member_id"].unique())


# --------------------------------------------------------------------------
# Early Day Motions (oralquestionsandmotions-api): motions + signatories
# --------------------------------------------------------------------------

EDM_PAGE_SIZE = 100  # documented API maximum for `parameters.take`
#: Both fetch functions below share this cache source directory. The
#: cache key's URL-derived slug ("list" vs. the numeric motion id) keeps
#: list pages and detail pages from colliding on disk.
EDM_SOURCE = "edms"


def _edm_list_params(tabled_start: dt.date, tabled_end: dt.date, skip: int) -> dict[str, Any]:
    return {
        "parameters.tabledStartDate": tabled_start.isoformat(),
        "parameters.tabledEndDate": tabled_end.isoformat(),
        "parameters.skip": skip,
        "parameters.take": EDM_PAGE_SIZE,
    }


def _flatten_edm(item: dict[str, Any]) -> dict[str, Any]:
    """One row of raw-but-flattened fields from an EarlyDayMotions/list item.

    ``sponsors_count`` is carried through verbatim for provenance only: it
    is unreliable on the detail endpoint (see module docstring) and real
    signature counts must be derived from ``edm_signatures``, not this
    field.
    """
    sponsor = item.get("PrimarySponsor") or {}
    return {
        "motion_id": item["Id"],
        "member_id": item.get("MemberId"),
        "primary_sponsor_name": sponsor.get("Name"),
        "primary_sponsor_party": sponsor.get("Party"),
        "status": item.get("Status"),
        "status_date": item.get("StatusDate"),
        "title": item.get("Title"),
        "motion_text": item.get("MotionText"),
        "uin": item.get("UIN"),
        "uin_with_amendment_suffix": item.get("UINWithAmendmentSuffix"),
        "amendment_to_motion_id": item.get("AmendmentToMotionId"),
        "date_tabled": item.get("DateTabled"),
        "sponsors_count": item.get("SponsorsCount"),
    }


def _paginate_edms(
    client: httpx.Client | None,
    settings: Settings,
    tabled_start: dt.date,
    tabled_end: dt.date,
    *,
    force: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Walk every EDM-search page, from cache or network.

    The total is embedded in every page's own ``PagingInfo.Total``: same
    shape as Members/Search's ``totalResults``, so this mirrors
    ``_paginate_members`` (learn the total from page 0, page until an empty
    response confirms exhaustion) rather than the divisions module's
    separate-total-endpoint pattern.
    """
    rows: list[dict[str, Any]] = []
    skip = 0
    total: int | None = None
    while True:
        payload = _fetch_cached(
            client, settings, EDM_SOURCE, EDM_LIST_BASE,
            _edm_list_params(tabled_start, tabled_end, skip), force=force,
        )
        items = payload.get("Response") or []
        if total is None:
            total = payload["PagingInfo"]["Total"]
        if not items:
            break
        rows.extend(_flatten_edm(item) for item in items)
        skip += EDM_PAGE_SIZE
    assert total is not None, "EDM list API returned no pages at all"
    return rows, total


def fetch_edms(
    tabled_start: dt.date,
    tabled_end: dt.date,
    client: httpx.Client | None,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Every EDM tabled in ``[tabled_start, tabled_end]`` (inclusive both
    ends: verified live against the Easter 2026 recess, where the API
    correctly returns nothing tabled after the last pre-recess sitting day
    even with no explicit upper bound), one row per motion. No signatories
    here: see :func:`fetch_edm_signatures` for those.
    """
    settings = settings or get_settings()
    rows, total = _paginate_edms(client, settings, tabled_start, tabled_end, force=force)
    assert len(rows) == total, (
        f"collected {len(rows)} EDMs but API reports PagingInfo.Total={total}"
    )
    df = pd.DataFrame(rows)
    write_interim(df, "edms", settings=settings)
    return df


def cached_edms(
    tabled_start: dt.date, tabled_end: dt.date, settings: Settings | None = None
) -> pd.DataFrame | None:
    """Reassemble the EDM table purely from disk, or ``None`` if the cache

    doesn't yet cover every page. Lets a script find which motion ids need
    signature detail without a network call.
    """
    settings = settings or get_settings()
    try:
        rows, total = _paginate_edms(None, settings, tabled_start, tabled_end, force=False)
    except CollectError:
        return None
    if len(rows) != total:
        return None
    return pd.DataFrame(rows)


def plan_edms(
    tabled_start: dt.date, tabled_end: dt.date, settings: Settings | None = None
) -> FetchPlan:
    """What a full EDM-list pull would need, read entirely from disk.

    Same caveat as :func:`plan_members`: ``total_needed`` counts only the
    data-bearing pages implied by the total, not the final empty-page call
    that confirms exhaustion, so a real run makes one more request than
    this reports once the list is otherwise fully cached.
    """
    settings = settings or get_settings()
    page0_path = _cache_path(settings, EDM_SOURCE, EDM_LIST_BASE, _edm_list_params(tabled_start, tabled_end, 0))
    if not page0_path.exists():
        return FetchPlan(source="edms", total_needed=None, cached=0, missing=1,
                          note="first page not cached; total page count unknown until fetched")

    total = _read_cache(page0_path)["PagingInfo"]["Total"]
    n_pages = (total + EDM_PAGE_SIZE - 1) // EDM_PAGE_SIZE if total else 0
    cached = 0
    for skip in range(0, n_pages * EDM_PAGE_SIZE, EDM_PAGE_SIZE):
        params = _edm_list_params(tabled_start, tabled_end, skip)
        if _cache_path(settings, EDM_SOURCE, EDM_LIST_BASE, params).exists():
            cached += 1
    return FetchPlan(source="edms", total_needed=n_pages, cached=cached,
                      missing=n_pages - cached)


def _flatten_edm_signatures(motion_id: int, detail: dict[str, Any]) -> pd.DataFrame:
    """One row per (motion, signatory), long format: same shape as

    ``_flatten_division_votes``. ``Sponsors`` includes the primary sponsor
    (``sponsoring_order == 1``) as well as every subsequent signatory;
    nothing here distinguishes "tabled it" from "signed it" beyond that
    order field, which is a downstream construction call, not a parsing
    one.
    """
    rows: list[dict[str, Any]] = []
    for s in detail.get("Sponsors") or []:
        member = s.get("Member") or {}
        rows.append({
            "motion_id": motion_id,
            "member_id": s.get("MemberId"),
            "member_name": member.get("Name"),
            "member_party": member.get("Party"),
            "sponsoring_order": s.get("SponsoringOrder"),
            "created_when": s.get("CreatedWhen"),
            "is_withdrawn": s.get("IsWithdrawn"),
            "withdrawn_date": s.get("WithdrawnDate"),
        })
    return pd.DataFrame(
        rows,
        columns=[
            "motion_id", "member_id", "member_name", "member_party",
            "sponsoring_order", "created_when", "is_withdrawn", "withdrawn_date",
        ],
    )


def fetch_edm_signatures(
    motion_id: int,
    client: httpx.Client | None,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Every signatory for one motion, long format (one row per signatory).

    This is the endpoint the list search doesn't carry: the list's
    ``SponsorsCount`` is a summary field only; the named, member-id-keyed
    list lives here, on the singular ``EarlyDayMotion/{id}`` detail route.
    A motion with no additional signatories beyond its primary sponsor
    still yields at least one row (the primary sponsor's own entry), same
    "real zero, not missing" convention as ``fetch_division_votes``.
    """
    settings = settings or get_settings()
    url = f"{EDM_DETAIL_BASE}/{motion_id}"
    payload = _fetch_cached(client, settings, EDM_SOURCE, url, None, force=force)
    detail = payload.get("Response") or {}
    return _flatten_edm_signatures(motion_id, detail)


def fetch_all_edm_signatures(
    motion_ids: list[int],
    client: httpx.Client | None,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Signatories for many motions, concatenated and written to interim.

    One request per motion: the same "long pull" shape as
    ``fetch_all_member_contributions``/``fetch_all_member_biographies``, and
    individually cached the same way, so it is resumable on the same terms.
    For the full 2024-07-04..2026-03-31 window (~3,069 motions) this is the
    dominant cost of the whole EDM collection at ~2 req/s.
    """
    settings = settings or get_settings()
    parts = [
        fetch_edm_signatures(mid, client, settings=settings, force=force)
        for mid in motion_ids
    ]
    columns = [
        "motion_id", "member_id", "member_name", "member_party",
        "sponsoring_order", "created_when", "is_withdrawn", "withdrawn_date",
    ]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    write_interim(df, "edm_signatures", settings=settings)
    return df


def plan_edm_signatures(motion_ids: list[int], settings: Settings | None = None) -> FetchPlan:
    settings = settings or get_settings()
    cached = sum(
        1 for mid in motion_ids
        if cache_exists(settings, EDM_SOURCE, f"{EDM_DETAIL_BASE}/{mid}", None)
    )
    return FetchPlan(
        source="edm_signatures", total_needed=len(motion_ids), cached=cached,
        missing=len(motion_ids) - cached,
    )


# --------------------------------------------------------------------------
# interim output
# --------------------------------------------------------------------------


def write_interim(df: pd.DataFrame, name: str, *, settings: Settings | None = None) -> Path:
    """Write a parsed-but-not-derived table to ``data_interim/<name>.parquet``."""
    settings = settings or get_settings()
    settings.data_interim.mkdir(parents=True, exist_ok=True)
    path = settings.data_interim / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def read_interim(name: str, *, settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    return pd.read_parquet(settings.data_interim / f"{name}.parquet")


# --------------------------------------------------------------------------
# planning (for --dry-run)
# --------------------------------------------------------------------------


@dataclass
class FetchPlan:
    """What a collection step needs, and how much of it is already cached.

    ``total_needed`` is ``None`` when it can't be known without a network
    call (e.g. the first page of a paginated source hasn't been fetched yet).
    """

    source: str
    total_needed: int | None
    cached: int
    missing: int
    note: str = ""

    def summary(self) -> str:
        total = "?" if self.total_needed is None else str(self.total_needed)
        line = f"{self.source}: {self.cached}/{total} cached, {self.missing} missing"
        return f"{line} ({self.note})" if self.note else line


@dataclass
class CollectResult:
    """Outcome of a real (non-dry-run) collection step, for the run summary."""

    source: str
    rows: int
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
