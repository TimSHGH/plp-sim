#!/usr/bin/env python
"""Thin CLI wrapper around ``plp_sim.collect``.

Every fetch is defined, cached, and independently testable in
``plp_sim/collect.py``; this script only sequences the calls and prints a
summary. No parsing, filtering, or derived logic lives here.

Default pipeline:
  1. members: all current Commons members
  2. divisions: division overviews since ``--since``
  3. division_votes: member-level Aye/No detail for the
                             ``DIVISION_VOTES_SAMPLE`` most recent divisions,
                             enough to prove the endpoint end-to-end without
                             committing to a full historical pull before
                             attribute construction has said which specific
                             divisions (welfare, winter fuel allowance, ...)
                             it actually needs vote detail for
  4. member_contributions: spoken-contribution counts for every PLP member
                             (the long pull: ~400+ requests at ~2/s)
"""

from __future__ import annotations

import argparse
import datetime as dt

import httpx

from plp_sim import collect
from plp_sim.config import get_settings

#: Not a population decision: just enough recent divisions to exercise
#: fetch_division_votes against the live API on every run. Call
#: collect.fetch_all_division_votes(ids, ...) directly with a specific id
#: list once the real target set (e.g. welfare/WFA divisions) is known.
DIVISION_VOTES_SAMPLE = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        type=dt.date.fromisoformat,
        default=collect.CURRENT_PARLIAMENT_START,
        help="earliest division date to collect, YYYY-MM-DD "
        f"(default: current Parliament start, {collect.CURRENT_PARLIAMENT_START.isoformat()})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be fetched and what's already cached; make no network calls",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="bypass the cache and refetch everything",
    )
    return parser.parse_args()


def _sample_division_ids(divisions) -> list[int]:
    return (
        divisions.sort_values("division_id", ascending=False)
        .head(DIVISION_VOTES_SAMPLE)["division_id"]
        .tolist()
    )


def _dry_run(since: dt.date) -> None:
    settings = get_settings()
    print(f"Dry run (since={since.isoformat()}): no network calls made.")

    print(" ", collect.plan_members(settings).summary())
    print(" ", collect.plan_divisions(since, settings).summary())

    divisions = collect.cached_divisions(since, settings)
    if divisions is not None:
        print(" ", collect.plan_division_votes(_sample_division_ids(divisions), settings).summary())
    else:
        print("  division_votes: unknown (divisions not fully cached yet)")

    members = collect.cached_members(settings)
    if members is not None:
        plp_ids = collect.plp_member_ids(members, settings)
        print(" ", collect.plan_member_contributions(plp_ids, settings).summary())
    else:
        print("  member_contributions: unknown (members not fully cached yet)")


def _real_run(since: dt.date, *, force: bool) -> None:
    settings = get_settings()
    with httpx.Client(headers={"Accept": "application/json"}) as client:
        before_members = collect.plan_members(settings)
        members = collect.fetch_members(client, settings=settings, force=force)
        print(f"members: {len(members)} rows ({before_members.summary()})")

        before_divisions = collect.plan_divisions(since, settings)
        divisions = collect.fetch_divisions(since, client, settings=settings, force=force)
        print(f"divisions: {len(divisions)} rows since {since.isoformat()} ({before_divisions.summary()})")

        sample_ids = _sample_division_ids(divisions)
        before_votes = collect.plan_division_votes(sample_ids, settings)
        votes = collect.fetch_all_division_votes(sample_ids, client, settings=settings, force=force)
        print(f"division_votes: {len(votes)} rows across {len(sample_ids)} divisions "
              f"({before_votes.summary()})")

        plp_ids = collect.plp_member_ids(members, settings)
        before_contrib = collect.plan_member_contributions(plp_ids, settings)
        contributions = collect.fetch_all_member_contributions(
            plp_ids, client, settings=settings, force=force
        )
        print(f"member_contributions: {len(contributions)} rows across {len(plp_ids)} PLP members "
              f"({before_contrib.summary()})")


def main() -> None:
    args = _parse_args()
    get_settings().ensure_dirs()
    if args.dry_run:
        _dry_run(args.since)
    else:
        _real_run(args.since, force=args.force)


if __name__ == "__main__":
    main()
