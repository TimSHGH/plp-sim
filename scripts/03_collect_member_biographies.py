#!/usr/bin/env python
"""Collect post and committee-membership history for every PLP member.

``is_payroll`` and ``committee_count`` are hardcoded (False / 0) in
``attributes.py`` today because ``collect.py`` never called anything but
Members/Search, which carries no posts data. This script calls the sibling
``/api/Members/{id}/Biography`` endpoint (verified against live data: see
``collect.py``'s module docstring) for every PLP member, using the same
per-member cache and ~2 req/s rate limit as the Hansard contributions pull.

~405 members at ~2 req/s is ~3-4 minutes. Resumable: rerun after an
interruption and every member already cached is skipped at zero network cost.
"""

from __future__ import annotations

import argparse

import httpx

from plp_sim import collect
from plp_sim.config import get_settings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    settings.ensure_dirs()

    members = collect.cached_members(settings)
    if members is None:
        raise SystemExit("members not fully cached yet: run scripts/01_collect.py first")
    plp_ids = collect.plp_member_ids(members, settings)

    plan = collect.plan_member_biographies(plp_ids, settings)
    if args.dry_run:
        print(f"Dry run: no network calls made.\n  {plan.summary()}")
        return

    print(f"Fetching biographies for {len(plp_ids)} PLP members ({plan.summary()}) ...")
    with httpx.Client(headers={"Accept": "application/json"}) as client:
        bios = collect.fetch_all_member_biographies(
            plp_ids, client, settings=settings, force=args.force
        )
    print(f"member_biography: {len(bios)} post/committee rows across {bios['member_id'].nunique()} members")


if __name__ == "__main__":
    main()
