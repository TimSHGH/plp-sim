#!/usr/bin/env python
"""Collect member-level vote detail for every PRE-CUTOFF division.

``scripts/01_collect.py`` only ever pulled a fixed sample of the *most
recent* divisions (``DIVISION_VOTES_SAMPLE = 10``), enough to prove the
endpoint end-to-end. All 10 of those happen to postdate
``settings.cutoff_date``, which leaves ``rebellion_rate`` -- a SEG_VAR built
strictly from pre-cutoff divisions, by construction (see
``attributes._rebellion_rate``) -- with zero non-leaky observations to draw
on.

This script closes that gap: it fetches vote detail for every division dated
strictly before ``settings.cutoff_date`` in the already-collected
``divisions.parquet`` (~466 of ~575), using the same per-division cache
(`fetch_division_votes`) and the same ~2 req/s rate limit as everything else
in ``collect.py``. It also keeps whatever post-cutoff divisions are already
cached (the original 10) rather than dropping them, so nothing already
collected is lost -- ``fetch_all_division_votes`` reads any already-cached
division from disk at zero network cost, so passing the union of pre-cutoff
+ already-covered-post-cutoff ids costs nothing extra for the ones already
on disk.

Resumable by construction: rerun after an interruption and every division
already cached is skipped (zero network calls for it); only the remainder
hits the API. Progress can be polled with ``--dry-run`` at any time (reads
the cache directory only, no network).
"""

from __future__ import annotations

import argparse

import httpx
import pandas as pd

from plp_sim import collect
from plp_sim.config import get_settings


def _target_division_ids(settings) -> list[int]:
    """Pre-cutoff division ids, unioned with whatever's already cached.

    The union (rather than pre-cutoff alone) means a rerun never loses the
    post-cutoff divisions ``01_collect.py`` already fetched -- reading them
    back off disk is free, so there is no reason to narrow the target set
    and risk shrinking ``division_votes.parquet`` on a rebuild.
    """
    divisions = collect.read_interim("divisions", settings=settings)
    dates = pd.to_datetime(divisions["date"])
    pre_cutoff_ids = set(divisions.loc[dates < pd.Timestamp(settings.cutoff_date), "division_id"])

    try:
        already = collect.read_interim("division_votes", settings=settings)
        covered_ids = set(already["division_id"].unique())
    except FileNotFoundError:
        covered_ids = set()

    return sorted(pre_cutoff_ids | covered_ids)


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

    target_ids = _target_division_ids(settings)
    plan = collect.plan_division_votes(target_ids, settings)

    if args.dry_run:
        print(f"Dry run: no network calls made.\n  {plan.summary()}")
        return

    print(f"Fetching division votes for {len(target_ids)} divisions ({plan.summary()}) ...")
    with httpx.Client(headers={"Accept": "application/json"}) as client:
        votes = collect.fetch_all_division_votes(
            target_ids, client, settings=settings, force=args.force
        )
    print(f"division_votes: {len(votes)} rows across {votes['division_id'].nunique()} divisions")


if __name__ == "__main__":
    main()
