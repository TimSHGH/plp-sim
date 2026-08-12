#!/usr/bin/env python
"""Turn the cached API responses into the four tables everything downstream reads.

Order matters:

1. **attributes** first, because everything else is derived from it.
2. **holdout** second, and deliberately before any panel is selected. The
   observed outcomes are fixed before anything gets to look at them, so no
   selection decision can be tuned, even accidentally, against the answers.
3. **frames** last. F1 is the analyst-designed panel, F2 the algorithmic one.
   Frame error is computed here, with no model involved at all, which makes it
   the honest ceiling on what any persona method can achieve.

Reads only from the local cache, so it makes no network calls and can be re-run
freely.
"""

from __future__ import annotations

import argparse

import pandas as pd

from plp_sim import attributes, frames, holdout
from plp_sim.config import get_settings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild tables that already exist")
    args = ap.parse_args()

    cfg = get_settings()
    cfg.ensure_dirs()

    # The cutoff is the foundation of every comparison downstream, so it is
    # checked before a single table is written rather than trusted.
    holdout.assert_cutoff_precedes_events(cfg)

    attrs_path = cfg.data_processed / "attributes.parquet"
    if args.force or not attrs_path.exists():
        attrs = attributes.build_attributes(cfg)
        attrs.to_parquet(attrs_path, index=False)
        print(f"attributes  {len(attrs):>4} MPs -> {attrs_path.name}")
    else:
        attrs = pd.read_parquet(attrs_path)
        print(f"attributes  {len(attrs):>4} MPs (cached, --force to rebuild)")

    hold_path = cfg.data_processed / "holdout.parquet"
    if args.force or not hold_path.exists():
        hold, meta = holdout.write_holdout(cfg)
        print(f"holdout     {len(hold):>4} rows across {hold['event_id'].nunique()} events")
        for k, v in sorted(meta.items()):
            print(f"              {k}: {v}")
    else:
        hold = pd.read_parquet(hold_path)
        print(f"holdout     {len(hold):>4} rows (cached)")

    for name, build in (("F1", frames.build_f1), ("F2", frames.build_f2)):
        path = cfg.data_processed / f"frame_{name}.parquet"
        if args.force or not path.exists():
            frame = build(attrs, cfg)
            frame.to_parquet(path, index=False)
        else:
            frame = pd.read_parquet(path)
        err = frames.frame_error(attrs, frame)
        multivariate = err.loc[err["metric"] == "energy_distance", "error"]
        print(f"frame {name}    {len(frame):>4} personas, "
              f"multivariate frame error {float(multivariate.iloc[0]):.4f}")

    # Frame error is the one number here that involves no model at all. It is
    # the ceiling: no prompt recovers a panel that does not represent the group.
    err_path = cfg.data_processed / "frame_error.parquet"
    pd.concat(
        [frames.frame_error(attrs, pd.read_parquet(cfg.data_processed / f"frame_{n}.parquet"))
         .assign(frame=n) for n in ("F1", "F2")],
        ignore_index=True,
    ).to_parquet(err_path, index=False)
    print(f"\nwrote {err_path.name}. Next: scripts/05_run_ladder.py")


if __name__ == "__main__":
    main()
