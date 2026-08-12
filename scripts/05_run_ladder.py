#!/usr/bin/env python
"""Run the full persona ladder.

Builds every panel, generates P3's biographies, then elicits the validation
items (scored against observed outcomes) and the decision items (the actual
survey) for every method.

Resumable: every call is disk-cached on
(method, persona, item, model, prompt_version, option_order, draw_index,
temperature), so re-running after an interruption costs nothing for work
already done.
"""

from __future__ import annotations

import argparse
import asyncio
import time

import openai
import pandas as pd

from plp_sim import elicit, personas, schemas
from plp_sim.config import get_settings


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", choices=["validation", "decision", "both"], default="both")
    ap.add_argument("--orders", choices=["forward", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="cap personas per panel (smoke test)")
    args = ap.parse_args()

    cfg = get_settings()
    cfg.ensure_dirs()
    attributes = pd.read_parquet(cfg.data_processed / "attributes.parquet")
    frame = pd.read_parquet(cfg.data_processed / "frame_F1.parquet")
    instrument = elicit.load_instrument()
    client = openai.AsyncOpenAI(api_key=cfg.openai_api_key)
    cache = elicit.open_cache(cfg.cache_dir)

    # ---- panels ---------------------------------------------------------
    panels: dict[str, pd.DataFrame] = {
        m: personas.build_panel(m, attributes, cfg) for m in ("P0", "P1", "P2")
    }
    panels |= {m: personas.build_panel(m, attributes, cfg, frame=frame) for m in ("P4", "P5")}

    # P3 is P2's vectors plus generated prose, so the only difference between
    # the two conditions is the biography.
    print("generating P3 biographies...", flush=True)
    panels["P3"] = await personas.generate_biographies(client, cfg, panels["P2"])

    # RECALL needs real names -- it is the leakage control, not a persona.
    panels["RECALL"] = panels["P5"].assign(method="RECALL")

    # P3S is P3's biographies permuted onto the wrong personas. Same length,
    # same register, same total information; only the match between prose and
    # vector is destroyed. P3 - P3S is what the biography is actually worth.
    panels["P3S"] = personas.shuffle_biographies(panels["P3"], cfg)

    if args.limit:
        panels = {m: p.head(args.limit) for m, p in panels.items()}

    for m, panel in panels.items():
        print(f"  {m:<7} {len(panel):>4} personas", flush=True)
    pd.concat(panels.values()).to_parquet(cfg.data_processed / "personas.parquet", index=False)

    orders = ("forward", "reversed") if args.orders == "both" else ("forward",)

    # ---- elicit ---------------------------------------------------------
    blocks = []
    for kind, key in (("validation", "items"),):  # one set now: all three are scored
        if args.items not in (kind, "both"):
            continue
        items = instrument[key]
        n = sum(len(p) for p in panels.values()) * len(items) * len(orders)
        print(f"\n{kind}: {len(items)} items x {len(orders)} order(s) = {n} calls", flush=True)
        t0 = time.time()
        df = await elicit.run_ladder(
            client, cfg, instrument=instrument, panels=panels,
            items=items, option_orders=orders, cache=cache,
        )
        df["item_kind"] = kind
        blocks.append(df)
        cached = int(df["cached"].sum())
        print(f"  {len(df)} rows in {time.time()-t0:.0f}s "
              f"({cached} cached, {len(df)-cached} live)", flush=True)

    out = pd.concat(blocks, ignore_index=True)
    out.to_parquet(cfg.data_processed / "elicitation.parquet", index=False)

    low = (out["captured_mass"] < schemas.MIN_CAPTURED_MASS).sum()
    print(f"\nwrote {len(out)} rows -> data/processed/elicitation.parquet")
    print(f"captured_mass below {schemas.MIN_CAPTURED_MASS}: {low}/{len(out)}")
    print(out.groupby("method")["captured_mass"].mean().round(4).to_string())


if __name__ == "__main__":
    asyncio.run(main())
