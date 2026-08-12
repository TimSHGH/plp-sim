#!/usr/bin/env python
"""Position-conditional political physics, gain-framed.

Earlier manifesto layers applied one sentence to all 100 personas. The observed
data says that is the wrong shape:

    payroll     0.000 backed   0.054 exit   0.946 SILENT
    backbench   0.333 backed   0.301 exit   0.365 silent

Payroll MPs did not back the leader. They said nothing, and they did so almost
unanimously -- that is collective responsibility, a rule about who may speak,
not a position on the leadership. Ministers and PPSs cannot comment publicly
without resigning the post first. Backbenchers are the ones actually making a
political choice, and they split nearly evenly three ways.

Two corrections to the previous framing, both from the same observation:

* **Not "both misjudgements end careers."** That is loss-framed and symmetric,
  and it is wrong for a backbencher, who has no post to lose. The real pull
  toward a challenger is UPSIDE: early public support for the eventual winner
  is how a backbencher gets promoted.
* **The trade-off is position-dependent**, so the text is conditioned on the
  persona's own ``is_payroll`` rather than applied uniformly. That is persona
  modelling rather than prompt engineering: the same manifesto says different
  things to different personas because their situations genuinely differ.

Scored by group, not just on the topline. A method that hits the aggregate by
getting both subgroups wrong in opposite directions has learned nothing.
"""

from __future__ import annotations

import asyncio

import numpy as np
import openai
import pandas as pd

from plp_sim import dossier, elicit, metrics, personas
from plp_sim.config import get_settings

SUCC = (" A senior colleague is widely seen as a stronger candidate and consistently "
        "polls better than the current leader; if the leader went, they would be the "
        "likely successor.")
SOCIAL = {"none": "",
          "some": (" Around fifty of your colleagues have already publicly called for "
                   "the leader to go.")}

PAYROLL = ("\n\nWhere you stand: you hold a government post. Ministers and parliamentary "
           "private secretaries are bound by collective responsibility and do not comment "
           "publicly on the leadership; an MP who wants to say something about it resigns "
           "the post first. Your career runs through the government, whoever leads it.")
BACKBENCH = ("\n\nWhere you stand: you hold no government post. You are free to speak and "
             "you have no post to lose by speaking. Backbenchers who publicly support the "
             "eventual winner early are the ones a new leader promotes; those who are late, "
             "or who back the side that loses, wait another parliament for the chance.")


def _manifesto(p: pd.Series) -> str:
    return PAYROLL if bool(p.get("is_payroll")) else BACKBENCH


async def main() -> None:
    cfg = get_settings()
    attrs = pd.read_parquet(cfg.data_processed / "attributes.parquet")
    instrument = elicit.load_instrument()
    item = next(i for i in instrument["validation_items"] if i["id"] == "v_loyalty")
    client = openai.AsyncOpenAI(api_key=cfg.openai_api_key)
    cache = elicit.open_cache(cfg.cache_dir)
    p2c = personas.add_situational_context(personas.build_panel("P2", attrs, cfg), attrs, cfg)

    hold = pd.read_parquet(cfg.data_processed / "holdout.parquet")
    ev = hold[hold.event_id == "starmer_loyalty"].merge(
        attrs[["member_id", "is_payroll"]], on="member_id")
    cols = list(item["outcome_map"])
    obs_all = ev.outcome.value_counts(normalize=True).reindex(cols).fillna(0).to_numpy()
    obs_grp = {g: d.outcome.value_counts(normalize=True).reindex(cols).fillna(0).to_numpy()
               for g, d in ev.groupby("is_payroll")}

    w = p2c.set_index("persona_id")["weight"]
    pay = p2c.set_index("persona_id")["is_payroll"].astype(bool)

    tvd = metrics.one_minus_tvd

    print(f"{'arm':22}{'group':11}{'n':>4}{'back':>8}{'exit':>8}{'silent':>8}{'1-TVD':>8}")
    arms = (("L0 control", None),
            ("L5 position", {"payroll": PAYROLL, "backbench": BACKBENCH}))
    for arm, man in arms:
        with dossier.using("P2C", dossier.P2CSituated(manifesto=man)):
          for sname, stext in SOCIAL.items():
             it = dict(item)
             it["text"] = item["text"].strip() + SUCC + stext
             it["id"] = f"v_loyalty__pos_{arm.split()[0]}_{sname}"
             rows = [await elicit.elicit_item(client, cfg, method="P2C", frame="PANEL",
                                              persona=p, item=it, option_order="forward",
                                              draw_index=0, cache=cache)
                     for _, p in p2c.iterrows()]
             probs = np.vstack([r["probs"] for r in rows])
             pid = np.array([int(r["persona_id"]) for r in rows])
             wt = np.array([w[p] for p in pid])
             lab = f"{arm} / {sname}"
             s = (probs * wt[:, None]).sum(0) / wt.sum()
             print(f"{lab:22}{'ALL':11}{len(pid):>4}{s[0]:8.3f}{s[1]:8.3f}{s[2]:8.3f}"
                   f"{tvd(s, obs_all):8.3f}")
             for g, gname in ((True, "payroll"), (False, "backbench")):
                 m = np.array([bool(pay[p]) == g for p in pid])
                 if not m.any():
                     continue
                 sg = (probs[m] * wt[m, None]).sum(0) / wt[m].sum()
                 print(f"{'':22}{gname:11}{int(m.sum()):>4}{sg[0]:8.3f}{sg[1]:8.3f}{sg[2]:8.3f}"
                       f"{tvd(sg, obs_grp[g]):8.3f}")
    print(f"\n{'OBSERVED':22}{'ALL':11}{len(ev):>4}"
          f"{obs_all[0]:8.3f}{obs_all[1]:8.3f}{obs_all[2]:8.3f}")
    for g, gname in ((True, "payroll"), (False, "backbench")):
        o = obs_grp[g]
        print(f"{'':22}{gname:11}{int((ev.is_payroll == g).sum()):>4}"
              f"{o[0]:8.3f}{o[1]:8.3f}{o[2]:8.3f}")


if __name__ == "__main__":
    asyncio.run(main())
