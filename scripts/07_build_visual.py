#!/usr/bin/env python
"""The artificial society under rising pressure -- final build.

Four views of the SAME 100 personas. Only one thing changes between them: how
many colleagues have already broken cover. Everything else -- personas, network,
layout coordinates, item wording, decoding -- is held fixed, so a node that
changes colour has changed its mind rather than drifted.

Built on the best defensible configuration the study reached, not the first one
that ran:

* **P2C personas** -- archetype vector plus pre-cutoff situational context (a
  January 2026 constituency MRP and the polling trend to 30 March 2026).
* **Position-conditional political physics** -- the manifesto is written from
  each persona's own ``is_payroll``, because the observed data says the two
  groups are not doing the same thing. Payroll MPs were 94.6% silent: that is
  collective responsibility, a rule about who may speak, not a view on the
  leadership. Backbenchers split nearly three ways.

Deliberately NOT included: the one-sided framing that put 8% on the exit option,
and the explicit-norms layer that put 100% on it. Both work by arguing for the
answer being scored, so both are diagnostics rather than methods.

Encoding: colour is the position taken, shape is the persona's own position in
the party. Shape carries the payroll split so it survives greyscale and colour
blindness, and so the two findings can be read at once.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import networkx as nx
import numpy as np
import openai
import pandas as pd

from plp_sim import cascade, dossier, elicit, metrics, network, personas
from plp_sim.config import get_settings

STATES = ["back", "exit", "silent"]

#: Every string that defines this panel now lives in config/instrument.yaml, so
#: the published artefact is reproducible from config rather than from a copy of
#: it that happens to sit in this file. Rewording any of them changes the
#: published numbers -- see the warning on `pressure_panel` in that file.
_PP = elicit.load_instrument()["pressure_panel"]
SUCC = _PP["successor_premise"]
PAYROLL = _PP["manifesto"]["payroll"]
BACKBENCH = _PP["manifesto"]["backbench"]
ARMS = [(lv["id"], lv["label"], lv["social"]) for lv in _PP["levels"]]


async def main() -> None:
    cfg = get_settings()
    attrs = pd.read_parquet(cfg.data_processed / "attributes.parquet")
    votes = pd.read_parquet(cfg.data_interim / "division_votes.parquet")
    instrument = elicit.load_instrument()
    item = next(i for i in instrument["validation_items"] if i["id"] == "v_loyalty")
    client = openai.AsyncOpenAI(api_key=cfg.openai_api_key)
    cache = elicit.open_cache(cfg.cache_dir)

    p2 = personas.build_panel("P2", attrs, cfg)
    p2c = personas.add_situational_context(p2, attrs, cfg)
    neigh = cascade.cluster_network(attrs, p2, votes, cfg)

    g = nx.Graph()
    g.add_nodes_from(int(p) for p in p2c["persona_id"])
    for a, ns in neigh.items():
        for b in ns:
            g.add_edge(int(a), int(b))
    pos = nx.kamada_kawai_layout(g)
    comm = network.detect_communities(g, seed=cfg.random_seed)

    hold = pd.read_parquet(cfg.data_processed / "holdout.parquet")
    ev = hold[hold.event_id == "starmer_loyalty"].merge(
        attrs[["member_id", "is_payroll"]], on="member_id")
    cols = list(item["outcome_map"])
    obs = {"all": ev.outcome.value_counts(normalize=True).reindex(cols).fillna(0).to_numpy()}
    for grp, d in ev.groupby("is_payroll"):
        obs["payroll" if grp else "backbench"] = (
            d.outcome.value_counts(normalize=True).reindex(cols).fillna(0).to_numpy())
    obs_n = {"all": len(ev), "payroll": int(ev.is_payroll.sum()),
             "backbench": int((~ev.is_payroll).sum())}

    renderer = dossier.P2CSituated(manifesto={"payroll": PAYROLL, "backbench": BACKBENCH})
    w = p2c.set_index("persona_id")["weight"]
    pay = p2c.set_index("persona_id")["is_payroll"].astype(bool)
    tvd = lambda s, o: round(metrics.one_minus_tvd(s, o), 3)

    tabs = []
    with dossier.using("P2C", renderer):
      for tag, label, social in ARMS:
         it = dict(item)
         it["text"] = item["text"].strip() + SUCC + social
         it["id"] = f"v_loyalty__pos_L5_{tag}"
         rows = [await elicit.elicit_item(client, cfg, method="P2C", frame="PANEL",
                                          persona=p, item=it, option_order="forward",
                                          draw_index=0, cache=cache)
                 for _, p in p2c.iterrows()]
         probs = np.vstack([r["probs"] for r in rows])
         pid = np.array([int(r["persona_id"]) for r in rows])
         wt = np.array([w[p] for p in pid])
         share, acc = {}, {}
         for gname, mask in (("all", np.ones(len(pid), bool)),
                             ("payroll", np.array([bool(pay[p]) for p in pid])),
                             ("backbench", np.array([not bool(pay[p]) for p in pid]))):
             s = (probs[mask] * wt[mask, None]).sum(0) / wt[mask].sum()
             share[gname] = {st: round(float(s[i]), 4) for i, st in enumerate(STATES)}
             acc[gname] = tvd(s, obs[gname])
         tabs.append({
             "id": tag, "label": label, "social": social.strip() or "Nothing. You would be among the first to speak.",
             "share": share, "tvd": acc,
             "answer": {str(p): {"state": STATES[int(pr.argmax())],
                                 "p": [round(float(x), 6) for x in pr]}
                        for p, pr in zip(pid, probs)},
         })
         print(f"{tag:6} all {acc['all']:.3f}   payroll {acc['payroll']:.3f}   "
               f"backbench {acc['backbench']:.3f}")

    idx = p2c.set_index("persona_id")
    nodes = []
    for p in sorted(int(x) for x in p2c["persona_id"]):
        r = idx.loc[p]
        nodes.append({
            "id": p, "x": round(float(pos[p][0]), 4), "y": round(float(pos[p][1]), 4),
            "w": round(float(w[p]), 4), "community": int(comm.get(p, 0)),
            "payroll": bool(r.get("is_payroll")),
            "majority": None if pd.isna(r.get("majority_pct")) else round(float(r["majority_pct"]), 1),
            "runner_up": r.get("runner_up_party"),
            "intake": "2024" if r.get("is_2024_intake") else "pre-2024",
            "rebellion": None if pd.isna(r.get("rebellion_rate")) else round(float(r["rebellion_rate"]), 3),
            "risk": None if pd.isna(r.get("seat_at_risk_share")) else round(float(r["seat_at_risk_share"]), 2),
            "proj_lab": None if pd.isna(r.get("projected_labour_share")) else round(float(r["projected_labour_share"]), 1),
            "proj_win": r.get("projected_winner"),
        })
    # Does defection concentrate in one part of the network? The personas are never
    # told anything about communities -- the edges come from how the real MPs behind
    # each cluster actually voted, and the grouping is Louvain over those edges. So a
    # concentration here is emergent, not designed in. Tested rather than eyeballed.
    from scipy import stats as _st

    final = tabs[-1]["answer"]
    comm = {n["id"]: n["community"] for n in nodes}
    tally: dict[int, list[int]] = {}
    for pid, v in final.items():
        row = tally.setdefault(comm[int(pid)], [0, 0])
        row[0 if v["state"] == "exit" else 1] += 1
    table = [r for r in tally.values() if sum(r) > 0]
    chi2, pval = (_st.chi2_contingency(table)[:2] if len(table) > 1 else (float("nan"),) * 2)
    clustering = {
        "chi2": round(float(chi2), 2), "p": float(pval),
        "communities": sorted(
            ({"id": c, "exit": r[0], "n": sum(r), "rate": round(r[0] / sum(r), 3)}
             for c, r in tally.items()), key=lambda d: -d["rate"]),
    }
    print(f"\ndefection clusters by community: chi2={chi2:.2f} p={pval:.4f}")
    for c in clustering["communities"]:
        print(f"  community {c['id']}: {c['exit']}/{c['n']} ({c['rate']:.0%})")

    payload = {
        "clustering": clustering,
        "nodes": nodes, "edges": [[int(a), int(b)] for a, b in g.edges()], "tabs": tabs,
        "observed": {k: {st: round(float(v[i]), 4) for i, st in enumerate(STATES)}
                     for k, v in obs.items()},
        "observed_n": obs_n,
        "item": " ".join(item["text"].split()) + SUCC,
        "options": dict(zip(STATES, item["options"])),
        "n_personas": len(nodes), "model": cfg.model,
    }
    tpl = Path("plp_sim/cascade_template.html").read_text()
    out = cfg.outputs / "AS_network_viz.html"
    out.write_text(tpl.replace("__DATA__", json.dumps(payload).replace("</", "<\\/")))
    print(f"\nwrote {out}  ({out.stat().st_size/1024:.0f} KB, self-contained)")


if __name__ == "__main__":
    asyncio.run(main())
