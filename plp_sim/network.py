"""Co-voting similarity between MPs, and the communities it implies.

Two MPs are linked by how often they voted the same way on the divisions where
Labour split. Whipped votes are excluded by construction: when 300 Labour MPs
vote identically, agreement carries no information, and including those votes
would push every pair near 1.0 and produce a hairball. Only the divisions with
real dissent are used.

This is revealed behaviour rather than stated position, which is what makes the
resulting communities worth something: nothing tells the personas that groups
exist.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

#: A division counts as contested if at least this many PLP members broke from
#: the Labour majority. Below this, agreement is near-universal and tells us
#: nothing about faction.
MIN_DISSENTERS = 3

#: Each node keeps its strongest ``TOP_K`` links. Without sparsification every
#: pair has some similarity and the graph is a hairball with no visible
#: structure. 6 is enough to keep the graph connected without over-linking.
TOP_K = 6

#: Pairs must share at least this many contested divisions before their
#: agreement rate is trusted. Two MPs who overlap on three votes can agree
#: perfectly by luck.
MIN_SHARED = 5


def contested_divisions(
    votes: pd.DataFrame, member_ids: set[int], min_dissenters: int = MIN_DISSENTERS
) -> pd.Index:
    """Divisions where at least ``min_dissenters`` PLP members broke ranks.

    The whip proxy is the Labour majority position in that division; there is
    no independent whip-instruction field in the source.
    """
    plp_votes = votes[votes["member_id"].isin(member_ids)]
    majority = plp_votes.groupby("division_id")["vote"].agg(lambda s: s.value_counts().idxmax())
    dissent = (
        plp_votes.assign(_rebel=plp_votes["vote"].ne(plp_votes["division_id"].map(majority)))
        .groupby("division_id")["_rebel"]
        .sum()
    )
    return dissent[dissent >= min_dissenters].index






def detect_communities(g: nx.Graph, seed: int = 0) -> dict[int, int]:
    """Louvain communities, relabelled largest-first.

    Uses networkx's built-in implementation rather than the stale
    `python-louvain` package the original design brief named.

    Largest-first ordering matters downstream: the chart can only colour three
    communities before the palette stops being colourblind-safe when every
    group is on screen at once, so the fourth and beyond fold into a neutral
    "Other". Ordering by size makes that fold take the least information.
    """
    parts = nx.community.louvain_communities(g, weight="weight", seed=seed)
    parts = sorted(parts, key=len, reverse=True)
    return {node: idx for idx, part in enumerate(parts) for node in part}


