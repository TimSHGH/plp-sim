"""Builds the network that links the 100 personas.

The network is over archetypes, not MPs. A synthetic persona has no voting
record, so a real-MP graph cannot be attached to it without smuggling real
individuals back in. Instead each persona is a cluster, and two clusters are
linked by how often their members voted together on the divisions where Labour
actually split. Synthetic nodes, real edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plp_sim import frames, network, schemas
from plp_sim.config import Settings

#: Neighbours shown to each persona. Small on purpose: an MP watches a handful
#: of colleagues they are politically close to, not a running tally of all 405.
#: Showing more would smuggle the aggregate back in through the side door.
N_NEIGHBOURS = 8


def cluster_network(
    attributes: pd.DataFrame, panel: pd.DataFrame, votes: pd.DataFrame, settings: Settings
) -> dict[int, list[int]]:
    """Neighbour lists over P2 archetypes, from their members' real co-voting.

    Rebuilds the same clustering ``personas.build_p2`` used (same Gower matrix,
    same seed, so the labels line up), then scores every pair of clusters by the
    mean agreement of their members on contested divisions.
    """
    df = attributes.reset_index(drop=True)
    d = frames.gower_matrix(df, list(schemas.SEG_VARS))
    from kmedoids import KMedoids

    labels = pd.Series(
        KMedoids(settings.n_personas, method="fasterpam", metric="precomputed",
                 random_state=settings.random_seed).fit(d).labels_,
        index=df.index,
    )
    members = {int(c): df.loc[idx, "member_id"].tolist()
               for c, idx in labels.groupby(labels).groups.items()}

    contested = network.contested_divisions(votes, set(df["member_id"]))
    sub = votes[votes["division_id"].isin(contested)]
    grid = sub.pivot_table(index="member_id", columns="division_id", values="vote",
                           aggfunc="first")

    cluster_ids = sorted(members)
    profile = {}
    for c in cluster_ids:
        rows = grid.reindex(members[c]).dropna(how="all")
        if rows.empty:
            profile[c] = None
            continue
        # modal vote per division = the cluster's revealed position
        profile[c] = rows.mode(axis=0).iloc[0] if len(rows) else None

    sim = pd.DataFrame(np.nan, index=cluster_ids, columns=cluster_ids, dtype=float)
    for i, a in enumerate(cluster_ids):
        for b in cluster_ids[i + 1:]:
            pa, pb = profile[a], profile[b]
            if pa is None or pb is None:
                continue
            both = pa.notna() & pb.notna()
            if both.sum() < network.MIN_SHARED:
                continue
            agree = float((pa[both] == pb[both]).mean())
            sim.loc[a, b] = sim.loc[b, a] = agree

    # panel persona_id is assigned in cluster-label order by build_p2
    pid_of = {c: int(p) for c, p in zip(cluster_ids, sorted(panel["persona_id"]))}
    return {
        pid_of[c]: [pid_of[o] for o in sim.loc[c].dropna().nlargest(N_NEIGHBOURS).index]
        for c in cluster_ids
    }
