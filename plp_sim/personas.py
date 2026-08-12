"""Persona construction: the seven methods.

The task is to model a group with 100 personas, not to simulate 100 named
individuals. A method that needs a specific person's public record cannot be
pointed at "undecided voters in Red Wall seats", and cannot be told apart from
the model simply remembering that person.

So the ladder is ordered by how close each method gets to a real individual,
with the deployable line drawn explicitly:

    P0  stereotype        synthetic   deployable
    P1  quota             synthetic   deployable
    P2  archetype         synthetic   deployable
    P3  archetype + bio   synthetic   deployable
    ----------------------------------------------- the line
    P4  real, anonymised  real MP     ceiling only
    P5  real, named       real MP     ceiling only

P5 minus P2 is the headline: how much apparent accuracy comes from the model
already knowing these people rather than reasoning from the profile it was
given. On an audience it has never read about, that is the part that vanishes.

Individual accuracy is undefined for P0-P3 by construction. There is no real MP
behind a synthetic persona, so there is nothing to score one answer against.
Only the group distribution can be compared, which is what the task asks for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plp_sim import frames, schemas
from plp_sim.config import Settings

#: Attribute columns a persona carries. Deliberately a subset of ATTRIBUTES:
#: post-cutoff outcomes are excluded (they are what we predict), and so are the
#: HELDOUT_VARS the frame-error measurement depends on never having seen.
PERSONA_ATTRS: tuple[str, ...] = (
    "majority_pct",
    "vote_share",
    "runner_up_party",
    "is_2024_intake",
    "is_payroll",
    "committee_count",
    "rebellion_rate",
    "speech_count",
)

_NUMERIC = ("majority_pct", "vote_share", "committee_count", "rebellion_rate", "speech_count")
_CATEGORICAL = ("runner_up_party", "is_2024_intake", "is_payroll")


def _empty() -> pd.DataFrame:
    return schemas.PERSONA.empty()


def _finalise(rows: pd.DataFrame, method: str, n_pop: int) -> pd.DataFrame:
    """Coerce to the PERSONA contract and validate."""
    out = _empty()
    for col in out.columns:
        out[col] = rows[col] if col in rows else pd.Series([None] * len(rows), dtype=out[col].dtype)
    out["method"] = method
    out = out.astype(
        {
            "persona_id": "int64",
            "weight": "float64",
            "committee_count": "Int64",
            "speech_count": "Int64",
            "member_id": "Int64",
            "is_2024_intake": "boolean",
            "is_payroll": "boolean",
        }
    )
    assert abs(out["weight"].sum() - n_pop) < 1e-6, "weights must sum to the population size"
    return schemas.validate(out.reset_index(drop=True), schemas.PERSONA)


# --------------------------------------------------------------------------
# P0 -- stereotype
# --------------------------------------------------------------------------


def build_p0(attributes: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """One generic persona carrying the whole population's weight.

    Emitting a single row rather than 100 identical ones is the honest
    representation, and it is itself a finding worth a slide: a stereotype
    persona has **no within-panel variation by construction**. Asking it a
    question yields one opinion, not a distribution. Any spread a stereotype
    panel appears to show comes from sampling noise, not from modelling the
    group -- which is precisely the failure mode the sector baseline hides by
    running 100 copies and reporting the scatter.
    """
    rows = pd.DataFrame(
        {"persona_id": [0], "source": ["stereotype"], "weight": [float(len(attributes))],
         "cell": [None]}
    )
    return _finalise(rows, "P0", len(attributes))


# --------------------------------------------------------------------------
# P1 -- quota
# --------------------------------------------------------------------------


def build_p1(attributes: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Synthetic personas drawn so the panel's margins match the population.

    Reuses F1's stratification (``frames._stratum_labels``) to define quota
    cells, then, within a cell, draws **each attribute independently** from that
    cell's own empirical distribution.

    Independent draws are the point, not a shortcut. Copying a real MP's whole
    row would reproduce an individual and reintroduce exactly the retrieval
    problem this ladder exists to measure. Drawing per attribute preserves the
    cell's marginals (so the panel still represents the population) while making
    it very unlikely any persona coincides with a real member -- and
    ``no_synthetic_persona_matches_a_real_mp`` asserts that it does not.

    The cost is real and worth stating: within-cell correlation between
    attributes is broken. Cells are defined on the variables that matter most,
    so the correlation that survives is the between-cell structure.
    """
    df = attributes.reset_index(drop=True)
    rng = np.random.default_rng(settings.random_seed)

    stratum = frames._stratum_labels(df)
    sizes = stratum.value_counts()
    alloc = frames._proportional_allocation(sizes, settings.n_personas)

    records: list[dict[str, object]] = []
    pid = 0
    for label in sorted(alloc.index):
        k = int(alloc[label])
        if k <= 0:
            continue
        cell = df[(stratum == label).to_numpy()]
        share = float(sizes[label]) / k
        for _ in range(k):
            rec: dict[str, object] = {
                "persona_id": pid, "source": "quota", "weight": share, "cell": str(label),
            }
            for col in PERSONA_ATTRS:
                pool = cell[col].dropna()
                rec[col] = None if pool.empty else pool.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
            records.append(rec)
            pid += 1
    return _finalise(pd.DataFrame(records), "P1", len(df))


# --------------------------------------------------------------------------
# P2 -- archetype
# --------------------------------------------------------------------------


def build_p2(attributes: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Cluster central tendencies: the profile of a *type*, not of a person.

    Clusters exactly as F2 does (Gower distance, k-medoids) but then discards
    the medoid member and describes the cluster instead -- mean for numeric
    attributes, mode for categorical. The medoid is a real MP; its cluster's
    centre is not anyone.

    That one substitution is the whole difference between person simulation and
    persona modelling, and it costs almost nothing: the profile still sits at
    the centre of a real group of members.
    """
    df = attributes.reset_index(drop=True)
    d = frames.gower_matrix(df, list(schemas.SEG_VARS))
    from kmedoids import KMedoids

    km = KMedoids(
        settings.n_personas, method="fasterpam", metric="precomputed",
        random_state=settings.random_seed,
    ).fit(d)
    labels = pd.Series(km.labels_, index=df.index)

    records: list[dict[str, object]] = []
    for pid, (label, idx) in enumerate(labels.groupby(labels).groups.items()):
        block = df.loc[idx]
        rec: dict[str, object] = {
            "persona_id": pid, "source": "archetype", "weight": float(len(block)),
            "cell": f"cluster_{label}",
        }
        for col in _NUMERIC:
            v = block[col].dropna()
            rec[col] = None if v.empty else (round(float(v.mean()), 4)
                                             if col not in ("committee_count", "speech_count")
                                             else round(v.mean()))
        for col in _CATEGORICAL:
            v = block[col].dropna()
            rec[col] = None if v.empty else v.mode().iloc[0]
        records.append(rec)
    return _finalise(pd.DataFrame(records), "P2", len(df))


# --------------------------------------------------------------------------
# P4 / P5 -- the ceiling
# --------------------------------------------------------------------------


def build_real(attributes: pd.DataFrame, frame: pd.DataFrame, method: str) -> pd.DataFrame:
    """P4/P5: real MPs selected by a frame. The contamination ceiling.

    Kept because the gap between these and P2 is the number the whole study
    exists to produce. They are never presented as deployable: both require a
    real individual's public record, which a client audience does not have.
    """
    if method not in ("P4", "P5"):
        raise ValueError(f"build_real is for P4/P5, got {method!r}")
    a = attributes.set_index("member_id")
    ids = [int(m) for m in frame["member_id"]]
    w = frame.set_index("member_id")["weight"]

    records = []
    for pid, m in enumerate(ids):
        row = a.loc[m]
        rec: dict[str, object] = {
            "persona_id": pid, "source": "real", "member_id": m,
            "weight": float(w[m]), "cell": None,
        }
        for col in PERSONA_ATTRS:
            rec[col] = None if pd.isna(row[col]) else row[col]
        rec["role"] = None if pd.isna(row.get("role")) else row.get("role")
        if method == "P5":  # the name is the whole difference between P4 and P5
            rec["name"] = row["name"]
            rec["constituency"] = row["constituency"]
        records.append(rec)
    return _finalise(pd.DataFrame(records), method, round(float(w.sum())))


BUILDERS = {"P0": build_p0, "P1": build_p1, "P2": build_p2}


def build_panel(
    method: str, attributes: pd.DataFrame, settings: Settings,
    frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the persona panel for ``method``."""
    if method in BUILDERS:
        return BUILDERS[method](attributes, settings)
    if method in ("P4", "P5"):
        if frame is None:
            raise ValueError(f"{method} needs a frame: it selects real members")
        return build_real(attributes, frame, method)
    raise ValueError(f"no panel builder for method {method!r}")


# --------------------------------------------------------------------------
# P3 -- generated biographies
# --------------------------------------------------------------------------

#: Deliberately tight. The biography must dramatise the vector, not extend it:
#: every invented specific is a stereotype the attributes never justified, and
#: P3-vs-P2 on the identical vector is what measures exactly that.
BIOGRAPHY_SYSTEM = (
    "You write short, plausible profiles of British Members of Parliament for a "
    "research simulation. You will be given an attribute record. Write 90-120 "
    "words in the third person describing this MP's outlook, priorities and "
    "political instincts.\n\n"
    "Work only from the record. Do not invent a name, a constituency, an age, a "
    "gender, an ethnicity, a profession before politics, or any biographical "
    "fact the record does not contain. Do not name real people, real places or "
    "real factions. Describe disposition and motivation, which the record "
    "implies, rather than life history, which it does not. Write plain prose "
    "with no preamble and no headings."
)


def biography_prompt(persona: pd.Series) -> str:
    """The user message for one biography. Imported lazily to avoid a cycle."""
    from plp_sim import dossier

    return "Attribute record:\n\n" + dossier._attribute_bundle(persona)


async def generate_biographies(client, cfg: Settings, panel: pd.DataFrame) -> pd.DataFrame:
    """Return ``panel`` with a ``biography`` per row, generated once and cached.

    Takes a P2 panel and returns a P3 panel over the *same vectors*, which is
    what makes the P3-vs-P2 comparison clean: the only difference between the
    two conditions is the prose.
    """
    import asyncio
    import hashlib

    import diskcache

    cache = diskcache.Cache(str(cfg.cache_dir))
    sem = asyncio.Semaphore(cfg.max_concurrency)

    async def one(row: pd.Series) -> str:
        prompt = biography_prompt(row)
        key = "bio|" + hashlib.sha256(
            f"{cfg.model}|{cfg.prompt_version}|{prompt}".encode()
        ).hexdigest()
        hit = cache.get(key)
        if hit is not None:
            return str(hit)
        async with sem:
            resp = await client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "system", "content": BIOGRAPHY_SYSTEM},
                          {"role": "user", "content": prompt}],
                max_tokens=260,
            )
        text = resp.choices[0].message.content.strip()
        cache.set(key, text)
        return text

    bios = await asyncio.gather(*(one(panel.iloc[i]) for i in range(len(panel))))
    out = panel.copy()
    out["method"] = "P3"
    out["biography"] = list(bios)
    return schemas.validate(out.reset_index(drop=True), schemas.PERSONA)


def shuffle_biographies(p3: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """P3S: the same biographies, permuted onto the wrong personas.

    A derangement (no persona keeps its own text), so the control is total
    rather than partial. Everything about the prompt except relevance is held
    fixed, which makes ``P3 - P3S`` the cleanest available estimate of what the
    biography's *content* contributes over its *length*.
    """
    rng = np.random.default_rng(seed)
    n = len(p3)
    perm = np.arange(n)
    while True:
        rng.shuffle(perm)
        if n < 2 or not (perm == np.arange(n)).any():
            break
    out = p3.copy()
    out["method"] = "P3S"
    out["biography"] = p3["biography"].to_numpy()[perm]
    return schemas.validate(out.reset_index(drop=True), schemas.PERSONA)


# --------------------------------------------------------------------------
# P2C -- archetype plus situational context
# --------------------------------------------------------------------------


def national_context(settings: Settings, *, weeks: int = 8) -> str:
    """The shared political weather: identical text for every persona.

    Derived from the polling table rather than written by hand, so it is
    reproducible and cannot drift from the data. Reports levels and the change
    over the window, and stops there -- it does not say the leader is the cause
    of the decline, which is the inference under measurement.

    Admissibility is by **publication** date, not fieldwork date. A poll whose
    fieldwork closed in March but which was published in April describes a world
    no MP could have seen before the cutoff; four such polls are excluded
    upstream in ``data/manual/POLLING_NOTES.md``.
    """
    df = pd.read_csv(settings.data_manual / "polling_context.csv")
    df["d"] = pd.to_datetime(df["fieldwork_end"])
    recent = df[df["d"] >= df["d"].max() - pd.Timedelta(weeks=weeks)]
    early = df[df["d"] <= df["d"].min() + pd.Timedelta(weeks=weeks)]
    lab, ref = recent["labour"].mean(), recent["reform"].mean()
    grn, net = recent["green"].mean(), recent["leader_net"].dropna()
    rank = 1 + sum(recent[c].mean() > lab for c in ("reform", "conservative", "green", "libdem"))
    place = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}[rank]
    out = [
        (f"- National voting intention over the last {weeks} weeks: Labour {lab:.0f}%, "
         f"Reform UK {ref:.0f}%, Green {grn:.0f}%. Labour is in {place} place."),
        (f"- Labour polled {early['labour'].mean():.0f}% at the start of this Parliament's "
         f"second year, so this is a fall of {early['labour'].mean() - lab:.0f} points."),
    ]
    if len(net):
        out.append(f"- The party leader's net satisfaction rating is {net.mean():+.0f}.")
    return "\n".join(out)


def add_situational_context(panel: pd.DataFrame, attributes: pd.DataFrame,
                            settings: Settings, *, include_national: bool = True) -> pd.DataFrame:
    """Attach seat-risk context to an archetype panel, producing P2C.

    Every other attribute describes the persona. These describe the *world* it
    is in, and their absence is the leading explanation for the study's biggest
    failure: no persona, under any framing or model, will say a leader should
    go. A real MP calls for a leader to go when that leader is costing them
    their seat, and until now nothing in the profile said anything about that.

    An archetype is a cluster, not a person, so "is your seat at risk" becomes
    "what share of MPs like you are projected to lose theirs" -- an honest
    aggregation rather than a borrowed individual fact.
    """
    proj = pd.read_csv(settings.data_manual / "seat_projections.csv")
    risk = proj.set_index("constituency")

    df = attributes.reset_index(drop=True)
    d = frames.gower_matrix(df, list(schemas.SEG_VARS))
    from kmedoids import KMedoids

    labels = pd.Series(
        KMedoids(settings.n_personas, method="fasterpam", metric="precomputed",
                 random_state=settings.random_seed).fit(d).labels_,
        index=df.index,
    )
    out = panel.copy()
    out["method"] = "P2C"
    shares, winners, lab = [], [], []
    for pid, (_, idx) in zip(sorted(out["persona_id"]), labels.groupby(labels).groups.items()):
        seats = df.loc[idx, "constituency"]
        block = risk.reindex(seats).dropna(subset=["at_risk"])
        if block.empty:
            shares.append(None); winners.append(None); lab.append(None); continue
        shares.append(round(float(block["at_risk"].mean()), 3))
        atrisk = block[block["at_risk"]]
        winners.append(atrisk["projected_winner"].mode().iloc[0] if len(atrisk) else None)
        lab.append(round(float(block["projected_labour_share"].mean()), 1))
    if include_national:
        out["national_context"] = national_context(settings)
    order = out.sort_values("persona_id").index
    out.loc[order, "seat_at_risk_share"] = shares
    out.loc[order, "projected_winner"] = winners
    out.loc[order, "projected_labour_share"] = lab
    return schemas.validate(out.reset_index(drop=True), schemas.PERSONA)
