"""Table contracts for plp-sim.

Every module reads and writes tables described here. Nothing else negotiates
column names. If a column is not in this file it does not exist.

Three tables:

- ``ATTRIBUTES``  one row per PLP member, the construction input
- ``HOLDOUT``     one row per (member, event), the validation target
- ``ELICITATION`` one row per model call, the raw simulation output

Use :func:`validate` at every module boundary. It fails loudly on a missing
column, an unexpected column, a wrong dtype, or a value outside an allowed set,
which is the point: code that drifted from the contract should break at
integration rather than quietly at eval time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------
# column spec
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """One column of a table contract.

    ``dtype`` is a pandas dtype string. ``allowed`` constrains the value set for
    categoricals; ``None`` means unconstrained. ``minimum`` / ``maximum`` are
    inclusive bounds, checked only on non-null values.
    """

    dtype: str
    nullable: bool = False
    allowed: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    doc: str = ""


@dataclass(frozen=True)
class Schema:
    name: str
    columns: dict[str, Column]
    key: tuple[str, ...]
    doc: str = ""

    @property
    def names(self) -> list[str]:
        return list(self.columns)

    def empty(self) -> pd.DataFrame:
        """An empty frame with the right columns and dtypes: useful in tests."""
        return pd.DataFrame({n: pd.Series(dtype=c.dtype) for n, c in self.columns.items()})


class SchemaError(AssertionError):
    """Raised when a DataFrame violates its contract."""


def validate(df: pd.DataFrame, schema: Schema, *, allow_extra: bool = False) -> pd.DataFrame:
    """Check ``df`` against ``schema``. Returns ``df`` unchanged, or raises.

    Collects every problem before raising so one run surfaces the whole diff
    rather than one error at a time.
    """
    problems: list[str] = []

    missing = [n for n in schema.names if n not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")

    extra = [n for n in df.columns if n not in schema.columns]
    if extra and not allow_extra:
        problems.append(f"unexpected columns: {extra}")

    for name, col in schema.columns.items():
        if name not in df.columns:
            continue
        s = df[name]

        actual = str(s.dtype)
        if actual != col.dtype and not _dtype_compatible(actual, col.dtype):
            problems.append(f"{name}: dtype {actual!r}, expected {col.dtype!r}")

        n_null = int(s.isna().sum())
        if n_null and not col.nullable:
            problems.append(f"{name}: {n_null} null(s) in a non-nullable column")

        present = s.dropna()
        if col.allowed is not None and len(present):
            bad = sorted(set(present.unique()) - set(col.allowed))
            if bad:
                problems.append(f"{name}: values outside allowed set: {bad[:8]}")
        if col.minimum is not None and len(present) and present.min() < col.minimum:
            problems.append(f"{name}: min {present.min()} below {col.minimum}")
        if col.maximum is not None and len(present) and present.max() > col.maximum:
            problems.append(f"{name}: max {present.max()} above {col.maximum}")

    if schema.key and all(k in df.columns for k in schema.key):
        dupes = int(df.duplicated(subset=list(schema.key)).sum())
        if dupes:
            problems.append(f"{dupes} duplicate row(s) on key {schema.key}")

    if problems:
        raise SchemaError(
            f"{schema.name} failed validation ({len(problems)} problem(s)):\n  - "
            + "\n  - ".join(problems)
        )
    return df


def _dtype_compatible(actual: str, expected: str) -> bool:
    """Tolerate distinctions pandas blurs that the contract does not care about.

    Datetime *resolution* is one of these: pandas 2.x yields ``datetime64[s]``
    when broadcasting a scalar Timestamp and ``datetime64[ns]`` from
    ``to_datetime``, and both mean the same thing here.
    """
    if actual.startswith("datetime64[") and expected.startswith("datetime64["):
        return True
    equivalent = [
        {"int64", "Int64"},
        {"float64", "Float64"},
        {"bool", "boolean"},
        # pandas 3.0 turns on `future.infer_string` by default, so a column of
        # plain Python strings now reports "str" where 2.x reported "object".
        # The contract cares that it holds text, not which backend stores it.
        # Widening here rather than pinning pandas<3 or globally disabling the
        # future default: the new behaviour is where pandas is going.
        {"object", "string", "string[python]", "str"},
    ]
    return any({actual, expected} <= group for group in equivalent)


# --------------------------------------------------------------------------
# value sets
# --------------------------------------------------------------------------

#: Parties that count as PLP for the default population definition. Co-op
#: members take the Labour whip, so they are in.
PLP_PARTIES: tuple[str, ...] = ("Labour", "Labour (Co-op)")

#: Runner-up party is coarsened to this set; anything rarer becomes "Other".
#: A Reform-facing marginal and a Green-facing urban seat imply opposite answers
#: to the same item, so this distinction is load-bearing, not cosmetic.
RUNNER_UP_PARTIES: tuple[str, ...] = (
    "Conservative",
    "Reform UK",
    "Liberal Democrat",
    "Green Party",
    "Scottish National Party",
    "Plaid Cymru",
    "Independent",
    "Other",
)

#: The persona-construction ladder, ordered by how close each method gets to a
#: specific real individual.
#:
#: The brief asks for *personas modelling a group*, not simulations of named
#: people. That distinction is the whole design: a method that needs a named
#: individual's public record cannot be pointed at "undecided voters in Red Wall
#: seats", which is what the product actually has to do. So P0-P3 are synthetic
#, and deployable; P4/P5 require a real person and are reported as a ceiling.
#:
#:   P0      Generic stereotype. One shared dossier, no per-persona attributes.
#:           The published baseline; exists so the gain above it is measurable.
#:   P1      Quota persona. A SYNTHETIC attribute vector drawn so the panel's
#:           weighted margins match the population. No real MP behind it.
#:   P2      Archetype persona. A cluster's central tendency, again synthetic:
#:           the profile of a type, not of whoever happens to sit nearest.
#:   P3      P2 plus an LLM-written biography generated from that vector. The
#:           most creative method, and the one most able to invent detail the
#:           vector never contained -- which is why it is scored against P2 on
#:           the identical vector.
#:   P4      A real MP's attribute record, anonymised. Not deployable (needs a
#:           real person's record) but isolates what individual-level data buys.
#:   P5      P4 plus the real name. The contamination ceiling: whatever P5 beats
#:           P2 by is what the model already knew rather than what it inferred.
#:   RECALL  No persona at all. The leakage floor.
METHODS: tuple[str, ...] = ("P0", "P1", "P2", "P2C", "P3", "P3S", "P4", "P5", "RECALL")

#: Methods that could be pointed at an audience the model has never read about.
#: The headline claim rests on these; everything else is diagnostic.
DEPLOYABLE_METHODS: tuple[str, ...] = ("P0", "P1", "P2", "P2C", "P3")

#: P3S is a CONTROL, not a method: P3's biographies shuffled onto the wrong
#: personas. Same length, same style, same total prose -- only the match
#: between biography and vector is destroyed. P3 - P3S is therefore what
#: the biography's *relevance* buys, with prompt length held exactly fixed.
CONTROL_METHODS: tuple[str, ...] = ("P3S",)

#: Methods requiring a specific real individual's record. Reported, never sold.
CEILING_METHODS: tuple[str, ...] = ("P4", "P5", "RECALL")

#: Where a persona's attribute vector came from.
PERSONA_SOURCES: tuple[str, ...] = ("stereotype", "quota", "archetype", "real")
#: "PANEL" is used by the persona ladder, where the persona panel *is* the
#: frame: each method builds its own, so there is no separate frame to name.
FRAMES: tuple[str, ...] = ("F1", "F2", "FULL", "PANEL")
OPTION_ORDERS: tuple[str, ...] = ("forward", "reversed")
EVENT_TYPES: tuple[str, ...] = ("nomination", "loyalty", "rebellion", "signature", "free_vote")

#: Declaration timing in the 2026 leadership contest. Three buckets, not a
#: continuous variable: the only source is a live tally page captured by the
#: Wayback Machine four times, so every date is "nominated on or before this
#: crawl". `none` is derived as roster minus nominators, and is the informative
#: cell: 379 of 380 nominators backed the same candidate, so *whether* an MP
#: nominated at all carries far more signal than *whom* they nominated.
NOMINATION_BUCKETS: tuple[str, ...] = ("day1", "mid", "late", "none")

#: The primary holdout outcome. Binary rather than four-way because the timing
#: buckets are crawl artefacts, while "moved with the herd immediately" versus
#: "did not" is a real behavioural distinction with a 79/21 split.
BANDWAGON_OUTCOMES: tuple[str, ...] = ("day1_bandwagon", "held_back")

#: Public position during the pressure on Starmer that preceded his 22 June 2026
#: resignation. Unlike the nomination this was a genuine choice under
#: uncertainty: the outcome was not yet known, so it is expected to carry the
#: variance the coronation lacks. `silent` is a real category, not missing data:
#: declining to take a public position under pressure is itself a decision.
LOYALTY_OUTCOMES: tuple[str, ...] = ("backed_starmer", "called_for_exit", "silent")

#: The subset who took any public position at all. `silent` is deliberately a
#: first-class outcome in LOYALTY_OUTCOMES rather than missing data: declining
#: to speak while your leader is under public pressure is a behaviour. This
#: binary is the secondary view over the ~200 who did speak.
LOYALTY_BINARY_OUTCOMES: tuple[str, ...] = ("backed_starmer", "called_for_exit")


# --------------------------------------------------------------------------
# ATTRIBUTES
# --------------------------------------------------------------------------

ATTRIBUTES = Schema(
    name="ATTRIBUTES",
    key=("member_id",),
    doc="One row per PLP member. The single construction input for every frame and dossier.",
    columns={
        # identity
        "member_id": Column("int64", doc="Parliament Members API id"),
        "name": Column("object", doc="Display name"),
        "constituency": Column("object"),
        "party_name": Column("object", allowed=PLP_PARTIES),
        # electoral
        "majority_pct": Column("float64", minimum=0.0, maximum=100.0,
                               doc="2024 majority as % of valid votes cast"),
        "vote_share": Column("float64", minimum=0.0, maximum=100.0),
        "runner_up_party": Column("object", allowed=RUNNER_UP_PARTIES),
        "first_elected": Column("datetime64[ns]"),
        "is_2024_intake": Column("bool"),
        # position
        "is_payroll": Column("bool", doc="Minister, whip, or PPS"),
        "role": Column("object", nullable=True),
        "committee_count": Column("int64", minimum=0),
        # behavioural
        "rebellion_rate": Column("float64", nullable=True, minimum=0.0, maximum=1.0),
        "rebellions_welfare": Column("Int64", nullable=True, minimum=0),
        "rebellions_wfa": Column("Int64", nullable=True, minimum=0),
        "did_nominate": Column("boolean", nullable=True,
                               doc="Did this MP nominate anyone in the 2026 leadership contest. "
                                   "Separate from nomination_day because NaN there would otherwise "
                                   "conflate 'declined to nominate' with 'no source found': "
                                   "opposite behaviours. Declining to nominate a near-unanimous "
                                   "winner is the informative low-base-rate signal, so it needs "
                                   "its own field. Null means genuinely unknown."),
        "nomination_bucket": Column("object", nullable=True, allowed=NOMINATION_BUCKETS,
                                    doc="Coarse declaration timing. Only three crawl-derived "
                                        "buckets exist in the source, so this is deliberately "
                                        "ordinal rather than continuous."),
        "nomination_day": Column("float64", nullable=True, minimum=0.0,
                                 doc="Days from nominations opening to first appearance in an "
                                     "archived tally. CENSORED: takes only the values 0/4/7, and "
                                     "each is 'nominated on or before', not a declaration date. "
                                     "Do not fit a continuous hazard model to this."),
        # constituency context (held out of stratification: used to measure frame error)
        "deprivation_score": Column("float64", nullable=True),
        "median_age": Column("float64", nullable=True),
        "degree_share": Column("float64", nullable=True, minimum=0.0, maximum=100.0),
        # activity
        "speech_count": Column("Int64", nullable=True, minimum=0),
    },
)

#: Columns recording behaviour that happened AFTER ``config.cutoff_date``.
#:
#: These are outcomes, not construction inputs, and using any of them to build a
#: frame or a dossier is leakage. The original design brief listed
#: ``nomination_day`` as a construction attribute *and* made leadership
#: nomination the primary validation target, which is circular: stratify on the
#: outcome and the weighted panel reproduces the outcome distribution by
#: construction, making distribution accuracy trivially perfect and meaningless.
#: Kept in ATTRIBUTES because it is the master MP table, but fenced off here and
#: asserted disjoint from SEG_VARS in tests.
POST_CUTOFF_VARS: tuple[str, ...] = (
    "did_nominate",
    "nomination_bucket",
    "nomination_day",
    "rebellions_welfare",
    "rebellions_wfa",
)

#: Variables the analyst-specified strata are built from (F1) and that F2's
#: Gower distance sees. Frame error is measured on what is NOT in here.
#:
#: ``rebellion_rate`` must be computed from divisions strictly BEFORE the cutoff.
#: The specific welfare and winter-fuel rebellions are holdout events and live in
#: POST_CUTOFF_VARS above.
SEG_VARS: tuple[str, ...] = (
    "majority_pct",
    "runner_up_party",
    "is_payroll",
    "is_2024_intake",
    "rebellion_rate",
)

#: Deliberately excluded from frame construction so they can serve as an honest
#: out-of-sample test of whether the weighted panel reproduces the population.
HELDOUT_VARS: tuple[str, ...] = (
    "deprivation_score",
    "median_age",
    "degree_share",
    "speech_count",
)

#: Which SEG_VARS are categorical, for Gower and for stratum construction.
CATEGORICAL_SEG_VARS: tuple[str, ...] = ("runner_up_party", "is_payroll", "is_2024_intake")


# --------------------------------------------------------------------------
# HOLDOUT
# --------------------------------------------------------------------------

HOLDOUT = Schema(
    name="HOLDOUT",
    key=("member_id", "event_id"),
    doc=(
        "Observed post-cutoff behaviour, one row per (member, event). Every event "
        "must have real variance: whipped divisions are excluded by construction, "
        "because a null model that predicts 'follows the whip' scores >95% and no "
        "method can distinguish itself against it."
    ),
    columns={
        "member_id": Column("int64"),
        "event_id": Column("object", doc="Stable slug, e.g. 'leadership_nomination_2026'"),
        "event_type": Column("object", allowed=EVENT_TYPES),
        "outcome": Column("object", doc="Observed label, one of the event's option set"),
        "outcome_index": Column("int64", minimum=0, doc="Index into the event's option list"),
        "n_options": Column("int64", minimum=2),
        "base_rate": Column("float64", minimum=0.0, maximum=1.0,
                            doc="Modal-class share for this event. Every accuracy figure is "
                                "reported against this, never against zero."),
        "observed_at": Column("datetime64[ns]"),
    },
)


# --------------------------------------------------------------------------
# ELICITATION
# --------------------------------------------------------------------------

ELICITATION = Schema(
    name="ELICITATION",
    key=("method", "frame", "persona_id", "item_id", "option_order", "draw_index"),
    doc=(
        "One row per model call. `probs` is the renormalised distribution over the "
        "item's options; `captured_mass` is how much probability the option tokens "
        "actually accounted for before renormalisation. A low captured_mass means "
        "the model answered something other than the item: that is a finding, not "
        "a rounding error, so it is stored rather than discarded."
    ),
    columns={
        "method": Column("object", allowed=METHODS),
        "frame": Column("object", allowed=FRAMES),
        "persona_id": Column("int64", doc="member_id of the MP this persona represents"),
        "item_id": Column("object"),
        "option_order": Column("object", allowed=OPTION_ORDERS),
        "draw_index": Column("int64", minimum=0,
                             doc="0 for logprob elicitation; 0..n-1 for sampled dispersion draws"),
        "model": Column("object"),
        "prompt_version": Column("object"),
        "temperature": Column("float64", nullable=True,
                              doc="Null for logprob calls, set for dispersion sampling"),
        "probs": Column("object", doc="list[float] aligned to the item's forward option order"),
        "top_option": Column("object"),
        "top_prob": Column("float64", minimum=0.0, maximum=1.0),
        "captured_mass": Column("float64", minimum=0.0, maximum=1.0),
        "cached": Column("bool"),
        "latency_ms": Column("float64", nullable=True, minimum=0.0),
    },
)

#: Below this, treat the call as suspect and surface it in the run report rather
#: than silently renormalising a distribution built from almost no probability.
MIN_CAPTURED_MASS = 0.30

# --------------------------------------------------------------------------
# FRAME
# --------------------------------------------------------------------------

FRAME = Schema(
    name="FRAME",
    key=("frame", "member_id"),
    doc=(
        "The selected panel. One row per persona, with the population weight it "
        "carries. Both F1 and F2 select *real* members: F1 samples within stratum, "
        "F2 medoids are members, which is what makes individual accuracy scoring "
        "well defined: every persona has its own observed outcome to be scored "
        "against, while the weighted panel is compared to the true population marginal."
    ),
    columns={
        "frame": Column("object", allowed=FRAMES),
        "member_id": Column("int64"),
        "weight": Column("float64", minimum=0.0,
                         doc="Population weight. Weights sum to the population size, "
                             "not to 1, so weighted counts read directly as MPs."),
        "stratum": Column("object", nullable=True,
                          doc="F1 stratum label; null for F2"),
    },
)

# --------------------------------------------------------------------------
# PERSONA
# --------------------------------------------------------------------------

PERSONA = Schema(
    name="PERSONA",
    key=("method", "persona_id"),
    doc=(
        "The panel actually put to the model. One row per persona per method.\n\n"
        "This is the table that separates persona modelling from person "
        "simulation. For P1/P2/P3 the attribute values are SYNTHESISED and "
        "``member_id`` is null: there is no real MP behind the row, so nothing "
        "the model produces can be retrieval. For P4/P5 ``member_id`` points at "
        "a real member, which is exactly why those two are a ceiling rather "
        "than a method.\n\n"
        "Attribute columns deliberately mirror ATTRIBUTES so one renderer works "
        "for both synthetic and real personas."
    ),
    columns={
        "method": Column("object", allowed=METHODS),
        "persona_id": Column("int64", doc="Stable within (method); not a member_id"),
        "source": Column("object", allowed=PERSONA_SOURCES),
        "member_id": Column("Int64", nullable=True,
                            doc="Null for synthetic personas. Set only for P4/P5, "
                                "and its presence is what makes those non-deployable."),
        "weight": Column("float64", minimum=0.0,
                         doc="How many real members this persona stands for. Weights "
                             "sum to the population size."),
        "cell": Column("object", nullable=True,
                       doc="Quota cell (P1) or cluster label (P2/P3); null for P0"),
        # mirrored attribute block
        "majority_pct": Column("float64", nullable=True, minimum=0.0, maximum=100.0),
        "vote_share": Column("float64", nullable=True, minimum=0.0, maximum=100.0),
        "runner_up_party": Column("object", nullable=True, allowed=RUNNER_UP_PARTIES),
        "is_2024_intake": Column("boolean", nullable=True),
        "is_payroll": Column("boolean", nullable=True),
        "role": Column("object", nullable=True),
        "committee_count": Column("Int64", nullable=True, minimum=0),
        "rebellion_rate": Column("float64", nullable=True, minimum=0.0, maximum=1.0),
        "speech_count": Column("Int64", nullable=True, minimum=0),
        # situational context (P2C only) -- what the political weather is, and
        # whether this kind of MP is personally exposed to it. Every other
        # attribute describes the persona; these describe the world it is in.
        # Their absence is the leading explanation for why no persona will say
        # a leader should go: nothing told it the leader was costing it anything.
        "seat_at_risk_share": Column("float64", nullable=True, minimum=0.0, maximum=1.0,
                                     doc="Share of this archetype's members projected to "
                                         "lose their seat (Electoral Calculus MRP, Jan 2026)"),
        "projected_winner": Column("object", nullable=True,
                                   doc="Modal projected winner where members are at risk"),
        "projected_labour_share": Column("float64", nullable=True, minimum=0.0, maximum=100.0),
        "national_context": Column("object", nullable=True,
                                   doc="Shared pre-cutoff polling summary, identical across personas"),
        # P3 only
        "biography": Column("object", nullable=True,
                            doc="LLM-written prose for P3. Null everywhere else."),
        # P4/P5 only
        "name": Column("object", nullable=True),
        "constituency": Column("object", nullable=True),
    },
)

ALL_SCHEMAS: dict[str, Schema] = {
    s.name: s for s in (ATTRIBUTES, HOLDOUT, ELICITATION, FRAME, PERSONA)
}
