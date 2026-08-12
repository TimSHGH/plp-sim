"""Observed post-cutoff behaviour: one row per (member, event).

Built before anything else touches the data, so no later decision can be tuned
against the answers.

Score only on events with genuine variance. A whipped division is useless as a
target because a model that always predicts "follows the whip" scores above
95%. Every event is therefore reported against its own ``base_rate``.

The four events, and what each is worth:

``starmer_loyalty`` is primary. Backed the leader, called for them to go, or
said nothing. The only event with balanced variance, and a real choice made
before the outcome was known.

``leadership_nomination`` is weak. The contest was a coronation, 379 of 380
nominating MPs backing the same candidate, so *whom* they nominated carries
almost no information. What survives is whether and how fast they moved.

``post_cutoff_rebellion`` pools across divisions. No single one is usable: of
107 with member detail, only 15 show any dissent and none exceeds 10%. So it
becomes "rebelled at least once".

Welfare and Winter Fuel rebellions are **dropped**. All five divisions predate
the cutoff, which means they are already inside ``rebellion_rate``, a
construction variable. Scoring on them would test the model against its own
inputs. They look like ideal targets until you check the dates.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from plp_sim import attributes as attrs_mod
from plp_sim import schemas
from plp_sim.config import Settings, get_settings

#: Earliest date on which any validation event became observable. The cutoff
#: must strictly precede this or construction data leaks into the targets.
#: 7 May 2026 is when organised public pressure on the leadership began; the
#: first named Westminster MP break was 9 May.
EARLIEST_EVENT_DATE = dt.date(2026, 5, 7)

EVENT_DATES: dict[str, dt.date] = {
    "starmer_loyalty": dt.date(2026, 5, 7),
    "leadership_nomination": dt.date(2026, 7, 9),
    "post_cutoff_rebellion": dt.date(2026, 4, 14),
}


class CutoffViolation(AssertionError):
    """Raised when a validation event predates the construction cutoff."""


def assert_cutoff_precedes_events(settings: Settings) -> None:
    """The whole design rests on this and nothing else was checking it.

    ``post_cutoff_rebellion`` starts 14 April 2026, only two weeks after the
    1 April cutoff: comfortable, but not so comfortable that a future change
    to either date can be assumed safe.
    """
    bad = {k: v for k, v in EVENT_DATES.items() if v <= settings.cutoff_date}
    if bad:
        raise CutoffViolation(
            f"cutoff_date {settings.cutoff_date} does not strictly precede "
            f"event(s): {bad}. Construction data would leak into the targets."
        )


# --------------------------------------------------------------------------
# event builders
# --------------------------------------------------------------------------


def _with_base_rate(df: pd.DataFrame, event_id: str, event_type: str, when: dt.date) -> pd.DataFrame:
    """Attach outcome_index, n_options and the event's modal-class share.

    ``base_rate`` is the modal share rather than 1/n_options: the honest thing
    to beat is the best trivial predictor, not a uniform guess.
    """
    labels = sorted(df["outcome"].unique())
    return pd.DataFrame(
        {
            "member_id": df["member_id"].astype("int64").to_numpy(),
            "event_id": event_id,
            "event_type": event_type,
            "outcome": df["outcome"].to_numpy(),
            "outcome_index": df["outcome"].map({v: i for i, v in enumerate(labels)}).astype("int64").to_numpy(),
            "n_options": len(labels),
            "base_rate": float(df["outcome"].value_counts(normalize=True).max()),
            "observed_at": pd.Timestamp(when),
        }
    )


def build_starmer_loyalty(roster: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict[str, object]]:
    """PRIMARY. Public position during the pressure before the resignation.

    Three-way over the whole PLP: ``backed_starmer`` / ``called_for_exit`` /
    ``silent``.

    Silence is kept as a class rather than dropped. Declining to speak while your
    leader is under public pressure is a behaviour, not missing data, and it is the
    largest class. Keeping it takes the target from 203 MPs to all 405, and turns
    the question into "would this MP speak at all, and if so which way".

    The binary over the 203 who did speak is available as
    ``starmer_loyalty_binary``.

    Trackers over-report MPs calling for a leader to go, so ``silent`` is
    over-populated relative to private opinion. See
    ``data/manual/STARMER_LOYALTY_NOTES.md``.
    """
    raw = pd.read_csv(settings.data_manual / "starmer_loyalty.csv")

    def _key(s: pd.Series) -> pd.Series:
        return attrs_mod._strip_honorifics(s).str.casefold().str.strip()

    left = roster.assign(
        _name=_key(roster["name"]),
        _con=attrs_mod._normalize_constituency_name(roster["constituency"]),
    )
    right = raw.assign(_name=_key(raw["mp_name"]))
    has_con = right["constituency"].notna()
    right.loc[has_con, "_con"] = attrs_mod._normalize_constituency_name(
        right.loc[has_con, "constituency"]
    )

    # Two-stage join, and the split matters. Only the 101 `called_for_exit`
    # rows come from a tracker that lists constituencies; 96 of the 110
    # `backed_starmer` rows are from a letter-signatory list that is a bare run
    # of names. Requiring name+constituency therefore drops almost every
    # loyalist and silently converts a balanced 48/52 target into an 88/12 one
    # -- a far worse target, arrived at invisibly. Stage 2 exists for them.
    stage1 = left.merge(
        right.loc[has_con, ["_name", "_con", "position"]], on=["_name", "_con"], how="inner"
    )
    remaining = right[~right["_name"].isin(stage1["_name"])]

    # Name-only matching is weaker, so refuse it wherever the name is not
    # unique on either side rather than guessing which MP was meant.
    dup_left = set(left.loc[left["_name"].duplicated(keep=False), "_name"])
    dup_right = set(remaining.loc[remaining["_name"].duplicated(keep=False), "_name"])
    ambiguous = dup_left | dup_right
    safe = remaining[~remaining["_name"].isin(ambiguous)]
    stage2 = left.merge(safe[["_name", "position"]], on="_name", how="inner")

    joined = pd.concat([stage1, stage2], ignore_index=True).drop_duplicates("member_id")

    matched_names = set(joined["_name"])
    unmatched = right[~right["_name"].isin(matched_names)]
    report = {
        "source_rows": len(raw),
        "matched": len(joined),
        "matched_on_name_and_constituency": len(stage1),
        "matched_on_name_only": len(stage2),
        "ambiguous_names_refused": sorted(ambiguous),
        "unmatched": len(unmatched),
        "unmatched_names": sorted(unmatched["mp_name"].tolist())[:20],
        "unclassified_plp": len(roster) - len(joined),
    }
    classified = joined.rename(columns={"position": "outcome"})[["member_id", "outcome"]]

    # Three-way over the whole PLP: everyone not in a tracker was silent.
    silent_ids = set(roster["member_id"]) - set(classified["member_id"])
    three_way = pd.concat(
        [
            classified,
            pd.DataFrame({"member_id": sorted(silent_ids), "outcome": "silent"}),
        ],
        ignore_index=True,
    )
    report["silent"] = len(silent_ids)

    out = pd.concat(
        [
            _with_base_rate(
                three_way, "starmer_loyalty", "loyalty", EVENT_DATES["starmer_loyalty"]
            ),
            # Binary over just those who spoke, kept as a secondary view: it is
            # the cleaner side-picking contrast, at the cost of half the PLP.
            _with_base_rate(
                classified,
                "starmer_loyalty_binary", "loyalty", EVENT_DATES["starmer_loyalty"],
            ),
        ],
        ignore_index=True,
    )
    return out, report


def build_leadership_nomination(roster: pd.DataFrame, attributes: pd.DataFrame) -> pd.DataFrame:
    """SECONDARY. Day-one bandwagon versus held back.

    Binary rather than the brief's four-way early/mid/late/abstained, because
    the three timing buckets are Wayback crawl artefacts rather than real
    declaration dates. "Moved with the herd within hours" versus "did not" is
    a distinction the source can actually support.
    """
    a = attributes.set_index("member_id")
    bucket = a.loc[roster["member_id"], "nomination_bucket"]
    outcome = bucket.map(lambda b: "day1_bandwagon" if b == "day1" else "held_back")
    df = pd.DataFrame({"member_id": roster["member_id"].to_numpy(), "outcome": outcome.to_numpy()})
    df = df[pd.notna(df["outcome"])]
    return _with_base_rate(df, "leadership_nomination", "nomination", EVENT_DATES["leadership_nomination"])


def build_post_cutoff_rebellion(roster: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Pooled: did this MP rebel on ANY post-cutoff division.

    Pooled deliberately. Taken one at a time, no post-cutoff division is a
    usable target: 15 of 107 show any dissent and the largest is 7%, so a
    null model scores 93-100% on every one. Pooled, the outcome is ~9% and
    behaves sensibly: payroll MPs rebel at 0.0%, backbenchers at 11.5%, the
    2024 intake at 4.6% against 15.2% for the pre-2024 cohort.

    The positive class is small (36 members), so individual accuracy here will
    be noisy. Its value is the cross-tab gradient, not the topline.
    """
    votes = pd.read_parquet(settings.data_interim / "division_votes_postcutoff.parquet")
    votes = votes[votes["member_id"].isin(set(roster["member_id"]))]
    majority = votes.groupby("division_id")["vote"].agg(lambda s: s.value_counts().idxmax())
    rebelled = (
        votes.assign(_rebel=votes["vote"].ne(votes["division_id"].map(majority)))
        .groupby("member_id")["_rebel"].any()
    )
    present = roster[roster["member_id"].isin(rebelled.index)]
    df = pd.DataFrame(
        {
            "member_id": present["member_id"].to_numpy(),
            "outcome": present["member_id"].map(rebelled).map(
                {True: "rebelled", False: "loyal"}
            ).to_numpy(),
        }
    )
    return _with_base_rate(df, "post_cutoff_rebellion", "rebellion", EVENT_DATES["post_cutoff_rebellion"])


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def build_holdout(
    attributes: pd.DataFrame | None = None, settings: Settings | None = None
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Assemble every validation event. Returns (holdout, report)."""
    settings = settings or get_settings()
    assert_cutoff_precedes_events(settings)

    if attributes is None:
        attributes = pd.read_parquet(settings.data_processed / "attributes.parquet")
    roster = attributes[["member_id", "name", "constituency"]].copy()

    loyalty, loyalty_report = build_starmer_loyalty(roster, settings)
    nomination = build_leadership_nomination(roster, attributes)
    rebellion = build_post_cutoff_rebellion(roster, settings)

    holdout = pd.concat([loyalty, nomination, rebellion], ignore_index=True)
    holdout["observed_at"] = pd.to_datetime(holdout["observed_at"])
    schemas.validate(holdout, schemas.HOLDOUT)

    report: dict[str, object] = {
        "cutoff_date": settings.cutoff_date.isoformat(),
        "earliest_event": min(EVENT_DATES.values()).isoformat(),
        "loyalty_join": loyalty_report,
        "events": {
            e: {
                "n": int((holdout.event_id == e).sum()),
                "base_rate": float(holdout.loc[holdout.event_id == e, "base_rate"].iloc[0]),
                "outcomes": holdout.loc[holdout.event_id == e, "outcome"]
                .value_counts().to_dict(),
            }
            for e in holdout.event_id.unique()
        },
        "dropped_events": {
            "welfare_wfa_rebellions": (
                "All 5 welfare/WFA divisions are pre-cutoff (2024-09 to 2025-11) and are "
                "already inside rebellion_rate, a construction variable. Using them as "
                "targets would score the model against its own inputs."
            )
        },
    }
    return holdout, report


def write_holdout(settings: Settings | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    settings = settings or get_settings()
    settings.ensure_dirs()
    holdout, report = build_holdout(settings=settings)
    holdout.to_parquet(settings.data_processed / "holdout.parquet", index=False)
    return holdout, report
