"""Synthetic tables conforming to the contracts in ``plp_sim.schemas``.

Waves 1-3 are built and tested against these, so no module needs real data or an
API key to be developed or verified. The generator is seeded and deterministic.

Deliberate properties, because modules downstream depend on them:

* correlated structure: ``nomination_day`` depends on ``majority_pct`` and
  ``is_payroll``, so a frame that ignores those margins produces measurably
  worse frame error. A fixture of pure noise would let a broken frame look fine.
* realistic missingness: ``nomination_day`` is ~15% null, matching what a
  hand-compiled press-tracker dataset actually looks like.
* a genuinely skewed target: the holdout outcome is not uniform, so
  ``base_rate`` is meaningfully above 1/n_options and accuracy-vs-chance is a
  real test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plp_sim import schemas


def make_attributes(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """A synthetic attribute table of ``n`` members."""
    rng = np.random.default_rng(seed)

    majority = np.clip(rng.gamma(shape=2.2, scale=7.0, size=n), 0.1, 70.0)
    is_payroll = rng.random(n) < 0.30
    is_2024 = rng.random(n) < 0.55
    runner_up = rng.choice(
        schemas.RUNNER_UP_PARTIES,
        size=n,
        p=[0.34, 0.30, 0.14, 0.08, 0.05, 0.03, 0.03, 0.03],
    )

    # Payroll MPs declare early; MPs in marginal seats hedge and declare late.
    base_day = 6.0 - 3.0 * is_payroll + 0.16 * (40.0 - np.clip(majority, 0, 40))
    nomination_day = np.clip(base_day + rng.normal(0, 2.4, n), 0.0, 28.0)
    nomination_day[rng.random(n) < 0.15] = np.nan  # unsourced declarations

    # Censor to the three crawl-derived buckets the real source actually supports
    # (0 / 4 / 7 days), plus a `none` category for MPs who declined to nominate.
    # Mirroring the real skew matters: ~79% day-1, and the non-nominators are the
    # informative low-base-rate cell.
    did_nominate = rng.random(n) >= 0.06
    nomination_day = np.where(
        ~did_nominate, np.nan,
        np.select(
            [nomination_day <= 5.0, nomination_day <= 9.0],
            [0.0, 4.0], default=7.0,
        ),
    )
    bucket = np.where(
        ~did_nominate, "none",
        np.select([nomination_day == 0.0, nomination_day == 4.0],
                  ["day1", "mid"], default="late"),
    )
    bucket = np.where(pd.isna(nomination_day) & did_nominate, None, bucket)

    rebellion_rate = np.clip(
        0.035 + 0.0016 * majority - 0.030 * is_payroll + rng.normal(0, 0.018, n), 0.0, 1.0
    )

    first_elected = pd.to_datetime(
        np.where(is_2024, "2024-07-04", "2019-12-12")
    ) + pd.to_timedelta(rng.integers(0, 900, n) * ~is_2024, unit="D")

    df = pd.DataFrame(
        {
            "member_id": np.arange(100_000, 100_000 + n, dtype="int64"),
            "name": [f"Member {i:03d}" for i in range(n)],
            "constituency": [f"Constituency {i:03d}" for i in range(n)],
            "party_name": rng.choice(schemas.PLP_PARTIES, size=n, p=[0.89, 0.11]),
            "majority_pct": majority,
            "vote_share": np.clip(32.0 + 0.55 * majority + rng.normal(0, 4.0, n), 0.0, 100.0),
            "runner_up_party": runner_up,
            "first_elected": first_elected,
            "is_2024_intake": is_2024,
            "is_payroll": is_payroll,
            "role": np.where(is_payroll, "Minister", None),
            "committee_count": rng.integers(0, 3, n).astype("int64"),
            "rebellion_rate": rebellion_rate,
            "rebellions_welfare": pd.array(rng.integers(0, 3, n), dtype="Int64"),
            "rebellions_wfa": pd.array(rng.integers(0, 2, n), dtype="Int64"),
            "did_nominate": pd.array(did_nominate, dtype="boolean"),
            "nomination_bucket": bucket,
            "nomination_day": nomination_day,
            # held out of stratification, correlated with it: this is what makes
            # the frame-error measurement non-trivial
            "deprivation_score": 22.0 + 0.30 * majority + rng.normal(0, 5.0, n),
            "median_age": 44.0 - 0.10 * majority + rng.normal(0, 4.0, n),
            "degree_share": np.clip(31.0 + 0.22 * majority + rng.normal(0, 7.0, n), 0.0, 100.0),
            "speech_count": pd.array(
                np.clip(rng.poisson(28, n) + (~is_payroll) * 14, 0, None), dtype="Int64"
            ),
        }
    )
    return schemas.validate(df, schemas.ATTRIBUTES)


def make_holdout(attributes: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A synthetic holdout table covering every member in ``attributes``."""
    rng = np.random.default_rng(seed + 1)
    options = ("early", "late", "abstained")

    # Outcome depends on the same drivers as nomination_day, so a frame that
    # reproduces those margins should score better than one that does not.
    day = attributes["nomination_day"].fillna(attributes["nomination_day"].median()).to_numpy()
    score = (day - day.mean()) / (day.std() or 1.0) + rng.normal(0, 0.7, len(attributes))
    idx = np.digitize(score, bins=[-0.45, 0.75])  # skewed, not uniform

    df = pd.DataFrame(
        {
            "member_id": attributes["member_id"].to_numpy(),
            "event_id": "leadership_nomination_2026",
            "event_type": "nomination",
            "outcome": [options[i] for i in idx],
            "outcome_index": idx.astype("int64"),
            "n_options": np.int64(len(options)),
            "observed_at": pd.Timestamp("2026-05-15"),
        }
    )
    modal_share = df["outcome"].value_counts(normalize=True).max()
    df["base_rate"] = float(modal_share)
    return schemas.validate(df, schemas.HOLDOUT)


def make_elicitation(
    attributes: pd.DataFrame, *, method: str = "P2", frame: str = "F2", seed: int = 0
) -> pd.DataFrame:
    """A synthetic elicitation result set, one call per member for one item."""
    rng = np.random.default_rng(seed + 2)
    n = len(attributes)
    raw = rng.dirichlet(np.array([2.4, 3.1, 1.5]), size=n)
    top = raw.argmax(axis=1)
    labels = ("early", "late", "abstained")

    df = pd.DataFrame(
        {
            "method": method,
            "frame": frame,
            "persona_id": attributes["member_id"].to_numpy(),
            "item_id": "leadership_nomination_2026",
            "option_order": "forward",
            "draw_index": np.zeros(n, dtype="int64"),
            "model": "fixture-model",
            "prompt_version": "v1",
            "temperature": pd.array([None] * n, dtype="float64"),
            "probs": [list(map(float, row)) for row in raw],
            "top_option": [labels[i] for i in top],
            "top_prob": raw.max(axis=1),
            "captured_mass": np.clip(rng.beta(9, 1.6, n), 0.0, 1.0),
            "cached": np.zeros(n, dtype=bool),
            "latency_ms": rng.uniform(180, 900, n),
        }
    )
    return schemas.validate(df, schemas.ELICITATION)
