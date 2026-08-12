"""Scoring.

Every accuracy figure is reported against the event's own ``base_rate``, never
against zero or uniform chance. The main holdout outcome is roughly 79/21
skewed, so raw accuracy is misleading: a model that always guesses the modal
class scores 79% while doing nothing. :func:`balanced_accuracy` and
:func:`matthews_corrcoef` report that correctly as no better than chance, which
is why they are the headline per-event numbers.

:func:`one_minus_tvd` is the distribution measure the project quotes throughout.

:func:`leakage_gain` is signed and never clamped. A method scoring below the
no-persona control is retrieval wearing a costume, and that is a result worth
reporting rather than a bug to hide.

Grouped functions take a ``group`` keyword so callers can re-aggregate without
touching the metric logic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as _balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef as _matthews_corrcoef_score

from plp_sim import frames, schemas
from plp_sim.config import get_settings

#: Default grouping granularity for every per-event metric in this module.
DEFAULT_GROUP: tuple[str, ...] = ("method", "frame", "event_id")

#: Attributes the cross-tab logistic model is fit on. Fixed rather than a
#: parameter: these are the two gradients the design brief calls out
#: ("a method that hits the topline with the wrong gradient is worse than
#: useless"), and letting callers vary them would make results across runs
#: incomparable.
CROSSTAB_PREDICTORS: tuple[str, ...] = ("majority_pct", "is_2024_intake")

#: Human test-retest Shannon entropy, in bits, for a comparable political
#: attitude item with ~3-4 forced-choice options.
#:
#: ASSUMPTION, not a measured constant: back-computed from the published
#: political-survey re-interview literature (e.g. the Achen 1975 "response
#: instability" tradition and ANES panel re-interview studies), which report
#: test-retest agreement in roughly the 65-80% range for items of this shape.
#: Entropy of a distribution whose modal share matches the middle of that
#: band (~72%) over 4 options is close to this value. Treat this as a
#: documented modelling choice to be revisited if a project-specific
#: benchmark becomes available, not as ground truth.
HUMAN_TEST_RETEST_ENTROPY_BITS = 0.80


# --------------------------------------------------------------------------
# joining elicitation to holdout truth
# --------------------------------------------------------------------------


def score_calls(
    elicitation: pd.DataFrame, holdout: pd.DataFrame, *, draw_index: int = 0
) -> pd.DataFrame:
    """Join elicitation predictions to their holdout ground truth.

    Inner join on ``persona_id == member_id`` and ``item_id == event_id``,
    restricted to ``draw_index`` (the logprob elicitation call, not a
    dispersion draw) by default. Standalone instrument items with no matching
    holdout event (e.g. ``polling_threshold``) drop out of this join by
    construction -- they have no ground truth to be scored against, which is
    correct, not a bug.
    """
    calls = elicitation[elicitation["draw_index"] == draw_index]
    return calls.merge(
        holdout,
        left_on=["persona_id", "item_id"],
        right_on=["member_id", "event_id"],
        how="inner",
        validate="many_to_one",
    )


# --------------------------------------------------------------------------
# individual accuracy: balanced accuracy, MCC
# --------------------------------------------------------------------------


def balanced_accuracy(
    scored: pd.DataFrame, *, group: Sequence[str] = DEFAULT_GROUP
) -> pd.DataFrame:
    """Balanced accuracy per ``group``, alongside raw accuracy and the
    event's own ``base_rate`` so the raw number is never read in isolation.

    Balanced accuracy (mean per-class recall) is the headline, not raw
    accuracy: on a ~79/21 skew, a constant predictor that always guesses the
    modal class scores ~79% raw accuracy and ~50% balanced accuracy -- only
    the second number correctly reports that as no better than chance.
    """
    group = list(group)

    def _one(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "balanced_accuracy": _balanced_accuracy_score(g["outcome"], g["top_option"]),
                "accuracy": float((g["outcome"] == g["top_option"]).mean()),
                "base_rate": float(g["base_rate"].iloc[0]),
                "n": len(g),
            }
        )

    return scored.groupby(group, dropna=False).apply(_one, include_groups=False).reset_index()


def matthews_corrcoef(
    scored: pd.DataFrame, *, group: Sequence[str] = DEFAULT_GROUP
) -> pd.DataFrame:
    """Matthews correlation coefficient per ``group``, alongside ``base_rate``.

    MCC is ~0 for a predictor uncorrelated with the truth regardless of class
    skew, which is what makes it a useful second headline figure next to
    balanced accuracy: the two can disagree (e.g. a method that gets the
    marginal skew right by luck but the per-member calls wrong) and both are
    worth reporting.
    """
    group = list(group)

    def _one(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "matthews_corrcoef": _matthews_corrcoef_score(g["outcome"], g["top_option"]),
                "base_rate": float(g["base_rate"].iloc[0]),
                "n": len(g),
            }
        )

    return scored.groupby(group, dropna=False).apply(_one, include_groups=False).reset_index()


# --------------------------------------------------------------------------
# calibration: Brier score, reliability curve
# --------------------------------------------------------------------------


def brier_score(scored: pd.DataFrame, *, group: Sequence[str] = DEFAULT_GROUP) -> pd.DataFrame:
    """Multiclass Brier score per ``group``: mean over calls of
    ``sum((probs - onehot(outcome))**2)``, the standard (unnormalised,
    Brier 1950) generalisation of the binary Brier score to more than two
    options. Zero is a perfect, fully-confident-and-correct call; the
    maximum is 2.0.
    """
    group = list(group)

    def _row_brier(row: pd.Series) -> float:
        probs = np.asarray(row["probs"], dtype=float)
        onehot = np.zeros_like(probs)
        onehot[int(row["outcome_index"])] = 1.0
        return float(np.sum((probs - onehot) ** 2))

    with_brier = scored.assign(_brier=scored.apply(_row_brier, axis=1))
    return (
        with_brier.groupby(group, dropna=False)
        .agg(brier_score=("_brier", "mean"), base_rate=("base_rate", "first"), n=("_brier", "size"))
        .reset_index()
    )



def one_minus_tvd(simulated: np.ndarray, observed: np.ndarray) -> float:
    """Agreement between two distributions over the same options, 1.0 at a
    perfect match.

    A one-liner, but it was written out by hand in six places and is the number
    every headline in this project is quoted in. One definition means one place
    to be wrong, and one place to check.
    """
    sim = np.asarray(simulated, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if sim.shape != obs.shape:
        raise ValueError(f"shape mismatch: simulated {sim.shape} vs observed {obs.shape}")
    return float(1.0 - 0.5 * np.abs(sim - obs).sum())


def distribution_accuracy(
    elicitation: pd.DataFrame, frame: pd.DataFrame, holdout: pd.DataFrame, *, draw_index: int = 0
) -> pd.DataFrame:
    """1 - TVD between the frame-WEIGHTED simulated marginal and the true
    population marginal, per (method, frame, event_id).

    The simulated marginal is the expected value of ``probs`` (not the hard
    ``top_option``), weighted by ``frame["weight"]`` -- getting that weight
    join wrong (or dropping it) silently turns this into an unweighted
    comparison of the ~100 personas against the population, which flatters
    the method by hiding exactly the distortion frame error is supposed to
    surface. The population marginal comes from the FULL ``holdout`` table
    for that event (every member, not just the panel), which is the true
    target being reproduced.
    """
    calls = elicitation[elicitation["draw_index"] == draw_index]
    weighted = calls.merge(
        frame[["frame", "member_id", "weight"]],
        left_on=["frame", "persona_id"],
        right_on=["frame", "member_id"],
        how="inner",
        validate="many_to_one",
    )

    results = []
    for event_id, event_holdout in holdout.groupby("event_id"):
        n_options = int(event_holdout["n_options"].iloc[0])
        pop_share = (
            event_holdout["outcome_index"]
            .value_counts(normalize=True)
            .reindex(range(n_options), fill_value=0.0)
            .sort_index()
            .to_numpy()
        )
        subset = weighted[weighted["item_id"] == event_id]
        for (method, frame_name), g in subset.groupby(["method", "frame"]):
            probs = np.vstack(g["probs"].to_numpy())
            w = g["weight"].to_numpy(dtype=float)
            sim_share = (probs * w[:, None]).sum(axis=0) / w.sum()
            tvd = float(0.5 * np.abs(pop_share - sim_share).sum())
            results.append(
                {
                    "method": method,
                    "frame": frame_name,
                    "event_id": event_id,
                    "tvd": tvd,
                    "distribution_accuracy": 1.0 - tvd,
                    "n_personas": len(g),
                }
            )
    return pd.DataFrame(
        results, columns=["method", "frame", "event_id", "tvd", "distribution_accuracy", "n_personas"]
    )


# --------------------------------------------------------------------------
# cross-tab recovery: does the method get the gradient right, not just the topline
# --------------------------------------------------------------------------


def _fit_logistic(
    predictors: pd.DataFrame, y: pd.Series, attributes: pd.DataFrame
) -> dict[str, float]:
    """Fit a logistic model of ``y`` on :data:`CROSSTAB_PREDICTORS`,
    z-scoring ``majority_pct`` against the POPULATION's mean/std (not the
    fitting sample's) so coefficients from the real (full population) and
    simulated (~100-persona panel) models are on the same scale and directly
    comparable. Returns NaN coefficients rather than raising when ``y`` has
    fewer than two classes -- a degenerate fit is a real, reportable finding
    for that (method, frame, event), not a crash.
    """
    mean = attributes["majority_pct"].mean()
    std = attributes["majority_pct"].std(ddof=0) or 1.0
    coef_names = ("intercept", *CROSSTAB_PREDICTORS)
    if y.nunique() < 2:
        return dict.fromkeys(coef_names, float("nan"))

    design = np.column_stack(
        [
            (predictors["majority_pct"].to_numpy(dtype=float) - mean) / std,
            predictors["is_2024_intake"].to_numpy(dtype=float),
        ]
    )
    # C=np.inf is an unregularised fit -- the ``penalty=None`` spelling is
    # deprecated as of scikit-learn 1.8 and removed in 1.10.
    model = LogisticRegression(C=np.inf, max_iter=1000)
    model.fit(design, y.to_numpy())
    return {
        "intercept": float(model.intercept_[0]),
        "majority_pct": float(model.coef_[0][0]),
        "is_2024_intake": float(model.coef_[0][1]),
    }



def leakage_gain(
    metric_df: pd.DataFrame, *, value_col: str, group: Sequence[str] = ("frame", "event_id")
) -> pd.DataFrame:
    """(method score) - (RECALL score), per ``group``, for every method other
    than RECALL. Signed, and deliberately NOT clamped at zero: a negative
    leakage_gain -- a method that underperforms the no-persona recall control
    -- is a real, reportable result (the model is retrieving what it already
    knows about the named MP, not simulating from the dossier), not a
    failure to hide by flooring it at zero.

    ``metric_df`` is any per-(method, frame, event) metric table with a
    ``method`` column and a ``value_col`` score column (e.g. the output of
    :func:`balanced_accuracy`). For a lower-is-better metric such as
    ``brier_score``, the caller is responsible for negating before
    interpreting sign as "better".
    """
    group = list(group)
    recall = metric_df.loc[metric_df["method"] == "RECALL", [*group, value_col]].rename(
        columns={value_col: "recall_score"}
    )
    other = metric_df[metric_df["method"] != "RECALL"]
    merged = other.merge(recall, on=group, how="inner")
    merged["leakage_gain"] = merged[value_col] - merged["recall_score"]
    return merged


# --------------------------------------------------------------------------
# dispersion: is the panel under-confident-by-repetition
# --------------------------------------------------------------------------


def within_persona_entropy(
    draws: pd.DataFrame, *, group: Sequence[str] = ("method", "frame", "persona_id", "item_id")
) -> pd.DataFrame:
    """Shannon entropy (bits) of ``top_option`` across repeated draws, per
    ``group``. Zero means the persona answered identically on every draw --
    that is under-dispersion, not confidence, and it is what produces
    falsely-confident toplines if left uncalibrated.
    """
    group = list(group)

    def _entropy(g: pd.DataFrame) -> float:
        p = g["top_option"].value_counts(normalize=True).to_numpy()
        return float(-(p * np.log2(p)).sum())

    return (
        draws.groupby(group, dropna=False)
        .apply(lambda g: _entropy(g), include_groups=False)
        .reset_index(name="entropy_bits")
    )


def dispersion_ratio(
    draws: pd.DataFrame,
    *,
    benchmark: float = HUMAN_TEST_RETEST_ENTROPY_BITS,
    group: Sequence[str] = ("method", "frame", "item_id"),
) -> pd.DataFrame:
    """Mean simulated within-persona entropy over ``benchmark``, per
    ``group``. A ratio near 1 means the panel is about as internally
    unstable as a real human test-retest sample; well below 1 means the
    panel is under-dispersed and any topline built from it will read more
    confident than the underlying behaviour actually is.
    """
    group = list(group)
    per_persona = within_persona_entropy(draws, group=(*group, "persona_id"))
    agg = per_persona.groupby(group, dropna=False)["entropy_bits"].mean().reset_index()
    agg["dispersion_ratio"] = agg["entropy_bits"] / benchmark
    return agg


# --------------------------------------------------------------------------
# error decomposition: frame error vs response error
# --------------------------------------------------------------------------


def decompose_error(
    attributes: pd.DataFrame,
    frame: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    group: Sequence[str] = DEFAULT_GROUP,
) -> pd.DataFrame:
    """Split total error per ``group`` into frame and response components.

    Superseded by :func:`decompose_error_tvd`, which puts both on the same scale.
    This one composes ``1 - balanced_accuracy`` with a mean per-margin TVD, which
    are not commensurable, so the shares summing to 1 is arithmetic rather than a
    decomposition.

    Frame error is taken from :func:`plp_sim.frames.frame_error`, not recomputed.
    It is one number per frame, so ``scored`` and ``frame`` must both already be
    restricted to a single named frame or one frame's error is applied to another's
    rows.

    The split is additive, not causal: ``response = max(total - frame, 0)``.
    """
    group = list(group)
    ba = balanced_accuracy(scored, group=group)

    frame_err = frames.frame_error(attributes, frame)
    frame_tvd = float(frame_err.loc[frame_err["metric"] == "tvd", "error"].mean())

    rows = []
    for _, row in ba.iterrows():
        total_error = 1.0 - float(row["balanced_accuracy"])
        if total_error <= 1e-12:
            frame_component, response_component = total_error, 0.0
            frame_share, response_share = 1.0, 0.0
        else:
            frame_component = min(frame_tvd, total_error)
            response_component = total_error - frame_component
            frame_share = frame_component / total_error
            response_share = response_component / total_error
        rows.append(
            {
                **{col: row[col] for col in group},
                "total_error": total_error,
                "frame_error": frame_component,
                "response_error": response_component,
                "frame_share": frame_share,
                "response_share": response_share,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# bootstrap: resample PERSONAS, not rows
# --------------------------------------------------------------------------


def bootstrap(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    *,
    persona_col: str = "persona_id",
    n_resamples: int = 1000,
    seed: int | None = None,
    ci: float = 0.95,
) -> dict[str, float]:
    """Persona-level (cluster) bootstrap CI for any scalar ``metric_fn(df)``.

    Resamples PERSONAS with replacement, not rows: every row belonging to a
    resampled persona travels together (and is duplicated if that persona is
    drawn more than once), preserving within-persona correlation across
    items and draws. Resampling rows directly would treat every row as an
    independent unit and understate uncertainty by exactly the amount that
    matters here -- the panel is ~100 personas, not the however-many rows
    that fan out from them, and point estimates from ~100 simulated
    respondents invite exactly the challenge a presentation does not want.

    ``seed`` defaults to ``config.get_settings().random_seed`` so a bootstrap
    run without an explicit seed is still reproducible and tied to the same
    provenance as the rest of the run.
    """
    if seed is None:
        seed = get_settings().random_seed
    rng = np.random.default_rng(seed)

    reset = df.reset_index(drop=True)
    indices = reset.groupby(persona_col).indices  # persona -> positional index array
    personas = np.array(list(indices))

    point = float(metric_fn(reset))
    replicates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        drawn = rng.choice(personas, size=len(personas), replace=True)
        positions = np.concatenate([indices[p] for p in drawn])
        replicates[i] = float(metric_fn(reset.iloc[positions]))

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(replicates, [alpha, 1.0 - alpha])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "se": float(np.std(replicates, ddof=1)),
        "ci": ci,
        "n_resamples": n_resamples,
        "n_personas": len(personas),
    }


# --------------------------------------------------------------------------
# low captured_mass diagnostic
# --------------------------------------------------------------------------


def low_captured_mass(
    elicitation: pd.DataFrame,
    *,
    threshold: float = schemas.MIN_CAPTURED_MASS,
    group: Sequence[str] = ("method", "frame", "item_id"),
) -> pd.DataFrame:
    """Count calls where ``captured_mass < threshold`` per ``group``.

    Per ``schemas.MIN_CAPTURED_MASS``: a low-captured-mass call means the
    model answered something other than the item, before renormalisation
    papered over it. This is a finding to surface in the run report, not
    noise to drop, so it is counted here rather than filtered out anywhere
    upstream.
    """
    group = list(group)
    flagged = elicitation.assign(_low=elicitation["captured_mass"] < threshold)
    if group:
        out = (
            flagged.groupby(group, dropna=False)
            .agg(n_calls=("_low", "size"), n_low_captured_mass=("_low", "sum"))
            .reset_index()
        )
    else:
        out = pd.DataFrame(
            [{"n_calls": len(flagged), "n_low_captured_mass": int(flagged["_low"].sum())}]
        )
    out["share_low_captured_mass"] = out["n_low_captured_mass"] / out["n_calls"]
    return out


# --------------------------------------------------------------------------
# single-scale decomposition
# --------------------------------------------------------------------------


def decompose_error_tvd(
    scored: pd.DataFrame,
    frame: pd.DataFrame,
    population_outcomes: pd.DataFrame,
    *,
    group: Sequence[str] = DEFAULT_GROUP,
) -> pd.DataFrame:
    """Decompose error into frame and response components on one common scale.

    Every quantity is a total variation distance over the same outcome space, so
    the three are directly comparable:

    ``frame``     panel's own observed outcomes vs the population. The error left
                  with a perfect simulator, purely because 100 personas are not
                  405 MPs. No model call needed, which makes it the honest ceiling.
    ``response``  panel's simulated outcomes vs its own observed outcomes. What
                  the model adds, holding the panel fixed.
    ``total``     panel's simulated outcomes vs the population. End to end.

    All three use the frame's population weights.

    **These do not sum.** TVD obeys the triangle inequality, so
    ``total <= frame + response``. The gap is information: it means the model's
    errors partly cancel the panel's bias, which is luck, not skill. Report the
    three numbers rather than shares.
    """
    group = list(group)
    weights = frame.set_index("member_id")["weight"]

    def _marginal(labels: pd.Series, w: pd.Series | None = None) -> pd.Series:
        if w is None:
            w = pd.Series(1.0, index=labels.index)
        tot = float(w.sum())
        if tot <= 0:
            return pd.Series(dtype="float64")
        return w.groupby(labels).sum() / tot

    def _tvd(a: pd.Series, b: pd.Series) -> float:
        idx = a.index.union(b.index)
        return float(0.5 * (a.reindex(idx, fill_value=0.0) - b.reindex(idx, fill_value=0.0)).abs().sum())

    rows: list[dict[str, object]] = []
    for keys, block in scored.groupby(group, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        event = dict(zip(group, keys)).get("event_id")

        pop = population_outcomes[population_outcomes["event_id"] == event]
        pop_marginal = _marginal(pop["outcome"])

        w = block["persona_id"].map(weights)
        observed = _marginal(block["outcome"], w)
        simulated = _marginal(block["top_option"], w)

        frame_err = _tvd(observed, pop_marginal)
        response_err = _tvd(simulated, observed)
        total_err = _tvd(simulated, pop_marginal)

        rows.append(
            dict(zip(group, keys))
            | {
                "frame_error": frame_err,
                "response_error": response_err,
                "total_error": total_err,
                # total <= frame + response by the triangle inequality; a large
                # slack means the two errors are cancelling rather than adding.
                "cancellation_slack": frame_err + response_err - total_err,
                "n_personas": int(block["persona_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)
