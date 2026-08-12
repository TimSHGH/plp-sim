"""Tests for plp_sim.metrics.

The module under test turns raw elicitation + holdout tables into the
the headline figures. The properties worth testing are the ones a
*wrong* implementation would pass by accident: balanced accuracy that
secretly reports raw accuracy on skewed data, an MCC that isn't actually
penalising randomness, a distribution-accuracy calculation whose weight join
is silently a no-op, a leakage gain that gets clamped at zero and hides a
real negative result, a bootstrap that resamples rows instead of personas
and so understates uncertainty, and a captured-mass diagnostic that drops
the very calls it exists to surface.

Most tests build small, hand-constructed tables rather than routing through
the full synthetic fixtures, so the expected answer can be worked out by
hand and the test is checking the metric, not the fixture generator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plp_sim import metrics
from tests.fixtures import synthetic

# --------------------------------------------------------------------------
# balanced_accuracy / matthews_corrcoef -- the headline, not raw accuracy
# --------------------------------------------------------------------------


def _skewed_scored(n_majority: int = 79, n_minority: int = 21) -> pd.DataFrame:
    """A ~79/21 skewed (method, frame, event) group: 'outcome' is the truth,
    'top_option' is supplied by each test."""
    outcome = ["A"] * n_majority + ["B"] * n_minority
    return pd.DataFrame(
        {
            "method": "P4",
            "frame": "F2",
            "event_id": "e1",
            "outcome": outcome,
            "base_rate": n_majority / (n_majority + n_minority),
        }
    )


def test_balanced_accuracy_of_constant_predictor_on_skewed_data_is_near_half():
    # A predictor that always guesses the modal class scores ~79% RAW
    # accuracy on this skew but does zero discriminative work -- balanced
    # accuracy must report that as ~50%, not ~79%.
    scored = _skewed_scored()
    scored["top_option"] = "A"

    out = metrics.balanced_accuracy(scored)
    assert out.loc[0, "balanced_accuracy"] == pytest.approx(0.5, abs=1e-9)
    assert out.loc[0, "accuracy"] == pytest.approx(0.79, abs=1e-9)
    assert out.loc[0, "base_rate"] == pytest.approx(0.79, abs=1e-9)


def test_matthews_corrcoef_of_random_predictor_is_near_zero():
    rng = np.random.default_rng(0)
    n = 4000
    outcome = rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.3, 0.1])
    # top_option drawn independently of outcome -- genuinely uncorrelated.
    top_option = rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.3, 0.1])
    scored = pd.DataFrame(
        {
            "method": "P4",
            "frame": "F2",
            "event_id": "e1",
            "outcome": outcome,
            "top_option": top_option,
            "base_rate": 0.6,
        }
    )
    out = metrics.matthews_corrcoef(scored)
    assert abs(out.loc[0, "matthews_corrcoef"]) < 0.05


def test_matthews_corrcoef_of_perfect_predictor_is_one():
    scored = _skewed_scored()
    scored["top_option"] = scored["outcome"]
    out = metrics.matthews_corrcoef(scored)
    assert out.loc[0, "matthews_corrcoef"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# distribution_accuracy -- the weight join must not be a silent no-op
# --------------------------------------------------------------------------


def _hand_built_distribution_case(weight_a: float, weight_b: float):
    """Two personas, a 2-option item. Population truth is an even 50/50
    split; each persona's simulated distribution is confidently wrong about
    itself (A leans option 0, B leans option 1). Only the frame weights
    determine whether the weighted panel marginal comes out even (matching
    the population) or skewed (not matching it).
    """
    elicitation = pd.DataFrame(
        {
            "method": ["P4", "P4"],
            "frame": ["F2", "F2"],
            "persona_id": [1, 2],
            "item_id": ["evt", "evt"],
            "draw_index": [0, 0],
            "probs": [[0.9, 0.1], [0.1, 0.9]],
        }
    )
    frame = pd.DataFrame(
        {"frame": ["F2", "F2"], "member_id": [1, 2], "weight": [weight_a, weight_b]}
    )
    holdout = pd.DataFrame(
        {
            "member_id": [1, 2],
            "event_id": ["evt", "evt"],
            "outcome_index": [0, 1],
            "n_options": [2, 2],
        }
    )
    return elicitation, frame, holdout


def test_distribution_accuracy_actually_uses_the_weights():
    # Equal weights: the panel's own confident-but-opposite errors cancel,
    # reproducing the population's 50/50 split exactly (tvd == 0).
    elicitation, equal_frame, holdout = _hand_built_distribution_case(5.0, 5.0)
    equal_result = metrics.distribution_accuracy(elicitation, equal_frame, holdout)
    assert equal_result.loc[0, "tvd"] == pytest.approx(0.0, abs=1e-9)

    # Deliberately unequal weights (9:1): persona 1 dominates the weighted
    # marginal, which must now diverge from the population split.
    _, unequal_frame, _ = _hand_built_distribution_case(9.0, 1.0)
    unequal_result = metrics.distribution_accuracy(elicitation, unequal_frame, holdout)
    assert unequal_result.loc[0, "tvd"] == pytest.approx(0.32, abs=1e-9)

    # If the weighting were silently dead (e.g. an unweighted mean), these
    # two results would be identical -- they must differ.
    assert equal_result.loc[0, "tvd"] != pytest.approx(unequal_result.loc[0, "tvd"])
    assert equal_result.loc[0, "distribution_accuracy"] > unequal_result.loc[0, "distribution_accuracy"]


# --------------------------------------------------------------------------
# leakage_gain -- signed, and never clamped
# --------------------------------------------------------------------------


def test_leakage_gain_goes_negative_when_recall_beats_the_method():
    metric_df = pd.DataFrame(
        {
            "method": ["P4", "P5", "RECALL"],
            "frame": ["F2", "F2", "F2"],
            "event_id": ["e1", "e1", "e1"],
            "balanced_accuracy": [0.50, 0.55, 0.80],
        }
    )
    out = metrics.leakage_gain(metric_df, value_col="balanced_accuracy")
    m2 = out.loc[out["method"] == "P4", "leakage_gain"].iloc[0]
    m3 = out.loc[out["method"] == "P5", "leakage_gain"].iloc[0]

    # RECALL (0.80) beats both methods: the gain is negative, reported
    # exactly, not floored at 0.
    assert m2 == pytest.approx(0.50 - 0.80)
    assert m3 == pytest.approx(0.55 - 0.80)
    assert m2 < 0
    assert m3 < 0


def test_leakage_gain_positive_when_method_beats_recall():
    metric_df = pd.DataFrame(
        {
            "method": ["P5", "RECALL"],
            "frame": ["F2", "F2"],
            "event_id": ["e1", "e1"],
            "balanced_accuracy": [0.75, 0.55],
        }
    )
    out = metrics.leakage_gain(metric_df, value_col="balanced_accuracy")
    assert out.loc[0, "leakage_gain"] == pytest.approx(0.20)


# --------------------------------------------------------------------------
# bootstrap -- resamples personas, CIs contain the point and widen with less data
# --------------------------------------------------------------------------


def test_bootstrap_ci_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"persona_id": np.arange(60), "x": rng.normal(0.4, 1.0, 60)})
    result = metrics.bootstrap(df, lambda d: d["x"].mean(), n_resamples=1000, seed=0)
    assert result["ci_low"] <= result["point"] <= result["ci_high"]


def test_bootstrap_ci_widens_as_resample_size_falls():
    rng = np.random.default_rng(2)
    small = pd.DataFrame({"persona_id": np.arange(5), "x": rng.normal(0.0, 1.0, 5)})
    large = pd.DataFrame({"persona_id": np.arange(200), "x": rng.normal(0.0, 1.0, 200)})

    small_result = metrics.bootstrap(small, lambda d: d["x"].mean(), n_resamples=1000, seed=0)
    large_result = metrics.bootstrap(large, lambda d: d["x"].mean(), n_resamples=1000, seed=0)

    small_width = small_result["ci_high"] - small_result["ci_low"]
    large_width = large_result["ci_high"] - large_result["ci_low"]
    assert small_width > large_width


def test_bootstrap_resamples_whole_personas_not_rows():
    # Every row for a given persona must move together: a persona with an
    # extreme, internally-consistent value should show up as that whole
    # value in a replicate, never blended row-by-row with another persona's.
    df = pd.DataFrame(
        {
            "persona_id": [1, 1, 1, 2, 2, 2],
            "x": [10.0, 10.0, 10.0, 0.0, 0.0, 0.0],
        }
    )

    def per_persona_mean_of_means(d: pd.DataFrame) -> float:
        return float(d.groupby("persona_id")["x"].mean().mean())

    result = metrics.bootstrap(df, per_persona_mean_of_means, n_resamples=200, seed=0)
    # The only possible replicate values are 10.0, 0.0, or 5.0 (one of
    # each) -- never anything else, which is what a row-level (rather than
    # persona-level) resample would produce.
    assert result["point"] == pytest.approx(5.0)
    assert result["ci_low"] in (0.0, 5.0, 10.0)
    assert result["ci_high"] in (0.0, 5.0, 10.0)


# --------------------------------------------------------------------------
# decompose_error -- shares sum to 1, frame_error delegates to frames.frame_error
# --------------------------------------------------------------------------


def test_decompose_error_shares_sum_to_one(attributes, holdout):
    elicitation = synthetic.make_elicitation(attributes, method="P4", frame="F2", seed=3)
    scored = metrics.score_calls(elicitation, holdout)

    # A deliberately partial, unequally-weighted "frame" -- the first half of
    # the population, re-weighted to stand in for the whole -- so frame_tvd
    # is nonzero and the decomposition is doing real work, not a degenerate
    # 0/100 split.
    half = attributes.iloc[: len(attributes) // 2]
    frame = pd.DataFrame(
        {
            "frame": "F2",
            "member_id": half["member_id"].to_numpy(),
            "weight": len(attributes) / len(half),
            "stratum": None,
        }
    )

    out = metrics.decompose_error(attributes, frame, scored)
    assert len(out) > 0
    totals = out["frame_share"] + out["response_share"]
    assert np.allclose(totals, 1.0)
    assert (out["frame_error"] >= 0).all()
    assert (out["response_error"] >= 0).all()
    # frame_error + response_error must reconstruct total_error exactly.
    assert np.allclose(out["frame_error"] + out["response_error"], out["total_error"])


# --------------------------------------------------------------------------
# low_captured_mass -- counted, not dropped
# --------------------------------------------------------------------------


def test_low_captured_mass_counts_rather_than_drops(attributes):
    elicitation = synthetic.make_elicitation(attributes, method="P4", frame="F2", seed=4)
    n = len(elicitation)
    # Force a known number of rows below threshold.
    elicitation = elicitation.copy()
    elicitation.loc[elicitation.index[:7], "captured_mass"] = 0.10
    assert (elicitation["captured_mass"] < metrics.schemas.MIN_CAPTURED_MASS).sum() == 7

    out = metrics.low_captured_mass(elicitation, group=())
    assert out.loc[0, "n_calls"] == n  # nothing dropped from the total
    assert out.loc[0, "n_low_captured_mass"] == 7
    assert out.loc[0, "share_low_captured_mass"] == pytest.approx(7 / n)


def test_low_captured_mass_grouped_matches_manual_count(attributes):
    elicitation = synthetic.make_elicitation(attributes, method="P4", frame="F2", seed=5)
    elicitation = elicitation.copy()
    elicitation.loc[elicitation.index[:3], "captured_mass"] = 0.05

    out = metrics.low_captured_mass(elicitation, group=("method", "frame"))
    row = out.iloc[0]
    assert row["n_calls"] == len(elicitation)
    assert row["n_low_captured_mass"] == 3


# --------------------------------------------------------------------------
# score_calls -- the join underlying every per-event metric
# --------------------------------------------------------------------------


def test_score_calls_joins_on_persona_and_item(attributes, holdout):
    elicitation = synthetic.make_elicitation(attributes, method="P4", frame="F2", seed=6)
    scored = metrics.score_calls(elicitation, holdout)
    assert len(scored) == len(attributes)  # one holdout event, every member scored
    assert set(scored["persona_id"]) == set(attributes["member_id"])


def test_score_calls_drops_items_with_no_holdout_event(attributes, holdout):
    elicitation = synthetic.make_elicitation(attributes, method="P4", frame="F2", seed=7)
    elicitation = elicitation.copy()
    elicitation["item_id"] = "not_a_holdout_event"
    scored = metrics.score_calls(elicitation, holdout)
    assert len(scored) == 0


def test_score_calls_respects_draw_index(attributes, holdout):
    draw0 = synthetic.make_elicitation(attributes, method="P4", frame="F2", seed=8)
    draw1 = draw0.copy()
    draw1["draw_index"] = 1
    both = pd.concat([draw0, draw1], ignore_index=True)
    scored = metrics.score_calls(both, holdout, draw_index=0)
    assert (scored["draw_index"] == 0).all()
    assert len(scored) == len(attributes)


# --------------------------------------------------------------------------
# brier_score -- bounds and perfect/worst cases
# --------------------------------------------------------------------------


def test_brier_score_is_zero_for_a_perfectly_confident_correct_call():
    scored = pd.DataFrame(
        {
            "method": ["P4"],
            "frame": ["F2"],
            "event_id": ["e1"],
            "probs": [[1.0, 0.0, 0.0]],
            "outcome_index": [0],
            "base_rate": [0.5],
        }
    )
    out = metrics.brier_score(scored)
    assert out.loc[0, "brier_score"] == pytest.approx(0.0)


def test_brier_score_is_two_for_a_perfectly_confident_wrong_call():
    scored = pd.DataFrame(
        {
            "method": ["P4"],
            "frame": ["F2"],
            "event_id": ["e1"],
            "probs": [[1.0, 0.0, 0.0]],
            "outcome_index": [1],
            "base_rate": [0.5],
        }
    )
    out = metrics.brier_score(scored)
    assert out.loc[0, "brier_score"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# dispersion_ratio -- zero entropy for an identical-every-draw persona
# --------------------------------------------------------------------------


def test_within_persona_entropy_is_zero_when_answer_never_varies():
    draws = pd.DataFrame(
        {
            "method": ["P4"] * 5,
            "frame": ["F2"] * 5,
            "persona_id": [1] * 5,
            "item_id": ["evt"] * 5,
            "top_option": ["A"] * 5,
        }
    )
    out = metrics.within_persona_entropy(draws)
    assert out.loc[0, "entropy_bits"] == pytest.approx(0.0)


def test_dispersion_ratio_is_zero_when_fully_under_dispersed():
    draws = pd.DataFrame(
        {
            "method": ["P4"] * 6,
            "frame": ["F2"] * 6,
            "persona_id": [1, 1, 1, 2, 2, 2],
            "item_id": ["evt"] * 6,
            "top_option": ["A", "A", "A", "B", "B", "B"],
        }
    )
    out = metrics.dispersion_ratio(draws)
    assert out.loc[0, "dispersion_ratio"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# single-scale TVD decomposition
# --------------------------------------------------------------------------


def _decomp_inputs(n=120, seed=3):
    """Panel + population + simulated answers, on one shared outcome space."""
    rng = np.random.default_rng(seed)
    attrs = synthetic.make_attributes(n=n, seed=seed)
    labels = np.array(["held_back", "day1_bandwagon"])
    pop_outcome = rng.choice(labels, size=n, p=[0.21, 0.79])
    population = pd.DataFrame(
        {"member_id": attrs.member_id, "event_id": "loyalty", "outcome": pop_outcome}
    )
    picked = attrs.member_id.iloc[: n // 2].to_numpy()
    frame = pd.DataFrame(
        {"frame": "F2", "member_id": picked, "weight": n / len(picked), "stratum": None}
    )
    obs = population.set_index("member_id").loc[picked, "outcome"].to_numpy()
    scored = pd.DataFrame(
        {
            "method": "P4", "frame": "F2", "event_id": "loyalty",
            "persona_id": picked, "outcome": obs, "top_option": obs,
        }
    )
    return population, frame, scored


def test_tvd_decomposition_response_error_is_zero_for_a_perfect_simulator():
    population, frame, scored = _decomp_inputs()
    out = metrics.decompose_error_tvd(scored, frame, population)
    assert out["response_error"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    # frame error survives even a perfect simulator -- that is the whole point:
    # it is the cost of 100 personas standing in for the full population.
    assert out["frame_error"].iloc[0] >= 0.0
    assert out["total_error"].iloc[0] == pytest.approx(out["frame_error"].iloc[0], abs=1e-12)


def test_tvd_decomposition_response_error_grows_as_the_simulator_degrades():
    population, frame, scored = _decomp_inputs()
    flipped = scored.copy()
    flip = {"held_back": "day1_bandwagon", "day1_bandwagon": "held_back"}
    flipped["top_option"] = flipped["outcome"].map(flip)
    good = metrics.decompose_error_tvd(scored, frame, population)["response_error"].iloc[0]
    bad = metrics.decompose_error_tvd(flipped, frame, population)["response_error"].iloc[0]
    assert bad > good


def test_tvd_decomposition_obeys_the_triangle_inequality():
    """total <= frame + response. If this fails the numbers are not TVDs."""
    population, frame, scored = _decomp_inputs()
    noisy = scored.copy()
    rng = np.random.default_rng(9)
    mask = rng.random(len(noisy)) < 0.35
    flip = {"held_back": "day1_bandwagon", "day1_bandwagon": "held_back"}
    noisy.loc[mask, "top_option"] = noisy.loc[mask, "outcome"].map(flip)
    out = metrics.decompose_error_tvd(noisy, frame, population)
    row = out.iloc[0]
    assert row["total_error"] <= row["frame_error"] + row["response_error"] + 1e-12
    assert row["cancellation_slack"] >= -1e-12


def test_tvd_decomposition_components_are_all_on_the_zero_one_scale():
    population, frame, scored = _decomp_inputs()
    out = metrics.decompose_error_tvd(scored, frame, population)
    for col in ("frame_error", "response_error", "total_error"):
        assert 0.0 <= out[col].iloc[0] <= 1.0


def test_tvd_decomposition_uses_frame_weights():
    """Unequal weights must change the answer, or the weighting is dead code."""
    population, frame, scored = _decomp_inputs()
    skewed = frame.copy()
    skewed["weight"] = np.linspace(0.1, 5.0, len(skewed))
    a = metrics.decompose_error_tvd(scored, frame, population)["frame_error"].iloc[0]
    b = metrics.decompose_error_tvd(scored, skewed, population)["frame_error"].iloc[0]
    assert a != pytest.approx(b)
