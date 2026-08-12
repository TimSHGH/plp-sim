"""Tests for plp_sim.frames.

The module under test builds the ~100-persona panels that stand in for the
full PLP and measures the error that substitution alone introduces. The
properties worth testing are the ones a *wrong* implementation could pass by
accident: a Gower matrix that quietly produces NaN, a rake that stops at
max-iter without actually matching margins, a frame-error metric that can't
tell a deliberately bad panel from a good one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plp_sim import frames, schemas
from plp_sim.config import Settings
from tests.fixtures import synthetic

# --------------------------------------------------------------------------
# gower_matrix
# --------------------------------------------------------------------------


def test_gower_symmetric_and_zero_diagonal(attributes_large):
    d = frames.gower_matrix(attributes_large, list(schemas.SEG_VARS))
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


def test_gower_range_is_zero_to_one(attributes_large):
    d = frames.gower_matrix(attributes_large, list(schemas.SEG_VARS))
    assert d.min() >= 0.0
    assert d.max() <= 1.0 + 1e-12


def test_gower_nan_does_not_produce_nan_distance(attributes_large):
    # nomination_day is ~15% null by construction; those pairs must fall
    # back to averaging over the remaining SEG_VARS, never to NaN.
    assert attributes_large["nomination_day"].isna().any()
    d = frames.gower_matrix(attributes_large, list(schemas.SEG_VARS))
    assert not np.isnan(d).any()


def test_gower_numeric_distance_is_range_normalised():
    df = pd.DataFrame({"x": [0.0, 5.0, 10.0]})
    d = frames.gower_matrix(df, ["x"])
    assert d[0, 2] == pytest.approx(1.0)  # full range apart
    assert d[0, 1] == pytest.approx(0.5)  # half the range apart


def test_gower_categorical_distance_is_0_or_1():
    df = pd.DataFrame({"c": ["a", "a", "b"]})
    d = frames.gower_matrix(df, ["c"])
    assert d[0, 1] == pytest.approx(0.0)
    assert d[0, 2] == pytest.approx(1.0)


def test_gower_missing_value_excluded_from_pair_not_whole_row():
    # Row 1 is missing `y`; its distance to row 0 must fall back to `x`
    # alone -- not zero (which would understate it) and not NaN.
    df = pd.DataFrame({"x": [0.0, 10.0, 10.0], "y": [0.0, np.nan, 10.0]})
    d = frames.gower_matrix(df, ["x", "y"])
    assert not np.isnan(d[0, 1])
    assert d[0, 1] == pytest.approx(1.0)  # |0-10| / range(10) on x alone


# --------------------------------------------------------------------------
# build_f1 / build_f2 -- shared frame contract
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def settings():
    return Settings()


@pytest.fixture(scope="module")
def f1(attributes_large, settings):
    return frames.build_f1(attributes_large, settings)


@pytest.fixture(scope="module")
def f2(attributes_large, settings):
    return frames.build_f2(attributes_large, settings)


@pytest.mark.parametrize("name", ["f1", "f2"])
def test_frame_conforms_to_schema(name, request):
    schemas.validate(request.getfixturevalue(name), schemas.FRAME)


@pytest.mark.parametrize("name", ["f1", "f2"])
def test_weights_sum_to_population_size(name, request, attributes_large):
    df = request.getfixturevalue(name)
    assert df["weight"].sum() == pytest.approx(len(attributes_large))


@pytest.mark.parametrize("name", ["f1", "f2"])
def test_every_member_id_exists_in_attributes(name, request, attributes_large):
    df = request.getfixturevalue(name)
    assert set(df["member_id"]) <= set(attributes_large["member_id"])


@pytest.mark.parametrize("name", ["f1", "f2"])
def test_frame_has_n_personas_rows(name, request, settings):
    assert len(request.getfixturevalue(name)) == settings.n_personas


def test_f1_stratum_is_labelled(f1):
    assert f1["stratum"].notna().all()


def test_f2_stratum_is_null(f2):
    assert f2["stratum"].isna().all()


def test_f2_medoids_are_distinct_real_members(f2, attributes_large):
    # F2 selects k-medoids, i.e. actual rows -- not centroids that might not
    # correspond to any real MP.
    assert f2["member_id"].is_unique
    assert set(f2["member_id"]) <= set(attributes_large["member_id"])


# --------------------------------------------------------------------------
# stratification
# --------------------------------------------------------------------------


def test_strata_collapse_below_min_size(attributes):
    # The 40-row fixture crossed with tercile x 8 parties x payroll x intake
    # has mostly near-empty cells; collapsing must leave nothing below
    # MIN_STRATUM_SIZE rather than erroring or sampling from a group of one.
    counts = frames._stratum_labels(attributes).value_counts()
    assert (counts >= frames.MIN_STRATUM_SIZE).all()


def test_strata_partition_the_population(attributes_large):
    # Collapsing must relabel, never drop, a row.
    stratum = frames._stratum_labels(attributes_large)
    assert stratum.notna().all()
    assert len(stratum) == len(attributes_large)


def test_proportional_allocation_sums_to_total_and_respects_capacity():
    sizes = pd.Series({"a": 3, "b": 47, "c": 50})
    alloc = frames._proportional_allocation(sizes, 10)
    assert int(alloc.sum()) == 10
    assert (alloc <= sizes).all()


def test_proportional_allocation_never_exceeds_a_tiny_stratum():
    # A stratum smaller than its proportional share must be capped at its
    # own population, with the remainder going elsewhere -- not erroring.
    sizes = pd.Series({"tiny": 1, "rest": 99})
    alloc = frames._proportional_allocation(sizes, 50)
    assert int(alloc.sum()) == 50
    assert alloc["tiny"] <= 1


# --------------------------------------------------------------------------
# raking
# --------------------------------------------------------------------------


def test_f1_raking_reproduces_seg_var_margins(attributes_large, f1, settings):
    edges = frames._rake_bin_edges(attributes_large)
    pop_categories = frames._seg_var_categories(attributes_large, edges)
    merged = f1.merge(attributes_large, on="member_id")
    sample_categories = frames._seg_var_categories(merged, edges)

    worst_gap = 0.0
    for col in pop_categories.columns:
        target = pop_categories[col].value_counts()
        achieved = merged["weight"].groupby(sample_categories[col]).sum()
        achieved = achieved.reindex(target.index, fill_value=0.0)
        worst_gap = max(worst_gap, float((achieved - target).abs().max()))

    assert worst_gap <= settings.rake_tolerance + 1e-6


def test_rake_raises_on_unsatisfiable_target():
    # Category "b" has zero mass in the sample -- no rescaling conjures a
    # nonzero weight where there was none, so this must never converge.
    categories = pd.DataFrame({"x": ["a", "a", "a"]})
    targets = {"x": pd.Series({"a": 2.0, "b": 1.0})}
    weights = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="did not converge"):
        frames.rake(categories, targets, weights, max_iter=50, tol=1e-6)


def test_rake_error_message_reports_achieved_discrepancy():
    categories = pd.DataFrame({"x": ["a", "a", "a"]})
    targets = {"x": pd.Series({"a": 2.0, "b": 1.0})}
    weights = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match=r"discrepancy 1\b"):
        frames.rake(categories, targets, weights, max_iter=5, tol=1e-6)


def test_rake_converges_on_a_satisfiable_target():
    categories = pd.DataFrame({"x": ["a", "a", "b", "b"]})
    targets = {"x": pd.Series({"a": 10.0, "b": 30.0})}
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    result = frames.rake(categories, targets, weights, max_iter=50, tol=1e-9)
    assert result[:2].sum() == pytest.approx(10.0)
    assert result[2:].sum() == pytest.approx(30.0)


# --------------------------------------------------------------------------
# frame_error
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bad_frame(attributes_large):
    # Deliberately bad: the 100 highest-majority MPs, equally weighted --
    # ignores every population margin the good frames are built to match.
    worst = attributes_large.nlargest(100, "majority_pct")
    return pd.DataFrame(
        {
            "frame": "F1",
            "member_id": worst["member_id"].to_numpy(),
            "weight": len(attributes_large) / 100.0,
            "stratum": None,
        }
    )


def test_frame_error_output_shape(attributes_large, f1):
    err = frames.frame_error(attributes_large, f1)
    assert set(err["variable"]) == set(schemas.HELDOUT_VARS) | {"multivariate"}
    assert set(err["metric"]) == {"tvd", "energy_distance"}
    assert (err["error"] >= 0).all()


def test_frame_error_zero_for_the_population_against_itself(attributes_large):
    # The sanity floor for "lower is better": the full, equally-weighted
    # population scored against itself must read (numerically) zero.
    full = pd.DataFrame(
        {
            "frame": "FULL",
            "member_id": attributes_large["member_id"].to_numpy(),
            "weight": 1.0,
            "stratum": None,
        }
    )
    err = frames.frame_error(attributes_large, full)
    assert (err["error"] < 1e-9).all()


def test_bad_frame_scores_worse_than_f1_and_f2(attributes_large, f1, f2, bad_frame):
    err_f1 = frames.frame_error(attributes_large, f1)
    err_f2 = frames.frame_error(attributes_large, f2)
    err_bad = frames.frame_error(attributes_large, bad_frame)

    def mean_tvd(err):
        return err.loc[err["metric"] == "tvd", "error"].mean()

    def energy(err):
        return err.loc[err["metric"] == "energy_distance", "error"].iloc[0]

    assert mean_tvd(err_bad) > mean_tvd(err_f1)
    assert mean_tvd(err_bad) > mean_tvd(err_f2)
    assert energy(err_bad) > energy(err_f1)
    assert energy(err_bad) > energy(err_f2)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_f1_is_deterministic(attributes_large, settings):
    a = frames.build_f1(attributes_large, settings)
    b = frames.build_f1(attributes_large, settings)
    pd.testing.assert_frame_equal(a, b)


def test_f2_is_deterministic(attributes_large, settings):
    a = frames.build_f2(attributes_large, settings)
    b = frames.build_f2(attributes_large, settings)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------
# regression: runtime leakage of a post-cutoff outcome into the raking margins
# --------------------------------------------------------------------------


def test_raking_columns_are_derived_from_the_schema_not_hardcoded():
    """The fence in schemas.py has to bind at RUNTIME, not just on paper.

    `_rake_bin_edges` and `_seg_var_categories` previously hardcoded
    ("majority_pct", "rebellion_rate", "nomination_day"). nomination_day is a
    POST_CUTOFF_VAR -- the outcome -- so F1 was raked to reproduce the true
    nomination split, hitting it to ~5e-8 by construction and making every F1
    distribution-accuracy number on that event meaningless.

    test_schemas.py asserted SEG_VARS and POST_CUTOFF_VARS were disjoint, which
    was true and useless: nothing checked what this module actually consumed.
    This test closes that gap by inspecting the real categories.
    """
    attrs = synthetic.make_attributes(n=200, seed=5)
    edges = frames._rake_bin_edges(attrs)
    cats = frames._seg_var_categories(attrs, edges)

    assert set(cats.columns) == set(schemas.SEG_VARS), (
        "raking margins must be exactly SEG_VARS; got "
        f"{sorted(set(cats.columns) ^ set(schemas.SEG_VARS))} as the difference"
    )
    leaked = set(cats.columns) & set(schemas.POST_CUTOFF_VARS)
    assert not leaked, f"post-cutoff outcome(s) used as raking margins: {sorted(leaked)}"
    assert not set(edges) & set(schemas.POST_CUTOFF_VARS)


def test_f1_does_not_reproduce_a_post_cutoff_outcome_by_construction():
    """The behavioural version of the test above.

    If a post-cutoff outcome is a raking margin, the weighted panel matches the
    population on it to floating-point precision. Genuine sampling error is
    orders of magnitude larger, so a tiny error here is the signature of
    leakage rather than of a good frame.
    """
    attrs = synthetic.make_attributes(n=400, seed=1)
    frame = frames.build_f1(attrs, Settings(n_personas=100, random_seed=0))
    w = frame.set_index("member_id")["weight"]
    picked = attrs[attrs.member_id.isin(w.index)].copy()
    picked["_w"] = picked.member_id.map(w)

    pop = attrs["nomination_bucket"].value_counts()
    panel = picked.groupby("nomination_bucket")["_w"].sum()
    max_err = max(abs(panel.get(k, 0.0) - v) for k, v in pop.items())
    assert max_err > 1e-3, (
        f"F1 reproduces nomination_bucket to {max_err:.2e} -- that is raking "
        "forcing the outcome, not sampling accuracy"
    )


def test_rake_raises_when_a_sample_category_has_no_target():
    """An untargeted category is never rescaled and keeps its initial weight,
    while the loop still reports convergence -- a silent wrong answer."""
    cats = pd.DataFrame({"x": ["a", "a", "b"]})
    targets = {"x": pd.Series({"a": 30.0})}  # 'b' has no target
    with pytest.raises(ValueError, match="no target"):
        frames.rake(cats, targets, np.array([1.0, 1.0, 1000.0]), max_iter=50, tol=1e-6)


def test_collapse_absorbs_a_lone_undersized_category():
    """Relabelling one 1-member cell as "__rare__" leaves a 1-member cell.

    Exactly one MP has a null rebellion_rate (the by-election winner, with no
    pre-cutoff votes). That made a "missing" cell of size 1 whose margin no
    sample could satisfy, and every seed failed with a discrepancy of 1.
    """
    cats = pd.DataFrame({"x": ["big"] * 40 + ["tiny"]})
    out = frames.collapse_rare_categories(cats, min_size=8)
    assert out["x"].value_counts().min() >= 8
    assert "tiny" not in set(out["x"])
