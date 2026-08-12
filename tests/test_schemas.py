"""The contract must actually catch drift, or it is decoration.

These tests are deliberately about the validator failing, not passing: a
schema check that never rejects anything provides no integration guarantee.
"""

from __future__ import annotations

import pandas as pd
import pytest

from plp_sim import schemas
from tests.fixtures import synthetic


def test_fixtures_conform(attributes, holdout, elicitation):
    schemas.validate(attributes, schemas.ATTRIBUTES)
    schemas.validate(holdout, schemas.HOLDOUT)
    schemas.validate(elicitation, schemas.ELICITATION)


def test_seg_and_heldout_vars_exist_and_do_not_overlap():
    cols = set(schemas.ATTRIBUTES.names)
    assert set(schemas.SEG_VARS) <= cols
    assert set(schemas.HELDOUT_VARS) <= cols
    # Frame error is only honest if it is measured on variables the frame
    # never saw. Any overlap here silently invalidates figure 1.
    assert not set(schemas.SEG_VARS) & set(schemas.HELDOUT_VARS)


def test_categorical_seg_vars_are_a_subset_of_seg_vars():
    assert set(schemas.CATEGORICAL_SEG_VARS) <= set(schemas.SEG_VARS)


def test_rejects_missing_column(attributes):
    with pytest.raises(schemas.SchemaError, match="missing columns"):
        schemas.validate(attributes.drop(columns=["majority_pct"]), schemas.ATTRIBUTES)


def test_rejects_unexpected_column(attributes):
    df = attributes.assign(vibe="good")
    with pytest.raises(schemas.SchemaError, match="unexpected columns"):
        schemas.validate(df, schemas.ATTRIBUTES)
    schemas.validate(df, schemas.ATTRIBUTES, allow_extra=True)  # opt-out works


def test_rejects_null_in_non_nullable(attributes):
    df = attributes.copy()
    df.loc[df.index[0], "majority_pct"] = None
    with pytest.raises(schemas.SchemaError, match="non-nullable"):
        schemas.validate(df, schemas.ATTRIBUTES)


def test_rejects_value_outside_allowed_set(attributes):
    df = attributes.copy()
    df.loc[df.index[0], "runner_up_party"] = "Monster Raving Loony"
    with pytest.raises(schemas.SchemaError, match="outside allowed set"):
        schemas.validate(df, schemas.ATTRIBUTES)


def test_rejects_out_of_range(attributes):
    df = attributes.copy()
    df.loc[df.index[0], "majority_pct"] = 140.0
    with pytest.raises(schemas.SchemaError, match="above"):
        schemas.validate(df, schemas.ATTRIBUTES)


def test_rejects_duplicate_key(attributes):
    df = pd.concat([attributes, attributes.head(1)], ignore_index=True)
    with pytest.raises(schemas.SchemaError, match="duplicate row"):
        schemas.validate(df, schemas.ATTRIBUTES)


def test_reports_every_problem_at_once(attributes):
    df = attributes.drop(columns=["majority_pct"]).assign(vibe="good")
    with pytest.raises(schemas.SchemaError) as exc:
        schemas.validate(df, schemas.ATTRIBUTES)
    assert "missing columns" in str(exc.value)
    assert "unexpected columns" in str(exc.value)


def test_empty_frame_round_trips():
    for schema in schemas.ALL_SCHEMAS.values():
        schemas.validate(schema.empty(), schema)


def test_holdout_base_rate_beats_uniform(holdout):
    # If the target were uniform, accuracy-vs-chance would be untestable.
    assert holdout["base_rate"].iloc[0] > 1.0 / holdout["n_options"].iloc[0]


def test_nomination_day_has_realistic_missingness(attributes_large):
    frac = attributes_large["nomination_day"].isna().mean()
    assert 0.05 < frac < 0.30


def test_heldout_vars_correlate_with_seg_vars(attributes_large):
    # The fixture must have exploitable structure, otherwise a broken frame
    # would score the same as a good one and figure 1 would prove nothing.
    r = attributes_large["deprivation_score"].corr(attributes_large["majority_pct"])
    assert abs(r) > 0.2


def test_make_attributes_is_deterministic():
    a = synthetic.make_attributes(n=25, seed=7)
    b = synthetic.make_attributes(n=25, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_seg_vars_contain_no_post_cutoff_outcome():
    """Stratifying on the outcome would make distribution accuracy trivially perfect.

    The original design brief listed nomination_day as both a construction
    attribute and the primary validation target. That is circular: a frame raked
    on the outcome reproduces the outcome marginal by construction, so 1-TVD
    reads ~1.0 regardless of whether the method works at all.
    """
    assert not set(schemas.SEG_VARS) & set(schemas.POST_CUTOFF_VARS)


def test_post_cutoff_vars_all_exist():
    assert set(schemas.POST_CUTOFF_VARS) <= set(schemas.ATTRIBUTES.names)


def test_did_nominate_distinguishes_declined_from_unsourced(attributes_large):
    """NaN in nomination_day must not be the only signal for 'declined'.

    Declining to nominate a near-unanimous winner is the informative cell; an
    unsourced row is missing data. Conflating them would put the study's most
    discriminating signal into the same bucket as its noise.
    """
    df = attributes_large
    declined = df["did_nominate"] == False
    assert declined.sum() > 0
    assert (df.loc[declined, "nomination_bucket"] == "none").all()
    assert df.loc[declined, "nomination_day"].isna().all()
