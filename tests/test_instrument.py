"""The instrument is a contract with the holdout, and it fails silently.

An `outcome_map` naming a label that does not exist in holdout.parquet raises
nothing. `reindex` fills the missing rows with NaN, `fillna(0)` turns them into
zeros, and the observed marginal becomes all-zero -- at which point every method
scores an identical, plausible-looking 1-TVD and nobody notices. That happened:
`v_nomination` was written with `[early, late]` against real labels
`[day1_bandwagon, held_back]` and produced 0.500 for all seven methods.

These tests are cheap and they run against the real artefacts, because the bug
only exists in the relationship between two files.
"""

from __future__ import annotations

import pandas as pd
import pytest

from plp_sim import elicit
from plp_sim.config import get_settings

HOLDOUT = get_settings().data_processed / "holdout.parquet"
requires_holdout = pytest.mark.skipif(
    not HOLDOUT.exists(), reason="holdout.parquet not built in this checkout"
)


@pytest.fixture(scope="module")
def instrument() -> dict:
    return elicit.load_instrument()


def test_every_item_is_internally_consistent(instrument):
    for it in instrument["items"]:
        n = len(it["options"])
        assert len(it["tokens"]) == n, f"{it['id']}: tokens/options length mismatch"
        assert len(it["outcome_map"]) == n, f"{it['id']}: outcome_map/options mismatch"
        assert len(set(it["tokens"])) == n, f"{it['id']}: duplicate tokens"


def test_no_option_was_coerced_to_a_bool(instrument):
    """YAML 1.1 turns a bare `No` into False, and the model is then shown the
    literal string "False" as an answer. This nearly shipped."""
    for it in instrument["items"]:
        for o in it["options"]:
            assert isinstance(o, str), f"{it['id']}: option {o!r} is {type(o).__name__}, quote it"


def test_tokens_are_single_tokens_after_strip(instrument):
    for it in instrument["items"]:
        for t in it["tokens"]:
            assert isinstance(t, str) and len(t.strip()) == 1, f"{it['id']}: bad token {t!r}"


@requires_holdout
def test_outcome_maps_match_the_real_holdout_labels(instrument):
    holdout = pd.read_parquet(HOLDOUT)
    for it in instrument["items"]:
        actual = set(holdout.loc[holdout["event_id"] == it["holdout_event"], "outcome"])
        assert actual, f"{it['id']}: no rows for event {it['holdout_event']!r}"
        declared = set(it["outcome_map"])
        assert declared == actual, (
            f"{it['id']}: outcome_map {sorted(declared)} does not match holdout "
            f"labels {sorted(actual)} -- this would silently score 0.500 everywhere"
        )


@requires_holdout
def test_observed_marginal_is_never_degenerate(instrument):
    """The symptom the silent failure produces, asserted directly."""
    holdout = pd.read_parquet(HOLDOUT)
    for it in instrument["items"]:
        ev = holdout[holdout["event_id"] == it["holdout_event"]]
        obs = ev["outcome"].value_counts(normalize=True).reindex(it["outcome_map"]).fillna(0)
        assert obs.sum() == pytest.approx(1.0), f"{it['id']}: observed marginal sums to {obs.sum()}"
        assert (obs > 0).all(), f"{it['id']}: some outcome has zero observed mass"


@requires_holdout
def test_pressure_panel_points_at_a_real_item(instrument):
    pp = instrument["pressure_panel"]
    ids = {it["id"] for it in instrument["items"]}
    assert pp["base_item"] in ids
    assert len(pp["levels"]) >= 2
    assert pp["levels"][0]["social"] == "", "the first level must be the no-pressure baseline"
    # only the trailing sentence may differ between levels
    assert len({lv["id"] for lv in pp["levels"]}) == len(pp["levels"]), "duplicate level ids"
