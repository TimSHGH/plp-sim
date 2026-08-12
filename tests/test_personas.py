"""Tests for the persona ladder.

The properties worth testing here are the ones a wrong implementation would
pass by accident, and above all the one the whole reframe rests on: that a
"synthetic" persona is actually synthetic. A quota generator that happened to
reproduce real MPs would look identical from the outside, score well, and be
measuring retrieval.
"""

from __future__ import annotations

import pandas as pd
import pytest

from plp_sim import frames, personas, schemas
from plp_sim.config import Settings
from tests.fixtures import synthetic

SYNTHETIC_METHODS = ("P0", "P1", "P2")
REAL_METHODS = ("P4", "P5")


@pytest.fixture(scope="module")
def attrs():
    return synthetic.make_attributes(n=400, seed=1)


@pytest.fixture(scope="module")
def cfg():
    return Settings(n_personas=100, random_seed=0)


@pytest.fixture(scope="module")
def frame(attrs, cfg):
    return frames.build_f1(attrs, cfg)


@pytest.fixture(scope="module")
def panels(attrs, cfg, frame):
    out = {m: personas.build_panel(m, attrs, cfg) for m in SYNTHETIC_METHODS}
    out |= {m: personas.build_panel(m, attrs, cfg, frame=frame) for m in REAL_METHODS}
    return out


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", [*SYNTHETIC_METHODS, *REAL_METHODS])
def test_panel_conforms_to_schema(panels, method):
    schemas.validate(panels[method], schemas.PERSONA)


@pytest.mark.parametrize("method", [*SYNTHETIC_METHODS, *REAL_METHODS])
def test_weights_sum_to_population(panels, attrs, method):
    assert panels[method]["weight"].sum() == pytest.approx(len(attrs))


# --------------------------------------------------------------------------
# the property the reframe depends on
# --------------------------------------------------------------------------


def test_synthetic_personas_have_no_member_id(panels):
    """A member_id on a synthetic persona means a real person leaked in."""
    for m in SYNTHETIC_METHODS:
        assert panels[m]["member_id"].isna().all(), f"{m} carries a real member_id"


def test_no_synthetic_persona_reproduces_a_real_mp(panels, attrs):
    """The load-bearing test.

    If a quota draw happened to reconstruct a real MP's full attribute vector,
    that persona would be a person, and any accuracy on it would be
    indistinguishable from the model recalling that individual. Independent
    per-attribute draws are what prevent it, and this asserts they do.
    """
    cols = list(personas.PERSONA_ATTRS)
    real = {tuple(r) for r in attrs[cols].astype(str).itertuples(index=False, name=None)}
    for m in ("P1", "P2"):
        syn = [tuple(r) for r in panels[m][cols].astype(str).itertuples(index=False, name=None)]
        assert not [r for r in syn if r in real], f"{m} reproduced a real MP's vector"


def test_real_methods_do_carry_member_ids(panels):
    """P4/P5 are the ceiling precisely because they need a real person."""
    for m in REAL_METHODS:
        assert panels[m]["member_id"].notna().all()


def test_only_p5_carries_a_name(panels):
    """The name is the entire difference between the two ceiling methods."""
    assert panels["P5"]["name"].notna().all()
    assert panels["P4"]["name"].isna().all()


# --------------------------------------------------------------------------
# representativeness
# --------------------------------------------------------------------------


def test_p1_reproduces_population_margins(panels, attrs):
    p1 = panels["P1"]
    total = p1["weight"].sum()
    for col in ("runner_up_party", "is_payroll", "is_2024_intake"):
        pop = attrs[col].value_counts(normalize=True)
        panel = p1.groupby(col)["weight"].sum() / total
        gap = (panel.reindex(pop.index).fillna(0.0) - pop).abs().max()
        assert gap < 0.08, f"{col} margin off by {gap:.3f}"


def test_p1_beats_a_deliberately_skewed_panel(panels, attrs):
    """A representativeness check that can actually fail.

    Asserting only that P1's margins are close proves little unless a bad panel
    would miss. This builds one (the safest 100 seats) and requires P1 to be
    closer on majority_pct.
    """
    p1 = panels["P1"]
    p1_mean = (p1["majority_pct"] * p1["weight"]).sum() / p1["weight"].sum()
    bad_mean = attrs.nlargest(100, "majority_pct")["majority_pct"].mean()
    pop_mean = attrs["majority_pct"].mean()
    assert abs(p1_mean - pop_mean) < abs(bad_mean - pop_mean)


def test_p0_is_a_single_persona(panels, attrs):
    """A stereotype has no within-panel variation, so one row is the honest
    representation. Emitting 100 identical copies would manufacture a spread
    that is sampling noise rather than modelled disagreement."""
    assert len(panels["P0"]) == 1
    assert panels["P0"]["weight"].iloc[0] == pytest.approx(len(attrs))


def test_p2_archetypes_are_not_simply_medoid_members(panels, attrs):
    """P2 must describe a cluster, not the member nearest its centre.

    Taking the medoid would be person simulation wearing a persona label.
    """
    cols = ["majority_pct", "vote_share"]
    real = {tuple(round(float(v), 4) for v in r)
            for r in attrs[cols].itertuples(index=False, name=None)}
    arche = [tuple(round(float(v), 4) for v in r)
             for r in panels["P2"][cols].itertuples(index=False, name=None)]
    assert sum(a in real for a in arche) < len(arche) * 0.2


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", SYNTHETIC_METHODS)
def test_panels_are_deterministic(attrs, cfg, method):
    a = personas.build_panel(method, attrs, cfg)
    b = personas.build_panel(method, attrs, cfg)
    pd.testing.assert_frame_equal(a, b)


def test_build_panel_rejects_unknown_method(attrs, cfg):
    with pytest.raises(ValueError, match="no panel builder"):
        personas.build_panel("NOPE", attrs, cfg)


def test_real_methods_require_a_frame(attrs, cfg):
    with pytest.raises(ValueError, match="needs a frame"):
        personas.build_panel("P4", attrs, cfg)


# --------------------------------------------------------------------------
# P3 prompt hygiene
# --------------------------------------------------------------------------


def test_biography_prompt_carries_no_identifying_detail(panels):
    """The generator must not be handed a name or seat to build prose around."""
    prompt = personas.biography_prompt(panels["P2"].iloc[0])
    assert "Member " not in prompt and "Constituency " not in prompt


def test_biography_system_prompt_forbids_invention():
    """P3's whole risk is inventing detail the vector never contained."""
    s = personas.BIOGRAPHY_SYSTEM.lower()
    assert "do not invent" in s
    for banned in ("name", "gender", "ethnicity"):
        assert banned in s
