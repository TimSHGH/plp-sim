"""Manifesto injection and the registry override.

Four scripts used to reassign ``P2CSituated.body`` at runtime and restore it on
the last line of the function. That mutates module-global state, duplicates the
same lambda four times, and leaves the class permanently patched for the rest of
the process if anything raises before the restore runs.

These tests pin the replacement: composition for the manifesto's content, a
context manager for the swap.
"""

from __future__ import annotations

import pandas as pd
import pytest

from plp_sim import dossier

MANIFESTO = {"payroll": "\n\nPAYROLL LINE.", "backbench": "\n\nBACKBENCH LINE."}


def _persona(*, payroll: bool) -> pd.Series:
    """A minimal PERSONA row, built here rather than borrowed from a fixture so
    the test states exactly which fields the renderer depends on."""
    return pd.Series({
        "persona_id": 1, "method": "P2C", "weight": 4.05,
        "majority_pct": 12.3, "vote_share": 44.1, "runner_up_party": "Reform UK",
        "is_2024_intake": False, "is_payroll": payroll, "role": "Minister of State",
        "committee_count": 1, "rebellion_rate": 0.04, "speech_count": 120,
        "seat_at_risk_share": 0.8, "projected_winner": "Reform UK",
        "projected_labour_share": 21.0, "national_context": "- Labour 19%.",
    })


def test_manifesto_is_chosen_by_the_personas_own_payroll_status():
    r = dossier.P2CSituated(manifesto=MANIFESTO)
    on_payroll = r.render(_persona(payroll=True))
    backbench = r.render(_persona(payroll=False))

    assert "PAYROLL LINE." in on_payroll
    assert "BACKBENCH LINE." not in on_payroll
    assert "BACKBENCH LINE." in backbench
    assert "PAYROLL LINE." not in backbench


def test_no_manifesto_means_no_extra_text():
    plain = dossier.P2CSituated().render(_persona(payroll=True))
    assert "PAYROLL LINE." not in plain
    assert "Where you stand" not in plain


def test_renderers_do_not_share_state():
    """The old approach patched the class, so every instance changed at once."""
    with_man = dossier.P2CSituated(manifesto=MANIFESTO)
    without = dossier.P2CSituated()
    assert "PAYROLL LINE." in with_man.render(_persona(payroll=True))
    assert "PAYROLL LINE." not in without.render(_persona(payroll=True))


def test_the_situational_block_survives_the_manifesto():
    """The manifesto is appended, not substituted for the context."""
    out = dossier.P2CSituated(manifesto=MANIFESTO).render(_persona(payroll=False))
    assert "The current political situation" in out
    assert "Labour 19%" in out
    assert out.index("The current political situation") < out.index("BACKBENCH LINE.")


def test_using_restores_the_registry():
    before = dossier.METHOD_REGISTRY["P2C"]
    with dossier.using("P2C", dossier.P2CSituated(manifesto=MANIFESTO)):
        assert dossier.METHOD_REGISTRY["P2C"] is not before
    assert dossier.METHOD_REGISTRY["P2C"] is before


def test_using_restores_even_when_the_body_raises():
    """The whole reason this is a context manager and not two statements."""
    before = dossier.METHOD_REGISTRY["P2C"]
    with pytest.raises(RuntimeError), \
            dossier.using("P2C", dossier.P2CSituated(manifesto=MANIFESTO)):
        raise RuntimeError("boom")
    assert dossier.METHOD_REGISTRY["P2C"] is before


def test_build_dossier_uses_the_overridden_renderer():
    persona = _persona(payroll=True)
    assert "PAYROLL LINE." not in dossier.build_dossier("P2C", persona)
    with dossier.using("P2C", dossier.P2CSituated(manifesto=MANIFESTO)):
        assert "PAYROLL LINE." in dossier.build_dossier("P2C", persona)
    assert "PAYROLL LINE." not in dossier.build_dossier("P2C", persona)
