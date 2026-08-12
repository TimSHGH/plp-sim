"""Offline tests for :mod:`plp_sim.elicit` and :mod:`plp_sim.dossier`.

Everything here runs with no network and no API key: the OpenAI client is
always the small fake object built below (``make_client``), never a real
``AsyncOpenAI``. There is no ``pytest-asyncio`` in this project's dependency
set, so async code under test is driven with a plain ``asyncio.run(...)``
inside an ordinary ``def test_...`` function rather than an ``async def``
test with a marker.

Persona rows come from the shared ``attributes`` fixture in
``tests/conftest.py`` (``tests/fixtures/synthetic.py``, per the task brief).
"""

from __future__ import annotations

import asyncio
import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pandas as pd
import pytest
from tenacity import wait_none

from plp_sim import dossier, elicit, schemas
from plp_sim.config import Settings

# ---------------------------------------------------------------------------
# fixture data
# ---------------------------------------------------------------------------


@pytest.fixture
def personas(attributes) -> pd.DataFrame:
    return attributes.iloc[:12]


@pytest.fixture
def persona(personas) -> pd.Series:
    return personas.iloc[0]


@pytest.fixture
def instrument() -> dict:
    return elicit.load_instrument()


@pytest.fixture
def item(instrument) -> dict:
    return elicit.item_by_id(instrument, "v_loyalty")  # tokens A, B, C


@pytest.fixture
def cfg(tmp_path) -> Settings:
    return Settings(
        data_interim=tmp_path / "interim",
        cache_dir=tmp_path / "cache",
        model="gpt-4o-mini",
        top_logprobs=20,
        max_concurrency=4,
        max_retries=3,
        prompt_version="v1",
    )


# ---------------------------------------------------------------------------
# a fake OpenAI client -- only `client.chat.completions.create(...)` is used
# ---------------------------------------------------------------------------


def _logprob_response(top_logprobs: dict[str, float], sampled_token: str | None = None):
    """Build a fake Chat Completions response exposing exactly the attribute
    path `elicit.extract_top_logprobs` reads:
    `response.choices[0].logprobs.content[0].top_logprobs[i].{token,logprob}`.
    """
    sampled_token = sampled_token or max(top_logprobs, key=top_logprobs.__getitem__)
    top = [SimpleNamespace(token=t, logprob=lp) for t, lp in top_logprobs.items()]
    token_logprob = SimpleNamespace(
        token=sampled_token,
        logprob=top_logprobs.get(sampled_token, -9999.0),
        top_logprobs=top,
    )
    logprobs = SimpleNamespace(content=[token_logprob])
    return SimpleNamespace(choices=[SimpleNamespace(logprobs=logprobs)])


def _sample_response(token: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=token))])


def make_client(*, return_value=None, side_effect=None) -> SimpleNamespace:
    """A minimal duck-typed stand-in for `AsyncOpenAI`."""
    create = AsyncMock()
    if side_effect is not None:
        create.side_effect = side_effect
    if return_value is not None:
        create.return_value = return_value
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


# ---------------------------------------------------------------------------
# (a) tokenisation
# ---------------------------------------------------------------------------


def test_extract_top_logprobs_normalises_whitespace():
    response = _logprob_response({" A": -0.1, "B ": -2.0})
    result = elicit.extract_top_logprobs(response)
    # approx, not exact: colliding tokens are now merged in probability space,
    # so every logprob makes a log->exp->log round trip and -0.1 comes back as
    # -0.10000000000000006. Merging is required for correctness (see
    # test_colliding_tokens_have_their_mass_summed_not_overwritten); demanding
    # bit-exact floats here would just re-break that.
    assert result.keys() == {"A", "B"}
    assert result["A"] == pytest.approx(-0.1)
    assert result["B"] == pytest.approx(-2.0)


def test_probe_model_passes_when_all_tokens_present(cfg, instrument):
    all_tokens = sorted({t for it in instrument["items"] for t in it["tokens"]})
    top_logprobs = {t: math.log(1.0 / len(all_tokens)) for t in all_tokens}
    client = make_client(return_value=_logprob_response(top_logprobs))

    result = asyncio.run(elicit.probe_model(client, cfg, instrument))
    assert set(all_tokens) <= set(result)


def test_probe_model_raises_when_a_token_never_surfaces(cfg, instrument):
    top_logprobs = {"A": math.log(0.9), "B": math.log(0.1)}  # C, D never appear
    client = make_client(return_value=_logprob_response(top_logprobs))

    with pytest.raises(AssertionError, match="never appeared"):
        asyncio.run(elicit.probe_model(client, cfg, instrument))


# ---------------------------------------------------------------------------
# (b) logprob -> probability, captured mass
# ---------------------------------------------------------------------------


def test_probs_from_top_logprobs_hand_computed():
    top_logprobs = {"A": math.log(0.5), "B": math.log(0.3), "C": math.log(0.1)}
    probs, captured_mass = elicit.probs_from_top_logprobs(top_logprobs, ["A", "B", "C", "D"])

    assert captured_mass == pytest.approx(0.9)
    assert probs == pytest.approx([0.5 / 0.9, 0.3 / 0.9, 0.1 / 0.9, 0.0])
    assert sum(probs) == pytest.approx(1.0)


def test_captured_mass_is_computed_before_renormalising():
    # Raw mass is 0.4, well under 1 -- probs must still sum to 1 afterwards.
    top_logprobs = {"A": math.log(0.25), "B": math.log(0.15)}
    probs, captured_mass = elicit.probs_from_top_logprobs(top_logprobs, ["A", "B", "C"])

    assert captured_mass == pytest.approx(0.40)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[2] == 0.0  # C never appeared


def test_missing_option_token_is_zero_not_dropped():
    top_logprobs = {"A": math.log(0.9)}  # B, C, D never appear
    probs, captured_mass = elicit.probs_from_top_logprobs(top_logprobs, ["A", "B", "C", "D"])

    assert len(probs) == 4  # nothing silently dropped
    assert probs[1:] == [0.0, 0.0, 0.0]
    assert captured_mass == pytest.approx(0.9)


def test_zero_captured_mass_does_not_crash():
    probs, captured_mass = elicit.probs_from_top_logprobs({}, ["A", "B", "C"])
    assert captured_mass == 0.0
    assert probs == [0.0, 0.0, 0.0]


def test_low_captured_mass_is_flagged_not_swallowed():
    with pytest.warns(elicit.LowCapturedMassWarning, match="low captured_mass"):
        elicit.check_captured_mass(0.10, method="P4", item_id="v_loyalty", persona_id=1)
    assert 0.10 < schemas.MIN_CAPTURED_MASS


def test_healthy_captured_mass_does_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        elicit.check_captured_mass(0.95, method="P4", item_id="v_loyalty", persona_id=1)


# ---------------------------------------------------------------------------
# (d) reversed option order
# ---------------------------------------------------------------------------


def test_reversing_twice_is_the_identity():
    values = [0.1, 0.2, 0.3, 0.4]
    once = elicit.realign_to_forward("reversed", values)
    twice = elicit.realign_to_forward("reversed", once)
    assert twice == values
    assert once != values  # the intermediate step actually did something


def test_forward_order_is_a_no_op():
    values = [0.1, 0.2, 0.3, 0.4]
    assert elicit.realign_to_forward("forward", values) == values


def test_displayed_options_reversed_matches_re_lettering(item):
    forward = elicit.displayed_options(item, "forward")
    reversed_ = elicit.displayed_options(item, "reversed")
    assert forward == item["options"]
    assert reversed_ == list(reversed(item["options"]))
    assert list(reversed(reversed_)) == forward


def test_elicit_item_stores_probs_against_forward_order_for_reversed_arm(cfg, persona, item):
    # All mass on letter "A". Under the reversed arm, letter A is re-lettered
    # onto the LAST forward option -- so the stored, forward-order probs must
    # peak at the last index, not the first.
    top_logprobs = {
        "A": math.log(0.97),
        "B": math.log(0.01),
        "C": math.log(0.01),
        "D": math.log(0.01),
    }
    client = make_client(return_value=_logprob_response(top_logprobs))

    row = asyncio.run(
        elicit.elicit_item(
            client,
            cfg,
            method="P4",
            frame="F2",
            persona=persona,
            item=item,
            option_order="reversed",
        )
    )
    n = len(item["options"])
    assert row["probs"][n - 1] == max(row["probs"])
    assert row["top_option"] == item["options"][n - 1]


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def test_cache_hit_makes_zero_client_calls(cfg, persona, item):
    cache = elicit.open_cache(cfg.cache_dir)
    top_logprobs = {
        "A": math.log(0.7),
        "B": math.log(0.2),
        "C": math.log(0.05),
        "D": math.log(0.02),
    }
    client = make_client(return_value=_logprob_response(top_logprobs))

    async def go():
        first = await elicit.elicit_item(
            client, cfg, method="P4", frame="F2", persona=persona, item=item, cache=cache
        )
        second = await elicit.elicit_item(
            client, cfg, method="P4", frame="F2", persona=persona, item=item, cache=cache
        )
        return first, second

    first, second = asyncio.run(go())

    assert client.chat.completions.create.await_count == 1  # second was a cache hit
    assert first["cached"] is False
    assert second["cached"] is True
    assert first["probs"] == second["probs"]


def test_changing_prompt_version_alone_misses_cache(cfg, persona, item):
    cache = elicit.open_cache(cfg.cache_dir)
    persona_id = int(persona["member_id"])
    key_v1 = elicit.cache_key(
        method="P4",
        persona_id=persona_id,
        item_id=item["id"],
        model=cfg.model,
        prompt_version="v1",
        option_order="forward",
        draw_index=0,
    )
    key_v2 = elicit.cache_key(
        method="P4",
        persona_id=persona_id,
        item_id=item["id"],
        model=cfg.model,
        prompt_version="v2",
        option_order="forward",
        draw_index=0,
    )

    assert key_v1 != key_v2
    cache[key_v1] = {"top_logprobs": {"A": -0.1}}
    assert cache.get(key_v1) is not None  # sanity: the v1 entry really is there
    assert cache.get(key_v2) is None  # prompt_version changed -> miss


def test_cache_key_changes_with_each_component():
    base = {
        "method": "P4",
        "persona_id": 1,
        "item_id": "x",
        "model": "gpt-4o-mini",
        "prompt_version": "v1",
        "option_order": "forward",
        "draw_index": 0,
    }
    baseline = elicit.cache_key(**base)

    for field, other in [
        ("method", "P5"),
        ("persona_id", 2),
        ("item_id", "y"),
        ("model", "gpt-4o"),
        ("prompt_version", "v2"),
        ("option_order", "reversed"),
        ("draw_index", 1),
    ]:
        changed = elicit.cache_key(**{**base, field: other})
        assert changed != baseline, f"{field} did not change the cache key"

    assert elicit.cache_key(**base) == baseline  # deterministic


# ---------------------------------------------------------------------------
# (c) prefix caching precondition -- byte-identical system prompt
# ---------------------------------------------------------------------------


def test_system_prompt_byte_identical_across_items(instrument, persona):
    rendered = {
        it["id"]: dossier.build_dossier("P4", persona, prompt_version="v1")
        for it in instrument["items"]
    }
    values = list(rendered.values())
    assert len(set(values)) == 1, "dossier for a fixed (method, persona) must not vary by item"
    # and no accidental leakage of one item's text into the shared dossier
    for it in instrument["items"]:
        assert it["text"].strip() not in values[0]


# ---------------------------------------------------------------------------
# dossier content: RECALL leakage control, length parity
# ---------------------------------------------------------------------------


def test_recall_has_no_persona_attributes_or_biography(persona):
    text = dossier.build_dossier("RECALL", persona)
    lowered = text.lower()

    for forbidden in (
        "majority",
        "vote share",
        "rebellion",
        "committee",
        "payroll",
        "intake",
        "nomination",
        "hansard",
        "speeches recorded",
        "safe seat",
        "marginal seat",
        "backbencher",
    ):
        assert forbidden not in lowered, f"RECALL leaked a biography/attribute term: {forbidden!r}"
    assert f"{float(persona['majority_pct']):.1f}" not in text

    # It DOES name the real MP -- that's the whole point of the leakage control.
    assert persona["name"] in text
    assert persona["constituency"] in text


#: Methods whose dossier is an attribute record. Parity is only meaningful
#: across these: P0 carries no attributes by design, P3 carries generated prose,
#, and RECALL carries no persona at all, so holding those to the same length
#: would be measuring nothing.
_PARITY_METHODS = ("P1", "P2", "P4", "P5")


def test_dossier_length_parity_across_attribute_methods(attributes):
    """Length parity so the comparison measures information, not prompt size."""
    worst = 0.0
    for _, row in attributes.iterrows():
        lens = dossier.dossier_lengths(row, _PARITY_METHODS)
        spread = (max(lens.values()) - min(lens.values())) / min(lens.values())
        worst = max(worst, spread)
    assert worst <= 0.15, f"dossier length spread {worst:.1%} exceeds the 15% budget"


def test_p0_carries_no_persona_detail(attributes):
    """The stereotype baseline must be identical for every persona.

    If P0 varied by persona it would not be a stereotype, and the baseline
    would be measuring grounding it is supposed to lack.
    """
    rendered = {dossier.build_dossier("P0", row) for _, row in attributes.iterrows()}
    assert len(rendered) == 1


def test_p3_refuses_to_render_without_a_biography(attributes):
    """Silently falling back to P2 would make the two conditions identical
    while still being reported as different methods."""
    with pytest.raises(ValueError, match="no biography"):
        dossier.build_dossier("P3", attributes.iloc[0])


def test_build_dossier_rejects_unknown_method(persona):
    with pytest.raises(ValueError, match="unknown method"):
        dossier.build_dossier("BOGUS", persona)


# ---------------------------------------------------------------------------
# end-to-end: schema conformance
# ---------------------------------------------------------------------------


def test_elicit_item_output_conforms_to_schema(cfg, persona, item):
    top_logprobs = {
        "A": math.log(0.6),
        "B": math.log(0.25),
        "C": math.log(0.1),
        "D": math.log(0.03),
    }
    client = make_client(return_value=_logprob_response(top_logprobs))

    row = asyncio.run(
        elicit.elicit_item(client, cfg, method="P5", frame="F1", persona=persona, item=item)
    )
    df = elicit.to_dataframe([row])
    schemas.validate(df, schemas.ELICITATION)


def test_run_elicitation_output_conforms_to_schema(cfg, instrument, personas):
    top_logprobs = {
        "A": math.log(0.6),
        "B": math.log(0.25),
        "C": math.log(0.1),
        "D": math.log(0.03),
    }
    client = make_client(return_value=_logprob_response(top_logprobs))
    small_personas = personas.iloc[:3]
    # derived, not hardcoded: the instrument is a live design document and its
    # item count changes when the study's questions change. A test that assumes
    # a fixed count fails on a legitimate edit and says nothing about the code.
    small_instrument = {**instrument, "items": instrument["items"][:2]}
    n_items = len(small_instrument["items"])

    df = asyncio.run(
        elicit.run_elicitation(
            client,
            cfg,
            instrument=small_instrument,
            personas=small_personas,
            methods=["P4", "RECALL"],
            frame="F2",
        )
    )
    schemas.validate(df, schemas.ELICITATION)
    assert len(df) == 3 * n_items * 2  # personas x items x methods, forward order only
    assert set(df["method"]) == {"P4", "RECALL"}


def test_sample_draws_output_conforms_to_schema(cfg, persona, item):
    client = make_client(return_value=_sample_response("B"))

    df = asyncio.run(
        elicit.sample_draws(client, cfg, method="P4", frame="F2", persona=persona, item=item, n=3)
    )
    schemas.validate(df, schemas.ELICITATION)
    assert list(df["draw_index"]) == [0, 1, 2]
    assert (df["temperature"] == cfg.dispersion_temperature).all()
    assert (df["top_option"] == item["options"][item["tokens"].index("B")]).all()


# ---------------------------------------------------------------------------
# call logging
# ---------------------------------------------------------------------------


def test_calls_are_logged_to_calls_jsonl(cfg, persona, item):
    top_logprobs = {
        "A": math.log(0.9),
        "B": math.log(0.05),
        "C": math.log(0.03),
        "D": math.log(0.02),
    }
    client = make_client(return_value=_logprob_response(top_logprobs))

    asyncio.run(
        elicit.elicit_item(client, cfg, method="P4", frame="F2", persona=persona, item=item)
    )

    log_path = cfg.data_interim / "calls.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["item_id"] == item["id"]
    assert record["method"] == "P4"


# ---------------------------------------------------------------------------
# retry on 429 / 5xx
# ---------------------------------------------------------------------------


def test_retries_on_rate_limit_then_succeeds(cfg, monkeypatch):
    # Patch tenacity's wait strategy to zero so the test doesn't sleep.
    monkeypatch.setattr(elicit, "wait_exponential", lambda *a, **k: wait_none())

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    rate_limited = openai.RateLimitError(
        "slow down", response=httpx.Response(429, request=request), body=None
    )
    ok_response = _logprob_response({"A": math.log(0.9), "B": math.log(0.1)})
    client = make_client(side_effect=[rate_limited, ok_response])

    async def go():
        return await elicit._create_chat_completion(
            client,
            cfg,
            messages=[{"role": "user", "content": "hi"}],
            logprobs=True,
            top_logprobs=5,
        )

    response = asyncio.run(go())
    assert response is ok_response
    assert client.chat.completions.create.await_count == 2


def test_non_retryable_error_raises_immediately(cfg):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    bad_request = openai.BadRequestError(
        "nope", response=httpx.Response(400, request=request), body=None
    )
    client = make_client(side_effect=[bad_request])

    async def go():
        return await elicit._create_chat_completion(
            client,
            cfg,
            messages=[{"role": "user", "content": "hi"}],
            logprobs=True,
            top_logprobs=5,
        )

    with pytest.raises(openai.BadRequestError):
        asyncio.run(go())
    assert client.chat.completions.create.await_count == 1  # no retry attempted


# --------------------------------------------------------------------------
# regression: logprob vs sampled-draw cache collision
# --------------------------------------------------------------------------


def test_cache_key_separates_logprob_from_sampled_draw():
    """A logprob call and dispersion draw 0 must not share a cache entry.

    The original spec omitted `temperature` from the key, so both produced an
    identical hash and the first sampled draw read back the deterministic
    logprob answer. That biases within-persona entropy toward zero: it
    manufactures the very under-dispersion the calibration step exists to
    measure, which is the worst direction for a bug to point.
    """
    common = {
        "method": "P4",
        "persona_id": 100_001,
        "item_id": "v_loyalty",
        "model": "gpt-4o-mini",
        "prompt_version": "v1",
        "option_order": "forward",
        "draw_index": 0,
    }
    logprob_key = elicit.cache_key(**common, temperature=None)
    sampled_key = elicit.cache_key(**common, temperature=1.0)
    assert logprob_key != sampled_key


def test_cache_key_separates_temperatures():
    common = {
        "method": "P4",
        "persona_id": 100_001,
        "item_id": "v_loyalty",
        "model": "gpt-4o-mini",
        "prompt_version": "v1",
        "option_order": "forward",
        "draw_index": 3,
    }
    assert elicit.cache_key(**common, temperature=0.7) != elicit.cache_key(
        **common, temperature=1.0
    )


# --------------------------------------------------------------------------
# regressions found by adversarial review
# --------------------------------------------------------------------------


def _fake_logprob_response(raw_token_probs):
    """Response whose top_logprobs carries RAW (un-stripped) tokens."""
    from types import SimpleNamespace
    top = [
        SimpleNamespace(token=t, logprob=math.log(p))
        for t, p in raw_token_probs.items()
    ]
    return SimpleNamespace(
        choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=top)]))]
    )


def test_colliding_tokens_have_their_mass_summed_not_overwritten():
    """" A" and "A" both normalise to "A"; their mass must add.

    A dict comprehension keeps whichever came last. With 0.50 on " A" and 0.05
    on "A", last-wins keeps 0.05 and flips the reported answer from A to B --
    while captured_mass stays above MIN_CAPTURED_MASS so nothing warns.
    """
    resp = _fake_logprob_response({" A": 0.50, "A": 0.05, "B": 0.30, "C": 0.10, "D": 0.02})
    lp = elicit.extract_top_logprobs(resp)
    assert math.exp(lp["A"]) == pytest.approx(0.55, abs=1e-9)

    probs, captured = elicit.probs_from_top_logprobs(lp, ["A", "B", "C", "D"])
    assert probs[0] == pytest.approx(0.55 / 0.97, abs=1e-6)
    assert captured == pytest.approx(0.97, abs=1e-6)
    # the whole point: A, not B, is on top
    assert max(range(len(probs)), key=probs.__getitem__) == 0


def test_collision_merge_is_order_independent():
    a = elicit.extract_top_logprobs(_fake_logprob_response({" A": 0.5, "A": 0.05, "B": 0.45}))
    b = elicit.extract_top_logprobs(_fake_logprob_response({"A": 0.05, " A": 0.5, "B": 0.45}))
    assert math.exp(a["A"]) == pytest.approx(math.exp(b["A"]), abs=1e-12)


def test_cache_key_is_unambiguous_across_field_boundaries():
    """A separator-bearing field must not let two different tuples collide."""
    base = {
        "method": "P4", "persona_id": 1, "prompt_version": "v1",
        "option_order": "forward", "draw_index": 0, "temperature": None,
    }
    a = elicit.cache_key(item_id="foo|bar", model="X", **base)
    b = elicit.cache_key(item_id="foo", model="bar|X", **base)
    assert a != b


def test_captured_mass_is_clamped_at_one_within_float_error():
    """Real API calls overshoot 1.0 when all mass sits on the option tokens.

    The first live call returned 1.0000000578, which fails schemas.ELICITATION's
    [0, 1] bound. Summing exp() of near-zero logprobs does that; it is float
    error, not a modelling problem.
    """
    lp = {t: math.log(0.25) for t in "ABCD"}
    probs, captured = elicit.probs_from_top_logprobs(lp, list("ABCD"))
    assert captured <= 1.0
    assert sum(probs) == pytest.approx(1.0)


def test_captured_mass_above_float_error_raises():
    """Clamping must not hide a genuinely impossible value."""
    lp = {t: math.log(0.5) for t in "ABCD"}  # sums to 2.0
    with pytest.raises(ValueError, match="double-counting"):
        elicit.probs_from_top_logprobs(lp, list("ABCD"))
