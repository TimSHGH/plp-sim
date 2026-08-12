"""Reads answers as probability distributions rather than by sampling.

Each item is constrained to one output token (``max_tokens=1``) and the
distribution is read straight off ``logprobs``. One call gives the full spread
of opinion instead of a single draw, which removes sampling noise and costs an
order of magnitude less. Chat Completions and OpenAI only: the Responses API
exposes logprobs differently and Anthropic exposes none.

Four things that bite if skipped:

* **Tokenisation.** "A" and " A" are different tokens. Normalise with
  ``.strip()`` and probe the model before trusting an option token.
* **Captured mass.** ``top_logprobs`` is truncated. Sum ``exp(logprob)`` over
  the option tokens *before* renormalising, and treat low mass as a finding.
* **Prefix caching.** OpenAI's cache is positional. The dossier is a long
  constant reused across items, so it must be byte-identical call to call, with
  the varying item text last.
* **Reversed option order.** The reversed arm re-letters the reversed list.
  Probabilities are always stored back against forward order so the two arms
  are comparable.

The HTTP boundary is a single injected ``client``, duck-typed to
``client.chat.completions.create``. Every test mocks it; no test needs a key.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import diskcache
import openai
import pandas as pd
import yaml
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from plp_sim import schemas
from plp_sim.config import Settings
from plp_sim.dossier import build_dossier

DEFAULT_INSTRUMENT_PATH = Path(__file__).resolve().parent.parent / "config" / "instrument.yaml"

#: 429 and 5xx only -- anything else (auth, bad request, content filter) is a
#: real error that retrying can't fix and should surface immediately.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class LowCapturedMassWarning(UserWarning):
    """Raised when option tokens accounted for too little of the truncated
    top-`k` distribution. Per schemas.MIN_CAPTURED_MASS: this means the model
    answered something other than the item -- a finding to surface, never a
    rounding error to silently renormalise away."""


# --------------------------------------------------------------------------
# instrument
# --------------------------------------------------------------------------


def load_instrument(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config/instrument.yaml``. Not memoised: it's tiny, and a cached
    copy that silently outlives an edit is worse than re-reading a few KB."""
    with open(path or DEFAULT_INSTRUMENT_PATH) as fh:
        instrument = yaml.safe_load(fh)
    return instrument


def item_by_id(instrument: dict[str, Any], item_id: str) -> dict[str, Any]:
    for item in instrument["items"]:
        if item["id"] == item_id:
            return item
    raise KeyError(f"no item {item_id!r} in instrument")


# --------------------------------------------------------------------------
# (d) option order
# --------------------------------------------------------------------------


def displayed_options(item: dict[str, Any], option_order: str) -> list[str]:
    """The option text in the order actually shown to the model."""
    if option_order not in schemas.OPTION_ORDERS:
        raise ValueError(f"unknown option_order {option_order!r}; expected {schemas.OPTION_ORDERS}")
    options = list(item["options"])
    return options if option_order == "forward" else list(reversed(options))


def render_item_text(item: dict[str, Any], option_order: str) -> str:
    """The user-message text for one item: the question, then its options
    lettered per ``item["tokens"]`` in ``option_order``. The instrument's
    ``tokens`` list is reused unchanged for both arms -- only which option
    text sits behind each letter changes, which is what "re-lettering" means
    for the reversed arm."""
    options = displayed_options(item, option_order)
    lines = [f"{tok}. {opt}" for tok, opt in zip(item["tokens"], options, strict=True)]
    return item["text"].strip() + "\n\n" + "\n".join(lines)


def realign_to_forward(option_order: str, values: list[Any]) -> list[Any]:
    """Map a per-option list from displayed order back to the item's forward
    order.

    The reversed arm displays ``list(reversed(options))`` behind the same
    token letters, so undoing it is exactly reversing the list again --
    reversing twice is the identity, which is the round-trip property
    ``tests/test_elicit.py`` checks. Forward order is passed through
    unchanged.
    """
    if option_order == "forward":
        return list(values)
    if option_order == "reversed":
        return list(reversed(values))
    raise ValueError(f"unknown option_order {option_order!r}; expected {schemas.OPTION_ORDERS}")


# --------------------------------------------------------------------------
# (a) + (b) logprobs -> probabilities
# --------------------------------------------------------------------------


def extract_top_logprobs(response: Any) -> dict[str, float]:
    """Pull ``{token.strip(): logprob}`` for the single generated token's
    alternatives, **summing probability mass across tokens that collide**.

    ``.strip()`` matters: "A" and " A" are different tokens and the instrument
    uses bare letters. But normalising two raw tokens onto one key means the
    key can collide, and a plain dict comprehension would keep whichever came
    last and silently discard the other's mass.

    That is not a rounding error. With raw tokens ``{" A": 0.50, "A": 0.05,
    "B": 0.30, ...}`` the true mass on option A is 0.55, but last-wins keeps
    0.05 and renormalises to A=0.11, B=0.64 -- reporting **B as the top answer
    when the model actually chose A**. Captured mass lands at 0.47, comfortably
    above ``MIN_CAPTURED_MASS``, so the diagnostic never fires: a well-formed,
    schema-valid, confidently wrong distribution. Since single-letter
    completions are exactly where BPE tokenisers emit both spaced and bare
    variants, this is a likely case, not a contrived one.

    Summing in probability space (not logprob space) is the only correct merge:
    P(A) = P(" A") + P("A").
    """
    top = response.choices[0].logprobs.content[0].top_logprobs
    merged: dict[str, float] = {}
    for entry in top:
        key = entry.token.strip()
        merged[key] = merged.get(key, 0.0) + math.exp(entry.logprob)
    return {k: math.log(v) if v > 0.0 else -math.inf for k, v in merged.items()}


def probs_from_top_logprobs(
    top_logprobs: dict[str, float], tokens: list[str]
) -> tuple[list[float], float]:
    """Convert a token -> logprob map into a renormalised probability vector
    aligned to ``tokens``, plus the captured mass BEFORE renormalisation.

    ``top_logprobs`` is a truncated (top-k) distribution, so
    ``sum(exp(logprob))`` over just the option tokens is generally < 1; that
    sum is ``captured_mass``, computed here before anything is renormalised.
    A token absent from ``top_logprobs`` contributes zero raw mass -- it is
    not dropped from the output, so a broken/never-surfacing option shows up
    as a structural zero in ``probs`` rather than silently vanishing.
    """
    raw = [math.exp(top_logprobs[t]) if t in top_logprobs else 0.0 for t in tokens]
    captured_mass = float(sum(raw))

    # When the model puts essentially all its mass on the option tokens -- the
    # normal, healthy case once the dossier's forced-choice instruction bites --
    # summing exp() of near-zero logprobs overshoots 1.0 in floating point.
    # The first real API call produced 1.0000000578, which fails the schema's
    # [0, 1] bound. Clamp only within tolerance: a genuinely impossible value
    # means the token map is wrong (double-counted collisions, say), and that
    # must surface rather than be quietly rounded into range.
    if captured_mass > 1.0:
        if captured_mass > 1.0 + 1e-6:
            raise ValueError(
                f"captured_mass={captured_mass!r} exceeds 1 by more than float error; "
                "the token->logprob map is probably double-counting."
            )
        captured_mass = 1.0

    if captured_mass > 0:
        total = float(sum(raw))
        probs = [r / total for r in raw]
    else:
        probs = [0.0 for _ in raw]
    return probs, captured_mass


def check_captured_mass(
    captured_mass: float, *, method: str, item_id: str, persona_id: int
) -> None:
    """Warn (never silently swallow) when ``captured_mass`` is suspiciously
    low. The value itself is still stored in the output row regardless --
    this is an operator-facing signal on top of that, not a substitute for
    it."""
    if captured_mass < schemas.MIN_CAPTURED_MASS:
        warnings.warn(
            f"low captured_mass={captured_mass:.4f} (< {schemas.MIN_CAPTURED_MASS}) for "
            f"method={method!r} item_id={item_id!r} persona_id={persona_id}: the model "
            "likely answered something other than the item.",
            LowCapturedMassWarning,
            stacklevel=2,
        )


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def cache_key(
    *,
    method: str,
    persona_id: int,
    item_id: str,
    model: str,
    prompt_version: str,
    option_order: str,
    draw_index: int,
    temperature: float | None = None,
    dossier_text: str | None = None,
) -> str:
    """sha256 over everything that determines a call's answer.

    ``frame`` is excluded on purpose: the same (method, persona, item) call gives
    the same answer whichever frame selected that persona, so F1 and F2 share
    entries.

    ``temperature`` is included. Without it a logprob call (temperature None) and
    sampled draw 0 (temperature 1.0) collide, and the first draw silently reads
    back the deterministic result, biasing every entropy estimate toward zero.

    ``dossier_text`` is included for the same reason. Identifying a persona by
    ``(method, persona_id)`` is not the same as identifying the prompt it produces:
    enriching a persona's content left both unchanged and silently replayed the old
    answers. Hashing the rendered system message makes the key cover what was
    actually sent.
    """
    # repr() + a separator that cannot appear unescaped in a repr. A plain
    # "|".join is ambiguous: item_id="foo|bar" with model="X" hashes the same
    # as item_id="foo" with model="bar|X". Inert today -- no current item id or
    # model string contains a pipe -- but a future model name like
    # "vendor|model" would silently serve one call's cached answer to another,
    # and a wrong-but-plausible cached probability is unfindable after the fact.
    payload = "\x1f".join(
        repr(part)
        for part in (
            method, persona_id, item_id, model, prompt_version,
            option_order, draw_index, temperature,
            None if dossier_text is None
            else hashlib.sha256(dossier_text.encode("utf-8")).hexdigest(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def open_cache(cache_dir: str | Path) -> diskcache.Cache:
    return diskcache.Cache(str(cache_dir))


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------

_PROBE_SYSTEM = "You answer multiple-choice survey questions with a single capital letter."
_PROBE_USER = "Respond with exactly one capital letter and nothing else."


async def probe_model(client: Any, cfg: Settings, instrument: dict[str, Any]) -> dict[str, float]:
    """Send one throwaway prompt and assert every option token used anywhere
    in the instrument actually appears in the returned top-``cfg.top_logprobs``.

    An option that never surfaces is a silently broken item, not a rare
    answer -- finding that out mid-run costs a day, so this runs once at
    startup, before any real elicitation call.
    """
    all_tokens = sorted({tok for item in instrument["items"] for tok in item["tokens"]})
    response = await _create_chat_completion(
        client,
        cfg,
        messages=[
            {"role": "system", "content": _PROBE_SYSTEM},
            {"role": "user", "content": _PROBE_USER},
        ],
        logprobs=True,
        top_logprobs=cfg.top_logprobs,
    )
    top_logprobs = extract_top_logprobs(response)
    missing = [t for t in all_tokens if t not in top_logprobs]
    if missing:
        raise AssertionError(
            f"probe failed: token(s) {missing} never appeared in the top-{cfg.top_logprobs} "
            f"for model {cfg.model!r}. Fix the instrument or the model before running the "
            "full elicitation -- this is exactly the silently-broken-item failure mode."
        )
    return top_logprobs


# --------------------------------------------------------------------------
# the OpenAI call itself, with retry
# --------------------------------------------------------------------------


async def _create_chat_completion(client: Any, cfg: Settings, *, messages: list[dict], **kwargs):
    async for attempt in AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(max(1, cfg.max_retries)),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=30.0),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    ):
        with attempt:
            return await client.chat.completions.create(
                model=cfg.model, messages=messages, max_tokens=1, **kwargs
            )


def _log_call(cfg: Settings, record: dict[str, Any]) -> None:
    """Append one call record to ``data_interim/calls.jsonl``. A wall-clock
    timestamp is fine here -- this is a log, not the system prompt that
    prefix caching depends on being byte-identical."""
    cfg.data_interim.mkdir(parents=True, exist_ok=True)
    line = {**record, "logged_at": datetime.now(UTC).isoformat()}
    with open(cfg.data_interim / "calls.jsonl", "a") as fh:
        fh.write(json.dumps(line, default=str) + "\n")


# --------------------------------------------------------------------------
# single-item logprob elicitation
# --------------------------------------------------------------------------


def _persona_key(persona: pd.Series) -> int:
    """The identifier for a persona row.

    PERSONA rows carry ``persona_id``; ATTRIBUTES rows carry ``member_id``.
    Synthetic personas have a NULL ``member_id`` on purpose -- there is no real
    MP behind them -- so reading member_id here would crash on exactly the
    methods the study is built around. Prefer persona_id and fall back only for
    the older ATTRIBUTES-shaped callers.
    """
    pid = persona.get("persona_id")
    if pid is not None and not pd.isna(pid):
        return int(pid)
    return int(persona["member_id"])


async def elicit_item(
    client: Any,
    cfg: Settings,
    *,
    method: str,
    frame: str,
    persona: pd.Series,
    item: dict[str, Any],
    option_order: str = "forward",
    draw_index: int = 0,
    cache: diskcache.Cache | None = None,
) -> dict[str, Any]:
    """Run (or fetch from cache) one logprob elicitation call and return a
    dict matching one row of ``schemas.ELICITATION``."""
    if method not in schemas.METHODS:
        raise ValueError(f"unknown method {method!r}; expected {schemas.METHODS}")
    if frame not in schemas.FRAMES:
        raise ValueError(f"unknown frame {frame!r}; expected {schemas.FRAMES}")

    persona_id = _persona_key(persona)
    item_id = item["id"]
    dossier = build_dossier(method, persona, prompt_version=cfg.prompt_version)
    key = cache_key(
        method=method,
        persona_id=persona_id,
        item_id=item_id,
        model=cfg.model,
        prompt_version=cfg.prompt_version,
        option_order=option_order,
        draw_index=draw_index,
        dossier_text=dossier,
    )

    cached_entry = cache.get(key) if cache is not None else None
    if cached_entry is not None:
        top_logprobs = cached_entry["top_logprobs"]
        latency_ms = None
        cached = True
    else:
        t0 = time.monotonic()
        item_text = render_item_text(item, option_order)
        response = await _create_chat_completion(
            client,
            cfg,
            messages=[
                {"role": "system", "content": dossier},
                {"role": "user", "content": item_text},
            ],
            logprobs=True,
            top_logprobs=cfg.top_logprobs,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0
        top_logprobs = extract_top_logprobs(response)
        if cache is not None:
            cache[key] = {"top_logprobs": top_logprobs}
        cached = False

    displayed_probs, captured_mass = probs_from_top_logprobs(top_logprobs, item["tokens"])
    forward_probs = realign_to_forward(option_order, displayed_probs)
    check_captured_mass(captured_mass, method=method, item_id=item_id, persona_id=persona_id)

    top_index = max(range(len(forward_probs)), key=forward_probs.__getitem__)
    row = {
        "method": method,
        "frame": frame,
        "persona_id": persona_id,
        "item_id": item_id,
        "option_order": option_order,
        "draw_index": int(draw_index),
        "model": cfg.model,
        "prompt_version": cfg.prompt_version,
        "temperature": None,
        "probs": forward_probs,
        "top_option": item["options"][top_index],
        "top_prob": float(forward_probs[top_index]),
        "captured_mass": float(captured_mass),
        "cached": cached,
        "latency_ms": latency_ms,
    }
    _log_call(cfg, {**row, "top_logprobs": top_logprobs})
    return row


# --------------------------------------------------------------------------
# dispersion sampling -- the only place this module samples
# --------------------------------------------------------------------------


async def _sample_one_draw(
    client: Any,
    cfg: Settings,
    *,
    method: str,
    frame: str,
    persona: pd.Series,
    item: dict[str, Any],
    option_order: str,
    draw_index: int,
    cache: diskcache.Cache | None,
) -> dict[str, Any]:
    persona_id = _persona_key(persona)
    item_id = item["id"]
    dossier = build_dossier(method, persona, prompt_version=cfg.prompt_version)
    key = cache_key(
        method=method,
        persona_id=persona_id,
        item_id=item_id,
        model=cfg.model,
        prompt_version=cfg.prompt_version,
        option_order=option_order,
        draw_index=draw_index,
        dossier_text=dossier,
        # This is the sampling path, so temperature is part of the call's
        # identity: see cache_key's docstring for why omitting it here would
        # silently manufacture the under-dispersion result.
        temperature=float(cfg.dispersion_temperature),
    )

    cached_entry = cache.get(key) if cache is not None else None
    if cached_entry is not None:
        token = cached_entry["token"]
        latency_ms = None
        cached = True
    else:
        t0 = time.monotonic()
        item_text = render_item_text(item, option_order)
        response = await _create_chat_completion(
            client,
            cfg,
            messages=[
                {"role": "system", "content": dossier},
                {"role": "user", "content": item_text},
            ],
            temperature=cfg.dispersion_temperature,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0
        token = response.choices[0].message.content.strip()
        if cache is not None:
            cache[key] = {"token": token}
        cached = False

    tokens = item["tokens"]
    n_options = len(tokens)
    if token in tokens:
        pos = tokens.index(token)
        displayed_probs = [1.0 if i == pos else 0.0 for i in range(n_options)]
        captured_mass = 1.0
    else:
        displayed_probs = [0.0] * n_options
        captured_mass = 0.0
    check_captured_mass(captured_mass, method=method, item_id=item_id, persona_id=persona_id)

    forward_probs = realign_to_forward(option_order, displayed_probs)
    top_index = max(range(len(forward_probs)), key=forward_probs.__getitem__)
    top_option = item["options"][top_index] if captured_mass > 0 else f"UNRECOGNISED:{token!r}"
    row = {
        "method": method,
        "frame": frame,
        "persona_id": persona_id,
        "item_id": item_id,
        "option_order": option_order,
        "draw_index": int(draw_index),
        "model": cfg.model,
        "prompt_version": cfg.prompt_version,
        "temperature": float(cfg.dispersion_temperature),
        "probs": forward_probs,
        "top_option": top_option,
        "top_prob": float(forward_probs[top_index]),
        "captured_mass": float(captured_mass),
        "cached": cached,
        "latency_ms": latency_ms,
    }
    _log_call(cfg, {**row, "raw_token": token})
    return row


async def sample_draws(
    client: Any,
    cfg: Settings,
    *,
    method: str,
    frame: str,
    persona: pd.Series,
    item: dict[str, Any],
    option_order: str = "forward",
    n: int | None = None,
    cache: diskcache.Cache | None = None,
) -> pd.DataFrame:
    """Temperature-sampled draws for dispersion calibration: same prompt,
    ``temperature=cfg.dispersion_temperature``, ``draw_index`` 0..n-1. This is
    the ONLY path in this module that samples rather than reading logprobs
    directly -- everywhere else uses :func:`elicit_item`.
    """
    n = cfg.dispersion_draws if n is None else n
    rows = [
        await _sample_one_draw(
            client,
            cfg,
            method=method,
            frame=frame,
            persona=persona,
            item=item,
            option_order=option_order,
            draw_index=i,
            cache=cache,
        )
        for i in range(n)
    ]
    return to_dataframe(rows)


# --------------------------------------------------------------------------
# assembling + validating results
# --------------------------------------------------------------------------


def to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble elicitation rows into a ``schemas.ELICITATION``-conforming
    frame, validating before returning."""
    if not rows:
        return schemas.ELICITATION.empty()

    df = pd.DataFrame(rows)
    df["persona_id"] = df["persona_id"].astype("int64")
    df["draw_index"] = df["draw_index"].astype("int64")
    df["temperature"] = df["temperature"].astype("float64")
    df["top_prob"] = df["top_prob"].astype("float64")
    df["captured_mass"] = df["captured_mass"].astype("float64")
    df["latency_ms"] = df["latency_ms"].astype("float64")
    df["cached"] = df["cached"].astype(bool)
    df = df[schemas.ELICITATION.names]
    return schemas.validate(df, schemas.ELICITATION)


# --------------------------------------------------------------------------
# batch runner
# --------------------------------------------------------------------------


async def run_elicitation(
    client: Any,
    cfg: Settings,
    *,
    instrument: dict[str, Any],
    personas: pd.DataFrame,
    methods: Sequence[str],
    frame: str,
    option_orders: Sequence[str] = ("forward",),
    cache: diskcache.Cache | None = None,
) -> pd.DataFrame:
    """Run every (method, persona, item, option_order) combination at
    ``draw_index=0``, bounded by ``cfg.max_concurrency`` concurrent calls."""
    if cache is None:
        cache = open_cache(cfg.cache_dir)
    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    async def _bound(method: str, persona: pd.Series, item: dict[str, Any], option_order: str):
        async with semaphore:
            return await elicit_item(
                client,
                cfg,
                method=method,
                frame=frame,
                persona=persona,
                item=item,
                option_order=option_order,
                draw_index=0,
                cache=cache,
            )

    tasks = [
        _bound(method, persona, item, option_order)
        for method in methods
        for _, persona in personas.iterrows()
        for item in instrument["items"]
        for option_order in option_orders
    ]
    rows = await asyncio.gather(*tasks)
    return to_dataframe(list(rows))


async def run_ladder(
    client: Any,
    cfg: Settings,
    *,
    instrument: dict[str, Any],
    panels: dict[str, pd.DataFrame],
    items: Sequence[dict[str, Any]] | None = None,
    option_orders: Sequence[str] = ("forward",),
    cache: diskcache.Cache | None = None,
) -> pd.DataFrame:
    """Run the persona ladder: every method against **its own** panel.

    ``run_elicitation`` assumes one shared persona table across methods, which
    was right when every method described the same real MPs. It is wrong now:
    P1's quota personas, P2's archetypes and P5's real members are different
    panels of different sizes, and pairing a method with the wrong one would
    silently score a method against personas it never saw.

    ``panels`` maps method -> PERSONA frame, so each method can only ever be run
    against the panel it was built for.
    """
    if cache is None:
        cache = open_cache(cfg.cache_dir)
    items = list(items if items is not None else instrument["items"])
    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    async def _bound(method: str, persona: pd.Series, item: dict[str, Any], order: str):
        async with semaphore:
            return await elicit_item(
                client, cfg, method=method, frame="PANEL", persona=persona,
                item=item, option_order=order, draw_index=0, cache=cache,
            )

    tasks = [
        _bound(method, persona, item, order)
        for method, panel in panels.items()
        for _, persona in panel.iterrows()
        for item in items
        for order in option_orders
    ]
    rows = await asyncio.gather(*tasks)
    return to_dataframe(list(rows))
