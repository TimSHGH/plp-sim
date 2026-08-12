"""System-prompt rendering: one renderer per method, dispatched through a registry.

The rendered text is the ``system`` message; the item text goes in the ``user``
message. ``FORCED_CHOICE_INSTRUCTIONS`` is byte-identical across every method,
so a measured difference between methods is grounding rather than wording.

Renderers take a PERSONA row, not an ATTRIBUTES row. That lets one renderer
serve a synthetic archetype and a real MP alike, and it is why the attribute
block is limited to :data:`plp_sim.personas.PERSONA_ATTRS`, which excludes every
post-cutoff column by construction.

That exclusion is load-bearing. An earlier version listed nomination timing in
the attribute block, which is both a post-cutoff variable and the answer to one
of the scored questions. Excluding them at the contract level makes the mistake
impossible to reintroduce without editing the contract first.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import pandas as pd

from plp_sim import schemas

KNOWN_PROMPT_VERSIONS: frozenset[str] = frozenset({"v1", "v2"})

#: Shared by every method. Guards the default failure mode: a balanced,
#: cautious, high-agreement panel that produces confident-looking garbage
#: instead of a forced choice.
FORCED_CHOICE_INSTRUCTIONS = (
    "Respond with exactly one capital letter matching your chosen option, and "
    "nothing else: no words, no punctuation, no explanation. You must pick a "
    "single option even under uncertainty. Never hedge, never average across "
    "options, never say you cannot decide, and never decline to answer. Do "
    "not mention that you are an AI, a model, or a simulation, and do not "
    "comment on how confident, certain, or difficult the question is. Answer "
    "only in character, as the person described would, all the way through."
)

_PLP = "the governing Parliamentary Labour Party (PLP)"


# --------------------------------------------------------------------------
# shared fragments
# --------------------------------------------------------------------------


def _majority_band(pct: float) -> str:
    if pct < 5.0:
        return "an ultra-marginal seat that could easily be lost at the next election"
    if pct < 15.0:
        return "a marginal seat where re-election is not at all guaranteed"
    return "a safe seat with a comfortable cushion over the runner-up"


def _payroll_clause(is_payroll: object, role: object) -> str:
    if bool(is_payroll):
        role_txt = role if isinstance(role, str) and role else "a payroll post"
        return f"on the government payroll, holding {role_txt}"
    return "a backbencher, holding no government or opposition frontbench role"


def _nullable(value: object, fmt: str, none_text: str) -> str:
    if value is None or pd.isna(value):
        return none_text
    return fmt.format(value)


def _attribute_bundle(p: pd.Series) -> str:
    """The attribute record. Pre-cutoff columns only, by contract."""
    lines = []
    mp = p.get("majority_pct")
    if mp is not None and not pd.isna(mp):
        lines.append(
            f"- Majority at the 2024 election: {float(mp):.1f}% of valid votes cast "
            f"({_majority_band(float(mp))})"
        )
    lines += [
        "- Vote share: " + _nullable(p.get("vote_share"), "{:.1f}%", "not recorded"),
        f"- Runner-up party in their constituency: {p.get('runner_up_party') or 'not recorded'}",
        f"- Intake: {'2024 intake' if p.get('is_2024_intake') else 'pre-2024 intake'}",
        f"- Position: {_payroll_clause(p.get('is_payroll'), p.get('role'))}",
        "- Select/other committees served on: "
        + _nullable(p.get("committee_count"), "{:.0f}", "not recorded"),
        "- Rebellion rate against the whip across divisions before the cutoff: "
        + _nullable(p.get("rebellion_rate"), "{:.1%}", "no rebellions on record"),
        "- Speeches recorded in Hansard since the 2024 election: "
        + _nullable(p.get("speech_count"), "{:.0f}", "not recorded"),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------


class PersonaMethod:
    """One way of turning a PERSONA row into a system-message body.

    A registry of small classes rather than an if/elif chain on ``method``:
    that chain was the Switch Statements smell, and the ladder would have taken
    it to seven branches. A new method is now a class plus one registry line,
    with no existing branch touched.
    """

    name: str = ""
    deployable: bool = True

    def body(self, persona: pd.Series) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def render(self, persona: pd.Series) -> str:
        return self.body(persona) + "\n\n" + FORCED_CHOICE_INSTRUCTIONS


class P0Stereotype(PersonaMethod):
    """Generic stereotype. No attributes at all: the sector baseline.

    Carries no persona-specific content, so every P0 call is byte-identical.
    That is the honest baseline: a stereotype panel has one opinion, not a
    distribution.
    """

    name = "P0"

    def body(self, persona: pd.Series) -> str:
        return (
            f"You are role-playing a Member of Parliament in {_PLP}, answering a "
            "series of internal party questions. You know nothing specific about "
            "yourself beyond being a Labour MP in this Parliament: do not invent a "
            "name, a constituency, a majority, or a personal history. Answer as a "
            "typical Labour MP would, weighing party loyalty against electoral "
            "pressure the way the average member of this group tends to."
        )


class _AttributePersona(PersonaMethod):
    """Shared body for any method described by an attribute record."""

    headline = ""

    def body(self, persona: pd.Series) -> str:
        return self.headline + "\n\n" + _attribute_bundle(persona)


class P1Quota(_AttributePersona):
    """Synthetic quota persona. No real MP behind the numbers."""

    name = "P1"
    headline = (
        f"You are role-playing a Member of Parliament in {_PLP}, answering a series "
        "of internal party questions. You are not any particular real MP: the "
        "record below describes the kind of member you are. Answer as someone with "
        "exactly this profile would."
    )


class P2Archetype(_AttributePersona):
    """Synthetic archetype persona: a cluster's central tendency."""

    name = "P2"
    headline = (
        f"You are role-playing a Member of Parliament in {_PLP}, answering a series "
        "of internal party questions. You represent a recognisable type within the "
        "party rather than any individual; the record below is the typical profile "
        "of that group. Answer as a representative member of it would."
    )


class P3Biography(PersonaMethod):
    """P2's vector plus an LLM-written biography.

    Scored against P2 on the identical underlying vector, which is the only way
    to separate what the prose *adds* (coherence, a sense of a person, better
    role-play) from what it *invents* (detail the vector never contained, which
    is stereotyping dressed as specificity).
    """

    name = "P3"

    def body(self, persona: pd.Series) -> str:
        bio = persona.get("biography")
        if bio is None or pd.isna(bio) or not str(bio).strip():
            raise ValueError(
                f"P3 persona {persona.get('persona_id')} has no biography; "
                "run personas.generate_biographies first"
            )
        return (
            f"You are role-playing a Member of Parliament in {_PLP}, answering a "
            "series of internal party questions. You are not any particular real "
            "MP. This is who you are:\n\n"
            f"{str(bio).strip()}\n\n"
            "Your underlying record:\n" + _attribute_bundle(persona)
        )


class P4RealAnonymised(_AttributePersona):
    """A real MP's record, anonymised. Ceiling, not a method."""

    name = "P4"
    deployable = False
    headline = (
        f"You are role-playing an anonymised Member of Parliament in {_PLP}, "
        "answering a series of internal party questions, described by the "
        "attribute record below."
    )


class P5RealNamed(PersonaMethod):
    """A real MP, named. The contamination ceiling."""

    name = "P5"
    deployable = False

    def body(self, persona: pd.Series) -> str:
        return (
            f"You are role-playing a named Member of Parliament in {_PLP}, "
            "answering a series of internal party questions.\n\n"
            f"You are {persona['name']}, the Member of Parliament for "
            f"{persona['constituency']}.\n\n" + _attribute_bundle(persona)
        )


class RecallControl(PersonaMethod):
    """No persona at all. The leakage floor."""

    name = "RECALL"
    deployable = False

    def body(self, persona: pd.Series) -> str:
        return (
            f"You are being asked about {persona['name']}, the real Member of "
            f"Parliament for {persona['constituency']}. Using your own knowledge "
            "of this specific, real person and their public record, answer how "
            "they would actually respond in real life. Do not invent, infer, or "
            "fall back on a generic biography, persona, or attribute profile for "
            "them: rely only on what you actually know about this named "
            "individual. If you are not sure, still give your single best answer "
            "for this specific person rather than a generic one."
        )


class P3ShuffledControl(P3Biography):
    """P3 with the biographies permuted onto the wrong personas.

    The length control. P3's prompt is ~57% longer than P2's, so a P3 win could
    be prompt length rather than biography content. Generic filler prose would
    control length but also change style and register. Shuffling P3's own
    biographies holds length, style, register and total information identical
    and destroys exactly one thing: whether the biography describes the persona
    it is attached to.

    Costs no extra API calls -- it reuses text already generated.
    """

    name = "P3S"


class P2CSituated(P2Archetype):
    """P2's vector plus the situational context: what the polls say, and whether
    an MP of this type is personally exposed to them.

    Scored against P2 on the identical archetype vector, so P2C - P2 isolates
    the added evidence exactly, in the same way P3 - P2 isolates the biography.

    **The block states the situation and stops.** It does not say the leader is
    the cause, that colleagues are unhappy, or that anything should follow. That
    inference is the thing being measured; supplying it would answer the
    question in the prompt.
    """

    name = "P2C"

    def __init__(self, manifesto: Mapping[str, str] | None = None) -> None:
        """``manifesto`` maps "payroll"/"backbench" to a sentence about what that
        MP's job permits them to say.

        Injected rather than patched on. Four scripts used to reassign
        ``P2CSituated.body`` at runtime and restore it afterwards, which mutates
        global state, duplicates the same lambda four times, and leaves the class
        permanently patched if anything raises before the restore line.
        """
        self.manifesto = dict(manifesto or {})

    def _where_you_stand(self, p: pd.Series) -> str:
        if not self.manifesto:
            return ""
        return self.manifesto["payroll" if bool(p.get("is_payroll")) else "backbench"]

    def _context(self, p: pd.Series) -> str:
        share = p.get("seat_at_risk_share")
        if share is None or pd.isna(share):
            return ""
        lines = ["\nThe current political situation, as of the most recent published data:\n"]
        nat = p.get("national_context")
        if isinstance(nat, str) and nat.strip():
            lines.append(nat.strip())
        pct = float(share)
        who = p.get("projected_winner")
        lab = p.get("projected_labour_share")
        if pct == 0.0:
            lines.append(
                "- On the most recent constituency-level projection (MRP, January 2026), "
                "MPs with your profile are projected to hold their seats at the next "
                "general election."
            )
        else:
            band = ("nearly all" if pct >= 0.9 else "most" if pct >= 0.6
                    else "around half" if pct >= 0.4 else "a minority of")
            lines.append(
                f"- On the most recent constituency-level projection (MRP, January 2026), "
                f"{band} MPs with your profile ({pct:.0%}) are projected to lose their "
                f"seat at the next general election"
                + (f", most often to {who}" if isinstance(who, str) and who else "")
                + "."
            )
        if lab is not None and not pd.isna(lab):
            lines.append(f"- Projected Labour vote share in seats like yours: {float(lab):.0f}%.")
        return "\n".join(lines)

    def body(self, persona: pd.Series) -> str:
        return super().body(persona) + "\n" + self._context(persona) + self._where_you_stand(persona)


METHOD_REGISTRY: dict[str, PersonaMethod] = {
    m.name: m
    for m in (
        P0Stereotype(), P1Quota(), P2Archetype(), P2CSituated(), P3Biography(), P3ShuffledControl(),
        P4RealAnonymised(), P5RealNamed(), RecallControl(),
    )
}


@contextmanager
def using(method: str, renderer: PersonaMethod) -> Iterator[None]:
    """Temporarily swap the renderer for ``method``, restoring it on the way out.

    The registry is module-global, so a script that swaps a renderer and forgets
    to put it back silently changes every later call in the process. ``finally``
    makes that impossible, including when the body raises.
    """
    previous = METHOD_REGISTRY[method]
    METHOD_REGISTRY[method] = renderer
    try:
        yield
    finally:
        METHOD_REGISTRY[method] = previous


def build_dossier(method: str, persona: pd.Series, *, prompt_version: str = "v2") -> str:
    """Render the system-message dossier for ``method`` from one PERSONA row.

    Deterministic in ``(method, persona, prompt_version)`` only. No timestamps
    or per-call ids: the dossier is reused as the system message across every
    item for a given persona, and OpenAI's prefix caching is positional, so
    anything varying here breaks the cache for everything after it.
    """
    if method not in schemas.METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {schemas.METHODS}")
    if prompt_version not in KNOWN_PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt_version {prompt_version!r}")
    return METHOD_REGISTRY[method].render(persona)


def dossier_lengths(persona: pd.Series, methods=None, *, prompt_version: str = "v2") -> dict[str, int]:
    """Rendered character length per method, for the length-parity check."""
    return {
        m: len(build_dossier(m, persona, prompt_version=prompt_version))
        for m in (methods or schemas.METHODS)
    }
