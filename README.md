# plp-sim

100 LLM personas of the Parliamentary Labour Party, scored against what the real MPs actually did.

**[AS_methods_brief.docx](AS_methods_brief.docx)** is the write-up.
**`outputs/AS_network_viz.html`** is the result. Download it and open it in a browser, it needs
nothing else.

This repository holds the code behind both.

## What it does

Builds personas of the 405 sitting Labour MPs seven different ways, asks them a single-select
question about a leadership crisis, and scores the answers against real behaviour recorded after a
cutoff date the personas never see.

Asked plainly, the panel says 100% would back the leader. Reality was 26% back, 24% against, 50%
silent, and all seven construction methods score identically. Adding the situation and the role
rule takes it from 0.26 to 0.77, against 0.50 for guessing.

## Why the answers can be checked

Everything a persona is told comes from before 1 April 2026. Everything it is scored on happened
after. The two streams meet only at the comparison.

Leakage across that line does not make results look wrong, it makes them look good, so it never
announces itself in the output. It happened twice during the build.

## Layout

```
plp_sim/
  collect.py      Parliament APIs, every response cached to disk
  attributes.py   one row per MP, eight recorded facts
  frames.py       panel selection, and frame error measured with no LLM involved
  personas.py     the seven persona constructions
  dossier.py      one renderer per method, dispatched through a registry
  elicit.py       logprob elicitation, disk cache, async runner
  holdout.py      observed outcomes, built before anything else touches the data
  cascade.py      the network over archetypes
  network.py      co-voting similarity and community detection
  metrics.py      scoring
  schemas.py      the column contracts every table is validated against
scripts/          numbered, run in order
config/           the instrument: the questions and the pressure panel
data/manual/      hand-compiled data, committed because the run needs it
tests/            226 tests
```

## Running it

```bash
uv sync
cp .env.example .env                              # add an OpenAI key
uv run python scripts/01_collect.py               # 02 and 03 next
uv run python scripts/04_build_tables.py          # attributes, holdout, panels
uv run python scripts/05_run_ladder.py            # the seven methods
uv run python scripts/06_role_rule.py             # the role rule, by group
uv run python scripts/07_build_visual.py          # writes the HTML
```

Every model call is cached on its full inputs, including the rendered prompt, so re-running costs
nothing for work already done. Collection is cached too: a second run makes zero network calls.

Answers are read as a probability distribution over the option letters rather than by sampling
repeatedly. One call gives the full spread of opinion, which is cheaper and has no sampling noise.
This is why the project is OpenAI only: Anthropic exposes no logprobs.

## Reading the code

Two files carry the argument.

`config/instrument.yaml` is the questions, with the reasoning for each design decision in
comments, including three warnings about wordings that must not be changed because they move the
answer by 60 to 80 points.

`plp_sim/dossier.py` is how a persona becomes a prompt: one renderer per method, and the
situational context and role rule that turned the result around.

Several comments record mistakes rather than decisions. The cutoff notes in `frames.py` and
`schemas.py` exist because that boundary was breached twice.
