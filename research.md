# Research Notes

Running record of research framing, decisions, and clarifications for this
project. Appended to over time; newest sections go at the end unless they
belong with an existing topic.

Scope note: this file records *how we think about the problem*. It is not an
architecture reference (`README.md` and the per-module READMEs cover how the
pipeline runs), and it is not the original project intent (`project_description.md`
carries that, unmodified on purpose so the plan of record stays separable from
what we later learned).

---

## 1. What the project is studying

We study the effectiveness of prefix caching during LLM inference, using vLLM
as the serving backend.

The mechanism: vLLM retains the KV cache produced during prefill. When a new
request shares a leading token sequence with something already cached, that
prefill work is reused instead of recomputed, which reduces time to first
token (TTFT).

The problem: real prompts are rarely byte-identical. Small differences break
reuse. The project measures how badly, for which kinds of differences, and how
much can be recovered by reorganizing the prompt alone — without resorting to
more advanced cache-composition methods.

The strategy studied so far: place content that is unlikely to change at the
front of the prompt, and content that varies at the end. In a RAG prompt this
means system instruction and retrieved context early, the user question last.

### 1.1 Clarifications on the mechanism

**Reuse is a cliff, not a gradient.** vLLM reuses tokens from position 0 up to
the *first divergence*, at 16-token block granularity. Similarity after that
point contributes nothing. This is why the framing "partially similar prompts
get partial reuse" is misleading — a prompt can be 99% identical and still get
near-zero reuse if the 1% lands early. The v5 RAG results show this directly:
a single injected typo in the user query collapses the TTFT gain from ~5.5 ms
to statistical noise.

**"Retrieved context early, question last" is traffic-dependent, not a general
rule.** It is correct when the query varies while retrieval stays stable. When
retrieval varies per request — the common RAG case, since a new query produces
new chunks — the retrieved chunks *are* the volatile content, and putting them
early is the worst available choice. The defensible rule is weaker and more
interesting: *whatever is stable for your traffic pattern goes first.*

**One layout strategy has been studied, not several.** `prompt_organization/`
registers three names, but `volatile_last` is a literal alias of `stable_first`
and `original` is the no-op baseline. Only `stable_first` is a real strategy.
It is also an **oracle**: it reads the ground-truth `mutation_type` to look up
which section the mutation touched. A deployed system does not have that
information in advance, so current numbers are an upper bound on what a
practical layout policy could achieve.

The nuance matters for how this is reported. The *policy* is deployable —
volatility is usually knowable from a prompt's role structure without any
mutation label (the system prompt is fixed, the user query changes every
request, retrieved chunks change when the query does), and real deployments
already exploit exactly this. What the oracle buys is per-mutation-type
precision a real system would not have: it knows `typo` hit the query and
`chunk_reorder` hit the chunks, and reorders differently for each. A deployed
version would commit to one static order for the whole workload. So the honest
framing is that the evaluation, not the idea, uses a shortcut — and
`stable_first_blind` (§2.1) is the experiment that quantifies the gap.

### 1.2 The headline result so far

The strongest finding to date is not "reordering helps." It is the
**asymmetry** in when reordering helps:

- Volatile content is a *small* part of the prompt (typo, synonym substitution)
  → `stable_first` recovers essentially the full exact-reuse ceiling
  (+5.5 ms, ~100% recovery).
- Volatile content *is* the payload (chunk reorder) → reordering does nothing,
  and measured slightly negative (−0.54 ms).

Stated as a rule: **layout recovery scales with the stable mass that can be
moved ahead of the first divergence.** The strategy is the demonstration; the
asymmetry is the finding.

---

## 2. Agreed direction

Decisions taken so far. Priority order at the end of this section.

### 2.1 Additional layout strategies

`stable_first` is the only strategy that exists today. Four candidates, in
descending value:

**`canonicalize`** — make logically-identical prompts *byte*-identical.
Purely rule-based (regex and string normalization); **no LLM involved**.
Operations: collapse whitespace, unify case and punctuation, sort structured
keys, sort retrieved chunks by a stable document ID.

It does **not** fix typos. A typo is a genuine content change; reversing it
would require a spellchecker, which would also rewrite legitimate text and risk
altering meaning. `canonicalize` targets *formatting* and *reorder* mutations;
typos remain `stable_first`'s job. The two strategies are complementary and
should be measured both separately and combined.

**`static_preamble`** — cross-request rather than pairwise reuse. "Boilerplate"
here means the fixed text repeated in every prompt (e.g. the RAG system
instruction, identical across all 3000 cases). The current harness measures one
base → one followup, i.e. a pair. Real serving has thousands of requests sharing
that opening text, so a single cache entry serves all of them. This strategy
guarantees the first N tokens are byte-identical workload-wide. Note this is a
different question from "how far into a given pair do the prompts match."

**`stable_first_blind`** — the non-oracle variant. One fixed section order per
workload, applied to every request, with no knowledge of which mutation occurred
or where. Measures how much of the oracle's gain survives in a deployable
policy.

**`block_aligned`** — lowest priority. vLLM checks the cache in 16-token blocks,
and a block is a hit only if all 16 tokens match. If divergence falls at token
100, the block covering tokens 96–111 is recomputed entirely, so tokens 96–99
are recomputed despite being identical. The fix is to pad the *end of a stable
section* with semantically inert filler (newlines/spaces) until the next section
starts on a multiple of 16.

Padding introduces no new divergence because it is applied identically to both
prompts, in a section whose content is already shared — so the filler is part of
the shared prefix. Only ever pad stable sections, and only at section
boundaries, never mid-text. Maximum saving is 15 tokens per boundary, so this is
a footnote-sized result, not a headline.

### 2.2 Workload coverage

Current results are RAG QA seeded from Natural Questions.

**Decision: add agentic / tool-use as the second workload.** Its geometry is
small volatile fields (timestamps, IDs, request-specific arguments) sitting
*early*, among large stable tool schemas — the point on the
where-is-the-volatile-content axis where layout should win biggest, and where
production deployments actually suffer.

- **Multi-turn chat: not needed.** History appends at the tail, so the prefix is
  naturally stable and reuse is close to free. It confirms what is already
  predictable. Cheap to add later from ShareGPT if a third family is wanted.
- **Scientific: dropped.** Code exists but has never been run, and few people
  serve such prompts at scale, so external validity is low. If a structured
  -template geometry is wanted later, JSON-schema extraction is the better proxy.

### 2.3 Meaning-changing mutations

**Decision: included in the next run.** Code already exists. This is the entire
correctness axis: when a prompt changes meaning while keeping its prefix, the
cache can serve a fast hit on a prompt that should produce a *different* answer.
Fast-but-wrong makes this a correctness story rather than only a latency story.

### 2.4 Model scale

**Decision: move to `Llama-3.1-8B-Instruct`** (`Qwen2.5-7B-Instruct` is an
acceptable substitute). Rationale: it is the common baseline in serving papers,
so numbers are comparable, and ~16 GB in bf16 fits a single A100/H100 alongside
the KV cache.

**Keep the TinyLlama-1.1B runs.** Reporting both sizes shows the effect growing
with prefill cost, which is itself a result and preempts the objection that
single-digit-millisecond gains are measurement noise.

### 2.5 Comparison against position-independent reuse

**Decision: compare against at least one such method.** Chosen:
**CacheBlend**, shipped as part of **LMCache** (https://github.com/LMCache/LMCache,
EuroSys'25 best paper). It reuses KV blocks at *any* position by selectively
recomputing a small number of tokens to restore quality — precisely the method
that fixes `chunk_reorder`, which no layout strategy can.

Practical notes: LMCache integrates directly with vLLM, which we already run, so
this is closer to a configuration change than a new backend. Watch for V1
KV-connector issues (LMCache issue #3238) and pin a known-good vLLM/LMCache
version pair.

### 2.6 Priority order

1. **Model scale** — no new code required; highest value per unit effort.
2. **Meaning-changing mutations** — unlocks the correctness story.
3. **`canonicalize`** — the direct fix for the one negative result.
4. **Agentic / tool-use workload** — second geometry.
5. `stable_first_blind`, `static_preamble`, CacheBlend comparison.
6. `block_aligned` — last; bounded, small payoff.

---

## 3. Documentation conventions

- `project_description.md` — original project intent. Deliberately **not**
  updated as findings arrive, so the plan of record stays comparable against
  what was actually found.
- `research.md` (this file) — research framing, clarifications, decisions.
- `FINDINGS.md` — planned, not yet created. Will hold the results record
  (result tables, limitations, guidance extracted) once the study expands
  beyond the current single-workload scope.
- `README.md` and per-module READMEs — how the pipeline actually runs.
