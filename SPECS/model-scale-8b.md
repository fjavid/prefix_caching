# PRD: Model-parameterized pipeline and Llama-3.1-8B benchmark run

Priority item 1 of `research.md` §2.6.

## Goal

Make the model a first-class pipeline parameter, stage `Llama-3.1-8B-Instruct`
on the cluster, and produce a complete benchmark result set at 8B.

Two outcomes:

1. Any supported model can be selected by setting one variable, and its
   artifacts never collide with another model's.
2. A full 8B result set exists for the four RAG meaning-preserving mutation
   types, replacing the discarded v5 results.

## Motivation

Measured TTFT gains at TinyLlama-1.1B are ~5.5 ms against a ~0.02 ms noise
floor. Single-digit milliseconds invite the objection that the effect is a
measurement artifact. Prefill cost scales with parameter count, so the same
experiment at 8B should produce gains roughly 7× larger without changing any
conclusion.

Separately, the v5 results are void: they were produced before the chat-template
fix, so the `original` layout emitted EOS on the first token in ~97% of requests
(`research.md` §2.7). They are not reused and not retained as a baseline.

The 1.1B configuration must keep working — it will be re-run later — but no
cross-model comparison is in scope.

## User-facing behavior

Selecting a model is one variable:

```bash
MODEL_TAG=Llama-3.1-8B-Instruct ./run_pipeline.sh
```

`MODEL_TAG` determines the local weights directory, the tokenizer used for the
prompt-token budget, and the artifact namespace under `$SCRATCH`. Running a
different model writes to a different tree; nothing is overwritten.

Staging a model is idempotent:

```bash
MODEL_TAG=Llama-3.1-8B-Instruct bash prep_login.sh
```

Existing weights are detected and not re-downloaded.

The default remains TinyLlama-1.1B, so existing invocations behave as before
apart from the new path namespacing.

## Non-goals

- Cross-model comparison, combined plots, or a `model_name` dimension in the
  analysis. Each model is analyzed independently.
- Re-running the 1.1B baseline. Deferred to a later task.
- Adding models beyond Llama-3.1-8B-Instruct.
- New mutation types, meaning-changing mutations, new layout strategies, or a
  second workload. These are separate priority items.
- Migrating or reconciling the discarded v5 artifacts under `outputs/`.

## Requirements

### R1 — Model registry

A single place mapping `MODEL_TAG` to the values that vary per model:

| Field | Purpose |
|---|---|
| `MODEL_REPO` | Hugging Face repo id used by `prep_login.sh` |
| `MODEL_PATH` | local weights directory, `$PROJECT_ROOT/models/$MODEL_TAG` |
| `MAX_MODEL_LEN` | engine context cap |
| `MAX_PROMPT_TOKENS` | prompt-token budget for `prepare_data.sh` |
| `N_PARAMS` | parameter count for `analysis/plot_report.py` |
| `GATED` | whether the download needs `HF_TOKEN` |

Registry lives in `pipeline_config.sh`. Two entries: `TinyLlama-1.1B-Chat-v1.0`
(default) and `Llama-3.1-8B-Instruct`. An unknown `MODEL_TAG` fails with a clear
message listing valid tags.

### R2 — Artifact namespacing

All `$SCRATCH_ROOT` derived paths gain a model component:

```
$SCRATCH_ROOT/$MODEL_TAG/{processed,mutation,prompt_organization,benchmark_results,analysis}
```

This is required, not cosmetic. Current paths carry only
`TAG=${WORKLOAD}_${MUTATION_TYPE}`, so a 1.1B run after an 8B run would silently
overwrite every benchmark JSONL.

Each model gets its own `processed/rag_examples.jsonl`, because each model
filters the example set with its own tokenizer (R4).

### R3 — Model-parameterized staging

`prep_login.sh` stages the model named by `MODEL_TAG` rather than TinyLlama
unconditionally (currently hardcoded at lines 39-43).

- Skip the download when `$MODEL_PATH` already exists and is non-empty.
- For a gated repo, check `HF_TOKEN` is set before attempting the download and
  fail with an actionable message if it is not.
- The venv build, vLLM install, and dataset pre-caching stay as they are.

### R4 — Per-model prompt budget

`prepare_data.sh` uses the selected model's tokenizer and `MAX_PROMPT_TOKENS`
from the registry, writing to that model's `processed/` directory.

Each model filters independently and keeps its own example set. The two models
therefore run on different prompt subsets; this is accepted because no
cross-model comparison is in scope.

### R5 — Engine context cap

`submit_interface_benchmark.sh` passes `--max-model-len $MAX_MODEL_LEN` to
`benchmark_prefix_cache.py`. The flag exists in the working tree but no caller
sets it.

Registry values: 2048 for both models. Llama-3.1 declares 131072; without the
cap vLLM sizes the KV cache for the full window and fails allocation on a 40 GB
device.

### R6 — SLURM resources

`submit_interface_benchmark.sh` currently hardcodes `--mem=32G`,
`--time=02:30:00`, `--gres=gpu:1`. These must be settable per model, with 8B
defaults of `--mem=64G` and a time limit derived from R7.

GPU constraint for 8B: a device with ≥40 GB.

### R7 — Pilot run

Before the full sweep, a short run confirms the 8B path executes and produces
the timing data needed to size the full run.

- One mutation type, both layouts, both cache modes, ~50 cases.
- Must confirm: engine starts under the context cap; the chat template is
  applied; `original` no longer emits EOS on the first token; `reference_answers`
  is present in the output records.
- Emits measured per-case cost, which replaces the estimated timing table in
  `RUNBOOK.md`.

### R8 — Full 8B run

Four mutation types (`chunk_reorder`, `typo`, `formatting`,
`synonym_substitution`) × 2 layouts × 2 cache modes, case count and time limit
set from R7.

### R9 — Analysis parameterization

`analysis/plot_report.py` hardcodes `n_params=1.1e9` (line 202) and a
`"TinyLlama-1.1B on A100"` caption (line 271). Both take the value from the
registry so roofline annotations are correct at 8B.

### R10 — Documentation

`RUNBOOK.md` updated for the model-parameterized workflow, with the estimated
timing table replaced by pilot measurements.

## Inputs and outputs

**Inputs:** `MODEL_TAG`; `HF_TOKEN` for gated repos; the mutation and layout
knobs already in `pipeline_config.sh`.

**Outputs:** per-model artifact tree under `$SCRATCH_ROOT/$MODEL_TAG/`; benchmark
JSONLs plus `.summary.json` per layout and cache mode; analysis CSVs, summary
JSON, and plots.

The `.summary.json` written by `benchmark_prefix_cache.py` already records the
full backend config; it must additionally reflect `max_model_len` and
`apply_chat_template` so a result set is self-describing.

## Edge cases

- `MODEL_TAG` not in the registry → fail with the list of valid tags.
- Gated repo, no `HF_TOKEN` → fail before the download attempt.
- Weights directory exists but is incomplete (interrupted download) → detect and
  report rather than proceeding into a job that will fail on a compute node.
- Tokenizer has no chat template → backend warns and passes prompts through
  (already implemented in `backend_base.py`).
- Prompt exceeding `MAX_MODEL_LEN` after chat-template wrapping: the template
  adds a constant 6-token header at TinyLlama, so the effective budget is
  `MAX_PROMPT_TOKENS + template overhead + max_new_tokens`. Confirm this stays
  under `MAX_MODEL_LEN` for the selected model.
- Existing `$SCRATCH_ROOT` artifacts from before namespacing sit outside the new
  per-model tree and are simply not found. Acceptable; they are void anyway.

## Compatibility constraints

- The 1.1B configuration must remain runnable by setting `MODEL_TAG` back.
- `pipeline_config.sh` knobs keep their current names and override semantics.
- Compute nodes stay offline: no new runtime download path.
- Dependencies come from the cluster wheelhouse; no new package may be
  introduced without checking `avail_wheels`. `transformers` is already a
  dependency, so the tokenizer load in the backend adds nothing.
- The `analysis/` stage keeps running without a GPU on a login node or laptop.

## Testing and validation

There is no automated test suite in this repo (`AGENT_CONFIG.md`), so validation
is explicit and manual.

Locally, without a GPU:

- Registry resolution for both tags, and the failure path for an unknown tag.
- Derived paths contain the model component and differ between tags.
- `prep_login.sh` skips staging when the weights directory exists — verified with
  a stub directory, not a real download.
- Chat-template wrapping leaves first-divergence position shifted by a constant
  across records. Already verified: exactly +6 tokens over 25 records in each of
  four layout files.
- `reference_answers` never appears in a rendered prompt.

On the cluster, from R7:

- Engine starts at `max_model_len=2048` on the allocated device.
- `original` layout produces non-empty output text; the ~97% empty-output rate
  is gone.
- `reference_answers` present in benchmark output records.
- `exact_reuse` gain is clearly above the `unrelated_control` noise floor.

## Rollout plan

1. Registry and path namespacing (R1, R2) — local, no cluster.
2. Staging, data prep, engine cap, SLURM resources (R3–R6) — local edits.
3. Analysis parameterization (R9) — local.
4. Commit and push; pull on the cluster.
5. Stage 8B weights on a login node; run `prepare_data.sh` for that model.
6. Pilot (R7). Re-derive timing.
7. Full run (R8).
8. Update `RUNBOOK.md` with measured numbers (R10).

Steps 1–3 and 9 are agent work. Steps 4–7 are run by the user on the cluster.

## Open questions

- Time limit for the full 8B run: an output of the pilot, not decidable now.
- Case count: hold at 3000 if the pilot's measured per-case cost allows it under
  a 6 h allocation; otherwise reduce and record the reduction.
- Whether the void v5 artifacts under `outputs/` should be deleted or archived.
  Out of scope here; flagged for a decision.
