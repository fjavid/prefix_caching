# Review

## Scope

Reviewed Batch 2 in `prep_login.sh` against R3 of
`SPECS/model-scale-8b.md` and Batch 2 of `TASKS/model-scale-8b.md`.
Cross-checked `pipeline_config.sh`, `prepare_data.sh`, and the four SLURM stage
scripts only for sourcing side effects, environment propagation, and artifact
directory consumers.

Batches 3-10 and all non-shell implementation changes were not reviewed. No
real model or dataset download, venv build, data-prep run, SLURM submission, or
GPU path was executed.

## Blocking issues

1. **`model_dir_complete` accepts a partially downloaded sharded model as
   complete.** `prep_login.sh:63-70` returns success after finding any one
   `*.safetensors` or `*.bin` file. A directory containing `config.json`,
   `tokenizer_config.json`, `model.safetensors.index.json`, and only
   `model-00001-of-00002.safetensors` returned 0 even though the second shard
   was absent. An interrupted `hf download` can leave exactly this state:
   individual completed shards remain in the local directory while unfinished
   files remain absent or incomplete. `stage_model` then takes the skip path at
   `prep_login.sh:81-84`, so the missing shard is first discovered on an
   offline compute node. The same predicate accepts weights plus
   `tokenizer_config.json` when the tokenizer vocabulary (`tokenizer.json` or
   `tokenizer.model`) is absent; the chat-template configuration alone is not a
   loadable tokenizer. This violates R3's incomplete-directory edge case and
   the review acceptance criterion.

## Non-blocking issues

None.

Workflow practicality judgment: the repeated shared setup is not a Batch 2
defect worth fixing now. `virtualenv` and the wheelhouse installs run before
model staging. The bert-score *download* is conditional on its wheel being
absent. The fixed dataset/validation-model pre-cache runs after model staging,
not before it, and per-model `prepare_data.sh` must run for the new tokenizer.
Staging a second model therefore repeats shared venv/package checks and reloads
fixed cached assets, but this preserves the R3 requirement that those steps
stay as they are and does not affect correctness. Splitting environment setup
from model staging can be a later operational optimization; it should not
expand this repair.

## Missing tests

- An indexed multi-shard directory with one referenced shard absent must be
  rejected both before staging and after a nominally successful `hf download`.
- A directory with `tokenizer_config.json` but no loadable tokenizer vocabulary
  must be rejected on the login node.

The repository has no maintained automated test suite. All validation below
was manual and used temporary files removed after the checks.

## Commands and results

- `bash -n prep_login.sh` — passed.
- Sourced `pipeline_config.sh` under `set -euo pipefail` for both registered
  tags with temporary project/scratch roots. TinyLlama resolved to its repo,
  path, `GATED=0`, and registry values; Llama-3.1-8B resolved to its repo, path,
  `GATED=1`, and registry values. Sourcing did not change `PATH`, set
  `VIRTUAL_ENV`, or set any HF cache/offline variables. `activate_venv` was only
  defined, so sourcing before first-run venv creation has no side effect on the
  build or later HF exports.
- Sourced with `MODEL_TAG=unknown` — exited 1, listed both valid tags, and did
  not execute the following statement.
- Exercised `model_dir_complete` under `set -e` for absent, empty,
  config-only, weights-without-`tokenizer_config`, complete single-file,
  sharded safetensors, and sharded bin directories. Expected incomplete states
  returned 1 without terminating an enclosing `if`; complete states returned
  0. The `ls` globs in the conditional statements did not trigger unintended
  errexit behavior.
- Exercised `stage_model` as a bare statement under `set -euo pipefail` with an
  executable `hf` stub placed first on `PATH`. Absent ungated and token-bearing
  gated directories downloaded and passed the post-check; complete and
  sharded-complete directories skipped without invoking `hf`; empty,
  config-only, and weights-without-tokenizer-config directories exited 1
  without invoking `hf`; gated-without-token exited 1 before invoking `hf` and
  printed the model-page URL; a stub download returning an incomplete directory
  exited 1; an `hf` exit 42 propagated as exit 42. No statement after the bare
  staging call ran on any failure.
- Tested the gated path with `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, and neither.
  Both token variables pass the guard; neither fails before the stub. The local
  installed `huggingface_hub` authentication code also recognizes
  `HUGGING_FACE_HUB_TOKEN` as the backward-compatible fallback to `HF_TOKEN`.
- Inspected child-process inputs. The inline Python pre-cache block uses only
  fixed dataset/validation-model identifiers and is independent of model
  selection. It receives the HF cache exports. No earlier child process needs
  `MODEL_TAG`; `export MODEL_TAG` immediately before `bash prepare_data.sh`
  makes the child resolve the same registry entry and namespace.
- Scanned all SLURM producers and consumers. The required root set is complete:
  `PROCESSED_DIR`, `MUTATION_ROOT`/`MUTATION_DIR`, `LAYOUT_DIR`, `BENCH_DIR`, and
  `ANALYSIS_DIR`. Each stage also creates its own direct destination. No cluster
  stage reads or writes the removed un-namespaced `benchmark_results/` or
  `analysis/` directories.
- Reproduced the blocker: a one-of-two-shard indexed directory and a directory
  lacking tokenizer vocabulary both returned success from
  `model_dir_complete`.

## Suggested fixes

For sharded weights, parse `model.safetensors.index.json` or
`pytorch_model.bin.index.json` and require every unique file named in
`weight_map`. Treat a `*-of-*` weight file without a usable index as
incomplete. For an unsharded model, require the canonical single weight file.
Also validate that the tokenizer is locally loadable, for example with
`AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)`, so
`tokenizer_config.json` cannot stand in for missing vocabulary files. Apply the
same predicate after `hf download` returns.

Narrow execute prompt:

> Fix only the Batch 2 completeness blocker in
> `TASKS/handoffs/2026-08-19-codex-review-model-scale-batch2.md`. Make
> `model_dir_complete` reject partial indexed shard sets, sharded files without
> their index, and missing tokenizer payloads while preserving both registered
> models, the existing token guard, bare-call failure propagation, and
> idempotent skips. Extend the stubbed CPU-only validation for those states. Do
> not change the repeated venv/install/pre-cache workflow or any file outside
> `prep_login.sh`.

## Final git status

The working tree contains modified `prep_login.sh` and
`TASKS/model-scale-8b.md`, plus this untracked handoff. Nothing is staged. The
task-file modification was present before this review and was not edited by
the reviewer.

## Approval recommendation

Not approved. Model selection, token preflight, failure propagation,
environment propagation, and namespaced directory creation are workable, but
R3 is not met while an interrupted Llama multi-shard download can be accepted
as complete and deferred to an offline compute-node failure.
