# Review

## Scope

Reviewed Batch 3 in `prepare_data.sh` and
`prompt_mutation/prepare_rag_data.py` against R4 of
`SPECS/model-scale-8b.md` and Batch 3 of `TASKS/model-scale-8b.md`.
Cross-checked `pipeline_config.sh`, `prep_login.sh`, and
`submit_build_mutation.sh` for registry resolution, child-process propagation,
and the processed-file consumer. Repository-wide searches were limited to
hardcoded prompt budgets and processed-path references.

Batches 4-10 and all other implementation were not reviewed. No dataset or
model was downloaded, no complete data-prep run was executed, and no SLURM or
GPU job was submitted.

## Blocking issues

None.

## Non-blocking issues

1. `prompt_mutation/prepare_rag_data.py:11-14` states that
   `MAX_PROMPT_TOKENS + max_new_tokens` is the required engine window, but data
   prep counts the untemplated prompt. The actual bound also includes the chat
   template. Current values are safe: TinyLlama's measured overhead gives
   `1800 + 6 + 64 = 1870`, leaving 178 tokens below 2048. Llama-3.1's standard
   one-user-message template is also far below the maximum allowable overhead
   of 184 tokens, although its exact tokenizer/template was not available for
   local execution. This wording can become operationally wrong if the budget
   is raised to `MAX_MODEL_LEN - max_new_tokens`.

2. The echoed model, tokenizer, and budget are sufficient to inspect an
   interactive data-prep run but are not durable provenance. The JSONL has no
   manifest, and `submit_build_mutation.sh` does not validate how an existing
   file was produced. Per-model namespacing prevents the two standard registry
   invocations from colliding, but an explicit tokenizer/budget override or a
   later registry change can leave a same-namespace stale file that cannot be
   attributed after stdout is lost. This is a real reproducibility gap, not an
   R4 blocker: R4 and Batch 3 require model namespacing and the echoed summary,
   both of which are implemented.

3. The repository still documents legacy un-namespaced paths at `README.md:45`
   and `RUNBOOK.md:49`. The primary SLURM consumer uses the correct namespaced
   `$PROCESSED_DIR`, so the pipeline cannot select the old file. A user following
   the README's direct command can load
   `$SCRATCH/prefix_caching/processed/rag_examples.jsonl`, however, which may be
   a pre-namespacing example set produced for a different tokenizer. The
   RUNBOOK update is assigned to Batch 6; the README reference should be
   corrected with it.

## Missing tests

- A complete login-node data-prep run with each real tokenizer was not run,
  because it would require the prohibited dataset/model access. In particular,
  Llama-3.1's exact chat-template overhead remains to be measured during the
  staged-model validation/pilot.
- The repository has no maintained automated test suite. The registry,
  `prep_login.sh` child-process path, empty-export behavior, explicit overrides,
  output namespacing, syntax, and compilation were checked manually.

## Commands and results

- `bash -n prepare_data.sh` — passed.
- `.venv/bin/python` `py_compile` on
  `prompt_mutation/prepare_rag_data.py`, with the bytecode target under
  `/private/tmp` — passed.
- Sourced `pipeline_config.sh` and the exact variable-resolution lines
  `prepare_data.sh:35-58` under `set -euo pipefail` in clean environments for
  both tags. TinyLlama resolved tokenizer
  `/clean/home/work/prefix_caching/models/TinyLlama-1.1B-Chat-v1.0`, budget
  1800, and output
  `/clean/scratch/prefix_caching/TinyLlama-1.1B-Chat-v1.0/processed/rag_examples.jsonl`.
  Llama-3.1 resolved the corresponding Llama tokenizer, budget 1800, and its
  distinct model-namespaced output.
- Repeated the exact-block check with `MAX_PROMPT_TOKENS` and `TOKENIZER_PATH`
  exported as empty strings. `pipeline_config.sh`'s `:=` restored budget 1800,
  and `prepare_data.sh`'s `:-` restored the registry-derived tokenizer for both
  tags; neither value remained empty.
- Repeated with `MAX_PROMPT_TOKENS=999` and
  `TOKENIZER_PATH=/explicit/tokenizer`. Both explicit non-empty overrides were
  preserved for both tags while the output stayed under the selected
  `MODEL_TAG` namespace.
- Simulated the `prep_login.sh` boundary: a parent sourced the registry,
  exported `MODEL_TAG`, and launched a clean child that re-sourced the registry
  and resolved the data-prep variables. Both default and explicit-override
  cases preserved the expected tokenizer, budget, and namespace. Thus deleting
  the local shell default does not leave `MAX_PROMPT_TOKENS` unbound on either
  direct or `prep_login.sh` invocation paths.
- An initial exact-line extraction harness using `source /dev/stdin` and then
  process substitution did not feed the fragment into the child shell and
  raised `TOKENIZER_PATH: unbound variable`. Re-running the same extracted
  lines through `eval` exercised the intended code and passed; this was a
  validation-harness failure, not a repository-script failure.
- Repository-wide `git grep` found executable hardcoded 1800 defaults only in
  the two registry arms and the direct Python CLI default at
  `prompt_mutation/prepare_rag_data.py:71`. `prepare_data.sh` always passes the
  registry value explicitly, and both current registry entries are 1800, so
  the Python fallback cannot affect a pipeline run. It remains a second default
  for direct module invocation and can drift if a registry budget changes.
- Processed-path scan found `submit_build_mutation.sh` reading only
  `$PROCESSED_DIR/${WORKLOAD}_examples.jsonl`; `PROCESSED_DIR` is derived as
  `$SCRATCH_ROOT/$MODEL_TAG/processed`. `paths.py` defines only the separate
  repo-local `outputs/processed` tree and is not a cluster consumer. The only
  stale cluster paths found were the README and RUNBOOK references above.
- `git diff --check -- prepare_data.sh prompt_mutation/prepare_rag_data.py` —
  passed.

## Suggested fixes

- Amend the in-scope docstring inequality to include chat-template overhead.
- In the scheduled documentation pass, change the README and RUNBOOK examples
  to `$SCRATCH_ROOT/$MODEL_TAG/processed/rag_examples.jsonl`.
- If post-run attribution is required, write a sidecar manifest containing at
  least `MODEL_TAG`, tokenizer path/model id, prompt budget, dataset, split, and
  extraction settings, then validate it before mutation building. This is a
  follow-up requirement rather than a Batch 3 repair.
- When registry budgets diverge, remove or require the direct Python CLI's 1800
  fallback so it cannot become a second behavioral source of truth.

## Final git status

The working tree contains modified `prepare_data.sh`,
`prompt_mutation/prepare_rag_data.py`, and `TASKS/model-scale-8b.md`, plus this
untracked handoff. Nothing is staged. The task-file modification was present
before this review and was not edited by the reviewer.

## Approval recommendation

Approved. R4 is met. On every requested shell invocation path,
`MAX_PROMPT_TOKENS` is non-empty and registry-derived, `TOKENIZER_PATH` defaults
to the selected model while preserving an explicit override, and each model
writes and consumes a distinct processed JSONL. The 2048 context cap leaves
178 tokens after TinyLlama's measured template overhead and 184 tokens before
Llama-3.1 template overhead; the latter's standard template is well within
that bound. The remaining issues concern documentation and durable provenance,
not corruption of the standard Batch 3 data-prep path.

## Commit instructions

```bash
git add -- prepare_data.sh prompt_mutation/prepare_rag_data.py
git commit -m "Use per-model data preparation settings"
```
