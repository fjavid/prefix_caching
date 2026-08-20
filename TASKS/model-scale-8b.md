# Task: Model-parameterized pipeline and Llama-3.1-8B benchmark run

## Source spec

`SPECS/model-scale-8b.md`

## Overall goal

Make the model a first-class pipeline parameter selected by one variable
(`MODEL_TAG`), with per-model artifact namespacing so runs never collide, then
stage and benchmark `Llama-3.1-8B-Instruct` on the cluster.

## Global constraints

- Compute nodes are offline. Anything needing network belongs in `prep_login.sh`
  or `prepare_data.sh`.
- Dependencies come from the cluster wheelhouse (`pip install --no-index`). No
  new package without checking `avail_wheels`. `transformers` is already present.
- No automated test suite exists. Validation is explicit and manual; state what
  was and was not checked.
- The TinyLlama-1.1B configuration must stay runnable by setting `MODEL_TAG`
  back to it.
- Agents do not submit SLURM jobs or download models. Batches 7-10 are run by
  the user.
- Batches 1-6 are local, no GPU, and must leave the repo working.

## Batches

### Batch 1: Model registry and artifact namespacing

Status: done

Scope:
- `pipeline_config.sh`: add `MODEL_TAG` (default `TinyLlama-1.1B-Chat-v1.0`) and
  a registry resolving it to `MODEL_REPO`, `MODEL_PATH`, `MAX_MODEL_LEN`,
  `MAX_PROMPT_TOKENS`, `N_PARAMS`, `GATED`. Unknown tag exits non-zero listing
  valid tags.
- Insert `$MODEL_TAG` into the derived roots: `MUTATION_DIR`, `LAYOUT_DIR`,
  `BENCH_DIR`, `ANALYSIS_DIR`, `PROCESSED_DIR` all become
  `$SCRATCH_ROOT/$MODEL_TAG/...`. Introduce `MUTATION_ROOT="$SCRATCH_ROOT/$MODEL_TAG/mutation"`
  so callers stop composing that path themselves.
- Update the two direct composers: `submit_build_mutation.sh:33,43`
  (`--output-root`) and `run_pipeline.sh:166-169` (the printed path summary).
- `analyze_local.sh:32-33` picks up the namespaced defaults automatically via
  `SCRATCH_ROOT`; confirm and adjust only if the sourcing order breaks.
- `MODEL_PATH` keeps its current override semantics: an explicitly exported
  `MODEL_PATH` still wins over the registry value.

Out of scope: staging, data prep, engine flags, analysis.

Implementation steps:
1. Add the registry as a case statement or associative array in
   `pipeline_config.sh`, resolved before the derived-path block at lines 37-45.
2. Rewrite lines 38-42 to include `$MODEL_TAG`.
3. Update `submit_build_mutation.sh` and `run_pipeline.sh` to use the derived
   variables rather than re-composing `$SCRATCH_ROOT/mutation`.
4. Extend the knobs comment block at the top of `pipeline_config.sh`.

Validation:
- `MODEL_TAG=TinyLlama-1.1B-Chat-v1.0 bash -c 'source pipeline_config.sh; echo ...'`
  prints paths containing the tag; same for `Llama-3.1-8B-Instruct`; the two
  differ in every derived path.
- Unknown `MODEL_TAG` exits non-zero with the valid-tag list.
- Explicit `MODEL_PATH=/tmp/x` still overrides the registry.
- `bash -n` on every modified shell script.

Review notes:

### Batch 2: Model-parameterized staging in prep_login.sh

Status: done

Scope:
- Replace the hardcoded TinyLlama download (`prep_login.sh:39-43`) with staging
  of `$MODEL_TAG` from `$MODEL_REPO` into `$MODEL_PATH`.
- Skip when the weights directory exists and is non-empty.
- Detect an incomplete directory (present but missing `config.json` or any
  weight shard) and fail with an actionable message rather than proceeding.
- For `GATED=1`, verify `HF_TOKEN` is set before attempting the download; fail
  with the license-acceptance instruction if not.
- Source `pipeline_config.sh` instead of re-deriving `PROJECT_ROOT` and
  `SCRATCH_ROOT` locally (`prep_login.sh:9-10`), so the registry is available.
- `mkdir -p "$SCRATCH_ROOT/$MODEL_TAG"/{...}` for the namespaced tree.

Out of scope: actually downloading anything.

Implementation steps:
1. Source `pipeline_config.sh` after `load_modules`, keeping the module loads
   that must precede venv creation.
2. Extract the staging block into a `stage_model()` function.
3. Add the existence, completeness, and token guards.

Validation:
- Stub `$PROJECT_ROOT/models/<tag>/` containing `config.json` → staging is
  skipped, message printed, no network call attempted.
- Empty stub directory → fails as incomplete.
- `GATED=1` with `HF_TOKEN` unset → fails before any download, message names the
  model page.
- `bash -n`.
- Not validated: a real download. Requires network and a login node.

Review notes:

### Batch 3: Per-model prompt budget in prepare_data.sh

Status: done

Scope:
- `prepare_data.sh` takes `TOKENIZER_PATH` and `MAX_PROMPT_TOKENS` from the
  registry instead of the hardcoded defaults at lines 40-41.
- Output goes to the namespaced `$PROCESSED_DIR` from Batch 1.
- Update the comment at line 39 which states the budget is sized for
  TinyLlama's 2048-token window.
- Record the resolved model tag, tokenizer path, and token budget in the script's
  echoed summary so a stale processed file can be attributed.

Out of scope: changing the filter algorithm; the tokenizer-vocabulary difference
between models is accepted per the spec.

Implementation steps:
1. Replace the `: "${TOKENIZER_PATH:=$MODEL_PATH}"` and
   `: "${MAX_PROMPT_TOKENS:=1800}"` defaults with registry lookups that remain
   overridable by explicit export.
2. Extend the echoed summary block.

Validation:
- Both tags resolve to different `TOKENIZER_PATH` and `PROCESSED_DIR` values;
  print without executing the Python step.
- Explicit `MAX_PROMPT_TOKENS=999` still overrides.
- `bash -n`.
- Not validated: an actual data-prep run. Needs network for the dataset.

Review notes:

### Batch 4: Engine context cap and per-model SLURM resources

Status: done

Scope:
- `submit_interface_benchmark.sh` passes `--max-model-len "$MAX_MODEL_LEN"` to
  `benchmark_prefix_cache.py`. The flag exists in the working tree but no caller
  sets it.
- Make `--mem`, `--time`, and the GPU constraint settable per model rather than
  hardcoded at lines 6-10. SBATCH directives are static, so the mechanism is
  `run_pipeline.sh` passing `--mem`/`--time`/`--gres` overrides on the `sbatch`
  command line from registry values.
- Add the resolved values to the echoed benchmark-grid summary.
- 8B registry defaults: `--mem=64G`, `--gres=gpu:1` on a ≥40 GB device. Time
  limit stays at the current value until Batch 8 measures it; record that it is
  provisional.

Out of scope: choosing the final time limit; that is an output of the pilot.

Implementation steps:
1. Add `SBATCH_MEM`, `SBATCH_TIME`, `SBATCH_GRES` to the registry.
2. In `run_pipeline.sh`, add these as `sbatch` command-line overrides for the
   benchmark stage.
3. Add `--max-model-len` to the `python -m inference_benchmark.benchmark_prefix_cache`
   invocation.

Validation:
- Echo the constructed `sbatch` and `python` command lines for both tags without
  submitting; confirm `--max-model-len 2048` and the per-model resources appear.
- `bash -n`.
- Not validated: engine startup under the cap. Needs a GPU; covered by Batch 8.

Review notes:

### Batch 5: Analysis roofline parameterization

Status: done

Scope:
- `analysis/plot_report.py` takes the parameter count from the caller rather
  than defaulting to TinyLlama. `--n-params` already exists (line 443) with
  default `1.1e9`; the hardcoded caption at line 271
  (`"TinyLlama-1.1B on A100 @ ..."`) must use a `--model-label` argument.
- Add `--gpu-name` or fold the device into `--model-label` so the caption is not
  wrong when the run is on an H100 rather than an A100.

Out of scope: cross-model plots, a `model_name` column, any change to
`analyze_prefix_cache.py` or `plot_prefix_cache_results.py`.

Implementation steps:
1. Add `--model-label`, default derived from `--n-params` or an explicit string.
2. Replace the f-string at line 271 and the print at line 283.

Validation:
- Run `plot_report.py` against an existing merged CSV under `outputs/analysis/`
  with both `--n-params 1.1e9` and `--n-params 8e9`; confirm the caption and the
  roofline reference line change.
- Note in the handoff: the v5 CSVs used for this check are void as results, but
  are structurally valid inputs for exercising the plotting path.

Review notes:

### Batch 6: Runbook and configuration documentation

Status: done

Scope:
- `RUNBOOK.md`: rewrite the staging, data-prep, and submission sections for the
  `MODEL_TAG` workflow; document the namespaced artifact tree; document the
  registry fields.
- Mark the case-count and time-limit table as provisional pending Batch 8.
- `AGENT_CONFIG.md` document map: add `SPECS/` and `TASKS/` entries if absent.
- `README.md` and `inference_benchmark/README.md`: update the example commands
  that name TinyLlama explicitly.

Out of scope: `research.md` §2.6 status updates; that file records decisions, not
progress.

Validation:
- Every path, line reference, and command in the changed sections checked against
  the working tree.

Review notes:

### Batch 7: Stage 8B weights and prepare data (user, cluster)

Status: pending

Scope: on a login node, `MODEL_TAG=Llama-3.1-8B-Instruct bash prep_login.sh`
then `MODEL_TAG=Llama-3.1-8B-Instruct bash prepare_data.sh`.

Prerequisites: `HF_TOKEN` exported; Llama-3.1 license accepted; Batches 1-3
merged and pulled on the cluster.

Validation:
- `$PROJECT_ROOT/models/Llama-3.1-8B-Instruct` contains config and weight shards.
- `$SCRATCH_ROOT/Llama-3.1-8B-Instruct/processed/rag_examples.jsonl` exists;
  record the kept/dropped counts printed by the token-budget filter.

Review notes:

### Batch 8: Pilot run (user, cluster)

Status: pending

Scope: one mutation type, both layouts, both cache modes, ~50 cases.

Validation, all from the produced JSONLs:
- Engine started at `max_model_len=2048`.
- `original` layout produces non-empty `text`; the ~97% empty-output rate is
  gone.
- `reference_answers` present in record metadata.
- `exact_reuse` gain clearly exceeds the `unrelated_control` noise floor.
- Record measured mean per-request wall-clock and TTFT per layout and cache mode.
  These replace the estimated timing table in `RUNBOOK.md`.

Blocks Batch 9 until the measured per-case cost sets the case count and time
limit.

Review notes:

### Batch 9: Full 8B run (user, cluster)

Status: pending

Scope: `chunk_reorder`, `typo`, `formatting`, `synonym_substitution` × 2 layouts
× 2 cache modes, at the case count and time limit set by Batch 8.

Validation:
- All 16 benchmark JSONLs plus summaries present under the 8B namespace.
- Analysis stage completes; `<tag>.summary.json` produced per mutation type.
- Noise floor confirms `unrelated_control` remains near zero.

Review notes:

### Batch 10: Record results and finalize documentation

Status: pending

Scope:
- Replace the provisional timing table in `RUNBOOK.md` with Batch 8 measurements.
- Create `FINDINGS.md` with the 8B result set, per the structure agreed in
  `research.md` §3.
- Record in `research.md` §2.6 that item 1 is complete.

Validation: numbers in the documents match the produced summary JSONs.

Review notes:

## Review history

- **Batch 1** — reviewed by Codex, approved, no blocking or non-blocking issues.
  Report: `TASKS/handoffs/2026-08-19-codex-review-model-scale-batch1.md`.
  Claims independently re-verified by Claude (export propagation to all four
  sbatch stages, namespaced consumers, `paths.py` unused, `prep_login.sh:13`
  legacy dirs unread). Committed.
- **Prerequisite fixes** (chat template, reference answers, `max_model_len`) —
  reviewed by Codex, NOT approved, two blockers raised. Report:
  `TASKS/handoffs/2026-08-19-codex-review-prereq-fixes.md`. Both confirmed and
  fixed:
  - per-request reference answers in `case_builder.py`, so an
    `unrelated_control` followup no longer carries the wrong record's answer and
    a meaning-changing followup carries none;
  - engine-visible model-token overlap (`BackendBase.token_overlap`) recorded
    per case and used for plot axes, replacing whitespace-word counts that were
    mislabelled as tokens.
  The post-review changes were committed without a second Codex pass. Worth a
  follow-up review pass; not a blocker for Batch 2, which touches different
  files.
- **Batch 2** — reviewed by Codex, NOT approved, one blocking issue. Report:
  `TASKS/handoffs/2026-08-19-codex-review-model-scale-batch2.md`. Confirmed and
  fixed: `model_dir_complete` accepted a partially downloaded *sharded* model,
  because it returned success on finding any one weight file. Llama-3.1-8B is
  sharded, so an interrupted download of the model this batch exists to stage
  would have been treated as complete and failed later on an offline compute
  node. It also accepted `tokenizer_config.json` standing in for a missing
  tokenizer vocabulary. The predicate now verifies every shard named in
  `*.index.json`, rejects `*-of-*` files with no index, and requires the
  tokenizer to load with `local_files_only=True`.
  Codex judged the repeated venv/install/pre-cache on a second staging run to
  be an operational optimization rather than a Batch 2 defect; recorded as an
  open item rather than fixed here.
- **Batch 3** — reviewed by Codex, approved, R4 met, no blocking issues. Report:
  `TASKS/handoffs/2026-08-19-codex-review-model-scale-batch3.md`. Verified that
  deleting the duplicate `MAX_PROMPT_TOKENS` default leaves it bound on every
  path, including the `prep_login.sh` parent-to-child boundary and when exported
  empty.
  Two non-blocking findings were fixed: the chat-template overhead is **15**
  tokens, not the 6 quoted in the review and in the earlier implementation note
  (6 is the leading header alone, which is what shifts the divergence position;
  15 includes the trailing `</s>` and assistant marker), giving a margin of 169
  against `max_model_len=2048`; and `README.md:45` documented the
  pre-namespacing processed path, which would silently load another model's
  example set.
  Deferred: `RUNBOOK.md:49` carries the same legacy path (Batch 6); provenance
  is stdout-only, so an explicit override or registry change can leave an
  unattributable same-namespace processed file — a sidecar manifest is a new
  requirement, not a Batch 3 repair; and
  `prompt_mutation/prepare_rag_data.py:71` keeps a second `1800` default that
  applies only to direct invocation.
- **Batch 4** — reviewed by Codex, approved, R5 and R6 met, no blocking issues.
  Report: `TASKS/handoffs/2026-08-19-codex-review-model-scale-batch4.md`.
  A bug introduced during implementation was found and fixed before review:
  `"${extra[@]}"` on an empty array is an unbound-variable error under `set -u`
  in bash < 4.4 (macOS ships 3.2), and three of the four stages pass no extra
  args, so 3 of 4 submissions failed and the benchmark stage received
  `--dependency=afterok:` with an empty job id. Replaced with
  `${extra[@]+"${extra[@]}"}`.
  Codex verified the guard under bash 3.2 including arguments containing spaces
  and empty strings, ran all 16 SKIP_* combinations against both tags (32 runs,
  all correctly chained, no empty dependency), confirmed resource overrides
  appear only on the benchmark stage, and confirmed `--max-model-len` reaches
  both vLLM construction paths with omission preserving prior behavior. It found
  no sibling array-expansion defect: `layout_jsonls` is unguarded but
  unreachable-empty because `pipeline_config.sh:115` uses `:=`, so an empty
  `STRATEGIES` is replaced by the default. Independently re-verified.
  Codex endorsed the deviation from the plan on the 8B time limit: provisional
  `06:00:00` over the planned `02:30:00`, on the grounds that a longer limit does
  not force longer runtime whereas the shorter one risks killing the sweep after
  consuming the allocation. Batch 8 must still replace it with a measured value.

  **Open operational risk:** `SBATCH_GRES=gpu:1` does not guarantee a device with
  the >= 40 GB that R6 requires. On a heterogeneous GPU pool the 8B job can be
  allocated an undersized device and fail at engine startup. Set a typed request
  (e.g. `SBATCH_GRES=gpu:a100:1`) for the cluster in use. Recorded in
  `RUNBOOK.md` during Batch 6.
- **Batches 5 and 6** — reviewed together by Codex, NOT approved, four blocking
  issues. Report: `TASKS/handoffs/2026-08-20-codex-review-model-scale-batch56.md`.
  All four confirmed and fixed:
  1. *Unit incoherence.* Batch 5 fixed only the x-axis label; the title, legend,
     stdout, and roofline still called a whitespace-word slope `mu_s/token` and
     compared it against a per-model-token peak. Plots 4 and 5 repeated it, and
     plot 5 used the word-based `first_divergence_token` while calling it cached
     tokens despite `first_divergence_model_token` now existing. Fixed with a
     single `_length_unit()` resolver threaded through all three plots; the
     theoretical line and achieved-peak figure are now WITHHELD when the unit is
     words rather than shown with a caveat. The theoretical legend text also
     disagreed with the computation (`2*params / peak * utilization` vs
     `2*params / (peak * utilization)`) and was corrected.
  2. *Identity decoupled from values.* `--model-label` and `--n-params` were
     independent, so a figure could name TinyLlama while computing an 8B
     roofline. Labels are now always rendered with the value they claim —
     `Llama-3.1-8B-Instruct (8.0e+09 params)` — making a mismatch visible.
  3. *Documented commands did not bind `MODEL_TAG`.* Worse than the original
     defect: `ROOT=$SCRATCH/prefix_caching/$MODEL_TAG` with the variable unset
     collapses to the legacy pre-namespacing path the surrounding text warns
     about. Bound in every command across all four documents.
  4. *The timing table was withdrawn, not annotated.* Codex found the request
     count understated by 2x: `submit_interface_benchmark.sh` loops over both
     cache modes and each invocation sweeps both layouts, so a job issues
     `2 x 2 x 2 = 8` requests per case, not 4. Combined with the artifact-derived
     source measurements, the table produced an actionable recommendation from
     invalid data. Replaced with the measurement procedure and the formula.

  Also fixed, beyond the review's scope: the pilot command written into
  `RUNBOOK.md` did not work. `--max-cases` existed in the Python CLI but nothing
  wired it from the shell, so Batch 8 had no way to run a small pilot without
  rewriting the shared processed example set. `MAX_CASES` is now plumbed through
  `pipeline_config.sh` -> the `run_pipeline.sh` child-job export ->
  `submit_interface_benchmark.sh`, passed only when non-empty.

  Verified all three token-source states (engine column populated, absent, and
  present-but-all-null): unit, axis, legend, title, and stdout agree in each, and
  all five figures render in both modes.

## Final acceptance criteria

- One variable selects the model; artifacts for different models never collide.
- `Llama-3.1-8B-Instruct` stages idempotently, with a clear failure when
  `HF_TOKEN` is missing.
- A full 8B result set exists for four RAG meaning-preserving mutation types,
  produced with the chat template applied.
- The `original` layout generates real output; the empty-output artifact is gone.
- `reference_answers` is present in benchmark records and absent from prompts.
- TinyLlama-1.1B remains runnable by setting `MODEL_TAG` back.
- `RUNBOOK.md` reflects the model-parameterized workflow with measured timings.
