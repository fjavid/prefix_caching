# Review

## Scope

Reviewed Batch 1 against `SPECS/model-scale-8b.md` R1 and R2 in
`pipeline_config.sh`, `submit_build_mutation.sh`, `run_pipeline.sh`, and
`analyze_local.sh`. Cross-checked the artifact consumers in
`submit_apply_layouts.sh`, `submit_interface_benchmark.sh`,
`submit_analysis.sh`, `prepare_data.sh`, `prep_login.sh`, and `paths.py`.

The inference and prompt-mutation Python changes, `AGENT_CONFIG.md`, and
Batches 2-10 were not reviewed.

## Blocking issues

None.

## Non-blocking issues

None.

## Missing tests

None for the Batch 1 gate. The repository has no maintained automated test
suite; the required registry, override, path-resolution, and shell-syntax
checks were run manually.

## Commands and results

- `bash -n` on every root-level shell script: all nine scripts passed.
- Sourced `pipeline_config.sh` with `MODEL_TAG` unset: resolved the TinyLlama
  registry entry and all five artifact directories under
  `/scratch/test/prefix_caching/TinyLlama-1.1B-Chat-v1.0/`.
- Sourced with `MODEL_TAG=Llama-3.1-8B-Instruct`: resolved the 8B registry entry
  and all five artifact directories under
  `/scratch/test/prefix_caching/Llama-3.1-8B-Instruct/`.
- Sourced with `MODEL_TAG=UnknownModel`: exited 1 and listed both valid tags.
- Sourced with explicit non-empty overrides for `MODEL_REPO`, `MODEL_PATH`,
  `MAX_MODEL_LEN`, `MAX_PROMPT_TOKENS`, `N_PARAMS`, and `GATED`: all six values
  were preserved by the `:=` defaults.
- `git diff --check -- pipeline_config.sh submit_build_mutation.sh run_pipeline.sh analyze_local.sh`: passed.
- Repository-wide artifact reference scan: every cluster producer and consumer
  uses the namespaced variables from `pipeline_config.sh`. Mutation building
  reads `$PROCESSED_DIR` and writes below `$MUTATION_ROOT`; layout generation
  reads `$MUTATION_DIR` and writes `$LAYOUT_DIR`; benchmarking reads
  `$LAYOUT_DIR` and writes `$BENCH_DIR`; both analysis entry points read
  `$BENCH_DIR` and write `$ANALYSIS_DIR`.
- `run_pipeline.sh` includes `MODEL_TAG` in the single export bundle passed to
  build, layout, benchmark, and analysis jobs. Each stage then sources
  `pipeline_config.sh`, so every stage reconstructs the same model namespace.
- `prep_login.sh:13` creates empty legacy `benchmark_results/` and `analysis/`
  directories but no current stage reads or writes them. The submitted stages
  create their namespaced destinations themselves. Replacing this setup with
  the full per-model tree is correctly assigned to Batch 2 and does not create
  a Batch 1 collision path.
- `paths.py` defines the repo-local `outputs/` layout and is not imported by the
  cluster pipeline or any other Python module. It cannot redirect a cluster
  artifact into an un-namespaced path; the local-mirror documentation/layout
  can remain separate from R2.

## Suggested fixes

None.

## Final git status

The working tree is dirty. The four scoped files are modified:
`pipeline_config.sh`, `submit_build_mutation.sh`, `run_pipeline.sh`, and
`analyze_local.sh`. Eight modified files are outside this review:
`AGENT_CONFIG.md`, four files under `inference_benchmark/`, and three files
under `prompt_mutation/`. `RUNBOOK.md`, `SPECS/`, and `TASKS/` are untracked;
the last contains this report. Nothing is staged.

## Approval recommendation

Approved. R1 and R2 are met: both registry entries resolve correctly, unknown
tags fail clearly, and every cluster artifact path includes `MODEL_TAG`. No path
was found by which the two registered models can write the same cluster
artifact file.

## Commit instructions

```bash
git add -- pipeline_config.sh submit_build_mutation.sh run_pipeline.sh analyze_local.sh
git commit -m "Parameterize pipeline model artifact paths"
```
