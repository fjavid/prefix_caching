# Review

## Scope

Reviewed the Batch 4 diff in `pipeline_config.sh`, `run_pipeline.sh`, and
`submit_interface_benchmark.sh` against R5 and R6 in
`SPECS/model-scale-8b.md`. Cross-checked `RUNBOOK.md`, the other `submit_*.sh`
scripts, and the existing `BackendConfig`/vLLM paths only where required to
verify submission behavior and `max_model_len` propagation. Batches 5-10 were
not reviewed. No SLURM job or GPU inference was run.

## Blocking issues

None.

## Non-blocking issues

- `SBATCH_GRES=gpu:1` does not itself enforce the R6 requirement of a device
  with at least 40 GB. This matches the explicitly requested registry value and
  is individually overridable, but a heterogeneous cluster or partition must
  use a typed request such as `SBATCH_GRES=gpu:a100:1`; otherwise the 8B job can
  allocate an undersized GPU and fail during engine startup.

## Missing tests

- The real SLURM allocation and vLLM startup remain cluster-only checks for the
  Batch 8 pilot. `sbatch` is not installed locally, so command-line precedence
  was verified from Slurm's documented option precedence and the constructed
  argv, not by querying a scheduler.
- Only Bash 3.2.57 is installed on this host. The empty/non-empty guard was
  executed under 3.2; no modern Bash executable was available for the requested
  second runtime check.

## Commands and results

- `bash -n pipeline_config.sh run_pipeline.sh submit_interface_benchmark.sh`:
  passed.
- Bash 3.2 empty-array probes: unguarded `"${a[@]}"` failed with
  `a[@]: unbound variable`; `${a[@]+"${a[@]}"}` emitted zero words for an empty
  array and preserved all three non-empty test arguments, including an argument
  containing a space and an empty-string argument. The guard at
  `run_pipeline.sh:122,125` is correct.
- Array-expansion scan of the three in-scope files found no sibling instance of
  the same defect. `vars` is statically non-empty, `pairs` always receives the
  pinned pipeline variables, and `bench_args` always receives three registry
  options. `layout_jsonls` is unguarded, but supported configurations cannot
  reach it empty: unset or empty `STRATEGIES` is replaced by
  `original stable_first`, while the single-strategy override requires a
  non-empty `STRATEGY`. A whitespace-only strategy list is invalid input, not a
  supported empty layout sweep.
- Stubbed-`sbatch` matrix: ran both registered model tags across all 16
  `SKIP_MUTATION`/`SKIP_LAYOUTS`/`SKIP_BENCHMARK`/`SKIP_ANALYSIS` combinations
  under Bash 3.2 with temporary `HOME` and `SCRATCH`. All 32 runs passed. Each
  submitted subset was chained in stage order; the first submitted stage had no
  dependency; subsequent stages used the preceding non-skipped job ID; no
  `--dependency=afterok:` with an empty ID appeared.
- The same matrix confirmed that `--mem`, `--time`, and `--gres` occur only on
  `submit_interface_benchmark.sh`: TinyLlama received
  `32G/02:30:00/gpu:1`, and Llama received `64G/06:00:00/gpu:1`. A separate
  run confirmed independent overrides `77G/07:08:09/gpu:h100:2` replaced all
  three registry defaults.
- Slurm defines command-line `sbatch` options as taking precedence over
  `#SBATCH` directives. The constructed resource arguments precede the script
  path, so they are command-line options. The static
  `32G/02:30:00/gpu:1` directives remain a valid TinyLlama fallback for bare
  `sbatch submit_interface_benchmark.sh`.
- Direct benchmark-script probes with a stubbed `python` command confirmed both
  registered tags issue both cache-mode invocations with
  `--max-model-len 2048` and their model-specific path.
- `.venv/bin/python` CLI/engine probe confirmed
  `--max-model-len 2048` parses into `BackendConfig`, passes through
  `make_backend`, and reaches both `AsyncEngineArgs` and offline `LLM` as
  `max_model_len=2048`. Omitting the flag leaves `BackendConfig.max_model_len`
  as `None` and omits the kwarg from both constructors, preserving prior
  behavior. The first version of this test harness left chat templating enabled
  and attempted to load the fake tokenizer path; rerunning with
  `--no-chat-template` isolated the engine-argument test and passed.
- The cap covers the configured upper bound:
  `1800 + 15 + 64 = 1879 < 2048`, leaving 169 tokens.

## Suggested fixes

No code fix is required for Batch 4. Before an 8B run on a GPU pool containing
devices below 40 GB, set `SBATCH_GRES` to a cluster-supported GPU type that
guarantees the required memory.

The provisional `06:00:00` limit is the correct choice. `02:30:00` has a known
risk of killing the current 3000-case sweep after consuming the allocation.
The v5-derived 4.4-hour estimate is not suitable as a final measurement because
decode lengths were affected by the pre-chat-template behavior, but that
uncertainty does not justify retaining a limit already below the available
estimate. A six-hour limit does not force six hours of runtime; Batch 8 should
replace it with the measured value before the full sweep.

## Final git status

At review completion:

```text
 M TASKS/model-scale-8b.md
 M pipeline_config.sh
 M run_pipeline.sh
 M submit_interface_benchmark.sh
?? TASKS/handoffs/2026-08-19-codex-review-model-scale-batch4.md
```

`TASKS/model-scale-8b.md` contains the existing Batch 4 status change from
`pending` to `in_progress (implemented, awaiting review)` and was outside the
implementation review scope. No generated-output directory was modified.

## Approval recommendation

Approved. R5 and R6 are met: the engine cap reaches both vLLM construction
paths, resource overrides have the required precedence and stage scope, and the
dependency chain is correct for every skip combination. The real 8B allocation
still requires the Batch 8 pilot and a GPU request that guarantees at least
40 GB on the selected cluster.
