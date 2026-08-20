# Review

## Scope

Reviewed Batches 5 and 6 in the idea-to-code workflow: `analysis/plot_report.py`,
`RUNBOOK.md`, `README.md`, `analysis/README.md`, and
`inference_benchmark/README.md`. Cross-checked their claims read-only against the
specified pipeline, analysis, and backend files. Batches 7-10, historical
handoffs, and changes outside the five in-scope files were not reviewed. No
SLURM job was submitted and no model or dataset was downloaded. The v5 outputs
were used only as structurally valid plotting inputs; they are not valid results.

## Blocking issues

1. `analysis/plot_report.py:256-337` still produces a false roofline figure when
   engine token counts are unavailable. The requested all-null-column test does
   make `_estimate_total_tokens` return the whitespace-word reconstruction, and
   the x-axis correctly says words. The title, fit legend, theoretical legend,
   and stdout nevertheless label the measured slope as `mu_s/token` and compare
   it with a model-token roofline. The function's own warning says that
   comparison is invalid. `plot_ttft_on_vs_shared_ratio` repeats the error in
   its colorbar, title, and stdout (`analysis/plot_report.py:340-420`), and
   `plot_ttft_gain_vs_first_divergence` always uses the whitespace-word
   `first_divergence_token` while calling it cached tokens
   (`analysis/plot_report.py:425-485`), despite the merged schema now carrying
   `first_divergence_model_token`. The theoretical legend is also inconsistent
   with the computation: it prints `2*params / peak * utilization`, while line
   273 computes `2*params / (peak * utilization)`. These paths can put wrong
   units and a meaningless hardware comparison into a writeup figure.

2. Model/device names and the values used by the roofline are independent
   inputs (`analysis/plot_report.py:494-521`). For example,
   `--n-params 8e9 --model-label TinyLlama-1.1B --gpu-peak-tflops 990
   --gpu-label A100` is accepted; stdout reports an 8B/990-TFLOPS computation
   and an `A100 dense peak`, while the title identifies TinyLlama on A100. The
   value-derived defaults are internally derived from the numeric defaults, but
   they are not run provenance: applying default flags to 8B data asserts
   `1.1e+09 params on 312 TFLOPS device`. Since the CSVs do not identify the
   model or allocated GPU, the current implementation cannot meet the acceptance
   criterion that a figure never asserts a model or device it was not produced
   with. R9 is therefore not met as stated.

3. Several documented commands do not bind the model whose namespace they use.
   The re-benchmark command at `RUNBOOK.md:153` omits `MODEL_TAG`; in a fresh
   shell it silently selects TinyLlama even when the surrounding procedure is
   for Llama-3.1-8B. `analysis/README.md:73` constructs
   `$SCRATCH/prefix_caching/$MODEL_TAG` without setting `MODEL_TAG`; when it is
   unset, this collapses to the legacy pre-namespacing path that lines 68-70
   explicitly warn can contain another model's data. `README.md:43-45` has the
   same failure mode, and the direct benchmark example at
   `inference_benchmark/README.md:45-48` also assumes both `PROJECT_ROOT` and
   `MODEL_TAG` already exist. These are not self-contained commands and can
   fail or read/benchmark the wrong model.

4. The timing table should be removed, not retained under an annotation.
   `submit_interface_benchmark.sh` loops over both cache modes and both layouts,
   so a complete job makes `2 cache modes x 2 layouts x 2 requests` per case.
   `RUNBOOK.md:310-318` counts only `2 layouts x 2 requests`. More fundamentally,
   the source measurements are not estimates of post-fix decode cost: in the
   checked `chunk_reorder` files, 97.5% of `original` followups emitted at most
   one token, while 99.2-99.3% of `stable_first` followups reached 64 tokens.
   Keeping the derived table and the bold 3000-case/6-hour working plan gives an
   actionable recommendation from invalid data. Leave only the pilot gate and a
   provisional pilot allocation; add a timing table after Batch 8. R10's PRD
   requirement explicitly calls for pilot measurements, so R10 is not yet met.

## Non-blocking issues

- The only surviving `file:NN` reference, `RUNBOOK.md:138`, points to
  `pipeline_config.sh:110-116`. That range contains the scalar defaults shown in
  the table, but not `MUTATION_TYPES`; the plural override is resolved in
  `run_pipeline.sh`, not in the cited range.

## Missing tests

- No maintained automated test checks plot units and annotations for all three
  token-source states: engine-token column populated, absent, and present but
  entirely null.
- No test ensures the model/device identity shown in the title is coupled to
  the numeric roofline inputs.
- No documentation smoke check expands the command blocks in a clean shell and
  rejects an empty `MODEL_TAG` or `PROJECT_ROOT`.

## Commands and results

- `.venv/bin/python -m py_compile analysis/plot_report.py` — passed.
- `python -m analysis.plot_report` with default model/GPU flags, with output
  redirected to `tmp/review-batch56.xAarvc/default` — completed. The default
  title was visible and not truncated. The input CSVs lack engine token counts,
  exposing the word/token roofline defect above.
- The requested 8B/H100 invocation, output to
  `tmp/review-batch56.xAarvc/custom` — completed. The
  `Llama-3.1-8B-Instruct on H100` title was visible and not truncated; the
  numeric line changed from 12.821 to 29.385 microseconds per token.
- Four CSV copies with `followup_prompt_model_tokens` present but entirely null,
  output to `tmp/review-batch56.xAarvc/null_output` — completed. The estimator
  values exactly matched the missing-column fallback, and the x-axis correctly
  said `whitespace words`; the title, legend, and stdout still said tokens.
- TinyLlama tokenizer loaded from the existing local Hugging Face snapshot with
  offline flags. Across 400 actual rendered prompts from all eight layout files,
  total chat-template overhead was always 15 tokens. Across 200 base/mutated
  pairs, the first-divergence shift was always +6 tokens. The RUNBOOK's 15-token
  total overhead, `1800 + 15 + 64 = 1879`, and 169-token margin are correct; the
  backend's +6 statement correctly refers only to the leading header.
- Namespaced-path grep found only the intentional warning at
  `analysis/README.md:68`; command expansion found the unbound-variable cases
  listed above. Registry values and namespaced roots matched the documentation
  for both registered tags. Unknown-tag handling listed both valid tags.
- `git diff --check` on the five in-scope files — passed.

## Suggested fixes

Narrow execute prompt:

> Modify only `analysis/plot_report.py`, `RUNBOOK.md`, `README.md`,
> `analysis/README.md`, and `inference_benchmark/README.md`. Select the plot unit
> once from the actual source column and propagate it through axes, colorbars,
> legends, titles, and stdout. Suppress the per-model-token theoretical line and
> achieved-peak claim when the fallback unit is words. Prefer
> `first_divergence_model_token` for plot 5, with an explicitly word-labelled
> fallback. Correct the roofline formula text and make model/device identity a
> coherent, provenance-safe configuration rather than independent free-form
> claims. Bind `MODEL_TAG` (and `PROJECT_ROOT` where used) in every documented
> command. Remove the invalid timing table and 3000-case recommendation pending
> Batch 8. Add focused tests for populated, absent, and all-null engine-token
> columns, then repeat the requested visual checks.

## Final git status

```text
 M README.md
 M RUNBOOK.md
 M TASKS/model-scale-8b.md
 M analysis/README.md
 M analysis/plot_report.py
 M inference_benchmark/README.md
?? TASKS/handoffs/2026-08-20-codex-review-model-scale-batch56.md
```

`TASKS/model-scale-8b.md` was already modified and is outside the implementation
scope of this review. The handoff report is the only file written by this review.

## Approval recommendation

Not approved. R9 can still emit internally inconsistent or falsely attributed
roofline figures, and R10 contains commands that can select the wrong namespace
plus a timing recommendation derived from invalid measurements and incomplete
request-count arithmetic.
