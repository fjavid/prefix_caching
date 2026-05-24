# inference_benchmark

This subdirectory contains the benchmark harness for step 3 of the project:
running inference with and without prefix caching on prompt pairs built from
the `prompt_mutation/` pipeline.

## Files

- `backend_base.py` — common backend interface and generation result schema
- `vllm_backend.py` — first working backend using offline vLLM
- `sglang_backend.py` — scaffold for a future SGLang backend
- `benchmark_config.py` — experiment configuration dataclasses
- `metrics_utils.py` — request-level metrics helpers
- `case_builder.py` — converts mutation JSONL records into benchmark cases:
  exact reuse, partial reuse, and unrelated controls
- `inference_runner.py` — executes benchmark cases through a backend
- `benchmark_prefix_cache.py` — top-level CLI to run the benchmark

## Expected workflow

1. Generate mutated prompts with `prompt_mutation/build_mutation_dataset_v2.py`
2. Apply one or more layout strategies to each mutation JSONL via
   `prompt_organization/apply_layouts.py`; this writes per-layout JSONLs
   like `rag_chunk_reorder_original.jsonl`, `rag_chunk_reorder_stable_first.jsonl`.
3. Invoke `benchmark_prefix_cache.py` ONCE per cache mode, passing it ALL the
   layout JSONLs at the same time. The script starts a single vLLM engine
   and sweeps every layout back-to-back on it, writing one output JSONL
   per layout. Doing this per layout (instead of per-engine) eliminates a
   ~0.5 ms per-case variance previously caused by spinning up a fresh engine
   between layouts.
4. Compare follow-up request wall-clock time and TTFT across:
   - exact reuse
   - partial reuse
   - unrelated control

## Example

Cache OFF: one engine sweeping `original` and `stable_first`:

```bash
python -m inference_benchmark.benchmark_prefix_cache \
  --backend-name vllm \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --disable-prefix-caching \
  --layouts        original                                   stable_first \
  --mutation-jsonls outputs/prompt_organization/rag_chunk_reorder_original.jsonl \
                    outputs/prompt_organization/rag_chunk_reorder_stable_first.jsonl \
  --output-dir outputs/benchmark_results \
  --run-name-template 'rag_chunk_reorder_{layout}_cache_off'
```

Cache ON: identical command with `--enable-prefix-caching` and
`_cache_on` in the template. Each run produces one JSONL per layout.

## Notes

- Two key timing fields per request:
  - `ttft_seconds` — time-to-first-token (prefill-dominated; this is the
    metric prefix caching most directly affects).
  - `wall_clock_seconds` — end-to-end request time (prefill + decode). This
    is what we previously called `latency_seconds`. Renamed to avoid the
    common conflation of "latency" with TTFT in serving literature.
- The vLLM backend supports two modes:
  - `--use-async-ttft` (default): `AsyncLLMEngine` with streaming; measures
    both `ttft_seconds` and `wall_clock_seconds`.
  - `--no-async-ttft`: blocking offline `LLM.generate` path; `ttft_seconds`
    is `None`, `wall_clock_seconds` is still recorded.
- The runner does `warmup_iters` warmup requests ONCE per engine (default 2)
  and resets the prefix cache between every `(base, followup)` pair so the
  `unrelated_control` baseline is not contaminated by earlier cases.
- `sglang_backend.py` is scaffolded but not implemented yet.
