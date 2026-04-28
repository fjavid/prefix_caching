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
2. Point `benchmark_prefix_cache.py` at the resulting JSONL file
3. Run the benchmark twice:
   - prefix caching disabled
   - prefix caching enabled
4. Compare follow-up request latency across:
   - exact reuse
   - partial reuse
   - unrelated control

## Example

Run without prefix caching:

```bash
python -m inference_benchmark.benchmark_prefix_cache   --backend-name vllm   --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0   --disable-prefix-caching   --mutation-jsonl-path ../data/mutation/scientific/meaning_changing/algorithmic/parameter_change/<file>.jsonl   --run-name tinyllama_no_cache
```

Run with prefix caching:

```bash
python -m inference_benchmark.benchmark_prefix_cache   --backend-name vllm   --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0   --enable-prefix-caching   --mutation-jsonl-path ../data/mutation/scientific/meaning_changing/algorithmic/parameter_change/<file>.jsonl   --run-name tinyllama_with_cache
```

## Notes

- The current vLLM backend uses the offline Python API.
- End-to-end latency is measured directly.
- TTFT is left as `None` for now in the offline vLLM path because the basic API
  does not expose it directly; a server-based timing path can be added later.
- `sglang_backend.py` is scaffolded but not implemented yet.
