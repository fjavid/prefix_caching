# Runbook

Operational procedure for running this pipeline on Alliance Canada clusters.

For what each stage does, see `README.md` and the per-module READMEs. This file
covers order of operations, node placement, and per-model settings.

## Execution model

The pipeline is designed for a SLURM cluster with the following constraint:
**login nodes have network access; compute nodes do not.** Everything requiring
a download — Python wheels, model weights, Hugging Face datasets — must happen
on a login node in advance. `pipeline_config.sh:59-68` (`set_offline_env`) sets
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1` for
every batch job, so a missing asset fails at job start rather than downloading.

Path layout:

| Root | Variable | Contents |
|---|---|---|
| `$HOME/work/prefix_caching` | `PROJECT_ROOT` | repo, `.venv/`, `models/`, `hf_cache/`, `wheelhouse/` |
| `$SCRATCH/prefix_caching` | `SCRATCH_ROOT` | `processed/`, `mutation/`, `prompt_organization/`, `benchmark_results/`, `analysis/` |
| `outputs/` in the repo | — | local mirror; rsync target after a run |

Default account is `def-mmehride` (`pipeline_config.sh:23`).

## Stage 1 — Login node, once per cluster

```bash
cd $HOME/work/prefix_caching
bash prep_login.sh
```

Builds `.venv` from the Alliance wheelhouse (`pip install --no-index`), installs
`vllm==0.20.0`, downloads TinyLlama-1.1B, and pre-caches the datasets and
validation models the compute nodes will need offline.

Note `requirements.txt` must not contain `vllm`; the version is pinned inside
`prep_login.sh` because it comes from the cluster wheelhouse. Check what is
available with `avail_wheels "vllm"` before changing the pin.

## Stage 2 — Login node, once per data configuration

```bash
bash prepare_data.sh
```

Downloads and preprocesses the RAG source data into
`$SCRATCH_ROOT/processed/rag_examples.jsonl`. Idempotent.

Defaults: `LLukas22/nq-simplified`, `SPLIT=train[:1000]`, `MAX_SAMPLES=1000`,
`MIN_CHUNKS=3`, `MAX_CHUNKS=4`, `MAX_CHUNK_WORDS=200`, `MAX_PROMPT_TOKENS=1800`.

`MAX_PROMPT_TOKENS` is a context budget sized for a 2048-token window with
`max_new_tokens=64`. It is applied after rendering, using the tokenizer at
`TOKENIZER_PATH` (defaults to `MODEL_PATH`). Changing the model changes the
tokenizer, which changes which examples survive the filter — see
[Changing the model](#changing-the-model).

## Stage 3 — Submit the pipeline

```bash
MUTATION_TYPES="chunk_reorder typo formatting synonym_substitution" \
  ./run_pipeline.sh
```

Submits one independent `afterok`-chained SLURM sequence per mutation type:
`build_mutation` → `apply_layouts` → `benchmark` → `analyze`. Derived paths embed
`TAG=${WORKLOAD}_${MUTATION_TYPE}`, so parallel chains do not collide.

Knobs, all overridable by export (`pipeline_config.sh:29-35`):

| Variable | Default |
|---|---|
| `WORKLOAD` | `rag` |
| `SEMANTIC_CLASS` | `meaning_preserving` |
| `GENERATION_CLASS` | `algorithmic` |
| `MUTATION_TYPE` / `MUTATION_TYPES` | `chunk_reorder` |
| `STRATEGIES` | `original stable_first` |
| `CACHE_MODES` | `off on` |

Stage skips: `SKIP_MUTATION`, `SKIP_LAYOUTS`, `SKIP_BENCHMARK`, `SKIP_ANALYSIS`
(set to `1`). To re-benchmark without regenerating prompts:

```bash
SKIP_MUTATION=1 SKIP_LAYOUTS=1 ./run_pipeline.sh
```

The benchmark stage starts one vLLM engine per cache mode and sweeps all layouts
on it. This is deliberate: constructing a fresh engine per layout introduced
~0.5 ms of per-case variance, which is comparable to the effect being measured.

## Stage 4 — Monitor

```bash
squeue -u $USER
```

Logs land in the submit directory as `<jobname>-<jobid>.out` / `.err`.

## Stage 5 — Retrieve and analyze

The analysis stage runs automatically at the end of each chain. To re-run it
without SLURM — on the login node or locally after rsync:

```bash
rsync -av <cluster>:/scratch/$USER/prefix_caching/benchmark_results outputs/
RESULTS_ROOT=outputs/benchmark_results ANALYSIS_DIR=outputs/analysis \
  bash analyze_local.sh
```

Analysis requires only pandas, numpy, and matplotlib. Headline metric is
`ttft_gain_seconds`. `<prefix>.summary.json` is the canonical text report; see
`analysis/README.md` for how to read it.

## Changing the model

Per-model settings. Only `MODEL_PATH` is a rename; the rest require action.

| Setting | Location | TinyLlama-1.1B-Chat-v1.0 | Llama-3.1-8B-Instruct |
|---|---|---|---|
| `MODEL_PATH` | `pipeline_config.sh:27` | `$PROJECT_ROOT/models/TinyLlama-1.1B-Chat-v1.0` | `$PROJECT_ROOT/models/Llama-3.1-8B-Instruct` |
| Weights on disk | — | 2.2 GB (fp16) | ~16 GB (bf16) |
| Hugging Face access | login node | ungated | gated; accept license, export `HF_TOKEN` |
| Declared context | model config | 2048 | 131072 |
| `max_model_len` to set | not currently settable | 2048 | **2048** (not the declared 131072) |
| `--gpu-memory-utilization` | `benchmark_prefix_cache.py:149` | 0.85 | 0.85 |
| `--gres` | `submit_interface_benchmark.sh:10` | `gpu:1`, ≥16 GB | `gpu:1`, ≥40 GB (A100-40G / H100) |
| `--mem` (host) | `submit_interface_benchmark.sh:9` | 32G | 64G |
| `--time` | `submit_interface_benchmark.sh:6` | 02:30:00 | 06:00:00 at 3000 cases (see below) |
| `--max-new-tokens` | CLI default | 64 | 64 |
| Tokenizer vocab | `TOKENIZER_PATH` | 32k | 128k |
| `MAX_PROMPT_TOKENS` | `prepare_data.sh` | 1800 | re-derive |

### Required code change before running an 8B model

`max_model_len` is never passed to vLLM. `inference_benchmark/vllm_backend.py:54-59`
constructs `AsyncEngineArgs` with only `model`, `enable_prefix_caching`,
`gpu_memory_utilization`, and `trust_remote_code`; the synchronous path at line
70 is the same. A model declaring a 131072-token context will cause vLLM to size
the KV cache for the full window and fail allocation on a 40 GB device.

Add a `--max-model-len` flag through `benchmark_prefix_cache.py` →
`benchmark_config.py` → both engine constructors in `vllm_backend.py`.

### Sizing `max_model_len`

Required window is `MAX_PROMPT_TOKENS + max_new_tokens = 1800 + 64 = 1864`.

Measured against the v5 runs (all `outputs/benchmark_results/*.jsonl`), the
largest prompt actually produced was **1615 tokens**, p99 was 1295, and the mean
was 584. So 2048 covers the workload with margin.

Set `max_model_len=2048` for **both** models. For TinyLlama this equals its
native window. For Llama-3.1 it is a deliberate cap far below the declared
131072: prompts are budget-limited by `MAX_PROMPT_TOKENS`, so allocating a 128k
window would reserve KV cache for context the workload never uses, and fails
allocation on a 40 GB device.

If `MAX_PROMPT_TOKENS` is raised later, raise `max_model_len` to
`MAX_PROMPT_TOKENS + max_new_tokens`, rounded up to a multiple of the 16-token
block size.

### How many cases fit at 8B

**KV cache is not the binding constraint; wall-clock time is.**

KV footprint per token, bf16, from each model's attention configuration
(`2 × layers × kv_heads × head_dim × 2 bytes`):

| Model | Layers | KV heads | Head dim | Per token |
|---|---|---|---|---|
| TinyLlama-1.1B | 22 | 4 | 64 | 22 KiB |
| Llama-3.1-8B | 32 | 8 | 128 | 128 KiB |

On a 40 GB device at `gpu_memory_utilization=0.85` (~34 GB usable), 8B weights
take ~16 GB and runtime overhead ~2 GB, leaving ~16 GB for KV. At 128 KiB/token
that is ~131k tokens, or ~64 concurrent sequences at the full 2048 window and
~220 at the 584-token mean. The benchmark runs one `(base, followup)` pair at a
time and resets the cache between pairs, so it never approaches this.

Time budget, measured from the v5 runs. Mean TTFT was 12–13 ms across every
layout and cache mode. Mean end-to-end request time varied by layout because
decode length varies: 16 ms for `original`, up to 190 ms for `stable_first` on
`chunk_reorder`. Total measured request time per layout ranged from 95 s
(`chunk_reorder_original`) to 1138 s (`chunk_reorder_stable_first`) at 3000
cases.

Scaling to 8B: prefill is compute-bound and decode is bandwidth-bound, both
roughly linear in parameter count, giving ~7× (8B / 1.1B = 7.3, partly offset by
better GPU utilization at the larger size). A benchmark job sweeps both layouts
on one engine, so per-case cost is `2 layouts × 2 requests × mean request time`.

| | Best case (`original`-like) | Worst case (`chunk_reorder` + `stable_first`) |
|---|---|---|
| Mean request time at 8B | ~110 ms | ~1330 ms |
| Cost per case (2 layouts × 2 requests) | ~0.44 s | ~5.3 s |
| Cases in `--time=02:30:00` | ~19000 | ~1580 |
| Cases in `--time=06:00:00` | ~47000 | ~3950 |

Subtract ~10 min per job for engine startup; loading 16 GB of weights from
`$PROJECT_ROOT/models` is substantially slower than TinyLlama's 2.2 GB.

**Recommendation: keep 3000 cases and raise `--time` to `06:00:00`.** At 3000
cases the worst mutation type needs ~4.4 h of compute plus startup, which
exceeds the current `02:30:00` limit and would be killed mid-run. Holding the
case count keeps 1000 samples per relation cell, so confidence intervals stay
comparable to the 1.1B results and the two model sizes remain directly
comparable.

If a 6 h allocation is unavailable, cap at **1500 cases** under `02:30:00`.
This halves per-cell sample size, widening confidence intervals by ~1.41×
(they scale as 1/sqrt(N)). Acceptable, because the effect size at 8B should be
roughly 7× larger while the noise floor grows by less.

### Downloading a gated model

On the login node, with network available:

```bash
export HF_HOME="$PROJECT_ROOT/hf_cache"
export HF_HUB_DISABLE_XET=1
hf download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir "$PROJECT_ROOT/models/Llama-3.1-8B-Instruct"
```

Requires prior license acceptance on the model page and an `HF_TOKEN` in the
environment. `Qwen2.5-7B-Instruct` is ungated if that step is undesirable.

### Comparing results across models

The prompt-budget filter tokenizes with `TOKENIZER_PATH`. Llama-3.1's 128k vocab
encodes the same text in fewer tokens than TinyLlama's 32k vocab, so a different
subset of examples passes `MAX_PROMPT_TOKENS`. For a controlled scaling
comparison, either fix the example set before filtering or record that the two
runs used different prompt sets.

Also re-derive the case count. The current configuration runs 3000 cases per
mutation type; per-case prefill cost at 8B is roughly 7× that of 1.1B.

## Known operational hazards

- `HF_HUB_OFFLINE=1` on compute nodes: any asset not pre-cached on the login
  node fails at job start.
- vLLM version is pinned to the cluster wheelhouse, not PyPI. `AsyncEngineArgs`
  fields drift across versions; `vllm_backend.py:50-53` filters kwargs against
  the installed dataclass rather than assuming a schema.
- `$SCRATCH` is subject to purge policy. Sync results to `outputs/` or off-cluster
  before they expire.
- The benchmark resets the prefix cache between every `(base, followup)` pair so
  the `unrelated_control` baseline is not contaminated by earlier cases.
