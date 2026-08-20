# Runbook

Operational procedure for running this pipeline on Alliance Canada clusters.

For what each stage does, see `README.md` and the per-module READMEs. This file
covers order of operations, node placement, and per-model settings.

## Execution model

The pipeline is designed for a SLURM cluster with the following constraint:
**login nodes have network access; compute nodes do not.** Everything requiring
a download — Python wheels, model weights, Hugging Face datasets — must happen
on a login node in advance. `set_offline_env` in `pipeline_config.sh` sets
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1` for
every batch job, so a missing asset fails at job start rather than downloading.

Path layout:

| Root | Variable | Contents |
|---|---|---|
| `$HOME/work/prefix_caching` | `PROJECT_ROOT` | repo, `.venv/`, `models/<MODEL_TAG>/`, `hf_cache/`, `wheelhouse/` |
| `$SCRATCH/prefix_caching/<MODEL_TAG>` | `MODEL_SCRATCH_ROOT` | `processed/`, `mutation/`, `prompt_organization/`, `benchmark_results/`, `analysis/` |
| `outputs/` in the repo | — | local mirror; rsync target after a run |

Default account is `def-mmehride` (`SLURM_ACCOUNT` in `pipeline_config.sh`).

## Model selection

**One variable selects the model, and it namespaces everything.**

```bash
MODEL_TAG=Llama-3.1-8B-Instruct bash prep_login.sh
MODEL_TAG=Llama-3.1-8B-Instruct bash prepare_data.sh
MODEL_TAG=Llama-3.1-8B-Instruct ./run_pipeline.sh
```

`MODEL_TAG` must be set on **every** stage. It defaults to
`TinyLlama-1.1B-Chat-v1.0`, and an unknown tag exits with the list of valid ones.

Every `$SCRATCH_ROOT` artifact directory sits under `$SCRATCH_ROOT/$MODEL_TAG/`,
so a second model never overwrites the first model's results. Each model also
gets its own `processed/` example set, because the prompt-token budget is
filtered with that model's tokenizer.

The registry lives in `pipeline_config.sh` and resolves:

| Field | Purpose |
|---|---|
| `MODEL_REPO` | Hugging Face repo id used by `prep_login.sh` |
| `MODEL_PATH` | local weights, `$PROJECT_ROOT/models/$MODEL_TAG` |
| `MAX_MODEL_LEN` | engine context cap passed as `--max-model-len` |
| `MAX_PROMPT_TOKENS` | prompt-token budget enforced by `prepare_data.sh` |
| `N_PARAMS` | parameter count for the `plot_report.py` roofline |
| `GATED` | 1 if the repo needs license acceptance and `HF_TOKEN` |
| `SBATCH_MEM` / `SBATCH_TIME` / `SBATCH_GRES` | benchmark-job resources |

Every field is individually overridable by exporting it. To add a model, add a
case arm to the registry and stage it with `MODEL_TAG=<tag> bash prep_login.sh`.

## Stage 1 — Login node, once per cluster and per model

```bash
cd $HOME/work/prefix_caching
MODEL_TAG=Llama-3.1-8B-Instruct bash prep_login.sh
```

Builds `.venv` from the Alliance wheelhouse (`pip install --no-index`), installs
`vllm==0.20.0`, stages the selected model, and pre-caches the datasets and
validation models the compute nodes will need offline. Ends by running
`prepare_data.sh` for the same `MODEL_TAG`.

Staging is idempotent: an already-complete weights directory is left alone. A
directory is complete only when it has `config.json`, every weight shard named
in its `*.index.json`, and a tokenizer that loads with `local_files_only=True`.
An interrupted download therefore fails here on the login node rather than on an
offline compute node. If it reports an incomplete directory, remove it and re-run.

For a gated repo, accept the license on the model page and export `HF_TOKEN`
first; staging fails before attempting the download if no token is present.

Note `requirements.txt` must not contain `vllm`; the version is pinned inside
`prep_login.sh` because it comes from the cluster wheelhouse. Check what is
available with `avail_wheels "vllm"` before changing the pin.

**Staging a second model re-runs the whole venv build, the vLLM install, and the
dataset pre-cache before reaching the download.** That is wasteful but harmless;
the model download itself is skipped if already complete.

## Stage 2 — Login node, once per data configuration

```bash
MODEL_TAG=Llama-3.1-8B-Instruct bash prepare_data.sh
```

Downloads and preprocesses the RAG source data into
`$SCRATCH_ROOT/$MODEL_TAG/processed/rag_examples.jsonl`. Idempotent.

Defaults: `LLukas22/nq-simplified`, `SPLIT=train[:1000]`, `MAX_SAMPLES=1000`,
`MIN_CHUNKS=3`, `MAX_CHUNKS=4`, `MAX_CHUNK_WORDS=200`. `MAX_PROMPT_TOKENS` and
`TOKENIZER_PATH` come from the registry.

The budget is applied to the **untemplated** prompt, but the engine sees the
prompt after the chat template is applied, so the constraint is:

```
max_model_len >= max_prompt_tokens + chat_template_overhead + max_new_tokens
```

At current values that is `1800 + 15 + 64 = 1879` against `2048`, a margin of
169 tokens; 15 is the measured full-template overhead for TinyLlama-1.1B. Do not
raise `MAX_PROMPT_TOKENS` to `MAX_MODEL_LEN - max_new_tokens` — it would overflow.

**The processed file records no provenance.** Only the stdout of this stage says
which tokenizer and budget produced it, so keep the job log if you override
either. Two models cannot collide because of namespacing, but an override within
one namespace leaves an unattributable file.

## Stage 3 — Submit the pipeline

```bash
MODEL_TAG=Llama-3.1-8B-Instruct \
MUTATION_TYPES="chunk_reorder typo formatting synonym_substitution" \
  ./run_pipeline.sh
```

Submits one independent `afterok`-chained SLURM sequence per mutation type:
`build_mutation` → `apply_layouts` → `benchmark` → `analyze`. Derived paths embed
both `MODEL_TAG` and `TAG=${WORKLOAD}_${MUTATION_TYPE}`, so neither parallel
chains nor different models collide.

The benchmark stage receives `--max-model-len $MAX_MODEL_LEN` and per-model
`--mem` / `--time` / `--gres` as sbatch **command-line** options, which override
the static `#SBATCH` directives in `submit_interface_benchmark.sh`. Those
directives are the TinyLlama fallback and cannot read the registry, so
submitting `submit_interface_benchmark.sh` directly for an 8B model
under-requests memory and time. Go through `run_pipeline.sh`.

Knobs, all overridable by export. The scalar defaults are in
`pipeline_config.sh`; `MUTATION_TYPES` (plural) is resolved in `run_pipeline.sh`
and takes precedence over `MUTATION_TYPE` when set.

| Variable | Default |
|---|---|
| `WORKLOAD` | `rag` |
| `SEMANTIC_CLASS` | `meaning_preserving` |
| `GENERATION_CLASS` | `algorithmic` |
| `MUTATION_TYPE` / `MUTATION_TYPES` | `chunk_reorder` |
| `STRATEGIES` | `original stable_first` |
| `CACHE_MODES` | `off on` |
| `MAX_CASES` | unset (all cases) — set for a pilot |

Stage skips: `SKIP_MUTATION`, `SKIP_LAYOUTS`, `SKIP_BENCHMARK`, `SKIP_ANALYSIS`
(set to `1`). To re-benchmark without regenerating prompts:

```bash
MODEL_TAG=Llama-3.1-8B-Instruct SKIP_MUTATION=1 SKIP_LAYOUTS=1 ./run_pipeline.sh
```

`MODEL_TAG` is required here too. Omitted, it silently resolves to the default
tag and re-benchmarks TinyLlama against the 8B namespace's layout files — or
fails because they do not exist.

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
rsync -av <cluster>:/scratch/$USER/prefix_caching/<MODEL_TAG>/benchmark_results outputs/
RESULTS_ROOT=outputs/benchmark_results ANALYSIS_DIR=outputs/analysis \
  bash analyze_local.sh
```

Without those two overrides, `analyze_local.sh` reads the `MODEL_TAG`-namespaced
directories, which is what you want on the login node and not what you want
against an rsync'd copy.

Analysis requires only pandas, numpy, and matplotlib. Headline metric is
`ttft_gain_seconds`. `<prefix>.summary.json` is the canonical text report; see
`analysis/README.md` for how to read it.

For the roofline figure, pass the model and device explicitly — the defaults
describe TinyLlama on an A100 and will silently mislabel anything else:

```bash
python -m analysis.plot_report \
  --analysis-dir outputs/analysis --output-dir outputs/analysis/report \
  --n-params 8.0e9 --model-label Llama-3.1-8B-Instruct \
  --gpu-peak-tflops 990 --gpu-label H100
```

`N_PARAMS` is carried in the registry for exactly this purpose.

## Per-model settings reference

`MODEL_TAG` resolves all of these; the table is for checking what a tag implies.

| Setting | TinyLlama-1.1B-Chat-v1.0 | Llama-3.1-8B-Instruct |
|---|---|---|
| `MODEL_REPO` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `meta-llama/Llama-3.1-8B-Instruct` |
| Weights on disk | 2.2 GB (fp16) | ~16 GB (bf16), sharded |
| Hugging Face access | ungated | gated; accept license, export `HF_TOKEN` |
| Declared context | 2048 | 131072 |
| `MAX_MODEL_LEN` | 2048 | **2048** — not the declared 131072 |
| `MAX_PROMPT_TOKENS` | 1800 | 1800 |
| `N_PARAMS` | 1.1e9 | 8.0e9 |
| `SBATCH_MEM` | 32G | 64G |
| `SBATCH_TIME` | 02:30:00 | 06:00:00 (**provisional**) |
| `SBATCH_GRES` | `gpu:1` | `gpu:1` — see caveat below |
| `--gpu-memory-utilization` | 0.85 | 0.85 |
| `--max-new-tokens` | 64 | 64 |
| Tokenizer vocab | 32k | 128k |

`MAX_MODEL_LEN` is capped well below Llama-3.1's declared window on purpose:
prompts are budget-limited, so reserving KV cache for 131072 tokens fails
allocation on a 40 GB device.

### GPU size is not enforced by the request

**`SBATCH_GRES=gpu:1` does not guarantee a device with the ≥40 GB the 8B model
needs.** On a heterogeneous GPU pool the job can be allocated a smaller card and
fail at engine startup, after queueing. Use a typed request for the cluster in
use:

```bash
MODEL_TAG=Llama-3.1-8B-Instruct SBATCH_GRES=gpu:a100:1 ./run_pipeline.sh
```

Check what your cluster and partition accept before the full sweep; the pilot is
the place to confirm it.

### `SBATCH_TIME` for 8B is provisional

`06:00:00` is an estimate, not a measurement. It comes from scaling the v5
TinyLlama timings by ~7×, and those timings are unreliable: they predate the
chat-template fix, so `stable_first` generated to the 64-token cap while
`original` emitted EOS immediately, inflating the spread. Replace it with the
pilot's measured per-case cost before the full run.

### Historical note

`max_model_len` was formerly never passed to vLLM. `inference_benchmark/vllm_backend.py`
constructed `AsyncEngineArgs` with only `model`, `enable_prefix_caching`,
`gpu_memory_utilization`, and `trust_remote_code`; the synchronous path was the
same. It is now plumbed through `benchmark_prefix_cache.py` →
`benchmark_config.py` → both engine constructors, and `run_pipeline.sh` supplies
the registry value. Omitting the flag still reproduces the old behavior.

### Sizing `max_model_len`

Required window is `MAX_PROMPT_TOKENS + chat_template_overhead + max_new_tokens`
= `1800 + 15 + 64 = 1879`.

Measured against the v5 runs (all `outputs/benchmark_results/*.jsonl`), the
largest untemplated prompt produced was **1615 tokens**, p99 1295, mean 584. The
longest templated prompt observed across the layout files was 1554 tokens, so
1618 with generation. 2048 covers the workload with 169 tokens of headroom
against the budget bound.

Set `MAX_MODEL_LEN=2048` for **both** models. For TinyLlama this equals its
native window; for Llama-3.1 it is a deliberate cap as described above.

If `MAX_PROMPT_TOKENS` is raised later, raise `MAX_MODEL_LEN` to cover the
template overhead as well, rounded up to a multiple of the 16-token block size.
Setting the budget to `MAX_MODEL_LEN - max_new_tokens` overflows.

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

### Wall-clock budget — measure it, do not estimate it

**There is deliberately no timing table here.** An earlier revision carried one
derived from the v5 TinyLlama runs. It was withdrawn because it was wrong in two
independent ways, and a wrong table that produces an actionable case-count
recommendation is worse than no table:

1. **The source measurements were an artifact.** The v5 runs predate the
   chat-template fix. `original` emitted EOS on the first token in ~97.5% of
   followups while `stable_first` reached the 64-token cap in ~99% of them, so the
   16 ms vs 190 ms per-layout spread reflects that bug, not the workload. With the
   template applied both layouts generate, so the spread narrows and the cheap
   case gets more expensive.
2. **The request count was understated by 2×.** `submit_interface_benchmark.sh`
   loops over both cache modes, and each invocation sweeps both layouts, so one
   job issues `2 cache modes × 2 layouts × 2 requests = 8` requests per case. The
   withdrawn table counted 4.

What to do instead — this is Batch 8 of `TASKS/model-scale-8b.md`:

```bash
MODEL_TAG=Llama-3.1-8B-Instruct MAX_CASES=50 MUTATION_TYPES=chunk_reorder \
  ./run_pipeline.sh
```

From the resulting JSONLs, record mean per-request wall-clock and TTFT per layout
and cache mode. Then:

```
seconds per case = 2 cache modes × 2 layouts × 2 requests × mean request time
required time    = cases × seconds per case + engine startup
```

Allow ~10 min per job for engine startup; loading 16 GB of weights is
substantially slower than TinyLlama's 2.2 GB. Pick `SBATCH_TIME` from that, then
record the measured numbers here.

`SBATCH_TIME=06:00:00` in the registry is a **placeholder chosen to be safely
large**, not a derived value. It exists so a first run is not killed by the
`02:30:00` default; it is not evidence that 3000 cases fit.

On case count: 3000 per mutation type gives 1000 samples per relation cell, which
is what the 1.1B runs used. Halving it widens confidence intervals by ~1.41×
(they scale as 1/sqrt(N)). Decide from the measured per-case cost, and state the
count actually used in `FINDINGS.md`.

### Downloading a gated model

`prep_login.sh` handles this; the manual form is only for recovery.

```bash
# 1. Accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# 2. Then, on a login node:
export HF_TOKEN=<your token>
MODEL_TAG=Llama-3.1-8B-Instruct bash prep_login.sh
```

Staging fails before attempting the download if `HF_TOKEN` (or
`HUGGING_FACE_HUB_TOKEN`) is unset. `Qwen2.5-7B-Instruct` is ungated if the
license step is undesirable — it would need a registry arm added.

Manual equivalent:

```bash
export HF_HOME="$PROJECT_ROOT/hf_cache"
export HF_HUB_DISABLE_XET=1
hf download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir "$PROJECT_ROOT/models/Llama-3.1-8B-Instruct"
```

### Models are analyzed separately, not compared

Each model filters its own example set, because the budget is applied with that
model's tokenizer and vocabulary size changes how the same text tokenizes. The
two runs therefore use **different prompt subsets** and are not directly
comparable case-for-case. This is intended: results are namespaced per model and
analyzed independently. Nothing in `analysis/` carries a model dimension.

## Known operational hazards

- `MODEL_TAG` must be set on every stage. It is pinned into the child-job
  environment by `run_pipeline.sh`, but a bare `sbatch submit_*.sh` re-resolves
  the default tag and writes into the wrong namespace.
- `SBATCH_GRES=gpu:1` does not guarantee a ≥40 GB device for the 8B model. See
  [GPU size is not enforced by the request](#gpu-size-is-not-enforced-by-the-request).
- `HF_HUB_OFFLINE=1` on compute nodes: any asset not pre-cached on the login
  node fails at job start.
- vLLM version is pinned to the cluster wheelhouse, not PyPI. `AsyncEngineArgs`
  fields drift across versions; `vllm_backend.py` filters kwargs against the
  installed dataclass rather than assuming a schema.
- `$SCRATCH` is subject to purge policy. Sync results to `outputs/` or off-cluster
  before they expire.
- The benchmark resets the prefix cache between every `(base, followup)` pair so
  the `unrelated_control` baseline is not contaminated by earlier cases.
- The processed example set carries no provenance record; only the
  `prepare_data.sh` stdout identifies the tokenizer and budget used.
- Analysis figure defaults describe TinyLlama on an A100. Pass `--n-params`,
  `--model-label`, `--gpu-peak-tflops`, and `--gpu-label` to `plot_report.py`
  for any other configuration.
