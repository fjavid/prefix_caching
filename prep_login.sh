#!/bin/bash
# Login-node setup: builds .venv from the cluster wheelhouse, stages model
# weights, pre-caches datasets and validation models, then runs data prep.
# Must run on a LOGIN NODE — compute nodes have no network access.
#
# The model staged is selected by MODEL_TAG, resolved through the registry in
# pipeline_config.sh:
#   bash prep_login.sh                                      # default model
#   MODEL_TAG=Llama-3.1-8B-Instruct bash prep_login.sh      # a registered model
#
# Staging is idempotent: an already-complete weights directory is left alone.
# Gated repos require HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) plus prior license
# acceptance on the model's Hugging Face page.
set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3 arrow/24.0.0 opencv/4.13 python/3.12
module load scipy-stack/2025a

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Single source of truth for PROJECT_ROOT, SCRATCH_ROOT, the model registry
# (MODEL_TAG -> MODEL_REPO / MODEL_PATH / GATED), and the MODEL_TAG-namespaced
# artifact directories. Sourced rather than re-derived so this script and the
# SLURM stages cannot disagree about where anything lives.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/pipeline_config.sh"

mkdir -p "$PROJECT_ROOT"/{wheelhouse,hf_cache}
# Namespaced per model, matching what the SLURM stages read and write.
mkdir -p "$PROCESSED_DIR" "$MUTATION_ROOT" "$LAYOUT_DIR" "$BENCH_DIR" "$ANALYSIS_DIR"

virtualenv --no-download "$PROJECT_ROOT/.venv"
source "$PROJECT_ROOT/.venv/bin/activate"

pip install --no-index --upgrade pip

# requirements.txt should NOT contain vllm
pip install --no-index -r requirements.txt

echo "Available vLLM wheels:"
avail_wheels "vllm"
pip install --no-index "vllm==0.20.0"

# Optional: only if bert-score is not already available from the wheelhouse
BERT_WHEEL="$PROJECT_ROOT/wheelhouse/bert_score-0.3.13-py3-none-any.whl"
if [ ! -f "$BERT_WHEEL" ]; then
    python -m pip download --no-deps -d "$PROJECT_ROOT/wheelhouse" "bert-score==0.3.13"
fi
pip install --no-index "$BERT_WHEEL"

export HF_HOME="$PROJECT_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/hf_cache/datasets"

mkdir -p "$PROJECT_ROOT/models"

# A staged model directory is usable only if every weight shard is present and
# the tokenizer actually loads offline. Both matter concretely: an interrupted
# `hf download` of a sharded model (Llama-3.1-8B ships several shards) leaves
# the completed shards behind and the rest absent, and the chat template applied
# at benchmark time needs a loadable tokenizer, not just tokenizer_config.json.
# Anything missed here surfaces on a compute node with no network to recover
# from, after the job has already been queued and started.
#
# Counting weight files is not sufficient, so the shard set is verified against
# the index's weight_map when one exists, and a `*-of-*` file with no index is
# treated as unverifiable. Runs after the venv is active, so the interpreter and
# transformers are available.
model_dir_complete() {
    local d="$1"
    [ -f "$d/config.json" ] || return 1
    python - "$d" <<'PY'
import glob
import json
import os
import sys

d = sys.argv[1]


def incomplete(msg: str) -> None:
    print(f"     incomplete: {msg}")
    sys.exit(1)


index_path = None
for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
    candidate = os.path.join(d, name)
    if os.path.isfile(candidate):
        index_path = candidate
        break

if index_path is not None:
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map", {})
    if not weight_map:
        incomplete(f"{os.path.basename(index_path)} lists no weights")
    missing = sorted(
        {v for v in weight_map.values() if not os.path.isfile(os.path.join(d, v))}
    )
    if missing:
        incomplete(
            f"{len(missing)} of {len(set(weight_map.values()))} weight shard(s) "
            f"named in {os.path.basename(index_path)} are absent, "
            f"first: {missing[0]}"
        )
else:
    sharded = (
        glob.glob(os.path.join(d, "*-of-*.safetensors"))
        + glob.glob(os.path.join(d, "*-of-*.bin"))
    )
    if sharded:
        incomplete(
            "sharded weight files are present but no *.index.json exists to "
            "verify the shard set against"
        )
    if not (
        glob.glob(os.path.join(d, "*.safetensors"))
        or glob.glob(os.path.join(d, "*.bin"))
    ):
        incomplete("no *.safetensors or *.bin weight file")

try:
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(d, local_files_only=True)
except Exception as exc:  # noqa: BLE001 - any failure means unusable offline
    incomplete(f"tokenizer does not load offline ({type(exc).__name__}: {exc})")
PY
}

stage_model() {
    echo ""
    echo "Model staging:"
    echo "  MODEL_TAG  = $MODEL_TAG"
    echo "  MODEL_REPO = $MODEL_REPO"
    echo "  MODEL_PATH = $MODEL_PATH"
    echo "  GATED      = $GATED"

    if [ -d "$MODEL_PATH" ]; then
        if model_dir_complete "$MODEL_PATH"; then
            echo "  -> already staged, skipping download."
            return 0
        fi
        echo "ERROR: $MODEL_PATH exists but is incomplete (see reason above)." >&2
        echo "       A complete directory has config.json, every weight shard" >&2
        echo "       named in its *.index.json, and a tokenizer that loads" >&2
        echo "       offline. This is what an interrupted download leaves" >&2
        echo "       behind." >&2
        echo "       Inspect it, then remove and re-run:" >&2
        echo "         rm -rf '$MODEL_PATH'" >&2
        echo "         MODEL_TAG=$MODEL_TAG bash prep_login.sh" >&2
        return 1
    fi

    if [ "$GATED" = "1" ] && [ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]; then
        echo "ERROR: $MODEL_REPO is a gated Hugging Face repo and no token is set." >&2
        echo "       1. Accept the license at https://huggingface.co/$MODEL_REPO" >&2
        echo "       2. export HF_TOKEN=<your token>" >&2
        echo "       3. re-run: MODEL_TAG=$MODEL_TAG bash prep_login.sh" >&2
        return 1
    fi

    echo "  -> downloading..."
    export HF_HUB_DISABLE_XET=1
    hf download "$MODEL_REPO" --local-dir "$MODEL_PATH"

    if ! model_dir_complete "$MODEL_PATH"; then
        echo "ERROR: download finished but $MODEL_PATH is still incomplete." >&2
        return 1
    fi
    echo "  -> staged."
}

stage_model

python - <<'PY'
"""Pre-cache datasets and models used by the offline compute nodes."""
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Datasets used at step 0 (RAG data preparation).
load_dataset("LLukas22/nq-simplified", split="train[:5]")

# Sentence-transformer model used by overlap_analyzer + mutation_validation + severity_calibration.
SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# NLI model used by mutation_validation (only when --validation-backend in {nli, hybrid}).
nli_model = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
AutoTokenizer.from_pretrained(nli_model)
AutoModelForSequenceClassification.from_pretrained(nli_model)
PY

pip freeze > "$PROJECT_ROOT/frozen-requirements.txt"

deactivate
echo "Base prep complete."

# Step 0: pre-process RAG data into this model's processed/ directory. Required
# because compute nodes have no internet. Safe to re-run.
#
# MODEL_TAG is exported so the child process resolves the same registry entry
# and writes into the same namespace; prepare_data.sh would otherwise fall back
# to the default model.
export MODEL_TAG
echo ""
echo "Running data preparation step for MODEL_TAG=$MODEL_TAG..."
bash "$SCRIPT_DIR/prepare_data.sh"
echo ""
echo "Login-node setup complete for MODEL_TAG=$MODEL_TAG."
echo "Submit SLURM jobs with:  MODEL_TAG=$MODEL_TAG ./run_pipeline.sh"
