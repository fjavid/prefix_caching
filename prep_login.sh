#!/bin/bash
set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3 arrow/24.0.0 opencv/4.13 python/3.12
module load scipy-stack/2025a

PROJECT_ROOT="$HOME/work/prefix_caching"
SCRATCH_ROOT="$SCRATCH/prefix_caching"

mkdir -p "$PROJECT_ROOT"/{wheelhouse,hf_cache}
mkdir -p "$SCRATCH_ROOT"/{benchmark_results,analysis}

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

if [ ! -d "$PROJECT_ROOT/models/TinyLlama-1.1B-Chat-v1.0" ]; then
    export HF_HUB_DISABLE_XET=1
    hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --local-dir "$PROJECT_ROOT/models/TinyLlama-1.1B-Chat-v1.0"
fi

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

# Step 0: pre-process RAG data into $SCRATCH/prefix_caching/processed/rag_examples.jsonl
# This is required because compute nodes have no internet. Safe to re-run.
echo ""
echo "Running data preparation step..."
bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/prepare_data.sh"
echo "Login-node setup complete. You can now submit SLURM jobs (e.g. ./run_pipeline.sh)."
