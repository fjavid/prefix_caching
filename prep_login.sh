#!/bin/bash
set -euo pipefail

module load python/3.11
module load gcc opencv/4.9.0

PROJECT_ROOT="$HOME/work/prefix_caching"
SCRATCH_ROOT="$SCRATCH/prefix_caching"

mkdir -p "$PROJECT_ROOT"/{env,wheelhouse,hf_cache,raw_data}
mkdir -p "$SCRATCH_ROOT"/{processed,mutation,prompt_organization,benchmark_results,analysis}

virtualenv --no-download "$PROJECT_ROOT/env"
source "$PROJECT_ROOT/env/bin/activate"

python -m pip install --upgrade pip
pip install --no-index -r requirements.txt

BERT_WHEEL="$PROJECT_ROOT/wheelhouse/bert_score-0.3.13-py3-none-any.whl"
if [ ! -f "$BERT_WHEEL" ]; then
    python -m pip download --no-deps -d "$PROJECT_ROOT/wheelhouse" "bert-score==0.3.13"
fi
pip install --no-index "$BERT_WHEEL"

export HF_HOME="$PROJECT_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/hf_cache/datasets"

python - <<'PY'
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

load_dataset("LLukas22/nq-simplified", split="train[:5]")
SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
PY

deactivate
echo "Prep complete."