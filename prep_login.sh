#!/bin/bash
set -euo pipefail

module --force purge
module load gcc python/3.12 arrow/24.0.0

PROJECT_ROOT="$HOME/work/prefix_caching"
SCRATCH_ROOT="$SCRATCH/prefix_caching"

mkdir -p "$PROJECT_ROOT"/{env_base,wheelhouse,hf_cache,raw_data}
mkdir -p "$SCRATCH_ROOT"/{processed,mutation,prompt_organization,benchmark_results,analysis}

virtualenv --no-download "$PROJECT_ROOT/env_base"
source "$PROJECT_ROOT/env_base/bin/activate"

pip install --no-index --upgrade pip

# requirements.txt should NOT contain vllm
pip install --no-index -r requirements.txt

# Optional: only if bert-score is not already available from the wheelhouse
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

pip freeze > "$PROJECT_ROOT/base-requirements-frozen.txt"

deactivate
echo "Base prep complete."