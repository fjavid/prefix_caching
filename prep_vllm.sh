#!/bin/bash
set -euo pipefail

module --force purge
module load python/3.12 gcc opencv/4.11

PROJECT_ROOT="$HOME/work/prefix_caching"
mkdir -p "$PROJECT_ROOT"

VLLM_ENV="$PROJECT_ROOT/env_vllm"

virtualenv --no-download "$VLLM_ENV"
source "$VLLM_ENV/bin/activate"

pip install --no-index --upgrade pip

echo "Available vLLM wheels:"
avail_wheels "vllm"

# Replace X.Y.Z with the exact version shown by avail_wheels
pip install --no-index "vllm==X.Y.Z"

pip freeze > "$PROJECT_ROOT/vllm-requirements.txt"

deactivate
echo "vLLM prep complete."