#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
BASE="$REPO_ROOT/Drag-and-Drop-LLMs/models/Qwen2.5-0.5B-Instruct"
exec "${CONDA_EXE:-$HOME/miniconda3/bin/conda}" run --no-capture-output -n dnd python -m workspace.code_generator.evaluate_generated_downstream \
  --base-model "$BASE" \
  --original Loradatasets/common_sense_reasoning/ARC-e/220.safetensors \
  --generated outputs/code_generator_holdout_arc_c/generated_adapters/ARC-e/220/adapter_model.safetensors \
  --data prepare/data/ARC-e_test.json --task ARC-e \
  --output outputs/code_generator_holdout_arc_c/arc_e_downstream_comparison.json "$@"
