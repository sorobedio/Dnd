#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
exec "${CONDA_EXE:-$HOME/miniconda3/bin/conda}" run --no-capture-output -n dnd python -m workspace.code_generator.run evaluate-heldout \
  --model outputs/code_generator_holdout_arc_c/best_train.pt \
  --codes outputs/vqvae_train_arc_c_residual/train_codes.pt --task ARC-c \
  --prompts prepare/data/ARC-c_train.json \
  --output outputs/code_generator_holdout_arc_c/arc_c_heldout_evaluation.pt "$@"
