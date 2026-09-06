#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
CONDA_EXE="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
ENCODER="$REPO_ROOT/models/all-MiniLM-L12-v2"
if [[ ! -d "$ENCODER" ]]; then
    ENCODER="$REPO_ROOT/Drag-and-Drop-LLMs/models/all-MiniLM-L12-v2"
fi
# One launcher at a time; preparation can resume its per-task embedding caches.
mkdir -p outputs/code_generator_holdout_arc_c
exec 9>outputs/code_generator_holdout_arc_c/launcher.lock
flock -n 9 || { echo "Another code-generator launcher is running" >&2; exit 1; }
if [[ ! -f outputs/code_generator_data_holdout_arc_c/dataset.pt ]]; then
    "$CONDA_EXE" run --no-capture-output -n dnd python -m workspace.code_generator.run prepare --encoder "$ENCODER" \
        --exclude-tasks ARC-c --output-dir outputs/code_generator_data_holdout_arc_c
fi
TRAIN_ARGS=(--wandb
    --dataset outputs/code_generator_data_holdout_arc_c/dataset.pt
    --output-dir outputs/code_generator_holdout_arc_c)
if [[ -f outputs/code_generator_holdout_arc_c/last.pt ]]; then
    TRAIN_ARGS+=(--resume outputs/code_generator_holdout_arc_c/last.pt)
fi
exec "$CONDA_EXE" run --no-capture-output -n dnd python -m workspace.code_generator.run train "${TRAIN_ARGS[@]}" "$@"
