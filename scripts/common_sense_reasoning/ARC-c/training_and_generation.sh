#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5
export NUM_PROCESSES=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export DND_MICRO_BATCH_SIZE="${DND_MICRO_BATCH_SIZE:-64}"
python workspace/main/tasks/common_sense_reasoning/train_qwen0.5lora_ARC-c.py
for checkpoint in ARC-c1000 ARC-c2000 ARC-c3000 ARC-c; do
    python workspace/main/generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset "$checkpoint" --test_dataset ARC-c
done
