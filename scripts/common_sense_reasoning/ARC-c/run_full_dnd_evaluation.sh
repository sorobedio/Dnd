#!/usr/bin/env bash
set -euo pipefail

cd /data-vol1/soro/Projects/Dnd/Drag-and-Drop-LLMs
export CUDA_VISIBLE_DEVICES=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DND_EXTRACTOR=/data-vol1/soro/Projects/Dnd/Drag-and-Drop-LLMs/models/all-MiniLM-L12-v2
export DND_DATASET_ROOT=/data-vol1/soro/Projects/Dnd/Loradatasets/common_sense_reasoning
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# Existing pretrained DnD checkpoint; ARC-c is the held-out target.
for task in ARC-e BoolQ PIQA HellaSwag ARC-c OBQA WinoGrande; do
    echo "===== DnD ARC-c checkpoint on ${task} ====="
    python workspace/main/generate/qwen0.5lora_generation_common_sense_reasoning.py \
        --eval_dataset ARC-c \
        --test_dataset "$task"
done

echo "===== Full DnD common-sense evaluation complete ====="
