#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export DND_EXTRACTOR="$REPO_ROOT/Drag-and-Drop-LLMs/models/all-MiniLM-L12-v2"

# The pretrained DnD checkpoint was trained on ARC-e, BoolQ, PIQA, and HellaSwag.
# ARC-c is therefore the explicitly held-out target; the other tasks are useful
# in-domain/cross-task reference measurements.
EVAL_DATASET="${DND_EVAL_DATASET:-ARC-c}"
TASKS=(ARC-e BoolQ PIQA HellaSwag ARC-c OBQA WinoGrande)
LOG_DIR="${DND_LOG_DIR:-logs/dnd_common_sense_all}"
mkdir -p "$LOG_DIR"

for task in "${TASKS[@]}"; do
  log="$LOG_DIR/${EVAL_DATASET}_on_${task}.log"
  heldout=false; [[ "$task" == ARC-c ]] && heldout=true
  echo "Evaluating pretrained DnD ${EVAL_DATASET} on ${task}; ARC-c held-out=${heldout}" | tee "$log"
  python workspace/main/generate/qwen0.5lora_generation_common_sense_reasoning.py \
    "checkpoints/qwen0.5lora__${EVAL_DATASET}.pth" \
    --eval_dataset "$EVAL_DATASET" --test_dataset "$task" 2>&1 | tee -a "$log"
done

echo
echo "Completed pretrained DnD evaluation for: ${TASKS[*]}"
echo "Each target uses 10 generated adapters with 128 randomly sampled task prompts."
echo "Results are under results/common_sense_reasoning/ and logs are under $LOG_DIR."
