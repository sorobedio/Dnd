#!/usr/bin/env bash
# Evaluate the DnD generator pretrained by training_and_generation.sh on every
# common-sense task, including the held-out ARC-c: five adapters generated per
# task against the last five original LoRA checkpoints of the same task.
#
# Runs from this repository root, not the nested Drag-and-Drop-LLMs checkout,
# which has no prepare/data prompts. Every stage is restartable: prepared tasks
# and saved predictions are reused, so an interrupted run continues where it stopped.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

MODULE=workspace.dnd_downstream.evaluate
OUTPUT="${DND_OUTPUT:-outputs/dnd_pretrained_common_sense}"
LOG_DIR="${DND_LOG_DIR:-logs}"
# Cheapest first, with the held-out target scored before anything else.
read -r -a TASKS <<< "${DND_TASKS:-ARC-c OBQA WinoGrande ARC-e PIQA BoolQ HellaSwag}"
if [[ -n "${DND_PYTHON:-}" ]]; then
    read -r -a PYTHON <<< "$DND_PYTHON"
else
    PYTHON=("${CONDA_EXE:-$HOME/miniconda3/bin/conda}" run --no-capture-output -n "${DND_CONDA_ENV:-dnd}" python)
fi
mkdir -p "$LOG_DIR" "$OUTPUT"

# PIQA is scored on the official dev labels, because the repository PIQA test
# file labels every example A. They are downloaded once into $OUTPUT; point
# DND_PIQA_VALIDATION at an earlier copy when the host has no network access.
if [[ -n "${DND_PIQA_VALIDATION:-}" && ! -f "$OUTPUT/PIQA_validation.json" ]]; then
    cp "$DND_PIQA_VALIDATION" "$OUTPUT/PIQA_validation.json"
fi

"${PYTHON[@]}" -m "$MODULE" prepare \
    --output "$OUTPUT" \
    --samples "${DND_SAMPLES:-5}" \
    --originals "${DND_ORIGINALS:-5}" \
    --device "${DND_DEVICE:-cuda:0}" \
    --tasks "${TASKS[@]}" 2>&1 | tee "$LOG_DIR/dnd_pretrained_prepare.log"

EVALUATE=(--output "$OUTPUT" --tasks "${TASKS[@]}"
          --max-new-tokens "${DND_MAX_NEW_TOKENS:-1024}"
          --gpu-memory-utilization "${DND_VLLM_MEMORY_UTILIZATION:-0.25}")
[[ -n "${DND_LIMIT:-}" ]] && EVALUATE+=(--limit "$DND_LIMIT")
"${PYTHON[@]}" -m "$MODULE" evaluate "${EVALUATE[@]}" 2>&1 | tee "$LOG_DIR/dnd_pretrained_evaluate.log"

"${PYTHON[@]}" -m "$MODULE" report --output "$OUTPUT" --markdown "$OUTPUT/report.md"
echo "Adapters, predictions and reports are under $OUTPUT"
