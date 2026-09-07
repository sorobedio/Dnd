#!/usr/bin/env bash
# Evaluate the prefix-GPT code generator on every common-sense task: five
# adapters generated per task, decoded through the VQ-VAE and scored downstream.
# The original LoRA checkpoints are not part of this evaluation; set
# DND_ORIGINALS to a positive number to score them alongside.
#
# Scoring is the shared path used by evaluate_pretrained_dnd_all_tasks.sh, so
# the two generators' numbers are directly comparable. Every stage is
# restartable: prepared tasks and saved predictions are reused.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

MODULE=workspace.code_generator.evaluate_tasks
GENERATOR="${DND_GENERATOR_CHECKPOINT:-outputs/code_generator_holdout_arc_c/best_train.pt}"
OUTPUT="${DND_OUTPUT:-outputs/code_generator_common_sense}"
LOG_DIR="${DND_LOG_DIR:-logs}"
# Cheapest first, with the held-out target scored before anything else.
read -r -a TASKS <<< "${DND_TASKS:-ARC-c OBQA WinoGrande ARC-e PIQA BoolQ HellaSwag}"
if [[ -n "${DND_PYTHON:-}" ]]; then
    read -r -a PYTHON <<< "$DND_PYTHON"
else
    PYTHON=("${CONDA_EXE:-$HOME/miniconda3/bin/conda}" run --no-capture-output -n "${DND_CONDA_ENV:-dnd}" python)
fi
mkdir -p "$LOG_DIR" "$OUTPUT"

if [[ ! -f "$GENERATOR" ]]; then
    echo "Missing generator checkpoint: $GENERATOR" >&2
    exit 1
fi

# PIQA is scored on the official dev labels, because the repository PIQA test
# file labels every example A. They are downloaded once into $OUTPUT; point
# DND_PIQA_VALIDATION at an earlier copy when the host has no network access.
if [[ -n "${DND_PIQA_VALIDATION:-}" && ! -f "$OUTPUT/PIQA_validation.json" ]]; then
    cp "$DND_PIQA_VALIDATION" "$OUTPUT/PIQA_validation.json"
fi

"${PYTHON[@]}" -m "$MODULE" prepare \
    --output "$OUTPUT" \
    --generator-checkpoint "$GENERATOR" \
    --samples "${DND_SAMPLES:-5}" \
    --originals "${DND_ORIGINALS:-0}" \
    --prompt-split "${DND_PROMPT_SPLIT:-evaluation}" \
    --temperature "${DND_TEMPERATURE:-0}" \
    --top-k "${DND_TOP_K:-0}" \
    --device "${DND_DEVICE:-cuda:0}" \
    --tasks "${TASKS[@]}" 2>&1 | tee "$LOG_DIR/code_generator_prepare.log"

# vLLM is sized to the memory actually free on the GPU; override only to pin it.
EVALUATE=(--output "$OUTPUT" --tasks "${TASKS[@]}" --max-new-tokens "${DND_MAX_NEW_TOKENS:-1024}")
[[ -n "${DND_VLLM_MEMORY_UTILIZATION:-}" ]] && EVALUATE+=(--gpu-memory-utilization "$DND_VLLM_MEMORY_UTILIZATION")
[[ -n "${DND_LIMIT:-}" ]] && EVALUATE+=(--limit "$DND_LIMIT")
"${PYTHON[@]}" -m "$MODULE" evaluate "${EVALUATE[@]}" 2>&1 | tee "$LOG_DIR/code_generator_evaluate.log"

"${PYTHON[@]}" -m "$MODULE" report --output "$OUTPUT" --markdown "$OUTPUT/report.md"
echo "Adapters, predictions and reports are under $OUTPUT"
