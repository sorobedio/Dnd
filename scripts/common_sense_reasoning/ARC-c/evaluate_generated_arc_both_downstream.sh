#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
CONDA_EXE="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
BASE="$REPO_ROOT/Drag-and-Drop-LLMs/models/Qwen2.5-0.5B-Instruct"
GEN=outputs/code_generator_holdout_arc_c/best_train.pt
VQ=outputs/vqvae_train_arc_c_residual/train_codes.pt
PROMPT_ROOT=prepare/data

for task in ARC-e ARC-c; do
  if [[ ! -f "outputs/code_generator_holdout_arc_c/generated_adapters/$task/$( [[ $task == ARC-e ]] && echo 220 || echo 250 )/adapter_model.safetensors" ]]; then
    step=$( [[ "$task" == ARC-e ]] && echo 220 || echo 250 )
    "$CONDA_EXE" run --no-capture-output -n dnd python -m workspace.code_generator.run generate \
      --model "$GEN" --task "$task" --checkpoint-step "$step" \
      --prompts "$PROMPT_ROOT/${task}_train.json" \
      --output "outputs/code_generator_holdout_arc_c/${task}_generated_codes.pt" \
      --decode-dir "outputs/code_generator_holdout_arc_c/generated_adapters"
  fi
  step=$( [[ "$task" == "ARC-e" ]] && echo 220 || echo 250 )
  "$CONDA_EXE" run --no-capture-output -n dnd python -m workspace.code_generator.evaluate_generated_downstream \
    --base-model "$BASE" --original "Loradatasets/common_sense_reasoning/$task/$step.safetensors" \
    --generated "outputs/code_generator_holdout_arc_c/generated_adapters/$task/$step/adapter_model.safetensors" \
    --data "$PROMPT_ROOT/${task}_test.json" --task "$task" \
    --output "outputs/code_generator_holdout_arc_c/${task}_downstream_comparison.json"
done
