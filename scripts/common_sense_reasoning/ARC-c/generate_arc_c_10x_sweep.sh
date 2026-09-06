#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
CONDA_EXE="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
MODEL=outputs/code_generator_holdout_arc_c/best_train.pt
PROMPTS=prepare/data/ARC-c_train.json
ROOT=outputs/code_generator_holdout_arc_c/sweep_arc_c
mkdir -p "$ROOT"
run_one() {
  local family="$1" index="$2" temperature="$3" topk="$4"
  "$CONDA_EXE" run --no-capture-output -n dnd python -m workspace.code_generator.run generate \
    --model "$MODEL" --task ARC-c --checkpoint-step 250 --prompts "$PROMPTS" \
    --seed "$((999 + index))" --temperature "$temperature" --top-k "$topk" \
    --output "$ROOT/${family}_${index}_codes.pt" --decode-dir "$ROOT/${family}_${index}_adapter"
}
for i in $(seq 0 9); do run_one greedy "$i" 0 0; done
for i in $(seq 0 9); do run_one topk "$i" 0 "$((8 * (i + 1)))"; done
for i in $(seq 0 9); do
  temp=$(awk "BEGIN { printf \"%.2f\", 0.10 * ($i + 1) }")
  run_one temperature "$i" "$temp" 0
done
echo "Generated 30 ARC-c adapters under $ROOT"
