#!/usr/bin/env bash
set -euo pipefail
ROOT=outputs/code_generator_holdout_arc_c/sweep_arc_c
BASE=Drag-and-Drop-LLMs/models/Qwen2.5-0.5B-Instruct
ORIGINAL=Loradatasets/common_sense_reasoning/ARC-c/250.safetensors
DATA=prepare/data/ARC-c_test.json
CONDA_EXE="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
export CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false
REPORT="$ROOT/downstream_results.jsonl"; : > "$REPORT"
for family in greedy topk temperature; do
  for i in $(seq 0 9); do
    if [[ "$family" == greedy ]]; then setting="temperature=0,top_k=0"; else if [[ "$family" == topk ]]; then setting="temperature=0,top_k=$((8*(i+1)))"; else setting="temperature=$(awk "BEGIN {printf \"%.2f\",.1*($i+1)}")"; fi; fi
    adapter="$ROOT/${family}_${i}_adapter/ARC-c/250/adapter_model.safetensors"
    out="$ROOT/${family}_${i}_downstream.json"
    "$CONDA_EXE" run --no-capture-output -n dnd python -m workspace.code_generator.evaluate_generated_downstream --base-model "$BASE" --original "$ORIGINAL" --generated "$adapter" --data "$DATA" --task ARC-c --output "$out" >/dev/null
    python3 -c 'import json,sys; x=json.load(open(sys.argv[1]))["report"]; x.update(family=sys.argv[2],index=int(sys.argv[3]),setting=sys.argv[4]); print(json.dumps(x))' "$out" "$family" "$i" "$setting" >> "$REPORT"
  done
done
python3 - "$REPORT" <<'PY'
import json,sys
r=[json.loads(x) for x in open(sys.argv[1])]
print('Family       #  Setting                    Original   Generated  Change(pp)  Agreement')
print('-'*88)
for x in r: print(f"{x['family']:<12} {x['index']:>1}  {x['setting']:<25} {100*x['original_accuracy']:>8.2f}%  {100*x['generated_accuracy']:>9.2f}%  {x['delta_percentage_points']:>+9.2f}  {100*x['answer_agreement']:>8.2f}%")
for f in ('greedy','topk','temperature'):
 y=[x for x in r if x['family']==f]; print(f"{f+' mean':<12}    {'':25} {100*sum(x['generated_accuracy'] for x in y)/10:>9.2f}%  {sum(x['delta_percentage_points'] for x in y)/10:>+9.2f}")
PY
