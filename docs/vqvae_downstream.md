# Paired downstream reconstruction evaluation

`workspace.vqvae.evaluate_downstream` compares the highest-numbered original
LoRA file for each commonsense task with its reconstruction from a specified
VQ-VAE model. It records checkpoint and dataset hashes, VQ training membership,
token reconstruction error, weight relative L2 error, and weight cosine similarity.

Run from the repository root in Conda `dnd` on physical GPU 4:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m workspace.vqvae.evaluate_downstream prepare \
  --output outputs/vqvae_arc_c/downstream_last_latest \
  --model outputs/vqvae_arc_c/last.pt \
  --base-model Drag-and-Drop-LLMs/models/Qwen2.5-0.5B-Instruct \
  --data-root Loradatasets/common_sense_reasoning
python -m workspace.vqvae.evaluate_downstream evaluate \
  --output outputs/vqvae_arc_c/downstream_last_latest
```

Adjust the base model and data directories for other checkouts. Preparation
requires a new output directory. Evaluation reuses completed prediction files
from that directory when resuming; use a separate output for another protocol
or sample limit. `--tasks` restricts preparation to selected tasks.

The comparison uses the repository's prepared test JSON files, except HellaSwag
and WinoGrande, which use its labeled validation files, and PIQA, which uses
the official labeled validation set (1,838 examples). The repository PIQA test
file incorrectly assigns A to all 3,084 examples; its accuracy is not used.
Preparation downloads the official train/dev archive and records its source.
Both adapter variants
use the same Qwen chat template, supplied system prompts, rank 8/alpha 16 adapter
configuration, and greedy generation with up to 1,024 new tokens (configurable
with `--max-new-tokens`). Answers must
contain an explicit bracketed answer or start with an answer label. Numeric ARC
choices 1–5 are mapped to A–E for comparison. Unparseable
answers count as incorrect. These are controlled paired measurements, not a
claim of reproducing paper scores; the older inference script uses stochastic
decoding and a different, permissive answer parser.

`results.json` contains task accuracy, percentage-point change, answer agreement,
invalid/truncated response counts, and counts of correct-to-wrong and
wrong-to-correct answers. Each task also has original and reconstructed adapters
and per-example prediction JSONL files for inspection. `comparison.json` records
the exact source selection and reconstruction metadata.

The final VQ model (`last.pt`) differs from `best_train_reconstruction.pt`.
Automatic post-training `train_codes.pt` uses the latter. This evaluation
re-encodes the selected adapters with the explicitly requested model.
