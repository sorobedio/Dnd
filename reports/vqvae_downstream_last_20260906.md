# Final VQ-VAE downstream reconstruction evaluation

Evaluated on GPU 4 in Conda `dnd`, using `last.pt` at step 10,000 and the highest-numbered LoRA checkpoint per task. The automatically exported training codes use the best VQ model instead; these adapters were reconstructed again using the final model.

| Task | LoRA checkpoint | Examples | Original accuracy | Reconstructed accuracy | Change (pp) | Answer agreement |
|---|---:|---:|---:|---:|---:|---:|
| ARC-e | 220.safetensors | 2376 | 76.73% | 76.81% | +0.08 | 97.22% |
| BoolQ | 300.safetensors | 3270 | 62.60% | 63.82% | +1.22 | 94.50% |
| PIQA | 215.safetensors | 1838 | 48.53% | 51.52% | +2.99 | 17.68% |
| HellaSwag | 300.safetensors | 10042 | 54.02% | 52.67% | -1.35 | 74.62% |
| ARC-c | 250.safetensors | 1172 | 56.14% | 0.43% | -55.72 | 0.77% |
| OBQA | 300.safetensors | 500 | 70.40% | 2.40% | -68.00 | 2.20% |
| WinoGrande | 250.safetensors | 1267 | 51.07% | 27.62% | -23.44 | 32.83% |

Accuracy is largely retained on ARC-e, BoolQ, and HellaSwag. PIQA has similar near-chance accuracy but changes most answers. ARC-c, OBQA, and WinoGrande reconstructions degrade severely, including failure to follow the answer format.

The VQ training tasks were ARC-e, BoolQ, PIQA, and HellaSwag. Of the latest adapters evaluated, only ARC-e/220 was in the frozen training selection. The latest BoolQ, PIQA, and HellaSwag files were unseen checkpoints of represented tasks. ARC-c was a held-out task; OBQA and WinoGrande were absent from VQ training and evaluation.

| Task | Weight relative L2 error | Weight cosine | Reconstructed invalid answers | Reconstructed responses reaching 1,024-token limit |
|---|---:|---:|---:|---:|
| ARC-e | 5.21% | 0.9992 | 0/2376 | 0/2376 |
| BoolQ | 6.55% | 0.9984 | 0/3270 | 0/3270 |
| PIQA | 8.08% | 0.9971 | 0/1838 | 0/1838 |
| HellaSwag | 11.74% | 0.9950 | 0/10042 | 0/10042 |
| ARC-c | 139.39% | 0.2057 | 1154/1172 | 1168/1172 |
| OBQA | 139.73% | 0.2061 | 454/500 | 472/500 |
| WinoGrande | 148.31% | 0.1424 | 556/1267 | 750/1267 |

Both variants use Qwen2.5-0.5B-Instruct, rank 8 / alpha 16, the same prepared system/user prompts, Qwen chat template, and greedy decoding up to 1,024 new tokens. All original generations finished without truncation and were parseable. Explicit bracketed or leading answer labels are scored; numeric ARC labels 1–5 map to A–E. Unparseable answers count as incorrect. These are controlled paired generation scores, not a reproduction of the paper's stochastic inference and permissive parser.

An initial 32-token run was retained separately. All generations that finished before that cap were reused unchanged; all truncated variant/task sets were regenerated at 1,024 tokens. Extended generation did not resolve the severe ARC-c and OBQA failures. Strict answer-format accuracy should not be interpreted as a format-independent reasoning score.

PIQA uses 1,838 official labeled validation examples: the repository test JSON assigns A to every example, while official test labels are unavailable. Source: [official dataset loader](https://huggingface.co/datasets/ybisk/piqa/blob/main/piqa.py). HellaSwag and WinoGrande use repository validation files; other tasks use repository test files. Total: 20,465 examples per variant.

Server results and per-example predictions: `/data-vol1/soro/Projects/Dnd/outputs/vqvae_arc_c/downstream_last_latest_1024/`. Adapters and source selection are in `../downstream_last_latest/`; the extended directory links to those same adapters. Log: `/data-vol1/soro/Projects/Dnd/logs/vqvae_downstream_eval_1024.log`.

Verification: seven tasks and both variants present; every prediction index and count checked; accuracies recomputed from saved predictions; answer-parser unit tests pass. Full source hashes, membership, reconstruction errors, counts, and scores are in the adjacent JSON report.
