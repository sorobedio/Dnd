# Training-only VQ-VAE reconstruction improvement

Completed 3,000 steps on physical GPU 4 in Conda `dnd`. Training uses 254 complete adapters: 50 ARC-e and 51 each of BoolQ, PIQA, HellaSwag, and ARC-c. The previous 50 ARC-c held-out adapters and each task's latest missing checkpoint were added to training. There is no held-out split in this run. No downstream question/answer benchmark was used for optimization or checkpoint selection.

Full-training weighted MSE fell from **0.13542335 to 0.00158530**, a **98.83% reduction**. The baseline is the old model evaluated on the expanded training set with the same FP32 targets. ARC-c was not a training task for that old model. These are reconstruction measurements, not downstream accuracy scores.

| Task | Training adapters | Old model weighted MSE | New model weighted MSE | Reduction |
|---|---:|---:|---:|---:|
| ARC-e | 50 | 0.005509 | 0.001442 | 73.82% |
| BoolQ | 51 | 0.007960 | 0.001638 | 79.42% |
| PIQA | 51 | 0.009877 | 0.002259 | 77.12% |
| HellaSwag | 51 | 0.010006 | 0.001824 | 81.77% |
| ARC-c | 51 | 0.641217 | 0.000760 | 99.88% |

The residual decoder improves beyond shared references alone: their initial full-training weighted MSE was **0.00715479**, versus **0.00158530** after training (77.84% lower). Best-model selection evaluates the actual encode/decode path on all 254 training adapters every 250 steps; the winning checkpoint is step **3,000**.

The audit below compares actual restored matrices against the original source weights, using the highest-numbered checkpoint for each task. All five audited adapters are confirmed training members. BA error is `||B_hat A_hat - BA||_F / ||BA||_F`, aggregated across projection matrices; smaller is better. The common LoRA scale cancels in the relative error. The audit reproduces the old model's BF16 preprocessing/export and the new model's FP32 preprocessing/export.

| Task | Checkpoint | Old A error | New A error | Old B error | New B error | Old BA error | New BA error |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARC-e | 220.safetensors | 4.79% | 0.36% | 28.30% | 4.22% | 30.54% | 4.24% |
| BoolQ | 300.safetensors | 5.37% | 0.66% | 42.80% | 6.92% | 44.65% | 6.98% |
| PIQA | 215.safetensors | 6.42% | 1.12% | 61.38% | 10.20% | 63.19% | 10.26% |
| HellaSwag | 300.safetensors | 7.92% | 0.97% | 76.65% | 7.58% | 82.23% | 7.78% |
| ARC-c | 250.safetensors | 91.20% | 0.23% | 1427.58% | 2.55% | 881.08% | 2.57% |

The changes are FP32 tokenization/training/export, a restored-weight loss term that includes mean/std recovery, residual VQ encoding around five shared training references, shuffled complete epochs, a warmup/decay schedule, and final decoder refinement with encoder/codebook frozen. BF16 tokenization alone introduced 3.86% A and 23.25% B relative error on ARC-c/250; FP32 reduces this preprocessing round-trip error below 0.001%.

Each adapter uses **2,561 indices**: one reference index and 2,560 residual codes, compared with 2,560 previously. The five frozen shared references add **106.52 MiB** to the model. This extra shared storage is an explicit tradeoff, and means the model is not an equal-storage comparison to the original VQ-VAE. Reference selection uses the input weights, not an inference-time task label. Decode requires only the trained model, codes, and matrix shapes.

The best model and exported training codes are on the server:

- `/data-vol1/soro/Projects/Dnd/outputs/vqvae_train_arc_c_residual/best_train_reconstruction.pt`
- `/data-vol1/soro/Projects/Dnd/outputs/vqvae_train_arc_c_residual/train_codes.pt` — shape `[254, 2561]`
- The same folder contains `training_comparison.json`, `weight_audit.json`, `baseline.json`, `initial_reference.json`, `metrics.jsonl`, and the frozen data manifest.

Launcher: `bash scripts/common_sense_reasoning/ARC-c/training_vqvae_residual.sh`. Workflow and resume details are in `docs/vqvae_train_refinement.md`. The earlier direct-refinement experiment and original model are retained separately.

Verification: 13 tests pass, including reference-code round-trip without original weights/task labels, scale-aware loss gradients, ARC-c inclusion, and BA-error agreement with explicit matrix multiplication. A real saved-code decode restored all 336 finite LoRA matrices with the original shapes. All 254 final code sequences have valid index ranges and match the best model's SHA256. All GPU jobs launched for this refinement have finished. Downstream test accuracy has not been measured for this revised model.
