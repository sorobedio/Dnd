# Improved VQ-VAE downstream comparison

Best training-reconstruction checkpoint: step 3000. GPU 4, conda dnd. Fresh paired evaluation with identical greedy decoding, maximum 1024 new tokens, and explicit answer scoring. All five latest LoRA adapters were included in VQ-VAE training; benchmark questions were used only for downstream evaluation.

| Task | LoRA step | Examples | Original accuracy | Reconstructed accuracy | Change (pp) | Answer agreement |
|---|---:|---:|---:|---:|---:|---:|
| ARC-e | 220 | 2376 | 76.98% | 76.56% | -0.42 | 98.23% |
| BoolQ | 300 | 3270 | 62.63% | 63.09% | +0.46 | 98.07% |
| PIQA | 215 | 1838 | 48.75% | 49.08% | +0.33 | 90.32% |
| HellaSwag | 300 | 10042 | 54.11% | 54.02% | -0.09 | 91.18% |
| ARC-c | 250 | 1172 | 55.89% | 56.23% | +0.34 | 97.44% |

All 37,396 predictions checked for counts, indices, correctness, accuracy, agreement, and nontruncated valid answers.

PIQA uses the official labeled validation set. Original baselines were rerun and differ slightly from the previous comparison; source adapter and dataset hashes match that comparison. OBQA and WinoGrande were not evaluated in this five-task training reconstruction run.

The residual model stores five shared reference tensors (106.52 MiB) in addition to the VQ-VAE parameters and per-adapter codes.
