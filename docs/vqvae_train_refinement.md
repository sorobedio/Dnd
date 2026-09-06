# Training reconstruction refinement including ARC-c

Run in the repository root:

```bash
bash scripts/common_sense_reasoning/ARC-c/training_vqvae_residual.sh
```

The launcher uses Conda `dnd`, physical GPU 4, and the existing W&B login. It
starts from `outputs/vqvae_arc_c/last.pt` and writes a new run to
`outputs/vqvae_train_arc_c_residual/`. The previous model and its outputs remain
available for comparison. Resume this workflow with the same script:

```bash
bash scripts/common_sense_reasoning/ARC-c/training_vqvae_residual.sh \
  --resume outputs/vqvae_train_arc_c_residual/last.pt
```

The frozen 200 training adapters are retained, all 50 ARC-c adapters are moved
into training, and each represented task's highest-numbered checkpoint is added
if missing. For this dataset the result is 254 complete adapters: 50 ARC-e and
51 each of BoolQ, PIQA, HellaSwag, and ARC-c. There is no held-out split in this
run. OBQA, WinoGrande, and benchmark question/answer data are not used for training
or checkpoint selection.

The model uses the same whole-adapter HyperConv encoder/decoder and 1,024-vector
codebook, operating on residuals from five shared training references. Each
reference is a mean token tensor from one of the five training tasks, computed
only from these training adapters. Encoding chooses the nearest reference from
the input weights, without a task label, then quantizes the normalized residual.
Decoding restores the reference plus the decoded residual. References and residual
scales are frozen buffers stored in the model checkpoint.

The code contains one reference index plus 2,560 residual indices (2,561 total),
compared with 2,560 previously. The five shared references add about 106.5 MiB to
the model; this is an explicit storage-for-fidelity tradeoff. Decoding needs only
the trained model, indices, and matrix shapes, never the original weight values.

The initial decoder residual output is zero, so the initial model reconstructs
the nearest reference. `initial_reference.json` records this baseline separately
from `baseline.json`, which measures the old model on the expanded training set.
Further changes improve precision and optimization:

- FP32 tokenization preserves the means and standard deviations needed to restore
  original weight values. The original BF16 preprocessing introduced 3.86% relative
  L2 error for A and 23.25% for B on ARC-c/250 before any autoencoder was used.
  FP32 preprocessing reduced those errors to about 0.000062% and 0.000810%.
- FP32 training and export, with TF32 disabled, avoid reduced precision in predicted
  scale metadata. The manifest records the precision; token caches use separate keys.
- The loss is importance-weighted token MSE plus 0.01 times restored-weight relative
  MSE, divided by the mean reference residual variance for stable gradient scale,
  plus commitment loss during joint training. Restored-weight error differentiates
  through the same mean/std inverse transform as decoding, normalizing each chunk by
  its target standard deviation. Thus small B weights contribute meaningfully.
- Each epoch shuffles all training adapters without replacement, including the short
  final batch. AdamW uses no weight decay, gradient clipping, LR warmup, and cosine
  decay with a nonzero floor.
- Default training runs for 3,000 steps: 2,000 joint encoder/codebook/decoder steps,
  then 1,000 decoder refinement steps with encoder and codebook frozen. The LR schedule
  warms up again when the refinement phase begins. Default batch size is 5 and peak
  LR is 1e-4. These are experiment settings, not a claim of optimal hyperparameters.

Every 250 steps, at phase boundaries, on completion, and on a graceful stop,
the code evaluates the actual encode/decode path across all 254 training adapters.
`best_train_reconstruction.pt` contains the full model and optimizer state selected
by **mean weighted reconstruction MSE over the full training set**. It no longer
selects the lowest random minibatch loss. The report also includes per-task token
and restored-weight errors. `baseline.json` measures the old model on the exact
same training adapters and new FP32 representation before parameters change.

`last.pt` saves progress every 100 steps and on SIGINT/SIGTERM. Both checkpoint
types include model, optimizer, shuffle order and position, RNG states, data
manifest, best score, and W&B run identity. Use this refinement launcher to resume
these checkpoints; the legacy trainer has a different scheduler format.

After completion, the launcher exports all training codes using the best model.
The existing `workspace.vqvae.run encode/decode` commands recognize FP32 manifests
and preserve full precision. Do not reuse old BF16 token caches or code files with
the new model.

The earlier direct refinement experiment is retained in
`outputs/vqvae_train_arc_c_refined/`. It reduced ARC-c error while increasing
ARC-e error, motivating residual encoding. Its launcher is
`training_vqvae_refined.sh`; the residual launcher passes the tested reference
and optimization settings to that common implementation.
