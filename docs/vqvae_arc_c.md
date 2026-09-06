# VQ-VAE for the complete ARC-c-experiment LoRA checkpoints

One sample is one complete Qwen2.5-0.5B rank-8 adapter: all 336 A/B
matrices, packed by the existing DnD tokenizer into `[4296, 10, 130]`.
Matrices and BA deltas are not separated into samples. Sentence-BERT is
not needed for this weight autoencoder.

The manifest reads the training dataset names and `real_length` from
`workspace/main/tasks/common_sense_reasoning/train_qwen0.5lora_ARC-c.py`:
50 each from ARC-e, BoolQ, PIQA, and HellaSwag (200 training samples).
The 50 ARC-c adapters are held out for reconstruction evaluation only.
Selection matches DnD's `os.listdir(folder)[:50]` at manifest creation.
DnD did not record a historical filename manifest; the original run's exact
selection cannot be independently recovered if that directory order changed.
The new manifest records filenames and SHA-256 hashes so subsequent VQ-VAE
runs and resumes use the same files. No held-out weights enter gradient updates
or codebook EMA updates; no checkpoint is selected using held-out scores.

## Architecture

```text
[B, 4296, 10, 130]
    → HyperConv → [B, 1024, 10, 128]
    → HyperConv → [B,  256, 10, 128]
    → EMA vector quantizer (1024 entries × 128 dimensions)
    → codes [B, 256, 10]
    → codebook lookup [B, 256, 10, 128]
    → HyperConv → [B, 1024, 10, 128]
    → HyperConv → [B, 4296, 10, 130]
```

The encoder and decoder reuse DnD's hyper-convolution blocks, with kernel
size 3. Training minimizes the same importance-weighted token MSE plus a
0.25-weighted commitment loss. The codebook uses EMA updates rather than
an optimizer/codebook-loss term. Initialization samples encoder vectors
from the first training batch. Monitor code usage and perplexity for collapse.
These architecture and optimization defaults are a starting experiment,
not a configuration reported by the DnD paper.

There are 2,560 codes per adapter. Files store int32 indices (10 KiB of
index payload per adapter), with additional metadata. A 1,024-entry codebook
could theoretically use 10 bits per index (3.125 KiB), but bit packing is
not implemented. The shared encoder, decoder, and codebook are additional
storage. The reconstructed factors retain rank 8; reconstruction accuracy
and downstream ARC-c accuracy must be measured separately.

## Run on the server

```bash
conda activate dnd
cd /data-vol1/soro/Projects/Dnd/Drag-and-Drop-LLMs

# Record the exact files; optionally precompute token caches (~5.6 GB).
python -m workspace.vqvae.run prepare --cache

# The launcher always uses Conda dnd and physical GPU 4.
bash scripts/common_sense_reasoning/ARC-c/training_vqvae.sh
```

The launcher selects the `dnd` environment even if invoked from another Conda
environment. Physical GPU 4 is exposed as logical `cuda:0`. The data root is
detected beside the repository or inside it (`Loradatasets/common_sense_reasoning`);
`--data-root` or `DND_DATASET_ROOT` overrides detection.

The launcher waits until GPU 4 has at least 16 GiB free before importing
PyTorch or initializing CUDA. It checks every 30 seconds and leaves existing
jobs running. `VQVAE_MIN_FREE_MIB` and `VQVAE_POLL_SECONDS` configure this
headroom check; it is not a GPU memory limit or a reservation. A per-output
lock prevents duplicate queued/running launches. CPU runs and `--help` skip
the GPU-memory wait.

Defaults: 10,000 optimizer steps, batch size 4 complete adapters, AdamW
at 2e-4 with cosine decay, and W&B project `DnD-VQVAE` using the existing
login. Outputs are separate from the DnD run, in `outputs/vqvae_arc_c/`.
The launcher encodes all 200 training adapters into `train_codes.pt` after
training finishes, using `best_train_reconstruction.pt`. For example, override training settings with
`--batch-size 8 --steps 4000`. Do not assume DnD's batch size fits this model.

The first optimizer step, every 100 steps, the final step, and a requested
SIGINT/SIGTERM stop save `last.pt` atomically. Snapshots are also retained
every 1,000 steps. Checkpoints include model, EMA codebook, optimizer,
scheduler, sampler state, and Torch RNG states. Interrupts save after the
current optimizer step completes. A hard kill or hardware failure can still
lose steps since the last save. Resume with the same steps/batch/lr settings:

Whenever the observed training minibatch `weighted_mse` strictly improves,
`best_train_reconstruction.pt` is atomically replaced with a full resumable
checkpoint. Commitment loss and held-out loss are excluded from this decision.
The best loss and step are logged and preserved on resume. This is a best
minibatch loss, not an average over the entire training dataset. Resuming an
older checkpoint without these fields starts tracking at the next step; it
cannot recover an unsaved historical best model. Checkpoints record that
starting step as `best_tracking_start_step`.

```bash
bash scripts/common_sense_reasoning/ARC-c/training_vqvae.sh \
  --resume outputs/vqvae_arc_c/last.pt
```

## Export and reconstruct adapters

Use a fixed checkpoint, such as a `step_*.pt` snapshot, if training is still
running. Encoded files require the exact model/codebook used for encoding.

```bash
python -m workspace.vqvae.run encode \
  --model outputs/vqvae_arc_c/last.pt \
  --split held_out --output outputs/vqvae_arc_c/arc_c_codes.pt

python -m workspace.vqvae.run decode \
  --model outputs/vqvae_arc_c/last.pt \
  --codes outputs/vqvae_arc_c/arc_c_codes.pt \
  --output-dir outputs/vqvae_arc_c/reconstructed
```

Decoding uses only the saved codes, shared VQ-VAE checkpoint, and the matrix
names/shapes stored in the code file, not original LoRA weight values.
It writes a PEFT adapter folder for each sample, including all A/B matrices
and `adapter_config.json`. This command does not measure language-model
accuracy. Reconstruction evaluation reports weighted/unweighted token MSE
and code utilization separately in `metrics.jsonl` and W&B.

## Checks

```bash
python -m unittest discover -s tests -p test_vqvae.py -v
# Real-data, small run with resumable state and a separate output directory:
python -m workspace.vqvae.run train --device cpu --no-wandb \
  --steps 2 --stop-after 1 --batch-size 1 --eval-samples 1 \
  --output-dir outputs/vqvae_smoke
python -m workspace.vqvae.run train --device cpu --no-wandb \
  --steps 2 --batch-size 1 --eval-samples 1 \
  --output-dir outputs/vqvae_smoke --resume outputs/vqvae_smoke/last.pt
```
