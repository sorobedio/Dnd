# Prompt- and step-conditioned LoRA code generator

This generator predicts the existing residual VQ-VAE's latent indices. It does not generate or retrain the shared codebook vectors. A sample is one complete LoRA checkpoint, labelled with its task and numeric save step extracted from its source filename. Dataset preparation preserves the original code-row mapping and source hashes in `outputs/code_generator_data_holdout_arc_c/manifest.json` and exports `checkpoint_steps.csv` for inspection.

## Representation and architecture

- Frozen `all-MiniLM-L12-v2`, maximum 384 tokens per prompt. Attention-mask mean pooling of last hidden states produces one 384-dimensional vector per prompt, without L2 normalization. Unlike original DnD, this model pools the token dimension to control context length.
- Each training presentation samples 128 prompts without replacement from that checkpoint's task training prompts. No answer labels are embedded. Prompt embeddings are cached with source and encoder hashes.
- Prefix: 128 projected prompt vectors followed by one numeric step vector. The step MLP receives normalized step, normalized `log1p(step)`, and a constant; steps need not be categorical IDs seen in training.
- GPT-2-style decoder: 6 pre-LayerNorm causal transformer blocks, width 384, 6 attention heads, GELU MLPs, learned absolute position embeddings, tied token/output embeddings, and SDPA attention. No pretrained GPT-2 weights are required.
- Target: one reference ID (0–4), then 2,560 VQ indices (0–1023). Reference and code token vocabularies have disjoint input IDs and separate valid output ranges. BOS is input-only. The maximum input context is 2,690 positions.
- Teacher forcing shifts the target internally. Causal attention prevents future target leakage. Loss is mean VQ-token cross entropy plus reference cross entropy, keeping the single reference prediction from being diluted by 2,560 code predictions.
- Inference uses a KV cache and generates every code autoregressively. Greedy decoding is default; temperature and top-k sampling are optional.

## Training

On the server, from `/data-vol1/soro/Projects/Dnd`:

```bash
bash scripts/common_sense_reasoning/ARC-c/training_code_generator.sh
```

The launcher forces conda `dnd` and physical GPU 4. Defaults: 3,000 optimizer updates, batch 4, AdamW learning rate 0.0002 with warmup and cosine decay, W&B project `DnD-CodeGPT`. Preparation uses existing VQ codes and the local MiniLM model; no downloads or API key entry are needed.

The ARC-c launcher excludes all 51 ARC-c checkpoints and all ARC-c training prompts from generator training. It uses 203 checkpoints from ARC-e, BoolQ, PIQA, and HellaSwag. The earlier all-five-task experiment remains separately in `outputs/code_generator/` and is not a held-out ARC-c result. No validation split is made. Every 250 updates, all training checkpoints are evaluated using reproducible sampled training prompts. `best_train.pt` saves the lowest mean training objective; `last.pt` includes optimizer, shuffle order/cursor, RNG, configuration, data hash, and W&B run ID for exact schedule resume. Teacher-forced code accuracy is not a downstream accuracy measurement.

```bash
bash scripts/common_sense_reasoning/ARC-c/training_code_generator.sh \
  --resume outputs/code_generator_holdout_arc_c/last.pt
```

`--stop-after N` supports a bounded smoke run; it saves resumable state. Keep the same `--steps`, batch size, learning rate, seed, and evaluation interval when resuming. SIGTERM/SIGINT finish the current update, evaluate, and save.

## Generation and decoding

```bash
CUDA_VISIBLE_DEVICES=4 "$HOME/miniconda3/bin/conda" run --no-capture-output -n dnd \
  python -m workspace.code_generator.run generate \
  --model outputs/code_generator_holdout_arc_c/best_train.pt \
  --task ARC-c --checkpoint-step 250 --prompts prepare/data/ARC-c_train.json \
  --output outputs/generated_arc_c_codes.pt \
  --decode-dir outputs/generated_arc_c_adapter
```

Add `--checkpoint-step 100` to request a specific saving step. For held-out ARC-c, `--checkpoint-step 250` compares against its step-250 checkpoint. If omitted, the filename-only checkpoint catalogue still resolves ARC-c to 250. This catalogue supplies only numeric defaults; no ARC-c codes or prompt embeddings enter generator training. Tasks in the filename catalogue use their own recorded latest step; unknown task names fall back to the maximum catalogued step. The output explicitly records the resolved step and the fallback policy. This is a conditioning value, not a promise of equivalent task progress across datasets.

Prompt files can be lists of strings or repository-style records with `prompt` or `conversations`. Generation samples 128 prompts; if fewer are supplied, it repeats them to fill the prefix and records the chosen indices. Generated code files use the existing VQ decoder format and require the exact VQ checkpoint hash. Decoding writes a complete adapter and its configuration without loading the original target checkpoint.

## Unseen tasks

To exclude ARC-c from generator training, prepare a separate directory using `prepare --exclude-tasks ARC-c --output-dir ...`, then train with `--dataset .../dataset.pt --output-dir ...`. The current VQ-VAE and reference tensors already include ARC-c. Excluding it only from this generator is not a fully unseen-task experiment; that requires a separate VQ-VAE and reference construction without ARC-c, followed by re-encoding and generator training.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m unittest tests.test_code_generator -v
```

The tests check causal isolation, cached/full decoding equivalence, checkpoint step defaults, valid output alphabets, prefix gradients, and exact uninterrupted/resumed CPU training equivalence. Hiding GPUs for this CPU test avoids accelerator initialization by the optimizer on occupied devices.
