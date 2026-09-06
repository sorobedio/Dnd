"""Prepare, train, encode, and decode complete Qwen0.5B LoRA checkpoints."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import time

import torch
from safetensors.torch import save_file

from .data import (ROOT, WholeLoRADataset, build_manifest, importance_weights,
                   sha256, unpack_tokens, validate_manifest, weight_schema, default_data_root)
from .model import LoRAVQVAE, reconstruction_loss


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def autocast(device, enabled=True):
    return (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and enabled
            else contextlib.nullcontext())


def resolve_device(name):
    device = torch.device(name)
    # CUDA_VISIBLE_DEVICES=4 maps physical GPU 4 to logical cuda:0.
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    return device


def manifest_for(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        validate_manifest(manifest)
    else:
        manifest = build_manifest(args.data_root)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare(args):
    manifest = manifest_for(args)
    for split in ("train", "held_out"):
        dataset = WholeLoRADataset(manifest, split, Path(args.output_dir) / "token_cache")
        print(f"{split}: {len(dataset)} COMPLETE adapters; token shape [4296,10,130]", flush=True)
        if args.cache:
            for i in range(len(dataset)):
                dataset[i]
                if (i + 1) % 10 == 0:
                    print(f"Cached {split} {i + 1}/{len(dataset)}", flush=True)
    print("Manifest ready:", Path(args.output_dir) / "manifest.json", flush=True)


@torch.no_grad()
def evaluate(model, dataset, weights, device, limit):
    model.eval()
    weighted, mse, count = 0.0, 0.0, 0
    usage = torch.zeros(model.codebook.size, device=device)
    for i in range(min(limit, len(dataset))):
        target = dataset[i][None].to(device)
        with autocast(device):
            prediction, codes, _, _ = model(target)
            loss, plain = reconstruction_loss(prediction, target, weights)
        weighted += loss.item()
        mse += plain.item()
        count += 1
        usage += torch.bincount(codes.flatten(), minlength=model.codebook.size)
    probabilities = usage / usage.sum()
    model.train()
    return dict(held_out_weighted_mse=weighted / count, held_out_mse=mse / count,
                held_out_codes_used=(usage > 0).sum().item(),
                held_out_perplexity=torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item())


def train(args):
    if min(args.steps, args.batch_size, args.eval_samples, args.eval_every, args.save_every) < 1:
        raise ValueError("steps, batch size, evaluation counts, and save intervals must be positive")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output_dir)
    if (output / "last.pt").exists() and not args.resume:
        raise ValueError("A training checkpoint exists; use --resume or a new output directory")
    manifest = manifest_for(args)
    data = WholeLoRADataset(manifest, "train", output / "token_cache")
    held_out = WholeLoRADataset(manifest, "held_out", output / "token_cache")
    weights = importance_weights(manifest).to(device)
    state = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if state and state["manifest"] != manifest:
        raise ValueError("Resume manifest differs from the saved training data")
    if state and any(state["training"][key] != getattr(args, key)
                     for key in ("steps", "batch_size", "learning_rate")):
        raise ValueError("Resume must preserve steps, batch_size, and learning_rate")
    model = LoRAVQVAE(**state["model_config"]) if state else LoRAVQVAE(codebook_size=args.codebook_size)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    sampler = torch.Generator().manual_seed(args.seed)
    step = 0
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
    best_train_reconstruction = state.get("best_train_reconstruction", float("inf")) if state else float("inf")
    best_train_step = state.get("best_train_step") if state else None
    best_tracking_start_step = state.get("best_tracking_start_step", step + 1) if state else 1
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="DnD-VQVAE", name="ARC-c_whole_lora",
                         id=state.get("wandb_id") if state else None,
                         resume="allow" if state else None,
                         config={**vars(args), "model": model.config,
                                 "train_checkpoints": len(data), "held_out_checkpoints": len(held_out)},
                         dir=str(output))
    if state:
        sampler.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    del state
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested: will save after the current optimizer step.", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)

    def save(save_latest=True, save_best=False):
        snapshot = dict(version=1, model_config=model.config, model=model.state_dict(),
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), step=step,
                        training=vars(args), manifest=manifest, sampler_rng=sampler.get_state(),
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        wandb_id=run.id if run else None,
                        best_train_reconstruction=best_train_reconstruction,
                        best_train_step=best_train_step, best_tracking_start_step=best_tracking_start_step)
        if save_latest:
            atomic_save(snapshot, output / "last.pt")
            if step % 1000 == 0:
                atomic_save(snapshot, output / f"step_{step:06d}.pt")
        if save_best:
            atomic_save(snapshot, output / "best_train_reconstruction.pt")
            print(f"Saved best training reconstruction: step={step}, "
                  f"weighted_mse={best_train_reconstruction:.8f}", flush=True)

    print(f"Train: {len(data)} complete checkpoints; held out: {len(held_out)}; "
          f"batch={args.batch_size}; codes/checkpoint=2560; device={device}", flush=True)
    model.train()
    with (output / "metrics.jsonl").open("a") as metrics_file:
        while step < args.steps:
            started = time.monotonic()
            # Sample whole adapters with replacement, as DnD's repeated dataset does.
            indices = torch.randint(len(data), (args.batch_size,), generator=sampler).tolist()
            target = torch.stack([data[i] for i in indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                prediction, codes, commitment, perplexity = model(target)
                reconstruction, mse = reconstruction_loss(prediction, target, weights)
                loss = reconstruction + commitment
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            metrics = dict(step=step, weighted_mse=reconstruction.item(), mse=mse.item(),
                           commitment=commitment.item(), perplexity=perplexity.item(),
                           codes_used=codes.unique().numel(), gradient_norm=float(grad_norm),
                           learning_rate=optimizer.param_groups[0]["lr"],
                           step_seconds=time.monotonic() - started)
            del target, prediction, codes, loss, reconstruction, mse, commitment, perplexity
            if device.type == "cuda":
                metrics["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
            if step % args.eval_every == 0 or step == args.steps:
                metrics.update(evaluate(model, held_out, weights, device, args.eval_samples))
            improved = metrics["weighted_mse"] < best_train_reconstruction
            if improved:
                best_train_reconstruction = metrics["weighted_mse"]
                best_train_step = step
            metrics.update(best_train_reconstruction=best_train_reconstruction,
                           best_train_step=best_train_step)
            print(json.dumps(metrics), flush=True)
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()
            if run:
                run.log(metrics, step=step)
            stopping = stop_requested or (args.stop_after and step >= args.stop_after)
            save_latest = step == 1 or step % args.save_every == 0 or step == args.steps or stopping
            if save_latest or improved:
                save(save_latest=save_latest, save_best=improved)
            if stopping:
                break
    if run:
        run.finish()
    print(f"Saved resumable checkpoint at step {step}: {output / 'last.pt'}", flush=True)
    if args.encode_after and step == args.steps:
        # Free optimizer/gradients before exporting all training adapters.
        del optimizer, scheduler, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        best_path = output / "best_train_reconstruction.pt"
        encode(argparse.Namespace(model=str(best_path if best_path.exists() else output / "last.pt"), output=str(output / "train_codes.pt"),
                                  split="train", limit=None, device=args.device))


def load_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = LoRAVQVAE(**state["model_config"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    model.requires_float32 = state['manifest'].get('tokenization_dtype') == 'float32'
    if model.requires_float32:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
    return model, state["manifest"], state["step"]


@torch.no_grad()
def encode(args):
    device = resolve_device(args.device)
    checkpoint_hash = sha256(args.model)
    model, manifest, step = load_model(args.model, device)
    validate_manifest(manifest)
    dataset = WholeLoRADataset(manifest, args.split, Path(args.model).parent / "token_cache")
    number = min(args.limit, len(dataset)) if args.limit else len(dataset)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    schema = weight_schema(dataset.entries[0]["path"])
    codes = []
    for i in range(number):
        if weight_schema(dataset.entries[i]["path"]) != schema:
            raise ValueError("Adapter structures differ; cannot share a decoding schema")
        with autocast(device, enabled=not model.requires_float32):
            indices = model.encode(dataset[i][None].to(device))
        codes.append(indices[0].cpu().to(torch.int32))
        print(f"Encoded complete adapter {i + 1}/{number}", flush=True)
    if sha256(args.model) != checkpoint_hash:
        raise RuntimeError("Model checkpoint changed during encoding; use a fixed snapshot")
    atomic_save(dict(version=1, codes=torch.stack(codes), schema=schema,
                     entries=dataset.entries[:number], model_sha256=checkpoint_hash, step=step,
                     token_shape=manifest["token_shape"], codebook_size=model.codebook.size), args.output)
    print(f"Saved codes {tuple(torch.stack(codes).shape)}: {args.output}", flush=True)


@torch.no_grad()
def decode(args):
    device = resolve_device(args.device)
    encoded = torch.load(args.codes, map_location="cpu", weights_only=True)
    if sha256(args.model) != encoded["model_sha256"]:
        raise ValueError("Codes require the exact model/codebook checkpoint used to encode them")
    model, _, _ = load_model(args.model, device)
    number = min(args.limit, len(encoded["codes"])) if args.limit else len(encoded["codes"])
    for i in range(number):
        entry = encoded["entries"][i]
        folder = Path(args.output_dir) / entry["dataset"] / Path(entry["path"]).stem
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / "adapter_model.safetensors"
        if destination.exists():
            raise FileExistsError(destination)
        with autocast(device, enabled=not model.requires_float32):
            tokens = model.decode(encoded["codes"][i:i + 1].to(device))
        weights = unpack_tokens(tokens[0], encoded["schema"])
        save_file(weights, str(destination))
        config = json.loads((ROOT / "configs/Qwen0.5/adapter_config.json").read_text())
        candidates = [ROOT / 'models/Qwen2.5-0.5B-Instruct',
                      ROOT / 'Drag-and-Drop-LLMs/models/Qwen2.5-0.5B-Instruct']
        config["base_model_name_or_path"] = str(next((p for p in candidates if (p / 'config.json').exists()), candidates[0]))
        (folder / "adapter_config.json").write_text(json.dumps(config, indent=2))
        print(f"Decoded complete adapter {i + 1}/{number}: {folder}", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--threads", type=int, default=8)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train"):
        p = sub.add_parser(name)
        p.add_argument("--data-root", default=str(default_data_root()))
        p.add_argument("--output-dir", default=str(ROOT / "outputs/vqvae_arc_c"))
        if name == "prepare":
            p.add_argument("--cache", action="store_true")
        else:
            p.add_argument("--device", default="cuda")
            p.add_argument("--steps", type=int, default=10000)
            p.add_argument("--batch-size", type=int, default=4)
            p.add_argument("--codebook-size", type=int, default=1024)
            p.add_argument("--learning-rate", type=float, default=2e-4)
            p.add_argument("--seed", type=int, default=999)
            p.add_argument("--save-every", type=int, default=100)
            p.add_argument("--eval-every", type=int, default=500)
            p.add_argument("--eval-samples", type=int, default=50)
            p.add_argument("--resume")
            p.add_argument("--stop-after", type=int)
            p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
            p.add_argument("--encode-after", action="store_true")
    p = sub.add_parser("encode")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "held_out"], default="train")
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("decode")
    p.add_argument("--model", required=True)
    p.add_argument("--codes", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
