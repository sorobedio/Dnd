"""Training-only LoRA reconstruction refinement, including ARC-c."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import time

import torch

from .data import ROOT, WholeLoRADataset, importance_weights, sha256, validate_manifest
from .model import LoRAVQVAE, reconstruction_loss
from .run import atomic_save, encode, resolve_device


def training_manifest(source):
    """Preserve the frozen samples, promote ARC-c, and include latest train-task files."""
    manifest = copy.deepcopy(source)
    tasks = list(dict.fromkeys(e['dataset'] for e in manifest['entries']))
    for entry in manifest['entries']:
        entry['split'] = 'train'
    selected = {e['path'] for e in manifest['entries']}
    for task in tasks:
        folder = Path(next(e['path'] for e in manifest['entries'] if e['dataset'] == task)).parent
        latest = max(folder.glob('*.safetensors'), key=lambda p: int(p.stem))
        if str(latest) not in selected:
            manifest['entries'].append(dict(split='train', dataset=task, path=str(latest), sha256=sha256(latest)))
    manifest['selection'] = 'Frozen original train + ARC-c entries; latest file included for each represented task'
    manifest['tokenization_dtype'] = 'float32'
    manifest['settings']['datasets'] = tasks
    manifest['settings']['dataset_tag'] = None
    manifest['purpose'] = 'Training reconstruction only; no held-out split or benchmark selection'
    # Retain the same four-task importance weighting for comparable token losses.
    return manifest


def token_scales(tokens):
    """Differentiable inverse of DnD's repeated mean/std metadata."""
    width = tokens[:, :, -2:, :].unfold(-1, 2, 2).mean(-2)
    height = tokens[:, :, :, -2:].reshape(*tokens.shape[:2], -1, 2, 2).mean(-3)
    scales = (width + height) * 0.5
    mean = scales[..., 0].mean(-1) / 8.0
    std = ((scales[..., 1] - 1.6).exp() - 0.1).mean(-1) / 0.9
    return mean, std


def physical_reconstruction(prediction, target):
    """Relative error of restored weight chunks, including mean/std prediction errors."""
    mean, std = token_scales(target)
    predicted_mean, predicted_std = token_scales(prediction)
    std = std.clamp_min(1e-7)
    payload = target[:, :, :-2, :-2]
    valid = torch.isfinite(payload)
    normalized = (prediction[:, :, :-2, :-2] * (predicted_std / std)[..., None, None]
                  + ((predicted_mean - mean) / std)[..., None, None])
    error = normalized - torch.nan_to_num(payload, nan=0.0)
    return error.square()[valid].mean()


@torch.inference_mode()
def evaluate_training(model, data, weights, device):
    model.eval()
    sums = {}
    for index, entry in enumerate(data.entries):
        target = data[index][None].to(device)
        # Evaluate the actual code export/decode path, not the straight-through training approximation.
        prediction = model.decode(model.encode(target))
        weighted, plain = reconstruction_loss(prediction, target, weights)
        physical = physical_reconstruction(prediction, target)
        row = sums.setdefault(entry['dataset'], dict(n=0, weighted_mse=0., mse=0., physical_relative_mse=0.))
        row['n'] += 1
        for key, value in [('weighted_mse', weighted), ('mse', plain), ('physical_relative_mse', physical)]:
            row[key] += value.item()
    metrics = {}
    for key in ['weighted_mse', 'mse', 'physical_relative_mse']:
        metrics['train_' + key] = sum(row[key] for row in sums.values()) / len(data)
    metrics['per_task'] = {task: {key: value / row['n'] if key != 'n' else value
                                 for key, value in row.items()} for task, row in sums.items()}
    if not all(math.isfinite(metrics[key]) for key in ['train_weighted_mse', 'train_mse', 'train_physical_relative_mse']):
        raise RuntimeError('Non-finite full-training reconstruction metric')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialize-from', default=str(ROOT / 'outputs/vqvae_arc_c/last.pt'))
    parser.add_argument('--output-dir', default=str(ROOT / 'outputs/vqvae_train_arc_c_refined'))
    parser.add_argument('--resume')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--joint-steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--physical-weight', type=float, default=0.01)
    parser.add_argument('--commitment', type=float, default=0.25)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--encode-after', action='store_true')
    parser.add_argument('--reference-residual', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.save_every) < 1 or not 0 <= args.joint_steps <= args.steps:
        parser.error('Invalid steps, batch size, or evaluation intervals')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / '.refinement.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Use --resume or a new output directory')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    state = torch.load(args.resume or args.initialize_from, map_location='cpu', weights_only=True)
    manifest = state['manifest'] if args.resume else training_manifest(state['manifest'])
    validate_manifest(manifest)
    if args.resume:
        for key in ['steps', 'joint_steps', 'batch_size', 'learning_rate', 'physical_weight', 'commitment', 'warmup', 'seed', 'reference_residual']:
            if state['training'].get(key, False) != getattr(args, key):
                raise ValueError(f'Resume must preserve {key}')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    data = WholeLoRADataset(manifest, 'train', output / 'token_cache')
    weights = importance_weights(manifest).to(device)
    config = dict(state['model_config'], commitment=args.commitment)
    model = LoRAVQVAE(**config).to(device)
    model.load_state_dict(state['model'])
    old_baseline = None
    if args.reference_residual and not args.resume:
        old_baseline = evaluate_training(model, data, weights, device)
        tasks = list(dict.fromkeys(e['dataset'] for e in data.entries))
        references, scales = [], []
        for task in tasks:
            indices = [i for i,e in enumerate(data.entries) if e['dataset'] == task]
            center = sum(torch.nan_to_num(data[i], nan=0.) for i in indices) / len(indices)
            variance = sum((torch.nan_to_num(data[i], nan=0.) - center).square().mean().item()
                           for i in indices) / len(indices)
            references.append(center)
            scales.append(max(variance**.5, 1e-4))
        del model
        config['reference_count'] = len(tasks)
        model = LoRAVQVAE(**config).to(device)
        model.reference_tokens.copy_(torch.stack(references).to(device))
        model.reference_scale.copy_(torch.tensor(scales, device=device))
        model.references_initialized.fill_(True)
        torch.nn.init.zeros_(model.decoder.module_list[-1].linear2.weight)
        torch.nn.init.zeros_(model.decoder.module_list[-1].linear2.bias)
        initialization = torch.stack([data[next(i for i,e in enumerate(data.entries) if e['dataset'] == task)]
                                      for task in tasks]).to(device)
        with torch.no_grad():
            model(initialization)
        del initialization, references
        manifest['reference_tasks'] = tasks
        manifest['reference_selection'] = 'Nearest shared training reference, determined from input weights; no task label at inference'
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    generator = torch.Generator().manual_seed(args.seed)
    step, order, position = 0, [], 0
    best_value, best_step = float('inf'), None
    run = None
    if args.resume:
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng'])
        order, position, step = state['order'], state['position'], state['step']
        best_value, best_step = state['best_train_reconstruction'], state['best_train_step']
    if args.wandb:
        import wandb
        run = wandb.init(project='DnD-VQVAE', name='five_tasks_train_reconstruction',
                         id=state.get('wandb_id') if args.resume else None,
                         resume='allow' if args.resume else None, dir=str(output),
                         config={**vars(args), 'n_training_checkpoints': len(data), 'tokenization_dtype': 'float32'})
    if args.resume:
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    del state
    stop_requested = False

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print('Stop requested; saving at optimizer-step boundary', flush=True)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)

    def save(name, metrics=None):
        atomic_save(dict(version=2, model_config=model.config, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), step=step, training=vars(args), manifest=manifest,
                         sampler_rng=generator.get_state(), order=order, position=position,
                         torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                         best_train_reconstruction=best_value, best_train_step=best_step,
                         best_selection_metric='full_training_mean_weighted_mse', evaluation=metrics,
                         wandb_id=run.id if run else None), output / name)

    print(f'Training {len(data)} complete adapters, no held-out split; FP32; batch={args.batch_size}', flush=True)
    # Evaluate the old weights on the exact new training representation before changing parameters.
    if not args.resume:
        baseline = evaluate_training(model, data, weights, device)
        (output / 'baseline.json').write_text(json.dumps(old_baseline or baseline, indent=2) + '\n')
        if old_baseline is not None:
            (output / 'initial_reference.json').write_text(json.dumps(baseline, indent=2) + '\n')
        best_value, best_step = baseline['train_weighted_mse'], 0
        save('best_train_reconstruction.pt', baseline)
        save('last.pt', baseline)
        print('BASELINE ' + json.dumps(baseline), flush=True)
    with (output / 'metrics.jsonl').open('a') as log:
        while step < args.steps:
            started = time.monotonic()
            if position == len(order):
                order = torch.randperm(len(data), generator=generator).tolist()
                position = 0
            indices = order[position:position + args.batch_size]
            position += len(indices)
            target = torch.stack([data[i] for i in indices]).to(device)
            refining = step >= args.joint_steps
            model.train()
            if refining:
                model.encoder.eval()
                model.codebook.eval()
            for parameter in model.encoder.parameters():
                parameter.requires_grad_(not refining)
            optimizer.zero_grad(set_to_none=True)
            if refining:
                with torch.no_grad():
                    model.eval()
                    codes = model.encode(target)
                    counts = torch.bincount(codes.flatten(), minlength=model.codebook.size).float()
                    probabilities = counts / counts.sum()
                    perplexity = (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp()
                    commitment = torch.zeros((), device=device)
                model.decoder.train()
                prediction = model.decode(codes)
            else:
                prediction, _, commitment, perplexity = model(target)
            weighted, plain = reconstruction_loss(prediction, target, weights)
            physical = physical_reconstruction(prediction, target)
            normalization = model.reference_scale.square().mean() if model.reference_count else 1.
            loss = (weighted + args.physical_weight * physical) / normalization + (0 if refining else commitment)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite reconstruction loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            # Restart a gentle decay when codes freeze, then polish at a small final LR.
            phase_start = args.joint_steps if refining else 0
            phase_length = args.steps - args.joint_steps if refining else max(args.joint_steps, 1)
            progress = (step - phase_start) / max(phase_length - 1, 1)
            factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            if args.warmup:
                factor *= min(1., (step - phase_start + 1) / args.warmup)
            for group in optimizer.param_groups:
                group['lr'] = args.learning_rate * factor
            optimizer.step()
            step += 1
            metrics = dict(step=step, phase='decoder_refinement' if refining else 'joint',
                           weighted_mse=weighted.item(), mse=plain.item(), physical_relative_mse=physical.item(),
                           commitment=commitment.item(), perplexity=perplexity.item(), gradient_norm=float(grad),
                           learning_rate=optimizer.param_groups[0]['lr'], step_seconds=time.monotonic() - started)
            del target, prediction, loss, weighted, plain, physical, commitment, perplexity
            stopping = stop_requested or (args.stop_after and step >= args.stop_after)
            if step % args.eval_every == 0 or step in [args.joint_steps, args.steps] or stopping:
                evaluation = evaluate_training(model, data, weights, device)
                metrics.update(evaluation)
                if evaluation['train_weighted_mse'] < best_value:
                    best_value, best_step = evaluation['train_weighted_mse'], step
                    save('best_train_reconstruction.pt', evaluation)
                    print(f'BEST MODEL step={step} full_train_weighted_mse={best_value:.8f}', flush=True)
            metrics.update(best_train_reconstruction=best_value, best_train_step=best_step)
            log.write(json.dumps(metrics) + '\n'); log.flush()
            if step % 10 == 0 or step == 1 or 'train_weighted_mse' in metrics:
                print(json.dumps(metrics), flush=True)
            if run:
                wandb_metrics = {key: value for key, value in metrics.items() if key != 'per_task'}
                if 'per_task' in metrics:
                    wandb_metrics.update({f'train/{task}/{key}': value for task, row in metrics['per_task'].items()
                                          for key, value in row.items() if key != 'n'})
                run.log(wandb_metrics, step=step)
            if step % args.save_every == 0 or step == args.steps or stopping:
                save('last.pt', metrics if 'train_weighted_mse' in metrics else None)
            if stopping:
                break
    if run:
        run.finish()
    print(f'Saved step {step}, best model step {best_step}; {output}', flush=True)
    if args.encode_after and step == args.steps:
        del optimizer, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        encode(argparse.Namespace(model=str(output / 'best_train_reconstruction.pt'),
                                 output=str(output / 'train_codes.pt'), split='train', limit=None, device=args.device))


if __name__ == '__main__':
    main()
