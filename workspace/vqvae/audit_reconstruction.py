"""Audit restored A, B, and BA tensors on the latest training adapters only."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .data import WholeLoRADataset, unpack_tokens, weight_schema, sha256
from .run import load_model, autocast, resolve_device


def weight_errors(original, reconstructed):
    result = {}
    for kind in ['lora_A', 'lora_B']:
        error = norm = 0.
        for key, value in original.items():
            if kind not in key:
                continue
            a, b = value.double(), reconstructed[key].double()
            error += (a - b).square().sum().item()
            norm += a.square().sum().item()
        result[kind + '_relative_l2'] = (error / max(norm, 1e-30))**.5
    error = norm = 0.
    for key in original:
        if not key.endswith('lora_A.weight'):
            continue
        other = key.replace('lora_A.weight', 'lora_B.weight')
        a, b = original[key].double(), original[other].double()
        ah, bh = reconstructed[key].double(), reconstructed[other].double()
        # Frobenius norms via rank-8 Gram matrices; never materialize dense BA.
        n = ((b.T @ b) * (a @ a.T)).sum().item()
        nh = ((bh.T @ bh) * (ah @ ah.T)).sum().item()
        cross = ((bh.T @ b) * (ah @ a.T)).sum().item()
        error += max(0., n + nh - 2 * cross)
        norm += n
    result['delta_BA_relative_l2'] = (error / max(norm, 1e-30))**.5
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(8)
    device = resolve_device(args.device)
    state = torch.load(args.model, map_location='cpu', weights_only=True)
    entries = state['manifest']['entries']
    assert all(e['split'] == 'train' for e in entries)
    tasks = list(dict.fromkeys(e['dataset'] for e in entries))
    selected = [max((e for e in entries if e['dataset'] == task), key=lambda e: int(Path(e['path']).stem))
                for task in tasks]
    report = dict(scope='Latest checkpoint per task, all confirmed training members',
                  model=str(Path(args.model).resolve()), model_sha256=sha256(args.model),
                  model_step=state['step'], tasks=[])
    del state
    errors = {e['dataset']: dict(task=e['dataset'], source=e['path']) for e in selected}
    for variant, path in [('old', args.baseline), ('new', args.model)]:
        model, manifest, _ = load_model(path, device)
        for entry in selected:
            assert sha256(entry['path']) == entry['sha256']
            data = WholeLoRADataset(dict(manifest, entries=[entry]), 'train', Path(path).parent / 'token_cache')
            with torch.inference_mode(), autocast(device, enabled=not model.requires_float32):
                codes = model.encode(data[0][None].to(device))
                prediction = model.decode(codes)
            reconstructed = unpack_tokens(prediction[0], weight_schema(entry['path']))
            original = load_file(entry['path'])
            errors[entry['dataset']][variant] = weight_errors(original, reconstructed)
            print(entry['dataset'], variant, errors[entry['dataset']][variant], flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report['tasks'] = list(errors.values())
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
