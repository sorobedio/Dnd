"""Paired downstream evaluation of original and VQ-reconstructed complete LoRAs."""
import argparse
import json
import re
import shutil
from pathlib import Path

TASKS = ['ARC-e', 'BoolQ', 'PIQA', 'HellaSwag', 'ARC-c', 'OBQA', 'WinoGrande']


def answer(text, task):
    # Require an explicit answer; do not guess from option text/character counts.
    choices = r'true|false' if task == 'BoolQ' else r'A|B|C|D|E|1|2|3|4|5'
    match = re.search(r'\[\s*(' + choices + r')\s*\]', text, re.I)
    if not match:
        match = re.match(r'\s*(?:the answer is\s*|answer\s*:\s*)?(' + choices + r')(?!\w)', text, re.I)
    result = match.group(1).upper() if match else None
    return dict(zip('12345', 'ABCDE')).get(result, result)


def evaluation_data(task, root, output):
    if task != 'PIQA':
        split = 'validation' if task in ['HellaSwag', 'WinoGrande'] else 'test'
        return root / 'prepare/data' / f'{task}_{split}.json'
    # The repository PIQA test file has placeholder labels. Use official dev labels.
    import urllib.request
    import zipfile
    path = output.resolve() / 'PIQA_validation.json'
    if not path.exists():
        url = 'https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip'
        archive = output / 'physicaliqa-train-dev.zip'
        if not archive.exists():
            with urllib.request.urlopen(url, timeout=60) as source, archive.open('wb') as dest:
                shutil.copyfileobj(source, dest)
        with zipfile.ZipFile(archive) as z:
            data = [json.loads(line) for line in z.read('physicaliqa-train-dev/dev.jsonl').decode().splitlines()]
            labels = [int(x) for x in z.read('physicaliqa-train-dev/dev-labels.lst').decode().splitlines()]
        assert len(data) == len(labels) == 1838 and set(labels) == {0, 1}
        system = json.loads((root / 'prepare/data/PIQA_test.json').read_text())[0]['system']
        samples = [dict(prompt=f"{x['goal']}\nA: {x['sol1']}\nB: {x['sol2']}",
                        response='[' + 'AB'[label] + ']', system=system) for x, label in zip(data, labels)]
        path.write_text(json.dumps(samples, indent=2) + '\n')
        (output / 'PIQA_source.json').write_text(json.dumps(dict(url=url, split='validation',
            reason='Repository PIQA test labels are all A; official test labels are unavailable'), indent=2))
    return path


def prepare(args):
    import torch
    from safetensors.torch import load_file, save_file
    from .data import (ROOT, WholeLoRADataset, importance_weights, sha256,
                       unpack_tokens, weight_schema)
    from .model import reconstruction_loss
    from .run import autocast, load_model, resolve_device

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'comparison.json').exists():
        raise FileExistsError('Comparison already prepared; choose a new output directory')
    device = resolve_device(args.device)
    model, manifest, step = load_model(args.model, device)
    criterion = importance_weights(manifest).to(device)
    record = dict(vq_model=str(Path(args.model).resolve()), vq_sha256=sha256(args.model),
                  vq_step=step, base_model=str(Path(args.base_model).resolve()),
                  selection='highest numeric LoRA filename per task', tasks=[])
    for task in args.tasks:
        files = list((Path(args.data_root) / task).glob('*.safetensors'))
        source = max(files, key=lambda p: int(p.stem)).resolve()
        entry = dict(path=str(source), dataset=task, split='evaluation', sha256=sha256(source))
        dataset = WholeLoRADataset(dict(manifest, entries=[entry]), 'evaluation')
        with torch.inference_mode(), autocast(device, enabled=not model.requires_float32):
            target = dataset[0][None].to(device)
            codes = model.encode(target)
            prediction = model.decode(codes)
            weighted, plain = reconstruction_loss(prediction, target, criterion)
        recovered = unpack_tokens(prediction[0], weight_schema(source))
        original = load_file(str(source))
        assert set(recovered) == set(original)
        squared_error = squared_norm = dot = rec_norm = 0.0
        for name, weight in original.items():
            a, b = weight.double(), recovered[name].double()
            assert a.shape == b.shape
            squared_error += (a - b).square().sum().item()
            squared_norm += a.square().sum().item()
            rec_norm += b.square().sum().item()
            dot += (a * b).sum().item()
        for variant in ['original', 'reconstructed']:
            folder = output / task / variant
            folder.mkdir(parents=True, exist_ok=True)
            if variant == 'original':
                shutil.copyfile(source, folder / 'adapter_model.safetensors')
            else:
                save_file(recovered, str(folder / 'adapter_model.safetensors'))
                torch.save(codes.cpu(), folder / 'codes.pt')
            config = json.loads((ROOT / 'configs/Qwen0.5/adapter_config.json').read_text())
            config['base_model_name_or_path'] = record['base_model']
            (folder / 'adapter_config.json').write_text(json.dumps(config, indent=2))
        datafile = evaluation_data(task, ROOT, output)
        membership = next((e['split'] for e in manifest['entries'] if e['sha256'] == entry['sha256']), 'not_selected')
        info = dict(task=task, source=str(source), source_sha256=entry['sha256'],
                    vq_membership=membership, datafile=str(datafile), data_sha256=sha256(datafile),
                    weighted_mse=weighted.item(), token_mse=plain.item(),
                    weight_relative_l2=(squared_error / squared_norm)**0.5,
                    weight_cosine=dot / (squared_norm * rec_norm)**0.5)
        record['tasks'].append(info)
        print(json.dumps(info), flush=True)
    assert sha256(args.model) == record['vq_sha256'], 'VQ model changed during reconstruction'
    (output / 'comparison.json').write_text(json.dumps(record, indent=2) + '\n')


def evaluate(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from .data import sha256

    output = Path(args.output)
    record = json.loads((output / 'comparison.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(record['base_model'])
    engine = LLM(model=record['base_model'], dtype='bfloat16', tensor_parallel_size=1,
                 enable_lora=True, max_lora_rank=8, max_loras=1, max_cpu_loras=16,
                 max_model_len=4096, max_num_seqs=128, gpu_memory_utilization=0.25,
                 enforce_eager=True, seed=999)
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=999)
    report = dict(protocol='Repository prompts, Qwen chat template, greedy generation, explicit answer extraction',
                  max_new_tokens=args.max_new_tokens, vq_model=record['vq_model'], vq_step=record['vq_step'], tasks=[])
    request_id = 0
    for task_info in record['tasks']:
        task = task_info['task']
        assert sha256(task_info['datafile']) == task_info['data_sha256']
        samples = json.loads(Path(task_info['datafile']).read_text())
        if args.limit:
            samples = samples[:args.limit]
        prompts = []
        for sample in samples:
            messages = [{'role': 'system', 'content': sample['system']}] if sample.get('system') else []
            messages.append({'role': 'user', 'content': sample['prompt']})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            assert len(tokenizer.encode(prompt)) + args.max_new_tokens <= 4096, 'Prompt exceeds context limit'
            prompts.append(prompt)
        row = dict(task_info, n=len(samples))
        predictions = {}
        for variant in ['original', 'reconstructed']:
            request_id += 1
            destination = output / task / f'{variant}_predictions.jsonl'
            if destination.exists():
                results = [json.loads(line) for line in destination.read_text().splitlines()]
                assert len(results) == len(samples), 'Existing predictions have wrong sample count'
            else:
                generated = engine.generate(prompts, sampling, lora_request=LoRARequest(
                    f'{task}_{variant}', request_id, str((output / task / variant).resolve())))
                results = []
                for i, (sample, result) in enumerate(zip(samples, generated)):
                    text = result.outputs[0].text
                    predicted = answer(text, task)
                    label = answer(sample['response'], task)
                    assert label is not None
                    results.append(dict(index=i, predict=text, answer=predicted, label=label,
                                        correct=predicted == label, finish_reason=result.outputs[0].finish_reason))
                temporary = destination.with_suffix('.tmp')
                temporary.write_text(''.join(json.dumps(r) + '\n' for r in results))
                temporary.replace(destination)
            predictions[variant] = results
            row[variant + '_accuracy'] = sum(r['correct'] for r in results) / len(results)
            row[variant + '_invalid'] = sum(r['answer'] is None for r in results)
            row[variant + '_truncated'] = sum(r['finish_reason'] == 'length' for r in results)
            print(task, variant, row[variant + '_accuracy'], flush=True)
        row['delta_percentage_points'] = 100 * (row['reconstructed_accuracy'] - row['original_accuracy'])
        row['answer_agreement'] = sum(a['answer'] == b['answer'] for a, b in zip(
            predictions['original'], predictions['reconstructed'])) / len(samples)
        row['correct_to_wrong'] = sum(a['correct'] and not b['correct'] for a,b in zip(
            predictions['original'], predictions['reconstructed']))
        row['wrong_to_correct'] = sum(not a['correct'] and b['correct'] for a,b in zip(
            predictions['original'], predictions['reconstructed']))
        report['tasks'].append(row)
        (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'evaluate'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--model')
    parser.add_argument('--base-model')
    parser.add_argument('--data-root')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tasks', nargs='+', default=TASKS, choices=TASKS)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    args = parser.parse_args()
    if args.command == 'prepare' and not all([args.model, args.base_model, args.data_root]):
        parser.error('prepare requires --model, --base-model, --data-root')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
