"""Compare a freely generated held-out-task adapter with its original LoRA."""
import argparse
import json
import shutil
from pathlib import Path

import torch

from .run import ROOT, sha256


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--generator', required=True)
    p.add_argument('--codes', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    args = p.parse_args()
    state = torch.load(args.generator, map_location='cpu', weights_only=True)
    encoded = torch.load(args.codes, map_location='cpu', weights_only=True)
    generation = encoded['generation']
    assert generation['generator_sha256'] == sha256(args.generator)
    assert encoded['model_sha256'] == state['metadata']['vq_model_sha256']
    assert len(encoded['entries']) == 1
    task = encoded['entries'][0]['dataset']
    assert all(e['dataset'] != task for e in state['metadata']['entries']), 'Task was used in generator training'
    assert task not in state['metadata']['prompt_sources'], 'Task prompts were used in generator training'
    output = Path(args.output)
    if (output/'comparison.json').exists():
        raise FileExistsError('Choose a fresh comparison output directory')
    step = generation['checkpoint_step']
    original = Path(args.data_root)/task/f'{step}.safetensors'
    generated = Path(args.adapter)/'adapter_model.safetensors'
    config = json.loads((Path(args.adapter)/'adapter_config.json').read_text())
    config['base_model_name_or_path'] = str(Path(args.base_model).resolve())
    for variant, source in [('original', original), ('reconstructed', generated)]:
        folder = output/task/variant
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, folder/'adapter_model.safetensors')
        (folder/'adapter_config.json').write_text(json.dumps(config, indent=2)+'\n')
    from workspace.vqvae.evaluate_downstream import evaluation_data, evaluate
    datafile = evaluation_data(task, ROOT, output)
    row = dict(task=task, source=str(original.resolve()), source_sha256=sha256(original),
               datafile=str(datafile.resolve()), data_sha256=sha256(datafile),
               generator_membership='held_out', generator_sha256=sha256(args.generator),
               generator_step=state['step'], checkpoint_step=step,
               generated_adapter_sha256=sha256(generated), generated_codes_sha256=sha256(args.codes),
               generation=generation, variant_meaning='reconstructed denotes the freely generated adapter, not target reconstruction')
    record = dict(vq_model=state['metadata']['vq_model'], vq_step=None,
                  vq_sha256=encoded['model_sha256'], base_model=str(Path(args.base_model).resolve()),
                  selection='Original saving step matched to free generation condition', tasks=[row])
    (output/'comparison.json').write_text(json.dumps(record, indent=2)+'\n')
    # The shared evaluator uses the same prompts, decoding and answer parser for both adapters.
    # Its legacy "reconstructed" field is labelled explicitly above as generated.
    del state, encoded
    evaluate(argparse.Namespace(output=str(output), max_new_tokens=args.max_new_tokens, limit=None))


if __name__ == '__main__':
    main()
