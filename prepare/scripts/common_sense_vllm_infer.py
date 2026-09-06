"""Evaluate prepared common-sense LoRAs with the installed vLLM API."""
import json
import os
from pathlib import Path

import fire
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main(model_name_or_path: str, adapter_name_or_path: str, dataset: str, save_name: str, max_samples: int = None, max_new_tokens: int = 1024):
    samples = json.loads((Path('data') / f'{dataset}.json').read_text())
    if max_samples is not None:
        samples = samples[:max_samples]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    prompts = []
    for sample in samples:
        messages = []
        if sample.get('system'):
            messages.append({'role': 'system', 'content': sample['system']})
        messages.append({'role': 'user', 'content': sample['prompt']})
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    engine = LLM(
        model=model_name_or_path, dtype='bfloat16', tensor_parallel_size=1,
        enable_lora=True, max_lora_rank=64, max_model_len=3072,
        gpu_memory_utilization=float(os.environ.get('DND_VLLM_MEMORY_UTILIZATION', '0.25')),
        max_num_seqs=32, enforce_eager=True,
    )
    outputs = engine.generate(
        prompts,
        SamplingParams(temperature=0.95, top_p=0.7, top_k=50, max_tokens=max_new_tokens, seed=999),
        lora_request=LoRARequest('dnd', 1, str(Path(adapter_name_or_path).resolve())),
    )
    output_path = Path(save_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        for sample, output in zip(samples, outputs):
            f.write(json.dumps({'predict': output.outputs[0].text, 'label': sample['response']}, ensure_ascii=False) + '\n')
    print(f'Saved {len(outputs)} predictions to {output_path}', flush=True)


if __name__ == '__main__':
    fire.Fire(main)
