"""Compare an original and generated LoRA adapter on a benchmark task."""
import argparse, json, re, shutil
from pathlib import Path


def parse_answer(text, task):
    choices = r'true|false' if task == 'BoolQ' else r'A|B|C|D|E|1|2|3|4|5'
    m = re.search(r'\[\s*(' + choices + r')\s*\]', text, re.I)
    if not m: m = re.match(r'\s*(?:the answer is\s*|answer\s*:\s*)?(' + choices + r')(?!\w)', text, re.I)
    value = m.group(1).upper() if m else None
    return dict(zip('12345','ABCDE')).get(value, value)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-model',required=True);p.add_argument('--original',required=True)
    p.add_argument('--generated',required=True);p.add_argument('--data',required=True)
    p.add_argument('--task',default='ARC-e');p.add_argument('--output',required=True)
    p.add_argument('--max-new-tokens',type=int,default=1024);p.add_argument('--limit',type=int)
    args=p.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rows=json.loads(Path(args.data).read_text())
    if args.limit: rows=rows[:args.limit]
    tokenizer=AutoTokenizer.from_pretrained(args.base_model,local_files_only=True)
    prompts=[]
    for row in rows:
        messages=[]
        if row.get('system'): messages.append({'role':'system','content':row['system']})
        messages.append({'role':'user','content':row['prompt']})
        prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
    staging=Path(args.output).parent / '.adapter_staging'
    staging.mkdir(parents=True,exist_ok=True)
    adapter_paths={}
    for name,path in [('original',Path(args.original)),('generated',Path(args.generated))]:
        if path.is_dir():
            adapter_paths[name]=str(path)
            continue
        if not path.is_file(): raise FileNotFoundError(path)
        folder=staging/name; folder.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,folder/'adapter_model.safetensors')
        sibling=path.parent/'adapter_config.json'
        if sibling.exists():
            shutil.copyfile(sibling,folder/'adapter_config.json')
        else:
            (folder/'adapter_config.json').write_text(json.dumps(dict(
                peft_type='LORA',task_type='CAUSAL_LM',r=8,lora_alpha=16,
                lora_dropout=0.0,bias='none',target_modules=['down_proj','v_proj','gate_proj','o_proj','up_proj','k_proj','q_proj'],
                base_model_name_or_path=args.base_model),indent=2)+'\n')
        adapter_paths[name]=str(folder)
    llm=LLM(model=args.base_model,enable_lora=True,max_lora_rank=8,max_loras=2,
            max_cpu_loras=2,gpu_memory_utilization=.25,dtype='bfloat16',seed=999)
    sampling=SamplingParams(temperature=0,max_tokens=args.max_new_tokens)
    outputs={}
    for name,path in [('original',args.original),('generated',args.generated)]:
        result=llm.generate(prompts,sampling,lora_request=LoRARequest(name,1,adapter_paths[name]))
        predictions=[]
        for i,out in enumerate(result):
            text=out.outputs[0].text; answer=parse_answer(text,args.task); label=parse_answer(rows[i].get('response',''),args.task)
            predictions.append(dict(index=i,predict=text,answer=answer,label=label,correct=answer==label,
                                    finish_reason=out.outputs[0].finish_reason))
        outputs[name]=predictions
    n=len(rows); original=sum(x['correct'] for x in outputs['original'])/n; generated=sum(x['correct'] for x in outputs['generated'])/n
    report=dict(task=args.task,n=n,original_accuracy=original,generated_accuracy=generated,
                delta_percentage_points=100*(generated-original),
                answer_agreement=sum(a['answer']==b['answer'] for a,b in zip(outputs['original'],outputs['generated']))/n,
                original_invalid=sum(x['answer'] is None for x in outputs['original']),
                generated_invalid=sum(x['answer'] is None for x in outputs['generated']),
                protocol='same base model, chat template, greedy decoding, and answer parser')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(dict(report=report,predictions=outputs),indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
