"""Prepare step-labelled codes, train a prefix GPT, and generate decodable LoRA codes."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import torch

from .model import PrefixCodeGPT
ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def resolve_device(name):
    device = torch.device(name)
    return torch.device('cuda', 0) if device.type == 'cuda' and device.index is None else device


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()


def checkpoint_step(path):
    name = Path(path).stem
    if name.startswith('checkpoint-'):
        name = name[len('checkpoint-'):]
    if not name.isdecimal():
        raise ValueError(f'Cannot infer numeric checkpoint step: {path}')
    return int(name)


def default_step(task, latest, explicit=None):
    if explicit is not None:
        if explicit < 0:
            raise ValueError('Checkpoint step must be nonnegative')
        return explicit, 'explicit'
    if task in latest:
        return latest[task], 'latest_checkpoint_for_task'
    return max(latest.values()), 'latest_checkpoint_in_training_manifest'


def read_prompts(path):
    data = json.loads(Path(path).read_text())
    prompts = []
    for row in data:
        if isinstance(row, str):
            text = row
        elif 'prompt' in row:
            text = row['prompt']
        else:
            text = row['conversations'][0]['value']
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Empty/invalid prompt in {path}')
        prompts.append(text)
    if not prompts:
        raise ValueError('No prompts')
    return prompts


def encoder_hash(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError('Use a downloaded local MiniLM directory')
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in ('.json', '.txt', '.bin', '.safetensors'))
    return hashlib.sha256(''.join(p.name + sha256(p) for p in files).encode()).hexdigest()


@torch.no_grad()
def embed(prompts, encoder, device, batch_size=64, max_length=384):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder, local_files_only=True)
    model = AutoModel.from_pretrained(encoder, local_files_only=True).to(device).eval()
    rows = []
    for i in range(0, len(prompts), batch_size):
        tokens = tokenizer(prompts[i:i+batch_size], padding=True, truncation=True,
                           max_length=max_length, return_tensors='pt').to(device)
        with autocast(device):
            hidden = model(**tokens).last_hidden_state
        mask = tokens.attention_mask[..., None].float()
        pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
        rows.append(pooled.cpu())
        if i % (batch_size * 20) == 0:
            print(f'Embedded prompts {min(i+batch_size,len(prompts))}/{len(prompts)}', flush=True)
    return torch.cat(rows)


def prepare(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'dataset.pt').exists():
        raise FileExistsError('Prepared dataset exists; choose a new output directory')
    encoded = torch.load(args.codes, map_location='cpu', weights_only=True)
    if sha256(args.vq_model) != encoded['model_sha256']:
        raise ValueError('VQ model and code file do not match')
    vq = torch.load(args.vq_model, map_location='cpu', weights_only=True)
    references = vq['model_config'].get('reference_count', 0)
    if not references or encoded['codes'].ndim != 2:
        raise ValueError('This generator expects reference-residual VQ codes')
    vq_tasks = sorted({e['dataset'] for e in vq['manifest']['entries'] if e['split']=='train'})
    del vq
    selected = [i for i,e in enumerate(encoded['entries']) if e['dataset'] not in args.exclude_tasks]
    if not selected:
        raise ValueError('No training checkpoints remain')
    entries = [dict(encoded['entries'][i], checkpoint_step=checkpoint_step(encoded['entries'][i]['path']), code_row=i)
               for i in selected]
    codes = encoded['codes'][selected].long()
    k = encoded['codebook_size']
    if (codes[:,0]<0).any() or (codes[:,0]>=references).any() or (codes[:,1:]<0).any() or (codes[:,1:]>=k).any():
        raise ValueError('Invalid latent code indices')
    latest = {t:max(e['checkpoint_step'] for e in entries if e['dataset']==t)
              for t in sorted({e['dataset'] for e in entries})}
    # Filename-only catalogue preserves the requested generation default even
    # for tasks whose codes and prompts are excluded from generator training.
    available_steps = {}
    for entry in encoded['entries']:
        task = entry['dataset']
        available_steps[task] = max(available_steps.get(task, 0), checkpoint_step(entry['path']))
    metadata = dict(version=1, entries=entries, latest_steps=latest, codebook_size=k, references=references,
                    available_checkpoint_steps=available_steps,
                    code_length=codes.shape[1]-1, codes_sha256=sha256(args.codes),
                    vq_model=str(Path(args.vq_model).resolve()), vq_model_sha256=encoded['model_sha256'],
                    vq_training_tasks=vq_tasks, excluded_generator_tasks=args.exclude_tasks,
                    encoder=str(Path(args.encoder).resolve()), encoder_sha256=encoder_hash(args.encoder),
                    pooling='attention-mask mean of last_hidden_state; no L2 normalization',
                    max_text_length=384, num_prompts=args.num_prompts, prompt_sources={})
    features = {}
    for task in latest:
        source = Path(args.prompt_root)/f'{task}_train.json'
        prompts = read_prompts(source)
        if len(prompts) < args.num_prompts:
            raise ValueError(f'{task} has fewer than {args.num_prompts} prompts')
        metadata['prompt_sources'][task] = dict(path=str(source.resolve()), sha256=sha256(source), count=len(prompts))
        cache = out/f'prompts_{task}.pt'
        signature = dict(source=sha256(source), encoder=metadata['encoder_sha256'], pooling=metadata['pooling'], max_length=384)
        cached = torch.load(cache, weights_only=True) if cache.exists() else None
        if cached is not None and cached['signature'] == signature:
            features[task] = cached['embeddings']
        else:
            features[task] = embed(prompts, args.encoder, resolve_device(args.device))
            atomic_save(dict(signature=signature, embeddings=features[task]), cache)
    atomic_save(dict(metadata=metadata, codes=codes, prompts=features, schema=encoded['schema'],
                     token_shape=encoded['token_shape']), out/'dataset.pt')
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    with (out/'checkpoint_steps.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task', 'checkpoint_step', 'code_row', 'source_path', 'source_sha256'])
        writer.writerows((e['dataset'], e['checkpoint_step'], e['code_row'], e['path'], e['sha256']) for e in entries)
    print(json.dumps(dict(checkpoints=len(entries), latest_steps=latest, code_shape=list(codes.shape))), flush=True)


def batch(data, indices, rng, device):
    n = data['metadata']['num_prompts']
    entries = [data['metadata']['entries'][i] for i in indices]
    prompts = torch.stack([data['prompts'][e['dataset']][rng.sample(range(len(data['prompts'][e['dataset']])), n)]
                           for e in entries]).to(device)
    steps = torch.tensor([e['checkpoint_step'] for e in entries], device=device)
    return prompts, steps, data['codes'][indices].to(device)


@torch.no_grad()
def evaluate(model, data, device, batch_size):
    model.eval()
    rng = random.Random(1701)  # Same training-prompt subset on every checkpoint comparison.
    totals = {}
    for i in range(0, len(data['codes']), batch_size):
        indices = list(range(i,min(i+batch_size,len(data['codes']))))
        with autocast(device):
            _, metrics = model(*batch(data, indices, rng, device))
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.) + value.item()*len(indices)
    model.train()
    return {key:value/len(data['codes']) for key,value in totals.items()}


def train(args):
    if min(args.steps,args.batch_size,args.eval_every,args.save_every) < 1:
        raise ValueError('Training counts must be positive')
    out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not args.resume:
        raise FileExistsError('Use --resume or a new output directory')
    data = torch.load(args.dataset, map_location='cpu', weights_only=True)
    fingerprint = sha256(args.dataset)
    m = data['metadata']; device = resolve_device(args.device)
    torch.manual_seed(args.seed); rng = random.Random(args.seed)
    state = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    settings = {k:getattr(args,k) for k in ('steps','batch_size','learning_rate','seed','eval_every')}
    if state and (state['dataset_sha256']!=fingerprint or state['training']!=settings):
        raise ValueError('Resume requires identical data and training schedule')
    config = dict(codebook_size=m['codebook_size'], references=m['references'], code_length=m['code_length'],
                  num_prompts=m['num_prompts'], prompt_dim=next(iter(data['prompts'].values())).shape[-1],
                  width=args.width, layers=args.layers, heads=args.heads, step_scale=max(m['latest_steps'].values()))
    model = PrefixCodeGPT(**(state['model_config'] if state else config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    start, best, order, cursor = 0, float('inf'), [], 0
    run = None
    if state:
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start,best,order,cursor=state['step'],state['best_train_loss'],state['order'],state['cursor']
    if args.wandb:
        import wandb
        run=wandb.init(project='DnD-CodeGPT',config={**settings,**model.config,'tasks':list(m['latest_steps'])},
                       id=state.get('wandb_id') if state else None,resume='allow' if state else None,dir=str(out))
    if state:
        rng.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
    stop=False
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),config=model.config,training=settings)),flush=True)
    model.train()
    for step in range(start+1,args.steps+1):
        began=time.monotonic()
        if cursor>=len(order):
            order=list(range(len(data['codes'])));rng.shuffle(order);cursor=0
        indices=order[cursor:cursor+args.batch_size];cursor+=len(indices)
        warmup=min(100,max(1,args.steps//10))
        lr=args.learning_rate*min(step/warmup,1.)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,args.steps-warmup))))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with autocast(device):loss,metrics=model(*batch(data,indices,rng,device))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        row={k:v.item() for k,v in metrics.items()}
        row.update(step=step,learning_rate=lr,gradient_norm=norm.item(),step_seconds=time.monotonic()-began)
        improved=False
        stopping=stop or (args.stop_after and step>=args.stop_after)
        if step%args.eval_every==0 or step==args.steps or stopping:
            ev=evaluate(model,data,device,args.batch_size)
            row.update({'train_eval/'+k:v for k,v in ev.items()})
            improved=ev['loss']<best
            if improved:best=ev['loss']
        row['best_train_loss']=best if math.isfinite(best) else None
        if step==1 or step%10==0 or improved or stopping:print(json.dumps(row),flush=True)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if run:run.log(row,step=step)
        if improved or step%args.save_every==0 or step==args.steps or stopping:
            checkpoint=dict(model=model.state_dict(),model_config=model.config,optimizer=optimizer.state_dict(),
                            step=step,best_train_loss=best,metadata=m,schema=data['schema'],token_shape=data['token_shape'],
                            dataset_sha256=fingerprint,training=settings,order=order,cursor=cursor,
                            python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],wandb_id=run.id if run else None)
            atomic_save(checkpoint,out/'last.pt')
            if improved:atomic_save(checkpoint,out/'best_train.pt')
        if stopping:break
    if run:run.finish()


@torch.no_grad()
def generate(args):
    if not args.task or Path(args.task).name!=args.task or args.task in ('.','..'):
        raise ValueError('Task must be a simple nonempty name')
    if Path(args.output).exists():raise FileExistsError(args.output)
    state=torch.load(args.model,map_location='cpu',weights_only=True);m=state['metadata']
    encoder=args.encoder or m['encoder']
    if encoder_hash(encoder)!=m['encoder_sha256']:raise ValueError('Prompt encoder differs from training')
    prompts=read_prompts(args.prompts)
    n=m['num_prompts'];rng=random.Random(args.seed);torch.manual_seed(args.seed)
    # Inference can accept a handful of prompts; repeat to fill the fixed prefix if necessary.
    indices=rng.sample(range(len(prompts)),n) if len(prompts)>=n else [i%len(prompts) for i in range(n)]
    device=resolve_device(args.device)
    embeddings=embed([prompts[i] for i in indices],encoder,device)[None].to(device)
    step,source=default_step(args.task,m.get('available_checkpoint_steps',m['latest_steps']),args.checkpoint_step)
    model=PrefixCodeGPT(**state['model_config']).to(device);model.load_state_dict(state['model']);model.eval()
    with autocast(device):codes=model.generate(embeddings,torch.tensor([step],device=device),args.temperature,args.top_k)
    atomic_save(dict(version=1,codes=codes.cpu().to(torch.int32),schema=state['schema'],token_shape=state['token_shape'],
                     entries=[dict(dataset=args.task,path=f'{step}.safetensors',checkpoint_step=step,generated=True)],
                     model_sha256=m['vq_model_sha256'],codebook_size=m['codebook_size'],
                     generation=dict(generator_sha256=sha256(args.model),generator_step=state['step'],
                                     checkpoint_step=step,step_source=source,prompt_sha256=sha256(args.prompts),
                                     prompt_indices=indices,seed=args.seed,temperature=args.temperature,top_k=args.top_k)),args.output)
    print(json.dumps(dict(output=args.output,checkpoint_step=step,step_source=source,codes=list(codes.shape))),flush=True)
    if args.decode_dir:
        from workspace.vqvae.run import decode
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
        decode(argparse.Namespace(model=args.vq_model or m['vq_model'],codes=args.output,
                                  output_dir=args.decode_dir,limit=None,device=args.device))


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--codes',default=str(ROOT/'outputs/vqvae_train_arc_c_residual/train_codes.pt'))
    prep.add_argument('--vq-model',default=str(ROOT/'outputs/vqvae_train_arc_c_residual/best_train_reconstruction.pt'))
    prep.add_argument('--encoder',default=str(ROOT/'models/all-MiniLM-L12-v2'))
    prep.add_argument('--prompt-root',default=str(ROOT/'prepare/data'))
    prep.add_argument('--output-dir',default=str(ROOT/'outputs/code_generator_data'))
    prep.add_argument('--exclude-tasks',nargs='*',default=[])
    prep.add_argument('--num-prompts',type=int,default=128)
    tr=sub.add_parser('train')
    tr.add_argument('--dataset',default=str(ROOT/'outputs/code_generator_data/dataset.pt'))
    tr.add_argument('--output-dir',default=str(ROOT/'outputs/code_generator'))
    for key,default in [('steps',3000),('batch-size',4),('width',384),('layers',6),('heads',6),('eval-every',250),('save-every',100),('seed',999),('stop-after',0)]:
        tr.add_argument('--'+key,type=int,default=default)
    tr.add_argument('--learning-rate',type=float,default=2e-4);tr.add_argument('--resume');tr.add_argument('--wandb',action='store_true')
    gen=sub.add_parser('generate');gen.add_argument('--model',required=True);gen.add_argument('--prompts',required=True)
    gen.add_argument('--task',required=True);gen.add_argument('--checkpoint-step',type=int);gen.add_argument('--output',required=True)
    gen.add_argument('--encoder');gen.add_argument('--decode-dir');gen.add_argument('--vq-model')
    gen.add_argument('--temperature',type=float,default=0.);gen.add_argument('--top-k',type=int,default=0);gen.add_argument('--seed',type=int,default=999)
    for cmd in (prep,tr,gen):cmd.add_argument('--device',default='cuda:0')
    args=p.parse_args();torch.set_num_threads(8);globals()[args.command](args)


if __name__=='__main__':main()
