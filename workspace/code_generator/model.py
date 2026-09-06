"""GPT-2-style pre-norm causal decoder with continuous prefix embeddings."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(approximate='tanh'),
                                 nn.Linear(4 * width, width))

    def forward(self, x, past=None, cache=False):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if past is not None:
            if n != 1:
                raise ValueError('Cached decoding accepts one new token at a time')
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=past is None)
        x = x + self.out(a.transpose(1, 2).reshape(b, n, d))
        x = x + self.mlp(self.norm2(x))
        return x, (k, v) if cache else None


class PrefixCodeGPT(nn.Module):
    def __init__(self, codebook_size=1024, references=5, code_length=2560, prompt_dim=384,
                 num_prompts=128, width=384, layers=6, heads=6, step_scale=300):
        super().__init__()
        if width % heads or min(codebook_size, references, code_length, num_prompts, step_scale) < 1:
            raise ValueError('Invalid model dimensions')
        self.config = dict(codebook_size=codebook_size, references=references, code_length=code_length,
                           prompt_dim=prompt_dim, num_prompts=num_prompts, width=width,
                           layers=layers, heads=heads, step_scale=step_scale)
        self.bos = codebook_size + references
        self.token = nn.Embedding(self.bos + 1, width)
        self.position = nn.Embedding(num_prompts + 2 + code_length, width)
        self.prompt = nn.Linear(prompt_dim, width)
        self.step = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=.02)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

    def prefix(self, prompts, steps):
        if prompts.shape[1:] != (self.config['num_prompts'], self.config['prompt_dim']):
            raise ValueError('Incorrect prompt embedding shape')
        if not torch.isfinite(steps).all() or (steps < 0).any():
            raise ValueError('Checkpoint steps must be finite and nonnegative')
        s = steps.float() / self.config['step_scale']
        features = torch.stack((s, torch.log1p(steps.float()) / math.log1p(self.config['step_scale']),
                                torch.ones_like(s)), dim=-1)
        return torch.cat((self.prompt(prompts), self.step(features)[:, None]), dim=1)

    def hidden(self, x, past=None, offset=0, cache=False):
        x = x + self.position(torch.arange(offset, offset + x.shape[1], device=x.device))[None]
        states = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, None if past is None else past[i], cache)
            states.append(kv)
        return self.norm(x), states

    def forward(self, prompts, steps, targets):
        """targets: raw reference ID then raw VQ IDs. Shift internally; never expose future labels."""
        k, r = self.config['codebook_size'], self.config['references']
        if targets.shape != (len(prompts), self.config['code_length'] + 1):
            raise ValueError('Incorrect target sequence shape')
        if (targets[:, 0] < 0).any() or (targets[:, 0] >= r).any() or (targets[:, 1:] < 0).any() or (targets[:, 1:] >= k).any():
            raise ValueError('Invalid reference or VQ index')
        encoded = targets.clone()
        encoded[:, 0] += k
        bos = torch.full((len(prompts), 1), self.bos, device=prompts.device, dtype=torch.long)
        prefix = self.prefix(prompts, steps)
        x = torch.cat((prefix, self.token(torch.cat((bos, encoded[:, :-1]), dim=1))), dim=1)
        h, _ = self.hidden(x)
        h = h[:, prefix.shape[1]:]
        # Separate valid alphabets; BOS is never a prediction target.
        ref_logits = F.linear(h[:, 0], self.token.weight[k:k+r])
        code_logits = F.linear(h[:, 1:], self.token.weight[:k])
        ref_loss = F.cross_entropy(ref_logits.float(), targets[:, 0])
        code_loss = F.cross_entropy(code_logits.float().reshape(-1, k), targets[:, 1:].reshape(-1))
        loss = code_loss + ref_loss
        metrics = dict(loss=loss.detach(), code_ce=code_loss.detach(), reference_ce=ref_loss.detach(),
                       code_accuracy=(code_logits.argmax(-1) == targets[:, 1:]).float().mean(),
                       reference_accuracy=(ref_logits.argmax(-1) == targets[:, 0]).float().mean())
        return loss, metrics

    @torch.no_grad()
    def generate(self, prompts, steps, temperature=0., top_k=0):
        if temperature < 0 or top_k < 0:
            raise ValueError('temperature and top_k must be nonnegative')
        self.eval()
        k, r = self.config['codebook_size'], self.config['references']
        bos = torch.full((len(prompts), 1), self.bos, device=prompts.device, dtype=torch.long)
        x = torch.cat((self.prefix(prompts, steps), self.token(bos)), dim=1)
        h, cache = self.hidden(x, cache=True)
        offset = x.shape[1]
        result = []
        for i in range(self.config['code_length'] + 1):
            weights = self.token.weight[k:k+r] if i == 0 else self.token.weight[:k]
            logits = F.linear(h[:, -1], weights).float()
            if temperature == 0:
                chosen = logits.argmax(-1)
            else:
                logits = logits / temperature
                if top_k:
                    cutoff = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float('-inf'))
                chosen = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
            result.append(chosen)
            if i < self.config['code_length']:
                token = chosen + k if i == 0 else chosen
                h, cache = self.hidden(self.token(token[:, None]), past=cache, offset=offset, cache=True)
                offset += 1
        return torch.stack(result, dim=1)
