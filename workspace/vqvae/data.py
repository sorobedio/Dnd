"""Reuse DnD checkpoint selection, tensor ordering, dtype, and tokenization."""
import ast
import hashlib
import json
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from workspace.dnd.dataset.register import Text2Qwen25LoRA_CheckpointDataset as DnDReader
from workspace.dnd.tokenizer import Qwen2505LoRA_Tokenizer2D

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/main/tasks/common_sense_reasoning/train_qwen0.5lora_ARC-c.py"


def default_data_root():
    if os.environ.get("DND_DATASET_ROOT"):
        return Path(os.environ["DND_DATASET_ROOT"])
    candidates = [ROOT.parent / "Loradatasets/common_sense_reasoning",
                  ROOT / "Loradatasets/common_sense_reasoning"]
    return next((path for path in candidates if path.is_dir()), candidates[0])


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dnd_settings():
    tree = ast.parse(SOURCE.read_text())
    settings = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("datasets", "dataset_tag"):
                    settings[target.id] = ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "config":
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and key.value in ("real_length", "token_size"):
                    settings[key.value] = ast.literal_eval(value)
    return settings


def build_manifest(data_root):
    settings = dnd_settings()
    settings["token_size"] = list(settings["token_size"])
    data_root = Path(data_root).resolve()
    entries = []
    for split, datasets in [("train", settings["datasets"]), ("held_out", [settings["dataset_tag"]])]:
        for dataset in datasets:
            folder = data_root / dataset
            # DnD takes os.listdir(folder)[:50], without sorting. Freeze that order.
            selected = os.listdir(folder)[:settings["real_length"]]
            if len(selected) != settings["real_length"]:
                raise ValueError(f"Need {settings['real_length']} checkpoints in {folder}")
            for name in selected:
                path = folder / name
                if not path.is_file() or path.suffix != ".safetensors":
                    raise ValueError(f"DnD selection contains a non-checkpoint: {path}")
                entries.append(dict(split=split, dataset=dataset, path=str(path), sha256=sha256(path)))
    weight_paths = [ROOT / "workspace/datasets/common_sense_reasoning" / ds / "criterion_weight.pt"
                    for ds in settings["datasets"]]
    return dict(version=1, unit="complete_lora_checkpoint", token_shape=[4296, 10, 130],
                source_script=str(SOURCE), source_script_sha256=sha256(SOURCE),
                selection="os.listdir(folder)[:real_length], matching DnD at manifest creation",
                settings=settings, entries=entries,
                importance_weights=[dict(path=str(p), sha256=sha256(p)) for p in weight_paths])


def validate_manifest(manifest):
    for entry in manifest["entries"] + manifest["importance_weights"]:
        if sha256(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Source changed since manifest was created: {entry['path']}")


def importance_weights(manifest):
    weights = torch.stack([torch.load(p["path"], map_location="cpu", weights_only=True)
                           for p in manifest["importance_weights"]]).mean(0).float()
    if weights.shape != (4296,) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid importance weights")
    return weights


def read_weights(path, raw=False, dtype=torch.bfloat16):
    weights = {k: v.to(dtype) for k, v in load_file(str(path), device="cpu").items()}
    if not raw:
        weights = DnDReader.post_process(weights)
    key = DnDReader.sort_key_raw if raw else DnDReader.sort_key
    return OrderedDict(sorted(weights.items(), key=key))


class WholeLoRADataset(Dataset):
    def __init__(self, manifest, split, cache_dir=None):
        self.entries = [e for e in manifest["entries"] if e["split"] == split]
        self.dtype = torch.float32 if manifest.get("tokenization_dtype") == "float32" else torch.bfloat16
        self.tokenizer = Qwen2505LoRA_Tokenizer2D(token_size=(10, 130))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_id = hashlib.sha256(
            (ROOT / "workspace/dnd/tokenizer/tokenizer.py").read_bytes()
            + (ROOT / "workspace/dnd/tokenizer/register.py").read_bytes()
        ).hexdigest()[:12]
        if self.dtype == torch.float32:
            self.tokenizer_id += '_float32'
        if not self.entries:
            raise ValueError(f"No entries for split {split}")

    def __len__(self):
        return len(self.entries)

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        entry = self.entries[index]
        cache = (self.cache_dir / f"{entry['sha256']}_{self.tokenizer_id}.pt"
                 if self.cache_dir else None)
        if cache and cache.exists():
            return torch.load(cache, map_location="cpu", weights_only=True)
        tokens, _ = self.tokenizer.tokenize(read_weights(self.entries[index]["path"], dtype=self.dtype))
        if tuple(tokens.shape) != (4296, 10, 130) or torch.isinf(tokens).any():
            raise ValueError(f"Unexpected tokens for {self.entries[index]['path']}")
        if cache:
            temporary = cache.with_suffix(f".{os.getpid()}.tmp")
            torch.save(tokens, temporary)
            os.replace(temporary, cache)
        return tokens


def weight_schema(path):
    return [(key, list(value.shape)) for key, value in read_weights(path, raw=True).items()]


def unpack_tokens(tokens, schema):
    # Only key names and shapes are required; no original weight values are used.
    template = OrderedDict((key, torch.empty(shape)) for key, shape in schema)
    tokenizer = Qwen2505LoRA_Tokenizer2D(token_size=(10, 130))
    weights = tokenizer.detokenize(template, tokens.float().cpu())
    if not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Decoded adapter contains non-finite weights")
    return {key: value.contiguous() for key, value in weights.items()}
