import os

SEED = 999
import json
import os
import sys

from pathlib import Path
root = str(Path(__file__).resolve().parents[4])
sys.path.append(root)
os.chdir(root)
with open("./workspace/main/config.json", "r") as f:
    workspace_config = json.load(f)
USE_WANDB = workspace_config["use_wandb"]
if USE_WANDB:
    import wandb

    if workspace_config.get("wandb_api_key"):
        os.environ["WANDB_API_KEY"] = workspace_config["wandb_api_key"]

# torch
import torch

torch.set_float32_matmul_precision("high")
import accelerate.utils
from accelerate import Accelerator
from bitsandbytes.optim import AdamW8bit as Optimizer
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

accelerate.utils.set_seed(SEED)
dataset_tag = "ARC-c"

DATASET_ROOT = os.environ.get("DND_DATASET_ROOT", str(Path(root).parent / "Loradatasets/common_sense_reasoning"))
CONFIG_ROOT = f"./workspace/datasets/common_sense_reasoning"
COND_ROOT = "./prepare/data"
extractor = "./models/all-MiniLM-L12-v2"

from workspace.dnd.dataset import Text2Qwen25LoRA_FullCondDataset as Dataset
from workspace.dnd.model import HyperConvDecoderModel_FullCond as Model
from workspace.dnd.module import WarmupScheduler
from workspace.dnd.tokenizer import Qwen2505LoRA_Tokenizer2D as Tokenizer
from workspace.dnd.tools import calculate_mean_criterion_weight, start_monitor

datasets = ["ARC-e", "BoolQ",  "PIQA", "HellaSwag"]


micro_batch_size = int(os.environ.get("DND_MICRO_BATCH_SIZE", "64"))
global_batch_size = int(os.environ.get("DND_BATCH_SIZE", "64"))
num_processes = int(os.environ.get("NUM_PROCESSES", "1"))
if global_batch_size % (micro_batch_size * num_processes):
    raise ValueError(f"Microbatch times process count must divide {global_batch_size}")
accumulation_steps = global_batch_size // (micro_batch_size * num_processes)
accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=accumulation_steps)
max_text_length = 384
config: dict[str, [float, int, str, dict]] = {
    # global setting
    "seed": SEED,
    "model_tag": os.path.basename(__file__)[:-3].split("_")[1],
    "need_test": False,
    "use_wandb": True,
    # data setting
    "token_size": (10, 130),
    "real_length": 50,
    "train_checkpoint_folders": [f"{DATASET_ROOT}/{dataset}" for dataset in datasets],
    "test_checkpoint_folder": "",
    "dataset_tag": dataset_tag,
    "generated_file": f"{CONFIG_ROOT}/{dataset_tag}/",
    # train setting
    "max_num_gpus": num_processes,
    "batch_size": global_batch_size,
    "num_workers": int(os.environ.get("DND_NUM_WORKERS", "4")),
    "prefetch_factor": 1,
    "warmup_steps": 1,
    "total_steps": int(os.environ.get("DND_TOTAL_STEPS", "4000")),
    "micro_batch_size": micro_batch_size,
    "effective_batch_size": global_batch_size,
    "gradient_accumulation_steps": accumulation_steps,
    "learning_rate": 3e-5,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "save_every": 100,
    "print_every": 20,
    "num_texts": 128,
    "save_folder": "./checkpoints",
    "noise_enhance": 0.0001,
    "criterion_weight": calculate_mean_criterion_weight(
        [f"{CONFIG_ROOT}/{dataset}/criterion_weight.pt" for dataset in datasets]
    ),
    "extractor_type": "BERT",
    "text_tokenizer": AutoTokenizer.from_pretrained(extractor),
    "extra_condition_module": AutoModel.from_pretrained(extractor, torch_dtype="auto").to(accelerator.device),
    "max_text_length": max_text_length,
    "model_config": {
        "features": [
            (128, max_text_length, 384),
            (128, 200, 300),
            (128, 100, 256),
            (256, 50, 200),
            (512, 50, 200),
            (1024, 25, 200),
            (1024, 10, 200),
            (2048, 10, 200),
            (4296, 10, 130),
        ],
        "condition_dim": (128, max_text_length, 384),
        "kernel_size": 9,
    },
}

# Data
if accelerator.is_main_process:
    print("==> Preparing tokenizer and dataset...")
tokenizer = Tokenizer(token_size=config["token_size"])
Dataset.dtype = torch.bfloat16
expected_iteration = config["total_steps"] * config["batch_size"] * config["max_num_gpus"]


train_set = Dataset(
    checkpoint_folders=config["train_checkpoint_folders"],
    tokenizer=tokenizer,
    expected_iteration=expected_iteration,
    num_texts=config["num_texts"],
    text_tokenizer=config["text_tokenizer"],
    max_text_length=config["max_text_length"],
    real_length=config["real_length"],
    texts=[json.load(open(f"{COND_ROOT}/{dataset}_train.json", "r", encoding="utf-8")) for dataset in datasets],
)  # train set


test_set = Dataset(
    checkpoint_folders=[f"{DATASET_ROOT}/ARC-c"],
    tokenizer=tokenizer,
    expected_iteration=expected_iteration,
    num_texts=config["num_texts"],
    text_tokenizer=config["text_tokenizer"],
    max_text_length=config["max_text_length"],
    real_length=config["real_length"],
    texts=[json.load(open(f"{COND_ROOT}/ARC-c_train.json", "r", encoding="utf-8"))],
)


# process dataloader
config["batch_size"] = micro_batch_size
if accelerator.is_main_process:
    print(f"microbatch:{micro_batch_size}; accumulation:{accumulation_steps}; effective batch:{global_batch_size}")
train_loader = DataLoader(
    dataset=train_set,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    prefetch_factor=config["prefetch_factor"],
    collate_fn=train_set.collate_fn_train,
    drop_last=True,
    shuffle=True,
)  # train dataloader
eval_loader = DataLoader(
    dataset=train_set,
    batch_size=1,
    num_workers=0,
    collate_fn=train_set.collate_fn_test,
    shuffle=False,
)  # eval dataloader
eval_iterator = iter(eval_loader)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=1,
    num_workers=0,
    collate_fn=test_set.collate_fn_test,
    shuffle=False,
)  # test dataloader
test_iterator = iter(test_loader)


model = Model(
    config=config["model_config"],
    criterion_weight=config["criterion_weight"].view(1, -1, 1, 1),
    extractor_type=config["extractor_type"],
    extra_condition_module=config["extra_condition_module"],
)

# Optimizer
if accelerator.is_main_process:
    print("==> Building optimizer and scheduler...")
optimizer = Optimizer(
    params=model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)  # use 8bit optimizer
scheduler = WarmupScheduler(
    optimizer=optimizer,
    warmup_steps=config["warmup_steps"],
    total_steps=config["total_steps"],
)  # use cosine annealing


# Accelerator
if __name__ == "__main__":
    # accelerator
    # from accelerate.utils import DistributedDataParallelKwargs
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator(kwargs_handlers=[kwargs,],)

    # noinspection PyTypeChecker
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)  # define everything


# Wandb
config["tag"] = config["model_tag"] + "__" + config["dataset_tag"]
# noinspection PyUnboundLocalVariable
if __name__ == "__main__" and USE_WANDB and accelerator.is_main_process:
    wandb.login()
    wandb.init(
        project="DnD",
        name=config["tag"],
        config=config,
    )
    if os.environ.get("DND_GPU_MONITOR") == "1":
        start_monitor(second=20)


# Train
def train():
    print("==> Training...", flush=True)
    model.train()
    optimizer.zero_grad()
    update_step = 0
    accumulated_loss = 0.0
    for tokens, cond_id, cond_mask in train_loader:
        with accelerator.accumulate(model):
            conditions = {"input_ids": cond_id.to(accelerator.device), "attention_mask": cond_mask.to(accelerator.device)}
            tokens = tokens.to(accelerator.device)
            mask = ~torch.isnan(tokens)
            tokens = torch.nan_to_num_(tokens, nan=0.0)
            with accelerator.autocast():
                loss = model(source=None, mask=mask, condition=conditions, target=tokens,
                             noise_enhance=config.get("noise_enhance"))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at update {update_step + 1}")
            accumulated_loss += loss.detach().float().item() / accumulation_steps
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()
        if not accelerator.sync_gradients:
            continue
        scheduler.step(update_step)
        update_step += 1
        if accelerator.is_main_process:
            metrics = {"train_loss": accumulated_loss, "learning_rate": optimizer.param_groups[0]["lr"], "optimizer_step": update_step}
            if USE_WANDB:
                wandb.log(metrics, step=update_step)
            print(f"step:{update_step}/{config['total_steps']} Loss:{accumulated_loss:.6f}", flush=True)
            if update_step % config["save_every"] == 0 or update_step == config["total_steps"]:
                os.makedirs(config["save_folder"], exist_ok=True)
                state = {k: v for k, v in accelerator.get_state_dict(model).items() if not k.startswith("condition_module")}
                accelerator.save(state, os.path.join(config["save_folder"], config["tag"] + ".pth"))
                if update_step % 1000 == 0:
                    accelerator.save(state, os.path.join(config["save_folder"], config["tag"] + f"{update_step}.pth"))
                del state
                generate(iterator=eval_iterator, idx=update_step // config["save_every"])
                generate(iterator=test_iterator, idx=update_step // config["save_every"])
                torch.cuda.empty_cache()
        accumulated_loss = 0.0
        if update_step >= config["total_steps"]:
            break
    if USE_WANDB and accelerator.is_main_process:
        wandb.finish()


# Generate
def generate(iterator, idx):
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        print("==> Generating...")
    model.eval()
    # prepare data
    tokens, cond_id, cond_mask, tag = next(iterator)
    conditions = {"input_ids": cond_id.to(accelerator.device), "attention_mask": cond_mask.to(accelerator.device)}
    # generate
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        mask = ~torch.isnan(tokens)
        tokens = torch.nan_to_num_(tokens, nan=0.0)
        predict = accelerator.unwrap_model(model).generate(
            source=None,
            mask=mask.to(accelerator.device),
            condition=conditions,
            target=None,
            # generate=True,
        )  # generate
    # save and log
    generated_norm = torch.square(predict[mask]).mean().item()
    original_norm = torch.square(tokens[mask]).mean().item()
    if accelerator.is_main_process:
        print("generated_start:", predict.flatten()[0:5].tolist())
        print("original_start:", tokens.flatten()[0:5].tolist())
        print("generated_end:", predict[0, -1, 0, 0:5].tolist())
        print("original_end:", tokens[0, -1, 0, 0:5].tolist())
        print("generated_l2norm:", generated_norm)
        print("original_l2norm:", original_norm)
    if USE_WANDB:
        wandb.log(
            {
                "generated_l2norm": generated_norm,
                "original_l2norm:": original_norm,
            }
        )  # log generated info
    # noinspection PyTypeChecker save checkpoint
    _ = train_set.save_checkpoint(
        save_path=config["generated_file"], tokens=predict[0], tag=tag, number=idx
    )  # save files
    torch.cuda.empty_cache()
    model.train()
    return predict


if __name__ == "__main__":
    train()
