"""Evaluate the pretrained DnD generator on every common-sense task.

Three stages, each restartable:

  prepare   generate N LoRA adapters per task with the pretrained DnD checkpoint
            and stage the last N original LoRA checkpoints of the same task
  evaluate  score every adapter on the task benchmark with one vLLM engine
  report    render the markdown and summary tables

The generator, its conditioning and its token format follow
workspace/main/generate/qwen0.5lora_generation_common_sense_reasoning.py.
The answer parser and the PIQA label source are shared with
workspace/vqvae/evaluate_downstream.py so both reports score identically.
"""
import argparse
import json
import os
import shutil
import statistics
from pathlib import Path

from workspace.vqvae.evaluate_downstream import TASKS, answer, evaluation_data

SEED = 999
# Only the tensor shapes of this folder are used: it supplies the key names and
# shapes that detokenization writes the generated values into.
TEMPLATE_TASK = "ARC-e"
# The importance weights are a loss-only buffer; the generation script loads BoolQ's.
CRITERION_TASK = "BoolQ"
NUM_TEXTS = 128
MAX_TEXT_LENGTH = 384
TOKEN_SIZE = (10, 130)
# Mirrors the config in the generation script; a mismatch is caught when the
# checkpoint is loaded, because loading requires every generator key to be present.
MODEL_CONFIG = {
    "features": [
        (128, MAX_TEXT_LENGTH, 384),
        (128, 200, 300),
        (128, 100, 256),
        (256, 50, 200),
        (512, 50, 200),
        (1024, 25, 200),
        (1024, 10, 200),
        (2048, 10, 200),
        (4296, 10, 130),
    ],
    "condition_dim": (128, MAX_TEXT_LENGTH, 384),
    "kernel_size": 9,
}


def repository_root():
    return Path(__file__).resolve().parents[2]


def resolve_asset(explicit, *relative):
    """Assets live either in this repository or in the nested upstream checkout."""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path.resolve()
    root = repository_root()
    candidates = [root / r for r in relative] + [root / "Drag-and-Drop-LLMs" / r for r in relative]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise FileNotFoundError(f"None of these exist: {[str(p) for p in candidates]}")
    return found.resolve()


def select_originals(folder, count):
    """The last `count` original LoRA checkpoints, by training step in the filename."""
    files = [p for p in Path(folder).glob("*.safetensors") if p.stem.isdigit()]
    if len(files) < count:
        raise ValueError(f"Need {count} numbered checkpoints in {folder}, found {len(files)}")
    return sorted(files, key=lambda p: int(p.stem))[-count:]


def training_selection(data_root, settings):
    """The checkpoint paths DnD trained on, in the frozen os.listdir order it used."""
    selection = {}
    for dataset in settings["datasets"]:
        folder = Path(data_root) / dataset
        names = os.listdir(folder)[: settings["real_length"]]
        if len(names) != settings["real_length"]:
            raise ValueError(f"DnD trained on {settings['real_length']} checkpoints from {folder}, found {len(names)}")
        selection[dataset] = {str((folder / name).resolve()) for name in names}
    return selection


def membership(task, path, settings, selection):
    """How a checkpoint relates to what the DnD generator saw during training."""
    if task == settings["dataset_tag"]:
        return "held_out_task"
    if task not in settings["datasets"]:
        return "unseen_task"
    return "train" if str(Path(path).resolve()) in selection[task] else "unseen_checkpoint"


def spread(values):
    values = list(values)
    return dict(
        mean=statistics.fmean(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        min=min(values),
        max=max(values),
    )


def write_adapter(folder, base_model):
    config = json.loads((repository_root() / "configs/Qwen0.5/adapter_config.json").read_text())
    config["base_model_name_or_path"] = str(base_model)
    (folder / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")


def weight_distance(generated, original):
    """Parameter-space distance between two adapters, over shared keys."""
    from workspace.vqvae.data import read_weights

    a, b = read_weights(generated, raw=True), read_weights(original, raw=True)
    if set(a) != set(b):
        raise ValueError(f"Adapter keys differ between {generated} and {original}")
    error = norm = other = dot = 0.0
    for key, value in b.items():
        x, y = value.double(), a[key].double()
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch for {key}: {x.shape} vs {y.shape}")
        error += (x - y).square().sum().item()
        norm += x.square().sum().item()
        other += y.square().sum().item()
        dot += (x * y).sum().item()
    return dict(relative_l2=(error / norm) ** 0.5, cosine=dot / (norm * other) ** 0.5)


def build_generator(checkpoint, extractor, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    from workspace.dnd.model import HyperConvDecoderModel_FullCond as Model
    from workspace.dnd.tokenizer import Qwen2505LoRA_Tokenizer2D as Tokenizer

    weights = repository_root() / "workspace/datasets/common_sense_reasoning" / CRITERION_TASK / "criterion_weight.pt"
    criterion = torch.load(weights, map_location="cpu", weights_only=True)
    model = Model(
        config=MODEL_CONFIG,
        criterion_weight=criterion.view(1, -1, 1, 1),
        extractor_type="BERT",
        extra_condition_module=AutoModel.from_pretrained(extractor, torch_dtype="auto"),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The frozen text encoder is excluded when the checkpoint is saved; nothing else may be.
    missing = [key for key in missing if not key.startswith("condition_module.")]
    if missing or unexpected:
        raise ValueError(
            f"{checkpoint} does not match the generator architecture: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    return model.to(device).eval(), Tokenizer(token_size=TOKEN_SIZE), AutoTokenizer.from_pretrained(extractor)


def generate_adapters(model, tokenizer, text_tokenizer, task, data_path, args, output):
    """Sample `args.samples` adapters for one task, each on its own random prompt subset."""
    import torch
    from torch.utils.data import DataLoader

    from workspace.dnd.dataset import Text2Qwen25LoRA_FullCondDataset as Dataset

    dataset = Dataset(
        checkpoint_folders=[str(Path(args.data_root) / TEMPLATE_TASK)],
        tokenizer=tokenizer,
        expected_iteration=None,
        real_length=args.samples,
        texts=[json.loads(Path(data_path).read_text())],
        num_texts=NUM_TEXTS,
        text_tokenizer=text_tokenizer,
        max_text_length=MAX_TEXT_LENGTH,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=dataset.collate_fn_test, shuffle=False)
    staging = output / task / ".generated"
    variants = []
    for index, (tokens, cond_id, cond_mask, tag) in enumerate(loader):
        with torch.no_grad(), torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16):
            mask = ~torch.isnan(tokens)
            predict = model.generate(
                source=None,
                mask=mask.to(args.device),
                condition={"input_ids": cond_id.to(args.device), "attention_mask": cond_mask.to(args.device)},
                target=None,
            )
        norm = torch.square(predict[mask.to(predict.device)]).mean().item()
        dataset.save_checkpoint(save_path=str(staging), tokens=predict[0], tag=tag, number=index)
        folder = output / task / f"dnd_{index}"
        folder.mkdir(parents=True, exist_ok=True)
        (staging / f"{index}.safetensors").replace(folder / "adapter_model.safetensors")
        write_adapter(folder, args.base_model)
        variants.append(dict(name=f"dnd_{index}", kind="dnd", sample=index, token_l2norm=norm,
                             token_template=str(Path(tag).resolve())))
        print(f"{task} dnd_{index}: generated token l2norm {norm:.6f}", flush=True)
        del predict
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if staging.exists():
        shutil.rmtree(staging)
    return variants


def prepare(args):
    import accelerate.utils
    import torch

    from workspace.vqvae.data import dnd_settings, sha256

    torch.set_float32_matmul_precision("high")
    accelerate.utils.set_seed(SEED)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    args.base_model = resolve_asset(args.base_model, "models/Qwen2.5-0.5B-Instruct")
    args.extractor = resolve_asset(args.extractor, "models/all-MiniLM-L12-v2")
    args.dnd_checkpoint = resolve_asset(args.dnd_checkpoint, "checkpoints/qwen0.5lora__ARC-c.pth")
    args.data_root = resolve_asset(args.data_root, "Loradatasets/common_sense_reasoning")

    settings = dnd_settings()
    selection = training_selection(args.data_root, settings)
    digest = sha256(args.dnd_checkpoint)
    manifest = output / "comparison.json"
    record = dict(
        dnd_checkpoint=str(args.dnd_checkpoint), dnd_sha256=digest,
        dnd_train_tasks=settings["datasets"], dnd_held_out_task=settings["dataset_tag"],
        base_model=str(args.base_model), extractor=str(args.extractor),
        data_root=str(args.data_root), samples=args.samples, originals=args.originals,
        condition_prompts=NUM_TEXTS, seed=SEED,
        selection="last originals by training step; DnD samples differ only in the sampled conditioning prompts",
        tasks=[],
    )
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["dnd_sha256"] != digest:
            raise ValueError(f"{manifest} was built from a different DnD checkpoint; use a fresh --output")
        for key in ("samples", "originals"):
            if previous[key] != record[key]:
                raise ValueError(f"{manifest} was built with --{key} {previous[key]}; use a fresh --output")
        record["tasks"] = [t for t in previous["tasks"] if t["task"] in args.tasks]

    done = {t["task"] for t in record["tasks"]}
    pending = [t for t in args.tasks if t not in done]
    if not pending:
        print(f"All requested tasks already prepared in {output}", flush=True)
        return
    model, tokenizer, text_tokenizer = build_generator(args.dnd_checkpoint, args.extractor, args.device)
    for task in pending:
        data_path = evaluation_data(task, repository_root(), output)
        prompts = json.loads(Path(data_path).read_text())
        variants = generate_adapters(model, tokenizer, text_tokenizer, task, data_path, args, output)
        for source in select_originals(Path(args.data_root) / task, args.originals):
            folder = output / task / f"original_{source.stem}"
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, folder / "adapter_model.safetensors")
            write_adapter(folder, args.base_model)
            variants.append(dict(name=folder.name, kind="original", step=int(source.stem),
                                 source=str(source), source_sha256=sha256(source),
                                 dnd_membership=membership(task, source, settings, selection)))
        reference = max((v for v in variants if v["kind"] == "original"), key=lambda v: v["step"])["name"]
        for variant in variants:
            if variant["kind"] == "dnd":
                variant.update(weight_distance(
                    output / task / variant["name"] / "adapter_model.safetensors",
                    output / task / reference / "adapter_model.safetensors",
                ))
        entry = dict(task=task, datafile=str(data_path), data_sha256=sha256(data_path), n=len(prompts),
                     condition_source=str(data_path), reference=reference, variants=variants)
        record["tasks"].append(entry)
        record["tasks"].sort(key=lambda t: args.tasks.index(t["task"]))
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Prepared {task}: {len(variants)} adapters over {len(prompts)} examples", flush=True)
    if sha256(args.dnd_checkpoint) != digest:
        raise ValueError("The DnD checkpoint changed while adapters were being generated")


def evaluate(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from workspace.vqvae.data import sha256

    output = Path(args.output)
    record = json.loads((output / "comparison.json").read_text())
    tasks = [t for t in record["tasks"] if t["task"] in args.tasks]
    if not tasks:
        raise ValueError(f"No prepared tasks among {args.tasks}; run prepare first")
    tokenizer = AutoTokenizer.from_pretrained(record["base_model"])
    engine = LLM(model=record["base_model"], dtype="bfloat16", tensor_parallel_size=1,
                 enable_lora=True, max_lora_rank=8, max_loras=1, max_cpu_loras=16,
                 max_model_len=4096, max_num_seqs=128, gpu_memory_utilization=args.gpu_memory_utilization,
                 enforce_eager=True, seed=SEED)
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=SEED)
    results = output / "results.json"
    report = json.loads(results.read_text()) if results.exists() else None
    if report is None or report.get("max_new_tokens") != args.max_new_tokens or report.get("limit") != args.limit:
        report = dict(protocol="Repository prompts, Qwen chat template, greedy generation, explicit answer extraction",
                      max_new_tokens=args.max_new_tokens, limit=args.limit,
                      dnd_checkpoint=record["dnd_checkpoint"], dnd_sha256=record["dnd_sha256"],
                      dnd_train_tasks=record["dnd_train_tasks"], dnd_held_out_task=record["dnd_held_out_task"],
                      samples=record["samples"], originals=record["originals"], tasks=[])
    scored = {t["task"] for t in report["tasks"]}
    request_id = 0
    for entry in tasks:
        task = entry["task"]
        if task in scored:
            print(f"{task} already scored in {results}", flush=True)
            continue
        if sha256(entry["datafile"]) != entry["data_sha256"]:
            raise ValueError(f"Evaluation data changed since prepare: {entry['datafile']}")
        samples = json.loads(Path(entry["datafile"]).read_text())
        if args.limit:
            samples = samples[: args.limit]
        prompts = []
        for sample in samples:
            messages = [{"role": "system", "content": sample["system"]}] if sample.get("system") else []
            messages.append({"role": "user", "content": sample["prompt"]})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if len(tokenizer.encode(prompt)) + args.max_new_tokens > 4096:
                raise ValueError(f"{task} prompt plus {args.max_new_tokens} new tokens exceeds the context limit")
            prompts.append(prompt)
        row = dict(entry, n=len(samples), variants=[])
        predictions = {}
        for variant in entry["variants"]:
            request_id += 1
            folder = output / task / variant["name"]
            destination = folder / "predictions.jsonl"
            if destination.exists():
                answers = [json.loads(line) for line in destination.read_text().splitlines()]
                if len(answers) != len(samples):
                    raise ValueError(f"{destination} holds {len(answers)} predictions, expected {len(samples)}")
            else:
                generated = engine.generate(prompts, sampling, lora_request=LoRARequest(
                    f"{task}_{variant['name']}", request_id, str(folder.resolve())))
                answers = []
                for index, (sample, result) in enumerate(zip(samples, generated)):
                    text = result.outputs[0].text
                    label = answer(sample["response"], task)
                    if label is None:
                        raise ValueError(f"{task} example {index} has no parseable label")
                    predicted = answer(text, task)
                    answers.append(dict(index=index, predict=text, answer=predicted, label=label,
                                        correct=predicted == label, finish_reason=result.outputs[0].finish_reason))
                temporary = destination.with_suffix(".tmp")
                temporary.write_text("".join(json.dumps(a) + "\n" for a in answers))
                temporary.replace(destination)
            predictions[variant["name"]] = answers
            scores = dict(variant, accuracy=sum(a["correct"] for a in answers) / len(answers),
                          invalid=sum(a["answer"] is None for a in answers),
                          truncated=sum(a["finish_reason"] == "length" for a in answers))
            row["variants"].append(scores)
            print(f"{task} {variant['name']}: accuracy {scores['accuracy']:.4f}", flush=True)
        row.update(summarize(row, predictions))
        report["tasks"].append(row)
        report["tasks"].sort(key=lambda t: [x["task"] for x in record["tasks"]].index(t["task"]))
        results.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({t["task"]: {v["name"]: v["accuracy"] for v in t["variants"]} for t in report["tasks"]},
                     indent=2), flush=True)


def agreement(left, right):
    return sum(a["answer"] == b["answer"] for a, b in zip(left, right)) / len(left)


def summarize(row, predictions):
    """Aggregate the DnD samples and the original checkpoints of one task."""
    groups = {kind: [v for v in row["variants"] if v["kind"] == kind] for kind in ("dnd", "original")}
    summary = {f"{kind}_{key}": value
               for kind, variants in groups.items()
               for key, value in spread(v["accuracy"] for v in variants).items()}
    summary["delta_percentage_points"] = 100 * (summary["dnd_mean"] - summary["original_mean"])
    reference = predictions[row["reference"]]
    summary["dnd_agreement_with_reference"] = statistics.fmean(
        agreement(predictions[v["name"]], reference) for v in groups["dnd"])
    others = [v for v in groups["original"] if v["name"] != row["reference"]]
    summary["original_agreement_with_reference"] = statistics.fmean(
        agreement(predictions[v["name"]], reference) for v in others) if others else 1.0
    summary["dnd_relative_l2"] = statistics.fmean(v["relative_l2"] for v in groups["dnd"])
    summary["dnd_cosine"] = statistics.fmean(v["cosine"] for v in groups["dnd"])
    return summary


def report(args):
    output = Path(args.output)
    result = json.loads((output / "results.json").read_text())
    lines = ["# Pretrained DnD generator on the common-sense tasks", ""]
    lines += [f"Generator checkpoint `{result['dnd_checkpoint']}`, trained on "
              f"{', '.join(result['dnd_train_tasks'])} with {result['dnd_held_out_task']} held out. "
              f"Per task: {result['samples']} DnD samples, each conditioned on a different random subset of "
              f"the task's evaluation prompts, against the last {result['originals']} original LoRA checkpoints. "
              f"{result['protocol']}, up to {result['max_new_tokens']} new tokens.", ""]
    header = (f"| Task | DnD sees | Examples | Last-{result['originals']} originals | "
              f"{result['samples']} DnD samples | Change (pp) | DnD agreement | Original agreement |")
    lines += [header, "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task in result["tasks"]:
        sees = {v["dnd_membership"] for v in task["variants"] if v["kind"] == "original"}
        lines.append(
            f"| {task['task']} | {'/'.join(sorted(sees))} | {task['n']} | "
            f"{100 * task['original_mean']:.2f}% ± {100 * task['original_std']:.2f} | "
            f"{100 * task['dnd_mean']:.2f}% ± {100 * task['dnd_std']:.2f} | "
            f"{task['delta_percentage_points']:+.2f} | "
            f"{100 * task['dnd_agreement_with_reference']:.2f}% | "
            f"{100 * task['original_agreement_with_reference']:.2f}% |")
    lines += ["", "Agreement is measured against the highest-step original checkpoint of the same task. "
                  "Original agreement is the same measurement between the remaining originals and that "
                  "reference, so it shows how much of the DnD disagreement is ordinary checkpoint noise.", ""]
    lines += ["| Task | Adapter | Kind | Accuracy | Invalid | Truncated | Weight relative L2 | Weight cosine |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for task in result["tasks"]:
        for variant in task["variants"]:
            distance = (f"{100 * variant['relative_l2']:.2f}% | {variant['cosine']:.4f}"
                        if variant["kind"] == "dnd" else "\u2014 | \u2014")
            lines.append(
                f"| {task['task']} | {variant['name']} | {variant.get('dnd_membership', 'generated')} | "
                f"{100 * variant['accuracy']:.2f}% | {variant['invalid']}/{task['n']} | "
                f"{variant['truncated']}/{task['n']} | {distance} |")
    lines += ["", "Weight distances compare each generated adapter with the reference original checkpoint.", ""]
    text = "\n".join(lines) + "\n"
    if args.markdown:
        Path(args.markdown).write_text(text)
        print(f"Wrote {args.markdown}", flush=True)
    else:
        print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "evaluate", "report"])
    parser.add_argument("--output", required=True, help="directory holding adapters, predictions and reports")
    parser.add_argument("--dnd-checkpoint", help="pretrained DnD generator (.pth)")
    parser.add_argument("--base-model", help="Qwen2.5-0.5B-Instruct")
    parser.add_argument("--extractor", help="all-MiniLM-L12-v2 condition encoder")
    parser.add_argument("--data-root", help="folder of per-task original LoRA checkpoints")
    parser.add_argument("--samples", type=int, default=5, help="DnD adapters generated per task")
    parser.add_argument("--originals", type=int, default=5, help="last original checkpoints per task")
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="score only the first N examples per task")
    parser.add_argument("--markdown", help="report: write the tables to this file")
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
