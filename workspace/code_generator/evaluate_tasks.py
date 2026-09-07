"""Evaluate the prefix-GPT code generator on every common-sense task.

Generates adapters for each task with workspace.code_generator.run, decodes them
through the VQ-VAE, and scores them downstream. Scoring, prompt selection and the
answer parser are the shared ones from workspace.dnd_downstream.evaluate, so these
numbers are directly comparable with the DnD generator's.

Samples differ only in the conditioning prompts drawn from the task, matching how
the DnD run varies its samples; decoding stays greedy unless asked otherwise.
"""
import argparse
import json
import shutil
from pathlib import Path

from workspace.dnd_downstream.evaluate import (SCHEMA, TASKS, evaluate, evaluation_data, repository_root,
                                               report, resolve_asset, reusable, select_originals,
                                               task_membership, weight_distance, write_adapter)

SEED = 999
GENERATOR = "code"
DEFAULT_MODEL = "outputs/code_generator_holdout_arc_c/best_train.pt"


def generator_metadata(model):
    import torch

    state = torch.load(model, map_location="cpu", weights_only=True)
    metadata = state["metadata"]
    return dict(
        train_tasks=sorted(set(metadata["latest_steps"]) - set(metadata.get("excluded_generator_tasks", []))),
        held_out_tasks=sorted(metadata.get("excluded_generator_tasks", [])),
        known_steps=metadata.get("available_checkpoint_steps", metadata["latest_steps"]),
        vq_model=metadata["vq_model"],
        encoder=metadata["encoder"],
        num_prompts=metadata["num_prompts"],
        generator_step=state["step"],
    )


def prompt_file(task, split, output):
    """Which prompts condition the generator: the evaluation set, or the training set it saw."""
    if split == "train":
        return repository_root() / "prepare/data" / f"{task}_train.json"
    return Path(evaluation_data(task, repository_root(), output))


def generate_adapters(task, args, output):
    """One adapter per seed: run.generate writes codes, then the VQ decoder writes weights."""
    from workspace.code_generator.run import generate as run_generate

    prompts = prompt_file(task, args.prompt_split, output)
    if not Path(prompts).exists():
        raise FileNotFoundError(prompts)
    variants = []
    for index in range(args.samples):
        folder = output / task / f"code_{index}"
        staging = output / task / f".codes_{index}"
        codes = staging / "codes.pt"
        if not (folder / "adapter_model.safetensors").exists():
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            run_generate(argparse.Namespace(
                model=str(args.generator_checkpoint), prompts=str(prompts), task=task,
                checkpoint_step=args.checkpoint_step, output=str(codes), encoder=None,
                decode_dir=str(staging / "decoded"), vq_model=args.vq_model,
                temperature=args.temperature, top_k=args.top_k, seed=SEED + index, device=args.device))
            decoded = next((staging / "decoded").glob(f"{task}/*/adapter_model.safetensors"))
            folder.mkdir(parents=True, exist_ok=True)
            decoded.replace(folder / "adapter_model.safetensors")
            write_adapter(folder, args.base_model)
            import torch

            generation = torch.load(codes, map_location="cpu", weights_only=True)["generation"]
            (folder / "generation.json").write_text(json.dumps(generation, indent=2) + "\n")
            shutil.rmtree(staging)
        # Written by the generation above, or by an earlier interrupted run.
        generation = json.loads((folder / "generation.json").read_text())
        variants.append(dict(name=folder.name, kind=GENERATOR, sample=index, seed=generation["seed"],
                             temperature=generation["temperature"], top_k=generation["top_k"],
                             checkpoint_step=generation["checkpoint_step"],
                             step_source=generation["step_source"]))
        print(f"{task} {folder.name}: adapter ready", flush=True)
    return variants


def prepare(args):
    from workspace.vqvae.data import dnd_settings, sha256

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    args.base_model = resolve_asset(args.base_model, "models/Qwen2.5-0.5B-Instruct")
    args.generator_checkpoint = resolve_asset(args.generator_checkpoint, DEFAULT_MODEL)
    args.data_root = resolve_asset(args.data_root, "Loradatasets/common_sense_reasoning")
    info = generator_metadata(args.generator_checkpoint)
    digest = sha256(args.generator_checkpoint)
    manifest = output / "comparison.json"
    record = dict(
        schema=SCHEMA, generator=GENERATOR, generator_checkpoint=str(args.generator_checkpoint),
        generator_sha256=digest, generator_train_tasks=info["train_tasks"],
        generator_held_out_task=", ".join(info["held_out_tasks"]) or "none",
        generator_step=info["generator_step"], vq_model=info["vq_model"], encoder=info["encoder"],
        base_model=str(args.base_model), data_root=str(args.data_root),
        samples=args.samples, originals=args.originals, condition_prompts=info["num_prompts"], seed=SEED,
        selection=f"greedy code decoding at temperature {args.temperature} and top_k {args.top_k}; "
                  f"samples differ only in the conditioning prompts",
        prompt_split=args.prompt_split, tasks=[],
    )
    record.update(reusable(manifest, record, args.tasks))
    pending = [t for t in args.tasks if t not in {entry["task"] for entry in record["tasks"]}]
    if not pending:
        print(f"All requested tasks already prepared in {output}", flush=True)
        return
    settings = dnd_settings()
    for task in pending:
        datafile = evaluation_data(task, repository_root(), output)
        examples = json.loads(Path(datafile).read_text())
        variants = generate_adapters(task, args, output)
        for source in select_originals(Path(args.data_root) / task, args.originals):
            folder = output / task / f"original_{source.stem}"
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, folder / "adapter_model.safetensors")
            write_adapter(folder, args.base_model)
            variants.append(dict(name=folder.name, kind="original", step=int(source.stem),
                                 source=str(source), source_sha256=sha256(source),
                                 membership="staged_original"))
        staged = [v for v in variants if v["kind"] == "original"]
        reference = max(staged, key=lambda v: v["step"])["name"] if staged else None
        for variant in variants:
            if variant["kind"] != "original" and reference:
                variant.update(weight_distance(
                    output / task / variant["name"] / "adapter_model.safetensors",
                    output / task / reference / "adapter_model.safetensors",
                ))
        record["tasks"].append(dict(
            task=task, datafile=str(datafile), data_sha256=sha256(datafile), n=len(examples),
            membership=task_membership(task, info["train_tasks"], info["held_out_tasks"]),
            vq_membership=task_membership(task, settings["datasets"], settings["dataset_tag"]),
            condition_source=str(prompt_file(task, args.prompt_split, output)),
            reference=reference, variants=variants))
        record["tasks"].sort(key=lambda entry: args.tasks.index(entry["task"]))
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Prepared {task}: {len(variants)} adapters over {len(examples)} examples", flush=True)
    if sha256(args.generator_checkpoint) != digest:
        raise ValueError("The generator checkpoint changed while adapters were being generated")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "evaluate", "report"])
    parser.add_argument("--output", required=True, help="directory holding adapters, predictions and reports")
    parser.add_argument("--generator-checkpoint", help=f"prefix-GPT checkpoint (default {DEFAULT_MODEL})")
    parser.add_argument("--vq-model", help="VQ-VAE decoder; defaults to the one recorded in the generator")
    parser.add_argument("--base-model", help="Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-root", help="folder of per-task original LoRA checkpoints")
    parser.add_argument("--samples", type=int, default=5, help="adapters generated per task")
    parser.add_argument("--originals", type=int, default=0,
                        help="last original checkpoints per task to score alongside; 0 leaves them out")
    parser.add_argument("--checkpoint-step", type=int,
                        help="step to condition on; defaults to the task's latest in the generator manifest")
    parser.add_argument("--prompt-split", choices=["evaluation", "train"], default="evaluation",
                        help="prompts used as conditioning")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-utilization", type=float,
                        help="fraction of GPU memory for vLLM; sized to what is free by default")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="score only the first N examples per task")
    parser.add_argument("--markdown", help="report: write the tables to this file")
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
