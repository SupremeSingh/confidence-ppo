"""Paired, greedy GSM8K evaluation of the base model and two saved pilot actors."""
import argparse
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from math_rl.prompts import encode_completion, validate_assets
from math_rl.provenance import sha256, snapshot, write_json
from math_rl.reward import parse_number

VERL_COMMIT = "8d9e350ea58c7ad4b50dd14d9dcb50577242c55f"
OFFSET = 1024  # Reserve the earlier development/audit portion of the shuffled train split.
SETTINGS = dict(n=1, temperature=0.0, top_p=1.0, top_k=-1, max_tokens=2048, seed=42)
LABELS = ("base", "ppo", "grpo")
TRAINING_FILES = ("configs/assets.json", "data/gsm8k/train.parquet", "data/gsm8k/val.parquet",
                  "src/math_rl/prompts.py", "src/math_rl/math_verify_reward.py", "src/math_rl/verifier_batch.py")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(path, rows):
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def normalize(question):
    return " ".join(question.split())


def select_questions(source, tokenizer, count, excluded_ids, excluded_questions):
    if count < 1:
        raise ValueError("Evaluation count must be positive")
    selected, seen = [], set(excluded_questions)
    for row in source:
        identity = f"gsm8k/train/{row['original_index']}"
        question = row["question"]
        key = normalize(question)
        if identity in excluded_ids or key in seen:
            continue
        seen.add(key)
        prompt, tokens = encode_completion(tokenizer, question)
        if len(tokens) > 512:
            continue
        _, separator, answer = row["answer"].rpartition("####")
        if not separator or parse_number(answer.strip()) is None:
            raise ValueError(f"Invalid reference for {identity}")
        selected.append(dict(id=identity, question=question, prompt=prompt,
                             prompt_token_ids=tokens, ground_truth=answer.strip()))
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} eligible questions; need {count}")


def prepare_questions(root, count):
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer

    assets = json.loads((root / "configs/assets.json").read_text())
    validate_assets(assets)
    tokenizer = AutoTokenizer.from_pretrained(root / "models/qwen-math", local_files_only=True)
    excluded_ids, excluded_questions = set(), set()
    for split in ("train", "val"):
        for row in Dataset.from_parquet(str(root / f"data/gsm8k/{split}.parquet")):
            excluded_ids.add(row["extra_info"]["prompt_id"])
            excluded_questions.add(normalize(row["prompt"][1]["content"]))
    # Exclude saved diagnostic/audit examples too, even if their locations changed.
    for path in (root / "outputs").rglob("metadata.json"):
        metadata = json.loads(path.read_text())
        if metadata.get("mode") in ("audit", "diagnostic"):
            for row in read_jsonl(path.parent / "responses.jsonl"):
                excluded_ids.add(row["prompt_id"])

    source = load_dataset("openai/gsm8k", "main", split="train", revision=assets["dataset_revision"])
    source = source.add_column("original_index", list(range(len(source)))).shuffle(seed=assets["seed"])
    excluded_ids.update(f"gsm8k/train/{source[i]['original_index']}" for i in range(OFFSET))
    # Exclude duplicate wording even when it has a different dataset index.
    excluded_questions.update(normalize(row["question"]) for row in source
                              if f"gsm8k/train/{row['original_index']}" in excluded_ids)
    selected = select_questions(source, tokenizer, count, excluded_ids, excluded_questions)
    path = root / f"data/gsm8k/eval-{count}.json"
    contract = dict(assets=assets, count=count, offset=OFFSET,
                    prompt_code_sha256=sha256(root / "src/math_rl/prompts.py"))
    if path.exists():
        frozen = json.loads(path.read_text())
        if frozen["contract"] != contract:
            raise ValueError("Frozen evaluation settings changed; preserve and explicitly version the set")
        rows = frozen["questions"]
        validate_questions(rows, count, excluded_ids, excluded_questions)
        if rows != selected:
            raise ValueError("Frozen questions, references, tokenizer or exclusions changed")
        return path

    rows = selected
    validate_questions(rows, count, excluded_ids, excluded_questions)
    with path.open("x") as handle:
        json.dump(dict(contract=contract, excluded_ids=sorted(excluded_ids), questions=rows), handle, indent=2)
        handle.write("\n")
    return path


def validate_questions(rows, count, excluded_ids=(), excluded_questions=()):
    if count < 1 or len(rows) != count or len({r["id"] for r in rows}) != count:
        raise ValueError("Incomplete evaluation set or duplicate IDs")
    if len({normalize(r["question"]) for r in rows}) != count:
        raise ValueError("Duplicate evaluation questions")
    for row in rows:
        if row["id"] in excluded_ids or normalize(row["question"]) in excluded_questions:
            raise ValueError(f"Evaluation overlaps development/audit data: {row['id']}")
        if parse_number(row["ground_truth"]) is None or not 0 < len(row["prompt_token_ids"]) <= 512:
            raise ValueError("Invalid evaluation reference or prompt")


def inspect_checkpoint(root, method, pilot_job, step):
    run = root / f"checkpoints/{method}-{pilot_job}"
    checkpoint = run / f"global_step_{step}"
    record = json.loads((run / "run.json").read_text())
    config = record["config"]
    if config["algorithm"]["adv_estimator"] != {"ppo": "gae", "grpo": "grpo"}[method]:
        raise ValueError(f"Wrong algorithm in {run}")
    # Keep the evaluation contract tied to the actual training data and reward.
    for key, expected in (("train_files", "data/gsm8k/train.parquet"), ("val_files", "data/gsm8k/val.parquet")):
        if config["data"][key] != expected:
            raise ValueError(f"Unexpected {key}; explicitly exclude this run's data first")
    for name in TRAINING_FILES:
        if record["provenance"]["files"].get(name) != sha256(root / name):
            raise ValueError(f"Training/evaluation provenance differs: {name}")
    if not (checkpoint / "data.pt").is_file():
        raise ValueError(f"Incomplete checkpoint: {checkpoint}")
    actor = checkpoint / "actor"
    fsdp = json.loads((actor / "fsdp_config.json").read_text())
    world = config["trainer"]["n_gpus_per_node"] * config["trainer"]["nnodes"]
    if fsdp["world_size"] != world or not (actor / "huggingface/config.json").is_file():
        raise ValueError("Checkpoint export metadata missing or inconsistent")
    files = [actor / f"model_world_size_{world}_rank_{rank}.pt" for rank in range(world)]
    if any(not p.is_file() or p.stat().st_size == 0 for p in files):
        raise ValueError(f"Missing actor weight shards in {actor}")
    return actor, dict(run=str(run), step=step, run_sha256=sha256(run / "run.json"),
                       actor_shards={p.name: sha256(p) for p in files},
                       verifier=record["provenance"]["verifier"])


def export_actor(actor, model, log_path):
    # Use the training version's merger, then check exported tensors against shards.
    command = [sys.executable, "-m", "verl.model_merger"]
    common = ["--backend", "fsdp", "--local_dir", str(actor)]
    with log_path.open("x") as log:
        subprocess.run(command + ["merge", *common, "--target_dir", str(model)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        if not list(model.glob("*.safetensors")):
            raise RuntimeError(f"No exported weights: {model}")
        subprocess.run(command + ["test", *common, "--test_hf_dir", str(model)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def generate(model, questions, out):
    import torch
    from vllm import LLM, SamplingParams
    from math_rl.ppo_reward import compute_score

    rows = json.loads(questions.read_text())["questions"]
    out.mkdir()  # Refuse overwrites or partially completed outputs.
    engine = LLM(model=str(model), tokenizer=str(ROOT / "models/qwen-math"),
                 dtype="bfloat16", tensor_parallel_size=1, max_model_len=2560,
                 gpu_memory_utilization=0.6, max_num_seqs=8, max_num_batched_tokens=2560,
                 enforce_eager=True, seed=42, generation_config="vllm")
    params = SamplingParams(**SETTINGS)
    # Identical batches/order for every model; persist each completed batch.
    with (out / "responses.jsonl").open("x") as handle:
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            outputs = engine.generate([{"prompt_token_ids": row["prompt_token_ids"]} for row in batch], params)
            if len(outputs) != len(batch) or any(len(o.outputs) != 1 for o in outputs):
                raise RuntimeError("Incomplete generation batch")
            answers = [output.outputs[0] for output in outputs]
            scores = compute_score(["gsm8k"] * len(batch), [a.text for a in answers],
                                   [r["ground_truth"] for r in batch])
            for row, answer, score in zip(batch, answers, scores, strict=True):
                handle.write(json.dumps(dict(id=row["id"], ground_truth=row["ground_truth"],
                    response=answer.text, response_tokens=len(answer.token_ids),
                    finish_reason=answer.finish_reason, **score)) + "\n")
            handle.flush()
    write_json(out / "metadata.json", dict(mode="evaluation", model=str(model),
               questions_sha256=sha256(questions), responses_sha256=sha256(out / "responses.jsonl"),
               sampling=SETTINGS, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
               vllm=version("vllm"), transformers=version("transformers")))


def paired_counts(base, trained):
    improved = sum(a == 0 and b == 1 for a, b in zip(base, trained, strict=True))
    regressed = sum(a == 1 and b == 0 for a, b in zip(base, trained, strict=True))
    # Exact two-sided paired sign/McNemar test on discordant questions.
    from math import comb
    discordant = improved + regressed
    p = min(1., 2 * sum(comb(discordant, k) for k in range(min(improved, regressed) + 1)) / 2**discordant)
    return dict(wrong_to_right=improved, right_to_wrong=regressed,
                unchanged=len(base) - discordant, net_correct=improved - regressed,
                accuracy_change_percentage_points=100 * (improved - regressed) / len(base),
                paired_p_value=p)


def report(out):
    questions = json.loads((out / "questions.json").read_text())["questions"]
    validate_questions(questions, len(questions))
    ids = [q["id"] for q in questions]
    results, summary = {}, {}
    for label in LABELS:
        folder = out / label
        metadata = json.loads((folder / "metadata.json").read_text())
        if label != "base" and any(metadata[k] != json.loads((out / "base/metadata.json").read_text())[k]
                                   for k in ("gpu", "torch", "vllm", "transformers")):
            raise ValueError(f"Unmatched generation runtime: {label}")
        if metadata["questions_sha256"] != sha256(out / "questions.json") or metadata["sampling"] != SETTINGS:
            raise ValueError(f"Unmatched questions or decoding settings: {label}")
        if metadata["responses_sha256"] != sha256(folder / "responses.jsonl"):
            raise ValueError(f"Responses changed: {label}")
        rows = read_jsonl(folder / "responses.jsonl")
        if [r["id"] for r in rows] != ids:
            raise ValueError(f"Incomplete, duplicated or reordered responses: {label}")
        for question, row in zip(questions, rows, strict=True):
            if row["ground_truth"] != question["ground_truth"] or row["score"] not in (0, 1):
                raise ValueError(f"Invalid reference or reward: {label}")
            if row["verifier_status"] not in {"correct", "incorrect", "parse_failure", "unsupported_parse"}:
                raise ValueError(f"Verifier runtime error: {label}")
            if row["score"] != int(row["verifier_status"] == "correct"):
                raise ValueError(f"Reward disagrees with verifier status: {label}")
        results[label] = rows
        summary[label] = dict(correct=sum(r["score"] for r in rows), questions=len(rows),
                              accuracy=sum(r["score"] for r in rows) / len(rows),
                              truncation_rate=sum(r["finish_reason"] == "length" for r in rows) / len(rows))
    comparisons = {label: paired_counts([r["score"] for r in results["base"]],
                                        [r["score"] for r in results[label]]) for label in ("ppo", "grpo")}
    details = [dict(id=q["id"], question=q["question"], ground_truth=q["ground_truth"],
                    models={label: results[label][i] for label in LABELS}) for i, q in enumerate(questions)]
    write_jsonl(out / "changes.jsonl", [r for r in details if len({v["score"] for v in r["models"].values()}) > 1])
    sample = random.Random(42).sample(details, min(20, len(details)))
    with (out / "review.txt").open("x") as handle:
        for row in sample:
            handle.write(f"\n{row['id']}\nQUESTION: {row['question']}\nREFERENCE: {row['ground_truth']}\n")
            for label, result in row["models"].items():
                handle.write(f"\n{label.upper()} | score={result['score']} | extracted={result['extracted']}\n{result['response']}\n")
    result = dict(models=summary, versus_base=comparisons,
                  scope="One seed, fixed greedy decoding, automated Math-Verify scores. Review answers before claiming improvement.",
                  statistics="Paired p-values are descriptive, unadjusted for two comparisons; no automatic superiority claim.")
    write_json(out / "summary.json", result)
    print(json.dumps(result, indent=2))


def run(args):
    import torch
    from math_rl.ppo_reward import verifier_command

    direct = json.loads(distribution("verl").read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != VERL_COMMIT:
        raise RuntimeError("Evaluation requires the same pinned VERL as training")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Request one BF16-capable GPU")
    args.out.mkdir(parents=True, exist_ok=False)
    verifier = json.loads(subprocess.check_output(verifier_command() + ["--provenance"], text=True,
                         env=dict(os.environ, PYTHONPATH=str(ROOT / "src"))))
    actors, checkpoints = {}, {}
    for method in ("ppo", "grpo"):
        actors[method], checkpoints[method] = inspect_checkpoint(ROOT, method, args.pilot_job, args.step)
        if checkpoints[method]["verifier"] != verifier:
            raise ValueError("Training and evaluation verifier environments differ")
    questions = prepare_questions(ROOT, args.count)
    # Copy the frozen manifest so this run is independently inspectable.
    (args.out / "questions.json").write_bytes(questions.read_bytes())
    manifest = dict(pilot_job=args.pilot_job, checkpoints=checkpoints, verifier=verifier,
                    questions_sha256=sha256(questions), sampling=SETTINGS, provenance=snapshot(ROOT),
                    phase="prepared")
    write_json(args.out / "run.json", manifest)
    for label in LABELS:
        model = ROOT / "models/qwen-math"
        if label != "base":
            model = args.out / f"{label}-model"
            print(f"Exporting {label} actor from step {args.step}", flush=True)
            manifest["phase"] = f"exporting-{label}"
            write_json(args.out / "run.json", manifest)
            export_actor(actors[label], model, args.out / f"{label}-export.log")
        manifest.setdefault("model_weights_sha256", {})[label] = {p.name: sha256(p) for p in sorted(model.glob("*.safetensors"))}
        if not manifest["model_weights_sha256"][label]:
            raise RuntimeError(f"No model weights: {model}")
        manifest["phase"] = f"evaluating-{label}"
        write_json(args.out / "run.json", manifest)
        print(f"Evaluating {label} on {args.count} questions", flush=True)
        with (args.out / f"{label}.log").open("x") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "generate", "--model", str(model),
                            "--questions", str(args.out / "questions.json"), "--out", str(args.out / label)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    report(args.out)
    manifest["phase"] = "complete"
    write_json(args.out / "run.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--pilot-job", required=True, type=int)
    runner.add_argument("--step", type=int, default=10)
    runner.add_argument("--count", type=int, choices=range(200, 501), default=500, metavar="200..500")
    runner.add_argument("--out", type=Path, required=True)
    generator = commands.add_parser("generate")
    generator.add_argument("--model", type=Path, required=True)
    generator.add_argument("--questions", type=Path, required=True)
    generator.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.model, args.questions, args.out)
    else:
        run(args)


if __name__ == "__main__":
    main()
