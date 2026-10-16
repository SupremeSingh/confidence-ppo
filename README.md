# Confidence or correctness?

Does expressed confidence or self-criticism change how effectively an LLM learns from correctness rewards?

This is a standalone Duke-cluster experiment. It contains its own PPO implementation snapshot, configuration, launchers, tests and dependency setup. It never imports code or Python environments from the original project. The actor and critic are separate Transformers; the small linear critic is not used.

## Experiment

Three conditions start from the same Qwen2.5-Math-1.5B weights: neutral, confident and self-critical instructions. Confident and critical instructions both permit checking calculations and correcting mistakes. The ordinary math instruction and answer format are identical. This manipulates prompted behavior, not psychological confidence.

- **Data:** 512 training, 64 validation and 300 test questions from a copied Numina split. Normalized question families cannot cross splits. Every selected question must fit all three prompts.
- **Training:** five matched seeds per condition; 20 PPO updates × 64 answers = 1,280 generated training answers per run. Same two GPUs, sampling, response ceiling, correctness reward and predeclared `more_fitting` PPO profile. Different styles can consume different numbers of tokens.
- **Evaluation:** before/after greedy answers under the assigned instruction and a common neutral instruction. Full responses, token counts, verifier statuses and truncation are saved.
- **Primary endpoint:** learning gain under the common neutral instruction, confident versus neutral and critical versus neutral. Assigned-style results are secondary. Paired question/seed bootstrap intervals include 97.5% intervals for the two planned comparisons.
- **Interpretation:** five seeds and previously inspected questions make this an exploratory pilot. Significance is not guaranteed. Manually check style compliance and verifier errors. If neutral PPO does not learn, investigate the baseline before scaling. Differences can involve checking behavior, prompt wording and response length as well as confidence.

The code in `src/math_rl/` is a private snapshot of the existing runtime, with explicit evaluation under both prompts. Some shared helpers retain other critic implementations, but these are not selected. No changes to the original repository are needed.

## 1. Make this a separate repository

On your Mac, copy this folder outside the original project:

```bash
rsync -a --exclude='__pycache__' --exclude='.pytest_cache' \
  /Users/manmit/Desktop/rl_ms_project/sidequests/confidence_ppo/ \
  /Users/manmit/Desktop/confidence_ppo/
cd /Users/manmit/Desktop/confidence_ppo
git init
git add .
git commit -m "Initialize standalone confidence PPO study"
```

You can add an empty GitHub repository as `origin` and push this repository independently. Alternatively, copy the folder directly to Duke with your usual SSH alias:

```bash
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='.pytest_cache' \
  /Users/manmit/Desktop/confidence_ppo/ \
  ms785@compsci-login-02:/usr/xtmp/ms785/confidence_ppo/
```

## 2. Import assets once on Duke

The multi-gigabyte model and prepared data live on Duke, not in the local source checkout. Make physical copies into this separate directory:

```bash
cd /usr/xtmp/ms785/confidence_ppo
python3 scripts/import_assets.py \
  --from-project /usr/xtmp/ms785/rl_ms_project \
  --study outputs/numina-online-768-large-v1
```

This copies the question snapshot, base model/tokenizer and required asset/container metadata, verifies checksums, and records `inputs/assets.json`. The original project is read only during import. Model cache symlinks are dereferenced: this copy remains usable if the original checkout is later moved. It requires enough disk space for another model copy.

After import, no original-project path is used during setup or training. Data and model weights are ignored by Git; copying only Git history to another machine does not transfer those assets. Transfer `inputs/`, `models/` and the imported `configs/assets.json` and `configs/container.sha256` separately if relocating the study. Dataset/model usage terms still apply.

The Singularity/Apptainer container is a machine runtime. By default, the existing image at `/usr/xtmp/ms785/containers/verl-v041.sif` is reused and checksum checked. To make a private physical copy too:

```bash
mkdir -p containers
cp /usr/xtmp/ms785/containers/verl-v041.sif containers/
export MATH_RL_IMAGE="$PWD/containers/verl-v041.sif"
```

Export that variable in future submission shells if using the private image. Container and cache paths can otherwise stay shared, while code, environments, data and outputs are independent.

## 3. Prepare and smoke-test

From this standalone repository root:

```bash
source scripts/cluster_env.sh
bash scripts/submit.sh
```

This schedules private environment setup, question preparation and a three-condition GPU smoke test in dependency order. Setup installs pinned VERL and the isolated Math-Verify environment in this folder. It does not modify the original project's environments. Existing cluster runtime, network access for dependency installation, and A5000 GPUs are required.

After completion:

```bash
cat outputs/confidence-v1/smoke.json
```

Only if it says `"passed": true`:

```bash
bash scripts/submit.sh full
```

Fifteen training jobs run at most two at a time, followed by a report. Inspect:

```bash
squeue -u "$USER"
cat outputs/confidence-v1/jobs.tsv
cat outputs/confidence-v1/report.txt
```

Outputs include `summary.json`, per-run critic health, checkpoints and `base-test.jsonl`, `final-test.jsonl`, `base-neutral-test.jsonl`, `final-neutral-test.jsonl`. The JSONL files contain complete responses for manual inspection and later diagnostics.

Smoke takes time because each condition loads two models and evaluates before and after training under both prompts. Measure its runtime to budget the full pilot; queue time is additional. Two simultaneous jobs consume four GPUs and can compete with your main experiment for capacity. Launch after the main timing study for cleaner runtime comparisons.

Use a new output directory if changing the protocol. Partial training runs do not resume automatically; do not launch duplicate writers. A custom output is supported with `bash scripts/submit.sh smoke outputs/another-study` and the matching `full` command.

## Tests and later diagnostics

```bash
PYTHONPATH=src python3 -m pytest -q
```

Local tests cover paired statistics, seed assignment, safe physical asset copying and Slurm dependencies. Only the cluster smoke validates the actual GPU/VERL training path.

J-lens is a later diagnostic: inspect selected errors and corrections using saved responses and checkpoints. Vocabulary readouts can suggest mechanisms, but do not establish that confidence caused a learning difference. No J-lens dependencies or fitting are required by this pilot.
