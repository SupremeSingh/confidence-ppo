# Confidence or correctness?

**Does the way a language model expresses confidence change how well it learns from experience?**

A language model writes an answer one token at a time. Reinforcement learning lets it practice: generate an answer, receive a reward, and adjust its parameters so useful behavior becomes more likely.

The same model can be prompted to sound decisive or to question its reasoning. These instructions may change what it tries, whether it catches mistakes, and how long it spends answering. This project asks whether those differences change **learning**, beyond changing the wording of a single answer.

## From answers to learning

Suppose the question is “What is 17 × 18?” The correct answer is 306. A confident answer might state it directly; a self-critical answer might check the multiplication first. Either can still be right or wrong.

We use **Proximal Policy Optimization (PPO)** with two neural networks:

1. The **actor** generates a solution. Its state is the question plus the answer written so far; its next action is a token.
2. **Math-Verify** checks the final answer and supplies a correctness reward. Confident wording earns no extra reward.
3. A separate **critic** estimates expected future reward at each prefix. Generalized Advantage Estimation (GAE) combines these estimates with the observed reward to construct a learning signal.
4. PPO updates the actor with a clipped objective that limits the incentive for large probability changes on sampled actions. The critic learns to better fit its value targets.

The model learns from answers it generates and an external correctness signal. It does not decide for itself which answers deserve rewards.

## The experiment

Every run starts from the same **Qwen2.5-Math-1.5B** weights. We vary an instruction before the question:

| Condition | Requested behavior |
|---|---|
| Neutral | Work through the problem carefully. |
| Confident | Express confidence while checking calculations and correcting mistakes when needed. |
| Self-critical | Examine reasoning critically while checking calculations and correcting mistakes when needed. |

The answer format, reward, training questions, PPO settings and generation limits are identical. Confidence and self-criticism are **prompted behaviors**, not claims about internal feelings or beliefs.

Each condition runs with five matched training seeds. Each run uses **20 updates × 64 generated answers = 1,280 training answers**. We select **512 training, 64 validation and 300 test questions** from a saved Numina split. Normalized question families cannot cross splits, and each selected question must fit every prompt.

The actor and critic are separate Transformers. We use the predeclared `more_fitting` conventional PPO configuration. The original project's small linear critic is not used.

## What would count as evidence?

We evaluate before and after training in two ways:

- **Assigned prompt:** does the model improve with the instruction it trained with?
- **Common neutral prompt:** does the improvement remain when all models receive the same neutral instruction?

The primary measurement is **learning gain under the common neutral prompt**:

```text
learning gain = accuracy after training − accuracy before training
style effect  = gain with a style − gain in the neutral condition
```

A higher final score alone is insufficient if a condition already started higher. Comparing gains helps distinguish starting performance from improvement during training.

We report paired question/seed bootstrap intervals, including 97.5% intervals for the two planned comparisons against neutral. Full responses, verifier statuses, token lengths, truncation and critic health checks are saved.

This is an **exploratory pilot** on previously inspected questions. Five seeds do not guarantee significance. Manually check style compliance and grading errors before interpreting results. Differences could involve checking behavior, response length or wording—not confidence alone. If neutral PPO fails to learn, investigate that baseline before drawing broad conclusions about the styles.

## Run on Duke

This repository has its own code, configs, Python environments and outputs. Its runtime is a private snapshot of the original project, with explicit evaluation under both prompts. After asset import, it does not import from the original project.

**On the Duke login terminal**, clone once:

```bash
cd /usr/xtmp/ms785
git clone https://github.com/SupremeSingh/confidence-ppo.git
cd confidence-ppo
```

Import physical copies of the model, tokenizer, question snapshot and required metadata once:

```bash
python3 scripts/import_assets.py \
  --from-project /usr/xtmp/ms785/rl_ms_project \
  --study outputs/numina-online-768-large-v1
```

The importer verifies checksums and leaves the source unchanged. Large assets are ignored by Git; cloning alone does not transfer them. The existing checksum-verified Singularity/Apptainer image is shared by default. See `scripts/cluster_env.sh` for runtime paths; set `MATH_RL_IMAGE` if using a private image copy.

Submit setup, preparation and a three-condition smoke test:

```bash
source scripts/cluster_env.sh
bash scripts/submit.sh
```

After completion:

```bash
cat outputs/confidence-v1/smoke.json
```

If it shows `"passed": true`, launch all fifteen training runs and the final report:

```bash
bash scripts/submit.sh full
```

Monitor and read results:

```bash
squeue -u "$USER"
cat outputs/confidence-v1/jobs.tsv
cat outputs/confidence-v1/report.txt
```

At most two training jobs run simultaneously, each using two A5000 GPUs. Queue time is additional. Avoid duplicate submissions; partial training runs do not resume automatically. Use a new output directory if changing the protocol.

## Inspect the evidence

`outputs/confidence-v1/summary.json` contains measurements and comparisons. Each condition's run folder contains checkpoints, critic diagnostics and complete answers:

```text
base-test.jsonl            Before training, assigned prompt
final-test.jsonl           After training, assigned prompt
base-neutral-test.jsonl    Before training, common neutral prompt
final-neutral-test.jsonl   After training, common neutral prompt
```

A later **J-lens** study could inspect matched examples around errors and corrections to explore mechanisms behind a behavioral result. These readouts would be diagnostics, not proof of causation.

Run local checks with:

```bash
PYTHONPATH=src python3 -m pytest -q
```

## Neutral GRPO pilot

To check whether removing the learned critic improves learning, first run a
neutral-only GRPO pilot. On the **Duke login terminal**, from this repository:

```bash
bash scripts/grpo_pilot.sh submit
```

This submits one two-A5000 job with seed 301, 20 updates and 1,280 answers,
keeping the actor learning rate at `1e-6`. It copies and checksum-verifies the
prepared neutral training and validation data from `outputs/confidence-v1`.
Evaluation uses the 64 validation questions; test questions are not evaluated.
The existing runtime performs GRPO numerical and reward checks before training.

Results go to a new `outputs/confidence-grpo-neutral-v1` directory. Submission
refuses to reuse an existing destination. A manifest records the pilot settings
and data hashes; the runtime records code provenance and the resolved config.
Completed PPO results are preserved. Cancel unwanted PPO jobs explicitly before
starting this pilot; the script does not cancel jobs or rebuild environments.
Updating repository scripts invalidates the old study's prepared code hashes,
so do not update the checkout while its PPO jobs are still running.

```bash
tail -f outputs/confidence-grpo-neutral-v1/grpo-seed301.log
bash scripts/grpo_pilot.sh report
```

The job also generates `report.txt` automatically on success. Missing results
mean the job is pending, running or failed; inspect its training and Slurm logs.
One seed on 64 validation questions is diagnostic evidence. Establish a useful
neutral baseline before expanding to multiple seeds and prompt conditions.

The aim is a small, interpretable test of whether how we ask a model to reason changes what it learns from reward—and whether that learning persists without the original instruction.
