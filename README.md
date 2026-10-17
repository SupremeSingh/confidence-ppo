# Confidence or correctness?

**Current workflow:** [Screened GSM8K GRPO study](#screened-gsm8k-grpo-study-current-protocol).
Establish neutral learning on 128 screened practice questions before comparing
prompt styles. The PPO and unscreened pilots below are historical protocols.

**Does the way a language model expresses confidence change how well it learns from experience?**

A language model writes an answer one token at a time. Reinforcement learning lets it practice: generate an answer, receive a reward, and adjust its parameters so useful behavior becomes more likely.

The same model can be prompted to sound decisive or to question its reasoning. These instructions may change what it tries, whether it catches mistakes, and how long it spends answering. This project asks whether those differences change **learning**, beyond changing the wording of a single answer.

## Original PPO pilot: learning from answers

Suppose the question is “What is 17 × 18?” The correct answer is 306. A confident answer might state it directly; a self-critical answer might check the multiplication first. Either can still be right or wrong.

We use **Proximal Policy Optimization (PPO)** with two neural networks:

1. The **actor** generates a solution. Its state is the question plus the answer written so far; its next action is a token.
2. **Math-Verify** checks the final answer and supplies a correctness reward. Confident wording earns no extra reward.
3. A separate **critic** estimates expected future reward at each prefix. Generalized Advantage Estimation (GAE) combines these estimates with the observed reward to construct a learning signal.
4. PPO updates the actor with a clipped objective that limits the incentive for large probability changes on sampled actions. The critic learns to better fit its value targets.

The model learns from answers it generates and an external correctness signal. It does not decide for itself which answers deserve rewards.

## Original PPO experimental design

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

## Original PPO pilot commands

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
python3 scripts/diagnose_grpo.py
```

The read-only diagnostic summarizes actor learning rates, gradients, policy
metrics, training correctness, validation answer changes, lengths and verifier
statuses. It reports mixed-reward groups when saved rollout dumps contain
recognizable groups of four identical input prompts; otherwise it explicitly
marks this measurement unavailable. It needs only Python's standard library.

The job also generates `report.txt` automatically on success. Missing results
mean the job is pending, running or failed; inspect its training and Slurm logs.
One seed on 64 validation questions is diagnostic evidence. Establish a useful
neutral baseline before expanding to multiple seeds and prompt conditions.

To compare a higher actor learning rate while retaining seed 301 and the
20-update budget, submit a separate pilot:

```bash
bash scripts/grpo_pilot.sh submit outputs/confidence-grpo-neutral-lr3e-6-v1 3e-6
bash scripts/grpo_pilot.sh report outputs/confidence-grpo-neutral-lr3e-6-v1
python3 scripts/diagnose_grpo.py --out outputs/confidence-grpo-neutral-lr3e-6-v1
```

The optional third argument to `submit` is the actor learning rate (default
`1e-6`). It is stored in the pilot manifest and used by the compute job. The
pinned console logger prints three decimal places, so these small learning
rates appear as `0.000`; consult `run.json` for the configured value. Select
settings using validation, then confirm across seeds before expanding the study.

The aim is a small, interpretable test of whether how we ask a model to reason changes what it learns from reward—and whether that learning persists without the original instruction.

## Screened GSM8K GRPO study (current protocol)

Use this workflow instead of the unscreened diagnostic below. On the **Duke
login terminal**, after syncing the repository and finishing older training jobs:

```bash
bash scripts/gsm8k_study.sh pilot
```

This queues screening, a two-update GPU smoke test, three neutral training runs,
and an automatic learning-gate report. Dependencies stop later stages if an
earlier stage fails. Screening uses one A5000 GPU; training uses two GPUs per job,
with at most two training jobs at once. The default output directory is
`outputs/gsm8k-screened-grpo-v1`; duplicate submissions to it are refused.

The protocol reserves 512 randomly ordered, prompt-length-eligible validation
questions from GSM8K's official training split before screening. It then screens
at most 512 separate candidates with eight neutral base-model answers each,
selecting the first 128 with exactly 2–6 correct answers. Any candidate with a
verifier status other than correct/incorrect is ineligible; all statuses and
responses are saved for inspection. Generation uses temperature 1, top-p 1,
top-k -1 and a 2,048-token response ceiling. Screening stops after the first
completed batch containing 128 qualifying questions and fails if the candidate
budget is insufficient. Normalized duplicate wording cannot cross pools, and
every question must fit every style's 512-token prompt ceiling. Dataset revision,
selected IDs, data/model/code hashes and screening cost are recorded. The
official GSM8K test split is never loaded.

Neutral pilot seeds are 201, 202 and 203. Each starts from the original base
weights and uses 40 updates × 16 questions × 4 answers = 2,560 training answers,
with actor learning rate `1e-6` and correctness-only rewards. Before/after
evaluation includes greedy practice accuracy, eight independently seeded fresh
practice samples per question, assigned-style validation and common-neutral
validation. Screening and sampled-evaluation seeds are separate; screening
answers are never reused as a learning baseline. Full-precision telemetry records
learning rates, actor gradients and policy-change proxies, plus per-update mixed,
all-correct and all-incorrect group counts. These proxies do not measure exact
parameter displacement. The runtime's sampled-evaluation extension is exercised
by the GPU smoke test before any 40-update runs are allowed.

Inspect progress and the gate:

```bash
squeue -u "$USER"
bash scripts/gsm8k_study.sh status
```

The frozen gate requires complete finite audits, positive learning rates,
nonzero actor gradients and a nonzero policy-change proxy; positive greedy
practice gain in **each** pilot seed; and positive mean independently sampled
practice gain. Validation improvement is reported but does not determine this
optimization gate. A failed gate exits unsuccessfully and blocks the style
comparison. Inspect `logs/`, `screening-summary.json`, `smoke.log`, and the
per-seed training logs for failures. Partial runs do not automatically resume.
Do not change the thresholds in place after seeing results; changed protocols
need a new output directory.

Only after a passed gate, submit the full comparison:

```bash
bash scripts/gsm8k_study.sh full
# Once all runs finish:
bash scripts/gsm8k_study.sh report
```

This launches 15 fresh runs: neutral, confident and self-critical prompts × five
matched seeds (301–305). Pilot seeds are excluded from this comparison. Every
condition uses the same screened questions and settings, starting from the
original base weights. The two primary contrasts compare held-out neutral-prompt
learning gains against neutral training, with paired seed/question bootstrap
95% and 97.5% intervals. Assigned-style gains are secondary. Five seeds provide
exploratory uncertainty estimates, not guaranteed significance. Validation is a
development set once results guide decisions. Equal answer budgets need not
mean equal token or compute budgets.

The final `summary.json` includes per-seed evidence, verifier statuses, lengths,
truncation, group counts, pilot/full training costs and separate screening cost.
Training GPU hours exclude explicit evaluation and setup; each completed run's
`execution.json` separately records driver wall time including evaluation.
`review.jsonl` contains a fixed random sample of matched before/after answers
from every style for manual grading/style-compliance review. Higher mixed-group
fractions are potential mechanism evidence, not proof of causation.

Two-thirds-layer error probes and J-lens remain conditional follow-ups. Before
probing, define local-error versus eventual-answer labels, match prefixes before
the answer, split by question, and include probe-capacity/control-task checks.
The behavioral pipeline does not automatically launch probes or claim a causal
mechanism. Do not edit this checkout or shared environments while jobs run:
prepared hashes are checked before subsequent runs.

## Earlier unscreened GSM8K learning check

Before expanding a style study, use a separate neutral GRPO diagnostic to ask
whether the model can improve on a small, repeatedly practiced set. On Duke:

```bash
bash scripts/gsm8k_pilot.sh submit
# After completion:
bash scripts/gsm8k_pilot.sh report
```

The job downloads only the official GSM8K training split, resolves and records
an immutable dataset revision, and selects 32 practice and 256 held-out validation
questions by a deterministic shuffle. Normalized duplicate wording cannot cross
splits. Every selected question must fit the 512-token prompt budget; insufficient
counts fail explicitly. The official test set is unused. Preparation requires
Hugging Face access; failures are recorded in `preparation.log`.

It uses the same base model, neutral completion prompt, Math-Verify correctness
reward and GRPO runtime, with seed 301, learning rate `1e-6`, 20 updates and
1,280 answers. The 32 practice questions repeat during training. An optional
training-probe evaluation records greedy accuracy on those questions before and
after training, alongside the 256-question validation evaluation. Improvement
on practice questions is an optimization check, not evidence of generalization;
near-ceiling initial accuracy would limit its usefulness. No improvement does
not by itself establish a code bug.

Outputs are isolated under `outputs/gsm8k-grpo-learning-check-v1`, with request,
data/protocol manifest, code provenance, logs, full before/after answers,
checkpoint and `report.txt`. Submission refuses to reuse the output directory.
This is a new diagnostic protocol, not a continuation of the MATH pilot. Do not
update the checkout while training jobs are active. The optional training-probe
setting defaults to off for existing protocols.
