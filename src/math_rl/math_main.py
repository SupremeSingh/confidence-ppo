"""Pinned VERL entry point for the four-method, on-policy MATH comparison."""
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import hydra
from omegaconf import OmegaConf
import ray

from math_rl.provenance import sha256, snapshot, write_json

ROOT = Path(__file__).resolve().parents[2]
VERL_COMMIT = '8d9e350ea58c7ad4b50dd14d9dcb50577242c55f'


def validate(config):
    if config.experiment.method not in ('ppo', 'grpo', 'ridge', 'lstd'):
        raise ValueError('Unknown comparison method')
    expected = 'grpo' if config.experiment.method == 'grpo' else 'gae'
    if config.algorithm.adv_estimator != expected:
        raise ValueError('Method/advantage estimator mismatch')
    if (config.algorithm.gamma != 1 or config.algorithm.lam != .95
            or config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss):
        raise ValueError('Comparison uses gamma=1, actor GAE lambda=.95, no KL shaping')
    cumulative = config.experiment.critic_statistics == 'cumulative'
    if config.experiment.critic_statistics not in ('batch', 'cumulative'):
        raise ValueError('Unknown statistics mode')
    if cumulative:
        import math
        e = config.experiment
        if (e.method not in ('lstd', 'ridge') or config.actor_rollout_ref.model.critic_feature_mode != 'frozen'
                or e.replay_capacity or not math.isfinite(e.cumulative_epsilon) or e.cumulative_epsilon <= 0
                or not 0 <= e.critic_lambda <= 1 or e.statistics_reset_interval < 0
                or e.encoder_refresh_interval < 0
                or (e.encoder_refresh_interval and e.statistics_reset_interval != e.encoder_refresh_interval)):
            raise ValueError('Cumulative LSTD requires frozen features, no buffer, and resets on every refresh')
    elif (config.experiment.critic_lambda != .99 or config.experiment.critic_alpha != .01
          or config.experiment.encoder_refresh_interval or config.experiment.statistics_reset_interval):
        raise ValueError('Original comparison requires LSTD(.99), alpha=.01, no refresh/reset')
    from math_rl.ppo_baseline import PPO_PROFILES
    profile = config.experiment.ppo_profile
    if profile != 'legacy':
        chosen = PPO_PROFILES.get(profile)
        if chosen is None or config.experiment.method != 'ppo':
            raise ValueError('Invalid conventional PPO profile')
        if (config.critic.optim.lr != chosen['lr'] or config.critic.ppo_epochs != chosen['epochs']
                or config.critic.loss_agg_mode != 'seq-mean-token-mean'
                or config.critic.strategy != 'fsdp' or config.critic.use_dynamic_bsz
                or config.critic.ppo_micro_batch_size_per_gpu != 1
                or config.critic.ulysses_sequence_parallel_size != 1):
            raise ValueError('Baseline profile must use its locked critic settings')
    if config.experiment.evaluation_split not in ('test', 'val'):
        raise ValueError('Invalid evaluation split')
    if config.experiment.evaluation_split == 'val' and str(config.experiment.test_file) != str(config.data.val_files):
        raise ValueError('Calibration must evaluate validation data only')
    mode = config.actor_rollout_ref.model.critic_feature_mode
    layer = config.actor_rollout_ref.model.get('critic_feature_layer', 'final')
    if layer not in ('final', 'two_thirds', 'bottleneck') or (layer != 'final' and mode != 'frozen'):
        raise ValueError('Intermediate features require a frozen encoder')
    precision = config.actor_rollout_ref.model.get('critic_encoder_precision', 'bf16')
    if precision not in ('bf16', 'fp32'):
        raise ValueError('Unknown frozen encoder precision')
    if layer == 'bottleneck':
        model = config.actor_rollout_ref.model
        if (not cumulative or precision != 'fp32' or config.experiment.encoder_refresh_interval
                or not model.critic_bottleneck_path or not model.critic_bottleneck_sha256
                or not model.critic_bottleneck_layers):
            raise ValueError('Bottleneck requires immutable FP32 cumulative features and a hashed checkpoint')
        if sha256(model.critic_bottleneck_path) != model.critic_bottleneck_sha256:
            raise ValueError('Bottleneck checkpoint changed')
    capacity = config.experiment.replay_capacity
    if mode not in ('actor', 'frozen') or capacity < 0:
        raise ValueError('Invalid critic feature/replay mode')
    if capacity and (mode != 'frozen' or capacity < 64):
        raise ValueError('Replay requires frozen features and capacity >= 64 answers')
    if config.experiment.replay_max_transitions < 64 * 2048 or config.experiment.replay_max_age < 0:
        raise ValueError('Replay bounds must accommodate a complete batch and nonnegative age')
    if mode == 'frozen' and not config.actor_rollout_ref.actor.fsdp_config.param_offload:
        raise ValueError('Frozen encoder requires actor parameter offload between passes')
    rollout = config.actor_rollout_ref.rollout
    if (config.data.train_batch_size, rollout.n, rollout.temperature, rollout.top_p, rollout.top_k) != (16, 4, 1., 1., -1):
        raise ValueError('All methods require 16 prompts x 4 answers with matched sampling')
    if config.trainer.resume_mode != 'disable' or config.trainer.nnodes != 1:
        raise ValueError('Initial comparison supports fresh single-node runs only')
    if config.trainer.n_gpus_per_node != 2 or config.trainer.critic_warmup != 0 or config.trainer.val_only:
        raise ValueError('Comparison requires two GPUs and an actor update at every step')
    if (rollout.val_kwargs.n != 1 or rollout.val_kwargs.do_sample
            or config.data.max_prompt_length != 512 or config.data.max_response_length != 2048):
        raise ValueError('Fixed evaluation requires greedy n=1 and the 512/2048 token budgets')
    if config.actor_rollout_ref.actor.strategy != 'fsdp' or rollout.mode != 'sync':
        raise ValueError('Only the pinned synchronous FSDP path is supported')
    direct = json.loads(distribution('verl').read_text('direct_url.json') or '{}')
    if direct.get('vcs_info', {}).get('commit_id') != VERL_COMMIT:
        raise ValueError('Install the pinned VERL before running the custom trainer')


@ray.remote(num_cpus=1)
def run(config):
    import numpy as np
    import torch
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
    from math_rl.math_trainer import MathTrainer
    from math_rl.math_workers import FeatureActorWorker

    seed = config.data.seed
    os.environ['MATH_RL_SEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
    linear = config.experiment.method in ('ridge', 'lstd')
    roles = {Role.ActorRollout: ray.remote(FeatureActorWorker if linear else ActorRolloutRefWorker)}
    if config.experiment.method == 'ppo':
        from math_rl.ppo_baseline_worker import AuditedCriticWorker
        roles[Role.Critic] = ray.remote(CriticWorker if config.experiment.ppo_profile == 'legacy' else AuditedCriticWorker)
    pool = ResourcePoolManager(resource_pool_spec={'global_pool': [config.trainer.n_gpus_per_node]},
                               mapping={role: 'global_pool' for role in roles})
    train = create_rl_dataset(config.data.train_files, config.data, tokenizer, None)
    val = create_rl_dataset(config.data.val_files, config.data, tokenizer, None)
    trainer = MathTrainer(config=config, tokenizer=tokenizer, processor=None,
        role_worker_mapping=roles, resource_pool_manager=pool, ray_worker_group_cls=RayWorkerGroup,
        reward_fn=load_reward_manager(config, tokenizer, num_examine=0),
        val_reward_fn=load_reward_manager(config, tokenizer, num_examine=0),
        train_dataset=train, val_dataset=val, collate_fn=collate_fn,
        train_sampler=create_rl_sampler(config.data, train), device_name=config.trainer.device)
    started = time.perf_counter()
    trainer.init_workers()
    init_seconds = time.perf_counter() - started
    calibration = config.experiment.evaluation_split == 'val'
    initial = trainer.evaluate_test('base-validation' if calibration else 'base')
    def neutral_evaluation(label):
        saved = config.experiment.test_file
        config.experiment.test_file = config.experiment.neutral_test_file
        try:
            return trainer.evaluate_test(label + '-neutral')
        finally:
            config.experiment.test_file = saved
    initial_neutral = neutral_evaluation('base-validation' if calibration else 'base')
    started = time.perf_counter()
    trainer.fit()
    training_seconds = time.perf_counter() - started
    final = trainer.evaluate_test('final-validation' if calibration else 'final')
    final_neutral = neutral_evaluation('final-validation' if calibration else 'final')
    result = dict(neutral_initial=initial_neutral, neutral_final=final_neutral, method=config.experiment.method, seed=seed, initial_test=initial, final_test=final,
        updates=trainer.completed_updates, answers=trainer.completed_updates * 64,
        initialization_seconds=init_seconds, training_seconds=training_seconds,
        training_gpu_hours=training_seconds * config.trainer.n_gpus_per_node / 3600,
        scope='Training time includes validation, checkpointing, feature transport and critic fitting; excludes setup and test evaluation')
    result['ppo_profile'] = config.experiment.ppo_profile
    result['evaluation_split'] = config.experiment.evaluation_split
    if calibration:
        result['initial_validation'] = result.pop('initial_test')
        result['final_validation'] = result.pop('final_test')
    write_json(trainer.output / 'result.json', result)
    return result


@hydra.main(config_path='../../configs', config_name='math_online', version_base=None)
def main(config):
    OmegaConf.resolve(config)
    validate(config)
    out = Path(config.trainer.default_local_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new method output directory; automatic resume is deliberately disabled')
    out.mkdir(parents=True, exist_ok=True)
    from math_rl.ppo_reward import verifier_command
    verifier = json.loads(subprocess.check_output(verifier_command() + ['--provenance'], text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / 'src'))))
    write_json(out / 'run.json', dict(config=OmegaConf.to_container(config), provenance=snapshot(ROOT),
        verifier=verifier, data_hashes={str(p): sha256(p) for p in
            (config.data.train_files, config.data.val_files, config.experiment.test_file)}))
    # Preflight verifies the actual MATH files, not the old GSM8K pilot data.
    subprocess.run([sys.executable, str(ROOT / 'scripts/preflight.py'),
        '--grpo' if config.experiment.method == 'grpo' else '--ppo',
        '--expected-gpus', str(config.trainer.n_gpus_per_node),
        '--train-file', str(config.data.train_files), '--val-file', str(config.data.val_files)], check=True)
    os.environ['MATH_RL_SEED'] = str(config.data.seed)
    ray.init(num_cpus=config.ray_init.num_cpus, runtime_env={'env_vars': {
        'MATH_RL_SEED': str(config.data.seed), 'TOKENIZERS_PARALLELISM': 'false',
        'NCCL_DEBUG': 'WARN', 'VLLM_LOGGING_LEVEL': 'WARN', 'VLLM_USE_V1': '1'}})
    try:
        print(json.dumps(ray.get(run.remote(config)), indent=2))
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
