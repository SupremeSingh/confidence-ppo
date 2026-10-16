"""PPO/GRPO comparison using the pinned VERL loop, losses, GAE and checkpoints.

For linear critics only, the scoped advantage hook supplies cross-fitted values
where VERL normally reads transformer-critic values. No transformer critic is
allocated or updated in those runs. The pinned loop remains otherwise unchanged.
"""
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from verl.trainer.ppo import ray_trainer

from math_rl.completion_dataset import CompletionDataset
from math_rl.critic_replay import CriticReplay
from math_rl.online_critic import cross_fitted_values
from math_rl.provenance import write_json


class MathTrainer(ray_trainer.RayPPOTrainer):
    def _validate_config(self):
        # The parent infers use_critic from GAE. Linear PPO supplies values itself;
        # disable the transformer before either config checks or worker creation.
        if self.config.experiment.method in ('lstd', 'ridge'):
            self.use_critic = False
        super()._validate_config()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = self.config.experiment.method
        self.linear = self.method in ('lstd', 'ridge')
        self.output = Path(self.config.trainer.default_local_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.completed_updates = 0

    def fit(self):
        original = ray_trainer.compute_advantage
        capacity = getattr(self.config.experiment, 'replay_capacity', 0)
        model_config = getattr(getattr(self.config, 'actor_rollout_ref', None), 'model', None)
        feature_mode = getattr(model_config, 'critic_feature_mode', 'actor')
        replay = None
        cumulative = None
        original_log_prob = None
        if getattr(self.config.experiment, 'critic_statistics', 'batch') == 'cumulative':
            from datasets import Dataset
            from math_rl.cumulative_lstd import CumulativeLSTD
            rows = Dataset.from_parquet(str(self.config.data.train_files))
            questions = [str(row['extra_info']['prompt_id']) for row in rows]
            cumulative = CumulativeLSTD(questions, self.config.data.seed,
                self.config.experiment.cumulative_epsilon, self.config.experiment.critic_lambda,
                self.config.experiment.statistics_reset_interval, method=self.config.experiment.method)
            original_log_prob = self.actor_rollout_wg.compute_log_prob
            def log_prob(data):
                data.meta_info['critic_update_step'] = int(self.global_steps)
                data.meta_info['critic_refresh_interval'] = self.config.experiment.encoder_refresh_interval
                return original_log_prob(data)
            self.actor_rollout_wg.compute_log_prob = log_prob
        if self.linear and capacity:
            if feature_mode != 'frozen':
                raise ValueError('Replay requires immutable frozen features')
            replay = CriticReplay(capacity, self.config.experiment.replay_max_transitions,
                                  self.config.experiment.replay_max_age)

        def advantages(data, *args, **kwargs):
            if self.linear:
                device = data.batch['token_level_rewards'].device
                features = data.non_tensor_batch.pop('pretoken_features')
                questions = [str(info['prompt_id']) for info in data.non_tensor_batch['extra_info']]
                if cumulative is not None:
                    from math_rl.cumulative_lstd import encoder_epoch
                    epoch = encoder_epoch(self.global_steps, self.config.experiment.encoder_refresh_interval)
                    versions = data.non_tensor_batch.pop('critic_encoder_epoch')
                    if any(int(v) != epoch for v in versions):
                        raise RuntimeError('Workers returned features from a different encoder epoch')
                    values, heads, metrics = cumulative.update(features,
                        data.batch['token_level_rewards'], data.batch['response_mask'],
                        questions, self.global_steps, epoch)
                else:
                    values, heads, metrics = cross_fitted_values(features,
                        data.batch['token_level_rewards'], data.batch['response_mask'], questions,
                        self.method, alpha=self.config.experiment.critic_alpha,
                        trace_lambda=self.config.experiment.critic_lambda,
                        seed=self.config.data.seed + self.global_steps, replay=replay, step=self.global_steps)
                metrics['feature_mode'] = feature_mode
                data.batch['values'] = values.to(device)
                folder = self.output / 'linear-critic'
                folder.mkdir(exist_ok=True)
                torch.save(dict(heads=heads, step=self.global_steps, method=self.method,
                    feature_policy=feature_mode,
                    replay=metrics.get('replay')),
                    folder / f'{self.global_steps}.pt')
                if cumulative is not None:
                    temporary = folder / 'statistics.tmp'
                    torch.save(cumulative.state_dict(), temporary)
                    temporary.replace(folder / 'statistics.pt')
                with (folder / 'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(metrics, step=self.global_steps)) + '\n')
            result = original(data, *args, **kwargs)
            if not all(torch.isfinite(result.batch[k]).all() for k in ('advantages', 'returns')):
                raise RuntimeError('Nonfinite advantages or returns')
            return result

        # A single trainer runs in its own Ray driver process. Restore the module
        # function even on failure; never patch GAE or the PPO update equations.
        ray_trainer.compute_advantage = advantages
        try:
            super().fit()
        finally:
            ray_trainer.compute_advantage = original
            if original_log_prob is not None:
                self.actor_rollout_wg.compute_log_prob = original_log_prob
        self.completed_updates = self.global_steps - 1
        if self.completed_updates != self.total_training_steps:
            raise RuntimeError('Training ended before the requested update budget')

    def evaluate_test(self, label):
        """Fixed greedy test; scores never enter optimization or model selection."""
        from verl.utils.dataset.rl_dataset import collate_fn
        dataset = CompletionDataset(self.config.experiment.test_file, self.tokenizer, self.config.data)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        old_loader, old_reward = self.val_dataloader, self.val_reward_fn
        old_dir = self.config.trainer.validation_data_dir
        rows = []

        def record(batch, return_dict=False):
            result = old_reward(batch, return_dict=True)
            scores = result['reward_tensor'].sum(-1).cpu().tolist()
            extra = result.get('reward_extra_info', {})
            width = batch.batch['responses'].shape[1]
            counts = batch.batch['attention_mask'][:, -width:].sum(-1).long().cpu().tolist()
            for i, (score, count) in enumerate(zip(scores, counts, strict=True)):
                info = batch.non_tensor_batch['extra_info'][i]
                tokens = batch.batch['responses'][i, :count].cpu().tolist()
                rows.append(dict(id=info['prompt_id'], level=int(info['level']),
                    ground_truth=batch.non_tensor_batch['reward_model'][i]['ground_truth'],
                    response=self.tokenizer.decode(tokens, skip_special_tokens=True), score=float(score),
                    response_tokens=count,
                    capped=count == self.config.data.max_response_length and bool(tokens)
                        and tokens[-1] != self.tokenizer.eos_token_id,
                    verifier_status=str(extra['verifier_status'][i]), extracted=str(extra['extracted'][i])))
            return result if return_dict else result['reward_tensor']

        started = time.perf_counter()
        self.val_dataloader, self.val_reward_fn = loader, record
        self.config.trainer.validation_data_dir = None
        self.global_steps = self.completed_updates
        try:
            self._validate()
        finally:
            self.val_dataloader, self.val_reward_fn = old_loader, old_reward
            self.config.trainer.validation_data_dir = old_dir
        expected = [r['extra_info']['prompt_id'] for r in dataset.rows]
        if [r['id'] for r in rows] != expected:
            raise RuntimeError('Test output order/count differs from the frozen question list')
        with (self.output / f'{label}-test.jsonl').open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row) + '\n')
        result = dict(accuracy=sum(r['score'] for r in rows) / len(rows), questions=len(rows),
                      truncation_rate=sum(r['capped'] for r in rows) / len(rows),
                      seconds=time.perf_counter() - started)
        write_json(self.output / f'{label}-test.json', result)
        return result
