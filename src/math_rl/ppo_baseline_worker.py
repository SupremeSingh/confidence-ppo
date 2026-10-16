"""VERL conventional critic, with zero initialization and measured fitting checks."""
import json
import time
from pathlib import Path

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import CriticWorker

from math_rl.ppo_baseline import critic_diagnostics, zero_value_head


class AuditedCriticWorker(CriticWorker):
    def _build_critic_model_optimizer(self, config):
        model, optimizer, scheduler = super()._build_critic_model_optimizer(config)
        if not isinstance(model, FSDP):
            raise ValueError('Audited baseline currently requires pinned FSDP1')
        # Before the first optimizer step; retain pretrained transformer weights.
        with FSDP.summon_full_params(model, writeback=True, rank0_only=False):
            zero_value_head(model)
        return model, optimizer, scheduler

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data):
        before = data.batch['values'].detach().cpu().clone()
        targets = data.batch['returns'].detach().cpu().clone()
        width = data.batch['responses'].shape[1]
        mask = data.batch['attention_mask'][:, -width:].detach().cpu().clone()
        result = super().update_critic(data)
        # Extra forward included in measured training cost, also in final runs.
        started = time.perf_counter()
        after = super().compute_values(data).batch['values']
        audit_seconds = time.perf_counter() - started
        row = critic_diagnostics(before, after, targets, mask)
        row['audit_seconds'] = audit_seconds
        metrics = result.meta_info['metrics']
        gradients = torch.as_tensor(metrics['critic/grad_norm']).float()
        if not torch.isfinite(gradients).all():
            raise RuntimeError('Nonfinite conventional critic gradients')
        self._audit_updates = getattr(self, '_audit_updates', 0) + 1
        row.update(update=self._audit_updates, grad_norm_max=float(gradients.max()),
                   lr=float(metrics['critic/lr']))
        directory = Path(self.config.audit_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f'rank-{self.rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        return result
