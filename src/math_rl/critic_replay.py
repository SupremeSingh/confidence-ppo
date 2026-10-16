"""Bounded, recent-policy critic data. Never supplies PPO actor training samples."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Trajectory:
    question: str
    states: torch.Tensor
    reward: float
    step: int


class CriticReplay:
    """FIFO of whole answers on an immutable feature map; no importance correction.

    max_age counts completed policy-update differences. At step t, step t-max_age
    is still eligible. Capacity and token limits apply BEFORE fitting.
    """
    def __init__(self, capacity, max_transitions, max_age):
        if capacity < 1 or max_transitions < 1 or max_age < 0:
            raise ValueError('Invalid replay bounds')
        self.capacity, self.max_transitions, self.max_age = capacity, max_transitions, max_age
        self.entries = []
        self.last_step = -1
        self.evicted = 0
        self.dimension = None

    def append(self, states, rewards, questions, step):
        if step <= self.last_step:
            raise ValueError('Replay expects one fresh batch per increasing update')
        if not states or not (len(states) == len(rewards) == len(questions)):
            raise ValueError('Invalid replay batch')
        if len(states) > self.capacity or sum(len(s) for s in states) > self.max_transitions:
            raise ValueError('Replay limits must accommodate the entire fresh batch')
        dimension = self.dimension if self.dimension is not None else states[0].shape[-1]
        new = []
        for s, r, q in zip(states, rewards, questions, strict=True):
            s = torch.as_tensor(s).detach().cpu().float().clone()
            if s.ndim != 2 or not len(s) or s.shape[1] != dimension or not torch.isfinite(s).all():
                raise ValueError('Invalid or changing replay feature dimension')
            if r not in (0., 1.):
                raise ValueError('Replay needs binary terminal outcomes')
            new.append(Trajectory(str(q), s, float(r), step))
        entries = [e for e in self.entries if step - e.step <= self.max_age] + new
        tokens = sum(len(e.states) for e in entries)
        while len(entries) > self.capacity or tokens > self.max_transitions:
            tokens -= len(entries.pop(0).states)
        self.evicted += len(self.entries) + len(new) - len(entries)
        self.entries, self.dimension, self.last_step = entries, dimension, step

    def metrics(self):
        return dict(policy='recent_uncorrected', capacity_answers=self.capacity,
                    max_transitions=self.max_transitions, max_age_updates=self.max_age,
                    answers=len(self.entries), transitions=sum(len(e.states) for e in self.entries),
                    bytes=sum(e.states.numel() * e.states.element_size() for e in self.entries),
                    historical_answers=sum(e.step < self.last_step for e in self.entries),
                    oldest_age=max((self.last_step-e.step for e in self.entries), default=0),
                    evicted_answers=self.evicted)
