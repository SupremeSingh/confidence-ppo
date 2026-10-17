"""Pinned synchronous actor with optional, independently seeded evaluation."""
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from math_rl.study_checks import sampled_evaluation


class StudyActorWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts):
        if not prompts.meta_info.get('study_sampled_evaluation', False):
            return super().generate_sequences(prompts)
        if not prompts.meta_info.get('validate') or not prompts.meta_info.get('do_sample'):
            raise ValueError('Sampled evaluation must use the validation generation path')
        seeds = prompts.non_tensor_batch['study_sample_seed'].tolist()
        with sampled_evaluation(self.rollout, seeds):
            return super().generate_sequences(prompts)
