"""Numerical checks against the pinned VERL functions used by the pilot."""
from pathlib import Path


def checkpoint_complete(path, world_size, roles=("actor", "critic")):
    path = Path(path)
    files = [path / "data.pt"]
    for role in roles:
        for rank in range(world_size):
            for kind in ("model", "optim", "extra_state"):
                files.append(path / role / f"{kind}_world_size_{world_size}_rank_{rank}.pt")
    return all(p.is_file() and p.stat().st_size > 0 for p in files)


def grpo_numerical_checks():
    import numpy as np
    import torch
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    # Interleaved prompt groups: balancing across GPUs must not mix the baselines.
    # Group A rewards [0,1,0,1]; B all correct; C all incorrect.
    groups = np.array(["a", "b", "c"] * 4)
    outcome = torch.tensor([0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 1., 0.])
    mask = torch.ones(12, 3)
    mask[::2, -1] = 0
    rewards = torch.zeros_like(mask)
    rewards[torch.arange(12), mask.sum(-1).long() - 1] = outcome
    advantages, _ = compute_grpo_outcome_advantage(rewards, mask, groups)
    expected = torch.zeros_like(mask)
    scale = .5 / (torch.tensor(1. / 3.).sqrt() + 1e-6)
    expected[[0, 6]] = -scale * mask[[0, 6]]
    expected[[3, 9]] = scale * mask[[3, 9]]
    torch.testing.assert_close(advantages, expected)
    assert not advantages.requires_grad
    print("GRPO numerical checks passed: within-prompt normalization, mixed/equal rewards, padding")


def numerical_checks():
    import torch
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return, compute_policy_loss, compute_value_loss
    from verl.utils.torch_functional import masked_whiten

    # A three-token response and a two-token response with right padding.
    # V is evaluated BEFORE each action. EOS/budget exhaustion both terminate.
    rewards = torch.tensor([[0., 0., 1.], [0., 1., 0.]])
    values = torch.tensor([[.2, .3, .1], [.4, .2, 0.]], requires_grad=True)
    mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    advantages, returns = compute_gae_advantage_return(rewards, values, mask, .9, .8)
    expected = torch.tensor([[.58536, .738, 1.], [.756, 1., 0.]])
    torch.testing.assert_close(returns, expected)
    torch.testing.assert_close(advantages, masked_whiten(expected - values.detach(), mask))
    assert not advantages.requires_grad and not returns.requires_grad
    _, mc_returns = compute_gae_advantage_return(rewards, values, mask, 1., 1.)
    torch.testing.assert_close(mc_returns, mask)

    # At identical policies, ratio=1 and clipping/KL are zero.
    old = torch.zeros(1, 2)
    adv = torch.tensor([[1., -1.]])
    full_mask = torch.ones_like(adv)
    loss, clipped, kl, _ = compute_policy_loss(old, old, adv, full_mask,
                                               cliprange=.2, clip_ratio_c=1e9)
    torch.testing.assert_close(torch.stack([loss, clipped, kl]), torch.zeros(3))
    logp = torch.tensor([[1.5, .5]]).log().requires_grad_()
    loss, _, _, _ = compute_policy_loss(old, logp, adv, full_mask,
                                        cliprange=.2, clip_ratio_c=1e9)
    torch.testing.assert_close(loss, torch.tensor(-.2))
    loss.backward()
    torch.testing.assert_close(logp.grad, torch.zeros_like(logp))

    # Fit fixed, detached returns with the actual value loss; actor stays isolated.
    actor = torch.nn.Parameter(torch.zeros(1, 2))
    critic = torch.nn.Parameter(torch.zeros(1, 2))
    target = torch.tensor([[.2, .4]])
    optimizer = torch.optim.Adam([critic], lr=.03)
    def value_loss():
        return compute_value_loss(critic, target, torch.zeros_like(critic), full_mask,
                                  cliprange_value=.5)[0]
    initial = value_loss().item()
    for _ in range(30):
        optimizer.zero_grad()
        value_loss().backward()
        optimizer.step()
    assert value_loss().item() < initial / 10
    assert actor.grad is None and optimizer.state
    optimizer.zero_grad(set_to_none=True)
    policy_loss = compute_policy_loss(old, actor, adv, full_mask,
                                      cliprange=.2, clip_ratio_c=1e9)[0]
    policy_loss.backward()
    assert actor.grad.abs().sum() > 0 and critic.grad is None
    print("PPO numerical checks passed: GAE, masks, detached returns, clipping, critic fit, gradient isolation")
