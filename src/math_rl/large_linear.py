"""Raw-sum LSTD(.99) and matched Monte Carlo ridge on complete trajectories.

Gamma=1, terminal-only binary reward. Normalization is fixed before accumulation.
No sigmoid, averaging of matrices, or trace carry across answer boundaries.
"""
import torch
from math_rl.lstd import design


def statistics(dimension, device='cpu'):
    k = dimension + 1
    return dict(a=torch.zeros(k, k, dtype=torch.float64, device=device),
        gram=torch.zeros(k, k, dtype=torch.float64, device=device),
        b=torch.zeros(k, dtype=torch.float64, device=device),
        returns_rhs=torch.zeros(k, dtype=torch.float64, device=device),
        transitions=0, trajectories=0)


def accumulate(stats, states, reward, mean, scale, trace_lambda=.99, chunk=512):
    """states[t] is BEFORE action t; no post-terminal context is supplied."""
    if not len(states) or reward not in (0, 1) or not 0 <= trace_lambda <= 1 or chunk < 1:
        raise ValueError('Invalid trajectory or trace configuration')
    if (states.ndim != 2 or mean.shape != (states.shape[1],) or scale.shape != mean.shape
            or stats['a'].shape != (states.shape[1]+1, states.shape[1]+1)):
        raise ValueError('Incompatible fixed feature coordinates')
    if (not torch.isfinite(states).all() or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('Nonfinite states or invalid fixed scale')
    device = stats['a'].device
    previous = torch.zeros(len(mean)+1, dtype=torch.float64, device=device)
    for start in range(0, len(states), chunk):
        stop = min(start+chunk, len(states))
        phi = design(states[start:stop].to(device), mean.to(device), scale.to(device))
        following = torch.empty_like(phi)
        following[:-1] = phi[1:]
        following[-1] = (design(states[stop:stop+1].to(device), mean.to(device), scale.to(device))[0]
                         if stop < len(states) else 0)
        z = phi.clone()
        offset = 1
        while offset < len(z):
            z[offset:] = z[offset:] + trace_lambda**offset*z[:-offset]
            offset *= 2
        powers = trace_lambda**torch.arange(1, len(z)+1, dtype=z.dtype, device=device)
        z += powers[:, None]*previous
        previous = z[-1].clone()
        stats['a'].add_(z.T @ (phi-following))
        if stop == len(states):
            stats['b'].add_(z[-1]*reward)
        stats['gram'].add_(phi.T @ phi)
        stats['returns_rhs'].add_(phi.sum(0)*reward)
    stats['transitions'] += len(states)
    stats['trajectories'] += 1


def solve(stats, method, epsilon=.01):
    if method not in ('ridge', 'lstd-0.99') or epsilon <= 0 or not stats['transitions']:
        raise ValueError('Invalid linear fit')
    a, b = (stats['gram'], stats['returns_rhs']) if method == 'ridge' else (stats['a'], stats['b'])
    system = a.clone()
    system.diagonal().add_(epsilon)  # Includes intercept; exactly epsilon*I on raw sums.
    weights = torch.linalg.solve(system, b)
    residual = torch.linalg.vector_norm(system@weights-b)
    relative = residual/(torch.linalg.matrix_norm(system)*torch.linalg.vector_norm(weights)
                        + torch.linalg.vector_norm(b)).clamp_min(1e-30)
    if not torch.isfinite(weights).all() or not torch.isfinite(relative) or relative > 1e-8:
        raise ValueError('Linear system failed numerical check; no silent fallback')
    return weights, dict(relative_residual=float(relative), weight_norm=float(weights.norm()),
        epsilon=epsilon, transitions=stats['transitions'], trajectories=stats['trajectories'],
        convention='Raw sums plus epsilon*I, including intercept; no matrix normalization')
