"""Frozen-policy LSTD(lambda), following Lagoudakis & Parr (2003) and Boyan (2002).

z_t = lambda*z_{t-1} + phi_t; A = mean(z_t (phi_t - phi_next)^T), b = mean(z_t r_t).
Gamma=1 for finite capped responses. Terminal next features, including bias, are zero.
"""
import torch


def design(x, mean, scale):
    x = (x.double() - mean.double()) / scale.double()
    return torch.cat((x, torch.ones((len(x), 1), dtype=x.dtype, device=x.device)), 1)


def empty_statistics(dimension):
    return dict(a=torch.zeros(dimension + 1, dimension + 1, dtype=torch.float64),
                gram=torch.zeros(dimension + 1, dimension + 1, dtype=torch.float64),
                b=torch.zeros(dimension + 1, dtype=torch.float64),
                returns_rhs=torch.zeros(dimension + 1, dtype=torch.float64),
                transitions=0, trajectories=0)


def eligibility(phi, trace_lambda, previous):
    """Parallel prefix scan of the trace recurrence; carry only within one episode."""
    if not 0 <= trace_lambda <= 1:
        raise ValueError('Trace lambda must be in [0, 1]')
    z = phi.clone()
    offset = 1
    while offset < len(z):
        z[offset:] = z[offset:] + trace_lambda ** offset * z[:-offset]
        offset *= 2
    powers = trace_lambda ** torch.arange(1, len(z) + 1, dtype=phi.dtype)
    return z + powers[:, None] * previous


def accumulate(stats, states, outcome, mean, scale, batch_size=4096, trace_lambda=0., include_ridge=True):
    """Every pre-action state once; ridge uses identical state weighting.

    T+1 vectors include terminal context, which is never a training input.
    The last action receives outcome at EOS OR the response cap.
    """
    if len(states) < 2 or outcome not in (0, 1) or batch_size < 1:
        raise ValueError('Invalid trajectory or batch size')
    if not torch.isfinite(states).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Nonfinite features or invalid normalization')
    if not 0 <= trace_lambda <= 1:
        raise ValueError('Trace lambda must be in [0, 1]')
    trace = torch.zeros(len(mean) + 1, dtype=torch.float64)
    count = len(states) - 1
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        phi = design(states[start:stop], mean, scale)
        following = design(states[start + 1:stop + 1], mean, scale)
        reward = torch.zeros(len(phi), dtype=torch.float64)
        if stop == count:
            following[-1] = 0
            reward[-1] = outcome
        z = phi if trace_lambda == 0 else eligibility(phi, trace_lambda, trace)
        trace = z[-1].clone()
        stats['a'] += z.T @ (phi - following)
        stats['b'] += z.T @ reward
        if include_ridge:
            stats['gram'] += phi.T @ phi
            stats['returns_rhs'] += phi.sum(0) * outcome
    stats['transitions'] += count
    stats['trajectories'] += 1


def solve(stats, method, alpha):
    """Regularized fixed-point solve, NOT minimization of squared TD residuals.

    alpha*D shifts the fixed point; D leaves the intercept unpenalized.
    No inverse, sigmoid, clipping or silent pseudoinverse fallback.
    """
    if method not in ('lstd', 'ridge') or alpha < 0 or stats['transitions'] < 1:
        raise ValueError('Invalid solver settings')
    a, b = (stats['a'], stats['b']) if method == 'lstd' else (stats['gram'], stats['returns_rhs'])
    a, b = a / stats['transitions'], b / stats['transitions']
    penalty = torch.eye(len(b), dtype=torch.float64)
    penalty[-1, -1] = 0
    system = a + alpha * penalty
    weights = torch.linalg.solve(system, b)
    if not torch.isfinite(weights).all():
        raise ValueError('Nonfinite solution')
    residual = torch.linalg.vector_norm(system @ weights - b)
    relative = residual / (torch.linalg.matrix_norm(system) * torch.linalg.vector_norm(weights)
                           + torch.linalg.vector_norm(b)).clamp_min(1e-30)
    return weights, dict(equation_residual=float(residual), relative_residual=float(relative),
                        weight_norm=float(torch.linalg.vector_norm(weights)),
                        unregularized_residual=float(torch.linalg.vector_norm(a @ weights - b)))


def predict(weights, data, mean, scale):
    return torch.cat([design(x, mean, scale) @ weights for x in data['x'].split(4096)]).numpy()
