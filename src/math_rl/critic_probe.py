"""Frozen-feature value probes. No actor training and no TD/LSTD fitting here."""
import math
import time

import numpy as np
import torch
from torch import nn

HEADS = ("linear_logit", "mlp2", "resnet10", "linear_value")
PREFIX_SAMPLING = dict(version="random-prefix-v1", seed=1729, per_response=8,
    rule="L=0 once; seven iid uniform draws with replacement from 1..T-1 (0 if T=1). "
         "T is realized response length; no terminal state. Equal response weight; reward-blind.")


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, x):
        return x + self.net(x) / 2


def make_head(kind, dimension, width=256):
    if kind in ("linear_logit", "linear_value"):
        return nn.Linear(dimension, 1)
    if kind == "mlp2":
        return nn.Sequential(nn.Linear(dimension, width), nn.GELU(), nn.Linear(width, 1))
    if kind == "resnet10":
        return nn.Sequential(nn.Linear(dimension, width), nn.GELU(),
                             *(ResidualBlock(width) for _ in range(4)), nn.Linear(width, 1))
    raise ValueError(f"Unknown head: {kind}")


def prefix_states(backbone, token_ids, prompt_length, response_length):
    """T+1 post-normalization vectors: pre-action states 0..T-1 and terminal context T."""
    if prompt_length < 1 or response_length < 1 or token_ids.shape[1] != prompt_length + response_length:
        raise ValueError("Invalid prompt/response boundary")
    with torch.inference_mode():
        hidden = backbone(input_ids=token_ids, attention_mask=torch.ones_like(token_ids),
                          use_cache=False, return_dict=True).last_hidden_state
        states = hidden[0, prompt_length - 1:prompt_length + response_length].detach().cpu().clone()
    if states.shape[0] != response_length + 1 or not torch.isfinite(states).all():
        raise ValueError("Incomplete or nonfinite prefix features")
    return states


def transitions(states, reward):
    """LSTD-ready alignment; the last response action receives R and ends the episode."""
    count = len(states) - 1
    if count < 1 or reward not in (0, 1):
        raise ValueError("Invalid trajectory")
    rewards = torch.zeros(count)
    rewards[-1] = reward
    done = torch.zeros(count, dtype=torch.bool)
    done[-1] = True
    following = states[1:].clone()
    following[-1] = 0  # Never bootstrap from post-terminal context, including at the cap.
    return states[:-1], following, rewards, done


def sample_shard(shard, question_index):
    """Stratified random prefixes; repeatable per question/response, independent of rewards."""
    xs, ys, positions = [], [], []
    for i, reward in enumerate(shard["rewards"]):
        start, stop = shard["offsets"][i:i + 2].tolist()
        length = stop - start - 1
        if length < 1:
            raise ValueError("Empty trajectory")
        response_index = int(shard['response_indices'][i]) if 'response_indices' in shard else i
        rng = np.random.default_rng(np.random.SeedSequence([PREFIX_SAMPLING["seed"], question_index, response_index]))
        draws = PREFIX_SAMPLING["per_response"] - 1
        selected = [0] + (rng.integers(1, length, size=draws).tolist() if length > 1 else [0] * draws)
        xs.append(shard["features"][start + torch.tensor(selected)])
        ys.extend([float(reward)] * len(selected))
        positions.extend(selected)
    x = torch.cat(xs)
    return dict(x=x, y=torch.tensor(ys), position=torch.tensor(positions),
                question=torch.full((len(ys),), question_index, dtype=torch.long))


def auc(y, prediction):
    positive = y == 1
    npos, nneg = positive.sum(), (~positive).sum()
    if not npos or not nneg:
        return None
    order = np.argsort(prediction, kind="stable")
    _, start, counts = np.unique(prediction[order], return_index=True, return_counts=True)
    ranks = np.empty(len(y))
    ranks[order] = np.repeat(start + (counts + 1) / 2, counts)
    return float((ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if not len(y) or y.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Empty, mismatched or nonfinite predictions")
    p = np.clip(prediction, 1e-7, 1 - 1e-7)
    calibration = []
    bins = np.minimum((np.clip(prediction, 0, 1) * 10).astype(int), 9)
    for i in range(10):
        mask = bins == i
        if mask.any():
            calibration.append(dict(count=int(mask.sum()), prediction=float(prediction[mask].mean()),
                                    success_rate=float(y[mask].mean())))
    return dict(n=len(y), accuracy=float(((prediction >= .5) == y).mean()),
                brier=float(np.mean((prediction - y) ** 2)),
                cross_entropy=float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
                auroc=auc(y, prediction), outside_probability_range=float(((prediction < 0) | (prediction > 1)).mean()),
                calibration=calibration,
                ece=sum(b["count"] * abs(b["prediction"] - b["success_rate"]) for b in calibration) / len(y))


def assessment(data, prediction, baseline):
    y, q, pos = (data[k].numpy() for k in ("y", "question", "position"))
    result = metrics(y, prediction)
    masks = {"question_only": pos == 0, "early_1_32": (pos > 0) & (pos <= 32),
         "middle_33_128": (pos > 32) & (pos <= 128), "late_129_plus": pos > 128}
    result["by_prefix"] = {name: metrics(y[mask], prediction[mask]) for name, mask in masks.items() if mask.any()}
    result["constant_by_prefix"] = {name: metrics(y[mask], np.full(mask.sum(), baseline))
                                     for name, mask in masks.items() if mask.any()}
    # Cluster uncertainty by question: all its responses and prefixes move together.
    losses = (prediction - y) ** 2 - (baseline - y) ** 2
    _, group = np.unique(q, return_inverse=True)
    per_question = np.bincount(group, weights=losses) / np.bincount(group)
    rng = np.random.default_rng(42)
    samples = [rng.choice(per_question, len(per_question), replace=True).mean() for _ in range(1000)]
    result["question_mean_brier_difference_vs_constant"] = float(per_question.mean())
    result["question_bootstrap_95pct_interval"] = np.quantile(samples, [.025, .975]).tolist()
    return result


def predict(head, kind, data, mean, scale, device, batch_size=512):
    head.eval()
    values = []
    with torch.inference_mode():
        for batch in data["x"].split(batch_size):
            output = head((batch.to(device).float() - mean) / scale).flatten()
            values.append((output if kind == "linear_value" else output.sigmoid()).cpu())
    return torch.cat(values).numpy()


def fit_trial(kind, train, val, mean, scale, device, seed, lr, epochs, width=256, patience=30, min_epochs=30):
    if epochs < 1 or patience < 1 or min_epochs < 1 or lr <= 0:
        raise ValueError("Fitting budgets and learning rate must be positive")
    torch.manual_seed(seed)
    head = make_head(kind, train["x"].shape[1], width).to(device)
    # Start every output at the empirical training prior, not near zero success.
    prior = float(train["y"].mean().clamp(1e-5, 1 - 1e-5))
    output = [m for m in head.modules() if isinstance(m, nn.Linear)][-1]
    with torch.no_grad():
        output.weight.zero_()
        output.bias.fill_(prior if kind == "linear_value" else math.log(prior / (1 - prior)))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss() if kind == "linear_value" else nn.BCEWithLogitsLoss()
    rng = torch.Generator().manual_seed(seed)
    initial = predict(head, kind, val, mean, scale, device)
    best = float(np.mean((initial - val["y"].numpy()) ** 2))
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    curve = [dict(epoch=0, seconds=0., validation_brier=best)]
    best_epoch, stale = 0, 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        head.train()
        total_loss = torch.zeros((), device=device)
        for indices in torch.randperm(len(train["y"]), generator=rng).split(512):
            x, y = train["x"][indices].to(device).float(), train["y"][indices].to(device)
            loss = criterion(head((x - mean) / scale).flatten(), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {kind}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(indices)
        prediction = predict(head, kind, val, mean, scale, device)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Nonfinite validation predictions in {kind}")
        error = float(np.mean((prediction - val["y"].numpy()) ** 2))
        curve.append(dict(epoch=epoch + 1, seconds=time.perf_counter() - started, validation_brier=error,
                          train_loss=float(total_loss.cpu()) / len(train["y"])))
        stale += 1
        if error < best - 1e-6:
            best_epoch, stale = epoch + 1, 0
            best = error
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"{kind} seed={seed} lr={lr}: epoch={epoch + 1}, "
                  f"validation Brier={error:.5f}, best={best:.5f} at epoch {best_epoch}", flush=True)
        if epoch + 1 >= min_epochs and stale >= patience:
            break
    return dict(state=best_state, kind=kind, seed=seed, lr=lr, width=width, validation_brier=best,
                best_epoch=best_epoch, epochs_run=len(curve) - 1,
                stopped_early=len(curve) - 1 < epochs, best_at_budget_limit=best_epoch == epochs,
                curve=curve, fitting_seconds=time.perf_counter() - started,
                parameters=sum(p.numel() for p in head.parameters()),
                peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None)


def fit_ridge(train, val, mean, scale, alphas=(1e-5, 1e-4, 1e-3, .01, .1, 1., 10.)):
    """Direct float64 solve of mean squared return error + alpha*||w||^2; bias unpenalized.

    Supervised Monte Carlo regression, NOT LSTD. Accumulate on CPU in batches to
    avoid a full float64 copy of a large feature cache. Select alpha on validation.
    """
    started = time.perf_counter()
    mean, scale = mean.cpu().double(), scale.cpu().double()
    dimension = train["x"].shape[1]
    gram = torch.zeros(dimension + 1, dimension + 1, dtype=torch.float64)
    rhs = torch.zeros(dimension + 1, dtype=torch.float64)
    for start in range(0, len(train["y"]), 4096):
        x = (train["x"][start:start + 4096].double() - mean) / scale
        design = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], dim=1)
        gram += design.T @ design
        rhs += design.T @ train["y"][start:start + 4096].double()
    gram /= len(train["y"])
    rhs /= len(train["y"])
    penalty = torch.eye(dimension + 1, dtype=gram.dtype)
    penalty[-1, -1] = 0
    trials, best = [], None
    for alpha in alphas:
        if alpha <= 0:
            raise ValueError("Ridge regularization must be positive")
        system = gram + alpha * penalty
        weights = torch.linalg.solve(system, rhs)
        head = make_head("linear_value", dimension)
        with torch.no_grad():
            head.weight.copy_(weights[:-1].float()[None])
            head.bias.copy_(weights[-1:].float())
        prediction = predict(head, "linear_value", val, mean.float(), scale.float(), torch.device("cpu"))
        error = float(np.mean((prediction - val["y"].numpy()) ** 2))
        if not np.isfinite(error):
            raise RuntimeError("Nonfinite ridge result")
        trial = dict(alpha=alpha, validation_brier=error,
                     equation_residual=float(torch.linalg.vector_norm(system @ weights - rhs)))
        trials.append(trial)
        if best is None or error < best["validation_brier"]:
            best = dict(**trial, state=head.state_dict())
    return dict(**best, trials=trials, fitting_seconds=time.perf_counter() - started)
