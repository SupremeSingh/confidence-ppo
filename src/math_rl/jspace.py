"""Our sparse reconstruction extension to Anthropic's Jacobian Lens.

The upstream package computes J and vocabulary readouts, not sparse decomposition.
Here nonnegative matching pursuit approximates a J-space component in the original
hidden coordinates. This is not an exact projection or the paper's gradient pursuit.
"""
import torch

JLENS_COMMIT = '581d398613e5602a5af361e1c34d3a92ea82ba8e'
REPRESENTATIONS = ('final', 'two_thirds', 'jspace', 'jspace_final',
                   'two_thirds_final', 'residual', 'random', 'random_final')


def geometric_layers(n_layers, base=2):
    """Zero-based blocks ordered by 1, base, base**2, ... from the end.

    For 28 blocks/base 2: indices [27, 26, 24, 20, 12], i.e. blocks
    [28, 27, 25, 21, 13]. The final block is always first.
    """
    if n_layers < 1 or base < 2:
        raise ValueError('Positive layer count and integer base >= 2 required')
    layers, distance = [], 1
    while distance <= n_layers:
        layers.append(n_layers-distance)
        distance *= base
    return layers


def load_readout_weights(model_path):
    """Resolve the actual Qwen readout through HF, including tied/sharded weights.

    CPU loading only; no forward pass or feature regeneration. Looking solely for
    an explicit lm_head tensor in safetensors misses valid shared-weight exports.
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation='eager', device_map=None).cpu().eval()
    head = model.get_output_embeddings()
    if head is None or getattr(head, 'bias', None) is not None:
        raise ValueError('Expected an unbiased output embedding')
    norm = getattr(getattr(model, 'model', None), 'norm', None)
    if norm is None or not hasattr(norm, 'weight'):
        raise ValueError('Expected Qwen final RMSNorm')
    unembedding = head.weight.detach().cpu()
    gain = norm.weight.detach().cpu()
    if unembedding.ndim != 2 or gain.shape != (unembedding.shape[1],):
        raise ValueError('Incompatible output embedding and final norm')
    if not torch.isfinite(unembedding).all() or not torch.isfinite(gain).all():
        raise ValueError('Nonfinite readout weights')
    return {'lm_head.weight': unembedding, 'model.norm.weight': gain}


def vocabulary_directions(unembedding, norm_weight, jacobian):
    """Qwen RMSNorm gain folded into U; the state-dependent RMS divisor is scalar.

    Rows are directions in *source block-output coordinates*. U diag(g) J h
    agrees with official lens logits up to a positive per-state RMS divisor.
    No final norm is applied to h before reconstruction.
    """
    if unembedding.ndim != 2 or norm_weight.shape != (unembedding.shape[1],):
        raise ValueError('Invalid unembedding/norm dimensions')
    if jacobian.shape != (unembedding.shape[1], unembedding.shape[1]):
        raise ValueError('Invalid Jacobian dimensions')
    directions = (unembedding.float() * norm_weight.float()) @ jacobian.float()
    if not torch.isfinite(directions).all():
        raise ValueError('Nonfinite J-lens directions')
    norms = directions.norm(dim=1, keepdim=True)
    # Zero rows cannot contribute to positive pursuit; keep them zero.
    return directions / norms.clamp_min(1e-12)


def rotated_dictionary(dictionary, seed=913):
    """Fixed signed-coordinate permutation: preserves all atom norms/inner products.

    A geometry-matched random-orientation control, not a Haar-uniform rotation.
    Token labels in this dictionary carry no semantic interpretation.
    """
    rng = torch.Generator(device='cpu').manual_seed(seed)
    order = torch.randperm(dictionary.shape[1], generator=rng).to(dictionary.device)
    signs = (torch.randint(0, 2, (dictionary.shape[1],), generator=rng)*2-1).to(dictionary)
    return dictionary[:, order] * signs


@torch.no_grad()
def reconstruct(hidden, dictionary, steps=16):
    """At most `steps` nonnegative atoms, unit rows, greedy residual line search.

    Repeated atoms are allowed and their positive coefficients add. Each step
    cannot increase squared reconstruction error (up to roundoff). Zero residual
    or nonpositive correlations produce zero additions. Not a global optimum.
    """
    if steps < 1 or hidden.ndim != 2 or hidden.shape[1] != dictionary.shape[1]:
        raise ValueError('Invalid reconstruction shape/budget')
    if not torch.isfinite(hidden).all():
        raise ValueError('Nonfinite hidden features')
    residual = hidden.float().clone()
    result = torch.zeros_like(residual)
    ids, weights = [], []
    for _ in range(steps):
        scores = residual @ dictionary.T
        coefficient, index = scores.max(dim=1)
        coefficient = coefficient.clamp_min(0)
        atom = dictionary[index]
        # Guard against small norm drift in floating point.
        coefficient = coefficient / atom.square().sum(1).clamp_min(1e-12)
        addition = coefficient[:, None] * atom
        result += addition
        residual -= addition
        ids.append(index)
        weights.append(coefficient)
    return result, torch.stack(ids, 1), torch.stack(weights, 1)


def representations(two_thirds, final, jspace, random):
    return dict(final=final, two_thirds=two_thirds, jspace=jspace,
                jspace_final=torch.cat((jspace, final), 1),
                two_thirds_final=torch.cat((two_thirds, final), 1),
                residual=two_thirds-jspace, random=random,
                random_final=torch.cat((random, final), 1))


@torch.no_grad()
def pursuit_path(hidden, dictionary, budgets=(16, 25, 64), method='gradient', refinements=4):
    """Sparse nonnegative reconstruction at several budgets in a single pass.

    Our gradient-pursuit variant: select a new positive-correlation atom, then
    update all active coefficients along their projected gradient with backtracking.
    Not upstream code, an exact cone projection, or a bitwise paper reproduction.
    Matching preserves the original algorithm (including repeated atoms).
    """
    budgets = sorted(set(budgets))
    if (not budgets or budgets[0] < 1 or refinements < 1
            or method not in ('gradient', 'matching') or hidden.ndim != 2
            or dictionary.ndim != 2 or hidden.shape[1] != dictionary.shape[1]
            or budgets[-1] > len(dictionary)):
        raise ValueError('Invalid pursuit configuration')
    if not torch.isfinite(hidden).all() or not torch.isfinite(dictionary).all():
        raise ValueError('Nonfinite pursuit inputs')
    h = hidden.float()
    dictionary = dictionary.float()
    result = torch.zeros_like(h)
    ids = torch.empty((len(h), 0), dtype=torch.long, device=h.device)
    weights = torch.empty((len(h), 0), device=h.device)
    snapshots = {}
    for step in range(1, budgets[-1] + 1):
        residual = h - result
        scores = residual @ dictionary.T
        if method == 'gradient' and ids.shape[1]:
            scores.scatter_(1, ids, -torch.inf)
        correlation, index = scores.max(1)
        atom = dictionary[index]
        weight = correlation.clamp_min(0) / atom.square().sum(1).clamp_min(1e-12)
        ids = torch.cat((ids, index[:, None]), 1)
        weights = torch.cat((weights, weight[:, None]), 1)
        result = result + weight[:, None] * atom
        if method == 'gradient':
            atoms = dictionary[ids]
            for _ in range(refinements):
                residual = h - result
                gradient = torch.einsum('bkd,bd->bk', atoms, residual)
                gradient = torch.where((weights > 0) | (gradient > 0), gradient, 0.)
                direction = torch.einsum('bk,bkd->bd', gradient, atoms)
                rate = gradient.square().sum(1) / direction.square().sum(1).clamp_min(1e-20)
                old_error = residual.square().sum(1)
                accepted = torch.zeros(len(h), dtype=torch.bool, device=h.device)
                for _ in range(16):
                    candidate_weights = (weights + rate[:, None] * gradient).clamp_min(0)
                    candidate = torch.einsum('bk,bkd->bd', candidate_weights, atoms)
                    good = (~accepted) & ((h-candidate).square().sum(1) <= old_error)
                    weights = torch.where(good[:, None], candidate_weights, weights)
                    result = torch.where(good[:, None], candidate, result)
                    accepted |= good
                    if accepted.all():
                        break
                    rate = torch.where(accepted, rate, rate * .5)
        if step in budgets:
            snapshots[step] = (result.clone(), ids.clone(), weights.clone())
    return snapshots
