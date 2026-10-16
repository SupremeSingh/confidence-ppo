"""Lossless compact coordinates for full pre-softmax J-lens readouts.

For B = U diag(g), logits z = B x with x = RMS(Jh). If B.T B = Q diag(s²) Q.T,
phi = diag(s) Q.T x has identical inner products to z. No vocabulary selection,
sparse pursuit, truncated SVD, or variance threshold is used. This equivalence
does NOT apply to fitting a linear head on softmax probabilities.
"""
import torch


@torch.no_grad()
def readout_factor(unembedding, gain, chunk_size=4096):
    dimension = unembedding.shape[1]
    gram = torch.zeros(dimension, dimension, device=unembedding.device, dtype=torch.float64)
    for chunk in unembedding.split(chunk_size):
        b = chunk.double() * gain.double()
        gram += b.T @ b
    gram = gram.cpu()
    eigenvalues, vectors = torch.linalg.eigh((gram+gram.T)/2)
    if not torch.isfinite(eigenvalues).all() or eigenvalues.max() <= 0:
        raise ValueError('Degenerate or nonfinite vocabulary readout')
    if eigenvalues.min() < -1e-10 * eigenvalues.max():
        raise ValueError('Vocabulary Gram is not numerically positive semidefinite')
    values = eigenvalues.clamp_min(0)  # Only roundoff negatives; no positive eigenvalue truncation.
    factor = vectors * values.sqrt()[None, :]
    residual = torch.linalg.vector_norm(factor@factor.T-gram)/torch.linalg.vector_norm(gram)
    if residual > 1e-8:
        raise ValueError('Inaccurate full-readout factorization')
    return dict(factor=factor, eigenvalues=values, factorization_relative_error=float(residual),
                vocabulary_size=len(unembedding), dimension=dimension, truncation='none')


def rms_coordinates(transported, eps):
    return transported * torch.rsqrt(transported.square().mean(-1, keepdim=True)+eps)


def compact_readout(transported, factor, eps):
    return rms_coordinates(transported, eps) @ factor.to(transported)


def isotropic_normalization(x):
    """One training-only scale, preserving the full-readout Euclidean metric."""
    mean = x.mean(0)
    scale = ((x-mean).double().square().mean().sqrt().clamp_min(1e-8)).float()
    return mean, scale.expand_as(mean).clone()
