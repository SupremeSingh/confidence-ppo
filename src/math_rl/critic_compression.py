"""Training-only PCA and exact affine projection of raw-sum TD statistics."""
import torch


def pca_from_statistics(stats):
    """All-token centered PCA in the source's fixed normalized coordinates.

    Gram includes an intercept. No whitening, reward labels or test data enter PCA.
    """
    gram = stats['gram'].double()
    n = stats['transitions']
    if n < 2 or not torch.isfinite(gram).all():
        raise ValueError('PCA requires finite training statistics and at least two states')
    if not torch.isclose(gram[-1, -1], gram.new_tensor(float(n))):
        raise ValueError('Intercept/count mismatch')
    center = gram[:-1, -1] / n
    covariance = gram[:-1, :-1] / n - torch.outer(center, center)
    covariance = (covariance + covariance.T) / 2
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    tolerance = max(float(eigenvalues.abs().max()), 1.) * 1e-8
    if eigenvalues.min() < -tolerance or eigenvalues.max() <= 0:
        raise ValueError('Invalid covariance spectrum')
    return center, eigenvalues.flip(0).clamp_min(0), vectors.flip(1)


def affine_projection(center, components):
    """Row design phi maps to phi @ T; terminal zero maps to zero too."""
    dimension, rank = components.shape
    if center.shape != (dimension,) or not 0 < rank <= dimension:
        raise ValueError('Invalid projection shape')
    transform = components.new_zeros((dimension + 1, rank + 1))
    transform[:-1, :-1] = components
    transform[-1, :-1] = -center @ components
    transform[-1, -1] = 1
    return transform


def project_statistics(stats, transform):
    """Project BEFORE adding regularization; traces commute with fixed T."""
    return dict(a=transform.T @ stats['a'] @ transform,
                gram=transform.T @ stats['gram'] @ transform,
                b=transform.T @ stats['b'],
                returns_rhs=transform.T @ stats['returns_rhs'],
                transitions=stats['transitions'], trajectories=stats['trajectories'])
