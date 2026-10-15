"""Synthetic and bundled datasets with exactly controlled empirical covariance."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

DTYPE = torch.float64


def _orthonormal_columns(n: int, d: int, gen: torch.Generator) -> Tensor:
    """``(n, d)`` matrix with exactly orthonormal columns (``n >= d``), Haar-random."""
    if n < d:
        raise ValueError(f"need n >= d, got n={n}, d={d}")
    G = torch.randn(n, d, generator=gen, dtype=DTYPE)
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    return U @ Vh  # random orthonormal frame, same column space as G


def make_whitened_inputs(n: int, d: int, seed: int = 0) -> Tensor:
    """Gaussian inputs ``X (n, d)`` whitened so that ``X^T X / n = I_d`` to machine precision."""
    gen = torch.Generator().manual_seed(int(seed))
    return np.sqrt(n) * _orthonormal_columns(n, d, gen)


def make_covariance_imbalanced_inputs(
    n: int, d: int, spectrum: Union[Sequence[float], np.ndarray, Tensor], seed: int = 0
) -> Tensor:
    """Inputs ``X (n, d)`` whose empirical covariance ``X^T X / n`` equals ``diag(spectrum)`` exactly."""
    spec = torch.as_tensor(np.asarray(spectrum, dtype=np.float64))
    if spec.shape != (d,):
        raise ValueError(f"spectrum must have length d={d}")
    if (spec < 0).any():
        raise ValueError("spectrum must be non-negative")
    gen = torch.Generator().manual_seed(int(seed))
    return np.sqrt(n) * _orthonormal_columns(n, d, gen) * torch.sqrt(spec).unsqueeze(0)


def geometric_spectrum(d: int, ratio: float) -> np.ndarray:
    """``[1, ratio, ratio^2, ...]`` rescaled to mean 1 (same trace as the whitened case)."""
    s = ratio ** np.arange(d, dtype=np.float64)
    return s / s.mean()


def make_rank_r_targets(
    X: Tensor,
    r: int,
    d_out: int,
    seed: int = 0,
    singular_values: Optional[Union[float, Sequence[float]]] = None,
    noise_std: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Rank-``r`` linear targets ``Y = X Wstar^T (+ noise)``.

    ``Wstar = U diag(sv) V^T`` is ``d_out x d`` with Haar-random orthonormal ``U (d_out x r)`` and
    ``V (d x r)``; ``sv`` defaults to ``ones(r)`` (balanced cross-covariance).  A float
    ``singular_values`` means ``ones(r) * c``.
    """
    n, d = X.shape
    if r > min(d, d_out):
        raise ValueError(f"r={r} exceeds min(d_in, d_out)={min(d, d_out)}")
    gen = torch.Generator().manual_seed(int(seed))
    if singular_values is None:
        sv = torch.ones(r, dtype=DTYPE)
    elif isinstance(singular_values, (int, float)):
        sv = torch.full((r,), float(singular_values), dtype=DTYPE)
    else:
        sv = torch.as_tensor(np.asarray(singular_values, dtype=np.float64))
        if sv.shape != (r,):
            raise ValueError("singular_values must have length r")
    U = _orthonormal_columns(d_out, r, gen)
    V = _orthonormal_columns(d, r, gen)
    Wstar = (U * sv.unsqueeze(0)) @ V.T
    Y = X.to(DTYPE) @ Wstar.T
    if noise_std > 0:
        Y = Y + noise_std * torch.randn(n, d_out, generator=gen, dtype=DTYPE)
    return Y, Wstar


def load_digits_dataset(
    d: Optional[int] = None,
    whiten: bool = True,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Dict[str, Tensor]:
    """``sklearn.datasets.load_digits`` (bundled, 8x8 digits, 10 classes), standardized.

    Args:
        d: keep ``d`` PCA components (fit on the training split); ``None`` keeps all 64 pixels.
        whiten: divide the PCA scores by their training standard deviations (PCA whitening);
            ignored when ``d`` is None.
    Returns:
        dict with ``X_train, Y_train, X_test, Y_test`` (one-hot float64 targets) and
        ``y_train, y_test`` (integer labels).
    """
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_frac, random_state=int(seed), stratify=y)
    mean = X_tr.mean(0, keepdims=True)
    std = X_tr.std(0, keepdims=True)
    std[std == 0] = 1.0
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std
    if d is not None:
        pca = PCA(n_components=d, whiten=whiten, random_state=int(seed)).fit(X_tr)
        X_tr = pca.transform(X_tr)
        X_te = pca.transform(X_te)
    n_classes = 10
    return {
        "X_train": torch.as_tensor(X_tr, dtype=DTYPE),
        "Y_train": torch.nn.functional.one_hot(torch.as_tensor(y_tr), n_classes).to(DTYPE),
        "y_train": torch.as_tensor(y_tr),
        "X_test": torch.as_tensor(X_te, dtype=DTYPE),
        "Y_test": torch.nn.functional.one_hot(torch.as_tensor(y_te), n_classes).to(DTYPE),
        "y_test": torch.as_tensor(y_te),
    }
