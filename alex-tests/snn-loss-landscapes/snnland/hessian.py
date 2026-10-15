"""Hessians of functional models in float64: full (surrogate) Hessian, Gauss-Newton part, checks.

All routines take a ``ModelSpec`` (``model``) and a parameter dict, flatten the parameters with
``model.params_to_vec`` and differentiate the scalar loss with ``torch.func``.  Parameter index
``i`` of the flattened vector corresponds to ``model.layer_slices`` for per-layer blocks.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.func import grad, jacrev, jvp, vmap

from .models import ModelSpec, Params

LOSSES = ("mse", "ce")


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def per_output_loss(Yhat: Tensor, Y: Tensor, loss: str) -> Tensor:
    """Loss averaged over the batch (``Yhat, Y`` of shape ``(n, d_out)``)."""
    if loss == "mse":
        return 0.5 * ((Yhat - Y) ** 2).sum() / Yhat.shape[0]
    if loss == "ce":
        logp = torch.log_softmax(Yhat, dim=-1)
        if Y.dtype in (torch.int64, torch.int32):
            return -logp.gather(-1, Y.reshape(-1, 1)).mean()
        return -(Y * logp).sum(-1).mean()
    raise ValueError(f"unknown loss {loss!r}; choose from {LOSSES}")


def loss_fn(model: ModelSpec, params: Params, X: Tensor, Y: Tensor, loss: str = "mse") -> Tensor:
    """``mse``: ``(1/(2n)) ||Yhat - Y||_F^2``; ``ce``: mean cross-entropy (one-hot or int labels)."""
    return per_output_loss(model.forward(params, X), Y, loss)


def make_vec_loss(model: ModelSpec, X: Tensor, Y: Tensor, loss: str = "mse") -> Callable[[Tensor], Tensor]:
    """Scalar loss as a function of the flattened parameter vector."""
    X = X.to(model.dtype)

    def f(vec: Tensor) -> Tensor:
        return per_output_loss(model.forward(model.vec_to_params(vec), X), Y, loss)

    return f


def surrogate_gradient(model: ModelSpec, params: Params, X: Tensor, Y: Tensor, loss: str = "mse") -> Tensor:
    """Flattened (surrogate) gradient of the loss, via ``torch.func.grad``."""
    return grad(make_vec_loss(model, X, Y, loss))(model.params_to_vec(params))


# --------------------------------------------------------------------------- #
# Full Hessian = Jacobian of the gradient field
# --------------------------------------------------------------------------- #
def _hessian_func(f: Callable[[Tensor], Tensor], vec: Tensor, chunk_size: int) -> Tensor:
    """``jacfwd(jacrev(f))`` computed column-chunk by column-chunk (forward-over-reverse)."""
    P = vec.numel()
    grad_f = jacrev(f)
    eye = torch.eye(P, dtype=vec.dtype)

    def column(d: Tensor) -> Tensor:
        return jvp(grad_f, (vec,), (d,))[1]

    cols: List[Tensor] = []
    for start in range(0, P, chunk_size):
        cols.append(vmap(column)(eye[start : start + chunk_size]))
    return torch.cat(cols, 0).T  # H[i, j] = d g_i / d theta_j


def _hessian_jvp_loop(f: Callable[[Tensor], Tensor], vec: Tensor) -> Tensor:
    """Column ``j`` of the Jacobian of the gradient by one forward-over-reverse pass with ``e_j``
    (no ``vmap``)."""
    P = vec.numel()
    grad_f = jacrev(f)
    eye = torch.eye(P, dtype=vec.dtype)
    return torch.stack([jvp(grad_f, (vec,), (eye[j],))[1] for j in range(P)], 1)


def _hessian_hvp(f: Callable[[Tensor], Tensor], vec: Tensor) -> Tensor:
    """Row ``i`` of the Hessian by a double-backward pass with ``e_i`` (reverse-over-reverse)."""
    v = vec.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(f(v), v, create_graph=True)[0]
    P = v.numel()
    rows = []
    for i in range(P):
        e = torch.zeros(P, dtype=v.dtype)
        e[i] = 1.0
        (row,) = torch.autograd.grad(g, v, grad_outputs=e, retain_graph=True, allow_unused=True)
        rows.append(torch.zeros(P, dtype=v.dtype) if row is None else row)
    return torch.stack(rows, 0)  # row i = J^T e_i = (d g_i / d theta)


HESSIAN_METHODS = ("auto", "func", "jvp_loop", "hvp")


def full_hessian(
    model: ModelSpec,
    params: Params,
    X: Tensor,
    Y: Tensor,
    loss: str = "mse",
    method: str = "auto",
    chunk_size: int = 128,
) -> Tensor:
    """Jacobian ``H[i, j] = d g_i / d theta_j`` of the (surrogate) gradient field ``g``.

    Methods:
      * ``"func"``: ``torch.func`` forward-over-reverse (``jacfwd(jacrev)``), ``vmap`` over
        column chunks of ``chunk_size`` basis directions;
      * ``"jvp_loop"``: the same forward-over-reverse products, one basis direction at a time
        (slower, no ``vmap``);
      * ``"hvp"``: ``P`` double-backward (reverse-over-reverse) products.  Not available for the
        ``snn`` kind: reverse-over-reverse differentiates the forward spike values with ``sg'``
        and returns the symmetric Hessian of a smooth stand-in function instead of ``dg/dtheta``;
      * ``"auto"``: ``"func"``, falling back to ``"jvp_loop"`` (snn) or ``"hvp"`` (other kinds)
        on a ``RuntimeError``.

    The raw matrix is returned.  For ``linear`` / ``relu`` / ``smooth_snn`` it is the ordinary
    (symmetric) Hessian; for ``snn`` it is in general not symmetric because the surrogate
    gradient is not a gradient field (use :func:`symmetrize` / :func:`asymmetry`).  All methods
    agree with central finite differences of ``g`` (:func:`check_hessian_fd`).
    """
    if method not in HESSIAN_METHODS:
        raise ValueError(f"method must be one of {HESSIAN_METHODS}")
    if method == "hvp" and model.kind == "snn":
        raise ValueError("method='hvp' (reverse-over-reverse) is not the Jacobian of the surrogate "
                         "gradient for kind='snn'; use 'func' or 'jvp_loop'")
    f = make_vec_loss(model, X, Y, loss)
    vec = model.params_to_vec(params).detach()
    if method == "func":
        return _hessian_func(f, vec, chunk_size)
    if method == "jvp_loop":
        return _hessian_jvp_loop(f, vec)
    if method == "hvp":
        return _hessian_hvp(f, vec)
    try:
        return _hessian_func(f, vec, chunk_size)
    except RuntimeError:
        return _hessian_jvp_loop(f, vec) if model.kind == "snn" else _hessian_hvp(f, vec)


def symmetrize(H: Tensor) -> Tensor:
    """``(H + H^T) / 2``."""
    return 0.5 * (H + H.T)


def asymmetry(H: Tensor) -> float:
    """``||H - H^T||_F / ||H||_F`` (0 for a symmetric matrix)."""
    denom = float(torch.linalg.norm(H))
    return 0.0 if denom == 0.0 else float(torch.linalg.norm(H - H.T)) / denom


# --------------------------------------------------------------------------- #
# Gauss-Newton part
# --------------------------------------------------------------------------- #
def per_sample_jacobian(model: ModelSpec, params: Params, X: Tensor) -> Tensor:
    """``J (n, d_out, P)``: ``J[i, a, :] = d Yhat[i, a] / d theta`` via ``vmap(jacrev)`` over samples.

    With ``center_features`` the (detached) batch feature mean is computed once from the whole
    batch and passed to every per-sample evaluation, matching the batched forward exactly.
    """
    X = X.to(model.dtype)
    vec = model.params_to_vec(params).detach()
    feature_mean = None
    if model.center_features:
        with torch.no_grad():
            feature_mean = model.hidden_features(params, X).mean(0)

    def f_single(v: Tensor, x: Tensor) -> Tensor:
        return model.forward(model.vec_to_params(v), x, feature_mean=feature_mean)

    return vmap(jacrev(f_single, argnums=0), in_dims=(None, 0))(vec, X)


def gauss_newton(
    model: ModelSpec,
    params: Params,
    X: Tensor,
    Y: Tensor,
    loss: str = "mse",
    return_blocks: bool = False,
    return_J: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, object]]:
    """Gauss-Newton matrix ``H_o = (1/n) sum_i J_i^T (d^2 loss_i / d Yhat_i^2) J_i``.

    For ``mse`` this is ``J^T J / n``; for ``ce`` the output Hessian is
    ``diag(p_i) - p_i p_i^T`` (softmax probabilities).  With ``return_blocks`` / ``return_J`` a
    second dict is returned holding ``"blocks"`` (per-layer diagonal blocks, by
    ``model.layer_slices``) and/or ``"J"`` (the per-sample surrogate Jacobian ``(n, d_out, P)``).
    """
    J = per_sample_jacobian(model, params, X)
    n, d_out, P = J.shape
    if loss == "mse":
        Jf = J.reshape(n * d_out, P)
        H_o = Jf.T @ Jf / n
    elif loss == "ce":
        with torch.no_grad():
            p = torch.softmax(model.forward(params, X.to(model.dtype)), dim=-1)  # (n, d_out)
        W = torch.diag_embed(p) - p.unsqueeze(2) * p.unsqueeze(1)  # (n, d_out, d_out)
        H_o = torch.einsum("iap,iab,ibq->pq", J, W, J) / n
    else:
        raise ValueError(f"unknown loss {loss!r}; choose from {LOSSES}")
    if not (return_blocks or return_J):
        return H_o
    extra: Dict[str, object] = {}
    if return_blocks:
        extra["blocks"] = layer_blocks(H_o, model.layer_slices)
    if return_J:
        extra["J"] = J
    return H_o, extra


def layer_blocks(H: Tensor, layer_slices: List[slice]) -> List[Tensor]:
    """Diagonal blocks ``H[sl, sl]`` for each layer slice."""
    return [H[sl, sl] for sl in layer_slices]


def functional_part(H: Tensor, H_o: Tensor) -> Tensor:
    """``H_f = H - H_o`` (the residual-weighted second-derivative term)."""
    return H - H_o


# --------------------------------------------------------------------------- #
# Eigen-decompositions
# --------------------------------------------------------------------------- #
def eigh(H_sym: Tensor) -> Tuple[Tensor, Tensor]:
    """Eigenvalues (descending) and matching eigenvectors (columns) of a symmetric matrix."""
    evals, evecs = torch.linalg.eigh(H_sym)
    idx = torch.argsort(evals, descending=True)
    return evals[idx], evecs[:, idx]


def eigvals(H: Tensor) -> Tensor:
    """Descending eigenvalues of ``symmetrize(H)``."""
    return eigh(symmetrize(H))[0]


def top_k(H: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """Top-``k`` eigenpairs of ``symmetrize(H)`` (descending); ``scipy.sparse.linalg.eigsh`` for
    large ``P``, dense ``eigh`` when ``k`` is close to ``P``."""
    Hs = symmetrize(H)
    P = Hs.shape[0]
    if k >= P - 1 or P <= 512:
        evals, evecs = eigh(Hs)
        return evals[:k], evecs[:, :k]
    from scipy.sparse.linalg import eigsh

    w, v = eigsh(Hs.detach().numpy(), k=k, which="LA")
    order = np.argsort(w)[::-1]
    return torch.as_tensor(w[order].copy()), torch.as_tensor(v[:, order].copy())


# --------------------------------------------------------------------------- #
# Correctness checks
# --------------------------------------------------------------------------- #
def check_hessian_fd(
    model: ModelSpec,
    params: Params,
    X: Tensor,
    Y: Tensor,
    loss: str = "mse",
    H: Optional[Tensor] = None,
    n_dirs: int = 3,
    eps: float = 1e-5,
    seed: int = 0,
) -> List[float]:
    """Relative errors ``||H d - FD(d)|| / ||FD(d)||`` for random unit directions ``d``, where
    ``FD(d)`` is the central finite difference of the (surrogate) gradient with step ``eps``."""
    f = make_vec_loss(model, X, Y, loss)
    g = grad(f)
    vec = model.params_to_vec(params).detach()
    if H is None:
        H = full_hessian(model, params, X, Y, loss)
    gen = torch.Generator().manual_seed(int(seed))
    errs: List[float] = []
    for _ in range(n_dirs):
        d = torch.randn(vec.numel(), generator=gen, dtype=vec.dtype)
        d = d / torch.linalg.norm(d)
        fd = (g(vec + eps * d) - g(vec - eps * d)) / (2 * eps)
        errs.append(float(torch.linalg.norm(H @ d - fd) / torch.linalg.norm(fd)))
    return errs


def linear_gauss_newton_analytic(model: ModelSpec, params: Params, X: Tensor) -> Tensor:
    """Closed-form ``H_o`` of the deep *linear* model with MSE loss (no bias).

    With ``A_l = W_L ... W_{l+1}``, ``B_l = W_{l-1} ... W_1`` and ``Sigma = X^T X / n`` the block
    ``(l, m)`` in row-major weight coordinates is ``(A_l^T A_m) kron (B_l Sigma B_m^T)``.
    """
    if model.kind != "linear" or model.bias:
        raise ValueError("analytic formula is for the bias-free linear model")
    X = X.to(model.dtype)
    Sigma = X.T @ X / X.shape[0]
    Ws = [params[f"W{l}"] for l in range(1, model.L + 1)]
    A: List[Tensor] = []
    B: List[Tensor] = []
    for l in range(model.L):
        a = torch.eye(model.d_out, dtype=model.dtype)
        for m in range(model.L - 1, l, -1):
            a = a @ Ws[m]
        b = torch.eye(model.d_in, dtype=model.dtype)
        for m in range(l - 1, -1, -1):
            b = b @ Ws[m]
        A.append(a)
        B.append(b)
    H = torch.zeros(model.n_params, model.n_params, dtype=model.dtype)
    for l, sl in enumerate(model.layer_slices):
        for m, sm in enumerate(model.layer_slices):
            H[sl, sm] = torch.kron(A[l].T @ A[m], B[l] @ Sigma @ B[m].T)
    return H
