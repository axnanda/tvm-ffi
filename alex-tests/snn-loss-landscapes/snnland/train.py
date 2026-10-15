"""Full-batch training with (surrogate) gradients and Hessian-summary callbacks."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor
from torch.func import grad

from .analysis import cluster_stats, layer_coherence
from .hessian import eigvals, full_hessian, functional_part, gauss_newton, make_vec_loss
from .models import ModelSpec, Params

Callback = Callable[[int, Params, Tensor, float], Dict[str, object]]


def hessian_summary_callback(
    model: ModelSpec, X: Tensor, Y: Tensor, loss: str, r: int, d_in: int, d_out: int, tol: float = 1e-8
) -> Callback:
    """Callback computing ``H``, ``H_o``, ``H_f`` norms, cluster statistics (of the symmetrized
    ``H`` and of ``H_o``) and layer coherence of ``H_o``."""

    def cb(step: int, params: Params, g: Tensor, loss_value: float) -> Dict[str, object]:
        H = full_hessian(model, params, X, Y, loss)
        H_o = gauss_newton(model, params, X, Y, loss)
        H_f = functional_part(H, H_o)
        eig_H = eigvals(H)
        eig_Ho = eigvals(H_o)
        return {
            "H_norm": float(torch.linalg.norm(H)),
            "H_o_norm": float(torch.linalg.norm(H_o)),
            "H_f_norm": float(torch.linalg.norm(H_f)),
            "eig_H": eig_H,
            "eig_H_o": eig_Ho,
            "cluster_H": cluster_stats(eig_H, r, d_in, d_out, tol),
            "cluster_H_o": cluster_stats(eig_Ho, r, d_in, d_out, tol),
            "coherence_H_o": layer_coherence(H_o, model.layer_slices),
        }

    return cb


def gd_train(
    model: ModelSpec,
    params: Params,
    X: Tensor,
    Y: Tensor,
    lr: float,
    steps: int,
    loss: str = "mse",
    log_every: int = 0,
    callbacks: Optional[List[Callback]] = None,
    optimizer: str = "gd",
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
) -> Dict[str, object]:
    """Full-batch gradient descent (or Adam) on the flattened parameters.

    Gradients come from ``torch.func.grad`` of the loss (the surrogate gradient for ``snn``).
    ``history["loss"][t]`` and ``["grad_norm"][t]`` are recorded before update ``t``; at steps
    ``0, log_every, 2 log_every, ...`` (and the final step) every callback is called with
    ``(step, params, grad, loss)`` and its dict stored in ``history["summaries"]``.
    Returns ``{"params": final params, "loss", "grad_norm", "logged_steps", "summaries"}``.
    """
    if optimizer not in ("gd", "adam"):
        raise ValueError("optimizer must be 'gd' or 'adam'")
    f = make_vec_loss(model, X, Y, loss)
    g_fn = grad(f)
    vec = model.params_to_vec(params).detach().clone()
    m = torch.zeros_like(vec)
    v = torch.zeros_like(vec)
    history: Dict[str, object] = {"loss": [], "grad_norm": [], "logged_steps": [], "summaries": []}
    callbacks = callbacks or []
    for step in range(steps + 1):
        loss_value = float(f(vec))
        g = g_fn(vec)
        history["loss"].append(loss_value)
        history["grad_norm"].append(float(torch.linalg.norm(g)))
        if callbacks and ((log_every > 0 and step % log_every == 0) or step == steps):
            p = model.vec_to_params(vec)
            history["logged_steps"].append(step)
            history["summaries"].append({k: v_ for cb in callbacks for k, v_ in cb(step, p, g, loss_value).items()})
        if step == steps:
            break
        if optimizer == "gd":
            vec = vec - lr * g
        else:
            b1, b2 = betas
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            m_hat = m / (1 - b1 ** (step + 1))
            v_hat = v / (1 - b2 ** (step + 1))
            vec = vec - lr * m_hat / (torch.sqrt(v_hat) + eps)
    history["params"] = model.vec_to_params(vec)
    return history
