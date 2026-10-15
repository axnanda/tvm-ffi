"""Spectral summaries: cluster statistics, layer coherence, feature second moments."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from .hessian import eigh, symmetrize
from .models import ModelSpec, Params, spike_diagnostics

ArrayLike = Union[Tensor, np.ndarray, List[float]]


def _as_numpy(eigs: ArrayLike) -> np.ndarray:
    if isinstance(eigs, Tensor):
        eigs = eigs.detach().cpu().numpy()
    return np.asarray(eigs, dtype=np.float64).reshape(-1)


def _largest_log_gap(eigs: np.ndarray, start: int, stop: int) -> Optional[int]:
    """Index ``i`` in ``[start, stop)`` maximising ``log(eig_i / eig_{i+1})`` over positive pairs."""
    best, best_gap = None, -np.inf
    for i in range(start, min(stop, len(eigs) - 1)):
        if eigs[i] > 0 and eigs[i + 1] > 0:
            gap = np.log(eigs[i] / eigs[i + 1])
            if gap > best_gap:
                best, best_gap = i, gap
    return best


def cluster_stats(eigs: ArrayLike, r: int, d_in: int, d_out: int, tol: float = 1e-8) -> Dict[str, object]:
    """Dominant / bulk / zero cluster statistics of a descending eigenvalue list.

    The paper predicts ``r^2`` dominant eigenvalues, ``r (d_in + d_out - 2r)`` bulk ones and the
    rest zero.  Besides the statistics at those predicted indices, a data-driven split is
    reported: ``largest_log_gap_index`` is the ``argmax`` of ``log(eig_i / eig_{i+1})`` among
    positive eigenvalues with ``i < min(4 r^2, len)``; the dominant cluster is then
    ``eigs[:index + 1]`` and the bulk runs to the next largest log gap (searched up to
    ``4 * n_bulk_expected`` further eigenvalues).  ``zero_count`` counts ``|eig| < tol * max|eig|``.
    """
    e = _as_numpy(eigs)
    n = len(e)
    n_dom = r * r
    n_bulk = r * (d_in + d_out - 2 * r)
    scale = float(np.max(np.abs(e))) if n else 0.0
    zero_mask = np.abs(e) < tol * scale
    dom = e[:n_dom]
    bulk = e[n_dom : n_dom + n_bulk]
    out: Dict[str, object] = {
        "n_eigs": n,
        "n_dominant_expected": n_dom,
        "n_bulk_expected": n_bulk,
        "dominant_mean": float(dom.mean()) if dom.size else float("nan"),
        "dominant_min": float(dom.min()) if dom.size else float("nan"),
        "dominant_max": float(dom.max()) if dom.size else float("nan"),
        "bulk_mean": float(bulk.mean()) if bulk.size else float("nan"),
        "bulk_min": float(bulk.min()) if bulk.size else float("nan"),
        "bulk_max": float(bulk.max()) if bulk.size else float("nan"),
        "zero_count": int(zero_mask.sum()),
        "rank": int(n - zero_mask.sum()),
        "negative_count": int((e < -tol * scale).sum()),
        "most_negative": float(e.min()) if n else float("nan"),
    }
    out["ratio_paper"] = out["dominant_mean"] / out["bulk_mean"] if bulk.size and out["bulk_mean"] != 0 else float("nan")
    out["ratio_gap"] = float(e[n_dom - 1] / e[n_dom]) if n > n_dom and e[n_dom] != 0 else float("nan")
    idx = _largest_log_gap(e, 0, min(4 * n_dom, n))
    out["largest_log_gap_index"] = idx
    if idx is None:
        out.update(
            split_n_dominant=None, split_n_bulk=None, split_dominant_mean=float("nan"),
            split_bulk_mean=float("nan"), ratio_split=float("nan"),
        )
    else:
        k1 = idx + 1
        idx2 = _largest_log_gap(e, k1, min(k1 + 4 * max(n_bulk, 1), n))
        k2 = idx2 + 1 if idx2 is not None else int(np.sum(e > 0))
        split_dom = e[:k1]
        split_bulk = e[k1:k2]
        out["split_n_dominant"] = k1
        out["split_n_bulk"] = int(k2 - k1)
        out["split_dominant_mean"] = float(split_dom.mean())
        out["split_bulk_mean"] = float(split_bulk.mean()) if split_bulk.size else float("nan")
        out["ratio_split"] = (
            out["split_dominant_mean"] / out["split_bulk_mean"]
            if split_bulk.size and out["split_bulk_mean"] != 0 else float("nan")
        )
    return out


def layer_coherence(H: Tensor, layer_slices: List[slice]) -> Dict[str, object]:
    """``lambda_max(H) / max_l lambda_max(H_ll)`` (``== L`` for the linear balanced-uniform case).

    ``H`` is symmetrized first.  Also returns the top eigenvalue of each diagonal block, the
    per-layer energy ``||v_l||^2`` of the top eigenvector ``v`` and the cross-layer block
    alignment ``tr(B_lm B_lm^T) / sqrt(tr(B_ll B_ll^T) tr(B_mm B_mm^T))``.
    """
    Hs = symmetrize(H)
    evals, evecs = eigh(Hs)
    lam_max = float(evals[0])
    top = evecs[:, 0]
    block_lams = [float(eigh(Hs[sl, sl])[0][0]) for sl in layer_slices]
    energy = [float((top[sl] ** 2).sum()) for sl in layer_slices]
    L = len(layer_slices)
    align = torch.zeros(L, L, dtype=Hs.dtype)
    fro = [float(torch.linalg.norm(Hs[sl, sl])) for sl in layer_slices]
    for l, sl in enumerate(layer_slices):
        for m, sm in enumerate(layer_slices):
            denom = fro[l] * fro[m]
            align[l, m] = float(torch.linalg.norm(Hs[sl, sm])) ** 2 / denom if denom > 0 else float("nan")
    return {
        "lambda_max": lam_max,
        "lambda_max_blocks": block_lams,
        "coherence": lam_max / max(block_lams) if max(block_lams) > 0 else float("nan"),
        "top_eigvec_layer_energy": energy,
        "block_alignment": align,
    }


def top_eigvec_layer_energy(H_sym: Tensor, layer_slices: List[slice], k: int) -> Tensor:
    """``(k, L)`` matrix: fraction of the squared norm of each of the top-``k`` eigenvectors of
    ``H_sym`` (symmetrized defensively) that lies in each layer's parameter slice."""
    _, evecs = eigh(symmetrize(H_sym))
    V = evecs[:, :k]
    energy = torch.stack([(V[sl] ** 2).sum(0) for sl in layer_slices], dim=1)  # (k, L)
    return energy / (V**2).sum(0).unsqueeze(1)


def feature_second_moment_stats(model: ModelSpec, params: Params, X: Tensor) -> Dict[str, object]:
    """Second-moment structure of the last hidden features ``a (n, h)`` fed to the readout.

    Returns ``mu_sq_norm = ||mu||^2`` (``mu = a.mean(0)``), the top-3 eigenvalues of the centred
    covariance ``Cov_a = (a - mu)^T (a - mu) / n`` and of the uncentred second moment
    ``M = a^T a / n``, ``frac_var_along_mu = mu^T Cov_a mu / (||mu||^2 tr Cov_a)``,
    ``frac_second_moment_along_mu = mu^T M mu / (||mu||^2 tr M)`` and
    ``mean_frac_of_second_moment = ||mu||^2 / tr M`` (the share of ``M`` due to the mean).
    """
    with torch.no_grad():
        a = model.hidden_features(params, X.to(model.dtype))
    n = a.shape[0]
    mu = a.mean(0)
    ac = a - mu
    cov = ac.T @ ac / n
    M = a.T @ a / n
    cov_eigs = eigh(cov)[0]
    M_eigs = eigh(M)[0]
    mu_sq = float(mu @ mu)
    tr_cov, tr_M = float(torch.trace(cov)), float(torch.trace(M))
    return {
        "mu_sq_norm": mu_sq,
        "cov_top3": [float(x) for x in cov_eigs[:3]],
        "second_moment_top3": [float(x) for x in M_eigs[:3]],
        "frac_var_along_mu": float(mu @ cov @ mu) / (mu_sq * tr_cov) if mu_sq > 0 and tr_cov > 0 else float("nan"),
        "frac_second_moment_along_mu": float(mu @ M @ mu) / (mu_sq * tr_M) if mu_sq > 0 and tr_M > 0 else float("nan"),
        "mean_frac_of_second_moment": mu_sq / tr_M if tr_M > 0 else float("nan"),
        "trace_cov": tr_cov,
        "trace_second_moment": tr_M,
    }


def spike_stats(model: ModelSpec, params: Params, X: Tensor) -> Dict[str, object]:
    """Pass-through to :func:`snnland.models.spike_diagnostics`."""
    return spike_diagnostics(model, params, X)
