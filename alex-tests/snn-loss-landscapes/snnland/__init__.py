"""snnland: Hessian spectra of deep linear nets, ReLU MLPs and LIF spiking nets (CPU, float64).

Tooling to test the claims of arXiv 2602.00545 ("Depth, Not Data: An Analysis of Hessian
Spectral Bifurcation") on spiking networks trained with surrogate gradients.

USAGE
-----
>>> import snnland as sl
>>> sl.set_threads(1)
>>> X = sl.make_whitened_inputs(n=64, d=8, seed=0)              # X^T X / n == I exactly
>>> Y, Wstar = sl.make_rank_r_targets(X, r=2, d_out=4, seed=0)   # rank-2 balanced targets
>>> model = sl.ModelSpec("linear", L=3, d_in=8, widths=8, d_out=4)
>>> params = model.init_params(seed=0, init="balanced_uniform", r=2, s=1.0)
>>> H = sl.full_hessian(model, params, X, Y)                     # Jacobian of the gradient field
>>> H_o = sl.gauss_newton(model, params, X, Y)                   # J^T J / n
>>> stats = sl.cluster_stats(sl.eigvals(H_o), r=2, d_in=8, d_out=4)
>>> stats["ratio_paper"]                                         # == L == 3
>>> sl.layer_coherence(H_o, model.layer_slices)["coherence"]     # == L
>>> snn = sl.ModelSpec("snn", L=3, d_in=8, widths=16, d_out=4, T=8, beta=0.9, k=4.0)
>>> p = snn.init_params(seed=0)                                  # kaiming, ~45% neurons spike
>>> sl.spike_diagnostics(snn, p, X)                              # rates / surrogate window
>>> H = sl.full_hessian(snn, p, X, Y); sl.asymmetry(H)           # surrogate Hessian, non-symmetric
>>> hist = sl.gd_train(snn, p, X, Y, lr=0.1, steps=100, log_every=50,
...                    callbacks=[sl.hessian_summary_callback(snn, X, Y, "mse", 2, 8, 4)])

Modules: ``data`` (whitened / imbalanced inputs, rank-r targets, digits), ``models``
(functional ``ModelSpec``, LIF + straight-through surrogate spikes, inits, spike diagnostics),
``hessian`` (``full_hessian``, ``gauss_newton``, ``functional_part``, ``eigh``, checks),
``analysis`` (``cluster_stats``, ``layer_coherence``, ``top_eigvec_layer_energy``,
``feature_second_moment_stats``), ``train`` (``gd_train``).
"""

from __future__ import annotations

import warnings

import torch

# torch's own forward-mode decompositions (torch/_decomp/decompositions_for_jvp.py) still call
# the deprecated torch.jit.script on first use of jvp; the warning is not actionable here.
warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)

from .analysis import (
    cluster_stats,
    feature_second_moment_stats,
    layer_coherence,
    spike_stats,
    top_eigvec_layer_energy,
)
from .data import (
    geometric_spectrum,
    load_digits_dataset,
    make_covariance_imbalanced_inputs,
    make_rank_r_targets,
    make_whitened_inputs,
)
from .hessian import (
    asymmetry,
    check_hessian_fd,
    eigh,
    eigvals,
    full_hessian,
    functional_part,
    gauss_newton,
    layer_blocks,
    linear_gauss_newton_analytic,
    loss_fn,
    make_vec_loss,
    per_sample_jacobian,
    surrogate_gradient,
    symmetrize,
    top_k,
)
from .models import (
    ModelSpec,
    lif_run,
    params_to_vec,
    spike_diagnostics,
    straight_through_spike,
    surrogate_fn,
    vec_to_params,
)
from .train import gd_train, hessian_summary_callback

__version__ = "0.1.0"


def set_threads(n: int = 1) -> None:
    """Set the number of intra-op CPU threads (experiments share 4 CPUs between processes)."""
    torch.set_num_threads(int(n))


__all__ = [
    "ModelSpec", "lif_run", "params_to_vec", "vec_to_params", "spike_diagnostics",
    "straight_through_spike", "surrogate_fn",
    "make_whitened_inputs", "make_covariance_imbalanced_inputs", "geometric_spectrum",
    "make_rank_r_targets", "load_digits_dataset",
    "loss_fn", "make_vec_loss", "surrogate_gradient", "full_hessian", "symmetrize", "asymmetry",
    "gauss_newton", "per_sample_jacobian", "layer_blocks", "functional_part", "eigh", "eigvals",
    "top_k", "check_hessian_fd", "linear_gauss_newton_analytic",
    "cluster_stats", "layer_coherence", "top_eigvec_layer_energy", "feature_second_moment_stats",
    "spike_stats",
    "gd_train", "hessian_summary_callback",
    "set_threads",
]
