"""Functional model definitions: deep linear nets, ReLU MLPs and LIF spiking nets.

Every model is a pure function ``forward(params, X) -> Y`` of an ordered
parameter dict ``{"W1": (h1 x d_in), ..., "WL": (d_out x h_{L-1})}`` (plus
``"b1".."bL"`` when ``bias=True``), so that ``torch.func`` transforms
(``grad``, ``jacrev``, ``jacfwd``, ``vmap``) can be applied directly.

The ``snn`` kind uses the straight-through surrogate trick

    s = H(v) + (sg(v) - sg(v).detach()),   v = u - theta,

whose forward value is the exact Heaviside spike while its reverse-mode derivative is
``sg'(v)``.  The surrogate gradient field ``g(theta)`` is therefore not the gradient of any
function and its Jacobian (``snnland.hessian.full_hessian``) is in general *not* symmetric.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

Params = Dict[str, Tensor]

KINDS = ("linear", "relu", "tanh", "smooth_snn", "snn")
SURROGATES = ("sigmoid", "atan", "fast_sigmoid", "triangle")
RESETS = ("soft", "hard", "none")
READOUTS = ("rate", "membrane")
INITS = ("gaussian", "kaiming", "balanced_uniform")


# --------------------------------------------------------------------------- #
# Surrogate functions
# --------------------------------------------------------------------------- #
def surrogate_fn(name: str, k: float) -> Callable[[Tensor], Tensor]:
    """Return the smooth surrogate ``sg(v)`` (a CDF-like function of ``v = u - theta``).

    ``sg'(v)`` is the surrogate derivative used for the spike:

    * ``"sigmoid"``:       sg(v) = sigmoid(k v)
    * ``"atan"``:          sg(v) = atan(pi/2 * k v) / pi + 1/2          (SpikingJelly, alpha = k)
    * ``"fast_sigmoid"``:  sg(v) = 1/2 (1 + k v / (1 + k|v|))           (sg' = k/2 / (1 + k|v|)^2)
    * ``"triangle"``:      sg'(v) = k max(0, 1 - k|v|), i.e. sg is the piecewise quadratic
                           0.5 + k v - k^2 v|v| / 2 on |v| <= 1/k and the step outside.
    """
    if name == "sigmoid":
        return lambda v: torch.sigmoid(k * v)
    if name == "atan":
        return lambda v: torch.atan(0.5 * math.pi * k * v) / math.pi + 0.5
    if name == "fast_sigmoid":
        return lambda v: 0.5 * (1.0 + k * v / (1.0 + k * v.abs()))
    if name == "triangle":

        def tri(v: Tensor) -> Tensor:
            inside = 0.5 + k * v - 0.5 * k * k * v * v.abs()
            step = (v > 0).to(v.dtype)
            return torch.where(v.abs() <= 1.0 / k, inside, step)

        return tri
    raise ValueError(f"unknown surrogate {name!r}; choose from {SURROGATES}")


def heaviside(v: Tensor) -> Tensor:
    """Exact spike: 1 where ``v >= 0`` (i.e. ``u >= theta``), else 0.  No gradient."""
    return (v >= 0).to(v.dtype)


class _SurrogateIncrement(torch.autograd.Function):
    """Identity in value and in reverse mode, zero in forward mode.

    Wrapping the zero-valued increment ``sg(v) - sg(v).detach()`` of the straight-through spike
    in this op makes nested ``jvp(grad(.))`` compute the Jacobian of the *surrogate gradient
    field*: the inner reverse pass still routes ``sg'(v)`` through the spike, but the outer
    forward pass sees the spike values as the (piecewise) constants they are.  Without it, the
    outer pass would differentiate the spike values with ``sg'`` as well, which yields the
    symmetric Hessian of a different, smooth function rather than ``d g / d theta``.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return x.clone()

    @staticmethod
    def setup_context(ctx: object, inputs: tuple, output: Tensor) -> None:
        pass

    @staticmethod
    def backward(ctx: object, grad: Tensor) -> Tensor:
        return grad

    @staticmethod
    def jvp(ctx: object, tangent: Tensor) -> Tensor:
        return torch.zeros_like(tangent)


def straight_through_spike(v: Tensor, sg: Callable[[Tensor], Tensor]) -> Tensor:
    """``H(v) + (sg(v) - sg(v).detach())``: exact {0, 1} value, surrogate first derivative.

    The bracket is exactly zero in value, so the spike is exactly ``H(v)``; its reverse-mode
    derivative is ``sg'(v)`` and (through :class:`_SurrogateIncrement`) its forward-mode
    tangent is zero, see :func:`snnland.hessian.full_hessian`.
    """
    smooth = sg(v)
    return heaviside(v) + _SurrogateIncrement.apply(smooth - smooth.detach())


# --------------------------------------------------------------------------- #
# LIF dynamics (functional, no in-place ops)
# --------------------------------------------------------------------------- #
def lif_run(
    current: Tensor,
    T: int,
    beta: float,
    theta: float,
    spike_fn: Optional[Callable[[Tensor], Tensor]],
    reset: str,
    detach_reset: bool,
    constant_current: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Run a layer of LIF neurons for ``T`` steps.

    Args:
        current: input current ``(T, *batch, h)``, or ``(*batch, h)`` held constant over time
            when ``constant_current`` is set (direct encoding).
        spike_fn: maps ``v = u - theta`` to the spike (straight-through or smooth); ``None`` for
            a non-spiking leaky integrator (no reset).
        reset: ``"soft"``: ``u_t = beta u_{t-1} + I_t - theta s_{t-1}``;
            ``"hard"``: ``u_t = beta (1 - s_{t-1}) u_{t-1} + I_t`` (reset the carried-over
            membrane, then integrate the new input); ``"none"``: no reset.
        detach_reset: use ``s_{t-1}.detach()`` in the reset term (no gradient through resets).

    Returns:
        ``(spikes, v)`` each of shape ``(T, *batch, h)`` where ``v = u - theta``.  For a
        non-spiking integrator ``spikes`` is the membrane ``u`` itself.
    """
    if reset not in RESETS:
        raise ValueError(f"unknown reset {reset!r}; choose from {RESETS}")
    if constant_current:
        current_t = current.unsqueeze(0).expand(T, *current.shape)
    else:
        if current.shape[0] != T:
            raise ValueError(f"current has {current.shape[0]} time steps, expected T={T}")
        current_t = current
    u = torch.zeros_like(current_t[0])
    s = torch.zeros_like(u)
    spikes: List[Tensor] = []
    vs: List[Tensor] = []
    for t in range(T):
        s_reset = s.detach() if detach_reset else s
        if spike_fn is None or reset == "none":
            u = beta * u + current_t[t]
        elif reset == "soft":
            u = beta * u + current_t[t] - theta * s_reset
        else:  # hard
            u = beta * (1.0 - s_reset) * u + current_t[t]
        v = u - theta
        s = u if spike_fn is None else spike_fn(v)
        spikes.append(s)
        vs.append(v)
    return torch.stack(spikes, 0), torch.stack(vs, 0)


# --------------------------------------------------------------------------- #
# Parameter flattening
# --------------------------------------------------------------------------- #
def params_to_vec(params: Params) -> Tensor:
    """Concatenate all parameters (in dict order) into one flat vector."""
    return torch.cat([p.reshape(-1) for p in params.values()])


def vec_to_params(vec: Tensor, shapes: Union["ModelSpec", Dict[str, Tuple[int, ...]]]) -> Params:
    """Inverse of :func:`params_to_vec` given a ``ModelSpec`` or a ``{name: shape}`` dict."""
    if isinstance(shapes, ModelSpec):
        shapes = shapes.shapes
    out: Params = {}
    offset = 0
    for name, shape in shapes.items():
        numel = int(math.prod(shape))
        out[name] = vec[offset : offset + numel].reshape(shape)
        offset += numel
    if offset != vec.numel():
        raise ValueError(f"vector has {vec.numel()} entries but shapes need {offset}")
    return out


def random_orthogonal(n: int, gen: torch.Generator, dtype: torch.dtype = torch.float64) -> Tensor:
    """Haar-random orthogonal ``n x n`` matrix (QR of a Gaussian with sign fix)."""
    G = torch.randn(n, n, generator=gen, dtype=dtype)
    Q, R = torch.linalg.qr(G)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Model specification
# --------------------------------------------------------------------------- #
class ModelSpec:
    """Architecture + hyperparameters of a functional model.

    Args:
        kind: ``"linear"``, ``"relu"``, ``"tanh"``, ``"smooth_snn"`` or ``"snn"``.
        L: number of weight matrices (depth).
        widths: hidden widths, an int (shared) or a sequence of length ``L - 1``.
        d_in, d_out: input / output dimensions.
        bias: add bias vectors ``b_l`` (default: none).
        center_features: (relu / smooth_snn / snn only) subtract the *detached* batch mean of
            the last hidden features (time-averaged rates for spiking nets) before the readout.
        T, beta, theta: LIF steps, leak and threshold.
        k: surrogate slope (``sigmoid(k v)``; ``alpha`` for ``"atan"``).
        surrogate: one of ``SURROGATES``.
        reset: ``"soft"`` (subtract ``theta s_{t-1}``), ``"hard"`` or ``"none"``.
        detach_reset: detach the spike in the reset term.
        readout: ``"rate"``: ``Y = W_L mean_t s_t^{(L-1)}``; ``"membrane"``: ``Y = mean_t u_t``
            of a non-spiking leaky integrator driven by ``W_L s_t^{(L-1)}`` (by linearity this is
            ``W_L`` applied to the time-averaged leaky-integrated rates).

    Layers ``1 .. L-1`` are spiking for the ``snn`` kinds (``L - 1`` LIF layers); layer ``L`` is
    the linear readout.  Default dtype is float64 throughout.
    """

    def __init__(
        self,
        kind: str,
        L: int,
        d_in: int,
        widths: Union[int, Sequence[int]],
        d_out: int,
        bias: bool = False,
        center_features: bool = False,
        T: int = 8,
        beta: float = 0.9,
        theta: float = 1.0,
        k: float = 4.0,
        surrogate: str = "sigmoid",
        reset: str = "soft",
        detach_reset: bool = False,
        readout: str = "rate",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
        if L < 1:
            raise ValueError("L must be >= 1")
        if kind in ("snn", "smooth_snn") and L < 2:
            raise ValueError("spiking models need L >= 2 (at least one LIF layer + readout)")
        if isinstance(widths, int):
            widths = [widths] * (L - 1)
        widths = list(widths)
        if len(widths) != L - 1:
            raise ValueError(f"need {L - 1} hidden widths, got {len(widths)}")
        if surrogate not in SURROGATES:
            raise ValueError(f"unknown surrogate {surrogate!r}; choose from {SURROGATES}")
        if reset not in RESETS:
            raise ValueError(f"unknown reset {reset!r}; choose from {RESETS}")
        if readout not in READOUTS:
            raise ValueError(f"unknown readout {readout!r}; choose from {READOUTS}")
        if center_features and kind == "linear":
            raise ValueError("center_features is only defined for relu / tanh / smooth_snn / snn")
        self.kind = kind
        self.L = L
        self.d_in = d_in
        self.d_out = d_out
        self.widths = widths
        self.dims: List[int] = [d_in] + widths + [d_out]
        self.bias = bias
        self.center_features = center_features
        self.T, self.beta, self.theta, self.k = T, beta, theta, k
        self.surrogate, self.reset, self.detach_reset, self.readout = surrogate, reset, detach_reset, readout
        self.dtype = dtype
        self._sg = surrogate_fn(surrogate, k)

        self.shapes: Dict[str, Tuple[int, ...]] = {}
        for l in range(1, L + 1):
            self.shapes[f"W{l}"] = (self.dims[l], self.dims[l - 1])
            if bias:
                self.shapes[f"b{l}"] = (self.dims[l],)
        self.param_names: List[str] = list(self.shapes)
        self.layer_slices: List[slice] = []
        offset = 0
        for l in range(1, L + 1):
            start = offset
            offset += int(math.prod(self.shapes[f"W{l}"]))
            if bias:
                offset += self.dims[l]
            self.layer_slices.append(slice(start, offset))
        self.n_params = offset

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        extra = ""
        if self.kind in ("snn", "smooth_snn"):
            extra = (
                f", T={self.T}, beta={self.beta}, theta={self.theta}, k={self.k}, "
                f"surrogate={self.surrogate!r}, reset={self.reset!r}, detach_reset={self.detach_reset}, "
                f"readout={self.readout!r}"
            )
        return (
            f"ModelSpec(kind={self.kind!r}, L={self.L}, dims={self.dims}, bias={self.bias}, "
            f"center_features={self.center_features}{extra})"
        )

    @property
    def is_spiking(self) -> bool:
        return self.kind in ("snn", "smooth_snn")

    def params_to_vec(self, params: Params) -> Tensor:
        return torch.cat([params[name].reshape(-1) for name in self.param_names])

    def vec_to_params(self, vec: Tensor) -> Params:
        return vec_to_params(vec, self.shapes)

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def init_params(
        self,
        seed: int = 0,
        init: Optional[str] = None,
        gain: Optional[float] = None,
        r: Optional[int] = None,
        s: Union[float, Sequence[float]] = 1.0,
        tail_value: float = 0.0,
    ) -> Params:
        """Initialise parameters deterministically.

        Args:
            init: ``"gaussian"`` (``N(0, gain^2 / fan_in)``, default gain 1), ``"kaiming"``
                (gaussian with gain ``sqrt(2)``; default for relu / snn kinds) or
                ``"balanced_uniform"``.  Default: ``"gaussian"`` for linear, ``"kaiming"`` otherwise.
            gain: overrides the gaussian gain.
            r: rank of the balanced init (required for ``"balanced_uniform"``).
            s: singular value (float) or per-layer list ``[s_1, ..., s_L]`` (non-uniform case).
            tail_value: singular value on the remaining ``min(dims) - r`` directions.

        ``"balanced_uniform"`` sets ``W_l = U_{l+1} diag(s_l on r dirs, tail elsewhere) U_l^T`` with
        shared Haar-random orthogonal factors ``U_1 (d_in x d_in), ..., U_{L+1} (d_out x d_out)``,
        so the end-to-end map is ``U_{L+1} diag(prod_l s_l) U_1^T`` on an ``r``-dim subspace.

        Spiking heuristic for gaussian / kaiming: with whitened inputs the layer-1 currents
        have std ``gain``; a neuron with constant current ``I`` first spikes within ``T`` steps iff
        ``I > theta (1 - beta) / (1 - beta^T)`` (``~0.175`` for theta=1, beta=0.9, T=8), so with
        gain ``sqrt(2)`` roughly 45% of the neurons of layer 1 spike.  Deeper layers receive
        spike inputs with a similar scale.  Use :func:`spike_diagnostics` to verify.
        Biases (if any) are initialised to zero.
        """
        if init is None:
            init = "gaussian" if self.kind == "linear" else "kaiming"
        if init not in INITS:
            raise ValueError(f"unknown init {init!r}; choose from {INITS}")
        gen = torch.Generator().manual_seed(int(seed))
        params: Params = {}
        if init in ("gaussian", "kaiming"):
            g = gain if gain is not None else (math.sqrt(2.0) if init == "kaiming" else 1.0)
            for l in range(1, self.L + 1):
                fan_out, fan_in = self.shapes[f"W{l}"]
                params[f"W{l}"] = torch.randn(fan_out, fan_in, generator=gen, dtype=self.dtype) * (
                    g / math.sqrt(fan_in)
                )
                if self.bias:
                    params[f"b{l}"] = torch.zeros(fan_out, dtype=self.dtype)
            return params
        # balanced_uniform
        if r is None:
            raise ValueError("balanced_uniform init needs r")
        if r > min(self.dims):
            raise ValueError(f"r={r} exceeds the smallest layer dimension {min(self.dims)}")
        s_layers = [float(s)] * self.L if isinstance(s, (int, float)) else [float(x) for x in s]
        if len(s_layers) != self.L:
            raise ValueError(f"need {self.L} per-layer singular values, got {len(s_layers)}")
        U = [random_orthogonal(d, gen, self.dtype) for d in self.dims]  # U_1 .. U_{L+1}
        for l in range(1, self.L + 1):
            m = min(self.dims[l], self.dims[l - 1])
            s_vec = torch.full((m,), tail_value, dtype=self.dtype)
            s_vec[:r] = s_layers[l - 1]
            params[f"W{l}"] = (U[l][:, :m] * s_vec.unsqueeze(0)) @ U[l - 1][:, :m].T
            if self.bias:
                params[f"b{l}"] = torch.zeros(self.dims[l], dtype=self.dtype)
        return params

    # ------------------------------------------------------------------ #
    # Forward passes
    # ------------------------------------------------------------------ #
    def _affine(self, params: Params, h: Tensor, l: int) -> Tensor:
        out = h @ params[f"W{l}"].T
        if self.bias:
            out = out + params[f"b{l}"]
        return out

    def forward(self, params: Params, X: Tensor, feature_mean: Optional[Tensor] = None) -> Tensor:
        """Evaluate the model: ``X`` of shape ``(n, d_in)`` (or ``(d_in,)`` under ``vmap``).

        ``feature_mean`` (only with ``center_features``): the detached batch mean of the last
        hidden features to subtract; computed from ``X`` when omitted (``X`` must then be batched).
        """
        a = self.hidden_features(params, X)
        if self.center_features:
            if feature_mean is None:
                if X.dim() != 2:
                    raise ValueError("center_features needs a batched X (or an explicit feature_mean)")
                feature_mean = a.detach().mean(0)
            a = a - feature_mean.detach()
        return self._readout(params, a)

    def hidden_features(self, params: Params, X: Tensor) -> Tensor:
        """The last hidden features ``a`` that the readout ``W_L`` acts on (before centering).

        For spiking kinds and ``readout="rate"`` these are the time-averaged spikes of layer
        ``L - 1``; for ``"membrane"`` the time-averaged leaky-integrated rates (so that
        ``Y = W_L a`` in both cases).  For ``linear`` / ``relu`` the activations after layer ``L-1``.
        """
        X = X.to(self.dtype)
        if self.kind == "linear":
            h = X
            for l in range(1, self.L):
                h = self._affine(params, h, l)
            return h
        if self.kind == "relu":
            h = X
            for l in range(1, self.L):
                h = torch.relu(self._affine(params, h, l))
            return h
        if self.kind == "tanh":
            h = X
            for l in range(1, self.L):
                h = torch.tanh(self._affine(params, h, l))
            return h
        spikes, _ = self._spiking_stack(params, X)
        rates = spikes[-1]  # (T, *batch, h_{L-1})
        if self.readout == "rate":
            return rates.mean(0)
        integrated, _ = lif_run(rates, self.T, self.beta, self.theta, None, "none", False)
        return integrated.mean(0)

    def _readout(self, params: Params, a: Tensor) -> Tensor:
        return self._affine(params, a, self.L)

    def _spike_fn(self) -> Callable[[Tensor], Tensor]:
        if self.kind == "snn":
            return lambda v: straight_through_spike(v, self._sg)
        return self._sg  # smooth_snn: the surrogate is the forward nonlinearity

    def _spiking_stack(self, params: Params, X: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Run the ``L - 1`` LIF layers; returns per-layer spike and ``v`` tensors ``(T, *batch, h)``."""
        spike_fn = self._spike_fn()
        current = self._affine(params, X, 1)  # direct encoding: I_t = W_1 x for all t
        spikes_all: List[Tensor] = []
        vs_all: List[Tensor] = []
        for l in range(1, self.L):
            spikes, vs = lif_run(
                current, self.T, self.beta, self.theta, spike_fn, self.reset, self.detach_reset,
                constant_current=(l == 1),
            )
            spikes_all.append(spikes)
            vs_all.append(vs)
            if l < self.L - 1:
                current = self._affine(params, spikes, l + 1)  # (T, *batch, h_{l+1})
        return spikes_all, vs_all

    __call__ = forward


# --------------------------------------------------------------------------- #
# Spike diagnostics
# --------------------------------------------------------------------------- #
def spike_diagnostics(spec: ModelSpec, params: Params, X: Tensor) -> Dict[str, object]:
    """Spike statistics of a spiking model on a batch ``X`` of shape ``(n, d_in)``.

    Returns a dict with overall numbers and per-layer lists:
        ``ever_spike_frac``: fraction of neurons that spike at least once over (batch, time);
        ``mean_rate``: mean spike value over (time, batch, neuron);
        ``window_frac``: fraction of (sample, time, neuron) with ``|v| < 2 / k`` (inside the
        surrogate window);
        ``min_abs_v``: smallest ``|v|`` (distance to threshold; finite-difference checks of the
        surrogate gradient are only meaningful when the perturbation stays below this);
        ``silent_sample_frac``: fraction of samples with no spike at all in the last LIF layer.
    """
    if not spec.is_spiking:
        raise ValueError("spike_diagnostics needs a spiking model")
    with torch.no_grad():
        spikes_all, vs_all = spec._spiking_stack(params, X)
    width = 2.0 / spec.k
    per_layer_ever: List[float] = []
    per_layer_rate: List[float] = []
    per_layer_window: List[float] = []
    per_layer_min_abs_v: List[float] = []
    n_neurons = 0
    n_ever = 0.0
    for spikes, vs in zip(spikes_all, vs_all):
        fired = (spikes > 0.5).any(dim=0).any(dim=0)  # (h,)
        per_layer_ever.append(float(fired.to(torch.float64).mean()))
        per_layer_rate.append(float(spikes.mean()))
        per_layer_window.append(float((vs.abs() < width).to(torch.float64).mean()))
        per_layer_min_abs_v.append(float(vs.abs().min()))
        n_neurons += fired.numel()
        n_ever += float(fired.sum())
    last = spikes_all[-1]
    silent = float(((last > 0.5).sum(dim=(0, 2)) == 0).to(torch.float64).mean())
    return {
        "ever_spike_frac": n_ever / n_neurons,
        "mean_rate": float(sum(per_layer_rate) / len(per_layer_rate)),
        "window_frac": float(sum(per_layer_window) / len(per_layer_window)),
        "min_abs_v": float(min(per_layer_min_abs_v)),
        "silent_sample_frac": silent,
        "per_layer_ever_spike_frac": per_layer_ever,
        "per_layer_mean_rate": per_layer_rate,
        "per_layer_window_frac": per_layer_window,
        "per_layer_min_abs_v": per_layer_min_abs_v,
    }
