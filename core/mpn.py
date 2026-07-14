"""
Efficiency-optimized MPN implementation (the default; wired in everywhere).

The reference implementation this was derived from is kept as core/mpn_archive.py.
Same networks (MultiPlasticLayer / MultiPlasticNet / DeepMultiPlasticNet) and the
same public API (bptt_gradients / local_* / sequence_gradients), producing results
IDENTICAL to mpn_archive.py up to floating-point round-off (some changes are
bitwise-exact, the speedups reorder ops). Verified by tests/test_mpn_revise.py:
float64 diffs are ~1e-16 (pure round-off → same computation), float32 well within
1e-5, across BPTT and all local rules on both nets.

What is optimized (vs mpn_archive.py), all result-preserving:
  Forward / BPTT
    - MP-layer forward splits W_eff·x into F.linear(x, W) [static GEMM] +
      bmm(W⊙M, x) [plastic] — no materialized (B,i,I) W_eff.
    - update_M_matrix computes M_pre = λM + η·postᵀpre directly (no −M+λM, no
      delta_M alloc); masked path vectorized. Readout via F.linear.
    - BPTT uses a loss-only helper (skips the unused analytic grad_output);
      dropped redundant clones.
  Local rules (the main win here)
    - local_direct / local_diag_rflo FUSE eligibility+grad+trace into
      local_grad_step_fast WITHOUT materializing the (B,i,I) E tensor:
      direct contracts ℓ·φ'·(1+M)·x directly; diag reuses factor=1+M+W·A for
      both grad_W and the A-update. exact still builds E/P (algorithmic).
    - update_M_matrix_local_fast skips the general branch/clamp path (valid only
      in the clean local config assert_local_config guarantees: mult, linear
      m_act, no bounds — hard-sets self.M = M_pre).
    - Input-embedding backprojection via backproject_through_modulated_weights_fast
      (ℓ_pre@W + bmm(ℓ_pre, W⊙M), no W_eff).
    - eta/lam expanded once per unroll; batched readout/embedding grads; streams
      the scalar loss and supports return_outputs=False to skip storing outputs.
    - Tier-A overhead cuts (long-unroll): the mode branch is hoisted OUT of the
      per-step loop (step_fn_for(mode) → _local_step_{direct,diag,exact} resolved
      once); inputs/labels/masks are made time-major + contiguous so per-step reads
      are contiguous (B,·) rows; the readout/embedding scratch buffers (go/hid/ga/u)
      are persistent (self._scratch, reused across calls, never checkpointed) and
      time-major so writes are contiguous. All calc-preserving.
  Measured: local_direct ~2.0×, local_diag_rflo ~1.6×, BPTT ~1.4× vs archive (CPU).

core/mpn_archive.py is left untouched as the reference implementation.
"""
import torch
from torch import nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch.nn.init import orthogonal_

import math
import numpy as np
import copy
import time
import os


# ─── Tier-B: optional torch.compile of the local per-step core (opt-in) ───────
# The direct/diag local rules have a fixed per-timestep computation with no
# data-dependent Python control flow, which makes it a clean torch.compile /
# CUDA-graph target: compiling fuses the elementwise ops and cuts launch overhead
# over a long unroll (T=784 seq-MNIST-pixel). To keep this safe when it cannot be
# validated (no GPU in dev), it is:
#   * OPT-IN and default OFF (set MPN_COMPILE_LOCAL=1 or call
#     mpn.set_compile_local(True)); the eager path is untouched when off;
#   * a PURE-FUNCTIONAL core (no self / no in-place attribute writes inside the
#     compiled region) so the same function runs eager and compiled — one source
#     of truth, CPU-verifiable — with the network doing the state assignment;
#   * wrapped in a try/except that falls back to the eager core if compilation
#     raises, so a compile failure can never break a training run.
COMPILE_LOCAL = os.environ.get("MPN_COMPILE_LOCAL", "0") == "1"
_COMPILED = {}   # (fn_name -> compiled callable) cache


def set_compile_local(flag: bool):
    """Enable/disable torch.compile of the local per-step cores at runtime.
    Clears the compiled-fn cache so the next call recompiles as needed."""
    global COMPILE_LOCAL
    COMPILE_LOCAL = bool(flag)
    _COMPILED.clear()


def _maybe_compile(fn):
    """Return fn, or a torch.compile'd version cached by name when COMPILE_LOCAL
    is on. Falls back to the eager fn if torch.compile raises (e.g. unsupported
    backend) so the caller never has to care."""
    if not COMPILE_LOCAL:
        return fn
    key = fn.__name__
    cached = _COMPILED.get(key)
    if cached is None:
        try:
            cached = torch.compile(fn, dynamic=False)
        except Exception as e:   # pragma: no cover - environment dependent
            print(f"[mpn] torch.compile({key}) failed ({e}); using eager.")
            cached = fn
        _COMPILED[key] = cached
    return cached


# ─── Pure-functional local per-step cores (no self, no in-place attr writes) ──
# These implement exactly the same math as MultiPlasticLayer._local_step_{direct,
# diag} + update_M_matrix_local_fast, but as pure tensor->tensor functions so they
# can be torch.compile'd. Each returns the per-step grads AND the NEW state
# (M_new for direct; M_new/A_new/Q_new for diag); the layer assigns state. Bias
# eligibility R / phi' need not be returned (self.E/self.R were dead diagnostics).

def _core_direct_step(x, hidden, phi_prime, ell, M, W, eta, lam, um):
    """Full fused direct step: grads (using M_{t-1}) + the hebb_assoc M update.
    grad_W = sum_B (ell*phi')_i (1+M_iI) x_I ; grad_b = sum_B (ell*phi')_i ;
    M_new = lam*M + eta*(hiddenᵀx). Returns (grad_W_t, grad_b_t, M_new).
    Used only for hebb_assoc (a=1); hebb_pre keeps the eager path (its M update
    uses a post-independent constant, handled by update_M_matrix_local_fast)."""
    ell_phi = ell * phi_prime
    grad_W_t = (ell_phi.unsqueeze(-1) * (1.0 + M) * x.unsqueeze(1)).sum(0)
    grad_b_t = ell_phi.sum(0)
    outer = hidden.unsqueeze(-1) * x.unsqueeze(1)                 # (B,i,I) hebb_assoc
    M_new = lam.unsqueeze(0) * M + eta.unsqueeze(0) * outer
    if um is not None:
        m = um.view(-1, 1, 1)
        M_new = m * M_new + (1.0 - m) * M
    return grad_W_t, grad_b_t, M_new


def _core_diag_step(x, hidden, phi_prime, ell, M, A, Q, W, eta, lam, a, um):
    """Full fused diagonal-RFLO step: grad_W/grad_b (from t-1 traces) + A,Q,M
    updates. Same math as _local_step_diag + update_M_matrix_local_fast. Returns
    (grad_W_t, grad_b_t, M_new, A_new, Q_new)."""
    W0 = W.unsqueeze(0)
    factor = 1.0 + M + W0 * A                                    # (B,i,I)
    ell_phi = ell * phi_prime

    grad_W_t = (ell_phi.unsqueeze(-1) * x.unsqueeze(1) * factor).sum(0)

    row_recurrent_b = torch.bmm(Q * W0, x.unsqueeze(-1)).squeeze(-1)
    R = phi_prime * (1.0 + row_recurrent_b)
    grad_b_t = torch.einsum('Bi,Bi->i', ell, R)

    A_new = lam.unsqueeze(0) * A + a * eta.unsqueeze(0) * phi_prime.unsqueeze(-1) * x.square().unsqueeze(1) * factor
    Q_new = lam.unsqueeze(0) * Q + a * eta.unsqueeze(0) * R.unsqueeze(-1) * x.unsqueeze(1)
    outer = hidden.unsqueeze(-1) * x.unsqueeze(1)                 # hebb_assoc M update
    M_new = lam.unsqueeze(0) * M + eta.unsqueeze(0) * outer
    if um is not None:
        m3 = um.view(-1, 1, 1)
        A_new = m3 * A_new + (1.0 - m3) * A
        Q_new = m3 * Q_new + (1.0 - m3) * Q
        M_new = m3 * M_new + (1.0 - m3) * M
    return grad_W_t, grad_b_t, M_new, A_new, Q_new

from net_helpers import BaseNetwork, BaseNetworkFunctions
from net_helpers import rand_weight_init, get_activation_function


# ─── Feedback (learning-signal) modes ─────────────────────────────────────────
# How each hidden layer's learning signal ("ell", the surrogate dL/dh) is formed.
# exact_spatial differs from random feedback (layerwise_fa/direct_fa) at ANY depth
# — even one hidden layer: grad_output @ W_output (exact) vs @ B (random).
# layerwise_fa and direct_fa differ from EACH OTHER only when there is more than
# one trainable activity boundary (multiple MP layers, or one MP layer plus a
# trainable input embedding); with a single boundary both are ordinary one-hidden-
# layer feedback alignment (one random matrix).
#
#   'exact_spatial'  actual readout + actual (modulated) lower weights — the exact
#                    same-time SPATIAL gradient (weight transport everywhere).
#                    "exact" qualifies the FEEDBACK PATHWAY, not the full deep
#                    gradient: paired with local_exact_rowlocal the TOP plastic
#                    layer's gradient is exact vs BPTT (its only paths to the loss
#                    are the spatial one — delivered by the true W_output — and its
#                    own temporal M-path — captured by the exact P trace), but the
#                    LOWER plastic layers still get SURROGATE gradients (they omit
#                    temporal paths through the plastic state of the layers above).
#                    Paired with local_diag_rflo / local_direct even the top layer
#                    is only approximate, because the eligibility itself is. Hence
#                    the literal name 'exact_spatial'/'backprop_spatial'; the
#                    behavior is mathematically correct either way. Legacy name:
#                    'exact_readout' (kept as an alias; mpn_archive.py still uses it).
#   'layerwise_fa'   conventional recursive feedback alignment: a fixed random matrix
#                    at EVERY adjacent boundary (readout→top via B_feedback, each
#                    inter-hidden boundary via B_inter[n]). No actual W or M is used
#                    in the feedback pathway, removing weight transport at every layer.
#   'direct_fa'      direct feedback alignment: the readout error is projected DIRECTLY
#                    to every hidden layer through its own random matrix B_direct[k]
#                    (no sequential backward chain at all).
_FEEDBACK_MODES = ('exact_spatial', 'layerwise_fa', 'direct_fa')
_FEEDBACK_ALIASES = {'exact_readout': 'exact_spatial'}   # legacy name → canonical


def canonical_feedback_mode(mode):
    """Map a (possibly legacy) feedback_mode string to its canonical name and
    validate it against _FEEDBACK_MODES. 'exact_readout' is the previous name for
    'exact_spatial' and aliases to it (byte-identical behavior); mpn_archive.py
    and older configs still use it. Raises ValueError for an unrecognized mode."""
    mode = _FEEDBACK_ALIASES.get(mode, mode)
    if mode not in _FEEDBACK_MODES:
        raise ValueError(
            f"unknown feedback_mode '{mode}'; expected one of {_FEEDBACK_MODES} "
            f"(or the legacy alias {tuple(_FEEDBACK_ALIASES)})")
    return mode


def masked_mse_loss_and_output_grad(output, labels, mask):
    """Masked MSE identical to net_helpers.compute_loss (float 'cost' mask, mean
    reduction over all B*T*n_out elements) plus its analytic output gradient.

    output, labels, mask: (B, T, n_out); mask is a float cost mask.
    returns (scalar loss, grad_output (B, T, n_out) = dL/d output).
    """
    N = output.numel()  # B * T * n_out, matches F.mse_loss(reduction='mean')
    diff = mask * output - mask * labels           # = mask * (output - labels)
    loss = (diff ** 2).sum() / N
    grad_output = (2.0 / N) * mask * diff          # 2/N * mask^2 * (output - labels)
    return loss, grad_output


def masked_mse_loss_only(output, labels, mask):
    """Scalar masked MSE only — BIT-IDENTICAL to the loss from
    masked_mse_loss_and_output_grad (same ops, same order) but WITHOUT building
    the unused grad_output tensor. The BPTT path uses this: autograd supplies the
    parameter gradients, so the analytic output gradient is dead work. The loss
    tensor and its backward graph are unchanged, so autograd.grad returns exactly
    the same parameter gradients as before (bitwise)."""
    N = output.numel()
    diff = mask * output - mask * labels           # identical expression → identical loss
    return (diff ** 2).sum() / N


def masked_cross_entropy_loss_and_grad(output, labels, mask):
    """Softmax cross-entropy on the mask-selected positions; identical semantics
    to mpn_archive.masked_cross_entropy_loss_and_grad (see it for the full
    contract). output is LOGITS; the per-position weight is w[b,t] = max_c
    mask[b,t,c] (CE normalizes over channels, so a per-channel graded mask is not
    meaningful). Being a non-default loss, it routes the local rules through their
    two-pass path. Kept here so this impl mirrors the archive's loss surface."""
    w = mask.amax(dim=-1)                              # (B, T) per-position weight
    n_scored = w.sum().clamp_min(1.0)
    logp = torch.log_softmax(output, dim=-1)
    ce = -(labels * logp).sum(dim=-1)                  # (B, T)
    loss = (w * ce).sum() / n_scored
    grad_output = w.unsqueeze(-1) * (logp.exp() - labels) / n_scored
    return loss, grad_output


class MultiPlasticLayer(BaseNetworkFunctions):
    """
    Fully-connected layer with multi-plasticity.

    This functions very similarly to a fully connected PyTorch layer. However,
    in addition to the implementation of the layer in a regular forward pass,
    the modulations need to be correctly kept track of through the "reset_state"
    and "update_M_matrix" functions. The latter needs to be called after every
    forward pass where the modulations should be updated.
    """
    def __init__(self, ml_params, output_matrix, verbose=True):
        super().__init__()

        init_string=''

        self.verbose=verbose
        # Name appended to various parameters to disti
        self.mp_layer_name = ml_params.get('mpl_name', '')

        self.n_input = ml_params['n_input']
        self.n_output = ml_params['n_output']

        init_string += '  MP Layer{} parameters:\n'.format(self.mp_layer_name)
        init_string += '    n_neurons - input: {}, output: {}'.format(
            self.n_input, self.n_output
        )

        # Determines whether or not layer weights are trainable parameters
        self.layer_bias = ml_params.get('bias', True)
        self.freeze_layer = ml_params.get('freeze_layer', False)
        if self.freeze_layer:
            init_string += '    W: Frozen // '
            if self.layer_bias:
                init_string += 'b: Frozen //'
        else:
            self.params = ['W',] # This is a local params list that can be merged with full network params if needed
            if self.layer_bias:
                self.params.append('b')

        ### Weight/bias initialization ###
        # Input weights
        self.W_init = ml_params.get('W_init', 'xavier')
        W_freeze = ml_params.get('W_freeze', False)

        # Initialize the weight tensor once.
        W_tensor = torch.tensor(
            rand_weight_init(self.n_input, self.n_output, init_type=self.W_init, cell_types=None),
            dtype=torch.float
        )

        if W_freeze:
            print("MPN Layer W Frozen")
            self.register_buffer('W', W_tensor)
        else:
            self.parameter_or_buffer('W', W_tensor)


        # Bias term
        if self.layer_bias:
            self.b_init = 'gaussian'
        else:
            self.b_init = 'zeros'
        self.parameter_or_buffer('b', torch.tensor(
            rand_weight_init(self.n_output, init_type=self.b_init),
            dtype=torch.float)
        )

        ###### M matrix-related specs ########
        init_string += '\n    M matrix parameters:'

        self.mp_type = ml_params.get('mp_type', 'mult')
        # Controls the update equation of the M matrix (calculation of \Delta M)
        self.m_update_type = ml_params.get('m_update_type', 'hebb_assoc')
        # Activation function to pass M through after update (can enforce bounds)
        self.m_act = ml_params.get('m_activation', 'linear')
        self.m_act_fn, self.m_act_fn_np, self.m_act_fn_p = get_activation_function(self.m_act)

        # Initial modulation values
        self.register_buffer('M_init', torch.zeros((self.n_output, self.n_input,), dtype=torch.float))

        # Controls maximum and minimum values of modulations so weights don't change signs
        self.modulation_bounds = ml_params.get('modulation_bounds', True)
        if self.modulation_bounds:
            self.M_bound_vals = ml_params.get('m_bounds', (-1.0, 1.0,)) # (min, max)

            M_bounds, init_string = self.build_M_bounds(init_string=init_string)
            self.register_buffer('M_bounds', M_bounds)

            # These bounds will need to be continually updated if W is variable and M is additive, which is not yet implemented
            if self.mp_type == 'add' and 'W' in self.params:
                raise NotImplementedError('Need to continuously update bounds in this case.')

        init_string += '      type: {} // Update - type: {} // Act fn: {}'.format(
            self.mp_type, self.m_update_type, self.m_act
        )

        ### Eta dependencies ###
        self.eta_train = ml_params.get('eta_train', True)
        eta_train_str = 'fixed'
        if self.eta_train:
            self.params.append('eta')
            eta_train_str = 'train'
        self.eta_type = ml_params.get('eta_type', 'scalar')
        self.eta_init = ml_params.get('eta_init', 'eta_clamp')

        self.eta_clamp = ml_params.get('eta_clamp', 1.00)

        self.parameter_or_buffer('eta', torch.tensor(
            self.init_M_parameter(param_type=self.eta_type, init_type=self.eta_init),
        dtype=torch.float))

        if self.eta_type in ('scalar', 'pre_vector', 'post_vector', 'matrix'):
            init_string += '\n      Eta: {} ({}) // '.format(self.eta_type, eta_train_str)
        else:
            raise ValueError('eta_type: {} not recognized'.format(self.eta_type))

        ### Lambda dependencies ###
        self.lam_train = ml_params.get('lam_train', True)
        lam_train_str = 'fixed'
        if self.lam_train:
            self.params.append('lam')
            lam_train_str = 'train'
        self.lam_type = ml_params.get('lam_type', 'scalar')
        self.lam_init = ml_params.get('lam_init', 'lam_clamp')
        # Maximum lambda value/corresponding decay time constant (both always computed)
        if 'm_time_scale' in ml_params:
            self.m_time_scale = ml_params.get('m_time_scale')
            self.lam_clamp = 1. - ml_params['dt'] / self.m_time_scale
        else:
            self.lam_clamp = ml_params.get('lam_clamp', 0.95)
            self.m_time_scale = ml_params['dt'] / (1. - self.lam_clamp)

        self.parameter_or_buffer('lam', torch.tensor(
            np.abs(self.init_M_parameter(param_type=self.lam_type, init_type=self.lam_init)), # Always positive
        dtype=torch.float))

        if self.lam_type in ('scalar', 'pre_vector', 'post_vector', 'matrix'):
            init_string += 'Lambda: {} ({}) // Lambda_max: {:.2f} (tau: {:.1e})'.format(
                self.lam_type, lam_train_str, self.lam_clamp, self.m_time_scale
            )
        else:
            raise ValueError('lam_type: {} not recognized'.format(self.lam_type))

        if self.verbose: # Full summary of mp_layer parameters
            print(init_string)

    def reset_state(self, B=1):
        """
        Resets/initializes modulations values
        """

        self.M = torch.ones(B, *self.W.shape, device=self.W.device) #shape: (B, n_output, n_input) = (B, post, pre)
        self.M = self.M * self.M_init.unsqueeze(0) # (B, n_input, n_output) x (1, n_input, n_output)

        self.M_pre = torch.zeros_like(self.M)

        # Snapshot M_init values at frozen positions so freeze_M restores them each step.
        if hasattr(self, '_plasticity_freeze_mask') and self._plasticity_freeze_mask is not None:
            self._M_frozen_vals = self.M[:, self._plasticity_freeze_mask[0],
                                            self._plasticity_freeze_mask[1]].clone()

    def set_plasticity_freeze(self, post_indices, pre_indices):
        """Freeze M evolution at specific (post, pre) positions.

        After each update_M_matrix call, M[:, post, pre] is reset to its
        value at trial onset (from reset_state).  Pass torch long tensors.
        """
        self._plasticity_freeze_mask = (post_indices, pre_indices)
        self._M_frozen_vals = None  # populated by reset_state

    def clear_plasticity_freeze(self):
        """Remove any plasticity freeze mask."""
        self._plasticity_freeze_mask = None
        self._M_frozen_vals = None

    # ─── Exact row-local eligibility learning (MPN expressions) ───────────────
    # State parallel to M for computing dL/dW, dL/db locally in time (RTRL-style
    # forward-mode traces) instead of via BPTT. Only valid in the "clean"
    # derivation config: mp_type='mult', m_activation='linear', no modulation
    # bounds. Indexing throughout: i = post (n_output), I = param-pre index of
    # W_{iI}, J = plastic-pre index. See run_sequence_local_mpn_exact.

    def assert_local_config(self):
        """Guard for the local (eligibility-trace) rules. They are derived for a
        multiplicative modulation with a linear M-activation and no clamping, and
        for one of two Hebbian M-updates:

            hebb_assoc: M_{iI,t} = lam M_{iI,t-1} + eta h_{i,t} x_{I,t}
            hebb_pre:   M_{iI,t} = lam M_{iI,t-1} + eta c        x_{I,t}   (c const)

        For hebb_assoc M depends on the postsynaptic activity h (hence on W, b),
        so the plastic-sensitivity traces P/A/Q are live. For hebb_pre M depends
        only on the input, so dM/dW = dM/db = 0 identically: every plastic trace
        is zero and the local rule is EXACT (see self._assoc). Other updates
        (oja) / nonlinear m_act / bounds would need extra terms, so are refused
        here rather than silently giving wrong gradients."""
        if self.mp_type != 'mult':
            raise NotImplementedError(
                f"local rules derived for mp_type='mult', got '{self.mp_type}'")
        if self.m_update_type not in ('hebb_assoc', 'hebb_pre'):
            raise NotImplementedError(
                f"local rules derived for m_update_type in (hebb_assoc, hebb_pre), "
                f"got '{self.m_update_type}'")
        if self.m_act != 'linear':
            raise NotImplementedError(
                f"local rules require m_activation='linear', got '{self.m_act}'")
        if self.modulation_bounds:
            raise NotImplementedError(
                "local rules require modulation_bounds=False (clamping gives "
                "zero/undefined dM_t/dM_pre in saturated regions)")

    # Back-compat aliases (older callers / tests used these names).
    assert_local_assoc_config = assert_local_config
    assert_exact_rowlocal_config = assert_local_config

    @property
    def _assoc(self):
        """delta_assoc in the derivation: 1.0 when M depends on h (hebb_assoc),
        so dM/dW and dM/db are nonzero and the plastic traces P/A/Q are live;
        0.0 for hebb_pre, where M is input-only so every plastic-sensitivity
        trace is identically zero and the local rule reduces to the exact
        instantaneous (direct) gradient."""
        return 1.0 if self.m_update_type == 'hebb_assoc' else 0.0

    def reset_local_learning_state(self, B=1):
        """Allocate the exact row-local eligibility traces (zeroed).

        P[b, i, I, J] = dM[b, i, J] / dW[i, I]   shape (B, n_output, n_input, n_input)
        Q[b, i, J]    = dM[b, i, J] / db[i]      shape (B, n_output, n_input)
        """
        dev, dt = self.W.device, self.W.dtype
        self.P = torch.zeros(B, self.n_output, self.n_input, self.n_input, device=dev, dtype=dt)
        self.Q = torch.zeros(B, self.n_output, self.n_input, device=dev, dtype=dt)
        self.E = None  # last computed dh/dW  (B, n_output, n_input)
        self.R = None  # last computed dh/db  (B, n_output)

    def compute_exact_rowlocal_eligibility(self, x, phi_prime):
        """Compute E = dh_t/dW and R = dh_t/db from the *previous* traces.

        Must be called AFTER the forward pass (which uses M_{t-1}) but BEFORE
        update_M_matrix / update_exact_rowlocal_traces, so self.M, self.P,
        self.Q still hold their time-(t-1) values.

        x:         (B, n_input)   input to this MP layer at time t (= x_t)
        phi_prime: (B, n_output)  phi'(h_tilde_t), hidden-activation derivative
        returns E (B, n_output, n_input), R (B, n_output)
        """
        M_prev = self.M  # (B, i, I) = M_{t-1}, forward already consumed it

        # E^I_{i} = phi'_i * [ (1 + M_{iI}) x_I + sum_J W_{iJ} x_J P^I_{iJ} ]
        direct = (1.0 + M_prev) * x.unsqueeze(1)                       # (B, i, I)
        row_recurrent = torch.einsum('iJ,BJ,BiIJ->BiI', self.W, x, self.P)
        E = phi_prime.unsqueeze(-1) * (direct + row_recurrent)         # (B, i, I)

        # R_i = phi'_i * [ 1 + sum_J W_{iJ} x_J Q_{iJ} ]
        row_recurrent_b = torch.einsum('iJ,BJ,BiJ->Bi', self.W, x, self.Q)
        R = phi_prime * (1.0 + row_recurrent_b)                        # (B, i)

        self.E, self.R = E, R
        return E, R

    def _eta_lam_full(self):
        """Expand eta and lam to full (n_output, n_input) = (i, J) matrices so
        the trace-update broadcasting is unambiguous across all param types
        (build_M_parameter returns a 1-D (1,) for scalars, 2-D otherwise)."""
        def expand(p, ptype):
            n_out, n_in = self.n_output, self.n_input
            if ptype == 'scalar':
                return p.reshape(1, 1).expand(n_out, n_in)
            elif ptype == 'pre_vector':
                return p.reshape(1, n_in).expand(n_out, n_in)
            elif ptype == 'post_vector':
                return p.reshape(n_out, 1).expand(n_out, n_in)
            elif ptype == 'matrix':
                return p.reshape(n_out, n_in)
            raise ValueError(f"param_type {ptype} not recognized")
        return expand(self.eta, self.eta_type), expand(self.lam, self.lam_type)

    def update_exact_rowlocal_traces(self, x, E, R, update_mask=None, eta_lam=None):
        """Advance the eligibility traces one step (uses M_t's eta/lam):

        P^I_{iJ,t} = lam_{iJ} P^I_{iJ,t-1} + eta_{iJ} x_J E^I_{i,t}
        Q_{iJ,t}   = lam_{iJ} Q_{iJ,t-1}   + eta_{iJ} x_J R_{i,t}

        Call AFTER compute_exact_rowlocal_eligibility, in step with
        update_M_matrix. update_mask (B,) freezes traces for inactive batch rows.
        eta_lam: optional precomputed (eta, lam) from _eta_lam_full() — eta/lam are
        constant during an unroll, so the loop hoists this out (identical values)."""
        eta, lam = eta_lam if eta_lam is not None else self._eta_lam_full()  # each (i, J)
        a = self._assoc  # 0 for hebb_pre → dM/dW = dM/db = 0, traces stay zero

        outerP = torch.einsum('BiI,BJ->BiIJ', E, x)            # x_J E^I_i
        P_new = lam[None, :, None, :] * self.P + a * eta[None, :, None, :] * outerP

        outerQ = torch.einsum('Bi,BJ->BiJ', R, x)              # x_J R_i
        Q_new = lam[None, :, :] * self.Q + a * eta[None, :, :] * outerQ

        if update_mask is not None:
            mP = update_mask.view(-1, 1, 1, 1).to(P_new.dtype)
            mQ = update_mask.view(-1, 1, 1).to(Q_new.dtype)
            P_new = mP * P_new + (1.0 - mP) * self.P
            Q_new = mQ * Q_new + (1.0 - mQ) * self.Q

        self.P, self.Q = P_new, Q_new

    # ─── Diagonal / same-synapse RFLO approximation ──────────────────────────
    # Replaces the exact fourth-order trace P^I_{iJ} (B,post,pre,pre) with a
    # single same-synapse trace A_{iI} ~= P^I_{iI} (B,post,pre), dropping all
    # off-synapse (J != I) plastic sensitivities. The bias trace Q (B,post,pre)
    # stays EXACT — it is only O(d), so there is no reason to approximate it.
    # Intentionally an approximation to BPTT (grows with seq length and eta),
    # except when n_input == 1, where there are no off-diagonal terms to drop.

    def reset_diag_rflo_state(self, B=1):
        """Allocate the diagonal-RFLO traces (zeroed).

        A[b, i, I] ~= dM[b, i, I] / dW[i, I]   shape (B, n_output, n_input)
        Q[b, i, J]  = dM[b, i, J] / db[i]       shape (B, n_output, n_input)  (exact)
        """
        dev, dt = self.W.device, self.W.dtype
        self.A = torch.zeros(B, self.n_output, self.n_input, device=dev, dtype=dt)
        self.Q = torch.zeros(B, self.n_output, self.n_input, device=dev, dtype=dt)
        self.E = None  # last computed (approx) dh/dW  (B, n_output, n_input)
        self.R = None  # last computed dh/db  (B, n_output)

    def compute_diag_rflo_eligibility(self, x, phi_prime):
        """Diagonal approximation to dh_t/dW: drop the sum over J != I, keeping
        only the same-synapse term W_{iI} x_I A_{iI}.

            E_hat^I_{i,t} = phi'_i * x_I * (1 + M_{iI,t-1} + W_{iI} A_{iI,t-1}).

        The bias eligibility R stays exact (uses the exact bias trace Q). Call
        AFTER the forward pass and BEFORE update_diag_rflo_traces / update_M,
        so self.M, self.A, self.Q still hold time-(t-1) values.
        returns E_hat (B, n_output, n_input), R (B, n_output).
        """
        M_prev = self.M   # (B, i, I) = M_{t-1}
        A_prev = self.A   # (B, i, I)

        E = (phi_prime.unsqueeze(-1) * x.unsqueeze(1)
             * (1.0 + M_prev + self.W.unsqueeze(0) * A_prev))          # (B, i, I)

        # Exact row-local bias trace (same as the exact rule).
        row_recurrent_b = torch.einsum('iJ,BJ,BiJ->Bi', self.W, x, self.Q)
        R = phi_prime * (1.0 + row_recurrent_b)                        # (B, i)

        self.E, self.R = E, R
        return E, R

    def update_diag_rflo_traces(self, x, E, R, update_mask=None, eta_lam=None):
        """Advance the diagonal-RFLO traces one step (uses M_t's eta/lam):

        A_{iI,t} = lam_{iI} A_{iI,t-1} + eta_{iI} x_I E_hat^I_{i,t}
        Q_{iJ,t} = lam_{iJ} Q_{iJ,t-1} + eta_{iJ} x_J R_{i,t}          (exact)

        Call AFTER compute_diag_rflo_eligibility, in step with update_M_matrix.
        update_mask (B,) freezes traces for inactive batch rows.
        eta_lam: optional precomputed (eta, lam) — constant over the unroll, hoisted
        out of the time loop (identical values, avoids re-expanding every step)."""
        eta, lam = eta_lam if eta_lam is not None else self._eta_lam_full()  # each (i, I)
        a = self._assoc  # 0 for hebb_pre → dM/dW = dM/db = 0, traces stay zero

        A_new = lam[None] * self.A + a * eta[None] * x.unsqueeze(1) * E  # x_I E_hat^I_i

        outerQ = torch.einsum('Bi,BJ->BiJ', R, x)                      # x_J R_i
        Q_new = lam[None] * self.Q + a * eta[None] * outerQ

        if update_mask is not None:
            mA = update_mask.view(-1, 1, 1).to(A_new.dtype)
            mQ = update_mask.view(-1, 1, 1).to(Q_new.dtype)
            A_new = mA * A_new + (1.0 - mA) * self.A
            Q_new = mQ * Q_new + (1.0 - mQ) * self.Q

        self.A, self.Q = A_new, Q_new

    # ─── Direct / instantaneous local approximation ──────────────────────────
    # The strongest approximation: treat M_{t-1} as a stop-gradient modulatory
    # state and drop ALL sensitivity through the plasticity dynamics. No trace
    # (no P, no A, no Q) — only the current M, x_t, phi'(h_tilde), and ell_t.
    #   E_dir^I_{i} = phi'_i (1 + M_{iI,t-1}) x_I     (diag RFLO with A = 0)
    #   R_dir_{i}   = phi'_i                          (no bias trace Q)
    # For the WEIGHT gradient this is exactly diagonal RFLO with A forced to 0;
    # the bias differs from diagonal RFLO (which keeps an exact Q).

    def compute_direct_local_eligibility(self, x, phi_prime):
        """Direct/instantaneous local eligibility (no trace, stop-gradient
        through the plasticity history). Call AFTER the forward pass so self.M
        holds M_{t-1}. returns E (B, n_output, n_input), R (B, n_output)."""
        M_prev = self.M   # (B, i, I) = M_{t-1}, treated as a stop-gradient state

        E = phi_prime.unsqueeze(-1) * (1.0 + M_prev) * x.unsqueeze(1)   # (B, i, I)
        R = phi_prime                                                  # (B, i)

        self.E, self.R = E, R
        return E, R

    # ─── Fast fused local-learning helpers ───────────────────────────────────
    # These helpers implement the same local rules as compute_*_eligibility plus
    # update_*_traces, but fuse the hot operations for local_direct and
    # local_diag_rflo so the full eligibility tensor E is not materialized unless
    # exact row-local learning requires it.

    def _apply_update_mask_3d(self, new, old, update_mask):
        if update_mask is None:
            return new
        m = update_mask.view(-1, 1, 1).to(dtype=new.dtype, device=new.device)
        return m * new + (1.0 - m) * old

    def _apply_update_mask_4d(self, new, old, update_mask):
        if update_mask is None:
            return new
        m = update_mask.view(-1, 1, 1, 1).to(dtype=new.dtype, device=new.device)
        return m * new + (1.0 - m) * old

    def update_M_matrix_local_fast(self, pre, post, eta=None, lam=None, update_mask=None):
        """Fast M update for the clean local-rule regime.

        Algebraically identical to update_M_matrix for the configurations allowed
        by assert_local_config(): multiplicative MP, linear M activation, no
        modulation bounds, and Hebbian associative/pre-only updates. It skips the
        general-purpose branch/clamp path and reuses preexpanded eta/lam.
        """
        if eta is None or lam is None:
            eta, lam = self._eta_lam_full()

        M_prev = self.M
        if self.m_update_type == 'hebb_pre':
            c = 1.0 / math.sqrt(post.shape[-1])
            outer = c * pre.unsqueeze(1).expand(-1, self.n_output, -1)
        elif self.m_update_type == 'hebb_assoc':
            outer = post.unsqueeze(-1) * pre.unsqueeze(1)
        else:
            raise NotImplementedError("fast local M update supports hebb_assoc/hebb_pre only")

        M_pre = lam.unsqueeze(0) * M_prev + eta.unsqueeze(0) * outer
        M_pre = self._apply_update_mask_3d(M_pre, M_prev, update_mask)

        self.M_pre = M_pre
        self.M = M_pre  # assert_local_config guarantees linear m_act and no bounds.

        if hasattr(self, '_plasticity_freeze_mask') and self._plasticity_freeze_mask is not None:
            self.M[:, self._plasticity_freeze_mask[0],
                      self._plasticity_freeze_mask[1]] = self._M_frozen_vals

        return M_pre - M_prev

    # ── Mode-specialized fused local step (dispatched once, not per timestep) ──
    # Each _local_step_<mode> is one fused local-learning step for MP-layer W,b and
    # its traces, mathematically identical to compute_*_eligibility + update_*_traces
    # but without materializing the (B,i,I) eligibility E. The sequence loop resolves
    # the right one ONCE (via step_fn_for) instead of branching on `mode` every step.

    def _local_step_direct(self, x, phi_prime, ell, eta, lam, update_mask=None):
        """Direct/instantaneous: grad_W = sum_B ell_i phi'_i (1 + M_iI) x_I,
        grad_b = sum_B ell_i phi'_i. No trace (P/A/Q) touched."""
        ell_phi = ell * phi_prime
        grad_W_t = (ell_phi.unsqueeze(-1) * (1.0 + self.M) * x.unsqueeze(1)).sum(0)
        grad_b_t = ell_phi.sum(0)
        self.E, self.R = None, phi_prime
        return grad_W_t, grad_b_t

    def _local_step_diag(self, x, phi_prime, ell, eta, lam, update_mask=None):
        """Diagonal RFLO: factor = 1 + M + W*A fused into grad_W and the A update
        (E_hat = phi'*x*factor never built); exact bias trace Q kept."""
        W, M_prev = self.W, self.M
        a = self._assoc
        A_prev, Q_prev = self.A, self.Q
        factor = 1.0 + M_prev + W.unsqueeze(0) * A_prev
        ell_phi = ell * phi_prime

        grad_W_t = (ell_phi.unsqueeze(-1) * x.unsqueeze(1) * factor).sum(0)

        row_recurrent_b = torch.bmm(Q_prev * W.unsqueeze(0), x.unsqueeze(-1)).squeeze(-1)
        R = phi_prime * (1.0 + row_recurrent_b)
        grad_b_t = torch.einsum('Bi,Bi->i', ell, R)

        A_new = (lam.unsqueeze(0) * A_prev
                 + a * eta.unsqueeze(0) * phi_prime.unsqueeze(-1)
                 * x.square().unsqueeze(1) * factor)
        Q_new = (lam.unsqueeze(0) * Q_prev
                 + a * eta.unsqueeze(0) * R.unsqueeze(-1) * x.unsqueeze(1))
        self.A = self._apply_update_mask_3d(A_new, A_prev, update_mask)
        self.Q = self._apply_update_mask_3d(Q_new, Q_prev, update_mask)
        self.E, self.R = None, R
        return grad_W_t, grad_b_t

    def _local_step_exact(self, x, phi_prime, ell, eta, lam, update_mask=None):
        """Exact row-local: E is still built (the full P trace update needs it),
        but eta/lam are reused and the P/Q update is done inline."""
        a = self._assoc
        E, R = self.compute_exact_rowlocal_eligibility(x, phi_prime)
        grad_W_t = torch.einsum('Bi,BiI->iI', ell, E)
        grad_b_t = torch.einsum('Bi,Bi->i', ell, R)

        P_prev, Q_prev = self.P, self.Q
        outerP = torch.einsum('BiI,BJ->BiIJ', E, x)
        P_new = lam[None, :, None, :] * P_prev + a * eta[None, :, None, :] * outerP
        Q_new = lam[None, :, :] * Q_prev + a * eta[None, :, :] * R.unsqueeze(-1) * x.unsqueeze(1)
        self.P = self._apply_update_mask_4d(P_new, P_prev, update_mask)
        self.Q = self._apply_update_mask_3d(Q_new, Q_prev, update_mask)
        return grad_W_t, grad_b_t

    def step_fn_for(self, mode):
        """Return the mode's fused per-step function, resolved ONCE before a loop
        (so the per-timestep call has no `mode` branch)."""
        return {'direct': self._local_step_direct,
                'diag': self._local_step_diag,
                'exact': self._local_step_exact}[mode]

    # ── torch.compile fast path (opt-in; see set_compile_local / COMPILE_LOCAL) ──
    def can_compile_step(self, mode):
        """True if this layer/mode can use the compiled pure-functional core: only
        direct/diag with the hebb_assoc M update (hebb_pre's M update uses a
        post-independent constant, kept on the eager path)."""
        return (COMPILE_LOCAL and mode in ('direct', 'diag')
                and self.m_update_type == 'hebb_assoc')

    def compiled_step_and_update(self, mode, x, hidden, phi_prime, ell, eta, lam,
                                 update_mask=None):
        """Run the fused (compiled) core for `mode`, ASSIGN the returned state
        (M, and A/Q for diag), and return (grad_W_t, grad_b_t). Same result as
        step_fn_for(mode)(...) followed by update_M_matrix_local_fast(...), but the
        grads + all trace/M updates happen inside one compiled region."""
        if mode == 'direct':
            core = _maybe_compile(_core_direct_step)
            grad_W_t, grad_b_t, M_new = core(
                x, hidden, phi_prime, ell, self.M, self.W, eta, lam, update_mask)
            self.M_pre = M_new
            self.M = M_new
            self.E, self.R = None, phi_prime
        else:  # 'diag'
            core = _maybe_compile(_core_diag_step)
            grad_W_t, grad_b_t, M_new, A_new, Q_new = core(
                x, hidden, phi_prime, ell, self.M, self.A, self.Q, self.W,
                eta, lam, self._assoc, update_mask)
            self.A, self.Q = A_new, Q_new
            self.M_pre = M_new
            self.M = M_new
            self.E, self.R = None, None
        # Plasticity-freeze (rare) still handled here to match the eager path.
        if getattr(self, '_plasticity_freeze_mask', None) is not None:
            self.M[:, self._plasticity_freeze_mask[0],
                      self._plasticity_freeze_mask[1]] = self._M_frozen_vals
        return grad_W_t, grad_b_t

    def local_grad_step_fast(self, x, phi_prime, ell, mode, eta=None, lam=None,
                             update_mask=None):
        """Back-compat single-call dispatcher (kept for external callers). The
        sequence loops use step_fn_for(mode) to hoist this dispatch out of the loop.
        Returns (grad_W_t, grad_b_t)."""
        if eta is None or lam is None:
            eta, lam = self._eta_lam_full()
        return self.step_fn_for(mode)(x, phi_prime, ell, eta, lam, update_mask)

    def backproject_through_modulated_weights_fast(self, ell_pre):
        """Compute ell_x = ell_pre @ W_eff without materializing W_eff.

        ell_pre: (B, post), usually ell * phi_prime.
        returns: (B, pre), sum_i ell_pre_i W_iI (1 + M_iI).
        """
        base = ell_pre.matmul(self.W)
        plastic = torch.bmm(ell_pre.unsqueeze(1), self.W.unsqueeze(0) * self.M).squeeze(1)
        return base + plastic

    @torch.no_grad()
    def param_clamp(self):
        """ Enforce lambda bounds. Doesn't track gradients, since this is always called after weight updates. """
        self.lam.data.clamp_(0., self.lam_clamp)

    def init_M_parameter(self, param_type='scalar', init_type='gaussian'):
        """
        Initialize different forms of the various M parameters (e.g. eta and lambda).
        Default is just one of each parameter for each layer, but can make them
        post- and/or presynaptic cell dependent.

        Turned into a buffer/parameter externally.
        """

        if type(init_type) == float:
            if param_type == 'scalar': # Just directly set to init_type
                param = init_type
            elif param_type == 'pre_vector': # Serves as mean to distribution
                param = init_type + rand_weight_init(self.n_input, init_type='guassian', weight_norm=1.0)
            elif param_type == 'post_vector':
                param = init_type + rand_weight_init(self.n_output, init_type='guassian', weight_norm=1.0)
            elif param_type == 'matrix':
                param = init_type + rand_weight_init(self.n_input, self.n_output, init_type='guassian', weight_norm=1.0)
        elif type(init_type) == str:
            if param_type == 'scalar':
                if init_type in ('lam_clamp',):
                    param = self.lam_clamp
                elif init_type in ('eta_clamp',):
                    param = self.eta_clamp
                else:
                    param = rand_weight_init(1, init_type=init_type, weight_norm=1.0)
            elif param_type == 'pre_vector':
                if init_type in ('lam_clamp',):
                    param = self.lam_clamp * np.ones((self.n_input,))
                elif init_type in ('eta_clamp',):
                    param = self.eta_clamp * np.ones((self.n_input,))
                else:
                    param = rand_weight_init(self.n_input, init_type=init_type, weight_norm=1.0)
            elif param_type == 'post_vector':
                if init_type in ('lam_clamp',):
                    param = self.lam_clamp * np.ones((self.n_output,))
                elif init_type in ('eta_clamp',):
                    param = self.eta_clamp * np.ones((self.n_output,))
                else:
                    param = rand_weight_init(self.n_output, init_type=init_type, weight_norm=1.0)
            elif param_type == 'matrix':
                if init_type in ('lam_clamp',):
                    param = self.lam_clamp * np.random.rand(self.n_output, self.n_input,)
                elif init_type in ('eta_clamp',):
                    param = self.eta_clamp * np.random.rand(self.n_output, self.n_input,)
                else:
                    param = rand_weight_init(self.n_input, self.n_output, init_type=init_type, weight_norm=1.0)
        else:
            raise ValueError('Init of type {} not recognized: {}'.format(type(init_type), init_type))

        return param

    def build_M_parameter(self, param, param_type='scalar'):
        """
        Returns M parameters (e.g. eta and lambda) of the appropriate dimensions
        given their type

        OUTPUTS:
        param_matrix shape is simply something that can be cast to the shape of
        M without batch dim: (n_output, n_input), so len(param_expanded.shape)
        == 2 always
        """

        if param_type == 'scalar':
            param_expanded = param.unsqueeze(0) # shape (1, 1)
        elif param_type == 'pre_vector':
            param_expanded = param.unsqueeze(0) # shape (1, n_input)
        elif param_type == 'post_vector':
            param_expanded = param.unsqueeze(-1) # shape (n_output, 1)
        elif param_type == 'matrix':
            param_expanded = param # shape (n_output, n_input)

        return param_expanded

    def build_M_bounds(self, init_string=''):
        """
        Controls maximum and minimum values of modulations. Generally used so
        weights don't change signs (since these are often tied to cell type).

        Bounds are in order: (upper_vals, lower_vals)
        """

        W_fixed = self.W.detach()

        if self.mp_type == 'add':
            MAX_ADD = self.M_bound_vals[1] # Default: 1.0, Controls how much a weight can be strengthened, 1.0 means the weight's mag can be doubled, 0.0 means it cant be strengthened
            MIN_ADD = self.M_bound_vals[0] # Default: 0.0, Any value >0 prevents weight from being fully weakened, e.g. 0.2 means the weight can be weakened to at most 20% of its original value

            # Expanation of this expression: (with example values MAX_ADD = 2.0 and MIN_ADD = 0.2)
            #   First line: Upper bounds on Ms
            #       For W_ij > 0: Maximum M value is MAX_ADD * W_ij, so 2 * W_ij > 0, meaning positive weights can be strengthened to 3x their initial value
            #       For W_ij < 0: Maximum M value is -1 * (1 - MIN_ADD) * W_ij = -0.8 * W_ij > 0, since W_ij is negative. Since a positive M_ij would cancel the
            #           negative W_ij, this means that at most W_ij can be weakened to 0.2 x its original value
            #   Second line: Lower bounds for M
            #       For W_ij > 0: Minimum M value is -1 * (1 - MIN_ADD) * W_ij = -0.8 * W_ij < 0, since W_ij is positive, a negative M_ij that saturates this bound
            #           would reduce W_ij to 0.2 x its original value.
            #       For W_ij < 0: Minimum M value is MAX_ADD * W_ij = 2 * W_ij < 0, since W_ij is negative. So can strengthen negative weight to 3x its initial value

            M_bounds = torch.cat((
                (MAX_ADD * W_fixed * (W_fixed > 0) - 1 * (1 - MIN_ADD) * W_fixed * (W_fixed < 0)).unsqueeze(0),
                (MAX_ADD * W_fixed * (W_fixed < 0) - 1 * (1 - MIN_ADD) * W_fixed * (W_fixed > 0)).unsqueeze(0)
            ))

            init_string += '    update bounds - Max add: {}, Min add: {}\n'.format(MAX_ADD, MIN_ADD)
        elif self.mp_type == 'mult':
            max_mult = self.M_bound_vals[1] # Controls how much a weight can be enhanced, 1.0 means the weight's mag can be doubled
            min_mult = self.M_bound_vals[0] # Controls how much a weight can be depressed, -1.0 means it can be fully depressed
            M_bounds = torch.cat((
                max_mult * torch.ones_like(W_fixed).unsqueeze(0),
                min_mult * torch.ones_like(W_fixed).unsqueeze(0)
            ))
            init_string += '    update bounds - Max mult: {}, Min mult: {}\n'.format(max_mult, min_mult)
        else:
            raise ValueError('MP type not recognized in build_M_bounds.')

        return M_bounds, init_string

    def update_M_matrix(self, pre, post, update_mask=None, eta_lam_build=None):
        """
        Updates the modulation matrix from one time step to the next.
        Should only be called in the network_step pass once. Directly updates self.M.

        Note that this is NOT automatically called in MP layer's "forward" call, since the
        postsynaptic activity could be dependent on other factors (e.g. other layers).

        M updates can be frozen from the update_mask, if a given bactch idx is
        False (e.g. because it is beyond the end of the sequence)

        pre.shape: (B, n_input)
        post.shape: (B, n_output)
        update_mask: (B,)
        eta_lam_build: optional precomputed (eta, lam) from build_M_parameter —
            these are constant over an unroll, so the sequence loops hoist the
            expansion out (identical values, one fewer op per step).
        """

        if eta_lam_build is not None:
            eta, lam = eta_lam_build
        else:
            eta = self.build_M_parameter(self.eta, self.eta_type)
            lam = self.build_M_parameter(self.lam, self.lam_type)
        M = self.M

        # Compute the pre-activation modulation M_pre directly, batched, without the
        # `-M + λM` cancellation or a zeros(delta_M) allocation. Both branches below
        # implement the same recurrence as the original:
        #   hebb: M_pre = λ M_{t-1} + η · postᵀpre   (hebb_pre: post → 1/√n_out const)
        #   oja : M_pre = M_{t-1} + η·postᵀpre − |η|·post²·M_{t-1}
        # update_mask (B,) freezes inactive rows: M_pre = M there (Δ = 0), matching
        # the original's per-row skip — done vectorized instead of a Python loop.
        if self.m_update_type in ('hebb_pre',):
            post = 1 / math.sqrt(post.shape[-1]) * torch.ones_like(post)

        if self.m_update_type in ('hebb_assoc', 'hebb_pre',):
            outer = torch.einsum('Bi,BI->BiI', post, pre)
            M_pre = lam.unsqueeze(0) * M + eta.unsqueeze(0) * outer
        elif self.m_update_type in ('oja',):
            outer = torch.einsum('Bi,BI->BiI', post, pre)
            M_pre = (M + eta.unsqueeze(0) * outer
                     - torch.abs(eta).unsqueeze(0) * (post ** 2).unsqueeze(-1) * M)
        else:
            raise ValueError(f"unknown m_update_type '{self.m_update_type}'")

        if update_mask is not None:
            # Freeze inactive batch rows (Δ = 0 → M_pre = M) without a Python loop.
            m = update_mask.view(-1, 1, 1).to(M_pre.dtype)
            M_pre = m * M_pre + (1.0 - m) * M

        self.M_pre = M_pre
        self.M = self.m_act_fn(M_pre)

        # Update M matrices, while being sure update holds matrix within bounds
        # (this may error if not self.ei_types, but this is always true in our settings)
        # (note: updates to restristed cell types is built into the eta matrix)
        if self.modulation_bounds:
            self.M = torch.clamp(self.M, min=self.M_bounds[1], max=self.M_bounds[0])

        # Freeze plasticity at masked positions: restore M to its initial-state values
        if hasattr(self, '_plasticity_freeze_mask') and self._plasticity_freeze_mask is not None:
            self.M[:, self._plasticity_freeze_mask[0],
                      self._plasticity_freeze_mask[1]] = self._M_frozen_vals

        # Returned only for "theory matching" consumers; keep it correct (M_pre - M).
        return self.M_pre - M

    def get_modulated_weights(self, M=None):
        """
        Returns modulated weights, taking into account exactly how M and W are combined.
        Note the batch size of the modulated weights is set by the batch size of M.

        If M is not None, the passed M matrix is used, otherwise just uses stored Ms (this
        is be used for analysis of the network after training)
        """

        W = self.W
        if M is None:
            M = self.M

        # Fixed weights and M matrix, either multiplicative or additive
        if self.mp_type == 'mult':
            modulated_weights = W.unsqueeze(0) + W.unsqueeze(0) * M
        elif self.mp_type == 'add':
            modulated_weights = W.unsqueeze(0) + M

        return modulated_weights

    def forward(self, x, run_mode='minimal'):
        """
        Passes inputs through the modulated weights. Activation are handled
        externally to allow for more complicated architectures.

        Updating of modulations is handled externally

        INPUTS:
        x.shape: [B, n_input]

        INTERNALS:
        b.shape: [n_output]
        modulated_weights.shape: [B, n_output, n_input],

        OUTPUTS:
        pre_act.shape: [B, n_output]

        """

        # pre_act_no_bias[b,i] = sum_I W_eff[b,i,I] x[b,I], with the modulated weight
        # W_eff = W + W⊙M (mult) or W + M (add). Instead of materializing the full
        # (B,i,I) W_eff and contracting it, split into:
        #   static  = x @ Wᵀ                         (batch-independent → one GEMM)
        #   plastic = bmm(W⊙M or M, x)               (the genuinely batched part)
        # Same math, reordered: one fewer (B,i,I) temporary (no W_eff add) and the
        # static term routes through an optimized dense-linear kernel.
        static = F.linear(x, self.W)                          # (B, i) = x @ Wᵀ
        if self.mp_type == 'mult':
            plastic_w = self.W.unsqueeze(0) * self.M          # (B, i, I) = W⊙M
        else:  # 'add'
            plastic_w = self.M                                # (B, i, I)
        plastic = torch.bmm(plastic_w, x.unsqueeze(-1)).squeeze(-1)   # (B, i)
        pre_act_no_bias = static + plastic

        pre_act = pre_act_no_bias + self.b.unsqueeze(0)

        if run_mode in ('track_states',):
            db = {
                'pre_act_no_bias': pre_act_no_bias.detach(),
                'M': self.M.detach(),
                'b': self.b.detach().unsqueeze(0), # record bias information as well 
            }
            if self.m_act:
                db['M_pre'] = self.M_pre.detach()
        else:
            db = None

        return pre_act, db

class MultiPlasticNetBase(BaseNetwork):
    """
    Base network for the multiplastic network. Initializes things like the output
    layers and activation functions that are mostly shared across all types of
    MPNs, no matter the connections that lead from input to output.
    """

    def __init__(self, net_params, n_output_pre, output_matrix="", verbose=False):
        # Note that this assumes self.output has already been set in child
        assert hasattr(self, 'n_output')

        super().__init__(net_params, verbose=verbose)

        init_string = 'MultiPlastic Net:\n'

        init_string += '  output neurons: {}\n'.format(
            self.n_output
        )

        # Get list that should be parameters of the network (for later matching to experiment)
        self.params = ['W_output',] # Biases can be added below if used.

        # Numpy equivalents only used for debugging purposes
        self.act = net_params.get('activation', 'linear')
        self.act_fn, self.act_fn_np, self.act_fn_p = get_activation_function(self.act)

        init_string += '  Act: {}\n'.format(
            self.act,
        )

        self.b_output_active = net_params.get('output_bias', True)
        if self.b_output_active:
            self.params.append('b_output')
            self.b_output_init = 'gaussian'
        else:
            self.b_output_init = 'zeros'

        # By default, these are set to none but are overridden in the initialization if cell types are used
        self.ei_balance = None
        self.input_cell_types = None
        self.hidden_cell_types = None

        self.cell_types = net_params.get('cell_types', None)
        if self.cell_types:
            raise NotImplementedError()

        # Output weights
        self.W_output_init = net_params.get('W_output_init', 'xavier')
        self.parameter_or_buffer('W_output', torch.tensor(
            rand_weight_init(n_output_pre, self.n_output, init_type=self.W_output_init,
                             cell_types=self.hidden_cell_types),
            dtype=torch.float)
        )

        # overwrite 
        if output_matrix == "":
            pass
        elif output_matrix == "untrained":
            print("Output Matrix Untrained")
            self.W_output.requires_grad = False
        elif output_matrix == "orthogonal":
            print("Output Matrix Orthogonal and Untrained")
            W_output_init = torch.empty(self.n_output, n_output_pre)  # Create a matrix of size (n_output, n_output_pre)
            orthogonal_(W_output_init)  # In-place orthogonal initialization
            W_output_init = W_output_init.T
            self.parameter_or_buffer('W_output', torch.tensor(W_output_init, dtype=torch.float))
            self.W_output.requires_grad = False
        else:
            raise Exception("Output Matrix not recognized")

        self.parameter_or_buffer('b_output', torch.tensor(
            rand_weight_init(self.n_output, init_type=self.b_output_init),
            dtype=torch.float)
        )

        if verbose: # Full summary of readout parameters (MP layer prints out internally)
            print(init_string)

    def reset_state(self, B=1):
        """ Resets states of all internal layer M matrices """

        for mp_layer in self.mp_layers:
            mp_layer.reset_state(B=B)

    def _scratch(self, name, shape, dtype, device):
        """Return a persistent scratch buffer (allocated once, reused across calls)
        keyed by `name`; reallocated only if shape/dtype/device change. For buffers
        the local loop OVERWRITES every element each step, so no zeroing is needed —
        this just avoids re-allocating (B,T,·) tensors on every sequence_gradients
        call (allocator/‑churn win, especially at long T). Stored under
        self._scratch_bufs (a plain dict, not a registered buffer, so it never lands
        in state_dict / checkpoints)."""
        cache = getattr(self, '_scratch_bufs', None)
        if cache is None:
            cache = {}
            self._scratch_bufs = cache
        buf = cache.get(name)
        if buf is None or buf.shape != shape or buf.dtype != dtype or buf.device != device:
            buf = torch.empty(shape, dtype=dtype, device=device)
            cache[name] = buf
        return buf

    def param_clamp(self):
        # mp_layer call doesn't track gradients, since this is always called after weight updates
        for mp_layer in self.mp_layers:
            mp_layer.param_clamp()

    @torch.no_grad()
    def _monitor_init(self, train_params, train_data, train_trails=None, valid_batch=None, valid_trails=None):

        # Additional quantities to track during training, note initializes these first so that _monitor call
        # inside super()._monitor_init can append additional quantities.
        if self.hist is None:
            self.hist = {
                'iter': 0,
            }
            for mpl_idx, mp_layer in enumerate(self.mp_layers):
                self.hist['eta{}'.format(mp_layer.mp_layer_name)] = []
                self.hist['lam{}'.format(mp_layer.mp_layer_name)] = []

        super()._monitor_init(train_params, train_data, train_trails=train_trails, valid_batch=valid_batch, valid_trails=valid_trails)

    @torch.no_grad()
    def _monitor(self, train_batch, train_go_info_batch, valid_go_info_batch, train_type='supervised', output=None, loss=None, loss_components=None,
                 acc=None, valid_batch=None, nowiter=None):

        super()._monitor(train_batch, train_go_info_batch, valid_go_info_batch, output=output, loss=loss, loss_components=loss_components,
                         valid_batch=valid_batch, nowiter=nowiter)

        for mpl_idx, mp_layer in enumerate(self.mp_layers):
            self.hist['eta{}'.format(mp_layer.mp_layer_name)].append(
                mp_layer.eta.detach().cpu().numpy()
            )
            self.hist['lam{}'.format(mp_layer.mp_layer_name)].append(
                mp_layer.lam.detach().cpu().numpy()
            )

class MultiPlasticNet(MultiPlasticNetBase):
    """
    Two-layer feedforward setup, with single multi-plastic layer followed by a readout layer.
    """

    def __init__(self, net_params, verbose=False):

        if 'n_neurons' in net_params:
            # assert len(net_params['n_neurons']) == 3
            self.n_input = net_params['n_neurons'][0]
            self.n_hidden = net_params['n_neurons'][1]
            self.n_output = net_params['n_neurons'][2]
        else:
            self.n_input = net_params['n_input']
            self.n_hidden = net_params['n_hidden']
            self.n_output = net_params['n_output']

        # output_matrix controls readout trainability (see MultiPlasticNetBase);
        # set before super().__init__ so the MP layer construction can read it.
        self.output_matrix = net_params.get('output_matrix', '')

        # Learning rule for sequence_gradients():
        #   'bptt'                 — autograd through the unrolled forward+M-update
        #   'local_exact_rowlocal' — exact row-local eligibility traces (== BPTT
        #                            in the clean config); alias 'local'
        #   'local_diag_rflo'      — diagonal/same-synapse RFLO approximation
        #   'local_direct'         — direct/instantaneous approximation (no trace;
        #                            stops gradient through the plasticity history)
        _rule = net_params.get('learning_rule', 'bptt')
        if _rule == 'local':                       # back-compat
            _rule = 'local_exact_rowlocal'
        assert _rule in ('bptt', 'local_exact_rowlocal', 'local_diag_rflo',
                         'local_direct'), f"unknown learning_rule '{_rule}'"
        self.learning_rule = _rule

        # Learning signal for the hidden layer. This net has exactly ONE trainable
        # activity boundary (readout → hidden; no input embedding), so the two
        # random-feedback variants ('layerwise_fa', 'direct_fa') coincide — each
        # just replaces the readout transpose with one fixed random matrix
        # B_feedback, i.e. ordinary one-hidden-layer feedback alignment. They would
        # only differ with a second trainable boundary (see DeepMultiPlasticNet).
        # 'exact_spatial' (true gradient via W_output) is distinct from them here
        # and at any depth. Buffer registered in super().__init__ once W_output's
        # shape is known.
        self.feedback_mode = canonical_feedback_mode(
            net_params.get('feedback_mode', 'exact_spatial'))

        super().__init__(net_params, self.n_hidden, verbose=verbose)

        # Fixed random feedback matrix for feedback alignment (same shape as
        # W_output, never trained). Allocated for any non-exact mode (they all
        # reduce to one random boundary for a single hidden layer).
        if self.feedback_mode != 'exact_spatial':
            self.register_buffer('B_feedback', torch.tensor(
                rand_weight_init(self.n_hidden, self.n_output,
                                 init_type=net_params.get('B_feedback_init', 'xavier')),
                dtype=self.W_output.dtype))

        # Creates the input MP layer
        self.param_clamping = True # Always have param clamping for MP layers because lam bounds
        net_params['ml_params']['n_input'] = self.n_input
        net_params['ml_params']['n_output'] = self.n_hidden
        net_params['ml_params']['dt'] = self.dt
        self.mp_layer = MultiPlasticLayer(net_params['ml_params'], output_matrix=self.output_matrix, verbose=verbose)
        self.params.extend(self.mp_layer.params)

        self.mp_layers = [self.mp_layer,] # List of all mp_layers in this network

    def forward(self, inputs, run_mode='minimal', verbose=False):

        x = inputs  # read-only downstream (never mutated in-place) → no clone needed

        # Returns pre-activation
        hidden_pre, db_mp = self.mp_layer(x, run_mode=run_mode)

        hidden = self.act_fn(hidden_pre)

        output = F.linear(hidden, self.W_output, self.b_output)

        if run_mode in ('track_states'):
            db = {
                'M': db_mp['M'],
                'hidden_pre': hidden_pre.detach(),
                'hidden': hidden.detach(),
                "input": x.detach(), 
            }
        else:
            db = None

        return output, hidden, db

    def network_step(self, current_input, run_mode='minimal', verbose=False):
        """
        Performs a single batch pass forward for the network. This mostly consists of a forward pass and
        the associated updates to internal states (i.e. the modulations)

        This should not be passed a full sequence of data, only data from a given time point
        """

        assert len(current_input.shape) == 2

        output, current_hidden, db = self.forward(current_input, run_mode=run_mode, verbose=verbose)

        # M updated internally when this is called, M here is only used if finding fixed points (not yet implemented)
        M = self.mp_layer.update_M_matrix(current_input, current_hidden)

        return output, db

    # ─── Sequence gradients: single switch between BPTT and local learning ────
    def _trainable_params(self):
        """Trainable tensors this net computes gradients for, keyed by name."""
        ps = {'W': self.mp_layer.W, 'W_output': self.W_output}
        if self.mp_layer.layer_bias:
            ps['b'] = self.mp_layer.b
        if self.b_output_active:
            ps['b_output'] = self.b_output
        return {k: v for k, v in ps.items() if v.requires_grad}

    def bptt_gradients(self, inputs, labels, masks,
                       loss_and_grad=masked_mse_loss_and_output_grad,
                       return_outputs=True):
        """Full BPTT gradients via autograd through the unrolled forward +
        update_M_matrix loop. Returns {param_name: grad, ..., 'loss', 'outputs'}.
        With m_activation='linear' and no bounds, update_M_matrix is fully
        differentiable, so autograd through it is exactly BPTT. Records a fwd/bwd
        wall-time split in self._bptt_fwd_s / _bptt_bwd_s (CUDA-synced) for timing."""
        B, T, _ = inputs.shape
        _cuda = self.W_output.is_cuda
        if _cuda:
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        self.reset_state(B=B)
        outs = []
        for t in range(T):
            x_t = inputs[:, t, :]
            hidden_pre, _ = self.mp_layer(x_t)
            hidden = self.act_fn(hidden_pre)
            out = F.linear(hidden, self.W_output, self.b_output)
            outs.append(out)
            self.mp_layer.update_M_matrix(x_t, hidden)
        outputs = torch.stack(outs, dim=1)

        # BPTT only needs the SCALAR loss (autograd supplies the param grads); the
        # default helper's analytic grad_output would be dead work, so skip it.
        # Bit-identical loss + backward graph → identical gradients.
        if loss_and_grad is masked_mse_loss_and_output_grad:
            loss = masked_mse_loss_only(outputs, labels, masks)
        else:
            loss, _ = loss_and_grad(outputs, labels, masks)
        params = self._trainable_params()
        if _cuda:
            torch.cuda.synchronize()
        _t1 = time.perf_counter()
        grads = torch.autograd.grad(loss, list(params.values()))
        if _cuda:
            torch.cuda.synchronize()
        _t2 = time.perf_counter()
        self._bptt_fwd_s = _t1 - _t0
        self._bptt_bwd_s = _t2 - _t1
        # autograd.grad returns fresh tensors we own; detach is enough (no clone).
        result = {k: g.detach() for k, g in zip(params, grads)}
        result['loss'] = loss.detach()
        result['outputs'] = outputs.detach() if return_outputs else None
        return result

    @torch.no_grad()
    def _prepass_output_grad(self, inputs, labels, masks, loss_and_grad, eta, lam,
                             update_masks):
        """Forward-only pre-pass for a CUSTOM loss. Advances M with the same
        clean-config fast update the accumulation loop uses (M/traces never depend
        on grad_output, so the M-trajectory is identical), builds the full output
        sequence, and returns (loss, grad_output_seq) from loss_and_grad. A general
        loss's dL/d output_t can couple across time, so the per-step output gradient
        cannot be formed until every output exists; the accumulation pass then
        consumes grad_output_seq[:, t]. Keeps grads CONSISTENT with the reported
        loss, matching bptt_gradients()."""
        mp = self.mp_layer
        B, T, _ = inputs.shape
        dev, dt = inputs.device, inputs.dtype
        self.reset_state(B=B)
        outputs = torch.empty(B, T, self.n_output, dtype=dt, device=dev)
        for t in range(T):
            x_t = inputs[:, t, :]
            hidden_pre, _ = mp(x_t)
            hidden = self.act_fn(hidden_pre)
            outputs[:, t, :] = F.linear(hidden, self.W_output, self.b_output)
            um = None if update_masks is None else update_masks[:, t]
            mp.update_M_matrix_local_fast(x_t, hidden, eta=eta, lam=lam, update_mask=um)
        loss, grad_output_seq = loss_and_grad(outputs, labels, masks)
        return loss, grad_output_seq

    @torch.no_grad()
    def _local_sequence_gradients(self, inputs, labels, masks, mode,
                                  loss_and_grad=masked_mse_loss_and_output_grad,
                                  update_masks=None,
                                  return_outputs=True):
        """Shared forward-mode local-learning loop.

        This keeps the local rules mathematically identical but fuses the hot
        direct/diagonal operations:
          * local_direct never materializes E_dir;
          * local_diag_rflo never materializes E_hat;
          * eta/lam expansion is cached once per sequence;
          * the clean local M update skips the general update_M_matrix path;
          * return_outputs=False avoids storing outputs when using the default
            masked-MSE loss.
        """
        mp = self.mp_layer
        mp.assert_local_config()
        assert mode in ('exact', 'diag', 'direct')
        if mp.m_update_type == 'hebb_pre':
            mode = 'direct'

        B, T, _ = inputs.shape
        dev, dt = inputs.device, inputs.dtype
        eta, lam = mp._eta_lam_full()   # constant over the unroll; also used by the pre-pass

        # Custom loss: a forward-only pre-pass supplies grad_output_seq (dL/d
        # outputs) and the loss from the SUPPLIED loss_and_grad, so the returned
        # grads match the reported loss instead of the hard-coded masked-MSE one.
        # M/traces never depend on grad_output, so re-running the forward after
        # reset_state reproduces the identical M-trajectory.
        default_loss = (loss_and_grad is masked_mse_loss_and_output_grad)
        need_outputs = return_outputs or (not default_loss)
        prepass_loss, grad_output_seq = (None, None)
        if not default_loss:
            prepass_loss, grad_output_seq = self._prepass_output_grad(
                inputs, labels, masks, loss_and_grad, eta, lam, update_masks)

        self.reset_state(B=B)
        if mode == 'exact':
            mp.reset_local_learning_state(B=B)
        elif mode == 'diag':
            mp.reset_diag_rflo_state(B=B)

        # Single hidden boundary: exact_spatial uses W_output, the random modes
        # use the one fixed B_feedback (they coincide here — ordinary FA).
        feedback = self.W_output if self.feedback_mode == 'exact_spatial' else self.B_feedback
        step_fn = mp.step_fn_for(mode)      # resolve the per-step rule ONCE (no in-loop branch)
        # Tier-B: if compilation is on and this mode/config supports it, use the
        # fused compiled core (grads + M/A/Q update in one region); else eager.
        use_compiled = mp.can_compile_step(mode)

        grad_W = torch.zeros_like(mp.W)
        grad_b = torch.zeros_like(mp.b)

        # Time-major, contiguous views: per-step rows inputs_T[t] are contiguous
        # (B, ·) instead of strided slices of a (B, T, ·) tensor — one upfront copy
        # for faster per-step reads over a long unroll. Same math.
        inputs_T = inputs.transpose(0, 1).contiguous()
        labels_T = labels.transpose(0, 1).contiguous()
        masks_T = masks.transpose(0, 1).contiguous()
        um_T = update_masks.transpose(0, 1).contiguous() if update_masks is not None else None

        # Persistent scratch for the readout-gradient contraction, stored time-major
        # (T, B, ·) so the per-step writes are contiguous; fully overwritten each
        # step so no zeroing. These NEVER escape (consumed by the einsum below), so
        # reusing one buffer across calls is safe. outputs is a FRESH allocation
        # (it is returned via .detach(), which shares storage — a scratch buffer
        # would be clobbered on the next call), (B, T, ·) for the caller.
        go_seq = self._scratch('local_go', (T, B, self.n_output), dt, dev)
        hid_seq = self._scratch('local_hid', (T, B, self.n_hidden), dt, dev)
        outputs = torch.empty(B, T, self.n_output, dtype=dt, device=dev) if need_outputs else None
        loss_sum = torch.zeros((), dtype=dt, device=dev)

        N = B * T * self.n_output

        for t in range(T):
            x_t = inputs_T[t]

            hidden_pre, _ = mp(x_t)
            hidden = self.act_fn(hidden_pre)
            output = F.linear(hidden, self.W_output, self.b_output)
            if need_outputs:
                outputs[:, t, :] = output

            m_t = masks_T[t]
            if default_loss:
                diff_t = m_t * output - m_t * labels_T[t]
                grad_output = (2.0 / N) * m_t * diff_t
                loss_sum = loss_sum + (diff_t * diff_t).sum()
            else:
                grad_output = grad_output_seq[:, t, :]

            ell = grad_output @ feedback
            phi_prime = self.act_fn_p(hidden_pre)
            um = None if um_T is None else um_T[t]

            if use_compiled:
                # Fused core computes grads AND advances M/A/Q — no separate update.
                grad_W_t, grad_b_t = mp.compiled_step_and_update(
                    mode, x_t, hidden, phi_prime, ell, eta, lam, um)
            else:
                grad_W_t, grad_b_t = step_fn(x_t, phi_prime, ell, eta, lam, um)
                mp.update_M_matrix_local_fast(x_t, hidden, eta=eta, lam=lam, update_mask=um)
            grad_W += grad_W_t
            grad_b += grad_b_t

            go_seq[t] = grad_output
            hid_seq[t] = hidden

        # Readout grads from the time-major buffers (one contraction each).
        grad_Wout = torch.einsum('TBa,TBi->ai', go_seq, hid_seq)
        grad_bout = go_seq.sum(dim=(0, 1))

        if default_loss:
            loss = masked_mse_loss_only(outputs, labels, masks) if outputs is not None else (loss_sum / N)
        else:
            loss = prepass_loss   # from the pre-pass, same outputs → identical value

        params = self._trainable_params()
        all_grads = {'W': grad_W, 'b': grad_b, 'W_output': grad_Wout, 'b_output': grad_bout}
        result = {k: all_grads[k] for k in params}
        result['loss'] = loss.detach()
        result['outputs'] = outputs.detach() if return_outputs and outputs is not None else None
        return result

    def local_gradients(self, inputs, labels, masks, **kwargs):
        """Exact row-local (eligibility-trace) gradients — no BPTT. Equals BPTT
        in the clean config. See _local_sequence_gradients."""
        return self._local_sequence_gradients(inputs, labels, masks, 'exact', **kwargs)

    def local_diag_rflo_gradients(self, inputs, labels, masks, **kwargs):
        """Diagonal / same-synapse RFLO gradients — an approximation to BPTT
        (drops off-synapse plastic sensitivities). See _local_sequence_gradients."""
        return self._local_sequence_gradients(inputs, labels, masks, 'diag', **kwargs)

    def local_direct_gradients(self, inputs, labels, masks, **kwargs):
        """Direct/instantaneous local gradients — the strongest approximation:
        treats M_{t-1} as a stop-gradient state and drops all plasticity-mediated
        temporal credit (no P/A/Q trace). See _local_sequence_gradients."""
        return self._local_sequence_gradients(inputs, labels, masks, 'direct', **kwargs)

    def sequence_gradients(self, inputs, labels, masks, **kwargs):
        """Dispatch on self.learning_rule and write the gradients into each
        parameter's .grad (so an optimizer.step() works exactly as with
        loss.backward()). Returns the gradient/loss dict."""
        if self.learning_rule == 'bptt':
            grads = self.bptt_gradients(inputs, labels, masks, **kwargs)
        elif self.learning_rule == 'local_exact_rowlocal':
            grads = self.local_gradients(inputs, labels, masks, **kwargs)
        elif self.learning_rule == 'local_diag_rflo':
            grads = self.local_diag_rflo_gradients(inputs, labels, masks, **kwargs)
        elif self.learning_rule == 'local_direct':
            grads = self.local_direct_gradients(inputs, labels, masks, **kwargs)
        else:
            raise ValueError(f"unknown learning_rule '{self.learning_rule}'")

        for name, p in self._trainable_params().items():
            p.grad = grads[name].clone()
        return grads

class DeepMultiPlasticNet(MultiPlasticNetBase):
    """
    N-layer feedforward setup: an optional trainable input embedding, then one or
    more multi-plastic layers stacked in sequence, followed by a single readout
    layer, i.e.
        u -> [W_initial_linear, act] -> [MP_1] -> [MP_2] -> ... -> [W_output] -> y
    The number of MP layers is len(n_neurons) - 2 (one per hidden width); pass
    n_neurons=[in, h1, h2, out] for two MP layers, etc.

    BPTT (bptt_gradients / sequence_gradients with learning_rule='bptt') supports
    ANY number of MP layers — it is autograd through the unrolled forward, which
    already iterates over self.mp_layers. The local rules (local_*) ALSO support
    any number of MP layers: each layer keeps its own intra-layer eligibility
    (exact row-local / diagonal RFLO / direct) and is credited by a same-time
    inter-layer learning signal (see _same_time_boundary_signals). Those hidden-
    layer updates are surrogates — they omit temporal paths through the plastic
    state of the layers above them — even when the intra-layer eligibility is exact.

    feedback_mode selects HOW the inter-layer learning signal is formed (see the
    module-level _FEEDBACK_MODES table). exact_spatial differs from the random modes
    at ANY depth; layerwise_fa and direct_fa differ from each other only when there
    is more than one trainable activity boundary (multiple MP layers, or one MP
    layer plus a trainable input embedding):
      'exact_spatial'  exact same-time SPATIAL gradient (weight transport
                       everywhere). With local_exact_rowlocal the TOP plastic
                       layer's gradient is exact vs BPTT (spatial path via the true
                       W_output + its own temporal M-path via the exact P trace);
                       LOWER layers stay surrogates, and diag/direct make even the
                       top only approximate. Legacy alias: 'exact_readout'.
      'layerwise_fa'   conventional recursive feedback alignment — a fixed random
                       matrix at EVERY boundary, so the whole feedback pathway is
                       weight-transport-free.
      'direct_fa'      direct feedback alignment — the readout error is projected
                       directly onto every hidden layer through its own random matrix.
    For a single MP layer with no trainable input embedding, layerwise_fa and
    direct_fa reduce to ordinary one-hidden-layer feedback alignment. With a
    trainable embedding the MP-layer signals coincide in form, but the embedding
    receives recursive feedback under layerwise_fa and a direct output projection
    under direct_fa.
    """

    def __init__(self, net_params, verbose=False):
        cfg = copy.deepcopy(net_params)

        # Mar 16th: add input layer
        self.input_layer_active = cfg.get('input_layer_add', False)
        self.input_layer_active_trainable = cfg.get('input_layer_add_trainable', False)

        arch = list(cfg['n_neurons'])
        if self.input_layer_active:
            arch.insert(1, cfg.get('linear_embed', 128)) # add an initial linear embedding layer

        n_layers = len(arch) - 1
        self.n_input = cfg['n_neurons'][0]
        self.n_hidden = cfg['n_neurons'][1]
        self.n_output = cfg['n_neurons'][-1]

        self.output_matrix = cfg['output_matrix']

        # Learning rule / feedback for sequence_gradients() (see the rule methods
        # below). Same options as MultiPlasticNet. 'local' aliases exact.
        _rule = cfg.get('learning_rule', 'bptt')
        if _rule == 'local':
            _rule = 'local_exact_rowlocal'
        assert _rule in ('bptt', 'local_exact_rowlocal', 'local_diag_rflo',
                         'local_direct'), f"unknown learning_rule '{_rule}'"
        self.learning_rule = _rule
        # input_mode decouples the TRAINABLE INPUT EMBEDDING's learning rule from the
        # MP-layer learning_rule, so any RULES_TO_RUN × input-rule combination can be
        # compared. Only meaningful when input_layer_add_trainable is on.
        #   'match'        — the embedding follows learning_rule (historical default:
        #                    exact autograd under bptt, the 3-factor local rule under
        #                    a local rule). Zero behavior change from before.
        #   'exact'        — the embedding is ALWAYS trained by the true BPTT gradient
        #                    (dL/dW_in via autograd), even during a local MP run.
        #   'three_factor' — the embedding ALWAYS uses the DIRECT 3-factor local rule
        #                    (ell_h[0] ⊙ phi'(embed_pre)) · uᵀ, even during a bptt run.
        # The splice lives in sequence_gradients (recompute the OTHER method's grad
        # and overwrite only W_in/b_in) so no new gradient math is introduced.
        self.input_mode = cfg.get('input_mode', 'match')
        assert self.input_mode in ('match', 'exact', 'three_factor'), \
            f"unknown input_mode '{self.input_mode}'"
        # feedback_mode governs how each hidden layer's learning signal is formed
        # in the local rules (see the _FEEDBACK_MODES table and
        # _same_time_boundary_signals). B_feedback_init is stashed for the
        # per-layer FA buffers built after the MP layers exist.
        self.feedback_mode = canonical_feedback_mode(cfg.get('feedback_mode', 'exact_spatial'))
        self._B_feedback_init = cfg.get('B_feedback_init', 'xavier')

        # ── Homeostatic gain control on the FA learning signals (opt-in) ──────
        # Deep random-feedback pathways can attenuate/amplify each lower boundary's
        # learning signal ell (products of random matrices × phi'). This keeps a
        # per-boundary running RMS estimate v_k and rescales ell_k -> gain_k * ell_k
        # / sqrt(v_k + eps) — a fully local, weight-transport-free gain control
        # (per-layer RMSNorm on the teaching signal). OFF by default and applied
        # ONLY to the random feedback modes: exact_spatial must stay bit-identical to
        # BPTT, so it is never rescaled (asserted in __init__). See
        # _normalize_learning_signal / _same_time_boundary_signals.
        self.feedback_normalize = bool(cfg.get('feedback_normalize', False))
        self.ell_rms_beta = float(cfg.get('feedback_normalize_beta', 0.9))
        self.ell_rms_eps = float(cfg.get('feedback_normalize_eps', 1e-8))
        self._ell_gain_val = float(cfg.get('feedback_normalize_gain', 1.0))
        if self.feedback_normalize and self.feedback_mode == 'exact_spatial':
            raise ValueError(
                "feedback_normalize is only valid for the random feedback modes "
                "('layerwise_fa'/'direct_fa'); it would break exact_spatial's "
                "exact==BPTT guarantee. Got feedback_mode='exact_spatial'.")

        super().__init__(cfg, cfg['n_neurons'][-2], output_matrix=self.output_matrix, verbose=verbose)

        # Fixed random feedback matrix for the TOP (readout → top-hidden) boundary,
        # used by 'layerwise_fa' (top + every inter-layer boundary). Must match
        # W_output's shape — (n_output, top_hidden) where top_hidden = last
        # plastic-layer width = cfg['n_neurons'][-2], NOT the first hidden width
        # self.n_hidden (wrong for a deep unequal-width stack). 'direct_fa' does NOT
        # use B_feedback (it projects the output error directly to every layer,
        # incl. the top).
        if self.feedback_mode == 'layerwise_fa':
            self.register_buffer('B_feedback', torch.tensor(
                rand_weight_init(cfg['n_neurons'][-2], self.n_output,
                                 init_type=self._B_feedback_init),
                dtype=self.W_output.dtype))

        # Creates all the MP layers
        self.param_clamping = True # Always have param clamping for MP layers because lam bounds
        
        input_init_type = net_params.get("input_init_type", "xavier")

        if self.input_layer_active:
            self.W_initial_linear = nn.Linear(arch[0], arch[1])

            # self.W_initial_linear.weight.data = torch.tensor(
            #     rand_weight_init(arch[0], arch[1], init_type=net_params.get('W_init', 'xavier')),
            #     dtype=torch.float
            # )
            
            with torch.no_grad():
                if input_init_type == "identity":
                    W = self.W_initial_linear.weight
                    W.zero_()
                    k = min(W.size(0), W.size(1))
                    W[:k, :k].copy_(torch.eye(k, device=W.device, dtype=W.dtype))

                elif input_init_type == "identity_noise":
                    eps = net_params.get("identity_eps", 1e-3)
                    W = self.W_initial_linear.weight
                    W.zero_()
                    k = min(W.size(0), W.size(1))
                    W[:k, :k].copy_(torch.eye(k, device=W.device, dtype=W.dtype))
                    W.add_(eps * torch.randn_like(W))

                elif input_init_type == "orthogonal":
                    torch.nn.init.orthogonal_(self.W_initial_linear.weight, gain=1.0)

                else:
                    self.W_initial_linear.weight.copy_(torch.tensor(
                        rand_weight_init(arch[0], arch[1], init_type=input_init_type),
                        device=self.W_initial_linear.weight.device,
                        dtype=self.W_initial_linear.weight.dtype
                    ))
            
            if not self.input_layer_active_trainable:
                print(f'  Input Layer Frozen.')
                self.W_initial_linear.weight.requires_grad = False

            if net_params.get('input_layer_bias', False):
                self.W_initial_linear.bias.data = torch.tensor(
                    rand_weight_init(arch[1], init_type='gaussian'),
                    dtype=torch.float
                )
            else:
                self.W_initial_linear.bias = None

        self.mp_layers = []

        start_layer_count = 1 if self.input_layer_active else 0
        # if additional input layer is added, shift the starting index of layer counting from 1.
        # One MP layer per hidden width: range below yields len(n_neurons) - 2 layers
        # (e.g. n_neurons=[in, h1, h2, out] -> two MP layers).
        for mpl_idx in range(start_layer_count, n_layers - 1):
            ml_key = f'ml_params{mpl_idx}' if f'ml_params{mpl_idx}' in cfg else 'ml_params'
            # Updates some parameters for each new MPL
            cfg[ml_key]['dt'] = self.dt
            cfg[ml_key]['mpl_name'] = str(mpl_idx)
            cfg[ml_key]['n_input'] = arch[mpl_idx]
            cfg[ml_key]['n_output'] = arch[mpl_idx + 1]

            setattr(
                self,
                f'mp_layer{mpl_idx}',
                MultiPlasticLayer(cfg[ml_key], output_matrix=self.output_matrix, verbose=verbose)
            )

            self.mp_layers.append(getattr(self, 'mp_layer{}'.format(mpl_idx)))
            self.params.extend([param+str(mpl_idx) for param in self.mp_layers[-1].params])

        # ── Per-boundary fixed random feedback for layerwise / direct FA ──────
        # (built AFTER the MP layers so their widths are known; B_feedback for the
        # top boundary is already registered above.) Buffers are registered under
        # names, and only the NAMES are cached in a list — never the tensor objects,
        # so .to()/.double() moves are always seen via getattr at use time.
        #
        #   layerwise_fa: a fixed random B_inter[n] (shape = layers[n].W.shape =
        #     (post, pre)) at EVERY backprojected boundary, replacing the modulated-
        #     weight transpose. With the top B_feedback this makes the ENTIRE
        #     feedback pathway weight-transport-free (conventional recursive FA).
        #   direct_fa: a fixed random B_direct[k] projecting the readout error
        #     DIRECTLY onto h[k] for every k (no sequential backward chain).
        #     B_direct[k] maps (n_output) -> dim(h[k]); k == L reproduces the
        #     readout -> top-hidden role that B_feedback plays in the other modes.
        self._B_inter_names = []
        self._B_direct_names = []
        L = len(self.mp_layers)
        if self.feedback_mode == 'layerwise_fa':
            for n, mp in enumerate(self.mp_layers):
                name = f'B_inter{n}'                 # shape (post, pre) == mp.W.shape
                self.register_buffer(name, torch.tensor(
                    rand_weight_init(mp.n_input, mp.n_output, init_type=self._B_feedback_init),
                    dtype=self.W_output.dtype))
                self._B_inter_names.append(name)
        elif self.feedback_mode == 'direct_fa':
            # dim(h[k]) = input width of layer k for k < L, output width of the top
            # layer for k == L. Register B_direct[0..L] (index 0 credits the embedding).
            hdims = [mp.n_input for mp in self.mp_layers] + [self.mp_layers[-1].n_output]
            for k, dim_hk in enumerate(hdims):
                name = f'B_direct{k}'                # shape (n_output, dim(h[k]))
                self.register_buffer(name, torch.tensor(
                    rand_weight_init(dim_hk, self.n_output, init_type=self._B_feedback_init),
                    dtype=self.W_output.dtype))
                self._B_direct_names.append(name)

        # Homeostatic-gain state, allocated only when feedback_normalize is on (so
        # the default net's buffer set — hence its state_dict — is unchanged). One
        # entry per boundary index 0..L (0 = embedding boundary, 1..L = MP layers).
        # ell_gain is a fixed target-gain buffer (moves with .to()); the running RMS
        # v_k is a per-sequence TEMPORAL state (a plain tensor attribute, reset each
        # unroll like self.M — never a buffer, so it stays out of state_dict).
        if self.feedback_normalize:
            self.register_buffer(
                'ell_gain', torch.full((L + 1,), self._ell_gain_val, dtype=self.W_output.dtype))
            self.ell_rms = None   # allocated/zeroed by reset_feedback_norm_state(B)


    def forward(self, inputs, run_mode='minimal', verbose=False):
        """
        """
        # 2025-11-19: the x_t in the MPN paper is the "input to MPN layer"
        # namely after the MLP 
        if self.input_layer_active:
            x = self.W_initial_linear(inputs)
            x = self.act_fn(x)
        else:
            x = inputs

        layer_input = x  # read-only downstream (never mutated in-place) → no clone

        mpl_activities = [x,] # Used for updating the M matrices

        db = {} if run_mode in ('track_states',) else None

        for mpl_idx, mp_layer in enumerate(self.mp_layers):
            # The pre-layer activity is only needed for the track_states log below;
            # capture it there (as a detached snapshot) instead of cloning every step.
            layer_input_old = layer_input
            hidden_pre, db_mp = mp_layer(layer_input, run_mode=run_mode)

            layer_input = self.act_fn(hidden_pre)

            if run_mode in ('debug',):
                print(f'  MP Layer {mpl_idx} forward.')
                print('   Pre-act mean {:.2e} Post-act mean {:.2e}'.format(
                    torch.mean(hidden_pre.detach()), torch.mean(layer_input.detach())
                ))

            mpl_activities.append(layer_input) # Postsyn activity
            if run_mode in ('track_states',):
                db['hidden_pre{}'.format(mp_layer.mp_layer_name)] = hidden_pre.detach()
                db['hidden{}'.format(mp_layer.mp_layer_name)] = layer_input.detach()
                db['M{}'.format(mp_layer.mp_layer_name)] = db_mp['M']
                db['b{}'.format(mp_layer.mp_layer_name)] = db_mp['b']
                db['input{}'.format(mp_layer.mp_layer_name)] = layer_input_old.detach()

        if run_mode in ('debug',):
            print(f'  Output layer forward.')

        output = F.linear(layer_input, self.W_output, self.b_output)
        
        return output, mpl_activities, db

    def network_step(self, current_input, run_mode='minimal', verbose=False, seq_idx=None):
        """
        Performs a single batch pass forward for the network. This mostly consists of a forward pass and
        the associated updates to internal states (i.e. the modulations)

        This should not be passed a full sequence of data, only data from a given time point
        """

        assert len(current_input.shape) == 2
        if run_mode in ('debug',):
            print(f' Network step:')

        # current_input is per-time input, so has shape (batch_size, input_size)
        output, mpl_activities, db = self.forward(current_input, run_mode=run_mode, verbose=verbose)

        # M updated internally when this is called
        for mpl_idx, mp_layer in enumerate(self.mp_layers):
            if run_mode in ('debug',):
                print(f'  MP Layer {mpl_idx} M update.')
                print('   Pre mean {:.2e} post mean {:.2e}'.format(
                    torch.mean(mpl_activities[mpl_idx].detach()), torch.mean(mpl_activities[mpl_idx + 1].detach())
                ))
                print('   M mag mean {:.2e} max {:.2e}'.format(
                    torch.mean(torch.abs(mp_layer.M.detach())), torch.max(torch.abs(mp_layer.M.detach()))
                ))

            _ = mp_layer.update_M_matrix(mpl_activities[mpl_idx], mpl_activities[mpl_idx + 1])

        return output, mpl_activities, db

    # ─── Learning rules for the single-MP-layer deep net (+ optional input embed) ──
    # Mirrors MultiPlasticNet's rule suite for the architecture
    #   u -> [W_initial_linear, act] -> x -> [MP_1] -> ... -> [MP_k] -> [W_output] -> y
    # For k == 1 this is directly comparable to the RNN (input -> hidden -> output),
    # with the plastic M playing the role the RNN's recurrence plays.
    # BPTT trains ALL params (every MP layer + the input embedding) exactly via
    # autograd through the unrolled forward, for ANY number of MP layers. The local
    # rules train EVERY MP layer with its own eligibility traces, credited by a
    # same-time inter-layer learning signal backprojected through the modulated
    # weights of the layers above (M-mediated temporal paths through upper layers
    # dropped), and train the input embedding with the same DIRECT 3-factor rule
    # (backproject the boundary signal through layer 0's modulated weights, then
    # multiply by the embedding activation derivative and the raw input — the RNN's
    # RFLO treatment of its input weights). Works for any number of MP layers.

    def _has_trainable_embed(self):
        return (self.input_layer_active and self.W_initial_linear.weight.requires_grad)

    # ── Deep-local forward + same-time boundary-signal helpers ────────────────
    # These implement the two ingredients the multi-MP-layer local rules need
    # (see the class docstring): a forward that stops short of the M update, and
    # a top-down learning-signal pass that backprojects the readout error through
    # every layer's MODULATED weights W_eff = W ⊙ (1 + M_{t-1}). Both read the
    # frozen M_{t-1}, so they must run before any layer advances its M.

    def _forward_local_stack(self, u_t):
        """Forward one time step through the whole stack WITHOUT updating any M.

        Returns (output, h, z, phi_p, embed_pre) with the zero-based indexing the
        deep-local loop uses:
            h[0]      input representation (= embedding output, or raw u_t)
            z[n]      preactivation of MP layer n           (B, d_{n+1})
            phi_p[n]  phi'(z[n])                            (B, d_{n+1})
            h[n+1]    output of MP layer n                  (B, d_{n+1})
            output    readout(h[L])
        embed_pre is the embedding PRE-activation (for the embedding grad) or None.
        Every mp(...) call consumes that layer's own M_{t-1}; nothing is advanced.
        """
        if self.input_layer_active:
            embed_pre = self.W_initial_linear(u_t)
            h0 = self.act_fn(embed_pre)
        else:
            embed_pre = None
            h0 = u_t

        h = [h0]
        z = []
        phi_p = []
        for mp in self.mp_layers:
            z_n, _ = mp(h[-1])              # uses this layer's M_{t-1}
            phi_p.append(self.act_fn_p(z_n))
            z.append(z_n)
            h.append(self.act_fn(z_n))

        output = F.linear(h[-1], self.W_output, self.b_output)
        return output, h, z, phi_p, embed_pre

    def reset_feedback_norm_state(self, B=1):
        """Zero the per-sequence running RMS of every boundary learning signal.
        No-op unless feedback_normalize is on. Called once per unroll (parallel to
        reset_state), so the homeostatic gain control starts fresh each sequence and
        never leaks state across calls. B is unused (the RMS is a scalar per
        boundary, averaged over batch and units) but kept for signature symmetry."""
        if not getattr(self, 'feedback_normalize', False):
            return
        L = len(self.mp_layers)
        self.ell_rms = torch.zeros(L + 1, dtype=self.W_output.dtype, device=self.W_output.device)

    def _normalize_learning_signal(self, k, ell):
        """Homeostatic gain control on boundary k's learning signal (opt-in).

        Advances the per-boundary running RMS estimate and rescales ell to a fixed
        target gain — a fully local, weight-transport-free normalization (per-layer
        RMSNorm on the teaching signal):
            v_k   <- beta v_k + (1 - beta) mean_{B,i}[ ell_{i}^2 ]
            ell~  =  gain_k * ell / sqrt(v_k + eps)
        Uses only the activity of the neurons receiving the signal (no forward
        weights/gradients). ell is (B, dim h[k]); returns the rescaled signal."""
        power = ell.square().mean()
        v = self.ell_rms[k] * self.ell_rms_beta + power * (1.0 - self.ell_rms_beta)
        self.ell_rms[k] = v
        return self.ell_gain[k] * ell / torch.sqrt(v + self.ell_rms_eps)

    def _maybe_normalize_boundary_signals(self, ell_h, need_input_signal):
        """Post-pass homeostatic gain control over a fully-built ell_h list; a no-op
        unless feedback_normalize is on (so the default/exact path is untouched).
        Every filled boundary (1..L, plus 0 when the embedding is trainable) is
        rescaled independently. Because the layerwise recursion is LINEAR, rescaling
        as a post-pass preserves each boundary's direction and only controls its
        magnitude, so it is equivalent (up to the running-RMS history) to normalizing
        inside the recursion — and matches the reference _normalize_learning_signal
        loop. exact_spatial never reaches here with the flag on (asserted off)."""
        if not self.feedback_normalize:
            return ell_h
        for k in range(1, len(self.mp_layers) + 1):
            ell_h[k] = self._normalize_learning_signal(k, ell_h[k])
        if need_input_signal and ell_h[0] is not None:
            ell_h[0] = self._normalize_learning_signal(0, ell_h[0])
        return ell_h

    def _same_time_boundary_signals(self, grad_output, phi_p, need_input_signal):
        """Same-time layer-boundary learning signals using the frozen M_{t-1}.

        ell_h[n] is the surrogate dL/dh[n] (the signal at the INPUT boundary of MP
        layer n). How it is formed depends on self.feedback_mode (see _FEEDBACK_MODES):

          exact_spatial — exact same-time spatial gradient:
            ell_h[L] = grad_output @ W_output
            ell_h[n] = W_eff^{(n)T} ( ell_h[n+1] ⊙ phi'(z[n]) )      for n < L
          layerwise_fa — conventional recursive feedback alignment: a fixed random
            matrix at EVERY boundary (no actual W or M in the feedback pathway):
            ell_h[L] = grad_output @ B_feedback
            ell_h[n] = ( ell_h[n+1] ⊙ phi'(z[n]) ) @ B_inter[n]      for n < L
          direct_fa — direct feedback alignment: the readout error is projected
            DIRECTLY onto every hidden layer through its own fixed random matrix
            (no backward chain, no phi'/M in the feedback pathway):
            ell_h[k] = grad_output @ B_direct[k]                      for all k

        For exact_spatial / layerwise_fa the recursion drops temporal paths through
        the plastic state of upper layers but keeps the same-time spatial structure.
        Layer n's eligibility routine is fed ell_h[n+1] (its OUTPUT-boundary signal)
        — NOT premultiplied by phi'.

        Returns ell_h, a length-(L+1) list. ell_h[1..L] are always filled (every
        MP layer's step needs its output signal); ell_h[0] is filled only when
        need_input_signal (the trainable input embedding). For a single MP layer
        with no embedding, exact_spatial / layerwise_fa compute exactly one matmul.

        When feedback_normalize is on (random modes only), each fully-constructed
        boundary signal is passed through the homeostatic gain control before return
        (a post-pass, so the layerwise recursion itself is built from raw signals and
        only its outputs are rescaled — matching _normalize_learning_signal).
        """
        L = len(self.mp_layers)
        ell_h = [None] * (L + 1)

        # direct_fa: every boundary is an independent random projection of the
        # readout error — no backward chain, no phi'/M in the feedback pathway.
        if self.feedback_mode == 'direct_fa':
            for k in range(1, L + 1):
                ell_h[k] = grad_output @ getattr(self, self._B_direct_names[k])
            if need_input_signal:
                ell_h[0] = grad_output @ getattr(self, self._B_direct_names[0])
            return self._maybe_normalize_boundary_signals(ell_h, need_input_signal)

        # exact_spatial vs layerwise_fa: the former uses the exact (modulated)
        # weight transpose at every boundary (weight transport), the latter a fixed
        # random matrix at every boundary (transport-free). Top boundary first, then
        # recurse down; ell_h[0] only when the trainable embedding needs it.
        layerwise = (self.feedback_mode == 'layerwise_fa')
        ell_h[L] = grad_output @ (self.B_feedback if layerwise else self.W_output)
        for n in range(L - 1, 0, -1):
            delta_n = ell_h[n + 1] * phi_p[n]
            if layerwise:
                ell_h[n] = delta_n @ getattr(self, self._B_inter_names[n])
            else:
                ell_h[n] = self.mp_layers[n].backproject_through_modulated_weights_fast(delta_n)
        if need_input_signal:
            delta_0 = ell_h[1] * phi_p[0]
            if layerwise:
                ell_h[0] = delta_0 @ getattr(self, self._B_inter_names[0])
            else:
                ell_h[0] = self.mp_layers[0].backproject_through_modulated_weights_fast(delta_0)

        return self._maybe_normalize_boundary_signals(ell_h, need_input_signal)

    def _trainable_params(self):
        """Trainable tensors this net computes gradients for, keyed by name.
        The first MP layer keeps the bare keys 'W'/'b' (back-compat with the
        single-MP-layer local rules and validate); any further MP layers are keyed
        'W{j}'/'b{j}' by their position j in self.mp_layers. Readout is 'W_output'/
        'b_output'; the optional trainable input embedding is 'W_in'/'b_in'."""
        ps = {'W_output': self.W_output}
        for j, mp in enumerate(self.mp_layers):
            suffix = '' if j == 0 else str(j)
            ps[f'W{suffix}'] = mp.W
            if mp.layer_bias:
                ps[f'b{suffix}'] = mp.b
        if self.b_output_active:
            ps['b_output'] = self.b_output
        if self.input_layer_active:
            ps['W_in'] = self.W_initial_linear.weight
            if self.W_initial_linear.bias is not None:
                ps['b_in'] = self.W_initial_linear.bias
        return {k: v for k, v in ps.items() if v.requires_grad}

    def bptt_gradients(self, inputs, labels, masks,
                       loss_and_grad=masked_mse_loss_and_output_grad,
                       return_outputs=True):
        """Full BPTT via autograd through the unrolled deep forward + M-update.
        Trains every parameter (all MP layers + the input embedding) exactly, for
        any number of MP layers. Records a fwd/bwd wall-time split in
        self._bptt_fwd_s / _bptt_bwd_s (CUDA-synced) for the timing readout."""
        B, T, _ = inputs.shape
        _cuda = self.W_output.is_cuda
        if _cuda:
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        self.reset_state(B=B)
        outs = []
        for t in range(T):
            out, _, _ = self.network_step(inputs[:, t, :], seq_idx=t)
            outs.append(out)
        outputs = torch.stack(outs, dim=1)

        # BPTT only needs the SCALAR loss (autograd supplies the param grads); the
        # default helper's analytic grad_output would be dead work, so skip it.
        # Bit-identical loss + backward graph → identical gradients.
        if loss_and_grad is masked_mse_loss_and_output_grad:
            loss = masked_mse_loss_only(outputs, labels, masks)
        else:
            loss, _ = loss_and_grad(outputs, labels, masks)
        params = self._trainable_params()
        if _cuda:
            torch.cuda.synchronize()
        _t1 = time.perf_counter()
        grads = torch.autograd.grad(loss, list(params.values()))
        if _cuda:
            torch.cuda.synchronize()
        _t2 = time.perf_counter()
        self._bptt_fwd_s = _t1 - _t0
        self._bptt_bwd_s = _t2 - _t1
        # autograd.grad returns fresh tensors we own; detach is enough (no clone).
        result = {k: g.detach() for k, g in zip(params, grads)}
        result['loss'] = loss.detach()
        result['outputs'] = outputs.detach() if return_outputs else None
        return result

    @torch.no_grad()
    def _prepass_output_grad(self, inputs, labels, masks, loss_and_grad, eta_lam,
                             update_masks):
        """Forward-only pre-pass for a CUSTOM loss (deep net). Runs the embedding +
        full MP stack with the same clean-config fast M update the accumulation
        loop uses (for EVERY layer), builds the full output sequence, and returns
        (loss, grad_output_seq) from loss_and_grad. See
        MultiPlasticNet._prepass_output_grad. eta_lam is the per-layer list of
        (eta, lam) expansions (one entry per MP layer)."""
        layers = self.mp_layers
        B, T, _ = inputs.shape
        dev, dt = inputs.device, inputs.dtype
        self.reset_state(B=B)
        outputs = torch.empty(B, T, self.n_output, dtype=dt, device=dev)
        for t in range(T):
            u_t = inputs[:, t, :]
            output, h, z, phi_p, embed_pre = self._forward_local_stack(u_t)
            outputs[:, t, :] = output
            um = None if update_masks is None else update_masks[:, t]
            for n, mp in enumerate(layers):
                eta_n, lam_n = eta_lam[n]
                mp.update_M_matrix_local_fast(h[n], h[n + 1], eta=eta_n, lam=lam_n,
                                              update_mask=um)
        loss, grad_output_seq = loss_and_grad(outputs, labels, masks)
        return loss, grad_output_seq

    @torch.no_grad()
    def _local_sequence_gradients(self, inputs, labels, masks, mode,
                                  loss_and_grad=masked_mse_loss_and_output_grad,
                                  update_masks=None,
                                  return_outputs=True):
        """Forward-mode local learning for the deep net — ANY number of MP layers.

        Per time step (see the class docstring and the module derivation) the
        ordering is strict:
          1. forward the whole stack with M_{t-1}                (_forward_local_stack)
          2. readout error grad_output
          3. every same-time boundary signal ell_h[.] with M_{t-1}
             (_same_time_boundary_signals) BEFORE any M is advanced
          4. each layer's local gradient from its own eligibility, credited by its
             OUTPUT-boundary signal ell_h[n+1]
          5. (fused into 4 for direct/diag) advance each layer's traces
          6. only now advance every layer's M from M_{t-1} to M_t

        Each layer keeps its own intra-layer eligibility (exact P/Q, diagonal A/Q,
        or direct); the inter-layer credit is the same-time spatial backprojection
        through the modulated weights, which drops temporal paths through upper
        plastic layers. For a single MP layer this reduces to the previous
        single-layer computation bit-for-bit (one top boundary matmul, same
        scratch buffers, same einsums).

        Loss handling mirrors MultiPlasticNet: default masked MSE is inline; a
        custom loss_and_grad triggers a forward-only pre-pass whose grad_output_seq
        feeds the accumulation pass (grads stay consistent with the reported loss).
        """
        layers = self.mp_layers
        L = len(layers)
        for mp in layers:
            mp.assert_local_config()
        assert mode in ('exact', 'diag', 'direct')
        # hebb_pre: M input-only → dM/dW = dM/db = 0, every plastic trace is zero,
        # so the trace modes collapse to the direct rule (per layer). _assoc already
        # zeroes the trace terms, but routing to 'direct' also skips the unused
        # trace allocation, exactly as the single-layer path does.
        if all(mp.m_update_type == 'hebb_pre' for mp in layers):
            mode = 'direct'

        B, T, _ = inputs.shape
        dev, dt = inputs.device, inputs.dtype
        eta_lam = [mp._eta_lam_full() for mp in layers]   # per-layer (eta, lam), constant over the unroll

        # Custom loss: forward-only pre-pass supplies grad_output_seq and the loss
        # (see MultiPlasticNet._local_sequence_gradients / _prepass_output_grad).
        default_loss = (loss_and_grad is masked_mse_loss_and_output_grad)
        need_outputs = return_outputs or (not default_loss)
        prepass_loss, grad_output_seq = (None, None)
        if not default_loss:
            prepass_loss, grad_output_seq = self._prepass_output_grad(
                inputs, labels, masks, loss_and_grad, eta_lam, update_masks)

        self.reset_state(B=B)
        self.reset_feedback_norm_state(B=B)   # per-sequence RMS state (no-op unless on)
        for mp in layers:
            if mode == 'exact':
                mp.reset_local_learning_state(B=B)
            elif mode == 'diag':
                mp.reset_diag_rflo_state(B=B)

        embed = self._has_trainable_embed()
        step_fns = [mp.step_fn_for(mode) for mp in layers]   # per-layer rule, resolved once
        # Tier-B fused compiled core (opt-in). Safe for a deep stack too: all
        # boundary signals are computed up front, and each layer's compiled step
        # reads/advances only its OWN M/traces, so advancing M[n] inside the
        # per-layer loop never disturbs another layer's step.
        use_compiled = all(mp.can_compile_step(mode) for mp in layers)

        grad_W = [torch.zeros_like(mp.W) for mp in layers]
        grad_b = [torch.zeros_like(mp.b) for mp in layers]

        # Time-major, contiguous views (see MultiPlasticNet._local_sequence_gradients).
        inputs_T = inputs.transpose(0, 1).contiguous()
        labels_T = labels.transpose(0, 1).contiguous()
        masks_T = masks.transpose(0, 1).contiguous()
        um_T = update_masks.transpose(0, 1).contiguous() if update_masks is not None else None

        # Persistent scratch (time-major, contiguous writes, fully overwritten →
        # no zeroing). go_seq/hid_seq/ga_seq/u_seq never escape (consumed by the
        # einsums below), so reuse across calls is safe; outputs is a fresh alloc
        # since it is returned (its .detach() shares storage). hid_seq holds the TOP
        # hidden activity h[L] fed to the readout, so it is sized by W_output's input
        # width (= last plastic width), NOT self.n_hidden (wrong for unequal widths).
        top_hidden_dim = self.W_output.shape[1]
        go_seq = self._scratch('dlocal_go', (T, B, self.n_output), dt, dev)
        hid_seq = self._scratch('dlocal_hid', (T, B, top_hidden_dim), dt, dev)
        outputs = torch.empty(B, T, self.n_output, dtype=dt, device=dev) if need_outputs else None
        loss_sum = torch.zeros((), dtype=dt, device=dev)
        if embed:
            has_bin = self.W_initial_linear.bias is not None
            ga_seq = self._scratch('dlocal_ga', (T, B, self.W_initial_linear.weight.shape[0]), dt, dev)
            u_seq = self._scratch('dlocal_u', (T, B, self.n_input), dt, dev)

        N = B * T * self.n_output

        for t in range(T):
            u_t = inputs_T[t]

            # 1. Forward the whole stack with M_{t-1} (no M advance yet).
            output, h, z, phi_p, embed_pre = self._forward_local_stack(u_t)
            if need_outputs:
                outputs[:, t, :] = output

            # 2. Readout error.
            m_t = masks_T[t]
            if default_loss:
                diff_t = m_t * output - m_t * labels_T[t]
                grad_output = (2.0 / N) * m_t * diff_t
                loss_sum = loss_sum + (diff_t * diff_t).sum()
            else:
                grad_output = grad_output_seq[:, t, :]

            um = None if um_T is None else um_T[t]

            # 3. All same-time boundary signals, using the still-frozen M_{t-1}.
            #    ell_h[n+1] credits MP layer n; ell_h[0] (only if the embedding is
            #    trainable) credits the input embedding.
            ell_h = self._same_time_boundary_signals(grad_output, phi_p,
                                                      need_input_signal=embed)

            # 4. Trainable input embedding (3-factor direct rule); uses M_{t-1} via
            #    ell_h[0], so it MUST precede the M updates.
            if embed:
                ga_seq[t] = ell_h[0] * self.act_fn_p(embed_pre)
                u_seq[t] = u_t

            # 5. Per-layer local gradient + trace advance.
            if use_compiled:
                # Fused core computes grads AND advances this layer's M/A/Q.
                for n, mp in enumerate(layers):
                    eta_n, lam_n = eta_lam[n]
                    grad_W_t, grad_b_t = mp.compiled_step_and_update(
                        mode, h[n], h[n + 1], phi_p[n], ell_h[n + 1], eta_n, lam_n, um)
                    grad_W[n] += grad_W_t
                    grad_b[n] += grad_b_t
            else:
                for n, mp in enumerate(layers):
                    eta_n, lam_n = eta_lam[n]
                    grad_W_t, grad_b_t = step_fns[n](
                        h[n], phi_p[n], ell_h[n + 1], eta_n, lam_n, um)
                    grad_W[n] += grad_W_t
                    grad_b[n] += grad_b_t
                # 6. Only now advance every layer's M from M_{t-1} to M_t.
                for n, mp in enumerate(layers):
                    eta_n, lam_n = eta_lam[n]
                    mp.update_M_matrix_local_fast(
                        h[n], h[n + 1], eta=eta_n, lam=lam_n, update_mask=um)

            go_seq[t] = grad_output
            hid_seq[t] = h[-1]

        grad_Wout = torch.einsum('TBa,TBi->ai', go_seq, hid_seq)
        grad_bout = go_seq.sum(dim=(0, 1))

        if default_loss:
            loss = masked_mse_loss_only(outputs, labels, masks) if outputs is not None else (loss_sum / N)
        else:
            loss = prepass_loss   # from the pre-pass, same outputs → identical value

        # Assemble grads under the same keys as _trainable_params: layer 0 → 'W'/'b',
        # layer n≥1 → 'W{n}'/'b{n}'; readout 'W_output'/'b_output'; embedding 'W_in'/'b_in'.
        all_grads = {'W_output': grad_Wout, 'b_output': grad_bout}
        for n in range(L):
            suffix = '' if n == 0 else str(n)
            all_grads[f'W{suffix}'] = grad_W[n]
            all_grads[f'b{suffix}'] = grad_b[n]
        if embed:
            all_grads['W_in'] = torch.einsum('TBo,TBI->oI', ga_seq, u_seq)
            if has_bin:
                all_grads['b_in'] = ga_seq.sum(dim=(0, 1))

        params = self._trainable_params()
        result = {k: all_grads[k] for k in params}
        result['loss'] = loss.detach()
        result['outputs'] = outputs.detach() if return_outputs and outputs is not None else None
        return result

    def local_gradients(self, inputs, labels, masks, **kwargs):
        """Exact intra-layer row-local eligibility per MP layer + same-time
        inter-layer learning signals + direct 3-factor input embedding. ONLY with
        feedback_mode='exact_spatial' is the TOP plastic layer's gradient exact vs
        BPTT (exact eligibility + true spatial feedback); the LOWER plastic layers
        stay surrogates (they omit temporal paths through upper plastic layers), and
        under the FA feedback modes even the top layer is no longer exact."""
        return self._local_sequence_gradients(inputs, labels, masks, 'exact', **kwargs)

    def local_diag_rflo_gradients(self, inputs, labels, masks, **kwargs):
        """Diagonal RFLO eligibility per MP layer + same-time inter-layer learning
        signals + direct 3-factor input embedding. APPROXIMATE for EVERY plastic
        layer — including the top — regardless of feedback_mode, because the
        diagonal eligibility itself drops off-synapse plastic sensitivities (exact
        only when n_input == 1). The readout gradient stays exact."""
        return self._local_sequence_gradients(inputs, labels, masks, 'diag', **kwargs)

    def local_direct_gradients(self, inputs, labels, masks, **kwargs):
        """Direct/instantaneous eligibility per MP layer + same-time inter-layer
        learning signals + direct 3-factor input embedding. Treats every M_{t-1} as
        a stop-gradient state (spatial backprop through the deep feedforward net with
        frozen modulation). APPROXIMATE for EVERY plastic layer — including the top —
        regardless of feedback_mode (no plasticity-mediated temporal credit at all);
        exact only at T=1 or eta=0. The readout gradient stays exact."""
        return self._local_sequence_gradients(inputs, labels, masks, 'direct', **kwargs)

    def _grads_for_rule(self, rule, inputs, labels, masks, **kwargs):
        """Full gradient dict for an ARBITRARY rule (not necessarily self.learning_
        rule). Used both by sequence_gradients and by the input_mode splice, which
        needs the OTHER rule's embedding gradient."""
        if rule == 'bptt':
            return self.bptt_gradients(inputs, labels, masks, **kwargs)
        elif rule == 'local_exact_rowlocal':
            return self.local_gradients(inputs, labels, masks, **kwargs)
        elif rule == 'local_diag_rflo':
            return self.local_diag_rflo_gradients(inputs, labels, masks, **kwargs)
        elif rule == 'local_direct':
            return self.local_direct_gradients(inputs, labels, masks, **kwargs)
        raise ValueError(f"unknown learning_rule '{rule}'")

    def _apply_input_mode(self, grads, inputs, labels, masks, **kwargs):
        """Override W_in/b_in in `grads` when input_mode requests a rule the native
        pass did not already produce for the embedding.

        The embedding's NATIVE rule is 'exact' under bptt and 'three_factor' under
        any local rule (all local rules give the SAME embedding gradient — the
        3-factor rule does not depend on the MP eligibility mode). So a splice is
        needed only when input_mode disagrees with that native rule:
          input_mode='exact'        on a LOCAL run → take W_in/b_in from a BPTT pass.
          input_mode='three_factor' on a BPTT run  → take W_in/b_in from a local pass
                                                      (local_direct is the cheapest).
        'match', a non-trainable embedding, or an already-matching native rule → no-op.
        The extra pass recomputes the full gradient but only W_in/b_in are kept."""
        if self.input_mode == 'match' or not self._has_trainable_embed():
            return grads
        native_is_exact = (self.learning_rule == 'bptt')
        want_exact = (self.input_mode == 'exact')
        if want_exact == native_is_exact:
            return grads                      # native pass already produced it
        other_rule = 'bptt' if want_exact else 'local_direct'
        other = self._grads_for_rule(other_rule, inputs, labels, masks, **kwargs)
        for k in ('W_in', 'b_in'):
            if k in grads and k in other:
                grads[k] = other[k]
        return grads

    def sequence_gradients(self, inputs, labels, masks, **kwargs):
        """Dispatch on self.learning_rule; write grads into each param's .grad.
        If input_mode != 'match', the trainable input embedding's W_in/b_in grad is
        then overridden to the input_mode's rule (see _apply_input_mode)."""
        grads = self._grads_for_rule(self.learning_rule, inputs, labels, masks, **kwargs)
        grads = self._apply_input_mode(grads, inputs, labels, masks, **kwargs)

        for name, p in self._trainable_params().items():
            p.grad = grads[name].clone()
        return grads
