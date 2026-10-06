"""
Efficiency-optimized MPN implementation (the default; wired in everywhere).

The reference implementation this was derived from is kept as core/mpn_archive.py.
Same networks (MultiPlasticLayer / MultiPlasticNet / DeepMultiPlasticNet) and the
same public API (bptt_gradients / local_* / sequence_gradients), producing results
IDENTICAL to mpn_archive.py up to floating-point round-off (some changes are
bitwise-exact, the speedups reorder ops). Verified by tests/test_mpn_revise.py:
float64 diffs are ~1e-16 (pure round-off → same computation), float32 well within
1e-5, across BPTT and all local rules on both nets for shared configurations.
Optional rflo_trace_rho adds a capped-gain RFLO approximation beyond the archive.

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
    - update_M_matrix_local_fast skips the general activation/update branches
      in the local config (mult, linear m_act, Hebbian updates); optional bounds
      and frozen plastic states share their derivative gates with local traces.
    - Input-embedding backprojection via backproject_through_modulated_weights_fast
      (ℓ_pre@W + bmm(ℓ_pre, W⊙M), no W_eff).
    - eta/lam expanded once per unroll; batched readout/embedding grads; streams
      the scalar loss and supports return_outputs=False to skip storing outputs.
    - Long-unroll overhead cuts: the mode branch is hoisted OUT of the per-step
      loop (step_fn_for(mode) → _local_step_{direct,diag,exact} resolved once);
      inputs/labels/masks are made time-major + contiguous so per-step reads are
      contiguous (B,·) rows; the readout/embedding scratch buffers (go/hid/ga/u)
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

from net_helpers import BaseNetwork, BaseNetworkFunctions
from net_helpers import rand_weight_init, get_activation_function


def resolve_local_bias_mode(local_bias_mode, learning_rule):
    """Resolve MP bias eligibility for a rule without changing the requested policy.

    BPTT always differentiates biases with autograd and is reported as 'autograd'
    (never 'exact', which names the row-local Q TRACE); local_direct always uses
    the instantaneous factor. match selects direct for diagonal RFLO and exact for
    row-local. Explicit exact/direct retain their historical local-rule behavior.
    """
    if local_bias_mode not in ('exact', 'direct', 'match'):
        raise ValueError("local_bias_mode must be 'exact', 'direct', or 'match'")
    if learning_rule == 'bptt':
        return 'autograd'
    if learning_rule == 'local_direct':
        return 'direct'
    if learning_rule not in ('local_diag_rflo', 'local_exact_rowlocal'):
        raise ValueError(f"unknown learning_rule {learning_rule!r}")
    if local_bias_mode == 'match':
        return 'exact' if learning_rule == 'local_exact_rowlocal' else 'direct'
    return local_bias_mode


def resolve_input_mode(input_mode, learning_rule):
    """Resolve the input-gradient algorithm for a rule, before trainability checks."""
    if input_mode == 'paired':
        paired = {'bptt': 'exact', 'local_direct': 'three_factor',
                  'local_diag_rflo': 'diag_mtrace'}
        if learning_rule not in paired:
            raise ValueError(f"input_mode='paired' does not support {learning_rule!r}; "
                             "use match or an explicit input mode")
        return paired[learning_rule]
    if input_mode == 'match':
        return 'exact' if learning_rule == 'bptt' else 'three_factor'
    if input_mode == 'diag_mtrace':
        return 'exact' if learning_rule == 'bptt' else 'diag_mtrace'
    if input_mode in ('exact', 'three_factor'):
        return input_mode
    raise ValueError(f"unknown input_mode {input_mode!r}")


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


# ─── Learning-signal SOURCE (deep net only) ───────────────────────────────────
# Orthogonal to feedback_mode (HOW a signal is transported) and to learning_rule
# (WHICH eligibility consumes it): learning_signal selects WHERE each MP layer's
# learning signal ell_h[n+1] comes from.
#   'global'        the main readout's error, delivered to every layer by the
#                   same-time inter-layer pathway of feedback_mode (the historical
#                   behavior; default, byte-identical to before).
#   'local_readout' every NON-TOP MP layer n < L-1 owns an auxiliary linear head
#                   q^(n) = C_n h[n+1] + c_n trained on the SAME task loss/labels/
#                   mask as the main readout; layer n's eligibility is credited by
#                   its own head's error projected through C_n (ell_h[n+1] = e_n C_n)
#                   and NOTHING is propagated down from the layers above. The top
#                   layer keeps the true readout; the input embedding shares module
#                   0's head through the unchanged layer-0 backprojection. Requires
#                   exact_spatial feedback and cross_layer_steps=0 (both other
#                   mechanisms are inter-module by construction).
#   'mixed'         ell_h[n+1] = ell_global + local_signal_alpha * ell_local, the two
#                   computed INDEPENDENTLY (the head errors never enter the global
#                   recursion). Same constraints as local_readout.
# The heads train only via the local passes; bptt ignores learning_signal (its heads
# receive no gradient), so the BPTT baseline keeps its meaning.
_LEARNING_SIGNALS = ('global', 'local_readout', 'mixed')


def canonical_learning_signal(signal):
    """Validate a learning_signal string against _LEARNING_SIGNALS."""
    if signal not in _LEARNING_SIGNALS:
        raise ValueError(
            f"unknown learning_signal '{signal}'; expected one of {_LEARNING_SIGNALS}")
    return signal


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

        # Independent of the weight-eligibility rule. Default preserves legacy
        # exact bias traces; 'direct' uses R=phi' and allocates no Q trace.
        # 'match' is resolved by the actual trace initializer on EVERY pass, not
        # the net's current learning_rule (public gradient methods may override it).
        self.local_bias_mode = ml_params.get('local_bias_mode', 'exact')
        resolve_local_bias_mode(self.local_bias_mode, 'local_exact_rowlocal')
        # Optional stabilization of MP-weight diagonal traces, not forward M,
        # bias traces, input sensitivities, or the other gradient algorithms.
        self.rflo_trace_rho = ml_params.get('rflo_trace_rho')
        if self.rflo_trace_rho is not None:
            self.rflo_trace_rho = float(self.rflo_trace_rho)
            if not math.isfinite(self.rflo_trace_rho) or not 0 < self.rflo_trace_rho < 1:
                raise ValueError("rflo_trace_rho must be finite and strictly between 0 and 1")

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
        # m_scale is a fixed hyperparameter, saved in net_params like m_activation.
        self.m_scale = float(ml_params.get('m_scale', 1.0))
        if self.m_act == 'scaled_tanh' and (not math.isfinite(self.m_scale) or self.m_scale <= 0):
            raise ValueError("scaled_tanh requires a finite positive m_scale")
        base_activation = 'linear' if self.m_act == 'scaled_tanh' else self.m_act
        self.m_act_fn, self.m_act_fn_np, self.m_act_fn_p = get_activation_function(base_activation)

        # Initial modulation values
        self.register_buffer('M_init', torch.zeros((self.n_output, self.n_input,), dtype=torch.float))

        # Controls maximum and minimum values of modulations so weights don't change signs
        self.modulation_bounds = ml_params.get('modulation_bounds', self.m_act != 'scaled_tanh')
        if self.m_act == 'scaled_tanh' and self.modulation_bounds:
            raise ValueError("scaled_tanh supplies its own bounds; set modulation_bounds=False")
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
    # derivation config: mp_type='mult', linear or scaled-tanh writes.
    # Linear writes optionally have hard bounds. Indexing: i = post, I = param-pre,
    # W_{iI}, J = plastic-pre index. See run_sequence_local_mpn_exact.

    def assert_local_config(self):
        """Guard for the local (eligibility-trace) rules. They are derived for a
        multiplicative modulation with linear/clipped or scaled-tanh writes, and
        for one of two Hebbian M-updates:

            hebb_assoc: M_{iI,t} = lam M_{iI,t-1} + eta h_{i,t} x_{I,t}
            hebb_pre:   M_{iI,t} = lam M_{iI,t-1} + eta c        x_{I,t}   (c const)

        For hebb_assoc M depends on the postsynaptic activity h (hence on W, b),
        so the plastic-sensitivity traces P/A/Q are live. For hebb_pre M depends
        only on the input, so dM/dW = dM/db = 0 identically: every plastic trace
        is zero and the local rule is EXACT (see self._assoc). Other updates
        (oja) / other nonlinear m_act are refused here. The shared write
        finalizer applies the appropriate derivative and mask to P/A/Q."""
        if self.mp_type != 'mult':
            raise NotImplementedError(
                f"local rules derived for mp_type='mult', got '{self.mp_type}'")
        if self.m_update_type not in ('hebb_assoc', 'hebb_pre'):
            raise NotImplementedError(
                f"local rules derived for m_update_type in (hebb_assoc, hebb_pre), "
                f"got '{self.m_update_type}'")
        if self.m_act not in ('linear', 'scaled_tanh'):
            raise NotImplementedError(
                f"local rules require linear or scaled_tanh modulation, got '{self.m_act}'")

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
        Q is None when local_bias_mode='direct'; 'match' selects exact here.
        """
        dev, dt = self.W.device, self.W.dtype
        self.P = torch.zeros(B, self.n_output, self.n_input, self.n_input, device=dev, dtype=dt)
        self._local_trace_name = 'P'
        self._local_traces_pending = False
        self._smooth_trace_previous = {}
        self.Q = (torch.zeros(B, self.n_output, self.n_input, device=dev, dtype=dt)
                  if resolve_local_bias_mode(self.local_bias_mode, 'local_exact_rowlocal') == 'exact'
                  else None)
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
        R = self._local_bias_eligibility(x, phi_prime)

        self.E, self.R = E, R
        return E, R

    def _local_bias_eligibility(self, x, phi_prime):
        """Exact row-local bias sensitivity or direct neuron-level factor."""
        # Q allocation resolves the requested policy for the actual pass; never
        # overwrite local_bias_mode, which must survive rule switches/reloading.
        if self.Q is None:
            return phi_prime
        recurrent = torch.einsum('iJ,BJ,BiJ->Bi', self.W, x, self.Q)
        return phi_prime * (1.0 + recurrent)

    def _advance_local_bias_trace(self, x, R, eta, lam, update_mask):
        self._local_traces_pending = True
        if self.Q is None:
            return
        Q_new = (lam[None] * self.Q
                 + self._assoc * eta[None] * R.unsqueeze(-1) * x.unsqueeze(1))
        self.Q = self._prepare_local_trace('Q', Q_new, self.Q, update_mask)

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

        self.P = self._prepare_local_trace('P', P_new, self.P, update_mask)
        self._advance_local_bias_trace(x, R, eta, lam, update_mask)

    # ─── Diagonal / same-synapse RFLO approximation ──────────────────────────
    # Replaces the exact fourth-order trace P^I_{iJ} (B,post,pre,pre) with a
    # single same-synapse trace A_{iI} ~= P^I_{iI} (B,post,pre), dropping all
    # off-synapse (J != I) plastic sensitivities. The bias trace Q (B,post,pre)
    # is exact by default. local_bias_mode='direct' instead uses R=phi' with no Q.
    # Intentionally an approximation to BPTT; omitted terms can matter more
    # with longer sequences or stronger plasticity.
    # With rflo_trace_rho=None, n_input == 1 has no off-diagonal terms to drop.
    # Capping the recurrence introduces an additional approximation even then.

    def reset_diag_rflo_state(self, B=1):
        """Allocate the diagonal-RFLO traces (zeroed).

        A[b, i, I] ~= dM[b, i, I] / dW[i, I]   shape (B, n_output, n_input)
        With rflo_trace_rho set, A is a stabilized surrogate sensitivity.
        Q[b, i, J]  = dM[b, i, J] / db[i]       shape (B, n_output, n_input)
        Q is None when local_bias_mode='direct' or 'match'.
        """
        dev, dt = self.W.device, self.W.dtype
        self.A = torch.zeros(B, self.n_output, self.n_input, device=dev, dtype=dt)
        self._diag_phi_prime = None
        self._local_trace_name = 'A'
        self._local_traces_pending = False
        self._smooth_trace_previous = {}
        self.Q = (torch.zeros(B, self.n_output, self.n_input, device=dev, dtype=dt)
                  if resolve_local_bias_mode(self.local_bias_mode, 'local_diag_rflo') == 'exact'
                  else None)
        self.E = None  # last computed (approx) dh/dW  (B, n_output, n_input)
        self.R = None  # last computed dh/db  (B, n_output)

    def compute_diag_rflo_eligibility(self, x, phi_prime):
        """Diagonal approximation to dh_t/dW: drop the sum over J != I, keeping
        only the same-synapse term W_{iI} x_I A_{iI}.

            E_hat^I_{i,t} = phi'_i * x_I * (1 + M_{iI,t-1} + W_{iI} A_{iI,t-1}).

        The bias eligibility is exact by default; local_bias_mode='direct'
        uses R=phi_prime and no Q trace. Call
        AFTER the forward pass and BEFORE update_diag_rflo_traces / update_M,
        so self.M, self.A, self.Q still hold time-(t-1) values.
        returns E_hat (B, n_output, n_input), R (B, n_output).
        """
        M_prev = self.M   # (B, i, I) = M_{t-1}
        A_prev = self.A   # (B, i, I)
        # The explicit trace update needs phi' itself: recovering it by dividing
        # E by x*(1+M+W*A) would be undefined at zero inputs/factors.
        if self.rflo_trace_rho is not None:
            self._diag_phi_prime = phi_prime

        E = (phi_prime.unsqueeze(-1) * x.unsqueeze(1)
             * (1.0 + M_prev + self.W.unsqueeze(0) * A_prev))          # (B, i, I)

        # Exact row-local bias trace (same as the exact rule).
        R = self._local_bias_eligibility(x, phi_prime)

        self.E, self.R = E, R
        return E, R

    def update_diag_rflo_traces(self, x, E, R, update_mask=None, eta_lam=None):
        """Advance the diagonal-RFLO traces one step (uses M_t's eta/lam):

        Without a gain cap:
        A_{iI,t} = lam_{iI} A_{iI,t-1} + eta_{iI} x_I E_hat^I_{i,t}
        Q_{iJ,t} = lam_{iJ} Q_{iJ,t-1} + eta_{iJ} x_J R_{i,t}          (exact)
        With rflo_trace_rho set, cap the A recurrence via _capped_diag_trace.
        The write derivative and frozen-state gate are applied after the M write.

        Call AFTER compute_diag_rflo_eligibility, in step with update_M_matrix.
        update_mask (B,) freezes traces for inactive batch rows.
        eta_lam: optional precomputed (eta, lam) — constant over the unroll, hoisted
        out of the time loop (identical values, avoids re-expanding every step)."""
        eta, lam = eta_lam if eta_lam is not None else self._eta_lam_full()  # each (i, I)
        a = self._assoc  # 0 for hebb_pre → dM/dW = dM/db = 0, traces stay zero

        if self.rflo_trace_rho is None:
            A_new = lam[None] * self.A + a * eta[None] * x.unsqueeze(1) * E  # x_I E_hat^I_i
        else:
            if self._diag_phi_prime is None:
                raise RuntimeError("Compute diagonal eligibility before advancing capped traces")
            A_new = self._capped_diag_trace(x, self._diag_phi_prime, eta, lam)
            self._diag_phi_prime = None

        self.A = self._prepare_local_trace('A', A_new, self.A, update_mask)
        self._advance_local_bias_trace(x, R, eta, lam, update_mask)

    def _capped_diag_trace(self, x, phi_prime, eta, lam):
        """Candidate A update with bounded recurrence gain, before masks/gates.

        k = assoc*eta*phi'*x^2; gain = clip(lam + k*W, -rho, rho).
        Only the coefficient of old A is capped. The drive k*(1+M) is retained.
        This is a surrogate eligibility, not the exact modulation derivative.
        """
        k = (self._assoc * eta.unsqueeze(0) * phi_prime.unsqueeze(-1)
             * x.square().unsqueeze(1))
        gain = lam.unsqueeze(0) + k * self.W.unsqueeze(0)
        gain = gain.clamp(-self.rflo_trace_rho, self.rflo_trace_rho)
        return gain * self.A + k * (1.0 + self.M)

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
        """Advance M and finalize the already-advanced local sensitivity traces.

        Algebraically identical to update_M_matrix for the configurations allowed
        by assert_local_config(): multiplicative MP with linear/hard-clipped or
        scaled-tanh writes and Hebbian associative/pre-only updates. The shared
        finalizer handles activation derivatives, masks and frozen states.
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
        self._finish_modulation_update(M_pre, M_prev, update_mask)
        return self.M_pre - M_prev

    def _finish_modulation_update(self, raw, previous, update_mask):
        """Shared BPTT/local write map. Smooth masks mix AFTER activation.

        smooth: M_new = r * B*tanh(raw/B) + (1-r)*M_old.
        hard:   M_new = clip(r*raw + (1-r)*M_old), preserving legacy behavior.
        M_pre is the argument of the activation in the selected convention.
        """
        if self.m_act == 'scaled_tanh':
            self.M_pre = raw
            candidate = self.m_scale * torch.tanh(raw / self.m_scale)
            self.M = self._apply_update_mask_3d(candidate, previous, update_mask)
        else:
            self.M_pre = self._apply_update_mask_3d(raw, previous, update_mask)
            self.M = self.m_act_fn(self.M_pre)
            if self.modulation_bounds:
                self.M = self._apply_modulation_bounds(self.M)
        self._finalize_local_traces()
        frozen = getattr(self, '_plasticity_freeze_mask', None)
        if frozen is not None:
            self.M[:, frozen[0], frozen[1]] = self._M_frozen_vals

    def _prepare_local_trace(self, name, new, old, update_mask):
        """Defer smooth masking until its nonlinear write derivative is known."""
        if self.m_act == 'scaled_tanh':
            if update_mask is not None:
                self._smooth_trace_previous[name] = (old, update_mask)
            return new
        mask_fn = self._apply_update_mask_4d if new.ndim == 4 else self._apply_update_mask_3d
        return mask_fn(new, old, update_mask)

    def _apply_modulation_bounds(self, value):
        """Hard clipping with d(output)/d(value)=1 at either endpoint.

        Strict comparisons choose the unchanged value at equality, so autograd
        and the explicit local gate use the same convention, independently of
        torch.clamp's backend/version-specific endpoint behavior. Bounds are
        fixed buffers; derivatives through the bound values are not required.
        """
        lower, upper = self.M_bounds[1], self.M_bounds[0]
        return torch.where(value < lower, lower,
                           torch.where(value > upper, upper, value))

    def _modulation_clamp_derivative(self):
        """Inclusive endpoint derivative of _apply_modulation_bounds."""
        return (self.M_pre >= self.M_bounds[1]) & (self.M_pre <= self.M_bounds[0])

    def _modulation_write_derivative(self):
        """Derivative of the write nonlinearity, including frozen-state zeros.

        None denotes an identity gate. A frozen plastic state is restored to
        its parameter-independent initial value, so all of its sensitivities
        vanish. For P, this gate acts on the state index J, not parameter index I.
        Does not include the update mask: trace finalization and cross-layer
        sources apply that mask separately with the correct ordering.
        """
        if self.m_act == 'scaled_tanh':
            gate = 1.0 - torch.tanh(self.M_pre / self.m_scale).square()
        else:
            gate = self._modulation_clamp_derivative() if self.modulation_bounds else None
        frozen = getattr(self, '_plasticity_freeze_mask', None)
        if frozen is not None:
            if gate is None:
                gate = torch.ones_like(self.M_pre, dtype=torch.bool)
            gate[:, frozen[0], frozen[1]] = False
        return gate

    def _finalize_local_traces(self):
        """Apply the nonlinear derivative/mask once to newly advanced P/A/Q."""
        if not getattr(self, '_local_traces_pending', False):
            return
        write_gate = self._modulation_write_derivative()
        if self.m_act == 'scaled_tanh':
            names = [self._local_trace_name] + (['Q'] if self.Q is not None else [])
            frozen = getattr(self, '_plasticity_freeze_mask', None)
            for name in names:
                gate = write_gate.unsqueeze(-2) if name == 'P' else write_gate
                advanced = getattr(self, name) * gate
                if name in self._smooth_trace_previous:
                    old, mask = self._smooth_trace_previous[name]
                    mask_fn = self._apply_update_mask_4d if name == 'P' else self._apply_update_mask_3d
                    advanced = mask_fn(advanced, old, mask)
                if frozen is not None:
                    if name == 'P':
                        advanced[:, frozen[0], :, frozen[1]] = 0
                    else:
                        advanced[:, frozen[0], frozen[1]] = 0
                setattr(self, name, advanced)
            self._smooth_trace_previous.clear()
            self._local_traces_pending = False
            return
        if write_gate is not None:
            trace = getattr(self, self._local_trace_name)
            gate = write_gate.unsqueeze(-2) if self._local_trace_name == 'P' else write_gate
            setattr(self, self._local_trace_name, trace * gate)
            if self.Q is not None:
                self.Q = self.Q * write_gate
        self._local_traces_pending = False

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
        (E_hat never built); optional A-gain cap and configurable exact/direct bias."""
        W, M_prev = self.W, self.M
        a = self._assoc
        A_prev = self.A
        factor = 1.0 + M_prev + W.unsqueeze(0) * A_prev
        ell_phi = ell * phi_prime

        grad_W_t = (ell_phi.unsqueeze(-1) * x.unsqueeze(1) * factor).sum(0)

        R = self._local_bias_eligibility(x, phi_prime)
        grad_b_t = torch.einsum('Bi,Bi->i', ell, R)

        if self.rflo_trace_rho is None:
            A_new = (lam.unsqueeze(0) * A_prev
                     + a * eta.unsqueeze(0) * phi_prime.unsqueeze(-1)
                     * x.square().unsqueeze(1) * factor)
        else:
            A_new = self._capped_diag_trace(x, phi_prime, eta, lam)
        self.A = self._prepare_local_trace('A', A_new, A_prev, update_mask)
        self._advance_local_bias_trace(x, R, eta, lam, update_mask)
        self.E, self.R = None, R
        return grad_W_t, grad_b_t

    def _local_step_exact(self, x, phi_prime, ell, eta, lam, update_mask=None):
        """Exact row-local: E is still built (the full P trace update needs it),
        but eta/lam are reused and the P/Q update is done inline."""
        a = self._assoc
        E, R = self.compute_exact_rowlocal_eligibility(x, phi_prime)
        grad_W_t = torch.einsum('Bi,BiI->iI', ell, E)
        grad_b_t = torch.einsum('Bi,Bi->i', ell, R)

        P_prev = self.P
        outerP = torch.einsum('BiI,BJ->BiIJ', E, x)
        P_new = lam[None, :, None, :] * P_prev + a * eta[None, :, None, :] * outerP
        self.P = self._prepare_local_trace('P', P_new, P_prev, update_mask)
        self._advance_local_bias_trace(x, R, eta, lam, update_mask)
        return grad_W_t, grad_b_t

    def step_fn_for(self, mode):
        """Return the mode's fused per-step function, resolved ONCE before a loop
        (so the per-timestep call has no `mode` branch)."""
        return {'direct': self._local_step_direct,
                'diag': self._local_step_diag,
                'exact': self._local_step_exact}[mode]

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

        self._finish_modulation_update(M_pre, M, update_mask)

        # Difference before the selected nonlinearity; not the bounded M delta.
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

    @property
    def resolved_local_bias_modes(self):
        """Effective bias rule per MP layer, recomputed after cloning/rule changes:
        'direct' (phi' only), 'exact' (row-local Q trace) or 'autograd' (bptt)."""
        return [resolve_local_bias_mode(layer.local_bias_mode, self.learning_rule)
                for layer in self.mp_layers]

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

        # ── Fixed input standardization (opt-in) ─────────────────────────────
        # The raw input u_t feeds straight into the modulated forward W(1+M)x AND
        # into the Hebbian M update (η·h·xᵀ), so its scale strongly conditions the
        # modulation dynamics. When enabled, u_t is affinely standardized by FIXED
        # per-feature statistics — u_norm = (u - loc) / scale — applied IDENTICALLY
        # in every forward/gradient/eval path (so it conditions all learning rules
        # equally and just removes a scale artifact; it is NOT a learned/adaptive
        # norm). The statistics are frozen buffers (they move with .to()/.double()
        # and ride in state_dict so a reloaded net normalizes identically); estimate
        # them once from a task sample via set_input_norm_stats(). The buffers are
        # allocated ONLY when the flag is on (matching the rest of the codebase), so
        # a default net's buffer set — hence its state_dict — is byte-for-byte
        # unchanged and old checkpoints still load. When off, _standardize_input is a
        # pure identity, so every existing gradient path is unchanged.
        self.input_normalize = net_params.get('input_normalize', False)
        if self.input_normalize:
            self._alloc_input_norm_buffers()

        if verbose: # Full summary of readout parameters (MP layer prints out internally)
            print(init_string)

    def reset_state(self, B=1):
        """ Resets states of all internal layer M matrices """

        for mp_layer in self.mp_layers:
            mp_layer.reset_state(B=B)

    def _alloc_input_norm_buffers(self):
        """Register the fixed input-standardization buffers (identity init: loc=0,
        scale=1). Allocated only when input_normalize is on, so a default net's
        state_dict is unchanged and old checkpoints still load. Idempotent. Matches
        W_output's dtype/device so it is correct whether called at construction or
        later (e.g. after .double()/.to(device))."""
        if 'input_loc' not in self._buffers:
            ref = self.W_output
            self.register_buffer('input_loc', torch.zeros(self.n_input, dtype=ref.dtype, device=ref.device))
            self.register_buffer('input_scale', torch.ones(self.n_input, dtype=ref.dtype, device=ref.device))

    @torch.no_grad()
    def set_input_norm_stats(self, sample_inputs, eps=1e-5):
        """Freeze the fixed input-standardization statistics from a data SAMPLE.

        Estimates per-feature mean/std over all non-feature axes of `sample_inputs`
        (shape (..., n_input); e.g. a (B, T, n_input) batch or a stack of them) and
        stores them in the input_loc / input_scale buffers so _standardize_input maps
        u -> (u - mean) / max(std, eps) thereafter. `eps` floors the scale so a
        constant channel (std 0, e.g. a fixation bit) is centered but not blown up.
        The statistics are FIXED once set (this is not called inside any training
        step) and applied identically to train + validation for every rule. Enables
        normalization (input_normalize=True), allocating the buffers if needed, since
        freezing real statistics is the whole point of computing them. The buffers
        move with .to()/.double() and persist in state_dict, so a reloaded net
        normalizes identically."""
        self.input_normalize = True
        self._alloc_input_norm_buffers()
        x = sample_inputs.reshape(-1, self.n_input).to(self.input_loc)
        self.input_loc.copy_(x.mean(dim=0))
        self.input_scale.copy_(x.std(dim=0).clamp_min(eps))

    def _standardize_input(self, inputs):
        """Apply the FIXED input standardization u -> (u - loc) / scale.

        Identity (returns `inputs` unchanged, no copy) unless input_normalize is on,
        so every default path is byte-for-byte unchanged. Called once at the top of
        each terminal gradient method and the eval forward on the whole (B, T,
        n_input) sequence, so the standardized input then propagates identically into
        the modulated forward W(1+M)x, the Hebbian M update, and (deep net) the input
        embedding's own gradient. Broadcasts over any leading axes."""
        if not getattr(self, 'input_normalize', False):
            return inputs
        return (inputs - self.input_loc) / self.input_scale

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
        #                            in the clean config)
        #   'local_diag_rflo'      — diagonal/same-synapse RFLO approximation
        #   'local_direct'         — direct/instantaneous approximation (no trace;
        #                            stops gradient through the plasticity history)
        _rule = net_params.get('learning_rule', 'bptt')
        assert _rule in ('bptt', 'local_exact_rowlocal', 'local_diag_rflo',
                         'local_direct'), f"unknown learning_rule '{_rule}'"
        self.learning_rule = _rule
        # Local readout heads need a non-top MP layer to attach to; this net has
        # exactly one MP layer and no embedding, so only the global signal exists.
        if canonical_learning_signal(net_params.get('learning_signal', 'global')) != 'global':
            raise ValueError(
                "learning_signal requires DeepMultiPlasticNet (net_type 'dmpn'); "
                "MultiPlasticNet has a single MP layer and no local readout heads.")

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
        inputs = self._standardize_input(inputs)   # fixed norm (identity unless on)
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

        # Fixed input standardization (identity unless on), applied ONCE here so both
        # the custom-loss pre-pass and the accumulation loop below see the same
        # standardized u_t (the pre-pass itself must NOT re-normalize).
        inputs = self._standardize_input(inputs)

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
        # below). Same options as MultiPlasticNet.
        _rule = cfg.get('learning_rule', 'bptt')
        assert _rule in ('bptt', 'local_exact_rowlocal', 'local_diag_rflo',
                         'local_direct'), f"unknown learning_rule '{_rule}'"
        self.learning_rule = _rule
        # input_mode decouples the TRAINABLE INPUT EMBEDDING's learning rule from the
        # MP-layer learning_rule, so any RULES_TO_RUN × input-rule combination can be
        # compared. Only meaningful when an embedding weight or bias is trainable.
        #   'match'        — the embedding follows learning_rule (historical default:
        #                    exact autograd under bptt, the 3-factor local rule under
        #                    a local rule). Zero behavior change from before.
        #   'exact'        — the embedding is ALWAYS trained by the true BPTT gradient
        #                    (dL/dW_in via autograd), even during a local MP run.
        #   'three_factor' — the embedding ALWAYS uses the DIRECT 3-factor local rule
        #                    (ell_h[0] ⊙ phi'(embed_pre)) · uᵀ, even during a bptt run.
        #   'diag_mtrace'  — local runs track the first MP layer's modulation-column
        #                    sensitivity to the corresponding embedding row. BPTT
        #                    stays exact. Requires exact_spatial feedback.
        #   'paired'       — bptt: exact; local_direct: three_factor;
        #                    local_diag_rflo: diag_mtrace. No row-local mapping.
        # exact/three_factor use a gradient splice; diag_mtrace runs forward-mode
        # sensitivities inside the local pass, without an additional BPTT pass.
        self.input_mode = cfg.get('input_mode', 'match')
        resolve_input_mode(self.input_mode, self.learning_rule)
        # feedback_mode governs how each hidden layer's learning signal is formed
        # in the local rules (see the _FEEDBACK_MODES table and
        # _same_time_boundary_signals). B_feedback_init is stashed for the
        # per-layer FA buffers built after the MP layers exist.
        self.feedback_mode = canonical_feedback_mode(cfg.get('feedback_mode', 'exact_spatial'))
        if self.input_mode in ('diag_mtrace', 'paired'):
            if not self.input_layer_active or self.feedback_mode != 'exact_spatial':
                raise ValueError(f"{self.input_mode} requires an input embedding and exact_spatial feedback")
        self._B_feedback_init = cfg.get('B_feedback_init', 'xavier')
        # cross_layer_steps: depth of the cross-layer TEMPORAL correction added to the
        # surrogate lower-layer local gradients (see _cross_layer_correction). 0 (the
        # default) = the pure same-time rule, byte-identical to before. 1 = the exact
        # depth-1 (one-temporal-hop) correction: each layer n is credited through the
        # Hebbian M-writes of EVERY upper layer m>n at the previous step, which makes
        # every layer's gradient exact vs BPTT at T=2 (the dropped series has one term
        # there). Only k=1 is implemented; k>1 (deeper truncated BPTT-through-plasticity)
        # is a future extension and is rejected below. The correction is orthogonal to
        # the weight-eligibility mode. It is separate from the pure DFA algorithms:
        # direct_fa requires zero correction, because this routine uses forward weights.
        # Exact T=2 statements require exact_spatial feedback and exact bias/weight traces.
        self.cross_layer_steps = int(cfg.get('cross_layer_steps', 0))
        if self.feedback_mode == 'direct_fa' and self.cross_layer_steps != 0:
            raise ValueError(
                "direct_fa requires cross_layer_steps=0: the one-step correction "
                "uses upper forward weights and is a separate reference algorithm.")
        if self.cross_layer_steps not in (0, 1):
            raise NotImplementedError(
                f"cross_layer_steps={self.cross_layer_steps}: only 0 (same-time) and 1 "
                f"(exact depth-1 temporal correction) are implemented.")
        # learning_signal: WHERE each MP layer's learning signal comes from (see the
        # module-level _LEARNING_SIGNALS table). 'global' is the default and leaves
        # every path byte-identical. The local modes replace/augment the inter-layer
        # signal with per-layer auxiliary readout heads (built last, below), so the
        # two inter-MODULE mechanisms — random/recursive feedback transport and the
        # cross-layer temporal correction — are rejected with them.
        self.learning_signal = canonical_learning_signal(cfg.get('learning_signal', 'global'))
        self.local_signal_alpha = float(cfg.get('local_signal_alpha', 1.0))
        if self.learning_signal != 'global':
            if self.feedback_mode != 'exact_spatial':
                raise ValueError(
                    f"learning_signal='{self.learning_signal}' requires feedback_mode="
                    "'exact_spatial': the local heads define the only inter-layer signal, "
                    "so layerwise_fa/direct_fa have nothing to transport.")
            if self.cross_layer_steps != 0:
                raise ValueError(
                    f"learning_signal='{self.learning_signal}' requires cross_layer_steps=0: "
                    "the depth-1 correction credits a layer through the layers above it, "
                    "which is inter-module by construction.")
            if not math.isfinite(self.local_signal_alpha):
                raise ValueError("local_signal_alpha must be finite")

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

        # ── Identity skip connections (opt-in) ───────────────────────────────
        # When mp_residual is on, each MP layer whose input/output widths MATCH runs
        # a parameter-free IDENTITY residual on the stream:
        #     a_n     = act_fn(z_n)     # BLOCK activation → drives the M-update + phi'
        #     h_{n+1} = h_n + residual_scale * a_n  # stream → next layer/readout
        # The fixed branch gain multiplies parameter sensitivities and the branch
        # part of spatial Jacobians, never the identity path or the raw Hebbian post.
        # P/A/Q track raw block sensitivities, so their recurrences are unchanged.
        # Raw a_n is retained explicitly (also valid at scale=0). A skip
        # is inserted ONLY where d_in==d_out (an identity map needs equal widths);
        # unequal-width layers are skipped WITH A WARNING (a projection skip would need
        # its own gradient rule — an extension). No state is registered (identity adds
        # no params/buffers), so state_dict is unchanged and old checkpoints load. OFF
        # by default → _residual_at is all-False and every path is byte-for-byte the
        # non-residual net.
        self.mp_residual = cfg.get('mp_residual', False)
        self.residual_scale = float(cfg.get('residual_scale', 1.0))
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0:
            raise ValueError('residual_scale must be finite and nonnegative')
        if not self.mp_residual and self.residual_scale != 1.0:
            raise ValueError('non-default residual_scale requires mp_residual=True')
        self._residual_at = []
        for n, mp in enumerate(self.mp_layers):
            ok = self.mp_residual and (mp.n_input == mp.n_output)
            if self.mp_residual and not ok:
                print(f'  [mp_residual] skip DISABLED at MP layer {n}: width '
                      f'{mp.n_input}->{mp.n_output} (identity residual needs equal widths).')
            self._residual_at.append(ok)
        # Non-residual blocks keep their ordinary, unscaled activation.
        self._branch_scales = tuple(self.residual_scale if active else 1.0
                                    for active in self._residual_at)

        # ── Local readout heads (opt-in; learning_signal != 'global') ────────
        # One auxiliary linear readout per NON-TOP MP layer n = 0..L-2, reading that
        # layer's output stream h[n+1] and trained on the SAME task loss/labels/mask
        # as the main readout (bias present iff the main readout's is). The head's
        # output error projected through its weights forms layer n's learning
        # signal in the local passes (see _same_time_boundary_signals); the top
        # layer keeps the true readout W_output and the embedding shares module 0's
        # head via the unchanged layer-0 backprojection. Heads take part in
        # TRAINING ONLY — forward()/network_step() and every evaluation path are
        # untouched — and are keyed separately from _trainable_params (see
        # _aux_params) so bptt_gradients, the alignment diagnostic and the weight-
        # decay grouping never see them.
        #
        # Construction is deliberately LAST and draws from a PRIVATE numpy stream
        # (seeded from torch.initial_seed(), i.e. run_seed's per-seed manual_seed,
        # or from local_head_seed when given) with the global numpy state saved and
        # restored around rand_weight_init. The main parameters above and the
        # training-data stream drawn after construction are therefore byte-identical
        # to a learning_signal='global' net built under the same seed, making the
        # two modes a PAIRED comparison. A single MP layer has no head at all, so
        # local_readout then coincides with global.
        self._head_names = []          # [(weight_name, bias_name_or_None)] per head
        if self.learning_signal != 'global':
            head_seed = cfg.get('local_head_seed', None)
            if head_seed is None:
                head_seed = torch.initial_seed() + 0x5EED
            np_state = np.random.get_state()
            np.random.seed(int(head_seed) % (2 ** 32))
            try:
                for n in range(L - 1):
                    d_n = self.mp_layers[n].n_output          # width of h[n+1]
                    w_name = f'head_W{n}'
                    b_name = f'head_b{n}' if self.b_output_active else None
                    self.register_parameter(w_name, nn.Parameter(torch.tensor(
                        rand_weight_init(d_n, self.n_output, init_type=self.W_output_init),
                        dtype=self.W_output.dtype)))
                    if b_name is not None:
                        self.register_parameter(b_name, nn.Parameter(torch.tensor(
                            rand_weight_init(self.n_output, init_type=self.b_output_init),
                            dtype=self.W_output.dtype)))
                    self._head_names.append((w_name, b_name))
            finally:
                np.random.set_state(np_state)


    def _combine_mp_branch(self, n, inputs, activation):
        """Keep the identity path at unit gain; scale only enabled MP branches."""
        if self._residual_at[n]:
            return inputs + self._branch_scales[n] * activation
        return activation

    def forward(self, inputs, run_mode='minimal', verbose=False):
        """Public forward API; raw block activities are internal to the M write."""
        output, activities, db, _ = self._forward_stack(inputs, run_mode, verbose)
        return output, activities, db

    def _forward_stack(self, inputs, run_mode='minimal', verbose=False):
        """Forward with raw block activities, retaining their autograd graph."""
        # 2025-11-19: the x_t in the MPN paper is the "input to MPN layer"
        # namely after the MLP 
        if self.input_layer_active:
            x = self.W_initial_linear(inputs)
            x = self.act_fn(x)
        else:
            x = inputs

        layer_input = x  # read-only downstream (never mutated in-place) → no clone

        mpl_activities = [x,] # Used for updating the M matrices
        block_activities = []

        db = {} if run_mode in ('track_states',) else None

        for mpl_idx, mp_layer in enumerate(self.mp_layers):
            # The pre-layer activity is the residual stream h[n] feeding this layer;
            # also used by the track_states log below.
            layer_input_old = layer_input
            hidden_pre, db_mp = mp_layer(layer_input, run_mode=run_mode)

            # Keep raw a_n for the Hebbian write; mpl_activities stores the stream.
            a_n = self.act_fn(hidden_pre)
            block_activities.append(a_n)
            layer_input = self._combine_mp_branch(mpl_idx, layer_input_old, a_n)

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
        
        return output, mpl_activities, db, block_activities

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
        output, mpl_activities, db, block_activities = self._forward_stack(
            current_input, run_mode=run_mode, verbose=verbose)

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

            # The write uses raw a_n, not the scaled residual increment. Retaining
            # it explicitly avoids cancellation and division by a possibly zero gain.
            pre = mpl_activities[mpl_idx]
            post = block_activities[mpl_idx]
            _ = mp_layer.update_M_matrix(pre, post)

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
    # dropped), and by default train the input embedding with a DIRECT 3-factor rule
    # (backproject the boundary signal through layer 0's modulated weights, then
    # multiply by the embedding activation derivative and the raw input). The
    # diag_mtrace input mode adds first-MP-layer temporal sensitivities below.

    def _embed_grad_flags(self):
        """(need_W_in, need_b_in): whether the input embedding's weight / bias each
        need a local gradient this pass. Decided PER TENSOR on requires_grad — the
        weight and bias can be frozen independently (e.g. input_layer_add_trainable=
        False freezes only the weight) — so it matches _trainable_params's per-tensor
        keys and never emits/drops a key the params dict disagrees with."""
        if not self.input_layer_active:
            return (False, False)
        need_W_in = self.W_initial_linear.weight.requires_grad
        need_b_in = (self.W_initial_linear.bias is not None
                     and self.W_initial_linear.bias.requires_grad)
        return (need_W_in, need_b_in)

    def _has_trainable_embed(self):
        """True if the input embedding contributes ANY trainable tensor (weight OR
        bias), i.e. the local pass must run its selected embedding rule. The two
        tensors are gated independently in _embed_grad_flags / _trainable_params; a
        frozen weight with a trainable bias (or vice-versa) still counts."""
        return any(self._embed_grad_flags())

    def _input_mtrace_step(self, trace, features, embed_prime, x, post,
                           phi_prime, signal, eta, lam):
        """Advance a diagonal-column input sensitivity, before the M write.

        trace[b,i,j,k] approximates dM0[b,i,j]/dU[j,k]; derivatives of
        M0[b,i,J!=j] w.r.t. U[j,k] and all deeper M states are omitted.
        features contains raw (standardized) inputs for trainable weights and
        optionally a constant 1 for the input bias. No parameter gradients are
        propagated through optimizer steps: weights are fixed during the unroll.

        D[j,k] = phi_embed'[j] * features[k]
        C[i,j,k] = phi0'[i] W0[i,j] ((1+M0[i,j]) D[j,k] + x[j] trace[i,j,k])
        raw_trace = lam*trace + eta*(post[i]*D[j,k] + x[j]*C[i,j,k])

        C is the raw first MP block activation's sensitivity. Its branch scale
        multiplies C only in the gradient; the identity adds D there. Neither
        scaling nor the identity path enters the Hebbian post sensitivity C.
        For hebb_pre, replace post by its constant and omit the x*C term.
        The caller finalizes raw_trace AFTER M0's write using its actual gate.
        Storage is O(batch * first_MP_width * embed_width * feature_count).
        """
        layer = self.mp_layers[0]
        direct = embed_prime.unsqueeze(-1) * features.unsqueeze(1)
        eligibility = (phi_prime[:, :, None, None] * layer.W[None, :, :, None]
                       * ((1.0 + layer.M).unsqueeze(-1) * direct.unsqueeze(1)
                          + x[:, None, :, None] * trace))
        gradient = torch.einsum('bi,bijk->jk', self._branch_scales[0] * signal, eligibility)
        if self._residual_at[0]:
            gradient = gradient + torch.einsum('bj,bjk->jk', signal, direct)
        raw_trace = lam[None, :, :, None] * trace
        if layer.m_update_type == 'hebb_assoc':
            source = (post[:, :, None, None] * direct.unsqueeze(1)
                      + x[:, None, :, None] * eligibility)
        else:  # hebb_pre still depends on the embedding, unlike its MP W trace.
            source = direct.unsqueeze(1) / math.sqrt(layer.n_output)
        raw_trace = raw_trace + eta[None, :, :, None] * source
        return gradient, raw_trace

    def _finish_input_mtrace(self, proposed, previous, update_mask):
        """Apply M0's write derivative, mask convention, and frozen-state zeros."""
        layer = self.mp_layers[0]
        gate = layer._modulation_write_derivative()
        if layer.m_act == 'scaled_tanh':
            proposed = proposed * gate.unsqueeze(-1)
            updated = layer._apply_update_mask_4d(proposed, previous, update_mask)
        else:
            updated = layer._apply_update_mask_4d(proposed, previous, update_mask)
            if gate is not None:
                updated = updated * gate.unsqueeze(-1)
        frozen = getattr(layer, '_plasticity_freeze_mask', None)
        if frozen is not None:
            updated[:, frozen[0], frozen[1], :] = 0
        return updated

    # ── Deep-local forward + same-time boundary-signal helpers ────────────────
    # These implement the two ingredients the multi-MP-layer local rules need
    # (see the class docstring): a forward that stops short of the M update, and
    # a top-down learning-signal pass that backprojects the readout error through
    # every layer's MODULATED weights W_eff = W ⊙ (1 + M_{t-1}). Both read the
    # frozen M_{t-1}, so they must run before any layer advances its M.

    def _forward_local_stack(self, u_t, *, return_blocks=False):
        """Forward one time step through the whole stack WITHOUT updating any M.

        Returns (output, h, z, phi_p, embed_pre) with the zero-based indexing the
        deep-local loop uses:
            h[0]      input representation (= embedding output, or raw u_t)
            z[n]      preactivation of MP layer n           (B, d_{n+1})
            phi_p[n]  phi'(z[n])                            (B, d_{n+1})
            h[n+1]    output of MP layer n                  (B, d_{n+1})
            output    readout(h[L])
        embed_pre is the embedding PRE-activation (for the embedding grad) or None.
        With return_blocks=True, append the raw block activations for M writes.
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
        blocks = []
        for n, mp in enumerate(self.mp_layers):
            z_n, _ = mp(h[-1])              # uses this layer's M_{t-1}
            phi_p.append(self.act_fn_p(z_n))
            z.append(z_n)
            # phi_p and blocks refer to raw a_n; only the output stream is scaled.
            # Own-layer eligibility recurrences continue to describe raw a_n.
            a_n = self.act_fn(z_n)
            blocks.append(a_n)
            h.append(self._combine_mp_branch(n, h[-1], a_n))

        output = F.linear(h[-1], self.W_output, self.b_output)
        result = (output, h, z, phi_p, embed_pre)
        return (*result, blocks) if return_blocks else result

    def _same_time_boundary_signals(self, grad_output, phi_p, need_input_signal,
                                    head_errors=None):
        """Same-time layer-boundary learning signals using the frozen M_{t-1}.

        head_errors (local readout heads; see _LEARNING_SIGNALS): the per-head
        output errors e_n = dL_n/dq^(n) for n = 0..L-2, or None (global signal).
          local_readout — ell_h[n+1] = e_n @ C_n for every non-top layer; the
            inter-layer recursion is SKIPPED (nothing descends from upper layers);
            ell_h[L] stays grad_output @ W_output.
          mixed — the global recursion runs unchanged on the MAIN readout error,
            then alpha * e_n @ C_n is ADDED to ell_h[n+1]; the head errors never
            enter the recursion, so no auxiliary loss reaches another module.
        In both modes ell_h[0] (the embedding) is still derived from the FINAL
        ell_h[1] through layer 0's backprojection (+ identity skip), i.e. the
        embedding belongs to module 0 and follows module 0's signal.

        ell_h[n] is the surrogate dL/dh[n] (the signal at the INPUT boundary of MP
        layer n). The equations below show non-residual blocks; enabled skips
        scale the branch term and add the identity term as described below.
        The feedback pathway is selected by self.feedback_mode (see _FEEDBACK_MODES):

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
        Layer n's eligibility routine is fed branch_scale[n] * ell_h[n+1]
        (the signal to raw a_n), never premultiplied by phi'. The P/A/Q recurrences
        still use raw phi' and raw block sensitivities.

        IDENTITY SKIP: h[n+1]=h[n]+alpha*a_n gives the same-time Jacobian
        dh[n+1]/dh[n] = I + alpha*phi'(z[n])·W_eff^{(n)}. Only the
        weight-path term is scaled; the identity term ADDS ell_h[n+1]
        straight through (dimensionally safe because a skip requires equal widths). So
        ell_h[n] becomes (weight-path backprojection) + ell_h[n+1]. This makes ell_h[n]
        the exact same-time spatial gradient of the RESIDUAL net actually being run.
        direct_fa is unaffected (its boundaries are fixed random projections of the
        readout error, independent of the forward Jacobian).

        Returns ell_h, a length-(L+1) list. ell_h[1..L] are always filled (every
        MP layer's step needs its output signal); ell_h[0] is filled only when
        need_input_signal (the trainable input embedding). For a single MP layer
        with no embedding, exact_spatial / layerwise_fa compute exactly one matmul.
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
            return ell_h

        # exact_spatial vs layerwise_fa: the former uses the exact (modulated)
        # weight transpose at every boundary (weight transport), the latter a fixed
        # random matrix at every boundary (transport-free). Top boundary first, then
        # recurse down; ell_h[0] only when the trainable embedding needs it.
        layerwise = (self.feedback_mode == 'layerwise_fa')
        ell_h[L] = grad_output @ (self.B_feedback if layerwise else self.W_output)
        # Pure local readout: no signal descends from the layers above, so the
        # recursion is skipped entirely (each non-top layer is set from its head).
        pure_local = head_errors is not None and self.learning_signal == 'local_readout'
        if not pure_local:
            for n in range(L - 1, 0, -1):
                delta_n = self._branch_scales[n] * ell_h[n + 1] * phi_p[n]
                if layerwise:
                    ell_h[n] = delta_n @ getattr(self, self._B_inter_names[n])
                else:
                    ell_h[n] = self.mp_layers[n].backproject_through_modulated_weights_fast(delta_n)
                # Identity-skip Jacobian term adds ell_h[n+1] straight
                # through (equal widths guaranteed by _residual_at[n]).
                if self._residual_at[n]:
                    ell_h[n] = ell_h[n] + ell_h[n + 1]
        if head_errors is not None:
            # Local head signals: e_n @ C_n lives at layer n's OUTPUT boundary h[n+1]
            # (same row-vector convention as grad_output @ W_output above). Replaces
            # the boundary signal under local_readout, is added under mixed.
            for n, (w_name, _) in enumerate(self._head_names):
                local_n = head_errors[n] @ getattr(self, w_name)
                ell_h[n + 1] = local_n if pure_local else ell_h[n + 1] + self.local_signal_alpha * local_n
        if need_input_signal:
            delta_0 = self._branch_scales[0] * ell_h[1] * phi_p[0]
            if layerwise:
                ell_h[0] = delta_0 @ getattr(self, self._B_inter_names[0])
            else:
                ell_h[0] = self.mp_layers[0].backproject_through_modulated_weights_fast(delta_0)
            # Embedding→layer-0 identity skip → add ell_h[1] through unscaled.
            if self._residual_at[0]:
                ell_h[0] = ell_h[0] + ell_h[1]

        return ell_h

    def _cross_layer_correction(self, grad_W, grad_b, ell_h, phi_p, h, eta_lam,
                                prev_E, prev_R, prev_ablock, prev_hstream, prev_phi,
                                prev_Weff, prev_update_mask=None):
        """Add the exact depth-1 (one-temporal-hop) cross-layer correction to grad_W
        and grad_b IN PLACE, for the current step t (>0), from the PREVIOUS step's
        Hebbian M-writes. Only the same-time surrogate is dropped by the base local
        rule; this restores the leading temporal term, making every hidden layer's
        WEIGHT AND BIAS gradient exact vs BPTT at T=2 (where the dropped series has a
        single term) under exact spatial feedback and exact weight/bias eligibility.
        This corrects MP parameters only; the input embedding retains its selected
        learning rule. A masked previous write contributes no delayed source.

        Derivation (verified to machine precision by the T=2 oracle). Layer q-1's
        weight/bias reaches L_t one temporal hop earlier through the M-write of EVERY
        upper layer m≥q at step t-1: {W,b}[q-1] → h_stream[q]_{t-1} (sensitivity
        branch_scale[q-1] * prev_E/prev_R[q-1]) → M[m]_{t-1} → z[m]_t → … → L_t. The
        write M[m]_{ki}=η a[m]_k h[m]_i depends on {W,b}[q-1] through BOTH factors:
          PRE  (∂/∂ the pre h[m]_i):  a[m]_{k,t-1} carried as prev_ablock[m];
          POST (∂/∂ the post a[m]_k): a[m]_{k,t-1} itself depends on h_stream[m]_{t-1},
               backprojected through W_eff[m]_{t-1} (= prev_Weff[m] = W(1+M_{t-2})).
        The loss-sensitivity covector to that write is S_ki = δ[m]_k W[m]_{ki} h[m]_{i,t}
        with δ[m]_t = branch_scale[m]*ell_h[m+1]_t·φ'(z[m]_t). The earlier POST
        derivative uses raw a[m], without another branch gain. Both brackets share
        the same downward route from h[m]_{t-1} to h_stream[q]_{t-1} through frozen
        same-time Jacobians, so the whole thing regroups into ONE O(L·width²) adjoint
        sweep: build a source covector v[m] at each upper layer, then descend, folding
        each source in and backprojecting through the t-1 layer Jacobians (weight path +
        residual identity). At each boundary the SAME accumulated adjoint `acc`,
        multiplied by branch_scale[q-1], contracts with prev_E[q-1] for the weight
        AND prev_R[q-1] for the bias
        (b[q-1] feeds h_stream[q]_{t-1} exactly as W[q-1] does, so it shares the whole
        downstream path — only the intra-layer eligibility differs). NOTE the two weight
        operators differ: the PRE bracket uses the RAW W[m] (current-step pre h[m]_t),
        the POST bracket and the downward Jacobian use the EFFECTIVE W_eff[m]_{t-1};
        conflating them breaks T=2 exactness. For pre-only plasticity the PRE
        factor is the same constant 1/sqrt(n_output) used by the forward write,
        and the POST source is zero."""
        layers = self.mp_layers
        L = len(layers)

        # Source covector v[m] living at stream position h[m], for each upper plastic
        # layer m = 1..L-1 (layer 0 has nothing below it to credit; the readout is
        # non-plastic so no adjoint descends from h[L]).
        v = [None] * L
        for m in range(1, L):
            delta = self._branch_scales[m] * ell_h[m + 1] * phi_p[m]
            We = layers[m].W * eta_lam[m][0]                      # W[m]·η[m]  (out_m, in_m)
            # PRE bracket uses the actual postsynaptic factor in the earlier write.
            associative = layers[m].m_update_type == 'hebb_assoc'
            post_factor = (prev_ablock[m] if associative
                           else 1.0 / math.sqrt(layers[m].n_output))
            wk = delta * post_factor                              # (B, out_m)
            write_gate = layers[m]._modulation_write_derivative()
            if write_gate is not None:
                gated_We = We.unsqueeze(0) * write_gate
                u_pre = torch.einsum('Bk,Bki->Bi', wk, gated_We) * h[m]
            else:
                u_pre = torch.einsum('Bk,ki->Bi', wk, We) * h[m]
            # POST bracket: pre factor h[m]_{t-1} fixed, post a[m]_{t-1} backprojected
            # one layer-m hop through W_eff[m]_{t-1}.
            if associative:
                if write_gate is not None:
                    coef_post = delta * prev_phi[m] * torch.einsum(
                        'Bki,Bi,Bi->Bk', gated_We, h[m], prev_hstream[m])
                else:
                    coef_post = delta * prev_phi[m] * torch.einsum(
                        'ki,Bi,Bi->Bk', We, h[m], prev_hstream[m])
                u_post = torch.einsum('Bk,Bki->Bi', coef_post, prev_Weff[m])
                v[m] = u_pre + u_post
            else:
                v[m] = u_pre
            # Mask the source of the write at t-1, not the current loss or the
            # same-time transport Jacobians. Supports both binary and graded masks.
            if prev_update_mask is not None:
                v[m] = v[m] * prev_update_mask.to(v[m]).reshape(-1, 1)

        # Downward adjoint sweep (t-1 Jacobians). acc holds dL_t/dh_stream[q]_{t-1}
        # along the one-hop path; convert it to a raw-block signal before crediting
        # W/b via prev_E/prev_R (weight & bias share the same downstream path).
        acc = None
        for q in range(L - 1, 0, -1):
            if acc is None:
                acc = v[q]
            else:
                # backproject the adjoint at h[q+1] down through layer q (t-1 forward):
                # Scale the weight path only; the identity adds the unscaled acc.
                r = self._branch_scales[q] * acc * prev_phi[q]
                down = torch.einsum('Ba,Bai->Bi', r, prev_Weff[q])
                if self._residual_at[q]:
                    down = down + acc
                acc = v[q] + down
            branch_acc = self._branch_scales[q - 1] * acc
            grad_W[q - 1] = grad_W[q - 1] + torch.einsum('Bc,Bcd->cd', branch_acc, prev_E[q - 1])
            grad_b[q - 1] = grad_b[q - 1] + torch.einsum('Bc,Bc->c', branch_acc, prev_R[q - 1])

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

    def _aux_params(self):
        """Auxiliary (local readout head) parameters, keyed 'head_W{n}'/'head_b{n}'
        by the MP layer n they read. Kept APART from _trainable_params: they are
        trained only by the local passes (bptt_gradients never differentiates them,
        so autograd sees no unused parameter), excluded from the BPTT-alignment
        columns and from the weight-decay group (neither key starts with 'W'), and
        get their .grad written by sequence_gradients only when the pass produced
        one. Empty for learning_signal='global' or a single MP layer."""
        ps = {}
        for w_name, b_name in self._head_names:
            ps[w_name] = getattr(self, w_name)
            if b_name is not None:
                ps[b_name] = getattr(self, b_name)
        return {k: v for k, v in ps.items() if v.requires_grad}

    def _head_outputs(self, h):
        """Auxiliary head predictions q^(n) = C_n h[n+1] + c_n for every head, from
        the per-step stream list h of _forward_local_stack. Linear, so dL/dC_n is
        e_n^T h[n+1] and dL/dc_n is sum_B e_n (accumulated by the local loop)."""
        return [F.linear(h[n + 1], getattr(self, w_name),
                         None if b_name is None else getattr(self, b_name))
                for n, (w_name, b_name) in enumerate(self._head_names)]

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
        inputs = self._standardize_input(inputs)   # fixed norm (identity unless on)
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
                             update_masks, with_heads=True):
        """Forward-only pre-pass for a CUSTOM loss (deep net). Runs the embedding +
        full MP stack with the same clean-config fast M update the accumulation
        loop uses (for EVERY layer), builds the full output sequence, and returns
        (loss, grad_output_seq, aux_losses, aux_grad_seqs) from loss_and_grad. See
        MultiPlasticNet._prepass_output_grad. eta_lam is the per-layer list of
        (eta, lam) expansions (one entry per MP layer). The two aux lists hold one
        entry per local readout head (the SAME loss helper applied to that head's
        prediction sequence, so mask handling and normalization match the main
        readout exactly); both are empty without heads."""
        layers = self.mp_layers
        B, T, _ = inputs.shape
        dev, dt = inputs.device, inputs.dtype
        self.reset_state(B=B)
        outputs = torch.empty(B, T, self.n_output, dtype=dt, device=dev)
        aux_outputs = [torch.empty(B, T, self.n_output, dtype=dt, device=dev)
                       for _ in (self._head_names if with_heads else [])]
        for t in range(T):
            u_t = inputs[:, t, :]
            output, h, z, phi_p, embed_pre, blocks = self._forward_local_stack(
                u_t, return_blocks=True)
            outputs[:, t, :] = output
            if aux_outputs:
                for k, q_k in enumerate(self._head_outputs(h)):
                    aux_outputs[k][:, t, :] = q_k
            um = None if update_masks is None else update_masks[:, t]
            for n, mp in enumerate(layers):
                eta_n, lam_n = eta_lam[n]
                # The Hebbian write uses the retained raw activation, at any scale.
                post = blocks[n]
                mp.update_M_matrix_local_fast(h[n], post, eta=eta_n, lam=lam_n,
                                              update_mask=um)
        loss, grad_output_seq = loss_and_grad(outputs, labels, masks)
        aux = [loss_and_grad(q, labels, masks) for q in aux_outputs]
        return loss, grad_output_seq, [a[0] for a in aux], [a[1] for a in aux]

    @torch.no_grad()
    def _local_sequence_gradients(self, inputs, labels, masks, mode,
                                  loss_and_grad=masked_mse_loss_and_output_grad,
                                  update_masks=None,
                                  return_outputs=True,
                                  use_local_heads=True):
        """Forward-mode local learning for the deep net — ANY number of MP layers.

        use_local_heads=False runs the pass with the GLOBAL signal even on a net
        that owns local readout heads (no head errors, no head gradients, no
        aux_* keys). _apply_input_mode uses it for the three_factor embedding
        splice of a bptt run, so bptt ignores learning_signal in every input mode.

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
        plastic layers. Input mode diag_mtrace additionally propagates the
        first MP layer's diagonal-column sensitivities to embedding parameters;
        it does not change MP parameter gradients or add deeper temporal paths.

        Loss handling mirrors MultiPlasticNet: default masked MSE is inline; a
        custom loss_and_grad triggers a forward-only pre-pass whose grad_output_seq
        feeds the accumulation pass (grads stay consistent with the reported loss).
        """
        layers = self.mp_layers
        L = len(layers)
        for mp in layers:
            mp.assert_local_config()
        assert mode in ('exact', 'diag', 'direct')
        # Resolve before hebb_pre can collapse the MP eligibility mode to direct.
        # Public gradient methods can run independently of self.learning_rule.
        input_mode = resolve_input_mode(self.input_mode, {
            'exact': 'local_exact_rowlocal', 'diag': 'local_diag_rflo',
            'direct': 'local_direct',
        }[mode])
        # hebb_pre: M input-only → dM/dW = dM/db = 0, every plastic trace is zero,
        # so the trace modes collapse to the direct rule (per layer). _assoc already
        # zeroes the trace terms, but routing to 'direct' also skips the unused
        # trace allocation, exactly as the single-layer path does.
        if all(mp.m_update_type == 'hebb_pre' for mp in layers):
            mode = 'direct'

        B, T, _ = inputs.shape
        dev, dt = inputs.device, inputs.dtype
        eta_lam = [mp._eta_lam_full() for mp in layers]   # per-layer (eta, lam), constant over the unroll

        # Fixed input standardization (identity unless on), applied ONCE here so both
        # the custom-loss pre-pass and the accumulation loop see the same standardized
        # u_t (the pre-pass itself must NOT re-normalize). The standardized u_t then
        # feeds the embedding forward AND the embedding's own input gradient (u_seq).
        inputs = self._standardize_input(inputs)

        # Custom loss: forward-only pre-pass supplies grad_output_seq and the loss
        # (see MultiPlasticNet._local_sequence_gradients / _prepass_output_grad).
        default_loss = (loss_and_grad is masked_mse_loss_and_output_grad)
        need_outputs = return_outputs or (not default_loss)
        # Local readout heads (learning_signal != 'global'; none on a single MP
        # layer, in which case this pass is byte-identical to the global one).
        heads = self._head_names if use_local_heads else []
        n_heads = len(heads)
        prepass_loss, grad_output_seq = (None, None)
        aux_prepass_loss, aux_grad_seq = ([], [])
        if not default_loss:
            prepass_loss, grad_output_seq, aux_prepass_loss, aux_grad_seq = \
                self._prepass_output_grad(inputs, labels, masks, loss_and_grad,
                                          eta_lam, update_masks,
                                          with_heads=use_local_heads)

        self.reset_state(B=B)
        for mp in layers:
            if mode == 'exact':
                mp.reset_local_learning_state(B=B)
            elif mode == 'diag':
                mp.reset_diag_rflo_state(B=B)

        # Embedding gradients are gated PER TENSOR (weight/bias frozen independently):
        # `embed` runs the selected input rule if EITHER is trainable; need_W_in /
        # need_b_in then decide which key to emit, matching _trainable_params exactly.
        need_W_in, need_b_in = self._embed_grad_flags()
        embed = need_W_in or need_b_in
        input_mtrace = embed and input_mode == 'diag_mtrace'
        if input_mtrace:
            if self.feedback_mode != 'exact_spatial':
                raise ValueError("diag_mtrace requires exact_spatial feedback")
            feature_count = (self.n_input if need_W_in else 0) + int(need_b_in)
            input_trace = torch.zeros(B, layers[0].n_output, layers[0].n_input,
                                      feature_count, dtype=dt, device=dev)
            grad_input = torch.zeros(layers[0].n_input, feature_count, dtype=dt, device=dev)
        step_fns = [mp.step_fn_for(mode) for mp in layers]   # per-layer rule, resolved once

        grad_W = [torch.zeros_like(mp.W) for mp in layers]
        grad_b = [torch.zeros_like(mp.b) for mp in layers]

        # Cross-layer depth-1 temporal correction (see _cross_layer_correction). Only
        # active with cross_layer_steps>0 AND more than one MP layer (a single layer has
        # no upper neighbor to route temporal credit through). It needs each layer's
        # per-step eligibility E[n] and one step of history, so it keeps the PREVIOUS
        # step's (E, block activation, stream, phi', W_eff) per layer. When off, none of
        # this is allocated and the loop is byte-identical to before.
        do_cross = (self.cross_layer_steps >= 1) and (L > 1)
        if do_cross:
            prev_E = [None] * L        # E[n]_{t-1} = dh_stream[n+1]/dW[n]  (B,out_n,in_n)
            prev_R = [None] * L        # R[n]_{t-1} = dh_stream[n+1]/db[n]  (B,out_n)
            prev_ablock = [None] * L   # block activation a[n]_{t-1}        (B,out_n)
            prev_hstream = [None] * (L + 1)   # residual stream h[.]_{t-1}
            prev_phi = [None] * L      # phi'(z[n])_{t-1}                   (B,out_n)
            prev_Weff = [None] * L     # W_eff[n] the t-1 forward CONSUMED = W(1+M_{t-2})
            prev_update_mask = None    # which batch rows wrote their M at t-1
            # Per-layer eligibility + trace-advance callables for the explicit path the
            # correction needs (resolved once, like step_fns). Direct mode has no trace.
            compute_elig, update_trace = [], []
            for mp in layers:
                if mode == 'exact':
                    compute_elig.append(mp.compute_exact_rowlocal_eligibility)
                    update_trace.append(mp.update_exact_rowlocal_traces)
                elif mode == 'diag':
                    compute_elig.append(mp.compute_diag_rflo_eligibility)
                    update_trace.append(mp.update_diag_rflo_traces)
                else:  # direct: no plastic trace to advance
                    compute_elig.append(mp.compute_direct_local_eligibility)
                    update_trace.append(None)

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
        if embed and not input_mtrace:
            # ga_seq (= ell_h[0]⊙φ') feeds BOTH the W_in grad (⊗ u) and the b_in grad
            # (Σ), so it is needed whenever the branch runs; u_seq only when W_in does.
            ga_seq = self._scratch('dlocal_ga', (T, B, self.W_initial_linear.weight.shape[0]), dt, dev)
            if need_W_in:
                u_seq = self._scratch('dlocal_u', (T, B, self.n_input), dt, dev)

        N = B * T * self.n_output

        # Local readout heads: per-head gradient accumulators (outer products summed
        # per step — no (T,B,·) scratch), streamed aux losses, and (optionally) the
        # head prediction sequences for accuracy logging.
        if n_heads:
            grad_head_W = [torch.zeros_like(getattr(self, w_name)) for w_name, _ in heads]
            grad_head_b = [torch.zeros(self.n_output, dtype=dt, device=dev) for _ in heads]
            aux_loss_sum = [torch.zeros((), dtype=dt, device=dev) for _ in heads]
            aux_outputs = ([torch.empty(B, T, self.n_output, dtype=dt, device=dev) for _ in heads]
                           if need_outputs else None)

        for t in range(T):
            u_t = inputs_T[t]

            # 1. Forward the whole stack with M_{t-1} (no M advance yet).
            output, h, z, phi_p, embed_pre, blocks = self._forward_local_stack(
                u_t, return_blocks=True)
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

            # 2b. Local readout heads: each head's own output error on the SAME
            #     labels/mask (same loss formula and 1/N normalization as the main
            #     readout; custom losses come from the pre-pass), plus the head's
            #     own gradient — q is linear in (C_n, c_n), so dL_n/dC_n = e_n^T h[n+1].
            head_errors = None
            if n_heads:
                q = self._head_outputs(h)
                head_errors = []
                for k in range(n_heads):
                    if default_loss:
                        diff_k = m_t * q[k] - m_t * labels_T[t]
                        e_k = (2.0 / N) * m_t * diff_k
                        aux_loss_sum[k] = aux_loss_sum[k] + (diff_k * diff_k).sum()
                    else:
                        e_k = aux_grad_seq[k][:, t, :]
                    head_errors.append(e_k)
                    grad_head_W[k] += torch.einsum('Ba,Bi->ai', e_k, h[k + 1])
                    grad_head_b[k] += e_k.sum(0)
                    if aux_outputs is not None:
                        aux_outputs[k][:, t, :] = q[k]

            um = None if um_T is None else um_T[t]

            # 3. All same-time boundary signals, using the still-frozen M_{t-1}.
            #    ell_h[n+1] credits MP layer n. Direct input gradients use ell_h[0];
            #    diag_mtrace instead credits its first-layer eligibility with ell_h[1].
            #    With local readout heads, head_errors replace (local_readout) or
            #    augment (mixed) the non-top boundary signals.
            ell_h = self._same_time_boundary_signals(grad_output, phi_p,
                                                      need_input_signal=embed and not input_mtrace,
                                                      head_errors=head_errors)

            # 4. Input gradients consume M_{t-1}; the new trace is finalized only
            #    after the first MP layer's write. The ordinary branch is unchanged.
            if input_mtrace:
                features = u_t if need_W_in else u_t[:, :0]
                if need_b_in:
                    features = torch.cat((features, torch.ones(B, 1, dtype=dt, device=dev)), dim=1)
                post0 = blocks[0]
                grad_t, proposed_input_trace = self._input_mtrace_step(
                    input_trace, features, self.act_fn_p(embed_pre), h[0], post0,
                    phi_p[0], ell_h[1], *eta_lam[0])
                grad_input += grad_t
            elif embed:
                ga_seq[t] = ell_h[0] * self.act_fn_p(embed_pre)
                if need_W_in:
                    u_seq[t] = u_t

            # 5. Per-layer local gradient + trace advance.
            if not do_cross:
                # Fast path (default): the fused per-mode step (no E materialized for
                # diag/direct). Traces describe raw blocks; only ell is scaled.
                for n, mp in enumerate(layers):
                    eta_n, lam_n = eta_lam[n]
                    grad_W_t, grad_b_t = step_fns[n](
                        h[n], phi_p[n], self._branch_scales[n] * ell_h[n + 1],
                        eta_n, lam_n, um)
                    grad_W[n] += grad_W_t
                    grad_b[n] += grad_b_t
            else:
                # Cross-layer correction needs each layer's per-step eligibility E, so
                # use the explicit compute+update path (algebraically identical to the
                # fused step — same grads — but exposes E). W_eff at THIS step uses the
                # frozen M_{t-1}, captured BEFORE the step-6 M advance for next step's
                # correction (it must read M_{t-2}).
                cur_E = [None] * L
                cur_R = [None] * L
                cur_Weff = [mp.W * (1.0 + mp.M) for mp in layers]   # W(1+M_{t-1})
                for n, mp in enumerate(layers):
                    E_n, R_n = compute_elig[n](h[n], phi_p[n])
                    branch_signal = self._branch_scales[n] * ell_h[n + 1]
                    grad_W[n] += torch.einsum('Bi,BiI->iI', branch_signal, E_n)
                    grad_b[n] += torch.einsum('Bi,Bi->i', branch_signal, R_n)
                    cur_E[n] = E_n
                    cur_R[n] = R_n
                # Depth-1 temporal correction from the PREVIOUS step's writes (t>0).
                # Corrects BOTH grad_W (via prev_E) and grad_b (via prev_R).
                if t > 0:
                    self._cross_layer_correction(
                        grad_W, grad_b, ell_h, phi_p, h, eta_lam,
                        prev_E, prev_R, prev_ablock, prev_hstream, prev_phi, prev_Weff,
                        prev_update_mask=prev_update_mask)
                # Advance the intra-layer traces (exact/diag keep a trace; direct has
                # none). Uses the just-computed cur_E/cur_R, exactly like the fused step.
                for n, mp in enumerate(layers):
                    if update_trace[n] is not None:
                        update_trace[n](h[n], cur_E[n], cur_R[n],
                                        update_mask=um, eta_lam=eta_lam[n])
            # 6. Advance M with input stream h[n] and raw block activation blocks[n].
            #    The branch scale changes output sensitivities, not this write rule.
            for n, mp in enumerate(layers):
                eta_n, lam_n = eta_lam[n]
                post = blocks[n]
                mp.update_M_matrix_local_fast(
                    h[n], post, eta=eta_n, lam=lam_n, update_mask=um)
                if input_mtrace and n == 0:
                    input_trace = self._finish_input_mtrace(proposed_input_trace, input_trace, um)

            # Stash this step's state for next step's depth-1 correction (post-M-advance
            # is fine: the correction reads none of the just-advanced M; W_eff was
            # captured pre-advance as cur_Weff = W(1+M_{t-1})).
            if do_cross:
                for n in range(L):
                    prev_E[n] = cur_E[n]
                    prev_R[n] = cur_R[n]
                    prev_ablock[n] = blocks[n]
                    prev_phi[n] = phi_p[n]
                    prev_Weff[n] = cur_Weff[n]
                for n in range(L + 1):
                    prev_hstream[n] = h[n]
                prev_update_mask = um

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
        if input_mtrace:
            if need_W_in:
                all_grads['W_in'] = grad_input[:, :self.n_input]
            if need_b_in:
                all_grads['b_in'] = grad_input[:, -1]
        elif embed:
            # Emit each embedding key by its OWN trainability (weight/bias frozen
            # independently), matching _trainable_params so result has exactly its keys.
            if need_W_in:
                all_grads['W_in'] = torch.einsum('TBo,TBI->oI', ga_seq, u_seq)
            if need_b_in:
                all_grads['b_in'] = ga_seq.sum(dim=(0, 1))
        # Local readout heads: 'head_W{n}'/'head_b{n}' (see _aux_params).
        if n_heads:
            for k, (w_name, b_name) in enumerate(heads):
                all_grads[w_name] = grad_head_W[k]
                if b_name is not None:
                    all_grads[b_name] = grad_head_b[k]

        params = self._trainable_params()
        result = {k: all_grads[k] for k in params}
        if n_heads:                     # heads active in THIS pass (not under use_local_heads=False)
            for k in self._aux_params():
                result[k] = all_grads[k]
        result['loss'] = loss.detach()
        result['outputs'] = outputs.detach() if return_outputs and outputs is not None else None
        if n_heads:
            # Per-head task losses (same objective as the main readout, on that
            # head's prediction) and, when requested, the head prediction sequences.
            # Reported SEPARATELY from 'loss', which stays the main readout's.
            if default_loss:
                result['aux_loss'] = [
                    (masked_mse_loss_only(aux_outputs[k], labels, masks) if aux_outputs is not None
                     else aux_loss_sum[k] / N).detach() for k in range(n_heads)]
            else:
                result['aux_loss'] = [l.detach() for l in aux_prepass_loss]
            result['aux_outputs'] = ([o.detach() for o in aux_outputs]
                                     if return_outputs and aux_outputs is not None else None)
        return result

    def local_gradients(self, inputs, labels, masks, **kwargs):
        """Exact intra-layer row-local eligibility per MP layer + same-time
        inter-layer learning signals + the selected local input rule. ONLY with
        feedback_mode='exact_spatial' is the TOP plastic layer's gradient exact vs
        BPTT for weights (and biases when local_bias_mode='exact'); LOWER plastic layers
        stay surrogates (they omit temporal paths through upper plastic layers), and
        under the FA feedback modes even the top layer is no longer exact."""
        return self._local_sequence_gradients(inputs, labels, masks, 'exact', **kwargs)

    def local_diag_rflo_gradients(self, inputs, labels, masks, **kwargs):
        """Diagonal RFLO eligibility per MP layer + same-time inter-layer learning
        signals + the selected local input rule. APPROXIMATE for EVERY plastic
        layer — including the top — regardless of feedback_mode, because the
        diagonal eligibility itself drops off-synapse plastic sensitivities (exact
        only when n_input == 1). The readout gradient stays exact."""
        return self._local_sequence_gradients(inputs, labels, masks, 'diag', **kwargs)

    def local_direct_gradients(self, inputs, labels, masks, **kwargs):
        """Direct/instantaneous eligibility per MP layer + same-time inter-layer
        learning signals + the selected local input rule. For MP parameter gradients,
        treats M_{t-1} as a stop-gradient state (spatial backprop through the deep
        feedforward net with frozen modulation). APPROXIMATE for EVERY plastic
        layer — including the top —
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

    @property
    def resolved_input_mode(self):
        """Input algorithm for the current rule; recomputed after cloning/reloading."""
        return resolve_input_mode(self.input_mode, self.learning_rule)

    def _apply_input_mode(self, grads, inputs, labels, masks, **kwargs):
        """Override W_in/b_in in `grads` when input_mode requests a rule the native
        pass did not already produce for the embedding.

        The embedding's NATIVE rule is 'exact' under bptt and 'three_factor' under
        any local rule unless diag_mtrace is selected. Both local input rules are
        independent of the MP eligibility mode. A splice is
        needed only when input_mode disagrees with that native rule:
          input_mode='exact'        on a LOCAL run → take W_in/b_in from a BPTT pass.
          input_mode='three_factor' on a BPTT run  → take W_in/b_in from a local pass
                                                      (local_direct is the cheapest).
        diag_mtrace and paired are handled inside the selected pass; BPTT stays exact.
        'match', a non-trainable embedding, or an already-matching native rule → no-op.
        The extra pass recomputes the full gradient but only W_in/b_in are kept."""
        if self.input_mode in ('match', 'diag_mtrace', 'paired') or not self._has_trainable_embed():
            return grads
        native_is_exact = (self.learning_rule == 'bptt')
        want_exact = (self.input_mode == 'exact')
        if want_exact == native_is_exact:
            return grads                      # native pass already produced it
        other_rule = 'bptt' if want_exact else 'local_direct'
        if other_rule == 'bptt' and self.learning_signal != 'global':
            raise ValueError(
                "input_mode='exact' splices a full-BPTT embedding gradient into a local "
                f"run, which is not local; under learning_signal='{self.learning_signal}' "
                "use match, three_factor, diag_mtrace or paired.")
        if other_rule == 'local_direct':
            # bptt ignores learning_signal: its three_factor embedding splice is the
            # direct 3-factor rule under the GLOBAL (main readout) signal, never a
            # local head's error (the heads are untrained under bptt anyway).
            kwargs = dict(kwargs, use_local_heads=False)
        other = self._grads_for_rule(other_rule, inputs, labels, masks, **kwargs)
        for k in ('W_in', 'b_in'):
            if k in grads and k in other:
                grads[k] = other[k]
        return grads

    def sequence_gradients(self, inputs, labels, masks, **kwargs):
        """Dispatch on self.learning_rule; write grads into each param's .grad.
        Input modes exact/three_factor may splice W_in/b_in from another pass.
        diag_mtrace/paired run inside the selected pass and leave BPTT unchanged."""
        grads = self._grads_for_rule(self.learning_rule, inputs, labels, masks, **kwargs)
        grads = self._apply_input_mode(grads, inputs, labels, masks, **kwargs)

        for name, p in self._trainable_params().items():
            p.grad = grads[name].clone()
        # Local readout heads are trained only by the local passes: bptt produces no
        # head gradient, so its heads get .grad=None (also CLEARING any gradient left
        # by an earlier local pass on the same model) and the optimizer skips them.
        for name, p in self._aux_params().items():
            p.grad = grads[name].clone() if name in grads else None
        return grads
