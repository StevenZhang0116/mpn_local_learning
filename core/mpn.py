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

    def assert_local_assoc_config(self):
        """Guard shared by the exact row-local and diagonal-RFLO rules. Both are
        derived for the associative Hebbian M-update

            M_{iI,t} = lam M_{iI,t-1} + eta h_{i,t} x_{I,t}

        with multiplicative modulation, a linear M-activation, and no clamping.
        Other m_update_types (hebb_pre, oja) / nonlinear m_act / bounds would
        need extra terms (a dM_t/dM_pre factor, zero/undefined saturated grads),
        so they are refused here rather than silently giving wrong gradients."""
        if self.mp_type != 'mult':
            raise NotImplementedError(
                f"local rules derived for mp_type='mult', got '{self.mp_type}'")
        if self.m_update_type != 'hebb_assoc':
            raise NotImplementedError(
                f"local rules derived for m_update_type='hebb_assoc', got "
                f"'{self.m_update_type}'")
        if self.m_act != 'linear':
            raise NotImplementedError(
                f"local rules require m_activation='linear', got '{self.m_act}'")
        if self.modulation_bounds:
            raise NotImplementedError(
                "local rules require modulation_bounds=False (clamping gives "
                "zero/undefined dM_t/dM_pre in saturated regions)")

    # Back-compat alias (older callers / tests used the exact-only name).
    assert_exact_rowlocal_config = assert_local_assoc_config

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

    def update_exact_rowlocal_traces(self, x, E, R, update_mask=None):
        """Advance the eligibility traces one step (uses M_t's eta/lam):

        P^I_{iJ,t} = lam_{iJ} P^I_{iJ,t-1} + eta_{iJ} x_J E^I_{i,t}
        Q_{iJ,t}   = lam_{iJ} Q_{iJ,t-1}   + eta_{iJ} x_J R_{i,t}

        Call AFTER compute_exact_rowlocal_eligibility, in step with
        update_M_matrix. update_mask (B,) freezes traces for inactive batch rows.
        """
        eta, lam = self._eta_lam_full()  # each (i, J)

        outerP = torch.einsum('BiI,BJ->BiIJ', E, x)            # x_J E^I_i
        P_new = lam[None, :, None, :] * self.P + eta[None, :, None, :] * outerP

        outerQ = torch.einsum('Bi,BJ->BiJ', R, x)              # x_J R_i
        Q_new = lam[None, :, :] * self.Q + eta[None, :, :] * outerQ

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

    def update_diag_rflo_traces(self, x, E, R, update_mask=None):
        """Advance the diagonal-RFLO traces one step (uses M_t's eta/lam):

        A_{iI,t} = lam_{iI} A_{iI,t-1} + eta_{iI} x_I E_hat^I_{i,t}
        Q_{iJ,t} = lam_{iJ} Q_{iJ,t-1} + eta_{iJ} x_J R_{i,t}          (exact)

        Call AFTER compute_diag_rflo_eligibility, in step with update_M_matrix.
        update_mask (B,) freezes traces for inactive batch rows.
        """
        eta, lam = self._eta_lam_full()  # each (i, I)

        A_new = lam[None] * self.A + eta[None] * x.unsqueeze(1) * E     # x_I E_hat^I_i

        outerQ = torch.einsum('Bi,BJ->BiJ', R, x)                      # x_J R_i
        Q_new = lam[None] * self.Q + eta[None] * outerQ

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

    def update_M_matrix(self, pre, post, update_mask=None):
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
        """

        eta = self.build_M_parameter(self.eta, self.eta_type)
        lam = self.build_M_parameter(self.lam, self.lam_type)
        M = self.M

        delta_M = torch.zeros_like(M)

        if update_mask is not None: # Update each batch_idx individually using the update_mask
            for batch_idx in range(M.shape[0]):
                if update_mask[batch_idx]: # Only calculates delta_M if batch is being updated (I think this saves time?)
                    if self.m_update_type in ('hebb_pre',):
                        post = 1 / math.sqrt(post.shape[-1]) * torch.ones_like(post)

                    if self.m_update_type in ('hebb_assoc', 'hebb_pre',):
                        delta_M[batch_idx] = - M[batch_idx] + lam * M[batch_idx] + eta * torch.einsum(
                            'i, I -> iI', post[batch_idx], pre[batch_idx]
                        )
                    elif self.m_update_type in ('oja',):
                        delta_M[batch_idx] = (eta * torch.einsum('i, I -> iI', post[batch_idx], pre[batch_idx]) -
                                              torch.abs(eta) * torch.einsum('i, iI -> iI', post[batch_idx]**2, M[batch_idx]))
        else: # Update all M at once
            if self.m_update_type in ('hebb_pre',):
                post = 1 / math.sqrt(post.shape[-1]) * torch.ones_like(post)

            if self.m_update_type in ('hebb_assoc', 'hebb_pre',):
                delta_M = - M + lam.unsqueeze(0) * M + eta.unsqueeze(0) * torch.einsum(
                    'Bi, BI -> BiI', post, pre
                )

            elif self.m_update_type in ('oja',):
                raise NotImplementedError('Need to update to a batched version.')
                # delta_M[batch_idx] = (eta * torch.einsum('i, I -> iI', post[batch_idx], pre[batch_idx]) -
                #                       torch.abs(eta) * torch.einsum('i, iI -> iI', post[batch_idx]**2, M[batch_idx]))

        self.M_pre = self.M + delta_M
        self.M = self.m_act_fn(self.M_pre)

        # # Masks batches of delta_M
        # delta_M_masked = torch.einsum('B, BiI -> BiI', update_mask, delta_M)

        # Update M matrices, while being sure update holds matrix within bounds
        # (this may error if not self.ei_types, but this is always true in our settings)
        # (note: updates to restristed cell types is built into the eta matrix)
        if self.modulation_bounds:
            self.M = torch.clamp(self.M, min=self.M_bounds[1], max=self.M_bounds[0])

        # Freeze plasticity at masked positions: restore M to its initial-state values
        if hasattr(self, '_plasticity_freeze_mask') and self._plasticity_freeze_mask is not None:
            self.M[:, self._plasticity_freeze_mask[0],
                      self._plasticity_freeze_mask[1]] = self._M_frozen_vals

        return delta_M # This is only used for theory matching

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

        modulated_weights = self.get_modulated_weights()
        pre_act_no_bias =  torch.einsum('BiI, BI -> Bi', modulated_weights, x)

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

        # Learning signal for the hidden layer: 'exact_readout' uses W_output
        # (true gradient); 'random_fixed' uses a fixed random matrix B_feedback
        # (feedback alignment). Buffer registered in super().__init__ once
        # W_output's shape is known.
        self.feedback_mode = net_params.get('feedback_mode', 'exact_readout')
        assert self.feedback_mode in ('exact_readout', 'random_fixed'), \
            f"unknown feedback_mode '{self.feedback_mode}'"

        super().__init__(net_params, self.n_hidden, verbose=verbose)

        # Fixed random feedback matrix for feedback alignment (same shape as
        # W_output, never trained). Only allocated when requested.
        if self.feedback_mode == 'random_fixed':
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

        x = torch.clone(inputs)

        # Returns pre-activation
        hidden_pre, db_mp = self.mp_layer(x, run_mode=run_mode)

        hidden = self.act_fn(hidden_pre)

        output_hidden = torch.einsum('iI, BI -> Bi', self.W_output, hidden)
        output = output_hidden + self.b_output.unsqueeze(0)

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
                       loss_and_grad=masked_mse_loss_and_output_grad):
        """Full BPTT gradients via autograd through the unrolled forward +
        update_M_matrix loop. Returns {param_name: grad, ..., 'loss', 'outputs'}.
        With m_activation='linear' and no bounds, update_M_matrix is fully
        differentiable, so autograd through it is exactly BPTT."""
        B, T, _ = inputs.shape
        self.reset_state(B=B)
        outs = []
        for t in range(T):
            x_t = inputs[:, t, :]
            hidden_pre, _ = self.mp_layer(x_t)
            hidden = self.act_fn(hidden_pre)
            out = torch.einsum('iI,BI->Bi', self.W_output, hidden) + self.b_output.unsqueeze(0)
            outs.append(out)
            self.mp_layer.update_M_matrix(x_t, hidden)
        outputs = torch.stack(outs, dim=1)

        loss, _ = loss_and_grad(outputs, labels, masks)
        params = self._trainable_params()
        grads = torch.autograd.grad(loss, list(params.values()))
        result = {k: g.detach().clone() for k, g in zip(params, grads)}
        result['loss'] = loss.detach()
        result['outputs'] = outputs.detach()
        return result

    @torch.no_grad()
    def _local_sequence_gradients(self, inputs, labels, masks, mode,
                                  loss_and_grad=masked_mse_loss_and_output_grad,
                                  update_masks=None):
        """Shared forward-mode local-learning loop for the three local rules:
          mode='exact'  — exact row-local (full trace P, Q)
          mode='diag'   — diagonal / same-synapse RFLO (trace A, exact Q)
          mode='direct' — direct/instantaneous (no trace; stops gradient through
                          the plasticity history)
        They differ only in the eligibility / trace equations; the loop order,
        learning signal, and gradient accumulation are identical.

        Per-time-step order (must match the theory):
            forward (uses M_{t-1}) -> learning signal ell -> eligibility E,R
            (from t-1 traces) -> gradient accumulation -> trace update -> M update.

        ell_{i,t} = (dL_t/dy_t) . F[:, i], where F = W_output (exact_readout) or
        a fixed random B_feedback (random_fixed, feedback alignment). The readout
        gradients (W_output, b_output) always use the true grad_output. Returns
        the same dict shape as bptt_gradients().
        """
        mp = self.mp_layer
        mp.assert_local_assoc_config()
        assert mode in ('exact', 'diag', 'direct')

        B, T, _ = inputs.shape
        dev, dt = inputs.device, inputs.dtype
        self.reset_state(B=B)
        if mode == 'exact':
            mp.reset_local_learning_state(B=B)
        elif mode == 'diag':
            mp.reset_diag_rflo_state(B=B)
        # mode == 'direct' needs no eligibility state.

        # Feedback matrix for the hidden learning signal (W_output or random).
        feedback = self.W_output if self.feedback_mode == 'exact_readout' else self.B_feedback

        grad_W = torch.zeros_like(mp.W)
        grad_b = torch.zeros_like(mp.b)
        grad_Wout = torch.zeros_like(self.W_output)
        grad_bout = torch.zeros_like(self.b_output)
        outputs = torch.zeros(B, T, self.n_output, dtype=dt, device=dev)

        # Global normalizer for the per-timestep output gradient: the loss is a
        # mean over ALL B*T*n_out elements, so dL/d output_t carries the full
        # 1/N (not a per-step count). The per-step grad below is the analytic
        # masked-MSE gradient this rule is derived for; the loss_and_grad kwarg
        # is used only for the returned scalar loss.
        N = B * T * self.n_output

        for t in range(T):
            x_t = inputs[:, t, :]

            # Forward at time t (consumes M_{t-1}).
            hidden_pre, _ = mp(x_t)
            hidden = self.act_fn(hidden_pre)
            output = torch.einsum('iI,BI->Bi', self.W_output, hidden) + self.b_output.unsqueeze(0)
            outputs[:, t, :] = output

            # Eligibility E=dh_t/dW, R=dh_t/db (exact / diagonal / direct).
            phi_prime = self.act_fn_p(hidden_pre)
            if mode == 'exact':
                E, R = mp.compute_exact_rowlocal_eligibility(x_t, phi_prime)
            elif mode == 'diag':
                E, R = mp.compute_diag_rflo_eligibility(x_t, phi_prime)
            else:  # 'direct'
                E, R = mp.compute_direct_local_eligibility(x_t, phi_prime)

            # Per-timestep output gradient (masked MSE, global 1/N). Hidden
            # learning signal uses the feedback matrix; readout grads use the
            # true grad_output.
            m_t, y_t = masks[:, t, :], labels[:, t, :]
            grad_output = (2.0 / N) * m_t * (m_t * output - m_t * y_t)
            ell = grad_output @ feedback                    # (B, n_hidden)

            grad_W += torch.einsum('Bi,BiI->iI', ell, E)
            grad_b += torch.einsum('Bi,Bi->i', ell, R)
            grad_Wout += torch.einsum('Ba,Bi->ai', grad_output, hidden)
            grad_bout += grad_output.sum(0)

            # Advance traces (if any) then modulations (M -> t), both mask-aware.
            # 'direct' keeps no trace, but STILL updates M — the rule uses the
            # true MPN forward dynamics; it only stops eligibility flow through
            # the history that produced M.
            um = None if update_masks is None else update_masks[:, t]
            if mode == 'exact':
                mp.update_exact_rowlocal_traces(x_t, E, R, update_mask=um)
            elif mode == 'diag':
                mp.update_diag_rflo_traces(x_t, E, R, update_mask=um)
            mp.update_M_matrix(x_t, hidden, update_mask=um)

        loss, _ = loss_and_grad(outputs, labels, masks)
        params = self._trainable_params()
        all_grads = {'W': grad_W, 'b': grad_b, 'W_output': grad_Wout, 'b_output': grad_bout}
        result = {k: all_grads[k] for k in params}
        result['loss'] = loss.detach()
        result['outputs'] = outputs.detach()
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
    N-layer feedforward setup, with N-1 multi-plastic layers followed by a single readout layer.
    """

    def __init__(self, net_params, verbose=False, forzihan=True):
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
        self.feedback_mode = cfg.get('feedback_mode', 'exact_readout')
        assert self.feedback_mode in ('exact_readout', 'random_fixed'), \
            f"unknown feedback_mode '{self.feedback_mode}'"

        super().__init__(cfg, cfg['n_neurons'][-2], output_matrix=self.output_matrix, verbose=verbose)

        # Fixed random feedback matrix for feedback alignment (shape of W_output).
        if self.feedback_mode == 'random_fixed':
            self.register_buffer('B_feedback', torch.tensor(
                rand_weight_init(self.n_hidden, self.n_output,
                                 init_type=cfg.get('B_feedback_init', 'xavier')),
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
        # if additional input layer is added, shift the starting index of layer counting from 1
        for mpl_idx in range(start_layer_count, n_layers - 1): # (e.g. three-layer has two MP layers)
            if forzihan:
                assert n_layers - 1 - start_layer_count == 1, "2025-10-29: One-Layer MPN Now"

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

        layer_input = torch.clone(x)

        mpl_activities = [x,] # Used for updating the M matrices

        db = {} if run_mode in ('track_states',) else None

        for mpl_idx, mp_layer in enumerate(self.mp_layers):
            # Returns pre-activation
            layer_input_old = layer_input.clone()
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
            
        output_hidden = torch.einsum('iI, BI -> Bi', self.W_output, layer_input)
        output = output_hidden + self.b_output.unsqueeze(0)
        
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
    #     u -> [W_initial_linear, act] -> x -> [MP: W, M] -> h -> [W_output] -> y
    # which is directly comparable to the RNN (input -> hidden -> output), with the
    # plastic M playing the role the RNN's recurrence plays. Requires exactly one
    # MP layer (the forzihan config). BPTT trains ALL params (incl. the input
    # embedding) exactly via autograd; the local rules train the MP layer with
    # eligibility traces and the input embedding with a DIRECT 3-factor rule
    # (backproject the hidden learning signal through the modulated weights, then
    # multiply by the embedding activation derivative and the raw input — the RNN's
    # RFLO treatment of its input weights; M-mediated history is dropped).

    def _assert_single_mp(self):
        assert len(self.mp_layers) == 1, \
            "deep-net learning rules support exactly one MP layer (forzihan config)"

    def _has_trainable_embed(self):
        return (self.input_layer_active and self.W_initial_linear.weight.requires_grad)

    def _trainable_params(self):
        """Trainable tensors this net computes gradients for, keyed by name."""
        mp = self.mp_layers[0]
        ps = {'W': mp.W, 'W_output': self.W_output}
        if mp.layer_bias:
            ps['b'] = mp.b
        if self.b_output_active:
            ps['b_output'] = self.b_output
        if self.input_layer_active:
            ps['W_in'] = self.W_initial_linear.weight
            if self.W_initial_linear.bias is not None:
                ps['b_in'] = self.W_initial_linear.bias
        return {k: v for k, v in ps.items() if v.requires_grad}

    def bptt_gradients(self, inputs, labels, masks,
                       loss_and_grad=masked_mse_loss_and_output_grad):
        """Full BPTT via autograd through the unrolled deep forward + M-update.
        Trains every parameter (input embedding included) exactly."""
        self._assert_single_mp()
        B, T, _ = inputs.shape
        self.reset_state(B=B)
        outs = []
        for t in range(T):
            out, _, _ = self.network_step(inputs[:, t, :], seq_idx=t)
            outs.append(out)
        outputs = torch.stack(outs, dim=1)

        loss, _ = loss_and_grad(outputs, labels, masks)
        params = self._trainable_params()
        grads = torch.autograd.grad(loss, list(params.values()))
        result = {k: g.detach().clone() for k, g in zip(params, grads)}
        result['loss'] = loss.detach()
        result['outputs'] = outputs.detach()
        return result

    @torch.no_grad()
    def _local_sequence_gradients(self, inputs, labels, masks, mode,
                                  loss_and_grad=masked_mse_loss_and_output_grad,
                                  update_masks=None):
        """Forward-mode local learning for the deep net (one MP layer + optional
        trainable input embedding). Same three modes as MultiPlasticNet for the MP
        layer ('exact'/'diag'/'direct'); the input embedding always uses the direct
        3-factor rule. Per-step order: forward (uses M_{t-1}) -> learning signal ->
        eligibility -> grad accumulation -> trace update -> M update."""
        self._assert_single_mp()
        mp = self.mp_layers[0]
        mp.assert_local_assoc_config()
        assert mode in ('exact', 'diag', 'direct')

        B, T, _ = inputs.shape
        dev, dt = inputs.device, inputs.dtype
        self.reset_state(B=B)
        if mode == 'exact':
            mp.reset_local_learning_state(B=B)
        elif mode == 'diag':
            mp.reset_diag_rflo_state(B=B)

        feedback = self.W_output if self.feedback_mode == 'exact_readout' else self.B_feedback
        embed = self._has_trainable_embed()

        grad_W = torch.zeros_like(mp.W)
        grad_b = torch.zeros_like(mp.b)
        grad_Wout = torch.zeros_like(self.W_output)
        grad_bout = torch.zeros_like(self.b_output)
        if embed:
            grad_Win = torch.zeros_like(self.W_initial_linear.weight)
            has_bin = self.W_initial_linear.bias is not None
            grad_bin = torch.zeros_like(self.W_initial_linear.bias) if has_bin else None
        outputs = torch.zeros(B, T, self.n_output, dtype=dt, device=dev)
        N = B * T * self.n_output

        for t in range(T):
            u_t = inputs[:, t, :]                       # raw input (B, n_input)

            # Input embedding forward: a = W_in u (+ b_in); x = act(a).
            if self.input_layer_active:
                a_t = self.W_initial_linear(u_t)
                x_t = self.act_fn(a_t)
            else:
                a_t, x_t = None, u_t

            # MP layer forward (consumes M_{t-1}); readout.
            hidden_pre, _ = mp(x_t)
            hidden = self.act_fn(hidden_pre)
            output = torch.einsum('iI,BI->Bi', self.W_output, hidden) + self.b_output.unsqueeze(0)
            outputs[:, t, :] = output

            # MP-layer eligibility E=dh/dW, R=dh/db from prev traces.
            phi_prime = self.act_fn_p(hidden_pre)
            if mode == 'exact':
                E, R = mp.compute_exact_rowlocal_eligibility(x_t, phi_prime)
            elif mode == 'diag':
                E, R = mp.compute_diag_rflo_eligibility(x_t, phi_prime)
            else:
                E, R = mp.compute_direct_local_eligibility(x_t, phi_prime)

            # Output gradient + hidden learning signal (feedback matrix).
            m_t, y_t = masks[:, t, :], labels[:, t, :]
            grad_output = (2.0 / N) * m_t * (m_t * output - m_t * y_t)
            ell = grad_output @ feedback                 # (B, n_hidden)

            grad_W += torch.einsum('Bi,BiI->iI', ell, E)
            grad_b += torch.einsum('Bi,Bi->i', ell, R)
            grad_Wout += torch.einsum('Ba,Bi->ai', grad_output, hidden)
            grad_bout += grad_output.sum(0)

            # Input embedding, local (direct 3-factor) rule. Backproject ell
            # through the modulated MP weights to the embedding output x, then
            # through act'(a).
            #   ell_x_I = sum_i ell_i * phi'(h~_i) * W_eff_iI ;  grad_a = ell_x * act'(a)
            #   dL/dW_in = grad_a^T u ;  dL/db_in = sum_B grad_a
            if embed:
                W_eff = mp.get_modulated_weights()       # (B, i, I) = W + W⊙M_{t-1}
                ell_pre = ell * phi_prime                # (B, i)  = dL/dh~_i
                ell_x = torch.einsum('Bi,BiI->BI', ell_pre, W_eff)   # (B, I=embed dim)
                grad_a = ell_x * self.act_fn_p(a_t)      # through x = act(a)
                grad_Win += torch.einsum('Bo,BI->oI', grad_a, u_t)
                if has_bin:
                    grad_bin += grad_a.sum(0)

            # Advance traces then M.
            um = None if update_masks is None else update_masks[:, t]
            if mode == 'exact':
                mp.update_exact_rowlocal_traces(x_t, E, R, update_mask=um)
            elif mode == 'diag':
                mp.update_diag_rflo_traces(x_t, E, R, update_mask=um)
            mp.update_M_matrix(x_t, hidden, update_mask=um)

        loss, _ = loss_and_grad(outputs, labels, masks)
        all_grads = {'W': grad_W, 'b': grad_b, 'W_output': grad_Wout, 'b_output': grad_bout}
        if embed:
            all_grads['W_in'] = grad_Win
            if has_bin:
                all_grads['b_in'] = grad_bin

        params = self._trainable_params()
        result = {k: all_grads[k] for k in params}
        result['loss'] = loss.detach()
        result['outputs'] = outputs.detach()
        return result

    def local_gradients(self, inputs, labels, masks, **kwargs):
        """Exact row-local MP-layer gradients + direct 3-factor input embedding."""
        return self._local_sequence_gradients(inputs, labels, masks, 'exact', **kwargs)

    def local_diag_rflo_gradients(self, inputs, labels, masks, **kwargs):
        """Diagonal RFLO MP-layer gradients + direct 3-factor input embedding."""
        return self._local_sequence_gradients(inputs, labels, masks, 'diag', **kwargs)

    def local_direct_gradients(self, inputs, labels, masks, **kwargs):
        """Direct/instantaneous MP-layer gradients + direct 3-factor input embedding."""
        return self._local_sequence_gradients(inputs, labels, masks, 'direct', **kwargs)

    def sequence_gradients(self, inputs, labels, masks, **kwargs):
        """Dispatch on self.learning_rule; write grads into each param's .grad."""
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
