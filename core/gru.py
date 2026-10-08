"""
Gated recurrent unit (GRU) with two learning rules, mirroring rnn.LeakyRNN so
the same lockstep BPTT-vs-local training loop (scripts/train_common.py) and the
same tasks apply:

  'bptt'            — autograd through the unrolled recurrence.
  'local_diag_rflo' — RFLO-style forward-mode eligibility traces (Murray & Escola
                      2019) adapted to the GRU: every parameter keeps a per-unit
                      trace of dh_i/dtheta that follows ONLY the unit's own
                      update-gate leak z_i (the GRU analog of the leaky RNN's
                      alpha) and DROPS the recurrent sensitivity of h_{t-1} that
                      enters the gates through W_rec. It is an APPROXIMATION to
                      BPTT, exact when W_rec = 0 (no recurrence through the gates)
                      or T = 1. Same API key as the MPN diagonal rule / RNN RFLO.

Dynamics (PyTorch GRUCell convention; gate order r, z, n in the stacked weights):
    r_t = sigmoid(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)
    z_t = sigmoid(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)
    m_t = W_hn h_{t-1} + b_hn
    n_t = tanh(W_in x_t + b_in + r_t * m_t)
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    y_t = W_output h_t + b_output

RFLO eligibility (unit i; theta a parameter in unit i's row of one gate block).
With h~ = h_{t-1} held CONSTANT inside the gates (the dropped recurrent term):
    A_i = (1 - z_i) (1 - n_i^2)            dh_i / d(pre_n)_i
    C_i = (h~_i - n_i) z_i (1 - z_i)       dh_i / d(pre_z)_i
    D_i = A_i m_i r_i (1 - r_i)            dh_i / d(pre_r)_i   (r enters n via r*m)
    immediate_t(theta) = coef_i * input_of_theta_t
        rows of W_input / b_input:   coef = (D, C, A)        x_t  / 1
        rows of W_rec   / b_rec:     coef = (D, C, A * r)    h~_t / 1   (b_hn is scaled by r)
    P_t(theta) = z_{i,t} * P_{t-1}(theta) + immediate_t(theta)
    ell_{i,t}  = (dL_t/dy_t) . F[:, i]        F = W_output (exact_spatial) or random B
    dL/dtheta  = sum_t ell_{i,t} P_t(theta)
Readout gradients stay EXACT. Under 'local_diag_rflo' the whole recurrent block
trains by these traces; under 'bptt' everything trains by autograd.

Trainable params: W_input (3H x I), W_rec (3H x H), W_output (O x H),
b_input (3H), b_rec (3H) when hidden_bias, b_output when output_bias.
Trace memory: batch x 3H x (I + H + 2) floats, independent of sequence length.
"""
import time

import torch

from net_helpers import BaseNetwork
from net_helpers import rand_weight_init
from mpn import masked_mse_loss_and_output_grad  # one shared masked-MSE definition
from mpn import canonical_feedback_mode           # shared feedback-mode vocabulary


class GRU(BaseNetwork):
    def __init__(self, net_params, verbose=False):
        super().__init__(net_params, verbose=verbose)

        if 'n_neurons' in net_params:
            self.n_input, self.n_hidden, self.n_output = net_params['n_neurons']
        else:
            self.n_input = net_params['n_input']
            self.n_hidden = net_params['n_hidden']
            self.n_output = net_params['n_output']

        self.b_hidden_active = net_params.get('hidden_bias', True)
        self.b_output_active = net_params.get('output_bias', False)

        self.learning_rule = net_params.get('learning_rule', 'bptt')
        assert self.learning_rule in ('bptt', 'local_diag_rflo'), \
            f"unknown learning_rule '{self.learning_rule}'"
        # One hidden layer → both random-feedback variants are plain feedback
        # alignment on the readout → hidden boundary; only exact_spatial differs.
        self.feedback_mode = canonical_feedback_mode(
            net_params.get('feedback_mode', 'exact_spatial'))

        self.param_clamping = False

        self.params = ['W_input', 'W_rec', 'W_output']
        if self.b_hidden_active:
            self.params.extend(['b_input', 'b_rec'])
        if self.b_output_active:
            self.params.append('b_output')

        H3 = 3 * self.n_hidden
        # rand_weight_init(n_in, n_out) -> (n_out, n_in); the three gate blocks are
        # stacked along the output dim in PyTorch order (r, z, n).
        self.parameter_or_buffer('W_input', torch.tensor(
            rand_weight_init(self.n_input, H3,
                             init_type=net_params.get('W_input_init', 'xavier')),
            dtype=torch.float))
        self.parameter_or_buffer('W_rec', torch.tensor(
            rand_weight_init(self.n_hidden, H3,
                             init_type=net_params.get('W_rec_init', 'xavier')),
            dtype=torch.float))
        self.parameter_or_buffer('W_output', torch.tensor(
            rand_weight_init(self.n_hidden, self.n_output,
                             init_type=net_params.get('W_output_init', 'xavier')),
            dtype=torch.float))
        bias_init = 'gaussian' if self.b_hidden_active else 'zeros'
        self.parameter_or_buffer('b_input', torch.tensor(
            rand_weight_init(H3, init_type=bias_init), dtype=torch.float))
        self.parameter_or_buffer('b_rec', torch.tensor(
            rand_weight_init(H3, init_type=bias_init), dtype=torch.float))
        self.parameter_or_buffer('b_output', torch.tensor(
            rand_weight_init(self.n_output,
                             init_type='gaussian' if self.b_output_active else 'zeros'),
            dtype=torch.float))

        if self.feedback_mode != 'exact_spatial':
            self.register_buffer('B_feedback', torch.tensor(
                rand_weight_init(self.n_hidden, self.n_output,
                                 init_type=net_params.get('B_feedback_init', 'xavier')),
                dtype=self.W_output.dtype))

        if verbose:
            print(f"GRU: in={self.n_input} hidden={self.n_hidden} out={self.n_output} "
                  f"rule={self.learning_rule} feedback={self.feedback_mode}")

    # ── state / params ───────────────────────────────────────────────────────
    def reset_state(self, B=1):
        self.hidden = torch.zeros(B, self.n_hidden, device=self.W_rec.device,
                                  dtype=self.W_rec.dtype)

    def _trainable_params(self):
        ps = {'W_input': self.W_input, 'W_rec': self.W_rec, 'W_output': self.W_output}
        if self.b_hidden_active:
            ps['b_input'] = self.b_input
            ps['b_rec'] = self.b_rec
        if self.b_output_active:
            ps['b_output'] = self.b_output
        return {k: v for k, v in ps.items() if v.requires_grad}

    # ── forward ──────────────────────────────────────────────────────────────
    def _gates(self, x_t, h_prev):
        """(r, z, n, m) for one step; m = W_hn h_prev + b_hn (the hidden part of
        the candidate pre-activation, which the reset gate scales)."""
        H = self.n_hidden
        gi = x_t @ self.W_input.t() + self.b_input            # (B, 3H)
        gh = h_prev @ self.W_rec.t() + self.b_rec             # (B, 3H)
        r = torch.sigmoid(gi[:, :H] + gh[:, :H])
        z = torch.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        m = gh[:, 2 * H:]
        n = torch.tanh(gi[:, 2 * H:] + r * m)
        return r, z, n, m

    def _step(self, x_t, h_prev):
        """One GRU step. Returns ((r, z, n, m), h_t, y_t)."""
        r, z, n, m = self._gates(x_t, h_prev)
        h = (1.0 - z) * n + z * h_prev
        y = h @ self.W_output.t() + self.b_output
        return (r, z, n, m), h, y

    @torch.no_grad()
    def forward_outputs(self, inputs):
        """No-grad unrolled forward → outputs (B, T, n_output); scores held-out data."""
        B, T, _ = inputs.shape
        h = torch.zeros(B, self.n_hidden, device=inputs.device, dtype=inputs.dtype)
        outs = []
        for t in range(T):
            _, h, y = self._step(inputs[:, t, :], h)
            outs.append(y)
        return torch.stack(outs, dim=1)

    # ── learning rules ───────────────────────────────────────────────────────
    def bptt_gradients(self, inputs, labels, masks,
                       loss_and_grad=masked_mse_loss_and_output_grad):
        """Full BPTT via autograd through the unrolled recurrence; records the
        fwd/bwd wall-time split like rnn.LeakyRNN for the timing readout."""
        B, T, _ = inputs.shape
        _cuda = self.W_output.is_cuda
        if _cuda:
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        h = torch.zeros(B, self.n_hidden, device=inputs.device, dtype=inputs.dtype)
        outs = []
        for t in range(T):
            _, h, y = self._step(inputs[:, t, :], h)
            outs.append(y)
        outputs = torch.stack(outs, dim=1)
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
        result = {k: g.detach().clone() for k, g in zip(params, grads)}
        result['loss'] = loss.detach()
        result['outputs'] = outputs.detach()
        return result

    @torch.no_grad()
    def local_diag_rflo_gradients(self, inputs, labels, masks,
                                  loss_and_grad=masked_mse_loss_and_output_grad,
                                  update_masks=None):
        """GRU-RFLO gradients: forward-mode per-unit eligibility traces (see the
        module docstring), gated by each unit's own update gate z_i, dropping the
        recurrent sensitivity of h_{t-1} inside the gates. Readout exact. Returns
        the same dict shape as bptt_gradients()."""
        B, T, _ = inputs.shape
        dev, dt = inputs.device, inputs.dtype
        H = self.n_hidden

        h = torch.zeros(B, H, device=dev, dtype=dt)
        P_in = torch.zeros(B, 3 * H, self.n_input, device=dev, dtype=dt)
        P_rec = torch.zeros(B, 3 * H, H, device=dev, dtype=dt)
        p_bin = torch.zeros(B, 3 * H, device=dev, dtype=dt)
        p_brec = torch.zeros(B, 3 * H, device=dev, dtype=dt)

        grad = {k: torch.zeros_like(v) for k, v in
                (('W_input', self.W_input), ('W_rec', self.W_rec), ('b_input', self.b_input),
                 ('b_rec', self.b_rec), ('W_output', self.W_output), ('b_output', self.b_output))}
        outputs = torch.zeros(B, T, self.n_output, dtype=dt, device=dev)

        feedback = self.W_output if self.feedback_mode == 'exact_spatial' else self.B_feedback
        N = B * T * self.n_output

        for t in range(T):
            x_t = inputs[:, t, :]
            h_prev = h
            (r, z, n, m), h, y = self._step(x_t, h_prev)
            outputs[:, t, :] = y

            # Immediate sensitivities of h_i to its own gate pre-activations, with
            # h_prev treated as constant inside the gates (the dropped RFLO term).
            A = (1.0 - z) * (1.0 - n * n)                 # candidate gate
            C = (h_prev - n) * z * (1.0 - z)              # update gate
            D = A * m * r * (1.0 - r)                     # reset gate (via r*m in n)
            coef_in = torch.cat((D, C, A), dim=1)         # rows of W_input / b_input
            coef_rec = torch.cat((D, C, A * r), dim=1)    # rows of W_rec / b_rec
            z3 = z.repeat(1, 3)                           # per-unit leak for each gate block

            P_in_new = z3.unsqueeze(-1) * P_in + coef_in.unsqueeze(-1) * x_t.unsqueeze(1)
            P_rec_new = z3.unsqueeze(-1) * P_rec + coef_rec.unsqueeze(-1) * h_prev.unsqueeze(1)
            p_bin_new = z3 * p_bin + coef_in
            p_brec_new = z3 * p_brec + coef_rec

            if update_masks is not None:
                # Frozen batch rows keep their traces / hidden state (padding).
                mk = update_masks[:, t].to(dt)
                keep = (1.0 - mk)
                P_in = mk.view(-1, 1, 1) * P_in_new + keep.view(-1, 1, 1) * P_in
                P_rec = mk.view(-1, 1, 1) * P_rec_new + keep.view(-1, 1, 1) * P_rec
                p_bin = mk.view(-1, 1) * p_bin_new + keep.view(-1, 1) * p_bin
                p_brec = mk.view(-1, 1) * p_brec_new + keep.view(-1, 1) * p_brec
                h = mk.view(-1, 1) * h + keep.view(-1, 1) * h_prev
            else:
                P_in, P_rec, p_bin, p_brec = P_in_new, P_rec_new, p_bin_new, p_brec_new

            m_t, y_t = masks[:, t, :], labels[:, t, :]
            grad_output = (2.0 / N) * m_t * (m_t * y - m_t * y_t)   # (B, a)
            ell = grad_output @ feedback                            # (B, H), per unit
            ell3 = ell.repeat(1, 3)                                 # same signal for each gate row

            grad['W_input'] += torch.einsum('Bg,BgI->gI', ell3, P_in)
            grad['W_rec'] += torch.einsum('Bg,Bgj->gj', ell3, P_rec)
            grad['b_input'] += torch.einsum('Bg,Bg->g', ell3, p_bin)
            grad['b_rec'] += torch.einsum('Bg,Bg->g', ell3, p_brec)
            grad['W_output'] += torch.einsum('Ba,Bi->ai', grad_output, h)
            grad['b_output'] += grad_output.sum(0)

        loss, _ = loss_and_grad(outputs, labels, masks)
        params = self._trainable_params()
        result = {k: grad[k] for k in params}
        result['loss'] = loss.detach()
        result['outputs'] = outputs.detach()
        return result

    def sequence_gradients(self, inputs, labels, masks, **kwargs):
        """Dispatch on self.learning_rule and write each parameter's .grad."""
        if self.learning_rule == 'bptt':
            grads = self.bptt_gradients(inputs, labels, masks, **kwargs)
        elif self.learning_rule == 'local_diag_rflo':
            grads = self.local_diag_rflo_gradients(inputs, labels, masks, **kwargs)
        else:
            raise ValueError(f"unknown learning_rule '{self.learning_rule}'")
        for name, p in self._trainable_params().items():
            p.grad = grads[name].clone()
        return grads
