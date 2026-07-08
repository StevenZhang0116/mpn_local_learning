"""
Leaky vanilla RNN with two learning rules, mirroring the MPN's BPTT / local
split (see mpn.MultiPlasticNet):

  'bptt'            — autograd through the unrolled recurrent forward.
  'local_diag_rflo' — RFLO (Murray & Escola 2019, "Local online learning in
                      recurrent networks with random feedback"): forward-mode
                      eligibility traces that DROP the network-mediated recurrent
                      sensitivity term. This is the RNN analog of the MPN
                      diagonal / same-synapse approximation, so it shares the
                      API key 'local_diag_rflo'. It is an APPROXIMATION to BPTT
                      (exact only when W_rec = 0, i.e. no recurrence).

Dynamics (leaky, alpha = 1 - dt/tau):
    u_t   = W_input x_t + W_rec h_{t-1} + b_hidden
    h_t   = alpha h_{t-1} + (1 - alpha) phi(u_t)
    y_t   = W_output h_t + b_output

RFLO eligibility traces (i = post/hidden unit; j = pre hidden; I = pre input):
    p^rec_{ij,t} = alpha p^rec_{ij,t-1} + (1-alpha) phi'(u_{i,t}) h_{j,t-1}
    q^in_{iI,t}  = alpha q^in_{iI,t-1}  + (1-alpha) phi'(u_{i,t}) x_{I,t}
    s^b_{i,t}    = alpha s^b_{i,t-1}    + (1-alpha) phi'(u_{i,t})
    ell_{i,t}    = (dL_t/dy_t) . F[:, i]          # F = W_output or random B_feedback
    dL/dW_rec_{ij}   = sum_t ell_{i,t} p^rec_{ij,t}
    dL/dW_input_{iI} = sum_t ell_{i,t} q^in_{iI,t}
    dL/db_{i}        = sum_t ell_{i,t} s^b_{i,t}
Readout grads stay EXACT (dL/dW_output, dL/db_output use the true dL_t/dy_t).
Under 'local_diag_rflo' ALL weights (input, recurrent, bias) train by RFLO; under
'bptt' everything trains by exact autograd — the learning_rule governs the whole
network, there is no separate per-layer control.

Trainable params: W_input, W_rec, W_output, b_hidden, (b_output if enabled).
"""
import torch

from net_helpers import BaseNetwork
from net_helpers import rand_weight_init, get_activation_function
from mpn import masked_mse_loss_and_output_grad  # one shared masked-MSE definition


class LeakyRNN(BaseNetwork):
    def __init__(self, net_params, verbose=False):
        super().__init__(net_params, verbose=verbose)

        if 'n_neurons' in net_params:
            self.n_input, self.n_hidden, self.n_output = net_params['n_neurons']
        else:
            self.n_input = net_params['n_input']
            self.n_hidden = net_params['n_hidden']
            self.n_output = net_params['n_output']

        self.act = net_params.get('activation', 'tanh')
        self.act_fn, self.act_fn_np, self.act_fn_p = get_activation_function(self.act)

        # Leaky integration factor (alpha = 1 - dt/tau; set by
        # convert_and_init_multitask_params, default 0.8 as in LD's paper).
        self.leaky = net_params.get('leaky', True)
        self.alpha = net_params.get('alpha', 0.8) if self.leaky else 0.0

        self.b_hidden_active = net_params.get('hidden_bias', True)
        self.b_output_active = net_params.get('output_bias', False)

        # Learning rule / feedback (mirrors MultiPlasticNet).
        self.learning_rule = net_params.get('learning_rule', 'bptt')
        assert self.learning_rule in ('bptt', 'local_diag_rflo'), \
            f"unknown learning_rule '{self.learning_rule}'"
        self.feedback_mode = net_params.get('feedback_mode', 'exact_readout')
        assert self.feedback_mode in ('exact_readout', 'random_fixed'), \
            f"unknown feedback_mode '{self.feedback_mode}'"

        self.param_clamping = False

        # Trainable parameter list (used by parameter_or_buffer).
        self.params = ['W_input', 'W_rec', 'W_output']
        if self.b_hidden_active:
            self.params.append('b_hidden')
        if self.b_output_active:
            self.params.append('b_output')

        # ── Weights (rand_weight_init(n_in, n_out) -> shape (n_out, n_in)) ──
        self.parameter_or_buffer('W_input', torch.tensor(
            rand_weight_init(self.n_input, self.n_hidden,
                             init_type=net_params.get('W_input_init', 'xavier')),
            dtype=torch.float))
        self.parameter_or_buffer('W_rec', torch.tensor(
            rand_weight_init(self.n_hidden, self.n_hidden,
                             init_type=net_params.get('W_rec_init', 'xavier')),
            dtype=torch.float))
        self.parameter_or_buffer('W_output', torch.tensor(
            rand_weight_init(self.n_hidden, self.n_output,
                             init_type=net_params.get('W_output_init', 'xavier')),
            dtype=torch.float))

        self.parameter_or_buffer('b_hidden', torch.tensor(
            rand_weight_init(self.n_hidden,
                             init_type='gaussian' if self.b_hidden_active else 'zeros'),
            dtype=torch.float))
        self.parameter_or_buffer('b_output', torch.tensor(
            rand_weight_init(self.n_output,
                             init_type='gaussian' if self.b_output_active else 'zeros'),
            dtype=torch.float))

        # Fixed random feedback matrix (same shape as W_output = (n_output, n_hidden)).
        if self.feedback_mode == 'random_fixed':
            self.register_buffer('B_feedback', torch.tensor(
                rand_weight_init(self.n_hidden, self.n_output,
                                 init_type=net_params.get('B_feedback_init', 'xavier')),
                dtype=self.W_output.dtype))

        if verbose:
            print(f"LeakyRNN: in={self.n_input} hidden={self.n_hidden} "
                  f"out={self.n_output} act={self.act} alpha={self.alpha} "
                  f"rule={self.learning_rule} feedback={self.feedback_mode}")

    def reset_state(self, B=1):
        self.hidden = torch.zeros(B, self.n_hidden, device=self.W_rec.device,
                                  dtype=self.W_rec.dtype)

    def _trainable_params(self):
        ps = {'W_input': self.W_input, 'W_rec': self.W_rec, 'W_output': self.W_output}
        if self.b_hidden_active:
            ps['b_hidden'] = self.b_hidden
        if self.b_output_active:
            ps['b_output'] = self.b_output
        return {k: v for k, v in ps.items() if v.requires_grad}

    def _step(self, x_t, h_prev):
        """One leaky recurrent step. Returns (u_t pre-activation, h_t, y_t)."""
        u = (torch.einsum('iI,BI->Bi', self.W_input, x_t)
             + torch.einsum('ij,Bj->Bi', self.W_rec, h_prev)
             + self.b_hidden.unsqueeze(0))
        h = self.alpha * h_prev + (1.0 - self.alpha) * self.act_fn(u)
        y = torch.einsum('ai,Bi->Ba', self.W_output, h) + self.b_output.unsqueeze(0)
        return u, h, y

    @torch.no_grad()
    def forward_outputs(self, inputs):
        """No-grad unrolled forward, returning outputs (B, T, n_output). Used to
        score held-out data."""
        B, T, _ = inputs.shape
        h = torch.zeros(B, self.n_hidden, device=inputs.device, dtype=inputs.dtype)
        outs = []
        for t in range(T):
            _, h, y = self._step(inputs[:, t, :], h)
            outs.append(y)
        return torch.stack(outs, dim=1)

    def bptt_gradients(self, inputs, labels, masks,
                       loss_and_grad=masked_mse_loss_and_output_grad):
        """Full BPTT gradients via autograd through the unrolled recurrence.
        Returns {param_name: grad, ..., 'loss', 'outputs'}."""
        B, T, _ = inputs.shape
        h = torch.zeros(B, self.n_hidden, device=inputs.device, dtype=inputs.dtype)
        outs = []
        for t in range(T):
            _, h, y = self._step(inputs[:, t, :], h)
            outs.append(y)
        outputs = torch.stack(outs, dim=1)

        loss, _ = loss_and_grad(outputs, labels, masks)
        params = self._trainable_params()
        grads = torch.autograd.grad(loss, list(params.values()))
        result = {k: g.detach().clone() for k, g in zip(params, grads)}
        result['loss'] = loss.detach()
        result['outputs'] = outputs.detach()
        return result

    @torch.no_grad()
    def local_diag_rflo_gradients(self, inputs, labels, masks,
                                  loss_and_grad=masked_mse_loss_and_output_grad,
                                  update_masks=None):
        """RFLO gradients — forward-mode eligibility traces, no BPTT. Drops the
        network-mediated recurrent sensitivity term (approximation to BPTT).
        Readout gradients stay exact. Returns the same dict shape as
        bptt_gradients()."""
        B, T, _ = inputs.shape
        dev, dt = inputs.device, inputs.dtype
        a1 = 1.0 - self.alpha

        h = torch.zeros(B, self.n_hidden, device=dev, dtype=dt)
        p_rec = torch.zeros(B, self.n_hidden, self.n_hidden, device=dev, dtype=dt)
        q_in = torch.zeros(B, self.n_hidden, self.n_input, device=dev, dtype=dt)
        s_b = torch.zeros(B, self.n_hidden, device=dev, dtype=dt)

        grad_W_rec = torch.zeros_like(self.W_rec)
        grad_W_input = torch.zeros_like(self.W_input)
        grad_b_hidden = torch.zeros_like(self.b_hidden)
        grad_W_output = torch.zeros_like(self.W_output)
        grad_b_output = torch.zeros_like(self.b_output)
        outputs = torch.zeros(B, T, self.n_output, dtype=dt, device=dev)

        feedback = self.W_output if self.feedback_mode == 'exact_readout' else self.B_feedback
        N = B * T * self.n_output

        for t in range(T):
            x_t = inputs[:, t, :]
            h_prev = h
            u, h, y = self._step(x_t, h_prev)
            outputs[:, t, :] = y

            # Advance eligibility traces to time t (RFLO: local, drops W_rec-
            # mediated cross-neuron term). Uses phi'(u_t) and h_{t-1} / x_t.
            phi_p = self.act_fn_p(u)                                # (B, i)
            p_rec_new = self.alpha * p_rec + a1 * torch.einsum('Bi,Bj->Bij', phi_p, h_prev)
            q_in_new = self.alpha * q_in + a1 * torch.einsum('Bi,BI->BiI', phi_p, x_t)
            s_b_new = self.alpha * s_b + a1 * phi_p

            if update_masks is not None:
                # Frozen batch rows keep the previous trace / hidden state and
                # contribute no gradient (padding beyond sequence end).
                m = update_masks[:, t].to(dt)
                p_rec = m.view(-1, 1, 1) * p_rec_new + (1.0 - m).view(-1, 1, 1) * p_rec
                q_in = m.view(-1, 1, 1) * q_in_new + (1.0 - m).view(-1, 1, 1) * q_in
                s_b = m.view(-1, 1) * s_b_new + (1.0 - m).view(-1, 1) * s_b
                h = m.view(-1, 1) * h + (1.0 - m).view(-1, 1) * h_prev
            else:
                p_rec, q_in, s_b = p_rec_new, q_in_new, s_b_new

            # Per-timestep output gradient (masked MSE, global 1/N) + learning signal.
            m_t, y_t = masks[:, t, :], labels[:, t, :]
            grad_output = (2.0 / N) * m_t * (m_t * y - m_t * y_t)   # (B, a)
            ell = grad_output @ feedback                            # (B, i)

            grad_W_rec += torch.einsum('Bi,Bij->ij', ell, p_rec)
            grad_W_input += torch.einsum('Bi,BiI->iI', ell, q_in)
            grad_b_hidden += torch.einsum('Bi,Bi->i', ell, s_b)
            grad_W_output += torch.einsum('Ba,Bi->ai', grad_output, h)
            grad_b_output += grad_output.sum(0)

        loss, _ = loss_and_grad(outputs, labels, masks)
        all_grads = {'W_input': grad_W_input, 'W_rec': grad_W_rec,
                     'b_hidden': grad_b_hidden, 'W_output': grad_W_output,
                     'b_output': grad_b_output}

        params = self._trainable_params()
        result = {k: all_grads[k] for k in params}
        result['loss'] = loss.detach()
        result['outputs'] = outputs.detach()
        return result

    def sequence_gradients(self, inputs, labels, masks, **kwargs):
        """Dispatch on self.learning_rule and write grads into each parameter's
        .grad (so optimizer.step() works as with loss.backward())."""
        if self.learning_rule == 'bptt':
            grads = self.bptt_gradients(inputs, labels, masks, **kwargs)
        elif self.learning_rule == 'local_diag_rflo':
            grads = self.local_diag_rflo_gradients(inputs, labels, masks, **kwargs)
        else:
            raise ValueError(f"unknown learning_rule '{self.learning_rule}'")

        for name, p in self._trainable_params().items():
            p.grad = grads[name].clone()
        return grads
