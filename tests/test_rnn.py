"""core/rnn.py LeakyRNN: RFLO under a custom (cross-entropy) loss and under
update_masks, against a detached-recurrence autograd oracle (exactly the RFLO
approximation: h_{t-1} constant inside the recurrence, the alpha leak kept),
plus the all-frozen limit. Complements validate_local_learning tier 7 (MSE).
Run from tests/: python -m unittest test_rnn -v"""
import contextlib
import io
import unittest

import numpy as np
import torch

import _bootstrap  # noqa: F401
import mpn
import rnn

H, I, O, B, T = 5, 3, 2, 4, 6


def make_net(rule='local_diag_rflo', feedback='exact_spatial', seed=3):
    np.random.seed(seed)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        net = rnn.LeakyRNN(dict(n_neurons=[I, H, O], output_bias=True, hidden_bias=True,
                                learning_rule=rule, feedback_mode=feedback, loss_type='MSE',
                                alpha=0.7), verbose=False).double()
    return net


def data(seed=11):
    torch.manual_seed(seed)
    x = torch.randn(B, T, I, dtype=torch.double)
    y = torch.randn(B, T, O, dtype=torch.double)
    mask = torch.rand(B, T, O, dtype=torch.double)
    mask[0, 0] = 0
    labels = torch.zeros_like(y)
    labels[torch.arange(B), :, torch.randint(0, O, (B,))] = 1
    return x, y, mask, labels


def mse(out, y, mask):
    return ((mask * out - mask * y) ** 2).sum() / out.numel()


def rflo_oracle(net, x, y, mask, loss_fn=None, update_masks=None):
    """RFLO as autograd: u_t uses the DETACHED h_{t-1} (dropped recurrent term), the
    leak alpha*h_{t-1} keeps the live state; frozen rows keep h BEFORE the readout."""
    ps = {k: v.detach().clone().requires_grad_() for k, v in net._trainable_params().items()}
    h = torch.zeros(B, H, dtype=torch.double)
    outs = []
    for t in range(T):
        u = x[:, t] @ ps['W_input'].t() + h.detach() @ ps['W_rec'].t() + ps['b_hidden']
        h_new = net.alpha * h + (1 - net.alpha) * torch.tanh(u)
        if update_masks is not None:
            m = update_masks[:, t].view(-1, 1)
            h = m * h_new + (1 - m) * h
        else:
            h = h_new
        outs.append(h @ ps['W_output'].t() + ps['b_output'])
    out = torch.stack(outs, 1)
    loss = mse(out, y, mask) if loss_fn is None else loss_fn(out, y, mask)[0]
    return dict(zip(ps, torch.autograd.grad(loss, list(ps.values()))))


class TestLeakyRNNRFLO(unittest.TestCase):
    def test_mse_matches_oracle_and_readout_exact(self):
        x, y, mask, _ = data()
        net = make_net()
        got = net.local_diag_rflo_gradients(x, y, mask)
        ref = rflo_oracle(net, x, y, mask)
        for k, v in ref.items():
            torch.testing.assert_close(got[k], v, rtol=1e-10, atol=1e-12)
        bptt = net.bptt_gradients(x, y, mask)
        for k in ('W_output', 'b_output'):
            torch.testing.assert_close(got[k], bptt[k], rtol=1e-10, atol=1e-12)

    def test_custom_loss_drives_rflo(self):
        x, y, mask, labels = data()
        ce = mpn.masked_cross_entropy_loss_and_grad
        net = make_net()
        got = net.local_diag_rflo_gradients(x, labels, mask, loss_and_grad=ce)
        ref = rflo_oracle(net, x, labels, mask, loss_fn=ce)
        for k, v in ref.items():
            torch.testing.assert_close(got[k], v, rtol=1e-10, atol=1e-12)
        bptt = net.bptt_gradients(x, labels, mask, loss_and_grad=ce)
        for k in ('W_output', 'b_output'):
            torch.testing.assert_close(got[k], bptt[k], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(got['loss'], ce(got['outputs'], labels, mask)[0])
        under_mse = net.local_diag_rflo_gradients(x, labels, mask)
        self.assertGreater((got['W_output'] - under_mse['W_output']).abs().max().item(), 1e-3)
        net.sequence_gradients(x, labels, mask, loss_and_grad=ce)
        torch.testing.assert_close(net.W_output.grad, got['W_output'])

    def test_update_masks_freeze_state_before_readout(self):
        x, y, mask, labels = data()
        net = make_net()
        full = net.local_diag_rflo_gradients(x, y, mask)
        um1 = torch.ones(B, T, dtype=torch.double)
        same = net.local_diag_rflo_gradients(x, y, mask, update_masks=um1)
        for k in net._trainable_params():
            torch.testing.assert_close(same[k], full[k], rtol=0, atol=0)
        um0 = torch.zeros(B, T, dtype=torch.double)
        got0 = net.local_diag_rflo_gradients(x, y, mask, update_masks=um0)
        torch.testing.assert_close(got0['outputs'], net.b_output.expand(B, T, O), rtol=0, atol=0)
        for k in ('W_input', 'W_rec', 'b_hidden', 'W_output'):
            self.assertEqual(got0[k].abs().max().item(), 0.0, k)
        um = um1.clone()
        um[0, 2:] = 0
        um[1, 3] = 0.25
        # (The CE helper's analytic gradient assumes one-hot targets, so use labels.)
        for loss_fn, target in ((None, y), (mpn.masked_cross_entropy_loss_and_grad, labels)):
            kw = {} if loss_fn is None else {'loss_and_grad': loss_fn}
            got = net.local_diag_rflo_gradients(x, target, mask, update_masks=um, **kw)
            ref = rflo_oracle(net, x, target, mask, loss_fn=loss_fn, update_masks=um)
            for k, v in ref.items():
                torch.testing.assert_close(got[k], v, rtol=1e-10, atol=1e-12)
        # forward_outputs == the unmasked unroll used by evaluation.
        torch.testing.assert_close(net.forward_outputs(x), full['outputs'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
