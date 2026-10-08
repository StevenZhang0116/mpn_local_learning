"""core/gru.py + scripts/train_gru.py: BPTT against an independent torch.nn.GRUCell
autograd oracle, GRU-RFLO traces against a detached-recurrence autograd oracle
(exactly the approximation they implement), exactness limits (W_rec = 0, T = 1),
exact readout, feedback modes, update masks, the training-loop integration and
the CLI. Small CPU double tensors. Run from tests/: python -m unittest test_gru -v"""
import contextlib
import copy
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import _bootstrap  # noqa: F401
import gru
import train_common
import train_gru

H, I, O, B, T = 5, 3, 2, 4, 6


def make_net(rule='bptt', feedback='exact_spatial', hidden_bias=True, output_bias=True, seed=3):
    np.random.seed(seed)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        net = gru.GRU(dict(n_neurons=[I, H, O], output_bias=output_bias, hidden_bias=hidden_bias,
                           learning_rule=rule, feedback_mode=feedback, loss_type='MSE'),
                      verbose=False).double()
    return net


def data(seed=11, T=T):
    torch.manual_seed(seed)
    x = torch.randn(B, T, I, dtype=torch.double)
    y = torch.randn(B, T, O, dtype=torch.double)
    mask = torch.rand(B, T, O, dtype=torch.double)
    mask[0, 0] = 0
    return x, y, mask


def mse(out, y, mask):
    return ((mask * out - mask * y) ** 2).sum() / out.numel()


def grucell_oracle(net, x, y, mask):
    """Independent BPTT reference: torch.nn.GRUCell with the net's weights copied
    in (same PyTorch gate layout r, z, n), unrolled with autograd."""
    cell = torch.nn.GRUCell(net.n_input, net.n_hidden, bias=True).double()
    with torch.no_grad():
        cell.weight_ih.copy_(net.W_input)
        cell.weight_hh.copy_(net.W_rec)
        cell.bias_ih.copy_(net.b_input)
        cell.bias_hh.copy_(net.b_rec)
    Wo = net.W_output.detach().clone().requires_grad_()
    bo = net.b_output.detach().clone().requires_grad_()
    h = torch.zeros(x.shape[0], net.n_hidden, dtype=torch.double)
    outs = []
    for t in range(x.shape[1]):
        h = cell(x[:, t], h)
        outs.append(h @ Wo.t() + bo)
    out = torch.stack(outs, 1)
    params = [cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh, Wo, bo]
    g = torch.autograd.grad(mse(out, y, mask), params)
    return out.detach(), dict(zip(['W_input', 'W_rec', 'b_input', 'b_rec', 'W_output', 'b_output'], g))


def rflo_oracle(net, x, y, mask):
    """What GRU-RFLO computes, written as autograd: h_{t-1} is DETACHED inside the
    gates (the dropped recurrent sensitivity) while the live h_{t-1} stays in the
    update-gate leak z * h_{t-1} (the kept per-unit path)."""
    ps = {k: v.detach().clone().requires_grad_() for k, v in net._trainable_params().items()}
    b_in = ps.get('b_input', net.b_input)
    b_rec = ps.get('b_rec', net.b_rec)
    b_out = ps.get('b_output', net.b_output)
    Hn = net.n_hidden
    h = torch.zeros(x.shape[0], Hn, dtype=torch.double)
    outs = []
    for t in range(x.shape[1]):
        hd = h.detach()
        gi = x[:, t] @ ps['W_input'].t() + b_in
        gh = hd @ ps['W_rec'].t() + b_rec
        r = torch.sigmoid(gi[:, :Hn] + gh[:, :Hn])
        z = torch.sigmoid(gi[:, Hn:2 * Hn] + gh[:, Hn:2 * Hn])
        n = torch.tanh(gi[:, 2 * Hn:] + r * gh[:, 2 * Hn:])
        h = (1 - z) * n + z * h
        outs.append(h @ ps['W_output'].t() + b_out)
    out = torch.stack(outs, 1)
    return dict(zip(ps, torch.autograd.grad(mse(out, y, mask), list(ps.values()))))


class TestGRUGradients(unittest.TestCase):
    def test_bptt_matches_torch_grucell_oracle(self):
        x, y, mask = data()
        net = make_net()
        out, ref = grucell_oracle(net, x, y, mask)
        got = net.bptt_gradients(x, y, mask)
        torch.testing.assert_close(got['outputs'], out)
        torch.testing.assert_close(net.forward_outputs(x), out)
        for k, v in ref.items():
            torch.testing.assert_close(got[k], v, rtol=1e-10, atol=1e-12)
        self.assertEqual(set(got) - {'loss', 'outputs'}, set(ref))

    def test_rflo_matches_detached_recurrence_oracle(self):
        x, y, mask = data()
        net = make_net(rule='local_diag_rflo')
        got = net.local_diag_rflo_gradients(x, y, mask)
        ref = rflo_oracle(net, x, y, mask)
        for k, v in ref.items():
            torch.testing.assert_close(got[k], v, rtol=1e-10, atol=1e-12)
        # An approximation: recurrent-block grads differ from BPTT; the readout's
        # are exact; outputs and loss are the same forward.
        bptt = net.bptt_gradients(x, y, mask)
        self.assertGreater(max((got[k] - bptt[k]).abs().max().item()
                               for k in ('W_input', 'W_rec', 'b_input', 'b_rec')), 1e-4)
        for k in ('W_output', 'b_output'):
            torch.testing.assert_close(got[k], bptt[k], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(got['outputs'], bptt['outputs'])
        torch.testing.assert_close(got['loss'], bptt['loss'])

    def test_rflo_exact_without_recurrence_and_at_one_step(self):
        x, y, mask = data()
        net = make_net()
        with torch.no_grad():
            net.W_rec.zero_()
        a, b = net.local_diag_rflo_gradients(x, y, mask), net.bptt_gradients(x, y, mask)
        for k in net._trainable_params():
            torch.testing.assert_close(a[k], b[k], rtol=1e-10, atol=1e-12)
        net = make_net()
        a = net.local_diag_rflo_gradients(x[:, :1], y[:, :1], mask[:, :1])
        b = net.bptt_gradients(x[:, :1], y[:, :1], mask[:, :1])
        for k in net._trainable_params():
            torch.testing.assert_close(a[k], b[k], rtol=1e-10, atol=1e-12)

    def test_feedback_modes_and_grad_writeback(self):
        x, y, mask = data()
        exact = make_net(rule='local_diag_rflo').local_diag_rflo_gradients(x, y, mask)
        for feedback in ('layerwise_fa', 'direct_fa'):
            net = make_net(rule='local_diag_rflo', feedback=feedback)
            self.assertTrue(hasattr(net, 'B_feedback'))
            got = net.sequence_gradients(x, y, mask)
            self.assertFalse(torch.allclose(got['W_rec'], exact['W_rec']))
            torch.testing.assert_close(got['W_output'], exact['W_output'])
            for name, p in net._trainable_params().items():
                torch.testing.assert_close(p.grad, got[name])
        with self.assertRaises(AssertionError):
            make_net(rule='local_direct')

    def test_update_masks_freeze_rows(self):
        x, y, mask = data()
        net = make_net(rule='local_diag_rflo')
        full = net.local_diag_rflo_gradients(x, y, mask)
        um1 = torch.ones(B, T, dtype=torch.double)
        same = net.local_diag_rflo_gradients(x, y, mask, update_masks=um1)
        for k in net._trainable_params():
            torch.testing.assert_close(same[k], full[k], rtol=0, atol=0)
        um = um1.clone()
        um[0, 2:] = 0          # row 0 frozen from t=2 on: state and traces stop advancing
        got = net.local_diag_rflo_gradients(x, y, mask, update_masks=um)
        for k in net._trainable_params():
            self.assertTrue(torch.isfinite(got[k]).all())
        self.assertFalse(torch.allclose(got['W_rec'], full['W_rec']))

    def test_no_bias_variant(self):
        x, y, mask = data()
        net = make_net(hidden_bias=False, output_bias=False)
        self.assertEqual(set(net._trainable_params()), {'W_input', 'W_rec', 'W_output'})
        got = net.local_diag_rflo_gradients(x, y, mask)
        ref = rflo_oracle(net, x, y, mask)
        for k in ref:
            torch.testing.assert_close(got[k], ref[k], rtol=1e-10, atol=1e-12)


class TestTrainGRU(unittest.TestCase):
    def test_cli_and_config(self):
        with patch.object(sys, 'argv', ['train_gru.py']):
            a = train_gru._parse_args()
        self.assertEqual(a.feedback, train_gru.FEEDBACK_MODE)
        with patch.object(sys, 'argv', ['train_gru.py', '--learning-signal', 'dfa', '--hidden', '7']):
            a = train_gru._parse_args()
        self.assertEqual((a.feedback, a.hidden), ('direct_fa', 7))
        cfg = train_gru._cfg()
        self.assertEqual((cfg.file_prefix, cfg.ckpt_prefix, cfg.signal_mode),
                         ('train_gru', 'gru', 'exact_spatial'))
        self.assertIn('GRU', cfg.title)
        _, train_params, net_params = train_gru.build_params()
        self.assertEqual(net_params['net_type'], 'gru')
        self.assertNotIn('alpha', net_params)
        self.assertEqual(train_params['reg_lambda'], 0.0)

    def test_run_seed_trains_both_rules_in_lockstep(self):
        x, y, mask = data(T=5)
        cfg = train_gru._cfg()
        cfg.rules_to_run = ['bptt', 'local_diag_rflo']
        cfg.n_datasets, cfg.batch, cfg.log_grad_align, cfg.save_nets = 3, B, True, False
        cfg.device, cfg.dtype = torch.device('cpu'), torch.double
        net_params = dict(n_neurons=[I, H, O], output_bias=False, hidden_bias=True,
                          learning_rule='bptt', feedback_mode='exact_spatial', loss_type='MSE')
        cfg.build_params = lambda: ({}, {}, copy.deepcopy(net_params))
        cfg.task = SimpleNamespace(init_params=lambda *p: p, valid_batch=lambda *a: (x, y, mask),
                                   train_batch=lambda *a: (x, y, mask),
                                   accuracy=lambda net, out, labels, m, inputs, isvalid=False:
                                       float(-mse(out, labels, m)), loss_and_grad=None)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            curves, align = train_common.run_seed(cfg, 5, [0, 2])
        text = log.getvalue()
        self.assertIn('BPTT', text)
        self.assertIn('RFLO', text)
        self.assertEqual(set(align), {'local_diag_rflo'})
        self.assertEqual(set(align['local_diag_rflo']), {'W_input', 'W_rec', 'W_output'})
        for values in curves['bptt'].values():
            self.assertTrue(np.isfinite(values).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
