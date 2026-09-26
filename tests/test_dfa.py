"""Independent autograd oracles for local MPN+DFA (small CPU double tensors).
Run: python -m unittest discover -s tests -v
"""
import copy
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
sys.path.insert(0, str(ROOT / 'scripts'))
import mpn


def make_net(rule='local_exact_rowlocal', bias='exact', residual=False, kinds=None):
    np.random.seed(37)
    torch.manual_seed(37)
    cfg = dict(n_neurons=[2, 3, 3, 4, 2], dt=1, activation='tanh',
               output_matrix='', output_bias=True, input_layer_add=True,
               input_layer_add_trainable=True, input_layer_bias=True, linear_embed=3,
               learning_rule=rule, feedback_mode='direct_fa', input_mode='match',
               cross_layer_steps=0, mp_residual=residual,
               ml_params=dict(bias=True, mp_type='mult', m_update_type='hebb_assoc',
                              m_activation='linear', modulation_bounds=False,
                              eta_type='scalar', eta_train=False, lam_type='scalar',
                              lam_train=False, m_time_scale=10, local_bias_mode=bias))
    with contextlib.redirect_stdout(io.StringIO()):
        net = mpn.DeepMultiPlasticNet(cfg, verbose=False).double()
    with torch.no_grad():
        for n, layer in enumerate(net.mp_layers):
            layer.eta.fill_(.18)
            layer.lam.fill_(.67)
            layer.M_init.normal_(0, .04)
            if kinds:
                layer.m_update_type = kinds[n]
    return net


def data():
    torch.manual_seed(11)
    x = torch.randn(2, 4, 2, dtype=torch.double) * .25
    y = torch.randn(2, 4, 2, dtype=torch.double) * .2
    masks = torch.rand_like(y)
    masks[0, 0] = 0
    um = torch.tensor([[1., 1., 0., 1.], [1., 0., 1., 1.]], dtype=torch.double)
    return x, y, masks, um


def write(layer, M, x, a, active):
    eta, lam = layer._eta_lam_full()
    post = a if layer.m_update_type == 'hebb_assoc' else torch.ones_like(a) / a.shape[-1] ** .5
    new = lam * M + eta * post.unsqueeze(-1) * x.unsqueeze(1)
    return active[:, None, None] * new + (1 - active[:, None, None]) * M


@torch.no_grad()
def independent_forward(net, x, um):
    """No calls to the production forward, eligibility or signal routines."""
    layers = net.mp_layers
    M = [l.M_init[None].expand(len(x), -1, -1).clone() for l in layers]
    inputs = [[] for _ in layers]
    olds = [[] for _ in layers]
    top = []
    out = []
    for t in range(x.shape[1]):
        h = torch.tanh(torch.nn.functional.linear(x[:, t], net.W_initial_linear.weight,
                                                  net.W_initial_linear.bias))
        for n, layer in enumerate(layers):
            inputs[n].append(h.clone())
            olds[n].append(M[n].clone())
            a = torch.tanh((layer.W * (1 + M[n]) * h[:, None]).sum(-1) + layer.b)
            M[n] = write(layer, M[n], h, a, um[:, t])
            h = a + h if net._residual_at[n] else a
        top.append(h)
        out.append(torch.nn.functional.linear(h, net.W_output, net.b_output))
    return torch.stack(out, 1), inputs, olds, torch.stack(top, 1)


def oracle(net, x, y, mask, um, diag):
    """Row oracle: differentiate each isolated row-layer trajectory.
    Diagonal oracle: replay each scalar synapse with all OTHER modulation
    histories detached. This truncates graph paths, without coding an A recursion.
    """
    outputs, xs, olds, top = independent_forward(net, x, um)
    go = 2 * mask.square() * (outputs - y) / outputs.numel()
    result = {'W_output': torch.einsum('bta,bti->ai', go, top),
              'b_output': go.sum((0, 1))}
    for n, layer in enumerate(net.mp_layers):
        ell = go @ getattr(net, net._B_direct_names[n+1])
        suffix = '' if n == 0 else str(n)
        if not diag:
            W = layer.W.detach().clone().requires_grad_()
            b = layer.b.detach().clone().requires_grad_()
            M = layer.M_init[None].expand(len(x), -1, -1).clone()
            objective = 0
            for t in range(x.shape[1]):
                a = torch.tanh((W * (1 + M) * xs[n][t][:, None]).sum(-1) + b)
                objective = objective + (ell[:, t] * a).sum()
                M = write(layer, M, xs[n][t], a, um[:, t])
            gw, gb = torch.autograd.grad(objective, (W, b))
        else:
            gw = torch.zeros_like(layer.W)
            eta, lam = layer._eta_lam_full()
            for i in range(layer.n_output):
                for j in range(layer.n_input):
                    w = layer.W[i, j].detach().clone().requires_grad_()
                    m = layer.M_init[i, j].expand(len(x)).clone()
                    objective = 0
                    keep = torch.ones(layer.n_input, dtype=torch.double)
                    keep[j] = 0
                    for t in range(x.shape[1]):
                        inp = xs[n][t]
                        other = (keep * layer.W[i].detach() * (1 + olds[n][t][:, i]) * inp).sum(-1)
                        a = torch.tanh(other + w * (1 + m) * inp[:, j] + layer.b[i].detach())
                        objective = objective + (ell[:, t, i] * a).sum()
                        post = a if layer.m_update_type == 'hebb_assoc' else torch.ones_like(a) / layer.n_output ** .5
                        new = lam[i, j] * m + eta[i, j] * post * inp[:, j]
                        m = um[:, t] * new + (1 - um[:, t]) * m
                    gw[i, j] = torch.autograd.grad(objective, w)[0]
            # Compute exact bias via an independent own-layer graph when requested.
            b = layer.b.detach().clone().requires_grad_()
            M = layer.M_init[None].expand(len(x), -1, -1).clone()
            objective = 0
            for t in range(x.shape[1]):
                a = torch.tanh((layer.W.detach() * (1+M) * xs[n][t][:, None]).sum(-1)+b)
                objective = objective + (ell[:, t] * a).sum()
                M = write(layer, M, xs[n][t], a, um[:, t])
            gb = torch.autograd.grad(objective, b)[0]
        if layer.local_bias_mode == 'direct':
            b = layer.b.detach().clone().requires_grad_()
            objective = sum((ell[:, t] * torch.tanh(
                (layer.W.detach() * (1+olds[n][t]) * xs[n][t][:, None]).sum(-1)+b)).sum()
                for t in range(x.shape[1]))
            gb = torch.autograd.grad(objective, b)[0]
        result['W'+suffix], result['b'+suffix] = gw, gb
    U = net.W_initial_linear.weight.detach().clone().requires_grad_()
    b = net.W_initial_linear.bias.detach().clone().requires_grad_()
    emb = torch.tanh(torch.nn.functional.linear(x, U, b))
    ell0 = go @ getattr(net, net._B_direct_names[0])
    result['W_in'], result['b_in'] = torch.autograd.grad((ell0*emb).sum(), (U, b))
    return outputs, result


class TestDFA(unittest.TestCase):
    def test_both_rules_against_independent_autograd(self):
        x, y, masks, um = data()
        for diag in (False, True):
            for bias in ('exact', 'direct'):
                for residual in (False, True):
                    for kinds in (None, ['hebb_pre', 'hebb_assoc', 'hebb_pre'], ['hebb_pre']*3):
                        with self.subTest(diag=diag, bias=bias, residual=residual, kinds=kinds):
                            rule = 'local_diag_rflo' if diag else 'local_exact_rowlocal'
                            net = make_net(rule, bias, residual, kinds)
                            expected_out, expected = oracle(net, x, y, masks, um, diag)
                            # No hidden BPTT/autograd is allowed in either local DFA run.
                            with patch.object(net, 'bptt_gradients', side_effect=AssertionError('BPTT called')):
                                actual = net.sequence_gradients(x, y, masks, update_masks=um)
                            torch.testing.assert_close(actual['outputs'], expected_out, rtol=1e-10, atol=1e-11)
                            for key, val in expected.items():
                                torch.testing.assert_close(actual[key], val, rtol=1e-9, atol=1e-10)
                            if bias == 'direct' and kinds != ['hebb_pre']*3:
                                self.assertTrue(all(l.Q is None for l in net.mp_layers))

    def test_feedback_is_independent_fixed_and_serialized(self):
        net = make_net()
        signal = torch.randn(2, 2, dtype=torch.double)
        phi = [torch.randn(2, l.n_output, dtype=torch.double) for l in net.mp_layers]
        expected = [signal @ getattr(net, name) for name in net._B_direct_names]
        with torch.no_grad():
            for l in net.mp_layers:
                l.reset_state(B=2)
                l.W.add_(1)
                l.M.add_(2)
            net.W_output.add_(3)
        actual = net._same_time_boundary_signals(signal, phi, True)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e)
        param_names = dict(net.named_parameters())
        before = {k: v.clone() for k, v in net.named_buffers() if k.startswith('B_direct')}
        self.assertTrue(before)
        self.assertFalse(set(before) & set(param_names))
        x, y, masks, _ = data()
        optimizer = torch.optim.SGD(net.parameters(), lr=.001)
        net.sequence_gradients(x, y, masks)
        optimizer.step()
        for name, tensor in before.items():
            torch.testing.assert_close(getattr(net, name), tensor, rtol=0, atol=0)
        other = make_net()
        other.load_state_dict(net.state_dict())
        for name, tensor in before.items():
            torch.testing.assert_close(getattr(other, name), tensor, rtol=0, atol=0)

    def test_explicit_and_fused_trace_paths(self):
        for mode in ('exact', 'diag'):
            for bias in ('exact', 'direct'):
                a = make_net(bias=bias).mp_layers[0]
                b = copy.deepcopy(a)
                for layer in (a, b):
                    layer.reset_state(B=2)
                    (layer.reset_local_learning_state if mode=='exact' else layer.reset_diag_rflo_state)(B=2)
                for t in range(3):
                    x = torch.randn(2, a.n_input, dtype=torch.double)*.2
                    f = torch.rand(2, a.n_output, dtype=torch.double)
                    ell = torch.randn_like(f)
                    um = torch.tensor([1., t % 2], dtype=torch.double)
                    eta, lam = a._eta_lam_full()
                    g, h = a.step_fn_for(mode)(x, f, ell, eta, lam, um)
                    E, R = (b.compute_exact_rowlocal_eligibility if mode=='exact' else b.compute_diag_rflo_eligibility)(x, f)
                    torch.testing.assert_close(g, torch.einsum('bi,bij->ij', ell, E))
                    torch.testing.assert_close(h, (ell*R).sum(0))
                    (b.update_exact_rowlocal_traces if mode=='exact' else b.update_diag_rflo_traces)(x, E, R, um)
                    torch.testing.assert_close(getattr(a, 'P' if mode=='exact' else 'A'), getattr(b, 'P' if mode=='exact' else 'A'))
                    if bias=='exact': torch.testing.assert_close(a.Q, b.Q)
                    else: self.assertIsNone(a.Q)

    def test_preset_and_conflicts(self):
        import train_mpn
        with patch.object(sys, 'argv', ['train_mpn.py', '--dfa']):
            args = train_mpn._parse_args()
        self.assertEqual((args.feedback,args.input_mode,args.cross_layer_steps,args.local_bias_mode),
                         ('direct_fa','match',0,'direct'))
        self.assertEqual(args.rules, ['bptt','local_exact_rowlocal','local_diag_rflo'])
        self.assertFalse(args.grad_align)
        for conflicting in (['--input-mode','exact'], ['--cross-layer-steps','1'], ['--feedback','exact_spatial']):
            with self.subTest(conflicting=conflicting), patch.object(sys,'argv',['train_mpn.py','--dfa',*conflicting]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): train_mpn._parse_args()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
