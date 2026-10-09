"""Optional per-step MP-input RMS normalization (net_params['mp_input_norm']).

What is checked (float64, CPU, tight tolerances):
  * norm OFF is the previous computation: the key absent and 'none' build the same
    net (no extra buffers/state), and gradients are byte-identical for every rule.
  * the transpose-Jacobian helper matches autograd's VJP of x / rms(x).
  * norm ON, single MP layer: local_exact_rowlocal == BPTT for T > 1 (the layer's
    own eligibility is unaffected by the input form).
  * norm ON, deep stack, exact_spatial: the TOP MP layer's W/b == BPTT for T > 1;
    EVERY parameter (all MP layers + the trainable embedding, three_factor) == BPTT
    at T = 1, which isolates the same-time Jacobian incl. the norm at every
    boundary and the identity skip; with cross_layer_steps=1 all MP W/b == BPTT at
    T = 2 (the one-hop correction must map its sources and sweep through the norm
    at t-1). Covered with/without residual skips, scalar/matrix rates and hard bounds.
  * layerwise_fa / direct_fa run under the norm; direct_fa's boundary signals are
    unchanged by it; layerwise_fa's embedding signal equals the autograd VJP of the
    same B-matrix chain through the norm.
  * local_readout heads at T = 1 under the norm: each non-top module's W/b equal
    autograd of its OWN head loss, the top layer equals autograd of the main loss.
  * diag_mtrace / paired input modes are rejected with the norm on.
  * train_mpn CLI: --mp-input-norm / --mp-input-norm-eps validation, net_params
    wiring, and a checkpoint round-trip through the saved configuration.
"""
import _bootstrap  # noqa: F401

import copy
import json
import os
import tempfile
import unittest

import numpy as np
import torch

import mpn
import train_mpn


def _ml_params(bounds=False, matrix=False):
    return {'bias': True, 'mp_type': 'mult', 'm_update_type': 'hebb_assoc',
            'm_activation': 'linear', 'modulation_bounds': bounds,
            'm_bounds': (-0.5, 0.5),
            'eta_type': 'matrix' if matrix else 'scalar', 'eta_train': False,
            'lam_type': 'matrix' if matrix else 'scalar', 'lam_train': False,
            'm_time_scale': 400, 'W_freeze': False}


def build_deep(arch, *, norm='rms', residual=False, cross=0, rule='local_exact_rowlocal',
               feedback='exact_spatial', embed=True, input_mode='match',
               learning_signal='global', bounds=False, matrix=False, seed=0, eps=1e-5):
    torch.manual_seed(seed)
    npar = {'n_neurons': arch, 'loss_type': 'MSE', 'activation': 'tanh',
            'output_bias': True, 'output_matrix': '', 'dt': 40,
            'feedback_mode': feedback, 'mp_residual': residual,
            'cross_layer_steps': cross, 'learning_rule': rule,
            'input_mode': input_mode, 'learning_signal': learning_signal,
            'ml_params': _ml_params(bounds, matrix)}
    if norm is not None:
        npar['mp_input_norm'] = norm
        npar['mp_input_norm_eps'] = eps
    if embed:
        npar.update({'linear_embed': arch[1], 'input_layer_add': True,
                     'input_layer_add_trainable': True, 'input_layer_bias': True,
                     'input_init_type': 'xavier'})
    net = mpn.DeepMultiPlasticNet(npar, verbose=False).double()
    with torch.no_grad():
        for j, m in enumerate(net.mp_layers):
            if matrix:
                m.eta.copy_(0.6 + 0.3 * torch.rand_like(m.eta))
                m.lam.copy_(0.5 * m.lam_clamp + 0.3 * m.lam_clamp * torch.rand_like(m.lam))
            else:
                # Large eta: with ||x_hat||^2 = d the write is O(1), so the norm's
                # Jacobian terms are far from negligible in every check below.
                m.eta.fill_(0.8 + 0.1 * j)
                m.lam.fill_(0.6 * m.lam_clamp)
    return net


def build_single(n_in, n_hid, n_out, *, norm='rms', rule='local_exact_rowlocal', seed=0):
    torch.manual_seed(seed)
    npar = {'n_neurons': [n_in, n_hid, n_out], 'loss_type': 'MSE', 'activation': 'tanh',
            'output_bias': True, 'output_matrix': '', 'dt': 40,
            'learning_rule': rule, 'feedback_mode': 'exact_spatial',
            'ml_params': _ml_params()}
    if norm is not None:
        npar['mp_input_norm'] = norm
    net = mpn.MultiPlasticNet(npar, verbose=False).double()
    with torch.no_grad():
        net.mp_layer.eta.fill_(0.7)
        net.mp_layer.lam.fill_(0.6 * net.mp_layer.lam_clamp)
    return net


def make_data(B, T, n_in, n_out, seed=1, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    # Per-sample input magnitudes spread over an order of magnitude, so the norm
    # actually changes the trajectory (and r differs across the batch).
    amp = (0.3 + scale * torch.rand(B, 1, 1, generator=g)).double()
    inputs = amp * torch.randn(B, T, n_in, generator=g).double()
    labels = torch.randn(B, T, n_out, generator=g).double()
    masks = torch.rand(B, T, n_out, generator=g).double()
    return inputs, labels, masks


def mp_keys(net, bias=True):
    ks = []
    for n in range(len(net.mp_layers)):
        sfx = '' if n == 0 else str(n)
        ks.append(f'W{sfx}')
        if bias:
            ks.append(f'b{sfx}')
    return ks


def rel(a, b):
    return ((a - b).norm() / (b.norm() + 1e-300)).item()


class TestNormHelpers(unittest.TestCase):
    def test_absent_key_is_none_and_adds_no_state(self):
        net_a = build_deep([5, 6, 6, 3], norm=None, seed=2)
        net_b = build_deep([5, 6, 6, 3], norm='none', seed=2)
        self.assertEqual(net_a.mp_input_norm, 'none')
        self.assertEqual(net_b.mp_input_norm, 'none')
        self.assertFalse(net_a._mp_norm_active)
        self.assertEqual(list(net_a.state_dict().keys()), list(net_b.state_dict().keys()))
        net_c = build_deep([5, 6, 6, 3], norm='rms', seed=2)
        # The norm is parameter-free and stateless: no new buffers/params, so a
        # norm-off state_dict loads into a norm-on net (and vice versa) unchanged.
        self.assertEqual(list(net_a.state_dict().keys()), list(net_c.state_dict().keys()))
        net_c.load_state_dict(net_a.state_dict())
        for k in net_a.state_dict():
            self.assertTrue(torch.equal(net_a.state_dict()[k], net_c.state_dict()[k]))

    def test_invalid_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "mp_input_norm"):
            build_deep([5, 6, 3], norm='layer')
        with self.assertRaisesRegex(ValueError, "mp_input_norm_eps"):
            build_deep([5, 6, 3], norm='rms', eps=0.0)

    def test_forward_and_transpose_jacobian_match_autograd(self):
        net = build_deep([5, 7, 3], norm='rms', seed=3)
        x = (2.0 * torch.randn(4, 7, dtype=torch.float64)).requires_grad_(True)
        x_hat, r = net._mp_input_norm_forward(x)
        self.assertEqual(r.shape, (4, 1))
        d = x.shape[1]
        torch.testing.assert_close(x_hat.square().sum(-1).detach(),
                                   torch.full((4,), float(d), dtype=torch.float64),
                                   rtol=0, atol=d * 1e-4)   # eps shifts it slightly
        g = torch.randn(4, 7, dtype=torch.float64)
        vjp, = torch.autograd.grad(x_hat, x, grad_outputs=g)
        ours = net._mp_input_norm_backward(g, x_hat.detach(), r.detach())
        torch.testing.assert_close(ours, vjp, rtol=1e-12, atol=1e-12)
        # Off: identity, no copy.
        net.mp_input_norm = 'none'
        same, r0 = net._mp_input_norm_forward(x)
        self.assertIs(same, x)
        self.assertIsNone(r0)
        self.assertIs(net._mp_input_norm_backward(g, None, None), g)

    def test_norm_off_is_byte_identical_for_every_rule(self):
        inp, lab, msk = make_data(3, 5, 5, 3)
        for rule in ('bptt', 'local_exact_rowlocal', 'local_diag_rflo', 'local_direct'):
            ref = build_deep([5, 6, 6, 3], norm=None, rule=rule, residual=True, seed=4)
            new = build_deep([5, 6, 6, 3], norm='none', rule=rule, residual=True, seed=4)
            new.load_state_dict(ref.state_dict())
            ga = ref.sequence_gradients(inp, lab, msk)
            gb = new.sequence_gradients(inp, lab, msk)
            for k in ga:
                if k == 'outputs' or ga[k] is None:
                    continue
                self.assertTrue(torch.equal(ga[k], gb[k]), f'{rule}:{k}')

    def test_norm_changes_the_trajectory(self):
        inp, lab, msk = make_data(3, 4, 5, 3)
        a = build_deep([5, 6, 3], norm='none', rule='bptt', seed=5)
        b = build_deep([5, 6, 3], norm='rms', rule='bptt', seed=5)
        b.load_state_dict(a.state_dict())
        ga, gb = a.sequence_gradients(inp, lab, msk), b.sequence_gradients(inp, lab, msk)
        self.assertGreater(rel(ga['W'], gb['W']), 1e-3)
        self.assertFalse(torch.allclose(ga['outputs'], gb['outputs']))


class TestSingleLayer(unittest.TestCase):
    def test_exact_rowlocal_matches_bptt_under_norm(self):
        inp, lab, msk = make_data(4, 7, 5, 3)
        for bptt_net, loc in [(build_single(5, 6, 3, rule='bptt', seed=6),
                               build_single(5, 6, 3, rule='local_exact_rowlocal', seed=6))]:
            loc.load_state_dict(bptt_net.state_dict())
            gb = bptt_net.sequence_gradients(inp, lab, msk)
            gl = loc.sequence_gradients(inp, lab, msk)
            for k in ('W', 'b', 'W_output', 'b_output'):
                torch.testing.assert_close(gl[k], gb[k], rtol=1e-10, atol=1e-12, msg=k)
            torch.testing.assert_close(gl['loss'], gb['loss'], rtol=1e-12, atol=1e-12)

    def test_eval_forward_consumes_normalized_input(self):
        net = build_single(5, 6, 3, rule='bptt', seed=6)
        inp, _, _ = make_data(2, 3, 5, 3)
        net.reset_state(B=2)
        x0 = inp[:, 0, :]
        out, _ = net.network_step(x0)
        x_hat = x0 / torch.sqrt(x0.square().mean(-1, keepdim=True) + net.mp_input_norm_eps)
        # M after one step is eta * h * x_hat^T (M_0 = 0, lam irrelevant at t=1).
        net2 = build_single(5, 6, 3, rule='bptt', seed=6); net2.load_state_dict(net.state_dict())
        net2.reset_state(B=2)
        z, _ = net2.mp_layer(x_hat)
        h = torch.tanh(z)
        expect = net.mp_layer.eta * h.unsqueeze(-1) * x_hat.unsqueeze(1)
        torch.testing.assert_close(net.mp_layer.M, expect, rtol=1e-12, atol=1e-12)


class TestDeepExactness(unittest.TestCase):
    def _pair(self, arch, **kw):
        kw_b = dict(kw); kw_b['rule'] = 'bptt'
        b = build_deep(arch, **kw_b)
        l = build_deep(arch, **kw)
        l.load_state_dict(b.state_dict())
        return b, l

    def test_top_layer_exact_for_long_sequences(self):
        for residual, bounds, matrix in [(False, False, False), (True, False, True),
                                         (True, True, False), (False, True, True)]:
            b, l = self._pair([5, 6, 6, 6, 3], residual=residual, bounds=bounds,
                              matrix=matrix, seed=7, norm='rms')
            inp, lab, msk = make_data(4, 6, 5, 3, seed=residual + 2 * bounds)
            gb, gl = b.sequence_gradients(inp, lab, msk), l.sequence_gradients(inp, lab, msk)
            top = len(l.mp_layers) - 1
            for k in (f'W{top}', f'b{top}', 'W_output', 'b_output'):
                torch.testing.assert_close(gl[k], gb[k], rtol=1e-9, atol=1e-11,
                                           msg=f'{k} res={residual} bounds={bounds}')
            # Lower layers are surrogates: they must differ (sanity that the test bites).
            self.assertGreater(rel(gl['W'], gb['W']), 1e-4)

    def test_all_params_exact_at_T1(self):
        """At T=1 there are no temporal paths at all, so every local rule with
        exact_spatial feedback must reproduce BPTT for EVERY parameter, including the
        embedding (three_factor) — this isolates the same-time spatial Jacobian with
        the norm at every boundary (and the pre-norm residual skip)."""
        for rule in ('local_exact_rowlocal', 'local_diag_rflo', 'local_direct'):
            for residual in (False, True):
                b, l = self._pair([5, 6, 6, 6, 3], rule=rule, residual=residual,
                                  seed=8, norm='rms')
                inp, lab, msk = make_data(5, 1, 5, 3, seed=11)
                gb, gl = b.sequence_gradients(inp, lab, msk), l.sequence_gradients(inp, lab, msk)
                for k in mp_keys(l) + ['W_in', 'b_in', 'W_output', 'b_output']:
                    torch.testing.assert_close(gl[k], gb[k], rtol=1e-9, atol=1e-11,
                                               msg=f'{rule} {k} res={residual}')

    def test_cross_layer_one_hop_exact_at_T2(self):
        for residual, bounds, matrix in [(False, False, False), (True, False, False),
                                         (False, True, True), (True, True, True)]:
            b, l = self._pair([5, 6, 6, 6, 3], residual=residual, bounds=bounds, matrix=matrix,
                              cross=1, embed=False, seed=9, norm='rms')
            inp, lab, msk = make_data(4, 2, 5, 3, seed=13)
            gb, gl = b.sequence_gradients(inp, lab, msk), l.sequence_gradients(inp, lab, msk)
            for k in mp_keys(l) + ['W_output', 'b_output']:
                torch.testing.assert_close(gl[k], gb[k], rtol=1e-9, atol=1e-11,
                                           msg=f'{k} res={residual} bounds={bounds} mat={matrix}')
            # Without the correction the lower layers are NOT exact at T=2.
            l0 = build_deep([5, 6, 6, 6, 3], residual=residual, bounds=bounds, matrix=matrix,
                            cross=0, embed=False, seed=9, norm='rms')
            l0.load_state_dict(b.state_dict())
            g0 = l0.sequence_gradients(inp, lab, msk)
            self.assertGreater(rel(g0['W'], gb['W']), 1e-5)

    def test_cross_layer_T3_improves_lower_layers(self):
        b, l1 = self._pair([5, 6, 6, 3], cross=1, embed=False, seed=10, norm='rms')
        l0 = build_deep([5, 6, 6, 3], cross=0, embed=False, seed=10, norm='rms')
        l0.load_state_dict(b.state_dict())
        inp, lab, msk = make_data(4, 3, 5, 3, seed=17)
        gb = b.sequence_gradients(inp, lab, msk)
        g1, g0 = l1.sequence_gradients(inp, lab, msk), l0.sequence_gradients(inp, lab, msk)
        self.assertLess(rel(g1['W'], gb['W']), rel(g0['W'], gb['W']))


class TestFeedbackModesAndHeads(unittest.TestCase):
    def test_direct_fa_boundary_signals_unaffected(self):
        net = build_deep([5, 6, 6, 3], feedback='direct_fa', seed=12, norm='rms')
        inp, _, _ = make_data(3, 1, 5, 3)
        net.reset_state(B=3)
        out, h, z, phi, _, blocks, norm = net._forward_local_stack(inp[:, 0], return_blocks=True,
                                                                   return_norm=True)
        go = torch.randn(3, 3, dtype=torch.float64)
        with_norm = net._same_time_boundary_signals(go, phi, True, norm=norm)
        without = net._same_time_boundary_signals(go, phi, True, norm=[(None, None)] * 2)
        for a, b_ in zip(with_norm, without):
            self.assertTrue(torch.equal(a, b_))
        g = net.sequence_gradients(inp, torch.randn(3, 1, 3, dtype=torch.float64),
                                   torch.ones(3, 1, 3, dtype=torch.float64))
        self.assertTrue(all(torch.isfinite(v).all() for k, v in g.items() if torch.is_tensor(v)))

    def test_layerwise_fa_signal_passes_through_norm_jacobian(self):
        net = build_deep([5, 6, 6, 3], feedback='layerwise_fa', residual=True, seed=14, norm='rms')
        inp, _, _ = make_data(3, 1, 5, 3)
        x = inp[:, 0]
        net.reset_state(B=3)
        out, h, z, phi, _, blocks, norm = net._forward_local_stack(x, return_blocks=True,
                                                                   return_norm=True)
        go = torch.randn(3, 3, dtype=torch.float64)
        ell = net._same_time_boundary_signals(go, phi, True, norm=norm)
        # Oracle: the same FA chain, ell_h[n] = J_n^T(delta_n @ B_inter[n]) + skip *
        # ell_h[n+1], with J_n^T taken from autograd's VJP of the norm at h[n].
        expect = [None] * (len(net.mp_layers) + 1)
        expect[-1] = go @ net.B_feedback
        for n in range(len(net.mp_layers) - 1, -1, -1):
            hn = h[n].detach().requires_grad_(True)
            x_hat, _ = net._mp_input_norm_forward(hn)
            delta = net._branch_scales[n] * expect[n + 1] * phi[n]
            at_xhat = delta @ getattr(net, net._B_inter_names[n])
            vjp, = torch.autograd.grad(x_hat, hn, grad_outputs=at_xhat)
            expect[n] = vjp + (expect[n + 1] if net._residual_at[n] else 0.0)
        for n in range(len(expect)):
            torch.testing.assert_close(ell[n], expect[n], rtol=1e-12, atol=1e-12, msg=str(n))

    def test_local_readout_heads_T1_match_autograd_of_own_head_loss(self):
        net = build_deep([5, 6, 6, 6, 3], learning_signal='local_readout', seed=15, norm='rms')
        inp, lab, msk = make_data(4, 1, 5, 3, seed=19)
        g = net.sequence_gradients(inp, lab, msk)
        L = len(net.mp_layers)
        # Autograd: forward once with M_0 (T=1), per-head masked-MSE losses.
        net.reset_state(B=4)
        u = net._standardize_input(inputs=inp)[:, 0]
        out, h, z, phi, embed_pre, blocks, norm = net._forward_local_stack(u, return_blocks=True,
                                                                           return_norm=True)
        N = 4 * 1 * 3
        def mse(o):
            d = msk[:, 0] * o - msk[:, 0] * lab[:, 0]
            return (d * d).sum() / N
        main = mse(out)
        heads = [mse(q) for q in net._head_outputs(h)]
        for n in range(L):
            sfx = '' if n == 0 else str(n)
            W, b = net.mp_layers[n].W, net.mp_layers[n].b
            loss = main if n == L - 1 else heads[n]
            gW, gb = torch.autograd.grad(loss, [W, b], retain_graph=True)
            torch.testing.assert_close(g[f'W{sfx}'], gW, rtol=1e-9, atol=1e-11, msg=f'W{sfx}')
            torch.testing.assert_close(g[f'b{sfx}'], gb, rtol=1e-9, atol=1e-11, msg=f'b{sfx}')
        # Embedding follows module 0's head through layer 0's backprojection + norm.
        gWin, = torch.autograd.grad(heads[0], [net.W_initial_linear.weight], retain_graph=True)
        torch.testing.assert_close(g['W_in'], gWin, rtol=1e-9, atol=1e-11)

    def test_diag_mtrace_rejected_with_norm(self):
        with self.assertRaisesRegex(ValueError, "diag_mtrace"):
            build_deep([5, 6, 3], input_mode='diag_mtrace', norm='rms')
        with self.assertRaisesRegex(ValueError, "paired"):
            build_deep([5, 6, 3], input_mode='paired', rule='local_diag_rflo', norm='rms')
        # Still allowed with the norm off.
        build_deep([5, 6, 3], input_mode='diag_mtrace', norm='none')


class TestTrainingScriptEval(unittest.TestCase):
    """train_mpn.forward_outputs (the held-out validation forward) must run the SAME
    network the gradient paths trained: with the norm on, both net types consume
    x_hat there too. Regression: the single-layer branch drove mp_layer on the raw
    standardized input, so validation evaluated a different network than training."""

    def test_single_layer_eval_matches_training_forward(self):
        inp, lab, msk = make_data(4, 5, 5, 3)
        b = build_single(5, 6, 3, rule='bptt', seed=6)
        torch.testing.assert_close(train_mpn.forward_outputs(b, inp),
                                   b.bptt_gradients(inp, lab, msk)['outputs'],
                                   rtol=1e-12, atol=1e-12)
        loc = build_single(5, 6, 3, rule='local_exact_rowlocal', seed=6)
        loc.load_state_dict(b.state_dict())
        torch.testing.assert_close(train_mpn.forward_outputs(loc, inp),
                                   loc.sequence_gradients(inp, lab, msk)['outputs'],
                                   rtol=1e-12, atol=1e-12)
        # Sanity: the un-normalized forward is a different network.
        b.mp_input_norm = 'none'
        self.assertGreater(rel(train_mpn.forward_outputs(b, inp),
                               train_mpn.forward_outputs(loc, inp)), 1e-2)

    def test_deep_eval_matches_training_forward(self):
        inp, lab, msk = make_data(3, 4, 5, 3)
        for residual in (False, True):
            b = build_deep([5, 6, 6, 3], rule='bptt', residual=residual, seed=7)
            torch.testing.assert_close(train_mpn.forward_outputs(b, inp),
                                       b.bptt_gradients(inp, lab, msk)['outputs'],
                                       rtol=1e-12, atol=1e-12, msg=f'res={residual}')


class TestCLI(unittest.TestCase):
    def test_parser_and_params(self):
        args = train_mpn._parse_args(['--mp-input-norm', 'rms', '--mp-input-norm-eps', '1e-6'])
        self.assertEqual(args.mp_input_norm, 'rms')
        self.assertEqual(args.mp_input_norm_eps, 1e-6)
        self.assertEqual(train_mpn._parse_args([]).mp_input_norm, 'none')
        with self.assertRaises(SystemExit):
            train_mpn._parse_args(['--mp-input-norm', 'layer'])
        with self.assertRaises(SystemExit):
            train_mpn._parse_args(['--mp-input-norm', 'rms', '--mp-input-norm-eps', '0'])
        with self.assertRaises(SystemExit):
            train_mpn._parse_args(['--mp-input-norm', 'rms', '--input-mode', 'diag_mtrace'])
        saved = train_mpn.MP_INPUT_NORM, train_mpn.MP_INPUT_NORM_EPS
        try:
            train_mpn.MP_INPUT_NORM, train_mpn.MP_INPUT_NORM_EPS = 'rms', 1e-6
            _, _, net_params = train_mpn.build_params()
            self.assertEqual(net_params['mp_input_norm'], 'rms')
            self.assertEqual(net_params['mp_input_norm_eps'], 1e-6)
            train_mpn.MP_INPUT_NORM = 'none'
            _, _, net_params = train_mpn.build_params()
            self.assertEqual(net_params['mp_input_norm'], 'none')
        finally:
            train_mpn.MP_INPUT_NORM, train_mpn.MP_INPUT_NORM_EPS = saved

    def test_config_round_trip(self):
        net = build_deep([5, 6, 6, 3], residual=True, seed=21, norm='rms', eps=2e-5)
        cfg = {'mp_input_norm': net.mp_input_norm, 'mp_input_norm_eps': net.mp_input_norm_eps}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'config.json')
            with open(path, 'w') as f:
                json.dump(cfg, f)
            with open(path) as f:
                loaded = json.load(f)
        again = build_deep([5, 6, 6, 3], residual=True, seed=21, norm=loaded['mp_input_norm'],
                           eps=loaded['mp_input_norm_eps'])
        again.load_state_dict(net.state_dict())
        inp, lab, msk = make_data(3, 3, 5, 3)
        ga, gb = net.sequence_gradients(inp, lab, msk), again.sequence_gradients(inp, lab, msk)
        self.assertTrue(torch.equal(ga['W'], gb['W']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
