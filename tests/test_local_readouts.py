"""Local readout heads (learning_signal = 'local_readout' / 'mixed') on the deep MPN.

Independent autograd oracles (no production signal/eligibility routines), locality,
limiting cases, guards, and the runner/CLI/checkpoint surface. Small CPU double
tensors. Run from tests/:  python -m unittest test_local_readouts -v
"""
import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401  (core/ + scripts/ on sys.path)
import mpn
import train_common
import train_mpn
from test_dfa import write   # the defining masked plastic-state recurrence


# ─── Fixtures ─────────────────────────────────────────────────────────────────
def make_cfg(widths=(4, 4, 4), signal='global', rule='local_direct', residual=True,
             input_mode='match', cross=0, feedback='exact_spatial', output_bias=True,
             bias='exact', alpha=1.0, head_seed=None, include_signal_key=True,
             residual_scale=1.0):
    cfg = dict(net_type='dmpn', n_neurons=[3, *widths, 2], dt=1, activation='tanh', output_matrix='',
               output_bias=output_bias, input_layer_add=True, input_layer_add_trainable=True,
               input_layer_bias=True, linear_embed=widths[0], learning_rule=rule,
               feedback_mode=feedback, input_mode=input_mode, cross_layer_steps=cross,
               mp_residual=residual, residual_scale=residual_scale, local_signal_alpha=alpha,
               ml_params=dict(bias=True, mp_type='mult', m_update_type='hebb_assoc',
                              m_activation='linear', m_scale=1.0, modulation_bounds=True,
                              m_bounds=(-1.0, 1.0), eta_type='scalar', eta_train=False,
                              lam_type='scalar', lam_train=False, m_time_scale=10,
                              local_bias_mode=bias))
    if include_signal_key:
        cfg['learning_signal'] = signal
    if head_seed is not None:
        cfg['local_head_seed'] = head_seed
    return cfg


def make_net(cfg, seed=5):
    np.random.seed(seed)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        net = mpn.DeepMultiPlasticNet(copy.deepcopy(cfg), verbose=False).double()
    with torch.no_grad():
        for layer in net.mp_layers:
            layer.eta.fill_(.3)
            layer.lam.fill_(.7)
            layer.M_init.normal_(0, .05)
    return net


def data(B=3, T=5, n_in=3, n_out=2, seed=11):
    torch.manual_seed(seed)
    x = torch.randn(B, T, n_in, dtype=torch.double) * .5
    y = torch.randn(B, T, n_out, dtype=torch.double) * .3
    mask = torch.rand(B, T, n_out, dtype=torch.double)
    mask[0, 0] = 0
    um = (torch.rand(B, T, dtype=torch.double) > .3).double()
    um[:, 0] = 1
    return x, y, mask, um


def mse_loss_and_grad(q, y, mask):
    """Same contract as mpn.masked_mse_loss_and_output_grad, written independently."""
    N = q.numel()
    loss = ((mask * q - mask * y) ** 2).sum() / N
    return loss, 2.0 / N * mask * (mask * q - mask * y)


# ─── Independent forward / oracles ────────────────────────────────────────────
@torch.no_grad()
def independent_forward(net, x, um=None):
    """Replay the deep MPN with heads from its defining equations — no production
    forward/signal/eligibility routine. Returns per-step lists: streams h[t][0..L],
    frozen modulations M_prev[t][n] (the M each layer CONSUMED at t), the main
    output sequence and each head's prediction sequence."""
    layers = net.mp_layers
    B, T, _ = x.shape
    M = [l.M_init[None].expand(B, -1, -1).clone() for l in layers]
    streams, M_prev, outs = [], [], []
    heads = [[] for _ in net._head_names]
    for t in range(T):
        h = [torch.tanh(F.linear(x[:, t], net.W_initial_linear.weight, net.W_initial_linear.bias))]
        M_prev.append([m.clone() for m in M])
        for n, layer in enumerate(layers):
            a = torch.tanh((layer.W * (1 + M[n]) * h[-1][:, None]).sum(-1) + layer.b)
            active = torch.ones(B, dtype=x.dtype) if um is None else um[:, t]
            M[n] = write(layer, M[n], h[-1], a, active)
            h.append(net.residual_scale * a + h[-1] if net._residual_at[n] else a)
        streams.append(h)
        outs.append(F.linear(h[-1], net.W_output, net.b_output))
        for k, (w_name, b_name) in enumerate(net._head_names):
            heads[k].append(F.linear(h[k + 1], getattr(net, w_name),
                                     None if b_name is None else getattr(net, b_name)))
    return (streams, M_prev, torch.stack(outs, 1), [torch.stack(q, 1) for q in heads])


def head_oracle(net, streams, y, mask, loss_and_grad):
    """Autograd of each head's own loss w.r.t. its (C, c), with the streams fixed.
    Also returns the per-head output-error sequence e_n = dL_n/dq_n (B, T, n_out)."""
    grads, errors = {}, []
    for k, (w_name, b_name) in enumerate(net._head_names):
        C = getattr(net, w_name).detach().clone().requires_grad_()
        c = (getattr(net, b_name).detach().clone().requires_grad_()
             if b_name is not None else None)
        h_seq = torch.stack([s[k + 1] for s in streams], 1)          # (B, T, d)
        q = F.linear(h_seq, C, c)
        loss, e = loss_and_grad(q, y, mask)
        params = (C,) if c is None else (C, c)
        g = torch.autograd.grad(loss, params)
        grads[w_name] = g[0]
        if c is not None:
            grads[b_name] = g[1]
        errors.append(e.detach())
    return grads, errors


def layer_oracle(net, n, ell_seq, streams, um):
    """Exact gradient of sum_t <ell_t, h_out_t> for the ISOLATED layer-n trajectory
    (inputs h[n]_t fixed), differentiating through the layer's own plastic state.
    ell_seq is the signal at the layer's OUTPUT boundary h[n+1] (its residual input
    h[n] does not depend on W_n; only the scaled branch contributes to its gradient)."""
    layer = net.mp_layers[n]
    B, T = ell_seq.shape[:2]
    W = layer.W.detach().clone().requires_grad_()
    b = layer.b.detach().clone().requires_grad_()
    M = layer.M_init[None].expand(B, -1, -1).clone()
    objective = 0
    for t in range(T):
        x_t = streams[t][n]
        a = torch.tanh((W * (1 + M) * x_t[:, None]).sum(-1) + b)
        h_out = x_t + net.residual_scale * a if net._residual_at[n] else a
        objective = objective + (ell_seq[:, t] * h_out).sum()
        active = torch.ones(B, dtype=x_t.dtype) if um is None else um[:, t]
        M = write(layer, M, x_t, a, active)
    return torch.autograd.grad(objective, (W, b))


def embed_oracle(net, ell1_seq, x, streams, M_prev):
    """Direct three-factor embedding gradient: the same-time spatial gradient of
    sum_t <ell1_t, h[1]_t> w.r.t. (U, b_in) with layer 0's modulation M0_{t-1}
    treated as a constant (what input_mode 'three_factor' defines)."""
    U = net.W_initial_linear.weight.detach().clone().requires_grad_()
    b_in = net.W_initial_linear.bias.detach().clone().requires_grad_()
    layer = net.mp_layers[0]
    objective = 0
    for t in range(x.shape[1]):
        emb = torch.tanh(F.linear(x[:, t], U, b_in))
        a = torch.tanh((layer.W * (1 + M_prev[t][0]) * emb[:, None]).sum(-1) + layer.b)
        h1 = net.residual_scale * a + emb if net._residual_at[0] else a
        objective = objective + (ell1_seq[:, t] * h1).sum()
    return torch.autograd.grad(objective, (U, b_in))


def head_signal(net, k, e_seq):
    """ell at layer k's output boundary from head k's error: e_k @ C_k, per step."""
    return e_seq @ getattr(net, net._head_names[k][0])


MAIN_KEYS = ('W', 'b', 'W1', 'b1', 'W2', 'b2', 'W_output', 'b_output', 'W_in', 'b_in')


# ─── Gradient correctness ─────────────────────────────────────────────────────
class TestLocalReadoutGradients(unittest.TestCase):
    def test_default_global_has_no_heads_and_identical_results(self):
        x, y, mask, um = data()
        for rule in ('bptt', 'local_direct', 'local_diag_rflo', 'local_exact_rowlocal'):
            with self.subTest(rule=rule):
                implicit = make_net(make_cfg(rule=rule, include_signal_key=False))
                explicit = make_net(make_cfg(rule=rule, signal='global'))
                self.assertEqual(implicit.learning_signal, 'global')
                self.assertEqual(implicit._head_names, [])
                self.assertEqual(implicit._aux_params(), {})
                self.assertEqual(list(implicit.state_dict()), list(explicit.state_dict()))
                self.assertFalse(any(k.startswith('head_') for k in implicit.state_dict()))
                a = implicit.sequence_gradients(x, y, mask, update_masks=um) if rule != 'bptt' \
                    else implicit.sequence_gradients(x, y, mask)
                b = explicit.sequence_gradients(x, y, mask, update_masks=um) if rule != 'bptt' \
                    else explicit.sequence_gradients(x, y, mask)
                self.assertEqual(set(a), set(b))
                self.assertNotIn('aux_loss', a)
                for k in b:
                    if b[k] is not None:
                        self.assertTrue(torch.equal(a[k], b[k]), k)

    def test_single_layer_local_readout_coincides_with_global(self):
        x, y, mask, um = data()
        for rule in ('local_direct', 'local_diag_rflo', 'local_exact_rowlocal'):
            with self.subTest(rule=rule):
                local = make_net(make_cfg(widths=(5,), signal='local_readout', rule=rule))
                glob = make_net(make_cfg(widths=(5,), signal='global', rule=rule))
                self.assertEqual(local._head_names, [])
                self.assertEqual(list(local.state_dict()), list(glob.state_dict()))
                a = local.sequence_gradients(x, y, mask, update_masks=um)
                b = glob.sequence_gradients(x, y, mask, update_masks=um)
                self.assertEqual(set(a), set(b))
                for k in b:
                    if b[k] is not None:
                        self.assertTrue(torch.equal(a[k], b[k]), k)

    def test_head_gradients_match_autograd(self):
        x, y, mask, um = data()
        for residual, output_bias, loss_name in (
                (True, True, 'mse'), (False, True, 'mse'), (True, False, 'mse'),
                (True, True, 'ce'), (False, False, 'ce')):
            with self.subTest(residual=residual, output_bias=output_bias, loss=loss_name):
                net = make_net(make_cfg(signal='local_readout', residual=residual,
                                        output_bias=output_bias))
                if loss_name == 'ce':
                    labels = torch.zeros_like(y)
                    labels[torch.arange(3), :, torch.randint(0, 2, (3,))] = 1
                    loss_and_grad = mpn.masked_cross_entropy_loss_and_grad
                    kw = dict(loss_and_grad=loss_and_grad)
                else:
                    labels, loss_and_grad, kw = y, mse_loss_and_grad, {}
                streams, _, outputs, head_outs = independent_forward(net, x, um)
                expected, _ = head_oracle(net, streams, labels, mask, loss_and_grad)
                actual = net.sequence_gradients(x, labels, mask, update_masks=um, **kw)
                self.assertEqual(len(net._head_names), 2)
                for name, value in expected.items():
                    torch.testing.assert_close(actual[name], value, rtol=1e-10, atol=1e-12)
                # Head bias present iff the main readout has one.
                self.assertEqual('head_b0' in actual, output_bias)
                # Reported aux losses are each head's OWN task loss, main loss untouched.
                torch.testing.assert_close(actual['outputs'], outputs)
                torch.testing.assert_close(actual['loss'], loss_and_grad(outputs, labels, mask)[0])
                for k, q in enumerate(head_outs):
                    torch.testing.assert_close(actual['aux_outputs'][k], q)
                    torch.testing.assert_close(actual['aux_loss'][k], loss_and_grad(q, labels, mask)[0])

    def test_rowlocal_layers_are_exact_for_their_own_head_loss(self):
        """Under local_readout each non-top layer is the TOP of its own module, so
        exact row-local eligibility + its head signal = the exact gradient of that
        head's loss (independent oracle), with and without residuals/update masks."""
        x, y, mask, um_all = data()
        for residual, um in ((True, um_all), (False, um_all), (True, None)):
            with self.subTest(residual=residual, masked=um is not None):
                net = make_net(make_cfg(signal='local_readout', rule='local_exact_rowlocal',
                                        residual=residual, bias='exact'))
                streams, M_prev, outputs, _ = independent_forward(net, x, um)
                _, errors = head_oracle(net, streams, y, mask, mse_loss_and_grad)
                actual = net.sequence_gradients(x, y, mask, update_masks=um)
                # Non-top layers: own head signal.
                for n in range(2):
                    ell = head_signal(net, n, errors[n])
                    gW, gb = layer_oracle(net, n, ell, streams, um)
                    suffix = '' if n == 0 else str(n)
                    torch.testing.assert_close(actual['W' + suffix], gW, rtol=1e-9, atol=1e-11)
                    torch.testing.assert_close(actual['b' + suffix], gb, rtol=1e-9, atol=1e-11)
                # Top layer: the main readout's error through W_output (unchanged).
                _, go = mse_loss_and_grad(outputs, y, mask)
                gW, gb = layer_oracle(net, 2, go @ net.W_output, streams, um)
                torch.testing.assert_close(actual['W2'], gW, rtol=1e-9, atol=1e-11)
                torch.testing.assert_close(actual['b2'], gb, rtol=1e-9, atol=1e-11)
                # Embedding: module 0's head signal through the direct 3-factor rule.
                gU, gbin = embed_oracle(net, head_signal(net, 0, errors[0]), x, streams, M_prev)
                torch.testing.assert_close(actual['W_in'], gU, rtol=1e-9, atol=1e-11)
                torch.testing.assert_close(actual['b_in'], gbin, rtol=1e-9, atol=1e-11)
                # Readout gradient is the usual exact one.
                torch.testing.assert_close(actual['W_output'],
                                           torch.einsum('bta,bti->ai', go, torch.stack([s[-1] for s in streams], 1)))

    def test_locality_perturbing_one_head_changes_only_its_module(self):
        x, y, mask, um = data()
        module_of = {0: ('W', 'b', 'W_in', 'b_in'), 1: ('W1', 'b1')}
        for rule in ('local_direct', 'local_diag_rflo', 'local_exact_rowlocal'):
            base = make_net(make_cfg(signal='local_readout', rule=rule))
            ref = base.sequence_gradients(x, y, mask, update_masks=um)
            for k, own in module_of.items():
                with self.subTest(rule=rule, head=k):
                    net = copy.deepcopy(base)
                    with torch.no_grad():
                        getattr(net, f'head_W{k}').add_(.5)
                    got = net.sequence_gradients(x, y, mask, update_masks=um)
                    torch.testing.assert_close(got['outputs'], ref['outputs'], rtol=0, atol=0)
                    for key in MAIN_KEYS:
                        if key in own:
                            self.assertFalse(torch.equal(got[key], ref[key]), key)
                        else:
                            self.assertTrue(torch.equal(got[key], ref[key]), key)
                    # Other heads' gradients are untouched (they read a forward the
                    # heads never influence).
                    for j in (0, 1):
                        same = torch.equal(got[f'head_W{j}'], ref[f'head_W{j}'])
                        self.assertEqual(same, j != k, f'head_W{j}')

    def test_unequal_widths_disable_residual_where_needed_and_still_run(self):
        x, y, mask, um = data()
        for rule in ('local_direct', 'local_diag_rflo', 'local_exact_rowlocal'):
            with self.subTest(rule=rule):
                with contextlib.redirect_stdout(io.StringIO()):
                    net = make_net(make_cfg(widths=(4, 3, 5), signal='local_readout', rule=rule))
                # embedding 4 -> MP0 4->4 (skip ON), MP1 4->3 and MP2 3->5 (skip OFF).
                self.assertEqual(net._residual_at, [True, False, False])
                self.assertEqual([tuple(getattr(net, w).shape) for w, _ in net._head_names],
                                 [(2, 4), (2, 3)])
                streams, _, _, _ = independent_forward(net, x, um)
                expected, _ = head_oracle(net, streams, y, mask, mse_loss_and_grad)
                actual = net.sequence_gradients(x, y, mask, update_masks=um)
                for name, value in expected.items():
                    torch.testing.assert_close(actual[name], value, rtol=1e-10, atol=1e-12)
                self.assertEqual(len(actual['aux_loss']), 2)
                self.assertTrue(all(torch.isfinite(actual[k]).all() for k in net._trainable_params()))

    def test_bptt_ignores_the_signal_and_reference_grads_exclude_heads(self):
        x, y, mask, _ = data()
        local = make_net(make_cfg(signal='local_readout', rule='bptt'))
        glob = make_net(make_cfg(signal='global', rule='bptt'))
        self.assertEqual(len(local._head_names), 2)
        a = local.sequence_gradients(x, y, mask)
        b = glob.sequence_gradients(x, y, mask)
        self.assertEqual(set(a), set(b))
        for k in b:
            if b[k] is not None:
                self.assertTrue(torch.equal(a[k], b[k]), k)
        self.assertTrue(all(p.grad is None for p in local._aux_params().values()))
        # The alignment diagnostic's BPTT reference on a LOCAL net: main keys only,
        # and the cosine columns never include head parameters.
        net = make_net(make_cfg(signal='local_readout', rule='local_direct'))
        grads = net.sequence_gradients(x, y, mask)
        ref = train_common.bptt_reference_grads(net, x, y, mask, {})
        self.assertEqual(set(ref), set(net._trainable_params()))
        keys = train_common._grad_align_keys(net._trainable_params())
        self.assertEqual(keys, ['W_in', 'W', 'W1', 'W2', 'W_output'])
        cos = train_common.cosine_alignment(grads, ref, keys)
        self.assertTrue(all(np.isfinite(v) for v in cos.values()))
        self.assertEqual(net.learning_rule, 'local_direct')   # restored by the helper

    def test_bptt_three_factor_splice_uses_the_global_signal(self):
        """bptt ignores learning_signal in EVERY input mode: its three_factor embedding
        splice (an extra local_direct pass) must use the main readout's signal, not an
        untrained head's error. Regression for the b552af1 review (W_in differed by ~0.32)."""
        x, y, mask, _ = data()
        for input_mode in ('three_factor', 'match', 'paired'):
            with self.subTest(input_mode=input_mode):
                g = make_net(make_cfg(signal='global', rule='bptt', input_mode=input_mode))
                l = make_net(make_cfg(signal='local_readout', rule='bptt', input_mode=input_mode))
                gg, gl = g.sequence_gradients(x, y, mask), l.sequence_gradients(x, y, mask)
                self.assertEqual(set(gg), set(gl))
                for k in gg:
                    if gg[k] is not None:
                        self.assertTrue(torch.equal(gg[k], gl[k]), k)
                self.assertTrue(all(p.grad is None for p in l._aux_params().values()))
        # The flag itself: a local rule run with use_local_heads=False reproduces the
        # global pass bitwise (no aux keys), with the heads left untouched.
        g = make_net(make_cfg(signal='global', rule='local_direct'))
        l = make_net(make_cfg(signal='local_readout', rule='local_direct'))
        gg = g.local_direct_gradients(x, y, mask)
        gl = l.local_direct_gradients(x, y, mask, use_local_heads=False)
        self.assertEqual(set(gg), set(gl))
        self.assertNotIn('aux_loss', gl)
        for k in gg:
            if gg[k] is not None:
                self.assertTrue(torch.equal(gg[k], gl[k]), k)

    def test_switching_a_model_to_bptt_clears_stale_head_grads(self):
        """Regression for the b552af1 review: a head .grad written by a local pass must
        not survive a later bptt pass on the SAME model (else Adam keeps moving the heads)."""
        x, y, mask, _ = data()
        net = make_net(make_cfg(signal='local_readout', rule='local_direct'))
        net.sequence_gradients(x, y, mask)
        self.assertTrue(all(p.grad is not None for p in net._aux_params().values()))
        net.learning_rule = 'bptt'
        net.sequence_gradients(x, y, mask)
        self.assertTrue(all(p.grad is None for p in net._aux_params().values()))
        # And back: the local pass writes them again.
        net.learning_rule = 'local_diag_rflo'
        net.sequence_gradients(x, y, mask)
        self.assertTrue(all(p.grad is not None for p in net._aux_params().values()))

    def test_mixed_is_linear_in_alpha(self):
        """Traces never depend on ell, so grad(mixed) = grad(global) + alpha*grad(local)
        for every MP/embedding parameter; top layer, readout and heads are shared."""
        x, y, mask, um = data()
        for rule in ('local_direct', 'local_diag_rflo', 'local_exact_rowlocal'):
            with self.subTest(rule=rule):
                g = make_net(make_cfg(signal='global', rule=rule))
                l = make_net(make_cfg(signal='local_readout', rule=rule, head_seed=3))
                m = make_net(make_cfg(signal='mixed', rule=rule, head_seed=3, alpha=.37))
                gg = g.sequence_gradients(x, y, mask, update_masks=um)
                gl = l.sequence_gradients(x, y, mask, update_masks=um)
                gm = m.sequence_gradients(x, y, mask, update_masks=um)
                for k in ('W', 'b', 'W1', 'b1', 'W_in', 'b_in'):
                    torch.testing.assert_close(gm[k], gg[k] + .37 * gl[k], rtol=1e-12, atol=1e-14)
                for k in ('W2', 'b2', 'W_output', 'b_output'):
                    self.assertTrue(torch.equal(gm[k], gg[k]) and torch.equal(gl[k], gg[k]), k)
                for k in ('head_W0', 'head_b0', 'head_W1', 'head_b1'):
                    self.assertTrue(torch.equal(gm[k], gl[k]), k)
                # alpha = 0 reproduces the global MP gradients exactly (heads still train).
                z = make_net(make_cfg(signal='mixed', rule=rule, head_seed=3, alpha=0.))
                gz = z.sequence_gradients(x, y, mask, update_masks=um)
                for k in MAIN_KEYS:
                    torch.testing.assert_close(gz[k], gg[k], rtol=0, atol=0)

    def test_head_init_uses_a_private_rng_stream(self):
        """Building heads must not disturb the global numpy/torch streams: the main
        parameters and anything drawn afterwards (training data) match a global net
        built under the same seed; heads differ across seeds and are reproducible
        from local_head_seed."""
        local = make_net(make_cfg(signal='local_readout'), seed=5)
        after_local = (np.random.rand(), torch.rand(1).item())
        glob = make_net(make_cfg(signal='global'), seed=5)
        after_global = (np.random.rand(), torch.rand(1).item())
        self.assertEqual(after_local, after_global)
        for k, p in glob._trainable_params().items():
            self.assertTrue(torch.equal(local._trainable_params()[k], p), k)
        other = make_net(make_cfg(signal='local_readout'), seed=6)
        self.assertFalse(torch.equal(local.head_W0, other.head_W0))
        a = make_net(make_cfg(signal='local_readout', head_seed=9), seed=5)
        b = make_net(make_cfg(signal='local_readout', head_seed=9), seed=6)
        self.assertTrue(torch.equal(a.head_W0, b.head_W0))
        self.assertFalse(torch.equal(a.W_output, b.W_output))


# ─── Guards ───────────────────────────────────────────────────────────────────
class TestLocalReadoutGuards(unittest.TestCase):
    def test_model_guards(self):
        x, y, mask, _ = data()
        for bad, pattern in ((dict(feedback='direct_fa'), 'exact_spatial'),
                             (dict(feedback='layerwise_fa'), 'exact_spatial'),
                             (dict(cross=1), 'cross_layer_steps'),
                             (dict(alpha=float('nan')), 'finite')):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, pattern):
                make_net(make_cfg(signal='local_readout', **bad))
        with self.assertRaisesRegex(ValueError, 'unknown learning_signal'):
            make_net(make_cfg(signal='nope'))
        # A BPTT input splice is not local: rejected at gradient time for local rules.
        net = make_net(make_cfg(signal='local_readout', rule='local_direct', input_mode='exact'))
        with self.assertRaisesRegex(ValueError, 'not local'):
            net.sequence_gradients(x, y, mask)
        # ... but the bptt rule itself (native exact embedding) is fine.
        make_net(make_cfg(signal='local_readout', rule='bptt', input_mode='exact')).sequence_gradients(x, y, mask)
        # The other input modes run.
        for mode in ('match', 'three_factor', 'diag_mtrace', 'paired'):
            for rule in ('local_direct', 'local_diag_rflo'):
                make_net(make_cfg(signal='local_readout', rule=rule, input_mode=mode)).sequence_gradients(x, y, mask)
        # Single-MP-layer net has nowhere to attach a head.
        cfg = make_cfg(widths=(4,), signal='local_readout')
        cfg['n_neurons'] = [3, 4, 2]
        with self.assertRaisesRegex(ValueError, 'DeepMultiPlasticNet'), contextlib.redirect_stdout(io.StringIO()):
            mpn.MultiPlasticNet(cfg)

    def test_cli_defaults_and_guards(self):
        with patch.object(sys, 'argv', ['train_mpn.py']):
            args = train_mpn._parse_args()
        self.assertEqual((args.learning_signal, args.input_mode, args.cross_layer_steps, args.seed),
                         ('global', train_mpn.INPUT_MODE, train_mpn.CROSS_LAYER_STEPS, None))
        with patch.object(sys, 'argv', ['train_mpn.py', '--learning-signal', 'local_readout', '--seed', '7']):
            args = train_mpn._parse_args()
        # The two conflicting module defaults ('exact', 1) switch; everything else is kept.
        self.assertEqual((args.learning_signal, args.input_mode, args.cross_layer_steps,
                          args.feedback, args.seed),
                         ('local_readout', 'match', 0, 'exact_spatial', 7))
        with patch.object(sys, 'argv', ['train_mpn.py', '--learning-signal', 'mixed',
                                        '--local-signal-alpha', '0.5', '--input-mode', 'paired']):
            args = train_mpn._parse_args()
        self.assertEqual((args.learning_signal, args.input_mode, args.local_signal_alpha),
                         ('mixed', 'paired', .5))
        # A non-conflicting module default is respected.
        with patch.multiple(train_mpn, INPUT_MODE='diag_mtrace'), \
                patch.object(sys, 'argv', ['train_mpn.py', '--learning-signal', 'local_readout']):
            self.assertEqual(train_mpn._parse_args().input_mode, 'diag_mtrace')
        for bad in (['--input-mode', 'exact'], ['--cross-layer-steps', '1'],
                    ['--feedback', 'direct_fa'], ['--feedback', 'layerwise_fa'],
                    ['--net', 'mpn1'], ['--dfa'], ['--local-signal-alpha', 'nan']):
            with self.subTest(bad=bad), patch.object(
                    sys, 'argv', ['train_mpn.py', '--learning-signal', 'local_readout', *bad]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train_mpn._parse_args()
        # build_params records the signal and rejects it for mpn1.
        with patch.multiple(train_mpn, LEARNING_SIGNAL='local_readout', LOCAL_SIGNAL_ALPHA=.25):
            _, _, params = train_mpn.build_params()
            self.assertEqual((params['learning_signal'], params['local_signal_alpha']), ('local_readout', .25))
            with patch.object(train_mpn, 'NET_TYPE', 'mpn1'), patch.object(train_mpn, 'MP_RESIDUAL', False):
                with self.assertRaisesRegex(ValueError, 'dmpn'):
                    train_mpn.build_params()


# ─── Runner / persistence ─────────────────────────────────────────────────────
class TestLocalReadoutRunner(unittest.TestCase):
    def test_checkpoint_roundtrip_and_legacy_params(self):
        x, y, mask, _ = data()
        cfg = make_cfg(signal='local_readout', rule='local_diag_rflo', head_seed=4)
        net = make_net(cfg)
        expected = net.sequence_gradients(x, y, mask)
        self.assertTrue({'head_W0', 'head_b0', 'head_W1', 'head_b1'} <= set(net.state_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'local.pt'
            torch.save(dict(net_params=cfg, state_dict=net.state_dict(),
                            learning_rule=net.learning_rule), path)
            with contextlib.redirect_stdout(io.StringIO()):
                loaded = train_mpn.load_net(path, device=torch.device('cpu'), dtype=torch.double)
            self.assertEqual(loaded.learning_signal, 'local_readout')
            actual = loaded.sequence_gradients(x, y, mask)
            for key in expected:
                if key in ('aux_loss', 'aux_outputs'):
                    for a, e in zip(actual[key], expected[key]):
                        torch.testing.assert_close(a, e, rtol=0, atol=0)
                elif expected[key] is not None:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            # A pre-feature checkpoint (no learning_signal key) builds a head-less
            # net whose state_dict matches the stored one exactly.
            legacy_cfg = make_cfg(rule='local_diag_rflo', include_signal_key=False)
            legacy = make_net(legacy_cfg)
            torch.save(dict(net_params=legacy_cfg, state_dict=legacy.state_dict(),
                            learning_rule='local_diag_rflo'), path)
            with contextlib.redirect_stdout(io.StringIO()):
                loaded = train_mpn.load_net(path, device=torch.device('cpu'), dtype=torch.double)
            self.assertEqual((loaded.learning_signal, loaded._head_names), ('global', []))

    def test_run_seed_clips_separately_logs_heads_and_saves_metadata(self):
        x, y, mask, _ = data(B=4, T=5)
        net_params = make_cfg(signal='local_readout', rule='bptt')
        cfg = train_mpn._cfg()
        cfg.rules_to_run = ['bptt', 'local_direct', 'local_diag_rflo']
        cfg.learning_signal, cfg.input_mode, cfg.cross_layer_steps = 'local_readout', 'match', 0
        cfg.input_normalize, cfg.log_grad_align, cfg.save_nets = False, True, True
        cfg.n_datasets, cfg.batch, cfg.grad_clip = 2, 4, 1e-3     # tiny clip → always active
        cfg.device, cfg.dtype = torch.device('cpu'), torch.double
        cfg.build_params = lambda: ({}, {'reg_lambda': 1e-3, 'weight_reg': 'L2'}, copy.deepcopy(net_params))
        cfg.net_factory = lambda params, verbose: mpn.DeepMultiPlasticNet(params, verbose=False)
        cfg.task = SimpleNamespace(
            init_params=lambda *params: params,
            valid_batch=lambda *args: (x, y, mask),
            train_batch=lambda *args: (x, y, mask),
            accuracy=lambda net, out, labels, m, inputs, isvalid=False: float(-((out - labels) * m).pow(2).mean()),
            loss_and_grad=None)
        clip_calls = []
        real_clip = torch.nn.utils.clip_grad_norm_

        def spy(params, *a, **k):
            clip_calls.append([id(p) for p in params])
            return real_clip(params, *a, **k)

        with tempfile.TemporaryDirectory() as directory:
            cfg.ckpt_dir = cfg.fig_dir = cfg.data_dir = directory
            log = io.StringIO()
            with patch.object(torch.nn.utils, 'clip_grad_norm_', spy), contextlib.redirect_stdout(log):
                curves, align = train_common.run_seed(cfg, 13, [0, 1])
            text = log.getvalue()
            self.assertIn('learning_signal=local_readout: 2 local readout head(s)', text)
            self.assertIn('head0: acc', text)
            self.assertIn('head1: acc', text)
            # Every clip call is either all-main or all-head: the two groups never mix.
            heads_seen = 0
            for rule in cfg.rules_to_run:
                ckpt = torch.load(train_common.ckpt_path(cfg, rule, 13), weights_only=False)
                self.assertEqual(ckpt['learning_signal'], 'local_readout')
                self.assertEqual(ckpt['net_params']['learning_signal'], 'local_readout')
                self.assertEqual(ckpt['net_params']['learning_rule'], rule)
                self.assertTrue({'head_W0', 'head_b0', 'head_W1', 'head_b1'} <= set(ckpt['state_dict']))
                with contextlib.redirect_stdout(io.StringIO()):
                    reloaded = train_mpn.load_net(train_common.ckpt_path(cfg, rule, 13),
                                                  device=torch.device('cpu'), dtype=torch.double)
                head_ids_any = {id(p) for p in reloaded._aux_params().values()}
                self.assertEqual(len(head_ids_any), 4)
            # Alignment columns exclude heads; curves are finite.
            self.assertEqual(set(align), {'local_direct', 'local_diag_rflo'})
            for per_key in align.values():
                self.assertEqual(set(per_key), {'W_in', 'W', 'W1', 'W2', 'W_output'})
            for split_curves in curves.values():
                for values in split_curves.values():
                    self.assertTrue(np.isfinite(values).all())
            # Separate clipping: 2 steps × (bptt: main only [aux group has no grads but
            # is still a separate call] ; local rules: main + aux) → no call mixes groups.
            self.assertGreaterEqual(len(clip_calls), 2 * 3)
            # Saved config carries the signal.
            with contextlib.redirect_stdout(io.StringIO()):
                record_path = train_common.save_config(cfg)
            import json
            record = json.loads(Path(record_path).read_text())
            self.assertEqual(record['learning_signal'], 'local_readout')
            self.assertEqual(record['net_params']['learning_signal'], 'local_readout')
        # The clip spy saw parameter-id lists; verify no list contains both a head
        # and a non-head parameter by reconstructing from a fresh net of the same
        # architecture is not possible (ids differ) — instead check group sizes:
        # main group = 10 tensors (W_in,b_in, W,b,W1,b1,W2,b2, W_output,b_output),
        # aux group = 4 tensors (two heads × weight+bias).
        sizes = sorted(set(len(c) for c in clip_calls))
        self.assertEqual(sizes, [4, 10])

    def test_legacy_tag_and_header_mention_the_signal(self):
        cfg = train_mpn._cfg()
        cfg.run_id = ''
        base_tag = train_common.param_tag(cfg)
        cfg.learning_signal = 'local_readout'
        self.assertEqual(train_common.param_tag(cfg), base_tag + '_ls-local_readout')
        cfg.learning_signal = 'global'
        self.assertEqual(train_common.param_tag(cfg), base_tag)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
