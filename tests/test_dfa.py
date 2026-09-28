"""Independent autograd oracles for local MPN+DFA (small CPU double tensors).
Run: python -m unittest discover -s tests -v
"""
import copy
import contextlib
import io
from itertools import product
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


def make_net(rule='local_exact_rowlocal', bias='exact', residual=False, kinds=None,
             bounded=False, smooth_scale=None):
    np.random.seed(37)
    torch.manual_seed(37)
    cfg = dict(n_neurons=[2, 3, 3, 4, 2], dt=1, activation='tanh',
               output_matrix='', output_bias=True, input_layer_add=True,
               input_layer_add_trainable=True, input_layer_bias=True, linear_embed=3,
               learning_rule=rule, feedback_mode='direct_fa', input_mode='match',
               cross_layer_steps=0, mp_residual=residual,
               ml_params=dict(bias=True, mp_type='mult', m_update_type='hebb_assoc',
                              m_activation='linear' if smooth_scale is None else 'scaled_tanh',
                              m_scale=1.0 if smooth_scale is None else smooth_scale,
                              modulation_bounds=bounded,
                              m_bounds=(-1.0, 1.0),
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
    if layer.m_act == 'scaled_tanh':
        new = layer.m_scale * torch.tanh(new / layer.m_scale)
    updated = active[:, None, None] * new + (1 - active[:, None, None]) * M
    return updated.clamp(-1.0, 1.0) if layer.modulation_bounds else updated


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
                        if layer.m_act == 'scaled_tanh':
                            new = layer.m_scale * torch.tanh(new / layer.m_scale)
                        m = um[:, t] * new + (1 - um[:, t]) * m
                        if layer.modulation_bounds:
                            m = m.clamp(-1.0, 1.0)
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
    def test_modulation_cli_modes_and_scales(self):
        import train_mpn

        for mode, tag in (('none', ''), ('hard', '_mb-0.4-0.4'), ('scaled_tanh', '_mtanh-0.4')):
            with self.subTest(mode=mode):
                with patch.object(sys, 'argv', ['train_mpn.py', '--dfa', '--modulation-mode', mode,
                                               '--modulation-bound', '0.4']):
                    args = train_mpn._parse_args()
                self.assertEqual(args.modulation_mode, mode)
                self.assertEqual(args.modulation_bound, .4)
                with patch.multiple(train_mpn, MODULATION_MODE=mode, MODULATION_BOUND=.4,
                                    MODULATION_BOUNDS=mode == 'hard'):
                    _, _, params = train_mpn.build_params()
                    layer_params = params['ml_params']
                    self.assertEqual(layer_params['m_scale'], .4)
                    self.assertEqual(layer_params['modulation_bounds'], mode == 'hard')
                    self.assertEqual(layer_params['m_activation'],
                                     'scaled_tanh' if mode == 'scaled_tanh' else 'linear')
                    actual_tag = train_mpn._cfg().tag_extra
                    if tag:
                        self.assertIn(tag, actual_tag)
                    else:
                        self.assertNotIn('_mb-', actual_tag)
                        self.assertNotIn('_mtanh-', actual_tag)
        for invalid in (0., -1., float('nan'), float('inf')):
            with self.subTest(invalid=invalid):
                with patch.object(sys, 'argv', ['train_mpn.py', f'--modulation-bound={invalid}']):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        train_mpn._parse_args()
                with self.assertRaises(ValueError):
                    make_net(smooth_scale=invalid)
        with self.assertRaises(ValueError):
            make_net(bounded=True, smooth_scale=1.)

    def test_runner_defaults_bound_modulation_without_regularization(self):
        import train_common
        import train_mpn

        self.assertTrue(train_mpn.MODULATION_BOUNDS)
        self.assertEqual(train_mpn.REG_LAMBDA, 0.0)
        _, train_params, net_params = train_mpn.build_params()
        self.assertTrue(net_params['ml_params']['modulation_bounds'])
        self.assertEqual(net_params['ml_params']['m_bounds'], (-1.0, 1.0))
        self.assertEqual(train_params['reg_lambda'], 0.0)
        self.assertIsNone(train_params['activity_reg'])
        _, optimizer, _ = train_common.make_optim(
            make_net(), .001, weight_decay=train_params['reg_lambda'])
        self.assertTrue(all(group['weight_decay'] == 0 for group in optimizer.param_groups))
        config = train_mpn._cfg()
        tag = config.tag_extra
        self.assertIn('_mb-1-1', tag)
        self.assertNotIn('_l2-', tag)
        with patch.object(train_mpn, 'MODULATION_BOUNDS', False):
            unbounded_config = train_mpn._cfg()
        for path_builder in (train_common.fig_path, train_common.data_path,
                             train_common.config_path, train_common.align_fig_path):
            bounded_path = path_builder(config)
            self.assertIn('_mb-1-1', bounded_path)
            self.assertEqual(bounded_path.replace('_mb-1-1', ''), path_builder(unbounded_config))
        for rule in ('bptt', 'local_diag_rflo', 'local_direct'):
            self.assertIn('_mb-1-1', train_common.ckpt_path(config, rule, 291))

    def test_runner_modulation_modes_and_regularization_for_all_rules(self):
        import train_common
        import train_mpn

        for net_type, mode in product(('dmpn', 'mpn1'), ('none', 'hard', 'scaled_tanh')):
            with self.subTest(net_type=net_type, mode=mode), patch.multiple(
                    train_mpn, NET_TYPE=net_type, N_HIDDEN=[3, 3] if net_type == 'dmpn' else 3,
                    RULESET='delaygo', N_DATASETS=2, N_RUNS=1, BATCH=2,
                    DEVICE=torch.device('cpu'), DTYPE=torch.double, MP_RESIDUAL=False,
                    MODULATION_MODE=mode, MODULATION_BOUND=.4,
                    MODULATION_BOUNDS=mode == 'hard', REG_LAMBDA=1e-4,
                    INPUT_MODE='match', LOG_GRAD_ALIGN=False, SAVE_NETS=False):
                config = train_mpn._cfg()
                original_builder = config.build_params

                def small_params():
                    task_params, train_params, net_params = original_builder()
                    train_params['valid_n_batch'] = 2
                    self.assertEqual(train_params['reg_lambda'], 1e-4)
                    self.assertIsNone(train_params['activity_reg'])
                    self.assertEqual(net_params['ml_params']['modulation_bounds'], mode == 'hard')
                    self.assertEqual(net_params['ml_params']['m_bounds'], (-.4, .4))
                    return task_params, train_params, net_params

                config.build_params = small_params
                with patch.object(train_common, 'make_optim', wraps=train_common.make_optim) as factory:
                    with contextlib.redirect_stdout(io.StringIO()):
                        curves, _ = train_common.run_seed(config, 37, [0, 1])
                self.assertEqual(factory.call_count, len(config.rules_to_run))
                for call in factory.call_args_list:
                    self.assertEqual(call.kwargs['weight_decay'], 1e-4)
                    network = call.args[0]
                    layers = network.mp_layers if net_type == 'dmpn' else [network.mp_layer]
                    for layer in layers:
                        if mode == 'hard':
                            self.assertTrue((layer.M >= layer.M_bounds[1]).all())
                            self.assertTrue((layer.M <= layer.M_bounds[0]).all())
                        elif mode == 'scaled_tanh':
                            self.assertTrue((layer.M.abs() <= layer.m_scale).all())
                for split_curves in curves.values():
                    for values in split_curves.values():
                        self.assertTrue(np.isfinite(values).all())

    def test_bounded_general_and_fast_write_paths(self):
        layer = make_net(bounded=True).mp_layers[0]
        layer.reset_state(B=2)
        layer.reset_local_learning_state(B=2)
        with torch.no_grad():
            layer.eta.fill_(10)
        reference = copy.deepcopy(layer)
        inputs = torch.tensor([[1., -1., .1], [-1., 1., -.1]], dtype=torch.double)
        post = torch.ones(2, layer.n_output, dtype=torch.double)
        eligibility = torch.ones_like(layer.M)
        bias_eligibility = torch.ones_like(post)
        update_mask = torch.tensor([1., 0.], dtype=torch.double)
        with torch.no_grad():
            for current in (layer, reference):
                current.update_exact_rowlocal_traces(
                    inputs, eligibility, bias_eligibility, update_mask=update_mask)
            layer.update_M_matrix_local_fast(inputs, post, update_mask=update_mask)
            reference.update_M_matrix(inputs, post, update_mask=update_mask)
        torch.testing.assert_close(layer.M, reference.M)
        torch.testing.assert_close(layer.P, reference.P)
        torch.testing.assert_close(layer.Q, reference.Q)
        self.assertTrue((layer.M.abs() <= 1).all())

    def test_clamp_derivative_matches_autograd_at_endpoints(self):
        layer = make_net(bounded=True).mp_layers[0]
        raw = torch.tensor([[[-2., -1., -.5], [0., .5, 1.], [2., 2., -2.]]],
                           dtype=torch.double, requires_grad=True)
        expected = torch.autograd.grad(raw.clamp(-1., 1.).sum(), raw)[0]
        clipped = layer._apply_modulation_bounds(raw)
        torch.testing.assert_close(clipped, raw.detach().clamp(-1., 1.))
        torch.testing.assert_close(torch.autograd.grad(clipped.sum(), raw)[0], expected)
        layer.M_pre = raw.detach()
        torch.testing.assert_close(layer._modulation_clamp_derivative().to(raw.dtype), expected)

    @torch.no_grad()
    def test_frozen_state_traces_for_general_and_fast_updates(self):
        for mode in ('exact', 'diag'):
            for bounded in (False, True):
                for bias in ('exact', 'direct'):
                    for fast in (False, True):
                        with self.subTest(mode=mode, bounded=bounded, bias=bias, fast=fast):
                            layer = make_net(bias=bias, bounded=bounded).mp_layers[0]
                            layer.set_plasticity_freeze(torch.tensor([0]), torch.tensor([1]))
                            layer.reset_state(B=2)
                            if mode == 'exact':
                                layer.reset_local_learning_state(B=2)
                            else:
                                layer.reset_diag_rflo_state(B=2)
                            frozen_value = layer.M[:, 0, 1].clone()
                            inputs = torch.full((2, layer.n_input), .2, dtype=torch.double)
                            phi_prime = torch.full((2, layer.n_output), .5, dtype=torch.double)
                            eta, lam = layer._eta_lam_full()
                            layer.step_fn_for(mode)(inputs, phi_prime, torch.ones_like(phi_prime), eta, lam)
                            trace_name = 'P' if mode == 'exact' else 'A'
                            expected_trace = getattr(layer, trace_name).clone()
                            if mode == 'exact':
                                expected_trace[:, 0, :, 1] = 0
                            else:
                                expected_trace[:, 0, 1] = 0
                            expected_bias_trace = None if layer.Q is None else layer.Q.clone()
                            if expected_bias_trace is not None:
                                expected_bias_trace[:, 0, 1] = 0
                            update = layer.update_M_matrix_local_fast if fast else layer.update_M_matrix
                            update(inputs, torch.full_like(phi_prime, .3))
                            torch.testing.assert_close(layer.M[:, 0, 1], frozen_value)
                            torch.testing.assert_close(getattr(layer, trace_name), expected_trace)
                            if expected_bias_trace is None:
                                self.assertIsNone(layer.Q)
                            else:
                                torch.testing.assert_close(layer.Q, expected_bias_trace)

    def test_weight_regularization_matches_explicit_l2_gradient(self):
        import train_common

        coefficient = 1e-4
        net = make_net()
        reference = copy.deepcopy(net)
        _, optimizer, _ = train_common.make_optim(net, .001, weight_decay=coefficient)
        baseline = torch.optim.Adam(reference.parameters(), lr=.001)
        for name, parameter in net._trainable_params().items():
            parameter.grad = torch.full_like(parameter, .2)
            reference_parameter = reference._trainable_params()[name]
            reference_parameter.grad = parameter.grad.clone()
            if name.startswith('W'):
                reference_parameter.grad.add_(reference_parameter.detach(), alpha=coefficient)
        optimizer.step()
        baseline.step()
        for name, parameter in net._trainable_params().items():
            torch.testing.assert_close(parameter, reference._trainable_params()[name],
                                       rtol=1e-12, atol=1e-12)
        for group in optimizer.param_groups:
            for parameter in group['params']:
                name = next(name for name, candidate in net._trainable_params().items()
                            if candidate is parameter)
                self.assertEqual(group['weight_decay'], coefficient if name.startswith('W') else 0)

    def test_both_rules_against_independent_autograd(self):
        x, y, masks, um = data()
        for diag, (bounded, smooth_scale) in product((False, True), ((False, None), (True, None), (False, .4))):
            write_masks = um if smooth_scale is None else torch.tensor(
                [[1., .25, 0., 1.], [.75, 0., 1., 1.]], dtype=x.dtype)
            for bias in ('exact', 'direct'):
                for residual in (False, True):
                    for kinds in (None, ['hebb_pre', 'hebb_assoc', 'hebb_pre'], ['hebb_pre']*3):
                        with self.subTest(diag=diag, bounded=bounded, smooth_scale=smooth_scale,
                                          bias=bias, residual=residual, kinds=kinds):
                            rule = 'local_diag_rflo' if diag else 'local_exact_rowlocal'
                            net = make_net(rule, bias, residual, kinds, bounded=bounded, smooth_scale=smooth_scale)
                            if bounded or smooth_scale is not None:
                                with torch.no_grad():
                                    for layer in net.mp_layers:
                                        layer.eta.fill_(1000 if bounded else 20)
                            expected_out, expected = oracle(net, x, y, masks, write_masks, diag)
                            # No hidden BPTT/autograd is allowed in either local DFA run.
                            with patch.object(net, 'bptt_gradients', side_effect=AssertionError('BPTT called')):
                                actual = net.sequence_gradients(x, y, masks, update_masks=write_masks)
                            torch.testing.assert_close(actual['outputs'], expected_out, rtol=1e-10, atol=1e-11)
                            for key, val in expected.items():
                                torch.testing.assert_close(actual[key], val, rtol=1e-9, atol=1e-10)
                            if bounded:
                                self.assertTrue(any((layer.M.abs() == 1).any() for layer in net.mp_layers))
                            if smooth_scale is not None:
                                self.assertTrue(all((layer.M.abs() <= smooth_scale).all() for layer in net.mp_layers))
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
