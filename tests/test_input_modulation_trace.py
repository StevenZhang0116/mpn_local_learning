"""Input modulation-column sensitivities checked against independent autograd graphs."""
import contextlib
import copy
import io
from itertools import product
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import _bootstrap
import mpn
import train_mpn


def make_net(embed=3, widths=(3, 2), write_mode='linear', kind='hebb_assoc',
             residual=False, rule='local_diag_rflo', input_mode='diag_mtrace'):
    torch.manual_seed(13)
    cfg = dict(net_type='dmpn', n_neurons=[2, *widths, 2], dt=1,
               activation='tanh', output_matrix='', output_bias=True,
               input_layer_add=True, input_layer_add_trainable=True,
               input_layer_bias=True, linear_embed=embed, input_init_type='xavier',
               learning_rule=rule, feedback_mode='exact_spatial', input_mode=input_mode,
               cross_layer_steps=0, mp_residual=residual,
               ml_params=dict(bias=True, mp_type='mult', m_update_type=kind,
                              m_activation='scaled_tanh' if write_mode == 'smooth' else 'linear',
                              m_scale=.4, modulation_bounds=write_mode == 'hard',
                              m_bounds=(-.3, .3), eta_type='scalar', eta_train=False,
                              lam_type='scalar', lam_train=False, m_time_scale=10,
                              local_bias_mode='direct'))
    with contextlib.redirect_stdout(io.StringIO()):
        net = mpn.DeepMultiPlasticNet(cfg, verbose=False).double()
    with torch.no_grad():
        for layer in net.mp_layers:
            layer.eta.fill_(3.0 if write_mode != 'linear' else .7)
            layer.lam.fill_(.8)
            layer.M_init.uniform_(-.1, .1)
    return net, cfg


def data():
    torch.manual_seed(29)
    inputs = torch.randn(2, 5, 2, dtype=torch.double) * .8
    labels = torch.randn(2, 5, 2, dtype=torch.double) * .4
    mask = torch.rand_like(labels)
    active = torch.tensor([[1., .25, 0., 1., .7], [.5, 1., 1., 0., 1.]], dtype=torch.double)
    return inputs, labels, mask, active


def fourth_power_loss(outputs, labels, mask):
    diff = outputs - labels
    return (mask * diff.pow(4)).mean(), 4 * mask * diff.pow(3) / outputs.numel()


def oracle(net, inputs, labels, mask, active=None, column=None, custom=False):
    """Differentiate a full unroll or a graph retaining one first-MP state column.

    No production trace/signal/write helpers. With column=j, detach all embedding
    outputs except j, all other first-layer M columns, and all later-layer states.
    Forward VALUES remain unchanged. This defines the approximation independently
    of the explicit sensitivity recurrence used in production.
    """
    if net.input_normalize:
        inputs = (inputs - net.input_loc) / net.input_scale
    states = [layer.M_init[None].expand(len(inputs), -1, -1).clone()
              for layer in net.mp_layers]
    outputs = []
    for t in range(inputs.shape[1]):
        h = torch.tanh(net.W_initial_linear(inputs[:, t]))
        if column is not None:
            selector = torch.zeros_like(h)
            selector[:, column] = 1
            h = h.detach() + selector * (h - h.detach())
        for n, layer in enumerate(net.mp_layers):
            a = torch.tanh((layer.W * (1 + states[n]) * h[:, None]).sum(-1) + layer.b)
            post = a if layer.m_update_type == 'hebb_assoc' else torch.ones_like(a) / a.shape[1] ** .5
            raw = layer.lam * states[n] + layer.eta * post[:, :, None] * h[:, None]
            r = 1 if active is None else active[:, t, None, None]
            if layer.m_act == 'scaled_tanh':
                candidate = layer.m_scale * torch.tanh(raw / layer.m_scale)
                updated = r * candidate + (1-r) * states[n]
            else:
                updated = r * raw + (1-r) * states[n]
                if layer.modulation_bounds:
                    updated = torch.where(updated < layer.M_bounds[1], layer.M_bounds[1],
                                          torch.where(updated > layer.M_bounds[0], layer.M_bounds[0], updated))
            frozen = getattr(layer, '_plasticity_freeze_mask', None)
            if frozen is not None:
                keep = torch.ones_like(updated, dtype=torch.bool)
                keep[:, frozen[0], frozen[1]] = False
                updated = torch.where(keep, updated, layer.M_init[None])
            if column is not None:
                keep = torch.zeros_like(updated)
                if n == 0:
                    keep[:, :, column] = 1
                updated = updated.detach() + keep * (updated - updated.detach())
            states[n] = updated
            h = a + h if net._residual_at[n] else a
        outputs.append(F.linear(h, net.W_output, net.b_output))
    outputs = torch.stack(outputs, 1)
    loss = fourth_power_loss(outputs, labels, mask)[0] if custom else ((outputs-labels)*mask).square().mean()
    parameters = {k: p for k, p in net._trainable_params().items() if k in ('W_in', 'b_in')}
    gradients = torch.autograd.grad(loss, list(parameters.values()))
    return outputs.detach(), dict(zip(parameters, gradients))


class TestInputModulationTrace(unittest.TestCase):
    def assert_input_close(self, actual, expected):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-9, atol=2e-11)

    def test_single_embedding_coordinate_matches_full_bptt(self):
        x, y, mask, active = data()
        for write_mode, kind in product(('linear', 'hard', 'smooth'), ('hebb_assoc', 'hebb_pre')):
            with self.subTest(write_mode=write_mode, kind=kind):
                net, _ = make_net(embed=1, widths=(3,), write_mode=write_mode, kind=kind)
                expected_out, expected = oracle(net, x, y, mask, active)
                actual = net.sequence_gradients(x, y, mask, update_masks=active)
                self.assert_input_close(actual, expected)
                torch.testing.assert_close(actual['outputs'], expected_out)
                # The public full-BPTT path also agrees when no update mask is used.
                reference = net.bptt_gradients(x, y, mask)
                actual = net.sequence_gradients(x, y, mask)
                self.assert_input_close(actual, {k: reference[k] for k in expected})

    def test_pre_only_write_matches_full_bptt_with_multiple_embedding_coordinates(self):
        x, y, mask, active = data()
        for write_mode in ('linear', 'hard', 'smooth'):
            with self.subTest(write_mode=write_mode):
                net, _ = make_net(widths=(3,), kind='hebb_pre', residual=True, write_mode=write_mode)
                _, expected = oracle(net, x, y, mask, active)
                self.assert_input_close(net.sequence_gradients(x, y, mask, update_masks=active), expected)

    def test_deep_column_approximation_matches_independent_autograd(self):
        x, y, mask, active = data()
        for write_mode, residual, kind in product(('linear', 'hard', 'smooth'),
                                                  (False, True), ('hebb_assoc', 'hebb_pre')):
            with self.subTest(write_mode=write_mode, residual=residual, kind=kind):
                net, _ = make_net(widths=(3, 3), write_mode=write_mode, residual=residual, kind=kind)
                net.mp_layers[0].set_plasticity_freeze(torch.tensor([0]), torch.tensor([1]))
                net.set_input_norm_stats(x.reshape(-1, 2))
                expected = {}
                for j in range(3):
                    expected_out, per_column = oracle(net, x, y, mask, active, column=j)
                    for key, value in per_column.items():
                        expected[key] = expected.get(key, 0) + value
                actual = net.sequence_gradients(x, y, mask, update_masks=active)
                self.assert_input_close(actual, expected)
                torch.testing.assert_close(actual['outputs'], expected_out, rtol=1e-10, atol=1e-11)

    def test_only_input_gradients_change_and_bptt_stays_exact(self):
        x, y, mask, _ = data()
        for rule, correction in product(('bptt', 'local_direct', 'local_diag_rflo', 'local_exact_rowlocal'), (0, 1)):
            with self.subTest(rule=rule, correction=correction):
                net, _ = make_net(rule=rule, widths=(3, 3))
                net.cross_layer_steps = correction
                baseline = copy.deepcopy(net)
                baseline.input_mode = 'match'
                expected = baseline.sequence_gradients(x, y, mask)
                with patch.object(net, 'bptt_gradients', wraps=net.bptt_gradients) as bptt:
                    actual = net.sequence_gradients(x, y, mask)
                    self.assertEqual(bptt.call_count, int(rule == 'bptt'))
                for key in expected:
                    if rule == 'bptt' or key not in ('W_in', 'b_in'):
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                if rule != 'bptt':
                    self.assertGreater((actual['W_in'] - expected['W_in']).abs().max().item(), 1e-7)
                for key, parameter in net._trainable_params().items():
                    torch.testing.assert_close(parameter.grad, actual[key])

    def test_delayed_loss_credits_past_stimulus(self):
        net, _ = make_net(embed=1, widths=(3,), rule='local_direct')
        x, y, mask, _ = data()
        x[:, 1:] = 0
        mask[:, :-1] = 0
        baseline = copy.deepcopy(net)
        baseline.input_mode = 'match'
        direct = baseline.sequence_gradients(x, y, mask)
        actual = net.sequence_gradients(x, y, mask)
        reference = net.bptt_gradients(x, y, mask)
        self.assertEqual(direct['W_in'].abs().max().item(), 0)
        self.assertGreater(actual['W_in'].abs().max().item(), 1e-7)
        self.assert_input_close(actual, {k: reference[k] for k in ('W_in', 'b_in')})

    def test_custom_loss_partial_freeze_and_repeatability(self):
        x, y, mask, active = data()
        for train_weight, train_bias in ((True, False), (False, True), (True, True), (False, False)):
            with self.subTest(train_weight=train_weight, train_bias=train_bias):
                net, _ = make_net(embed=1, widths=(1,), write_mode='smooth', residual=True)
                net.W_initial_linear.weight.requires_grad_(train_weight)
                net.W_initial_linear.bias.requires_grad_(train_bias)
                kwargs = dict(update_masks=active, loss_and_grad=fourth_power_loss)
                actual = net.sequence_gradients(x, y, mask, **kwargs)
                if train_weight or train_bias:
                    _, expected = oracle(net, x, y, mask, active, custom=True)
                    self.assert_input_close(actual, expected)
                self.assertEqual('W_in' in actual, train_weight)
                self.assertEqual('b_in' in actual, train_bias)
                repeated = net.sequence_gradients(x, y, mask, return_outputs=False, **kwargs)
                self.assertIsNone(repeated['outputs'])
                for key in actual:
                    if key != 'outputs':
                        torch.testing.assert_close(actual[key], repeated[key], rtol=0, atol=0)

    def test_zero_plasticity_reduces_to_instantaneous_input_gradient(self):
        x, y, mask, _ = data()
        for setting in ('eta_zero', 'no_writes', 'T_one'):
            with self.subTest(setting=setting):
                net, _ = make_net()
                kwargs = {}
                if setting == 'eta_zero':
                    net.mp_layers[0].eta.zero_()
                elif setting == 'no_writes':
                    kwargs['update_masks'] = torch.zeros(x.shape[:2], dtype=x.dtype)
                baseline = copy.deepcopy(net)
                baseline.input_mode = 'match'
                xx, yy, mm = (x[:, :1], y[:, :1], mask[:, :1]) if setting == 'T_one' else (x, y, mask)
                expected = baseline.sequence_gradients(xx, yy, mm, **kwargs)
                actual = net.sequence_gradients(xx, yy, mm, **kwargs)
                self.assert_input_close(actual, {k: expected[k] for k in ('W_in', 'b_in')})

    def test_cli_guards_and_checkpoint_roundtrip(self):
        command = ['train_mpn.py', '--input-mode', 'diag_mtrace', '--feedback', 'exact_spatial']
        with patch.object(sys, 'argv', command):
            args = train_mpn._parse_args()
        self.assertEqual(args.input_mode, 'diag_mtrace')
        for extra in (['--net', 'mpn1'], ['--feedback', 'direct_fa'], ['--feedback', 'layerwise_fa']):
            with patch.object(sys, 'argv', command + extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train_mpn._parse_args()
        net, cfg = make_net()
        cfg['feedback_mode'] = 'direct_fa'
        with self.assertRaisesRegex(ValueError, 'exact_spatial'):
            mpn.DeepMultiPlasticNet(cfg)
        cfg['feedback_mode'] = 'exact_spatial'
        x, y, mask, _ = data()
        expected = net.sequence_gradients(x, y, mask)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.pt'
            torch.save(dict(net_params=cfg, state_dict=net.state_dict(), learning_rule=net.learning_rule), path)
            with contextlib.redirect_stdout(io.StringIO()):
                loaded = train_mpn.load_net(path, device=torch.device('cpu'), dtype=torch.double)
            self.assertEqual(loaded.input_mode, 'diag_mtrace')
            actual = loaded.sequence_gradients(x, y, mask)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


class TestPairedInputMode(unittest.TestCase):
    mapping = {'bptt': 'exact', 'local_direct': 'three_factor',
               'local_diag_rflo': 'diag_mtrace'}

    def test_pairing_matches_explicit_algorithms_after_cloning(self):
        x, y, mask, active = data()
        for kind in ('hebb_assoc', 'hebb_pre'):
            base, _ = make_net(rule='bptt', input_mode='paired', kind=kind)
            for rule, expected_mode in self.mapping.items():
                with self.subTest(rule=rule, kind=kind):
                    net = copy.deepcopy(base)
                    net.learning_rule = rule
                    reference = copy.deepcopy(net)
                    reference.input_mode = expected_mode
                    self.assertEqual(net.resolved_input_mode, expected_mode)
                    kwargs = {} if rule == 'bptt' else {'update_masks': active}
                    expected = reference.sequence_gradients(x, y, mask, **kwargs)
                    with patch.object(net, 'bptt_gradients', wraps=net.bptt_gradients) as bptt:
                        actual = net.sequence_gradients(x, y, mask, **kwargs)
                    self.assertEqual(bptt.call_count, int(rule == 'bptt'))
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                    for key, parameter in net._trainable_params().items():
                        torch.testing.assert_close(parameter.grad, actual[key], rtol=0, atol=0)

    def test_direct_gradient_methods_resolve_the_requested_rule(self):
        # A diagnostic can call a local method on a net configured as BPTT.
        x, y, mask, _ = data()
        net, _ = make_net(rule='bptt', input_mode='paired', kind='hebb_pre')
        for rule in ('local_direct', 'local_diag_rflo'):
            reference = copy.deepcopy(net)
            reference.input_mode = self.mapping[rule]
            actual = getattr(net, rule + '_gradients')(x, y, mask)
            expected = getattr(reference, rule + '_gradients')(x, y, mask)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_cli_defaults_and_guards(self):
        command = ['train_mpn.py', '--input-mode', 'paired', '--feedback', 'exact_spatial']
        with patch.object(sys, 'argv', command):
            args = train_mpn._parse_args()
        self.assertEqual(args.input_mode, 'paired')
        self.assertEqual(args.rules, ['bptt', 'local_diag_rflo', 'local_direct'])
        for extra in (['--net', 'mpn1'], ['--feedback', 'direct_fa'],
                      ['--feedback', 'layerwise_fa'], ['--dfa'],
                      ['--rules', 'local_exact_rowlocal']):
            with self.subTest(extra=extra), patch.object(sys, 'argv', command + extra):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train_mpn._parse_args()
        with patch.object(sys, 'argv', command + ['--rules', 'local_direct']):
            self.assertEqual(train_mpn._parse_args().rules, ['local_direct'])

    def test_model_guards_and_legacy_mapping(self):
        with self.assertRaisesRegex(ValueError, 'does not support'):
            make_net(rule='local_exact_rowlocal', input_mode='paired')
        net, cfg = make_net(rule='bptt', input_mode='paired')
        net.learning_rule = 'local_exact_rowlocal'
        x, y, mask, _ = data()
        with self.assertRaisesRegex(ValueError, 'does not support'):
            net.sequence_gradients(x, y, mask)
        for feedback in ('direct_fa', 'layerwise_fa'):
            with self.assertRaisesRegex(ValueError, 'exact_spatial'):
                mpn.DeepMultiPlasticNet({**cfg, 'feedback_mode': feedback})
        with self.assertRaisesRegex(ValueError, 'input embedding'):
            mpn.DeepMultiPlasticNet({**cfg, 'input_layer_add': False})
        for rule in (*self.mapping, 'local_exact_rowlocal'):
            self.assertEqual(mpn.resolve_input_mode('match', rule),
                             'exact' if rule == 'bptt' else 'three_factor')
            self.assertEqual(mpn.resolve_input_mode('diag_mtrace', rule),
                             'exact' if rule == 'bptt' else 'diag_mtrace')

    def test_checkpoint_reload_resolves_per_rule(self):
        x, y, mask, _ = data()
        base, cfg = make_net(rule='bptt', input_mode='paired')
        with tempfile.TemporaryDirectory() as directory:
            for rule, expected_mode in self.mapping.items():
                net = copy.deepcopy(base)
                net.learning_rule = rule
                expected = net.sequence_gradients(x, y, mask)
                path = Path(directory) / f'{rule}.pt'
                # Also covers older checkpoint structure: base BPTT in net_params,
                # actual rule at the top level, restored after construction.
                torch.save(dict(net_params=cfg, state_dict=net.state_dict(),
                                learning_rule=rule, resolved_input_mode=expected_mode), path)
                with contextlib.redirect_stdout(io.StringIO()):
                    loaded = train_mpn.load_net(path, device=torch.device('cpu'), dtype=torch.double)
                self.assertEqual(loaded.input_mode, 'paired')
                self.assertEqual(loaded.resolved_input_mode, expected_mode)
                actual = loaded.sequence_gradients(x, y, mask)
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_training_checkpoint_and_log_record_resolved_modes(self):
        # Exercise production cloning and persistence with zero optimizer steps.
        _, net_params = make_net(rule='bptt', input_mode='paired')
        x, y, mask, _ = data()
        cfg = train_mpn._cfg()
        cfg.rules_to_run = list(self.mapping)
        cfg.input_mode = 'paired'
        cfg.input_normalize = False
        cfg.n_datasets = 0
        cfg.log_grad_align = False
        cfg.device, cfg.dtype = torch.device('cpu'), torch.double
        cfg.build_params = lambda: ({}, {}, net_params)
        cfg.net_factory = lambda params, verbose: mpn.DeepMultiPlasticNet(params, verbose=False)
        cfg.task = SimpleNamespace(init_params=lambda *params: params,
                                   valid_batch=lambda *args: (x, y, mask))
        cfg.save_nets = True
        with tempfile.TemporaryDirectory() as directory:
            cfg.ckpt_dir = directory
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                train_mpn.tc.run_seed(cfg, 13, [])
            for rule, expected_mode in self.mapping.items():
                checkpoint = torch.load(train_mpn.tc.ckpt_path(cfg, rule, 13), weights_only=False)
                self.assertEqual(checkpoint['input_mode'], 'paired')
                self.assertEqual(checkpoint['net_params']['input_mode'], 'paired')
                self.assertEqual(checkpoint['net_params']['learning_rule'], rule)
                self.assertEqual(checkpoint['resolved_input_mode'], expected_mode)
                self.assertEqual(checkpoint['run_id'], cfg.run_id)
                self.assertIn(f'{rule}: input_mode=paired, resolved_input_mode={expected_mode}',
                              log.getvalue())
            self.assertEqual(Path(train_mpn.tc.ckpt_path(cfg, rule, 13)).relative_to(directory),
                             Path(cfg.run_id) / 'seed13' / f'{rule}.pt')


if __name__ == '__main__':
    unittest.main()
