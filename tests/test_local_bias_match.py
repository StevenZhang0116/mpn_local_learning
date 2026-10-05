"""Rule-matched MP bias eligibility: numerical equivalence and provenance.

Run from tests/: python -m unittest test_local_bias_match -v
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import _bootstrap  # noqa: F401
import mpn
import train_common as tc
import train_mpn as tm
from test_local_readouts import (
    data, make_cfg, make_net, independent_forward, head_oracle,
    head_signal, layer_oracle, mse_loss_and_grad,
)


# Resolved label per rule under 'match' ('autograd' is a resolved label for bptt,
# never a requestable policy) and the explicit REQUESTED policy that reproduces
# each rule's match behavior (bptt ignores the bias policy, so any valid one).
MODES = {'local_direct': 'direct', 'local_diag_rflo': 'direct',
         'local_exact_rowlocal': 'exact', 'bptt': 'autograd'}
REQUESTED = {rule: ('direct' if mode == 'autograd' else mode) for rule, mode in MODES.items()}


class TestLocalBiasMatch(unittest.TestCase):
    def assert_grads_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            if expected[key] is not None:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_cli_mapping_and_legacy_defaults(self):
        args = tm._parse_args(['--learning-signal', 'local_readout', '--no-residual',
                               '--hidden', *['128'] * 7, '--rules', *MODES,
                               '--local-bias-mode', 'match', '--input-mode', 'match'])
        self.assertEqual(args.local_bias_mode, 'match')
        self.assertEqual(args.input_mode, 'match')
        self.assertFalse(args.residual)
        self.assertEqual(tm._parse_args([]).local_bias_mode, 'match')
        self.assertEqual(tm._parse_args(['--dfa']).local_bias_mode, 'direct')
        self.assertEqual(tm._parse_args(['--local-bias-mode', 'direct']).local_bias_mode, 'direct')
        self.assertEqual(tm._parse_args(['--dfa', '--local-bias-mode', 'match']).local_bias_mode,
                         'match')
        for rule, mode in MODES.items():
            self.assertEqual(mpn.resolve_local_bias_mode('match', rule), mode)
        with self.assertRaises(ValueError):
            make_net(make_cfg(bias='invalid'))
        with self.assertRaises(ValueError):
            mpn.resolve_local_bias_mode('match', 'invalid')

    def test_deep_match_equals_explicit_rules(self):
        x, y, mask, um = data()
        for signal, feedback in (('global', 'exact_spatial'), ('global', 'direct_fa'),
                                 ('local_readout', 'exact_spatial'), ('mixed', 'exact_spatial')):
            for rule, explicit in MODES.items():
                with self.subTest(signal=signal, feedback=feedback, rule=rule):
                    net = make_net(make_cfg(signal=signal, feedback=feedback, rule=rule, bias='match'))
                    ref = make_net(make_cfg(signal=signal, feedback=feedback, rule=rule,
                                            bias=REQUESTED[rule]))
                    kw = {} if rule == 'bptt' else {'update_masks': um}
                    actual = net.sequence_gradients(x, y, mask, **kw)
                    expected = ref.sequence_gradients(x, y, mask, **kw)
                    self.assert_grads_equal(actual, expected)
                    self.assertEqual(net.resolved_local_bias_modes, [explicit] * 3)
                    self.assertTrue(all(layer.local_bias_mode == 'match' for layer in net.mp_layers))
                    if rule in ('local_diag_rflo', 'local_exact_rowlocal'):
                        self.assertTrue(all((layer.Q is None) == (explicit == 'direct')
                                            for layer in net.mp_layers))

    def test_rule_switches_and_direct_gradient_entry_points(self):
        x, y, mask, _ = data()
        methods = {'local_exact_rowlocal': 'local_gradients',
                   'local_diag_rflo': 'local_diag_rflo_gradients',
                   'local_direct': 'local_direct_gradients'}
        net = make_net(make_cfg(rule='bptt', signal='local_readout', bias='match'))
        # Call an arbitrary method while the declared rule remains BPTT. In
        # particular, diag must not accidentally allocate the row-local bias Q.
        for rule in ('local_exact_rowlocal', 'local_diag_rflo', 'local_direct',
                     'local_exact_rowlocal', 'local_diag_rflo'):
            ref = make_net(make_cfg(rule=rule, signal='local_readout', bias=REQUESTED[rule]))
            self.assert_grads_equal(getattr(net, methods[rule])(x, y, mask),
                                    getattr(ref, methods[rule])(x, y, mask))
            self.assertEqual(net.learning_rule, 'bptt')
        # Cloning, changing the declared rule and gradient diagnostics must also
        # preserve the requested policy; resolution cannot be a one-time preset.
        for rule in (*MODES, 'local_exact_rowlocal', 'local_diag_rflo'):
            net.learning_rule = rule
            clone = copy.deepcopy(net)
            ref = make_net(make_cfg(rule=rule, signal='local_readout', bias=REQUESTED[rule]))
            self.assert_grads_equal(clone.sequence_gradients(x, y, mask),
                                    ref.sequence_gradients(x, y, mask))
            self.assertEqual(clone.resolved_local_bias_modes, [MODES[rule]] * 3)
            self.assertTrue(all(l.local_bias_mode == 'match' for l in clone.mp_layers))

    def test_rowlocal_bias_matches_independent_autograd(self):
        x, y, mask, um = data()
        net = make_net(make_cfg(signal='local_readout', rule='local_exact_rowlocal', bias='match'))
        streams, _, outputs, _ = independent_forward(net, x, um)
        _, errors = head_oracle(net, streams, y, mask, mse_loss_and_grad)
        _, output_error = mse_loss_and_grad(outputs, y, mask)
        signals = [head_signal(net, n, e) for n, e in enumerate(errors)] + [output_error @ net.W_output]
        actual = net.sequence_gradients(x, y, mask, update_masks=um)
        for n, signal in enumerate(signals):
            weight, bias = layer_oracle(net, n, signal, streams, um)
            suffix = '' if n == 0 else str(n)
            torch.testing.assert_close(actual['W' + suffix], weight, rtol=1e-9, atol=1e-11)
            torch.testing.assert_close(actual['b' + suffix], bias, rtol=1e-9, atol=1e-11)

    def test_single_layer_match_and_bptt_exactness(self):
        x, y, mask, _ = data()
        params = make_cfg(widths=(4,), rule='bptt', bias='match', residual=False)
        params['net_type'] = 'mpn1'
        np.random.seed(5)
        with contextlib.redirect_stdout(io.StringIO()):
            net = mpn.MultiPlasticNet(params, verbose=False).double()
        with torch.no_grad():
            net.mp_layer.eta.fill_(.3)
            net.mp_layer.lam.fill_(.7)
        initial = copy.deepcopy(net)
        bptt = net.sequence_gradients(x, y, mask)
        for rule, mode in MODES.items():
            net = copy.deepcopy(initial)
            net.learning_rule = rule
            ref = copy.deepcopy(net)
            ref.mp_layer.local_bias_mode = REQUESTED[rule]
            actual = net.sequence_gradients(x, y, mask)
            self.assert_grads_equal(actual, ref.sequence_gradients(x, y, mask))
            self.assertEqual(net.resolved_local_bias_modes, [mode])
            if rule == 'local_exact_rowlocal':
                for key in net._trainable_params():
                    torch.testing.assert_close(actual[key], bptt[key], rtol=1e-9, atol=1e-11)

    def test_training_metadata_and_reload(self):
        x, y, mask, _ = data(B=4, T=5)
        params = make_cfg(signal='local_readout', rule='bptt', bias='match')
        cfg = tm._cfg()
        cfg.rules_to_run = list(MODES)
        cfg.learning_signal = cfg.signal_mode = 'local_readout'
        cfg.input_mode, cfg.cross_layer_steps = 'match', 0
        cfg.input_normalize, cfg.log_grad_align, cfg.save_nets = False, True, True
        cfg.n_datasets, cfg.batch = 1, 4
        cfg.device, cfg.dtype = torch.device('cpu'), torch.double
        cfg.build_params = lambda: ({}, {}, copy.deepcopy(params))
        cfg.net_factory = lambda p, verbose: mpn.DeepMultiPlasticNet(p, verbose=False)
        cfg.task = SimpleNamespace(init_params=lambda *p: p, valid_batch=lambda *a: (x,y,mask),
                                   train_batch=lambda *a: (x,y,mask), accuracy=lambda *a, **k: 0.,
                                   loss_and_grad=None)
        with tempfile.TemporaryDirectory() as directory:
            cfg.ckpt_dir = cfg.fig_dir = cfg.data_dir = directory
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                tc.run_seed(cfg, 17, [0])
                config_path = tc.save_config(cfg)
            self.assertEqual(json.loads(Path(config_path).read_text())['net_params']['ml_params']['local_bias_mode'],
                             'match')
            for rule, mode in MODES.items():
                path = tc.ckpt_path(cfg, rule, 17)
                saved = torch.load(path, weights_only=False)
                self.assertEqual(saved['net_params']['ml_params']['local_bias_mode'], 'match')
                self.assertEqual(saved['resolved_local_bias_modes'], [mode] * 3)
                if rule != 'bptt':
                    self.assertIn('bias=' + mode, saved['effective_config'])
                with contextlib.redirect_stdout(io.StringIO()):
                    loaded = tm.load_net(path, device=torch.device('cpu'), dtype=torch.double)
                self.assertEqual(loaded.resolved_local_bias_modes, [mode] * 3)
                loaded.learning_rule = 'local_exact_rowlocal'
                self.assertEqual(loaded.resolved_local_bias_modes, ['exact'] * 3)
                loaded.sequence_gradients(x,y,mask)
                self.assertTrue(all(l.Q is not None for l in loaded.mp_layers))
                loaded.learning_rule = 'local_diag_rflo'
                loaded.sequence_gradients(x,y,mask)
                self.assertTrue(all(l.Q is None for l in loaded.mp_layers))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
