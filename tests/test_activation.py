"""The network activation (train_mpn --activation): the derivative the local rules use
equals autograd's for every offered activation, the exactness statements that hold
for tanh hold for the others, and the choice reaches the run's metadata."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import _bootstrap  # noqa: F401
import mpn
import net_helpers
import tasks
import train_common
import train_mpn

RTOL, ATOL = 1e-9, 1e-12


def build(task_name, activation, *, hidden=(6, 5), batch=4, cross=0, routing='dense'):
    """A float64 deep MPN from train_mpn's parameter builder (no residual, exact bias
    rule) and one training batch. Returns (net, (x, y, mask), loss_kw)."""
    with patch.multiple(train_mpn, N_HIDDEN=list(hidden), RULESET=task_name, MP_RESIDUAL=False,
                        ETA=[0.3], LAM=0.99, ACTIVATION=activation, MP_INPUT_ROUTING=routing,
                        CROSS_LAYER_STEPS=cross, LOCAL_BIAS_MODE='exact'):
        np.random.seed(0)
        torch.manual_seed(0)
        task = tasks.make_task(task_name)
        task_params, train_params, net_params = task.init_params(*train_mpn.build_params())
        with contextlib.redirect_stdout(io.StringIO()):
            net = mpn.DeepMultiPlasticNet(net_params, verbose=False).double()
        batch = task.train_batch(task_params, train_params, batch, torch.device('cpu'), torch.float64)
    loss_fn = getattr(task, 'loss_and_grad', None)
    return net, batch, ({} if loss_fn is None else {'loss_and_grad': loss_fn})


def gradients(base, rule, batch, loss_kw):
    """{parameter name: gradient} of one rule, on a copy of `base` (as run_seed does)."""
    net = copy.deepcopy(base)
    net.learning_rule = rule
    net.sequence_gradients(*batch, **loss_kw)
    return {name: p.grad.clone() for name, p in net.named_parameters() if p.grad is not None}


def assert_same(test, got, want, names):
    for name in names:
        test.assertGreater(want[name].norm().item(), 0.0, name)
        torch.testing.assert_close(got[name], want[name], rtol=RTOL, atol=ATOL, msg=name)


class TestActivation(unittest.TestCase):
    def test_derivative_equals_autograd_including_at_zero(self):
        points = torch.tensor([-3.0, -0.5, -1e-9, 0.0, 1e-9, 0.5, 3.0], dtype=torch.float64)
        for name in train_mpn.ACTIVATIONS:
            with self.subTest(activation=name):
                fn, _, derivative = net_helpers.get_activation_function(name)
                x = points.clone().requires_grad_(True)
                auto, = torch.autograd.grad(fn(x).sum(), x)
                torch.testing.assert_close(derivative(points), auto, rtol=0, atol=1e-12)
        # 'relu' is the CLI spelling of the existing 'ReLU'.
        self.assertEqual(net_helpers.get_activation_function('relu'),
                         net_helpers.get_activation_function('ReLU'))
        relu = net_helpers.get_activation_function('relu')[0]
        self.assertEqual(relu(points).tolist(), [0.0, 0.0, 0.0, 0.0, 1e-9, 0.5, 3.0])

    def test_cli_default_choices_and_metadata(self):
        self.assertEqual(train_mpn._parse_args([]).activation, 'tanh')
        self.assertEqual(train_mpn.build_params()[2]['activation'], 'tanh')
        self.assertNotIn('_act-', train_common.param_tag(train_mpn._cfg()))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            train_mpn._parse_args(['--activation', 'tanh_re'])
        for name in train_mpn.ACTIVATIONS:
            args = train_mpn._parse_args(['--activation', name])
            with patch.object(train_mpn, 'ACTIVATION', args.activation):
                self.assertEqual(train_mpn.build_params()[2]['activation'], name)
                cfg = train_mpn._cfg()
                self.assertEqual(cfg.activation, name)
                self.assertEqual(f'_act-{name}' in train_common.param_tag(cfg), name != 'tanh')

    def test_net_uses_the_activation_in_the_embedding_and_the_mp_layers(self):
        net, (x, _, _), _ = build('wh_mid_T12', 'relu')
        self.assertEqual(net.act, 'relu')
        net.reset_state(B=x.shape[0])
        with torch.no_grad():
            _, h, z, phi_p, embed_pre, blocks, _ = net._forward_local_stack(
                x[:, 0, :], return_blocks=True, return_norm=True)
        torch.testing.assert_close(h[0], torch.relu(embed_pre))
        for n in range(len(net.mp_layers)):
            torch.testing.assert_close(blocks[n], torch.relu(z[n]))
            torch.testing.assert_close(phi_p[n], (z[n] > 0).to(z[n].dtype))
        tanh_net, _, _ = build('wh_mid_T12', 'tanh')
        self.assertEqual(tanh_net.act, 'tanh')
        self.assertEqual({k: v.shape for k, v in net.state_dict().items()},
                         {k: v.shape for k, v in tanh_net.state_dict().items()})

    def test_single_mp_layer_row_local_is_bptt(self):
        for name in train_mpn.ACTIVATIONS:
            for task_name in ('wh_mid_T12', 'twohop_K4_d2'):
                with self.subTest(activation=name, task=task_name):
                    net, batch, loss_kw = build(task_name, name, hidden=(6,))
                    ref = gradients(net, 'bptt', batch, loss_kw)
                    row = gradients(net, 'local_exact_rowlocal', batch, loss_kw)
                    assert_same(self, row, ref, ('mp_layer1.W', 'mp_layer1.b', 'W_output'))

    def test_deep_top_layer_and_two_step_cross_layer_are_exact(self):
        for name in train_mpn.ACTIVATIONS:
            with self.subTest(activation=name, check='top layer'):
                net, batch, loss_kw = build('twohop_K4_d2', name)
                ref = gradients(net, 'bptt', batch, loss_kw)
                row = gradients(net, 'local_exact_rowlocal', batch, loss_kw)
                assert_same(self, row, ref, ('mp_layer2.W', 'mp_layer2.b', 'W_output'))
            with self.subTest(activation=name, check='T=2, one cross-layer step'):
                net, batch, loss_kw = build('wh_mid_T2', name, cross=1)
                ref = gradients(net, 'bptt', batch, loss_kw)
                row = gradients(net, 'local_exact_rowlocal', batch, loss_kw)
                assert_same(self, row, ref, ('mp_layer1.W', 'mp_layer1.b',
                                             'mp_layer2.W', 'mp_layer2.b'))

    def test_relu_agrees_with_bptt_at_exactly_zero_preactivations(self):
        # Zero biases and blank input steps put pre-activations at exactly 0, where
        # ReLU has a kink: BPTT (autograd) and the local rules must use the same
        # derivative there.
        net, (x, y, mask), loss_kw = build('wh_mid_T12', 'relu', hidden=(6,))
        with torch.no_grad():
            net.W_initial_linear.bias.zero_()
            net.mp_layer1.b.zero_()
        x = x.clone()
        x[:, 3:6, :] = 0.0
        batch = (x, y, mask)
        ref = gradients(net, 'bptt', batch, loss_kw)
        row = gradients(net, 'local_exact_rowlocal', batch, loss_kw)
        assert_same(self, row, ref, ('mp_layer1.W', 'mp_layer1.b', 'W_output'))
        # The batch does sit on the kink: a ReLU whose autograd splits the tie at 0
        # (torch.maximum) changes BPTT's gradient.
        split = copy.deepcopy(net)
        split.act_fn = lambda v: torch.maximum(v, torch.zeros_like(v))
        other = gradients(split, 'bptt', batch, loss_kw)
        self.assertGreater((other['mp_layer1.b'] - ref['mp_layer1.b']).norm().item(), 1e-8)

    def test_top1_routing_equivalence_holds_for_every_activation(self):
        # ReLU makes many inputs exactly zero, so the winner is often picked among ties.
        for name in train_mpn.ACTIVATIONS:
            for cross in (0, 1):
                with self.subTest(activation=name, cross=cross):
                    net, batch, loss_kw = build('twohop_K4_d2', name, cross=cross,
                                                routing='top1_abs')
                    diag = gradients(net, 'local_diag_rflo', batch, loss_kw)
                    row = gradients(net, 'local_exact_rowlocal', batch, loss_kw)
                    self.assertEqual(diag.keys(), row.keys())
                    assert_same(self, diag, row, ('mp_layer1.W', 'mp_layer1.b',
                                                 'mp_layer2.W', 'mp_layer2.b'))

    def test_short_run_saves_and_restores_the_activation(self):
        rules = ['bptt', 'local_direct', 'local_diag_rflo', 'local_exact_rowlocal']
        with tempfile.TemporaryDirectory() as directory, patch.multiple(
                train_mpn, CKPT_DIR=directory, FIG_DIR=directory, DATA_DIR=directory,
                SAVE_NETS=True, N_HIDDEN=[6, 5], N_DATASETS=6, BATCH=4, N_RUNS=1,
                RULESET='twohop_K4_d2', DEVICE=torch.device('cpu'), LOG_GRAD_ALIGN=True,
                MP_RESIDUAL=False, ETA=[0.3], LAM=0.99, CROSS_LAYER_STEPS=1,
                ACTIVATION='relu', RULES_TO_RUN=rules, _RUN_IDS={}):
            cfg = train_mpn._cfg()
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                curves, _ = train_common.run_seed(cfg, 3, [0, 5])
                config = json.loads(Path(train_common.save_config(cfg)).read_text())
            self.assertEqual(config['activation'], 'relu')
            self.assertEqual(config['net_params']['activation'], 'relu')
            for rule in rules:
                self.assertTrue(np.isfinite(curves[rule]['train']).all(), rule)
                self.assertTrue(np.isfinite(curves[rule]['valid']).all(), rule)
                path = train_common.ckpt_path(cfg, rule, 3)
                self.assertEqual(torch.load(path, weights_only=False)['activation'], 'relu')
                net = train_mpn.load_net(path, device=torch.device('cpu'))
                self.assertEqual(net.act, 'relu')
                self.assertIs(net.act_fn, net_helpers.relu_fn)


if __name__ == '__main__':
    unittest.main()
