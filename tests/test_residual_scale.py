"""Fixed MP-branch gains: independent autograd references and runner coverage."""
import contextlib
import io
import json
from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401
import mpn
import train_common as tc
import train_mpn
import wandb_logging
import test_dfa as dfa
import test_input_modulation_trace as imt
import test_local_readouts as heads
from test_residual_correction import full_sequence_oracle


RULES = ('local_direct', 'local_diag_rflo', 'local_exact_rowlocal', 'bptt')


class TestResidualScale(unittest.TestCase):
    def assert_gradients(self, actual, expected):
        for key, value in expected.items():
            with self.subTest(parameter=key):
                torch.testing.assert_close(actual[key], value, rtol=2e-9, atol=2e-11)

    def test_default_and_no_residual_compatibility(self):
        x, y, mask, _ = heads.data(B=2, T=3)
        for residual, rule in product((False, True), RULES):
            with self.subTest(residual=residual, rule=rule):
                cfg = heads.make_cfg(residual=residual, rule=rule)
                explicit = heads.make_net(cfg)
                del cfg['residual_scale']  # legacy checkpoint/config
                implicit = heads.make_net(cfg)
                self.assertEqual(implicit.residual_scale, 1.)
                self.assertEqual(explicit.state_dict().keys(), implicit.state_dict().keys())
                a = explicit.sequence_gradients(x, y, mask)
                b = implicit.sequence_gradients(x, y, mask)
                self.assert_gradients(a, {k: b[k] for k in implicit._trainable_params()})
                # The independent formula at scale=1 is the original h_in + a.
                _, _, expected, _ = heads.independent_forward(implicit, x)
                torch.testing.assert_close(a['outputs'], expected)

    def test_bptt_matches_independent_full_unroll(self):
        x, y, mask, _ = dfa.data()
        for scale, mode in product((0., .35, 1.7), ('linear', 'hard', 'smooth')):
            with self.subTest(scale=scale, mode=mode):
                net = dfa.make_net(rule='bptt', residual=True, residual_scale=scale,
                                   bounded=mode == 'hard', smooth_scale=.4 if mode == 'smooth' else None)
                with torch.no_grad():
                    for layer in net.mp_layers:
                        layer.eta.fill_(20.)
                expected_out, expected = full_sequence_oracle(net, x * 3, y, mask, None)
                actual = net.sequence_gradients(x * 3, y, mask)
                torch.testing.assert_close(actual['outputs'], expected_out)
                self.assert_gradients(actual, expected)
                # Width-changing last block has neither a skip nor a branch gain.
                self.assertEqual(net._branch_scales, (scale, scale, 1.))

    def test_raw_hebbian_write_and_trace_at_zero_and_nonunit_scales(self):
        x, _, _, _ = heads.data(B=2, T=1)
        for scale in (0., .25, 1.5):
            net = heads.make_net(heads.make_cfg(widths=(4,), residual_scale=scale,
                                               rule='local_diag_rflo'))
            net.reset_state(B=len(x))
            layer = net.mp_layers[0]
            with torch.no_grad():
                emb = torch.tanh(net.W_initial_linear(x[:, 0]))
                old = layer.M.clone()
                a = torch.tanh((layer.W * (1 + old) * emb[:, None]).sum(-1) + layer.b)
                expected = dfa.write(layer, old, emb, a, torch.ones(len(x), dtype=x.dtype))
                _, stream, _ = net.network_step(x[:, 0])
                torch.testing.assert_close(stream[1], emb + scale * a)
                torch.testing.assert_close(layer.M, expected, rtol=1e-12, atol=1e-13)
                self.assertGreater(float((expected - old).norm()), 0.)
            # At one layer the raw trace is independent of the output gain.
            _, y, mask, _ = heads.data(B=2, T=1)
            net.sequence_gradients(x, y, mask)
            eta, _ = layer._eta_lam_full()
            raw = eta[None] * (1 - a.square())[:, :, None] * emb[:, None].square() * (1 + old)
            gate = ((layer.M_pre >= -1.) & (layer.M_pre <= 1.)).to(raw)
            torch.testing.assert_close(layer.A, raw * gate, rtol=1e-10, atol=1e-12)

    def test_zero_scale_identity_and_zero_mp_gradients(self):
        x, y, mask, _ = heads.data(B=2, T=4)
        for rule, signal, input_mode in product(RULES, ('global', 'local_readout'),
                                                 ('match', 'diag_mtrace')):
            with self.subTest(rule=rule, signal=signal, input_mode=input_mode):
                net = heads.make_net(heads.make_cfg(rule=rule, signal=signal,
                                     input_mode=input_mode, residual_scale=0.))
                result = net.sequence_gradients(x, y, mask)
                expected = F.linear(torch.tanh(net.W_initial_linear(x)), net.W_output, net.b_output)
                torch.testing.assert_close(result['outputs'], expected)
                for n in range(3):
                    suffix = '' if n == 0 else str(n)
                    for prefix in ('W', 'b'):
                        self.assertEqual(torch.count_nonzero(result[prefix + suffix]), 0)
                self.assertGreater(float(result['W_in'].norm()), 0.)
                self.assertGreater(float(result['W_output'].norm()), 0.)
                if signal == 'local_readout' and rule != 'bptt':
                    self.assertGreater(float(net.head_W0.grad.norm()), 0.)

    def test_no_plasticity_local_rules_match_bptt(self):
        x, y, mask, _ = heads.data(B=2, T=4)
        for rule in RULES[:-1]:
            net = heads.make_net(heads.make_cfg(rule=rule, residual_scale=.4))
            with torch.no_grad():
                for layer in net.mp_layers:
                    layer.eta.zero_()
            expected = net.bptt_gradients(x, y, mask)
            actual = net.sequence_gradients(x, y, mask)
            self.assert_gradients(actual, {k: expected[k] for k in net._trainable_params()})

    def test_local_head_rowlocal_and_embedding_match_isolated_autograd(self):
        x, y, mask, um = heads.data(B=2, T=4)
        for scale, custom in product((.4, 1.3), (False, True)):
            with self.subTest(scale=scale, custom=custom):
                loss_fn = imt.fourth_power_loss if custom else heads.mse_loss_and_grad
                net = heads.make_net(heads.make_cfg(signal='local_readout', rule='local_exact_rowlocal',
                                                   residual_scale=scale))
                streams, states, output, aux = heads.independent_forward(net, x, um)
                head_grads, errors = heads.head_oracle(net, streams, y, mask, loss_fn)
                kwargs = {'loss_and_grad': loss_fn} if custom else {}
                result = net.sequence_gradients(x, y, mask, update_masks=um, **kwargs)
                _, go = loss_fn(output, y, mask)
                signals = [heads.head_signal(net, n, e) for n, e in enumerate(errors)]
                signals.append(go @ net.W_output)
                expected = {}
                for n, ell in enumerate(signals):
                    suffix = '' if n == 0 else str(n)
                    expected['W' + suffix], expected['b' + suffix] = heads.layer_oracle(
                        net, n, ell, streams, um)
                expected['W_in'], expected['b_in'] = heads.embed_oracle(net, signals[0], x, streams, states)
                expected['W_output'] = torch.einsum('bta,bti->ai', go, torch.stack([s[-1] for s in streams], 1))
                self.assert_gradients(result, expected)
                self.assert_gradients({k: p.grad for k, p in net._aux_params().items()}, head_grads)
                torch.testing.assert_close(result['outputs'], output)
                for actual, wanted in zip(result['aux_outputs'], aux):
                    torch.testing.assert_close(actual, wanted)

    def test_direct_head_gradient_freezes_modulation_history(self):
        x, y, mask, um = heads.data(B=2, T=4)
        net = heads.make_net(heads.make_cfg(signal='local_readout', residual_scale=.3))
        streams, states, output, _ = heads.independent_forward(net, x, um)
        _, errors = heads.head_oracle(net, streams, y, mask, heads.mse_loss_and_grad)
        _, go = heads.mse_loss_and_grad(output, y, mask)
        signals = [heads.head_signal(net, n, e) for n, e in enumerate(errors)] + [go @ net.W_output]
        actual = net.sequence_gradients(x, y, mask, update_masks=um)
        expected = {}
        for n, layer in enumerate(net.mp_layers):
            W = layer.W.detach().clone().requires_grad_()
            b = layer.b.detach().clone().requires_grad_()
            objective = 0
            for t in range(x.shape[1]):
                inp = streams[t][n]
                a = torch.tanh((W * (1 + states[t][n]) * inp[:, None]).sum(-1) + b)
                objective = objective + (signals[n][:, t] * (inp + .3 * a)).sum()
            suffix = '' if n == 0 else str(n)
            expected['W' + suffix], expected['b' + suffix] = torch.autograd.grad(objective, (W, b))
        self.assert_gradients(actual, expected)

    def test_dfa_diagonal_and_rowlocal_against_scalar_and_row_oracles(self):
        x, y, mask, um = dfa.data()
        for scale, diag, bias in product((0., .4, 1.3), (False, True), ('direct', 'exact')):
            with self.subTest(scale=scale, diag=diag, bias=bias):
                net = dfa.make_net(rule='local_diag_rflo' if diag else 'local_exact_rowlocal',
                                   bias=bias, residual=True, residual_scale=scale, smooth_scale=.4)
                expected_out, expected = dfa.oracle(net, x, y, mask, um, diag=diag)
                actual = net.sequence_gradients(x, y, mask, update_masks=um)
                torch.testing.assert_close(actual['outputs'], expected_out)
                self.assert_gradients(actual, expected)

    def test_spatial_signals_via_independent_autograd(self):
        x, _, _, _ = heads.data(B=2, T=1)
        for feedback in ('exact_spatial', 'layerwise_fa'):
            net = heads.make_net(heads.make_cfg(residual_scale=.3, feedback=feedback))
            net.reset_state(B=len(x))
            h = [torch.tanh(net.W_initial_linear(x[:, 0])).detach().requires_grad_()]
            phi = []
            for n, layer in enumerate(net.mp_layers):
                z = (layer.W.detach() * (1 + layer.M) * h[-1][:, None]).sum(-1) + layer.b.detach()
                if feedback == 'layerwise_fa':
                    # Same z value, but an independently constructed random backward map.
                    z = z.detach() + F.linear(h[-1] - h[-1].detach(), getattr(net, net._B_inter_names[n]))
                a = torch.tanh(z)
                phi.append((1 - a.square()).detach())
                h.append(h[-1] + .3 * a)
            go = torch.randn(len(x), net.n_output, dtype=x.dtype)
            readout = net.W_output if feedback == 'exact_spatial' else net.B_feedback
            expected = torch.autograd.grad(((go @ readout) * h[-1]).sum(), h)
            actual = net._same_time_boundary_signals(go, phi, True)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)

    def test_mixed_signal_alpha_is_distinct_from_residual_scale(self):
        x, y, mask, _ = heads.data(B=2, T=3)
        results = []
        for alpha in (0., 1., .3):
            net = heads.make_net(heads.make_cfg(signal='mixed', alpha=alpha, residual_scale=.4))
            results.append(net.sequence_gradients(x, y, mask))
        for key in net._trainable_params():
            torch.testing.assert_close(results[2][key], results[0][key] + .3 * (results[1][key] - results[0][key]))

    def test_two_step_cross_correction_matches_full_unroll(self):
        x, y, mask, _ = dfa.data()
        x, y, mask = x[:, :2] * 3, y[:, :2], mask[:, :2]
        um = torch.tensor([[.25, 1.], [1., 0.]], dtype=x.dtype)
        for scale, mode, kind in product((0., .4, 1.3), ('linear', 'hard', 'smooth'),
                                         ('hebb_assoc', 'hebb_pre')):
            with self.subTest(scale=scale, mode=mode, kind=kind):
                net = dfa.make_net(residual=True, residual_scale=scale, kinds=[kind] * 3,
                                   bounded=mode == 'hard', smooth_scale=.4 if mode == 'smooth' else None)
                net.feedback_mode, net.cross_layer_steps = 'exact_spatial', 1
                for layer in net.mp_layers:
                    layer.set_plasticity_freeze(torch.tensor([0]), torch.tensor([0]))
                    with torch.no_grad():
                        layer.eta.fill_(20.)
                out, expected = full_sequence_oracle(net, x, y, mask, um)
                actual = net.sequence_gradients(x, y, mask, update_masks=um)
                torch.testing.assert_close(actual['outputs'], out)
                self.assert_gradients(actual, {k: v for k, v in expected.items() if k not in ('W_in', 'b_in')})

    def test_input_trace_against_independent_column_graphs(self):
        x, y, mask, um = imt.data()
        for scale, mode, kind in product((0., .4), ('linear', 'hard', 'smooth'),
                                         ('hebb_assoc', 'hebb_pre')):
            with self.subTest(scale=scale, mode=mode, kind=kind):
                net, _ = imt.make_net(widths=(3, 3), residual=True, residual_scale=scale,
                                      write_mode=mode, kind=kind)
                net.mp_layers[0].set_plasticity_freeze(torch.tensor([0]), torch.tensor([1]))
                net.set_input_norm_stats(x.reshape(-1, 2))
                expected = {}
                for j in range(3):
                    out, part = imt.oracle(net, x, y, mask, um, column=j, custom=True)
                    for key, value in part.items():
                        expected[key] = expected.get(key, 0) + value
                actual = net.sequence_gradients(x, y, mask, update_masks=um, loss_and_grad=imt.fourth_power_loss)
                torch.testing.assert_close(actual['outputs'], out)
                self.assert_gradients(actual, expected)

    def test_capped_rflo_fused_and_explicit_paths_agree(self):
        x, y, mask, _ = heads.data(B=2, T=1)
        cfg = heads.make_cfg(rule='local_diag_rflo', residual_scale=.4)
        cfg['ml_params']['rflo_trace_rho'] = .99
        net = heads.make_net(cfg)
        fused = net.sequence_gradients(x, y, mask)
        net.cross_layer_steps = 1  # at T=1 this changes only the computation path
        explicit = net.sequence_gradients(x, y, mask)
        self.assert_gradients(fused, {k: explicit[k] for k in net._trainable_params()})

    def test_cli_validation_and_checkpoint_roundtrip(self):
        for scale in ('nan', 'inf', '-.1'):
            with self.subTest(scale=scale), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train_mpn._parse_args(['--residual-scale=' + scale])
                with self.assertRaises(ValueError):
                    heads.make_net(heads.make_cfg(residual_scale=float(scale)))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_mpn._parse_args(['--no-residual', '--residual-scale', '.5'])
        with self.assertRaises(ValueError):
            heads.make_net(heads.make_cfg(residual=False, residual_scale=.5))
        for scale in (0., .5, 1.5):
            self.assertEqual(train_mpn._parse_args(['--residual', '--residual-scale', str(scale)]).residual_scale, scale)
        cfg = heads.make_cfg(signal='local_readout', residual_scale=.5)
        net = heads.make_net(cfg)
        x, y, mask, _ = heads.data(B=2, T=3)
        expected = net.sequence_gradients(x, y, mask)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save({'net_params': cfg, 'state_dict': net.state_dict()}, path)
            saved = torch.load(path, weights_only=False)
            restored = heads.make_net(saved['net_params'])
            restored.load_state_dict(saved['state_dict'])
            actual = restored.sequence_gradients(x, y, mask)
            self.assertEqual(restored.residual_scale, .5)
            self.assert_gradients(actual, {k: expected[k] for k in net._trainable_params()})

    def test_cli_wiring_and_saved_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            def inspect(cfg):
                self.assertEqual(cfg.residual_scale, .5)
                self.assertEqual(cfg.build_params()[2]['residual_scale'], .5)
                cfg.fig_dir = cfg.ckpt_dir = cfg.data_dir = directory
                path = tc.save_config(cfg, str(Path(directory) / 'config.json'))
                saved = json.loads(Path(path).read_text())
                self.assertEqual(saved['residual_scale'], .5)
                self.assertEqual(saved['net_params']['residual_scale'], .5)
                self.assertEqual(wandb_logging._base_config(cfg, 'test')['residual_scale'], .5)
                cfg.rules_to_run = ['local_direct']
                runs = {'local_direct': {'train': [[.1]], 'valid': [[.2]]}}
                agg = {'local_direct': {split: {'mean': [.1], 'std': [0.]} for split in ('train', 'valid')}}
                path = str(Path(directory) / 'data.npz')
                tc.save_plot_data(cfg, [0], runs, agg, path=path)
                with np.load(path) as saved:
                    self.assertEqual(float(saved['residual_scale']), .5)
                with patch.object(tc, 'plot') as plot:
                    tc.replot_from_npz(cfg, path)
                    self.assertIn('residual scale=0.5', plot.call_args.args[4])
                first_id = cfg.run_id
                train_mpn.RESIDUAL_SCALE = .25
                self.assertNotEqual(first_id, train_mpn._cfg().run_id)

            argv = ['train_mpn.py', '--hidden', '3', '3', '--residual', '--residual-scale', '.5',
                    '--learning-signal', 'local_readout']
            with patch.dict(train_mpn.__dict__), patch.object(sys, 'argv', argv), \
                    patch.object(tc, 'run_experiment', side_effect=inspect) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                train_mpn.main()
                run.assert_called_once()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
