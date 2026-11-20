"""Signed top-k routing at MP boundaries (top1_abs and top<k>_abs), including its
spatial and temporal VJPs."""
import contextlib
import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

import _bootstrap
import mpn
import tasks
import train_common
import train_mpn
from test_mp_input_norm import build_deep, build_single, make_data, mp_keys, rel


def script_rule_gradients(task_name, routing, bias, cross, hidden=(6, 5), batch=4):
    """Every parameter gradient of local_diag_rflo and local_exact_rowlocal at the
    same weights and on the same batch, with the nets built as train_common.run_seed
    builds them: train_mpn's parameter builder, then one copy per rule. No residual
    connections; the gradients are taken under the task's own loss.
    Returns ({name: grad} for diag, the same for row-local, the two rules'
    resolved bias modes)."""
    with patch.multiple(train_mpn, N_HIDDEN=list(hidden), RULESET=task_name, MP_RESIDUAL=False,
                        ETA=[0.3], LAM=0.99, MP_INPUT_ROUTING=routing,
                        CROSS_LAYER_STEPS=cross, LOCAL_BIAS_MODE=bias):
        np.random.seed(0)
        torch.manual_seed(0)
        task = tasks.make_task(task_name)
        task_params, train_params, net_params = task.init_params(*train_mpn.build_params())
        with contextlib.redirect_stdout(io.StringIO()):
            base = mpn.DeepMultiPlasticNet(net_params, verbose=False).double()
        x, y, mask = task.train_batch(task_params, train_params, batch,
                                      torch.device('cpu'), torch.float64)
    loss_fn = getattr(task, 'loss_and_grad', None)
    loss_kw = {} if loss_fn is None else {'loss_and_grad': loss_fn}
    grads, bias_modes = [], []
    for rule in ('local_diag_rflo', 'local_exact_rowlocal'):
        net = copy.deepcopy(base)
        net.learning_rule = rule
        net.sequence_gradients(x, y, mask, **loss_kw)
        grads.append({name: p.grad.clone() for name, p in net.named_parameters()
                      if p.grad is not None})
        bias_modes.append(net.resolved_local_bias_modes)
    return grads[0], grads[1], bias_modes


class TestInputRouting(unittest.TestCase):
    def test_dense_and_scalar_inputs_keep_the_identity_path(self):
        net = build_single(3, 5, 2, norm='none')
        x = torch.randn(4, 3, dtype=torch.double)
        self.assertEqual(net.mp_input_routing, 'dense')
        for routing, values in [('dense', x), ('top1_abs', x[:, :1])]:
            net.mp_input_routing = routing
            output, context = net._prepare_mp_input(values, layer_idx=0)
            self.assertIs(output, values)
            self.assertIsNone(context)

    def test_signed_winner_ties_zero_and_rms_backward(self):
        x = torch.tensor([[-3., 3., 1.], [0., 0., 0.], [.2, -.5, 2.]],
                         dtype=torch.double, requires_grad=True)
        upstream = torch.tensor([[1., 2., 3.]], dtype=torch.double).expand_as(x)
        mask = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=x.dtype)
        for norm in ('none', 'rms'):
            net = build_deep([3, 4, 2], norm=norm, embed=False)
            net.mp_input_routing = 'top1_abs'
            value, context = net._prepare_mp_input(x, layer_idx=0)
            normalized = x if norm == 'none' else x / (
                x.square().mean(-1, keepdim=True) + net.mp_input_norm_eps).sqrt()
            expected = mask * normalized
            torch.testing.assert_close(value, expected)
            expected_grad, = torch.autograd.grad(expected, x, upstream)
            torch.testing.assert_close(net._mp_input_backward(upstream, context), expected_grad)
            auto_grad, = torch.autograd.grad(value, x, upstream)
            torch.testing.assert_close(auto_grad, expected_grad)

    def test_single_layer_reads_writes_and_evaluates_routed_sequence(self):
        net = build_single(4, 5, 2, norm='none')
        net.mp_input_routing = 'top1_abs'
        x, y, mask = make_data(5, 7, 4, 2)
        selector = torch.nn.functional.one_hot(x.abs().argmax(-1), num_classes=4)
        dense = build_single(4, 5, 2, norm='none')
        dense.load_state_dict(net.state_dict())
        ref = dense.bptt_gradients(x * selector, y, mask)
        local = net.local_diag_rflo_gradients(x, y, mask)
        for key in ref:
            torch.testing.assert_close(local[key], ref[key], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(train_mpn.forward_outputs(net, x), ref['outputs'])
        # The public step also updates the synaptic state with the routed pre.
        net.reset_state(B=len(x))
        dense.reset_state(B=len(x))
        for t in range(x.shape[1]):
            out, _ = net.network_step(x[:, t])
            ref_out, _ = dense.network_step((x * selector)[:, t])
            torch.testing.assert_close(out, ref_out)
            torch.testing.assert_close(net.mp_layer.M, dense.mp_layer.M)

    def test_deep_mp_weight_and_bias_equivalence(self):
        x, y, mask = make_data(8, 7, 5, 3)
        for cross in (0, 1):
            for norm in ('none', 'rms'):
                with self.subTest(cross=cross, norm=norm):
                    net = build_deep([5, 6, 6, 3], norm=norm, residual=True, cross=cross)
                    net.mp_input_routing = 'top1_abs'
                    diag = net.local_diag_rflo_gradients(x, y, mask)
                    row = net.local_gradients(x, y, mask)
                    for layer in net.mp_layers:
                        diag_P = torch.diag_embed(layer.P.diagonal(dim1=-2, dim2=-1))
                        self.assertTrue(torch.equal(layer.P, diag_P))
                    for key in mp_keys(net):
                        torch.testing.assert_close(diag[key], row[key], rtol=1e-11, atol=1e-12)
                    torch.testing.assert_close(train_mpn.forward_outputs(net, x), diag['outputs'])

    def test_training_script_rules_agree_only_with_exact_bias(self):
        # The same equivalence for the nets the training script builds, without
        # residual connections, under a cross-entropy loss (two-hop recall) and
        # under the default masked MSE (wh_mid).
        def mp_bias(name):
            return name.startswith('mp_layer') and name.endswith('.b')

        for task_name in ('twohop_K4_d2', 'wh_mid_T12'):
            for cross in (0, 1):
                with self.subTest(task=task_name, cross=cross, bias='exact'):
                    diag, row, modes = script_rule_gradients(task_name, 'top1_abs', 'exact', cross)
                    self.assertEqual(modes, [['exact', 'exact'], ['exact', 'exact']])
                    self.assertEqual(diag.keys(), row.keys())
                    self.assertEqual(sum(map(mp_bias, row)), 2)
                    for name in row:
                        torch.testing.assert_close(diag[name], row[name], rtol=1e-10, atol=1e-12)
                with self.subTest(task=task_name, cross=cross, bias='match'):
                    # The CLI default: diagonal RFLO resolves to the direct bias
                    # rule, so the MP biases differ and nothing else does.
                    diag, row, modes = script_rule_gradients(task_name, 'top1_abs', 'match', cross)
                    self.assertEqual(modes, [['direct', 'direct'], ['exact', 'exact']])
                    for name in row:
                        if mp_bias(name):
                            self.assertGreater(rel(diag[name], row[name]), 1e-3, name)
                        else:
                            torch.testing.assert_close(diag[name], row[name],
                                                       rtol=1e-10, atol=1e-12)
            with self.subTest(task=task_name, routing='dense'):
                # Control: with dense inputs the MP weight gradients do differ, so
                # the agreement above comes from the routing.
                diag, row, _ = script_rule_gradients(task_name, 'dense', 'exact', 0)
                for name in ('mp_layer1.W', 'mp_layer2.W'):
                    self.assertGreater(rel(diag[name], row[name]), 1e-3, name)

    def test_one_hop_uses_previous_winner_with_residual_and_rms(self):
        x, y, mask = make_data(16, 2, 5, 3, seed=13)
        net = build_deep([5, 6, 6, 6, 3], norm='rms', residual=True, cross=1,
                         embed=False, seed=9)
        net.mp_input_routing = 'top1_abs'
        ref = net.bptt_gradients(x, y, mask)
        diag = net.local_diag_rflo_gradients(x, y, mask)
        for key in mp_keys(net):
            torch.testing.assert_close(diag[key], ref[key], rtol=1e-10, atol=1e-12)
        net.cross_layer_steps = 0
        truncated = net.local_diag_rflo_gradients(x, y, mask)
        self.assertGreater((truncated['W'] - ref['W']).norm().item(), 1e-6)

    def test_routed_embedding_column_trace_with_identity_skip(self):
        # With one MP layer and one active presynaptic column at each step,
        # diag_mtrace also captures this embedding's full temporal derivative.
        net = build_deep([5, 6, 3], norm='none', input_mode='diag_mtrace', residual=True)
        net.mp_input_routing = 'top1_abs'
        x, y, mask = make_data(6, 8, 5, 3)
        ref = net.bptt_gradients(x, y, mask)
        diag = net.local_diag_rflo_gradients(x, y, mask)
        for key in ('W_in', 'b_in'):
            torch.testing.assert_close(diag[key], ref[key], rtol=1e-10, atol=1e-12)

    def test_cli_default_and_run_metadata(self):
        self.assertEqual(train_mpn._parse_args([]).mp_input_routing, 'dense')
        args = train_mpn._parse_args(['--mp-input-routing', 'top1_abs'])
        with patch.object(train_mpn, 'MP_INPUT_ROUTING', args.mp_input_routing):
            self.assertEqual(train_mpn.build_params()[2]['mp_input_routing'], 'top1_abs')
            cfg = train_mpn._cfg()
            self.assertEqual(cfg.mp_input_routing, 'top1_abs')
            self.assertIn('_routing-top1_abs', train_common.param_tag(cfg))


class TestTopKRouting(unittest.TestCase):
    """top<k>_abs for k > 1: the same read / write / backward machinery as top1_abs,
    without the diagonal-trace equivalence."""

    def test_routing_names(self):
        self.assertIsNone(mpn.mp_input_routing_k('dense'))
        self.assertEqual(mpn.mp_input_routing_k('top1_abs'), 1)
        self.assertEqual(mpn.mp_input_routing_k('top12_abs'), 12)
        for bad in ('top0_abs', 'top_abs', 'topk_abs', 'top1', 'top-1_abs', 'top01_abs',
                    'TOP1_ABS', 'top1.5_abs', 'top2_abs ', '', None, 1):
            with self.subTest(name=bad):
                with self.assertRaises(ValueError):
                    mpn.mp_input_routing_k(bad)
        # A net refuses an unknown name when it is built, the CLI before that.
        with self.assertRaises(ValueError):
            script_rule_gradients('wh_mid_T12', 'top0_abs', 'exact', 0)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            train_mpn._parse_args(['--mp-input-routing', 'top0_abs'])
        args = train_mpn._parse_args(['--mp-input-routing', 'top3_abs'])
        with patch.object(train_mpn, 'MP_INPUT_ROUTING', args.mp_input_routing):
            self.assertEqual(train_mpn.build_params()[2]['mp_input_routing'], 'top3_abs')
            self.assertIn('_routing-top3_abs', train_common.param_tag(train_mpn._cfg()))

    def test_top_k_winners_ties_zero_and_rms_backward(self):
        x = torch.tensor([[1., -1., 1., .5], [0., 0., 0., 0.], [.2, -.5, 2., -3.], [4., 0., 0., 0.]],
                         dtype=torch.double, requires_grad=True)
        # Among equal magnitudes the lower index wins, on every device.
        winners = [[0, 1], [0, 1], [3, 2], [0, 1]]
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            self.assertEqual(mpn._top_abs_indices(x.detach().to(device), 2).tolist(), winners)
            self.assertEqual(mpn._top_abs_indices(x.detach().to(device), 1).tolist(),
                             [[0], [0], [3], [0]])
        mask = torch.tensor([[1., 1., 0., 0.], [1., 1., 0., 0.], [0., 0., 1., 1.], [1., 1., 0., 0.]],
                            dtype=x.dtype)
        upstream = torch.tensor([[1., 2., 3., 4.]], dtype=torch.double).expand_as(x)
        for norm in ('none', 'rms'):
            net = build_deep([4, 4, 2], norm=norm, embed=False)
            net.mp_input_routing = 'top2_abs'
            value, context = net._prepare_mp_input(x, layer_idx=0)
            self.assertEqual(context.winners.tolist(), winners)
            normalized = x if norm == 'none' else x / (
                x.square().mean(-1, keepdim=True) + net.mp_input_norm_eps).sqrt()
            expected = mask * normalized
            torch.testing.assert_close(value, expected)
            expected_grad, = torch.autograd.grad(expected, x, upstream)
            torch.testing.assert_close(net._mp_input_backward(upstream, context), expected_grad)
            auto_grad, = torch.autograd.grad(value, x, upstream)
            torch.testing.assert_close(auto_grad, expected_grad)

    def test_k_at_least_the_width_is_dense(self):
        net = build_single(3, 5, 2, norm='none')
        x = torch.randn(4, 3, dtype=torch.double)
        for routing in ('top3_abs', 'top7_abs'):
            net.mp_input_routing = routing
            output, context = net._prepare_mp_input(x, layer_idx=0)
            self.assertIs(output, x)
            self.assertIsNone(context)
        # Both MP layers of the script-built net read 6 components.
        dense = script_rule_gradients('wh_mid_T12', 'dense', 'exact', 1)
        whole = script_rule_gradients('wh_mid_T12', 'top6_abs', 'exact', 1)
        fewer = script_rule_gradients('wh_mid_T12', 'top5_abs', 'exact', 1)
        for rule in (0, 1):
            for name in dense[rule]:
                self.assertTrue(torch.equal(whole[rule][name], dense[rule][name]), name)
            self.assertGreater(rel(fewer[rule]['mp_layer1.W'], dense[rule]['mp_layer1.W']), 1e-3)

    def test_single_layer_reads_writes_and_evaluates_top_k(self):
        net = build_single(4, 5, 2, norm='none')
        net.mp_input_routing = 'top2_abs'
        x, y, mask = make_data(5, 7, 4, 2)
        keep = torch.zeros_like(x).scatter_(-1, x.abs().topk(2, dim=-1).indices, 1.0)
        dense = build_single(4, 5, 2, norm='none')
        dense.load_state_dict(net.state_dict())
        ref = dense.bptt_gradients(x * keep, y, mask)
        # Row-local is exact for a single MP layer; BPTT differentiates the routing itself.
        row = net.local_gradients(x, y, mask)
        bptt = net.bptt_gradients(x, y, mask)
        for key in ref:
            torch.testing.assert_close(row[key], ref[key], rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(bptt[key], ref[key], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(train_mpn.forward_outputs(net, x), ref['outputs'])
        # Two components active at once couple, so the diagonal trace is approximate.
        diag = net.local_diag_rflo_gradients(x, y, mask)
        self.assertGreater(rel(diag['W'], row['W']), 1e-3)
        net.reset_state(B=len(x))
        dense.reset_state(B=len(x))
        for t in range(x.shape[1]):
            out, _ = net.network_step(x[:, t])
            ref_out, _ = dense.network_step((x * keep)[:, t])
            torch.testing.assert_close(out, ref_out)
            torch.testing.assert_close(net.mp_layer.M, dense.mp_layer.M)

    def test_deep_top_k_keeps_exactness_but_not_the_diagonal_equivalence(self):
        x, y, mask = make_data(8, 7, 5, 3)
        for cross in (0, 1):
            for norm in ('none', 'rms'):
                with self.subTest(cross=cross, norm=norm):
                    net = build_deep([5, 6, 6, 3], norm=norm, residual=True, cross=cross)
                    net.mp_input_routing = 'top2_abs'
                    ref = net.bptt_gradients(x, y, mask)
                    diag = net.local_diag_rflo_gradients(x, y, mask)
                    row = net.local_gradients(x, y, mask)
                    # The top MP layer's row-local gradient is still BPTT's.
                    for key in ('W1', 'b1'):
                        torch.testing.assert_close(row[key], ref[key], rtol=1e-10, atol=1e-12)
                    for layer in net.mp_layers:
                        diag_P = torch.diag_embed(layer.P.diagonal(dim1=-2, dim2=-1))
                        self.assertGreater((layer.P - diag_P).abs().max().item(), 1e-6)
                    for key in mp_keys(net, bias=False):
                        self.assertGreater(rel(diag[key], row[key]), 1e-3, key)
                    torch.testing.assert_close(train_mpn.forward_outputs(net, x), row['outputs'])

    def test_one_hop_uses_previous_winners_with_residual_and_rms(self):
        # At T=2 one cross-layer step makes row-local exact for every MP layer, so
        # this checks the routed backward maps at t and at t-1 against autograd.
        x, y, mask = make_data(16, 2, 5, 3, seed=13)
        net = build_deep([5, 6, 6, 6, 3], norm='rms', residual=True, cross=1,
                         embed=False, seed=9)
        net.mp_input_routing = 'top2_abs'
        ref = net.bptt_gradients(x, y, mask)
        row = net.local_gradients(x, y, mask)
        for key in mp_keys(net):
            torch.testing.assert_close(row[key], ref[key], rtol=1e-10, atol=1e-12)
        net.cross_layer_steps = 0
        truncated = net.local_gradients(x, y, mask)
        self.assertGreater((truncated['W'] - ref['W']).norm().item(), 1e-6)

    def test_training_script_rules_differ_for_k_above_one(self):
        for task_name in ('twohop_K4_d2', 'wh_mid_T12'):
            with self.subTest(task=task_name):
                diag, row, _ = script_rule_gradients(task_name, 'top2_abs', 'exact', 1)
                for name in ('mp_layer1.W', 'mp_layer2.W'):
                    self.assertGreater(rel(diag[name], row[name]), 1e-3, name)


if __name__ == '__main__':
    unittest.main()
