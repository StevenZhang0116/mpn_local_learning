"""Regression checks for residual depth settings and delayed credit sources."""
import contextlib
import io
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from test_dfa import make_net, data, write
import mpn
import train_mpn


def full_sequence_oracle(net, x, y, mask, update_mask):
    """Independent differentiable forward; no production signal/trace routines.

    `write` implements the defining masked plastic-state recurrence directly.
    This oracle supports update masks, unlike the public BPTT entry point.
    """
    states = [layer.M_init[None].expand(len(x), -1, -1).clone()
              for layer in net.mp_layers]
    outputs = []
    for t in range(x.shape[1]):
        h = torch.tanh(net.W_initial_linear(x[:, t]))
        for n, layer in enumerate(net.mp_layers):
            a = torch.tanh((layer.W * (1 + states[n]) * h[:, None]).sum(-1)
                           + layer.b)
            active = (torch.ones(len(x), dtype=x.dtype, device=x.device)
                      if update_mask is None else update_mask[:, t])
            states[n] = write(layer, states[n], h, a, active)
            frozen = getattr(layer, '_plasticity_freeze_mask', None)
            if frozen is not None:
                states[n][:, frozen[0], frozen[1]] = layer.M_init[frozen[0], frozen[1]]
            h = net.residual_scale * a + h if net._residual_at[n] else a
        outputs.append(F.linear(h, net.W_output, net.b_output))
    outputs = torch.stack(outputs, 1)
    loss = ((outputs - y) * mask).square().mean()
    params = net._trainable_params()
    gradients = torch.autograd.grad(loss, list(params.values()))
    return outputs, dict(zip(params, gradients))


class TestResidualCorrection(unittest.TestCase):
    def test_scaled_tanh_bptt_matches_full_oracle(self):
        inputs, labels, masks, _ = data()
        for scale in (.4, 2.):
            with self.subTest(scale=scale):
                net = make_net(rule='bptt', residual=True, smooth_scale=scale)
                with torch.no_grad():
                    for layer in net.mp_layers:
                        layer.eta.fill_(20)
                expected_out, expected = full_sequence_oracle(net, inputs, labels, masks, None)
                actual = net.sequence_gradients(inputs, labels, masks)
                torch.testing.assert_close(actual['outputs'], expected_out)
                for key, value in expected.items():
                    torch.testing.assert_close(actual[key], value, rtol=1e-9, atol=1e-10)

    def test_frozen_two_step_gradients_match_full_oracle(self):
        inputs, labels, masks, _ = data()
        inputs, labels, masks = inputs[:, :2] * 3, labels[:, :2], masks[:, :2]
        update_mask = torch.tensor([[.25, 1.], [1., 0.]], dtype=inputs.dtype)
        for bounded, smooth_scale in ((False, None), (True, None), (False, .4)):
            for residual in (False, True):
                for kinds in (None, ['hebb_pre'] * 3,
                              ['hebb_assoc', 'hebb_pre', 'hebb_assoc']):
                    with self.subTest(bounded=bounded, smooth_scale=smooth_scale, residual=residual, kinds=kinds):
                        net = make_net(residual=residual, kinds=kinds, bounded=bounded, smooth_scale=smooth_scale)
                        net.feedback_mode = 'exact_spatial'
                        net.cross_layer_steps = 1
                        for layer in net.mp_layers:
                            post_indices = torch.arange(layer.n_output)
                            pre_indices = post_indices % layer.n_input
                            layer.set_plasticity_freeze(post_indices, pre_indices)
                            with torch.no_grad():
                                layer.eta.fill_(100 if bounded else 20 if smooth_scale is not None else .18)
                        expected_out, expected = full_sequence_oracle(
                            net, inputs, labels, masks, update_mask)
                        actual = net.sequence_gradients(
                            inputs, labels, masks, update_masks=update_mask)
                        torch.testing.assert_close(actual['outputs'], expected_out)
                        for key, value in expected.items():
                            if key not in ('W_in', 'b_in'):
                                torch.testing.assert_close(actual[key], value,
                                                           rtol=1e-9, atol=1e-10)

    def test_bounded_two_step_gradients(self):
        inputs, labels, masks, _ = data()
        inputs, labels, masks = inputs[:, :2] * 3, labels[:, :2], masks[:, :2]
        for residual in (False, True):
            for kinds in (None, ['hebb_pre'] * 3,
                          ['hebb_assoc', 'hebb_pre', 'hebb_assoc']):
                for correction in (0, 1):
                    with self.subTest(residual=residual, kinds=kinds, correction=correction):
                        net = make_net(residual=residual, kinds=kinds, bounded=True)
                        net.feedback_mode = 'exact_spatial'
                        net.cross_layer_steps = correction
                        with torch.no_grad():
                            for layer in net.mp_layers:
                                layer.eta.fill_(100)
                        update_mask = torch.tensor([[.25, 1.], [1., 0.]], dtype=inputs.dtype)
                        expected_out, expected = full_sequence_oracle(
                            net, inputs, labels, masks, update_mask)
                        actual = net.sequence_gradients(
                            inputs, labels, masks, update_masks=update_mask)
                        torch.testing.assert_close(actual['outputs'], expected_out)
                        for key in ('W', 'b', 'W1', 'b1', 'W2', 'b2', 'W_output', 'b_output'):
                            if correction or key in ('W2', 'b2', 'W_output', 'b_output'):
                                torch.testing.assert_close(actual[key], expected[key],
                                                           rtol=1e-9, atol=1e-10)
                        self.assertTrue(any((layer.M.abs() == 1).any() for layer in net.mp_layers))
                        for layer in net.mp_layers:
                            self.assertTrue((layer.M.abs() <= 1).all())

    def test_two_step_mp_gradients_match_full_oracle(self):
        x, y, mask, _ = data()
        x, y, mask = x[:, :2], y[:, :2], mask[:, :2]
        update_masks = [
            None,
            torch.ones(2, 2, dtype=x.dtype),
            # Different previous/current flags catch an off-by-one mask error.
            torch.tensor([[0., 1.], [1., 0.]], dtype=x.dtype),
            torch.tensor([[.25, 1.], [.75, 0.]], dtype=x.dtype),
        ]
        for residual in (False, True):
            for kinds in (None, ['hebb_pre'] * 3,
                          ['hebb_assoc', 'hebb_pre', 'hebb_assoc']):
                for mask_idx, update_mask in enumerate(update_masks):
                    with self.subTest(residual=residual, kinds=kinds, mask=mask_idx):
                        net = make_net(bias='exact', residual=residual, kinds=kinds)
                        net.feedback_mode = 'exact_spatial'
                        net.cross_layer_steps = 1
                        expected_out, expected = full_sequence_oracle(
                            net, x, y, mask, update_mask)
                        actual = net.sequence_gradients(
                            x, y, mask, update_masks=update_mask)
                        torch.testing.assert_close(actual['outputs'], expected_out,
                                                   rtol=1e-10, atol=1e-11)
                        for key, value in expected.items():
                            if key in ('W_in', 'b_in'):
                                # The one-step correction only credits MP parameters.
                                continue
                            torch.testing.assert_close(actual[key], value,
                                                       rtol=1e-9, atol=1e-10)

    def test_disabled_writes_produce_no_correction(self):
        x, y, mask, _ = data()
        update_mask = torch.zeros(x.shape[:2], dtype=x.dtype)
        for rule in ('local_exact_rowlocal', 'local_diag_rflo', 'local_direct'):
            with self.subTest(rule=rule):
                net = make_net(rule=rule, residual=True)
                net.feedback_mode = 'exact_spatial'
                base = net.sequence_gradients(x, y, mask, update_masks=update_mask)
                net.cross_layer_steps = 1
                corrected = net.sequence_gradients(x, y, mask, update_masks=update_mask)
                for key in net._trainable_params():
                    torch.testing.assert_close(corrected[key], base[key],
                                               rtol=1e-10, atol=1e-11)

    def test_cli_residual_at_every_depth(self):
        for depth in (1, 2, 3):
            for enabled in (False, True):
                with self.subTest(depth=depth, residual=enabled):
                    argv = ['train_mpn.py', '--dfa', '--hidden', *(['3'] * depth),
                            '--residual' if enabled else '--no-residual']
                    with patch.object(sys, 'argv', argv):
                        args = train_mpn._parse_args()
                    with patch.multiple(train_mpn, N_HIDDEN=args.hidden,
                                        MP_RESIDUAL=args.residual, NET_TYPE='dmpn'):
                        _, _, cfg = train_mpn.build_params()
                        self.assertEqual(cfg['mp_residual'], enabled)
                        self.assertEqual(train_mpn._cfg().mp_residual, enabled)
                        with contextlib.redirect_stdout(io.StringIO()):
                            net = mpn.DeepMultiPlasticNet(cfg)
                        self.assertEqual(net._residual_at, [enabled] * depth)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
