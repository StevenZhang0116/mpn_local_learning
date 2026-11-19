"""The in-place exact row-local trace update (MultiPlasticLayer._advance_exact_trace)
against the out-of-place arithmetic it replaces. CPU float64.
Run: python -m unittest test_exact_trace_inplace -v
"""
from itertools import product
import unittest

import torch

import _bootstrap  # noqa: F401  (core/ and scripts/{training,tasks}/ on sys.path)
from test_additive_mpn import make_net, data

MODES = ('linear', 'hard', 'smooth')        # unbounded, hard-clipped, scaled-tanh write
MASKS = (None, 'binary', 'fractional')      # per-step update mask (B, T)


def update_mask(kind):
    active = data()[3]
    return None if kind is None else (active > .5).to(active.dtype) if kind == 'binary' else active


def gradients(inplace, mask_kind, freeze=False, **net_kw):
    """Exact row-local gradients plus every layer's final P / Q / M."""
    net, _ = make_net(rule='local_exact_rowlocal', **net_kw)
    for layer in net.mp_layers:
        layer.exact_trace_inplace = inplace
        if freeze:
            post = torch.arange(layer.n_output)
            layer.set_plasticity_freeze(post, post % layer.n_input)
    x, y, mask, _ = data()
    um = update_mask(mask_kind)
    out = net.sequence_gradients(x, y, mask, **({} if um is None else {'update_masks': um}))
    result = {k: v for k, v in out.items() if torch.is_tensor(v)}
    for n, layer in enumerate(net.mp_layers):
        result.update({f'P{n}': layer.P, f'Q{n}': layer.Q, f'M{n}': layer.M})
    return result


def one_step(layer, mask=None, inplace=True, seed=3):
    """One explicit eligibility / trace / write step from a random trace; returns
    the trace before (a copy), the trace object before, E, x and the write gate."""
    g = torch.Generator().manual_seed(seed)
    B = 2
    layer.exact_trace_inplace = inplace
    layer.reset_state(B=B)
    layer.reset_local_learning_state(B=B)
    layer.P.copy_(torch.randn(layer.P.shape, generator=g, dtype=torch.double) * .3)
    x = torch.randn(B, layer.n_input, generator=g, dtype=torch.double)
    post = torch.tanh(torch.randn(B, layer.n_output, generator=g, dtype=torch.double))
    before, obj = layer.P.clone(), layer.P
    with torch.no_grad():
        E, R = layer.compute_exact_rowlocal_eligibility(x, 1 - post.square())
        layer.update_exact_rowlocal_traces(x, E, R, update_mask=mask)
        layer.update_M_matrix_local_fast(x, post, update_mask=mask)
        gate = layer._modulation_write_derivative()
    return before, obj, E, x, gate


class TestExactTraceInPlace(unittest.TestCase):
    def test_gradients_and_traces_match_the_reference_arithmetic(self):
        cases = live = 0
        nets = [dict(kind='mpn1', widths=(3,))]
        nets += [dict(kind='dmpn', widths=(3, 2, 3), cross=cross, residual=residual)
                 for cross, residual in product((0, 1), (False, True))]
        for net_kw, mode, mp_type, mask_kind, freeze in product(
                nets, MODES, ('mult', 'add'), MASKS, (False, True)):
            kw = dict(net_kw, mode=mode, mp_type=mp_type)
            with self.subTest(mask=mask_kind, freeze=freeze, **kw):
                new = gradients(True, mask_kind, freeze, **kw)
                ref = gradients(False, mask_kind, freeze, **kw)
                self.assertEqual(new.keys(), ref.keys())
                # a live final trace (zero only where a hard write clips every entry)
                live += any(float(v.abs().max()) > 0 for k, v in ref.items() if k.startswith('P'))
                for key in ref:
                    torch.testing.assert_close(new[key], ref[key], rtol=1e-12, atol=1e-14,
                                               msg=lambda m, key=key: f'{key}: {m}')
                cases += 1
        self.assertEqual(cases, 5 * 3 * 2 * 3 * 2)
        self.assertGreater(live, .9 * cases)

    def test_hard_write_follows_the_masked_recursion(self):
        """P <- gate * (m (lam P + eta x_J E^I_i) + (1-m) P): the update mask mixes
        new and old BEFORE the clip gate; written out here with plain tensor ops."""
        for mode, mask in product(('linear', 'hard'), (None, torch.tensor([1., 0.]),
                                                       torch.tensor([.3, .8]))):
            with self.subTest(mode=mode, mask=mask):
                net, _ = make_net(kind='mpn1', widths=(3,), mode=mode, mp_type='mult')
                layer = net.mp_layers[0]
                mask = None if mask is None else mask.double()
                before, _, E, x, gate = one_step(layer, mask)
                eta, lam = layer._eta_lam_full()
                new = lam[None, :, None, :] * before + eta[None, :, None, :] * E[..., None] * x[:, None, None, :]
                if mask is not None:
                    m = mask.view(-1, 1, 1, 1)
                    new = m * new + (1 - m) * before
                if gate is not None:
                    self.assertFalse(bool(gate.all()), 'no entry is clipped: the gate is untested')
                    new = new * gate.unsqueeze(-2)
                torch.testing.assert_close(layer.P, new, rtol=1e-12, atol=1e-14)

    def test_smooth_write_follows_the_gated_recursion(self):
        """P <- tanh'(M_pre / s) * (lam P + eta x_J E^I_i), no update mask."""
        net, _ = make_net(kind='mpn1', widths=(3,), mode='smooth', mp_type='mult')
        layer = net.mp_layers[0]
        before, _, E, x, gate = one_step(layer)
        eta, lam = layer._eta_lam_full()
        new = lam[None, :, None, :] * before + eta[None, :, None, :] * E[..., None] * x[:, None, None, :]
        torch.testing.assert_close(layer.P, new * gate.unsqueeze(-2), rtol=1e-12, atol=1e-14)

    def test_trace_buffer_is_reused(self):
        """In place means the SAME tensor object survives the step; the reference
        arithmetic (switched off, or a smooth write with an update mask) builds a
        new one."""
        mask = torch.tensor([.3, 1.], dtype=torch.double)
        for mode, um, inplace, reused in (
                ('linear', None, True, True), ('hard', None, True, True),
                ('hard', mask, True, True), ('smooth', None, True, True),
                ('smooth', mask, True, False),       # needs the old trace after the gate
                ('hard', None, False, False), ('smooth', None, False, False)):
            with self.subTest(mode=mode, masked=um is not None, inplace=inplace):
                net, _ = make_net(kind='mpn1', widths=(3,), mode=mode, mp_type='mult')
                layer = net.mp_layers[0]
                _, obj, _, _, _ = one_step(layer, um, inplace=inplace)
                self.assertIs(layer.P is obj, reused)

    def test_sequence_gradients_takes_the_in_place_path(self):
        """The training entry point runs under no_grad, so every step of every
        layer is in place: the reference arithmetic, which hands its new trace to
        _prepare_local_trace('P', ...), is never reached. Switched off, it runs
        once per step and layer. (Both give the same numbers, so only this count
        would notice a silent fall-back to the slow path.)"""
        x, y, mask, _ = data()
        for inplace, per_layer in ((True, 0), (False, x.shape[1])):
            with self.subTest(inplace=inplace):
                net, _ = make_net(rule='local_exact_rowlocal', kind='dmpn', widths=(3, 2, 3),
                                  mode='hard', mp_type='mult')
                calls = []
                for layer in net.mp_layers:
                    layer.exact_trace_inplace = inplace

                    def spy(name, *args, _original=layer._prepare_local_trace):
                        if name == 'P':
                            calls.append(name)
                        return _original(name, *args)
                    layer._prepare_local_trace = spy
                net.sequence_gradients(x, y, mask)
                self.assertEqual(len(calls), per_layer * len(net.mp_layers))

    def test_autograd_callers_keep_the_reference_path(self):
        """With autograd recording, P is rebuilt out of place, so a graph through
        the trace stays valid: d(sum P_2)/dW exists and backward() does not raise."""
        net, _ = make_net(kind='mpn1', widths=(3,), mode='linear', mp_type='mult')
        layer = net.mp_layers[0]
        layer.reset_state(B=2)
        layer.reset_local_learning_state(B=2)
        g = torch.Generator().manual_seed(7)
        first = layer.P
        for _ in range(2):
            x = torch.randn(2, layer.n_input, generator=g, dtype=torch.double)
            post = torch.tanh(torch.randn(2, layer.n_output, generator=g, dtype=torch.double))
            E, R = layer.compute_exact_rowlocal_eligibility(x, 1 - post.square())
            self.assertTrue(E.requires_grad)
            layer.update_exact_rowlocal_traces(x, E, R)
            with torch.no_grad():
                layer.update_M_matrix_local_fast(x, post)
        self.assertIsNot(layer.P, first)
        self.assertEqual(float(first.abs().max()), 0.0)          # the initial trace is untouched
        layer.P.sum().backward()
        self.assertGreater(float(layer.W.grad.abs().max()), 0)

    def test_grad_mode_alone_selects_the_reference_path(self):
        """Autograd on, nothing tracked (every parameter frozen): still out of
        place. The rule is the grad mode, not which operands require grad."""
        net, _ = make_net(kind='mpn1', widths=(3,), mode='hard', mp_type='mult')
        layer = net.mp_layers[0]
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
        layer.reset_state(B=2)
        layer.reset_local_learning_state(B=2)
        g = torch.Generator().manual_seed(2)
        x = torch.randn(2, layer.n_input, generator=g, dtype=torch.double)
        post = torch.tanh(torch.randn(2, layer.n_output, generator=g, dtype=torch.double))
        E, R = layer.compute_exact_rowlocal_eligibility(x, 1 - post.square())
        self.assertFalse(E.requires_grad)
        first = layer.P
        layer.update_exact_rowlocal_traces(x, E, R)
        self.assertIsNot(layer.P, first)
        self.assertEqual(float(first.abs().max()), 0.0)

    def test_update_mask_gradient_with_the_old_trace_in_another_graph(self):
        """Only the update mask requires grad, and the trace before the step is
        saved by another loss. An in-place step would bump that trace's version
        and make the other loss's backward() raise; out of place, both gradients
        exist, and d(sum c P_new)/dm equals the closed form of
        P_new = gate (m (lam P + eta x_J E^I_i) + (1-m) P)."""
        for mode in ('linear', 'hard'):
            with self.subTest(mode=mode):
                net, _ = make_net(kind='mpn1', widths=(3,), mode=mode, mp_type='mult')
                layer = net.mp_layers[0]
                for parameter in layer.parameters():
                    parameter.requires_grad_(False)
                layer.reset_state(B=2)
                layer.reset_local_learning_state(B=2)
                g = torch.Generator().manual_seed(9)

                def rand(*shape):
                    return torch.randn(*shape, generator=g, dtype=torch.double)

                layer.P.copy_(rand(*layer.P.shape) * .3)
                x, post, c = rand(2, layer.n_input), torch.tanh(rand(2, layer.n_output)), rand(*layer.P.shape)
                mask = torch.tensor([.3, .8], dtype=torch.double, requires_grad=True)
                weight = rand(*layer.P.shape).requires_grad_()
                before = layer.P
                value = before.clone()
                other = (before * weight).sum()            # this graph saves the old trace
                E, R = layer.compute_exact_rowlocal_eligibility(x, 1 - post.square())
                self.assertFalse(E.requires_grad)
                layer.update_exact_rowlocal_traces(x, E, R, update_mask=mask)
                layer.update_M_matrix_local_fast(x, post, update_mask=mask)
                gate = layer._modulation_write_derivative()

                self.assertIsNot(layer.P, before)
                self.assertTrue(torch.equal(before, value))
                other.backward()                           # raises if `before` was modified
                torch.testing.assert_close(weight.grad, value)

                eta, lam = layer._eta_lam_full()
                step = (lam[None, :, None, :] * value
                        + eta[None, :, None, :] * E[..., None] * x[:, None, None, :] - value)
                if gate is not None:
                    self.assertFalse(bool(gate.all()), 'no entry is clipped: the gate is untested')
                    step = step * gate.unsqueeze(-2)
                actual, = torch.autograd.grad((c * layer.P).sum(), mask)
                torch.testing.assert_close(actual, (c * step).sum(dim=(1, 2, 3)), rtol=1e-12, atol=1e-14)

    def test_reset_allocates_a_fresh_zero_trace(self):
        """A reset never zeroes the old buffer in place: a trace kept from the
        previous sequence stays intact."""
        net, _ = make_net(kind='mpn1', widths=(3,), mode='hard', mp_type='mult')
        layer = net.mp_layers[0]
        _, obj, _, _, _ = one_step(layer)
        kept = obj.clone()
        layer.reset_local_learning_state(B=2)
        self.assertIsNot(layer.P, obj)
        self.assertEqual(float(layer.P.abs().max()), 0.0)
        self.assertGreater(float(kept.abs().max()), 0.0)
        self.assertTrue(torch.equal(obj, kept))


if __name__ == '__main__':
    unittest.main()
