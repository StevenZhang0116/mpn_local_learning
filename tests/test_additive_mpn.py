"""Additive MPN: independent autograd oracles, local approximations and CLI training.

Run on CPU: cd tests && python -m unittest test_additive_mpn -v
No production forward, signal, write or eligibility helper is used by the oracles.
"""
import contextlib
import copy
import io
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap
import mpn
import train_mpn
import train_common as tc

RULES = ('local_direct', 'local_diag_rflo', 'local_exact_rowlocal', 'bptt')


def make_net(kind='dmpn', widths=(3, 3), rule='local_exact_rowlocal', mode='hard',
             signal='global', feedback='exact_spatial', residual=False, cross=0,
             bias='match', rho=None, input_mode='match', plasticity='hebb_assoc',
             mp_type='add', embed=3):
    np.random.seed(17)
    torch.manual_seed(19)
    cfg = dict(net_type=kind, n_neurons=[2, *widths, 2], activation='tanh', dt=1,
               output_matrix='', output_bias=True, learning_rule=rule,
               feedback_mode=feedback, learning_signal=signal, local_signal_alpha=.6,
               input_mode=input_mode, mp_residual=residual,
               residual_scale=.4 if residual else 1., cross_layer_steps=cross,
               input_layer_add=True, input_layer_add_trainable=True,
               input_layer_bias=True, linear_embed=embed,
               ml_params=dict(mp_type=mp_type, bias=True, local_bias_mode=bias,
                              rflo_trace_rho=rho, m_update_type=plasticity,
                              m_activation='scaled_tanh' if mode == 'smooth' else 'linear',
                              m_scale=.2, modulation_bounds=mode == 'hard',
                              m_bounds=(-.2, .2), eta_type='scalar', eta_train=False,
                              lam_type='scalar', lam_train=False, m_time_scale=10))
    cls = mpn.DeepMultiPlasticNet if kind == 'dmpn' else mpn.MultiPlasticNet
    with contextlib.redirect_stdout(io.StringIO()):
        net = cls(copy.deepcopy(cfg), verbose=False).double()
    with torch.no_grad():
        for layer in net.mp_layers:
            layer.eta.fill_(1.2)
            layer.lam.fill_(.8)
            layer.M_init.uniform_(-.08, .08)
    return net, cfg


def data(T=5):
    torch.manual_seed(31)
    x = torch.randn(2, T, 2, dtype=torch.double) * .8
    y = torch.randn_like(x) * .3
    mask = torch.rand_like(y)
    mask[0, 0] = 0
    active = torch.tensor([[1., .25, 0., .7, 1.], [.5, 1., .8, 0., 1.]],
                          dtype=torch.double)[:, :T]
    return x, y, mask, active


def loss_fn(q, y, mask):
    return ((q-y)*mask).square().mean()


def fourth_loss(q, y, mask):
    return (mask*(q-y).pow(4)).mean(), 4*mask*(q-y).pow(3)/q.numel()


def effective(layer, W, M):
    return W + M if layer.mp_type == 'add' else W * (1+M)


def write(layer, M, x, post, active):
    if layer.m_update_type == 'hebb_pre':
        post = torch.ones_like(post) / layer.n_output**.5
    raw = layer.lam*M + layer.eta*post[:, :, None]*x[:, None, :]
    r = 1 if active is None else active[:, None, None]
    if layer.m_act == 'scaled_tanh':
        value = r*layer.m_scale*torch.tanh(raw/layer.m_scale) + (1-r)*M
    else:
        value = r*raw + (1-r)*M
        if layer.modulation_bounds:
            value = torch.where(value < layer.M_bounds[1], layer.M_bounds[1],
                                torch.where(value > layer.M_bounds[0], layer.M_bounds[0], value))
    frozen = getattr(layer, '_plasticity_freeze_mask', None)
    if frozen is not None:
        keep = torch.ones_like(value, dtype=torch.bool)
        keep[:, frozen[0], frozen[1]] = False
        value = torch.where(keep, value, layer.M_init[None])
    return value


def forward(net, x, active=None, stop_states=False, column=None):
    """Full graph, spatial-only graph, or one-column input-trace graph."""
    x = (x-net.input_loc)/net.input_scale if net.input_normalize else x
    deep = isinstance(net, mpn.DeepMultiPlasticNet)
    states = [l.M_init[None].expand(len(x), -1, -1).clone() for l in net.mp_layers]
    outputs, streams, consumed = [], [], []
    for t in range(x.shape[1]):
        h = torch.tanh(net.W_initial_linear(x[:, t])) if deep and net.input_layer_active else x[:, t]
        if column is not None:
            keep = torch.zeros_like(h)
            keep[:, column] = 1
            h = h.detach() + keep*(h-h.detach())
        hs = [h]
        consumed.append([s.detach().clone() for s in states])
        for n, layer in enumerate(net.mp_layers):
            a = torch.tanh((effective(layer, layer.W, states[n])*h[:, None]).sum(-1)+layer.b)
            updated = write(layer, states[n], h, a, None if active is None else active[:, t])
            if stop_states:
                updated = updated.detach()
            elif column is not None:
                keep = torch.zeros_like(updated)
                if n == 0:
                    keep[:, :, column] = 1
                updated = updated.detach() + keep*(updated-updated.detach())
            states[n] = updated
            h = h+net.residual_scale*a if deep and net._residual_at[n] else a
            hs.append(h)
        streams.append(hs)
        outputs.append(F.linear(h, net.W_output, net.b_output))
    return torch.stack(outputs, 1), streams, consumed


def full_oracle(net, x, y, mask, active=None, column=None):
    out, _, _ = forward(net, x, active, column=column)
    params = net._trainable_params()
    if column is not None:
        params = {k: p for k, p in params.items() if k in ('W_in', 'b_in')}
    grads = torch.autograd.grad(loss_fn(out, y, mask), list(params.values()))
    return out.detach(), dict(zip(params, grads))


def isolated_oracle(layer, xs, ell, active, mode, bias_exact):
    """Differentiate isolated neurons; RFLO keeps one M column per parameter column.

    Stop-gradient edges encode the approximation, independently of trace recurrences.
    Biases are differentiated in a separate exact/direct graph, as configured.
    """
    def run(column=None, stop=False):
        W = layer.W.detach().clone().requires_grad_()
        b = layer.b.detach().clone().requires_grad_()
        M = layer.M_init[None].expand(len(xs), -1, -1).clone()
        objective = 0
        for t in range(xs.shape[1]):
            a = torch.tanh((effective(layer, W, M)*xs[:, t, None]).sum(-1)+b)
            objective = objective + (ell[:, t]*a).sum()
            M = write(layer, M, xs[:, t], a, None if active is None else active[:, t])
            if stop:
                M = M.detach()
            elif column is not None:
                keep = torch.zeros_like(M)
                keep[:, :, column] = 1
                M = M.detach() + keep*(M-M.detach())
        return torch.autograd.grad(objective, (W, b))
    Wg = (torch.stack([run(column=j)[0][:, j] for j in range(layer.n_input)], 1)
          if mode == 'local_diag_rflo' else run(stop=mode == 'local_direct')[0])
    return Wg, run(stop=not bias_exact)[1]


def local_oracle(net, x, y, mask, active, rule, loss=loss_fn):
    """Independent spatial signals (autograd) times isolated local sensitivities."""
    out, streams, _ = forward(net, x, active, stop_states=True)
    deep = isinstance(net, mpn.DeepMultiPlasticNet)
    L = len(net.mp_layers)
    nodes = [s[n+1] for s in streams for n in range(L)]
    main = loss(out, y, mask)
    signals = list(torch.autograd.grad(main, nodes, retain_graph=True))
    signals = [torch.stack([signals[t*L+n] for t in range(len(streams))], 1).detach()
               for n in range(L)]
    expected = {}
    for key in ('W_output', 'b_output'):
        expected[key] = torch.autograd.grad(main, net._trainable_params()[key], retain_graph=True)[0]
    if deep and net.learning_signal != 'global':
        for n, (wn, bn) in enumerate(net._head_names):
            hs = torch.stack([s[n+1] for s in streams], 1)
            head = F.linear(hs, getattr(net, wn), getattr(net, bn))
            obj = loss(head, y, mask)
            sig = torch.autograd.grad(obj, hs, retain_graph=True)[0].detach()
            signals[n] = sig if net.learning_signal == 'local_readout' else signals[n]+net.local_signal_alpha*sig
            for key in (wn, bn):
                expected[key] = torch.autograd.grad(obj, getattr(net, key), retain_graph=True)[0]
    for n, layer in enumerate(net.mp_layers):
        xs = torch.stack([s[n].detach() for s in streams], 1)
        scale = net._branch_scales[n] if deep else 1.
        bias_exact = (rule != 'local_direct' and (layer.local_bias_mode == 'exact' or
                      (layer.local_bias_mode == 'match' and rule == 'local_exact_rowlocal')))
        gW, gb = isolated_oracle(layer, xs, scale*signals[n], active, rule, bias_exact)
        suffix = '' if n == 0 else str(n)
        expected['W'+suffix], expected['b'+suffix] = gW, gb
    if deep and net.input_layer_active:
        objective = sum((signals[0][:, t]*s[1]).sum() for t, s in enumerate(streams))
        for key in ('W_in', 'b_in'):
            expected[key] = torch.autograd.grad(objective, net._trainable_params()[key], retain_graph=True)[0]
    return out.detach(), expected


class TestAdditiveMPN(unittest.TestCase):
    def close_grads(self, actual, expected):
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, rtol=3e-9, atol=3e-11, msg=key)

    def test_full_bptt_against_independent_unroll(self):
        x, y, mask, _ = data()
        for kind, widths in (('mpn1', (3,)), ('dmpn', (3,)), ('dmpn', (3, 2, 3))):
            for mode, residual in product(('none', 'hard', 'smooth'), (False, True)):
                if kind == 'mpn1' and residual:
                    continue
                with self.subTest(kind=kind, widths=widths, mode=mode, residual=residual):
                    net, _ = make_net(kind=kind, widths=widths, rule='bptt', mode=mode, residual=residual)
                    net.mp_layers[0].set_plasticity_freeze(torch.tensor([0]), torch.tensor([1]))
                    out, expected = full_oracle(net, x, y, mask)
                    actual = net.sequence_gradients(x, y, mask)
                    torch.testing.assert_close(actual['outputs'], out)
                    self.close_grads(actual, expected)

    def test_local_rules_against_detached_graph_oracles(self):
        x, y, mask, active = data()
        for kind, widths, signal in (('mpn1', (3,), 'global'), ('dmpn', (3,), 'global'),
                                    ('dmpn', (3, 2), 'global'), ('dmpn', (3, 3), 'local_readout'),
                                    ('dmpn', (3, 3), 'mixed')):
            for rule, mode in product(RULES[:-1], ('none', 'hard', 'smooth')):
                with self.subTest(kind=kind, widths=widths, signal=signal, rule=rule, mode=mode):
                    net, _ = make_net(kind=kind, widths=widths, signal=signal, rule=rule,
                                      mode=mode, residual=kind == 'dmpn')
                    net.mp_layers[0].set_plasticity_freeze(torch.tensor([0]), torch.tensor([1]))
                    out, expected = local_oracle(net, x, y, mask, active, rule)
                    actual = net.sequence_gradients(x, y, mask, update_masks=active)
                    actual.update({k: p.grad for k, p in getattr(net, '_aux_params', lambda: {})().items()})
                    torch.testing.assert_close(actual['outputs'], out)
                    self.close_grads(actual, expected)

    def test_exact_bias_and_pre_only_writes(self):
        x, y, mask, active = data()
        for rule, kind in product(RULES[:-1], ('hebb_assoc', 'hebb_pre')):
            net, _ = make_net(rule=rule, bias='exact', plasticity=kind, mode='smooth')
            _, expected = local_oracle(net, x, y, mask, active, rule)
            self.close_grads(net.sequence_gradients(x, y, mask, update_masks=active), expected)

    def test_single_layer_rowlocal_matches_bptt(self):
        x, y, mask, _ = data()
        for mode in ('none', 'hard', 'smooth'):
            net, _ = make_net(kind='mpn1', widths=(3,), mode=mode)
            expected = net.bptt_gradients(x, y, mask)
            self.close_grads(net.sequence_gradients(x, y, mask),
                             {k: expected[k] for k in net._trainable_params()})

    def test_two_step_cross_layer_exactness(self):
        x, y, mask, active = data(T=2)
        for mode, residual, kind in product(('none', 'hard', 'smooth'), (False, True),
                                            ('hebb_assoc', 'hebb_pre')):
            net, _ = make_net(widths=(3, 3, 2), mode=mode, residual=residual,
                              plasticity=kind, cross=1)
            net.mp_layers[1].set_plasticity_freeze(torch.tensor([0]), torch.tensor([1]))
            out, expected = full_oracle(net, x, y, mask, active)
            expected = {k: g for k, g in expected.items() if k not in ('W_in', 'b_in')}
            actual = net.sequence_gradients(x, y, mask, update_masks=active)
            torch.testing.assert_close(actual['outputs'], out)
            self.close_grads(actual, expected)

    def test_cross_correction_leaves_top_layer_unchanged(self):
        x, y, mask, active = data()
        for rule in RULES[:-1]:
            net, _ = make_net(rule=rule, rho=.6, mode='smooth')
            base = net.sequence_gradients(x, y, mask, update_masks=active)
            net.cross_layer_steps = 1
            corrected = net.sequence_gradients(x, y, mask, update_masks=active)
            self.close_grads(corrected, {k: base[k] for k in ('W1', 'b1', 'W_output', 'b_output')})
            self.assertGreater(float((corrected['W']-base['W']).norm()), 1e-8)

    def test_input_column_trace_against_detached_unroll(self):
        x, y, mask, active = data()
        for mode, kind, residual in product(('none', 'hard', 'smooth'),
                                             ('hebb_assoc', 'hebb_pre'), (False, True)):
            net, _ = make_net(mode=mode, plasticity=kind, residual=residual,
                              rule='local_diag_rflo', input_mode='diag_mtrace')
            net.mp_layers[0].set_plasticity_freeze(torch.tensor([0]), torch.tensor([1]))
            net.set_input_norm_stats(x.reshape(-1, 2))
            actual = net.sequence_gradients(x, y, mask, update_masks=active)
            expected = {k: torch.zeros_like(actual[k]) for k in ('W_in', 'b_in')}
            for j in range(3):
                out, per_column = full_oracle(net, x, y, mask, active, column=j)
                for k in expected:
                    expected[k][j] = per_column[k][j]
            torch.testing.assert_close(actual['outputs'], out)
            self.close_grads(actual, expected)

    def test_explicit_and_fused_paths_and_capped_recurrence(self):
        xseq, _, _, masks = data()
        for mode, rule, rho in product(('none', 'hard', 'smooth'), RULES[:-1], (None, .4)):
            net, _ = make_net(kind='mpn1', widths=(3,), mode=mode, rule=rule, rho=rho, bias='exact')
            layer = net.mp_layer
            layer.set_plasticity_freeze(torch.tensor([0]), torch.tensor([1]))
            layer.reset_state(B=2)
            short = {'local_direct': 'direct', 'local_diag_rflo': 'diag', 'local_exact_rowlocal': 'exact'}[rule]
            if short == 'diag':
                layer.reset_diag_rflo_state(B=2)
            elif short == 'exact':
                layer.reset_local_learning_state(B=2)
            other = copy.deepcopy(layer)
            compute = {'direct': 'compute_direct_local_eligibility', 'diag': 'compute_diag_rflo_eligibility',
                       'exact': 'compute_exact_rowlocal_eligibility'}[short]
            update = {'diag': 'update_diag_rflo_traces', 'exact': 'update_exact_rowlocal_traces'}
            with torch.no_grad():
                for t in range(5):
                    x = xseq[:, t]
                    post = torch.tanh((layer.W+layer.M).mul(x[:, None]).sum(-1)+layer.b)
                    phi, ell = 1-post.square(), torch.full_like(post, .7)
                    eta, lam = layer._eta_lam_full()
                    old_A = layer.A.clone() if short == 'diag' else None
                    raw_gain = lam[None] + eta[None]*phi[..., None]*x[:, None].square()
                    expected_A = None
                    if short == 'diag' and rho is not None:
                        k = eta[None]*phi[..., None]*x[:, None].square()
                        candidate = raw_gain.clamp(-rho, rho)*old_A+k
                        # Apply the write's derivative to the candidate trace independently.
                        raw = lam*layer.M+eta*post[..., None]*x[:, None]
                        r = masks[:, t, None, None]
                        if mode == 'smooth':
                            expected_A = r*(1-torch.tanh(raw/.2).square())*candidate+(1-r)*old_A
                        else:
                            argument = r*raw+(1-r)*layer.M
                            gate = ((argument >= -.2) & (argument <= .2)) if mode == 'hard' else 1
                            expected_A = gate*(r*candidate+(1-r)*old_A)
                        expected_A[:, 0, 1] = 0
                    E, R = getattr(other, compute)(x, phi)
                    expected_W = (ell[..., None]*E).sum(0)
                    expected_b = (ell*R).sum(0)
                    if short in update:
                        getattr(other, update[short])(x, E, R, masks[:, t])
                    gW, gb = layer.local_grad_step_fast(x, phi, ell, short, update_mask=masks[:, t])
                    other.update_M_matrix(x, post, masks[:, t])
                    layer.update_M_matrix_local_fast(x, post, update_mask=masks[:, t])
                    self.close_grads({'W': gW, 'b': gb}, {'W': expected_W, 'b': expected_b})
                    torch.testing.assert_close(layer.M, other.M)
                    if short != 'direct':
                        trace = 'A' if short == 'diag' else 'P'
                        torch.testing.assert_close(getattr(layer, trace), getattr(other, trace))
                        torch.testing.assert_close(layer.Q, other.Q)
                    if expected_A is not None:
                        torch.testing.assert_close(layer.A, expected_A, rtol=1e-10, atol=1e-12)

    def test_no_plasticity_limit_and_random_feedback(self):
        x, y, mask, _ = data()
        for feedback, rule in product(('exact_spatial', 'layerwise_fa', 'direct_fa'), RULES[:-1]):
            net, _ = make_net(rule=rule, feedback=feedback, residual=True)
            with torch.no_grad():
                for l in net.mp_layers:
                    l.eta.zero_()
            actual = net.sequence_gradients(x, y, mask)
            if feedback == 'exact_spatial':
                _, expected = full_oracle(net, x, y, mask)
            else:
                # No temporal derivative: all local eligibility choices coincide.
                reference = copy.deepcopy(net)
                reference.learning_rule = 'local_direct'
                expected = {k: g for k, g in reference.sequence_gradients(x, y, mask).items()
                            if k in reference._trainable_params()}
            self.close_grads(actual, expected)

    def test_custom_loss_local_heads(self):
        x, y, mask, active = data()
        net, _ = make_net(signal='local_readout', mode='smooth')
        _, expected = local_oracle(net, x, y, mask, active, 'local_exact_rowlocal',
                                   loss=lambda q, y, m: fourth_loss(q, y, m)[0])
        actual = net.sequence_gradients(x, y, mask, update_masks=active, loss_and_grad=fourth_loss)
        actual.update({k: p.grad for k, p in net._aux_params().items()})
        self.close_grads(actual, expected)

    def test_fixed_bounds_and_invalid_type(self):
        net, _ = make_net()
        layer = net.mp_layers[0]
        before = layer.M_bounds.clone()
        with torch.no_grad():
            layer.W.mul_(-20)
        rebuilt, _ = layer.build_M_bounds()
        torch.testing.assert_close(before, rebuilt)
        self.assertTrue(torch.all(before[0] == torch.tensor(.2, dtype=torch.float32).double()))
        with self.assertRaisesRegex(ValueError, 'mp_type'):
            make_net(mp_type='bad')

    def test_additive_trace_diagnostics(self):
        sys.path.insert(0, str(_bootstrap.ROOT/'notebooks'))
        import diagnose_gradients as diagnostic
        net, _ = make_net(rule='local_diag_rflo', rho=.4)
        layer = net.mp_layers[0]
        layer.reset_state(B=2)
        layer.reset_diag_rflo_state(B=2)
        with torch.no_grad():
            layer.W.fill_(-8.)  # Multiplicative formulas would report different metrics.
            layer.A.fill_(.25)
            layer.M.fill_(.1)
            recorder = diagnostic.TraceRecorder(net)
            x = torch.ones(2, 3, dtype=torch.double)
            phi = ell = torch.ones(2, 3, dtype=torch.double)
            eta, lam = layer._eta_lam_full()
            layer._local_step_diag(x, phi, ell, eta, lam)
        row = recorder.rows[0]
        self.assertEqual(row['base_rms'], 1.)
        self.assertEqual(row['wa_rms'], .25)
        self.assertEqual(row['factor_rms'], 1.25)
        self.assertEqual(row['trace_gain_clipped_fraction'], 1.)


class TestAdditiveSurface(unittest.TestCase):
    def test_contextdelaydm1_short_training(self):
        # Exercise the real task adapter and optimizer, not a synthetic stand-in.
        for kind, widths, signal in (('mpn1', 3, 'global'), ('dmpn', [3, 3], 'local_readout')):
            with patch.multiple(train_mpn, NET_TYPE=kind, N_HIDDEN=widths,
                                MP_TYPE='add', MP_RESIDUAL=False, RESIDUAL_SCALE=1.,
                                RULESET='contextdelaydm1', RULES_TO_RUN=list(RULES),
                                LEARNING_SIGNAL=signal, FEEDBACK_MODE='exact_spatial',
                                INPUT_MODE='match', LOCAL_BIAS_MODE='match',
                                CROSS_LAYER_STEPS=0, RFLO_TRACE_RHO=.99,
                                MODULATION_MODE='hard', MODULATION_BOUNDS=True,
                                MODULATION_BOUND=.1, BATCH=2, N_DATASETS=2,
                                LOG_GRAD_ALIGN=False, N_RUNS=1):
                cfg = train_mpn._cfg()
                cfg.device, cfg.dtype = torch.device('cpu'), torch.float32
                cfg.save_nets = True
                records = []
                logger = SimpleNamespace(log_step=lambda rule, step, values:
                                         records.append((rule, step, values)))
                with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                    cfg.ckpt_dir = cfg.fig_dir = cfg.data_dir = directory
                    tc.run_seed(cfg, 23, [0, 1], wandb_logger=logger)
                    self.assertEqual(len(records), 2*len(RULES))
                    for rule in RULES:
                        loaded = train_mpn.load_net(tc.ckpt_path(cfg, rule, 23),
                                                    device=torch.device('cpu'))
                        self.assertTrue(all(l.mp_type == 'add' for l in loaded.mp_layers))
                        self.assertTrue(all(torch.isfinite(p).all() for p in loaded.parameters()))

    def test_cli_config_and_checkpoint(self):
        self.assertEqual(train_mpn._parse_args([]).mp_type, 'mult')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_mpn._parse_args(['--mp-type', 'bad'])
        for kind, hidden in (('mpn1', ['3']), ('dmpn', ['3', '3'])):
            for mode in ('none', 'hard', 'scaled_tanh'):
                with tempfile.TemporaryDirectory() as directory:
                    def inspect(cfg):
                        params = cfg.build_params()[2]
                        self.assertEqual(params['ml_params']['mp_type'], 'add')
                        net = cfg.net_factory(params, False).double()
                        self.assertTrue(all(l.mp_type == 'add' for l in net.mp_layers))
                        tc.save_config(cfg, str(Path(directory)/'config.json'))
                        saved = json.loads((Path(directory)/'config.json').read_text())
                        self.assertEqual(saved['net_params']['ml_params']['mp_type'], 'add')
                        path = Path(directory)/'net.pt'
                        torch.save(dict(net_params=params, state_dict=net.state_dict(), learning_rule='bptt'), path)
                        loaded = train_mpn.load_net(path, device=torch.device('cpu'), dtype=torch.double)
                        self.assertTrue(all(l.mp_type == 'add' for l in loaded.mp_layers))
                        for l, ll in zip(net.mp_layers, loaded.mp_layers):
                            if l.modulation_bounds:
                                torch.testing.assert_close(l.M_bounds, ll.M_bounds)
                    argv = ['train_mpn.py', '--mp-type', 'add', '--net', kind, '--hidden', *hidden,
                            '--no-residual', '--modulation-mode', mode, '--task', 'contextdelaydm1']
                    with patch.dict(train_mpn.__dict__), patch.object(sys, 'argv', argv), \
                            patch.object(tc, 'run_experiment', side_effect=inspect), \
                            contextlib.redirect_stdout(io.StringIO()):
                        train_mpn.main()

    def test_lockstep_training_and_reload_all_rules(self):
        x, y, mask, _ = data(T=3)
        for kind, widths, signal in (('mpn1', (3,), 'global'), ('dmpn', (3,), 'global'),
                                    ('dmpn', (3, 3), 'local_readout')):
            _, params = make_net(kind=kind, widths=widths, signal=signal, rule='bptt', rho=.99)
            cfg = train_mpn._cfg()
            cfg.rules_to_run = list(RULES)
            cfg.learning_signal, cfg.input_mode, cfg.cross_layer_steps = signal, 'match', 0
            cfg.lr, cfg.lr_schedule = .001, 'constant'
            cfg.input_normalize, cfg.log_grad_align, cfg.save_nets = False, False, True
            cfg.n_datasets, cfg.batch = 2, 2
            cfg.device, cfg.dtype = torch.device('cpu'), torch.double
            cfg.build_params = lambda: ({}, {}, copy.deepcopy(params))
            cls = mpn.DeepMultiPlasticNet if kind == 'dmpn' else mpn.MultiPlasticNet
            cfg.net_factory = lambda p, verbose: cls(p, verbose=False)
            cfg.task = SimpleNamespace(init_params=lambda *p: p,
                valid_batch=lambda *a: (x, y, mask), train_batch=lambda *a: (x, y, mask),
                loss_and_grad=None,
                accuracy=lambda net, out, labels, m, inputs, isvalid=False: float(-loss_fn(out, labels, m)))
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                cfg.ckpt_dir = cfg.fig_dir = cfg.data_dir = directory
                tc.run_seed(cfg, 7, [0, 1])
                for rule in RULES:
                    path = tc.ckpt_path(cfg, rule, 7)
                    loaded = train_mpn.load_net(path, device=torch.device('cpu'), dtype=torch.double)
                    self.assertEqual(loaded.learning_rule, rule)
                    self.assertTrue(all(l.mp_type == 'add' for l in loaded.mp_layers))
                    grads = loaded.sequence_gradients(x, y, mask)
                    self.assertTrue(torch.isfinite(grads['loss']))
                    for key in loaded._trainable_params():
                        self.assertTrue(torch.isfinite(grads[key]).all(), (rule, key))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
