"""Head-only optimizer rates, schedule ablations, and their CLI/logging surface.

CPU float64 tests include the pre-feature Adam/plateau implementation as an
independent compatibility reference. Run: python -m unittest test_local_head_optimization -v
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import _bootstrap  # noqa: F401
import mpn
import train_common as tc
import train_mpn
import wandb_logging
from test_local_readouts import data, make_cfg, make_net


def legacy_optimizer(net, lr, decay):
    """Original runner's optimizer, independent of the production group builder."""
    parameters = [p for p in net.parameters() if p.requires_grad]
    if decay:
        weights = {id(p) for name, p in net._trainable_params().items()
                   if name.startswith('W')}
        weights.update(id(p) for name, p in net._aux_params().items()
                       if name.startswith('head_W'))
        groups = [dict(params=[p for p in parameters if id(p) in weights], weight_decay=decay),
                  dict(params=[p for p in parameters if id(p) not in weights], weight_decay=0.)]
    else:
        groups = parameters
    opt = torch.optim.Adam(groups, lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=.95, patience=30, min_lr=1e-8)
    return opt, sch


class TestHeadOptimizer(unittest.TestCase):
    def test_default_matches_legacy_updates_and_plateau(self):
        x, y, mask, _ = data(B=2, T=3)
        for rule in ('local_direct', 'local_diag_rflo', 'local_exact_rowlocal', 'bptt'):
            for decay in (0., .02):
                with self.subTest(rule=rule, decay=decay):
                    old = make_net(make_cfg(widths=(3, 3), signal='local_readout', rule=rule))
                    new = copy.deepcopy(old)
                    opt_old, sch_old = legacy_optimizer(old, .001, decay)
                    _, opt_new, sch_new = tc.make_optim(new, .001, weight_decay=decay)
                    # Force a plateau reduction while exercising real gradients,
                    # Adam moments, auxiliary heads, and weight/bias decay.
                    for step in range(34):
                        for net, opt, sch in ((old, opt_old, sch_old), (new, opt_new, sch_new)):
                            opt.zero_grad()
                            net.sequence_gradients(x, y, mask)
                            opt.step()
                            sch.step(1. + step)
                    self.assertLess(opt_new.param_groups[0]['lr'], .001)
                    self.assertEqual([g['lr'] for g in opt_old.param_groups],
                                     [g['lr'] for g in opt_new.param_groups])
                    for a, b in zip(old.parameters(), new.parameters()):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_multiplier_changes_only_head_step_including_bias_and_decay(self):
        x, y, mask, _ = data(B=2, T=3)
        for rule in ('local_direct', 'local_diag_rflo', 'local_exact_rowlocal'):
            for signal in ('local_readout', 'mixed'):
                for decay in (0., .02):
                    with self.subTest(rule=rule, signal=signal, decay=decay):
                        normal = make_net(make_cfg(widths=(3, 3), signal=signal, rule=rule))
                        faster = copy.deepcopy(normal)
                        initial = {name: p.detach().clone() for name, p in normal.named_parameters()}
                        optimizers = []
                        for net, multiplier in ((normal, 1.), (faster, 3.)):
                            trainable, opt, sch = tc.make_optim(
                                net, .001, weight_decay=decay, head_lr_mult=multiplier,
                                lr_schedule='constant')
                            self.assertIsNone(sch)
                            ids = [id(p) for g in opt.param_groups for p in g['params']]
                            self.assertEqual(len(ids), len(set(ids)))
                            self.assertEqual(set(ids), {id(p) for p in trainable})
                            expected_weights = {id(p) for name, p in net._trainable_params().items()
                                                if name.startswith('W')}
                            expected_weights.update(id(p) for name, p in net._aux_params().items()
                                                    if name.startswith('head_W'))
                            for group in opt.param_groups:
                                for p in group['params']:
                                    self.assertEqual(group['weight_decay'],
                                                     decay if id(p) in expected_weights else 0.)
                            opt.zero_grad()
                            net.sequence_gradients(x, y, mask)
                            optimizers.append(opt)
                        # A learning-rate option must NOT scale any loss/signal/gradient.
                        for a, b in zip(normal.parameters(), faster.parameters()):
                            if a.grad is not None:
                                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                        for opt in optimizers:
                            opt.step()
                        for (name, a), (_, b) in zip(normal.named_parameters(), faster.named_parameters()):
                            if name.startswith('head_'):
                                torch.testing.assert_close(
                                    b - initial[name], 3 * (a - initial[name]), rtol=1e-10, atol=1e-15)
                            else:
                                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_bptt_ignores_head_multiplier_over_multiple_updates(self):
        baseline = make_net(make_cfg(signal='local_readout', rule='bptt'))
        changed = copy.deepcopy(baseline)
        opts = [tc.make_optim(n, .001, weight_decay=.02, head_lr_mult=m)[1]
                for n, m in ((baseline, 1.), (changed, 3.))]
        initial_heads = {k: p.detach().clone() for k, p in changed._aux_params().items()}
        for step in range(4):
            x, y, mask, _ = data(B=2, T=4, seed=step)
            for net, opt in zip((baseline, changed), opts):
                opt.zero_grad()
                net.sequence_gradients(x, y, mask)
                opt.step()
        for a, b in zip(baseline.parameters(), changed.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for name, p in changed._aux_params().items():
            torch.testing.assert_close(p, initial_heads[name], rtol=0, atol=0)
            self.assertIsNone(p.grad)
            self.assertNotIn(p, opts[1].state)
        rates = tc.learning_rate_snapshot(changed, opts[1])
        self.assertFalse(any(k.startswith('head_') for k in rates))
        self.assertEqual(set(rates.values()), {.001})

    def test_no_head_models_ignore_multiplier(self):
        for signal, widths in (('global', (3, 3)), ('local_readout', (3,))):
            with self.subTest(signal=signal, widths=widths):
                net = make_net(make_cfg(widths=widths, signal=signal))
                _, opt, _ = tc.make_optim(net, .001, head_lr_mult=3.)
                self.assertEqual(len(opt.param_groups), 1)
                self.assertEqual(set(tc.learning_rate_snapshot(net, opt).values()), {.001})

    def test_schedule_is_explicit_and_plateau_schedules_both_roles(self):
        for schedule in ('plateau', 'constant'):
            net = make_net(make_cfg(signal='local_readout'))
            _, opt, sch = tc.make_optim(net, .001, head_lr_mult=3., lr_schedule=schedule)
            for step in range(34):
                opt.zero_grad()
                for p in net.parameters():
                    if p.requires_grad:
                        p.grad = torch.ones_like(p)
                opt.step()
                if sch is not None:
                    sch.step(1. + step)
            factor = .95 if schedule == 'plateau' else 1.
            rates = tc.learning_rate_snapshot(net, opt)
            for name, rate in rates.items():
                self.assertAlmostEqual(rate, .001 * factor * (3 if name.startswith('head_') else 1))
            if sch is not None:
                np.testing.assert_allclose(sch.min_lrs, [1e-8, 3e-8], rtol=1e-15, atol=0)


class TestOptimizationSurface(unittest.TestCase):
    def test_defaults_validation_and_run_id(self):
        args = train_mpn._parse_args([])
        self.assertEqual(args.lr, train_mpn.LR)
        self.assertEqual(args.head_lr_mult, 1.)
        self.assertEqual(args.lr_schedule, 'plateau')
        for flag in ('--lr', '--head-lr-mult'):
            for value in ('0', '-1', 'nan', 'inf'):
                with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        train_mpn._parse_args([f'{flag}={value}'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_mpn._parse_args(['--lr-schedule', 'invalid'])
        with self.assertRaises(ValueError):
            tc.validate_optim_options(.001, float('nan'), 'plateau')
        with self.assertRaises(ValueError):
            tc.validate_optim_options(1e308, 10., 'constant')
        with patch.dict(train_mpn.__dict__):
            base = train_mpn._cfg().run_id
            train_mpn.HEAD_LR_MULT = 3.
            head = train_mpn._cfg().run_id
            train_mpn.LR_SCHEDULE = 'constant'
            schedule = train_mpn._cfg().run_id
            self.assertEqual(len({base, head, schedule}), 3)
            self.assertEqual(train_mpn._cfg().run_id, schedule)

    def test_cli_wiring_and_config_npz_wandb_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            def inspect(cfg):
                self.assertEqual((cfg.lr, cfg.head_lr_mult, cfg.lr_schedule), (.002, 3., 'constant'))
                self.assertIsNone(cfg.build_params()[1]['scheduler'])
                cfg.fig_dir = cfg.ckpt_dir = cfg.data_dir = directory
                cfg.n_runs = 1
                path = tc.save_config(cfg, str(Path(directory) / 'config.json'))
                record = json.loads(Path(path).read_text())
                self.assertEqual(record['head_lr_mult'], 3.)
                self.assertEqual(record['lr_schedule'], 'constant')
                self.assertIsNone(record['train_params']['scheduler'])
                self.assertEqual(wandb_logging._base_config(cfg, 'test')['head_lr_mult'], 3.)
                self.assertEqual(wandb_logging._base_config(cfg, 'test')['lr_schedule'], 'constant')
                cfg.rules_to_run = ['local_direct']
                runs = {'local_direct': {'train': [[.1]], 'valid': [[.2]]}}
                agg = {'local_direct': {split: {'mean': [.1], 'std': [0.]}
                                       for split in ('train', 'valid')}}
                path = str(Path(directory) / 'data.npz')
                tc.save_plot_data(cfg, [0], runs, agg, path=path)
                with np.load(path, allow_pickle=True) as saved:
                    self.assertEqual(float(saved['head_lr_mult']), 3.)
                    self.assertEqual(str(saved['lr_schedule']), 'constant')

            command = ['train_mpn.py', '--hidden', '3', '3', '--learning-signal', 'local_readout',
                       '--lr', '.002', '--head-lr-mult', '3', '--lr-schedule', 'constant']
            with patch.dict(train_mpn.__dict__), patch.object(sys, 'argv', command), \
                    patch.object(tc, 'run_experiment', side_effect=inspect) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                train_mpn.main()
                run.assert_called_once()

    def test_runner_logs_rates_and_heads_and_saves_reloadable_checkpoint(self):
        x, y, mask, _ = data(B=2, T=3)
        params = make_cfg(widths=(3, 3, 3), signal='local_readout', rule='bptt', bias='match')
        cfg = train_mpn._cfg()
        cfg.rules_to_run = ['local_direct', 'local_diag_rflo', 'local_exact_rowlocal', 'bptt']
        cfg.learning_signal, cfg.input_mode, cfg.cross_layer_steps = 'local_readout', 'match', 0
        cfg.lr, cfg.head_lr_mult, cfg.lr_schedule = .001, 3., 'constant'
        cfg.input_normalize, cfg.log_grad_align, cfg.save_nets = False, False, True
        cfg.n_datasets, cfg.batch = 2, 2
        cfg.device, cfg.dtype = torch.device('cpu'), torch.double
        cfg.build_params = lambda: ({}, {}, copy.deepcopy(params))
        cfg.net_factory = lambda p, verbose: mpn.DeepMultiPlasticNet(p, verbose=False)
        cfg.task = SimpleNamespace(
            init_params=lambda *p: p, valid_batch=lambda *args: (x, y, mask),
            train_batch=lambda *args: (x, y, mask), loss_and_grad=None,
            accuracy=lambda net, out, labels, m, inputs, isvalid=False:
                float(-((out - labels) * m).pow(2).mean()))
        records = []
        logger = SimpleNamespace(log_step=lambda rule, step, values: records.append((rule, step, values)))
        with tempfile.TemporaryDirectory() as directory:
            cfg.ckpt_dir = cfg.fig_dir = cfg.data_dir = directory
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                tc.run_seed(cfg, 13, [0, 1], wandb_logger=logger)
            self.assertIn('lr(next)', log.getvalue())
            self.assertIn('head_W0=3.00e-03', log.getvalue())
            self.assertIn('head0: acc', log.getvalue())
            self.assertEqual(len(records), 8)
            for rule, step, metrics in records:
                self.assertEqual(metrics['lr'], .001)
                self.assertEqual(metrics['lr/W_in'], .001)
                self.assertEqual(metrics['lr/W_output'], .001)
                self.assertEqual('lr/head_W0' in metrics, rule != 'bptt')
                if rule != 'bptt':
                    self.assertEqual(metrics['lr/head_W0'], .003)
                    self.assertIn('train/aux0_loss', metrics)
                    self.assertIn('train/aux0_accuracy', metrics)
            for rule in cfg.rules_to_run:
                path = tc.ckpt_path(cfg, rule, 13)
                checkpoint = torch.load(path, weights_only=False)
                self.assertEqual(checkpoint['head_lr_mult'], 3.)
                self.assertEqual(checkpoint['lr_schedule'], 'constant')
                self.assertEqual('head_W0' in checkpoint['learning_rates'], rule != 'bptt')
                with contextlib.redirect_stdout(io.StringIO()):
                    loaded = train_mpn.load_net(path, device=torch.device('cpu'), dtype=torch.double)
                self.assertEqual(loaded.learning_rule, rule)
                self.assertEqual(len(loaded._head_names), 2)


if __name__ == '__main__':
    unittest.main()
