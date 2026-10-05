"""The unified --learning-signal axis and the three-stage CLI (parse / resolve /
validate): signal-independent defaults, legacy translation (--dfa preset,
--feedback, --learning-signal global), conflict rejection independent of argument
order, grouped help, metadata, the RNN script's matching names, and the per-rule
effective-configuration line. Run from tests/: python -m unittest test_cli_signal_modes -v
"""
import contextlib
import copy
import io
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import _bootstrap  # noqa: F401
import mpn
import train_common
import train_mpn
import train_rnn
from test_local_readouts import make_cfg, data


def parse(*argv):
    with patch.object(sys, 'argv', ['train_mpn.py', *argv]):
        return train_mpn._parse_args()


def rejects(*argv):
    with patch.object(sys, 'argv', ['train_mpn.py', *argv]), \
            contextlib.redirect_stderr(io.StringIO()):
        try:
            train_mpn._parse_args()
        except SystemExit:
            return True
    return False


SIGNAL_INDEPENDENT = ('rules', 'input_mode', 'local_bias_mode', 'cross_layer_steps',
                      'grad_align', 'net', 'residual', 'lam', 'rflo_trace_rho')


class TestSignalModeCLI(unittest.TestCase):
    def test_defaults_are_signal_independent(self):
        base = parse()
        self.assertEqual((base.signal_mode, base.learning_signal, base.feedback),
                         ('exact_spatial', 'global', 'exact_spatial'))
        self.assertEqual(base.rules, ['bptt', 'local_direct', 'local_diag_rflo'])
        self.assertEqual((base.input_mode, base.local_bias_mode, base.cross_layer_steps,
                          base.grad_align), ('match', 'direct', 0, True))
        # The module globals the tests/notebooks patch agree with the parsed defaults.
        self.assertEqual((train_mpn.INPUT_MODE, train_mpn.LOCAL_BIAS_MODE,
                          train_mpn.CROSS_LAYER_STEPS, train_mpn.LOG_GRAD_ALIGN),
                         ('match', 'direct', 0, True))
        self.assertEqual(train_mpn.RULES_TO_RUN, ['bptt', 'local_direct', 'local_diag_rflo'])
        self.assertEqual((train_mpn.LEARNING_SIGNAL, train_mpn.FEEDBACK_MODE),
                         ('global', 'exact_spatial'))
        # Switching ONLY the signal changes only the signal.
        for mode, pair in train_mpn.SIGNAL_MODES.items():
            with self.subTest(mode=mode):
                args = parse('--learning-signal', mode)
                self.assertEqual(args.signal_mode, mode)
                self.assertEqual((args.learning_signal, args.feedback), pair)
                for key in SIGNAL_INDEPENDENT:
                    self.assertEqual(getattr(args, key), getattr(base, key), key)
                self.assertFalse(args.dfa)

    def test_legacy_spellings_translate(self):
        self.assertEqual(parse('--feedback', 'direct_fa').signal_mode, 'dfa')
        self.assertEqual(parse('--feedback', 'layerwise_fa').signal_mode, 'layerwise_fa')
        self.assertEqual(parse('--feedback', 'exact_readout').signal_mode, 'exact_spatial')
        self.assertEqual(parse('--learning-signal', 'global', '--feedback', 'layerwise_fa').signal_mode,
                         'layerwise_fa')
        self.assertEqual(parse('--learning-signal', 'global').signal_mode, 'exact_spatial')
        # An AGREEING legacy flag is accepted alongside the unified one.
        self.assertEqual(parse('--learning-signal', 'dfa', '--feedback', 'direct_fa').signal_mode, 'dfa')
        # The --dfa PRESET keeps its historical bundle; --learning-signal dfa does not.
        preset, plain = parse('--dfa'), parse('--learning-signal', 'dfa')
        self.assertEqual((preset.signal_mode, preset.feedback, preset.input_mode,
                          preset.local_bias_mode, preset.cross_layer_steps, preset.grad_align,
                          preset.rules),
                         ('dfa', 'direct_fa', 'match', 'direct', 0, False,
                          ['bptt', 'local_exact_rowlocal', 'local_diag_rflo']))
        self.assertEqual((plain.signal_mode, plain.feedback), ('dfa', 'direct_fa'))
        self.assertEqual(plain.rules, train_mpn.RULES_TO_RUN)
        self.assertTrue(plain.grad_align)
        self.assertTrue(parse('--dfa', '--learning-signal', 'dfa').dfa)
        # Module defaults pointing at another mode: the explicit flag still wins.
        with patch.multiple(train_mpn, LEARNING_SIGNAL='local_readout'):
            self.assertEqual(parse().signal_mode, 'local_readout')
            self.assertEqual(parse('--learning-signal', 'global').signal_mode, 'exact_spatial')
            self.assertEqual(parse('--learning-signal', 'dfa').signal_mode, 'dfa')
        with patch.multiple(train_mpn, FEEDBACK_MODE='direct_fa'):
            self.assertEqual(parse().signal_mode, 'dfa')
            self.assertEqual(parse('--learning-signal', 'local_readout').feedback, 'exact_spatial')
        # Old module defaults ('exact' input mode, cross-layer 1) still switch under a
        # local signal and are kept under a global one.
        with patch.multiple(train_mpn, INPUT_MODE='exact', CROSS_LAYER_STEPS=1):
            self.assertEqual((parse().input_mode, parse().cross_layer_steps), ('exact', 1))
            local = parse('--learning-signal', 'local_readout')
            self.assertEqual((local.input_mode, local.cross_layer_steps), ('match', 0))

    def test_conflicts_are_rejected_regardless_of_order(self):
        for bad in (['--learning-signal', 'exact_spatial', '--feedback', 'direct_fa'],
                    ['--feedback', 'direct_fa', '--learning-signal', 'exact_spatial'],
                    ['--learning-signal', 'dfa', '--cross-layer-steps', '1'],
                    ['--learning-signal', 'layerwise_fa', '--cross-layer-steps', '1'],
                    ['--feedback', 'layerwise_fa', '--cross-layer-steps', '1'],
                    ['--dfa', '--learning-signal', 'local_readout'],
                    ['--dfa', '--feedback', 'exact_spatial'],
                    ['--learning-signal', 'dfa', '--input-mode', 'paired'],
                    ['--learning-signal', 'dfa', '--input-mode', 'diag_mtrace'],
                    ['--learning-signal', 'local_readout', '--input-mode', 'exact'],
                    ['--learning-signal', 'local_readout', '--cross-layer-steps', '1'],
                    ['--learning-signal', 'mixed', '--net', 'mpn1'],
                    ['--learning-signal', 'nope']):
            with self.subTest(bad=bad):
                self.assertTrue(rejects(*bad))
        # The exact_spatial pathway may carry the cross-layer correction.
        self.assertEqual(parse('--learning-signal', 'exact_spatial', '--cross-layer-steps', '1')
                         .cross_layer_steps, 1)
        # Not supported by the model → not expressible on the CLI either.
        with patch.multiple(train_mpn, LEARNING_SIGNAL='local_readout', FEEDBACK_MODE='direct_fa'):
            self.assertTrue(rejects())

    def test_help_is_grouped_by_scope(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            train_mpn.parse_arguments(['--help'])
        self.assertEqual(cm.exception.code, 0)
        text = out.getvalue()
        for title in ('network and task', 'learning algorithm', 'approximation knobs',
                      'forward dynamics', 'training and logging', 'legacy options'):
            self.assertIn(title, text)
        for flag in ('--learning-signal', '--rules', '--input-mode', '--rflo-trace-rho',
                     '--local-signal-alpha', '--seed', '--dfa', '--feedback'):
            self.assertIn(flag, text)

    def test_cfg_and_params_record_the_signal_mode(self):
        self.assertEqual(train_mpn._cfg().signal_mode, 'exact_spatial')
        _, _, params = train_mpn.build_params()
        self.assertEqual((params['feedback_mode'], params['learning_signal']),
                         ('exact_spatial', 'global'))
        with patch.multiple(train_mpn, LEARNING_SIGNAL='local_readout'):
            self.assertEqual(train_mpn._cfg().signal_mode, 'local_readout')
        with patch.multiple(train_mpn, FEEDBACK_MODE='direct_fa'):
            self.assertEqual(train_mpn._cfg().signal_mode, 'dfa')
        with patch.multiple(train_mpn, LEARNING_SIGNAL='mixed'):
            self.assertEqual(train_mpn._cfg().signal_mode, 'mixed')
        # A pair the CLI cannot express is recorded faithfully, never as a CLI mode.
        self.assertEqual(train_mpn.signal_mode_from('local_readout', 'direct_fa'),
                         'local_readout+direct_fa')
        self.assertEqual(train_mpn.signal_mode_from('global', 'exact_readout'), 'exact_spatial')
        # Legacy filenames and the figure note are unchanged for the global modes.
        cfg = train_mpn._cfg()
        cfg.run_id = ''
        self.assertNotIn('_ls-', train_common.param_tag(cfg))

    def test_rnn_cli_shares_the_signal_names(self):
        with patch.object(sys, 'argv', ['train_rnn.py']):
            self.assertEqual(train_rnn._parse_args().feedback, train_rnn.FEEDBACK_MODE)
        with patch.object(sys, 'argv', ['train_rnn.py', '--learning-signal', 'dfa']):
            self.assertEqual(train_rnn._parse_args().feedback, 'direct_fa')
        with patch.object(sys, 'argv', ['train_rnn.py', '--feedback', 'direct_fa']):
            self.assertEqual(train_rnn._parse_args().feedback, 'direct_fa')
        with patch.object(sys, 'argv', ['train_rnn.py', '--learning-signal', 'exact_spatial',
                                        '--feedback', 'direct_fa']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_rnn._parse_args()
        self.assertEqual(train_rnn._cfg().signal_mode, 'exact_spatial')
        with patch.object(train_rnn, 'FEEDBACK_MODE', 'direct_fa'):
            self.assertEqual(train_rnn._cfg().signal_mode, 'dfa')


class TestEffectiveConfigLine(unittest.TestCase):
    def test_run_seed_prints_what_each_rule_actually_runs(self):
        x, y, mask, _ = data(B=4, T=5)
        net_params = make_cfg(signal='local_readout', rule='bptt')   # bias='exact', no rho
        cfg = train_mpn._cfg()
        cfg.rules_to_run = ['bptt', 'local_direct', 'local_diag_rflo']
        cfg.learning_signal, cfg.signal_mode = 'local_readout', 'local_readout'
        cfg.input_mode, cfg.cross_layer_steps = 'match', 0
        cfg.input_normalize, cfg.log_grad_align, cfg.save_nets, cfg.n_datasets = False, False, False, 0
        cfg.device, cfg.dtype = torch.device('cpu'), torch.double
        cfg.build_params = lambda: ({}, {}, copy.deepcopy(net_params))
        cfg.net_factory = lambda params, verbose: mpn.DeepMultiPlasticNet(params, verbose=False)
        cfg.task = SimpleNamespace(init_params=lambda *p: p,
                                   valid_batch=lambda *a: (x, y, mask),
                                   train_batch=lambda *a: (x, y, mask),
                                   accuracy=lambda *a, **k: 0.0, loss_and_grad=None)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            train_common.run_seed(cfg, 13, [])
        text = log.getvalue()
        self.assertIn('bptt: input_mode=match, resolved_input_mode=exact, '
                      'mp_update=full BPTT (feedback/bias/heads unused)', text)
        self.assertIn('local_direct: input_mode=match, resolved_input_mode=three_factor, '
                      'signal=local_readout, feedback=exact_spatial, bias=direct (rule-fixed), '
                      'heads=2 active', text)
        self.assertIn('local_diag_rflo: input_mode=match, resolved_input_mode=three_factor, '
                      'signal=local_readout, feedback=exact_spatial, bias=exact, rho=none, '
                      'heads=2 active', text)
        # bptt + three_factor is a HYBRID (the embedding uses the direct 3-factor rule
        # under the global signal through the feedback pathway) and must never be
        # summarized as a full-BPTT baseline — in the log and in the checkpoint.
        cfg_h = train_mpn._cfg()
        cfg_h.rules_to_run = ['bptt']
        cfg_h.learning_signal, cfg_h.signal_mode, cfg_h.feedback_mode = 'global', 'dfa', 'direct_fa'
        cfg_h.input_mode, cfg_h.cross_layer_steps = 'three_factor', 0
        cfg_h.input_normalize, cfg_h.log_grad_align, cfg_h.save_nets, cfg_h.n_datasets = False, False, True, 0
        cfg_h.device, cfg_h.dtype = torch.device('cpu'), torch.double
        hybrid_params = make_cfg(signal='global', rule='bptt', input_mode='three_factor',
                                 feedback='direct_fa')
        cfg_h.build_params = lambda: ({}, {}, copy.deepcopy(hybrid_params))
        cfg_h.net_factory = cfg.net_factory
        cfg_h.task = cfg.task
        expected_line = ('bptt: input_mode=three_factor, resolved_input_mode=three_factor, '
                         'algorithm=hybrid, mp_update=BPTT, input_update=three_factor, '
                         'input_signal=global, input_feedback=direct_fa, heads=unused')
        with tempfile.TemporaryDirectory() as directory:
            cfg_h.ckpt_dir = cfg_h.fig_dir = cfg_h.data_dir = directory
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                train_common.run_seed(cfg_h, 13, [])
            self.assertIn(expected_line, log.getvalue())
            ckpt = torch.load(train_common.ckpt_path(cfg_h, 'bptt', 13), weights_only=False)
            self.assertEqual(ckpt['effective_config'], expected_line.split(', ', 2)[2])
            self.assertTrue(ckpt['uses_dfa'])
        # Full BPTT (match) keeps the plain label even on the dfa pathway.
        full = mpn.DeepMultiPlasticNet(make_cfg(signal='global', rule='bptt', feedback='direct_fa'),
                                       verbose=False)
        self.assertEqual(train_common.effective_rule_summary(cfg_h, full, 'bptt'),
                         'mp_update=full BPTT (feedback/bias/heads unused)')
        # The same summary is what the checkpoint records (no nets saved here, so
        # exercise the helper directly on a head-less global net).
        glob = mpn.DeepMultiPlasticNet(make_cfg(signal='global', rule='local_diag_rflo'), verbose=False)
        cfg.signal_mode = 'exact_spatial'
        self.assertEqual(train_common.effective_rule_summary(cfg, glob, 'local_diag_rflo'),
                         'signal=exact_spatial, feedback=exact_spatial, bias=exact, rho=none, heads=none')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
