"""Compact output naming and metadata checks, without training."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import _bootstrap
import train_common as tc
import train_mpn
import train_rnn


class TestOutputLayout(unittest.TestCase):
    def test_stable_short_names_and_distinct_experiments(self):
        with patch.multiple(train_mpn, N_HIDDEN=[128] * 9, SEED=288):
            cfg = train_mpn._cfg()
            self.assertEqual(tc.data_path(cfg), train_mpn.data_path())
            self.assertLess(len(Path(tc.data_path(cfg)).name), 65)
            self.assertNotIn('_b128', tc.data_path(cfg))
            self.assertEqual(Path(tc.ckpt_path(cfg, 'bptt', 288)).relative_to(cfg.ckpt_dir),
                             Path(cfg.run_id) / 'seed288' / 'bptt.pt')
            with patch.object(train_mpn, 'BATCH', 16):
                self.assertNotEqual(cfg.run_id, train_mpn._cfg().run_id)
            # A fresh invocation gets a new ID even with identical settings.
            with patch.object(train_mpn, '_RUN_IDS', {}):
                self.assertNotEqual(cfg.run_id, train_mpn._cfg().run_id)
        rnn_cfg = train_rnn._cfg()
        self.assertEqual(tc.run_stem(rnn_cfg),
                         f'{rnn_cfg.file_prefix}_{tc.param_tag(rnn_cfg)}_runs{rnn_cfg.n_runs}')

    def test_full_configuration_is_saved_in_run_folder(self):
        with tempfile.TemporaryDirectory() as directory, patch.multiple(
                train_mpn, CKPT_DIR=directory, FIG_DIR=directory, SAVE_NETS=True,
                N_HIDDEN=[4, 6, 4], BATCH=7, N_RUNS=3, SEED=288, RULESET='delaygo',
                INPUT_MODE='paired', RULES_TO_RUN=['bptt', 'local_diag_rflo', 'local_direct']):
            cfg = train_mpn._cfg()
            with contextlib.redirect_stdout(io.StringIO()):
                path = Path(tc.save_config(cfg))
            self.assertEqual(path, Path(directory) / cfg.run_id / 'config.json')
            record = json.loads(path.read_text())
            self.assertEqual(record['run_id'], cfg.run_id)
            self.assertEqual(record['batch'], 7)
            self.assertEqual(record['train_params']['batch_size'], 7)
            self.assertEqual(record['seeds'], [288, 289, 290])
            self.assertEqual(record['net_params']['n_neurons'][1:-1], [4, 6, 4])
            self.assertEqual(record['task_params']['rules'], ['delaygo'])
            self.assertEqual(record['resolved_input_modes']['local_diag_rflo'], 'diag_mtrace')
            cfg.save_nets = False
            self.assertEqual(Path(tc.config_path(cfg)), Path(directory) / f'{cfg.run_id}.json')

    def test_dfa_flags_record_preset_and_per_model_behavior(self):
        # Exercise the real save path with tiny models and no optimizer steps.
        cases = [('direct_fa', True, 'match', True),
                 ('direct_fa', False, 'match', True),
                 ('exact_spatial', False, 'match', True),
                 ('layerwise_fa', False, 'match', True),
                 ('direct_fa', False, 'three_factor', True),
                 ('direct_fa', False, 'three_factor', False)]
        for feedback, preset, input_mode, trainable_embed in cases:
            with self.subTest(feedback=feedback, preset=preset, input_mode=input_mode,
                              trainable_embed=trainable_embed), tempfile.TemporaryDirectory() as directory:
                with patch.multiple(
                        train_mpn, CKPT_DIR=directory, SAVE_NETS=True, N_HIDDEN=[2, 2],
                        N_DATASETS=0, BATCH=2, N_RUNS=1, RULESET='delaygo',
                        DEVICE=torch.device('cpu'), LOG_GRAD_ALIGN=False, CROSS_LAYER_STEPS=0,
                        FEEDBACK_MODE=feedback, DFA_PRESET=preset, INPUT_MODE=input_mode,
                        RULES_TO_RUN=['bptt', 'local_direct', 'local_diag_rflo']):
                    cfg = train_mpn._cfg()
                    params = cfg.task.init_params(*cfg.build_params())
                net_params = params[2]
                net_params['input_layer_add_trainable'] = trainable_embed
                net_params['input_layer_bias'] = trainable_embed
                inputs = torch.zeros(2, 1, net_params['n_neurons'][0])
                labels = torch.zeros(2, 1, net_params['n_neurons'][-1])
                cfg.task = SimpleNamespace(init_params=lambda *args: params,
                                           valid_batch=lambda *args: (inputs, labels, torch.ones_like(labels)))
                expected = {r: feedback == 'direct_fa' and (
                    r != 'bptt' or (input_mode == 'three_factor' and trainable_embed))
                    for r in cfg.rules_to_run}
                with contextlib.redirect_stdout(io.StringIO()):
                    tc.run_seed(cfg, 13, [])
                    config = json.loads(Path(tc.save_config(cfg)).read_text())
                self.assertIs(config['dfa_preset'], preset)
                self.assertIs(config['uses_dfa'], any(expected.values()))
                self.assertEqual(config['uses_dfa_by_rule'], expected)
                for rule in cfg.rules_to_run:
                    checkpoint = torch.load(tc.ckpt_path(cfg, rule, 13), weights_only=False)
                    self.assertIs(checkpoint['dfa_preset'], preset)
                    self.assertIs(checkpoint['uses_dfa'], expected[rule])
                    self.assertEqual(checkpoint['feedback_mode'], feedback)


if __name__ == '__main__':
    unittest.main()
