"""Run converted analysis scripts on small fixtures and inspect every saved figure."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import _bootstrap
import mpn
import tasks
import train_mpn


class TestAnalysisScripts(unittest.TestCase):
    def test_default_output_directories_are_under_notebooks(self):
        class ParsedDefaults(Exception):
            pass

        for name in ('visualize_performance', 'visualize_trained_networks',
                     'compare_mpn_rnn_performance'):
            with self.subTest(script=name):
                script = _bootstrap.ROOT / 'notebooks' / f'{name}.py'
                expected = _bootstrap.ROOT / 'notebooks' / name

                def check_defaults(parser):
                    self.assertEqual(parser.get_default('output_dir'), expected)
                    raise ParsedDefaults

                with patch.object(argparse.ArgumentParser, 'parse_args', check_defaults):
                    with self.assertRaises(ParsedDefaults):
                        runpy.run_path(str(script), run_name='__main__')

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / 'figures' / 'nested'
        self.rules = ['bptt', 'local_diag_rflo', 'local_direct']
        self.stem = 'dmpn_delaygo_h4-4_test_'

    def run_script(self, name, *arguments):
        environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg')
        result = subprocess.run(
            [sys.executable, str(_bootstrap.ROOT / 'notebooks' / name),
             *map(str, arguments), '--output-dir', str(self.output)],
            cwd=self.root, env=environment, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)
        return result

    def assert_figures(self, suffixes):
        images = list(self.output.glob('*.png'))
        self.assertEqual(len(images), len(suffixes), [image.name for image in images])
        for suffix in suffixes:
            matches = [image for image in images if image.stem.endswith(suffix)]
            self.assertEqual(len(matches), 1, suffix)
            with Image.open(matches[0]) as image:
                image.load()
                self.assertGreater(image.width, 100)
                self.assertGreater(image.height, 100)
                pixels = np.asarray(image.convert('RGB'))
                self.assertGreater(pixels.std(), 1.0)

    def make_checkpoints(self):
        with patch.multiple(train_mpn, N_HIDDEN=[4, 4], NET_TYPE='dmpn', RULESET='delaygo',
                            MP_RESIDUAL=False, INPUT_MODE='match'):
            task_params, train_params, net_params = train_mpn.build_params()
        task_params, train_params, net_params = tasks.make_task('delaygo').init_params(
            task_params, train_params, net_params)
        for index, rule in enumerate(self.rules):
            torch.manual_seed(37)
            with contextlib.redirect_stdout(io.StringIO()):
                net = mpn.DeepMultiPlasticNet(net_params, verbose=False)
            with torch.no_grad():
                for name, parameter in net._trainable_params().items():
                    if name.startswith('b'):
                        parameter.fill_(.01 * (index + 1))
            torch.save({'net_params': net_params, 'task_params': task_params,
                        'state_dict': net.state_dict(), 'learning_rule': rule, 'ruleset': 'delaygo'},
                       self.root / f'{self.stem}{rule}_seed37.pt')

    def test_performance_saves_all_figures_and_accuracy(self):
        self.make_checkpoints()
        self.run_script('visualize_performance.py', '--ckpt-dir', self.root,
                        '--ckpt-stem', self.stem, '--seed', 37, '--trials', 8)
        self.assert_figures(['accuracy_angle_stimulus', 'example_trials', 'modulation_per_synapse'])
        records = list(self.output.glob('*accuracy_angle_stimulus.json'))
        self.assertEqual(len(records), 1)
        with records[0].open() as stream:
            record = json.load(stream)
        self.assertEqual(record['n_trials'], 8)
        self.assertEqual(record['rules'], self.rules)

    def test_networks_save_heatmaps_alignments_and_distributions(self):
        self.make_checkpoints()
        self.run_script('visualize_trained_networks.py', '--ckpt-dir', self.root,
                        '--ckpt-stem', self.stem, '--seed', 37)
        self.assert_figures(['weight_heatmaps', 'weight_alignment_to_bptt',
                             'bias_alignment_to_bptt', 'weight_distributions'])
        self.assertEqual(len(list(self.output.glob('*bias_alignment_to_bptt.json'))), 1)

    def test_comparison_uses_metadata_and_distinct_source_names(self):
        values = dict(rules=np.asarray(['bptt']), record_steps=np.asarray([0, 10, 20]),
                      ruleset='delaygo', n_hidden=4, n_runs=1, metric='accuracy',
                      acc_label='accuracy (%)', feedback_mode='exact_spatial')
        for split in ('train', 'valid'):
            values[f'mean__bptt__{split}'] = np.asarray([.1, .4, .8])
            values[f'std__bptt__{split}'] = np.asarray([.01, .02, .01])
        mpn_path, rnn_path = self.root / 'mpn.npz', self.root / 'rnn.npz'
        np.savez(mpn_path, **values)
        np.savez(rnn_path, **values)
        self.run_script('compare_mpn_rnn_performance.py', '--mpn-file', mpn_path, '--rnn-file', rnn_path)
        images = list(self.output.glob('*.png'))
        self.assertEqual(len(images), 1)
        self.assert_figures([images[0].stem])
        alternate = self.root / 'another_rnn.npz'
        np.savez(alternate, **values)
        self.run_script('compare_mpn_rnn_performance.py', '--mpn-file', mpn_path, '--rnn-file', alternate)
        self.assertEqual(len(list(self.output.glob('*.png'))), 2)


if __name__ == '__main__':
    unittest.main()
