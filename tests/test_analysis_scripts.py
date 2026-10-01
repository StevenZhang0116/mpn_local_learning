"""Check checkpoint selection and analysis entry points using small figure fixtures."""

import argparse
import contextlib
import io
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

sys.path.insert(0, str(_bootstrap.ROOT / 'notebooks'))
from visualize_trained_networks import select_checkpoints


class TestAnalysisScripts(unittest.TestCase):
    def test_default_output_directories_are_under_notebooks(self):
        class ParsedDefaults(Exception):
            pass

        for name in ('visualize_trained_networks',
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

    def run_script(self, name, *arguments, expected_returncode=0):
        environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg')
        result = subprocess.run(
            [sys.executable, str(_bootstrap.ROOT / 'notebooks' / name),
             *map(str, arguments), '--output-dir', str(self.output)],
            cwd=self.root, env=environment, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, expected_returncode, result.stdout + '\n' + result.stderr)
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
        result = self.run_script('visualize_trained_networks.py', '--analysis', 'performance',
                                 '--ckpt-dir', self.root,
                                 '--ckpt-stem', self.stem, '--seed', 37, '--trials', 8)
        self.assertEqual(result.stdout.count('Held-out masked MSE loss on 8 trials'), 2)
        for timing_mode in ('random', 'random_batch'):
            self.assertIn(f'mode_input={timing_mode}, task=delaygo, seed=37', result.stdout)
        for threshold in (0.3, 0.6, 0.9):
            self.assertIn(f'Active synapses (|M| > {threshold}); representative trials only:', result.stdout)
        self.assert_figures(['accuracy_angle_stimulus', 'example_trials', 'modulation_per_synapse',
                             'modulation_active_fraction_threshold0.3',
                             'modulation_active_fraction_threshold0.6',
                             'modulation_active_fraction_threshold0.9'])
        self.assertEqual(list(self.output.glob('*.json')), [])

    def test_networks_save_heatmaps_alignments_and_distributions(self):
        self.make_checkpoints()
        self.run_script('visualize_trained_networks.py', '--ckpt-dir', self.root,
                        '--ckpt-stem', self.stem, '--seed', 37, '--analysis', 'weights')
        self.assert_figures(['weight_heatmaps', 'weight_alignment_to_bptt',
                             'bias_alignment_to_bptt', 'parameter_cosine_diag_rflo_vs_direct',
                             'weight_distributions'])
        self.assertEqual(list(self.output.glob('*.json')), [])

    def test_combined_analysis_automatically_selects_run_and_seed(self):
        self.make_checkpoints()
        # A newer, incomplete seed must not displace the shared seed.
        (self.root / f'{self.stem}bptt_seed38.pt').touch()
        result = self.run_script('visualize_trained_networks.py', '--ckpt-dir', self.root,
                                 '--trials', 8)
        self.assertIn(f'Selected checkpoint group: {self.stem}seed37', result.stdout)
        for rule in self.rules:
            self.assertEqual(result.stdout.count(f'<- {self.stem}{rule}_seed37.pt'), 1)
        self.assert_figures(['weight_heatmaps', 'weight_alignment_to_bptt',
                             'bias_alignment_to_bptt', 'parameter_cosine_diag_rflo_vs_direct',
                             'weight_distributions', 'accuracy_angle_stimulus',
                             'example_trials', 'modulation_per_synapse',
                             'modulation_active_fraction_threshold0.3',
                             'modulation_active_fraction_threshold0.6',
                             'modulation_active_fraction_threshold0.9'])
        self.assertEqual(list(self.output.glob('*.json')), [])

    def test_selection_uses_complete_groups_and_respects_overrides(self):
        for seed in (37, 38):
            for rule in self.rules:
                path = self.root / f'{self.stem}{rule}_seed{seed}.pt'
                path.touch()
                os.utime(path, (seed, seed))
        stem, seed, paths = select_checkpoints(self.root, self.rules, self.stem)
        self.assertEqual((stem, seed), (self.stem, 38))
        self.assertEqual(list(paths), self.rules)
        self.assertEqual(select_checkpoints(self.root, self.rules, seed=37)[1], 37)
        # Never mix seeds or stems to fill in a missing rule.
        (self.root / 'other_bptt_seed99.pt').touch()
        with self.assertRaisesRegex(ValueError, 'missing local_diag_rflo, local_direct'):
            select_checkpoints(self.root, self.rules, stem='other_')
        with self.assertRaisesRegex(ValueError, 'No complete checkpoint group'):
            select_checkpoints(self.root, self.rules, self.stem, seed=99)
        self.assertEqual(select_checkpoints(self.root, ['bptt'], stem='other_')[1], 99)

    def test_nested_checkpoint_analysis(self):
        self.make_checkpoints()
        run_dir = self.root / 'dmpn_delaygo_abc123'
        seed_dir = run_dir / 'seed37'
        seed_dir.mkdir(parents=True)
        for rule in self.rules:
            (self.root / f'{self.stem}{rule}_seed37.pt').rename(seed_dir / f'{rule}.pt')
        for directory in (self.root, run_dir, seed_dir):
            stem, seed, paths = select_checkpoints(directory, self.rules, stem=run_dir.name)
            self.assertEqual((stem, seed), (run_dir.name + '_', 37))
            self.assertEqual(paths, {r: seed_dir / f'{r}.pt' for r in self.rules})
        result = self.run_script('visualize_trained_networks.py', '--run-dir', run_dir,
                                 '--analysis', 'weights')
        self.assertIn(f'Selected checkpoint group: {run_dir.name}_seed37', result.stdout)
        self.output = self.output / run_dir.name / 'seed37'
        self.assert_figures(['weight_heatmaps', 'weight_alignment_to_bptt',
                             'bias_alignment_to_bptt', 'parameter_cosine_diag_rflo_vs_direct',
                             'weight_distributions'])
        self.assertTrue((self.output / 'weight_heatmaps.png').is_file())

    def test_nested_and_legacy_groups_never_mix(self):
        self.make_checkpoints()
        run_dir = self.root / 'new_run'
        seed_dir = run_dir / 'seed38'
        seed_dir.mkdir(parents=True)
        (seed_dir / 'bptt.pt').touch()
        # Newer incomplete nested group must not displace complete legacy files.
        self.assertEqual(select_checkpoints(self.root, self.rules)[:2], (self.stem, 37))
        for rule in self.rules[1:]:
            (seed_dir / f'{rule}.pt').touch()
        self.assertEqual(select_checkpoints(self.root, self.rules)[:2], ('new_run_', 38))
        # Same run name and seed in different directories must not be combined.
        for branch, rule in zip(('a', 'b', 'c'), self.rules):
            directory = self.root / branch / 'duplicate' / 'seed37'
            directory.mkdir(parents=True)
            (directory / f'{rule}.pt').touch()
        with self.assertRaisesRegex(ValueError, 'No complete checkpoint group'):
            select_checkpoints(self.root, self.rules, stem='duplicate')

    def test_performance_requires_saved_task_setup(self):
        self.make_checkpoints()
        path = self.root / f'{self.stem}bptt_seed37.pt'
        checkpoint = torch.load(path, weights_only=False)
        del checkpoint['task_params']
        torch.save(checkpoint, path)
        result = subprocess.run(
            [sys.executable, str(_bootstrap.ROOT / 'notebooks/visualize_trained_networks.py'),
             '--ckpt-dir', str(self.root), '--output-dir', str(self.output)],
            capture_output=True, text=True, timeout=120,
            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES=''))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires saved ring-task task_params', result.stderr)
        self.assertFalse(self.output.exists())

    def curve_data(self):
        values = dict(rules=np.asarray(['bptt']), record_steps=np.asarray([0, 10, 20]),
                      ruleset='delaygo', n_hidden=4, n_runs=1, metric='accuracy',
                      acc_label='accuracy (%)', feedback_mode='exact_spatial')
        for split in ('train', 'valid'):
            values[f'mean__bptt__{split}'] = np.asarray([.1, .4, .8])
            values[f'std__bptt__{split}'] = np.asarray([.01, .02, .01])
        return values

    def test_comparison_uses_metadata_and_distinct_source_names(self):
        values = self.curve_data()
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

    def test_comparison_checks_full_architecture_with_short_filenames(self):
        values = self.curve_data()
        mpn_path, rnn_path = self.root / 'dmpn_delaygo_abc123.npz', self.root / 'rnn.npz'
        np.savez(mpn_path, **values, arch_tag='h4-4-4')
        np.savez(rnn_path, **values)
        result = self.run_script('compare_mpn_rnn_performance.py', '--mpn-file', mpn_path,
                                 '--rnn-file', rnn_path, expected_returncode=1)
        self.assertIn('hidden size differs', result.stderr)
        self.assertIn('MPN=(4, 4, 4) vs RNN=4', result.stderr)
        self.assertFalse(self.output.exists())

    def test_comparison_prefers_metadata_over_stale_filename(self):
        values = self.curve_data()
        mpn_path = self.root / 'train_dmpn_oldtask_h999_direct_fa_runs1.npz'
        rnn_path = self.root / 'rnn.npz'
        np.savez(mpn_path, **values, arch_tag='h4')
        np.savez(rnn_path, **values)
        result = self.run_script('compare_mpn_rnn_performance.py', '--mpn-file', mpn_path,
                                 '--rnn-file', rnn_path)
        self.assertIn('task = delaygo   hidden = 4   feedback = exact_spatial', result.stdout)

    def test_comparison_keeps_legacy_filename_fallback(self):
        values = self.curve_data()
        for field in ('ruleset', 'n_hidden', 'feedback_mode'):
            del values[field]
        mpn_path = self.root / 'train_dmpn_delaygo_h4_exact_readout_runs1.npz'
        rnn_path = self.root / 'train_rnn_delaygo_h4_exact_spatial_runs1.npz'
        np.savez(mpn_path, **values)
        np.savez(rnn_path, **values)
        result = self.run_script('compare_mpn_rnn_performance.py', '--mpn-file', mpn_path,
                                 '--rnn-file', rnn_path)
        self.assertIn('task = delaygo   hidden = 4   feedback = exact_spatial', result.stdout)


if __name__ == '__main__':
    unittest.main()
