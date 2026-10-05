"""Gradient invariants on tiny CPU models and CUDA-only CLI behavior, without training."""
from copy import deepcopy
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import _bootstrap
import mpn
import train_mpn
from test_input_modulation_trace import make_net, data

sys.path.insert(0, str(_bootstrap.ROOT / "notebooks"))
import diagnose_gradients as diagnostic


class TestGradientDiagnostics(unittest.TestCase):
    def test_pair_selection_never_mixes_seeds_or_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for folder, rules in (("run_a/seed1", diagnostic.RULES[:2]),
                                  ("run_a/seed2", ("local_direct",)),
                                  ("run_b/seed2", ("local_diag_rflo",))):
                seed_dir = root / folder
                seed_dir.mkdir(parents=True)
                for rule in rules:
                    path = seed_dir / f"{rule}.pt"
                    path.touch()
                    os.utime(path, ns=(100, 100) if folder.endswith("seed1") else (200, 200))
            paths = diagnostic.select_sources(run_dir=root)
            self.assertEqual({p.parent for p in paths.values()}, {root / "run_a/seed1"})
            self.assertEqual(paths, diagnostic.select_sources(run_dir=root / "run_a/seed1"))
            with self.assertRaisesRegex(ValueError, "No complete checkpoint group"):
                diagnostic.select_sources(run_dir=root / "run_a/seed2")
            for rule in diagnostic.RULES[:2]:
                path = root / f"run_a/seed3/{rule}.pt"
                path.parent.mkdir(exist_ok=True)
                path.touch()
                os.utime(path, ns=(300, 300))
            paths = diagnostic.select_sources(run_dir=root / "run_a")
            self.assertEqual({p.parent.name for p in paths.values()}, {"seed3"})

    def test_pair_rejects_mismatched_metadata_and_configuration(self):
        net, cfg = make_net(input_mode="match")
        checkpoints = {rule: dict(net_params=dict(cfg, learning_rule=rule),
                                 state_dict=net.state_dict(), learning_rule=rule,
                                 run_id="run_a", seed=13, ruleset="delaygo")
                       for rule in diagnostic.RULES[:2]}
        diagnostic.validate_pair(checkpoints)
        for key, value in (("run_id", "run_b"), ("seed", 14), ("ruleset", "delayanti"),
                           ("learning_rule", "bptt")):
            with self.subTest(key=key):
                changed = deepcopy(checkpoints)
                changed["local_diag_rflo"][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    diagnostic.validate_pair(changed)
        changed = deepcopy(checkpoints)
        changed["local_diag_rflo"]["net_params"]["feedback_mode"] = "random"
        with self.assertRaisesRegex(ValueError, "network configurations"):
            diagnostic.validate_pair(changed)

    def test_paired_reports_share_one_batch_and_keep_distinct_weights(self):
        net, cfg = make_net(input_mode="match")
        batch = data()[:3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_dir = root / "run_a/seed13"
            seed_dir.mkdir(parents=True)
            originals = {}
            for rule in diagnostic.RULES[:2]:
                state = deepcopy(net.state_dict())
                if rule == "local_diag_rflo":
                    state["W_output"] += .5
                path = seed_dir / f"{rule}.pt"
                torch.save(dict(net_params=dict(cfg, learning_rule=rule), state_dict=state,
                                learning_rule=rule, run_id="run_a", seed=13, ruleset="delaygo"), path)
                originals[path] = diagnostic.file_fingerprint(path)
            args = SimpleNamespace(checkpoint=None, run_dir=root / "run_a", batch_file=None,
                                   dtype="saved", feedback="saved", output_dir=None)
            reports_dir = root / "notebooks" / "diagnose_gradients" / "run_a"
            # Only the reusable orchestration helper runs on CPU; the CLI still requires CUDA.
            with patch.object(diagnostic._bootstrap, "ROOT", root), \
                    patch.object(diagnostic, "make_batch", return_value=batch) as generate:
                with contextlib.redirect_stdout(io.StringIO()):
                    diagnostic.run_diagnostics(args, torch.device("cpu"))
            generate.assert_called_once()
            summary = json.loads((reports_dir / "summary.json").read_text())
            reports = summary["checkpoints"]
            expected_hash = diagnostic.fingerprint(batch)
            self.assertEqual(summary["batch"]["sha256"], expected_hash)
            self.assertNotEqual(reports["local_direct"]["losses"]["bptt"],
                                reports["local_diag_rflo"]["losses"]["bptt"])
            for rule in diagnostic.RULES[:2]:
                self.assertEqual(reports[rule]["batch"]["sha256"], expected_hash)
                self.assertEqual(set(reports[rule]["losses"]), set(diagnostic.RULES))
                self.assertFalse((reports_dir / rule).exists())
            self.assertEqual({path.name for path in reports_dir.iterdir()},
                             {"batch.pt", "summary.json", "gradient_metrics.csv", "trace_steps.csv",
                              "trace_summary.csv", "checkpoint_comparison.png", "trace_comparison.png"})
            for name in ("gradient_metrics", "trace_steps", "trace_summary"):
                with (reports_dir / f"{name}.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual({row["source_checkpoint"] for row in rows}, set(diagnostic.RULES[:2]))
            payload = torch.load(reports_dir / "batch.pt", weights_only=False)
            self.assertEqual(diagnostic.fingerprint(tuple(payload[k] for k in ("inputs", "labels", "masks"))),
                             expected_hash)
            for path, before in originals.items():
                self.assertEqual(diagnostic.file_fingerprint(path), before)
            for name in ("checkpoint_comparison.png", "trace_comparison.png"):
                with Image.open(reports_dir / name) as image:
                    self.assertGreater(np.asarray(image.convert("RGB")).std(), 1)

    def test_zero_plasticity_matches_full_bptt(self):
        net, _ = make_net(input_mode="match")
        for layer in net.mp_layers:
            layer.eta.zero_()
        batch = data()[:3]
        result = diagnostic.diagnose(net, batch)
        for rule in diagnostic.RULES[:2]:
            for key, expected in result["gradients"]["bptt"].items():
                torch.testing.assert_close(result["gradients"][rule][key], expected,
                                           rtol=1e-10, atol=1e-12)
        self.assertTrue(all(row["wa_rms"] == 0 for row in result["trace_rows"]))
        self.assertTrue(all(check["forward_matches_bptt"] for check in result["forward_checks"].values()))

    def test_recorder_preserves_gradients_and_observes_pre_update_trace(self):
        net, _ = make_net(input_mode="match")
        batch = data()[:3]
        state = {k: v.clone() for k, v in net.state_dict().items()}
        batch_hash = diagnostic.fingerprint(batch)
        expected = {}
        for rule, method in (("bptt", "bptt_gradients"),
                             ("local_direct", "local_direct_gradients"),
                             ("local_diag_rflo", "local_diag_rflo_gradients")):
            clone = deepcopy(net)
            expected[rule] = getattr(clone, method)(*batch)
        result = diagnostic.diagnose(net, batch)
        for rule in diagnostic.RULES:
            for key, actual in result["gradients"][rule].items():
                torch.testing.assert_close(actual, expected[rule][key], rtol=0, atol=0)
        self.assertEqual(diagnostic.fingerprint(batch), batch_hash)
        self.assertTrue(all(torch.equal(value, net.state_dict()[key]) for key, value in state.items()))
        rows = result["trace_rows"]
        self.assertEqual(len(rows), len(net.mp_layers) * batch[0].shape[1])
        self.assertTrue(all(row["wa_rms"] == 0 for row in rows if row["step"] == 0))

        # Independently calculate the first trace write, which is consumed at t=1.
        oracle = deepcopy(net)
        oracle.reset_state(B=batch[0].shape[0])
        with torch.no_grad():
            _, h, _, phi, _ = oracle._forward_local_stack(batch[0][:, 0])
            for index, layer in enumerate(oracle.mp_layers):
                eta, _ = layer._eta_lam_full()
                trace = eta[None] * phi[index][..., None] * h[index][:, None, :].square() * (1 + layer.M)
                wa_rms = diagnostic.rms(layer.W[None] * trace)
                recorded = next(r for r in rows if r["layer"] == f"MP{index + 1}" and r["step"] == 1)
                self.assertAlmostEqual(recorded["wa_rms"], wa_rms, places=12)

    def test_clipping_and_pre_only_updates(self):
        for kind in ("hebb_assoc", "hebb_pre"):
            with self.subTest(kind=kind):
                net, _ = make_net(input_mode="match", write_mode="hard", kind=kind)
                for layer in net.mp_layers:
                    layer.eta.fill_(1000)
                result = diagnostic.diagnose(net, data()[:3])
                self.assertGreater(max(row["write_clipped_fraction"] for row in result["trace_rows"]), 0)
                if kind == "hebb_pre":
                    self.assertTrue(all(row["wa_rms"] == 0 for row in result["trace_rows"]))

    def test_undefined_metrics_are_not_reported_as_zero_alignment(self):
        value = diagnostic.compare_vectors(torch.zeros(2), torch.ones(2))
        self.assertIsNone(value["cosine"])
        self.assertEqual(value["left_norm"], 0)
        self.assertEqual(value["relative_l2_error"], 1)
        value = diagnostic.compare_vectors(torch.ones(2), torch.zeros(2))
        self.assertIsNone(value["norm_ratio"])
        self.assertIsNone(value["relative_l2_error"])
        value = diagnostic.compare_vectors(torch.tensor([float("inf")]), torch.ones(1))
        self.assertIsNone(value["cosine"])
        self.assertIsNone(value["left_norm"])
        self.assertEqual(value["left_nonfinite_fraction"], 1)
        json.dumps(value, allow_nan=False)

    def test_rejects_custom_loss_and_uninformative_batch(self):
        with self.assertRaisesRegex(ValueError, "masked-MSE"):
            diagnostic.build_model(dict(net_params=dict(net_type="dmpn", loss_type="cross_entropy")),
                                   torch.device("cpu"), torch.double)
        x, y, mask = data()[:3]
        with self.assertRaisesRegex(ValueError, "all-zero"):
            diagnostic.validate_batch((x, y, torch.zeros_like(mask)))

    def test_cli_requires_cuda_before_loading_checkpoint(self):
        command = [sys.executable, str(_bootstrap.ROOT / "notebooks/diagnose_gradients.py"),
                   "--checkpoint", "nonexistent_checkpoint.pt"]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        proc = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("CUDA is required", proc.stderr)
        self.assertNotIn("FileNotFoundError", proc.stderr)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CLI export integration")
    def test_cli_exports_reusable_batch_without_changing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.multiple(train_mpn, N_HIDDEN=[3, 3], RULESET="delaygo", INPUT_MODE="paired",
                                MP_RESIDUAL=False, CROSS_LAYER_STEPS=0, LOCAL_BIAS_MODE="exact"):
                cfg = train_mpn._cfg()
                task_params, train_params, net_params = cfg.task.init_params(*cfg.build_params())
            with contextlib.redirect_stdout(io.StringIO()):
                net = mpn.DeepMultiPlasticNet(net_params, verbose=False)
            path = root / "local_diag_rflo.pt"
            torch.save(dict(net_params=net_params, task_params=task_params, train_params=train_params,
                            state_dict=net.state_dict(), ruleset="delaygo", learning_rule="local_diag_rflo"), path)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            command = [sys.executable, str(_bootstrap.ROOT / "notebooks/diagnose_gradients.py"),
                       "--checkpoint", str(path), "--dtype", "float64"]
            environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                               OPENBLAS_NUM_THREADS="1")
            for name, options in (("first", []),
                                  ("second", ["--batch-file", str(root / "first/batch.pt")])):
                proc = subprocess.run(command + options + ["--output-dir", str(root / name)],
                                      cwd=root, env=environment, capture_output=True, text=True, timeout=120)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            first, second = [json.loads((root / name / "summary.json").read_text()) for name in ("first", "second")]
            self.assertEqual(first["batch"]["shape"][0], 128)
            self.assertEqual(first["batch"]["seed"], 0)
            self.assertTrue(first["diagnostic_settings"]["device"].startswith("cuda"))
            self.assertEqual(first["batch"]["sha256"], second["batch"]["sha256"])
            self.assertEqual(first["losses"], second["losses"])
            self.assertEqual(first["saved_settings"]["input_mode"], "paired")
            self.assertEqual(first["diagnostic_settings"]["input_mode"], "match")
            self.assertEqual(first["diagnostic_settings"]["local_bias_mode"], "direct")
            self.assertEqual((root / "first/gradient_metrics.csv").read_text(),
                             (root / "second/gradient_metrics.csv").read_text())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            for name in ("gradient_comparison.png", "trace_diagnostics.png"):
                with Image.open(root / "first" / name) as image:
                    self.assertGreater(image.width, 100)
                    self.assertGreater(np.asarray(image.convert("RGB")).std(), 1)
            self.assertTrue((root / "first/trace_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
