"""CLI decay overrides reach every MP layer and survive saved configuration."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import _bootstrap
import train_mpn
import train_common


class TestLambdaOverride(unittest.TestCase):
    def test_defaults_and_invalid_arguments(self):
        with patch.object(sys, "argv", ["train_mpn.py"]):
            self.assertIsNone(train_mpn._parse_args().lam)
        with patch.object(train_mpn, "LAM", None):
            params = train_mpn.build_params()[2]["ml_params"]
        self.assertEqual(params["m_time_scale"], 4000)
        self.assertNotIn("lam_clamp", params)
        for value in ("-0.1", "1", "1.1", "nan", "inf"):
            with self.subTest(value=value), patch.object(sys, "argv", ["train_mpn.py", "--lam", value]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train_mpn._parse_args()

    def test_cli_to_dynamics_metadata_and_reload(self):
        for kind, hidden in (("dmpn", ["2", "2"]), ("mpn1", ["2"])):
            for decay in (0., .9):
                with self.subTest(kind=kind, decay=decay), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)

                    def inspect_config(cfg):
                        _, _, params = cfg.task.init_params(*cfg.build_params())
                        self.assertEqual(params["ml_params"]["lam_clamp"], decay)
                        self.assertNotIn("m_time_scale", params["ml_params"])
                        net = cfg.net_factory(params, False).double()
                        layers = net.mp_layers if kind == "dmpn" else [net.mp_layer]
                        with torch.no_grad():
                            for layer in layers:
                                torch.testing.assert_close(layer.lam, torch.full_like(layer.lam, decay))
                                self.assertFalse(layer.lam_train)
                                layer.M_init.fill_(.5)
                                layer.reset_state(B=2)
                                layer.update_M_matrix_local_fast(
                                    torch.zeros(2, layer.n_input, dtype=torch.double),
                                    torch.zeros(2, layer.n_output, dtype=torch.double))
                                torch.testing.assert_close(layer.M, torch.full_like(layer.M, .5*decay))
                        path = root / "net.pt"
                        torch.save(dict(net_params=params, state_dict=net.state_dict(), learning_rule="bptt"), path)
                        loaded = train_mpn.load_net(path, device=torch.device("cpu"), dtype=torch.double)
                        restored = loaded.mp_layers if kind == "dmpn" else [loaded.mp_layer]
                        for layer in restored:
                            torch.testing.assert_close(layer.lam, torch.full_like(layer.lam, decay))
                            self.assertAlmostEqual(layer.m_time_scale, 40 / (1-decay))
                        train_common.save_config(cfg, str(root / "config.json"))
                        record = json.loads((root / "config.json").read_text())
                        self.assertEqual(record["net_params"]["ml_params"]["lam_clamp"], decay)

                    command = ["train_mpn.py", "--net", kind, "--hidden", *hidden,
                               "--task", "delaygo", "--lam", str(decay), "--no-residual"]
                    # Exercise main's argument-to-global wiring without launching training.
                    with patch.dict(train_mpn.__dict__), patch.object(sys, "argv", command):
                        with patch.object(train_mpn.tc, "run_experiment", side_effect=inspect_config) as run:
                            with contextlib.redirect_stdout(io.StringIO()):
                                train_mpn.main()
                            run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
