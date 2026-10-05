"""Training batch CLI reaches task sampling, validation, and saved metadata."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import _bootstrap
import tasks
import train_mpn
import train_common


class TestBatchSize(unittest.TestCase):
    def test_default_and_invalid_sizes(self):
        with patch.object(sys, "argv", ["train_mpn.py"]), patch.object(train_mpn, "BATCH", 128):
            self.assertEqual(train_mpn._parse_args().batch_size, 128)
        for value in ("0", "-1", "1.5"):
            with self.subTest(value=value), patch.object(sys, "argv", ["train_mpn.py", "--batch-size", value]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train_mpn._parse_args()

    def test_pixel_batch_shapes_and_metadata_through_cli(self):
        images = np.zeros((64, 28, 28), dtype=np.float32)
        labels = np.arange(64) % 10
        dataset = {split: (images, labels) for split in ("train", "test")}
        command = ["train_mpn.py", "--task", "seqmnist_pixel", "--hidden", "128", "128", "128",
                   "--batch-size", "16", "--no-grad-align", "--input-mode", "match",
                   "--cross-layer-steps", "0", "--local-bias-mode", "direct",
                   "--rules", "local_direct", "local_diag_rflo", "bptt"]

        def inspect(cfg):
            self.assertEqual(cfg.batch, 16)
            self.assertFalse(cfg.log_grad_align)
            task, train, net = cfg.task.init_params(*cfg.build_params())
            self.assertEqual(train["batch_size"], 16)
            self.assertEqual(train["n_batches"], 16)
            self.assertEqual(train["valid_n_batch"], 48)
            self.assertEqual((net["n_neurons"][0], net["n_neurons"][-1]), (1, 10))
            batch = cfg.task.train_batch(task, train, cfg.batch, torch.device("cpu"), torch.float32)
            valid = cfg.task.valid_batch(task, train, torch.device("cpu"), torch.float32)
            self.assertEqual(tuple(batch[0].shape), (16, 784, 1))
            self.assertEqual(tuple(valid[0].shape), (48, 784, 1))
            self.assertEqual(tuple(batch[1].shape), (16, 784, 10))
            self.assertEqual(float(batch[2][:, :-1].sum()), 0.)
            self.assertTrue(torch.all(batch[2][:, -1] == 1))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"config.json"
                train_common.save_config(cfg, str(path))
                record = json.loads(path.read_text())
                self.assertEqual(record["batch"], 16)
                self.assertEqual(record["train_params"]["batch_size"], 16)
                self.assertEqual(record["train_params"]["valid_n_batch"], 48)
                self.assertFalse(record["log_grad_align"])

        with patch.dict(train_mpn.__dict__), patch.object(sys, "argv", command):
            with patch.object(tasks.SeqMNISTTask, "_load", return_value=dataset):
                with patch.object(train_mpn.tc, "run_experiment", side_effect=inspect) as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        train_mpn.main()
                    run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
