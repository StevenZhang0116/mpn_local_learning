"""Console logging checks without running a training experiment."""

import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import _bootstrap
import run_logging


class TestRunLogging(unittest.TestCase):
    def test_mirrors_both_streams_and_flushes(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with run_logging.tee_output("train_mpn", directory) as log_path:
                    self.assertEqual(sys.stdout.write("training output\n"), 16)
                    print("warning output", file=sys.stderr)
                    sys.stdout.write("partial output")
                    sys.stdout.flush()
                    contents = log_path.read_text(encoding="utf-8")
                    self.assertIn("training output\nwarning output\npartial output", contents)
                    self.assertEqual(sys.stdout.isatty(), stdout.isatty())
                    self.assertEqual(sys.stdout.encoding, stdout.encoding)
                    self.assertTrue(sys.stdout.writable())
                self.assertIs(sys.stdout, stdout)
                self.assertIs(sys.stderr, stderr)
            self.assertIn(str(log_path), stdout.getvalue())
            self.assertIn("training output", stdout.getvalue())
            self.assertNotIn("warning output", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "warning output\n")

    def test_default_path_is_project_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(run_logging, "__file__", str(root / "core" / "run_logging.py")):
                with contextlib.redirect_stdout(io.StringIO()):
                    with run_logging.tee_output("train_mpn") as log_path:
                        self.assertEqual(log_path.parent, root / "log")
                        self.assertRegex(log_path.name, rf"^train_mpn_\d{{8}}_\d{{6}}_{os.getpid()}\.log$")
                self.assertTrue(log_path.is_file())

    def test_restores_streams_and_closes_log_on_exception(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    with run_logging.tee_output("train_mpn", directory) as log_path:
                        log_file = sys.stdout.log_file
                        print("before failure")
                        raise RuntimeError("training failed")
                self.assertIs(sys.stdout, stdout)
                self.assertIs(sys.stderr, stderr)
                self.assertTrue(log_file.closed)
            self.assertIn("before failure", log_path.read_text(encoding="utf-8"))

    def test_train_mpn_cli_captures_experiment_output(self):
        import train_common

        def fake_experiment(config):
            self.assertEqual(config.feedback_mode, "direct_fa")
            self.assertEqual(config.n_datasets, 1)
            print("experiment stdout")
            print("experiment stderr", file=sys.stderr)

        argv = ["train_mpn.py", "--dfa", "--task", "delaygo", "--steps", "1",
                "--runs", "1", "--hidden", "4", "4"]
        script = _bootstrap.ROOT / "scripts" / "train_mpn.py"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(run_logging, "__file__", str(root / "core" / "run_logging.py")):
                with patch.object(sys, "argv", argv):
                    with patch.object(train_common, "run_experiment", side_effect=fake_experiment) as runner:
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            runpy.run_path(str(script), run_name="__main__")
            runner.assert_called_once()
            logs = list((root / "log").glob("train_mpn_*.log"))
            self.assertEqual(len(logs), 1)
            contents = logs[0].read_text(encoding="utf-8")
            self.assertIn("experiment stdout", contents)
            self.assertIn("experiment stderr", contents)
            self.assertIn("experiment stdout", stdout.getvalue())
            self.assertIn("experiment stderr", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
