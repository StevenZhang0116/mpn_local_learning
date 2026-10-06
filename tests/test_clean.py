"""clean.py: dry run by default, --run deletes only the contents of the target
output folders, --keep spares entries, symlinks are unlinked not followed, and a
non-project root is refused. Run from tests/: python -m unittest test_clean -v"""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

import _bootstrap
sys.path.insert(0, str(_bootstrap.ROOT))
import clean  # noqa: E402


def make_fake_root(tmp):
    root = Path(tmp) / "proj"
    (root / "scripts").mkdir(parents=True)
    (root / "core").mkdir()
    (root / "scripts" / "train_mpn.py").write_text("# code\n")
    (root / "core" / "mpn.py").write_text("# code\n")
    for rel in clean.TARGET_DIRS.values():
        (root / rel).mkdir(parents=True)
    # outputs
    run = root / "checkpoints" / "dmpn_delaygo_20261005_140034_aa1b2ce0c926"
    (run / "seed3").mkdir(parents=True)
    (run / "config.json").write_text("{}")
    (run / "seed3" / "bptt.pt").write_bytes(b"x" * 2048)
    keep_run = root / "checkpoints" / "dmpn_seqmnist_20261005_000000_deadbeef0000"
    keep_run.mkdir()
    (keep_run / "config.json").write_text("{}")
    (root / "figure" / "a.png").write_bytes(b"p" * 10)
    (root / "figure_data" / "a.npz").write_bytes(b"n" * 10)
    (root / "log" / "a.log").write_text("log\n")
    (root / "notebooks" / "diagnose_gradients" / "r1").mkdir()
    (root / "notebooks" / "diagnose_gradients" / "r1" / "summary.json").write_text("{}")
    (root / "notebooks" / "visualize_trained_networks" / "w.png").write_bytes(b"w")
    # a symlink inside an output dir pointing at CODE must be unlinked, never followed
    outside = root / "scripts" / "precious.txt"
    outside.write_text("do not delete")
    os.symlink(outside, root / "figure" / "link_to_code.txt")
    os.symlink(root / "scripts", root / "log" / "link_to_scripts")
    return root, outside


class TestClean(unittest.TestCase):
    def run_main(self, argv, root):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = clean.main(argv, root=root)
        return code, out.getvalue()

    def test_dry_run_deletes_nothing_and_lists_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = make_fake_root(tmp)
            before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            code, text = self.run_main([], root)
            self.assertEqual(code, 0)
            self.assertIn("DRY RUN", text)
            self.assertIn("Nothing deleted", text)
            for rel in clean.TARGET_DIRS.values():
                self.assertIn(rel, text)
            self.assertEqual(sorted(str(p.relative_to(root)) for p in root.rglob("*")), before)

    def test_run_deletes_only_output_contents_and_keeps_dirs_and_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, outside = make_fake_root(tmp)
            code, text = self.run_main(["--run"], root)
            self.assertEqual(code, 0)
            self.assertIn("Deleted", text)
            for rel in clean.TARGET_DIRS.values():
                d = root / rel
                self.assertTrue(d.is_dir(), rel)
                self.assertEqual(os.listdir(d), [], rel)
            # code untouched, including the symlink targets
            self.assertTrue(outside.is_file())
            self.assertEqual(outside.read_text(), "do not delete")
            self.assertTrue((root / "scripts" / "train_mpn.py").is_file())
            self.assertTrue((root / "core" / "mpn.py").is_file())

    def test_keep_and_dir_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = make_fake_root(tmp)
            code, text = self.run_main(["checkpoints", "log", "--keep", "dmpn_seqmnist*", "--run"], root)
            self.assertEqual(code, 0)
            self.assertIn("(keep) dmpn_seqmnist_20261005_000000_deadbeef0000", text)
            self.assertEqual(os.listdir(root / "checkpoints"), ["dmpn_seqmnist_20261005_000000_deadbeef0000"])
            self.assertEqual(os.listdir(root / "log"), [])
            # untouched dirs
            self.assertEqual(os.listdir(root / "figure_data"), ["a.npz"])
            self.assertTrue((root / "figure" / "a.png").is_file())
            self.assertTrue((root / "notebooks" / "diagnose_gradients" / "r1" / "summary.json").is_file())

    def test_refuses_non_project_root_and_unknown_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "checkpoints").mkdir()
            (Path(tmp) / "checkpoints" / "x").write_text("x")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                clean.main(["--run"], root=tmp)
            self.assertTrue((Path(tmp) / "checkpoints" / "x").is_file())
            root, _ = make_fake_root(tmp)
            with self.assertRaises(ValueError):
                clean.plan(root, dirs=["scripts"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                clean.main(["scripts", "--run"], root=root)
            self.assertTrue((root / "scripts" / "train_mpn.py").is_file())

    def test_real_project_root_is_recognized_and_wrapper_forwards(self):
        self.assertTrue(clean.is_project_root(_bootstrap.ROOT))
        self.assertEqual(clean.ROOT, _bootstrap.ROOT.resolve())
        wrapper = (_bootstrap.ROOT / "utils" / "clean.py").read_text()
        self.assertIn("clean.main()", wrapper)


if __name__ == "__main__":
    unittest.main()
