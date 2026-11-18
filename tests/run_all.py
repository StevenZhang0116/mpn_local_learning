#!/usr/bin/env python
"""Run the whole test suite with one command and print one verdict line per file.

    python tests/run_all.py            # everything that can run on this machine
    python tests/run_all.py -k norm    # only files whose name contains "norm"

The suite mixes two styles, so this runner starts one subprocess per file:
  * unittest modules  (most test_*.py)        -> python -m unittest <module>
  * check scripts     (SCRIPTS below)         -> python <file>, pass = exit code 0
test_mpn_revise.py needs a CUDA GPU and is skipped, with a note, when there is none.
Exits non-zero if any file fails. Run it on a compute node, not on a login node.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ("validate_local_learning.py", "test_tasks.py", "test_mpn_revise.py")
NEEDS_CUDA = ("test_mpn_revise.py",)


def has_cuda():
    code = "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    return subprocess.run([sys.executable, "-c", code], cwd=HERE).returncode == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-k", metavar="TEXT", default="",
                        help="run only the files whose name contains TEXT")
    args = parser.parse_args(argv)

    modules = sorted(p.name for p in HERE.glob("test_*.py") if p.name not in SCRIPTS)
    files = [f for f in modules + list(SCRIPTS) if args.k in f]
    cuda = has_cuda() if any(f in NEEDS_CUDA for f in files) else False

    failed, skipped = [], []
    for name in files:
        if name in NEEDS_CUDA and not cuda:
            print(f"{name:<40} SKIP  (needs a CUDA GPU)")
            skipped.append(name)
            continue
        command = ([sys.executable, name] if name in SCRIPTS
                   else [sys.executable, "-m", "unittest", name[:-3]])
        start = time.perf_counter()
        result = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
        verdict = "ok" if result.returncode == 0 else "FAIL"
        print(f"{name:<40} {verdict:<5} {time.perf_counter() - start:6.1f}s", flush=True)
        if result.returncode != 0:
            failed.append(name)
            tail = (result.stderr or result.stdout).strip().splitlines()[-15:]
            print("\n".join("    " + line for line in tail))

    print(f"\n{len(files) - len(failed) - len(skipped)} passed, {len(failed)} failed, "
          f"{len(skipped)} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
