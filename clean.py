#!/usr/bin/env python
# coding: utf-8
"""Delete this project's STALE OUTPUTS — never its code.

Targets (the contents of each directory; the directories themselves are kept):
    checkpoints/                           trained nets + config.json per run
    figure/                                training figures
    figure_data/                           the .npz behind each figure
    log/                                   console logs (log/<run-id>.log)
    notebooks/diagnose_gradients/          analysis outputs
    notebooks/verify_rflo_scaling/         analysis outputs
    notebooks/visualize_trained_networks/  analysis outputs
All of them are git-ignored output folders written by scripts/train_mpn.py,
scripts/train_rnn.py and the notebooks/*.py analysis scripts.

This is destructive, so it is a DRY RUN unless --run is given: without --run it
only prints what would be deleted. It is anchored to the directory this file
lives in (the project root) and refuses to run from a copy that is not the
project. Running experiments write into these folders too — their outputs are
deleted as well, so do not clean while a training job is in progress.

Usage (from anywhere):
    python clean.py                          # preview everything (nothing deleted)
    python clean.py --run                    # delete everything listed
    python clean.py log figure --run         # only some target dirs
    python clean.py --keep 'dmpn_seqmnist*'  # spare entries matching a glob (repeatable)
    python clean.py --keep 'dmpn_seqmnist*' --run
"""
import argparse
import fnmatch
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# short name -> path relative to the project root
TARGET_DIRS = {
    "checkpoints": "checkpoints",
    "figure": "figure",
    "figure_data": "figure_data",
    "log": "log",
    "diagnose_gradients": "notebooks/diagnose_gradients",
    "verify_rflo_scaling": "notebooks/verify_rflo_scaling",
    "visualize_trained_networks": "notebooks/visualize_trained_networks",
}


def is_project_root(root):
    """Guard against running a stray copy of this file somewhere else."""
    root = Path(root)
    return (root / "scripts" / "train_mpn.py").is_file() and (root / "core" / "mpn.py").is_file()


def iter_entries(directory):
    """Top-level entries (files, dirs, symlinks) directly under `directory`, sorted."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return [directory / name for name in sorted(os.listdir(directory))]


def size_of(path):
    """Bytes under `path` (a symlink counts as 0; it is removed as a link, never followed)."""
    path = Path(path)
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        # do not descend into symlinked directories
        dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def plan(root=ROOT, dirs=None, keep=()):
    """What would be deleted. Returns {name: {'path', 'exists', 'delete', 'kept'}}.
    `keep` is a list of fnmatch globs matched against entry NAMES; matching
    entries are listed under 'kept' and never deleted."""
    root = Path(root).resolve()
    names = list(TARGET_DIRS) if not dirs else list(dirs)
    result = {}
    for name in names:
        if name not in TARGET_DIRS:
            raise ValueError(f"unknown target {name!r}; choose from {list(TARGET_DIRS)}")
        path = root / TARGET_DIRS[name]
        entries = iter_entries(path)
        kept = [e for e in entries if any(fnmatch.fnmatch(e.name, pat) for pat in keep)]
        delete = [e for e in entries if e not in kept]
        result[name] = {"path": path, "exists": path.is_dir(), "delete": delete, "kept": kept}
    return result


def delete(entries):
    """Remove each top-level entry: symlinks as links (never followed), files,
    or whole directory trees."""
    for e in entries:
        if e.is_symlink() or e.is_file():
            e.unlink()
        elif e.is_dir():
            shutil.rmtree(e)


def main(argv=None, root=ROOT):
    p = argparse.ArgumentParser(
        description="Delete stale outputs (checkpoints/, figure/, figure_data/, log/ and the "
                    "notebooks/* analysis output folders). DRY RUN unless --run is given.")
    # NOTE: no argparse `choices` here — combined with nargs="*" and no argument
    # given, Python < 3.12 validates the empty default against the choices and
    # fails with "invalid choice: []". Names are validated below instead.
    p.add_argument("dirs", nargs="*", metavar="DIR",
                   help="limit to these target dirs (default: all of them); one of: "
                        + ", ".join(TARGET_DIRS))
    p.add_argument("--run", action="store_true",
                   help="actually delete; without it nothing is removed")
    p.add_argument("--keep", action="append", default=[], metavar="GLOB",
                   help="spare top-level entries whose NAME matches this glob "
                        "(e.g. 'dmpn_seqmnist*'); repeatable")
    args = p.parse_args(argv)

    unknown = [d for d in args.dirs if d not in TARGET_DIRS]
    if unknown:
        p.error(f"unknown target dir(s) {unknown}; choose from {list(TARGET_DIRS)}")
    root = Path(root).resolve()
    if not is_project_root(root):
        p.error(f"{root} does not look like the mpn_local_learning project root; refusing to run")

    todo = plan(root, args.dirs or None, args.keep)
    n_entries = n_bytes = 0
    print(f"{'DELETING' if args.run else 'DRY RUN (pass --run to delete)'} under {root}")
    for name, info in todo.items():
        rel = TARGET_DIRS[name]
        if not info["exists"]:
            print(f"  {rel:40} (missing)")
            continue
        if not info["delete"] and not info["kept"]:
            print(f"  {rel:40} (empty)")
            continue
        size = sum(size_of(e) for e in info["delete"])
        n_entries += len(info["delete"])
        n_bytes += size
        print(f"  {rel:40} {len(info['delete'])} item(s), {human(size)}"
              + (f", {len(info['kept'])} kept" if info["kept"] else ""))
        for e in info["delete"][:10]:
            print(f"      - {e.name}")
        if len(info["delete"]) > 10:
            print(f"      … and {len(info['delete']) - 10} more")
        for e in info["kept"]:
            print(f"      (keep) {e.name}")
    print(f"Total: {n_entries} top-level item(s), {human(n_bytes)}.")
    if n_entries == 0:
        print("Nothing to delete.")
        return 0
    if not args.run:
        print("Nothing deleted (dry run). Re-run with --run to delete.")
        return 0
    for info in todo.values():
        delete(info["delete"])
    print(f"Deleted {n_entries} item(s), {human(n_bytes)}. The target directories themselves were kept.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
