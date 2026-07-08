#!/usr/bin/env python
# coding: utf-8
"""
Remove saved results: the contents of <root>/figure, <root>/figure_data, and
<root>/checkpoints (the outputs of train_mpn.py / train_rnn.py).

Safe by default: previews what would be deleted and asks for confirmation. The
directories themselves are kept (only their contents are removed).

Usage (run from anywhere):
    python clean.py                 # preview + confirm, then delete all three
    python clean.py --yes           # skip the confirmation prompt
    python clean.py --dry-run       # only show what would be deleted
    python clean.py figure          # limit to specific dirs (figure/figure_data/checkpoints)
    python clean.py checkpoints figure_data --yes
"""
import argparse
import os
import shutil

import _bootstrap  # exposes ROOT (the project root)

# The result directories this tool manages, keyed by their short name.
RESULT_DIRS = {
    "figure": _bootstrap.ROOT / "figure",
    "figure_data": _bootstrap.ROOT / "figure_data",
    "checkpoints": _bootstrap.ROOT / "checkpoints",
}


def _iter_entries(d):
    """Yield the paths directly under directory d (files and subdirs)."""
    if not d.is_dir():
        return
    for name in sorted(os.listdir(d)):
        yield d / name


def _summarize(dirs):
    """Return (per-dir entry lists, total file count, total bytes)."""
    plan, n_files, n_bytes = {}, 0, 0
    for name in dirs:
        entries = list(_iter_entries(RESULT_DIRS[name]))
        plan[name] = entries
        for e in entries:
            if e.is_file():
                n_files += 1
                n_bytes += e.stat().st_size
            elif e.is_dir():
                for root, _, files in os.walk(e):
                    for f in files:
                        n_files += 1
                        n_bytes += os.path.getsize(os.path.join(root, f))
    return plan, n_files, n_bytes


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def _delete(entries):
    for e in entries:
        if e.is_dir():
            shutil.rmtree(e)
        else:
            e.unlink()


def main():
    p = argparse.ArgumentParser(
        description="Clean saved results (figure/, figure_data/, checkpoints/).")
    p.add_argument("dirs", nargs="*", choices=list(RESULT_DIRS),
                   help="which result dirs to clean (default: all three)")
    p.add_argument("--yes", "-y", action="store_true", help="skip confirmation")
    p.add_argument("--dry-run", action="store_true",
                   help="only show what would be deleted")
    args = p.parse_args()

    dirs = args.dirs if args.dirs else list(RESULT_DIRS)
    plan, n_files, n_bytes = _summarize(dirs)

    print("Result directories to clean:")
    total_entries = 0
    for name in dirs:
        entries = plan[name]
        total_entries += len(entries)
        loc = RESULT_DIRS[name]
        if not loc.is_dir():
            print(f"  {name:12} (missing — nothing to do)")
        elif not entries:
            print(f"  {name:12} (empty)")
        else:
            print(f"  {name:12} {len(entries)} item(s)  [{loc}]")
            for e in entries[:8]:
                print(f"      - {e.name}")
            if len(entries) > 8:
                print(f"      … and {len(entries) - 8} more")
    print(f"\nTotal: {n_files} file(s), {_human(n_bytes)} across {total_entries} top-level item(s).")

    if total_entries == 0:
        print("Nothing to delete.")
        return 0
    if args.dry_run:
        print("Dry run — nothing deleted.")
        return 0
    if not args.yes:
        resp = input("Delete these? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return 1

    for name in dirs:
        _delete(plan[name])
    print(f"Deleted contents of: {', '.join(dirs)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
