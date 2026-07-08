"""Path bootstrap: make the shared library and sibling scripts importable.

Importing this module (``import _bootstrap``) walks up to the project root
(the ``mpn_local_learning`` directory) and prepends both ``core/`` (the shared
library: ``mpn``, ``mpn_tasks``, ...) and ``scripts/`` (so scripts/notebooks can
``import train_mpn``) to ``sys.path``. It also exposes ``ROOT`` so output dirs
(figure/, checkpoints/) anchor to the project root regardless of the working
directory a script or notebook is launched from.
"""
import sys as _sys
import pathlib as _pathlib

# This file lives at <root>/scripts/_bootstrap.py (or <root>/notebooks/_bootstrap.py);
# the project root is its parent's parent.
ROOT = _pathlib.Path(__file__).resolve().parent.parent

for _sub in ("core", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
