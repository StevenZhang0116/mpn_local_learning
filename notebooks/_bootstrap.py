"""Path bootstrap: make the shared library and the scripts importable.

Importing this module (``import _bootstrap``) walks up from its own location to
the project root (the ``mpn_local_learning`` directory, recognized by
``core/mpn.py``) and prepends three folders to ``sys.path``:

  core/              the shared library (``mpn``, ``rnn``, ``gru``, ``mpn_tasks``, ...)
  scripts/training/  the training entry points (``train_mpn``, ``train_rnn``,
                     ``train_gru``) and their shared loop (``train_common``)
  scripts/tasks/     the task adapters (``tasks``, ``adding_tasks``,
                     ``associative_recall_tasks``)

so every module keeps its flat name (``import train_mpn``, ``import tasks``)
wherever the importing file lives. It also exposes ``ROOT`` so output dirs
(figure/, checkpoints/) anchor to the project root regardless of the working
directory a script is launched from.

Identical copies of this file sit next to the code that runs as a script:
scripts/training/, scripts/tasks/, notebooks/, tests/ and utils/.
"""
import sys as _sys
import pathlib as _pathlib

# Depth-independent: the nearest ancestor that holds core/mpn.py is the project root.
ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents
            if (_p / "core" / "mpn.py").is_file())

for _sub in ("core", "scripts/training", "scripts/tasks"):
    _p = str(ROOT / _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
