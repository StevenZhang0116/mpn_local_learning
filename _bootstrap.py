"""Path bootstrap: make core/ importable via flat imports.

Importing this module (``import _bootstrap``) prepends this directory's
``core/`` to ``sys.path`` so flat imports such as ``import mpn`` /
``import mpn_tasks`` resolve. Run scripts from this directory
(e.g. ``python train_mpn.py``).
"""
import sys as _sys
import pathlib as _pathlib

_CORE = _pathlib.Path(__file__).resolve().parent / "core"
if str(_CORE) not in _sys.path:
    _sys.path.insert(0, str(_CORE))
