#!/usr/bin/env python
# coding: utf-8
"""Thin wrapper: the output cleaner now lives at <root>/clean.py (covers
checkpoints/, figure/, figure_data/, log/ and the notebooks/* analysis output
folders; dry run by default, --run to delete). This forwards to it so the old
path keeps working; see `python clean.py --help` at the project root."""
import sys

import _bootstrap  # exposes ROOT (the project root)

sys.path.insert(0, str(_bootstrap.ROOT))
import clean  # noqa: E402  (<root>/clean.py)

if __name__ == "__main__":
    sys.exit(clean.main())
