"""Small helpers for teeing script output to a log file.

The log opens under a timestamped name (so even argument errors are captured) and
is renamed to the run's output ID once that is known — see rename_active_log — so
a run's log, figure, plot data and checkpoint folder all share one name:
    log/<run-id>.log  figure/<run-id>.png  figure_data/<run-id>.npz  checkpoints/<run-id>/
"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import os
import sys
import threading


class _TeeStream:
    """Write to both the original terminal stream and a shared log file."""

    def __init__(self, terminal, log_file, lock):
        self.terminal = terminal
        self.log_file = log_file
        self.lock = lock

    def write(self, text):
        with self.lock:
            self.terminal.write(text)
            self.log_file.write(text)
        return len(text)

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    @property
    def encoding(self):
        return self.terminal.encoding

    def __getattr__(self, name):
        return getattr(self.terminal, name)


# Innermost-last stack of the tee logs currently open in this process; each entry
# is a mutable holder so rename_active_log can update the path the tee writes to.
_ACTIVE_LOGS = []


def active_log_path():
    """Path of the innermost active tee log, or None when no tee is open."""
    return _ACTIVE_LOGS[-1]["path"] if _ACTIVE_LOGS else None


def rename_active_log(stem):
    """Rename the innermost active tee log to ``<log_dir>/<stem>.log`` and keep
    writing to it (the open handle follows the renamed inode on POSIX). Called by
    train_mpn.main once the run ID shared by the figure / .npz / checkpoint folder
    is known, so the log carries that ID instead of a timestamp. Returns the new
    path, or None when no tee is active (e.g. main() called from a test). An
    unlikely name clash falls back to ``<stem>_<pid>.log``."""
    if not _ACTIVE_LOGS:
        return None
    holder = _ACTIVE_LOGS[-1]
    old = holder["path"]
    new = old.with_name(f"{stem}.log")
    if new == old:
        return old
    if new.exists():
        new = old.with_name(f"{stem}_{os.getpid()}.log")
    os.replace(old, new)
    holder["path"] = new
    print(f"Log renamed to: {new}", flush=True)
    return new


@contextmanager
def tee_output(script_name, log_dir=None):
    """Mirror stdout/stderr to ``log/<script>_<timestamp>_<pid>.log`` (renamed to
    ``log/<run-id>.log`` by rename_active_log once the run ID exists)."""
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent / "log"
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{script_name}_{timestamp}_{os.getpid()}.log"
    lock = threading.RLock()

    holder = {"path": log_path}
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        stdout_tee = _TeeStream(sys.stdout, log_file, lock)
        stderr_tee = _TeeStream(sys.stderr, log_file, lock)
        _ACTIVE_LOGS.append(holder)
        try:
            with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                print(f"Logging stdout/stderr to: {log_path}", flush=True)
                yield log_path
        finally:
            _ACTIVE_LOGS.pop()
