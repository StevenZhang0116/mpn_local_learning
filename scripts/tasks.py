#!/usr/bin/env python
# coding: utf-8
"""
Task adapters: the data/metric seam between the shared training loop
(train_common.run_seed) and a concrete task.

train_common used to call mpn_tasks directly (convert_and_init_multitask_params,
get_prefs, generate_trials_wrap, net.compute_acc). Those calls are now routed
through a `Task` object on the RunConfig, so the SAME lockstep BPTT-vs-local loop
can train on either the multitask ring tasks or a plain supervised dataset like
sequential MNIST. The networks and the loop don't change — only the task does.

A Task provides four hooks (everything run_seed needs to touch the data/metric):

  init_params(task_params, train_params, net_params) -> the same triple, adjusted
      so a net can be built (sets the input/output widths of net_params, etc.).
  valid_batch(task_params, train_params, device, dtype) -> (inputs, labels, mask)
      the held-out set, generated ONCE per seed and shared across rules.
  train_batch(task_params, train_params, batch, device, dtype) -> (inputs, labels,
      mask) one training batch (generated once per step, fed to every rule).
  accuracy(net, output, labels, mask, inputs, isvalid) -> float scalar accuracy.

All tensors are (B, T, ·). The mask is a float 'cost' mask (B, T, n_out) — the
loss/eligibility machinery already multiplies by it, so a task that only scores
its final step just sets the mask to 1 there and 0 elsewhere.

A Task also declares its OBJECTIVE via a `loss_and_grad` attribute (the
(loss, dL/d output) contract of mpn.masked_mse_loss_and_output_grad):
    None  -> the net's default masked MSE (ring / adding). Passing None keeps the
             net on its module-local fast path (the local rules' single-pass
             masked-MSE branch), so the default tasks are byte-for-byte unchanged.
    fn    -> a task-specific loss (seq-MNIST uses masked cross-entropy on logits).
             Any non-default loss routes the local rules through their two-pass
             path (forward once for outputs, then loss_and_grad for grad_output).
The objective is per-task by design; different tasks need not share one loss.

`make_task(ruleset)` maps the RULESET string to the right Task:
    'seqmnist'         -> SeqMNISTTask(mode='row')    28 steps of 28 pixels
    'seqmnist_pixel'   -> SeqMNISTTask(mode='pixel')  784 steps of 1 pixel
    anything else      -> MultiTaskAdapter()          the ring tasks (unchanged)
"""
import os
import re
import struct

import numpy as np
import torch

import mpn_tasks
from adding_tasks import AddingProblemTask
from mpn import masked_cross_entropy_loss_and_grad


# ─── Multitask ring tasks (the original behaviour, unchanged) ─────────────────
class MultiTaskAdapter:
    """Wraps the mpn_tasks pipeline exactly as train_common used to call it, so
    the default (ring-task) behaviour is byte-for-byte preserved."""

    # The training loop plots this metric (see train_common). 'accuracy' → the
    # accuracy() curve, percent, 0-110 y-range; 'loss' → the masked-MSE loss,
    # log-y (used by regression tasks where accuracy is uninformative).
    metric = "accuracy"
    y_label = "angle accuracy (%)"
    loss_and_grad = None            # default masked MSE (net's module-local fast path)

    def init_params(self, task_params, train_params, net_params):
        task_params, train_params, net_params = mpn_tasks.convert_and_init_multitask_params(
            (task_params, train_params, net_params))
        net_params["prefs"] = mpn_tasks.get_prefs(task_params["hp"])
        return task_params, train_params, net_params

    def valid_batch(self, task_params, train_params, device, dtype):
        vdata, _ = mpn_tasks.generate_trials_wrap(
            task_params, train_params["valid_n_batch"], rules=task_params["rules"],
            mode_input="random_batch", device=device)
        return tuple(d.to(dtype) for d in vdata)

    def train_batch(self, task_params, train_params, batch, device, dtype):
        data, _ = mpn_tasks.generate_trials_wrap(
            task_params, batch, rules=task_params["rules"],
            mode_input="random_batch", device=device)
        return tuple(d.to(dtype) for d in data)

    def accuracy(self, net, output, labels, mask, inputs, isvalid=False):
        """Library angle accuracy; nan on failure (same as the old try_accuracy)."""
        try:
            acc, _ = net.compute_acc(output.float(), labels.float(), mask.float(),
                                     inputs.float(), mode=net.acc_measure, isvalid=isvalid)
            return float(acc)
        except Exception:
            return float("nan")


# ─── Sequential MNIST (read raw idx files; no torchvision needed) ─────────────
# Default location of the extracted MNIST idx files (raw ubyte, already
# un-gzipped) in this workspace. Override with task_params['mnist_root'].
_DEFAULT_MNIST_ROOT = ("/allen/programs/mindscope/workgroups/auto-model/"
                       "zihan.zhang/MultiTaskMPN/data/MNIST/raw")
_MNIST_MEAN, _MNIST_STD = 0.1307, 0.3081


def _read_idx_images(path):
    """Read an idx3-ubyte image file -> float array (n, rows, cols) in [0, 1]."""
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad image magic {magic} in {path}"
        buf = f.read(n * rows * cols)
    return (np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255.0
            ).reshape(n, rows, cols)


def _read_idx_labels(path):
    """Read an idx1-ubyte label file -> int array (n,)."""
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad label magic {magic} in {path}"
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


class SeqMNISTTask:
    """Sequential MNIST as a classification task, trained with masked softmax
    CROSS-ENTROPY on the final step (loss_and_grad below). Cross-entropy is the
    standard classification objective; the local rules consume any loss through
    their (loss, dL/d output) seam, so — with the two-pass fix — CE fits them
    exactly (it only enters via grad_output). The final-step readout is treated
    as LOGITS (no softmax in the net forward; argmax accuracy is unchanged).

    Each 28x28 image is presented as a sequence:
      mode='row'   -> T=28 steps, 28 features/step (row by row).  [fast default]
      mode='pixel' -> T=784 steps, 1 feature/step (pixel by pixel; the hard,
                      long-memory benchmark — heavy for BPTT over 784 steps).

    The network reads the whole sequence; only the FINAL step is scored: the
    cost mask is 1 at t=T-1 (all 10 outputs) and 0 elsewhere, and the one-hot
    label sits at t=T-1. M/eligibility still evolve over every step; only the
    loss is applied at the end. Accuracy = argmax of the last-step output.
    """
    n_classes = 10
    metric = "accuracy"
    y_label = "classification accuracy (%)"
    loss_and_grad = staticmethod(masked_cross_entropy_loss_and_grad)

    def __init__(self, mode="row"):
        assert mode in ("row", "pixel"), f"unknown seq-MNIST mode '{mode}'"
        self.mode = mode
        self.seq_len = 28 if mode == "row" else 784
        self.feat_dim = 28 if mode == "row" else 1
        self._cache = {}   # root -> {'train': (X, y), 'test': (X, y)}

    # -- data loading / caching -------------------------------------------------
    def _load(self, root):
        if root in self._cache:
            return self._cache[root]
        pairs = {}
        for split, imgf, lblf in (
                ("train", "train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
                ("test", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")):
            X = _read_idx_images(os.path.join(root, imgf))          # (n, 28, 28)
            y = _read_idx_labels(os.path.join(root, lblf))          # (n,)
            X = (X - _MNIST_MEAN) / _MNIST_STD
            pairs[split] = (X, y)
        self._cache[root] = pairs
        return pairs

    def _root(self, task_params):
        return task_params.get("mnist_root", _DEFAULT_MNIST_ROOT)

    def _to_sequence(self, X):
        """(B, 28, 28) images -> (B, T, feat_dim) sequence for the chosen mode."""
        B = X.shape[0]
        if self.mode == "row":
            return X.reshape(B, 28, 28)               # T=28 rows, 28 features each
        return X.reshape(B, 784, 1)                    # T=784 pixels, 1 feature

    def _make_batch(self, X, y, idx, device, dtype):
        """Assemble one (inputs, labels, mask) batch for the rows selected by idx."""
        xb = torch.as_tensor(self._to_sequence(X[idx]), dtype=dtype, device=device)
        B, T, _ = xb.shape
        yb = y[idx]
        labels = torch.zeros(B, T, self.n_classes, dtype=dtype, device=device)
        mask = torch.zeros(B, T, self.n_classes, dtype=dtype, device=device)
        # One-hot target and cost mask on the final step only.
        labels[torch.arange(B), T - 1, torch.as_tensor(yb, device=device)] = 1.0
        mask[:, T - 1, :] = 1.0
        return xb, labels, mask

    # -- Task hooks -------------------------------------------------------------
    def init_params(self, task_params, train_params, net_params):
        """Set the net's input/output widths for seq-MNIST; no ring-task prefs."""
        in_dim, out_dim = self.feat_dim, self.n_classes
        if "n_neurons" in net_params:
            net_params["n_neurons"][0] = in_dim
            net_params["n_neurons"][-1] = out_dim
        else:
            net_params["n_input"] = in_dim
            net_params["n_output"] = out_dim
        net_params["prefs"] = None                     # no angle-readout preferences
        net_params.setdefault("loss_type", "MSE")
        # Record sizes where the multitask converter would have (for provenance).
        task_params["n_input"] = in_dim
        task_params["n_output"] = out_dim
        # Preload so the first batch call is fast and fails early if data is absent.
        self._load(self._root(task_params))
        return task_params, train_params, net_params

    def valid_batch(self, task_params, train_params, device, dtype):
        """A fixed held-out slice of the TEST set (first valid_n_batch images),
        so every rule and every seed is scored on the same examples."""
        X, y = self._load(self._root(task_params))["test"]
        n = min(int(train_params.get("valid_n_batch", 384)), X.shape[0])
        idx = np.arange(n)
        return self._make_batch(X, y, idx, device, dtype)

    def train_batch(self, task_params, train_params, batch, device, dtype):
        """A uniformly-random batch of TRAIN images (np.random, so it is seeded
        by run_seed's per-seed np.random.seed and identical across rules)."""
        X, y = self._load(self._root(task_params))["train"]
        idx = np.random.randint(0, X.shape[0], size=int(batch))
        return self._make_batch(X, y, idx, device, dtype)

    def accuracy(self, net, output, labels, mask, inputs, isvalid=False):
        """Fraction correct: argmax of the final-step output vs the one-hot label."""
        pred = output[:, -1, :].argmax(dim=-1)
        true = labels[:, -1, :].argmax(dim=-1)
        return float((pred == true).float().mean().item())


# ─── Task registry ────────────────────────────────────────────────────────────
def is_seqmnist(ruleset):
    """True if the RULESET string names a seq-MNIST variant."""
    return str(ruleset).lower() in (
        "seqmnist", "smnist", "mnist", "seqmnist_pixel", "pmnist", "pixelmnist")


def is_adding(ruleset):
    """True if the RULESET string names an adding-problem variant."""
    return str(ruleset).lower().split("_")[0] in ("adding", "add")


def _adding_task(ruleset):
    """Build an AddingProblemTask from the RULESET string. Bare 'adding' uses the
    defaults (seq_len=200, n_marks=2); a parametric form lets you sweep them
    without editing code, e.g. 'adding_L500_m3' -> seq_len=500, n_marks=3
    (the 'L<len>' and 'm<marks>' fragments are both optional, any order)."""
    s = str(ruleset).lower()
    m_len = re.search(r"l(\d+)", s)
    m_marks = re.search(r"m(\d+)", s)
    seq_len = int(m_len.group(1)) if m_len else 200
    n_marks = int(m_marks.group(1)) if m_marks else 2
    return AddingProblemTask(seq_len=seq_len, n_marks=n_marks)


def make_task(ruleset):
    """Map a RULESET string to a Task:
      'seqmnist' / 'seqmnist_pixel' -> SeqMNISTTask (row / pixel)
      'adding' (or 'adding_L<len>_m<marks>') -> AddingProblemTask
      anything else                 -> multitask ring-task pipeline (unchanged)."""
    key = str(ruleset).lower()
    if key in ("seqmnist", "smnist", "mnist"):
        return SeqMNISTTask(mode="row")
    if key in ("seqmnist_pixel", "pmnist", "pixelmnist"):
        return SeqMNISTTask(mode="pixel")
    if is_adding(key):
        return _adding_task(key)
    return MultiTaskAdapter()


def acc_label_for(ruleset):
    """y-axis label for the plotted panels, taken from the task the ruleset maps
    to (accuracy label for accuracy-metric tasks, loss label for loss-metric)."""
    return make_task(ruleset).y_label


def metric_for(ruleset):
    """Which curve the training loop plots for this task: 'accuracy' or 'loss'."""
    return make_task(ruleset).metric
