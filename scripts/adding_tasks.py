#!/usr/bin/env python
# coding: utf-8
"""
The adding problem (Hochreiter & Schmidhuber 1997; Le, Jaitly & Hinton 2015),
as a Task adapter for the mpn_local_learning training loop (see scripts/tasks.py).

Why this task here: it is THE canonical long-range credit-assignment benchmark.
The answer depends on two inputs that are far apart in time, so solving it needs
temporal credit propagated across the whole sequence — exactly the sensitivity
the local eligibility-trace rules (diagonal RFLO / direct) approximate or drop.
So the BPTT-vs-local gap should open up as the sequence gets longer, which is the
central question of this project.

Framing (so it drives the same masked-MSE / eligibility machinery as the ring
tasks and seq-MNIST — the local rules are derived for masked MSE):
  input  (B, T, 2): channel 0 = random values ~ U[0, high); channel 1 = a binary
      marker, 1 at exactly `n_marks` positions and 0 elsewhere.
  target: the SUM of the marked values (range [0, n_marks*high]), placed at the
      final step; the cost mask is 1 at t=T-1 and 0 elsewhere (only the final
      step is scored — M / eligibility still evolve over every step).
  metric: fraction of examples whose final-step prediction is within `tol` of the
      true sum (a constant predictor scores ~0; a perfect one scores 1).

Mark placement (`placement`):
  'segments' (default) — split the sequence into `n_marks` equal segments and put
      one mark in each. For n_marks=2 this is the standard "one mark in each half",
      guaranteeing a dependency that spans ~the whole sequence (the hard version).
  'uniform' — `n_marks` distinct uniformly-random positions (easier: marks can be
      adjacent).

Run standalone for a quick look:  python adding_tasks.py
"""
import numpy as np
import torch


def adding_problem_generator(batch_size, seq_len=200, n_marks=2, high=1.0,
                             placement="segments", rng=None):
    """Generate one adding-problem batch.

    Returns (X, target):
      X      : float tensor (batch_size, seq_len, 2) — [value, marker] channels.
      target : float tensor (batch_size, 1) — sum of the marked values.

    rng: an object exposing .uniform / .randint / .choice (the np.random module for
    training, or a np.random.RandomState for a reproducible held-out set). Defaults
    to np.random.
    """
    assert seq_len >= n_marks, f"seq_len {seq_len} must be >= n_marks {n_marks}"
    rng = rng if rng is not None else np.random

    vals = rng.uniform(0.0, high, size=(batch_size, seq_len, 1)).astype(np.float32)
    marks = np.zeros((batch_size, seq_len, 1), dtype=np.float32)

    if placement == "segments":
        # One mark per equal segment → marks spread across the whole sequence.
        bounds = np.linspace(0, seq_len, n_marks + 1).astype(int)
        for i in range(batch_size):
            for s in range(n_marks):
                p = rng.randint(bounds[s], bounds[s + 1])
                marks[i, p, 0] = 1.0
    elif placement == "uniform":
        # n_marks distinct uniformly-random positions.
        for i in range(batch_size):
            pos = rng.choice(seq_len, size=n_marks, replace=False)
            marks[i, pos, 0] = 1.0
    else:
        raise ValueError(f"unknown placement '{placement}'")

    X = np.concatenate([vals, marks], axis=2)          # (B, T, 2)
    target = (vals * marks).sum(axis=1)                # (B, 1) sum of marked values
    return torch.from_numpy(X), torch.from_numpy(target.astype(np.float32))


class AddingProblemTask:
    """Task adapter for the adding problem (see module docstring). Regression onto
    a single scalar (the sum), scored on the final step by masked MSE; accuracy is
    the fraction of predictions within `tol` of the true sum."""
    feat_dim = 2       # input channels: [value, marker]
    n_output = 1       # scalar sum
    # This is a regression task — accuracy (fraction within tol) is a crude summary,
    # so the training loop plots the masked-MSE LOSS instead (log-y). accuracy() is
    # still computed and logged to the console.
    metric = "loss"
    y_label = "masked-MSE loss"

    def __init__(self, seq_len=200, n_marks=2, high=1.0, tol=0.04,
                 placement="segments"):
        self.seq_len = int(seq_len)
        self.n_marks = int(n_marks)
        self.high = float(high)
        self.tol = float(tol)
        self.placement = placement
        self._valid = None   # cached fixed held-out batch (cpu tensors)

    # -- batch assembly (target/mask on the final step only) --------------------
    def _make_batch(self, X, target, device, dtype):
        xb = X.to(device=device, dtype=dtype)
        B, T, _ = xb.shape
        labels = torch.zeros(B, T, self.n_output, dtype=dtype, device=device)
        mask = torch.zeros(B, T, self.n_output, dtype=dtype, device=device)
        labels[:, T - 1, 0] = target[:, 0].to(device=device, dtype=dtype)
        mask[:, T - 1, 0] = 1.0
        return xb, labels, mask

    # -- Task hooks -------------------------------------------------------------
    def init_params(self, task_params, train_params, net_params):
        """Set the net's input/output widths (2 -> 1); no ring-task prefs."""
        if "n_neurons" in net_params:
            net_params["n_neurons"][0] = self.feat_dim
            net_params["n_neurons"][-1] = self.n_output
        else:
            net_params["n_input"] = self.feat_dim
            net_params["n_output"] = self.n_output
        net_params["prefs"] = None
        net_params.setdefault("loss_type", "MSE")
        task_params["n_input"] = self.feat_dim
        task_params["n_output"] = self.n_output
        return task_params, train_params, net_params

    def valid_batch(self, task_params, train_params, device, dtype):
        """A FIXED held-out batch (seeded RandomState), identical across every
        rule and seed so the test curve is comparable."""
        if self._valid is None:
            n = int(train_params.get("valid_n_batch", 384))
            rng = np.random.RandomState(int(task_params.get("valid_seed", 1234)))
            self._valid = adding_problem_generator(
                n, self.seq_len, self.n_marks, self.high, self.placement, rng)
        return self._make_batch(self._valid[0], self._valid[1], device, dtype)

    def train_batch(self, task_params, train_params, batch, device, dtype):
        """A fresh random batch drawn from np.random (seeded per-seed in
        run_seed, so all rules in a seed see identical data)."""
        X, target = adding_problem_generator(
            int(batch), self.seq_len, self.n_marks, self.high, self.placement)
        return self._make_batch(X, target, device, dtype)

    def accuracy(self, net, output, labels, mask, inputs, isvalid=False):
        """Fraction of examples whose final-step prediction is within tol of the
        true sum."""
        pred = output[:, -1, 0]
        true = labels[:, -1, 0]
        return float(((pred - true).abs() < self.tol).float().mean().item())


if __name__ == "__main__":
    X, y = adding_problem_generator(batch_size=4, seq_len=200, n_marks=2)
    print(f"X {tuple(X.shape)}  target {tuple(y.shape)}")
    for i in range(4):
        pos = torch.nonzero(X[i, :, 1]).flatten().tolist()
        print(f"  ex{i}: marks at {pos}  values "
              f"{[round(X[i, p, 0].item(), 3) for p in pos]}  sum={y[i, 0].item():.3f}")
