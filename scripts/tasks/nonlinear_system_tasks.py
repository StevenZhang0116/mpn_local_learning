#!/usr/bin/env python
# coding: utf-8
"""
Nonlinear system identification (a simplified Wiener-Hammerstein system), as a
Task adapter for the mpn_local_learning training loop (see scripts/tasks/tasks.py).

Each trial is a fresh white-noise waveform x_1 .. x_T ~ N(0, 1), one scalar per
step. The net sees only x_t and predicts y_t at EVERY step. The targets come from
a hidden system with a fast memory, a slow memory and one saturating
nonlinearity; all states start at zero, and the net is never shown the system's
internal states or its past outputs, so it has to hold the history itself.

    fast memory      s_t = alpha * s_{t-1} + (1 - alpha) * x_t

    'mid' (wh_mid)   y_t = beta * y_{t-1} + (1 - beta) * tanh(gain * s_t)
    'out' (wh_out)   r_t = beta * r_{t-1} + (1 - beta) * s_t,   y_t = tanh(gain * r_t)

The two variants are a paired control: same inputs, same two memories, same
supervision, and only the place of the nonlinearity moves. In 'out' the two
memories merge into one causal linear filter and the nonlinearity is applied once,
at the output. In 'mid' the slow memory accumulates a signal that has already been
transformed, so the computation is memory -> nonlinearity -> memory.

Framing (the same masked-MSE machinery as the ring tasks and the adding problem):
  input  (B, T, 1): the waveform.
  target (B, T, 1): y_t; the cost mask is 1 at every step.
  metric: the training loop plots the MSE. accuracy() is the skill relative to the
      best causal linear filter (FIR) on the same batch,
          1 - MSE_model / MSE_FIR,
      so 0 means "as good as a linear filter", 1 means exact, and a net that is
      worse than the linear filter scores below 0 (an untrained net is far below).
      A linear filter already explains about 90 % of the target variance at the
      defaults (alpha=0.6, beta=0.9, gain=6, T=64), for both variants; the skill
      measures what is learned of the remaining, nonlinear part.

The FIR baseline is fitted once, by least squares, on `fir_n` sequences drawn from
their own seed (`fir_seed`), independent of the training and the held-out data.

Run standalone for a quick look:  python nonlinear_system_tasks.py
"""
import numpy as np
import torch
import torch.nn.functional as F


VARIANTS = ("mid", "out")


def wh_system(x, variant, alpha=0.6, beta=0.9, gain=6.0):
    """Run the hidden system on waveforms x (B, T).

    Returns (y, s), both float64 (B, T): the target and the fast state. The slow
    state is y itself for 'mid' and arctanh(y) / gain for 'out'."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    x = np.asarray(x, dtype=np.float64)
    B, T = x.shape
    fast = np.zeros(B)
    slow = np.zeros(B)
    s = np.empty((B, T))
    y = np.empty((B, T))
    for t in range(T):
        fast = alpha * fast + (1.0 - alpha) * x[:, t]
        drive = np.tanh(gain * fast) if variant == "mid" else fast
        slow = beta * slow + (1.0 - beta) * drive
        s[:, t] = fast
        y[:, t] = slow if variant == "mid" else np.tanh(gain * slow)
    return y, s


# ─── The linear baseline: a causal FIR filter ─────────────────────────────────
def lagged_inputs(x, order):
    """(B, T) -> (B, T, order): entry k holds x_{t-k}, zero before the trial."""
    x = np.asarray(x, dtype=np.float64)
    B, T = x.shape
    lags = np.zeros((B, T, order))
    for k in range(min(order, T)):
        lags[:, k:, k] = x[:, :T - k]
    return lags


def fit_fir(x, y, order):
    """Least-squares causal linear filter y_t ~ sum_k w_k x_{t-k} + c over all
    trials and steps. Returns (w, c): the `order` lag weights and the intercept."""
    lags = lagged_inputs(x, order).reshape(-1, order)
    design = np.concatenate([lags, np.ones((lags.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=np.float64).reshape(-1), rcond=None)
    return coef[:order], float(coef[order])


def fir_predict(x, w, c):
    """The filter's prediction (B, T) for waveforms x (B, T) (numpy)."""
    return lagged_inputs(x, len(w)) @ w + c


class NonlinearSystemTask:
    """Task adapter for the two-timescale nonlinear system (see the module
    docstring). Regression onto one scalar per step, scored at every step by
    masked MSE; accuracy is the skill relative to the linear (FIR) baseline."""
    feat_dim = 1       # the scalar waveform
    n_output = 1       # the scalar target
    # A regression task: the training loop plots the MSE (log-y). accuracy() is
    # still computed and logged to the console.
    metric = "loss"
    y_label = "MSE loss"
    loss_and_grad = None            # default masked MSE (net's module-local fast path)

    def __init__(self, variant="mid", alpha=0.6, beta=0.9, gain=6.0, seq_len=64,
                 fir_order=64, fir_n=512, fir_seed=4321):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gain = float(gain)
        self.seq_len = int(seq_len)
        if not 0.0 <= self.alpha < 1.0 or not 0.0 <= self.beta < 1.0:
            raise ValueError("alpha and beta must be in [0, 1)")
        if not self.gain > 0.0:
            raise ValueError("gain must be > 0")
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        # A lag beyond the trial length never sees an input (the states start at 0).
        self.fir_order = min(int(fir_order), self.seq_len)
        self.fir_n = int(fir_n)
        self.fir_seed = int(fir_seed)
        self._fir = None    # (w, c), fitted on first use
        self._valid = {}    # (n, valid_seed) -> cached fixed held-out (x, y)

    # -- data ---------------------------------------------------------------
    def targets(self, x):
        """(y, s) for waveforms x (B, T): the target and the hidden fast state."""
        return wh_system(x, self.variant, self.alpha, self.beta, self.gain)

    def _sample(self, n, rng):
        """n fresh trials -> (x, y), float64 (n, T). rng is the np.random module
        (training) or a RandomState (held-out set, FIR fit)."""
        x = rng.standard_normal((int(n), self.seq_len))
        return x, self.targets(x)[0]

    def _valid_set(self, task_params, train_params):
        n = int(train_params.get("valid_n_batch", 384))
        seed = int(task_params.get("valid_seed", 1234))
        if (n, seed) not in self._valid:
            self._valid[(n, seed)] = self._sample(n, np.random.RandomState(seed))
        return self._valid[(n, seed)]

    def _make_batch(self, x, y, device, dtype):
        """inputs / labels / mask, each (B, T, 1); every step is scored."""
        inputs = torch.as_tensor(x, dtype=dtype, device=device).unsqueeze(-1)
        labels = torch.as_tensor(y, dtype=dtype, device=device).unsqueeze(-1)
        return inputs, labels, torch.ones_like(labels)

    # -- linear baseline ----------------------------------------------------
    def fir_coefficients(self):
        """(w, c) of the least-squares causal FIR filter for this system, fitted
        once on fir_n sequences from RandomState(fir_seed)."""
        if self._fir is None:
            x, y = self._sample(self.fir_n, np.random.RandomState(self.fir_seed))
            self._fir = fit_fir(x, y, self.fir_order)
        return self._fir

    def fir_predict(self, inputs):
        """The linear baseline's prediction for a batch of inputs (B, T, 1) or
        (B, T), as a tensor of the same shape, dtype and device."""
        x = inputs[..., 0] if inputs.dim() == 3 else inputs
        w, c = self.fir_coefficients()
        weights = torch.as_tensor(w, dtype=x.dtype, device=x.device)
        # windows[:, t, j] = x_{t-(order-1)+j}, i.e. lag order-1-j; zero before the trial.
        windows = F.pad(x, (len(w) - 1, 0)).unfold(1, len(w), 1)
        pred = windows @ weights.flip(0) + c
        return pred.unsqueeze(-1) if inputs.dim() == 3 else pred

    def fir_mse(self, x, y):
        """MSE of the linear baseline on waveforms x (B, T) with targets y (numpy)."""
        w, c = self.fir_coefficients()
        return float(np.mean((fir_predict(x, w, c) - y) ** 2))

    # -- Task hooks -----------------------------------------------------------
    def init_params(self, task_params, train_params, net_params):
        """Set the net's input/output widths (1 -> 1) and record the system, the
        held-out seed and the linear baseline's held-out MSE in task_params; no
        ring-task prefs."""
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
        task_params["variant"] = self.variant
        task_params["alpha"] = self.alpha
        task_params["beta"] = self.beta
        task_params["gain"] = self.gain
        task_params["seq_len"] = self.seq_len
        task_params.setdefault("valid_seed", 1234)
        task_params["fir_order"] = self.fir_order
        task_params["fir_n"] = self.fir_n
        task_params["fir_seed"] = self.fir_seed
        task_params["fir_valid_mse"] = self.fir_mse(*self._valid_set(task_params, train_params))
        return task_params, train_params, net_params

    def valid_batch(self, task_params, train_params, device, dtype):
        """A FIXED held-out batch (seeded RandomState), identical across every
        rule and seed, and across the two variants (same waveforms)."""
        return self._make_batch(*self._valid_set(task_params, train_params), device, dtype)

    def train_batch(self, task_params, train_params, batch, device, dtype):
        """A fresh random batch drawn from np.random (seeded per-seed in
        run_seed, so all rules in a seed see identical data)."""
        return self._make_batch(*self._sample(batch, np.random), device, dtype)

    def accuracy(self, net, output, labels, mask, inputs, isvalid=False):
        """Skill relative to the linear baseline on this batch:
        1 - MSE_model / MSE_FIR (0 = as good as the FIR filter, 1 = exact,
        negative = worse than the FIR filter)."""
        with torch.no_grad():
            model = ((output - labels) ** 2).mean()
            linear = ((self.fir_predict(inputs) - labels) ** 2).mean()
            return float(1.0 - model / linear)


if __name__ == "__main__":
    for variant in VARIANTS:
        task = NonlinearSystemTask(variant)
        x, y = task._sample(512, np.random.RandomState(0))
        mse = task.fir_mse(x, y)
        print(f"wh_{variant}: x {x.shape}  y in [{y.min():+.3f}, {y.max():+.3f}]  "
              f"std {y.std():.3f}   FIR({task.fir_order}) MSE {mse:.3e}  "
              f"R^2 {1.0 - mse / y.var():.3f}")
