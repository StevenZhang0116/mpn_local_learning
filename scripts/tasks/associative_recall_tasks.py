#!/usr/bin/env python
# coding: utf-8
"""
Two-hop associative recall, as a Task adapter for the mpn_local_learning training
loop (see scripts/tasks/tasks.py).

Why this task here: it separates ONE memory lookup from TWO under an identical
memory load. Every trial writes two random tables and then asks one question; the
two kinds of question share the stored history, the delay and even the correct
answer, and differ only in how many lookups the answer needs. A model that scores
well on one-hop but poorly on two-hop queries stores the tables and fails to chain
them, which in a deep MPN is the job of the layers working together.

One trial, with K symbols per set (default K=4, delay=4):

  two random permutations   f: A -> B   and   g: B -> C   (fresh every trial)

  step 0 .. K-1        the K pairs (A_a, B_f(a)), in random order
  step K .. 2K-1       the K pairs (B_b, C_g(b)), in random order
  next `delay` steps   blank (all-zero input)
  final step           the query, a single symbol

  one-hop query   input B_b            answer C_g(b)
  two-hop query   input A_a, b = f(a)  answer C_g(f(a))   (the same C symbol)

The A -> B table always comes first. With the opposite order a network could look
up C while the (A, B) pair is on screen and store it under the key A, which turns
the two-hop query into a single lookup.

Framing (same seam as seq-MNIST: final-step classification on logits):
  input  (B, T, 3K): one channel per symbol, A = 0..K-1, B = K..2K-1,
      C = 2K..3K-1. A pair step lights its two channels, the query step lights
      one, a blank step none. T = 2K + delay + 1.
  target: the C symbol, one-hot over K outputs at the final step; the cost mask is
      1 at t=T-1 and 0 elsewhere. Loss: masked softmax cross-entropy. Chance = 1/K.
  train batches: each trial is two-hop with probability `p_two_hop`, else one-hop.
  held-out set: PAIRED. Each stored history appears twice, once per query kind,
      with the same queried symbol and the same answer: the first half of the
      batch holds the one-hop trials, the second half the two-hop trials of the
      same histories. query_accuracy() scores the two kinds separately.

Run standalone for a quick look:  python associative_recall_tasks.py
"""
import numpy as np
import torch

import _bootstrap  # noqa: F401  (puts ../core on sys.path when run standalone)
from mpn import masked_cross_entropy_loss_and_grad


def two_hop_generator(n_histories, n_symbols=4, delay=4, p_two_hop=0.5, rng=None,
                      paired=False):
    """Generate one batch of two-hop recall trials.

    Returns (X, target, info):
      X      : float tensor (B, 2K + delay + 1, 3K), the symbol channels.
      target : long tensor (B,), the index of the correct C symbol.
      info   : numpy arrays describing each trial: 'two_hop' (B,) bool, the tables
               'f' and 'g' (B, K), and the queried 'a' and 'b' = f[a] (B,).

    paired=False: B = n_histories, each trial two-hop with probability p_two_hop.
    paired=True : B = 2 * n_histories. Trials [0, n) are the one-hop and trials
        [n, 2n) the two-hop version of the SAME histories and queried symbols, so
        row i and row n + i differ only in the query step and share their target.

    rng: an object exposing .random_sample / .randint (the np.random module for
    training, or a np.random.RandomState for a reproducible held-out set). Defaults
    to np.random.
    """
    K, n, delay = int(n_symbols), int(n_histories), int(delay)
    assert K >= 2, f"n_symbols must be >= 2, got {K}"
    assert delay >= 0, f"delay must be >= 0, got {delay}"
    rng = rng if rng is not None else np.random
    T = 2 * K + delay + 1
    rows = np.arange(n)

    def permutations():
        return np.argsort(rng.random_sample((n, K)), axis=1)

    f, g = permutations(), permutations()                 # f: A -> B,  g: B -> C
    order_a, order_b = permutations(), permutations()     # presentation order per table
    history = np.zeros((n, T, 3 * K), dtype=np.float32)
    for s in range(K):
        a = order_a[:, s]
        history[rows, s, a] = 1.0                         # A_a ...
        history[rows, s, K + f[rows, a]] = 1.0            # ... with B_f(a)
        b = order_b[:, s]
        history[rows, K + s, K + b] = 1.0                 # B_b ...
        history[rows, K + s, 2 * K + g[rows, b]] = 1.0    # ... with C_g(b)

    a_query = rng.randint(0, K, size=n)
    b_query = f[rows, a_query]
    target = g[rows, b_query]

    def with_query(two_hop):
        X = history.copy()
        X[rows, T - 1, np.where(two_hop, a_query, K + b_query)] = 1.0
        return X

    if paired:
        two_hop = np.repeat([False, True], n)
        X = np.concatenate([with_query(two_hop[:n]), with_query(two_hop[n:])])
        f, g, a_query, b_query, target = (np.concatenate([v, v]) for v in
                                          (f, g, a_query, b_query, target))
    else:
        two_hop = rng.random_sample(n) < p_two_hop
        X = with_query(two_hop)

    info = {"two_hop": two_hop, "f": f, "g": g, "a": a_query, "b": b_query}
    return torch.from_numpy(X), torch.from_numpy(target.astype(np.int64)), info


class TwoHopRecallTask:
    """Task adapter for two-hop associative recall (see the module docstring).
    Classification over the K symbols of C, scored on the final step with masked
    cross-entropy; accuracy is the argmax of the final-step logits."""
    metric = "accuracy"
    y_label = "classification accuracy (%)"
    loss_and_grad = staticmethod(masked_cross_entropy_loss_and_grad)

    def __init__(self, n_symbols=4, delay=4, p_two_hop=0.5):
        self.n_symbols = int(n_symbols)
        self.delay = int(delay)
        self.p_two_hop = float(p_two_hop)
        if self.n_symbols < 2:
            raise ValueError("n_symbols must be >= 2")
        if self.delay < 0:
            raise ValueError("delay must be >= 0")
        if not 0.0 <= self.p_two_hop <= 1.0:
            raise ValueError("p_two_hop must be in [0, 1]")
        self.feat_dim = 3 * self.n_symbols     # input channels: A, B and C symbols
        self.n_output = self.n_symbols         # one logit per C symbol
        self.seq_len = 2 * self.n_symbols + self.delay + 1
        self._valid = {}    # (n_histories, valid_seed) -> cached fixed held-out batch

    # -- batch assembly (one-hot target / mask on the final step only) ----------
    def _make_batch(self, X, target, device, dtype):
        xb = X.to(device=device, dtype=dtype)
        B, T, _ = xb.shape
        labels = torch.zeros(B, T, self.n_output, dtype=dtype, device=device)
        mask = torch.zeros(B, T, self.n_output, dtype=dtype, device=device)
        labels[torch.arange(B), T - 1, target.to(device)] = 1.0
        mask[:, T - 1, :] = 1.0
        return xb, labels, mask

    # -- Task hooks -------------------------------------------------------------
    def init_params(self, task_params, train_params, net_params):
        """Set the net's input/output widths (3K -> K) and record the task setup
        (K, delay, query ratio) in task_params; no ring-task prefs."""
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
        task_params["n_symbols"] = self.n_symbols
        task_params["delay"] = self.delay
        task_params["p_two_hop"] = self.p_two_hop
        task_params["seq_len"] = self.seq_len
        return task_params, train_params, net_params

    def valid_batch(self, task_params, train_params, device, dtype):
        """A FIXED, PAIRED held-out batch (seeded RandomState): valid_n_batch // 2
        histories, each asked once per query kind. Identical across every rule and
        seed; the first half is one-hop, the second half two-hop."""
        n = max(int(train_params.get("valid_n_batch", 384)) // 2, 1)
        seed = int(task_params.get("valid_seed", 1234))
        if (n, seed) not in self._valid:
            X, target, _ = two_hop_generator(
                n, self.n_symbols, self.delay, rng=np.random.RandomState(seed), paired=True)
            self._valid[(n, seed)] = (X, target)
        return self._make_batch(*self._valid[(n, seed)], device, dtype)

    def train_batch(self, task_params, train_params, batch, device, dtype):
        """A fresh random batch drawn from np.random (seeded per-seed in run_seed,
        so all rules in a seed see identical data); each trial is two-hop with
        probability p_two_hop."""
        X, target, _ = two_hop_generator(
            int(batch), self.n_symbols, self.delay, self.p_two_hop)
        return self._make_batch(X, target, device, dtype)

    def accuracy(self, net, output, labels, mask, inputs, isvalid=False):
        """Fraction correct over all trials: argmax of the final-step logits."""
        pred = output[:, -1, :].argmax(dim=-1)
        true = labels[:, -1, :].argmax(dim=-1)
        return float((pred == true).float().mean().item())

    def query_accuracy(self, output, labels, inputs):
        """Accuracy split by query kind: {'one_hop': .., 'two_hop': ..}. The kind is
        read off the final input step (an A symbol asks a two-hop question, a B
        symbol a one-hop one); a kind with no trial in the batch gives nan."""
        correct = output[:, -1, :].argmax(dim=-1) == labels[:, -1, :].argmax(dim=-1)
        two_hop = inputs[:, -1, :self.n_symbols].sum(dim=-1) > 0

        def fraction(select):
            return float(correct[select].float().mean().item()) if bool(select.any()) else float("nan")

        return {"one_hop": fraction(~two_hop), "two_hop": fraction(two_hop)}


if __name__ == "__main__":
    K, delay = 4, 4
    X, target, info = two_hop_generator(2, K, delay, paired=True)
    print(f"X {tuple(X.shape)}  target {tuple(target.shape)}")
    names = [f"{s}{i}" for s in "ABC" for i in range(K)]
    for i in range(X.shape[0]):
        kind = "two-hop" if info["two_hop"][i] else "one-hop"
        steps = ["+".join(names[j] for j in torch.nonzero(X[i, t]).flatten().tolist()) or "."
                 for t in range(X.shape[1])]
        print(f"  trial {i} ({kind}): {' '.join(steps)}  ->  C{int(target[i])}")
