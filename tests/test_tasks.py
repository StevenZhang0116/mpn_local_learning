#!/usr/bin/env python
# coding: utf-8
"""
Tests for the task adapters (scripts/tasks.py) — the data/metric seam that lets
train_common train either the ring tasks or sequential MNIST.

Checks, for seq-MNIST (both 'row' and 'pixel' modes):
  1. raw idx files load with the right shapes / value ranges (no torchvision).
  2. init_params sets the net's in/out widths correctly.
  3. batches have shape (B, T, feat), one-hot labels + cost mask on the FINAL
     step only, and the valid set is a FIXED slice (identical across calls).
  4. train_batch is deterministic under a fixed np.random seed (so all rules in a
     seed see identical data — the lockstep invariant).
  5. the accuracy metric is argmax-correct (perfect=1, wrong=0).
  6. registry: make_task / acc_label_for map ruleset strings correctly.
And an integration check:
  7. a few real training steps on seq-MNIST reduce the loss for BOTH the RNN and
     the deep MPN, under BPTT and a local rule (pipeline works end to end).

Run from this directory (mpn_local_learning/tests):  python test_tasks.py
"""
import os

import numpy as np
import torch

import _bootstrap  # prepends ../core + ../scripts to sys.path
import tasks
import mpn
import rnn


PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    ok = bool(cond)
    PASS += ok
    FAIL += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('   ' + extra) if extra else ''}")


def _mnist_available(t):
    root = t._root({})
    return os.path.isfile(os.path.join(root, "train-images-idx3-ubyte"))


# ─── 1-6: SeqMNISTTask unit checks ───────────────────────────────────────────
def test_seqmnist_task(mode, T_expected, feat_expected):
    print(f"── SeqMNISTTask(mode='{mode}') ──")
    t = tasks.SeqMNISTTask(mode=mode)
    if not _mnist_available(t):
        print(f"  [SKIP] MNIST idx files not found at {t._root({})}")
        return

    check("seq_len / feat_dim", t.seq_len == T_expected and t.feat_dim == feat_expected,
          f"T={t.seq_len} feat={t.feat_dim}")

    # raw load
    pairs = t._load(t._root({}))
    Xtr, ytr = pairs["train"]
    check("train set shape", Xtr.shape == (60000, 28, 28) and ytr.shape == (60000,),
          f"X={Xtr.shape} y={ytr.shape}")
    check("labels in 0..9", ytr.min() == 0 and ytr.max() == 9)

    # init_params sets net widths
    tp, trp, npar = {"dt": 40}, {"valid_n_batch": 50}, {"n_neurons": [1, 20, 1]}
    tp, trp, npar = t.init_params(tp, trp, npar)
    check("init_params net widths", npar["n_neurons"] == [feat_expected, 20, 10],
          f"n_neurons={npar['n_neurons']}")
    check("init_params clears prefs", npar["prefs"] is None)

    # train batch shape + masking + one-hot
    xb, yb, mb = t.train_batch(tp, trp, 8, "cpu", torch.float32)
    check("train batch shape", tuple(xb.shape) == (8, T_expected, feat_expected)
          and tuple(yb.shape) == (8, T_expected, 10),
          f"x={tuple(xb.shape)} y={tuple(yb.shape)}")
    check("cost mask on final step only",
          float(mb[:, :-1].sum()) == 0.0 and float(mb[:, -1, :].sum()) == 8 * 10)
    check("labels one-hot at final step",
          torch.allclose(yb[:, -1, :].sum(-1), torch.ones(8)) and float(yb[:, :-1].sum()) == 0.0)

    # valid set is a FIXED slice (identical across calls)
    v1 = t.valid_batch(tp, trp, "cpu", torch.float32)
    v2 = t.valid_batch(tp, trp, "cpu", torch.float32)
    check("valid batch fixed / reproducible",
          torch.equal(v1[0], v2[0]) and torch.equal(v1[1], v2[1]),
          f"valid shape {tuple(v1[0].shape)}")

    # train_batch deterministic under a fixed np seed (lockstep invariant)
    np.random.seed(123); a = t.train_batch(tp, trp, 8, "cpu", torch.float32)
    np.random.seed(123); b = t.train_batch(tp, trp, 8, "cpu", torch.float32)
    check("train batch deterministic per np-seed", torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))

    # accuracy metric: perfect=1, wrong=0
    B = yb.shape[0]
    perfect = yb.clone()
    wrong = torch.zeros_like(yb)
    wrong_idx = (yb[:, -1, :].argmax(-1) + 1) % 10          # shift every label by 1
    wrong[torch.arange(B), -1, wrong_idx] = 1.0             # per-row one-hot
    check("accuracy(perfect)=1", t.accuracy(None, perfect, yb, mb, xb) == 1.0)
    check("accuracy(wrong)=0", t.accuracy(None, wrong, yb, mb, xb) == 0.0)


def test_adding_task():
    print("── AddingProblemTask ──")
    from adding_tasks import AddingProblemTask, adding_problem_generator

    t = AddingProblemTask(seq_len=200, n_marks=2)
    check("seq_len / feat_dim / n_output",
          t.seq_len == 200 and t.feat_dim == 2 and t.n_output == 1)

    # init_params sets 2 -> 1 widths
    tp, trp, npar = {"dt": 40}, {"valid_n_batch": 32}, {"n_neurons": [1, 20, 1]}
    tp, trp, npar = t.init_params(tp, trp, npar)
    check("init_params net widths", npar["n_neurons"] == [2, 20, 1],
          f"n_neurons={npar['n_neurons']}")
    check("init_params clears prefs", npar["prefs"] is None)

    # batch shape + final-step masking; input channel 1 is a 0/1 marker
    xb, yb, mb = t.train_batch(tp, trp, 8, "cpu", torch.float32)
    check("train batch shape", tuple(xb.shape) == (8, 200, 2) and tuple(yb.shape) == (8, 200, 1),
          f"x={tuple(xb.shape)} y={tuple(yb.shape)}")
    check("cost mask on final step only",
          float(mb[:, :-1].sum()) == 0.0 and float(mb[:, -1, 0].sum()) == 8)
    marker = xb[:, :, 1]
    check("marker is 0/1 with exactly n_marks ones per row",
          bool(((marker == 0) | (marker == 1)).all()) and bool((marker.sum(1) == 2).all()))

    # target at the final step equals the sum of the marked values
    val = xb[:, :, 0]
    true_sum = (val * marker).sum(1)
    check("target = sum of marked values",
          torch.allclose(yb[:, -1, 0], true_sum, atol=1e-5))

    # 'segments' placement: for n_marks=2 exactly one mark in each half
    first_half = marker[:, :100].sum(1)
    second_half = marker[:, 100:].sum(1)
    check("segments placement: one mark per half",
          bool((first_half == 1).all()) and bool((second_half == 1).all()))

    # valid set is FIXED / reproducible across calls
    v1 = t.valid_batch(tp, trp, "cpu", torch.float32)
    v2 = t.valid_batch(tp, trp, "cpu", torch.float32)
    check("valid batch fixed / reproducible", torch.equal(v1[0], v2[0]) and torch.equal(v1[1], v2[1]))

    # train_batch deterministic under a fixed np seed (lockstep invariant)
    np.random.seed(7); a = t.train_batch(tp, trp, 8, "cpu", torch.float32)
    np.random.seed(7); b = t.train_batch(tp, trp, 8, "cpu", torch.float32)
    check("train batch deterministic per np-seed", torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))

    # accuracy: within-tol
    good = yb.clone()                        # exact prediction -> all within tol
    bad = yb.clone(); bad[:, -1, 0] = bad[:, -1, 0] + 10 * t.tol   # far off -> none within tol
    check("accuracy(exact)=1", t.accuracy(None, good, yb, mb, xb) == 1.0)
    check("accuracy(far off)=0", t.accuracy(None, bad, yb, mb, xb) == 0.0)


def test_registry():
    print("── registry (make_task / acc_label_for) ──")
    check("seqmnist -> SeqMNISTTask row",
          isinstance(tasks.make_task("seqmnist"), tasks.SeqMNISTTask)
          and tasks.make_task("seqmnist").mode == "row")
    check("seqmnist_pixel -> SeqMNISTTask pixel",
          tasks.make_task("seqmnist_pixel").mode == "pixel")
    check("adding -> AddingProblemTask (200, 2)",
          isinstance(tasks.make_task("adding"), tasks.AddingProblemTask)
          and tasks.make_task("adding").seq_len == 200
          and tasks.make_task("adding").n_marks == 2)
    check("adding_L500_m3 parses len/marks",
          tasks.make_task("adding_L500_m3").seq_len == 500
          and tasks.make_task("adding_L500_m3").n_marks == 3)
    check("ring task -> MultiTaskAdapter",
          isinstance(tasks.make_task("delaygo"), tasks.MultiTaskAdapter))
    check("acc_label seqmnist", tasks.acc_label_for("seqmnist") == "classification accuracy (%)")
    check("acc_label adding", tasks.acc_label_for("adding") == "masked-MSE loss")
    check("acc_label ring", tasks.acc_label_for("contextdelaydm1") == "angle accuracy (%)")
    # plotted metric per task: adding plots loss, the rest plot accuracy
    check("metric adding = loss", tasks.metric_for("adding") == "loss")
    check("metric seqmnist = accuracy", tasks.metric_for("seqmnist") == "accuracy")
    check("metric ring = accuracy", tasks.metric_for("delaygo") == "accuracy")


# ─── 7: integration — a few steps reduce the loss for both nets ──────────────
def _net_params(net_type, n_in, n_hidden, n_out):
    npar = {
        "n_neurons": [n_in, n_hidden, n_out], "loss_type": "MSE", "activation": "tanh",
        "output_bias": False, "output_matrix": "", "dt": 40,
        "learning_rule": "bptt", "feedback_mode": "exact_readout",
        "ml_params": {"bias": True, "mp_type": "mult", "m_update_type": "hebb_assoc",
                      "m_activation": "linear", "modulation_bounds": False,
                      "eta_type": "scalar", "eta_train": False, "lam_type": "scalar",
                      "lam_train": False, "m_time_scale": 4000, "W_freeze": False},
    }
    if net_type == "dmpn":
        npar.update({"linear_embed": n_hidden, "input_layer_add": True,
                     "input_layer_add_trainable": True, "input_layer_bias": True,
                     "input_init_type": "xavier"})
    if net_type == "vanilla":
        npar.update({"hidden_bias": True, "leaky": True, "alpha": 0.8})
    return npar


def _train_a_bit(net, t, tp, trp, steps=40, lr=5e-3, batch=32):
    """Run `steps` optimizer steps; return (loss_first, loss_last)."""
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    first = last = None
    np.random.seed(0)
    for s in range(steps):
        xb, yb, mb = t.train_batch(tp, trp, batch, "cpu", torch.float32)
        opt.zero_grad()
        g = net.sequence_gradients(xb, yb, mb)
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 10)
        opt.step()
        if net.param_clamping:
            net.param_clamp()
        if s == 0:
            first = float(g["loss"])
        last = float(g["loss"])
    return first, last


def _integration_for(task_name, t, n_out, steps=40):
    """A few optimizer steps on task `t` should reduce the loss for both nets and
    both rules (pipeline works end to end)."""
    tp, trp, _ = t.init_params({"dt": 40}, {"valid_n_batch": 100}, {"n_neurons": [1, 32, 1]})
    for net_type, cls, label in (("dmpn", mpn.DeepMultiPlasticNet, "MPN"),
                                 ("vanilla", rnn.LeakyRNN, "RNN")):
        for rule in ("bptt", "local_diag_rflo"):
            torch.manual_seed(0); np.random.seed(0)
            npar = _net_params(net_type, t.feat_dim, 32, n_out)
            npar["learning_rule"] = rule
            net = cls(npar, verbose=False).to(torch.float32)
            net.learning_rule = rule
            f, l = _train_a_bit(net, t, tp, trp, steps=steps)
            check(f"{task_name} {label} {rule}: loss decreases", l < f, f"{f:.3e} -> {l:.3e}")


def test_integration_seqmnist():
    print("── integration: loss decreases on seq-MNIST (both nets, both rules) ──")
    t = tasks.SeqMNISTTask(mode="row")
    if not _mnist_available(t):
        print(f"  [SKIP] MNIST idx files not found at {t._root({})}")
        return
    _integration_for("seqmnist", t, t.n_classes, steps=40)


def test_integration_adding():
    print("── integration: loss decreases on adding (short seq, both nets, both rules) ──")
    # Short sequence so the test stays fast; the framing is identical to seq_len=200.
    t = tasks.make_task("adding_L30_m2")
    _integration_for("adding", t, t.n_output, steps=40)


if __name__ == "__main__":
    print("=" * 68)
    test_seqmnist_task("row", 28, 28)
    test_seqmnist_task("pixel", 784, 1)
    test_adding_task()
    test_registry()
    test_integration_seqmnist()
    test_integration_adding()
    print("=" * 68)
    print(f"{'ALL TASK TESTS PASSED' if FAIL == 0 else f'{FAIL} CHECK(S) FAILED'}  "
          f"({PASS} passed, {FAIL} failed)")
    raise SystemExit(1 if FAIL else 0)
