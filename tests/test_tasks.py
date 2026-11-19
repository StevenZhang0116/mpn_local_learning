#!/usr/bin/env python
# coding: utf-8
"""
Tests for the task adapters (scripts/tasks/tasks.py) — the data/metric seam that lets
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
The adding problem and two-hop associative recall get the same kind of checks
(for two-hop: the tables decoded from the inputs reproduce every target, the
held-out set is paired and reproducible, and one real update runs for BPTT and a
local rule on a two-MP-layer net, the RNN and the GRU).

Run from this directory (mpn_local_learning/tests):  python test_tasks.py
"""
import os

import math

import numpy as np
import torch
import torch.nn.functional as F

import _bootstrap  # puts core/ and scripts/{training,tasks}/ on sys.path
import tasks
import mpn
import rnn
import gru


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
def check_seqmnist_task(mode, T_expected, feat_expected):
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
    from adding_tasks import AddingProblemTask

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


def _decode_tables(xb, K):
    """Read both tables back from the INPUTS alone: f[b, a] = B symbol paired with
    A_a in the first K steps, g[b, k] = C symbol paired with B_k in the next K."""
    A, Bs, C = xb[..., :K], xb[..., K:2 * K], xb[..., 2 * K:]
    f = torch.einsum('bsa,bsk->bak', A[:, :K], Bs[:, :K]).argmax(-1)
    g = torch.einsum('bsk,bsc->bkc', Bs[:, K:2 * K], C[:, K:2 * K]).argmax(-1)
    return f, g


def test_twohop_task():
    print("── TwoHopRecallTask ──")
    from associative_recall_tasks import TwoHopRecallTask, two_hop_generator

    K, delay = 4, 4
    t = TwoHopRecallTask(n_symbols=K, delay=delay)
    check("seq_len / feat_dim / n_output",
          t.seq_len == 2 * K + delay + 1 and t.feat_dim == 3 * K and t.n_output == K)

    tp, trp, npar = {"dt": 40}, {"valid_n_batch": 64}, {"n_neurons": [1, 20, 1]}
    tp, trp, npar = t.init_params(tp, trp, npar)
    check("init_params net widths", npar["n_neurons"] == [3 * K, 20, K],
          f"n_neurons={npar['n_neurons']}")
    check("init_params clears prefs", npar["prefs"] is None)
    check("task setup recorded in task_params",
          (tp["n_symbols"], tp["delay"], tp["p_two_hop"], tp["seq_len"]) == (K, delay, 0.5, 13))

    np.random.seed(3)
    B = 256
    xb, yb, mb = t.train_batch(tp, trp, B, "cpu", torch.float32)
    T = t.seq_len
    check("train batch shape", tuple(xb.shape) == (B, T, 3 * K) and tuple(yb.shape) == (B, T, K),
          f"x={tuple(xb.shape)} y={tuple(yb.shape)}")
    check("cost mask on final step only (all K outputs)",
          float(mb[:, :-1].sum()) == 0.0 and float(mb[:, -1].sum()) == B * K)
    check("one-hot label on final step only",
          float(yb[:, :-1].sum()) == 0.0 and bool((yb[:, -1].sum(-1) == 1).all()))

    A, Bs, C = xb[..., :K], xb[..., K:2 * K], xb[..., 2 * K:]
    check("steps 0..K-1 show one A with one B",
          bool((A[:, :K].sum(-1) == 1).all()) and bool((Bs[:, :K].sum(-1) == 1).all())
          and float(C[:, :K].sum()) == 0.0)
    check("steps K..2K-1 show one B with one C",
          bool((Bs[:, K:2 * K].sum(-1) == 1).all()) and bool((C[:, K:2 * K].sum(-1) == 1).all())
          and float(A[:, K:2 * K].sum()) == 0.0)
    check("every symbol appears exactly once per table",
          all(bool((part.sum(1) == 1).all())
              for part in (A[:, :K], Bs[:, :K], Bs[:, K:2 * K], C[:, K:2 * K])))
    check("delay steps are blank", float(xb[:, 2 * K:2 * K + delay].sum()) == 0.0)
    check("query step lights exactly one A or B symbol",
          bool((xb[:, -1].sum(-1) == 1).all()) and float(C[:, -1].sum()) == 0.0)

    # the target is the composition of the tables as they appear in the inputs
    f, g = _decode_tables(xb, K)
    rows = torch.arange(B)
    two_hop = A[:, -1].sum(-1) > 0
    a_q, b_q = A[:, -1].argmax(-1), Bs[:, -1].argmax(-1)
    expected = torch.where(two_hop, g[rows, f[rows, a_q]], g[rows, b_q])
    target = yb[:, -1].argmax(-1)
    check("target = g(f(a)) for two-hop, g(b) for one-hop", torch.equal(target, expected))
    frac = float(two_hop.float().mean())
    check("both query kinds drawn at about p_two_hop=0.5", 0.35 < frac < 0.65, f"two-hop fraction {frac:.2f}")
    check("tables are redrawn per trial",
          len({tuple(r.tolist()) for r in f}) > 1 and len({tuple(r.tolist()) for r in g}) > 1)
    check("all K answers occur", len(set(target.tolist())) == K)

    for p, want in ((0.0, False), (1.0, True)):
        xe, _, _ = TwoHopRecallTask(K, delay, p_two_hop=p).train_batch(tp, trp, 32, "cpu", torch.float32)
        check(f"p_two_hop={p:g} gives only {'two' if want else 'one'}-hop queries",
              bool(((xe[:, -1, :K].sum(-1) > 0) == want).all()))

    # the generator's own bookkeeping agrees with what the inputs show
    X, tgt, info = two_hop_generator(32, K, delay, rng=np.random.RandomState(0))
    f_i, g_i = _decode_tables(X, K)
    r = np.arange(32)
    check("generator info matches the inputs",
          np.array_equal(info["f"], f_i.numpy()) and np.array_equal(info["g"], g_i.numpy())
          and np.array_equal(info["b"], info["f"][r, info["a"]])
          and np.array_equal(tgt.numpy(), info["g"][r, info["b"]])
          and np.array_equal(info["two_hop"], (X[:, -1, :K].sum(-1) > 0).numpy()))

    # held-out set: paired, fixed, and independent of the global numpy stream
    state = np.random.get_state()[1].copy()
    xv, yv, mv = t.valid_batch(tp, trp, "cpu", torch.float32)
    check("valid batch does not consume np.random", np.array_equal(state, np.random.get_state()[1]))
    n = xv.shape[0] // 2
    check("valid batch size = valid_n_batch", xv.shape[0] == 64)
    check("paired: both halves share the stored history", torch.equal(xv[:n, :-1], xv[n:, :-1]))
    check("paired: both halves share the answer", torch.equal(yv[:n], yv[n:]))
    check("paired: first half one-hop, second half two-hop",
          bool((xv[:n, -1, K:2 * K].sum(-1) == 1).all()) and bool((xv[n:, -1, :K].sum(-1) == 1).all()))
    f_v, _ = _decode_tables(xv[:n], K)
    check("paired: the one-hop query is f(a) of the two-hop query",
          torch.equal(f_v[torch.arange(n), xv[n:, -1, :K].argmax(-1)], xv[:n, -1, K:2 * K].argmax(-1)))
    v2 = t.valid_batch(tp, trp, "cpu", torch.float32)
    np.random.seed(99)
    v3 = TwoHopRecallTask(K, delay).valid_batch(tp, trp, "cpu", torch.float32)
    check("valid batch fixed across calls and task instances",
          torch.equal(xv, v2[0]) and torch.equal(yv, v2[1])
          and torch.equal(xv, v3[0]) and torch.equal(yv, v3[1]))

    # train_batch deterministic under a fixed np seed (lockstep invariant)
    np.random.seed(7); a = t.train_batch(tp, trp, 8, "cpu", torch.float32)
    np.random.seed(7); b = t.train_batch(tp, trp, 8, "cpu", torch.float32)
    check("train batch deterministic per np-seed", torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))

    # accuracy: overall and split by query kind
    wrong = torch.roll(yv, shifts=1, dims=-1)          # every prediction off by one class
    check("accuracy(perfect)=1", t.accuracy(None, yv, yv, mv, xv) == 1.0)
    check("accuracy(all wrong)=0", t.accuracy(None, wrong, yv, mv, xv) == 0.0)
    mixed = yv.clone(); mixed[n:] = wrong[n:]         # right on one-hop, wrong on two-hop
    split = t.query_accuracy(mixed, yv, xv)
    check("query_accuracy separates the two kinds", split == {"one_hop": 1.0, "two_hop": 0.0}, str(split))
    check("overall accuracy is their mean on the paired set", t.accuracy(None, mixed, yv, mv, xv) == 0.5)
    only_one = t.query_accuracy(yv[:n], yv[:n], xv[:n])
    check("query_accuracy gives nan for an absent kind",
          only_one["one_hop"] == 1.0 and math.isnan(only_one["two_hop"]))


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
    check("twohop -> TwoHopRecallTask (K=4, delay=4)",
          isinstance(tasks.make_task("twohop"), tasks.TwoHopRecallTask)
          and tasks.make_task("twohop").n_symbols == 4
          and tasks.make_task("twohop").delay == 4)
    parsed = tasks.make_task("twohop_K6_d2")
    check("twohop_K6_d2 parses symbols/delay",
          (parsed.n_symbols, parsed.delay, parsed.seq_len, parsed.feat_dim) == (6, 2, 15, 18))
    check("twohop_d0 keeps the default K",
          (tasks.make_task("twohop_d0").n_symbols, tasks.make_task("twohop_d0").delay) == (4, 0))
    for bad in ("twohop_x3", "twohop_K4_K5", "twohop_K"):
        try:
            tasks.make_task(bad)
            rejected = False
        except ValueError:
            rejected = True
        check(f"malformed name {bad} is rejected", rejected)
    check("acc_label / metric twohop",
          tasks.acc_label_for("twohop") == "classification accuracy (%)"
          and tasks.metric_for("twohop") == "accuracy")
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
        "learning_rule": "bptt", "feedback_mode": "exact_spatial",
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


def test_integration_twohop():
    print("── integration: one real update on two-hop recall (cross-entropy) ──")
    t = tasks.make_task("twohop")
    tp, trp, _ = t.init_params({"dt": 40}, {"valid_n_batch": 32}, {"n_neurons": [1, 16, 16, 1]})
    np.random.seed(0)
    xb, yb, mb = t.train_batch(tp, trp, 16, "cpu", torch.float32)
    target = yb[:, -1].argmax(-1)
    cases = [("MPN (2 MP layers)", "dmpn", mpn.DeepMultiPlasticNet, rule, ("W", "W1", "W_output"))
             for rule in ("bptt", "local_direct")]
    cases += [(label, "vanilla", cls, rule, ("W_input", "W_rec", "W_output"))
              for label, cls in (("RNN", rnn.LeakyRNN), ("GRU", gru.GRU))
              for rule in ("bptt", "local_diag_rflo")]
    for label, net_type, cls, rule, keys in cases:
        torch.manual_seed(0); np.random.seed(0)
        npar = _net_params(net_type, t.feat_dim, 16, t.n_output)
        if net_type == "dmpn":
            npar["n_neurons"] = [t.feat_dim, 16, 16, t.n_output]      # embedding + two MP layers
        npar["learning_rule"] = rule
        net = cls(npar, verbose=False).to(torch.float32)
        net.learning_rule = rule
        if net_type == "dmpn":
            check(f"twohop {label}: two MP layers built", len(net.mp_layers) == 2)
        params = [p for p in net.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in params]
        opt = torch.optim.Adam(params, lr=1e-3)
        opt.zero_grad()
        g = net.sequence_gradients(xb, yb, mb, loss_and_grad=t.loss_and_grad)
        ce = F.cross_entropy(g["outputs"][:, -1], target)
        check(f"twohop {label} {rule}: loss is the final-step cross-entropy",
              abs(float(g["loss"]) - float(ce)) < 1e-5, f"{float(g['loss']):.4f} vs {float(ce):.4f}")
        check(f"twohop {label} {rule}: gradients finite and nonzero ({', '.join(keys)})",
              all(bool(torch.isfinite(g[k]).all()) and float(g[k].abs().sum()) > 0 for k in keys))
        opt.step()
        if net.param_clamping:
            net.param_clamp()
        check(f"twohop {label} {rule}: the update changes the parameters and keeps them finite",
              any(not torch.equal(p, b) for p, b in zip(params, before))
              and all(bool(torch.isfinite(p).all()) for p in params))


if __name__ == "__main__":
    print("=" * 68)
    check_seqmnist_task("row", 28, 28)
    check_seqmnist_task("pixel", 784, 1)
    test_adding_task()
    test_twohop_task()
    test_registry()
    test_integration_seqmnist()
    test_integration_adding()
    test_integration_twohop()
    print("=" * 68)
    print(f"{'ALL TASK TESTS PASSED' if FAIL == 0 else f'{FAIL} CHECK(S) FAILED'}  "
          f"({PASS} passed, {FAIL} failed)")
    raise SystemExit(1 if FAIL else 0)
