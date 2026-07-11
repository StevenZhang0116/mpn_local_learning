#!/usr/bin/env python
# coding: utf-8
"""
Verify core/mpn_revise.py matches core/mpn.py in forward AND backward dynamics,
within tolerance.

mpn_revise.py is an efficiency refactor. Some changes are bitwise-identical
(loss-only helper, dropped clones), but the main speedups REORDER floating-point
ops and so are only identical up to round-off:
  - MP-layer forward: W_eff·x split into F.linear(x, W) + bmm(W⊙M, x), instead of
    materializing W + W⊙M and one big einsum.
  - update_M_matrix: M_pre = λM + η·postᵀpre computed directly (no −M+λM, no
    delta_M zeros alloc), masked path vectorized.
  - readout: torch.einsum('iI,BI->Bi', W_out, h) + b → F.linear(h, W_out, b).
  - LOCAL rules (local_direct / local_diag_rflo): eligibility + grad + trace fused
    (step_fn_for(mode)) WITHOUT materializing the (B,i,I) E tensor; the clean local
    M update takes a fast path (update_M_matrix_local_fast); the dmpn input
    embedding backprojection avoids building W_eff. All reorderings.
  - Tier-A overhead cuts: mode branch hoisted out of the per-step loop; time-major
    contiguous input/label/mask slicing; persistent (reused) time-major scratch
    buffers for the readout/embedding contractions. Calc-preserving.
  - Tier-B (opt-in): torch.compile of a pure-functional fused per-step core for
    direct/diag (mpn_revise.set_compile_local(True) / MPN_COMPILE_LOCAL=1); default
    OFF, eager fallback if compilation raises. Reordering, checked within tolerance.

So we compare with torch.allclose at a tolerance (float64: 1e-9; float32: 1e-5),
checking the WHOLE forward (rolled-out outputs) and the WHOLE backward (loss +
every parameter gradient) for BPTT and all three local rules, on dmpn and mpn1.
Both nets start from identical weights (state_dict copy) and identical random
eta/lam, fed the same data.

Coverage:
  - synthetic random trials (float64 tight, float32 loose, + a larger/longer case);
  - the vectorized/ragged update_mask path;
  - return_outputs=False must NOT change any gradient (only skips storing outputs);
  - persistent-scratch reuse safety (repeated calls identical; returned outputs not
    clobbered by a later call's scratch writes);
  - Tier-B torch.compile fused core matches the eager path within tolerance (opt-in);
  - REAL sequential-MNIST-pixel (T=784, 1 feature/step) alignment — the long-unroll
    stress case (skipped cleanly if the MNIST idx files are absent);
  - a seq-MNIST-pixel timing benchmark (mpn vs mpn_revise) at the training batch
    size (B=128, T=784), swept over hidden size; reported, not asserted.

Runs on GPU (requires CUDA — raises at import if unavailable): nets, data, and
masks all live on cuda, and the benchmark brackets its timed region with
torch.cuda.synchronize() so wall-clock reflects completed GPU work.

Run from this directory (on a CUDA box):  python test_mpn_revise.py
"""
import copy
import os
import time

import numpy as np
import torch

import _bootstrap  # prepends ../core + ../scripts to sys.path
# The reference implementation is core/mpn_archive.py; the efficiency-optimized
# one (formerly mpn_revise) is now the default core/mpn.py. Alias them to the
# names this test uses throughout: `mpn` = reference, `mpn_revise` = optimized.
import mpn_archive as mpn
import mpn as mpn_revise
import tasks


# This test runs on GPU (the Tier-A opts target CUDA launch/alloc overhead, so
# timing is only meaningful there). Require CUDA and fail loudly if absent.
if not torch.cuda.is_available():
    raise RuntimeError(
        "test_mpn_revise.py requires a CUDA GPU (torch.cuda.is_available() is False). "
        "Run it on the GPU box — the alignment checks and the seq-MNIST benchmark "
        "are meant to exercise / time the CUDA path.")
DEVICE = torch.device("cuda")

# Disable TF32 for the ALIGNMENT comparison: on Ampere+ GPUs float32 matmuls
# default to TF32 (~10-bit mantissa), which would inflate the mpn-vs-mpn_revise
# reorder difference past the float32 tolerance (1e-5) and cause false failures.
# We want to measure the reordering error, not TF32 rounding, so force true fp32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _sync():
    """Wait for all queued CUDA work (so wall-clock timing is accurate)."""
    torch.cuda.synchronize()


PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    ok = bool(cond)
    PASS += ok
    FAIL += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('   ' + extra) if extra else ''}")


def close(a, b, rtol, atol):
    """(is_close, max_abs_diff) between two tensors."""
    d = (a - b).abs().max().item()
    return torch.allclose(a, b, rtol=rtol, atol=atol), d


def net_params(net_type, n_in, n_hidden, n_out, m_update="hebb_assoc"):
    npar = {
        "net_type": net_type,
        "n_neurons": [n_in, n_hidden, n_out], "loss_type": "MSE", "activation": "tanh",
        "output_bias": True, "output_matrix": "", "dt": 40,
        "learning_rule": "bptt", "feedback_mode": "exact_readout",
        "ml_params": {"bias": True, "mp_type": "mult", "m_update_type": m_update,
                      "m_activation": "linear", "modulation_bounds": False,
                      "eta_type": "scalar", "eta_train": False, "lam_type": "scalar",
                      "lam_train": False, "m_time_scale": 400, "W_freeze": False},
    }
    if net_type == "dmpn":
        npar.update({"linear_embed": n_hidden, "input_layer_add": True,
                     "input_layer_add_trainable": True, "input_layer_bias": True,
                     "input_init_type": "xavier"})
    return npar


def build_pair(net_type, n_in, n_hidden, n_out, dtype, seed=0, m_update="hebb_assoc"):
    """Same net from mpn and mpn_revise, weights copied so they start identical,
    with matching random eta/lam. Returns (net_orig, net_rev)."""
    cls_name = "DeepMultiPlasticNet" if net_type == "dmpn" else "MultiPlasticNet"
    torch.manual_seed(seed)
    net_o = getattr(mpn, cls_name)(net_params(net_type, n_in, n_hidden, n_out, m_update),
                                   verbose=False).to(DEVICE).to(dtype)
    net_r = getattr(mpn_revise, cls_name)(net_params(net_type, n_in, n_hidden, n_out, m_update),
                                          verbose=False).to(DEVICE).to(dtype)
    net_r.load_state_dict(copy.deepcopy(net_o.state_dict()))
    with torch.no_grad():
        for mp_o, mp_r in zip(net_o.mp_layers, net_r.mp_layers):
            mp_o.eta.copy_(0.1 + 0.05 * torch.rand_like(mp_o.eta))
            mp_o.lam.copy_(0.5 * mp_o.lam_clamp + 0.4 * mp_o.lam_clamp
                           * torch.rand_like(mp_o.lam))
            mp_r.eta.copy_(mp_o.eta); mp_r.lam.copy_(mp_o.lam)
    return net_o, net_r


def make_data(B, T, n_in, n_out, dtype, seed=1):
    # Device-side generator so the random data is reproducible ON the GPU.
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    inputs = torch.randn(B, T, n_in, generator=g, dtype=dtype, device=DEVICE)
    labels = torch.randn(B, T, n_out, generator=g, dtype=dtype, device=DEVICE)
    masks = torch.rand(B, T, n_out, generator=g, dtype=dtype, device=DEVICE)
    return inputs, labels, masks


def compare_rule(net_type, rule, dtype, rtol, atol, n_in=5, n_hidden=6, n_out=3,
                 B=4, T=7, seed=0):
    tag = f"{net_type}/{rule}/{str(dtype).split('.')[-1]}"
    net_o, net_r = build_pair(net_type, n_in, n_hidden, n_out, dtype, seed=seed)
    net_o.learning_rule = rule; net_r.learning_rule = rule
    inputs, labels, masks = make_data(B, T, net_o.n_input, net_o.n_output, dtype)

    go = net_o.sequence_gradients(inputs, labels, masks)
    gr = net_r.sequence_gradients(inputs, labels, masks)

    keys = sorted(k for k in go if k not in ("loss", "outputs"))
    ok, d = close(go["loss"], gr["loss"], rtol, atol)
    check(f"{tag}: loss", ok, f"|Δ|={d:.1e}")
    ok, d = close(go["outputs"], gr["outputs"], rtol, atol)
    check(f"{tag}: outputs (forward)", ok, f"max|Δ|={d:.1e}")
    worst = 0.0; allok = True
    for k in keys:
        ok, d = close(go[k], gr[k], rtol, atol)
        allok &= ok; worst = max(worst, d)
    check(f"{tag}: all {len(keys)} grads (backward)", allok, f"max|Δ|={worst:.1e}")


def compare_masked_update(dtype, rtol, atol):
    """update_M_matrix's masked path was vectorized; check it against the original
    with a variable-length update_mask (local rules take update_masks)."""
    tag = f"dmpn/local_diag_rflo/{str(dtype).split('.')[-1]}/masked"
    B, T = 5, 8
    net_o, net_r = build_pair("dmpn", 5, 6, 3, dtype, seed=4)
    net_o.learning_rule = "local_diag_rflo"; net_r.learning_rule = "local_diag_rflo"
    inputs, labels, masks = make_data(B, T, net_o.n_input, net_o.n_output, dtype, seed=5)
    um = torch.ones(B, T, dtype=dtype, device=DEVICE)
    for bi in range(B):                       # ragged: zero out a tail per row
        um[bi, T - (bi % 3):] = 0.0
    go = net_o.sequence_gradients(inputs, labels, masks, update_masks=um)
    gr = net_r.sequence_gradients(inputs, labels, masks, update_masks=um)
    worst = max((go[k] - gr[k]).abs().max().item()
                for k in go if k not in ("loss", "outputs"))
    ok = all(close(go[k], gr[k], rtol, atol)[0]
             for k in go if k not in ("loss", "outputs"))
    check(f"{tag}: grads w/ update_mask", ok, f"max|Δ|={worst:.1e}")


def check_return_outputs_flag(dtype=torch.float64):
    """mpn_revise adds return_outputs=False (skips storing the (B,T,·) outputs when
    the default masked-MSE loss is used). It must not change the learning: every
    GRADIENT stays bit-identical to the return_outputs=True call on the SAME net,
    and result['outputs'] becomes None. The scalar LOSS is only compared up to
    round-off — with outputs stored it is one batched sum masked_mse_loss_only(...),
    without them it is the streamed per-step sum loss_sum/N (same value, different
    reduction order). Covers all local rules + BPTT."""
    for net_type in ("dmpn", "mpn1"):
        for rule in ("bptt", "local_exact_rowlocal", "local_diag_rflo", "local_direct"):
            _, net = build_pair(net_type, 5, 6, 3, dtype, seed=11)
            net.learning_rule = rule
            inputs, labels, masks = make_data(4, 7, net.n_input, net.n_output, dtype, seed=12)
            g_full = net.sequence_gradients(inputs, labels, masks, return_outputs=True)
            g_skip = net.sequence_gradients(inputs, labels, masks, return_outputs=False)
            keys = [k for k in g_full if k not in ("loss", "outputs")]
            grads_equal = all(torch.equal(g_full[k], g_skip[k]) for k in keys)
            loss_close = torch.allclose(g_full["loss"], g_skip["loss"], rtol=1e-6, atol=1e-8)
            ok = grads_equal and (g_skip["outputs"] is None) and loss_close
            check(f"{net_type}/{rule}: return_outputs=False keeps grads (loss ~eq)", ok)


def check_scratch_reuse_safe(dtype=torch.float64):
    """Tier-A reuses persistent scratch buffers (self._scratch) across calls for the
    readout/embedding contractions, and slices inputs time-major. Two hazards to rule
    out: (a) calling sequence_gradients twice must give IDENTICAL grads (the loop is
    stateless across calls — reset_state each time), and (b) the outputs returned by
    an earlier call must NOT be clobbered by a later call's scratch writes (i.e.
    result['outputs'] must be an independent tensor, not a view of a reused buffer).
    Covers all local rules + BPTT on both nets."""
    for net_type in ("dmpn", "mpn1"):
        for rule in ("bptt", "local_exact_rowlocal", "local_diag_rflo", "local_direct"):
            _, net = build_pair(net_type, 5, 6, 3, dtype, seed=21)
            net.learning_rule = rule
            inputs, labels, masks = make_data(4, 7, net.n_input, net.n_output, dtype, seed=22)
            g1 = net.sequence_gradients(inputs, labels, masks, return_outputs=True)
            out1 = g1["outputs"].clone()                 # snapshot before the 2nd call
            keys = [k for k in g1 if k not in ("loss", "outputs")]
            g1_grads = {k: g1[k].clone() for k in keys}
            # A DIFFERENT batch on the 2nd call → its scratch writes differ; if
            # result['outputs'] aliased a reused buffer, out1 would change.
            inp2, lab2, msk2 = make_data(4, 7, net.n_input, net.n_output, dtype, seed=23)
            net.sequence_gradients(inp2, lab2, msk2, return_outputs=True)
            outputs_intact = torch.equal(g1["outputs"], out1)
            # Re-run the ORIGINAL batch → grads must reproduce exactly (idempotent).
            g3 = net.sequence_gradients(inputs, labels, masks, return_outputs=True)
            grads_repro = all(torch.equal(g1_grads[k], g3[k]) for k in keys)
            check(f"{net_type}/{rule}: scratch reuse safe (outputs intact + grads reproducible)",
                  outputs_intact and grads_repro)


def check_compile_local(dtype=torch.float32, rtol=1e-4, atol=1e-5):
    """Tier-B: torch.compile of the fused per-step core (opt-in via
    mpn_revise.set_compile_local) must match the eager path within tolerance.
    Only direct/diag with hebb_assoc take the compiled core (see can_compile_step);
    the M/A/Q traces + grads are produced by the compiled region. Runs the SAME net
    eager then compiled (fresh state each call via reset_state) and compares grads.
    Wider tolerance than the mpn-vs-mpn_revise check because torch.compile fuses/
    reorders float ops; the first compiled call also triggers compilation (slow).
    Falls back to eager internally if compilation raises, so it never hard-fails."""
    import mpn as mr  # the efficiency-optimized impl (formerly mpn_revise)
    for net_type in ("dmpn", "mpn1"):
        for rule in ("local_diag_rflo", "local_direct"):
            _, net = build_pair(net_type, 5, 8, 3, dtype, seed=31)
            net.learning_rule = rule
            inputs, labels, masks = make_data(4, 9, net.n_input, net.n_output, dtype, seed=32)
            mr.set_compile_local(False)
            ge = net.sequence_gradients(inputs, labels, masks)
            mr.set_compile_local(True)
            try:
                gc = net.sequence_gradients(inputs, labels, masks)   # same net → reset_state
            finally:
                mr.set_compile_local(False)
            keys = [k for k in ge if k not in ("loss", "outputs")]
            worst = max((ge[k] - gc[k]).abs().max().item() for k in keys)
            ok = all(close(ge[k], gc[k], rtol, atol)[0] for k in keys)
            check(f"{net_type}/{rule}: torch.compile core matches eager", ok, f"max|Δ|={worst:.1e}")


def _seqmnist_batch(task, B, dtype, seed=0):
    """A real seq-MNIST-pixel batch (B, 784, 1) via the task adapter, plus the
    one-hot labels / final-step mask. Returns (inputs, labels, mask)."""
    tp, trp, _ = task.init_params({"dt": 40}, {"valid_n_batch": B}, {"n_neurons": [1, 8, 1]})
    np.random.seed(seed)
    return task.train_batch(tp, trp, B, DEVICE, dtype)


def compare_rule_seqmnist(net_type, rule, dtype, rtol, atol, n_hidden=32, B=8, seed=0):
    """Alignment on REAL sequential-MNIST-pixel data (T=784, 1 feature/step) — the
    long-sequence stress case the synthetic tests don't cover. Same protocol as
    compare_rule but the data comes from tasks.SeqMNISTTask(mode='pixel')."""
    task = tasks.SeqMNISTTask(mode="pixel")
    n_in, n_out = task.feat_dim, task.n_classes          # 1 -> 10
    tag = f"{net_type}/{rule}/{str(dtype).split('.')[-1]}/seqmnist_pixel"
    net_o, net_r = build_pair(net_type, n_in, n_hidden, n_out, dtype, seed=seed)
    net_o.learning_rule = rule; net_r.learning_rule = rule
    inputs, labels, masks = _seqmnist_batch(task, B, dtype, seed=seed)

    go = net_o.sequence_gradients(inputs, labels, masks)
    gr = net_r.sequence_gradients(inputs, labels, masks)
    keys = sorted(k for k in go if k not in ("loss", "outputs"))
    ok, d = close(go["loss"], gr["loss"], rtol, atol)
    check(f"{tag}: loss", ok, f"|Δ|={d:.1e}")
    ok, d = close(go["outputs"], gr["outputs"], rtol, atol)
    check(f"{tag}: outputs (forward)", ok, f"max|Δ|={d:.1e}")
    worst = 0.0; allok = True
    for k in keys:
        ok, d = close(go[k], gr[k], rtol, atol)
        allok &= ok; worst = max(worst, d)
    check(f"{tag}: all {len(keys)} grads (backward)", allok, f"max|Δ|={worst:.1e}")


def benchmark_seqmnist(net_type, rule, n_hidden=64, B=16, reps=5, warmup=2):
    """Wall-time per sequence_gradients call on seq-MNIST-pixel (T=784), mpn vs
    mpn_revise. Prints ms/call and the speedup; the two nets share weights/data so
    it is a like-for-like comparison. (float32 — the training dtype.)"""
    task = tasks.SeqMNISTTask(mode="pixel")
    n_in, n_out = task.feat_dim, task.n_classes
    net_o, net_r = build_pair(net_type, n_in, n_hidden, n_out, torch.float32, seed=0)
    net_o.learning_rule = rule; net_r.learning_rule = rule
    inputs, labels, masks = _seqmnist_batch(task, B, torch.float32, seed=0)

    def timeit(net):
        # Warm up (allocations, cuDNN autotune, lazy init), then sync so t0 starts
        # clean; sync again before stopping so we time completed GPU work, not just
        # the async kernel-launch queue.
        for _ in range(warmup):
            net.sequence_gradients(inputs, labels, masks)
        _sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            net.sequence_gradients(inputs, labels, masks)
        _sync()
        return 1000.0 * (time.perf_counter() - t0) / reps

    import mpn as mr  # the efficiency-optimized impl (formerly mpn_revise)
    mr.set_compile_local(False)
    to, tr = timeit(net_o), timeit(net_r)
    speedup = to / tr if tr > 0 else float("nan")
    line = (f"  {net_type}/{rule:16} T={inputs.shape[1]} h={n_hidden} B={B}: "
            f"orig {to:8.1f} ms  revise {tr:8.1f} ms  ({speedup:.2f}x)")
    # Tier-B: also time the torch.compile'd local core (direct/diag hebb_assoc).
    if net_r.mp_layers[0].m_update_type == "hebb_assoc" and rule in ("local_diag_rflo", "local_direct"):
        mr.set_compile_local(True)
        try:
            tc_ = timeit(net_r)   # first call compiles (warmup absorbs it)
            line += f"  revise+compile {tc_:8.1f} ms ({to / tc_:.2f}x)"
        except Exception as e:
            line += f"  [compile failed: {e}]"
        finally:
            mr.set_compile_local(False)
    print(line)


RULES = ["bptt", "local_exact_rowlocal", "local_diag_rflo", "local_direct"]

if __name__ == "__main__":
    print("=" * 70)
    # float64: tight tolerance (pure round-off from reordering).
    print("── float64 (rtol=atol=1e-9) ──")
    for net_type in ("dmpn", "mpn1"):
        for rule in RULES:
            compare_rule(net_type, rule, torch.float64, 1e-9, 1e-9)
    compare_masked_update(torch.float64, 1e-9, 1e-9)

    # float32: the training dtype; looser tolerance as requested.
    print("── float32 (rtol=1e-5, atol=1e-6) ──")
    for net_type in ("dmpn", "mpn1"):
        for rule in RULES:
            compare_rule(net_type, rule, torch.float32, 1e-5, 1e-6)
    compare_masked_update(torch.float32, 1e-5, 1e-6)

    # A larger/longer float32 case to stress accumulation over time.
    print("── float32 larger (dmpn, hidden=16, T=20) ──")
    for rule in RULES:
        compare_rule("dmpn", rule, torch.float32, 1e-5, 1e-6,
                     n_in=8, n_hidden=16, n_out=4, B=6, T=20, seed=7)

    # return_outputs=False must not change grads/loss (v2 feature; self-consistency).
    print("── return_outputs=False self-consistency (float64) ──")
    check_return_outputs_flag(torch.float64)

    # Tier-A persistent-scratch reuse: repeated calls must be identical and a
    # returned outputs tensor must not be clobbered by a later call's scratch.
    print("── persistent-scratch reuse safety (float64) ──")
    check_scratch_reuse_safe(torch.float64)

    # Tier-B torch.compile of the fused per-step core must match eager (opt-in).
    print("── torch.compile local core matches eager (float32, opt-in path) ──")
    check_compile_local(torch.float32)

    # Real data, long sequence: seq-MNIST-pixel (T=784, 1 feature/step). This is
    # the true stress test for accumulation over a long unroll. Skips cleanly if
    # the raw MNIST idx files are absent.
    _task = tasks.SeqMNISTTask(mode="pixel")
    if not os.path.isfile(os.path.join(_task._root({}), "train-images-idx3-ubyte")):
        print(f"── seq-MNIST-pixel: SKIP (MNIST idx files not found at {_task._root({})}) ──")
    else:
        print("── seq-MNIST-pixel alignment (T=784, float32, rtol=1e-5, atol=1e-6) ──")
        for net_type in ("dmpn", "mpn1"):
            for rule in RULES:
                compare_rule_seqmnist(net_type, rule, torch.float32, 1e-5, 1e-6,
                                      n_hidden=32, B=8)

        # Timing benchmark on the full training batch size (B=128, T=784), swept
        # over hidden size. reps=10 timed calls (after warmup) to average out noise.
        # Reported, not asserted. (Slow at T=784 — this is the real workload.)
        print("── seq-MNIST-pixel benchmark (mpn vs mpn_revise, float32, B=128, "
              "reps=10) ──")
        for n_hidden in (64, 128):
            for rule in ("bptt", "local_diag_rflo", "local_direct"):
                benchmark_seqmnist("dmpn", rule, n_hidden=n_hidden, B=128, reps=20)

    print("=" * 70)
    print(f"{'ALL MATCH within tol' if FAIL == 0 else f'{FAIL} CHECK(S) FAILED'}  "
          f"({PASS} passed, {FAIL} failed)")
    raise SystemExit(1 if FAIL else 0)
