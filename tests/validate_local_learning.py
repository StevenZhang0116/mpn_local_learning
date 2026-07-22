#!/usr/bin/env python
# coding: utf-8
"""
Validate the local-learning rules for a single-MP-layer MultiPlasticNet.

Two rules, both forward-mode (RTRL-style) eligibility traces, no BPTT:
  - exact row-local ('local_exact_rowlocal'): full trace P (B,post,pre,pre).
    EXACT — equals BPTT in the clean config.
  - diagonal RFLO ('local_diag_rflo'): same-synapse trace A (B,post,pre),
    dropping off-synapse (J != I) plastic sensitivities. APPROXIMATE.

Tiers (see the plan in the extension description):
  Tier 1  exact row-local == BPTT (autograd), across many configs.
  Tier 2  diagonal RFLO == exact row-local == BPTT when n_input == 1 (no
          off-diagonal terms to drop → the approximation is exact there).
  Tier 3  diagonal RFLO == a reference built from the exact P-rule with all
          off-diagonal P^I_{iJ} (J != I) manually zeroed each step.
  Extra   with n_input > 1 the diagonal RFLO MISMATCHES BPTT (sanity: it really
          is an approximation, not accidentally exact).
  Tier 5  direct/instantaneous rule ('local_direct'): == BPTT at T=1 and at
          eta=0; grad_W == diagonal RFLO with A=0; an approximation otherwise.
  Tier 6  deep-net input embedding (direct 3-factor): exact at eta=0, approximate
          otherwise, reduces to the single-MP net when no embedding.
  Tier 7  leaky RNN RFLO: == BPTT at W_rec=0; approximates otherwise (readout exact).
  Tier 8  pre-only update (hebb_pre): all local rules == BPTT for the MP layer
          (M is input-only → dM/dW=0); deep-net input embedding stays approximate.
  Tier 4  the same claims on the REAL delaygo task (train_mpn config / pipeline):
          exact == BPTT, diagonal RFLO an appreciable approximation. This is the
          integration check (Tiers 1-3 use synthetic random tensors).

Runs in float64 for a tight tolerance.
Run from this directory (mpn_local_learning/tests):  python validate_local_learning.py
"""
import torch

import _bootstrap  # prepends ../core + ../scripts to sys.path; exposes ROOT
import mpn


def build_net(n_input, n_hidden, n_output, activation, layer_bias, output_bias,
              eta_type, lam_type, learning_rule='local_exact_rowlocal',
              feedback_mode='exact_spatial', m_update_type='hebb_assoc', seed=0):
    torch.manual_seed(seed)
    net_params = {
        'n_neurons': [n_input, n_hidden, n_output],
        'loss_type': 'MSE',
        'activation': activation,
        'output_bias': output_bias,
        'output_matrix': '',
        'dt': 40,
        'learning_rule': learning_rule,
        'feedback_mode': feedback_mode,
        'ml_params': {
            'bias': layer_bias,
            'mp_type': 'mult',
            'm_update_type': m_update_type,
            'm_activation': 'linear',
            'modulation_bounds': False,
            'eta_type': eta_type,
            'eta_train': False,
            'lam_type': lam_type,
            'lam_train': False,
            'm_time_scale': 400,
            'W_freeze': False,
        },
    }
    net = mpn.MultiPlasticNet(net_params, verbose=False)
    # Randomize eta/lam so vector/matrix shapes are actually exercised
    # (lam kept in (0, lam_clamp) so M stays well-behaved).
    with torch.no_grad():
        net.mp_layer.eta.copy_(0.1 + 0.05 * torch.rand_like(net.mp_layer.eta))
        net.mp_layer.lam.copy_(0.5 * net.mp_layer.lam_clamp
                               + 0.4 * net.mp_layer.lam_clamp * torch.rand_like(net.mp_layer.lam))
    return net.double()


def make_data(B, T, n_input, n_output, use_mask=False):
    dtype = torch.float64
    inputs = torch.randn(B, T, n_input, dtype=dtype)
    labels = torch.randn(B, T, n_output, dtype=dtype)
    masks = torch.rand(B, T, n_output, dtype=dtype)  # float 'cost' mask
    update_masks = None
    if use_mask:
        update_masks = torch.ones(B, T, dtype=dtype)
        for bi in range(B):
            cut = T - (bi % 3)
            update_masks[bi, cut:] = 0.0
            masks[bi, cut:, :] = 0.0
    return inputs, labels, masks, update_masks


def diag_rflo_reference(net, inputs, labels, masks, update_masks=None):
    """Reference diagonal RFLO built from the EXACT P-rule with all off-diagonal
    P^I_{iJ} (J != I) zeroed after every trace update. Should match the native
    A-based implementation exactly. Runs on a fresh copy so it is independent."""
    import copy
    net = copy.deepcopy(net)
    mp = net.mp_layer
    B, T, _ = inputs.shape
    dev, dt = inputs.device, inputs.dtype
    net.reset_state(B=B)
    mp.reset_local_learning_state(B=B)
    diag_idx = torch.arange(mp.n_input)

    grad_W = torch.zeros_like(mp.W); grad_b = torch.zeros_like(mp.b)
    grad_Wout = torch.zeros_like(net.W_output); grad_bout = torch.zeros_like(net.b_output)
    outputs = torch.zeros(B, T, net.n_output, dtype=dt, device=dev)
    feedback = net.W_output if net.feedback_mode == 'exact_spatial' else net.B_feedback
    N = B * T * net.n_output

    with torch.no_grad():
        for t in range(T):
            x_t = inputs[:, t, :]
            hidden_pre, _ = mp(x_t)
            hidden = net.act_fn(hidden_pre)
            output = torch.einsum('iI,BI->Bi', net.W_output, hidden) + net.b_output.unsqueeze(0)
            outputs[:, t, :] = output
            phi_prime = net.act_fn_p(hidden_pre)
            E, R = mp.compute_exact_rowlocal_eligibility(x_t, phi_prime)

            m_t, y_t = masks[:, t, :], labels[:, t, :]
            grad_output = (2.0 / N) * m_t * (m_t * output - m_t * y_t)
            ell = grad_output @ feedback
            grad_W += torch.einsum('Bi,BiI->iI', ell, E)
            grad_b += torch.einsum('Bi,Bi->i', ell, R)
            grad_Wout += torch.einsum('Ba,Bi->ai', grad_output, hidden)
            grad_bout += grad_output.sum(0)

            um = None if update_masks is None else update_masks[:, t]
            mp.update_exact_rowlocal_traces(x_t, E, R, update_mask=um)
            # Zero off-diagonal plastic sensitivities: keep only P^I_{iI}.
            P = mp.P                                    # (B, i, I, J)
            P_diag = P[:, :, diag_idx, diag_idx]        # (B, i, I) = P^I_{iI}
            P.zero_()
            P[:, :, diag_idx, diag_idx] = P_diag
            mp.update_M_matrix(x_t, hidden, update_mask=um)

    params = net._trainable_params()
    all_grads = {'W': grad_W, 'b': grad_b, 'W_output': grad_Wout, 'b_output': grad_bout}
    return {k: all_grads[k] for k in params}


def compare(a, b, keys, rtol=1e-6, atol=1e-8):
    ok, diffs = True, {}
    for k in keys:
        d = (a[k] - b[k]).abs()
        denom = b[k].abs().max().clamp_min(1e-12)
        diffs[k] = (d.max().item(), (d.max() / denom).item())
        if not torch.allclose(a[k], b[k], rtol=rtol, atol=atol):
            ok = False
    return ok, diffs


def grad_keys(layer_bias, output_bias):
    keys = ['W', 'W_output']
    if layer_bias:
        keys.append('b')
    if output_bias:
        keys.append('b_output')
    return keys


def fmt(diffs, keys):
    return '  '.join(f"{k}:rel={diffs[k][1]:.1e}" for k in keys)


def tier1_exact_vs_bptt():
    print("── Tier 1: exact row-local == BPTT ─────────────────────────────────")
    cases = [
        ("linear, no bias, B=1",          'linear', False, False, 1, 'scalar', 'scalar', False),
        ("linear, +both biases, B=1",     'linear', True,  True,  1, 'scalar', 'scalar', False),
        ("tanh, +both biases, B=8",       'tanh',   True,  True,  8, 'scalar', 'scalar', False),
        ("tanh, eta/lam matrix, B=8",     'tanh',   True,  True,  8, 'matrix', 'matrix', False),
        ("tanh, variable-length masks",   'tanh',   True,  True,  8, 'matrix', 'matrix', True),
        ("ReLU, +both biases, B=8",       'ReLU',   True,  True,  8, 'scalar', 'scalar', False),
    ]
    allok = True
    for i, (name, act, lb, ob, B, et, lt, um) in enumerate(cases):
        net = build_net(5, 7, 3, act, lb, ob, et, lt, 'local_exact_rowlocal', seed=i)
        inp, lab, msk, umask = make_data(B, 12, 5, 3, use_mask=um)
        ref = net.bptt_gradients(inp, lab, msk)
        loc = net.local_gradients(inp, lab, msk, update_masks=umask)
        keys = grad_keys(lb, ob)
        ok, diffs = compare(loc, ref, keys)
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<32} {fmt(diffs, keys)}")
    return allok


def tier2_diag_matches_when_ninput1():
    print("── Tier 2: diagonal RFLO == BPTT when n_input == 1 ─────────────────")
    allok = True
    for i, (act, B) in enumerate([('linear', 1), ('tanh', 8), ('ReLU', 4)]):
        net = build_net(1, 6, 3, act, True, True, 'scalar', 'scalar', 'local_diag_rflo', seed=i)
        inp, lab, msk, _ = make_data(B, 10, 1, 3)
        ref = net.bptt_gradients(inp, lab, msk)
        dia = net.local_diag_rflo_gradients(inp, lab, msk)
        keys = grad_keys(True, True)
        ok, diffs = compare(dia, ref, keys)
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] n_input=1, {act}, B={B:<10} {fmt(diffs, keys)}")
    return allok


def tier3_diag_matches_zeroed_reference():
    print("── Tier 3: native diagonal RFLO == off-diagonal-zeroed P reference ──")
    allok = True
    cases = [
        ("tanh, B=8, scalar",   'tanh', 8, 'scalar', 'scalar', 'exact_spatial', False),
        ("tanh, B=8, matrix",   'tanh', 8, 'matrix', 'matrix', 'exact_spatial', False),
        ("tanh, masks",         'tanh', 8, 'matrix', 'matrix', 'exact_spatial', True),
        ("tanh, random feedbk", 'tanh', 8, 'scalar', 'scalar', 'layerwise_fa',  False),
    ]
    for i, (name, act, B, et, lt, fb, um) in enumerate(cases):
        net = build_net(5, 7, 3, act, True, True, et, lt, 'local_diag_rflo', fb, seed=i)
        inp, lab, msk, umask = make_data(B, 12, 5, 3, use_mask=um)
        dia = net.local_diag_rflo_gradients(inp, lab, msk, update_masks=umask)
        ref = diag_rflo_reference(net, inp, lab, msk, update_masks=umask)
        keys = grad_keys(True, True)
        ok, diffs = compare(dia, ref, keys)
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<20} {fmt(diffs, keys)}")
    return allok


def extra_diag_differs_from_bptt():
    print("── Sanity: diagonal RFLO MISMATCHES BPTT when n_input > 1 ──────────")
    net = build_net(5, 7, 3, 'tanh', True, True, 'scalar', 'scalar', 'local_diag_rflo', seed=0)
    inp, lab, msk, _ = make_data(8, 12, 5, 3)
    ref = net.bptt_gradients(inp, lab, msk)
    dia = net.local_diag_rflo_gradients(inp, lab, msk)
    keys = grad_keys(True, True)
    _, diffs = compare(dia, ref, keys)
    # W should differ appreciably; readout W_output must still match (it's exact).
    w_rel = diffs['W'][1]
    wout_rel = diffs['W_output'][1]
    ok = (w_rel > 1e-3) and (wout_rel < 1e-8)
    print(f"  [{'PASS' if ok else 'FAIL'}] W differs (rel={w_rel:.2e} > 1e-3) while "
          f"W_output stays exact (rel={wout_rel:.1e})")
    return ok


def tier5_direct_local():
    """Characterize the direct/instantaneous rule ('local_direct'):
      (a) T=1  → direct == BPTT for W, b (no temporal plasticity path yet).
      (b) eta=0 → direct == BPTT across time (M carries no W→h→M credit).
      (c) direct grad_W == diagonal RFLO grad_W with the A trace forced to 0.
      (d) with n_input>1, eta>0, T>1: direct MISMATCHES BPTT (it's the strongest
          approximation), while readout W_output stays exact."""
    print("── Tier 5: direct/instantaneous local rule ─────────────────────────")
    allok = True
    keys = grad_keys(True, True)

    # (a) T=1 equivalence to BPTT.
    net = build_net(5, 7, 3, 'tanh', True, True, 'scalar', 'scalar', 'local_direct', seed=0)
    inp, lab, msk, _ = make_data(4, 1, 5, 3)
    ok_a, d_a = compare(net.local_direct_gradients(inp, lab, msk),
                        net.bptt_gradients(inp, lab, msk), keys)
    allok &= ok_a
    print(f"  [{'PASS' if ok_a else 'FAIL'}] T=1 direct == BPTT          {fmt(d_a, keys)}")

    # (b) eta=0 equivalence to BPTT across time.
    net = build_net(5, 7, 3, 'tanh', True, True, 'scalar', 'scalar', 'local_direct', seed=1)
    with torch.no_grad():
        net.mp_layer.eta.zero_()
    inp, lab, msk, _ = make_data(4, 10, 5, 3)
    ok_b, d_b = compare(net.local_direct_gradients(inp, lab, msk),
                        net.bptt_gradients(inp, lab, msk), keys)
    allok &= ok_b
    print(f"  [{'PASS' if ok_b else 'FAIL'}] eta=0 direct == BPTT        {fmt(d_b, keys)}")

    # (c) direct grad_W == diagonal RFLO grad_W with A forced to 0 each step.
    # The diag path is FUSED (_local_step_diag updates A inline); it no longer calls
    # update_diag_rflo_traces, so patch the fused step to zero A after each call
    # (patching update_diag_rflo_traces would be a dead no-op — the stale-test bug).
    net = build_net(5, 7, 3, 'tanh', True, True, 'scalar', 'scalar', 'local_direct', seed=2)
    inp, lab, msk, _ = make_data(4, 12, 5, 3)
    gd = net.local_direct_gradients(inp, lab, msk)
    mp_layer = net.mp_layer
    orig = mp_layer._local_step_diag
    def _a_zero(x, phi_prime, ell, eta, lam, update_mask=None):
        gW, gb = orig(x, phi_prime, ell, eta, lam, update_mask=update_mask)
        mp_layer.A.zero_()
        return gW, gb
    mp_layer._local_step_diag = _a_zero
    gdiag0 = net.local_diag_rflo_gradients(inp, lab, msk)
    mp_layer._local_step_diag = orig
    w_rel = (gd['W'] - gdiag0['W']).abs().max().item() / max(gdiag0['W'].abs().max().item(), 1e-12)
    ok_c = w_rel < 1e-10
    allok &= ok_c
    print(f"  [{'PASS' if ok_c else 'FAIL'}] direct grad_W == diag(A=0) grad_W   rel={w_rel:.1e}")

    # (d) direct is an approximation for n_input>1 (W differs, readout exact).
    net = build_net(5, 7, 3, 'tanh', True, True, 'scalar', 'scalar', 'local_direct', seed=3)
    inp, lab, msk, _ = make_data(8, 12, 5, 3)
    _, d_d = compare(net.local_direct_gradients(inp, lab, msk),
                     net.bptt_gradients(inp, lab, msk), keys)
    ok_d = (d_d['W'][1] > 1e-3) and (d_d['W_output'][1] < 1e-8)
    allok &= ok_d
    print(f"  [{'PASS' if ok_d else 'FAIL'}] direct approximates BPTT    "
          f"W:rel={d_d['W'][1]:.2e} (>1e-3)  W_output:rel={d_d['W_output'][1]:.1e} (exact)")
    return allok


def tier4_real_task():
    """Integration check on the ACTUAL train_mpn pipeline: real task trials (real
    seq length, cost masks) through the exact train_mpn net config, which is now
    a DeepMultiPlasticNet with a trainable input embedding. Confirms on the real
    task: exact-MP == BPTT for the MP layer + readout, and diagonal RFLO is an
    appreciable approximation (readout stays exact). The input embedding uses the
    direct 3-factor rule (an approximation under all local rules), so W_in/b_in
    are excluded from the exact==BPTT keys and characterized in Tier 6."""
    print("── Tier 4: real task (train_mpn deep-MPN config) ───────────────────")
    try:
        import numpy as np
        import mpn_tasks
        import train_mpn as tm
    except Exception as e:
        print(f"  [SKIP] could not import task pipeline: {e}")
        return True

    np.random.seed(tm.SEED)
    torch.manual_seed(tm.SEED)
    # Tier 4 is the MULTITASK (ring-task) integration check, so pin a ring task
    # here regardless of train_mpn's current default RULESET (which may be a
    # non-ring task like seq-MNIST that generate_trials_wrap doesn't handle).
    tm.RULESET = "delaygo"
    # This tier's invariant is exact-local == BPTT on the hidden W, which only holds
    # with exact_spatial feedback (the FA modes deliberately perturb it). Pin the
    # feedback config here so the check is independent of train_mpn's current
    # globals: exact_spatial, and input_mode='match' so no cross-rule splice runs.
    tm.FEEDBACK_MODE = "exact_spatial"
    tm.INPUT_MODE = "match"
    task_params, train_params, net_params = tm.build_params()
    # Correctness is size-independent; shrink hidden/embedding so long-sequence
    # tasks (e.g. contextdelaydm1) fit under a tight memory ceiling. This still
    # exercises the real task pipeline (real seq length, cost masks) end to end.
    net_params["n_neurons"] = [net_params["n_neurons"][0], 24, net_params["n_neurons"][-1]]
    net_params["linear_embed"] = 24
    task_params, train_params, net_params = mpn_tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params)
    )
    net_params["prefs"] = mpn_tasks.get_prefs(task_params["hp"])
    # Small batch too: the exact P-trace is (B, post, pre, pre).
    B = 4
    net = mpn.DeepMultiPlasticNet(net_params, verbose=False).double()

    data, _ = mpn_tasks.generate_trials_wrap(
        task_params, B, rules=task_params["rules"],
        mode_input="random_batch", device=torch.device("cpu"),
    )
    inp, lab, msk = (d.double() for d in data)
    # MP-layer + readout keys where exact-local must equal BPTT (embedding is
    # the direct 3-factor approximation, checked separately).
    keys = grad_keys(net.mp_layers[0].layer_bias, net.b_output_active)

    ref = net.bptt_gradients(inp, lab, msk)
    exact = net.local_gradients(inp, lab, msk)
    diag = net.local_diag_rflo_gradients(inp, lab, msk)

    ok_exact, d_exact = compare(exact, ref, keys)
    _, d_diag = compare(diag, ref, keys)
    diag_w_rel = d_diag['W'][1]
    diag_wout_rel = d_diag['W_output'][1]
    ok_diag = (diag_w_rel > 1e-3) and (diag_wout_rel < 1e-8)

    print(f"  [{'PASS' if ok_exact else 'FAIL'}] exact MP+readout == BPTT   "
          f"{fmt(d_exact, keys)}")
    print(f"  [{'PASS' if ok_diag else 'FAIL'}] diag RFLO approximates      "
          f"W:rel={diag_w_rel:.2e} (>1e-3)  W_output:rel={diag_wout_rel:.1e} (exact)")
    return ok_exact and ok_diag


def tier6_deep_input_embedding():
    """The deep net's trainable input embedding (W_in, b_in), direct 3-factor rule:
      (a) eta=0 → exact-local == BPTT for ALL params incl. W_in/b_in (no M history).
      (b) eta>0 → W_in is an approximation to BPTT, while the readout stays exact.
      (c) with NO input layer, dmpn reduces to the single-MP net: exact == BPTT."""
    print("── Tier 6: deep-net input embedding (direct 3-factor) ──────────────")

    def build(seed, input_layer, trainable, eta):
        torch.manual_seed(seed)
        npar = {
            'n_neurons': [6, 12, 3], 'loss_type': 'MSE', 'activation': 'tanh',
            'output_bias': True, 'output_matrix': '', 'dt': 40,
            'input_layer_add': input_layer, 'input_layer_add_trainable': trainable,
            'linear_embed': 8, 'input_layer_bias': True,
            'learning_rule': 'local_exact_rowlocal', 'feedback_mode': 'exact_spatial',
            'ml_params': {'bias': True, 'mp_type': 'mult', 'm_update_type': 'hebb_assoc',
                          'm_activation': 'linear', 'modulation_bounds': False,
                          'eta_type': 'scalar', 'eta_train': False, 'lam_type': 'scalar',
                          'lam_train': False, 'm_time_scale': 400, 'W_freeze': False}}
        net = mpn.DeepMultiPlasticNet(npar, verbose=False).double()
        with torch.no_grad():
            net.mp_layers[0].eta.fill_(eta)
            net.mp_layers[0].lam.fill_(0.6 * net.mp_layers[0].lam_clamp)
        return net

    def data(B=4, T=10, n_in=6, n_out=3):
        torch.manual_seed(0)
        return (torch.randn(B, T, n_in), torch.randn(B, T, n_out), torch.rand(B, T, n_out))

    allok = True

    # (a) eta=0 → all params (incl. W_in, b_in) exact vs BPTT.
    net = build(1, True, True, 0.0); inp, lab, msk = data()
    ref, loc = net.bptt_gradients(inp, lab, msk), net.local_gradients(inp, lab, msk)
    keys = [k for k in ref if k not in ('loss', 'outputs')]
    ok_a, d_a = compare(loc, ref, keys)
    allok &= ok_a
    print(f"  [{'PASS' if ok_a else 'FAIL'}] eta=0 exact==BPTT (incl W_in) {fmt(d_a, keys)}")

    # (b) eta>0 → W_in approximate, readout exact.
    net = build(2, True, True, 0.12); inp, lab, msk = data()
    ref, loc = net.bptt_gradients(inp, lab, msk), net.local_gradients(inp, lab, msk)
    _, d_b = compare(loc, ref, ['W_in', 'W_output'])
    ok_b = (d_b['W_in'][1] > 1e-3) and (d_b['W_output'][1] < 1e-8)
    allok &= ok_b
    print(f"  [{'PASS' if ok_b else 'FAIL'}] eta>0 W_in approximates       "
          f"W_in:rel={d_b['W_in'][1]:.2e} (>1e-3)  W_output:rel={d_b['W_output'][1]:.1e} (exact)")

    # (c) no input layer → reduces to single-MP net; exact == BPTT.
    net = build(3, False, False, 0.12); inp, lab, msk = data()
    ref, loc = net.bptt_gradients(inp, lab, msk), net.local_gradients(inp, lab, msk)
    keys = [k for k in ref if k not in ('loss', 'outputs')]
    ok_c, d_c = compare(loc, ref, keys)
    allok &= ok_c
    print(f"  [{'PASS' if ok_c else 'FAIL'}] no-embed exact==BPTT          {fmt(d_c, keys)}")
    return allok


def tier7_rnn_rflo():
    """LeakyRNN RFLO ('local_diag_rflo') vs BPTT:
      (a) W_rec=0 → RFLO == BPTT for ALL params (no recurrent credit to drop).
      (b) W_rec!=0 → RFLO approximates (W_rec/W_input differ), readout exact."""
    print("── Tier 7: leaky RNN RFLO (local_diag_rflo) ────────────────────────")
    import rnn

    def build(seed, zero_wrec=False):
        torch.manual_seed(seed)
        npar = {'n_neurons': [6, 12, 3], 'loss_type': 'MSE', 'activation': 'tanh',
                'output_bias': True, 'hidden_bias': True, 'dt': 40, 'leaky': True,
                'alpha': 0.8, 'learning_rule': 'local_diag_rflo',
                'feedback_mode': 'exact_spatial'}
        net = rnn.LeakyRNN(npar, verbose=False).double()
        if zero_wrec:
            with torch.no_grad():
                net.W_rec.zero_()
        return net

    def data(B=4, T=10, n_in=6, n_out=3):
        torch.manual_seed(0)
        return (torch.randn(B, T, n_in), torch.randn(B, T, n_out), torch.rand(B, T, n_out))

    allok = True

    # (a) W_rec = 0 → RFLO exact.
    net = build(1, zero_wrec=True); inp, lab, msk = data()
    ref, loc = net.bptt_gradients(inp, lab, msk), net.local_diag_rflo_gradients(inp, lab, msk)
    keys = [k for k in ref if k not in ('loss', 'outputs')]
    ok_a, d_a = compare(loc, ref, keys)
    allok &= ok_a
    print(f"  [{'PASS' if ok_a else 'FAIL'}] W_rec=0 RFLO==BPTT           {fmt(d_a, keys)}")

    # (b) W_rec != 0 → approximation; readout exact.
    net = build(2); inp, lab, msk = data()
    ref, loc = net.bptt_gradients(inp, lab, msk), net.local_diag_rflo_gradients(inp, lab, msk)
    _, d_b = compare(loc, ref, ['W_rec', 'W_output'])
    ok_b = (d_b['W_rec'][1] > 1e-3) and (d_b['W_output'][1] < 1e-8)
    allok &= ok_b
    print(f"  [{'PASS' if ok_b else 'FAIL'}] W_rec!=0 RFLO approximates   "
          f"W_rec:rel={d_b['W_rec'][1]:.2e} (>1e-3)  W_output:rel={d_b['W_output'][1]:.1e} (exact)")
    return allok


def tier8_hebb_pre():
    """Pre-only Hebbian update ('hebb_pre'): M is input-only, so dM/dW = dM/db = 0
    for the MP layer and every plastic trace is identically zero.
      (a) all three local rules == BPTT on a single-MP-layer net (exact, no approx).
      (b) contrast: on hebb_assoc the diag/direct rules DIFFER from BPTT (so (a)
          is a real property of pre-only, not the test being trivially satisfied).
      (c) deep net: MP-layer params (W, b, W_output, b_output) exact under
          hebb_pre, but the input embedding W_in still feeds M, so W_in/b_in are
          an approximation — same as hebb_assoc."""
    print("── Tier 8: pre-only Hebbian update (hebb_pre) ──────────────────────")
    allok = True

    # (a) single-MP-layer, hebb_pre: exact / diag / direct all == BPTT.
    keys = grad_keys(True, True)
    for rule_name, rule in (("exact", "local_exact_rowlocal"),
                            ("diag", "local_diag_rflo"),
                            ("direct", "local_direct")):
        net = build_net(5, 7, 3, 'tanh', True, True, 'scalar', 'scalar',
                        learning_rule=rule, m_update_type='hebb_pre', seed=1)
        inp, lab, msk, _ = make_data(6, 12, 5, 3)
        ref = net.bptt_gradients(inp, lab, msk)
        loc = net._local_sequence_gradients(
            inp, lab, msk, {'local_exact_rowlocal': 'exact',
                            'local_diag_rflo': 'diag', 'local_direct': 'direct'}[rule])
        ok, d = compare(loc, ref, keys)
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] hebb_pre {rule_name:6} == BPTT     {fmt(d, keys)}")

    # (b) contrast: hebb_assoc diag genuinely differs (pre-only exactness is real).
    net = build_net(5, 7, 3, 'tanh', True, True, 'scalar', 'scalar',
                    m_update_type='hebb_assoc', seed=2)
    inp, lab, msk, _ = make_data(6, 12, 5, 3)
    _, d_b = compare(net.local_diag_rflo_gradients(inp, lab, msk),
                     net.bptt_gradients(inp, lab, msk), ['W'])
    ok_b = d_b['W'][1] > 1e-3
    allok &= ok_b
    print(f"  [{'PASS' if ok_b else 'FAIL'}] hebb_assoc diag DIFFERS      W:rel={d_b['W'][1]:.2e} (>1e-3)")

    # (c) deep net hebb_pre: MP-layer params exact; embedding W_in an approximation.
    npar = {'n_neurons': [5, 7, 3], 'loss_type': 'MSE', 'activation': 'tanh',
            'output_bias': True, 'output_matrix': '', 'dt': 40,
            'input_layer_add': True, 'input_layer_add_trainable': True,
            'linear_embed': 6, 'input_layer_bias': True,
            'learning_rule': 'local_exact_rowlocal', 'feedback_mode': 'exact_spatial',
            'ml_params': {'bias': True, 'mp_type': 'mult', 'm_update_type': 'hebb_pre',
                          'm_activation': 'linear', 'modulation_bounds': False,
                          'eta_type': 'scalar', 'eta_train': False, 'lam_type': 'scalar',
                          'lam_train': False, 'm_time_scale': 400, 'W_freeze': False}}
    torch.manual_seed(3)
    net = mpn.DeepMultiPlasticNet(npar, verbose=False).double()
    inp, lab, msk, _ = make_data(6, 12, 5, 3)
    ref, loc = net.bptt_gradients(inp, lab, msk), net.local_gradients(inp, lab, msk)
    ok_mp, d_mp = compare(loc, ref, ['W', 'b', 'W_output', 'b_output'])
    w_in_rel = compare(loc, ref, ['W_in'])[1]['W_in'][1]
    ok_c = ok_mp and (w_in_rel > 1e-3)
    allok &= ok_c
    print(f"  [{'PASS' if ok_c else 'FAIL'}] dmpn hebb_pre: MP exact, W_in approx   "
          f"{fmt(d_mp, ['W', 'W_output'])}  W_in:rel={w_in_rel:.2e} (>1e-3)")
    return allok


def _quartic_loss_and_grad(output, labels, mask):
    """A non-MSE differentiable masked loss for the custom-loss tier: mean of
    (mask*(out-lab))^4, with its analytic dL/d output. Differs from masked MSE so
    the local rules must actually USE the supplied gradient (not the hard-coded
    masked-MSE one) to match BPTT."""
    N = output.numel()
    d = mask * (output - labels)
    return (d ** 4).sum() / N, (4.0 / N) * mask * (d ** 3)


def tier9_custom_loss():
    """Custom (non-MSE) loss through loss_and_grad must be HONORED by the local
    rules, not just reported. Regression guard for the two-pass fix: previously the
    local rules always accumulated masked-MSE gradients and used loss_and_grad only
    for the scalar loss, so a custom loss gave a matching loss but WRONG grads.
      (a) T=1: every local rule (exact/diag/direct) == BPTT for ANY differentiable
          loss (no temporal plasticity path to omit) — this is the reporter's case.
      (b) eta=0, T>1: local == BPTT for the custom loss (M carries no W->h->M credit).
      (c) the reported loss equals BPTT's loss (both use the supplied loss).
    Covers MultiPlasticNet AND DeepMultiPlasticNet."""
    print("── Tier 9: custom loss honored by local rules ──────────────────────")
    allok = True
    lg = _quartic_loss_and_grad
    keys = grad_keys(True, True)

    def dmpn(seed, eta):
        npar = {'n_neurons': [5, 7, 3], 'loss_type': 'MSE', 'activation': 'tanh',
                'output_bias': True, 'output_matrix': '', 'dt': 40,
                'input_layer_add': True, 'input_layer_add_trainable': True,
                'linear_embed': 6, 'input_layer_bias': True,
                'learning_rule': 'local_exact_rowlocal', 'feedback_mode': 'exact_spatial',
                'ml_params': {'bias': True, 'mp_type': 'mult', 'm_update_type': 'hebb_assoc',
                              'm_activation': 'linear', 'modulation_bounds': False,
                              'eta_type': 'scalar', 'eta_train': False, 'lam_type': 'scalar',
                              'lam_train': False, 'm_time_scale': 400, 'W_freeze': False}}
        torch.manual_seed(seed)
        net = mpn.DeepMultiPlasticNet(npar, verbose=False).double()
        with torch.no_grad():
            net.mp_layers[0].eta.fill_(eta)
            net.mp_layers[0].lam.fill_(0.6 * net.mp_layers[0].lam_clamp)
        return net

    # (a) T=1: exact / diag / direct == BPTT for the quartic loss (single-MP net).
    for name, rule in (("exact", "local_exact_rowlocal"),
                       ("diag", "local_diag_rflo"), ("direct", "local_direct")):
        net = build_net(5, 7, 3, 'tanh', True, True, 'scalar', 'scalar', rule, seed=1)
        inp, lab, msk, _ = make_data(6, 1, 5, 3)
        ref = net.bptt_gradients(inp, lab, msk, loss_and_grad=lg)
        loc = net.sequence_gradients(inp, lab, msk, loss_and_grad=lg)
        ok, d = compare(loc, ref, keys)
        lgap = (ref['loss'] - loc['loss']).abs().item()
        ok = ok and lgap < 1e-10
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] T=1 custom {name:6} == BPTT    "
              f"{fmt(d, keys)}  loss_gap={lgap:.1e}")

    # (b) eta=0, T>1: exact local == BPTT for the quartic loss.
    net = build_net(5, 7, 3, 'tanh', True, True, 'scalar', 'scalar',
                    'local_exact_rowlocal', seed=2)
    with torch.no_grad():
        net.mp_layer.eta.zero_()
    inp, lab, msk, _ = make_data(6, 10, 5, 3)
    ref = net.bptt_gradients(inp, lab, msk, loss_and_grad=lg)
    loc = net.sequence_gradients(inp, lab, msk, loss_and_grad=lg)
    ok_b, d_b = compare(loc, ref, keys)
    allok &= ok_b
    print(f"  [{'PASS' if ok_b else 'FAIL'}] eta=0 custom exact == BPTT   {fmt(d_b, keys)}")

    # (c) deep net, T=1: exact-local (incl. W_in) == BPTT for the quartic loss.
    net = dmpn(3, 0.12)
    inp, lab, msk, _ = make_data(6, 1, 5, 3)
    ref = net.bptt_gradients(inp, lab, msk, loss_and_grad=lg)
    loc = net.sequence_gradients(inp, lab, msk, loss_and_grad=lg)
    dkeys = [k for k in ref if k not in ('loss', 'outputs')]
    ok_c, d_c = compare(loc, ref, dkeys)
    allok &= ok_c
    print(f"  [{'PASS' if ok_c else 'FAIL'}] dmpn T=1 custom == BPTT      {fmt(d_c, dkeys)}")
    return allok


def _onehot_final_step(B, T, n_out, seed=0):
    """One-hot target on the final step + a {0,1} final-step cost mask (all
    channels), the seq-MNIST convention. Random logits as 'output' come from the
    net; here we only build labels/mask."""
    g = torch.Generator().manual_seed(seed)
    lab = torch.zeros(B, T, n_out, dtype=torch.float64)
    msk = torch.zeros(B, T, n_out, dtype=torch.float64)
    cls = torch.randint(0, n_out, (B,), generator=g)
    lab[torch.arange(B), T - 1, cls] = 1.0
    msk[:, T - 1, :] = 1.0
    return lab, msk


def tier10_cross_entropy():
    """Masked cross-entropy (mpn.masked_cross_entropy_loss_and_grad), the seq-MNIST
    objective, is a proper non-default loss the local rules honor via the two-pass
    path. Guards the CE loss AND its use as a task loss.
      (a) the analytic grad_output equals autograd d(loss)/d(logits), and the loss
          equals F.cross_entropy on the scored (final) step;
      (b) T=1: every local rule (exact/diag/direct) == BPTT under CE;
      (c) eta=0, T>1: exact local == BPTT under CE;
      (d) deep net, T=1: exact-local (incl. W_in) == BPTT under CE.
    A one-hot final-step target + {0,1} final-step mask is used throughout."""
    print("── Tier 10: masked cross-entropy (seq-MNIST objective) ─────────────")
    import torch.nn.functional as F
    ce = mpn.masked_cross_entropy_loss_and_grad
    allok = True
    keys = grad_keys(True, True)

    # (a) analytic grad + loss value vs torch references.
    B, T, C = 6, 4, 5
    logits = torch.randn(B, T, C, dtype=torch.float64, requires_grad=True)
    lab, msk = _onehot_final_step(B, T, C, seed=1)
    loss, grad = ce(logits, lab, msk)
    loss.backward()
    g_rel = (logits.grad - grad).abs().max().item()
    ref_ce = F.cross_entropy(logits[:, T - 1, :].detach(), lab[:, T - 1, :].argmax(-1),
                             reduction='mean')
    l_gap = (loss.detach() - ref_ce).abs().item()
    ok_a = g_rel < 1e-10 and l_gap < 1e-10
    allok &= ok_a
    print(f"  [{'PASS' if ok_a else 'FAIL'}] grad==autograd, loss==F.CE   "
          f"grad_maxdiff={g_rel:.1e}  loss_gap={l_gap:.1e}")

    # (b) T=1: exact / diag / direct == BPTT under CE (n_out=3 classes).
    for name, rule in (("exact", "local_exact_rowlocal"),
                       ("diag", "local_diag_rflo"), ("direct", "local_direct")):
        net = build_net(5, 7, 3, 'tanh', True, True, 'scalar', 'scalar', rule, seed=1)
        inp, _, _, _ = make_data(6, 1, 5, 3)
        lab, msk = _onehot_final_step(6, 1, 3, seed=2)
        ref = net.bptt_gradients(inp, lab, msk, loss_and_grad=ce)
        loc = net.sequence_gradients(inp, lab, msk, loss_and_grad=ce)
        ok, d = compare(loc, ref, keys)
        lgap = (ref['loss'] - loc['loss']).abs().item()
        ok = ok and lgap < 1e-10
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] T=1 CE {name:6} == BPTT       "
              f"{fmt(d, keys)}  loss_gap={lgap:.1e}")

    # (c) eta=0, T>1: exact local == BPTT under CE.
    net = build_net(5, 7, 3, 'tanh', True, True, 'scalar', 'scalar',
                    'local_exact_rowlocal', seed=2)
    with torch.no_grad():
        net.mp_layer.eta.zero_()
    inp, _, _, _ = make_data(6, 10, 5, 3)
    lab, msk = _onehot_final_step(6, 10, 3, seed=3)
    ref = net.bptt_gradients(inp, lab, msk, loss_and_grad=ce)
    loc = net.sequence_gradients(inp, lab, msk, loss_and_grad=ce)
    ok_c, d_c = compare(loc, ref, keys)
    allok &= ok_c
    print(f"  [{'PASS' if ok_c else 'FAIL'}] eta=0 CE exact == BPTT       {fmt(d_c, keys)}")

    # (d) deep net, T=1: exact-local (incl. W_in) == BPTT under CE.
    torch.manual_seed(4)
    npar = {'n_neurons': [5, 7, 3], 'loss_type': 'MSE', 'activation': 'tanh',
            'output_bias': True, 'output_matrix': '', 'dt': 40,
            'input_layer_add': True, 'input_layer_add_trainable': True,
            'linear_embed': 6, 'input_layer_bias': True,
            'learning_rule': 'local_exact_rowlocal', 'feedback_mode': 'exact_spatial',
            'ml_params': {'bias': True, 'mp_type': 'mult', 'm_update_type': 'hebb_assoc',
                          'm_activation': 'linear', 'modulation_bounds': False,
                          'eta_type': 'scalar', 'eta_train': False, 'lam_type': 'scalar',
                          'lam_train': False, 'm_time_scale': 400, 'W_freeze': False}}
    net = mpn.DeepMultiPlasticNet(npar, verbose=False).double()
    with torch.no_grad():
        net.mp_layers[0].eta.fill_(0.12)
        net.mp_layers[0].lam.fill_(0.6 * net.mp_layers[0].lam_clamp)
    inp, _, _, _ = make_data(6, 1, 5, 3)
    lab, msk = _onehot_final_step(6, 1, 3, seed=5)
    ref = net.bptt_gradients(inp, lab, msk, loss_and_grad=ce)
    loc = net.sequence_gradients(inp, lab, msk, loss_and_grad=ce)
    dkeys = [k for k in ref if k not in ('loss', 'outputs')]
    ok_d, d_d = compare(loc, ref, dkeys)
    allok &= ok_d
    print(f"  [{'PASS' if ok_d else 'FAIL'}] dmpn T=1 CE == BPTT          {fmt(d_d, dkeys)}")
    return allok


def tier11_feedback_modes():
    """Deep-net feedback modes (mpn._FEEDBACK_MODES), the FA revision:
      exact_spatial — exact same-time spatial gradient (top layer exact vs BPTT).
      layerwise_fa  — fixed random matrix at EVERY boundary (weight-transport-free).
      direct_fa     — output error projected directly onto every hidden layer.
    Invariants checked:
      (a) the deleted modes ('random_top' / legacy 'random_fixed') now RAISE
          ValueError — the vocabulary is exactly _FEEDBACK_MODES.
      (b) readout grads (W_output, b_output) == BPTT for all three modes — the
          readout gradient depends only on grad_output and h[L], not feedback_mode.
          [deep stack: two MP layers.]
      (c) single MP layer (one boundary): layerwise_fa AND direct_fa, with their
          top random matrix set to W_output, reproduce exact_spatial EXACTLY (this
          is the ordinary-FA reduction — the only boundary is readout→hidden).
      (d) deep stack: layerwise_fa / direct_fa give finite grads for every param and
          DIFFER from exact_spatial on the hidden weights (they drop weight transport).
      (e) single MP layer, NO embedding: layerwise_fa == direct_fa given the same
          top matrix (one boundary → both are ordinary feedback alignment).
      (f) single MP layer + TRAINABLE embedding (a second boundary): with the shared
          top matrix forced equal, the MP-layer signals (W, b) + readout still
          coincide, but the embedding grads (W_in, b_in) DIFFER — layerwise_fa
          credits it by recursive feedback, direct_fa by a direct output projection.
    """
    print("── Tier 11: deep-net feedback modes (feedback alignment) ────────────")
    allok = True

    def build(nn, fb, seed=0, m_update='hebb_assoc', embed=False):
        torch.manual_seed(seed)
        npar = {'n_neurons': nn, 'loss_type': 'MSE', 'activation': 'tanh',
                'output_bias': True, 'output_matrix': '', 'dt': 40,
                'input_layer_add': embed, 'input_layer_add_trainable': embed,
                'linear_embed': 6, 'input_layer_bias': True, 'input_init_type': 'xavier',
                'learning_rule': 'local_exact_rowlocal', 'feedback_mode': fb,
                'ml_params': {'bias': True, 'mp_type': 'mult', 'm_update_type': m_update,
                              'm_activation': 'linear', 'modulation_bounds': False,
                              'eta_type': 'scalar', 'eta_train': False, 'lam_type': 'scalar',
                              'lam_train': False, 'm_time_scale': 400, 'W_freeze': False}}
        net = mpn.DeepMultiPlasticNet(npar, verbose=False).double()
        with torch.no_grad():
            for m in net.mp_layers:
                m.eta.fill_(0.1)
                m.lam.fill_(0.6 * m.lam_clamp)
        return net

    def sync_forward(src, dst):
        """Copy every trainable forward param src->dst so the two nets share
        identical weights. Needed because different feedback modes draw a
        different number of RNG values at init (the random modes draw extra
        feedback matrices), so a shared seed does NOT give shared weights. The
        fixed feedback buffers (B_feedback/B_inter/B_direct) are NOT trainable
        params, so they are left untouched."""
        with torch.no_grad():
            sp, dp = src._trainable_params(), dst._trainable_params()
            for k in dp:
                dp[k].copy_(sp[k])

    ARCH = [5, 7, 6, 3]                    # two MP layers (widths 7 then 6)
    inp, lab, msk, _ = make_data(6, 12, ARCH[0], ARCH[-1])
    rkeys = ['W_output', 'b_output']

    # (a) the deleted modes raise ValueError (vocabulary == _FEEDBACK_MODES).
    ok_a = True
    for dead in ('random_top', 'random_fixed'):
        try:
            build(ARCH, dead, seed=1)
            ok_a = False
        except ValueError:
            pass
    allok &= ok_a
    print(f"  [{'PASS' if ok_a else 'FAIL'}] deleted modes (random_top/random_fixed) raise ValueError")

    # (b) readout grads == BPTT for every surviving mode.
    ok_b = True
    for fb in ('exact_spatial', 'layerwise_fa', 'direct_fa'):
        net = build(ARCH, fb, seed=2)
        ref = net.bptt_gradients(inp, lab, msk)
        loc = net.local_gradients(inp, lab, msk)
        okm, dm = compare(loc, ref, rkeys)
        ok_b &= okm
        print(f"  [{'PASS' if okm else 'FAIL'}] {fb:<14} readout == BPTT   {fmt(dm, rkeys)}")
    allok &= ok_b

    # (c) single MP layer (one boundary): layerwise_fa / direct_fa with the top
    #     random matrix := W_output reproduce exact_spatial EXACTLY (ordinary FA).
    SARCH = [5, 7, 3]                       # one MP layer → one hidden boundary
    net_ex = build(SARCH, 'exact_spatial', seed=3)
    g_ex = net_ex.local_gradients(inp, lab, msk)
    keys_s = [k for k in g_ex if k not in ('loss', 'outputs')]
    ok_c = True
    for fb in ('layerwise_fa', 'direct_fa'):
        net = build(SARCH, fb, seed=3)
        sync_forward(net_ex, net)                        # identical forward weights
        top = net.B_feedback if fb == 'layerwise_fa' else getattr(net, net._B_direct_names[-1])
        with torch.no_grad():
            top.copy_(net.W_output)                      # kill the randomness at the only boundary
        okm, dm = compare(net.local_gradients(inp, lab, msk), g_ex, keys_s)
        ok_c &= okm
        print(f"  [{'PASS' if okm else 'FAIL'}] single-layer {fb:<12}(B:=W_out) == exact_spatial   {fmt(dm, ['W'])}")
    allok &= ok_c

    # (d) deep: layerwise_fa / direct_fa finite everywhere + differ from exact on hidden W.
    ok_d = True
    net_ex4 = build(ARCH, 'exact_spatial', seed=4)
    g_ex4 = net_ex4.local_gradients(inp, lab, msk)
    for fb in ('layerwise_fa', 'direct_fa'):
        net = build(ARCH, fb, seed=4)
        sync_forward(net_ex4, net)          # same forward weights → the ONLY difference is feedback
        g = net.local_gradients(inp, lab, msk)
        finite = all(torch.isfinite(g[k]).all() for k in g if k not in ('loss', 'outputs'))
        w_rel = (g['W'] - g_ex4['W']).abs().max().item() / max(g_ex4['W'].abs().max().item(), 1e-12)
        okm = finite and (w_rel > 1e-3)
        ok_d &= okm
        print(f"  [{'PASS' if okm else 'FAIL'}] {fb:<14} finite grads, hidden W differs   W:rel={w_rel:.2e} (>1e-3)")
    allok &= ok_d

    # (e) single MP layer: layerwise_fa == direct_fa given identical forward weights
    #     AND the same top random matrix (one boundary → both are ordinary FA).
    net_lw = build(SARCH, 'layerwise_fa', seed=5)
    B_shared = net_lw.B_feedback.detach().clone()
    grads_e = {}
    for fb in ('layerwise_fa', 'direct_fa'):
        net = build(SARCH, fb, seed=5)
        sync_forward(net_lw, net)           # identical forward weights
        top = net.B_feedback if fb == 'layerwise_fa' else getattr(net, net._B_direct_names[-1])
        with torch.no_grad():
            top.copy_(B_shared)             # identical top random matrix for both
        grads_e[fb] = net.local_gradients(inp, lab, msk)
    ok_e, _ = compare(grads_e['direct_fa'], grads_e['layerwise_fa'], keys_s)
    allok &= ok_e
    print(f"  [{'PASS' if ok_e else 'FAIL'}] single-layer: layerwise_fa == direct_fa")

    # (f) single MP layer + TRAINABLE input embedding = a SECOND trainable boundary,
    #     so layerwise_fa and direct_fa no longer fully coincide even at L=1. Forcing
    #     the shared MP-output (top) random matrix equal, the MP-layer signals (W, b)
    #     and readout still match, but the embedding grads (W_in, b_in) DIFFER:
    #     layerwise_fa credits the embedding by recursive feedback (through B_inter[0]),
    #     direct_fa by a direct output projection (B_direct[0]) — independent matrices.
    net_lwe = build(SARCH, 'layerwise_fa', seed=6, embed=True)
    B_top = net_lwe.B_feedback.detach().clone()
    grads_f = {}
    for fb in ('layerwise_fa', 'direct_fa'):
        net = build(SARCH, fb, seed=6, embed=True)
        sync_forward(net_lwe, net)          # identical forward weights (incl. embedding)
        top = net.B_feedback if fb == 'layerwise_fa' else getattr(net, net._B_direct_names[-1])
        with torch.no_grad():
            top.copy_(B_top)                # same TOP matrix → MP-layer signal identical
        grads_f[fb] = net.local_gradients(inp, lab, msk)
    mp_keys = ['W', 'b', 'W_output', 'b_output']       # should coincide
    ok_f_same, d_same = compare(grads_f['direct_fa'], grads_f['layerwise_fa'], mp_keys)
    win_rel = ((grads_f['layerwise_fa']['W_in'] - grads_f['direct_fa']['W_in']).abs().max().item()
               / max(grads_f['layerwise_fa']['W_in'].abs().max().item(), 1e-12))
    ok_f = ok_f_same and (win_rel > 1e-3)              # embedding grads should DIFFER
    allok &= ok_f
    print(f"  [{'PASS' if ok_f else 'FAIL'}] L=1 + embedding: MP signal coincides, "
          f"embedding differs   W:rel={d_same['W'][1]:.1e}  W_in:rel={win_rel:.2e} (>1e-3)")
    return allok


def tier13_input_mode():
    """input_mode decouples the trainable input embedding's learning rule from the
    MP-layer learning_rule (see mpn.DeepMultiPlasticNet._apply_input_mode).
    Invariants (deep dmpn: trainable embedding + one MP layer):
      (a) 'match' (default) is a NO-OP: grads byte-identical to a net built with no
          input_mode key, for every rule (bptt / diag / direct).
      (b) bptt + 'three_factor': W_in/b_in switch to the DIRECT local rule (== a
          native local run's W_in/b_in), while W/W_output stay the exact BPTT grad.
      (c) local_direct + 'exact': W_in/b_in switch to the true BPTT grad, while the
          MP-layer W stays the local grad (only the embedding is spliced).
      (d) the splice touches ONLY W_in/b_in — every other grad key is unchanged from
          the native run.
    """
    print("── Tier 13: input_mode (decoupled input-layer learning rule) ───────")
    allok = True

    def build(rule, input_mode, seed=0):
        import numpy as _np
        torch.manual_seed(seed); _np.random.seed(seed)
        npar = {'n_neurons': [5, 7, 3], 'loss_type': 'MSE', 'activation': 'tanh',
                'output_bias': True, 'output_matrix': '', 'dt': 40,
                'input_layer_add': True, 'input_layer_add_trainable': True,
                'linear_embed': 6, 'input_layer_bias': True, 'input_init_type': 'xavier',
                'learning_rule': rule, 'feedback_mode': 'exact_spatial',
                'input_mode': input_mode,
                'ml_params': {'bias': True, 'mp_type': 'mult', 'm_update_type': 'hebb_assoc',
                              'm_activation': 'linear', 'modulation_bounds': False,
                              'eta_type': 'scalar', 'eta_train': False, 'lam_type': 'scalar',
                              'lam_train': False, 'm_time_scale': 400, 'W_freeze': False}}
        net = mpn.DeepMultiPlasticNet(npar, verbose=False).double()
        with torch.no_grad():
            for m in net.mp_layers:
                m.eta.fill_(0.1); m.lam.fill_(0.6 * m.lam_clamp)
        return net

    inp, lab, msk, _ = make_data(4, 8, 5, 3)
    def grads(rule, im):
        return build(rule, im, seed=0).sequence_gradients(inp, lab, msk)

    # (a) 'match' == native (no-op) for every rule.
    ok_a = True
    for rule in ('bptt', 'local_diag_rflo', 'local_direct'):
        gm = grads(rule, 'match')
        gref = build(rule, 'match', seed=0)._grads_for_rule(rule, inp, lab, msk)
        ok, _ = compare(gm, gref, [k for k in gm if k not in ('loss', 'outputs')])
        ok_a &= ok
    allok &= ok_a
    print(f"  [{'PASS' if ok_a else 'FAIL'}] 'match' is a no-op (native) for bptt/diag/direct")

    keys_all = ['W', 'b', 'W_output', 'b_output', 'W_in', 'b_in']

    # (b) bptt + three_factor: W_in/b_in := local rule; W/W_output := exact BPTT.
    g_b3 = grads('bptt', 'three_factor')
    g_bptt = grads('bptt', 'match')
    g_localnat = grads('local_direct', 'match')      # native 3-factor embedding
    win_d = max((g_b3[k] - g_localnat[k]).abs().max().item() for k in ('W_in', 'b_in'))
    mp_d = max((g_b3[k] - g_bptt[k]).abs().max().item()
               for k in ('W', 'W_output', 'b', 'b_output'))
    ok_b = win_d < 1e-10 and mp_d < 1e-10
    allok &= ok_b
    print(f"  [{'PASS' if ok_b else 'FAIL'}] bptt+three_factor: W_in==local (d={win_d:.1e}), "
          f"MP/readout==BPTT (d={mp_d:.1e})")

    # (c) local_direct + exact: W_in/b_in := BPTT; MP W := local.
    g_le = grads('local_direct', 'exact')
    win_d2 = max((g_le[k] - g_bptt[k]).abs().max().item() for k in ('W_in', 'b_in'))
    mp_d2 = max((g_le[k] - g_localnat[k]).abs().max().item()
                for k in ('W', 'W_output', 'b', 'b_output'))
    ok_c = win_d2 < 1e-10 and mp_d2 < 1e-10
    allok &= ok_c
    print(f"  [{'PASS' if ok_c else 'FAIL'}] local_direct+exact: W_in==BPTT (d={win_d2:.1e}), "
          f"MP/readout==local (d={mp_d2:.1e})")

    # (d) splice touches ONLY W_in/b_in (vs the native run of the SAME rule).
    ok_d = True
    for k in ('W', 'b', 'W_output', 'b_output'):
        if (g_b3[k] - g_bptt[k]).abs().max().item() > 1e-12:
            ok_d = False
    allok &= ok_d
    print(f"  [{'PASS' if ok_d else 'FAIL'}] splice touches only W_in/b_in (other keys untouched)")
    return allok


def tier14_input_normalize():
    """Fixed input standardization u -> (u - loc)/scale (see
    mpn.MultiPlasticNetBase.set_input_norm_stats / _standardize_input), a data-
    conditioning knob applied IDENTICALLY to every rule (it is not rule-specific).
    Invariants:
      (a) OFF (the default): grads are BYTE-IDENTICAL to a net built with no flag,
          and no input_loc/input_scale buffers are allocated (state_dict unchanged).
      (b) set_input_norm_stats freezes correct per-feature statistics: the
          standardized sample is per-feature zero-mean / unit-std, and the buffers
          survive a state_dict round-trip (so a reloaded net normalizes identically).
      (c) The DEFINING equivalence: an ON net fed RAW inputs produces EXACTLY the
          same gradients (every rule × feedback mode) as an OFF net fed the pre-
          standardized inputs — proving the standardization is inserted once, in
          every path (BPTT / exact / diag / direct, all feedback modes, the trainable
          embedding's own input gradient), and conditions all rules equally.
    """
    print("── Tier 14: fixed input standardization (opt-in) ───────────────────")
    allok = True

    def build(nn, rule, fb='exact_spatial', norm=False, embed=True, seed=1):
        import numpy as _np
        torch.manual_seed(seed); _np.random.seed(seed)
        npar = {'n_neurons': nn, 'loss_type': 'MSE', 'activation': 'tanh',
                'output_bias': True, 'output_matrix': '', 'dt': 40,
                'input_layer_add': embed, 'input_layer_add_trainable': embed,
                'linear_embed': 6, 'input_layer_bias': True, 'input_init_type': 'xavier',
                'learning_rule': rule, 'feedback_mode': fb, 'input_mode': 'match',
                'input_normalize': norm,
                'ml_params': {'bias': True, 'mp_type': 'mult', 'm_update_type': 'hebb_assoc',
                              'm_activation': 'linear', 'modulation_bounds': False,
                              'eta_type': 'scalar', 'eta_train': False, 'lam_type': 'scalar',
                              'lam_train': False, 'm_time_scale': 400, 'W_freeze': False}}
        net = mpn.DeepMultiPlasticNet(npar, verbose=False).double()
        with torch.no_grad():
            for m in net.mp_layers:
                m.eta.fill_(0.1); m.lam.fill_(0.6 * m.lam_clamp)
        return net

    ARCH = [8, 12, 12, 4]                       # embedding + 2 MP layers → real depth
    # Off-scale, shifted inputs so standardization actually does something.
    inp, lab, msk, _ = make_data(6, 10, ARCH[0], ARCH[-1])
    inp = 5.0 * inp - 2.0

    # (a) OFF byte-identical + no extra buffers.
    a = build(ARCH, 'local_exact_rowlocal', norm=False)
    b = build(ARCH, 'local_exact_rowlocal', norm=False)
    keys = [k for k in a.sequence_gradients(inp, lab, msk) if k not in ('loss', 'outputs')]
    ga, gb = a.sequence_gradients(inp, lab, msk), b.sequence_gradients(inp, lab, msk)
    ok_a = (compare(ga, gb, keys)[0]
            and 'input_loc' not in a.state_dict() and 'input_scale' not in a.state_dict())
    allok &= ok_a
    print(f"  [{'PASS' if ok_a else 'FAIL'}] OFF: byte-identical + no input_loc/input_scale buffers")

    # (b) statistics correct (zero-mean / unit-std) + state_dict round-trip.
    on = build(ARCH, 'bptt', norm=True); on.set_input_norm_stats(inp)
    xs = on._standardize_input(inp).reshape(-1, ARCH[0])
    mean_err = xs.mean(0).abs().max().item()
    std_err = (xs.std(0) - 1.0).abs().max().item()
    rt = build(ARCH, 'bptt', norm=True); rt.set_input_norm_stats(torch.zeros_like(inp))
    rt.load_state_dict(on.state_dict())
    rt_err = (rt.input_loc - on.input_loc).abs().max().item() + (rt.input_scale - on.input_scale).abs().max().item()
    ok_b = mean_err < 1e-10 and std_err < 1e-10 and rt_err < 1e-12
    allok &= ok_b
    print(f"  [{'PASS' if ok_b else 'FAIL'}] stats: standardized mean={mean_err:.1e} "
          f"std-1={std_err:.1e}; state_dict round-trip err={rt_err:.1e}")

    # (c) ON(raw) == OFF(pre-standardized), every rule × feedback mode.
    xs_full = on._standardize_input(inp).clone()
    ok_c = True
    worst = 0.0
    for fb in ('exact_spatial', 'layerwise_fa', 'direct_fa'):
        for rule in ('bptt', 'local_exact_rowlocal', 'local_diag_rflo', 'local_direct'):
            n_on = build(ARCH, rule, fb, norm=True); n_on.set_input_norm_stats(inp)
            n_off = build(ARCH, rule, fb, norm=False)
            g_on = n_on.sequence_gradients(inp, lab, msk)
            g_off = n_off.sequence_gradients(xs_full, lab, msk)
            gkeys = [k for k in g_on if k not in ('loss', 'outputs')]
            same, d = compare(g_on, g_off, gkeys)
            worst = max(worst, max(d[k][0] for k in gkeys))
            ok_c &= same
    allok &= ok_c
    print(f"  [{'PASS' if ok_c else 'FAIL'}] ON(raw)==OFF(pre-standardized) for every "
          f"rule × feedback (max abs diff={worst:.1e})")
    return allok


def tier15_embed_partial_freeze():
    """Independently freezing the input embedding's weight vs bias (see
    mpn.DeepMultiPlasticNet._embed_grad_flags / _has_trainable_embed). The two
    tensors are gated PER TENSOR on requires_grad, so the local path must emit
    EXACTLY _trainable_params's keys — no missing key (was KeyError: 'b_in' when the
    weight was frozen but the bias trainable) and no silently-dropped computed grad
    (mirror case: trainable weight, frozen bias). Invariants, for every rule:
      (a) frozen weight + trainable bias: no crash; grads == _trainable_params keys
          (b_in present, W_in absent); b_in == the fully-trainable net's b_in.
      (b) trainable weight + frozen bias: grads == keys (W_in present, b_in absent);
          W_in == the fully-trainable net's W_in (freezing the bias doesn't perturb it).
    A frozen tensor's gradient must equal the full net's because reset weights are
    identical and a per-tensor requires_grad flag changes only WHICH grads are
    RETURNED, never the forward or the surviving grads' values.
    """
    print("── Tier 15: embedding partial freeze (weight/bias independent) ─────")
    allok = True
    RULES = ('bptt', 'local_exact_rowlocal', 'local_diag_rflo', 'local_direct')

    def build(rule):
        import numpy as _np
        torch.manual_seed(1); _np.random.seed(1)
        npar = {'n_neurons': [5, 7, 3], 'loss_type': 'MSE', 'activation': 'tanh',
                'output_bias': True, 'output_matrix': '', 'dt': 40,
                'input_layer_add': True, 'input_layer_add_trainable': True,
                'linear_embed': 6, 'input_layer_bias': True, 'input_init_type': 'xavier',
                'learning_rule': rule, 'feedback_mode': 'exact_spatial', 'input_mode': 'match',
                'ml_params': {'bias': True, 'mp_type': 'mult', 'm_update_type': 'hebb_assoc',
                              'm_activation': 'linear', 'modulation_bounds': False,
                              'eta_type': 'scalar', 'eta_train': False, 'lam_type': 'scalar',
                              'lam_train': False, 'm_time_scale': 400, 'W_freeze': False}}
        net = mpn.DeepMultiPlasticNet(npar, verbose=False).double()
        with torch.no_grad():
            for m in net.mp_layers:
                m.eta.fill_(0.1); m.lam.fill_(0.6 * m.lam_clamp)
        return net

    inp, lab, msk, _ = make_data(4, 8, 5, 3)

    # (a) frozen weight + trainable bias (the reported KeyError: 'b_in').
    ok_a = True
    for rule in RULES:
        full = build(rule); g_full = full.sequence_gradients(inp, lab, msk)
        net = build(rule); net.W_initial_linear.weight.requires_grad = False
        pk = sorted(net._trainable_params().keys())
        try:
            g = net.sequence_gradients(inp, lab, msk)
        except Exception as e:                       # the pre-fix failure mode
            ok_a = False
            print(f"    {rule}: RAISED {type(e).__name__}: {e}")
            continue
        gk = sorted(k for k in g if k not in ('loss', 'outputs'))
        bin_err = (g['b_in'] - g_full['b_in']).abs().max().item()
        ok_a &= (gk == pk) and ('W_in' not in gk) and ('b_in' in gk) and bin_err < 1e-12
    allok &= ok_a
    print(f"  [{'PASS' if ok_a else 'FAIL'}] frozen weight + trainable bias: no crash, "
          f"keys match, b_in == full net")

    # (b) trainable weight + frozen bias (mirror: computed b_in must be dropped, W_in kept).
    ok_b = True
    for rule in RULES:
        full = build(rule); g_full = full.sequence_gradients(inp, lab, msk)
        net = build(rule); net.W_initial_linear.bias.requires_grad = False
        pk = sorted(net._trainable_params().keys())
        g = net.sequence_gradients(inp, lab, msk)
        gk = sorted(k for k in g if k not in ('loss', 'outputs'))
        win_err = (g['W_in'] - g_full['W_in']).abs().max().item()
        ok_b &= (gk == pk) and ('b_in' not in gk) and ('W_in' in gk) and win_err < 1e-12
    allok &= ok_b
    print(f"  [{'PASS' if ok_b else 'FAIL'}] trainable weight + frozen bias: keys match, "
          f"W_in == full net (b_in dropped)")
    return allok


def tier16_mp_residual():
    """Identity skip connections around each equal-width MP block (see
    mpn.DeepMultiPlasticNet mp_residual / _residual_at). The skip is a same-time,
    parameter-free, memoryless op: h_{n+1}=act(z_n)+h_n where widths match. It stays
    fully local — the Hebbian write uses the block activation a_n (recovered as
    h_{n+1}-h_n), and the inter-layer signal gains the residual's identity Jacobian
    term (+ell_h[n+1]) — and leaves BPTT exact (autograd through the residual forward).
    Invariants (equal-width deep dmpn stack so skips are active everywhere):
      (a) OFF (default): grads BYTE-IDENTICAL to a net built with no mp_residual key,
          and no skip is registered (identity adds no params/buffers).
      (b) skip inserted only where d_in==d_out: an unequal-width layer has it disabled.
      (c) ON, eta=0 all layers: exact_rowlocal == BPTT (~1e-14) for EVERY param — the
          same-time backward (incl. the identity Jacobian term) is exact with no plastic
          temporal path.
      (d) ON, top-layer exact_rowlocal == BPTT and readout exact (the skip passes the
          top signal through unchanged; nothing plastic sits above the top block).
      (e) ON, freeze UPPER-layer eta: lower layer returns to EXACT vs BPTT (~1e-14),
          confirming the ONLY dropped term is the upper-plastic TEMPORAL path; with the
          upper eta active the lower layer is a surrogate (differs).
      (f) ON changes the network: grads differ from the non-residual net (not a no-op).
    """
    print("── Tier 16: identity skip connections (mp_residual) ────────────────")
    allok = True
    RULES = ('bptt', 'local_exact_rowlocal', 'local_diag_rflo', 'local_direct')

    def build(rule, arch, resid, etas=None, resid_key=True):
        import numpy as _np
        torch.manual_seed(1); _np.random.seed(1)
        npar = {'n_neurons': arch, 'loss_type': 'MSE', 'activation': 'tanh',
                'output_bias': True, 'output_matrix': '', 'dt': 40,
                'input_layer_add': True, 'input_layer_add_trainable': True,
                'linear_embed': arch[1], 'input_layer_bias': True, 'input_init_type': 'xavier',
                'learning_rule': rule, 'feedback_mode': 'exact_spatial', 'input_mode': 'match',
                'ml_params': {'bias': True, 'mp_type': 'mult', 'm_update_type': 'hebb_assoc',
                              'm_activation': 'linear', 'modulation_bounds': False,
                              'eta_type': 'scalar', 'eta_train': False, 'lam_type': 'scalar',
                              'lam_train': False, 'm_time_scale': 400, 'W_freeze': False}}
        if resid_key:
            npar['mp_residual'] = resid
        net = mpn.DeepMultiPlasticNet(npar, verbose=False).double()
        with torch.no_grad():
            for i, m in enumerate(net.mp_layers):
                m.eta.fill_(etas[i] if etas else 0.1); m.lam.fill_(0.6 * m.lam_clamp)
        return net

    ARCH = [6, 12, 12, 12, 4]         # embed(12) + 3 MP layers, all width 12 → skips everywhere
    inp, lab, msk, _ = make_data(4, 8, ARCH[0], ARCH[-1])
    keys = lambda g: [k for k in g if k not in ('loss', 'outputs')]

    # (a) OFF byte-identical + no skip registered.
    ok_a = True
    for rule in RULES:
        gA = build(rule, ARCH, False).sequence_gradients(inp, lab, msk)
        gB = build(rule, ARCH, False, resid_key=False).sequence_gradients(inp, lab, msk)
        ok_a &= compare(gA, gB, keys(gA))[0]
    ok_a &= (not any(build('bptt', ARCH, False)._residual_at))
    allok &= ok_a
    print(f"  [{'PASS' if ok_a else 'FAIL'}] OFF: byte-identical + no skip active")

    # (b) skip only where widths match.
    net_eq = build('bptt', ARCH, True)
    net_uneq = build('bptt', [6, 12, 10, 12, 4], True)   # middle layer 12->10 breaks equality
    ok_b = (all(net_eq._residual_at)
            and net_uneq._residual_at == [True, False, False])
    allok &= ok_b
    print(f"  [{'PASS' if ok_b else 'FAIL'}] skip active iff equal widths "
          f"(equal={net_eq._residual_at}, unequal={net_uneq._residual_at})")

    # (c) ON, eta=0 everywhere → exact == BPTT for all params.
    g_loc = build('local_exact_rowlocal', ARCH, True, etas=[0., 0., 0.]).sequence_gradients(inp, lab, msk)
    g_ref = build('bptt', ARCH, True, etas=[0., 0., 0.]).sequence_gradients(inp, lab, msk)
    d_c = max((g_loc[k] - g_ref[k]).abs().max().item() for k in keys(g_loc))
    ok_c = d_c < 1e-12
    allok &= ok_c
    print(f"  [{'PASS' if ok_c else 'FAIL'}] ON eta=0: exact_rowlocal == BPTT all params (d={d_c:.1e})")

    # (d) ON, top-layer exact + readout exact.
    g_loc = build('local_exact_rowlocal', ARCH, True).sequence_gradients(inp, lab, msk)
    g_ref = build('bptt', ARCH, True).sequence_gradients(inp, lab, msk)
    L = len(build('bptt', ARCH, True).mp_layers)
    topW = 'W' + ('' if L - 1 == 0 else str(L - 1))
    d_top = (g_loc[topW] - g_ref[topW]).abs().max().item()
    d_wout = (g_loc['W_output'] - g_ref['W_output']).abs().max().item()
    ok_d = d_top < 1e-12 and d_wout < 1e-12
    allok &= ok_d
    print(f"  [{'PASS' if ok_d else 'FAIL'}] ON: top MP layer {topW} == BPTT (d={d_top:.1e}), "
          f"readout exact (d={d_wout:.1e})")

    # (e) ON, freeze upper eta → lower exact; upper eta on → lower surrogate.
    g_fz = build('local_exact_rowlocal', ARCH, True, etas=[0.1, 0., 0.]).sequence_gradients(inp, lab, msk)
    gb_fz = build('bptt', ARCH, True, etas=[0.1, 0., 0.]).sequence_gradients(inp, lab, msk)
    d_lo_fz = (g_fz['W'] - gb_fz['W']).abs().max().item()
    g_on = build('local_exact_rowlocal', ARCH, True, etas=[0.1, 0.1, 0.1]).sequence_gradients(inp, lab, msk)
    gb_on = build('bptt', ARCH, True, etas=[0.1, 0.1, 0.1]).sequence_gradients(inp, lab, msk)
    rel_lo = ((g_on['W'] - gb_on['W']).abs().max()
              / gb_on['W'].abs().max().clamp_min(1e-12)).item()
    ok_e = d_lo_fz < 1e-12 and rel_lo > 1e-3
    allok &= ok_e
    print(f"  [{'PASS' if ok_e else 'FAIL'}] ON: freeze-upper-eta → lower exact (d={d_lo_fz:.1e}); "
          f"upper-eta-on → lower surrogate (rel={rel_lo:.1e})")

    # (f) ON is not a no-op vs the non-residual net.
    ok_f = True
    for rule in ('local_exact_rowlocal', 'local_diag_rflo', 'local_direct'):
        g_res = build(rule, ARCH, True).sequence_gradients(inp, lab, msk)
        g_no = build(rule, ARCH, False).sequence_gradients(inp, lab, msk)
        rel = ((g_res['W'] - g_no['W']).abs().max()
               / g_no['W'].abs().max().clamp_min(1e-12)).item()
        ok_f &= rel > 1e-3
    allok &= ok_f
    print(f"  [{'PASS' if ok_f else 'FAIL'}] ON changes grads vs non-residual net (not a no-op)")
    return allok


def tier17_cross_layer_steps():
    """Depth-1 cross-layer TEMPORAL correction (mpn.DeepMultiPlasticNet
    cross_layer_steps / _cross_layer_correction). The base local rules drop the
    temporal paths through the plastic state of UPPER layers; cross_layer_steps=1
    restores the leading (one-temporal-hop) term. Ground truth: at T=2 the dropped
    series has exactly ONE term, so the corrected local grad must equal BPTT there.
    Invariants (deep dmpn stacks, exact_spatial):
      (a) cross_layer_steps=0 is a NO-OP: grads byte-identical to a net built with the
          key absent, for every rule (deepcopy so weights are identical).
      (b) at T=1 the correction is inert: cross=1's explicit eligibility path gives
          grads byte-identical to cross=0's fused path (isolates the base rewrite).
      (c) THE headline: at T=2, cross=1 == BPTT (~1e-12) for every MP-layer WEIGHT AND
          BIAS (the correction credits grad_b too, via prev_R), across 2/3/4-layer
          stacks, scalar & matrix eta/lam, and residual-on.
      (d) at T>=3 the correction is not exact (depth-2+ tail remains) but STRICTLY
          IMPROVES the lower-layer weights AND biases vs cross=0.
      (e) freeze ALL upper-layer eta → the correction term vanishes (its coefficient
          carries the upper eta), so cross=1 == cross=0 exactly (the dropped paths
          are exactly the ones an upper eta=0 removes).
    SCOPE of the T=2 exactness claim (all satisfied here): exact_spatial feedback
    (the correction reads the true W, not the FA random matrices) and exact row-local
    intra-layer eligibility (diag/direct's own-M path is itself a surrogate at t=1, so
    cross=1 IMPROVES but is not T=2-exact for them). The trainable input embedding
    W_in keeps its same-time 3-factor rule (its cross-layer temporal term is not
    corrected), so full-gradient T=2 exactness incl. W_in needs input_mode='exact' or
    a frozen embedding; here the tier nets have no embedding, so it does not arise.
    """
    print("── Tier 17: cross-layer depth-1 temporal correction (cross_layer_steps) ──")
    import copy as _copy
    allok = True

    def build(arch, cross, matrix_eta=False, residual=False, seed=3, etas=None):
        torch.manual_seed(seed)
        npar = {'n_neurons': arch, 'loss_type': 'MSE', 'activation': 'tanh',
                'output_bias': True, 'output_matrix': '', 'dt': 40,
                'feedback_mode': 'exact_spatial', 'mp_residual': residual,
                'cross_layer_steps': cross, 'learning_rule': 'local_exact_rowlocal',
                'ml_params': {'bias': True, 'mp_type': 'mult', 'm_update_type': 'hebb_assoc',
                              'm_activation': 'linear', 'modulation_bounds': False,
                              'eta_type': 'matrix' if matrix_eta else 'scalar', 'eta_train': False,
                              'lam_type': 'matrix' if matrix_eta else 'scalar',
                              'lam_train': False, 'm_time_scale': 400, 'W_freeze': False}}
        net = mpn.DeepMultiPlasticNet(npar, verbose=False).double()
        with torch.no_grad():
            for j, m in enumerate(net.mp_layers):
                if matrix_eta:
                    m.eta.copy_(0.08 + 0.05 * torch.rand_like(m.eta))
                    m.lam.copy_(0.5 * m.lam_clamp + 0.3 * m.lam_clamp * torch.rand_like(m.lam))
                else:
                    m.eta.fill_(etas[j] if etas else 0.10 + 0.02 * j)
                    m.lam.fill_(0.6 * m.lam_clamp)
        return net

    def wk(net):
        # ALL MP-layer trainable tensors — weights AND biases — so the T=2 exactness
        # check covers grad_b too (the correction must credit the bias, not just W).
        L = len(net.mp_layers)
        ks = []
        for n in range(L):
            sfx = '' if n == 0 else str(n)
            ks += [f'W{sfx}', f'b{sfx}']
        return ks

    # (a) no-op at cross=0 (identical weights via deepcopy).
    net0 = build([5, 6, 7, 3], 0, matrix_eta=True, seed=7)
    netK = _copy.deepcopy(net0); netK.cross_layer_steps = 0
    inp, lab, msk, _ = make_data(3, 6, 5, 3)
    g0, gK = net0.local_gradients(inp, lab, msk), netK.local_gradients(inp, lab, msk)
    d_a = max((g0[k] - gK[k]).abs().max().item() for k in g0 if k not in ('loss', 'outputs'))
    ok_a = d_a < 1e-14
    allok &= ok_a
    print(f"  [{'PASS' if ok_a else 'FAIL'}] cross=0 no-op (d={d_a:.1e})")

    # (b) T=1 correction inert: explicit(cross=1) == fused(cross=0).
    netf = build([5, 6, 7, 3], 0, matrix_eta=True, seed=2)
    netc = _copy.deepcopy(netf); netc.cross_layer_steps = 1
    inp1, lab1, msk1, _ = make_data(3, 1, 5, 3)
    d_b = max((netf.local_gradients(inp1, lab1, msk1)[k]
               - netc.local_gradients(inp1, lab1, msk1)[k]).abs().max().item()
              for k in ('W', 'W1', 'W_output'))
    ok_b = d_b < 1e-14
    allok &= ok_b
    print(f"  [{'PASS' if ok_b else 'FAIL'}] T=1 explicit==fused base path (d={d_b:.1e})")

    # (c) T=2 == BPTT across configs.
    ok_c = True
    for name, arch, meta, resid in [("2L", [4, 5, 6, 3], False, False),
                                    ("3L", [4, 5, 6, 7, 3], False, False),
                                    ("3L-mat", [4, 5, 6, 7, 3], True, False),
                                    ("4L-mat", [4, 5, 6, 7, 8, 3], True, False),
                                    ("3L-res", [6, 6, 6, 6, 3], False, True)]:
        net = build(arch, 1, matrix_eta=meta, residual=resid, seed=3)
        inp, lab, msk, _ = make_data(3, 2, arch[0], arch[-1])
        ref, loc = net.bptt_gradients(inp, lab, msk), net.local_gradients(inp, lab, msk)
        worst = max((loc[k] - ref[k]).abs().max().item() / max(ref[k].abs().max().item(), 1e-30)
                    for k in wk(net))
        okm = worst < 1e-10
        ok_c &= okm
        print(f"  [{'PASS' if okm else 'FAIL'}] {name:7} T=2 cross=1 == BPTT (worst rel={worst:.1e})")
    allok &= ok_c

    # (d) T=3 improves over cross=0 on the lower weights AND biases.
    net_c = build([4, 5, 6, 7, 3], 1, matrix_eta=True, seed=3)
    net_0 = build([4, 5, 6, 7, 3], 0, matrix_eta=True, seed=3)
    inp, lab, msk, _ = make_data(3, 3, 4, 3)
    ref = net_c.bptt_gradients(inp, lab, msk)
    lower = ('W', 'W1', 'b', 'b1')     # lower-layer weights + biases
    rc = max((net_c.local_gradients(inp, lab, msk)[k] - ref[k]).abs().max().item()
             / max(ref[k].abs().max().item(), 1e-30) for k in lower)
    r0 = max((net_0.local_gradients(inp, lab, msk)[k] - ref[k]).abs().max().item()
             / max(ref[k].abs().max().item(), 1e-30) for k in lower)
    ok_d = rc < r0
    allok &= ok_d
    print(f"  [{'PASS' if ok_d else 'FAIL'}] T=3 cross improves lower W+b ({rc:.1e} < {r0:.1e})")

    # (e) freeze all upper eta → correction vanishes (cross=1 == cross=0), W and b.
    net_c = build([4, 5, 6, 7, 3], 1, seed=3, etas=[0.12, 0.0, 0.0])
    net_0 = _copy.deepcopy(net_c); net_0.cross_layer_steps = 0
    inp, lab, msk, _ = make_data(3, 4, 4, 3)
    d_e = max((net_c.local_gradients(inp, lab, msk)[k]
               - net_0.local_gradients(inp, lab, msk)[k]).abs().max().item()
              for k in ('W', 'W1', 'b', 'b1'))
    ok_e = d_e < 1e-12
    allok &= ok_e
    print(f"  [{'PASS' if ok_e else 'FAIL'}] freeze upper eta → correction vanishes (d={d_e:.1e})")
    return allok


def main():
    torch.set_default_dtype(torch.float64)
    print("Validating local-learning rules vs autograd (BPTT), float64\n")
    results = [
        tier1_exact_vs_bptt(),
        tier2_diag_matches_when_ninput1(),
        tier3_diag_matches_zeroed_reference(),
        extra_diag_differs_from_bptt(),
        tier5_direct_local(),
        tier6_deep_input_embedding(),
        tier7_rnn_rflo(),
        tier8_hebb_pre(),
        tier9_custom_loss(),
        tier10_cross_entropy(),
        tier11_feedback_modes(),
        tier13_input_mode(),
        tier14_input_normalize(),
        tier15_embed_partial_freeze(),
        tier16_mp_residual(),
        tier17_cross_layer_steps(),
        tier4_real_task(),
    ]
    print("\n" + ("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
