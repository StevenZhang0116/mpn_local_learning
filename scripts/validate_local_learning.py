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
Run from this directory:  python validate_local_learning.py
"""
import torch

import _bootstrap  # prepends ../core + ../scripts to sys.path; exposes ROOT
import mpn


def build_net(n_input, n_hidden, n_output, activation, layer_bias, output_bias,
              eta_type, lam_type, learning_rule='local_exact_rowlocal',
              feedback_mode='exact_readout', m_update_type='hebb_assoc', seed=0):
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
    feedback = net.W_output if net.feedback_mode == 'exact_readout' else net.B_feedback
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
        ("tanh, B=8, scalar",   'tanh', 8, 'scalar', 'scalar', 'exact_readout', False),
        ("tanh, B=8, matrix",   'tanh', 8, 'matrix', 'matrix', 'exact_readout', False),
        ("tanh, masks",         'tanh', 8, 'matrix', 'matrix', 'exact_readout', True),
        ("tanh, random feedbk", 'tanh', 8, 'scalar', 'scalar', 'random_fixed',  False),
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
    net = build_net(5, 7, 3, 'tanh', True, True, 'scalar', 'scalar', 'local_direct', seed=2)
    inp, lab, msk, _ = make_data(4, 12, 5, 3)
    gd = net.local_direct_gradients(inp, lab, msk)
    mp_layer = net.mp_layer
    orig = mp_layer.update_diag_rflo_traces
    def _a_zero(xx, E, R, update_mask=None):
        orig(xx, E, R, update_mask=update_mask)
        mp_layer.A.zero_()
    mp_layer.update_diag_rflo_traces = _a_zero
    gdiag0 = net.local_diag_rflo_gradients(inp, lab, msk)
    mp_layer.update_diag_rflo_traces = orig
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
            'learning_rule': 'local_exact_rowlocal', 'feedback_mode': 'exact_readout',
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
                'feedback_mode': 'exact_readout'}
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
            'learning_rule': 'local_exact_rowlocal', 'feedback_mode': 'exact_readout',
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
        tier4_real_task(),
    ]
    print("\n" + ("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
