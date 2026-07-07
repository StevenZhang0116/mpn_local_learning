#!/usr/bin/env python
# coding: utf-8
"""
Compare BPTT vs RFLO local learning on a leaky vanilla RNN, over N runs.

The RNN analog of train_mpn.py. For each of N_RUNS seeds we train TWO networks
in lockstep:
  - 'bptt'            — autograd through the unrolled recurrent forward
  - 'local_diag_rflo' — RFLO (Murray & Escola 2019): forward-mode eligibility
                        traces that drop the recurrent sensitivity term. Shares
                        the API key with the MPN diagonal rule (see core/rnn.py).
Both start from the SAME initialization (deepcopy), see the SAME per-step data
and the SAME held-out validation set, and use their own Adam + scheduler. So the
only difference between the two curves within a seed is the learning rule; across
seeds we vary init + data.

Output: one figure (figure/train_rnn_accuracy.png) with two panels — training
and testing (held-out) accuracy vs step — each rule mean ± std over the N runs.
Accuracy is the library's angle accuracy (train isvalid=False, valid isvalid=True).

Hyperparameters mirror train_mpn.py / one_task.py: hidden=200, lr=1e-3, batch=128,
n_datasets=3000, gradient_clip=10, Adam, ReduceLROnPlateau, tanh, leaky alpha
from convert_and_init_multitask_params. output_bias=False; no regularization.

Run from this directory:  python train_rnn.py
NOTE: heavy on CPU — intended for a GPU box (device auto-selects cuda). Reduce
N_RUNS / N_DATASETS for a quick look.
"""
import copy
import gc
import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless: write PNG, no display
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401  -- prepends ./core to sys.path
import mpn_tasks
import rnn

# ─── Configuration (mirrors train_mpn.py) ─────────────────────────────────────
SEED = 42
RULESET = "delaygo"           # single task to train on
RULES_TO_RUN = ["bptt", "local_diag_rflo"]   # the two rules to compare
FEEDBACK_MODE = "exact_readout"   # 'exact_readout' or 'random_fixed' (feedback align)
N_RUNS = 3                    # independent seeds per rule
N_HIDDEN = 200                # one_task.py: n_hidden = 200
N_DATASETS = 3000             # one_task.py: n_datasets = 3000 (heavy on CPU)
BATCH = 128                   # one_task.py: n_batches = batch_size = 128
LR = 1e-3                     # one_task.py: lr = 1e-3
GRAD_CLIP = 10                # one_task.py: gradient_clip = 10
LOG_EVERY = 100               # record/print accuracy every this many steps
FIG_DIR = "figure"            # subfolder for saved figures


def fig_path():
    """Figure filename encoding the key params so runs don't overwrite each
    other: task, hidden size, batch, #steps, lr, #runs, feedback mode."""
    return os.path.join(
        FIG_DIR,
        f"train_rnn_{RULESET}_h{N_HIDDEN}_b{BATCH}_n{N_DATASETS}"
        f"_lr{LR:.0e}_runs{N_RUNS}_{FEEDBACK_MODE}.png")

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.float32         # float32 for speed (one_task.py also runs float32)

RULE_LABEL = {"bptt": "BPTT", "local_diag_rflo": "RFLO"}
RULE_COLOR = {"bptt": "#1f77b4", "local_diag_rflo": "#d62728"}


def build_params():
    """(task, train, net) param dicts for a leaky vanilla RNN on one task.
    Regularization disabled so the objective is pure masked-MSE (what the RFLO
    rule is derived for). net_type='vanilla' selects the RNN in the library
    param converter; the actual class used here is rnn.LeakyRNN."""
    task_params = {
        "task_type": "multitask",
        "rules": [RULESET],
        "dt": 40,
        "ruleset": RULESET,
        "n_eachring": 8,
        "in_out_mode": "low_dim",
        "sigma_x": 0.00,
        "mask_type": "cost",
        "fixate_off": False,
        "task_info": True,
        "randomize_inputs": False,
        "n_input": 20,
        "modality_diff": False,
        "label_strength": False,
        "long_fixation": "normal",
        "long_stimulus": "normal",
        "long_delay": "normal",
        "long_response": "normal",
        "adjust_task_prop": True,
        "adjust_task_decay": 0.9,
    }

    train_params = {
        "lr": LR,
        "n_batches": BATCH,
        "batch_size": BATCH,
        "gradient_clip": GRAD_CLIP,
        "valid_n_batch": BATCH,
        "n_datasets": N_DATASETS,
        "valid_check": None,
        "n_epochs_per_set": 1,
        "task_mask": None,
        "weight_reg": "L2",
        "activity_reg": "L2",
        "reg_lambda": 0.0,           # no regularization → objective = masked MSE
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.95,
            "patience": 30,
            "min_lr": 1e-8,
        },
    }

    net_params = {
        "net_type": "vanilla",
        "n_neurons": [1, N_HIDDEN, 1],   # [in, hidden, out]; in/out overwritten below
        "output_bias": False,            # one_task.py: output_bias = False
        "hidden_bias": True,
        "loss_type": "MSE",
        "activation": "tanh",
        "cuda": False,
        "monitor_freq": train_params["n_epochs_per_set"],
        "monitor_valid_out": False,
        "output_matrix": "",
        "acc_measure": "angle",
        "leaky": True,
        "alpha": 0.8,                    # overwritten by convert_and_init to 1 - dt/tau
        "learning_rule": "bptt",         # overwritten per rule below
        "feedback_mode": FEEDBACK_MODE,
    }
    return task_params, train_params, net_params


def try_accuracy(net, output, labels, mask, inputs, isvalid=False):
    """Best-effort angle accuracy via the library; returns nan on failure.
    train uses isvalid=False (current batch), validation isvalid=True."""
    try:
        acc, _ = net.compute_acc(output.float(), labels.float(), mask.float(),
                                 inputs.float(), mode=net.acc_measure, isvalid=isvalid)
        return float(acc)
    except Exception:
        return float("nan")


def make_optim(net):
    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.95, patience=30, min_lr=1e-8)
    return trainable, opt, sch


def run_seed(seed, record_steps):
    """Train all RULES_TO_RUN in lockstep for one seed: identical init, identical
    per-step data, identical held-out validation set. Returns
    {rule: {'train': [...], 'valid': [...]}} sampled at record_steps."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    task_params, train_params, net_params = build_params()
    task_params, train_params, net_params = mpn_tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params)
    )
    net_params["prefs"] = mpn_tasks.get_prefs(task_params["hp"])

    # One base net → deepcopy so every rule starts from the SAME weights.
    base = rnn.LeakyRNN(net_params, verbose=(seed == SEED)).to(DEVICE).to(DTYPE)
    nets, optims = {}, {}
    for rule in RULES_TO_RUN:
        net = copy.deepcopy(base)
        net.learning_rule = rule
        nets[rule] = net
        optims[rule] = make_optim(net)

    # Held-out validation set, generated ONCE and shared across rules.
    vdata, _ = mpn_tasks.generate_trials_wrap(
        task_params, train_params["valid_n_batch"], rules=task_params["rules"],
        mode_input="random_batch", device=DEVICE,
    )
    v_inputs, v_labels, v_mask = (d.to(DTYPE) for d in vdata)

    curves = {r: {"train": [], "valid": []} for r in RULES_TO_RUN}
    record_set = set(record_steps)

    for step in range(N_DATASETS):
        # One batch, generated once and fed identically to every rule.
        data, _ = mpn_tasks.generate_trials_wrap(
            task_params, BATCH, rules=task_params["rules"],
            mode_input="random_batch", device=DEVICE,
        )
        inputs, labels, mask = (d.to(DTYPE) for d in data)

        for rule in RULES_TO_RUN:
            net = nets[rule]
            trainable, opt, sch = optims[rule]
            opt.zero_grad()
            grads = net.sequence_gradients(inputs, labels, mask)
            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            opt.step()
            if net.param_clamping:
                net.param_clamp()
            sch.step(grads["loss"].item())

            if step in record_set:
                train_acc = try_accuracy(net, grads["outputs"], labels, mask,
                                         inputs, isvalid=False)
                v_out = net.forward_outputs(v_inputs)
                valid_acc = try_accuracy(net, v_out, v_labels, v_mask, v_inputs,
                                         isvalid=True)
                curves[rule]["train"].append(train_acc)
                curves[rule]["valid"].append(valid_acc)

        if step in record_set:
            msg = "  ".join(
                f"{RULE_LABEL[r]}: tr={curves[r]['train'][-1]:.3f} "
                f"va={curves[r]['valid'][-1]:.3f}" for r in RULES_TO_RUN)
            print(f"  seed {seed} step {step:>5}   {msg}")

    del nets, optims, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return curves


def plot(record_steps, agg):
    """One figure, two panels (train / test accuracy), each rule mean ± std."""
    steps = np.asarray(record_steps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, split, title in zip(axes, ("train", "valid"),
                                ("Training accuracy", "Testing (held-out) accuracy")):
        for rule in RULES_TO_RUN:
            # Accuracy is stored as a fraction in [0, 1]; plot as percent so a
            # fully-solved task (100%) sits below the 110 upper bound.
            mean = 100.0 * agg[rule][split]["mean"]
            std = 100.0 * agg[rule][split]["std"]
            color = RULE_COLOR.get(rule, None)
            ax.plot(steps, mean, color=color, label=RULE_LABEL.get(rule, rule), lw=2)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 110)
    axes[0].set_ylabel("angle accuracy (%)")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle(f"{RULESET} (leaky RNN): BPTT vs RFLO  "
                 f"(mean ± std over {N_RUNS} runs, hidden={N_HIDDEN})")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = fig_path()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {path}")


def main():
    print(f"Task: {RULESET} (leaky RNN)  |  rules: {RULES_TO_RUN}  |  runs: {N_RUNS}  |  "
          f"hidden={N_HIDDEN} batch={BATCH} steps={N_DATASETS} lr={LR} clip={GRAD_CLIP}")
    print(f"Device: {DEVICE}  dtype: {DTYPE}  feedback: {FEEDBACK_MODE}\n")

    record_steps = list(range(0, N_DATASETS, LOG_EVERY))
    if record_steps[-1] != N_DATASETS - 1:
        record_steps.append(N_DATASETS - 1)

    runs = {r: {"train": [], "valid": []} for r in RULES_TO_RUN}
    seeds = [SEED + k for k in range(N_RUNS)]
    for run_idx, seed in enumerate(seeds):
        print(f"── Run {run_idx + 1}/{N_RUNS}  (seed {seed}) ──")
        curves = run_seed(seed, record_steps)
        for r in RULES_TO_RUN:
            runs[r]["train"].append(curves[r]["train"])
            runs[r]["valid"].append(curves[r]["valid"])

    agg = {}
    for r in RULES_TO_RUN:
        agg[r] = {}
        for split in ("train", "valid"):
            arr = np.asarray(runs[r][split], dtype=float)   # (N_RUNS, n_points)
            agg[r][split] = {"mean": np.nanmean(arr, axis=0),
                             "std": np.nanstd(arr, axis=0)}

    plot(record_steps, agg)

    print("\nFinal accuracy (mean ± std over runs):")
    for r in RULES_TO_RUN:
        tr = agg[r]["train"]; va = agg[r]["valid"]
        print(f"  {RULE_LABEL.get(r, r):<16} train {tr['mean'][-1]:.3f} ± {tr['std'][-1]:.3f}"
              f"   test {va['mean'][-1]:.3f} ± {va['std'][-1]:.3f}")


if __name__ == "__main__":
    main()
