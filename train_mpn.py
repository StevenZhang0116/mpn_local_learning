#!/usr/bin/env python
# coding: utf-8
"""
Compare BPTT vs diagonal-RFLO local learning on a 1-layer MPN, over N runs.

For each of N_RUNS seeds we train TWO networks in lockstep:
  - 'bptt'            — autograd through the unrolled forward + M-update
  - 'local_diag_rflo' — diagonal / same-synapse RFLO local learning
Both start from the SAME initialization (deepcopy), see the SAME per-step data
and the SAME held-out validation set, and use their own Adam + scheduler. So the
only difference between the two curves within a seed is the learning rule; across
seeds we vary init + data. Each goes through MultiPlasticNet.sequence_gradients()
which writes .grad and lets optimizer.step() do the update.

Output: one figure (train_mpn_accuracy.png) with two panels — training accuracy
and testing (held-out) accuracy vs training step — each showing both rules as
mean ± std across the N runs. Accuracy is the library's angle accuracy, matching
MultiTaskMPN/one_task/one_task.py (train uses isvalid=False, valid isvalid=True).

Hyperparameters are aligned with one_task.py: hidden=200, lr=1e-3, batch=128,
n_datasets=3000, gradient_clip=10, Adam, ReduceLROnPlateau, output_bias=False,
tanh. Differences required for the local rule: net_type='mpn1' (single MP layer,
no input embedding) and regularization OFF (reg_lambda=0, pure masked-MSE).

Correctness of the rules (vs BPTT) is proven separately in
validate_local_learning.py; this script is about learning performance.

Run from this directory:  python train_mpn.py
NOTE: N_RUNS x 2 rules x N_DATASETS steps is heavy on CPU — intended for a GPU
box (device auto-selects cuda). Reduce N_RUNS / N_DATASETS for a quick look.
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
import mpn

# ─── Configuration (aligned with MultiTaskMPN/one_task/one_task.py) ───────────
SEED = 42
RULESET = "delaygo"           # single task to train on
RULES_TO_RUN = ["bptt", "local_diag_rflo", "local_direct"]   # the two rules to compare
FEEDBACK_MODE = "exact_readout"   # 'exact_readout' or 'random_fixed' (feedback align)
N_RUNS = 5                    # independent seeds per rule
N_HIDDEN = 200                # one_task.py: n_hidden = 200
N_DATASETS = 5000             # one_task.py: n_datasets = 3000 (heavy on CPU)
BATCH = 128                   # one_task.py: n_batches = batch_size = 128
LR = 1e-3                     # one_task.py: lr = 1e-3
GRAD_CLIP = 10                # one_task.py: gradient_clip = 10
LOG_EVERY = 100               # record/print accuracy every this many steps
FIG_DIR = "figure"            # subfolder for saved figures
CKPT_DIR = "checkpoints"      # subfolder for saved trained networks
SAVE_NETS = True              # save each trained network (per rule, per seed)


def _param_tag():
    """Shared parameter string identifying a run configuration."""
    return (f"{RULESET}_h{N_HIDDEN}_b{BATCH}_n{N_DATASETS}"
            f"_lr{LR:.0e}_{FEEDBACK_MODE}")


def fig_path():
    """Figure filename encoding the key params so runs don't overwrite each
    other: task, hidden size, batch, #steps, lr, #runs, feedback mode."""
    return os.path.join(FIG_DIR, f"train_mpn_{_param_tag()}_runs{N_RUNS}.png")


def ckpt_path(rule, seed):
    """Checkpoint filename for one trained network: run params + rule + seed, so
    every (rule, seed) is a distinct file that can be reloaded later."""
    return os.path.join(CKPT_DIR, f"mpn_{_param_tag()}_{rule}_seed{seed}.pt")

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.float32         # float32 for speed (one_task.py also runs float32)

RULE_LABEL = {"bptt": "BPTT", "local_diag_rflo": "diagonal RFLO",
              "local_exact_rowlocal": "exact row-local", "local_direct": "direct"}
RULE_COLOR = {"bptt": "#1f77b4", "local_diag_rflo": "#d62728",
              "local_exact_rowlocal": "#2ca02c", "local_direct": "#9467bd"}


def build_params():
    """(task, train, net) param dicts for a single-MP-layer MultiPlasticNet on
    one task, in the local-rule derivation config. net_type='mpn1' (no input
    embedding); regularization disabled so the objective is pure masked-MSE."""
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
        "scheduler": {               # one_task.py: ReduceLROnPlateau
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.95,
            "patience": 30,
            "min_lr": 1e-8,
        },
    }

    net_params = {
        "net_type": "mpn1",
        "n_neurons": [1, N_HIDDEN, 1],   # [in, hidden, out]; in/out overwritten below
        "output_bias": False,            # one_task.py: output_bias = False
        "loss_type": "MSE",
        "activation": "tanh",
        "cuda": False,
        "monitor_freq": train_params["n_epochs_per_set"],
        "monitor_valid_out": False,
        "output_matrix": "",
        "acc_measure": "angle",
        "learning_rule": "bptt",         # overwritten per rule below
        "feedback_mode": FEEDBACK_MODE,
        "ml_params": {
            "bias": True,
            "mp_type": "mult",
            "m_update_type": "hebb_assoc",
            "m_activation": "linear",      # required for the local rules
            "modulation_bounds": False,    # required for the local rules
            "eta_type": "scalar",
            "eta_train": False,
            "lam_type": "scalar",
            "m_time_scale": 400,
            "lam_train": False,
            "W_freeze": False,
        },
    }
    return task_params, train_params, net_params


def try_accuracy(net, output, labels, mask, inputs, isvalid=False):
    """Best-effort angle accuracy via the library; returns nan on failure.
    isvalid mirrors one_task.py: train uses isvalid=False (current batch),
    validation uses isvalid=True (task-aligned held-out evaluation)."""
    try:
        acc, _ = net.compute_acc(output.float(), labels.float(), mask.float(),
                                 inputs.float(), mode=net.acc_measure, isvalid=isvalid)
        return float(acc)
    except Exception:
        return float("nan")


def load_net(path, device=None, dtype=DTYPE):
    """Reload a network saved by run_seed. Returns the reconstructed
    MultiPlasticNet with its trained weights and learning_rule restored.
    Example:  net = load_net(ckpt_path('bptt', 42))"""
    device = device or DEVICE
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = mpn.MultiPlasticNet(ckpt["net_params"], verbose=False).to(device).to(dtype)
    net.load_state_dict(ckpt["state_dict"])
    net.learning_rule = ckpt.get("learning_rule", net.learning_rule)
    return net


@torch.no_grad()
def forward_outputs(net, inputs):
    """Unrolled forward over time (no grad), returning outputs (B, T, n_output).
    Used to evaluate the held-out validation set; MultiPlasticNet.network_step
    returns 2 values so the library's iterate_sequence_batch can't drive it."""
    B, T, _ = inputs.shape
    net.reset_state(B=B)
    outs = []
    for t in range(T):
        x_t = inputs[:, t, :]
        hidden_pre, _ = net.mp_layer(x_t)
        hidden = net.act_fn(hidden_pre)
        out = torch.einsum('iI,BI->Bi', net.W_output, hidden) + net.b_output.unsqueeze(0)
        outs.append(out)
        net.mp_layer.update_M_matrix(x_t, hidden)
    return torch.stack(outs, dim=1)


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
    base = mpn.MultiPlasticNet(net_params, verbose=(seed == SEED)).to(DEVICE).to(DTYPE)
    nets, optims = {}, {}
    for rule in RULES_TO_RUN:
        net = copy.deepcopy(base)
        net.learning_rule = rule
        nets[rule] = net
        optims[rule] = make_optim(net)

    # Held-out validation set, generated ONCE and shared across rules (as in
    # one_task.py). Drawn after seeding so it is reproducible per seed.
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

            # Held-out validation loss on the UPDATED net drives the scheduler
            # (as in one_task.py) — smoother than the fresh per-batch train loss.
            v_out = forward_outputs(net, v_inputs)
            v_loss, _ = mpn.masked_mse_loss_and_output_grad(v_out, v_labels, v_mask)
            sch.step(v_loss.item())

            if step in record_set:
                train_acc = try_accuracy(net, grads["outputs"], labels, mask,
                                         inputs, isvalid=False)
                valid_acc = try_accuracy(net, v_out, v_labels, v_mask, v_inputs,
                                         isvalid=True)
                curves[rule]["train"].append(train_acc)
                curves[rule]["valid"].append(valid_acc)

        if step in record_set:
            msg = "  ".join(
                f"{RULE_LABEL[r]}: tr={curves[r]['train'][-1]:.3f} "
                f"va={curves[r]['valid'][-1]:.3f}" for r in RULES_TO_RUN)
            print(f"  seed {seed} step {step:>5}   {msg}")

    # Save each trained network (per rule) so it can be reloaded later. Stores
    # state_dict + net_params (as in one_task.py) plus rule/seed metadata.
    if SAVE_NETS:
        os.makedirs(CKPT_DIR, exist_ok=True)
        for rule in RULES_TO_RUN:
            path = ckpt_path(rule, seed)
            torch.save({
                "state_dict": nets[rule].state_dict(),
                "net_params": net_params,
                "learning_rule": rule,
                "feedback_mode": FEEDBACK_MODE,
                "seed": seed,
            }, path)
            print(f"  saved network: {path}")

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
    fig.suptitle(f"{RULESET}: BPTT vs diagonal RFLO  "
                 f"(mean ± std over {N_RUNS} runs, hidden={N_HIDDEN})")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = fig_path()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {path}")


def main():
    print(f"Task: {RULESET}  |  rules: {RULES_TO_RUN}  |  runs: {N_RUNS}  |  "
          f"hidden={N_HIDDEN} batch={BATCH} steps={N_DATASETS} lr={LR} clip={GRAD_CLIP}")
    print(f"Device: {DEVICE}  dtype: {DTYPE}  feedback: {FEEDBACK_MODE}\n")

    record_steps = list(range(0, N_DATASETS, LOG_EVERY))
    if record_steps[-1] != N_DATASETS - 1:
        record_steps.append(N_DATASETS - 1)

    # runs[rule][split] -> list (over seeds) of accuracy curves.
    runs = {r: {"train": [], "valid": []} for r in RULES_TO_RUN}
    seeds = [SEED + k for k in range(N_RUNS)]
    for run_idx, seed in enumerate(seeds):
        print(f"── Run {run_idx + 1}/{N_RUNS}  (seed {seed}) ──")
        curves = run_seed(seed, record_steps)
        for r in RULES_TO_RUN:
            runs[r]["train"].append(curves[r]["train"])
            runs[r]["valid"].append(curves[r]["valid"])

    # Aggregate mean/std across seeds (ignoring any nan accuracies).
    agg = {}
    for r in RULES_TO_RUN:
        agg[r] = {}
        for split in ("train", "valid"):
            arr = np.asarray(runs[r][split], dtype=float)   # (N_RUNS, n_points)
            agg[r][split] = {"mean": np.nanmean(arr, axis=0),
                             "std": np.nanstd(arr, axis=0)}

    plot(record_steps, agg)

    # Final-accuracy summary.
    print("\nFinal accuracy (mean ± std over runs):")
    for r in RULES_TO_RUN:
        tr = agg[r]["train"]; va = agg[r]["valid"]
        print(f"  {RULE_LABEL.get(r, r):<16} train {tr['mean'][-1]:.3f} ± {tr['std'][-1]:.3f}"
              f"   test {va['mean'][-1]:.3f} ± {va['std'][-1]:.3f}")


if __name__ == "__main__":
    main()
