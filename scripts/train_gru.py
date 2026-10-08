#!/usr/bin/env python
# coding: utf-8
"""
Compare BPTT vs RFLO-style local learning on a GRU, over N runs.

The GRU analog of train_rnn.py / train_mpn.py. For each of N_RUNS seeds we train
RULES_TO_RUN in lockstep:
  - 'bptt'            — autograd through the unrolled GRU recurrence
  - 'local_diag_rflo' — GRU-RFLO (see core/gru.py): forward-mode per-unit
                        eligibility traces gated by each unit's own update gate,
                        dropping the recurrent sensitivity that enters the gates
                        through W_rec. Exact when W_rec = 0 or T = 1; otherwise an
                        approximation to BPTT. Readout gradients stay exact.
Same lockstep protocol as the other scripts (identical init / data / valid set
per seed), same figure + .npz output. The shared machinery lives in
train_common.py; this file only sets the GRU config and hooks.

Hyperparameters mirror train_rnn.py. A GRU has three gate blocks, so N_HIDDEN
defaults to 100 (about the same number of recurrent parameters as the 180-unit
leaky RNN); pass --hidden to change it. net_type='gru'; class = gru.GRU.

Under 'local_diag_rflo' the whole recurrent block (input, recurrent, biases)
trains by the traces; under 'bptt' everything trains by exact autograd.

Run from this directory:
    python train_gru.py                      # BPTT vs RFLO (default)
    python train_gru.py --hidden 64 --runs 3 --task delaygo
"""
import argparse
import torch
import numpy as np 

import _bootstrap  # prepends ../core + ../scripts to sys.path; exposes ROOT
import gru
import tasks
import train_common as tc

# ─── Configuration (mirrors train_mpn.py) ─────────────────────────────────────
SEED = np.random.randint(0, 1000)  # random seed for this run (small → short run names)
RULESET = "seqmnist_pixel"           # single task to train on
RULES_TO_RUN = ["bptt", "local_diag_rflo"]   # rules to compare
# 'exact_spatial' = true gradient; the random modes ('layerwise_fa', 'direct_fa')
# both coincide with ordinary feedback alignment for this single-hidden-layer RNN.
# See mpn._FEEDBACK_MODES. (Legacy 'exact_readout' still aliases to 'exact_spatial'.)
FEEDBACK_MODE = "exact_spatial"
# The RNN has no local readout heads, so only the three GLOBAL signal modes of
# train_mpn.SIGNAL_MODES apply; --learning-signal uses the same names as train_mpn.
RNN_SIGNAL_MODES = {"exact_spatial": "exact_spatial", "layerwise_fa": "layerwise_fa",
                    "dfa": "direct_fa"}
_FEEDBACK_TO_SIGNAL = {"exact_spatial": "exact_spatial", "exact_readout": "exact_spatial",
                       "layerwise_fa": "layerwise_fa", "direct_fa": "dfa"}
N_RUNS = 3                    # independent seeds per rule
N_HIDDEN = 100                # 3 gate blocks: ~the leaky RNN (180) recurrent parameter count
N_DATASETS = 5000             # one_task.py: n_datasets = 3000 (heavy on CPU)
BATCH = 128                   # one_task.py: n_batches = batch_size = 128
LR = 1e-3                     # one_task.py: lr = 1e-3
GRAD_CLIP = 10                # one_task.py: gradient_clip = 10
LOG_EVERY = 100               # record/print accuracy every this many steps
FIG_DIR = str(_bootstrap.ROOT / "figure")        # saved figures (anchored to root)
CKPT_DIR = str(_bootstrap.ROOT / "checkpoints")  # saved trained networks
DATA_DIR = str(_bootstrap.ROOT / "figure_data")  # saved plot data (.npz)
SAVE_NETS = False             # RNN comparison: figures only, no checkpoints by default

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.float32         # float32 for speed (one_task.py also runs float32)

RULE_LABEL = {"bptt": "BPTT", "local_diag_rflo": "RFLO"}
RULE_COLOR = {"bptt": "#1f77b4", "local_diag_rflo": "#d62728"}


def build_params():
    """(task, train, net) param dicts for a GRU on one task. Regularization
    disabled so the objective is pure masked-MSE (what the traces are derived
    for). net_type='gru'; class = gru.GRU."""
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
        "valid_n_batch": BATCH * 3,
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
        "net_type": "gru",
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
        "learning_rule": "bptt",         # overwritten per rule below
        "feedback_mode": FEEDBACK_MODE,
    }
    return task_params, train_params, net_params


def _cfg():
    """Package the current module globals + GRU-specific hooks into a RunConfig.
    Built fresh on each call so callers/--flags can override globals before use."""
    return tc.RunConfig(
        file_prefix="train_gru", ckpt_prefix="gru",
        title=f"{RULESET} (GRU): BPTT vs RFLO", header_note=" (GRU)",
        rule_label=RULE_LABEL, rule_color=RULE_COLOR,
        seed=SEED, ruleset=RULESET, rules_to_run=RULES_TO_RUN,
        feedback_mode=FEEDBACK_MODE, n_runs=N_RUNS, n_hidden=N_HIDDEN,
        signal_mode=_FEEDBACK_TO_SIGNAL.get(FEEDBACK_MODE, FEEDBACK_MODE),
        batch=BATCH, n_datasets=N_DATASETS, lr=LR, grad_clip=GRAD_CLIP,
        log_every=LOG_EVERY, device=DEVICE, dtype=DTYPE,
        fig_dir=FIG_DIR, ckpt_dir=CKPT_DIR, data_dir=DATA_DIR, save_nets=SAVE_NETS,
        build_params=build_params,
        net_factory=lambda np_, verbose: gru.GRU(np_, verbose=verbose),
        # GRU has its own no-grad forward for held-out evaluation.
        eval_outputs=lambda net, inputs: net.forward_outputs(inputs),
        task=tasks.make_task(RULESET),
        acc_label=tasks.acc_label_for(RULESET),
        metric=tasks.metric_for(RULESET),
    )


# ─── Public path/replot helpers (thin wrappers over train_common) ─────────────
def fig_path():   return tc.fig_path(_cfg())
def data_path():  return tc.data_path(_cfg())
def ckpt_path(rule, seed): return tc.ckpt_path(_cfg(), rule, seed)
def replot_from_npz(npz_path, save_to=None):
    return tc.replot_from_npz(_cfg(), npz_path, save_to=save_to)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare BPTT vs RFLO-style local learning on a GRU.")
    p.add_argument("--task", default=RULESET, help="ruleset / task (default: %(default)s)")
    p.add_argument("--runs", type=int, default=N_RUNS, help="independent seeds")
    p.add_argument("--hidden", type=int, default=N_HIDDEN, help="hidden units")
    p.add_argument("--steps", type=int, default=N_DATASETS, help="training batches")
    p.add_argument("--learning-signal", choices=list(RNN_SIGNAL_MODES), default=None,
                   help="hidden learning signal, same names as train_mpn: exact_spatial "
                        "(true readout weights), layerwise_fa / dfa (fixed random "
                        "feedback; the two coincide for this single-hidden GRU). "
                        f"Default: {_FEEDBACK_TO_SIGNAL.get(FEEDBACK_MODE, FEEDBACK_MODE)}.")
    p.add_argument("--feedback",
                   choices=["exact_spatial", "layerwise_fa", "direct_fa", "exact_readout"],
                   default=None,
                   help="LEGACY spelling of --learning-signal (direct_fa = dfa; "
                        "'exact_readout' is the old name for exact_spatial). Must agree "
                        "with --learning-signal when both are given.")
    args = p.parse_args()
    if args.learning_signal is not None:
        implied = RNN_SIGNAL_MODES[args.learning_signal]
        if args.feedback is not None and _FEEDBACK_TO_SIGNAL[args.feedback] != args.learning_signal:
            p.error(f"--learning-signal {args.learning_signal} implies feedback {implied}; "
                    f"drop the legacy --feedback {args.feedback} or make them agree")
        args.feedback = implied
    else:
        args.feedback = args.feedback or FEEDBACK_MODE
    return args


def main():
    global RULESET, N_RUNS, N_HIDDEN, N_DATASETS, FEEDBACK_MODE
    args = _parse_args()
    RULESET = args.task
    N_RUNS = args.runs
    N_HIDDEN = args.hidden
    N_DATASETS = args.steps
    FEEDBACK_MODE = args.feedback
    tc.run_experiment(_cfg())


if __name__ == "__main__":
    main()
