#!/usr/bin/env python
# coding: utf-8
"""
Compare BPTT vs RFLO local learning on a leaky vanilla RNN, over N runs.

The RNN analog of train_mpn.py. For each of N_RUNS seeds we train RULES_TO_RUN
in lockstep:
  - 'bptt'            — autograd through the unrolled recurrent forward
  - 'local_diag_rflo' — RFLO (Murray & Escola 2019): forward-mode eligibility
                        traces that drop the recurrent sensitivity term. Shares
                        the API key with the MPN diagonal rule (see core/rnn.py).
Same lockstep protocol as train_mpn.py (identical init / data / valid set per
seed), same figure + .npz plot-data output. The shared machinery lives in
train_common.py; this file only sets the RNN config and hooks.

Hyperparameters mirror train_mpn.py / one_task.py. output_bias=False; no
regularization. net_type='vanilla' selects the RNN in the param converter; the
class used here is rnn.LeakyRNN.

Under 'local_diag_rflo' the WHOLE network (input, recurrent, bias) trains by
RFLO; under 'bptt' everything trains by exact autograd. The learning rule governs
every layer; there is no separate per-layer control.

Run from this directory:
    python train_rnn.py                      # BPTT vs RFLO (default)
    python train_rnn.py --hidden 100 --runs 3 --task delaygo
"""
import argparse
import torch
import numpy as np 

import _bootstrap  # prepends ../core + ../scripts to sys.path; exposes ROOT
import rnn
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
N_RUNS = 3                    # independent seeds per rule
N_HIDDEN = 180                # one_task.py: n_hidden = 200
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
    """(task, train, net) param dicts for a leaky vanilla RNN on one task.
    Regularization disabled so the objective is pure masked-MSE (what RFLO is
    derived for). net_type='vanilla' for the param converter; class = rnn.LeakyRNN."""
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


def _cfg():
    """Package the current module globals + RNN-specific hooks into a RunConfig.
    Built fresh on each call so callers/--flags can override globals before use."""
    return tc.RunConfig(
        file_prefix="train_rnn", ckpt_prefix="rnn",
        title=f"{RULESET} (leaky RNN): BPTT vs RFLO", header_note=" (leaky RNN)",
        rule_label=RULE_LABEL, rule_color=RULE_COLOR,
        seed=SEED, ruleset=RULESET, rules_to_run=RULES_TO_RUN,
        feedback_mode=FEEDBACK_MODE, n_runs=N_RUNS, n_hidden=N_HIDDEN,
        batch=BATCH, n_datasets=N_DATASETS, lr=LR, grad_clip=GRAD_CLIP,
        log_every=LOG_EVERY, device=DEVICE, dtype=DTYPE,
        fig_dir=FIG_DIR, ckpt_dir=CKPT_DIR, data_dir=DATA_DIR, save_nets=SAVE_NETS,
        build_params=build_params,
        net_factory=lambda np_, verbose: rnn.LeakyRNN(np_, verbose=verbose),
        # LeakyRNN has its own no-grad forward for held-out evaluation.
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
        description="Compare BPTT vs RFLO local learning on a leaky vanilla RNN.")
    p.add_argument("--task", default=RULESET, help="ruleset / task (default: %(default)s)")
    p.add_argument("--runs", type=int, default=N_RUNS, help="independent seeds")
    p.add_argument("--hidden", type=int, default=N_HIDDEN, help="hidden units")
    p.add_argument("--steps", type=int, default=N_DATASETS, help="training batches")
    p.add_argument("--feedback",
                   choices=["exact_spatial", "layerwise_fa", "direct_fa", "exact_readout"],
                   default=FEEDBACK_MODE,
                   help="hidden learning-signal feedback (both random modes "
                        "coincide for this single-hidden RNN). 'exact_readout' is "
                        "the legacy name for 'exact_spatial'.")
    return p.parse_args()


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
