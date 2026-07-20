#!/usr/bin/env python
# coding: utf-8
"""
Compare BPTT vs local learning on an MPN, over N runs.

Network is selectable with --net (or the NET_TYPE global):
  'dmpn' (default) — DeepMultiPlasticNet: raw input -> trainable input embedding
      (W_initial_linear, act) -> one MP layer (plastic W, M) -> readout. Directly
      comparable to the RNN in train_rnn.py (input -> hidden -> output), with the
      plastic M playing the role the RNN's recurrence plays.
  'mpn1' — MultiPlasticNet: single MP layer -> readout (no input embedding).

For each of N_RUNS seeds we train the RULES_TO_RUN networks in lockstep:
  - 'bptt'            — autograd through the unrolled forward + M-update
  - 'local_diag_rflo' — diagonal / same-synapse RFLO local learning
  - 'local_direct'    — direct/instantaneous local approximation
BPTT trains every parameter (input embedding included) exactly. Under the LOCAL
rules the MP layer uses eligibility traces and the input embedding uses a DIRECT
3-factor rule (the RNN's RFLO treatment of its input weights — see
DeepMultiPlasticNet._local_sequence_gradients). All start from the SAME init
(deepcopy), SAME per-step data, SAME held-out valid set, own Adam + scheduler,
via sequence_gradients() which writes .grad for optimizer.step().

Output: one figure with two panels — training and testing (held-out) accuracy vs
step — each rule mean ± std across the N runs, plus a .npz of the plotted arrays
(replot_from_npz regenerates the figure without retraining). Accuracy is the
library's angle accuracy (train isvalid=False, valid isvalid=True).

Hyperparameters align with one_task.py (hidden=200, lr=1e-3, batch=128, clip=10,
Adam, ReduceLROnPlateau, tanh); regularization OFF (pure masked-MSE, what the
local rules are derived for). Shared machinery lives in train_common.py.

Under a local rule (local_diag_rflo / local_direct) the WHOLE network — including
the input embedding — trains locally; under 'bptt' the whole network trains by
exact autograd. The learning rule governs every layer; there is no separate
per-layer control.

Weights & Biases (https://wandb.ai): pass --wandb to log every run live. Each
invocation becomes ONE W&B experiment named after this run's output save-stem (the
same name the figure/.npz/JSON share), containing K = len(RULES_TO_RUN) × N_RUNS
runs — one per (rule × seed). Group/color by 'rule' in the W&B UI to see exactly K
colors, with each of the N_RUNS seeds drawn as its own separate curve; a summary
run also logs the aggregate mean±std figures. Off by default (no wandb import, all
outputs byte-for-byte unchanged). See scripts/wandb_logging.py.

Run from this directory:
    python train_mpn.py                        # deep MPN (default)
    python train_mpn.py --wandb                # + log to Weights & Biases
    python train_mpn.py --net mpn1             # single MP layer, no input embedding
    python train_mpn.py --net dmpn --hidden 100 --runs 3 --task delaygo
    python train_mpn.py --net dmpn --hidden 150 100   # deep MP stack (two MP layers)

The MP-layer stack depth follows --hidden: one width → one MP layer; several
widths → one MP layer per width (a deep stack, dmpn only). The architecture is
encoded in every output filename (figure / checkpoint / .npz) and the figure
title, so multi-layer runs are self-describing and don't collide with the
single-layer runs on disk.
"""
import argparse
import os
import torch
import numpy as np 

import _bootstrap  # prepends ../core + ../scripts to sys.path; exposes ROOT
import mpn         # core/mpn.py — the efficiency-optimized implementation
import tasks
import train_common as tc

# ─── Configuration (aligned with MultiTaskMPN/one_task/one_task.py) ───────────
SEED = np.random.randint(0, 1000)  # random seed for this run (small → short run names)
RULESET = "contextdelaydm1"           # single task to train on
# Network: 'dmpn' = DeepMultiPlasticNet (trainable input embedding + MP layer,
# RNN-comparable); 'mpn1' = MultiPlasticNet (single MP layer, no embedding).
# Overridable with --net on the command line (see main()).
NET_TYPE = "dmpn"
RULES_TO_RUN = ["bptt", "local_exact_rowlocal", "local_diag_rflo", "local_direct"]   # rules to compare
# Hidden learning-signal feedback. 'exact_spatial' = exact same-time spatial
# gradient (weight transport; top plastic layer exact vs BPTT under
# local_exact_rowlocal); 'layerwise_fa' = recursive per-boundary feedback
# alignment; 'direct_fa' = direct feedback alignment. See mpn._FEEDBACK_MODES.
# exact_spatial differs from the random modes at any depth; layerwise_fa vs
# direct_fa differ only with >1 trainable boundary — which the default dmpn net
# always has (MP layer + trainable input embedding), so they differ here even for
# a single MP layer. (Legacy 'exact_readout' still loads.)
FEEDBACK_MODE = "exact_spatial"
# Learning rule for the TRAINABLE INPUT EMBEDDING (dmpn only), decoupled from the
# per-rule MP-layer learning_rule so any RULES_TO_RUN × input-rule combo compares:
#   'match'        — embedding follows each rule (exact under bptt, 3-factor local
#                    under a local rule). The historical default; no behavior change.
#   'exact'        — embedding ALWAYS trained by the true BPTT gradient (even in a
#                    local run — costs an extra BPTT pass for that rule).
#   'three_factor' — embedding ALWAYS uses the direct 3-factor local rule (even in
#                    the bptt run — MP+readout stay exact-autograd). --input-mode on CLI.
INPUT_MODE = "exact"
# Fixed input standardization of the raw input u_t. Because u_t feeds straight into
# the modulated forward W(1+M)x AND the Hebbian M update (η·h·x), its scale strongly
# conditions the modulation dynamics; standardizing by FIXED per-feature statistics
# (estimated once from a task sample, then frozen and applied identically to every
# rule + validation) removes that scale artifact without adding any adaptive/learned
# norm. Off by default (identity → existing runs/filenames unchanged byte-for-byte).
# --input-normalize on CLI. The sample size used to estimate the stats:
INPUT_NORMALIZE = False
INPUT_NORM_SAMPLE = 2048      # #trials sampled to estimate the fixed input stats
# Identity skip (residual) connections around each MP block: h_{n+1} = act(z_n) + h_n
# (dmpn only). A same-time, parameter-free, memoryless op — it stays fully local (the
# local rules gain the residual's identity Jacobian term in the inter-layer signal;
# the Hebbian write still uses the block activation) and leaves BPTT exact. A skip is
# inserted only where an MP layer's input/output widths MATCH (identity needs equal
# widths); unequal-width layers are skipped with a warning. Use EQUAL stacked widths
# (e.g. --hidden 128 128) to exercise it on a deep stack. Off by default. --residual.
MP_RESIDUAL = True
N_RUNS = 2                    # independent seeds per rule
# Hidden width(s) of the MP-layer stack. A single int → one MP layer (the classic
# in→hidden→out net). A list of ints → one MP layer per width, i.e. a DEEP MP
# stack (in→h1→h2→...→out); the deep local rules train every layer. The deep
# local rules now support any depth (see DeepMultiPlasticNet._local_sequence_gradients),
# so e.g. N_HIDDEN=[150, 100] builds two stacked MP layers. --hidden accepts
# several ints on the CLI.
N_HIDDEN = 200                # one_task.py: n_hidden = 200
N_DATASETS = 5000             # one_task.py: n_datasets = 3000 (heavy on CPU)
BATCH = 128                   # one_task.py: n_batches = batch_size = 128
LR = 1e-3                     # one_task.py: lr = 1e-3
GRAD_CLIP = 10                # one_task.py: gradient_clip = 10
LOG_EVERY = 100               # record/print accuracy every this many steps
# Output dirs anchored to the project root so they land at <root>/figure etc.
# regardless of the working directory the script is run from.
FIG_DIR = str(_bootstrap.ROOT / "figure")        # saved figures
CKPT_DIR = str(_bootstrap.ROOT / "checkpoints")  # saved trained networks
DATA_DIR = str(_bootstrap.ROOT / "figure_data")  # saved plot data (.npz)
SAVE_NETS = True              # save each trained network (per rule, per seed)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.float32         # float32 for speed (one_task.py also runs float32)

# ─── Weights & Biases (https://wandb.ai) logging (opt-in) ─────────────────────
# When USE_WANDB is on (--wandb), each invocation opens ONE W&B run per
# (rule × seed), all grouped under this run's output save-stem — the SAME name the
# figure/.npz/JSON share (train_common.run_stem) — which becomes the W&B
# experiment/group. Group/color by 'rule' in the UI to see exactly K = len(
# RULES_TO_RUN) colors, with each of the N_RUNS seeds drawn as its own separate
# curve; a final summary run logs the aggregate mean±std figures. Off by default →
# wandb is never imported and all outputs are byte-for-byte unchanged.
USE_WANDB = False
WANDB_PROJECT = "mpn_local_learning"   # W&B project the runs land in
WANDB_ENTITY = None                    # None → your default W&B entity (user/team)
WANDB_MODE = None                      # None → online; "offline"/"disabled" also valid

RULE_LABEL = {"bptt": "BPTT", "local_diag_rflo": "diagonal RFLO",
              "local_exact_rowlocal": "exact row-local", "local_direct": "direct"}
RULE_COLOR = {"bptt": "#1f77b4", "local_diag_rflo": "#d62728",
              "local_exact_rowlocal": "#2ca02c", "local_direct": "#9467bd"}


def _hidden_widths():
    """N_HIDDEN normalized to a list of MP-layer hidden widths (one per MP layer).
    A bare int → [int] (one MP layer); a list/tuple is used as-is (a deep stack)."""
    if isinstance(N_HIDDEN, (list, tuple)):
        return [int(w) for w in N_HIDDEN]
    return [int(N_HIDDEN)]


def build_params():
    """(task, train, net) param dicts for the chosen network on one task, in the
    local-rule derivation config (regularization off → pure masked-MSE).
    NET_TYPE selects 'dmpn' (trainable input embedding + MP layer, RNN-comparable)
    or 'mpn1' (single MP layer, no embedding). N_HIDDEN may be a single width (one
    MP layer) or a list of widths (a deep MP stack: in→h1→h2→...→out)."""
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
        "scheduler": {               # one_task.py: ReduceLROnPlateau
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.95,
            "patience": 30,
            "min_lr": 1e-8,
        },
    }

    widths = _hidden_widths()            # one entry per MP layer
    if NET_TYPE == "mpn1" and len(widths) > 1:
        raise ValueError(
            "mpn1 (MultiPlasticNet) is a single MP layer; pass one --hidden width "
            "or use --net dmpn for a multi-MP-layer stack.")
    if MP_RESIDUAL and NET_TYPE != "dmpn":
        raise ValueError(
            "--residual (identity skip connections) is implemented for dmpn only; "
            "use --net dmpn.")

    net_params = {
        "net_type": NET_TYPE,            # 'dmpn' or 'mpn1'
        # [in, h1, h2, ..., out]; in/out overwritten below. One MP layer per hidden
        # width (len(n_neurons)-2 MP layers). For dmpn the embedding is inserted
        # before h1 inside DeepMultiPlasticNet.
        "n_neurons": [1, *widths, 1],
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
        "input_normalize": INPUT_NORMALIZE,  # fixed per-feature input standardization
        "mp_residual": MP_RESIDUAL,      # identity skip around each equal-width MP block
        "input_mode": INPUT_MODE,        # input-embedding rule (match/exact/three_factor)
        "ml_params": {
            "bias": True,
            "mp_type": "mult",
            "m_update_type": "hebb_assoc",
            "m_activation": "linear",      # required for the local rules
            "modulation_bounds": False,    # required for the local rules
            "eta_type": "scalar",
            "eta_train": False,
            "lam_type": "scalar",
            "m_time_scale": 4000,          # dt=40 → lambda = 1 - dt/4000 = 0.99
            "lam_train": False,
            "W_freeze": False,
        },
    }
    if NET_TYPE == "dmpn":
        # Deep net: trainable input embedding before the MP layer (RNN-comparable).
        # Embedding width tracks the first MP layer's width.
        net_params.update({
            "linear_embed": widths[0],    # input-embedding width
            "input_layer_add": True,      # add a trainable input layer
            "input_layer_add_trainable": True,
            "input_layer_bias": True,
            "input_init_type": "xavier",
        })
    return task_params, train_params, net_params


def _mpn():
    """The MPN implementation module (core/mpn.py, efficiency-optimized)."""
    return mpn


@torch.no_grad()
def forward_outputs(net, inputs):
    """Unrolled forward over time (no grad), returning outputs (B, T, n_output),
    for evaluating the held-out validation set. Works for both net types:
    DeepMultiPlasticNet.network_step runs the full forward (embedding + MP layer)
    and returns 3 values; MultiPlasticNet's returns 2, so drive its mp_layer
    directly (the library's iterate_sequence_batch can't unpack the 2-tuple)."""
    B, T, _ = inputs.shape
    # Apply the SAME fixed input standardization the gradient paths use (identity
    # unless the net has input_normalize on), so held-out eval matches training.
    inputs = net._standardize_input(inputs)
    net.reset_state(B=B)
    outs = []
    if isinstance(net, mpn.DeepMultiPlasticNet):
        for t in range(T):
            out, _, _ = net.network_step(inputs[:, t, :], seq_idx=t)
            outs.append(out)
    else:  # MultiPlasticNet (single MP layer, no embedding)
        for t in range(T):
            x_t = inputs[:, t, :]
            hidden_pre, _ = net.mp_layer(x_t)
            hidden = net.act_fn(hidden_pre)
            out = torch.einsum('iI,BI->Bi', net.W_output, hidden) + net.b_output.unsqueeze(0)
            outs.append(out)
            net.mp_layer.update_M_matrix(x_t, hidden)
    return torch.stack(outs, dim=1)


def _net_class():
    """The network class selected by NET_TYPE, from the MPN implementation module."""
    m = _mpn()
    return m.DeepMultiPlasticNet if NET_TYPE == "dmpn" else m.MultiPlasticNet


def _eta_lam_tag():
    """Filename fragment for the MP-layer eta/lambda init, e.g. 'eta1.00_lam0.99'.
    Derived from build_params the same way MultiPlasticLayer initializes them:
    eta = eta_clamp (default 1.0); lambda = 1 - dt/m_time_scale."""
    task_params, _, net_params = build_params()
    ml = net_params["ml_params"]
    eta0 = ml.get("eta_clamp", 1.00)                      # default eta init
    dt = task_params.get("dt", 40)
    lam0 = 1.0 - dt / ml["m_time_scale"]                  # default lambda init
    return f"eta{eta0:.2f}_lam{lam0:.2f}"


def _arch_tag():
    """Filename-safe architecture fragment for the MP-layer stack, so multi-MP-layer
    runs don't collide on disk. Returns "" for a SINGLE hidden width — param_tag
    then falls back to the historical 'h{n_hidden}', so existing single-layer
    checkpoints/figures resolve byte-for-byte. For a deep stack it returns the
    joined widths, e.g. N_HIDDEN=[150, 100] -> 'h150-100'. (Embedding presence is
    already encoded by the dmpn/mpn1 filename prefix; the FULL arch, embedding
    included, appears in the figure title/provenance via _arch_desc.)"""
    widths = _hidden_widths()
    if len(widths) == 1:
        return ""                                     # → legacy 'h{n_hidden}' tag
    return "h" + "-".join(str(w) for w in widths)     # e.g. 'h150-100'


def _arch_desc():
    """Human-readable full architecture for the figure title / provenance, e.g.
    'arch=[20, 200, 150, 100, 1]' (the actual n_neurons the net is built with,
    embedding inserted for dmpn). Falls back cleanly for a single hidden layer."""
    _, _, net_params = build_params()
    arch = list(net_params["n_neurons"])
    if NET_TYPE == "dmpn":
        arch.insert(1, net_params.get("linear_embed", arch[1]))
    return f"arch={arch}"


def _cfg():
    """Package the current module globals + MPN-specific hooks into a RunConfig.
    Built fresh on each call so notebooks/validate/--net can override globals
    (e.g. NET_TYPE, N_HIDDEN, FEEDBACK_MODE) before any path/build helper below."""
    net_cls = _net_class()
    desc = "deep MPN" if NET_TYPE == "dmpn" else "MPN"
    # NET_TYPE is part of the prefix so dmpn/mpn1 runs don't overwrite each other.
    # n_hidden stays a SCALAR (the first MP-layer width) for the legacy 'h{n_hidden}'
    # single-layer filename fallback + provenance; the full stack rides in
    # arch_tag (filenames) and arch_desc (title/provenance).
    return tc.RunConfig(
        file_prefix=f"train_{NET_TYPE}", ckpt_prefix=NET_TYPE,
        title=f"{RULESET} ({desc}): BPTT vs local", header_note=f" ({desc})",
        rule_label=RULE_LABEL, rule_color=RULE_COLOR,
        seed=SEED, ruleset=RULESET, rules_to_run=RULES_TO_RUN,
        feedback_mode=FEEDBACK_MODE,
        input_normalize=INPUT_NORMALIZE, input_norm_sample=INPUT_NORM_SAMPLE,
        mp_residual=MP_RESIDUAL,
        input_mode=INPUT_MODE, n_runs=N_RUNS, n_hidden=_hidden_widths()[0],
        batch=BATCH, n_datasets=N_DATASETS, lr=LR, grad_clip=GRAD_CLIP,
        log_every=LOG_EVERY, device=DEVICE, dtype=DTYPE,
        fig_dir=FIG_DIR, ckpt_dir=CKPT_DIR, data_dir=DATA_DIR, save_nets=SAVE_NETS,
        arch_tag=_arch_tag(), arch_desc=_arch_desc(),
        use_wandb=USE_WANDB, wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY, wandb_mode=WANDB_MODE,
        build_params=build_params,
        net_factory=lambda np_, verbose: net_cls(np_, verbose=verbose),
        eval_outputs=forward_outputs,
        task=tasks.make_task(RULESET),
        acc_label=tasks.acc_label_for(RULESET),
        metric=tasks.metric_for(RULESET),
        tag_extra=_eta_lam_tag(),
    )


# ─── Public path/replot helpers (thin wrappers over train_common) ─────────────
def fig_path():   return tc.fig_path(_cfg())
def data_path():  return tc.data_path(_cfg())
def config_path(): return tc.config_path(_cfg())
def ckpt_path(rule, seed): return tc.ckpt_path(_cfg(), rule, seed)
def replot_from_npz(npz_path, save_to=None):
    return tc.replot_from_npz(_cfg(), npz_path, save_to=save_to)


def load_net(path, device=None, dtype=DTYPE):
    """Reload a network saved during training. Reconstructs whichever class the
    checkpoint's net_params describes ('dmpn' → DeepMultiPlasticNet, else
    MultiPlasticNet), independent of the current NET_TYPE, with trained weights
    and learning_rule restored.  Example:  net = load_net(ckpt_path('bptt', 42))"""
    device = device or DEVICE
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = _mpn()  # the MPN implementation module (core/mpn.py)
    net_cls = (m.DeepMultiPlasticNet if ckpt["net_params"].get("net_type") == "dmpn"
               else m.MultiPlasticNet)
    net = net_cls(ckpt["net_params"], verbose=False).to(device).to(dtype)
    net.load_state_dict(ckpt["state_dict"])
    net.learning_rule = ckpt.get("learning_rule", net.learning_rule)
    return net


def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare BPTT vs local learning on an MPN (deep or single-layer).")
    p.add_argument("--net", choices=["dmpn", "mpn1"], default=NET_TYPE,
                   help="dmpn: DeepMultiPlasticNet (trainable input embedding + MP "
                        "layer, RNN-comparable). mpn1: MultiPlasticNet (single MP "
                        "layer, no embedding). Default: %(default)s.")
    p.add_argument("--task", default=RULESET, help="ruleset / task (default: %(default)s)")
    p.add_argument("--runs", type=int, default=N_RUNS, help="independent seeds")
    p.add_argument("--hidden", type=int, nargs="+", default=None,
                   help="hidden width(s): one int → a single MP layer (classic); "
                        "several ints → a deep MP stack, one MP layer per width "
                        "(dmpn only), e.g. --hidden 150 100. Default: the N_HIDDEN "
                        "global.")
    p.add_argument("--steps", type=int, default=N_DATASETS, help="training batches")
    p.add_argument("--feedback",
                   choices=["exact_spatial", "layerwise_fa", "direct_fa", "exact_readout"],
                   default=FEEDBACK_MODE,
                   help="hidden learning-signal feedback. exact_spatial differs from "
                        "the random modes at any depth; layerwise_fa vs direct_fa "
                        "differ only with >1 trainable boundary (dmpn's embedding "
                        "counts). 'exact_readout' is the legacy name for 'exact_spatial'.")
    p.add_argument("--input-mode", choices=["match", "exact", "three_factor"],
                   default=INPUT_MODE,
                   help="input-embedding learning rule (dmpn), decoupled from the "
                        "MP-layer rule: 'match' = per-rule native (default), 'exact' "
                        "= always BPTT gradient, 'three_factor' = always the direct "
                        "local rule. Default: %(default)s.")
    p.add_argument("--input-normalize", action=argparse.BooleanOptionalAction,
                   default=INPUT_NORMALIZE,
                   help="fixed per-feature standardization of the raw input u_t "
                        "(statistics estimated once from a task sample, then frozen "
                        "and applied identically to every rule + validation). "
                        "Conditions all rules equally by removing a scale artifact. "
                        "Use --no-input-normalize to force off. Default: %(default)s.")
    p.add_argument("--residual", action=argparse.BooleanOptionalAction,
                   default=MP_RESIDUAL,
                   help="identity skip connections around each equal-width MP block "
                        "(dmpn only): h_{n+1}=act(z_n)+h_n. Parameter-free, stays fully "
                        "local (adds the residual's identity Jacobian term to the "
                        "inter-layer signal), BPTT stays exact. Needs equal stacked "
                        "widths (e.g. --hidden 128 128); unequal layers skip it with a "
                        "warning. Default: %(default)s.")
    p.add_argument("--wandb", dest="use_wandb", action="store_true", default=USE_WANDB,
                   help="log to Weights & Biases (https://wandb.ai): one run per "
                        "(rule × seed), all grouped under this run's output save-stem "
                        "as the experiment name. Group/color by 'rule' in the UI for "
                        "K=len(RULES_TO_RUN) colors, each seed a separate curve. "
                        "Off by default.")
    p.add_argument("--wandb-project", default=WANDB_PROJECT,
                   help="W&B project name (default: %(default)s).")
    p.add_argument("--wandb-entity", default=WANDB_ENTITY,
                   help="W&B entity (user/team); default: your W&B default.")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                   default=WANDB_MODE,
                   help="W&B mode; default: online. Use 'offline' to log locally and "
                        "`wandb sync` later (no login needed).")
    return p.parse_args()


def main():
    global NET_TYPE, RULESET, N_RUNS, N_HIDDEN, N_DATASETS, FEEDBACK_MODE
    global INPUT_MODE, INPUT_NORMALIZE, MP_RESIDUAL
    global USE_WANDB, WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE
    args = _parse_args()
    NET_TYPE = args.net
    RULESET = args.task
    N_RUNS = args.runs
    if args.hidden is not None:
        # one int → scalar (single MP layer); several → list (deep MP stack).
        N_HIDDEN = args.hidden[0] if len(args.hidden) == 1 else args.hidden
    N_DATASETS = args.steps
    FEEDBACK_MODE = args.feedback
    INPUT_MODE = args.input_mode
    INPUT_NORMALIZE = args.input_normalize
    MP_RESIDUAL = args.residual
    USE_WANDB = args.use_wandb
    WANDB_PROJECT = args.wandb_project
    WANDB_ENTITY = args.wandb_entity
    WANDB_MODE = args.wandb_mode
    tc.run_experiment(_cfg())


if __name__ == "__main__":
    main()
