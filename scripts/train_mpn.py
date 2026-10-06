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
BPTT with input_mode=match/exact/diag_mtrace/paired trains every parameter exactly.
Local MP layers use the selected eligibility rule. Input mode match uses direct
three-factor input gradients; exact uses a BPTT input splice; diag_mtrace adds
first-MP-layer modulation-column sensitivities for input weights and biases.
paired uses diag_mtrace only for local_diag_rflo, direct three-factor input
updates for local_direct, and full BPTT for bptt; row-local is unsupported.
three_factor forces direct input gradients even under BPTT. All start from the
SAME init (deepcopy), SAME per-step data, SAME held-out valid set, own Adam + scheduler,
via sequence_gradients() which writes .grad for optimizer.step().

Output: one figure with two panels — training and testing (held-out) accuracy vs
step — each rule mean ± std across the N runs, plus a .npz of the plotted arrays
(replot_from_npz regenerates the figure without retraining). Accuracy is the
library's angle accuracy (train isvalid=False, valid isvalid=True).

Hyperparameters align with one_task.py (hidden=200, lr=1e-3, batch=128, clip=10,
Adam, ReduceLROnPlateau, tanh). Modulation bounds default to [-1, 1]; L2 weight
regularization is disabled by default. Logged losses remain task losses.
Shared machinery lives in train_common.py.

A local update is LEARNING SIGNAL x ELIGIBILITY, and the three CLI axes are kept
independent so one flag changes one thing:
  --learning-signal  WHERE each MP layer's signal comes from (one of five modes):
                     exact_spatial (default; main error through the true weights),
                     layerwise_fa (fixed random matrix at every boundary), dfa (main
                     error projected directly to every layer), local_readout (each
                     module's own auxiliary head), mixed (exact_spatial + alpha*local).
                     Internally this sets the model's feedback_mode + learning_signal.
  --rules            WHICH eligibility consumes it (local_direct / local_diag_rflo /
                     local_exact_rowlocal) or full bptt.
  --input-mode       HOW the input embedding is trained (match / exact / three_factor /
                     diag_mtrace / paired) — see INPUT_MODE below.
Defaults are signal-independent: rules bptt+local_direct+local_diag_rflo, input
mode match (bptt = full BPTT, local runs fully local), direct bias updates, no
cross-layer correction, alignment diagnostic on. Switching only --learning-signal
therefore changes only the signal. Example:
    python scripts/train_mpn.py --learning-signal dfa --hidden 64 64 --task delaygo --steps 500
--dfa is kept as the LEGACY preset (direct_fa + match + direct bias + rules
bptt/row-local/diagonal + alignment off); --feedback and --learning-signal global
are legacy spellings translated into the unified mode. Explicitly conflicting
flags are rejected rather than resolved by argument order. --local-bias-mode exact
retains exact bias traces (needed for the row-local rule's top-layer bias exactness).

Use --learning-signal local_readout for per-layer LOCAL READOUT heads (dmpn,
multi-MP-layer): every non-top MP layer gets an auxiliary linear head trained on
the task loss, and that head's error — not a signal descending from the layers
above — credits the layer's eligibility; the top layer keeps W_output and the
embedding shares module 0's head. It implies exact_spatial feedback, needs
cross-layer-steps 0 and cannot splice a BPTT input gradient; if the module
defaults were changed to 'exact' / 1 they switch to 'match' / 0 unless set
explicitly. bptt ignores the setting. Pair it with --seed so an exact_spatial and
a local_readout invocation share init and training data:
    python scripts/train_mpn.py --task seqmnist --hidden 128 128 --seed 7
    python scripts/train_mpn.py --task seqmnist --hidden 128 128 --seed 7 \
        --learning-signal local_readout

Weights & Biases (https://wandb.ai): pass --wandb to log every run live. Each
invocation becomes ONE W&B experiment named after this run's output ID (also used
for the figure/.npz and checkpoint folder), containing len(RULES_TO_RUN) × N_RUNS
runs — one per (rule × seed). Group/color by 'rule' in the W&B UI to use one color
per learning rule, with each seed drawn as its own separate curve; a summary
run also logs the aggregate mean±std figures. Off by default; wandb is only
imported when enabled. See scripts/wandb_logging.py.

Run from this directory:
    python train_mpn.py                        # deep MPN (default)
    python train_mpn.py --wandb                # + log to Weights & Biases
    python train_mpn.py --net mpn1 --no-residual  # single MP layer, no embedding
    python train_mpn.py --net dmpn --hidden 100 --runs 3 --task delaygo
    python train_mpn.py --net dmpn --hidden 150 100   # deep MP stack (two MP layers)

The MP-layer stack depth follows --hidden: one width → one MP layer; several
widths → one MP layer per width (a deep stack, dmpn only). Every output of one
invocation shares ONE run ID, <model>_<task>_<YYYYMMDD_HHMMSS>_<hash12>:
checkpoints/<run-id>/seed<N>/<rule>.pt (plus config.json in the run folder),
figure/<run-id>.png, figure_data/<run-id>.npz and log/<run-id>.log. The
timestamp is the ID's creation time (so names sort chronologically) and the hash
keeps invocations distinct even within the same second. Architecture is also in
figure titles and plot-data metadata.
"""
import argparse
from datetime import datetime
import json
import uuid
import torch
import numpy as np 

import _bootstrap  # prepends ../core + ../scripts to sys.path; exposes ROOT
import mpn         # core/mpn.py — the efficiency-optimized implementation
import tasks
import train_common as tc
from run_logging import tee_output, rename_active_log

# ─── Configuration (aligned with MultiTaskMPN/one_task/one_task.py) ───────────
SEED = np.random.randint(0, 1000)  # starting seed; subsequent runs increment it
# Stable path helpers within this process; separate configurations/invocations
# receive distinct IDs, including repeated CLI runs with identical seeds.
_RUN_IDS = {}
RULESET = "contextdelaydm1"           # single task to train on
# Network: 'dmpn' = DeepMultiPlasticNet (trainable input embedding + MP layer,
# RNN-comparable); 'mpn1' = MultiPlasticNet (single MP layer, no embedding).
# Overridable with --net on the command line (see main()).
NET_TYPE = "dmpn"
# Rules to compare (--rules). Default changed 2026-10 from
# [bptt, local_exact_rowlocal, local_diag_rflo, local_direct] to the three rules the
# project's comparisons use; add local_exact_rowlocal explicitly (with
# --local-bias-mode exact for its bias exactness) when you want it.
RULES_TO_RUN = ["bptt", "local_direct", "local_diag_rflo"]
# Hidden learning-signal feedback. 'exact_spatial' = exact same-time spatial
# gradient (weight transport; top plastic layer exact vs BPTT under
# local_exact_rowlocal); 'layerwise_fa' = recursive per-boundary feedback
# alignment; 'direct_fa' = direct feedback alignment. See mpn._FEEDBACK_MODES.
# exact_spatial differs from the random modes at any depth; layerwise_fa vs
# direct_fa differ only with >1 trainable boundary — which the default dmpn net
# always has (MP layer + trainable input embedding), so they differ here even for
# a single MP layer. (Legacy 'exact_readout' still loads.)
FEEDBACK_MODE = "exact_spatial"
DFA_PRESET = False             # records whether the CLI --dfa preset was selected
# Learning rule for the TRAINABLE INPUT EMBEDDING (dmpn only), decoupled from the
# per-rule MP-layer learning_rule so any RULES_TO_RUN × input-rule combo compares:
#   'match'        — embedding follows each rule (exact under bptt, 3-factor local
#                    under a local rule). The historical default; no behavior change.
#   'exact'        — embedding ALWAYS trained by the true BPTT gradient (even in a
#                    local run — costs an extra BPTT pass for that rule).
#   'three_factor' — embedding ALWAYS uses the direct 3-factor local rule (even in
#                    the bptt run — MP+readout stay exact-autograd). --input-mode on CLI.
#   'diag_mtrace'  — local runs use a first-MP-layer diagonal-column modulation
#                    sensitivity trace; BPTT stays exact. Needs exact_spatial.
#   'paired'       — bptt: exact; local_direct: three_factor; local_diag_rflo:
#                    diag_mtrace. Requires dmpn/exact_spatial; excludes row-local.
# Default changed 2026-10 from 'exact' (a HYBRID: local runs spliced a full-BPTT
# embedding gradient) to 'match', so bptt is full BPTT and local runs are fully local.
INPUT_MODE = "match"
# MP bias eligibility for the row-local / diagonal rules ('direct' = phi' only, no
# bias trace; 'exact' = the row-local bias trace; 'match' = each rule's own level of
# locality: direct for diagonal RFLO, exact for row-local). local_direct always
# uses direct and bptt always uses autograd. Default 'match' (changed 2026-10 from
# 'exact' via 'direct'): identical to 'direct' for the default rule set, and the
# row-local rule keeps its bias trace whenever it is added. Pass an explicit
# 'direct' or 'exact' when every trace-based rule must share ONE bias policy (the
# --dfa preset selects direct).
LOCAL_BIAS_MODE = "match"
RFLO_TRACE_RHO = None  # optional cap on the MP-weight diagonal trace recurrence gain
LAM = None  # optional fixed modulation decay; None retains dt / m_time_scale setup
LOG_GRAD_ALIGN = True  # --dfa disables the optional BPTT diagnostic by default
MODULATION_BOUNDS = True
MODULATION_MODE = "hard"       # none | hard | scaled_tanh
MODULATION_BOUND = 1.0         # symmetric hard bound / scaled-tanh scale B
REG_LAMBDA = 0.0
# Fixed input standardization of the raw input u_t. Because u_t feeds straight into
# the modulated forward W(1+M)x AND the Hebbian M update (η·h·x), its scale strongly
# conditions the modulation dynamics; standardizing by FIXED per-feature statistics
# (estimated once from a task sample, then frozen and applied identically to every
# rule + validation) removes that scale artifact without adding any adaptive/learned
# norm. Off by default (identity); recorded in checkpoint/config metadata.
# --input-normalize on CLI. The sample size used to estimate the stats:
INPUT_NORMALIZE = False
INPUT_NORM_SAMPLE = 2048      # #trials sampled to estimate the fixed input stats
# Identity skip (residual) connections around each MP block: h_{n+1} = act(z_n) + h_n
# (dmpn only). A same-time, parameter-free, memoryless op — it stays fully local (the
# local rules gain the residual's identity Jacobian term in the inter-layer signal;
# the Hebbian write still uses the block activation) and leaves BPTT exact. A skip is
# inserted only where an MP layer's input/output widths MATCH (identity needs equal
# widths); unequal-width layers are skipped with a warning. Use EQUAL stacked widths
# (e.g. --hidden 128 128) to exercise it on a deep stack. --residual.
# Single-MP-layer networks also honor this setting: their skip connects the
# embedding to the MP output. Use --no-residual to reproduce old single-layer runs.
MP_RESIDUAL = True
# Cross-layer TEMPORAL correction depth for the local rules (dmpn, multi-MP-layer).
# The base local rules credit each hidden layer with a same-time inter-layer signal
# only, DROPPING the temporal paths through the plastic state of the layers above it.
# CROSS_LAYER_STEPS=1 adds back the exact leading (one-temporal-hop) term, which makes
# MP-layer gradients exact vs BPTT at T=2 under exact spatial feedback and exact
# weight/bias eligibility. It need not improve gradient error at longer T (see mpn.DeepMultiPlasticNet._cross_layer_correction and
# tests/validate_local_learning tier17). It costs one extra downward adjoint sweep +
# a step of stored state per training step, and applies to every non-bptt rule
# (bptt is already exact). 0 selects the pure same-time rule. Only 0 and 1 are
# implemented. Default changed 2026-10 from 1 to 0 (the pure same-time local rules);
# it is rejected with the random-feedback and local signal modes (see validate_config).
CROSS_LAYER_STEPS = 0
# Learning-signal SOURCE for the local rules (dmpn, multi-MP-layer; see
# mpn._LEARNING_SIGNALS). 'global' (default) = the main readout's error reaches
# every layer through the inter-layer pathway — byte-identical to before.
# 'local_readout' = every non-top MP layer owns an auxiliary linear head trained on
# the same task loss; its error, projected through the head weights, is that layer's
# learning signal and nothing descends from the layers above (the top layer keeps
# W_output; the embedding shares module 0's head). 'mixed' adds alpha × the local
# signal to the global one. Both local modes require exact_spatial feedback and
# cross_layer_steps=0, and a local run may not splice a BPTT input gradient
# (input_mode 'exact'); when those two defaults would conflict the CLI switches them
# to 'match' / 0 unless you set them explicitly. bptt ignores the setting (its heads
# receive no gradient), so the BPTT baseline is unchanged. Single-MP-layer nets have
# no head, so local_readout then coincides with global. --learning-signal on CLI.
LEARNING_SIGNAL = "global"
LOCAL_SIGNAL_ALPHA = 1.0       # weight of the local signal under 'mixed' (--local-signal-alpha)

# ─── Unified signal modes: the CLI's single axis for "which signal each MP layer
# learns from". The model keeps TWO orthogonal fields (feedback_mode = how a global
# signal is transported; learning_signal = where the signal comes from), but the
# local modes pin feedback_mode to exact_spatial, so exactly these five combinations
# exist and --learning-signal names them directly. Everything else (eligibility
# rule, input mode, bias mode, cross-layer correction) is chosen independently.
#   mode            learning_signal   feedback_mode   signal each MP layer learns from
#   exact_spatial   global            exact_spatial   main error through the true weights
#   layerwise_fa    global            layerwise_fa    main error through a fixed random
#                                                     matrix at every boundary
#   dfa             global            direct_fa       main error projected directly to
#                                                     every layer
#   local_readout   local_readout     exact_spatial   each module's own auxiliary head
#   mixed           mixed             exact_spatial   exact_spatial + alpha * local head
# 'exact_spatial' qualifies the SPATIAL pathway only — temporal credit is still what
# the eligibility rule provides; none of these is full BPTT.
SIGNAL_MODES = {
    "exact_spatial": ("global", "exact_spatial"),
    "layerwise_fa": ("global", "layerwise_fa"),
    "dfa": ("global", "direct_fa"),
    "local_readout": ("local_readout", "exact_spatial"),
    "mixed": ("mixed", "exact_spatial"),
}


def signal_mode_from(learning_signal, feedback_mode):
    """Name of the unified signal mode for a (learning_signal, feedback_mode) pair,
    i.e. the inverse of SIGNAL_MODES. A pair the CLI cannot express (e.g.
    local_readout with random feedback, which the model also rejects) is returned
    as 'learning_signal+feedback_mode' so metadata never misreports it."""
    feedback_mode = mpn.canonical_feedback_mode(feedback_mode)
    for name, pair in SIGNAL_MODES.items():
        if pair == (learning_signal, feedback_mode):
            return name
    return f"{learning_signal}+{feedback_mode}"
N_RUNS = 1                    # independent seeds per rule
# Hidden width(s) of the MP-layer stack. A single int → one MP layer (the classic
# in→hidden→out net). A list of ints → one MP layer per width, i.e. a DEEP MP
# stack (in→h1→h2→...→out); the deep local rules train every layer. The deep
# local rules now support any depth (see DeepMultiPlasticNet._local_sequence_gradients),
# so e.g. N_HIDDEN=[150, 100] builds two stacked MP layers. --hidden accepts
# several ints on the CLI.
N_HIDDEN = 200                # one_task.py: n_hidden = 200
N_DATASETS = 5000             # one_task.py: n_datasets = 3000 (heavy on CPU)
BATCH = 128                   # --batch-size; validation uses 3*BATCH samples
LR = 1e-3                     # one_task.py: lr = 1e-3
HEAD_LR_MULT = 1.0            # auxiliary readouts only; BPTT heads are unused
LR_SCHEDULE = "plateau"       # legacy main-loss scheduling; "constant" disables it
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
# (rule × seed), all grouped under the output ID shared by the figure/.npz and
# checkpoint folder (train_common.run_stem), which becomes the W&B
# experiment/group. Group/color by 'rule' in the UI to see exactly K = len(
# RULES_TO_RUN) colors, with each of the N_RUNS seeds drawn as its own separate
# curve; a final summary run logs the aggregate mean±std figures. Off by default →
# wandb is not imported and no W&B runs are created.
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


def _mp_residual():
    """Honor the residual setting at every depth; the model checks widths.

    Shared by build_params and run metadata so the architecture and saved labels
    agree, including for a single MP layer.
    """
    return bool(MP_RESIDUAL)


def build_params():
    """(task, train, net) param dicts for the chosen network on one task, in the
    optional modulation bounds and L2 weight regularization configuration.
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
        "head_lr_mult": HEAD_LR_MULT,
        "lr_schedule": LR_SCHEDULE,
        "n_batches": BATCH,
        "batch_size": BATCH,
        "gradient_clip": GRAD_CLIP,
        "valid_n_batch": BATCH * 3,
        "n_datasets": N_DATASETS,
        "valid_check": None,
        "n_epochs_per_set": 1,
        "task_mask": None,
        "weight_reg": "L2",
        "activity_reg": None,
        "reg_lambda": REG_LAMBDA,
        "scheduler": {               # one_task.py: ReduceLROnPlateau
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.95,
            "patience": 30,
            "min_lr": 1e-8,
        } if LR_SCHEDULE == "plateau" else None,
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
    if LEARNING_SIGNAL != "global" and NET_TYPE != "dmpn":
        raise ValueError(
            f"--learning-signal {LEARNING_SIGNAL} (local readout heads) needs the deep "
            "net's non-top MP layers; use --net dmpn.")
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
        "mp_residual": _mp_residual(),   # identity skip around each equal-width MP block
        "cross_layer_steps": CROSS_LAYER_STEPS,  # depth-1 cross-layer temporal correction
        "input_mode": INPUT_MODE,        # requested input-embedding policy
        "learning_signal": LEARNING_SIGNAL,      # global | local_readout | mixed
        "local_signal_alpha": LOCAL_SIGNAL_ALPHA,  # local weight under 'mixed'
        "ml_params": {
            "bias": True,
            "local_bias_mode": LOCAL_BIAS_MODE,
            "rflo_trace_rho": RFLO_TRACE_RHO,
            "mp_type": "mult",
            "m_update_type": "hebb_assoc",
            "m_activation": "scaled_tanh" if MODULATION_MODE == "scaled_tanh" else "linear",
            "m_scale": MODULATION_BOUND,
            "modulation_bounds": MODULATION_BOUNDS and MODULATION_MODE == "hard",
            "m_bounds": (-MODULATION_BOUND, MODULATION_BOUND),
            "eta_type": "scalar",
            "eta_train": False,
            "lam_type": "scalar",
            "m_time_scale": 4000,          # default dt=40 → lambda=0.99; --lam overrides below
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
    if LAM is not None:
        # The core prioritizes m_time_scale over lam_clamp, so keep only the
        # explicit decay when requested; its time constant is derived from dt.
        net_params["ml_params"].pop("m_time_scale")
        net_params["ml_params"]["lam_clamp"] = LAM
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


def _arch_tag():
    """MP-stack metadata label; a single width uses the scalar n_hidden fallback."""
    widths = _hidden_widths()
    if len(widths) == 1:
        return ""                                     # use n_hidden metadata
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


def new_run_id(net_type, ruleset, now=None):
    """The run ID every output of an invocation shares — checkpoints/<id>/,
    figure/<id>.png, figure_data/<id>.npz, log/<id>.log (see train_common.run_stem
    and run_logging.rename_active_log): <model>_<task>_<YYYYMMDD_HHMMSS>_<hash12>.
    The timestamp (creation time of the ID) makes listings sort chronologically;
    the random hash keeps simultaneous or repeated invocations distinct."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{net_type}_{ruleset}_{stamp}_{uuid.uuid4().hex[:12]}"


def _cfg():
    """Package the current module globals + MPN-specific hooks into a RunConfig.
    Built fresh on each call so notebooks/validate/--net can override globals
    (e.g. NET_TYPE, N_HIDDEN, FEEDBACK_MODE) before any path/build helper below."""
    net_cls = _net_class()
    desc = "deep MPN" if NET_TYPE == "dmpn" else "MPN"
    # Repeated helper calls for the same setup must resolve the same output paths.
    # Keep configuration details in metadata rather than encoding them in names.
    signature = json.dumps(tc._json_safe([
        build_params(), SEED, N_RUNS, RULES_TO_RUN, LOG_EVERY, LOG_GRAD_ALIGN,
        INPUT_NORM_SAMPLE, str(DEVICE), str(DTYPE), DFA_PRESET,
    ]), sort_keys=True)
    if signature not in _RUN_IDS:
        _RUN_IDS[signature] = new_run_id(NET_TYPE, RULESET)
    return tc.RunConfig(
        run_id=_RUN_IDS[signature],
        dfa_preset=DFA_PRESET,
        file_prefix=f"train_{NET_TYPE}", ckpt_prefix=NET_TYPE,
        title=f"{RULESET} ({desc}): BPTT vs local", header_note=f" ({desc})",
        rule_label=({**RULE_LABEL,
                     "local_exact_rowlocal": "exact row-local + DFA",
                     "local_diag_rflo": "diagonal + DFA",
                     "local_direct": "direct + DFA"}
                    if FEEDBACK_MODE == "direct_fa" else RULE_LABEL),
        rule_color=RULE_COLOR,
        seed=SEED, ruleset=RULESET, rules_to_run=RULES_TO_RUN,
        feedback_mode=FEEDBACK_MODE,
        input_normalize=INPUT_NORMALIZE, input_norm_sample=INPUT_NORM_SAMPLE,
        mp_residual=_mp_residual(), cross_layer_steps=CROSS_LAYER_STEPS,
        input_mode=INPUT_MODE, learning_signal=LEARNING_SIGNAL,
        local_signal_alpha=LOCAL_SIGNAL_ALPHA,
        signal_mode=signal_mode_from(LEARNING_SIGNAL, FEEDBACK_MODE),
        n_runs=N_RUNS, n_hidden=_hidden_widths()[0],
        batch=BATCH, n_datasets=N_DATASETS, lr=LR, grad_clip=GRAD_CLIP,
        head_lr_mult=HEAD_LR_MULT, lr_schedule=LR_SCHEDULE,
        log_every=LOG_EVERY, log_grad_align=LOG_GRAD_ALIGN, device=DEVICE, dtype=DTYPE,
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
    and learning_rule restored. Pass an existing checkpoint path; ckpt_path()
    refers to the current process's experiment, not a previous CLI invocation."""
    device = device or DEVICE
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = _mpn()  # the MPN implementation module (core/mpn.py)
    net_cls = (m.DeepMultiPlasticNet if ckpt["net_params"].get("net_type") == "dmpn"
               else m.MultiPlasticNet)
    net = net_cls(ckpt["net_params"], verbose=False).to(device).to(dtype)
    net.load_state_dict(ckpt["state_dict"])
    net.learning_rule = ckpt.get("learning_rule", net.learning_rule)
    return net


def parse_arguments(argv=None):
    """Stage 1 of 3: pure argparse, grouped by what a flag acts on; no cross-flag
    logic. Returns (parser, args) so the later stages can report conflicts through
    parser.error (with the usage line)."""
    p = argparse.ArgumentParser(
        description="Compare BPTT vs local learning on an MPN (deep or single-layer). "
                    "A local update is LEARNING SIGNAL (--learning-signal) x ELIGIBILITY "
                    "(--rules); the input embedding's update is a third, independent "
                    "choice (--input-mode). Defaults are signal-independent, so changing "
                    "only --learning-signal changes only the signal.")

    g = p.add_argument_group("network and task")
    g.add_argument("--task", default=RULESET, help="ruleset / task (default: %(default)s)")
    g.add_argument("--net", choices=["dmpn", "mpn1"], default=NET_TYPE,
                   help="dmpn: DeepMultiPlasticNet (trainable input embedding + MP "
                        "layer, RNN-comparable). mpn1: MultiPlasticNet (single MP "
                        "layer, no embedding). Default: %(default)s.")
    g.add_argument("--hidden", type=int, nargs="+", default=None,
                   help="hidden width(s): one int → a single MP layer (classic); "
                        "several ints → a deep MP stack, one MP layer per width "
                        "(dmpn only), e.g. --hidden 150 100. Default: the N_HIDDEN "
                        "global.")
    g.add_argument("--residual", action=argparse.BooleanOptionalAction,
                   default=MP_RESIDUAL,
                   help="identity skip connections around each equal-width MP block "
                        "(dmpn only): h_{n+1}=act(z_n)+h_n. Parameter-free, stays fully "
                        "local (adds the residual's identity Jacobian term to the "
                        "inter-layer signal), BPTT stays exact. Needs equal stacked "
                        "widths (e.g. --hidden 128 128); unequal layers skip it with a "
                        "warning. Changes the forward net, so every rule is affected. "
                        "Default: %(default)s.")
    g.add_argument("--input-normalize", action=argparse.BooleanOptionalAction,
                   default=INPUT_NORMALIZE,
                   help="fixed per-feature standardization of the raw input u_t "
                        "(statistics estimated once from a task sample, then frozen "
                        "and applied identically to every rule + validation). "
                        "Conditions all rules equally by removing a scale artifact. "
                        "Use --no-input-normalize to force off. Default: %(default)s.")

    g = p.add_argument_group(
        "learning algorithm",
        "local update = learning signal x eligibility; the input embedding is a third, "
        "independent choice. bptt is full BPTT of the main loss whatever the signal, "
        "except that --input-mode three_factor makes its embedding a hybrid.")
    g.add_argument("--rules", nargs="+", choices=list(RULE_LABEL), default=None,
                   help="ELIGIBILITY axis — how MP parameters are updated: local_direct "
                        "(current activity and modulation, no history), local_diag_rflo "
                        "(diagonal / same-synapse eligibility), local_exact_rowlocal "
                        "(full intra-layer row-local eligibility), bptt (full BPTT of the "
                        f"main loss). Default: {RULES_TO_RUN}; --dfa defaults to bptt + "
                        "row-local + diagonal; --input-mode paired to bptt + diagonal + "
                        "direct.")
    g.add_argument("--learning-signal",
                   choices=[*SIGNAL_MODES, "global"], default=None,
                   help="SIGNAL axis — where each MP layer's learning signal comes from "
                        "(dmpn; see SIGNAL_MODES): exact_spatial = the main readout error "
                        "through the TRUE weights at every boundary (weight transport); "
                        "layerwise_fa = the main error through a FIXED RANDOM matrix at "
                        "every boundary; dfa = the main error projected DIRECTLY to every "
                        "layer through its own random matrix; local_readout = every "
                        "non-top MP layer learns from its OWN auxiliary head trained on "
                        "the task loss (nothing descends from upper layers; the top layer "
                        "keeps W_output; module 0 = embedding + first MP layer); mixed = "
                        "exact_spatial + alpha * local head. The local modes need "
                        "--cross-layer-steps 0 and cannot use --input-mode exact; a "
                        "single MP layer has no head, so local_readout then equals "
                        "exact_spatial. 'global' is the legacy spelling that defers to "
                        "--feedback. Default: the module's FEEDBACK_MODE/LEARNING_SIGNAL "
                        f"({signal_mode_from(LEARNING_SIGNAL, FEEDBACK_MODE)}).")
    g.add_argument("--input-mode",
                   choices=["match", "exact", "three_factor", "diag_mtrace", "paired"],
                   default=None,
                   help="EMBEDDING axis — how the trainable input embedding is updated "
                        "(dmpn), per rule: match = BPTT under bptt, direct three-factor "
                        "under every local rule (default: bptt stays full BPTT, local runs "
                        "stay local); exact = always the BPTT gradient (a HYBRID for local "
                        "runs: one extra BPTT pass; not allowed with local signals); "
                        "three_factor = always the direct three-factor rule (a HYBRID "
                        "bptt baseline); diag_mtrace = first-MP-layer modulation-column "
                        "traces for local rules, BPTT unchanged; paired = bptt exact, "
                        "local_direct three_factor, local_diag_rflo diag_mtrace (rejects "
                        "local_exact_rowlocal). diag_mtrace/paired need dmpn and the "
                        f"exact_spatial pathway. Default: {INPUT_MODE} (an 'exact' module "
                        "default switches to match under a local signal); --dfa selects "
                        "match.")
    g.add_argument("--local-bias-mode", choices=["exact", "direct", "match"], default=None,
                   help="MP bias eligibility for the row-local / diagonal rules: direct = "
                        "phi' only (no bias trace), exact = the row-local bias trace "
                        "(needed for local_exact_rowlocal's top-layer bias exactness). "
                        "match = direct under local_diag_rflo, exact under "
                        "local_exact_rowlocal. local_direct always uses direct; bptt "
                        "always uses autograd. Default: "
                        f"{LOCAL_BIAS_MODE}; --dfa selects direct.")
    g.add_argument("--cross-layer-steps", type=int, choices=[0, 1], default=None,
                   help="depth of the cross-layer TEMPORAL correction to the local rules "
                        "(dmpn, multi-MP-layer): 0 = pure same-time surrogate; 1 = add "
                        "the exact one-temporal-hop term (exact vs BPTT for MP layers at "
                        "T=2 only with exact_spatial and exact eligibility; no general "
                        "improvement guarantee). Uses the TRUE forward weights, so it is "
                        "rejected with layerwise_fa / dfa and with the local signals. "
                        f"Applies to every non-bptt rule. Default: {CROSS_LAYER_STEPS}.")

    g = p.add_argument_group("approximation knobs (each acts on ONE rule or mode)")
    g.add_argument("--rflo-trace-rho", type=float, default=RFLO_TRACE_RHO,
                   help="optional diagonal-RFLO trace-gain cap, 0 < rho < 1. Acts ONLY on "
                        "local_diag_rflo's MP-WEIGHT A traces (not the drive term, bias "
                        "traces, input traces, or any other rule). Default: disabled.")
    g.add_argument("--local-signal-alpha", type=float, default=LOCAL_SIGNAL_ALPHA,
                   help="weight of the local head signal under --learning-signal mixed "
                        "ONLY (ignored otherwise). Default: %(default)s.")

    g = p.add_argument_group(
        "forward dynamics (change the network itself, so EVERY rule including bptt is affected)")
    g.add_argument("--lam", type=float, default=LAM,
                   help="fixed modulation decay, 0 <= lambda < 1, in every MP layer; "
                        "default uses m_time_scale=4000 (lambda=0.99 at dt=40)")
    g.add_argument("--modulation-mode", choices=["none", "hard", "scaled_tanh"],
                   default=MODULATION_MODE if MODULATION_BOUNDS or MODULATION_MODE == "scaled_tanh" else "none",
                   help="modulation write: unbounded, hard clipping, or B*tanh(S/B)")
    g.add_argument("--modulation-bound", type=float, default=MODULATION_BOUND,
                   help="positive bound/scale B for hard or scaled_tanh (default: %(default)s)")

    g = p.add_argument_group("training and logging")
    g.add_argument("--lr", type=float, default=LR,
                   help="base Adam learning rate, including the main readout "
                        "(default: %(default)s)")
    g.add_argument("--head-lr-mult", type=float, default=HEAD_LR_MULT,
                   help="auxiliary readout learning rate = --lr times this multiplier. "
                        "Applies to local_readout/mixed heads under local rules only; "
                        "ignored by bptt and models without auxiliary heads. Does not "
                        "scale the local loss or learning signal. Default: %(default)s.")
    g.add_argument("--lr-schedule", choices=["plateau", "constant"], default=LR_SCHEDULE,
                   help="plateau: ReduceLROnPlateau on main validation loss (legacy); "
                        "constant: keep initial main/head rates throughout training, "
                        "independent of validation loss. Applies to every rule, "
                        "including bptt. Default: %(default)s.")
    g.add_argument("--steps", type=int, default=N_DATASETS, help="training batches")
    g.add_argument("--batch-size", type=int, default=BATCH,
                   help="training trials per update; validation uses 3 times this "
                        "many trials, evaluated in chunks (default: %(default)s)")
    g.add_argument("--runs", type=int, default=N_RUNS, help="independent seeds")
    g.add_argument("--seed", type=int, default=None,
                   help="starting seed (subsequent runs increment it). Default: a "
                        "random draw per invocation. Fix it to make two invocations "
                        "(e.g. --learning-signal exact_spatial vs local_readout) a PAIRED "
                        "comparison with identical init and training data.")
    g.add_argument("--grad-align", action=argparse.BooleanOptionalAction, default=None,
                   help="BPTT gradient-alignment diagnostic at record steps (one extra "
                        "autograd pass per local rule per record step). Default: "
                        f"{LOG_GRAD_ALIGN}; --dfa selects off.")
    g.add_argument("--wandb", dest="use_wandb", action="store_true", default=USE_WANDB,
                   help="log to Weights & Biases (https://wandb.ai): one run per "
                        "(rule × seed), all grouped under this run's output save-stem "
                        "as the experiment name. Group/color by 'rule' in the UI for "
                        "K=len(RULES_TO_RUN) colors, each seed a separate curve. "
                        "Off by default.")
    g.add_argument("--wandb-project", default=WANDB_PROJECT,
                   help="W&B project name (default: %(default)s).")
    g.add_argument("--wandb-entity", default=WANDB_ENTITY,
                   help="W&B entity (user/team); default: your W&B default.")
    g.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                   default=WANDB_MODE,
                   help="W&B mode; default: online. Use 'offline' to log locally and "
                        "`wandb sync` later (no login needed).")

    g = p.add_argument_group(
        "legacy options (still accepted; translated into the unified settings above)")
    g.add_argument("--dfa", action="store_true",
                   help="LEGACY PRESET for the manuscript's DFA comparison: "
                        "--learning-signal dfa plus input mode match, direct bias, rules "
                        "bptt + row-local + diagonal, cross-layer 0 and alignment off. "
                        "Prefer --learning-signal dfa, which changes only the signal.")
    g.add_argument("--feedback",
                   choices=["exact_spatial", "layerwise_fa", "direct_fa", "exact_readout"],
                   default=None,
                   help="LEGACY spelling of the signal pathway (exact_spatial / "
                        "layerwise_fa / direct_fa = --learning-signal exact_spatial / "
                        "layerwise_fa / dfa; 'exact_readout' is the old name for "
                        "exact_spatial). Must agree with --learning-signal when both are "
                        "given.")
    return p, p.parse_args(argv)


def resolve_defaults_and_legacy_options(p, args):
    """Stage 2 of 3: translate the legacy options (--dfa preset, --feedback,
    --learning-signal global) into the unified signal mode and fill every unset
    flag from the module defaults. Explicit flags always win; a flag that
    contradicts another is reported here or in validate_config, never resolved by
    argument order. Sets args.signal_mode and the model-facing pair
    (args.learning_signal, args.feedback)."""
    requested = args.learning_signal
    if args.dfa:
        if requested not in (None, "global", "dfa"):
            p.error(f"--dfa is the legacy DFA preset and conflicts with "
                    f"--learning-signal {requested}; drop one of them")
        if args.feedback is not None and mpn.canonical_feedback_mode(args.feedback) != "direct_fa":
            p.error("--dfa requires --feedback direct_fa, --input-mode match, "
                    "and --cross-layer-steps 0; match keeps BPTT exact and local runs local")
        requested = "dfa"
    elif requested is None or requested == "global":
        # Legacy spelling: the (legacy) --feedback flag, or the module defaults,
        # decide which unified mode is meant.
        feedback = mpn.canonical_feedback_mode(args.feedback or FEEDBACK_MODE)
        learning_signal = "global" if requested == "global" else LEARNING_SIGNAL
        requested = signal_mode_from(learning_signal, feedback)
        if requested not in SIGNAL_MODES:
            p.error(f"learning_signal={learning_signal} with feedback {feedback} is not a "
                    f"supported signal mode; choose --learning-signal from {list(SIGNAL_MODES)}")
    elif args.feedback is not None and \
            mpn.canonical_feedback_mode(args.feedback) != SIGNAL_MODES[requested][1]:
        p.error(f"--learning-signal {requested} implies feedback "
                f"{SIGNAL_MODES[requested][1]}; drop the legacy --feedback {args.feedback} "
                "or make them agree")
    args.signal_mode = requested
    args.learning_signal, args.feedback = SIGNAL_MODES[requested]
    local_signal = args.learning_signal != "global"

    if args.grad_align is None:
        args.grad_align = False if args.dfa else LOG_GRAD_ALIGN
    # A local signal swaps only the two module defaults that would conflict with it:
    # an 'exact' input mode (a BPTT splice is not local) and a non-zero cross-layer
    # correction (inter-module by construction). Explicit flags are left alone.
    default_input_mode = INPUT_MODE
    if local_signal and default_input_mode == "exact":
        default_input_mode = "match"
    default_cross = CROSS_LAYER_STEPS
    if local_signal and default_cross != 0:
        default_cross = 0
    args.input_mode = args.input_mode or ("match" if args.dfa else default_input_mode)
    if args.cross_layer_steps is None:
        args.cross_layer_steps = 0 if args.dfa else default_cross
    args.local_bias_mode = args.local_bias_mode or ("direct" if args.dfa else LOCAL_BIAS_MODE)
    if args.rules is None:
        args.rules = (["bptt", "local_exact_rowlocal", "local_diag_rflo"] if args.dfa
                      else ["bptt", "local_diag_rflo", "local_direct"]
                      if args.input_mode == 'paired' else list(RULES_TO_RUN))
    return args


def validate_config(p, args):
    """Stage 3 of 3: reject every unsupported combination with one clear message.
    Mirrors the model-level guards (so the error arrives before any net is built)
    and adds the CLI-only rules."""
    local_signal = args.learning_signal != "global"
    try:
        tc.validate_optim_options(args.lr, args.head_lr_mult, args.lr_schedule)
    except ValueError as exc:
        p.error(str(exc))
    if args.input_mode in ('diag_mtrace', 'paired') and (
            args.net != 'dmpn' or args.feedback != 'exact_spatial'):
        p.error(f"--input-mode {args.input_mode} requires --net dmpn and the exact_spatial "
                "signal pathway (--learning-signal exact_spatial, local_readout or mixed)")
    if args.input_mode == 'paired' and 'local_exact_rowlocal' in args.rules:
        p.error("--input-mode paired does not support local_exact_rowlocal; "
                "use match or an explicit input mode")
    if args.dfa and (args.feedback != "direct_fa" or args.input_mode != "match"
                     or args.cross_layer_steps != 0):
        p.error("--dfa requires --feedback direct_fa, --input-mode match, "
                "and --cross-layer-steps 0; match keeps BPTT exact and local runs local")
    if args.feedback == "direct_fa" and args.cross_layer_steps != 0:
        p.error("direct_fa requires --cross-layer-steps 0 (or use --dfa)")
    if args.feedback == "layerwise_fa" and args.cross_layer_steps != 0:
        p.error("--learning-signal layerwise_fa requires --cross-layer-steps 0: the depth-1 "
                "correction backprojects through the TRUE forward weights, which would make "
                "the run a hybrid of random feedback and weight transport; use "
                "exact_spatial if you want the correction")
    if local_signal:
        if args.net != "dmpn":
            p.error(f"--learning-signal {args.signal_mode} requires --net dmpn")
        if args.cross_layer_steps != 0:
            p.error(f"--learning-signal {args.signal_mode} requires --cross-layer-steps 0")
        if args.input_mode == "exact":
            p.error(f"--learning-signal {args.signal_mode} cannot use --input-mode exact "
                    "(a BPTT input splice is not local); use match, three_factor, "
                    "diag_mtrace or paired")
        if not np.isfinite(args.local_signal_alpha):
            p.error("--local-signal-alpha must be finite")
    if not np.isfinite(args.modulation_bound) or args.modulation_bound <= 0:
        p.error("--modulation-bound must be finite and positive")
    if args.rflo_trace_rho is not None and (
            not np.isfinite(args.rflo_trace_rho) or not 0 < args.rflo_trace_rho < 1):
        p.error("--rflo-trace-rho must be finite and strictly between 0 and 1")
    if args.lam is not None and (not np.isfinite(args.lam) or not 0 <= args.lam < 1):
        p.error("--lam must be finite and satisfy 0 <= lambda < 1")
    if args.batch_size <= 0:
        p.error("--batch-size must be a positive integer")
    return args


def _parse_args(argv=None):
    """parse → resolve defaults/legacy → validate. argv=None reads sys.argv."""
    p, args = parse_arguments(argv)
    args = resolve_defaults_and_legacy_options(p, args)
    return validate_config(p, args)


def main():
    global NET_TYPE, RULESET, N_RUNS, N_HIDDEN, N_DATASETS, FEEDBACK_MODE, BATCH
    global INPUT_MODE, INPUT_NORMALIZE, MP_RESIDUAL, CROSS_LAYER_STEPS, LOCAL_BIAS_MODE, RULES_TO_RUN, LOG_GRAD_ALIGN
    global USE_WANDB, WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE
    global MODULATION_MODE, MODULATION_BOUND, MODULATION_BOUNDS
    global DFA_PRESET
    global RFLO_TRACE_RHO, LAM
    global SEED, LEARNING_SIGNAL, LOCAL_SIGNAL_ALPHA
    global LR, HEAD_LR_MULT, LR_SCHEDULE
    args = _parse_args()
    DFA_PRESET = args.dfa
    NET_TYPE = args.net
    RULESET = args.task
    N_RUNS = args.runs
    if args.seed is not None:
        SEED = args.seed
    LEARNING_SIGNAL = args.learning_signal
    LOCAL_SIGNAL_ALPHA = args.local_signal_alpha
    if args.hidden is not None:
        # one int → scalar (single MP layer); several → list (deep MP stack).
        N_HIDDEN = args.hidden[0] if len(args.hidden) == 1 else args.hidden
    N_DATASETS = args.steps
    BATCH = args.batch_size
    LR = args.lr
    HEAD_LR_MULT = args.head_lr_mult
    LR_SCHEDULE = args.lr_schedule
    FEEDBACK_MODE = args.feedback
    INPUT_MODE = args.input_mode
    LOCAL_BIAS_MODE = args.local_bias_mode
    RFLO_TRACE_RHO = args.rflo_trace_rho
    LAM = args.lam
    MODULATION_MODE = args.modulation_mode
    MODULATION_BOUND = args.modulation_bound
    MODULATION_BOUNDS = args.modulation_mode == "hard"
    RULES_TO_RUN = args.rules
    LOG_GRAD_ALIGN = args.grad_align
    INPUT_NORMALIZE = args.input_normalize
    MP_RESIDUAL = args.residual
    CROSS_LAYER_STEPS = args.cross_layer_steps
    USE_WANDB = args.use_wandb
    WANDB_PROJECT = args.wandb_project
    WANDB_ENTITY = args.wandb_entity
    WANDB_MODE = args.wandb_mode
    cfg = _cfg()
    # Give the console log the run ID shared by the figure / .npz / checkpoint
    # folder (log/<run-id>.log); a no-op when no tee is active (tests, imports).
    rename_active_log(tc.run_stem(cfg))
    tc.run_experiment(cfg)


if __name__ == "__main__":
    with tee_output("train_mpn"):
        main()
