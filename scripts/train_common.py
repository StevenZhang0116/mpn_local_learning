#!/usr/bin/env python
# coding: utf-8
"""
Shared training / plotting machinery for train_mpn.py and train_rnn.py.

The two train scripts are identical except for a handful of model-specific bits
(the param dicts, the network class, how the held-out forward is run, and
cosmetic labels/titles). Everything else — the lockstep multi-rule training
loop, accuracy logging, figure, .npz plot-data save, and replot — lives here and
is driven by a `RunConfig` the caller assembles from its module globals.

A caller builds a fresh RunConfig each call (so notebooks/validate can override
globals like N_HIDDEN before calling ckpt_path/build_params) and delegates:

    import train_common as tc
    def _cfg(): return tc.RunConfig(seed=SEED, ..., build_params=build_params, ...)
    def fig_path():   return tc.fig_path(_cfg())
    def main():       tc.run_experiment(_cfg())
"""
import copy
import gc
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless: write PNG, no display
import matplotlib.pyplot as plt

import tasks  # Task adapters (data/metric seam); make_task(ruleset) picks one
from mpn import masked_mse_loss_and_output_grad, resolve_input_mode, resolve_local_bias_mode


@dataclass
class RunConfig:
    """All knobs + hooks for one train/compare experiment. Callers build this
    fresh from their module globals (see _cfg() in the train scripts)."""
    # identity / presentation
    file_prefix: str            # legacy figure/data prefix (when run_id is empty)
    ckpt_prefix: str            # legacy checkpoint prefix; empty disables net saving
    title: str                  # figure suptitle prefix (before the mean±std note)
    header_note: str            # extra text in the console header, e.g. " (leaky RNN)"
    rule_label: dict
    rule_color: dict
    # scalar knobs
    seed: int
    ruleset: str
    rules_to_run: list
    feedback_mode: str
    n_runs: int
    n_hidden: int
    batch: int
    n_datasets: int
    lr: float
    grad_clip: float
    log_every: int
    device: torch.device
    dtype: torch.dtype
    # output dirs
    fig_dir: str
    ckpt_dir: str
    data_dir: str
    save_nets: bool
    # model-specific hooks
    build_params: Callable      # () -> (task_params, train_params, net_params)
    net_factory: Callable       # (net_params, verbose) -> net
    eval_outputs: Callable      # (net, inputs) -> outputs (B, T, n_output), no grad
    # Nonempty opts into compact output names and run/seed checkpoint folders.
    # Callers keep this ID stable across path-helper calls for one experiment.
    run_id: str = ""
    # CLI provenance, distinct from whether an individual model uses DFA.
    dfa_preset: bool = False
    # Task adapter (the data/metric seam): provides init_params / valid_batch /
    # train_batch / accuracy. Defaults to None, resolved to tasks.make_task(ruleset)
    # in run_seed so callers that don't set it keep the ring-task behaviour.
    task: object = None
    # Input-embedding learning rule (dmpn), decoupled from the per-rule MP-layer
    # rule (see mpn.DeepMultiPlasticNet input_mode). Recorded in metadata and logs;
    # the net reads it from net_params. Legacy filenames also include this setting.
    # 'paired' is resolved per rule by the model after cloning.
    input_mode: str = "match"
    # Fixed input standardization (see mpn.MultiPlasticNetBase.set_input_norm_stats).
    # When on, run_seed estimates per-feature input mean/std from a task sample of
    # input_norm_sample trials and freezes them into the base net (shared by every
    # rule via the deepcopy) BEFORE training. Recorded in metadata and the
    # console/figure note (also an "_inorm" fragment in legacy filenames).
    input_normalize: bool = False
    input_norm_sample: int = 2048
    # Per-step RMS normalization of every MP layer's presynaptic input (see
    # mpn.MultiPlasticNetBase mp_input_norm). Recorded in metadata and the
    # console/figure notes; the net reads it from net_params. 'none' = off.
    mp_input_norm: str = "none"
    mp_input_norm_eps: float = 1e-5
    # Identity skip (residual) connections around each equal-width MP block (dmpn;
    # see mpn.DeepMultiPlasticNet mp_residual). Recorded in metadata and console/
    # figure notes; the net reads it from net_params. Legacy names include "_res".
    mp_residual: bool = False
    residual_scale: float = 1.0
    # Cross-layer temporal correction depth for the local rules (dmpn; see
    # mpn.DeepMultiPlasticNet cross_layer_steps). Recorded in metadata and figure
    # notes; the net reads it from net_params. Legacy names include "_xl{k}".
    cross_layer_steps: int = 0
    # Learning-signal SOURCE for the local rules (dmpn; see mpn._LEARNING_SIGNALS):
    # 'global' (default, unchanged) | 'local_readout' (per-layer auxiliary heads) |
    # 'mixed' (global + local_signal_alpha × local). Recorded in metadata and the
    # console/figure notes; the net reads it from net_params. Under the local modes
    # run_seed clips the head gradients SEPARATELY from the main parameters and logs
    # each head's train loss/accuracy next to the main readout's (which stay the
    # plotted metrics and the plateau monitor). Legacy names include "_ls-{signal}".
    learning_signal: str = "global"
    local_signal_alpha: float = 1.0
    # Auxiliary heads only; the main readout remains at lr. BPTT ignores this
    # multiplier because its auxiliary heads never participate in the loss.
    head_lr_mult: float = 1.0
    # plateau: legacy main-validation-loss scheduling; constant: no scheduler.
    lr_schedule: str = "plateau"
    # ReduceLROnPlateau knobs (plateau only): multiply every group's lr by lr_factor
    # after lr_patience consecutive non-improving validation steps. The defaults are
    # the historical values; a larger patience decays more slowly.
    lr_patience: int = 30
    lr_factor: float = 0.95
    # Unified name of the (learning_signal, feedback_mode) pair the run uses — the
    # CLI's --learning-signal value (train_mpn.SIGNAL_MODES: exact_spatial /
    # layerwise_fa / dfa / local_readout / mixed). Recorded next to the two model
    # fields in every metadata sink; empty when the caller does not set it.
    signal_mode: str = ""
    # Optional extra string appended to the legacy filename tag (e.g. eta/lambda);
    # keep it filename-safe. Empty by default.
    tag_extra: str = ""
    # Architecture metadata label, also used in legacy filenames. Empty uses
    # the scalar n_hidden fallback; full architecture is always in net_params.
    arch_tag: str = ""
    # Human-readable architecture for the figure title / suffix and .npz provenance
    # (e.g. "arch=[20, 200, 150, 100, 3]"). Empty → the suffix falls back to
    # "hidden={n_hidden}" as before.
    arch_desc: str = ""
    # y-axis label for the plotted panels (task-dependent: accuracy label for the
    # ring tasks / seq-MNIST, loss label for the adding problem).
    acc_label: str = "angle accuracy (%)"
    # Which curve the figure shows: 'accuracy' (percent, 0-110 y) or 'loss'
    # (masked-MSE, log-y). Regression tasks like the adding problem use 'loss'
    # because accuracy is uninformative there. Accuracy is always logged either way.
    metric: str = "accuracy"
    # At each recorded step, also log the per-weight-matrix cosine similarity
    # between every non-BPTT rule's gradient and the TRUE BPTT gradient at the same
    # weights (a "how well does this local rule align with the exact gradient?"
    # diagnostic). Costs one extra BPTT pass per local rule per recorded step
    # (record steps only, not every step), timed OUTSIDE the fwd/bwd/opt readout.
    log_grad_align: bool = True
    # ─ Weights & Biases logging (opt-in; see scripts/wandb_logging.py) ─
    # When use_wandb is True, run_seed opens ONE W&B run per (rule × seed) grouped
    # under the run's output save-stem (run_stem, the figure/.npz name), so
    # a single invocation shows K = len(rules_to_run) colors (group/color by 'rule')
    # with each of the n_runs seeds as its own curve; a summary run logs the
    # aggregate figures. False (the default) does not import wandb or create W&B runs.
    # wandb_project/entity/mode/dir
    # /group/tags are forwarded to wandb.init (None → wandb's own defaults; mode None
    # → online).
    use_wandb: bool = False
    wandb_project: str = "mpn_local_learning"
    wandb_entity: str = None
    wandb_mode: str = None          # None → wandb default (online); "offline"/"online"/"disabled"
    wandb_dir: str = None           # None → wandb default (./wandb)
    wandb_group: str = None         # None → run_stem (the output save-stem)
    wandb_tags: list = None


# ─── Path helpers (read the passed cfg, i.e. the caller's live globals) ───────
def param_tag(cfg):
    """Detailed parameter tag for legacy flat outputs (no run_id)."""
    # Full stack label when available; otherwise use the scalar hidden width.
    arch = cfg.arch_tag if getattr(cfg, "arch_tag", "") else f"h{cfg.n_hidden}"
    tag = (f"{cfg.ruleset}_{arch}_b{cfg.batch}_n{cfg.n_datasets}"
           f"_lr{cfg.lr:.0e}_{cfg.feedback_mode}")
    if getattr(cfg, "input_normalize", False):
        tag += "_inorm"
    if getattr(cfg, "mp_input_norm", "none") != "none":
        tag += f"_mpnorm-{cfg.mp_input_norm}"
    if getattr(cfg, "mp_residual", False):
        tag += "_res"
        if getattr(cfg, "residual_scale", 1.0) != 1.0:
            tag += f"-scale{cfg.residual_scale:g}"
    if getattr(cfg, "cross_layer_steps", 0):
        tag += f"_xl{cfg.cross_layer_steps}"
    # Legacy filenames record the input-embedding rule, including default 'match'.
    tag += f"_in-{getattr(cfg, 'input_mode', 'match')}"
    if getattr(cfg, "learning_signal", "global") != "global":
        tag += f"_ls-{cfg.learning_signal}"
    if cfg.head_lr_mult != 1.0:
        tag += f"_headlr{cfg.head_lr_mult:g}"
    if cfg.lr_schedule != "plateau":
        tag += f"_sched-{cfg.lr_schedule}"
    elif (getattr(cfg, "lr_patience", 30), getattr(cfg, "lr_factor", 0.95)) != (30, 0.95):
        tag += f"_pat{cfg.lr_patience}_fac{cfg.lr_factor:g}"
    if cfg.tag_extra:
        tag += f"_{cfg.tag_extra}"
    return tag


def run_stem(cfg):
    """Shared figure/data stem and W&B group; use legacy naming without run_id."""
    if cfg.run_id:
        return cfg.run_id
    return f"{cfg.file_prefix}_{param_tag(cfg)}_runs{cfg.n_runs}"


def fig_path(cfg):
    return os.path.join(cfg.fig_dir, f"{run_stem(cfg)}.png")


def data_path(cfg):
    return os.path.join(cfg.data_dir, f"{run_stem(cfg)}.npz")


def config_path(cfg):
    """Compact saved-net runs use <run-id>/config.json; otherwise beside the figure."""
    if cfg.run_id and cfg.save_nets and cfg.ckpt_prefix:
        return os.path.join(cfg.ckpt_dir, cfg.run_id, "config.json")
    return os.path.join(cfg.fig_dir, f"{run_stem(cfg)}.json")


def align_fig_path(cfg):
    """Gradient-alignment-vs-BPTT figure path (figure dir, '_gradalign' suffix)."""
    return os.path.join(cfg.fig_dir,
                        f"{run_stem(cfg)}_gradalign.png")


def ckpt_path(cfg, rule, seed):
    if cfg.run_id:
        return os.path.join(cfg.ckpt_dir, cfg.run_id, f"seed{seed}", f"{rule}.pt")
    return os.path.join(cfg.ckpt_dir, f"{cfg.ckpt_prefix}_{param_tag(cfg)}_{rule}_seed{seed}.pt")


def uses_dfa(feedback_mode, rule, input_mode="match", trainable_embed=False):
    """Whether parameter updates use DFA (including a hybrid BPTT input splice)."""
    return feedback_mode == "direct_fa" and (
        rule != "bptt" or (input_mode == "three_factor" and trainable_embed))


def arch_suffix(cfg):
    """Architecture note for the figure title/suffix: the full stack when known
    (cfg.arch_desc), else the historical single scalar 'hidden={n_hidden}'."""
    return cfg.arch_desc if getattr(cfg, "arch_desc", "") else f"hidden={cfg.n_hidden}"


# ─── Gradient-alignment diagnostic (local rule vs exact BPTT) ─────────────────
# Which weight matrices to report alignment for, in log order. Keys match the
# gradient dicts from sequence_gradients / bptt_gradients. Biases (keys starting
# 'b') are intentionally skipped. Handles both models:
#   MPN — 'W_in' (dmpn input embedding), 'W'/'W1'/'W2'/... (each MP layer's plastic
#         weight; layer 0 is the bare 'W'), 'W_output' (readout).
#   RNN — 'W_input', 'W_rec', 'W_output'.
def _grad_align_keys(grad_dict):
    """Weight-matrix keys present in grad_dict, ordered input → hidden → output.
    Every key starting with 'W' (biases 'b*' and 'loss'/'outputs' skipped)."""
    w_keys = [k for k in grad_dict if k.startswith("W")]

    def order(k):
        # input-side first, output last, hidden weights (incl numbered MP layers
        # and W_rec) in the middle by their numeric suffix (bare 'W' → 0).
        if k in ("W_in", "W_input"):
            return (0, 0, k)
        if k == "W_output":
            return (2, 0, k)
        if k == "W":
            return (1, 0, k)
        if k[1:].isdigit():                 # 'W1', 'W2', ... (MP-layer index)
            return (1, int(k[1:]), k)
        return (1, 0, k)                    # 'W_rec' and any other hidden weight
    return sorted(w_keys, key=order)


def cosine_alignment(g_local, g_ref, keys):
    """Per-key cosine similarity between two gradient dicts (flattened tensors).
    Returns {key: cos in [-1, 1]} (nan if either gradient is ~0). Cosine, not raw
    error, so the scale differences between rules/layers don't confound it."""
    out = {}
    for k in keys:
        a, b = g_local.get(k), g_ref.get(k)
        if a is None or b is None:
            out[k] = float("nan")
            continue
        a = a.reshape(-1).to(torch.float64)
        b = b.reshape(-1).to(torch.float64)
        na, nb = a.norm(), b.norm()
        out[k] = float(torch.dot(a, b) / (na * nb)) if (na > 0 and nb > 0) else float("nan")
    return out


def bptt_reference_grads(net, inputs, labels, mask, loss_kw):
    """Exact BPTT gradients for `net` at its CURRENT weights, as a detached
    {name: grad} dict — the reference the local rules are compared against. Must
    run with autograd ENABLED (bptt_gradients builds the backward graph), so this
    is deliberately NOT wrapped in torch.no_grad. Restores net.learning_rule + the
    net's _bptt timing attributes afterwards so this diagnostic call perturbs
    neither the rule dispatch nor the fwd/bwd ms readout of the real update."""
    _MISSING = object()
    saved_rule = net.learning_rule
    saved_fwd = getattr(net, "_bptt_fwd_s", _MISSING)
    saved_bwd = getattr(net, "_bptt_bwd_s", _MISSING)
    try:
        # No return_outputs kwarg: the MPN bptt_gradients accepts it but the RNN's
        # does not, and the returned 'outputs' is dropped below anyway — keep this
        # helper model-agnostic.
        ref = net.bptt_gradients(inputs, labels, mask, **loss_kw)
    finally:
        net.learning_rule = saved_rule
        # bptt_gradients SETS _bptt_fwd_s/_bwd_s. Fully restore the prior state so
        # the training loop's timing split (which reads these right after
        # sequence_gradients) is unaffected: on a LOCAL-rule net they were absent,
        # so delete them (else a stale reference timing would leak into the next
        # step and be wrongly charged to the local rule as bwd time).
        for attr, saved in (("_bptt_fwd_s", saved_fwd), ("_bptt_bwd_s", saved_bwd)):
            if saved is _MISSING:
                if hasattr(net, attr):
                    delattr(net, attr)
            else:
                setattr(net, attr, saved)
    return {k: v for k, v in ref.items() if k not in ("loss", "outputs")}


# ─── Training pieces ──────────────────────────────────────────────────────────
def try_accuracy(task, net, output, labels, mask, inputs, isvalid=False):
    """Accuracy via the task adapter; returns nan on failure. train uses
    isvalid=False (current batch), validation isvalid=True. The task decides the
    metric (angle for ring tasks, argmax-correct for seq-MNIST)."""
    try:
        return float(task.accuracy(net, output, labels, mask, inputs, isvalid=isvalid))
    except Exception:
        return float("nan")


def aux_param_split(net, trainable):
    """Split `trainable` into (main, aux): aux = the net's local readout head
    parameters (mpn.DeepMultiPlasticNet._aux_params; empty for nets without heads
    or without the method), main = everything else. Used to clip the two groups
    SEPARATELY so head gradients never change the main network's clipped step."""
    aux_fn = getattr(net, "_aux_params", None)
    aux_ids = {id(p) for p in aux_fn().values()} if aux_fn is not None else set()
    main = [p for p in trainable if id(p) not in aux_ids]
    aux = [p for p in trainable if id(p) in aux_ids]
    return main, aux


def effective_rule_summary(cfg, net, rule):
    """One line describing what `rule` ACTUALLY runs on `net`, for the console and
    the per-rule provenance: MP update, signal mode + feedback pathway, bias
    eligibility, the RFLO trace cap, and whether local readout heads are active.
    Separates 'which signal was selected' from 'is this run fully local'."""
    if rule == "bptt":
        # bptt's MP layers and readout are always full BPTT. Its EMBEDDING is too,
        # except under input_mode three_factor, where the direct 3-factor rule is
        # spliced in under the GLOBAL signal (never a local head) through the net's
        # feedback pathway — a HYBRID, so the feedback mode IS used there. Label it
        # as such so the summary never calls a hybrid a full-BPTT baseline.
        resolved = getattr(net, "resolved_input_mode", None)
        has_embed = getattr(net, "_has_trainable_embed", lambda: False)()
        if has_embed and resolved == "three_factor":
            return ("algorithm=hybrid, mp_update=BPTT, input_update=three_factor, "
                    "input_signal=global, "
                    f"input_feedback={getattr(net, 'feedback_mode', 'n/a')}, heads=unused")
        return "mp_update=full BPTT (feedback/bias/heads unused)"
    layers = getattr(net, "mp_layers", [])
    signal = getattr(cfg, "signal_mode", "") or getattr(net, "learning_signal", "global")
    parts = [f"signal={signal}", f"feedback={getattr(net, 'feedback_mode', 'n/a')}"]
    if rule == "local_direct":
        parts.append("bias=direct (rule-fixed)")
    else:
        bias = sorted({resolve_local_bias_mode(mp.local_bias_mode, rule) for mp in layers})
        parts.append(f"bias={'/'.join(bias) if bias else 'n/a'}")
    if rule == "local_diag_rflo":
        rho = sorted({mp.rflo_trace_rho for mp in layers
                      if getattr(mp, "rflo_trace_rho", None) is not None})
        parts.append(f"rho={rho[0]:g}" if rho else "rho=none")
    n_heads = len(getattr(net, "_head_names", []))
    parts.append(f"heads={n_heads} active" if n_heads else "heads=none")
    return ", ".join(parts)


def validate_optim_options(lr, head_lr_mult, lr_schedule, lr_patience=30, lr_factor=0.95):
    """Shared CLI/programmatic validation, before constructing an optimizer."""
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if not np.isfinite(head_lr_mult) or head_lr_mult <= 0:
        raise ValueError("--head-lr-mult must be finite and positive")
    if not np.isfinite(lr * head_lr_mult) or lr * head_lr_mult <= 0:
        raise ValueError("--lr * --head-lr-mult must be finite and positive")
    if lr_schedule not in ("plateau", "constant"):
        raise ValueError("--lr-schedule must be plateau or constant")
    if int(lr_patience) != lr_patience or lr_patience < 0:
        raise ValueError("--lr-patience must be a nonnegative integer")
    if not np.isfinite(lr_factor) or not 0 < lr_factor < 1:
        raise ValueError("--lr-factor must satisfy 0 < factor < 1")


def make_optim(net, lr, weight_decay=0.0, *, head_lr_mult=1.0,
               lr_schedule="plateau", lr_patience=30, lr_factor=0.95):
    """Adam with coupled L2: (weight_decay / 2) * sum(W**2), excluding biases.
    Weight matrices = every _trainable_params key starting with 'W' plus the local
    readout heads' weights ('head_W*'), which are readouts like W_output.
    The default keeps the original parameter-group layout and scheduler exactly.
    Head-only LR overrides never affect main parameters or a BPTT baseline.
    Returns (trainable, optimizer, scheduler); constant returns scheduler=None.
    lr_patience / lr_factor are the plateau scheduler's patience (steps without a
    new best validation loss) and decay factor; the defaults are the historical
    30 / 0.95, and --lr-schedule constant ignores both.
    """
    validate_optim_options(lr, head_lr_mult, lr_schedule, lr_patience, lr_factor)
    trainable = [p for p in net.parameters() if p.requires_grad]
    if weight_decay:
        weight_ids = {id(parameter) for name, parameter in net._trainable_params().items()
                      if name.startswith('W')}
        aux_fn = getattr(net, "_aux_params", None)
        if aux_fn is not None:
            weight_ids |= {id(parameter) for name, parameter in aux_fn().items()
                           if name.startswith('head_W')}
        groups = [
            {'params': [parameter for parameter in trainable if id(parameter) in weight_ids],
             'weight_decay': weight_decay},
            {'params': [parameter for parameter in trainable if id(parameter) not in weight_ids],
             'weight_decay': 0.0},
        ]
    else:
        groups = trainable
    aux_fn = getattr(net, "_aux_params", None)
    active_heads = (aux_fn() if aux_fn is not None
                    and getattr(net, "learning_rule", None) != "bptt" else {})
    head_ids = {id(p) for p in active_heads.values() if p.requires_grad}
    if head_ids and head_lr_mult != 1.0:
        # Split EACH decay group by role, preserving weight-vs-bias decay policy.
        # Main groups stay first, so param_groups[0]['lr'] keeps its old meaning.
        old_groups = groups if weight_decay else [{'params': trainable}]
        groups = []
        for is_head in (False, True):
            for group in old_groups:
                params = [p for p in group['params'] if (id(p) in head_ids) == is_head]
                if params:
                    groups.append({**group, 'params': params,
                                   'lr': lr * head_lr_mult if is_head else lr})
    opt = torch.optim.Adam(groups, lr=lr)
    sch = None
    if lr_schedule == "plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=lr_factor, patience=int(lr_patience),
            min_lr=[1e-8 * (group['lr'] / lr) for group in opt.param_groups])
    return trainable, opt, sch


def learning_rate_snapshot(net, opt):
    """Effective rates by trainable weight matrix; omit unused BPTT heads.

    MP names follow _trainable_params (W, W1, ...); W_in is the embedding and
    W_output the main readout. Biases share their associated matrix's rate.
    """
    by_id = {id(p): group['lr'] for group in opt.param_groups for p in group['params']}
    params = {name: p for name, p in net._trainable_params().items()
              if name.startswith('W')}
    if getattr(net, 'learning_rule', None) != 'bptt':
        aux_fn = getattr(net, '_aux_params', None)
        if aux_fn is not None:
            params.update({name: p for name, p in aux_fn().items()
                           if name.startswith('head_W')})
    return {name: float(by_id[id(p)]) for name, p in params.items() if id(p) in by_id}


@torch.no_grad()
def eval_outputs_chunked(cfg, net, v_inputs, chunk):
    """Run the held-out forward in batch-dim chunks of size `chunk` and concatenate
    the outputs, to cap peak memory when valid_n_batch > chunk (avoids CUDA OOM on
    the unrolled validation forward, whose per-step M/eligibility tensors scale with
    the batch). Validation samples never interact (cfg.eval_outputs calls
    reset_state(B=chunk) per chunk), so the concatenated outputs equal a single
    cfg.eval_outputs(net, v_inputs) call up to float round-off — the SAME
    computation, just tiled differently (bit-exact in float64; ~1e-7 in float32
    from kernel tiling), so downstream loss/accuracy are unchanged. When chunking
    is off or the whole set fits (chunk is None, or B not greater than chunk),
    it is a single call (bit-identical)."""
    B = v_inputs.shape[0]
    if chunk is None or B <= chunk:
        return cfg.eval_outputs(net, v_inputs)
    outs = [cfg.eval_outputs(net, v_inputs[i:i + chunk]) for i in range(0, B, chunk)]
    return torch.cat(outs, dim=0)


def run_seed(cfg, seed, record_steps, run_idx=0, wandb_logger=None):
    """Train all cfg.rules_to_run in lockstep for one seed: identical init
    (deepcopy of one base net), identical per-step data, identical held-out
    validation set. Returns (curves, align_curves):
      curves[rule][split]        -> list over record_steps (train/valid metric)
      align_curves[rule][key]    -> list over record_steps of the cosine similarity
                                    between this (non-bptt) rule's gradient and the
                                    exact BPTT gradient for weight `key` ({} if the
                                    alignment diagnostic is off / no non-bptt rules).
    Saves each trained net if cfg.save_nets. If wandb_logger is given (cfg.use_wandb),
    every recorded step is also logged to its K per-rule W&B runs for this seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Task adapter (data/metric seam). Defaults to the ring-task pipeline.
    task = cfg.task if cfg.task is not None else tasks.make_task(cfg.ruleset)

    task_params, train_params, net_params = cfg.build_params()
    task_params, train_params, net_params = task.init_params(
        task_params, train_params, net_params)

    # One base net → deepcopy so every rule starts from the SAME weights.
    base = cfg.net_factory(net_params, seed == cfg.seed).to(cfg.device).to(cfg.dtype)

    # Fixed input standardization: estimate per-feature input mean/std from a task
    # sample ONCE and freeze them into the base net BEFORE the deepcopy, so every
    # rule inherits the SAME frozen statistics (they must not differ across rules).
    # Applied identically to train + validation inside each net's gradient/eval path.
    if getattr(cfg, "input_normalize", False):
        sample_in, _, _ = task.train_batch(
            task_params, train_params, cfg.input_norm_sample, cfg.device, cfg.dtype)
        base.set_input_norm_stats(sample_in)

    weight_decay = (float(train_params.get('reg_lambda', 0.0))
                    if train_params.get('weight_reg') == 'L2' else 0.0)
    if weight_decay:
        print(f"  L2 weight regularization: {weight_decay:g} (Adam coupled decay; biases excluded)")
    nets, optims, clip_groups = {}, {}, {}
    for rule in cfg.rules_to_run:
        net = copy.deepcopy(base)
        net.learning_rule = rule
        resolved_input = getattr(net, 'resolved_input_mode', None)
        if resolved_input is not None:
            # Effective per-rule configuration (what this rule actually runs).
            print(f"  {rule}: input_mode={net.input_mode}, resolved_input_mode={resolved_input}, "
                  + effective_rule_summary(cfg, net, rule))
        nets[rule] = net
        optims[rule] = make_optim(net, cfg.lr, weight_decay=weight_decay,
                                 head_lr_mult=cfg.head_lr_mult,
                                 lr_schedule=cfg.lr_schedule,
                                 lr_patience=getattr(cfg, "lr_patience", 30),
                                 lr_factor=getattr(cfg, "lr_factor", 0.95))
        # Main parameters and local readout heads are norm-clipped as SEPARATE
        # groups (same threshold), so the heads' gradients never alter the main
        # network's clipped update. Without heads the aux group is empty and the
        # main group is exactly the old `trainable` list.
        clip_groups[rule] = aux_param_split(net, optims[rule][0])
    n_heads = len(getattr(base, "_head_names", []))
    if n_heads:
        print(f"  learning_signal={cfg.learning_signal}: {n_heads} local readout head(s) "
              f"(one per non-top MP layer); bptt ignores them.")

    # The task's objective. None → the net's default masked MSE; keeping it None
    # (not passing an explicit fn) preserves the local rules' single-pass fast
    # path AND the module-local default-loss identity check (mpn vs mpn_archive
    # each own their masked_mse symbol). A task-specific loss (e.g. seq-MNIST's
    # cross-entropy) is passed through to both the gradient and validation calls.
    task_loss = getattr(task, "loss_and_grad", None)
    loss_kw = {} if task_loss is None else {"loss_and_grad": task_loss}
    val_loss_fn = masked_mse_loss_and_output_grad if task_loss is None else task_loss

    # Held-out validation set, generated ONCE and shared across rules.
    v_inputs, v_labels, v_mask = task.valid_batch(
        task_params, train_params, cfg.device, cfg.dtype)

    curves = {r: {"train": [], "valid": []} for r in cfg.rules_to_run}
    record_set = set(record_steps)

    # Gradient-alignment diagnostic: per-weight-matrix cosine of each LOCAL rule's
    # gradient vs the exact BPTT gradient at the same weights. Only meaningful for
    # non-bptt rules, so only enabled if there is at least one. The columns (one per
    # W-matrix: input embedding, each MP layer, readout — biases skipped) are fixed
    # for the run, derived from any net's trainable params.
    align_on = cfg.log_grad_align and any(r != "bptt" for r in cfg.rules_to_run)
    align_keys = (_grad_align_keys(next(iter(nets.values()))._trainable_params())
                  if align_on else [])
    # Persisted alignment: per (non-bptt rule, weight-key) list of cosines over the
    # record steps (parallels `curves`; returned so run_experiment can aggregate +
    # save it). Only non-bptt rules populate it (bptt IS the reference).
    align_curves = {r: {k: [] for k in align_keys}
                    for r in cfg.rules_to_run if r != "bptt"} if align_on else {}
    def _short(k):
        return {"W_in": "Win", "W_input": "Win", "W_rec": "Wrec",
                "W_output": "Wout"}.get(k, k)

    # Aligned log table: one header per seed, then one row per rule per recorded
    # step (see the print block below). label_w keeps the rule column aligned.
    label_w = max(len(cfg.rule_label.get(r, r)) for r in cfg.rules_to_run)
    align_hdr = ("   " + " ".join(f"{'cos ' + _short(k):>9}" for k in align_keys)
                 if align_on else "")
    print(f"  seed {seed}:")
    print(f"    {'step':>6}  {'rule':<{label_w}}   {'acc tr':>7} {'acc va':>7}   "
          f"{'loss tr':>9} {'loss va':>9}   {'lr':>7}   "
          f"{'fwd ms':>7} {'bwd ms':>7} {'opt ms':>7}{align_hdr}")

    # Per-rule wall time, split into forward / backward / optimizer phases and
    # averaged over each logging window (single-step timings are too noisy). The
    # held-out eval is NOT timed (shared across rules). Phases:
    #   fwd — the forward pass that produces the gradients: for BPTT the unrolled
    #         forward+M loop; for the local rules the whole forward+eligibility loop
    #         (which IS the gradient computation — forward-mode, no backward).
    #   bwd — the backward pass: BPTT's torch.autograd.grad; ZERO for the local
    #         rules (they have no backward). BPTT reports its own fwd/bwd split via
    #         net._bptt_fwd_s / _bptt_bwd_s; local rules → fwd = grad-production, bwd = 0.
    #   opt — grad-clip + optimizer.step() + param_clamp (rule-independent).
    t_fwd = {r: 0.0 for r in cfg.rules_to_run}
    t_bwd = {r: 0.0 for r in cfg.rules_to_run}
    t_opt = {r: 0.0 for r in cfg.rules_to_run}
    n_accum = 0                                    # steps since last record

    for step in range(cfg.n_datasets):
        # One batch, generated once and fed identically to every rule.
        inputs, labels, mask = task.train_batch(
            task_params, train_params, cfg.batch, cfg.device, cfg.dtype)

        step_log = {}    # per-rule (train_loss, valid_loss, lr) for this step's log line
        step_align = {}  # per-rule {key: cosine vs BPTT} at record steps (local rules)
        step_aux = {}    # per-rule [(head loss, head train acc)] at record steps (local heads)
        step_rates = {}  # effective post-scheduler rates (for the NEXT update)
        for rule in cfg.rules_to_run:
            net = nets[rule]
            trainable, opt, sch = optims[rule]

            # Time the training update, split into fwd / bwd / opt (cuda-synced so
            # the wall-clock reflects completed GPU work, not the async launch queue).
            cuda = (cfg.device.type == "cuda")
            opt.zero_grad()
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            grads = net.sequence_gradients(inputs, labels, mask, **loss_kw)
            if cuda:
                torch.cuda.synchronize()
            t_grad = time.perf_counter() - t0
            # Split grad time into fwd/bwd: BPTT exposes its internal forward vs
            # autograd.grad split; local rules have no backward → all fwd, bwd 0.
            # Either way the TOTAL charged is the outer-measured t_grad, so every
            # rule is timed over the SAME interval (sequence_gradients incl. its
            # .grad write-back). BPTT's internal split ends at autograd.grad, so it
            # misses the wrapper's result-dict build + .grad clone; attribute that
            # remainder to fwd (as the comment always intended) rather than dropping
            # it — otherwise BPTT would be undercounted relative to the local rules.
            fwd_s = getattr(net, "_bptt_fwd_s", None)
            bwd_s = getattr(net, "_bptt_bwd_s", None)
            if fwd_s is not None and bwd_s is not None:
                t_bwd[rule] += bwd_s
                t_fwd[rule] += t_grad - bwd_s   # fwd + un-split wrapper remainder
            else:
                t_fwd[rule] += t_grad

            # Gradient-alignment diagnostic (record steps only, non-bptt rules).
            # Compare THIS rule's gradient to the exact BPTT gradient at the SAME
            # (pre-update) weights → per-layer cosine similarity. Done before
            # opt.step() (weights unchanged) and NOT inside the fwd/bwd/opt timers.
            if align_on and step in record_set and rule != "bptt":
                ref = bptt_reference_grads(net, inputs, labels, mask, loss_kw)
                step_align[rule] = cosine_alignment(grads, ref, align_keys)

            t0 = time.perf_counter()
            if cfg.grad_clip is not None:
                main_params, aux_params = clip_groups[rule]
                torch.nn.utils.clip_grad_norm_(main_params, cfg.grad_clip)
                if aux_params:      # heads: own norm (no-op for bptt, whose heads have no grad)
                    torch.nn.utils.clip_grad_norm_(aux_params, cfg.grad_clip)
            opt.step()
            if net.param_clamping:
                net.param_clamp()
            if cuda:
                torch.cuda.synchronize()
            t_opt[rule] += time.perf_counter() - t0

            # Legacy plateau uses the UPDATED net's main validation loss.
            # Constant mode still evaluates this loss, but it cannot affect LR.
            # The held-out set can be larger than the training batch (valid_n_batch
            # = batch*3 by default), so run the forward in chunks of cfg.batch to
            # bound peak memory (prevents CUDA OOM); validation samples don't
            # interact, so the loss is computed ONCE on the concatenated outputs
            # (its normalizer — 1/N for MSE, 1/n_scored for CE — is then applied
            # over the full held-out set, exact; no per-chunk averaging).
            v_out = eval_outputs_chunked(cfg, net, v_inputs, cfg.batch)
            v_loss, _ = val_loss_fn(v_out, v_labels, v_mask)
            if sch is not None:
                sch.step(v_loss.item())

            if step in record_set:
                train_acc = try_accuracy(task, net, grads["outputs"], labels, mask,
                                         inputs, isvalid=False)
                valid_acc = try_accuracy(task, net, v_out, v_labels, v_mask, v_inputs,
                                         isvalid=True)
                train_loss, valid_loss = float(grads["loss"]), float(v_loss)
                # curves hold the PLOTTED metric (accuracy or loss); the console
                # log below always shows both. lr is the current (post-scheduler) lr.
                if cfg.metric == "loss":
                    curves[rule]["train"].append(train_loss)
                    curves[rule]["valid"].append(valid_loss)
                else:
                    curves[rule]["train"].append(train_acc)
                    curves[rule]["valid"].append(valid_acc)
                step_log[rule] = (train_acc, valid_acc, train_loss, valid_loss,
                                  opt.param_groups[0]["lr"])
                step_rates[rule] = learning_rate_snapshot(net, opt)
                # Local readout heads (local rules under a local learning_signal):
                # each head's train loss + train accuracy on this batch, logged
                # BESIDE the main readout's (never mixed into it).
                aux_losses = grads.get("aux_loss")
                if aux_losses:
                    aux_outs = grads.get("aux_outputs") or [None] * len(aux_losses)
                    step_aux[rule] = [
                        (float(l), (try_accuracy(task, net, o, labels, mask, inputs, isvalid=False)
                                    if o is not None else float("nan")))
                        for l, o in zip(aux_losses, aux_outs)]

        n_accum += 1   # one more timed step since the last log line

        if step in record_set:
            # One aligned row per rule under this step (step shown once, then blank
            # so rules for the same step read as a group). Always show both acc and
            # loss, plus the mean fwd/bwd/opt ms per training update over this
            # logging window (averaged over n_accum steps to smooth jitter). For the
            # local rules bwd is ~0 (forward-mode, no backward); for BPTT fwd is the
            # unrolled forward and bwd is autograd.grad.
            m = max(n_accum, 1)
            for i, r in enumerate(cfg.rules_to_run):
                step_col = f"{step:>6}" if i == 0 else " " * 6
                tr_acc, va_acc, tr_loss, va_loss, lr = step_log[r]
                fwd_ms = 1000.0 * t_fwd[r] / m
                bwd_ms = 1000.0 * t_bwd[r] / m
                opt_ms = 1000.0 * t_opt[r] / m
                # Per-layer cosine vs BPTT (blank for bptt itself: it IS the ref).
                if align_on:
                    al = step_align.get(r, {})
                    # Persist this record step's cosines for the non-bptt rules
                    # (nan when a key was absent), aligned with record_steps.
                    if r in align_curves:
                        for k in align_keys:
                            align_curves[r][k].append(al.get(k, float("nan")))
                    align_cols = "   " + " ".join(
                        f"{al[k]:>9.4f}" if (r != 'bptt' and k in al and al[k] == al[k])
                        else f"{'—':>9}" for k in align_keys)
                else:
                    align_cols = ""
                print(f"    {step_col}  {cfg.rule_label.get(r, r):<{label_w}}   "
                      f"{tr_acc:>7.3f} {va_acc:>7.3f}   "
                      f"{tr_loss:>9.3e} {va_loss:>9.3e}   {lr:>7.1e}   "
                      f"{fwd_ms:>7.1f} {bwd_ms:>7.1f} {opt_ms:>7.1f}{align_cols}")
                rates_txt = "  ".join(f"{name}={rate:.2e}" for name, rate in step_rates[r].items())
                print(f"    {' ' * 6}  {'':<{label_w}}   lr(next)  {rates_txt}")
                # Local readout heads: one indented sub-line per rule that has them
                # (head index = the MP layer it reads; the top layer is the main readout).
                aux = step_aux.get(r)
                if aux:
                    heads_txt = "  ".join(f"head{k}: acc {a:.3f} loss {l:.3e}"
                                          for k, (l, a) in enumerate(aux))
                    print(f"    {' ' * 6}  {'':<{label_w}}   aux  {heads_txt}")
                # Mirror this step's metrics to W&B (opt-in). One row into this
                # rule's run for this seed; grouping/coloring by 'rule' in the UI
                # then yields K curves with each seed drawn separately.
                if wandb_logger is not None:
                    al = step_align.get(r, {}) if align_on else {}
                    metrics = {
                        "train/accuracy": tr_acc, "valid/accuracy": va_acc,
                        "train/loss": tr_loss, "valid/loss": va_loss,
                        "lr": lr,
                        "time/fwd_ms": fwd_ms, "time/bwd_ms": bwd_ms,
                        "time/opt_ms": opt_ms,
                    }
                    metrics.update({f"lr/{name}": rate for name, rate in step_rates[r].items()})
                    for k in align_keys:
                        metrics[f"grad_align/{_short(k)}"] = al.get(k)
                    for k, (l, a) in enumerate(step_aux.get(r, [])):
                        metrics[f"train/aux{k}_loss"] = l
                        metrics[f"train/aux{k}_accuracy"] = a
                    wandb_logger.log_step(r, step, metrics)
            # Reset the window accumulators after logging.
            t_fwd = {r: 0.0 for r in cfg.rules_to_run}
            t_bwd = {r: 0.0 for r in cfg.rules_to_run}
            t_opt = {r: 0.0 for r in cfg.rules_to_run}
            n_accum = 0

    # Save each trained network (per rule) so it can be reloaded later. Stores
    # state_dict + net_params (as in one_task.py) plus rule/seed metadata. Also
    # stores the (initialized) task_params/train_params so a downstream viewer can
    # draw held-out trials from the checkpoint ALONE — no need to re-run
    # build_params() with a matching config (see notebooks/visualize_trained_networks.py).
    # ruleset is duplicated at top level for convenient labeling.
    if cfg.save_nets and cfg.ckpt_prefix:
        for rule in cfg.rules_to_run:
            path = ckpt_path(cfg, rule, seed)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                "run_id": cfg.run_id,
                "state_dict": nets[rule].state_dict(),
                "net_params": {**net_params, "learning_rule": rule},
                "task_params": task_params,
                "train_params": train_params,
                "ruleset": cfg.ruleset,
                "learning_rule": rule,
                "feedback_mode": cfg.feedback_mode,
                "dfa_preset": cfg.dfa_preset,
                "uses_dfa": uses_dfa(
                    getattr(nets[rule], "feedback_mode", cfg.feedback_mode), rule,
                    getattr(nets[rule], "input_mode", cfg.input_mode),
                    getattr(nets[rule], "_has_trainable_embed", lambda: False)()),
                "input_normalize": cfg.input_normalize,
                "mp_input_norm": getattr(cfg, "mp_input_norm", "none"),
                "mp_residual": cfg.mp_residual,
                "residual_scale": getattr(cfg, "residual_scale", 1.0),
                "input_mode": cfg.input_mode,
                "resolved_input_mode": getattr(nets[rule], 'resolved_input_mode', None),
                "resolved_local_bias_modes": getattr(nets[rule], 'resolved_local_bias_modes', None),
                "learning_signal": getattr(nets[rule], 'learning_signal', cfg.learning_signal),
                "signal_mode": getattr(cfg, "signal_mode", ""),
                "effective_config": effective_rule_summary(cfg, nets[rule], rule),
                "head_lr_mult": cfg.head_lr_mult,
                "lr_schedule": cfg.lr_schedule,
                "lr_patience": getattr(cfg, "lr_patience", 30),
                "lr_factor": getattr(cfg, "lr_factor", 0.95),
                "learning_rates": learning_rate_snapshot(nets[rule], optims[rule][1]),
                "seed": seed,
            }, path)
            print(f"  saved network: {path}")

    del nets, optims, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return curves, align_curves


# ─── Plotting / persistence ───────────────────────────────────────────────────
def plot(cfg, record_steps, agg, rules, title_suffix, save_to):
    """One figure, two panels (train / test), each rule mean ± std. The plotted
    metric follows cfg.metric: 'accuracy' → percent, linear 0-110 y; 'loss' →
    masked-MSE, log y. Driven purely by the passed arrays so it works both live
    and from a reloaded .npz (replot_from_npz)."""
    metric = getattr(cfg, "metric", "accuracy")
    is_loss = (metric == "loss")
    scale = 1.0 if is_loss else 100.0     # accuracy stored as fraction → percent
    steps = np.asarray(record_steps)
    noun = "loss" if is_loss else "accuracy"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, split, title in zip(axes, ("train", "valid"),
                                (f"Training {noun}", f"Testing (held-out) {noun}")):
        for rule in rules:
            mean = scale * np.asarray(agg[rule][split]["mean"])
            std = scale * np.asarray(agg[rule][split]["std"])
            color = cfg.rule_color.get(rule, None)
            ax.plot(steps, mean, color=color, label=cfg.rule_label.get(rule, rule), lw=2)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        if is_loss:
            ax.set_yscale("log")          # loss spans orders of magnitude
        else:
            ax.set_ylim(0, 110)
    axes[0].set_ylabel(getattr(cfg, "acc_label", "angle accuracy (%)"))
    axes[1].legend(loc="best" if is_loss else "lower right", frameon=False)
    fig.suptitle(f"{cfg.title}  {title_suffix}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {save_to}")


def plot_alignment(cfg, record_steps, align_agg, align_keys, save_to):
    """Gradient-alignment-vs-BPTT figure: cosine similarity between each non-bptt
    rule's gradient and the exact BPTT gradient, per weight matrix, over training.
    One subplot per weight-key (input embedding, each MP layer, readout); within a
    subplot one mean±std curve per non-bptt rule. cos=1 → the local rule points
    exactly along the true gradient; lower → more approximation. No-op if empty."""
    rules = list(align_agg.keys())
    if not rules or not align_keys:
        return
    steps = np.asarray(record_steps)
    ncols = len(align_keys)
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 3.6),
                             squeeze=False, sharey=True)
    for j, key in enumerate(align_keys):
        ax = axes[0][j]
        for r in rules:
            ms = align_agg.get(r, {}).get(key)
            if ms is None:
                continue
            mean, std = np.asarray(ms["mean"]), np.asarray(ms["std"])
            c = cfg.rule_color.get(r)
            ax.plot(steps, mean, color=c, lw=2, label=cfg.rule_label.get(r, r))
            ax.fill_between(steps, mean - std, mean + std, color=c, alpha=0.15)
        ax.axhline(1.0, color="0.7", lw=0.8, ls=":")   # perfect alignment
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("training step")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("cosine(grad, BPTT grad)")
    axes[0][-1].legend(frameon=False, loc="lower right")
    fig.suptitle(f"{cfg.title}: gradient alignment vs BPTT "
                 f"(mean ± std over {cfg.n_runs} runs, {arch_suffix(cfg)})", y=1.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    print(f"Saved gradient-alignment figure: {save_to}")


def save_plot_data(cfg, record_steps, runs, agg, path=None,
                   align_runs=None, align_agg=None):
    """Save the arrays behind the figure to an .npz so it can be replotted later
    without retraining. Stores per-seed curves (runs), aggregated mean/std (agg),
    the x-axis (record_steps), and run metadata. When align_runs/align_agg are given
    (gradient alignment vs BPTT), stores per-(rule,weight-key) cosine curves too:
      align_runs__{rule}__{key}   (n_runs, n_points) per-seed cosines
      align_mean/std__{rule}__{key}   (n_points,) across-seed mean/std
      align_keys                  the ordered weight-key list (for replotting)
    Use replot_from_npz() to reload."""
    path = path or data_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {
        "record_steps": np.asarray(record_steps),
        "rules": np.asarray(cfg.rules_to_run),
        "seeds": np.asarray([cfg.seed + k for k in range(cfg.n_runs)]),
        # scalar/string config for provenance + title reconstruction
        "run_id": cfg.run_id,
        "ruleset": cfg.ruleset, "n_hidden": cfg.n_hidden, "batch": cfg.batch,
        "n_datasets": cfg.n_datasets, "lr": cfg.lr, "n_runs": cfg.n_runs,
        "head_lr_mult": cfg.head_lr_mult, "lr_schedule": cfg.lr_schedule,
        "lr_patience": getattr(cfg, "lr_patience", 30), "lr_factor": getattr(cfg, "lr_factor", 0.95),
        "feedback_mode": cfg.feedback_mode, "input_normalize": cfg.input_normalize,
        "mp_input_norm": getattr(cfg, "mp_input_norm", "none"),
        "mp_input_norm_eps": getattr(cfg, "mp_input_norm_eps", 1e-5),
        "mp_residual": cfg.mp_residual,
        "residual_scale": getattr(cfg, "residual_scale", 1.0),
        "cross_layer_steps": getattr(cfg, "cross_layer_steps", 0),
        "input_mode": cfg.input_mode, "title": cfg.title,
        "learning_signal": getattr(cfg, "learning_signal", "global"),
        "local_signal_alpha": getattr(cfg, "local_signal_alpha", 1.0),
        "signal_mode": getattr(cfg, "signal_mode", ""),
        # full architecture (multi-layer stacks) for provenance + replot suffix
        "arch_tag": getattr(cfg, "arch_tag", ""),
        "arch_desc": getattr(cfg, "arch_desc", ""),
        # what the stored curves represent, so replot renders the right axes
        "metric": cfg.metric, "acc_label": cfg.acc_label,
    }
    for r in cfg.rules_to_run:
        for split in ("train", "valid"):
            out[f"runs__{r}__{split}"] = np.asarray(runs[r][split], dtype=float)
            out[f"mean__{r}__{split}"] = np.asarray(agg[r][split]["mean"], dtype=float)
            out[f"std__{r}__{split}"] = np.asarray(agg[r][split]["std"], dtype=float)
    # Gradient-alignment-vs-BPTT arrays (only when the diagnostic ran).
    if align_agg:
        akeys = []
        for r, per_key in align_agg.items():
            for k, ms in per_key.items():
                out[f"align_mean__{r}__{k}"] = np.asarray(ms["mean"], dtype=float)
                out[f"align_std__{r}__{k}"] = np.asarray(ms["std"], dtype=float)
                if align_runs is not None:
                    out[f"align_runs__{r}__{k}"] = np.asarray(align_runs[r][k], dtype=float)
                if k not in akeys:
                    akeys.append(k)
        out["align_keys"] = np.asarray(akeys)
        out["align_rules"] = np.asarray(list(align_agg.keys()))
    np.savez(path, **out)
    print(f"Saved plot data: {path}")


def _json_safe(x):
    """Best-effort convert an arbitrary value to something json.dump can write:
    tensors/arrays → lists, torch dtype/device → str, dict/list recurse, other
    non-primitives → repr. Never raises (config recording must not break a run)."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return repr(x)   # torch.dtype, torch.device, callables, task adapters, …


def save_config(cfg, path=None):
    """Write training + network setup to config_path(cfg), or an explicit path.
    Records the RunConfig scalars plus the RESOLVED net_params / task_params /
    train_params (built once via the cfg hooks, exactly as run_seed builds them), so
    the JSON fully describes how the nets were configured. Best-effort: any value
    that is not natively JSON-serializable is coerced by _json_safe, and the whole
    thing is wrapped so a recording failure can never abort training."""
    path = path or config_path(cfg)
    try:
        task = cfg.task if cfg.task is not None else tasks.make_task(cfg.ruleset)
        task_params, train_params, net_params = cfg.build_params()
        task_params, train_params, net_params = task.init_params(
            task_params, train_params, net_params)
        trainable_embed = bool(
            net_params.get("net_type") == "dmpn" and net_params.get("input_layer_add", False)
            and (net_params.get("input_layer_add_trainable", False)
                 or net_params.get("input_layer_bias", False)))
        dfa_by_rule = {
            rule: uses_dfa(cfg.feedback_mode, rule, cfg.input_mode, trainable_embed)
            for rule in cfg.rules_to_run
        }
        record = {
            # run / experiment
            "run_id": cfg.run_id,
            "file_prefix": cfg.file_prefix, "title": cfg.title,
            "ruleset": cfg.ruleset, "rules_to_run": list(cfg.rules_to_run),
            "seed": cfg.seed, "n_runs": cfg.n_runs,
            "seeds": [cfg.seed + k for k in range(cfg.n_runs)],
            # optimization
            "batch": cfg.batch, "n_datasets": cfg.n_datasets, "lr": cfg.lr,
            "head_lr_mult": cfg.head_lr_mult, "lr_schedule": cfg.lr_schedule,
            "lr_patience": getattr(cfg, "lr_patience", 30), "lr_factor": getattr(cfg, "lr_factor", 0.95),
            "grad_clip": cfg.grad_clip, "log_every": cfg.log_every,
            "device": str(cfg.device), "dtype": str(cfg.dtype),
            "metric": cfg.metric, "acc_label": cfg.acc_label,
            # network / learning-rule setup
            "n_hidden": cfg.n_hidden,
            "arch_tag": getattr(cfg, "arch_tag", ""),
            "arch_desc": getattr(cfg, "arch_desc", ""),
            "feedback_mode": cfg.feedback_mode,
            "dfa_preset": cfg.dfa_preset,
            "uses_dfa": any(dfa_by_rule.values()),
            "uses_dfa_by_rule": dfa_by_rule,
            "input_normalize": cfg.input_normalize,
            "input_norm_sample": cfg.input_norm_sample,
            "mp_input_norm": getattr(cfg, "mp_input_norm", "none"),
            "mp_input_norm_eps": getattr(cfg, "mp_input_norm_eps", 1e-5),
            "log_grad_align": cfg.log_grad_align,
            "mp_residual": cfg.mp_residual,
            "residual_scale": getattr(cfg, "residual_scale", 1.0),
            "cross_layer_steps": getattr(cfg, "cross_layer_steps", 0),
            "input_mode": cfg.input_mode,
            "learning_signal": getattr(cfg, "learning_signal", "global"),
            "local_signal_alpha": getattr(cfg, "local_signal_alpha", 1.0),
            "signal_mode": getattr(cfg, "signal_mode", ""),
            "resolved_input_modes": {
                rule: resolve_input_mode(cfg.input_mode, rule) for rule in cfg.rules_to_run
            } if net_params.get("net_type") == "dmpn" else {},
            # the resolved param dicts the nets are actually built from
            "net_params": _json_safe(net_params),
            "task_params": _json_safe(task_params),
            "train_params": _json_safe(train_params),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(record, fh, indent=2)
        print(f"Saved config: {path}")
    except Exception as e:                      # never let recording break a run
        print(f"WARNING: could not save config JSON ({e})")
    return path


def replot_from_npz(cfg, npz_path, save_to=None):
    """Regenerate the accuracy figure from a saved .npz (no retraining). Uses the
    title stored in the .npz and the labels/colors from cfg. Returns the output
    figure path."""
    d = np.load(npz_path, allow_pickle=True)
    rules = [str(r) for r in d["rules"]]
    record_steps = d["record_steps"]
    agg = {r: {sp: {"mean": d[f"mean__{r}__{sp}"], "std": d[f"std__{r}__{sp}"]}
               for sp in ("train", "valid")} for r in rules}
    # Prefer the full architecture saved with the data; fall back to hidden=scalar
    # for older .npz files that predate arch_desc.
    arch_note = (str(d["arch_desc"]) if "arch_desc" in d.files and str(d["arch_desc"])
                 else f"hidden={int(d['n_hidden'])}")
    residual_note = ""
    if "mp_residual" in d.files and bool(d["mp_residual"]):
        scale = float(d["residual_scale"]) if "residual_scale" in d.files else 1.0
        residual_note = f", residual scale={scale:g}"
    suffix = f"(mean ± std over {int(d['n_runs'])} runs, {arch_note}{residual_note})"
    # Prefer the title / metric / label saved with the data; fall back to cfg.
    cfg_for_plot = copy.copy(cfg)
    if "title" in d.files:
        cfg_for_plot.title = str(d["title"])
    if "metric" in d.files:
        cfg_for_plot.metric = str(d["metric"])
    if "acc_label" in d.files:
        cfg_for_plot.acc_label = str(d["acc_label"])
    save_to = save_to or (os.path.splitext(str(npz_path))[0] + "_replot.png")
    plot(cfg_for_plot, record_steps, agg, rules, suffix, save_to)
    return save_to


def run_experiment(cfg):
    """Full experiment: train every rule in lockstep across cfg.n_runs seeds,
    aggregate mean/std, save plot data, render the figure, print a summary."""
    print(f"Run: {run_stem(cfg)}")
    # Keep setup available even if training stops before every seed finishes.
    if cfg.run_id:
        save_config(cfg)
    print(f"Task: {cfg.ruleset}{cfg.header_note}  |  rules: {cfg.rules_to_run}  |  "
          f"runs: {cfg.n_runs}  |  {arch_suffix(cfg)} batch={cfg.batch} "
          f"steps={cfg.n_datasets} lr={cfg.lr} head_lr_mult={cfg.head_lr_mult:g} "
          f"lr_schedule={cfg.lr_schedule}"
          + (f" (patience={getattr(cfg, 'lr_patience', 30)}, factor={getattr(cfg, 'lr_factor', 0.95):g})"
             if cfg.lr_schedule == "plateau" else "")
          + f" clip={cfg.grad_clip}")
    print(f"Device: {cfg.device}  dtype: {cfg.dtype}"
          f"{f'  signal: {cfg.signal_mode}' if getattr(cfg, 'signal_mode', '') else ''}"
          f"  feedback: {cfg.feedback_mode}"
          f"{'  (input norm ON)' if getattr(cfg, 'input_normalize', False) else ''}"
          f"{f'  (MP-input norm: {cfg.mp_input_norm})' if getattr(cfg, 'mp_input_norm', 'none') != 'none' else ''}"
          f"{f'  (residual scale={cfg.residual_scale:g})' if getattr(cfg, 'mp_residual', False) else ''}"
          f"{f'  (cross-layer x{cfg.cross_layer_steps})' if getattr(cfg, 'cross_layer_steps', 0) else ''}"
          f"  input_mode: {getattr(cfg, 'input_mode', 'match')}"
          f"{f'  learning_signal: {cfg.learning_signal}' if getattr(cfg, 'learning_signal', 'global') != 'global' else ''}"
          f"{f' (alpha={cfg.local_signal_alpha:g})' if getattr(cfg, 'learning_signal', 'global') == 'mixed' else ''}\n")

    # Weights & Biases (opt-in). Import lazily so non-W&B runs never touch wandb.
    # The experiment name == the output save-stem (figure/.npz share it), used
    # as the W&B group so all K rules × n_runs seeds compare on one page.
    wb = None
    experiment = run_stem(cfg)
    if getattr(cfg, "use_wandb", False):
        import wandb_logging as wb
        if not wb.wandb_available():
            raise RuntimeError(
                "use_wandb=True but the `wandb` package is not importable. "
                "Install it (`pip install wandb`) or drop --wandb.")
        print(f"W&B: logging to project '{getattr(cfg, 'wandb_project', None)}', "
              f"group/experiment '{experiment}' "
              f"({len(cfg.rules_to_run)} rules × {cfg.n_runs} seeds runs)\n")

    record_steps = list(range(0, cfg.n_datasets, cfg.log_every))
    if record_steps[-1] != cfg.n_datasets - 1:
        record_steps.append(cfg.n_datasets - 1)

    # runs[rule][split] -> list (over seeds) of accuracy curves.
    # align_runs[rule][key] -> list (over seeds) of per-record-step cosine curves
    # (gradient alignment vs BPTT); empty for bptt and when the diagnostic is off.
    runs = {r: {"train": [], "valid": []} for r in cfg.rules_to_run}
    align_runs = {}
    seeds = [cfg.seed + k for k in range(cfg.n_runs)]
    for run_idx, seed in enumerate(seeds):
        print(f"── Run {run_idx + 1}/{cfg.n_runs}  (seed {seed}) ──")
        # One W&B logger (K live per-rule runs) per seed, so each seed is a separate
        # curve within each rule's color. Closed before the next seed so at most K
        # runs are open at once.
        wandb_logger = wb.WandbLogger(cfg, seed, run_idx, experiment) if wb else None
        try:
            curves, align_curves = run_seed(
                cfg, seed, record_steps, run_idx=run_idx, wandb_logger=wandb_logger)
        finally:
            if wandb_logger is not None:
                wandb_logger.finish()
        for r in cfg.rules_to_run:
            runs[r]["train"].append(curves[r]["train"])
            runs[r]["valid"].append(curves[r]["valid"])
        for r, per_key in align_curves.items():
            ar = align_runs.setdefault(r, {})
            for k, series in per_key.items():
                ar.setdefault(k, []).append(series)

    # Aggregate mean/std across seeds (ignoring any nan accuracies).
    agg = {}
    for r in cfg.rules_to_run:
        agg[r] = {}
        for split in ("train", "valid"):
            arr = np.asarray(runs[r][split], dtype=float)   # (n_runs, n_points)
            agg[r][split] = {"mean": np.nanmean(arr, axis=0),
                             "std": np.nanstd(arr, axis=0)}

    # Aggregate the alignment cosines the same way (per rule × weight-key).
    align_agg = {}
    for r, per_key in align_runs.items():
        align_agg[r] = {}
        for k, seed_series in per_key.items():
            arr = np.asarray(seed_series, dtype=float)      # (n_runs, n_points)
            align_agg[r][k] = {"mean": np.nanmean(arr, axis=0),
                               "std": np.nanstd(arr, axis=0)}

    save_plot_data(cfg, record_steps, runs, agg, align_runs=align_runs,
                   align_agg=align_agg)
    if not cfg.run_id:
        save_config(cfg)   # legacy config beside the figure; compact runs save at startup
    # Append notes to the figure title only when the feature is on / non-default, so
    # existing (norm-off, match) figure titles are unchanged.
    inorm_note = ", input norm" if getattr(cfg, "input_normalize", False) else ""
    mpnorm_note = (f", MP-input {cfg.mp_input_norm} norm"
                   if getattr(cfg, "mp_input_norm", "none") != "none" else "")
    resid_note = (f", residual scale={cfg.residual_scale:g}"
                  if getattr(cfg, "mp_residual", False) else "")
    xl_note = (f", cross-layer x{cfg.cross_layer_steps}"
               if getattr(cfg, "cross_layer_steps", 0) else "")
    inmode_note = ("" if getattr(cfg, "input_mode", "match") == "match"
                   else f", input={cfg.input_mode}")
    signal_note = ("" if getattr(cfg, "learning_signal", "global") == "global"
                   else f", signal={cfg.learning_signal}"
                   + (f" (alpha={cfg.local_signal_alpha:g})" if cfg.learning_signal == "mixed" else ""))
    plot(cfg, record_steps, agg, cfg.rules_to_run,
         f"(mean ± std over {cfg.n_runs} runs, {arch_suffix(cfg)}{inorm_note}{mpnorm_note}{resid_note}{xl_note}{inmode_note}{signal_note})",
         fig_path(cfg))

    # Gradient-alignment-vs-BPTT figure (only when the diagnostic produced data).
    if align_agg:
        # Preserve the input→hidden→output key order any non-bptt rule recorded.
        any_rule = next(iter(align_agg))
        align_keys = _grad_align_keys(align_agg[any_rule])
        plot_alignment(cfg, record_steps, align_agg, align_keys, align_fig_path(cfg))

    # Final-metric summary (the plotted metric: accuracy or loss).
    metric_noun = "loss" if cfg.metric == "loss" else "accuracy"
    fmt = "{:.3e}" if cfg.metric == "loss" else "{:.3f}"
    print(f"\nFinal {metric_noun} (mean ± std over runs):")
    final_summary = {}
    for r in cfg.rules_to_run:
        tr = agg[r]["train"]; va = agg[r]["valid"]
        print(f"  {cfg.rule_label.get(r, r):<16} "
              f"train {fmt.format(tr['mean'][-1])} ± {fmt.format(tr['std'][-1])}"
              f"   test {fmt.format(va['mean'][-1])} ± {fmt.format(va['std'][-1])}")
        final_summary[cfg.rule_label.get(r, r)] = (
            float(tr["mean"][-1]), float(tr["std"][-1]),
            float(va["mean"][-1]), float(va["std"][-1]))

    # W&B summary run: the aggregate K-colored figures + the final-metric table, in
    # the same group, so the comparison is viewable without any UI grouping.
    if wb:
        align_fig = align_fig_path(cfg) if align_agg else None
        try:
            wb.log_summary(cfg, experiment, fig_path(cfg), align_fig, final_summary)
        except Exception as e:                  # never let logging break a finished run
            print(f"WARNING: could not log W&B summary ({e})")

    return agg
