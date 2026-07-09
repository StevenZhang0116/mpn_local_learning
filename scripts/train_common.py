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
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless: write PNG, no display
import matplotlib.pyplot as plt

import tasks  # Task adapters (data/metric seam); make_task(ruleset) picks one
from mpn import masked_mse_loss_and_output_grad  # one shared masked-MSE definition


@dataclass
class RunConfig:
    """All knobs + hooks for one train/compare experiment. Callers build this
    fresh from their module globals (see _cfg() in the train scripts)."""
    # identity / presentation
    file_prefix: str            # figure/data filename stem, e.g. "train_mpn"
    ckpt_prefix: str            # checkpoint filename stem, e.g. "mpn" ("" → no save)
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
    # Task adapter (the data/metric seam): provides init_params / valid_batch /
    # train_batch / accuracy. Defaults to None, resolved to tasks.make_task(ruleset)
    # in run_seed so callers that don't set it keep the ring-task behaviour.
    task: object = None
    # optional extra string appended to the filename param tag (e.g. eta/lambda);
    # keep it filename-safe. Empty by default.
    tag_extra: str = ""
    # y-axis label for the plotted panels (task-dependent: accuracy label for the
    # ring tasks / seq-MNIST, loss label for the adding problem).
    acc_label: str = "angle accuracy (%)"
    # Which curve the figure shows: 'accuracy' (percent, 0-110 y) or 'loss'
    # (masked-MSE, log-y). Regression tasks like the adding problem use 'loss'
    # because accuracy is uninformative there. Accuracy is always logged either way.
    metric: str = "accuracy"


# ─── Path helpers (read the passed cfg, i.e. the caller's live globals) ───────
def param_tag(cfg):
    tag = (f"{cfg.ruleset}_h{cfg.n_hidden}_b{cfg.batch}_n{cfg.n_datasets}"
           f"_lr{cfg.lr:.0e}_{cfg.feedback_mode}")
    if cfg.tag_extra:
        tag += f"_{cfg.tag_extra}"
    return tag


def fig_path(cfg):
    return os.path.join(cfg.fig_dir, f"{cfg.file_prefix}_{param_tag(cfg)}_runs{cfg.n_runs}.png")


def data_path(cfg):
    return os.path.join(cfg.data_dir, f"{cfg.file_prefix}_{param_tag(cfg)}_runs{cfg.n_runs}.npz")


def ckpt_path(cfg, rule, seed):
    return os.path.join(cfg.ckpt_dir, f"{cfg.ckpt_prefix}_{param_tag(cfg)}_{rule}_seed{seed}.pt")


# ─── Training pieces ──────────────────────────────────────────────────────────
def try_accuracy(task, net, output, labels, mask, inputs, isvalid=False):
    """Accuracy via the task adapter; returns nan on failure. train uses
    isvalid=False (current batch), validation isvalid=True. The task decides the
    metric (angle for ring tasks, argmax-correct for seq-MNIST)."""
    try:
        return float(task.accuracy(net, output, labels, mask, inputs, isvalid=isvalid))
    except Exception:
        return float("nan")


def make_optim(net, lr):
    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.95, patience=30, min_lr=1e-8)
    return trainable, opt, sch


def run_seed(cfg, seed, record_steps):
    """Train all cfg.rules_to_run in lockstep for one seed: identical init
    (deepcopy of one base net), identical per-step data, identical held-out
    validation set. Returns {rule: {'train': [...], 'valid': [...]}} sampled at
    record_steps. Saves each trained net if cfg.save_nets."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Task adapter (data/metric seam). Defaults to the ring-task pipeline.
    task = cfg.task if cfg.task is not None else tasks.make_task(cfg.ruleset)

    task_params, train_params, net_params = cfg.build_params()
    task_params, train_params, net_params = task.init_params(
        task_params, train_params, net_params)

    # One base net → deepcopy so every rule starts from the SAME weights.
    base = cfg.net_factory(net_params, seed == cfg.seed).to(cfg.device).to(cfg.dtype)
    nets, optims = {}, {}
    for rule in cfg.rules_to_run:
        net = copy.deepcopy(base)
        net.learning_rule = rule
        nets[rule] = net
        optims[rule] = make_optim(net, cfg.lr)

    # Held-out validation set, generated ONCE and shared across rules.
    v_inputs, v_labels, v_mask = task.valid_batch(
        task_params, train_params, cfg.device, cfg.dtype)

    curves = {r: {"train": [], "valid": []} for r in cfg.rules_to_run}
    record_set = set(record_steps)

    # Aligned log table: one header per seed, then one row per rule per recorded
    # step (see the print block below). label_w keeps the rule column aligned.
    label_w = max(len(cfg.rule_label.get(r, r)) for r in cfg.rules_to_run)
    print(f"  seed {seed}:")
    print(f"    {'step':>6}  {'rule':<{label_w}}   {'acc tr':>7} {'acc va':>7}   "
          f"{'loss tr':>9} {'loss va':>9}   {'lr':>7}   {'ms/step':>8}")

    # Accumulate the training-update wall time per rule so we can report the mean
    # ms/step over each logging window (single-step timings are too noisy). Timed
    # region = the training update only (sequence_gradients + clip + opt.step),
    # NOT the shared held-out eval — that is what differs across rules.
    t_accum = {r: 0.0 for r in cfg.rules_to_run}   # seconds since last record
    n_accum = 0                                    # steps since last record

    for step in range(cfg.n_datasets):
        # One batch, generated once and fed identically to every rule.
        inputs, labels, mask = task.train_batch(
            task_params, train_params, cfg.batch, cfg.device, cfg.dtype)

        step_log = {}  # per-rule (train_loss, valid_loss, lr) for this step's log line
        for rule in cfg.rules_to_run:
            net = nets[rule]
            trainable, opt, sch = optims[rule]

            # Time the training update only (the part that differs across rules).
            if cfg.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad()
            grads = net.sequence_gradients(inputs, labels, mask)
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            opt.step()
            if net.param_clamping:
                net.param_clamp()
            if cfg.device.type == "cuda":
                torch.cuda.synchronize()
            t_accum[rule] += time.perf_counter() - t0

            # Held-out validation loss on the UPDATED net drives the scheduler
            # (as in one_task.py) — smoother than the fresh per-batch train loss.
            v_out = cfg.eval_outputs(net, v_inputs)
            v_loss, _ = masked_mse_loss_and_output_grad(v_out, v_labels, v_mask)
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

        n_accum += 1   # one more timed step since the last log line

        if step in record_set:
            # One aligned row per rule under this step (step shown once, then blank
            # so rules for the same step read as a group). Always show both acc and
            # loss, plus the mean ms per training update over this logging window
            # (averaged over n_accum steps to smooth single-step jitter).
            for i, r in enumerate(cfg.rules_to_run):
                step_col = f"{step:>6}" if i == 0 else " " * 6
                tr_acc, va_acc, tr_loss, va_loss, lr = step_log[r]
                ms = 1000.0 * t_accum[r] / max(n_accum, 1)
                print(f"    {step_col}  {cfg.rule_label.get(r, r):<{label_w}}   "
                      f"{tr_acc:>7.3f} {va_acc:>7.3f}   "
                      f"{tr_loss:>9.3e} {va_loss:>9.3e}   {lr:>7.1e}   {ms:>8.1f}")
            # Reset the window accumulators after logging.
            t_accum = {r: 0.0 for r in cfg.rules_to_run}
            n_accum = 0

    # Save each trained network (per rule) so it can be reloaded later. Stores
    # state_dict + net_params (as in one_task.py) plus rule/seed metadata.
    if cfg.save_nets and cfg.ckpt_prefix:
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        for rule in cfg.rules_to_run:
            path = ckpt_path(cfg, rule, seed)
            torch.save({
                "state_dict": nets[rule].state_dict(),
                "net_params": net_params,
                "learning_rule": rule,
                "feedback_mode": cfg.feedback_mode,
                "seed": seed,
            }, path)
            print(f"  saved network: {path}")

    del nets, optims, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return curves


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


def save_plot_data(cfg, record_steps, runs, agg, path=None):
    """Save the arrays behind the figure to an .npz so it can be replotted later
    without retraining. Stores per-seed curves (runs), aggregated mean/std (agg),
    the x-axis (record_steps), and run metadata. Use replot_from_npz() to reload."""
    path = path or data_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {
        "record_steps": np.asarray(record_steps),
        "rules": np.asarray(cfg.rules_to_run),
        "seeds": np.asarray([cfg.seed + k for k in range(cfg.n_runs)]),
        # scalar/string config for provenance + title reconstruction
        "ruleset": cfg.ruleset, "n_hidden": cfg.n_hidden, "batch": cfg.batch,
        "n_datasets": cfg.n_datasets, "lr": cfg.lr, "n_runs": cfg.n_runs,
        "feedback_mode": cfg.feedback_mode, "title": cfg.title,
        # what the stored curves represent, so replot renders the right axes
        "metric": cfg.metric, "acc_label": cfg.acc_label,
    }
    for r in cfg.rules_to_run:
        for split in ("train", "valid"):
            out[f"runs__{r}__{split}"] = np.asarray(runs[r][split], dtype=float)
            out[f"mean__{r}__{split}"] = np.asarray(agg[r][split]["mean"], dtype=float)
            out[f"std__{r}__{split}"] = np.asarray(agg[r][split]["std"], dtype=float)
    np.savez(path, **out)
    print(f"Saved plot data: {path}")
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
    suffix = (f"(mean ± std over {int(d['n_runs'])} runs, "
              f"hidden={int(d['n_hidden'])})")
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
    print(f"Task: {cfg.ruleset}{cfg.header_note}  |  rules: {cfg.rules_to_run}  |  "
          f"runs: {cfg.n_runs}  |  hidden={cfg.n_hidden} batch={cfg.batch} "
          f"steps={cfg.n_datasets} lr={cfg.lr} clip={cfg.grad_clip}")
    print(f"Device: {cfg.device}  dtype: {cfg.dtype}  feedback: {cfg.feedback_mode}\n")

    record_steps = list(range(0, cfg.n_datasets, cfg.log_every))
    if record_steps[-1] != cfg.n_datasets - 1:
        record_steps.append(cfg.n_datasets - 1)

    # runs[rule][split] -> list (over seeds) of accuracy curves.
    runs = {r: {"train": [], "valid": []} for r in cfg.rules_to_run}
    seeds = [cfg.seed + k for k in range(cfg.n_runs)]
    for run_idx, seed in enumerate(seeds):
        print(f"── Run {run_idx + 1}/{cfg.n_runs}  (seed {seed}) ──")
        curves = run_seed(cfg, seed, record_steps)
        for r in cfg.rules_to_run:
            runs[r]["train"].append(curves[r]["train"])
            runs[r]["valid"].append(curves[r]["valid"])

    # Aggregate mean/std across seeds (ignoring any nan accuracies).
    agg = {}
    for r in cfg.rules_to_run:
        agg[r] = {}
        for split in ("train", "valid"):
            arr = np.asarray(runs[r][split], dtype=float)   # (n_runs, n_points)
            agg[r][split] = {"mean": np.nanmean(arr, axis=0),
                             "std": np.nanstd(arr, axis=0)}

    save_plot_data(cfg, record_steps, runs, agg)
    plot(cfg, record_steps, agg, cfg.rules_to_run,
         f"(mean ± std over {cfg.n_runs} runs, hidden={cfg.n_hidden})", fig_path(cfg))

    # Final-metric summary (the plotted metric: accuracy or loss).
    metric_noun = "loss" if cfg.metric == "loss" else "accuracy"
    fmt = "{:.3e}" if cfg.metric == "loss" else "{:.3f}"
    print(f"\nFinal {metric_noun} (mean ± std over runs):")
    for r in cfg.rules_to_run:
        tr = agg[r]["train"]; va = agg[r]["valid"]
        print(f"  {cfg.rule_label.get(r, r):<16} "
              f"train {fmt.format(tr['mean'][-1])} ± {fmt.format(tr['std'][-1])}"
              f"   test {fmt.format(va['mean'][-1])} ± {fmt.format(va['std'][-1])}")
    return agg
