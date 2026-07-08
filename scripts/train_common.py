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
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless: write PNG, no display
import matplotlib.pyplot as plt

import mpn_tasks
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
    # optional extra string appended to the filename param tag (e.g. eta/lambda);
    # keep it filename-safe. Empty by default.
    tag_extra: str = ""


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
def try_accuracy(net, output, labels, mask, inputs, isvalid=False):
    """Best-effort angle accuracy via the library; returns nan on failure.
    train uses isvalid=False (current batch), validation isvalid=True."""
    try:
        acc, _ = net.compute_acc(output.float(), labels.float(), mask.float(),
                                 inputs.float(), mode=net.acc_measure, isvalid=isvalid)
        return float(acc)
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

    task_params, train_params, net_params = cfg.build_params()
    task_params, train_params, net_params = mpn_tasks.convert_and_init_multitask_params(
        (task_params, train_params, net_params)
    )
    net_params["prefs"] = mpn_tasks.get_prefs(task_params["hp"])

    # One base net → deepcopy so every rule starts from the SAME weights.
    base = cfg.net_factory(net_params, seed == cfg.seed).to(cfg.device).to(cfg.dtype)
    nets, optims = {}, {}
    for rule in cfg.rules_to_run:
        net = copy.deepcopy(base)
        net.learning_rule = rule
        nets[rule] = net
        optims[rule] = make_optim(net, cfg.lr)

    # Held-out validation set, generated ONCE and shared across rules.
    vdata, _ = mpn_tasks.generate_trials_wrap(
        task_params, train_params["valid_n_batch"], rules=task_params["rules"],
        mode_input="random_batch", device=cfg.device,
    )
    v_inputs, v_labels, v_mask = (d.to(cfg.dtype) for d in vdata)

    curves = {r: {"train": [], "valid": []} for r in cfg.rules_to_run}
    record_set = set(record_steps)

    for step in range(cfg.n_datasets):
        # One batch, generated once and fed identically to every rule.
        data, _ = mpn_tasks.generate_trials_wrap(
            task_params, cfg.batch, rules=task_params["rules"],
            mode_input="random_batch", device=cfg.device,
        )
        inputs, labels, mask = (d.to(cfg.dtype) for d in data)

        step_log = {}  # per-rule (train_loss, valid_loss, lr) for this step's log line
        for rule in cfg.rules_to_run:
            net = nets[rule]
            trainable, opt, sch = optims[rule]
            opt.zero_grad()
            grads = net.sequence_gradients(inputs, labels, mask)
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            opt.step()
            if net.param_clamping:
                net.param_clamp()

            # Held-out validation loss on the UPDATED net drives the scheduler
            # (as in one_task.py) — smoother than the fresh per-batch train loss.
            v_out = cfg.eval_outputs(net, v_inputs)
            v_loss, _ = masked_mse_loss_and_output_grad(v_out, v_labels, v_mask)
            sch.step(v_loss.item())

            if step in record_set:
                train_acc = try_accuracy(net, grads["outputs"], labels, mask,
                                         inputs, isvalid=False)
                valid_acc = try_accuracy(net, v_out, v_labels, v_mask, v_inputs,
                                         isvalid=True)
                curves[rule]["train"].append(train_acc)
                curves[rule]["valid"].append(valid_acc)
                # grads['loss'] is the masked-MSE training loss for this batch;
                # lr is the current (post-scheduler) learning rate for this rule.
                step_log[rule] = (float(grads["loss"]), float(v_loss),
                                  opt.param_groups[0]["lr"])

        if step in record_set:
            msg = "  ".join(
                f"{cfg.rule_label[r]}: acc tr={curves[r]['train'][-1]:.3f} "
                f"va={curves[r]['valid'][-1]:.3f} | loss tr={step_log[r][0]:.3e} "
                f"va={step_log[r][1]:.3e} | lr={step_log[r][2]:.1e}"
                for r in cfg.rules_to_run)
            print(f"  seed {seed} step {step:>5}   {msg}")

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
    """One figure, two panels (train / test accuracy), each rule mean ± std.
    Driven purely by the passed arrays so it works both live and from a reloaded
    .npz (replot_from_npz)."""
    steps = np.asarray(record_steps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, split, title in zip(axes, ("train", "valid"),
                                ("Training accuracy", "Testing (held-out) accuracy")):
        for rule in rules:
            # Accuracy is stored as a fraction in [0, 1]; plot as percent so a
            # fully-solved task (100%) sits below the 110 upper bound.
            mean = 100.0 * np.asarray(agg[rule][split]["mean"])
            std = 100.0 * np.asarray(agg[rule][split]["std"])
            color = cfg.rule_color.get(rule, None)
            ax.plot(steps, mean, color=color, label=cfg.rule_label.get(rule, rule), lw=2)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 110)
    axes[0].set_ylabel("angle accuracy (%)")
    axes[1].legend(loc="lower right", frameon=False)
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
    # Prefer the title saved with the data; fall back to cfg.title.
    cfg_for_plot = cfg
    if "title" in d.files:
        cfg_for_plot = copy.copy(cfg)
        cfg_for_plot.title = str(d["title"])
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

    # Final-accuracy summary.
    print("\nFinal accuracy (mean ± std over runs):")
    for r in cfg.rules_to_run:
        tr = agg[r]["train"]; va = agg[r]["valid"]
        print(f"  {cfg.rule_label.get(r, r):<16} train {tr['mean'][-1]:.3f} ± {tr['std'][-1]:.3f}"
              f"   test {va['mean'][-1]:.3f} ± {va['std'][-1]:.3f}")
    return agg
