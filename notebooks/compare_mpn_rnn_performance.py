#!/usr/bin/env python
# coding: utf-8

"""Overlay saved MPN and RNN train/test curves from --mpn-file and --rnn-file.

Names (with optional .npz extension) resolve against figure_data/; explicit paths
are also accepted. Task, hidden widths, and feedback mode come from saved metadata,
with filename parsing as a fallback for legacy files. Hidden widths, feedback,
and plotted metric must match; different tasks produce a warning.
MPN curves are solid, RNN curves dashed, and colors identify learning rules.
"""

import argparse
import hashlib
import pathlib
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401  -- prepends ../core + ../scripts to sys.path; exposes ROOT
import train_mpn as tm   # RULE_LABEL / RULE_COLOR for the MPN rules
import train_rnn as tr   # RULE_LABEL / RULE_COLOR for the RNN rules

plt.rcParams["figure.dpi"] = 110

FIG_DATA_DIR = _bootstrap.ROOT / "figure_data"   # where the .npz files live
FIG_OUT_DIR = _bootstrap.ROOT / "notebooks" / "compare_mpn_rnn_performance"


parser = argparse.ArgumentParser(description="Compare saved MPN and RNN learning curves.")
parser.add_argument("--mpn-file", required=True, help="saved MPN .npz name or path")
parser.add_argument("--rnn-file", required=True, help="saved RNN .npz name or path")
parser.add_argument("--output-dir", type=pathlib.Path, default=FIG_OUT_DIR)
args = parser.parse_args()
MPN_FILE, RNN_FILE = args.mpn_file, args.rnn_file
FIG_OUT_DIR = args.output_dir.expanduser().resolve()


def resolve_npz(name):
    """Resolve a user-supplied name to an existing .npz path. Accepts a bare stem,
    a 'name.npz', or an absolute/relative path; bare names look in figure_data/."""
    name = str(name)
    if not name.endswith(".npz"):
        name += ".npz"
    p = pathlib.Path(name).expanduser()
    if not p.is_absolute() and not p.exists():
        p = FIG_DATA_DIR / p.name
    if not p.exists():
        raise FileNotFoundError(f"npz not found: {p}")
    return p


MPN_PATH = resolve_npz(MPN_FILE)
RNN_PATH = resolve_npz(RNN_FILE)
print(f"MPN <- {MPN_PATH}")
print(f"RNN <- {RNN_PATH}")


# Each `.npz` (written by `train_common.save_plot_data`) stores the x-axis
# (`record_steps`), the list of `rules`, and per-rule `mean__<rule>__<split>` /
# `std__<rule>__<split>` arrays for `split` in `{train, valid}`.


def load_curves(path):
    """Return (record_steps, rules, agg, meta) from a train_common .npz.
    agg[rule][split] = {'mean': array, 'std': array}; meta holds the scalars.
    meta['metric'] is 'accuracy' or 'loss' (what the stored curves represent);
    meta['acc_label'] is the y-axis label. Both default sensibly for older .npz."""
    d = np.load(path, allow_pickle=True)
    rules = [str(r) for r in d["rules"]]
    steps = np.asarray(d["record_steps"])
    agg = {r: {sp: {"mean": np.asarray(d[f"mean__{r}__{sp}"]),
                    "std": np.asarray(d[f"std__{r}__{sp}"])}
               for sp in ("train", "valid")}
           for r in rules}
    scalar = lambda k: (d[k].item() if k in d.files and d[k].shape == () else None)
    meta = {k: scalar(k) for k in ("ruleset", "n_hidden", "n_runs", "batch",
                                   "lr", "feedback_mode", "title", "metric",
                                   "acc_label", "arch_tag")}
    # metric/acc_label were added later; default to accuracy for older .npz.
    meta["metric"] = meta["metric"] or "accuracy"
    meta["acc_label"] = meta["acc_label"] or "angle accuracy (%)"
    d.close()
    return steps, rules, agg, meta


def parse_descriptors(name):
    """Legacy filename fallback: (task, hidden width or width tuple, feedback mode).

    Recognizes current feedback names and the older exact_readout/random_fixed
    names. Compact run IDs need metadata and return None for missing fields.
    """
    stem = pathlib.Path(str(name)).name
    task = re.search(r"train_(?:dmpn|mpn1|rnn)_(.+?)_h\d+", stem)
    hidden = re.search(r"_h(\d+(?:-\d+)*)_", stem)
    readout = re.search(r"_(exact_spatial|layerwise_fa|direct_fa|exact_readout|random_fixed)(?:_|$)", stem)
    widths = tuple(int(width) for width in hidden.group(1).split('-')) if hidden else ()
    return (task.group(1) if task else None,
            (widths[0] if len(widths) == 1 else widths) if widths else None,
            readout.group(1) if readout else None)


def run_descriptors(path, meta):
    """Prefer saved architecture/feedback over filenames, including deep stacks."""
    task, hidden, feedback = parse_descriptors(path)
    arch_tag = meta.get('arch_tag')
    if arch_tag:
        match = re.fullmatch(r'h(\d+(?:-\d+)*)', arch_tag)
        if not match:
            raise ValueError(f"Unsupported saved arch_tag {arch_tag!r} in {path}")
        widths = tuple(int(width) for width in match[1].split('-'))
        hidden = widths[0] if len(widths) == 1 else widths
    elif meta.get('n_hidden') is not None:
        hidden = meta['n_hidden']
    feedback = meta.get('feedback_mode') or feedback
    if feedback == 'exact_readout':
        feedback = 'exact_spatial'
    return meta.get('ruleset') or task, hidden, feedback


mpn_steps, mpn_rules, mpn_agg, mpn_meta = load_curves(MPN_PATH)
rnn_steps, rnn_rules, rnn_agg, rnn_meta = load_curves(RNN_PATH)

mpn_task, mpn_hidden, mpn_readout = run_descriptors(MPN_PATH, mpn_meta)
rnn_task, rnn_hidden, rnn_readout = run_descriptors(RNN_PATH, rnn_meta)

# Hidden widths, feedback mode, and plotted metric must agree for a
# same-axes comparison.
for label, mval, rval in [("hidden size", mpn_hidden, rnn_hidden),
                          ("feedback mode", mpn_readout, rnn_readout),
                          ("plotted metric", mpn_meta["metric"], rnn_meta["metric"])]:
    if mval is None or rval is None:
        raise ValueError(f"could not read {label} (MPN={mval!r}, RNN={rval!r}).")
    if mval != rval:
        raise ValueError(f"{label} differs between the two runs: "
                         f"MPN={mval!r} vs RNN={rval!r}. The combined figure "
                         "requires them to match.")

# Task is cross-checked too, but only warns (you may deliberately overlay tasks).
if mpn_task != rnn_task:
    print(f"WARNING: task differs between files (MPN={mpn_task!r}, RNN={rnn_task!r}) "
          "- overlaying anyway.")

TASK, HIDDEN, READOUT = mpn_task, mpn_hidden, mpn_readout
METRIC, Y_LABEL = mpn_meta["metric"], mpn_meta["acc_label"]
print(f"task = {TASK}   hidden = {HIDDEN}   feedback = {READOUT}   metric = {METRIC}")
print(f"MPN rules: {mpn_rules}  ->  {[tm.RULE_LABEL.get(r, r) for r in mpn_rules]}")
print(f"RNN rules: {rnn_rules}  ->  {[tr.RULE_LABEL.get(r, r) for r in rnn_rules]}")
print(f"n_runs: MPN={mpn_meta['n_runs']}  RNN={rnn_meta['n_runs']}")


# Two panels (train / test) like the per-model figures, but with **both** models on
# each: MPN solid, RNN dashed, color per learning rule (mean line + ± std band).
# The legend is tagged `MODEL: rule` so every curve is unambiguous, and the title
# records the task, hidden widths, and feedback mode.


# Per-model style: solid MPN, dashed RNN; reuse each script's rule color/label maps.
SOURCES = [
    dict(model="MPN", linestyle="-", steps=mpn_steps, rules=mpn_rules, agg=mpn_agg,
         color=tm.RULE_COLOR, label=tm.RULE_LABEL, meta=mpn_meta),
    dict(model="RNN", linestyle="--", steps=rnn_steps, rules=rnn_rules, agg=rnn_agg,
         color=tr.RULE_COLOR, label=tr.RULE_LABEL, meta=rnn_meta),
]

# The plotted metric follows the .npz: 'accuracy' -> percent, linear 0-110 y;
# 'loss' -> masked-MSE, log y (matches train_common.plot).
is_loss = (METRIC == "loss")
scale = 1.0 if is_loss else 100.0
noun = "loss" if is_loss else "accuracy"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, split, title in zip(axes, ("train", "valid"),
                            (f"Training {noun}", f"Testing (held-out) {noun}")):
    for src in SOURCES:
        for rule in src["rules"]:
            mean = scale * np.asarray(src["agg"][rule][split]["mean"])
            std = scale * np.asarray(src["agg"][rule][split]["std"])
            color = src["color"].get(rule, None)
            lbl = f"{src['model']}: {src['label'].get(rule, rule)}"
            ax.plot(src["steps"], mean, color=color, ls=src["linestyle"], lw=2, label=lbl)
            ax.fill_between(src["steps"], mean - std, mean + std, color=color, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3)
    if is_loss:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 110)
axes[0].set_ylabel(Y_LABEL)
axes[1].legend(loc="best" if is_loss else "lower right", frameon=False, fontsize=8)

n_runs = mpn_meta["n_runs"] if mpn_meta["n_runs"] == rnn_meta["n_runs"] else \
    f"{mpn_meta['n_runs']}/{rnn_meta['n_runs']}"
fig.suptitle(f"{TASK}: MPN vs RNN — BPTT vs local learning  "
             f"(hidden={HIDDEN}, feedback={READOUT}, "
             f"mean ± std over {n_runs} runs)")
fig.tight_layout()


# Writes to `notebooks/compare_mpn_rnn_performance/`; descriptors and a source-pair
# identifier distinguish different comparisons.


FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
source_tag = hashlib.sha256(f"{MPN_PATH.resolve()}\n{RNN_PATH.resolve()}".encode()).hexdigest()[:10]
out_path = FIG_OUT_DIR / f"compare_mpn_rnn_{TASK}_h{HIDDEN}_{READOUT}_{source_tag}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure: {out_path}")
