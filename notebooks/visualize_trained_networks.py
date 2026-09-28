#!/usr/bin/env python
# coding: utf-8

# # Visualize trained deep-MPN weights across learning rules
#
# Reload the deep-MPN (`dmpn`) networks saved by `train_mpn.py` (one seed, each
# learning rule) and compare the three learned **weight matrices `W`** (not the
# data-dependent modulation `M`):
#
# | Component | Parameter | Shape | Role |
# |---|---|---|---|
# | **Input layer** | `W_initial_linear.weight` | `(n_embed, n_input)` | raw input → embedding |
# | **Modulation-layer W** | `mp_layers[0].W` | `(n_hidden, n_embed)` | plastic MP weight (base of `W + W⊙M`) |
# | **Output layer** | `W_output` | `(n_output, n_hidden)` | hidden → readout |
#
# This is the RNN-comparable architecture (input → hidden → output), with the
# plastic `M` playing the recurrence role. Set `SEED` / `RULES` below to match a
# completed `train_mpn.py` run (checkpoints are assumed to be `dmpn`).

# In[1]:


import argparse
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401  -- prepends ../core + ../scripts to sys.path
import mpn_tasks
import mpn
import train_mpn as tm      # ONLY for load_net (checkpoint-driven; ignores tm globals)

plt.rcParams["figure.dpi"] = 110

# Rule display labels/colors kept LOCAL so this notebook does not depend on train_mpn globals.
RULE_LABEL = {"bptt": "BPTT", "local_diag_rflo": "diagonal RFLO",
              "local_exact_rowlocal": "exact row-local", "local_direct": "direct"}
RULE_COLOR = {"bptt": "#1f77b4", "local_diag_rflo": "#d62728",
              "local_exact_rowlocal": "#2ca02c", "local_direct": "#9467bd"}


# ## 1. Choose the run to load (by checkpoint name)

# In[2]:


# --- run selection: point directly at saved checkpoints ---
# CKPT_STEM is the checkpoint filename up to (and INCLUDING) the trailing "_" that
# precedes "{rule}_seed{SEED}.pt". Copy it straight from a saved checkpoint name, e.g.
#   dmpn_contextdelaydm1_h200-200_b128_n5000_lr1e-03_direct_fa_eta1.00_lam0.99_
# Loaded per rule:  {CKPT_DIR}/{CKPT_STEM}{rule}_seed{SEED}.pt
# The net AND (below) the task config come from the checkpoints themselves — nothing
# is regenerated from train_mpn's globals.
CKPT_DIR  = tm.CKPT_DIR    # default <root>/checkpoints; set to any path you like
CKPT_STEM = "dmpn_contextdelaydm1_h256_b128_n5000_lr1e-03_exact_spatial_xl1_in-match_eta1.00_lam0.99_bias-direct_"
SEED      = 587             # the seed suffix on the checkpoint files
RULES     = ["bptt", "local_diag_rflo", "local_direct"]
DEVICE    = torch.device("cpu")

parser = argparse.ArgumentParser(description="Save trained MPN weight and bias comparison figures.")
parser.add_argument("--ckpt-dir", type=Path, default=Path(CKPT_DIR))
parser.add_argument("--ckpt-stem", default=CKPT_STEM)
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--rules", nargs="+", choices=list(RULE_LABEL), default=RULES)
parser.add_argument("--output-dir", type=Path,
                    default=_bootstrap.ROOT / "notebooks" / "visualize_trained_networks")
args = parser.parse_args()
CKPT_DIR, CKPT_STEM = args.ckpt_dir.expanduser(), args.ckpt_stem
SEED, RULES = args.seed, args.rules

nets, ckpts = {}, {}
for rule in RULES:
    path = os.path.join(CKPT_DIR, f"{CKPT_STEM}{rule}_seed{SEED}.pt")
    ckpts[rule] = torch.load(path, map_location=DEVICE, weights_only=False)
    nets[rule]  = tm.load_net(path, device=DEVICE)   # rebuilds the net from ckpt net_params
    print(f"{rule:20} <- {os.path.basename(path)}")

any_net = nets[RULES[0]]
any_ck  = ckpts[RULES[0]]
RULESET = any_ck.get("ruleset") or any_ck.get("net_params", {}).get("ruleset", "?")
n_input  = any_net.n_input
n_hidden = any_net.n_hidden
n_output = any_net.n_output
print(f"\nruleset={RULESET}  n_input={n_input}  n_hidden={n_hidden}  n_output={n_output}")

# --- Output folder for saved figures (created if missing) ---
# Figures from this script land in <root>/notebooks/visualize_trained_networks/, named from the
# checkpoint stem + seed + a per-figure suffix (self-describing, no cross-run
# collisions). save_fig(fig, suffix) writes <stem>seed<SEED>_<suffix>.png.
import _bootstrap
FIG_OUT_DIR = args.output_dir.expanduser().resolve()
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)   # create the intermediate folder

def save_fig(fig, suffix, dpi=150):
    """Save `fig` to FIG_OUT_DIR as <CKPT_STEM>seed<SEED>_<suffix>.png and return the path."""
    fname = f"{CKPT_STEM}seed{SEED}_{suffix}.png"
    out = FIG_OUT_DIR / fname
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure -> {out}")
    return out

print(f"figures will be saved to: {FIG_OUT_DIR}")

import json as _json
def save_json(obj, suffix):
    """Save `obj` (JSON-serializable) to FIG_OUT_DIR as <CKPT_STEM>seed<SEED>_<suffix>.json
    and return the path. Same naming convention as save_fig, so a figure and its
    underlying numbers share a stem."""
    out = FIG_OUT_DIR / f"{CKPT_STEM}seed{SEED}_{suffix}.json"
    with open(out, "w", encoding="utf-8") as fh:
        _json.dump(obj, fh, indent=2)
    print(f"saved json   -> {out}")
    return out


# ## 2. Extract the trained weight matrices
#
# We visualize the learned **weights `W`** only (not the data-dependent modulation
# `M`). For `dmpn` there are three: `W_initial_linear.weight` (input→embedding),
# `mp_layers[0].W` (the plastic MP-layer weight), and `W_output` (readout).

# In[3]:


# Weight matrices per rule, adaptive to ANY number of MP layers. comp[rule] holds
# W_in (input embedding), one W_mp{k} per MP layer k (0-based), and W_out (readout).
# `cols` is the ordered (key, title) list every downstream figure iterates over, so
# a deep MP stack shows each modulation layer's static weight SEPARATELY.
n_layers = len(any_net.mp_layers)

cols = [("W_in", "input layer  (embed×input)")]
for k in range(n_layers):
    cols.append((f"W_mp{k}", f"modulation layer {k}  W  (post×pre)"))
cols.append(("W_out", "output layer  W_out  (output×hidden)"))

comp = {}   # comp[rule] = {'W_in', 'W_mp0', 'W_mp1', ..., 'W_out'}
for rule, net in nets.items():
    d = {"W_in": net.W_initial_linear.weight.detach().cpu().numpy()}   # (n_embed, n_input)
    for k, mp in enumerate(net.mp_layers):
        d[f"W_mp{k}"] = mp.W.detach().cpu().numpy()                    # (post, pre)
    d["W_out"] = net.W_output.detach().cpu().numpy()                   # (n_output, n_hidden)
    comp[rule] = d
    mp_mags = "  ".join(f"|W_mp{k}| {np.abs(d[f'W_mp{k}']).mean():.3e}" for k in range(n_layers))
    print(f"{rule:20} |W_in| {np.abs(d['W_in']).mean():.3e}  {mp_mags}  "
          f"|W_out| {np.abs(d['W_out']).mean():.3e}")


# ## 3. Weight heatmaps — one row per learning rule
#
# Columns: **input layer** `W_initial_linear`, **modulation-layer** `mp_layers[0].W`,
# and **output layer** `W_output`. A shared symmetric color scale per column makes
# rules directly comparable.

# In[4]:


# Shared symmetric vlim per column (comparable across rules).
vlim = {key: max(np.abs(comp[r][key]).max() for r in RULES) for key, _ in cols}

nrows, ncols = len(RULES), len(cols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                         squeeze=False)
for i, rule in enumerate(RULES):
    for j, (key, title) in enumerate(cols):
        ax = axes[i][j]
        v = vlim[key]
        im = ax.imshow(comp[rule][key], aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
        if i == 0:
            ax.set_title(title, fontsize=10)
        if j == 0:
            ax.set_ylabel(f"{RULE_LABEL.get(rule, rule)}\n(post unit)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle(f"Trained deep-MPN weights ({n_layers} MP layer"
             f"{'s' if n_layers != 1 else ''}) — seed {SEED}", y=1.002, fontsize=13)
fig.tight_layout()
save_fig(fig, "weight_heatmaps")


# ### 3b. Trained-weight alignment to BPTT (cosine similarity & L1 distance)
#
# All rules start from the SAME initialization (deepcopy) and see the SAME data, so
# after training the differences in the LEARNED weights measure how far each local
# rule's solution drifts from the exact-gradient (BPTT) solution, per weight matrix.
# We report, for every non-BPTT rule and each component (input embedding, each MP
# layer's W, readout):
#   • cosine similarity  cos(w_rule, w_bptt)  — direction match (1 = identical direction)
#   • L1 distance        mean |w_rule − w_bptt|  (per-weight; scale-comparable across
#                        matrices), plus the relative L1  ‖Δ‖₁ / ‖w_bptt‖₁.
# BPTT is the reference, so it is excluded from the comparison.

# In[5]:


# Compare each rule's TRAINED weights to BPTT's, per component, via cosine + L1.
REF = "bptt"
if REF not in comp or len(RULES) < 2:
    print(f"reference '{REF}' not among loaded rules {list(comp)}; skipping alignment.")
else:
    keys = [k for k, _ in cols]                     # W_in, W_mp0, ..., W_out
    others = [r for r in RULES if r != REF]

    def _cos(a, b):
        a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")

    def _l1(a, b):        # mean |Δ| per weight
        return float(np.abs(a.ravel() - b.ravel()).mean())

    def _rel_l1(a, b):    # ‖Δ‖₁ / ‖b‖₁  (scale-free)
        denom = np.abs(b.ravel()).sum()
        return float(np.abs(a.ravel() - b.ravel()).sum() / denom) if denom > 0 else float("nan")

    cos_tab = {r: {} for r in others}   # cos_tab[rule][key]
    l1_tab  = {r: {} for r in others}
    rel_tab = {r: {} for r in others}
    for r in others:
        for k in keys:
            a, b = comp[r][k], comp[REF][k]
            cos_tab[r][k] = _cos(a, b)
            l1_tab[r][k]  = _l1(a, b)
            rel_tab[r][k] = _rel_l1(a, b)

    # Table.
    print(f"Trained-weight alignment to {RULE_LABEL.get(REF, REF)} "
          f"(seed {SEED}, {n_layers} MP layer{'s' if n_layers != 1 else ''}):")
    print(f"  {'rule':16} {'component':10} {'cosine':>9} {'mean L1':>10} {'rel L1':>9}")
    for r in others:
        for k in keys:
            print(f"  {RULE_LABEL.get(r, r):16} {k:10} "
                  f"{cos_tab[r][k]:>9.4f} {l1_tab[r][k]:>10.3e} {rel_tab[r][k]:>9.3f}")

    # Grouped bar charts: cosine (left) and mean-L1 (right), x = component, bars per rule.
    xs = np.arange(len(keys))
    width = 0.8 / max(len(others), 1)
    fig, (axc, axl) = plt.subplots(1, 2, figsize=(6.5 + 1.2 * len(keys), 4.0))
    for i, r in enumerate(others):
        off = (i - (len(others) - 1) / 2) * width
        c = RULE_COLOR.get(r)
        axc.bar(xs + off, [cos_tab[r][k] for k in keys], width, color=c,
                label=RULE_LABEL.get(r, r))
        axl.bar(xs + off, [l1_tab[r][k] for k in keys], width, color=c,
                label=RULE_LABEL.get(r, r))
    axc.axhline(1.0, color="0.7", lw=0.8, ls=":")   # perfect direction match
    axc.set_ylabel(f"cosine similarity to {RULE_LABEL.get(REF, REF)}")
    axc.set_ylim(min(0.0, min(cos_tab[r][k] for r in others for k in keys)) - 0.05, 1.05)
    axl.set_ylabel(f"mean |Δw| to {RULE_LABEL.get(REF, REF)}  (L1)")
    for ax in (axc, axl):
        ax.set_xticks(xs); ax.set_xticklabels(keys, rotation=20, fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    axc.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Trained-weight alignment to {RULE_LABEL.get(REF, REF)} — seed {SEED}",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    save_fig(fig, "weight_alignment_to_bptt")


# ### 3c. Trained-bias alignment to BPTT (cosine similarity & L1 distance)
#
# Same comparison as §3b but for the trained **biases** rather than weights: for each
# non-BPTT rule and each bias vector present (input-embedding `b_in`, each MP layer's
# `b`/`b{k}`, readout `b_out`), report cosine similarity and L1 distance to the BPTT
# bias. Biases absent on the net (e.g. no bias on a layer, or a frozen/untrained
# embedding) are skipped. This cell only COMPUTES + plots into `bias_fig`; the next
# cell saves the figure and the numbers.

# In[6]:


# Extract each rule's trained BIAS vectors, then compare to BPTT (cosine + L1).
# Reuses the _cos/_l1/_rel_l1 helpers defined in §3b.
REF = "bptt"

def _bias_vectors(net):
    """Ordered {key: 1-D bias array} for a net: b_in (trainable embedding, if any),
    each MP layer's b (b, b1, b2, ...), then b_out. Missing biases are omitted."""
    d = {}
    if getattr(net, "input_layer_active", False) and net.W_initial_linear.bias is not None:
        d["b_in"] = net.W_initial_linear.bias.detach().cpu().numpy()
    for k, mp in enumerate(net.mp_layers):
        if getattr(mp, "layer_bias", False):
            d[f"b_mp{k}"] = mp.b.detach().cpu().numpy()
    if getattr(net, "b_output_active", False):
        d["b_out"] = net.b_output.detach().cpu().numpy()
    return d

bias_comp = {rule: _bias_vectors(net) for rule, net in nets.items()}
bias_align = {}   # filled below; consumed by the save cell

if REF not in bias_comp:
    print(f"reference '{REF}' not loaded; skipping bias alignment.")
    bias_fig = None
else:
    # Only compare bias keys present for EVERY rule (so shapes line up).
    bkeys = [k for k in bias_comp[REF] if all(k in bias_comp[r] for r in RULES)]
    others = [r for r in RULES if r != REF]
    if not bkeys or not others:
        print("no biases present on the nets; nothing to compare.")
        bias_fig = None
    else:
        bcos = {r: {} for r in others}
        bl1  = {r: {} for r in others}
        brel = {r: {} for r in others}
        for r in others:
            for k in bkeys:
                a, b = bias_comp[r][k], bias_comp[REF][k]
                bcos[r][k] = _cos(a, b); bl1[r][k] = _l1(a, b); brel[r][k] = _rel_l1(a, b)

        # Table.
        print(f"Trained-bias alignment to {RULE_LABEL.get(REF, REF)} "
              f"(seed {SEED}, {n_layers} MP layer{'s' if n_layers != 1 else ''}):")
        print(f"  {'rule':16} {'bias':8} {'cosine':>9} {'mean L1':>10} {'rel L1':>9}")
        for r in others:
            for k in bkeys:
                print(f"  {RULE_LABEL.get(r, r):16} {k:8} "
                      f"{bcos[r][k]:>9.4f} {bl1[r][k]:>10.3e} {brel[r][k]:>9.3f}")

        # Grouped bar charts: cosine (left), mean-L1 (right); x = bias, bars per rule.
        xs = np.arange(len(bkeys))
        width = 0.8 / max(len(others), 1)
        bias_fig, (axc, axl) = plt.subplots(1, 2, figsize=(6.5 + 1.2 * len(bkeys), 4.0))
        for i, r in enumerate(others):
            off = (i - (len(others) - 1) / 2) * width
            c = RULE_COLOR.get(r)
            axc.bar(xs + off, [bcos[r][k] for k in bkeys], width, color=c,
                    label=RULE_LABEL.get(r, r))
            axl.bar(xs + off, [bl1[r][k] for k in bkeys], width, color=c,
                    label=RULE_LABEL.get(r, r))
        axc.axhline(1.0, color="0.7", lw=0.8, ls=":")
        axc.set_ylabel(f"cosine similarity to {RULE_LABEL.get(REF, REF)}")
        axc.set_ylim(min(0.0, min(bcos[r][k] for r in others for k in bkeys)) - 0.05, 1.05)
        axl.set_ylabel(f"mean |Δb| to {RULE_LABEL.get(REF, REF)}  (L1)")
        for ax in (axc, axl):
            ax.set_xticks(xs); ax.set_xticklabels(bkeys, rotation=20, fontsize=8)
            ax.grid(alpha=0.3, axis="y")
        axc.legend(frameon=False, fontsize=8)
        bias_fig.suptitle(f"Trained-bias alignment to {RULE_LABEL.get(REF, REF)} — seed {SEED}",
                          y=1.02, fontsize=12)
        bias_fig.tight_layout()

        # JSON-ready record (saved in the next cell).
        bias_align = {
            "ruleset": RULESET, "seed": int(SEED), "ckpt_stem": CKPT_STEM,
            "reference": REF, "bias_keys": bkeys, "rules": list(RULES),
            "cosine": {r: {k: bcos[r][k] for k in bkeys} for r in others},
            "l1":     {r: {k: bl1[r][k]  for k in bkeys} for r in others},
            "rel_l1": {r: {k: brel[r][k] for k in bkeys} for r in others},
        }


# In[7]:


# Save the §3c bias-alignment figure + numbers (separate cell, per request).
if bias_fig is not None:
    save_fig(bias_fig, "bias_alignment_to_bptt")
    save_json(bias_align, "bias_alignment_to_bptt")
else:
    print("no bias-alignment figure to save (no comparable biases).")


# ## 4. Weight distributions (overlaid histograms)
#
# For each weight matrix, overlay the flattened weight distributions of
# the learning rules on shared bins. This shows how the *shape/spread* of
# the learned weights differs across rules (e.g. whether a local rule
# learns systematically larger / more sparse weights than BPTT), which the
# heatmaps above don't make quantitative.

# In[8]:


fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 4.0),
                         squeeze=False)
for j, (key, title) in enumerate(cols):
    ax = axes[0][j]
    # Shared symmetric bins across rules for a fair overlay.
    vmax = max(np.abs(comp[r][key]).max() for r in RULES)
    bins = np.linspace(-vmax, vmax, 50)
    for rule in RULES:
        w = comp[rule][key].ravel()
        ax.hist(w, bins=bins, density=True, histtype='step', linewidth=1.8,
                color=RULE_COLOR.get(rule), label=RULE_LABEL.get(rule, rule))
        ax.axvline(w.mean(), color=RULE_COLOR.get(rule), ls=':', lw=1, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('weight value')
    ax.grid(alpha=0.3)
axes[0][0].set_ylabel('density')
axes[0][-1].legend(frameon=False)
fig.suptitle(f'Weight distributions by learning rule — seed {SEED}', y=1.02)
fig.tight_layout()
save_fig(fig, "weight_distributions")

# Per-rule distribution statistics (mean, std, max|w|) for every component.
print(f"{'component':10} {'rule':16} {'mean':>9} {'std':>9} {'max|w|':>9}")
for key, _ in cols:
    for rule in RULES:
        w = comp[rule][key].ravel()
        print(f'{key:10} {RULE_LABEL.get(rule, rule):16} '
              f'{w.mean():>9.3e} {w.std():>9.3e} {np.abs(w).max():>9.3e}')


# In[ ]:




