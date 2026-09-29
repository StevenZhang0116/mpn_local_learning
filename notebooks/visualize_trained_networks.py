#!/usr/bin/env python
"""Plot trained deep-MPN parameters and held-out performance from saved checkpoints."""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap
import mpn
import mpn_tasks
from mpn import masked_mse_loss_only
import train_mpn as tm


RULE_LABEL = {"bptt": "BPTT", "local_diag_rflo": "diagonal RFLO",
              "local_exact_rowlocal": "exact row-local", "local_direct": "direct"}
RULE_COLOR = {"bptt": "#1f77b4", "local_diag_rflo": "#d62728",
              "local_exact_rowlocal": "#2ca02c", "local_direct": "#9467bd"}
DEFAULT_RULES = ("bptt", "local_diag_rflo", "local_direct")


def select_checkpoints(directory, rules, stem=None, seed=None):
    """Pick the newest complete group containing every requested rule.

    Rank groups by the newest requested member's file modification time, then
    stem and numeric seed (largest wins). This does not inspect training status.
    Explicit stem/seed values filter candidates; groups are never mixed.
    Return (stem, seed, paths_by_rule), preserving the requested rule order.
    """
    pattern = re.compile(r"^(?P<stem>.+_)(?P<rule>" +
                         "|".join(map(re.escape, RULE_LABEL)) +
                         r")_seed(?P<seed>\d+)\.pt$")
    groups = {}
    for path in Path(directory).expanduser().glob("*.pt"):
        match = pattern.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        found_stem, found_seed = match["stem"], int(match["seed"])
        if stem is not None and found_stem != stem:
            continue
        if seed is not None and found_seed != seed:
            continue
        groups.setdefault((found_stem, found_seed), {})[match["rule"]] = path
    complete = {key: paths for key, paths in groups.items()
                if all(rule in paths for rule in rules)}
    if not complete:
        available = "; ".join(
            f"{s}seed{n}: missing {', '.join(r for r in rules if r not in paths)}"
            for (s, n), paths in sorted(groups.items()))
        raise ValueError(
            f"No complete checkpoint group in {directory} for rules {', '.join(rules)} "
            f"(stem={stem!r}, seed={seed!r}). " +
            (available or "Check --ckpt-dir, --ckpt-stem, and --seed."))
    key = max(complete, key=lambda k: (
        max(complete[k][r].stat().st_mtime_ns for r in rules), k[0], k[1]))
    return *key, {rule: complete[key][rule] for rule in rules}


@dataclass
class AnalysisContext:
    """Loaded models and saved task settings shared by both plotting functions."""

    stem: str
    seed: int
    nets: dict
    ruleset: str
    task_params: Optional[dict]
    output_dir: Path
    trials: int

    @property
    def rules(self):
        return list(self.nets)

    def save_fig(self, fig, suffix, dpi=150):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.stem}seed{self.seed}_{suffix}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure -> {path}")
        return path


def load_context(args):
    """Load each model once on CPU; use the first requested rule's task metadata."""
    stem, seed, paths = select_checkpoints(
        args.ckpt_dir, args.rules, args.ckpt_stem, args.seed)
    print(f"Selected checkpoint group: {stem}seed{seed}")
    nets = {}
    first = None
    for rule, path in paths.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt["net_params"].get("net_type") != "dmpn":
            raise ValueError(f"{path.name}: these visualizations require a deep MPN (dmpn).")
        if first is None:
            first = ckpt
        net = mpn.DeepMultiPlasticNet(ckpt["net_params"], verbose=False)
        # Preserve the saved precision instead of relying on training-script globals.
        net = net.to(device="cpu", dtype=ckpt["state_dict"]["W_output"].dtype)
        net.load_state_dict(ckpt["state_dict"])
        net.learning_rule = ckpt.get("learning_rule", rule)
        net.eval()
        nets[rule] = net
        print(f"{rule:20} <- {path.name}")
    task_params = first.get("task_params")
    if args.analysis != "weights":
        if not task_params or not {"hp", "rules"} <= task_params.keys():
            raise ValueError("Performance analysis requires saved ring-task task_params "
                             "(hp and rules). Use --analysis weights for older or "
                             "non-ring checkpoints; current training settings are not substituted.")
    ruleset = first.get("ruleset") or first["net_params"].get("ruleset", "?")
    context = AnalysisContext(stem, seed, nets, ruleset, task_params,
                              args.output_dir.expanduser().resolve(), args.trials)
    print(f"ruleset={ruleset}; figures will be saved to: {context.output_dir}")
    return context


def plot_weights(context):
    """Save parameter comparisons using the networks already loaded in context."""
    nets, RULES, SEED = context.nets, context.rules, context.seed
    any_net = next(iter(nets.values()))
    save_fig = context.save_fig

    # Extract the input embedding, every MP layer's static W, and the readout.
    # The data-dependent modulation M is plotted by plot_performance.
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


    # Weight heatmaps: one row per rule, one column per weight matrix, with a
    # shared symmetric color scale within each column.
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


    # Compare weights to BPTT using cosine similarity, mean absolute difference,
    # and relative L1 distance. Shared initialization/data are a training-run
    # assumption; this analysis does not verify their provenance.
    def _cos(a, b):
        a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


    def _l1(a, b):
        return float(np.abs(a.ravel() - b.ravel()).mean())

    def _rel_l1(a, b):
        denom = np.abs(b.ravel()).sum()
        return float(np.abs(a.ravel() - b.ravel()).sum() / denom) if denom > 0 else float("nan")

    REF = "bptt"
    if REF not in comp or len(RULES) < 2:
        print("weight alignment requires BPTT and at least one other rule; skipping alignment.")
    else:
        keys = [k for k, _ in cols]                     # W_in, W_mp0, ..., W_out
        others = [r for r in RULES if r != REF]

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


    # Compare available bias vectors to BPTT using the same metrics.
    def _bias_vectors(net):
        """Return b_in, b_mp{k}, and b_out arrays for biases present on the net.

        Input biases are included when the input layer is active, even if frozen.
        """
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

    if REF not in bias_comp:
        print(f"reference '{REF}' not loaded; skipping bias alignment.")
    else:
        # Only compare bias keys present for every loaded rule.
        bkeys = [k for k in bias_comp[REF] if all(k in bias_comp[r] for r in RULES)]
        others = [r for r in RULES if r != REF]
        if not bkeys or not others:
            print("bias alignment requires shared biases and a non-BPTT rule; skipping alignment.")
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
            save_fig(bias_fig, "bias_alignment_to_bptt")


    # Learned-parameter cosine similarity: diagonal RFLO vs direct
    #
    # Compare flattened trained weights and available biases, one value per tensor.
    # This comparison needs only the two local rules; BPTT is optional. Zero-norm
    # tensors have undefined cosine similarity and are marked N/A.

    pair = ("local_diag_rflo", "local_direct")
    if not all(rule in comp for rule in pair):
        print("diagonal RFLO and direct are not both loaded; skipping their parameter alignment.")
    else:
        pair_cos = {}
        for group, components in (("Weights", comp), ("Biases", bias_comp)):
            left, right = (components[rule] for rule in pair)
            scores = {}
            for key in left:
                if key not in right or left[key].shape != right[key].shape:
                    print(f"skipping diagonal RFLO vs direct {key}: missing or mismatched parameter.")
                    continue
                value = _cos(left[key], right[key])
                scores[key] = value if np.isfinite(value) else None
            if scores:
                pair_cos[group] = scores

        if pair_cos:
            fig, axes = plt.subplots(1, len(pair_cos), squeeze=False,
                                     figsize=(sum(max(4.5, 1.2 * len(s)) for s in pair_cos.values()), 4.2))
            for ax, (group, scores) in zip(axes[0], pair_cos.items()):
                xs = np.arange(len(scores))
                values = [v if v is not None else np.nan for v in scores.values()]
                ax.bar(xs, values, color=RULE_COLOR[pair[0]], width=0.65)
                for x, (key, value) in enumerate(scores.items()):
                    label = f"{value:.3f}" if value is not None else "N/A"
                    y = value if value is not None else 0.0
                    ax.annotate(label, (x, y), xytext=(0, 4 if y >= 0 else -4),
                                textcoords="offset points", ha="center",
                                va="bottom" if y >= 0 else "top", fontsize=9)
                    print(f"diagonal RFLO vs direct {key:10} cosine={label}")
                ax.axhline(0.0, color="0.7", lw=0.8)
                ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
                ax.set_ylim(-1.15, 1.15)
                ax.set_xticks(xs)
                ax.set_xticklabels(list(scores), rotation=20, fontsize=9)
                ax.set_title(group)
                ax.set_ylabel("cosine similarity")
                ax.grid(alpha=0.3, axis="y")
            fig.suptitle(f"Learned parameters: diagonal RFLO vs direct — seed {SEED}")
            fig.tight_layout()
            save_fig(fig, "parameter_cosine_diag_rflo_vs_direct")
        else:
            print("no matching parameters for diagonal RFLO vs direct; skipping alignment.")


    # Weight distributions (overlaid histograms)
    #
    # For each weight matrix, overlay the flattened weight distributions of
    # the learning rules on shared bins. This shows how the *shape/spread* of
    # the learned weights differs across rules (e.g. whether a local rule
    # learns systematically larger / more sparse weights than BPTT), which the
    # heatmaps above don't make quantitative.


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


def plot_performance(context):
    """Compare random/random_batch accuracy and MSE using saved task settings.

    Each timing mode generates context.trials trials per task, shared across
    learning rules. Example and modulation figures use only random_batch data.
    """
    nets, RULES, SEED = context.nets, context.rules, context.seed
    any_net = next(iter(nets.values()))
    RULESET = context.ruleset
    DEVICE = torch.device("cpu")
    save_fig = context.save_fig
    task_params = context.task_params
    N_TRIALS = context.trials

    def generate_batch(mode_input):
        # Reset each RNG per mode so results do not depend on evaluation order.
        params = deepcopy(task_params)
        params["hp"]["rng"] = np.random.RandomState(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        data, _ = mpn_tasks.generate_trials_wrap(
            params, N_TRIALS, rules=params["rules"],
            mode_input=mode_input, device=DEVICE)
        dtype = next(any_net.parameters()).dtype
        return tuple(d.to(dtype) for d in data)

    # These trials and predictions also drive the example and modulation plots.
    inputs, labels, mask = generate_batch("random_batch")
    inputs_np = inputs.cpu().numpy()
    labels_np = labels.cpu().numpy()          # target output (B, T, n_output)

    print(f"random_batch trials: inputs {inputs_np.shape}  target {labels_np.shape}")
    print(f"task = {RULESET}  |  n_input={inputs_np.shape[-1]}  n_output={labels_np.shape[-1]}")


    # Within each timing mode, reuse predictions for both accuracy metrics and
    # masked MSE. Keep random_batch predictions for the example-trial plot.
    # Forward chunking bounds modulation state, not the full-batch output tensor.
    ACC_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ACC_BATCH  = 32          # forward minibatch (lower this if you still hit GPU OOM)
    ACC_MODES  = ["angle", "stimulus"]
    print(f"accuracy device: {ACC_DEVICE}  |  forward minibatch: {ACC_BATCH}  |  modes: {ACC_MODES}")


    @torch.no_grad()
    def forward_outputs_chunked(net, inputs, chunk, device):
        """Run minibatched forwards and concatenate (B, T, out) on `device`.

        The caller moves and restores the model. Each chunk resets its state;
        changing batch size can introduce floating-point differences.
        Outputs from all chunks remain on the device until scoring finishes.
        """
        B = inputs.shape[0]
        outs = []
        for i in range(0, B, chunk):
            xb = inputs[i:i + chunk].to(device)
            outs.append(tm.forward_outputs(net, xb))
        return torch.cat(outs, dim=0)


    def score_batch(batch_inputs, batch_labels, batch_mask, keep_outputs=False):
        """Score one timing-mode batch; optionally retain CPU predictions for plots."""
        batch_outputs = {}
        accs = {m: {} for m in ACC_MODES}
        mse_losses = {}
        lab_d = batch_labels.to(ACC_DEVICE)
        msk_d = batch_mask.to(ACC_DEVICE)
        for rule, net in nets.items():
            orig_dev = next(net.parameters()).device
            try:
                net.to(ACC_DEVICE)
                out_t = forward_outputs_chunked(net, batch_inputs, ACC_BATCH, ACC_DEVICE)   # (B,T,out), once
                if keep_outputs:
                    batch_outputs[rule] = out_t.cpu().numpy()
                # Match training's cost mask and mean over all B*T*n_output elements.
                # This is the prediction loss, without weight regularization.
                mse_losses[rule] = float(masked_mse_loss_only(out_t, lab_d, msk_d))
                inp_d = batch_inputs.to(ACC_DEVICE)
                for mode in ACC_MODES:
                    try:
                        acc, _ = net.compute_acc(out_t.float(), lab_d.float(), msk_d.float(),
                                                 inp_d.float(), mode=mode, isvalid=True)
                        accs[mode][rule] = 100.0 * float(acc)
                    except Exception as e:
                        accs[mode][rule] = float("nan")
                        print(f"  {rule} [{mode}]: accuracy failed ({e})")
            finally:
                net.to(orig_dev)  # Restore the device for the modulation rollout.
                if ACC_DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

        return accs, mse_losses, batch_outputs

    TIMING_MODES = ("random", "random_batch")
    results = {}
    for timing_mode in TIMING_MODES:
        batch = ((inputs, labels, mask) if timing_mode == "random_batch"
                 else generate_batch(timing_mode))
        accs, mse_losses, batch_outputs = score_batch(
            *batch, keep_outputs=(timing_mode == "random_batch"))
        results[timing_mode] = (accs, mse_losses)
        if timing_mode == "random_batch":
            outputs = batch_outputs
        for mode in ACC_MODES:
            print(f"\nHeld-out {mode} accuracy on {batch[0].shape[0]} trials "
                  f"(mode_input={timing_mode}, task={RULESET}, seed={SEED}):")
            for rule in RULES:
                print(f"  {RULE_LABEL.get(rule, rule):16} {accs[mode][rule]:6.1f}%")
        print(f"\nHeld-out masked MSE loss on {batch[0].shape[0]} trials "
              f"(mode_input={timing_mode}, task={RULESET}, seed={SEED}):")
        for rule in RULES:
            print(f"  {RULE_LABEL.get(rule, rule):16} {mse_losses[rule]:.6g}")

    # Rows are timing modes. Each metric shares its y limits across both rows.
    xs = np.arange(len(RULES))
    fig, axes = plt.subplots(len(TIMING_MODES), len(ACC_MODES) + 1,
                             figsize=((1.4 * len(RULES) + 1.5) * (len(ACC_MODES) + 1), 3.2 * len(TIMING_MODES)),
                             squeeze=False)
    finite_losses = [value for _, losses in results.values()
                     for value in losses.values() if np.isfinite(value)]
    maximum = max(finite_losses, default=0.0)
    for row, timing_mode in enumerate(TIMING_MODES):
        accs, mse_losses = results[timing_mode]
        for a, mode in enumerate(ACC_MODES):
            ax = axes[row][a]
            ax.bar(xs, [accs[mode][r] for r in RULES], color=[RULE_COLOR.get(r, "#888") for r in RULES])
            for x, r in zip(xs, RULES):
                v = accs[mode][r]
                if v == v:                                     # skip nan labels
                    ax.text(x, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(xs); ax.set_xticklabels([RULE_LABEL.get(r, r) for r in RULES], rotation=15)
            ax.set_title(f"{timing_mode}: {mode} accuracy", fontsize=10)
            ax.set_ylim(0, 110); ax.grid(alpha=0.3, axis="y")
        axes[row][0].set_ylabel("accuracy (%)")
        ax = axes[row][-1]
        values = [mse_losses[r] for r in RULES]
        ax.bar(xs, values, color=[RULE_COLOR.get(r, "#888") for r in RULES])
        for x, value in zip(xs, values):
            if np.isfinite(value):
                ax.annotate(f"{value:.3g}", (x, value), xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([RULE_LABEL.get(r, r) for r in RULES], rotation=15)
        ax.set_title(f"{timing_mode}: masked MSE loss", fontsize=10)
        ax.set_ylabel("MSE loss")
        ax.set_ylim(0, 1.2 * maximum if maximum > 0 else 1.0)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{RULESET}: held-out accuracy and MSE loss per rule (seed {SEED})",
                 y=1.03, fontsize=12)
    fig.tight_layout(); save_fig(fig, "accuracy_angle_stimulus")


    # Example-trial figure — input, target, and each rule's output
    #
    # One column per example trial. Top row: input channels. Following rows: one per
    # learning rule, showing the **target** (faded thick line) vs. that rule's
    # **network output** (solid) for every output channel. Output tracking the target
    # = the network solves the trial.


    N_SHOW = min(10, N_TRIALS)              # example trials (columns) to display
    n_out = labels_np.shape[-1]
    n_in = inputs_np.shape[-1]

    nrows = 1 + len(RULES)                 # input row + one row per rule
    fig, axes = plt.subplots(nrows, N_SHOW, figsize=(3.2 * N_SHOW, 2.3 * nrows),
                             squeeze=False, sharex=True)
    in_cmap = plt.cm.viridis(np.linspace(0, 1, n_in))
    out_cmap = plt.cm.tab10(np.arange(n_out))

    for c in range(N_SHOW):
        # Top: input channels for trial c.
        ax = axes[0][c]
        for ch in range(n_in):
            ax.plot(inputs_np[c, :, ch], color=in_cmap[ch], lw=1.2,
                    label=f"in{ch}" if c == 0 else None)
        ax.set_title(f"trial {c}", fontsize=10)
        if c == 0:
            ax.set_ylabel("input", fontsize=10)
            ax.legend(fontsize=6, ncol=2, frameon=False, loc="upper right")

        # One row per rule: target (faded) vs network output (solid).
        for r, rule in enumerate(RULES):
            ax = axes[1 + r][c]
            for o in range(n_out):
                ax.plot(labels_np[c, :, o], color=out_cmap[o], lw=4, alpha=0.35)
                ax.plot(outputs[rule][c, :, o], color=out_cmap[o], lw=1.3,
                        label=f"out{o}" if c == 0 else None)
            if c == 0:
                ax.set_ylabel(f"{RULE_LABEL.get(rule, rule)}\noutput", fontsize=9)
                if r == 0:
                    ax.legend(fontsize=6, ncol=n_out, frameon=False, loc="upper right")
    axes[-1][0].set_xlabel("time step", fontsize=10)
    fig.suptitle(f"{RULESET}: target (faded) vs network output — seed {SEED}",
                 y=1.005, fontsize=12)
    fig.tight_layout()
    save_fig(fig, "example_trials")


    # Modulation trajectories: one representative trial per occupied target-angle
    # bin, shared across rules. These can differ from the first N_SHOW trials above.
    # Record active fractions for every layer; keep full M histories only for
    # the first, middle, and last layers to bound memory use.
    # For trajectories, use the later middle layer at even depths, or all layers
    # for depths <= 3. Active-fraction plots always show every layer.
    # Layer labels retain their original indices.
    n_layers = len(nets[RULES[0]].mp_layers)
    shown_layers = (list(range(n_layers)) if n_layers <= 3
                    else [0, n_layers // 2, n_layers - 1])
    if len(shown_layers) < n_layers:
        print(f"{n_layers} MP layers present; full M trajectories use the first, middle, "
              f"and last (indices {shown_layers}); active fractions use every layer.")


    ACTIVE_THRESHOLDS = (0.3, 0.6, 0.9)

    @torch.no_grad()
    def rollout_M(net, inputs, keep):
        """Return selected M histories and active percentages for every layer.

        Returns (histories, active): histories contains one float32 array of
        shape (B, T, post, pre) per layer in keep; active maps each threshold
        to percentages of shape (n_layers, B, T). Both use M after each update.
        """
        B, T, _ = inputs.shape
        net.reset_state(B=B)
        inputs = net._standardize_input(inputs)
        buf = [np.zeros((B, T) + tuple(net.mp_layers[k].M.shape[1:]), dtype=np.float32)
               for k in keep]
        active = {threshold: np.zeros((len(net.mp_layers), B, T), dtype=np.float32)
                  for threshold in ACTIVE_THRESHOLDS}
        for t in range(T):
            net.network_step(inputs[:, t, :], seq_idx=t)
            for k, layer in enumerate(net.mp_layers):
                magnitude = layer.M.abs()
                for threshold in ACTIVE_THRESHOLDS:
                    active[threshold][k, :, t] = (
                        100.0 * (magnitude > threshold).float().mean(dim=(1, 2))
                    ).cpu().numpy()
            for j, k in enumerate(keep):
                buf[j][:, t] = net.mp_layers[k].M.cpu().numpy()   # M_t for TRUE layer k
        return buf, active


    # Bin trials by the target angle atan2(sin, cos), using positive cost-mask
    # timesteps, then select one trial per occupied preferred-direction bin.
    # In the trajectory figure, rows are rule/layer pairs; faint lines show
    # sampled synapses and black lines show the mean over all synapses.
    MAX_SYNAPSE = 400                 # cap lines drawn per panel (subsample if larger)

    # --- Preferred ring directions (the discrete stimuli) ---
    prefs = np.asarray(any_net.prefs.detach().cpu().numpy() if hasattr(any_net, "prefs")
                       else task_params["hp"]["pref"])

    # Per-trial target direction (ch1=sin, ch2=cos).
    def _stim_bin(b):
        m = mask_np[b, :, 0] > 0  # Includes scored pre-response timesteps too.
        if m.sum() == 0:
            return -1
        sin = labels_np[b, m, 1].mean(); cos = labels_np[b, m, 2].mean()
        ang = np.mod(np.arctan2(sin, cos), 2 * np.pi)
        d = np.abs((prefs - ang + np.pi) % (2 * np.pi) - np.pi)   # periodic distance
        return int(np.argmin(d))

    mask_np = mask.cpu().numpy()
    stim_bins = np.array([_stim_bin(b) for b in range(inputs_np.shape[0])])
    _rng = np.random.default_rng(SEED)
    occupied = sorted(int(u) for u in np.unique(stim_bins) if u >= 0)
    # one random trial index per occupied stimulus bin
    picks = [(u, int(_rng.choice(np.where(stim_bins == u)[0]))) for u in occupied]
    print(f"{len(prefs)} ring stimuli; {len(picks)} occupied → one trial each: "
          + ", ".join(f"stim{u}(θ={prefs[u]:.2f})→trial{b}" for u, b in picks))

    if not picks:
        raise ValueError("No scored ring-stimulus trials are available for the modulation figure.")
    selected_inputs = inputs[[trial for _, trial in picks]]
    rollouts = {rule: rollout_M(net, selected_inputs, shown_layers) for rule, net in nets.items()}
    M_by_rule = {rule: result[0] for rule, result in rollouts.items()}
    active_by_rule = {rule: result[1] for rule, result in rollouts.items()}
    for rule in RULES:
        shapes = [matrix.shape for matrix in M_by_rule[rule]]
        print(f"{RULE_LABEL.get(rule, rule):16} M per shown layer: "
              + ", ".join(f"L{layer}:{shape}" for layer, shape in zip(shown_layers, shapes)))

    ncols = len(picks)
    # Rows iterate over the selected layers only (shown_layers, at most three).
    # `j` indexes M_by_rule[rule][j]; `k` is the corresponding TRUE MP-layer index.
    rows = [(rule, j, k) for rule in RULES for j, k in enumerate(shown_layers)]
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 1.9 * nrows),
                             squeeze=False, sharex=True)
    rng = np.random.default_rng(SEED)
    for r, (rule, j, k) in enumerate(rows):
        M_seq = M_by_rule[rule][j]                       # (B, T, post, pre); j → buffered slot
        B, T, post, pre = M_seq.shape
        n_syn = post * pre
        flat = M_seq.reshape(B, T, n_syn)                # flatten synapses
        sel = (rng.choice(n_syn, MAX_SYNAPSE, replace=False) if n_syn > MAX_SYNAPSE
               else np.arange(n_syn))
        c = RULE_COLOR.get(rule, "#888")
        for col, (u, _) in enumerate(picks):
            ax = axes[r][col]
            traj = flat[col]
            ax.plot(traj[:, sel], color=c, lw=0.4, alpha=0.15)
            ax.plot(traj.mean(axis=1), color="k", lw=1.4)   # across-synapse mean
            ax.axhline(0.0, color="0.7", lw=0.6, ls=":")
            if r == 0:
                ax.set_title(f"stim {u}\nθ={prefs[u]:.2f}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{RULE_LABEL.get(rule, rule)}\nMP L{k}  (M, {n_syn} syn)",
                              fontsize=8)   # k = true layer index
    axes[-1][0].set_xlabel("time step", fontsize=10)
    fig.suptitle(f"{RULESET}: per-synapse modulation M over time — one trial per stimulus "
                 f"(seed {SEED}; faint = sampled synapses, black = across-synapse mean)",
                 y=1.005, fontsize=12)
    fig.tight_layout()
    save_fig(fig, "modulation_per_synapse")

    # Count all synapses, including negative modulation with large magnitude.
    # Use every layer and the same representative trials as the M trajectories.
    for threshold in ACTIVE_THRESHOLDS:
        fig, axes = plt.subplots(n_layers, ncols,
                                 figsize=(3.0 * ncols, 2.5 * n_layers),
                                 squeeze=False, sharex=True, sharey=True)
        print(f"\nActive synapses (|M| > {threshold:g}); representative trials only:")
        print("  rule             layer stimulus trial   time mean (%)   peak (%)   final (%)")
        for k in range(n_layers):
            for rule in RULES:
                # (trial, time): fraction over every post/pre pair, without subsampling.
                active_percent = active_by_rule[rule][threshold][k]
                for col, (u, trial) in enumerate(picks):
                    curve = active_percent[col]
                    axes[k][col].plot(curve, color=RULE_COLOR.get(rule, "#888"),
                                      label=RULE_LABEL.get(rule, rule), lw=1.5)
                    print(f"  {RULE_LABEL.get(rule, rule):16} {k:5d} {u:8d} {trial:5d} "
                          f"{curve.mean():15.2f} {curve.max():10.2f} {curve[-1]:11.2f}")
            for col, (u, _) in enumerate(picks):
                ax = axes[k][col]
                ax.set_ylim(0, 100)
                ax.grid(alpha=0.3)
                if k == 0:
                    ax.set_title(f"stim {u}\nθ={prefs[u]:.2f}", fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"MP L{k}\nactive synapses (%)", fontsize=9)
                if k == n_layers - 1:
                    ax.set_xlabel("time step", fontsize=10)
        axes[0][0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{RULESET}: active synapses (|M| > {threshold:g}) — "
                     f"one trial per stimulus (seed {SEED})",
                     y=1.005, fontsize=12)
        fig.tight_layout()
        save_fig(fig, f"modulation_active_fraction_threshold{threshold:g}")


def main():
    """Run parameter, performance, or both analyses from one checkpoint group."""
    parser = argparse.ArgumentParser(
        description="Save trained MPN parameter and performance figures from one checkpoint group.")
    parser.add_argument("--ckpt-dir", type=Path, default=_bootstrap.ROOT / "checkpoints")
    parser.add_argument("--ckpt-stem", help="checkpoint prefix; default: newest complete group")
    parser.add_argument("--seed", type=int, help="saved seed; default: newest complete group")
    parser.add_argument("--rules", nargs="+", choices=list(RULE_LABEL), default=list(DEFAULT_RULES))
    parser.add_argument("--analysis", choices=("all", "weights", "performance"), default="all")
    parser.add_argument("--trials", type=int, default=2000,
                        help="trials per task per timing mode (random and random_batch; default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=_bootstrap.ROOT / "notebooks" / "visualize_trained_networks")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    args.rules = list(dict.fromkeys(args.rules))
    try:
        context = load_context(args)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    plt.rcParams["figure.dpi"] = 110
    if args.analysis in ("all", "weights"):
        plot_weights(context)
    if args.analysis in ("all", "performance"):
        plot_performance(context)


if __name__ == "__main__":
    main()
