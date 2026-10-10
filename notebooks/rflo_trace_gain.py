#!/usr/bin/env python
"""Trace-gain diagnostic: exact row-local gain vs diagonal-RFLO gain, and when the cap matters.

For one batch of one task, a deep MPN is run through three local passes on the
SAME forward trajectory and compared with BPTT:
  exact row-local  (local_exact_rowlocal)  the full within-row eligibility P
  diagonal RFLO    (local_diag_rflo, no cap) the same-synapse trace A
  diagonal RFLO    (local_diag_rflo, cap rho) A with the recurrence gain clipped

Per step and per layer it records two loop gains of the modulation-trace
recurrence (k = assoc * eta * phi' * x^2):
  diagonal gain  g_diag[b,i,I] = lam + k_I * W_iI          (what RFLO's A obeys)
  exact row gain g_row[b,i]    = lam + sum_J k_J * W_iJ    (what the exact P obeys)
The diagonal approximation keeps one signed term of the row sum, so its gain can
exceed 1 for a synapse even when the row's net feedback is damped; A then grows
geometrically within a trial while the exact trace stays bounded. The figure
shows the gain distributions, the fraction above 1 over time, the growth of the
largest trace entry (exact diag(P), uncapped A, capped A) and each variant's
gradient cosine with BPTT. Outputs: <output-dir>/<tag>.png, <tag>_summary.json,
<tag>_curves.npz (default output dir notebooks/rflo_trace_gain/).

Whether the cap is needed depends on the loop gain lam + k*W: at the defaults
(eta=1, lam=0.99) the uncapped A diverges; with a small eta (e.g. 0.003 at lam=0.99),
with or without --mp-input-norm rms, all three variants stay bounded and aligned
(README, "MP-input RMS normalization"). --eta / --lam / --mp-input-norm /
--mp-input-norm-skip-first take the same values as train_mpn.py.

Run from the project root, e.g.
    python notebooks/rflo_trace_gain.py --task contextdelaydm1 --hidden 128 128 128
    python notebooks/rflo_trace_gain.py --task adding_L200 --batch 16 --rho 0.99
    python notebooks/rflo_trace_gain.py --task contextdelaydm1 --hidden 64 64 64 --batch 8 \
        --eta 0.003 --lam 0.99                      # no-norm control
    python notebooks/rflo_trace_gain.py --task contextdelaydm1 --hidden 64 64 64 --batch 8 \
        --mp-input-norm rms --mp-input-norm-skip-first --eta 0.003 --lam 0.99
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap
import mpn
import tasks
import train_common as tc
import train_mpn

# Validated reference palette (dataviz skill, categorical slots 1-3; all-pairs safe).
COLORS = {"exact": "#2a78d6", "diag": "#eb6834", "capped": "#1baf7a"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def build(task_name, widths, rule, rho, seed, batch, residual=False, device="cpu",
          dtype=torch.float32, data_seed=1, overrides=None):
    """A deep MPN from train_mpn's parameter builder (so every default matches the
    training script), plus one training batch. Returns (net, (x, y, mask)).
    overrides: extra train_mpn module globals to patch for this build, e.g.
    {'MP_INPUT_NORM': 'rms', 'ETA': [0.01], 'LAM': 0.99} (same semantics as the
    corresponding --mp-input-norm / --eta / --lam flags of train_mpn.py)."""
    with patch.multiple(train_mpn, N_HIDDEN=list(widths), RULESET=task_name,
                        MP_RESIDUAL=residual, RFLO_TRACE_RHO=rho, **(overrides or {})):
        # Seed BEFORE the task is initialized: the ring tasks draw their trial
        # generator's seed from the global numpy stream at init time.
        np.random.seed(seed)
        torch.manual_seed(seed)
        task = tasks.make_task(task_name)
        tp, trp, npar = task.init_params(*train_mpn.build_params())
        npar["learning_rule"] = rule
        with contextlib.redirect_stdout(io.StringIO()):
            net = mpn.DeepMultiPlasticNet(npar, verbose=False).to(device).to(dtype)
        np.random.seed(data_seed)
        x, y, mask = task.train_batch(tp, trp, batch, torch.device(device), dtype)
    return net, (x, y, mask)


def _lam_scalar(lam):
    """lam arrives as the (i, I) expansion; the train configs use a scalar lambda."""
    return lam[:, :1]


def record_diag(net, x, y, mask):
    """Run the diagonal RFLO pass once, recording per layer and step the diagonal
    gain, the exact row gain, their fractions above 1, and max|A|."""
    rec = [dict(frac_diag=[], frac_row=[], maxA=[], diag_gain=[], row_gain=[])
           for _ in net.mp_layers]
    originals = []
    for n, layer in enumerate(net.mp_layers):
        orig_step, orig_sel = layer._local_step_diag, layer.step_fn_for

        def spy(xx, phi_prime, ell, eta, lam, update_mask=None, _o=orig_step, _l=layer, _n=n):
            k = _l._assoc * eta.unsqueeze(0) * phi_prime.unsqueeze(-1) * xx.square().unsqueeze(1)
            w_read = _l._modulation_read_weight() if hasattr(_l, "_modulation_read_weight") else _l.W
            kw = k * w_read.unsqueeze(0)                          # (B, i, I)
            g_diag = lam.unsqueeze(0) + kw
            g_row = _lam_scalar(lam).unsqueeze(0).squeeze(-1) + kw.sum(-1)   # (B, i)
            out = _o(xx, phi_prime, ell, eta, lam, update_mask)
            r = rec[_n]
            r["frac_diag"].append((g_diag.abs() > 1).float().mean().item())
            r["frac_row"].append((g_row.abs() > 1).float().mean().item())
            r["maxA"].append(_l.A.abs().max().item())
            r["diag_gain"].append(g_diag.detach().flatten().cpu())
            r["row_gain"].append(g_row.detach().flatten().cpu())
            return out
        layer.step_fn_for = (lambda mode, _s=spy, _o=orig_sel: _s if mode == "diag" else _o(mode))
        originals.append((layer, orig_sel))
    try:
        grads = net.local_diag_rflo_gradients(x, y, mask)
    finally:
        for layer, orig_sel in originals:
            layer.step_fn_for = orig_sel
    for r in rec:
        r["diag_gain"] = torch.cat(r["diag_gain"]).numpy()
        r["row_gain"] = torch.cat(r["row_gain"]).numpy()
        for k in ("frac_diag", "frac_row", "maxA"):
            r[k] = np.asarray(r[k])
    return grads, rec


def record_exact(net, x, y, mask):
    """Run the exact row-local pass once, recording per layer and step max|P| and
    max|diag(P)| (the entries the diagonal trace A approximates)."""
    rec = [dict(maxP=[], maxPdiag=[]) for _ in net.mp_layers]
    originals = []
    for n, layer in enumerate(net.mp_layers):
        orig_step, orig_sel = layer._local_step_exact, layer.step_fn_for

        def spy(xx, phi_prime, ell, eta, lam, update_mask=None, _o=orig_step, _l=layer, _n=n):
            out = _o(xx, phi_prime, ell, eta, lam, update_mask)
            P = _l.P                                             # (B, i, I, J)
            rec[_n]["maxP"].append(P.abs().max().item())
            rec[_n]["maxPdiag"].append(torch.diagonal(P, dim1=-2, dim2=-1).abs().max().item())
            return out
        layer.step_fn_for = (lambda mode, _s=spy, _o=orig_sel: _s if mode == "exact" else _o(mode))
        originals.append((layer, orig_sel))
    try:
        grads = net.local_gradients(x, y, mask)
    finally:
        for layer, orig_sel in originals:
            layer.step_fn_for = orig_sel
    for r in rec:
        for k in ("maxP", "maxPdiag"):
            r[k] = np.asarray(r[k])
    return grads, rec


def analyze(task_name, widths=(128, 128, 128), batch=16, seed=0, rho=0.99, residual=False,
            device="cpu", dtype=torch.float32, overrides=None):
    """All measurements for one task/batch: three local variants on identical
    init and data, plus the BPTT reference. Returns a JSON-safe summary and the
    per-step curves. overrides: see build()."""
    L = len(widths)
    net_exact, (x, y, mask) = build(task_name, widths, "local_exact_rowlocal", None, seed, batch, residual, device, dtype, overrides=overrides)
    net_diag, _ = build(task_name, widths, "local_diag_rflo", None, seed, batch, residual, device, dtype, overrides=overrides)
    net_cap, _ = build(task_name, widths, "local_diag_rflo", rho, seed, batch, residual, device, dtype, overrides=overrides)
    for other in (net_diag, net_cap):          # same init by construction (same seed); assert it
        for k, p in net_exact._trainable_params().items():
            assert torch.equal(other._trainable_params()[k], p), k
    g_exact, rec_exact = record_exact(net_exact, x, y, mask)
    g_diag, rec_diag = record_diag(net_diag, x, y, mask)
    g_cap, rec_cap = record_diag(net_cap, x, y, mask)
    ref = net_exact.bptt_gradients(x, y, mask)
    keys = ["W" if n == 0 else f"W{n}" for n in range(L)]
    cos = {name: tc.cosine_alignment(g, ref, keys)
           for name, g in (("exact", g_exact), ("diag", g_diag), ("capped", g_cap))}
    norms = {name: {k: float(g[k].norm()) for k in keys}
             for name, g in (("exact", g_exact), ("diag", g_diag), ("capped", g_cap), ("bptt", ref))}
    T = x.shape[1]
    layers = []
    for n in range(L):
        dg, rg = rec_diag[n]["diag_gain"], rec_diag[n]["row_gain"]
        layers.append(dict(
            key=keys[n], width=int(widths[n]),
            frac_diag_gain_gt1=float((np.abs(dg) > 1).mean()),
            frac_row_gain_gt1=float((np.abs(rg) > 1).mean()),
            diag_gain_quantiles={q: float(np.quantile(dg, q)) for q in (0.01, 0.5, 0.99, 0.999)},
            row_gain_quantiles={q: float(np.quantile(rg, q)) for q in (0.01, 0.5, 0.99, 0.999)},
            maxA_uncapped=dict(quarter=float(rec_diag[n]["maxA"][T // 4]), half=float(rec_diag[n]["maxA"][T // 2]), end=float(rec_diag[n]["maxA"][-1])),
            maxA_capped=dict(quarter=float(rec_cap[n]["maxA"][T // 4]), half=float(rec_cap[n]["maxA"][T // 2]), end=float(rec_cap[n]["maxA"][-1])),
            maxPdiag_exact=dict(quarter=float(rec_exact[n]["maxPdiag"][T // 4]), half=float(rec_exact[n]["maxPdiag"][T // 2]), end=float(rec_exact[n]["maxPdiag"][-1])),
            cos_vs_bptt={name: cos[name][keys[n]] for name in cos},
            grad_norm={name: norms[name][keys[n]] for name in norms},
        ))
    summary = dict(task=task_name, T=int(T), batch=int(batch), widths=[int(w) for w in widths], seed=seed,
                   rho=rho, residual=residual, dtype=str(dtype),
                   mp_input_norm=net_exact.mp_input_norm,
                   mp_input_norm_skip_first=bool(getattr(net_exact, "mp_input_norm_skip_first", False)),
                   eta=float(net_exact.mp_layers[0].eta.mean()), lam=float(net_exact.mp_layers[0].lam.mean()),
                   layers=layers)
    curves = dict(T=T, keys=keys,
                  frac_diag=np.stack([r["frac_diag"] for r in rec_diag]),
                  frac_row=np.stack([r["frac_row"] for r in rec_diag]),
                  maxA_uncapped=np.stack([r["maxA"] for r in rec_diag]),
                  maxA_capped=np.stack([r["maxA"] for r in rec_cap]),
                  maxPdiag_exact=np.stack([r["maxPdiag"] for r in rec_exact]),
                  maxP_exact=np.stack([r["maxP"] for r in rec_exact]),
                  diag_gain=[r["diag_gain"] for r in rec_diag],
                  row_gain=[r["row_gain"] for r in rec_diag])
    return summary, curves


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def plot(summary, curves, path):
    """One row per MP layer, four panels: gain ECDFs, fraction of gains above 1
    over time, largest trace entry over time (log), gradient cosine vs BPTT."""
    L = len(summary["layers"])
    T = curves["T"]
    steps = np.arange(T)
    fig, axes = plt.subplots(L, 4, figsize=(17, 3.4 * L), squeeze=False)
    for n, lay in enumerate(summary["layers"]):
        ax = axes[n][0]
        for name, arr, color in (("exact row gain  λ + Σ_J k_J W_iJ", curves["row_gain"][n], COLORS["exact"]),
                                 ("diagonal gain  λ + k_I W_iI", curves["diag_gain"][n], COLORS["diag"])):
            v = np.sort(np.abs(arr))
            ax.plot(v, np.linspace(0, 1, len(v)), color=color, lw=2, label=name)
        ax.axvline(1.0, color=MUTED, lw=1, ls=":")
        lo = min(np.quantile(np.abs(curves["row_gain"][n]), 0.001), np.quantile(np.abs(curves["diag_gain"][n]), 0.001))
        hi = max(np.quantile(np.abs(curves["row_gain"][n]), 0.999), np.quantile(np.abs(curves["diag_gain"][n]), 0.999))
        ax.set_xlim(min(lo, 0.95), max(hi, 1.05))
        ax.set_ylabel(f"layer {n}  ({lay['key']}, width {lay['width']})\nECDF", color=TEXT, fontsize=9)
        ax.set_title("recurrence gain |g| (all steps, batch, synapses)" if n == 0 else "", color=TEXT, fontsize=9)
        # ECDFs rise from bottom-left to top-right: the upper-left corner is free for
        # the legend and the lower-right corner for the fraction-above-1 readout.
        ax.text(0.98, 0.05, f"> 1:  row {100 * lay['frac_row_gain_gt1']:.1f}%   diag {100 * lay['frac_diag_gain_gt1']:.1f}%",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=TEXT)
        _style(ax)
        if n == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")

        ax = axes[n][1]
        ax.plot(steps, 100 * curves["frac_row"][n], color=COLORS["exact"], lw=2, label="exact row gain")
        ax.plot(steps, 100 * curves["frac_diag"][n], color=COLORS["diag"], lw=2, label="diagonal gain")
        ax.set_ylim(0, max(1.0, 1.15 * 100 * max(curves["frac_row"][n].max(), curves["frac_diag"][n].max())))
        ax.set_title("% of gains above 1, per step" if n == 0 else "", color=TEXT, fontsize=9)
        if n == L - 1:
            ax.set_xlabel("time step", color=MUTED, fontsize=9)
        _style(ax)
        if n == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")

        ax = axes[n][2]
        ax.plot(steps, curves["maxPdiag_exact"][n], color=COLORS["exact"], lw=2, label="exact row-local  max|diag P|")
        ax.plot(steps, curves["maxA_uncapped"][n], color=COLORS["diag"], lw=2, label="diagonal RFLO  max|A|, no cap")
        ax.plot(steps, curves["maxA_capped"][n], color=COLORS["capped"], lw=2, label=f"diagonal RFLO  max|A|, cap ρ={summary['rho']}")
        ax.set_yscale("log")
        ax.set_title("largest modulation-trace entry (log)" if n == 0 else "", color=TEXT, fontsize=9)
        if n == L - 1:
            ax.set_xlabel("time step", color=MUTED, fontsize=9)
        _style(ax)
        if n == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")

        ax = axes[n][3]
        names = [("exact row-local", "exact", COLORS["exact"]), ("diag, no cap", "diag", COLORS["diag"]),
                 (f"diag, cap {summary['rho']}", "capped", COLORS["capped"])]
        vals = [lay["cos_vs_bptt"][k] for _, k, _ in names]
        xs = np.arange(len(names))
        ax.bar(xs, vals, width=0.5, color=[c for *_, c in names], edgecolor="white", linewidth=2)
        for xi, v, (_, k, _) in zip(xs, vals, names):
            ratio = lay["grad_norm"][k] / lay["grad_norm"]["bptt"] if lay["grad_norm"]["bptt"] else float("nan")
            ax.text(xi, v + 0.02, f"cos {v:.2f}\n|g|/|g_BPTT| {ratio:.2g}", ha="center", va="bottom", fontsize=7.5, color=TEXT)
        ax.set_xticks(xs)
        ax.set_xticklabels([t for t, *_ in names], fontsize=8, color=MUTED)
        ax.set_ylim(min(0, min(vals) - 0.1), 1.25)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_title("gradient cosine with BPTT (this batch)" if n == 0 else "", color=TEXT, fontsize=9)
        _style(ax)
        ax.grid(False, axis="x")
    fig.suptitle(f"{summary['task']}: diagonal-RFLO trace gain vs the exact row-local trace  "
                 f"(T={summary['T']}, batch={summary['batch']}, η={summary['eta']:g}, λ={summary['lam']:.3g}, "
                 f"widths={summary['widths']}, residual={'on' if summary['residual'] else 'off'}"
                 f"{', MP-input ' + summary['mp_input_norm'] + ' norm' + (' (layer 0 un-normalized)' if summary.get('mp_input_norm_skip_first') else '') if summary.get('mp_input_norm', 'none') != 'none' else ''})",
                 color=TEXT, fontsize=11, y=1.0)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def run(task_name, widths, batch, seed, rho, residual, output_dir=None, device="cpu", dtype=torch.float32,
        overrides=None):
    summary, curves = analyze(task_name, widths, batch, seed, rho, residual, device, dtype, overrides=overrides)
    tag = f"{task_name}_h{'-'.join(str(w) for w in widths)}_b{batch}_rho{rho:g}_seed{seed}"
    if summary.get("mp_input_norm", "none") != "none":
        tag += f"_mpnorm-{summary['mp_input_norm']}"
        if summary.get("mp_input_norm_skip_first"):
            tag += "-skip0"
    if overrides and overrides.get("ETA") is not None:
        tag += "_eta" + "-".join(f"{e:g}" for e in overrides["ETA"])
    if overrides and overrides.get("LAM") is not None:
        tag += f"_lam{overrides['LAM']:g}"
    out = Path(output_dir) if output_dir else _bootstrap.ROOT / "notebooks" / "rflo_trace_gain"
    out.mkdir(parents=True, exist_ok=True)
    fig_path = plot(summary, curves, out / f"{tag}.png")
    with open(out / f"{tag}_summary.json", "w") as fh:
        json.dump(tc._json_safe(summary), fh, indent=2)
    np.savez(out / f"{tag}_curves.npz", **{k: v for k, v in curves.items() if k not in ("diag_gain", "row_gain", "keys")},
             keys=np.asarray(curves["keys"]))
    print(f"Saved figure: {fig_path}")
    for lay in summary["layers"]:
        print(f"  {lay['key']:3s} gain>1: row {100 * lay['frac_row_gain_gt1']:.1f}% diag {100 * lay['frac_diag_gain_gt1']:.1f}% | "
              f"max|A| end: no cap {lay['maxA_uncapped']['end']:.2e} cap {lay['maxA_capped']['end']:.2e} | "
              f"exact max|diag P| end {lay['maxPdiag_exact']['end']:.2e} | cos vs BPTT: "
              + " ".join(f"{k}={v:.2f}" for k, v in lay['cos_vs_bptt'].items()))
    return summary, fig_path


def _parse(argv=None):
    p = argparse.ArgumentParser(description="Exact row gain vs diagonal RFLO gain, trace growth and BPTT alignment on one batch.")
    p.add_argument("--task", default="contextdelaydm1")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 128])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rho", type=float, default=0.99, help="trace-gain cap for the capped variant")
    p.add_argument("--residual", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mp-input-norm", choices=["none", "rms"], default="none",
                   help="per-step RMS norm of each MP layer's input (train_mpn --mp-input-norm)")
    p.add_argument("--mp-input-norm-skip-first", action="store_true", default=False,
                   help="leave MP layer 0's input un-normalized (train_mpn --mp-input-norm-skip-first)")
    p.add_argument("--eta", type=float, nargs="+", default=None,
                   help="fixed Hebbian write rate(s), as train_mpn --eta (default: core 1.0)")
    p.add_argument("--lam", type=float, default=None,
                   help="fixed modulation decay, as train_mpn --lam (default: m_time_scale setup)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = _parse(argv)
    overrides = {"MP_INPUT_NORM": a.mp_input_norm,
                 "MP_INPUT_NORM_SKIP_FIRST": bool(a.mp_input_norm_skip_first)}
    if a.eta is not None:
        overrides["ETA"] = list(a.eta)
    if a.lam is not None:
        overrides["LAM"] = a.lam
    return run(a.task, a.hidden, a.batch, a.seed, a.rho, a.residual, a.output_dir, a.device,
               torch.float64 if a.dtype == "float64" else torch.float32, overrides=overrides)


if __name__ == "__main__":
    main()
