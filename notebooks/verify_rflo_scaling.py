#!/usr/bin/env python
"""Test the RFLO-as-rescaling hypothesis at frozen single-MP checkpoints.

CUDA required. All complete direct/RFLO seed pairs in one experiment are used.
Eight shared batches of 128 trials (data seeds 0..7); first four fit scalar or
positive diagonal gains, last four test their predictions. MP weights only.
Local bias is direct; input_mode=match; feedback must be exact_spatial. BPTT
therefore supplies the exact MP-weight reference at this depth. No training or
checkpoint writes. Virtual Adam uses zero moments and raw frozen-weight batch
gradients, WITHOUT clipping/decay; it does not reconstruct training updates.
Outputs default to notebooks/verify_rflo_scaling/<run-id>_scaling/.
"""

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap
import diagnose_gradients as dg
from train_common import _json_safe


DATA_SEEDS = tuple(range(8))
FIT_BATCHES = 4


def scaling_fit(direct, rflo):
    """Unconstrained common gain, preserving its sign; zero vectors are undefined."""
    d, r = direct.double().reshape(-1), rflo.double().reshape(-1)
    if not torch.isfinite(d).all() or not torch.isfinite(r).all():
        raise ValueError("Scaling fits require finite gradients")
    dd, rr = float(d @ d), float(r @ r)
    gain = float(d @ r) / dd if dd else None
    residual = float((r - gain*d).norm()) / rr**.5 if dd and rr else None
    correction_energy = float((r-d).square().sum())
    parallel_fraction = ((gain-1)**2*dd/correction_energy
                         if gain is not None and correction_energy else None)
    return dict(gain=gain, residual=residual,
                parallel_correction_energy_fraction=parallel_fraction,
                **dg.compare_vectors(r, d))


def heldout_scaling(direct, rflo, fit_batches=FIT_BATCHES):
    """Fit gains across batches, then evaluate on independent held-out batches."""
    d, r = direct.double().flatten(1), rflo.double().flatten(1)
    if not 0 < fit_batches < len(d):
        raise ValueError("Need nonempty fit and held-out batches")
    train_d, test_d = d[:fit_batches], d[fit_batches:]
    train_r, test_r = r[:fit_batches], r[fit_batches:]
    energy = train_d.square().sum(0)
    supported = energy > 0
    # Unsupported coordinates predict zero: their held-out RFLO energy is NOT dropped.
    diagonal = torch.zeros_like(energy)
    diagonal[supported] = (train_d*train_r).sum(0)[supported] / energy[supported]
    scalar = scaling_fit(train_d, train_r)["gain"]
    if scalar is None:
        scalar = 0.
    positive_diagonal = diagonal.clamp_min(0)
    return dict(scalar_gain=scalar,
                scalar_test_error=dg.compare_vectors(scalar*test_d, test_r)["relative_l2_error"],
                positive_diagonal_test_error=dg.compare_vectors(positive_diagonal*test_d, test_r)["relative_l2_error"],
                signed_diagonal_test_error=dg.compare_vectors(diagonal*test_d, test_r)["relative_l2_error"],
                identity_test_error=dg.compare_vectors(test_d, test_r)["relative_l2_error"],
                negative_diagonal_fraction=float((diagonal[supported] < 0).double().mean()) if supported.any() else None,
                unsupported_diagonal_fraction=float((~supported).double().mean()))


def adam_directions(gradients, eps=1e-8):
    """Bias-corrected Adam directions at fixed weights, with fresh zero moments."""
    m, v = torch.zeros_like(gradients[0]), torch.zeros_like(gradients[0])
    directions = []
    for step, grad in enumerate(gradients, 1):
        m = .9*m + .1*grad
        v = .999*v + .001*grad.square()
        directions.append((m / (1 - .9**step)) / ((v / (1 - .999**step)).sqrt() + eps))
    return torch.stack(directions)


class ScalingRecorder(dg.TraceRecorder):
    """Read pre-update contributions; never divide by 1+M elementwise.

    Moment fits use all sample/time contributions u=ell*E_direct, v=ell*E_RFLO.
    This weights gains by u^2 and includes zero-base RFLO energy in the residual.
    Time curves sum over the batch first, exactly as the weight-gradient routine.
    """

    def __init__(self, net):
        if len(net.mp_layers) != 1:
            raise ValueError("Scaling validation requires exactly one MP layer")
        self.direct_steps, self.rflo_steps = [], []
        self.moments = torch.zeros((3, *net.mp_layers[0].W.shape),
                                   device=net.W_output.device, dtype=torch.double)
        self.zero_base_energy = torch.zeros((), device=net.W_output.device, dtype=torch.double)
        super().__init__(net)

    def _attach(self, layer, index):
        super()._attach(layer, index)
        for name in ("_local_step_diag", "_local_step_direct"):
            original = getattr(layer, name)

            def wrapped(this, x, phi_prime, ell, eta, lam, update_mask=None, original=original):
                base = 1 + this.M
                trace = getattr(this, "A", None)
                wa = torch.zeros_like(base) if trace is None else this.W[None]*trace
                pre = (ell*phi_prime)[..., None]*x[:, None, :]
                u, v = (pre*base).double(), (pre*(base+wa)).double()
                self.direct_steps.append(u.sum(0).cpu())
                self.rflo_steps.append(v.sum(0).cpu())
                self.moments[0] += u.square().sum(0)
                self.moments[1] += (u*v).sum(0)
                self.moments[2] += v.square().sum(0)
                self.zero_base_energy += (v.square()*(base.abs() <= self.near_zero)).sum()
                return original(x, phi_prime, ell, eta, lam, update_mask=update_mask)

            setattr(layer, name, MethodType(wrapped, layer))

    def finish(self, gradients):
        d, r = torch.stack(self.direct_steps), torch.stack(self.rflo_steps)
        # Check independently reconstructed sums against the production gradients.
        checks = {}
        for name, values in (("local_direct", d), ("local_diag_rflo", r)):
            expected = gradients[name]["W"].double()
            error = float((values.sum(0)-expected).norm())
            if error > 2e-5*float(expected.norm()) + 1e-9:
                raise RuntimeError(f"Recorded contributions do not reproduce {name}: {error}")
            checks[name] = dg.ratio(error, float(expected.norm()))
        uu, uv, vv = self.moments.cpu()
        supported = uu > 0
        gain = torch.zeros_like(uu)
        gain[supported] = uv[supported]/uu[supported]
        residual_energy = (vv - 2*gain*uv + gain.square()*uu).sum().clamp_min(0)
        total_energy = float(vv.sum())
        metrics = dict(
            contribution_diagonal_residual=dg.ratio(float(residual_energy.sqrt()), total_energy**.5),
            contribution_diagonal_gradient_error=dg.compare_vectors(gain*d.sum(0), r.sum(0))["relative_l2_error"],
            zero_base_rflo_energy_fraction=dg.ratio(float(self.zero_base_energy), total_energy),
            temporal_cancellation_factor=dg.ratio(float(d.abs().sum(0).norm()), float(d.sum(0).norm())),
            reconstruction_direct_error=checks["local_direct"],
            reconstruction_rflo_error=checks["local_diag_rflo"])
        return metrics, d, r


def select_pairs(run_dir):
    """Accept one experiment or seed folder; reject missing rules or mixed runs."""
    root = Path(run_dir).expanduser().resolve()
    folders = [root] if (root/"local_direct.pt").exists() else sorted(root.glob("seed*"))
    if not folders:
        raise ValueError(f"No seed directories found under {root}")
    pairs = []
    for folder in folders:
        paths = {rule: folder/f"{rule}.pt" for rule in dg.RULES[:2]}
        if any(not path.is_file() for path in paths.values()):
            raise ValueError(f"Incomplete direct/RFLO pair: {folder}")
        pairs.append(paths)
    return pairs


def plot_results(rows, heldout, steps, directory):
    groups = [(row["seed"], row["source_checkpoint"]) for row in heldout]
    labels = [f"{seed}\n{'direct' if rule == 'local_direct' else 'RFLO'}" for seed, rule in groups]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = plt.cm.tab10(np.linspace(0, .9, len(groups)))
    for i, (seed, rule) in enumerate(groups):
        selected = [r for r in rows if (r["seed"], r["source_checkpoint"]) == (seed, rule)]
        offset = np.linspace(-.15, .15, len(selected))
        for ax, key in zip(axes[0], ("cosine", "gain", "residual")):
            ax.scatter(i+offset, [r[key] for r in selected], color=colors[i], s=20)
        for dx, key, marker in ((-.12, "direct_bptt_cosine", "o"), (.12, "rflo_bptt_cosine", "x")):
            axes[1, 0].scatter(i+dx, np.mean([r[key] for r in selected]), marker=marker,
                               color="C0" if dx < 0 else "C1", label=key.replace("_bptt_cosine", "") if i == 0 else None)
        result = heldout[i]
        for dx, key, color in ((-.18, "scalar_test_error", "C0"), (0, "positive_diagonal_test_error", "C1"),
                               (.18, "identity_test_error", "C2")):
            axes[1, 1].bar(i+dx, result[key], width=.18, color=color, label=key.replace("_test_error", "") if i == 0 else None)
        selected_steps = [r for r in steps if (r["seed"], r["source_checkpoint"], r["batch_seed"]) == (seed, rule, 0)]
        axes[1, 2].plot([r["step"] for r in selected_steps], [r["wa_to_base_rms_ratio"] for r in selected_steps],
                       color=colors[i], label=labels[i].replace("\n", " "))
    titles = ["RFLO vs direct: MP-weight cosine", "Best common gain per batch", "Residual after common gain fit",
              "MP-weight cosine with BPTT (batch mean)", "Held-out gradient prediction error", "WA / (1+M) RMS ratio: batch seed 0"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.grid(alpha=.2)
    for ax in list(axes[0]) + list(axes[1, :2]):
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels)
    axes[1, 0].legend()
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].legend(fontsize=7)
    axes[1, 2].set_xlabel("Time step")
    fig.suptitle("Single MP layer: same weights/batches across gradient rules; 8 batches per checkpoint\n"
                 "Raw gradients, direct bias, exact spatial feedback; held-out fits use batches 0–3 and test 4–7")
    fig.tight_layout()
    fig.savefig(directory/"scaling_comparison.png", dpi=160)
    plt.close(fig)


def run(run_dir, output_dir=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for single-layer scaling validation")
    pairs = select_pairs(run_dir)
    first = torch.load(pairs[0]["local_direct"], map_location="cpu", weights_only=False)
    directory = Path(output_dir or _bootstrap.ROOT/"notebooks"/"verify_rflo_scaling"/
                     (first["run_id"] + "_scaling")).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Generate all batches once; every checkpoint consumes these exact tensors.
    batches = [tuple(v.cpu() for v in dg.make_batch(first, seed)) for seed in DATA_SEEDS]
    torch.save(dict(data_seeds=DATA_SEEDS, batches=batches), directory/"batches.pt")
    rows, heldout, all_steps, provenance, saved_gradients = [], [], [], [], {}
    for paths in pairs:
        checkpoints = {rule: torch.load(path, map_location="cpu", weights_only=False) for rule, path in paths.items()}
        dg.validate_pair(checkpoints)
        for rule, checkpoint in checkpoints.items():
            if checkpoint["run_id"] != first["run_id"] or checkpoint["ruleset"] != first["ruleset"]:
                raise ValueError("All seed pairs must belong to the same experiment")
            task = deepcopy(checkpoint["task_params"])
            reference_task = deepcopy(first["task_params"])
            for params in (task, reference_task):
                params["hp"].pop("rng", None)
                params["hp"].pop("seed", None)
            if _json_safe(task) != _json_safe(reference_task):
                raise ValueError("Task configurations differ across checkpoint seeds")
            net, original = dg.build_model(checkpoint, torch.device("cuda"), torch.float32)
            if len(net.mp_layers) != 1 or net.feedback_mode != "exact_spatial":
                raise ValueError("Requires one MP layer with exact_spatial feedback")
            state_hash = dg.file_fingerprint(paths[rule])
            source = dict(seed=checkpoint["seed"], source_checkpoint=rule)
            source_grads = {name: [] for name in dg.RULES}
            forward_checks = []
            for batch_seed, batch_cpu in zip(DATA_SEEDS, batches):
                batch = tuple(v.cuda() for v in batch_cpu)
                observers = []
                def observe(model):
                    observer = ScalingRecorder(model)
                    observers.append(observer)
                    return observer
                result = dg.diagnose(net, batch, recorder_factory=observe)
                metrics, time_d, time_r = observers[0].finish(result["gradients"])
                grads = result["gradients"]
                d, r, b = (grads[name]["W"].double() for name in dg.RULES)
                row = dict(**source, batch_seed=batch_seed, loss=result["losses"]["bptt"],
                           **scaling_fit(d, r), **metrics,
                           direct_bptt_cosine=dg.compare_vectors(d, b)["cosine"],
                           rflo_bptt_cosine=dg.compare_vectors(r, b)["cosine"])
                rows.append(row)
                forward_checks.append(result["forward_checks"])
                for name in dg.RULES:
                    source_grads[name].append(grads[name]["W"].double())
                for trace_row, dt, rt, dc, rc in zip(result["trace_rows"], time_d, time_r, time_d.cumsum(0), time_r.cumsum(0)):
                    all_steps.append(dict(**source, batch_seed=batch_seed, **trace_row,
                                          step_cosine=dg.compare_vectors(rt, dt)["cosine"],
                                          cumulative_cosine=dg.compare_vectors(rc, dc)["cosine"],
                                          cumulative_norm_ratio=dg.compare_vectors(rc, dc)["norm_ratio"]))
                print(f"seed={source['seed']} source={rule} batch={batch_seed} "
                      f"cos={row['cosine']:.5f} gain={row['gain']:.5f} residual={row['residual']:.5f}", flush=True)
                observers.clear()
                del result, batch
                gc.collect()
            source_grads = {k: torch.stack(v) for k, v in source_grads.items()}
            d, r = (source_grads[name] for name in dg.RULES[:2])
            adam_d, adam_r = adam_directions(d), adam_directions(r)
            held = dict(**source, **heldout_scaling(d, r),
                        virtual_adam_cosine=dg.compare_vectors(adam_r, adam_d)["cosine"],
                        virtual_adam_relative_error=dg.compare_vectors(adam_r, adam_d)["relative_l2_error"])
            heldout.append(held)
            saved_gradients[f"{source['seed']}/{rule}"] = source_grads
            if dg.file_fingerprint(paths[rule]) != state_hash:
                raise RuntimeError("Checkpoint file changed during validation")
            provenance.append(dict(**source, path=str(paths[rule]), sha256=state_hash,
                                   original_settings=original, net_params=checkpoint["net_params"],
                                   forward_checks=forward_checks))
            del net
            gc.collect()
            # Keep partial scalar results available if a later checkpoint fails.
            dg.save_csv(directory/"batch_metrics.csv", rows)
            dg.save_csv(directory/"heldout_metrics.csv", heldout)
    dg.save_csv(directory/"trace_steps.csv", all_steps)
    torch.save(saved_gradients, directory/"mp_weight_gradients.pt")
    summary = dict(run_id=first["run_id"], checkpoints=provenance,
                   batches=[dict(data_seed=s, shape=list(b[0].shape), sha256=dg.fingerprint(b)) for s, b in zip(DATA_SEEDS, batches)],
                   settings=dict(batch_size=dg.BATCH_SIZE, fit_batches=FIT_BATCHES, dtype="float32", device="cuda",
                                 input_mode="match", local_bias_mode="direct", cross_layer_steps=0,
                                 optimizer_steps=0, checkpoint_stage="saved final weights"),
                   heldout=heldout,
                   definitions=dict(scope="MP W only, excluding bias/input/readout; raw masked-MSE gradients",
                       parallel_correction_energy_fraction="||(c*-1)g_direct||^2 / ||g_RFLO-g_direct||^2; undefined for zero correction",
                       contribution_diagonal_residual="Best independent gain per synapse fitted over sample/time contributions; zero-base energy retained",
                       temporal_cancellation_factor="norm(sum_t abs(sum_b u_bt)) / norm(sum_bt u_bt); batch cancellation precedes this statistic",
                       heldout="Fit 4 batches; predict next 4; no per-test-batch refitting",
                       virtual_adam="Zero moments, beta=(0.9,0.999), eps=1e-8, fixed checkpoint gradients; no clipping/decay or weight updates",
                       inference="Final-state geometry across batches, not evidence of stable gains throughout training"))
    (directory/"summary.json").write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False)+"\n")
    plot_results(rows, heldout, all_steps, directory)
    print(f"Saved: {directory}", flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="default: notebooks/verify_rflo_scaling/<run-id>_scaling/")
    args = parser.parse_args()
    run(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
