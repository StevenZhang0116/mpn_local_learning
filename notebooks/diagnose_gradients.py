#!/usr/bin/env python
"""Compare direct, diagonal-RFLO and full BPTT at identical checkpoint weights.

Use --run-dir to compare direct-trained and RFLO-trained checkpoints from the
same experiment/seed on one shared batch, or --checkpoint for a single file.
All reports are saved directly in the output directory, without per-rule folders.
The default directory is notebooks/diagnose_gradients/<run-id>/, using the saved
experiment name. Use --output-dir to keep separate analyses of one experiment.
No optimizer steps. Each pass starts from a fresh copy, resets M to the saved
M_init, and consumes the same tensors. Local rules use input_mode=match, direct
MP bias gradients, and no cross-layer correction to isolate MP weight traces.
Feedback is preserved unless --feedback exact_spatial is explicitly requested.
The saved rflo_trace_rho setting is respected for diagonal MP-weight traces.
CUDA is required. New batches contain 128 trials with a fixed data seed of 0.
Only saved ring-task setup is supported for generating data; --batch-file reuses
an exported batch exactly. Reports contain raw masked-MSE gradients before clipping,
weight decay or Adam. A checkpoint describes weights, not a resumable mid-trial M.
"""

import argparse
from copy import deepcopy
import csv
import gc
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap
import mpn
import mpn_tasks
from train_common import _json_safe
from visualize_trained_networks import select_checkpoints


RULES = ("local_direct", "local_diag_rflo", "bptt")
LABELS = {"local_direct": "direct", "local_diag_rflo": "diagonal RFLO", "bptt": "BPTT"}
BATCH_SIZE = 128
BATCH_SEED = 0


def select_sources(checkpoint=None, run_dir=None):
    """Use one file, or the newest complete direct/RFLO group (same seed/directory)."""
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        return {path.stem: path}
    _, _, paths = select_checkpoints(Path(run_dir).expanduser().resolve(), RULES[:2])
    return {rule: path.resolve() for rule, path in paths.items()}


def validate_pair(checkpoints):
    """Reject incompatible metadata/configurations before comparing trained states."""
    direct, rflo = (checkpoints[rule] for rule in RULES[:2])
    for rule, checkpoint in checkpoints.items():
        saved_rule = checkpoint.get("learning_rule", checkpoint["net_params"].get("learning_rule"))
        if saved_rule != rule:
            raise ValueError(f"{rule} checkpoint declares learning_rule={saved_rule!r}.")
    for key in ("run_id", "ruleset", "seed"):
        if direct.get(key) != rflo.get(key):
            raise ValueError(f"Checkpoint pair has different {key}; use the same experiment and seed.")
    configs = [{k: v for k, v in cp["net_params"].items() if k != "learning_rule"}
               for cp in (direct, rflo)]
    if _json_safe(configs[0]) != _json_safe(configs[1]):
        raise ValueError("Checkpoint pair has different network configurations.")
    shapes = [{k: tuple(v.shape) for k, v in cp["state_dict"].items()} for cp in (direct, rflo)]
    if shapes[0] != shapes[1]:
        raise ValueError("Checkpoint pair has different state tensor shapes.")


def file_fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(batch):
    """Hash dtype, shape and values, not the batch filename or RNG seed."""
    digest = hashlib.sha256()
    for tensor in batch:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_batch(batch):
    if len(batch) != 3 or any(not torch.is_tensor(x) for x in batch):
        raise ValueError("A batch must contain inputs, labels and masks tensors.")
    x, y, mask = batch
    if x.ndim != 3 or y.ndim != 3 or mask.shape != y.shape or x.shape[:2] != y.shape[:2]:
        raise ValueError("Expected inputs [B,T,input], labels/masks [B,T,output].")
    if min(*x.shape, *y.shape) < 1 or any(not torch.isfinite(v).all() for v in batch):
        raise ValueError("Batch dimensions must be positive and all values finite.")
    if not torch.any(mask != 0):
        raise ValueError("An all-zero loss mask cannot diagnose learning gradients.")


def make_batch(checkpoint, seed=BATCH_SEED):
    """Generate once from saved task parameters, independently of model RNG use."""
    params = deepcopy(checkpoint.get("task_params"))
    if not params or not {"hp", "rules"} <= params.keys():
        raise ValueError("Need saved ring-task task_params (hp/rules), or --batch-file.")
    params["hp"]["rng"] = np.random.RandomState(seed)
    params["hp"]["seed"] = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    batch, _ = mpn_tasks.generate_trials_wrap(
        params, BATCH_SIZE, rules=params["rules"], mode_input="random_batch", device="cuda")
    return tuple(batch)


def build_model(checkpoint, device, dtype, feedback="saved"):
    """Restore saved tensors with explicitly documented diagnostic gradient settings."""
    params = deepcopy(checkpoint["net_params"])
    if params.get("net_type") != "dmpn":
        raise ValueError("This diagnostic currently supports dmpn checkpoints only.")
    if params.get("loss_type", "MSE") != "MSE":
        raise ValueError("This diagnostic requires a masked-MSE checkpoint; custom losses are not supported.")
    original = {"feedback_mode": params.get("feedback_mode", "exact_spatial"),
                "mp_type": params["ml_params"].get("mp_type", "mult"),
                "rflo_trace_rho": params["ml_params"].get("rflo_trace_rho"),
                "input_mode": params.get("input_mode", "match"),
                "cross_layer_steps": params.get("cross_layer_steps", 0),
                "local_bias_mode": params["ml_params"].get("local_bias_mode", "exact")}
    # Build with the saved feedback topology so every saved buffer loads strictly.
    params.update(input_mode="match", cross_layer_steps=0, learning_rule="bptt")
    params["ml_params"]["local_bias_mode"] = "direct"
    net = mpn.DeepMultiPlasticNet(params, verbose=False).to(device=device, dtype=dtype)
    net.load_state_dict(checkpoint["state_dict"])
    if feedback == "exact_spatial":
        net.feedback_mode = "exact_spatial"  # no new feedback buffers are needed
    net.eval()
    for layer in net.mp_layers:
        layer.assert_local_config()
    return net, original


def finite_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def rms(value):
    return finite_number(value.detach().double().square().mean().sqrt().item())


def ratio(numerator, denominator):
    return (finite_number(numerator / denominator)
            if numerator is not None and denominator is not None and denominator > 0 else None)


class TraceRecorder:
    """Observe existing local methods without changing gradients or retaining traces.

    The correction (W*A for mult, A for add) and base (1+M for mult, 1 for add)
    are measured BEFORE the step advances A or M, matching the values
    used in the RFLO gradient. Write/clipping statistics describe the subsequent
    M update. Fractions cover all batch/synapse entries, including zero-loss times.
    """

    def __init__(self, net, near_zero=1e-8):
        self.rows = []
        self.near_zero = near_zero
        for index, layer in enumerate(net.mp_layers):
            self._attach(layer, index)

    def _attach(self, layer, index):
        pending = []
        counter = [0]

        def observe_step(original):
            def wrapped(this, x, phi_prime, ell, eta, lam, update_mask=None):
                base = this._direct_weight_factor()
                trace = getattr(this, "A", None)
                correction = (torch.zeros_like(base) if trace is None
                              else this._modulation_read_weight().unsqueeze(0) * trace)
                factor = base + correction
                # Use absolute magnitude for a signed/zero base; never divide elementwise.
                valid = base.abs() > self.near_zero
                flips = ((factor * base < 0) & valid).sum().item()
                row = dict(layer=f"MP{index + 1}", step=counter[0],
                           base_rms=rms(base), wa_rms=rms(correction), factor_rms=rms(factor),
                           trace_rms=0.0 if trace is None else rms(trace),
                           base_near_zero_fraction=(~valid).double().mean().item(),
                           correction_dominates_fraction=(correction.abs() > base.abs()).double().mean().item(),
                           sign_flip_fraction=ratio(flips, valid.sum().item()),
                           nonfinite_factor_fraction=(~torch.isfinite(factor)).double().mean().item(),
                           phi_prime_rms=rms(phi_prime))
                row["wa_to_base_rms_ratio"] = ratio(row["wa_rms"], row["base_rms"])
                # Candidate gains before update masks or write gates; disabled
                # caps (and the trace-free hebb_pre path) report zero.
                row["trace_gain_clipped_fraction"] = 0.0
                if this.rflo_trace_rho is not None and trace is not None:
                    k = this._assoc * eta[None] * phi_prime[..., None] * x[:, None, :].square()
                    raw_gain = lam[None] + k * this._modulation_read_weight()[None]
                    row["trace_gain_clipped_fraction"] = (
                        raw_gain.abs() > this.rflo_trace_rho).double().mean().item()
                pending.append(row)
                self.rows.append(row)
                counter[0] += 1
                return original(x, phi_prime, ell, eta, lam, update_mask=update_mask)
            return wrapped

        # hebb_pre can collapse RFLO to the direct fast path; its correction is zero.
        for name in ("_local_step_diag", "_local_step_direct"):
            setattr(layer, name, MethodType(observe_step(getattr(layer, name)), layer))
        update = layer.update_M_matrix_local_fast

        def wrapped_update(this, *args, **kwargs):
            result = update(*args, **kwargs)
            row = pending.pop()
            if this.modulation_bounds and this.m_act != "scaled_tanh":
                clipped = (this.M_pre < this.M_bounds[1]) | (this.M_pre > this.M_bounds[0])
                row["write_clipped_fraction"] = clipped.double().mean().item()
            else:
                row["write_clipped_fraction"] = 0.0
            gate = this._modulation_write_derivative()
            row["write_gate_zero_fraction"] = 0.0 if gate is None else (gate == 0).double().mean().item()
            return result

        layer.update_M_matrix_local_fast = MethodType(wrapped_update, layer)


def compare_vectors(left, right):
    """Zero/nonfinite vectors have undefined cosine; report null rather than zero."""
    left, right = left.detach().double().cpu().reshape(-1), right.detach().double().cpu().reshape(-1)
    ln, rn = finite_number(left.norm()), finite_number(right.norm())
    cosine = (finite_number(torch.dot(left / ln, right / rn))
              if ln is not None and rn is not None and ln > 0 and rn > 0 else None)
    difference = finite_number((left - right).norm())
    return dict(left_norm=ln, right_norm=rn, cosine=cosine,
                norm_ratio=ratio(ln, rn), difference_norm=difference,
                relative_l2_error=ratio(difference, rn),
                left_nonfinite_fraction=(~torch.isfinite(left)).double().mean().item(),
                right_nonfinite_fraction=(~torch.isfinite(right)).double().mean().item())


def parameter_layer(key):
    if key in ("W_in", "b_in"):
        return "input"
    if key in ("W_output", "b_output"):
        return "readout"
    if key[0] in ("W", "b") and (not key[1:] or key[1:].isdigit()):
        return f"MP{int(key[1:] or 0) + 1}"
    return key


def gradient_metrics(gradients):
    keys = list(gradients["bptt"])
    groups = {}
    for key in keys:
        groups.setdefault(parameter_layer(key), []).append(key)
    order = (["input"] + sorted((name for name in groups if name.startswith("MP")),
                                 key=lambda name: int(name[2:])) + ["readout"])
    order += [name for name in groups if name not in order]
    selections = [("parameter", k, [k]) for k in keys]
    selections += [("layer", name, groups[name]) for name in order if name in groups]
    selections += [("global", "all", keys)]
    rows = []
    for scope, name, members in selections:
        vectors = {r: torch.cat([gradients[r][k].reshape(-1) for k in members]) for r in RULES}
        for left, right in ((RULES[0], "bptt"), (RULES[1], "bptt"), (RULES[1], RULES[0])):
            rows.append(dict(scope=scope, name=name, left=left, right=right,
                             **compare_vectors(vectors[left], vectors[right])))
    return rows


def diagnose(net, batch, recorder_factory=TraceRecorder):
    """Compute three gradient algorithms; optionally attach a richer RFLO observer."""
    validate_batch(batch)
    dtype, device = net.W_output.dtype, net.W_output.device
    batch = tuple(v.to(device=device, dtype=dtype) for v in batch)
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    batch_hash = fingerprint(batch)
    gradients, losses, checks, traces = {}, {}, {}, []
    reference_outputs = None
    # Full BPTT uses its dedicated method, never an input-gradient splice.
    for rule in ("bptt", "local_direct", "local_diag_rflo"):
        model = deepcopy(net)
        model.learning_rule = rule
        recorder = recorder_factory(model) if rule == "local_diag_rflo" else None
        method = {"bptt": model.bptt_gradients, "local_direct": model.local_direct_gradients,
                  "local_diag_rflo": model.local_diag_rflo_gradients}[rule]
        with torch.enable_grad():
            result = method(*batch, return_outputs=True)
        outputs = result["outputs"].detach().cpu()
        losses[rule] = finite_number(result["loss"])
        if not torch.isfinite(outputs).all() or losses[rule] is None:
            raise ValueError(f"{rule}: nonfinite forward outputs/loss; gradient comparison is invalid.")
        if reference_outputs is None:
            reference_outputs = outputs.clone()
        tolerances = (1e-4, 1e-5) if dtype == torch.float32 else (1e-8, 1e-10)
        same = torch.allclose(outputs, reference_outputs, rtol=tolerances[0], atol=tolerances[1])
        same_loss = bool(np.isclose(losses[rule], losses["bptt"], rtol=tolerances[0], atol=tolerances[1]))
        checks[rule] = dict(forward_matches_bptt=bool(same),
                            loss_matches_bptt=same_loss,
                            max_output_abs_difference=float((outputs - reference_outputs).abs().max()))
        if not same or not same_loss:
            raise ValueError(f"{rule}: forward differs from BPTT; cannot compare gradients under identical conditions. "
                             "Try --dtype float64 to check for numerical accumulation differences.")
        if any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items()):
            raise RuntimeError(f"{rule}: diagnostic changed checkpoint parameters/buffers.")
        checks[rule]["checkpoint_tensors_unchanged"] = True
        gradients[rule] = {k: result[k].detach().cpu().clone() for k in model._trainable_params()}
        if recorder is not None:
            traces = recorder.rows
        del method, model, result, recorder
    if fingerprint(batch) != batch_hash or any(not torch.equal(v, net.state_dict()[k]) for k, v in before.items()):
        raise RuntimeError("Diagnostic mutated the source model or batch.")
    return dict(gradients=gradients, gradient_rows=gradient_metrics(gradients),
                trace_rows=traces, losses=losses, forward_checks=checks, batch_sha256=batch_hash)


def trace_summary(rows):
    summaries = []
    for layer in dict.fromkeys(r["layer"] for r in rows):
        selected = [r for r in rows if r["layer"] == layer]
        entry = dict(layer=layer, steps=len(selected))
        for key in selected[0]:
            if key in ("layer", "step"):
                continue
            valid = [r[key] for r in selected if r[key] is not None]
            entry[f"mean_{key}"] = float(np.mean(valid)) if valid else None
            entry[f"max_{key}"] = float(np.max(valid)) if valid else None
        summaries.append(entry)
    return summaries


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_gradient_axes(result, axes):
    rows = [r for r in result["gradient_rows"] if r["scope"] == "layer" and r["right"] == "bptt"]
    layers = list(dict.fromkeys(r["name"] for r in rows))
    for rule in RULES[:2]:
        subset = [r for r in rows if r["left"] == rule]
        for ax, field in zip(axes, ("cosine", "left_norm", "relative_l2_error")):
            ax.plot(layers, [r[field] if r[field] is not None else np.nan for r in subset],
                    marker="o", label=LABELS[rule])
    axes[1].plot(layers, [r["right_norm"] for r in rows if r["left"] == RULES[0]],
                 marker="o", label="BPTT")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].axhline(0, color="grey", lw=.6)
    for ax, title in zip(axes, ("Cosine with full BPTT", "Gradient L2 norm", "Relative L2 error vs BPTT")):
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=60)
        ax.grid(alpha=.25)
        ax.legend()
    for ax in axes[1:]:
        ax.set_yscale("symlog", linthresh=1e-10)


def save_plots(result, directory):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plot_gradient_axes(result, axes)
    fig.tight_layout()
    fig.savefig(directory / "gradient_comparison.png", dpi=150)
    plt.close(fig)
    save_trace_plot({"saved": result}, directory)


def save_trace_plot(results, directory):
    """Plot RFLO traces at each set of weights, sharing each metric's color scale."""
    paired = len(results) > 1
    layers = list(dict.fromkeys(r["layer"] for r in next(iter(results.values()))["trace_rows"]))
    fig, axes = plt.subplots(len(results), 3, squeeze=False, sharex="col", sharey="row",
                             figsize=(16, len(results) * max(3.5, .35 * len(layers))),
                             constrained_layout=True)
    cmap = deepcopy(plt.get_cmap("viridis"))
    cmap.set_bad("#dddddd")
    for column, (field, title) in enumerate(zip(
            ("wa_to_base_rms_ratio", "correction_dominates_fraction", "write_clipped_fraction"),
            ("log10(1 + RMS(correction)/RMS(base))", "Fraction |correction| > |base|", "Fraction of clipped M writes"))):
        matrices = []
        for result in results.values():
            values = np.array([[r[field] if r[field] is not None else np.nan
                                for r in result["trace_rows"] if r["layer"] == layer] for layer in layers])
            matrices.append(np.log10(1 + values) if column == 0 else values)
        maximum = max((float(v[np.isfinite(v)].max()) for v in matrices if np.isfinite(v).any()), default=0)
        vmax = max(1.0, maximum) if column == 0 else 1.0
        for row, (source, values) in enumerate(zip(results, matrices)):
            ax = axes[row, column]
            plot = ax.imshow(np.ma.masked_invalid(values), aspect="auto", interpolation="nearest",
                             cmap=cmap, vmin=0, vmax=vmax)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers)
            ax.set_xlabel("Trial time step")
            if row == 0:
                ax.set_title(title)
            if column == 0:
                ax.set_ylabel(f"{LABELS[source]}-trained checkpoint" if paired else "MP layer")
        fig.colorbar(plot, ax=axes[:, column].tolist(), shrink=.85)
    fig.suptitle("Diagonal RFLO traces at each checkpoint's weights; shared color scales\n"
                 "Pre-update eligibility factors; subsequent modulation writes. Gray = undefined.")
    filename = "trace_comparison.png" if paired else "trace_diagnostics.png"
    fig.savefig(directory / filename, dpi=150)
    plt.close(fig)


def make_summary(path, checkpoint, checkpoint_hash, net, original, result, batch, batch_file):
    """Collect one checkpoint's settings and checks for the shared report."""
    return dict(checkpoint=str(path), checkpoint_sha256=checkpoint_hash,
                   source_rule=checkpoint.get("learning_rule"), run_id=checkpoint.get("run_id"),
                   saved_settings=original,
                   diagnostic_settings=dict(feedback_mode=net.feedback_mode, input_mode="match",
                       mp_type=[layer.mp_type for layer in net.mp_layers],
                       rflo_trace_rho=original["rflo_trace_rho"],
                       local_bias_mode="direct", cross_layer_steps=0, optimizer_steps=0,
                       dtype=str(net.W_output.dtype), device=str(net.W_output.device),
                       reset_state="saved M_init; zero eligibility traces"),
                   batch=dict(shape=list(batch[0].shape), sha256=result["batch_sha256"],
                       seed=BATCH_SEED if not batch_file else None,
                       source=str(batch_file) if batch_file else "saved task_params / random_batch"),
                   losses=result["losses"], forward_checks=result["forward_checks"],
                   definitions=dict(ratio="RMS(correction) / RMS(base); mult: correction=W*A, base=1+M; add: correction=A, base=1; undefined when denominator is zero",
                       trace_gain_clipped_fraction="candidate A gains outside [-rho,rho], before update masks/write gates; zero when disabled",
                       sign_flip_fraction="among entries with abs(base)>1e-8",
                       null="undefined ratio/cosine or nonfinite value; never interpreted as zero",
                       trace_summary="unweighted mean/max over time steps with defined values; not a pooled elementwise ratio",
                       scope="raw masked-MSE gradients; layer metrics concatenate weights and biases",
                       inference="one batch describes gradient geometry, not the cause of training failure"))


def save_reports(results, summaries, directory):
    """Write one set of reports; paired results are labeled by source checkpoint."""
    hashes = {result["batch_sha256"] for result in results.values()}
    if len(hashes) != 1:
        raise RuntimeError("Checkpoint diagnostics did not use the same batch.")
    paired = len(results) > 1
    for filename, key in (("gradient_metrics", "gradient_rows"), ("trace_steps", "trace_rows"),
                          ("trace_summary", None)):
        rows = []
        for source, result in results.items():
            values = result[key] if key else trace_summary(result["trace_rows"])
            rows.extend(dict(source_checkpoint=source, **row) if paired else row for row in values)
        save_csv(directory / f"{filename}.csv", rows)
    if paired:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col", sharey="col")
        for row_axes, (source, result) in zip(axes, results.items()):
            plot_gradient_axes(result, row_axes)
            row_axes[0].set_ylabel(f"{LABELS[source]}-trained checkpoint")
        fig.suptitle("Shared batch; each row uses its own checkpoint weights and BPTT reference")
        fig.tight_layout()
        fig.savefig(directory / "checkpoint_comparison.png", dpi=150)
        plt.close(fig)
        save_trace_plot(results, directory)
        summary = dict(batch=summaries[RULES[0]]["batch"],
                       shared_batch_file="batch.pt", checkpoints=summaries,
                       comparison="Each checkpoint has its own BPTT reference; losses may differ between checkpoints.")
    else:
        save_plots(next(iter(results.values())), directory)
        summary = next(iter(summaries.values()))
    (directory / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="diagnose one checkpoint")
    source.add_argument("--run-dir", type=Path,
                        help="experiment or seed directory; choose the newest complete direct/RFLO pair")
    parser.add_argument("--batch-file", type=Path, help="reuse a previously exported batch.pt")
    parser.add_argument("--dtype", choices=("saved", "float32", "float64"), default="saved")
    parser.add_argument("--feedback", choices=("saved", "exact_spatial"), default="saved")
    parser.add_argument("--output-dir", type=Path, help="default: notebooks/diagnose_gradients/<run-id>/")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gradient diagnostics, but no CUDA device is available.")
    run_diagnostics(args, torch.device("cuda"))


def run_diagnostics(args, device):
    """Evaluate selected checkpoints sequentially using a single immutable batch."""
    paths = select_sources(args.checkpoint, args.run_dir)
    checkpoints = {source: torch.load(path, map_location="cpu", weights_only=False)
                   for source, path in paths.items()}
    if args.run_dir:
        validate_pair(checkpoints)
    first = next(iter(checkpoints.values()))
    dtypes = {cp["state_dict"]["W_output"].dtype for cp in checkpoints.values()}
    if args.dtype == "saved" and len(dtypes) != 1:
        raise ValueError("Checkpoints have different precisions; choose --dtype float32 or float64.")
    dtype = next(iter(dtypes)) if args.dtype == "saved" else getattr(torch, args.dtype)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Use float32 or float64 for this diagnostic.")
    if args.batch_file:
        payload = torch.load(args.batch_file.expanduser(), map_location="cpu", weights_only=False)
        if payload.get("ruleset") != first.get("ruleset"):
            raise ValueError("Batch and checkpoint ruleset differ.")
        batch = tuple(payload[k] for k in ("inputs", "labels", "masks"))
    else:
        batch = make_batch(first)
    batch = tuple(v.to(dtype=dtype, device=device) for v in batch)
    validate_batch(batch)
    batch_hash = fingerprint(batch)
    checkpoint_hashes = {source: file_fingerprint(path) for source, path in paths.items()}
    checkpoint_path = next(iter(paths.values()))
    experiment_name = first.get("run_id") or (
        checkpoint_path.parent.parent.name if checkpoint_path.parent.name.startswith("seed")
        else checkpoint_path.stem)
    directory = (args.output_dir or _bootstrap.ROOT / "notebooks" / "diagnose_gradients" / experiment_name).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    batch_payload = dict(zip(("inputs", "labels", "masks"), [v.detach().cpu() for v in batch]))
    batch_payload.update(ruleset=first.get("ruleset"), batch_sha256=batch_hash)
    torch.save(batch_payload, directory / "batch.pt")
    print(f"Shared batch: {tuple(batch[0].shape)}; dtype={dtype}; device={device}")
    print("Comparing raw gradients at each checkpoint's weights; no optimizer updates.")
    results, summaries = {}, {}
    for source, path in paths.items():
        print(f"Checkpoint: {path}")
        net, original = build_model(checkpoints[source], device, dtype, args.feedback)
        result = diagnose(net, batch)
        if result["batch_sha256"] != batch_hash:
            raise RuntimeError("Diagnostic changed the shared batch.")
        summaries[source] = make_summary(path, checkpoints[source], checkpoint_hashes[source],
                                         net, original, result, batch, args.batch_file)
        for row in result["gradient_rows"]:
            if row["scope"] == "layer" and row["right"] == "bptt":
                print(f"{row['name']:8} {LABELS[row['left']]:14} cosine={row['cosine']} norm_ratio={row['norm_ratio']}")
        result.pop("gradients")  # retain scalar reports, not all parameter gradients
        results[source] = result
        del net
        gc.collect()  # release recorder/model reference cycles before the next checkpoint
    save_reports(results, summaries, directory)
    print(f"Saved diagnostic: {directory}")


if __name__ == "__main__":
    main()
