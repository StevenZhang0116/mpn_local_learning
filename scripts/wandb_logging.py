#!/usr/bin/env python
# coding: utf-8
"""
Optional Weights & Biases logging for the train_common experiment loop.

Isolated in this module so the `wandb` dependency is only touched when a run
opts in (RunConfig.use_wandb=True, e.g. `python train_mpn.py --wandb`);
train_common imports it lazily, so non-W&B runs never load wandb.

Run layout — ONE W&B run per (rule × seed):
  * project  : cfg.wandb_project
  * group    : the output save-stem shared by the figure / .npz
               (train_common.run_stem; also the MPN checkpoint-folder ID). This is the
               "experiment name": all K rules × N_RUNS seeds of one invocation
               land in it and compare on one page.
  * name     : "<rule>_<save-stem>_seed<seed>" — the rule + the output save-stem
               (== group) + the (short) seed, so the name is self-describing and
               unique per (rule × seed).
  * job_type : the rule, and config["rule"]=rule too — group / color by either in
               the W&B UI to get exactly K colors (K = len(rules_to_run)), with
               each of the N_RUNS seeds drawn as its own separate curve.
No tags are set by default (the name + config already carry the provenance);
cfg.wandb_tags, if provided, is passed through verbatim.
Per recorded step each run logs train/valid accuracy + loss, lr, per-phase
timing, and (non-bptt rules) the per-weight gradient-alignment cosine vs BPTT.
A final "summary" run in the same group logs the aggregate mean±std figures as
images, so the K colored curves are viewable without any UI grouping.
"""
import math
import os


def wandb_available():
    """True if `wandb` can be imported (so run_experiment can fail fast with a
    clear message instead of dying mid-training)."""
    try:
        import wandb  # noqa: F401
        return True
    except Exception:
        return False


def _clean(metrics):
    """Drop None / NaN entries so they don't pollute the W&B charts (accuracy is
    NaN for loss-only tasks; a gradient-alignment cosine is NaN when a weight's
    gradient is ~0)."""
    out = {}
    for k, v in metrics.items():
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        out[k] = v
    return out


def _base_config(cfg, experiment):
    """The shared config logged to every run of an invocation (full provenance)."""
    return {
        "experiment": experiment,                 # == the output save-stem / group
        "ruleset": cfg.ruleset,
        "rules_to_run": list(cfg.rules_to_run),
        "n_runs": cfg.n_runs,
        "n_hidden": cfg.n_hidden,
        "arch_tag": getattr(cfg, "arch_tag", ""),
        "arch_desc": getattr(cfg, "arch_desc", ""),
        "feedback_mode": cfg.feedback_mode,
        "input_mode": getattr(cfg, "input_mode", "match"),
        "input_normalize": getattr(cfg, "input_normalize", False),
        "mp_residual": getattr(cfg, "mp_residual", False),
        "cross_layer_steps": getattr(cfg, "cross_layer_steps", 0),
        "batch": cfg.batch,
        "n_datasets": cfg.n_datasets,
        "lr": cfg.lr,
        "grad_clip": cfg.grad_clip,
        "metric": cfg.metric,
        "acc_label": cfg.acc_label,
        "device": str(cfg.device),
        "dtype": str(cfg.dtype),
        "file_prefix": cfg.file_prefix,
    }


def _init_kwargs(cfg, experiment):
    """The wandb.init kwargs common to the per-rule runs and the summary run."""
    return {
        "project": getattr(cfg, "wandb_project", None) or "mpn_local_learning",
        "entity": getattr(cfg, "wandb_entity", None),
        "mode": getattr(cfg, "wandb_mode", None),        # None → wandb default (online)
        "dir": getattr(cfg, "wandb_dir", None),          # None → wandb default (./wandb)
        "group": getattr(cfg, "wandb_group", None) or experiment,
        # reinit="create_new" keeps K runs LIVE at once (legacy reinit=True would
        # finish the previous run on each init, breaking the lockstep loop that
        # trains every rule within one step).
        "reinit": "create_new",
    }


class WandbLogger:
    """K live W&B runs (one per rule) for a SINGLE seed. log_step() writes one
    metric row per rule per recorded training step; finish() closes them all.
    A fresh logger is built per seed, so at most K runs are open at once."""

    def __init__(self, cfg, seed, run_idx, experiment):
        import wandb
        base = _base_config(cfg, experiment)
        common = _init_kwargs(cfg, experiment)
        # No default tags — the run name (rule + save-stem + seed) and config already
        # carry all the provenance. Only user-supplied cfg.wandb_tags pass through.
        tags = list(cfg.wandb_tags) if getattr(cfg, "wandb_tags", None) else None
        self.runs = {}
        for rule in cfg.rules_to_run:
            # rule + the output save-stem (== experiment/group) + the short seed.
            run = wandb.init(
                name=f"{rule}_{experiment}_seed{seed}",
                job_type=rule,
                tags=tags,
                config={**base, "rule": rule, "seed": int(seed), "run_idx": run_idx},
                **common,
            )
            # Pin the TRAINING STEP as the x-axis for every metric so each metric
            # renders as a LINE over the learning trajectory (not a per-run bar).
            # Use the run-object method (NOT the global wandb.define_metric) because
            # K runs are live at once here — the global one is ambiguous across them.
            run.define_metric("step")
            run.define_metric("*", step_metric="step")
            self.runs[rule] = run

    def log_step(self, rule, step, metrics):
        run = self.runs.get(rule)
        if run is None:
            return
        payload = _clean(metrics)
        if payload:
            # The training step travels IN the payload as the "step" metric (the
            # x-axis defined above), so every key plots as a line vs training step,
            # shared across every run in the group. No step= kwarg — the custom
            # step_metric drives the axis.
            payload["step"] = int(step)
            run.log(payload)

    def finish(self):
        for run in self.runs.values():
            try:
                run.finish()
            except Exception:
                pass
        self.runs = {}


def log_summary(cfg, experiment, fig_path, align_fig_path, final_summary):
    """One extra 'summary' run in the same group: logs the aggregate mean±std
    figures (the K-colored curves) as images plus the final-metric table, so the
    comparison is viewable without configuring any UI grouping. Best-effort."""
    import wandb
    tags = list(cfg.wandb_tags) if getattr(cfg, "wandb_tags", None) else None
    run = wandb.init(
        name=f"summary_{experiment}",
        job_type="summary",
        tags=tags,
        config=_base_config(cfg, experiment),
        **_init_kwargs(cfg, experiment),
    )
    try:
        imgs = {}
        if fig_path and os.path.exists(fig_path):
            imgs["figures/accuracy"] = wandb.Image(fig_path)
        if align_fig_path and os.path.exists(align_fig_path):
            imgs["figures/grad_alignment"] = wandb.Image(align_fig_path)
        if imgs:
            run.log(imgs)
        # Final-metric table (rule × mean/std). NOTE: deliberately NOT logged as
        # per-rule summary scalars (`final/<rule>/...`) — W&B renders single-value-
        # per-run metrics as BAR panels, which is exactly the bar plot we want to
        # avoid. The learning trajectory lives in the per-rule runs' line charts;
        # this table is the only end-of-run scalar view.
        table = wandb.Table(
            columns=["rule", "train_mean", "train_std", "test_mean", "test_std"])
        for rule_label, vals in final_summary.items():
            table.add_data(rule_label, *vals)
        run.log({"final_metrics": table})
    finally:
        run.finish()
