# tests

Test / sanity-check / validation code for `mpn_local_learning`. Each file imports
`_bootstrap` first (a copy lives here, same as in `scripts/training/`,
`scripts/tasks/`, `notebooks/` and `utils/`), which puts `core/`,
`scripts/training/` and `scripts/tasks/` on the path so `mpn`, `rnn`, `tasks`,
`train_mpn`, etc. import cleanly under their flat names. Run a single file from this directory, or the
whole suite with `run_all.py` (see the end of this page).

| File | What it checks |
|---|---|
| `test_additive_mpn.py` | Additive single/deep BPTT against an independent unroll; direct, diagonal RFLO and row-local gradients against detached-graph oracles; local/mixed heads, exact/direct bias, nonlinear writes, masks/freezes, input-column traces, capped RFLO, one-hop cross-layer credit, CLI metadata and lockstep training/checkpoint reload for all four rules. CPU float64. |
| `test_residual_scale.py` | Scaled residuals against independent full-unroll, isolated-layer and scalar-synapse autograd oracles; identity/legacy limits, raw Hebbian writes, spatial signals, input traces, cross-layer correction, CLI and saved configuration. |
| `test_local_head_optimization.py` | Auxiliary head LR isolation (weights and biases), legacy Adam/plateau equivalence, BPTT/no-head invariance, constant and plateau scheduling, decay groups, CLI validation/wiring, per-layer logs and saved metadata. |
| `test_train_mpn_batch_size.py` | Training-batch CLI validation, pixel-MNIST train/validation shapes, final-step masks, and saved batch/alignment settings. |
| `test_train_mpn_lambda.py` | Lambda CLI validation, unchanged default time constant, scalar decay dynamics in single/deep MPNs, config metadata, and checkpoint reload. |
| `test_rflo_trace_gain.py` | Optional RFLO gain cap: original full-P diagonal reference, bounded constant-drive recurrence, fused/explicit agreement with scalar gate/mask oracles, forward and other-rule invariance, CLI validation, saved metadata, checkpoint reload, and diagnostic cap statistics. |
| `test_gradient_diagnostics.py` | Same-checkpoint gradient comparisons, zero-plasticity BPTT agreement, non-invasive trace timing, clipping/pre-only writes, undefined metrics, paired checkpoint selection/validation and shared-batch reports, CUDA requirement, and reusable diagnostic artifacts (CLI export integration requires CUDA). |
| `test_rflo_scaling.py` | Scalar/diagonal fits with held-out batches, cancellation and zero-base cases, virtual Adam against PyTorch, recorded contribution reconstruction, unchanged production gradients, and single-layer row-local/BPTT agreement. |
| `test_output_layout.py` | Compact run IDs, nested checkpoint paths, full configuration metadata, and legacy RNN output naming. |
| `test_analysis_scripts.py` | New/legacy checkpoint discovery, automatic seed selection, figure saving, and metadata-based MPN/RNN comparisons. |
| `test_mp_input_norm.py` | Optional per-step MP-input RMS normalization (`--mp-input-norm rms`): norm off is byte-identical for every rule and adds no state; the transpose Jacobian equals autograd's VJP; single-layer row-local == BPTT; deep stacks under `exact_spatial`: top layer exact for long sequences, EVERY parameter (incl. the three-factor embedding) exact at `T=1`, `--cross-layer-steps 1` exact at `T=2` (with/without residual skips, hard bounds, matrix rates); `layerwise_fa` through the norm vs an autograd oracle, `dfa` unchanged, local readout heads vs their own head losses; `diag_mtrace`/`paired` rejected; CLI and config round-trip; the training script's validation forward (`forward_outputs`) equals the trained forward for both net types; `mp_input_norm_skip_first` (layer 0 raw while layers 1.. are normalized, exactness re-checked, `diag_mtrace` accepted again and exact at `T=1`, single-MP-layer nets reject it, CLI/tag). CPU float64. |
| `test_input_modulation_trace.py` | Input modulation sensitivities against independent autograd graphs, BPTT limiting cases, delayed credit, nonlinear writes/masks, residuals, freezing, custom losses, CLI validation, and checkpoint reload. |
| `test_rnn.py` | `core/rnn.py` LeakyRNN RFLO against a detached-recurrence autograd oracle under masked MSE and cross-entropy (the loss now drives the traces), exact readout, and `update_masks` applied before the readout (all-frozen limit, partial and graded masks). |
| `test_rflo_trace_gain_analysis.py` | `notebooks/rflo_trace_gain.py`: the recorded diagonal gain equals `lam + k*W` and the exact row gain equals its row sum, the three local variants share init/data, curves and summaries have the right shapes, every gradient is taken under the task's own loss (masked MSE by default, cross-entropy for two-hop recall, where the top layer's exact row-local gradient equals BPTT's and differs from the MSE one), and the CLI writes the figure, JSON summary and `.npz` curves. |
| `test_gru.py` | `core/gru.py` + `scripts/training/train_gru.py`: BPTT against an independent `torch.nn.GRUCell` autograd oracle, GRU-RFLO traces against a detached-recurrence autograd oracle, exactness at `W_rec = 0` and `T = 1`, exact readout, feedback modes, update masks, the no-bias variant, CLI/config, and a lockstep `run_seed` integration run. |
| `test_clean.py` | The root `clean.py` output cleaner: dry run by default, `--run` deletes only the contents of the target output folders (directories and code kept, symlinks unlinked not followed), `--keep` globs and positional folder selection, refusal outside the project root. |
| `test_cli_signal_modes.py` | The unified `--learning-signal` axis and three-stage CLI: signal-independent defaults, legacy translation (`--dfa`, `--feedback`, `global`), order-independent conflict rejection, grouped `--help`, `signal_mode` metadata, the RNN script's matching names, the per-rule effective-configuration line, and `--eta` (one shared `eta_clamp` or per-layer `ml_params<idx>` overrides, rejection of bad lists). |
| `test_local_bias_match.py` | Rule-matched MP bias updates: equivalence to explicit direct/exact policies, independent row-local autograd, single-layer BPTT agreement, rule switching/direct gradient calls, CLI defaults, training diagnostics, and checkpoint reload/provenance. |
| `test_local_readouts.py` | Local readout heads (`learning_signal` = `local_readout` / `mixed`): head gradients vs independent autograd (MSE + CE), exact row-local gradients per layer against an isolated-layer oracle under each head's own loss, locality (one head perturbs one module), single-layer and `global` limiting cases (byte-identical), `mixed` linearity in alpha, private head RNG (paired init/data), bptt ignoring the heads, model/CLI guards, checkpoint round-trip, and the runner's separate clipping / head logging / metadata. |
| `test_dfa.py` | Layerwise/direct feedback alignment against independent autograd oracles, fixed and serialized feedback matrices, explicit vs fused trace paths, modulation modes (none/hard/scaled-tanh) and clamp-derivative endpoints, frozen plastic states, L2 regularization against an explicit gradient, and the `--dfa` preset with its conflict checks. |
| `test_residual_correction.py` | Residual stacks with `--cross-layer-steps 1`: two-step MP gradients against a full-sequence autograd oracle (bounded, frozen and scaled-tanh writes), no correction when writes are disabled, and the CLI's residual setting at every depth. |
| `test_run_logging.py` | `core/run_logging.py`: stdout/stderr mirroring and flushing, the project-relative default path, renaming the active log to the run ID, stream restoration on exceptions, capture of a `train_mpn` CLI run, and no log file for `--help`. |
| `test_mpn_revise.py` | The optimized `core/mpn.py` against the reference `core/mpn_archive.py`, for the baseline configuration the archive implements (features added since are covered by the other files): outputs, loss and every gradient for BPTT and the three local rules on `dmpn` and `mpn1` (float64 and float32), masked updates, `return_outputs=False`, scratch-buffer reuse, pixel-MNIST alignment and a timing benchmark. Requires CUDA; run as a script. |
| `validate_local_learning.py` | Correctness of the local-learning rules vs BPTT, as numbered tiers (1-11 and 13-17; there is no tier 12): exact row-local vs BPTT, diagonal RFLO limits and its zeroed reference, direct rule, real-task integration, deep input embedding, RNN RFLO, pre-only Hebbian writes, custom and cross-entropy losses, feedback modes, input modes, input normalization, partially frozen embedding, residual stacks and cross-layer steps. float64, tight tolerances. |
| `test_tasks.py` | The task adapters in `scripts/tasks/tasks.py`: seq-MNIST data loading / batching / masking / metric / determinism, the adding problem, two-hop associative recall (the tables decoded from the inputs reproduce every target, trial layout, the paired and reproducible held-out set, per-query accuracy, one real cross-entropy update for BPTT and a local rule on a two-MP-layer net, the RNN and the GRU), the registry including name parsing, and an end-to-end "loss decreases" integration check for both MPN and RNN. |

Run everything with one command, from the repository root:

```bash
python tests/run_all.py            # one verdict line per file; non-zero exit on any failure
python tests/run_all.py -k norm    # only the files whose name contains "norm"
```

`run_all.py` starts one subprocess per file, because the suite mixes two styles:
most files are `unittest` modules (`python -m unittest <name>` from this directory),
while `validate_local_learning.py`, `test_tasks.py` and `test_mpn_revise.py` are
scripts that print per-check `[PASS]/[FAIL]` lines and exit non-zero on a failure.
`test_mpn_revise.py` needs a CUDA GPU and is skipped when there is none. Network
weights are drawn from numpy's global RNG, so every builder in these files seeds
numpy as well as torch. Run the suite on a compute node, not on a login node; it
takes about fifteen minutes, most of it in `test_analysis_scripts.py` and the GPU
comparison.
