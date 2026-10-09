# tests

Test / sanity-check / validation code for `mpn_local_learning`. Each file imports
`_bootstrap` first (a copy lives here, same as `scripts/` and `notebooks/`), which
adds `../core` and `../scripts` to the path so `mpn`, `rnn`, `tasks`, `train_mpn`,
etc. import cleanly. Run from this directory.

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
| `test_mp_input_norm.py` | Optional per-step MP-input RMS normalization (`--mp-input-norm rms`): norm off is byte-identical for every rule and adds no state; the transpose Jacobian equals autograd's VJP; single-layer row-local == BPTT; deep stacks under `exact_spatial`: top layer exact for long sequences, EVERY parameter (incl. the three-factor embedding) exact at `T=1`, `--cross-layer-steps 1` exact at `T=2` (with/without residual skips, hard bounds, matrix rates); `layerwise_fa` through the norm vs an autograd oracle, `dfa` unchanged, local readout heads vs their own head losses; `diag_mtrace`/`paired` rejected; CLI and config round-trip; the training script's validation forward (`forward_outputs`) equals the trained forward for both net types. CPU float64. |
| `test_input_modulation_trace.py` | Input modulation sensitivities against independent autograd graphs, BPTT limiting cases, delayed credit, nonlinear writes/masks, residuals, freezing, custom losses, CLI validation, and checkpoint reload. |
| `test_rnn.py` | `core/rnn.py` LeakyRNN RFLO against a detached-recurrence autograd oracle under masked MSE and cross-entropy (the loss now drives the traces), exact readout, and `update_masks` applied before the readout (all-frozen limit, partial and graded masks). |
| `test_rflo_trace_gain_analysis.py` | `notebooks/rflo_trace_gain.py`: the recorded diagonal gain equals `lam + k*W` and the exact row gain equals its row sum, the three local variants share init/data, curves and summaries have the right shapes, and the CLI writes the figure, JSON summary and `.npz` curves. |
| `test_gru.py` | `core/gru.py` + `scripts/train_gru.py`: BPTT against an independent `torch.nn.GRUCell` autograd oracle, GRU-RFLO traces against a detached-recurrence autograd oracle, exactness at `W_rec = 0` and `T = 1`, exact readout, feedback modes, update masks, the no-bias variant, CLI/config, and a lockstep `run_seed` integration run. |
| `test_clean.py` | The root `clean.py` output cleaner: dry run by default, `--run` deletes only the contents of the target output folders (directories and code kept, symlinks unlinked not followed), `--keep` globs and positional folder selection, refusal outside the project root. |
| `test_cli_signal_modes.py` | The unified `--learning-signal` axis and three-stage CLI: signal-independent defaults, legacy translation (`--dfa`, `--feedback`, `global`), order-independent conflict rejection, grouped `--help`, `signal_mode` metadata, the RNN script's matching names, the per-rule effective-configuration line, and `--eta` (one shared `eta_clamp` or per-layer `ml_params<idx>` overrides, rejection of bad lists). |
| `test_local_bias_match.py` | Rule-matched MP bias updates: equivalence to explicit direct/exact policies, independent row-local autograd, single-layer BPTT agreement, rule switching/direct gradient calls, CLI defaults, training diagnostics, and checkpoint reload/provenance. |
| `test_local_readouts.py` | Local readout heads (`learning_signal` = `local_readout` / `mixed`): head gradients vs independent autograd (MSE + CE), exact row-local gradients per layer against an isolated-layer oracle under each head's own loss, locality (one head perturbs one module), single-layer and `global` limiting cases (byte-identical), `mixed` linearity in alpha, private head RNG (paired init/data), bptt ignoring the heads, model/CLI guards, checkpoint round-trip, and the runner's separate clipping / head logging / metadata. |
| `validate_local_learning.py` | Correctness of the local-learning rules vs BPTT (Tiers 1-8 + a real-task integration check). float64, tight tolerances. |
| `test_tasks.py` | The task adapters in `scripts/tasks.py`: seq-MNIST data loading / batching / masking / metric / determinism, the registry, and an end-to-end "loss decreases" integration check for both MPN and RNN. |

```bash
cd tests
python validate_local_learning.py   # rule correctness (all tiers)
python test_tasks.py                 # task adapters + integration
```

Both print per-check `[PASS]/[FAIL]` lines and exit non-zero if anything fails.
