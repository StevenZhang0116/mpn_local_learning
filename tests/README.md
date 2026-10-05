# tests

Test / sanity-check / validation code for `mpn_local_learning`. Each file imports
`_bootstrap` first (a copy lives here, same as `scripts/` and `notebooks/`), which
adds `../core` and `../scripts` to the path so `mpn`, `rnn`, `tasks`, `train_mpn`,
etc. import cleanly. Run from this directory.

| File | What it checks |
|---|---|
| `test_train_mpn_batch_size.py` | Training-batch CLI validation, pixel-MNIST train/validation shapes, final-step masks, and saved batch/alignment settings. |
| `test_train_mpn_lambda.py` | Lambda CLI validation, unchanged default time constant, scalar decay dynamics in single/deep MPNs, config metadata, and checkpoint reload. |
| `test_rflo_trace_gain.py` | Optional RFLO gain cap: original full-P diagonal reference, bounded constant-drive recurrence, fused/explicit agreement with scalar gate/mask oracles, forward and other-rule invariance, CLI validation, saved metadata, checkpoint reload, and diagnostic cap statistics. |
| `test_gradient_diagnostics.py` | Same-checkpoint gradient comparisons, zero-plasticity BPTT agreement, non-invasive trace timing, clipping/pre-only writes, undefined metrics, paired checkpoint selection/validation and shared-batch reports, CUDA requirement, and reusable diagnostic artifacts (CLI export integration requires CUDA). |
| `test_rflo_scaling.py` | Scalar/diagonal fits with held-out batches, cancellation and zero-base cases, virtual Adam against PyTorch, recorded contribution reconstruction, unchanged production gradients, and single-layer row-local/BPTT agreement. |
| `test_output_layout.py` | Compact run IDs, nested checkpoint paths, full configuration metadata, and legacy RNN output naming. |
| `test_analysis_scripts.py` | New/legacy checkpoint discovery, automatic seed selection, figure saving, and metadata-based MPN/RNN comparisons. |
| `test_input_modulation_trace.py` | Input modulation sensitivities against independent autograd graphs, BPTT limiting cases, delayed credit, nonlinear writes/masks, residuals, freezing, custom losses, CLI validation, and checkpoint reload. |
| `validate_local_learning.py` | Correctness of the local-learning rules vs BPTT (Tiers 1-8 + a real-task integration check). float64, tight tolerances. |
| `test_tasks.py` | The task adapters in `scripts/tasks.py`: seq-MNIST data loading / batching / masking / metric / determinism, the registry, and an end-to-end "loss decreases" integration check for both MPN and RNN. |

```bash
cd tests
python validate_local_learning.py   # rule correctness (all tiers)
python test_tasks.py                 # task adapters + integration
```

Both print per-check `[PASS]/[FAIL]` lines and exit non-zero if anything fails.
