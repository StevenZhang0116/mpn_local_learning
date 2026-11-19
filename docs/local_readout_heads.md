# Local readout heads (`--learning-signal local_readout`)

By default every MP layer's learning signal is the main readout's error delivered
through the inter-layer pathway (`--learning-signal exact_spatial`). With
`--learning-signal local_readout`, every non-top MP layer of a `dmpn` stack owns an
auxiliary linear head trained on the same task loss, labels and mask as the main
readout; the head's error projected through its weights is that layer's learning
signal, and nothing descends from the layers above. The top MP layer keeps the true
readout `W_output`, and the input embedding shares module 0's head through the
unchanged layer-0 backprojection (plus the residual identity term), so module 0 is
"embedding + first MP layer" and every other module is one MP layer. The
eligibility rule (`local_direct`, `local_diag_rflo`, `local_exact_rowlocal`) is
unchanged; only the signal it consumes changes. Under `local_exact_rowlocal` each
non-top layer's gradient is then exact for its own head's loss. `--learning-signal
mixed --local-signal-alpha a` adds `a` times the local signal to the global one,
computed independently (head errors never enter the global recursion).

Heads take part in training only; evaluation, the plotted metrics, the default
plateau scheduler's monitor, and
`grads["loss"]` stay the main readout's. The console prints an `aux` sub-line per
rule with each head's train loss and accuracy. Head gradients are norm-clipped as a separate group so they never
change the main network's clipped step. `bptt` ignores the setting (its heads get no
gradient), so the BPTT baseline is unchanged; the gradient-alignment columns exclude
the heads. The local modes imply exact_spatial feedback, require
`--cross-layer-steps 0`, and cannot use `--input-mode exact` (a BPTT input splice is
not local); should the module defaults be set to `exact` / `1`, they switch to `match`
and `0` unless you pass them explicitly. A single MP layer has no head, so
`local_readout` then coincides with `global`. Heads are initialized from a private RNG
stream, so an `exact_spatial` and a `local_readout` invocation with the same `--seed`
share init and training data (the defaults are signal-independent, so nothing else
differs between the two commands):

```bash
python scripts/training/train_mpn.py --task seqmnist --hidden 128 128 --seed 7
python scripts/training/train_mpn.py --task seqmnist --hidden 128 128 --seed 7 --learning-signal local_readout
```

## Head learning rate and LR schedule

Auxiliary readouts can use a separate Adam learning rate without changing the
local loss, learning signal, eligibility rule, or forward network:

```bash
# Append to a local_readout/mixed comparison command:
--lr 0.001 --head-lr-mult 3 --lr-schedule constant
```

`--head-lr-mult` multiplies the base `--lr` for auxiliary head weights **and biases**
only. The input embedding, MP layers, and main output readout retain the base rate.
It is ignored by BPTT (auxiliary heads remain untrained), global-signal models,
and single-MP-layer models without auxiliary heads. The default multiplier is 1.
`--lr-schedule plateau` preserves the existing `ReduceLROnPlateau` behavior driven
by the main validation loss; it schedules both main and active head groups. Its two
knobs are `--lr-patience` (training steps without a new best validation loss before a
decay; default 30) and `--lr-factor` (multiplicative decay per plateau; default 0.95).
The historical 30 / 0.95 decays fast on a noisy validation curve (about 1000x over
5000 steps in the October 2026 `contextdelaydm1` runs); `--lr-patience 200` decays
roughly seven times more slowly. Both are recorded in checkpoints, `config.json`
and the `.npz` metadata, and appear in the legacy filename as `_pat<p>_fac<f>`
when non-default. `--lr-schedule constant` disables the scheduler, so validation loss
cannot lower any rate. This schedule choice applies to **every** rule in the command, including
BPTT; it does not change their gradient definitions. Per-module loss-driven
schedulers are not introduced here.

Start with a paired 2-by-2 comparison of head multipliers 1/3 and schedules
plateau/constant, keeping the seed and all other flags fixed. These are optimization
options, not a guarantee that a faster head improves task accuracy. Existing
weight-vs-bias decay and separate main/head gradient clipping are preserved.

The console's `lr(next)` line reports the
effective rate for each trainable matrix after scheduling (for the next update):
`W_in`, `W`/`W1`/..., `W_output`, and active `head_W0`/... . Unused BPTT heads are
omitted. Per-head train loss/accuracy remain in the existing console `aux` lines.
Requested multiplier/schedule are saved in config,
checkpoint and plot-data metadata; checkpoints also record final effective
`learning_rates`. Old checkpoints still load without these optional metadata keys.

## Metadata and seeds

`learning_signal` / `local_signal_alpha` are saved in `net_params` (checkpoints and
`config.json`) and in the `.npz` metadata; checkpoints with heads reload through
`load_net`, and pre-feature checkpoints build head-less nets. `--seed` fixes the
starting seed for any task (ring-task trials are now seeded from the per-seed numpy
stream rather than an import-time draw, so they too repeat across invocations).
