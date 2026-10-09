# mpn_local_learning

A minimal offshoot of `MultiTaskMPN` for developing **local (eligibility-trace)
learning rules** for Multi-Plastic Networks (MPNs), as an alternative to BPTT,
and comparing them against a leaky RNN trained with RFLO.

## Learning rules

The `learning_rule` governs the **whole** network (input embedding included):

- `bptt` — exact autograd through the unrolled forward.
- `local_exact_rowlocal` — exact row-local eligibility traces (matches the MPN
  paper's expressions).
- `local_diag_rflo` — diagonal / same-synapse RFLO approximation (drops off-synapse
  terms); for the RNN this is RFLO (Murray & Escola 2019).
- `local_direct` — direct/instantaneous 3-factor rule, no trace.

A local update is **learning signal × eligibility**, and `train_mpn.py` keeps the
two choices (plus the input embedding's update) on separate flags: `--learning-signal`
picks where each MP layer's signal comes from (`exact_spatial`, `layerwise_fa`, `dfa`,
`local_readout`, `mixed`), `--rules` picks the eligibility above, and `--input-mode`
picks how the embedding is trained. See "Choosing the learning signal" below.
Readout gradients are always exact.

## Additive and multiplicative MPNs

`--mp-type mult` (default) uses `W_eff = W*(1+M)`; `--mp-type add` uses
`W_eff = W+M`. Both support `bptt`, `local_direct`, `local_diag_rflo`, and
`local_exact_rowlocal`, with `--net mpn1` or any depth of `--net dmpn`.
The choice is independent of the learning signal, residual connections,
input/bias policy, and `--cross-layer-steps` (their existing compatibility rules
still apply). BPTT remains full main-loss BPTT under `--input-mode match`.

For example, from the repository root:

```bash
python scripts/train_mpn.py --mp-type add --task contextdelaydm1 \
  --hidden 128 128 128 --no-residual --learning-signal local_readout \
  --rules local_direct local_diag_rflo local_exact_rowlocal bptt \
  --input-mode match --local-bias-mode match --cross-layer-steps 0 \
  --lam 0.99 --rflo-trace-rho 0.99 \
  --modulation-mode hard --modulation-bound 0.1 \
  --batch-size 128 --steps 5000 --runs 3 --lr 0.001
```

For one MP layer with an embedding, use `--hidden 128`; without an embedding,
also use `--net mpn1 --learning-signal exact_spatial`. For temporal cross-layer
correction, use `--learning-signal exact_spatial --cross-layer-steps 1`.

Additive `--modulation-bound B` bounds **absolute weight increments** in
`[-B,B]`, independently of W. It does not preserve weight signs. Multiplicative
bounds remain dimensionless gains. The example's `B=0.1` is a starting value,
not an established optimum: equal bounds/eta do not match the strength of the
two architectures. Tune the bound or smooth scale when comparing them.
The additive option changes neither the activation nor the Hebbian write rule;
eta remains at its existing setting. See [the equations and tests](docs/additive_mpn.md).
Configuration JSON and checkpoints save the choice as `net_params.ml_params.mp_type`.

## Layout

- `core/` — shared library: `mpn.py` (MultiPlasticNet, DeepMultiPlasticNet),
  `rnn.py` (LeakyRNN), `gru.py` (GRU with BPTT and a GRU-specific RFLO),
  `mpn_tasks.py`, `net_helpers.py`, `helper.py`,
  `run_logging.py` (console/file output mirroring).
- `scripts/` — `train_mpn.py`, `train_rnn.py`, `train_gru.py` (lockstep BPTT-vs-local over N seeds),
  `train_common.py` (shared train/plot machinery), `tasks.py` (task adapters — the
  data/metric seam; ring tasks + sequential MNIST + adding problem), `adding_tasks.py`
  (adding-problem generator + Task), `_bootstrap.py` (path setup).
- `tests/` — `validate_local_learning.py` (rule correctness vs BPTT, all tiers),
  `test_tasks.py` (task-adapter + integration tests). See `tests/README.md`.
- `notebooks/` — Python analysis scripts: `visualize_trained_networks.py`
  (checkpoint selection, loading, parameter and performance plots), and
  `compare_mpn_rnn_performance.py`.
- `clean.py` (project root) — deletes STALE OUTPUTS only: the contents of
  `checkpoints/`, `figure/`, `figure_data/`, `log/` and the `notebooks/diagnose_gradients/`,
  `notebooks/verify_rflo_scaling/`, `notebooks/visualize_trained_networks/` output folders.
  Dry run by default; `python clean.py --run` deletes, `--keep '<glob>'` spares matching
  entries, positional names limit it to some folders. `utils/clean.py` forwards to it.
- Outputs (git-ignored): `figure/` (PNGs), `figure_data/` (`.npz` behind each figure),
  `checkpoints/` (`.pt` trained nets), `log/` (`train_mpn` console output).

## Usage

```bash
cd scripts
python train_mpn.py --net dmpn --task delaygo --runs 3   # deep MPN: BPTT vs local
python train_rnn.py --task delaygo --runs 3              # leaky RNN: BPTT vs RFLO
python train_mpn.py --net dmpn --task seqmnist --runs 3  # deep MPN on sequential MNIST
python train_rnn.py --task seqmnist --runs 3             # leaky RNN on sequential MNIST
python train_mpn.py --net dmpn --task adding --runs 3    # deep MPN on the adding problem
python train_rnn.py --task adding --runs 3               # leaky RNN on the adding problem
python train_gru.py --task seqmnist --runs 3             # GRU: BPTT vs GRU-RFLO, same tasks/protocol

cd ../tests
python validate_local_learning.py                        # rule correctness checks
python test_tasks.py                                     # task-adapter tests
```

`--task` (the `RULESET`) selects the task:
- a ring-task name (`delaygo`, `contextdelaydm1`, …) → the multitask pipeline;
- `seqmnist` (28 rows × 28 px) or `seqmnist_pixel` (784 × 1 px) → sequential MNIST
  (reads the raw idx files directly, no torchvision);
- `adding` → the adding problem (seq_len=200, 2 marks); `adding_L<len>_m<marks>`
  sweeps the sequence length / mark count, e.g. `adding_L500_m3`.

All tasks run for all three models and both BPTT and local learning. `train_gru.py`
trains a GRU (`core/gru.py`, PyTorch gate layout r/z/n) with the same lockstep
protocol: `bptt` is autograd through the unrolled recurrence; `local_diag_rflo` is a
GRU-specific RFLO in which every parameter keeps a per-unit eligibility trace gated by
that unit's own update gate `z` (the GRU analog of the leaky RNN's `alpha`) while the
recurrent sensitivity entering the gates through `W_rec` is dropped. It is exact when
`W_rec = 0` or `T = 1` and otherwise an approximation; readout gradients stay exact.
The default hidden width is 100 (three gate blocks, so about the leaky RNN's recurrent
parameter count at 180). `--learning-signal exact_spatial|layerwise_fa|dfa` and the
legacy `--feedback` work as in `train_rnn.py`. In both recurrent models the RFLO pass
takes its per-step output error from the task's loss: masked MSE inline, any other
loss (seq-MNIST's cross-entropy) through a forward-only pre-pass, so RFLO and BPTT
optimize the same objective. Before October 2026 the RNN's RFLO always used the MSE
error while reporting the task loss, so earlier `train_rnn.py` seq-MNIST RFLO curves
optimized MSE and are not comparable with new ones. Sequential MNIST
uses cross-entropy at the final step; ring tasks use masked MSE at their scored
times, and the adding problem uses final-step MSE. The task supplies the same
objective to all learning rules. The adding problem is a long-range
credit-assignment benchmark: its BPTT-vs-local gap should widen with sequence
length — the central question this project studies.

For long sequences, `train_mpn.py --batch-size 16` reduces the number of trials
per training update (default 128). Validation uses three times the training batch
size and evaluates in chunks of the training batch size. Both sizes are saved in
the experiment metadata. `--no-grad-align` disables the extra full-BPTT reference
passes used to log local/BPTT gradient cosine similarities; it does not disable
the `bptt` training rule or an exact-BPTT input update.

For example, a smaller-batch pixel-MNIST comparison with three MP layers:

```bash
python scripts/train_mpn.py --task seqmnist_pixel --hidden 128 128 128 \
  --residual --rflo-trace-rho 0.99 --lam 0.99 \
  --learning-signal exact_spatial --input-mode match --local-bias-mode direct \
  --cross-layer-steps 0 --modulation-mode hard --modulation-bound 1 \
  --rules local_direct local_diag_rflo bptt \
  --batch-size 16 --no-grad-align --steps 5000 --runs 3
```

Full BPTT retains the 784-step computation graph, including the per-sample
modulation matrices; use a smaller batch if it exceeds available GPU memory.
Reducing batch size changes samples per optimizer step, not sequence length or
the BPTT gradient definition. At batch 16, 5000 steps sample 80,000 images per
rule/seed, versus 640,000 at batch 128 (sampling is with replacement).

To scale residual MP branches, add `--residual --residual-scale 0.5`.
Each equal-width block then computes `h_out = h_in + 0.5 * a`, where `a = tanh(z)`.
The fixed, nonnegative scale defaults to `1.0`; the identity path has unit gain.
Unequal-width blocks keep `h_out = a` and the existing disabled-skip warning.
Non-default scales require residuals to be enabled. Scale `0` is supported as an
identity-limit diagnostic, not a suggested training setting.

The Hebbian write uses raw `a`, not the scaled residual increment. Local MP weight
and bias gradients receive the branch-scaled learning signal, while the raw
eligibility-trace recurrences stay unchanged. Spatial Jacobians, input traces,
and the optional one-hop cross-layer correction include the corresponding branch
gain. BPTT differentiates the same forward dynamics exactly. Heads still read
`h_out`; their gradients are not separately scaled. This forward hyperparameter
applies to every learning rule and is distinct from `--local-signal-alpha`
(mixed learning signals). Checkpoints, JSON/NPZ metadata, W&B and plot labels
record it; old checkpoints without the field use `1.0`.

Each run writes a two-panel (train / test accuracy) figure to `figure/`, the arrays
behind it to `figure_data/`, (MPN) trained nets to `checkpoints/`, and the console
output to `log/`. All four share one run ID, `<model>_<task>_<YYYYMMDD_HHMMSS>_<hash>`
(the timestamp is when the ID was created, so listings sort chronologically; the
12-hex-digit hash keeps invocations distinct even within the same second), for example:

```text
checkpoints/dmpn_contextdelaydm1_20261005_135103_a1b2c3d4e5f6/
  config.json
  seed288/
    bptt.pt
    local_diag_rflo.pt
    local_direct.pt
  seed289/
    ...
figure/dmpn_contextdelaydm1_20261005_135103_a1b2c3d4e5f6.png
figure_data/dmpn_contextdelaydm1_20261005_135103_a1b2c3d4e5f6.npz
log/dmpn_contextdelaydm1_20261005_135103_a1b2c3d4e5f6.log
```

One CLI invocation gets one folder, with a subfolder for each seed. Separate
invocations get different IDs even with identical settings. Batch size, learning
rate, steps, full architecture, input mode, modulation settings, and task setup
are saved in metadata rather than filenames. `config.json` is written before
training; each checkpoint also contains its own model, task, and training setup.
New checkpoints include two explicit DFA flags: `dfa_preset` records whether
`--dfa` was passed, and `uses_dfa` records whether that model's parameter updates
use DFA. With the preset, local-rule checkpoints have `uses_dfa=true` and the
full-BPTT baseline has `uses_dfa=false`; all have `dfa_preset=true`. Manually
configuring `direct_fa` can enable DFA without selecting the preset. The config
JSON includes both flags and `uses_dfa_by_rule`; its `uses_dfa` means any selected
model uses DFA. Hybrid BPTT with a trainable `three_factor` input embedding also
counts as using DFA when its feedback is `direct_fa`. Older files lack these flags;
absence does not mean false and cannot establish whether the CLI preset was used.
When checkpoint saving is disabled, the config JSON goes beside the figure.
RNN output naming is unchanged. Reload a saved figure without retraining via
`replot_from_npz("figure_data/<run-id>.npz")`; path helpers such as `data_path()`
refer to the current process's experiment. Explore results with the analysis
scripts in `notebooks/`.

For temporal input-layer credit without an extra BPTT pass, select
`--input-mode diag_mtrace` (exact_spatial pathway) on a `dmpn` run. Local rules
then track the first MP layer's modulation-column sensitivities to the matching
embedding weights and bias; BPTT remains a full-gradient reference. This is a
diagonal-column approximation: other modulation columns and deeper-layer temporal
paths are omitted. MP parameter updates retain their selected learning rule.
It supports unbounded, hard-clipped, and scaled-tanh modulation, residuals,
input normalization, and independently frozen input weights/biases. Trace memory
scales as batch × first MP width × embedding width × (raw input width + bias),
independently of sequence length. See [the derivation and limitations](docs/input_modulation_trace.md).

For example, change `--input-mode match` to `--input-mode diag_mtrace` in a local
learning comparison. Keep `--cross-layer-steps 0` for the original MP update rules;
`1` adds its separate correction to MP parameters only. Metadata records
`input_mode="diag_mtrace"`, and checkpoint loading restores this input mode. Existing
`match`, `exact`, and `three_factor` modes retain their behavior.

For a comparison of complete algorithm pairings, use:

```bash
python scripts/train_mpn.py --input-mode paired --learning-signal exact_spatial --cross-layer-steps 0 --local-bias-mode direct
```

`paired` defaults to BPTT, diagonal RFLO, and direct when `--rules` is omitted:

| Learning rule | Input embedding update under `paired` |
|---|---|
| `bptt` | Full BPTT (`exact`) |
| `local_diag_rflo` | First-layer modulation-column trace (`diag_mtrace`) |
| `local_direct` | Direct three-factor update (`three_factor`) |

It requires `dmpn`, an input embedding, and `exact_spatial` feedback.
`local_exact_rowlocal` is rejected because its pairing is undefined. `match`
retains its historical mapping: full BPTT for BPTT and three-factor input updates
for every local rule. To isolate the input-trace contribution, compare paired
diagonal RFLO with diagonal RFLO using `--input-mode match`, keeping all other
settings fixed. The command above explicitly disables cross-layer temporal
corrections and selects direct MP bias updates; `paired` itself changes only
the input-update policy, not those independent settings.

Metadata records `input_mode="paired"`. Console logs and checkpoints record each
model's `resolved_input_mode`; the config JSON records `resolved_input_modes`
for all selected rules. The requested `input_mode` remains `paired`, and the
effective mapping is recomputed whenever the model's learning rule changes.

### Choosing the learning signal (`--learning-signal`)

`train_mpn.py` separates three independent choices so that changing one flag changes
one thing:

| Axis | Flag | Choices |
|---|---|---|
| Learning signal (where each MP layer's error comes from) | `--learning-signal` | `exact_spatial` (default), `layerwise_fa`, `dfa`, `local_readout`, `mixed` |
| Eligibility (how MP parameters consume it) | `--rules` | `local_direct`, `local_diag_rflo`, `local_exact_rowlocal`, `bptt` |
| Input embedding update | `--input-mode` | `match` (default), `exact`, `three_factor`, `diag_mtrace`, `paired` |

The five signal modes map onto the model's two fields (`learning_signal`, `feedback_mode`):

| `--learning-signal` | Signal each MP layer learns from | internal `learning_signal` | internal `feedback_mode` |
|---|---|---|---|
| `exact_spatial` | main readout error through the true weights at every boundary (same-time spatial path; weight transport) | `global` | `exact_spatial` |
| `layerwise_fa` | main readout error through a fixed random matrix at every boundary | `global` | `layerwise_fa` |
| `dfa` | main readout error projected directly to every layer through its own random matrix | `global` | `direct_fa` |
| `local_readout` | each module's own auxiliary head error (see the next section) | `local_readout` | `exact_spatial` |
| `mixed` | `exact_spatial` plus `--local-signal-alpha` × the local head error | `mixed` | `exact_spatial` |

`exact_spatial` qualifies the spatial pathway only; temporal credit is whatever the
eligibility rule provides, so none of these is full BPTT. `bptt`'s MP layers and
readout are full BPTT of the main loss under every signal mode (its auxiliary heads, if
any, receive no gradient), and so is its embedding except under `--input-mode
three_factor`, which splices the direct three-factor rule into the embedding under the
global signal through the selected feedback pathway. That combination is a hybrid, not a
full-BPTT baseline; the per-rule line below labels it `algorithm=hybrid`.

Defaults are signal-independent, so switching `--learning-signal` alone changes only the
signal. They changed in October 2026; old runs used the previous column:

| Setting | Previous default | Current default |
|---|---|---|
| `--rules` | `bptt local_exact_rowlocal local_diag_rflo local_direct` | `bptt local_direct local_diag_rflo` |
| `--input-mode` | `exact` (local runs spliced a BPTT embedding gradient: a hybrid) | `match` (bptt = full BPTT, local runs fully local) |
| `--local-bias-mode` | `exact` | `match` (each rule's own bias update: direct for diagonal RFLO, exact for row-local; pass `direct` or `exact` to force one shared policy) |
| `--cross-layer-steps` | `1` | `0` |
| `--grad-align` | on | on (unchanged) |

`--local-bias-mode match` selects the MP bias update separately for each rule:

| Rule | Effective bias update under `match` |
|---|---|
| `local_direct` | direct, instantaneous |
| `local_diag_rflo` | direct, no bias trace |
| `local_exact_rowlocal` | exact row-local bias trace |
| `bptt` | full autograd gradient (no local bias trace) |

`match` is the default: for the default rule set it is identical to `direct`
(diagonal RFLO resolves to direct), and the row-local rule keeps its bias trace
whenever it is added. Explicit `exact` or `direct` applies one shared policy to both
trace-based local rules (the `--dfa` preset selects `direct`); `local_direct` and BPTT
always keep their native bias updates. `match` only changes MP bias eligibility, independently of the learning
signal and input mode. In particular, `--input-mode match` still uses three-factor
embedding updates for every local rule: matching the bias does not make a local
network a full-BPTT baseline. For the four-rule comparison, use
`--rules local_direct local_diag_rflo local_exact_rowlocal bptt --local-bias-mode match`.
The requested policy stays in checkpoint/config `net_params`, while checkpoints
also record `resolved_local_bias_modes` (one value per MP layer: `direct`, `exact`,
or `autograd` for BPTT); the console's per-rule summary prints the effective bias
mode. Resolution happens again when
switching rules or directly calling another rule's gradient method.

Supported signal/input combinations for `dmpn` runs with local rules:

| Signal | Allowed `--input-mode` | `--cross-layer-steps` |
|---|---|---|
| `exact_spatial` | all five | `0` or `1` |
| `layerwise_fa` | `match`, `exact`, `three_factor` | `0` only (the correction uses the true forward weights) |
| `dfa` | `match`, `exact`, `three_factor` | `0` only |
| `local_readout`, `mixed` | `match`, `three_factor`, `diag_mtrace`, `paired` | `0` only |

`--input-mode exact` under a global signal and `--input-mode three_factor` under `bptt`
are hybrids (a BPTT embedding gradient in a local run, or a local embedding rule in the
BPTT baseline); the console's per-rule line records what each rule actually runs, e.g.

```text
  bptt: input_mode=match, resolved_input_mode=exact, mp_update=full BPTT (feedback/bias/heads unused)
  local_direct: input_mode=match, resolved_input_mode=three_factor, signal=dfa, feedback=direct_fa, bias=direct (rule-fixed), heads=none
  local_diag_rflo: input_mode=match, resolved_input_mode=three_factor, signal=dfa, feedback=direct_fa, bias=direct, rho=0.99, heads=none
```

The same summary is stored per checkpoint as `effective_config`, and `signal_mode` is
recorded next to `feedback_mode`/`learning_signal` in checkpoints, `config.json`, the
`.npz` metadata and W&B. Legacy spellings still work and are translated: `--feedback
exact_spatial|layerwise_fa|direct_fa` (= `--learning-signal exact_spatial|layerwise_fa|dfa`),
`--learning-signal global` (defers to `--feedback`), and the `--dfa` preset, which keeps
its historical bundle (`dfa` + `match` + direct bias + rules `bptt local_exact_rowlocal
local_diag_rflo` + alignment off). A flag that contradicts another is rejected whatever
the argument order. `train_rnn.py` accepts the same three global names. The parser runs
in three stages (`parse_arguments`, `resolve_defaults_and_legacy_options`,
`validate_config`), and `--help` groups flags by what they act on.

### Local readout heads (`--learning-signal local_readout`)

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
rule with each head's train loss and accuracy (also logged to W&B as
`train/aux{n}_*`). Head gradients are norm-clipped as a separate group so they never
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
python scripts/train_mpn.py --task seqmnist --hidden 128 128 --seed 7
python scripts/train_mpn.py --task seqmnist --hidden 128 128 --seed 7 --learning-signal local_readout
```

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
roughly seven times more slowly. Both are recorded in checkpoints, `config.json`,
the `.npz` metadata and W&B, and appear in the legacy filename as `_pat<p>_fac<f>`
when non-default. `--lr-schedule constant` disables the scheduler, so validation loss
cannot lower any rate. This schedule choice applies to **every** rule in the command, including
BPTT; it does not change their gradient definitions. Per-module loss-driven
schedulers are not introduced here.

Start with a paired 2-by-2 comparison of head multipliers 1/3 and schedules
plateau/constant, keeping the seed and all other flags fixed. These are optimization
options, not a guarantee that a faster head improves task accuracy. Existing
weight-vs-bias decay and separate main/head gradient clipping are preserved.

The console's `lr(next)` line and W&B `lr/{parameter_name}` fields report the
effective rate for each trainable matrix after scheduling (for the next update):
`W_in`, `W`/`W1`/..., `W_output`, and active `head_W0`/... . Unused BPTT heads are
omitted. Per-head train loss/accuracy remain in the existing console `aux` lines
and W&B `train/aux{n}_*` fields. Requested multiplier/schedule are saved in config,
checkpoint, plot-data and W&B metadata; checkpoints also record final effective
`learning_rates`. Old checkpoints still load without these optional metadata keys.

`learning_signal` / `local_signal_alpha` are saved in `net_params` (checkpoints and
`config.json`) and in the `.npz` metadata; checkpoints with heads reload through
`load_net`, and pre-feature checkpoints build head-less nets. `--seed` fixes the
starting seed for any task (ring-task trials are now seeded from the per-seed numpy
stream rather than an import-time draw, so they too repeat across invocations).

`train_mpn.py` defaults to hard modulation bounds `[-1, 1]` and no regularization:
`MODULATION_MODE="hard"`, `MODULATION_BOUND=1.0`, and `REG_LAMBDA=0.0`.
Use `--modulation-mode none` for unbounded writes,
`--modulation-mode hard --modulation-bound 1` for hard clipping, or
`--modulation-mode scaled_tanh --modulation-bound 1` for smooth `B*tanh(S/B)`
writes. The bound/scale must be finite and positive. Smooth writes apply the
update mask after the nonlinearity; hard clipping retains the existing order.
Local traces and cross-layer corrections include the selected write derivative
and frozen-state mask. Set `REG_LAMBDA=1e-4` for L2 weight regularization.
Adam applies coupled weight decay to
trainable weight matrices only, with penalty `(REG_LAMBDA / 2) * sum(W**2)`;
biases and activities are not regularized. Decay is applied after task-gradient
clipping; logged losses, scheduling, and alignment remain task-only. The config
JSON and checkpoints store modulation bounds/activation/scale in
`net_params.ml_params` and regularization in `train_params.reg_lambda`.

Use `--lam 0.9` to set the fixed modulation decay in every MP layer to 0.9 for
all training rules. The value must be finite and in `[0, 1)`. Without this flag,
the existing `m_time_scale=4000` setup gives lambda=0.99 at dt=40. An explicit
value is saved as `net_params.ml_params.lam_clamp`, replacing `m_time_scale`;
the core initializes lambda from it and derives the corresponding time constant.
Eta is set separately by `--eta` (next paragraph). This parameter is separate from
the RFLO trace-gain cap below.

Use `--eta 0.3` to set the fixed Hebbian write rate eta of every MP layer. It is
saved as `net_params.ml_params.eta_clamp`, which also initializes the (untrained)
eta parameter because `eta_init='eta_clamp'`; without the flag the core's 1.0
applies. One value per MP layer, e.g. `--eta 0.3 0.1 0.1` with
`--hidden 128 128 128` (bottom to top), instead writes per-layer
`net_params.ml_params<idx>` dictionaries: copies of the shared `ml_params` with
their own `eta_clamp`, indexed in the full architecture, so under `--net dmpn` MP
layer i is `ml_params<i+1>` (the input embedding occupies index 0).
`DeepMultiPlasticNet` reads a layer's own dictionary in preference to the shared
one, so checkpoints and `config.json` restore the per-layer rates unchanged; an
all-equal list collapses to the shared form. Values must be finite; `0` freezes M
at zero and a negative value is anti-Hebbian. Because the diagonal-RFLO trace gain
is `lambda + eta*phi'*x**2*W`, a smaller eta is the most direct way to keep that
gain below 1 without the `--rflo-trace-rho` cap; like `--lam`, it changes the
forward dynamics for every rule, so paired comparisons must share it. The first
seed's architecture printout shows each layer's `Eta_init`.

For diagonal RFLO, `--rflo-trace-rho 0.99` optionally caps the MP-weight trace
recurrence gain. With `k = assoc * eta * phi_prime * x**2`, the candidate update is
`A_new = clip(lambda + k*W, -rho, rho)*A_old + k*(1+M)`; existing update masks,
write derivatives, and frozen-state gates are then applied in their usual order.
For additive MPNs the candidate is `A_new = clip(lambda + k, -rho, rho)*A_old + k`.
`rho` must be finite and strictly between 0 and 1. Omit the flag to retain the
original recurrence. This is an additional eligibility approximation: it limits
repeated amplification of old traces, but does not guarantee better gradient
alignment or training performance. It does not cap the drive term, forward
modulation, exact bias traces, or input-layer sensitivity traces. Direct,
row-local, and BPTT gradients do not use this cap. For an isolated comparison,
use `--input-mode match --local-bias-mode direct --cross-layer-steps 0`.

The cap is saved as `net_params.ml_params.rflo_trace_rho` in checkpoints and
`config.json`, and restored by the diagnostic script. Legacy checkpoints with
no such field use the original recurrence. Diagnostic `summary.json` records
the setting; trace CSVs include `trace_gain_clipped_fraction`, the fraction of
candidate gains outside `[-rho,rho]` before update masks/write gates (zero when
the cap is disabled).

### MP-input RMS normalization (`--mp-input-norm rms`)

`--mp-input-norm rms` rescales the presynaptic vector that **every** MP layer
consumes, at every time step: `x_hat = x / sqrt(mean_J(x_J**2) + eps)`, so
`||x_hat||**2` equals the layer's fan-in. The modulated forward `W(1+M)x_hat`, the
Hebbian write `eta * a * x_hat^T` and all eligibility traces use the same `x_hat`;
an identity residual skip carries the **un-normalized** stream (pre-norm placement).
It is parameter-free and stateless (no learnable gain, no buffers), so checkpoints
and `state_dict`s are unchanged and load across the two settings; `none` (the
default) is the previous computation byte for byte. It is independent of
`--input-normalize`, which standardizes the raw input once with fixed statistics.
`--mp-input-norm-eps` sets `eps` (default `1e-5`). The setting is saved as
`net_params.mp_input_norm` / `mp_input_norm_eps` in checkpoints and `config.json`,
appears in run metadata and in the console/figure notes, and is restored on reload.
See [`docs/mp_input_norm.md`](docs/mp_input_norm.md) for the equations.

Why: the recurrence gain of a row's plastic eligibility (exact row-local and BPTT
alike) is `lambda + eta*phi'*sum_J W_iJ x_J**2`, and the diagonal RFLO trace of
synapse `(i,I)` has gain `lambda + eta*phi'*W_iI*x_I**2`. Both traces stay
bounded only while these gains stay below one along the trajectory. Without the
norm, `sum_J W_iJ x_J**2` scales with the fan-in, with the magnitude of the input at
that step and, with `--residual`, with the depth of the residual stream, so no single
`--eta` is right for every layer. With `||x_hat||**2 = d` and Xavier weights the row
sum is O(1) in every layer and at every step, and `eta/(1-lambda)` becomes the one
dimensionless knob. On `contextdelaydm1` with `--hidden 64 64 64`
(`notebooks/rflo_trace_gain.py --mp-input-norm rms --eta ETA --lam 0.99`, batch 8):

| setting | row gain > 1 (per step, layers 1–3) | diagonal gain > 1 | max trace at the end | cos vs BPTT (exact / diag) |
|---|---|---|---|---|
| default `eta=1, lam=0.99`, no norm | 47–52 % | 10–17 % | `A` up to 1.6e5 | −0.5…1.0 / 0.03–0.06 |
| `rms, eta=0.01` (`eta/(1-lam)=1`) | 7–10 % | 0 % | 2–10 | 0.88–1.0 / 0.77–0.94 |
| `rms, eta=0.005` (0.5) | 0.5–3 % | 0 % | 0.9–3 | 0.97–1.0 / 0.93–0.98 |
| `rms, eta=0.003` (0.3) | 0–0.2 % | 0 % | 0.5–1.5 | 0.99–1.0 / 0.98–0.99 |
| `rms, eta=0.015, lam=0.95` (0.3) | 0–0.2 % | 0 % | 0.8–3 | 0.94–1.0 / 0.89–0.97 |

So with the norm on, `eta ≈ 0.3*(1-lambda)` keeps every layer's traces bounded
without `--rflo-trace-rho` (the capped and uncapped diagonal variants then
coincide), while the modulation still reaches `O(0.1)` per synapse. The condition
depends on the weights, so a run that grows `W` substantially may re-enter the
unstable regime; `rflo_trace_gain.py` reports the exceedance fractions for a
saved configuration. Trace boundedness is not a convergence guarantee.

Learning rules under the norm. Each MP layer's own eligibility recursion is
unchanged (its input is simply `x_hat`, which does not depend on that layer's
parameters), so the exactness statements carry over: single-layer row-local equals
BPTT, the top layer is exact under `exact_spatial`, and `--cross-layer-steps 1` is
exact at `T=2`. Learning signals that travel from an MP layer back to the stream
below it (`exact_spatial`, `layerwise_fa`, the embedding's three-factor rule and
the cross-layer correction's sources and downward sweep) pass through the norm's
transpose Jacobian `J^T g = (g - x_hat * mean_J(g_J x_hat_J)) / r` before the
identity-skip term is added; `dfa` projects straight onto the stream and is
unchanged; local readout heads read the un-normalized stream and are unchanged;
`bptt` differentiates through the norm by autograd. `--input-mode diag_mtrace` (and
`paired`, which resolves to it) is rejected with the norm on: its one-column-per-
embedding-row bookkeeping does not describe a layer whose columns share a
normalizer. `tests/test_mp_input_norm.py` checks all of the above against autograd.

Every `train_mpn.py` CLI invocation also mirrors stdout and stderr to
`log/<run-id>.log`, the same `<model>_<task>_<YYYYMMDD_HHMMSS>_<hash>` name as the
run's figure, `.npz` and checkpoint folder. The file opens under a temporary timestamped name
(`train_mpn_YYYYMMDD_HHMMSS_PID.log`, so even argument errors are captured) and is
renamed as soon as the run ID exists; both paths are printed at startup. The log
directory is anchored to the project root regardless of the working directory, and
output remains visible in the terminal. No extra flag is needed; importing
`train_mpn` and running `train_rnn.py` do not enable this logging.

## Analysis figures

The former notebooks are standalone Python scripts. They use Matplotlib's Agg
backend, save every figure as a 150-dpi PNG with tight bounding boxes, and close
figures after saving. No Jupyter kernel or graphical display is required.

From the project root, run both parameter and performance analyses together:

```bash
python notebooks/visualize_trained_networks.py
```

This searches `checkpoints/` recursively, supporting both the new run/seed folders
and existing flat checkpoint filenames. It selects a complete group containing
BPTT, diagonal RFLO, and direct for the same run and seed. Groups are ranked by
the newest requested checkpoint's modification time, then stem, numeric seed,
and directory (largest wins).
Incomplete groups are skipped; training status is not checked. The selected stem
and seed are printed. Networks are loaded once and task settings come from the
first requested rule's checkpoint, so no task setup or manually chosen seed is
needed. These plots support deep MPN (`dmpn`) checkpoints.

Optional overrides and separate analyses:

```bash
python notebooks/visualize_trained_networks.py --run-dir checkpoints/dmpn_contextdelaydm1_20261005_135103_a1b2c3d4e5f6 --trials 2000
python notebooks/visualize_trained_networks.py --analysis weights --seed 37
python notebooks/visualize_trained_networks.py --analysis performance --ckpt-stem "$CKPT_STEM"
python notebooks/compare_mpn_rnn_performance.py --mpn-file "$MPN_NPZ" --rnn-file "$RNN_NPZ"
```

`--run-dir` (an alias for `--ckpt-dir`) accepts a checkpoint root, one run folder,
or one seed folder. `CKPT_STEM` is either a new run-folder name or the legacy
filename prefix before `<rule>_seed<seed>.pt`, including its trailing underscore.
`--seed` pins a saved seed; otherwise the
newest complete seed matching the requested stem/rules is selected automatically.
`MPN_NPZ` and `RNN_NPZ` select existing plot-data files (bare names also resolve in
`figure_data/`). Checkpoint scripts also accept `--ckpt-dir` and `--rules`.
All scripts expose `--help` and `--output-dir`. Performance figures require
ring-task settings stored in `task_params`; older checkpoints without them can
still use `--analysis weights`. `--trials` defaults to 2,000 per task rule per
timing mode; forward passes use minibatches of at most 32 trials.
Example-trial plots, modulation trajectories, and all three active-fraction
plots use `mode_input="random_batch"`, with independently randomized task-period
timing across trials. The accuracy/MSE figure compares this batch against an
additional `random` batch, whose trials share period timing within each task.
Both modes use the same saved seed and trial count; each learning rule is
evaluated on the same batch within a mode.

Default output folders depend on the script, even when `--analysis` changes.
They are project-root-relative, independent of the working directory:

- `notebooks/visualize_trained_networks/`: weight heatmaps, weight/bias
  alignment to BPTT, diagonal-RFLO/direct parameter cosine similarity, weight
  distributions, accuracy, example trials, and modulation trajectories.
  Use `--analysis weights` or `--analysis performance` for only that subset;
  the default `--analysis all` produces both. Alignment plots are skipped when
  their comparison rules are absent. New-layout checkpoints produce short PNG
  filenames under `<output-dir>/<run-id>/seed<N>/` (e.g. `weight_heatmaps.png`).
  Legacy checkpoints keep their existing figure naming.
  Modulation histories are stored only for the
  representative trials actually plotted, showing the first, middle, and last MP
  layers (all layers for depths up to three; the later middle layer for even depths).
  The `modulation_active_fraction_threshold0.3`,
  `modulation_active_fraction_threshold0.6`, and
  `modulation_active_fraction_threshold0.9` figures plot the percentage of all
  synapses with `abs(M)` strictly above the named threshold at each time step.
  These figures show every MP layer using the same representative trials as
  the modulation trajectories. Each panel overlays learning rules on a shared
  0–100% scale. Fractions are computed during rollout without storing full
  modulation histories for the additional layers.
  The console reports the time-averaged, peak, and final percentages for each
  rule/layer/trial, using M after each update and all time steps.
  The `accuracy_angle_stimulus` figure has two rows: `random` on top and
  `random_batch` below. Each row includes angle accuracy, stimulus accuracy,
  and masked MSE loss on that mode's trials. Matching metrics share y-axis
  limits across rows. MSE uses the training cost mask and averages over all
  batch/time/output elements, excluding weight regularization.
- `notebooks/compare_mpn_rnn_performance/`: combined learning curves;
  filenames include a source-pair identifier to distinguish different runs.
  Supply both `--mpn-file` and `--rnn-file`. The script reads task, hidden widths,
  and feedback mode from `.npz` metadata, with a legacy filename fallback when
  metadata is missing. It checks the full hidden-layer stack, feedback mode, and
  plotted metric for agreement; task differences produce a warning.

These analysis output directories are Git-ignored. Training-script figures
continue to use `figure/`.

### Compare gradients at direct-trained and RFLO-trained checkpoints

`notebooks/diagnose_gradients.py` compares `local_direct`, `local_diag_rflo`, and
full BPTT at each checkpoint's weights, using saved initial modulation and one
shared batch. Run both trained states from one experiment with a single command:

```bash
python notebooks/diagnose_gradients.py \
  --run-dir checkpoints/dmpn_contextdelaydm1_5e277d7bb0d8
```

The script automatically selects a seed containing both `local_direct.pt` and
`local_diag_rflo.pt`, using the same newest-complete-group selection as
`visualize_trained_networks.py`. It prints both selected paths. To select a
particular seed, pass its folder, e.g. `--run-dir checkpoints/<run_id>/seed979`.
It never pairs files across seed folders. The two checkpoints are evaluated
sequentially to limit GPU memory use, each with all three gradient algorithms.
No optimizer updates are performed; both checkpoint files remain untouched.

Single-checkpoint analysis is also supported:

```bash
python notebooks/diagnose_gradients.py \
  --checkpoint checkpoints/dmpn_contextdelaydm1_5e277d7bb0d8/seed979/local_diag_rflo.pt
```

The diagnostic supports deep MPNs with masked-MSE loss. It generates ring-task
trials from saved task metadata (`random_batch`) with a fixed batch size of 128
and data seed of 0, or loads an exported batch with its original size and values.
CUDA is required; the script raises an error immediately if CUDA is unavailable.
Checkpoint precision and saved feedback mode are used by default. Use
`--dtype float64` to reduce numerical differences, or `--feedback exact_spatial`
to examine traces independently of random-feedback approximations.

Both local passes use `input_mode=match`, direct MP bias updates, and
`cross_layer_steps=0` to isolate the diagonal MP weight trace. These may differ
from training settings; both original and diagnostic settings are recorded.
The BPTT reference always computes full gradients, including the input embedding.
Gradients are measured before clipping, regularization, and Adam. Each pass
resets M to saved `M_init`; eligibility traces start at zero. Forward outputs,
losses, and unchanged checkpoint tensors are checked before reporting results.

Outputs default to `notebooks/diagnose_gradients/<run-id>/`, matching the saved
training experiment name. Checkpoints without a saved run ID fall back to their
experiment folder name (for `seed<N>` folders) or checkpoint filename stem.
Repeated analyses of one experiment overwrite the same report filenames; use
`--output-dir` to keep different seeds/batches/settings separately.
All files are saved directly in this directory,
without `local_direct/` or `local_diag_rflo/` subfolders:

- `gradient_metrics.csv`: per-parameter, per-layer, and global cosine, norms,
  norm ratios, relative L2 errors, and nonfinite fractions; includes RFLO vs direct.
- `trace_steps.csv` and `trace_summary.csv`: pre-update RMS of `WA`, `1+M`, and
  their sum, correction/base RMS ratio, correction-dominance/sign-flip fractions,
  and subsequent modulation-write clipping/zero-derivative fractions.
- `checkpoint_comparison.png` (paired mode): gradient comparisons with one row
  per trained checkpoint.
- `trace_comparison.png` (paired mode): RFLO trace heatmaps with one row per
  trained checkpoint and columns for `log10(1 + RMS(WA)/RMS(1+M))`, the fraction
  `|WA| > |1+M|`, and the fraction of clipped modulation writes. Corresponding
  columns share color scales across checkpoints; gray cells are undefined.
  Single-checkpoint mode instead saves
  `gradient_comparison.png` and `trace_diagnostics.png`.
- `summary.json`: losses, consistency checks, settings, and checkpoint/batch
  fingerprints.

In paired mode, CSVs combine both checkpoints with a `source_checkpoint` column,
and `summary.json` includes settings and checks for both. Each plot row compares
against BPTT **at that row's own weights**; losses can differ between the two
checkpoints. Both use precisely
the same input/target/mask tensors and precision. The shared `batch.pt` is saved
at the output root in either mode; use `--batch-file <path>/batch.pt` to reuse it.

Layer metrics concatenate weights and biases; use the individual `W`, `W1`, …
rows to inspect MP weights alone. Trace statistics include all times, including
zero-loss periods. Near-zero bases are excluded from sign-flip fractions; RMS
ratios avoid unstable elementwise division. Undefined/nonfinite metrics are
JSON `null` or empty CSV cells, never silently replaced with zero. Trace summaries
average/maximize defined per-step values. A one-batch comparison diagnoses local
gradient geometry; it does not establish why an entire training run failed.

### Why diagonal RFLO needs the trace-gain cap

```bash
python notebooks/rflo_trace_gain.py --task contextdelaydm1 --hidden 128 128 128 --batch 16 --rho 0.99
```

On one batch, the same forward trajectory is run through exact row-local, diagonal
RFLO without a cap, and diagonal RFLO with `--rho`, and compared with BPTT. The
modulation-trace recurrence has gain `lam + k*W` per synapse in the diagonal
approximation (`k = eta * phi' * x^2`) but `lam + sum_J k_J W_iJ` for the exact row
trace; the diagonal rule keeps one signed term of that sum, so its gain can exceed 1
where the row's net feedback is damped, and the same-synapse trace then grows
geometrically within a trial while the exact trace stays bounded. The figure
(`notebooks/rflo_trace_gain/<tag>.png`, with a JSON summary and `.npz` curves) shows,
per MP layer, the ECDFs of both gains, the fraction above 1 per step, the largest
trace entry over time for the three variants, and each variant's gradient cosine with
BPTT. CPU is fine; it is a single-batch diagnostic, not a training run.

### Test RFLO scaling in a single MP layer

```bash
python notebooks/verify_rflo_scaling.py \
  --run-dir checkpoints/dmpn_contextdelaydm1_cc7fda16e6a4
```

This CUDA-only validation uses every direct/RFLO checkpoint pair in the experiment
(or one seed folder), with eight shared batches of 128 trials and fixed data
seeds 0–7. It requires exactly one MP layer and saved `exact_spatial` feedback.
It compares MP weights alone, keeping local bias direct and input mode `match`.
Full BPTT supplies the exact MP-weight reference at this depth. Checkpoints are
unchanged; all measurements use their saved final weights in float32.

`batch_metrics.csv` reports best common gains, residuals, gradient cosines,
the parallel fraction of correction energy, sample/time contribution fits,
temporal cancellation, and zero-base RFLO energy.
`heldout_metrics.csv` fits common or positive per-parameter gains on batches 0–3
and tests them on batches 4–7, without refitting. It also compares virtual Adam
directions from the eight frozen-weight gradient batches, starting with zero
moments and omitting clipping/decay. These are **not** the historical training
updates and do not establish scale invariance throughout training.

Outputs share one directory, by default
`notebooks/verify_rflo_scaling/<run-id>_scaling/`: the CSVs, `trace_steps.csv`,
`scaling_comparison.png`, `summary.json` (settings, hashes and consistency checks),
`batches.pt`, and `mp_weight_gradients.pt`. `--output-dir` overrides this location.
No per-rule result folders are created. Scalar residuals test a single shared
gain; a large scalar residual alone does not rule out diagonal preconditioning.

Validate all figure-saving workflows with small temporary fixtures:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p test_analysis_scripts.py -v
```

To add a new task, implement a `Task` in `tasks.py` (`init_params` / `valid_batch` /
`train_batch` / `accuracy`) and register it in `make_task`; the models and the
training loop don't change.
