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

`feedback_mode` ∈ {`exact_readout`, `random_fixed`} selects exact vs. fixed-random
feedback for the hidden learning signal. Readout gradients are always exact.

## Layout

- `core/` — shared library: `mpn.py` (MultiPlasticNet, DeepMultiPlasticNet),
  `rnn.py` (LeakyRNN), `mpn_tasks.py`, `net_helpers.py`, `helper.py`,
  `run_logging.py` (console/file output mirroring).
- `scripts/` — `train_mpn.py`, `train_rnn.py` (lockstep BPTT-vs-local over N seeds),
  `train_common.py` (shared train/plot machinery), `tasks.py` (task adapters — the
  data/metric seam; ring tasks + sequential MNIST + adding problem), `adding_tasks.py`
  (adding-problem generator + Task), `_bootstrap.py` (path setup).
- `tests/` — `validate_local_learning.py` (rule correctness vs BPTT, all tiers),
  `test_tasks.py` (task-adapter + integration tests). See `tests/README.md`.
- `notebooks/` — Python analysis scripts: `visualize_trained_networks.py`
  (checkpoint selection, loading, parameter and performance plots), and
  `compare_mpn_rnn_performance.py`.
- `utils/` — `clean.py` (remove saved outputs, safe by default).
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

All tasks run for both models and both BPTT and local learning, framed as
masked-MSE scored on the final step, so they drive the same loss/eligibility
machinery the local rules are derived for. The adding problem is a long-range
credit-assignment benchmark: its BPTT-vs-local gap should widen with sequence
length — the central question this project studies.

Each run writes a two-panel (train / test accuracy) figure to `figure/`, the arrays
behind it to `figure_data/`, and (MPN) trained nets to `checkpoints/`. Reload figures
without retraining via `replot_from_npz(data_path())`; explore results with the
analysis scripts in `notebooks/`.

For temporal input-layer credit without an extra BPTT pass, select
`--input-mode diag_mtrace --feedback exact_spatial` on a `dmpn` run. Local rules
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
`1` adds its separate correction to MP parameters only. The saved run stem contains
`_in-diag_mtrace`, and checkpoint loading restores this input mode. Existing
`match`, `exact`, and `three_factor` modes retain their behavior.

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
clipping; logged losses, scheduling, and alignment remain task-only. Filename
tags are `_mb-B-B` for hard clipping, `_mtanh-B` for smooth modulation, and
`_l2-1e-04` for that regularization setting. The unbounded mode has no modulation
tag. Existing hard-bound filenames stay unchanged at `B=1`.

Every `train_mpn.py` CLI invocation also mirrors stdout and stderr to
`log/train_mpn_YYYYMMDD_HHMMSS_PID.log`, following the `MultiTaskMPN` logging
pattern. The log directory is anchored to the project root regardless of the
working directory. The log path is printed at startup, and output remains
visible in the terminal. No extra flag is needed; importing `train_mpn` and
running `train_rnn.py` do not enable this logging.

## Analysis figures

The former notebooks are standalone Python scripts. They use Matplotlib's Agg
backend, save every figure as a 150-dpi PNG with tight bounding boxes, and close
figures after saving. No Jupyter kernel or graphical display is required.

From the project root, run both parameter and performance analyses together:

```bash
python notebooks/visualize_trained_networks.py
```

This selects a complete checkpoint group containing BPTT, diagonal RFLO, and
direct for the same run and seed. Groups are ranked by the newest requested
checkpoint's modification time, then stem and numeric seed (largest wins).
Incomplete groups are skipped; training status is not checked. The selected stem
and seed are printed. Networks are loaded once and task settings come from the
first requested rule's checkpoint, so no task setup or manually chosen seed is
needed. These plots support deep MPN (`dmpn`) checkpoints.

Optional overrides and separate analyses:

```bash
python notebooks/visualize_trained_networks.py --ckpt-stem "$CKPT_STEM" --trials 2000
python notebooks/visualize_trained_networks.py --analysis weights --seed 37
python notebooks/visualize_trained_networks.py --analysis performance --ckpt-stem "$CKPT_STEM"
python notebooks/compare_mpn_rnn_performance.py --mpn-file "$MPN_NPZ" --rnn-file "$RNN_NPZ"
```

`CKPT_STEM` is the checkpoint filename prefix before `<rule>_seed<seed>.pt`,
including its trailing underscore. `--seed` pins a saved seed; otherwise the
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
  their comparison rules are absent. Figures are saved as PNGs only.
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

These analysis output directories are Git-ignored. Training-script figures
continue to use `figure/`.

Validate all figure-saving workflows with small temporary fixtures:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p test_analysis_scripts.py -v
```

To add a new task, implement a `Task` in `tasks.py` (`init_params` / `valid_batch` /
`train_batch` / `accuracy`) and register it in `make_task`; the models and the
training loop don't change.
