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
- `notebooks/` — Python analysis scripts: `visualize_performance.py`,
  `visualize_trained_networks.py`, and `compare_mpn_rnn_performance.py`.
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

From the project root, select saved data with these options:

```bash
python notebooks/visualize_performance.py --ckpt-stem "$CKPT_STEM" --seed "$SEED" --trials 500
python notebooks/visualize_trained_networks.py --ckpt-stem "$CKPT_STEM" --seed "$SEED"
python notebooks/compare_mpn_rnn_performance.py --mpn-file "$MPN_NPZ" --rnn-file "$RNN_NPZ"
```

`CKPT_STEM` is the checkpoint filename prefix before `<rule>_seed<seed>.pt`,
including its trailing underscore. Set `SEED` to the saved seed, and `MPN_NPZ`
and `RNN_NPZ` to existing plot-data files (bare names also resolve in
`figure_data/`). The default selections retain the old notebook settings;
override them when those runs are not present. Checkpoint scripts also accept
`--ckpt-dir` and `--rules`. All scripts expose `--help` and `--output-dir`.

Default outputs are project-root-relative, independent of the working directory:

- `notebooks/visualize_performance/`: accuracy, example trials, modulation
  trajectories, and accuracy JSON. Modulation histories are stored only for the
  representative trials actually plotted.
- `notebooks/visualize_trained_networks/`: weight heatmaps, weight/bias
  alignment to BPTT, weight distributions, and bias-alignment JSON. Alignment
  plots are skipped if there is no BPTT reference or no other rule to compare.
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
