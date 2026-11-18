# Training scripts, tasks and outputs

Details of the training entry points (`scripts/train_mpn.py`, `train_rnn.py`,
`train_gru.py`). The quick-start commands are in the [README](../README.md).

## Tasks, models and objectives

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

## Batch size and long sequences

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

## Outputs and run IDs

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
scripts in `notebooks/` (see [analysis_scripts.md](analysis_scripts.md)).

## Console log

Every `train_mpn.py` CLI invocation also mirrors stdout and stderr to
`log/<run-id>.log`, the same `<model>_<task>_<YYYYMMDD_HHMMSS>_<hash>` name as the
run's figure, `.npz` and checkpoint folder. The file opens under a temporary timestamped name
(`train_mpn_YYYYMMDD_HHMMSS_PID.log`, so even argument errors are captured) and is
renamed as soon as the run ID exists; both paths are printed at startup. The log
directory is anchored to the project root regardless of the working directory, and
output remains visible in the terminal. No extra flag is needed; importing
`train_mpn` and running `train_rnn.py` do not enable this logging.

## Cleaning outputs

`clean.py` (project root) deletes STALE OUTPUTS only: the contents of
`checkpoints/`, `figure/`, `figure_data/`, `log/` and the `notebooks/diagnose_gradients/`,
`notebooks/rflo_trace_gain/`, `notebooks/verify_rflo_scaling/`,
`notebooks/visualize_trained_networks/` output folders.
Dry run by default; `python clean.py --run` deletes, `--keep '<glob>'` spares matching
entries, positional names limit it to some folders. `utils/clean.py` forwards to it.

## Adding a task

To add a new task, implement a `Task` in `tasks.py` (`init_params` / `valid_batch` /
`train_batch` / `accuracy`) and register it in `make_task`; the models and the
training loop don't change.
