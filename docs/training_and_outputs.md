# Training scripts, tasks and outputs

Details of the training entry points (`scripts/training/train_mpn.py`, `train_rnn.py`,
`train_gru.py`). The quick-start commands are in the [README](../README.md).

## Environment

`pip install -r requirements.txt` installs what the code imports: PyTorch, NumPy,
SciPy and Matplotlib (plus Pillow for two test files). The suite was
last verified with Python 3.9 and PyTorch 2.5.1.

The PyTorch build has to support the GPU. A Tesla V100 has compute capability 7.0,
which recent CUDA 13 builds no longer include: with such a build
`torch.cuda.is_available()` still returns `True`, but the first tensor moved to the
GPU fails with `CUDA error: no kernel image is available for execution on the
device`. `torch.cuda.get_arch_list()` shows what a build supports; it must contain
`sm_70` for a V100.

## Tasks, models and objectives

`--task` (the `RULESET`) selects the task:
- a ring-task name (`delaygo`, `contextdelaydm1`, …) → the multitask pipeline;
- `seqmnist` (28 rows × 28 px) or `seqmnist_pixel` (784 × 1 px) → sequential MNIST
  (reads the raw idx files directly, no torchvision);
- `adding` → the adding problem (seq_len=200, 2 marks); `adding_L<len>_m<marks>`
  sweeps the sequence length / mark count, e.g. `adding_L500_m3`;
- `twohop` → two-hop associative recall (4 symbols per set, 4 blank steps before
  the query); `twohop_K<symbols>_d<delay>` changes either, e.g. `twohop_K8_d6`.
  See [Two-hop associative recall](#two-hop-associative-recall) below.
- `wh_mid` / `wh_out` → nonlinear system identification, the two halves of a paired
  control; `_a<percent>`, `_b<percent>`, `_g<gain>` and `_T<steps>` change the
  system, e.g. `wh_mid_b95_g2p5_T32`.
  See [Nonlinear system identification](#nonlinear-system-identification) below.

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
and two-hop recall use cross-entropy at the final step; ring tasks use masked MSE
at their scored times, the adding problem uses final-step MSE, and the
nonlinear-system tasks use MSE at every step. The task supplies the same
objective to all learning rules. The adding problem is a long-range
credit-assignment benchmark: its BPTT-vs-local gap should widen with sequence
length — the central question this project studies.

## Two-hop associative recall

`--task twohop` (`scripts/tasks/associative_recall_tasks.py`) separates one memory lookup
from two under an identical memory load. Each trial draws two fresh random
permutations, `f: A -> B` and `g: B -> C`, over `K` symbols per set:

| Steps | Input |
|---|---|
| `0 .. K-1` | the `K` pairs `(A_a, B_f(a))`, in random order |
| `K .. 2K-1` | the `K` pairs `(B_b, C_g(b))`, in random order |
| next `delay` steps | blank |
| final step | the query: one symbol |

| Query | Final input | Answer |
|---|---|---|
| one-hop | `B_b` | `C_g(b)` |
| two-hop | `A_a`, with `b = f(a)` | the same `C_g(f(a))` |

The input has one channel per symbol (`3K` in all): a pair step lights two, the
query one. The output is a `K`-way classification scored with cross-entropy on the
final step only, so chance is `1/K`. The `A -> B` table always comes first: with
the opposite order a network could look `C` up while the `(A, B)` pair is shown and
store it under the key `A`, which would turn the two-hop query into a single lookup.

Training trials are two-hop with probability 0.5. The held-out set is fixed
(`task_params.valid_seed`, default 1234) and paired: each stored history is asked
once per query kind, with the same queried symbol and the same answer. The first
half of the batch holds the one-hop trials and the second half the two-hop trials
of the same histories, so a difference between the two halves is a difference in
the number of lookups and nothing else. `K`, the delay and the two-hop probability
are saved in `task_params` (`n_symbols`, `delay`, `p_two_hop`).

The training loop logs and plots the accuracy over the whole held-out set, in
which the two kinds weigh equally: solving one-hop while guessing two-hop gives
`0.5 + 0.5/K`, 62.5 % at `K = 4`. `TwoHopRecallTask.query_accuracy(output, labels,
inputs)` returns the two accuracies separately; it is not called by the training
loop, so apply it to a saved checkpoint's outputs on the held-out batch.

## Nonlinear system identification

`--task wh_mid` and `--task wh_out` (`scripts/tasks/nonlinear_system_tasks.py`) ask the
network to imitate a hidden system online. Each trial is a fresh white-noise
waveform `x_t ~ N(0, 1)`, one scalar per step; the network sees only `x_t` and
predicts `y_t` at every step. The system has a fast memory, a slow memory and one
saturating nonlinearity, and starts each trial at zero:

```
fast memory   s_t = alpha * s_{t-1} + (1 - alpha) * x_t

wh_mid        y_t = beta * y_{t-1} + (1 - beta) * tanh(gain * s_t)
wh_out        r_t = beta * r_{t-1} + (1 - beta) * s_t,    y_t = tanh(gain * r_t)
```

The two tasks are a paired control: the same waveforms (for the same seed and
architecture), the same two memories and the same supervision; only the place of
the nonlinearity moves. In `wh_out` the two memories merge into one linear filter
and the nonlinearity is applied once, at the output. In `wh_mid` the slow memory
accumulates a signal that has already been transformed, so a large input counts
no more than a moderate one.

| Name fragment | Sets | Default |
|---|---|---|
| `_a<percent>` | `alpha`, the fast memory | 0.6 (`_a60`) |
| `_b<percent>` | `beta`, the slow memory | 0.9 (`_b90`) |
| `_g<gain>` | `gain`, the input scale of the `tanh` | 6 |
| `_T<steps>` | sequence length | 64 |

`p` is the decimal point (`_g2p5`, `_b97p5`); the fragments go in any order after
`wh_mid` or `wh_out`.

The loss is the MSE over all steps, and the training loop plots it. The console's
accuracy columns show the skill relative to the best causal linear filter,
`1 - MSE_model / MSE_FIR`, on the same batch: 0 is as good as a linear filter, 1
is exact, and a network worse than the linear filter scores below 0 (an untrained
one far below). At the defaults a linear filter already explains about 90 % of the
target variance in both tasks, so the skill, not the raw MSE or R², says how much
of the nonlinear part was learned. The filter (`fir_order` lags plus an intercept)
is fitted by least squares on `fir_n` sequences from `fir_seed`, independent of the
training and held-out data. Its held-out MSE is saved as
`task_params.fir_valid_mse`, next to `variant`, `alpha`, `beta`, `gain`, `seq_len`
and `valid_seed`, so `1 - valid_MSE / fir_valid_mse` turns a saved validation
curve into the skill.

At `gain = 6` the two tasks have the same linear baseline but not the same
saturation: the `tanh` argument has a standard deviation of about 3.0 in `wh_mid`
and 1.25 in `wh_out`. `wh_mid_g2p5` matches the argument scale of the default
`wh_out` instead. `NonlinearSystemTask.targets(x)` also returns the fast state
`s_t`, which the training loop does not use.

## Activation function

`train_mpn.py --activation {tanh,relu,softplus,sigmoid,linear}` sets the activation
of the input embedding and of every MP layer; the readout stays linear. The default
`tanh` is the previous computation. All rules in a command use the same activation:
BPTT differentiates it by autograd, and the local rules use its derivative, so every
exactness statement (the top MP layer under `exact_spatial`, `--cross-layer-steps 1`
at `T=2`, `local_diag_rflo` = `local_exact_rowlocal` under `top1_abs` routing) holds
for each choice. At a pre-activation of exactly 0, `relu` has derivative 0 in both.

The choice is saved as `activation` in `config.json`, in every checkpoint and in the
`.npz` metadata, and `load_net` restores it. When it is not `tanh` it also appears in
the console header and the figure title.

With `relu` the activities are non-negative and unbounded, so the Hebbian write
`eta * h * x^T` is non-negative and can be much larger than with `tanh`; `--eta` and
`--lam` values tuned for `tanh` may need to change. `train_rnn.py` and `train_gru.py`
do not take this flag and keep `tanh`.

## Batch size and long sequences

For long sequences, `train_mpn.py --batch-size 16` reduces the number of trials
per training update (default 128). Validation uses three times the training batch
size and evaluates in chunks of the training batch size. Both sizes are saved in
the experiment metadata. `--no-grad-align` disables the extra full-BPTT reference
passes used to log local/BPTT gradient cosine similarities; it does not disable
the `bptt` training rule or an exact-BPTT input update.

For example, a smaller-batch pixel-MNIST comparison with three MP layers:

```bash
python scripts/training/train_mpn.py --task seqmnist_pixel --hidden 128 128 128 \
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
`train_mpn` and running `train_rnn.py` do not enable this logging, and
`train_mpn.py --help` prints the usage without creating a log file.

## Cleaning outputs

`clean.py` (project root) deletes STALE OUTPUTS only: the contents of
`checkpoints/`, `figure/`, `figure_data/`, `log/` and the
`notebooks/compare_mpn_rnn_performance/`, `notebooks/diagnose_gradients/`,
`notebooks/rflo_trace_gain/`, `notebooks/verify_rflo_scaling/`,
`notebooks/visualize_trained_networks/` output folders.
Dry run by default; `python clean.py --run` deletes, `--keep '<glob>'` spares matching
entries, positional names limit it to some folders. `utils/clean.py` forwards to it.

## Adding a task

To add a new task, implement a `Task` under `scripts/tasks/` (`init_params` / `valid_batch` /
`train_batch` / `accuracy`) and register it in `make_task`; the models and the
training loop don't change.
