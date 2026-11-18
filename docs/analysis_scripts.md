# Analysis scripts

## Trained-network figures

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

## Compare gradients at direct-trained and RFLO-trained checkpoints

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

## Trace-gain diagnostic: when diagonal RFLO needs the cap

```bash
python notebooks/rflo_trace_gain.py --task contextdelaydm1 --hidden 128 128 128 --batch 16 --rho 0.99
# same diagnostic at a small write rate, with / without the MP-input norm:
python notebooks/rflo_trace_gain.py --task contextdelaydm1 --hidden 64 64 64 --batch 8 --eta 0.003 --lam 0.99
python notebooks/rflo_trace_gain.py --task contextdelaydm1 --hidden 64 64 64 --batch 8 --eta 0.003 --lam 0.99 \
  --mp-input-norm rms --mp-input-norm-skip-first
```

On one batch, the same forward trajectory is run through exact row-local, diagonal
RFLO without a cap, and diagonal RFLO with `--rho`, and compared with BPTT. The
modulation-trace recurrence has gain `lam + k*W` per synapse in the diagonal
approximation (`k = eta * phi' * x^2`) but `lam + sum_J k_J W_iJ` for the exact row
trace; the diagonal rule keeps one signed term of that sum, so its gain can exceed 1
where the row's net feedback is damped, and the same-synapse trace then grows
geometrically within a trial while the exact trace stays bounded. That is the
picture at the default `eta=1, lambda=0.99`; whether it happens at all is set by
the loop gain `lambda + k*W`, and at a small `--eta` (0.003 at `lambda=0.99`, with
or without `--mp-input-norm rms`) all three variants stay bounded and aligned, see
the [measurements table](mp_input_norm.md#measurements). The figure
(`notebooks/rflo_trace_gain/<tag>.png`, with a JSON summary and `.npz` curves) shows,
per MP layer, the ECDFs of both gains, the fraction above 1 per step, the largest
trace entry over time for the three variants, and each variant's gradient cosine with
BPTT. CPU is fine; it is a single-batch diagnostic, not a training run.

## Test RFLO scaling in a single MP layer

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

## Validation

Validate all figure-saving workflows with small temporary fixtures:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p test_analysis_scripts.py -v
```
