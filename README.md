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
two choices (plus the input embedding's update) on separate flags, so changing one
flag changes one thing:

| Axis | Flag | Choices |
|---|---|---|
| Learning signal (where each MP layer's error comes from) | `--learning-signal` | `exact_spatial` (default), `layerwise_fa`, `dfa`, `local_readout`, `mixed` |
| Eligibility (how MP parameters consume it) | `--rules` | `local_direct`, `local_diag_rflo`, `local_exact_rowlocal`, `bptt` |
| Input embedding update | `--input-mode` | `match` (default), `exact`, `three_factor`, `diag_mtrace`, `paired` |

Readout gradients are always exact. `--mp-type mult` (default) uses
`W_eff = W*(1+M)`; `--mp-type add` uses `W_eff = W+M`.

## Layout

- `core/` — shared library: `mpn.py` (MultiPlasticNet, DeepMultiPlasticNet),
  `rnn.py` (LeakyRNN), `gru.py` (GRU with BPTT and a GRU-specific RFLO),
  `mpn_tasks.py`, `net_helpers.py`, `helper.py`, `run_logging.py`.
- `scripts/training/` — `train_mpn.py`, `train_rnn.py`, `train_gru.py` (lockstep
  BPTT-vs-local over N seeds) and `train_common.py` (shared train/plot machinery).
- `scripts/tasks/` — the task adapters: `tasks.py` (registry, ring tasks, sequential
  MNIST), `adding_tasks.py`, `associative_recall_tasks.py` and
  `nonlinear_system_tasks.py`.
- `notebooks/` — standalone analysis scripts (trained-network figures, gradient and
  trace diagnostics, MPN-vs-RNN curves).
- `tests/` — rule-correctness and integration tests; see [tests/README.md](tests/README.md).
- `docs/` — the detailed documentation indexed below.
- `clean.py` — deletes stale outputs (dry run by default).
- `requirements.txt` — the packages the code imports.
- Outputs (git-ignored): `figure/`, `figure_data/`, `checkpoints/`, `log/`.

## Usage

```bash
cd scripts/training
python train_mpn.py --net dmpn --task delaygo --runs 3   # deep MPN: BPTT vs local
python train_rnn.py --task delaygo --runs 3              # leaky RNN: BPTT vs RFLO
python train_mpn.py --net dmpn --task seqmnist --runs 3  # deep MPN on sequential MNIST
python train_rnn.py --task seqmnist --runs 3             # leaky RNN on sequential MNIST
python train_mpn.py --net dmpn --task adding --runs 3    # deep MPN on the adding problem
python train_rnn.py --task adding --runs 3               # leaky RNN on the adding problem
python train_gru.py --task seqmnist --runs 3             # GRU: BPTT vs GRU-RFLO, same tasks/protocol

cd ../..
python tests/run_all.py                                  # the whole test suite
```

Install the dependencies with `pip install -r requirements.txt`.

`--task` takes a ring-task name (`delaygo`, `contextdelaydm1`, …), `seqmnist` /
`seqmnist_pixel`, `adding` / `adding_L<len>_m<marks>`, `twohop` /
`twohop_K<symbols>_d<delay>`, or `wh_mid` / `wh_out` (nonlinear system
identification). `--hidden` sets the MP
stack (one width per MP layer). Every rule in a command starts from the same
initialization and sees the same data. One invocation writes one run ID,
`<model>_<task>_<YYYYMMDD_HHMMSS>_<hash>`, shared by `figure/`, `figure_data/`,
`checkpoints/` and `log/`; `python scripts/training/train_mpn.py --help` lists every flag.

### One active input per MP layer

With `--mp-input-routing top1_abs`, each MP layer selects the largest absolute
input per sample and time step, after any MP-input RMS normalization. It keeps
the signed value, chooses the first index on ties, and uses the same vector for
the forward read, Hebbian write and eligibility. Residual skips carry the full
stream. The default is `--mp-input-routing dense`.

This provides a control for temporal credit assignment in deep MPNs. Starting
from zero sensitivity, a Hebbian write can update only the winning diagonal
entry of each row's weight trace. With the RFLO gain cap disabled (default),
`local_diag_rflo` therefore gives the same **within-layer MP weight eligibility**
as `local_exact_rowlocal`, using one trace value per synapse. Use
`--local-bias-mode exact` to match their bias updates too. Lower-layer gradients
can still differ from BPTT through the plastic states of higher layers.

```bash
python scripts/training/train_mpn.py --net dmpn --task wh_mid --hidden 16 16 \
  --mp-input-routing top1_abs --local-bias-mode exact --learning-signal exact_spatial \
  --rules local_diag_rflo local_exact_rowlocal bptt --cross-layer-steps 0 --seed 27
```

## Documentation

| Topic | File |
|---|---|
| Tasks, models, batch size, output layout, console log, cleaning, adding a task | [docs/training_and_outputs.md](docs/training_and_outputs.md) |
| `--learning-signal`, defaults, `--local-bias-mode`, supported combinations | [docs/learning_signals.md](docs/learning_signals.md) |
| Local readout heads, `--head-lr-mult`, `--lr-schedule` | [docs/local_readout_heads.md](docs/local_readout_heads.md) |
| `--input-mode` (`diag_mtrace`, `paired`) | [docs/input_modes.md](docs/input_modes.md) |
| Input-trace derivation and limitations | [docs/input_modulation_trace.md](docs/input_modulation_trace.md) |
| `--mp-type`, modulation bounds, `--lam`, `--eta`, `--rflo-trace-rho`, `--residual-scale` | [docs/plasticity_settings.md](docs/plasticity_settings.md) |
| Additive MPN equations and tests | [docs/additive_mpn.md](docs/additive_mpn.md) |
| `--mp-input-norm rms`: equations, trace-gain measurements | [docs/mp_input_norm.md](docs/mp_input_norm.md) |
| Analysis and diagnostic scripts in `notebooks/` | [docs/analysis_scripts.md](docs/analysis_scripts.md) |
