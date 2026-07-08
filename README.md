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
  `rnn.py` (LeakyRNN), `mpn_tasks.py`, `net_helpers.py`, `helper.py`.
- `scripts/` — `train_mpn.py`, `train_rnn.py` (lockstep BPTT-vs-local over N seeds),
  `train_common.py` (shared train/plot machinery), `validate_local_learning.py`
  (correctness checks), `_bootstrap.py` (path setup).
- `notebooks/` — `visualize_performance.ipynb`, `visualize_trained_networks.ipynb`,
  `compare_mpn_rnn_performance.ipynb` (overlays an MPN and an RNN run on one figure).
- `utils/` — `clean.py` (remove saved outputs, safe by default).
- Outputs (git-ignored): `figure/` (PNGs), `figure_data/` (`.npz` behind each figure),
  `checkpoints/` (`.pt` trained nets).

## Usage

```bash
cd scripts
python train_mpn.py --net dmpn --task delaygo --runs 3   # deep MPN: BPTT vs local
python train_rnn.py --task delaygo --runs 3              # leaky RNN: BPTT vs RFLO
python validate_local_learning.py                        # correctness checks
```

Each run writes a two-panel (train / test angle-accuracy) figure to `figure/`, the
arrays behind it to `figure_data/`, and (MPN) trained nets to `checkpoints/`. Reload
figures without retraining via `replot_from_npz(data_path())`; explore results with
the notebooks.
