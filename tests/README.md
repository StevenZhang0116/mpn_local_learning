# tests

Test / sanity-check / validation code for `mpn_local_learning`. Each file imports
`_bootstrap` first (a copy lives here, same as `scripts/` and `notebooks/`), which
adds `../core` and `../scripts` to the path so `mpn`, `rnn`, `tasks`, `train_mpn`,
etc. import cleanly. Run from this directory.

| File | What it checks |
|---|---|
| `validate_local_learning.py` | Correctness of the local-learning rules vs BPTT (Tiers 1-8 + a real-task integration check). float64, tight tolerances. |
| `test_tasks.py` | The task adapters in `scripts/tasks.py`: seq-MNIST data loading / batching / masking / metric / determinism, the registry, and an end-to-end "loss decreases" integration check for both MPN and RNN. |

```bash
cd tests
python validate_local_learning.py   # rule correctness (all tiers)
python test_tasks.py                 # task adapters + integration
```

Both print per-check `[PASS]/[FAIL]` lines and exit non-zero if anything fails.
