# Input embedding updates (`--input-mode`)

`--input-mode` selects how the trainable input embedding of a `dmpn` network is
updated, independently of the MP layers' eligibility rule (`--rules`):

| `--input-mode` | Embedding update under `bptt` | Embedding update under a local rule |
|---|---|---|
| `match` (default) | full BPTT | direct three-factor rule |
| `exact` | full BPTT | full-BPTT embedding gradient spliced into the local run (a hybrid) |
| `three_factor` | direct three-factor rule (a hybrid) | direct three-factor rule |
| `diag_mtrace` | full BPTT | first-MP-layer modulation-column trace |
| `paired` | full BPTT | `diag_mtrace` for `local_diag_rflo`, three-factor for `local_direct` |

The combinations each learning signal accepts are listed in
[learning_signals.md](learning_signals.md#supported-combinations).

## Modulation-column trace (`diag_mtrace`)

For temporal input-layer credit without an extra BPTT pass, select
`--input-mode diag_mtrace` (exact_spatial pathway) on a `dmpn` run. Local rules
then track the first MP layer's modulation-column sensitivities to the matching
embedding weights and bias; BPTT remains a full-gradient reference. This is a
diagonal-column approximation: other modulation columns and deeper-layer temporal
paths are omitted. MP parameter updates retain their selected learning rule.
It supports unbounded, hard-clipped, and scaled-tanh modulation, residuals,
input normalization, and independently frozen input weights/biases. Trace memory
scales as batch × first MP width × embedding width × (raw input width + bias),
independently of sequence length. See [the derivation and limitations](input_modulation_trace.md).

For example, change `--input-mode match` to `--input-mode diag_mtrace` in a local
learning comparison. Keep `--cross-layer-steps 0` for the original MP update rules;
`1` adds its separate correction to MP parameters only. Metadata records
`input_mode="diag_mtrace"`, and checkpoint loading restores this input mode. Existing
`match`, `exact`, and `three_factor` modes retain their behavior.

## Paired comparison (`paired`)

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
