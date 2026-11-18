# Choosing the learning signal (`--learning-signal`)

`train_mpn.py` separates three independent choices so that changing one flag changes
one thing:

| Axis | Flag | Choices |
|---|---|---|
| Learning signal (where each MP layer's error comes from) | `--learning-signal` | `exact_spatial` (default), `layerwise_fa`, `dfa`, `local_readout`, `mixed` |
| Eligibility (how MP parameters consume it) | `--rules` | `local_direct`, `local_diag_rflo`, `local_exact_rowlocal`, `bptt` |
| Input embedding update | `--input-mode` | `match` (default), `exact`, `three_factor`, `diag_mtrace`, `paired` |

## Signal modes

The five signal modes map onto the model's two fields (`learning_signal`, `feedback_mode`):

| `--learning-signal` | Signal each MP layer learns from | internal `learning_signal` | internal `feedback_mode` |
|---|---|---|---|
| `exact_spatial` | main readout error through the true weights at every boundary (same-time spatial path; weight transport) | `global` | `exact_spatial` |
| `layerwise_fa` | main readout error through a fixed random matrix at every boundary | `global` | `layerwise_fa` |
| `dfa` | main readout error projected directly to every layer through its own random matrix | `global` | `direct_fa` |
| `local_readout` | each module's own auxiliary head error (see [local_readout_heads.md](local_readout_heads.md)) | `local_readout` | `exact_spatial` |
| `mixed` | `exact_spatial` plus `--local-signal-alpha` × the local head error | `mixed` | `exact_spatial` |

`exact_spatial` qualifies the spatial pathway only; temporal credit is whatever the
eligibility rule provides, so none of these is full BPTT. `bptt`'s MP layers and
readout are full BPTT of the main loss under every signal mode (its auxiliary heads, if
any, receive no gradient), and so is its embedding except under `--input-mode
three_factor`, which splices the direct three-factor rule into the embedding under the
global signal through the selected feedback pathway. That combination is a hybrid, not a
full-BPTT baseline; the per-rule line below labels it `algorithm=hybrid`.

## Defaults

Defaults are signal-independent, so switching `--learning-signal` alone changes only the
signal. They changed in October 2026; old runs used the previous column:

| Setting | Previous default | Current default |
|---|---|---|
| `--rules` | `bptt local_exact_rowlocal local_diag_rflo local_direct` | `bptt local_direct local_diag_rflo` |
| `--input-mode` | `exact` (local runs spliced a BPTT embedding gradient: a hybrid) | `match` (bptt = full BPTT, local runs fully local) |
| `--local-bias-mode` | `exact` | `match` (each rule's own bias update: direct for diagonal RFLO, exact for row-local; pass `direct` or `exact` to force one shared policy) |
| `--cross-layer-steps` | `1` | `0` |
| `--grad-align` | on | on (unchanged) |

## MP bias updates (`--local-bias-mode`)

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

## Supported combinations

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
recorded next to `feedback_mode`/`learning_signal` in checkpoints, `config.json` and the
`.npz` metadata. Legacy spellings still work and are translated: `--feedback
exact_spatial|layerwise_fa|direct_fa` (= `--learning-signal exact_spatial|layerwise_fa|dfa`),
`--learning-signal global` (defers to `--feedback`), and the `--dfa` preset, which keeps
its historical bundle (`dfa` + `match` + direct bias + rules `bptt local_exact_rowlocal
local_diag_rflo` + alignment off). A flag that contradicts another is rejected whatever
the argument order. `train_rnn.py` accepts the same three global names. The parser runs
in three stages (`parse_arguments`, `resolve_defaults_and_legacy_options`,
`validate_config`), and `--help` groups flags by what they act on.
