# Plasticity and architecture settings

Forward-dynamics options of `scripts/training/train_mpn.py`. Each one changes the network
that every rule trains, so paired comparisons must share it. The per-step RMS
normalization of MP-layer inputs has its own page: [mp_input_norm.md](mp_input_norm.md).

## Additive and multiplicative MPNs (`--mp-type`)

`--mp-type mult` (default) uses `W_eff = W*(1+M)`; `--mp-type add` uses
`W_eff = W+M`. Both support `bptt`, `local_direct`, `local_diag_rflo`, and
`local_exact_rowlocal`, with `--net mpn1` or any depth of `--net dmpn`.
The choice is independent of the learning signal, residual connections,
input/bias policy, and `--cross-layer-steps` (their existing compatibility rules
still apply). BPTT remains full main-loss BPTT under `--input-mode match`.

For example, from the repository root:

```bash
python scripts/training/train_mpn.py --mp-type add --task contextdelaydm1 \
  --hidden 128 128 128 --no-residual --learning-signal local_readout \
  --rules local_direct local_diag_rflo local_exact_rowlocal bptt \
  --input-mode match --local-bias-mode match --cross-layer-steps 0 \
  --lam 0.99 --rflo-trace-rho 0.99 \
  --modulation-mode hard --modulation-bound 0.1 \
  --batch-size 128 --steps 5000 --runs 3 --lr 0.001
```

For one MP layer with an embedding, use `--hidden 128`; without an embedding,
also use `--net mpn1 --learning-signal exact_spatial`. For temporal cross-layer
correction, use `--learning-signal exact_spatial --cross-layer-steps 1`.

Additive `--modulation-bound B` bounds **absolute weight increments** in
`[-B,B]`, independently of W. It does not preserve weight signs. Multiplicative
bounds remain dimensionless gains. The example's `B=0.1` is a starting value,
not an established optimum: equal bounds/eta do not match the strength of the
two architectures. Tune the bound or smooth scale when comparing them.
The additive option changes neither the activation nor the Hebbian write rule;
eta remains at its existing setting. See [the equations and tests](additive_mpn.md).
Configuration JSON and checkpoints save the choice as `net_params.ml_params.mp_type`.

## Modulation bounds and regularization

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
clipping; logged losses, scheduling, and alignment remain task-only. The config
JSON and checkpoints store modulation bounds/activation/scale in
`net_params.ml_params` and regularization in `train_params.reg_lambda`.

## Modulation decay (`--lam`)

Use `--lam 0.9` to set the fixed modulation decay in every MP layer to 0.9 for
all training rules. The value must be finite and in `[0, 1)`. Without this flag,
the existing `m_time_scale=4000` setup gives lambda=0.99 at dt=40. An explicit
value is saved as `net_params.ml_params.lam_clamp`, replacing `m_time_scale`;
the core initializes lambda from it and derives the corresponding time constant.
Eta is set separately by `--eta` (next section). This parameter is separate from
the RFLO trace-gain cap below.

## Hebbian write rate (`--eta`)

Use `--eta 0.3` to set the fixed Hebbian write rate eta of every MP layer. It is
saved as `net_params.ml_params.eta_clamp`, which also initializes the (untrained)
eta parameter because `eta_init='eta_clamp'`; without the flag the core's 1.0
applies. One value per MP layer, e.g. `--eta 0.3 0.1 0.1` with
`--hidden 128 128 128` (bottom to top), instead writes per-layer
`net_params.ml_params<idx>` dictionaries: copies of the shared `ml_params` with
their own `eta_clamp`, indexed in the full architecture, so under `--net dmpn` MP
layer i is `ml_params<i+1>` (the input embedding occupies index 0).
`DeepMultiPlasticNet` reads a layer's own dictionary in preference to the shared
one, so checkpoints and `config.json` restore the per-layer rates unchanged; an
all-equal list collapses to the shared form. Values must be finite; `0` freezes M
at zero and a negative value is anti-Hebbian. Because the diagonal-RFLO trace gain
is `lambda + eta*phi'*x**2*W`, a smaller eta is the most direct way to keep that
gain below 1 without the `--rflo-trace-rho` cap; like `--lam`, it changes the
forward dynamics for every rule, so paired comparisons must share it. The first
seed's architecture printout shows each layer's `Eta_init`.

## RFLO trace-gain cap (`--rflo-trace-rho`)

For diagonal RFLO, `--rflo-trace-rho 0.99` optionally caps the MP-weight trace
recurrence gain. With `k = assoc * eta * phi_prime * x**2`, the candidate update is
`A_new = clip(lambda + k*W, -rho, rho)*A_old + k*(1+M)`; existing update masks,
write derivatives, and frozen-state gates are then applied in their usual order.
For additive MPNs the candidate is `A_new = clip(lambda + k, -rho, rho)*A_old + k`.
`rho` must be finite and strictly between 0 and 1. Omit the flag to retain the
original recurrence. This is an additional eligibility approximation: it limits
repeated amplification of old traces, but does not guarantee better gradient
alignment or training performance. It does not cap the drive term, forward
modulation, exact bias traces, or input-layer sensitivity traces. Direct,
row-local, and BPTT gradients do not use this cap. For an isolated comparison,
use `--input-mode match --local-bias-mode direct --cross-layer-steps 0`.

The cap is saved as `net_params.ml_params.rflo_trace_rho` in checkpoints and
`config.json`, and restored by the diagnostic script. Legacy checkpoints with
no such field use the original recurrence. Diagnostic `summary.json` records
the setting; trace CSVs include `trace_gain_clipped_fraction`, the fraction of
candidate gains outside `[-rho,rho]` before update masks/write gates (zero when
the cap is disabled).

## Residual branch scaling (`--residual-scale`)

To scale residual MP branches, add `--residual --residual-scale 0.5`.
Each equal-width block then computes `h_out = h_in + 0.5 * a`, where `a` is the block's
activation (`tanh(z)` by default, see `--activation`).
The fixed, nonnegative scale defaults to `1.0`; the identity path has unit gain.
Unequal-width blocks keep `h_out = a` and the existing disabled-skip warning.
Non-default scales require residuals to be enabled. Scale `0` is supported as an
identity-limit diagnostic, not a suggested training setting.

The Hebbian write uses raw `a`, not the scaled residual increment. Local MP weight
and bias gradients receive the branch-scaled learning signal, while the raw
eligibility-trace recurrences stay unchanged. Spatial Jacobians, input traces,
and the optional one-hop cross-layer correction include the corresponding branch
gain. BPTT differentiates the same forward dynamics exactly. Heads still read
`h_out`; their gradients are not separately scaled. This forward hyperparameter
applies to every learning rule and is distinct from `--local-signal-alpha`
(mixed learning signals). Checkpoints, JSON/NPZ metadata and plot labels
record it; old checkpoints without the field use `1.0`.
