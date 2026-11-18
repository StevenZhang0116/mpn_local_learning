# MP-input RMS normalization (`--mp-input-norm rms`)

`net_params['mp_input_norm'] = 'rms'` (CLI `--mp-input-norm rms`) rescales the
presynaptic vector that every MP layer consumes (layers `1..L-1` only with
`mp_input_norm_skip_first`, see below), at every time step. It is
parameter-free and stateless, off by default (`'none'`), and recorded in checkpoints
and `config.json`. `mp_input_norm_eps` (CLI `--mp-input-norm-eps`, default `1e-5`)
is the constant inside the square root. There is no learnable gain and no buffer,
so checkpoints and `state_dict`s load across the two settings and `none` reproduces
the previous computation byte for byte. The setting appears in the run metadata and
the console/figure notes and is restored on reload. It is independent of
`--input-normalize`, which standardizes the raw input once with fixed statistics.

## Definition and placement

Let `h[n]_t` be the residual stream feeding MP layer `n` at step `t` (`h[0]` is the
embedding output or the raw input) and `d_n` its width. The layer consumes

```text
r[n]_t     = sqrt( mean_J (h[n]_{J,t})^2 + eps )          (B, 1)
x_hat[n]_t = h[n]_t / r[n]_t                               ||x_hat||^2 ≈ d_n
```

and everything that belongs to layer `n` uses `x_hat[n]_t`:

```text
z[n]_t      = b[n] + W_eff[n]_t x_hat[n]_t,     W_eff = W(1+M_{t-1})  (mult) | W+M_{t-1}  (add)
a[n]_t      = phi(z[n]_t)
M[n]_t      = lam M[n]_{t-1} + eta a[n]_t x_hat[n]_t^T    (then bounds / masks as before)
h[n+1]_t    = h[n]_t + scale * a[n]_t                      (residual on: the SKIP carries h, not x_hat)
            = a[n]_t                                       (residual off)
```

This is the pre-norm placement: only the branch entering the MP block is
normalized; the identity path, the readout and the local readout heads see the
un-normalized stream. Hard bounds, `scaled_tanh` writes, update masks and frozen
plastic entries are unchanged (they act on the write, not on its input).

The single-layer net (`--net mpn1`) has nothing trainable below its MP layer, so
the norm is simply applied to the (standardized) input sequence once; the layer,
its write and its traces all consume `x_hat`.

## Why

For a row `i`, substituting the exact eligibility `E` into the trace recursion for
`P^I_{iJ,t} = dM_{iJ,t}/dW_{iI}` gives a linear time-varying system whose transition
matrix is `lam*I + eta*phi'(z_i) x x^T diag(W_i)`-like, with the single non-trivial
eigenvalue

```text
mu_i,t = lam + eta * phi'(z_i,t) * sum_J W_iJ x_J,t^2        (exact row-local, BPTT; mult)
mu_i,t = lam + eta * phi'(z_i,t) * sum_J x_J,t^2              (additive MPN, W_eff = W + M)
```

and the diagonal RFLO trace `A_iI` has the scalar gain

```text
g_iI,t = lam + eta * phi'(z_i,t) * W_iI x_I,t^2              (diag RFLO)
```

Traces stay bounded only while these stay inside `(-1, 1)` along the trajectory.
`sum_J W_iJ x_J^2` scales with the fan-in, with `||x_t||^2` at that step and, under
`--residual`, with the growing norm of the residual stream, so the stability
condition on `eta` is not scale-free: the value that fits one layer, task or depth
need not fit another. For the MULTIPLICATIVE MPN, `||x_hat||^2 = d` and zero-mean
Xavier-scaled `W` make the row sum `O(1)` in every layer and at every step, and
`eta/(1-lam)` becomes the single dimensionless knob. The additive MPN has no `W` in
its row sum: its gain is `lam + eta*phi'*||x_hat||^2 = lam + eta*phi'*d`, which the
RMS norm pins at the fan-in, so the same `eta` is far less stable there (at
`eta=0.003, lam=0.99`, 3x128 `contextdelaydm1`: ~0.1% of unit-steps above 1 for
`mult` versus 5-6% for `add`); with `add`, scale `eta` by `1/d`, or normalize to
unit L2 norm. Measured on `contextdelaydm1` (`--hidden 64 64 64`, mult), `eta ≈
0.3*(1-lam)` keeps all three layers' traces bounded without `--rflo-trace-rho`;
see the measurements below; a no-norm control at the same `eta` is bounded and aligned
as well, so on that task the norm contributes scale invariance rather than a lower
gain. The row sum still depends on the weights, so it is a
trajectory-dependent condition, not a fixed hyperparameter range.

Normalization does not remove the trade-off between plasticity strength and
stability — the modulation's contribution to a row's preactivation,
`sum_J W_iJ M_iJ x_J ≈ (eta/(1-lam)) a_i sum_J W_iJ x_J^2`, is governed by the
same loop gain — it makes that gain uniform across layers, rows and time.

## Measurements

On `contextdelaydm1` with `--hidden 64 64 64`
(`notebooks/rflo_trace_gain.py --mp-input-norm rms --eta ETA --lam 0.99`, batch 8):

| setting | row gain > 1 (per step, layers 1–3) | diagonal gain > 1 | max trace at the end | cos vs BPTT (exact / diag) |
|---|---|---|---|---|
| default `eta=1, lam=0.99`, no norm | 47–52 % | 10–17 % | `A` up to 1.6e5 | −0.5…1.0 / 0.03–0.06 |
| **control:** `eta=0.003, lam=0.99`, no norm (same seed/batch) | 0 % | 0 % | 0.06–0.12 | 1.00 / 1.00 |
| `rms, eta=0.01` (`eta/(1-lam)=1`) | 7–10 % | 0 % | 2–10 | 0.88–1.0 / 0.77–0.94 |
| `rms, eta=0.005` (0.5) | 0.5–3 % | 0 % | 0.9–3 | 0.97–1.0 / 0.93–0.98 |
| `rms, eta=0.003` (0.3) | 0–0.2 % | 0 % | 0.5–1.5 | 0.99–1.0 / 0.98–0.99 |
| `rms, eta=0.015, lam=0.95` (0.3) | 0–0.2 % | 0 % | 0.8–3 | 0.94–1.0 / 0.89–0.97 |

The control row (added after the merge, from the same diagnostic with the same seed
and batch) shows that on this task the whole improvement over the default row comes
from the smaller `eta`, not from the norm: without the norm the stream has
`||x||**2 < d` (tanh activities plus the embedding), so at the same `eta` its loop
gain is lower still and its traces smaller. What the norm buys is scale invariance —
the same `eta` gives the same loop gain whatever the fan-in, activity level or
residual-stream depth — which a single-task, no-residual diagnostic does not
exercise; whether that matters for training has to be settled by paired runs
(`--seed`, same `--eta`, norm on/off).

So with the norm on (multiplicative MPN), `eta ≈ 0.3*(1-lambda)` keeps every layer's traces bounded
without `--rflo-trace-rho` (the capped and uncapped diagonal variants then
coincide), while the modulation still reaches `O(0.1)` per synapse. The condition
depends on the weights, so a run that grows `W` substantially may re-enter the
unstable regime; `rflo_trace_gain.py` reports the exceedance fractions for a
saved configuration. Trace boundedness is not a convergence guarantee.

## Jacobian

The norm is a map from the stream to the layer input with

```text
J_t      = (I - x_hat x_hat^T / d) / r_t
J_t^T g  = ( g - x_hat * mean_J(g_J x_hat_J) ) / r_t          (`_mp_input_norm_backward`)
```

Any learning signal that is formed at `x_hat[n]` and must be delivered to the
stream `h[n]` (and so to the layers and the embedding below) is mapped through
`J_t^T` **before** the identity-skip term `ell_h[n+1]` is added, because the skip
lives on the un-normalized stream:

```text
ell_h[n] = J[n]_t^T ( W_eff[n]_t^T (scale * ell_h[n+1] ⊙ phi'(z[n]_t)) ) + skip[n] * ell_h[n+1]
```

## What changes per learning mode

| Mode | Change | Reason |
|---|---|---|
| `bptt` | forward only (`_forward_stack`) | autograd differentiates through the norm |
| own-layer eligibility of `local_exact_rowlocal` / `local_diag_rflo` / `local_direct` | input is `x_hat[n]` | `x_hat[n]` does not depend on layer `n`'s parameters; the recursions are unchanged in form, so single-layer exactness, top-layer exactness under `exact_spatial`, and the `T=2` exactness of `--cross-layer-steps 1` all carry over |
| `exact_spatial` | `J^T` after each `W_eff^T` backprojection, before the skip term; also for the embedding signal `ell_h[0]` | exact same-time spatial gradient of the network actually run |
| `layerwise_fa` | `J^T` after each `B_inter` projection | the norm is parameter-free, not weight transport; standard FA through a norm layer keeps its Jacobian |
| `dfa` (`direct_fa`) | none | fixed projections onto the stream |
| `local_readout` / `mixed` heads | none | heads read the un-normalized stream `h[n+1]` |
| `--cross-layer-steps 1` | the write's PRE factor is `x_hat[m]_t`, the POST bracket uses `x_hat[m]_{t-1}`; each source `v[m]` and each `W_eff[q]_{t-1}` step of the downward sweep is followed by `J_{t-1}^T` (before the skip term) | mirrors the same-time pass one step back; verified exact at `T=2` |
| input mode `three_factor` (and `match` for local rules) | inherits `ell_h[0]` | — |
| input mode `exact` | none | BPTT splice |
| input mode `diag_mtrace` / `paired` | **rejected**, unless `mp_input_norm_skip_first` (layer 0 then consumes the raw embedding output) | its one-column-per-embedding-row trace assumes embedding row `j` feeds only column `j` of layer 0's input; the shared normalizer breaks that |
| hard bounds, `scaled_tanh`, masks, frozen states | none | gates act on the write |

## Skipping the first layer (`mp_input_norm_skip_first`)

`net_params['mp_input_norm_skip_first'] = True` (CLI `--mp-input-norm-skip-first`)
leaves MP layer 0's input un-normalized; layers `1..L-1` are normalized as above.
Layer 0 consumes `h[0]` — the embedding output, or the raw input for the single-layer
net — and a per-step rescaling of that vector erases its amplitude: in the adding
task the input `(v, 0)` becomes `(±sqrt 2, 0)` for every value `v`; with a trainable
embedding the amplitude survives only relative to the embedding bias. The upper
layers' inputs are activities whose scale carries no task information, which is
where the loop-gain argument applies.

Implementation: `_mp_input_norm_forward(x, layer_idx)` returns `(x, None)` for
`layer_idx == 0`, so layer 0's entry in `norm` is `(h[0], None)` and
`_mp_input_norm_backward` is the identity there; no other code path changes.
Consequences: every exactness statement above holds unchanged (re-tested with the
flag on); `diag_mtrace`/`paired` are accepted again, since layer 0's input is the raw
embedding output; a net with a single MP layer rejects the flag (nothing would be
normalized); with the norm off the flag is ignored by the model, and the CLI rejects
it without `--mp-input-norm rms`. Recorded in `net_params`,
`config.json`, checkpoints and the console/figure notes; the legacy
tag gains `-skip0`.

## Implementation map

* `mpn.MultiPlasticNetBase`: config parsing (incl. `mp_input_norm_skip_first`),
  `_mp_input_norm_forward(x, layer_idx)`,
  `_mp_input_norm_backward`, `_mp_input_norm_seq`.
* `mpn.MultiPlasticNet`: `forward`, `network_step`, `bptt_gradients`,
  `_prepass_output_grad`, `_local_sequence_gradients` consume `x_hat`.
* `mpn.DeepMultiPlasticNet`: `_forward_stack` (returns the consumed inputs),
  `_forward_local_stack(return_norm=True)`, `_same_time_boundary_signals(norm=...)`,
  `_cross_layer_correction(..., x_in, prev_xin, prev_norm=...)`, the local loop
  and the custom-loss pre-pass.
* `scripts/train_mpn.py`, `scripts/train_common.py`: CLI, `net_params`, metadata,
  notes. `notebooks/rflo_trace_gain.py`: `--mp-input-norm / --mp-input-norm-skip-first /
  --eta / --lam`.
* `tests/test_mp_input_norm.py`: see `tests/README.md`.
