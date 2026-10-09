# Additive MPN dynamics and local gradients

`mp_type='add'` (CLI `--mp-type add`) selects, for each layer,

\[
z_{i,t}=\sum_j(W_{ij}+M_{ij,t-1})x_{j,t}+b_i,\quad a_{i,t}=\phi(z_{i,t}),
\qquad M_t=\mathcal C(\lambda\odot M_{t-1}+\eta\odot a_tx_t^T).
\]

Here `x` is the layer's current input, not necessarily the raw network input.
With a residual, the output stream is `x + residual_scale*a`; writes still use
raw `a`. The single-layer and deep networks share the same MP-layer implementation.
This is additive **synaptic** modulation, independently of activation residuals.
It implements the additive mechanism used by
[Tyulmankov, Yang & Abbott (2022)](https://doi.org/10.1016/j.neuron.2021.11.009),
while retaining this repository's activation, task and training choices.

## Derivatives

Let `D = dW_eff/dW` holding M fixed, and `S = dW_eff/dM` holding W fixed:

| Mode | Effective weight | D | S |
|---|---|---|---|
| mult (default) | W*(1+M) | 1+M | W |
| add | W+M | 1 | 1 |

For fixed input trajectories to a layer, define
`P[b,i,I,J] = dM[b,i,J]/dW[i,I]`, `Q[b,i,J] = dM[b,i,J]/db[i]`,
and diagonal approximation `A[b,i,I] ~= P[b,i,I,I]`. All traces below on the
right-hand side are from t-1. In additive mode:

\[
E^{\rm direct}_{iI,t}=\phi'_{i,t}x_{I,t},\qquad
\widehat E^{\rm diag}_{iI,t}=\phi'_{i,t}x_{I,t}(1+A_{iI,t-1}),
\]
\[
E^{\rm row}_{iI,t}=\phi'_{i,t}
\left[x_{I,t}+\sum_Jx_{J,t}P^I_{iJ,t-1}\right],\qquad
R^{\rm exact}_{i,t}=\phi'_{i,t}
\left[1+\sum_Jx_{J,t}Q_{iJ,t-1}\right].
\]

Direct bias uses `R = phi_prime`; `local-bias-mode match` keeps direct bias for
direct/diagonal rules and exact Q for row-local. BPTT differentiates biases by
autograd, independently of this setting. Unbounded associative writes advance

\[
P^I_{iJ,t}=\lambda_{iJ}P^I_{iJ,t-1}+\eta_{iJ}x_{J,t}E_{iI,t},\quad
A_{iI,t}=\lambda_{iI}A_{iI,t-1}+\eta_{iI}x_{I,t}\widehat E_{iI,t},\quad
Q_{iJ,t}=\lambda_{iJ}Q_{iJ,t-1}+\eta_{iJ}x_{J,t}R_{i,t}.
\]

With `rflo_trace_rho`, let `k = eta*phi_prime*x**2`. The additive candidate is
`A_new = clip(lambda+k, -rho, rho)*A_old+k`. This caps only the coefficient of
old A, not the drive, the forward memory or the bias/input traces. It is an
additional approximation and does not make the complete Jacobian contractive.
For pre-only writes, own-layer W/b traces remain zero (the postsynaptic factor
is constant); input-embedding modulation traces still have a presynaptic source.

Both fused and explicit paths apply the existing write derivatives, graded
update masks and frozen-state gates. Hard writes clip after masked mixing;
scaled-tanh writes mix the activated candidate with old M. Frozen entries have
zero sensitivity. Full row-local eligibility is exact for an isolated layer
with fixed inputs and matching bias policy; it is not full deep-network BPTT.

## Spatial, input and cross-layer paths

- Same-time exact spatial feedback uses `W+M`.
- `diag_mtrace` uses `(W+M)*D_embed + x*P_input` inside the activation derivative.
  The embedding's direct and exact/BPTT policies retain their existing meanings.
- `cross_layer_steps=1` reads a previous upper-layer M-write with sensitivity
  `x`, replacing multiplicative `W*x`. The earlier spatial transport still uses
  the effective weight, now `W+M`. This applies to both the pre- and postsynaptic
  sources of a write and to both W and bias corrections.
- Cross-layer correction keeps the existing constraints: exact spatial signal,
  no local/mixed heads, MP parameters only. It restores full MP gradients at
  T=2 with exact row-local W/b eligibility; longer trajectories remain approximate.
- Local readout/mixed and random-feedback modes keep their existing definitions.
  BPTT ignores auxiliary heads and uses the main loss. `input-mode three_factor`
  remains an explicit hybrid even with BPTT; `match` keeps full BPTT.

## Bounds, defaults and persistence

The CLI default remains multiplicative. Both modes support no bounds, hard
bounds, and scaled-tanh writes. New additive hard bounds are fixed absolute M
bounds. This replaces the previously incomplete W-relative additive bounds,
which rejected trainable W; no weight-sign guarantee is made. Bounds and the
smooth scale have weight units for add, and are dimensionless for mult.
Compare memory-to-static-drive scales, rather than assuming equal numeric
bounds give equal modulation strength. No automatic retuning of eta is applied.

The model type is saved in `net_params.ml_params.mp_type`; old configurations
without it remain multiplicative. Existing saved `M_bounds` buffers load from
state_dict as usual. Diagnostic CSV fields with legacy `wa_*` names measure the
actual eligibility correction: W*A for mult, A for add.

## Validation

From the repository root:

```bash
OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m unittest discover -s tests -p test_additive_mpn.py -v
```

The float64 tests use independent forward/write graphs. Diagonal RFLO is checked
by detaching all modulation columns except the parameter's own column; direct
learning detaches every M history. Row-local uses an isolated layer graph with
fixed inputs and the same learning signal. Tests also cover full BPTT, scalar
capped recurrence, fused/explicit parity, exact bias, pre-only writes, residuals,
heterogeneous widths, masks/freezes, local and mixed heads, custom loss,
input-column traces, T=2 cross-layer exactness, configuration and checkpoint
round-trips, and lockstep optimizer steps for all four rules.
