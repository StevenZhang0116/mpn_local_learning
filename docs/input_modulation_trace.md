# Input-layer modulation sensitivities

`--input-mode diag_mtrace` adds a forward-mode sensitivity trace for input
embedding weights and biases during local learning. It requires `dmpn` and
`exact_spatial` feedback. It leaves the `bptt` rule unchanged, so one command can
compare local input traces against full BPTT. It is an experimental approximation;
improved training accuracy or gradient alignment is not guaranteed.

## Retained paths

Let `U[j,k]` be an input embedding weight, `u_t` the raw input after optional fixed
standardization, and `x_t = phi(U u_t + b_in)` the embedding activity. For the
first MP layer, let `W[i,j]`, `M[i,j]`, and `a[i]` denote its base weight,
modulation, and block activation. All expressions below suppress the batch index.

The stored trace is

```text
P[i,j,k] ≈ dM_first[i,j] / dU[j,k].
```

For each embedding row `j`, this retains its influence on column `j` of the first
MP layer's modulation matrix. It drops derivatives of other columns `J != j`
with respect to `U[j,k]`, even though those derivatives generally exist for
associative Hebbian writes. It also drops temporal derivatives through every
deeper MP layer's modulation. Current-time spatial credit still propagates
through the entire stack using the current modulated weights.

This is an RFLO-inspired diagonal approximation, not the original random-feedback
RFLO algorithm: `exact_spatial` uses forward weights for its learning signals,
and the trace accesses the first MP layer across the embedding boundary.

## Recurrence and input gradient

At time `t`, first run the forward pass with `M_(t-1)` and compute the same-time
learning signal `ell_first` at the first MP layer's output boundary. Using the
previous input trace `P_(t-1)`, compute

```text
D[j,k] = phi_embed'(pre_embed[j]) * u_t[k]
C[i,j,k] = phi_first'(pre_first[i]) * W[i,j] *
           ((1 + M_(t-1)[i,j]) * D[j,k] + x_t[j] * P_(t-1)[i,j,k])
g_U[j,k] += sum_batch,i ell_first[i] * C[i,j,k]
P_raw[i,j,k] = lambda[i,j] * P_(t-1)[i,j,k]
               + eta[i,j] * (a_t[i] * D[j,k] + x_t[j] * C[i,j,k])
```

These expressions use multiplicative weights. With `--mp-type add`, replace
only the expression for `C` by

```text
C[i,j,k] = phi_first'(pre_first[i]) *
           ((W[i,j] + M_(t-1)[i,j]) * D[j,k] + x_t[j] * P_(t-1)[i,j,k])
```

The input drive uses the effective weight `W+M`; the modulation sensitivity
has coefficient 1. The write, masks, residual handling and approximation
boundaries are unchanged. See [additive MPN](additive_mpn.md).

For an active first-layer identity residual with `--residual-scale alpha`,
multiply the `ell_first * C` gradient contribution by `alpha`, then add
`sum_batch ell_first[j] * D[j,k]` for the unscaled identity path. Keep `C` and
`P_raw` unchanged: the modulation update writes raw block activation `a_t`,
not the scaled residual increment or the residual stream. Downstream scaled
branch and identity contributions are already included in `ell_first`.
The default `alpha=1` recovers the original residual rule; without a residual,
use the gradient above with no branch gain or identity term.

For `hebb_pre`, the postsynaptic factor is the constant `1/sqrt(first_MP_width)`.
Use that constant in place of `a_t` and omit `x_t[j] * C[i,j,k]` from the trace
source. Unlike the MP weight trace, the input trace is nonzero for `hebb_pre`:
its modulation still depends on the embedding activity.

Input biases use the same recurrence with `u_t[k] = 1`. Only trainable input
tensors allocate features: weight-only uses `raw_input_width` features,
bias-only uses one, and jointly trainable weights/biases use their sum.

After writing the first layer's actual modulation, finalize the trace with the
same derivative and masking convention:

- Linear/unbounded: `P_t = r * P_raw + (1-r) * P_(t-1)`.
- Hard clipping: apply the clipping derivative **after** this mask mixture,
  evaluated at the actual masked pre-clipping write. The derivative at each
  clipping endpoint is one, matching the production forward convention.
- Scaled tanh: apply `1 - tanh(M_raw / bound)^2` to `P_raw` **before** mixing it
  with `P_(t-1)` using the update mask `r`.
- Frozen modulation entries have zero sensitivity, including when `r = 0`.

The update mask can be binary or graded. It controls state writes and is distinct
from the loss/cost mask. The input trace is reset to zero for each sequence batch;
parameters stay fixed throughout that unroll. The accumulated gradients are
written to `.grad` and passed to the existing optimizer. No extra BPTT pass is
performed for local rules.

## Scope and costs

`--input-mode paired` selects this input trace only for `local_diag_rflo`.
It uses direct three-factor input updates for `local_direct` and full BPTT for
`bptt`. The preset requires a deep MPN with an embedding and `exact_spatial`
feedback, and rejects `local_exact_rowlocal`. The explicit `diag_mtrace` mode
still selects the input trace for all local rules, and `match` remains unchanged.

- Only input-layer gradients change. MP weights/biases and readout gradients
  continue to use the selected learning rule.
- `--local-bias-mode` controls MP biases; input biases follow `diag_mtrace`.
- `--cross-layer-steps 1` adds the existing MP-parameter correction independently;
  it does not restore missing deeper temporal paths for input parameters.
- Forward activations and losses are unchanged at fixed parameters. Subsequent
  optimizer updates can of course change the learned trajectory.
- The persistent trace uses `B * H_first * E_embed * K` numbers; temporary
  eligibility/source tensors add further memory of the same order. For batch
  128, first-layer/embedding widths 128, six raw inputs and a bias, one trace
  requires 56 MiB in float32 or 112 MiB in float64. There is no sequence-length
  factor in this added trace storage; total training memory includes other buffers.
- Trace tensors are temporary sequence state, not checkpoint entries. The input
  mode is stored in `net_params` and the run's config JSON as `input_mode`.

## Validation

The independent autograd oracle retains one embedding coordinate and its first-MP
modulation column at a time, detaching other state histories while preserving
the forward values. It verifies the declared approximation without duplicating
the production sensitivity recurrence.

Full input-gradient agreement with BPTT is expected for a single MP layer when
the embedding width is one, or for a single MP layer with pre-only Hebbian writes
at any embedding width. General deep associative networks remain approximate.
The tests also verify a stimulus-only-at-the-first-step task with loss only at
the last step: this trace recovers a nonzero temporal input-weight gradient
where the instantaneous three-factor input update is zero.

Run the small CPU checks from the project root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p test_input_modulation_trace.py -v
```
