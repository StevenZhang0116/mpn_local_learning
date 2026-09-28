# Deep MPN: exact row-local + DFA and diagonal + DFA

This revision reuses the existing `direct_fa` pathway and the existing weight
eligibility implementations. It adds configurable bias eligibility and a DFA
training preset that matches the manuscript's assumptions.

## Run

From the extracted project root:

```bash
python scripts/train_mpn.py --dfa --hidden 64 64 --task delaygo --steps 500 --runs 1
```

This compares BPTT, exact row-local + DFA, and diagonal + DFA from the same
initial network, including the same fixed feedback matrices. Both local rules
use **direct bias updates** by default so their comparison isolates weight
eligibility. BPTT still uses exact gradients for all trainable parameters.

To run only the two local variants (no BPTT baseline):

```bash
python scripts/train_mpn.py --dfa --hidden 64 64 \
  --rules local_exact_rowlocal local_diag_rflo --task delaygo --steps 500
```

For the manuscript's exact row-local eligibility **including exact bias traces**:

```bash
python scripts/train_mpn.py --dfa --hidden 64 64 \
  --rules local_exact_rowlocal --local-bias-mode exact --task delaygo --steps 500
```

For the diagonal variant with direct bias:

```bash
python scripts/train_mpn.py --dfa --hidden 64 64 \
  --rules local_diag_rflo --local-bias-mode direct --task delaygo --steps 500
```

`--no-residual` disables residual connections. The runner enables identity
residuals for compatible widths at every depth, including a single MP layer.
Residuals are enabled by default; use `--no-residual` to reproduce the older
single-layer behavior, which previously disabled the skip regardless of the flag.
Start with small widths: exact row-local traces scale quadratically with fan-in.

The runner requires the project's usual dependencies: PyTorch, NumPy, SciPy,
Matplotlib, seaborn, and six. Optional task-specific dependencies and W&B are
only needed for the corresponding existing features. W&B remains opt-in.

## What the preset does

| Setting | `--dfa` value | Why |
| --- | --- | --- |
| `feedback_mode` | `direct_fa` | Fixed output-error projection to every activity boundary |
| `input_mode` | `match` | Local runs use direct DFA input updates; BPTT baseline stays exact |
| `cross_layer_steps` | `0` | No forward-weight-based temporal correction |
| `ml_params.local_bias_mode` | `direct` unless explicitly overridden | No row-wide bias trace; fair weight-trace comparison |
| `log_grad_align` | `False` unless `--grad-align` is passed | Prevent additional BPTT diagnostic passes |
| selected rules | BPTT, exact row-local, diagonal RFLO | Compare the requested algorithms with an exact baseline |

The original runner defaults were `input_mode="exact"` and
`cross_layer_steps=1`. Merely passing `--feedback direct_fa` was therefore not
enough to instantiate these algorithms. The preset rejects conflicting input,
feedback, or correction settings. At model construction, `direct_fa` with a
nonzero `cross_layer_steps` is also rejected. Without `--dfa`, the existing
non-DFA defaults are preserved.

`--grad-align` explicitly enables gradient comparisons against BPTT; these are
diagnostics, not the gradients used for local parameter updates. It adds
nonlocal/autograd computation and should be off when measuring local-learning
compute or memory.

## Direct model configuration

No new `learning_rule` strings are needed. Use either:

```python
net_params.update({
    "learning_rule": "local_exact_rowlocal",  # or "local_diag_rflo"
    "feedback_mode": "direct_fa",
    "input_mode": "match",
    "cross_layer_steps": 0,
})
net_params["ml_params"]["local_bias_mode"] = "direct"  # or "exact"
net_params["ml_params"]["modulation_bounds"] = True
net_params["ml_params"]["m_bounds"] = (-1.0, 1.0)
net = mpn.DeepMultiPlasticNet(net_params)
```

If using layer-specific dictionaries (`ml_params1`, `ml_params2`, ...), set
`local_bias_mode` in each dictionary actually used. Its default is `exact` for
backward compatibility. Set this before model construction.

Use the eligibility configuration required by this project:
`mp_type="mult"`, `m_activation="linear"` or `"scaled_tanh"`,
`m_update_type="hebb_assoc"` or `"hebb_pre"`, and fixed eta/lambda.
The local updates do not train eta/lambda.

The runner defaults to `modulation_bounds=True` with `m_bounds=(-1, 1)`.
For bounded models, each masked write is clamped, and the advanced P/A/Q traces
are multiplied by the clamp derivative before the next step. The optional
cross-layer correction uses the previous write's clamp derivative too. The
derivative is one at the endpoints and zero outside, matching PyTorch autograd.
When advancing explicit traces, complete the step with `update_M_matrix` or
`update_M_matrix_local_fast` to apply the write's clamp derivative.
Unbounded linear models remain supported by setting `modulation_bounds=False`.

The runner also accepts `--modulation-mode none|hard|scaled_tanh` and
`--modulation-bound B` (finite and positive). The default remains `hard` with
`B=1`. Smooth modulation uses `m_activation="scaled_tanh"`, `m_scale=B`, and
`modulation_bounds=False`; combining smooth modulation and hard clipping is
rejected. The scale is a fixed hyperparameter stored in the model configuration.

With raw Hebbian write `S = lambda*M_old + eta*post*pre`, smooth writes use
`M_new = r*B*tanh(S/B) + (1-r)*M_old`, where `r` is the write mask. The derivative
`1-tanh(S/B)**2` is applied to newly advanced traces before mixing with the old
traces. Thus zero masks preserve both the old state and its sensitivities, and
fractional masks retain their interpolation meaning. Frozen entries have zero
write sensitivity. BPTT and local methods share the same forward write map.

Use, for example:

```bash
python scripts/train_mpn.py --dfa --hidden 64 64 --modulation-mode scaled_tanh --modulation-bound 1
```

Hard filenames use `_mb-B-B`, smooth filenames use `_mtanh-B`, and unbounded
filenames omit the modulation tag. Changing modulation does not enable
regularization or change the existing DFA feedback/input/correction settings.

`train_mpn.py` defaults to `REG_LAMBDA=0.0` (no regularization). Setting it to
`1e-4` enables coupled Adam weight decay on
trainable weight matrices, with penalty `(REG_LAMBDA / 2) * sum(W**2)` and no
bias or activity penalty. This is separate from the model's gradient methods;
direct model users configure their optimizer separately. Decay is applied after
task-gradient clipping. Logged loss, scheduling, and alignment remain task-only.
The defaults `MODULATION_BOUNDS=True` and `REG_LAMBDA=0.0` give a bounded,
unregularized comparison. Filename tags record either feature when enabled.

## Mapping to the manuscript

For zero-based MP layer `n`, its output boundary is `h[n+1]`:

```python
ell_h[n + 1] = grad_output @ B_direct[n + 1]
```

`B_direct[0]` credits the trainable input embedding; `B_direct[L]` credits the
top plastic layer. All are registered buffers, not trainable parameters. They
are saved in `state_dict` and follow device/dtype moves. There is no additional
activation derivative in this projection: eligibility already contains it.

- **Row-local + DFA:** use exact within-row `P[b,i,I,J] = dM[b,i,J]/dW[i,I]`
  to compute `E`, then accumulate `sum_b ell[b,i] * E[b,i,I]`.
- **Diagonal + DFA:** use only the scalar sensitivity trace `A[b,i,I]`, then
  accumulate the same learning-signal/eligibility product.
- **Exact bias:** keep `Q` and use `R = phi_prime * (1 + sum_J W*x*Q)`.
- **Direct bias:** use `R = phi_prime`, with `Q=None`; no Q is allocated,
  read, or updated in the trace-based rules.
- **Input embedding:** its own DFA signal times its activation derivative and
  raw input. `input_mode="match"` prevents the extra BPTT splice for local runs.
- **Readout:** the usual exact output-layer gradient.

"Exact" in row-local + DFA qualifies eligibility only. Random feedback is a
surrogate boundary signal even at the top layer, so the full update is generally
not the exact task gradient. Diagonal + DFA with direct bias is synapse-local
conditional on a postsynaptic feedback signal. Row-local + DFA still uses
within-row cross-synapse information. Both require output-error communication.

Residual writes use the block activation, not the full residual stream, as in
the existing implementation. The repository's pre-only write includes the
constant `1/sqrt(post_width)`; the manuscript uses constant 1. This existing
normalization is preserved and can be absorbed into eta when comparing formulas.

The existing masked loss is `mean((mask * (output - target))**2)`. Thus
`grad_output` includes `2 * mask**2 / numel`, rather than the manuscript's
unnormalized half-MSE convention. Nonbinary masks must be interpreted accordingly.
This preserves the task-loss normalization; the runner can optionally add weight
regularization through its optimizer as described above.

The training runner accumulates sequence gradients and applies Adam per batch;
it does not change parameters at every time step. Eligibility is forward/causal,
but the optimized existing runner caches sequence arrays for readout/embedding
reductions and outputs. Consequently the *implemented total memory* is not
constant in sequence length, even though eligibility-state size is. No full
BPTT graph is built for the two local rules in this preset.

## Changed files

- `core/mpn.py`: configurable exact/direct bias eligibility in explicit and fused
  trace paths; direct mode omits Q; guard against DFA plus temporal correction.
- `scripts/train_mpn.py`: `--dfa`, `--rules`, `--local-bias-mode`, and
  `--[no-]grad-align`; DFA-specific labels and bias-mode output filename tag.
- `tests/test_dfa.py`: independent autograd graph oracles and configuration tests.

`core/mpn_archive.py` is an unchanged historical implementation. The runner
imports `core/mpn.py`.

## Validation

The residual/correction revision makes three changes:

- The runner honors residual settings at every depth, including one MP layer.
- The optional one-step correction uses the forward write's constant
  `1/sqrt(n_output)` for `hebb_pre` and omits its postsynaptic source.
- Delayed sources use the previous timestep's plasticity update mask, so a
  disabled write contributes no delayed credit. Graded masks are also supported.

`--dfa` continues to require `--cross-layer-steps 0`; the correction fixes apply
to the separate spatial-feedback reference algorithm. Exact two-step MP
gradients require exact spatial feedback and exact weight/bias eligibility.

```bash
python -m unittest discover -s tests -v
```

The tests cover 72 combinations of weight rule, bias rule, residual setting,
associative/pre-only/mixed plasticity, and unbounded/hard/smooth modulation, with unequal widths, nonzero initial
modulation, graded loss masks, and masked plasticity updates. The row-local
oracle differentiates an isolated layer trajectory. The diagonal oracle
replays each scalar synapse with other synapses' modulation histories detached;
it does not duplicate the A-trace recurrence. All parameter updates, including
input and readout, match within `rtol=1e-9, atol=1e-10` in float64.

Additional tests check explicit versus fused traces, no BPTT call in local
updates, fixed and serialized feedback buffers, no Q for direct bias, and
rejection of conflicting preset settings. A one-batch, two-plastic-layer
`delaygo` comparison was also run through the training/plotting pipeline.
These are correctness checks, not evidence of converged task performance.

`tests/test_residual_correction.py` additionally checks the CLI/model/metadata
residual setting at depths 1, 2, and 3; exact two-step MP/readout gradients in
24 residual/plasticity/update-mask configurations against an independent full
autograd graph; and zero correction when all writes are disabled for each of
the three local rules. The embedding is intentionally excluded from the
two-step exactness claim because its gradient retains its selected learning rule.

Additional bounded tests exercise saturation, clamp endpoints, graded write masks,
and agreement between the general and fast write paths. Small training runs check
all four learning rules for both deep and single-layer MPNs, including optimizer
regularization and bounded states. An independent optimizer test compares coupled
decay with explicitly adding the L2 gradient.
