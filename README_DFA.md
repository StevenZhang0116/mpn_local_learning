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

`--no-residual` disables residual connections. The existing runner enables
identity residuals only for compatible widths in a multi-MP-layer stack.
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
net = mpn.DeepMultiPlasticNet(net_params)
```

If using layer-specific dictionaries (`ml_params1`, `ml_params2`, ...), set
`local_bias_mode` in each dictionary actually used. Its default is `exact` for
backward compatibility. Set this before model construction.

Use the clean eligibility configuration already required by this project:
`mp_type="mult"`, `m_activation="linear"`, `modulation_bounds=False`,
`m_update_type="hebb_assoc"` or `"hebb_pre"`, and fixed eta/lambda.
The local updates do not train eta/lambda.

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
This revision preserves the objective and its normalization.

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

```bash
python -m unittest discover -s tests -v
```

The tests cover 24 combinations of weight rule, bias rule, residual setting,
and associative/pre-only/mixed plasticity, with unequal widths, nonzero initial
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
