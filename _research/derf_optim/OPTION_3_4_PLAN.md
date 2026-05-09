# Option 3 / Option 4 plan — TE-integrated Derf

This is the surgery plan for the two remaining throughput options.
Discovery work below is concrete; implementation is intentionally deferred
to a focused follow-up session because the engineering surface is genuine
even though the math is trivial.

## Key discovery

**TE's `LayerNormLinear` is not a single fused CUDA kernel.** It is, in
order, `apply_normalization()` (which calls `tex.rmsnorm_fwd` or
`tex.layernorm_fwd`) followed by a separate cuBLAS GEMM. The "fusion" is
careful pipelining + activation buffer reuse. This means we don't need to
rewrite a CUTLASS-style fused mainloop — we just need to land a Derf
forward kernel into the same dispatch.

Container TE version: **2.11.0+c188b533**.

## Two routes to TE-integrated Derf

### Option 4 (lower effort, ~8-12h focused work) — Python monkeypatch

No TE rebuild. Write a Derf forward kernel (our Option 2 Triton kernel
already works), monkeypatch `apply_normalization` to route Derf calls
there, and let TE's GEMM machinery handle the matmul. Use a subclass of
`TELayerNormColumnParallelLinear` to carry the extra `alpha`/`s` scalars.

### Option 3 (~2-3d focused work) — vendor TE fork

Clone TE @ `c188b533`, copy `ln_fwd_kernels.cuh`/`ln_bwd_kernels.cuh` →
`derf_fwd_kernels.cuh`/`derf_bwd_kernels.cuh`, strip the `Stats`
reduction, substitute the affine math with `gamma * f(alpha*x + s) + beta`.
Plumb alpha/s through `ForwardKernelParams` struct, register a new
`tex.derf_fwd` C++ binding, build wheel, override container's TE via
PYTHONPATH.

## Surgery sites (exact file:line in TE main, near c188b533)

### Python side
| file | what |
| --- | --- |
| `transformer_engine/pytorch/module/_common.py:19-32` | `_get_normalization_func` dispatch — **monkeypatch site for Option 4** |
| `transformer_engine/pytorch/module/_common.py:34-58` | `apply_normalization` body |
| `transformer_engine/pytorch/module/layernorm_linear.py:86` | `_LayerNormLinear` autograd Function — saves `(inp, ln_weight, ln_bias, weight, ...)`; for Derf we need to also save `alpha, s` |
| `transformer_engine/pytorch/module/layernorm_linear.py:231` | `apply_normalization()` call inside `_LayerNormLinear.forward` |
| `transformer_engine/pytorch/module/layernorm_linear.py:1055-1069` | norm backward dispatch — `tex.rmsnorm_bwd` for RMSNorm; need a `derf_bwd` branch |
| `transformer_engine/pytorch/module/layernorm_linear.py:1121` | `LayerNormLinear` user-facing class |
| `transformer_engine/pytorch/module/layernorm_linear.py:1173` | hardcoded assert `normalization in ["LayerNorm","RMSNorm"]` — needs allowlist update or subclass override |

### C++/CUDA side (Option 3 only)
| file | role |
| --- | --- |
| `transformer_engine/common/normalization/layernorm/ln_fwd_kernels.cuh` | template for the kernel — copy → `derf_fwd_kernels.cuh`, strip `Stats stats(...)` and `stats.compute(...)` calls (lines 55-91), replace `y_ij = rs * (x_ij - mu)` with `f(alpha * x_ij + s)` |
| `transformer_engine/common/normalization/layernorm/ln_fwd_cuda_kernel.cu` | launcher — duplicate as `derf_fwd_cuda_kernel.cu`, add `alpha`/`s` params |
| `transformer_engine/common/normalization/common.h:~80-150` | `ForwardKernelParams` struct — add `void *alpha; void *s;` fields |
| `transformer_engine/common/normalization/layernorm/ln_api.cpp` | C++ entry point — duplicate as `derf_api.cpp`, register binding |
| `transformer_engine/pytorch/csrc/extensions/normalization.cpp` (or similar) | pybind binding for `tex.derf_fwd` |
| `transformer_engine/common/CMakeLists.txt` | add `derf_*` source files to build |

### Backward
- `transformer_engine/common/normalization/layernorm/ln_bwd_kernels.cuh:~50-200`
  has the bwd template. For DyT/Derf the gradient through `f` is closed-form:
  - Derf: `d/dz erf(z) = 2/sqrt(pi) * exp(-z^2)`, then chain through `alpha`
  - DyT:  `d/dz tanh(z) = 1 - tanh(z)^2`, chain through `alpha`
  - No reduction needed — much simpler than RMSNorm/LayerNorm bwd which has
    the reductions for `dgamma`, `dbeta`, `dmu`, `dvar`.

## Why we stopped here

Option 1 (torch.compile, 271 TFLOP/s) already recovers ~58% of the lost
throughput in 50 lines of Python and ships immediately. Spending 1-2 days
on Option 3 to climb to ~310 is worth doing if the use case warrants it
(e.g., a long Derf training run where 15% throughput * many GPU-hours
adds up), but it's not the right shape for a single autonomous session
done in one go.

## Concrete next-session steps (Option 4 first)

1. `pip install transformer_engine==2.11.0` locally (just for source reading
   alignment with container).
2. Write `_research/derf_optim/option4_te_patch.py`:
   - Subclass `te.pytorch.LayerNormLinear` as `DerfLayerNormLinear`,
     override `__init__` to skip the normalization assert and store
     `alpha`/`s` Parameters.
   - At module-load time, monkeypatch
     `transformer_engine.pytorch.module._common._get_normalization_func` to
     return our Derf kernel for `normalization == "Derf"`. Read `alpha`/`s`
     from a thread-local or from attributes set on the gamma tensor.
   - The Derf kernel itself is the `_norm_linear_fwd_kernel` from
     `option2_triton.py`, but **only the norm part** (no matmul) — TE's
     GEMM machinery does the matmul.
3. Add `make_qkv_class("Derf")` factory returning `DerfLayerNormLinear`.
4. Wire `APERTUS_DERF_OPTIM=te_patch` in `gpt_layer_specs.py:336`.
5. Numerical correctness via `_research/derf_optim/test_correctness.py` (add
   `test_option4` that compares against eager reference at the
   `LayerNormLinear` level — only viable on cluster with CUDA + TE).
6. Throughput sbatch from `o2-derf-triton-throughput.sbatch` template.

## Owner notes

- Don't bother with `c10::optional<Tensor>` at the pybind boundary — see
  Option 3's first attempt traceback. Always pass an empty 0-element
  tensor and check `t.numel() > 0`.
- TE 2.11's `apply_normalization` returns `(ln_out, mu, rsigma)`. Our Derf
  variant needs to return `(ln_out, dummy, dummy)` (or save `(alpha, s)`
  state in the slot, depending on how the bwd consumes it).
- Numerical tolerances on bf16: rtol 1e-1, atol 5e-2 for param grads;
  rtol 5e-3, atol 1e-3 for fp32. See `test_correctness.py` for the helper.
