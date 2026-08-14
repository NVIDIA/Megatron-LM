# Forward H Kernel Design Notes

## Current Stage

The correctness-first CUDA rewrite and WMMA fallback have been removed. The
compiled entry point now routes only to the SM100 port kernel and raises for
unsupported inputs instead of falling back.

The exposed port path supports the scalar `g` gate used by GatedDeltaNet. The
Python wrapper accepts FLA layout `[B, T, H]` and passes contiguous port layout
`[B, H, T]` into the extension. Unsupported input shapes still raise directly
instead of falling back.

The bwd CUTE/TMA/TMEM files have been copied into
`csrc/ops/chunk_delta_h_fwd_sm100_bwd_port*.cuh` as the editable port scaffold.
The compiled source is `chunk_delta_h_fwd_sm100.cu`, and it launches
`chunk_delta_fwd_port_kernel` through the port scaffold. The scaffold's
top-level `ParamsHost`/`ParamsBase` contract has already been changed to fwd
tensors (`k`, `w`, `u`, optional `h0`, outputs `h`, `ht`, `v_new`).

Verified in Docker on `umbriel-b200-035`:

```text
Quick correctness check PASSED
Full correctness check PASSED
Performance case: B=2 T=8192 H=64 K=128 V=128
CUDA first baseline: 63.2480 ms
FLA Triton:    0.4127 ms
CUDA/FLA:      153.2485x
```

The full correctness run uses smaller random input scale for long sequences
(`T>=1024`) to avoid recurrence overflow unrelated to kernel correctness.

## Root Cause Fixed

FLA stores `v_new = u - w @ h` before applying the scalar `g` gate. The gated
value is used only for the hidden-state update. The initial CUDA baseline wrote
the gated value to `v_new`, which failed the FLA `USE_G=True` comparison. The
kernel and local PyTorch reference now store ungated `v_new`.

For the SM100 port, `H *= exp2(g_last)` must be applied by epilogue warps that
cover all TMEM subpartitions. Applying it only from the MMA consumer warp scales
one subpartition and leaves the other `V` lanes stale.

## Optimization Direction

The original baseline performed scalar loops in one CTA per `(B,H,V tile)` and
was expected to be far slower than FLA. The next implementation stage should
continue replacing the mainloop with the planned SM100/CuTe dataflow:

```text
store h_i = H
tmp_v[V,T] = H[V,K] @ W[T,K]^T
v_new[T,V] = u[T,V] - tmp_v[V,T]^T
apply gate to the TMEM-bound update value, not the stored v_new
H[V,K] += gated_v_new[V,T] @ K[K,T]^T
store ht after the final chunk
```

The existing tests should be kept unchanged while the CUDA implementation is
replaced under the wrapper.
