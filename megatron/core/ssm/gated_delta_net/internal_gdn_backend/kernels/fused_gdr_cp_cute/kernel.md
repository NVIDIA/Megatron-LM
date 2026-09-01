# Fused GDR CP CuTeDSL preprocessing

This package provides the SM100 context-parallel boundary preprocessing used by
the internal gated-delta-rule backend. It computes the cross-rank forward
initial state and backward terminal-state gradient; the local sequence backward
is still performed by `fused_gdr_bwd_cute`.

## Selection

The internal backend tries this package before FLA's CP preprocessing. Set
`MCORE_GDN_CP_CUTEDSL` to:

- `auto` (default): use CuTeDSL when the runtime and operands are supported.
- `1`: request CuTeDSL, while retaining a correctness-preserving FLA fallback
  for unsupported operands.
- `0`: disable this package and use FLA CP preprocessing.

`FLA_CP_CUTEDSL` is accepted as a compatibility alias when the Megatron
variable is unset.

## Supported fast path

- NVIDIA SM100 B200/GB200 with an NCCL CP group of size 2 through 8.
- BF16 operands, scalar `g` gate, chunk size 64.
- Equal key/value dimensions of 64 or 128.
- Dense BTHD forward preprocessing treats `(batch, value head)` as one native
  work-item space, so all batch elements run in one kernel launch. The
  warp-specialized backward does the same for `K = V = 128`.
- When a dense shape is not supported by the native-batch specialization, the
  dispatcher preserves the installed FLA kernel's `B = 1` contract by slicing
  the boundary preprocessing per batch element and concatenating the states.
- Packed THD uses the validated CPU sequence offsets attached to the CP context.
- A local sequence length of at least 64 that is not divisible by 64 is handled
  inside the kernels; inputs are not padded or copied. Dense backward shards
  shorter than one chunk use the FLA fallback.

Calls outside this contract fall back to the installed FLA implementation. The
dispatcher exposes launch counters and fallback reasons so benchmarks can prove
which path executed.

## Notes

The kernels use PyTorch symmetric memory for peer exchange. Every rank must
make the same dispatch decision; therefore the dispatcher classifies the whole
rank chain from immutable global CPU sequence offsets before launching.

The Python modules carry their upstream MIT copyright and license notice in
their source headers. No separate runtime checkout of an FLA fork is required.

## Verified CP4 case

On four GB200 GPUs, the production `GatedDeltaNet.forward` path was checked at
`B=2`, local `T=8192`, global `T=32768`, `H=64`, and `K=V=128`. Correctness
covered the output, input gradient, and the `q`, `k`, `v`, `g`, and `beta`
gradients. The non-64-aligned local `T=8190` case was checked separately.

For the aligned case, CUDA Event timing with 10 warmups and 20 samples, taking
the maximum time across CP ranks, measured:

| Path | Median | Speedup vs. pure FLA |
| --- | ---: | ---: |
| Pure FLA | 14.926 ms | 1.00x |
| FLA CP + fused GDR backward | 10.931 ms | 1.37x |
| CuTeDSL CP + FLA backward | 11.138 ms | 1.34x |
| CuTeDSL CP + fused GDR backward | 8.604 ms | 1.73x |

The final path executed one CuTeDSL forward boundary launch, one CuTeDSL
backward boundary launch, and one fused GDR backward launch per iteration,
with no CuTeDSL fallback. A separate post-review rerun of the final path
measured 8.603 ms median (P10 8.593 ms, P90 8.646 ms, CV 0.353%).
