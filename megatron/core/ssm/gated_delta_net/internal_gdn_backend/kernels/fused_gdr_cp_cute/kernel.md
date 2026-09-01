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
- Dense BTHD is orchestrated one batch slice at a time.
- Packed THD uses the validated CPU sequence offsets attached to the CP context.
- A local sequence length that is not divisible by 64 is handled inside the
  kernels; inputs are not padded or copied.

Calls outside this contract fall back to the installed FLA implementation. The
dispatcher exposes launch counters and fallback reasons so benchmarks can prove
which path executed.

## Notes

The kernels use PyTorch symmetric memory for peer exchange. Every rank must
make the same dispatch decision; therefore the dispatcher classifies the whole
rank chain from immutable global CPU sequence offsets before launching.

The Python modules carry their upstream MIT copyright and license notice in
their source headers. No separate runtime checkout of an FLA fork is required.
