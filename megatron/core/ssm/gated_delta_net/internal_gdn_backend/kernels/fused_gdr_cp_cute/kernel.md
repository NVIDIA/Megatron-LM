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
- Packed THD input with validated CPU sequence offsets attached to the CP
  context. Call `finalize_packed_seq_params()` after padding and before the
  model forward so the existing `cp_partition_route` owns that snapshot.
- BF16 operands, scalar `g` gate, chunk size 64.
- Equal key/value dimensions of 64 or 128.
- A local sequence length of at least 64 that is not divisible by 64 is handled
  inside the kernels; inputs are not padded or copied.

The production `GatedDeltaNet.forward` path preserves MCore's established
chunkwise-CP contract: dense SBHD input with `B > 1` is rejected. Use packed THD
input, or use dense input with `B = 1`. Calls outside the fast-path contract
fall back to the installed FLA implementation. The dispatcher exposes launch
counters and fallback reasons so benchmarks can prove which path executed.

## Notes

The kernels use PyTorch symmetric memory for peer exchange. Every rank must
make the same dispatch decision; therefore the dispatcher classifies the whole
rank chain from immutable global CPU sequence offsets before launching.
Replacing or mutating the device offsets, or mutating their CPU snapshot, makes
the route invalid and raises instead of risking a rank-divergent collective.
One symmetric-memory wrapper is cached per process group, device, and kernel
shape so every rank performs rendezvous in the same order. Because its
communication epoch is mutable, each cached wrapper marshals the complete
launch (including tensor preprocessing) onto its first CUDA stream; as with
NCCL, CP collectives must be submitted in identical order on every rank. The
common owner-stream path uses a raw stream handle and does not construct a
temporary ``torch.cuda.Stream`` object. A
cache-miss-only rank handshake raises an explicit error for mismatched
concurrent first-use shapes before allocating symmetric memory; it is absent
from the steady-state path.
Packed THD GatedDeltaNet with chunkwise CP rejects CUDA-graph configurations
that capture its attention region, before layout conversion. Partial graphs
that capture only MLP or MoE regions leave GDN eager and remain supported. FLA
derives rank-chain topology from capture-time CPU offsets, so replaying a
different packing cannot be made correct by falling back from CuTeDSL alone.

The Python modules carry their upstream MIT copyright and license notice in
their source headers. No separate runtime checkout of an FLA fork is required.

## Verified CP4 cases

The production packed-THD `GatedDeltaNet.forward` path was checked on four
GB200 GPUs for two logical sequences. Each sequence has global length 32768 and
local length 8192, with `H=64` and `K=V=128`. Each rank receives a local tensor
of shape `[16384, H, D]` and validated global CPU metadata
`cu_seqlens=[0, 32768, 65536]`; the normal MCore packed CP layout conversion is
enabled. Correctness covered the output, input gradient, and the `q`, `k`, `v`,
`g`, and `beta` gradients. For the non-64-aligned case, the two local
sequence lengths were 8190 and 8194; the combined path measured 11.114 ms
versus 15.117 ms for pure FLA (1.36x).

The aligned timings below use CUDA Events with 10 warmups and 20 samples. Each
sample is the maximum time across CP ranks, and the table reports the median.

| Path | Median | Speedup vs. pure FLA |
| --- | ---: | ---: |
| Pure FLA | 14.794 ms | 1.00x |
| FLA CP + fused GDR backward | 13.035 ms | 1.13x |
| CuTeDSL CP + FLA backward | 12.907 ms | 1.15x |
| CuTeDSL CP + fused GDR backward | 11.197 ms | 1.32x |

The combined path executed one CuTeDSL forward boundary launch, one CuTeDSL
backward boundary launch, and one fused GDR backward launch per iteration,
with no CuTeDSL fallback. The GB200 test
`tests/unit_tests/ssm/test_internal_gdn_backend_e2e.py` checks the forward and
backward launch-counter deltas and requires the fallback-reason map to remain
empty, in addition to comparing the output and five gradients with pure FLA.
