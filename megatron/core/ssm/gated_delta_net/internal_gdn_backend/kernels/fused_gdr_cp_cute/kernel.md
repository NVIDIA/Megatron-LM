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

## Supported fast path

- NVIDIA Blackwell data-center GPUs (SM100 B200/GB200, SM103 GB300) with an NCCL
  CP group of size 2 through 8.
- Packed THD input with validated CPU sequence offsets attached to the CP
  context. Call `finalize_packed_seq_params()` after padding and before the
  model forward so the existing `cp_partition_route` owns that snapshot.
- BF16 operands, chunk size 64, and exactly one gate: GDN's scalar `g`
  (`[1,T,HV]`, fp32 or bf16) or KDA's per-key `gk` (`[1,T,HV,K]` fp32, FLA's
  chunk-local log2 cumsum). The kernels read only the chunk-last row of `gk`
  and apply no per-token gating in that mode, so the `k` / `q` operands must
  arrive pre-gated as FLA's `kg` / `qg` -- the same contract as FLA's own
  Triton CP pre-process. `HV % 4 == 0` is required in both modes (only the
  `g` TMA box needs it; the dispatcher keeps one head-geometry contract).
- Equal key/value dimensions of 128.
- Any local window length that is not a multiple of 64 is handled inside the
  kernels (the TMA descriptors carry the logical `T`, so a ragged tail is
  zero-filled by the hardware); inputs are not padded or copied. Windows
  shorter than one chunk (`T < 64`) rely on the same zero-fill and are
  covered by the `cp_tiny_window` GDR harness case.

## Modules

Only the warp-specialized (tcgen05) engines remain:

| Module | Role |
| --- | --- |
| `backend.py` | dispatcher: predicates, chain classification, emit flags, wrapper cache and rendezvous, launch counters, fallback reasons |
| `protocol.py` | `CPProtocolMixin`: protocol geometry (`NV`, `NK`, `NSTRIPS`, `STRIP`, gate mode), sys-scope flag/ack spins and signals, `hm_sym` strip offsets, the cp.async staging + bf16 hi/lo warp-MMA fold both merge roles run, and the tcgen05 gemm issue helper |
| `fused_ws.py` | `PreProcessFwdFusedWS` (forward compute + merge roles) and its host wrapper `CuteDSLFusedCPPreProcessWS` |
| `bwd_ws.py` | `PreProcessBwdFusedWS` (reverse compute + suffix fold) and its host wrapper `CuteDSLFusedCPBwdPreProcessWS` |
| `_host.py` | host helpers shared by the wrappers: pointer / CUstream memoization, gate normalization |
| `symm.py` | `CPSymmBuffers`: the symmetric `hm_sym` / `flags` / `acks` tensors and peer pointer tables |

Each wrapper compiles exactly one program. The forward's aligned and ragged specializations are the same program under scalar `g` (`need_lam` is on either way) and the per-key `gk` wrapper is built as the ragged specialization; the backward wrapper runs the ragged (tail-masked) specialization for every `T` in every gate mode. Keying the backward on `T % 64 == 0` used to compile the ragged program lazily on only the ranks that own an unaligned boundary window, several minutes into a packed run while their peers spun, and mixed two programs across ranks; the single program costs the lambda table and the dV tail mask on aligned calls. The wrapper cache key and the
handshake signature carry the gate mode, so a GDN and a KDA layer of the same
head geometry never share a wrapper. The forward engine's K = 64 `gk` pairing
is unverified and unreachable: the dispatcher admits only K = V = 128.

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
The same file covers the per-key (`gk`) gate mode through the KDA adapter at
the Kimi head geometry (K == V == 128) under CP2 and CP4 in strict `cute`
mode: one launch per direction, no fallback, output and all seven gradients
against FLA's own CP path; a companion test pins down that head dims outside
the kernels' envelope raise in `cute` mode and are recorded as fallbacks in
`auto` mode instead of passing silently. The tests skip unless the pytest
process group has exactly `cp_size` ranks (run them under `torchrun
--nproc_per_node=2` and `=4`).
