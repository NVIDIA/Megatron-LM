# Fused wide-residual shortcut seams

For wide-residual hybrid models, with or without shortcut-connected MoE
(`--moe-shortcut-wide-residual-norm-fusion`), each residual seam is one Gluon kernel: it mixes the
wide-residual streams with the branch update, writes the new residual, and emits the RMSNorm
reader(s) of it — reader zero directly as MXFP8 for an exclusive TE Linear consumer under an
MXFP8 recipe, otherwise BF16. The residual stream is FP32 (`--fp32-residual-connection`) or BF16
and the connection logits FP32 or BF16; all arithmetic is FP32. The requirements (BF16
parameters, RMSNorm, 2-6 streams, a shared expert in shortcut blocks, no dropout/bias, no CP/PP,
at most 65536 local tokens per call, 16384 above hidden 4096, a GB300 for training, ...) are
listed in the `moe_shortcut_wide_residual_norm_fusion` docstring.

| boundary | updates | readers | kernel |
|---|--:|--:|---|
| compute → MoE inside a block | 1 | 1 | `shortcut_add_wide_residual_write_read_rmsnorm_quant`, no shared update |
| lone layer → lone layer (M → -, - → M) | 1 | 1 | `shortcut_add_wide_residual_write_read_rmsnorm_quant`, no shared update |
| block → lone layer ([* E] → M) | 2 (routed + shared) | 1 | `shortcut_add_wide_residual_write_read_rmsnorm_quant` |
| block → block (shortcut add) | 2 (routed + shared) | 2 | `shortcut_add_..._two_reader_rmsnorm_quant` |
| lone layer → block (M → [* E]) | 1 | 2 | `shortcut_add_..._two_reader_rmsnorm_quant`, no shared update |
| into a standalone MoE layer (M → E) | 1 | 1 | `shortcut_add_wide_residual_write_read_rmsnorm_quant`, BF16 read |

A standalone MoE layer (outside a shortcut block, or every MoE layer without
`--moe-shortcut-connection`) takes its read whole: one BF16 tensor for its router, shared expert
and dispatcher, whose gradients autograd sums. Under MXFP8 its Linears quantize that read
themselves, unlike a block's MoE read, which goes to the shared expert's first Linear as MXFP8.

A write that the next seam applies is passed forward as a `PendingWideResidualWrites`: the routed and
shared MoE outputs (the shortcut add, whose BF16 sum is rounded before the write) or a lone
layer's single update.

Modules:

- `api.py`: the seam functions and their autograd function (all state is call-local).
- `forward_kernel.py`: the forward kernel and its GB300 schedule selection (sweep-fit decision
  trees for training and prefill, a decode tree with CTA clusters for short rows, a fixed schedule
  in batch-invariant mode); inference launches save nothing for backward.
- `backward_kernel.py`: one persistent backward kernel per seam plus the final gradient
  reductions (GB300 only).
- `residual_replay.py`, `replay_kernel.py`: every training seam records its write into a replay
  interval (one write per interval, or N with `--residual-stream-recompute-num-layers N`). An
  interval keeps only its input residual and the writes' tapes; it is replayed bit-exactly once,
  before its backward GEMMs (in the next interval's first seam's backward, or from a hook on the
  stack's output), rebuilding the freed residuals and filling its MXFP8 readers' column slots.
  Forwards and backwards alternate on one stream (fusion rejects PP, overlapped EP communication
  and full recompute), so the replay needs no events; retained-graph backward is unsupported.
- MXFP8 handoff: the stack fixes each read's formats at link time from the per-layer precision
  its forward uses; an MXFP8 consumer TELinear (hooked in
  `megatron/core/transformer/wide_residual.py`) unpacks its read's replayed columnwise operand
  for its weight-gradient GEMM (TE runs its GEMMs unchanged).

Residual writes are bit-identical across schedules and to replay; reader outputs differ by at
most one BF16 ulp between schedules. Tests: `tests/unit_tests/fusions/test_wide_residual_shortcut.py`
(FP64/autograd references) and `tests/unit_tests/determinism/kernels/test_wide_residual_shortcut.py`.
