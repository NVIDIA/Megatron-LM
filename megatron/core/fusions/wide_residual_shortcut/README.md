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

Every seam is one `fused_seam` call:

| boundary | updates | readers | notes |
|---|--:|--:|---|
| compute → MoE inside a block | 1 | 1 | no shared update |
| lone layer → lone layer (M → -, - → M) | 1 | 1 | no shared update |
| block → lone layer ([* E] → M) | 2 (routed + shared) | 1 | |
| block → block (shortcut add) | 2 (routed + shared) | 2 | |
| lone layer → block (M → [* E]) | 1 | 2 | no shared update |
| into a standalone MoE layer (M → E) | 1 | 1 | BF16 read |

A standalone MoE layer (outside a shortcut block, or every MoE layer without
`--moe-shortcut-connection`) takes its read whole: one BF16 tensor for its router, shared expert
and dispatcher, whose gradients autograd sums. Under MXFP8 its Linears quantize that read
themselves, unlike a block's MoE read, which goes to the shared expert's first Linear as MXFP8.

A write that the next seam applies is passed forward as a `PendingWideResidualWrites`: the routed and
shared MoE outputs (the shortcut add, whose BF16 sum is rounded before the write) or a lone
layer's single update.

This package knows tensors, formats and kernel limits, not Megatron modules: the seam modules,
their linking and the per-layer precision live in `megatron/core/transformer/wide_residual.py`.

Modules:

- `api.py`: the seam op `fused_seam`, its autograd function and its call-time checks
  (`check_mxfp8_call`).
- `forward_kernel.py`: the forward kernel and its GB300 schedule selection (sweep-fit decision
  trees for training and prefill, a decode tree with CTA clusters for short rows, a fixed schedule
  in batch-invariant mode); inference launches save nothing for backward.
- `backward_kernel.py`: one persistent backward kernel per seam plus the final gradient
  reductions (GB300 only).
- `residual_replay.py`, `replay_kernel.py`: every training seam records its write into a replay
  interval (one write per interval, or N with `--residual-stream-recompute-num-layers N`). An
  interval keeps only its input residual and the writes' tapes; it is replayed bit-exactly once,
  before its backward GEMMs (in the next interval's first seam's backward, or from a hook on the
  stack's output), rebuilding the freed residuals and filling its MXFP8 readers' column slots,
  which consumer TE Linears unpack for their weight-gradient GEMMs (`install_columns`; TE runs
  its GEMMs unchanged). Forwards and backwards alternate on one stream (fusion rejects PP,
  overlapped EP communication and full recompute), so the replay needs no events;
  retained-graph backward is unsupported. The seam modules open intervals and the stack's last
  write closes the last one (`wide_residual.link_residual_seams`).

Residual writes are bit-identical across schedules and to replay; reader outputs differ by at
most one BF16 ulp between schedules. Tests: `tests/unit_tests/fusions/test_wide_residual_shortcut.py`
(FP64/autograd references) and `tests/unit_tests/determinism/kernels/test_wide_residual_shortcut.py`.
