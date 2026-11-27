# Fused wide-residual shortcut seams

For wide-residual hybrid models, with or without shortcut-connected MoE
(`--moe-shortcut-wide-residual-norm-fusion`), each residual seam is one Gluon kernel: it mixes the
wide-residual streams with the branch update, writes the new residual, and emits the RMSNorm
reader(s) of it. Each linked consumer GEMM determines the activation format it receives. Each
linked producer GEMM determines the format of its output gradient. Eligible GEMMs receive MXFP8
(E4M3 for gradients), and other handoffs use BF16. The residual stream is FP32
(`--fp32-residual-connection`) or BF16. Connection logits can be FP32 or BF16. The requirements (BF16
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

During training, `--residual-stream-recompute-num-layers` sets the replay interval length
(default one write). The interval retains
its first input residual and each write's tape. Before the interval's backward GEMMs run,
one replay kernel reconstructs the residuals and the columnwise MXFP8 activations needed
for weight gradients. The next interval triggers replay for its predecessor. A stack output
hook handles the final interval when its consumers need columns. Reusing a retained graph
for a second backward is unsupported.

Residual writes are bit-identical across schedules and to replay; reader outputs differ by at
most one BF16 ulp between schedules. Tests: `tests/unit_tests/fusions/test_wide_residual_shortcut.py`
(FP64/autograd references) and `tests/unit_tests/determinism/kernels/test_wide_residual_shortcut.py`.
