# GDN chunkwise context parallelism

`TransformerConfig.gdn_cp_mode="chunkwise"` selects opt-in chunkwise GDN
training with Transformer Engine and cuDNN. The default remains `"headwise"`.
No cuDNN kernel change is required. Transformer Engine must provide
`GatedDeltaNetAttention.get_context_parallel_backend()` and cuDNN must provide
the forward and backward state-summary operations.

The initial implementation supports TP1, no sequence parallelism, batch-one
packed THD or dense input, and fused pre-GDN preparation. Physical packed lengths
must be divisible by twice the CP size. When attention CUDA graphs are configured,
warmup and capture compute packing metadata on the GPU without device-to-host reads.
CUDA graph replay supports changing boundary values in fixed-size input buffers;
changing the number of documents or token-buffer shape requires another graph.
The CP process group is fixed for each graph.
Deterministic mode is unsupported. CP1 retains the ordinary local GDN path.

Input and output retain per-document zigzag ownership. Inside each GDN layer,
eager execution moves hidden states to globally contiguous shards and exchanges a
convolution halo. Configured attention graphs instead use contiguous shards of each document,
two fixed-size redistribution messages per rank, and per-document halos and state
summaries. This keeps communication sizes independent of the packed lengths.
Both paths enter the existing fused preparation and MCore's chunkwise CP protocol.
Normalized value features move back to zigzag ownership before the output projection.
GQA layers keep
their existing CP layout. Model parameters and checkpoint keys do not change.

Eager packing plans are built outside capture using host boundaries, then reused
across layers with bounded storage, weak tensor
references, mutation versions and CUDA stream events. Inference tensors without
mutation versions bypass reuse. Graph capture records metadata construction and
does not depend on the eager layout cache. Warmup must use the configured graph path
so its collective operations and kernels are initialized before capture. Exact packing
buffer identities and values need not match warmup. Graphs covering only MLP or MoE
retain the eager GDN path. The runtime process group is passed explicitly.

Gate parameter gradients are summed in FP32 across CP before rounding to their
original parameter dtype. They are divided by CP size so the ordinary DP+CP
gradient sum counts them once. This preserves the gradient-reduction contract
while avoiding rounding independent BF16 contributions before summation.

Validation should include full-module outputs, input gradients and every
parameter gradient against an independent recurrence, with packed padding,
document reset boundaries and recomputation. The local state operations also
have bitwise replay coverage. Fused preparation retains its existing
non-deterministic reductions; the module rejects deterministic mode.

Selecting chunkwise does not imply a universal performance win. Include
redistribution, halo exchange, preparation, projections, backward and recompute
when comparing algorithms, then validate the actual training workload and
convergence before changing recipe defaults.
