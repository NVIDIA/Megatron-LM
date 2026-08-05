# MXFP8 Support in M-FSDP v2

## Proposal

Add MXFP8 (block-scaled FP8, E4M3) **model-weight** support to Megatron-FSDP v2,
matching the v1 `fp8_param_gather` semantics:

1. **Model weights are the only fp8 state.** After each optimizer step, the fp32
   main weights are quantized into **two standalone MXFP8 E4M3 weights per
   parameter**: a row-wise quantized weight for the forward pass and a
   column-wise quantized weight for the backward pass, each with its own
   per-block scales. This mirrors TE's `cast_master_weights_to_fp8` and the
   v1 buffer's `weight_buffer` / `transpose_weight_buffer` pair.
2. **The parameter all-gather moves both fp8 payloads.** The unsharded
   parameters are TE `MXFP8Tensor` weights that Transformer Engine layers
   consume directly — fp8 GEMMs in forward and backward, with the transpose
   cache created at unshard and discarded at reshard (v1 parity).
3. **Main gradients stay fp32/bf16 — never fp8.** The existing v2 gradient
   path (bf16 partial grads -> reduce-scatter -> fp32/bf16 `main_grad`) is
   unchanged.
4. **Optimizer states stay fp32.** Standard master-weight fp8 training; the
   optimizer adapter and `main_weight` (fp32) are untouched.

Sharded weight storage and all-gather payload are 2x fp8 = 1x bf16 per
parameter — parity with the bf16 path — and the gains are fp8 GEMM throughput
and parity memory with fp8 compute, exactly like v1 `fp8_param_gather`.

Gating: `--fp8-recipe mxfp8 --fp8-param-gather` with `--megatron-fsdp-version 2`.
The current rejection in `mcore_fsdp_adapter.py:694` is lifted for this
configuration only.

Reference points: the v1 fp8 machinery this design reuses —
`fp8_quantize()` / `fp8_set_raw_data()` / `fp8_get_raw_data()` /
`fp8_create_transpose_cache()` in `megatron_fsdp/mixed_precision.py`, the
dual-storage buffer in `param_and_grad_buffer.py`
(`is_transpose_buffer=True/False` storages, buckets keyed on
`has_transpose_buffer and bwd`), and the fine-grained param-gather hooks in
`mcore_fsdp_adapter.py:226`. The v2 codebase state is
[PR #6197](https://github.com/NVIDIA/Megatron-LM/pull/6197).

## Background

### Current M-FSDP v2 data flow

From `experimental/parameter_group.py`:

- `main_weight`: flat DBuffer in `main_params_dtype` (fp32 default), optimizer
  placements (sharded). Owned by the optimizer.
- `model_weight`: flat DBuffer in the compute dtype (bf16), parameter
  placements (sharded). Source for the all-gather.
- `_unsharded_model_weight`: replicated bf16 buffer, filled by
  `unshard_parameters()` (all-gather + redistribute) and consumed by compute;
  storage released after forward/backward.
- `main_grad`: flat DBuffer in `main_grads_dtype` (fp32/bf16), lazily allocated
  on the reduce-scatter stream, filled by `reduce_partial_gradients` from bf16
  partial grads. **fp32/bf16 only — fp8 is never used for gradients.**
- `sync_model_weight_from_main_weight()`: after `optimizer.step()`, cast +
  redistribute `main_weight` -> `model_weight`. This is the single refresh
  touch point.
- `experimental/optimizer.py` `fully_shard_optimizer()`: step pre/post hooks
  that (a) temporarily cast sharded grads to the parameter dtype unless the
  optimizer is precision aware, and (b) call `sync_model_weight_from_main_weight`
  after the step.

### v1 fp8 machinery (reuse)

- `mixed_precision.py`: `fp8_quantize()` wraps TE `cast_master_weights_to_fp8`
  (with a fallback) and produces **both** the row-wise and the column-wise
  fp8 weight from the fp32 masters; `fp8_set_raw_data` / `fp8_get_raw_data`
  move fp8 payloads with an optional transpose flag
  (`fp8_need_transpose_data` is True for `MXFP8Tensor`);
  `fp8_create_transpose_cache` / `fp8_discard_transpose_cache` manage the
  backward transpose cache across the unshard/reshard lifecycle;
  `post_all_gather_processing` runs after the gather (CUDA-graph friendly).
- `param_and_grad_buffer.py:2657/2673`: v1 keeps **two independent buffer
  storages** per bucket — `is_transpose_buffer=False` (row-wise, forward) and
  `is_transpose_buffer=True` (column-wise, backward) — and bucket keys select
  the transpose storage for the backward all-gather (`get_bucket_key`).
- `mcore_fsdp_adapter.py:226`: fine-grained param-gather hooks are enabled for
  `fp8_recipe == "mxfp8" and fp8_param_gather`.

### Constraint: NCCL has no fp8 arithmetic collectives

All-gather is a byte-move and works on raw fp8 payloads (v1 already does
this). Reduce-scatter/all-reduce must SUM, so gradients are reduced in
bf16/fp32 and never stored in fp8.

## Design

### Model weights as dual-orientation MXFP8

Replace the bf16 `model_weight` DBuffer with **two quantized weight pairs**
while keeping all DBuffer layout semantics (placements, mesh, tensor shapes):

- `model_weight_fp8_fwd`: (uint8 payload, bf16 scales) — row-wise E4M3
  quantization for the forward GEMM.
- `model_weight_fp8_bwd`: (uint8 payload, bf16 scales) — column-wise E4M3
  quantization (block grid over the other dim) for the backward GEMM.
- `model_weight` (bf16) is no longer allocated in this mode.

Flow changes:

- `sync_model_weight_from_main_weight()`: quantize the fp32 main weights into
  both fp8 orientations (TE `cast_master_weights_to_fp8` when available,
  matching v1 numerics; a torch fallback keeps unit tests dependency-free),
  write payloads + scales. This is the only place fp8 is produced.
- `unshard_parameters()`: all-gather the forward and backward payloads +
  scales, then materialize the unsharded parameter as a TE `MXFP8Tensor`
  (raw fp8 data + block scales), creating the transpose cache per the v1
  lifecycle. TE layers compute directly on the fp8 weights — no explicit
  dequantization in the FSDP layer.
- The optimizer post-step hook and `main_grad` path are unchanged.

Benefits: fp8 GEMM compute with memory/comm parity vs bf16 (2x fp8 payload =
1x bf16), matching v1 `fp8_param_gather`. No change to gradient or optimizer
state precision.

### Non-goals (explicit)

- **No fp8 main gradients.** `main_grad` stays fp32/bf16 (the `main_grads_dtype`
  policy); grads are reduced in bf16 and stored in fp32/bf16.
- **No fp8 optimizer states.** Adam moments stay fp32 in `main_weight`'s
  optimizer buffers.
- No fp8 for activations (that is `--fp8` autocast, still rejected for v2).

### Gating and validation

- Lift the `mcore_fsdp_adapter.py:694` rejection for
  `fp8_recipe == "mxfp8" and fp8_param_gather`; keep rejecting other fp8/fp4
  combinations in v2.
- Unit parity test following the `test_mcore_nd_parallel.py` pattern: M-FSDP
  v2 + mxfp8 vs the ND reference (per-step loss, grad norm, params; 5% loss
  tolerance + strict per-step param snapshots). Kernel tests cover both
  orientations, block independence, saturation, padding, and error bounds.
- 8xH100 proxy convergence run with `--fp8-recipe mxfp8 --fp8-param-gather`,
  compared against the existing bf16 baseline runs; report peak memory,
  all-gather bytes, and fp8 GEMM throughput.

## Implementation status

### Revision 6 (split + Fp8ParameterGroup + DBuffers)

- **Detection-based split, no flag.** `FsdpModule.__init__` groups parameters by
  `is_float8tensor` + `fp8_need_transpose_data`; MXFP8 groups become
  `Fp8ParameterGroup`, everything else stays on the untouched regular path.
  `quantize_model_weight` is gone from `fully_shard` / `FsdpModule.__init__` /
  the adapter.
- **`Fp8ParameterGroup(FsdpParameterGroup)`** owns a `main_weight` DBuffer
  (fp32, shared plumbing) plus **rowwise and colwise uint8 DBuffers** with the
  same shapes/placements. No bf16 `model_weight`, no `_unsharded_model_weight`,
  no bf16 chunk.
- **Quantize** (`sync_model_weight_from_main_weight`): full-size temporary
  `MXFP8Tensor` per fp8 tensor (scale-inverse grids aliased to the real
  tensors' grids), `cast_master_weights_to_fp8` with fragments = flat slices
  of the temp raw data at the shard offsets, then the shard slices are copied
  into the two payload DBuffers and the temps are released.
- **Oriented unshard**: `unshard_parameters(orientation)` — forward
  (`"rowwise"`) gathers and binds only the rowwise payload; backward
  (`"colwise"`) only the column-wise payload — one byte per element per pass,
  mirroring v1's per-pass transpose gather and TE's `fsdp_pre_all_gather`
  usage. The orientation threads through `pre_forward` / `pre_backward`, the
  prefetches, the public API, and the fine-grained hooks (1F1B path).
  Reshard detaches the payloads (`clear_payloads`).
- Both payloads are `(rows, cols)` row-major uint8 (TE's `_columnwise_data`
  has the same shape as `_rowwise_data`; only the block direction and scale
  grid differ). The reference kernels/tests in `quantization.py` follow that
  convention.
- Main gradients and optimizer states stay fp32/bf16.

### Validation status

The end-to-end parity test `tests/unit_tests/distributed/mfsdp_v2/test_mxfp8_v1_parity.py`
**passes on Blackwell (B200, prenyx)**: the same GPT-MoE model built with
MXFP8 primary weights (`--fp8-recipe mxfp8 --fp8-param-gather` with `--fp8`
mode) trains 10 steps under M-FSDP v2 and M-FSDP v1 with per-step losses
matching to ~1e-4 and grad norms to ~1e-4. The fp8 primary weights rest in
different representations on the two sides at rest (v2: fp32 main DTensors;
v1: quantized tensors), so their parity is asserted via losses + grad norms,
while the non-fp8 parameters are compared strictly. The run drove out six
bugs in the v2 fp8 path: TE 2.16 `DType`/`shape`/`device` API differences,
the 1D master-shard contract, non-owned tensors, fp32 main-grad dtype, and
the both-orientations-at-forward requirement.

### Remaining

- PP+EP with the MFSDP adapters is not covered: `_get_dp_tp_mesh` assumes
  world = dp_cp x ep x tp (no PP ranks) — a pre-existing gap unrelated to fp8.
- The one-orientation-per-pass gather is deferred: Megatron's TE layers call
  `update_usage(rowwise=True, columnwise=True)` at forward, so both payloads
  are gathered and bound per unshard.

## Alternatives considered

- **Single orientation + bf16 dequant for compute.** Simpler (one payload, no
  TE dependency) but loses fp8 GEMM compute and diverges from v1's
  dual-orientation semantics; rejected in Revision 1.
- **TE `MXFP8Tensor` inside the DBuffer.** v1's buffers already do this, but
  v2's DBuffer/DTensor path assumes dtype-uniform flat buffers
  (`dbuffer.py:220` enforces one dtype per DBuffer); the (payload, scales)
  pairs keep the layout machinery untouched, and `MXFP8Tensor` is materialized
  only at unshard.
- **Reusing the v1 `DataParallelBuffer` / `DistributedOptimizer` for v2.**
  Rejected by the same reasoning as
  [PR #6186](https://github.com/NVIDIA/Megatron-LM/pull/6186): v2 owns a
  separate DBuffer/DTensor stack.
