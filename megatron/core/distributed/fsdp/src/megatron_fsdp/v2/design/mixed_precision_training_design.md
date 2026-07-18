# Mixed precision training support in Megatron FSDP v2

Megatron FSDP v2 uses `MixedPrecisionPolicy` to describe how parameters,
optimizer weights, and gradients are stored in the `fully_shard()` path.

The policy keeps the FSDP runtime independent of specific tensor formats. FSDP
owns buffer allocation, sharding, unshard/reshard, and gradient reduction;
`MixedPrecisionPolicy` owns dtype- and tensor-format-specific decisions.

## What the policy manages

`MixedPrecisionPolicy` controls:

- model-init contexts for quantized parameter formats;
- how parameters are grouped into compatible `ParameterGroup`s;
- model-weight buffer dtype and storage shape;
- extraction of raw parameter storage into FSDP buffers;
- rebinding a parameter to an unsharded FSDP buffer view;
- post-unshard and post-reshard tensor-format hooks;
- main-weight and main-gradient dtypes;
- copying or quantizing optimizer main weights back to model weights.

The MCore FSDP adapter builds the policy from DDP and transformer config fields,
including FP8 and FP4 parameter-gather settings.

## Buffer roles

Each `ParameterGroup` may own several `DataParallelBuffer`s:

| Buffer | Purpose |
| --- | --- |
| `model_weight_buffer` | Compute-weight storage used by FSDP unshard/all-gather. Dense params use their model dtype; FP8/NVFP4 params use raw `uint8` storage. |
| `transpose_weight_buffer` | Optional extra weight buffer for formats that need a backward/columnwise payload, currently MXFP8. |
| `main_weight_buffer` | Optimizer-facing high-precision weights. Created only when the policy asks for a separate main-weight dtype/layout. |
| `main_grad_buffer` | Optimizer-facing gradients. Its dtype comes from `main_grads_dtype`, `main_params_dtype`, or the parameter dtype depending on policy settings. |

When a separate `main_weight_buffer` would be redundant (same dtype and same
sharding layout as `model_weight_buffer`), `ParameterGroup._init_buffers()`
skips it and lets the optimizer mutate model-weight storage directly.
`copy_main_weights_to_model_weights()` still marks replicated HSDP storage
dirty in this case, because an outer optimizer shard was updated even though
there is no separate tensor payload to copy. The next unshard then refreshes
the outer replicas. Quantized parameters require a separate high-precision
main-weight buffer.

## Parameter grouping

One `ParameterGroup` must contain parameters with the same storage kind.

`group_key_dtype()` returns:

- the tensor dtype for ordinary dense parameters;
- `("quantized", type(tensor).__name__, recipe)` for FP8 or NVFP4 parameters.

This prevents dense, FP8, and NVFP4 parameters from sharing a buffer layout or
post-processing path.

## Dense BF16/FP32 path

For ordinary dense parameters:

1. `model_weight_buffer` stores the parameter data in the model dtype.
2. `main_weight_buffer` is optional. It is used when the optimizer should keep a
   different dtype or sharding layout, commonly FP32 main weights for BF16
   model weights.
3. `main_grad_buffer` stores optimizer-facing gradients.
4. After optimizer step, `copy_main_weights_to_model_weights()` copies main
   weights back to model weights when a separate main buffer exists.

No tensor-format-specific post-processing is required.

## FP8 path

FP8 support is configured with `FullyShardFP8Policy`.

When enabled, `model_init_context()` enters Transformer Engine's quantized
model-init context and asks TE to preserve a high-precision initialization value
when available.

For FP8 parameters:

- `model_weight_buffer_dtype()` returns `torch.uint8`;
- `get_param_data()` extracts TE's raw FP8 payload;
- `bind_unsharded_param()` rebinds TE raw payload fields to FSDP all-gathered
  buffer views;
- `post_unshard()` rebuilds TE recipe-specific state when needed;
- `post_reshard()` invalidates temporary transpose/cache views unless the FP8
  policy asks to keep them;
- `copy_main_weights_to_model_weights()` casts high-precision main weights back
  into FP8 model storage.

MXFP8 may use `transpose_weight_buffer` for the backward/columnwise payload.

## NVFP4 path

NVFP4 support is configured with `FullyShardNVFP4Policy`.

When enabled, `model_init_context()` enters Transformer Engine's quantized
model-init context with an `NVFP4BlockScaling()` recipe. FP8 and NVFP4 are
mutually exclusive.

NVFP4 parameters use packed rowwise `uint8` model storage: two logical FP4
values per byte. Optimizer weights and gradients remain in full logical shape.

| Buffer | Storage | Shape domain |
| --- | --- | --- |
| `model_weight_buffer` | packed `torch.uint8` | NVFP4 packed shape (`last_dim / 2`) |
| `main_weight_buffer` | usually FP32 | full logical parameter shape |
| `main_grad_buffer` | usually FP32 | full logical parameter shape |

`DataParallelBuffer` starts from logical shapes for every buffer. For NVFP4
model-weight buffers, it then compacts the index:

```python
self.buffer_index.compact(0.5, mp_policy.get_param_storage_shapes(params))
```

This keeps parameter ordering and proportional offsets aligned with the
full-shape buffers while reducing model-weight storage to the packed NVFP4
size.

After optimizer step, `copy_main_weights_to_model_weights()` dispatches NVFP4
groups to `quantize_main_weights_to_nvfp4()`, which calls Transformer Engine's
`quantize_master_weights()`.

The quantization start offsets are derived from `main_weight_buffer`, not
`model_weight_buffer`, because TE expects logical-element offsets rather than
packed-byte offsets.

## Main gradient dtype and decoupled gradients

`main_grads_dtype_for_param()` resolves the main-gradient dtype in this order:

1. explicit `main_grads_dtype`;
2. if `use_decoupled_grad=False`, the main-parameter dtype;
3. otherwise the parameter dtype.

When `use_decoupled_grad=True`, FSDP installs reduced gradients on
`dist_param.decoupled_grad` instead of `dist_param.grad`. The mixed-precision
policy only selects storage dtype; `ParameterGroup` and `FSDPModule` own the
actual gradient installation and reduction lifecycle.

## Current limitations

- FP8 and NVFP4 cannot be enabled at the same time.
- NVFP4 requires Transformer Engine support for `NVFP4Tensor`,
  `NVFP4BlockScaling`, and `quantize_master_weights()`.
- HSDP outer optimizer sharding is not supported for NVFP4.
- The non-distributed NVFP4 quantization path is not implemented; NVFP4
  quantization currently expects an inner-sharded model-weight buffer.
- NVFP4 uses rowwise packed storage only; there is no transpose-weight buffer
  or transpose-cache management path like MXFP8.

## Relevant files

| File | Role |
| --- | --- |
| `mixed_precision.py` | Defines `MixedPrecisionPolicy`, `FullyShardFP8Policy`, `FullyShardNVFP4Policy`, tensor-format hooks, and main-to-model weight updates. |
| `param_group.py` | Creates model/main/grad buffers and delegates dtype/storage behavior to the policy. |
| `dp_buffer.py` | Owns buffer layout and compacts NVFP4 model-weight buffer indices. |
| `mcore_fsdp_adapter.py` | Builds the policy from DDP and transformer config fields. |
| `lazy_grad_buffer_design.md` | Documents lazy `main_grad_buffer` allocation and release. |
