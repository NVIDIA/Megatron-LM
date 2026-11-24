# GDN recurrence activation offloading

`gdn_core_attn` offloads tensors saved by the BF16 FLA GatedDeltaNet recurrence
using the existing scheduler, transfer streams and pinned-memory pool. It is
opt-in and trades device memory for host-pinned storage and copy traffic.
Numerical kernels are unchanged.

## Usage

Add these arguments to an existing GDN training launch:

```bash
export NVTE_CPU_OFFLOAD_V1=1
# Append to the training arguments:
--bf16 --experimental-attention-variant gdn \
--fine-grained-activation-offloading --offload-modules gdn_core_attn
```

Programmatic configurations use `fine_grained_activation_offloading=True`,
`offload_modules=["gdn_core_attn"]` and `gdn_kernel_backend="fla"`.
`min_offloaded_tensor_size` is measured in elements (default 1,048,576).
`activation_offload_fraction` selects a fraction from 0 to 1; zero selects no
copies. The existing warmup policy and last-group margin apply. Use multiple
GDN layers when evaluating steady-state savings.

## Supported scope

The scope wraps only the FLA recurrence and commits before output normalization.
A differentiable recurrence input anchors backward prefetch, including a trainable
gate when Q is frozen. Evaluation, no-grad execution and fully frozen recurrences
bypass the scope. Inputs shared with neighboring operations are not force-released.

BF16 GDN, selective `gdn_norm_out` recomputation and native pre-GDR fusion are
supported. Configuration rejects GDN2, FP8/NVFP4, other recurrence backends, full
recomputation and CUDA graphs. Projection, convolution and output normalization
remain outside this scope. Projection/convolution recomputation and convolution
input offload are separate work in [#7852](https://github.com/NVIDIA/Megatron-LM/pull/7852).

## Validation

`tests/unit_tests/ssm/test_gated_delta_net_offloading.py` covers fractions,
thresholds, packed sequences, frozen inputs and pool reuse.
`tests/unit_tests/ssm/test_gdn_offload_training.py` compares native decoder training
with loss, gradients, weights and complete optimizer state, including pinned
ownership before schedule reset.

Historical Qwen measurements at `1f2c9ae4f7e617b12caa115e387d95e668517c62`
reduced peak allocated memory by 952 MiB on A6000, with step-time and pinned-memory
costs. They do not qualify the current revision. The [versioned evidence](https://github.com/WEIXILIN0307/Megatron-LM/tree/5af996df67ef2483242ea25be876b996d745449c)
retains commands, original results and experimental profiling/training tools.
