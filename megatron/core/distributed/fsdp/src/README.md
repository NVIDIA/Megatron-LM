<div align="center">

# Megatron-FSDP

</div>

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

</div>

## ✨ What is Megatron-FSDP?

**Megatron-FSDP** is an NVIDIA-developed distributed parallelism library written in native PyTorch that provides a high-performance implementation of **Fully Sharded Data Parallelism (FSDP)**. It offers seamless cross-compatibility with various deep learning frameworks and parallelism libraries such as Megatron-Core, and is performance-optimized to support training and inference of extremely large PyTorch models at data-center scale on NVIDIA GPUs.

For comprehensive information about Megatron-FSDP, refer to: [Megatron-FSDP | Megatron-Core Developer Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/)

### 🧩 Compatibility

- PyTorch **[DeviceMesh](https://docs.pytorch.org/docs/stable/distributed.html#devicemesh)**, **[DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html)**, and **[Distributed Checkpoint (DCP)](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)**
- **[Megatron Core](https://github.com/NVIDIA/Megatron-LM)**
- **[TransformerEngine](https://github.com/NVIDIA/TransformerEngine)**
- **[NVIDIA NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)**

## 📦 Installation

```
pip install megatron-fsdp
```

- PyPI: https://pypi.org/project/megatron-fsdp/
- Source Code: https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src

## 🚀 Quick Start

```python
import torch
from megatron_fsdp import (
    fully_shard_model,
    fully_shard_optimizer,
)

# Initialize Torch Distributed.
torch.distributed.init_process_group()
torch.cuda.set_device(torch.distributed.get_rank())

# Fully-shard the model.
model = torch.nn.Transformer()
fsdp_model = fully_shard_model(
    module=model,
    fsdp_unit_modules=[
        torch.nn.TransformerEncoder,
        torch.nn.TransformerDecoder
    ]
)

# Fully-shard the optimizer.
toy_adam = torch.optim.AdamW(params=fsdp_model.parameters(), lr=0.01)
optimizer = fully_shard_optimizer(optimizer=toy_adam)

# Forward pass.
inp = torch.randn(1, 512, 512).to("cuda")
tgt = torch.randn(1, 512, 512).to("cuda")
output = fsdp_model(inp, inp)

# Backward pass.
torch.nn.functional.mse_loss(output, tgt).backward()

# Optimizer step.
optimizer.step()
optimizer.zero_grad()

# Checkpoint the model and optimizer.
torch.distributed.checkpoint.save({
    "model": fsdp_model.state_dict(),
    "optimizer": optimizer.state_dict(),
}, checkpoint_id="ckpt/")

# Load the saved checkpoint.
ckpt = {
    "model": fsdp_model.state_dict(),
    "optimizer": optimizer.state_dict(),
}
torch.distributed.checkpoint.load(state_dict=ckpt, checkpoint_id="ckpt/")
fsdp_model.load_state_dict(ckpt["model"], strict=False)
optimizer.load_state_dict(ckpt["optimizer"])
```

## ⚙️ `fully_shard` / `MegatronFSDP` API - Advanced Features

Megatron-FSDP's `fully_shard_*` API has a comprehensive set of arguments for fine-tuning your model's performance.

- `fsdp_unit_modules` is a list of sub-module classes or `str` import-paths associated with modules that you want `MegatronFSDP` to fully-shard.
  - Required if `1`, `2`, or `3` are specified as the sharding strategy. Defaults to `None`, in which case Megatron-FSDP will replicate the parameters similar to DDP.
- `zero_dp_strategy` (and `outer_dp_sharding_strategy`) configure different degrees of zero-redundancy data parallelism as described in [ZeRO (Zero Redundancy Optimizer)](https://arxiv.org/abs/1910.02054). It reduces CUDA memory utilization during model training by distributing model parameters, gradients, and optimizer states across multiple devices in the DP `ProcessGroup`, and collectively communicating subsets of parameters and gradients to specific devices when needed for computation or differentiation. More aggressive sharding strategies will entail more communication overhead, with `no_shard` being the least memory efficient but most communication efficient, and `optim_grads_params` being the most memory efficient but least communication efficient. Additionally, `outer_dp_sharding_strategy` supports `no_shard` ([Hybrid-Sharded Data Parallelism (HSDP)](https://arxiv.org/pdf/2304.11277)) and `optim` (`HFSDP` = Fully-Sharded Optimizer State + _HSDP_, requires `zero_dp_strategy='optim_grads_params'`), after specifying the "outer" DP group (`dp_outer_dim` / `hybrid_fsdp_group`).
  - Default: `optim_grads_params` or `3` for `zero_dp_strategy` and `no_shard` or `0` for `outer_dp_sharding_strategy`
  - `0` or `no_shard` implies that your model is not sharded. Similar memory usage to `DDP`.
  - `1` or `optim` implies that your optimizer state is sharded for distributed optimization. Similar to optimizer state sharding in `ZeRO-DP`.
  - `2` or `optim_grads` implies that your optimizer state and gradients are sharded. Similar to `ZeRO-2`.
  - `3` or `optim_grads_params` implies that your optimizer state, gradients, and training parameters are sharded. Similar to `ZeRO-3`.
- `device_mesh` is a [`torch.distributed.DeviceMesh`](https://docs.pytorch.org/docs/stable/distributed.html#devicemesh) that informs `MegatronFSDP` of your distributed environment for sharding in conjunction with hardware configuration and other parallelisms. If not provided, `megatron_fsdp.fully_shard(_model)` will build an FSDP DeviceMesh for you automatically.
  - `dp_shard_dim` is the name of the sub-mesh required for FSDP sharding, and is commonly the flattened combination of the data parallel (DP) and context parallel (CP) sub-meshes.
    - When model parameters are replicated across DP-CP during the backward pass, resultant gradients across DP and CP ranks are reduced simultaneously, normalized by the DP-CP world size. For more information about how ring attention shards the sequence dimension through the attention and non-attention layers of the Transformer, refer to: [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889).
  - `dp_outer_dim` is the name of the sub-mesh corresponding to the "outer" DP group, which is required for replication or sharding in HSDP. `fully_shard` will perform HSDP if `dp_outer_dim` is specified.
  - `tp_dim` is the name of the sub-mesh used for tensor parallelism (TP), which is required for `(FSDP, TP)`-strided sharding when using Megatron-LM or Torch-native `DTensor` TP.
    - For more information about tensor parallelism, refer to: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053).
  - `hybrid_fsdp_group` is the `ProcessGroup` which contains all ranks in the flattened `dp_shard_dim` and `dp_outer_dim` sub-meshes utilized to specify the `(DP-Outer, DP-Shard)` sharded mesh coordinates for the weight and gradient buffers. Required for HSDP.
  - `hybrid_fsdp_expt_group` defines the data-parallel communication group for expert parameters. It is required for HSDP.
- `expt_device_mesh` is another [`torch.distributed.DeviceMesh`](https://docs.pytorch.org/docs/stable/distributed.html#devicemesh) tailored for the expert parallel (EP) modules in `MegatronFSDP`.
  - `dp_shard_dim` is the name of the sub-mesh required for FSDP sharding of the EP modules, enabling expert data parallelism (EDP).
  - `tp_dim` is the name of the sub-mesh used for expert tensor parallelism (ETP), which is required for `(FSDP, ETP)`-strided sharding when using Megatron-LM or Torch-native `DTensor` ETP.
- `init_model_with_meta_device` has `MegatronFSDP` initialize your `meta`-device model in shards on every CUDA device to avoid OOM when initializing extremely large models that cannot fit on a single device. Users can initialize their model on a [`meta`-device](https://docs.pytorch.org/docs/stable/meta.html) (`with torch.device('meta'): ...`), and ``MegatronFSDP`` will further shard and initialize the model parameters layer-by-layer adhering to the customizable `module.reset_parameters` method, which prevents the entire model from being allocated in memory at any point during runtime.
    - Defaults to `False`.
    - Note that the `device` argument which installs your model on a specific device or rank will be deactivated when `init_model_with_meta_device=True`.
- `mixed_precision_policy` takes a `megatron_fsdp.MixedPrecisionPolicy` that configures mixed-precision compute and communication for Megatron-FSDP. Configuration options include:
    - `main_params_dtype` controls the data-type for parameters responsible for distributed checkpointing, distributed optimization, and quantization.
        - Defaults to `torch.float32`.
        - If set to `None`, the native model compute parameter data-type will be utilized.
        - Requires specification (cannot be `None`) when using quantized parameters with Megatron-FSDP.
    - `main_grads_dtype` controls the data-type for gradients used in distributed optimization.
        - Defaults to `None`, in which the model native gradient data-type will be utilized.
        - While `torch.float32` (or higher) is recommended for accuracy at scale, as `main_grads_dtype` controls the data-type for gradient accumulation, `None` is more flexible and uses pre-determined parameter gradient logic in mixed-precision scenarios, such as `BF16` for `FP8`/`FP4` parameters quantized via TransformerEngine.
    - `grad_comm_dtype` controls the data-type for gradient communications when reducing gradients. Lower precision `grad_comm_dtype` improves (communication) performance, but may increase memory utilization or sacrifice gradient precision in certain cases.
        - Defaults to `None`, in which the `main_grads_dtype` data-type will be utilized. No additional memory is allocated when `grad_comm_dtype == main_grads_dtype`.
        - If using HSDP (either DP-Replicate or DP-Outer in `outer_dp_sharding_strategy`), `no_shard`, or `optim`, allocating `dtype`-custom gradient communication buffers may increase per-unit memory overhead, so users should consider the performance-memory trade-off when using this feature.
        - If using NCCL user buffer registration `v2.27+`, gradient reduction may be performed in high-precision depending on the network domain (NVLink or IB), and can enable mixed-precision communication and accumulation, e.g. setting grad_comm_dtype to `BF16` can support `FP32` reduction even though we have `BF16` input and output communication buffers. Otherwise, gradients will be reduced in `grad_comm_dtype` (and accumulated in `main_grads_dtype`) as usual.
- `overlap_grad_reduce` and `overlap_param_gather` will overlap gradient [`reduce-scatter`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reducescatter) and parameter [`all-gather`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather) group communications with backward and forward compute with asynchronous calls and pre-fetching. (In the case of `no_shard`, parameters are not gathered but gradient [`all-reduce`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce) is overlapped.)
    - Both default to `True`.
- `sync_model_each_microbatch` will trigger a `wait` (`MegatronFSDP.finish_grad_sync()`) on gradient reduction, parameter de-allocation, and optimizer parameter / gradient installation (in preparation for `optimizer.step()`) after every forward-backward pass. When using HSDP, parameters and gradients will be all-gathered and reduced respectively on the "outer" DP group each training step instead of each optimization cycle. This behavior is desirable for a transparent and user-friendly sharded training loop where post-backward transformations on the gradient and a clean compute / memory state are necessary within and between training iterations, but damages performance in situations where optimization is delayed (e.g. gradient accumulation) when the communications of the previous training iteration can be overlapped with the compute of the next training iteration. Will also override `is_last_microbatch` / `microbatch_count` logic in `MegatronFSDP`.
    - Defaults to `True` for `fully_shard`, but defaults to `False` when using the `MegatronFSDP` class directly.
    - Can also be controlled with the `MegatronFSDP.sync()` context manager, or through invoking `MegatronFSDP.set_model_auto_sync(bool)`.
    - WARNING: When this synchronization feature is activated in conjunction with `no_shard` / `0` or `optim` / `1` sharding strategies, the user is responsible for calling `MegatronFSDP.zero_grad_buffer()` or `optimizer.zero_grad()` after the subsequent forward-backward pass. This is because un-sharded gradients are all-reduced directly into the gradient accumulation buffer, and this buffer should not be all-reduced more than once per optimization cycle! Analogous to the justification for the [`no_sync()` API for PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync).
- `enable_fine_grained_param_gather` modifies FSDP to all-gather parameters with per-Module granularity instead of collectively unsharding all sub-modules of a unit module in Megatron-FSDP.
    - Defaults to `False`.
- `keep_fp8_transpose_cache` will keep the fp8 transpose cache when using `MegatronFSDP`. This option will cause (number of parameter $\times$ 1 Byte) of memory overhead, but can skip the weight transpose operation in the backward propagation. This feature will not give any benefit from the Blackwell architecture.
    - Defaults to `False`.
- `use_decoupled_grad` installs the reduced gradient into a separate buffer: `Parameter.decoupled_grad`. This buffer is utilized by specific optimizers, such as TransformerEngine's `FusedAdam`, and can be used to temporarily store your gradient for custom `torch.nn.Optimizer`(s).
    - Defaults to `False`.
    - Required for `transformer_engine.pytorch.optimizers.FusedAdam`.
- `nccl_ub` will allocate and register the NCCL userbuffer for param and grad buffers. This option enables an SM-efficient NCCL algorithm that could improve the performance of overlapped computations. This flag will be much more effective when used together with SHARP if the FSDP communication includes both NVL and IB domains. Enabling this option will cause additional memory overhead due to the requirement to enable the `fsdp_double_buffer` option.
    - **Only effective when using with Megatron-Core.**
    - Defaults to `False`.
    - By default we try to use NCCL window (symmetric) registration if it is available. If not it falls back to conventional local registration.
- `fsdp_manual_registration` will manually register the FSDP communication buffers with the NCCL user buffer. For symmetric registration with large models, the registration itself can take a significant amount of time. This option minimizes the number of registration calls to reduce the registration time. However, with this option enabled, you need to manually call the `ParamAndGradBuffer.manual_buffer_registration()` function after the first iteration. This is already implemented in the Megatron-LM training loop. In other use cases, users are expected to call this function themselves. 
    - This is an example of required modification in the training loop.
        ```python
        def train(...):
            ...
            # After the first iteration, user need to call the
            # ParamAndGradBuffer.manual_buffer_registration() function in the training loop
            if (iteration ==  start_iteration + 1):
                if isinstance(model, megatron_FSDP) and model.ddp_config.fsdp_manual_registration:
                    param_and_grad_buffer = getattr(model, "param_and_grad_buffer", None)
                    if param_and_grad_buffer is not None:
                        param_and_grad_buffer.manual_buffer_registration()
        ```
    - **Only effective when using with Megatron-Core.**
    - This option is only effective when `nccl_ub` is enabled.
    - Defaults to `False`, but will be automatically enabled in Megatron-LM.
- `disable_symmetric_registration` will disable NCCL window (i.e. symmetric) registration when using `nccl_ub`. 
    - Defaults to `False`.
- `fsdp_double_buffer` will use persistently allocated double buffers for temporarily-defined memory needed in `MegatronFSDP` communications. Having persistent double buffers may increase peak VRAM utilization, but is required to register NCCL user buffers (`nccl_ub=True`) for `MegatronFSDP`. Currently, this is only supported for simple repetitive model structures such as GPT.
    - Defaults to `False`. Automatically overridden to `True` when `nccl_ub` is enabled.
- `preproc_state_dict_for_dcp_ckpt` adds `model.state_dict()` and `optimizer.state_dict()` post-hooks that modify the model and optimizer state in preparation for `torch.distributed.checkpoint.{save,load}` ([Torch DCP](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)) checkpointing. Specifically, it adds `__create_write_items__` and `__create_chunk_list__` methods to Tensors utilized by Torch DCP to redistribute parameters when saving and loading model and optimizer checkpoints. Can be deactivated should the user need a custom distributed checkpointing strategy.
    - Defaults to `True`.

## 🧮 Using Megatron-FSDP with [`TransformerEngine`](https://github.com/NVIDIA/TransformerEngine)

Megatron-FSDP natively supports mixed-precision activations and parameter sharding in conjunction with [TransformerEngine](https://github.com/NVIDIA/TransformerEngine).

- Within the [`transformer_engine.pytorch.autocast(recipe: transformer_engine.common.recipe.Recipe)`](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.autocast) context, model activations are converted based on the recipe.
- Within the [`transformer_engine.pytorch.quantized_model_init(recipe: transformer_engine.common.recipe.Recipe)`](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.quantized_model_init) context, TransformerEngine native modules (e.g. [`transformer_engine.pytorch.TransformerLayer`](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.TransformerLayer)) have their parameters converted based on the recipe.
    - Requires quantized model activations, i.e. `transformer_engine.pytorch.autocast`.

```python
# FP8 Recipe
fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
    fp8_format=transformer_engine.common.recipe.Format.HYBRID,
)

# Construct TransformerEngine model with FP8 parameters.
with transformer_engine.pytorch.quantized_model_init(
    recipe=fp8_recipe,
    # Needed for FP8 parameters with Megatron-FSDP.
    preserve_high_precision_init_val=True,
):
    te_model = transformer_engine.pytorch.TransformerLayer(...)

# Fully-shard the model.
mfsdp_model = fully_shard_model(
    module=te_model,
    fsdp_unit_modules=[te.pytorch.TransformerLayer],
    # Only FSDP / ZeRO-3 supports FP8 parameters.
    zero_dp_strategy=3,
    # FP32 main weights needed for FP8 parameters.
    mixed_precision_policy=MixedPrecisionPolicy(
        main_params_dtype=torch.float32
    ),
    # Needed for select FP8 recipes.
    keep_fp8_transpose_cache=True,
)

# Evaluate and differentiate the model with FP8 activations.
with transformer_engine.pytorch.autocast(recipe=fp8_recipe):
    mfsdp_model(x).sum().backward()
```

ℹ️ `TransformerEngine` kernels have various constraints related to quantized Tensors, such as using fused QKV parameters or defining activations and parameters with shapes compatible to CuBLAS kernels on supported hardware from NVIDIA. To properly initialize `TransformerLayer`, you can refer to the example model used in our unit tests: `Megatron-LM/tests/unit_tests/distributed/fsdp/test_mfsdp_fully_shard.py::TestMegatronFsdpFullyShard::test_fully_shard_te_quantized`.