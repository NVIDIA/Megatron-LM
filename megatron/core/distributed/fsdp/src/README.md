<div align="center">

# 🚀 Megatron-FSDP

</div>

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

</div>

## ✨ What is Megatron-FSDP?

**Megatron-FSDP** is an NVIDIA-developed PyTorch extension that provides a high-performance implementation of Fully Sharded Data Parallelism (FSDP). It offers seamless cross-compatibility with major deep learning frameworks and parallelism libraries, making it easy to scale your PyTorch models across multiple GPUs and nodes.

Megatron-FSDP can provide up to 25% speed up and 23% memory savings compared to FSDP2.

### Compatibility

- **[PyTorch DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html)**
- **[Megatron Core](https://github.com/NVIDIA/Megatron-LM)**
- **[TransformerEngine](https://github.com/NVIDIA/TransformerEngine)**

## ✨ Features

- **Easy Integration**: Simple `fully_shard` function for quick model parallelization
- **High Performance**: Optimized for NVIDIA GPUs with efficient memory management
- **Cross-Framework**: Works seamlessly with PyTorch, Huggingface Transformers, Megatron-LM, Megatron Bridge and TransformerEngine
- **Scalable**: Supports both single-node multi-GPU and multi-node distributed training
- **Flexible Configuration**: Configurable sharding strategies and process groups

## ⚡ Optimizations

- **Advanced Bucketing**: Data-type aware bucketing system to minimize the overhead of collective operations
- **Buffer Management**: Zero copy communication is achieved by reorganizing the storage of parameters and main grad with `ParamAndGradBuffer` class
- **Communication Overlapping**: Improved communication overlap of paramter all-gather and gradient reduce-scatter
- **FP8 Mixed Precision with Transformer Engine**: Compatibility with Transformer Engine enables efficient FP8 mixed precision training
- **Gradient accumulate fusion support with Transformer Engine**: Remove the explicit gradient copy to the communication buffer in backwards pass

### Advanced Collective Communication
- **SM Usage Reduction with SHARP**: FSDP's `All-Gather` (AG) and `Reduce-Scatter` (RS) collectives are designed to overlap with compute kernels. However, standard NCCL communication kernels can consume a significant number of GPU SMs (e.g., 16-32 SMs), "stealing" resources from compute (GEMM) kernels and reducing overall TFLOPS.
- **In-Switch Processing**: We leverage **SHARP** (Scalable Hierarchical Aggregation and Reduction Protocol) to offload these collective operations. SHARP performs aggregation and reduction computations directly on the network switches (InfiniBand or NVLink Switch) instead of on the GPU SMs. This dramatically reduces the SM consumption for communication to **1-6 SM** freeing up GPU resources for compute. It also provides lower communication latency, especially in large, scaled-out workloads.
- **Symmetric Optimizations for MNNVL**: We support **symmetric-based optimizations**, introduced in NCCL v2.27, which enable switch offloading for **Multi-Node NVLink (MNNVL)** systems such as GB200/GB300. This allows the same SM-saving benefits over the high-bandwidth NVLink fabric itself.
- **Hierarchical Collectives**: When an FSDP sharding domain spans both NVLink and InfiniBand, the library utilizes **hierarchical SHARP collectives** (e.g., NVL-SHARP + IB-SHARP) to optimize the communication path across the entire system topology.
<!-- ## 📊 Performance  -->

## 📦 Installation

```
pip install megatron-fsdp
```

- PyPI: https://pypi.org/project/megatron-fsdp/
- Source Code: https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src

## 🚀 Quick Start

### Basic Usage

Transform your PyTorch model to use Fully Sharded Data Parallelism with just a few lines:

```python
import torch
from megatron_fsdp import fully_shard

# Your existing model and optimizer
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Enable FSDP with Megatron-FSDP.
# Alternatively, you can use fully_shard_model() followed by fully_shard_optimizer()!
model, optimizer = fully_shard(
    model,
    optimizer,
    device_mesh=device_mesh, # Your global DeviceMesh.
    dp_shard_dim="dp_shard_cp", # Sharding across the flattened DP-CP mesh.
    fsdp_unit_modules=[YourTransformerBlock], # Modules to shard.
)

# Your model is now ready for distributed training!
```

### Comparison with FSDP-2

`fully_shard` / `fully_shard_model` / `fully_shard_optimizer` are simple entrypoints into `MegatronFSDP`.

- No need to call `fully_shard` on all the sub-modules, just pass your sub-module classes or import paths to `fully_shard`!
- One liner for the sharding change, which seamlessly preserves the identity of your training loop.

Compare this with FSDP2:

```python
import torch
from torch.distributed.fsdp import fully_shard

# Your existing model and optimizer
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Enable FSDP with FSDP2
for module in model.modules():
    if isinstance(module, YourTransformerBlock): # Sub-Modules to shard
        fully_shard(module)
fully_shard(model)

# Your model is now ready for distributed training!
```

## `fully_shard` / `MegatronFSDP` API - Advanced Features

```python
import torch
from megatron_fsdp import fully_shard

# Initialize DeviceMesh.
device_mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    mesh_shape=(dp_outer_size, dp_shard_size, cp_size, tp_size),
    mesh_dim_names=("dp_outer", "dp_shard", "cp", "tp"),
)
# Only relevant when using HSDP, where we also need the full DP group for data parallelism,
# This sub-mesh can be provided to distributed samplers or dataloaders.
device_mesh[("dp_outer", "dp_shard")]._flatten("dp")
# Only required if using CP. Otherwise, just pass dp_shard to FSDP.
device_mesh[("dp_shard", "cp")]._flatten("dp_shard_cp")
# Only required if using HSDP. Otherwise, don't pass hybrid_fsdp_group.
device_mesh[("dp_outer", "dp_shard", "cp")]._flatten("hsdp")
hsdp_group = device_mesh["hsdp"].get_group()
# Initialize DeviceMesh for expert parallel (EP) modules when using FSDP + EP.
expert_device_mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    mesh_shape=(expt_dp_shard_size, expt_tp_size),
    mesh_dim_names=("dp_shard", "tp"),
)

# Fully-shards your model and distributes your optimizer.
model, optimizer = fully_shard(
    # PyTorch (Root) Module
    model,
    # PyTorch Optimizer
    optimizer,
    # Device Mesh
    device_mesh=device_mesh
    # Always required for FSDP or HSDP.
    dp_shard_dim="dp_shard_cp",
    # Set this required argument to use HSDP instead of FSDP. Otherwise, set this to None.
    dp_outer_dim="dp_outer",
    # Only required for TP-sensitive models (i.e. Megatron-LM / TransformerEngine) or when using DTensor-based TP.
    # Otherwise, set this to None.
    tp_dim="tp",
    # Only required when using HSDP. Otherwise, set this to None.
    hybrid_fsdp_group=hsdp_group,
    # Only required for FSDP + EP. Otherwise, set this to None.
    expt_device_mesh=expt_device_mesh,
    # FSDP Sharding Strategy: no_shard (0) / optim (1) / optim_grads (2) / optim_grads_params (3)
    zero_dp_strategy=3,
    outer_dp_sharding_strategy=1,
    # Sharded Modules
    fsdp_unit_modules=[...],
    # Initialize the model on devices in shards to avoid OOM. Requires device("meta")-init for model.
    init_model_with_meta_device=True,
    # Reduce gradients in FP32.
    grad_reduce_in_fp32=False,
    # Store distributed optimization state in FP32.
    preserve_fp32_weights=True,
    # Sync parameters and gradients each step. Allows for gradient transformations after backward pass,
    # and synchronizes parameters and gradients across HSDP groups, but deactivates compute-communication
    # overlap going into the subsequent training step.
    sync_model_each_microbatch=True,
    # Preprocess state dict for DCP checkpointing. Required for Torch Distributed Checkpoint.
    preproc_state_dict_for_dcp_ckpt=True,
)

# Save model and optimizer state.
torch.distributed.checkpoint.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_id=str(CKPT_DIR))

# Load model and optimizer state.
ckpt_state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
torch.distributed.checkpoint.load(state_dict=ckpt_state_dict, checkpoint_id=str(CKPT_DIR))
# model.load_state_dict(strict=False) is only necessary to ignore TE FP8 extra state
# that is missing from the DCP checkpoint but present in TEBaseModule.
# Megatron-FSDP does not support TE FP8 extra state checkpointing with DCP.
model.load_state_dict(ckpt_state_dict["model"], strict=False)
optimizer.load_state_dict(ckpt_state_dict["optimizer"])
```

- `zero_dp_strategy` (and `outer_dp_sharding_strategy`) configure different degrees of zero-redundancy data parallelism as described in [ZeRO (Zero Redundancy Optimizer)](https://arxiv.org/abs/1910.02054). It reduces CUDA memory utilization during model training by distributing model parameters, gradients, and optimizer states across multiple devices in the DP `ProcessGroup`, and collectively communicating subsets of parameters and gradients to specific devices when needed for computation or differentiation. More aggressive sharding strategies will entail more communication overhead, with `no_shard` being the least memory efficient but most communication efficient, and `optim_grads_params` being the most memory efficient but least communication efficient. `outer_dp_sharding_strategy` has the same options, except for the (required) "outer" DP group (`dp_outer_dim` / `hybrid_fsdp_group`) when using [Hybrid-Sharded Data Parallelism (HSDP)](https://arxiv.org/pdf/2304.11277), and only `no_shard` (DP Replication) and `optim` (Optimizer State Hybrid Sharding, requires `zero_dp_strategy='optim_grads_params`) are supported.
  - Default: `optim_grads_params` or `3` for `zero_dp_strategy` and `no_shard` or `0` for `outer_dp_sharding_strategy`
  - `0` or `no_shard` implies that your model is not sharded. Similar memory usage to `DDP`.
  - `1` or `optim` implies that your optimizer state is sharded for distributed optimization. Similar to optimizer state sharding in `ZeRO-DP`.
  - `2` or `optim_grads` implies that your optimizer state and gradients are sharded. Similar to `ZeRO-2`.
  - `3` or `optim_grads_params` implies that your optimizer state, gradients, and training parameters are sharded. Similar to `ZeRO-3`.
- `fsdp_unit_modules` is a list of sub-module classes or `str` import-paths associated with modules that you want `MegatronFSDP` to fully-shard.
  - Required if `1`, `2`, or `3` are specified as the sharding strategy. Defaults to `None`.
- `device_mesh` is a [`torch.distributed.DeviceMesh`](https://docs.pytorch.org/docs/stable/distributed.html#devicemesh) that informs `MegatronFSDP` of your distributed environment for sharding in conjunction with hardware configuration and other parallelisms.
  - `dp_shard_dim` is the name of the sub-mesh required for FSDP sharding, and is commonly the flattened combination of the data parallel (DP) and context parallel (CP) sub-meshes.
    - When model parameters are replicated across DP-CP during the backward pass, resultant gradients across DP and CP ranks are reduced simultaneously, normalized by the DP-CP world size. For more information about how ring attention shards the sequence dimension through the attention and non-attention layers of the Transformer, refer to: [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889).
  - `dp_outer_dim` is the name of the sub-mesh corresponding to the "outer" DP group, which is required for replication or sharding in HSDP. `fully_shard` will perform HSDP if `dp_outer_dim` is specified.
  - `tp_dim` is the name of the sub-mesh used for tensor parallelism (TP), which is required for `(FSDP, TP)`-strided sharding when using Megatron-LM or Torch-native `DTensor` TP.
    - For more information about tensor parallelism, refer to: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053).
  - `hybrid_fsdp_group` is the `ProcessGroup` which contains all ranks in the flattened `dp_shard_dim` and `dp_outer_dim` sub-meshes utilized to specify the `(DP-Outer, DP-Shard)` sharded coordinate system for the weight and gradient buffers. Required for HSDP.
- `expt_device_mesh` is another [`torch.distributed.DeviceMesh`](https://docs.pytorch.org/docs/stable/distributed.html#devicemesh) tailored for the expert parallel (EP) modules in `MegatronFSDP`.
  - `dp_shard_dim` is the name of the sub-mesh required for FSDP sharding of the EP modules, enabling expert data parallelism (EDP).
  - `tp_dim` is the name of the sub-mesh used for expert tensor parallelism (ETP), which is required for `(FSDP, ETP)`-strided sharding when using Megatron-LM or Torch-native `DTensor` ETP.
- `init_model_with_meta_device` has `MegatronFSDP` initialize your `meta`-device model in shards on every CUDA device to avoid OOM when initializing extremely large models that cannot fit on a single device. Users can initialize their model on a [`meta`-device](https://docs.pytorch.org/docs/stable/meta.html) (`with torch.device('meta'): ...`), and ``MegatronFSDP`` will further shard and initialize the model parameters layer-by-layer adhering to the customizable `module.reset_parameters` method, which prevents the entire model from being allocated in memory at any point during runtime.
    - Defaults to `False`.
    - Note that the `device` argument which installs your model on a specific device or rank will be deactivated when `init_model_with_meta_device=True`.
- `grad_reduce_in_fp32` will reduce gradients in `FP32` precision (in contrast to the lower `BF16` or `FP8` model training precision).
    - Defaults to `False`.
    - `torch.distributed.fsdp.MixedPrecisionPolicy` will be supported in the near future.
- `preserve_fp32_weights` will preserve a `FP32` precision version of model parameters utilized for optimization.
    - Defaults to `True`.
    - `torch.distributed.fsdp.MixedPrecisionPolicy` will be supported in the near future.
- `overlap_grad_reduce` and `overlap_param_gather` will overlap gradient [`reduce-scatter`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reducescatter) and parameter [`all-gather`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather) group communications with backward and forward compute with asynchronous calls and pre-fetching. (In the case of `no_shard`, parameters are not gathered but gradient [`all-reduce`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce) is overlapped.)
    - Both default to `True`.
- `sync_model_each_microbatch` will trigger a `wait` (`MegatronFSDP.finish_grad_sync()`) on gradient reduction, parameter de-allocation, and optimizer parameter / gradient installation (in preparation for `optimizer.step()`) after every forward-backward pass. When using HSDP, parameters and gradients will be all-gathered and reduced respectively on the "outer" DP group each training step instead of each optimization cycle. This behavior is desirable for a transparent and user-friendly sharded training loop where post-backward transformations on the gradient and a clean compute / memory state are necessary between training iterations, but damages performance in situations where optimization is delayed (e.g. gradient accumulation) where the communications of the previous training iteration can be overlapped with the compute of the next training iteration. Will also override `is_last_microbatch` / `microbatch_count` logic in `MegatronFSDP`.
    - Defaults to `True` for `fully_shard`, but defaults to `False` when using the `MegatronFSDP` class directly.
- `keep_fp8_transpose_cache_when_using_custom_fsdp` will keep the fp8 transpose cache when using `MegatronFSDP`. This option will cause (number of parameter $\times$ 1 Byte) of memory overhead, but can skip the weight transpose operation in the backward propagation. This feature will not give any benefit from the Blackwell architecture.
    - **Only effective when using Megatron-LM.**
    - Defaults to `False`.
- `nccl_ub` will allocate and register the NCCL userbuffer for param and grad buffers. This option enables an SM-efficient NCCL algorithm that could improve the performance of overlapped computations. This flag will be much more effective when used together with SHARP if the FSDP communication includes both NVL and IB domains. Enabling this option will cause additional memory overhead due to the requirement to enable the `fsdp_double_buffer` option.
    - **Only effective when using Megatron-LM.**
    - Defaults to `False`.
    - By default we try to use NCCL window (symmetric) registration if it is available. If not it falls back to conventional local registraion.
- `disable_symmetric_registration` will disable NCCL window (i.e. symmetric) registraion when using `nccl_ub`. 
    - Dafaults to `False`.
- `fsdp_double_buffer` will use persistently allocated double buffers for temporarily-defined memory needed in `MegatronFSDP` communications. Having persistent double buffers may increase peak VRAM utilization, but is required to register NCCL user buffers (`nccl_ub=True`) for `MegatronFSDP`. Currently, this is only supported for simple repetitive model structures such as GPT.
    - **Only effective when using Megatron-LM.**
    - Defaults to `False`. Automatically overridden to `True` when `nccl_ub` is enabled.
- `preproc_state_dict_for_dcp_ckpt` adds `model.state_dict()` and `optimizer.state_dict()` post-hooks that modify the model and optimizer state in preparation for `torch.distributed.checkpoint.{save,load}` ([Torch DCP](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)) checkpointing. Specifically, it adds `__create_write_items__` and `__create_chunk_list__` methods to Tensors utilized by Torch DCP to redistribute parameters when saving and loading model and optimizer checkpoints. Can be deactivated should the user need a custom distributed checkpointing strategy.
    - Defaults to `True`.
