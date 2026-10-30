<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# distributed package

This package contains various utilities to finalize model weight gradients
on each rank before the optimizer step. This includes a distributed data
parallelism wrapper to all-reduce or reduce-scatter the gradients across
data-parallel replicas, and a `finalize_model_grads` method to
synchronize gradients across different parallelism modes (e.g., 'tied'
layers on different pipeline stages, or gradients for experts in a MoE on
different ranks due to expert parallelism).

## Configuring Megatron-FSDP

`FullyShardedDataParallelConfig` is independent of
`DistributedDataParallelConfig`. It configures Megatron-FSDP v1 or v2; PyTorch
FSDP continues to use `TorchFullyShardedDataParallelConfig`.

```python
import torch
from megatron.core.distributed import (
    FullyShardedDataParallel,
    FullyShardedDataParallelConfig,
)

fsdp_config = FullyShardedDataParallelConfig(
    version=2,
    data_parallel_sharding_strategy="optim_grads_params",
    main_params_dtype=torch.float32,
    main_grads_dtype=torch.float32,
)
wrapped_model = FullyShardedDataParallel(
    config=transformer_config,
    fsdp_config=fsdp_config,
    module=model,
    pg_collection=pg_collection,
)
```

`version` defaults to 1, preserving the previous default. FSDP-specific fields
omit the former `megatron_fsdp_` or `fsdp_` prefix. For example,
`megatron_fsdp_main_grads_dtype` becomes `main_grads_dtype`, and
`fsdp_double_buffer` becomes `double_buffer`. Existing CLI flags are unchanged.

Model builders accept `data_parallel_config=fsdp_config`; native
`PretrainConfigContainer` instances accept `data_parallel=fsdp_config`. The config
type selects the wrapper. Pass process groups explicitly; streams and other
runtime state remain owned by the FSDP context.

Existing `ddp_config=` wrapper arguments and `ddp` training configuration payloads
remain supported. Do not supply both the old and new inputs. To migrate an
existing configuration explicitly, use
`FullyShardedDataParallelConfig.from_ddp_config(old_config)`.

Validation does not mutate the configuration. The adapter resolves inherited
expert sharding strategies and buffer settings on a copy during construction.
`wrapped_model.fsdp_config` exposes that effective config. The deprecated
`wrapped_model.ddp_config` interface forwards FSDP reads and writes to the same
effective config, including v1's runtime adjustments to overlap and buffering.

