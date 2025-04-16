# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Optional, Set

import numpy as np
import torch

from megatron.core.device_utils import (
    get_current_device_type, 
    get_xla_model, 
    get_xla_runtime, 
    get_xla_spmd
)
from megatron.core.wrapped_process_group import WrappedProcessGroup

try:
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        SpmdFullyShardedDataParallel as fully_shard
    )
    HAVE_FSDP = True
except ImportError:
    try:
        from torch.distributed import DeviceMesh
        from torch.distributed._composable.fsdp import fully_shard

        HAVE_FSDP = True
    except ImportError:
        HAVE_FSDP = False

from megatron.core.fp8_utils import is_float8tensor

from .. import parallel_state, tensor_parallel
from ..models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from ..models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from ..transformer.transformer_config import TransformerConfig
from ..transformer.transformer_layer import TransformerLayer
from .data_parallel_base import _BaseDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig

xm = get_xla_model()
xr = get_xla_runtime()
xs = get_xla_spmd()

class TorchFullyShardedDataParallel(_BaseDataParallel):
    """
    Enables fully sharded data parallelism by wrapping the given model with
    the PyTorch FSDP2 API:
    https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md
    To utilize this class, PyTorch version >= 2.4.0 is required.

    Args:
        config: Transformer config object.
        ddp_config: DistributedDataParallel config object.
        module: Underlying model.
        sub_modules_to_wrap: Set of sub_modules to shard with FSDP.
            Parameters within each sub_module will be all-gathered just-in-time.
            The default set includes the following submodules derived from the
            GPT model architecture:
                TransformerLayer (all Transformer layers)
                LanguageModelEmbedding (initial embedding layer)
                RotaryEmbedding  (initial RoPE layer)
                tensor_parallel.ColumnParallelLinear (final output layer)

            User can set _fsdp_modules attribute on submodules to set additional
            submodules to shard with FSDP.
        process_group: Optional ProcessGroup to use for distributed operations.
            If None (default), the data parallel process group will be obtained from
            parallel_state.get_data_parallel_group(with_context_parallel=True).
    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        sub_modules_to_wrap: Set[torch.nn.Module] = {
            TransformerLayer,
            LanguageModelEmbedding,
            RotaryEmbedding,
            tensor_parallel.ColumnParallelLinear,
        },
        group: Optional[WrappedProcessGroup] = None,
    ):

        assert (
            HAVE_FSDP
        ), 'TorchFullyShardedDataParallel requires PyTorch >= 2.4.0 with FSDP 2 support.'

        super().__init__(config=config, module=module)

        if group is None:
            self.group = parallel_state.get_data_parallel_group(with_context_parallel=True, wrapped=True)
        else:
            self.group = group

        if xm:
            # Define the mesh following common SPMD practice
            num_devices = xr.global_runtime_device_count()
            device_ids = np.array(range(num_devices))
            mesh_shape = (len(self.group.rank_groups[0]), len(self.group.rank_groups))
            # To be noted, the mesh must have an axis named 'fsdp', 
            # # which the weights and activations will be sharded on.
            self.mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'model'))
            kwargs = {"mesh": self.mesh}
        else:
            self.mesh = DeviceMesh.from_group(self.group.process_group, get_current_device_type())
            kwargs = {"mesh": self.mesh}

        def save_custom_attrs(module):
            custom_attrs = {}
            for name, param in module.named_parameters():
                attrs = vars(param)
                if is_float8tensor(param):
                    # disable fp8 transpose cache and perform transposing fp8 weights
                    # at each micro-batch because torch-FSDP doesn't recognize the
                    # micro-batch id, thus removing unnecessary memory stores
                    attrs['_fp8_attrs']['transpose_invalid'] = False
                    del attrs['_fp8_attrs']['transpose']
                custom_attrs[name] = {k: v for k, v in attrs.items()}
            return custom_attrs

        def restore_custom_attrs(module, custom_attrs):
            for name, param in module.named_parameters():
                if name in custom_attrs:
                    for attr_name, attr_value in custom_attrs[name].items():
                        setattr(param, attr_name, attr_value)

        # Save the custom attributes on Parameters before FSDP overwrites them.
        # See https://github.com/pytorch/pytorch/issues/136929.
        attrs = save_custom_attrs(self.module)

        sub_modules_to_wrap = set(sub_modules_to_wrap)
        for sub_module in self.module.modules():
            fsdp_modules = getattr(sub_module, "_fsdp_modules", [])
            for f in fsdp_modules:
                sub_modules_to_wrap.add(f)

        prev_module = None
        for sub_module in self.module.modules():
            # Wrap individual submodules to fetch parameters just-in-time rather than
            # conservatively fetching all parameters at the start of each iteration.
            # See https://github.com/pytorch/pytorch/issues/114299.
            if any(
                isinstance(sub_module, sub_module_to_wrap)
                for sub_module_to_wrap in sub_modules_to_wrap
            ):
                fully_shard(sub_module, **kwargs)

                # Explicitly set the FSDP backward prefetch schedule to prevent activation
                # recomputation from disrupting the automatically generated default schedule.
                if config.recompute_granularity is not None:
                    sub_module.set_modules_to_backward_prefetch(
                        [prev_module] if prev_module else []
                    )
                prev_module = sub_module

        # Wrap the root module as required by the FSDP API.
        # See https://github.com/pytorch/pytorch/issues/114299.
        fully_shard(self.module, **kwargs)

        restore_custom_attrs(self.module, attrs)

    def load_state_dict(self, state_dict, strict=True):
        """
        No-op because tensors are already loaded in-place by
        `_load_base_checkpoint` with FSDP2."""
        pass
