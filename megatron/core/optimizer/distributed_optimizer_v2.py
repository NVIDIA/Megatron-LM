import torch
from typing import Any, Callable, List, Optional
from logging import getLogger
import copy

from ..config_logger import has_config_logger_enabled, log_config_to_disk
from .optimizer_config import OptimizerConfig
from .optimizer import MixedPrecisionOptimizer
from .grad_scaler import MegatronGradScaler
from ..transformer.module import MegatronModule
from ..transformer.fsdp_dtensor_checkpoint import handle_experts_in_state_dict

logger = getLogger(__name__)


class DistributedOptimizer_V2(MixedPrecisionOptimizer):
    """
    DTensor-based distributed optimizer implementation for Megatron-LM.

    This optimizer extends MixedPrecisionOptimizer to provide distributed training capabilities
    using DTensor for parameter and optimizer state management across multiple devices.

    Args:
        optimizer (torch.optim.Optimizer): The base PyTorch optimizer to wrap.
        config (OptimizerConfig): Configuration object containing optimizer settings.
        grad_scaler (MegatronGradScaler): Gradient scaler for mixed precision training.
        init_state_fn (Optional[Callable]): Optional function to initialize optimizer states.
        model_chunks (List[MegatronModule]): List of model chunks to optimize. Must be non-empty.
            All chunks must share the same DDP configuration.

    Raises:
        ValueError: If model_chunks is empty or if model chunks have different DDP configurations.

    Attributes:
        model_chunks (List[MegatronModule]): The list of model chunks being optimized.
        ddp_config: The DDP configuration shared across all model chunks.
        param_to_name (dict): Cached mapping from parameters to their names (lazy-initialized).

    Methods:
        state_dict(): Returns a state dictionary with globally unique parameter names.
        load_state_dict(state_dict): Loads optimizer state from a state dictionary.
        zero_grad(set_to_none): Zeros out gradients for all parameters.
        
    Notes:
        - This optimizer requires Megatron-LM's FSDP or DDP configuration.
        - Compatible with distributed checkpointing for saving and loading optimizer states.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: MegatronGradScaler,
        init_state_fn: Optional[Callable],
        model_chunks: List[MegatronModule],
    ):
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        super().__init__(optimizer, config, grad_scaler, init_state_fn)
        self.model_chunks = model_chunks
        if not self.model_chunks:
            raise ValueError(
                "DistributedOptimizer_V2 requires a non-empty 'model_chunks' list."
            )
        self.ddp_config = self.model_chunks[0].ddp_config
        for model_chunk in self.model_chunks:
            assert self.ddp_config == model_chunk.ddp_config

    def state_dict(self):
        # Generate global unique named optimizer states
        named_state = {
            (self._param_name(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        named_param_groups = self._param_groups_to_named_param_groups(self.param_groups)

        return {"state": named_state, "named_param_groups": named_param_groups}        

    def load_state_dict(self, state_dict):
        state_dict["param_groups"] = self._named_param_groups_to_param_groups(
            state_dict["named_param_groups"], self.param_groups,
        )
        del state_dict["named_param_groups"]
        self.optimizer.load_state_dict(state_dict)

    def _param_groups_to_named_param_groups(
        self, param_groups: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Convert a parameter group to a mapping of parameter names to group metadata."""
        param_to_group_meta = {}
        for group in param_groups:
            group_meta = copy.deepcopy(group)
            del group_meta["params"]
            for p in group["params"]:
                param_to_group_meta[self._param_name(p)] = group_meta
        return param_to_group_meta

    def _named_param_groups_to_param_groups(
        self,
        param_to_group_meta: dict[str, Any],
        param_groups: list[dict[str, Any]],
        strict: bool = True,
    ) -> list[dict[str, Any]]:
        """Convert a mapping of parameter names to group metadata to a list of parameter groups."""
        new_param_groups = []
        for group in param_groups:
            new_group = {"params": []}
            for param in group["params"]:
                param_name = self._param_name(param)
                if param_name not in param_to_group_meta:
                    if strict:
                        raise ValueError(
                            f"Parameter {param_name} not found in param_to_group_meta mapping."
                        )
                    continue
                group_meta = param_to_group_meta[param_name]
                new_group_wo_params = copy.deepcopy(new_group)
                del new_group_wo_params["params"]
                if new_group_wo_params and new_group_wo_params != group_meta:
                    error_info = (
                        f"Parameter {param_name} and the parameters in the same group "
                        f"{new_group['params']} have different metadata. Please check "
                        "that whether the checkpoint and current param_groups match. "
                        f"Parameter {param_name} has metadata {group_meta}, "
                        f"while others group metadata is {new_group}."
                    )
                    if strict:
                        raise ValueError(error_info)
                    else:
                        logger.warning(error_info)
                        continue
                new_group["params"].append(param)
                new_group.update(group_meta)
            new_param_groups.append(new_group)
        return new_param_groups

    def _param_name(self, param: torch.nn.Parameter) -> str:
        """Get the name of the parameter."""
        if not hasattr(self, "param_to_name"):
            name_to_param = {}
            for model_chunk in self.model_chunks:
                _name_to_param = dict(model_chunk.named_parameters())
                common_keys = name_to_param.keys() & _name_to_param.keys()
                if common_keys:
                    raise ValueError(
                        f"Parameter names conflict between model chunks: {common_keys}. "
                        "Ensure that each model chunk has unique parameter names."
                    )
                name_to_param.update(_name_to_param)
            num_experts = self.model_chunks[0].config.num_moe_experts if self.model_chunks else None
            name_to_param = handle_experts_in_state_dict(name_to_param, num_experts)
            self.param_to_name = {param: name for name, param in name_to_param.items()}
        assert (
            param in self.param_to_name
        ), f"Parameter {param} not found in param_to_name mapping. "
        return self.param_to_name[param]

    def _init_optimizer_states_with_dummy_values(self):
        # Initializes optimizer states with dummy values.

        # This is necessary to ensure that the optimizer's states are
        # initialized correctly. These dummy states will be replaced in-place
        # during the loading of distributed checkpoints.
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.numel() == 0 or (
                    hasattr(param, "_local_tensor") and param._local_tensor.numel() == 0
                ):
                    # Avoid FusedAdam errors on empty tensor input.
                    continue
                param.grad = torch.zeros_like(param)
        self.optimizer.step()
        self.zero_grad()

    def zero_grad(self, set_to_none: bool = True):
        """
        Zeroes grads for the model related parameters, i.e., model_float16_groups
        and model_fp32_groups. We additionally zero the remaining groups as a
        memory optimization to reduce fragmentation; in the case of
        set_to_none==True, the space used by this field can be safely deallocated.

        Args:
            set_to_none (bool): if true, set grads to None.
        """
        if self.ddp_config.use_megatron_fsdp:
            for model_chunk in self.model_chunks:
                model_chunk.zero_grad_buffer()
        else:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            p.grad.zero_()
