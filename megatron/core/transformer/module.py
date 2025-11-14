# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron Module."""
from typing import Optional, Tuple

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)

_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)


def param_is_not_shared(param):  # pylint: disable=missing-function-docstring
    return not hasattr(param, 'shared') or not param.shared


class MegatronModule(torch.nn.Module):
    """Base Megatron module inhertied by all Models.

    Megatron specific extensions of torch Module with support
    for pipelining

    Args:
        config (TransformerConfig): Transformer config
    """

    # def __init__(self, config: TransformerConfig, share_word_embeddings=True):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

    def state_dict_for_save_checkpoint(self, prefix: str = '', keep_vars: bool = False):
        """Override state dict for saving checkpoints Use this function to override the
        state dict for saving checkpoints.

        Args:
            prefix (str, optional): _description_. Defaults to ''.
            keep_vars (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        return self.state_dict(prefix=prefix, keep_vars=keep_vars)

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Default implementation for sharded state dict for distributed checkpointing.

        General definition of sharded_state_dict simply calls `sharded_state_dict_default`
        (which call sharded_state_dict method if possible or a default implementation otherwise)
        recursively on all submodules.

        Args:
            prefix (str): prefix for the state dict keys
            sharded_offsets (Tuple[Tuple[int, int, int]], optional): sharding already
                applied (e.g. PP related) by sup-modules. Passed along to ShardedTensor
            metadata (dict, optional): metadata passed recursively to sharded_state_dict methods

        Returns:
            dict: dictionary of state dict keys mapped to ShardedTensors
        """
        sharded_state_dict = {}
        # Save parameters
        self._save_to_state_dict(sharded_state_dict, '', keep_vars=True)
        if not hasattr(self, 'tp_group'):
            # some model interface hasn't updated for m4, fallback needed
            tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            tp_group = self.tp_group
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            sharded_offsets=sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=metadata['dp_cp_group'],
        )
        # Recurse into submodules
        for name, module in self.named_children():
            sharded_state_dict.update(
                sharded_state_dict_default(
                    module, f'{prefix}{name}.', sharded_offsets, metadata, tp_group=tp_group
                )
            )
        return sharded_state_dict

    def set_is_first_microbatch(self):
        """Sets the is_first_microbatch flag if it exists and config.fp8==True.
        When this flag is set, TE modules will update their fp8 parameter cache.
        If kitchen is being used, kitchen controls quantization level.
        """
        if (
            self.config.fp8 is not None
            or self.config.fp4 is not None
            or getattr(self.config, 'use_kitchen', False)
        ):
            if not hasattr(self, "modules_with_is_first_microbatch"):
                self.modules_with_is_first_microbatch = []
                for m in self.modules():
                    if hasattr(m, "is_first_microbatch"):
                        self.modules_with_is_first_microbatch.append(m)
            for m in self.modules_with_is_first_microbatch:
                m.is_first_microbatch = True

    def set_symmetric_ar(self, set_to: Optional[str] = None) -> None:
        """
        Set symmetric all-reduce functionality across all eligible modules.

        This method traverses the model's module hierarchy to find all modules
        with the 'symmetric_ar_type' attribute, caches them, and then sets their
        '_symmetric_ar_cache' attribute to the specified value to enable or
        disable symmetric all-reduce operations.

        Args:
            set_to (Any, optional): Value to set for the 'symmetric_ar_type' to.
            Allowed choices ['two_shot', "one_shot", "multimem_all_reduce", None]
        """
        assert set_to in ['two_shot', "one_shot", "multimem_all_reduce", None]

        # Recursive function to find all modules with our target attributes
        def create_ar_cache(module):
            # Check if this module has any of our target attributes
            if hasattr(module, "symmetric_ar_type"):
                self._symmetric_ar_cache.append(module)

            # Check all children modules recursively
            for child in module._modules.values():
                if child is not None:
                    create_ar_cache(child)

        if not hasattr(self, "_symmetric_ar_cache"):
            self._symmetric_ar_cache = []
            create_ar_cache(self)

        for module in self._symmetric_ar_cache:
            module._symmetric_ar_cache = set_to


class GraphableMegatronModule(MegatronModule):
    """Megatron module that can be used to capture and replay CUDA graphs.
    Now only TransformerLayer and MambaLayer are graphable.

    Args:
        config (TransformerConfig): Transformer config
    """

    def __init__(self, config: TransformerConfig, vp_stage: Optional[int] = None):
        super().__init__(config)

        assert isinstance(config, TransformerConfig), "config must be a TransformerConfig"

        # Enable cuda graphs.
        if config.cuda_graph_impl == "local":
            from megatron.core.transformer.cuda_graphs import CudaGraphManager

            self.cudagraph_manager = CudaGraphManager(config, vp_stage=vp_stage)
        elif config.cuda_graph_impl == "transformer_engine":
            # List to store CUDA graphs. A list of `N` CUDA graphs for this layer where N is
            # the number of microbatches. Multiple CUDA graphs per layer is required to support
            # pipelining which requires running FWD graph of multiple microbatches before BWD
            # graph. To enable CUDA graph, this list should be populated in the model training
            # script with the graphs returned by make_graphed_callables API before the first
            # training step.
            self.cuda_graphs = []
            # List to store forward pre-hooks. Forward pre-hooks are not captured into CUDA
            # graphs. Those hooks and args are collected in this list and should be manually
            # triggered before CUDA Graph running. This is required to ensure the correct param
            # all-gather overlap with forward compute.
            self.cuda_graph_manual_hooks = []

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """
        Get the static inputs for the layer.
        We assume that the module has one hidden_states input, whose shape is inferred
        from the seq_length, micro_batch_size, and parallel config.
        Override this method if the module has other inputs.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the static inputs for the layer.
        """
        # Calculate data shape related values.
        context_parallel_size = self.config.context_parallel_size
        slen_per_cp = seq_length // context_parallel_size
        sequence_parallel = self.config.sequence_parallel
        tensor_model_parallel_size = self.config.tensor_model_parallel_size
        slen_per_cptp = (
            slen_per_cp // tensor_model_parallel_size if sequence_parallel else slen_per_cp
        )

        static_inputs = {}
        static_inputs["hidden_states"] = torch.ones(
            (slen_per_cptp, micro_batch_size, self.config.hidden_size),
            dtype=torch.bfloat16,
            requires_grad=True,
            device=torch.cuda.current_device(),
        )
        return static_inputs

    def setup_manual_hooks(self, make_hook_func):
        """
        Set CUDA Graph manual hooks for the submodules that contain direct parameters and are
        covered by cudagraphs.
        """
        self.cuda_graph_manual_hooks = []

        # Select the modules who contain direct parameters and are covered by cudagraphs.
        # Add these modules to the `cuda_graph_manual_hooks` because their hooks will not
        # be automatically triggered when they go through the CUDA Graph path.
        param_modules = {}
        for submodule in self._get_submodules_under_cudagraphs():
            for module in submodule.modules():
                if next(module.parameters(recurse=False), None) is not None:
                    # Module contains direct parameters.
                    param_modules[id(module)] = module
        for module in param_modules.values():
            self.cuda_graph_manual_hooks.append((make_hook_func(), (module,)))

    def _get_submodules_under_cudagraphs(self):
        """
        Get the submodules that are covered by cudagraphs. Return a list that only contains the
        module itself if the whole layer is covered by cudagraphs.
        """
        return [self]

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """
        CUDA Graph capture for this layer using TE interface.
        Normally it's just a forward pass if we're capturing the entire layer.
        """
        return self.forward(*args, **kwargs)

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """
        CUDA graph replay for this layer and microbatch `self.current_microbatch` using TE
        interface. TransformerEngine versions>=1.10 allow keyword arguments with CUDA graph.
        However, CUDA graph accepts only Tensor inputs.
        Hence, check if the arguments are all tensors.
        """
        for arg in args:
            assert isinstance(arg, torch.Tensor), "CUDA graph accepts only Tensor inputs."
        for _, v in kwargs.items():
            assert v is None or isinstance(
                v, torch.Tensor
            ), "CUDA graph accepts only Tensor inputs."

        cg_index = getattr(self, 'current_microbatch', 0) % len(self.cuda_graphs)
        cudagraph_args, cudagraph_kwargs = self._get_te_cuda_graph_replay_args(*args, **kwargs)

        for hook, hook_args in self.cuda_graph_manual_hooks:
            hook(*hook_args)
        return self.cuda_graphs[cg_index](*cudagraph_args, **cudagraph_kwargs)

    def _get_te_cuda_graph_replay_args(self, *args, **kwargs):
        """Helper function to get tensor arguments for TE CUDA graph."""
        if len(args) == 0:
            assert 'hidden_states' in kwargs, "hidden_states is required."
            hidden_states = kwargs.pop('hidden_states')
            cudagraph_args = (hidden_states,)
        else:
            assert (
                'hidden_states' not in kwargs
            ), "hidden_states should only be passed as either a positional or keyword argument."
            cudagraph_args = tuple(args)

        cudagraph_kwargs = kwargs.copy()
        cudagraph_kwargs['is_first_microbatch'] = getattr(self, 'current_microbatch', 0) == 0
        return cudagraph_args, cudagraph_kwargs

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        return hasattr(self, 'cudagraph_manager')

    def _should_call_te_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the TE cudagraph path.
        """
        from megatron.core.transformer.cuda_graphs import is_graph_capturing

        return (
            self.config.cuda_graph_impl == "transformer_engine"
            and self.training
            and (is_graph_capturing() or self.cuda_graphs)
        )

    def __call__(self, *args, **kwargs):

        if self._should_call_local_cudagraph(*args, **kwargs):
            # Set the is_first_microbatch flag for weight caching
            current_microbatch = getattr(self, 'current_microbatch', 0)
            self.cudagraph_manager.set_is_first_microbatch(current_microbatch == 0)
            return self.cudagraph_manager(self, args, kwargs)
        elif self._should_call_te_cudagraph(*args, **kwargs):
            if not self.cuda_graphs:
                # Do CUDA Graphs capture.
                cuda_graph_func = self._te_cuda_graph_capture
            else:
                # Do CUDA Graphs replay.
                cuda_graph_func = self._te_cuda_graph_replay
            return cuda_graph_func(*args, **kwargs)
        return super().__call__(*args, **kwargs)


def conversion_helper(val, conversion):
    """Recursively applies a conversion function to values in nested data structures.

    Args:
        val: A single value or a nested structure (tuple/list) of values to convert
        conversion (callable): A function that performs the desired conversion on a single value

    Returns:
        The converted value, maintaining the same nested structure as the input.
        If input is a single value, returns the converted value.
        If input is a tuple/list, returns a tuple/list with all elements converted.
    """
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_convertor):
    """Converts floating-point values from fp32 to fp16.

    Args:
        val: The value to convert. Can be a single number, a tuple, or a list.
        float16_convertor: A function that converts a single fp32 value to fp16
    """

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Converts floating-point values from fp16 to fp32.

    Args:
        val: The value to convert. Can be a single number, a tuple, or a list.
    """

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


class Float16Module(MegatronModule):
    """Float 16 Module.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(Float16Module, self).__init__(config)
        self.config = config
        self.fp16 = config.fp16
        self.bf16 = config.bf16
        self.vp_stage = getattr(module, 'vp_stage', None)

        if self.fp16:
            self.add_module('module', module.half())

            def float16_convertor(val):
                return val.half()

        elif self.bf16:
            self.add_module('module', module.bfloat16())

            def float16_convertor(val):
                return val.bfloat16()

        else:
            raise Exception('Either config.fp16 or config.bf16 should be True.')

        self.float16_convertor = float16_convertor

    def set_input_tensor(self, input_tensor):  # pylint: disable=missing-function-docstring
        return self.module.set_input_tensor(input_tensor)

    def forward(self, *inputs, fp32_output=True, **kwargs):
        """
        Execute the wrapped module in model precision and optionally upcast outputs to fp32.

        On the first pipeline stage, positional/keyword tensor inputs are converted to the
        module precision (fp16 or bf16) before invoking the wrapped module. The wrapped module
        is called with the provided inputs and keyword arguments. On the last pipeline stage
        only, outputs are upcast to fp32 if ``fp32_output`` is True; otherwise, outputs are
        returned in the model precision (fp16/bf16).

        Args:
            *inputs: Positional inputs forwarded to the wrapped module (converted to fp16/bf16 on
                the pipeline first stage).
            fp32_output (bool, keyword-only): If True (default), upcast outputs to fp32 on the
                pipeline last stage. Has no effect on non-last stages. Set to False to keep outputs
                in model precision when downstream consumers expect half precision or to avoid
                extra casts.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            The wrapped module's outputs, potentially upcast to fp32 depending on pipeline stage
            and ``fp32_output``.
        """
        if parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=self.vp_stage):
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if (
            parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=self.vp_stage)
            and fp32_output is True
        ):
            outputs = float16_to_fp32(outputs)
        return outputs

    def state_dict(
        self, destination=None, prefix='', keep_vars=False
    ):  # pylint: disable=missing-function-docstring
        return self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """Retrieve state_dict from the module being wrapped."""
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def sharded_state_dict(self, prefix='', *args, **kwargs):
        """Retrieve sharded_state_dict from the module being wrapped."""
        return self.module.sharded_state_dict(prefix, *args, **kwargs)

    def load_state_dict(
        self, state_dict, strict=True
    ):  # pylint: disable=missing-function-docstring
        self.module.load_state_dict(state_dict, strict=strict)
