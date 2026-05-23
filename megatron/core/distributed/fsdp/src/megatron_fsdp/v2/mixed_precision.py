# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mixed precision policy helpers for Megatron-FSDP2.

This module owns the v2 policy data model. Translation from Megatron/MCore
config objects belongs in the adapter layer.
"""

import inspect
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import List, Optional, Tuple

import torch
from packaging.version import Version as PkgVersion

try:
    import transformer_engine as te

    if hasattr(te, "__version__"):
        TE_VERSION = PkgVersion(str(te.__version__))
    else:
        TE_VERSION = PkgVersion(version("transformer-engine"))
except Exception:
    TE_VERSION = None

try:
    from transformer_engine.pytorch.tensor import QuantizedTensor

    FP8_TENSOR_CLASS = QuantizedTensor
    HAVE_TE_FP8 = True
except Exception:
    try:
        from transformer_engine.pytorch.float8_tensor import Float8Tensor

        FP8_TENSOR_CLASS = Float8Tensor
        HAVE_TE_FP8 = True
    except Exception:
        FP8_TENSOR_CLASS = None
        HAVE_TE_FP8 = False

try:
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    HAVE_TE_FLOAT8TENSOR = True
except Exception:
    try:
        from transformer_engine.pytorch.float8_tensor import Float8Tensor

        HAVE_TE_FLOAT8TENSOR = True
    except Exception:
        Float8Tensor = None
        HAVE_TE_FLOAT8TENSOR = False

try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    HAVE_TE_MXFP8 = True
except Exception:
    MXFP8Tensor = None
    HAVE_TE_MXFP8 = False

try:
    from transformer_engine.pytorch.tensor.utils import cast_master_weights_to_fp8

    HAVE_TE_CAST_MASTER_WEIGHTS_TO_FP8 = True
except Exception:
    HAVE_TE_CAST_MASTER_WEIGHTS_TO_FP8 = False

try:
    from transformer_engine.pytorch.tensor.utils import post_all_gather_processing

    HAVE_TE_POST_ALL_GATHER_PROCESSING = True
except Exception:
    HAVE_TE_POST_ALL_GATHER_PROCESSING = False

try:
    from transformer_engine.pytorch import quantized_model_init

    QUANTIZED_MODEL_INIT_CLASS = quantized_model_init
except Exception:
    try:
        from transformer_engine.pytorch import fp8_model_init

        QUANTIZED_MODEL_INIT_CLASS = fp8_model_init
    except Exception:
        QUANTIZED_MODEL_INIT_CLASS = nullcontext

try:
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
except Exception:
    TransformerEngineBaseModule = None

try:
    from megatron.core.tensor_parallel import get_cuda_rng_tracker
except Exception:
    from ..utils import get_cuda_rng_tracker

from ..utils import _MODEL_PARALLEL_RNG_TRACKER_NAME

if not HAVE_TE_CAST_MASTER_WEIGHTS_TO_FP8:
    try:
        from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_scale

        multi_tensor_scale_impl = multi_tensor_scale
    except ImportError:
        try:
            import amp_C
            from apex.multi_tensor_apply import multi_tensor_applier

            multi_tensor_scale_impl = amp_C.multi_tensor_scale
        except ImportError:

            def local_multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
                return op(2048 * 32, noop_flag_buffer, tensor_lists, *args)

            def local_multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
                for src, dst in zip(tensor_lists[0], tensor_lists[1]):
                    dst.copy_(src * scale)

            multi_tensor_applier = local_multi_tensor_applier
            multi_tensor_scale_impl = local_multi_tensor_scale

    def _multi_tensor_copy_this_to_that(
        this: List[torch.Tensor],
        that: List[torch.Tensor],
        overflow_buf: Optional[torch.Tensor] = None,
    ):
        if overflow_buf is not None:
            overflow_buf.fill_(0)
            multi_tensor_applier(multi_tensor_scale_impl, overflow_buf, [this, that], 1.0)
        else:
            for this_, that_ in zip(this, that):
                that_.copy_(this_)


@dataclass(frozen=True)
class FullyShardFP8Policy:
    """FP8 recipe settings owned by the v2 ``fully_shard`` path."""

    enabled: bool = False
    recipe: Optional[str] = None
    keep_transpose_cache: bool = False


@dataclass(frozen=True)
class FullyShardMixedPrecisionPolicy:
    """Mixed precision policy owned by the v2 ``fully_shard`` path."""

    main_params_dtype: Optional[torch.dtype] = None
    main_grads_dtype: Optional[torch.dtype] = None
    grad_comm_dtype: Optional[torch.dtype] = None
    fp8_param_gather: bool = False
    fp8_recipe: Optional[str] = None
    keep_fp8_transpose_cache: bool = False
    use_decoupled_grad: bool = False
    fp8: Optional[FullyShardFP8Policy] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.fp8 is None:
            fp8 = FullyShardFP8Policy(
                enabled=self.fp8_param_gather,
                recipe=self.fp8_recipe,
                keep_transpose_cache=self.keep_fp8_transpose_cache,
            )
            object.__setattr__(self, "fp8", fp8)
            return

        object.__setattr__(self, "fp8_param_gather", self.fp8.enabled)
        object.__setattr__(self, "fp8_recipe", self.fp8.recipe)
        object.__setattr__(self, "keep_fp8_transpose_cache", self.fp8.keep_transpose_cache)

    @contextmanager
    def model_init_context(self, module: Optional[torch.nn.Module] = None):
        """Return the model-init context for mixed precision parameter creation."""
        with ExitStack() as stack:
            if self.fp8.enabled:
                # TE initializes FP8 parameters while preserving a high-precision value
                # that can seed the optimizer main-weight buffer.
                assert (
                    QUANTIZED_MODEL_INIT_CLASS is not nullcontext
                ), "Transformer Engine is required for FP8 parameter initialization"
                args = {"enabled": True}
                if (
                    "preserve_high_precision_init_val"
                    in inspect.signature(QUANTIZED_MODEL_INIT_CLASS).parameters
                ):
                    args["preserve_high_precision_init_val"] = True
                stack.enter_context(QUANTIZED_MODEL_INIT_CLASS(**args))

            if (
                module is not None
                and TransformerEngineBaseModule is not None
                and TE_VERSION is not None
                and TE_VERSION >= PkgVersion("0.9.0")
                and not isinstance(module, TransformerEngineBaseModule)
            ):
                cuda_rng_tracker = get_cuda_rng_tracker()
                if _MODEL_PARALLEL_RNG_TRACKER_NAME in cuda_rng_tracker.states_:
                    stack.enter_context(cuda_rng_tracker.fork())

            yield

    def group_key_dtype(self, tensor: torch.Tensor):
        """Return the parameter grouping dtype key."""
        if not self.is_fp8_param(tensor):
            return tensor.dtype
        return ("quantized", type(tensor).__name__, self.fp8.recipe)

    def validate_param_group(self, params: List[torch.Tensor]) -> None:
        """Validate that one ParameterGroup has one mixed-precision storage kind."""
        if len(params) == 0:
            return
        group_key = self.group_key_dtype(params[0])
        assert all(self.group_key_dtype(param) == group_key for param in params), (
            "Parameters with different mixed-precision storage kinds must not "
            "share a ParameterGroup"
        )

    def is_fp8_param(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` is managed as an FP8 parameter."""
        return is_fp8_param(tensor)

    def fine_grained_forward_hooks_required(self, param_groups) -> bool:
        """Return whether submodule forward hooks are needed for these parameter groups."""
        for param_group in param_groups:
            for param in param_group.params:
                if self.is_fp8_param(param):
                    return True
        return False

    @staticmethod
    def model_weight_buffer_dtype(tensor: torch.Tensor) -> torch.dtype:
        """Return the model-weight buffer dtype for ``tensor``."""
        return torch.uint8 if is_fp8_param(tensor) else tensor.dtype

    def get_param_data(self, tensor: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        """Return the parameter data view to store in the model-weight buffer."""
        if self.is_fp8_param(tensor):
            return get_fp8_raw_data(tensor, transpose=transpose)
        return tensor.detach()

    def bind_unsharded_param(
        self, tensor: torch.Tensor, data: torch.Tensor, buffer_role: str
    ) -> None:
        """Bind a parameter to an unsharded model-weight buffer view."""
        if self.is_fp8_param(tensor):
            if buffer_role == "transpose_weight":
                assert self.needs_transpose_weight_buffer(
                    tensor
                ), f"Type {type(tensor)} has no transpose payload"
                attr = "_columnwise_data"
            else:
                attr = "_rowwise_data" if hasattr(tensor, "_rowwise_data") else "_data"

            # Rebind TE's raw uint8 payload to the all-gathered FSDP buffer view.
            old_data = getattr(tensor, attr)
            if old_data is not None:
                assert (
                    old_data.dtype == data.dtype
                ), f"FP8 raw dtype mismatch: {old_data.dtype} vs {data.dtype}"
                assert (
                    old_data.shape == data.shape
                ), f"FP8 raw shape mismatch: {old_data.shape} vs {data.shape}"
            setattr(tensor, attr, data)
            return
        tensor.data = data

    def storage_tensors_to_free(
        self, tensor: torch.Tensor, model_weight_buffer, main_weight_buffer
    ) -> List[torch.Tensor]:
        """Return original parameter storages that FSDP buffers have replaced."""
        # The buffers are ownership signals, not data sources: non-FP8 params can
        # be backed by either model or main weight storage, while FP8 params are
        # only safe to free after their raw payload is copied to the model buffer.
        if model_weight_buffer is None and main_weight_buffer is None:
            return []

        if not self.is_fp8_param(tensor):
            return [tensor.data]

        if model_weight_buffer is None:
            return []

        tensors = [get_fp8_raw_data(tensor)]
        if self.needs_transpose_weight_buffer(tensor):
            tensors.append(get_fp8_raw_data(tensor, transpose=True))
        return tensors

    def weight_buffers_for_unshard(
        self, model_weight_buffer, transpose_weight_buffer=None, *, bwd_pass: bool = False
    ) -> List:
        """Return the weight buffer needed for forward or backward compute."""
        if bwd_pass and transpose_weight_buffer is not None:
            return [transpose_weight_buffer]
        return [model_weight_buffer]

    def needs_transpose_weight_buffer(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` needs an extra transpose/columnwise buffer."""
        return HAVE_TE_MXFP8 and isinstance(tensor, MXFP8Tensor)

    def main_params_dtype_for_param(self, tensor: torch.Tensor) -> Optional[torch.dtype]:
        """Return the main-parameter dtype for a parameter group."""
        if self.is_fp8_param(tensor) and self.main_params_dtype is None:
            return torch.float32
        return self.main_params_dtype

    def main_grads_dtype_for_param(self, tensor: torch.Tensor) -> torch.dtype:
        """Return the main-gradient dtype for a parameter group.

        Defaults to float32 for stable gradient accumulation (avoids
        precision loss in reduce-scatter and optimizer steps).
        """
        if self.main_grads_dtype is not None:
            return self.main_grads_dtype
        return torch.float32

    def get_high_precision_value(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a high-precision value for initializing optimizer main weights."""
        if not self.is_fp8_param(tensor):
            return tensor.detach()
        if hasattr(tensor, "get_high_precision_init_val"):
            item = tensor.get_high_precision_init_val()
            tensor.clear_high_precision_init_val()
            return item

        # If TE did not preserve the FP32 init value, recover a main-weight
        # value from the FP8 tensor before the original full parameter is freed.
        assert isinstance(TE_VERSION, PkgVersion) and TE_VERSION >= PkgVersion(
            "2.0"
        ), "Transformer Engine >= 2.0 is required for FP8 dequantize"
        return tensor.dequantize()

    def post_unshard(self, params: List[torch.Tensor], bwd_pass: bool = False) -> None:
        """Run post-unshard mixed precision processing for a parameter group."""
        params = [param for param in params if self.is_fp8_param(param)]
        if len(params) == 0:
            return

        if self.needs_transpose_weight_buffer(params[0]):
            # Match v1: forward only rebinds rowwise raw data. Do not mark the
            # MXFP8 columnwise payload unavailable since TE may request the
            # backward workspace from inside the forward call stack.
            if bwd_pass:
                if HAVE_TE_POST_ALL_GATHER_PROCESSING:
                    post_all_gather_processing(params)
                else:
                    for param in params:
                        if hasattr(param, "_create_transpose"):
                            param._create_transpose()
                        else:
                            param._create_columnwise()
            return

        # TE rebuilds recipe-specific state after FSDP all-gather for recipes
        # where columnwise data is derived from the all-gathered rowwise data.
        if HAVE_TE_POST_ALL_GATHER_PROCESSING:
            post_all_gather_processing(params)
        else:
            for param in params:
                if hasattr(param, "_create_transpose"):
                    param._create_transpose()
                else:
                    param._create_columnwise()
        for param in params:
            if hasattr(param, "update_usage"):
                param.update_usage(rowwise_usage=not bwd_pass, columnwise_usage=True)

    def post_reshard(self, params: List[torch.Tensor]) -> None:
        """Run post-reshard mixed precision processing for a parameter group."""
        if self.fp8.keep_transpose_cache:
            return
        for param in params:
            if not self.is_fp8_param(param):
                continue
            # Drop full-weight transpose/cache views after FSDP reshard releases
            # the unsharded payload.
            if hasattr(param, "_transpose_invalid"):
                param._transpose_invalid = True
                param._transpose = None
            elif not self.needs_transpose_weight_buffer(param):
                param.update_usage(rowwise_usage=True, columnwise_usage=False)

    def copy_main_weights_to_model_weights(
        self,
        params: List[torch.Tensor],
        param_idx: dict,
        data_parallel_group: torch.distributed.ProcessGroup,
        model_weight_buffer,
        main_weight_buffer,
        transpose_weight_buffer=None,
    ) -> None:
        """Install optimized main weights into model compute weights."""
        if main_weight_buffer is None:
            return

        assert model_weight_buffer is not None, "main weights require a model-weight buffer"
        if not self.is_fp8_param(params[0]):
            model_weight_buffer.data.copy_(main_weight_buffer.data)
            return

        fp8_params = []
        main_params = []
        start_offsets = []
        model_param_shards = []
        for param in params:
            item_id = param_idx[param]
            model_shard = model_weight_buffer.get_item(item_id, only_shard=True)
            if model_shard.numel() == 0:
                fp8_params.append(param)
                main_params.append(None)
                start_offsets.append(None)
                model_param_shards.append((None, None))
                continue

            transpose_shard = None
            if transpose_weight_buffer is not None:
                transpose_shard = transpose_weight_buffer.get_item(item_id, only_shard=True)
            main_weight = main_weight_buffer.get_item(item_id, only_shard=True)
            start_offset, _ = model_weight_buffer.buffer_index._get_item_slice_in_shard(item_id)
            fp8_params.append(param)
            main_params.append(main_weight)
            start_offsets.append(start_offset)
            model_param_shards.append((model_shard, transpose_shard))

        quantize_main_weights_to_fp8(
            fp8_params, main_params, start_offsets, data_parallel_group, model_param_shards
        )


def is_fp8_param(tensor: torch.Tensor) -> bool:
    """Return True if the parameter is backed by a Transformer Engine FP8 tensor."""
    return HAVE_TE_FP8 and isinstance(tensor, FP8_TENSOR_CLASS)


def get_fp8_raw_data(tensor: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    """Return the raw uint8 storage owned by a Transformer Engine FP8 tensor."""
    assert is_fp8_param(tensor), f"Type {type(tensor)} is not a Transformer Engine FP8 tensor"
    if transpose:
        assert HAVE_TE_MXFP8 and isinstance(
            tensor, MXFP8Tensor
        ), f"Type {type(tensor)} has no transpose payload"
        return getattr(tensor, "_columnwise_data")
    attr = "_rowwise_data" if hasattr(tensor, "_rowwise_data") else "_data"
    return getattr(tensor, attr)


def quantize_main_weights_to_fp8(
    model_params: List[torch.Tensor],
    main_params: List[Optional[torch.Tensor]],
    start_offsets: List[Optional[int]],
    data_parallel_group: torch.distributed.ProcessGroup,
    fsdp_shard_model_params: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
) -> None:
    """Quantize FP32 main-weight shards into FP8/MXFP8 model-weight shards."""
    if len(model_params) == 0:
        return

    fsdp_shard_model_params = [x[0] if x[1] is None else x for x in fsdp_shard_model_params]
    if HAVE_TE_CAST_MASTER_WEIGHTS_TO_FP8:
        args = [
            model_params,
            main_params,
            start_offsets,
            data_parallel_group,
            fsdp_shard_model_params,
        ]
        kwargs = {}
        if HAVE_TE_POST_ALL_GATHER_PROCESSING:
            kwargs["manual_post_all_gather_processing"] = True
        cast_master_weights_to_fp8(*args, **kwargs)
        return

    assert HAVE_TE_FLOAT8TENSOR, "Transformer Engine Float8Tensor is required for FP8 fallback"
    for model_param, main_param, start_offset, shard_model_param in zip(
        model_params, main_params, start_offsets, fsdp_shard_model_params
    ):
        if main_param is None:
            continue
        if shard_model_param is None:
            shard_model_param = get_fp8_raw_data(model_param).view(-1)[
                start_offset : start_offset + main_param.numel()
            ]

        quantizer = model_param._quantizer
        main_param = main_param.to(model_param.dtype)
        out = Float8Tensor(
            shape=main_param.size(),
            dtype=model_param.dtype,
            requires_grad=False,
            data=shard_model_param,
            fp8_scale_inv=model_param._scale_inv,
            fp8_dtype=model_param._fp8_dtype,
            quantizer=quantizer,
        )
        quantizer.update_quantized(main_param, out)

    amaxes = []
    scales = []
    scale_invs = []
    for model_param in model_params:
        quantizer = model_param._quantizer
        amaxes.append(quantizer.amax.view(1))
        scales.append(quantizer.scale.view(1))
        scale_invs.append(model_param._scale_inv.view(1))
        model_param._reset_caches()

    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
    packed_scales = torch.empty(len(scales), dtype=torch.float32, device=scales[0].device)
    packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
    _multi_tensor_copy_this_to_that(scales, packed_scale_views, dummy_overflow_buf)
    torch.reciprocal(packed_scales, out=packed_scales)
    _multi_tensor_copy_this_to_that(packed_scale_views, scale_invs, dummy_overflow_buf)

    packed_amaxes = torch.empty(len(amaxes), dtype=torch.float32, device=amaxes[0].device)
    packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
    _multi_tensor_copy_this_to_that(amaxes, packed_amax_views, dummy_overflow_buf)
    torch.distributed.all_reduce(
        packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
    )
    _multi_tensor_copy_this_to_that(packed_amax_views, amaxes, dummy_overflow_buf)
