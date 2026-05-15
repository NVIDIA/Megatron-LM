# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mixed precision policy helpers for Megatron-FSDP2.

This module owns the v2 policy data model. Translation from Megatron/MCore
config objects belongs in the adapter layer.
"""

import inspect
from contextlib import nullcontext
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

    def model_init_context(self):
        """Return the model-init context for mixed precision parameter creation."""
        if not self.fp8.enabled:
            return nullcontext()

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
        return QUANTIZED_MODEL_INIT_CLASS(**args)

    def group_key_dtype(self, tensor: torch.Tensor):
        """Return the parameter grouping dtype key."""
        if not self.is_fp8_param(tensor):
            return tensor.dtype
        return ("quantized", type(tensor).__name__, self.fp8.recipe)

    def is_fp8_param(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` is managed as an FP8 parameter."""
        return is_fp8_param(tensor)

    def model_weight_buffer_dtype(self, tensor: torch.Tensor) -> torch.dtype:
        """Return the model-weight buffer dtype for ``tensor``."""
        return torch.uint8 if self.is_fp8_param(tensor) else tensor.dtype

    def get_param_data(self, tensor: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        """Return the parameter data view to store in the model-weight buffer."""
        if self.is_fp8_param(tensor):
            return get_fp8_raw_data(tensor, transpose=transpose)
        return tensor.detach()

    def needs_transpose_weight_buffer(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` needs an extra transpose/columnwise buffer."""
        return HAVE_TE_MXFP8 and isinstance(tensor, MXFP8Tensor)

    def main_params_dtype_for_param(self, tensor: torch.Tensor) -> Optional[torch.dtype]:
        """Return the main-parameter dtype for a parameter group."""
        if self.is_fp8_param(tensor) and self.main_params_dtype is None:
            return torch.float32
        return self.main_params_dtype

    def main_grads_dtype_for_param(self, tensor: torch.Tensor) -> torch.dtype:
        """Return the main-gradient dtype for a parameter group."""
        if self.main_grads_dtype is not None:
            return self.main_grads_dtype
        return torch.bfloat16 if self.is_fp8_param(tensor) else tensor.dtype

    def initial_main_weight(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the initial high-precision optimizer weight for ``tensor``."""
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

    def set_unsharded_weight(
        self, tensor: torch.Tensor, data: torch.Tensor, transpose: bool = False
    ) -> None:
        """Attach an unsharded FP8 payload to an FP8 parameter."""
        assert self.is_fp8_param(
            tensor
        ), f"Type {type(tensor)} is not a Transformer Engine FP8 tensor"
        if transpose:
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

    def post_unshard(self, params: List[torch.Tensor], is_bwd: bool = False) -> None:
        """Run post-unshard mixed precision processing for a parameter group."""
        params = [param for param in params if self.is_fp8_param(param)]
        if len(params) == 0:
            return

        if self.needs_transpose_weight_buffer(params[0]):
            # Match v1: forward only rebinds rowwise raw data. Do not mark the
            # MXFP8 columnwise payload unavailable since TE may request the
            # backward workspace from inside the forward call stack.
            if is_bwd:
                self.create_transpose_cache(params)
            return

        # TE rebuilds recipe-specific state after FSDP all-gather for recipes
        # where columnwise data is derived from the all-gathered rowwise data.
        self.create_transpose_cache(params)
        self.set_weight_usage(params, rowwise=not is_bwd, columnwise=True)

    def create_transpose_cache(self, params: List[torch.Tensor]) -> None:
        """Run TE post-all-gather processing for FP8 parameters."""
        if HAVE_TE_POST_ALL_GATHER_PROCESSING:
            post_all_gather_processing(params)
            return
        for param in params:
            if hasattr(param, "_create_transpose"):
                param._create_transpose()
            else:
                param._create_columnwise()

    def set_weight_usage(
        self, params: List[torch.Tensor], *, rowwise: bool, columnwise: bool
    ) -> None:
        """Set FP8 rowwise/columnwise usage flags when TE exposes them."""
        for param in params:
            if self.is_fp8_param(param) and hasattr(param, "update_usage"):
                param.update_usage(rowwise_usage=rowwise, columnwise_usage=columnwise)

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

    def quantize_main_weights_to_model(
        self,
        model_params: List[torch.Tensor],
        main_params: List[torch.Tensor],
        start_offsets: List[int],
        data_parallel_group: torch.distributed.ProcessGroup,
        fsdp_shard_model_params: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    ) -> None:
        """Quantize high-precision main weights into FP8 model-weight shards."""
        quantize_main_weights_to_fp8(
            model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params
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
    main_params: List[torch.Tensor],
    start_offsets: List[int],
    data_parallel_group: torch.distributed.ProcessGroup,
    fsdp_shard_model_params: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
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
