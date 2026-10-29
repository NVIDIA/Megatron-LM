# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""BF16 params with int16 remainders as an exact stand-in for FP32 main params.

An FP32 value is its upper 16 bits (a BF16 value) followed by 16 low bits. Keeping the BF16
model param plus those low bits as an int16 "remainder" stores the FP32 main param in 2 extra
bytes instead of 4, which is what TE's ``FusedAdam(store_param_remainders=True)`` does for Adam.
Here it serves optimizers that update FP32 main params themselves (layer-sharded Muon).

The BF16 half is the FP32 value rounded half away from zero: the upper bits are incremented
when the dropped low half is at least half a BF16 ulp, which shows up as a negative int16
remainder. That sign is how :func:`merge_main_param` undoes the rounding, so
merge(split(x)) == x bit for bit. It differs from PyTorch's round-to-nearest-even cast only on
exact ties.
"""

from typing import Callable, Dict, List, Optional
from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

from ..dist_checkpointing.mapping import ShardedStateDict
from ..dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
    optim_state_to_sharding_state,
)
from .grad_scaler import MegatronGradScaler
from .optimizer import MixedPrecisionOptimizer, _backfill_gtp_sharded_param_map
from .optimizer_config import OptimizerConfig

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()

_BLOCK_SIZE = 4096


def _to_int16(value: torch.Tensor) -> torch.Tensor:
    """Wrap int32 values into int16 two's complement without relying on cast overflow."""
    return (((value + 0x8000) & 0xFFFF) - 0x8000).to(torch.int16)


def split_main_param(main: torch.Tensor, param: torch.Tensor, remainder: torch.Tensor) -> None:
    """Store FP32 ``main`` as the BF16 ``param`` plus the int16 ``remainder``, in place."""
    assert main.dtype == torch.float32 and param.dtype == torch.bfloat16
    assert remainder.dtype == torch.int16
    bits = main.detach().view(torch.int32)
    low = bits & 0xFFFF
    rounded_up = (low >= 0x8000).to(torch.int32)
    param.detach().view(torch.int16).copy_(_to_int16((bits >> 16) + rounded_up))
    remainder.copy_(_to_int16(low))


def merge_main_param(param: torch.Tensor, remainder: torch.Tensor) -> torch.Tensor:
    """Rebuild the FP32 main param that :func:`split_main_param` stored."""
    assert param.dtype == torch.bfloat16 and remainder.dtype == torch.int16
    low = remainder.to(torch.int32)
    high = param.detach().view(torch.int16).to(torch.int32) - (low < 0).to(torch.int32)
    return ((high << 16) | (low & 0xFFFF)).view(torch.float32)


@triton.jit
def _remainder_update_kernel(
    param_ptr,
    remainder_ptr,
    update_ptr,
    num_elements,
    weight_decay_alpha,
    update_alpha,
    APPLY_WEIGHT_DECAY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    low = tl.load(remainder_ptr + offsets, mask=mask).to(tl.int32)
    high = tl.load(param_ptr + offsets, mask=mask).to(tl.int32)
    high = high - (low < 0).to(tl.int32)
    main = ((high << 16) | (low & 0xFFFF)).to(tl.float32, bitcast=True)
    if APPLY_WEIGHT_DECAY:
        # Same op as ``main.add_(main, alpha=weight_decay_alpha)``.
        main = main + weight_decay_alpha * main
    update = tl.load(update_ptr + offsets, mask=mask)
    main = main + update_alpha * update
    bits = main.to(tl.int32, bitcast=True)
    low = bits & 0xFFFF
    rounded_up = (low >= 0x8000).to(tl.int32)
    high = (bits >> 16) + rounded_up
    tl.store(param_ptr + offsets, (((high + 0x8000) & 0xFFFF) - 0x8000).to(tl.int16), mask=mask)
    tl.store(remainder_ptr + offsets, (((low + 0x8000) & 0xFFFF) - 0x8000).to(tl.int16), mask=mask)


def apply_update_with_remainder(
    param: torch.Tensor,
    remainder: torch.Tensor,
    update: torch.Tensor,
    update_alpha: float,
    weight_decay_alpha: Optional[float] = None,
) -> None:
    """``main += weight_decay_alpha * main; main += update_alpha * update`` on the FP32 main
    param held as ``param`` + ``remainder``, in one pass.

    Each step is the FP32 op ``Tensor.add_(other, alpha=...)`` runs on an FP32 main param, so
    the stored main value matches that path bit for bit.
    """
    assert param.dtype == torch.bfloat16 and remainder.dtype == torch.int16
    assert update.dtype == torch.float32 and update.shape == param.shape
    assert param.is_contiguous() and remainder.is_contiguous()
    update = update.contiguous()
    if not HAVE_TRITON or not param.is_cuda:
        main = merge_main_param(param, remainder)
        if weight_decay_alpha is not None:
            main.add_(main, alpha=weight_decay_alpha)
        main.add_(update, alpha=update_alpha)
        split_main_param(main, param, remainder)
        return
    num_elements = param.numel()
    if num_elements == 0:
        return
    _remainder_update_kernel[(triton.cdiv(num_elements, _BLOCK_SIZE),)](
        param.detach().view(torch.int16),
        remainder,
        update,
        num_elements,
        0.0 if weight_decay_alpha is None else weight_decay_alpha,
        update_alpha,
        APPLY_WEIGHT_DECAY=weight_decay_alpha is not None,
        BLOCK_SIZE=_BLOCK_SIZE,
    )


class Float16OptimizerWithParamRemainders(MixedPrecisionOptimizer):
    """BF16 mixed-precision wrapper that keeps int16 remainders instead of FP32 main params.

    The wrapped optimizer steps the BF16 model params directly: their FP32 gradients are handed
    over as ``param.decoupled_grad`` (the precision-aware convention, so grad-norm and clipping
    read them there) and the FP32 main value of each param is ``param`` + its remainder, which
    the wrapped optimizer rebuilds and writes back on update. It must accept them through
    ``set_param_remainders`` (layer-sharded Muon with ``store_param_remainders=True``). FP32
    params get no remainder: they are their own main params, as in
    :class:`Float16OptimizerWithFloat16Params`.

    Checkpoints carry the rebuilt FP32 main params under ``fp32_from_fp16_params``, the same
    entry :class:`Float16OptimizerWithFloat16Params` writes, so either wrapper loads the
    other's checkpoints.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer stepping BF16 params with remainders.
        config (OptimizerConfig): configuration object for optimizer.
        grad_scaler (MegatronGradScaler): used for scaling gradients; None for BF16.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: Optional[MegatronGradScaler],
        init_state_fn: Callable,
    ):
        super().__init__(optimizer, config, grad_scaler, init_state_fn)
        # Per param group: the BF16 params (also the wrapped optimizer's params), their int16
        # remainders, and the FP32 params, which are their own main params.
        self.float16_groups: List[List[torch.nn.Parameter]] = []
        self.remainder_groups: List[List[torch.Tensor]] = []
        self.fp32_from_fp32_groups: List[List[torch.nn.Parameter]] = []
        if not optimizer:
            self.is_stub_optimizer = True
            return
        if not getattr(optimizer, 'store_param_remainders', False):
            raise TypeError(
                f'{type(optimizer).__name__} does not step BF16 params with remainders; '
                'construct it with store_param_remainders=True.'
            )

        remainders: Dict[int, torch.Tensor] = {}
        for param_group in self.optimizer.param_groups:
            float16_params_this_group = []
            remainders_this_group = []
            fp32_params_this_group = []
            for param in param_group['params']:
                if not param.requires_grad:
                    continue
                if param.type() == 'torch.cuda.FloatTensor':
                    fp32_params_this_group.append(param)
                    continue
                if param.type() != 'torch.cuda.BFloat16Tensor':
                    raise TypeError(
                        'Param remainders need plain torch.cuda.BFloat16Tensor or '
                        'torch.cuda.FloatTensor params. Received {}'.format(param.type())
                    )
                remainder = torch.zeros_like(param, dtype=torch.int16)
                # Start from the FP32 initializer when the model kept one, as the FP32
                # main params of Float16OptimizerWithFloat16Params do.
                get_high_precision_init_val = getattr(param, 'get_high_precision_init_val', None)
                high_precision_init_val = (
                    get_high_precision_init_val()
                    if get_high_precision_init_val is not None
                    else None
                )
                if high_precision_init_val is not None:
                    split_main_param(
                        high_precision_init_val.to(device=param.device, dtype=torch.float32),
                        param.data,
                        remainder,
                    )
                    param.clear_high_precision_init_val()
                float16_params_this_group.append(param)
                remainders_this_group.append(remainder)
                remainders[id(param)] = remainder
            self.float16_groups.append(float16_params_this_group)
            self.remainder_groups.append(remainders_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)
        self.optimizer.set_param_remainders(remainders)
        self.is_stub_optimizer = False

    def _params_and_remainders(self):
        for params, remainders in zip(self.float16_groups, self.remainder_groups, strict=True):
            yield from zip(params, remainders, strict=True)

    def _all_params(self):
        for param, _ in self._params_and_remainders():
            yield param
        for group in self.fp32_from_fp32_groups:
            yield from group

    def zero_grad(self, set_to_none=True):
        """Zero the model grads. ``decoupled_grad`` aliases the persistent DDP ``main_grad``
        buffer, which DDP zeroes itself, so it is left in place."""
        if self.is_stub_optimizer:
            return
        for param in self._all_params():
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()

    def _collect_main_grad_data_for_unscaling(self):
        if self.is_stub_optimizer:
            return
        return [
            param.decoupled_grad.data
            for param in self._all_params()
            if getattr(param, 'decoupled_grad', None) is not None
        ]

    def _copy_model_grads_to_main_grads(self):
        for param in self._all_params():
            if hasattr(param, 'main_grad'):
                param.decoupled_grad = param.main_grad.float()
            elif param.grad is not None:
                param.decoupled_grad = param.grad.float()
            param.grad = None

    def _copy_main_params_to_model_params(self):
        # The wrapped optimizer writes the model params during its step.
        pass

    def _copy_model_params_to_main_params(self, state_dict=None):
        assert state_dict is None, "Initialize main params from state dict is not supported"
        # The BF16 model params become the exact main params.
        for _, remainder in self._params_and_remainders():
            remainder.zero_()

    def _main_params_per_group(self) -> List[List[torch.Tensor]]:
        """Rebuilt FP32 main params, one fresh tensor per param."""
        return [
            [merge_main_param(param, remainder) for param, remainder in zip(params, remainders)]
            for params, remainders in zip(self.float16_groups, self.remainder_groups)
        ]

    def state_dict(self, is_loading: bool = False):
        if is_loading:
            self.init_state_fn(self.optimizer, self.config)

        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['fp32_from_fp16_params'] = self._main_params_per_group()
        return state_dict

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ):
        if is_loading:
            self.init_state_fn(self.optimizer, self.config)

        state_dict = self.state_dict()

        # The wrapped optimizer's params are the model params themselves.
        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict,
            (param for group in self.optimizer.param_groups for param in group['params']),
        )
        _backfill_gtp_sharded_param_map(
            id_to_sharded_param_map, self.float16_groups, model_sharded_state_dict
        )

        # State ids of the trainable BF16 params, skipping FP32 and frozen ones, in group order.
        trainable_param_ids = {id(param) for param, _ in self._params_and_remainders()}
        float16_param_ids_per_group = [
            [
                param_id
                for param_id, param in zip(
                    state_group['params'], inner_group['params'], strict=True
                )
                if id(param) in trainable_param_ids
            ]
            for state_group, inner_group in zip(
                state_dict['optimizer']['param_groups'], self.optimizer.param_groups, strict=True
            )
        ]
        state_dict['fp32_from_fp16_params'] = [
            [
                make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id],
                    main_param,
                    prefix=f'optimizer.state.fp32_param',
                )
                for param_id, main_param in zip(param_ids, main_group, strict=True)
            ]
            for main_group, param_ids in zip(
                state_dict['fp32_from_fp16_params'], float16_param_ids_per_group, strict=True
            )
        ]

        step = self._extract_common_per_param_step(state_dict['optimizer'])
        optim_state_to_sharding_state(
            state_dict['optimizer'], id_to_sharded_param_map, exclude_keys="step"
        )
        if step:
            state_dict['optimizer']['state']['common_step'] = step
        return state_dict

    def load_state_dict(self, state_dict):
        if 'common_step' in state_dict['optimizer']['state']:
            common_step = state_dict['optimizer']['state'].pop('common_step')
            self._restore_common_per_param_step(state_dict['optimizer'], common_step)
        state_dict['optimizer']['param_groups'] = self._filter_and_reorder_param_groups(
            self.optimizer.param_groups, state_dict['optimizer']['param_groups']
        )
        self.optimizer.load_state_dict(state_dict['optimizer'])

        if self.grad_scaler and 'grad_scaler' in state_dict:
            self.grad_scaler.load_state_dict(state_dict['grad_scaler'])

        for params, remainders, saved_group in zip(
            self.float16_groups, self.remainder_groups, state_dict['fp32_from_fp16_params']
        ):
            for param, remainder, saved_main_param in zip(params, remainders, saved_group):
                split_main_param(
                    saved_main_param.data.to(device=param.device, dtype=torch.float32),
                    param.data,
                    remainder,
                )
