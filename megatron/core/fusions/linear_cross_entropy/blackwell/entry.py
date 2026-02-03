# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import logging
import os
import typing
from dataclasses import dataclass, field
from functools import lru_cache

try:
    import cuda.bindings.driver as cuda  # type: ignore
    import cutlass
    import cutlass.cute as cute
    import torch
    import torch.distributed as dist
    import triton  # type: ignore
    from cutlass.cute.runtime import from_dlpack

    import megatron.core.fusions.linear_cross_entropy.utils as utils
    from megatron.core.fusions.linear_cross_entropy.blackwell import (
        bwd_partial_dlogits as bwd_partial_dlogits,
    )
    from megatron.core.fusions.linear_cross_entropy.blackwell import fwd_mainloop as fwd_mainloop
    from megatron.core.fusions.linear_cross_entropy.blackwell import triton as triton_kernels

    @dataclass
    class FwdConfig:
        """
        The configuration for the forward pass.
        """

        _dedicated_stream: torch.cuda.Stream = field(default_factory=torch.cuda.Stream)
        _dedicated_events: typing.List[torch.cuda.Event] = field(default_factory=list)
        _initialized: bool = field(default=False)
        _fwd_mainloop_kernels: typing.Dict[str, cute.kernel] = field(default_factory=dict)
        _vocab_per_split: int = field(
            default=int(os.environ.get("LCE_FWD_VOCAB_SPLIT_SIZE", 512 * 6))
        )

    @dataclass
    class BwdConfig:
        """
        The configuration for the backward pass.
        """

        _bwd_kernel: typing.Dict[str, cute.kernel] = field(default_factory=dict)
        _vocab_per_split: int = field(
            default=int(os.environ.get("LCE_BWD_VOCAB_SPLIT_SIZE", 512 * 6))
        )
        _backward_method: utils.BackwardMethodEnum = field(
            default=utils.BackwardMethodEnum.kDlogitsSplitN
        )

    @lru_cache(maxsize=1)
    def _get_fwd_config() -> FwdConfig:
        """
        Helper function to lazy initialize the forward configuration.
        """
        return FwdConfig()

    @lru_cache(maxsize=1)
    def _get_bwd_config() -> BwdConfig:
        """
        Helper function to lazy initialize the backward configuration.
        """
        return BwdConfig()

    def forward(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        tp_group: typing.Optional[torch.distributed.ProcessGroup] = None,
        reduction: typing.Literal["none", "sum", "mean"] = "mean",
        ignore_index: int = -100,
        sequence_parallel: bool = False,
    ) -> typing.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor
    ]:
        """
        forward host function
        """
        tp_rank = 0 if tp_group is None else torch.distributed.get_rank(tp_group)
        tp_world_size = 1 if tp_group is None else torch.distributed.get_world_size(tp_group)
        in_tp_mode = (tp_group is not None) and (tp_world_size > 1)

        assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
        assert weight.device == hidden.device and labels.device == hidden.device

        # hidden could be [batch, seqlen, dim] or [seqlen, batch, dim] or [tokens, dim]
        assert hidden.dim() == 2 or hidden.dim() == 3
        # weight must be [vocab_size, dim]
        assert weight.dim() == 2
        # labels could be [batch, seqlen] or [seqlen, batch] or [tokens]
        assert (hidden.dim() == 2 and labels.dim() == 1) or (
            hidden.dim() == 3 and labels.dim() == 2
        )
        assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()

        hidden_view = hidden.view(-1, hidden.shape[-1])
        labels_view = labels.view(-1)

        assert (
            sequence_parallel and hidden_view.shape[0] * tp_world_size == labels_view.shape[0]
        ) or (not sequence_parallel and hidden_view.shape[0] == labels_view.shape[0])
        assert hidden_view.shape[1] == weight.shape[1]

        global_hidden = hidden
        if in_tp_mode and sequence_parallel:
            partial_hidden_shape = hidden.shape
            global_hidden_shape = (
                partial_hidden_shape[0] * tp_world_size,
                *partial_hidden_shape[1:],
            )
            global_hidden = torch.empty(
                global_hidden_shape, dtype=hidden.dtype, device=hidden.device
            )
            dist.all_gather_into_tensor(global_hidden, hidden, group=tp_group)
            assert global_hidden.is_contiguous()
            hidden_view = global_hidden.view(-1, global_hidden.shape[-1])

        num_tokens, dim = hidden_view.shape
        vocab_size, _ = weight.shape

        if not _get_fwd_config()._initialized:
            _get_fwd_config()._dedicated_stream = torch.cuda.Stream(hidden.device)
            _get_fwd_config()._dedicated_events = [torch.cuda.Event() for _ in range(2)]
            _get_fwd_config()._initialized = True

        REDUCTION = utils.str_to_reduction_enum(reduction)
        # declare logprobs
        if REDUCTION == utils.EntropyReductionEnum.kNone:
            logprobs = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
            if in_tp_mode:
                logprobs.zero_()
        else:
            logprobs = torch.zeros((), device=hidden.device, dtype=torch.float32)
        # declare auxiliary tensors
        maximum = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
        accumulate = torch.empty_like(maximum, dtype=torch.float32)
        num_valid_tokens = torch.empty((), device=hidden.device, dtype=torch.int64)
        assert (
            maximum.is_contiguous()
            and accumulate.is_contiguous()
            and num_valid_tokens.is_contiguous()
        )
        # declare intermediate tensors
        # NOTE: this is a parameter for tuning
        num_splits = (
            vocab_size + _get_fwd_config()._vocab_per_split - 1
        ) // _get_fwd_config()._vocab_per_split
        _max = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
        _accu = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
        if REDUCTION == utils.EntropyReductionEnum.kNone:
            _logprobs = logprobs
        else:
            _logprobs = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
            if in_tp_mode:
                _logprobs.zero_()
        assert _max.is_contiguous() and _accu.is_contiguous() and _logprobs.is_contiguous()

        triton_kernels.get_num_valid_tokens[(1,)](
            num_tokens, ignore_index, labels_view, labels_view.stride(0), num_valid_tokens
        )

        # need to compile the kernel for the first time
        hidden_packed = from_dlpack(
            hidden_view.detach(), assumed_align=16
        ).mark_compact_shape_dynamic(mode=0)
        weight_packed = from_dlpack(weight.detach(), assumed_align=16)
        labels_packed = from_dlpack(
            labels_view.detach(), assumed_align=8
        ).mark_compact_shape_dynamic(mode=0)
        logprobs_packed = from_dlpack(_logprobs, assumed_align=16).mark_compact_shape_dynamic(
            mode=0
        )
        _max_packed = from_dlpack(_max, assumed_align=8).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        _accu_packed = from_dlpack(_accu, assumed_align=8).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        cuda_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # VocabSize and Dim are fixed for a given model,
        # only the number of tokens can vary
        key = f"vocab_size:{vocab_size}+dim:{dim}+dtype:{hidden_view.dtype}"
        if _get_fwd_config()._fwd_mainloop_kernels.get(key) is None:
            fwd_mainloop_kernel = fwd_mainloop.FwdMainLoop(
                vocab_per_split=_get_fwd_config()._vocab_per_split
            )
            fwd_mainloop_compiled_kernel = cute.compile(
                fwd_mainloop_kernel,
                hidden_packed,
                weight_packed,
                labels_packed,
                logprobs_packed,
                _max_packed,
                _accu_packed,
                ignore_index,
                tp_rank,
                cuda_stream,
            )
            _get_fwd_config()._fwd_mainloop_kernels[key] = fwd_mainloop_compiled_kernel
        else:
            fwd_mainloop_compiled_kernel = _get_fwd_config()._fwd_mainloop_kernels[key]
        fwd_mainloop_compiled_kernel(
            hidden_packed,
            weight_packed,
            labels_packed,
            logprobs_packed,
            _max_packed,
            _accu_packed,
            ignore_index,
            tp_rank,
            cuda_stream,
        )

        if not in_tp_mode:

            def grid(meta):
                return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]),)

            triton_kernels.forward_dp_epilogue[grid](
                num_tokens,
                num_splits,
                ignore_index,
                labels_view,
                labels_view.stride(0),
                num_valid_tokens,
                _max,
                _max.stride(0),
                _max.stride(1),
                _accu,
                _accu.stride(0),
                _accu.stride(1),
                maximum,
                maximum.stride(0),
                accumulate,
                maximum.stride(0),
                _logprobs,
                _logprobs.stride(0),
                logprobs,
                triton.language.constexpr(REDUCTION.value),
            )
        else:
            _max_backup = _max.clone()
            dist.all_reduce(_max, op=dist.ReduceOp.MAX, group=tp_group)

            torch.cuda.current_stream().record_event(_get_fwd_config()._dedicated_events[0])
            with torch.cuda.stream(_get_fwd_config()._dedicated_stream):
                _get_fwd_config()._dedicated_stream.wait_event(
                    _get_fwd_config()._dedicated_events[0]
                )
                dist.all_reduce(_logprobs, op=dist.ReduceOp.SUM, group=tp_group)
                _get_fwd_config()._dedicated_stream.record_event(
                    _get_fwd_config()._dedicated_events[1]
                )

            def grid(meta):
                return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]),)

            triton_kernels.forward_tp_epilogue[grid](
                num_tokens,
                num_splits,
                _max,
                _max.stride(0),
                _max.stride(1),
                _max_backup,
                _max_backup.stride(0),
                _max_backup.stride(1),
                _accu,
                _accu.stride(0),
                _accu.stride(1),
                maximum,
                maximum.stride(0),
                accumulate,
                maximum.stride(0),
            )
            # reduce accumulate
            dist.all_reduce(accumulate, op=dist.ReduceOp.SUM, group=tp_group)

            # update logprobs
            torch.cuda.current_stream().wait_event(_get_fwd_config()._dedicated_events[1])
            triton_kernels.forward_tp_epilogue_update_logprobs[grid](
                num_tokens,
                ignore_index,
                num_valid_tokens,
                labels_view,
                labels_view.stride(0),
                _logprobs,
                _logprobs.stride(0),
                maximum,
                maximum.stride(0),
                accumulate,
                accumulate.stride(0),
                logprobs,
                REDUCTION.value,
            )

        return (
            logprobs,
            maximum,
            accumulate,
            num_valid_tokens,
            tp_rank,
            tp_world_size,
            global_hidden,
        )

    def backward(
        dlogprobs: torch.Tensor,
        global_hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        maximum: torch.Tensor,
        accu: torch.Tensor,
        num_valid_tokens: torch.Tensor,
        reduction: typing.Literal["none", "sum", "mean"] = "mean",
        ignore_index: int = -100,
        tp_group: typing.Optional[dist.ProcessGroup] = None,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        sequence_parallel: bool = False,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        backward host function
        """
        in_tp_mode = (tp_group is not None) and (tp_world_size > 1)

        hidden_view = global_hidden.view(-1, global_hidden.shape[-1])
        labels_view = labels.view(-1)

        num_tokens, dim = hidden_view.shape
        vocab_size, _ = weight.shape

        REDUCTION = utils.str_to_reduction_enum(reduction)
        dlogprobs_view = dlogprobs.view(-1)
        assert (
            REDUCTION == utils.EntropyReductionEnum.kNone and dlogprobs.shape == (num_tokens,)
        ) or (REDUCTION != utils.EntropyReductionEnum.kNone and dlogprobs.dim() == 0)
        assert dlogprobs.is_contiguous() and dlogprobs.is_cuda

        assert (
            num_valid_tokens.dim() == 0
            and num_valid_tokens.is_cuda
            and num_valid_tokens.dtype == torch.int64
        )

        d_hidden = torch.empty_like(global_hidden)
        d_weight = torch.empty_like(weight)
        assert d_hidden.is_contiguous() and d_weight.is_contiguous()

        # FIXME: implement different backward methods
        _backward_method = _get_bwd_config()._backward_method
        if _backward_method == utils.BackwardMethodEnum.kDlogitsSplitN:
            vocab_per_split = _get_bwd_config()._vocab_per_split
            num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

            _d_logits = torch.empty(
                (num_tokens, vocab_per_split),
                device=global_hidden.device,
                dtype=global_hidden.dtype,
            )

            hidden_packed = from_dlpack(
                hidden_view.detach(), assumed_align=16
            ).mark_compact_shape_dynamic(mode=0)
            weight_packed = from_dlpack(weight.detach(), assumed_align=16)
            labels_packed = from_dlpack(
                labels_view.detach(), assumed_align=8
            ).mark_compact_shape_dynamic(mode=0)
            dlogprobs_packed = from_dlpack(
                dlogprobs_view.detach(), assumed_align=8
            ).mark_compact_shape_dynamic(mode=0)
            maximum_packed = from_dlpack(
                maximum.detach(), assumed_align=8
            ).mark_compact_shape_dynamic(mode=0)
            accu_packed = from_dlpack(accu.detach(), assumed_align=8).mark_compact_shape_dynamic(
                mode=0
            )
            dlogits_packed = from_dlpack(_d_logits, assumed_align=32).mark_compact_shape_dynamic(
                mode=0
            )
            scalarNumValidTokens_packed = cute.runtime.make_ptr(
                cutlass.Int64, num_valid_tokens.data_ptr(), cute.AddressSpace.gmem, assumed_align=8
            )

            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

            key = (
                f"vocab_size:{vocab_size}+dim:{dim}+reduction:{REDUCTION}+dtype:{hidden_view.dtype}"
            )
            if _get_bwd_config()._bwd_kernel.get(key) is None:
                bwd_kernel = bwd_partial_dlogits.BwdPartialDlogits(
                    reduction=REDUCTION.value, vocab_per_split=vocab_per_split
                )
                bwd_kernel_compiled = cute.compile(
                    bwd_kernel,
                    0,  # split_idx
                    hidden_packed,
                    weight_packed,
                    labels_packed,
                    dlogprobs_packed,
                    maximum_packed,
                    accu_packed,
                    dlogits_packed,
                    scalarNumValidTokens_packed,
                    ignore_index,
                    tp_rank,
                    stream,
                )
                _get_bwd_config()._bwd_kernel[key] = bwd_kernel_compiled
            else:
                bwd_kernel_compiled = _get_bwd_config()._bwd_kernel.get(key)

            for split_idx in range(num_splits):
                bwd_kernel_compiled(
                    split_idx,
                    hidden_packed,
                    weight_packed,
                    labels_packed,
                    dlogprobs_packed,
                    maximum_packed,
                    accu_packed,
                    dlogits_packed,
                    scalarNumValidTokens_packed,
                    ignore_index,
                    tp_rank,
                    stream,
                )
                # remove padding areas
                # cublas can handle non-contiguous tensors
                # therefore, we do not need to contiguous the tensor
                vocab_right_bound = (
                    min((split_idx + 1) * vocab_per_split, vocab_size) - split_idx * vocab_per_split
                )
                valid_d_logits = _d_logits[:, :vocab_right_bound]

                torch.addmm(
                    input=d_hidden.view(-1, dim),
                    mat1=valid_d_logits,
                    mat2=weight[split_idx * vocab_per_split : (split_idx + 1) * vocab_per_split, :],
                    beta=(split_idx != 0),
                    alpha=1.0,
                    out=d_hidden.view(-1, dim),
                )
                torch.matmul(
                    valid_d_logits.T,
                    hidden_view,
                    out=d_weight[
                        split_idx * vocab_per_split : (split_idx + 1) * vocab_per_split, :
                    ],
                )
        else:
            raise NotImplementedError(f"Unsupported backward method: {_backward_method}")

        if in_tp_mode:
            dist.all_reduce(d_hidden, op=dist.ReduceOp.SUM, group=tp_group)
            if sequence_parallel:
                partial_hidden_shape = (
                    global_hidden.shape[0] // tp_world_size,
                    *global_hidden.shape[1:],
                )
                partial_num_tokens = num_tokens // tp_world_size
                d_hidden = d_hidden.view(-1, d_hidden.shape[-1])[
                    tp_rank * partial_num_tokens : (tp_rank + 1) * partial_num_tokens, :
                ]
                d_hidden = d_hidden.view(partial_hidden_shape).clone()

        return d_hidden, d_weight

except ImportError:
    logging.warning(
        "Cutlass or CUDA bindings not found. LinearCrossEntropy Blackwell entry "
        "points will not be available."
    )
