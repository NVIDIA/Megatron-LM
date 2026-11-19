# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License for these changes: BSD-3-Clause.
"""Opt-in exact FA2 forward/LSE with native cuDNN SBHD backward.

Only the qualified BF16 self-attention geometry is selected. Other calls retain
TE's original backend, inputs, return value and process groups unchanged.
"""

import dis
import inspect
import types
import weakref
from typing import Any

import torch

_SAVED_LSE_STATISTICS: list[dict[str, Any]] = []


def saved_lse_statistics():
    """Return scalar execution counters without retaining models or CUDA tensors."""
    return [{**row['geometry'], **row['counters']} for row in _SAVED_LSE_STATISTICS]


class _NativeForwardCudnnBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, module, bound, original, stats, rng_state):
        import transformer_engine.pytorch.attention.dot_product_attention.backends as backend

        saved = []
        flash = backend.flash_attn_func

        def capture(*args, **kwargs):
            assert not kwargs.get('return_attn_probs', False)
            kwargs['return_attn_probs'] = True
            output, lse, unused_mask = flash(*args, **kwargs)
            assert unused_mask.numel() == 0
            assert lse.dtype == torch.float32 and lse.shape == (q.shape[1], 32, q.shape[0])
            saved.append((*args[:3], output, lse))
            return output

        # Run the actual native body without changing its class or process-global
        # FA2 alias. Mutable referenced objects retain their native identities.
        function = original.__func__
        namespace = dict(function.__globals__)
        assert namespace['flash_attn_func'] is flash
        namespace['flash_attn_func'] = capture
        invoke = types.FunctionType(
            function.__code__,
            namespace,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
        invoke.__kwdefaults__ = function.__kwdefaults__
        output = types.MethodType(invoke, module)(*bound.args, **bound.kwargs)
        assert backend.flash_attn_func is flash
        assert len(saved) == 1
        fq, fk, fv, fo, lse = saved[0]
        assert all(x.is_contiguous() for x in (fq, fk, fv, fo, lse))
        ctx.save_for_backward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            output.reshape(q.shape).contiguous(),
            lse.unsqueeze(-1),
            bound.arguments['cu_seqlens_q'],
            bound.arguments['cu_seqlens_kv'],
            rng_state,
        )
        ctx.scale = module.softmax_scale
        ctx.deterministic = module.deterministic
        ctx.window = bound.arguments['window_size'] or (-1, 0)
        ctx.stats = stats
        stats['forward'] += 1
        return output

    @staticmethod
    def backward(ctx, grad):
        import transformer_engine.pytorch.attention.dot_product_attention.backends as backend

        q, k, v, o, lse, cuq, cukv, rng = ctx.saved_tensors
        do = grad.reshape(q.shape).contiguous()
        dq, dk, dv, *unused = backend.fused_attn_bwd(
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k.shape[0],
            cu_seqlens_q=cuq,
            cu_seqlens_kv=cukv,
            q=q,
            k=k,
            v=v,
            o=o,
            d_o=do,
            fake_dtype=q.dtype,
            aux_ctx_tensors=[lse, rng],
            fused_attention_backend=backend.FusedAttnBackend['F16_arbitrary_seqlen'],
            attn_scale=ctx.scale,
            dropout=0.0,
            fast_zero_fill=True,
            qkv_layout='sbhd_sbhd_sbhd',
            o_format='sbhd',
            do_format='sbhd',
            dqkv_layout='sbhd_sbhd_sbhd',
            attn_bias_type='no_bias',
            attn_mask_type='causal',
            softmax_type='vanilla',
            window_size=ctx.window,
            bottom_right_diagonal=False,
            deterministic=ctx.deterministic,
            cuda_graph=torch.cuda.is_current_stream_capturing(),
        )
        ctx.stats['backward'] += 1
        return dq, dk, dv, None, None, None, None, None


def install_saved_lse_attention(owner: Any) -> bool:
    """Wrap only this TEDotProductAttention's FlashAttention child, if compatible.

    Called after the native TE constructor, only for an explicit config opt-in.
    Returns whether an instance wrapper was installed; backend choice is still TE's.
    """
    if not getattr(owner.config, 'attention_saved_lse_cudnn_backward', False):
        return False
    import transformer_engine.pytorch.attention.dot_product_attention.backends as backend

    flash: Any = getattr(owner, 'flash_attention', None)
    if type(flash) is not backend.FlashAttention:
        return False
    if getattr(flash, '_saved_lse_adapter_installed', False):
        return True
    original = flash.forward
    if not isinstance(original, types.MethodType) or not isinstance(
        original.__func__, types.FunctionType
    ):
        return False
    if original.__func__.__closure__ is not None:
        return False
    if any(
        op.opname in ('STORE_GLOBAL', 'DELETE_GLOBAL')
        for op in dis.get_instructions(original.__func__)
    ):
        return False  # Snapshotting a body that mutates its global namespace is unsupported.
    # This integration needs the fixed-length FA2 call in the undecorated native body.
    if (
        original.__func__.__globals__ is not vars(backend)
        or 'flash_attn_func' not in original.__func__.__code__.co_names
    ):
        return False
    signature = inspect.signature(original)
    required = {
        'query_layer',
        'key_layer',
        'value_layer',
        'qkv_layout',
        'cp_group',
        'attn_mask_type',
        'fp8',
        'fp8_output',
        'pad_between_seqs',
        'inference_params',
        'attention_mask',
        'alibi_slopes',
        'window_size',
        'cu_seqlens_q',
        'cu_seqlens_kv',
        'flash_attention_backend',
        'max_seqlen_q',
        'max_seqlen_kv',
    }
    if not required <= signature.parameters.keys() or not callable(
        getattr(backend, 'fused_attn_bwd', None)
    ):
        return False
    reference = weakref.ref(owner)
    rng_states = {}
    stats = dict(forward=0, backward=0, fallback=0)

    def forward(module, *args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        d = bound.arguments
        q, k, v = (d[name] for name in ('query_layer', 'key_layer', 'value_layer'))
        parent = reference()
        config = parent.config if parent is not None else None
        tp = getattr(parent, '_tp_group', None)
        cp = d['cp_group']
        supported = (
            config is not None
            and parent.training
            and module.training
            and torch.is_grad_enabled()
            and config.tensor_model_parallel_size == config.context_parallel_size == 1
            and tp is not None
            and tp.size() == 1
            and not config.fp8
            and not config.fp4
            and config.softmax_type == 'vanilla'
            and not config.qk_clip
            and not config.log_max_attention_logit
            and not torch.cuda.is_current_stream_capturing()
            and not torch.compiler.is_compiling()
            and type(q) is type(k) is type(v) is torch.Tensor
            and q.ndim == k.ndim == v.ndim == 4
            and q.device == k.device == v.device
            and q.is_cuda
            and q.dtype == k.dtype == v.dtype == torch.bfloat16
            and q.shape[:2] == k.shape[:2] == v.shape[:2]
            and tuple(q.shape[-2:]) == (32, 128)
            and tuple(k.shape[-2:]) == tuple(v.shape[-2:]) == (2, 128)
            and tuple(q.shape[:2]) in ((512, 1), (8192, 1), (8192, 4))
            and d['qkv_layout'] == 'sbhd_sbhd_sbhd'
            and d['attn_mask_type'] == 'causal'
            and module.attention_type == 'self'
            and module.attention_dropout == 0
            and not d['fp8']
            and not d['fp8_output']
            and not d['pad_between_seqs']
            and d['inference_params'] is None
            and d['attention_mask'] is None
            and d['alibi_slopes'] is None
            and d.get('softcap', 0.0) == 0.0
            and (cp is None or (not isinstance(cp, list) and cp.size() == 1))
            and d['window_size'] in (None, (-1, -1), (-1, 0))
            and d['max_seqlen_q'] == q.shape[0]
            and d['max_seqlen_kv'] == k.shape[0]
            and getattr(d['flash_attention_backend'], 'major', None) == 2
        )
        if supported:
            # Use TE's canonical full rectangular metadata by identity; never inspect
            # CUDA values or accept arbitrary ragged cu_seqlens under an SBHD label.
            cu = backend.dpa_utils.get_full_cu_seqlens(q.shape[1], q.shape[0], q.device)
            supported = cu._version == 0 and all(
                d[name] is None or d[name] is cu for name in ('cu_seqlens_q', 'cu_seqlens_kv')
            )
        if not supported:
            stats['fallback'] += 1
            return original(*args, **kwargs)
        bound.arguments['cu_seqlens_q'] = cu
        bound.arguments['cu_seqlens_kv'] = cu
        if q.device not in rng_states:
            rng_states[q.device] = torch.zeros(2, dtype=torch.int64, device=q.device)
        return _NativeForwardCudnnBackward.apply(
            q, k, v, module, bound, original, stats, rng_states[q.device]
        )

    flash.forward = types.MethodType(forward, flash)
    flash._saved_lse_adapter_installed = True
    flash._saved_lse_adapter_statistics = stats
    _SAVED_LSE_STATISTICS.append(
        dict(
            counters=stats,
            geometry=dict(
                num_attention_heads=owner.config.num_attention_heads,
                num_query_groups=owner.config.num_query_groups,
                kv_channels=owner.config.kv_channels,
            ),
        )
    )
    return True
