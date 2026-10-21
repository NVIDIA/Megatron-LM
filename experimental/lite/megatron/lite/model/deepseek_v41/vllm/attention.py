# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Caller-owned sparse FlashMLA with native parameter names and owners."""
import torch

from .linear import linear


def _rotate(x, positions, cache, inverse=False):
    width = cache.shape[-1]
    angles = cache[positions].float()
    cos, sin = angles.chunk(2, -1)
    while cos.ndim < x.ndim:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    if inverse:
        sin = -sin
    tail = x[..., -width:].float().unflatten(-1, (-1, 2))
    a, b = tail.unbind(-1)
    rotated = torch.stack((a * cos - b * sin, b * cos + a * sin), -1).flatten(-2)
    return torch.cat((x[..., :-width], rotated.to(x.dtype)), -1)


class _QueryInputs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw, qw, kw, eps):
        from vllm.model_executor.kernels.batch_invariant import rms_norm_cuda
        from vllm.models.deepseek_v41.common.ops.query_quant import fused_q_kv_rmsnorm_quant

        q, kv = raw.split((qw.numel(), kw.numel()), -1)
        qa, native_kv = fused_q_kv_rmsnorm_quant(q, kv, qw.bfloat16(), kw.bfloat16(), eps)
        qr = rms_norm_cuda(q.contiguous(), qw.bfloat16(), eps)
        ctx.save_for_backward(q, kv, qw, kw)
        ctx.eps = eps
        ctx.mark_non_differentiable(qa.data, qa.scale)
        return qr, native_kv, qa.data, qa.scale

    @staticmethod
    def backward(ctx, dq, dk, unused_codes, unused_scales):
        q, kv, qw, kw = ctx.saved_tensors

        def norm_vjp(x, w, grad):
            x = x.float()
            g = grad.float()
            inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + ctx.eps)
            scaled = g * w.bfloat16().float()
            dx = inv * (scaled - x * inv.square() * (scaled * x).mean(-1, keepdim=True))
            return dx, (g * x * inv).sum(0).to(w.dtype)

        a, b = norm_vjp(q, qw, dq)
        c, d = norm_vjp(kv, kw, dk)
        return torch.cat((a, c), -1).to(q.dtype), b, d, None


class _SWAInsert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, positions, cache, eps):
        ctx.empty = q.shape[0] == 0
        if ctx.empty:
            if kv.shape[0] != 0:
                raise ValueError('Empty SWA queries require empty KV rows')
            return q.clone(), kv.clone()
        from vllm.models.deepseek_v41.common.ops.cache_utils import dequantize_and_gather_k_cache

        rows, heads, dim = q.shape
        block = 64
        pages = (rows + block - 1) // block
        table = torch.arange(pages, dtype=torch.int32, device=q.device)[None]
        slots = torch.arange(rows, dtype=torch.int64, device=q.device)
        packed = torch.zeros((pages, block, 528), dtype=torch.uint8, device=q.device)
        padded = 64 if heads <= 64 else 128
        rotated = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q.contiguous(),
            kv.contiguous(),
            packed.view(pages, -1),
            slots,
            positions,
            cache,
            padded,
            eps,
            block,
            False,
            True,
            True,
            False,
        )
        decoded = kv.new_empty((1, rows, dim))
        lengths = torch.tensor([rows], dtype=torch.int32, device=q.device)
        dequantize_and_gather_k_cache(decoded, packed, lengths, None, table, block, 0)
        ctx.save_for_backward(positions, cache)
        return rotated[:, :heads], decoded[0]

    @staticmethod
    def backward(ctx, dq, dkv):
        if ctx.empty:
            return dq, dkv, None, None, None
        positions, cache = ctx.saved_tensors
        return (
            _rotate(dq, positions, cache, True),
            _rotate(dkv, positions, cache, True),
            None,
            None,
            None,
        )


class _Sparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, sink, indices, lengths, scale):
        from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

        heads = q.shape[1]
        padded = 64 if heads <= 64 else 128
        query = torch.nn.functional.pad(q, (0, 0, 0, padded - heads)).contiguous()
        bias = torch.nn.functional.pad(sink.float(), (0, padded - heads), value=-torch.inf)
        output = torch.empty_like(query)
        flash_mla_sparse_fwd(
            q=query,
            kv=kv[:, None].contiguous(),
            indices=indices[:, None],
            sm_scale=scale,
            attn_sink=bias,
            topk_length=lengths,
            out=output,
        )
        ctx.save_for_backward(q, kv, sink, indices)
        ctx.scale = scale
        return output[:, :heads]

    @staticmethod
    def backward(ctx, grad):
        q, kv, sink, indices = ctx.saved_tensors
        if q.shape[0] == 0:
            return (
                torch.zeros_like(q),
                torch.zeros_like(kv),
                torch.zeros_like(sink),
                None,
                None,
                None,
            )
        if not q.is_cuda or q.shape[-1] != 512:
            # CPU/small-width common-activation reference; production accepts D512.
            with torch.enable_grad():
                query = q.detach().float().requires_grad_()
                values = kv.detach().float().requires_grad_()
                bias = sink.detach().float().requires_grad_()
                gathered = values[indices.clamp_min(0).long()]
                logits = (query @ gathered.transpose(-1, -2)) * ctx.scale
                logits = logits.masked_fill((indices < 0)[:, None], -torch.inf)
                logits = torch.cat((logits, bias[None, :, None].expand(q.shape[0], -1, 1)), -1)
                out = logits.softmax(-1)[..., :-1] @ gathered
                grads = torch.autograd.grad(out, (query, values, bias), grad.float())
            return (
                *(value.to(source.dtype) for value, source in zip(grads, (q, kv, sink))),
                None,
                None,
                None,
            )
        # Replay exact FlashMLA statistics for the whole-query cuDNN adjoint.
        from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

        from megatron.core.transformer.experimental_attention_variant.csa_utils import (
            fused_sparse_attention as sparse,
        )

        heads = q.shape[1]
        padded = 64 if heads <= 64 else 128
        query = torch.nn.functional.pad(q, (0, 0, 0, padded - heads)).contiguous()
        bias = torch.nn.functional.pad(sink.float(), (0, padded - heads), value=-torch.inf)
        lengths = None
        output, _, lse = flash_mla_sparse_fwd(
            q=query,
            kv=kv[:, None].contiguous(),
            indices=indices[:, None],
            sm_scale=ctx.scale,
            attn_sink=bias,
            topk_length=lengths,
        )
        dout = torch.nn.functional.pad(grad, (0, 0, 0, padded - heads)).contiguous()
        sparse._ensure_dsa_namespace()
        result = sparse._DSA.sparse_attention_backward_wrapper(
            query,
            kv,
            output,
            dout,
            lse,
            bias,
            indices,
            softmax_scale=ctx.scale,
            topk_length=lengths,
            **sparse._deterministic_sparse_bwd_kwargs(
                query,
                kv,
                output,
                lse,
                bias,
                indices,
                lengths,
                ctx.scale,
                requested=torch.are_deterministic_algorithms_enabled(),
            ),
        )
        return (
            result["dq"][:, :heads],
            result["dkv"].to(kv.dtype),
            result["d_sink"][:heads].to(sink.dtype),
            None,
            None,
            None,
        )


class _InverseRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, positions, cache, groups):
        if not x.is_cuda:
            raise RuntimeError("Deployment inverse RoPE requires CUDA")
        from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
            fused_inv_rope_fp8_quant,
        )

        ctx.save_for_backward(positions, cache)
        ctx.shape = x.shape
        out, _ = fused_inv_rope_fp8_quant(
            x,
            positions,
            cache,
            n_groups=groups,
            heads_per_group=x.shape[1] // groups,
            nope_dim=x.shape[-1] - cache.shape[-1],
            rope_dim=cache.shape[-1],
            quant_group_size=128,
            tma_aligned_scales=True,
            quantize=False,
        )
        return out

    @staticmethod
    def backward(ctx, grad):
        positions, cache = ctx.saved_tensors
        return _rotate(grad.reshape(ctx.shape), positions, cache), None, None, None


def inverse_rope_bf16(x, positions, cache, groups):
    """LoRA input: native BF16 inverse-RoPE before the base path FP8 quantizer."""
    return _InverseRoPE.apply(x, positions, cache, groups)


class _FP8OutputProjection(torch.autograd.Function):
    """Native inverse-RoPE/FP8 einsum, with the decoded-operand STE adjoint."""

    @staticmethod
    def forward(ctx, x, weight, positions, cache, groups):
        if not x.is_cuda or not weight.is_cuda:
            raise RuntimeError("Deployment FP8 wo_a requires CUDA")
        from types import SimpleNamespace

        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            deepgemm_post_process_fp8_weight_block,
        )
        from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
            fused_inv_rope_fp8_quant,
        )
        from vllm.models.deepseek_v4.nvidia.ops.o_proj import _compile_einsum_configs
        from vllm.utils.deep_gemm import fp8_einsum

        from megatron.lite.model.deepseek_v41.lite.export import to_release

        heads, dim = x.shape[-2:]
        inner = heads * dim // groups
        rank = weight.numel() // (groups * inner)
        ctx.shape, ctx.weight_shape, ctx.weight_dtype = (x.shape, weight.shape, weight.dtype)
        ctx.like = weight
        if x.shape[0] == 0:
            ctx.save_for_backward(
                x.new_empty((0, groups, inner)),
                weight.reshape(groups, rank, inner),
                positions,
                cache,
            )
            return x.new_empty((0, groups, rank))
        encoded, scale = fused_inv_rope_fp8_quant(
            x,
            positions,
            cache,
            n_groups=groups,
            heads_per_group=heads // groups,
            nope_dim=dim - cache.shape[-1],
            rope_dim=cache.shape[-1],
            quant_group_size=32,
            tma_aligned_scales=True,
            quantize=True,
        )
        # Exactly the serialization used for live publication, including its
        # canonicalization. Do not independently requantize each output row.
        tensors = dict(
            to_release("layers.0.attn.wo_a.weight", weight.reshape(groups * rank, inner))
        )
        codes, descale = (tensors["layers.0.attn.wo_a.weight"], tensors["layers.0.attn.wo_a.scale"])
        saved_codes, saved_scale = codes.clone(), descale.clone()
        codes, native_scale = deepgemm_post_process_fp8_weight_block(
            codes, descale, (32, 32), True, is_bmm=True, bmm_batch_size=groups
        )
        _compile_einsum_configs(
            x.shape[0],
            SimpleNamespace(weight=codes),
            native_scale,
            cache,
            n_groups=groups,
            heads_per_group=heads // groups,
            nope_dim=dim - cache.shape[-1],
            rope_dim=cache.shape[-1],
            o_lora_rank=rank,
            einsum_recipe=(1, 1, 32),
            tma_aligned_scales=True,
        )
        result = x.new_empty((x.shape[0], groups, rank))
        fp8_einsum(
            "bhr,hdr->bhd", (encoded, scale), (codes, native_scale), result, recipe=(1, 1, 32)
        )
        ctx.compact, ctx.input_dtype = True, x.dtype
        ctx.save_for_backward(encoded, scale, saved_codes, saved_scale, positions, cache)
        return result

    @staticmethod
    def backward(ctx, grad):
        if getattr(ctx, 'compact', False):
            from megatron.lite.primitive.quantization.block_fp8 import dequantize_block_fp8

            encoded, scale, codes, sf, positions, cache = ctx.saved_tensors
            groups, rank = grad.shape[-2:]
            inner = codes.shape[-1]
            # FP8 code times a power of two is exact in BF16: decode all groups at once.
            dtype = torch.bfloat16 if ctx.input_dtype == torch.bfloat16 else torch.float32
            rows = encoded.shape[0]
            raw = scale.contiguous().view(torch.uint8).reshape(rows, groups, -1)[..., : inner // 32]
            sf_rows = (raw.to(torch.int32) << 23).view(torch.float32).to(dtype)
            activation = (
                (encoded.to(dtype).reshape(rows, groups, -1, 32) * sf_rows.unsqueeze(-1))
                .reshape(rows, groups, inner)
                .to(ctx.input_dtype)
            )
            weight = (
                dequantize_block_fp8(codes, sf, (32, 32))
                .reshape(groups, rank, inner)
                .to(ctx.input_dtype)
            )
        else:
            activation, weight, positions, cache = ctx.saved_tensors
        with torch.autocast(device_type=grad.device.type, enabled=False):
            g = grad.permute(1, 0, 2).bfloat16()
            a = activation.permute(1, 0, 2).bfloat16()
            w = weight.bfloat16()

            def bmm(left, right):
                return (
                    torch.bmm(left, right, out_dtype=torch.float32)
                    if left.is_cuda
                    else torch.bmm(left.float(), right.float())
                )

            dx = bmm(g, w).permute(1, 0, 2).to(activation.dtype)
            dw = bmm(g.transpose(1, 2), a).reshape(ctx.weight_shape)
        like = getattr(ctx, 'like', None)
        from .linear import accumulate_wgrad

        return (
            _rotate(dx.reshape(ctx.shape), positions, cache),
            dw.to(ctx.weight_dtype) if like is None else accumulate_wgrad((like,), dw, like),
            None,
            None,
            None,
        )


def _projections(attention, x):
    raw = linear(
        x, (attention.linear_q_down_proj.weight, attention.linear_kv_proj.weight), fp8=True
    )
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

    qr, kv, codes, scales = _QueryInputs.apply(
        raw,
        attention.q_layernorm.weight,
        attention.kv_layernorm.weight,
        attention.config.layernorm_epsilon,
    )
    qa = QuantizedActivation(codes, scales, qr.dtype, qr.shape, kMxfp8Dynamic)
    q = linear(qr, (attention.linear_q_up_proj.weight,), fp8=True, activation=qa).bfloat16()
    q = q.reshape(-1, attention.num_attention_heads_per_partition, attention.q_head_dim)
    from .linear import _row_scales

    return raw, qr, kv, q, qa, _row_scales(qa.scale, *qr.shape)


def _projection_rows(projected, lo, hi):
    from dataclasses import replace

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale

    raw, qr, kv, q, qa, scales = projected
    sliced = replace(
        qa,
        data=qa.data[lo:hi],
        scale=swizzle_mxfp8_scale(scales[lo:hi].contiguous(), hi - lo, qr.shape[-1]),
        orig_shape=qr[lo:hi].shape,
    )
    return raw[lo:hi], qr[lo:hi], kv[lo:hi], q[lo:hi], sliced


def _documents(documents, device):
    """THD document row ranges as vLLM requests: per-document positions, query start
    locations and lengths, plus per-ratio index remaps (see :func:`swa_forward`)."""
    from types import SimpleNamespace

    lengths = [hi - lo for lo, hi in documents]
    starts = [lo for lo, _ in documents] + [documents[-1][1] if documents else 0]
    return SimpleNamespace(
        documents=documents,
        lengths=lengths,
        positions=torch.cat(
            [torch.zeros(0, dtype=torch.int64)] + [torch.arange(n) for n in lengths]
        ).to(device),
        cu=torch.tensor(starts, dtype=torch.int32, device=device),
        seq=torch.tensor(lengths, dtype=torch.int32, device=device),
        remap={},
    )


def _remap(layout, ratio, width, total, device):
    """``combine_topk_swa_indices`` addresses request ``b``'s workspace as
    ``M*b + [compressed (N rows) | SWA rows]``. Map that to the packed KV
    ``[every document's compressed rows | every THD row]`` actually passed."""
    if ratio not in layout.remap:
        counts = [n // ratio if ratio else 0 for n in layout.lengths]
        first = [sum(counts[:d]) for d in range(len(counts))]
        rows = [lo for lo, _ in layout.documents]
        stride = width + max(layout.lengths, default=0)
        v = torch.arange(len(counts) * stride, device=device)
        b, r = v // stride, v % stride
        compressed = torch.tensor(first, dtype=torch.int64, device=device)[b] + r
        swa = total + torch.tensor(rows, dtype=torch.int64, device=device)[b] + r - width
        layout.remap[ratio] = stride, torch.where(r < width, compressed, swa).int()
    return layout.remap[ratio]


def swa_forward(
    attention,
    hidden,
    positions,
    rope_cache,
    *,
    observe=None,
    source=None,
    projected=None,
    layout=None,
    project=True,
):
    """All packed documents in one pass: every op is row-wise or per request, so each
    document's rows carry the bits of a single-document call."""
    emit = observe or (lambda name, value: None)
    x = hidden.reshape(-1, hidden.shape[-1])
    raw, qr, kv, q, qa = (_projections(attention, x) if projected is None else projected)[:5]
    for name, value in (
        ('qkv_projection', raw),
        ('q_norm', qr),
        ('kv_norm', kv),
        ('q_projection', q),
    ):
        emit(name, value)
    q, kv = _SWAInsert.apply(q, kv, positions, rope_cache, attention.config.layernorm_epsilon)
    emit('q_rope', q)
    emit('kv_cache_decoded', kv)
    from vllm.models.deepseek_v41.common.ops import combine_topk_swa_indices

    rows = x.shape[0]
    if layout is None:
        layout = _documents([(0, rows)], x.device)
    ratio, width, total = 0, 0, 0
    topk = torch.empty((rows, 0), device=x.device, dtype=torch.int32)
    if source is not None:
        compressed, topk, ratio = source(x, qr, qa)
        width = max((n // ratio for n in layout.lengths), default=0)
        total = compressed.shape[0]
        kv = torch.cat((compressed, kv), 0)
    stride, table = _remap(layout, ratio, width, total, x.device)
    indices, lens = combine_topk_swa_indices(
        topk,
        layout.cu,
        layout.seq,
        layout.seq,
        attention.config.csa_window_size,
        ratio,
        topk.shape[1],
        stride,
        width,
    )
    indices = torch.where(indices >= 0, table[indices.clamp_min(0).long()], indices)
    emit('swa_indices', indices)
    sink = attention.core_attention.native.attn_sink
    out = _Sparse.apply(q, kv, sink, indices, lens, attention.q_head_dim**-0.5)
    emit('core_attn', out)
    if not project:
        return out
    return output_projection(attention, out, positions, rope_cache, emit).reshape_as(hidden)


def output_projection(attention, out, positions, rope_cache, emit=None):
    """wo_a/wo_b over all THD rows at once: both are row-wise, so batching documents
    keeps every output row's bits and encodes each weight once per layer call."""
    emit = emit or (lambda name, value: None)
    weight = attention.linear_o_group_proj
    groups = attention.config.o_groups
    z = _FP8OutputProjection.apply(out, weight, positions, rope_cache, groups)
    emit('wo_a', z)
    result = linear(z.flatten(1), (attention.linear_proj.weight,), fp8=True).bfloat16()
    emit('wo_b', result)
    return result


def build_attention(config, index, context):
    """Reuse the native allocation/spec and replace only the attention forward."""
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELinear,
        TERowParallelLinear,
    )
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
        DSv4HybridSelfAttentionSubmodules,
    )
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.lite.model.deepseek_v41.lite.components.assembly import ModelNorm
    from megatron.lite.model.deepseek_v41.lite.model import CoreAttentionSources

    attention = SWAAttention(
        config,
        DSv4HybridSelfAttentionSubmodules(
            q_layernorm=ModelNorm,
            kv_layernorm=ModelNorm,
            linear_q_down_proj=TELinear,
            linear_q_up_proj=TEColumnParallelLinear,
            linear_kv_proj=TEColumnParallelLinear,
            core_attention=ModuleSpec(CoreAttentionSources, params={'context': context}),
            linear_proj=TERowParallelLinear,
        ),
        index + 1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=context.groups,
        compress_ratio=config.csa_compress_ratios[index],
    )
    attention.release = context.release
    return attention


from megatron.lite.model.deepseek_v41.lite.components.attention import DS41SelfAttention


class SWAAttention(DS41SelfAttention):
    def forward(self, hidden_states, attention_mask=None, *, packed_seq_params=None, **kwargs):
        if self.pg_collection.cp.size() != 1:
            raise NotImplementedError('Deployment SWA CP exchange is not yet validated')
        if packed_seq_params is None or packed_seq_params.qkv_format != 'thd':
            raise ValueError('Deployment SWA requires native packed THD metadata')
        if not hasattr(self, '_deployment_rope'):
            from types import SimpleNamespace

            from vllm.config import VllmConfig, set_current_vllm_config
            from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope

            r = self.release
            hf = SimpleNamespace(
                rope_theta=r.rope_theta,
                compress_rope_theta=r.compress_rope_theta,
                rope_parameters={
                    'rope_type': 'yarn',
                    'factor': r.rotary_scaling_factor,
                    'original_max_position_embeddings': r.original_max_position_embeddings,
                    'beta_fast': r.beta_fast,
                    'beta_slow': r.beta_slow,
                },
            )
            with set_current_vllm_config(VllmConfig()), torch.device(hidden_states.device):
                self._deployment_rope = build_deepseek_v4_rope(
                    hf,
                    head_dim=r.head_dim,
                    rope_head_dim=r.qk_rope_head_dim,
                    max_position_embeddings=r.max_position_embeddings,
                    compress_ratio=self._dsv4_compress_ratio,
                ).cos_sin_cache
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        cached = getattr(packed_seq_params, '_deployment_documents', None)
        if cached is None or cached[0] is not cu or cached[1] != cu._version:
            boundaries = cu.tolist()
            if boundaries[0] != 0 or boundaries[-1] != hidden_states.shape[0]:
                raise ValueError('Deployment SWA requires THD documents tiling all rows')
            documents = _documents(list(zip(boundaries, boundaries[1:])), cu.device)
            cached = (cu, cu._version, documents)
            packed_seq_params._deployment_documents = cached
        if hidden_states.shape[0] == 0:
            return hidden_states * 0, None
        layout = cached[2]

        def sources(x, qr, qa):
            from .compressed import source

            decoded, topk = source(
                self, x, qr, qa, self._deployment_rope, layout.documents, lambda *_: None
            )
            return decoded, topk, self._dsv4_compress_ratio

        out = swa_forward(
            self,
            hidden_states,
            layout.positions,
            self._deployment_rope,
            source=sources if self._dsv4_compress_ratio else None,
            projected=_projections(self, hidden_states.reshape(-1, hidden_states.shape[-1])),
            layout=layout,
        )
        return out, None
