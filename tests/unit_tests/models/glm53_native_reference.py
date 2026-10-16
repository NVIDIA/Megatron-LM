# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent Torch math for production-geometry GLM53 composite tests.

Arithmetic: huggingface/transformers@93c8b7b485963a10800c91f55304db6be211c2bd,
src/transformers/models/glm5_next/modeling_glm5_next.py, SHA256
2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b.
RMSNorm, KDA, Indexer/Attention, MoE, HyperConnection and DecoderLayer are
translated to ordinary Torch operations. No target/kernel imports are allowed.

Routed experts follow HF's single gate_up F.linear and slot-major dispatch;
dense/shared FFNs retain separate gate/up projections. Separate reference leaf
names are retained for exhaustive mappings, concatenating only the expert weight
at execution. Routing specializes the pinned n_group=topk_group=1 profile.

The MTP topology has a separate serving-source provenance, not an official-HF
MTP claim; see NativeMTP's documentation and the source-pinned feature design.

NativeAttention has two core contracts. 'hf_eager' (the default, and the only
one the hybrid tests use) is HF93c8's expanded BF16 K/V with eager attention.
'absorbed' replaces only the attention core with production's absorbed-MLA
contract (absorbed_attention_core, which consumes kv_b_proj's weight as
W_UK/W_UV); the q_a/q_b/kv_a/o_proj projections, norms and the indexer stay HF
arithmetic. The MTP composite tests select it because a pre-registered causal
diagnostic traced the single-stream MTP [none] gap to the attention core's
rounding contract (FP32 scores plus the W_UK/W_UV reassociation; neither alone
closes it); the cudnn and real HybridModel MTP cases were not covered. A local
attention-layer test keeps 'hf_eager' under direct comparison with production.

Two comparison contracts serve the composite mHC tests.

ClampReplay replays the target's SwiGLU clamp decisions inside the reference
FFNs. The reference keeps its own FC1 values and arithmetic; only which
elements clamp is decided from the target's observed FC1 output, with the
reference threshold. Rounding-level input differences otherwise move a few
coverage-channel values across the threshold, and each moved decision switches
a gradient between zero and one. Every replayed decision that differs from the
reference's own must be a near tie within its column's rounding-level spread
(CLAMP_TIE_FLOOR, CLAMP_TIE_SIGMAS).

assert_mhc_decomposition replaces the shape-[1] mHC scale gradient links. Each
scale gradient is the sum of per-token contributions (dL/dh * proj) over its
group, and that sum cancels heavily, so a 0.999 bar on the scalar is a few
percent tolerance on a cancelled sum, which equally valid reference executions
fail against each other. The contribution tensors are compared at the standard
bars instead, and each side's scalar must equal the sum of its own
contributions.
"""

import json
import resource
from contextvars import ContextVar
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

HIDDEN = 4096
STREAMS = 4
HEADS = 64
KDA_DIM = 128
KDA_WIDTH = HEADS * KDA_DIM
RANK = 128
EPS = 1e-5
EXPERTS = 288
EXPERT_WIDTH = 2048
VOCAB = 154880
CLAMP = 10.0
# A replayed clamp decision may span two BF16 ulps at the threshold, or six
# times the call's relative FC1 difference scaled to its output column's RMS.
CLAMP_TIE_FLOOR = 0.125
CLAMP_TIE_SIGMAS = 6.0
MHC_GROUPS = (('pre', slice(0, 4)), ('post', slice(4, 8)), ('res', slice(8, 24)))
MHC_RECONSTRUCTION_RTOL = 1e-5
_LEDGER = ContextVar('glm53_composite_similarity_ledger', default=None)


class ComparisonLedger:
    """One bounded stdout summary and full named metrics in the JUnit property."""

    def __init__(self, name, rank):
        self.name, self.rank = name, rank
        self.comparisons, self.details = [], {}
        self.finite_operands = 0
        self.zero_operands = 0

    def __enter__(self):
        self.token = _LEDGER.set(self)
        # Initialize the current allocator before observing its peak. This is
        # test instrumentation, not an uninitialized-ordinal memory probe.
        warmup = torch.empty(1, device='cuda')
        del warmup
        torch.cuda.reset_peak_memory_stats()
        return self

    def publish(self, record_property):
        report = {
            'schema': 'glm53-composite-native-v1',
            'test': self.name,
            'rank': self.rank,
            'scope': 'BF16 independent TP1/PP1/EP1/CP1 replica; synthetic weights',
            'finite_operands': self.finite_operands,
            'zero_operands': self.zero_operands,
            'comparisons': self.comparisons,
            'details': self.details,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'host_process_max_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        record_property('glm53_composite_receipt', json.dumps(report, sort_keys=True))
        compact = {key: value for key, value in report.items() if key != 'comparisons'}
        compact['compared_tensor_count'] = len(self.comparisons)
        compact['worst_five'] = sorted(
            self.comparisons, key=lambda value: min(value['cosine'], value['tensor_similarity'])
        )[:5]
        print('GLM53_COMPOSITE_RECEIPT ' + json.dumps(compact, sort_keys=True))

    def __exit__(self, *_):
        _LEDGER.reset(self.token)


def record_details(key, value):
    ledger = _LEDGER.get()
    if ledger is not None:
        ledger.details[key] = value


class NativeLinear(nn.Module):
    """Uninitialized independent storage, populated by an exhaustive weight map."""

    def __init__(self, inputs, outputs, device, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outputs, inputs, device=device, dtype=dtype))

    def forward(self, hidden):
        return F.linear(hidden, self.weight)


class NativeRMSNorm(nn.Module):
    """HF93c8: cast normalized activation BEFORE multiplying the BF16 weight."""

    def __init__(self, width, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, device=device, dtype=torch.bfloat16))

    def forward(self, hidden):
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + EPS)
        return self.weight * value.to(hidden.dtype)


class NativeForgetGate(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.f_a_proj = NativeLinear(HIDDEN, RANK, device)
        self.f_b_proj = NativeLinear(RANK, KDA_WIDTH, device)
        self.dt_bias = nn.Parameter(torch.empty(KDA_WIDTH, device=device, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.empty(HEADS, device=device, dtype=torch.float32))

    def forward(self, hidden):
        raw = self.f_b_proj(self.f_a_proj(hidden))
        gate = (raw.float() + self.dt_bias).unflatten(-1, (HEADS, KDA_DIM))
        return -5.0 * torch.sigmoid(self.A_log.exp()[None, None, :, None] * gate)


class NativeGatedNorm(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(KDA_DIM, device=device, dtype=torch.bfloat16))

    def forward(self, hidden, gate):
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + EPS)
        return (value * self.weight.float() * gate.float().sigmoid()).to(hidden.dtype)


class NativeKDA(nn.Module):
    """Source recurrent delta rule; never invokes FLA or a target convolution."""

    def __init__(self, device):
        super().__init__()
        self.q_proj = NativeLinear(HIDDEN, KDA_WIDTH, device)
        self.k_proj = NativeLinear(HIDDEN, KDA_WIDTH, device)
        self.v_proj = NativeLinear(HIDDEN, KDA_WIDTH, device)
        self.conv1d = nn.Module()
        self.conv1d.weight = nn.Parameter(
            torch.empty(3 * KDA_WIDTH, 1, 4, device=device, dtype=torch.float32)
        )
        self.forget_gate = NativeForgetGate(device)
        self.b_proj = NativeLinear(HIDDEN, HEADS, device)
        self.g_a_proj = NativeLinear(HIDDEN, RANK, device)
        self.g_b_proj = NativeLinear(RANK, KDA_WIDTH, device)
        self.o_norm = NativeGatedNorm(device)
        self.o_proj = NativeLinear(KDA_WIDTH, HIDDEN, device)

    def forward(self, hidden):
        x = hidden.transpose(0, 1).contiguous()
        batch, sequence = x.shape[:2]
        qkv = torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), -1)
        qkv = F.conv1d(
            qkv.transpose(1, 2).float(), self.conv1d.weight, padding=3, groups=3 * KDA_WIDTH
        )[..., :sequence]
        qkv = F.silu(qkv).to(x.dtype).transpose(1, 2)
        query, key, value = (
            tensor.reshape(batch, sequence, HEADS, KDA_DIM) for tensor in qkv.split(KDA_WIDTH, -1)
        )
        query, key, value = (tensor.float() for tensor in (query, key, value))
        query = query / torch.sqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        query = query * (KDA_DIM**-0.5)
        key = key / torch.sqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        decay = self.forget_gate(x).exp()
        # The released beta sigmoid runs in the projection dtype, before float.
        beta = self.b_proj(x).sigmoid().float()
        state = torch.zeros(batch, HEADS, KDA_DIM, KDA_DIM, device=x.device, dtype=torch.float32)
        outputs = []
        for token in range(sequence):
            q, k, v = query[:, token], key[:, token], value[:, token]
            state = state * decay[:, token].unsqueeze(-1)
            memory = (state * k.unsqueeze(-1)).sum(-2)
            delta = (v - memory) * beta[:, token].unsqueeze(-1)
            state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((state * q.unsqueeze(-1)).sum(-2))
        output = torch.stack(outputs, 1).to(x.dtype)
        gate = self.g_b_proj(self.g_a_proj(x)).unflatten(-1, (HEADS, KDA_DIM))
        output = self.o_norm(output, gate).flatten(-2)
        return self.o_proj(output).transpose(0, 1).contiguous()


class NativeMHC(nn.Module):
    """HF93c8 normalize-before-projection, twenty Sinkhorn iterations."""

    def __init__(self, device):
        super().__init__()
        self.fn = nn.Parameter(
            torch.empty(24, STREAMS * HIDDEN, device=device, dtype=torch.float32)
        )
        self.base = nn.Parameter(torch.empty(24, device=device, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, device=device, dtype=torch.float32))

    def mappings(self, hidden):
        x = hidden.float()
        normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
        mix = F.linear(normalized, self.fn)
        # Kept for assert_mhc_decomposition: scale gradients are sums of
        # (dL/dmix / scale) * mix over each group.
        if mix.requires_grad:
            mix.retain_grad()
        self.last_mix = mix
        pre, post, comb = mix.split((4, 4, 16), -1)
        pre_bias, post_bias, comb_bias = self.base.split((4, 4, 16))
        pre = (pre * self.scale[0] + pre_bias).sigmoid() + 1e-6
        post = 2 * (post * self.scale[1] + post_bias).sigmoid()
        comb = (comb * self.scale[2] + comb_bias).unflatten(-1, (4, 4))
        comb = comb.softmax(-1) + 1e-6
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
        for _ in range(19):
            comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
            comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
        streams = hidden.unflatten(-1, (STREAMS, HIDDEN))
        aggregated = (pre.unsqueeze(-1) * streams).sum(-2).to(hidden.dtype)
        return aggregated, post.to(hidden.dtype), comb.to(hidden.dtype)

    @staticmethod
    def residual(hidden, branch, post, comb):
        streams = hidden.unflatten(-1, (STREAMS, HIDDEN))
        mixed = torch.matmul(comb.transpose(-1, -2), streams)
        return (post.unsqueeze(-1) * branch.unsqueeze(-2) + mixed).flatten(-2)


class NativeFFN(nn.Module):
    def __init__(self, width, device, *, combined_gate_up=False):
        super().__init__()
        self.combined_gate_up = combined_gate_up
        self.gate_proj = NativeLinear(HIDDEN, width, device)
        self.up_proj = NativeLinear(HIDDEN, width, device)
        self.down_proj = NativeLinear(width, HIDDEN, device)

    def pre_activation(self, hidden):
        if self.combined_gate_up:
            # HF93c8 Glm5NextTextExperts:132; keep the existing two leaf names.
            weight = torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0)
            return F.linear(hidden, weight).chunk(2, dim=-1)
        # HF93c8 Glm5NextTextMLP:99-100 (dense and shared experts).
        return self.gate_proj(hidden), self.up_proj(hidden)

    def forward(self, hidden, rows=None):
        gate, up = self.pre_activation(hidden)
        self.clamp_counts = torch.stack(
            (
                (gate > 10).sum(),
                (up > 10).sum(),
                (up < -10).sum(),
                ((gate <= 10) & (up >= -10) & (up <= 10)).sum(),
            )
        )
        # ClampReplay binds clamp_replay; rows are the routed token ids.
        replay = getattr(self, 'clamp_replay', None)
        if replay is None:
            gate = gate.clamp(max=CLAMP)
            up = up.clamp(min=-CLAMP, max=CLAMP)
        else:
            gate, up = replay(gate, up, rows)
        return self.down_proj(F.silu(gate) * up)


class NativeIndexer(nn.Module):
    """Frozen source KPool selection; topk2048 is never replaced by a toy budget."""

    def __init__(self, device):
        super().__init__()
        self.wq_b = NativeLinear(1536, 32 * 128, device)
        self.wk = NativeLinear(HIDDEN, 128, device)
        self.k_norm = nn.LayerNorm(128, eps=1e-6, device=device, dtype=torch.bfloat16)
        self.weights_proj = NativeLinear(HIDDEN, 32, device)
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(128, HIDDEN, device=device, dtype=torch.bfloat16)
        )
        self.index_kpool_compress_ape = nn.Parameter(
            torch.empty(4, 128, device=device, dtype=torch.float32)
        )
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, hidden, qr):
        batch, sequence = hidden.shape[:2]
        query = self.wq_b(qr).reshape(batch, sequence, 32, 128)
        keys = self.k_norm(self.wk(hidden))
        gates = F.linear(hidden, self.index_kpool_compress_gate)
        pools = sequence // 4
        key_groups = keys[:, : pools * 4].reshape(batch, pools, 4, 128)
        gate_groups = gates[:, : pools * 4].reshape_as(key_groups)
        probabilities = (gate_groups.float() + self.index_kpool_compress_ape[None, None]).softmax(2)
        pooled = (probabilities.to(keys.dtype) * key_groups).sum(2)
        scores = torch.matmul(query.float(), pooled.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores * (128**-0.5))
        weights = self.weights_proj(hidden).float() * (32**-0.5)
        scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
        pool_tokens = torch.arange(pools * 4, device=hidden.device).reshape(pools, 4)
        valid = pool_tokens[:, -1][None] <= torch.arange(sequence, device=hidden.device)[:, None]
        selected = (
            scores.masked_fill(~valid[None], torch.finfo(scores.dtype).min)
            .topk(min(2048 // 4, pools), dim=-1)
            .indices
        )
        selected_valid = valid[None].expand(batch, -1, -1).gather(-1, selected)
        tokens = pool_tokens[selected].masked_fill(~selected_valid[..., None], -1).flatten(-2)
        result = torch.full((batch, sequence, 2051), -1, device=hidden.device, dtype=torch.int32)
        result[..., : tokens.size(-1)] = tokens.to(torch.int32)
        for token in range(sequence):
            tail = (token + 1) % 4
            result[:, token, 2048 : 2048 + tail] = torch.arange(
                token + 1 - tail, token + 1, device=hidden.device, dtype=torch.int32
            )
        return result


ATTENTION_CONTRACTS = ('hf_eager', 'absorbed')


def absorbed_index_mask(selected, sequence):
    """Production's FP32 0/-inf top-k index mask plus plain causal validity.

    dsa_masking.scatter_topk_into_index_mask (amax scatter, -1 selections stay
    -inf) followed by the default causal row mask, as the unfused DSA core uses.
    """
    batch = selected.size(0)
    device = selected.device
    index_mask = torch.full((batch, sequence, sequence), float('-inf'), device=device)
    indices = selected.to(dtype=torch.int64)
    source = torch.where(
        indices >= 0,
        torch.zeros((), dtype=index_mask.dtype, device=device),
        torch.full((), float('-inf'), dtype=index_mask.dtype, device=device),
    )
    index_mask.scatter_reduce_(-1, indices.clamp_min(0), source, reduce='amax', include_self=True)
    positions = torch.arange(sequence, device=device)
    causal = positions[None, :] <= positions[:, None]
    return index_mask.masked_fill(~causal.unsqueeze(0), float('-inf'))


def absorbed_masked_softmax(logits, valid, eps=1e-10):
    """Production's composite FP32 masked softmax (dsa_masking.masked_softmax)."""
    masked = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    row_has_valid = valid.any(dim=-1, keepdim=True)
    row_max = masked.max(dim=-1, keepdim=True).values
    row_max = torch.where(row_has_valid, row_max, torch.zeros_like(row_max))
    probabilities = torch.exp(masked - row_max)
    probabilities = probabilities.masked_fill(~valid, 0.0)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)
    return probabilities.masked_fill(~valid, 0.0)


def absorbed_attention_core(q_flat, latent, kv_b_weight, selected):
    """Production's absorbed-MLA attention core ('none' DSA backend) in pure Torch.

    Takes the B-major q_b_proj output, the normalized 512-wide latent, the
    kv_b_proj weight and the indexer selection; returns the B-major input of
    o_proj. Order and dtypes follow the q absorption in absorbed_mla.py, the
    unfused absorbed DSA core and the V up-projection einsum in dsa.py:
    bf16(q x W_UK) with the strided per-head W_UK view, FP32 QK logits against
    the latent times 256**-0.5 (production's softmax_scale, exactly 1/16), the
    FP32 index mask, the composite FP32 masked softmax, bf16(P x latent), then
    bf16(... x W_UV) with a contiguous W_UV. This is algebraically the HF eager
    attention; the BF16 rounding points, the FP32 score dtype and the -inf
    index mask with the composite softmax differ, all equivalent on valid keys.

    Why the MTP composite tests use it: a pre-registered causal diagnostic
    (eight workers agree, all eleven validity gates passed) examined the
    single-stream MTP [none] case. Every stage up to q, the latent and kv is
    bit-identical to production and the first divergence is the attention core.
    With this contract the reference reproduces production's MTP FFN input bit
    for bit and passes every 0.999 bar (minimum gradient cosine 0.99996). With
    the HF eager core 13.6% of that input differs, about ten SwiGLU clamp
    crossings move and three gradient links fall below the bar (expert82 up
    0.99805, minimum 0.9911). The score dtype and the reassociation are both
    needed; neither alone closes the gap. The cudnn (FlashMLA) backend is not
    bitwise to this contract; it and the real HybridModel MTP case were not
    covered by the diagnostic.
    """
    batch, sequence = q_flat.shape[:2]
    # S-major [s, b, heads, 256] query and [s, b, 512] latent, as production holds them.
    q = q_flat.transpose(0, 1).contiguous().view(sequence, batch, HEADS, 256)
    c = latent.transpose(0, 1).contiguous()
    # Per-head [W_UK; W_UV] rows of kv_b_proj; W_UK stays a strided view.
    weight = kv_b_weight.view(HEADS, 2 * 256, 512)
    w_k, w_v = weight[:, :256, :], weight[:, 256:, :]
    q_abs = torch.einsum('...nd,ndk->...nk', q, w_k).contiguous()
    key = torch.unsqueeze(c, -2).clone()
    up_v = w_v.to(dtype=q_abs.dtype).contiguous()
    q_perm = q_abs.permute(1, 2, 0, 3)
    k_perm = key.permute(1, 2, 3, 0)
    scores = torch.matmul(q_perm.float(), k_perm.float()) * (256**-0.5)
    index_mask = absorbed_index_mask(selected, sequence)
    scores = scores + index_mask.unsqueeze(1)
    valid = torch.isfinite(index_mask)
    probabilities = absorbed_masked_softmax(
        scores.float(), valid.unsqueeze(1).expand(batch, HEADS, sequence, sequence)
    )
    value = key[..., :512].permute(1, 2, 0, 3)
    o_lat = torch.matmul(probabilities.to(value.dtype), value).permute(2, 0, 1, 3).contiguous()
    core_sb = torch.einsum('sbhc,hdc->sbhd', o_lat, up_v).contiguous()
    core_sb = core_sb.view(core_sb.size(0), core_sb.size(1), -1)
    return core_sb.transpose(0, 1).contiguous()


class NativeAttention(nn.Module):
    """HF93c8 MLA projections and frozen indexer with a selectable attention core.

    core_contract='hf_eager' (default): expanded HF K/V and eager attention,
    independent of matrix absorption. core_contract='absorbed': production's
    absorbed-MLA core (absorbed_attention_core, which documents why the MTP
    composite tests use it); everything outside the core is unchanged.
    """

    def __init__(self, device, *, core_contract='hf_eager'):
        super().__init__()
        if core_contract not in ATTENTION_CONTRACTS:
            raise ValueError(
                f'core_contract must be one of {ATTENTION_CONTRACTS}, got {core_contract!r}'
            )
        self.core_contract = core_contract
        self.q_a_proj = NativeLinear(HIDDEN, 1536, device)
        self.q_a_layernorm = NativeRMSNorm(1536, device)
        self.q_b_proj = NativeLinear(1536, HEADS * 256, device)
        self.kv_a_proj_with_mqa = NativeLinear(HIDDEN, 512, device)
        self.kv_a_layernorm = NativeRMSNorm(512, device)
        self.kv_b_proj = NativeLinear(512, HEADS * 2 * 256, device)
        self.o_proj = NativeLinear(HEADS * 256, HIDDEN, device)
        self.indexer = NativeIndexer(device)

    def absorbed_forward(self, hidden):
        """HF q_a/q_b/kv_a/o_proj projections, norms and indexer; kv_b_proj absorbed in the core."""
        x = hidden.transpose(0, 1)
        qr = self.q_a_layernorm(self.q_a_proj(x))
        q_flat = self.q_b_proj(qr)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        selected = self.indexer(x, qr)
        self.last_indices = selected.detach()
        output = absorbed_attention_core(q_flat, latent, self.kv_b_proj.weight, selected)
        return self.o_proj(output).transpose(0, 1).contiguous()

    def forward(self, hidden):
        if self.core_contract == 'absorbed':
            return self.absorbed_forward(hidden)
        x = hidden.transpose(0, 1)
        batch, sequence = x.shape[:2]
        qr = self.q_a_layernorm(self.q_a_proj(x))
        query = self.q_b_proj(qr).reshape(batch, sequence, HEADS, 256).transpose(1, 2)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        kv = self.kv_b_proj(latent).reshape(batch, sequence, HEADS, 512).transpose(1, 2)
        keys, values = kv.split(256, -1)
        selected = self.indexer(x, qr)
        self.last_indices = selected.detach()
        visible = torch.zeros(batch, sequence, sequence, device=x.device, dtype=torch.int32)
        visible.scatter_add_(-1, selected.clamp_min(0).long(), (selected >= 0).to(torch.int32))
        mask = torch.where(visible[:, None].bool(), 0.0, torch.finfo(query.dtype).min).to(
            query.dtype
        )
        scores = torch.matmul(query, keys.transpose(-1, -2)) * (256**-0.5)
        probabilities = (scores + mask).softmax(-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(probabilities, values).transpose(1, 2).contiguous().flatten(-2)
        return self.o_proj(output).transpose(0, 1).contiguous()


class NativeMoE(nn.Module):
    """Full288 expert inventory with source routing and ordinary per-expert GEMMs."""

    def __init__(self, device):
        super().__init__()
        self.gate = NativeLinear(HIDDEN, EXPERTS, device)
        self.register_buffer(
            'e_score_correction_bias', torch.empty(EXPERTS, device=device, dtype=torch.float32)
        )
        self.experts = nn.ModuleList(
            NativeFFN(EXPERT_WIDTH, device, combined_gate_up=True) for _ in range(EXPERTS)
        )
        self.shared_experts = NativeFFN(EXPERT_WIDTH, device)

    def routing(self, hidden):
        x = hidden.reshape(-1, HIDDEN)
        scores = F.linear(x.float(), self.gate.weight.float()).sigmoid()
        indices = (scores + self.e_score_correction_bias).topk(8, dim=-1, sorted=False).indices
        probabilities = scores.gather(-1, indices)
        probabilities = probabilities / (probabilities.sum(-1, keepdim=True) + 1e-20) * 2.5
        return indices, probabilities

    def forward(self, hidden):
        shape = hidden.shape
        x = hidden.reshape(-1, HIDDEN)
        indices, probabilities = self.routing(x)
        routing = torch.zeros(x.size(0), EXPERTS, device=x.device, dtype=torch.bool)
        routing.scatter_(1, indices, True)
        self.last_routing = routing.detach()
        self.last_probabilities = (
            torch.zeros_like(routing, dtype=probabilities.dtype)
            .scatter(1, indices, probabilities)
            .detach()
        )
        counts = routing.sum(0)
        assert (counts > 0).all(), f'fixture must route all288 experts: {counts.cpu().tolist()}'
        # HF93c8 Experts:124-131 traverses top-k slots before token rows.
        with torch.no_grad():
            expert_mask = F.one_hot(indices, num_classes=EXPERTS).permute(2, 1, 0)
        output = torch.zeros_like(x)
        for expert, module in enumerate(self.experts):
            slot, token = torch.where(expert_mask[expert])
            branch = module(x[token], rows=token) * probabilities[token, slot, None]
            output.index_add_(0, token, branch.to(x.dtype))
        return output.reshape(shape) + self.shared_experts(hidden)


class NativeHybridBlock(nn.Module):
    """One logical attention/FFN block, including both mHC sites and final norm."""

    def __init__(self, device, *, expand=False, sparse=False):
        super().__init__()
        self.expand = expand
        self.attn_hc = NativeMHC(device)
        self.ffn_hc = NativeMHC(device)
        self.input_layernorm = NativeRMSNorm(HIDDEN, device)
        self.post_attention_layernorm = NativeRMSNorm(HIDDEN, device)
        self.self_attn = NativeAttention(device) if sparse else NativeKDA(device)
        self.mlp = NativeMoE(device) if sparse else NativeFFN(12288, device)
        self.final_norm = NativeRMSNorm(HIDDEN, device)

    def attention_stage(self, hidden):
        if self.expand:
            hidden = hidden.unsqueeze(-2).expand(*hidden.shape[:-1], STREAMS, HIDDEN).flatten(-2)
        collapsed, post, comb = self.attn_hc.mappings(hidden)
        attention = self.self_attn(self.input_layernorm(collapsed))
        hidden = self.attn_hc.residual(hidden, attention, post, comb)
        return hidden

    def ffn_input(self, hidden):
        hidden = self.attention_stage(hidden)
        collapsed, _, _ = self.ffn_hc.mappings(hidden)
        return self.post_attention_layernorm(collapsed)

    def forward(self, hidden):
        hidden = self.attention_stage(hidden)
        collapsed, post, comb = self.ffn_hc.mappings(hidden)
        ffn = self.mlp(self.post_attention_layernorm(collapsed))
        hidden = self.ffn_hc.residual(hidden, ffn, post, comb)
        self.last_streams = hidden
        return self.final_norm(hidden.unflatten(-1, (STREAMS, HIDDEN)).mean(-2))


@torch.no_grad()
def balanced_router_weight(ffn_input):
    """A synthetic weight fixture, not a target routing override or oracle.

    Fit ordinary router weights to margin-separated desired logits using the
    independent prefix activations. Every group of eight experts gets real
    tokens. Both implementations then execute their own unmodified routing.
    """
    x = ffn_input.reshape(-1, HIDDEN).double()
    assert x.size(0) >= EXPERTS // 8
    indices = (torch.arange(x.size(0), device=x.device) % (EXPERTS // 8))[:, None] * 8
    indices = indices + torch.arange(8, device=x.device)[None]
    logits = torch.full((x.size(0), EXPERTS), -1.0, device=x.device, dtype=torch.float64)
    desired = torch.linspace(0.4, 1.1, 8, device=x.device, dtype=torch.float64).expand_as(indices)
    logits.scatter_(1, indices, desired)
    gram = x @ x.T
    gram.diagonal().add_(1e-6)
    weight = (x.T @ torch.linalg.solve(gram, logits)).T.to(torch.bfloat16).contiguous()
    observed = F.linear(x.float(), weight.float())
    assert (observed - logits.float()).abs().max().item() < 0.05, 'router fixture lost its margin'
    return weight


class HostParameters:
    """Own CPU gold storage after target backward; never executes target math."""

    def __init__(self, module):
        self.parameters = {}
        persistent, runtime = buffer_inventory(module)
        self.persistent_buffers = {
            name: value.detach().to('cpu', copy=True) for name, value in persistent.items()
        }
        self.runtime_buffers = {
            name: value.detach().to('cpu', copy=True) for name, value in runtime.items()
        }
        for name, parameter in module.named_parameters():
            value = nn.Parameter(
                parameter.detach().to('cpu', copy=True), requires_grad=parameter.requires_grad
            )
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                value.grad = parameter.grad.detach().to('cpu', copy=True)
            else:
                assert parameter.grad is None, name
            self.parameters[name] = value

    def named_parameters(self):
        return self.parameters.items()


def buffer_inventory(module):
    """Enumerate serialized buffers and nonpersistent runtime tensors separately."""
    if isinstance(module, HostParameters):
        return module.persistent_buffers, module.runtime_buffers
    persistent, runtime = {}, {}
    inactive = []
    for prefix, child in module.named_modules():
        for name, value in child._buffers.items():
            full_name = f'{prefix}.{name}' if prefix else name
            if value is None:
                # PyTorch does not serialize None buffers; retain their names as
                # inactive metadata, not silently as a numerical checkpoint key.
                inactive.append(full_name)
            elif name in child._non_persistent_buffers_set:
                runtime[full_name] = value
            else:
                persistent[full_name] = value
    record_details('inactive_registered_buffers', inactive)
    return persistent, runtime


class BufferContract:
    """Exhaustive semantic-buffer map plus an explicit runtime-counter allowlist."""

    def __init__(self, actual, native, links=(), runtime_names=()):
        self.links = tuple(links)
        self.runtime_names = set(runtime_names)
        actual_persistent, actual_runtime = buffer_inventory(actual)
        native_persistent, native_runtime = buffer_inventory(native)
        assert set(actual_persistent) == {left for left, _ in self.links}
        assert set(native_persistent) == {right for _, right in self.links}
        assert set(actual_runtime) == self.runtime_names, sorted(actual_runtime)
        assert not native_runtime, sorted(native_runtime)
        self.values = {}
        self.runtime_before = {}
        for left, right in self.links:
            a, n = actual_persistent[left], native_persistent[right]
            assert a.shape == n.shape == (EXPERTS,), (left, right)
            assert a.dtype == n.dtype == torch.float32, (left, right)
            assert not a.requires_grad and not n.requires_grad
            n.copy_(a)
            self.values[left] = a.detach().to('cpu', copy=True)
        for name, value in actual_runtime.items():
            assert name.endswith('.router.local_tokens_per_expert'), name
            assert value.shape == (EXPERTS,) and not value.requires_grad, name
            self.runtime_before[name] = value.detach().to('cpu', copy=True)
        record_details(
            'buffer_inventory',
            {
                'persistent_map': [list(link) for link in self.links],
                'runtime_allowlist': sorted(self.runtime_names),
                'note': 'quantile/global-aux/hash buffers are forbidden by this declared profile',
            },
        )

    def assert_actual_unchanged(self, actual, router_records=()):
        persistent, runtime = buffer_inventory(actual)
        assert set(persistent) == set(self.values)
        assert set(runtime) == self.runtime_names
        for name, before in self.values.items():
            assert persistent[name].dtype == torch.float32, name
            torch.testing.assert_close(persistent[name].detach().cpu(), before, rtol=0, atol=0)
        if self.runtime_names:
            assert len(router_records) == 1
            counts = router_records[0][1].sum(0)
            for name, value in runtime.items():
                delta = value.detach().cpu() - self.runtime_before[name]
                torch.testing.assert_close(delta.double(), counts.double(), rtol=0, atol=0)
            record_details(
                'runtime_counter_values',
                {
                    name: {'dtype': str(value.dtype), 'counts': value.detach().cpu().tolist()}
                    for name, value in runtime.items()
                },
            )
        record_details('persistent_buffers_unchanged', True)

    def copy_native(self, actual, native):
        actual_persistent, actual_runtime = buffer_inventory(actual)
        native_persistent, native_runtime = buffer_inventory(native)
        assert set(actual_persistent) == set(self.values)
        assert set(actual_runtime) == self.runtime_names and not native_runtime
        assert set(native_persistent) == {right for _, right in self.links}
        for left, right in self.links:
            assert actual_persistent[left].dtype == native_persistent[right].dtype == torch.float32
            torch.testing.assert_close(
                actual_persistent[left].cpu(), self.values[left], rtol=0, atol=0
            )
            native_persistent[right].copy_(actual_persistent[left])

    def assert_native_unchanged(self, native):
        persistent, runtime = buffer_inventory(native)
        assert not runtime and set(persistent) == {right for _, right in self.links}
        for left, right in self.links:
            assert persistent[right].dtype == torch.float32
            torch.testing.assert_close(
                persistent[right].detach().cpu(), self.values[left], rtol=0, atol=0
            )


class NativeEmbedding(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(VOCAB, HIDDEN, device=device, dtype=torch.bfloat16))

    def forward(self, tokens):
        return F.embedding(tokens, self.weight).transpose(0, 1).contiguous()


def successor_tokens(tokens):
    """CP1 training successor: copy, zero-fill the end, never circularly wrap."""
    successor = torch.zeros_like(tokens)
    successor[:, :-1] = tokens[:, 1:]
    return successor


class NativeMTP(nn.Module):
    """Source-backed single-stream contract, NOT complete official-HF MTP parity.

    Structure: vllm-project/vllm@10d2c6083d3b7508f2409a127a1d03dd489e427c,
    vllm/models/glm5next/common/mtp.py51-115 (blob2bb8b10d074676fad5ee06da087913302b031649),
    common/model.py368-442,532-549 and nvidia/ops/fused_eh_norm.py39-49.
    Those independently confirm embedding-first, DE, no inner mHC and one final
    norm. Component arithmetic deliberately follows HF93c8, including its
    cast-before-weight RMSNorm, NOT vLLM's fused EH rounding or speculative
    position-zero handling. The successor training definition is above.
    attention_contract selects only the attention core (NativeAttention).
    """

    def __init__(self, device, *, attention_contract='hf_eager'):
        super().__init__()
        self.enorm = NativeRMSNorm(HIDDEN, device)
        self.hnorm = NativeRMSNorm(HIDDEN, device)
        self.eh_proj = NativeLinear(2 * HIDDEN, HIDDEN, device)
        self.input_layernorm = NativeRMSNorm(HIDDEN, device)
        self.self_attn = NativeAttention(device, core_contract=attention_contract)
        self.post_attention_layernorm = NativeRMSNorm(HIDDEN, device)
        self.mlp = NativeMoE(device)
        self.final_norm = NativeRMSNorm(HIDDEN, device)

    def attention_stage(self, hidden, embedding):
        hidden = self.eh_proj(torch.cat((self.enorm(embedding), self.hnorm(hidden)), -1))
        return hidden + self.self_attn(self.input_layernorm(hidden))

    def ffn_input(self, hidden, embedding):
        return self.post_attention_layernorm(self.attention_stage(hidden, embedding))

    def forward(self, hidden, embedding):
        hidden = self.attention_stage(hidden, embedding)
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return self.final_norm(hidden)


class NativeMTPHarness(nn.Module):
    """Independent shared embedding/head and one MTP depth, no base decoder."""

    def __init__(self, device, *, attention_contract='hf_eager'):
        super().__init__()
        self.embedding = NativeEmbedding(device)
        self.output_layer = NativeLinear(HIDDEN, VOCAB, device)
        self.mtp = NativeMTP(device, attention_contract=attention_contract)

    def ffn_input(self, hidden, tokens):
        return self.mtp.ffn_input(hidden, self.embedding(successor_tokens(tokens)))

    def forward(self, hidden, tokens):
        predicted = self.mtp(hidden, self.embedding(successor_tokens(tokens)))
        return (
            torch.cat((hidden, predicted), 0),
            self.output_layer(hidden),
            self.output_layer(predicted),
        )


class NativeCompositeModel(NativeMTPHarness):
    """One real-width early KDA+dense block followed by single-depth DE MTP.

    attention_contract reaches only the MTP attention; the decoder block is KDA.
    """

    def __init__(self, device, *, wire=False, attention_contract='hf_eager'):
        super().__init__(device, attention_contract=attention_contract)
        self.decoder = NativeHybridBlock(device, expand=not wire)

    def ffn_input(self, tokens, wire=None):
        hidden = self.decoder(self.embedding(tokens) if wire is None else wire)
        return self.mtp.ffn_input(hidden, self.embedding(successor_tokens(tokens)))

    def forward(self, tokens, wire=None):
        self.decoder_input = self.embedding(tokens) if wire is None else wire
        self.decoder_input.retain_grad()
        hidden = self.decoder(self.decoder_input)
        return super().forward(hidden, tokens)


@dataclass(frozen=True)
class ParameterLink:
    """An explicit contiguous slice on either side of a packed parameter map."""

    native: str
    actual: str
    native_slice: slice | None = None
    actual_slice: slice | None = None


def linked_view(tensor, selection):
    return tensor if selection is None else tensor[selection]


def assert_complete_mapping(actual, native, links):
    """Prove whole-parameter coverage without a billion-element coverage mask."""
    actual_parameters, native_parameters = dict(actual.named_parameters()), dict(
        native.named_parameters()
    )
    assert set(actual_parameters) == {link.actual for link in links}
    assert set(native_parameters) == {link.native for link in links}
    record_details(
        'parameter_inventory',
        {
            'actual_named': len(actual_parameters),
            'native_named': len(native_parameters),
            'mapped_slices': len(links),
            'actual_parameter_elements': sum(
                parameter.numel() for parameter in actual_parameters.values()
            ),
            'frozen_actual_names': [
                name for name, parameter in actual_parameters.items() if not parameter.requires_grad
            ],
        },
    )
    for parameters, name_key, slice_key in (
        (actual_parameters, 'actual', 'actual_slice'),
        (native_parameters, 'native', 'native_slice'),
    ):
        for name, parameter in parameters.items():
            intervals = []
            for link in links:
                if getattr(link, name_key) == name:
                    view = linked_view(parameter, getattr(link, slice_key))
                    assert view.is_contiguous(), name
                    start = view.storage_offset() - parameter.storage_offset()
                    intervals.append((start, start + view.numel()))
            cursor = 0
            for start, end in sorted(intervals):
                assert start == cursor and end > start, (name, intervals)
                cursor = end
            assert cursor == parameter.numel(), (name, intervals)
    for link in links:
        a = linked_view(actual_parameters[link.actual], link.actual_slice)
        n = linked_view(native_parameters[link.native], link.native_slice)
        assert a.shape == n.shape, link
        assert a.dtype == n.dtype, link
        assert (
            actual_parameters[link.actual].requires_grad
            == native_parameters[link.native].requires_grad
        ), link
    return actual_parameters, native_parameters


@torch.no_grad()
def copy_parameters(actual, native, links):
    actual_parameters, native_parameters = assert_complete_mapping(actual, native, links)
    for link in links:
        linked_view(native_parameters[link.native], link.native_slice).copy_(
            linked_view(actual_parameters[link.actual], link.actual_slice)
        )


def assert_similarity(actual, expected, label, *, exact_zero=False):
    """Bound temporary storage; both skill metrics are required above0.999."""
    assert actual is not None and expected is not None, label
    assert actual.shape == expected.shape and actual.numel() > 0, label
    assert actual.dtype == expected.dtype, label
    a, b = actual.detach().reshape(-1), expected.detach().reshape(-1)
    dot = torch.zeros((), dtype=torch.float64, device=expected.device)
    norm_a, norm_b = dot.clone(), dot.clone()
    for start in range(0, a.numel(), 262144):
        x = a[start : start + 262144].to(device=b.device, dtype=torch.float64)
        y = b[start : start + 262144].double()
        assert torch.isfinite(x).all() and torch.isfinite(y).all(), label
        dot += (x * y).sum()
        norm_a += x.square().sum()
        norm_b += y.square().sum()
    if exact_zero:
        assert norm_a.item() == norm_b.item() == 0.0, label
        ledger = _LEDGER.get()
        if ledger is not None:
            ledger.zero_operands += 2
        return
    assert norm_a.item() > 0 and norm_b.item() > 0, f'{label}: unexpected all-zero operand'
    cosine = (dot / (norm_a * norm_b).sqrt()).item()
    similarity = (2 * dot / (norm_a + norm_b)).item()
    ledger = _LEDGER.get()
    if ledger is not None:
        ledger.finite_operands += 2
        ledger.comparisons.append(
            {
                'name': label,
                'cosine': cosine,
                'tensor_similarity': similarity,
                'shape': list(actual.shape),
                'dtype': str(actual.dtype),
            }
        )
    assert cosine > 0.999, f'{label}: cosine={cosine}'
    assert similarity > 0.999, f'{label}: tensor_similarity={similarity}'


def is_mhc_scale(name):
    return 'hyper_connection.' in name and name.rsplit('.', 1)[-1] in (
        'alpha_pre',
        'alpha_post',
        'alpha_res',
    )


def assert_all_gradients(actual, native, links, *, decomposed=None):
    """Compare every linked gradient at the 0.999 bars.

    decomposed, when given, must be exactly the set of mHC scale links in this
    inventory, each verified by assert_mhc_decomposition; those shape-[1] links
    are then accounted for by their contribution tensors, never silently.
    """
    actual_parameters, native_parameters = assert_complete_mapping(actual, native, links)
    if decomposed is not None:
        scales = {link.actual for link in links if is_mhc_scale(link.actual)}
        assert set(decomposed) == scales, (sorted(decomposed), sorted(scales))
    decomposed = decomposed or {}
    for link in links:
        a, n = actual_parameters[link.actual], native_parameters[link.native]
        if not a.requires_grad:
            assert a.grad is n.grad is None, link
        else:
            assert a.grad is not None and n.grad is not None, link
            if link.actual in decomposed:
                assert decomposed[link.actual] == link.native, link
                group = link.actual.rsplit('_', 1)[-1]
                index = [name for name, _ in MHC_GROUPS].index(group)
                assert link.native_slice == slice(index, index + 1), link
                continue
            assert_similarity(
                linked_view(a.grad, link.actual_slice),
                linked_view(n.grad, link.native_slice),
                f'{link.actual} <- {link.native}',
            )


def _relative_difference(target, reference):
    target, reference = target.double(), reference.double()
    return ((target - reference).norm() / reference.norm().clamp_min(1e-300)).item()


class ClampReplay:
    """Replay target SwiGLU clamp decisions inside reference FFNs.

    Bind each reference FFN to the target's observed pre-activation FC1 output
    of the same site. The decision (gate > 10, up > 10, up < -10) is computed
    from the target values; the reference keeps its own values, so
    where(decision, +-10, value) is bitwise clamp() in value and gradient when
    both sides agree. A decision that differs from the reference's own must be
    a near tie: |v_target - v_ref| <= max(CLAMP_TIE_FLOOR,
    CLAMP_TIE_SIGMAS * r * rms_j), where r is the call's relative FC1
    difference for that half without the moved elements and rms_j is the
    reference RMS of the element's output column over the call. A column's
    difference scales with its own spread: the std-0.5 coverage columns, the
    only ones that reach the threshold, are about 25 times wider than the
    rest. Every bound site must run exactly once. Records reach the ledger
    before any assertion.
    """

    def __init__(self):
        self.sites, self.records = {}, []

    def bind(self, label, module, target):
        """Dense or shared-expert FFN; target is its [..., 2W] FC1 output."""
        gate, up = target.chunk(2, dim=-1)
        self._attach(label, module, lambda rows: (gate, up), rows_expected=False)

    def bind_routed(self, label, moe, target, routing, tokens_per_expert, permuted_probs, probs):
        """Routed experts; target rows are expert-major with ascending tokens.

        routing/probs are the target router's [tokens, experts] outputs. The
        rebuilt order must reproduce the target's per-expert counts and its
        permuted probabilities bitwise, so a misaligned replay cannot pass.
        """
        routing, probs = routing.cpu(), probs.cpu()
        experts = routing.size(1)
        assert len(moe.experts) == experts, label
        counts = routing.sum(0)
        assert counts.tolist() == [int(value) for value in tokens_per_expert], label
        mask = routing.T.contiguous()
        tokens = torch.arange(routing.size(0)).expand(experts, -1).masked_select(mask)
        expected_probs = probs.T.contiguous().masked_select(mask)
        assert torch.equal(permuted_probs.cpu(), expected_probs), f'{label}: permuted order'
        assert target.size(0) == tokens.numel(), label
        starts = [0] + torch.cumsum(counts, 0).tolist()
        for expert, module in enumerate(moe.experts):
            span = slice(starts[expert], starts[expert + 1])
            order, values = tokens[span], target[span]

            def select(rows, order=order, values=values, expert=expert):
                rows = rows.cpu()
                assert rows.numel() == order.numel(), (label, expert)
                index = torch.searchsorted(order, rows)
                assert torch.equal(order[index.clamp(max=order.numel() - 1)], rows), (label, expert)
                return values[index.to(values.device)].chunk(2, dim=-1)

            self._attach(f'{label}.{expert}', module, select, rows_expected=True)

    def _attach(self, label, module, select, *, rows_expected):
        assert label not in self.sites and getattr(module, 'clamp_replay', None) is None, label
        self.sites[label] = 0

        def replay(gate, up, rows):
            assert (rows is not None) is rows_expected, label
            self.sites[label] += 1
            target_gate, target_up = (value.to(device=gate.device) for value in select(rows))
            return self._apply(label, gate, up, target_gate, target_up)

        module.clamp_replay = replay

    def _apply(self, label, gate, up, target_gate, target_up):
        assert target_gate.shape == gate.shape and target_up.shape == up.shape, label
        assert target_gate.dtype == gate.dtype and target_up.dtype == up.dtype, label
        decided = (target_gate > CLAMP, target_up > CLAMP, target_up < -CLAMP)
        own = (gate.detach() > CLAMP, up.detach() > CLAMP, up.detach() < -CLAMP)
        moved = {'gate': own[0] ^ decided[0], 'up': (own[1] ^ decided[1]) | (own[2] ^ decided[2])}
        record = {'site': label, 'elements': gate.numel(), 'flips': {}}
        self.records.append(record)
        for name, half, target, value, mine, theirs in (
            ('gate_upper', 'gate', target_gate, gate, own[0], decided[0]),
            ('up_upper', 'up', target_up, up, own[1], decided[1]),
            ('up_lower', 'up', target_up, up, own[2], decided[2]),
        ):
            flipped = mine ^ theirs
            count = int(flipped.sum())
            record['flips'][name] = count
            if count:
                # The moved elements are excluded, so a gross one cannot widen its own band.
                kept = ~moved[half]
                relative = _relative_difference(target[kept], value.detach()[kept])
                reference = value.detach().double()
                columns = reference.reshape(-1, reference.size(-1)).square().mean(0).sqrt()
                band = (CLAMP_TIE_SIGMAS * relative * columns).clamp_min(CLAMP_TIE_FLOOR)
                spread = (target.double() - reference).abs()
                fraction = (spread / band)[flipped]
                worst = int(fraction.argmax())
                record.setdefault('ties', {})[name] = {
                    'max_fraction_of_band': fraction[worst].item(),
                    'abs_difference_at_max': spread[flipped][worst].item(),
                    'band_at_max': band.expand_as(spread)[flipped][worst].item(),
                    'relative_difference': relative,
                }
                if fraction[worst] > 1:
                    record_details('clamp_replay', self.summary())
                    raise AssertionError(
                        f'{label}: {name} decision is not a near tie {record["ties"][name]}'
                    )
        gate = torch.where(decided[0], CLAMP, gate)
        up = torch.where(decided[1], CLAMP, torch.where(decided[2], -CLAMP, up))
        return gate, up

    def summary(self):
        totals = {}
        for record in self.records:
            for name, count in record['flips'].items():
                totals[name] = totals.get(name, 0) + count
        return {
            'sites': len(self.sites),
            'calls': len(self.records),
            'flip_totals': totals,
            'tie_records': [record for record in self.records if 'ties' in record],
        }

    def assert_consumed(self):
        summary = self.summary()
        record_details('clamp_replay', summary)
        assert self.sites and all(count == 1 for count in self.sites.values()), {
            label: count for label, count in self.sites.items() if count != 1
        }
        return summary['flip_totals']


def mhc_contributions(gradient, projection, scales):
    """Per-group addends of the three scale gradients, in float64.

    gradient is dL/dproj (or dL/dmix), whose columns carry their group's scale
    factor; scales are the three scale values (pre, post, res).
    """
    gradient, projection = gradient.detach().double(), projection.detach().double()
    return {
        group: gradient[..., columns] / float(scales[index]) * projection[..., columns]
        for index, (group, columns) in enumerate(MHC_GROUPS)
    }


def assert_mhc_decomposition(actual, native, captures, sites):
    """Replace each shape-[1] mHC scale link with its contribution tensor.

    captures maps an actual mHC prefix to the target's _compute_h input
    {'proj', 'grad'}; sites lists (actual prefix, native NativeMHC name).
    Per group: the contribution tensors meet the 0.999 bars, and each side's
    scale gradient equals the sum of its contributions within
    MHC_RECONSTRUCTION_RTOL of their absolute sum. Returns
    {actual scale name: native scale name} for assert_all_gradients.
    """
    actual_parameters = dict(actual.named_parameters())
    verified, report = {}, []
    try:
        for prefix, native_name in sites:
            verified.update(
                _decompose_site(actual_parameters, native, captures, prefix, native_name, report)
            )
    finally:
        # Recorded even when a group fails, for the diagnosis.
        record_details('mhc_scale_decomposition', report)
    return verified


def _decompose_site(actual_parameters, native, captures, prefix, native_name, report):
    mhc = native.get_submodule(native_name)
    mix = mhc.last_mix
    assert mix.grad is not None, native_name
    target = captures[prefix]
    assert target['grad'] is not None, prefix
    alphas = [actual_parameters[f'{prefix}.alpha_{group}'] for group, _ in MHC_GROUPS]
    actual_parts = mhc_contributions(
        target['grad'], target['proj'], [float(alpha.detach()) for alpha in alphas]
    )
    native_parts = mhc_contributions(mix.grad, mix, mhc.scale.detach().tolist())
    verified = {}
    for index, (group, _) in enumerate(MHC_GROUPS):
        a = actual_parts[group]
        n = native_parts[group].to(a.device)
        sums = {}
        for side, parts, scalar in (
            ('actual', a, alphas[index].grad.detach().double().reshape(())),
            ('native', n, mhc.scale.grad.detach().double()[index].to(a.device)),
        ):
            sums[side] = (
                scalar.item(),
                parts.abs().sum().item(),
                (scalar - parts.sum()).abs().item(),
            )
        value_a, value_n = sums['actual'][0], sums['native'][0]
        report.append(
            {
                'link': f'{prefix}.alpha_{group}',
                'actual_scale_gradient': value_a,
                'native_scale_gradient': value_n,
                'scalar_similarity': 2 * value_a * value_n / (value_a**2 + value_n**2),
                'native_cancellation': sums['native'][1] / max(abs(value_n), 1e-300),
                'difference_over_abs_sum': abs(value_a - value_n) / sums['native'][1],
                'reconstruction_residuals': [sums['actual'][2], sums['native'][2]],
            }
        )
        for side, (scalar, scale, residual) in sums.items():
            assert residual <= MHC_RECONSTRUCTION_RTOL * scale, (
                f'{prefix}.alpha_{group} {side}: scale gradient {scalar} is not the sum '
                f'of its contributions (residual {residual}, abs sum {scale})'
            )
        assert_similarity(a, n, f'{prefix}.alpha_{group} contributions <- {native_name}')
        verified[f'{prefix}.alpha_{group}'] = f'{native_name}.scale'
    return verified
