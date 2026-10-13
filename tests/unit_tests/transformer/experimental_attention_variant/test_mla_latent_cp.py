# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for the experimental latent-KV context-parallel MLA (``MLAWithLatentCP``)."""

import itertools
import os
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
)
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_layer_specs,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import multi_latent_attention as mla_module
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import mla_latent_cp
from megatron.core.transformer.experimental_attention_variant.mla_latent_cp import (
    MLAWithLatentCP,
    apply_mla_latent_cp_spec,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

HIDDEN = 64
HEADS = 8
QK_HEAD_DIM, POS_DIM, V_HEAD_DIM = 128, 64, 128
CU_SEQLENS_PADDED = [0, 64, 192, 256, 512]
# Actual lengths well below the padded ones, so that whole chunks (and, at CP8 with an a2a
# group of 4, whole coarse chunks) consist of padding rows.
CU_SEQLENS_ACTUAL = [0, 48, 160, 200, 300]
TOTAL_TOKENS = CU_SEQLENS_PADDED[-1]
_IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
try:
    mla_latent_cp._fa4()
    HAVE_FA4_B11 = True
except Exception:  # pylint: disable=broad-except
    HAVE_FA4_B11 = False

requires_te = pytest.mark.skipif(
    not is_te_min_version("2.5.0", check_equality=True),
    reason="Requires TransformerEngine >= 2.5.0",
)
# The reference runs the production configuration: TE's cuDNN backend (NVTE_* are read at call
# time and the conftest disables both backends) on the native (192, 128) MLA shape. The old
# path pads V to the Q head dim for THD unless experimental_attention_variant is set (Kimi uses
# 'kda'); cuDNN has no (192, 192) training kernel on Blackwell, so skip that pad here. TE's
# ring is asked to batch its point-to-point sends/receives (as MLAWithLatentCP does): the
# per-peer NCCL communicators of unbatched isend/irecv can stall the test session's
# destroy_process_group on some clusters.
force_cudnn = mock.patch.dict(
    os.environ, {"NVTE_FUSED_ATTN": "1", "NVTE_FLASH_ATTN": "0", "NVTE_BATCH_MHA_P2P_COMM": "1"}
)
no_v_pad = mock.patch.object(
    mla_module,
    "_prepare_mla_core_attention_value",
    lambda attn, query, value, psp: (value, False, value.shape[-1], value.shape[-1]),
)


def _make_config(tp, cp, **overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=QK_HEAD_DIM,
        qk_pos_emb_head_dim=POS_DIM,
        v_head_dim=V_HEAD_DIM,
        qk_layernorm=True,
        attention_output_gate=True,
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
        context_parallel_size=cp,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.fused,
        mla_latent_cp=True,
    )
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


def _hcp_sizes(comm, cp, a2a):
    return [a2a, cp // a2a] if comm == "a2a+p2p" else None


def _attention_spec(kv_norm=False):
    spec = get_gpt_layer_with_transformer_engine_spec(
        multi_latent_attention=True, qk_layernorm=True
    ).submodules.self_attention
    if kv_norm:  # a separate KV norm module instead of the one fused into the up-projection
        spec.submodules.kv_layernorm = TENorm
        spec.submodules.linear_kv_up_proj = TEColumnParallelLinear
    return spec


def _build(spec, config, **kwargs):
    model_parallel_cuda_manual_seed(123)
    module = build_module(spec, config=config, layer_number=1, **kwargs).cuda()
    return module if config.fp8_param else module.bfloat16()  # .bfloat16() dequantizes fp8 params


def _packed(padded, **extra):
    cu = torch.tensor(CU_SEQLENS_ACTUAL if padded else CU_SEQLENS_PADDED, dtype=torch.int32).cuda()
    cup = torch.tensor(CU_SEQLENS_PADDED, dtype=torch.int32).cuda() if padded else None
    max_len = max(b - a for a, b in zip(CU_SEQLENS_PADDED[:-1], CU_SEQLENS_PADDED[1:]))
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cup,
        cu_seqlens_kv_padded=cup,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
        **extra,
    )


def _thd_sp_shard_token_indices(cu_seqlens_padded, cp_size, cp_rank, tp_size, tp_rank):
    """Global packed-token indices of one (cp_rank, tp_rank) sequence-parallel shard in the zigzag
    CP layout (rank r holds chunks r and 2P-1-r of every sequence)."""
    token_indices = []
    for seq_start, seq_end in zip(cu_seqlens_padded[:-1], cu_seqlens_padded[1:]):
        chunk_len = (seq_end - seq_start) // (2 * cp_size)
        for chunk in (cp_rank, 2 * cp_size - cp_rank - 1):
            chunk_start = seq_start + chunk * chunk_len
            token_indices.extend(range(chunk_start, chunk_start + chunk_len))
    cp_local = torch.tensor(token_indices, dtype=torch.long)
    assert cp_local.numel() % tp_size == 0
    return cp_local.chunk(tp_size)[tp_rank].contiguous()


def _gather_full_thd(local, token_indices, total_tokens, group):
    """Reassemble the full packed sequence from equal-sized shards of a process group."""
    local = local.detach().contiguous()
    token_indices = token_indices.to(local.device)
    gathered = [torch.empty_like(local) for _ in range(group.size())]
    gathered_indices = [torch.empty_like(token_indices) for _ in range(group.size())]
    dist.all_gather(gathered, local, group=group)
    dist.all_gather(gathered_indices, token_indices, group=group)
    full = torch.empty(
        (total_tokens,) + tuple(local.shape[1:]), dtype=local.dtype, device=local.device
    )
    for shard, indices in zip(gathered, gathered_indices):
        full[indices] = shard
    return full


def _assert_close(actual, ref, rtol, mre, what):
    """Whole-tensor max bound plus a mean-relative bound (bf16 noise on near-zero elements)."""
    actual, ref = actual.float(), ref.float()
    diff = (actual - ref).abs()
    max_ref, mean_ref = ref.abs().max().item(), ref.abs().mean().item()
    print(
        f"[parity] {what}: max|d|/max|ref|={diff.max().item() / max(max_ref, 1e-12):.2e} "
        f"mean|d|/mean|ref|={diff.mean().item() / max(mean_ref, 1e-12):.2e}"
    )
    assert (
        diff.max().item() <= rtol * max_ref + 1e-5
    ), f"{what}: max|d|={diff.max().item():.3e} > {rtol}*max|ref|={rtol * max_ref:.3e}"
    assert (
        diff.mean().item() <= mre * mean_ref + 1e-6
    ), f"{what}: mean|d|={diff.mean().item():.3e} > {mre}*mean|ref|={mre * mean_ref:.3e}"


def _compare_param_grads(new, ref, cp_group, tp_group, grad_tol):
    """Per-rank partial weight gradients differ between the paths (the up-projection runs on
    other ranks' latents, its norm on the whole local sequence with this TP rank's partial
    gradient, a2a ranks on their head slice); they agree after the reductions DDP performs: CP for
    every parameter, TP for the sequence-parallel (replicated) norm weights."""
    ref_params = dict(ref.named_parameters())
    for name, param in new.named_parameters():
        if param.grad is None:
            assert ref_params[name].grad is None, name
            continue
        # With gradient_accumulation_fusion TE accumulated into main_grad and flagged the param.
        fused = getattr(param, "grad_added_to_main_grad", False)
        new_grad = (param.main_grad if fused else param.grad).float().clone()
        ref_grad = ref_params[name].grad.float().clone()
        groups = [cp_group] + ([tp_group] if getattr(param, "sequence_parallel", False) else [])
        for group in groups:
            dist.all_reduce(new_grad, group=group)
            dist.all_reduce(ref_grad, group=group)
        _assert_close(new_grad, ref_grad, *grad_tol, name)


def _run_parity(
    tp,
    cp,
    padded,
    dynamic=False,
    kv_norm=False,
    fp8=False,
    fp8_param=False,
    two_stream=False,
    comm="p2p",
    a2a=1,
    heads=HEADS,
    qk_head_dim=QK_HEAD_DIM,
    v_head_dim=V_HEAD_DIM,
    kv_channels=None,
    fused_wgrad=False,
):
    """Old TE-CP MLA (p2p ring) vs MLAWithLatentCP on identical weights, inputs and upstream
    gradient; ``comm``/``a2a`` select the new path's communication scheme, ``fused_wgrad`` runs
    the new module with TE's fused weight-gradient accumulation into a manual ``main_grad``,
    ``fp8_param`` builds both modules with MXFP8 parameters (``fp8_model_init``)."""
    dims = dict(
        num_attention_heads=heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        kv_channels=kv_channels,
    )
    world = tp * cp
    if Utils.world_size < world or Utils.world_size % world != 0:
        pytest.skip(f"Needs a multiple of {world} CUDA ranks.")
    Utils.initialize_model_parallel(
        tp,
        1,
        context_parallel_size=cp,
        dynamic_context_parallel=dynamic,
        hierarchical_context_parallel_sizes=_hcp_sizes(comm, cp, a2a),
    )
    try:
        cp_group = parallel_state.get_context_parallel_group()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_cp_group = parallel_state.get_tensor_and_context_parallel_group()
        eff_group, eff_cp, extra = cp_group, cp, {}
        if dynamic:
            # A microbatch scheduled on a dynamic CP group of 2 inside the static CP group.
            eff_group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=2)
            eff_cp = 2
            extra = dict(local_cp_size=2, cp_group=eff_group)
        device = torch.device("cuda", torch.cuda.current_device())
        cp_rank, tp_rank = eff_group.rank(), tp_group.rank()
        tokens = TOTAL_TOKENS // eff_cp

        spec = _attention_spec(kv_norm)
        precision = dict(fp8="e4m3", fp8_recipe="mxfp8") if fp8 else {}
        if fp8_param:  # TE creates the parameters quantized, in params_dtype
            precision.update(fp8_param=True, params_dtype=torch.bfloat16)
        ref_config = _make_config(tp, cp, mla_latent_cp=False, **dims, **precision)
        new_config = _make_config(
            tp,
            cp,
            **dims,
            mla_latent_cp_two_stream=two_stream,
            cp_comm_type=comm,
            hierarchical_context_parallel_sizes=_hcp_sizes(comm, cp, a2a),
            gradient_accumulation_fusion=fused_wgrad,
            **precision,
        )
        # The full collection carries the hierarchical groups (as TransformerLayer passes it).
        pgs = ProcessGroupCollection.use_mpu_process_groups()
        with get_fp8_context(new_config, is_init=True):  # fp8_model_init under fp8_param
            ref = _build(spec, ref_config)
            new = _build(apply_mla_latent_cp_spec(spec), new_config, pg_collection=pgs)
        assert isinstance(new, MLAWithLatentCP) and isinstance(new.core_attention, IdentityOp)
        assert new.cp_comm_type == comm
        new.load_state_dict(ref.state_dict(), strict=False)
        if fused_wgrad:  # what DDP provides: an fp32 main_grad and the flag TE sets when it fused
            for p in new.parameters():
                p.main_grad = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                p.grad_added_to_main_grad = False

        torch.manual_seed(7)
        full_hidden = torch.randn(TOTAL_TOKENS, 1, HIDDEN, dtype=torch.bfloat16, device=device)
        full_grad = torch.randn_like(full_hidden)
        zigzag = _thd_sp_shard_token_indices(CU_SEQLENS_PADDED, eff_cp, cp_rank, tp, tp_rank).to(
            device
        )
        ref_in = full_hidden.index_select(0, zigzag).requires_grad_(True)
        with force_cudnn, no_v_pad, get_fp8_context(new_config):
            try:
                ref_out, _ = ref(ref_in, None, packed_seq_params=_packed(padded, **extra))
            except ValueError as error:
                if "No dot product attention backend" not in str(error):
                    raise
                pytest.skip("TE has no backend for padded THD MLA on this build (reference path)")
            ref_out.mul(full_grad.index_select(0, zigzag)).sum().backward()

        new_idx, new_psp = zigzag, _packed(padded, **extra)
        new_in = full_hidden.index_select(0, new_idx).requires_grad_(True)
        saved_shapes, hook_fires = [], []
        new.linear_kv_up_proj.weight.register_post_accumulate_grad_hook(
            lambda p: hook_fires.append(1)
        )

        def pack(t):
            saved_shapes.append(tuple(t.shape))
            return t

        save_hooks = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        with save_hooks, get_fp8_context(new_config):
            new_out, _ = new(new_in, None, packed_seq_params=new_psp)
        new_out.mul(full_grad.index_select(0, new_idx)).sum().backward()
        if kv_norm:  # the separate norm is differentiated inside the ring (not fused into TE)
            assert new.kv_layernorm.weight.grad is not None

        # The memory property: only the query (and the output) are saved, never K/V. With an a2a
        # group the ring saves the exchanged query (A times the tokens, H/A heads), not `query`.
        heads_tp = heads // tp
        q_shape = (tokens, heads_tp, qk_head_dim + POS_DIM)
        if a2a > 1:
            q_x = (a2a * tokens, heads_tp // a2a, q_shape[-1])
            out_x = (a2a * tokens, heads_tp // a2a, v_head_dim)
            same = q_x == out_x  # qk + pos == v: the saved ring output has the query's shape
            assert saved_shapes.count(q_x) == 1 + same, saved_shapes
            assert saved_shapes.count(q_shape) == 0, saved_shapes
            assert saved_shapes.count(out_x) <= 1 + same, saved_shapes
        else:
            assert saved_shapes.count(q_shape) <= 1, saved_shapes
            assert saved_shapes.count((tokens, heads_tp, v_head_dim)) <= 1, saved_shapes
        # The up-projection weight accumulates its gradient exactly once per microbatch.
        assert len(hook_fires) == 1

        # Both paths quantize the forward identically, so the output bound is the bf16 one. In the
        # backward, MXFP8 (E4M3, 3 mantissa bits) quantizes every rank's partial dK/dV before the
        # up-projection in the ring but their CP sum once in TE's ring, so the two paths carry
        # independent rounding noise of the same size: measured 8e-2 max / 5e-2 mean relative on
        # GB300 (tp=2, cp=2) against <= 1e-2 measured in bf16.
        out_tol = (1e-2, 2e-2)
        grad_tol = (1.5e-1, 1e-1) if fp8 else (2e-2, 3e-2)
        full_ref = _gather_full_thd(ref_out, zigzag, TOTAL_TOKENS, tp_cp_group)
        full_new = _gather_full_thd(new_out, new_idx, TOTAL_TOKENS, tp_cp_group)
        _assert_close(full_new, full_ref, *out_tol, "output")
        full_ref_grad = _gather_full_thd(ref_in.grad, zigzag, TOTAL_TOKENS, tp_cp_group)
        full_new_grad = _gather_full_thd(new_in.grad, new_idx, TOTAL_TOKENS, tp_cp_group)
        _assert_close(full_new_grad, full_ref_grad, *grad_tol, "input grad")
        _compare_param_grads(new, ref, cp_group, tp_group, grad_tol)
    finally:
        Utils.destroy_model_parallel()


# (tp, cp, cp_comm_type, a2a group size, padded); the last four need 8 GPUs.
A2A_CASES = (
    (1, 4, "a2a+p2p", 2, False),
    (1, 4, "a2a+p2p", 2, True),
    (1, 4, "a2a", 4, False),
    (1, 4, "a2a", 4, True),
    (2, 2, "a2a", 2, False),
    (2, 4, "a2a+p2p", 2, False),
    (1, 8, "a2a+p2p", 2, False),
    (1, 8, "a2a+p2p", 4, True),
    (1, 8, "a2a", 8, True),
)


@pytest.mark.experimental
@requires_te
class TestMLALatentCPParity:
    """Old vs new path with the cuDNN backend (the CI backend)."""

    @pytest.mark.parametrize(("tp", "cp"), ((1, 2), (2, 2), (1, 4), (2, 4), (4, 2)))
    @pytest.mark.parametrize("padded", (False, True), ids=("packed", "padded"))
    def test_parity(self, tp, cp, padded):
        _run_parity(tp, cp, padded)

    def test_parity_cp1(self):
        _run_parity(2, 1, False)

    def test_parity_dynamic_cp(self):
        _run_parity(1, 4, False, dynamic=True)

    def test_parity_separate_kv_norm(self):
        _run_parity(1, 2, False, kv_norm=True)

    @pytest.mark.skipif(not _IS_BLACKWELL, reason="MXFP8 needs Blackwell")
    def test_parity_mxfp8(self):
        _run_parity(2, 2, False, fp8=True)

    @pytest.mark.parametrize(("tp", "cp", "padded"), ((1, 2, False), (2, 4, True)))
    def test_parity_two_stream(self, tp, cp, padded):
        _run_parity(tp, cp, padded, two_stream=True)

    @pytest.mark.parametrize(("tp", "cp", "comm", "a2a", "padded"), A2A_CASES)
    def test_parity_a2a(self, tp, cp, comm, a2a, padded):
        _run_parity(tp, cp, padded, comm=comm, a2a=a2a)

    @pytest.mark.skipif(not _IS_BLACKWELL, reason="MXFP8 needs Blackwell")
    def test_parity_a2a_mxfp8(self):
        _run_parity(1, 4, False, fp8=True, comm="a2a+p2p", a2a=2)

    @pytest.mark.skipif(not _IS_BLACKWELL, reason="MXFP8 needs Blackwell")
    def test_parity_a2a_fp8_param(self):
        """The production combination: MXFP8 parameters, the weight slab as TE's quantized row
        split of the parameter, and the fused wgrad into the slab's main_grad rows."""
        _run_parity(1, 4, False, fp8=True, fp8_param=True, fused_wgrad=True, comm="a2a+p2p", a2a=2)

    @pytest.mark.skipif(not _IS_BLACKWELL, reason="MXFP8 needs Blackwell")
    def test_parity_a2a_fp8_param_padded_scales(self):
        """A slab of 3 heads x (64 + 128) = 576 weight rows: TE's quantized row split still
        applies (576 % 32 == 0) but pads copies of the slab's scales (576 % 128 != 0; 18
        columnwise scale rows % 4 != 0) instead of viewing them; with 256-wide heads (the
        fixture, Kimi-K3) every slab row count is a multiple of 128 and only the pure-view split
        is reached. kv_channels: Attention.__init__ quantizes a kv_channels * heads output
        projection before MLA replaces it, and the default hidden // 12 = 5 is not quantizable."""
        _run_parity(
            1,
            4,
            False,
            fp8=True,
            fp8_param=True,
            fused_wgrad=True,
            comm="a2a",
            a2a=4,
            heads=12,
            qk_head_dim=64,
            kv_channels=V_HEAD_DIM,
        )

    def test_parity_a2a_odd_heads(self):
        _run_parity(1, 4, True, comm="a2a", a2a=4, heads=12)  # three heads per a2a rank

    @pytest.mark.parametrize(("tp", "cp", "comm", "a2a"), ((1, 2, "p2p", 1), (1, 4, "a2a+p2p", 2)))
    def test_parity_fused_wgrad(self, tp, cp, comm, a2a):
        """TE's fused weight-gradient accumulation: the full weight's main_grad (p2p) and the
        head-slab view into it (a2a modes) match the reference after the CP reduction."""
        _run_parity(tp, cp, False, comm=comm, a2a=a2a, fused_wgrad=True)

    def test_a2a_rejects_dynamic_cp(self):
        if Utils.world_size < 4 or Utils.world_size % 4 != 0:
            pytest.skip("Needs a multiple of 4 CUDA ranks.")
        Utils.initialize_model_parallel(
            1, 1, context_parallel_size=4, dynamic_context_parallel=True
        )
        try:
            new = _build(
                apply_mla_latent_cp_spec(_attention_spec()), _make_config(1, 4, cp_comm_type="a2a")
            )
            group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=2)
            x = torch.randn(TOTAL_TOKENS // 2, 1, HIDDEN, dtype=torch.bfloat16, device="cuda")
            with pytest.raises(NotImplementedError, match="dynamic"):
                new(x, None, packed_seq_params=_packed(False, local_cp_size=2, cp_group=group))
        finally:
            Utils.destroy_model_parallel()


@pytest.mark.experimental
@requires_te
def test_old_path_a2a_p2p_matches_p2p():
    """TE's own a2a+p2p on THD MLA (the perf reference of the hierarchical recipes) agrees with
    its p2p ring; both are the old path, so this only qualifies the reference."""
    tp, cp, a2a = 1, 4, 2
    if Utils.world_size < cp or Utils.world_size % cp != 0:
        pytest.skip(f"Needs a multiple of {cp} CUDA ranks.")
    Utils.initialize_model_parallel(
        tp, 1, context_parallel_size=cp, hierarchical_context_parallel_sizes=[a2a, cp // a2a]
    )
    try:
        cp_group = parallel_state.get_context_parallel_group()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_cp_group = parallel_state.get_tensor_and_context_parallel_group()
        config = _make_config(tp, cp, mla_latent_cp=False)
        pgs = ProcessGroupCollection.use_mpu_process_groups()
        spec = _attention_spec()
        p2p = _build(spec, config, pg_collection=pgs)
        hier = _build(spec, config, pg_collection=pgs, cp_comm_type="a2a+p2p")
        hier.load_state_dict(p2p.state_dict())
        torch.manual_seed(7)
        full_hidden = torch.randn(TOTAL_TOKENS, 1, HIDDEN, dtype=torch.bfloat16).cuda()
        idx = _thd_sp_shard_token_indices(
            CU_SEQLENS_PADDED, cp, cp_group.rank(), tp, tp_group.rank()
        ).cuda()
        outs, grads = [], []
        with force_cudnn, no_v_pad:
            for module in (p2p, hier):
                x = full_hidden.index_select(0, idx).requires_grad_(True)
                out, _ = module(x, None, packed_seq_params=_packed(True))
                out.sum().backward()
                outs.append(_gather_full_thd(out, idx, TOTAL_TOKENS, tp_cp_group))
                grads.append(_gather_full_thd(x.grad, idx, TOTAL_TOKENS, tp_cp_group))
        _assert_close(outs[1], outs[0], 1e-2, 2e-2, "output")
        _assert_close(grads[1], grads[0], 2e-2, 3e-2, "input grad")
        _compare_param_grads(hier, p2p, cp_group, tp_group, (2e-2, 3e-2))
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.experimental
@pytest.mark.skipif(not (_IS_BLACKWELL and HAVE_FA4_B11), reason="FlashAttention-4 needs Blackwell")
class TestMLALatentCPFlashAttn4:
    """FA4 backend against the cuDNN backend of the same module."""

    @pytest.mark.parametrize(
        ("tp", "cp", "comm", "a2a"), ((1, 2, "p2p", 1), (1, 4, "p2p", 1), (1, 4, "a2a+p2p", 2))
    )
    def test_flash_matches_fused(self, tp, cp, comm, a2a):
        world = tp * cp
        if Utils.world_size < world or Utils.world_size % world != 0:
            pytest.skip(f"Needs a multiple of {world} CUDA ranks.")
        Utils.initialize_model_parallel(
            tp,
            1,
            context_parallel_size=cp,
            hierarchical_context_parallel_sizes=_hcp_sizes(comm, cp, a2a),
        )
        try:
            cp_rank = parallel_state.get_context_parallel_group().rank()
            tp_rank = parallel_state.get_tensor_model_parallel_group().rank()
            spec = apply_mla_latent_cp_spec(_attention_spec())
            mode = dict(
                cp_comm_type=comm, hierarchical_context_parallel_sizes=_hcp_sizes(comm, cp, a2a)
            )
            pgs = ProcessGroupCollection.use_mpu_process_groups()
            fused = _build(spec, _make_config(tp, cp, **mode), pg_collection=pgs)
            flash_config = _make_config(tp, cp, attention_backend=AttnBackend.flash, **mode)
            flash = _build(spec, flash_config, pg_collection=pgs)
            flash.load_state_dict(fused.state_dict())
            torch.manual_seed(7)
            full_hidden = torch.randn(TOTAL_TOKENS, 1, HIDDEN, dtype=torch.bfloat16).cuda()
            idx = _thd_sp_shard_token_indices(CU_SEQLENS_PADDED, cp, cp_rank, tp, tp_rank).cuda()
            outs, grads = [], []
            for module in (fused, flash):
                x = full_hidden.index_select(0, idx).requires_grad_(True)
                out, _ = module(x, None, packed_seq_params=_packed(True))
                out.sum().backward()
                outs.append(out)
                grads.append(x.grad)
            _assert_close(outs[1], outs[0], 1e-2, 2e-2, "output")
            _assert_close(grads[1], grads[0], 1e-2, 2e-2, "input grad")
            for (name, p_flash), p_fused in zip(flash.named_parameters(), fused.parameters()):
                if p_fused.grad is not None:
                    _assert_close(p_flash.grad, p_fused.grad, 2e-2, 3e-2, name)
        finally:
            Utils.destroy_model_parallel()


def _backends():
    yield pytest.param((mla_latent_cp._cudnn_fwd, mla_latent_cp._cudnn_bwd), id="cudnn")
    if _IS_BLACKWELL and HAVE_FA4_B11:
        yield pytest.param((mla_latent_cp._fa4_fwd, mla_latent_cp._fa4_bwd), id="fa4")


@pytest.mark.experimental
@requires_te
@pytest.mark.parametrize("backend", list(_backends()))
def test_backend_half_kv_zero_copy(backend):
    """A ring step that attends only the first chunk of a received latent passes the full padded
    offsets with half lengths; the kernel must read the first half and leave the rest zero."""
    fwd, bwd = backend
    torch.manual_seed(0)
    n, h = 128, 2
    q = torch.randn(n, h, QK_HEAD_DIM + POS_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn(n, h, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    off = torch.tensor([0, n], dtype=torch.int32, device="cuda")
    cu_half = torch.tensor([0, n // 2], dtype=torch.int32, device="cuda")
    scale = 0.1
    step = mla_latent_cp._Step(False, off, cu_half, off, off, n, n // 2, False)
    out, lse, rng = fwd(q, k, v, step, scale)
    dout = torch.randn_like(out)
    dq, dk, dv = bwd(q, k, v, out, dout, lse, rng, step, scale, False)

    qf, kf, vf = (
        t.float().transpose(0, 1).contiguous().requires_grad_(True) for t in (q, k, v)
    )  # [h, n, d]
    scores = scale * qf @ kf[:, : n // 2].transpose(1, 2)
    probs = scores.softmax(-1)
    ref_out = probs @ vf[:, : n // 2]
    ref_lse = torch.logsumexp(scores, -1)
    ref_out.backward(dout.float().transpose(0, 1))
    _assert_close(out.transpose(0, 1), ref_out, 2e-2, 2e-2, "out")
    _assert_close(lse.transpose(0, 1), ref_lse, 1e-2, 1e-2, "lse")
    _assert_close(dq.transpose(0, 1), qf.grad, 2e-2, 3e-2, "dq")
    _assert_close(dk.transpose(0, 1), kf.grad, 2e-2, 3e-2, "dk")
    _assert_close(dv.transpose(0, 1), vf.grad, 2e-2, 3e-2, "dv")
    assert torch.equal(dk[n // 2 :], torch.zeros_like(dk[n // 2 :]))
    assert torch.equal(dv[n // 2 :], torch.zeros_like(dv[n // 2 :]))


@pytest.mark.parametrize(("cp", "a2a"), ((16, 4), (16, 8), (16, 2), (8, 4), (4, 4)))
def test_a2a_permutation_matches_chunk_bookkeeping(cp, a2a):
    """The gather over the a2a group (rank-major) permuted with ``_a2a_permutation`` is the
    sequence order of the p2p zigzag: for every p2p rank, the global chunk ids of both halves are
    ascending and contiguous (natural order when the a2a group is the whole CP group)."""
    lengths = [2 * cp * n for n in (1, 3, 2, 4)]
    cup = torch.tensor([0] + list(itertools.accumulate(lengths)), dtype=torch.int32)
    psp = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cup, cu_seqlens_q_padded=cup, max_seqlen_q=max(lengths)
    )
    tokens = cup[-1].item() // cp
    perm, inv = mla_latent_cp._a2a_permutation(psp, cp, a2a, tokens)
    assert torch.equal(perm.index_select(0, inv), torch.arange(a2a * tokens))

    def chunks(rank_chunks):  # global token ids of the given chunk ids, sequence by sequence
        ids = []
        for s0, s1 in zip(cup[:-1].tolist(), cup[1:].tolist()):
            c = (s1 - s0) // (2 * cp)
            for first, last in rank_chunks:
                ids += list(range(s0 + first * c, s0 + (last + 1) * c))
        return torch.tensor(ids)

    for p in range(cp // a2a):
        # Rank r holds chunks r and 2P-1-r; the a2a group of p2p rank p is ranks pA .. pA+A-1.
        gathered = torch.cat(
            [
                chunks([(r, r), (2 * cp - 1 - r, 2 * cp - 1 - r)])
                for r in range(p * a2a, (p + 1) * a2a)
            ]
        )
        expected = chunks(
            [(p * a2a, (p + 1) * a2a - 1), (2 * cp - (p + 1) * a2a, 2 * cp - 1 - p * a2a)]
        )
        assert torch.equal(gathered.index_select(0, perm), expected), (cp, a2a, p)


@pytest.mark.parametrize(
    "overrides",
    (
        dict(multi_latent_attention=False),
        dict(cp_comm_type="all_gather"),
        dict(cp_comm_type="a2a+p2p"),  # no hierarchical sizes
        dict(
            cp_comm_type="a2a+p2p",
            context_parallel_size=4,
            hierarchical_context_parallel_sizes=[2, 4],
        ),
        dict(
            cp_comm_type="a2a",
            context_parallel_size=4,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
        ),  # 8 heads / TP4 = 2 heads per rank, not divisible by the a2a group of 4
        dict(cp_comm_type="a2a", dynamic_context_parallel=True),
        dict(
            cp_comm_type="a2a", recompute_granularity="selective", recompute_modules=["mla_up_proj"]
        ),
        dict(attention_backend=AttnBackend.unfused),
        dict(tensor_model_parallel_size=2, sequence_parallel=False),
        dict(attention_dropout=0.1),
        dict(recompute_granularity="selective", recompute_modules=["core_attn"]),
        dict(delay_wgrad_compute=True),
        dict(tp_comm_overlap=True),
        dict(fp8="e4m3", fp8_recipe="delayed"),
        dict(mla_down_proj_fusion=True),
        dict(mla_latent_cp=False, mla_latent_cp_two_stream=True),
    ),
)
def test_config_rejections(overrides):
    with pytest.raises(ValueError, match=r"mla_latent_cp( does not support|_two_stream requires)"):
        _make_config(1, 1, **overrides)


@requires_te
def test_module_rejections():
    Utils.initialize_model_parallel(1, 1)
    try:
        spec = apply_mla_latent_cp_spec(_attention_spec())
        with pytest.raises(ValueError, match="mla_latent_cp=True"):
            _build(spec, _make_config(1, 1, mla_latent_cp=False))
        local = _attention_spec(kv_norm=True)
        local.submodules.linear_kv_up_proj = ColumnParallelLinear
        with pytest.raises(NotImplementedError, match="Transformer Engine linear"):
            _build(apply_mla_latent_cp_spec(local), _make_config(1, 1))
        # a2a+p2p needs the hierarchical groups; a collection without them is rejected.
        no_hcp = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
            tp_cp=parallel_state.get_tensor_and_context_parallel_group(),
        )
        hier = _make_config(
            1, 1, cp_comm_type="a2a+p2p", hierarchical_context_parallel_sizes=[1, 1]
        )
        with pytest.raises(ValueError, match="hcp"):
            _build(spec, hier, cp_comm_type="a2a+p2p", pg_collection=no_hcp)
        module = _build(spec, _make_config(1, 1))
        x = torch.randn(16, 1, HIDDEN, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="THD"):
            module(x, None)
    finally:
        Utils.destroy_model_parallel()


def test_spec_helper_is_pure_and_default_unchanged():
    spec = _attention_spec()
    new_spec = apply_mla_latent_cp_spec(spec)
    assert new_spec.module is MLAWithLatentCP
    assert new_spec.submodules.core_attention is IdentityOp
    # The old spec (what every model builds with mla_latent_cp=False) is untouched.
    assert spec.module is MLASelfAttention
    assert spec.submodules.core_attention is TEDotProductAttention


@requires_te
@pytest.mark.parametrize(
    ("flag", "module", "core"),
    ((False, MLASelfAttention, TEDotProductAttention), (True, MLAWithLatentCP, IdentityOp)),
)
def test_gated_spec_swap_gpt_and_hybrid(flag, module, core):
    """The two call sites swap the MLA attention only when mla_latent_cp is set."""
    Utils.initialize_model_parallel(1, 1)
    try:
        # One dense and one MoE layer, so both GPT layer specs are checked.
        config = _make_config(1, 1, mla_latent_cp=flag, num_moe_experts=4, moe_layer_freq=[0, 1])
        layer_specs = get_gpt_decoder_layer_specs(config, use_transformer_engine=True)
        assert len(layer_specs) == 2
        for layer_spec in layer_specs:
            attention = layer_spec.submodules.self_attention
            assert attention.module is module and attention.submodules.core_attention is core
        stack = HybridStack(  # one MLA layer ('+'), as the Kimi hybrid pattern builds it
            config,
            hybrid_stack_spec.submodules,
            layer_type_list=["+"],
            post_layer_norm=False,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        assert type(stack.layers[0].self_attention) is module
        assert type(stack.layers[0].self_attention.core_attention) is core
    finally:
        Utils.destroy_model_parallel()
