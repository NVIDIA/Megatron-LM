# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import copy
import os
import types
from typing import List, Optional, Tuple

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state as mpu
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.resharding.refit import clear_all_caches, swap_model_weights
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    import nvshmem.core

    has_nvshmem = True
except Exception:
    has_nvshmem = False


def _build_pg_collection(
    tp_size: int, pp_size: int = None, ep_size: int = 1
) -> ProcessGroupCollection:
    cp_size = mpu.get_context_parallel_world_size()
    if pp_size is None:
        pp_size = mpu.get_pipeline_model_parallel_world_size()
    world_size = dist.get_world_size()
    dp_size = world_size // (tp_size * cp_size * ep_size * pp_size)
    assert dp_size >= 1 and (tp_size * cp_size * ep_size * pp_size * dp_size) == world_size

    grid = HyperCommGrid(
        [tp_size, cp_size, ep_size, pp_size, dp_size], ["tp", "cp", "ep", "pp", "dp"]
    )
    tp_group = grid.create_pg("tp")
    cp_group = grid.create_pg("cp")
    pp_group = grid.create_pg("pp")
    ep_group = grid.create_pg("ep")
    dp_group = grid.create_pg("dp")
    # Composite groups required by MoE/router and some utilities
    tp_cp_group = grid.create_pg(["tp", "cp"])
    mp_group = grid.create_pg(["tp", "cp", "ep", "pp"])
    tp_ep_group = grid.create_pg(["tp", "ep"])
    tp_ep_pp_group = grid.create_pg(["tp", "ep", "pp"])
    dp_cp_group = grid.create_pg(["cp", "dp"])
    tp_dp_cp_group = grid.create_pg(["tp", "cp", "dp"])
    embd_group_ranks = mpu.default_embedding_ranks(dist.get_process_group_ranks(pp_group))
    embd_group = dist.new_group(ranks=embd_group_ranks)
    pos_embd_group_ranks = mpu.default_position_embedding_ranks(
        dist.get_process_group_ranks(pp_group)
    )
    pos_embd_group = dist.new_group(ranks=pos_embd_group_ranks)
    return ProcessGroupCollection(
        tp=tp_group,
        cp=cp_group,
        pp=pp_group,
        ep=ep_group,
        embd=embd_group,
        pos_embd=pos_embd_group,
        dp=dp_group,
        tp_cp=tp_cp_group,
        mp=mp_group,
        expt_tp=tp_group,
        expt_dp=dp_group,
        tp_ep=tp_ep_group,
        tp_ep_pp=tp_ep_pp_group,
        dp_cp=dp_cp_group,
        tp_dp_cp=tp_dp_cp_group,
    )


def _build_gpt(
    config: TransformerConfig,
    vocab_size: int,
    seq_len: int,
    pg_collection,
    parallel_output: bool = True,
    num_moe_experts: Optional[int] = None,
) -> GPTModel:
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=num_moe_experts, moe_grouped_gemm=(num_moe_experts is not None)
    )
    mtp_block_spec = None
    if config.mtp_num_layers:
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=layer_spec, use_transformer_engine=True
        )
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        pre_process=True,
        post_process=True,
        fp16_lm_cross_entropy=False,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        pg_collection=pg_collection,
        mtp_block_spec=mtp_block_spec,
    )
    return model


def _mp_config() -> ModelParallelConfig:
    return ModelParallelConfig(
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
    )


def _set_pg_collection(module, tp_group, dp_group):
    module.pg_collection = types.SimpleNamespace(tp=tp_group, dp=dp_group, ep=None, pp=None)
    return module


@pytest.mark.parametrize(
    "refit_backend",
    [
        pytest.param(
            "nvshmem",
            marks=pytest.mark.skipif(
                not has_nvshmem,
                reason="nvshmem.core is not available (NVSHMEM Python bindings not installed)",
            ),
        ),
        "nccl",
        "gloo",
    ],
)
@pytest.mark.parametrize(
    "src_tp,src_pp,src_ep,dst_tp,dst_pp,dst_ep,num_experts",
    [
        # TP only changes
        (2, 1, 1, 1, 1, 1, None),  # TP2 -> TP1
        (1, 1, 1, 2, 1, 1, None),  # TP1 -> TP2
        (2, 1, 1, 4, 1, 1, None),  # TP2 -> TP4
        # PP only changes
        (1, 2, 1, 1, 1, 1, None),  # PP2 -> PP1
        (1, 1, 1, 1, 2, 1, None),  # PP1 -> PP2
        # Both TP and PP change
        (2, 2, 1, 1, 1, 1, None),  # TP2,PP2 -> TP1,PP1
        (1, 1, 1, 2, 2, 1, None),  # TP1,PP1 -> TP2,PP2
        (2, 1, 1, 1, 2, 1, None),  # TP2,PP1 -> TP1,PP2
        (1, 2, 1, 2, 1, 1, None),  # TP1,PP2 -> TP2,PP1
        (1, 2, 1, 2, 4, 1, None),  # TP1,PP2 -> TP2,PP4
        # MoE EP changes
        (1, 1, 2, 1, 1, 4, 4),  # EP2 -> EP4
        (1, 1, 2, 1, 1, 1, 4),  # EP2 -> EP1
        (1, 1, 1, 1, 1, 2, 4),  # EP1 -> EP2
        (1, 1, 2, 1, 2, 2, 4),  # EP2 -> PP2,EP2
        # MoE mixed TP + EP
        (2, 1, 2, 1, 1, 1, 4),  # TP2,EP2 -> TP1,EP1
        (1, 1, 1, 2, 1, 2, 4),  # TP1,EP1 -> TP2,EP2
        (4, 1, 1, 2, 1, 2, 4),  # TP4,EP1 -> TP2,EP2
        (2, 1, 2, 4, 1, 1, 4),  # TP2,EP2 -> TP4,EP1
        (4, 1, 1, 1, 1, 4, 4),  # TP4,EP1 -> TP1,EP4
        (1, 1, 4, 4, 1, 1, 4),  # EP4 -> TP4,EP1
    ],
)
@pytest.mark.parametrize(
    "moe_mode",
    [None, "latent", "latent_mtp"],
)
def test_swap_gpt_parametrized(
    refit_backend: str,
    src_tp: int,
    src_pp: int,
    src_ep: int,
    dst_tp: int,
    dst_pp: int,
    dst_ep: int,
    num_experts: Optional[int],
    moe_mode: Optional[str],
):
    # latent/mtp modes only apply to MoE configs
    if moe_mode is not None and num_experts is None:
        pytest.skip("latent/mtp modes only apply to MoE configurations")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=src_tp, pipeline_model_parallel_size=src_pp
    )
    world = dist.get_world_size()
    if (world % (src_tp * src_pp * src_ep) != 0) or (world % (dst_tp * dst_pp * dst_ep) != 0):
        Utils.destroy_model_parallel()
        pytest.skip(
            "WORLD_SIZE must be divisible by both src_tp*src_pp*src_ep and dst_tp*dst_pp*dst_ep"
        )
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    seq_len = 8
    vocab_size = 128
    cfg = TransformerConfig(
        num_layers=4 if (src_pp > 1 or dst_pp > 1) else 2,
        hidden_size=32,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        moe_router_dtype="fp64",
        moe_token_dispatcher_type="alltoall",
        num_query_groups=4,
    )

    src_pgs = _build_pg_collection(tp_size=src_tp, pp_size=src_pp, ep_size=src_ep)
    dst_pgs = _build_pg_collection(tp_size=dst_tp, pp_size=dst_pp, ep_size=dst_ep)
    src_cfg = copy.deepcopy(cfg)
    dst_cfg = copy.deepcopy(cfg)

    if num_experts is not None:
        for c, ep in [(src_cfg, src_ep), (dst_cfg, dst_ep)]:
            c.num_moe_experts = num_experts
            c.moe_ffn_hidden_size = c.ffn_hidden_size
            c.expert_model_parallel_size = ep
            c.moe_grouped_gemm = True
            c.add_bias_linear = False
            if moe_mode in ("latent", "latent_mtp"):
                c.moe_latent_size = 16
                c.moe_shared_expert_intermediate_size = 64
                c.activation_func = torch.nn.functional.silu
                c.gated_linear_unit = True
            if moe_mode == "latent_mtp":
                c.mtp_num_layers = 1
        try:
            import transformer_engine
        except Exception:
            Utils.destroy_model_parallel()
            pytest.skip("Transformer Engine not available; skipping MoE refit test")

    src_model = (
        _build_gpt(
            src_cfg, vocab_size, seq_len, src_pgs,
            parallel_output=False, num_moe_experts=num_experts,
        )
        .to(device)
        .eval()
    )
    dst_model = (
        _build_gpt(
            dst_cfg, vocab_size, seq_len, dst_pgs,
            parallel_output=False, num_moe_experts=num_experts,
        )
        .to(device)
        .eval()
    )

    batch = 2
    tokens = torch.randint(
        low=0, high=vocab_size, size=(batch, seq_len), device=device, dtype=torch.long
    )
    position_ids = (
        torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    )
    attention_mask = torch.ones((batch, 1, seq_len, seq_len), device=device, dtype=torch.bool)

    # Collect source reference logits
    ref_logits = torch.empty(batch, seq_len, vocab_size, device=device, dtype=torch.float32)
    src_pp_ranks = dist.get_process_group_ranks(src_pgs.pp)
    src_last_pp_rank = src_pp_ranks[-1]
    with torch.no_grad():
        src_out = src_model(tokens, position_ids, attention_mask)
        if dist.get_rank() == src_last_pp_rank:
            ref_logits.copy_(src_out)
    dist.broadcast(ref_logits, src=src_last_pp_rank, group=src_pgs.pp)

    # Swap weights
    swap_model_weights([src_model], [dst_model], refit_method=refit_backend)

    # Collect destination logits
    dst_logits = torch.empty(batch, seq_len, vocab_size, device=device, dtype=torch.float32)
    dst_pp_ranks = dist.get_process_group_ranks(dst_pgs.pp)
    dst_last_pp_rank = dst_pp_ranks[-1]
    with torch.no_grad():
        dst_out = dst_model(tokens, position_ids, attention_mask)
        if dist.get_rank() == dst_last_pp_rank:
            dst_logits.copy_(dst_out)
    dist.broadcast(dst_logits, src=dst_last_pp_rank, group=dst_pgs.pp)

    # Compare
    assert ref_logits.shape == dst_logits.shape
    max_diff = (dst_logits - ref_logits).abs().max().item()
    assert torch.allclose(
        dst_logits, ref_logits, atol=5e-4, rtol=5e-4
    ), (
        f"Refit src(TP={src_tp},PP={src_pp},EP={src_ep})"
        f"->dst(TP={dst_tp},PP={dst_pp},EP={dst_ep}) "
        f"moe_mode={moe_mode} outputs differ (max_diff={max_diff:.6f})"
    )
    dist.barrier()

    clear_all_caches()
    Utils.destroy_model_parallel()
