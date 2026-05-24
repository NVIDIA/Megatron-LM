# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
import copy
import gc
import os
import types
from dataclasses import fields
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

try:
    import mamba_ssm  # noqa: F401

    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel

    has_mamba_deps = True
except Exception:
    has_mamba_deps = False


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


def _destroy_pg_collection(pgc: ProcessGroupCollection):
    """Destroy all process groups in a ProcessGroupCollection to free NCCL communicator memory."""
    destroyed = set()
    for f in fields(pgc):
        pg = getattr(pgc, f.name, None)
        if pg is not None and id(pg) not in destroyed:
            destroyed.add(id(pg))
            dist.destroy_process_group(pg)


def _pp_flags(pg_collection) -> Tuple[bool, bool]:
    """Return (pre_process, post_process) based on pipeline-parallel rank."""
    pp_group = pg_collection.pp
    pp_rank = dist.get_rank(pp_group)
    pp_size = dist.get_world_size(pp_group)
    return pp_rank == 0, pp_rank == pp_size - 1


def _run_forward(model, tokens, position_ids, attention_mask, pg_collection):
    """Run a forward pass using Megatron's pipeline schedule.

    For PP=1 this is a simple forward call.  For PP>1 this delegates to the
    Megatron pipeline schedule which handles P2P communication between stages.

    Returns logits on the last PP stage, None on other stages.
    """
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

    pp_group = pg_collection.pp
    pp_size = dist.get_world_size(pp_group)
    batch, seq_len = tokens.shape

    def forward_step_func(data_iterator, model):
        output = model(tokens, position_ids, attention_mask)

        def loss_func(output_tensor, non_loss_data=False):
            if non_loss_data:
                return output_tensor
            return output_tensor.sum(), {"logits": output_tensor}

        return output, loss_func

    forward_backward_func = get_forward_backward_func(pp_size=pp_size, vp_size=None)
    kwargs = dict(
        forward_step_func=forward_step_func,
        data_iterator=iter([None]),
        model=[model],
        num_microbatches=1,
        seq_length=seq_len,
        micro_batch_size=batch,
        forward_only=True,
        collect_non_loss_data=True,
        pg_collection=pg_collection,
    )
    if pp_size > 1:
        kwargs["p2p_communicator"] = P2PCommunicator(pp_group, model.config)
    result = forward_backward_func(**kwargs)
    # result is a list of per-microbatch outputs; only populated on last PP stage
    if result and result[0] is not None:
        return result[0]
    return None


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
    pre_process, post_process = _pp_flags(pg_collection)
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=False,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_percent=1.0,
        pg_collection=pg_collection,
        mtp_block_spec=mtp_block_spec,
    )
    return model


def _build_mamba(
    config: TransformerConfig,
    vocab_size: int,
    seq_len: int,
    pg_collection,
    hybrid_layer_pattern: str,
    parallel_output: bool = True,
):
    pre_process, post_process = _pp_flags(pg_collection)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        hybrid_layer_pattern=hybrid_layer_pattern,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=False,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=False,
        pg_collection=pg_collection,
    )
    return model


def _mamba_layer_pattern(base: str, num_layers: int, pp_size: int) -> str:
    """Build hybrid_layer_pattern with '|' pipeline stage boundaries."""
    layers_per_stage = num_layers // pp_size
    unit_len = len(base)
    repeats_per_stage = layers_per_stage // unit_len
    stage = base * repeats_per_stage
    return "|".join([stage] * pp_size)


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
    "src_tp,src_pp,src_ep,dst_tp,dst_pp,dst_ep,num_experts,moe_mode",
    [
        # ---- Non-MoE: TP only changes ----
        (2, 1, 1, 1, 1, 1, None, None),  # TP2 -> TP1
        (1, 1, 1, 2, 1, 1, None, None),  # TP1 -> TP2
        (2, 1, 1, 4, 1, 1, None, None),  # TP2 -> TP4
        # ---- Non-MoE: PP only changes ----
        (1, 2, 1, 1, 1, 1, None, None),  # PP2 -> PP1
        (1, 1, 1, 1, 2, 1, None, None),  # PP1 -> PP2
        # ---- Non-MoE: Both TP and PP change ----
        (2, 2, 1, 1, 1, 1, None, None),  # TP2,PP2 -> TP1,PP1
        (1, 1, 1, 2, 2, 1, None, None),  # TP1,PP1 -> TP2,PP2
        (2, 1, 1, 1, 2, 1, None, None),  # TP2,PP1 -> TP1,PP2
        (1, 2, 1, 2, 1, 1, None, None),  # TP1,PP2 -> TP2,PP1
        (1, 2, 1, 2, 4, 1, None, None),  # TP1,PP2 -> TP2,PP4
        # ---- MoE: EP changes (standard) ----
        (1, 1, 2, 1, 1, 4, 4, None),  # EP2 -> EP4
        (1, 1, 2, 1, 1, 1, 4, None),  # EP2 -> EP1
        (1, 1, 1, 1, 1, 2, 4, None),  # EP1 -> EP2
        (1, 1, 2, 1, 2, 2, 4, None),  # EP2 -> PP2,EP2
        # ---- MoE: mixed TP + EP (standard) ----
        (2, 1, 2, 1, 1, 1, 4, None),  # TP2,EP2 -> TP1,EP1
        (1, 1, 1, 2, 1, 2, 4, None),  # TP1,EP1 -> TP2,EP2
        (4, 1, 1, 2, 1, 2, 4, None),  # TP4,EP1 -> TP2,EP2
        (2, 1, 2, 4, 1, 1, 4, None),  # TP2,EP2 -> TP4,EP1
        (4, 1, 1, 1, 1, 4, 4, None),  # TP4,EP1 -> TP1,EP4
        (1, 1, 4, 4, 1, 1, 4, None),  # EP4 -> TP4,EP1
        # ---- MoE latent: representative configs ----
        (1, 1, 2, 1, 1, 1, 4, "latent"),  # EP2 -> EP1
        (2, 1, 2, 1, 1, 1, 4, "latent"),  # TP2,EP2 -> TP1,EP1
        (1, 1, 1, 2, 1, 2, 4, "latent"),  # TP1,EP1 -> TP2,EP2
        # ---- MoE latent + MTP: representative configs ----
        (1, 1, 1, 1, 1, 2, 4, "latent_mtp"),  # EP1 -> EP2
        (2, 1, 2, 1, 1, 1, 4, "latent_mtp"),  # TP2,EP2 -> TP1,EP1
        (1, 1, 1, 2, 1, 2, 4, "latent_mtp"),  # TP1,EP1 -> TP2,EP2
    ],
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

    # Small GPT config
    seq_len = 8
    vocab_size = 128
    # --group-query-attention   --num-query-groups 8
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

    # Build PGs and models (always use unified PG builder so we can set EP)
    src_pgs = _build_pg_collection(tp_size=src_tp, pp_size=src_pp, ep_size=src_ep)
    dst_pgs = _build_pg_collection(tp_size=dst_tp, pp_size=dst_pp, ep_size=dst_ep)
    # Apply PP/EP configuration to TransformerConfigs
    src_cfg = copy.deepcopy(cfg)
    dst_cfg = copy.deepcopy(cfg)
    src_cfg.pipeline_model_parallel_size = src_pp
    dst_cfg.pipeline_model_parallel_size = dst_pp

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
            src_cfg,
            vocab_size,
            seq_len,
            src_pgs,
            parallel_output=False,
            num_moe_experts=num_experts,
        )
        .to(device)
        .eval()
    )
    dst_model = (
        _build_gpt(
            dst_cfg,
            vocab_size,
            seq_len,
            dst_pgs,
            parallel_output=False,
            num_moe_experts=num_experts,
        )
        .to(device)
        .eval()
    )

    # Inputs
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
        src_out = _run_forward(src_model, tokens, position_ids, attention_mask, src_pgs)
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
        dst_out = _run_forward(dst_model, tokens, position_ids, attention_mask, dst_pgs)
        if dist.get_rank() == dst_last_pp_rank:
            dst_logits.copy_(dst_out)
    dist.broadcast(dst_logits, src=dst_last_pp_rank, group=dst_pgs.pp)

    # Compare
    assert ref_logits.shape == dst_logits.shape
    max_diff = (dst_logits - ref_logits).abs().max().item()
    assert torch.allclose(dst_logits, ref_logits, atol=5e-4, rtol=5e-4), (
        f"Refit src(TP={src_tp},PP={src_pp},EP={src_ep})"
        f"->dst(TP={dst_tp},PP={dst_pp},EP={dst_ep}) "
        f"moe_mode={moe_mode} outputs differ (max_diff={max_diff:.6f})"
    )
    dist.barrier()

    # Free GPU memory to prevent OOM across the many parametrized test cases
    del src_model, dst_model
    # Clear refit caches before destroying model parallel to avoid stale plans
    clear_all_caches()
    _destroy_pg_collection(src_pgs)
    _destroy_pg_collection(dst_pgs)
    Utils.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


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
    "src_tp,src_ep,dst_tp,dst_ep",
    [
        (2, 2, 1, 1),  # TP2,EP2 -> TP1,EP1 (cross-cluster shape)
        (1, 1, 2, 2),  # TP1,EP1 -> TP2,EP2 (reverse)
        (1, 2, 2, 2),  # TP=1->TP=2 with EP unchanged
    ],
)
def test_router_expert_bias_refit(
    refit_backend: str, src_tp: int, src_ep: int, dst_tp: int, dst_ep: int
):
    """Regression test: MoE router ``expert_bias`` (a *persistent buffer*, not a
    Parameter) must travel with weights during refit/resharding.

    This was the root cause of stale routing on the inference model when
    refit was used to re-shard a Nemotron-style MoE+Mamba checkpoint across
    different TP/EP layouts: the router buffer carried aux-loss-free load
    balancing state on the trainer but stayed at zero on the inference model
    because ``swap_model_weights`` only enumerated ``named_parameters``.
    """
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=src_tp, pipeline_model_parallel_size=1
    )
    world = dist.get_world_size()
    if (world % (src_tp * src_ep) != 0) or (world % (dst_tp * dst_ep) != 0):
        Utils.destroy_model_parallel()
        pytest.skip("WORLD_SIZE must be divisible by both src_tp*src_ep and dst_tp*dst_ep")

    try:
        import transformer_engine
    except Exception:
        Utils.destroy_model_parallel()
        pytest.skip("Transformer Engine not available")

    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    num_experts = 4
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=8,
        num_query_groups=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=64,
        moe_grouped_gemm=True,
        add_bias_linear=False,
        moe_router_dtype="fp64",
        moe_token_dispatcher_type="alltoall",
        # The flag this regression covers: routers register a persistent
        # ``expert_bias`` buffer used for aux-loss-free load balancing.
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
    )
    src_cfg = copy.deepcopy(cfg)
    dst_cfg = copy.deepcopy(cfg)
    src_cfg.expert_model_parallel_size = src_ep
    dst_cfg.expert_model_parallel_size = dst_ep

    src_pgs = _build_pg_collection(tp_size=src_tp, pp_size=1, ep_size=src_ep)
    dst_pgs = _build_pg_collection(tp_size=dst_tp, pp_size=1, ep_size=dst_ep)

    src_model = (
        _build_gpt(
            src_cfg,
            vocab_size=128,
            seq_len=8,
            pg_collection=src_pgs,
            parallel_output=False,
            num_moe_experts=num_experts,
        )
        .to(device)
        .eval()
    )
    dst_model = (
        _build_gpt(
            dst_cfg,
            vocab_size=128,
            seq_len=8,
            pg_collection=dst_pgs,
            parallel_output=False,
            num_moe_experts=num_experts,
        )
        .to(device)
        .eval()
    )

    # Stamp a recognizable pattern into every router.expert_bias on src,
    # promoting it to fp32 to mirror what _maintain_float32_expert_bias does on
    # the trainer's first forward.  dst is left at its bf16/init state so the
    # refit must transfer the value AND harmonize the dtype.
    test_pattern = torch.arange(num_experts, dtype=torch.float32, device=device) + 0.25
    src_buffers: dict[str, torch.Tensor] = {}
    for name, mod in src_model.named_modules():
        bias = getattr(mod, "expert_bias", None)
        if isinstance(bias, torch.Tensor):
            with torch.no_grad():
                if bias.dtype != torch.float32:
                    fp32_bias = bias.detach().to(torch.float32)
                    fp32_bias.copy_(test_pattern)
                    mod._buffers["expert_bias"] = fp32_bias
                else:
                    bias.copy_(test_pattern)
            src_buffers[f"{name}.expert_bias"] = mod._buffers["expert_bias"]

    # Sanity: dst's buffers should NOT yet match src (they're zero-init).
    pre_swap_match = all(
        torch.allclose(
            dict(dst_model.named_buffers()).get(n, torch.zeros_like(b)).float(),
            b.float(),
            atol=1e-5,
        )
        for n, b in src_buffers.items()
    )
    assert not pre_swap_match, "test setup wrong: dst already matches src before refit"

    swap_model_weights([src_model], [dst_model], refit_method=refit_backend)
    torch.cuda.synchronize()

    # Verify each router.expert_bias on dst now matches src's stamped pattern.
    dst_named_buffers = dict(dst_model.named_buffers())
    mismatches = []
    for name, src_buf in src_buffers.items():
        dst_buf = dst_named_buffers.get(name)
        assert dst_buf is not None, f"dst missing buffer {name}"
        if not torch.allclose(dst_buf.float(), src_buf.float(), atol=1e-5):
            mismatches.append((name, (dst_buf - src_buf).abs().max().item()))
    assert not mismatches, (
        f"router.expert_bias not transferred during refit "
        f"(src_tp={src_tp}, src_ep={src_ep} -> dst_tp={dst_tp}, dst_ep={dst_ep}, "
        f"backend={refit_backend}): {mismatches}"
    )
    dist.barrier()

    del src_model, dst_model
    clear_all_caches()
    _destroy_pg_collection(src_pgs)
    _destroy_pg_collection(dst_pgs)
    Utils.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


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
def test_router_expert_bias_refit_non_collocated(refit_backend: str):
    """Non-collocated counterpart of ``test_router_expert_bias_refit``.

    Splits the world into disjoint src and dst rank sets so dst-only ranks
    have no local view of the src model.  Exercises the ``all_gather_object``-
    based dtype harmonization path: src ranks hold ``expert_bias`` in fp32 and
    dst ranks in bf16, and the only way dst can learn the expected dtype is
    via the gathered map.
    """
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    world = dist.get_world_size()
    src_tp, src_ep, dst_tp, dst_ep = 2, 1, 2, 1
    src_world = src_tp * src_ep
    dst_world = dst_tp * dst_ep
    if world < src_world + dst_world:
        Utils.destroy_model_parallel()
        pytest.skip(f"Non-collocated test requires WORLD_SIZE >= {src_world + dst_world}")

    try:
        import transformer_engine  # noqa: F401
    except Exception:
        Utils.destroy_model_parallel()
        pytest.skip("Transformer Engine not available")

    from megatron.rl.parallel_utils import build_inference_pg_collection

    rank = dist.get_rank()
    is_src = rank < src_world
    is_dst = src_world <= rank < src_world + dst_world

    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    num_experts = 4
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=8,
        num_query_groups=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=64,
        moe_grouped_gemm=True,
        add_bias_linear=False,
        moe_router_dtype="fp64",
        moe_token_dispatcher_type="alltoall",
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
    )
    src_cfg = copy.deepcopy(cfg)
    dst_cfg = copy.deepcopy(cfg)
    src_cfg.expert_model_parallel_size = src_ep
    dst_cfg.expert_model_parallel_size = dst_ep

    # Both pg collections are built collectively on every rank (dist.new_group
    # requires it) but each one's groups only contain the ranks for that side.
    src_pgs = build_inference_pg_collection(
        world_size=src_world, tp_size=src_tp, ep_size=src_ep, rank_offset=0
    )
    dst_pgs = build_inference_pg_collection(
        world_size=dst_world, tp_size=dst_tp, ep_size=dst_ep, rank_offset=src_world
    )

    src_model = None
    dst_model = None
    if is_src:
        src_model = (
            _build_gpt(
                src_cfg,
                vocab_size=128,
                seq_len=8,
                pg_collection=src_pgs,
                parallel_output=False,
                num_moe_experts=num_experts,
            )
            .to(device)
            .eval()
        )
    elif is_dst:
        dst_model = (
            _build_gpt(
                dst_cfg,
                vocab_size=128,
                seq_len=8,
                pg_collection=dst_pgs,
                parallel_output=False,
                num_moe_experts=num_experts,
            )
            .to(device)
            .eval()
        )

    test_pattern = torch.arange(num_experts, dtype=torch.float32, device=device) + 0.25
    if is_src and src_model is not None:
        for name, mod in src_model.named_modules():
            bias = getattr(mod, "expert_bias", None)
            if isinstance(bias, torch.Tensor):
                with torch.no_grad():
                    # Promote to fp32 to mirror what _maintain_float32_expert_bias
                    # does on the trainer's first forward, while dst remains at
                    # its bf16/fp32-from-init state.  This forces the dtype
                    # harmonization path to do work for non-collocated transfer
                    # (dst-only ranks have no local view of src's dtype).
                    if bias.dtype != torch.float32:
                        fp32_bias = bias.detach().to(torch.float32)
                        fp32_bias.copy_(test_pattern)
                        mod._buffers["expert_bias"] = fp32_bias
                    else:
                        bias.copy_(test_pattern)

    dist.barrier()

    swap_model_weights(
        [src_model] if src_model is not None else None,
        [dst_model] if dst_model is not None else None,
        refit_method=refit_backend,
    )
    torch.cuda.synchronize()
    dist.barrier()

    if is_dst and dst_model is not None:
        dst_named_buffers = dict(dst_model.named_buffers())
        mismatches = []
        for name, dst_buf in dst_named_buffers.items():
            if not name.endswith("expert_bias"):
                continue
            # Replicated buffer: expected value is the stamped test_pattern.
            # dst_buf should also be fp32 thanks to dtype harmonization.
            if dst_buf.dtype != torch.float32:
                mismatches.append((name, f"dtype not harmonized: {dst_buf.dtype}"))
                continue
            if not torch.allclose(dst_buf, test_pattern, atol=1e-5):
                mismatches.append((name, (dst_buf - test_pattern).abs().max().item()))
        assert not mismatches, (
            f"Non-collocated refit did not transfer router.expert_bias correctly "
            f"(backend={refit_backend}): {mismatches}"
        )

    dist.barrier()
    if src_model is not None:
        del src_model
    if dst_model is not None:
        del dst_model
    clear_all_caches()
    _destroy_pg_collection(src_pgs)
    _destroy_pg_collection(dst_pgs)
    Utils.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


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
    "src_tp,src_pp,dst_tp,dst_pp",
    [
        # TP only changes (exercises block-interleaved planner for Mamba in_proj)
        (2, 1, 1, 1),  # TP2 -> TP1
        (1, 1, 2, 1),  # TP1 -> TP2
        (2, 1, 4, 1),  # TP2 -> TP4
        # TP + PP change together
        (1, 1, 2, 2),  # TP1,PP1 -> TP2,PP2
        (2, 1, 1, 2),  # TP2,PP1 -> TP1,PP2
    ],
)
def test_swap_mamba_parametrized(
    refit_backend: str, src_tp: int, src_pp: int, dst_tp: int, dst_pp: int
):
    if not has_mamba_deps:
        pytest.skip("Mamba dependencies (mamba_ssm, einops) not available")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=src_tp, pipeline_model_parallel_size=src_pp
    )
    world = dist.get_world_size()
    if (world % (src_tp * src_pp) != 0) or (world % (dst_tp * dst_pp) != 0):
        Utils.destroy_model_parallel()
        pytest.skip("WORLD_SIZE must be divisible by both src_tp*src_pp and dst_tp*dst_pp")

    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Small Mamba config — use "M*" hybrid pattern to test both Mamba layers
    # (block-interleaved in_proj resharding) and attention layers together.
    seq_len = 8
    vocab_size = 128
    base_pattern = "M*"
    # Ensure enough layers for both PP configs (at least len(base_pattern) per stage)
    min_layers = max(src_pp, dst_pp) * len(base_pattern)
    num_layers = max(min_layers, 4 if (src_pp > 1 or dst_pp > 1) else 2)
    # Round up to be divisible by both pp_size * unit_len
    from math import lcm

    factor = lcm(src_pp, dst_pp) * len(base_pattern)
    num_layers = ((num_layers + factor - 1) // factor) * factor

    cfg = TransformerConfig(
        num_layers=num_layers,
        hidden_size=256,
        num_attention_heads=8,
        num_query_groups=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    src_pgs = _build_pg_collection(tp_size=src_tp, pp_size=src_pp)
    dst_pgs = _build_pg_collection(tp_size=dst_tp, pp_size=dst_pp)

    src_pattern = _mamba_layer_pattern(base_pattern, num_layers, src_pp)
    dst_pattern = _mamba_layer_pattern(base_pattern, num_layers, dst_pp)

    src_model = (
        _build_mamba(cfg, vocab_size, seq_len, src_pgs, src_pattern, parallel_output=False)
        .to(device)
        .eval()
    )
    dst_model = (
        _build_mamba(cfg, vocab_size, seq_len, dst_pgs, dst_pattern, parallel_output=False)
        .to(device)
        .eval()
    )

    # Inputs
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
        src_out = _run_forward(src_model, tokens, position_ids, attention_mask, src_pgs)
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
        dst_out = _run_forward(dst_model, tokens, position_ids, attention_mask, dst_pgs)
        if dist.get_rank() == dst_last_pp_rank:
            dst_logits.copy_(dst_out)
    dist.broadcast(dst_logits, src=dst_last_pp_rank, group=dst_pgs.pp)

    # Compare
    assert ref_logits.shape == dst_logits.shape
    max_diff = (dst_logits - ref_logits).abs().max().item()
    assert torch.allclose(dst_logits, ref_logits, atol=1e-3, rtol=1e-3), (
        f"Mamba refit src(TP={src_tp},PP={src_pp})"
        f"->dst(TP={dst_tp},PP={dst_pp}) "
        f"outputs differ (max_diff={max_diff:.6f})"
    )
    dist.barrier()

    # Free GPU memory to prevent OOM across the many parametrized test cases
    del src_model, dst_model
    clear_all_caches()
    _destroy_pg_collection(src_pgs)
    _destroy_pg_collection(dst_pgs)
    Utils.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
