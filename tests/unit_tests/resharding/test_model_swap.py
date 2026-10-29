# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
import copy
import gc
import importlib.util
import types
from dataclasses import fields
from functools import partial
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
from megatron.core.resharding.refit import (
    clear_all_caches,
    prepare_swap_model_weights,
    swap_model_weights,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    import nvshmem.core

    has_nvshmem = True
except Exception:
    has_nvshmem = False

try:
    has_nccl_m2n = (
        importlib.util.find_spec("nccl.core") is not None
        and importlib.util.find_spec("nccl.m2n") is not None
    )
except (ImportError, ModuleNotFoundError):
    has_nccl_m2n = False

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
    transformer_impl: str = "transformer_engine",
    share_embeddings_and_output_weights: bool = False,
) -> GPTModel:
    spec_builder = (
        get_gpt_layer_local_spec
        if transformer_impl == "local"
        else get_gpt_layer_with_transformer_engine_spec
    )
    layer_spec = spec_builder(
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
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        position_embedding_type="rope",
        rotary_percent=1.0,
        pg_collection=pg_collection,
        mtp_block_spec=mtp_block_spec,
    )
    return model


def _build_hybrid(
    config: TransformerConfig,
    vocab_size: int,
    seq_len: int,
    pg_collection,
    hybrid_layer_pattern: str,
    parallel_output: bool = True,
    stack_spec=None,
):
    pre_process, post_process = _pp_flags(pg_collection)
    if stack_spec is None:
        stack_spec = hybrid_stack_spec
    model = HybridModel(
        config=config,
        hybrid_stack_spec=stack_spec,
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


def _hybrid_layer_pattern(base: str, num_layers: int, pp_size: int) -> str:
    """Build hybrid_layer_pattern with '|' pipeline stage boundaries."""
    layers_per_stage = num_layers // pp_size
    unit_len = len(base)
    repeats_per_stage = layers_per_stage // unit_len
    stage = base * repeats_per_stage
    return "|".join([stage] * pp_size)


class _DSGQACoreAttention(DSAttention):
    """Wire the production DSA core to explicit GQA query, key, and value tensors."""

    def set_hidden_states(self, hidden_states: torch.Tensor) -> None:
        self._dsa_hidden_states = hidden_states

    def clear_hidden_states(self) -> None:
        self._dsa_hidden_states = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        hidden_states = getattr(self, "_dsa_hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("DSA-GQA core attention requires the self-attention input")
        return super().forward(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            x=hidden_states,
            qr=hidden_states,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )


class _DSGroupedSelfAttention(SelfAttention):
    """Self-attention layer that executes the production DSA core on GQA tensors."""

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        self.core_attention.set_hidden_states(hidden_states)
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self.core_attention.clear_hidden_states()


def _dsa_gqa_stack_spec():
    """Return a hybrid stack spec with a non-absorbed, single-group GQA DSA layer."""
    stack_spec = copy.deepcopy(hybrid_stack_spec)
    dsa_layer_spec = stack_spec.submodules.dsa_layer
    mla_attention_submodules = dsa_layer_spec.submodules.self_attention.submodules
    dsa_submodules = DSAttentionSubmodules(
        indexer=copy.deepcopy(mla_attention_submodules.core_attention.submodules.indexer)
    )
    dsa_layer_spec.submodules.self_attention = ModuleSpec(
        module=_DSGroupedSelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=mla_attention_submodules.linear_q_proj,
            core_attention=partial(_DSGQACoreAttention, submodules=dsa_submodules),
            linear_proj=mla_attention_submodules.linear_proj,
        ),
    )
    return stack_spec


def _dsa_config(num_layers: int, tp_size: int, pp_size: int, *, gqa: bool):
    """Build a compact config for absorbed MLA DSA or explicit single-group GQA DSA."""
    hidden_size = 128
    return MLATransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        num_query_groups=1 if gqa else 8,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        sequence_parallel=not gqa and tp_size > 1,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        params_dtype=torch.float32,
        add_bias_linear=False,
        normalization="RMSNorm",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        multi_latent_attention=not gqa,
        q_lora_rank=hidden_size if gqa else 64,
        kv_lora_rank=32,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        rope_type="rope",
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=16,
        dsa_indexer_topk=4,
        dsa_indexer_rotate_activation=False,
        dsa_kernel_backend="none",
    )


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


@pytest.mark.parametrize("src_tp,precision", [(1, "bf16"), (2, "bf16"), (1, "fp8")])
def test_repeated_refit_matches_fresh_models_and_reuses_plan(src_tp: int, precision: str):
    """A->B->C->A refits execute new weights without moving target storage."""
    Utils.initialize_distributed()
    world = dist.get_world_size()
    if world % src_tp:
        pytest.skip("WORLD_SIZE must be divisible by src_tp")
    Utils.initialize_model_parallel(tensor_model_parallel_size=src_tp)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    seq_len, vocab_size = 8, 64
    config = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=2,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp8="hybrid" if precision == "fp8" else None,
        fp8_recipe="tensorwise" if precision == "fp8" else None,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    src_pgs = _build_pg_collection(tp_size=src_tp)
    dst_pgs = _build_pg_collection(tp_size=1)

    def build(seed, pg_collection):
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(
            seed,
            tp_rank=dist.get_rank(pg_collection.tp),
            ep_rank=dist.get_rank(pg_collection.ep),
            etp_rank=dist.get_rank(pg_collection.expt_tp),
        )
        return (
            _build_gpt(config, vocab_size, seq_len, pg_collection, parallel_output=False)
            .to(device)
            .eval()
        )

    sources = [build(seed, src_pgs) for seed in (101, 202, 303)]
    fresh_targets = [build(seed, dst_pgs) for seed in (101, 202, 303)]
    target = build(404, dst_pgs)
    fp8_activation = []
    handles = []
    if precision == "fp8":
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        def record_fp8_activation(_module, _inputs):
            fp8_activation.append(FP8GlobalStateManager.is_fp8_enabled())

        for module in target.modules():
            if not any(module.children()):
                handles.append(module.register_forward_pre_hook(record_fp8_activation))
    tokens = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    positions = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.triu(
        torch.ones((1, 1, seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1
    )

    def logits(model, pg_collection):
        with torch.no_grad():
            output = _run_forward(model, tokens, positions, attention_mask, pg_collection)
        assert output is not None
        return output.detach().clone()

    source_outputs = [logits(model, src_pgs) for model in sources]
    references = [logits(model, dst_pgs) for model in fresh_targets]
    initial = logits(target, dst_pgs)
    # TP2/TP1 variation measured one BF16 step; parameters are also compared exactly below.
    cross_topology_tolerance = 2e-3 if src_tp > 1 else 5e-4
    for source_output, reference in zip(source_outputs, references):
        assert torch.equal(source_output.argmax(dim=-1), reference.argmax(dim=-1))
        assert torch.allclose(
            source_output, reference, atol=cross_topology_tolerance, rtol=cross_topology_tolerance
        )
    if precision == "fp8":
        assert any(fp8_activation), "No module executed inside Transformer Engine FP8 autocast"
    for left, right in zip(references, references[1:]):
        assert not torch.allclose(left, right, atol=1e-3, rtol=1e-3)
    assert all(not torch.allclose(initial, ref, atol=1e-3, rtol=1e-3) for ref in references)

    target_layout = {
        name: (tuple(param.shape), param.dtype, tuple(param.stride()), param.data_ptr())
        for name, param in target.named_parameters()
    }
    clear_all_caches()
    prepare_swap_model_weights([sources[0]], [target])
    from megatron.core.resharding import refit as refit_module

    assert len(refit_module._plan_cache) == 1
    prepared_plan = next(iter(refit_module._plan_cache.values()))
    observed = []
    for source_index in (0, 1, 2, 0):
        fp8_activation.clear()
        swap_model_weights([sources[source_index]], [target], refit_method="nccl")
        torch.cuda.synchronize()
        actual = logits(target, dst_pgs)
        if precision == "fp8":
            assert any(fp8_activation), "The post-refit target step did not execute FP8"
        expected = references[source_index]
        observed.append(actual)
        assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
        assert torch.allclose(actual, expected, atol=5e-4, rtol=5e-4)
        assert next(iter(refit_module._plan_cache.values())) is prepared_plan
        assert {
            name: (tuple(param.shape), param.dtype, tuple(param.stride()), param.data_ptr())
            for name, param in target.named_parameters()
        } == target_layout
        reference_params = dict(fresh_targets[source_index].named_parameters())
        assert all(
            torch.equal(param, reference_params[name]) for name, param in target.named_parameters()
        )

    assert torch.equal(observed[0].argmax(dim=-1), observed[-1].argmax(dim=-1))
    assert not torch.allclose(observed[0], observed[1], atol=1e-3, rtol=1e-3)
    assert not torch.allclose(observed[1], observed[2], atol=1e-3, rtol=1e-3)
    for handle in handles:
        handle.remove()

    dist.barrier()
    del sources, fresh_targets, target
    clear_all_caches()
    _destroy_pg_collection(src_pgs)
    _destroy_pg_collection(dst_pgs)
    Utils.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


def test_refit_reshards_expert_tensor_parallel_weights_and_executes_them():
    """ETP2 expert shards become an executing ETP1 model matching a fresh load."""
    Utils.initialize_distributed()
    world = dist.get_world_size()
    if world % 2:
        pytest.skip("ETP2 refit requires an even WORLD_SIZE")
    try:
        import transformer_engine  # noqa: F401
    except Exception:
        pytest.skip("Transformer Engine is required for grouped MoE experts")

    from megatron.core.inference.shards import build_inference_pg_collection

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_tensor_parallel_size=2)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    source_pgs = build_inference_pg_collection(
        world_size=world, tp_size=1, ep_size=1, expt_tp_size=2
    )
    reference_pgs = build_inference_pg_collection(
        world_size=world, tp_size=1, ep_size=1, expt_tp_size=1
    )
    target_pgs = build_inference_pg_collection(
        world_size=world, tp_size=1, ep_size=1, expt_tp_size=1
    )

    def build(seed, pg_collection, expert_tp):
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(
            seed, tp_rank=0, ep_rank=0, etp_rank=dist.get_rank(pg_collection.expt_tp)
        )
        config = TransformerConfig(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=2,
            num_moe_experts=4,
            moe_ffn_hidden_size=64,
            moe_grouped_gemm=True,
            moe_router_dtype="fp64",
            moe_token_dispatcher_type="alltoall",
            add_bias_linear=False,
            expert_tensor_parallel_size=expert_tp,
            use_cpu_initialization=True,
            pipeline_dtype=torch.bfloat16,
            params_dtype=torch.bfloat16,
            bf16=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        return (
            _build_gpt(
                config,
                vocab_size=64,
                seq_len=8,
                pg_collection=pg_collection,
                parallel_output=False,
                num_moe_experts=4,
            )
            .to(device)
            .eval()
        )

    reference = build(1701, reference_pgs, 1)
    source = build(1701, source_pgs, 2)
    target = build(1802, target_pgs, 1)
    source_expert_pointers = {
        name: param.data_ptr() for name, param in source.named_parameters() if ".experts." in name
    }
    clear_all_caches()
    swap_model_weights([reference], [source], refit_method="nccl")
    torch.cuda.synchronize()
    assert source_expert_pointers == {
        name: param.data_ptr() for name, param in source.named_parameters() if ".experts." in name
    }
    clear_all_caches()
    tokens = torch.arange(8, device=device, dtype=torch.long).unsqueeze(0)
    positions = torch.arange(8, device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.triu(
        torch.ones((1, 1, 8, 8), device=device, dtype=torch.bool), diagonal=1
    )
    expert_steps = {"source": 0, "reference": 0, "target": 0}
    handles = []
    for label, model in (("source", source), ("reference", reference), ("target", target)):
        for module in model.modules():
            if hasattr(module, "experts") and hasattr(module, "router"):
                handles.append(
                    module.register_forward_hook(
                        lambda _module, _inputs, _output, key=label: expert_steps.__setitem__(
                            key, expert_steps[key] + 1
                        )
                    )
                )

    def run(model, pg_collection):
        with torch.no_grad():
            return (
                _run_forward(model, tokens, positions, attention_mask, pg_collection)
                .detach()
                .clone()
            )

    source_output = run(source, source_pgs)
    expected = run(reference, reference_pgs)
    initial = run(target, target_pgs)
    assert expert_steps["source"] and expert_steps["reference"] and expert_steps["target"]
    # ETP2 grouped GEMM measured one BF16 quantum; retain exact greedy identity.
    assert torch.equal(source_output.argmax(dim=-1), expected.argmax(dim=-1))
    assert torch.allclose(source_output, expected, atol=2e-3, rtol=2e-3)
    assert not torch.allclose(initial, expected, atol=1e-3, rtol=1e-3)

    target_expert_pointers = {
        name: param.data_ptr() for name, param in target.named_parameters() if ".experts." in name
    }
    swap_model_weights([source], [target], refit_method="nccl")
    torch.cuda.synchronize()
    refitted = run(target, target_pgs)
    assert expert_steps["target"] >= 2, "The post-refit step did not execute the experts"
    assert torch.equal(refitted.argmax(dim=-1), expected.argmax(dim=-1))
    assert torch.allclose(refitted, expected, atol=5e-4, rtol=5e-4)
    reference_params = dict(reference.named_parameters())
    assert all(
        torch.equal(param, reference_params[name]) for name, param in target.named_parameters()
    )
    assert target_expert_pointers == {
        name: param.data_ptr() for name, param in target.named_parameters() if ".experts." in name
    }

    for handle in handles:
        handle.remove()
    dist.barrier()
    del source, reference, target
    clear_all_caches()
    for pg_collection in (source_pgs, reference_pgs, target_pgs):
        _destroy_pg_collection(pg_collection)
    Utils.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


def test_tied_embedding_refit_preserves_alias_and_executes_new_weight():
    """A tied input/output weight remains one destination object after a real refit."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    source_pgs = _build_pg_collection(tp_size=1)
    target_pgs = _build_pg_collection(tp_size=1)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=2,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        params_dtype=torch.bfloat16,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    def build(seed, pg_collection):
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        return (
            _build_gpt(
                config,
                vocab_size=64,
                seq_len=8,
                pg_collection=pg_collection,
                parallel_output=False,
                share_embeddings_and_output_weights=True,
            )
            .to(device)
            .eval()
        )

    source, target = build(1201, source_pgs), build(1302, target_pgs)
    source_weight = source.shared_embedding_or_output_weight()
    target_weight = target.shared_embedding_or_output_weight()
    assert source_weight is source.embedding.word_embeddings.weight
    assert target_weight is target.embedding.word_embeddings.weight
    assert target.output_layer.weight is None
    target_pointer = target_weight.data_ptr()
    tokens = torch.arange(8, device=device, dtype=torch.long).unsqueeze(0)
    positions = torch.arange(8, device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.triu(
        torch.ones((1, 1, 8, 8), device=device, dtype=torch.bool), diagonal=1
    )
    with torch.no_grad():
        expected = _run_forward(source, tokens, positions, attention_mask, source_pgs).clone()
        initial = _run_forward(target, tokens, positions, attention_mask, target_pgs).clone()
    assert not torch.allclose(initial, expected, atol=1e-3, rtol=1e-3)

    swap_model_weights([source], [target], refit_method="nccl")
    torch.cuda.synchronize()
    with torch.no_grad():
        actual = _run_forward(target, tokens, positions, attention_mask, target_pgs)
    assert target.shared_embedding_or_output_weight() is target.embedding.word_embeddings.weight
    assert target.shared_embedding_or_output_weight().data_ptr() == target_pointer
    assert torch.equal(target.shared_embedding_or_output_weight(), source_weight)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)

    dist.barrier()
    del source, target
    clear_all_caches()
    _destroy_pg_collection(source_pgs)
    _destroy_pg_collection(target_pgs)
    Utils.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("refit_backend", ["nccl", "nixl"])
def test_gtp_shards_execute_real_transfer_and_match_fresh_oracle(refit_backend: str):
    """GTP shards from every rank reconstruct a full destination used by inference."""
    if refit_backend == "nixl":
        pytest.importorskip("nixl._api", reason="NIXL is not installed in this runtime")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2:
        Utils.destroy_model_parallel()
        pytest.skip("GTP transfer requires at least two ranks")
    if refit_backend == "nixl" and world > 2:
        # Each rank sends its one shard to several peers. NixlCopyService registers that
        # region once per send, and peers then fail to load the metadata (NOT_ALLOWED).
        Utils.destroy_model_parallel()
        pytest.skip("NixlCopyService registers a region sent to several peers more than once")

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    src_pgs = _build_pg_collection(tp_size=1)
    dst_pgs = _build_pg_collection(tp_size=1)
    gtp_group = dist.new_group(ranks=list(range(world)))
    src_pgs.gtp_remat = gtp_group

    class TinyWeightModel(torch.nn.Module):
        def __init__(self, weight, pg_collection):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)
            self.pg_collection = pg_collection
            self.config = types.SimpleNamespace(num_moe_experts=None)

        def forward(self, inputs):
            return inputs @ self.weight.t()

    rows_per_rank, columns = 2, 4
    full_weight = torch.arange(
        world * rows_per_rank * columns, device=device, dtype=torch.float32
    ).reshape(world * rows_per_rank, columns)
    source = TinyWeightModel(
        full_weight[rank * rows_per_rank : (rank + 1) * rows_per_rank].clone(), src_pgs
    )
    source.weight.is_gtp_weight_remat = True
    source.weight.group = gtp_group
    source.weight.pad_length = 0
    target = TinyWeightModel(torch.full_like(full_weight, -17), dst_pgs)
    oracle = TinyWeightModel(full_weight.clone(), dst_pgs)
    inputs = torch.arange(3 * columns, device=device, dtype=torch.float32).reshape(3, columns)
    expected = oracle(inputs)
    initial = target(inputs)
    target_ptr = target.weight.data_ptr()
    assert source.weight.shape != target.weight.shape
    assert not torch.equal(initial, expected)

    clear_all_caches()
    prepare_swap_model_weights([source], [target])
    from megatron.core.resharding import refit as refit_module

    plan = next(iter(refit_module._plan_cache.values()))
    assert {op.peer_rank for op in plan.recv_ops} == set(range(world))
    assert len(plan.recv_ops) == world
    swap_model_weights([source], [target], refit_method=refit_backend)
    torch.cuda.synchronize()

    assert target.weight.data_ptr() == target_ptr
    assert torch.equal(target.weight, oracle.weight)
    assert torch.equal(target(inputs), expected)

    dist.barrier()
    del source, target, oracle
    clear_all_caches()
    _destroy_pg_collection(src_pgs)
    _destroy_pg_collection(dst_pgs)
    Utils.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


def test_refit_executes_context_parallel_collectives_with_new_weights(monkeypatch):
    Utils.initialize_distributed()
    world = dist.get_world_size()
    if world < 2 or world % 2:
        pytest.skip("Mamba context parallelism requires an even world of at least two ranks")
    if not has_mamba_deps:
        pytest.skip("Mamba dependencies are not available")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    from megatron.core.ssm import mamba_context_parallel as mamba_cp

    source_pgs = _build_pg_collection(tp_size=1)
    target_pgs = _build_pg_collection(tp_size=1)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    class TinyContextParallelModel(torch.nn.Module):
        def __init__(self, seed, pg_collection):
            super().__init__()
            torch.manual_seed(seed)
            self.pg_collection = pg_collection
            self.config = types.SimpleNamespace(num_moe_experts=None)
            self.conv_weight = torch.nn.Parameter(torch.randn(16, 1, 3, device=device))
            self.conv_bias = torch.nn.Parameter(torch.randn(16, device=device))
            self.dt_bias = torch.nn.Parameter(torch.randn(4, device=device))
            self.A_log = torch.nn.Parameter(torch.randn(4, device=device))
            self.D = torch.nn.Parameter(torch.randn(8, device=device))

        def forward(self, local_sequence):
            cp = mamba_cp.MambaContextParallel(
                cp_group=self.pg_collection.cp,
                d_inner_local_tp=8,
                nheads_local_tp=4,
                ngroups_local_tp=2,
                d_state=2,
                conv1d_weight_cp1=self.conv_weight,
                conv1d_bias_cp1=self.conv_bias,
                conv1d_padding=2,
                dt_bias_cp1=self.dt_bias,
                A_log_cp1=self.A_log,
                D_cp1=self.D,
                D_has_hdim=True,
            )
            redistributed = cp.pre_conv_ssm(local_sequence)
            head_scale = (cp.get_dt_bias() + cp.get_A_log()).repeat_interleave(2)
            recurrent = cp.post_conv_ssm(redistributed[..., :4] * (cp.get_D() + head_scale))
            convolution = cp.conv1d(redistributed[..., 4:12].permute(1, 2, 0))
            return recurrent, convolution

    source = TinyContextParallelModel(1403, source_pgs)
    target = TinyContextParallelModel(1504, target_pgs)
    cp_rank = target_pgs.cp.rank()
    local_input = (
        torch.arange(2 * 28, device=device, dtype=torch.float32).reshape(2, 1, 28) + 100 * cp_rank
    )
    with torch.no_grad():
        expected = tuple(tensor.clone() for tensor in source(local_input))
        initial = tuple(tensor.clone() for tensor in target(local_input))
    assert any(not torch.equal(before, after) for before, after in zip(initial, expected))
    calls = {"sequence_to_head": 0, "head_to_sequence": 0}
    real_sequence_to_head = mamba_cp.all_to_all_sp2hp
    real_head_to_sequence = mamba_cp.all_to_all_hp2sp

    def observe_sequence_to_head(*args, **kwargs):
        calls["sequence_to_head"] += 1
        return real_sequence_to_head(*args, **kwargs)

    def observe_head_to_sequence(*args, **kwargs):
        calls["head_to_sequence"] += 1
        return real_head_to_sequence(*args, **kwargs)

    monkeypatch.setattr(mamba_cp, "all_to_all_sp2hp", observe_sequence_to_head)
    monkeypatch.setattr(mamba_cp, "all_to_all_hp2sp", observe_head_to_sequence)
    swap_model_weights([source], [target], refit_method="nccl")
    calls.update(sequence_to_head=0, head_to_sequence=0)
    with torch.no_grad():
        actual = target(local_input)

    source_parameters = dict(source.named_parameters())
    assert all(
        torch.equal(parameter, source_parameters[name])
        for name, parameter in target.named_parameters()
    )
    assert calls == {"sequence_to_head": 5, "head_to_sequence": 1}
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)

    dist.barrier()
    del source, target
    clear_all_caches()
    _destroy_pg_collection(source_pgs)
    _destroy_pg_collection(target_pgs)
    Utils.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


def test_noncollocated_multi_destination_pools_receive_and_execute_same_weights():
    """One source refits two disjoint destination pools through real NCCL copies."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 3:
        Utils.destroy_model_parallel()
        pytest.skip("Two destination pools require one source and two destination ranks")

    from megatron.core.inference.shards import build_inference_pg_collection

    source_pgs = build_inference_pg_collection(world_size=1, tp_size=1, rank_offset=0)
    destination_pgs = [
        build_inference_pg_collection(world_size=1, tp_size=1, rank_offset=offset)
        for offset in (1, 2)
    ]
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    class TinyModel(torch.nn.Module):
        def __init__(self, value, pg_collection):
            super().__init__()
            self.weight = torch.nn.Parameter(value)
            self.pg_collection = pg_collection
            self.config = types.SimpleNamespace(num_moe_experts=None)

        def forward(self, inputs):
            return inputs @ self.weight.t()

    expected_weight = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4)
    source = TinyModel(expected_weight.clone(), source_pgs) if rank == 0 else None
    pool_index = rank - 1 if rank in (1, 2) else 0
    target = (
        TinyModel(torch.full_like(expected_weight, -11 - rank), destination_pgs[pool_index])
        if rank in (1, 2)
        else None
    )
    inputs = torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4)
    expected_output = inputs @ expected_weight.t()
    if target is not None:
        assert not torch.equal(target(inputs), expected_output)
        target_ptr = target.weight.data_ptr()

    clear_all_caches()
    model_source = [source] if source is not None else None
    model_target = [target] if target is not None else None
    prepare_swap_model_weights(
        model_source, model_target, num_dst_pools=2, dst_pool_index=pool_index
    )
    swap_model_weights(
        model_source, model_target, refit_method="nccl", num_dst_pools=2, dst_pool_index=pool_index
    )
    torch.cuda.synchronize()

    if target is not None:
        assert target.weight.data_ptr() == target_ptr
        assert torch.equal(target.weight, expected_weight)
        assert torch.equal(target(inputs), expected_output)
    received = [None] * world
    local_received = rank in (1, 2) and torch.equal(target.weight, expected_weight)
    dist.all_gather_object(received, local_received)
    assert received[1:3] == [True, True]

    dist.barrier()
    del source, target
    clear_all_caches()
    _destroy_pg_collection(source_pgs)
    for pg_collection in destination_pgs:
        _destroy_pg_collection(pg_collection)
    Utils.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("variant", ["gdn", "gdp"])
def test_swap_gated_recurrent_variants_matches_fresh_model(variant: str):
    """GDN and GDP weights execute the fresh source model's fixed-history output."""
    if not has_mamba_deps:
        pytest.skip("HybridModel dependencies are not available")

    if variant == "gdn":
        from megatron.core.ssm.gated_delta_net import HAVE_FLA

        if not HAVE_FLA:
            pytest.skip("GDN requires FLA")
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_query_groups=2,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[1, 1],
            gdn_kernel_backend="fla",
            activation_func=torch.nn.functional.silu,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        stack_spec, pattern = hybrid_stack_spec, "G*"
    else:
        from megatron.core.models.hybrid.hybrid_layer_specs import (
            gated_delta_product_inference_stack_spec,
        )
        from megatron.core.ssm.gated_delta_product import HAVE_FLA, HAVE_MAMBA_SSM

        if not (HAVE_FLA and HAVE_MAMBA_SSM):
            pytest.skip("GDP dependencies are not available")
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=4,
            ffn_hidden_size=128,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            mamba_num_heads=16,
            mamba_head_dim=16,
            mamba_num_groups=4,
            mamba_state_dim=16,
            gdp_num_householder=2,
            is_hybrid_model=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        stack_spec, pattern = gated_delta_product_inference_stack_spec, "M*"

    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    source_pgs = _build_pg_collection(tp_size=1)
    target_pgs = _build_pg_collection(tp_size=1)

    def build(seed, pg_collection):
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        return (
            _build_hybrid(
                config,
                vocab_size=64,
                seq_len=8,
                pg_collection=pg_collection,
                hybrid_layer_pattern=pattern,
                parallel_output=False,
                stack_spec=stack_spec,
            )
            .to(device)
            .eval()
        )

    source, target = build(717, source_pgs), build(818, target_pgs)
    tokens = torch.arange(8, device=device, dtype=torch.long).unsqueeze(0)
    positions = torch.arange(8, device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.triu(
        torch.ones((1, 1, 8, 8), device=device, dtype=torch.bool), diagonal=1
    )
    with torch.no_grad():
        expected = _run_forward(source, tokens, positions, attention_mask, source_pgs).clone()
        initial = _run_forward(target, tokens, positions, attention_mask, target_pgs).clone()
    assert not torch.allclose(initial, expected, atol=5e-2, rtol=5e-2)

    swap_model_weights([source], [target], refit_method="nccl")
    torch.cuda.synchronize()
    with torch.no_grad():
        actual = _run_forward(target, tokens, positions, attention_mask, target_pgs)
    source_parameters = dict(source.named_parameters())
    assert all(
        torch.equal(parameter, source_parameters[name])
        for name, parameter in target.named_parameters()
    )
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)

    dist.barrier()
    del source, target
    clear_all_caches()
    _destroy_pg_collection(source_pgs)
    _destroy_pg_collection(target_pgs)
    Utils.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


def test_refit_executes_sliding_window_and_learnable_sink_attention():
    """Refit transfers the sink parameter and preserves the windowed output oracle."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    source_pgs = _build_pg_collection(tp_size=1)
    target_pgs = _build_pg_collection(tp_size=1)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        params_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        window_size=(4, 0),
        softmax_type="learnable",
    )

    def build(seed, pg_collection, model_config=config):
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        return (
            _build_gpt(
                model_config,
                vocab_size=64,
                seq_len=8,
                pg_collection=pg_collection,
                parallel_output=False,
                transformer_impl="local",
            )
            .to(device)
            .eval()
        )

    source, target = build(919, source_pgs), build(1020, target_pgs)
    source_sinks = {
        name: parameter
        for name, parameter in source.named_parameters()
        if name.endswith("softmax_offset")
    }
    target_sinks = {
        name: parameter
        for name, parameter in target.named_parameters()
        if name.endswith("softmax_offset")
    }
    assert source_sinks.keys() == target_sinks.keys() and source_sinks
    with torch.no_grad():
        for index, parameter in enumerate(source_sinks.values()):
            parameter.copy_(torch.linspace(-2 - index, 2 + index, parameter.numel(), device=device))

    full_config = copy.deepcopy(config)
    full_config.window_size = None
    full_attention = build(1121, source_pgs, model_config=full_config)
    full_attention.load_state_dict(source.state_dict())
    tokens = torch.arange(8, device=device, dtype=torch.long).unsqueeze(0)
    positions = torch.arange(8, device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.triu(
        torch.ones((1, 1, 8, 8), device=device, dtype=torch.bool), diagonal=1
    )
    with torch.no_grad():
        expected = _run_forward(source, tokens, positions, attention_mask, source_pgs).clone()
        without_window = _run_forward(
            full_attention, tokens, positions, attention_mask, source_pgs
        ).clone()
        sink_values = {name: parameter.clone() for name, parameter in source_sinks.items()}
        for parameter in source_sinks.values():
            parameter.zero_()
        without_sink = _run_forward(source, tokens, positions, attention_mask, source_pgs).clone()
        for name, parameter in source_sinks.items():
            parameter.copy_(sink_values[name])
        initial = _run_forward(target, tokens, positions, attention_mask, target_pgs).clone()
    assert not torch.allclose(expected, without_window, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(expected, without_sink, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(expected, initial, atol=1e-5, rtol=1e-5)

    swap_model_weights([source], [target], refit_method="nccl")
    torch.cuda.synchronize()
    with torch.no_grad():
        actual = _run_forward(target, tokens, positions, attention_mask, target_pgs)
    for name, parameter in target_sinks.items():
        assert torch.equal(parameter, source_sinks[name])
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)

    dist.barrier()
    del source, target, full_attention
    clear_all_caches()
    _destroy_pg_collection(source_pgs)
    _destroy_pg_collection(target_pgs)
    Utils.destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("dsa_type", ["mla", "gqa"])
@pytest.mark.parametrize(
    "src_tp,src_pp,dst_tp,dst_pp",
    [
        # Four-GPU Blackwell coverage.
        (1, 1, 4, 1),
        (4, 1, 1, 1),
        (4, 1, 2, 2),
        (2, 2, 4, 1),
        # Eight-GPU H100 coverage.
        (1, 1, 8, 1),
        (8, 1, 1, 1),
        (8, 1, 4, 2),
        (4, 2, 8, 1),
    ],
)
@pytest.mark.launch_on_gb200
def test_swap_dsa_parallelism(dsa_type: str, src_tp: int, src_pp: int, dst_tp: int, dst_pp: int):
    """Refit real MLA and single-group GQA DSA models across full-node TP/PP layouts."""
    if not has_mamba_deps:
        pytest.skip("HybridModel dependencies (mamba_ssm, einops) not available")

    world = Utils.world_size
    if (world % (src_tp * src_pp) != 0) or (world % (dst_tp * dst_pp) != 0):
        pytest.skip("WORLD_SIZE must be divisible by both src_tp*src_pp and dst_tp*dst_pp")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=src_tp, pipeline_model_parallel_size=src_pp
    )
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    from math import lcm

    seq_len = 8
    vocab_size = 64
    base_pattern = "D-"
    num_layers = lcm(src_pp, dst_pp) * len(base_pattern)
    src_cfg = _dsa_config(num_layers, src_tp, src_pp, gqa=dsa_type == "gqa")
    dst_cfg = _dsa_config(num_layers, dst_tp, dst_pp, gqa=dsa_type == "gqa")
    stack_spec = _dsa_gqa_stack_spec() if dsa_type == "gqa" else hybrid_stack_spec
    src_pgs = _build_pg_collection(tp_size=src_tp, pp_size=src_pp)
    dst_pgs = _build_pg_collection(tp_size=dst_tp, pp_size=dst_pp)
    src_pattern = _hybrid_layer_pattern(base_pattern, num_layers, src_pp)
    dst_pattern = _hybrid_layer_pattern(base_pattern, num_layers, dst_pp)

    src_model = _build_hybrid(
        src_cfg,
        vocab_size,
        seq_len,
        src_pgs,
        src_pattern,
        parallel_output=False,
        stack_spec=stack_spec,
    ).to(device)
    dst_model = _build_hybrid(
        dst_cfg,
        vocab_size,
        seq_len,
        dst_pgs,
        dst_pattern,
        parallel_output=False,
        stack_spec=stack_spec,
    ).to(device)
    src_model.eval()
    dst_model.eval()

    for model in (src_model, dst_model):
        dsa_modules = [module for module in model.modules() if isinstance(module, DSAttention)]
        assert dsa_modules
        assert all(isinstance(module, _DSGQACoreAttention) for module in dsa_modules) == (
            dsa_type == "gqa"
        )

    batch = 1
    tokens = torch.arange(seq_len, device=device, dtype=torch.long).view(batch, seq_len)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(batch, seq_len)
    attention_mask = torch.triu(
        torch.ones((batch, 1, seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1
    )

    ref_logits = torch.empty(batch, seq_len, vocab_size, device=device)
    src_last_pp_rank = dist.get_process_group_ranks(src_pgs.pp)[-1]
    with torch.no_grad():
        src_out = _run_forward(src_model, tokens, position_ids, attention_mask, src_pgs)
        if dist.get_rank() == src_last_pp_rank:
            ref_logits.copy_(src_out)
    dist.broadcast(ref_logits, src=src_last_pp_rank, group=src_pgs.pp)

    swap_model_weights([src_model], [dst_model], refit_method="nccl")

    dst_logits = torch.empty_like(ref_logits)
    dst_last_pp_rank = dist.get_process_group_ranks(dst_pgs.pp)[-1]
    with torch.no_grad():
        dst_out = _run_forward(dst_model, tokens, position_ids, attention_mask, dst_pgs)
        if dist.get_rank() == dst_last_pp_rank:
            dst_logits.copy_(dst_out)
    dist.broadcast(dst_logits, src=dst_last_pp_rank, group=dst_pgs.pp)

    max_diff = (dst_logits - ref_logits).abs().max().item()
    assert torch.allclose(dst_logits, ref_logits, atol=2e-3, rtol=2e-3), (
        f"{dsa_type.upper()} DSA refit src(TP={src_tp},PP={src_pp})"
        f"->dst(TP={dst_tp},PP={dst_pp}) outputs differ (max_diff={max_diff:.6f})"
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
            "nccl_m2n",
            marks=pytest.mark.skipif(
                not has_nccl_m2n, reason="NVIDIA/nccl-extensions and NCCL4Py are not installed"
            ),
        ),
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
    if refit_backend == "nccl_m2n" and world != src_world + dst_world:
        Utils.destroy_model_parallel()
        pytest.skip("NCCL M2N requires a world containing exactly the source and destination ranks")

    try:
        import transformer_engine  # noqa: F401
    except Exception:
        Utils.destroy_model_parallel()
        pytest.skip("Transformer Engine not available")

    from megatron.core.inference.shards import build_inference_pg_collection

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
            "nccl_m2n",
            marks=pytest.mark.skipif(
                not has_nccl_m2n, reason="NVIDIA/nccl-extensions and NCCL4Py are not installed"
            ),
        ),
        "nccl",
    ],
)
@pytest.mark.parametrize(
    "src_world,src_tp,src_pp,dst_world,dst_tp,dst_pp",
    [
        # Four-rank Blackwell CI: two disjoint ranks per side, including native M2N.
        (2, 1, 1, 2, 2, 1),  # TP1,DP2 -> TP2
        (2, 2, 1, 2, 1, 1),  # TP2 -> TP1,DP2
        (2, 1, 1, 2, 1, 2),  # PP1,DP2 -> PP2
        (2, 1, 2, 2, 1, 1),  # PP2 -> PP1,DP2
        (2, 2, 1, 2, 1, 2),  # TP2,PP1 -> TP1,PP2
        (2, 1, 2, 2, 2, 1),  # TP1,PP2 -> TP2,PP1
        # Eight-rank H100 CI: four disjoint ranks per side (NCCL, plus M2N if installed).
        (4, 1, 1, 4, 4, 1),  # TP1,DP4 -> TP4
        (4, 4, 1, 4, 1, 1),  # TP4 -> TP1,DP4
        (4, 1, 1, 4, 1, 2),  # PP1,DP4 -> PP2,DP2
        (4, 1, 2, 4, 1, 1),  # PP2,DP2 -> PP1,DP4
        (4, 4, 1, 4, 2, 2),  # TP4,PP1 -> TP2,PP2
        (4, 2, 2, 4, 4, 1),  # TP2,PP2 -> TP4,PP1
    ],
)
@pytest.mark.parametrize("dsa_type", ["mla", "gqa"])
@pytest.mark.launch_on_gb200
def test_swap_dsa_non_collocated(
    refit_backend: str,
    src_world: int,
    src_tp: int,
    src_pp: int,
    dst_world: int,
    dst_tp: int,
    dst_pp: int,
    dsa_type: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Refit real MLA and GQA DSA layers across disjoint rank windows."""
    if not has_mamba_deps:
        pytest.skip("Mamba dependencies (mamba_ssm, einops) not available")
    if Utils.world_size != src_world + dst_world:
        pytest.skip("Test requires a world containing exactly the source and destination ranks")
    if refit_backend == "nccl_m2n":
        if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10:
            pytest.skip("Native NCCL M2N coverage runs on the GIN-enabled Blackwell CI lane")
        # PACK is M2N's default, supported copy transport. Keep submissions to one
        # complete parameter so this model test does not depend on the extension's
        # cross-tensor fusion path; grouped fusion has its own focused unit coverage.
        monkeypatch.setenv("NCCL_RESHARD_COPY_ALGORITHM", "PACK")

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    from math import lcm

    from megatron.core.inference.shards import build_inference_pg_collection

    rank = dist.get_rank()
    is_src = rank < src_world
    is_dst = src_world <= rank < src_world + dst_world
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    src_pgs = build_inference_pg_collection(
        world_size=src_world, tp_size=src_tp, pp_size=src_pp, rank_offset=0
    )
    dst_pgs = build_inference_pg_collection(
        world_size=dst_world, tp_size=dst_tp, pp_size=dst_pp, rank_offset=src_world
    )

    base_pattern = "D-"
    num_layers = lcm(src_pp, dst_pp) * len(base_pattern)
    src_cfg = _dsa_config(num_layers, src_tp, src_pp, gqa=dsa_type == "gqa")
    dst_cfg = _dsa_config(num_layers, dst_tp, dst_pp, gqa=dsa_type == "gqa")
    stack_spec = _dsa_gqa_stack_spec() if dsa_type == "gqa" else hybrid_stack_spec
    src_pattern = _hybrid_layer_pattern(base_pattern, num_layers, src_pp)
    dst_pattern = _hybrid_layer_pattern(base_pattern, num_layers, dst_pp)

    src_model = None
    dst_model = None
    if is_src:
        src_model = _build_hybrid(
            src_cfg,
            vocab_size=64,
            seq_len=8,
            pg_collection=src_pgs,
            hybrid_layer_pattern=src_pattern,
            parallel_output=False,
            stack_spec=stack_spec,
        )
        src_model = src_model.to(device).eval()
    elif is_dst:
        dst_model = _build_hybrid(
            dst_cfg,
            vocab_size=64,
            seq_len=8,
            pg_collection=dst_pgs,
            hybrid_layer_pattern=dst_pattern,
            parallel_output=False,
            stack_spec=stack_spec,
        )
        dst_model = dst_model.to(device).eval()

    local_model = src_model if src_model is not None else dst_model
    dsa_modules = [module for module in local_model.modules() if isinstance(module, DSAttention)]
    assert dsa_modules
    assert all(isinstance(module, _DSGQACoreAttention) for module in dsa_modules) == (
        dsa_type == "gqa"
    )

    batch = 1
    seq_len = 8
    vocab_size = 64
    tokens = torch.arange(seq_len, device=device, dtype=torch.long).view(batch, seq_len)
    position_ids = (
        torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    )
    attention_mask = torch.triu(
        torch.ones((batch, 1, seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1
    )

    ref_logits = torch.empty(batch, seq_len, vocab_size, device=device, dtype=torch.float32)
    if is_src:
        with torch.no_grad():
            src_out = _run_forward(src_model, tokens, position_ids, attention_mask, src_pgs)
        if rank == src_world - 1:
            ref_logits.copy_(src_out)
    dist.broadcast(ref_logits, src=src_world - 1)

    swap_model_weights(
        [src_model] if src_model is not None else None,
        [dst_model] if dst_model is not None else None,
        refit_method=refit_backend,
        execution_batch_bytes=1 if refit_backend == "nccl_m2n" else None,
    )
    torch.cuda.synchronize()

    dst_logits = torch.empty_like(ref_logits)
    if is_dst:
        with torch.no_grad():
            dst_out = _run_forward(dst_model, tokens, position_ids, attention_mask, dst_pgs)
        if rank == src_world + dst_world - 1:
            dst_logits.copy_(dst_out)
    dist.broadcast(dst_logits, src=src_world + dst_world - 1)

    max_diff = (dst_logits - ref_logits).abs().max().item()
    assert torch.allclose(dst_logits, ref_logits, atol=2e-3, rtol=2e-3), (
        f"Non-collocated {dsa_type.upper()} DSA refit via {refit_backend} "
        f"src(TP={src_tp},PP={src_pp})->dst(TP={dst_tp},PP={dst_pp}) "
        f"outputs differ (max_diff={max_diff:.6f})"
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

    world = Utils.world_size
    if (world % (src_tp * src_pp) != 0) or (world % (dst_tp * dst_pp) != 0):
        pytest.skip("WORLD_SIZE must be divisible by both src_tp*src_pp and dst_tp*dst_pp")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=src_tp, pipeline_model_parallel_size=src_pp
    )

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

    src_pattern = _hybrid_layer_pattern(base_pattern, num_layers, src_pp)
    dst_pattern = _hybrid_layer_pattern(base_pattern, num_layers, dst_pp)

    src_model = _build_hybrid(
        cfg, vocab_size, seq_len, src_pgs, src_pattern, parallel_output=False
    ).to(device)
    dst_model = _build_hybrid(
        cfg, vocab_size, seq_len, dst_pgs, dst_pattern, parallel_output=False
    ).to(device)
    src_model.eval()
    dst_model.eval()

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
