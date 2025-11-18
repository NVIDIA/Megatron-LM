import os
import copy
import types
import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.test_utilities import Utils
from megatron.core.model_refitting import swap_model_weights
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.model_parallel_config import ModelParallelConfig
from mcore_reshard import reshard_with_general_planner
from typing import Tuple


def _build_pg_collection(tp_size: int, pp_size: int = None) -> ProcessGroupCollection:
    cp_size = mpu.get_context_parallel_world_size()
    if pp_size is None:
        pp_size = mpu.get_pipeline_model_parallel_world_size()
    world_size = dist.get_world_size()
    dp_size = world_size // (tp_size * cp_size * pp_size)
    assert dp_size >= 1 and (tp_size * cp_size * pp_size * dp_size) == world_size

    grid = HyperCommGrid([tp_size, cp_size, 1, pp_size, dp_size], ["tp", "cp", "ep", "pp", "dp"])
    tp_group = grid.create_pg("tp")
    cp_group = grid.create_pg("cp")
    pp_group = grid.create_pg("pp")
    ep_group = grid.create_pg("ep")
    dp_group = grid.create_pg("dp")
    embd_group_ranks = mpu.default_embedding_ranks(dist.get_process_group_ranks(pp_group))
    embd_group = dist.new_group(ranks=embd_group_ranks)
    return ProcessGroupCollection(tp=tp_group, cp=cp_group, pp=pp_group, ep=ep_group, embd=embd_group, dp=dp_group)


def _build_gpt(config: TransformerConfig, vocab_size: int, seq_len: int, pg_collection, parallel_output: bool = True) -> GPTModel:
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
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
    "src_tp,src_pp,dst_tp,dst_pp",
    [
        (2, 1, 1, 1),  # TP2 -> TP1
        (1, 1, 2, 1),  # TP1 -> TP2
        (1, 2, 1, 1),  # PP2 -> PP1
        (1, 1, 1, 2),  # PP1 -> PP2
        (2, 2, 1, 1),  # TP2,PP2 -> TP1,PP1
        (1, 1, 2, 2),  # TP1,PP1 -> TP2,PP2
        (2, 1, 1, 2),  # TP2,PP1 -> TP1,PP2
        (1, 2, 2, 1),  # TP1,PP2 -> TP2,PP1
    ],
)
def test_nccl_swap_gpt_parametrized(src_tp: int, src_pp: int, dst_tp: int, dst_pp: int):
    # Initialize environment with source MP sizing
    Utils.initialize_model_parallel(tensor_model_parallel_size=src_tp, pipeline_model_parallel_size=src_pp)
    # Validate divisibility post-init using the default PG safely
    world = dist.get_world_size()
    if (world % (src_tp * src_pp) != 0) or (world % (dst_tp * dst_pp) != 0):
        Utils.destroy_model_parallel()
        pytest.skip("WORLD_SIZE must be divisible by both src_tp*src_pp and dst_tp*dst_pp")
    model_parallel_cuda_manual_seed(1234)

    torch.manual_seed(1234)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Small GPT config
    seq_len = 8
    vocab_size = 128
    cfg = TransformerConfig(
        num_layers=4 if (src_pp > 1 or dst_pp > 1) else 2,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    # Build PGs and models
    src_pgs = ProcessGroupCollection.use_mpu_process_groups()
    dst_pgs = _build_pg_collection(tp_size=dst_tp, pp_size=dst_pp)
    # Use parallel_output=False to gather vocab-parallel outputs inside model and emit only on last PP stage
    src_model = _build_gpt(copy.deepcopy(cfg), vocab_size, seq_len, src_pgs, parallel_output=False).to(device).eval()
    dst_model = _build_gpt(copy.deepcopy(cfg), vocab_size, seq_len, dst_pgs, parallel_output=False).to(device).eval()

    # Inputs
    batch = 2
    tokens = torch.randint(low=0, high=vocab_size, size=(batch, seq_len), device=device, dtype=torch.long)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    attention_mask = torch.ones((batch, 1, seq_len, seq_len), device=device, dtype=torch.bool)

    # Collect source reference logits (parallel_output=False ensures full vocab on last PP stage)
    ref_logits = torch.empty(batch, seq_len, vocab_size, device=device, dtype=torch.float32)
    src_pp_ranks = dist.get_process_group_ranks(src_pgs.pp)
    src_last_pp_rank = src_pp_ranks[-1]
    with torch.no_grad():
        src_out = src_model(tokens, position_ids, attention_mask)
        if dist.get_rank() == src_last_pp_rank:
            ref = src_out  # [b, s, vocab]
            ref_logits.copy_(ref)
    dist.broadcast(ref_logits, src=src_last_pp_rank, group=src_pgs.pp)

    # Swap weights
    swap_model_weights([src_model], [dst_model], refit_method="nccl")

    # Collect destination logits (parallel_output=False ensures full vocab on last PP stage)
    dst_logits = torch.empty(batch, seq_len, vocab_size, device=device, dtype=torch.float32)
    dst_pp_ranks = dist.get_process_group_ranks(dst_pgs.pp)
    dst_last_pp_rank = dst_pp_ranks[-1]
    with torch.no_grad():
        dst_out = dst_model(tokens, position_ids, attention_mask)  # last stage returns tensor, others return None
        if dist.get_rank() == dst_last_pp_rank:
            dst_logits.copy_(dst_out)  # [b, s, vocab]
    dist.broadcast(dst_logits, src=dst_last_pp_rank, group=dst_pgs.pp)

    # Compare
    assert ref_logits.shape == dst_logits.shape
    assert torch.allclose(dst_logits, ref_logits, atol=1e-4, rtol=1e-4), f"Refit src(TP={src_tp},PP={src_pp})->dst(TP={dst_tp},PP={dst_pp}) GPT outputs differ"

    dist.barrier()
    Utils.destroy_model_parallel()

# def test_nccl_swap_row_parallel_linear_tp2_to_tp1():
#     Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
#     model_parallel_cuda_manual_seed(1234)
#     device = torch.device(f"cuda:{torch.cuda.current_device()}")

#     # Build TP=2 source and TP=1 dest groups
#     src_pgs = ProcessGroupCollection.use_mpu_process_groups()
#     infer_pgs = _build_pg_collection(tp_size=1)

#     in_features = 12
#     out_features = 16
#     cfg = _mp_config()

#     # Source RowParallelLinear (TP=2), input_is_parallel=False so it scatters internally
#     src_layer = RowParallelLinear(
#         input_size=in_features,
#         output_size=out_features,
#         config=cfg,
#         init_method=torch.nn.init.zeros_,
#         bias=False,
#         input_is_parallel=False,
#         skip_bias_add=True,
#         tp_group=src_pgs.tp,
#     ).to(device)
#     _set_pg_collection(src_layer, src_pgs.tp, src_pgs.dp)
#     # Ensure TP metadata is present for planner (row-parallel shards input dim=1)
#     src_layer.weight.tensor_model_parallel = True
#     src_layer.weight.partition_dim = 1
#     src_layer.weight.partition_stride = 1

#     # Deterministic per-rank weights (sharded along dim=1)
#     rank = dist.get_rank(src_pgs.tp)
#     with torch.no_grad():
#         src_layer.weight.copy_(
#             torch.arange(src_layer.weight.numel(), device=device, dtype=torch.float32).reshape_as(
#                 src_layer.weight
#             )
#             + rank * 1000.0
#         )

#     # Destination RowParallelLinear (TP=1)
#     dst_layer = RowParallelLinear(
#         input_size=in_features,
#         output_size=out_features,
#         config=_mp_config(),
#         init_method=torch.nn.init.zeros_,
#         bias=False,
#         input_is_parallel=False,
#         skip_bias_add=True,
#         tp_group=infer_pgs.tp,
#     ).to(device)
#     _set_pg_collection(dst_layer, infer_pgs.tp, infer_pgs.dp)
#     # Destination is unsharded (TP=1) but keep metadata consistent
#     dst_layer.weight.tensor_model_parallel = False
#     dst_layer.weight.partition_dim = 1
#     dst_layer.weight.partition_stride = 1

#     # Use layers directly to simplify parameter name matching
#     src = src_layer
#     dst = dst_layer
#     # Attach pg_collection to layers so reshard can find process groups
#     src.pg_collection = src_pgs
#     dst.pg_collection = infer_pgs

#     # Input and reference (gather master weight along dim=1 from TP=2)
#     x = torch.randn(4, in_features, device=device)
#     parts = [torch.empty_like(src_layer.weight) for _ in range(dist.get_world_size(src_pgs.tp))]
#     dist.all_gather(parts, src_layer.weight.contiguous(), group=src_pgs.tp)
#     master_w = torch.cat(parts, dim=1).contiguous()  # [out, in]
#     ref = x @ master_w.t()

#     # Use resharder directly for per-layer validation and inspect plan
#     plan = reshard_with_general_planner(src, dst)
#     assert (len(plan.recv_ops) + len(plan.local_copy_ops)) > 0, "No transfers scheduled for RowParallelLinear"
#     # Verify weights transferred correctly
#     with torch.no_grad():
#         assert dst_layer.weight.shape == master_w.shape
#         assert torch.allclose(dst_layer.weight, master_w, atol=1e-6, rtol=1e-6), "RowParallelLinear weights mismatch after transfer"
#     y, _ = dst(x)
#     assert torch.allclose(y, ref, atol=1e-4, rtol=1e-4), "RowParallelLinear TP2->TP1 mismatch"

#     dist.barrier()
#     Utils.destroy_model_parallel()

# def test_nccl_swap_column_parallel_linear_tp2_to_tp1():
#     Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
#     model_parallel_cuda_manual_seed(1234)
#     device = torch.device(f"cuda:{torch.cuda.current_device()}")

#     # Build TP=2 source and TP=1 dest groups
#     src_pgs = ProcessGroupCollection.use_mpu_process_groups()
#     infer_pgs = _build_pg_collection(tp_size=1)

#     in_features = 12
#     out_features = 16
#     cfg = _mp_config()

#     # Source ColumnParallelLinear (TP=2)
#     src_layer = ColumnParallelLinear(
#         input_size=in_features,
#         output_size=out_features,
#         config=cfg,
#         init_method=torch.nn.init.zeros_,
#         bias=False,
#         gather_output=False,
#         tp_group=src_pgs.tp,
#     ).to(device)
#     _set_pg_collection(src_layer, src_pgs.tp, src_pgs.dp)
#     # Ensure TP metadata is present for planner
#     src_layer.weight.tensor_model_parallel = True
#     src_layer.weight.partition_dim = 0
#     src_layer.weight.partition_stride = 1

#     # Deterministic per-rank weights
#     rank = dist.get_rank(src_pgs.tp)
#     with torch.no_grad():
#         src_layer.weight.copy_(
#             torch.arange(src_layer.weight.numel(), device=device, dtype=torch.float32).reshape_as(
#                 src_layer.weight
#             )
#             + rank * 1000.0
#         )

#     # Destination ColumnParallelLinear (TP=1)
#     dst_layer = ColumnParallelLinear(
#         input_size=in_features,
#         output_size=out_features,
#         config=_mp_config(),
#         init_method=torch.nn.init.zeros_,
#         bias=False,
#         gather_output=False,
#         tp_group=infer_pgs.tp,
#     ).to(device)
#     _set_pg_collection(dst_layer, infer_pgs.tp, infer_pgs.dp)
#     # Destination is unsharded (TP=1) but keep metadata consistent
#     dst_layer.weight.tensor_model_parallel = False
#     dst_layer.weight.partition_dim = 0
#     dst_layer.weight.partition_stride = 1

#     # Use layers directly to simplify parameter name matching
#     src = src_layer
#     dst = dst_layer
#     # Attach pg_collection to layers so reshard can find process groups
#     src.pg_collection = src_pgs
#     dst.pg_collection = infer_pgs

#     # Input and reference (gather master weight from TP=2)
#     x = torch.randn(4, in_features, device=device)
#     parts = [torch.empty_like(src_layer.weight) for _ in range(dist.get_world_size(src_pgs.tp))]
#     dist.all_gather(parts, src_layer.weight.contiguous(), group=src_pgs.tp)
#     master_w = torch.cat(parts, dim=0).contiguous()  # [out, in]
#     ref = x @ master_w.t()

#     # Use resharder directly for per-layer validation and inspect plan
#     plan = reshard_with_general_planner(src, dst)
#     assert (len(plan.recv_ops) + len(plan.local_copy_ops)) > 0, "No transfers scheduled for ColumnParallelLinear"
#     # Verify weights transferred correctly
#     with torch.no_grad():
#         assert dst_layer.weight.shape == master_w.shape
#         assert torch.allclose(dst_layer.weight, master_w, atol=1e-6, rtol=1e-6), "ColumnParallelLinear weights mismatch after transfer"
#     y, _ = dst(x)
#     assert torch.allclose(y, ref, atol=1e-4, rtol=1e-4), "ColumnParallelLinear TP2->TP1 mismatch"

#     dist.barrier()
#     Utils.destroy_model_parallel()