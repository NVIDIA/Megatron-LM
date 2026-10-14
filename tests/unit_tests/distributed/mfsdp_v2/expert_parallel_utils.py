# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared MoE model construction and training for MFSDP v2 expert-parallel tests."""

import torch
import torch.distributed as dist

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig


def transformer_config(
    num_layers: int, num_experts: int, ep_size: int, hidden: int, ffn_hidden: int
) -> TransformerConfig:
    """Build a small FP32 MoE configuration with dropout disabled."""
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        num_attention_heads=4,
        num_moe_experts=num_experts,
        expert_model_parallel_size=ep_size,
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_aux_loss_coeff=0.0,
        moe_grouped_gemm=True,
        moe_ffn_hidden_size=ffn_hidden,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        # unfused (native-PyTorch) attention: flash/fused don't support the fp32 params we use.
        attention_backend=AttnBackend.unfused,
    )


def build_process_group_collection(
    one: dist.ProcessGroup,
    dp: dist.ProcessGroup,
    ep: dist.ProcessGroup,
    expert_dp: dist.ProcessGroup,
) -> ProcessGroupCollection:
    """A ProcessGroupCollection for a TP=PP=CP=1 MoE model. `one` is a size-1 group for the
    trivial TP/PP/CP axes; dp, ep, expert_dp are the data-, expert-, and expert-data-parallel
    groups.
    """
    return ProcessGroupCollection(
        tp=one,
        expt_tp=one,
        cp=one,
        pp=one,
        tp_cp=one,
        tp_dp_cp=dp,
        ep=ep,
        tp_ep=ep,
        expt_dp=expert_dp,
        dp=dp,
        dp_cp=dp,
        embd=None,
        pos_embd=None,
    )


def build_hybrid_model(
    config: TransformerConfig,
    pg_collection: ProcessGroupCollection,
    vocab: int,
    seq: int,
    pattern: str,
) -> HybridModel:
    """Build a CUDA HybridModel with explicit process groups."""
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=vocab,
        max_sequence_length=seq,
        hybrid_layer_pattern=pattern,
        pg_collection=pg_collection,
    ).cuda()


def train_hybrid_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ids: torch.Tensor,
    pos: torch.Tensor,
    mask: torch.Tensor | None,
    target: torch.Tensor,
    loss_reduce_group: dist.ProcessGroup | None = None,
) -> list[torch.Tensor]:
    """Run five optimizer steps; return the per-step losses (globally averaged if loss_reduce_group given)."""
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(
            model(input_ids=ids, position_ids=pos, attention_mask=mask), target
        )
        loss.backward()
        optimizer.step()
        # The model sees a shard, so average the loss across ranks for the global loss.
        loss = loss.detach()
        if loss_reduce_group is not None:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=loss_reduce_group)
        losses.append(loss)
    return losses
