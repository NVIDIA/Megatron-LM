# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-FSDP v2 composed with expert parallelism through a real MCore HybridModel.

Checks that an ``EP=4`` transformer-MoE ``HybridModel`` (an attention layer + a MoE layer)
sharded with mFSDP v2 (experts over the expert-DP sub-mesh, dense params over the full DP
mesh), consuming its ``1/dp`` shard of a global batch, reproduces a single **full-batch
``EP=1`` reference**.

The reference processes the whole global batch on every rank, so its gradients are
identical across ranks and need no reduction -- it has no distributed logic. The model's
gradients are reduced only by mFSDP. So this independently validates EP all-to-all
dispatch, FSDP sharding, and the gradient reduction/scaling: a broken reduction (or a
missing expert-grad scaling factor) would diverge from full-batch training.

Both models are built from explicit ``ProcessGroupCollection``s (no global
``parallel_state`` / ``initialize_model_parallel``): the reference with a size-1 ``ep``
group (all experts local), the model with the 2-way ``ep`` group. Model shapes and the
``(ep, dp)`` split are test-local so different tests can vary them.
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    fully_shard_context,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import MoETransformerLayer

_FLAT_SHARD = Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def _transformer_config(
    num_layers: int, num_experts: int, ep_size: int, hidden: int, ffn_hidden: int
) -> TransformerConfig:
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


def _build_process_group_collection(
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


def _build_hybrid_model(
    config: TransformerConfig,
    pg_collection: ProcessGroupCollection,
    vocab: int,
    seq: int,
    pattern: str,
) -> HybridModel:
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=vocab,
        max_sequence_length=seq,
        hybrid_layer_pattern=pattern,
        pg_collection=pg_collection,
    ).cuda()


def _train(
    model: torch.nn.Module,
    ids: torch.Tensor,
    pos: torch.Tensor,
    mask: torch.Tensor | None,
    target: torch.Tensor,
    loss_reduce_group: dist.ProcessGroup | None = None,
) -> list[torch.Tensor]:
    """Run 5 SGD steps; return the per-step losses (globally averaged if loss_reduce_group given)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, foreach=False)
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


def test_ep_fsdp_matches_fullbatch_reference(distributed_setup):
    """EP=4 + mFSDP on 1/dp-sharded data reproduces single full-batch EP=1 training."""
    device = distributed_setup.device
    world_size, rank = distributed_setup.world_size, distributed_setup.rank

    num_experts, ep_size = 8, 4
    # hidden=64 with 4 heads -> head_dim=16, large enough for the attention backend.
    hidden, ffn_hidden, vocab, seq, b_local = 64, 128, 128, 8, 2
    # HybridModel builds one layer per pattern symbol, so "*E" is a two-layer stack: "*" a
    # self-attention-only layer, "E" a MoE layer -- i.e. a real transformer-MoE block
    # (attention then MoE, two residuals; equivalent to a Mixtral-style layer). This exercises
    # mFSDP sharding the dense attention params over full DP alongside EP-sharded experts.
    layer_pattern = "*E"
    num_layers = len(layer_pattern)
    if world_size % ep_size != 0 or num_experts % ep_size != 0:
        pytest.skip(f"world_size {world_size} is incompatible with EP={ep_size}.")
    edp_size = world_size // ep_size
    global_batch = world_size * b_local  # one shard per rank

    # Process groups (no global parallel_state). world_mesh: the full DP group; moe_mesh: the
    # ep (ep_size-way) and expert-DP (edp_size-way) groups for the EP=4 model. Meshes also
    # initialize the default process group, so build them before the size-1 group below.
    world_mesh = init_device_mesh(device.type, (world_size,))
    world = world_mesh.get_group()
    moe_mesh = init_device_mesh(device.type, (edp_size, ep_size), mesh_dim_names=("edp", "ep"))
    ep_group, expert_dp_group = moe_mesh.get_group("ep"), moe_mesh.get_group("edp")
    # This rank's size-1 group: the trivial TP=PP=CP axes and the EP=1 reference's ep group.
    one = dist.new_group([rank], use_local_synchronization=True)

    # Reference EP=1 (all experts local); model EP=4. Seed once so the reference is
    # deterministic and identical across ranks (CPU init); the model's own init is irrelevant
    # since its weights are copied from the reference below.
    torch.manual_seed(123)
    reference = _build_hybrid_model(
        _transformer_config(num_layers, num_experts, 1, hidden, ffn_hidden),
        _build_process_group_collection(one, dp=one, ep=one, expert_dp=one),
        vocab,
        seq,
        layer_pattern,
    )
    model = _build_hybrid_model(
        _transformer_config(num_layers, num_experts, ep_size, hidden, ffn_hidden),
        _build_process_group_collection(one, dp=world, ep=ep_group, expert_dp=expert_dp_group),
        vocab,
        seq,
        layer_pattern,
    )

    # Dense params line up by name (load_state_dict); the experts do not -- EP=1 stores all
    # experts as weight0.., EP=4 stores num_experts/EP as weight0.. per rank -- so patch them
    # by global index (model local weight i == reference global weight local_expert_indices[i]).
    model.load_state_dict(reference.state_dict(), strict=False)
    for model_layer, reference_layer in zip(model.decoder.layers, reference.decoder.layers):
        if not isinstance(model_layer, MoETransformerLayer):
            continue  # only MoE layers have experts to remap; the attention layer has none
        for fc in ("linear_fc1", "linear_fc2"):
            model_fc = getattr(model_layer.mlp.experts, fc)
            reference_fc = getattr(reference_layer.mlp.experts, fc)
            for local, global_ in enumerate(model_layer.mlp.local_expert_indices):
                getattr(model_fc, f"weight{local}").data.copy_(
                    getattr(reference_fc, f"weight{global_}").data
                )

    # Shard the model: experts over the expert-DP sub-mesh, dense params over the full DP mesh.
    # Experts additionally need grad_divisor=ep_size; see fully_shard.
    with fully_shard_context(device=device):
        for decoder_layer in model.decoder.layers:
            if isinstance(decoder_layer, MoETransformerLayer):
                fully_shard(
                    decoder_layer.mlp.experts,
                    mesh=moe_mesh["edp"],
                    placements=_FLAT_SHARD,
                    grad_divisor=ep_size,
                )
        fully_shard(model, mesh=world_mesh, placements=_FLAT_SHARD)

    # One global batch, identical on every rank; the reference sees all of it, the model its shard.
    torch.manual_seed(4321)
    ids = torch.randint(0, vocab, (global_batch, seq), dtype=torch.int64, device=device)
    pos = torch.arange(seq, dtype=torch.int64, device=device).repeat(global_batch, 1)
    mask = None  # attention layer is attn_mask_type=causal, so TE builds the causal mask itself
    target = torch.randn(global_batch, seq, vocab, device=device)
    shard = slice(rank * b_local, (rank + 1) * b_local)

    reference_losses = _train(reference, ids, pos, mask, target)
    model_losses = _train(
        model, ids[shard], pos[shard], mask, target[shard], loss_reduce_group=world
    )

    torch.testing.assert_close(
        torch.stack(model_losses),
        torch.stack(reference_losses),
        msg="EP=4 mFSDP model did not reproduce full-batch EP=1 training.",
    )

    # Destroy the groups this test created; leave the default (world) group for later tests.
    for group in (one, ep_group, expert_dp_group):
        dist.destroy_process_group(group)
