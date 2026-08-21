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

Both MoE expert weight layouts are covered:

* per-expert grouped weights -- ``weight0``, ``weight1``, ... one parameter per local expert;
* ``moe_single_grouped_weight`` -- one fused TE ``GroupedTensor`` per ``GroupedLinear`` holding
  every local expert, which keeps its values in ``rowwise_data`` rather than in ``.data``.

Both models are built from explicit ``ProcessGroupCollection``s (no global
``parallel_state`` / ``initialize_model_parallel``): the reference with a size-1 ``ep``
group (all experts local), the model with the 2-way ``ep`` group. Model shapes and the
``(ep, dp)`` split are test-local so different tests can vary them.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import MoETransformerLayer
from megatron.core.utils import is_te_min_version

_FLAT_SHARD = Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])

_WEIGHT_LAYOUTS = [
    pytest.param(False, id="per_expert_weights"),
    pytest.param(
        True,
        id="single_grouped_weight",
        marks=pytest.mark.skipif(
            not is_te_min_version("2.14.0"),
            reason="moe_single_grouped_weight requires Transformer Engine >= 2.14.0",
        ),
    ),
]


@pytest.fixture
def te_single_grouped_env(request):
    """Set the env vars TE reads for the fused path, for the cases that use it.

    TE reads these both when constructing GroupedLinear and later when matching fused ops on
    the first forward, so they have to stay set for the whole test, not just model build.
    """
    if not request.getfixturevalue("single_grouped_weight"):
        yield
        return

    names = ("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "NVTE_CUTEDSL_FUSED_GROUPED_MLP")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = "1"
    yield
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _transformer_config(
    num_layers: int,
    num_experts: int,
    ep_size: int,
    hidden: int,
    ffn_hidden: int,
    single_grouped_weight: bool,
) -> TransformerConfig:
    # The fused single-weight layout is only supported under the TE op fuser, so the two
    # layouts differ by exactly the flags that name the feature under test. Everything else,
    # including the SwiGLU the op fuser needs, is shared.
    fused_layout = (
        dict(moe_single_grouped_weight=True, use_transformer_engine_op_fuser=True)
        if single_grouped_weight
        else {}
    )

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
        hidden_dropout=0.0,
        attention_dropout=0.0,
        # SwiGLU: an ordinary MoE activation, and the one the TE op fuser can fuse.
        gated_linear_unit=True,
        activation_func=F.silu,
        params_dtype=torch.bfloat16,
        bf16=True,
        # GPU initialization, because model.cuda() cannot move a fused grouped weight: it
        # migrates the GroupedTensor wrapper but leaves the rowwise_data buffer holding its
        # values on the host, so the grouped GEMM gets a host pointer and faults. Both layouts
        # initialize the same way rather than only the one that has to.
        use_cpu_initialization=False,
        # unfused (native-PyTorch) attention keeps the reference deterministic.
        attention_backend=AttnBackend.unfused,
        **fused_layout,
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


def _expert_tensors(linear: torch.nn.Module, num_experts: int) -> list:
    """Per-expert views of an unsharded grouped linear's weights.

    ``num_experts`` is how many experts this linear holds: all of them on the EP=1 reference,
    only this rank's on the EP model. Handles both layouts -- a fused GroupedTensor is one
    parameter covering every expert, and its values live in ``rowwise_data`` rather than in a
    directly viewable ``.data``, so index a reshaped view of that instead.
    """
    fused = getattr(linear, "weight", None)
    if fused is not None:
        rowwise_data = getattr(fused, "rowwise_data", None)
        storage = fused.data if rowwise_data is None else rowwise_data
        return list(storage.view(fused.shape))
    return [getattr(linear, f"weight{index}").data for index in range(num_experts)]


def _expert_parameters(linear: torch.nn.Module, num_local_experts: int) -> list:
    """The parameters holding a grouped linear's experts, for either layout.

    Unlike ``_expert_tensors`` these are the parameters themselves, not per-expert views: after
    ``fully_shard`` each is a DTensor over an arbitrary flat slice of the group's buffer, which
    no longer lines up with expert boundaries.
    """
    fused = getattr(linear, "weight", None)
    if fused is not None:
        return [fused]
    return [getattr(linear, f"weight{index}") for index in range(num_local_experts)]


def _storage_pointer(parameter: torch.nn.Parameter) -> int:
    """Address of the buffer the compute kernels actually read for ``parameter``.

    For a fused GroupedTensor that is ``rowwise_data``, not ``.data`` -- which is the whole
    reason FSDP has to remap grouped storage rather than reassign ``.data``.
    """
    rowwise_data = getattr(parameter, "rowwise_data", None)
    return (parameter.data if rowwise_data is None else rowwise_data).data_ptr()


def _moe_layers(model: HybridModel, reference: HybridModel):
    """Yield (fc name, model linear, reference linear, local expert indices) for each MoE layer."""
    for model_layer, reference_layer in zip(model.decoder.layers, reference.decoder.layers):
        if not isinstance(model_layer, MoETransformerLayer):
            continue  # only MoE layers have experts; the attention layer has none
        for fc in ("linear_fc1", "linear_fc2"):
            yield (
                fc,
                getattr(model_layer.mlp.experts, fc),
                getattr(reference_layer.mlp.experts, fc),
                list(model_layer.mlp.local_expert_indices),
            )


def _copy_expert_weights(model: HybridModel, reference: HybridModel, num_experts: int) -> None:
    """Copy each of the model's local experts out of the reference's all-expert weights.

    Experts cannot be matched by name the way dense parameters can: the reference holds all
    ``num_experts`` of them and the model only ``num_experts / ep_size``, so the model's local
    expert ``i`` is the reference's expert ``local_expert_indices[i]``.
    """
    for _, model_linear, reference_linear, local_expert_indices in _moe_layers(model, reference):
        model_experts = _expert_tensors(model_linear, len(local_expert_indices))
        reference_experts = _expert_tensors(reference_linear, num_experts)
        for local, global_ in enumerate(local_expert_indices):
            model_experts[local].copy_(reference_experts[global_])


def _train(
    model: torch.nn.Module,
    ids: torch.Tensor,
    pos: torch.Tensor,
    mask: torch.Tensor | None,
    target: torch.Tensor,
    loss_reduce_group: dist.ProcessGroup | None = None,
    is_fsdp: bool = False,
) -> list[torch.Tensor]:
    """Run 5 SGD steps; return the per-step losses (globally averaged if loss_reduce_group given)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, foreach=False)
    if is_fsdp:
        # When main weights and compute weights are different buffers (they are whenever the
        # parameter dtype is not the optimizer dtype), only this hook refreshes the compute
        # weights after a step. Without it the model silently trains nothing.
        fully_shard_optimizer(optimizer)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(
            model(input_ids=ids, position_ids=pos, attention_mask=mask).float(), target
        )
        loss.backward()
        optimizer.step()
        # The model sees a shard, so average the loss across ranks for the global loss.
        loss = loss.detach()
        if loss_reduce_group is not None:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=loss_reduce_group)
        losses.append(loss)
    return losses


@pytest.mark.parametrize("single_grouped_weight", _WEIGHT_LAYOUTS)
def test_ep_fsdp_matches_fullbatch_reference(
    distributed_setup, single_grouped_weight, te_single_grouped_env
):
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

    # Reference EP=1 (all experts local); model EP=4. Seed the default generator and the
    # model-parallel RNG tracker -- with explicit ranks, so GPU initialization needs no global
    # parallel_state and every rank draws the same weights. The model's own init is irrelevant
    # since its weights are copied from the reference below.
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123, tp_rank=0, ep_rank=0, etp_rank=0)
    reference = _build_hybrid_model(
        _transformer_config(num_layers, num_experts, 1, hidden, ffn_hidden, single_grouped_weight),
        _build_process_group_collection(one, dp=one, ep=one, expert_dp=one),
        vocab,
        seq,
        layer_pattern,
    )
    model = _build_hybrid_model(
        _transformer_config(
            num_layers, num_experts, ep_size, hidden, ffn_hidden, single_grouped_weight
        ),
        _build_process_group_collection(one, dp=world, ep=ep_group, expert_dp=expert_dp_group),
        vocab,
        seq,
        layer_pattern,
    )

    # Dense params line up by name. The expert weights do not -- the two sides hold a different
    # number of experts -- so exclude them from the load and copy them by index.
    dense_state = {
        name: tensor
        for name, tensor in reference.state_dict().items()
        if ".mlp.experts." not in name
    }
    model.load_state_dict(dense_state, strict=False)
    with torch.no_grad():
        _copy_expert_weights(model, reference, num_experts)

    # Record where each expert weight's values currently live. fully_shard must move them into
    # its own buffer. This matters for the fused layout in particular: a GroupedTensor's kernels
    # read rowwise_data, not .data, so a fully_shard that only reassigned .data would leave TE
    # reading storage FSDP never writes -- silently freezing every expert. One backward cannot
    # see that (the first gradient is correct either way), so assert on the remap directly.
    # Hold the parameter objects, not views of them: FSDP remaps their storage in place, so a
    # view captured now would keep reporting the old address.
    original_expert_storage = [
        (fc, parameter, _storage_pointer(parameter))
        for fc, model_linear, _, indices in _moe_layers(model, reference)
        for parameter in _expert_parameters(model_linear, len(indices))
    ]

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

    for fc, parameter, original_pointer in original_expert_storage:
        assert _storage_pointer(parameter) != original_pointer, (
            f"{fc}: fully_shard left an expert weight pointing at its original storage, so "
            "the compute kernels would keep reading values FSDP never updates."
        )

    # Experts shard over just this rank's expert-DP group; every other (dense) param shards over
    # the full DP mesh. Without this the comparisons below would still pass if fully_shard had
    # quietly skipped the expert parameters.
    expert_dp_ranks = sorted(dist.get_process_group_ranks(expert_dp_group))
    expert_parameter_ids = {
        id(parameter)
        for layer in model.decoder.layers
        if isinstance(layer, MoETransformerLayer)
        for parameter in layer.mlp.experts.parameters()
    }
    num_sharded_experts = 0
    for name, parameter in model.named_parameters():
        assert isinstance(parameter, DTensor), f"param {name!r} should be a DTensor."
        is_expert = id(parameter) in expert_parameter_ids
        expected = expert_dp_ranks if is_expert else list(range(world_size))
        assert parameter.device_mesh.mesh.tolist() == expected, (
            f"param {name!r} sharded over ranks {parameter.device_mesh.mesh.tolist()}, "
            f"expected {expected}."
        )
        num_sharded_experts += is_expert
    assert num_sharded_experts > 0, "No expert parameters were sharded by mFSDP."

    # One global batch, identical on every rank; the reference sees all of it, the model its shard.
    torch.manual_seed(4321)
    ids = torch.randint(0, vocab, (global_batch, seq), dtype=torch.int64, device=device)
    pos = torch.arange(seq, dtype=torch.int64, device=device).repeat(global_batch, 1)
    mask = None  # attention layer is attn_mask_type=causal, so TE builds the causal mask itself
    target = torch.randn(global_batch, seq, vocab, device=device)
    shard = slice(rank * b_local, (rank + 1) * b_local)

    reference_losses = _train(reference, ids, pos, mask, target)
    model_losses = _train(
        model, ids[shard], pos[shard], mask, target[shard], loss_reduce_group=world, is_fsdp=True
    )

    # BF16 compute, so this is the tolerance the MCore single-grouped-weight DDP parity test
    # uses. Note this is a coarse check at BF16: a gradient scaled wrong by ep_size still lands
    # inside it over five steps, so it is a smoke test for the training loop, not a proof of
    # the reduction's scaling.
    torch.testing.assert_close(
        torch.stack(model_losses),
        torch.stack(reference_losses),
        rtol=5e-3,
        atol=5e-3,
        msg="EP=4 mFSDP model did not reproduce full-batch EP=1 training.",
    )

    # Destroy the groups this test created; leave the default (world) group for later tests.
    for group in (one, ep_group, expert_dp_group):
        dist.destroy_process_group(group)
