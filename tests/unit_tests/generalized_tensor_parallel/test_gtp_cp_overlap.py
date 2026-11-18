# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GTP sequence sharding with CP attention and DDP/Adam, including CP-changing resume.

Run with torchrun on both four and eight GPUs to cover all TP/CP/GTP layouts.
The reference uses ordinary CP and replicated weights, with identical tokens.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    GTP_CONFIG,
    GTPShardedParam,
    reset_gtp_state,
    update_gtp_config,
)
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)


@pytest.fixture(autouse=True)
def _enable_cp_attention(set_env, monkeypatch):
    # The common unit fixture disables both TE attention backends. CP requires
    # the fused backend here; restore its environment after each test.
    monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
    original = {
        name: getattr(GTP_CONFIG, name)
        for name in (
            "pad_for_alignment",
            "calculate_per_token_loss",
            "reduce_scatter_with_fp32_accumulation",
        )
    }
    update_gtp_config(reduce_scatter_with_fp32_accumulation=False)
    yield
    update_gtp_config(**original)


class _Model(torch.nn.Module):
    def __init__(
        self, config: TransformerConfig, pg: ProcessGroupCollection, num_moe_experts=None
    ) -> None:
        super().__init__()
        self.config = config
        self.share_embeddings_and_output_weights = False
        spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=num_moe_experts is not None
        )
        self.layer = spec.module(config, spec.submodules, layer_number=1, pg_collection=pg)

    def forward(self, x):
        return self.layer(x, attention_mask=None)[0]

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return self.layer.sharded_state_dict(prefix + "layer.", sharded_offsets, metadata)


def _build(
    tp,
    cp,
    weight,
    distopt,
    per_token,
    initial=None,
    num_sequence_shards=1,
    num_optimizer_instances=1,
    num_moe_experts=None,
):
    ps.destroy_model_parallel()
    reset_gtp_state()
    total_cp = cp * num_sequence_shards
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        context_parallel_size=total_cp,
        gtp_remat_size=weight,
        gtp_remat_num_sequence_shards=num_sequence_shards,
        num_distributed_optimizer_instances=num_optimizer_instances,
        expert_model_parallel_size=2 if num_moe_experts is not None else 1,
        expert_gtp_remat_size=2 if num_moe_experts is not None else 1,
    )
    pg = ProcessGroupCollection.use_mpu_process_groups()
    model_parallel_cuda_manual_seed(42)
    update_gtp_config(pad_for_alignment=0, calculate_per_token_loss=per_token)
    moe_config = (
        dict(
            num_moe_experts=num_moe_experts,
            moe_router_topk=2,
            moe_ffn_hidden_size=256,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.0,
            expert_model_parallel_size=2,
            expert_tensor_parallel_num_weight_shards=2,
        )
        if num_moe_experts is not None
        else {}
    )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=512,
        tensor_model_parallel_size=tp,
        tensor_parallel_num_weight_shards=tp * weight,
        context_parallel_size=total_cp,
        tensor_parallel_num_sequence_shards=tp * num_sequence_shards,
        sequence_parallel=tp > 1,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.fused,
        cp_comm_type="p2p",
        add_bias_linear=True,
        gradient_accumulation_fusion=False,
        calculate_per_token_loss=per_token,
        **moe_config,
    )
    assert pg.cp.size() == config.context_parallel_size
    assert pg.gtp_remat.size() == weight
    assert config.gtp_remat_num_sequence_shards == num_sequence_shards
    model = _Model(config, pg, num_moe_experts).cuda()
    if initial is not None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                value = initial[name]
                if isinstance(param, GTPShardedParam):
                    if value.shape != param.shape:
                        value = value.chunk(weight, dim=0)[pg.gtp_remat.rank()]
                param.copy_(value)
    if num_moe_experts is not None:
        for param in model.parameters():
            is_sharded = isinstance(param, GTPShardedParam)
            if not getattr(param, "allreduce", True):
                group = ps.get_expert_data_parallel_group(with_gtp_remat=not is_sharded)
            else:
                group = ps.get_data_parallel_group(
                    with_context_parallel=True, with_gtp_remat=not is_sharded
                )
            dist.broadcast(param.data, src=dist.get_process_group_ranks(group)[0], group=group)
    saved = {n: p.detach().clone() for n, p in model.named_parameters()}
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=distopt,
            overlap_grad_reduce=False,
            grad_reduce_in_fp32=True,
            num_distributed_optimizer_instances=num_optimizer_instances,
        ),
        model,
        pg_collection=pg,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=0.001,
            bf16=True,
            use_distributed_optimizer=distopt,
            use_precision_aware_optimizer=False,
            clip_grad=0.1,
        ),
        [ddp],
        pg_collection=pg,
        use_gloo_process_groups=False,
    )
    if num_moe_experts is not None:
        from megatron.core.tensor_parallel.gtp_api import classify_gtp_remat_chains

        classify_gtp_remat_chains([ddp])
    return ddp, optimizer, pg, saved


def _step(model, optimizer, pg, step):
    optimizer.zero_grad()
    model.zero_grad_buffer()
    # Every CP/TP peer of a sequence uses the same complete input before slicing.
    generator = torch.Generator(device="cuda").manual_seed(
        1000 + step * 31 + pg.dp_gtp_remat.rank()
    )
    x = torch.randn(128, 1, 256, generator=generator, device="cuda", dtype=torch.bfloat16)
    cp, cp_rank = pg.cp.size(), pg.cp.rank()
    chunks = x.reshape(2 * cp, -1, 1, 256)
    x = torch.cat((chunks[cp_rank], chunks[2 * cp - cp_rank - 1]), dim=0)
    if model.module.config.sequence_parallel:
        x = scatter_to_sequence_parallel_region(x, group=pg.tp)
    output = model(x)
    # A non-linear loss avoids a near-zero gradient through LayerNorm.
    losses = output.float().square().mean(dim=-1)
    per_token = model.module.config.calculate_per_token_loss
    loss = losses.sum() if per_token else losses.mean()
    loss.backward()
    tokens = torch.tensor(losses.numel(), device="cuda", dtype=torch.int64) if per_token else None
    finalize_model_grads([model], num_tokens=tokens, pg_collection=pg)
    gradients = {}
    if not model.ddp_config.use_distributed_optimizer:
        gradients = {n: p.main_grad.clone() for n, p in model.module.named_parameters()}
    success, norm, _ = optimizer.step()
    assert success
    if model.ddp_config.param_sync_via_bucket_group:
        model.start_param_sync(force_sync=True)
    return (
        float(norm),
        gradients,
        {n: p.detach().clone() for n, p in model.module.named_parameters()},
    )


def _assert_two_node_site_locality(pg):
    """Treat each four-GPU node as a site when requested by the test launcher."""
    if os.environ.get("GTP_TEST_TWO_SITE_SIMULATION") != "1":
        return
    assert os.environ.get("NCCL_NET") == "Socket"
    hosts = [None] * dist.get_world_size()
    dist.all_gather_object(hosts, socket.gethostname())
    assert len(hosts) == 8 and len(set(hosts)) == 2, hosts
    assert len(set(hosts[:4])) == len(set(hosts[4:])) == 1, hosts
    assert hosts[0] != hosts[4], hosts
    my_host = hosts[dist.get_rank()]
    for name in (
        "tp",
        "cp",
        "gtp_remat",
        "ep",
        "expt_gtp_remat",
        "mp",
        "tp_ep_pp_with_egtp_remat",
        "intra_dp_cp",
        "intra_expt_dp",
        "intra_dist_opt",
    ):
        ranks = dist.get_process_group_ranks(getattr(pg, name))
        assert {hosts[rank] for rank in ranks} == {my_host}, (name, ranks, hosts)
    inter_ranks = dist.get_process_group_ranks(pg.inter_dist_opt)
    assert len(inter_ranks) == 2 and {hosts[rank] for rank in inter_ranks} == set(hosts), (
        inter_ranks,
        hosts,
    )


def test_replica_all_gather_group_uses_weight_layout():
    """Optimizer AG must use the same replicas as its gradient reduction."""
    if dist.get_world_size() not in (4, 8):
        pytest.skip("Requires four or eight ranks")
    ps.destroy_model_parallel()
    try:
        ps.initialize_model_parallel(
            context_parallel_size=dist.get_world_size(),
            gtp_remat_size=2,
            gtp_remat_num_sequence_shards=2,
        )
        pg = ProcessGroupCollection.use_mpu_process_groups()
        all_gather, _ = ps.create_all_gather_groups()
        assert dist.get_process_group_ranks(all_gather) == dist.get_process_group_ranks(pg.dp_cp)
        assert pg.dp_gtp_remat.size() == 1
        assert pg.dp_cp.size() == dist.get_world_size() // 2
    finally:
        ps.destroy_model_parallel()


@pytest.mark.parametrize(
    "tp,cp,weight,num_sequence_shards",
    [(1, 2, 2, 2), (1, 1, 4, 2), (2, 2, 2, 2), (1, 2, 4, 2), (2, 1, 4, 4), (2, 2, 2, 1)],
)
@pytest.mark.parametrize("distopt", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
def test_attention_gradients_and_adam_updates(
    tp, cp, weight, num_sequence_shards, distopt, per_token
):
    world = tp * cp * weight
    if dist.get_world_size() != world:
        pytest.skip(f"Requires {world} ranks")
    try:
        reference, opt, pg, initial = _build(tp, cp * num_sequence_shards, 1, distopt, per_token)
        expected = [_step(reference, opt, pg, i) for i in range(3)]
        model, opt, pg, _ = _build(
            tp,
            cp,
            weight,
            distopt,
            per_token,
            initial=initial,
            num_sequence_shards=num_sequence_shards,
        )
        assert pg.dp_cp.size() * pg.gtp_remat.size() == pg.dp_cp_gtp_remat.size()
        for i, (expected_norm, expected_grads, expected_weights) in enumerate(expected):
            norm, grads, weights = _step(model, opt, pg, i)
            torch.testing.assert_close(norm, expected_norm, rtol=0.03, atol=1e-4)
            for name, param in model.module.named_parameters():
                target = expected_weights[name]
                grad_target = expected_grads.get(name)
                if isinstance(param, GTPShardedParam):
                    target = target.chunk(weight, dim=0)[pg.gtp_remat.rank()]
                    if grad_target is not None:
                        grad_target = grad_target.chunk(weight, dim=0)[pg.gtp_remat.rank()]
                torch.testing.assert_close(weights[name], target, rtol=0.03, atol=2e-3)
                if grad_target is not None:
                    torch.testing.assert_close(grads[name], grad_target, rtol=0.03, atol=2e-3)
    finally:
        ps.destroy_model_parallel()
        reset_gtp_state()


def test_sequence_sharding_two_optimizer_instances():
    """Two DistOpt instances preserve updates when GTP peers split a sequence."""
    if dist.get_world_size() != 8:
        pytest.skip("Requires eight ranks")
    try:
        reference, optimizer, pg, initial = _build(1, 1, 2, True, False, num_sequence_shards=2)
        expected = [_step(reference, optimizer, pg, step) for step in range(2)]

        model, optimizer, pg, _ = _build(
            1, 1, 2, True, False, initial=initial, num_sequence_shards=2, num_optimizer_instances=2
        )
        assert pg.cp.size() == 2
        assert (
            ps.get_data_parallel_group(
                with_context_parallel=True, with_gtp_remat=False, partial_data_parallel=True
            ).size()
            == 2
        )
        for step, (expected_norm, _, expected_weights) in enumerate(expected):
            norm, _, weights = _step(model, optimizer, pg, step)
            torch.testing.assert_close(norm, expected_norm, rtol=1e-4, atol=1e-5)
            for name, expected_weight in expected_weights.items():
                torch.testing.assert_close(
                    weights[name], expected_weight, rtol=1e-4, atol=1e-4, msg=name
                )
    finally:
        ps.destroy_model_parallel()
        reset_gtp_state()


def test_moe_sequence_sharding_two_optimizer_instances():
    """Two DistOpt instances preserve dense and expert updates with GTP sequence CP."""
    if dist.get_world_size() != 8:
        pytest.skip("Requires eight ranks")
    try:
        reference, optimizer, pg, initial = _build(
            1, 1, 2, True, False, num_sequence_shards=2, num_moe_experts=4
        )
        expert_names = [
            name
            for name, param in reference.module.named_parameters()
            if not getattr(param, "allreduce", True)
        ]
        assert expert_names
        assert any(
            isinstance(dict(reference.module.named_parameters())[name], GTPShardedParam)
            for name in expert_names
        )
        expected = [_step(reference, optimizer, pg, step) for step in range(2)]
        assert any(
            not torch.equal(expected[-1][2][name], initial[name]) for name in expert_names
        ), "Expert weights did not change"

        model, optimizer, pg, _ = _build(
            1,
            1,
            2,
            True,
            False,
            initial=initial,
            num_sequence_shards=2,
            num_optimizer_instances=2,
            num_moe_experts=4,
        )
        assert pg.cp.size() == 2
        assert pg.expt_gtp_remat.size() == 2
        _assert_two_node_site_locality(pg)
        for step, (expected_norm, _, expected_weights) in enumerate(expected):
            norm, _, weights = _step(model, optimizer, pg, step)
            torch.testing.assert_close(norm, expected_norm, rtol=1e-4, atol=1e-5)
            for name, expected_weight in expected_weights.items():
                torch.testing.assert_close(
                    weights[name], expected_weight, rtol=1e-4, atol=1e-4, msg=name
                )
    finally:
        ps.destroy_model_parallel()
        reset_gtp_state()


@pytest.mark.parametrize("tp", [1, 2])
def test_model_and_adam_checkpoint_after_cp_extension(tp, tmp_path_dist_ckpt):
    """CP1 -> CP2 with two GTP sequence shards preserves model and optimizer ownership."""
    from megatron.core.dist_checkpointing import ShardedTensor, load, save
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    if dist.get_world_size() != 4 * tp:
        pytest.skip(f"Requires {4 * tp} ranks")
    metadata = {"distrib_optim_sharding_type": "dp_reshardable"}

    def state(model, optimizer, pg, loading=False):
        model_state = model.module.sharded_state_dict(metadata={"dp_cp_group": pg.dp_cp_gtp_remat})
        return {
            "model": model_state,
            "optimizer": optimizer.sharded_state_dict(
                model_state, is_loading=loading, metadata=metadata
            ),
        }

    def state_tensors(value):
        if isinstance(value, ShardedTensor):
            yield value
        elif isinstance(value, dict):
            if getattr(value.get("padding"), "data", False):
                return  # Padding is reallocated, not live optimizer state.
            for child in value.values():
                yield from state_tensors(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from state_tensors(child)

    def snapshot(state_dict):
        return {
            (leaf.key, leaf.global_offset): leaf.data.clone()
            for leaf in state_tensors(state_dict)
            if leaf.data is not None
        }

    try:
        model, optimizer, pg, _ = _build(tp, 1, 2, True, False, num_sequence_shards=2)
        _step(model, optimizer, pg, 0)
        saved_state = state(model, optimizer, pg)
        expected_weights = {n: p.detach().clone() for n, p in model.module.named_parameters()}
        # Distinct moments expose accidental checkpoint writer/key aliasing.
        for leaf in state_tensors(saved_state["optimizer"]):
            if "exp_avg" in leaf.key and leaf.data is not None:
                leaf.data.fill_(dist.get_rank() + 1)
        expected = snapshot(saved_state["optimizer"])
        assert expected and any("exp_avg" in key for key, _ in expected)
        with TempNamedDir(tmp_path_dist_ckpt / f"gtpcp_tp{tp}", sync=True) as directory:
            save(saved_state, directory)
            model, optimizer, pg, _ = _build(tp, 2, 2, True, False, num_sequence_shards=2)
            restored = load(state(model, optimizer, pg, loading=True), directory)
            model.module.load_state_dict(restored["model"])
            optimizer.load_state_dict(restored["optimizer"])
            actual = snapshot(state(model, optimizer, pg)["optimizer"])
            assert actual.keys() == expected.keys()
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            for name, param in model.module.named_parameters():
                torch.testing.assert_close(param, expected_weights[name], rtol=0, atol=0)
    finally:
        ps.destroy_model_parallel()
        reset_gtp_state()
