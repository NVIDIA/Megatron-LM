# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Canonical expert FC1 checkpoints across GLU layouts and GTP partitions."""

import gc

import pytest
import torch
import torch.nn.functional as F

from megatron.core import dist_checkpointing as dcp
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.optimizer import make_sharded_optimizer_tensor
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    is_gtp_param,
    reset_gtp_state,
    update_gtp_config,
    wait_for_gtp_grad_reduction_on_current_stream,
)


@pytest.fixture(autouse=True)
def distributed():
    Utils.initialize_distributed()
    if Utils.world_size != 4:
        pytest.skip("Requires four torchrun ranks")
    yield
    torch.cuda.synchronize()
    reset_gtp_state()
    Utils.destroy_model_parallel()
    gc.collect()


def _canonical(ffn, hidden=128):
    # Construct the two semantic weights independently. Values distinguish experts,
    # channels and columns, and are small enough for the execution comparison.
    generator = torch.Generator().manual_seed(937)
    gate = torch.randn(2, ffn, hidden, generator=generator).mul_(0.035).bfloat16()
    up = torch.randn(2, ffn, hidden, generator=generator).mul_(0.065).bfloat16()
    fc2 = torch.randn(2, hidden, ffn, generator=generator).mul_(0.045).bfloat16()
    return {"linear_fc1.weight": torch.cat((gate, up), dim=1), "linear_fc2.weight": fc2}


def _external_checkpoint(canonical, directory):
    # No model factory or production GLU conversion constructs the input checkpoint.
    dcp.save(
        {
            key: ShardedTensor.from_rank_offsets(
                "experts." + key, tensor.cuda(), replica_id=Utils.rank
            )
            for key, tensor in canonical.items()
        },
        directory,
    )


def _model(ffn, degree, etp=1, alignment=1, interleave=32):
    reset_gtp_state()
    Utils.initialize_model_parallel(expert_tensor_parallel_size=etp, expert_gtp_remat_size=degree)
    update_gtp_config(pad_for_alignment=alignment, calculate_per_token_loss=False)
    model_parallel_cuda_manual_seed(1234, force_reset_rng=True)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_ffn_hidden_size=ffn,
        moe_grouped_gemm=True,
        moe_single_grouped_weight=False,
        moe_mlp_glu_interleave_size=interleave,
        expert_tensor_parallel_size=etp,
        expert_tensor_parallel_num_weight_shards=etp * degree,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bias_activation_fusion=False,
        gradient_accumulation_fusion=False,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    model = TEGroupedMLP(
        2,
        config,
        GroupedMLPSubmodules(TEColumnParallelGroupedLinear, TERowParallelGroupedLinear),
        pg_collection=pg,
    ).cuda()
    assert not model._with_fused_impl
    return model, pg


def _weights(model, pg):
    return {
        key: value
        for key, value in model.sharded_state_dict(metadata={"dp_cp_group": pg.expt_dp}).items()
        if "weight" in key
    }


def _expected(canonical, model, pg):
    """Explicit channel enumeration, independent of factory interval/count arithmetic."""
    expected = {}
    for layer in ("linear_fc1", "linear_fc2"):
        for expert, tensor in enumerate(canonical[layer + ".weight"]):
            if layer == "linear_fc1":
                gate, up = tensor.chunk(2, dim=0)
                gate = gate.chunk(pg.expt_tp.size(), dim=0)[pg.expt_tp.rank()]
                up = up.chunk(pg.expt_tp.size(), dim=0)[pg.expt_tp.rank()]
                block = model.config.moe_mlp_glu_interleave_size
                tensor = (
                    torch.cat(
                        [part for pair in zip(gate.split(block), up.split(block)) for part in pair]
                    )
                    if block is not None
                    else torch.cat((gate, up))
                )
            else:
                tensor = tensor.chunk(pg.expt_tp.size(), dim=1)[pg.expt_tp.rank()]
            param = getattr(getattr(model, layer), f"weight{expert}")
            rows = param.shape[0]
            padded = tensor.new_zeros(rows * pg.expt_gtp_remat.size(), tensor.shape[1])
            padded[: tensor.shape[0]].copy_(tensor)
            start = pg.expt_gtp_remat.rank() * rows
            expected[f"{layer}.weight{expert}"] = padded[start : start + rows].cuda()
    return expected


def _assert_parameters(model, expected):
    for name, tensor in expected.items():
        layer, weight = name.split(".")
        torch.testing.assert_close(getattr(getattr(model, layer), weight), tensor, rtol=0, atol=0)


@pytest.mark.parametrize(
    "ffn,alignment,source,target",
    [
        (96, 1, (1, 1), (1, 2)),
        (96, 1, (1, 2), (1, 1)),
        (96, 1, (1, 2), (1, 4)),
        (96, 32, (1, 2), (1, 4)),  # full-padding destination rank
        (160, 32, (1, 4), (1, 2)),  # partial-padding source rank
        (192, 1, (1, 2), (2, 2)),
        (192, 1, (2, 2), (1, 2)),
    ],
)
@pytest.mark.parametrize("target_interleave", [None, 32])
def test_canonical_disk_and_resharding(
    tmp_path_dist_ckpt, ffn, alignment, source, target, target_interleave
):
    canonical = _canonical(ffn)
    with (
        TempNamedDir(tmp_path_dist_ckpt / "external_glu", sync=True) as external,
        TempNamedDir(tmp_path_dist_ckpt / "saved_glu", sync=True) as saved,
    ):
        _external_checkpoint(canonical, external)
        model, pg = _model(ffn, source[1], source[0], alignment)
        model.load_state_dict(dcp.load(_weights(model, pg), external), strict=False)
        expected = _expected(canonical, model, pg)
        _assert_parameters(model, expected)
        identities = {name: id(param) for name, param in model.named_parameters()}
        model_state = _weights(model, pg)
        optimizer_state = {
            key: make_sharded_optimizer_tensor(
                factory, factory.data.float() + 7, "optimizer.exp_avg"
            )
            for key, factory in model_state.items()
            if key.startswith("linear_fc1.")
        }
        dcp.save({"model": model_state, "optimizer": optimizer_state}, saved)
        _assert_parameters(model, expected)
        assert identities == {name: id(param) for name, param in model.named_parameters()}
        disk = dcp.load_plain_tensors(saved)
        for name, tensor in canonical.items():
            torch.testing.assert_close(disk["experts." + name].cpu(), tensor, rtol=0, atol=0)
        moment = disk["optimizer.exp_avg.experts.linear_fc1.weight"]
        assert moment.dtype == torch.float32
        torch.testing.assert_close(
            moment.cpu(), canonical["linear_fc1.weight"].float() + 7, rtol=0, atol=0
        )
        del model, pg, expected, disk
        model, pg = _model(ffn, target[1], target[0], alignment, target_interleave)
        model.load_state_dict(dcp.load(_weights(model, pg), saved), strict=False)
        _assert_parameters(model, _expected(canonical, model, pg))


def test_checkpoint_uses_current_interleave_block(tmp_path_dist_ckpt):
    test_canonical_disk_and_resharding(tmp_path_dist_ckpt, 128, 1, (1, 2), (1, 4), 64)


@pytest.mark.parametrize("interleave", [None, 32])
def test_bf16_checkpoint_forward_backward(tmp_path_dist_ckpt, interleave):
    canonical = _canonical(96)
    with TempNamedDir(tmp_path_dist_ckpt / "execute_glu", sync=True) as directory:
        _external_checkpoint(canonical, directory)
        model, pg = _model(96, 4, interleave=interleave)
        model.load_state_dict(dcp.load(_weights(model, pg), directory), strict=False)
        for param in model.parameters():
            param.main_grad = torch.zeros(param.shape, device="cuda", dtype=torch.float32)
        x = torch.randn(128, 128, generator=torch.Generator().manual_seed(47)).cuda().bfloat16()
        x.requires_grad_()
        expected_x = x.detach().clone().requires_grad_()
        reference_weights = {key: value.cuda().requires_grad_() for key, value in canonical.items()}
        expected_outputs = []
        for expert, inputs in enumerate(expected_x.chunk(2)):
            gate, up = F.linear(inputs, reference_weights["linear_fc1.weight"][expert]).chunk(2, -1)
            expected_outputs.append(
                F.linear(F.silu(gate) * up, reference_weights["linear_fc2.weight"][expert])
            )
        expected_output = torch.cat(expected_outputs)
        expected_output.float().square().mean().backward()
        actual, _ = model(x, torch.tensor([64, 64]), torch.ones(128, device="cuda"))
        actual.float().square().mean().backward()
        wait_for_gtp_grad_reduction_on_current_stream()
        torch.testing.assert_close(actual, expected_output, rtol=0.035, atol=0.002)
        torch.testing.assert_close(x.grad, expected_x.grad, rtol=0.06, atol=2e-5)
        expected_grads = _expected(
            {key: value.grad for key, value in reference_weights.items()}, model, pg
        )
        for name, expected in expected_grads.items():
            layer, weight = name.split(".")
            param = getattr(getattr(model, layer), weight)
            grad = param.main_grad if is_gtp_param(param) else param.grad
            torch.testing.assert_close(grad.float(), expected.float(), rtol=0.08, atol=3e-5)
        with TempNamedDir(tmp_path_dist_ckpt / "execute_glu_saved", sync=True) as saved:
            dcp.save(_weights(model, pg), saved)
            with torch.no_grad():
                after_save, _ = model(
                    x.detach(), torch.tensor([64, 64]), torch.ones(128, device="cuda")
                )
            torch.testing.assert_close(after_save, actual.detach(), rtol=0, atol=0)
