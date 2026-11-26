# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DP4 numerical parity of grouped and discrete expert optimizer parameters.

Run with four GB200 GPUs. This deliberately uses a complete TransformerLayer,
including attention, learned routing, and a gated shared expert. The reference
uses the same native TE grouped GEMMs with separate expert Parameters; only the
parameter representation changes. Neither routing nor optimizer updates are mocked.

The four-step BF16 runs use a shared batch across DP ranks (new data each step)
with the default Muon precision. A separate single-step run uses distinct data
per rank and full-precision Newton-Schulz. Distinct-rank reductions can round
differently when whole-parameter ownership changes; tiny FP32 update differences
can cross BF16 weight rounding boundaries and diverge on subsequent steps. The
shared-batch control isolates representation parity without replacing gradients.
"""

import gc
import hashlib
import inspect
import math
import os
import re

import pytest
import torch
import torch.nn.functional as F

from megatron.core.dist_checkpointing import load, save
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.fp8_utils import (
    copy_tensor_to_quantized_param,
    get_fp8_context,
    get_grouped_quantized_members,
    is_grouped_mxfp8tensor,
    is_grouped_tensor,
    is_mxfp8tensor,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_te_min_version
from megatron.training.training import wrap_model_chunks_with_ddp
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

try:
    from transformer_engine.pytorch import GroupedLinear

    _TE_SINGLE_WEIGHT = all(
        name in inspect.signature(GroupedLinear.__init__).parameters
        for name in ("single_grouped_weight", "use_grouped_tensor")
    )
except ImportError:
    _TE_SINGLE_WEIGHT = False

pytestmark = [
    pytest.mark.internal,
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        not is_te_min_version("2.14.0") or not _TE_SINGLE_WEIGHT,
        reason="Single grouped weights require the TE >= 2.14 GroupedLinear API",
    ),
    pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) != 4,
        reason="This numerical reference requires torchrun --nproc-per-node=4 (DP4)",
    ),
]

_EXPERT_WEIGHT = re.compile(r"^(.*\.experts\.linear_fc[12])\.weight(\d*)$")
_STEPS = 4
_ATOL = 1.0e-6
_RTOL = 1.0e-5
_MXFP8_FIELDS = ("_rowwise_data", "_columnwise_data", "_rowwise_scale_inv", "_columnwise_scale_inv")


def _logical_tensors(name, tensor):
    """Name the same expert matrices independently of parent parameter storage."""
    match = _EXPERT_WEIGHT.fullmatch(name)
    if match and not match[2]:
        assert tensor.ndim == 3 and tensor.shape[0] == 4, (name, tensor.shape)
        return {f"{match[1]}.weight{expert}": tensor[expert] for expert in range(4)}
    return {name: tensor}


def _parameter_storage(param):
    # A BF16 GroupedTensor's ordinary Tensor payload is not its grouped GEMM storage.
    if is_grouped_tensor(param):
        return param.rowwise_data.view(param.shape)
    return param.detach()


def _is_quantized(param):
    return is_grouped_mxfp8tensor(param) or is_mxfp8tensor(param)


@torch.no_grad()
def _initialize_canonical_weights(layer):
    """Explicit, rank-independent initialization, before DDP and master creation."""
    for name, param in layer.named_parameters():
        # The optimizer must start from the same preserved BF16 initializer, not
        # from a dequantized approximation of that initializer.
        storage = (
            param.get_high_precision_init_val()
            if _is_quantized(param)
            else _parameter_storage(param)
        )
        for logical_name, target in _logical_tensors(name, storage.view(param.shape)).items():
            seed = int.from_bytes(hashlib.sha256(logical_name.encode()).digest()[:8], "little")
            generator = torch.Generator(device="cpu").manual_seed(seed)
            value = (
                torch.ones(target.shape, dtype=torch.float32)
                if target.ndim == 1
                else torch.randn(target.shape, generator=generator) * 0.02
            )
            target.copy_(value.to(device=target.device, dtype=target.dtype))
        if _is_quantized(param):
            if is_grouped_mxfp8tensor(param):
                get_grouped_quantized_members(param, create_if_missing=True)
            copy_tensor_to_quantized_param(
                param, storage.to(device=param.device, dtype=torch.bfloat16).contiguous()
            )


def _create_model(
    single_weight,
    layout,
    overlap,
    pg_collection,
    *,
    muon_fp32_matmul_prec="highest",
    optimizer_kind="muon",
    fp8_primary=False,
    reuse=False,
    distributed=True,
):
    use_layer_wise = optimizer_kind == "muon"
    assert distributed or (not use_layer_wise and not overlap and not reuse)
    if fp8_primary:
        import transformer_engine.pytorch as te

        available, reason = te.is_mxfp8_available(return_reason=True)
        if not available:
            pytest.skip(reason)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        ffn_hidden_size=512,
        num_moe_experts=4,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=256,
        moe_shared_expert_gate=True,
        moe_router_topk=2,
        moe_router_pre_softmax=True,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        # All 256 real tokens may select an expert. Capacity 256 therefore never
        # drops a token, and is valid for native MXFP8 grouped activation casts.
        moe_expert_capacity_factor=2.0 if fp8_primary else None,
        moe_pad_expert_input_to_capacity=fp8_primary,
        moe_grouped_gemm=True,
        moe_use_grouped_tensor=True,
        moe_single_grouped_weight=single_weight,
        moe_token_dispatcher_type="alltoall",
        use_transformer_engine_op_fuser=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        normalization="RMSNorm",
        add_bias_linear=False,
        add_qkv_bias=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_backend=AttnBackend.unfused,
        gradient_accumulation_fusion=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp8="e4m3" if fp8_primary else None,
        fp8_recipe="mxfp8" if fp8_primary else "delayed",
        fp8_param=fp8_primary,
        # Construct TE GroupedTensor storage on CUDA: moving a CPU-initialized wrapper
        # with .cuda() can leave its live rowwise storage on CPU in current TE versions.
        use_cpu_initialization=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)
    with get_fp8_context(config, 0, is_init=True):
        layer = TransformerLayer(
            config,
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=4, moe_grouped_gemm=True, use_te_op_fuser=False
            ),
            layer_number=1,
            pg_collection=pg_collection,
        ).cuda()
    _initialize_canonical_weights(layer)
    assert tuple(layer.self_attention.linear_qkv.weight.shape) == (768, 256)
    assert tuple(layer.mlp.shared_experts.gate_weight.shape) == (1, 256)
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=distributed,
        fp8_param_gather=fp8_primary,
        reuse_grad_buf_for_mxfp8_param_ag=reuse,
        overlap_param_gather=overlap,
        overlap_grad_reduce=overlap,
        grad_reduce_in_fp32=True,
        bucket_size=40960,
    )
    model = wrap_model_chunks_with_ddp(
        [layer],
        config,
        ddp_config,
        use_layer_wise_distributed_optimizer=use_layer_wise,
        layer_wise_param_layout=layout,
        pg_collection=pg_collection,
    )[0]
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="muon" if use_layer_wise else "adam",
            muon_scalar_optimizer="adam",
            lr=3.0e-4,
            weight_decay=0.01,
            decoupled_weight_decay=True,
            bf16=True,
            fp8_recipe="mxfp8" if fp8_primary else None,
            clip_grad=1.0,
            use_distributed_optimizer=distributed,
            use_layer_wise_distributed_optimizer=use_layer_wise,
            reuse_grad_buf_for_mxfp8_param_ag=reuse,
            layer_wise_param_layout=layout,
            overlap_param_gather=overlap,
            muon_tp_mode="duplicated",
            muon_fp32_matmul_prec=muon_fp32_matmul_prec,
        ),
        [model],
        pg_collection=pg_collection,
        use_gloo_process_groups=False,
    )
    return model, optimizer


def _optimizer_leaves(optimizer, is_muon=False):
    is_muon = is_muon or isinstance(optimizer, LayerWiseDistributedOptimizer)
    if hasattr(optimizer, "chained_optimizers"):
        for child in optimizer.chained_optimizers:
            yield from _optimizer_leaves(child, is_muon)
    elif not optimizer.is_stub_optimizer:
        yield optimizer, "muon" if is_muon else "adam"


def _owned_parameters(model, optimizer):
    """Read actual optimizer leaves, including byte-sharded Adam norm parameters."""
    names = {id(param): name for name, param in model.module.named_parameters()}
    for leaf, kind in _optimizer_leaves(optimizer):
        assert ("Muon" if kind == "muon" else "Adam") in type(leaf.optimizer).__name__
        pairs = []
        if isinstance(leaf, DistributedOptimizer):
            for model_groups, master_groups in (
                (leaf.model_float16_groups, leaf.shard_fp32_from_float16_groups),
                (leaf.model_fp32_groups, leaf.shard_fp32_groups),
            ):
                for group, masters in zip(model_groups, master_groups, strict=True):
                    pairs.extend(zip(group, masters, strict=True))
        else:
            for group, masters in zip(
                leaf.float16_groups, leaf.fp32_from_float16_groups, strict=True
            ):
                pairs.extend(zip(group, masters, strict=True))
            pairs.extend((param, param) for group in leaf.fp32_from_fp32_groups for param in group)

        actual = {id(param) for group in leaf.optimizer.param_groups for param in group["params"]}
        assert actual == {id(master) for _, master in pairs}, "Unobserved optimizer parameters"
        for param, master in pairs:
            if isinstance(leaf, DistributedOptimizer):
                span = leaf._get_model_param_range_map(param)["param"]
                start, end = span.start, span.end
            else:
                start, end = 0, param.numel()
            yield names[id(param)], param.shape, kind, start, end, master, leaf.optimizer


def _snapshot(model, optimizer, *, after_step, include_grad=True, consumed_grads=None):
    """Reconstruct owner states as logical matrices, never assume equal rank ownership."""
    replicated = not model.ddp_config.use_distributed_optimizer
    local = []
    for name, shape, kind, start, end, master, base in _owned_parameters(model, optimizer):
        assert master.dtype == torch.float32, (name, "master must be FP32")
        tensors = {"master": _parameter_storage(master).float().cpu().clone().flatten()}
        if after_step:
            if include_grad:
                gradient = master.grad if consumed_grads is None else consumed_grads[id(master)]
                assert gradient is not None, (name, "missing optimizer gradient")
                tensors["grad"] = _parameter_storage(gradient).float().cpu().clone().flatten()
            keys = ("momentum_buffer",) if kind == "muon" else ("exp_avg", "exp_avg_sq")
            for key in keys:
                assert key in base.state[master], (name, key, tuple(base.state[master]))
                tensors[key] = (
                    _parameter_storage(base.state[master][key]).float().cpu().clone().flatten()
                )
        local.append((name, tuple(shape), kind, start, end, tensors))
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local)
    parents = {}
    for rank, records in enumerate(gathered):
        for name, shape, kind, start, end, tensors in records:
            if name not in parents:
                parents[name] = {
                    "kind": kind,
                    "shape": shape,
                    "coverage": torch.zeros(math.prod(shape), dtype=torch.int32),
                    "owners": [],
                    "tensors": {key: torch.empty(math.prod(shape)) for key in tensors},
                }
            parent = parents[name]
            assert parent["kind"] == kind and parent["shape"] == shape
            if replicated and parent["owners"]:
                assert (start, end) == (0, math.prod(shape)), name
                for key, value in tensors.items():
                    torch.testing.assert_close(
                        parent["tensors"][key],
                        value,
                        rtol=0,
                        atol=0,
                        msg=f"{name}/{key}: replicated optimizer states differ across ranks",
                    )
            parent["owners"].append(rank)
            parent["coverage"][start:end] += 1
            for key, value in tensors.items():
                assert value.numel() == end - start, (name, key, value.shape, start, end)
                parent["tensors"][key][start:end] = value
    expected = {name for name, param in model.module.named_parameters() if param.requires_grad}
    assert parents.keys() == expected
    logical = {}
    for name, parent in parents.items():
        expected_coverage = torch.distributed.get_world_size() if replicated else 1
        assert torch.all(parent["coverage"] == expected_coverage), (
            name,
            parent["owners"],
            "invalid ownership",
        )
        if parent["kind"] == "muon":
            assert len(parent["owners"]) == 1, (name, "whole-parent ownership required")
        for key, value in parent["tensors"].items():
            for logical_name, matrix in _logical_tensors(name, value.view(parent["shape"])).items():
                logical.setdefault(logical_name, {})[key] = matrix
    return logical, parents


def _assert_close(actual, expected, label, *, atol=_ATOL, rtol=_RTOL):
    """Keep strict thresholds; include useful diagnostics if a BF16 path diverges."""
    actual, expected = actual.float(), expected.float()
    difference = actual - expected
    max_abs = difference.abs().max().item()
    relative_l2 = difference.norm().item() / max(expected.norm().item(), 1.0e-30)
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        msg=lambda message: (
            f"{label}: max_abs={max_abs:.9g}, relative_l2={relative_l2:.9g}\n{message}"
        ),
    )
    return max_abs


def _compare_snapshots(actual, expected, previous_actual=None, previous_expected=None):
    assert actual.keys() == expected.keys()
    maxima = {}
    for name in expected:
        assert actual[name].keys() == expected[name].keys()
        for key in expected[name]:
            difference = _assert_close(actual[name][key], expected[name][key], f"{name}/{key}")
            maxima[key] = max(maxima.get(key, 0.0), difference)
        if previous_actual is not None:
            actual_delta = actual[name]["master"] - previous_actual[name]["master"]
            expected_delta = expected[name]["master"] - previous_expected[name]["master"]
            difference = _assert_close(actual_delta, expected_delta, f"{name}/update_delta")
            maxima["delta"] = max(maxima.get("delta", 0.0), difference)
            # Every expert, attention, router, shared gate, and norm must participate.
            assert torch.count_nonzero(actual[name]["grad"]) > 0, (name, "zero gradient")
            assert torch.count_nonzero(expected[name]["grad"]) > 0, (
                name,
                "zero reference gradient",
            )
            assert torch.count_nonzero(actual_delta) > 0, (name, "zero update")
            assert torch.count_nonzero(expected_delta) > 0, (name, "zero reference update")
    return maxima


def _assert_optimizer_counts(parents, single_weight, optimizer_kind="muon"):
    expected_muon = {
        "self_attention.linear_qkv.weight": (768, 256),
        "self_attention.linear_proj.weight": (256, 256),
        "mlp.router.weight": (4, 256),
        "mlp.shared_experts.linear_fc1.weight": (512, 256),
        "mlp.shared_experts.linear_fc2.weight": (256, 256),
        "mlp.shared_experts.gate_weight": (1, 256),
    }
    for projection, shape in (("linear_fc1", (1024, 256)), ("linear_fc2", (256, 512))):
        if single_weight:
            expected_muon[f"mlp.experts.{projection}.weight"] = (4, *shape)
        else:
            for expert in range(4):
                expected_muon[f"mlp.experts.{projection}.weight{expert}"] = shape
    expected_adam = {
        "self_attention.linear_qkv.layer_norm_weight": (256,),
        "pre_mlp_layernorm.weight": (256,),
    }
    if optimizer_kind == "adam":
        expected_adam.update(expected_muon)
        expected_muon = {}
    for kind, expected in (("muon", expected_muon), ("adam", expected_adam)):
        actual = {
            name: parent["shape"] for name, parent in parents.items() if parent["kind"] == kind
        }
        assert actual == expected, (kind, actual, expected)
    counts = {
        kind: sum(parent["kind"] == kind for parent in parents.values())
        for kind in ("muon", "adam")
    }
    expected_counts = (
        {"muon": 8 if single_weight else 14, "adam": 2}
        if optimizer_kind == "muon"
        else {"muon": 0, "adam": 10 if single_weight else 16}
    )
    assert counts == expected_counts, counts
    if torch.distributed.get_rank() == 0:
        details = {
            name: {key: parent[key] for key in ("kind", "shape", "owners")}
            for name, parent in parents.items()
        }
        print(f"single_weight={single_weight}: actual optimizer Parameters={counts}; {details}")


@torch.no_grad()
def _assert_published_weights(model, snapshot):
    """Every rank must consume exactly the values published by the previous owner step."""
    quantized = {}
    for name, param in model.module.named_parameters():
        if _is_quantized(param):
            if is_grouped_mxfp8tensor(param):
                assert not param._with_gemm_swizzled_scales, name
                members = get_grouped_quantized_members(param)
                prefix = _EXPERT_WEIGHT.fullmatch(name)[1]
                logical_members = {
                    f"{prefix}.weight{i}": member for i, member in enumerate(members)
                }
                for member in members:
                    for field, parent_field in zip(
                        _MXFP8_FIELDS,
                        ("rowwise_data", "columnwise_data", "scale_inv", "columnwise_scale_inv"),
                        strict=True,
                    ):
                        actual = getattr(member, field)
                        assert actual is not None, (name, field)
                        assert (
                            actual.untyped_storage().data_ptr()
                            == getattr(param, parent_field).untyped_storage().data_ptr()
                        ), (name, field, "detached member storage")
            else:
                logical_members = {name: param}
            for logical_name, member in logical_members.items():
                quantized[logical_name] = _assert_mxfp8_published(
                    member, snapshot[logical_name]["master"], logical_name
                )
            continue
        for logical_name, value in _logical_tensors(name, _parameter_storage(param)).items():
            expected = snapshot[logical_name]["master"].to(dtype=value.dtype)
            _assert_close(
                value.cpu(), expected, f"{logical_name}/published replica", atol=0, rtol=0
            )
    if quantized:
        digest = hashlib.sha256()
        for name, fields in sorted(quantized.items()):
            digest.update(name.encode())
            for field in _MXFP8_FIELDS:
                digest.update(fields[field].numpy().tobytes())
        replicas = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(replicas, digest.hexdigest())
        assert len(set(replicas)) == 1, "Published MXFP8 payloads/scales differ across DP ranks"
    return quantized


def _assert_mxfp8_published(member, master, name):
    """Check exact payloads and persistent linear scales against the BF16 source."""
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    rows, columns = member.shape
    # Every quantized projection in this fixture is 128-aligned, including both
    # scale dimensions after blocking. No uninitialized scale padding is compared.
    assert rows % 128 == columns % 128 == 0, (name, member.shape)
    assert not member._with_gemm_swizzled_scales, name
    reference_quantizer = MXFP8Quantizer(member._fp8_dtype, rowwise=True, columnwise=True)
    reference_quantizer.optimize_for_gemm = False
    expected = reference_quantizer(master.to(device="cuda", dtype=torch.bfloat16).contiguous())
    for field in _MXFP8_FIELDS:
        actual_field, expected_field = getattr(member, field), getattr(expected, field)
        assert actual_field is not None and expected_field is not None, (name, field)
        torch.testing.assert_close(
            actual_field, expected_field, rtol=0, atol=0, msg=f"{name}/{field}"
        )

    return {field: getattr(member, field).cpu().clone() for field in _MXFP8_FIELDS}


def _compare_published_mxfp8(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        for field in _MXFP8_FIELDS:
            torch.testing.assert_close(
                actual[name][field],
                expected[name][field],
                rtol=0,
                atol=0,
                msg=f"{name}/{field}: single/discrete MXFP8 differs",
            )


def _train_step(
    model,
    optimizer,
    hidden_states,
    target,
    attention_mask,
    *,
    previous_snapshot=None,
    published_snapshots=None,
):
    routes = []
    consumed_grads = {}

    def observe_optimizer(base, _args, _kwargs):
        # A synchronous parameter publish can reuse and zero gradient storage.
        # Observe the actual clipped gradients consumed by the real base step.
        for group in base.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    consumed_grads[id(param)] = param.grad.detach().clone()

    def observe_router(_module, _inputs, output):
        probabilities, routing_map = output
        routes.append((probabilities.detach().float().cpu(), routing_map.detach().cpu()))

    hook = model.module.mlp.router.register_forward_hook(observe_router)
    optimizer_hooks = [
        leaf.optimizer.register_step_pre_hook(observe_optimizer)
        for leaf, _ in _optimizer_leaves(optimizer)
    ]
    optimizer.zero_grad()
    model.zero_grad_buffer()
    if model.ddp_config.reuse_grad_buf_for_mxfp8_param_ag and model.ddp_config.overlap_param_gather:
        optimizer.prepare_model_params_for_param_sync()
    # GEMM may create temporary swizzled scales, but must keep the persistent
    # publication buffers and their linear layout intact through backward.
    grouped_fields = ("rowwise_data", "columnwise_data", "scale_inv", "columnwise_scale_inv")
    grouped_pointers = {
        name: tuple(getattr(param, field).data_ptr() for field in grouped_fields)
        for name, param in model.module.named_parameters()
        if is_grouped_mxfp8tensor(param)
    }
    model.module.set_is_first_microbatch()
    try:
        with get_fp8_context(model.module.config, 0):
            output, _ = model(hidden_states, attention_mask=attention_mask)
        if previous_snapshot is not None:
            published = _assert_published_weights(model, previous_snapshot)
            if published_snapshots is not None:
                published_snapshots.append(published)
        loss = F.mse_loss(output.float(), target.float())
        loss.backward()
        model.finish_grad_sync()
        for name, param in model.module.named_parameters():
            assert param.main_grad.dtype == torch.float32, (name, "gradient must be FP32")
            if name in grouped_pointers:
                assert not param._with_gemm_swizzled_scales, name
                assert (
                    tuple(getattr(param, field).data_ptr() for field in grouped_fields)
                    == grouped_pointers[name]
                ), (name, "GEMM replaced persistent MXFP8 storage")
                for member in get_grouped_quantized_members(param):
                    assert not member._with_gemm_swizzled_scales, name
        success, grad_norm, _ = optimizer.step()
        assert success
        # Read gradients before the next forward can reuse their storage for param AG.
        snapshot, _ = _snapshot(model, optimizer, after_step=True, consumed_grads=consumed_grads)
    finally:
        hook.remove()
        for optimizer_hook in optimizer_hooks:
            optimizer_hook.remove()
    assert len(routes) == 1
    return (
        output.detach().float().cpu(),
        loss.detach().float().cpu(),
        grad_norm,
        routes[0],
        snapshot,
    )


@pytest.fixture
def dp4(monkeypatch):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    assert torch.distributed.get_world_size() == 4
    assert pg_collection.dp_cp.size() == pg_collection.expt_dp.size() == 4
    assert all(
        getattr(pg_collection, name).size() == 1 for name in ("tp", "pp", "cp", "ep", "expt_tp")
    )
    yield pg_collection
    gc.collect()
    Utils.destroy_model_parallel()


def _compare_transformer_training(
    dp4,
    layout,
    overlap,
    *,
    distinct_rank_batches,
    steps,
    optimizer_kind="muon",
    fp8_primary=False,
    reuse=False,
    distributed=True,
):
    model_options = {
        "muon_fp32_matmul_prec": "highest" if distinct_rank_batches else "medium",
        "optimizer_kind": optimizer_kind,
        "fp8_primary": fp8_primary,
        "reuse": reuse,
        "distributed": distributed,
    }
    reference, reference_optimizer = _create_model(False, layout, overlap, dp4, **model_options)
    candidate, candidate_optimizer = _create_model(True, layout, overlap, dp4, **model_options)
    previous_reference, reference_parents = _snapshot(
        reference, reference_optimizer, after_step=False
    )
    previous_candidate, candidate_parents = _snapshot(
        candidate, candidate_optimizer, after_step=False
    )
    _assert_optimizer_counts(reference_parents, False, optimizer_kind)
    _assert_optimizer_counts(candidate_parents, True, optimizer_kind)
    assert previous_candidate.keys() == previous_reference.keys()
    for name in previous_reference:
        _assert_close(
            previous_candidate[name]["master"],
            previous_reference[name]["master"],
            f"{name}/initial_master",
            atol=0,
            rtol=0,
        )

    attention_mask = torch.triu(torch.ones(128, 128, device="cuda", dtype=torch.bool), diagonal=1)
    attention_mask = attention_mask[None, None]
    for step in range(steps):
        # Sharing the BF16 batch controls floating-point summation order without
        # replacing any backward gradients or bypassing the native DP collectives.
        rank_seed = Utils.rank if distinct_rank_batches else 0
        generator = torch.Generator(device="cuda").manual_seed(10000 + 17 * step + rank_seed)
        hidden_states = torch.randn(
            128, 2, 256, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        target = torch.randn(128, 2, 256, generator=generator, device="cuda", dtype=torch.float32)
        reference_published, candidate_published = [], []
        reference_result = _train_step(
            reference,
            reference_optimizer,
            hidden_states.clone(),
            target,
            attention_mask,
            previous_snapshot=previous_reference,
            published_snapshots=reference_published,
        )
        candidate_result = _train_step(
            candidate,
            candidate_optimizer,
            hidden_states.clone(),
            target,
            attention_mask,
            previous_snapshot=previous_candidate,
            published_snapshots=candidate_published,
        )
        assert len(reference_published) == len(candidate_published) == 1
        _compare_published_mxfp8(candidate_published[0], reference_published[0])
        ref_output, ref_loss, ref_norm, (ref_probs, ref_routes), ref_snapshot = reference_result
        output, loss, norm, (probs, routes), snapshot = candidate_result
        assert routes.dtype == ref_routes.dtype == torch.bool
        assert torch.equal(routes, ref_routes), f"step {step}: router dispatch masks differ"
        assert torch.isfinite(probs).all() and torch.all(probs >= 0)
        real_routes, reference_real_routes = probs > 0, ref_probs > 0
        assert torch.equal(
            real_routes, reference_real_routes
        ), f"step {step}: learned top-2 router indices differ"
        assert torch.all(real_routes.sum(dim=-1) == 2)
        assert torch.all(routes[real_routes]), "A real top-2 assignment was removed from dispatch"
        if fp8_primary:
            # Capacity 256 equals the number of real tokens: dispatch pads every
            # expert to 256 entries while only top-2 entries carry nonzero scores.
            assert routes.all(), "Every expert must have its full 256-token capacity"
            assert torch.all((routes & ~real_routes).sum(dim=-1) == 2)
        else:
            assert torch.equal(routes, real_routes), "Unpadded dispatch contains dummy routes"
        counts = real_routes.sum(dim=0).to(device="cuda")
        torch.distributed.all_reduce(counts)
        assert torch.all(counts > 0), "Every expert must receive real routed tokens"
        _assert_close(probs, ref_probs, f"step {step}/router_probabilities")
        _assert_close(output, ref_output, f"step {step}/output", atol=0, rtol=0)
        _assert_close(loss, ref_loss, f"step {step}/loss", atol=1.0e-6, rtol=1.0e-4)
        _assert_close(
            torch.as_tensor(norm), torch.as_tensor(ref_norm), f"step {step}/global_grad_norm"
        )
        maxima = _compare_snapshots(snapshot, ref_snapshot, previous_candidate, previous_reference)
        if Utils.rank == 0:
            print(
                f"distinct_rank_batches={distinct_rank_batches} "
                f"optimizer={optimizer_kind} fp8_primary={fp8_primary} "
                f"distributed={distributed} layout={layout} overlap={overlap} reuse={reuse} step={step}: "
                f"loss={loss.item():.9g}; max_abs={maxima}"
            )
        previous_candidate, previous_reference = snapshot, ref_snapshot

    # The fifth shared-batch forward publishes the fourth update through overlap hooks.
    # The distinct-rank case deliberately compares one update before BF16 rounding can
    # feed back into a new forward; its synchronous gather has already published it.
    if not distinct_rank_batches:
        if reuse and overlap:
            reference_optimizer.prepare_model_params_for_param_sync()
            candidate_optimizer.prepare_model_params_for_param_sync()
        with torch.no_grad(), get_fp8_context(reference.module.config, 0):
            reference_output, _ = reference(hidden_states.clone(), attention_mask=attention_mask)
        with torch.no_grad(), get_fp8_context(candidate.module.config, 0):
            candidate_output, _ = candidate(hidden_states.clone(), attention_mask=attention_mask)
        _assert_close(candidate_output, reference_output, "final published output", atol=0, rtol=0)
    final_published = []
    for model, snapshot in ((reference, previous_reference), (candidate, previous_candidate)):
        final_published.append(_assert_published_weights(model, snapshot))
    _compare_published_mxfp8(final_published[1], final_published[0])
    for model in (reference, candidate):
        if overlap and model.remove_forward_pre_hook_handles:
            model.disable_forward_pre_hook(param_sync=False)


@pytest.mark.parametrize("layout", ["padded", "compact"])
@pytest.mark.parametrize("overlap", [False, True])
def test_muon_single_grouped_weight_transformer_dp4(dp4, layout, overlap):
    """Four native BF16/Muon steps with a shared DP batch controlling reduction rounding."""
    _compare_transformer_training(dp4, layout, overlap, distinct_rank_batches=False, steps=_STEPS)


@pytest.mark.parametrize(
    "layout,overlap,reuse",
    [
        pytest.param("padded", False, True, id="padded-sync-reuse"),
        pytest.param("padded", True, True, id="padded-overlap-reuse"),
        pytest.param("compact", False, False, id="compact-sync"),
        pytest.param("compact", True, False, id="compact-overlap"),
        pytest.param("compact", False, True, id="compact-sync-reuse"),
        pytest.param("compact", True, True, id="compact-overlap-reuse"),
    ],
)
def test_muon_single_grouped_weight_mxfp8_transformer_dp4(dp4, layout, overlap, reuse):
    """All legal MXFP8 LayerWise policies run real routed MoE forward/backward updates."""
    _compare_transformer_training(
        dp4,
        layout,
        overlap,
        distinct_rank_batches=False,
        steps=_STEPS,
        fp8_primary=True,
        reuse=reuse,
    )


@pytest.mark.parametrize("fp8_primary", [False, True], ids=["bf16", "mxfp8"])
@pytest.mark.parametrize("overlap", [False, True], ids=["sync", "overlap"])
def test_adamw_single_grouped_weight_transformer_dp4(dp4, fp8_primary, overlap):
    """Classic DistributedOptimizer shards the full grouped AdamW parent, including moments."""
    _compare_transformer_training(
        dp4,
        "padded",
        overlap,
        distinct_rank_batches=False,
        steps=_STEPS,
        optimizer_kind="adam",
        fp8_primary=fp8_primary,
        reuse=fp8_primary,
    )


@pytest.mark.parametrize("fp8_primary", [False, True], ids=["bf16", "mxfp8"])
def test_adamw_single_grouped_weight_replicated_transformer_dp4(dp4, fp8_primary):
    """Replicated AdamW updates every rank's full parent without parameter all-gather."""
    _compare_transformer_training(
        dp4,
        "padded",
        False,
        distinct_rank_batches=False,
        steps=_STEPS,
        optimizer_kind="adam",
        fp8_primary=fp8_primary,
        distributed=False,
    )


def test_muon_single_grouped_weight_distinct_rank_step_dp4(dp4):
    """Real distinct-rank inputs and strict one-step FP32 optimizer-state comparison."""
    _compare_transformer_training(dp4, "padded", False, distinct_rank_batches=True, steps=1)


def _checkpoint_state(model, optimizer, *, is_loading=False):
    # DDP inherits MegatronModule.sharded_state_dict, which prefixes these keys with
    # "module."; its load_state_dict forwards to the unwrapped layer. Keep both sides
    # in the unwrapped model's namespace, as the training checkpoint path does.
    model_state = model.module.sharded_state_dict()
    return {
        "model": model_state,
        "optimizer": optimizer.sharded_state_dict(
            model_state,
            is_loading=is_loading,
            # Model-space Adam shards are needed when the buffer layout changes.
            metadata={"distrib_optim_sharding_type": "fully_reshardable"},
        ),
    }


@pytest.mark.parametrize("target_single", [True, False], ids=["single", "discrete"])
def test_muon_single_grouped_weight_checkpoint_resume_dp4(dp4, tmp_path_dist_ckpt, target_single):
    """Restore real tensors into compact ownership, optionally splitting grouped parents."""
    source, source_optimizer = _create_model(True, "padded", False, dp4)
    attention_mask = torch.triu(torch.ones(128, 128, device="cuda", dtype=torch.bool), diagonal=1)[
        None, None
    ]

    def batch(step):
        generator = torch.Generator(device="cuda").manual_seed(20000 + 17 * step + Utils.rank)
        hidden_states = torch.randn(
            128, 2, 256, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        target = torch.randn(128, 2, 256, generator=generator, device="cuda")
        return hidden_states, target

    for step in range(2):
        hidden_states, target = batch(step)
        _train_step(source, source_optimizer, hidden_states, target, attention_mask)
    before_save, source_parents = _snapshot(
        source, source_optimizer, after_step=True, include_grad=False
    )
    _assert_optimizer_counts(source_parents, True)

    with TempNamedDir(tmp_path_dist_ckpt / "muon_single_weight_padded_to_compact") as checkpoint:
        save(_checkpoint_state(source, source_optimizer), checkpoint)
        restored, restored_optimizer = _create_model(target_single, "compact", False, dp4)
        # _create_model deliberately uses canonical initializers, so a different seed alone
        # would not establish that load replaced the destination. Poison both live weights
        # and optimizer masters explicitly before constructing the loading state dictionary.
        with torch.no_grad():
            for param in restored.module.parameters():
                _parameter_storage(param).add_(0.125)
            for _, _, _, _, _, master, _ in _owned_parameters(restored, restored_optimizer):
                master.add_(0.25)
        poisoned, _ = _snapshot(restored, restored_optimizer, after_step=False)
        assert all(
            not torch.equal(poisoned[name]["master"], before_save[name]["master"])
            for name in before_save
        )
        loaded = load(_checkpoint_state(restored, restored_optimizer, is_loading=True), checkpoint)
        # TE's ordinary load pre-hook merges the indexed expert weights/extra states
        # from DCP into its single grouped parent; do not manually rewrite that layout.
        restored.module.load_state_dict(loaded["model"])
        restored_optimizer.load_state_dict(loaded["optimizer"])
        after_load, restored_parents = _snapshot(
            restored, restored_optimizer, after_step=True, include_grad=False
        )
        _assert_optimizer_counts(restored_parents, target_single)
        assert after_load.keys() == before_save.keys()
        for name in before_save:
            assert after_load[name].keys() == before_save[name].keys()
            for key in before_save[name]:
                _assert_close(
                    after_load[name][key],
                    before_save[name][key],
                    f"{name}/restored_{key}",
                    atol=0,
                    rtol=0,
                )
        for name, param in restored.module.named_parameters():
            for logical_name, value in _logical_tensors(name, _parameter_storage(param)).items():
                _assert_close(
                    value.cpu(),
                    before_save[logical_name]["master"].to(dtype=value.dtype),
                    f"{logical_name}/restored_model_weight",
                    atol=0,
                    rtol=0,
                )

        hidden_states, target = batch(2)
        uninterrupted_result = _train_step(
            source, source_optimizer, hidden_states.clone(), target, attention_mask
        )
        resumed_result = _train_step(
            restored, restored_optimizer, hidden_states.clone(), target, attention_mask
        )
        ref_output, ref_loss, ref_norm, (ref_probs, ref_routes), ref_snapshot = uninterrupted_result
        output, loss, norm, (probs, routes), snapshot = resumed_result
        assert torch.equal(routes, ref_routes), "Resumed learned router indices differ"
        _assert_close(probs, ref_probs, "resumed router probabilities")
        _assert_close(output, ref_output, "resumed output", atol=0, rtol=0)
        _assert_close(loss, ref_loss, "resumed loss", atol=1.0e-6, rtol=1.0e-4)
        _assert_close(torch.as_tensor(norm), torch.as_tensor(ref_norm), "resumed global_grad_norm")
        maxima = _compare_snapshots(snapshot, ref_snapshot, after_load, before_save)
        if Utils.rank == 0:
            print(
                f"single grouped weight DCP padded -> compact target_single={target_single} "
                f"resume: max_abs={maxima}"
            )
