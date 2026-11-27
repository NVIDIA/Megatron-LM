# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real TE storage checks for whole-parent LayerWise parameter publication."""

import inspect
import os

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear
from megatron.core.fp8_utils import (
    get_fp8_context,
    get_grouped_quantized_members,
    prepare_grouped_mxfp8_weight_,
    requantize_whole_params_from_high_precision,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS, TensorParallelMuon
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal
from megatron.training.training import (
    disable_forward_pre_hook,
    enable_forward_pre_hook,
    force_param_sync,
    wrap_model_chunks_with_ddp,
)
from tests.unit_tests.test_utilities import Utils

te = pytest.importorskip("transformer_engine.pytorch")
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

pytestmark = [
    pytest.mark.internal,
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        "single_grouped_weight" not in inspect.signature(te.GroupedLinear.__init__).parameters,
        reason="Requires the TE single grouped weight API",
    ),
]

_SHAPE = (3, 128, 256)
_MXFP8_FIELDS = ("_rowwise_data", "_columnwise_data", "_rowwise_scale_inv", "_columnwise_scale_inv")
_PARENT_FIELDS = ("rowwise_data", "columnwise_data", "scale_inv", "columnwise_scale_inv")


@pytest.fixture(autouse=True)
def _distributed(monkeypatch):
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _require_mxfp8():
    available, reason = te.is_mxfp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)


def _assert_quantized_members_equal(grouped, values):
    """Compare parent-backed views with independent, ordinary 2D TE casts."""
    assert not grouped._with_gemm_swizzled_scales
    reference_quantizer = grouped.quantizer.copy()
    reference_quantizer.set_usage(rowwise=True, columnwise=True)
    reference_quantizer.optimize_for_gemm = False
    # These dimensions require no scale padding, so every byte is meaningful.
    for member, value in zip(get_grouped_quantized_members(grouped), values, strict=True):
        assert not member._with_gemm_swizzled_scales
        reference = reference_quantizer(value.contiguous())
        for field in _MXFP8_FIELDS:
            actual = getattr(member, field)
            expected = getattr(reference, field)
            assert actual is not None and expected is not None, field
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=field)


def _assert_cached_quantized_members_live(grouped, cached_members):
    """Existing member objects must still alias the parent's published payloads and scales."""
    assert not grouped._with_gemm_swizzled_scales
    current_members = get_grouped_quantized_members(grouped)
    assert [id(member) for member in current_members] == [id(member) for member in cached_members]
    for member in cached_members:
        assert not member._with_gemm_swizzled_scales
        for member_field, parent_field in zip(_MXFP8_FIELDS, _PARENT_FIELDS, strict=True):
            member_data = getattr(member, member_field)
            parent_data = getattr(grouped, parent_field)
            assert member_data is not None and parent_data is not None, member_field
            assert (
                member_data.untyped_storage().data_ptr() == parent_data.untyped_storage().data_ptr()
            ), member_field


@torch.no_grad()
def test_grouped_mxfp8_requantization_updates_parent_storage():
    """Both payload directions must change in place, including existing cached views."""
    _require_mxfp8()
    quantizer = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=True, columnwise=True)
    grouped = GroupedTensor.make_grouped_tensor_with_shapes(
        num_tensors=_SHAPE[0],
        shapes=[_SHAPE[1:]] * _SHAPE[0],
        quantizer=quantizer,
        dtype=torch.bfloat16,
        device="cuda",
    )
    param = torch.nn.Parameter(grouped)
    prepare_grouped_mxfp8_weight_(param)
    members = get_grouped_quantized_members(param, create_if_missing=True)
    cached_ids = [id(member) for member in members]
    cached_pointers = [
        tuple(getattr(member, field).data_ptr() for field in _MXFP8_FIELDS) for member in members
    ]
    generator = torch.Generator(device="cuda").manual_seed(1863)
    source = torch.randn(_SHAPE, device="cuda", generator=generator, dtype=torch.bfloat16)
    for step in range(2):
        values = source * (0.125 if step == 0 else -4.0)
        requantize_whole_params_from_high_precision([param], [values])
        current_members = get_grouped_quantized_members(param)
        assert [id(member) for member in current_members] == cached_ids
        assert [
            tuple(getattr(member, field).data_ptr() for field in _MXFP8_FIELDS)
            for member in current_members
        ] == cached_pointers
        _assert_quantized_members_equal(param, values)


@pytest.mark.parametrize(
    "fp8_primary,preserve_init",
    [(False, True), (True, True), (True, False)],
    ids=["bf16", "mxfp8", "mxfp8-missing-init"],
)
@torch.no_grad()
def test_grouped_local_publish_preserves_master_and_updates_live_storage(
    fp8_primary, preserve_init
):
    """DP1 masters use preserved initializers or the ordinary logical-value fallback."""
    if fp8_primary:
        _require_mxfp8()
    torch.manual_seed(3701)
    with (
        torch.enable_grad(),
        te.fp8_model_init(
            enabled=fp8_primary,
            recipe=MXFP8BlockScaling(),
            preserve_high_precision_init_val=preserve_init,
        ),
    ):
        layer = te.GroupedLinear(
            _SHAPE[0],
            _SHAPE[2],
            _SHAPE[1],
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
            single_grouped_weight=True,
            use_grouped_tensor=True,
        )
    param = layer.weight
    if fp8_primary:
        # The ordinary clone/float path materializes the parameter's logical
        # BF16 values before promotion. Read independent member views as its
        # reference rather than repeating the parent's clone implementation.
        dequantized = torch.stack(
            [
                member.dequantize(dtype=torch.bfloat16).float()
                for member in get_grouped_quantized_members(param)
            ]
        )
        if preserve_init:
            initial = param.get_high_precision_init_val().to(device="cuda", dtype=torch.float32)
            assert not torch.equal(
                initial, dequantized
            ), "Initializer must distinguish lossless masters"
        else:
            assert not hasattr(param, "get_high_precision_init_val")
            initial = dequantized
    else:
        initial = param.rowwise_data.view(_SHAPE).float().clone()

    # No DP group: every rank exercises the actual single-owner publication path.
    # SGD isolates storage correctness from Muon's separately tested matrix update.
    optimizer = LayerWiseDistributedOptimizer(
        [torch.optim.SGD([param], lr=0.125)], OptimizerConfig(optimizer="sgd", bf16=True)
    )
    master = param.main_param
    assert type(master) is torch.Tensor
    assert master.dtype == torch.float32 and tuple(master.shape) == _SHAPE
    torch.testing.assert_close(master, initial, rtol=0, atol=0)
    if fp8_primary and preserve_init:
        assert param.get_high_precision_init_val() is None
    reference = torch.nn.Parameter(initial.clone())
    reference_optimizer = torch.optim.SGD([reference], lr=0.125)
    generator = torch.Generator(device="cuda").manual_seed(1902)
    for step in range(2):
        gradient = torch.randn(_SHAPE, device="cuda", generator=generator) * (step + 1)
        master.grad = gradient.clone()
        reference.grad = gradient.clone()
        reference_optimizer.step()
        assert optimizer.step_with_ready_grads()
        torch.testing.assert_close(master, reference, rtol=0, atol=0)
        if fp8_primary:
            _assert_quantized_members_equal(param, reference.to(torch.bfloat16))
        else:
            torch.testing.assert_close(
                param.rowwise_data.view(_SHAPE), reference.to(torch.bfloat16), rtol=0, atol=0
            )


def _mxfp8_grouped_wrapper(shape=_SHAPE):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=shape[0],
        moe_grouped_gemm=True,
        moe_use_grouped_tensor=True,
        moe_single_grouped_weight=True,
        fp8="e4m3",
        fp8_recipe="mxfp8",
        fp8_param=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
    )
    model_parallel_cuda_manual_seed(4131)
    # Construct a training weight with both GEMM layouts.
    with torch.enable_grad(), get_fp8_context(config, 0, is_init=True):
        return TEColumnParallelGroupedLinear(
            shape[0],
            shape[2],
            shape[1],
            config=config,
            init_method=init_method_normal(0.02),
            bias=False,
            skip_bias_add=False,
            is_expert=True,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            name="decoder.layers.0.mlp.experts.linear_fc1",
        )


@pytest.mark.parametrize("shape", [(65, 128, 256), (3, 96, 256), (3, 128, 160)])
def test_grouped_mxfp8_wrapper_shape_error_identifies_parameter(shape):
    _require_mxfp8()
    with pytest.raises(ValueError) as error:
        _mxfp8_grouped_wrapper(shape)
    assert "decoder.layers.0.mlp.experts.linear_fc1.weight" in str(error.value)
    assert str(shape) in str(error.value)


class _GroupedExpertStack(torch.nn.Module):
    """Six projections span two padded buckets even after filling four DP owners."""

    def __init__(self, config, pg_collection):
        super().__init__()
        self.config = config
        with get_fp8_context(config, 0, is_init=True):
            self.experts = torch.nn.ModuleList(
                [
                    TEColumnParallelGroupedLinear(
                        _SHAPE[0],
                        in_features,
                        out_features,
                        config=config,
                        init_method=init_method_normal(0.02),
                        bias=False,
                        skip_bias_add=False,
                        is_expert=True,
                        pg_collection=pg_collection,
                    )
                    for in_features, out_features in ((256, 128), (128, 256)) * 3
                ]
            )

    def forward(self, value, splits):
        for expert in self.experts:
            value, _ = expert(value, splits)
        return value


def test_compact_grouped_cpu_initialization_checks_live_storage_device():
    """Reject TE builds that migrate the wrapper while leaving its live data on CPU."""
    layer = te.GroupedLinear(
        _SHAPE[0],
        _SHAPE[2],
        _SHAPE[1],
        bias=False,
        params_dtype=torch.bfloat16,
        device="cpu",
        single_grouped_weight=True,
        use_grouped_tensor=True,
    )
    initial = layer.weight.rowwise_data.view(_SHAPE).float().clone()
    layer.cuda()
    param = layer.weight

    def make_optimizer():
        return LayerWiseDistributedOptimizer(
            [torch.optim.SGD([param], lr=0.125)],
            OptimizerConfig(optimizer="sgd", bf16=True),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )

    if param.rowwise_data.device != param.device:
        # Every replica must reject the broken move, including ranks that would
        # own no parent and therefore never construct an FP32 master for it.
        with pytest.raises(RuntimeError, match="use_cpu_initialization=False"):
            make_optimizer()
    else:
        # A future TE implementation may correctly migrate all grouped metadata.
        make_optimizer()
        torch.testing.assert_close(
            param.rowwise_data.view(_SHAPE).float().cpu(), initial, rtol=0, atol=0
        )
        if getattr(param, "main_param", None) is not None:
            assert param.main_param.device == param.device
            torch.testing.assert_close(param.main_param.cpu(), initial, rtol=0, atol=0)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4 or not HAVE_EMERGING_OPTIMIZERS,
    reason="Requires DP4 and emerging_optimizers",
)
@pytest.mark.parametrize("layout", ["padded", "compact"])
@pytest.mark.parametrize("overlap", [False, True])
def test_grouped_mxfp8_muon_dp4_publication(layout, overlap, monkeypatch):
    """Real GEMMs and eval/checkpoint-sync transitions preserve unswizzled storage."""
    _require_mxfp8()
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    assert pg_collection.dp_cp.size() == pg_collection.expt_dp.size() == 4
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        ffn_hidden_size=256,
        num_moe_experts=_SHAPE[0],
        moe_grouped_gemm=True,
        moe_use_grouped_tensor=True,
        moe_single_grouped_weight=True,
        fp8="e4m3",
        fp8_recipe="mxfp8",
        fp8_param=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=True,
    )
    model_parallel_cuda_manual_seed(3121)
    raw_model = _GroupedExpertStack(config, pg_collection)
    parameters = list(raw_model.parameters())
    assert len(parameters) == 6 and all(param.is_grouped_expert_weight for param in parameters)
    references = [
        [
            torch.nn.Parameter(member.to(device="cuda", dtype=torch.float32))
            for member in param.get_high_precision_init_val()
        ]
        for param in parameters
    ]
    reuse_grad_buffer = layout == "padded"
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        fp8_param_gather=True,
        overlap_param_gather=overlap,
        overlap_grad_reduce=overlap,
        reuse_grad_buf_for_mxfp8_param_ag=reuse_grad_buffer,
        grad_reduce_in_fp32=True,
        bucket_size=4096,
    )
    model = wrap_model_chunks_with_ddp(
        [raw_model],
        config,
        ddp_config,
        use_layer_wise_distributed_optimizer=True,
        layer_wise_param_layout=layout,
        pg_collection=pg_collection,
    )[0]
    optimizer_config = OptimizerConfig(
        optimizer="muon",
        bf16=True,
        lr=0.01,
        weight_decay=0.01,
        clip_grad=0.0,
        use_distributed_optimizer=True,
        use_layer_wise_distributed_optimizer=True,
        layer_wise_param_layout=layout,
        reuse_grad_buf_for_mxfp8_param_ag=reuse_grad_buffer,
        overlap_param_gather=overlap,
    )
    base_optimizer = TensorParallelMuon(parameters, lr=0.01, weight_decay=0.01)
    optimizer = LayerWiseDistributedOptimizer(
        [base_optimizer], optimizer_config, pg_collection=pg_collection, model_chunks=[model]
    )
    reference_optimizer = TensorParallelMuon(
        [member for members in references for member in members], lr=0.01, weight_decay=0.01
    )
    owners = []
    for param in parameters:
        owns_param = getattr(param, "main_param", None) is not None
        owner_count = torch.tensor(int(owns_param), device="cuda")
        torch.distributed.all_reduce(owner_count, group=pg_collection.dp_cp)
        assert owner_count.item() == 1
        owner_rank = torch.tensor(Utils.rank if owns_param else -1, device="cuda")
        torch.distributed.all_reduce(
            owner_rank, op=torch.distributed.ReduceOp.MAX, group=pg_collection.dp_cp
        )
        owners.append(owner_rank.item())
    if overlap:
        assert len(model.bucket_groups) + len(model.expert_parallel_bucket_groups) >= 2
    cached_members = [tuple(get_grouped_quantized_members(param)) for param in parameters]
    parent_pointers = [
        tuple(getattr(param, field).data_ptr() for field in _PARENT_FIELDS) for param in parameters
    ]

    generator = torch.Generator(device="cuda").manual_seed(9841 + Utils.rank)
    # Native grouped MXFP8 casts require each expert's token dimension to be
    # divisible by 128; keep unequal workloads within that supported domain.
    splits = torch.tensor([128, 256, 128], device="cuda", dtype=torch.int64)
    try:
        # Three updates cover ordinary training, train -> eval -> train, then
        # train -> eval -> checkpoint's force_param_sync -> train. A final forward
        # consumes the last update, including deferred overlap gathers.
        for step in range(4):
            value = torch.randn(
                512, 256, device="cuda", dtype=torch.bfloat16, generator=generator
            ).requires_grad_()
            eval_output = None
            if step in (1, 2):
                if overlap:
                    disable_forward_pre_hook([model], optimizer=optimizer)
                model.eval()
                with torch.no_grad(), get_fp8_context(config, 0):
                    eval_output = model(value.detach(), splits)
                # evaluate() restores train mode before checkpointing or the next step.
                model.train()
                if overlap:
                    enable_forward_pre_hook([model])
                if step == 2 and overlap:
                    force_param_sync([model], optimizer=optimizer)
                for param, members, views in zip(
                    parameters, references, cached_members, strict=True
                ):
                    _assert_cached_quantized_members_live(param, views)
                    _assert_quantized_members_equal(
                        param,
                        torch.stack([member.detach().to(torch.bfloat16) for member in members]),
                    )

            optimizer.zero_grad()
            model.zero_grad_buffer()
            if reuse_grad_buffer and overlap:
                optimizer.prepare_model_params_for_param_sync()
            with get_fp8_context(config, 0):
                output = model(value, splits)
            assert torch.isfinite(output).all(), f"{layout=}, {overlap=}, {step=}"
            if eval_output is not None:
                torch.testing.assert_close(output, eval_output, rtol=0, atol=0)
            for param, views, pointers in zip(
                parameters, cached_members, parent_pointers, strict=True
            ):
                _assert_cached_quantized_members_live(param, views)
                assert (
                    tuple(getattr(param, field).data_ptr() for field in _PARENT_FIELDS) == pointers
                )
            if step:
                for param, members in zip(parameters, references, strict=True):
                    _assert_quantized_members_equal(
                        param,
                        torch.stack([member.detach().to(torch.bfloat16) for member in members]),
                    )
            if step == 3:
                break
            upstream = torch.randn(
                output.shape, device="cuda", dtype=output.dtype, generator=generator
            )
            # Forward has completed any pending AG. Backward may swizzle local
            # GEMM operands, but must not mutate the persistent weight storage.
            published_storage = [
                tuple(getattr(param, field).clone() for field in _PARENT_FIELDS)
                for param in parameters
            ]
            output.backward(upstream)
            model.finish_grad_sync()
            for param, views, pointers, published in zip(
                parameters, cached_members, parent_pointers, published_storage, strict=True
            ):
                _assert_cached_quantized_members_live(param, views)
                assert (
                    tuple(getattr(param, field).data_ptr() for field in _PARENT_FIELDS) == pointers
                )
                for field, before in zip(_PARENT_FIELDS, published, strict=True):
                    torch.testing.assert_close(
                        getattr(param, field), before, rtol=0, atol=0, msg=field
                    )
            gradients = [param.main_grad.detach().clone() for param in parameters]
            # Padded reduction need only retain the complete gradient on its owner.
            # Broadcast that authoritative value before reuse can clear the buffer.
            for gradient, owner, members in zip(gradients, owners, references, strict=True):
                torch.distributed.broadcast(gradient, src=owner, group=pg_collection.dp_cp)
                for member, member_gradient in zip(members, gradient, strict=True):
                    member.grad = member_gradient.clone()
            reference_optimizer.step()
            success, _, _ = optimizer.step()
            assert success
            for param, members in zip(parameters, references, strict=True):
                master = getattr(param, "main_param", None)
                if master is not None:
                    torch.testing.assert_close(master, torch.stack(members), rtol=0, atol=0)
                    torch.testing.assert_close(
                        base_optimizer.state[master]["momentum_buffer"],
                        torch.stack(
                            [
                                reference_optimizer.state[member]["momentum_buffer"]
                                for member in members
                            ]
                        ),
                        rtol=0,
                        atol=0,
                    )
    finally:
        if overlap and model.remove_forward_pre_hook_handles:
            model.disable_forward_pre_hook(param_sync=False)
