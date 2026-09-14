# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 contiguous THD CP layout, SWA, compression and shared attention parity.

Run distributed cases with torchrun on 2 or 4 ranks. Native projections retain
production attention, RoPE, boundary communication and Hybrid/mHC execution;
CPU hosts use Gloo. CUDA sparse kernels are covered separately below.
"""

import copy
import gc
import inspect
import os
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import fields, replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import pretrain_hybrid as hybrid_entry
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineDataIterator,
    PipelinePayload,
    PipelinePayloadPlan,
)
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_with_interleaving,
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.pipeline_parallel.typed_p2p_communication import _pack_header, _unpack_header
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import mappings
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import transformer_config as config_runtime
from megatron.core.transformer.cuda_graphs import _set_capture_end, _set_capture_start
from megatron.core.transformer.experimental_attention_variant import csa2 as csa2_runtime
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CSA2State,
    apply_csa2_thd_rope,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
    CSA2CandidateBlocks,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_context_parallel import (
    build_csa2_cp_compression_layout,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_cuda_graph import (
    CSA2CudaGraphAdapter,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_hybrid_adapter import (
    CSA2HybridAdapter,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_indexer import (
    prepare_csa2_indexer_inputs,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_pipeline import (
    CSA2PipelinePayload,
    build_csa2_pipeline_plan,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    build_csa2_thd_layout,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.hyper_connection import SinglePassMHCState
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.pipeline_parallel.test_typed_pipeline import _Timers
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant import (
    test_csa2_cuda_graph as graph_tests,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2 import (
    _complex_rotation,
    _CPUFrequencyTable,
    _packed,
    _require_sparse_kernels,
    _RMSNorm,
    _rotary_module,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_cuda_graph import (
    cpu_graph_slots as cpu_graph_slots,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_distributed_pipeline import (
    _model as _pipeline_model,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_distributed_pipeline import (
    _reference_parameter_name,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_pipeline import _FFN
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_pipeline import (
    _config as _pipeline_config,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_pipeline import (
    _pattern as _pipeline_pattern,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_recompute import (
    cpu_checkpoint_rng as cpu_checkpoint_rng,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_recompute import (
    native_attention as native_attention,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv41 import _make_config


@pytest.fixture(autouse=True)
def native_packing_dependency(monkeypatch):
    """Native linears do not use the TE THD implementation covered by its version gate."""
    if not HAVE_TE:
        original = config_runtime.is_te_min_version
        monkeypatch.setattr(
            config_runtime,
            "is_te_min_version",
            lambda version, *args, **kwargs: version == "2.9.0"
            or original(version, *args, **kwargs),
        )


class _LayoutGroup:
    """Rank metadata for layout tests; no collectives are replaced by this object."""

    def __init__(self, size, rank):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _swa_config(cp_size=1, dtype=torch.float32, **options):
    values = dict(
        params_dtype=dtype,
        num_layers=4,
        csa_compress_ratios=[0] * 4,
        csa2_kv_source_layers=[],
        csa2_index_source_layers=[],
        csa2_candidate_source_layer=None,
        csa2_candidate_topk_blocks=0,
        csa2_candidate_block_size=0,
        context_parallel_size=cp_size,
        cp_partition_mode="contiguous",
        sequence_packing_scheduler="dp_balanced",
        moe_token_dispatcher_type="alltoall",
        dsa_indexer_loss_coeff=0,
        use_fused_mhc=False,
        bias_dropout_fusion=False,
    )
    values.update(options)
    return _make_config(**values)


def _batch(case="ragged", device="cpu"):
    if case == "ragged":
        return _packed([3, 0, 7, 0, 9], [5, 0, 9, 6, 10], tail=2, device=device)
    if case == "all-padding":
        return _packed([0], [32], device=device)
    if case == "no-sequences":
        return _packed([], [], tail=32, device=device)
    return _packed([32], [32], device=device)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("case", ["ragged", "all-padding", "no-sequences"])
def test_cp_layout_matches_global_sequence_positions(cp_size, case):
    params, _, valid = _batch(case)
    local_rows = valid.numel() // cp_size
    expected_ids, expected_positions = [], []
    for sequence, (logical, physical) in enumerate(
        zip(params.cu_seqlens_q.diff().tolist(), params.cu_seqlens_q_padded.diff().tolist())
    ):
        expected_ids.extend([sequence] * physical)
        expected_positions.extend([i if i < logical else 0 for i in range(physical)])
    tail = valid.numel() - len(expected_ids)
    expected_ids.extend([-1] * tail)
    expected_positions.extend([0] * tail)
    params.cp_partition_mode = "contiguous"
    for rank in range(cp_size):
        group = _LayoutGroup(cp_size, rank)
        layout = build_csa2_thd_layout(params, local_rows, cp_group=group)
        rows = slice(rank * local_rows, (rank + 1) * local_rows)
        torch.testing.assert_close(layout.sequence_ids, torch.tensor(expected_ids[rows]))
        torch.testing.assert_close(layout.position_ids, torch.tensor(expected_positions[rows]))
        torch.testing.assert_close(layout.valid_tokens, valid[rows])
        assert layout.global_start == rank * local_rows
        assert layout.global_total_tokens == valid.numel()
        explicit = replace(params, local_cp_size=cp_size, cp_group=group)
        layout.validate_compatible(explicit, local_rows)
    if case == "no-sequences":
        params.max_seqlen_q = params.max_seqlen_kv = 1
    else:
        params.cu_seqlens_q_padded.zero_()
    with pytest.raises(ValueError, match="layout|length|sequence"):
        layout.validate_compatible(params, local_rows, group)


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cp_token_rope_matches_complex_rotation(cp_size, dtype):
    params, _, valid = _batch()
    params.cp_partition_mode = "contiguous"
    rows = valid.numel() // cp_size
    rotary = _CPUFrequencyTable(8)
    config = _swa_config(cp_size, dtype)
    for rank in range(cp_size):
        group = _LayoutGroup(cp_size, rank)
        layout = build_csa2_thd_layout(params, rows, cp_group=group)
        torch.manual_seed(517 + rank)
        x = torch.randn(rows, 1, 2, 16, dtype=dtype)
        x[~layout.valid_tokens] = float("nan")
        x.requires_grad_()
        reference_x = x.detach().clone().requires_grad_()
        output = apply_csa2_thd_rope(x, rotary, config, layout, group)
        expected = _complex_rotation(
            reference_x, 8, layout.position_ids, layout.valid_tokens, rotary
        )
        tolerance = (
            dict(atol=2e-6, rtol=2e-6) if dtype == torch.float32 else dict(atol=3e-2, rtol=3e-2)
        )
        torch.testing.assert_close(output, expected, **tolerance)
        probe = torch.randn_like(output)
        output.backward(probe)
        expected.backward(probe)
        torch.testing.assert_close(x.grad, reference_x.grad, **tolerance)
        assert torch.isfinite(x.grad).all()


def test_cp_layout_rejects_reuse_on_another_rank_and_wrong_partition():
    params, _, _ = _batch("long")
    params.cp_partition_mode = "contiguous"
    first = build_csa2_thd_layout(params, 16, cp_group=_LayoutGroup(2, 0))
    second = build_csa2_thd_layout(params, 16, cp_group=_LayoutGroup(2, 1))
    with pytest.raises(ValueError, match="different THD layout"):
        first.validate_layout(second)
    assert first.for_compression(2).total_tokens == 32
    with pytest.raises(ValueError, match="contiguous"):
        build_csa2_thd_layout(
            replace(params, cp_partition_mode="zigzag"), 16, cp_group=_LayoutGroup(2, 0)
        )


def test_cp_configuration_rejects_unimplemented_shared_state_combinations():
    _swa_config(2)
    _make_config(
        context_parallel_size=2,
        cp_partition_mode="contiguous",
        sequence_packing_scheduler="dp_balanced",
        moe_token_dispatcher_type="alltoall",
    )
    _swa_config(2, pipeline_model_parallel_size=2, pipeline_dtype=torch.float32)
    _cp_graph_config(2)


@pytest.fixture(scope="module", params=[2, 4], ids=["cp2", "cp4"])
def cp_groups(request):
    size = request.param
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < size or world_size % size:
        pytest.skip(f"Requires torchrun with a world size divisible by CP={size}")
    if torch.cuda.is_available():
        Utils.initialize_model_parallel(context_parallel_size=size)
        model_parallel_cuda_manual_seed(172)
        try:
            yield ProcessGroupCollection.use_mpu_process_groups(), torch.device(
                "cuda", torch.cuda.current_device()
            )
        finally:
            Utils.destroy_model_parallel()
        return
    if not dist.is_initialized():
        dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    rank = dist.get_rank()
    singletons = [dist.new_group([i], timeout=timedelta(seconds=90)) for i in range(world_size)]
    contexts = [
        dist.new_group(list(range(i, i + size)), timeout=timedelta(seconds=90))
        for i in range(0, world_size, size)
    ]
    groups = ProcessGroupCollection()
    groups.tp = groups.pp = singletons[rank]
    groups.cp = groups.dp_cp = contexts[rank // size]
    try:
        yield groups, torch.device("cpu")
    finally:
        dist.barrier()
        dist.destroy_process_group(groups.cp)
        dist.destroy_process_group(groups.tp)


def _stack(config, groups, attention):
    return HybridStack(
        config,
        HybridStackSubmodules(
            forward_adapter=CSA2HybridAdapter,
            dsa_layer=ModuleSpec(
                TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=_RMSNorm,
                    self_attention=attention,
                    self_attn_bda=get_bias_dropout_add,
                ),
            ),
            moe_layer=ModuleSpec(
                TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=_RMSNorm, mlp=_FFN, mlp_bda=get_bias_dropout_add
                ),
            ),
        ),
        layer_type_list=list("DEDE"),
        post_layer_norm=False,
        pg_collection=groups,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", ["ragged", "long", "all-padding", "no-sequences"])
def test_swa_cp_matches_full_attention_forward_backward(
    cp_groups,
    native_attention,
    dtype,
    case,
    backend="none",
    rope_fusion=False,
    explicit_group=False,
):
    groups, device = cp_groups
    options = dict(dsa_kernel_backend=backend, apply_rope_fusion=rope_fusion)
    if backend == "cudnn":
        options.update(v_head_dim=512, num_attention_heads=64)
    config = _swa_config(groups.cp.size(), dtype, **options)
    reference_groups = copy.copy(groups)
    reference_groups.cp = groups.tp
    torch.manual_seed(613)
    reference = native_attention(
        replace(
            config, context_parallel_size=1, dsa_kernel_backend="none", apply_rope_fusion=False
        ),
        1,
        reference_groups,
    ).to(device)
    actual_groups = copy.copy(reference_groups if explicit_group else groups)
    actual = native_attention(config, 1, actual_groups).to(device)
    actual.load_state_dict(reference.state_dict())
    if rope_fusion:
        for module in (reference, actual):
            module.rotary_pos_emb = _rotary_module(module.config, False, module.pg_collection.cp)
    saved_norm_outputs = []
    actual.kv_layernorm.register_forward_hook(
        lambda module, args, output: saved_norm_outputs.append((output, output.detach().clone()))
    )
    params, _, valid = _batch(case, device)
    params.cp_partition_mode = "contiguous"
    local_params = replace(params, cp_group=groups.cp) if explicit_group else params
    local_rows = valid.numel() // groups.cp.size()
    start = groups.cp.rank() * local_rows
    rows = slice(start, start + local_rows)
    pending = []
    for _ in range(2):
        full_x = torch.randn(valid.numel(), 1, config.hidden_size, device=device, dtype=dtype)
        full_x[~valid] = float("nan")
        reference_x = full_x.detach().clone().requires_grad_()
        local_x = full_x[rows].detach().clone().requires_grad_()
        expected, _ = reference(
            reference_x.masked_fill(~valid[:, None, None], 0), None, packed_seq_params=params
        )
        state = CSA2State()
        output, _ = actual(local_x, None, packed_seq_params=local_params, csa2_state=state)
        assert actual.pg_collection.cp is (groups.tp if explicit_group else groups.cp)
        tolerance = (
            dict(atol=3e-6, rtol=3e-5) if dtype == torch.float32 else dict(atol=3e-3, rtol=3e-2)
        )
        torch.testing.assert_close(output, expected[rows], **tolerance)
        assert state.thd_layout.global_start == start and state.last_layer == 0
        assert torch.isfinite(output).all()
        probe = torch.randn_like(expected)
        pending.append((output, expected, probe, local_x, reference_x))
    for tensor, before_rope in saved_norm_outputs:
        torch.testing.assert_close(tensor, before_rope, rtol=0, atol=0)
    for output, expected, probe, local_x, reference_x in reversed(pending):
        output.backward(probe[rows])
        expected.backward(probe)
        torch.testing.assert_close(local_x.grad, reference_x.grad[rows], **tolerance)
        assert torch.isfinite(local_x.grad).all()
    for (name, parameter), ref_parameter in zip(actual.named_parameters(), reference.parameters()):
        assert parameter.grad is not None, name
        dist.all_reduce(parameter.grad, group=groups.cp)
        if dtype == torch.bfloat16:
            # Local and halo KV gradients round separately before native BF16
            # weight-gradient GEMMs and accumulation. Use a normwise BF16 bound;
            # the FP32 cases check every element without this rounding allowance.
            difference = (parameter.grad.float() - ref_parameter.grad.float()).norm()
            bound = 2 * torch.finfo(dtype).eps * ref_parameter.grad.float().norm()
            assert difference <= bound + 1e-6, f"{name}: gradient error {difference} > {bound}"
        else:
            torch.testing.assert_close(
                parameter.grad,
                ref_parameter.grad,
                **tolerance,
                msg=lambda message: f"{name}: {message}",
            )


@pytest.mark.parametrize("case", ["ragged", "long"])
def test_fused_swa_cp_matches_native_reference(cp_groups, native_attention, case):
    _require_sparse_kernels()
    test_swa_cp_matches_full_attention_forward_backward(
        cp_groups, native_attention, torch.bfloat16, case, backend="cudnn", rope_fusion=True
    )


def test_packed_cp_group_overrides_module_fallback(cp_groups, native_attention):
    test_swa_cp_matches_full_attention_forward_backward(
        cp_groups, native_attention, torch.float32, "ragged", explicit_group=True
    )


@pytest.mark.parametrize("mhc", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_cp_hybrid_stack_preserves_local_mhc_and_norm_recompute(
    cp_groups, native_attention, cpu_checkpoint_rng, mhc, recompute, compress_ratio=0
):
    groups, device = cp_groups
    config = _swa_config(groups.cp.size(), enable_hyper_connections=mhc)
    if compress_ratio:
        first_ratio, last_ratio = (
            compress_ratio
            if isinstance(compress_ratio, tuple)
            else (compress_ratio, compress_ratio)
        )
        config = replace(
            config,
            csa_compress_ratios=[first_ratio, 0, last_ratio, 0],
            csa2_kv_source_layers=[0, 2],
            csa2_index_source_layers=[0, 2],
        )
    reference_groups = copy.copy(groups)
    reference_groups.cp = groups.tp
    torch.manual_seed(782)
    reference = _stack(
        replace(config, context_parallel_size=1), reference_groups, native_attention
    ).to(device)
    if recompute:
        config = replace(
            config,
            recompute_granularity="selective",
            recompute_modules=["mla_up_proj", "layernorm"],
        )
    actual = _stack(config, groups, native_attention).to(device)
    actual.load_state_dict(reference.state_dict())
    params, _, valid = _batch(device=device)
    params.cp_partition_mode = "contiguous"
    local_rows = valid.numel() // groups.cp.size()
    rows = slice(groups.cp.rank() * local_rows, (groups.cp.rank() + 1) * local_rows)
    reference_x = torch.randn(32, 1, config.hidden_size, device=device, requires_grad=True)
    local_x = reference_x[rows].detach().clone().requires_grad_()
    expected = reference(reference_x, None, packed_seq_params=params)
    output = actual(local_x, None, packed_seq_params=params)
    torch.testing.assert_close(output, expected[rows], atol=5e-6, rtol=5e-5)
    probe = torch.randn_like(expected)
    output.backward(probe[rows])
    expected.backward(probe)
    torch.testing.assert_close(local_x.grad, reference_x.grad[rows], atol=5e-6, rtol=5e-5)
    for (name, parameter), ref_parameter in zip(actual.named_parameters(), reference.parameters()):
        if ref_parameter.grad is None:
            assert parameter.grad is None, name
            continue
        assert parameter.grad is not None, name
        dist.all_reduce(parameter.grad, group=groups.cp)
        torch.testing.assert_close(
            parameter.grad, ref_parameter.grad, atol=5e-6, rtol=5e-5, msg=name
        )


# Full compression and shared-key communication.


def _full_config(cp_size, ratio, dtype=torch.float32, **options):
    values = dict(
        csa_compress_ratios=[ratio, 0, ratio, 0],
        csa2_kv_source_layers=[0, 2],
        csa2_index_source_layers=[0, 2],
    )
    values.update(options)
    return _swa_config(cp_size, dtype, **values)


def _compression_batch(case, device="cpu"):
    if case == "boundary":
        # r2 groups at [7,8], [15,16] and [23,24] straddle CP=2/4 cuts.
        return _packed([1, 30], [1, 31], device=device)
    if case == "short":
        return _packed([1, 1, 1, 1], [3, 5, 7, 17], device=device)
    return _batch(case, device)


def _assert_layout(actual, expected):
    for field in fields(expected):
        value, reference = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, reference, atol=0, rtol=0, msg=field.name)
        else:
            assert value == reference, field.name


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("case", ["boundary", "ragged", "short", "all-padding", "no-sequences"])
def test_compression_layout_selects_unique_group_owners(cp_size, ratio, case):
    params, _, valid = _compression_batch(case)
    params.cp_partition_mode = "contiguous"
    full = build_csa2_thd_layout(params, valid.numel()).for_compression(ratio)
    local_rows = valid.numel() // cp_size
    layouts, rank_sources = [], []
    for rank in range(cp_size):
        token_layout = build_csa2_thd_layout(
            params, local_rows, cp_group=_LayoutGroup(cp_size, rank)
        )
        layout = build_csa2_cp_compression_layout(token_layout, ratio, 4)
        _assert_layout(layout.global_layout, full)
        global_sources = layout.local.source_indices + rank * local_rows - 4
        rank_sources.append(global_sources)
        layouts.append(layout)
        assigned = layout.local.valid_groups
        assert (
            (global_sources[assigned] >= rank * local_rows - 4)
            & (global_sources[assigned] < (rank + 1) * local_rows)
        ).all()
    gathered_sources = torch.cat(rank_sources)
    output = layouts[0].to_sequence_major(gathered_sources)
    # Each sequence is grouped independently. Complete groups must come from
    # the rank owning their final token, including when the first token is remote.
    expected_sources = torch.zeros_like(output)
    offset, compressed_row = 0, 0
    for logical, physical in zip(
        params.cu_seqlens_q.diff().tolist(), params.cu_seqlens_q_padded.diff().tolist()
    ):
        for start in range(0, physical - ratio + 1, ratio):
            if start + ratio <= logical:
                expected_sources[compressed_row] = torch.arange(
                    offset + start, offset + start + ratio
                )
                owner = (offset + start + ratio - 1) // local_rows
                assert (
                    layouts[0].gather_indices[compressed_row] // layouts[0].local.capacity == owner
                )
            compressed_row += 1
        offset += physical
    torch.testing.assert_close(output, expected_sources, atol=0, rtol=0)
    for layout in layouts[1:]:
        torch.testing.assert_close(layout.gather_indices, layouts[0].gather_indices)


def test_cp_full_configuration_accepts_shared_supervision():
    for ratio in (1, 2):
        _full_config(2, ratio)
        _full_config(2, ratio, dsa_indexer_loss_coeff=0.1)
        _full_config(2, ratio, csa2_kv_source_layers=[0])


def _full_modules(groups, device, native_attention, ratio, dtype, **options):
    config = _full_config(groups.cp.size(), ratio, dtype, **options)
    reference_groups = copy.copy(groups)
    reference_groups.cp = groups.tp
    torch.manual_seed(891)
    reference = native_attention(
        replace(
            config, context_parallel_size=1, dsa_kernel_backend="none", apply_rope_fusion=False
        ),
        1,
        reference_groups,
    ).to(device)
    actual = native_attention(config, 1, groups).to(device)
    actual.load_state_dict(reference.state_dict())
    if config.apply_rope_fusion:
        for module in (reference, actual):
            module.rotary_pos_emb = _rotary_module(module.config, True, module.pg_collection.cp)
            module.core_attention.rotary_pos_emb = module.rotary_pos_emb
    return reference, actual


def _check_parameter_gradients(actual, reference, group, dtype, *, atol=4e-6):
    for (name, parameter), ref_parameter in zip(actual.named_parameters(), reference.parameters()):
        if ref_parameter.grad is None:
            assert parameter.grad is None, name
            continue
        assert parameter.grad is not None, name
        dist.all_reduce(parameter.grad, group=group)
        if dtype == torch.bfloat16:
            error = (parameter.grad.float() - ref_parameter.grad.float()).norm()
            bound = 2 * torch.finfo(dtype).eps * ref_parameter.grad.float().norm()
            assert error <= bound + 1e-6, f"{name}: {error} > {bound}"
        else:
            torch.testing.assert_close(
                parameter.grad,
                ref_parameter.grad,
                atol=atol,
                rtol=4e-5,
                msg=lambda message: f"{name}: {message}",
            )


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", ["boundary", "ragged", "short", "all-padding", "no-sequences"])
def test_full_cp_matches_attention_and_gradients(
    cp_groups, native_attention, monkeypatch, ratio, dtype, case, fused=False
):
    groups, device = cp_groups
    options = {}
    if fused:
        options = dict(
            dsa_kernel_backend="cudnn",
            apply_rope_fusion=True,
            num_attention_heads=64,
            v_head_dim=512,
            dsa_indexer_n_heads=32,
            dsa_indexer_head_dim=128,
        )
    reference, actual = _full_modules(groups, device, native_attention, ratio, dtype, **options)
    params, _, valid = _compression_batch(case, device)
    params.cp_partition_mode = "contiguous"
    local_rows = valid.numel() // groups.cp.size()
    rows = slice(groups.cp.rank() * local_rows, (groups.cp.rank() + 1) * local_rows)
    gather_shapes = []
    original_gather = csa2_runtime.async_gather_from_sequence_parallel_region

    def record_gather(tensor, **kwargs):
        gather_shapes.append(tensor.shape)
        return original_gather(tensor, **kwargs)

    monkeypatch.setattr(csa2_runtime, "async_gather_from_sequence_parallel_region", record_gather)
    saved = []
    for norm in (actual.core_attention.compressor.norm, actual.core_attention.indexer.k_norm):
        norm.register_forward_hook(
            lambda module, args, output: saved.append((output, output.detach().clone()))
        )
    pending = []
    tolerance = dict(atol=4e-6, rtol=4e-5) if dtype == torch.float32 else dict(atol=3e-3, rtol=3e-2)
    for _ in range(2):
        full_x = torch.randn(
            valid.numel(), 1, actual.config.hidden_size, device=device, dtype=dtype
        )
        full_x[~valid] = float("nan")
        ref_x = full_x.detach().clone().requires_grad_()
        local_x = full_x[rows].detach().clone().requires_grad_()
        state, reference_state = CSA2State(), CSA2State()
        expected = reference(
            ref_x.masked_fill(~valid[:, None, None], 0),
            None,
            packed_seq_params=params,
            csa2_state=reference_state,
        )[0]
        output = actual(local_x, None, packed_seq_params=params, csa2_state=state)[0]
        torch.testing.assert_close(output, expected[rows], **tolerance)
        _assert_layout(state.compressed_layout, reference_state.compressed_layout)
        torch.testing.assert_close(state.global_kv, reference_state.global_kv, **tolerance)
        torch.testing.assert_close(state.indexer_k, reference_state.indexer_k, **tolerance)
        torch.testing.assert_close(state.global_indices, reference_state.global_indices[rows])
        probe = torch.randn_like(expected)
        pending.append((output, expected, probe, local_x, ref_x))
    assert gather_shapes == [
        torch.Size((cp_utils.get_cp_compressor_capacity(local_rows, ratio), 1, dim))
        for _ in range(2)
        for dim in (actual.config.dsa_indexer_head_dim, actual.config.v_head_dim)
    ]
    for tensor, before_rope in saved:
        torch.testing.assert_close(tensor, before_rope, atol=0, rtol=0)
    for output, expected, probe, local_x, ref_x in reversed(pending):
        output.backward(probe[rows])
        expected.backward(probe)
        torch.testing.assert_close(local_x.grad, ref_x.grad[rows], **tolerance)
        assert torch.isfinite(local_x.grad).all()
    _check_parameter_gradients(actual, reference, groups.cp, dtype)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("case", ["boundary", "ragged"])
def test_fused_full_cp_matches_native_reference(
    cp_groups, native_attention, monkeypatch, ratio, case
):
    _require_sparse_kernels()
    test_full_cp_matches_attention_and_gradients(
        cp_groups, native_attention, monkeypatch, ratio, torch.bfloat16, case, fused=True
    )


@pytest.mark.parametrize("ratio", [1, 2, (2, 1)], ids=["r1", "r2", "r2-to-r1"])
@pytest.mark.parametrize("mhc", [False, True])
def test_full_cp_hybrid_norm_and_mla_recompute(
    cp_groups, native_attention, cpu_checkpoint_rng, ratio, mhc
):
    test_cp_hybrid_stack_preserves_local_mhc_and_norm_recompute(
        cp_groups, native_attention, cpu_checkpoint_rng, mhc, True, compress_ratio=ratio
    )


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("field", ["global_kv", "indexer_k"])
def test_shared_cp_key_gradients_reduce_once_after_multiple_consumers(
    cp_groups, native_attention, monkeypatch, ratio, field
):
    groups, device = cp_groups
    reference, actual = _full_modules(groups, device, native_attention, ratio, torch.float32)
    params, _, valid = _compression_batch("boundary", device)
    params.cp_partition_mode = "contiguous"
    local_rows = valid.numel() // groups.cp.size()
    rows = slice(groups.cp.rank() * local_rows, (groups.cp.rank() + 1) * local_rows)
    ref_x = torch.randn(32, 1, actual.config.hidden_size, device=device, requires_grad=True)
    local_x = ref_x[rows].detach().clone().requires_grad_()
    qr = torch.randn(32, 1, actual.config.q_lora_rank, device=device)
    reference_state = CSA2State(thd_layout=build_csa2_thd_layout(params, 32))
    state = CSA2State(thd_layout=build_csa2_thd_layout(params, local_rows, cp_group=groups.cp))
    boundary = cp_utils.exchange_cp_boundary_hidden(
        local_x, ratio, actual.config.csa_window_size, groups.cp
    )
    reference.core_attention._shared_global_attention(
        ref_x, qr, reference_state, use_indexer_loss=True
    )
    actual.core_attention._shared_global_attention(
        local_x, qr[rows], state, use_indexer_loss=True, boundary_hidden=boundary
    )
    value, expected = getattr(state, field), getattr(reference_state, field)
    torch.testing.assert_close(value, expected, atol=4e-6, rtol=4e-5)
    probes = torch.randn(2, groups.cp.size(), *value.shape, device=device)
    # Normalize over shared rows/channels as for a training mean objective.
    # The reference still sums the distinct contributions from every CP rank.
    loss = (value * probes[0, groups.cp.rank()]).mean() + (
        value.square() * probes[1, groups.cp.rank()]
    ).mean()
    ref_loss = (expected * probes[0].sum(0)).mean() + (expected.square() * probes[1].sum(0)).mean()
    reductions = []
    original_rs = mappings.dist_reduce_scatter_func

    def record_rs(output, input_, **kwargs):
        reductions.append(tuple(output.shape))
        return original_rs(output, input_, **kwargs)

    monkeypatch.setattr(mappings, "dist_reduce_scatter_func", record_rs)
    loss.backward()
    ref_loss.backward()
    assert reductions == [
        (cp_utils.get_cp_compressor_capacity(local_rows, ratio), 1, value.shape[-1])
    ]
    if field == "indexer_k":
        assert ref_x.grad is None and local_x.grad is None
        assert all(
            parameter.grad is None for parameter in actual.core_attention.compressor.parameters()
        )
    else:
        torch.testing.assert_close(local_x.grad, ref_x.grad[rows], atol=5e-7, rtol=4e-5)
    _check_parameter_gradients(actual, reference, groups.cp, torch.float32, atol=5e-7)


# Reindex/Reuse, candidate selection and indexer supervision.


def _sharing_config(cp_size, dtype=torch.float32, **options):
    values = dict(
        num_layers=8,
        csa_compress_ratios=[2, 0, 2, 2, 1, 0, 1, 1],
        csa2_kv_source_layers=[0, 4],
        csa2_index_source_layers=[0, 2, 4, 6],
        csa2_candidate_source_layer=4,
        csa2_candidate_topk_blocks=2,
        csa2_candidate_block_size=2,
    )
    values.update(options)
    return _swa_config(cp_size, dtype, **values)


def _sharing_batch(case, device="cpu"):
    if case == "empty-ranks":
        # Only rank zero has real queries, including when CP=4.
        return _packed([3], [32], device=device)
    return _compression_batch(case, device)


def test_cp_sharing_configuration_preserves_remaining_guards():
    for options in ({}, {"calculate_per_token_loss": True}, {"csa_window_size": 1}):
        _sharing_config(2, dsa_indexer_loss_coeff=0.1, **options)
    _sharing_config(2, pipeline_model_parallel_size=2, pipeline_dtype=torch.float32)
    _cp_graph_config(2)
    with pytest.raises(ValueError, match="dynamic CP"):
        _sharing_config(
            2,
            dynamic_context_parallel=True,
            sequence_packing_scheduler="default_dynamic_cp",
            calculate_per_token_loss=True,
        )


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("case", ["boundary", "ragged", "empty-ranks", "no-sequences"])
def test_cp_candidate_score_chunks_keep_original_sequence_positions(cp_size, ratio, case):
    params, _, valid = _sharing_batch(case)
    params.cp_partition_mode = "contiguous"
    rows = valid.numel() // cp_size
    for rank in range(cp_size):
        layout = build_csa2_thd_layout(params, rows, cp_group=_LayoutGroup(cp_size, rank))
        compressed = layout.for_compression(ratio)
        inputs = prepare_csa2_indexer_inputs(
            torch.zeros(rows, 2, 8),
            torch.zeros(compressed.capacity, 8),
            torch.zeros(rows, 2),
            ratio,
            thd_layout=layout,
            compressed_layout=compressed,
        )
        cu_q, cu_k, _, _ = inputs.packed_metadata
        assert cu_q[0] == cu_k[0] == 0
        assert cu_q[-1] == rows and cu_k[-1] == compressed.capacity
        assert (cu_q.diff() >= 0).all() and (cu_k.diff() >= 0).all()
        assert cu_q.shape == cu_k.shape
        assert inputs.loss_divisor(False) == max(valid.sum(), 1)
        assert inputs.loss_divisor(True) == 1
        # A small tile cuts both sequences and CP shards. Reconstruct the scorer's
        # positions independently and compare with the original packed sequence.
        for start in range(0, rows, 3):
            end = min(rows, start + 3)
            prefix = cu_q.clamp(start, end) - start
            offsets = inputs.packed_causal_offsets + (start - cu_q[:-1]).clamp_min(0)
            for seq, (left, right) in enumerate(zip(prefix[:-1], prefix[1:])):
                for row in range(int(left), int(right)):
                    local_row = start + row
                    if layout.valid_tokens[local_row]:
                        assert seq == layout.sequence_ids[local_row]
                        position = offsets[seq] + row - left
                        assert position == layout.position_ids[local_row]
                        assert inputs.visible[local_row] == (position + 1) // ratio
                        assert inputs.starts[local_row] == cu_k[seq]


def _sharing_modules(
    groups, device, native_attention, dtype, mode, *, fused=False, explicit_group=False, **options
):
    if fused:
        options.update(
            dsa_kernel_backend="cudnn",
            apply_rope_fusion=True,
            num_attention_heads=64,
            v_head_dim=512,
            dsa_indexer_n_heads=32,
            dsa_indexer_head_dim=128,
        )
    config = _sharing_config(
        groups.cp.size(),
        dtype,
        dsa_indexer_loss_coeff=0 if mode == "off" else 0.3,
        dsa_indexer_use_sparse_loss=mode.startswith("sparse"),
        calculate_per_token_loss=mode.endswith("token"),
        **options,
    )
    reference_groups = copy.copy(groups)
    reference_groups.cp = groups.tp
    torch.manual_seed(891)
    reference = torch.nn.ModuleList(
        native_attention(
            replace(
                config, context_parallel_size=1, dsa_kernel_backend="none", apply_rope_fusion=False
            ),
            i + 1,
            reference_groups,
        )
        for i in range(config.num_layers)
    ).to(device)
    actual_groups = copy.copy(reference_groups if explicit_group else groups)
    actual = torch.nn.ModuleList(
        native_attention(config, i + 1, actual_groups) for i in range(config.num_layers)
    ).to(device)
    actual.load_state_dict(reference.state_dict())
    if config.apply_rope_fusion:
        for module in (*reference, *actual):
            module.rotary_pos_emb = _rotary_module(module.config, True, module.pg_collection.cp)
            module.core_attention.rotary_pos_emb = module.rotary_pos_emb
    return reference, actual


@pytest.fixture
def loss_tracker(monkeypatch):
    # Use the real tracker on CPU too; production normally initializes it on CUDA.
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", {"values": torch.zeros(12)})
    monkeypatch.setattr(DSAIndexerLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    records = []
    original = DSAIndexerLossLoggingHelper.save_loss_to_tracker

    def save(**kwargs):
        if DSAIndexerLossLoggingHelper.tracker["values"].device != kwargs["loss"].device:
            DSAIndexerLossLoggingHelper.tracker["values"] = kwargs["loss"].new_zeros(12)
        original(**kwargs)
        records.append(kwargs)

    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "save_loss_to_tracker", save)
    return records


def _run_layers(layers, x, params, *, defer=False):
    state = CSA2State(defer_indexer_loss=defer)
    outputs, snapshots, losses = [], [], []
    for i, layer in enumerate(layers):
        state.indexer_loss = None
        attention = layer(x, None, packed_seq_params=params, csa2_state=state)[0]
        x = x.masked_fill(~state.thd_layout.valid_tokens[:, None, None], 0) + 0.25 * attention
        outputs.append(x)
        if state.indexer_loss is not None:
            losses.append((i, state.indexer_loss))
        snapshots.append(
            (
                state.global_kv,
                state.indexer_k,
                state.global_indices,
                state.candidates,
                state.fused_indices,
                state.fused_topk_length,
            )
        )
    return outputs, snapshots, losses


@pytest.mark.parametrize("mode", ["off", "dense", "sparse", "dense-token", "sparse-token"])
@pytest.mark.parametrize(
    "case", ["boundary", "ragged", "empty-ranks", "short", "all-padding", "no-sequences"]
)
def test_cp_sharing_matches_outputs_losses_and_gradients(
    cp_groups,
    native_attention,
    loss_tracker,
    monkeypatch,
    mode,
    case,
    dtype=torch.float32,
    fused=False,
    **options,
):
    groups, device = cp_groups
    reference, actual = _sharing_modules(
        groups, device, native_attention, dtype, mode, fused=fused, **options
    )
    config = actual[0].config
    params, _, valid = _sharing_batch(case, device)
    params.cp_partition_mode = "contiguous"
    local_params = replace(params, cp_group=groups.cp) if options.get("explicit_group") else params
    local_rows = valid.numel() // groups.cp.size()
    rows = slice(groups.cp.rank() * local_rows, (groups.cp.rank() + 1) * local_rows)
    tolerance = dict(atol=5e-6, rtol=5e-5) if dtype == torch.float32 else dict(atol=6e-3, rtol=3e-2)
    gather_shapes, reductions = [], []
    original_gather = csa2_runtime.async_gather_from_sequence_parallel_region
    original_reduce = mappings.dist_reduce_scatter_func

    def gather(tensor, **kwargs):
        gather_shapes.append(tensor.shape)
        return original_gather(tensor, **kwargs)

    def reduce(output, tensor, **kwargs):
        reductions.append(tensor.shape)
        return original_reduce(output, tensor, **kwargs)

    monkeypatch.setattr(csa2_runtime, "async_gather_from_sequence_parallel_region", gather)
    monkeypatch.setattr(mappings, "dist_reduce_scatter_func", reduce)
    pending = []
    for _ in range(2):
        ref_x = torch.randn(valid.numel(), 1, config.hidden_size, device=device, dtype=dtype)
        ref_x[~valid] = float("nan")
        ref_x.requires_grad_()
        x = ref_x[rows].detach().clone().requires_grad_()
        first_log = len(loss_tracker)
        expected, ref_states, _ = _run_layers(
            reference, ref_x.masked_fill(~valid[:, None, None], 0), params
        )
        ref_logs = loss_tracker[first_log:]
        first_log = len(loss_tracker)
        outputs, states, _ = _run_layers(actual, x, local_params)
        logs = loss_tracker[first_log:]
        for output, target, state, ref_state in zip(outputs, expected, states, ref_states):
            torch.testing.assert_close(output, target[rows], **tolerance)
            for value, ref_value in zip(state[:2], ref_state[:2]):
                if value is not None:
                    torch.testing.assert_close(value, ref_value, **tolerance)
            if state[2] is not None:
                torch.testing.assert_close(state[2], ref_state[2][rows], atol=0, rtol=0)
            if state[3] is not None:
                torch.testing.assert_close(
                    state[3].indices, ref_state[3].indices[rows], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    state[3].lengths, ref_state[3].lengths[rows], atol=0, rtol=0
                )
        for reuse, source in ((3, 2), (7, 6)):
            assert states[reuse][2] is states[source][2]
            if fused:
                assert states[reuse][4] is states[source][4]
                assert states[reuse][5] is states[source][5]
        if fused:
            assert states[5][4] is states[4][4], "SWA must preserve the compressed index cache"
        for i in (2, 3, 6, 7):
            owner = 0 if i < 4 else 4
            assert states[i][0] is states[owner][0]
            assert states[i][1] is states[owner][1]
        assert [entry["layer_number"] for entry in logs] == ([] if mode == "off" else [1, 3, 5, 7])
        for log, ref_log in zip(logs, ref_logs):
            assert log["reduce_group"] is groups.cp and ref_log["reduce_group"] is None
            total = log["loss"].detach().clone()
            dist.all_reduce(total, group=groups.cp)
            torch.testing.assert_close(total, ref_log["loss"], **tolerance)
        probe = torch.randn_like(expected[-1]) / expected[-1].numel()
        pending.append((outputs[-1], expected[-1], probe, x, ref_x))
    assert len(gather_shapes) == 8, "Only two K gathers per Full owner and microbatch"
    for output, target, probe, x, ref_x in reversed(pending):
        reductions.clear()
        output.backward(probe[rows])
        target.backward(probe)
        assert len(reductions) == (2 if mode == "off" else 4), "One RS per shared main/indexer K"
        torch.testing.assert_close(x.grad, ref_x.grad[rows], **tolerance)
        assert torch.isfinite(x.grad).all()
    _check_parameter_gradients(actual, reference, groups.cp, dtype, atol=5e-6)


@pytest.mark.parametrize("mode", ["off", "dense", "sparse-token"])
def test_native_bf16_cp_sharing(cp_groups, native_attention, loss_tracker, monkeypatch, mode):
    test_cp_sharing_matches_outputs_losses_and_gradients(
        cp_groups,
        native_attention,
        loss_tracker,
        monkeypatch,
        mode,
        "boundary",
        dtype=torch.bfloat16,
    )


def test_cp_sharing_uses_explicit_packed_group(
    cp_groups, native_attention, loss_tracker, monkeypatch
):
    test_cp_sharing_matches_outputs_losses_and_gradients(
        cp_groups,
        native_attention,
        loss_tracker,
        monkeypatch,
        "dense",
        "ragged",
        explicit_group=True,
    )


@pytest.mark.parametrize("mode", ["dense", "sparse"])
@pytest.mark.parametrize("window", [1, 4])
def test_fused_cp_sharing(cp_groups, native_attention, loss_tracker, monkeypatch, mode, window):
    _require_sparse_kernels()
    test_cp_sharing_matches_outputs_losses_and_gradients(
        cp_groups,
        native_attention,
        loss_tracker,
        monkeypatch,
        mode,
        "boundary",
        dtype=torch.bfloat16,
        fused=True,
        csa_window_size=window,
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_cp_reindex_only_loss_trains_owner_keys_without_compressor_gradients(
    cp_groups, native_attention, loss_tracker, monkeypatch, sparse
):
    groups, device = cp_groups
    reference, actual = _sharing_modules(
        groups, device, native_attention, torch.float32, "sparse" if sparse else "dense"
    )
    params, _, valid = _sharing_batch("boundary", device)
    params.cp_partition_mode = "contiguous"
    local_rows = valid.numel() // groups.cp.size()
    rows = slice(groups.cp.rank() * local_rows, (groups.cp.rank() + 1) * local_rows)
    ref_x = torch.randn(32, 1, actual[0].config.hidden_size, device=device, requires_grad=True)
    x = ref_x[rows].detach().clone().requires_grad_()
    _, _, ref_losses = _run_layers(reference, ref_x, params, defer=True)
    _, _, losses = _run_layers(actual, x, params, defer=True)
    reductions = []
    original_reduce = mappings.dist_reduce_scatter_func

    def reduce(output, tensor, **kwargs):
        reductions.append(tensor.shape)
        return original_reduce(output, tensor, **kwargs)

    monkeypatch.setattr(mappings, "dist_reduce_scatter_func", reduce)
    sum(loss for i, loss in losses if i in (2, 6)).backward()
    sum(loss for i, loss in ref_losses if i in (2, 6)).backward()
    assert len(reductions) == 2
    assert x.grad is ref_x.grad is None
    for name, parameter in actual.named_parameters():
        if ".indexer." not in name:
            assert parameter.grad is None, name
    for owner in (0, 4):
        indexer = actual[owner].core_attention.indexer
        assert indexer.linear_wk.weight.grad is not None
        assert indexer.k_norm.weight.grad is not None
        assert indexer.linear_wq_b.weight.grad is None
    _check_parameter_gradients(actual, reference, groups.cp, torch.float32)


@pytest.mark.parametrize("training", [False, True])
def test_cp_eval_or_no_grad_skips_indexer_supervision(
    cp_groups, native_attention, loss_tracker, training
):
    groups, device = cp_groups
    reference, actual = _sharing_modules(groups, device, native_attention, torch.float32, "dense")
    reference.train(training)
    actual.train(training)
    params, _, valid = _sharing_batch("ragged", device)
    params.cp_partition_mode = "contiguous"
    count = valid.numel() // groups.cp.size()
    rows = slice(groups.cp.rank() * count, (groups.cp.rank() + 1) * count)
    x = torch.randn(32, 1, actual[0].config.hidden_size, device=device)
    # Check eval with autograd enabled, and train under no_grad (checkpoint first pass).
    with torch.set_grad_enabled(not training):
        expected, _, _ = _run_layers(reference, x, params)
        output, _, _ = _run_layers(actual, x[rows].clone(), params)
    assert not loss_tracker
    torch.testing.assert_close(output[-1], expected[-1][rows], atol=5e-6, rtol=5e-5)


def test_cp_sharing_recomputes_local_projections(
    cp_groups, native_attention, loss_tracker, monkeypatch, cpu_checkpoint_rng
):
    test_cp_sharing_matches_outputs_losses_and_gradients(
        cp_groups,
        native_attention,
        loss_tracker,
        monkeypatch,
        "dense",
        "ragged",
        recompute_granularity="selective",
        recompute_modules=["layernorm", "mla_up_proj"],
    )


# CP coordinates and shared tensors across physical/virtual pipeline stages.


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("cp_rank", [0, 1])
def test_cp_pipeline_roundtrip_keeps_local_queries_and_global_keys(cp_size, cp_rank):
    config = replace(
        _pipeline_config(),
        context_parallel_size=cp_size,
        cp_partition_mode="contiguous",
        sequence_packing_scheduler="dp_balanced",
        moe_token_dispatcher_type="alltoall",
    )
    params, _, valid = _sharing_batch("boundary")
    params.cp_partition_mode = "contiguous"
    group = _LayoutGroup(cp_size, cp_rank)
    rows = valid.numel() // cp_size
    layout = build_csa2_thd_layout(params, rows, cp_group=group)
    global_layout = build_csa2_thd_layout(params, valid.numel())
    plan = build_csa2_pipeline_plan(
        config, _pipeline_pattern((1, 2, 4, 6, 7, 8, 10)), qkv_format="thd"
    )
    for chunk in plan[:-1]:
        boundary = chunk.outgoing
        descriptor = boundary.payload_spec(config, rows, 1, params, cp_group=group)
        assert descriptor.metadata == (boundary.layer_offset, params.max_seqlen_q, cp_size, cp_rank)
        values = {
            spec.name: torch.zeros(spec.shape, dtype=spec.dtype, requires_grad=spec.requires_grad)
            for spec in descriptor.tensor_specs
        }
        state = CSA2State(
            last_layer=boundary.last_attention_layer,
            kv_source_layer=boundary.kv_source_layer,
            index_source_layer=boundary.index_source_layer,
            candidate_source_layer=boundary.candidate_source_layer,
            global_kv=values.get("global_kv"),
            indexer_k=values.get("indexer_k"),
            global_indices=values.get("global_indices"),
            thd_layout=layout,
        )
        if boundary.candidate_source_layer is not None:
            state.candidates = CSA2CandidateBlocks(
                values["candidate_indices"],
                values["candidate_lengths"],
                config.csa2_candidate_block_size,
            )
        payload = boundary.export_payload(
            values["hidden_states"],
            state,
            SinglePassMHCState(values["pre_mix"]),
            packed_seq_params=params,
            cp_group=group,
        )
        assert isinstance(payload, CSA2PipelinePayload)
        assert payload.tensor_specs == descriptor.tensor_specs
        if state.global_kv is not None:
            assert (
                state.global_kv.shape[0]
                == global_layout.for_compression(boundary.compress_ratio).capacity
            )
        specs, metadata = _unpack_header(
            _pack_header(payload.tensor_specs, payload.metadata, 3, 1), 3, 1
        )
        assert metadata == payload.metadata
        assert [spec.shape for spec in specs] == [spec.shape for spec in payload.tensor_specs]
        hidden, restored, mhc, received_params = payload.restore(cp_group=group)
        assert hidden is values["hidden_states"] and mhc.pre_mix is values["pre_mix"]
        assert (
            received_params.cp_group is group and received_params.cp_partition_mode == "contiguous"
        )
        assert restored.thd_layout.global_start == cp_rank * rows
        layout.validate_layout(restored.thd_layout)
        if state.global_kv is not None:
            _assert_layout(
                restored.compressed_layout, global_layout.for_compression(boundary.compress_ratio)
            )
            assert restored.global_kv is state.global_kv
        with pytest.raises(ValueError, match="CP shard"):
            payload.restore(cp_group=_LayoutGroup(cp_size, 1 - cp_rank))
        with pytest.raises(ValueError, match="CP shard"):
            payload.restore()
        with pytest.raises(ValueError, match="invalid CP metadata"):
            replace(payload, cp_rank=cp_size).validate()
        # Planning works with content-free prefixes and needs no device scalar reads.
        meta_params = replace(
            params,
            cu_seqlens_q=torch.empty_like(params.cu_seqlens_q, device="meta"),
            cu_seqlens_q_padded=torch.empty_like(params.cu_seqlens_q_padded, device="meta"),
            total_tokens=None,
            seq_idx=None,
        )
        assert boundary.payload_spec(config, rows, 1, meta_params, cp_group=group) == descriptor
        with pytest.raises(ValueError, match="explicit CP group"):
            boundary.payload_spec(config, rows, 1, params)
        explicit_params = replace(params, cp_group=group)
        assert boundary.payload_spec(config, rows, 1, explicit_params) == descriptor


@contextmanager
def _cp_pipeline_groups(cp_size=2, pp_size=2):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world % (cp_size * pp_size):
        pytest.skip(f"Requires torchrun with world size divisible by CP={cp_size} * PP={pp_size}")
    if torch.cuda.is_available():
        Utils.initialize_model_parallel(
            context_parallel_size=cp_size, pipeline_model_parallel_size=pp_size
        )
        model_parallel_cuda_manual_seed(719)
        try:
            yield ProcessGroupCollection.use_mpu_process_groups(), torch.device(
                "cuda", torch.cuda.current_device()
            )
        finally:
            Utils.destroy_model_parallel()
        return
    if not dist.is_initialized():
        dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    rank = dist.get_rank()
    singletons = [dist.new_group([i], timeout=timedelta(seconds=90)) for i in range(world)]
    contexts = [
        dist.new_group(list(range(i, i + cp_size)), timeout=timedelta(seconds=90))
        for i in range(0, world, cp_size)
    ]
    pipelines = [
        dist.new_group(
            [base + cp + pp * cp_size for pp in range(pp_size)], timeout=timedelta(seconds=90)
        )
        for base in range(0, world, pp_size * cp_size)
        for cp in range(cp_size)
    ]
    groups = ProcessGroupCollection()
    groups.tp = singletons[rank]
    groups.cp = groups.dp_cp = contexts[rank // cp_size]
    groups.pp = pipelines[rank // (pp_size * cp_size) * cp_size + rank % cp_size]
    groups.embd = groups.pos_embd = None
    try:
        yield groups, torch.device("cpu")
        dist.barrier()
    finally:
        for group in (groups.pp, groups.cp, groups.tp):
            dist.destroy_process_group(group)


@pytest.mark.parametrize(
    "cuts,overlap,warmup,planned",
    [
        ((1,), False, False, True),
        ((4,), False, False, True),
        ((6,), False, False, True),
        ((8,), False, False, True),
        ((7, 8, 10), False, False, True),
        ((7, 8, 10), True, False, True),
        ((7, 8, 10), True, True, True),
        ((7, 8, 10), False, False, False),
    ],
)
def test_cp_pp_vpp_matches_single_stage(
    native_attention,
    loss_tracker,
    monkeypatch,
    cuts,
    overlap,
    warmup,
    planned,
    recompute=None,
    dtype=torch.float32,
    coefficient=0.3,
    mhc=True,
    forward_only=False,
    per_token=False,
    sparse_loss=False,
    batch_p2p=False,
    fused=False,
    graph_mode=None,
):
    with _cp_pipeline_groups() as (groups, device):
        monkeypatch.setattr(
            DSAIndexerLossAutoScaler, "main_loss_backward_scale", torch.ones(1, device=device)
        )
        cp_size, pp_size = groups.cp.size(), groups.pp.size()
        vp_size = (len(cuts) + 1) // pp_size
        kernel_options = (
            dict(
                num_attention_heads=64,
                v_head_dim=512,
                dsa_indexer_n_heads=32,
                dsa_indexer_head_dim=128,
            )
            if fused
            else {}
        )
        config = replace(
            _pipeline_config(dtype, coefficient, mhc),
            context_parallel_size=cp_size,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vp_size if vp_size > 1 else None,
            microbatch_group_size_per_vp_stage=pp_size,
            pipeline_dtype=dtype,
            batch_p2p_comm=batch_p2p,
            overlap_p2p_comm=overlap,
            overlap_p2p_comm_warmup_flush=warmup,
            deallocate_pipeline_outputs=True,
            cp_partition_mode="contiguous",
            sequence_packing_scheduler="dp_balanced",
            moe_token_dispatcher_type="alltoall",
            cross_entropy_loss_fusion=False,
            multi_latent_attention=True,
            calculate_per_token_loss=per_token,
            dsa_indexer_use_sparse_loss=sparse_loss,
            dsa_kernel_backend="cudnn" if fused else "none",
            **kernel_options,
        )
        if isinstance(recompute, str):
            config = replace(
                config,
                recompute_granularity="full",
                recompute_method=recompute,
                recompute_num_layers=2,
            )
        elif recompute:
            config = replace(config, recompute_granularity="selective", recompute_modules=recompute)
        if graph_mode:
            config = replace(
                config,
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=["attn"],
                max_seqlen_per_dp_cp_rank=16,
                thd_max_packed_sequences=6,
                pad_packed_seq_alignment="max",
                is_hybrid_model=True,
            )
        reference_groups = copy.copy(groups)
        # FP32 compares the full CP=1/PP=1 model. In BF16, use the same CP
        # partition to isolate PP from rounding/cancellation in CP-local GEMMs
        # and mHC reductions. BF16 CP=1 parity is covered by the attention tests.
        reference_cp = cp_size if dtype == torch.bfloat16 else 1
        reference_groups.cp = groups.cp if reference_cp > 1 else groups.tp
        reference_groups.pp = groups.tp
        torch.manual_seed(719)
        reference = _pipeline_model(
            replace(
                config,
                context_parallel_size=reference_cp,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                overlap_p2p_comm=False,
                overlap_p2p_comm_warmup_flush=False,
                recompute_granularity=None,
                recompute_method=None,
                recompute_num_layers=None,
                dsa_kernel_backend="none",
                cuda_graph_impl="none",
            ),
            reference_groups,
            "DE" * 6,
            True,
            True,
            device,
            attention=native_attention,
        )
        pattern = _pipeline_pattern(cuts)
        rank = groups.pp.rank()
        models = [
            _pipeline_model(
                config,
                groups,
                pattern,
                rank == 0 and vp == 0,
                rank == pp_size - 1 and vp == vp_size - 1,
                device,
                vp_stage=vp if vp_size > 1 else None,
                attention=native_attention,
            )
            for vp in range(vp_size)
        ]
        plan = build_csa2_pipeline_plan(config, pattern, pp_size=pp_size, qkv_format="thd")
        chunks = [plan[vp * pp_size + rank] for vp in range(vp_size)]
        ref_parameters = dict(reference.named_parameters())
        with torch.no_grad():
            for stage, chunk in zip(models, chunks):
                for name, parameter in stage.named_parameters():
                    parameter.copy_(
                        ref_parameters[_reference_parameter_name(name, chunk.layer_offset)]
                    )
        batches = []
        generator = torch.Generator().manual_seed(718)
        for case in ("boundary", "ragged", "empty-ranks", "no-sequences"):
            if graph_mode:
                params, valid = _cp_graph_batch(case, device)
            else:
                params, _, valid = _sharing_batch(case, device)
            params.cp_partition_mode = "contiguous"
            tokens = torch.randint(0, 32, (1, valid.numel()), generator=generator).to(device)
            labels = torch.randint(0, 32, tokens.shape, generator=generator).to(device)
            batches.append((tokens, labels, valid.unsqueeze(0), params))
        count = len(batches)
        if graph_mode:
            _install_cp_graph_callables(
                [model.decoder for model in models],
                monkeypatch,
                joint_backward=graph_mode == "joint",
                num_microbatches=count,
            )
        total_tokens = sum(valid.sum() for _, _, valid, _ in batches).clamp_min(1)
        local_rows = batches[0][0].shape[1] // cp_size
        rows = slice(groups.cp.rank() * local_rows, (groups.cp.rank() + 1) * local_rows)
        tolerance = (
            dict(atol=8e-6, rtol=8e-5) if dtype == torch.float32 else dict(atol=3e-3, rtol=5e-2)
        )
        if planned:

            def no_header(*args, **kwargs):
                pytest.fail("Prepared CP+PP must not parse or send device headers")

            monkeypatch.setattr(
                "megatron.core.pipeline_parallel.typed_p2p_communication._pack_header", no_header
            )
            monkeypatch.setattr(
                "megatron.core.pipeline_parallel.typed_p2p_communication._unpack_header", no_header
            )
        for _ in range(2):
            expected, expected_outputs = [], []
            reference.zero_grad(set_to_none=True)
            for stage in models:
                stage.zero_grad(set_to_none=True)
            loss_tracker.clear()
            DSAIndexerLossAutoScaler.set_loss_scale(
                torch.tensor([1.0 if per_token else 1.0 / count], device=device)
            )
            with torch.set_grad_enabled(not forward_only):
                for tokens, labels, valid, params in batches:
                    ref_rows = rows if reference_cp > 1 else slice(None)
                    losses = reference(
                        tokens[:, ref_rows],
                        None,
                        None,
                        labels=labels[:, ref_rows],
                        packed_seq_params=params,
                    )
                    expected_outputs.append(losses.detach())
                    loss = (losses.float() * valid[:, ref_rows]).sum()
                    logged = (loss / valid.sum().clamp_min(1)).detach()
                    if reference_cp > 1:
                        dist.all_reduce(logged, group=groups.cp)
                    expected.append(logged)
                    if not forward_only:
                        (loss if per_token else loss / valid.sum().clamp_min(1) / count).backward()
            if not forward_only:
                for parameter in reference.parameters():
                    if parameter.grad is not None:
                        if reference_cp > 1:
                            dist.all_reduce(parameter.grad, group=groups.cp)
                        if per_token:
                            parameter.grad.div_(total_tokens)
            reference_aux = list(loss_tracker)
            loss_tracker.clear()
            actual, payload_refs, finalized_counts = [], [], []

            def finalize(models, num_tokens, **kwargs):
                assert per_token and kwargs["pg_collection"] is groups
                if models[-1].post_process:
                    local_tokens = sum(valid[:, rows].sum() for _, _, valid, _ in batches)
                    torch.testing.assert_close(num_tokens, local_tokens.to(num_tokens))
                # Match the normal finalizer's PP broadcast and DP-with-CP token sum.
                dist.broadcast(
                    num_tokens, src=dist.get_global_rank(groups.pp, pp_size - 1), group=groups.pp
                )
                dist.all_reduce(num_tokens, group=groups.cp)
                torch.testing.assert_close(num_tokens, total_tokens.to(num_tokens))
                finalized_counts.append(num_tokens.clone())

            config.finalize_model_grads_func = finalize if per_token else None

            def forward_step(iterator, stage):
                tokens, labels, valid, params = next(iterator)
                incoming = stage.decoder.input_tensor
                if isinstance(incoming, PipelinePayload):
                    payload_refs.append(weakref.ref(incoming))
                    assert all(tensor.is_leaf for tensor in incoming.tensors)
                output = stage(
                    tokens[:, rows] if stage.pre_process else None,
                    None,
                    None,
                    labels=labels[:, rows] if stage.post_process else None,
                    packed_seq_params=params if stage.pre_process else None,
                )
                assert stage.decoder.input_tensor is None
                if isinstance(output, PipelinePayload):
                    payload_refs.append(weakref.ref(output))

                def loss_func(losses):
                    expected_rows = slice(None) if reference_cp > 1 else rows
                    torch.testing.assert_close(
                        losses, expected_outputs[len(actual)][:, expected_rows], **tolerance
                    )
                    loss = (losses.float() * valid[:, rows]).sum()
                    logged = (loss / valid.sum().clamp_min(1)).detach().clone()
                    dist.all_reduce(logged, group=groups.cp)
                    actual.append(logged)
                    if per_token:
                        return loss, valid[:, rows].sum().to(torch.int), {"loss": logged}
                    return loss / valid.sum().clamp_min(1), {"loss": logged}

                return output, loss_func

            if planned:

                def prepare(iterator, stage, num_microbatches, *, forward_only):
                    inputs, incoming, outgoing = [], [], []
                    for _ in range(num_microbatches):
                        batch = next(iterator)
                        recv, send = stage.pipeline_payload_spec(
                            local_rows, 1, batch[-1], requires_grad=not forward_only
                        )
                        inputs.append(batch)
                        incoming.append(recv)
                        outgoing.append(send)
                    return PipelineDataIterator(
                        inputs, PipelinePayloadPlan(tuple(incoming), tuple(outgoing))
                    )

                forward_step.prepare_pipeline_inputs = prepare
            schedule = (
                forward_backward_pipelining_with_interleaving
                if vp_size > 1
                else forward_backward_pipelining_without_interleaving
            )
            with torch.set_grad_enabled(not forward_only):
                schedule(
                    forward_step_func=forward_step,
                    data_iterator=[iter(batches) for _ in models] if vp_size > 1 else iter(batches),
                    model=models if vp_size > 1 else models[0],
                    num_microbatches=count,
                    seq_length=32,
                    micro_batch_size=1,
                    forward_only=forward_only,
                    p2p_communicator=P2PCommunicator(groups.pp, config),
                    pg_collection=groups,
                )
            gc.collect()
            assert all(ref() is None for ref in payload_refs)
            assert len(finalized_counts) == int(per_token and not forward_only)
            if models[-1].post_process:
                torch.testing.assert_close(torch.stack(actual), torch.stack(expected), **tolerance)
            local_layers = {
                chunk.layer_offset + i + 1
                for chunk in chunks
                for i in range(len(chunk.layer_pattern))
            }
            for layer in sorted(local_layers):
                records = [
                    entry["loss"] for entry in loss_tracker if entry["layer_number"] == layer
                ]
                target = [
                    entry["loss"] for entry in reference_aux if entry["layer_number"] == layer
                ]
                assert len(records) == len(target)
                if records:
                    value = torch.stack(records)
                    dist.all_reduce(value, group=groups.cp)
                    target = torch.stack(target)
                    if reference_cp > 1:
                        dist.all_reduce(target, group=groups.cp)
                    torch.testing.assert_close(
                        value.sort().values, target.sort().values, **tolerance
                    )
            if not forward_only:
                for stage, chunk in zip(models, chunks):
                    for name, parameter in stage.named_parameters():
                        target = ref_parameters[_reference_parameter_name(name, chunk.layer_offset)]
                        assert (parameter.grad is None) == (target.grad is None), name
                        if target.grad is None:
                            continue
                        dist.all_reduce(parameter.grad, group=groups.cp)
                        # Mean mode averages replicated gradients; per-token mode sums
                        # them and normalizes once by the global batch token count.
                        parameter.grad.div_(total_tokens if per_token else cp_size)
                        if dtype == torch.float32:
                            torch.testing.assert_close(
                                parameter.grad, target.grad, **tolerance, msg=name
                            )
                        else:
                            error = (parameter.grad.float() - target.grad.float()).norm()
                            bound = 3 * torch.finfo(dtype).eps * target.grad.float().norm() + 1e-6
                            assert error <= bound, f"{name}: {error} > {bound}"
                with torch.no_grad():
                    for stage in models:
                        for parameter in stage.parameters():
                            if parameter.grad is not None:
                                parameter.add_(parameter.grad, alpha=-0.01)
                    for parameter in reference.parameters():
                        if parameter.grad is not None:
                            parameter.add_(parameter.grad, alpha=-0.01)
                    for stage, chunk in zip(models, chunks):
                        for name, parameter in stage.named_parameters():
                            target = ref_parameters[
                                _reference_parameter_name(name, chunk.layer_offset)
                            ]
                            torch.testing.assert_close(parameter, target, **tolerance, msg=name)


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(dict(dtype=torch.bfloat16, coefficient=0.0), id="bf16-aux-off"),
        pytest.param(dict(dtype=torch.bfloat16), id="bf16-aux-on"),
        pytest.param(dict(per_token=True), id="per-token-dense"),
        pytest.param(dict(per_token=True, sparse_loss=True), id="per-token-sparse"),
        pytest.param(dict(mhc=False), id="ordinary-residual"),
        pytest.param(dict(forward_only=True), id="forward-only"),
        pytest.param(dict(batch_p2p=True), id="batched-p2p"),
    ],
)
def test_cp_pp_vpp_training_modes(native_attention, loss_tracker, monkeypatch, options):
    test_cp_pp_vpp_matches_single_stage(
        native_attention, loss_tracker, monkeypatch, (7, 8, 10), False, False, True, **options
    )


@pytest.mark.parametrize(
    "recompute",
    [
        "uniform",
        "block",
        ["mhc"],
        ["layernorm", "mla_up_proj"],
        ["mhc", "layernorm", "mla_up_proj"],
    ],
)
def test_cp_pp_vpp_recompute(
    native_attention, loss_tracker, monkeypatch, cpu_checkpoint_rng, recompute
):
    test_cp_pp_vpp_matches_single_stage(
        native_attention,
        loss_tracker,
        monkeypatch,
        (7, 8, 10),
        True,
        True,
        True,
        recompute=recompute,
    )


@pytest.mark.parametrize("per_token", [False, True])
def test_fused_cp_pp_vpp(native_attention, loss_tracker, monkeypatch, per_token):
    _require_sparse_kernels()
    test_cp_pp_vpp_matches_single_stage(
        native_attention,
        loss_tracker,
        monkeypatch,
        (7, 8, 10),
        True,
        True,
        True,
        dtype=torch.bfloat16,
        fused=True,
        per_token=per_token,
    )


@pytest.mark.parametrize("stage", range(4))
@pytest.mark.parametrize("cp_size,cp_rank", [(2, 0), (2, 1), (4, 0), (4, 3)])
def test_cp_pipeline_entry_plans_local_capacity_without_device_reads(
    monkeypatch, stage, cp_size, cp_rank
):
    args = SimpleNamespace(
        tensor_model_parallel_size=1,
        context_parallel_size=cp_size,
        seq_length=32,
        micro_batch_size=1,
        sft=False,
        dataloader_inter_document_masking=False,
        sequence_packing_scheduler="dp_balanced",
    )
    timers = _Timers()
    monkeypatch.setattr(hybrid_entry, "get_args", lambda: args)
    monkeypatch.setattr(hybrid_entry, "get_timers", lambda: timers)
    monkeypatch.setattr(hybrid_entry, "stimer", lambda **kwargs: nullcontext())
    # Dataset routing is upstream; exercise the prepared entry with its already
    # partitioned batch and global prefixes, as returned by the packing scheduler.
    group = _LayoutGroup(cp_size, cp_rank)
    config = replace(
        _pipeline_config(),
        context_parallel_size=cp_size,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        cp_partition_mode="contiguous",
        sequence_packing_scheduler="dp_balanced",
        moe_token_dispatcher_type="alltoall",
        pipeline_dtype=torch.float32,
    )
    pattern = _pipeline_pattern((7, 8, 10))
    adapter = CSA2HybridAdapter(
        config,
        layer_type_list=list(pattern.split("|")[stage]),
        pp_layer_offset=(0, 7, 8, 10)[stage],
        pre_process=stage == 0,
        post_process=stage == 3,
        is_mtp_layer=False,
        pg_collection=SimpleNamespace(cp=group),
    )
    adapter.configure_distributed_pipeline(pattern, _LayoutGroup(2, stage % 2), vp_stage=stage // 2)
    model = SimpleNamespace(
        pipeline_payload_spec=adapter.pipeline_payload_spec, vp_stage=stage // 2
    )
    batches, descriptors = [], []
    for case in ("boundary", "ragged", "empty-ranks", "no-sequences", "graph-prefixes"):
        if case == "graph-prefixes":
            params, valid = _cp_graph_batch("ragged", "cpu")
        else:
            params, _, valid = _sharing_batch(case)
        local_rows = valid.numel() // cp_size
        params.cp_partition_mode, params.local_cp_size, params.total_tokens = (
            "contiguous",
            cp_size,
            None if case == "graph-prefixes" else local_rows,
        )
        params.cp_group = group
        rows = slice(cp_rank * local_rows, (cp_rank + 1) * local_rows)
        batch = [None] * 12
        batch[-1] = params
        if stage in (0, 3):
            batch[9 if stage == 0 else 4] = torch.zeros(1, local_rows, dtype=torch.int64)
        if case != "no-sequences":
            batch[10] = (~valid[rows]).unsqueeze(0)
        batches.append(tuple(batch))
        descriptors.append(adapter.pipeline_payload_spec(local_rows, 1, params))
    original_get_batch = hybrid_entry.get_batch
    fetched = []

    def get_batch(iterator, vp_stage=None):
        if isinstance(iterator, PipelineDataIterator):
            return original_get_batch(iterator, vp_stage)
        assert vp_stage == stage // 2
        fetched.append(vp_stage)
        return next(iterator)

    monkeypatch.setattr(hybrid_entry, "get_batch", get_batch)

    def no_item(*args, **kwargs):
        pytest.fail("Prepared CP descriptors and consumption must not read device scalars")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", no_item)
        prepared = hybrid_entry.prepare_pipeline_inputs(iter(batches), model, len(batches))
        assert len(fetched) == len(batches)
        for index, expected in enumerate(descriptors):
            plan = prepared.pipeline_payload_plan
            assert (plan.incoming[index], plan.outgoing[index]) == expected
            batch = hybrid_entry.get_batch(prepared, model.vp_stage)
            assert batch[-1] is batches[index][-1]
            assert batch[-1].cp_group is group
            for spec in expected:
                if spec is not None:
                    assert spec.tensor_specs[0].shape[:2] == (32 // cp_size, 1)
                    assert spec.metadata[2:] == (cp_size, cp_rank)
                    if index == len(batches) - 1:
                        # Static graph prefixes reserve the global r1 capacity,
                        # although each rank holds only 32 // cp_size hidden rows.
                        for tensor_spec in spec.tensor_specs:
                            if tensor_spec.name in ("global_kv", "indexer_k"):
                                assert tensor_spec.shape[0] == 32
        with pytest.raises(StopIteration):
            hybrid_entry.get_batch(prepared, model.vp_stage)
    assert len(fetched) == len(batches)
    assert all(timer.started == timer.stopped for timer in timers.values())


# Static CP layout and shared state through TE graph tensor boundaries.


def _cp_graph_config(cp_size, dtype=torch.float32, coefficient=0.3, mhc=True, **options):
    values = dict(
        context_parallel_size=cp_size,
        cp_partition_mode="contiguous",
        sequence_packing_scheduler="dp_balanced",
        moe_token_dispatcher_type="alltoall",
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=["attn"],
        max_seqlen_per_dp_cp_rank=32 // cp_size,
        thd_max_packed_sequences=6,
        pad_packed_seq_alignment="max",
        multi_latent_attention=True,
        is_hybrid_model=True,
    )
    values.update(options)
    return replace(_pipeline_config(dtype, coefficient, mhc), **values)


def _cp_graph_batch(case, device):
    params, _, valid = _sharing_batch(case, device)
    prefixes = {}
    for suffix in ("q", "kv", "q_padded", "kv_padded"):
        name = "cu_seqlens_" + suffix
        prefix = getattr(params, name)
        prefixes[name] = torch.cat((prefix, prefix[-1:].expand(7 - prefix.numel())))
    params = replace(
        params,
        **prefixes,
        max_seqlen_q=32,
        max_seqlen_kv=32,
        cp_partition_mode="contiguous",
        total_tokens=None,
        seq_idx=None,
    )
    return params, valid


def _install_cp_graph_callables(stacks, monkeypatch, *, joint_backward=False, num_microbatches=2):
    def static_inputs(layer):
        # Exercise the real THD sample builder. Only its allocation device changes
        # on CPU; CP geometry, graph inputs and production collectives are retained.
        with monkeypatch.context() as context:
            if not next(layer.parameters()).is_cuda:
                context.setattr(torch.cuda, "current_device", lambda: "cpu")
            static = layer.get_layer_static_inputs(32, 1)
        return layer._te_cuda_graph_adapter.get_static_inputs(static)

    monkeypatch.setattr(graph_tests, "_static_inputs", static_inputs)
    return graph_tests._install_eager_graph_callables(
        stacks, joint_backward=joint_backward, num_microbatches=num_microbatches
    )


@pytest.mark.parametrize("cp_size,cp_rank", [(2, 0), (2, 1), (4, 0), (4, 3)])
def test_cp_cuda_graph_static_shapes_and_group_validation(cp_size, cp_rank):
    config = _cp_graph_config(cp_size)
    group = _LayoutGroup(cp_size, cp_rank)
    with pytest.raises(ValueError, match="explicit CP group"):
        CSA2CudaGraphAdapter(config, layer_number=1, is_attention=True)
    with pytest.raises(ValueError, match="group size"):
        CSA2CudaGraphAdapter(
            config, layer_number=1, is_attention=True, cp_group=_LayoutGroup(cp_size * 2, cp_rank)
        )
    for index in (0, 2, 4, 6, 8, 10):
        adapter = CSA2CudaGraphAdapter(
            config, layer_number=index + 1, is_attention=True, cp_group=group
        )
        static = {"hidden_states": torch.empty(32 // cp_size, 1, 128, device="meta")}
        static.update(
            {
                "cu_seqlens_" + suffix: torch.empty(7, dtype=torch.int32, device="meta")
                for suffix in ("q", "kv", "q_padded", "kv_padded")
            }
        )
        values = adapter.get_static_inputs(static)
        params = adapter._packed_params(static)
        assert params.max_seqlen_q == params.max_seqlen_kv == 32
        assert params.cp_group is group and params.local_cp_size == cp_size
        for name, tensor in values.items():
            if name in ("csa2_graph_global_kv", "csa2_graph_indexer_k"):
                assert tensor.shape[0] == 32 // adapter.ratio
            elif name in (
                "csa2_graph_global_indices",
                "csa2_graph_candidate_indices",
                "csa2_graph_candidate_lengths",
            ):
                assert tensor.shape[0] == 32 // cp_size

        def no_replay(*args, **kwargs):
            pytest.fail("Invalid CP metadata must fail before invoking a captured collective")

        for bad in (
            replace(params, cp_group=_LayoutGroup(cp_size, 1 - cp_rank if cp_size == 2 else 1)),
            replace(
                params, cp_group=_LayoutGroup(cp_size, cp_rank)
            ),  # same coordinates, another communicator
            replace(params, local_cp_size=cp_size * 2),
            replace(params, cp_partition_mode="zigzag"),
        ):
            with pytest.raises(ValueError, match="CP|contiguous"):
                adapter.replay(no_replay, values["hidden_states"], packed_seq_params=bad)
        with pytest.raises(ValueError, match="THD"):
            adapter.replay(no_replay, values["hidden_states"])
        wrong_state = CSA2State(
            thd_layout=SimpleNamespace(cp_size=cp_size, cp_rank=(cp_rank + 1) % cp_size)
        )
        with pytest.raises(ValueError, match="different CP shard"):
            adapter.replay(
                no_replay,
                values["hidden_states"],
                packed_seq_params=params,
                cross_layer_state=wrong_state,
            )


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(dict(mhc=False), id="ordinary"),
        pytest.param(dict(coefficient=0.0), id="aux-off"),
        pytest.param({}, id="mhc"),
        pytest.param(dict(joint_backward=True), id="joint-backward"),
        pytest.param(dict(mixed=True), id="graph-eager-consumers"),
        pytest.param(
            dict(joint_backward=True, recompute=["layernorm", "mla_up_proj"]), id="norm-mla"
        ),
        pytest.param(
            dict(joint_backward=True, recompute=["mhc", "layernorm", "mla_up_proj"]),
            id="mhc-norm-mla",
        ),
    ],
)
def test_cp_cuda_graph_boundaries_match_eager(
    cp_groups,
    native_attention,
    loss_tracker,
    monkeypatch,
    cpu_checkpoint_rng,
    cpu_graph_slots,
    options,
):
    _check_cp_graph_boundaries(cp_groups, native_attention, loss_tracker, monkeypatch, **options)


def _check_cp_graph_boundaries(
    cp_groups,
    native_attention,
    loss_tracker,
    monkeypatch,
    *,
    mhc=True,
    coefficient=0.3,
    joint_backward=False,
    mixed=False,
    recompute=None,
    real_te=False,
):
    groups, device = cp_groups
    groups = copy.copy(groups)
    groups.embd = groups.pos_embd = None
    config = _cp_graph_config(groups.cp.size(), coefficient=coefficient, mhc=mhc)
    if recompute:
        config = replace(config, recompute_granularity="selective", recompute_modules=recompute)
    torch.manual_seed(845)
    reference = _pipeline_model(
        replace(config, cuda_graph_impl="none", recompute_granularity=None),
        groups,
        "DE" * 6,
        True,
        True,
        device,
        attention=native_attention,
    ).decoder
    actual = _pipeline_model(
        config, groups, "DE" * 6, True, True, device, attention=native_attention
    ).decoder
    actual.load_state_dict(reference.state_dict())
    calls = (
        _install_real_cp_graph_callables(actual)
        if real_te
        else _install_cp_graph_callables([actual], monkeypatch, joint_backward=joint_backward)
    )
    if mixed:
        for index in (4, 10):  # Eager Reuse must consume the graph's restored CP layout and K.
            actual.layers[index].cuda_graphs = []
    local_rows = 32 // groups.cp.size()
    for cases in (("boundary", "ragged"), ("empty-ranks", "no-sequences")):
        reference.zero_grad(set_to_none=True)
        actual.zero_grad(set_to_none=True)
        pending = []
        for slot, case in enumerate(cases):
            params, _ = _cp_graph_batch(case, device)
            x = torch.randn(local_rows, 1, config.hidden_size, device=device, requires_grad=True)
            ref_x = x.detach().clone().requires_grad_()
            loss_tracker.clear()
            DSAIndexerLossLoggingHelper.tracker["values"].zero_()
            expected = reference(ref_x, None, packed_seq_params=params)
            expected_logs = list(loss_tracker)
            expected_values = DSAIndexerLossLoggingHelper.tracker["values"].detach().clone()
            loss_tracker.clear()
            DSAIndexerLossLoggingHelper.tracker["values"].zero_()
            for layer in actual.layers:
                layer.current_microbatch = slot
            output = actual(x, None, packed_seq_params=params)
            torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(
                DSAIndexerLossLoggingHelper.tracker["values"], expected_values, atol=2e-6, rtol=2e-5
            )
            if not real_te:
                assert len(loss_tracker) == len(expected_logs)
                for record, target in zip(loss_tracker, expected_logs):
                    assert record["reduce_group"] is groups.cp
                    torch.testing.assert_close(record["loss"], target["loss"], atol=2e-6, rtol=2e-5)
            pending.append((output, output.detach().clone(), expected, x, ref_x))
        for output, snapshot, expected, x, ref_x in reversed(pending):
            torch.testing.assert_close(output, snapshot, atol=0, rtol=0)
            probe = torch.randn_like(output) * 0.01
            (output * probe).sum().backward()
            (expected * probe).sum().backward()
            graph_tests._assert_gradient(x.grad, ref_x.grad)
        for (name, parameter), ref_parameter in zip(
            actual.named_parameters(), reference.parameters()
        ):
            try:
                graph_tests._assert_gradient(parameter.grad, ref_parameter.grad)
            except AssertionError as error:
                raise AssertionError(name) from error
    assert calls and {slot for _, slot in calls} == {0, 1}


@pytest.mark.parametrize(
    "cuts,options",
    [
        ((8,), dict(graph_mode="eager")),
        ((7, 8, 10), dict(graph_mode="joint")),
        (
            (7, 8, 10),
            dict(graph_mode="joint", per_token=True, recompute=["mhc", "layernorm", "mla_up_proj"]),
        ),
    ],
)
def test_cp_pp_vpp_cuda_graph_boundaries(
    native_attention, loss_tracker, monkeypatch, cpu_checkpoint_rng, cpu_graph_slots, cuts, options
):
    overlap = len(cuts) > 1
    test_cp_pp_vpp_matches_single_stage(
        native_attention, loss_tracker, monkeypatch, cuts, overlap, overlap, True, **options
    )


def _install_real_cp_graph_callables(stack):
    te_graph = pytest.importorskip("transformer_engine.pytorch.graph")
    make_graphs = te_graph.make_graphed_callables
    if "_num_layers_per_chunk" not in inspect.signature(make_graphs).parameters:
        pytest.skip("Requires the TE pipeline capture interface")
    bodies = tuple(graph_tests._NativeCaptureBody(layer) for layer in stack.layers)
    sample_args, sample_kwargs = [], []
    for _ in range(2):
        for layer in stack.layers:
            values = layer._te_cuda_graph_adapter.get_static_inputs(
                layer.get_layer_static_inputs(32, 1)
            )
            sample_args.append((values.pop("hidden_states"),))
            sample_kwargs.append(values)
    options = dict(
        sample_kwargs=tuple(sample_kwargs),
        _order=[1, 1, -1, -1],
        _num_layers_per_chunk=[len(stack.layers)],
        num_warmup_iters=3,
        allow_unused_input=True,
    )
    if "_reuse_graph_input_output_buffers" in inspect.signature(make_graphs).parameters:
        options["_reuse_graph_input_output_buffers"] = False
    _set_capture_start()
    try:
        graphs = make_graphs(bodies, tuple(sample_args), **options)
    finally:
        _set_capture_end()
    calls = []
    for index, layer in enumerate(stack.layers):
        slots = []
        for slot in range(2):
            i = slot * len(stack.layers) + index
            graph, names = graphs[i], tuple(sample_kwargs[i])

            def replay(*args, graph=graph, names=names, slot=slot, index=index, **kwargs):
                calls.append((index + 1, slot))
                return graph(
                    *args,
                    **{name: kwargs[name] for name in names},
                    is_first_microbatch=kwargs.get("is_first_microbatch", False),
                )

            slots.append(replay)
        layer.cuda_graphs = slots
        if getattr(layer, "_uses_mhc_recompute_cuda_graph_split", lambda: False)():
            layer.set_te_cuda_graph_static_hidden_inputs(
                sample_args[slot * len(stack.layers) + index][0] for slot in range(2)
            )
    return calls


@pytest.mark.parametrize("recompute", [None, ["mhc", "layernorm", "mla_up_proj"]])
def test_cp_transformer_engine_cuda_graph_replay(
    cp_groups, native_attention, loss_tracker, monkeypatch, cpu_checkpoint_rng, recompute
):
    if not torch.cuda.is_available():
        pytest.skip("Real TE graph replay with CP requires CUDA/NCCL")
    _check_cp_graph_boundaries(
        cp_groups, native_attention, loss_tracker, monkeypatch, real_te=True, recompute=recompute
    )
