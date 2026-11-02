# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed-THD DSA with CP under a raw CUDA graph, fed by prebuilt layout tables.

The packed-CP query positions and gathered-key reorder are built at batch
construction (``dsa_layout.prebuild_packed_cp_layout``) and attached to
``PackedSeqParams``. Together with ``cu_seqlens`` they are the only per-pack inputs
of a DSA layer, so one captured forward/backward can serve a different packing of
the same geometry after refreshing both in place.

Needs 2 GPUs (CP=2; extra ranks become data-parallel replicas), Transformer Engine,
and runs the unfused DSA backend. Launch from the Megatron-LM root, e.g.::

    torchrun --nproc-per-node 2 -m pytest -q \
        tests/unit_tests/transformer/experimental_attention_variant/\
test_dsa_prebuilt_cp_layout_cuda_graph.py
"""

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.context_parallel_layout import prebuild_thd_cp_partition_routes
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    copy_prebuilt_packed_cp_layout_,
    get_prebuilt_packed_cp_layout,
    prebuild_packed_cp_layout,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from tests.unit_tests.test_utilities import Utils

_CP_SIZE = 2
_LOCAL_ROWS = 64
_GLOBAL_ROWS = _LOCAL_ROWS * _CP_SIZE
# Same entry count and total for every pack, so cu_seqlens and every table keep their
# shapes; each sequence length is a multiple of 2 * CP. Pack B has a zero-length entry.
_PACK_A = (0, 32, 96, _GLOBAL_ROWS)
_PACK_B = (0, 72, 72, _GLOBAL_ROWS)
_SEED = 1234


def _make_config() -> MLATransformerConfig:
    num_heads = 8
    return MLATransformerConfig(
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        num_layers=1,
        hidden_size=512,
        num_attention_heads=num_heads,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_head_dim=64,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_use_sparse_loss=False,
        dsa_indexer_rotate_activation=False,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        qk_layernorm=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=_CP_SIZE,
        cp_comm_type="allgather",
        apply_rope_fusion=False,
        rope_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        rotary_interleaved=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, multiplier=2.0),
        kv_channels=64,
        num_query_groups=num_heads,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=False,
        attention_backend=AttnBackend.unfused,
        dsa_kernel_backend="none",
    )


def _make_attention(config, pg_collection) -> DSAttention:
    from megatron.core.extensions.transformer_engine import TELinear, TENorm

    indexer = ModuleSpec(
        module=DSAIndexer,
        submodules=DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        ),
    )
    return DSAttention(
        config=config,
        submodules=DSAttentionSubmodules(indexer=indexer),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type="allgather",
        pg_collection=pg_collection,
    ).cuda()


def _packed_seq_params(boundaries, cp_group, *, prebuild: bool) -> PackedSeqParams:
    """Batch-construction metadata: routes + host boundaries, optionally the DSA tables."""
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        # Graph-style static metadata: the packed capacity, as pad_sequence_for_thd
        # sets it for fixed-capacity cu_seqlens. RoPE tables depend on it.
        max_seqlen_q=_GLOBAL_ROWS,
        max_seqlen_kv=_GLOBAL_ROWS,
        cp_group=cp_group,
    )
    prebuild_thd_cp_partition_routes(params, cp_group)
    if prebuild:
        layout = prebuild_packed_cp_layout(params)
        assert layout is not None and layout.geometry.local_rows == _LOCAL_ROWS
    return params


def _make_inputs(config, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    absorbed_dim = config.kv_lora_rank + config.qk_pos_emb_head_dim

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=generator) * scale

    return {
        "query": randn(_LOCAL_ROWS, 1, config.num_attention_heads, absorbed_dim),
        "key": randn(_LOCAL_ROWS, 1, 1, absorbed_dim),
        "x": randn(_LOCAL_ROWS, 1, config.hidden_size, scale=8.0),
        "qr": randn(_LOCAL_ROWS, 1, config.q_lora_rank, scale=8.0),
        "up_v_weight": randn(config.num_attention_heads, config.v_head_dim, config.kv_lora_rank),
        "grad_output": randn(_LOCAL_ROWS, 1, config.num_attention_heads * config.v_head_dim),
    }


_GRAD_INPUTS = ("query", "key", "up_v_weight")


def _forward_backward(attention, inputs, packed_seq_params):
    output = attention(
        query=inputs["query"],
        key=inputs["key"],
        value=None,
        attention_mask=None,
        x=inputs["x"],
        qr=inputs["qr"],
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=packed_seq_params,
        up_v_weight=inputs["up_v_weight"],
    )
    output.backward(inputs["grad_output"].view_as(output))
    return output


def _eager(attention, inputs, packed_seq_params):
    """Eager forward/backward on private copies; returns output and input grads."""
    leaves = {name: value.detach().clone() for name, value in inputs.items()}
    for name in _GRAD_INPUTS:
        leaves[name].requires_grad_(True)
    output = _forward_backward(attention, leaves, packed_seq_params)
    torch.cuda.synchronize()
    return output.detach().clone(), {name: leaves[name].grad.clone() for name in _GRAD_INPUTS}


def _assert_matches(actual, expected, label):
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2, msg=label)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestDSAPrebuiltPackedCPLayoutCudaGraph:
    """A captured packed-THD DSA layer replays correctly after an in-place layout refresh."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_method(self, request):
        if Utils.world_size < _CP_SIZE:
            pytest.skip(f"requires at least {_CP_SIZE} distributed ranks")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=_CP_SIZE,
        )
        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)
        cls = request.cls
        cls.config = _make_config()
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        cls.cp_group = cls.pg_collection.cp
        cls.attention = _make_attention(cls.config, cls.pg_collection)
        yield
        Utils.destroy_model_parallel()

    def test_eager_prebuilt_layout_matches_lazy_layout_on_device(self):
        """The H2D-copied prebuilt tables give the eager result of the lazy memo path."""
        inputs = _make_inputs(self.config, _SEED + parallel_state.get_context_parallel_rank())
        for boundaries in (_PACK_A, _PACK_B):
            lazy = _packed_seq_params(boundaries, self.cp_group, prebuild=False)
            prebuilt = _packed_seq_params(boundaries, self.cp_group, prebuild=True)
            lazy_output, lazy_grads = _eager(self.attention, inputs, lazy)
            prebuilt_output, prebuilt_grads = _eager(self.attention, inputs, prebuilt)

            memo = getattr(lazy, DSAttention._LAYOUT_HOLDER_ATTR)
            layout = get_prebuilt_packed_cp_layout(prebuilt)
            ((memo_positions, memo_reorder, memo_inverse),) = memo.values()
            assert torch.equal(memo_positions, layout.query_positions)
            assert torch.equal(memo_reorder, layout.kv_reorder_idx)
            assert torch.equal(memo_inverse, layout.kv_reorder_inv_idx)
            assert getattr(prebuilt, DSAttention._LAYOUT_HOLDER_ATTR) == {}
            # The tables are bit-identical; the kernels are not guaranteed deterministic.
            _assert_matches(prebuilt_output, lazy_output, "output")
            for name in _GRAD_INPUTS:
                _assert_matches(prebuilt_grads[name], lazy_grads[name], f"d{name}")

    def test_graph_replay_after_in_place_refresh_matches_eager(self):
        """Capture with pack A, refresh cu_seqlens + prebuilt tables for pack B, replay."""
        cp_rank = parallel_state.get_context_parallel_rank()
        inputs_a = _make_inputs(self.config, _SEED + cp_rank)
        inputs_b = _make_inputs(self.config, _SEED + 100 + cp_rank)

        # Eager references, built through the lazy memo path on fresh metadata.
        expected_a = _eager(
            self.attention, inputs_a, _packed_seq_params(_PACK_A, self.cp_group, prebuild=False)
        )
        expected_b = _eager(
            self.attention, inputs_b, _packed_seq_params(_PACK_B, self.cp_group, prebuild=False)
        )
        # The layout change must be observable, or the refresh below proves nothing.
        stale_b = _eager(
            self.attention, inputs_b, _packed_seq_params(_PACK_A, self.cp_group, prebuild=False)
        )
        assert not torch.allclose(stale_b[0].float(), expected_b[0].float(), rtol=2e-2, atol=2e-2)

        static_params = _packed_seq_params(_PACK_A, self.cp_group, prebuild=True)
        static_layout = get_prebuilt_packed_cp_layout(static_params)
        static_tables = {
            name: (table, table.data_ptr()) for name, table in static_layout.named_tables().items()
        }
        static_cu_seqlens = static_params.cu_seqlens_q
        static_inputs = {name: value.detach().clone() for name, value in inputs_a.items()}
        for name in _GRAD_INPUTS:
            static_inputs[name].requires_grad_(True)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                for name in _GRAD_INPUTS:
                    static_inputs[name].grad = None
                _forward_backward(self.attention, static_inputs, static_params)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        assert getattr(static_params, DSAttention._LAYOUT_HOLDER_ATTR) == {}

        for name in _GRAD_INPUTS:
            static_inputs[name].grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            static_output = _forward_backward(self.attention, static_inputs, static_params)
        torch.cuda.synchronize()
        # The capture read the prebuilt tables, not a lazily built memo entry.
        assert getattr(static_params, DSAttention._LAYOUT_HOLDER_ATTR) == {}

        def replay_and_check(expected, label):
            graph.replay()
            torch.cuda.synchronize()
            _assert_matches(static_output, expected[0], f"{label} output")
            for name in _GRAD_INPUTS:
                _assert_matches(static_inputs[name].grad, expected[1][name], f"{label} d{name}")

        replay_and_check(expected_a, "pack A")

        # Refresh every per-pack input in place, as a graph integration would before replay.
        source_params = _packed_seq_params(_PACK_B, self.cp_group, prebuild=True)
        with torch.no_grad():
            static_cu_seqlens.copy_(source_params.cu_seqlens_q)
            for name, value in inputs_b.items():
                static_inputs[name].copy_(value)
        # Negative control: with pack B's cu_seqlens and inputs but pack A's tables, the
        # replay must not reproduce pack B, or the table refresh below proves nothing.
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.allclose(
            static_output.float(), expected_b[0].float(), rtol=2e-2, atol=2e-2
        ), "stale layout tables went unnoticed"
        copy_prebuilt_packed_cp_layout_(static_params, source_params)
        assert static_params.cu_seqlens_q is static_cu_seqlens
        for name, table in static_layout.named_tables().items():
            assert table is static_tables[name][0]
            assert table.data_ptr() == static_tables[name][1]
            assert torch.equal(
                table, get_prebuilt_packed_cp_layout(source_params).named_tables()[name]
            )

        replay_and_check(expected_b, "pack B")

        # And back again, to show the replay follows the refreshed inputs both ways.
        source_params = _packed_seq_params(_PACK_A, self.cp_group, prebuild=True)
        with torch.no_grad():
            static_cu_seqlens.copy_(source_params.cu_seqlens_q)
            for name, value in inputs_a.items():
                static_inputs[name].copy_(value)
        copy_prebuilt_packed_cp_layout_(static_params, source_params)
        replay_and_check(expected_a, "pack A again")

        graph.reset()
        del graph, static_output
        torch.cuda.synchronize()
