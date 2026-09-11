# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import functools
import os
from copy import deepcopy
from datetime import timedelta
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts import BaseInferenceContext, StaticInferenceContext
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.common.embeddings.rotary_pos_embedding import MultimodalRotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.models.hybrid.hybrid_block import (
    HybridStack,
    HybridStackSubmodules,
    HyperConnectionHybridLayer,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec, hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel, _hybrid_logging_pg_kwargs
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant.csa2 import CompressedSparseAttention2
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import divide, is_fa_min_version, is_torch_min_version
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv41 import (
    _make_config as _make_dsv41_config,
)

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    _HAVE_HADAMARD = True
except ImportError:
    _HAVE_HADAMARD = False
    _hadamard_transform = None


def _mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Identity-with-scale stand-in for `fast_hadamard_transform.hadamard_transform`.

    Mirrors the helper in `tests/unit_tests/transformer/experimental_attention_variant/
    test_attention_variant_dsa.py` so that DSA forward tests run in containers that
    don't ship the upstream library.
    """
    return x * scale


def _is_dataclass_instance(value):
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _assert_equal_with_partial_contents(left, right, path="root"):
    """Assert recursive equality while comparing `partial` objects structurally."""
    if isinstance(left, functools.partial) or isinstance(right, functools.partial):
        assert isinstance(left, functools.partial), f"{path}: left is not `partial`"
        assert isinstance(right, functools.partial), f"{path}: right is not `partial`"
        _assert_equal_with_partial_contents(left.func, right.func, f"{path}.func")
        _assert_equal_with_partial_contents(left.args, right.args, f"{path}.args")
        _assert_equal_with_partial_contents(
            left.keywords or {}, right.keywords or {}, f"{path}.keywords"
        )
        return

    if _is_dataclass_instance(left) or _is_dataclass_instance(right):
        assert _is_dataclass_instance(left), f"{path}: left is not a dataclass"
        assert _is_dataclass_instance(right), f"{path}: right is not a dataclass"
        assert type(left) is type(right), f"{path}: dataclass types differ"
        for field in dataclasses.fields(left):
            if field.compare:
                _assert_equal_with_partial_contents(
                    getattr(left, field.name), getattr(right, field.name), f"{path}.{field.name}"
                )
        return

    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict), f"{path}: left is not a dict"
        assert isinstance(right, dict), f"{path}: right is not a dict"
        assert left.keys() == right.keys(), f"{path}: dict keys differ"
        for key in left:
            _assert_equal_with_partial_contents(left[key], right[key], f"{path}[{key!r}]")
        return

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert type(left) is type(right), f"{path}: sequence types differ"
        assert len(left) == len(right), f"{path}: sequence lengths differ"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_equal_with_partial_contents(left_item, right_item, f"{path}[{index}]")
        return

    assert left == right, f"{path}: values differ"


class _DummyHybridLayer(MegatronModule):
    """Minimal same-shape layer used to test HybridModel/mHC plumbing."""

    def __init__(self, config: TransformerConfig, layer_number: int, **_kwargs):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.seen_hidden_shapes = []

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        inference_context=None,
        packed_seq_params=None,
        **_kwargs,
    ):
        self.seen_hidden_shapes.append(tuple(hidden_states.shape))
        return hidden_states + 0.125 * self.proj(hidden_states)


def _get_dummy_hybrid_stack_spec() -> ModuleSpec:
    """Build a HybridStack spec whose layer symbols all resolve to dummy layers."""
    dummy_layer_spec = ModuleSpec(module=_DummyHybridLayer)
    return ModuleSpec(
        module=HybridStack,
        params={"post_layer_norm": False},
        submodules=HybridStackSubmodules(
            mamba_layer=dummy_layer_spec,
            gdn_layer=dummy_layer_spec,
            attention_layer=dummy_layer_spec,
            dsa_layer=dummy_layer_spec,
            mlp_layer=dummy_layer_spec,
            moe_layer=dummy_layer_spec,
        ),
    )


def test_hybrid_logging_process_groups_are_paired():
    tp_group = object()
    dp_cp_group = object()

    assert _hybrid_logging_pg_kwargs(SimpleNamespace()) == {}
    assert _hybrid_logging_pg_kwargs(SimpleNamespace(tp=tp_group, dp_cp=dp_cp_group)) == {
        'tp_group': tp_group,
        'dp_cp_group': dp_cp_group,
    }

    with pytest.raises(ValueError, match="tp.*dp_cp"):
        _hybrid_logging_pg_kwargs(SimpleNamespace(tp=tp_group))
    with pytest.raises(ValueError, match="tp.*dp_cp"):
        _hybrid_logging_pg_kwargs(SimpleNamespace(dp_cp=dp_cp_group))
    with pytest.raises(ValueError, match="tp.*dp_cp"):
        _hybrid_logging_pg_kwargs(SimpleNamespace(tp=tp_group, dp_cp=None))
    with pytest.raises(ValueError, match="tp.*dp_cp"):
        _hybrid_logging_pg_kwargs(SimpleNamespace(tp=None, dp_cp=dp_cp_group))


def test_hybrid_model_constructor_with_mrope():
    Utils.initialize_model_parallel(1, 1)
    try:
        # use_cpu_initialization keeps the embedding off the CUDA RNG tracker; it
        # does not make this a GPU-free test, because MultimodalRotaryEmbedding
        # builds inv_freq on the current CUDA device regardless of the flag.
        model_config = TransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            mrope_section=[2, 3, 3],
            mrope_interleaved=True,
        )
        model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="*-",
            position_embedding_type="mrope",
            rotary_percent=0.25,
        )

        assert isinstance(model.rotary_pos_emb, MultimodalRotaryEmbedding)
        assert model.mrope_section == [2, 3, 3]
    finally:
        Utils.destroy_model_parallel()


def test_hybrid_model_mrope_uses_injected_cp_group():
    """The injected CP group must reach MRoPE, not the parallel_state global."""
    from megatron.core.process_groups_config import ProcessGroupCollection

    Utils.initialize_model_parallel(1, 1)
    try:
        # A distinct group object over the same ranks: if the mrope branch falls
        # back to parallel_state, the identity assertion below fails.
        custom_cp_group = torch.distributed.new_group(
            ranks=list(range(torch.distributed.get_world_size()))
        )
        assert custom_cp_group is not parallel_state.get_context_parallel_group()

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.cp = custom_cp_group

        model_config = TransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            mrope_section=[2, 3, 3],
            mrope_interleaved=True,
        )
        model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="*-",
            position_embedding_type="mrope",
            rotary_percent=0.25,
            pg_collection=pg_collection,
        )

        assert model.rotary_pos_emb.cp_group is custom_cp_group
    finally:
        Utils.destroy_model_parallel()


def test_mrope_stored_cp_group_drives_unpacked_slicing():
    """HybridModel.forward passes cp_group=None on the unpacked path, so the
    group stored at construction time is the one that slices the frequencies."""

    class _StubCPGroup:
        def size(self):
            return 2

    stub_cp_group = _StubCPGroup()
    rotary = MultimodalRotaryEmbedding(
        kv_channels=32, rotary_percent=0.25, interleaved_mrope=True, cp_group=stub_cp_group
    )
    position_ids = torch.arange(4, device=torch.cuda.current_device()).repeat(3, 1, 1)
    seen = {}

    def fake_slice(tensor, seq_dim, cp_group):
        seen['cp_group'] = cp_group
        return tensor

    with patch(
        'megatron.core.models.common.embeddings.rotary_pos_embedding.'
        'get_pos_emb_on_this_cp_rank',
        fake_slice,
    ):
        rotary(position_ids, mrope_section=[1, 1, 2])

    assert seen['cp_group'] is stub_cp_group


@pytest.mark.skipif(
    not is_torch_min_version("2.4.0"),
    reason="torch.distributed.init_device_mesh requires torch >= 2.4.0",
)
@pytest.mark.parametrize("tp_size,cp_size,pp_size", [(2, 1, 4), (1, 1, 8), (8, 1, 1)])
def test_hybrid_model_with_custom_process_groups(tmp_path, tp_size, cp_size, pp_size):
    """Test HybridModel with custom process groups."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        pipeline_model_parallel_size=pp_size,
    )

    try:
        # Create device mesh for custom process groups
        assert torch.distributed.get_world_size() == 8, "Test requires 8 GPUs"

        # Initialize torch.distributed if not already initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')

        dp_size = 1

        # Create HyperCommGrid with dimensions tp, cp, pp, dp.
        grid = HyperCommGrid([tp_size, cp_size, pp_size, dp_size], ["tp", "cp", "pp", "dp"])

        pp_group = grid.create_pg("pp")
        cp_group = grid.create_pg("cp")
        tp_group = grid.create_pg("tp")
        dp_cp_group = grid.create_pg(["cp", "dp"])
        embd_group_ranks = parallel_state.default_embedding_ranks(
            torch.distributed.get_process_group_ranks(pp_group)
        )
        embd_group = torch.distributed.new_group(
            ranks=embd_group_ranks, timeout=timedelta(minutes=30)
        )

        # Create model with custom process groups
        from megatron.core.process_groups_config import ProcessGroupCollection

        pg_collection = ProcessGroupCollection(
            tp=tp_group, cp=cp_group, pp=pp_group, embd=embd_group, dp_cp=dp_cp_group
        )

        # Build pattern with '|' pipeline stage separators: 2 layers per PP stage
        hybrid_layer_pattern = "|".join(["*-"] * pp_size)

        # Configure model with appropriate sizes for parallelism
        model_config = TransformerConfig(
            num_layers=2 * pp_size,  # Scale layers with PP size
            hidden_size=256 * tp_size,
            num_attention_heads=4 * tp_size,  # Scale heads with TP size
            use_cpu_initialization=True,
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            pipeline_model_parallel_size=pp_size,
            pipeline_dtype=torch.bfloat16,
        )

        model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=4,
            hybrid_layer_pattern=hybrid_layer_pattern,
            pg_collection=pg_collection,
        )

        # Basic forward test
        micro_batch_size = 2
        sequence_length = model.max_sequence_length

        model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == divide(model.vocab_size, tp_size)
    finally:
        Utils.destroy_model_parallel()


class TestHybridModel:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        model_config = TransformerConfig(
            num_layers=3,  # 1 Mamba layer, 1 attention layer, 1 MLP layer
            hidden_size=256,  # The Mamba layer places several constraints on this
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        self.model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",  # 1 Mamba, 1 attention, 1 MLP
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.model, HybridModel)

        assert self.model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in self.model.parameters()])
        assert num_weights == 1774872

    def test_constructor_with_hyper_connections(self):
        model_config = TransformerConfig(
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
        )
        model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",
        )

        assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in model.decoder.layers)
        assert model.decoder.hc_head_fn.shape == (
            model_config.num_residual_streams,
            model_config.hidden_size * model_config.num_residual_streams,
        )
        assert model.decoder.hc_head_base.shape == (model_config.num_residual_streams,)
        assert model.decoder.hc_head_scale.shape == (1,)
        assert "decoder.hc_head_fn" in model.state_dict()
        decoder_sharded_state = model.decoder.sharded_state_dict(prefix="decoder.", metadata={})
        assert "decoder.hc_head_fn" in decoder_sharded_state
        assert "decoder.hc_head_base" in decoder_sharded_state
        assert "decoder.hc_head_scale" in decoder_sharded_state
        num_weights = sum([p.numel() for p in model.parameters()])
        assert num_weights > sum([p.numel() for p in self.model.parameters()])

    def test_hyper_connection_recompute_skips_boundary_bda_checkpoint(self, monkeypatch):
        model_config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
            mhc_sinkhorn_iterations=3,
        )
        layer = HyperConnectionHybridLayer(
            config=model_config, layer=_DummyHybridLayer(model_config, layer_number=1)
        )
        hidden_states = torch.randn(
            4, 2, model_config.hidden_size * model_config.num_residual_streams, requires_grad=True
        )
        manager = type("_FakeManager", (), {})()
        manager.is_last_layer_in_recompute_block = True
        seen_bda_managers = []
        seen_inner_managers = []

        def fake_hyper_connection_forward(hidden_states, mhc_recompute_manager=None):
            assert mhc_recompute_manager is manager
            s, b, _ = hidden_states.shape
            n = model_config.num_residual_streams
            c = model_config.hidden_size
            aggregated = hidden_states.view(s, b, n, c).mean(dim=2)
            h_res = torch.empty(s, b, n, n, dtype=hidden_states.dtype)
            h_post = torch.empty(s, b, n, dtype=hidden_states.dtype)
            return aggregated, h_res, h_post, hidden_states

        def fake_fused_h_res_h_post_bda(
            h_res,
            original_residual,
            h_post,
            layer_output_with_bias,
            dropout_prob,
            training,
            fused,
            manager=None,
        ):
            seen_bda_managers.append(manager)
            return original_residual

        def fake_inner_fast_path(*_args, mhc_recompute_manager=None, **_kwargs):
            seen_inner_managers.append(mhc_recompute_manager)
            return None

        monkeypatch.setattr(layer.hyper_connection, "forward", fake_hyper_connection_forward)
        monkeypatch.setattr(
            layer.hyper_connection, "fused_h_res_h_post_bda", fake_fused_h_res_h_post_bda
        )
        monkeypatch.setattr(
            layer, "_call_inner_transformer_layer_without_local_bda", fake_inner_fast_path
        )

        output, _ = layer(hidden_states, attention_mask=None, mhc_recompute_manager=manager)
        assert output is hidden_states
        assert seen_bda_managers == [None]
        assert seen_inner_managers == [manager]

        manager.is_last_layer_in_recompute_block = False
        layer(hidden_states, attention_mask=None, mhc_recompute_manager=manager)
        assert seen_bda_managers[-1] is manager
        assert seen_inner_managers[-1] is manager

    def test_forward_with_hyper_connections(self):
        model_config = TransformerConfig(
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
        )
        model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",
        )
        model.cuda()

        sequence_length = model.max_sequence_length
        micro_batch_size = 2
        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == model.vocab_size

    def test_dummy_hybrid_model_with_hyper_connections_forward_backward(self):
        model_config = TransformerConfig(
            num_layers=3,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
            mhc_sinkhorn_iterations=3,
        )
        model = HybridModel(
            config=model_config,
            hybrid_stack_spec=_get_dummy_hybrid_stack_spec(),
            vocab_size=64,
            max_sequence_length=8,
            hybrid_layer_pattern="M*-",
            parallel_output=False,
        )

        assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in model.decoder.layers)
        assert all(
            isinstance(layer.inner_layer, _DummyHybridLayer) for layer in model.decoder.layers
        )

        model.cuda()
        sequence_length = model.max_sequence_length
        micro_batch_size = 2
        data = torch.arange(sequence_length, dtype=torch.int64, device='cuda')
        input_ids = data.repeat((micro_batch_size, 1))
        position_ids = data.repeat((micro_batch_size, 1))

        logits = model.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=None)

        assert logits.shape == (micro_batch_size, sequence_length, model.vocab_size)
        assert torch.isfinite(logits).all()

        logits.float().mean().backward()

        for layer in model.decoder.layers:
            assert layer.inner_layer.seen_hidden_shapes == [
                (sequence_length, micro_batch_size, model_config.hidden_size)
            ]
            assert layer.inner_layer.proj.weight.grad is not None
            assert layer.hyper_connection.mapping_proj.weight.grad is not None
            assert torch.isfinite(layer.inner_layer.proj.weight.grad).all()
            assert torch.isfinite(layer.hyper_connection.mapping_proj.weight.grad).all()

    def test_set_input_tensor(self):
        config: TransformerConfig = self.model.config
        sequence_length = self.model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        self.model.set_input_tensor(input_tensor)

        assert self.model.decoder.input_tensor.shape[0] == sequence_length
        assert self.model.decoder.input_tensor.shape[1] == micro_batch_size
        assert self.model.decoder.input_tensor.shape[2] == config.hidden_size

    def test_forward(self):
        sequence_length = self.model.max_sequence_length
        micro_batch_size = 2

        self.model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = self.model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.model.vocab_size

    def test_forward_packed_sequence(self):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        model_config = TransformerConfig(
            num_layers=3,  # 1 Mamba layer, 1 attention layer, 1 MLP layer
            hidden_size=256,  # The Mamba layer places several constraints on this
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,  # Needed for backend=flash
            params_dtype=torch.bfloat16,  # Needed for backend=flash
            attention_backend=AttnBackend.flash,  # Needed for packed sequence
        )
        vocab_size = 100
        model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=12,
            hybrid_layer_pattern="M*-",  # 1 Mamba, 1 attention, 1 MLP
        )

        sequence_length = model.max_sequence_length
        micro_batch_size = 1  # must be 1 for packed sequence

        model.cuda()

        data = [i % vocab_size for i in range(sequence_length)]
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        lengths = [4, 3, 5]
        assert sum(lengths) == sequence_length
        positions = [i for n in lengths for i in range(n)]
        position_ids = (
            torch.tensor(positions, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        )
        attention_mask = None

        cumsum = [0] + list(accumulate(lengths))
        cu_seqlens = torch.tensor(cumsum, dtype=torch.int32).cuda()
        max_seqlen = max(lengths)

        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            total_tokens=sequence_length,
        )

        logits = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == model.vocab_size

    def test_inference(self):
        micro_batch_size = 2
        inference_context: BaseInferenceContext = StaticInferenceContext(
            max_batch_size=micro_batch_size, max_sequence_length=self.model.max_sequence_length
        )
        prompt_length = self.model.max_sequence_length - 1

        self.model.cuda()

        with InferenceMode.active():
            # load-context/first-output-token, step/generate
            for offset in (0, prompt_length):
                if offset == 0:
                    sequence_length = prompt_length
                else:
                    sequence_length = 1
                inference_context.sequence_len_offset = offset

                data = list(range(sequence_length))
                input_ids = (
                    torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
                )
                position_ids = (
                    torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
                )
                attention_mask = torch.ones(
                    (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
                ).cuda()

                logits = self.model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    inference_context=inference_context,
                    runtime_gather_output=True,
                )

                assert logits.shape[0] == micro_batch_size
                # StaticInferenceContext always sets materialize_only_last_token_logits=True.
                assert logits.shape[1] == 1
            assert logits.shape[2] == self.model.vocab_size

    def test_save_load(self, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(self.model.state_dict(), path)

        self.model.load_state_dict(torch.load(path))

    def test_layer_numbers(self):
        """
        The layer numbers should start at one (for the embedding # layer) and go up
        incrementally from there. This is required for PEFT to work.
        """
        model = self.model
        for expected, layer in enumerate(model.decoder.layers, start=1):
            assert expected == layer.layer_number, "layer numbers are incorrect"


class TestHybridQKLayernorm:

    # Subclasses override these to retarget the same tests at MLA's
    # `mla_layer.kv_layernorm` or DSA's `dsa_layer.kv_layernorm`. The base class
    # exercises the SelfAttention path with `attention_layer.k_layernorm`.
    _attention_layer_attr = 'attention_layer'
    _k_norm_attr = 'k_layernorm'

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_model(self, spec=None, **config_overrides):
        if spec is None:
            spec = hybrid_stack_spec
        config = TransformerConfig(
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            **config_overrides,
        )
        return HybridModel(
            config=config,
            hybrid_stack_spec=spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",
        )

    def _get_attention_layer(self, model):
        """Return the self-attention submodule that owns a `q_layernorm`."""
        for layer in model.decoder.layers:
            if hasattr(layer, 'self_attention') and hasattr(layer.self_attention, 'q_layernorm'):
                return layer.self_attention
        return None

    def _get_k_norm(self, attn):
        return getattr(attn, self._k_norm_attr)

    def test_trivial_qk_norm_by_default(self):
        """Without qk_layernorm, attention has trivial q/k layernorm."""
        from megatron.core.transformer.identity_op import IdentityOp

        model = self._build_model()
        attn = self._get_attention_layer(model)
        assert attn is not None
        assert attn.q_layernorm is None or isinstance(attn.q_layernorm, IdentityOp)
        k_norm = self._get_k_norm(attn)
        assert k_norm is None or isinstance(k_norm, IdentityOp)

    def test_qk_layernorm_from_config(self):
        """config.qk_layernorm=True creates q/k layernorm even with static spec."""
        model = self._build_model(qk_layernorm=True)
        attn = self._get_attention_layer(model)
        assert attn is not None
        # TENorm is a factory (__new__ returns a TE LayerNorm/RMSNorm), so we
        # verify the norm was created rather than checking for a specific type.
        assert attn.q_layernorm is not None
        assert self._get_k_norm(attn) is not None

    def test_qk_l2_norm_from_config(self):
        """config.qk_l2_norm=True creates L2Norm q/k layernorm."""
        from megatron.core.transformer.torch_norm import L2Norm

        model = self._build_model(qk_l2_norm=True)
        attn = self._get_attention_layer(model)
        assert attn is not None
        assert isinstance(attn.q_layernorm, L2Norm)
        assert isinstance(self._get_k_norm(attn), L2Norm)

    def test_spec_provided_norm_not_overwritten(self):
        """When the spec already provides q/k layernorm, config doesn't override it."""
        import copy

        from megatron.core.transformer.identity_op import IdentityOp

        # Build a spec that explicitly sets q/k layernorm to IdentityOp on the
        # attention layer that this subclass exercises.
        spec = copy.deepcopy(hybrid_stack_spec)
        attn_submodules = getattr(
            spec.submodules, self._attention_layer_attr
        ).submodules.self_attention.submodules
        attn_submodules.q_layernorm = IdentityOp
        setattr(attn_submodules, self._k_norm_attr, IdentityOp)

        model = self._build_model(spec=spec, qk_layernorm=True)
        attn = self._get_attention_layer(model)
        assert attn is not None
        assert isinstance(attn.q_layernorm, IdentityOp)
        assert isinstance(self._get_k_norm(attn), IdentityOp)

    def test_forward_with_qk_layernorm(self):
        """HybridModel forward pass works with qk_layernorm enabled."""
        model = self._build_model(qk_layernorm=True)
        model.cuda()

        sequence_length = 4
        micro_batch_size = 2
        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == 100


class TestHybridMLAQKLayernorm(TestHybridQKLayernorm):
    """Tests QK norm configuration of HybridModel with MLA."""

    _attention_layer_attr = 'mla_layer'
    _k_norm_attr = 'kv_layernorm'

    def _build_model(self, spec=None, **config_overrides):
        if spec is None:
            spec = hybrid_stack_spec
        config = MLATransformerConfig(
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            **config_overrides,
        )
        return HybridModel(
            config=config,
            hybrid_stack_spec=spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M+-",
        )

    def test_qk_l2_norm_from_config(self):
        with pytest.raises(ValueError, match="qk_l2_norm is not supported"):
            super().test_qk_l2_norm_from_config()


class TestHybridDSAQKLayernorm(TestHybridQKLayernorm):
    """Tests QK norm configuration of HybridModel with DSA."""

    _attention_layer_attr = 'dsa_layer'
    _k_norm_attr = 'kv_layernorm'

    @pytest.fixture(autouse=True)
    def _patch_hadamard_if_needed(self):
        if not _HAVE_HADAMARD:
            with patch(
                'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
                _mock_hadamard_transform,
            ):
                yield
        else:
            yield

    def test_spec_provided_norm_not_overwritten(self):
        # DSA cannot fuse the QK norm into the up-projection, so a trivial
        # `IdentityOp` spec is auto-promoted to `TENorm` when `qk_layernorm=True`.
        # Finer-grained spec-respect behavior is covered by TestDSAQKNormResolution.
        pytest.skip("DSA auto-promotes IdentityOp to TENorm; covered by TestDSAQKNormResolution.")

    def _build_model(self, spec=None, **config_overrides):
        if spec is None:
            spec = hybrid_stack_spec
        config_kwargs = dict(
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            add_bias_linear=False,
            # AbsorbedMLASelfAttention forwards `x` and `qr` to the DSA core attention; without
            # this, the DSA core attention's forward fails on missing positional arguments.
            experimental_attention_variant="dsa",
            # DSA-specific settings; defaults are None and DSAIndexer requires them.
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            # The indexer-loss path runs in training mode and multiplies by this coefficient;
            # leaving it at the default `None` raises `TypeError: ... 'Tensor' and 'NoneType'`.
            dsa_indexer_loss_coeff=1.0,
            # DSA's `rotate_activation` (Hadamard rotation) only supports bf16 input.
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        config_kwargs.update(config_overrides)
        config = MLATransformerConfig(**config_kwargs)
        return HybridModel(
            config=config,
            hybrid_stack_spec=spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="MD-",
        )

    def test_qk_l2_norm_from_config(self):
        with pytest.raises(ValueError, match="qk_l2_norm is not supported"):
            super().test_qk_l2_norm_from_config()


class _MLAQKNormTestBase:
    """Common machinery for MLA/DSA QK-norm spec tests.

    Subclasses override `experimental_attention_variant` and
    `hybrid_layer_pattern` to target the MLA vs. DSA code path.
    """

    experimental_attention_variant = None
    hybrid_layer_pattern = "M+-"
    mla_layer_attr = "mla_layer"

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _make_spec(self, **submodule_overrides):
        """Return a copy of `hybrid_stack_spec` with MLA/DSA submodule overrides."""
        import copy

        spec = copy.deepcopy(hybrid_stack_spec)
        mla_submodules = getattr(
            spec.submodules, self.mla_layer_attr
        ).submodules.self_attention.submodules
        for key, value in submodule_overrides.items():
            setattr(mla_submodules, key, value)
        return spec

    def _build_model(self, spec=None, **config_overrides):
        if spec is None:
            spec = hybrid_stack_spec
        config_kwargs = dict(
            num_layers=3, hidden_size=256, num_attention_heads=4, use_cpu_initialization=True
        )
        if self.experimental_attention_variant is not None:
            config_kwargs["experimental_attention_variant"] = self.experimental_attention_variant
            if self.experimental_attention_variant == "dsa":
                # Must not be True for DSA.
                config_kwargs.setdefault("add_bias_linear", False)
                # DSAIndexer requires these; their config defaults are None.
                config_kwargs.setdefault("dsa_indexer_n_heads", 8)
                config_kwargs.setdefault("dsa_indexer_head_dim", 64)
                config_kwargs.setdefault("dsa_indexer_topk", 32)

        config_kwargs.update(config_overrides)
        config = MLATransformerConfig(**config_kwargs)
        return HybridModel(
            config=config,
            hybrid_stack_spec=spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern=self.hybrid_layer_pattern,
        )

    def _get_mla_attention(self, model):
        """Return the attention submodule for the selected MLA variant, or None."""
        if self.experimental_attention_variant == "dsa":
            from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
                AbsorbedMLASelfAttention,
            )

            attention_cls = AbsorbedMLASelfAttention
        else:
            from megatron.core.transformer.multi_latent_attention import MLASelfAttention

            attention_cls = MLASelfAttention

        for layer in model.decoder.layers:
            if hasattr(layer, 'self_attention') and isinstance(layer.self_attention, attention_cls):
                return layer.self_attention
        return None


class TestMLAQKNormSpecValidation(_MLAQKNormTestBase):
    """Tests QK norm spec validation in `MLASelfAttention`.

    These errors guard against silently ignoring a configured norm or
    double-applying one through a fused norm+linear.
    """

    experimental_attention_variant = None
    hybrid_layer_pattern = "M+-"
    mla_layer_attr = "mla_layer"

    def test_q_norm_without_q_lora_rank_raises(self):
        """When `q_lora_rank is None`, a non-trivial `q_layernorm` would
        never be reached and must error out.
        """
        from megatron.core.extensions.transformer_engine import TENorm

        spec = self._make_spec(q_layernorm=TENorm)
        with pytest.raises(ValueError, match=r"q_lora_rank is None"):
            self._build_model(spec=spec, q_lora_rank=None)

    def test_q_norm_without_q_lora_rank_hint_for_non_fused_linear(self):
        """Error message hints at fused linear when `linear_q_proj` is non-fused."""
        from megatron.core.extensions.transformer_engine import TENorm

        spec = self._make_spec(q_layernorm=TENorm)
        with pytest.raises(ValueError, match=r"fused norm\+linear for"):
            self._build_model(spec=spec, q_lora_rank=None)

    def test_fused_linear_q_up_with_q_norm_raises(self):
        """Non-trivial `q_layernorm` combined with a fused `linear_q_up_proj`
        would apply the norm twice.
        """
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
            TENorm,
        )

        spec = self._make_spec(q_layernorm=TENorm, linear_q_up_proj=TELayerNormColumnParallelLinear)
        with pytest.raises(ValueError, match=r"fused norm\+linear"):
            self._build_model(spec=spec)

    def test_fused_linear_kv_up_with_kv_norm_raises(self):
        """Non-trivial `kv_layernorm` combined with a fused `linear_kv_up_proj`
        would apply the norm twice.
        """
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
            TENorm,
        )

        spec = self._make_spec(
            kv_layernorm=TENorm, linear_kv_up_proj=TELayerNormColumnParallelLinear
        )
        with pytest.raises(ValueError, match=r"fused norm\+linear"):
            self._build_model(spec=spec)


class TestMLAQKNormResolution(_MLAQKNormTestBase):
    """Tests `_resolve_qk_norm_config` for MLA.

    Covers fusion auto-selection, spec overrides, and the "disabled"-path
    guards that reject fused/explicit norms when `qk_layernorm` is off.
    """

    experimental_attention_variant = None
    hybrid_layer_pattern = "M+-"
    mla_layer_attr = "mla_layer"

    def test_qk_layernorm_fuses_kv_up_by_default(self):
        """With default (trivial) `kv_layernorm`, enabling `qk_layernorm`
        auto-selects the fused `TELayerNormColumnParallelLinear` for KV up.
        """
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
        from megatron.core.transformer.identity_op import IdentityOp

        model = self._build_model(qk_layernorm=True)
        attn = self._get_mla_attention(model)
        assert attn is not None
        assert isinstance(attn.linear_kv_up_proj, TELayerNormColumnParallelLinear)
        assert isinstance(attn.kv_layernorm, IdentityOp)

    def test_spec_q_norm_disables_q_up_fusion(self):
        """A non-trivial `q_layernorm` from the spec must force a non-fused
        `linear_q_up_proj` so the norm isn't applied on top of a fused one.
        """
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TELayerNormColumnParallelLinear,
            TENorm,
        )

        spec = self._make_spec(q_layernorm=TENorm)
        model = self._build_model(spec=spec, qk_layernorm=True)
        attn = self._get_mla_attention(model)
        assert attn is not None
        assert isinstance(attn.linear_q_up_proj, TEColumnParallelLinear)
        assert not isinstance(attn.linear_q_up_proj, TELayerNormColumnParallelLinear)
        # The spec's norm is actually used; it's not reset to IdentityOp.
        assert attn.q_layernorm is not None
        from megatron.core.transformer.identity_op import IdentityOp

        assert not isinstance(attn.q_layernorm, IdentityOp)

    def test_spec_kv_norm_disables_kv_up_fusion(self):
        """Mirror of `test_spec_q_norm_disables_q_up_fusion` for KV."""
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TELayerNormColumnParallelLinear,
            TENorm,
        )

        spec = self._make_spec(kv_layernorm=TENorm)
        model = self._build_model(spec=spec, qk_layernorm=True)
        attn = self._get_mla_attention(model)
        assert attn is not None
        assert isinstance(attn.linear_kv_up_proj, TEColumnParallelLinear)
        assert not isinstance(attn.linear_kv_up_proj, TELayerNormColumnParallelLinear)
        from megatron.core.transformer.identity_op import IdentityOp

        assert not isinstance(attn.kv_layernorm, IdentityOp)

    def test_disabled_qk_layernorm_rejects_fused_linear_q_up(self):
        """When `qk_layernorm` is off, spec must not force fused linear_q_up_proj."""
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

        spec = self._make_spec(linear_q_up_proj=TELayerNormColumnParallelLinear)
        with pytest.raises(ValueError, match=r"supposed to be disabled"):
            self._build_model(spec=spec)

    def test_disabled_qk_layernorm_rejects_fused_linear_kv_up(self):
        """When `qk_layernorm` is off, spec must not force fused linear_kv_up_proj."""
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

        spec = self._make_spec(linear_kv_up_proj=TELayerNormColumnParallelLinear)
        with pytest.raises(ValueError, match=r"supposed to be disabled"):
            self._build_model(spec=spec)

    def test_disabled_qk_layernorm_rejects_spec_norms(self):
        """When `qk_layernorm` is off, spec must not carry explicit q/kv layernorms."""
        from megatron.core.extensions.transformer_engine import TENorm

        for overrides in (
            {"q_layernorm": TENorm},
            {"kv_layernorm": TENorm},
            {"q_layernorm": TENorm, "kv_layernorm": TENorm},
        ):
            spec = self._make_spec(**overrides)
            with pytest.raises(ValueError, match=r"supposed to be disabled"):
                self._build_model(spec=spec)


class TestDSAQKNormResolution(_MLAQKNormTestBase):
    """Tests `_resolve_qk_norm_config` for DSA.

    DSA requires non-fused Q/KV up projections and explicit norms;
    the fused optimization valid for MLA must be rejected here.
    """

    experimental_attention_variant = "dsa"
    hybrid_layer_pattern = "MD-"
    mla_layer_attr = "dsa_layer"

    def test_qk_layernorm_uses_unfused_linear_and_te_norm(self):
        """With default spec, DSA + `qk_layernorm=True` uses non-fused
        `TEColumnParallelLinear` and `TENorm` for Q/KV.
        """
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TELayerNormColumnParallelLinear,
        )
        from megatron.core.transformer.identity_op import IdentityOp

        model = self._build_model(qk_layernorm=True)
        attn = self._get_mla_attention(model)
        assert attn is not None
        assert isinstance(attn.linear_q_up_proj, TEColumnParallelLinear)
        assert not isinstance(attn.linear_q_up_proj, TELayerNormColumnParallelLinear)
        assert isinstance(attn.linear_kv_up_proj, TEColumnParallelLinear)
        assert not isinstance(attn.linear_kv_up_proj, TELayerNormColumnParallelLinear)
        assert not isinstance(attn.q_layernorm, IdentityOp)
        assert not isinstance(attn.kv_layernorm, IdentityOp)

    def test_qk_layernorm_without_q_lora_rank_raises(self):
        """DSA cannot apply Q norm when `q_lora_rank is None`."""
        with pytest.raises(ValueError, match=r"q_lora_rank is None.*not supported for DSA"):
            self._build_model(qk_layernorm=True, q_lora_rank=None)

    def test_qk_layernorm_rejects_fused_linear_q_up(self):
        """DSA does not support the fused norm+linear optimization."""
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

        spec = self._make_spec(linear_q_up_proj=TELayerNormColumnParallelLinear)
        with pytest.raises(ValueError, match=r"not supported for DSA"):
            self._build_model(spec=spec, qk_layernorm=True)

    def test_qk_layernorm_without_q_lora_rejects_fused_linear_q(self):
        """DSA does not support fused `linear_q_proj` when `q_lora_rank=None`."""
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

        spec = self._make_spec(linear_q_proj=TELayerNormColumnParallelLinear)
        with pytest.raises(ValueError, match=r"not supported for DSA"):
            self._build_model(spec=spec, qk_layernorm=True, q_lora_rank=None)

    def test_disabled_qk_layernorm_rejects_fused_linear_kv_up(self):
        """When `qk_layernorm` is off, spec must not force fused linear_kv_up_proj."""
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

        spec = self._make_spec(linear_kv_up_proj=TELayerNormColumnParallelLinear)
        with pytest.raises(ValueError, match=r"supposed to be disabled"):
            self._build_model(spec=spec)

    def test_disabled_qk_layernorm_rejects_spec_norms(self):
        """When `qk_layernorm` is off, spec must not carry explicit q/kv layernorms."""
        from megatron.core.extensions.transformer_engine import TENorm

        for overrides in (
            {"q_layernorm": TENorm},
            {"kv_layernorm": TENorm},
            {"q_layernorm": TENorm, "kv_layernorm": TENorm},
        ):
            spec = self._make_spec(**overrides)
            with pytest.raises(ValueError, match=r"supposed to be disabled"):
                self._build_model(spec=spec)


class TestMLADownProjFusion:
    """Tests `HybridStack._fuse_mla_down_proj`.

    The method rewrites the MLA `ModuleSpec` in place on a deep-copied
    `HybridStackSubmodules` when `config.mla_down_proj_fusion=True`, swapping
    the self-attention module to `FusedMLASelfAttention` and collapsing the
    separate q/kv down projections into a single fused `linear_qkv_down_proj`
    that also absorbs the input layernorm.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _fresh_submodules(self):
        """Return a deep copy of `hybrid_stack_spec.submodules` so tests don't
        share state through `hybrid_stack_spec`.
        """
        import copy

        return copy.deepcopy(hybrid_stack_spec.submodules)

    def _call_fuse(self, submodules, *, mla_down_proj_fusion):
        """Invoke `_fuse_mla_down_proj` as an unbound method with a minimal
        stub for `self`. The method only reads `self.config`, so we can avoid
        constructing a full `HybridStack`.
        """
        from megatron.core.models.hybrid.hybrid_block import HybridStack

        stub = SimpleNamespace(config=SimpleNamespace(mla_down_proj_fusion=mla_down_proj_fusion))
        # Mimic the call-site check in `HybridStack.__init__`.
        if getattr(stub.config, "mla_down_proj_fusion", False):
            submodules = HybridStack._fuse_mla_down_proj(stub, submodules)
        return submodules

    def _build_model(self, pattern="M+-", **config_overrides):
        config_kwargs = dict(
            num_layers=3, hidden_size=256, num_attention_heads=4, use_cpu_initialization=True
        )
        config_kwargs.update(config_overrides)
        config = MLATransformerConfig(**config_kwargs)
        return HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern=pattern,
        )

    def _get_layer_with_mla(self, model):
        """Return the layer whose self-attention is an `MLASelfAttention`
        (which includes its `FusedMLASelfAttention` subclass).
        """
        from megatron.core.transformer.multi_latent_attention import MLASelfAttention

        for layer in model.decoder.layers:
            if hasattr(layer, 'self_attention') and isinstance(
                layer.self_attention, MLASelfAttention
            ):
                return layer
        return None

    def test_disabled_returns_spec_unchanged(self):
        """Flag off: method returns the same object, no copying or rewriting."""
        submodules = self._fresh_submodules()
        result = self._call_fuse(submodules, mla_down_proj_fusion=False)
        assert result is submodules

    def test_enabled_rewrites_mla_spec(self):
        """Flag on: MLA spec is swapped to the fused module and fused linear."""
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
        from megatron.core.transformer.identity_op import IdentityOp
        from megatron.core.transformer.multi_latent_attention import FusedMLASelfAttention

        submodules = self._fresh_submodules()
        result = self._call_fuse(submodules, mla_down_proj_fusion=True)

        mla_spec = result.mla_layer
        assert mla_spec.submodules.input_layernorm is IdentityOp
        assert mla_spec.submodules.self_attention.module is FusedMLASelfAttention

        attn_submodules = mla_spec.submodules.self_attention.submodules
        assert attn_submodules.linear_qkv_down_proj is TELayerNormColumnParallelLinear
        assert attn_submodules.linear_q_down_proj is None
        assert attn_submodules.linear_kv_down_proj is None

    def test_enabled_sets_sharded_state_dict_keys_map(self):
        """The keys map is written on the MLA layer submodules for checkpoint
        compatibility with pre-fusion checkpoints.
        """
        submodules = self._fresh_submodules()
        result = self._call_fuse(submodules, mla_down_proj_fusion=True)

        keys_map = result.mla_layer.submodules.sharded_state_dict_keys_map
        assert keys_map == {
            "self_attention.linear_q_down_proj.layer_norm_": "input_layernorm.",
            "self_attention.linear_kv_down_proj.layer_norm_": "input_layernorm.",
            "self_attention.linear_qkv_down_proj.layer_norm_": "input_layernorm.",
        }

    def test_enabled_deep_copies_input_submodules(self):
        """The caller's submodules object must not be mutated – the method
        deep-copies before rewriting, so callers can safely reuse their spec.
        """
        from megatron.core.transformer.multi_latent_attention import (
            FusedMLASelfAttention,
            MLASelfAttention,
        )

        submodules = self._fresh_submodules()
        original_mla_module = submodules.mla_layer.submodules.self_attention.module
        original_q_down_proj = (
            submodules.mla_layer.submodules.self_attention.submodules.linear_q_down_proj
        )
        assert original_mla_module is MLASelfAttention  # sanity check of baseline

        result = self._call_fuse(submodules, mla_down_proj_fusion=True)

        # Original is unchanged.
        assert submodules.mla_layer.submodules.self_attention.module is original_mla_module
        assert (
            submodules.mla_layer.submodules.self_attention.submodules.linear_q_down_proj
            is original_q_down_proj
        )
        # And result is a different object than the input.
        assert result is not submodules
        assert result.mla_layer is not submodules.mla_layer
        # Plus the fused module only shows up on the returned copy.
        assert result.mla_layer.submodules.self_attention.module is FusedMLASelfAttention

    def test_enabled_leaves_dsa_layer_alone(self):
        """MLA fusion must not rewrite the absorbed DSA attention specification."""
        from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
            AbsorbedMLASelfAttention,
        )
        from megatron.core.transformer.multi_latent_attention import FusedMLASelfAttention

        submodules = self._fresh_submodules()
        result = self._call_fuse(submodules, mla_down_proj_fusion=True)

        assert result.dsa_layer.submodules.self_attention.module is AbsorbedMLASelfAttention
        assert result.dsa_layer.submodules.self_attention.module is not FusedMLASelfAttention
        # DSA's down projections must remain non-`None` (they're still used
        # via the unfused path).
        assert result.dsa_layer.submodules.self_attention.submodules.linear_q_down_proj is not None
        assert result.dsa_layer.submodules.self_attention.submodules.linear_kv_down_proj is not None

    def test_enabled_leaves_non_mla_layers_alone(self):
        """Unrelated layer specs (mamba, attention, mlp) must survive unchanged."""
        submodules = self._fresh_submodules()
        original_mamba = submodules.mamba_layer
        original_attention = submodules.attention_layer
        original_mlp = submodules.mlp_layer

        result = self._call_fuse(submodules, mla_down_proj_fusion=True)

        _assert_equal_with_partial_contents(result.mamba_layer, original_mamba)
        _assert_equal_with_partial_contents(result.attention_layer, original_attention)
        _assert_equal_with_partial_contents(result.mlp_layer, original_mlp)

    def test_model_uses_fused_mla_when_enabled(self):
        """Integration: a full HybridModel built with the flag uses
        `FusedMLASelfAttention`.
        """
        from megatron.core.transformer.multi_latent_attention import FusedMLASelfAttention

        model = self._build_model(mla_down_proj_fusion=True)
        layer = self._get_layer_with_mla(model)
        assert layer is not None
        assert isinstance(layer.self_attention, FusedMLASelfAttention)
        # And the fused down projection is present on the attention module.
        assert hasattr(layer.self_attention, "linear_qkv_down_proj")

    def test_model_uses_unfused_mla_when_disabled(self):
        """Integration: with the flag off, MLA layers use the standard
        `MLASelfAttention` (never the fused subclass).
        """
        from megatron.core.transformer.multi_latent_attention import (
            FusedMLASelfAttention,
            MLASelfAttention,
        )

        model = self._build_model(mla_down_proj_fusion=False)
        layer = self._get_layer_with_mla(model)
        assert layer is not None
        assert isinstance(layer.self_attention, MLASelfAttention)
        assert not isinstance(layer.self_attention, FusedMLASelfAttention)

    def test_enabled_replaces_input_layernorm_with_identity(self):
        """Integration: because the fused down-proj absorbs the input
        layernorm, the transformer layer's own `input_layernorm` must be
        `IdentityOp`.
        """
        from megatron.core.transformer.identity_op import IdentityOp

        model = self._build_model(mla_down_proj_fusion=True)
        layer = self._get_layer_with_mla(model)
        assert layer is not None
        assert isinstance(layer.input_layernorm, IdentityOp)

    def test_forward_with_fused_mla(self):
        """Integration: forward pass works with `mla_down_proj_fusion=True`."""
        model = self._build_model(mla_down_proj_fusion=True)
        model.cuda()

        sequence_length = 4
        micro_batch_size = 2
        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == 100


class TestHybridWithDynamicInference:
    """Tests HybridModel with dynamic inference."""

    @torch.inference_mode()
    def setup_method(self, method):
        fp8_available, reason_for_no_fp8 = check_fp8_support()
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)

        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        model_config = TransformerConfig(
            num_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            fp8="hybrid",
            fp8_recipe="tensorwise",
        )

        self.model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=DynamicInferenceContext.TOKEN_ROUNDER,
            hybrid_layer_pattern="M*",  # 1 Mamba, 1 attention
        )
        self.model = Float16Module(self.model.config, self.model)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_dynamic_inference_padding_with_fp8(self):
        """
        Tests that logits for padded tokens are zeroed out for fp8 inference.
        """
        self.model.cuda()
        self.model.eval()
        config = self.model.config

        mamba_inference_state_config = MambaInferenceStateConfig.from_model(self.model.module)

        inference_context = DynamicInferenceContext(
            model_config=self.model.config,
            inference_config=InferenceConfig(
                max_sequence_length=self.model.module.max_sequence_length,
                buffer_size_gb=1.0,
                block_size_tokens=256,
                materialize_only_last_token_logits=False,
                mamba_inference_state_config=mamba_inference_state_config,
            ),
        )

        # Add a request with 10 tokens. Since 10 is not a multiple of 64 (TOKEN_ROUNDER),
        # this will create padding up to the padded length of 64.
        active_token_count = 10
        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=torch.arange(0, active_token_count, dtype=torch.long, device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )
        inference_context.add_request(request)

        # Prepares the context, including calculating the padded token count.
        inference_context.initialize_attention_state()

        assert inference_context.active_token_count == active_token_count
        assert inference_context.padded_active_token_count == DynamicInferenceContext.TOKEN_ROUNDER

        # Prepare inputs for the forward pass.
        padded_token_count = inference_context.padded_active_token_count
        input_ids, position_ids = inference_context.current_input_and_position_ids()

        # Run the forward pass with inference parameters.
        with InferenceMode.active():
            logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                inference_context=inference_context,
                runtime_gather_output=True,
            )

        # Verify the output shape.
        assert logits.shape[0] == 1
        assert logits.shape[1] == padded_token_count
        assert logits.shape[2] == self.model.module.vocab_size

        # Extract the logits corresponding to the padding tokens (from index 10 to 63).
        padding_start_idx = inference_context.active_token_count
        padding_end_idx = inference_context.padded_active_token_count
        padding_logits = logits[0, padding_start_idx:padding_end_idx, :]

        # Assert that all padding logits are zero.
        assert torch.all(padding_logits == 0.0), "Logits for padding tokens are not all zero."


def _make_yarn_config(**kwargs):
    """Build a TransformerConfig with yarn positional embedding attributes."""
    cfg = TransformerConfig(
        num_layers=3,  # 1 Mamba layer, 1 attention layer, 1 MLP layer
        hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=True,
        **kwargs,
    )
    # Yarn-specific attributes are set dynamically on the config (not TransformerConfig fields).
    cfg.yarn_rotary_scaling_factor = 2.0
    cfg.yarn_original_max_position_embeddings = 4
    cfg.yarn_beta_fast = 32.0
    cfg.yarn_beta_slow = 1.0
    cfg.yarn_mscale = 1.0
    cfg.yarn_mscale_all_dim = 0.0
    cfg.yarn_correction_range_round_to_int = True
    return cfg


class TestHybridModelWithYarn:
    """Tests for HybridModel with YaRN positional embeddings."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        model_config = _make_yarn_config()
        self.model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",  # 1 Mamba, 1 attention, 1 MLP
            position_embedding_type='yarn',
            rotary_base=10000,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.model, HybridModel)
        assert self.model.max_sequence_length == 4
        assert self.model.position_embedding_type == 'yarn'
        # YaRN creates a YarnRotaryEmbedding rather than a plain RotaryEmbedding.
        assert isinstance(self.model.rotary_pos_emb, YarnRotaryEmbedding)

    def test_forward(self):
        sequence_length = self.model.max_sequence_length
        micro_batch_size = 2

        self.model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = self.model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.model.vocab_size

    def test_inference(self):
        micro_batch_size = 2
        inference_context: BaseInferenceContext = StaticInferenceContext(
            max_batch_size=micro_batch_size, max_sequence_length=self.model.max_sequence_length
        )
        prompt_length = self.model.max_sequence_length - 1

        self.model.cuda()

        with InferenceMode.active():
            # load-context/first-output-token, step/generate
            for offset in (0, prompt_length):
                sequence_length = prompt_length if offset == 0 else 1
                inference_context.sequence_len_offset = offset

                data = list(range(sequence_length))
                input_ids = (
                    torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
                )
                position_ids = (
                    torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
                )
                attention_mask = torch.ones(
                    (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
                ).cuda()

                logits = self.model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    inference_context=inference_context,
                    runtime_gather_output=True,
                )

                assert logits.shape[0] == micro_batch_size
                # StaticInferenceContext always sets materialize_only_last_token_logits=True.
                assert logits.shape[1] == 1
                assert logits.shape[2] == self.model.vocab_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Production Hybrid model requires CUDA")
@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is not installed")
class TestHybridDSv41Model:
    """Exercise the V4.1 text backbone with real attention, mHC, MoE, and LM endpoints."""

    @pytest.fixture
    def pg_collection(self, monkeypatch):
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(1234)
        monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", {})
        monkeypatch.setattr(
            DSAIndexerLossAutoScaler, "main_loss_backward_scale", torch.ones((), device="cuda")
        )
        try:
            yield ProcessGroupCollection.use_mpu_process_groups()
        finally:
            Utils.destroy_model_parallel()

    @staticmethod
    def _build_model(pg_collection, dtype, *, indexer_loss_coeff=0.0, sparse_loss=False):
        # Hybrid numbers attention and MoE sublayers independently. Zero entries on
        # the E sublayers are placeholders, not additional attention modules.
        config = _make_dsv41_config(
            params_dtype=dtype,
            num_layers=12,
            csa_compress_ratios=[0, 0, 2, 0, 2, 0, 1, 0, 1, 0, 1, 0],
            csa2_kv_source_layers=[2, 6],
            csa2_index_source_layers=[2, 6, 8],
            csa2_candidate_source_layer=6,
            dsa_indexer_loss_coeff=indexer_loss_coeff,
            dsa_indexer_use_sparse_loss=sparse_loss,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0,
            moe_shared_expert_gate=False,
            moe_shared_expert_overlap=False,
            moe_token_dispatcher_type="allgather",
            moe_grouped_gemm=False,
            cross_entropy_loss_fusion=False,
        )
        model_config = HybridModelConfig(
            transformer=config,
            hybrid_stack_spec=hybrid_dsv4_stack_spec(config),
            hybrid_layer_pattern="DE" * 6,
            vocab_size=64,
            seq_length=16,
            position_embedding_type="none",
            share_embeddings_and_output_weights=False,
            parallel_output=False,
        )
        bare_model = HybridModelBuilder(model_config).build_model(pg_collection).cuda()
        # Use the training wrapper, which preserves the FP32-marked compressor,
        # attention sink, and mHC tensors. A blanket .bfloat16() would change them.
        model = Float16Module(config, bare_model) if dtype == torch.bfloat16 else bare_model
        model.train()
        return model, bare_model

    @staticmethod
    def _batch(length):
        tokens = torch.arange(2 * (length + 1), device="cuda").view(2, length + 1) % 64
        input_ids = tokens[:, :-1].contiguous()
        labels = tokens[:, 1:].contiguous()
        position_ids = torch.arange(length, device="cuda").expand_as(input_ids)
        loss_mask = torch.ones_like(labels, dtype=torch.float32)
        loss_mask[:, 1::3] = 0
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
        )

    @staticmethod
    def _masked_loss(token_losses, loss_mask):
        # HybridModel returns per-token CE. The training loss function owns masking
        # and reduction; the model's loss_mask argument also serves the MTP path.
        return (token_losses.float() * loss_mask).sum() / loss_mask.sum()

    @staticmethod
    def _assert_live_gradient(parameter, name):
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.float().abs().sum() > 0, name

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_hybrid_dsv41_logits_masked_ce_and_backbone_gradients(self, pg_collection, dtype):
        """The configured DE stack reaches both LM endpoints and every trained branch."""
        model, bare_model = self._build_model(pg_collection, dtype)
        assert bare_model.pg_collection is pg_collection
        assert bare_model.embedding.word_embeddings.weight is not bare_model.output_layer.weight
        assert bare_model.embedding.word_embeddings.weight.dtype == dtype
        assert bare_model.output_layer.weight.dtype == dtype
        assert len(bare_model.decoder.layers) == 12
        assert all(
            isinstance(layer, HyperConnectionHybridLayer) for layer in bare_model.decoder.layers
        )
        assert not any("hc_head_" in name for name, _ in bare_model.named_parameters())

        marked_parameters = [
            parameter
            for parameter in bare_model.parameters()
            if getattr(parameter, "keep_in_fp32", False)
        ]
        assert marked_parameters
        assert all(parameter.dtype == torch.float32 for parameter in marked_parameters)

        attention_layers = [
            layer.inner_layer.self_attention for layer in bare_model.decoder.layers[::2]
        ]
        moe_layers = [layer.inner_layer.mlp for layer in bare_model.decoder.layers[1::2]]
        assert all(
            isinstance(layer.core_attention, CompressedSparseAttention2)
            for layer in attention_layers
        )
        assert all(isinstance(layer, MoELayer) for layer in moe_layers)
        assert all(isinstance(layer.experts, SequentialMLP) for layer in moe_layers)
        assert all(isinstance(layer.shared_experts, SharedExpertMLP) for layer in moe_layers)

        batch = self._batch(7)
        with torch.no_grad():
            logits = model(**{key: value for key, value in batch.items() if key != "labels"})
        assert logits.shape == (2, 7, 64)
        assert logits.dtype == torch.float32
        expected_losses = F.cross_entropy(
            logits.flatten(0, 1), batch["labels"].flatten(), reduction="none"
        ).view_as(batch["labels"])

        selected_experts = {}

        def record_routing(router, inputs, output):
            # Router flattens [sequence, batch]. Experts used only by masked final
            # tokens can correctly receive zero gradient, so require supervised use.
            supervised_tokens = batch["loss_mask"].T.reshape(-1).bool()
            selected_experts[router] = output[1].detach()[supervised_tokens].any(dim=0)

        handles = [layer.router.register_forward_hook(record_routing) for layer in moe_layers]
        try:
            token_losses = model(**batch)
        finally:
            for handle in handles:
                handle.remove()
        assert token_losses.shape == batch["labels"].shape
        tolerance = 2e-4 if dtype == torch.bfloat16 else 2e-6
        torch.testing.assert_close(token_losses, expected_losses, atol=tolerance, rtol=tolerance)
        token_losses.retain_grad()
        loss = self._masked_loss(token_losses, batch["loss_mask"])
        torch.testing.assert_close(
            loss, expected_losses[batch["loss_mask"].bool()].mean(), atol=tolerance, rtol=tolerance
        )
        loss.backward()
        torch.testing.assert_close(
            token_losses.grad, batch["loss_mask"] / batch["loss_mask"].sum(), atol=0, rtol=0
        )

        self._assert_live_gradient(bare_model.embedding.word_embeddings.weight, "embedding")
        self._assert_live_gradient(bare_model.output_layer.weight, "LM head")
        self._assert_live_gradient(bare_model.decoder.final_norm.weight, "final norm")
        for index, layer in enumerate(bare_model.decoder.layers):
            self._assert_live_gradient(layer.hyper_connection.mapping_proj.weight, f"mHC {index}")
        for index, attention in enumerate(attention_layers):
            # Checking the main query projection reaches all attention modes without
            # requiring an auxiliary indexer loss in this backbone integration step.
            self._assert_live_gradient(attention.linear_q_up_proj.weight, f"attention Q {index}")
            indexer = attention.core_attention.indexer
            if indexer is not None:
                assert all(parameter.grad is None for parameter in indexer.parameters())
            compressor = attention.core_attention.compressor
            if compressor is not None:
                self._assert_live_gradient(compressor.linear_wkv.weight, f"global KV owner {index}")
                if compressor.linear_wgate is not None:
                    self._assert_live_gradient(
                        compressor.linear_wgate.weight, f"KV compressor gate {index}"
                    )
        for index, moe in enumerate(moe_layers):
            self._assert_live_gradient(moe.router.weight, f"router {index}")
            self._assert_live_gradient(moe.shared_experts.linear_fc1.weight, f"shared FC1 {index}")
            self._assert_live_gradient(moe.shared_experts.linear_fc2.weight, f"shared FC2 {index}")
            selected = selected_experts[moe.router].nonzero().flatten().tolist()
            assert selected
            for expert_index in selected:
                expert = moe.experts.local_experts[expert_index]
                self._assert_live_gradient(
                    expert.linear_fc1.weight, f"expert FC1 {index}/{expert_index}"
                )
                self._assert_live_gradient(
                    expert.linear_fc2.weight, f"expert FC2 {index}/{expert_index}"
                )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("indexer_loss_coeff", [0.0, 0.1], ids=["lm_only", "lm_and_indexer"])
    def test_hybrid_dsv41_live_forwards_and_state_dict_roundtrip(
        self, pg_collection, dtype, indexer_loss_coeff
    ):
        """Restored serial execution matches two live graphs backpropagated in reverse."""
        model, bare_model = self._build_model(
            pg_collection, dtype, indexer_loss_coeff=indexer_loss_coeff
        )
        restored, restored_bare = self._build_model(
            pg_collection, dtype, indexer_loss_coeff=indexer_loss_coeff
        )
        incompatible = restored_bare.load_state_dict(deepcopy(bare_model.state_dict()), strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        batches = [self._batch(5), self._batch(7)]

        # Both CSA2 sharing and Single-Pass mHC must keep the earlier graph intact.
        outputs = [model(**batch) for batch in batches]
        forward_tolerance = 2e-4 if dtype == torch.bfloat16 else 2e-6
        for index in (1, 0):
            expected = restored(**batches[index])
            torch.testing.assert_close(
                outputs[index], expected, atol=forward_tolerance, rtol=forward_tolerance
            )
            self._masked_loss(expected, batches[index]["loss_mask"]).backward()
            self._masked_loss(outputs[index], batches[index]["loss_mask"]).backward()

        restored_parameters = dict(restored_bare.named_parameters())
        tolerance = dict(atol=2e-6, rtol=2e-5)
        if dtype == torch.bfloat16:
            tolerance = dict(atol=2e-5, rtol=2e-2)
        for name, parameter in bare_model.named_parameters():
            expected_gradient = restored_parameters[name].grad
            if expected_gradient is None:
                assert parameter.grad is None, name
            else:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                torch.testing.assert_close(parameter.grad, expected_gradient, **tolerance, msg=name)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("sparse_loss", [False, True], ids=["dense", "sparse"])
    def test_hybrid_dsv41_indexer_aux_loss_preserves_backbone_gradients(
        self, pg_collection, dtype, sparse_loss
    ):
        """One LM backward also trains the shared indexer, without updating its input producers."""
        lm_only, lm_bare = self._build_model(pg_collection, dtype, sparse_loss=sparse_loss)
        model, bare_model = self._build_model(
            pg_collection, dtype, indexer_loss_coeff=0.1, sparse_loss=sparse_loss
        )
        bare_model.load_state_dict(deepcopy(lm_bare.state_dict()), strict=True)
        batch = self._batch(7)
        forward_tolerance = 2e-4 if dtype == torch.bfloat16 else 2e-6

        with torch.no_grad():
            inputs = {key: value for key, value in batch.items() if key != "labels"}
            expected_logits = lm_only(**inputs)
            logits = model(**inputs)
        torch.testing.assert_close(
            logits, expected_logits, atol=forward_tolerance, rtol=forward_tolerance
        )
        expected_losses = lm_only(**batch)
        token_losses = model(**batch)
        torch.testing.assert_close(
            token_losses, expected_losses, atol=forward_tolerance, rtol=forward_tolerance
        )
        # The model exposes the unchanged LM loss. The real CSA2 KL terms are
        # attached by DSAIndexerLossAutoScaler and participate in this same backward.
        self._masked_loss(expected_losses, batch["loss_mask"]).backward()
        self._masked_loss(token_losses, batch["loss_mask"]).backward()

        index_source_layers = bare_model.config.csa2_index_source_layers
        tracker_values = DSAIndexerLossLoggingHelper.tracker["values"]
        assert torch.isfinite(tracker_values).all()
        assert (tracker_values[index_source_layers] > 0).all()
        inactive = torch.ones_like(tracker_values, dtype=torch.bool)
        inactive[index_source_layers] = False
        assert torch.count_nonzero(tracker_values[inactive]) == 0

        for layer_index in index_source_layers:
            indexer = bare_model.decoder.layers[
                layer_index
            ].inner_layer.self_attention.core_attention.indexer
            self._assert_live_gradient(indexer.linear_wq_b.weight, f"indexer query {layer_index}")
            self._assert_live_gradient(
                indexer.linear_weights_proj.weight, f"indexer weights {layer_index}"
            )
            if indexer.owns_k:
                self._assert_live_gradient(
                    indexer.linear_wk.weight, f"shared indexer key {layer_index}"
                )
                self._assert_live_gradient(
                    indexer.k_norm.weight, f"shared indexer key norm {layer_index}"
                )
            else:
                assert indexer.linear_wk is None and indexer.k_norm is None

        reference_parameters = dict(lm_bare.named_parameters())
        tolerance = dict(atol=2e-6, rtol=2e-5)
        if dtype == torch.bfloat16:
            tolerance = dict(atol=2e-5, rtol=2e-2)
        for name, parameter in bare_model.named_parameters():
            reference_gradient = reference_parameters[name].grad
            if ".indexer." in name:
                assert reference_gradient is None, name
            elif reference_gradient is None:
                assert parameter.grad is None, name
            else:
                assert parameter.grad is not None, name
                torch.testing.assert_close(
                    parameter.grad, reference_gradient, **tolerance, msg=name
                )
