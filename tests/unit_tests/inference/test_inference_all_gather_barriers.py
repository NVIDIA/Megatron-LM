# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""All-gather barriers on the inference "tp" symmetric buffer.

Inference-optimized column-parallel linears all-gather their sequence-parallel input into one
shared "tp" symmetric buffer, and their GEMMs read it there. Each all-gather barriers before it
writes unless the module that knows the order of operations on the buffer proves the barrier
redundant (chain_tp_buffer_collectives in megatron.core.inference.symmetric_memory): the
all-gather directly follows a reduce-scatter with the same per-rank segments, or a collective
that synchronizes the TP group. The declarations are static, so CUDA graph capture and replay
agree.

The GPU tests run an eval forward of each inference layer type at TP=2 with sequence
parallelism and NVLS, record the operations on the buffer and the synchronizing collectives,
check that none writes a buffer a peer may still read, and pin which all-gathers barrier. The
CPU tests check the declaration rules and that every inference layer spec has a case.
"""

import ast
import dataclasses
import pathlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

import megatron.core.inference.symmetric_memory as symmetric_memory
import megatron.core.tensor_parallel.inference_layers as inference_layers
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import (
    gated_delta_product_inference_stack_spec,
    hybrid_inference_stack_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA, HAVE_FLA_GDN2, HAVE_FLA_KDA
from megatron.core.ssm.gated_delta_product import HAVE_FLA as HAVE_GDP_FLA
from megatron.core.ssm.gated_delta_product import HAVE_MAMBA_SSM as HAVE_GDP_MAMBA_SSM
from megatron.core.ssm.kda_layer_config import KDALayerConfig
from megatron.core.ssm.mamba_mixer import HAVE_MAMBA_SSM
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import ModuleSpec, TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.inference.symmetric_buffer_test_utils import (
    ALL_GATHER,
    REDUCE_SCATTER,
    barrier_trace,
    find_symmetric_buffer_races,
    record_symmetric_buffer_ops,
    skip_unless_nvls,
)
from tests.unit_tests.test_utilities import Utils

AG = (ALL_GATHER, False)
BARRIERED_AG = (ALL_GATHER, True)
RS = (REDUCE_SCATTER, None)

# A sublayer that all-gathers its input after the previous sublayer's reduce-scatter. The first
# one of a block follows an operation the block cannot know, so it keeps its barrier.
_IN_RS = [AG, RS]
_FIRST_IN_RS = [BARRIERED_AG, RS]
# MLA: the query projection follows the NCCL gather of the RoPE key or latent, which
# synchronizes the TP group; linear_kv_up_proj follows the query projection's all-gather.
_MLA = [AG, BARRIERED_AG, RS]

_MLA_CONFIG = dict(
    q_lora_rank=64,
    kv_lora_rank=64,
    qk_head_dim=64,
    qk_pos_emb_head_dim=64,
    v_head_dim=128,
    rope_type="rope",
    # Resolves the hybrid specs' up-projections to inference-optimized linears.
    qk_layernorm=True,
)
_MOE_CONFIG = dict(
    num_moe_experts=4,
    moe_router_topk=2,
    moe_ffn_hidden_size=128,
    moe_shared_expert_intermediate_size=256,
    moe_grouped_gemm=True,
)
_GDN_CONFIG = dict(
    linear_conv_kernel_dim=4,
    linear_key_head_dim=64,
    linear_value_head_dim=64,
    linear_num_key_heads=4,
    linear_num_value_heads=4,
    activation_func=F.silu,
)
_KDA_CONFIG = dict(
    linear_conv_kernel_dim=4,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_num_key_heads=4,
    linear_num_value_heads=4,
    kda_safe_gate=True,
    kda_lower_bound=-5.0,
    experimental_attention_variant="kda",
    is_hybrid_model=True,
    activation_func=F.silu,
)


@dataclasses.dataclass(frozen=True)
class _GPTCase:
    """A two-layer GPT TransformerBlock of inference layers."""

    trace: list
    config_class: type = TransformerConfig
    config_overrides: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class _HybridCase:
    """A HybridStack of inference layers, and the stack-spec layers it builds."""

    stack_spec: ModuleSpec
    pattern: str
    layer_specs: tuple[str, ...]
    trace: list
    config_class: type = TransformerConfig
    config_overrides: dict = dataclasses.field(default_factory=dict)


_GPT_CASES = [
    pytest.param(_GPTCase(_FIRST_IN_RS + _IN_RS * 3), id="dense"),
    # The shared expert's linear_fc1 follows linear_proj's reduce-scatter.
    pytest.param(_GPTCase(_FIRST_IN_RS + _IN_RS * 3, config_overrides=_MOE_CONFIG), id="moe"),
    pytest.param(
        _GPTCase((_MLA + _IN_RS) * 2, MLATransformerConfig, config_overrides=_MLA_CONFIG), id="mla"
    ),
    pytest.param(
        _GPTCase(
            (_MLA + _IN_RS) * 2,
            MLATransformerConfig,
            config_overrides={**_MLA_CONFIG, "q_lora_rank": None, "qk_layernorm": False},
        ),
        id="mla_q_proj",
    ),
]

_HYBRID_CASES = [
    pytest.param(
        _HybridCase(
            hybrid_inference_stack_spec,
            "*-*-",
            ("attention_layer", "mlp_layer"),
            _FIRST_IN_RS + _IN_RS * 3,
        ),
        id="attention_mlp",
    ),
    pytest.param(
        _HybridCase(
            hybrid_inference_stack_spec,
            "*E",
            ("moe_layer",),
            _FIRST_IN_RS + _IN_RS,
            config_overrides=_MOE_CONFIG,
        ),
        id="moe",
    ),
    pytest.param(
        _HybridCase(hybrid_inference_stack_spec, "MM", ("mamba_layer",), _FIRST_IN_RS + _IN_RS),
        marks=pytest.mark.skipif(not HAVE_MAMBA_SSM, reason="mamba_ssm is not installed"),
        id="mamba",
    ),
    pytest.param(
        _HybridCase(
            gated_delta_product_inference_stack_spec,
            "MM",
            ("mamba_layer",),
            _FIRST_IN_RS + _IN_RS,
            config_overrides=dict(
                mamba_num_heads=4, mamba_head_dim=64, mamba_num_groups=4, mamba_state_dim=16
            ),
        ),
        marks=pytest.mark.skipif(
            not (HAVE_GDP_FLA and HAVE_GDP_MAMBA_SSM), reason="GDP dependencies are not installed"
        ),
        id="gdp",
    ),
    pytest.param(
        _HybridCase(
            hybrid_inference_stack_spec,
            "GG",
            ("gdn_layer",),
            _FIRST_IN_RS + _IN_RS,
            config_overrides=_GDN_CONFIG,
        ),
        marks=pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed"),
        id="gdn",
    ),
    pytest.param(
        _HybridCase(
            hybrid_inference_stack_spec,
            "GG",
            ("gdn2_layer",),
            _FIRST_IN_RS + _IN_RS,
            config_overrides={**_GDN_CONFIG, "experimental_attention_variant": "gdn2"},
        ),
        marks=pytest.mark.skipif(not HAVE_FLA_GDN2, reason="FLA with GDN2 is not installed"),
        id="gdn2",
    ),
    # beta_proj shares in_proj's all-gather.
    pytest.param(
        _HybridCase(
            hybrid_inference_stack_spec,
            "KK",
            ("kda_layer",),
            _FIRST_IN_RS + _IN_RS,
            config_class=KDALayerConfig,
            config_overrides=_KDA_CONFIG,
        ),
        marks=pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA is not installed"),
        id="kda",
    ),
    # Two-stage gates add f_b_proj's and g_b_proj's all-gathers, each after another all-gather.
    pytest.param(
        _HybridCase(
            hybrid_inference_stack_spec,
            "KK",
            ("kda_layer",),
            [BARRIERED_AG] * 3 + [RS, AG] + [BARRIERED_AG] * 2 + [RS],
            config_class=KDALayerConfig,
            config_overrides={**_KDA_CONFIG, "kda_two_stage_gates": True},
        ),
        marks=pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA is not installed"),
        id="kda_two_stage",
    ),
    pytest.param(
        _HybridCase(
            hybrid_inference_stack_spec,
            "++",
            ("mla_layer",),
            _MLA * 2,
            config_class=MLATransformerConfig,
            config_overrides=_MLA_CONFIG,
        ),
        id="mla",
    ),
    pytest.param(
        _HybridCase(
            hybrid_inference_stack_spec,
            "++",
            ("mla_fused_down_proj_layer",),
            _MLA * 2,
            config_class=MLATransformerConfig,
            config_overrides={**_MLA_CONFIG, "mla_down_proj_fusion": True},
        ),
        id="mla_fused_down_proj",
    ),
]

# Inference hybrid layer specs that a test other than test_hybrid_stack checks.
_OTHER_TESTS = {
    "dsa_layer": (
        "tests/unit_tests/inference/test_inference_all_gather_barriers.py::"
        "TestInferenceAllGatherBarriers::test_hybrid_dsa_layer"
    ),
    "mtp_block_spec": (
        "tests/unit_tests/inference/text_generation_controllers/"
        "test_text_generation_controller.py::TestTextGenerationController::"
        "test_mtp_inference_optimized_hybrid_no_unbarriered_consecutive_all_gathers"
    ),
}


@pytest.mark.skipif(
    not inference_layers.HAVE_TE, reason="inference_optimized requires Transformer Engine"
)
@pytest.mark.skipif(Utils.world_size < 2, reason="requires tensor parallelism of 2")
class TestInferenceAllGatherBarriers:
    """Forward each inference layer type at TP=2 with sequence parallelism and NVLS."""

    tensor_parallel_size = 2
    sequence_length = 64
    micro_batch_size = 2
    hidden_size = 256

    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(self.tensor_parallel_size, 1)
        SymmetricMemoryManager.destroy()
        model_parallel_cuda_manual_seed(123)
        yield
        torch.cuda.synchronize()
        SymmetricMemoryManager.destroy()  # its buffer is registered on this TP group
        Utils.destroy_model_parallel()

    def _config(self, config_class=TransformerConfig, **overrides):
        return config_class(
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            normalization="RMSNorm",
            add_bias_linear=False,
            transformer_impl="inference_optimized",
            bf16=True,
            params_dtype=torch.bfloat16,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
            tensor_model_parallel_size=self.tensor_parallel_size,
            sequence_parallel=True,
            **overrides,
        )

    @staticmethod
    def _hybrid_stack(stack_spec, pattern, config):
        return HybridStack(
            config,
            stack_spec.submodules,
            layer_config_list=validate_segment_layers(pattern, config),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "pp", "cp"]
            ),
        )

    def _check(self, module, expected_trace):
        """Forward ``module`` and check the operations it ran on the symmetric buffer."""
        module = module.cuda().eval()
        generator = torch.Generator(device="cuda").manual_seed(1234)
        hidden_states = torch.randn(
            self.sequence_length // self.tensor_parallel_size,
            self.micro_batch_size,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        with record_symmetric_buffer_ops() as ops, torch.no_grad():
            # Its NCCL all-reduce also synchronizes the TP group before the forward.
            skip_unless_nvls(module.tp_group)
            module(hidden_states, attention_mask=None)
        races = find_symmetric_buffer_races(ops)
        assert not races, f"{races}\nrecorded: {ops}"
        assert barrier_trace(ops) == expected_trace

    @pytest.mark.parametrize("case", _GPT_CASES)
    def test_gpt_block(self, case):
        config = self._config(case.config_class, num_layers=2, **case.config_overrides)
        spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False)
        self._check(TransformerBlock(config, spec), case.trace)

    @pytest.mark.parametrize("case", _HYBRID_CASES)
    def test_hybrid_stack(self, case):
        config = self._config(
            case.config_class, num_layers=len(case.pattern), **case.config_overrides
        )
        self._check(self._hybrid_stack(case.stack_spec, case.pattern, config), case.trace)

    def test_hybrid_dsa_layer(self):
        """DSA runs no all-gather on the buffer; the next layer's follows its reduce-scatter.

        The hybrid inference spec keeps DSA's projections in Transformer Engine, which
        communicates through NCCL, so linear_proj's reduce-scatter is DSA's only operation on
        the buffer. Checked on the built layers: DSA's core attention needs kernels this test
        should not depend on.
        """
        config = self._config(
            MLATransformerConfig,
            num_layers=2,
            **{
                **_MLA_CONFIG,
                "qk_layernorm": False,
                "dsa_indexer_n_heads": 8,
                "dsa_indexer_head_dim": 64,
                "dsa_indexer_topk": 32,
            },
        )
        dsa_layer, mlp_layer = self._hybrid_stack(hybrid_inference_stack_spec, "D-", config).layers
        all_gathering = (
            inference_layers.InferenceColumnParallelLinear,
            inference_layers.InferenceLayerNormColumnParallelLinear,
        )
        assert not any(isinstance(module, all_gathering) for module in dsa_layer.modules())
        linear_proj = dsa_layer.self_attention.linear_proj
        assert isinstance(linear_proj, inference_layers.InferenceRowParallelLinear)
        assert symmetric_memory.chain_tp_buffer_collectives(dsa_layer, None) is linear_proj
        assert mlp_layer.mlp.linear_fc1.barrier_before_all_gather is False


def test_every_inference_hybrid_layer_spec_has_a_case():
    """A new inference layer type needs a case, so its all-gathers get checked."""
    covered = set()
    for param in _HYBRID_CASES:
        (case,) = param.values
        covered.update(id(getattr(case.stack_spec.submodules, name)) for name in case.layer_specs)
    for stack_spec in (hybrid_inference_stack_spec, gated_delta_product_inference_stack_spec):
        for field in dataclasses.fields(stack_spec.submodules):
            spec = getattr(stack_spec.submodules, field.name)
            if spec is None or spec is IdentityOp or field.name in _OTHER_TESTS:
                continue
            assert id(spec) in covered, f"no all-gather barrier case for {field.name}"


@pytest.mark.parametrize("node_id", sorted(_OTHER_TESTS.values()))
def test_other_tests_exist(node_id):
    """Each test that _OTHER_TESTS names exists."""
    path, class_name, function_name = node_id.split("::")
    tree = ast.parse((pathlib.Path(__file__).parents[3] / path).read_text())
    functions = {
        item.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
        for item in node.body
        if isinstance(item, ast.FunctionDef)
    }
    assert function_name in functions, f"{node_id} does not exist"


class _StandIn:
    """An inference linear reduced to the attributes the declaration rules read."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)


class _ColumnParallel(_StandIn):
    pass


class _LayerNormColumnParallel(_StandIn):
    pass


class _RowParallel(_StandIn):
    pass


class TestDeclarationRules:
    """The declaration rules of megatron.core.inference.symmetric_memory, on stand-ins (CPU)."""

    tp_group = object()

    @pytest.fixture(autouse=True)
    def stand_in_linears(self, monkeypatch):
        monkeypatch.setattr(inference_layers, "HAVE_TE", True)
        monkeypatch.setattr(inference_layers, "InferenceColumnParallelLinear", _ColumnParallel)
        monkeypatch.setattr(
            inference_layers, "InferenceLayerNormColumnParallelLinear", _LayerNormColumnParallel
        )
        monkeypatch.setattr(inference_layers, "InferenceRowParallelLinear", _RowParallel)

    def _all_gather(self, cls=_LayerNormColumnParallel, in_features=256, fp32_residual=False):
        return cls(
            tp_group=self.tp_group,
            in_features=in_features,
            config=SimpleNamespace(fp32_residual_connection=fp32_residual),
            barrier_before_all_gather=True,
        )

    def _reduce_scatter(self, out_features=256, tp_group=None):
        return _RowParallel(tp_group=tp_group or self.tp_group, out_features=out_features)

    @pytest.mark.parametrize(
        "all_gather_kwargs, reduce_scatter_kwargs, barrier",
        [
            ({}, {}, False),
            ({"cls": _ColumnParallel}, {}, False),
            # A different width writes into the segments that peers may still reduce.
            ({"in_features": 128}, {}, True),
            ({}, {"tp_group": object()}, True),
            # An FP32 residual stream doubles the bytes each rank all-gathers.
            ({"fp32_residual": True}, {}, True),
        ],
    )
    def test_after_reduce_scatter(self, all_gather_kwargs, reduce_scatter_kwargs, barrier):
        all_gather = self._all_gather(**all_gather_kwargs)
        symmetric_memory.skip_all_gather_barrier_after_reduce_scatter(
            self._reduce_scatter(**reduce_scatter_kwargs), all_gather
        )
        assert all_gather.barrier_before_all_gather is barrier

    def test_after_unknown_or_other_reduce_scatter(self):
        all_gather = self._all_gather()
        symmetric_memory.skip_all_gather_barrier_after_reduce_scatter(None, all_gather)
        symmetric_memory.skip_all_gather_barrier_after_reduce_scatter(
            _StandIn(out_features=256), all_gather
        )
        assert all_gather.barrier_before_all_gather is True
        other = _StandIn(barrier_before_all_gather=True)
        symmetric_memory.skip_all_gather_barrier_after_reduce_scatter(self._reduce_scatter(), other)
        assert other.barrier_before_all_gather is True

    def test_after_synchronizing_collective(self):
        all_gather = self._all_gather(_ColumnParallel)
        symmetric_memory.skip_all_gather_barrier_after_synchronizing_collective(all_gather)
        assert all_gather.barrier_before_all_gather is False
        other = _StandIn(barrier_before_all_gather=True)
        symmetric_memory.skip_all_gather_barrier_after_synchronizing_collective(other)
        assert other.barrier_before_all_gather is True

    def test_chain(self):
        reduce_scatter = self._reduce_scatter()
        # Modules that do nothing on the buffer pass the preceding reduce-scatter on.
        assert symmetric_memory.chain_tp_buffer_collectives(None, reduce_scatter) is reduce_scatter
        assert (
            symmetric_memory.chain_tp_buffer_collectives(IdentityOp(), reduce_scatter)
            is reduce_scatter
        )
        # A module that does not declare its order may do anything there.
        assert (
            symmetric_memory.chain_tp_buffer_collectives(torch.nn.Linear(2, 2), reduce_scatter)
            is None
        )
        declaring = mock.Mock(spec=["chain_tp_buffer_collectives"])
        declaring.chain_tp_buffer_collectives.return_value = "last"
        assert symmetric_memory.chain_tp_buffer_collectives(declaring, reduce_scatter) == "last"
        declaring.chain_tp_buffer_collectives.assert_called_once_with(reduce_scatter)
