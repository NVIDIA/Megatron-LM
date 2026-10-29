# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""All-gather barriers of the inference-optimized layer specs.

Inference-optimized column-parallel linears all-gather their sequence-parallel input into one
shared "tp" symmetric buffer, and their GEMMs read it from there. multimem_all_gather ends with
a barrier but does not start with one; a reduce-scatter barriers on entry. So an all-gather that
directly follows another all-gather on the buffer must request barrier_before, or a fast rank
overwrites a peer's buffer while the peer's GEMM still reads it. Layers request it statically
(set_barrier_before_all_gather), so CUDA graph capture and replay agree.

These tests record the symmetric-memory collectives of a forward of each inference-spec layer
type at TP=2 with sequence parallelism and NVLS, and check that rule.
"""

import dataclasses
from typing import Optional
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

import megatron.core.tensor_parallel.inference_layers as inference_layers
from megatron.core.inference.communication.torch_symm_triton.utils import is_device_nvls_capable
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
from tests.unit_tests.test_utilities import Utils

_ALL_GATHER = ("all_gather", False)
_BARRIERED_ALL_GATHER = ("all_gather", True)
_REDUCE_SCATTER = ("reduce_scatter", None)


@dataclasses.dataclass(frozen=True)
class _HybridCase:
    """A HybridStack of inference layers to run, and the stack-spec layers it builds."""

    stack_spec: ModuleSpec
    pattern: str
    layer_specs: tuple[str, ...]
    config_class: type = TransformerConfig
    config_overrides: dict = dataclasses.field(default_factory=dict)
    # Expected collectives of one layer, when pinned exactly.
    layer_trace: Optional[list] = None


_GDN = dict(
    linear_conv_kernel_dim=4,
    linear_key_head_dim=64,
    linear_value_head_dim=64,
    linear_num_key_heads=4,
    linear_num_value_heads=4,
    activation_func=F.silu,
)
_KDA = dict(
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

_HYBRID_CASES = [
    pytest.param(
        _HybridCase(hybrid_inference_stack_spec, "*-*-", ("attention_layer", "mlp_layer")),
        id="attention_mlp",
    ),
    pytest.param(
        _HybridCase(hybrid_inference_stack_spec, "MM", ("mamba_layer",)),
        marks=pytest.mark.skipif(not HAVE_MAMBA_SSM, reason="mamba_ssm is not installed"),
        id="mamba",
    ),
    pytest.param(
        _HybridCase(
            gated_delta_product_inference_stack_spec,
            "MM",
            ("mamba_layer",),
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
        _HybridCase(hybrid_inference_stack_spec, "GG", ("gdn_layer",), config_overrides=_GDN),
        marks=pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed"),
        id="gdn",
    ),
    pytest.param(
        _HybridCase(
            hybrid_inference_stack_spec,
            "GG",
            ("gdn2_layer",),
            config_overrides={**_GDN, "experimental_attention_variant": "gdn2"},
        ),
        marks=pytest.mark.skipif(not HAVE_FLA_GDN2, reason="FLA with GDN2 is not installed"),
        id="gdn2",
    ),
    # in_proj follows the previous layer's reduce-scatter; beta_proj follows in_proj.
    pytest.param(
        _HybridCase(
            hybrid_inference_stack_spec,
            "KK",
            ("kda_layer",),
            config_class=KDALayerConfig,
            config_overrides=_KDA,
            layer_trace=[_ALL_GATHER, _BARRIERED_ALL_GATHER, _REDUCE_SCATTER],
        ),
        marks=pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA is not installed"),
        id="kda",
    ),
    # Two-stage gates add f_b_proj and g_b_proj all-gathers right after beta_proj's.
    pytest.param(
        _HybridCase(
            hybrid_inference_stack_spec,
            "KK",
            ("kda_layer",),
            config_class=KDALayerConfig,
            config_overrides={**_KDA, "kda_two_stage_gates": True},
            layer_trace=[_ALL_GATHER] + [_BARRIERED_ALL_GATHER] * 3 + [_REDUCE_SCATTER],
        ),
        marks=pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA is not installed"),
        id="kda_two_stage",
    ),
]

# Inference HybridStack layer specs whose all-gather barriers are tested elsewhere.
_TESTED_ELSEWHERE = {
    # MLASelfAttention / FusedMLASelfAttention: tests/unit_tests/transformer/
    # test_multi_latent_attention.py.
    "dsa_layer",
    "mla_layer",
    "mla_fused_down_proj_layer",
    # Routed tokens use the MoE dispatcher's own symmetric buffers; the shared expert is an
    # MLP (the attention_mlp case).
    "moe_layer",
    # MultiTokenPredictionLayer: test_text_generation_controller.py,
    # test_mtp_inference_optimized_hybrid_no_unbarriered_consecutive_all_gathers.
    "mtp_block_spec",
}


def _unbarriered_all_gathers(events):
    """Indices of all-gathers that directly follow another all-gather without barrier_before."""
    return [
        index
        for index, ((previous, _), (kind, barrier_before)) in enumerate(
            zip(events, events[1:]), start=1
        )
        if previous == kind == "all_gather" and not barrier_before
    ]


@pytest.mark.skipif(
    not inference_layers.HAVE_TE, reason="inference_optimized requires Transformer Engine"
)
@pytest.mark.skipif(Utils.world_size < 2, reason="requires tensor parallelism of 2")
class TestInferenceAllGatherBarriers:
    """Forward each inference-spec layer type at TP=2 with sequence parallelism and NVLS."""

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

    def _symmetric_collectives(self, module):
        """Run ``module`` and return its symmetric-memory collectives in order: all-gathers
        with their barrier_before flag, and reduce-scatters."""
        # Collective: create the buffer on every rank, then skip on all ranks or none.
        buffer = SymmetricMemoryManager.get_buffer("tp", process_group=module.tp_group)
        handle = buffer.symm_mem_hdl
        nvls = is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device()))
        nvls = torch.tensor(
            [float(nvls and handle is not None and bool(handle.multicast_ptr))], device="cuda"
        )
        torch.distributed.all_reduce(nvls, op=torch.distributed.ReduceOp.MIN)
        if nvls.item() != 1.0:
            pytest.skip("NVLS multicast unavailable")

        generator = torch.Generator(device="cuda").manual_seed(1234)
        hidden_states = torch.randn(
            self.sequence_length // self.tensor_parallel_size,
            self.micro_batch_size,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )

        events = []
        all_gather = inference_layers.multimem_all_gather
        reduce_scatter = inference_layers.multimem_reduce_scatter

        def record_all_gather(*args, barrier_before=False, **kwargs):
            events.append(("all_gather", barrier_before))
            return all_gather(*args, barrier_before=barrier_before, **kwargs)

        def record_reduce_scatter(*args, **kwargs):
            events.append(_REDUCE_SCATTER)
            return reduce_scatter(*args, **kwargs)

        with (
            mock.patch.object(inference_layers, "multimem_all_gather", record_all_gather),
            mock.patch.object(inference_layers, "multimem_reduce_scatter", record_reduce_scatter),
            torch.no_grad(),
        ):
            module(hidden_states, attention_mask=None)
        return events

    @staticmethod
    def _check(events, expected=None):
        assert any(kind == "all_gather" for kind, _ in events), f"no NVLS all-gather: {events}"
        assert not _unbarriered_all_gathers(events), (
            f"all-gathers {_unbarriered_all_gathers(events)} overwrite the symmetric buffer "
            f"right after another all-gather without barrier_before: {events}"
        )
        if expected is not None:
            assert events == expected

    @pytest.mark.parametrize("case", _HYBRID_CASES)
    def test_hybrid_layer(self, case):
        config = self._config(
            case.config_class, num_layers=len(case.pattern), **case.config_overrides
        )
        stack = HybridStack(
            config,
            case.stack_spec.submodules,
            layer_config_list=validate_segment_layers(case.pattern, config),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "pp", "cp"]
            ),
        )
        expected = None if case.layer_trace is None else case.layer_trace * len(case.pattern)
        self._check(self._symmetric_collectives(stack.cuda().eval()), expected)

    def test_gpt_decoder_block(self):
        config = self._config(num_layers=2)
        spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False)
        block = TransformerBlock(config, spec)
        self._check(self._symmetric_collectives(block.cuda().eval()))


def test_every_inference_hybrid_layer_spec_has_a_case():
    """A new inference layer type needs a case above, so its all-gathers get checked."""
    covered = set()
    for param in _HYBRID_CASES:
        (case,) = param.values
        covered.update(id(getattr(case.stack_spec.submodules, name)) for name in case.layer_specs)
    for stack_spec in (hybrid_inference_stack_spec, gated_delta_product_inference_stack_spec):
        for field in dataclasses.fields(stack_spec.submodules):
            spec = getattr(stack_spec.submodules, field.name)
            if spec is None or spec is IdentityOp or field.name in _TESTED_ELSEWHERE:
                continue
            assert id(spec) in covered, f"no all-gather barrier case for {field.name}"
