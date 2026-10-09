# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA, GatedDeltaNet
from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.ssm.gated_delta_net_test_utils import GatedDeltaNetTestBase

try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
except ImportError:
    HAVE_FUSED_PRE_GDR = False
else:
    HAVE_FUSED_PRE_GDR = callable(causal_conv1d_bwd_function)


def _build_gdn(config):
    """Build one GDN/GDN2 layer on the current CUDA device."""
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)
    gdn_spec = get_experimental_attention_variant_module_spec(config=config)
    gdn = gdn_spec.module(
        config,
        submodules=gdn_spec.submodules,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=pg_collection,
    )
    return gdn.cuda().bfloat16()


def _hidden_states(test, micro_batch_size, seq_length):
    return torch.randn(
        (seq_length // test.sp_size // test.cp_size, micro_batch_size, test.gdn.config.hidden_size),
        device=torch.cuda.current_device(),
        dtype=torch.bfloat16,
        requires_grad=True,
    )


def _forward_backward(gdn, hidden_states):
    output, _ = gdn(hidden_states, None)
    output.float().sum().backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in gdn.named_parameters()
        if param.grad is not None
    }
    return output.detach(), grads, hidden_states.grad.detach().clone()


def _assert_identical(reference, candidate):
    ref_out, ref_grads, ref_input_grad = reference
    out, grads, input_grad = candidate
    rank = torch.distributed.get_rank()
    assert torch.equal(out, ref_out), f"Output not identical ({rank=})"
    assert torch.equal(input_grad, ref_input_grad), f"Input grad not identical ({rank=})"
    assert set(grads) == set(ref_grads)
    for name in ref_grads:
        assert torch.equal(grads[name], ref_grads[name]), f"Grad not identical for {name} ({rank=})"


def _zero_grads(gdn, hidden_states):
    hidden_states.grad = None
    for param in gdn.parameters():
        param.grad = None


def _strided_gate_pre_gated_delta_rule(
    module, qkvzba, cu_seqlens_q=None, seq_idx=None, cp_size_headwise=1, cp_group_headwise=None
):
    """Stand in for the fused pre-GDR kernel, returning its strided output-gate view.

    Query, key, value, and the scalar gates come from the unfused preparation. The gate
    is then replaced with the same strided view of the projection's Z channels that
    ``fused_streamed_pre_gated_delta_rule`` returns, so a recompute pass that frees the
    projection too early zeros the norm.
    """
    del seq_idx, cp_group_headwise
    batch = qkvzba.shape[1]
    seq_len = qkvzba.shape[0]
    query, key, value, gate, beta, g = module.pre_gated_delta_rule(
        qkvzba, batch, seq_len, module.cp_size, module.pg_collection.cp, cu_seqlens_q
    )
    qk_channels = module.qk_dim_local_tp // cp_size_headwise
    v_channels = module.v_dim_local_tp // cp_size_headwise
    num_value_heads = v_channels // module.value_head_dim
    z_offset = 2 * qk_channels + v_channels
    gate_view = (
        qkvzba[:, :, z_offset : z_offset + v_channels]
        .view(seq_len, batch, num_value_heads, module.value_head_dim)
        .permute(1, 0, 2, 3)
    )
    assert gate_view.untyped_storage().data_ptr() == qkvzba.untyped_storage().data_ptr()
    assert torch.equal(gate_view, gate), "strided Z view does not match the prepared gate"
    return query, key, value, gate_view, beta, g


@pytest.mark.parametrize("use_gdn2", [False, True], ids=["gdn", "gdn2"])
@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size"),
    [(1, False, 1), (2, False, 1), (2, True, 1), (1, False, 2), (2, False, 2), (2, True, 2)],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNet(GatedDeltaNetTestBase):
    @pytest.mark.parametrize(
        "recompute_modules",
        [
            ["gdn_norm_out"],
            ["gdn_in_proj"],
            ["gdn_qkv"],
            ["gdn_in_proj", "gdn_qkv"],
            ["gdn_in_proj", "gdn_qkv", "gdn_norm_out"],
        ],
        ids=lambda modules: "+".join(modules),
    )
    def test_selective_recompute(self, recompute_modules):
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        def build_gdn(config):
            gdn_spec = get_experimental_attention_variant_module_spec(config=config)
            gdn = gdn_spec.module(
                config,
                submodules=gdn_spec.submodules,
                layer_number=1,
                bias=False,
                conv_bias=False,
                conv_init=1.0,
                use_qk_l2norm=True,
                A_init_range=(1, 16),
                pg_collection=pg_collection,
            )
            return gdn.cuda().bfloat16()

        def run(gdn, hidden_states):
            output, _ = gdn(hidden_states, None)
            output.float().sum().backward()
            grads = {
                name: param.grad.detach()
                for name, param in gdn.named_parameters()
                if param.grad is not None
            }
            input_grad = hidden_states.grad.detach().clone()
            return output.detach(), grads, input_grad

        micro_batch_size = 2
        seq_length = 64
        base_config = copy.deepcopy(self.transformer_config)
        rec_config = copy.deepcopy(self.transformer_config)
        rec_config.recompute_granularity = "selective"
        rec_config.recompute_modules = recompute_modules

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        hidden_states = torch.randn(
            (
                seq_length // self.sp_size // self.cp_size,
                micro_batch_size,
                self.gdn.config.hidden_size,
            ),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        # --- Baseline (no recompute) ---
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        base_gdn = build_gdn(base_config)
        assert base_gdn.recompute_norm_out is False
        assert base_gdn.recompute_in_proj is False
        assert base_gdn.recompute_qkv is False
        base_output, base_grads, base_input_grad = run(base_gdn, hidden_states)
        hidden_states.grad = None
        assert base_gdn.norm_out_checkpoint is None
        del base_gdn
        torch.cuda.empty_cache()

        # --- Recompute ---
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        rec_gdn = build_gdn(rec_config)
        assert rec_gdn.recompute_norm_out == ("gdn_norm_out" in recompute_modules)
        assert rec_gdn.recompute_in_proj == ("gdn_in_proj" in recompute_modules)
        assert rec_gdn.recompute_qkv == ("gdn_qkv" in recompute_modules)

        # Every requested checkpoint must release its outputs during the forward pass.
        discarded = []
        original_discard = CheckpointWithoutOutput._discard_outputs

        def recording_discard(ckpt):
            original_discard(ckpt)
            discarded.append([out.untyped_storage().nbytes() for out in ckpt.outputs])

        with mock.patch.object(CheckpointWithoutOutput, "_discard_outputs", recording_discard):
            rec_output, rec_grads, rec_input_grad = run(rec_gdn, hidden_states)
        assert len(discarded) == len(recompute_modules)
        assert all(nbytes == 0 for outputs in discarded for nbytes in outputs)
        if "gdn_norm_out" in recompute_modules:
            assert rec_gdn.norm_out_checkpoint is not None

        rank = torch.distributed.get_rank()
        assert torch.equal(rec_output, base_output), f"Output not identical ({rank=})"
        assert torch.equal(rec_input_grad, base_input_grad), f"Input grad not identical ({rank=})"
        assert set(rec_grads.keys()) == set(base_grads.keys())
        for name in base_grads:
            assert torch.equal(
                rec_grads[name], base_grads[name]
            ), f"Grad not identical for {name} ({rank=})"

    @pytest.mark.parametrize(
        "recompute_modules",
        [["gdn_in_proj"], ["gdn_in_proj", "gdn_norm_out"]],
        ids=lambda modules: "+".join(modules),
    )
    def test_in_proj_recompute_with_strided_gate_view(self, recompute_modules):
        """gdn_in_proj must not free the fused path's strided output-gate view early."""
        if self.use_gdn2:
            pytest.skip("The fused pre-GDR gate is a strided view on GDN1 only.")

        base_config = copy.deepcopy(self.transformer_config)
        base_config.gdn_pre_gated_delta_rule_fusion = True
        rec_config = copy.deepcopy(base_config)
        rec_config.recompute_granularity = "selective"
        rec_config.recompute_modules = list(recompute_modules)

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        hidden_states = _hidden_states(self, micro_batch_size=2, seq_length=64)
        with mock.patch.object(
            GatedDeltaNet,
            "_fused_streamed_pre_gated_delta_rule",
            _strided_gate_pre_gated_delta_rule,
        ):
            model_parallel_cuda_manual_seed(42)
            torch.manual_seed(42)
            base_gdn = _build_gdn(base_config)
            reference = _forward_backward(base_gdn, hidden_states)
            _zero_grads(base_gdn, hidden_states)
            del base_gdn
            torch.cuda.empty_cache()

            model_parallel_cuda_manual_seed(42)
            torch.manual_seed(42)
            rec_gdn = _build_gdn(rec_config)
            reference_after_base = _forward_backward(rec_gdn, hidden_states)
        _assert_identical(reference, reference_after_base)

    def test_in_proj_recompute_with_real_pre_gated_delta_rule_fusion(self):
        """Real fused pre-GDR plus gdn_in_proj stays bit-exact, including the gate view."""
        if self.use_gdn2:
            pytest.skip("Pre-GDR fusion is GDN1-specific.")
        if not HAVE_FUSED_PRE_GDR:
            pytest.skip("causal-conv1d fused backward is not installed.")

        base_config = copy.deepcopy(self.transformer_config)
        base_config.gdn_pre_gated_delta_rule_fusion = True
        rec_config = copy.deepcopy(base_config)
        rec_config.recompute_granularity = "selective"
        rec_config.recompute_modules = ["gdn_in_proj", "gdn_norm_out"]

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        hidden_states = _hidden_states(self, micro_batch_size=2, seq_length=64)

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        base_gdn = _build_gdn(base_config)
        reference = _forward_backward(base_gdn, hidden_states)
        _zero_grads(base_gdn, hidden_states)
        del base_gdn
        torch.cuda.empty_cache()

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        rec_gdn = _build_gdn(rec_config)
        _assert_identical(reference, _forward_backward(rec_gdn, hidden_states))


def _init_qkv_offload():
    """Start a one-chunk offload handler that keeps every small activation."""
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        FineGrainedActivationOffloadingInterface,
        PipelineOffloadManager,
    )

    PipelineOffloadManager.reset_instance()
    FineGrainedActivationOffloadingInterface.init_chunk_handler(
        pp_rank=0,
        vp_size=None,
        vp_stage=None,
        min_offloaded_tensor_size=1,
        delta_offload_bytes_across_pp_ranks=0,
        activation_offload_fraction=1.0,
    )


def _warmup_gdn_qkv_offload_bytes():
    """Bytes copied to CPU during warmup, before the last-group margin disables them."""
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        PipelineOffloadManager,
    )

    manager = PipelineOffloadManager.get_instance()
    return sum(
        group.total_offload_bytes
        for chunk in manager._cached_chunks_forward
        for group in chunk.offload_groups
        if group._name == "gdn_qkv"
    )


@pytest.mark.parametrize("use_gdn2", [False, True], ids=["gdn", "gdn2"])
@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size"),
    [(1, False, 1), (2, False, 1), (2, True, 1), (1, False, 2), (2, False, 2), (2, True, 2)],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNetQKVOffload(GatedDeltaNetTestBase):
    """gdn_qkv offload matches the eager layer on the same GPU grid as recompute."""

    @pytest.mark.parametrize(
        "recompute_modules",
        [[], ["gdn_qkv"], ["gdn_norm_out"], ["gdn_qkv", "gdn_norm_out"]],
        ids=lambda modules: "+".join(modules) if modules else "offload-only",
    )
    def test_qkv_offload(self, recompute_modules):
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            PipelineOffloadManager,
        )

        base_config = copy.deepcopy(self.transformer_config)
        off_config = copy.deepcopy(self.transformer_config)
        off_config.fine_grained_activation_offloading = True
        off_config.offload_modules = ["gdn_qkv"]
        if recompute_modules:
            off_config.recompute_granularity = "selective"
            off_config.recompute_modules = list(recompute_modules)

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        hidden_states = _hidden_states(self, micro_batch_size=2, seq_length=64)

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        base_gdn = _build_gdn(base_config)
        reference = _forward_backward(base_gdn, hidden_states)
        _zero_grads(base_gdn, hidden_states)
        del base_gdn
        torch.cuda.empty_cache()

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        off_gdn = _build_gdn(off_config)
        assert off_gdn.offload_gdn_qkv is True
        _init_qkv_offload()
        try:
            # Warmup performs the D2H copy. A single group is then kept on GPU by the
            # offload margin, so the following iteration exercises that steady state too.
            warmup = _forward_backward(off_gdn, hidden_states)
            _assert_identical(reference, warmup)
            assert _warmup_gdn_qkv_offload_bytes() > 0, "gdn_qkv offload copied no bytes"
            _zero_grads(off_gdn, hidden_states)
            PipelineOffloadManager.get_instance().reset()
            steady = _forward_backward(off_gdn, hidden_states)
            _assert_identical(reference, steady)
        finally:
            PipelineOffloadManager.reset_instance()


def _gdn_config(**overrides):
    """A minimal GDN TransformerConfig; constructing it runs the recompute/offload checks."""
    kwargs = dict(
        hidden_size=64,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_layers=1,
        linear_attention_freq=1,
        num_attention_heads=4,
        normalization="RMSNorm",
        activation_func=F.silu,
        experimental_attention_variant="gdn",
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.mark.parametrize("module", ["gdn_in_proj", "gdn_qkv"])
def test_gdn_recompute_modules_are_accepted(module):
    config = _gdn_config(recompute_granularity="selective", recompute_modules=[module])
    assert module in config.recompute_modules


@pytest.mark.parametrize("module", ["gdn_in_proj", "gdn_qkv"])
def test_gdn_recompute_modules_require_gdn_variant(module):
    with pytest.raises(ValueError, match=f"{module} in recompute_modules is only supported"):
        TransformerConfig(
            hidden_size=64,
            num_layers=1,
            num_attention_heads=4,
            recompute_granularity="selective",
            recompute_modules=[module],
        )


def test_gdn_qkv_recompute_rejects_pre_gated_delta_rule_fusion():
    with pytest.raises(ValueError, match="gdn_qkv in recompute_modules is not supported"):
        _gdn_config(
            recompute_granularity="selective",
            recompute_modules=["gdn_qkv"],
            gdn_pre_gated_delta_rule_fusion=True,
        )


def test_gdn_in_proj_recompute_allows_pre_gated_delta_rule_fusion():
    config = _gdn_config(
        recompute_granularity="selective",
        recompute_modules=["gdn_in_proj"],
        gdn_pre_gated_delta_rule_fusion=True,
    )
    assert config.recompute_modules == ["gdn_in_proj"]


def test_gdn_qkv_offload_is_accepted():
    config = _gdn_config(fine_grained_activation_offloading=True, offload_modules=["gdn_qkv"])
    assert config.offload_modules == ["gdn_qkv"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (dict(experimental_attention_variant=None), "gdn_qkv in offload_modules is only supported"),
        (
            dict(gdn_pre_gated_delta_rule_fusion=True),
            "gdn_qkv in offload_modules is not supported with gdn_pre_gated_delta_rule_fusion",
        ),
        (
            dict(recompute_granularity="selective", recompute_modules=["gdn_in_proj"]),
            "gdn_qkv cannot be set in offload_modules together with gdn_in_proj",
        ),
    ],
    ids=["non-gdn", "pre-gdr-fusion", "with-gdn-in-proj"],
)
def test_gdn_qkv_offload_rejections(overrides, match):
    with pytest.raises(ValueError, match=match):
        _gdn_config(
            fine_grained_activation_offloading=True, offload_modules=["gdn_qkv"], **overrides
        )
