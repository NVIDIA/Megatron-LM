# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions import transformer_engine as transformer_engine_module
from megatron.core.extensions.transformer_engine import TELinear, TERMSNormDuplicatedLinear
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe import moe_layer as moe_layer_module
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class _RecordingLinear(torch.nn.Module):
    calls = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.parallel_mode = None
        self._tp_group = None
        self.register_parameter("weight", torch.nn.Parameter(torch.empty(1, 1)))
        self.calls.append((args, kwargs))

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return transformer_engine_module.TELinear.sharded_state_dict(
            self, prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )


class _RecordingRMSNormLinear(_RecordingLinear):
    calls = []


class _RecordingInferenceLinear(torch.nn.Module):
    calls = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.calls.append((args, kwargs))


class _DummyDispatcher(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _FakeProcessGroup:
    def rank(self):
        return 0

    def size(self):
        return 1


def _build_dummy_module(*args, **kwargs):
    return torch.nn.Module()


def _record_checkpoint_call(checkpoint_calls, state_dict, prefix, *args, **kwargs):
    checkpoint_calls[prefix] = (state_dict, kwargs)
    return {}


def test_latent_projections_use_owning_tp_group_for_checkpoint_only(monkeypatch):
    """Latent projections use the owning TP group only for checkpoint metadata."""
    tp_group = _FakeProcessGroup()
    ep_group = _FakeProcessGroup()
    gtp_remat_group = _FakeProcessGroup()
    gtp_replica_group = _FakeProcessGroup()
    checkpoint_dp_cp_group = _FakeProcessGroup()
    pg_collection = ProcessGroupCollection(
        tp=tp_group, ep=ep_group, dp_cp=gtp_replica_group, gtp_remat=gtp_remat_group
    )
    checkpoint_calls = {}
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=1,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_token_dispatcher_type="allgather",
        moe_ffn_hidden_size=16,
        moe_latent_size=4,
        use_cpu_initialization=True,
        add_bias_linear=False,
        gtp_remat_opt_in_modules=["moe_latent_proj"],
    )
    submodules = MoESubmodules(experts=_build_dummy_module, router=_build_dummy_module)

    _RecordingLinear.calls.clear()
    monkeypatch.setattr(moe_layer_module, "HAVE_TE", True)
    monkeypatch.setattr(moe_layer_module, "TELinear", _RecordingLinear)
    monkeypatch.setattr(moe_layer_module, "MoEAllGatherTokenDispatcher", _DummyDispatcher)
    monkeypatch.setattr(
        transformer_engine_module,
        "make_sharded_tensors_for_checkpoint",
        lambda state_dict, prefix, *args, **kwargs: _record_checkpoint_call(
            checkpoint_calls, state_dict, prefix, *args, **kwargs
        ),
    )

    layer = MoELayer(config, submodules, pg_collection=pg_collection)

    assert len(_RecordingLinear.calls) == 2
    assert all(kwargs["parallel_mode"] == "duplicated" for _, kwargs in _RecordingLinear.calls)
    assert all("tp_group" not in kwargs for _, kwargs in _RecordingLinear.calls)
    assert all(kwargs["gtp_remat_group"] is gtp_remat_group for _, kwargs in _RecordingLinear.calls)
    assert all(
        kwargs["gtp_replica_group"] is gtp_replica_group for _, kwargs in _RecordingLinear.calls
    )
    assert layer.fc1_latent_proj._tp_group is tp_group
    assert layer.fc2_latent_proj._tp_group is tp_group

    layer.sharded_state_dict(prefix="moe.", metadata={"dp_cp_group": checkpoint_dp_cp_group})

    for name in ("fc1_latent_proj", "fc2_latent_proj"):
        state_dict, checkpoint_kwargs = checkpoint_calls[f"moe.{name}."]
        assert set(state_dict) == {"weight"}
        assert checkpoint_kwargs["tp_group"] is tp_group
        assert checkpoint_kwargs["dp_cp_group"] is checkpoint_dp_cp_group

    layer.fc1_latent_proj.parallel_mode = "column"
    with pytest.raises(
        AssertionError, match="TELinear sharded_state_dict can only be used with duplicated"
    ):
        layer.sharded_state_dict(prefix="moe.", metadata={"dp_cp_group": checkpoint_dp_cp_group})

    from megatron.core.tensor_parallel import inference_layers as inference_layers_module

    _RecordingInferenceLinear.calls.clear()
    monkeypatch.setattr(inference_layers_module, "InferenceLinear", _RecordingInferenceLinear)
    config.transformer_impl = "inference_optimized"
    MoELayer(config, submodules, pg_collection=pg_collection)

    assert len(_RecordingInferenceLinear.calls) == 2
    assert all(
        "gtp_remat_group" not in kwargs and "gtp_replica_group" not in kwargs
        for _, kwargs in _RecordingInferenceLinear.calls
    )


def test_latent_rmsnorm_up_projection_receives_gtp_groups(monkeypatch):
    """The fused up-projection receives GTP groups while keeping duplicated TP execution."""
    tp_group = _FakeProcessGroup()
    ep_group = _FakeProcessGroup()
    gtp_remat_group = _FakeProcessGroup()
    gtp_replica_group = _FakeProcessGroup()
    pg_collection = ProcessGroupCollection(
        tp=tp_group, ep=ep_group, dp_cp=gtp_replica_group, gtp_remat=gtp_remat_group
    )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=1,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_token_dispatcher_type="allgather",
        moe_ffn_hidden_size=16,
        moe_latent_size=4,
        moe_latent_up_projection_rmsnorm=True,
        use_cpu_initialization=True,
        add_bias_linear=False,
        gtp_remat_opt_in_modules=["moe_latent_proj"],
    )
    submodules = MoESubmodules(experts=_build_dummy_module, router=_build_dummy_module)

    _RecordingLinear.calls.clear()
    _RecordingRMSNormLinear.calls.clear()
    monkeypatch.setattr(moe_layer_module, "HAVE_TE", True)
    monkeypatch.setattr(moe_layer_module, "TELinear", _RecordingLinear)
    monkeypatch.setattr(moe_layer_module, "TERMSNormDuplicatedLinear", _RecordingRMSNormLinear)
    monkeypatch.setattr(moe_layer_module, "MoEAllGatherTokenDispatcher", _DummyDispatcher)

    layer = MoELayer(config, submodules, pg_collection=pg_collection)

    assert len(_RecordingLinear.calls) == 1
    assert len(_RecordingRMSNormLinear.calls) == 1
    _, kwargs = _RecordingRMSNormLinear.calls[0]
    assert kwargs["parallel_mode"] == "duplicated"
    assert kwargs["tp_group"] is tp_group
    assert kwargs["gtp_remat_group"] is gtp_remat_group
    assert kwargs["gtp_replica_group"] is gtp_replica_group
    assert layer.fc2_latent_proj._tp_group is tp_group


@pytest.mark.skipif(
    not is_te_min_version("1.7.0.dev0"),
    reason="Transformer Engine RMSNormLinear requires TE 1.7.0 or later.",
)
def test_rmsnorm_duplicated_linear_checkpoint_metadata(monkeypatch):
    """Duplicated RMSNormLinear keeps its explicit TP group in checkpoint metadata."""
    tp_group = _FakeProcessGroup()
    dp_cp_group = _FakeProcessGroup()
    checkpoint_call = {}
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
    )
    layer = TERMSNormDuplicatedLinear(
        16,
        32,
        parallel_mode="duplicated",
        config=config,
        init_method=config.output_layer_init_method,
        bias=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        tp_group=tp_group,
    )

    def record_checkpoint_call(state_dict, prefix, axis_map, sharded_offsets, **kwargs):
        checkpoint_call.update(
            state_dict=state_dict, prefix=prefix, axis_map=axis_map, kwargs=kwargs
        )
        return {}

    monkeypatch.setattr(
        transformer_engine_module, "make_sharded_tensors_for_checkpoint", record_checkpoint_call
    )
    layer.sharded_state_dict(prefix="latent_up.", metadata={"dp_cp_group": dp_cp_group})

    assert layer._tp_group is tp_group
    assert checkpoint_call["prefix"] == "latent_up."
    assert checkpoint_call["axis_map"] is None
    assert checkpoint_call["kwargs"]["tp_group"] is tp_group
    assert checkpoint_call["kwargs"]["dp_cp_group"] is dp_cp_group


class TestLatentMoELayer:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"),
        reason="Expert with TE Linear is only supported in TE 1.7.0 and later.",
    )
    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [4])
    @pytest.mark.parametrize("use_te,grouped_gemm", [(True, True), (True, False), (False, False)])
    @pytest.mark.parametrize("moe_latent_size", [8, 16])
    def test_latent_moe_layer(
        self, num_moe_experts, moe_token_dispatcher_type, use_te, grouped_gemm, moe_latent_size
    ):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            moe_ffn_hidden_size=128,
            moe_shared_expert_intermediate_size=128,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            moe_latent_size=moe_latent_size,
        )
        if use_te:
            transformer_layer_submodules = get_gpt_layer_with_transformer_engine_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
            )
        else:
            transformer_layer_submodules = get_gpt_layer_local_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
            )
        submodules = get_submodules(transformer_layer_submodules.mlp)
        assert isinstance(submodules, MoESubmodules)
        moe_layer = MoELayer(self.transformer_config, submodules)
        moe_layer.cuda()
        config = moe_layer.config

        assert (
            moe_layer.shared_experts.linear_fc1.weight.shape[1] == config.hidden_size
        ), "Shared expert computation has to happen in hidden dimension."
        assert (
            moe_layer.shared_experts.linear_fc2.weight.shape[0] == config.hidden_size
        ), "Shared expert computation has to happen in hidden dimension."
        if grouped_gemm:
            for i in range(num_moe_experts):
                fc1_weight = getattr(moe_layer.experts.linear_fc1, f"weight{i}")
                fc2_weight = getattr(moe_layer.experts.linear_fc2, f"weight{i}")
                assert (
                    fc1_weight.shape[1] == config.moe_latent_size
                ), f"Shape mismatch for expert {i} {fc1_weight.shape=}"
                assert (
                    fc2_weight.shape[0] == config.moe_latent_size
                ), f"Shape mismatch for expert {i} {fc2_weight.shape=}"
        else:
            for i in range(num_moe_experts):
                expert = moe_layer.experts.local_experts[i]
                assert (
                    expert.linear_fc1.weight.shape[1] == config.moe_latent_size
                ), f"Shape mismatch for expert {i} {fc1_weight.shape=}"
                assert (
                    expert.linear_fc2.weight.shape[0] == config.moe_latent_size
                ), f"Shape mismatch for expert {i} {fc2_weight.shape=}"
        assert (
            moe_layer.router.weight.shape[1] == config.hidden_size
        ), "MoE routing has to happen in hidden dimension."

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((32, 2, config.hidden_size))
        hidden_states = hidden_states.cuda()
        output, _ = moe_layer(hidden_states)
        assert output.shape[2] == config.hidden_size

        Utils.destroy_model_parallel()

    def test_latent_up_projection_rmsnorm_requires_latent_size(self):
        with pytest.raises(
            ValueError, match="moe_latent_up_projection_rmsnorm requires moe_latent_size"
        ):
            TransformerConfig(
                num_layers=1,
                hidden_size=32,
                num_attention_heads=4,
                moe_latent_up_projection_rmsnorm=True,
            )

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"),
        reason="Expert with TE Linear is only supported in TE 1.7.0 and later.",
    )
    @pytest.mark.parametrize(
        "tp_size",
        [
            1,
            pytest.param(
                2,
                marks=pytest.mark.skipif(
                    Utils.world_size < 2 or Utils.world_size % 2,
                    reason="TP2/SP runtime test requires an even world size >= 2.",
                ),
            ),
        ],
    )
    def test_latent_up_projection_rmsnorm(self, tp_size):
        Utils.initialize_model_parallel(tp_size, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_moe_experts=4,
            moe_token_dispatcher_type="alltoall",
            moe_router_topk=2,
            moe_grouped_gemm=True,
            moe_ffn_hidden_size=128,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=tp_size > 1,
            moe_latent_size=16,
            moe_latent_up_projection_rmsnorm=True,
        )
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=4, moe_grouped_gemm=True
            ).mlp
        )
        moe_layer = MoELayer(config, submodules).cuda()

        assert isinstance(moe_layer.fc1_latent_proj, TELinear)
        assert isinstance(moe_layer.fc2_latent_proj, TERMSNormDuplicatedLinear)
        assert moe_layer.fc2_latent_proj.normalization == "RMSNorm"
        assert moe_layer.fc2_latent_proj.parallel_mode is None
        assert moe_layer.fc2_latent_proj.tp_size == 1
        assert (
            moe_layer.fc2_latent_proj._tp_group is parallel_state.get_tensor_model_parallel_group()
        )
        assert all(
            getattr(param, "allreduce", False)
            and getattr(param, "sequence_parallel", False) == (tp_size > 1)
            and not getattr(param, "tensor_model_parallel", True)
            for param in moe_layer.fc2_latent_proj.parameters()
        )
        if tp_size > 1:
            for param in moe_layer.fc2_latent_proj.parameters():
                gathered = [torch.empty_like(param) for _ in range(tp_size)]
                torch.distributed.all_gather(
                    gathered, param, group=moe_layer.fc2_latent_proj._tp_group
                )
                for replica in gathered[1:]:
                    torch.testing.assert_close(gathered[0], replica, rtol=0, atol=0)

        latent = 4 * torch.randn(8, config.moe_latent_size, device="cuda", dtype=torch.bfloat16) + 2
        latent.requires_grad_(True)
        output, output_bias = moe_layer.fc2_latent_proj(latent)
        assert output_bias is None

        latent_ref = latent.detach().float().requires_grad_(True)
        norm_weight_ref = (
            moe_layer.fc2_latent_proj.layer_norm_weight.detach().float().requires_grad_(True)
        )
        linear_weight_ref = moe_layer.fc2_latent_proj.weight.detach().float().requires_grad_(True)
        normalized_ref = latent_ref * torch.rsqrt(
            latent_ref.square().mean(dim=-1, keepdim=True) + config.layernorm_epsilon
        )
        normalized_ref = normalized_ref * norm_weight_ref
        output_ref = torch.nn.functional.linear(normalized_ref, linear_weight_ref)
        torch.testing.assert_close(output.float(), output_ref, rtol=1e-2, atol=1e-2)

        output_grad = torch.randn_like(output)
        output.backward(output_grad)
        output_ref.backward(output_grad.float())
        torch.testing.assert_close(latent.grad.float(), latent_ref.grad, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(
            moe_layer.fc2_latent_proj.layer_norm_weight.grad.float(),
            norm_weight_ref.grad,
            rtol=3e-2,
            atol=3e-2,
        )
        torch.testing.assert_close(
            moe_layer.fc2_latent_proj.weight.grad.float(),
            linear_weight_ref.grad,
            rtol=3e-2,
            atol=3e-2,
        )

        Utils.destroy_model_parallel()

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="MXFP8 requires compute capability >= 10.0 (Blackwell).",
    )
    @pytest.mark.skipif(
        not is_te_min_version("2.1.0"),
        reason="Transformer Engine MXFP8 requires TE 2.1.0 or later.",
    )
    def test_latent_up_projection_rmsnorm_mxfp8(self):
        """MXFP8 RMSNorm up-projection produces finite forward and backward tensors."""
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            params_dtype=torch.bfloat16,
            fp8="e4m3",
            fp8_recipe="mxfp8",
        )
        layer = TERMSNormDuplicatedLinear(
            32,
            64,
            parallel_mode="duplicated",
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            tp_group=parallel_state.get_tensor_model_parallel_group(),
        ).cuda()
        latent = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        with get_fp8_context(config):
            output, output_bias = layer(latent)
        assert output_bias is None
        assert layer.fp8
        assert output.dtype == torch.bfloat16
        assert output.shape == (32, 64)
        assert torch.isfinite(output).all()

        output.square().mean().backward()
        for tensor in (latent.grad, layer.layer_norm_weight.grad, layer.weight.grad):
            assert tensor is not None
            assert torch.isfinite(tensor).all()
            assert torch.count_nonzero(tensor)

        Utils.destroy_model_parallel()
