# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed MoE coverage for the TE grouped-tensor expert path.

The dispatchers place expert-token padding at different boundaries:

* All-to-All returns unpadded expert segments and TEGroupedMLP pads them before FC1.
* DeepEP communicates first, then its local fused permutation pads expert segments.
* HybridEP fuses communication, permutation, and expert-segment padding.
* NCCL-EP returns aligned expert segments from fused dispatch, like HybridEP. Its non-op-fuser
  grouped-tensor integration is not enabled yet, so its parity and lifecycle cases remain skipped.

The numerical tests compare each grouped-tensor configuration with the old discrete-parameter,
CPU-split path on the same dispatcher. The lifecycle tests inspect the actual expert-compute
boundary to ensure padding rows are zero and the dispatcher-specific inverse removes them.
"""

import inspect
import os
from typing import Dict

import pytest
import torch
import torch.nn.functional as F

from megatron.core import config as mcore_config
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.fused_a2a import (
    HAVE_DEEP_EP,
    HAVE_HYBRIDEP,
    reset_hybrid_ep_buffer,
)
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import fused_permute_and_pad_with_probs
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.internal,
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]


_ALIGN_SIZE = 256
_HIDDEN_SIZE = 256
_MOE_FFN_HIDDEN_SIZE = 256
_NUM_LOCAL_EXPERTS = 2
_NUM_LOCAL_TOKENS = 128
_TOLERANCES = {"rtol": 1e-2, "atol": 1e-2}
_NCCL_EP_GROUPED_TENSOR_UNSUPPORTED_REASON = (
    "NCCL-EP support for the non-op-fuser grouped-tensor expert path is not implemented yet"
)

_PARAMETER_LAYOUTS = (
    pytest.param(False, False, False, id="discrete-weight-no-bias"),
    pytest.param(True, False, False, id="single-weight-no-bias"),
    pytest.param(False, True, False, id="discrete-weight-discrete-bias"),
    pytest.param(True, True, False, id="single-weight-discrete-bias"),
    pytest.param(False, True, True, id="discrete-weight-single-bias"),
    pytest.param(True, True, True, id="single-weight-single-bias"),
)


def _require_test_environment(dispatcher: str) -> int:
    """Validate runtime support and return the EP world size used by the test."""
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        pytest.skip("DeepEP/HybridEP parity requires at least two distributed ranks")
    if dispatcher == "deepep" and not HAVE_DEEP_EP:
        pytest.skip("DeepEP is not available")
    if dispatcher == "hybridep" and not HAVE_HYBRIDEP:
        pytest.skip("HybridEP is not available")
    if dispatcher == "deepep" and fused_permute_and_pad_with_probs is None:
        pytest.skip("DeepEP grouped-tensor padding requires TE fused permute-and-pad")

    try:
        from transformer_engine.pytorch.module import GroupedLinear
    except ImportError:
        pytest.skip("Transformer Engine GroupedLinear is not available")
    parameters = inspect.signature(GroupedLinear.__init__).parameters
    if "use_grouped_tensor" not in parameters or "single_grouped_bias" not in parameters:
        pytest.skip("Installed TE lacks native grouped-tensor parameter support")
    return world_size


def _dispatcher_options(dispatcher: str) -> Dict[str, object]:
    """Return the production padding configuration for one dispatcher."""
    if dispatcher == "alltoall":
        return {
            "moe_token_dispatcher_type": "alltoall",
            "moe_flex_dispatcher_backend": None,
            "moe_permute_fusion": True,
        }
    if dispatcher == "deepep":
        # DeepEP communicates first. Its fused local permutation then groups and pads tokens.
        return {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "deepep",
            "moe_permute_fusion": True,
        }
    if dispatcher == "hybridep":
        # HybridEP owns permutation and padding inside its fused dispatch/combine kernels.
        return {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "hybridep",
            "moe_permute_fusion": True,
        }
    if dispatcher == "ncclep":
        # NCCL-EP dispatch packs aligned expert segments, but the module grouped-tensor path is
        # intentionally config-rejected until its end-to-end numerics and padding are validated.
        return {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "ncclep",
            "moe_permute_fusion": True,
            "moe_expert_rank_capacity_factor": 8.0,
            # The lifecycle assertions inspect the dynamically narrowed expert buffer. Static
            # NCCL-EP intentionally exposes the full receive-capacity buffer instead.
            "moe_ncclep_static_shape": False,
        }
    raise ValueError(f"Unknown dispatcher {dispatcher!r}")


def _build_moe_layer(
    dispatcher: str,
    *,
    ep_size: int,
    use_grouped_tensor: bool,
    single_grouped_weight: bool,
    use_bias: bool,
    single_grouped_bias: bool,
) -> MoELayer:
    """Build a small real TE MoE layer without using the TE operation fuser."""
    options = _dispatcher_options(dispatcher)
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=8,
        num_moe_experts=ep_size * _NUM_LOCAL_EXPERTS,
        moe_ffn_hidden_size=_MOE_FFN_HIDDEN_SIZE,
        use_cpu_initialization=False,
        add_bias_linear=use_bias,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_router_load_balancing_type="none",
        moe_router_topk=2,
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_use_grouped_tensor=use_grouped_tensor,
        moe_single_grouped_weight=single_grouped_weight,
        moe_single_grouped_bias=single_grouped_bias,
        use_transformer_engine_op_fuser=False,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
        sequence_parallel=False,
        **options,
    )
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            num_experts=transformer_config.num_moe_experts, moe_grouped_gemm=True
        ).mlp
    )
    assert isinstance(submodules, MoESubmodules)
    layer = MoELayer(transformer_config, submodules)
    layer = Float16Module(layer.config, layer).module
    layer.cuda()
    layer.set_layer_number(0)
    return layer


def _copy_linear_parameters(reference, target) -> None:
    """Copy discrete expert parameters into either a discrete or packed target layout."""
    for parameter_name in ("weight", "bias"):
        if parameter_name == "bias" and not reference.use_bias:
            continue
        reference_parts = torch.stack(
            [
                getattr(reference, f"{parameter_name}{idx}").detach()
                for idx in range(reference.num_gemms)
            ]
        )
        target_is_grouped = getattr(target, f"single_grouped_{parameter_name}")
        if target_is_grouped:
            grouped_parameter = getattr(target, parameter_name)
            grouped_parameter.rowwise_data.view_as(reference_parts).copy_(reference_parts)
        else:
            for idx, part in enumerate(reference_parts):
                getattr(target, f"{parameter_name}{idx}").copy_(part)


@torch.no_grad()
def _copy_layer_parameters(reference: MoELayer, target: MoELayer) -> None:
    """Give reference and target identical router and expert parameters."""
    target_parameters = dict(target.named_parameters())
    for name, parameter in reference.named_parameters():
        if not name.startswith("experts."):
            target_parameters[name].copy_(parameter)
    _copy_linear_parameters(reference.experts.linear_fc1, target.experts.linear_fc1)
    _copy_linear_parameters(reference.experts.linear_fc2, target.experts.linear_fc2)


def _canonical_gradient(linear, parameter_name: str) -> torch.Tensor:
    """Return expert gradients as one [experts, ...] FP32 tensor for either layout."""
    if getattr(linear, f"single_grouped_{parameter_name}"):
        gradient = getattr(linear, parameter_name).grad
        assert gradient is not None, f"Grouped {parameter_name} parent has no gradient"
        return gradient.reshape(linear.num_gemms, -1).float()

    gradients = [getattr(linear, f"{parameter_name}{idx}").grad for idx in range(linear.num_gemms)]
    assert all(gradient is not None for gradient in gradients)
    return torch.stack([gradient.reshape(-1) for gradient in gradients]).float()


def _run_forward_backward(
    layer: MoELayer, base_input: torch.Tensor, grad_output: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Run one MoE step and collect values sensitive to dispatch and parameter layout."""
    layer.zero_grad(set_to_none=True)
    hidden_states = base_input.detach().clone().requires_grad_(True)
    output, _ = layer(hidden_states)
    output.backward(grad_output)

    result = {
        "output": output.detach(),
        "input_grad": hidden_states.grad.detach(),
        "router_grad": layer.router.weight.grad.detach(),
        "fc1_weight_grad": _canonical_gradient(layer.experts.linear_fc1, "weight"),
        "fc2_weight_grad": _canonical_gradient(layer.experts.linear_fc2, "weight"),
    }
    if layer.config.add_bias_linear:
        result["fc1_bias_grad"] = _canonical_gradient(layer.experts.linear_fc1, "bias")
        result["fc2_bias_grad"] = _canonical_gradient(layer.experts.linear_fc2, "bias")
    return result


def _run_numerical_parity_case(
    dispatcher: str, *, single_grouped_weight: bool, use_bias: bool, single_grouped_bias: bool
) -> None:
    """Compare grouped-tensor execution with the old path on the same dispatcher."""
    ep_size = _require_test_environment(dispatcher)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, expert_model_parallel_size=ep_size
    )
    mcore_config.ENABLE_EXPERIMENTAL = True

    _set_random_seed(seed_=1234, data_parallel_random_init=False)
    reference = _build_moe_layer(
        dispatcher,
        ep_size=ep_size,
        use_grouped_tensor=False,
        single_grouped_weight=False,
        use_bias=use_bias,
        single_grouped_bias=False,
    )
    target = _build_moe_layer(
        dispatcher,
        ep_size=ep_size,
        use_grouped_tensor=True,
        single_grouped_weight=single_grouped_weight,
        use_bias=use_bias,
        single_grouped_bias=single_grouped_bias,
    )
    _copy_layer_parameters(reference, target)

    base_input = torch.randn(
        _NUM_LOCAL_TOKENS, 1, _HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    grad_output = torch.randn_like(base_input)

    reference_result = _run_forward_backward(reference, base_input, grad_output)
    target_result = _run_forward_backward(target, base_input, grad_output)

    assert target.experts._use_grouped_tensor
    assert not reference.experts._use_grouped_tensor
    assert reference_result.keys() == target_result.keys()
    for name in reference_result:
        torch.testing.assert_close(
            target_result[name],
            reference_result[name],
            msg=lambda message, value=name: f"{dispatcher} {value} mismatch: {message}",
            **_TOLERANCES,
        )


def _make_padding_mask(
    real_tokens_per_expert: torch.Tensor, padded_tokens_per_expert: torch.Tensor
) -> torch.Tensor:
    """Build a mask for the padding suffix in every expert-major segment.

    For example, using an alignment of four for readability:

    ``real_tokens_per_expert   = tensor([3, 2])``
    ``padded_tokens_per_expert = tensor([4, 4])``

    The packed expert-major rows are ``[e0 real x3][e0 pad x1][e1 real x2][e1 pad x2]``,
    so this function returns ``[F, F, F, T, F, F, T, T]``. Indexing the packed hidden states or
    probabilities with that mask selects only the synthetic rows that must contain exact zeros.
    """
    masks = []
    for real_count, padded_count in zip(
        real_tokens_per_expert.cpu().tolist(), padded_tokens_per_expert.cpu().tolist()
    ):
        # Each expert contributes a False prefix for real rows followed by a True padding suffix.
        masks.append(torch.zeros(real_count, dtype=torch.bool, device="cuda"))
        masks.append(torch.ones(padded_count - real_count, dtype=torch.bool, device="cuda"))
    return torch.cat(masks)


def _infer_real_tokens_per_expert(
    padded_probs: torch.Tensor, padded_tokens_per_expert: torch.Tensor
) -> torch.Tensor:
    """Infer real expert counts from the zero suffix in every padded probability segment.

    This dropless test uses FP32 softmax router probabilities, so every routed token has a
    nonzero probability. All padding implementations use exact zeros and append them after the
    real rows in each expert-major segment. This gives one dispatcher-independent source of truth
    without inspecting backend-specific routing metadata.
    """
    real_counts = []
    offset = 0
    for padded_count in padded_tokens_per_expert.cpu().tolist():
        segment = padded_probs[offset : offset + padded_count]
        nonzero_rows = segment != 0
        real_count = int(nonzero_rows.sum().item())

        # Padding must be one contiguous suffix. Interspersed zero rows would preserve the total
        # nonzero count while still violating the expert-major layout expected by grouped GEMM.
        assert torch.all(nonzero_rows[:real_count])
        assert not torch.any(nonzero_rows[real_count:])
        real_counts.append(real_count)
        offset += padded_count

    assert offset == padded_probs.numel()
    return torch.tensor(real_counts, dtype=torch.int64, device=padded_probs.device)


def _install_padding_probes(layer: MoELayer, monkeypatch):
    """Observe the tensors crossing each padding boundary without changing execution.

    ``register_forward_pre_hook`` runs immediately before the selected module's ``forward``.
    The hook receives the module and the tuple of positional arguments that forward is about to
    consume. Returning ``None`` leaves those arguments unchanged, so these hooks are read-only
    probes rather than replacements for any production operation.

    There are two relevant boundaries. ``layer.experts`` sees what the token dispatcher hands to
    TEGroupedMLP, while ``linear_fc1`` sees the final padded tensors and CUDA expert counts that
    TE's grouped-tensor GEMM actually consumes. They are different for AllToAll, where TEGroupedMLP
    owns padding, but identical for DeepEP and HybridEP, whose fused dispatch paths already pad.
    """
    captured = {"padding_calls": 0, "unpadding_calls": 0}

    def capture_dispatcher_input(_module, args):
        # TEGroupedMLP.forward(hidden, tokens_per_expert, permuted_probs) is about to run.
        # DeepEP and HybridEP have already padded at this boundary, so preserve their router
        # probabilities for the generic real-row inference below. Detach so the test does not
        # retain the graph, and clone in case downstream computation reuses the input storage.
        captured["dispatcher_probs"] = args[2].detach().clone()

    def capture_fc1_input(_module, args):
        # GroupedLinear.forward(hidden, m_splits, ...) is about to run. This is the authoritative
        # view of the rows and device-side split tensor presented to the grouped GEMM.
        captured["padded_hidden"] = args[0].detach().clone()
        captured["padded_counts"] = args[1].detach().clone()

    # Pre-hooks observe module inputs before either module can transform them. The returned hook
    # handles need not be retained because each test owns this layer and executes one forward.
    layer.experts.register_forward_pre_hook(capture_dispatcher_input)
    layer.experts.linear_fc1.register_forward_pre_hook(capture_fc1_input)

    def capture_padding(_module, args, output):
        # quantization_padding is called once for hidden states and once for router probabilities
        # in the AllToAll path. A forward hook is used here because the padded tensor is its output.
        captured["padding_calls"] += 1
        padded_tensor = output[0]
        # Probability padding receives [tokens, 1], whereas hidden-state padding receives
        # [tokens, hidden_size]. Preserve the padded probabilities for an exact-zero assertion.
        if args[0].shape[-1] == 1:
            captured["padded_probs"] = padded_tensor.detach().clone().reshape(-1)

    def capture_unpadding(_module, _args, output):
        # AllToAll uses TEGroupedMLP's explicit unpadding module after expert compute. Recording
        # the call proves that its locally inserted padding reaches the matching removal path.
        captured["unpadding_calls"] += 1

    layer.experts.quantization_padding.register_forward_hook(capture_padding)
    layer.experts.quantization_unpadding.register_forward_hook(capture_unpadding)

    # All dispatchers share this final restoration API.
    original_combine_postprocess = layer.token_dispatcher.combine_postprocess

    def capture_combine_postprocess(hidden_states, *args, **kwargs):
        output = original_combine_postprocess(hidden_states, *args, **kwargs)
        captured["restored_shape"] = output.shape
        return output

    monkeypatch.setattr(layer.token_dispatcher, "combine_postprocess", capture_combine_postprocess)

    return captured


def _run_padding_lifecycle_case(dispatcher: str, monkeypatch) -> None:
    """Verify exact zero padding, 256 alignment, no double-padding, and unpadding.

    This test follows one real MoE forward through dispatch, expert compute, and restoration. It
    independently reconstructs the number of real tokens assigned to each local expert, then
    compares that metadata with the padded CUDA ``m_splits`` observed directly at FC1. Numerical
    parity with the legacy backend is tested separately; this case focuses on padding ownership
    and the structural contract required by TE's grouped-tensor kernels.
    """
    # EP spans the whole torchrun world so every tested dispatcher performs real communication.
    # TP remains one to keep the independently reconstructed local-expert counts unambiguous.
    ep_size = _require_test_environment(dispatcher)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, expert_model_parallel_size=ep_size
    )
    mcore_config.ENABLE_EXPERIMENTAL = True

    # Use the strictest parameter layout. If packed weight and packed bias reach the native
    # grouped-tensor path correctly, discrete parameter layouts use the same padding lifecycle.
    _set_random_seed(seed_=1357, data_parallel_random_init=False)
    layer = _build_moe_layer(
        dispatcher,
        ep_size=ep_size,
        use_grouped_tensor=True,
        single_grouped_weight=True,
        use_bias=True,
        single_grouped_bias=True,
    )
    # Install observers before the forward so they capture dispatcher output, FC1 input, and the
    # common dispatcher boundary that returns to the original token layout.
    captured = _install_padding_probes(layer, monkeypatch)

    torch.manual_seed(9753)
    hidden_states = torch.randn(
        _NUM_LOCAL_TOKENS, 1, _HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    output, _ = layer(hidden_states)

    padded_counts = captured["padded_counts"]

    # TE's grouped-tensor API requires device-resident int64 splits. Every expert segment must be
    # represented by its m_split, and the physical FC1 input must contain their total.
    assert padded_counts.device.type == "cuda"
    assert padded_counts.dtype == torch.int64
    assert torch.all(padded_counts % _ALIGN_SIZE == 0)
    assert captured["padded_hidden"].shape[0] == padded_counts.sum().item()

    if dispatcher == "alltoall":
        # All-to-All returns real expert rows; TEGroupedMLP pads hidden states and probs itself.
        assert captured["padding_calls"] == 2
        assert captured["unpadding_calls"] == 1
    else:
        # DeepEP and HybridEP already return padded rows. TEGroupedMLP must not pad them again.
        assert captured["padding_calls"] == 0
        assert captured["unpadding_calls"] == 0
        # For fused dispatchers, probabilities are already padded when they enter TEGroupedMLP,
        # so the experts pre-hook is the correct observation point for the zero check below.
        captured["padded_probs"] = captured["dispatcher_probs"].reshape(-1)

    # Router probabilities provide a common representation across all dispatchers: real routed
    # rows are nonzero and padded rows are an exact-zero suffix.
    real_counts = _infer_real_tokens_per_expert(captured["padded_probs"], padded_counts)
    expected_padded_counts = ((real_counts + _ALIGN_SIZE - 1) // _ALIGN_SIZE) * _ALIGN_SIZE
    torch.testing.assert_close(padded_counts, expected_padded_counts, rtol=0, atol=0)

    # Expert-major layout is [expert 0 real][expert 0 pad][expert 1 real][expert 1 pad]...
    # Build that exact mask and require both hidden states and routing probabilities to use
    # numerical zero for every synthetic row. Merely allocating the right shape is insufficient.
    padding_mask = _make_padding_mask(real_counts, padded_counts)
    # An expert may already have a 256-aligned token count and legitimately need no padding.
    # Boolean indexing with an empty mask is valid; otherwise these checks inspect every pad row.
    assert not torch.any(captured["padded_hidden"][padding_mask])
    assert not torch.any(captured["padded_probs"][padding_mask])

    # Regardless of where a backend removes padding, the common dispatcher postprocess contract
    # must restore exactly the shape that entered this MoE layer.
    assert captured["restored_shape"] == hidden_states.shape

    # The public MoE contract is unchanged by internal alignment. Run backward as a final check
    # that padding/unpadding preserved a connected, finite autograd path to the original input.
    assert output.shape == hidden_states.shape
    output.float().square().mean().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


class TestGroupedTensorDispatcherNumerics:
    """Distributed numerical and padding coverage for grouped-tensor dispatchers."""

    def setup_method(self, method):
        if not torch.distributed.is_available() or Utils.world_size < 2:
            pytest.skip("Distributed dispatcher tests must be launched with torchrun")
        self._old_single_param_env = os.environ.get("NVTE_GROUPED_LINEAR_SINGLE_PARAM")
        self._previous_experimental = mcore_config.ENABLE_EXPERIMENTAL
        os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
        Utils.initialize_distributed()

    def teardown_method(self, method):
        try:
            mcore_config.ENABLE_EXPERIMENTAL = self._previous_experimental
            reset_hybrid_ep_buffer()
            Utils.destroy_model_parallel()
        finally:
            if self._old_single_param_env is None:
                os.environ.pop("NVTE_GROUPED_LINEAR_SINGLE_PARAM", None)
            else:
                os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = self._old_single_param_env

    @pytest.mark.parametrize(
        "single_grouped_weight,use_bias,single_grouped_bias", _PARAMETER_LAYOUTS
    )
    @pytest.mark.timeout(180)
    def test_alltoall_grouped_tensor_moe_parity(
        self, single_grouped_weight, use_bias, single_grouped_bias
    ):
        """All-to-All grouped-tensor MoE forward/backward matches its legacy expert path."""
        _run_numerical_parity_case(
            "alltoall",
            single_grouped_weight=single_grouped_weight,
            use_bias=use_bias,
            single_grouped_bias=single_grouped_bias,
        )

    @pytest.mark.parametrize(
        "single_grouped_weight,use_bias,single_grouped_bias", _PARAMETER_LAYOUTS
    )
    @pytest.mark.timeout(180)
    def test_deepep_grouped_tensor_moe_parity(
        self, single_grouped_weight, use_bias, single_grouped_bias
    ):
        """DeepEP grouped-tensor MoE forward/backward matches its legacy expert path."""
        _run_numerical_parity_case(
            "deepep",
            single_grouped_weight=single_grouped_weight,
            use_bias=use_bias,
            single_grouped_bias=single_grouped_bias,
        )

    @pytest.mark.parametrize(
        "single_grouped_weight,use_bias,single_grouped_bias", _PARAMETER_LAYOUTS
    )
    @pytest.mark.timeout(180)
    def test_hybridep_grouped_tensor_moe_parity(
        self, single_grouped_weight, use_bias, single_grouped_bias
    ):
        """HybridEP grouped-tensor MoE forward/backward matches its legacy expert path."""
        _run_numerical_parity_case(
            "hybridep",
            single_grouped_weight=single_grouped_weight,
            use_bias=use_bias,
            single_grouped_bias=single_grouped_bias,
        )

    @pytest.mark.skip(reason=_NCCL_EP_GROUPED_TENSOR_UNSUPPORTED_REASON)
    @pytest.mark.parametrize(
        "single_grouped_weight,use_bias,single_grouped_bias", _PARAMETER_LAYOUTS
    )
    @pytest.mark.timeout(180)
    def test_ncclep_grouped_tensor_moe_parity(
        self, single_grouped_weight, use_bias, single_grouped_bias
    ):
        """NCCL-EP grouped-tensor parity coverage reserved for future enablement."""
        _run_numerical_parity_case(
            "ncclep",
            single_grouped_weight=single_grouped_weight,
            use_bias=use_bias,
            single_grouped_bias=single_grouped_bias,
        )

    @pytest.mark.timeout(180)
    def test_alltoall_grouped_tensor_padding_lifecycle(self, monkeypatch):
        """All-to-All explicitly pads in TEGroupedMLP and removes it before combine."""
        _run_padding_lifecycle_case("alltoall", monkeypatch)

    @pytest.mark.timeout(180)
    def test_deepep_grouped_tensor_padding_lifecycle(self, monkeypatch):
        """DeepEP fused local permutation pads, and local unpermute removes those rows."""
        _run_padding_lifecycle_case("deepep", monkeypatch)

    @pytest.mark.timeout(180)
    def test_hybridep_grouped_tensor_padding_lifecycle(self, monkeypatch):
        """HybridEP fused dispatch pads, and fused combine returns the original token shape."""
        _run_padding_lifecycle_case("hybridep", monkeypatch)

    @pytest.mark.skip(reason=_NCCL_EP_GROUPED_TENSOR_UNSUPPORTED_REASON)
    @pytest.mark.timeout(180)
    def test_ncclep_grouped_tensor_padding_lifecycle(self, monkeypatch):
        """NCCL-EP padding lifecycle coverage reserved for future enablement."""
        _run_padding_lifecycle_case("ncclep", monkeypatch)
