# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact determinism check for the end-to-end Transformer Engine MegaMoE path."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.moe.fused_a2a import HAVE_TE_EP, nccl_ep_finalize
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.configs import gpt_base
from tests.unit_tests.determinism.utils import (
    assert_bit_exact,
    capture_rng_state,
    collect_grads,
    reset_quantizer_state,
    restore_rng_state,
    zero_grads,
)
from tests.unit_tests.test_utilities import Utils

_EP_SIZE = 4
_SEQ_LEN = 32
_MICRO_BATCH = 4
_VOCAB_SIZE = 128


def _is_megamoe_available() -> bool:
    """Check the runtime gates needed to execute FusedMoeEp."""
    if not HAVE_TE_EP or not torch.cuda.is_available():
        return False
    try:
        from transformer_engine.pytorch.ops import (
            GroupedLinear,
            MoeCombine,
            MoeDispatch,
            ScaledSwiGLU,
        )
        from transformer_engine.pytorch.ops.fused.moe_ep import (
            _cudnn_megamoe_supported,
            _import_cudnn_moe_ep,
        )
    except ImportError:
        return False
    del GroupedLinear, MoeCombine, MoeDispatch, ScaledSwiGLU
    return (
        torch.cuda.get_device_capability() == (10, 7)
        and _cudnn_megamoe_supported()
        and _import_cudnn_moe_ep() is not None
    )


requires_megamoe = pytest.mark.skipif(
    not _is_megamoe_available(),
    reason="requires NCCL-EP and the cuDNN MegaMoE Transformer Engine fusion on SM107",
)


@pytest.fixture(autouse=True)
def _restore_test_state():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    nccl_ep_finalize()
    Utils.destroy_model_parallel()
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _build_model(cuda_graph_impl: str = "none") -> GPTModel:
    config = TransformerConfig(
        **(
            gpt_base()
            | {
                "num_layers": 1,
                "hidden_size": 256,
                "ffn_hidden_size": 1024,
                "num_attention_heads": 8,
                "tensor_model_parallel_size": 1,
                "expert_model_parallel_size": _EP_SIZE,
                "num_moe_experts": 8,
                "moe_router_topk": 2,
                "moe_router_load_balancing_type": "aux_loss",
                "moe_token_dispatcher_type": "flex",
                "moe_flex_dispatcher_backend": "ncclep",
                "moe_grouped_gemm": True,
                "moe_single_grouped_weight": True,
                "moe_use_transformer_engine_fused_moe": True,
                "gated_linear_unit": True,
                "activation_func": F.silu,
                "add_bias_linear": False,
                "fp8": "e4m3",
                "fp8_recipe": "mxfp8",
                "fp8_param": True,
                "moe_mlp_glu_interleave_size": 32,
                "moe_expert_rank_capacity_factor": 8.0,
                "cuda_graph_impl": cuda_graph_impl,
                "cuda_graph_modules": [],
                "cuda_graph_warmup_steps": 1,
            }
        )
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=True,
        use_te_op_fuser=(
            config.use_transformer_engine_op_fuser
            or config.moe_use_transformer_engine_fused_moe
        ),
    )
    return GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=_SEQ_LEN,
        position_embedding_type="rope",
    ).cuda()


def _make_inputs() -> dict[str, torch.Tensor]:
    input_ids = torch.randint(
        0, _VOCAB_SIZE, (_MICRO_BATCH, _SEQ_LEN), device="cuda", dtype=torch.long
    )
    return {
        "input_ids": input_ids,
        "position_ids": torch.arange(_SEQ_LEN, device="cuda", dtype=torch.long)
        .unsqueeze(0)
        .repeat(_MICRO_BATCH, 1),
        "attention_mask": torch.ones(
            _MICRO_BATCH, 1, _SEQ_LEN, _SEQ_LEN, device="cuda", dtype=torch.bool
        ),
        # Supplying labels makes GPTModel run its normal cross-entropy path after MegaMoE.
        "labels": torch.roll(input_ids, shifts=-1, dims=1),
    }


def _assert_megamoe_selected(model: GPTModel) -> None:
    fused_ops = []
    for module in model.modules():
        sequences = getattr(module, "_fused_moe_ops", None)
        if sequences is None:
            continue
        for sequence in sequences:
            fused_ops.extend(
                op
                for group in sequence._module_groups[0]._forward_ops
                for op in group
                if type(op).__name__ == "FusedMoeEp"
            )
    assert fused_ops, "the determinism test must exercise FusedMoeEp"


# TODO: Add input-gradient determinism coverage once the MegaMoE kernel outputs deterministic
# dprob. dprob contributes to the MoE input gradient through the router.
def _collect_expert_grads(model: GPTModel) -> dict[str, torch.Tensor]:
    """Collect routed-expert gradients, which do not depend on dprob."""
    all_grads = collect_grads([model])
    grads = {name: grad for name, grad in all_grads.items() if ".experts." in name}
    assert grads, "expected routed-expert gradients"
    return grads


@pytest.mark.internal
@requires_megamoe
def test_megamoe_cross_entropy_replays_bit_exactly():
    """Replay cross-entropy and routed-expert gradients bit-for-bit."""
    if Utils.world_size < _EP_SIZE:
        pytest.skip(f"requires at least {_EP_SIZE} GPUs")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=_EP_SIZE,
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(123)

    model = _build_model()
    inputs = _make_inputs()

    def fwd_bwd():
        loss = model(**inputs)
        loss.float().mean().backward()
        return loss.detach().clone(), _collect_expert_grads(model)

    state = capture_rng_state()
    loss_a, grads_a = fwd_bwd()
    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    restore_rng_state(state)
    zero_grads(model)
    reset_quantizer_state([model])
    loss_b, grads_b = fwd_bwd()

    assert_bit_exact(loss_a, grads_a, loss_b, grads_b)
    _assert_megamoe_selected(model)


@pytest.mark.internal
@requires_megamoe
def test_megamoe_local_graph_replay_matches_eager():
    """Exercise MegaMoE through Megatron's local graph record/capture/replay machinery."""
    if Utils.world_size < _EP_SIZE:
        pytest.skip(f"requires at least {_EP_SIZE} GPUs")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=_EP_SIZE,
    )
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(123)
    model = _build_model(cuda_graph_impl="local")
    inputs = _make_inputs()

    def fwd_bwd():
        loss = model(**inputs).float().mean()
        loss.backward()
        return loss.detach().clone(), _collect_expert_grads(model)

    eager_loss, eager_grads = fwd_bwd()
    _assert_megamoe_selected(model)
    fused_experts = next(
        module for module in model.modules() if getattr(module, "_fused_moe_ops", None) is not None
    )
    sequence_before_capture = fused_experts._fused_moe_ops[0]
    zero_grads(model)
    reset_quantizer_state([model])

    create_cudagraphs()
    replay_loss, replay_grads = fwd_bwd()
    assert fused_experts._fused_moe_ops[0] is sequence_before_capture

    torch.testing.assert_close(replay_loss, eager_loss, rtol=0.0, atol=0.0)
    assert replay_grads.keys() == eager_grads.keys()
    for name in eager_grads:
        torch.testing.assert_close(replay_grads[name], eager_grads[name], rtol=0.0, atol=0.0)

    for module in model.modules():
        manager = getattr(module, "cudagraph_manager", None)
        if manager is None:
            continue
        for runner in manager.cudagraph_runners:
            if hasattr(runner, "fwd_graph"):
                del runner.fwd_graph
            if hasattr(runner, "bwd_graph"):
                del runner.bwd_graph
    torch.cuda.synchronize()
