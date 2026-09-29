# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed tests for the inference-kernel training forward.

``moe_inference_training_forward`` runs the training MoE forward through the
expert kernel generation uses, and takes its backward from a TE recompute pass.
Its purpose is train/generation parity in RL, where the importance ratio compares
a training log-prob against a generation log-prob, so any kernel difference
between the two shows up as a bias no amount of tuning removes. Three properties
matter, and they are what these tests cover:

1. The path runs at all, forward and backward, under expert parallelism, and the
   value pass really reaches the inference kernel rather than silently falling
   back to TE (which still gives finite outputs and gradients).
2. The value pass reproduces the generation forward bitwise.
3. The NVLS symmetric heap, which cannot grow, is sized for the largest pass
   rather than the first one.

Runs on any GPU count that is a power of two and divides the expert count; the
expert-parallel size is the world size. The construction-time checks that need no
GPU are in ``test_inference_training_forward_config.py``.
"""

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.token_dispatcher_inference import (
    InferenceAllGatherDispatcherBase,
    NCCLAllGatherDispatcher,
    NVLSAllGatherVDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.test_utilities import Utils

NUM_EXPERTS = 16
# topk=8 rather than a small value: the two paths obtain their expert indices
# differently -- generation reads them from torch.topk on the router scores, the
# value pass recovers them from TopKRouter's boolean map -- and at topk=2 the
# combine is a two-term sum, which commutes exactly and cannot see an ordering
# difference that topk=8 can.
ROUTER_TOPK = 8
LOCAL_TOKENS = 8

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _config(**overrides):
    """A squared-ReLU inference-optimized MoE layer config, the nano-v3 activation.

    Squared ReLU is the point: it has no gated weight layout, which is what rules
    out kernels that implement only SwiGLU, so the vLLM and torch kernels are the
    ones this mode has to work for.
    """
    base = dict(
        num_layers=1,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=128,
        moe_router_topk=ROUTER_TOPK,
        moe_router_score_function="softmax",
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        activation_func=squared_relu,
        gated_linear_unit=False,
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="inference_optimized",
        inference_grouped_gemm_backend="vllm",
        inference_moe_token_dispatcher_type="nccl",
        expert_model_parallel_size=Utils.world_size,
        # The value pass stores no activations, so the backward comes from a
        # recompute pass.
        recompute_granularity="selective",
        recompute_modules=["moe"],
    )
    base.update(overrides)
    return TransformerConfig(**base)


def _generation_and_training_configs(backend, dispatcher, **overrides):
    """``(generation, training)`` configs differing only in the parity flag."""
    common = dict(
        inference_grouped_gemm_backend=backend,
        inference_moe_token_dispatcher_type=dispatcher,
        **overrides,
    )
    return _config(**common), _config(**common, moe_inference_training_forward=True)


def _skip_unless_supported(backend, dispatcher):
    if NUM_EXPERTS % Utils.world_size:
        pytest.skip(f"num_experts={NUM_EXPERTS} must divide EP={Utils.world_size}")
    if dispatcher == "nvls" and (Utils.world_size & (Utils.world_size - 1)):
        pytest.skip("NVLS Triton symmetric-memory barrier requires power-of-two EP size.")
    if backend == "torch" and not (
        is_torch_min_version("2.10") and hasattr(torch, '_grouped_mm')
    ):
        pytest.skip("Requires PyTorch >= 2.10 with torch._grouped_mm")


def _build_layer(config, for_inference: bool = False, max_tokens: int = LOCAL_TOKENS):
    """Build the MoE layer.

    ``for_inference`` stands up what the dynamic inference context allocates in
    production for a generation layer: the valid-tokens scalar and, on NVLS, the
    symmetric heap. A training-only layer deliberately leaves both unallocated,
    which is the real situation the value pass has to handle itself.
    """
    from megatron.core import parallel_state

    if for_inference:
        NCCLAllGatherDispatcher.allocate_buffers()
        if config.inference_moe_token_dispatcher_type == 'nvls':
            NVLSAllGatherVDispatcher.allocate_buffers(
                per_rank_worst_case_token_count=max_tokens,
                topk=config.moe_router_topk,
                hidden_size=config.moe_latent_size or config.hidden_size,
                ep_group=parallel_state.get_expert_model_parallel_group(),
            )
    return get_inference_optimized_moe_spec()(config=config).cuda()


def _copy_expert_weights(src, dst):
    """Give two layers identical expert and router weights.

    Done before either layer's first forward, while parameters still own their
    storage: the inference path redirects ``param.data`` into a concatenated
    buffer on first use.
    """
    dst.load_state_dict(src.state_dict())


def _hidden(config, seed, tokens=None):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        LOCAL_TOKENS if tokens is None else tokens,
        1,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )


def _elements_that_differ(got, want) -> int:
    """How many elements differ, on the worst rank.

    Reduced because each rank holds different tokens and different local experts,
    so the honest number is the worst one rather than whichever rank pytest
    happens to quote, and asserting on it makes the verdict rank-independent.
    """
    differing = torch.tensor([float((got != want).sum().item())], device="cuda")
    torch.distributed.all_reduce(differing, op=torch.distributed.ReduceOp.MAX)
    return int(differing.item())


@pytest.fixture(autouse=True)
def _parallel_state():
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
    # Registers the 'expert-parallel-rng' tracker state that expert weight init
    # requires. Utils.initialize_model_parallel does not do this.
    model_parallel_cuda_manual_seed(123)

    def reset():
        # Process-wide state. Leaked, one test's inference setup would mask a
        # missing allocation in the next, which is exactly the check the NVLS
        # arm has to make.
        InferenceAllGatherDispatcherBase._valid_tokens_tensor = None
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()

    reset()
    yield
    reset()
    Utils.destroy_model_parallel()


class TestValuePassRuns:
    """(1) The path runs, and the value pass reaches the inference kernel."""

    @pytest.mark.parametrize("dispatcher", ["nccl", "nvls"])
    @pytest.mark.parametrize("backend", ["torch", "vllm"])
    def test_forward_backward_runs(self, backend, dispatcher):
        _skip_unless_supported(backend, dispatcher)
        config = _config(
            inference_grouped_gemm_backend=backend,
            inference_moe_token_dispatcher_type=dispatcher,
            moe_inference_training_forward=True,
        )
        layer = _build_layer(config).train()
        hidden = _hidden(config, seed=0).requires_grad_(True)

        out, _ = layer(hidden)
        out.sum().backward()

        assert out.shape == hidden.shape
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
        # The backward must reach the expert weights through the recompute pass,
        # not just the input.
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        fc1_grad = layer.experts.linear_fc1.weight0.grad
        assert fc1_grad is not None, "expert weights received no gradient"
        assert torch.isfinite(fc1_grad).all()
        assert fc1_grad.abs().sum() > 0

    @pytest.mark.parametrize("backend", ["torch", "vllm"])
    def test_value_pass_is_handed_a_routing_map(self, backend):
        """The value pass reaches the inference kernel exactly once, with a routing map.

        These kernels route internally, so the routing map is what tells them
        which expert a token belongs to; without it there is nothing to run.

        Asserted separately from the forward above because the two failures look
        nothing alike. A layer that stops taking the inference kernel at all
        still produces a finite output and finite gradients -- through TE -- and
        the only thing wrong with it is that it is no longer the forward
        generation runs, which is the entire point of the mode.
        """
        _skip_unless_supported(backend, "nccl")
        config = _config(
            inference_grouped_gemm_backend=backend, moe_inference_training_forward=True
        )
        layer = _build_layer(config).train()
        experts = layer.experts
        original = experts._grouped_gemm_training_forward_pass
        seen = []

        def spy(hidden_states, probs, routing_map):
            seen.append(routing_map)
            return original(hidden_states, probs, routing_map=routing_map)

        experts._grouped_gemm_training_forward_pass = spy
        out, _ = layer(_hidden(config, seed=0).requires_grad_(True))
        out.sum().backward()

        assert len(seen) == 1, (
            f"expected exactly one value pass through the inference kernel, saw {len(seen)}; "
            "0 means the layer fell back to TE, >1 means the recompute took this path too"
        )
        assert seen[0] is not None, "the value pass ran without a routing map"
        assert seen[0].shape[0] > 0

    @pytest.mark.parametrize("backend", ["torch", "vllm"])
    def test_eval_forward_takes_the_inference_kernel(self, backend):
        """The log-prob pass runs under ``eval()`` and must not fall back to TE.

        RL frameworks compute the training log-prob under ``model.eval()``, so a
        gate on ``self.training`` would send exactly the pass that has to match
        generation down the TE path. No backward, so no recompute pairs with it.
        """
        _skip_unless_supported(backend, "nccl")
        config = _config(
            inference_grouped_gemm_backend=backend, moe_inference_training_forward=True
        )
        layer = _build_layer(config).eval()
        experts = layer.experts
        original = experts._grouped_gemm_training_forward_pass
        calls = []

        def spy(hidden_states, probs, routing_map):
            calls.append(1)
            return original(hidden_states, probs, routing_map=routing_map)

        experts._grouped_gemm_training_forward_pass = spy
        with torch.no_grad():
            layer(_hidden(config, seed=0))

        assert len(calls) == 1, "the eval forward did not take the inference kernel"


class TestBf16TrainGenParity:
    """(2) The value pass reproduces the generation forward bitwise, from BF16 weights.

    Run under ``eval()`` and ``no_grad``, which is the log-prob pass RL actually
    compares and the one whose disagreement shows up as KL. It is also the
    simpler pass to read: without a backward there is no recompute, so a
    mismatch is the value pass and nothing else.

    Parametrized over the dispatcher because the two answer different
    questions. ``nccl`` puts both sides on the same dispatcher, leaving the
    kernel and the weights as the only difference. ``nvls`` is production's
    default and the harder case: the value pass has to allocate the symmetric
    heap generation gets from the inference context, and the two combines
    reduce across EP ranks through different collectives if it does not.
    """

    @pytest.mark.parametrize("dispatcher", ["nccl", "nvls"])
    @pytest.mark.parametrize("backend", ["torch", "vllm"])
    def test_training_forward_matches_generation_forward(self, backend, dispatcher):
        _skip_unless_supported(backend, dispatcher)
        gen_config, train_config = _generation_and_training_configs(backend, dispatcher)
        gen_layer = _build_layer(gen_config, for_inference=True).eval()
        train_layer = _build_layer(train_config).eval()
        _copy_expert_weights(gen_layer, train_layer)

        hidden = _hidden(train_config, seed=1)
        with torch.no_grad(), InferenceMode.active():
            gen_out, _ = gen_layer(hidden)
        with torch.no_grad():
            train_out, _ = train_layer(hidden)

        differing = _elements_that_differ(train_out, gen_out)
        assert differing == 0, (
            f"bf16 train/gen parity broken on {backend}/{dispatcher}: "
            f"{differing}/{train_out.numel()} elements differ. Same weights and the same "
            "kernel, so this is routing, the dispatcher, or the weight layout, not rounding"
        )


class TestSymmetricHeapSizing:
    """(3) The NVLS symmetric heap is sized for the largest pass, not the first.

    Symmetric memory is allocated once and cannot grow. A log-prob microbatch is
    padded to its own longest sequence, so a run can see 512 tokens and later 576;
    sizing from the first would then fail on the second, at Step 2 of a job whose
    Step 1 passed.
    """

    SMALL, LARGE = LOCAL_TOKENS, 2 * LOCAL_TOKENS

    def _train_layer(self, **overrides):
        config = _config(
            inference_moe_token_dispatcher_type="nvls",
            moe_inference_training_forward=True,
            **overrides,
        )
        return config, _build_layer(config).eval()

    def test_a_configured_bound_covers_a_larger_later_pass(self):
        _skip_unless_supported("vllm", "nvls")
        config, layer = self._train_layer(moe_inference_training_max_tokens_per_rank=self.LARGE)
        with torch.no_grad():
            layer(_hidden(config, seed=0, tokens=self.SMALL))
            out, _ = layer(_hidden(config, seed=1, tokens=self.LARGE))
        assert torch.isfinite(out).all()

    def test_without_a_bound_a_larger_later_pass_fails_loudly(self):
        """A named assertion on every rank, rather than a hang or a bad answer."""
        _skip_unless_supported("vllm", "nvls")
        config, layer = self._train_layer()
        with torch.no_grad():
            layer(_hidden(config, seed=0, tokens=self.SMALL))
            with pytest.raises(AssertionError, match="moe_inference_training_max_tokens_per_rank"):
                layer(_hidden(config, seed=1, tokens=self.LARGE))

    def test_the_bound_is_a_floor_not_a_cap(self):
        """A first pass larger than the bound still gets buffers that fit it."""
        _skip_unless_supported("vllm", "nvls")
        config, layer = self._train_layer(moe_inference_training_max_tokens_per_rank=self.SMALL)
        with torch.no_grad():
            out, _ = layer(_hidden(config, seed=0, tokens=self.LARGE))
        assert torch.isfinite(out).all()
