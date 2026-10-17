# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed tests for the FlashInfer inference-kernel training forward.

``moe_inference_training_forward`` runs the training MoE forward through the
FlashInfer routed MXFP8 or per-token NVFP4 kernel generation uses, and takes its
backward from a TE recompute pass. Those kernels are batch invariant but do not
match TE training, so without this a training log-prob and a generation log-prob of
the same token differ, and the importance ratio in RL carries that bias. The tests
cover:

1. The path runs, forward and backward, and the value pass really reaches the
   FlashInfer kernel rather than silently falling back to TE (which still gives
   finite outputs and gradients).
2. The value pass keeps reproducing the generation forward bitwise after the weights
   change, and independently of how many tokens share the launch. Single-batch
   parity is ``test_flashinfer_value_pass_matches_generation`` in
   ``test_moe_dispatching_and_routing.py``, next to the TE comparison it replaces.
3. The NVLS symmetric heap, which cannot grow, is sized for the largest pass
   rather than the first one.

Batch-invariant mode throughout, which the mode requires. Runs on any GPU count
that is a power of two and divides the expert count; the expert-parallel size is
the world size, with the NVLS dispatcher when it is larger than one. The
construction-time checks that need no GPU are in
``test_inference_training_forward_config.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.inference.moe.flashinfer_mxfp8 import HAVE_FLASHINFER_ROUTED_MXFP8
from megatron.core.inference.moe.flashinfer_nvfp4 import HAVE_FLASHINFER_ROUTED_NVFP4
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.token_dispatcher_inference import (
    InferenceAllGatherDispatcherBase,
    NCCLAllGatherDispatcher,
    NVLSAllGatherVDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

NUM_EXPERTS = 16
# topk=8 rather than a small value: generation reads its expert indices from
# torch.topk on the router scores, the value pass recovers them from TopKRouter's
# boolean map, and the kernel sums routes in the order it is given. At topk=2 the
# sum commutes exactly and cannot see an ordering difference that topk=8 can.
ROUTER_TOPK = 8
LOCAL_TOKENS = 8
# Generation decodes a few tokens per rank while a training forward takes a whole
# microbatch, so RL parity rests on a token's output not depending on how many
# others shared the launch.
GEN_TOKENS = 8
TRAIN_TOKEN_COUNTS = (512, 2048)

# FlashInfer routed MXFP8 supports only non-gated squared ReLU; per-token NVFP4
# supports squared ReLU and SwiGLU.
RECIPES = {
    "mxfp8": dict(fp8="hybrid", fp8_recipe="mxfp8", fp8_param=True, use_cpu_initialization=False),
    "nvfp4-squared_relu": dict(inference_moe_nvfp4_recipe="per_token"),
    "nvfp4-swiglu": dict(
        inference_moe_nvfp4_recipe="per_token", gated_linear_unit=True, activation_func=F.silu
    ),
}

pytestmark = [
    pytest.mark.internal,
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="FlashInfer routed MXFP8 and per-token NVFP4 MoE need SM100/SM103",
    ),
]


def _dispatcher():
    return "nvls" if Utils.world_size > 1 else "nccl"


def _config(recipe, **overrides):
    """A batch-invariant inference-optimized FlashInfer MoE layer config."""
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
        inference_grouped_gemm_backend="flashinfer",
        inference_moe_token_dispatcher_type=_dispatcher(),
        expert_model_parallel_size=Utils.world_size,
        attention_backend=AttnBackend.flash,
        flash_attention_version=4,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        batch_invariant_backend="te_native",
        # The value pass stores no activations, so the backward comes from a
        # recompute pass.
        recompute_granularity="selective",
        recompute_modules=["moe"],
    )
    base.update(RECIPES[recipe])
    base.update(overrides)
    return TransformerConfig(**base)


def _generation_and_training_configs(recipe, **overrides):
    """``(generation, training)`` configs differing only in the parity flag."""
    return (
        _config(recipe, **overrides),
        _config(recipe, **overrides, moe_inference_training_forward=True),
    )


def _skip_unless_supported(recipe):
    if NUM_EXPERTS % Utils.world_size:
        pytest.skip(f"num_experts={NUM_EXPERTS} must divide EP={Utils.world_size}")
    if Utils.world_size & (Utils.world_size - 1):
        pytest.skip("NVLS Triton symmetric-memory barrier requires power-of-two EP size.")
    if recipe == "mxfp8" and not HAVE_FLASHINFER_ROUTED_MXFP8:
        pytest.skip("FlashInfer routed MXFP8 MoE is unavailable")
    if recipe.startswith("nvfp4") and not HAVE_FLASHINFER_ROUTED_NVFP4:
        pytest.skip("FlashInfer per-token NVFP4 MoE is unavailable")


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


def _fp8_context(config, is_init=False):
    """TE's MXFP8 context for an MXFP8 config, a no-op otherwise."""
    import contextlib

    from megatron.core.fp8_utils import get_fp8_context

    if not config.fp8:
        return contextlib.nullcontext()
    return get_fp8_context(config, 0, is_init=is_init)


def _layer_pair(recipe, gen_config, train_config, max_tokens=LOCAL_TOKENS):
    """Generation and training layers holding the same weights, each in its own form.

    Generation converts as production does: MXFP8 replaces TE's MXFP8 parameters
    with MCore ``MXFP8Tensor``s at load, NVFP4 quantizes the BF16 stacks on the
    first forward. Training keeps its parameters and the value pass re-derives the
    kernel weights from them on every forward.
    """
    from megatron.core.inference.quantization.utils import quantize_model_to_mxfp8

    with _fp8_context(gen_config, is_init=True):
        gen_layer = _build_layer(gen_config, for_inference=True, max_tokens=max_tokens).eval()
    with _fp8_context(train_config, is_init=True):
        train_layer = _build_layer(train_config).eval()
    train_layer.load_state_dict(gen_layer.state_dict())
    if recipe == "mxfp8":
        assert train_layer.experts._expert_params_use_te_mxfp8()
        quantize_model_to_mxfp8(gen_layer, backend="triton")
    return gen_layer, train_layer


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
    """How many elements differ, on the worst rank."""
    differing = torch.tensor([float((got != want).sum().item())], device="cuda")
    torch.distributed.all_reduce(differing, op=torch.distributed.ReduceOp.MAX)
    return int(differing.item())


def _spy_on_value_pass(experts):
    """Count value passes and record the routing map each one is handed."""
    original = experts._inference_training_forward_pass
    seen = []

    def spy(hidden_states, probs, routing_map):
        seen.append(routing_map)
        return original(hidden_states, probs, routing_map=routing_map)

    experts._inference_training_forward_pass = spy
    return seen


@pytest.fixture(autouse=True)
def _parallel_state(monkeypatch):
    from megatron.core.inference import utils as inference_utils
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    # The routed MXFP8 kernel JIT-compiles without flashinfer-jit-cache; the layer's
    # check only guards production against the compile time.
    monkeypatch.setattr(inference_utils, "check_flashinfer_jit_cache_installed", lambda: None)

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
    # Global kernel patching: on before any layer is built, off again afterwards,
    # or it would silently change the next test.
    enable_batch_invariant_mode(backend="te_native")
    yield
    disable_batch_invariant_mode()
    reset()
    Utils.destroy_model_parallel()


class TestValuePassRuns:
    """(1) The path runs, and the value pass reaches the FlashInfer kernel."""

    @pytest.mark.parametrize("recipe", ["nvfp4-squared_relu", "nvfp4-swiglu"])
    def test_forward_backward_runs(self, recipe):
        """NVFP4 keeps BF16 parameters, so the TE recompute gives every weight a gradient."""
        _skip_unless_supported(recipe)
        config = _config(recipe, moe_inference_training_forward=True)
        layer = _build_layer(config).train()
        seen = _spy_on_value_pass(layer.experts)
        hidden = _hidden(config, seed=0).requires_grad_(True)

        out, _ = layer(hidden)
        out.sum().backward()

        assert len(seen) == 1, (
            f"expected exactly one value pass through the FlashInfer kernel, saw {len(seen)}; "
            "0 means the layer fell back to TE, >1 means the recompute took this path too"
        )
        assert seen[0] is not None and seen[0].shape[0] > 0
        assert out.shape == hidden.shape
        assert torch.isfinite(out).all()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        fc1_grad = layer.experts.linear_fc1.weight0.grad
        assert fc1_grad is not None, "expert weights received no gradient"
        assert torch.isfinite(fc1_grad).all()
        assert fc1_grad.abs().sum() > 0

    @pytest.mark.parametrize("recipe", list(RECIPES))
    def test_eval_forward_takes_the_flashinfer_kernel(self, recipe):
        """The log-prob pass runs under ``eval()`` and must not fall back to TE.

        RL frameworks compute the training log-prob under ``model.eval()``, so a
        gate on ``self.training`` would send exactly the pass that has to match
        generation down the TE path.
        """
        _skip_unless_supported(recipe)
        _, train_config = _generation_and_training_configs(recipe)
        _, layer = _layer_pair(recipe, *_generation_and_training_configs(recipe))
        seen = _spy_on_value_pass(layer.experts)
        with torch.no_grad(), _fp8_context(train_config):
            layer(_hidden(train_config, seed=0))
        assert len(seen) == 1, "the eval forward did not take the FlashInfer kernel"


class TestTrainGenParity:
    """(2) The value pass keeps reproducing the generation forward bitwise.

    Run under ``eval()`` and ``no_grad``, which is the log-prob pass RL compares.

    Not covered: the refit. Both layers start from the same parameters, whereas in
    RL generation's weights arrive as BF16 slices and are quantized on the receiver.
    """

    @pytest.mark.parametrize("recipe", ["nvfp4-squared_relu", "nvfp4-swiglu"])
    def test_value_pass_follows_a_weight_update(self, recipe):
        """After the parameters move, the value pass matches a refreshed generation.

        A value pass that cached its kernel weights would keep matching the old
        generation and miss this. The update is applied to both layers' BF16
        parameters, and generation requantizes as a refit would.
        """
        _skip_unless_supported(recipe)
        gen_config, train_config = _generation_and_training_configs(recipe)
        gen_layer, train_layer = _layer_pair(recipe, gen_config, train_config)
        hidden = _hidden(train_config, seed=2)
        with torch.no_grad(), InferenceMode.active():
            stale_gen_out, _ = gen_layer(hidden)
        with torch.no_grad():
            train_layer(hidden)

        with torch.no_grad():
            for layer in (gen_layer, train_layer):
                for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
                    for i in range(layer.experts.num_local_experts):
                        weight = getattr(linear, f"weight{i}")
                        weight.mul_(1.5)
        assert gen_layer.experts.refresh_flashinfer_nvfp4_weights()

        with torch.no_grad(), InferenceMode.active():
            gen_out, _ = gen_layer(hidden)
        with torch.no_grad():
            train_out, _ = train_layer(hidden)

        assert _elements_that_differ(gen_out, stale_gen_out) > 0, "the update moved nothing"
        differing = _elements_that_differ(train_out, gen_out)
        assert differing == 0, (
            f"{recipe}: after a weight update, {differing}/{train_out.numel()} elements "
            "differ from refreshed generation; the value pass read stale kernel weights"
        )


class TestTokenCountParity:
    """Does a token's output depend on how many tokens shared the launch?

    In RL the counts never match: generation decodes a few tokens per step and the
    training log-prob pass takes a whole microbatch. Two comparisons, so a failure
    says which side moved:

    * generation wide vs generation chunked: the kernel alone.
    * training wide vs generation chunked: the production case.
    """

    @pytest.mark.parametrize("recipe", list(RECIPES))
    def test_training_forward_matches_chunked_generation(self, recipe):
        _skip_unless_supported(recipe)
        gen_config, train_config = _generation_and_training_configs(recipe)
        # The symmetric heap is sized once, for the widest forward either side will
        # make; the training pass reuses generation's rather than growing it.
        gen_layer, train_layer = _layer_pair(
            recipe, gen_config, train_config, max_tokens=max(TRAIN_TOKEN_COUNTS)
        )

        results = []
        for count in TRAIN_TOKEN_COUNTS:
            hidden = _hidden(train_config, seed=count, tokens=count)
            with torch.no_grad(), _fp8_context(gen_config), InferenceMode.active():
                gen_wide, _ = gen_layer(hidden)
                gen_chunked = torch.cat(
                    [
                        gen_layer(hidden[start : start + GEN_TOKENS])[0]
                        for start in range(0, count, GEN_TOKENS)
                    ],
                    dim=0,
                )
            with torch.no_grad(), _fp8_context(train_config):
                train_wide, _ = train_layer(hidden)
            results.append(
                (
                    count,
                    _elements_that_differ(gen_wide, gen_chunked),
                    _elements_that_differ(train_wide, gen_chunked),
                )
            )

        # Asserted after every count has run, so one failure does not hide the
        # numbers for the others.
        for count, kernel_differs, production_differs in results:
            assert kernel_differs == 0, (
                f"{recipe}: generation output depends on batch size at {count} vs "
                f"{GEN_TOKENS} tokens ({kernel_differs} elements differ). The kernel is not "
                "batch-invariant, independent of the training path."
            )
            assert production_differs == 0, (
                f"{recipe}: training at {count} tokens differs from generation at "
                f"{GEN_TOKENS} ({production_differs} elements differ) although the kernel "
                "itself is batch-invariant."
            )


class TestSymmetricHeapSizing:
    """(3) The NVLS symmetric heap is sized for the largest pass, not the first.

    Symmetric memory is allocated once and cannot grow. A log-prob microbatch is
    padded to its own longest sequence, so a run can see 512 tokens and later 576;
    sizing from the first would then fail on the second.
    """

    SMALL, LARGE = LOCAL_TOKENS, 2 * LOCAL_TOKENS
    RECIPE = "nvfp4-squared_relu"

    def _train_layer(self, **overrides):
        _skip_unless_supported(self.RECIPE)
        if _dispatcher() != "nvls":
            pytest.skip("the symmetric heap is only used by the NVLS dispatcher (EP > 1)")
        config = _config(self.RECIPE, moe_inference_training_forward=True, **overrides)
        return config, _build_layer(config).eval()

    def test_a_configured_bound_covers_a_larger_later_pass(self):
        config, layer = self._train_layer(moe_inference_training_max_tokens_per_rank=self.LARGE)
        with torch.no_grad():
            layer(_hidden(config, seed=0, tokens=self.SMALL))
            out, _ = layer(_hidden(config, seed=1, tokens=self.LARGE))
        assert torch.isfinite(out).all()

    def test_without_a_bound_a_larger_later_pass_fails_loudly(self):
        """A named assertion on every rank, rather than a hang or a bad answer."""
        config, layer = self._train_layer()
        with torch.no_grad():
            layer(_hidden(config, seed=0, tokens=self.SMALL))
            with pytest.raises(AssertionError, match="moe_inference_training_max_tokens_per_rank"):
                layer(_hidden(config, seed=1, tokens=self.LARGE))

    def test_the_bound_is_a_floor_not_a_cap(self):
        """A first pass larger than the bound still gets buffers that fit it."""
        config, layer = self._train_layer(moe_inference_training_max_tokens_per_rank=self.SMALL)
        with torch.no_grad():
            out, _ = layer(_hidden(config, seed=0, tokens=self.LARGE))
        assert torch.isfinite(out).all()
