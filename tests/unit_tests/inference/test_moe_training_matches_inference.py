# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for ``moe_training_matches_inference``.

``moe_training_matches_inference`` runs the training MoE forward through the
FlashInfer routed MXFP8 or per-token NVFP4 kernel generation uses, and takes its
backward from a TE recompute pass. Those kernels are batch invariant but do not
match TE training, so without this a training log-prob and a generation log-prob of
the same token differ, and the importance ratio in RL carries that bias. The tests
cover:

1. Construction-time validation and generation's route order (CPU, no GPU): the
   checks that fail at ``TransformerConfig`` construction rather than in arithmetic,
   because a rejected setting found ten minutes into a two-node job is the failure
   they exist to move to the first second.
2. The path runs, forward and backward, and the value pass really reaches the
   FlashInfer kernel rather than silently falling back to TE (which still gives
   finite outputs and gradients).
3. The value pass keeps reproducing the generation forward bitwise after the weights
   change, and independently of how many tokens share the launch.
4. A value pass larger than the NVLS symmetric heap, which cannot grow, is split
   into heap-sized launches with unchanged output; and without generation in its
   process, as when the two run apart, the value pass allocates its own heap and
   still matches generation.
5. The value pass widens the forward/backward mismatch of neither recipe it applies
   to, TE MXFP8 forward and backward or NVFP4 forward with a BF16 backward, beyond
   the mismatch the default TE recipe already accepts.

The GPU tests run under batch-invariant mode, which the flag requires, on any GPU
count that is a power of two and divides the expert count; the expert-parallel size
is the world size, with the NVLS dispatcher when it is larger than one.
"""

import contextlib
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core import fp4_utils
from megatron.core.activations import squared_relu
from megatron.core.fp8_utils import dequantize_fp8_tensor, is_float8tensor
from megatron.core.inference.moe.flashinfer_mxfp8 import HAVE_FLASHINFER_ROUTED_MXFP8
from megatron.core.inference.moe.flashinfer_nvfp4 import HAVE_FLASHINFER_ROUTED_NVFP4
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.moe_module_specs import (
    get_inference_optimized_moe_spec,
    get_moe_module_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
    set_batch_invariant_mode,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe import training_matches_inference
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
# Per-rank NVLS heap for MXFP8 tests that would otherwise size it to LOCAL_TOKENS.
HEAP_TOKENS = 512
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

_MXFP8 = dict(fp8="hybrid", fp8_recipe="mxfp8", fp8_param=True)


def _on_flashinfer_gpu(*fixtures):
    """Marks for a GPU test class: GB200 launch, SM100/SM103, and its fixtures."""
    marks = (
        pytest.mark.internal,
        pytest.mark.launch_on_gb200,
        pytest.mark.skipif(
            not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
            reason="FlashInfer routed MXFP8 and per-token NVFP4 MoE need SM100/SM103",
        ),
        pytest.mark.usefixtures(*fixtures),
    )

    def apply(cls):
        for mark in marks:
            cls = mark(cls)
        return cls

    return apply


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
        _config(recipe, **overrides, moe_training_matches_inference=True),
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
    if for_inference:
        _allocate_dispatch_buffers(config, max_tokens)
    return get_inference_optimized_moe_spec()(config=config).cuda()


def _allocate_dispatch_buffers(config, max_tokens):
    """The dispatcher buffers a generation context allocates, ``max_tokens`` per rank."""
    from megatron.core import parallel_state

    NCCLAllGatherDispatcher.allocate_buffers()
    if config.inference_moe_token_dispatcher_type == "nvls":
        NVLSAllGatherVDispatcher.allocate_buffers(
            per_rank_worst_case_token_count=max_tokens,
            topk=config.moe_router_topk,
            hidden_size=config.moe_latent_size or config.hidden_size,
            tp_ep_group=parallel_state.get_expert_model_parallel_group(),
        )


def _free_dispatch_buffers():
    """Drop the process-wide dispatcher buffers, as in a process without generation."""
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    InferenceAllGatherDispatcherBase._valid_tokens_tensor = None
    NVLSAllGatherVDispatcher._delete_buffers()
    SymmetricMemoryManager.destroy()


def _fp8_context(config, is_init=False):
    """TE's MXFP8 context for an MXFP8 config, a no-op otherwise."""
    from megatron.core.fp8_utils import get_fp8_context

    if not config.fp8:
        return contextlib.nullcontext()
    return get_fp8_context(config, 0, is_init=is_init)


def _layer_pair(recipe, gen_config, train_config, max_tokens=LOCAL_TOKENS):
    """Generation and training layers holding the same weights, each in its own form.

    Generation converts as production does: MXFP8 replaces TE's MXFP8 parameters
    with MCore ``MXFP8Tensor``s at load, NVFP4 quantizes the BF16 stacks on the
    first forward. Training keeps its parameters and the value pass derives the
    kernel weights from them once per optimizer step.
    """
    from megatron.core.inference.quantization.utils import quantize_model_to_mxfp8

    with _fp8_context(gen_config, is_init=True):
        gen_layer = _build_layer(gen_config, for_inference=True, max_tokens=max_tokens).eval()
    with _fp8_context(train_config, is_init=True):
        train_layer = _build_layer(train_config).eval()
    train_layer.load_state_dict(gen_layer.state_dict())
    if recipe == "mxfp8":
        assert training_matches_inference.expert_params_use_te_mxfp8(train_layer.experts)
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


def _optimizer_step():
    """An optimizer step, which is what tells the value pass the parameters moved."""
    torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0).step()


@pytest.fixture
def spy_on_value_pass(monkeypatch):
    """Record the routing map each value pass of an experts module is handed."""
    original = training_matches_inference.expert_value_pass
    seen = {}

    def spy(experts, hidden_states, probs, routing_map):
        seen.setdefault(id(experts), []).append(routing_map)
        return original(experts, hidden_states, probs, routing_map)

    monkeypatch.setattr(training_matches_inference, "expert_value_pass", spy)
    return lambda experts: seen.setdefault(id(experts), [])


@pytest.fixture
def _parallel_state(monkeypatch):
    from megatron.core.inference import utils as inference_utils

    # The routed MXFP8 kernel JIT-compiles without flashinfer-jit-cache; the layer's
    # check only guards production against the compile time.
    monkeypatch.setattr(inference_utils, "check_flashinfer_jit_cache_installed", lambda **_: None)

    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
    # Registers the 'expert-parallel-rng' tracker state that expert weight init
    # requires. Utils.initialize_model_parallel does not do this.
    model_parallel_cuda_manual_seed(123)

    # Process-wide state. Leaked, one test's inference setup would mask a missing
    # allocation in the next, which is exactly the check the NVLS arm has to make.
    _free_dispatch_buffers()
    yield
    _free_dispatch_buffers()
    Utils.destroy_model_parallel()


@pytest.fixture
def _batch_invariant(_parallel_state):
    # Global kernel patching: on before any layer is built, off again afterwards,
    # or it would silently change the next test.
    enable_batch_invariant_mode(backend="te_native")
    yield
    disable_batch_invariant_mode()


# ---------------------------------------------------------------------------
# (1) Construction-time validation and generation's route order (CPU).
# ---------------------------------------------------------------------------


def _validated_config(**overrides):
    """A valid per-token NVFP4 parity config, which each test then breaks one way."""
    kwargs = dict(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_grouped_gemm=True,
        add_bias_linear=False,
        gated_linear_unit=False,
        activation_func=squared_relu,
        transformer_impl="inference_optimized",
        # Required by inference_optimized: a BF16 router would cost a per-decode
        # dtype conversion, and its expert module assumes RMSNorm. Unrelated to
        # what is tested here, but without them every test dies before reaching
        # its assertion.
        moe_router_dtype="fp32",
        normalization="RMSNorm",
        inference_grouped_gemm_backend="flashinfer",
        inference_moe_nvfp4_recipe="per_token",
        attention_backend=AttnBackend.flash,
        flash_attention_version=4,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        batch_invariant_backend="te_native",
        moe_training_matches_inference=True,
        # The value pass stores no activations, so the backward has to come from
        # a recompute pass. Validated, not assumed.
        recompute_granularity="selective",
        recompute_modules=["moe"],
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestBackendAllowList:
    @pytest.mark.parametrize("recipe", ["nvfp4", "mxfp8"])
    def test_flashinfer_quantized_recipes_are_accepted(self, recipe):
        """MXFP8 needs no extra opt-in: its TE recompute is MXFP8 too."""
        overrides = dict(inference_moe_nvfp4_recipe=None, **_MXFP8) if recipe == "mxfp8" else {}
        assert _validated_config(**overrides).moe_training_matches_inference

    @pytest.mark.parametrize("backend", ["vllm", "torch"])
    def test_backends_that_already_match_training_are_rejected(self, backend):
        """Under batch_invariant_mode their kernels already match TE training."""
        with pytest.raises(ValueError, match="supports only inference_grouped_gemm_backend"):
            _validated_config(
                inference_grouped_gemm_backend=backend, inference_moe_nvfp4_recipe=None
            )

    def test_flashinfer_bf16_is_rejected(self):
        """The FlashInfer BF16 kernel is not batch invariant, so nothing could match it."""
        # batch_invariant_mode rejects it first, with its own message naming the recipes.
        with pytest.raises(ValueError, match="inference_moe_nvfp4_recipe"):
            _validated_config(inference_moe_nvfp4_recipe=None)

    def test_batch_invariant_mode_is_required(self):
        """Generation and the value pass launch the kernel on different token counts."""
        with pytest.raises(ValueError, match="requires batch_invariant_mode"):
            _validated_config(batch_invariant_mode=False)

    def test_requires_the_inference_optimized_transformer_impl(self):
        """Only that spec builds the inference-capable expert module."""
        with pytest.raises(ValueError, match="inference_optimized"):
            _validated_config(transformer_impl="transformer_engine")


class TestRecomputeRequirement:
    def test_recompute_is_required(self):
        """Without it there is no backward graph for the experts at all."""
        with pytest.raises(ValueError, match="recompute"):
            _validated_config(recompute_granularity=None, recompute_modules=None)

    @pytest.mark.parametrize("modules", [None, [], ["layernorm"]])
    def test_moe_must_be_among_the_recomputed_modules(self, modules):
        """None becomes ["core_attn"] later in construction, so it does not recompute MoE."""
        with pytest.raises(ValueError, match="recompute"):
            _validated_config(recompute_modules=modules)

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(recompute_modules=["moe", "layernorm"]),
            dict(moe_layer_recompute=True, recompute_granularity=None, recompute_modules=None),
        ],
    )
    def test_moe_recompute_is_accepted_in_every_supported_form(self, overrides):
        config = _validated_config(**overrides)
        assert config.recompute_granularity == "selective"
        assert "moe" in config.recompute_modules

    def test_local_cuda_graphs_are_rejected(self):
        """They disable MoE-layer recompute, leaving the value pass with no backward."""
        with pytest.raises(ValueError, match="cuda_graph_impl"):
            _validated_config(cuda_graph_impl="local")


class TestParityBreakingSettings:
    """Settings that make the training router pick different experts than generation.

    Each is rejected rather than tolerated, because the loss of parity they cause
    is otherwise silent: the layer runs, the loss moves, and only the KL shows it.
    """

    def test_overlapped_schedule_is_rejected(self):
        """It calls the layer's stages directly and never reaches the pass switch."""
        with pytest.raises(ValueError, match="overlap_moe_expert_parallel_comm"):
            _validated_config(overlap_moe_expert_parallel_comm=True)

    @pytest.mark.parametrize("balancing", ["sinkhorn", "quantile_balancing"])
    def test_router_balancing_that_changes_selection_is_rejected(self, balancing):
        with pytest.raises(ValueError, match="moe_router_load_balancing_type"):
            _validated_config(moe_router_load_balancing_type=balancing)

    def test_input_jitter_is_rejected(self):
        with pytest.raises(ValueError, match="moe_input_jitter_eps"):
            _validated_config(moe_input_jitter_eps=0.01)

    def test_forced_load_balancing_is_rejected(self):
        with pytest.raises(ValueError, match="moe_router_force_load_balancing"):
            _validated_config(moe_router_force_load_balancing=True)

    def test_expert_tensor_parallelism_is_rejected(self):
        with pytest.raises(ValueError, match="expert tensor parallelism"):
            _validated_config(
                expert_tensor_parallel_size=2, inference_moe_token_dispatcher_type="nvls"
            )


class TestGenerationRouteOrder:
    """The value pass routes through generation's router path.

    The kernel sums a token's routes in the order given, so the value pass must hand
    it generation's top-k order, not one recovered from the training router's map.
    """

    def test_value_pass_keeps_generation_top_k_order(self):
        from types import SimpleNamespace

        from megatron.core.transformer.moe.training_matches_inference import _route

        probs = torch.tensor([[0.5, 0.1, 0.3]])
        ids = torch.tensor([[5, 1, 3]])
        seen = {}

        def generation_route(hidden_states, input_ids=None):
            seen["shape"], seen["input_ids"] = hidden_states.shape, input_ids
            return probs, ids

        stub = SimpleNamespace(route=generation_route)
        input_ids = torch.tensor([[7, 8], [9, 10], [11, 12]])
        out_probs, out_ids = _route(stub, torch.zeros(2, 3, 4), input_ids)

        assert out_ids is ids and out_probs is probs
        assert seen["shape"] == (6, 1, 4) and seen["input_ids"] is input_ids

    def test_generation_router_path_under_inference_mode(self):
        """Under InferenceMode the router skips the training path and returns generation's."""
        from types import SimpleNamespace

        from megatron.core.transformer.moe.router import InferenceTopKRouter

        stub = SimpleNamespace(_forward=lambda *args: "generation")
        with InferenceMode.active():
            assert InferenceTopKRouter.forward(stub, torch.zeros(6, 1, 4)) == "generation"


# ---------------------------------------------------------------------------
# (2)-(4) The value pass on GPU.
# ---------------------------------------------------------------------------


@_on_flashinfer_gpu("_batch_invariant")
class TestValuePassRuns:
    """(2) The path runs, and the value pass reaches the FlashInfer kernel."""

    @pytest.mark.parametrize("mode", ["train", "eval"])
    @pytest.mark.parametrize("recipe", ["nvfp4-squared_relu", "nvfp4-swiglu"])
    def test_forward_backward_runs(self, recipe, mode, spy_on_value_pass):
        """NVFP4 keeps BF16 parameters, so the TE recompute gives every weight a gradient.

        ``eval()`` disables training behavior, not autograd: a grad-enabled eval
        forward still needs the recompute for its backward.
        """
        _skip_unless_supported(recipe)
        config = _config(recipe, moe_training_matches_inference=True)
        layer = _build_layer(config).train(mode == "train")
        seen = spy_on_value_pass(layer.experts)
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
    def test_eval_forward_takes_the_flashinfer_kernel(self, recipe, spy_on_value_pass):
        """The log-prob pass runs under ``eval()`` and must not fall back to TE.

        RL frameworks compute the training log-prob under ``model.eval()``, so a
        gate on ``self.training`` would send exactly the pass that has to match
        generation down the TE path.
        """
        _skip_unless_supported(recipe)
        gen_config, train_config = _generation_and_training_configs(recipe)
        # The routed MXFP8 kernel runs on the whole heap and has no tile config for
        # one sized to a few tokens per rank.
        _, layer = _layer_pair(recipe, gen_config, train_config, max_tokens=HEAP_TOKENS)
        seen = spy_on_value_pass(layer.experts)
        with torch.no_grad(), _fp8_context(train_config):
            layer(_hidden(train_config, seed=0))
        assert len(seen) == 1, "the eval forward did not take the FlashInfer kernel"

    @pytest.mark.parametrize("recipe", list(RECIPES))
    def test_kernel_weights_are_built_once_per_optimizer_step(self, recipe, monkeypatch):
        """Passes between optimizer steps reuse the kernel weights; a step drops them.

        Each build first runs the expert linears' pre-forward hooks, which is where DDP
        finishes an overlapped parameter all-gather before the weights are read.
        """
        _skip_unless_supported(recipe)
        gen_config, train_config = _generation_and_training_configs(recipe)
        _, layer = _layer_pair(recipe, gen_config, train_config, max_tokens=HEAP_TOKENS)
        experts = layer.experts
        builds, hook_calls = [], []
        build = training_matches_inference._build_value_pass_weights
        monkeypatch.setattr(
            training_matches_inference,
            "_build_value_pass_weights",
            lambda module: builds.append(1) or build(module),
        )
        experts.linear_fc1.register_forward_pre_hook(lambda *_: hook_calls.append(1))
        hidden = _hidden(train_config, seed=0)
        with torch.no_grad(), _fp8_context(train_config):
            _optimizer_step()
            layer(hidden)
            layer(hidden)
            _optimizer_step()
            layer(hidden)
        assert len(builds) == 2
        assert len(hook_calls) == 2

    @pytest.mark.parametrize("recipe", list(RECIPES))
    def test_release_frees_the_kernel_weights_and_the_next_pass_rebuilds_them(self, recipe):
        """A caller can free the kernel weights between optimizer steps.

        The next value pass rebuilds them and matches the pass before the release.
        """
        _skip_unless_supported(recipe)
        gen_config, train_config = _generation_and_training_configs(recipe)
        _, layer = _layer_pair(recipe, gen_config, train_config, max_tokens=HEAP_TOKENS)
        hidden = _hidden(train_config, seed=0)
        with torch.no_grad(), _fp8_context(train_config):
            before = layer(hidden)[0].clone()
            assert layer.experts in training_matches_inference._value_pass_weights
            training_matches_inference.release_inference_value_pass_weights(layer)
            assert layer.experts not in training_matches_inference._value_pass_weights
            after = layer(hidden)[0]
        assert layer.experts in training_matches_inference._value_pass_weights
        assert torch.equal(before, after)


@_on_flashinfer_gpu("_batch_invariant")
class TestTrainGenParity:
    """(3) The value pass keeps reproducing the generation forward bitwise.

    Run under ``eval()`` and ``no_grad``, which is the log-prob pass RL compares.

    Not covered: the refit. Both layers start from the same parameters, whereas in
    RL generation's weights arrive as BF16 slices and are quantized on the receiver.
    """

    @pytest.mark.parametrize("recipe", ["nvfp4-squared_relu", "nvfp4-swiglu"])
    def test_value_pass_follows_a_weight_update(self, recipe):
        """After the parameters move, the value pass matches a refreshed generation.

        The update is applied to both layers' BF16 parameters, and generation
        requantizes as a refit would. The value pass keeps its kernel weights until
        the next optimizer step and rebuilds them after it.
        """
        _skip_unless_supported(recipe)
        gen_config, train_config = _generation_and_training_configs(recipe)
        gen_layer, train_layer = _layer_pair(recipe, gen_config, train_config)
        hidden = _hidden(train_config, seed=2)
        with torch.no_grad(), InferenceMode.active():
            stale_gen_out, _ = gen_layer(hidden)
        with torch.no_grad():
            stale_train_out, _ = train_layer(hidden)

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
            same_step_out, _ = train_layer(hidden)
            _optimizer_step()
            train_out, _ = train_layer(hidden)

        assert _elements_that_differ(gen_out, stale_gen_out) > 0, "the update moved nothing"
        assert (
            _elements_that_differ(same_step_out, stale_train_out) == 0
        ), "the value pass rebuilt its kernel weights without an optimizer step"
        differing = _elements_that_differ(train_out, gen_out)
        assert differing == 0, (
            f"{recipe}: after a weight update, {differing}/{train_out.numel()} elements "
            "differ from refreshed generation; the value pass read stale kernel weights"
        )


@_on_flashinfer_gpu("_batch_invariant")
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
        # Generation's wide pass needs a heap that fits it; training reuses it.
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


@_on_flashinfer_gpu("_batch_invariant")
class TestSymmetricHeapSizing:
    """(4) A value pass larger than the NVLS symmetric heap is split to fit it.

    The heap is allocated once and cannot grow, so a pass larger than generation's
    runs in heap-sized launches as generation does.
    """

    SMALL, LARGE = LOCAL_TOKENS, 3 * LOCAL_TOKENS
    RECIPE = "nvfp4-squared_relu"

    def test_a_larger_pass_matches_heap_sized_passes(self):
        _skip_unless_supported(self.RECIPE)
        if _dispatcher() != "nvls":
            pytest.skip("the symmetric heap is only used by the NVLS dispatcher (EP > 1)")
        config = _config(self.RECIPE, moe_training_matches_inference=True)
        layer = _build_layer(config, for_inference=True, max_tokens=self.SMALL).eval()
        hidden = _hidden(config, seed=1, tokens=self.LARGE)
        with torch.no_grad():
            chunked = torch.cat(
                [
                    layer(hidden[start : start + self.SMALL])[0]
                    for start in range(0, self.LARGE, self.SMALL)
                ],
                dim=0,
            )
            assert NVLSAllGatherVDispatcher._per_rank_worst_case_token_count == self.SMALL
            wide, _ = layer(hidden)
        assert _elements_that_differ(wide, chunked) == 0

    def _heap_sized_layer(self, recipe=RECIPE, max_tokens=SMALL):
        _skip_unless_supported(recipe)
        if _dispatcher() != "nvls":
            pytest.skip("the symmetric heap is only used by the NVLS dispatcher (EP > 1)")
        config = _config(recipe, moe_training_matches_inference=True)
        with _fp8_context(config, is_init=True):
            layer = _build_layer(config, for_inference=True, max_tokens=max_tokens).eval()
        return config, layer

    def test_each_launch_covers_only_rows_that_hold_tokens(self, spy_on_value_pass):
        """The kernel's cost follows the microbatch, not the generation-sized heap."""
        config, layer = self._heap_sized_layer()
        seen = spy_on_value_pass(layer.experts)
        group_size = torch.distributed.get_world_size(layer._inference_token_dispatcher.tp_ep_group)
        with torch.no_grad():
            layer(_hidden(config, seed=3, tokens=self.SMALL + self.SMALL // 2))
        assert [routing_map.shape[0] for routing_map in seen] == [
            group_size * self.SMALL,
            group_size * (self.SMALL // 2),
        ]

    def test_mxfp8_launches_keep_the_largest_kernel_tile(self, spy_on_value_pass):
        """MXFP8 launches follow the microbatch down to the kernel's largest-tile floor.

        The routed MXFP8 kernel has no tile config for a launch of a few tokens per
        rank at these shapes, so a small microbatch launches on the floor's rows.
        """
        config, layer = self._heap_sized_layer("mxfp8", max_tokens=HEAP_TOKENS)
        seen = spy_on_value_pass(layer.experts)
        group_size = torch.distributed.get_world_size(layer._inference_token_dispatcher.tp_ep_group)
        floor = 128 * layer.num_local_experts // config.moe_router_topk + 1
        assert group_size * LOCAL_TOKENS < floor < group_size * (HEAP_TOKENS // 2)
        with torch.no_grad(), _fp8_context(config):
            layer(_hidden(config, seed=3, tokens=LOCAL_TOKENS))
            layer(_hidden(config, seed=3, tokens=HEAP_TOKENS // 2))
        assert [routing_map.shape[0] for routing_map in seen] == [
            floor,
            group_size * (HEAP_TOKENS // 2),
        ]

    def test_a_layer_that_skips_the_metadata_sync_matches_one_that_runs_it(self):
        """Generation refreshes dispatch metadata on its first MoE layer only.

        A one-token tail chunk leaves that layer's metadata stale for the next pass,
        so every launch of the value pass must refresh it.
        """
        config, layer = self._heap_sized_layer()
        hidden = _hidden(config, seed=1, tokens=self.LARGE + 1)
        dispatcher = layer._inference_token_dispatcher
        with torch.no_grad():
            want, _ = layer(hidden)
            dispatcher._runs_metadata_sync = False
            got, _ = layer(hidden)
        assert _elements_that_differ(got, want) == 0

    @pytest.mark.parametrize("real_tokens", [0, LOCAL_TOKENS // 2])
    def test_generation_padding_count_masks_no_training_row(self, monkeypatch, real_tokens):
        """A colocated context binds its real-token count to the NVLS dispatcher class."""
        config, layer = self._heap_sized_layer()
        hidden = _hidden(config, seed=2, tokens=self.LARGE)
        with torch.no_grad():
            want, _ = layer(hidden)
            monkeypatch.setattr(
                NVLSAllGatherVDispatcher,
                "_real_token_count_tensor",
                torch.tensor([real_tokens], dtype=torch.int32, device="cuda"),
            )
            got, _ = layer(hidden)
        assert _elements_that_differ(got, want) == 0


@_on_flashinfer_gpu("_batch_invariant")
class TestSeparateGeneration:
    """(4) Training without generation in its process, as when the two run apart.

    Generation runs, its buffers are freed, and the value pass allocates its own:
    either sized like generation's, as a trainer does for a generation that sets
    ``max_tokens``, or by default. The buffer size is the row count of the FlashInfer
    launch, so the default case also checks that it does not change the output.
    """

    TOKENS = 64

    @pytest.mark.parametrize("training_heap", ["generation", "default"])
    @pytest.mark.parametrize("recipe", list(RECIPES))
    def test_value_pass_with_its_own_buffers_matches_generation(self, recipe, training_heap):
        _skip_unless_supported(recipe)
        if _dispatcher() != "nvls":
            pytest.skip("only the NVLS dispatcher has sized buffers (EP > 1)")
        gen_config, train_config = _generation_and_training_configs(recipe)
        gen_layer, train_layer = _layer_pair(
            recipe, gen_config, train_config, max_tokens=HEAP_TOKENS
        )
        hidden = _hidden(train_config, seed=3, tokens=self.TOKENS)
        with torch.no_grad(), _fp8_context(gen_config), InferenceMode.active():
            gen_out, _ = gen_layer(hidden)

        _free_dispatch_buffers()
        if training_heap == "generation":
            _allocate_dispatch_buffers(train_config, HEAP_TOKENS)
        with torch.no_grad(), _fp8_context(train_config):
            train_out, _ = train_layer(hidden)

        heap = NVLSAllGatherVDispatcher._per_rank_worst_case_token_count
        assert (heap == HEAP_TOKENS) == (training_heap == "generation")
        differing = _elements_that_differ(train_out, gen_out)
        assert differing == 0, (
            f"{recipe}: with its own {heap}-token heap against generation's "
            f"{HEAP_TOKENS}, {differing}/{train_out.numel()} elements differ"
        )

    def test_buffers_of_another_expert_parallel_size_are_rejected(self, monkeypatch):
        recipe = "nvfp4-squared_relu"
        _skip_unless_supported(recipe)
        if _dispatcher() != "nvls":
            pytest.skip("only the NVLS dispatcher has sized buffers (EP > 1)")
        config = _config(recipe, moe_training_matches_inference=True)
        layer = _build_layer(config, for_inference=True, max_tokens=HEAP_TOKENS).eval()
        # The same rows split over half the ranks: buffers of a generation at EP / 2.
        monkeypatch.setattr(
            NVLSAllGatherVDispatcher, "_per_rank_worst_case_token_count", 2 * HEAP_TOKENS
        )
        with torch.no_grad(), pytest.raises(RuntimeError, match="expert-parallel size"):
            layer(_hidden(config, seed=4))


# ---------------------------------------------------------------------------
# (5) Forward/backward mismatch against the default TE recipe.
# ---------------------------------------------------------------------------

MISMATCH_TOKENS = 512
MISMATCH_SEEDS = (11, 12, 13)
# Fraction of tokens scaled up, and by how much, in the outlier inputs, where the
# arms' quantization is most likely to part ways.
OUTLIER_FRACTION = 0.05
OUTLIER_SCALE = 30.0
# The value pass's forward may be this much farther from BF16 than the default
# recipe's. Against TE per-token NVFP4 the two land within 0.5% of each other,
# either side; 1% stays well under the ~20% both are from BF16.
ALPHA = 0.01


def _te_config(recipe, **quantization):
    """TE training config with the value-pass config's shape and activation."""
    activation = {
        key: value
        for key, value in RECIPES[recipe].items()
        if key in ("activation_func", "gated_linear_unit")
    }
    return TransformerConfig(
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
        activation_func=activation.get("activation_func", squared_relu),
        gated_linear_unit=activation.get("gated_linear_unit", False),
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="transformer_engine",
        expert_model_parallel_size=Utils.world_size,
        **quantization,
    )


def _te_layer(config):
    return (
        get_moe_module_spec(use_te=True, num_experts=config.num_moe_experts, moe_grouped_gemm=True)(
            config=config
        )
        .cuda()
        .eval()
    )


def _copy_parameters(source, target):
    """Copy by name, dequantizing TE MXFP8 parameters into a BF16 target."""
    params = dict(source.named_parameters())
    with torch.no_grad():
        for name, param in target.named_parameters():
            assert name in params, f"source layer has no parameter {name}"
            value = params[name]
            param.copy_(dequantize_fp8_tensor(value) if is_float8tensor(value) else value)


def _all_sum(value: float) -> float:
    tensor = torch.tensor([value], device="cuda", dtype=torch.float64)
    torch.distributed.all_reduce(tensor)
    return tensor.item()


def _relative_error(output, reference) -> float:
    diff = _all_sum((output.double() - reference.double()).pow(2).sum().item())
    return (diff / _all_sum(reference.double().pow(2).sum().item())) ** 0.5


def _worst_token_error(output, reference) -> float:
    """Largest per-token relative error, over all ranks."""
    per_token = (output.float() - reference.float()).norm(dim=-1) / reference.float().norm(
        dim=-1
    ).clamp_min(1e-30)
    worst = per_token.max().reshape(1).double()
    torch.distributed.all_reduce(worst, op=torch.distributed.ReduceOp.MAX)
    return worst.item()


def _mismatch_tokens(hidden_size, seed, outliers):
    generator = torch.Generator(device="cuda").manual_seed(seed * 1009 + Utils.rank)
    hidden = torch.randn(
        MISMATCH_TOKENS, 1, hidden_size, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    if outliers:
        count = max(1, int(OUTLIER_FRACTION * MISMATCH_TOKENS))
        rows = torch.randperm(MISMATCH_TOKENS, device="cuda", generator=generator)[:count]
        hidden[rows] *= OUTLIER_SCALE
    return hidden


def _te_per_token_nvfp4_recipe():
    """TE's per-token NVFP4 with a BF16 backward: the training form of FlashInfer's recipe.

    Row-scaled activations and 1D weight blocks, without the random Hadamard transform
    or stochastic rounding, as in ``test_batch_invariant_nvfp4_moe_matches_training``.
    TE's default NVFP4 scales activations per tensor and weights in 2D blocks, so
    against it the comparison would measure the scaling scheme, not the kernel.
    """
    recipe_module = pytest.importorskip("transformer_engine.common.recipe")
    fields = getattr(recipe_module.NVFP4BlockScaling, "__dataclass_fields__", {})
    if not {"row_scaled_activation", "backward_override"} <= fields.keys():
        pytest.skip("Requires Transformer Engine with row-scaled NVFP4 and backward_override.")
    return recipe_module.NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        row_scaled_activation=True,
        backward_override="high_precision",
    )


def _default_recipe_forward(recipe, value_pass_config, value_pass):
    """The default TE recipe's forward on the value pass's weights, as a callable.

    Both are plain TE layers: the inference-optimized layer rejects MXFP8 in
    training unless the flag is on. MXFP8: TE MXFP8, which is what the value pass's
    recompute runs, under batch-invariant mode as the recompute is. NVFP4: TE
    per-token NVFP4 with ``backward_override='high_precision'``, a BF16 backward
    like the value pass's recompute. Batch-invariant mode rejects TE ``fp4``, so
    that layer runs outside it.
    """
    invariant = value_pass_config.batch_invariant_backend
    if recipe == "mxfp8":
        # The config flag, not just the runtime mode, makes the AllToAll dispatcher
        # build the inverse map the batch-invariant unpermute needs.
        te_config = _te_config(
            recipe,
            **_MXFP8,
            batch_invariant_mode=True,
            batch_invariant_backend=invariant,
            attention_backend=AttnBackend.flash,
            flash_attention_version=4,
            attention_dropout=0.0,
        )
        with _fp8_context(te_config, is_init=True):
            te = _te_layer(te_config)
        te.load_state_dict(value_pass.state_dict())

        def forward(hidden):
            with set_batch_invariant_mode(True, backend=invariant), _fp8_context(te_config):
                return te(hidden)[0]

        return forward

    te_recipe = _te_per_token_nvfp4_recipe()
    te_config = _te_config(recipe, fp4="e2m1", fp4_recipe="nvfp4")
    te = _te_layer(te_config)
    _copy_parameters(value_pass, te)

    def forward(hidden):
        with (
            mock.patch.object(fp4_utils, "get_fp4_recipe", lambda config: te_recipe),
            fp4_utils.get_fp4_context(te_config, 0),
        ):
            return te(hidden)[0]

    return forward


def measure_forward_mismatch(recipe, seeds=MISMATCH_SEEDS):
    """``[(seed, outliers, {arm: (relative error, worst token error)})]`` against BF16.

    The arms are the value pass and the default TE recipe's forward, on the same
    weights. The reference is a TE BF16 forward on those weights, dequantized for
    MXFP8: the layer whose gradient both recipes' backward computes, since it ignores
    the forward's quantization.
    """
    value_pass_config = _config(recipe, moe_training_matches_inference=True)
    torch.manual_seed(2028)
    torch.cuda.manual_seed(2028)
    with _fp8_context(value_pass_config, is_init=True):
        value_pass = _build_layer(value_pass_config).eval()
    default = _default_recipe_forward(recipe, value_pass_config, value_pass)
    bf16 = _te_layer(_te_config(recipe))
    _copy_parameters(value_pass, bf16)
    invariant = value_pass_config.batch_invariant_backend

    results = []
    for seed in seeds:
        for outliers in (False, True):
            hidden = _mismatch_tokens(value_pass_config.hidden_size, seed, outliers)
            with torch.no_grad():
                with (
                    set_batch_invariant_mode(True, backend=invariant),
                    _fp8_context(value_pass_config),
                ):
                    value_pass_out, _ = value_pass(hidden)
                default_out = default(hidden)
                bf16_out, _ = bf16(hidden)
            assert not torch.equal(default_out, bf16_out), "the default recipe did not quantize"
            assert not torch.equal(value_pass_out, bf16_out), "the value pass did not run"
            results.append(
                (
                    seed,
                    outliers,
                    {
                        arm: (_relative_error(out, bf16_out), _worst_token_error(out, bf16_out))
                        for arm, out in (("value_pass", value_pass_out), ("te", default_out))
                    },
                )
            )
    return results


@_on_flashinfer_gpu("_parallel_state")
class TestForwardBackwardMismatch:
    """(5) Is the value pass's forward/backward mismatch within the default recipe's?

    The flag applies only to recipes whose backward is already not the derivative of
    their forward. Both ignore the forward's quantization and compute the BF16
    layer's gradient:

    * MXFP8: forward and backward in TE MXFP8. The value pass's recompute is that
      same TE MXFP8 backward, so only the forward changes.
    * NVFP4: the value pass's recompute runs in BF16, the exact derivative of the
      BF16 forward. TE per-token NVFP4 with ``backward_override='high_precision'``
      is also a BF16 backward, evaluated at its NVFP4 forward's intermediates.

    In both, the gap the backward bridges is how far the forward that produced the
    loss is from BF16. The value pass is acceptable if its forward is no farther:

        |FI - BF16| <= (1 + ALPHA) * |TE - BF16|

    on the MoE layer output, relative to |BF16|, for Gaussian tokens and for tokens
    with outlier scales. Only forwards run, so the comparison is deterministic.
    Batch-invariant mode stays off process-wide, since TE NVFP4 must run without it;
    the inference-optimized layers turn it on around their calls.
    """

    @pytest.mark.parametrize("recipe", list(RECIPES))
    def test_value_pass_mismatch_within_default_recipe(self, recipe):
        _skip_unless_supported(recipe)
        results = measure_forward_mismatch(recipe)

        def describe(errors):
            return ", ".join(
                f"{arm}: {relative:.4g} (worst token {worst:.4g})"
                for arm, (relative, worst) in errors.items()
            )

        if Utils.rank == 0:
            for seed, outliers, errors in results:
                kind = "outliers" if outliers else "gaussian"
                print(
                    f"\n{recipe} seed {seed} {kind}: relative error vs BF16 -- {describe(errors)}"
                )
        for seed, outliers, errors in results:
            bound = (1 + ALPHA) * errors["te"][0]
            assert errors["value_pass"][0] <= bound, (
                f"{recipe} seed {seed} {"outliers" if outliers else "gaussian"}: the value "
                f"pass's forward is farther from BF16 than the default TE recipe's by more "
                f"than {ALPHA:.0%} ({describe(errors)})"
            )
