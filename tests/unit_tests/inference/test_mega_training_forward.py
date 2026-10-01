# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed tests for the FlashInfer mega MoE training forward.

This is ``moe_inference_training_forward`` with
``inference_grouped_gemm_backend='flashinfer_mega'``; the backend-independent
behaviour of that mode is covered in ``test_inference_training_forward.py``. The
properties that matter, and what these tests cover:

1. The path runs, forward and backward, under expert parallelism, and the value
   pass really reaches the megakernel rather than silently falling back to TE.
2. The value pass reproduces the generation forward bitwise, for bf16 and mxfp8,
   including when the two see different token counts.
3. Forward and gradients stay close to the standard TE bf16 path.
4. Generation serves the weights it was last refit with, since the megakernel
   holds its own transformed copy of the expert weights.
5. The MoE layer behaves the same inside a whole transformer layer.

Needs Blackwell (sm100 megakernels) and a FlashInfer that ships ``moe_ep``. The
expert-parallel size is the world size, and it has to divide the expert count.
The construction-time checks that need no GPU are in
``test_flashinfer_mega_moe.py``.
"""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.moe.mega import MegatronMegaMoEAdapter
from megatron.core.inference.moe.mega._deps import _HAVE_FLASHINFER_MOE_EP
from megatron.core.inference.moe.mega.training_weights import reset_training_scratches
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
    MegaLocalPassthroughDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

NUM_EXPERTS = 16
# topk=8 rather than a small value: generation reads the expert indices from
# torch.topk on the router scores while the value pass recovers them from
# TopKRouter's boolean map, and at topk=2 the combine is a two-term sum, which
# commutes exactly and cannot see an ordering difference that topk=8 can.
ROUTER_TOPK = 8
LOCAL_TOKENS = 8
# Generation decodes a few tokens per rank while a training forward takes a whole
# microbatch, so parity rests on a token's output not depending on how many others
# shared the launch.
GEN_TOKENS = 8
TRAIN_TOKEN_COUNTS = (512, 2048)
# The precisions whose kernel weights Megatron packs and owns, i.e. the ones that
# can drive the training forward and be refit. Tracks
# ``InferenceGroupedMLP._MEGA_CALLER_OWNED_PRECISIONS``.
OWNED_PRECISIONS = ("bf16", "mxfp8")

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not _HAVE_FLASHINFER_MOE_EP, reason="FlashInfer moe_ep not installed"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="sm100 mega kernels require Blackwell",
    ),
]


def _config(mega_training: bool, **overrides):
    """An inference-optimized SwiGLU MoE layer config on the mega backend.

    ``mega_training`` selects the training forward; everything else is held fixed
    so a generation/training pair differs only in which kernel produces the value.
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
        activation_func=F.silu,
        gated_linear_unit=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="inference_optimized",
        inference_grouped_gemm_backend="flashinfer_mega",
        inference_mega_max_tokens_per_rank=64,
        expert_model_parallel_size=Utils.world_size,
        # The value pass stores no activations, so the backward comes from a
        # recompute pass.
        recompute_granularity="selective",
        recompute_modules=["moe"],
        moe_inference_training_forward=mega_training,
    )
    base.update(overrides)
    return TransformerConfig(**base)


def _parity_configs(precision, **overrides):
    """``(generation, training)`` configs differing only in the training-forward flag."""
    return (
        _config(mega_training=False, inference_mega_precision=precision, **overrides),
        _config(mega_training=True, inference_mega_precision=precision, **overrides),
    )


def _build_layer(config, for_inference: bool = False):
    """Build the MoE layer.

    ``for_inference`` stands up what the dynamic inference context allocates in
    production for a generation layer (the valid-tokens scalar). A training-only
    layer deliberately leaves it unallocated, which is the situation the value
    pass has to handle itself.
    """
    if for_inference:
        MegaLocalPassthroughDispatcher.allocate_buffers()
    return get_inference_optimized_moe_spec()(config=config).cuda()


def _copy_expert_weights(src, dst):
    """Give two layers identical weights.

    Done before either layer's first forward, while parameters still own their
    storage: the inference path redirects ``param.data`` into a concatenated
    buffer on first use.
    """
    dst.load_state_dict(src.state_dict())


def _bump_expert_weights(layer, delta=0.01):
    """Rewrite the expert parameters in place, standing in for an optimizer step or a refit."""
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if "expert" in name or "linear_fc" in name:
                param.add_(delta)


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


def _generate(layer, hidden):
    with torch.no_grad(), InferenceMode.active():
        out, _ = layer(hidden)
    # Cloned because the kernel may return a view of a workspace that the next
    # launch overwrites.
    return out.clone()


def _chunked_generation(layer, hidden, chunk):
    """Replay ``hidden`` through the generation layer ``chunk`` tokens at a time."""
    return torch.cat(
        [
            _generate(layer, hidden[start : start + chunk])
            for start in range(0, hidden.shape[0], chunk)
        ]
    )


def _elements_that_differ(got, want) -> int:
    """How many elements differ, on the worst rank.

    Reduced because each rank holds different tokens and different local experts,
    and asserting on the reduced value makes the verdict rank-independent.
    """
    differing = torch.tensor([float((got != want).sum().item())], device="cuda")
    torch.distributed.all_reduce(differing, op=torch.distributed.ReduceOp.MAX)
    return int(differing.item())


def _rel_rms(got, want) -> float:
    """Relative RMS error on the worst rank, for comparisons that are not bitwise."""
    got, want = got.float(), want.float()
    error = ((got - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt()).reshape(1)
    torch.distributed.all_reduce(error, op=torch.distributed.ReduceOp.MAX)
    return error.item()


def _count_value_passes(experts):
    """Wrap the megakernel value pass of ``experts`` and return the list of its calls."""
    calls = []
    original = experts._mega_training_forward_pass

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    experts._mega_training_forward_pass = counting
    return calls


@pytest.fixture(autouse=True)
def _parallel_state():
    if NUM_EXPERTS % Utils.world_size:
        pytest.skip(f"num_experts={NUM_EXPERTS} must divide EP={Utils.world_size}")
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
    # Registers the 'expert-parallel-rng' tracker state that expert weight init
    # requires. Utils.initialize_model_parallel does not do this.
    model_parallel_cuda_manual_seed(123)

    def reset():
        # Process-wide state. The shared training layer is pinned to one geometry,
        # so leaking it would raise in the next test that builds a different one.
        reset_training_scratches()
        InferenceAllGatherDispatcherBase._valid_tokens_tensor = None
        MegatronMegaMoEAdapter.reset_shared_training()

    reset()
    yield
    reset()
    Utils.destroy_model_parallel()


@pytest.fixture
def batch_invariant():
    """Batch-invariant mode, which is global kernel patching: on before any layer is built,
    off again afterwards, or it would silently change the next test."""
    enable_batch_invariant_mode(backend="te_native")
    yield
    disable_batch_invariant_mode()


class TestValuePassRuns:
    """(1) The path runs, and the value pass reaches the megakernel."""

    @pytest.mark.parametrize("precision", OWNED_PRECISIONS)
    def test_forward_backward_runs(self, precision):
        config = _config(mega_training=True, inference_mega_precision=precision)
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

    def test_value_pass_reaches_the_megakernel_once(self):
        """Exactly one megakernel call per forward, none from the recompute.

        A layer that stops taking the megakernel still produces finite outputs and
        gradients through TE, so only a count can see the fallback. More than one
        call means the recompute pass took this path too, leaving the backward
        without a graph.
        """
        config = _config(mega_training=True)
        layer = _build_layer(config).train()
        calls = _count_value_passes(layer.experts)

        out, _ = layer(_hidden(config, seed=0).requires_grad_(True))
        out.sum().backward()

        assert len(calls) == 1, f"expected one megakernel value pass, saw {len(calls)}"

    def test_eval_forward_takes_the_megakernel(self):
        """The log-prob pass runs under ``eval()`` and must not fall back to TE."""
        config = _config(mega_training=True)
        layer = _build_layer(config).eval()
        calls = _count_value_passes(layer.experts)

        with torch.no_grad():
            layer(_hidden(config, seed=0))

        assert len(calls) == 1, "the eval forward did not take the megakernel"


class TestTrainGenParity:
    """(2) The value pass reproduces the generation forward bitwise.

    For mxfp8 this is also what makes the straight-through gradient worth taking:
    the backward is knowingly bf16, so the forward has to match generation exactly.
    Each forward is run once before the comparison so kernel compilation and
    workspace warmup are out of the picture.
    """

    @pytest.mark.parametrize("precision", OWNED_PRECISIONS)
    @pytest.mark.parametrize("mode", ["train", "eval"])
    def test_training_forward_matches_generation_forward(self, precision, mode):
        gen_config, train_config = _parity_configs(precision)
        gen_layer = _build_layer(gen_config, for_inference=True).eval()
        train_layer = _build_layer(train_config)
        getattr(train_layer, mode)()
        _copy_expert_weights(gen_layer, train_layer)

        # The log-prob pass runs eval() under no_grad; the training pass keeps grad on.
        context = torch.no_grad if mode == "eval" else contextlib.nullcontext

        hidden = _hidden(train_config, seed=1)
        _generate(gen_layer, hidden)
        with context():
            train_layer(hidden)

        gen_out = _generate(gen_layer, hidden)
        with context():
            train_out, _ = train_layer(hidden)

        differing = _elements_that_differ(train_out, gen_out)
        assert differing == 0, (
            f"{precision} {mode}-mode train/gen parity broken: {differing}/{train_out.numel()} "
            "elements differ. Same weights and the same kernel, so this is routing, the "
            "weight layout or the repack, not rounding"
        )

    @pytest.mark.parametrize("precision", OWNED_PRECISIONS)
    def test_parity_holds_after_a_weight_update(self, precision):
        """Guards the repack: a snapshotted kernel weight would drift here."""
        gen_config, train_config = _parity_configs(precision)
        gen_layer = _build_layer(gen_config, for_inference=True).eval()
        train_layer = _build_layer(train_config).train()
        _copy_expert_weights(gen_layer, train_layer)

        hidden = _hidden(train_config, seed=2)
        _generate(gen_layer, hidden)
        train_layer(hidden)

        # Stand in for an optimizer step, applied to both layers.
        for layer in (gen_layer, train_layer):
            _bump_expert_weights(layer)
        assert gen_layer.experts.refresh_mega_weights() is True

        gen_out = _generate(gen_layer, hidden)
        with torch.no_grad():
            train_out, _ = train_layer(hidden)

        differing = _elements_that_differ(train_out, gen_out)
        assert differing == 0, f"{precision} parity lost after a weight update: {differing} differ"


class TestTokenCountParity:
    """(2b) A token's output does not depend on how many tokens shared the launch.

    The tests above hand both sides the same token count, but in RL they never
    match: generation decodes a few tokens per step and the training log-prob pass
    takes a whole microbatch. Two comparisons, so a failure says which side moved:

    * generation wide vs generation chunked -- the kernel alone;
    * training wide vs generation chunked -- the production case.

    Batch-invariant mode, which is what the RL recipe runs and which keeps the
    router from wobbling with batch size.
    """

    @staticmethod
    def _configs(precision, capacity):
        try:
            return _parity_configs(
                precision,
                inference_mega_max_tokens_per_rank=capacity,
                attention_backend=AttnBackend.flash,
                flash_attention_version=3,
                attention_dropout=0.0,
                batch_invariant_mode=True,
                batch_invariant_backend="te_native",
            )
        except (ValueError, AssertionError, ImportError) as error:
            pytest.skip(f"batch-invariant mode is unavailable in this environment: {error}")

    @pytest.mark.parametrize("precision", OWNED_PRECISIONS)
    def test_training_forward_matches_chunked_generation(self, precision, batch_invariant):
        # One workspace capacity for every comparison; the adapter rejects a
        # forward wider than its workspace.
        capacity = max(GEN_TOKENS, *TRAIN_TOKEN_COUNTS)
        gen_config, train_config = self._configs(precision, capacity)
        gen_layer = _build_layer(gen_config, for_inference=True).eval()
        train_layer = _build_layer(train_config).eval()
        _copy_expert_weights(gen_layer, train_layer)

        results = []
        for count in TRAIN_TOKEN_COUNTS:
            hidden = _hidden(train_config, seed=count, tokens=count)
            gen_wide = _generate(gen_layer, hidden)
            gen_chunked = _chunked_generation(gen_layer, hidden, GEN_TOKENS)
            with torch.no_grad():
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
                f"{precision}: generation output depends on batch size at {count} vs "
                f"{GEN_TOKENS} tokens ({kernel_differs} elements differ); the kernel is not "
                "batch-invariant, independent of the training path"
            )
            assert production_differs == 0, (
                f"{precision}: training at {count} tokens differs from generation at "
                f"{GEN_TOKENS} ({production_differs} elements differ) although the kernel "
                "itself is batch-invariant"
            )


class TestKernelDeterminism:
    """Launching the same work twice returns the same bits.

    Distinct from batch invariance, which varies the launch width: here the layer,
    weights, routing and token count are all fixed. A launch-dependent result at
    fixed width (atomics, split-K, workspace state carried across launches) would
    mean a rollout cannot be reproduced even by re-running it unchanged.
    """

    RELAUNCHES = 8

    @pytest.mark.parametrize("precision", OWNED_PRECISIONS)
    def test_repeated_launches_return_identical_bits(self, precision):
        overrides = {}
        if precision != "bf16":
            # Megatron's fused quantize kernels cover squared-ReLU only; the mega
            # path quantizes inside FlashInfer regardless.
            overrides["inference_moe_disable_fused_quant_kernels"] = True
        config = _config(mega_training=False, inference_mega_precision=precision, **overrides)
        layer = _build_layer(config, for_inference=True).eval()

        hidden = _hidden(config, seed=3)
        reference = _generate(layer, hidden)
        differed = sum(
            _elements_that_differ(_generate(layer, hidden), reference) > 0
            for _ in range(self.RELAUNCHES)
        )

        assert differed == 0, (
            f"the {precision} megakernel returned different bits in {differed} of "
            f"{self.RELAUNCHES} identical relaunches"
        )


class TestAgainstStandardBf16:
    """(3) Forward and gradients stay close to the standard TE bf16 path.

    The megakernel reduces the expert combine in fp32 in one pass while the TE
    path accumulates in bf16 across dispatch and combine, so the two differ by
    rounding rather than by algorithm. The bounds are not bitwise claims; they
    leave headroom for seed variation and stay far below the O(0.1) error that a
    wrong expert, a wrong gate/up interleave or divergent routing produces.
    """

    FORWARD_TOL = 1.5e-2
    GRAD_TOL = 5e-2

    @staticmethod
    def _layers():
        te_layer = _build_layer(_config(mega_training=False)).train()
        mega_layer = _build_layer(_config(mega_training=True)).train()
        _copy_expert_weights(te_layer, mega_layer)
        return te_layer, mega_layer

    def test_forward_matches_te_bf16(self):
        te_layer, mega_layer = self._layers()
        hidden = _hidden(mega_layer.config, seed=3)

        te_out, _ = te_layer(hidden)
        mega_out, _ = mega_layer(hidden)

        error = _rel_rms(mega_out, te_out)
        assert error < self.FORWARD_TOL, f"forward diverges from TE bf16: rel_rms={error:.3e}"

    def test_input_and_weight_gradients_match_te_bf16(self):
        te_layer, mega_layer = self._layers()
        te_hidden = _hidden(mega_layer.config, seed=4).requires_grad_(True)
        mega_hidden = te_hidden.detach().clone().requires_grad_(True)

        te_out, _ = te_layer(te_hidden)
        mega_out, _ = mega_layer(mega_hidden)
        # A fixed non-uniform cotangent, so the reduction does not mask
        # per-position differences the way sum() would.
        grad_out = torch.randn(
            te_out.shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=torch.Generator(device="cuda").manual_seed(5),
        )
        te_out.backward(grad_out)
        mega_out.backward(grad_out)

        dgrad_error = _rel_rms(mega_hidden.grad, te_hidden.grad)
        assert dgrad_error < self.GRAD_TOL, f"dgrad diverges: rel_rms={dgrad_error:.3e}"
        wgrad_error = _rel_rms(
            mega_layer.experts.linear_fc1.weight0.grad, te_layer.experts.linear_fc1.weight0.grad
        )
        assert wgrad_error < self.GRAD_TOL, f"wgrad diverges: rel_rms={wgrad_error:.3e}"


def _use_flashinfer_preprocessing(layer):
    """Put one generation layer back on FlashInfer's own weight preprocessing.

    bf16 and mxfp8 generation own their transformed weights so that a refit can
    rewrite them, which leaves no in-tree way to reach the path FlashInfer takes
    when it preprocesses and snapshots the weights itself. Rebuilding the adapter
    is enough: each one builds its FlashInfer layer lazily.
    """
    experts = layer.experts
    experts._mega_adapter = MegatronMegaMoEAdapter(
        config=experts.config, ep_group=experts.ep_group, owns_transformed_weights=False
    )
    # Instance attribute shadowing the method, so the adapter is handed the raw
    # per-expert stacks and preprocesses them itself.
    experts._mega_inference_weights = lambda: None


class TestGenerationWeightOwnership:
    """(4) Generation serves the weights it was last refit with.

    The megakernel reads its own transformed copy of the expert weights, not the
    parameters. Letting FlashInfer preprocess and snapshot them is right exactly
    once; bf16 and mxfp8 instead own the buffer and rewrite it in place, so a
    refit is a repack rather than a teardown (an EP collective, a symmetric-heap
    reallocation and a CuTeDSL recompile).
    """

    @pytest.mark.parametrize("precision", OWNED_PRECISIONS)
    def test_caller_owned_weights_match_flashinfer_preprocessing(self, precision):
        """The repack gives the same forward as letting FlashInfer build the pack itself.

        ``test_mega_training_weights.py`` checks the packed bytes; this checks that
        handing them over as ``transformed_weights`` is wired correctly. For mxfp8
        it is also the only check that the scale planes reach the kernel.
        """
        config = _config(mega_training=False, inference_mega_precision=precision)
        owned = _build_layer(config, for_inference=True).eval()
        preprocessed = _build_layer(config, for_inference=True).eval()
        _copy_expert_weights(owned, preprocessed)
        _use_flashinfer_preprocessing(preprocessed)

        hidden = _hidden(config, seed=6)
        differing = _elements_that_differ(_generate(owned, hidden), _generate(preprocessed, hidden))
        assert differing == 0, f"caller-owned and FlashInfer-preprocessed {precision} differ"

    def test_refit_without_a_refresh_serves_stale_weights(self):
        """Negative control: the staleness the refit hook exists to prevent is real."""
        config = _config(mega_training=False)
        layer = _build_layer(config, for_inference=True).eval()
        hidden = _hidden(config, seed=6)

        # inference_mode rather than no_grad, because the engine generates under
        # it: the weight buffer is allocated by the first generation forward and
        # would be an inference tensor, which a refit cannot write in place from
        # ordinary mode.
        with torch.inference_mode(), InferenceMode.active():
            before = layer(hidden)[0].clone()

        _bump_expert_weights(layer)
        with torch.inference_mode(), InferenceMode.active():
            stale = layer(hidden)[0].clone()
        assert torch.equal(before, stale), "the kernel picked up a parameter write with no refresh"

        assert layer.experts.refresh_mega_weights() is True
        with torch.inference_mode(), InferenceMode.active():
            refreshed = layer(hidden)[0].clone()
        assert not torch.equal(before, refreshed), "refresh_mega_weights() did not reach the kernel"

    @pytest.mark.parametrize("precision", OWNED_PRECISIONS)
    def test_refresh_after_a_refit_matches_a_freshly_built_layer(self, precision):
        """A refit layer is indistinguishable from one that had the new weights all along.

        For mxfp8 the refresh requantizes rather than copies, so this also catches a
        requantization that writes somewhere the kernel is not reading.
        """
        config = _config(mega_training=False, inference_mega_precision=precision)
        refitted = _build_layer(config, for_inference=True).eval()
        hidden = _hidden(config, seed=6)

        # The first forward binds the kernel to the pre-refit weights; without it
        # there would be nothing stale to refresh.
        _generate(refitted, hidden)
        _bump_expert_weights(refitted)
        assert refitted.experts.refresh_mega_weights() is True

        reference = _build_layer(config, for_inference=True).eval()
        _copy_expert_weights(refitted, reference)

        differing = _elements_that_differ(_generate(refitted, hidden), _generate(reference, hidden))
        assert differing == 0, f"a refreshed {precision} layer differs from a fresh one"

    def test_refit_discovers_the_refresh_hook(self):
        """``resharding.refit`` finds the hook by name, so keep it findable.

        The refit calls whatever module answers to ``refresh_mega_weights``.
        Nothing type-checks that, so renaming the method or moving it off the
        experts module would silently leave generation on snapshotted weights.
        """
        layer = _build_layer(_config(mega_training=False))

        hooks = [
            module
            for module in layer.modules()
            if getattr(module, "refresh_mega_weights", None) is not None
        ]
        assert hooks == [layer.experts]
        assert hooks[0].refresh_mega_weights() is True

    def test_each_generation_layer_owns_its_weight_buffer(self):
        """Generation buffers are per layer, unlike the shared training scratch."""
        config = _config(mega_training=False)
        first = _build_layer(config, for_inference=True).eval()
        second = _build_layer(config, for_inference=True).eval()
        # Deliberately not synced: identical experts would make a shared buffer
        # undetectable.
        hidden = _hidden(config, seed=6)

        first_before = _generate(first, hidden)
        second_out = _generate(second, hidden)
        first_after = _generate(first, hidden)

        assert not torch.equal(first_before, second_out), "the layers hold identical weights"
        assert torch.equal(first_before, first_after), "layers share a weight buffer"

    def test_refresh_refuses_an_unowned_precision(self):
        """A precision Megatron does not pack refuses rather than drifting.

        nvfp4 and fp8_fp4 still let FlashInfer preprocess and snapshot, so there is
        no buffer to rewrite and serving the pre-refit weights would be silent.
        """
        layer = _build_layer(_config(mega_training=False))
        # Read at refresh time, so this reaches the guard without nvfp4 parameters.
        layer.experts.config.inference_mega_precision = "nvfp4"

        with pytest.raises(NotImplementedError, match="quantizes the weights"):
            layer.experts.refresh_mega_weights()


class TestWholeTransformerLayer:
    """(5) The MoE layer inside the transformer layer it ships in.

    Enabling the training forward also puts the training side on
    ``transformer_impl='inference_optimized'``, so attention, the projections and
    the norms change implementation along with the experts. The MoE-only tests
    cannot see that. Generation is out of scope here: it needs a dynamic inference
    context and a KV cache.
    """

    SEQ_LEN = 16
    BATCH = 2

    @staticmethod
    def _build_block(config):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_inference_submodules
        from megatron.core.transformer.transformer_layer import TransformerLayer

        submodules = get_gpt_layer_with_inference_submodules(
            num_experts=config.num_moe_experts, moe_grouped_gemm=True
        )
        return TransformerLayer(config, submodules).cuda()

    def _inputs(self, config, seed):
        hidden = torch.randn(
            (self.SEQ_LEN, self.BATCH, config.hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        mask = torch.ones((1, 1, self.SEQ_LEN, self.SEQ_LEN), dtype=bool, device="cuda")
        return hidden, mask

    def test_layer_forward_backward_runs(self):
        # Dropout off so the train and eval forwards below differ only in the kernel.
        config = _config(mega_training=True, hidden_dropout=0.0)
        block = self._build_block(config).train()
        calls = _count_value_passes(block.mlp.experts)

        hidden, mask = self._inputs(config, seed=11)
        hidden.requires_grad_(True)
        out, _ = block(hidden_states=hidden, attention_mask=mask)
        out.sum().backward()

        assert out.shape == (self.SEQ_LEN, self.BATCH, config.hidden_size)
        assert torch.isfinite(out).all()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        assert len(calls) == 1, f"expected one megakernel value pass, saw {len(calls)}"

    def test_eval_forward_matches_train_forward(self):
        config = _config(mega_training=True, hidden_dropout=0.0)
        block = self._build_block(config)
        hidden, mask = self._inputs(config, seed=12)

        block.train()
        with torch.no_grad():
            train_out, _ = block(hidden_states=hidden, attention_mask=mask)
        block.eval()
        calls = _count_value_passes(block.mlp.experts)
        with torch.no_grad():
            eval_out, _ = block(hidden_states=hidden, attention_mask=mask)

        assert len(calls) == 1, "the eval-mode forward did not take the megakernel"
        differing = _elements_that_differ(eval_out, train_out)
        assert differing == 0, f"eval and train forwards differ in {differing} elements"
