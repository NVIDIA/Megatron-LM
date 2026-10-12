# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed tests for the FlashInfer mega MoE generation backend.

This is ``inference_grouped_gemm_backend='flashinfer_mega'``, the fused EP
transport plus SwiGLU expert-MLP megakernel used by dynamic inference. The
properties that matter, and what these tests cover:

1. A token's output does not depend on how many tokens shared the launch
   (batch invariance), for the caller-owned precisions.
2. Relaunching the same work returns the same bits.
3. Generation serves the weights it was last refit with, since the megakernel
   holds its own transformed copy of the expert weights.

Needs Blackwell (sm100 megakernels) and a FlashInfer that ships ``moe_ep``. The
expert-parallel size is the world size, and it has to divide the expert count.
The construction-time checks that need no GPU are in
``test_flashinfer_mega_moe.py``, and the kernel-layout packers are checked
against FlashInfer's own preprocessing in ``test_mega_kernel_weights.py``.
"""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.moe.mega import MegatronMegaMoEAdapter
from megatron.core.inference.moe.mega._deps import _HAVE_FLASHINFER_MOE_EP
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
ROUTER_TOPK = 8
LOCAL_TOKENS = 8
# Decode takes a few tokens per rank per step while prefill takes many, so batch
# invariance rests on a token's output not depending on how many others shared
# the launch.
GEN_TOKENS = 8
WIDE_TOKEN_COUNTS = (512, 2048)
# The precisions whose kernel weights Megatron packs and owns, i.e. the ones that
# can be refit. Tracks ``_MEGA_CALLER_OWNED_PRECISIONS`` in
# ``megatron/core/transformer/moe/experts.py``. bf16_nvfp4 is included only when
# this FlashInfer build has the kernel (flashinfer-python >= 0.7.1); older
# builds still run the bf16 and mxfp8 cases.
_BF16_NVFP4_KERNEL = (
    "flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_nvfp4_bf16_cutedsl"
)
OWNED_PRECISIONS = ("bf16", "mxfp8") + (
    ("bf16_nvfp4",) if importlib.util.find_spec(_BF16_NVFP4_KERNEL) is not None else ()
)

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not _HAVE_FLASHINFER_MOE_EP, reason="FlashInfer moe_ep not installed"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="sm100 mega kernels require Blackwell",
    ),
]


def _config(**overrides):
    """An inference-optimized SwiGLU MoE layer config on the mega backend."""
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
    )
    base.update(overrides)
    return TransformerConfig(**base)


def _build_layer(config):
    """Build the MoE layer, with what the dynamic inference context allocates for it.

    That is the valid-tokens scalar the dispatcher records the local token count in.
    """
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


@pytest.fixture(autouse=True)
def _parallel_state():
    if NUM_EXPERTS % Utils.world_size:
        pytest.skip(f"num_experts={NUM_EXPERTS} must divide EP={Utils.world_size}")
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
    # Registers the 'expert-parallel-rng' tracker state that expert weight init
    # requires. Utils.initialize_model_parallel does not do this.
    model_parallel_cuda_manual_seed(123)

    def reset():
        # Process-wide state, which would otherwise leak into the next test.
        InferenceAllGatherDispatcherBase._valid_tokens_tensor = None

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


class TestBatchInvariance:
    """(1) A token's output does not depend on how many tokens shared the launch.

    RL never matches token counts between passes: decode takes a few tokens per
    step and prefill or a log-prob pass takes a whole microbatch. Generation wide
    vs generation chunked isolates the kernel. Batch-invariant mode is what the RL
    recipe runs and what keeps the router from wobbling with batch size.
    """

    @staticmethod
    def _config(precision, capacity):
        try:
            return _config(
                inference_mega_precision=precision,
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
    def test_wide_launch_matches_chunked_launches(self, precision, batch_invariant):
        # One workspace capacity for every comparison; the adapter rejects a
        # forward wider than its workspace.
        capacity = max(GEN_TOKENS, *WIDE_TOKEN_COUNTS)
        config = self._config(precision, capacity)
        layer = _build_layer(config).eval()

        results = []
        for count in WIDE_TOKEN_COUNTS:
            hidden = _hidden(config, seed=count, tokens=count)
            wide = _generate(layer, hidden)
            chunked = _chunked_generation(layer, hidden, GEN_TOKENS)
            results.append((count, _elements_that_differ(wide, chunked)))

        # Asserted after every count has run, so one failure does not hide the
        # numbers for the others.
        for count, differs in results:
            assert differs == 0, (
                f"{precision}: output depends on batch size at {count} vs {GEN_TOKENS} "
                f"tokens ({differs} elements differ); the kernel is not batch-invariant"
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
        config = _config(inference_mega_precision=precision, **overrides)
        layer = _build_layer(config).eval()

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
    """(3) Generation serves the weights it was last refit with.

    The megakernel reads its own transformed copy of the expert weights, not the
    parameters. Letting FlashInfer preprocess and snapshot them is right exactly
    once; bf16 and mxfp8 instead own the buffer and rewrite it in place, so a
    refit is a repack rather than a teardown (an EP collective, a symmetric-heap
    reallocation and a CuTeDSL recompile).
    """

    @pytest.mark.parametrize("precision", OWNED_PRECISIONS)
    def test_caller_owned_weights_match_flashinfer_preprocessing(self, precision):
        """The repack gives the same forward as letting FlashInfer build the pack itself.

        ``test_mega_kernel_weights.py`` checks the packed bytes; this checks that
        handing them over as ``transformed_weights`` is wired correctly. For mxfp8
        it is also the only check that the scale planes reach the kernel.
        """
        config = _config(inference_mega_precision=precision)
        owned = _build_layer(config).eval()
        preprocessed = _build_layer(config).eval()
        _copy_expert_weights(owned, preprocessed)
        _use_flashinfer_preprocessing(preprocessed)

        hidden = _hidden(config, seed=6)
        differing = _elements_that_differ(_generate(owned, hidden), _generate(preprocessed, hidden))
        assert differing == 0, f"caller-owned and FlashInfer-preprocessed {precision} differ"

    def test_refit_without_a_refresh_serves_stale_weights(self):
        """Negative control: the staleness the refit hook exists to prevent is real."""
        config = _config()
        layer = _build_layer(config).eval()
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
        config = _config(inference_mega_precision=precision)
        refitted = _build_layer(config).eval()
        hidden = _hidden(config, seed=6)

        # The first forward binds the kernel to the pre-refit weights; without it
        # there would be nothing stale to refresh.
        _generate(refitted, hidden)
        _bump_expert_weights(refitted)
        assert refitted.experts.refresh_mega_weights() is True

        reference = _build_layer(config).eval()
        _copy_expert_weights(refitted, reference)

        differing = _elements_that_differ(_generate(refitted, hidden), _generate(reference, hidden))
        assert differing == 0, f"a refreshed {precision} layer differs from a fresh one"

    def test_refit_discovers_the_refresh_hook(self):
        """``resharding.refit`` finds the hook by name, so keep it findable.

        The refit calls whatever module answers to ``refresh_mega_weights``.
        Nothing type-checks that, so renaming the method or moving it off the
        experts module would silently leave generation on snapshotted weights.
        """
        layer = _build_layer(_config())

        hooks = [
            module
            for module in layer.modules()
            if getattr(module, "refresh_mega_weights", None) is not None
        ]
        assert hooks == [layer.experts]
        assert hooks[0].refresh_mega_weights() is True

    def test_each_generation_layer_owns_its_weight_buffer(self):
        """Each layer's kernel-weight buffer is its own, never shared with another layer."""
        config = _config()
        first = _build_layer(config).eval()
        second = _build_layer(config).eval()
        # Deliberately not synced: identical experts would make a shared buffer
        # undetectable.
        hidden = _hidden(config, seed=6)

        first_before = _generate(first, hidden)
        second_out = _generate(second, hidden)
        first_after = _generate(first, hidden)

        assert not torch.equal(first_before, second_out), "the layers hold identical weights"
        assert torch.equal(first_before, first_after), "layers share a weight buffer"

    def test_refresh_refuses_an_unowned_precision(self):
        """A precision this backend does not pack refuses rather than drifting.

        bf16, mxfp8, and bf16_nvfp4 own their buffers. Anything else has no
        packer, so serving the pre-refit weights would be silent.
        """
        layer = _build_layer(_config())
        # Read at refresh time, so this reaches the guard without constructing
        # a layer in an unsupported precision.
        layer.experts.config.inference_mega_precision = "nvfp4"

        with pytest.raises(NotImplementedError, match="quantizes the weights"):
            layer.experts.refresh_mega_weights()
