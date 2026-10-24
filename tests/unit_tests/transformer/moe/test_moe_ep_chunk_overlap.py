# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Core tests for the experimental EP chunk overlap path.

Two things are covered. First, that the chunked custom autograd node is numerically transparent: a
fake expert compute runs the **real** squared-ReLU expert math and differentiates it with autograd,
so the chunked path can be compared against the same math run in one shot. Second, that ``MoELayer``
routes to the facade when the feature is on and to the token dispatcher when it is off.
"""

import sys
from inspect import signature
from types import ModuleType

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.extensions.transformer_engine import cpu_offload
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import ChunkOffloadHandler
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.tensor_parallel import gtp_api
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe import moe_ep_chunk_overlap as overlap_module
from megatron.core.transformer.moe.moe_ep_chunk_overlap import ep_chunk_overlap
from megatron.core.transformer.moe.moe_ep_chunk_overlap.ep_chunk_overlap import (
    CachedForwardChunkActivation,
    MoEEPChunkOverlapRuntime,
)
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.token_dispatcher import _HybridEPManager
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

_HIDDEN_SIZE = 256  # TE fused grouped MLP needs % 64, MXFP8 % 32; 256 is the only measured value
_FFN_HIDDEN_SIZE = 128
_NUM_EXPERTS = 4
_ROUTER_TOPK = 2
# A concrete SM count for the facade, which takes its geometry as scalars. There is no
# overlap-specific config knob: production reads moe_flex_dispatcher_num_sms.
_NUM_COMM_SMS = 32
_SEQUENCE_LENGTH = 7
_MICRO_BATCH_SIZE = 2
_NUM_TOKENS = _SEQUENCE_LENGTH * _MICRO_BATCH_SIZE  # 14: deliberately not divisible by 4 chunks

_DISPATCHER_METHODS = (
    "dispatch_preprocess",
    "token_dispatch",
    "dispatch_postprocess",
    "combine_preprocess",
    "token_combine",
    "combine_postprocess",
)


def _expert_math(hidden, topk_indices, topk_weights, fc1_weight, fc2_weight):
    """Weighted squared-ReLU MoE in FP32: ``sum_k w[t,k] * fc2(srelu(fc1(x[t])))``.

    Row-independent by construction, so chunking it must be exact up to FP32 associativity in the
    weight-gradient reduction.
    """
    preactivation = torch.einsum("th,tkih->tki", hidden, fc1_weight[topk_indices])
    activation = torch.relu(preactivation) ** 2
    expert_output = torch.einsum("tki,tkhi->tkh", activation, fc2_weight[topk_indices])
    return (expert_output * topk_weights.unsqueeze(-1)).sum(dim=1)


class _FakeDispatchManager:
    """One chunk's collectives, as the identity plus the padding the schedule insists on.

    The schedule holds every dispatch at ``max_num_expanded_tokens_per_chunk`` and
    ``check_dispatch_capacity`` raises otherwise, so an identity leg would not even run: this
    one pads to capacity and takes the padding back in the combine. Both legs are ordinary
    differentiable ops, which is the point -- the schedule gets its adjoints by differentiating
    the nodes these build, so a fake that returned a detached tensor would make the backward
    this file exists to test unrunnable.

    What it does NOT do is permute into expert-major order. The production manager does; what
    is on trial here is the chunking and the custom backward, and a permutation would only
    make the one-shot reference harder to write without covering anything new.
    """

    def __init__(self) -> None:
        #: Set by `_ensure_built`, which is where the geometry is first known.
        self.capacity: int | None = None
        self._live_rows: int | None = None
        self.dispatched_probs: torch.Tensor | None = None
        #: The schedule pins this before every dispatch; production reads it to skip a host
        #: sync. Nothing here needs it, but it must be assignable or the schedule would fail.
        self.num_permuted_tokens: int | None = None

    def setup_metadata(self, routing_map, probs, padding_mask=None):
        """Take this chunk's routing, exactly as the trunk manager is handed it."""
        self._routing_map = routing_map
        self._probs = probs

    def dispatch(self, hidden_states):
        self._live_rows = hidden_states.shape[0]
        self.dispatched_probs = self._pad(self._probs)
        return self._pad(hidden_states)

    def combine(self, hidden_states):
        return hidden_states[: self._live_rows]

    def get_number_of_tokens_per_expert(self):
        """Per-LOCAL-expert receive counts, summing to the live rows, with no permutation to
        describe: one expert nominally owns them all."""
        counts = torch.zeros(_NUM_EXPERTS, dtype=torch.long, device=self._probs.device)
        counts[0] = self._live_rows
        return counts

    def _pad(self, tensor):
        """Zero-pad to capacity. ``torch.cat`` so the padding is part of the graph: its backward
        is the slice that drops those rows' gradients, which is what the real combine does."""
        rows = self.capacity - tensor.shape[0]
        return torch.cat([tensor, tensor.new_zeros((rows, *tensor.shape[1:]))], dim=0)


def _routing_map_of(dense_probs: torch.Tensor) -> torch.Tensor:
    """The boolean map the router hands the layer: true exactly where a probability is live."""
    return dense_probs.detach() != 0


class _AutogradRuntime(MoEEPChunkOverlapRuntime):
    """The real runtime with its native compute replaced by autograd over the same expert math.

    Only the build step and the per-chunk stages are overridden, so the two-stream schedule, the
    generation lifecycle, the chunking and the autograd node under test are production code; the
    collectives become `_FakeDispatchManager`'s pad-and-unpad, leaving the stream and event
    plumbing running for real over compute autograd can differentiate. Its wgrad methods simply
    OVERRIDE the runtime's own, so the schedule's ``self.<stage>()`` reaches this class by
    ordinary inheritance.
    """

    fc1_weight: torch.Tensor
    fc2_weight: torch.Tensor
    #: The wgrad stage in call order, produced by the real ``moe_backward``. Bound by
    #: `_make_facade` so the publish probe appends to the same list this fake does.
    wgrad_log: list

    def _ensure_built(self, flat_hidden):
        # Phase TWO only: the inherited __init__ is the keyword surface, so a renamed keyword
        # raises there. No super() -- it reads `weight{i}` and makes a real comm stream.
        if self.comm_stream is not None:
            return
        self.config = ep_chunk_overlap._MoEEPChunkOverlapConfig(
            num_local_tokens=flat_hidden.shape[0],
            hidden_size=self._hidden_size,
            ffn_hidden_size=self._ffn_hidden_size,
            num_experts=self._num_experts,
            router_topk=self._router_topk,
            num_token_chunks=self._num_chunks,
            world_size=torch.distributed.get_world_size(self._tp_ep_group),
            num_comm_SMs=self._num_comm_SMs,
            num_local_experts=self._num_local_experts,
        )
        self.device = torch.device(flat_hidden.device)
        self.comm_stream = torch.cuda.Stream(device=self.device)
        # The managers were built by `_build_facade` and validated by the inherited __init__;
        # only the capacity is new here, because the geometry is what this phase resolves.
        for manager in self._managers:
            manager.capacity = self.config.max_num_expanded_tokens_per_chunk
        # Dropped, not left at the constructor's None: both compute stages are overridden, so an
        # accidental `expert_ops()` must be an AttributeError, not a real `build_expert_ops`.
        del self._ops, self._fc1_op, self._fc2_op

        # The Function's weight inputs, cached at build time as the production runtime does.
        # These are not a `TEGroupedMLP`'s `weight{i}` parameters, which is why the fake has them.
        self.expert_params = (self.fc1_weight, self.fc2_weight)

        self.fc1_main_grad = torch.zeros_like(self.fc1_weight)
        self.fc2_main_grad = torch.zeros_like(self.fc2_weight)
        # Mirrors TE's per-op WeightGradStore: a FIFO with no chunk identity, which the schedule
        # must drain exactly one entry from per chunk.
        self._wgrad_queue: list = []
        #: One entry per chunk: the offload gate the runtime threaded down to that chunk's compute.
        self.offload_log: list = []

    def begin_wgrad_accumulation(self):
        """Nothing to ready: this fake accumulates into buffers its own build allocated."""
        self.wgrad_log.append("begin")

    def end_wgrad_accumulation(self):
        """The EGTP flush leg; this fake owns no GTP group, so only its POSITION is asserted.

        It must land after every chunk's drain (or the staging buffers hold a partial sum) and
        after the publish (or DDP's post hook adds a stale ``param.grad`` into ``main_grad``:
        ``zero_grad_buffer`` clears only the flag -- distributed_data_parallel.py:575-578, :699).
        """
        self.wgrad_log.append("flush")

    def routed_experts_compute_forward(self, dispatched):
        # The gate no longer rides this call: the experts' own fused forward owns the group, so
        # the fake reads the flag off the experts module, which is where production reads it.
        self.offload_log.append(self._experts.offload_fused_group_mlp)
        # The WHOLE capacity window, as the real compute takes it: its dX must come back at
        # capacity. Padded rows carry zero probabilities, so they contribute and take nothing.
        hidden = dispatched.recv_x.detach().float().requires_grad_(True)
        probs = dispatched.recv_probs.detach().float().requires_grad_(True)
        with torch.enable_grad():
            # Top-k off the DENSE row, which is the form the schedule now dispatches; the
            # one-shot reference takes the same two lines, so the two stay comparable.
            weights, indices = torch.topk(probs, k=self._router_topk, dim=-1, sorted=True)
            output = _expert_math(hidden, indices, weights, self.fc1_weight, self.fc2_weight)
        # The saved chunk carries this fake's own inner graph in the same three slots the real
        # one carries TE's, plus the dispatch NODE, which only the schedule differentiates.
        saved = CachedForwardChunkActivation(
            dispatched_tokens=hidden,
            routing_probs=probs,
            fc2_output=output,
            dispatch_graph=dispatched.graph,
        )
        return output.detach().to(dispatched.recv_x.dtype), saved

    def routed_experts_compute_backward(self, saved, grad_recv):
        inputs = [saved.dispatched_tokens, saved.routing_probs, self.fc1_weight, self.fc2_weight]
        grad_hidden, grad_probs, grad_fc1, grad_fc2 = torch.autograd.grad(
            saved.fc2_output, inputs, grad_recv.float()
        )
        # Parked rather than returned, exactly as TE parks a deferred wgrad: the weight gradients
        # stay the wgrad stage's work and never cross the seam.
        self._wgrad_queue.append((grad_fc1, grad_fc2))
        # Flat, as the production stage returns it; the schedule views it back onto the
        # dispatch node's own probability output.
        return grad_hidden.to(grad_recv.dtype), grad_probs.reshape(-1)

    def run_chunk_wgrad(self, *, stream):
        """Wgrad leg. The stream is checked here now that the schedule supplies it directly."""
        # `==` not `is`: torch.cuda.current_stream() builds a fresh wrapper per call, so identity
        # fails even for the same underlying cudaStream_t, which __eq__ compares.
        assert stream == torch.cuda.current_stream(self.device), (
            "the wgrad drain is handed the compute stream, never asks for one; the schedule passes "
            f"the same stream its events are recorded on, got {stream}"
        )
        self.wgrad_log.append("chunk")
        assert len(self._wgrad_queue) == 1, (
            f"the deferred wgrad queue holds {len(self._wgrad_queue)} entries at the drain point, "
            "not exactly one; this chunk's wgrad is missing or belongs to another chunk"
        )
        grad_fc1, grad_fc2 = self._wgrad_queue.pop()
        # Straight into main_grad, as TE's `backward_dw` accumulates: nothing is held back for a
        # later publish, because publishing no longer moves gradients -- it hands DDP placeholders.
        self.fc1_main_grad += grad_fc1
        self.fc2_main_grad += grad_fc2


class _StubOverlap:
    """Stand-in facade that records its calls and keeps its inputs in the autograd graph."""

    def __init__(self):
        self.calls = []

    def forward(self, hidden_states, dense_probs, routing_map):
        self.calls.append((hidden_states, dense_probs, routing_map))
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        scale = dense_probs.sum(dim=-1, keepdim=True).to(flat.dtype)
        return (flat * scale).view(hidden_states.shape), None


def _make_config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=4,
        num_moe_experts=_NUM_EXPERTS,
        moe_ffn_hidden_size=_FFN_HIDDEN_SIZE,
        moe_router_topk=_ROUTER_TOPK,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=False,
        use_cpu_initialization=False,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        # The facade requires a communication-SM budget even though the feature stays off in most
        # of these configs; the value is inert unless the flex DeepEP dispatcher is selected.
        moe_flex_dispatcher_num_sms=40,
        # EXPLICIT: the field still defaults to 'deepep' for the trunk dispatcher, and that name
        # no longer selects a backend here -- it raises. These configs used to take that default.
        moe_flex_dispatcher_backend="hybridep",
    )
    kwargs.update(overrides)
    if kwargs.get("moe_ep_chunk_overlap"):
        # The schedule now runs the experts' OWN fused forward, so __post_init__ requires all of
        # these, plus the dispatcher pair that makes skip_routed_expert_padding() true.
        required = dict(
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            # TEGroupedMLP asserts _is_fused_impl_supported() under the op fuser, and only the
            # three scaled TE activations qualify; the default GELU does not.
            activation_func=squared_relu,
            use_fused_weighted_squared_relu=True,
            gated_linear_unit=False,
            use_transformer_engine_op_fuser=True,
            moe_use_grouped_tensor=True,
            gradient_accumulation_fusion=True,
            overlap_dispatch_backward_with_experts_wgrad=True,
        )
        required.update(overrides)
        kwargs.update(required)
    return TransformerConfig(**kwargs)


def _install_publish_probe(monkeypatch, published: list, wgrad_log: list | None = None) -> None:
    """Shadow the one TransformerEngine symbol ``moe_backward`` imports for itself.

    ``publish_wgrad`` was inlined, so this is the only seam left at which the publish can be
    observed, and the only real TE touch on either fake's path. Each call's parameters are
    appended to ``published`` in call order.
    """

    def get_dummy_wgrads_for_params(params):
        params = list(params)
        if wgrad_log is not None:
            wgrad_log.append("publish")
        published.append(tuple(params))
        # ``None`` where TE returns a real placeholder: autograd accepts it for an input that
        # takes no gradient, and these two tensors are not DDP-managed parameters.
        return [None] * len(params)

    common = ModuleType("transformer_engine.pytorch.ops._common")
    common.get_dummy_wgrads_for_params = get_dummy_wgrads_for_params
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.ops._common", common)


def _patch_schedule_stages(monkeypatch) -> None:
    """Replace the five schedule stages both fakes below stand in for.

    By NAME, so a rename raises here -- the rename detector as well as the substitution.
    ``ensure_expert_parameters_ready`` is a module function; the other four are METHODS, patched
    on the class, where each is bound at call time and the runtime arrives as ``self`` -- exactly
    what the module functions were handed as ``runtime``. The three wgrad entries are
    ``_AutogradRuntime``'s own overrides, which already win by MRO, so patching the base class
    changes no behaviour and keeps the detector.
    """
    monkeypatch.setattr(ep_chunk_overlap, "ensure_expert_parameters_ready", lambda experts: None)
    for name, replacement in (
        ("quantize_expert_weights", lambda runtime: None),
        ("begin_wgrad_accumulation", _AutogradRuntime.begin_wgrad_accumulation),
        ("run_chunk_wgrad", _AutogradRuntime.run_chunk_wgrad),
        ("end_wgrad_accumulation", _AutogradRuntime.end_wgrad_accumulation),
    ):
        monkeypatch.setattr(MoEEPChunkOverlapRuntime, name, replacement)


def _build_facade(*, hidden_size, ffn_hidden_size, num_chunks, offload) -> MoEEPChunkOverlapRuntime:
    """Whatever runtime class is currently patched into the package, built as production does."""
    # The one field the runtime reads off `experts` itself: its activation-offload gate, read by
    # NAME, so a rename on the trunk's side is an AttributeError in every test here.
    experts = torch.nn.Identity()
    experts.offload_fused_group_mlp = offload
    # Resolved through the package, not through the name imported above: that attribute is the
    # switch `_make_facade` flips to put a fake runtime class in production's place.
    return overlap_module.MoEEPChunkOverlapRuntime(
        num_experts=_NUM_EXPERTS,
        router_topk=_ROUTER_TOPK,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_chunks=num_chunks,
        num_comm_SMs=_NUM_COMM_SMS,
        experts=experts,
        tp_ep_group=get_default_pg_collection().tp_ep,
        num_local_experts=_NUM_EXPERTS,
        # ONE PER CHUNK, as `moe_layer.py` builds them; the inherited __init__ refuses any
        # other count.
        managers=[_FakeDispatchManager() for _ in range(num_chunks)],
    )


def _make_inputs(generator, device, hidden_shape, num_tokens):
    """Hidden states plus the sparse dense row the router produces: only top-k is non-zero."""
    hidden = (
        torch.randn(hidden_shape, generator=generator, device=device, dtype=torch.float32)
        .mul_(0.5)
        .to(torch.bfloat16)
        .requires_grad_(True)
    )
    logits = torch.randn(
        (num_tokens, _NUM_EXPERTS), generator=generator, device=device, dtype=torch.float32
    )
    top_values, top_indices = torch.topk(logits.softmax(dim=-1), k=_ROUTER_TOPK, dim=-1)
    dense_probs = torch.zeros_like(logits).scatter(1, top_indices, top_values)
    return hidden, dense_probs.requires_grad_(True)


def _make_layer(config: TransformerConfig) -> MoELayer:
    submodules = get_submodules(
        get_gpt_layer_local_submodules(num_experts=_NUM_EXPERTS, moe_grouped_gemm=False).mlp
    )
    assert isinstance(submodules, MoESubmodules)
    return MoELayer(config, submodules).cuda()


def _spy_on_dispatcher(monkeypatch, layer: MoELayer) -> list[str]:
    """Record every token-dispatcher phase method the layer invokes, in call order."""
    called: list[str] = []
    dispatcher = layer.token_dispatcher
    for method_name in _DISPATCHER_METHODS:
        original = getattr(dispatcher, method_name)

        def make_spy(name, func):
            def spy(*args, **kwargs):
                called.append(name)
                return func(*args, **kwargs)

            return spy

        monkeypatch.setattr(dispatcher, method_name, make_spy(method_name, original))
    return called


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestOverlapAutograd:
    """Drive the real facade and autograd node over the fake expert compute."""

    @pytest.fixture(autouse=True)
    def _monkeypatch(self, monkeypatch):
        self.monkeypatch = monkeypatch

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=1234, data_parallel_random_init=False)
        self.device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=self.device).manual_seed(7)
        self.fc1_weight = (
            torch.randn(
                (_NUM_EXPERTS, _FFN_HIDDEN_SIZE, _HIDDEN_SIZE),
                generator=generator,
                device=self.device,
            )
            * 0.05
        ).requires_grad_(True)
        self.fc2_weight = (
            torch.randn(
                (_NUM_EXPERTS, _HIDDEN_SIZE, _FFN_HIDDEN_SIZE),
                generator=generator,
                device=self.device,
            )
            * 0.05
        ).requires_grad_(True)
        self.generator = generator

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _make_facade(self, num_chunks: int = 2, offload: bool = False) -> MoEEPChunkOverlapRuntime:
        """Build the production runtime with its native compute swapped for autograd."""
        self.wgrad_log: list = []
        runtime_cls = type(
            "_BoundAutogradRuntime",
            (_AutogradRuntime,),
            {
                "fc1_weight": self.fc1_weight,
                "fc2_weight": self.fc2_weight,
                # Shared with the publish probe below: `publish_wgrad` was inlined into
                # `moe_backward`, so its log entry can only come from TE's own seam now.
                "wgrad_log": self.wgrad_log,
            },
        )
        self.monkeypatch.setattr(overlap_module, "MoEEPChunkOverlapRuntime", runtime_cls)
        _patch_schedule_stages(self.monkeypatch)
        self.published: list = []
        _install_publish_probe(self.monkeypatch, self.published, self.wgrad_log)
        return _build_facade(
            hidden_size=_HIDDEN_SIZE,
            ffn_hidden_size=_FFN_HIDDEN_SIZE,
            num_chunks=num_chunks,
            offload=offload,
        )

    def _make_inputs(self):
        hidden_shape = (_SEQUENCE_LENGTH, _MICRO_BATCH_SIZE, _HIDDEN_SIZE)
        return _make_inputs(self.generator, self.device, hidden_shape, _NUM_TOKENS)

    @pytest.mark.parametrize("num_chunks", [1, 2, 3, 4])
    def test_chunked_gradients_match_a_single_shot_autograd_reference(self, num_chunks):
        """The gate: chunking and the custom backward must be numerically transparent."""
        facade = self._make_facade(num_chunks=num_chunks)
        hidden, dense_probs = self._make_inputs()

        output, _ = facade.forward(hidden, dense_probs, _routing_map_of(dense_probs))
        output.float().pow(2).sum().backward()

        # Reference: the identical math in one shot, through ordinary autograd.
        reference_hidden = hidden.detach().clone().requires_grad_(True)
        reference_probs = dense_probs.detach().clone().requires_grad_(True)
        reference_fc1 = self.fc1_weight.detach().clone().requires_grad_(True)
        reference_fc2 = self.fc2_weight.detach().clone().requires_grad_(True)
        topk_weights, topk_indices = torch.topk(
            reference_probs, k=_ROUTER_TOPK, dim=-1, sorted=True
        )
        reference_output = _expert_math(
            reference_hidden.view(-1, _HIDDEN_SIZE).float(),
            topk_indices,
            topk_weights,
            reference_fc1,
            reference_fc2,
        ).to(torch.bfloat16)
        reference_output.float().pow(2).sum().backward()

        # Tolerances follow each tensor's dtype, not the difference hoped for: BF16 activations
        # carry ~3 decimal digits; the FP32 grads differ only by the chunked reduction's order.
        torch.testing.assert_close(
            output, reference_output.view(hidden.shape), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=2e-2, atol=2e-4)
        torch.testing.assert_close(dense_probs.grad, reference_probs.grad, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(facade.fc1_main_grad, reference_fc1.grad, rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(facade.fc2_main_grad, reference_fc2.grad, rtol=1e-3, atol=1e-5)
        # Order and per-chunk count over the real `moe_backward`: ready once, drain exactly one
        # chunk's wgrad inside each chunk, publish once -- the publish being TE's own call.
        assert facade.wgrad_log == (["begin"] + ["chunk"] * num_chunks + ["publish", "flush"])
        # ...over the runtime's ONE cached parameter tuple, which is also what the facade handed
        # the Function as its weight inputs. By IDENTITY: a rebuilt tuple would compare equal.
        assert [tuple(map(id, handed)) for handed in self.published] == [
            tuple(map(id, facade.expert_params))
        ]
        # And the gate above is the offload-free arm, so the numbers are the schedule's alone.
        assert facade.offload_log == [False] * num_chunks

    def test_gradients_accumulate_across_microbatches(self):
        facade = self._make_facade()
        # The SAME input twice, so accumulate and overwrite are distinguishable: with different
        # inputs any second value differs from the first and the assertion proves nothing.
        hidden, dense_probs = self._make_inputs()

        output, _ = facade.forward(hidden, dense_probs, _routing_map_of(dense_probs))
        output.float().sum().backward()
        first_grad = facade.fc1_main_grad.clone()

        hidden.grad = None
        dense_probs.grad = None
        output, _ = facade.forward(hidden, dense_probs, _routing_map_of(dense_probs))
        output.float().sum().backward()

        assert len(self.published) == 2
        torch.testing.assert_close(facade.fc1_main_grad, 2 * first_grad, rtol=1e-5, atol=1e-7)

    def test_no_grad_bypasses_the_autograd_node(self):
        """Eval runs the same kernels as training and must not touch a weight gradient.

        Bypassed, not tolerated: under no_grad ``apply()`` would still run ``forward()`` and
        stash the saved chunks on ctx while building no node, so the graph they hold is never
        freed. Observable here as the publish: an eval forward reaching ``moe_backward``'s
        hand-off would tell DDP a gradient was ready for a backward that never ran. The second
        forward asserts eval left nothing behind that training then trips over.
        """
        facade = self._make_facade()
        hidden, dense_probs = self._make_inputs()

        with torch.no_grad():
            output, bias = facade.forward(hidden, dense_probs, _routing_map_of(dense_probs))
        assert bias is None
        assert output.shape == hidden.shape
        assert self.published == [], "eval must never publish weight gradients"

        # The generation was released, so training still works afterwards.
        trained, _ = facade.forward(hidden, dense_probs, _routing_map_of(dense_probs))
        trained.float().sum().backward()
        assert len(self.published) == 1

    @pytest.mark.parametrize("num_chunks", [1, 3])
    def test_the_offload_gate_reaches_every_chunk(self, num_chunks):
        """``fused_group_mlp`` offloading is ONE group per token chunk, so the gate the facade
        reads off the experts has to arrive at every chunk's compute, not at the region once."""
        facade = self._make_facade(num_chunks=num_chunks, offload=True)
        hidden, dense_probs = self._make_inputs()

        output, _ = facade.forward(hidden, dense_probs, _routing_map_of(dense_probs))
        output.float().sum().backward()

        assert facade.offload_log == [True] * num_chunks

    @pytest.mark.xfail(
        strict=True,
        reason="Asserts an eval-time gate the runtime does not implement. There is no grad-mode "
        "check anywhere in moe_ep_chunk_overlap -- 0 hits for is_grad_enabled/no_grad across all "
        "five files -- _MoEEPChunkOverlapFunction.apply still runs its forward under no_grad, and "
        "offload is forwarded verbatim from experts.offload_fused_group_mlp, so offload_log comes "
        "back [True, True]. Strict, so it flags itself for removal once the gate lands.",
    )
    def test_eval_opens_no_offload_group_even_when_the_experts_ask_for_one(self):
        """No backward runs over a no_grad forward, so a group committed there is never reloaded:
        it would sit in the offload manager's reload list until the iteration boundary."""
        facade = self._make_facade(num_chunks=2, offload=True)
        hidden, dense_probs = self._make_inputs()

        with torch.no_grad():
            facade.forward(hidden, dense_probs, _routing_map_of(dense_probs))

        assert facade.offload_log == [False, False]
        # And the training path that follows still gets the gate, so this is a no_grad rule and
        # not the facade having dropped the flag altogether.
        output, _ = facade.forward(hidden, dense_probs, _routing_map_of(dense_probs))
        output.float().sum().backward()
        assert facade.offload_log == [False, False, True, True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoELayerSeam:
    """The ``MoELayer`` wiring: which of the two paths a forward actually takes."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _make_hidden_states(self):
        return torch.randn(
            _SEQUENCE_LENGTH,
            _MICRO_BATCH_SIZE,
            _HIDDEN_SIZE,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )

    def test_disabled_path_runs_every_dispatcher_phase_in_order(self, monkeypatch):
        layer = _make_layer(_make_config())
        called = _spy_on_dispatcher(monkeypatch, layer)

        layer(self._make_hidden_states())

        assert called == list(_DISPATCHER_METHODS)

    def test_enabled_path_calls_the_facade_once_and_bypasses_the_dispatcher(self, monkeypatch):
        layer = _make_layer(_make_config())
        called = _spy_on_dispatcher(monkeypatch, layer)
        stub = _StubOverlap()
        layer.moe_ep_chunk_overlap = stub
        hidden_states = self._make_hidden_states()

        output, _ = layer(hidden_states)
        output.sum().backward()

        assert len(stub.calls) == 1
        assert called == []
        # The router must stay differentiable through the facade.
        assert hidden_states.grad is not None
        assert layer.router.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestOverlapBackendSelection:
    """``moe_flex_dispatcher_backend`` is what selects this path's collectives, and nothing else.

    There is no transport type any more and no mapping of its own to get wrong: the layer builds
    the trunk's dispatch managers through ``make_dispatch_manager``, the single place a flex
    backend name becomes an implementation. So what is left to gate is that the overlap path
    reaches that function and asks it for one manager per chunk.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_the_backend_name_selects_its_manager(self):
        """Only 'hybridep' is reachable here. 'ncclep' used to be checked alongside it because
        the mapping was a dict this path owned and nothing was constructed; now the manager IS
        built, and ``make_dispatch_manager`` asserts TP x EP > 1 for that backend, which this
        test's expert-parallel size of 1 cannot satisfy."""
        config = _make_config(moe_ep_chunk_overlap=True, moe_flex_dispatcher_backend="hybridep")

        layer = _make_grouped_layer(config)

        managers = layer.moe_ep_chunk_overlap._managers  # noqa: SLF001
        assert all(isinstance(manager, _HybridEPManager) for manager in managers)

    def test_there_is_one_manager_per_token_chunk(self):
        """Not one shared manager: chunk c's combine runs while c+1's dispatch is still in
        flight, and HybridEP's cached replay reads buffer-resident staging from offset 0, so a
        second dispatch into the same buffer invalidates the first one's handle."""
        config = _make_config(
            moe_ep_chunk_overlap=True,
            moe_flex_dispatcher_backend="hybridep",
            moe_ep_chunk_overlap_num_chunks=3,
        )

        layer = _make_grouped_layer(config)

        managers = layer.moe_ep_chunk_overlap._managers  # noqa: SLF001
        assert len(managers) == 3
        assert len({id(manager) for manager in managers}) == 3, "distinct objects, not one reused"

    def test_a_manager_count_that_does_not_match_the_chunk_count_is_refused(self):
        """The runtime's own guard, reached directly: ``moe_layer`` builds the list from the same
        field, so only this test would catch the two drifting apart."""
        with pytest.raises(ValueError, match="one dispatch manager per token chunk"):
            overlap_module.MoEEPChunkOverlapRuntime(
                num_experts=_NUM_EXPERTS,
                router_topk=_ROUTER_TOPK,
                hidden_size=_HIDDEN_SIZE,
                ffn_hidden_size=_FFN_HIDDEN_SIZE,
                num_chunks=2,
                num_comm_SMs=_NUM_COMM_SMS,
                experts=torch.nn.Identity(),
                tp_ep_group=get_default_pg_collection().tp_ep,
                num_local_experts=_NUM_EXPERTS,
                managers=[_FakeDispatchManager()],
            )

    def test_an_unmapped_backend_raises_instead_of_silently_taking_a_default(self):
        config = _make_config(moe_ep_chunk_overlap=True)
        config.moe_flex_dispatcher_backend = "not-a-backend"

        with pytest.raises(ValueError, match="Invalid backend"):
            _make_grouped_layer(config)


def _make_grouped_layer(config: TransformerConfig) -> MoELayer:
    """A layer whose experts are the ``TEGroupedMLP`` this path requires; ``'moe_act'`` is derived
    there (experts.py:275-278) and the config refuses it without ``moe_grouped_gemm``."""
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            num_experts=_NUM_EXPERTS, moe_grouped_gemm=True
        ).mlp
    )
    assert isinstance(submodules, MoESubmodules)
    return MoELayer(config, submodules).cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSplitExecutionMap:
    """The three-phase split that partial CUDA graphs drive must equal the single call.

    ``MoETransformerLayer._forward_mlp_{router,expert_compute,postprocess}`` rewrite
    ``fwd_execution_map`` and thread ``intermediate_tensors`` between three separate
    ``MoELayer.forward`` calls, capturing the first and last. The overlap path used to refuse
    that outright; these assert the refusal was conservative rather than necessary, by driving
    the split by hand -- no graph capture, so a failure here is the SPLIT, not the capture.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _layer_with_stub_runtime():
        """A real MoELayer whose routed-expert region is a deterministic stand-in.

        The runtime is stubbed, not the layer: what is under test is ``custom_forward``'s
        phase plumbing, and a real transport would add EP collectives this test does not need.
        """
        layer = _make_grouped_layer(_make_config(moe_ep_chunk_overlap=True, moe_grouped_gemm=True))
        assert layer.moe_ep_chunk_overlap is not None, "the overlap path must be the active one"

        class _StubRuntime:
            calls = 0

            def forward(self, hidden_states, probs, routing_map):
                """Deterministic in both inputs, so a dropped or reordered phase shows up.

                Shape contract copied from the real runtime: it flattens to [tokens, hidden]
                and returns ``flat_output.view(input_shape), None``. The routing map arrives
                as the third argument because ``preprocess`` stashes it for the expert phase,
                which under 1F1B is a separate callable that is handed nothing else.
                """
                type(self).calls += 1
                flat = hidden_states.reshape(-1, hidden_states.shape[-1])
                out = flat * 2.0 + probs.sum(dim=-1, keepdim=True)
                return out.view(hidden_states.shape), None

        layer.moe_ep_chunk_overlap = _StubRuntime()
        return layer

    @staticmethod
    def _input(layer):
        generator = torch.Generator(device="cuda").manual_seed(11)
        return torch.randn(
            (_SEQUENCE_LENGTH, _MICRO_BATCH_SIZE, _HIDDEN_SIZE),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )

    def test_split_execution_equals_the_single_call(self):
        """The headline: three phases threaded by intermediate_tensors == one unsplit call."""
        layer = self._layer_with_stub_runtime()
        hidden_states = self._input(layer)

        layer.fwd_execution_map = ["route", "expert_compute", "postprocess"]
        whole, whole_bias = layer(hidden_states)

        layer.fwd_execution_map = "route"
        routed, probs, shared = layer(hidden_states, intermediate_tensors=())
        layer.fwd_execution_map = "expert_compute"
        expert_out, split_bias = layer(None, intermediate_tensors=(routed, probs))
        layer.fwd_execution_map = "postprocess"
        split = layer(None, intermediate_tensors=(expert_out, shared))

        assert split.shape == whole.shape
        assert torch.equal(
            split, whole
        ), f"split != whole: {int(torch.ne(split, whole).sum())}/{whole.numel()} differ"
        assert split_bias == whole_bias

    def test_the_split_really_went_through_the_overlap_runtime(self):
        """Guards the test itself: a stub that never fired would make the above vacuous."""
        layer = self._layer_with_stub_runtime()
        hidden_states = self._input(layer)
        before = type(layer.moe_ep_chunk_overlap).calls

        layer.fwd_execution_map = "route"
        routed, probs, _shared = layer(hidden_states, intermediate_tensors=())
        layer.fwd_execution_map = "expert_compute"
        layer(None, intermediate_tensors=(routed, probs))

        assert type(layer.moe_ep_chunk_overlap).calls == before + 1

    def test_padding_mask_is_still_refused(self):
        """The one per-forward refusal that was NOT conservative: it stays."""
        layer = self._layer_with_stub_runtime()
        hidden_states = self._input(layer)
        mask = torch.ones((_MICRO_BATCH_SIZE, _SEQUENCE_LENGTH), dtype=torch.bool, device="cuda")

        with pytest.raises(NotImplementedError, match="padding_mask"):
            layer(hidden_states, padding_mask=mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSelectiveRecomputeAdmission:
    """Which ``--recompute-modules`` entries survive ``MoELayer.__init__`` on this path.

    'moe' and full-layer recompute re-run the facade's forward inside the backward and SILENTLY
    misorder the chunked wgrad FIFO, so both are refused. 'moe_act' is not that: it recomputes
    inside TE's own fused backward, never re-entering the schedule, so it must be admitted and
    must arrive on the experts, where the package reads it.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _config(self, modules: list[str]) -> TransformerConfig:
        return _make_config(
            moe_ep_chunk_overlap=True,
            moe_grouped_gemm=True,
            recompute_granularity='selective',
            recompute_modules=modules,
        )

    def test_moe_act_is_admitted_and_arrives_on_the_experts(self):
        layer = _make_grouped_layer(self._config(['moe_act']))

        assert layer.moe_ep_chunk_overlap is not None
        # The one attribute `build_expert_ops` reads to decide what to ask TE for.
        assert layer.experts.activation_recompute is True

    def test_the_experts_carry_no_recompute_request_when_moe_act_is_absent(self):
        layer = _make_grouped_layer(self._config(['core_attn']))

        assert layer.experts.activation_recompute is False

    def test_moe_is_still_refused(self):
        """The contrast that makes the admission above a decision rather than an absent check.

        Refused while BUILDING THE CONFIG, not while building the layer: the check lives in
        ``TransformerConfig.__post_init__`` so a bad config is rejected once, up front.
        """
        with pytest.raises(ValueError, match="'moe' in recompute_modules"):
            self._config(['moe'])

    def test_full_granularity_is_still_refused(self):
        """Also refused at config time. ``recompute_method`` and ``recompute_num_layers`` are
        supplied because ``__post_init__`` requires them of any non-selective granularity, and
        this test is about the overlap refusal rather than that pre-existing one."""
        with pytest.raises(ValueError, match="recompute_granularity='full'"):
            _make_config(
                moe_ep_chunk_overlap=True,
                moe_grouped_gemm=True,
                recompute_granularity='full',
                recompute_method='uniform',
                recompute_num_layers=1,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestExpertGTPWeightRematAdmission:
    """EGTP is ADMITTED at construction, inside a narrow envelope that is refused by name.

    The schedule never shows TE a distributed weight (``release_expert_weights`` keeps the
    full-shaped quantized shadow and hangs a full-sized staging buffer off it as ``main_grad``),
    so ``delay_wgrad_compute=True`` stays legal. Two conditions remain, each a raise because
    each alternative is SILENT:

    * one local expert -- TE's ``num_groups == 1`` shortcuts read past ``sum(split_sizes)``, so
      expert wgrads come back wrong or NaN with a bit-identical forward and dgrad;
    * CUDA-graph capture that REACHES the routed experts -- whole-layer, ``moe``, or
      full_iteration -- because then the routed experts land on GTP's GRAPHED chain and
      ``allocate_graph_wgrad_rings`` hands their accumulator a ring slot that is process-shared
      at ring_size 2, never zeroed and unfenced. Partial capture above the experts is admitted.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _egtp_config(**overrides):
        config = _make_config(moe_ep_chunk_overlap=True, moe_grouped_gemm=True, **overrides)
        # Set post-construction: the field is derived in ModelParallelConfig.__post_init__ from
        # --expert-tensor-parallel-num-weight-shards, which _make_config does not plumb.
        config.expert_gtp_weight_remat_size = 2
        return config

    @classmethod
    def _one_local_expert_config(cls):
        """``num_local_experts == 1``: ``num_moe_experts // ep_size`` with EP=1 at world 1.
        ``moe_router_pre_softmax`` is the config's own requirement at topk 1, not this path's."""
        return cls._egtp_config(num_moe_experts=1, moe_router_topk=1, moe_router_pre_softmax=True)

    @staticmethod
    def _set_have_gtp(monkeypatch, value: bool) -> None:
        """Patch the flag at its source module: ``moe_layer`` imports it INSIDE the branch."""
        monkeypatch.setattr(gtp_api, "HAVE_GTP", value)

    def test_expert_gtp_weight_remat_is_admitted(self, monkeypatch):
        """T1a. The headline: EGTP no longer raises at construction on a supported build."""
        self._set_have_gtp(monkeypatch, True)
        layer = _make_grouped_layer(self._egtp_config())

        assert layer.num_local_experts >= 2, "the geometry this admission test needs"
        assert layer.moe_ep_chunk_overlap is not None

    def test_one_local_expert_is_refused(self, monkeypatch):
        """T1b. TE's ``num_groups == 1`` wgrad corruption is silent, so the floor is a raise."""
        self._set_have_gtp(monkeypatch, True)
        with pytest.raises(RuntimeError, match="at least 2 local experts"):
            _make_grouped_layer(self._one_local_expert_config())

    @pytest.mark.parametrize(
        "modules",
        [
            [CudaGraphModule.moe_router],
            [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
            [CudaGraphModule.attn, CudaGraphModule.moe_router],
        ],
        ids=["router", "router+preprocess", "attn+router"],
    )
    @pytest.mark.parametrize("impl", ["transformer_engine", "local"])
    def test_partial_capture_before_the_experts_is_admitted(self, monkeypatch, impl, modules):
        """Both partial mechanisms leave the experts eager: transformer_engine early-returns
        above ``expert_compute``, local splits the layer and captures only the outer phases."""
        self._set_have_gtp(monkeypatch, True)
        config = self._egtp_config()
        config.cuda_graph_impl = impl
        config.cuda_graph_modules = modules

        assert _make_grouped_layer(config).moe_ep_chunk_overlap is not None

    def test_full_iteration_is_admitted(self, monkeypatch):
        """full_iteration captures the experts, but allocate_graph_wgrad_rings early-returns for
        it, so no ring slot exists on any chain -- measured 0 slots with EGTP live, and the
        AG/RS join lands inside the capture via finalize_model_grads."""
        self._set_have_gtp(monkeypatch, True)
        config = self._egtp_config()
        config.cuda_graph_impl = "full_iteration"
        config.cuda_graph_modules = []

        assert _make_grouped_layer(config).moe_ep_chunk_overlap is not None

    def test_cuda_graphs_are_admitted_without_egtp(self):
        """The contrast: the CUDA-graph refusal belongs to EGTP, not to the overlap path."""
        config = _make_config(moe_ep_chunk_overlap=True, moe_grouped_gemm=True)
        config.cuda_graph_impl = "local"

        assert _make_grouped_layer(config).moe_ep_chunk_overlap is not None

    def test_the_default_still_builds(self):
        """The contrast that makes the envelope a decision rather than a broken constructor."""
        config = _make_config(moe_ep_chunk_overlap=True, moe_grouped_gemm=True)

        assert config.expert_gtp_weight_remat_size == 1
        assert _make_grouped_layer(config).moe_ep_chunk_overlap is not None

    def test_no_egtp_message_cites_a_document_that_does_not_exist(self, monkeypatch):
        """``docs/moe_ep_overlap_egtp_support_plan.md`` has never existed in the tree or in git
        history, and this was its one USER-FACING citation."""
        phantom = "moe_ep_overlap_egtp_support_plan"
        assert phantom not in TestExpertGTPWeightRematAdmission.__doc__

        self._set_have_gtp(monkeypatch, True)
        with pytest.raises(RuntimeError) as excinfo:
            _make_grouped_layer(self._one_local_expert_config())
        assert phantom not in str(excinfo.value)


# Big enough that a chunk's saved tensors clear the 1 Mi-element default offload floor, and legal
# under the overlap config: hidden is a multiple of 256 and ffn a multiple of 128.
_OFFLOAD_HIDDEN_SIZE = 3072
_OFFLOAD_FFN_HIDDEN_SIZE = 6144
_OFFLOAD_NUM_TOKENS = 1024


def _linear_expert_math(hidden, topk_indices, topk_weights, fc1_slices, fc2_slices):
    """The same weighted squared-ReLU MoE as ``_expert_math`` without the ``[T, K, I, H]`` gather:
    memory-linear in the token count, so the saved tensors are ``[T, H]`` and ``[T, I]``."""
    output = torch.zeros_like(hidden)
    for expert, (fc1_weight, fc2_weight) in enumerate(zip(fc1_slices, fc2_slices)):
        # Top-k experts are distinct per token, so this sums at most one of the k weights.
        selected = (topk_indices == expert).to(topk_weights.dtype)
        weight_per_token = (topk_weights * selected).sum(dim=1, keepdim=True)
        activation = torch.relu(hidden @ fc1_weight.t()) ** 2
        output = output + (activation @ fc2_weight.t()) * weight_per_token
    return output


class _OffloadingRuntime(_AutogradRuntime):
    """``_AutogradRuntime`` whose compute stage opens the REAL per-chunk offload group, as
    ``run_expert_forward`` does: on the ambient stream, under ``enable_grad``, in the Function."""

    def routed_experts_compute_forward(self, dispatched):
        offload = self._experts.offload_fused_group_mlp
        self.offload_log.append(offload)
        dispatched_tokens = dispatched.recv_x.detach().float().requires_grad_(True)
        routing_probs = dispatched.recv_probs.detach().float().requires_grad_(True)
        stream = torch.cuda.current_stream(self.device)
        offload_group = off_interface(offload, dispatched_tokens, "fused_group_mlp")
        with torch.cuda.stream(stream), torch.enable_grad():
            topk_weights, topk_indices = torch.topk(
                routing_probs, k=self._router_topk, dim=-1, sorted=True
            )
            # Sliced and opted out BEFORE the group, as the real path opts its quantized weight
            # shadows out: a weight saved per chunk would be offloaded once per chunk.
            fc1_slices = [weight for weight in self.fc1_weight]
            fc2_slices = [weight for weight in self.fc2_weight]
            for weight in fc1_slices + fc2_slices:
                off_interface.mark_not_offload(weight)
            with offload_group as group_input:
                output = _linear_expert_math(
                    group_input, topk_indices, topk_weights, fc1_slices, fc2_slices
                )
            # The commit is the root the backward starts from, so it is what `saved` carries.
            output = offload_group.group_offload(output)
        return output.detach().to(dispatched.recv_x.dtype), CachedForwardChunkActivation(
            dispatched_tokens=dispatched_tokens,
            routing_probs=routing_probs,
            fc2_output=output,
            dispatch_graph=dispatched.graph,
        )


class _OffloadProbe:
    """Counters and failures collected by ``_install_offload_probe``."""

    def __init__(self):
        self.offload_calls = 0
        self.compared = 0
        # Bytes this probe actually copied out. `offload_summary_bytes` counts only groups left
        # offloadable, so it reads 0 at num_chunks=1 where the single group is marked resident.
        self.offload_bytes = 0
        self.failures: list[str] = []
        self.references: dict[int, torch.Tensor] = {}


def _install_offload_probe(monkeypatch) -> _OffloadProbe:
    """Poison every pinned buffer before its D2H, delay that D2H, and check every reload: a bare
    comparison would only test a timing race that a clean run proves nothing about."""
    probe = _OffloadProbe()
    original_reload = ChunkOffloadHandler.reload

    def offload(handler, src_tensor, pin_memory=True, use_cpu_pool=True):
        # Loud if the group-name list in `OffloadTensorGroup.__init__` is renamed: the CPU-pool
        # branch has a free-before-H2D of its own that this path must never reach.
        assert use_cpu_pool is False, "fused_group_mlp must not use the shared CPU pool"
        if not src_tensor.is_contiguous():
            src_tensor = src_tensor.contiguous()
        cpu_backup = torch.empty(
            src_tensor.shape, dtype=src_tensor.dtype, device="cpu", pin_memory=pin_memory
        )
        cpu_backup.fill_(float("nan") if cpu_backup.is_floating_point() else -1)
        # Lands on d2h_stream, whose context `bulk_offload_group` already holds, so the copy
        # below cannot have completed by the time the backward reaches this group.
        torch.cuda._sleep(int(2e7))
        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        probe.offload_calls += 1
        probe.offload_bytes += src_tensor.numel() * src_tensor.element_size()
        probe.references[id(cpu_backup)] = src_tensor.clone()
        # FOUR elements, as the real `offload` returns: a 3-tuple raised ValueError unpacking
        # `view_meta` before comparing anything -- SILENTLY. None: contiguous() bars alias views.
        return (src_tensor.device, cpu_backup, use_cpu_pool, None)

    def reload(handler, state, non_blocking=None):
        tensor = original_reload(handler, state, non_blocking=non_blocking)
        reference = probe.references.pop(id(state[1]), None)
        if reference is not None:
            reference.record_stream(torch.cuda.current_stream())
            probe.compared += 1
            # `torch.equal` syncs the CURRENT stream only and never d2h_stream, so a reload that
            # races its own D2H stays observable here instead of being synchronized away.
            if not torch.equal(tensor, reference):
                probe.failures.append(
                    f"reloaded tensor {tuple(tensor.shape)} differs from the offloaded source"
                )
        return tensor

    monkeypatch.setattr(ChunkOffloadHandler, "offload", offload)
    monkeypatch.setattr(ChunkOffloadHandler, "reload", reload)
    return probe


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(cpu_offload is None, reason="TE CPU offload hooks are not available")
@pytest.mark.xfail(
    strict=True,
    reason="Upstream defect: fine_grained_activation_offload's inline tensor_pop reload enqueues "
    "its H2D with no wait on the group's D2H event, so a backward reaching a group before any "
    "prefetch ran reads the pinned buffer unfilled. Every group takes that path because "
    "bulk_reload_group never drains _groups_to_reload. Strict, so this flags itself for removal "
    "once the fix lands: NVIDIA/Megatron-LM#7399.",
)
class TestOffloadedExpertComputeRoundTrip:
    """Every activation the per-chunk group offloads must come back bit-identical."""

    @pytest.fixture(autouse=True)
    def _monkeypatch(self, monkeypatch):
        self.monkeypatch = monkeypatch

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=1234, data_parallel_random_init=False)
        # The manager is a process-wide singleton; a leaked one poisons every later test here.
        off_interface.reset_instance()
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.generator = torch.Generator(device=self.device).manual_seed(11)
        self.fc1_weight = (
            torch.randn(
                (_NUM_EXPERTS, _OFFLOAD_FFN_HIDDEN_SIZE, _OFFLOAD_HIDDEN_SIZE),
                generator=self.generator,
                device=self.device,
            )
            * 0.02
        ).requires_grad_(True)
        self.fc2_weight = (
            torch.randn(
                (_NUM_EXPERTS, _OFFLOAD_HIDDEN_SIZE, _OFFLOAD_FFN_HIDDEN_SIZE),
                generator=self.generator,
                device=self.device,
            )
            * 0.02
        ).requires_grad_(True)

    def teardown_method(self, method):
        off_interface.reset_instance()
        Utils.destroy_model_parallel()

    def _make_facade(self, num_chunks: int) -> MoEEPChunkOverlapRuntime:
        """The offloading runtime, with the offload gate ON."""
        self.wgrad_log: list = []
        runtime_cls = type(
            "_BoundOffloadingRuntime",
            (_OffloadingRuntime,),
            {
                "fc1_weight": self.fc1_weight,
                "fc2_weight": self.fc2_weight,
                "wgrad_log": self.wgrad_log,
            },
        )
        self.monkeypatch.setattr(overlap_module, "MoEEPChunkOverlapRuntime", runtime_cls)
        _patch_schedule_stages(self.monkeypatch)
        _install_publish_probe(self.monkeypatch, [])
        return _build_facade(
            hidden_size=_OFFLOAD_HIDDEN_SIZE,
            ffn_hidden_size=_OFFLOAD_FFN_HIDDEN_SIZE,
            num_chunks=num_chunks,
            offload=True,
        )

    def _make_inputs(self):
        """Fresh inputs on EVERY iteration: reused ones make a stale reload bit-identical."""
        hidden_shape = (_OFFLOAD_NUM_TOKENS, 1, _OFFLOAD_HIDDEN_SIZE)
        return _make_inputs(self.generator, self.device, hidden_shape, _OFFLOAD_NUM_TOKENS)

    @pytest.mark.parametrize("min_offloaded_tensor_size", [0, 1024 * 1024])
    @pytest.mark.parametrize("num_chunks", [1, 2])
    def test_offloaded_activations_come_back_unchanged(self, num_chunks, min_offloaded_tensor_size):
        """The gate: warmup included, and with no synchronize between forward and backward."""
        probe = _install_offload_probe(self.monkeypatch)
        facade = self._make_facade(num_chunks=num_chunks)
        manager = PipelineOffloadManager.get_instance()
        resident_bytes = 0

        for _iteration in range(3):
            off_interface.init_chunk_handler(
                pp_rank=0,
                vp_size=1,
                vp_stage=0,
                min_offloaded_tensor_size=min_offloaded_tensor_size,
                delta_offload_bytes_across_pp_ranks=0,
                activation_offload_fraction=1.0,
            )
            hidden, dense_probs = self._make_inputs()
            output, _ = facade.forward(hidden, dense_probs, _routing_map_of(dense_probs))
            # No synchronize here: that is what retires every D2H before any H2D can race it.
            output.float().pow(2).sum().backward()
            resident_bytes = manager.offload_summary_bytes.get("fused_group_mlp", 0)
            off_interface.reset()

        failures = list(probe.failures)
        # Non-vacuity first: a probe that moved no bytes would report no failures either.
        if probe.offload_bytes <= 0:
            failures.append("the fused_group_mlp group offloaded nothing")
        # Distinct from the probe's count, which includes warmup: `should_bulk_offload` offloads
        # unconditionally there, so only this sees a group left un-offloadable in steady state.
        if num_chunks > 1 and resident_bytes <= 0:
            failures.append("no fused_group_mlp group stayed offloadable after warmup")
        if probe.offload_calls <= 0 or probe.compared <= 0:
            failures.append(
                f"probe never fired: {probe.offload_calls} offloads, {probe.compared} reloads"
            )
        if facade.offload_log != [True] * (num_chunks * 3):
            failures.append(f"the offload gate did not reach every chunk: {facade.offload_log}")

        # Scored across ranks: raising rank-locally strands the others on the teardown barrier.
        gathered: list = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, failures)
        assert all(not per_rank for per_rank in gathered), f"offload round trip failed: {gathered}"
