# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The expert-compute seam, exercised with no GPU, no DeepEP and no TransformerEngine.

``test_moe_ep_chunk_overlap.py`` fakes the whole runtime, so nothing there reaches
:meth:`MoEEPChunkOverlapRuntime.routed_experts_compute_forward` or the runtime's construction.
This file covers the seam itself: the guarantees the runtime makes on the expert compute's behalf,
each of them "silent rather than loud when broken". Each is made loud by a stand-in that does the
least the seam allows -- records its arguments, declares nothing to the allocator, touches no
stream of its own -- so a guarantee that drifted into the TransformerEngine functions fails here
instead of becoming an intermittent wrong answer on eight GPUs: a dropped ``record_stream`` was
measured at ``rel = 4.6e+02`` with nothing raised.

The substitution is by NAME -- on the module for ``run_expert_forward`` and
``run_expert_backward``, on :class:`MoEEPChunkOverlapRuntime` for the three wgrad stages, which are
methods -- so ``monkeypatch.setattr`` raises if one is renamed or moved out: the rename detector
as well as the substitution. That the runtime wires itself to TE's functions and
no others is asserted in :class:`TestRuntimeConstruction`; that TE's op shell is built exactly
once, in :class:`TestOpShellIsBuiltOnce`.

CPU-only and single-process on purpose: no ``torch.distributed`` init and no CUDA context.
``torch.cuda.current_stream`` becomes a sentinel whose identity is all anyone reads, and the
dispatch payload is a ``torch.Tensor`` subclass recording ``record_stream`` instead of needing a
CUDA allocation. Both substitutions sit OUTSIDE the code under test: the two
``routed_experts_compute_*`` methods that run are the production ones.
"""

import ast
import inspect
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.pipeline_parallel.fine_grained_activation_offload import _te_do_not_offload
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_ep_chunk_overlap import ep_chunk_overlap
from megatron.core.transformer.moe.moe_ep_chunk_overlap.ep_chunk_overlap import (
    CachedForwardChunkActivation,
    DispatchedChunk,
    MoEEPChunkOverlapRuntime,
    _MoEEPChunkOverlapConfig,
)

_HIDDEN_SIZE = 256  # TE fused grouped MLP needs % 64, MXFP8 % 32; 256 is the only measured value
_FFN_HIDDEN_SIZE = 128
_NUM_EXPERTS = 2
_ROUTER_TOPK = 2
_NUM_LOCAL_TOKENS = 8
_NUM_CHUNKS = 2
_WORLD_SIZE = 1
_NUM_COMM_SMS = 32


def _make_config() -> _MoEEPChunkOverlapConfig:
    """The smallest geometry the config validator accepts."""
    return _MoEEPChunkOverlapConfig(
        num_local_tokens=_NUM_LOCAL_TOKENS,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        num_experts=_NUM_EXPERTS,
        router_topk=_ROUTER_TOPK,
        num_token_chunks=_NUM_CHUNKS,
        world_size=_WORLD_SIZE,
        num_comm_SMs=_NUM_COMM_SMS,
        # Spelled as the old derivation so _CAPACITY below is unchanged by the field's arrival.
        num_local_experts=_NUM_EXPERTS // _WORLD_SIZE,
    )


#: Expanded rows one dispatch produces for that geometry -- read from the config rather than
#: written down, so the padding formula stays the package's business and not this file's.
_CAPACITY = _make_config().max_num_expanded_tokens_per_chunk


class _Opaque:
    """A value the runtime must carry without reading. Any attribute access IS the failure."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attribute: str) -> object:
        if attribute.startswith("__"):
            # Let Python's own protocols (repr helpers, copy, pickle) fail the normal way.
            raise AttributeError(attribute)
        raise AssertionError(f"the runtime read {attribute!r} off the opaque {self._name}")

    def __repr__(self) -> str:
        return f"<opaque {self._name}>"


class _AmbientStream:
    """Stand-in for the caller's compute stream; identity is the only thing anyone reads."""

    def __repr__(self) -> str:
        return "<ambient compute stream>"


class _RecordingTensor(torch.Tensor):
    """A CPU tensor that records ``record_stream`` instead of needing a CUDA allocation.

    The prologue under test is a sequence of ``record_stream`` calls on buffers DeepEP allocated
    on its own stream. Recording them lets the real ``routed_experts_compute_forward`` run with
    no CUDA context, and a dropped claim is what this file exists to catch.
    """

    @staticmethod
    def wrapping(data: torch.Tensor, role: str, seam: "_Seam") -> "_RecordingTensor":
        """Return ``data`` as a recording tensor labelled ``role``, logging into ``seam``."""
        tensor = data.as_subclass(_RecordingTensor)
        tensor.role = role
        tensor.seam = seam
        return tensor

    def record_stream(self, stream) -> None:
        """Record the claim the runtime is making on this buffer."""
        # Views of this tensor inherit neither attribute, so a claim made on one raises here
        # rather than being counted as a claim on the tensor it aliases.
        self.seam.stream_claims.append((self.role, stream))
        self.seam.log.append(f"record_stream:{self.role}")


class _FakeExpertCompute:
    """An expert compute that does the least the seam allows.

    It records its arguments, allocates nothing, declares nothing to the allocator and touches no
    stream of its own, so everything the runtime guarantees has to be visible from outside it.
    Every method takes its arguments exactly as the module function it stands in for is called,
    so a kwarg renamed on the runtime's side of the seam raises ``TypeError`` here.
    """

    def __init__(self, log: list) -> None:
        self._log = log
        self.forward_calls: list = []
        self.backward_calls: list = []
        self.wgrad_calls: list = []
        #: One entry per ``run_expert_forward``: the saved chunk it built for that call.
        self.saved_returns: list = []
        #: One entry per ``run_expert_backward``: that chunk's routing gradient. The seam requires
        #: a FRESH flat tensor per call, so no later chunk holds a reference that can overwrite it.
        self.dprob_returns: list = []

    def run_expert_forward(self, runtime, *, dispatched, stream):
        """Record the call and return one row of nothing per received row.

        The chunk arrives as one ``DispatchedChunk`` now that the schedule keeps the dispatch
        NODE rather than a transport handle. Its fields are unpacked into the recorded call so
        the storage-aliasing assertions below still see them by name; ``dispatched`` itself is
        kept for the identity checks and is not a tensor, so it never enters that comparison.
        """
        self._log.append("run_expert_forward")
        self.forward_calls.append(
            dict(
                runtime=runtime,
                dispatched=dispatched,
                recv_x=dispatched.recv_x,
                recv_probs=dispatched.recv_probs,
                split_sizes=dispatched.split_sizes,
                stream=stream,
            )
        )
        self.saved_returns.append(_make_saved(dispatch_graph=dispatched.graph))
        return torch.zeros((dispatched.recv_x.shape[0], _HIDDEN_SIZE)), self.saved_returns[-1]

    def run_expert_backward(self, *, recv_x, saved, stream):
        """Record the call; return dX and a FRESH flat routing gradient, as the seam requires."""
        self._log.append("run_expert_backward")
        self.backward_calls.append(dict(recv_x=recv_x, saved=saved, stream=stream))
        self.dprob_returns.append(torch.arange(_CAPACITY, dtype=torch.float32))
        return torch.zeros((_CAPACITY, _HIDDEN_SIZE)), self.dprob_returns[-1]

    def run_chunk_wgrad(self, runtime, *, stream) -> None:
        """Record the call; the weight gradients themselves are not this file's subject."""
        self._log.append("run_chunk_wgrad")
        self.wgrad_calls.append(dict(runtime=runtime, stream=stream))

    def begin_wgrad_accumulation(self, runtime) -> None:
        """Ready this backward's accumulation buffers. A side effect and no value."""
        self._log.append("begin_wgrad_accumulation")
        self.wgrad_calls.append(dict(runtime=runtime))

    def end_wgrad_accumulation(self, runtime) -> None:
        """Flush the EGTP staging buffers. A side effect and no value; inert without EGTP."""
        self._log.append("end_wgrad_accumulation")
        self.wgrad_calls.append(dict(runtime=runtime))


class _Seam:
    """The two collaborators the runtime talks to, plus the one log that orders their calls."""

    def __init__(self) -> None:
        self.log: list = []
        self.stream_claims: list = []
        self.experts = _FakeExperts()
        self.expert_compute = _FakeExpertCompute(self.log)

    def install(self, monkeypatch) -> None:
        """Put this seam's expert compute where the runtime reads TransformerEngine's.

        By NAME, where each is looked up at call time, so ``monkeypatch.setattr`` raises on a
        rename -- the rename detector as well as the substitution. The two expert-compute
        functions are still module-level; the three wgrad stages are METHODS, so they are patched
        on the class. They are patched although nothing here should call them: being able to say
        so is the assertion. ``publish_wgrad`` is absent because it no longer exists --
        ``moe_backward`` calls TE's ``get_dummy_wgrads_for_params`` directly, asserted over the
        real backward in ``test_moe_ep_chunk_overlap.py``.
        """
        compute = self.expert_compute
        # Spelled out rather than looked up by string, so a name that exists on only one side is
        # a NameError here instead of a silently unpatched production function.
        for name, replacement in (
            ("run_expert_forward", compute.run_expert_forward),
            ("run_expert_backward", compute.run_expert_backward),
        ):
            monkeypatch.setattr(ep_chunk_overlap, name, replacement)
        # A bound method put on a CLASS is not re-bound on access, so the runtime never reaches the
        # fake. These thunks take it as ``self`` and pass it on as the ``runtime`` positional.
        for name, replacement in (
            (
                "run_chunk_wgrad",
                lambda runtime, *, stream: compute.run_chunk_wgrad(runtime, stream=stream),
            ),
            ("begin_wgrad_accumulation", lambda runtime: compute.begin_wgrad_accumulation(runtime)),
            ("end_wgrad_accumulation", lambda runtime: compute.end_wgrad_accumulation(runtime)),
        ):
            monkeypatch.setattr(MoEEPChunkOverlapRuntime, name, replacement)


class _StubbedRuntime(MoEEPChunkOverlapRuntime):
    """The production runtime with only its constructor replaced.

    Deliberately does NOT call ``super().__init__``: that builds a DeepEP buffer, checks
    TransformerEngine's fused-op registration and makes a CUDA stream, none of which exists in a
    CPU unit-test process. Every method the tests below call is inherited and unmodified.
    """

    def __init__(self, config: _MoEEPChunkOverlapConfig, seam: _Seam) -> None:
        self.config = config
        self.device = torch.device("cpu")
        self._experts = seam.experts
        # The schedule's own collectives, which nothing here issues: the two
        # `routed_experts_compute_*` methods under test never read them.
        self._managers = ()
        # Cached exactly as the production constructor caches it: `quantize_expert_weights`
        # and the wgrad stages read the tuple off the runtime rather than rebuilding it.
        self.expert_params = ep_chunk_overlap.expert_weight_parameters(
            seam.experts, config.num_local_experts
        )
        # The op-shell cache, empty exactly as the production constructor leaves it: the shell is
        # built by the first forward, and `expert_ops` is the only thing that may build one.
        self._ops = None
        self._fc1_op = None
        self._fc2_op = None
        # The schedule owns the collectives, and nothing here issues one. No per-chunk buffers:
        # the production runtime has none either, which is what this stub has to keep mirroring.
        self.comm_stream = None


def _make_runtime(seam: _Seam, monkeypatch) -> MoEEPChunkOverlapRuntime:
    """Build the production runtime around ``seam``'s fakes.

    Annotated with the real class so the static contract checker resolves every
    ``runtime.<method>(...)`` below against the package rather than against the stub.
    """
    seam.install(monkeypatch)
    return _StubbedRuntime(_make_config(), seam)


def _runtime_over(experts: torch.nn.Module) -> MoEEPChunkOverlapRuntime:
    """A stubbed runtime over one particular experts module, with nothing else substituted."""
    seam = _Seam()
    seam.experts = experts
    return _StubbedRuntime(_make_config(), seam)


def _make_dispatched(seam: _Seam) -> DispatchedChunk:
    """One chunk's receive, with payload tensors that record the runtime's allocator claims.

    ``graph`` is opaque: the runtime only carries it into the saved chunk, where the SCHEDULE
    differentiates it. Anything this seam touched on it would be a claim the schedule has
    already made on the comm stream, so an ``_Opaque`` here is the assertion that it does not.
    """
    return DispatchedChunk(
        recv_x=_RecordingTensor.wrapping(torch.zeros((_CAPACITY, _HIDDEN_SIZE)), "recv_x", seam),
        recv_probs=_RecordingTensor.wrapping(torch.zeros((_CAPACITY,)), "recv_probs", seam),
        split_sizes=_RecordingTensor.wrapping(
            torch.tensor([_CAPACITY // 2, _CAPACITY // 2]), "split_sizes", seam
        ),
        graph=_Opaque("dispatch graph"),
    )


def _make_grad_recv(seam: _Seam) -> torch.Tensor:
    """The backward dispatch's receive: one tensor now, not a whole chunk.

    The backward takes the gradient the schedule got by differentiating the combine node, so
    there is no second ``DispatchedChunk`` and no second set of routing metadata to claim.
    """
    return _RecordingTensor.wrapping(torch.zeros((_CAPACITY, _HIDDEN_SIZE)), "grad_recv", seam)


def _make_saved(*, dispatch_graph: object | None = None) -> CachedForwardChunkActivation:
    """What a forward chunk hands its backward: three opaque tensors and the dispatch node."""
    return CachedForwardChunkActivation(
        dispatched_tokens=_Opaque("dispatched tokens"),
        routing_probs=_Opaque("routing probs"),
        fc2_output=_Opaque("fc2 output"),
        dispatch_graph=_Opaque("dispatch graph") if dispatch_graph is None else dispatch_graph,
    )


class _RecordingConstruction:
    """Stands in for something the runtime builds: records the call and returns one marker."""

    def __init__(self, name: str, log: list) -> None:
        self._name = name
        self._log = log
        self.calls: list = []
        self.instance = _Opaque(name)

    def __call__(self, *args, **kwargs):
        """Record this construction, in order, and hand back the marker instance."""
        self._log.append(self._name)
        self.calls.append((args, kwargs))
        return self.instance


class _FakeGroupedLinear(torch.nn.Module):
    """One linear's per-expert ``weight{i}`` parameters and the geometry ``_make_fused_ops``
    hands TE, which is all of a linear either of them resolves."""

    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.num_gemms = _NUM_EXPERTS
        self.in_features = in_features
        self.out_features = out_features
        for index in range(_NUM_EXPERTS):
            self.register_parameter(
                f"weight{index}", torch.nn.Parameter(torch.zeros((out_features, in_features)))
            )


class _DuckTypedExperts(torch.nn.Module):
    """Everything the package READS off ``experts``, on a module that is not a ``TEGroupedMLP``.

    The package builds no TransformerEngine ops of its own any more: it BORROWS the experts' own
    sequence and CALLS the experts' own forward. So this stand-in has to be callable and has to
    carry ``_fused_ops`` / ``_make_fused_ops``, and the construction below mirrors
    ``TEGroupedMLP._make_fused_ops`` (experts.py:548-720) -- TE's and the trunk's behaviour, not
    the package's. That construction moving in here is a real cost: nothing in this file now
    pins WHICH activation op the production path picks, only that it checks whatever it got.
    """

    def __init__(self, *, gated: bool = False, activation_recompute: bool = False) -> None:
        # `torch.nn.Module.__init__`, never `super()`: `_FakeExperts` below mixes this with
        # `TEGroupedMLP`, whose own __init__ takes a config, a module spec and TransformerEngine.
        torch.nn.Module.__init__(self)
        # FC1 carries gate and up when gated, as ``TEGroupedMLP`` builds it (experts.py:218-221).
        fc1_out = 2 * _FFN_HIDDEN_SIZE if gated else _FFN_HIDDEN_SIZE
        self.linear_fc1 = _FakeGroupedLinear(fc1_out, _HIDDEN_SIZE)
        self.linear_fc2 = _FakeGroupedLinear(_HIDDEN_SIZE, _FFN_HIDDEN_SIZE)
        # The trunk's own 'moe_act' predicate, derived on the experts (experts.py:275-278) and
        # read back by ``_make_fused_ops`` at experts.py:637.
        self.activation_recompute = activation_recompute
        self.config = SimpleNamespace(
            activation_func=F.silu if gated else squared_relu,
            gated_linear_unit=gated,
            activation_func_clamp_value=None,
            moe_mlp_glu_interleave_size=None,
        )
        # In a TUPLE and None until the first build, exactly as experts.py:351 leaves it: a bare
        # module here registers the op shells as submodules and doubles every expert parameter.
        self._fused_ops: tuple | None = None
        #: One entry per ``_make_fused_ops`` call, so a second shell is counted, not inferred.
        self.make_fused_ops_calls = 0
        #: What the package asked the builder for, so the two overrides can be asserted on.
        self.fused_ops_overrides: tuple | None = None
        #: Per ``forward``: ``(hidden, tokens_per_expert, probs, the sequence already existed)``.
        self.forward_calls: list = []
        #: ``forward``'s second return value; a non-None one is what the seam has to refuse.
        self.fc2_bias: torch.Tensor | None = None

    def _make_activation_op(self, te_ops) -> torch.nn.Module:
        """The activation ``TEGroupedMLP`` picks for this config (experts.py:637-708)."""
        recompute = {"activation_recompute_in_mlp": self.activation_recompute}
        if self.config.gated_linear_unit and self.config.activation_func is F.silu:
            return te_ops.ScaledSwiGLU(
                glu_interleave_size=self.config.moe_mlp_glu_interleave_size, **recompute
            )
        if not self.config.gated_linear_unit and self.config.activation_func is squared_relu:
            return te_ops.ScaledSReLU(**recompute)
        raise RuntimeError("_make_fused_ops expected SwiGLU, quick_gelu, or weighted squared_relu")

    def make_fused_ops(
        self, *, delay_wgrad_compute=None, accumulate_into_main_grad=None
    ) -> torch.nn.Module:
        """``TEGroupedMLP``'s public builder, which is the seam the package borrows through.

        The overrides are recorded rather than applied: what the package must prove is that it
        asks for both, since neither global config flag is required of the model any more.
        """
        self.fused_ops_overrides = (delay_wgrad_compute, accumulate_into_main_grad)
        return self._make_fused_ops()

    def _make_fused_ops(self) -> torch.nn.Module:
        """TE's FC1/activation/FC2 container, carrying the experts' REAL weight parameters.

        ``register_grouped_linear_params`` (experts.py:554-573) puts each linear's own
        ``weight{i}`` onto the meta op shell, which is what makes a shadow written into that
        shell's ``__dict__`` hide a live parameter rather than a copy of one.
        """
        from transformer_engine.pytorch import ops as te_ops

        self.make_fused_ops_calls += 1
        linear_ops = []
        for linear in (self.linear_fc1, self.linear_fc2):
            weights = [linear.get_parameter(f"weight{i}") for i in range(linear.num_gemms)]
            op = te_ops.GroupedLinear(
                linear.num_gemms,
                linear.in_features,
                linear.out_features,
                bias=False,
                device="meta",
                dtype=weights[0].dtype,
                accumulate_into_main_grad=True,
                delay_wgrad_compute=True,
            )
            for index, weight in enumerate(weights):
                op.register_parameter(f"weight{index}", weight)
            linear_ops.append(op)
        return te_ops.Sequential(linear_ops[0], self._make_activation_op(te_ops), linear_ops[1])

    def forward(self, hidden_states, tokens_per_expert, permuted_probs):
        """``TEGroupedMLP.forward``'s shape: record the call, return ``(output, bias)``."""
        self.forward_calls.append(
            (hidden_states, tokens_per_expert, permuted_probs, self._fused_ops is not None)
        )
        output = torch.zeros((hidden_states.shape[0], _HIDDEN_SIZE), requires_grad=True)
        return output, self.fc2_bias

    #: ``TEGroupedMLP`` resolves this at construction; the schedule refuses to borrow unless it
    #: is True, because the unfused forward defers no weight gradient for it to drain.
    _with_fused_impl = True

    def _ensure_main_grad_for_fused_impl(self) -> None:
        """``TEGroupedMLP``'s own lazy ``main_grad`` hook, fired through the seam."""


class _FakeExperts(_DuckTypedExperts, TEGroupedMLP):
    """The same structure carrying the type the trunk actually builds, kept so these fakes stay
    honest about what a real ``TEGroupedMLP`` is -- the runtime no longer checks it."""


class _StubScaledSReLU(torch.nn.Module):
    """TE's ungated squared ReLU: the ONE op whose fused backward honours the recompute flag."""

    def __init__(self, *, activation_recompute_in_mlp: bool = False) -> None:
        super().__init__()
        self.activation_recompute_in_mlp = activation_recompute_in_mlp


class _StubScaledSwiGLU(torch.nn.Module):
    """TE's gated silu, modelled as TE 2.18 behaves: it ACCEPTS the recompute flag and its fused
    backward then SILENTLY drops it. (te220's own ``swiglu.py:473-477`` raises instead -- later,
    and TE's refusal rather than the package's, which is why the package refuses first.)"""

    def __init__(
        self, *, glu_interleave_size: int | None = None, activation_recompute_in_mlp: bool = False
    ) -> None:
        super().__init__()
        self.glu_interleave_size = glu_interleave_size
        self.activation_recompute_in_mlp = activation_recompute_in_mlp


class _StubScaledClampedQGeGLU(_StubScaledSwiGLU):
    """The other gated op the package can build; same acceptance, same silent drop on 2.18."""

    def __init__(self, *, alpha=None, limit=None, glu_linear_offset=None, **gated) -> None:
        super().__init__(**gated)
        self.alpha, self.limit, self.glu_linear_offset = alpha, limit, glu_linear_offset


#: MXFP8's four inner tensors, in ``prepare_for_saving`` order (mxfp8_tensor_storage.py:158-168).
#: An unrequested usage is present-and-``None``, never absent: that is how the boundary decides.
_MXFP8_INNER = ("_rowwise_data", "_columnwise_data", "_rowwise_scale_inv", "_columnwise_scale_inv")


class _StubQuantizedTensor(torch.Tensor):
    """What a TE quantizer returns: a tensor, and deliberately NOT a Parameter, so the offload
    hooks would manage it unless the package marks it.

    Carries MXFP8's four inner slots, because the ONE thing TE's fused dgrad reads off a saved
    weight is ``_columnwise_data`` / ``_columnwise_scale_inv`` (``grouped_mlp.py:2179-2180`` FC2,
    ``:2465-2466`` FC1); without them a rowwise-only gather looks like a both-usage one.
    """


def _as_quantized(tensor: torch.Tensor, *, rowwise: bool, columnwise: bool):
    """A stub MXFP8 tensor with exactly the usages asked for; the other slots are None.

    ``rowwise=True, columnwise=False`` is what GTP's FORWARD all-gather produces
    (``set_usage(rowwise=fwd, columnwise=not fwd)``, generalized_tensor_parallelism.py:1373), and
    it is the shape of defect D2.
    """
    shadow = tensor.detach().clone().as_subclass(_StubQuantizedTensor)
    shadow._rowwise_data = tensor.detach().clone() if rowwise else None
    shadow._rowwise_scale_inv = torch.zeros(4) if rowwise else None
    shadow._columnwise_data = tensor.detach().clone() if columnwise else None
    shadow._columnwise_scale_inv = torch.ones(4) if columnwise else None
    return shadow


class _StubQuantizer:
    """TE's per-weight quantizer: records what the package asks of it, returns one shadow."""

    def __init__(self) -> None:
        # TE's own default; an internal quantizer returns a storage with no requires_grad.
        self.internal = True
        self.usage: tuple | None = None

    def set_usage(self, *, rowwise: bool, columnwise: bool) -> None:
        """Record the usage demanded; columnwise is what the fused backward's wgrad reads."""
        self.usage = (rowwise, columnwise)

    def __call__(self, weight: torch.Tensor) -> "_StubQuantizedTensor":
        """One fresh shadow per call, so a reused or aliased one is visible to the tests.

        The usages honour whatever ``set_usage`` was last told, exactly as TE does: this branch
        is asked for BOTH, which is why a BF16 expert parameter's shadow serves the dgrad and
        needs no backward all-gather.
        """
        rowwise, columnwise = self.usage if self.usage is not None else (True, True)
        return _as_quantized(weight, rowwise=rowwise, columnwise=columnwise)


class _StubGroupedLinear(torch.nn.Module):
    """TE's grouped-linear op shell; the package registers the real expert weights onto it."""

    def __init__(self, num_gemms: int, in_features: int, out_features: int, **options) -> None:
        super().__init__()
        self.num_gemms, self.in_features, self.out_features = num_gemms, in_features, out_features
        self.options = options
        # None until the recipe state exists, as TE leaves it until the fuser's FIRST forward
        # (``ops/fuser.py:651-653``); making the package call for it is half of what is tested.
        self._quantizers: dict = {}
        self.recipes: list = []

    def get_quantizer(self, mode: str, index: int) -> "_StubQuantizer | None":
        """TE's accessor, absent-until-reset exactly as TE's is."""
        return self._quantizers.get((mode, index))

    def reset_recipe_state(self, *, recipe) -> None:
        """Build this op's quantizers, as TE's fuser would on its first forward."""
        self.recipes.append(recipe)
        self._quantizers = {("forward", i): _StubQuantizer() for i in range(2 * self.num_gemms)}


class _StubSequential(torch.nn.Module):
    """TE's op container, modelled on ``transformer_engine/pytorch/ops/sequential.py:82-107``.

    The ops are real submodules keyed by position, exactly as TE's ``append`` registers them, so
    ``children()`` yields the triple itself rather than a wrapper holding it -- which is how the
    package reaches FC1, the activation and FC2 off a sequence it did not build.
    """

    def __init__(self, *ops: torch.nn.Module) -> None:
        super().__init__()
        for index, op in enumerate(ops):
            self.add_module(str(index), op)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __getitem__(self, index: int) -> torch.nn.Module:
        return list(self._modules.values())[index]


class _StubFusedGLU:
    """The registration ``check_fused_grouped_mlp`` demands for a gated triple."""

    @staticmethod
    def is_supported() -> bool:
        """Registered, so the only thing left that can raise is the package's 'moe_act' rule."""
        return True


class _StubFusedUnary(_StubFusedGLU):
    """The same registration for the ungated triple."""


class _StubMXFP8Recipe:
    """The one recipe this path implements; ``autocast_recipe`` raises on any other."""

    def mxfp8(self) -> bool:
        """Yes, so a raise below can only come from the package's own rule."""
        return True


class _StubFP8GlobalStateManager:
    """The live-autocast lookup ``autocast_recipe`` reads. Enabled and MXFP8, as this path needs."""

    recipe = _StubMXFP8Recipe()

    @staticmethod
    def is_fp8_enabled() -> bool:
        """There is a live quantization autocast around this forward."""
        return True

    @classmethod
    def get_fp8_recipe(cls) -> _StubMXFP8Recipe:
        """The recipe that autocast carries, which the package inherits rather than invents."""
        return cls.recipe


def _install_fake_transformer_engine(monkeypatch) -> list:
    """Shadow every TransformerEngine module the expert-compute section imports lazily, and
    return the list ``mark_not_offload`` records into."""
    ops = ModuleType("transformer_engine.pytorch.ops")
    ops.GroupedLinear = _StubGroupedLinear
    ops.Sequential = _StubSequential
    ops.ScaledSReLU = _StubScaledSReLU
    ops.ScaledSwiGLU = _StubScaledSwiGLU
    ops.ScaledClampedQGeGLU = _StubScaledClampedQGeGLU

    grouped_mlp = ModuleType("transformer_engine.pytorch.ops.fused.grouped_mlp")
    grouped_mlp.GroupedMLP_CuTeGEMMGLU = _StubFusedGLU
    grouped_mlp.GroupedMLP_CuTeGEMMUnary = _StubFusedUnary
    grouped_mlp.ScaledSReLU = _StubScaledSReLU
    grouped_mlp._grouped_gemm_dsrelu_backward_supported = lambda: True
    grouped_mlp._nvidia_cudnn_frontend_supports_wgrad = lambda: True
    grouped_mlp.is_glu_activation = lambda op: isinstance(op, _StubScaledSwiGLU)
    grouped_mlp.validate_grouped_mlp_dims = lambda fc1_op, activation_op, fc2_op: None

    #: Every tensor handed to TE's ``mark_not_offload``, in call order.
    marks: list = []
    cpu_offload = ModuleType("transformer_engine.pytorch.cpu_offload")
    cpu_offload.mark_not_offload = marks.append

    common = ModuleType("transformer_engine.pytorch.ops._common")
    common.is_quantized_tensor = lambda tensor: isinstance(tensor, _StubQuantizedTensor)

    quantization = ModuleType("transformer_engine.pytorch.quantization")
    quantization.FP8GlobalStateManager = _StubFP8GlobalStateManager

    pytorch = ModuleType("transformer_engine.pytorch")
    pytorch.ops = ops
    pytorch.autocast = lambda *, enabled, recipe: nullcontext()
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", pytorch)
    for name, module in (
        ("transformer_engine.pytorch.ops.fused.grouped_mlp", grouped_mlp),
        ("transformer_engine.pytorch.cpu_offload", cpu_offload),
        ("transformer_engine.pytorch.ops._common", common),
        ("transformer_engine.pytorch.quantization", quantization),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return marks


class TestActivationRecomputeReachesTransformerEngine:
    """What ``'moe_act'`` in ``--recompute-modules`` does to the triple this path RUNS.

    TE honours ``activation_recompute_in_mlp`` on ``ScaledSReLU`` only
    (``grouped_mlp.py:1753-1758``). On TE 2.18 the two gated ops take the keyword and their fused
    backward SILENTLY drops it, saving the FC2 input anyway; on te220 they raise instead
    (``swiglu.py:473-477``) -- later, and TE's refusal rather than this package's.

    The package no longer BUILDS that triple: ``TEGroupedMLP._make_fused_ops`` does and the
    runtime borrows it. What is left to the package, and what these four pin, is the CHECK --
    ``expert_ops`` runs ``check_fused_grouped_mlp`` over the sequence it borrowed and refuses the
    combinations TE accepts and ignores. ACCEPTED-AND-IGNORED is invisible from the gradients,
    which are identical either way, so a borrowed-but-unchecked sequence would be silent.

    ``expert_ops`` runs for real here over a shadowed TE, which also keeps ``_DuckTypedExperts``
    honest: every attribute the package reads off ``experts`` is resolved for real, so one
    dropped from the stand-in is an ``AttributeError`` in this class.
    """

    @pytest.fixture(autouse=True)
    def _shadow_transformer_engine(self, monkeypatch):
        _install_fake_transformer_engine(monkeypatch)

    def test_the_flag_is_carried_onto_the_op_te_honours_it_on(self):
        """The ungated squared-ReLU triple: accepted, and cached as the ops the drain reads."""
        runtime = _runtime_over(_DuckTypedExperts(activation_recompute=True))

        sequential = runtime.expert_ops()

        fc1, activation, fc2 = list(sequential.children())
        assert (runtime._fc1_op, runtime._fc2_op) == (fc1, fc2), "the checked triple is TE's own"
        assert sequential is runtime._experts._fused_ops[0], "borrowed, never a second shell"
        assert isinstance(activation, _StubScaledSReLU)
        assert activation.activation_recompute_in_mlp is True

    def test_without_moe_act_the_same_triple_asks_te_for_nothing(self):
        """The negative arm: ``activation_recompute=False`` must not set the flag at all."""
        runtime = _runtime_over(_DuckTypedExperts(activation_recompute=False))

        activation = list(runtime.expert_ops().children())[1]
        assert activation.activation_recompute_in_mlp is False

    def test_a_gated_activation_with_moe_act_is_refused(self):
        """On 2.18 ``ScaledSwiGLU`` takes the flag and ignores it, so the package refuses first --
        over a sequence it did not build, which is the whole of what it still owns here."""
        runtime = _runtime_over(_DuckTypedExperts(gated=True, activation_recompute=True))

        with pytest.raises(RuntimeError, match=r"'moe_act' in recompute_modules"):
            runtime.expert_ops()

    def test_the_same_gated_triple_is_built_when_moe_act_is_not_asked_for(self):
        """The refusal is about the REQUEST, not the gated activation: same shape, no raise."""
        runtime = _runtime_over(_DuckTypedExperts(gated=True, activation_recompute=False))

        activation = list(runtime.expert_ops().children())[1]
        assert isinstance(activation, _StubScaledSwiGLU)
        assert activation.activation_recompute_in_mlp is False


class TestQuantizedWeightShadow:
    """``quantize_expert_weights`` for real over a shadowed TE. HAZARD (silent): the shadow is no
    ``Parameter`` and TE saves it per TOKEN CHUNK, so unmarked it is a D2H per chunk freeing 0."""

    @pytest.fixture(autouse=True)
    def _shadow_transformer_engine(self, monkeypatch):
        self.marks = _install_fake_transformer_engine(monkeypatch)
        self.runtime = _StubbedRuntime(_make_config(), _Seam())

    def test_every_shadow_stored_on_the_op_is_the_tensor_that_was_marked(self):
        """Identity, not shape: marking one tensor and storing another frees nothing and raises
        nothing, and is exactly what the static test this replaces could not see."""
        self.runtime.quantize_expert_weights()

        fc1, fc2 = self.runtime._fc1_op, self.runtime._fc2_op
        stored = [fc1.weight0, fc1.weight1, fc2.weight0, fc2.weight1]
        assert len(self.marks) == 2 * _NUM_EXPERTS, "one shadow per expert weight, marked once"
        assert [id(shadow) for shadow in self.marks] == [id(shadow) for shadow in stored]

    def test_the_shadow_is_offloadable_material_and_carries_the_wgrad_requirement(self):
        """Not a Parameter, so the offload hooks WOULD take it; requires_grad, or TE reads
        weight_requires_grad off it and SKIPS the wgrad entirely."""
        self.runtime.quantize_expert_weights()

        for shadow in self.marks:
            assert not isinstance(shadow, torch.nn.Parameter)
            assert shadow.requires_grad is True

    def test_the_shadow_hides_the_parameter_from_getattr_but_not_from_parameters(self):
        """The instance dict, not ``register_parameter``: the fuser's one-time snapshot and DDP
        must keep pointing at the real parameters while TE's live re-read finds the shadow."""
        real = ep_chunk_overlap.expert_weight_parameters(self.runtime._experts, _NUM_EXPERTS)
        self.runtime.quantize_expert_weights()

        fc1, fc2 = self.runtime._fc1_op, self.runtime._fc2_op
        assert [id(p) for p in fc1.parameters()] == [id(p) for p in real[:_NUM_EXPERTS]]
        assert [id(p) for p in fc2.parameters()] == [id(p) for p in real[_NUM_EXPERTS:]]

        # And dropped again by the backward's first stage, or the wgrad would accumulate into a
        # shadow instead of into mcore's main_grad on the real parameter.
        self.runtime.release_expert_weights()
        assert [id(fc1.weight0), id(fc1.weight1)] == [id(p) for p in real[:_NUM_EXPERTS]]

    def test_the_quantizers_are_built_by_tes_own_reset_and_asked_for_both_usages(self):
        """Columnwise is not optional here: this path always enables grad, so TE's own
        ``columnwise=input_requires_grad`` would be True on every call that reaches it."""
        self.runtime.quantize_expert_weights()

        for op in (self.runtime._fc1_op, self.runtime._fc2_op):
            assert len(op.recipes) == 1, "the fuser's own call, made once per op"
            for index in range(_NUM_EXPERTS):
                quantizer = op.get_quantizer("forward", 2 * index + 1)
                assert quantizer.internal is False
                assert quantizer.usage == (True, True)

    def test_the_flag_transformer_engine_sets_is_the_one_the_offload_checker_honours(self):
        """TE marks the DATA tensors, not the wrapper, and mcore's checker walks
        ``get_data_tensors``; a mark that missed that path would free nothing and say nothing."""
        marked, plain = torch.zeros(4), torch.zeros(4)
        marked._TE_do_not_offload = True

        assert _te_do_not_offload(SimpleNamespace(get_data_tensors=lambda: (marked,))) is True
        assert _te_do_not_offload(SimpleNamespace(get_data_tensors=lambda: (plain,))) is False


class TestTheChunkRunsThroughTheExpertsOwnForward:
    """One chunk is handed to the EXPERTS module, not to a sequence this package assembled.

    What stood here was the per-chunk offload group this package opened itself. That group, the
    quantization padding and the NCCL-EP ``op_kwargs`` all live in ``TEGroupedMLP._fused_forward``
    (experts.py:792-920) now, which is why the schedule calls the module rather than the op
    Sequential. HAZARD (silent): calling the Sequential directly skips every one of them and
    leaves the numbers right, with the offload doing nothing -- the same shape of failure the
    deleted tests were written against, one level up.
    """

    @pytest.fixture(autouse=True)
    def _shadow_transformer_engine(self, monkeypatch):
        _install_fake_transformer_engine(monkeypatch)
        # Outside the code under test, as `torch.cuda.current_stream` already is: there is no
        # CUDA context here and the stream the schedule passes is a sentinel.
        monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
        self.seam = _Seam()
        self.runtime = _StubbedRuntime(_make_config(), self.seam)
        self.dispatched = None

    def _run(self):
        """One chunk through the production ``run_expert_forward``."""
        self.dispatched = DispatchedChunk(
            recv_x=torch.zeros((_CAPACITY, _HIDDEN_SIZE)),
            recv_probs=torch.zeros((_CAPACITY,)),
            split_sizes=torch.tensor([_CAPACITY // 2, _CAPACITY // 2]),
            graph=_Opaque("dispatch graph"),
        )
        return ep_chunk_overlap.run_expert_forward(
            self.runtime, dispatched=self.dispatched, stream=_AmbientStream()
        )

    def test_the_module_itself_runs_the_chunk_with_the_counts_undigested(self):
        """``TEGroupedMLP.forward``'s three positional arguments, in its order and once per chunk.

        The leaf it gets is this chunk's own detached ``recv_x``, which is what the dgrad later
        differentiates; handing it the receive itself would differentiate into the transport.
        """
        _combine_input, saved = self._run()

        assert len(self.seam.experts.forward_calls) == 1, "one expert call per chunk"
        hidden, split_sizes, probs, _built = self.seam.experts.forward_calls[0]
        assert hidden is saved.dispatched_tokens and hidden.requires_grad is True
        assert hidden is not self.dispatched.recv_x
        # Row layout is TE's business: it gets the manager's per-expert counts, undigested.
        assert split_sizes is self.dispatched.split_sizes
        assert probs is saved.routing_probs and probs.shape == (_CAPACITY,)
        assert saved.dispatch_graph is self.dispatched.graph

    def test_the_fused_sequence_exists_before_the_experts_are_called(self):
        """``expert_ops`` is FORCED first. The quantization and EGTP steps write into the ops'
        weight slots and run before any forward, so a sequence the forward built lazily would be
        a second one, carrying neither the shadows nor the accumulators."""
        self._run()

        assert self.seam.experts.forward_calls[0][3] is True
        assert self.runtime._ops is self.seam.experts._fused_ops[0]

    def test_the_combine_gets_a_detach_and_the_saved_root_keeps_the_graph(self):
        """The transport reads a detached view; the saved root is what the dgrad starts from.
        Handing the graph-bearing tensor to the combine would make the collective part of it."""
        combine_input, saved = self._run()

        assert saved.fc2_output.requires_grad is True
        assert combine_input is not saved.fc2_output
        assert combine_input.requires_grad is False

    def test_the_package_opens_no_offload_group_of_its_own(self):
        """Statically: the group is the experts'. One opened here too would wrap a forward that
        already opened one, offloading each chunk's activations twice and reloading them once."""
        source = inspect.getsource(ep_chunk_overlap)

        for name in ("off_interface", "fine_grained_activation_offload", "group_offload"):
            assert name not in source

    def test_an_expert_bias_is_refused(self):
        """``_fused_forward`` returns ``bias=None``; anything else means the UNFUSED path ran,
        whose bias this schedule has nowhere to put and would drop on the floor."""
        self.seam.experts.fc2_bias = torch.zeros(_HIDDEN_SIZE)

        with pytest.raises(RuntimeError, match="does not support an expert FC2 bias"):
            self._run()


class TestOpShellIsBuiltOnce:
    """The one piece of state the deleted backend class used to own.

    HAZARD (silent): ``OperationFuser.__init__`` snapshots the parameter list once
    (``ops/fuser.py:548``) while the fused forward re-reads weights live
    (``ops/fused/grouped_mlp.py:1211``, ``:1248``), so re-registering the expert parameters on a
    ``Sequential`` that has already run computes with the NEW weight and attributes the gradient
    to the OLD tensor -- measured: the optimizer sees no update and nothing raises.

    The sequence is the EXPERTS' now, which adds a THIRD way to end up with two of them: build
    one here while ``TEGroupedMLP._fused_forward`` lazily builds its own on the trunk's next
    forward. Hence three tests -- one place may build, it builds at most once, and one that
    already exists is borrowed rather than replaced.
    """

    @pytest.fixture(autouse=True)
    def _shadow_transformer_engine(self, monkeypatch):
        _install_fake_transformer_engine(monkeypatch)

    def test_only_expert_ops_ever_builds_a_shell(self):
        """Statically: ``make_fused_ops`` has exactly one call site, inside ``expert_ops``.

        A second call site anywhere -- a convenience rebuild, a reset path, a promoted test
        helper -- would re-register the real parameters on a fresh fuser and lose every later
        weight gradient without raising. Read off the module's source, so a new one fails here.
        """
        module = ast.parse(inspect.getsource(ep_chunk_overlap))
        call_sites = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "make_fused_ops"
        ]
        assert len(call_sites) == 1

        accessor = [
            method
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "MoEEPChunkOverlapRuntime"
            for method in node.body
            if isinstance(method, ast.FunctionDef) and method.name == "expert_ops"
        ]
        assert len(accessor) == 1, "the runtime's one op-shell accessor"
        assert call_sites[0] in list(ast.walk(accessor[0])), "the one call site is that accessor's"
        # Every step is guarded, not unconditional: the guards are the whole of what refuses an
        # unfused experts module, builds the sequence once, and caches this layer's view once.
        guards = [node for node in accessor[0].body if isinstance(node, ast.If)]
        assert len(guards) == 3, "one refuses the unfused forward, one builds, one caches"
        assert call_sites[0] in list(ast.walk(guards[1]))

        stores = [
            node
            for node in ast.walk(accessor[0])
            if isinstance(node, ast.Attribute)
            and node.attr == "_ops"
            and isinstance(node.ctx, ast.Store)
        ]
        assert len(stores) == 1 and stores[0] in list(ast.walk(guards[2]))

    def test_both_wgrad_modes_are_asked_for_rather_than_required_of_the_model(self):
        """The override is the whole reason neither wgrad flag is a model-wide requirement.

        Dropping either one is silent: without the delay there is nothing for run_chunk_wgrad to
        drain, and without the accumulate the wgrad lands in ``.grad`` while the EGTP flush reads
        a ``main_grad`` nobody wrote. Both are invisible from the loss.
        """
        experts = _DuckTypedExperts()
        runtime = _runtime_over(experts)

        runtime.expert_ops()

        assert experts.fused_ops_overrides == (True, True)

    def test_expert_ops_builds_on_the_first_call_and_caches_it(self, monkeypatch):
        """Behaviourally: the second call returns the first shell without building another."""
        seam = _Seam()
        runtime = _make_runtime(seam, monkeypatch)

        first = runtime.expert_ops()
        second = runtime.expert_ops()

        assert first is second
        assert seam.experts.make_fused_ops_calls == 1, "a rebuilt shell grads the OLD tensor"
        assert first is seam.experts._fused_ops[0], "and it is the EXPERTS', not a private one"
        # The two per-linear ops the wgrad drain reads are cached by that same one call.
        fc1_op, _activation_op, fc2_op = list(first.children())
        assert (runtime._fc1_op, runtime._fc2_op) == (fc1_op, fc2_op)

    def test_a_sequence_the_experts_already_built_is_borrowed_untouched(self, monkeypatch):
        """``_fused_forward`` builds one lazily, so by the time a layer runs BOTH paths there may
        already be one. Building a second here would leave the weight shadows and the EGTP
        accumulators on a shell TransformerEngine never runs, losing every expert gradient."""
        seam = _Seam()
        runtime = _make_runtime(seam, monkeypatch)
        already_built = seam.experts._make_fused_ops()
        seam.experts._fused_ops = (already_built,)
        builds_before = seam.experts.make_fused_ops_calls

        assert runtime.expert_ops() is already_built
        assert seam.experts.make_fused_ops_calls == builds_before


class TestExpertComputeSeam:
    """What the runtime guarantees its expert compute, against one that guarantees nothing."""

    @pytest.fixture(autouse=True)
    def _ambient_stream(self, monkeypatch):
        """Hand the runtime a sentinel compute stream, so none of this needs a CUDA context."""
        self.ambient = _AmbientStream()
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: self.ambient)

    def test_routed_experts_compute_forward_declares_the_dispatch_buffers(self, monkeypatch):
        """The forward prologue: three allocator claims, made before the expert compute runs.

        DeepEP allocated all three on its comm stream. Without the claims the caching allocator
        hands the block back to that stream and the payload is overwritten mid-GEMM -- measured at
        ``rel = 4.6e+02`` with nothing raised. This stand-in makes no claim of its own, which is
        what proves the runtime made them.
        """
        seam = _Seam()
        runtime = _make_runtime(seam, monkeypatch)
        dispatched = _make_dispatched(seam)

        fc2_output, saved = runtime.routed_experts_compute_forward(dispatched=dispatched)

        prologue, delegated = seam.log[:-1], seam.log[-1]
        assert delegated == "run_expert_forward", "the prologue must precede the GEMMs, not follow"
        # Exactly the three that cross from the comm stream to this one. The rest is read back
        # there, by the combine or the dispatch node's backward, so it must NOT be claimed.
        assert set(prologue) == {
            "record_stream:recv_x",
            "record_stream:recv_probs",
            "record_stream:split_sizes",
        }
        # Every claim names the stream the expert compute was handed, the one its GEMMs run on.
        assert {stream for _role, stream in seam.stream_claims} == {self.ambient}

        call = seam.expert_compute.forward_calls[0]
        assert call["stream"] is self.ambient, "it is handed its stream, never asks for one"
        assert call["dispatched"] is dispatched
        assert call["recv_x"] is dispatched.recv_x
        # Row layout is TE's business: it gets the manager's per-expert counts, undigested.
        assert call["split_sizes"] is dispatched.split_sizes
        # And THIS runtime, so the op shell it resolves is this layer's and not another's.
        assert call["runtime"] is runtime
        assert fc2_output.shape[0] == _CAPACITY
        # The dispatch NODE rides in the saved chunk, which is where the backward differentiates
        # it to get the dispatch adjoint.
        assert saved.dispatch_graph is dispatched.graph

    def test_routed_experts_compute_backward_declares_the_backward_receive(self, monkeypatch):
        """The backward prologue: the same claim on the backward dispatch's own receive."""
        seam = _Seam()
        runtime = _make_runtime(seam, monkeypatch)
        grad_recv = _make_grad_recv(seam)

        runtime.routed_experts_compute_backward(saved=_make_saved(), grad_recv=grad_recv)

        assert seam.log == ["record_stream:grad_recv", "run_expert_backward"]
        assert seam.stream_claims == [("grad_recv", self.ambient)]
        assert seam.expert_compute.backward_calls[0]["stream"] is self.ambient
        assert seam.expert_compute.backward_calls[0]["recv_x"] is grad_recv

    def test_moe_forward_fires_the_parameter_all_gather(self):
        """Statically: ``moe_forward`` calls ``ensure_expert_parameters_ready``.

        The collective half is the schedule's, TE keeps the purely local half. A forward that
        never called it would leave this rank reading un-gathered shards and one collective out
        of step with its peers, raising nothing. Static because the call was inlined out of a
        runtime method, and because this file never runs ``moe_forward``, which needs streams.
        """
        module = ast.parse(inspect.getsource(ep_chunk_overlap))
        forwards = [
            method
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "MoEEPChunkOverlapRuntime"
            for method in node.body
            if isinstance(method, ast.FunctionDef) and method.name == "moe_forward"
        ]
        assert len(forwards) == 1
        # Both spellings, since the two halves now sit on opposite sides of the class boundary:
        # the module function by bare name, the sibling method off ``self``.
        called = {
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(forwards[0])
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                or (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                )
            )
        }
        assert "ensure_expert_parameters_ready" in called
        assert "quantize_expert_weights" in called, "the expert compute's own once-per-forward half"

    def test_the_only_buffer_shared_with_the_dispatch_is_its_receive(self, monkeypatch):
        """The expert compute is handed the dispatch's receive to READ and no destination to write.

        The trunk passes none either -- ``MoETokenDispatcher.get_expert_zero_copy_buffers``
        returns ``(None, None)`` (token_dispatcher.py:222-230) -- so TE allocates FC2's output and
        dX itself: one ``[capacity, hidden]`` bf16 tensor per chunk in flight, spent to drop the
        aliasing.
        """
        seam = _Seam()
        runtime = _make_runtime(seam, monkeypatch)
        forward_dispatch = _make_dispatched(seam)
        backward_grad = _make_grad_recv(seam)

        fc2_output, _ = runtime.routed_experts_compute_forward(dispatched=forward_dispatch)
        grad_input, _ = runtime.routed_experts_compute_backward(
            saved=_make_saved(), grad_recv=backward_grad
        )

        # By STORAGE, not shape or start address: a `.view()` of the receive under any name lands
        # in `shared` below -- the destination that was removed. Keyword-only args catch the rest.
        for call, receive_tensor, produced in (
            (seam.expert_compute.forward_calls[0], forward_dispatch.recv_x, fc2_output),
            (seam.expert_compute.backward_calls[0], backward_grad, grad_input),
        ):
            receive = receive_tensor.untyped_storage().data_ptr()
            shared = {
                name
                for name, value in call.items()
                if isinstance(value, torch.Tensor) and value.untyped_storage().data_ptr() == receive
            }
            assert shared == {"recv_x"}, shared
            assert call["recv_x"] is receive_tensor
            # And what comes back -- the payload the collective reads -- is its own allocation.
            assert produced.untyped_storage().data_ptr() != receive

    def test_the_saved_chunk_crosses_the_runtime_with_nothing_read_off_it(self, monkeypatch):
        """One saved chunk crosses the seam and the runtime reads NOTHING off it.

        Its four fields are ``_Opaque``, so any attribute the runtime touches -- a shape, a
        ``record_stream`` -- raises here instead of becoming a wrong answer on eight GPUs. That
        opacity is what lets the expert compute put its whole autograd graph in those slots, and
        makes keeping alive anything they imply its job. The handle that used to be the one
        exception is gone with the transport: the dispatch NODE now rides in its slot and only
        the SCHEDULE differentiates it, on the comm stream, outside this seam.

        Nothing carries the weight gradients -- they stay parked in TE's FIFOs from the dgrad
        that enqueued them until ``run_chunk_wgrad`` drains them inside the same chunk, so the
        runtime holds no wgrad STATE, which the last three assertions pin.
        """
        seam = _Seam()
        runtime = _make_runtime(seam, monkeypatch)
        dispatched = _make_dispatched(seam)

        _output, saved = runtime.routed_experts_compute_forward(dispatched=dispatched)
        assert saved is seam.expert_compute.saved_returns[-1]
        assert saved.dispatch_graph is dispatched.graph

        handed_back = _make_saved()
        runtime.routed_experts_compute_backward(saved=handed_back, grad_recv=_make_grad_recv(seam))
        assert seam.expert_compute.backward_calls[0]["saved"] is handed_back

        # STATE, not surface: the three wgrad stages are methods on the runtime, but a wgrad
        # SLOT -- per-chunk or class-wide -- is what would outlive the chunk that enqueued it.
        assert [name for name in vars(runtime) if "wgrad" in name] == []
        assert [
            name
            for name, value in vars(MoEEPChunkOverlapRuntime).items()
            if "wgrad" in name and not callable(value)
        ] == []
        assert seam.expert_compute.wgrad_calls == [], "no wgrad call reaches TE through here"

    def test_the_routing_gradient_reaches_the_combine_as_the_dgrads_own_tensor(self, monkeypatch):
        """The one dgrad output that leaves the compute stream is handed on, flat and uncopied.

        Replaces the assertion that it landed in a runtime-owned per-chunk slot. That slot kept
        the next chunk from overwriting what this chunk's combine was still reading; a tensor the
        dgrad allocated for this chunk alone has no later holder, so slot and copy are both gone.
        """
        seam = _Seam()
        runtime = _make_runtime(seam, monkeypatch)

        _grad_input, grad_probs = runtime.routed_experts_compute_backward(
            saved=_make_saved(), grad_recv=_make_grad_recv(seam)
        )

        # By STORAGE: a copy anywhere on this path would show up as a different allocation.
        assert grad_probs.data_ptr() == seam.expert_compute.dprob_returns[-1].data_ptr()
        assert grad_probs.shape == (_CAPACITY,)

        _grad_input, other_chunk = runtime.routed_experts_compute_backward(
            saved=_make_saved(), grad_recv=_make_grad_recv(seam)
        )
        assert other_chunk.data_ptr() != grad_probs.data_ptr(), "one allocation per chunk"


# EGTP on CPU: `expert_gtp_weight_remat_size > 1` shards every `weight{i}` on dim 0 and TE must
# never see one -- the forward gathers once into a full-shaped shadow, the backward KEEPS it.

#: Not allocator noise: generalized_tensor_parallelism.py:338-346 retains pool buffers keyed on
#: (shape, dtype), so a recycled one holds the PREVIOUS prescaled wgrad: no `zero_()` blows up.
_WGRAD_POOL_POISON = 7.0

#: The EGTP degree the fakes shard by. 2 is enough: the shard is half the logical dim-0, which is
#: what makes an unsharded-vs-sharded confusion a shape error rather than a silent aliasing.
_EGTP_SIZE = 2


class _FakeGTPWgradPool:
    """GTP's ``_wgrad_pool_get``, with its two properties that matter kept: it is NOT zeroed, and
    a second call hands back a DIFFERENT buffer (so a double acquire silently loses the first)."""

    def __init__(self) -> None:
        self.handed_out: list = []
        #: Every ``finalize_group_grads`` in call order, by leader. The ORDER is load-bearing:
        #: the reduce-scatter is a cascade finalized by the chain PREDECESSOR's flush.
        self.flush_order: list = []

    def get(self, shape, dtype) -> torch.Tensor:
        buffer = torch.full(tuple(shape), _WGRAD_POOL_POISON, dtype=dtype)
        self.handed_out.append(buffer)
        return buffer


def _make_gtp_weight_group(
    out_features: int, in_features: int, num_experts: int, pool: _FakeGTPWgradPool
) -> list:
    """One linear's EGTP group: ``num_experts`` dim-0 shards, the leader at index 0.

    The five members the package touches and no more: ``is_gtp_weight_remat`` /
    ``_unsharded_shape`` (Site 0), ``weight_list`` + ``materialize_group_for_forward`` (Site 1),
    ``get_wgrad_tensor`` (Site 3), ``finalize_group_grads`` (Site 4). ``weight_list`` is set on
    element 0 ONLY, as ``attach_gtp_to_presharded_module`` sets it
    (generalized_tensor_parallelism.py:716), and every leader-only member raises on a non-leader
    the way ``list(None)`` does at ``:1129``.
    """
    assert out_features % _EGTP_SIZE == 0
    shard_rows = out_features // _EGTP_SIZE
    weights = []
    for index in range(num_experts):
        weight = torch.nn.Parameter(torch.zeros((shard_rows, in_features)))
        weight.is_gtp_weight_remat = True
        # A real GTPShardedParam carries this, which is exactly the flag TE reads
        # (ops/fused/grouped_mlp.py:660). The design's whole premise is that it never reaches TE.
        weight.is_distributed_weight = True
        weight.is_routed_expert = True
        weight.expert_idx = index
        weight.weight_list = None
        weight._unsharded_shape = (out_features, in_features)
        # Shard-shaped, as DDP sizes it: param_and_grad_buffer.py:1412-1415 is not GTP-aware.
        weight.main_grad = torch.zeros((shard_rows, in_features))
        # The full weight the all-gather would rebuild. Distinct per expert, so a shadow taken
        # from the wrong index is visible rather than plausible.
        weight._full_weight = torch.full((out_features, in_features), float(index + 1))
        # What the BACKWARD all-gather rebuilds: the same weight COLUMNWISE, as TE's dgrad reads
        # it. Distinct per expert AND from `_full_weight`, so a wrong graft is a wrong NUMBER.
        weight._full_weight_bwd = _as_quantized(
            torch.full((out_features, in_features), float(100 + index)),
            rowwise=False,
            columnwise=True,
        )
        weight._pool = pool
        weight.materialize_calls = 0
        weight.materialize_bwd_calls = 0
        weight.wgrad_tensor_calls = 0
        weight.flushes = []
        _bind_gtp_methods(weight)
        weights.append(weight)
    weights[0].weight_list = weights
    return weights


def _gtp_weights_of(param):
    """``GTPShardedParam._weights``, including its ``list(None)`` failure on a non-leader."""
    return list(param.weight_list)


def _bind_gtp_methods(param) -> None:
    """Attach the four GTP methods as closures; a real shard gets them from its class."""

    def materialize_group_for_forward():
        param.materialize_calls += 1
        return [weight._full_weight for weight in _gtp_weights_of(param)]

    def materialize_group_for_backward():
        """GTP's public protocol method (generalized_tensor_parallelism.py:2244-2246), a
        three-line delegator to ``all_gather_and_prefetch_bwd``. Called with ``fwd=False``, so
        ``set_usage(rowwise=fwd, columnwise=not fwd)`` yields COLUMNWISE -- which is why the
        fakes hand back tensors carrying only that representation."""
        param.materialize_bwd_calls += 1
        return [weight._full_weight_bwd for weight in _gtp_weights_of(param)]

    def get_wgrad_tensor():
        param.wgrad_tensor_calls += 1
        return param._pool.get(param._unsharded_shape, param.main_grad.dtype)

    def finalize_group_grads(wgrads):
        # Mirrors the real one where it matters: zips against _weights and zip TRUNCATES
        # (generalized_tensor_parallelism.py:2098, :2003) -- the silent mode a length check traps.
        param.flushes.append(list(zip(_gtp_weights_of(param), list(wgrads))))
        param._pool.flush_order.append(param)

    param.materialize_group_for_forward = materialize_group_for_forward
    param.materialize_group_for_backward = materialize_group_for_backward
    param.get_wgrad_tensor = get_wgrad_tensor
    param.finalize_group_grads = finalize_group_grads


class _EGTPGroupedLinear(torch.nn.Module):
    """``_FakeGroupedLinear`` with EGTP weights: SHARD storage, LOGICAL declared geometry.

    The split is the whole point of Site 0. Megatron restores the logical ``out_features`` after
    GTP pre-shards it (``extensions/transformer_engine.py:466-470``), so the op shell, TE's
    ``validate_grouped_mlp_dims`` and the fused kernels all run on the logical geometry while
    ``weight{i}.shape`` is half of it.
    """

    def __init__(self, out_features: int, in_features: int, pool: _FakeGTPWgradPool) -> None:
        super().__init__()
        self.num_gemms = _NUM_EXPERTS
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = False
        self.wgrad_accumulation_and_reduce_hooks = None
        for index, weight in enumerate(
            _make_gtp_weight_group(out_features, in_features, _NUM_EXPERTS, pool)
        ):
            self.register_parameter(f"weight{index}", weight)


class _EGTPExperts(_DuckTypedExperts):
    """``_DuckTypedExperts`` whose two linears carry EGTP shards."""

    def __init__(self, pool: _FakeGTPWgradPool) -> None:
        super().__init__()
        self.linear_fc1 = _EGTPGroupedLinear(_FFN_HIDDEN_SIZE, _HIDDEN_SIZE, pool)
        self.linear_fc2 = _EGTPGroupedLinear(_HIDDEN_SIZE, _FFN_HIDDEN_SIZE, pool)


class _EGTPSeam(_Seam):
    """The seam with EGTP-sharded experts and one shared wgrad pool to watch."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = _FakeGTPWgradPool()
        self.experts = _EGTPExperts(self.pool)


def _make_fp8_param(leaders) -> None:
    """Turn each group's forward gather into what ``--fp8-param-gather`` really produces.

    Under MXFP8 that flag is MANDATORY (``arguments.py`` asserts it for ``fp8_recipe=mxfp8``), so
    this is the ONLY arm production reaches -- and the arm every recorded green measurement
    missed, because a BF16 parameter takes ``quantize_expert_weights``' copy-out branch instead.
    The gathered tensor is already quantized, so TE reuses it untouched
    (``ops/fused/grouped_mlp.py:1250-1256``) and it is installed as the shadow -- ROWWISE ONLY,
    which is exactly what the dgrad cannot read.
    """
    for leader in leaders:
        for weight in leader.weight_list:
            weight._full_weight = _as_quantized(weight._full_weight, rowwise=True, columnwise=False)


class _EGTPFixture:
    """Shared setup: a stubbed runtime over EGTP experts, with the op shell already built.

    The classes below drive the REAL ``quantize_expert_weights``,
    ``release_expert_weights`` and ``end_wgrad_accumulation`` over a GTP surface faked exactly as
    narrowly as the package uses it -- five members and a leader rule.
    """

    @pytest.fixture(autouse=True)
    def _shadow_transformer_engine(self, monkeypatch):
        self.marks = _install_fake_transformer_engine(monkeypatch)
        self.seam = _EGTPSeam()
        self.runtime = _StubbedRuntime(_make_config(), self.seam)

    @property
    def _ops(self):
        return (self.runtime._fc1_op, self.runtime._fc2_op)

    @property
    def _leaders(self):
        return (self.seam.experts.linear_fc1.weight0, self.seam.experts.linear_fc2.weight0)

    def _forward(self) -> None:
        self.runtime.quantize_expert_weights()


class TestEGTPForwardMaterialization(_EGTPFixture):
    """Site 1: ONE all-gather per group per forward, on the leader, and a shadow in EVERY slot."""

    def test_the_group_is_gathered_once_per_linear_on_the_leader(self):
        self._forward()

        for leader in self._leaders:
            assert leader.materialize_calls == 1, "once per forward, not once per expert"
        for linear in (self.seam.experts.linear_fc1, self.seam.experts.linear_fc2):
            for index in range(1, _NUM_EXPERTS):
                non_leader = getattr(linear, f"weight{index}")
                assert non_leader.materialize_calls == 0, "leader-only, or _weights raises"

    def test_every_weight_slot_holds_a_full_shaped_shadow(self):
        self._forward()

        for op, out_features in zip(self._ops, (_FFN_HIDDEN_SIZE, _HIDDEN_SIZE)):
            for index in range(_NUM_EXPERTS):
                shadow = op.__dict__[f"weight{index}"]
                assert tuple(shadow.shape) == (out_features, op.in_features)

    def test_the_shadow_is_not_a_distributed_weight(self):
        """The premise, asserted on the object TE reads (``_get_weight_tensors()[0]`` is a plain
        ``getattr``): with this flag True TE takes the distributed branch and refuses
        ``delay_wgrad_compute``. The shard the shadow hides carries it, which is the point."""
        self._forward()

        linears = (self.seam.experts.linear_fc1, self.seam.experts.linear_fc2)
        for op, linear in zip(self._ops, linears):
            for index in range(_NUM_EXPERTS):
                shadow = op.__dict__[f"weight{index}"]
                # What TE resolves, and what DDP and the optimizer still resolve.
                assert getattr(op, f"weight{index}") is shadow
                assert op._parameters[f"weight{index}"] is getattr(linear, f"weight{index}")
                assert not getattr(shadow, "is_distributed_weight", False)
                assert getattr(linear, f"weight{index}").is_distributed_weight is True

    def test_the_shadow_of_expert_i_is_the_materialized_weight_of_expert_i(self):
        """Index alignment, checked by VALUE: the fakes give expert ``i`` the constant ``i + 1``,
        so a transposed or truncated group is a wrong number rather than a wrong shape."""
        self._forward()

        for op in self._ops:
            for index in range(_NUM_EXPERTS):
                shadow = op.__dict__[f"weight{index}"]
                assert torch.equal(shadow, torch.full_like(shadow, float(index + 1)))

    def test_an_already_quantized_materialized_weight_still_becomes_a_shadow(self):
        """T1f, the ``--fp8-param-gather`` hole. Without this branch the slot keeps resolving to
        the SHARDED parameter and TE raises at ops/fused/grouped_mlp.py:711 -- after a full
        forward's work, and only because that raise happens to exist."""
        leader = self.seam.experts.linear_fc1.weight0
        for weight in leader.weight_list:
            weight._full_weight = weight._full_weight.as_subclass(_StubQuantizedTensor)

        self._forward()

        fc1_op = self.runtime._fc1_op
        for index in range(_NUM_EXPERTS):
            shadow = fc1_op.__dict__[f"weight{index}"]
            assert isinstance(shadow, _StubQuantizedTensor)
            assert tuple(shadow.shape) == (_FFN_HIDDEN_SIZE, _HIDDEN_SIZE)
        # Marked, or TE's offload pass dequantizes the shadow once per TOKEN CHUNK. FC1 is
        # quantized first, so its marks lead; FC2 arrives plain and takes the ordinary branch.
        assert [id(mark) for mark in self.marks[:_NUM_EXPERTS]] == [
            id(fc1_op.__dict__[f"weight{index}"]) for index in range(_NUM_EXPERTS)
        ]
        assert len(self.marks) == 2 * _NUM_EXPERTS
        # The contrast, by IDENTITY: FC1's shadow IS the materialized weight (installed untouched,
        # which is the branch under test), FC2's is the quantizer's fresh output.
        materialized = {id(weight._full_weight) for weight in leader.weight_list}
        assert {id(fc1_op.__dict__[f"weight{i}"]) for i in range(_NUM_EXPERTS)} == materialized
        fc2_op = self.runtime._fc2_op
        for index in range(_NUM_EXPERTS):
            fc2_weight = getattr(self.seam.experts.linear_fc2, f"weight{index}")
            assert id(fc2_op.__dict__[f"weight{index}"]) != id(fc2_weight._full_weight)


class TestEGTPStagingAccumulator(_EGTPFixture):
    """Sites 2 and 3: the shadow SURVIVES into the backward, carrying a zeroed full-sized buffer."""

    def _release(self) -> None:
        self._forward()
        self.runtime.release_expert_weights()

    def test_the_shadow_is_kept_and_carries_a_full_sized_accumulator(self):
        self._release()

        for op, out_features in zip(self._ops, (_FFN_HIDDEN_SIZE, _HIDDEN_SIZE)):
            for index in range(_NUM_EXPERTS):
                shadow = op.__dict__[f"weight{index}"]
                assert shadow is not None, "popping it re-exposes the sharded GTP parameter"
                assert tuple(shadow.main_grad.shape) == (out_features, op.in_features)

    def test_the_accumulator_is_zeroed(self):
        """S4. The pool hands back the previous iteration's PRESCALED wgrad, so skipping the
        ``zero_()`` is a geometric blow-up that reads as a bad learning rate."""
        self._release()

        assert self.seam.pool.handed_out, "the accumulator must come from GTP's own wgrad pool"
        for buffer in self.seam.pool.handed_out:
            assert torch.count_nonzero(buffer) == 0

    def test_get_wgrad_tensor_is_called_exactly_once_per_weight(self):
        """Z2/S3. A second call returns a DIFFERENT uninitialized buffer on the ordinary pool and
        the first chunks' gradients are simply lost, with nothing raised."""
        self._release()

        for linear in (self.seam.experts.linear_fc1, self.seam.experts.linear_fc2):
            for index in range(_NUM_EXPERTS):
                assert getattr(linear, f"weight{index}").wgrad_tensor_calls == 1
        assert len(self.seam.pool.handed_out) == 2 * _NUM_EXPERTS

    def test_the_real_parameters_keep_their_own_sharded_main_grad(self):
        """The staging buffer must NEVER be the DDP ``main_grad``: a full-shaped wgrad written
        into a shard-shaped slice of DDP's bucket overruns into the neighbouring parameter."""
        self._release()

        for linear in (self.seam.experts.linear_fc1, self.seam.experts.linear_fc2):
            for index in range(_NUM_EXPERTS):
                param = getattr(linear, f"weight{index}")
                assert tuple(param.main_grad.shape) == tuple(param.shape)

    def test_without_egtp_the_shadows_are_still_popped(self):
        """N2, on this file's own non-EGTP seam: the old behaviour, bit for bit."""
        seam = _Seam()
        runtime = _StubbedRuntime(_make_config(), seam)
        runtime.quantize_expert_weights()

        runtime.release_expert_weights()

        for op in (runtime._fc1_op, runtime._fc2_op):
            for index in range(_NUM_EXPERTS):
                assert f"weight{index}" not in op.__dict__


class TestEGTPFlush(_EGTPFixture):
    """Site 4: ONE reduce-scatter per group per backward, on the leader, with the FULL list."""

    def _backward(self) -> None:
        """Forward, release (which acquires and zeroes the accumulators), then flush.

        The accumulators are recorded between the two, because the flush drops the slots that
        hold them -- and it is the identity of what was flushed that this class checks.
        """
        self._forward()
        self.runtime.release_expert_weights()
        self._accumulators = {
            id(op): [op.__dict__[f"weight{index}"].main_grad for index in range(_NUM_EXPERTS)]
            for op in self._ops
        }
        self.runtime.end_wgrad_accumulation()

    def test_each_group_is_flushed_exactly_once_on_its_leader(self):
        self._backward()

        for leader in self._leaders:
            assert len(leader.flushes) == 1
        for linear in (self.seam.experts.linear_fc1, self.seam.experts.linear_fc2):
            for index in range(1, _NUM_EXPERTS):
                assert getattr(linear, f"weight{index}").flushes == []

    def test_the_flush_carries_the_full_index_aligned_group(self):
        """F1/S1, the worst silent mode in the design. ``finalize_group_grads`` zips against
        ``self._weights`` twice and ``zip`` TRUNCATES, so a short list drops the tail experts'
        reduce-scatters while still firing their DDP grad-ready."""
        self._backward()

        for leader, op in zip(self._leaders, self._ops):
            paired = leader.flushes[0]
            assert len(paired) == _NUM_EXPERTS == len(leader.weight_list)
            for index, (weight, wgrad) in enumerate(paired):
                assert weight is leader.weight_list[index]
                assert wgrad is self._accumulators[id(op)][index]

    def test_fc2_is_flushed_before_fc1(self):
        """REVERSE CHAIN ORDER, and it is not cosmetic.

        A weight's async reduce-scatter is finalized by its chain PREDECESSOR's flush
        (``generalized_tensor_parallelism.py:2194-2224``). FC1 and FC2 split onto separate chains
        only once ``classify_gtp_remat_chains`` has run AND the names carry ``.linear_fc1.`` /
        ``.linear_fc2.`` (``_classify_param_chain:204-217``); otherwise both sit on the single
        UNGRAPHED chain in first-forward order FC1 -> FC2, where flushing FC1 first leaves FC2's
        reduce-scatter in flight forever -- no ``main_grad.add_`` and no DDP grad-ready. FC2 first
        is reverse-chain order there and irrelevant in the split layout.
        """
        self._backward()

        fc1_leader, fc2_leader = self._leaders
        assert self.seam.pool.flush_order == [fc2_leader, fc1_leader]

    def test_the_slots_are_empty_after_the_flush(self):
        """F8: the post-backward state must be exactly the one the non-EGTP path leaves, or the
        next forward's ``quantize_expert_weights`` writes over a live accumulator."""
        self._backward()

        for op in self._ops:
            for index in range(_NUM_EXPERTS):
                assert f"weight{index}" not in op.__dict__

    @pytest.mark.xfail(
        strict=True,
        reason="Asserts a length check end_wgrad_accumulation does not perform. It sizes wgrads "
        "from config.num_local_experts and never reads leader.weight_list, so a group longer than "
        "the slots goes unnoticed: the method has 0 raise statements and no message matching this "
        "pattern exists in megatron/. Strict, so it flags itself for removal once the guard lands.",
    )
    def test_a_short_group_is_refused(self):
        """T4b: simulate the zip truncation by giving the leader a group LONGER than the slots."""
        self._forward()
        self.runtime.release_expert_weights()
        leader = self.seam.experts.linear_fc1.weight0
        leader.weight_list = list(leader.weight_list) + [leader.weight_list[-1]]

        with pytest.raises(RuntimeError, match="weight gradients for a GTP group of"):
            self.runtime.end_wgrad_accumulation()

    @pytest.mark.xfail(
        strict=True,
        reason="Asserts a presence check end_wgrad_accumulation does not perform. A missing "
        "main_grad becomes None via getattr(shadow, 'main_grad', None) and is handed to "
        "finalize_group_grads unremarked: the method has 0 raise statements and no message "
        "matching this pattern exists in megatron/. Strict, so it flags itself for removal once "
        "the guard lands.",
    )
    def test_a_missing_accumulator_is_refused(self):
        """The flush must never silently reduce-scatter fewer buffers than experts."""
        self._forward()
        self.runtime.release_expert_weights()
        del self.runtime._fc2_op.__dict__["weight1"].main_grad

        with pytest.raises(RuntimeError, match="expert 1 has no wgrad accumulator"):
            self.runtime.end_wgrad_accumulation()

    def test_a_second_backward_gets_a_FRESH_zeroed_accumulator(self):
        """The shape of R1, ranked first among this design's silent failure modes.

        The staging buffer is per BACKWARD and the real ``main_grad`` is per ITERATION. Two full
        cycles must therefore acquire twice and zero twice: a buffer carried over would make the
        second microbatch's gradient the SUM of both, i.e. a factor of ~num_microbatches too large
        on expert weights only, with a healthy loss curve and a bit-identical forward and dX.
        """
        for _ in range(2):
            self._forward()
            self.runtime.release_expert_weights()
            for op in self._ops:
                for index in range(_NUM_EXPERTS):
                    accumulator = op.__dict__[f"weight{index}"].main_grad
                    assert torch.count_nonzero(accumulator) == 0, "the pool hands back poison"
            self.runtime.end_wgrad_accumulation()

        for linear in (self.seam.experts.linear_fc1, self.seam.experts.linear_fc2):
            for index in range(_NUM_EXPERTS):
                assert getattr(linear, f"weight{index}").wgrad_tensor_calls == 2
        for leader in self._leaders:
            assert len(leader.flushes) == 2, "one reduce-scatter per weight group per BACKWARD"

    def test_without_egtp_the_flush_is_a_no_op(self):
        """N3. Two ``getattr`` calls and a ``continue``; the shadows are already gone."""
        seam = _Seam()
        runtime = _StubbedRuntime(_make_config(), seam)
        runtime.quantize_expert_weights()
        runtime.release_expert_weights()

        runtime.end_wgrad_accumulation()

        for op in (runtime._fc1_op, runtime._fc2_op):
            for index in range(_NUM_EXPERTS):
                assert f"weight{index}" not in op.__dict__

    def test_the_flush_before_any_forward_is_a_no_op(self):
        """``op is None`` is reached when no forward ever built the ops. ``continue``, not
        ``return``: with ``return`` the FC2 op would be skipped the moment the two stopped
        being built together (D8, the same defect the old ``release_expert_weights`` carried)."""
        runtime = _StubbedRuntime(_make_config(), _EGTPSeam())

        runtime.end_wgrad_accumulation()
        runtime.release_expert_weights()


class TestEGTPBackwardGather(_EGTPFixture):
    """Site 5: the forward/backward boundary re-materializes the group COLUMNWISE.

    TWO defects, one cause, and this is the one call that removes both.

    Under ``--fp8-param-gather`` -- mandatory under MXFP8, so the only arm production reaches --
    ``quantize_expert_weights`` installs GTP's ROWWISE-ONLY forward gather as the shadow. TE's
    fused dgrad reads ``_columnwise_data`` / ``_columnwise_scale_inv`` off the weight the forward
    SAVED (``ops/fused/grouped_mlp.py:1900-1904`` unpacks, ``:2179-2180`` and ``:2465-2466``
    read), found ``None`` and raised ``TypeError`` from
    ``swizzle_scales_and_pack_ptrs_for_discrete_weights`` on every rank -- defect D2. Making one
    tensor serve both passes also required the forward view to survive into the backward, which
    GTP's two-buffer parity does not guarantee (``generalized_tensor_parallelism.py:303-313``):
    a later same-parity forward gather overwrote it and ``N-2`` of ``N`` layers dgradded against
    another layer's weights, SILENTLY -- defect D1.

    Re-gathering fixes D2 by construction (``fwd=False`` yields ``columnwise=True``) and D1
    structurally: nothing gathered by the forward is read after it, so the one-block-ahead
    lifetime the parity DOES guarantee suffices.

    THE FIELDS ARE GRAFTED rather than the slot re-pointed because the two readers resolve the
    weight differently: the wgrad reads the SLOT at backward time (``_get_weight_tensors()``,
    ``:659``), which is what lets the staging accumulator work, while the dgrad reads the
    forward's SAVED OBJECT and re-reads the slot only for a distributed weight (``:2103-2104``)
    -- which this path never presents, and could not, since ``is_dist_weight and delay_wgrad``
    raises at ``:711-714``. Both readers must be served on the same object.
    """

    def _fp8_forward(self) -> None:
        _make_fp8_param(self._leaders)
        self._forward()

    def _fp8_release(self) -> None:
        self._fp8_forward()
        self.runtime.release_expert_weights()

    @property
    def _shadows(self):
        return [op.__dict__[f"weight{index}"] for op in self._ops for index in range(_NUM_EXPERTS)]

    def test_the_forward_gather_alone_cannot_serve_the_dgrad(self):
        """The premise, asserted rather than assumed: after the forward every shadow carries
        rowwise data and NO columnwise data. That is defect D2, in one line."""
        self._fp8_forward()

        for shadow in self._shadows:
            assert shadow._rowwise_data is not None
            assert shadow._columnwise_data is None, "D2's precondition is gone; retune this test"
            assert ep_chunk_overlap._needs_backward_gather(shadow) is True

    def test_the_boundary_regathers_each_group_once_on_its_leader(self):
        """ONE backward all-gather per linear per LAYER -- not per token chunk, which is what
        made TE's own protocol path cost N gathers with N-1 blocking."""
        self._fp8_release()

        for linear in (self.seam.experts.linear_fc1, self.seam.experts.linear_fc2):
            assert linear.weight0.materialize_bwd_calls == 1, "leader gathers the whole group"
            for index in range(1, _NUM_EXPERTS):
                assert getattr(linear, f"weight{index}").materialize_bwd_calls == 0

    def test_the_dgrad_representation_is_grafted_onto_the_object_te_saved(self):
        """THE central property. Read off the object the FORWARD installed -- captured before
        the boundary runs -- and NOT off the weight slot, since the two are the same object only
        if the graft happened. Re-pointing the slot at the gathered tensor (the tempting design)
        satisfies a slot-based assertion while leaving TE's saved object, and the dgrad, broken.

        By IDENTITY and by VALUE: identity because TE's saved-tensor round trip returns the same
        Python object, so mutating it is the only route to the dgrad; value because expert ``i``
        carries the constant ``100 + i``, so a transposed or truncated group is a wrong NUMBER.
        """
        self._fp8_forward()
        saved_by_the_forward = [
            [op.__dict__[f"weight{index}"] for index in range(_NUM_EXPERTS)] for op in self._ops
        ]

        self.runtime.release_expert_weights()

        linears = (self.seam.experts.linear_fc1, self.seam.experts.linear_fc2)
        for saved, linear in zip(saved_by_the_forward, linears):
            for index, shadow in enumerate(saved):
                gathered = getattr(linear, f"weight{index}")._full_weight_bwd
                assert shadow._columnwise_data is gathered._columnwise_data
                assert shadow._columnwise_scale_inv is gathered._columnwise_scale_inv
                assert torch.equal(
                    shadow._columnwise_data,
                    torch.full_like(shadow._columnwise_data, float(100 + index)),
                )

    def test_no_forward_view_survives_the_boundary(self):
        """THE property the whole fix rests on, and the one worth a standing test.

        D1 was a held view being recycled under its holder. It cannot recur while nothing
        gathered by the forward is still referenced once the backward starts, so assert exactly
        that: every rowwise slot -- data and scale -- is dropped at the boundary. Read off the
        objects the FORWARD installed, since those are the ones TE saved and therefore the ones
        whose lifetime is in question."""
        self._fp8_forward()
        saved_by_the_forward = [
            op.__dict__[f"weight{index}"] for op in self._ops for index in range(_NUM_EXPERTS)
        ]

        self.runtime.release_expert_weights()

        for shadow in saved_by_the_forward:
            assert shadow._rowwise_data is None, "a forward view survived into the backward"
            assert shadow._rowwise_scale_inv is None
            assert shadow._columnwise_data is not None, "and the backward view is installed"

    def test_the_weight_slot_keeps_the_same_object_so_the_accumulator_lands_where_te_looks(self):
        """The wgrad reads ``main_grad`` off the SLOT (``:659``, ``:696``) and the dgrad off the
        forward's saved object. Re-pointing the slot at the freshly gathered tensor would split
        the two, so the slot must still hold the object the forward saved."""
        self._fp8_forward()
        before = [id(shadow) for shadow in self._shadows]

        self.runtime.release_expert_weights()

        assert [id(shadow) for shadow in self._shadows] == before
        for op in self._ops:
            for index in range(_NUM_EXPERTS):
                shadow = op.__dict__[f"weight{index}"]
                assert shadow.main_grad is not None
                assert tuple(shadow.main_grad.shape) == (op.out_features, op.in_features)

    def test_the_backward_gathers_wrapper_is_marked_not_offloadable(self):
        """Every chunk's fused backward saves this shadow, so an unmarked view would be
        dequantized once per TOKEN CHUNK and free nothing."""
        self._fp8_forward()
        marks_before = len(self.marks)

        self.runtime.release_expert_weights()

        wrappers = [
            getattr(linear, f"weight{index}")._full_weight_bwd
            for linear in (self.seam.experts.linear_fc1, self.seam.experts.linear_fc2)
            for index in range(_NUM_EXPERTS)
        ]
        newly = {id(mark) for mark in self.marks[marks_before:]}
        assert newly == {id(wrapper) for wrapper in wrappers}

    def test_the_wrappers_are_dropped_by_the_flush(self):
        """The lifetime is exactly one layer's backward. Holding the wrappers past the flush
        would keep GTP's gather buffer referenced across the NEXT layer's backward gather, which
        is the two-buffer parity's one-block-ahead window and no wider."""
        self._fp8_release()
        for op in self._ops:
            assert "_egtp_bwd_weight_group" in op.__dict__

        self.runtime.end_wgrad_accumulation()

        for op in self._ops:
            assert "_egtp_bwd_weight_group" not in op.__dict__
            for index in range(_NUM_EXPERTS):
                assert f"weight{index}" not in op.__dict__

    def test_the_flush_is_still_one_reduce_scatter_per_group_under_fp8_parameters(self):
        """The added gather must not disturb the wgrad half: still ONE
        ``finalize_group_grads`` per group per backward, on the leader, with the full list."""
        self._fp8_release()

        self.runtime.end_wgrad_accumulation()

        for leader in self._leaders:
            assert len(leader.flushes) == 1
            assert len(leader.flushes[0]) == _NUM_EXPERTS

    def test_bf16_expert_parameters_take_no_backward_gather(self):
        """V4/N6, and the reason every previously recorded green measurement stays valid: a BF16
        expert parameter is quantized into a FRESH both-usage tensor, which serves the dgrad, so
        the boundary is two ``getattr`` calls and a return."""
        self._forward()

        self.runtime.release_expert_weights()

        for leader in self._leaders:
            assert leader.materialize_bwd_calls == 0, "a copy-out shadow needs no re-gather"
        for shadow in self._shadows:
            assert shadow._columnwise_data is not None
            # Untouched: the copy-out branch's rowwise data is its OWN, not a view of GTP's
            # cache, so there is nothing to drop and dropping it would break any later reader.
            assert shadow._rowwise_data is not None
        for op in self._ops:
            assert "_egtp_bwd_weight_group" not in op.__dict__

    def test_without_egtp_the_backward_gather_is_never_reached(self):
        """N7: with ``expert_gtp_weight_remat_size == 1`` every line added here is unreachable,
        not merely inert. Demonstrated by counting calls, not by reading the branch."""
        calls = []
        original = ep_chunk_overlap._install_backward_gather

        def counting(op, weights):
            calls.append(op)
            return original(op, weights)

        ep_chunk_overlap._install_backward_gather = counting
        try:
            runtime = _StubbedRuntime(_make_config(), _Seam())
            runtime.quantize_expert_weights()
            runtime.release_expert_weights()
            runtime.end_wgrad_accumulation()
        finally:
            ep_chunk_overlap._install_backward_gather = original

        assert calls == [], "the non-EGTP path must not reach the backward gather at all"
        for op in (runtime._fc1_op, runtime._fc2_op):
            for index in range(_NUM_EXPERTS):
                assert f"weight{index}" not in op.__dict__


class TestEGTPWritesNoGTPState(_EGTPFixture):
    """The compliance property, as a test rather than as a review note.

    GTP is shared code, and this package's EGTP integration is read-only against it: read one
    flag, call the public protocol methods, never write GTP's own state. A previous revision
    wrapped ``param._grad_accum_hook`` in place -- an attribute GTP initialises
    (``generalized_tensor_parallelism.py:917``), sets (``:1838``) and CALLS (``:1858``) -- and
    never restored it.
    """

    #: Attributes owned and invoked by GTP itself. Writing any of them from here replaces GTP
    #: behaviour from outside GTP, with no file changed and nothing to restore it.
    GTP_OWNED = (
        "_grad_accum_hook",
        "_grad_accum_node",
        "_ag_ticket_fwd",
        "_ag_ticket_bwd",
        "_prefetch_handle",
        "_need_weight_prefetch",
        "_buf_parity",
        "weight_list",
        "chain_id",
        "state",
    )

    def test_the_schedule_never_assigns_to_a_gtp_owned_attribute(self):
        source = inspect.getsource(ep_chunk_overlap)

        for attribute in self.GTP_OWNED:
            for holder in ("param", "leader", "weight", "w", "shard"):
                assert f"{holder}.{attribute} =" not in source
                assert f"{holder}.{attribute}=" not in source
            assert f'setattr(param, "{attribute}"' not in source

    def test_the_expert_parameters_state_is_unchanged_by_a_whole_forward_and_backward(self):
        """The behavioural half: snapshot every GTP-owned attribute the fakes carry, run the
        real forward, boundary and flush, and require every one of them back unchanged."""
        _make_fp8_param(self._leaders)
        watched = [weight for leader in self._leaders for weight in leader.weight_list]
        before = [
            {name: id(getattr(weight, name)) for name in self.GTP_OWNED if hasattr(weight, name)}
            for weight in watched
        ]

        self.runtime.quantize_expert_weights()
        self.runtime.release_expert_weights()
        self.runtime.end_wgrad_accumulation()

        for weight, snapshot in zip(watched, before):
            for name, identity in snapshot.items():
                assert id(getattr(weight, name)) == identity, f"{name} was replaced"


class TestEGTPHooksAreInertWithoutIt(_EGTPFixture):
    """N1/N5: the package still imports no GTP symbol, and every added branch reads one flag."""

    def test_the_schedule_touches_gtp_only_through_its_public_api(self):
        """``gtp_api`` is the one GTP dependency allowed: it degrades to HAVE_GTP=False on a CPU-
        only path with no TransformerEngine, so importing it costs those paths nothing. Reaching
        past it into GTP internals would couple this schedule to a private layout instead."""
        source = inspect.getsource(ep_chunk_overlap)

        for forbidden in ("generalized_tensor_parallelism", "GTPShardedParam"):
            assert f"import {forbidden}" not in source
            assert f"from megatron.core.tensor_parallel.{forbidden}" not in source

    def test_every_added_branch_is_gated_on_the_one_attribute(self):
        """Read rather than imported, and the SAME attribute ``is_gtp_param`` is
        (generalized_tensor_parallelism.py:767-769)."""
        source = inspect.getsource(ep_chunk_overlap)

        assert source.count("is_gtp_param(weights[0])") == 2
        assert source.count("is_gtp_param(leader)") == 1
        # The raw attribute must not be read directly: the helper is the one supported spelling.
        assert "is_gtp_weight_remat" not in source
