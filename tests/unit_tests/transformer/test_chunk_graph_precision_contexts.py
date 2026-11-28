# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Precision handling inside chunk-wise CUDA graph captures: BF16 boundary layers must opt
out of the block-wide FP8 / NVFP4 context (and MTP layers back in), activation checkpointing
must go through TE's checkpoint (mcore's tensor_parallel.checkpoint skips the checkpoint node
while a graph is captured), and a graph replay must open the FP8 / NVFP4 autocast the block
forward would have opened. ``TestNvfp4ChunkGraphEndToEnd`` captures and replays real NVFP4
models and checks that every layer runs in the precision eager runs it in."""

from contextlib import contextmanager, nullcontext
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from megatron.core import fp4_utils, fp8_utils
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp4_utils import (
    get_fp4_recipe,
    get_layer_fp4_context,
    get_layer_quantization_context,
)
from megatron.core.fp8_utils import get_fp8_recipe, get_layer_fp8_context
from megatron.core.tensor_parallel import random as tensor_parallel_random
from megatron.core.transformer.chunk_cuda_graph import (
    ChunkCudaGraphBlockMixin,
    ChunkCudaGraphPostProcessBlock,
)
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import use_te_checkpoint

# Every test here reads private members of the chunk-graph and quantization helpers.
pytestmark = pytest.mark.internal


def _config(**overrides):
    kwargs = dict(
        num_layers=4, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True, bf16=True
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


_CHUNK = dict(
    cuda_graph_impl="chunk_wise",
    cuda_graph_modules=[],
    cuda_graph_dynamic_microbatches=True,
    sequence_packing_scheduler="dp_balanced",
    pad_packed_seq_alignment="max",
    max_seqlen_per_dp_cp_rank=128,
    thd_max_packed_sequences=8,
)

# ``--first-last-layers-bf16`` on the 4-layer models below: decoder layers 0 and 3 are BF16,
# layers 1 and 2 (and every MTP layer) NVFP4.
_NVFP4_FIRST_LAST_BF16 = dict(
    fp4="e2m1",
    first_last_layers_bf16=True,
    num_layers_at_start_in_bf16=1,
    num_layers_at_end_in_bf16=1,
)
# Full activation recompute: every decoder layer is its own checkpoint.
_FULL_RECOMPUTE = dict(
    recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1
)
# MXFP8 with the same BF16 boundary layers.
_MXFP8_FIRST_LAST_BF16 = dict(
    fp8="e4m3",
    fp8_recipe="mxfp8",
    first_last_layers_bf16=True,
    num_layers_at_start_in_bf16=1,
    num_layers_at_end_in_bf16=1,
)
_BF16 = "BF16"
_NVFP4 = "NVFP4BlockScaling"
_MXFP8 = "MXFP8BlockScaling"
_SEQ_LENGTH = _CHUNK["max_seqlen_per_dp_cp_rank"]
_VOCAB = 128


def _skip_unless_nvfp4_is_available():
    """Skip with the accurate reason on a TE build or device that cannot run NVFP4."""
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    is_nvfp4_available = getattr(FP8GlobalStateManager, "is_nvfp4_available", None)
    if is_nvfp4_available is None:
        pytest.skip("This Transformer Engine has no NVFP4 support")
    nvfp4_available, reason = is_nvfp4_available()
    if not nvfp4_available:
        pytest.skip(f"NVFP4 not available on this device: {reason}")


def _skip_unless_mxfp8_is_available():
    """Skip with the accurate reason on a device that cannot run MXFP8."""
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    mxfp8_available, reason = FP8GlobalStateManager.is_mxfp8_available()
    if not mxfp8_available:
        pytest.skip(f"MXFP8 not available on this device: {reason}")


class _LayerPrecisionRecorder:
    """Records the precision every TE GEMM module of the decoder and MTP stacks executes in.

    A forward pre-hook reads Transformer Engine's quantization state at the moment the module
    runs: ``"BF16"`` when quantization is disabled, otherwise the class name of the active
    recipe. Observations are grouped by phase: ``phase`` as set by the test (``"eager"``, then
    ``"replay"``), ``"warmup"`` for the eager passes TE runs inside ``make_graphed_callables``
    and ``"capture"`` for a forward recorded into a CUDA graph; ``"/recompute"`` is appended
    while an activation checkpoint re-runs its function in the backward pass.

    The LM head is called by the post-process outside every layer context, so the state at its
    call is the autocast the post-process itself runs under; it is kept apart in
    ``lm_head_observed``.
    """

    def __init__(self, model):
        import transformer_engine.pytorch as te

        self.phase = "eager"
        self.observed = {}  # phase -> {module name: set of precisions}
        self.lm_head_observed = {}  # phase -> set of precisions
        self.module_names = []
        self._handles = [model.output_layer.register_forward_pre_hook(self._observe_lm_head)]
        for prefix in ("decoder", "mtp"):
            stack = getattr(model, prefix, None)
            if stack is None:
                continue
            for name, module in stack.named_modules(prefix=prefix):
                if isinstance(module, (te.Linear, te.LayerNormLinear, te.LayerNormMLP)):
                    self.module_names.append(name)
                    self._handles.append(
                        module.register_forward_pre_hook(partial(self._observe, name))
                    )

    def _current(self):
        """The phase and the quantization state at this moment."""
        from transformer_engine.pytorch.distributed import in_fp8_activation_recompute_phase
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        from megatron.core.transformer import cuda_graphs

        if torch.cuda.is_current_stream_capturing():
            phase = "capture"
        elif cuda_graphs.is_graph_capturing():
            phase = "warmup"
        else:
            phase = self.phase
        if in_fp8_activation_recompute_phase():
            phase += "/recompute"
        if FP8GlobalStateManager.is_fp8_enabled():
            precision = type(FP8GlobalStateManager.get_fp8_recipe()).__name__
        else:
            precision = _BF16
        return phase, precision

    def _observe(self, name, module, args):
        phase, precision = self._current()
        self.observed.setdefault(phase, {}).setdefault(name, set()).add(precision)

    def _observe_lm_head(self, module, args):
        phase, precision = self._current()
        self.lm_head_observed.setdefault(phase, set()).add(precision)

    def expected(self, num_layers, quantized=_NVFP4, mtp=None):
        """The eager precision of every hooked module when the first and the last decoder layer
        are BF16: ``quantized`` for the other decoder layers and, unless ``mtp`` names another
        precision, for the MTP stack."""
        precisions = {}
        for name in self.module_names:
            stack, _, layer_index = name.split(".")[:3]
            if stack == "decoder":
                is_boundary_layer = int(layer_index) in (0, num_layers - 1)
                precisions[name] = {_BF16 if is_boundary_layer else quantized}
            else:
                precisions[name] = {mtp or quantized}
        return precisions

    def check(self, phases, expected, recompute):
        """Every module ran in its ``expected`` precision in each of ``phases``, and nothing ran
        in any other phase. With activation recompute, the modules a phase recomputes ran in the
        same precision again, and every phase recomputes the same modules."""
        recompute_phases = [f"{phase}/recompute" for phase in phases] if recompute else []
        assert set(self.observed) == {*phases, *recompute_phases}
        for phase in phases:
            assert self.observed[phase] == expected, phase
        for phase in recompute_phases:
            recomputed = self.observed[phase]
            assert recomputed == {name: expected[name] for name in recomputed}, phase
            assert recomputed.keys() == self.observed[recompute_phases[0]].keys(), phase

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _relative_error(values, reference):
    """Relative L2 error of a tensor, or of a ``{name: tensor}`` dict taken as one vector."""
    if torch.is_tensor(values):
        values, reference = {"": values}, {"": reference}
    assert values.keys() == reference.keys()
    error = sum((values[name] - ref).double().pow(2).sum() for name, ref in reference.items())
    norm = sum(ref.double().pow(2).sum() for ref in reference.values())
    return (error / norm).sqrt().item()


def _build_nvfp4_gpt_model(cuda_graph_impl, **config_overrides):
    """Fixed-length 4-layer GPTModel under ``_NVFP4_FIRST_LAST_BF16``. Every build holds the
    same weights and starts from the same RNG state (NVFP4 rounds gradients stochastically)."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend

    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    config = _config(
        hidden_size=256,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.unfused,  # like the other fixed-length chunk-graph tests
        hidden_dropout=0.0,
        attention_dropout=0.0,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_modules=[],
        use_te_rng_tracker=True,
        **_NVFP4_FIRST_LAST_BF16,
        **config_overrides,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=_VOCAB,
        max_sequence_length=_SEQ_LENGTH,
    )
    return model.bfloat16().cuda()


def _build_quantized_thd_model(model_type, quantization=None, **config_overrides):
    """Packed-sequence (THD) chunk-graph model with one MTP layer under ``quantization``
    (default ``_NVFP4_FIRST_LAST_BF16``): a 4-layer GPTModel, or a HybridModel whose ``*-*-``
    decoder holds attention and MLP as separate layers (0: attention and 3: MLP are the BF16
    ones) and whose MTP is a ``*-`` stack of the decoder's class."""
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    config = _config(
        hidden_size=256,
        num_query_groups=4,
        ffn_hidden_size=512,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mtp_num_layers=1,
        is_hybrid_model=model_type == "hybrid",
        **(_NVFP4_FIRST_LAST_BF16 if quantization is None else quantization),
        **_CHUNK,
        **config_overrides,
    )
    model_parallel_cuda_manual_seed(42)
    torch.manual_seed(42)
    common = dict(
        config=config,
        vocab_size=_VOCAB,
        max_sequence_length=_SEQ_LENGTH,
        position_embedding_type="rope",
    )
    if model_type == "hybrid":
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
        from megatron.core.models.hybrid.hybrid_model import HybridModel

        model = HybridModel(
            hybrid_stack_spec=hybrid_stack_spec, hybrid_layer_pattern="*-*-/*-", **common
        )
    else:
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
            get_gpt_mtp_block_spec,
        )
        from megatron.core.models.gpt.gpt_model import GPTModel

        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        model = GPTModel(
            transformer_layer_spec=layer_spec,
            mtp_block_spec=get_gpt_mtp_block_spec(config, layer_spec, use_transformer_engine=True),
            **common,
        )
    return model.cuda()


def _fixed_length_step(model, tokens, attention_mask):
    """One next-token-prediction microbatch without an optimizer step; returns the logits and
    ``{name: grad}`` in float32."""
    for param in model.parameters():
        # Caller-owned leaf-grad buffers: with a leaf grad of None, AccumulateGrad may adopt
        # TE's recyclable static grad buffer instead of copying out of it.
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        else:
            param.grad.zero_()
    # Eager: the first microbatch of a step re-quantizes the NVFP4 weights, as the schedule's
    # forward_step requests. A replay gets the same from its microbatch index.
    model.set_is_first_microbatch()
    logits = model(input_ids=tokens, position_ids=tokens, attention_mask=attention_mask)
    # Under graphs the logits are computed from TE's recyclable static output, owned only until
    # the backward: retain the values first.
    values = logits.detach().float().clone()
    labels = (tokens + 1) % logits.shape[-1]
    torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), labels.flatten()).backward()
    torch.cuda.synchronize()
    grads = {name: param.grad.detach().float().clone() for name, param in model.named_parameters()}
    return values, grads


def _init_two_microbatches(micro_batch_size):
    """Two microbatches per step: a capture sizes its TE warm-up from the micro-batch count."""
    from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator

    init_num_microbatches_calculator(
        rank=0,
        rampup_batch_size=None,
        global_batch_size=2 * micro_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
    )


def _perturb_parameters(model, seed):
    """Stand-in for an optimizer step: the same relative perturbation for equal ``seed``."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.shape, generator=generator)
            param.add_((0.02 * param.abs().mean().float().cpu() * noise).to(param))


class TestChunkGraphPrecisionContexts:
    @pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not available")
    def test_use_te_checkpoint_decision(self):
        # FP8 always goes through TE's checkpoint.
        assert use_te_checkpoint(_config(fp8="e4m3", fp8_recipe="mxfp8"))
        # BF16: mcore's checkpoint outside graphs ...
        assert not use_te_checkpoint(_config())
        assert not use_te_checkpoint(_config(cuda_graph_impl="transformer_engine"))
        # ... but TE's checkpoint inside a chunk capture, where mcore's would skip checkpointing.
        assert use_te_checkpoint(_config(**_CHUNK))

    @pytest.mark.skipif(not (HAVE_TE and torch.cuda.is_available()), reason="TE and CUDA required")
    def test_selective_core_attn_recompute_checkpoints_inside_chunk_capture(self):
        """Selective ``core_attn`` recompute: eager builds one checkpoint node per layer, mcore's
        ``tensor_parallel.checkpoint`` builds none while a graph is captured, and under chunk
        granularity the attention uses TE's checkpoint, which does build its node in the capture."""
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer import cuda_graphs
        from megatron.core.transformer.spec_utils import build_module
        from tests.unit_tests.test_utilities import Utils

        def checkpoint_nodes(tensor):
            names, seen, stack = [], set(), [tensor.grad_fn]
            while stack:
                fn = stack.pop()
                if fn is None or fn in seen:
                    continue
                seen.add(fn)
                if "Checkpoint" in type(fn).__name__:
                    names.append(type(fn).__name__)
                stack.extend(next_fn for next_fn, _ in fn.next_functions)
            return sorted(names)

        def run(**overrides):
            config = _config(
                num_layers=1,
                params_dtype=torch.bfloat16,
                attention_dropout=0.0,  # graphed attention recompute asserts no dropout
                hidden_dropout=0.0,
                recompute_granularity="selective",
                recompute_modules=["core_attn"],
                **overrides,
            )
            layer = build_module(
                get_gpt_layer_with_transformer_engine_spec(), config=config, layer_number=1
            ).cuda()
            layer.train()
            hidden = torch.randn(16, 2, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
            output, _ = layer(hidden, attention_mask=None)
            return checkpoint_nodes(output)

        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        model_parallel_cuda_manual_seed(42)
        try:
            # Autograd names the node of a custom Function "<class name>Backward".
            megatron_node = tensor_parallel_random.CheckpointFunction.__name__ + "Backward"
            eager_nodes = run()
            assert eager_nodes == [megatron_node]  # eager: mcore's node
            cuda_graphs._set_capture_start()
            try:
                assert run() == []  # mcore's checkpoint skips its node while a graph is captured
                chunk_nodes = run(**_CHUNK)  # Transformer Engine's checkpoint does not
                assert len(chunk_nodes) == 1 and chunk_nodes != eager_nodes
            finally:
                cuda_graphs._set_capture_end()
        finally:
            Utils.destroy_model_parallel()

    @pytest.mark.launch_on_gb200  # MXFP8 needs Blackwell; elsewhere a per-tensor recipe stands in
    @pytest.mark.skipif(not (HAVE_TE and torch.cuda.is_available()), reason="TE and CUDA required")
    def test_layer_fp8_context_opts_bf16_boundary_layers_out_of_a_chunk_capture(self):
        """Layer 0 and the last layer are BF16; inside an outer FP8 context (what the chunk
        capture applies to the whole block) their per-layer context must disable FP8."""
        import transformer_engine.pytorch as te
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        fp8_available, reason = FP8GlobalStateManager.is_fp8_available()
        if not fp8_available:
            pytest.skip(f"FP8 not available on this device: {reason}")
        # The test observes the FP8 *state* only, so any recipe first_last_layers_bf16 accepts
        # will do: MXFP8 needs Blackwell, per-tensor current scaling also runs on Hopper.
        recipe = "mxfp8" if FP8GlobalStateManager.is_mxfp8_available()[0] else "tensorwise"
        config = _config(
            fp8="e4m3",
            fp8_recipe=recipe,
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=1,
            num_layers_at_end_in_bf16=1,
            **_CHUNK,
        )
        with te.fp8_autocast(enabled=True, fp8_recipe=get_fp8_recipe(config)):
            assert FP8GlobalStateManager.is_fp8_enabled()
            # decoder layers 0 and 3 are BF16; an MTP layer (index >= num_layers) keeps the
            # FP8 of the enclosing MTP-block context, exactly as in eager.
            for layer_no, expect_fp8 in ((0, False), (1, True), (2, True), (3, False), (4, True)):
                with get_layer_fp8_context(config, layer_no):
                    assert FP8GlobalStateManager.is_fp8_enabled() is expect_fp8, layer_no
            # The MTP stack numbers its layers from 1: layer 0 there is an FP8 layer, not the
            # first BF16 decoder layer.
            with get_layer_fp8_context(config, 0, is_mtp_layer=True):
                assert FP8GlobalStateManager.is_fp8_enabled()
            # The block's per-layer context is this helper (layers are numbered from 1).
            block = object.__new__(TransformerBlock)
            object.__setattr__(block, "config", config)
            with block._get_inner_quantization_context(SimpleNamespace(layer_number=1)):
                assert not FP8GlobalStateManager.is_fp8_enabled()
            with block._get_inner_quantization_context(SimpleNamespace(layer_number=2)):
                assert FP8GlobalStateManager.is_fp8_enabled()
        # During warm-up and capture, TE's autocast wrapper on the graphed block *class* runs
        # the MTP stack (same class, but captured inside the post-process block's graph rather
        # than a graphed callable itself) with FP8 *disabled*; its layers must switch FP8 back
        # on themselves (their parameters are FP8), while a BF16 decoder layer stays off.
        with te.fp8_autocast(enabled=False, fp8_recipe=get_fp8_recipe(config)):
            with get_layer_fp8_context(config, 0, is_mtp_layer=True):
                assert FP8GlobalStateManager.is_fp8_enabled()
            with get_layer_fp8_context(config, 1):
                assert FP8GlobalStateManager.is_fp8_enabled()
            with get_layer_fp8_context(config, 0):
                assert not FP8GlobalStateManager.is_fp8_enabled()
        # Outside chunk captures the boundary layers simply get no FP8 context of their own
        # (per-layer capture never runs them under an outer one).
        config.cuda_graph_impl = "transformer_engine"
        with te.fp8_autocast(enabled=True, fp8_recipe=get_fp8_recipe(config)):
            with get_layer_fp8_context(config, 0):
                assert FP8GlobalStateManager.is_fp8_enabled()
        # A chunk graph replay runs under the block-wide FP8 autocast TransformerBlock.forward
        # would have opened (the replay does not run the forward).
        config.cuda_graph_impl = "chunk_wise"
        assert not FP8GlobalStateManager.is_fp8_enabled()
        with block._chunk_graph_replay_quantization_context():
            assert FP8GlobalStateManager.is_fp8_enabled()
        assert not FP8GlobalStateManager.is_fp8_enabled()

    def test_chunk_replay_runs_under_the_block_fp8_context(self, monkeypatch):
        """Replay enters the graphed callable without running the block forward, so the chunk
        callable itself opens the FP8 autocast that TE's graphed callable reads its amax-reduction
        and weight-cache bookkeeping from (delayed scaling never updates its scales otherwise).
        Without FP8 the replay opens nothing."""
        events = []

        @contextmanager
        def recording_fp8_context(config, layer_no=-1, is_init=False):
            events.append(("fp8", layer_no, is_init))
            yield
            events.append("exit")

        monkeypatch.setattr(fp8_utils, "get_fp8_context", recording_fp8_context)

        @contextmanager
        def recording_fp4_context(config, layer_no=-1, is_init=False):
            events.append(("fp4", layer_no, is_init))
            yield
            events.append("exit")

        monkeypatch.setattr(fp4_utils, "get_fp4_context", recording_fp4_context)

        class GraphedBase:
            def _te_cuda_graph_replay(self, *args, **kwargs):
                events.append(("replay", len(args), tuple(sorted(kwargs))))
                return "output"

        class Block(ChunkCudaGraphBlockMixin, GraphedBase):
            def __init__(self, config):
                self.config = config
                self.pre_process = True
                self.cuda_graphs = [object()]
                self.current_microbatch = 0

        def replay(**config):
            events.clear()
            block = Block(
                SimpleNamespace(cuda_graph_impl="chunk_wise", sequence_parallel=False, **config)
            )
            hidden_states = torch.ones(2, 1, 4)
            assert block._te_cuda_graph_replay(hidden_states, attention_mask=None) == "output"
            return list(events)

        # layer_no=-1 is the block-wide context, exactly what TransformerBlock.forward opens.
        assert replay(fp8="hybrid", pipeline_model_parallel_size=1) == [
            ("fp8", -1, False),
            ("replay", 1, ()),
            "exit",
        ]
        assert replay(fp8=None, pipeline_model_parallel_size=1) == [("replay", 1, ())]
        # NVFP4 replays open the block-wide FP4 autocast the same way (TE's weight cache reads
        # is_first_microbatch from it; without it every replay re-quantizes the BF16 weights).
        assert replay(fp8=None, fp4="e2m1", pipeline_model_parallel_size=1) == [
            ("fp4", -1, False),
            ("replay", 1, ()),
            "exit",
        ]
        assert replay(fp8=None, fp4=None, pipeline_model_parallel_size=1) == [("replay", 1, ())]

    def test_layer_quantization_context_dispatch_under_chunk_graphs(self, monkeypatch):
        """``get_layer_quantization_context`` mirrors ``get_layer_fp8_context`` for NVFP4: under
        chunk graphs a BF16 boundary layer gets the quantization-*disabled* context (it would
        otherwise inherit the block-wide NVFP4 capture autocast), an MTP layer opens the FP4
        context itself (TE runs the MTP's inner stack with quantization disabled during the
        capture), and every other layer gets its FP4 context. Outside chunk graphs the FP4 path
        is exactly ``get_fp4_context``; FP8 configs go to ``get_layer_fp8_context``."""
        calls = []
        monkeypatch.setattr(
            fp4_utils,
            "get_fp4_context",
            lambda config, layer_no=-1, is_init=False: calls.append(("fp4", layer_no)),
        )
        monkeypatch.setattr(
            fp4_utils,
            "get_fp8_disabled_context",
            lambda config, is_init=False: calls.append(("disabled",)),
        )
        monkeypatch.setattr(
            fp4_utils,
            "get_layer_fp8_context",
            lambda config, layer_no, is_mtp_layer=False: calls.append(
                ("fp8", layer_no, is_mtp_layer)
            ),
        )

        def config(**overrides):
            kwargs = dict(
                fp8=None,
                fp4="e2m1",
                cuda_graph_impl="chunk_wise",
                num_layers=4,
                first_last_layers_bf16=True,
                num_layers_at_start_in_bf16=1,
                num_layers_at_end_in_bf16=1,
            )
            kwargs.update(overrides)
            return SimpleNamespace(**kwargs)

        def dispatch(cfg, layer_no, **kwargs):
            calls.clear()
            get_layer_quantization_context(cfg, layer_no, **kwargs)
            return calls[-1]

        chunk = config()
        # Decoder layers 0 and 3 are BF16, 1 and 2 are FP4; an index past num_layers is an MTP
        # layer of the GPT MTP block, which keeps the FP4 of its enclosing context like eager.
        assert dispatch(chunk, 0) == ("disabled",)
        assert dispatch(chunk, 1) == ("fp4", 1)
        assert dispatch(chunk, 2) == ("fp4", 2)
        assert dispatch(chunk, 3) == ("disabled",)
        assert dispatch(chunk, 4) == ("fp4", 4)
        # The hybrid MTP stack numbers its layers from 1: layer 0 there is an FP4 layer.
        assert dispatch(chunk, 0, is_mtp_layer=True) == ("fp4", -1)
        # Same for the FP4-only helper.
        calls.clear()
        get_layer_fp4_context(chunk, 0)
        assert calls == [("disabled",)]
        # Without first_last_layers_bf16 every layer is FP4.
        assert dispatch(config(first_last_layers_bf16=False), 0) == ("fp4", 0)
        # Eager / per-layer TE graphs: unchanged, the boundary layer gets no context of its own.
        for impl in ("none", "transformer_engine", "local"):
            assert dispatch(config(cuda_graph_impl=impl), 0) == ("fp4", 0)
            assert dispatch(config(cuda_graph_impl=impl), 0, is_mtp_layer=True) == ("fp4", 0)
        # FP8 configs are handled by get_layer_fp8_context; FP4 is never consulted.
        assert dispatch(config(fp8="e4m3", fp4=None), 0, is_mtp_layer=True) == ("fp8", 0, True)
        # Neither: no context at all.
        calls.clear()
        assert isinstance(get_layer_quantization_context(config(fp4=None), 0), nullcontext)
        assert calls == []
        # The decoder block's per-layer context is this helper (layers are numbered from 1).
        block = object.__new__(TransformerBlock)
        object.__setattr__(block, "config", chunk)
        calls.clear()
        block._get_inner_quantization_context(SimpleNamespace(layer_number=1))
        block._get_inner_quantization_context(SimpleNamespace(layer_number=2))
        assert calls == [("disabled",), ("fp4", 1)]
        if HAVE_TE:
            # The same combination as a real TransformerConfig: it builds without NVFP4
            # hardware (the recipe is only created when a context is opened) and selects the
            # contexts the namespace above does.
            real = _config(**_NVFP4_FIRST_LAST_BF16, **_CHUNK)
            assert real.cuda_graph_impl == "chunk_wise" and real.fp4 == "e2m1" and not real.fp8
            assert use_te_checkpoint(real)
            assert [dispatch(real, layer_no) for layer_no in range(5)] == [
                ("disabled",),
                ("fp4", 1),
                ("fp4", 2),
                ("disabled",),
                ("fp4", 4),
            ]
            assert dispatch(real, 0, is_mtp_layer=True) == ("fp4", -1)

    @pytest.mark.launch_on_gb200  # NVFP4 needs Blackwell: skipped everywhere else
    @pytest.mark.skipif(not (HAVE_TE and torch.cuda.is_available()), reason="TE and CUDA required")
    def test_layer_fp4_context_opts_bf16_boundary_layers_out_of_a_chunk_capture(self):
        """NVFP4 variant of the FP8 test above: inside an outer NVFP4 autocast (what the chunk
        capture applies to the whole block) the BF16 boundary layers' per-layer context must
        disable quantization, and inside a disabled one (how TE runs the MTP's inner stack during
        the capture) the MTP layers must re-enable it."""
        import transformer_engine.pytorch as te
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        _skip_unless_nvfp4_is_available()
        config = _config(
            fp4="e2m1",
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=1,
            num_layers_at_end_in_bf16=1,
            **_CHUNK,
        )
        recipe = get_fp4_recipe(config)

        def quantization_enabled():
            # TE drives NVFP4 through the FP8 autocast state; the recipe tells the two apart.
            if not FP8GlobalStateManager.is_fp8_enabled():
                return False
            assert type(FP8GlobalStateManager.get_fp8_recipe()) is type(recipe)
            return True

        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            assert quantization_enabled()
            for layer_no, expect_fp4 in ((0, False), (1, True), (2, True), (3, False), (4, True)):
                with get_layer_fp4_context(config, layer_no):
                    assert quantization_enabled() is expect_fp4, layer_no
                with get_layer_quantization_context(config, layer_no):
                    assert quantization_enabled() is expect_fp4, layer_no
            with get_layer_fp4_context(config, 0, is_mtp_layer=True):
                assert quantization_enabled()
            block = object.__new__(TransformerBlock)
            object.__setattr__(block, "config", config)
            with block._get_inner_quantization_context(SimpleNamespace(layer_number=1)):
                assert not quantization_enabled()
            with block._get_inner_quantization_context(SimpleNamespace(layer_number=2)):
                assert quantization_enabled()
        # TE's wrapper runs the (non-graphed) MTP stack with quantization disabled during the
        # capture: its FP4 layers switch NVFP4 back on, a BF16 decoder layer stays off.
        with te.fp8_autocast(enabled=False, fp8_recipe=recipe):
            with get_layer_fp4_context(config, 0, is_mtp_layer=True):
                assert quantization_enabled()
            with get_layer_fp4_context(config, 1):
                assert quantization_enabled()
            with get_layer_fp4_context(config, 0):
                assert not quantization_enabled()
        # Eager: the boundary layer gets no context of its own, as before.
        config.cuda_graph_impl = "transformer_engine"
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            with get_layer_fp4_context(config, 0):
                assert quantization_enabled()
        # A chunk graph replay runs under the block-wide NVFP4 autocast.
        config.cuda_graph_impl = "chunk_wise"
        assert not FP8GlobalStateManager.is_fp8_enabled()
        with block._chunk_graph_replay_quantization_context():
            assert quantization_enabled()
        assert not FP8GlobalStateManager.is_fp8_enabled()

    def test_postprocess_chunk_replay_opens_fp8_only_with_quantized_modules(self):
        """The post-process callable (LM head, loss and optionally MTP) holds TE modules that run
        under FP8 only with MTP layers on this stage or the MXFP8 output projection; otherwise its
        replay opens no autocast, exactly like the eager post-process."""

        def block(fp8, mtp_num_layers, mtp_process, **config_overrides):
            config = dict(
                cuda_graph_impl="chunk_wise",
                fp8=fp8,
                fp8_recipe="delayed",
                fp8_output_proj=False,
                mtp_num_layers=mtp_num_layers,
            )
            config.update(config_overrides)
            post = object.__new__(ChunkCudaGraphPostProcessBlock)
            object.__setattr__(post, "config", SimpleNamespace(**config))
            object.__setattr__(post, "_owner", SimpleNamespace(mtp_process=mtp_process))
            return post

        assert not block("hybrid", None, False)._chunk_graph_has_quantized_modules()
        # The MTP block lives on another stage.
        assert not block("hybrid", 1, False)._chunk_graph_has_quantized_modules()
        assert block("hybrid", 1, True)._chunk_graph_has_quantized_modules()
        assert (
            block(
                "e4m3", None, False, fp8_output_proj=True, fp8_recipe="mxfp8"
            )._chunk_graph_has_quantized_modules()
            is HAVE_TE
        )
        assert isinstance(
            block("hybrid", None, False)._chunk_graph_replay_quantization_context(), nullcontext
        )
        assert isinstance(
            block(None, 1, True)._chunk_graph_replay_quantization_context(), nullcontext
        )
        # An MTP that keep_mtp_in_bf16 holds out of the quantization is no quantized module:
        # the callable is captured and replayed without an autocast ...
        kept_in_bf16 = block("hybrid", 1, True, keep_mtp_in_bf16=True)
        assert not kept_in_bf16._chunk_graph_has_quantized_modules()
        assert isinstance(kept_in_bf16._chunk_graph_replay_quantization_context(), nullcontext)
        # ... unless the MXFP8 output projection next to it needs one.
        assert (
            block(
                "e4m3", 1, True, keep_mtp_in_bf16=True, fp8_output_proj=True, fp8_recipe="mxfp8"
            )._chunk_graph_has_quantized_modules()
            is HAVE_TE
        )

    @pytest.mark.launch_on_gb200  # MXFP8 needs Blackwell; elsewhere a per-tensor recipe stands in
    @pytest.mark.skipif(not (HAVE_TE and torch.cuda.is_available()), reason="TE and CUDA required")
    def test_mtp_layer_kept_in_bf16_opts_out_of_an_enclosing_chunk_capture_autocast(self):
        """With ``keep_mtp_in_bf16`` the MTP layer's copy of the config has no FP8, so the layer
        opens no context of its own. A chunk capture records the post-process under the model's
        autocast when the LM head is quantized; the layer then disables quantization itself.
        Without an enclosing autocast (eager) the context changes nothing, and outside chunk
        graphs or for a layer that is not held out of a quantized model it is ``nullcontext``."""
        import transformer_engine.pytorch as te
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

        fp8_available, reason = FP8GlobalStateManager.is_fp8_available()
        if not fp8_available:
            pytest.skip(f"FP8 not available on this device: {reason}")
        recipe = "mxfp8" if FP8GlobalStateManager.is_mxfp8_available()[0] else "tensorwise"
        config = _config(
            fp8="e4m3", fp8_recipe=recipe, mtp_num_layers=1, keep_mtp_in_bf16=True, **_CHUNK
        )
        layer = object.__new__(MultiTokenPredictionLayer)
        # Not held out of a quantized model (the class default).
        assert isinstance(layer._get_bf16_context(), nullcontext)

        object.__setattr__(layer, "quantized_model_config", config)
        with te.fp8_autocast(enabled=True, fp8_recipe=get_fp8_recipe(config)):
            with layer._get_bf16_context():
                assert not FP8GlobalStateManager.is_fp8_enabled()
            assert FP8GlobalStateManager.is_fp8_enabled()
        with layer._get_bf16_context():
            assert not FP8GlobalStateManager.is_fp8_enabled()
        assert not FP8GlobalStateManager.is_fp8_enabled()

        config.cuda_graph_impl = "transformer_engine"
        assert isinstance(layer._get_bf16_context(), nullcontext)


@pytest.mark.launch_on_gb200  # NVFP4 needs Blackwell: skipped everywhere else
@pytest.mark.skipif(not (HAVE_TE and torch.cuda.is_available()), reason="TE and CUDA required")
class TestNvfp4ChunkGraphEndToEnd:
    """Real chunk-graph capture and replay of NVFP4 models with BF16 first / last layers.

    For warm-up and capture, Transformer Engine wraps the ``__call__`` of the graphed block
    *class* in an enabled NVFP4 autocast. A BF16 boundary layer that opens no context of its own
    (``get_fp4_context`` returns ``nullcontext`` for it) inherited that autocast: eager ran it in
    BF16, every replay in NVFP4. An MTP stack of the decoder's class (hybrid models; recorded
    inside the post-process graph, not a graphed callable itself) got the wrapper with
    quantization *disabled*, so its first layer, an NVFP4 layer in eager, was recorded in BF16.
    ``get_layer_fp4_context`` makes both kinds of layer state their precision; these tests fail
    without it.

    ``keep_mtp_in_bf16`` holds the MTP out of the model's quantization: its layers have no
    context of their own, so they run in whatever state the post-process callable runs under.
    A post-process without a quantized module is therefore captured without an autocast, and
    one whose LM head is quantized (the MXFP8 output projection) is captured under it while
    the MTP layer disables quantization for itself; the ``mtp-bf16`` cases fail without these.

    Compared against eager:

    * the precision every TE GEMM module executes in during TE's warm-up passes and in the
      recorded forward (``_LayerPrecisionRecorder``): deterministic, and what a replay runs;
    * numerics. NVFP4 rounds stochastically only where it quantizes gradients, so the forward
      (logits, losses) and the gradients of the parameters between the last NVFP4 layer and the
      loss are deterministic and compared tightly. Measured on GB200: bit-identical to eager
      with the fix; without it the fixed-length logits are off by 25 % (relative L2), the
      per-token THD loss by up to 0.3 and those gradients by 17-32 %.
    """

    @pytest.fixture
    def capture_cleanups(self):
        """Process-global state of a capture test, put back even when the test fails. Yields
        the list of cleanup callables the test appends to (run in reverse order)."""
        from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
        from megatron.core.tensor_parallel.random import initialize_rng_tracker
        from megatron.core.transformer import cuda_graphs
        from tests.unit_tests.test_utilities import Utils

        _skip_unless_nvfp4_is_available()
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        # A failed earlier GPU test can leave the calculator initialized.
        destroy_num_microbatches_calculator()
        # The conftest fixture puts the previous tracker back after the test.
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        stream = torch.cuda.current_stream()
        cleanups = []
        try:
            yield cleanups
        finally:
            torch.cuda.synchronize()
            for cleanup in reversed(cleanups):
                cleanup()
            # Only still set if the capture raised.
            cuda_graphs._set_capture_end()
            cuda_graphs.te_set_capture_end()
            torch.cuda.set_stream(stream)
            destroy_num_microbatches_calculator()
            Utils.destroy_model_parallel()

    @staticmethod
    def _capture(model, micro_batch_size, cleanups):
        """Capture the chunk graphs of ``model`` (decoder block and, for packed sequences, the
        post-process block)."""
        from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

        # The helper's post-capture reset calls the DDP wrapper's zero_grad_buffer(); these
        # tests drive the bare model and zero the gradients themselves.
        model.zero_grad_buffer = lambda: None
        helper = TECudaGraphHelper(
            model=[model],
            config=model.config,
            seq_length=_SEQ_LENGTH,
            micro_batch_size=micro_batch_size,
        )
        # TE wraps the __call__ of the graphed classes while it captures and unwraps them only
        # when the capture returns.
        class_calls = {
            type(block): vars(type(block)).get("__call__")
            for block in (model.decoder, getattr(model, "postprocess_block", None))
            if block is not None
        }

        def release():
            if helper.graphs_created():
                helper.delete_cuda_graphs()
            for cls, call in class_calls.items():
                if call is not None:
                    cls.__call__ = call
                elif "__call__" in vars(cls):
                    del cls.__call__

        cleanups.append(release)
        helper.create_cudagraphs()
        assert helper.graphs_created()
        assert len(model.decoder.cuda_graphs) > 0

    @pytest.mark.parametrize("recompute", [None, "full"])
    def test_fixed_length_capture_and_replay_keep_the_eager_layer_precision(
        self, capture_cleanups, recompute
    ):
        """Fixed-length (SBHD) GPTModel: layers 0 and 3 BF16, layers 1 and 2 NVFP4, against an
        eager reference model without graphs. With full recompute the per-layer context is
        ``recompute.checkpointed_forward``'s, and the recompute recorded into the backward graph
        has to select the same precisions again (TE's checkpoint re-enters the block-wide
        capture autocast around it).

        Beyond the class-level checks: a weight update reaches the replay (the first microbatch
        of a step re-quantizes the NVFP4 weights inside the graph), and the autocast the replay
        opens (``_chunk_graph_replay_quantization_context``) is what hands ``is_first_microbatch``
        to TE, so later microbatches reuse the cached NVFP4 weights."""
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        from megatron.core.transformer import cuda_graphs

        num_layers, micro_batch_size = 4, 2
        _init_two_microbatches(micro_batch_size)
        tokens = torch.arange(_SEQ_LENGTH, device="cuda").repeat(micro_batch_size, 1)
        # Causal mask (True: masked), the layout the block's static capture input has.
        attention_mask = (
            ~torch.tril(torch.ones((_SEQ_LENGTH, _SEQ_LENGTH), dtype=torch.bool, device="cuda"))
        ).tile(micro_batch_size, 1, 1, 1)
        config_overrides = _FULL_RECOMPUTE if recompute else {}

        def build(cuda_graph_impl):
            model = _build_nvfp4_gpt_model(cuda_graph_impl, **config_overrides)
            model.train()
            recorder = _LayerPrecisionRecorder(model)
            capture_cleanups.append(recorder.remove)
            return model, recorder

        def step(model):
            return _fixed_length_step(model, tokens, attention_mask)

        # Eager reference, twice: the forward is deterministic, the difference of the two
        # gradients is NVFP4's stochastic rounding.
        eager_model, eager_recorder = build("none")
        eager_logits, eager_grads = step(eager_model)
        rerun_logits, rerun_grads = step(eager_model)
        expected = eager_recorder.expected(num_layers)
        assert {_BF16} in expected.values() and {_NVFP4} in expected.values()
        eager_recorder.check(["eager"], expected, recompute)
        torch.testing.assert_close(rerun_logits, eager_logits, rtol=1e-2, atol=1e-2)
        rounding_noise = _relative_error(rerun_grads, eager_grads)

        # The same weights under chunk graphs: eager before the capture, like the reference.
        graph_model, recorder = build("chunk_wise")
        logits, _ = step(graph_model)
        torch.testing.assert_close(logits, eager_logits, rtol=1e-2, atol=1e-2)

        self._capture(graph_model, micro_batch_size, capture_cleanups)
        # Each layer is warmed up and recorded in the precision eager runs it in.
        recorder.check(["eager", "warmup", "capture"], expected, recompute)

        recorder.phase = "replay"
        logits, grads = step(graph_model)
        torch.testing.assert_close(logits, eager_logits, rtol=1e-2, atol=1e-2)
        assert grads.keys() == eager_grads.keys()
        for name, grad in grads.items():
            assert torch.isfinite(grad).all(), name
        # The last (BF16) layer, the final norm and the LM head sit between the last NVFP4 layer
        # and the loss: no stochastic rounding reaches their gradients.
        deterministic = {
            name: grad
            for name, grad in eager_grads.items()
            if name.startswith(
                (f"decoder.layers.{num_layers - 1}.", "decoder.final_layernorm.", "output_layer.")
            )
        }
        assert len(deterministic) > 2
        assert _relative_error({name: grads[name] for name in deterministic}, deterministic) <= 1e-2
        # Every other gradient carries the stochastic rounding of the NVFP4 backward, so the
        # replay differs from eager about as much as eager differs from itself (measured: 1.0x
        # that noise). A sanity bound on the quantized backward, not a precision check.
        assert _relative_error(grads, eager_grads) <= max(2 * rounding_noise, 1e-2)

        # A weight update reaches the replay.
        for model in (eager_model, graph_model):
            _perturb_parameters(model, seed=7)
        updated_logits, _ = step(eager_model)
        assert _relative_error(updated_logits, eager_logits) > 5e-2
        logits, _ = step(graph_model)
        torch.testing.assert_close(logits, updated_logits, rtol=1e-2, atol=1e-2)

        # That was microbatch 0. The graphed callable is the first module of the autocast each
        # replay opens, which is how TE's skip-weight-update flag follows the microbatch: later
        # microbatches of a step reuse the NVFP4 weights microbatch 0 cached and still compute
        # what eager computes. Without the autocast the flag would not follow the index.
        def skip_weight_update():
            return FP8GlobalStateManager.get_skip_fp8_weight_update_tensor().item()

        assert skip_weight_update() == 0
        for microbatch in (1, 0, 1):
            cuda_graphs.set_current_microbatch(graph_model, microbatch)
            logits, _ = step(graph_model)
            assert skip_weight_update() == microbatch, microbatch
            torch.testing.assert_close(logits, updated_logits, rtol=1e-2, atol=1e-2)

        # No decoder module ran its forward after the capture: every step above was a replay.
        recorder.check(["eager", "warmup", "capture"], expected, recompute)

    # id -> (quantization, further config, precision of the quantized decoder layers, precision
    # of the MTP stack, precision the post-process callable is warmed up and captured under).
    _THD_PRECISIONS = {
        "nvfp4": (_NVFP4_FIRST_LAST_BF16, {}, _NVFP4, _NVFP4, _NVFP4),
        "nvfp4-mtp-bf16": (
            _NVFP4_FIRST_LAST_BF16,
            dict(keep_mtp_in_bf16=True),
            _NVFP4,
            _BF16,
            _BF16,
        ),
        "mxfp8-mtp-bf16": (
            _MXFP8_FIRST_LAST_BF16,
            dict(keep_mtp_in_bf16=True),
            _MXFP8,
            _BF16,
            _BF16,
        ),
        "mxfp8-output-proj-mtp-bf16": (
            _MXFP8_FIRST_LAST_BF16,
            dict(keep_mtp_in_bf16=True, fp8_output_proj=True),
            _MXFP8,
            _BF16,
            _MXFP8,
        ),
    }

    @pytest.mark.parametrize(
        "model_type, recompute, precision",
        [
            ("gpt", None, "nvfp4"),
            ("hybrid", None, "nvfp4"),
            ("hybrid", "full", "nvfp4"),
            ("gpt", None, "nvfp4-mtp-bf16"),
            ("hybrid", None, "nvfp4-mtp-bf16"),
            ("gpt", None, "mxfp8-mtp-bf16"),
            ("gpt", None, "mxfp8-output-proj-mtp-bf16"),
        ],
    )
    def test_thd_mtp_capture_and_replay_keep_the_eager_layer_precision(
        self, capture_cleanups, model_type, recompute, precision
    ):
        """Packed sequences (THD) with MTP: the post-process (MTP, LM head, loss) is a second
        captured callable. ``gpt``: the MTP layer opens the NVFP4 context around its transformer
        layer itself. ``hybrid``: the MTP's inner stack is of the decoder's class, so TE runs it
        with quantization disabled during the capture and its layers have to switch NVFP4 back
        on (the stack numbers its layers from 1: by index its first layer would be a BF16
        boundary layer); with full recompute both stacks take their per-layer context in
        ``recompute.checkpointed_forward``. ``mtp-bf16``: ``keep_mtp_in_bf16`` keeps every MTP
        module in BF16, also next to an MXFP8 LM head (``output-proj``, GPTModel only), for
        which the post-process is captured under the MXFP8 autocast. The eager reference is the
        same model before the capture."""
        from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
        from tests.unit_tests.transformer.test_thd_cuda_graph import _thd_batch, _train_step

        quantization, overrides, quantized, mtp_precision, callable_precision = (
            self._THD_PRECISIONS[precision]
        )
        if quantized == _MXFP8:
            _skip_unless_mxfp8_is_available()
        num_layers, micro_batch_size = 4, 1
        _init_two_microbatches(micro_batch_size)
        model = _build_quantized_thd_model(
            model_type, quantization, **overrides, **(_FULL_RECOMPUTE if recompute else {})
        )
        model.train()
        # MTP, LM head and loss are captured as the post-process callable.
        assert model.postprocess_block is not None and model.mtp_process
        recorder = _LayerPrecisionRecorder(model)
        capture_cleanups.append(recorder.remove)
        expected = recorder.expected(num_layers, quantized, mtp_precision)
        mtp_expected = [precisions for name, precisions in expected.items() if name[:4] == "mtp."]
        assert mtp_expected and all(precisions == {mtp_precision} for precisions in mtp_expected)
        assert {_BF16} in expected.values() and {quantized} in expected.values()

        # Three packings under the static capacity (the padding tail is a dummy sequence).
        batches = [
            _thd_batch(seqlens, _VOCAB, seed, _SEQ_LENGTH, _CHUNK["thd_max_packed_sequences"])
            for seed, seqlens in enumerate(([120], [60, 60], [16, 40, 60]))
        ]

        def step(batch):
            MTPLossLoggingHelper.clean_loss_in_tracker()
            per_token_loss, grads = _train_step(model, batch)
            torch.cuda.synchronize()
            # The MTP loss never reaches the model output; it is accumulated for logging.
            return per_token_loss, MTPLossLoggingHelper.tracker["loss_sums"].clone(), grads

        reference = [step(batch) for batch in batches]
        recorder.check(["eager"], expected, recompute)

        self._capture(model, micro_batch_size, capture_cleanups)
        assert len(model.postprocess_block.cuda_graphs) > 0
        # Decoder and MTP layers are warmed up and recorded in the precision eager runs them in.
        recorder.check(["eager", "warmup", "capture"], expected, recompute)
        # Eager opens no autocast around the post-process; the capture opens the model's only
        # for a post-process that holds a quantized module.
        assert recorder.lm_head_observed == {
            "eager": {_BF16},
            "warmup": {callable_precision},
            "capture": {callable_precision},
        }

        recorder.phase = "replay"
        for index in (2, 0, 1):
            per_token_loss, mtp_loss, grads = step(batches[index])
            ref_per_token_loss, ref_mtp_loss, ref_grads = reference[index]
            torch.testing.assert_close(per_token_loss, ref_per_token_loss, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(mtp_loss, ref_mtp_loss, rtol=1e-3, atol=1e-3)
            assert grads.keys() == ref_grads.keys()
            for name, grad in grads.items():
                assert torch.isfinite(grad).all(), (name, index)
            # Deterministic gradients: the LM head sees the outputs of the decoder (main loss)
            # and of the MTP layer (MTP loss); the MTP's final norm lies between the MTP's NVFP4
            # layer and the LM head.
            for name in ("output_layer.weight", "mtp.layers.0.final_layernorm.weight"):
                assert _relative_error(grads[name], ref_grads[name]) <= 1e-2, (name, index)
        # No decoder or MTP module ran its forward after the capture: both callables replayed.
        recorder.check(["eager", "warmup", "capture"], expected, recompute)


@contextmanager
def _te_skip_weight_update_tensor(tensor):
    """Run with ``tensor`` as Transformer Engine's process-global skip-weight-update flag and put
    the previous one back afterwards. TE creates the flag the first time a capture turns on its
    weight caching and keeps it for the life of the process, so ``None`` is the state of a
    training run whose model has no model-wide FP8 / FP4 recipe (in a test process an earlier
    test with such a recipe may have left the flag behind)."""
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    # TE >= 2.15 keeps the flag on quantization_state, older releases on the manager class.
    owner = getattr(FP8GlobalStateManager, "quantization_state", FP8GlobalStateManager)
    previous = owner.skip_fp8_weight_update_tensor
    owner.skip_fp8_weight_update_tensor = tensor
    try:
        yield
    finally:
        owner.skip_fp8_weight_update_tensor = previous


def _build_recipe_fp8_gpt_model(cuda_graph_impl):
    """Fixed-length 4-layer GPTModel *without* a model-wide FP8 / FP4 recipe whose MLP linears
    (``linear_fc1``: a TE LayerNormLinear, ``linear_fc2``: a TE Linear) run in per-tensor FP8
    through a per-module ``quant_recipe``. Their weights are stored in BF16 and quantized in
    the forward. Every build holds the same weights."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.quantization.quant_config import RecipeConfig
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend

    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    # What --te-precision-config-file loads.
    quant_recipe = RecipeConfig.from_config_dict(
        {
            "matchers": {
                "mlp": {
                    "type": "glob",
                    "enabled": True,
                    "pattern": "*mlp.linear_fc*",
                    "config": "fp8",
                }
            },
            "configs": {
                "fp8": {
                    "transformer_engine_config_type": "TEQuantizationParams",
                    "training_recipe": {
                        "fp8_quantization_recipe": "tensorwise",
                        "override_nonquantized_autocast": True,
                    },
                }
            },
        }
    )
    config = _config(
        hidden_size=256,
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.unfused,  # like the other fixed-length chunk-graph tests
        hidden_dropout=0.0,
        attention_dropout=0.0,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_modules=[],
        use_te_rng_tracker=True,
        quant_recipe=quant_recipe,
    )
    assert not config.fp8 and not config.fp4
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=_VOCAB,
        max_sequence_length=_SEQ_LENGTH,
    )
    return model.bfloat16().cuda()


@pytest.mark.launch_on_gb200
class TestQuantRecipeWeightsUnderChunkGraphs:
    """Linears quantized only through a per-module ``quant_recipe`` (``--te-precision-config-file``
    with ``override_nonquantized_autocast``; no model-wide FP8 / FP4 recipe) keep their weights
    in BF16 and quantize them in the forward. Eager, ``is_first_microbatch`` makes Transformer
    Engine quantize once per step and reuse the cached result for the other microbatches.

    A graph replay runs no Python, and the capture turns on TE's replay-driven weight cache
    (``fp8_weight_caching``) only with a model-wide recipe. Without one, a capture that handed
    TE the Python flag (``False`` after the eager passes before it) recorded a read of the
    cached quantized weights and no quantization: every replay computed with the weights of the
    last eager step. ``_resolve_is_first_microbatch`` therefore withholds the flag from such a
    capture, and TE records the quantization itself.
    """

    @pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not available")
    def test_first_microbatch_flag_is_withheld_from_a_capture_without_te_weight_caching(self):
        from megatron.core.extensions.transformer_engine import _resolve_is_first_microbatch
        from megatron.core.transformer import cuda_graphs

        def resolve(is_first_microbatch, cuda_graph_impl="chunk_wise", **quantization):
            config = dict(fp8=None, fp4=None, quant_recipe=object())
            config.update(quantization)
            module = SimpleNamespace(
                config=SimpleNamespace(cuda_graph_impl=cuda_graph_impl, **config),
                disable_parameter_transpose_cache=False,
                is_first_microbatch=is_first_microbatch,
            )
            return _resolve_is_first_microbatch(module)

        with _te_skip_weight_update_tensor(None):
            # Eager: the flag drives TE's weight cache.
            assert resolve(True) is True and resolve(False) is False
            cuda_graphs._set_capture_start()
            try:
                # Warm-up passes and capture of a chunk graph that nothing will tell the
                # microbatch. Per-layer TE graphs keep dev's behaviour.
                assert resolve(True) is None and resolve(False) is None
                assert resolve(True, "transformer_engine") is True
                assert resolve(False, "transformer_engine") is False
            finally:
                cuda_graphs._set_capture_end()
        # A capture that turned on TE's weight caching (model-wide FP8 / FP4 recipe) has TE's
        # flag: TE replaces the Python flag by it while it records, and the warm-up passes keep
        # using the Python flag as before.
        with _te_skip_weight_update_tensor(torch.zeros(1)):
            cuda_graphs._set_capture_start()
            try:
                for quantization in (dict(fp8="e4m3"), dict(fp4="e2m1")):
                    assert resolve(True, **quantization) is True
                    assert resolve(False, **quantization) is False
            finally:
                cuda_graphs._set_capture_end()

    @pytest.fixture
    def capture_cleanups(self):
        """``TestNvfp4ChunkGraphEndToEnd.capture_cleanups`` for any device with FP8, in the
        state of a process in which no capture turned on TE's weight caching."""
        import transformer_engine as te
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
        from megatron.core.tensor_parallel.random import initialize_rng_tracker
        from megatron.core.transformer import cuda_graphs
        from tests.unit_tests.test_utilities import Utils

        if not hasattr(te.common.recipe, "Float8CurrentScaling"):
            pytest.skip("This Transformer Engine has no per-tensor (current scaling) FP8 recipe")
        fp8_available, reason = FP8GlobalStateManager.is_fp8_available()
        if not fp8_available:
            pytest.skip(f"FP8 not available on this device: {reason}")
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        # A failed earlier GPU test can leave the calculator initialized.
        destroy_num_microbatches_calculator()
        # The conftest fixture puts the previous tracker back after the test.
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        stream = torch.cuda.current_stream()
        cleanups = []
        with _te_skip_weight_update_tensor(None):
            try:
                yield cleanups
            finally:
                torch.cuda.synchronize()
                for cleanup in reversed(cleanups):
                    cleanup()
                # Only still set if the capture raised.
                cuda_graphs._set_capture_end()
                cuda_graphs.te_set_capture_end()
                torch.cuda.set_stream(stream)
                destroy_num_microbatches_calculator()
                Utils.destroy_model_parallel()

    @pytest.mark.skipif(not (HAVE_TE and torch.cuda.is_available()), reason="TE and CUDA required")
    def test_weight_update_reaches_the_replay_of_recipe_quantized_linears(self, capture_cleanups):
        """Chunk-graph replays against an eager twin with the same weights, before and after an
        in-place update of the recipe-quantized weights (what an optimizer step does).

        Fails without the fix: the capture records no weight quantization, so after the update
        the replay still returns the logits of the old weights."""
        from megatron.core.transformer import cuda_graphs

        micro_batch_size = 2
        _init_two_microbatches(micro_batch_size)
        tokens = torch.arange(_SEQ_LENGTH, device="cuda").repeat(micro_batch_size, 1)
        # Causal mask (True: masked), the layout the block's static capture input has.
        attention_mask = (
            ~torch.tril(torch.ones((_SEQ_LENGTH, _SEQ_LENGTH), dtype=torch.bool, device="cuda"))
        ).tile(micro_batch_size, 1, 1, 1)

        def build(cuda_graph_impl):
            model = _build_recipe_fp8_gpt_model(cuda_graph_impl)
            model.train()
            quantized = {
                name: module
                for name, module in model.named_modules()
                if getattr(module, "te_quant_params", None) is not None
            }
            assert sorted(quantized) == [
                f"decoder.layers.{layer}.mlp.linear_fc{fc}" for layer in range(4) for fc in (1, 2)
            ]
            return model, list(quantized.values())

        def step(model):
            return _fixed_length_step(model, tokens, attention_mask)

        eager_model, eager_linears = build("none")
        graph_model, graph_linears = build("chunk_wise")
        eager_logits, eager_grads = step(eager_model)
        # The same weights under chunk graphs: eager before the capture, like the reference.
        logits, _ = step(graph_model)
        torch.testing.assert_close(logits, eager_logits, rtol=1e-2, atol=1e-2)
        # Without a model-wide recipe, the matched linears still ran under FP8.
        assert all(linear.fp8 for linear in eager_linears + graph_linears)

        TestNvfp4ChunkGraphEndToEnd._capture(graph_model, micro_batch_size, capture_cleanups)
        logits, grads = step(graph_model)
        torch.testing.assert_close(logits, eager_logits, rtol=1e-2, atol=1e-2)
        assert _relative_error(grads, eager_grads) <= 1e-2

        # The sign flip keeps the final layer norm from hiding the update.
        with torch.no_grad():
            for linear in eager_linears + graph_linears:
                linear.weight.mul_(-2.0)
        updated_logits, updated_grads = step(eager_model)
        assert _relative_error(updated_logits, eager_logits) > 5e-2
        # Every microbatch of the step computes with the updated weights, like eager.
        for microbatch in (0, 1):
            cuda_graphs.set_current_microbatch(graph_model, microbatch)
            logits, grads = step(graph_model)
            torch.testing.assert_close(logits, updated_logits, rtol=1e-2, atol=1e-2)
            assert _relative_error(grads, updated_grads) <= 1e-2, microbatch
