# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Chunk-wise CUDA graph capture and replay of the layer families beyond GQA / MHA and MLA.

``cuda_graph_impl="chunk_wise"`` hands the whole decoder block (and, with packed THD sequences,
the post-process block: LM head and loss) to Transformer Engine's ``make_graphed_callables``.
Every test here builds one small real model of a layer family, captures it with
``TECudaGraphHelper``, replays it on inputs that differ from the capture sample and from each
other, and compares the per-token loss and every parameter gradient with an eager twin
(``cuda_graph_impl="none"``, the same weights) run on the same inputs.

The comparison is as tight as the reference allows: the twin runs every batch twice, and where
its two passes agree bit for bit (every family here on GB200 with Transformer Engine 2.20) the
replay has to be bit-identical to it as well.

Every rank of a multi-rank launch runs the same single-rank model (tensor, context and pipeline
parallel sizes 1), except in the context-parallel DSA test, whose ranks pair up.
"""

import inspect
from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F

from megatron.core import full_cuda_graph, num_microbatches_calculator, parallel_state
from megatron.core.context_parallel import ContextParallelBatch
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer import cuda_graphs
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.utils import get_batch_on_this_cp_rank, is_te_min_version
from tests.unit_tests.test_utilities import Utils, restore_process_state, snapshot_process_state
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    _make_config,
)
from tests.unit_tests.transformer.test_thd_cuda_graph import _thd_batch

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not available"),
]

_VOCAB = 128
# Where the eager reference does not reproduce itself, a replay may be this many times further
# from it than its own second pass was.
_SPREAD_MARGIN = 4.0


_BF16 = dict(bf16=True, params_dtype=torch.bfloat16, use_cpu_initialization=True)


def _graph_config(cuda_graph_impl):
    """No dropout (numerical comparisons) and the CUDA graph implementation under test."""
    return dict(
        hidden_dropout=0.0,
        attention_dropout=0.0,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_modules=[],
    )


def _packed_config(cuda_graph_impl, tokens, max_num_seqs):
    """Packed-sequence (THD) training padded to a static capacity, as chunk graphs require."""
    return dict(
        **_graph_config(cuda_graph_impl),
        cuda_graph_dynamic_microbatches=cuda_graph_impl == "chunk_wise",
        max_seqlen_per_dp_cp_rank=tokens,
        thd_max_packed_sequences=max_num_seqs,
        sequence_packing_scheduler="dp_balanced",
        pad_packed_seq_alignment="max",
        calculate_per_token_loss=True,
    )


@contextmanager
def _capture_state(micro_batch_size, deterministic=False, **model_parallel_sizes):
    """Process state of a chunk capture; everything it sets is put back on exit.

    Yields the list of ``TECudaGraphHelper`` objects whose graphs are deleted on exit.
    """
    # Selects this rank's device, so the device-bound state below is read after it.
    Utils.initialize_model_parallel(**model_parallel_sizes)
    process_state = snapshot_process_state()
    previous_stream = torch.cuda.current_stream()
    previous_calculator = num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR
    previous_pool = full_cuda_graph._shared_graph_pool
    previous_capture_stream = full_cuda_graph._shared_capture_stream
    helpers = []
    try:
        # Chunk graphs are captured into the process-wide graph pool: start from a fresh handle.
        full_cuda_graph._shared_graph_pool = None
        # The capture sizes its Transformer Engine warm-up from the run's micro-batch count.
        num_microbatches_calculator.destroy_num_microbatches_calculator()
        num_microbatches_calculator.init_num_microbatches_calculator(
            rank=0,
            rampup_batch_size=None,
            global_batch_size=micro_batch_size,
            micro_batch_size=micro_batch_size,
            data_parallel_size=1,
        )
        # Graph-safe RNG tracker, as the training loop enables for CUDA graphs.
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        if deterministic:
            # PyTorch's deterministic kernels where it has them (a warning where it has none).
            torch.use_deterministic_algorithms(True, warn_only=True)
        # Like the training loop, run everything on a side stream: AccumulateGrad nodes keep the
        # stream they were created on, and the default stream must not be pulled into a capture.
        training_stream = torch.cuda.Stream()
        training_stream.wait_stream(previous_stream)
        torch.cuda.set_stream(training_stream)
        yield helpers
    finally:
        torch.cuda.synchronize()
        torch.cuda.set_stream(previous_stream)
        if cuda_graphs.is_graph_capturing():
            # A capture that raised leaves Megatron's and Transformer Engine's flags set.
            cuda_graphs._set_capture_end()
            cuda_graphs.te_set_capture_end()
        for helper in helpers:
            if helper.graphs_created():
                helper.delete_cuda_graphs()
        # Also after a capture that raised before any graph existed.
        full_cuda_graph._shared_graph_pool = previous_pool
        full_cuda_graph._shared_capture_stream = previous_capture_stream
        num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = previous_calculator
        Utils.destroy_model_parallel()
        restore_process_state(process_state)


@contextmanager
def _indexer_loss_state(models):
    """The DSA / CSA indexer loss goes through two process-wide objects: the metric tracker the
    layers write into (fixed storage that a captured graph keeps the address of, so it is sized
    before the first forward, like the training setup does) and the backward scale that
    ``forward_step`` publishes after every forward."""
    from megatron.core.transformer.experimental_attention_variant import dsa_logging
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossAutoScaler,
    )

    tracker = dsa_logging.DSAIndexerLossLoggingHelper.tracker
    saved_tracker = dict(tracker)
    saved_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
    tracker.clear()
    DSAIndexerLossAutoScaler.main_loss_backward_scale = None
    try:
        dsa_logging.initialize_dsa_metric_tracker(models)
        DSAIndexerLossAutoScaler.set_loss_scale(torch.ones(1, device="cuda"))
        yield
    finally:
        tracker.clear()
        tracker.update(saved_tracker)
        DSAIndexerLossAutoScaler.main_loss_backward_scale = saved_scale


def _module(model):
    """``model`` itself, or the module a ``DistributedDataParallel`` wrapper drives."""
    return model.module if isinstance(model, DistributedDataParallel) else model


def _start_step(model):
    """Host state and gradient buffers of one training iteration, as the schedules set them."""
    cuda_graphs.set_current_microbatch(model, 0)
    if isinstance(model, DistributedDataParallel):
        model.zero_grad_buffer()
        model.set_is_first_microbatch()
        return
    for param in model.parameters():
        if not param.requires_grad:
            continue
        # Caller-owned leaf-grad buffers: with a leaf grad of None, AccumulateGrad may adopt TE's
        # recyclable static grad buffer instead of copying out of it.
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        else:
            param.grad.zero_()


def _grads(model):
    """Every parameter gradient; under a ``DistributedDataParallel`` wrapper its ``main_grad``."""
    use_main_grad = isinstance(model, DistributedDataParallel)
    return {
        name: (param.main_grad if use_main_grad else param.grad).detach().float().clone()
        for name, param in _module(model).named_parameters()
        if param.requires_grad
    }


def _packed_step(model, batch, **model_kwargs):
    """Forward + backward of one packed micro-batch: (per-token loss, {name: grad})."""
    _start_step(model)
    per_token_loss = model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        attention_mask=None,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        padding_mask=batch["padding_mask"],
        packed_seq_params=batch["packed_seq_params"],
        **model_kwargs,
    )
    # Under graphs the model returns a view of TE's recyclable static output, owned only until
    # its backward: retain the values before crossing that boundary.
    loss_values = per_token_loss.detach().float().clone()
    loss = (per_token_loss.float() * batch["loss_mask"]).sum() / batch["loss_mask"].sum()
    loss.backward()
    return loss_values, _grads(model)


def _hybrid_packed_step(model, batch):
    """As ``pretrain_hybrid.py`` calls the model: with the layout-keyed batch views, which the
    stack receives as ``packed_seq_params_by_layout`` / ``padding_mask_by_layout``."""
    cp_batch = ContextParallelBatch.from_single_layout(
        model.config.linear_cp_layout,
        {"padding_mask": batch["padding_mask"]},
        batch["packed_seq_params"],
    )
    return _packed_step(model, batch, cp_batch=cp_batch)


def _fixed_length_batch(seed, micro_batch_size, seq_length):
    """One fixed-length micro-batch; identical on every rank (CPU seed)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(0, _VOCAB, (micro_batch_size, seq_length + 1), generator=generator)
    tokens = tokens.cuda()
    return dict(
        input_ids=tokens[:, :-1].contiguous(),
        labels=tokens[:, 1:].contiguous(),
        position_ids=torch.arange(seq_length, device="cuda").repeat(micro_batch_size, 1),
    )


def _fixed_length_step(model, batch):
    """Forward + backward of one fixed-length micro-batch: (per-token loss, {name: grad})."""
    _start_step(model)
    per_token_loss = model(batch["input_ids"], batch["position_ids"], None, labels=batch["labels"])
    loss_values = per_token_loss.detach().float().clone()
    per_token_loss.float().mean().backward()
    return loss_values, _grads(model)


def _relative_errors(actual, expected):
    """Relative L2 error of the loss and of every gradient (``inf`` where only one is zero)."""
    (loss, grads), (expected_loss, expected_grads) = actual, expected
    assert grads.keys() == expected_grads.keys()
    errors = {}
    for name, value, reference in [("loss", loss, expected_loss)] + [
        (name, grads[name], expected_grads[name]) for name in expected_grads
    ]:
        assert torch.isfinite(value).all() and torch.isfinite(reference).all(), name
        difference, norm = (value - reference).norm().item(), reference.norm().item()
        errors[name] = difference / norm if norm > 0 else (0.0 if difference == 0 else float("inf"))
    return errors


def _assert_matches(actual, expected, tolerance, what):
    """The loss and every gradient of ``actual`` are within ``tolerance`` of ``expected``."""
    errors = _relative_errors(actual, expected)
    worst = max(errors, key=errors.get)
    assert errors[worst] <= tolerance, (
        f"{what}: {worst}: relative L2 error {errors[worst]:.3e} exceeds {tolerance:.3e} "
        f"({sum(error > tolerance for error in errors.values())} of {len(errors)} tensors do)"
    )


def _eager_reference(twin, make_batches, step):
    """Eager result of every batch and the tolerance that reference supports.

    The twin runs every batch a second time. Where the two passes are bit-identical the
    tolerance is zero: a replay runs the same kernels on the same inputs. Otherwise it is
    ``_SPREAD_MARGIN`` times the largest relative L2 distance between the two passes.
    """
    reference = [step(twin, batch) for batch in make_batches()]
    spread = max(
        max(_relative_errors(step(twin, batch), expected).values())
        for batch, expected in zip(make_batches(), reference)
    )
    tolerance = _SPREAD_MARGIN * spread
    for first, second in zip(reference, reference[1:]):
        # A replay that returned another batch's result has to be visible.
        with pytest.raises(AssertionError, match="relative L2 error"):
            _assert_matches(first, second, tolerance, "distinct batches")
    return reference, tolerance


def _capture_replay_and_compare(
    monkeypatch, model, twin, make_batches, step, *, seq_length, micro_batch_size, helpers
):
    """Capture ``model``'s chunk graphs and compare every replay with the eager ``twin``.

    Returns ``(batches, reference, tolerance)`` for family-specific follow-up checks.
    """
    module = _module(model)
    module.train()
    _module(twin).train()
    if module is model:
        # The helper's post-capture reset calls the DDP wrapper's zero_grad_buffer(); a bare
        # model has its gradients zeroed by the step.
        model.zero_grad_buffer = lambda: None
    _module(twin).load_state_dict(module.state_dict())

    reference, tolerance = _eager_reference(twin, make_batches, step)
    batches = make_batches()
    # Eager warm-up iterations of the graphed model, as cuda_graph_warmup_steps runs them.
    for index in (0, 1):
        _assert_matches(step(model, batches[index]), reference[index], tolerance, "eager warm-up")

    helper = cuda_graphs.TECudaGraphHelper(
        model=[model],
        config=module.config,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
    )
    helpers.append(helper)
    # Transformer Engine wraps the __call__ of every captured class in its autocast for the
    # duration of a capture and takes the wrappers off at the end; a capture that raises leaves
    # them on those classes for the rest of the process.
    calls = {type(block): type(block).__call__ for block in helper.flattened_callables}
    try:
        helper.create_cudagraphs()
    finally:
        for block_class, call in calls.items():
            block_class.__call__ = call
    assert helper.graphs_created()
    callables = [module.decoder]
    if module.postprocess_block is not None:
        callables.append(module.postprocess_block)
    # One slot (a forward and a backward graph) per captured block.
    assert [len(block.cuda_graphs) for block in callables] == [1] * len(callables)

    replays = []
    original_replay = torch.cuda.CUDAGraph.replay

    def counting_replay(graph, *args, **kwargs):
        replays.append(graph)
        return original_replay(graph, *args, **kwargs)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counting_replay)

    # Two rounds in different orders: every replay must reflect the batch it was given.
    indices = list(range(len(batches)))
    for order in (indices, indices[::-1]):
        for index in order:
            before = len(replays)
            result = step(model, batches[index])
            # One forward and one backward graph per captured block.
            assert len(replays) - before == 2 * len(callables), (index, len(replays) - before)
            _assert_matches(result, reference[index], tolerance, f"replay of batch {index}")
    return batches, reference, tolerance


def _with_capture_sample_packing(model, batch):
    """``batch`` with the packing of the sample the block graphs were captured on."""
    capacity = model.config.max_seqlen_per_dp_cp_rank
    sample = model.decoder.get_layer_static_inputs(seq_length=capacity, micro_batch_size=1)
    stale = dict(batch)
    stale["packed_seq_params"] = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=sample["cu_seqlens_q"],
        cu_seqlens_kv=sample["cu_seqlens_kv"],
        cu_seqlens_q_padded=sample["cu_seqlens_q_padded"],
        cu_seqlens_kv_padded=sample["cu_seqlens_kv_padded"],
        max_seqlen_q=capacity,
        max_seqlen_kv=capacity,
    )
    return stale


def _assert_stale_packing_is_visible(model, step, batch, reference, tolerance):
    """A replay that kept the packing of the capture sample (one sequence filling the capacity)
    instead of the batch's own is not the eager result of the batch."""
    stale = step(model, _with_capture_sample_packing(model, batch))
    with pytest.raises(AssertionError, match="relative L2 error"):
        _assert_matches(stale, reference, tolerance, "capture-sample packing")


def _build_dsa_gpt(cuda_graph_impl, tokens, max_num_seqs=None, **overrides):
    """DSA over MLA: packed (THD) sequences when ``max_num_seqs`` is given, else fixed length."""
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel

    if max_num_seqs is None:
        # The batch carries no attention mask (causal attention).
        layout = dict(**_graph_config(cuda_graph_impl), create_attention_mask_in_dataloader=False)
    else:
        layout = _packed_config(cuda_graph_impl, tokens, max_num_seqs)
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        normalization="RMSNorm",
        add_bias_linear=False,
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        # Fewer keys than a sequence holds: the selection is sparse.
        dsa_indexer_topk=16,
        dsa_indexer_loss_coeff=0.01,
        **{**_BF16, **layout, **overrides},
    )
    model_parallel_cuda_manual_seed(42)
    return GPTModel(
        config=config,
        transformer_layer_spec=get_transformer_block_with_experimental_attention_variant_spec(
            config
        ),
        vocab_size=_VOCAB,
        max_sequence_length=tokens,
        position_embedding_type="rope",
    ).cuda()


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    "calculate_per_token_loss", [True, False], ids=["per_token_loss", "no_per_token_loss"]
)
def test_dsa_packed_capture_replay_matches_eager(monkeypatch, calculate_per_token_loss):
    """DSA over MLA (the indexer, its KL loss and the sparse attention) with packed sequences:
    decoder block and post-process block replayed on four packings."""
    pytest.importorskip("fast_hadamard_transform")
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention

    tokens, max_num_seqs = 128, 4
    packings = ([120], [60, 60], [32, 88], [16, 40, 60])
    overrides = dict(calculate_per_token_loss=calculate_per_token_loss)

    def make_batches():
        return [
            _thd_batch(seqlens, _VOCAB, seed, tokens, max_num_seqs)
            for seed, seqlens in enumerate(packings)
        ]

    with _capture_state(micro_batch_size=1) as helpers:
        model = _build_dsa_gpt("chunk_wise", tokens, max_num_seqs, **overrides)
        twin = _build_dsa_gpt("none", tokens, max_num_seqs, **overrides)
        assert sum(isinstance(module, DSAttention) for module in model.decoder.modules()) == 2
        # LM head + loss are captured as the post-process callable.
        assert model.postprocess_block is not None and twin.postprocess_block is None
        with _indexer_loss_state([model, twin]):
            batches, reference, tolerance = _capture_replay_and_compare(
                monkeypatch,
                model,
                twin,
                make_batches,
                _packed_step,
                seq_length=tokens,
                micro_batch_size=1,
                helpers=helpers,
            )
            _assert_stale_packing_is_visible(
                model, _packed_step, batches[1], reference[1], tolerance
            )


@pytest.mark.launch_on_gb200
def test_dsa_fixed_length_context_parallel_capture_replay_matches_eager(monkeypatch):
    """DSA with fixed-length batches and context parallelism over two ranks: every rank holds a
    zigzag shard of the sequence and all-gathers the keys of its peer inside the captured
    decoder block; LM head and loss stay eager."""
    pytest.importorskip("fast_hadamard_transform")
    if Utils.world_size % 2 != 0:
        pytest.skip("context parallelism over two ranks needs an even number of ranks")
    seq_length, micro_batch_size = 128, 2
    context_parallel = dict(context_parallel_size=2, cp_comm_type="all_gather")

    def make_batches():
        # This rank's shard of every micro-batch, as the data pipeline cuts it.
        cp_group = parallel_state.get_context_parallel_group()
        return [
            get_batch_on_this_cp_rank(
                _fixed_length_batch(seed, micro_batch_size, seq_length),
                dynamic_cp=False,
                cp_group=cp_group,
            )
            for seed in range(3)
        ]

    with _capture_state(micro_batch_size, context_parallel_size=2) as helpers:
        model = _build_dsa_gpt("chunk_wise", seq_length, **context_parallel)
        twin = _build_dsa_gpt("none", seq_length, **context_parallel)
        assert model.postprocess_block is None
        with _indexer_loss_state([model, twin]):
            _capture_replay_and_compare(
                monkeypatch,
                model,
                twin,
                make_batches,
                _fixed_length_step,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                helpers=helpers,
            )


def _dsv4_unbuildable_reason():
    """Why the installed Transformer Engine cannot build the DSv4 grouped output projection."""
    from megatron.core.transformer.experimental_attention_variant import (
        deepseek_v4_hybrid_attention_module_specs as dsv4_specs,
    )

    batched_linear = dsv4_specs.get_dsv4_o_group_proj_spec()
    if batched_linear is None:
        return None  # the torch.einsum projection
    if "input_batch_dim" not in inspect.signature(batched_linear.__init__).parameters:
        return "Transformer Engine's BatchedLinear does not take input_batch_dim"
    return None


def _build_dsv4_hybrid(cuda_graph_impl, tokens, max_num_seqs):
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel

    # Sliding-window, compressed sparse (ratio 4) and heavily compressed (ratio 128) attention.
    pattern = "W-C-H-"
    config = _make_config(
        num_layers=len(pattern),
        # Indexed by layer number; the pattern symbols select the same ratios.
        csa_compress_ratios=[0, 0, 4, 0, 128, 0],
        dsa_indexer_loss_coeff=0.01,
        is_hybrid_model=True,
        **_packed_config(cuda_graph_impl, tokens, max_num_seqs),
    )
    model_parallel_cuda_manual_seed(42)
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_dsv4_stack_spec(config),
        vocab_size=_VOCAB,
        max_sequence_length=tokens,
        hybrid_layer_pattern=pattern,
        position_embedding_type="rope",
    ).cuda()


@pytest.mark.launch_on_gb200
def test_dsv4_hybrid_packed_capture_replay_matches_eager(monkeypatch):
    """A hybrid stack of the three DSv4 attention layers with packed sequences, driven like
    ``pretrain_hybrid.py`` drives it.

    The PyTorch reference path of the sparse attention gathers the selected keys with
    ``index_select``, whose backward accumulates BF16 gradients with ``index_add_`` in an order
    that changes from run to run: two eager passes of one model differ by up to 3e-2 (relative
    L2) in the compressor gradients. With PyTorch's deterministic kernels the eager reference
    reproduces itself, and the replay is compared with it bit for bit.
    """
    pytest.importorskip("fast_hadamard_transform")
    reason = _dsv4_unbuildable_reason()
    if reason is not None:
        pytest.skip(f"DSv4HybridAttention cannot be built: {reason}")
    from megatron.core.transformer.experimental_attention_variant.csa import (
        CompressedSparseAttention,
    )

    tokens, max_num_seqs = 512, 4
    # Every sequence spans at least one block of the largest compression ratio.
    packings = ([448], [256, 192], [128, 320], [128, 128, 192])

    def make_batches():
        return [
            _thd_batch(seqlens, _VOCAB, seed, tokens, max_num_seqs)
            for seed, seqlens in enumerate(packings)
        ]

    with _capture_state(micro_batch_size=1, deterministic=True) as helpers:
        model = _build_dsv4_hybrid("chunk_wise", tokens, max_num_seqs)
        twin = _build_dsv4_hybrid("none", tokens, max_num_seqs)
        assert [
            module.compress_ratio
            for module in model.decoder.modules()
            if isinstance(module, CompressedSparseAttention)
        ] == [0, 4, 128]
        assert model.postprocess_block is not None and twin.postprocess_block is None
        with _indexer_loss_state([model, twin]):
            batches, reference, tolerance = _capture_replay_and_compare(
                monkeypatch,
                model,
                twin,
                make_batches,
                _hybrid_packed_step,
                seq_length=tokens,
                micro_batch_size=1,
                helpers=helpers,
            )
            _assert_stale_packing_is_visible(
                model, _hybrid_packed_step, batches[1], reference[1], tolerance
            )


def _build_gated_delta_net_gpt(cuda_graph_impl, variant, seq_length, **overrides):
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel

    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        experimental_attention_variant=variant,
        # Layer 1 is the linear-attention layer, layer 2 a softmax-attention layer.
        linear_attention_freq=2,
        linear_conv_kernel_dim=4,
        # The head geometry of test_kimi_k3_attnres_mtp.py's hybrid model: the linear-attention
        # kernels are compiled per head count and head dimension, so the KDA kernels are shared.
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        # The batch carries no attention mask (causal attention).
        create_attention_mask_in_dataloader=False,
        **{**_BF16, **_graph_config(cuda_graph_impl), **overrides},
    )
    model_parallel_cuda_manual_seed(42)
    return GPTModel(
        config=config,
        transformer_layer_spec=get_transformer_block_with_experimental_attention_variant_spec(
            config
        ),
        vocab_size=_VOCAB,
        max_sequence_length=seq_length,
        position_embedding_type="rope",
    ).cuda()


@pytest.mark.parametrize("variant", ["gdn", "kda"])
def test_gated_delta_net_fixed_length_capture_replay_matches_eager(monkeypatch, variant):
    """GDN / KDA (flash-linear-attention kernels) with fixed-length batches: the decoder block
    is captured, LM head and loss stay eager."""
    from megatron.core.ssm import gated_delta_net

    if not gated_delta_net.HAVE_FLA or (variant == "kda" and not gated_delta_net.HAVE_FLA_KDA):
        pytest.skip(f"flash-linear-attention does not provide the {variant} kernels")
    layer_class = {"gdn": gated_delta_net.GatedDeltaNet, "kda": gated_delta_net.KimiDeltaAttention}[
        variant
    ]
    # Two chunks of the linear-attention kernels per sequence.
    seq_length, micro_batch_size = 128, 2

    def make_batches():
        return [_fixed_length_batch(seed, micro_batch_size, seq_length) for seed in range(3)]

    with _capture_state(micro_batch_size) as helpers:
        model = _build_gated_delta_net_gpt("chunk_wise", variant, seq_length)
        twin = _build_gated_delta_net_gpt("none", variant, seq_length)
        assert isinstance(model.decoder.layers[0].self_attention, layer_class)
        assert model.postprocess_block is None
        _capture_replay_and_compare(
            monkeypatch,
            model,
            twin,
            make_batches,
            _fixed_length_step,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            helpers=helpers,
        )


def _build_mok_gpt(cuda_graph_impl, seq_length, pg_collection):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel

    num_experts = 4
    config = TransformerConfig(
        num_layers=1,
        # MoK needs at least 512 tokens per rank and sizes in multiples of 256.
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=256,
        moe_shared_expert_intermediate_size=256,
        moe_router_topk=2,
        moe_router_load_balancing_type="none",
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        gradient_accumulation_fusion=True,
        moe_megakernel_backend="mok",
        moe_megakernel_backend_config=dict(
            fwd_num_comm_sms=2,
            bwd_num_comm_sms=2,
            minibatch_size=256,
            macrobatch_size=256,
            schedule_capacity_multiplier=2.0,
        ),
        add_bias_linear=False,
        cuda_graph_warmup_steps=2,
        # The batch carries no attention mask (causal attention).
        create_attention_mask_in_dataloader=False,
        **_BF16,
        **_graph_config(cuda_graph_impl),
    )
    model_parallel_cuda_manual_seed(42)
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts, moe_grouped_gemm=True
        ),
        vocab_size=_VOCAB,
        max_sequence_length=seq_length,
        position_embedding_type="rope",
    ).cuda()
    # MoK accumulates its weight gradients into the DDP-owned main_grad buffers.
    return DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=False
        ),
        model,
        pg_collection=pg_collection,
    )


@pytest.mark.launch_on_gb200
def test_mok_bf16_fixed_length_capture_replay_matches_eager(monkeypatch):
    """The MoK megakernel with BF16 expert weights inside a one-layer block graph."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("MoK requires Blackwell SM100/SM103")
    functional = pytest.importorskip("mok.functional")
    if not is_te_min_version("2.14.0"):
        pytest.skip("CUDA graphs with the TE RNG tracker and grouped weights require TE >= 2.14.0")
    from megatron.core.process_groups_config import ProcessGroupCollection

    seq_length, micro_batch_size = 512, 1

    def make_batches():
        return [_fixed_length_batch(seed, micro_batch_size, seq_length) for seed in range(3)]

    try:
        with _capture_state(micro_batch_size, expert_model_parallel_size=1) as helpers:
            groups = ProcessGroupCollection.use_mpu_process_groups()
            model = _build_mok_gpt("chunk_wise", seq_length, groups)
            twin = _build_mok_gpt("none", seq_length, groups)
            assert model.module.decoder.layers[0].mlp.megakernel_experts is not None
            _capture_replay_and_compare(
                monkeypatch,
                model,
                twin,
                make_batches,
                _fixed_length_step,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
                helpers=helpers,
            )
    finally:
        functional.clear_workspace_cache()
